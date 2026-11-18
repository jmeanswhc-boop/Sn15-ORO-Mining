from __future__ import annotations

import json
import re
import threading
from contextlib import contextmanager
from typing import Any

from src.agent.proxy_client import ProxyClient


class _Tally:
    """The errand's clock, counted rather than read: waits add their own length, and a request in flight adds a
    quarter second for every quarter second it stays open."""
    EXCERPT = 0.25

    def __init__(self) -> None:
        self.seconds = 0.0
        self._barrier = threading.Event()

    def now(self) -> float:
        return self.seconds

    def halt_alt(self, length: float) -> None:
        if length > 0:
            threading.Event().wait(length)
            self.seconds += length

    @contextmanager
    def underway(self):
        done = threading.Event()

        def count() -> None:
            while not done.wait(self.EXCERPT):
                self.seconds += self.EXCERPT

        worker = threading.Thread(target=count, daemon=True)
        worker.start()
        try:
            yield
        finally:
            done.set()
            worker.join()


_TALLY = _Tally()

# ---------------------------------------------------------------------------------------------------------------- models
PERFORMER = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "anthropic/claude-haiku-5.5"]
PERUSER = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5", "openai/gpt-6-luna"]   # reads the request and the tools once
ASSESSOR = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]  # item checks and the rank check
INSPECTOR = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]  # the decline audit
WATCHER = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]  # shopper lines
STRATEGIST = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]   # the question plan (the hidden need hangs on it)
CONCLUDER = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]  # the finish check: stated steps still undone

# ---------------------------------------------------------------------------------------------------------------- clock
CUTOFF = 265.0          # every call ends before this many seconds
WRAPUP = 70.0            # with less time than this left, the errand heads for its finish
FINAL_INVOKE = 28.0          # with less than this left, the code sends the finish itself
TIMER_STAGE = 30           # (v605e) the status line's time left, once shown, in steps of this many seconds
CROSSCHECK_EQUIVALENT_LOWER = 4        # (v605e F3) read listings of a category carrying a field before a synonym of a value it never writes may meet it
SNAPSHOT_GLYPHS = 20000        # longest observation text shown to the actor
SEEK_PASSAGE = 600         # longest text of one field on a search page shown to the actor
CHRONICLE_PASSAGE = 2500        # longest text of one field in any other observation shown to the actor (store texts go whole)
GENTLE_OUTLAY = 0.03         # model spend (USD) after which a role's cheaper fallback model is used
OUTLAY_LID = 0.045         # model spend after which the errand heads for its finish (a run's key pays for every task)
FIRM_OUTLAY = 0.10         # model spend after which no model call goes out at all: the code finishes alone
CONCLUDE_STAGES = 2          # steps kept while a stated finishing step is on the list (its message and the finish itself)
CONCLUDE_CYCLES = 2          # tool calls the finish check may ask for before one finish
CYCLE_THRESHOLD = 80
TEMPO_INTERVAL = 0.15           # least gap in seconds between two HTTP requests of the errand, model and world calls alike (TA)
SLUGGISH_INTERVAL = 1.0            # the gap for SLOW_FOR seconds after a request was refused for load (429 or 503): one bucket per sandbox
SLUGGISH_TOWARD = 30.0
LOWERED_FOLLOWING = 45.0         # the models count as gone after this many seconds without a model reply (and two failed tries)
LLM_ATTEMPTS = 2           # attempts per model of one call after a load or transport failure, with a growing pause between
COSMOS_ATTEMPTS = 5           # attempts of one world call after a load or transport failure
ENGINE_CONCLUDES = 4         # finishing calls the code sends at most (a refused order is repaired and sent again)
ENGINE_SHEETS = 3            # further result pages the code reads of each product's best search
ENGINE_PERUSES = 8            # rounds of record reads of the code's picks at most (one record per product a round)
CONNECTORS = r"\s+and\s+|\s*&\s*|\s*/\s*|\s+\+\s+"   # (CA) how a store joins two kinds in one category name
PLACE_TERMS = {"add", "put", "hold", "set", "insert", "place", "keep", "include", "in", "plus", "increase", "append"}
DISCARD_TERMS = {"remove", "drop", "release", "delete", "take", "out", "clear", "discard", "minus", "decrease", "subtract", "off", "unhold"}
DEFECT_SLOTS = r"error|fault|refus|reject|denied|invalid|fail"
INQUIRE_STANDARD = 3           # messages assumed while no allowance was read anywhere; the store's own refusal sets the real cap
INITIAL_PURCHASE_VIA = 7        # with this many steps left after a refusal, only cart, view and finishing tools remain
INITIAL_PURCHASE_OWED = 10      # at most this many steps kept for the first order and one repair (a quarter of the steps, at least FIRST_ORDER_BY)
INACTIVE_THRESHOLD = 3            # replies without a tool call before the code finishes
STALLED_THRESHOLD = 4           # replies in a row where nothing was sent before the backup model takes a turn
SURVEY_PERUSES = 12          # at most this many shelf slices the code reads itself when a change leaves the whole shelf acceptable
TRAVERSE_SHEETS = 6            # under a cheapest rule in an id-ordered store: at most this many further pages read per product
TRAVERSE_AGGREGATE = 8            # and at most this many in all
SETTING_SHEETS = 4           # (CG2) an id-ordered all-words search led by a required value: at most this many further pages read at once per product
SETTING_AGGREGATE = 8           # and at most this many in all
SETTING_AGGREGATE_EACH = 5       # (EB3) several products: at least this many value pages in all per product
TAG_ATTEMPTS = 4           # (EB1) composed value + category-label searches of one product at most
TAG_EXAMINES = 3           # (EB7) no cheapest rule: rows of a one-page all-values composed search opened at most
ROOF_INVOKES = 4            # (CG3) price-filter listing calls of one product's ceiling walk at most (lowering the limit and reading pages together)
ROOF_EXAMINES = 3            # and at most this many of its rows opened (a pick and two backups)
BOND_TERMS = {"and", "or", "the", "for", "with", "of", "a", "an", "in", "on", "to", "by", "from", "at", "as", "is", "are", "be",
              "this", "that", "it", "its", "your", "our", "you", "all", "any", "not", "no"}
DEEPER_SHEETS = 4                 # pages a best/cheapest product's widest search may still get before its first cart put
INQUIRE_FOLLOWING_GROUPS = 2        # listing detail records read before the first planned question
INQUIRE_LEEWAY_STAGES = 3       # while fewer records were read and over 20 steps remain, this many more steps for them
INQUIRE_FOLLOWING_STAGES = 5       # or this many steps used

# ---------------------------------------------------------------------------------------------------------------- prompts
PERFORM_PRINCIPLES = """Your role: a meticulous shopping assistant serving a single shopper inside a simulated store, able to act only through the tools you are given. Each errand earns either 1 or 0, and the 1 comes only when the ending is correct on every count: what you order satisfies each requirement the shopper wrote down, each hidden need they disclose when you ask, and each change they announce as you go; quantity and budget are both correct; when the request wants the cheapest (or best) item, that is the one picked; and each procedural rule in the request is followed. If no item in the store satisfies all of it, the correct ending is a decline, carried out precisely the way the request describes.

HOW THE STORE WORKS
- Every tool call uses up one step, even a call that fails or is invalid, and the supply of steps is finite. Reasoning is free.
- You may put several tool calls in one reply as long as none depends on the output of another (opening the details of a few listings, say); they are executed in sequence. An order or a decline is always sent by itself.
- If a details tool accepts a LIST of listing ids, it returns every full record for ONE step (within its item limit): use it whenever you open two listings or more.
- Each store names its tools and arguments differently and shapes its results differently. Study every tool's description and JSON schema, and use the names and permitted values exactly as given. From the first results, work out which tool reveals which facts (price, stock, detailed properties).
- At the start, find out which category labels this store gives to listings of the kind requested. If similar labels sit next to each other, pick the one whose label fits the wording of the whole request, then search and buy within it.
- Search behaves differently in every store, so learn this one from its opening results. Rows may arrive best-match first or in a fixed id sequence; a keyword search may demand every word (an extra word lowers the total) or accept any word (an extra word raises the total, and listings hitting more of the words rank higher); only a few rows or pages may be shown; a filter may only work in a single mode (keyword mode may disregard the category or price filter, filter mode may disregard the words). Set any reported total against the rows you can actually see: if it is bigger, the fitting or cheapest listing could be on a later page or past the visible ones. If every word is demanded, tighten the search with more precise words (or the category and price filters with a lower price ceiling) until the total fits within the visible pages. Use short queries (one to three words) and put the most distinctive word FIRST: certain stores look only at the opening word or two and drop the rest, so a long query may in effect search only its first words (set a longer query's total and rows beside those of a shorter query with the same start; the notes say what they reveal). If every word is demanded, a required value plus the item kind ("blue chair", value in front) returns exactly the listings that carry that value. If any word can match, everyday words (the item kind, a unit) hit most of the shelf and hide everything else: the distinctive word or figure of a required value on its own ("herringbone", "30m", "0.4") returns only the listings that carry it, so search it alone and also in the other spellings listings use ("30 m", "3 atm"), go through every page of that search whenever its total fits on them, and combine the item kind with the values only to bring the nearest matches to the top. Keyword searches may also hit category names and property values. After each search, notes explain what the totals collected so far reveal about how this store searches.

THE SHOPPER AND HIDDEN NEEDS
- The shopper keeps some needs unwritten and discloses one only when a message asks about that property. Whatever a message asks FIRST gets the most dependable answer, so each message covers one property, or two at the most with the more probable one first: a brief, simple, warm question that names the property (one or two sample values are fine). No instructions, lists, labels, positions or rules.
- "Don't mind", "no preference" or "anything is fine" leaves that property open. Any preference or limit the shopper states becomes a hard requirement from that point, word for word, numbers and all.
- If questions are being planned for you, they are sent after you have opened the details of two or three listings of the kind requested, so make those openings among your first moves. Replies come back in the conversation along with their meaning; treat every disclosed need as a hard requirement and search again combining it with the values already stated.
- If you do the asking yourself: open the details of two or three listings first, next ask about whatever the story suggests (whom it is for, the look it needs, where or how it gets used), and after that about properties the listings carry that vary across the candidates. Do not ask anything the request has already decided, and never exceed the message budget. Your own messages are the means of performing the steps the instructions lay down (a confirmation, the basket, the total, a reply to the shopper): you may use the message tool any time the allowance still has a message available. As long as hidden-need questions are being planned for you, let the plan do the asking (it chooses properties from the listings you opened) and save your own messages for what the instructions require of you; ahead of an order, the basket is reported to the shopper on your behalf if you have not reported it since your last cart change.

RULES OF PROCEDURE
- The request can lay down rules for the way you work: how many listings to view or compare, what to verify, what to tell the shopper, what a remark has to contain, how to decline. These are graded just like requirements; carry out each one to the letter and in full. To "look at" or "open" N listings means N listings that fit completely, each opened through the details tool.
- If the shopper puts a question to you (to confirm a change, for instance), it gets answered on your behalf while messages are left; the change itself still binds you.
- The status line shows the INSTRUCTIONS IN FORCE: the rules in the request plus each instruction the shopper or the store adds afterwards (confirm with me before buying, tell me the total, a new allowance of messages or steps). Any of them can change at any moment, so read them again, along with the requirements and the cart, after each reply, notice or refusal, and base your next move on what they say right now, never on routine. Any stated step not yet done when you finish is performed for you just before the finish is sent, and a message is held back for it; a message you send yourself after the final cart change also counts.
- Things the shopper only hopes for ("ideally", "would be nice", "if possible") are not requirements: they merely decide between items that already meet everything, and they never override a cheapest rule.
- If the request leaves open something only the shopper can decide (no price limit, say, or two wishes that might not both be possible so the shopper must say which counts more), ask before choosing.

CHANGES WHILE YOU WORK
- The shopper can change things: a new budget (up or down), a requirement removed, swapped or added, a price increase, an item going out of stock. A single line may hold several changes. These lines can show up inside any tool result (an extra field containing sentences) or along with an order attempt. Every change applies from the moment it arrives: check the pick and the backups again, and if the cheapest or best is wanted, choose afresh from the new acceptable set.
- An order attempt may be REFUSED: the episode continues and the cart stays as it was. Read the refusal, apply each change it mentions, remove from the cart anything that no longer fits (a sold-out item, for instance), add the next verified fit, check its stock and price again, and order once more.
- Items may also go out of stock or change price silently, so check stock and price again just before each order attempt; an item whose present price multiplied by the quantity exceeds the present budget is no longer a fit.

SEVERAL PRODUCTS
- A request may want several distinct products, each with its own quantity, requirements and hidden needs. Handle all of them: search the category of each, verify a fit for each, and order them in one go, a single line per product at its quantity. A cart tool that takes a list of lines may overwrite the entire cart with that list, so include every product's line in each such call. The cheapest or best rule is applied inside each product.
- A requirement can state its value by referring to another listing or page ("the same colour as the jumpsuit sold by X"): locate that listing first (search the words that identify it, such as its shop or brand along with its kind), read its record, take the value from the field named, and then search for the requested product with that value.

THE CART
- The cart tool might ADD units to a line already in the cart instead of setting them: adding 1 more of a line you hold gives 2. Do not add a line again that the cart already contains; to adjust units, remove or add the difference. After each cart call, a note reports what the code counts in the cart.

HOW TO JUDGE A LISTING (this decides the score)
- The grader takes each requirement from the listing's field for that property, matching the value's own words as this store writes them in that field. A field whose name expresses the same property counts regardless of spelling: separators, a prefix, a shorter or longer name, or a common alternative name ("fit" for a pants fit type, "neckline" for a collar type, "dial size" for a watch dial size, "color family" for a colour). If a field bearing the property's own name exists, that field decides, and so does a field of that name with one extra word in front (pants length for length, clothing material for material): if either of them writes the value, the requirement is satisfied. Two fields whose names differ only in separators (size class, size_class) are a single field: the requirement is satisfied if either writes the value. Failing that, such an alternative field decides; if no field names the property at all, the options list or the title may display the value, except that the title alone (with no field, the options list included, writing the value) does not show it when other listings of the same category carry a field for that property (FIELD EVIDENCE gives the count): the grader reads that field, and this listing has none. The description never counts as evidence.
- Values are matched on their words as this store writes them: case, separators, singular versus plural and spelling variants make no difference; a more specific value containing the required one satisfies it ("80% cotton" for cotton, "forest green" for green, "rtx 3060 12gb" for rtx 3060); a single element of a list suffices. A synonym, an abbreviation, a related word or a shade ("tech wood" for technical wood, "win11" for windows 11, "embroidered" for embroidery, "solid" for plain, ivory or cream for off-white) satisfies a value only if no listing in the category writes the value itself in that field: if other listings give "matte" as their finish, a "satin" finish does not satisfy matte. A value combined with another by a plus ("wool + nylon") does not satisfy one of its parts if the field also holds the other part by itself ("nylon"). FIELD EVIDENCE lines, where supplied, are calculated from this store's records and state which case applies. A different value of the property does not satisfy it (quartz is not mechanical, rosewood is not maple), and neither does a word that merely resembles it. A value claiming to cover all values of the property ("suitable for all occasions", "all seasons", "universal") names none of them and does not satisfy a required value; a list that names the required value as well still satisfies it. A field naming another value of the property does not satisfy the required value merely because the two often occur together or one tends to suggest the other (a "pullover" style is not "zip-up", "slip-on" is not "lace-up"); only a shade normally sold as the required colour, or a more specific value containing it, counts.
- A field belonging to another property is not the requirement's property (a lining material is not the material, a length is not a style, a strap length is not a strap type): it can never replace a property whose own field the category's listings carry.
- Only what the shopper stated or disclosed matters. Never exclude a listing for something the shopper never asked for: an age group, gender, size, occasion, condition, quality or use you infer from the story or the description. The story is background; if it appears to matter, ask the shopper about that property rather than assume. Questions you ask set no requirements: only the shopper's answers do.
- Numbers are compared as quantities after unit conversion (4.5 cm equals 45 mm); limits include their boundary; a range qualifies only if the whole of it satisfies the limit. If no field names the property and the title serves as evidence, an exact required value is satisfied when the title displays its words and figures in the same sequence, even followed by extra detail (9' x 12' 4" shows 9' x 12'; 9' x 13' does not).
- The figures in a title ("55 x 94 Inch", "117 x 228 cm") do not tell which dimension each refers to: never exclude a listing, or skip opening it, because of its title figures; read the fields in its details.
- Honour exclusions ("leave out anything with X") and closed lists ("A or B and nothing else").
- The requested kind of product is satisfied by the listing's category alone: for the grader, a listing in the requested category IS that kind even if its title or description names some other object (a container, a case, a refill, a set, an accessory), so never pass over, reject or rank it lower for what it appears to be; judge it purely on the requirements, price and stock. No property needs to restate the category name. If the store labels its categories, the item has to sit in the one whose label fits the wording of the whole request (a request for "A or B" fits a category "A & B", not a narrower neighbour "B"). It has to be in stock if the request wants items on hand, and price multiplied by quantity has to stay within the budget (equal is acceptable). A listing's price is in the budget's unit unless its record names a different one (45 against a budget of 1000 US cents is 45 cents); never rescale either side.
- Declining is seldom the correct ending. If no listing satisfies the strict wording but one does satisfy each requirement under these rules (a more specific value, another name for the same field, or a synonym or shade where no listing in the category writes the value itself in that field), that listing is the fit; decline only if the search truly turned up nothing satisfying every requirement under these rules.

CHEAPEST OR BEST
- For the cheapest, the order has to be the lowest-priced item out of ALL items satisfying every requirement; finding one fitting item is not sufficient. If the keyword search demands every word and matches property values, the value-and-kind search returns the candidates carrying the value: go through all of its pages and open them in price order; a category listing with a price limit reveals the prices across the whole category but not its values: use it to learn which cheaper rows exist (drop the limit until the total fits), then check the properties of those rows. Search each required value together with the item kind as a short query, value in front ("blue chair", "30m watch"), and go through every page of it if its total fits on the readable pages (if not, add a second required value or use a narrower kind word); repeat this with the kind's other names ("stool", "bench") and with the value by itself; if a value cannot be searched (a figure the store drops), also search the product line of each fit or near-fit you come across, as two words, its maker plus one distinctive word from its title, every page: the other sizes and colours of a line are separate listings sharing its property values and may be cheaper; order the visible candidates by price yourself and open their details starting from the cheapest; the first one whose property fields satisfy everything, with every cheaper one opened and rejected, is the answer. Hold on to the next fitting ones as backups.
- For the best by some measure (most, largest, longest, highest...), compare ALL fitting items on that measure in the same manner and take the extreme. An item that gives no value for the measure cannot win.

FINISHING
- Maintain a ranked list of verified fits that includes at least one backup: the top pick may go out of stock at checkout.
- Ahead of each order attempt: the cart contains exactly one line per product requested (a single line when one product is requested), each being the chosen item at its required quantity, and nothing more; since your last cart change you have read the chosen item's stock and price again, and it is in stock and inside the budget.
- Decline only once a thorough search turns up nothing satisfying every requirement, and follow the request's decline rule exactly. If the request says not to offer substitutes, the decline remark names, describes or refers to no listing at all: no ids, no titles, no prices.
- Plan your steps: send the first order attempt while seven or more steps are left, so that a refusal can still be dealt with. Once 7 steps are left, only cart, stock-check and finishing tools are available. Make the decision once; do not switch the cart back and forth.
- Each reply includes at least one tool call. Keep the text brief: the current requirements, the verified fits with their price and stock, and the next move."""

PERUSE_PETITION = """Go through the shopper's request and the store's tool list given below, and reply with ONE compact JSON object (no indentation) and nothing more, containing these keys:

"product": the product kind requested, using the request's own words (when several products are requested, list them all joined with "and").
"items": an entry for each DIFFERENT product the request wants bought, in the request's order: {"product": its kind in the request's words, "quantity": how many units of one listing of it (same as "quantity" below), "requirements": its firm requirements (same as "requirements" below), "depends_on": the 1-based number of the other entry that this product has to work with, fit, suit or be compatible with according to the request, otherwise null}. A single entry when a single product is requested. A name joining two related kinds with "and" or "or" under one count and one set of requirements (a store category such as "lamps and lamp shades") is one product, one entry.
"quantity": how many units of one listing to order (an integer; 1 if the request does not ask for more than one). An item sold as one unit, like a pair of shoes or a set, has quantity 1. Only an explicit request to buy several units sets this ("I need three of them", "order 2 packs"); the number of people who will use or share it, or the number of old ones there used to be, is story and keeps it at 1. A count written before a product's name ("4 mugs", "two packs of tea") is that product's quantity even when the request adds that each product goes on one line, or one line apiece: such a remark counts the lines (one per product), never the units on a line.
"requirements": a list of strings, one for each firm requirement on the product (when several products are requested: the requirements of every product, each written "property (product): constraint"), each "property: constraint" carrying the shopper's exact value, limit, exclusion or closed list. Omit price, stock, quantity, budgets and mere wishes, and do not repeat the product kind or category: words that form part of the category's name (a season, an occasion or an audience within it) are handled by the category, not by a separate requirement. A requirement is a property of the product together with the value the request says it must take ("the pattern has to be herringbone", "with a mechanical movement", "at least 4 drawers", "the occasion has to be party"). The surrounding story never forms a requirement: who it is meant for, an event where it will be worn or used ("to wear at his wedding", "for the staff party"), where, when or how it will be used, how many people share it, or that it replaces an old one ("a new watch" asks for a replacement, not for a condition). An occasion, use or audience counts as a requirement only if the request makes it a property of the item itself ("the occasion has to be business", "a blazer made for business"). The value of a requirement is words or figures the request itself writes as the item's value; a quality, reason or wish given by the story (natural and breathable, slim enough for a case, works with what I already have, covers the whole drop) is a hint, never a requirement. When a value is taken from another listing ("the colour shown on the page of X's lamps with a brass base"), that requirement keeps the pointing words as its constraint, and the details that only describe which listing is meant (its seller, its kind, "with a brass base") are never requirements of the product itself.
"price_cap": the maximum total price as the number written in the request, in the request's own unit (not converted: "3050 US cents" gives 3050), or null if none is stated.
"currency": the request's price unit, or null.
"in_stock_only": true if the request wants only items that are on hand or in stock.
"priority": "cheapest" only if the request clearly asks for the cheapest or lowest-priced fitting item; "best" only if it clearly asks for the best by some measure; otherwise "none". Mentioning that money is tight or that every penny matters is not a request for the cheapest; it merely explains the cap.
"priority_detail": the measure when "best"; otherwise "".
"message_budget": the number of messages the shopper lets you send, as an integer, or null if not stated.
"step_budget": the number of moves or steps the request permits, as an integer, or null if not stated.
"rules": each procedural rule the request lays down for the assistant (how many listings to view, what to check or say, how to decline, what to avoid, what to include), quoted verbatim. Things the shopper will do personally are story, not rules.
"soft_wants": things the shopper would like but does not strictly need (ideally, preferably, would be nice, if possible). Keep them out of "requirements".
"instructions": the procedural instructions once more, one object for each: {"text": the instruction in the request's words, at most 20 words, "before_finish": true if it has to be done before or at the finish (before an order or a decline: confirming with the shopper first, telling them the basket contents or the total, a closing remark, a final check), otherwise false, "needs_message": true if doing it requires sending the shopper a message, otherwise false, "before_change": true if it asks the assistant to check with the shopper before changing, replacing or removing something already in the cart, otherwise false}. Include as well an instruction that only says how to finish or what to say at the finish. Empty if the request sets none.
"ask_first": what only the shopper can answer and ought to be asked before choosing: for instance the budget when no limit is given, which of two wishes counts more when the request says both may not be achievable (in that case put both wishes in soft_wants, not in requirements), or anything the request says to check with the shopper. Empty if none.
"hints": story details that point to a property of a product without giving its value (what it is used for or with, where, by whom, what it has to fit, match or withstand), each "property (product): the detail in a few words" ("property: detail" for a single product), naming the property the way listings would probably name it. If a detail could relate to more than one property, give the likeliest first and the next one after a slash ("colour / style (chair): must suit a room with one painted wall"): what the item has to match, suit or go well with (a room, a wall, an outfit, things used alongside it) and how it looks point first to its colour, then to its pattern or style. Omit a detail that only justifies a requirement the request already states (why it has to be the stated material or size). Weather, work, a schedule, travel, or a place where the item is neither worn nor used is background and never a hint; likewise a purpose that every listing of the kind serves (storage for a dresser, warmth for a coat). Empty if none.
Each sentence of the request ends up somewhere: a requirement, the priority, a soft want, a rule, a budget, a hint, or plain story. If you are unsure whether a sentence is a rule, put it in "rules".
"references": requirements whose value the request supplies only by pointing to ANOTHER listing or page ("the same colour as the jumpsuit sold by X", "within 5 of the figure on the page for Y", "a colour other than the one on Z's page"), each "property (product): as shown on <how the request names that listing>" ("property (product): different from the one shown on <how the request names that listing>" if the value has to differ from that listing's); also add each to "requirements" in that same form. Empty if none.
"reference_targets": for every entry in "references", the listing it points to: {"property": its property, "product": its product, "seller": the shop or brand name the request gives for that listing, written exactly as in the request, "kind": the kind of product that listing is, "details": [every other value the request names for that listing, brief and in the request's words ("blue", "4gb", "net weight 167.262186 g")], "relation": "same", "differ" (the value must NOT be that listing's: set apart from, different from, other than), "within", "at most" or "at least" (how the value has to relate to that listing's), "tolerance": the number for "within", otherwise null, "unit": its unit, otherwise ""}. Empty if none.
"decline_no_names": true if the request's decline rule forbids offering, naming or suggesting substitutes or alternatives; false if it asks you to name something; null if it states no rule.
"roles": an object giving the tool for each role, using the exact tool names from the list, or null if no tool suits: "search", "details", "view", "message", "cart", "cart_view", "order", "decline". "message" sends the shopper a message; "cart" adds to or removes from the cart; "order" buys or checks out; "decline" closes the errand without a purchase.
"""

RECEIVE = """Your job is to help a shopping assistant make sense of the shopper's latest words. Reply with ONE JSON object and nothing more, containing:
"meaning": a brief sentence on what the shopper intends.
"needs": new firm requirements on the product, each "property: constraint" carrying the exact value, limit, exclusion or closed list stated. Following a comparison word (falls short of, nothing less than, at least), a value made of words rather than a figure is the value itself, written the way the shopper writes it: "falls short of not available" gives "property: not available" ("not available" and "none" may be genuine values of a field).
"no_preference": properties the shopper said they are indifferent to.
"dropped": earlier requirements the shopper no longer requires.
"new_budget": the new maximum total price as a number in the unit of the request's budget (the figure the line writes when it uses that unit: "13075 cents" gives 13075 for a budget in cents), or null.
"unavailable": true if the shopper says an item can no longer be purchased (sold out, gone, out of stock, button disabled), otherwise false.
"meta_only": true if the line merely asks the assistant to ask less at a time (one question at a time, fewer things together, only the first now and the others later) and gives no value: it answers nothing so far, so put nothing in needs, no_preference or dropped for it, and its figurative words name no value; otherwise false. A line saying only that the shopper did not follow or understand the question, or asking for it to be repeated in shorter or simpler words, is not meta_only.
"requirements_now": the complete list of firm requirements on the product as they currently stand, each "property: constraint": begin with REQUIREMENTS SO FAR, retain each one unless the shopper dropped or replaced it, keep only the latest for a replaced value, and add every new need the shopper's lines state. Never add a requirement taken from the request's story that REQUIREMENTS SO FAR does not contain; requirements are added only by the shopper's lines. Leave out price, stock, quantity and product category.
"reply": if the shopper asks the assistant something or asks for a confirmation, a brief plain answer to send back that states the substance of the change (confirming the new value, for instance); otherwise "".
"instructions": procedural instructions these lines give the assistant from here on (confirm with me before buying, tell me the total or the basket, do or avoid something, how to finish or what to say at the finish), one object for each: {"text": the instruction in the lines' words, "before_finish": true if it has to be done before or at the finish (an order or a decline), otherwise false, "needs_message": true if doing it requires sending the shopper a message, otherwise false, "before_change": true if it asks the assistant to check with the shopper before changing, replacing or removing something already in the cart, otherwise false}; [] if the lines give none. A need, a budget or a dropped requirement does not count as an instruction.
"allowance": if the lines set a new limit on the assistant's messages, questions, steps, moves or calls: {"messages_total": the new total number of messages permitted, or null, "messages_left": how many messages can still be sent from here, or null, "steps_total": the new total number of steps permitted, or null, "steps_left": how many steps remain from here, or null}; null if no such limit is set.
"references": for every need in "needs" whose value the shopper supplies only by pointing to another listing or page, the listing it points to: {"property": its property, "seller": the shop or brand name the line gives for that listing, written exactly as in the line, "kind": the kind of product that listing is, "details": [every other value the line names for that listing, brief and in the line's words], "relation": "same", "differ" (the value must NOT be that listing's: set apart from, different from, other than), "within", "at most" or "at least", "tolerance": the number for "within", otherwise null, "unit": its unit, otherwise ""}; [] if none.
A single line may hold several changes; list each of them. A need on a different property, even a related one (a seat height next to an overall height, a strap width next to a width), is a fresh requirement and replaces nothing: an earlier requirement counts as dropped or replaced only when the line says so about that very property. A replaced requirement is both dropped (the old one) and a new need (the new one). If the shopper says which of two wishes counts more, the favoured one is a need. A budget that is stated is new_budget. A plain wish ("ideally", "would be nice") is not a need. Take the shopper literally and generously: homely, figurative or indirect words that name a value mean the shopper firmly wants that value; a value is excluded only if the shopper plainly says to avoid it. Words that merely emphasise how firmly the shopper holds a value (only, nothing but, nothing short of, strictly, stays, sticking with, locked to, holding out for) are not part of the value: write only the value words ("material: jute", "color: white"), never "<value> only"; an exclusion is added only when the shopper also rules out blends or other values in so many words. A number paired with a comparison (at most, at least, under, over, no more than) is a limit and keeps its unit. A new budget takes the place of the old one whether it is higher or lower, even if the shopper calls it good news. If the shopper merely says they don't mind, have no preference or anything works, put the property that was asked about in no_preference (if the message asked about two properties, only the FIRST one). A reply that merely says the shopper did not follow or understand the question ("that was rather dense", "you've lost me", "could you put it another way") is how this shopper expresses no preference when the question named the product the property belongs to: in that case put the property asked about in no_preference as well; if several products are requested and the question did not say which one, the property stays open (put nothing in no_preference). When several products are requested, write every need as "property (product): constraint", taking the product from the line (a line about "toe style of the women's boots" gives "toe style (women's boots): ..."). The assistant's own question never becomes a requirement: only what the shopper says does; values offered as examples in the question do not turn into needs. A "don't mind" answer releases just the property the assistant's message asked about; it never drops or alters a need the shopper stated earlier, and if the message asked nothing (it only informed), such an answer changes nothing. A need given by pointing to another listing keeps the line's own pointing words as its value ("finish: unlike the one shown on Y's kettle page"); the values the line names for that other listing describe that listing, not the product: they are not needs, drops or freed properties."""

PERUSE_PRINCIPLES = """- Judge each requirement by the listing's field for that property, using the value's own words exactly as this store writes them there: any field whose name means the same property counts, however it is spelled (separators, a prefix, a shorter or longer name, or a common alternative name such as "fit" for a pants fit type, "neckline" for a collar type, "color family" for a colour). If a field bearing the property's own name exists, that field decides, and so does a field of that name with one extra leading word (pants length for length, clothing material for material): the requirement is met when either of them writes the value. Two fields whose names differ only in separators (size class, size_class) count as a single field: the requirement is met when either writes the value. Failing that, such an alternative field decides; the options list or the title may show the value only when no field names the property at all, and the title alone (with no field, the options list included, writing the value) does not show it when other listings of the same category carry a field for that property (FIELD EVIDENCE gives the count): the grader reads that field, which this listing lacks. The description never counts as evidence.
- Values are matched on their words as this store writes them: differences of case, separators, singular versus plural and spelling variants do not matter; a more specific value that contains the required one satisfies it ("80% cotton" for cotton, "forest green" for green, "rtx 3060 12gb" for rtx 3060); a single element of a list suffices. A synonym, an abbreviation, a related word or a shade ("tech wood" for technical wood, "win11" for windows 11, "embroidered" for embroidery, "solid" for plain, ivory or cream for off-white) satisfies a value only if no listing in the category writes the value itself in that field: if other listings give "matte" as their finish, a "satin" finish does not satisfy matte. A value combined with another by a plus ("wool + nylon") does not satisfy one of its parts when the field also holds the other part on its own ("nylon"). FIELD EVIDENCE lines, where present, are derived from this store's records and tell which case applies. A different value of the property does not satisfy it (quartz is not mechanical, rosewood is not maple), and neither does a word that merely looks similar. A value claiming to cover every value of the property ("suitable for all occasions", "all seasons", "universal") names none of them and does not satisfy a required value; a list that names the required value as well still satisfies it. A field naming another value of the property does not satisfy the required value merely because the two often appear together or one tends to suggest the other (a "pullover" style is not "zip-up", "slip-on" is not "lace-up"); only a shade commonly sold as the required colour, or a more specific value containing it, counts.
- A field for a different property does not count (a lining material is not the material, a length is not a style, a strap length is not a strap type): it never substitutes for a property whose own field appears in the category's listings.
- Compare numbers as quantities once units are converted (4.5 cm equals 45 mm); limits are inclusive; a range counts only if the whole of it meets the limit. When no field names the property and the title serves as the evidence, an exact required value is met if the title shows its words and figures in the same order, even when further detail follows them (9' x 12' 4" shows 9' x 12'; 9' x 13' does not).
- A listing's price uses the same unit as the request's budget unless its record names a different currency or unit (a price of 45 against a budget of 1000 US cents means 45 cents, comfortably inside it); never rescale either the price or the budget.
- Only the listed requirements matter: never reject a listing over an age group, gender, size, occasion, condition, season, quality or use that the requirements do not mention.
- The category label alone satisfies the requested kind: a listing in the requested category is of that kind even if its title names some other object (a container, a case, a refill, a set, an accessory); judge it solely on the requirements, price and stock."""

ASSESS_ENTRY = """You are the final review before a shopping assistant commits to an item. The errand earns 1 only if the item satisfies every requirement below. Base your judgement ONLY on the records shown, following these rules:
""" + PERUSE_PRINCIPLES + """
- Words that form part of the category's own name are satisfied by the category.
- Respect exclusions and closed lists.
- The item has to be the requested kind; where listings carry a category label, it has to be the label that matches the request's wording as a whole, not a narrower or adjacent one.
- Price multiplied by quantity has to fit within the budget (equal is acceptable). It fails when the records show it out of stock.
- A simple wish ("ideally", "would be nice") marked as not required is no requirement.
Where records conflict (on price or stock, say), the most recent one wins.
Reply with ONE JSON object and nothing more: {"item": the item's id, "fits": true or false, "problems": [short strings, one per requirement that is unmet or not shown]}. Set fits=false only for a listed requirement that the item's fields plainly fail under these rules (a FIELD EVIDENCE line stating the item does NOT meet a requirement counts as such a failure), or that neither a field nor (if no field for it exists) the options list or the title states; never fail an item because a stock or price record is missing."""

STANDING_ASSESS = """A shopping assistant is about to place an order for a request that asks for the PRIORITY. Below you will find the requirements, the chosen item and the listing records seen. Applying the reading rules below (price times quantity inside the budget; not shown out of stock), name the listings that could satisfy every requirement AND outdo the chosen item on the priority (a lower price for the cheapest; a better value of the measure for the best). Omit listings whose records show them failing a requirement, out of stock or over budget. Figures in a ROW's title (such as 55 x 94) do not tell which dimension each one is, so they never rule a ROW out. Nor does a word of a ROW's title that seems to contradict a requirement (a model name carrying another network, size or colour): the record's own field decides, and titles often name a model line rather than the item's value.
READING RULES:
""" + PERUSE_PRINCIPLES + """
Reply with ONE JSON object and nothing more: {"better": [ids, best first, at most 4], "reason": "one sentence"}."""

INQUIRE_SCHEME = """You draw up the questions a shopping assistant will send to a shopper. The shopper has hidden needs: each one is a value of one product property, disclosed only when a message asks about that exact property; whatever a message asks first gets the most reliable answer, and questions on anything else get a polite "don't mind". Pick up to QUESTION_COUNT distinct properties, the most promising first:
0. first, anything listed below under MUST BE LEARNED FROM THE SHOPPER that has not yet been asked; ask it directly. For the budget, ask the most they are willing to spend. For a choice between two wishes, name both values together with their properties, for example "If one listing can't have both, would you rather have the <value> <property> or the <value> <property>?";
1. after that, properties that the listings' details below really contain. Read the story first (STORY HINTS below, if given, list its details and the property each relates to: they take precedence): the hidden need is usually the property that its concrete details relate to without stating the value. What the user will touch or press, look at closely or in photos, hear, measure or fit it against, read on the label, or how it was made points to that part or aspect of the item and to its property: tags, labels or the small print point to the material, fabric or ingredients; photos and appearance to the pattern, colour or the visible part; a body part or space it has to fit to that part's shape or size; buying for someone else, or for someone hardly known, to whom it is made for (gender, age group); how it has to feel to the hand, skin or foot first to the material or fabric; where, when or for which activity it is worn or used (a gym class, the office, outdoors) to its style or type, the property whose values name such uses; something fastened or adjusted over and over to its closure or mechanism; whatever does the work inside it to its core component. Weather, a job, a schedule or a place where the item is not worn or used count as background. Ask first about the property the story's details point to, using the name the listings give it. Whom the item is made for is a candidate like the rest: ask about it when the story points to someone else, or when nothing else is hinted. After that, favour properties describing what the product is like (pattern, material, shape, type or style variant, closure, length, function) over logistics (warranty, condition, pack size, variant or size lists, years). The keys are listed best first (most listings carry them and their values vary the most); among the listed keys, favour one marked as differing among the listings read (a hidden need separates listings; a key that every listing writes identically cannot), give a key marked as figures with a unit (a size, width, height, length, weight) a slot of its own whenever there is one, favour a key that most of that product's opened listings carry, and on a request for several products the product whose fitting listings are still numerous (a hidden need has to narrow the set), and never waste a slot on an option or variant picker or on a numbered duplicate of another key.
Never choose a property the request already settles (its requirements, the price, stock, quantity), never repeat one that was already asked, never use up two questions on near-duplicates (two kinds of style, two spellings of one property), and never ask about brand, maker, a product or model name, price, stock or delivery. Ask only about a property that shows up as a key in the listings' details below, using the listings' name for it; a property under STORY HINTS that matches no key below may be asked by its plain name, unless the hint is marked [no key] (enough of its product's listings were read without that field): in that case ask the key below whose values vary in the way its detail describes, or skip it. Every step-1 question is one short, plain, friendly sentence (15 words at most) asking which <property> they would like for the <item kind>, naming the property by the listings' own name for it (example values by themselves never reveal a need), optionally followed by one or two example values. No lead-in, no status update, no instructions or lists, and never a catch-all key such as options, variants, a description, a title or a name. Name the property by the exact key that one of that product's listings below writes (copied verbatim), and leave "second" empty. Each question covers ONE property of ONE product (the step-0 choice between two wishes excepted): never two properties in a single question and never properties of two products, because a reply reveals at most one need and a question naming two subjects may get no answer at all. In "second", put the next most promising property of the SAME product (not a different name for the first one), or leave it empty: the assistant may tack it onto its last message.
When the request is for SEVERAL products, the hidden needs can sit on any of them: spread the questions so that each product gets one before any gets a second (a product with no property left is set aside), take each product's properties from its own category's list below, write the property as "property (product)", and name the product in the question ("Which toe style would you like for the boots?").
Reply with ONE JSON object and nothing more: {"questions": [{"property": "...", "second": "... or empty", "question": "..."}, ...]}."""

SUBSEQUENT_SINGLE = """
Only ONE question goes out this time; the shopper's reply is read before the next one is picked. REMAINING question(s) are left altogether and each of them should be used: place the single most promising property first in the list (one or two more may follow as spares); once a need is revealed, the next question turns to a different property and story detail."""

PRIOR_ABSTAIN = """A shopping assistant is about to tell the shopper that nothing in the store fits; that is seldom the correct ending. Compare the listing records below with the requirements using these reading rules (the price has to fit the budget and the item must not be shown out of stock):
""" + PERUSE_PRINCIPLES + """
Some listings below are known only through search rows (no details were read, so their property fields are unknown). Such a row in the requested category (whatever object its title names) whose title, price and stock do not show it failing a requirement is still unchecked (figures in a title, such as 55 x 94, do not tell which dimension each one is, so they never show a row failing): put it under "unopened", the likeliest first (its title naming a required value, or the requested kind with nothing against it).
Reply with ONE JSON object and nothing more: {"possible_fits": [ids of listings that appear to meet every requirement, best first, at most 5], "unopened": [ids of rows-only listings to open before declining, at most 6], "reason": "one sentence"}."""

REFERENCE_JUMP = """A shopper's request borrows the PROPERTY of the product it wants from another listing in the store, described as: DESCRIPTION. Below are the records of listings that might be that listing. Reply with ONE JSON object and nothing more, containing:
"target": the id of the record that IS the described listing: sold by that seller, of that kind, and showing every named detail, read by meaning (a figure stated in another unit counts if it is the same amount); null when none of them is.
"missing": the details the target fails to show ([] when it shows them all).
"go_to": when the target's own text directs the reader to a DIFFERENT listing for the PROPERTY (take it from, use, refer to, look at or go by another listing), that listing as {"seller": its shop or brand name as written, "kind": its kind of product, "details": [each value the text gives for it, short and as written: a colour, a size with its unit, a material]}; otherwise null.
"value": when there is no go_to, the target's own PROPERTY value, as its record writes it.
"goes_with": the values that the target's text says go together with, pair with or count as the same as that value ([] when none).
A listing that the text names only as what the target is not ("not the green one from X"), or as a version, size or colour sold elsewhere, is not a go_to; neither is one it names for some other property. A record that does not show the described details is not the target, whatever its text claims."""

ALTERATION_SCRUTINY = """The conditions of a shopping errand have just changed: a store notice, a shopper line or a refused finish announced a new budget, a requirement dropped or added, or an item gone. Below are the request, what changed, the requirements currently in force, the priority, the cart and one card for each listing seen so far (a DETAILS card holds the full record). Re-assess the OPENED listings against the conditions as they are NOW, using these reading rules:
""" + PERUSE_PRINCIPLES + """
Reply with ONE JSON object and nothing more: {"fits": [ids of opened listings that meet every requirement now in force, fit the budget and are not shown out of stock, best first (cheapest first when the cheapest is requested)], "pick": the id to order now, or null when none fits, "cart_still_right": true when the cart's line is among the fits and is the correct pick (a basket that still fits is kept), false when the cart holds a line that no longer fits or a better pick exists under a cheapest or best rule, null when the cart is empty, "reason": "one sentence"}."""

CONCLUDE_ASSESS = """You are the final review before a shopping assistant ends an errand with an order or a decline. (Before an order, if no message reached the shopper after the last cart change, the code itself tells the shopper the basket and its total; list that only when a stated step asks for something more or different, such as a question to answer or waiting for a yes.) The request, the shopper's later lines and the store's notices may lay down steps the assistant has to carry out, some of them before or at the finish: confirming with the shopper first, telling them what the basket holds or its total, a closing remark, a final look at something. Read the instructions in force, the latest shopper lines and notices, and what was done since the last cart change, then decide whether any stated step remains undone before THIS finish. A basket step done before the last cart change does not count as done. Make nothing up: only a step the request, the shopper or the store really stated, and only one that one of the listed tools can perform. Reply with ONE JSON object and nothing more: {"pending": [one entry per undone step, in the order to run them: {"why": the stated step in a few words, "tool": the exact name of the tool that performs it, "args": its arguments exactly as that tool's schema requires; when the step is a message to the shopper, write it out in full as one or two short plain sentences with the facts the step asks for (for the basket: each cart line with its id, its count and its price, and the total)}], "ready": true when nothing is pending, else false}. Two entries at most; a step that no listed tool can perform is omitted."""

ABSTAIN_TERMS = r"refus|declin|cancel|abandon|reject|give.?up|abstain|withdraw|no.?(?:purchase|order|buy)|nothing|without"
PURCHASE_TERMS = r"commit|buy|order|purchas|check.?out|pay|confirm|complete|place|accept|proceed"
APOLOGY = "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything."
UID_RESEMBLING = r"[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}"
ENGINE_RESEMBLING = r"[A-Za-z0-9_\-]+"


# ----------------------------------------------------------------------------------------------------------- helpers
def into_passage(obj: Any, limit: int = SNAPSHOT_GLYPHS) -> str:
    # compact JSON when the value serialises (its str() otherwise); anything past `limit` characters is cut with a marker
    try:
        rendered = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except (ValueError, TypeError):
        rendered = str(obj)
    if len(rendered) > limit:
        rendered = rendered[:limit] + "...(cut)"
    return rendered


def truncated(obj: Any, chars: int, lvl: int = 0) -> Any:
    """A copy of the value in which each string is shortened to `chars`; whatever sits deeper than 8 levels becomes None."""
    if lvl >= 9:
        return None
    deeper = lvl + 1
    if isinstance(obj, str):
        return obj[:chars] + "..." if len(obj) > chars else obj
    if isinstance(obj, dict):
        return {k: truncated(v, chars, deeper) for k, v in obj.items()}
    if isinstance(obj, list):
        return [truncated(v, chars, deeper) for v in obj]
    return obj


def serialized_entity(reply: Any) -> dict[str, Any] | None:
    """Pulls the first JSON object out of a model reply; code fences and prose around it are tolerated."""
    if not isinstance(reply, str):
        return None
    core = reply.strip()
    if core.startswith("```"):
        core = re.sub(r"^```[A-Za-z]*", "", core).strip()
        if core.endswith("```"):
            core = core[:-3].strip()
    braced = core[core.find("{"): core.rfind("}") + 1] if "{" in core else ""

    def as_dict(attempt: str) -> dict[str, Any] | None:
        try:
            parsed = json.loads(attempt)
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return next((found for found in (as_dict(a) for a in (core, braced) if a) if found is not None), None)


def whether_numeric(obj: Any) -> bool:
    return not isinstance(obj, bool) and isinstance(obj, (int, float))


def lengthy_charseqs(obj: Any, collected: list[str], lvl: int = 0) -> None:
    """Gathers texts of at least four words from a string or a short list of strings (shopper lines, notices)."""
    if lvl >= 5:
        pass
    elif isinstance(obj, list):
        for member in obj[:10]:
            lengthy_charseqs(member, collected, lvl + 1)
    elif isinstance(obj, str) and len(obj.split()) >= 4:
        collected.append(obj)


def lengthy_charseqs_whole(record: dict[str, Any], out: list[str]) -> None:
    """Collects each non-blank string value of a record, and those of the maps nested directly inside it."""
    for field in record.values():
        if isinstance(field, dict):
            out.extend(str(x).strip() for x in field.values() if isinstance(x, str) and x.strip())
        elif isinstance(field, str) and field.strip():
            out.append(field.strip())


def traverse_mappings(node: Any, fn, lvl: int = 0, ceiling: int = 6) -> None:
    """Runs fn on each dict found inside the value, parents before their children."""
    if lvl <= ceiling:
        beneath = lvl + 1
        if isinstance(node, dict):
            fn(node)
            for child in node.values():
                if isinstance(child, (dict, list)):
                    traverse_mappings(child, fn, beneath, ceiling)
        elif isinstance(node, list):
            for member in node:
                traverse_mappings(member, fn, beneath, ceiling)


def rubbish_slot(field: Any) -> bool:
    """True for a field or planned property that names no product property (brand, shop, category, title, text, price,
    stock, ids), judged by its words with the planner's "(product)" suffix ignored."""
    head = re.split(r"\s*\(", str(field))[0]
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", head).lower()
    phrase = " ".join(re.findall(r"[a-z0-9]+", spaced))
    store_word = re.search(
        r"\b(?:brand|shop|seller|store|vendor|manufacturer|category|categories|department|title|headline|"
        r"heading|listing|description|overview|price|cost|amount|availability|sku|url)\b", phrase)
    return store_word is not None or re.fullmatch(
        r"(?:name|notes?|stock|in stock|on hand|images?|links?|make|maker|shelf|ids?|id)", phrase) is not None


def trait_slot(label: Any) -> str:
    """The comparison form of a property name: prop_words glued together, so "toeStyle" and "toe styles" both become
    "toestyle" (suffix in brackets dropped, camelCase split, letters only, plural s removed)."""
    return "".join(trait_terms(label))


def revisions_amid(left: str, right: str, most: int) -> bool:
    """Whether two short strings are no more than `most` edits apart."""
    if abs(len(left) - len(right)) > most:
        return False
    row = list(range(len(right) + 1))
    for i, ch_l in enumerate(left, 1):
        nxt = [i]
        for j, ch_r in enumerate(right, 1):
            nxt.append(min(row[j] + 1, nxt[j - 1] + 1, row[j - 1] + (ch_l != ch_r)))
        if min(nxt) > most:
            return False
        row = nxt
    return row[-1] <= most


def identical_trait(first: Any, second: Any) -> bool:
    """Loose test that two property names mean one property: identical after prop_key; a spelling variant sharing the
    first three letters and at most 1 edit away (2 once the shorter has 6+ letters: flavor, flavour); or the shorter, of
    5+ letters, opening the longer (sleeve, sleeve length)."""
    x, y = trait_slot(first), trait_slot(second)
    if not (x and y):
        return False
    n = min(len(x), len(y))
    if x == y or (n >= 4 and x[:3] == y[:3] and revisions_amid(x, y, 2 if n >= 6 else 1)):
        return True
    return n >= 5 and (x.startswith(y) or y.startswith(x))


def trait_terms(label: Any) -> list[str]:
    """A property name split into its comparison words: bracketed "(product)" suffix dropped, camelCase split, lower-case
    letter runs only, a plural s removed."""
    head = re.split(r"\s*\(", str(label))[0]
    tokens = re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", head).lower())

    def singular(tok: str) -> str:
        if tok[-1:] == "s" and tok[-2:] != "ss" and len(tok) > 3:
            return tok[:-1]
        return tok
    return [singular(t) for t in tokens]


def phrased_similar(base: str, variant: str) -> bool:
    """`variant` spells `base` differently: one or two extra letters, each a vowel or a doubled letter (color/colour,
    jewelry/jewellery), or the same letters with one neighbouring pair swapped (center/centre)."""
    extra = len(variant) - len(base)
    if extra == 0:
        moved = [k for k in range(len(base)) if base[k] != variant[k]]
        if len(moved) != 2:
            return False
        p, q = moved
        return q == p + 1 and base[p] == variant[q] and base[q] == variant[p]
    if extra != 1 and extra != 2:
        return False

    def soft(word: str, k: int) -> bool:
        if word[k] in "aeiouy":
            return True
        return (k > 0 and word[k - 1] == word[k]) or (k + 1 < len(word) and word[k + 1] == word[k])

    def drops(word: str):
        # the word with each soft letter left out in turn
        return (word[:k] + word[k + 1:] for k in range(len(word)) if soft(word, k))
    if extra == 1:
        return any(once == base for once in drops(variant))
    return any(twice == base for once in drops(variant) for twice in drops(once))


def rigorous_trait(first: Any, second: Any) -> bool:
    """Strict test (used when choosing questions) that two property names mean one property: identical after prop_key;
    a spelling variant of 4+ letters sharing the first three (spelled_alike); or the same words with one extra trailing
    token of 1-3 letters, a unit or code (height, height cm). A longer name opening with the other (display, display
    size; protein, protein source) or one changed letter (shade, shape; collar, colour) counts as a different property."""
    x, y = trait_slot(first), trait_slot(second)
    if not (x and y):
        return False
    shorter, longer = sorted((x, y), key=len)
    if x == y or (len(shorter) >= 4 and shorter[:3] == longer[:3] and phrased_similar(shorter, longer)):
        return True
    fewer, more = sorted((trait_terms(first), trait_terms(second)), key=len)
    n = len(fewer)
    if len(more) != n + 1 or more[:n] != fewer:
        return False
    return len(more[-1]) <= 3 and len("".join(fewer)) >= 5


BOUNDARY_TERMS = (r"\b(?:no|not)\s+(?:more|less|fewer|over|under|above|"
               r"below|beyond|exceeding)\b")
# (CH) the words of a limit
CROSSCHECK_BOUNDARY = (r"\b(?:at least|at most|no more than|no less than|more than|less than|up to|or more|or less|or larger|"
            r"or smaller|or higher|or lower|minimum|maximum|between|within|under|over|above|below|than)\b")
# (CJ2) comparison phrasing before a value
CROSSJUDGE_CONTRAST = (r"\b(?:falls?\s+short\s+of|short\s+of|nothing\s+(?:less|other)\s+than|"
              r"nothing\s+short\s+of|nothing\s+but|no\s+(?:less|worse)\s+than|at\s+least|"
              r"(?:less|more|better|worse)\s+than|exactly|only|just)\b")
CROSSCHECK_BYPASS = set("and the a an of".split())
# (CJ2) per errand: word values written by the store that a shopper line gave after a comparison. Module state (v605d),
# emptied by Errand.__init__; safe because a process runs one errand at a time and agent_main never runs concurrently.
VERBATIM_SETTINGS: set[str] = set()


def crosscheck_terms(raw: Any) -> list[str]:
    """CH: a value's words and figures, in order: lower-case runs of letters and digits (hyphens, slashes and plus signs
    divide them, so 'mid-heel' yields mid and heel), minus a few link words; negations are kept ('not available',
    'no warranty')."""
    tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", str(raw or "").lower())
    return [t for t in tokens if t not in CROSSCHECK_BYPASS]


def crosscheck_term_among(token: str, among: list[str]) -> bool:
    """CH: whether a value's word occurs among a field value's words, counting an identical word, its singular or
    plural (same_kind), and, when both have 6+ letters and the same first two, a one-letter slip (polyster, polyester)."""
    def matches(cand: str) -> bool:
        if cand == token:
            return True
        if not (token.isalpha() and cand.isalpha()):
            return False
        if identical_category(token, cand):
            return True
        return min(len(token), len(cand)) >= 6 and token[:2] == cand[:2] and revisions_amid(token, cand, 1)
    return any(matches(c) for c in among)


def crosscheck_variants(required: Any) -> list[list[str]]:
    """CH: the alternatives a required value allows (split at 'or'; a comma list stays one alternative, all of it
    wanted), each given as its words, after dropping '(stated)' / '(revealed ...)' marks, quotes and the emphasis words
    only, just, exactly, strictly."""
    text = re.sub(r"\((?:stated|revealed[^)]*)\)", " ", str(required or "").lower()).replace('"', " ")
    text = re.sub(r"\b(?:only|just|exactly|strictly)\b", " ", text)
    alternatives = []
    for chunk in re.split(r"\bor\b", text):
        chunk_words = crosscheck_terms(chunk)
        if chunk_words:
            alternatives.append(chunk_words)
    return alternatives


def crosscheck_inscribes(required: Any, field_text: Any) -> bool:
    """CH: whether a field value spells out a required value's own words (see ch_match)."""
    wanted = crosscheck_variants(required)
    have = crosscheck_terms(field_text)
    return crosscheck_correspond(wanted, have)


def crosscheck_correspond(options: list[list[str]], words: list[str]) -> bool:
    """CH: whether a field value's words spell one of a required value's alternatives (ch_alts): either each of its
    words and figures is found among them (containment is enough: 'forest green' covers green), or the alternative
    glued together equals a stretch of consecutive field words glued together (spacing variants: '18 k' / '18k',
    'wi-fi' / 'wifi')."""
    if not words:
        return False

    def glued_run(target: str) -> bool:
        for start in range(len(words)):
            acc = ""
            for piece in words[start:]:
                acc += piece
                if acc == target:
                    return True
                if len(acc) >= len(target):
                    break
        return False
    for option in options:
        if all(crosscheck_term_among(w, words) for w in option):
            return True
        glued = "".join(option)
        spacing_matters = len(option) >= 2 or re.search(r"\d", glued) or any(re.search(r"\d", p) for p in words)
        if spacing_matters and glued_run(glued):
            return True
    return False


def crosscheck_contacts(required: Any, field_text: Any) -> bool:
    """CH: whether a field value overlaps a required value at all: a shared word (ch_word_in), a shared figure, or the
    first three letters of one word opening a word on the other side (abbreviations such as 'win10' for windows)."""
    words = crosscheck_terms(field_text)
    lowered = str(field_text or "").lower()

    def overlaps(w: str) -> bool:
        if crosscheck_term_among(w, words):
            return True
        if re.fullmatch(r"\d+(?:\.\d+)?", w) and re.search(r"(?<![\d.])" + re.escape(w) + r"(?!\d|\.\d)", lowered):
            return True
        return len(w) >= 3 and any(len(p) >= 3 and (p.startswith(w[:3]) or w.startswith(p[:3])) for p in words)
    return any(overlaps(w) for option in crosscheck_variants(required) for w in option)


def crosscheck_portions(field_text: Any) -> list[str]:
    """CH: a field value's elements: the items of a list split at commas, semicolons and bars, or else the value itself."""
    whole = str(field_text or "").strip().lower()
    pieces = [p.strip() for p in re.split(r"[,;|]", whole) if p.strip()]
    if pieces:
        return pieces
    return [whole] if whole else []


def crosscheck_flanks(element: Any) -> list[str]:
    """CH: the two or more sides of an element joined with '+' ('a + b'), or [] when nothing is joined ('with' is not
    treated as a join, since it may name a trim or a component)."""
    halves = [s.strip() for s in re.split(r"\s*\+\s*", str(element)) if s.strip()]
    return halves if len(halves) > 1 else []


def numericval_inscribed(figure: str, request_low: str) -> bool:
    """CI: whether the request writes this figure as a standalone number (not part of a longer one) that either has a
    decimal point or is followed by a symbol, the end of the text or a unit of up to three letters ('6.5 m', '12"',
    '20 MP', '5 kg') - not by a longer word ('3 kids')."""
    decimal = "." in figure
    for hit in re.finditer(r"(?<![\d.])" + re.escape(figure) + r"(?!\d|\.\d)", request_low):
        unit = re.match(r"\s*([a-z]+)", request_low[hit.end():])
        if decimal or unit is None or len(unit.group(1)) <= 3:
            return True
    return False


OMIT_TERMS = (r"\b(?:no(?!-)|not|without|nothing|never|neither|nor|except|excluding|avoid\w*|"
                 r"anything but|other than|rather than|instead of)\b")


def affirmative_setting(value: Any, prop: Any = "") -> bool:
    """Whether a requirement's value says what the item is. Yes/no values do. Descriptions do not (an alternative or
    clause with 4+ content words: 'suits the newer kind of board'; a list of short alternatives still is a value).
    Exclusions (no, not, without, nothing, never, neither/nor, except, avoid, anything but, other than) do not either,
    unless a value precedes them ('cotton, nothing in leather' means cotton) or they only negate the property's own name
    ('machine washable: not machine washable' is a no). Bounds are not exclusions: 'no more than', a comparative ahead of
    'than' ('no thicker than'), or one inside a value carrying a figure ('0.39 inches, no thicker'). For exclusions and
    descriptions the property's real value stays unknown."""
    text = str(value).lower().strip()
    if re.fullmatch(r"\W*(?:yes|no|true|false|none|required|included|needed)\W*", text):
        return True
    if VERBATIM_SETTINGS:
        # (CJ2) a store-written value the shopper gave after a comparison, such as 'not available'
        bare = re.sub(r"\((?:stated|revealed[^)]*)\)", " ", text)
        if " ".join(re.findall(r"[a-z0-9]+", bare)) in VERBATIM_SETTINGS:
            return True

    def content_words(chunk: str) -> list[str]:
        return [w for w in re.findall(r"[a-z]{3,}", chunk) if w not in BOND_TERMS]
    text = re.sub(BOUNDARY_TERMS, " ", text)
    text = re.sub(r"\b(?:no|not)\s+[a-z]+er\b(?=\s+than\b)", " ", text)
    if re.search(r"\d", text):
        text = re.sub(r"\b(?:no|not)\s+[a-z]+er\b", " ", text)
    unlinked = re.sub(r"\b(?:" + SETTING_BONDS + r")\b", " ", text)
    clauses = re.split(r",|;|/|&|\(|\)|\bor\b|\band\b", unlinked)
    if any(len(content_words(c)) >= 4 for c in clauses):
        return False
    neg = re.search(OMIT_TERMS, text)
    if neg is None or content_words(text[:neg.start()]):
        return True
    after = content_words(text[neg.end():])
    own = set(trait_terms(prop)) | set(re.findall(r"[a-z]+", str(prop).lower()))
    return bool(after) and bool(own) and all(w in own or w.rstrip("s") in own for w in after)


def setting_observed(required: Any, sample_text: Any) -> bool:
    """Whether a field's sample values contain one of a requirement value's words (3+ letters, link and comparison words
    ignored, longer words matched by their stem) or one of its figure runs (4:3, 1920 x 1080, 11)."""
    hay = str(sample_text or "").lower()
    if not hay.strip():
        return False
    unlinked = re.sub(r"\b(?:" + SETTING_BONDS + r")\b", " ", str(required).lower())
    stems = [w[:-1] if len(w) > 4 else w for w in re.findall(r"[a-z]{3,}", unlinked) if w not in BOND_TERMS]
    if any(re.search(r"\b" + re.escape(s), hay) for s in stems):
        return True
    for run in re.findall(r"\d+(?:\.\d+)?(?:\s*[:x/]\s*\d+(?:\.\d+)?)*", unlinked):
        chain = r"\D{0,4}".join(re.escape(f) for f in re.findall(r"\d+(?:\.\d+)?", run))
        if re.search(r"(?<![\d.])" + chain + r"(?![\d.])", hay):
            return True
    return False


def category_terms(name: Any) -> set[str]:
    """The descriptive words of a product or category name: 3+ letters, link words dropped, trailing s removed."""
    found = re.findall(r"[a-z]{3,}", str(name).lower())
    return {w.rstrip("s") for w in found if w not in BOND_TERMS}


def category_lexemes(name: Any) -> list[str]:
    """CA: the descriptive words of a product or category name as written, in order: 3+ letters, with link words and
    possessive 's removed ("women's flats" yields women, flats)."""
    simple = re.sub(r"'s\b", "", str(name).lower().replace("’", "'"))
    return [w for w in re.findall(r"[a-z]{3,}", simple) if w not in BOND_TERMS]


def multiple_layouts(token: Any) -> set[str]:
    """CA: the singular and plural spellings a kind word may take, the word included: +s, +es, y/ies, f or fe/ves, and
    the reverse of each ('cookie'/'cookies', 'lens'/'lenses', 'battery'/'batteries')."""
    base = str(token)
    grown = [base, base + "s", base + "es"]
    if base[-1:] == "y":
        grown.append(base[:-1] + "ies")
    if base[-2:] == "fe":
        grown.append(base[:-2] + "ves")
    elif base[-1:] == "f":
        grown.append(base[:-1] + "ves")
    for tail, cut, endings in (("ies", 3, ("y",)), ("ves", 3, ("f", "fe")), ("es", 2, ("",)), ("s", 1, ("",))):
        if base.endswith(tail):
            grown.extend(base[:-cut] + e for e in endings)
    return {g for g in grown if len(g) >= 3}


def identical_category(first: Any, second: Any) -> bool:
    """CA: whether two kind words name the same kind: equal, or either is a singular/plural spelling of the other
    (checked both ways, so no one-sided stem must agree)."""
    if first == second:
        return True
    return str(second) in multiple_layouts(first) or str(first) in multiple_layouts(second)


def category_among(token: Any, among: Any) -> bool:
    """CA: whether the kind of a kind word appears among the words of `among`."""
    return any(identical_category(token, other) for other in among)


def categories_amid(tokens: Any, among: Any) -> bool:
    """CA: whether each kind word of `tokens` finds its kind among the words of `among`."""
    return all(category_among(t, among) for t in tokens)


def category_rating(mine: Any, theirs: Any, common: Any) -> float:
    """CC: how far `theirs` covers a name's kind words `mine` (singular and plural alike, via same_kind); a word that the
    other products' names also carry earns half."""
    total = 0
    for w in mine:
        if category_among(w, theirs):
            total += 0.5 if category_among(w, common) else 1.0
    return total


def moniker_motif(product_name: Any) -> str:
    """CA: a regex for a product name's kind words, in order, inside lower-case text - each in any singular or plural
    spelling, possessive allowed ('battery chargers' matches 'battery charger' and 'batteries chargers'); '' when the
    name has no kind word."""
    alternations = []
    for tok in category_lexemes(product_name):
        spellings = sorted(multiple_layouts(tok), key=lambda s: (-len(s), s))
        alternations.append("(?:" + "|".join(map(re.escape, spellings)) + r")(?:'s\b|'|\b)")
    return r"[\s'-]+".join(alternations)


SETTING_BONDS = (r"at least|at most|no more than|no less than|more than|less than|up to|or more|or less|or larger|"
               r"or smaller|or higher|or lower|minimum|maximum|between|within|under|over|above|below|exactly|only|"
               r"either|neither|nor|than|plus")


def single_trait_clue(line: str) -> str:
    """Turns a hint offering alternative properties ("colour / style (chair): detail") into one about the first, the
    rest moved into its detail ("colour (chair): detail (or style)")."""
    try:
        head, colon, rest = line.partition(":")
        if not colon or "/" not in head:
            return line
        tag = re.search(r"\(([^()]*)\)\s*$", head)
        listed = head if tag is None else head[: tag.start()]
        names = [n.strip() for n in listed.split("/") if n.strip()]
        if len(names) <= 1:
            return line
        suffix = "" if tag is None else " (" + tag.group(1).strip() + ")"
        others = ", ".join(names[1:])
        return f"{names[0]}{suffix}:{rest} (or {others})"
    except Exception:
        return line


def foremost_setting(hint_detail: Any, request_text: str, product_name: str) -> str:
    """The opening word of a hint's detail when it stands on its own there (followed by a comma, a bracket, a link word
    or the end) and the request puts it right before the product's own name ('a sexy jumpsuit or romper' for the detail
    'sexy, for a dinner' of the product 'jumpsuit or romper'); otherwise ''."""
    word = ""
    try:
        links = "|".join(sorted(BOND_TERMS))
        opener = re.match(r"\s*([a-z][a-z-]*[a-z])(?:\s*[,;.()]|\s*$|\s+(?:" + links + r")\b)", str(hint_detail).lower())
        if opener is not None:
            cand = opener.group(1)
            kinds = [k for k in re.findall(r"[a-z]{3,}", str(product_name).lower()) if k not in BOND_TERMS]
            viable = len(cand) >= 3 and cand not in BOND_TERMS and kinds
            if viable and cand.rstrip("s") not in {k.rstrip("s") for k in kinds}:
                head_kind = kinds[0].rstrip("s")
                seq = re.findall(r"[a-z][a-z'-]*", str(request_text).lower())
                if any(a == cand and b.split("'")[0].rstrip("s") == head_kind for a, b in zip(seq, seq[1:])):
                    word = cand
    except Exception:
        return ""
    return word


def narrative_estimate(need: Any, request_text: str, pointers: list[str]) -> list[str]:
    """The value words of a requirement missing from the request, in order ([] when all are written, or for a
    reference): words of 3+ letters (plurals, and words matching all but their last 1-3 letters, count as written) and
    figure runs (4:3, 1920 x 1080 - never a lone figure, which the request may spell out); comparison and link words are
    ignored; a yes/no value is judged by the name of its property."""
    need = str(need)
    head, _, tail = need.partition(":")
    wanted = tail.strip().lower()
    pointing = re.search(r"as shown|same as|the page|listing|sold by|redirect|refer", wanted) if wanted else None
    if not wanted or need in pointers or pointing:
        return []
    if re.fullmatch(r"\W*(?:yes|no|true|false|required|included|needed)\W*", wanted):
        wanted = re.split(r"\s*\(", head)[0].lower()
    unlinked = re.sub(r"\b(?:" + SETTING_BONDS + r")\b", " ", wanted)
    request_low = str(request_text).lower()
    figures = re.findall(r"\d+(?:\.\d+)?", unlinked)
    if figures and all(numericval_inscribed(f, request_low) for f in figures):
        # (CI) every figure in the value appears in the request; a differently spelt unit is not a guess
        return []
    seen_words = set(re.findall(r"[a-z]+", request_low))
    seen_stems = {s[:-1] if len(s) > 3 and s.endswith("s") else s for s in seen_words}

    def written(w: str) -> bool:
        # (written: an inflection or plural, a one-edit spelling variant (grey/gray), the opening of a compound the
        # request writes (water/waterproof), or a stem equal up to its final 3 letters (zipper/zips))
        if w.endswith("es") and len(w) > 4:
            wordbase = w[:-2]
        elif w.endswith("s") and len(w) > 3:
            wordbase = w[:-1]
        else:
            wordbase = w
        long_w = len(w) >= 4
        for s in seen_words:
            if s == w or (s.startswith(wordbase) and len(s) <= len(wordbase) + 3):
                return True
            if wordbase.startswith(s) and len(s) >= max(4, len(wordbase) - 2):
                return True
            if long_w and (s.startswith(w) or (len(s) >= 4 and s[0] == w[0] and revisions_amid(s, w, 1))):
                return True
        return any(len(s) >= 3 and wordbase.startswith(s) and len(s) >= len(wordbase) - 3 for s in seen_stems)
    words = [w for w in re.findall(r"[a-z]{3,}", unlinked) if w not in BOND_TERMS]
    missing = [w for w in words if not written(w)]
    if len(words) <= 2 and any(len(w) >= 4 and w not in missing for w in words):
        # (a short value whose core word the request does write counts as written: 'genuine leather' for 'real leather')
        return []
    for run in re.findall(r"\d+(?:\.\d+)?(?:\s*[:x/]\s*\d+(?:\.\d+)?)*", unlinked):
        numbers = re.findall(r"\d+(?:\.\d+)?", run)
        chain = r"\D{0,4}".join(re.escape(f) for f in numbers)
        if len(numbers) >= 2 and re.search(r"(?<![\d.])" + chain + r"(?!\d)", request_low) is None:
            missing.append(run)
    return missing


REFERENCE_TERMS = (r"\bas shown\b|\bshown on\b|\bsame as\b|\bpage\b|\bsold by\b|\bredirect|\brefer(?:s|red|ring)?\b|"
             r"\blisting points\b|\bpoints you\b")
# share of the steps the code's own reference reads and searches may take (8 at the least)
REFERENCE_STAGES = 0.4
# cap on reference hop calls in one errand
REFERENCE_INVOKES = 12
# cap on listings followed for a single reference
REFERENCE_JUMPS = 8
# (CE2) relation words turning a reference into an exclusion: the value has to differ from that listing's. 'set apart
# from' sits next to 'apart from' because it starts the pointing phrase at its verb, so the need reads 'set apart from
# ...' rather than 'apart from ...', which would mean 'besides'.
DIVERGE_TERMS = (r"\bset apart from\b|\bapart from\b|\bdistinct from\b|\bdifferent (?:from|to|than)\b"
                r"|\bdiffer(?:s|ent|ing)?\s+from\b|\bother than\b|\bunlike\b|\bnot the same as\b|\banything but\b")
# (CE2b) words that start the pointing part of a reference phrase (alongside REF_WORDS and DIFFER_WORDS)
INDICATE_TERMS = (r"\bwithin\b|\bat (?:most|least)\b|\bno (?:more|less) than\b|\bsame as\b|\bas shown\b"
               r"|\bthe value\b|\bthe one\b")
# (CE) when only differ references remain open, the code's reference steps end at this share (one excluded value)
REFERENCE_DIVERGE_STAGES = 0.3
# (CE) ... and at this count while no listing from any open differ reference's chain has been found
REFERENCE_DIVERGE_UNLOOKED = 5
# (CE1) field names meaning the listing's seller (generic shop words; the store's own seller fields are learnt in seller_keys)
VENDOR_SLOTS = (r"shop|seller|store|vendor|brand|make|manufactur|"
               r"merchant|retailer|supplier|sold.?by")


def reference_passage(required: Any) -> bool:
    """Whether a requirement value points at another listing or page ('as shown on the listing of the lamp sold by X')."""
    return re.search(REFERENCE_TERMS, str(required or "").lower()) is not None


def reference_pieces(raw: Any) -> list[str]:
    """A text's comparable tokens: lower-case words and figures, a figure fused with the short unit word right behind it
    ('12 v', '12-v' and '12v' all give 12v; '300 cd/m2' gives 300cd and m2), plural s removed, link words and single
    letters skipped."""
    def fuse(m):
        return m.group(0) if m.group(2) in BOND_TERMS else m.group(1) + m.group(2)
    fused = re.sub(r"(\d)(?:\s+|\s*-\s*)([a-z]{1,8})\b", fuse, str(raw or "").lower())

    def reduced(tok: str) -> str:
        if len(tok) <= 3 or not tok.endswith("s") or tok.endswith("ss") or tok[-2].isdigit():
            return tok
        return tok[:-2] if re.search(r"(?:ch|sh|x|z)es$", tok) else tok[:-1]
    found = re.findall(r"\d+(?:[.,:/]\d+)*[a-z]*|[a-z][a-z0-9]*", fused)
    return [reduced(t) for t in found if t not in BOND_TERMS and (len(t) != 1 or t.isdigit())]


def reference_numerals(raw: Any) -> set[tuple[float, str]]:
    """The (figure, unit word) pairs a text writes - a figure and the word just after it ('12 v', '1.35v', '24 inches'
    gives 24 and inch; '1066/1333mhz' pairs every figure with the unit) - with plurals reduced the way ref_toks does."""
    pairs: set[tuple[float, str]] = set()
    try:
        flat = str(raw or "").lower().replace(",", "")
        for m in re.finditer(r"(\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)*)\s*-?\s*([a-z]{1,8})\b", flat):
            unit = m.group(2)
            if len(unit) > 2 and unit[-1] == "s" and unit[-2:] != "ss":
                cut = 2 if re.search(r"(?:ch|sh|x|z)es$", unit) else 1
                unit = unit[:-cut]
            if unit not in BOND_TERMS:
                for number in re.split(r"\s*/\s*", m.group(1)):
                    pairs.add((float(number), unit))
    except Exception:
        pass
    return pairs


def reference_moniker(seller: Any) -> str:
    """A seller name in comparison form: only lower-case letters and digits, possessive 's removed."""
    no_possessive = re.sub(r"'s\b", "", str(seller or "").lower())
    return re.sub(r"[^a-z0-9]", "", no_possessive)


def reference_relationship(required: Any) -> tuple[str, float | None, str]:
    """The (relation, tolerance, unit) that a reference requirement's wording sets: 'differ' (CE2: a differ word such as
    'different from', 'other than', 'unlike' - checked before any bound word, so a bound word followed by a differ phrase
    is still a differ), 'within' with its figure and unit ('within 5 units above or below the one shown...'),
    'at most', 'at least', and otherwise 'same'."""
    wording = str(required or "").lower()
    if re.search(DIVERGE_TERMS, wording) is not None:
        return "differ", None, ""
    span = re.search(r"\bwithin\s+(\d+(?:\.\d+)?)\s*([a-z%]+)?", wording)
    if span:
        return "within", float(span.group(1)), (span.group(2) or "").strip()
    for relation, pattern in (("at most", r"\b(?:at most|no more than|up to|or less|or lower|or smaller)\b"),
                              ("at least", r"\b(?:at least|no less than|or more|or higher|or larger)\b")):
        if re.search(pattern, wording):
            return relation, None, ""
    return ("same", None, "")


def reference_decode(phrase: Any, accept: Any = None) -> dict[str, Any] | None:
    """{seller, kind, details} taken from a reference's own wording ('the kettle sold by Abcdef that is steel, 2l and
    new'; 'the Ghijkl mugs & cups with matte finish, ...'). The seller is the first capitalised name (a capital plus 3+
    lower-case letters, possessive 's allowed) not at the very start; the kind is the product words before 'sold by' or
    straight after the name; the details are the pieces after that, split at commas, 'and', 'with', 'in', 'at' and
    'that is/has/are/comes as'. Everything from '(or ...' on is ignored. None when no name is found. Given `accept`,
    the seller is the first such name `accept` takes (BW5: a capitalised product or category word ahead of it is
    skipped)."""
    try:
        text = re.split(r"\(\s*or\b", str(phrase or ""))[0]
        names = re.finditer(r"(?<![A-Za-z'])([A-Z][a-z]{3,})(?:'s)?(?![A-Za-z])", text)
        seller = next((m for m in names if m.start() > 0 and (accept is None or accept(m.group(1)))), None)
        if seller is None:
            return None
        ahead, behind = text[:seller.start()], text[seller.end():]
        kind = ""
        by_line = re.search(r".*(?:\bthe|\bfor|\ba|\ban)\s+(.*?)\s+(?:sold|made|offered|listed)\s+by\s*$", ahead,
                            re.I | re.S)
        opening = None if by_line else re.match(
            r"\s*([^,;(]*?)(?=\s+(?:with|that|which|in|at|having|from|of|for|sold)\b|[,;]|$)", behind, re.I)
        if by_line:
            kind = by_line.group(1)
        elif opening:
            kind, behind = opening.group(1), behind[opening.end():]
        splitter = r",|;|\band\b|\bwith\b|\bthat\s+(?:is|are|has|have|comes?(?:\s+as)?)\b|\bwhich\b|\bin\b|\bat\b"
        details = []
        for chunk in re.split(splitter, behind, flags=re.I):
            kept = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9.%:/&'-]*", chunk) if w.lower() not in BOND_TERMS]
            if kept:
                details.append(" ".join(kept))
        return dict(seller=seller.group(1), kind=kind.strip()[:80], details=details[:6])
    except Exception:
        return None


def trait_titled(prop_name: Any, raw: Any, exact: bool = False) -> bool:
    """CE2b: whether a text names a property: the property's words (bracketed "(product)" suffix dropped, camelCase and
    separators split) occur as one consecutive run of the text's words, each identical or, unless exact, its plural."""
    head = re.split(r"\s*\(", str(prop_name or ""))[0]
    want = re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", head).lower())
    have = re.findall(r"[a-z0-9]+", str(raw or "").lower())
    if not want or len(" ".join(want)) < 3:
        return False

    def same(tok: str, w: str) -> bool:
        if tok == w:
            return True
        if exact:
            return False
        return (len(w) >= 3 and tok in (w + "s", w + "es")) or (len(tok) >= 3 and w in (tok + "s", tok + "es"))
    span = len(want)
    return any(all(same(t, w) for t, w in zip(have[i: i + span], want)) for i in range(len(have) - span + 1))


def reference_stretch(prop_name: Any, sheet_passage: Any) -> str:
    """The text of `page_text` just after its first mention of the property, up to the end of that sentence (or
    '(or')."""
    try:
        head = re.split(r"\s*\(", str(prop_name))[0].lower()
        words = [w for w in re.findall(r"[a-z0-9]+", head) if w not in BOND_TERMS]
        hit = re.search(r"\b" + r"\W+".join(map(re.escape, words)) + r"\b", str(sheet_passage).lower()) if words else None
        if hit is None:
            return ""
        window = str(sheet_passage)[hit.end(): hit.end() + 500]
        return re.split(r"\(\s*or\b|(?<=[.!?])\s", window)[0]
    except Exception:
        return ""


def focused(obs: Any) -> Any:
    """A copy of an observation in which listing-like records that lack the plain names the code reads (price, in_stock,
    title) gain them, each copied from the single field whose name and value type fit (worlds rename these fields)."""
    try:
        copy = json.loads(json.dumps(obs))
    except (ValueError, TypeError):
        return obs

    def fill(rec: dict[str, Any]) -> None:
        if not any(isinstance(v, str) and "id" in str(k).lower() and re.fullmatch(UID_RESEMBLING, v) for k, v in rec.items()):
            return
        lowered = {str(k).lower(): k for k in rec}

        def adopt(target: str, fits) -> None:
            hits = [k for low, k in lowered.items() if fits(low, rec[k])]
            if len(hits) == 1:
                rec[target] = rec[hits[0]]
        if not any(re.search(r"price|cost", low) and whether_numeric(rec[k]) for low, k in lowered.items()):
            adopt("price", lambda low, v: whether_numeric(v) and re.search(r"amount|fee|charge|pay|rate|value", low)
                  and not re.search(r"total|count|qty|quant|unit|due|sum|rating|star|score|page|number|num|size|weight|"
                                    r"width|height|length|found|match", low))
        if not any(re.search(r"stock|avail", low) for low in lowered):
            adopt("in_stock", lambda low, v: isinstance(v, bool)
                  and re.search(r"hand|orderable|buyable|sellable|ready|ship|instock|purchas|in_store", low))
        if not any(re.search(r"title|heading", low) for low in lowered):
            adopt("title", lambda low, v: isinstance(v, str) and re.search(r"headline|name|label|caption", low)
                  and not re.search(r"shop|seller|store|vendor|brand|make|categor|shelf|department|id", low))
    traverse_mappings(copy, fill)
    return copy


# ----------------------------------------------------------------------------------------------------------- schema
class ParamAssess:
    """Nudges arguments a model wrote toward a JSON schema and reports whatever still violates it."""

    TYPES = dict(
        string=lambda x: isinstance(x, str),
        integer=lambda x: not isinstance(x, bool) and isinstance(x, int),
        number=whether_numeric,
        boolean=lambda x: isinstance(x, bool),
        object=lambda x: isinstance(x, dict),
        array=lambda x: isinstance(x, list),
        null=lambda x: x is None,
    )

    @classmethod
    def genre_roster(cls, spec: dict[str, Any]) -> list[str]:
        declared = spec.get("type")
        if isinstance(declared, str):
            return [declared]
        return list(declared or [])

    @classmethod
    def complies(cls, val: Any, tname: str) -> bool:
        check = cls.TYPES.get(tname)
        return True if check is None else check(val)

    @classmethod
    def transform(cls, val: Any, allowed: list[str]) -> Any:
        if not allowed:
            return val
        whole_float = isinstance(val, float) and val.is_integer()
        if any(cls.complies(val, t) for t in allowed):
            return int(val) if whole_float and "integer" in allowed else val
        text = val if isinstance(val, str) else None
        for wanted in allowed:
            if wanted == "integer":
                if whole_float:
                    return int(val)
                if text is not None and re.fullmatch(r"\s*-?\d+(\.0+)?\s*", text):
                    return int(float(text))
            elif wanted == "number":
                if text is not None and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", text):
                    as_float = float(text)
                    return as_float if not as_float.is_integer() else int(as_float)
            elif wanted == "string":
                if whether_numeric(val):
                    return str(val)
            elif wanted == "boolean" and text is not None and text.lower() in ("true", "false"):
                return text.lower() == "true"
        return val

    @classmethod
    def carryout(cls, val: Any, spec: Any, at: str = "args") -> tuple[Any, list[str]]:
        if not isinstance(spec, dict):
            return val, []
        kinds = cls.genre_roster(spec)
        value = cls.transform(val, kinds)
        if kinds and not any(cls.complies(value, k) for k in kinds):
            return value, [f"{at}: should be of type {'/'.join(kinds)}"]
        problems: list[str] = []
        for keyword, rule in cls.RULES:
            if keyword in spec:
                value, found = rule(value, spec, at)
                problems.extend(found)
        return value, problems

    @staticmethod
    def _enum(val, spec, at):
        choices = spec["enum"]
        if val in choices:
            return val, []
        wanted = val.strip().lower() if isinstance(val, str) else None
        if wanted is not None:
            hit = next((c for c in choices if isinstance(c, str) and c.lower() == wanted), None)
            if hit is not None:
                return hit, []
        return val, [f"{at}: has to be one of {choices}"]

    @staticmethod
    def _const(val, spec, at):
        fixed = spec["const"]
        return val, ([f"{at}: has to be exactly {fixed!r}"] if val != fixed else [])

    @staticmethod
    def _min_length(val, spec, at):
        floor = spec["minLength"]
        too_short = isinstance(val, str) and len(val) < floor
        return val, ([f"{at}: fewer than {floor} characters"] if too_short else [])

    @staticmethod
    def _max_length(val, spec, at):
        trimmed = val[: spec["maxLength"]] if isinstance(val, str) else val
        return trimmed, []

    @staticmethod
    def _boundary(keyword, broken, phrase):
        def check_bound(val, spec, at):
            edge = spec[keyword]
            violated = whether_numeric(val) and whether_numeric(edge) and broken(val, edge)
            return val, ([f"{at}: {phrase} {edge}"] if violated else [])
        return check_bound

    @staticmethod
    def _min_items(val, spec, at):
        least = spec["minItems"]
        short = isinstance(val, list) and len(val) < least
        return val, ([f"{at}: must hold at least {least} items"] if short else [])

    @staticmethod
    def _max_items(val, spec, at):
        most = spec["maxItems"]
        over = isinstance(val, list) and len(val) > most
        return val, ([f"{at}: may hold at most {most} items"] if over else [])

    @classmethod
    def _items(cls, val, spec, at):
        member_schema = spec["items"]
        if not (isinstance(val, list) and isinstance(member_schema, dict)):
            return val, []
        problems: list[str] = []
        repaired = []
        for pos, member in enumerate(val):
            fixed, found = cls.carryout(member, member_schema, f"{at}[{pos}]")
            repaired.append(fixed)
            problems.extend(found)
        return repaired, problems

    @classmethod
    def _qualities(cls, val, spec, at):
        if not isinstance(val, dict):
            return val, []
        declared = spec.get("properties") or {}
        problems = [f"{at}.{arg}: required" for arg in spec.get("required") or [] if arg not in val]
        if spec.get("additionalProperties") is False:
            problems.extend(f"{at}.{arg}: not an argument this tool accepts" for arg in val if arg not in declared)
        repaired = {}
        for arg, given in val.items():
            if arg in declared:
                given, found = cls.carryout(given, declared[arg], f"{at}.{arg}")
                problems.extend(found)
            repaired[arg] = given
        return repaired, problems


# (the rules run in this order: enum, const, string lengths, the four numeric bounds, item counts, items, then properties)
ParamAssess.RULES = (
    [(kw, getattr(ParamAssess, attr)) for kw, attr in (("enum", "_enum"), ("const", "_const"),
                                                     ("minLength", "_min_length"), ("maxLength", "_max_length"))]
    + [(kw, ParamAssess._boundary(kw, broken, phrase)) for kw, broken, phrase in (
        ("minimum", lambda got, edge: got < edge, "is below the minimum"),
        ("maximum", lambda got, edge: got > edge, "is above the maximum"),
        ("exclusiveMinimum", lambda got, edge: got <= edge, "has to exceed"),
        ("exclusiveMaximum", lambda got, edge: got >= edge, "has to stay under"))]
    + [(kw, getattr(ParamAssess, attr)) for kw, attr in (("minItems", "_min_items"), ("maxItems", "_max_items"),
                                                     ("items", "_items"))]
    + [("type", lambda got, sch, where: (got, []) if not isinstance(got, dict) else ParamAssess._qualities(got, sch, where))]
)


# ------------------------------------------------------------------------------------------------------------- wire
class TransmitDefect:
    """Stands in for the reply when the client itself raised: no data, and an error without a status (so it is retried
    like a dropped connection)."""

    def __init__(self, reason: str):
        self.data, self.error = None, {"status": None, "body": reason[:300]}


class Protocol:
    """Model and world requests through the sandbox proxy, both kept inside the errand's time budget."""

    def __init__(self, binding: dict[str, Any]):
        self.startmark = started = _TALLY.now()
        self.tether = dict(binding)
        self.llm_roster = ProxyClient(timeout=70, retry_delay=1, max_retries=1)
        # (TA3) the world client retries nothing itself: act() owns the retries
        self.cosmos = ProxyClient(timeout=95, retry_delay=1, max_retries=1)
        self.spend, self.pulse = 0.0, 0
        self.done = self.crashed = False
        self.journal = []
        # (TA) time the errand's latest HTTP request left
        self.final_transmit = 0.0
        # until this time requests stay SLOW_GAP apart (set by a 429 or 503)
        self.sluggish_till = 0.0
        # the next model request is held until this time (the back-off after a failed attempt)
        self.llm_restart = 0.0
        # time of the latest model answer, and the model attempts that failed after it
        self.final_response, self.misses = started, 0
        # no further model calls are paid for by the run key (402 or a key limit)
        self.without_llm = False
        # models the proxy rejected for this errand (401, 403, 404)
        self.forbidden = set()
        # models that no endpoint serves with tools attached (a 404 mentioning tools)
        self.instrument_forbidden = set()
        # allowed models named by a 403: the errand's fallback ladder
        self.surplus_llms = []
        # models the latest call attempted
        self.invoke_llms = set()
        # when set, the code finishes on its own and sends no model call
        self.silence = False
        # why the latest message-less call failed: "empty", "refused" or "" (see failed())
        self.invoke_breakdown = ""

    def surplus_alt(self) -> float:
        elapsed = _TALLY.now() - self.startmark
        return CUTOFF - elapsed

    def expended_depleted(self) -> bool:
        cap = OUTLAY_LID * self.__dict__.get("scale", 1.0)
        return self.spend >= cap

    def transmit(self, proxy: Any, route: str, payload: dict[str, Any]) -> Any:
        """Sends one paced HTTP request through the proxy (TA1): it goes at least PACE_GAP after the errand's previous
        request, or SLOW_GAP during the SLOW_FOR seconds after a load refusal (all episodes of a sandbox draw on one
        request bucket); model requests also wait for model_resume, the back-off a failed model attempt set."""
        spacing = TEMPO_INTERVAL if _TALLY.now() >= self.sluggish_till else SLUGGISH_INTERVAL
        hold = self.final_transmit + spacing - _TALLY.now()
        if route.find("/inference/") >= 0:
            hold = max(hold, self.llm_restart - _TALLY.now())
        hold = min(hold, self.surplus_alt() - 1)
        if hold > 0:
            _TALLY.halt_alt(hold)
        self.final_transmit = _TALLY.now()
        try:
            with _TALLY.underway():
                reply = proxy.post_verbose(route, payload)
        except Exception as exc:
            reply = TransmitDefect(f"{type(exc).__name__}: {exc}")
        load_refusal = reply.data is None and (reply.error or {}).get("status") in (429, 503)
        if load_refusal:
            self.sluggish_till = SLUGGISH_TOWARD + _TALLY.now()
        return reply

    def halt_alt(self, text: str, attempt: int, code: Any) -> float:
        """Back-off before a retry (TA2): 2 s after a 429 and 1 s after other failures, doubling per try up to 8 s, then
        spread by up to a quarter either way with a number derived from the request text and the step (never the clock),
        so errands refused together do not return together."""
        tail_total = sum(map(ord, text[-64:]))
        jitter = (len(text) * 7 + tail_total + 31 * attempt + 13 * self.pulse) % 101 / 100.0
        base = 2.0 if code == 429 else 1.0
        return min(8.0, base * 2 ** attempt) * (0.75 + 0.5 * jitter)

    def lowered_alt(self, when: float | None = None) -> bool:
        """True when no model can be called, or when the latest attempts failed and no model answered for DOWN_AFTER
        seconds as of `when`."""
        if self.without_llm:
            return True
        moment = _TALLY.now() if when is None else when
        return self.misses > 0 and moment - self.final_response > LOWERED_FOLLOWING

    def viable(self, models: list[str], with_tools: bool = False) -> bool:
        if self.without_llm or self.silence:
            return False
        candidates = list(models) + self.surplus_llms
        return any(m not in self.forbidden and not (with_tools and m in self.instrument_forbidden) for m in candidates)

    @staticmethod
    def simple(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The messages with every content flattened to one string and no cache marks: the simpler body tried once after
        a 400."""
        def flatten(original: dict[str, Any]) -> dict[str, Any]:
            msg = dict(original)
            parts = msg.get("content")
            if isinstance(parts, list):
                msg["content"] = "\n\n".join(str(p.get("text") or "") for p in parts if isinstance(p, dict))
            return msg
        return [flatten(m) for m in messages]

    def searchterm_llm(self, messages: list[dict[str, Any]], models: list[str], max_out: int,
                    tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        # (TA2) a load or transport failure is retried after a growing back-off, a 400 once more with plain contents; a
        # model the proxy rejects is dropped for the errand and a spent key ends model calls; the list is never hammered
        self.invoke_breakdown = ""
        self.invoke_llms = set()
        if self.without_llm or self.silence:
            return None

        def too_dear(rank: int, model: str) -> bool:
            # a sonnet-class model is passed over once spend is high, when a cheaper model follows it
            if "sonnet" not in model:
                return False
            cheaper_follows = rank + 1 < len(models) and "sonnet" not in models[rank + 1]
            # (projected cost: prompt characters at roughly 2.3 per token, plus the answer)
            projected = sum(len(json.dumps(m, default=str)) for m in messages) / 2.3 * 2e-6 + min(max_out, 1500) * 1e-5
            factor = self.__dict__.get("scale", 1.0)
            if self.expended_depleted():
                return True
            return bool(cheaper_follows and (self.spend >= GENTLE_OUTLAY * factor or self.spend + projected > OUTLAY_LID * factor))

        def single_left() -> bool:
            return len({m for m in ladder if m not in self.forbidden}) <= 1

        def hopeless(when: float | None = None) -> bool:
            # the models count as down and either no other model is left or two models already failed in this call
            return self.lowered_alt(when) and (lone or len(self.invoke_llms) >= 2)

        retry_seed = ""
        # how attempts failed: "empty" (answer without content), "hard" (refused), "soft" (load or network)
        outcomes: set[str] = set()
        # models refused outright during this call, not tried again in it
        dropped: set[str] = set()
        ladder = list(models) + [m for m in self.surplus_llms if m not in models]
        lone = single_left()
        for rank, model in enumerate(ladder):
            if not self.without_llm and self.spend >= FIRM_OUTLAY * self.__dict__.get("scale", 1.0):
                # (model spend reached its hard stop: no more model calls, the code finishes on its own)
                self.without_llm = True
                self.journal.append(dict(kind="spend_stop", spend=round(self.spend, 4)))
            if self.without_llm:
                return self.unsuccessful(outcomes)
            skipped = model in self.forbidden or model in dropped or (tools and model in self.instrument_forbidden)
            if skipped or too_dear(rank, model):
                continue
            simple_body = False
            for _ in range(LLM_ATTEMPTS if not self.lowered_alt() else 1):
                left = self.surplus_alt()
                if left < FINAL_INVOKE - 10:
                    return self.unsuccessful(outcomes)
                top = 75 if not self.lowered_alt() else 30
                self.llm_roster.timeout = int(max(8, min(top, left - FINAL_INVOKE + 15)))
                payload: dict[str, Any] = {"model": model}
                payload["messages"] = self.simple(messages) if simple_body else self.memoized(messages, model)
                payload["temperature"] = 0
                payload["max_tokens"] = max_out
                if tools:
                    payload.update(tools=tools, tool_choice="auto")
                self.invoke_llms.add(model)
                sent_at = _TALLY.now()
                reply = self.transmit(self.llm_roster, "/inference/chat/completions", payload)
                data = reply.data
                answered = isinstance(data, dict) and data.get("choices")
                if answered:
                    self.misses = 0
                    self.final_response = _TALLY.now()
                    cost = (data.get("usage") or {}).get("cost")
                    if whether_numeric(cost):
                        self.spend += float(cost)
                    message = data["choices"][0].get("message") or {}
                    if message.get("content") or message.get("tool_calls"):
                        return message
                    outcomes.add("empty")
                    self.journal.append(dict(kind="inference_failure", model=model, error="empty"))
                    break
                self.misses = self.misses + 1
                failure = str(reply.error)[:300] if reply.error else "empty"
                self.journal.append(dict(kind="inference_failure", model=model, error=failure))
                err = reply.error or {}
                status = err.get("status")
                raw = str(err.get("body") or "")
                said = raw.lower()
                key_spent = status == 402 or (status == 403 and re.search(r"key limit|insufficient|credit", said))
                if key_spent:
                    self.without_llm = True
                    return self.unsuccessful(outcomes)
                if status in (401, 403, 404):
                    outcomes.add("hard")
                    dropped.add(model)
                    if status == 404 and tools and "tool" in said:
                        # (no endpoint accepts tools; the model can still read)
                        self.instrument_forbidden.add(model)
                    elif status != 403 or not re.search(r"moderat|flagged", said):
                        self.forbidden.add(model)
                    allowed = None if status != 403 else re.search(r'allowed_models"?\s*:\s*\[([^\]]*)', raw)
                    if allowed and not self.surplus_llms:
                        # (the proxy lists the models this run may use: its first two become the errand's fallbacks)
                        named = re.findall(r'"([^"\s]{3,100})"', allowed.group(1))
                        self.surplus_llms = [m for m in named if m not in self.forbidden and m not in models][:2]
                        ladder.extend([m for m in self.surplus_llms if m not in ladder])
                        lone = single_left()
                    break
                if status in (400, 413, 422):
                    outcomes.add("hard")
                    if not simple_body:
                        # (the plain body goes out at once: a rejected body is not load)
                        simple_body = True
                        continue
                    dropped.add(model)
                    break
                client_error = isinstance(status, int) and 400 <= status < 500 and status not in (408, 425, 429)
                if client_error:
                    outcomes.add("hard")
                    dropped.add(model)
                    break
                outcomes.add("soft")
                if _TALLY.now() - sent_at > 20:
                    # (a hung or slow attempt moves on to the next model rather than waiting again; once the models
                    # count as down the call stops here only when two models failed this way, or there is no other)
                    if hopeless():
                        return self.unsuccessful(outcomes)
                    break
                retry_seed = retry_seed or json.dumps(messages[-1:], default=str)[-600:]
                # (the back-off grows with attempts failed since the last model answer, across calls: 2, 4, 8, 8 ... s;
                # send() makes the next model request wait it out, so a world call after a failure is not delayed)
                wait = self.halt_alt(retry_seed + model, min(self.misses - 1, 6), status)
                if FINAL_INVOKE - 10 > self.surplus_alt() - wait:
                    return self.unsuccessful(outcomes)
                self.llm_restart = wait + _TALLY.now()
                if hopeless(self.llm_restart):
                    # (silent for DOWN_AFTER: one attempt per model, two models per call)
                    return self.unsuccessful(outcomes)
        return self.unsuccessful(outcomes)

    def unsuccessful(self, outcomes: set[str]) -> None:
        """Closes a call that produced no message, recording why (TA4 reads it): "empty" if a model replied without
        content, "refused" if every attempt was rejected outright (4xx), "" if load or the network lost it."""
        if "empty" in outcomes:
            self.invoke_breakdown = "empty"
        else:
            self.invoke_breakdown = "refused" if outcomes == {"hard"} else ""

    @staticmethod
    def memoized(messages: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
        """Anthropic models get the system text and the latest message flagged for the prompt cache."""
        if not model.startswith("anthropic/") or len(messages) < 3:
            return messages
        copies = list(map(dict, messages))
        for at in (0, len(copies) - 1):
            text = copies[at].get("content")
            if isinstance(text, str) and text:
                copies[at]["content"] = [dict(type="text", text=text, cache_control=dict(type="ephemeral"))]
        return copies

    def confer(self, prompt: str, max_out: int, models: list[str]) -> dict[str, Any] | None:
        conversation = [dict(role="user", content=prompt)]
        reply = self.searchterm_llm(conversation, models, max_out)
        return serialized_entity((reply or {}).get("content") or "")

    def perform(self, actions: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Sends one or more actions as consecutive steps, resyncing the step number when the world expects another one.
        After a load refusal, a dropped connection or a server error the same envelope is resent after a growing back-off
        (TA3); the world replays a step it already ran from its record, so a resend never runs anything twice."""
        envelope: dict[str, Any] = dict(self.tether)
        envelope["step"] = self.pulse + 1
        envelope.update({"action": actions[0]} if len(actions) == 1 else {"actions": actions})
        retries = repeats = 0
        for _ in range(COSMOS_ATTEMPTS + 3):
            left = self.surplus_alt()
            if left < 2:
                break
            self.cosmos.timeout = int(max(3, min(95, left - 1)))
            reply = self.transmit(self.cosmos, "/environment/call", envelope)
            data = reply.data
            if isinstance(data, dict):
                step = data.get("step")
                self.pulse = step if isinstance(step, int) else self.pulse + len(actions)
                self.done = self.done or bool(data.get("finished"))
                return data
            err = reply.error or {}
            code = err.get("status")
            text = str(err.get("body") or "")
            self.journal.append(dict(kind="env_failure", status=code, body=text[:300]))
            if code == 402:
                self.crashed = True
                break
            if code == 409:
                if "finished" in text:
                    self.done = True
                    break
                named = re.search(r"expected (\d+)", text)
                if named:
                    self.pulse = int(named.group(1)) - 1
                    envelope["step"] = self.pulse + 1
                elif "reused" in text and repeats < 2:
                    # (TA3: one of this errand's steps ran but the reply was lost: the next number is free, or the
                    # world will name the one it wants)
                    repeats += 1
                    envelope["step"] += 1
                else:
                    break
                continue
            stop = code in (403, 422) or (code in (400, 401, 404) and retries >= 1)
            if stop or retries >= COSMOS_ATTEMPTS - 1:
                break
            wait = self.halt_alt(json.dumps(envelope, default=str)[-600:], retries, code)
            retries += 1
            if self.surplus_alt() - wait < 2:
                break
            _TALLY.halt_alt(wait)
        return None


# ---------------------------------------------------------------------------------------------------------- records
class Chronicles:
    """All observations the store sent back, plus the views of them that the checks and the question plan use."""

    def __init__(self):
        self.items = []
        self.ids = set()
        # ids whose details (a record holding a nested property map) have been seen
        self.read = set()
        # (CC) listing id -> category label carried by some record of that id
        self.label_of = {}

    def add(self, observation: Any) -> None:
        self.items.append(observation)

        def names_id(k: Any, v: Any) -> bool:
            if not isinstance(v, str):
                return False
            low = str(k).lower()
            return low in ("id", "_id") or (low.endswith("id") and bool(re.fullmatch(UID_RESEMBLING, v)))

        def has_props(v: Any) -> bool:
            return isinstance(v, dict) and sum(1 for x in v.values() if isinstance(x, (str, int, float))) >= 2

        def category_label(k: Any, v: Any) -> str:
            low = str(k).lower()
            fits = isinstance(v, str) and 0 < len(v.strip()) <= 60
            return v.strip() if fits and re.search(r"categor|shelf|department|aisle", low) and "id" not in low else ""

        def note(rec: dict[str, Any]) -> None:
            self.ids.update(v for k, v in rec.items()
                            if isinstance(v, str) and "id" in str(k).lower() and re.fullmatch(UID_RESEMBLING, v))
            ident = next((v for k, v in rec.items() if names_id(k, v)), None)
            if not ident:
                return
            # (any record holding a property map of 2+ fields counts as read, whatever tool returned it)
            if any(has_props(v) for v in rec.values()):
                self.read.add(ident)
            label = next((lab for lab in (category_label(k, v) for k, v in rec.items()) if lab), "")
            if label:
                # (CC: remember the listing's category label for its records lacking one - a spec sheet may carry just
                # the id, title and property map while the search row and the offer carry the label)
                self.__dict__.setdefault("label_of", {}).setdefault(ident, label)
        traverse_mappings(observation, note)

    def price(self, item_id: str) -> float | None:
        """The latest price seen on a record that carries the item's id."""
        seen: list[float] = []

        def note(rec: dict[str, Any]) -> None:
            if any(isinstance(v, str) and v == item_id and "id" in str(k).lower() for k, v in rec.items()):
                amount = next((float(v) for k, v in rec.items()
                               if whether_numeric(v) and not isinstance(v, bool) and re.search(r"price|cost", str(k).lower())),
                              None)
                if amount is not None:
                    seen.append(amount)
        for stored in self.items:
            traverse_mappings(stored, note)
        return seen[-1] if seen else None

    def concerning(self, item_id: str, keep: int = 4) -> list[str]:
        """The latest records (dicts) holding the item's id as a value, the outermost match only."""
        texts: list[str] = []

        def seek(node: Any, depth: int) -> None:
            if depth >= 7 or len(texts) > 7:
                return
            if isinstance(node, dict):
                if any(isinstance(v, str) and v == item_id for v in node.values()):
                    shown = into_passage(truncated(node, 400))[:2500]
                    if shown not in texts:
                        texts.append(shown)
                    return
                children = list(node.values())
            elif isinstance(node, list):
                children = node
            else:
                return
            for child in children:
                seek(child, depth + 1)

        for stored in self.items:
            seek(stored, 0)
        return texts[-keep:]

    def condense(self, budget: int = 24000, field_chars: int = 160) -> list[str]:
        """The records, latest first, kept within a character budget."""
        kept: list[str] = []
        employed = 0
        for rec in self.items[::-1]:
            shown = into_passage(truncated(rec, field_chars))
            if employed + len(shown) <= budget:
                kept.append(shown)
                employed += len(shown)
        return kept

    def profiles(self, budget: int = 30000, under_id: str = "", leave_out: set[str] | None = None,
              upfront: Any = ()) -> list[str]:
        """A line per listing seen, its search row merged with its details (details win), DETAILS listings ahead, every
        value clipped, all within a character budget. Given under_id, only that listing and those priced under it (or
        without a known price) remain. Listings in `upfront` go ahead of every other (EA1: they keep their card when
        the budget runs short)."""
        merged: dict[str, dict[str, Any]] = {}
        seen_order: list[str] = []

        def gather(rec: dict[str, Any]) -> None:
            ident = next((v for k, v in rec.items() if isinstance(v, str) and "id" in str(k).lower()
                          and re.fullmatch(UID_RESEMBLING, v) and v in self.ids), None)
            if not ident:
                return
            has_map = any(isinstance(v, dict) and len(v) >= 2 for v in rec.values())
            # (EA1: a record with 3 keys or fewer still counts when it holds a property map of 2+ fields - a spec sheet
            # may be just the id, the title and the map - the same rule add() applies to mark a read)
            if len(rec) <= 3 and not has_map:
                return
            if ident not in seen_order:
                seen_order.append(ident)
            merged.setdefault(ident, {}).update(rec)
            if has_map:
                self.read.add(ident)

        for stored in self.items:
            traverse_mappings(stored, gather)

        def price_of(ident: str) -> float | None:
            return next((float(v) for k, v in (merged.get(ident) or {}).items()
                         if whether_numeric(v) and not isinstance(v, bool) and re.search(r"price|cost", str(k).lower())), None)

        ceiling_price = price_of(under_id) if under_id else None
        if leave_out:
            seen_order = [i for i in seen_order if i == under_id or i not in leave_out]
        if ceiling_price is not None:
            seen_order = [i for i in seen_order if i == under_id or (p := price_of(i)) is None or p < ceiling_price]
        front = set(upfront or ())
        position = {ident: n for n, ident in enumerate(seen_order)}
        ranked = sorted(seen_order, key=lambda i: (i not in front, i not in self.read, position[i]))
        texts = {i: ("DETAILS " if i in self.read else "ROW ") + into_passage(truncated(merged[i], 160)) for i in ranked}
        if sum(map(len, texts.values())) > budget:
            # (v605e: when the cards cannot all fit, a read listing whose record says sold out goes behind the other
            # read ones)
            def gone(i: str) -> bool:
                return any(Errand.slot_inventory(k, v) is False for k, v in merged[i].items())
            ranked = sorted(ranked, key=lambda i: (i not in front, i not in self.read, gone(i), position[i]))
        lines: list[str] = []
        total = 0
        for ident in ranked:
            shown = texts[ident]
            if total + len(shown) <= budget:
                lines.append(shown)
                total += len(shown)
        return lines

    def tag_toward(self, rec: dict[str, Any]) -> str:
        """CC: the category label a record shows: a short category/shelf/aisle-like field of its own when it has one,
        otherwise the label a sibling record under the same listing id carried; '' if neither exists."""
        for field, entry in rec.items():
            if isinstance(entry, str):
                trimmed, lowered = entry.strip(), str(field).lower()
                if trimmed and len(trimmed) <= 60 and "id" not in lowered and re.search(r"categor|shelf|department|aisle", lowered):
                    return trimmed
        by_listing = vars(self).get("label_of") or {}
        borrowed = (by_listing[entry] for field, entry in rec.items()
                    if isinstance(entry, str) and entry in by_listing and "id" in str(field).lower())
        return next(borrowed, "")

    def detailed_peruses(self, shelf_name: str = "") -> int:
        """How many read listings have a record with a property map of at least 3 fields (for flat records: at least 5
        short text fields). A non-empty `shelf_name` keeps only records in that category."""
        target = shelf_name.strip().lower()
        hits: set[str] = set()

        def inspect(rec: dict[str, Any]) -> None:
            listing = next((v for k, v in rec.items() if isinstance(v, str) and "id" in str(k).lower() and v in self.read), None)
            # (CC: when the record names no category itself, the label of its listing stands in)
            if listing and not (shelf_name and self.tag_toward(rec).strip().lower() != target):
                nested = [v for v in rec.values() if isinstance(v, dict)]
                if nested:
                    dense = any(len(sub) >= 3 for sub in nested)
                else:
                    dense = len([v for v in rec.values() if isinstance(v, str) and 0 < len(v) <= 70]) >= 5
                if dense:
                    hits.add(listing)

        for record in self.items:
            traverse_mappings(record, inspect)
        return len(hits)

    def racks_peruse(self) -> list[str]:
        """Category labels of the read records, ordered from the most frequent down."""
        freq: dict[str, int] = {}

        def inspect(rec: dict[str, Any]) -> None:
            if next((v for k, v in rec.items() if isinstance(v, str) and "id" in str(k).lower() and v in self.read), None):
                # (CC: a spec sheet with no label borrows the label of its listing)
                label = self.tag_toward(rec)
                if label and any(isinstance(v, dict) for v in rec.values()):
                    freq[label] = 1 + freq.get(label, 0)

        for record in self.items:
            traverse_mappings(record, inspect)
        return sorted(freq, key=freq.__getitem__, reverse=True)

    def trait_picklist(self, shelf_name: str = "") -> tuple[str, int]:
        """Property names found on detail-like records (5+ short text fields) with how often they occur and a few sample
        values; a non-empty `shelf_name` limits this to that category. Non-property fields (brand, shop, category, title,
        text, price, stock, ids) are skipped, since asking about them uncovers nothing."""
        occur: dict[str, int] = {}
        samples: dict[str, list[str]] = {}
        value_freq: dict[str, dict[str, int]] = {}
        row_occur: dict[str, int] = {}
        record_n = full_n = 0
        shelf_low, shelf_norm = shelf_name.lower(), shelf_name.strip().lower()

        def in_category(rec: dict[str, Any]) -> bool:
            named_here = any(isinstance(entry, str) and entry.strip().lower() == shelf_low
                             and re.search(r"categor|shelf|department|aisle", str(field).lower()) for field, entry in rec.items())
            return named_here or self.tag_toward(rec).strip().lower() == shelf_norm      # (CC)

        def soakup(rec: dict[str, Any]) -> None:
            nonlocal record_n, full_n
            short = [(field, entry) for field, entry in rec.items()
                     if isinstance(entry, str) and 0 < len(entry) <= 70 and not rubbish_slot(field)]
            thin = len(short) < 5
            if thin and not (shelf_name and len(short) >= 2 and not any(isinstance(v, dict) for v in rec.values())):
                return
            record_n += 1
            # (CC: a thin record that holds a listing id is a listing row; without one it is a property map or a full record)
            listing_row = thin and any(isinstance(entry, str) and "id" in str(field).lower() for field, entry in rec.items())
            full_n += 0 if listing_row else 1
            for field, entry in short:
                if ("-" in entry or "_" in entry) and re.search(r"\d", entry) and re.fullmatch(ENGINE_RESEMBLING, entry):
                    continue
                label, low = re.sub(r"[_.]+", " ", str(field)).strip().lower(), entry.lower()
                occur[label] = 1 + occur.get(label, 0)
                if listing_row:
                    row_occur[label] = 1 + row_occur.get(label, 0)
                per_value = value_freq.setdefault(label, {})
                per_value[low] = 1 + per_value.get(low, 0)
                kept = samples.setdefault(label, [])
                if len(kept) < 5 and low not in kept:
                    kept.append(low)

        def descend(node: Any, within: bool, depth: int) -> None:
            if depth <= 6 and isinstance(node, list):
                for sub in node:
                    descend(sub, within, depth + 1)
            elif depth <= 6 and isinstance(node, dict):
                within = within or not shelf_name or in_category(node)
                if within:
                    soakup(node)
                for sub in node.values():
                    if isinstance(sub, (dict, list)):
                        descend(sub, within, depth + 1)

        for record in self.items:
            descend(record, False, 0)
        if full_n:
            # (CC: when the category's records carry their own properties, a field seen only on listing rows - the shop,
            # a blurb - belongs to the listing and is no property)
            occur = {label: n for label, n in occur.items() if row_occur.get(label, 0) < n}

        def variety(label: str) -> int:
            return len(samples.get(label, []))

        def varied(label: str) -> bool:
            return variety(label) >= 2 and occur[label] >= 2

        def with_unit(label: str) -> bool:
            seen_vals = samples.get(label, [])
            all_digits = bool(seen_vals) and all(re.search(r"\d", v) for v in seen_vals)
            return all_digits and any(re.search(r"\d\s*[a-z\"']", v) for v in seen_vals)

        # (fields whose values differ between the listings read lead: a hidden need is what separates listings)
        def weight(label: str) -> float:
            per_value = value_freq.get(label) or {}
            total = sum(per_value.values())
            share = max(per_value.values()) / total if total else 1.0
            return min(1.0, occur[label] / max(1, record_n)) * (1.0 - share)

        if record_n < 3:
            order = sorted(occur, key=lambda label: (-int(varied(label)), -occur[label], -variety(label)))
        else:
            order = sorted(occur, key=lambda label: (-weight(label), -occur[label], label))
        picklist_tablerows = []
        for label in order[:28]:
            tail = f" [values differ among the listings: {variety(label)} seen]" if varied(label) else ""
            if with_unit(label):
                tail += " [figures with a unit]"
            picklist_tablerows.append(f"- {label} (in {occur[label]} listings){tail}: " + " | ".join(samples.get(label, [])))
        return "\n".join(picklist_tablerows), record_n

    def tag_lineitems(self) -> str:
        """Short values repeated under the same key on several listings (category-like labels), each with its count."""
        by_key: dict[str, dict[str, int]] = {}

        def inspect(rec: dict[str, Any]) -> None:
            if len(rec) >= 4:
                for field, entry in rec.items():
                    if isinstance(entry, str) and 0 < len(entry) <= 40\
                            and not (re.search(r"\d", entry) and re.fullmatch(ENGINE_RESEMBLING, entry)):
                        tally = by_key.setdefault(str(field), {})
                        tally[entry] = 1 + tally.get(entry, 0)

        for record in self.items:
            traverse_mappings(record, inspect)
        out = []
        for field, tally in by_key.items():
            if sum(tally.values()) >= 4 and 2 <= len(tally) <= 8 and max(tally.values()) >= 2:
                ranked = sorted(tally.items(), key=lambda pair: pair[1], reverse=True)
                out.append(f"- {field}: " + " | ".join(f"{entry} (x{n})" for entry, n in ranked))
        return "\n".join(out[:12])


# ================================================================================================= the shopper's needs
class Requirements:
    """The shopper's current wishes: how the request was read, what was revealed, dropped or left open, and the budget."""

    def __init__(self, link: Protocol, text: str):
        self.protocol, self.request = link, text
        self.perusal = {}
        self.written, self.uncovered, self.let_go, self.indifferent = [], [], [], []
        self.now_reqs = None
        self.outlay_lid = None
        self.iteration = self.informed_iteration = 0       # told_version grows with each instruction added (the finish check re-runs)
        self.intercepted, self.dialogues = set(), []
        self.response_owed = ""
        self.broadened_when, self.broadened_pulse = -1, -10 ** 6
        self.surveyable = False     # True when the search tool pages, so the code itself can walk a shelf slice by slice
        self.resolved = []         # BT6: requirements whose value the code took from a listing the request points at
        self.ref_news = []         # BT1: needs stated by pointing at a listing, waiting for the resolver
        self.informed = []             # binding instructions: the request's first, then the shopper's and the store's
        self.allowance_news = None  # a message or step limit some line stated, for the errand to apply

    def attach_informed(self, entries: Any, origin: str) -> list[str]:
        """Records stated instructions (dicts with text / before_finish / needs_message, or bare strings whose flags stay
        unknown); identical wording is kept once. Gives back the texts that were new."""
        fresh: list[str] = []
        seen = {" ".join(str(entry.get("text") or "").lower().split()) for entry in self.informed}
        for entry in entries[:12] if isinstance(entries, list) else []:
            if isinstance(entry, str):
                entry = {"text": entry, "before_finish": None, "needs_message": None}
            if isinstance(entry, dict):
                text = " ".join(str(entry.get("text") or "").split())
                if text and text.lower() not in seen:
                    seen.add(text.lower())
                    marks = {flag: entry.get(flag) if isinstance(entry.get(flag), bool) else None
                             for flag in ("before_finish", "needs_message", "before_change")}
                    self.informed.append({"text": text[:400], "source": origin, **marks})
                    fresh.append(text[:120])
        if fresh:
            self.informed_iteration = self.informed_iteration + 1
        return fresh

    def conclude_informed(self) -> list[dict[str, Any]]:
        """Instructions that can still bind the finish: those marked before_finish and those whose flags are unknown."""
        return list(filter(lambda entry: entry.get("before_finish") is not False, self.informed))

    def informed_passage(self) -> str:
        parts = []
        for pos, entry in enumerate(self.informed[-10:], start=1):
            flags = [label for label, key in (("due before finishing", "before_finish"), ("to be done with a message", "needs_message"))
                     if entry.get(key)]
            origin = f"given by the {entry.get('source')}"
            parts.append(f"{pos}) {entry['text']} [" + (", ".join(flags) + "; " if flags else "") + origin + "]")
        return "\n[instructions in force] " + " ".join(parts) if parts else ""

    def items(self) -> list[dict[str, Any]]:
        """The separate products the request wants when it names two or more distinct ones; [] otherwise."""
        def whole(n: Any) -> bool:
            return isinstance(n, int) and not isinstance(n, bool) and n >= 1

        out = []
        for row in filter(lambda r: isinstance(r, dict), self.perusal.get("items") or []):
            name = str(row.get("product") or "").strip()
            if name:
                count, dep, raw_reqs = row.get("quantity"), row.get("depends_on"), row.get("requirements")
                reqs = raw_reqs if isinstance(raw_reqs, list) else []
                out.append({"product": name, "quantity": count if whole(count) else 1,
                            "requirements": [str(r) for r in reqs if r and ":" in str(r)],
                            "depends_on": dep - 1 if whole(dep) else None})
        return [] if len(out) < 2 else out

    def quantity(self) -> int:
        # (with several products every line carries its own count; the filters work per unit)
        wanted = None if self.items() else self.perusal.get("quantity")
        return wanted if isinstance(wanted, int) and not isinstance(wanted, bool) and wanted >= 1 else 1

    def headcounts(self) -> list[int]:
        """How many units each cart line has to carry, one entry per product."""
        per_product = [row["quantity"] for row in self.items()]
        return per_product if per_product else [self.quantity()]

    def entries_passage(self) -> str:
        products = self.items()
        listed = " | ".join(f"{pos}) {row['product']} x{row['quantity']}" for pos, row in enumerate(products, start=1))
        return ("THE REQUEST ASKS FOR SEVERAL PRODUCTS; the order must carry one line per product at that product's quantity and "
                "no other line; a listing is measured only against the requirements of the product it belongs to (its category "
                "tells which), and any cheapest or best rule works inside each product separately: " + listed) if products else ""

    def priority(self) -> str:
        rule = self.perusal.get("priority")
        return rule if rule in ("cheapest", "best") else "none"

    def traits_remaining(self) -> list[str]:
        """Property requirements still in force (price, budget, stock and quantity requirements excluded)."""
        pool = self.written + self.uncovered if self.now_reqs is None else self.now_reqs
        return [req for req in pool if req not in self.let_go
                and not re.search(r"\bprice\b|budget|quantity|\bstock\b|\bcents?\b", req.lower())]

    def passage_payload(self) -> str:
        # (the wording below is also read back by hear(): its words ground the listener's requirements, so it stays as is)
        if self.now_reqs is None:
            parts = [f"{req} (stated)" for req in self.written if req not in self.let_go]\
                + [f"{req} (revealed by the shopper)" for req in self.uncovered if req not in self.let_go]
        else:
            parts = list(self.now_reqs)
        if parts:
            body = "; ".join(parts)
        elif self.let_go:
            body = "no property requirement is left: every in-stock listing in the requested category fits"
        else:
            body = "see the request"
        if self.outlay_lid is not None:
            body += f"; total price at most {self.outlay_lid} {self.perusal.get('currency') or ''}".rstrip()
        body += "; " + self.entries_passage() if self.items() else f"; quantity {self.quantity()}"
        if self.let_go:
            body += "; no longer required: " + "; ".join(self.let_go)
        return body + ("; shopper does not care about: " + ", ".join(self.indifferent) if self.indifferent else "")

    def alterations(self) -> str:
        """For the checks that read the original request: the requirements the shopper has since given up."""
        if self.let_go:
            return ("\n\nCHANGES SINCE THE REQUEST (these win over its text): the shopper GAVE UP " + "; ".join(self.let_go)
                    + ". A requirement given up counts for nothing now: do not reject, rank or favour any listing by it.")
        return ""

    def synopsis(self, step_total: int, msg_total: int | None) -> str:
        first = self.perusal
        out = ["How the request was first read (compare it with the request's own words):"]
        if first.get("product"):
            out.append(f"- product: {first.get('product')}")
        products = self.items()
        if products:
            out.extend(f"- product {pos}: {row['product']}, quantity {row['quantity']} (its own cart line)"
                       for pos, row in enumerate(products, start=1))
            out.append("- the order carries exactly these lines, one for each product, and no other; when the cart tool "
                       "accepts a list of lines, every cart call sends all of them")
        else:
            out.append(f"- quantity: {self.quantity()}")
        out.extend(f"- stated requirement: {req}" for req in first.get("requirements") or [])
        out.extend(f"- value set by pointing at another listing (open that listing first, then search with its value): {req}"
                   for req in first.get("references") or [])
        if first.get("price_cap") is not None:
            out.append(f"- price cap (total): {first.get('price_cap')} {first.get('currency') or ''}".rstrip())
        if first.get("in_stock_only"):
            out.append("- only items that are in stock / available")
        rule = self.priority()
        if rule == "none":
            out.append("- no cheapest/best rule was found: any item that meets every requirement inside the budget is a right "
                       "answer, so spend no steps chasing the lowest price (unless the request itself asks for that)")
        else:
            extra = f" ({first.get('priority_detail')})" if first.get("priority_detail") else ""
            out.append(f"- pick the {rule}{extra} of all the fitting items")
        out.extend(f"- welcome but NOT required (use only to break a tie): {want}" for want in first.get("soft_wants") or [])
        out.extend(f"- to learn from the shopper before choosing: {topic}" for topic in first.get("ask_first") or [])
        out.extend(f"- rule: {rule_text}" for rule_text in first.get("rules") or [])
        if msg_total is not None:
            out.append(f"- messages the shopper may receive: {msg_total}")
        out.append(f"- step allowance: {step_total}")
        return "\n".join(out) if first else ""

    @staticmethod
    def lineitem_monikers(found: list[Any], said_lines: list[str]) -> dict[str, str]:
        """Maps a need to the same need under the property name used in the shopper's line, when the listener's name is no
        phrase of that line but every word of it is the start of the matching word in one stretch of the line (a short
        name standing for a longer one)."""
        tokens = re.findall(r"[a-z]+", " ".join(said_lines).lower())
        spoken = " ".join(tokens)
        mapped: dict[str, str] = {}
        for item in found:
            if isinstance(item, str) and ":" in item:
                head, _, tail = item.partition(":")
                words = re.findall(r"[a-z]+", re.split(r"\s*\(", head)[0].lower())
                if words and min(len(w) for w in words) >= 3 and not re.search(r"\b" + r"\s+".join(words) + r"\b", spoken):
                    width = len(words)
                    stretch = next((tokens[at: at + width] for at in range(len(tokens) - width + 1)
                                    if tokens[at: at + width] != words
                                    and all(full.startswith(short) for short, full in zip(words, tokens[at: at + width]))), None)
                    if stretch is not None:
                        mark = head[head.find("("):].strip() if "(" in head else ""
                        mapped[item] = " ".join(stretch) + (" " + mark if mark else "") + ":" + tail
        return mapped

    def discard_titled(self, gone_need: str, said_lines: list[str]) -> bool:
        """Whether a requirement the listener marked as dropped may really go. It may, unless it is a stated or revealed one
        and no phrase of 1-3 words in the shopper's lines names its property (same_prop) - so a line about a similarly
        named other property cannot drop it. CJ1: a phrase naming a field of the product's category (strict_prop) whose
        values hold the requirement's value also names it ('drop dietary needs' for 'sugar content: low sugar' when the
        store files low sugar under dietary needs); and a revealed need goes when the reply gives that property a new
        value (it is replaced)."""
        try:
            prop = str(gone_need).split(":", 1)[0]
            ptag = re.findall(r"\(([^)]*)\)", prop)

            def identical_category(req: str) -> bool:
                req_head = req.split(":", 1)[0]
                return identical_trait(req_head, prop) and (not ptag or re.findall(r"\(([^)]*)\)", req_head) in ([], ptag))

            stated = [req for req in self.written if identical_category(req)]
            if not stated:
                # (the reply's own new needs already sit in the ledger when its drops are read: they are not the old value)
                fresh = [n for n in vars(self).get("fresh_needs") or [] if isinstance(n, str) and ":" in n]
                stated = [req for req in self.uncovered if identical_category(req) and req not in self.written and req not in fresh]
                # (no such need at all, or the revealed value was swapped for a new one of the same property)
                if not stated or any(identical_category(n) and n != gone_need for n in fresh):
                    return True
            tokens = re.findall(r"[a-z]+", " ".join(said_lines).lower())
            grams = {" ".join(tokens[at: at + size]) for size in (1, 2, 3) for at in range(len(tokens) - size + 1)}
            grams = {g for g in grams if len(g) >= 3 and g not in BOND_TERMS}
            if any(identical_trait(req.split(":", 1)[0], g) for req in stated for g in grams):
                return True
            hook = vars(self).get("field_hook")
            for req in stated if hook else []:
                req_head, _, raw_value = req.partition(":")
                simple = re.sub(r"\((?:stated|revealed[^)]*)\)", " ", raw_value)
                # (a value taken from another listing has no field values to compare)
                if not reference_passage(simple) and crosscheck_variants(simple):
                    fields = hook(req_head) or {}
                    rivals = [r.split(":", 1)[0] for r in self.written + self.uncovered
                              if isinstance(r, str) and ":" in r and not identical_trait(r.split(":", 1)[0], prop)]
                    for g in grams:
                        # (a phrase that names another requirement still in force speaks of that one)
                        if not any(rigorous_trait(other, g) for other in rivals)\
                                and any(rigorous_trait(fld, g) and setting_observed(simple, " | ".join(vals)) for fld, vals in fields.items()):
                            self.protocol.journal.append(dict(kind="drop_by_field", need=req[:200], phrase=g))
                            return True
            return False
        except Exception:
            return True

    def verbatim_settings(self, reply: dict[str, Any]) -> None:
        """CJ2: when a new need's value, stripped of its comparison words, is a value the store itself writes in that
        property's field ('falls short of not available' where listings say 'not available'), the need becomes
        "property: <that value>" in the needs and in requirements_now, and positive_value() treats it as a value from then on."""
        hook = vars(self).get("field_hook")

        def alnum(text: str) -> str:
            return " ".join(re.findall(r"[a-z0-9]+", text))

        swaps: dict[str, str] = {}
        for item in (reply.get("needs") or []) if hook else []:
            if not (isinstance(item, str) and ":" in item):
                continue
            head, _, tail = item.partition(":")
            if re.search(r"\d", tail) or reference_passage(tail):
                continue
            low = tail.lower()
            core = alnum(re.sub(CROSSJUDGE_CONTRAST, " ", low))
            if not core or (core == alnum(low) and affirmative_setting(tail, head)):
                continue
            prop = re.split(r"\s*\(", head)[0].strip()
            for field, vals in (hook(head) or {}).items():
                match = next((v for v in vals if alnum(str(v).lower()) == core), None) if rigorous_trait(field, prop) else None
                if match is not None:
                    VERBATIM_SETTINGS.add(core)
                    swaps[item] = f"{head.strip()}: {str(match).strip().lower()}"
                    break
        if swaps:
            def swapped(seq: list[Any]) -> list[Any]:
                return [swaps.get(x, x) if isinstance(x, str) else x for x in seq]

            reply["needs"] = swapped(reply.get("needs") or [])
            if isinstance(reply.get("requirements_now"), list):
                reply["requirements_now"] = swapped(reply["requirements_now"])
            self.protocol.journal.append(dict(kind="literal_value", needs=list(swaps.values())[:4]))

    def anchored(self, asked_prop: str, asked_text: str | None) -> bool:
        """Whether a 'don't mind' must leave a property alone: some stated requirement in force carries a positive value
        written in the request itself, and it is that property by name, or the asked name ends with its name ('general
        style' for 'style')."""
        try:
            given = self.perusal.get("references")
            refs = [str(r) for r in given or []] if isinstance(given, list) else []
            refs += list(vars(self).get("resolved") or [])
            asked_tag = re.findall(r"\(([^)]*)\)", str(asked_prop))
            asked_key = trait_slot(asked_prop)
            for req in self.written:
                if req in self.let_go or ":" not in req:
                    continue
                req_head, _, req_value = req.partition(":")
                req_tag = re.findall(r"\(([^)]*)\)", req_head)
                if asked_tag and req_tag and not category_terms(asked_tag[-1]) & category_terms(req_tag[-1]):
                    continue
                if affirmative_setting(req_value, req_head) and not narrative_estimate(req, self.request, refs):
                    req_key = trait_slot(req_head)
                    if identical_trait(req_head, asked_prop) or (len(req_key) >= 4 and asked_key.endswith(req_key)):
                        return True
        except Exception:
            pass
        return False

    @staticmethod
    def valued(found: Any, said_lines: list[str]) -> bool:
        """CD: true when the lines themselves write one of the needs - every word of its property and of its value shows
        up in them (a reveal names both; a figurative word in a one-question-at-a-time reply names no property)."""
        spoken = " ".join(str(ln) for ln in said_lines).lower()
        for item in found if isinstance(found, list) else []:
            head, _, tail = str(item).partition(":")
            prop_bits = [w for w in trait_terms(head) if len(w) >= 3]
            value_bits = [w for w in re.findall(r"[a-z0-9]+", tail.lower()) if w not in BOND_TERMS]
            if prop_bits and value_bits and all(w in spoken for w in prop_bits + value_bits):
                return True
        return False

    def receive(self, said_lines: list[str], asked: str | None) -> str:
        """Takes shopper lines not seen before into the ledger and returns a note for the actor ('' when nothing was taken in)."""
        unseen = [ln for ln in said_lines if ln not in self.intercepted]
        if not unseen:
            return ""
        self.intercepted.update(unseen)
        if WRAPUP > self.protocol.surplus_alt():
            return ""
        so_far = self.passage_payload()
        sections = [RECEIVE, "THE SHOPPER'S REQUEST (for context):\n" + self.request, "REQUIREMENTS SO FAR:\n" + so_far]
        if self.dialogues:
            sections.append("EARLIER EXCHANGES WITH THE SHOPPER (oldest first):\n" + "\n".join(self.dialogues[-12:]))
        if asked:
            sections.append("THE ASSISTANT ASKED:\n" + asked)
        sections.append("THE SHOPPER NOW SAID:\n" + "\n".join(unseen))
        self.dialogues.append((f"asked: {asked} -> " if asked else "") + "shopper: " + " / ".join(unseen))
        reply = self.protocol.confer("\n\n".join(sections), 2500, WATCHER)
        if not reply:
            if self.protocol.misses > 0 or self.protocol.silence:
                # (TA4: load or the network ate the listener's call, or the code is finishing alone: the lines go back to
                # unread, to be read when a model answers again - Errand.catch_up - or by the code's own finish)
                self.intercepted.difference_update(unseen)
                del self.dialogues[-1]
                self.pending = list(vars(self).get("pending") or []) + [(list(unseen), asked)]
                self.pending_at = _TALLY.now()
            return ""
        try:
            if reply.get("meta_only") is True and asked and not self.valued(reply.get("needs"), unseen):
                # (CD: the line just asks for one question at a time and answers nothing - no need, nothing freed or
                # dropped; the asked property stays open and is sent again by itself)
                reply = dict(reply, needs=[], no_preference=[], dropped=[], references=[],
                             requirements_now=None, new_budget=None, unavailable=False, reply="")
                self.meta_last = asked
                self.protocol.journal.append(dict(kind="meta_only", line=" / ".join(unseen)[:200]))
        except Exception:
            pass
        try:
            renames = self.lineitem_monikers(reply.get("needs") or [], unseen)
        except Exception:
            renames = None
        if renames:
            # (a revealed need keeps the shopper's own property name: the listener had shortened "packaging type" to the
            # shorter stated name, which then overwrote the stated requirement)
            def renamed(seq: list[Any]) -> list[Any]:
                return [renames.get(x, x) if isinstance(x, str) else x for x in seq]

            reply["needs"] = renamed(reply.get("needs") or [])
            if isinstance(reply.get("requirements_now"), list):
                reply["requirements_now"] = renamed(reply["requirements_now"])
            self.protocol.journal.append(dict(kind="need_renamed", names=list(renames.values())[:4]))
        masked = unseen
        ref_hook = vars(self).get("ref_hook")
        if ref_hook:
            # (CE2b: a line that points at another listing carries one need, of the property it names outside the pointing
            # phrase; that listing's facts are not needs, drops or freed properties - the drop and restore tests read the
            # lines with that phrase blanked out)
            try:
                masked = ref_hook(reply, unseen, asked) or unseen
            except Exception as exc:
                self.protocol.journal.append(dict(kind="ref_line_error", error=f"{type(exc).__name__}: {exc}"[:300]))
        notes = []
        # (a requirement must rest on the first reading or on the shopper's own words: one whose words show up in neither
        # was made up from the story and is ignored)
        heard_text = " ".join([so_far, " ".join(self.dialogues), asked or ""]).lower()

        def grounded(req: str) -> bool:
            long_words = [w for w in re.findall(r"[a-z]+", req.lower()) if len(w) > 2]
            return not long_words or any(w in heard_text for w in long_words)

        try:
            self.verbatim_settings(reply)
        except Exception as exc:
            self.protocol.journal.append(dict(kind="literal_value_error", error=f"{type(exc).__name__}: {exc}"[:300]))
        self.fresh_needs = [n for n in reply.get("needs") or [] if isinstance(n, str)]     # (CJ1: read by drop_named)
        rebuilt = reply.get("requirements_now")
        if isinstance(rebuilt, list) and all(isinstance(req, str) for req in rebuilt):
            self.now_reqs = [req for req in rebuilt if req.strip() and grounded(req)]
        for item in reply.get("needs") or []:
            if isinstance(item, str) and item and item not in self.uncovered and grounded(item):
                self.uncovered.append(item)
                notes.append(f"requirement added: {item}")
        try:
            # (BT1: a need stated by pointing at another listing is handed to the code's reference resolver)
            given = reply.get("references")
            infos = [r for r in given or [] if isinstance(r, dict)] if isinstance(given, list) else []
            for item in reply.get("needs") or []:
                if isinstance(item, str) and item in self.uncovered\
                        and (reference_passage(item.partition(":")[2]) or item in (reply.get("ref_needs") or [])):
                    self.ref_news = list(vars(self).get("ref_news") or []) + [dict(need=item, infos=infos, line=" ".join(unseen))]
        except Exception:
            pass
        pool = self.written + self.uncovered if self.now_reqs is None else self.now_reqs
        retained = {req.split(":", 1)[0].strip().lower() for req in pool
                if isinstance(req, str) and ":" in req and req not in self.let_go}
        for free in reply.get("no_preference") or []:
            if isinstance(free, str) and free and free not in self.indifferent and free.strip().lower() not in retained:
                if self.anchored(free, asked):
                    # (the stated value stands; for the question's bookkeeping the reply still counts as a "don't mind")
                    self.pinned_heard = 1 + vars(self).get("pinned_heard", 0)
                else:
                    self.indifferent.append(free)
        widened = False
        budget_given = whether_numeric(reply.get("new_budget"))
        for gone in reply.get("dropped") or []:
            if not (isinstance(gone, str) and gone) or gone in self.let_go:
                continue
            if budget_given and re.search(r"\b(?:prices?|budget|costs?|spend\w*|cents?|dollars?)\b", gone.lower()):
                continue
            if self.discard_titled(gone, masked):
                self.let_go.append(gone)
                notes.append(f"no longer needed: {gone}")
                if not widened:
                    widened = self.priority() != "none"
            else:
                # (a stated requirement that the shopper's lines never name stays: the drop came from a reply about some
                # similarly named other property)
                self.protocol.journal.append(dict(kind="drop_kept", need=gone[:200]))
        new_cap = reply.get("new_budget")
        if whether_numeric(new_cap) and new_cap > 0 and new_cap != self.outlay_lid:
            self.outlay_lid = new_cap
            notes.append(f"budget is now {new_cap}")
        if reply.get("unavailable"):
            notes.append("an item can no longer be had; remove it from the cart and leave it out of the order")
        try:
            # (instructions in the lines bind from here on; a stated allowance is passed to the errand, which applies it)
            added = self.attach_informed(reply.get("instructions"), "store" if asked is None else "shopper")
            if added:
                notes.append("instruction added: " + "; ".join(added))
            limits = reply.get("allowance")
            if isinstance(limits, dict):
                limits = {key: val for key, val in limits.items() if whether_numeric(val) and not isinstance(val, bool)}
                if limits:
                    self.allowance_news = limits
                    notes.append("limits changed: " + json.dumps(self.allowance_news))
        except Exception as exc:
            self.protocol.journal.append(dict(kind="told_error", error=f"{type(exc).__name__}: {exc}"[:300]))
        if notes:
            self.iteration = self.iteration + 1
            if widened:
                self.broadened_when, self.broadened_pulse = self.iteration, self.protocol.pulse
        if self.now_reqs is not None:
            # (a revealed need holds until the shopper drops it: a rebuilt list that lost it after a "don't mind" about
            # something else gets it back)
            for item in self.uncovered:
                key = item.split(":", 1)[0].strip().lower()
                if key and not any(key in req.lower() for req in self.now_reqs) and not any(key in old.lower() for old in self.let_go):
                    self.now_reqs.append(item)

            def twin(head: str, mark: list[str], other: str) -> bool:
                other_head = other.split(":", 1)[0]
                return identical_trait(head, other_head) and re.findall(r"\(([^)]*)\)", other_head) == mark

            for item in self.written:
                # (a stated requirement holds as well unless dropped: a rebuilt list that lost it, with nothing of that
                # property left, gets it back; when a line names the property, reading it is the listener's job)
                try:
                    if item not in self.let_go and item not in self.now_reqs and not self.discard_titled(item, masked):
                        head = item.split(":", 1)[0]
                        mark = re.findall(r"\(([^)]*)\)", head)
                        if not any(twin(head, mark, old) for old in self.let_go)\
                                and not any(twin(head, mark, req) for req in self.now_reqs if isinstance(req, str)):
                            self.now_reqs.append(item)
                except Exception:
                    continue
        if widened:
            # (judged after the restore above: a revealed need missing from the rebuilt list still counts)
            if self.traits_remaining():
                notes.append("a dropped requirement MAKES MORE ACCEPTABLE: listings that failed on it alone fit now, and listings "
                             "never searched for may beat the current pick; search the item kind again without it - short "
                             "queries, each remaining required value plus the kind word, value first, every page read - and "
                             "take the cheapest of everything acceptable now")
            elif self.priority() == "cheapest" and self.surveyable:
                notes.append("no property requirement remains: any in-stock listing in the requested category fits, whatever "
                             "the object (an accessory, a part or a set shelved there counts too), and the cheapest on the whole "
                             "shelf wins. Rows arrive by relevance or id, not by price, so more slices of the shelf are read for "
                             "you next (the category listing's pages, the label's own words, words naming the kinds of object in "
                             "the titles); if no '[shelf sweep done]' list arrives, cover the shelf yourself with short queries, "
                             "first page of each - the category label, each kind word of the label by itself, each item name "
                             "that keeps coming back in the titles - plus the category listing's pages. Then make the "
                             "lowest-priced in-stock row of the requested category inside the budget the cart's single line, "
                             "re-check its stock and price, and order it")
            else:
                notes.append("no property requirement remains: any in-stock listing in the requested category fits, whatever "
                             "the object (an accessory, a part or a set shelved there counts too), and the cheapest on the whole "
                             "shelf wins. Rows arrive by relevance or id, not by price: cover the shelf with short queries, "
                             "first page of each - the category label, each kind word of the label by itself, each item name "
                             "that keeps coming back in the titles - plus the category listing's pages; then order the lowest "
                             "in-stock price")
        self.protocol.journal.append(dict(kind="heard", question=asked is not None, needs=len(reply.get("needs") or []),
                                  free=len(reply.get("no_preference") or []), dropped=len(reply.get("dropped") or []),
                                  budget=whether_numeric(new_cap)))
        # (only a question from the shopper earns a reply: answering a plain statement costs a message and draws a
        # "don't mind" that looks like a change)
        wants_reply = asked is None and any("?" in ln for ln in unseen) and not reply.get("no_preference")
        self.response_owed = str(reply.get("reply") or "").strip() if wants_reply else ""
        lead = "Meaning: " + str(reply.get("meaning") or "")
        return lead + " (" + "; ".join(notes) + ")" if notes else lead


# ================================================================================================================ the cart
class CarriageLedger:
    """What the code believes the cart holds: which form of the cart tool adds units (the enum values of the first call
    the store accepted), each line's units as counted from the calls and corrected by every cart view, and whether the
    tool adds to a line already held or overwrites it."""

    def __init__(self, tool_name: str | None, schema: dict[str, Any]):
        self.definite_instrument, self.trait_specifications = tool_name, (schema or {}).get("properties") or {}
        self.measures, self.pending = {}, {}      # pending: line -> units expected after the last call if the tool overwrites
        self.place_layout = None
        self.attaches = None          # True: a put on a held line adds to it; False: it overwrites; None: unknown so far
        self.employed, self.final_snapshot = False, None
        self.inspected_when, self.altered_when = -1, 0

    def layout(self, args: dict[str, Any]) -> str:
        specs = self.trait_specifications
        return into_passage({name: given for name, given in args.items()
                        if isinstance(specs.get(name), dict) and "enum" in specs[name]})

    def lineitem_specification(self) -> tuple[str, str, str] | None:
        """(list argument, id field, units field) when the cart tool takes a list of line objects; None otherwise."""
        for arg_name, spec in self.trait_specifications.items():
            row_spec = spec.get("items") if isinstance(spec, dict) and spec.get("type") == "array" else None
            row_fields = row_spec.get("properties") if isinstance(row_spec, dict) else None
            if isinstance(row_fields, dict):
                text_fields = [f for f, s in row_fields.items() if isinstance(s, dict) and s.get("type") == "string" and "enum" not in s]
                id_fields = [f for f in text_fields if "id" in f.lower()] or text_fields
                num_fields = [f for f, s in row_fields.items() if isinstance(s, dict) and s.get("type") in ("integer", "number")]
                if id_fields and num_fields:
                    return arg_name, id_fields[0], num_fields[0]
        return None

    def lines(self, args: dict[str, Any]) -> list[tuple[str, float | None]]:
        """Every (listing, units) line a cart call names: each row of a list argument, else its single flat line."""
        spec = self.lineitem_specification()
        if not spec or not isinstance(args.get(spec[0]), list):
            target = self.intent_uniform(args)
            return [(target, self.headcount_param(args))] if target else []
        list_arg, id_field, num_field = spec
        return [(row[id_field].strip(), float(row[num_field]) if whether_numeric(row.get(num_field)) else None)
                for row in args[list_arg]
                if isinstance(row, dict) and isinstance(row.get(id_field), str) and row[id_field].strip()]

    def aim(self, args: dict[str, Any]) -> str | None:
        """The listing a cart call adds (the first line of a list argument carrying units; otherwise its first line)."""
        named = self.lines(args)
        adding = [pair for pair in named if pair[1] is None or pair[1] > 0]
        return (adding or named)[0][0] if named else None

    def intent_uniform(self, args: dict[str, Any]) -> str | None:
        """The listing a flat cart call names: a single-word string argument that is not an enum."""
        for name, given in args.items():
            spec = self.trait_specifications.get(name)
            enum_arg = isinstance(spec, dict) and "enum" in spec
            if isinstance(given, str) and not enum_arg and given.strip() and " " not in given.strip():
                return given
        return None

    def headcount_param(self, args: dict[str, Any]) -> float | None:
        spec = self.lineitem_specification()
        if spec and isinstance(args.get(spec[0]), list):
            named = self.lines(args)
            adding = [pair for pair in named if pair[1] is None or pair[1] > 0]
            return (adding or named)[0][1] if named else None
        numeric_specs = [(name, s) for name, s in self.trait_specifications.items()
                         if isinstance(s, dict) and s.get("type") in ("integer", "number") and "enum" not in s]
        for name, s in numeric_specs:
            given = args.get(name, s.get("default"))
            if whether_numeric(given):
                return float(given)
        return None

    def whether_place(self, args: dict[str, Any]) -> bool:
        return True if self.place_layout is None else self.layout(args) == self.place_layout

    def retained(self, line: str) -> float:
        return self.measures.get(line, 0.0)

    def enrolled(self, args: dict[str, Any], answer: Any) -> str:
        """Records a cart call the store answered and returns a note on what the cart now holds."""
        self.employed = True
        if isinstance(answer, dict) and answer.get("error"):
            return ""
        if self.place_layout is None:
            self.place_layout = self.layout(args)
        spec = self.lineitem_specification()
        if spec and isinstance(args.get(spec[0]), list):
            # (a list of lines describes the whole basket: lines it omits are gone; a later cart view corrects any slip)
            self.attaches, self.measures = False, {}
            for target, measures in self.lines(args):
                if measures is None or measures > 0:
                    self.measures[target] = self.measures.get(target, 0.0) + (1.0 if measures is None else measures)
            shown = self.snapshot_measures(answer)
            if shown is not None:
                self.discover(shown)
            return self.recap_alt()
        target = self.aim(args)
        if not target:
            return ""
        given = self.headcount_param(args)
        measures = 1.0 if given is None else given
        before = self.retained(target)
        putting = self.whether_place(args)
        if not putting:
            self.measures[target] = max(0.0, before - measures) if measures > 0 else 0.0
        elif before > 0 and self.attaches is not False:
            self.measures[target] = before + measures
            self.pending = {target: measures}         # what the line would hold if the tool overwrote instead of adding
        else:
            self.measures[target] = measures if measures > 0 else 0.0
        if self.measures.get(target, 0) <= 0:
            del self.measures[target]
        shown = self.snapshot_measures(answer)
        if shown is None:
            if (measures <= 0 or not putting) and self.displays_blank(answer, target):
                # (v605e: a removal answered with a list of the cart's lines that lists none - the removed line is gone;
                # other counted lines stay until a view shows the cart, as such a list may carry only what changed)
                self.measures.pop(target, None)
                self.pending.pop(target, None)
        elif set(shown) <= {target}:
            # (a reply that names only the line just touched settles that line; the other lines keep their count)
            self.finalize(target, shown.get(target, 0.0))
        else:
            self.discover(shown)
        return self.recap_alt()

    def displays_blank(self, answer: Any, line: str) -> bool:
        """v605e: the reply to removing `line` shows an empty cart - its lists are all empty, it has at least one number and
        every number is 0, and it names no other line the code still counts (a bare acknowledgment with no list or no
        number proves nothing; naming the removed line is allowed)."""
        if isinstance(answer, dict) and line in self.measures:
            lists = [v for v in answer.values() if isinstance(v, list)]
            numbers = [v for v in answer.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if lists and not any(lists) and numbers and not any(v != 0 for v in numbers):
                rendered = into_passage(answer)
                return all(other not in rendered for other in self.measures if other != line)
        return False

    def snapshot_measures(self, answer: Any) -> dict[str, float] | None:
        """Line -> units taken from a cart-like observation: dicts that hold a known or single-word id string and exactly
        one positive integer field (the units). None when no such row appears."""
        found: dict[str, float] = {}
        known = set(self.pending) | set(self.measures)

        def inspect(rec: dict[str, Any]) -> None:
            ids = [v for v in rec.values()
                   if isinstance(v, str) and " " not in v and (v in known or re.fullmatch(UID_RESEMBLING, v))]
            numeric = [(f, v) for f, v in rec.items()
                       if isinstance(v, int) and not isinstance(v, bool) and 0 < v < 1000 and str(f) != "price"]
            if len(numeric) > 1:
                # (a row with both a unit price and a count: the field named like a count holds the units)
                countish = [(f, v) for f, v in numeric
                            if re.search(r"qty|quant|count|units|number|total|held|ordered", str(f).lower())
                            and not re.search(r"price|cost|due|sum|fee|charge", str(f).lower())]
                if len(countish) == 1:
                    numeric = countish
            if ids and len(numeric) > 1:
                # (no field is named like a count: take the one equal to the units counted for that line, else the smallest)
                lead = next((i for i in ids if i in known), ids[0])
                expected = self.measures.get(lead, self.pending.get(lead))
                equal = [] if expected is None else [(f, v) for f, v in numeric if abs(v - expected) < 1e-9]
                numeric = equal[:1] if len(equal) == 1 else [min(numeric, key=lambda pair: pair[1])]
            if ids and len(numeric) == 1:
                owner = next((i for i in ids if i in known), ids[0])
                found[owner] = found.get(owner, 0) + numeric[0][1]

        traverse_mappings(answer, inspect)
        if found and (not known or set(found) & known):
            return found
        return None

    def finalize(self, line: str, shown: float) -> None:
        """One line's units as a cart reply shows them: decides whether a put adds or overwrites, other lines untouched."""
        overwrite_value = self.pending.pop(line, None)
        if overwrite_value is not None:
            if abs(shown - overwrite_value) < 1e-9 and abs(self.measures.get(line, 0) - overwrite_value) > 1e-9:
                self.attaches = False
            elif shown > overwrite_value:
                self.attaches = True
        if shown > 0:
            self.measures[line] = shown
        else:
            self.measures.pop(line, None)

    def discover(self, view: dict[str, float]) -> None:
        """A cart view wins: it replaces the code's count and decides whether a put adds or overwrites."""
        for line, overwrite_value in self.pending.items():
            if line in view:
                if abs(view[line] - overwrite_value) < 1e-9 and abs(self.measures.get(line, 0) - overwrite_value) > 1e-9:
                    self.attaches = False
                elif view[line] > overwrite_value:
                    self.attaches = True
        self.pending, self.measures = {}, {line: measures for line, measures in view.items() if measures > 0}

    def inspected(self, answer: Any, step_no: int) -> None:
        self.inspected_when, self.final_snapshot = step_no, answer
        shown = self.snapshot_measures(answer)
        if shown is not None:
            self.discover(shown)
        elif isinstance(answer, (dict, list)) and not self.uids_among(answer):
            self.measures = {}

    def uids_among(self, answer: Any) -> bool:
        rendered = into_passage(answer)
        return any(line in rendered for line in self.measures)

    def recap_alt(self) -> str:
        shown = [f"{line} x{int(measures) if float(measures).is_integer() else measures}" for line, measures in self.measures.items()]
        return "[cart as counted] " + (", ".join(shown) if shown else "empty") + "."

    def defects(self, want_units: int, per_product: list[int] | None = None) -> list[str]:
        """What stops the cart from being exactly one line at the wanted units (one line per product, given `per_product`)."""
        def line_list() -> str:
            return ", ".join(f"{ident} x{int(measures)}" for ident, measures in self.measures.items())

        if not self.measures:
            return []
        if per_product and len(per_product) > 1:
            if len(self.measures) != len(per_product):
                return [f"the cart has {len(self.measures)} line(s) ({line_list()}) while the request names {len(per_product)} "
                        f"products, one line apiece; each product needs its own line and the cart nothing more"]
            held_units, wanted = sorted(int(measures) for measures in self.measures.values()), sorted(per_product)
            return [] if held_units == wanted else [
                f"the lines carry {held_units} units while the products call for {wanted} (one line per product, each at its quantity)"]
        if len(self.measures) > 1:
            return [f"the cart has more than one line ({line_list()}); remove every line except the chosen one"]
        return [f"line {line} has {int(measures)} units where the request needs {want_units}; "
                f"{'take out' if measures > want_units else 'put in'} {int(abs(measures - want_units))} with the cart tool"
                for line, measures in self.measures.items() if abs(measures - want_units) > 1e-9]


# ================================================================================================================ the errand
class OutingNucleus:
    """Errand piece: state, limits, tool roles, the clock and the status line the actor sees."""

    def __init__(self, payload: dict[str, Any]):
        env = payload["environment"]
        self.protocol = Protocol(env["binding"])
        self.policyview = view = env["policy_view"]
        self.protocol.done = bool(view.get("finished"))
        self.request = str(view.get("query") or "")
        self.stage_lid = int(view.get("max_steps") or 30)
        self.toolkit = [entry for entry in view.get("tools") or []
                        if isinstance(entry, dict) and isinstance(entry.get("function"), dict)]
        self.structures = {}
        for entry in self.toolkit:
            fn = entry["function"]
            self.structures[fn["name"]] = fn.get("parameters") or {"type": "object", "properties": {}}
        self.vault = Chronicles()
        self.needs = Requirements(self.protocol, self.request)
        self.needs.ref_hook = self.reference_remedy          # (CE2b: the shopper's pointing lines are read by the code)
        self.duty_table = {}
        self.handcart = CarriageLedger(None, {})
        self.letter_lid = None
        self.letters_transmitted = self.rejection_num = self.purchases_attempted = self.stuck = 0
        self.purchase_sanctioned = self.abstain_sanctioned = self.audit_done = False
        self.inquiring = True
        self.enquiries_journal, self.converse, self.final_invokes = [], [], []
        self.standing_assessed, self.assessed_attaches, self.cautioned = set(), set(), set()
        self.verdicts_alt = {}
        self.succeeded = set()             # item|version pairs the independent check found to meet everything
        self.rejected_lineitems = set()      # the lines in the cart when a checkout got refused
        self.rejections_observed = {}         # item -> refusals counted at the time it last went into the cart
        self.word_kind = ""             # 'any' / 'all': how the store's keyword search combines several words
        self.term_observed = []             # first pages seen: (argument and mode, other arguments, query words, total)
        self.foremost_observed = []             # (argument and mode, other arguments, query words, total, row marks)
        self.foremost_boundaries = {}           # argument and mode -> [largest k shown to be read, k values shown unread]
        self.foremost_terms = 0             # K if the store reads only a query's first K words; 0 while unknown
        self.foremost_context = None            # the argument and mode in which K was learned
        self.refs = []                  # requirements given by pointing at a listing, followed up by the code (BT)
        VERBATIM_SETTINGS.clear()          # (CJ2: word values are learned afresh for each errand)
        self.needs.field_hook = self.depot_attributes   # (CJ: the store's fields for a product's category, for drops and literals)

    # ----------------------------------------------------------------------------------------------- quick lookups
    @property
    def journal(self) -> list[dict[str, Any]]:
        return self.protocol.journal

    def shifts_remaining(self) -> int:
        return self.stage_lid - self.protocol.pulse

    def role(self, duty: str) -> str | None:
        return self.duty_table.get(duty)

    def inquire_threshold(self) -> int:
        return INQUIRE_STANDARD if self.letter_lid is None else self.letter_lid

    def cart_sealed(self) -> bool:
        """True once a message owed before the finish can no longer be sent again: every allowed message is spent while
        such an instruction is in force. A cart change then would leave the last message describing another cart, so the
        code stops pushing swaps (a cheaper fit found that late is not worth an unreported cart)."""
        try:
            if self.letter_lid is None or self.letters_transmitted < self.letter_lid:
                return False
            return any(entry.get("needs_message") for entry in self.needs.conclude_informed())
        except Exception:
            return False

    def letter_buffer(self) -> int:
        """Messages held back from questioning whenever a message tool exists: a single one, to tell the shopper the basket
        before ordering (or for the closing step the instructions require); all the rest go to questions."""
        return 1 if self.role("message") else 0

    def inquire_headroom(self) -> int:
        """Messages available for questions and replies: the allowance minus the closing reserve."""
        room = self.inquire_threshold() - self.letter_buffer()
        return room if room > 0 else 0

    def stage_buffer(self) -> int:
        """Steps held back for the closing message and the finish itself."""
        return 0 if not (self.role("message") or self.needs.conclude_informed()) else CONCLUDE_STAGES

    def enact_entitlement(self, limits: dict[str, Any]) -> None:
        """A message or step limit stated by a shopper line or a store notice (and read by the listener) holds from now on."""
        old_caps = (self.letter_lid, self.stage_lid)
        if whether_numeric(limits.get("messages_total")):
            self.letter_lid = max(self.letters_transmitted, int(limits.get("messages_total")))
        elif whether_numeric(limits.get("messages_left")):
            self.letter_lid = self.letters_transmitted + max(0, int(limits.get("messages_left")))
        step_total, step_left = limits.get("steps_total"), limits.get("steps_left")
        if whether_numeric(step_total) and self.protocol.pulse < int(step_total):
            self.stage_lid = min(self.stage_lid, int(step_total))
        elif whether_numeric(step_left) and int(step_left) >= 0:
            self.stage_lid = min(self.stage_lid, int(step_left) + self.protocol.pulse)
        if old_caps != (self.letter_lid, self.stage_lid):
            self.inquiring = self.inquiring and self.letters_transmitted < self.inquire_headroom()
            self.journal.append({"kind": "allowance", "news": limits, "caps": [self.letter_lid, self.stage_lid]})

    def invoke_category(self, tool: str, args: dict[str, Any] | None) -> str | None:
        """'order', 'decline' or None for a call; when one tool does both, its mode value decides."""
        order_tool, decline_tool = self.role("order"), self.role("decline")
        if not tool or tool not in (order_tool, decline_tool):
            return None
        if order_tool != decline_tool:
            return "decline" if tool != order_tool else "order"
        for given in (args or {}).values():
            low = given.lower() if isinstance(given, str) and len(given) <= 40 else ""
            if low and re.search(ABSTAIN_TERMS, low):
                return "decline"
            if low and re.search(PURCHASE_TERMS, low):
                break
        return "order"

    def concluders(self) -> set[str]:
        tools = {self.role("order"), self.role("decline")}
        tools.discard(None)
        return tools

    def timer_passage(self) -> str:
        """v605e: remaining time as the actor sees it - silent while over half of the errand's time is left, then rounded
        down to CLOCK_STEP seconds (exact seconds vary from run to run and would re-sample the actor). Display only: every
        decision that depends on time reads wire.spare() directly."""
        remaining = max(0.0, self.protocol.surplus_alt() - FINAL_INVOKE)
        if remaining > (CUTOFF - FINAL_INVOKE) / 2:
            return ""
        rounded = TIMER_STAGE * int(remaining // TIMER_STAGE)
        return f"roughly {rounded} s remain" if rounded > 0 else f"less than {TIMER_STAGE} s remain"

    def progress_lineitem(self) -> str:
        left_steps = self.shifts_remaining()
        sent = f"messages used {self.letters_transmitted}" + ("" if self.letter_lid is None else f" of {self.letter_lid}")
        pieces = [f"[status] {self.protocol.pulse} of {self.stage_lid} steps used, {left_steps} remaining", sent]
        pieces.extend(filter(None, [self.timer_passage()]))
        out = "; ".join(pieces) + ".\n[requirements now] " + self.needs.passage_payload() + self.ref_line() + self.ware_stages()
        out += self.needs.informed_passage()
        try:
            out += self.boundary_clue()
            out += self.prerequisite_clue()
        except Exception:
            pass
        if self.letter_buffer() and self.inquire_threshold() > self.letters_transmitted:
            out += (" (the planned questions keep one message back for the required closing step: send it yourself, "
                    "or the code sends it at the finish)")
        viable = [entry["function"]["name"] for entry in self.proposed_alt()]
        if len(self.toolkit) > len(viable):
            extra = "; questions about hidden needs are also planned and sent for you" if self.inquiring else ""
            out += "\n[tools usable now] " + ", ".join(viable) + " (any other tool call is turned down without costing a step" + extra + ")."
        if self.handcart.employed:
            out += "\n" + self.handcart.recap_alt()
        several = len(self.needs.items()) > 0
        if left_steps <= 2:
            out += (" LAST STEPS: order the cart at once if it holds a candidate for "
                    + ("each product asked for" if several else "the item")
                    + "; if not, add the best candidate (a verified fit, else the cheapest in-stock, in-budget listing whose "
                    "record shows no failing requirement) and order. Decline only if the records show nothing can fit.")
        elif left_steps <= 5 or WRAPUP > self.protocol.surplus_alt():
            out += (" FINISH NOW: set the cart to " + ("one line for each product asked for" if several else "a single line")
                    + " (a verified fit, else the cheapest in-stock, in-budget listing whose record shows no failing "
                    "requirement; drop any sold-out line), re-check its stock, then order. Decline only if the records show "
                    "nothing can fit.")
        elif self.protocol.expended_depleted():
            out += (" MODEL BUDGET USED: stop exploring; in the next few steps add the best candidate for "
                    + ("each product asked for (one line apiece, at its quantity)" if several else "the item asked for")
                    + " to the cart (a verified fit when there is one, else the closest fit the records show), re-check its "
                    "stock and price, and order. A decline is correct only if the records show "
                    + ("some product has" if several else "the store has") + " no listing able to meet its requirements.")
        elif self.purchases_attempted == 0:
            due = self.purchase_owed()
            if due < left_steps <= due + 7 and self.tightening():
                out += (f" Steps are getting short: in the next few steps add your best verified fit to the cart and order; "
                        f"searching and detail reads stop once {due} steps remain, and repairing a refused order takes "
                        f"about 4 steps.")
            elif left_steps <= due and self.tightening():
                out += (" Searching is over (details only for the cart's line, listings a check pointed at, or - while some "
                        "product still lacks a verified fit - at most 3 more reads or searches in total): add your best "
                        "verified fit to the cart, re-check its stock and price, and order; a refused order still takes "
                        "about 4 steps to repair.")
        return out

    # ----------------------------------------------------------------------------------------------- setting up


class NarrativeClues:
    """Errand piece: the one-time reading of the request, story values and the hints written into it."""

    def peruse_petition(self) -> None:
        def tool_row(entry: dict[str, Any]) -> str:
            fn = entry["function"]
            return f"- {fn['name']}: {fn.get('description', '')} | arguments: {into_passage(fn.get('parameters') or {})}"

        opening = self.policyview.get("opening")
        content = [{"type": "text", "text": PERUSE_PETITION + "\n\nTOOLS:\n" + "\n".join(map(tool_row, self.toolkit)),
                    "cache_control": {"type": "ephemeral"}},
                   {"type": "text", "text": "REQUEST:\n" + self.request
                    + ("" if opening is None else "\n\nOPENING OBSERVATION:\n" + into_passage(opening))}]
        self.peruse_substance = content
        parsed: dict[str, Any] = {}
        readers = PERUSER + WATCHER[:1]
        for turn, model_id in enumerate(readers * 3):
            # (TA5: further rounds only while load or the network has lost every call, up to DOWN_AFTER)
            if turn >= len(readers) and (self.protocol.misses == 0 or self.protocol.invoke_breakdown == "refused" or self.protocol.lowered_alt()
                                         or not self.protocol.viable(readers) or WRAPUP + 60 > self.protocol.surplus_alt()):
                break
            answer = self.protocol.searchterm_llm([{"role": "user", "content": content}], [model_id], 3000)
            parsed = serialized_entity((answer or {}).get("content") or "") or {}
            if isinstance(parsed.get("roles"), dict):
                break
            self.protocol.journal.append(dict(kind="setup_retry", model=model_id))
        if not isinstance(parsed.get("roles"), dict):
            # (TA5: no model managed to read the request - the plan is built from its own words and the tools' schemas, so
            # the errand still shops and can still finish by code while the models stay silent; a model's reading takes
            # over once one answers again - catch_up)
            try:
                parsed = {**self.code_reading(), **{key: val for key, val in parsed.items() if key != "roles"}}
            except Exception as exc:
                self.protocol.journal.append(dict(kind="code_reading_error", error=f"{type(exc).__name__}: {exc}"[:300]))
                parsed = {}
            self.read_at = _TALLY.now()
            self.protocol.journal.append(dict(kind="code_reading", roles=parsed.get("roles"), product=parsed.get("product"),
                                      price_cap=parsed.get("price_cap"), items=len(parsed.get("items") or [])))
        self.employ_perusal(parsed)

    def employ_perusal(self, parsed: dict[str, Any], redo: bool = False) -> None:
        """Puts a reading of the request into the errand's plan; with `redo` (TA4: a model's reading replacing the code's)
        it keeps the cart tool the cart has been used with and any budget the shopper changed since, and resets the step
        limit to the world's own."""
        if not redo:
            self.steps_given = self.stage_lid
        prior_cart = self.role("cart")
        roles = parsed.get("roles") if isinstance(parsed.get("roles"), dict) else {}
        self.duty_table = {duty: tool for duty, tool in roles.items() if isinstance(tool, str) and tool in self.structures}
        if redo and self.handcart.employed:
            if prior_cart:
                self.duty_table["cart"] = prior_cart
        else:
            cart_tool = self.role("cart")
            self.handcart = CarriageLedger(cart_tool, self.structures.get(cart_tool or "", {}))
        self.needs.surveyable = self.sheet_param() is not None if self.role("search") else False
        msg_budget = parsed.get("message_budget")
        if whether_numeric(msg_budget) and msg_budget >= 0:
            self.letter_lid = int(msg_budget)
        self.needs.perusal = parsed
        try:
            # (the request's own rules start the instruction ledger; later lines add to it)
            self.needs.attach_informed(parsed.get("instructions"), "request")
            self.needs.attach_informed(list(filter(lambda rule: isinstance(rule, str), parsed.get("rules") or [])), "request")
        except Exception as exc:
            self.protocol.journal.append(dict(kind="told_seed_error", error=f"{type(exc).__name__}: {exc}"[:300]))
        try:
            # (several products need reads and checks for each: the spend at which the errand turns to its finish rises by
            # three tenths per product beyond the first, two extra at most)
            extra_products = len(self.needs.items()) - 1
            self.protocol.scale = 1.0 + 0.3 * min(2, max(0, extra_products))
        except Exception:
            self.protocol.scale = 1.0

        def complete(req: str) -> bool:
            prop, sep, value = req.partition(":")
            return bool(sep) and bool(prop.strip()) and bool(value.strip())

        named = [str(req) for req in parsed.get("requirements") or [] if req]
        # (a requirement must name its property; a wish without one is story or a soft want, never a requirement)
        self.needs.written = list(filter(complete, named))
        loose = [req for req in named if not complete(req)]
        if loose:
            parsed["soft_wants"] = [str(w) for w in parsed.get("soft_wants") or [] if w] + loose
            parsed["requirements"] = self.needs.written[:]
        try:
            self.story_values(parsed)
        except Exception as exc:
            self.protocol.journal.append(dict(kind="story_values_error", error=f"{type(exc).__name__}: {exc}"[:300]))
        try:
            self.reference_enroll()
        except Exception as exc:
            self.protocol.journal.append(dict(kind="ref_register_error", error=f"{type(exc).__name__}: {exc}"[:300]))
        if not redo or self.needs.iteration <= 0:
            self.needs.outlay_lid = parsed.get("price_cap")
        if parsed and self.needs.outlay_lid is None and self.role("message"):
            earlier = [str(a) for a in parsed.get("ask_first") or [] if a]
            parsed["ask_first"] = ["the most the shopper is willing to spend, as the request sets no price limit (skip it "
                                   "if the same point is listed again here)"] + earlier
        if redo:
            self.stage_lid = vars(self).get("steps_given", self.stage_lid)
        step_budget = parsed.get("step_budget")
        if whether_numeric(step_budget) and 0 < step_budget < self.stage_lid:
            self.stage_lid = int(step_budget)
        # (a long step budget spreads the same work over more turns: the spend caps shrink with it, down to half)
        stretch = max(0.5, min(1.0, 40.0 / max(1, self.stage_lid)))
        self.protocol.scale = self.protocol.__dict__.get("scale", 1.0) * stretch
        if redo:
            self.needs.iteration = self.needs.iteration + 1       # (checks done under the code's plan stop counting)

    def story_values(self, parsed: dict[str, Any]) -> None:
        """A requirement whose value words never appear in the request is a guess drawn from the story: it leaves the
        requirements (and its product's entry) and becomes a hint - at most half of them, never the only one; references
        are exempt. The hints end up as a list of strings."""
        raw = parsed.get("hints")
        if isinstance(raw, dict):
            raw = [f"{v}" if ":" in str(v) else f"{key}: {v}"
                   for key, vals in raw.items() for v in (vals if isinstance(vals, list) else [vals]) if v]
        hints = [single_trait_clue(str(h).strip()) for h in raw or []
                 if isinstance(h, (str, int, float)) and str(h).strip()] if isinstance(raw, (list, tuple)) else []
        own_hints = hints[:]
        given = parsed.get("references")
        refs = [str(r) for r in given or []] if isinstance(given, list) else []
        written = self.needs.written
        guessed = [req for req in written if narrative_estimate(req, self.request, refs)]
        if guessed and len(written) >= 2 and len(guessed) * 2 <= len(written):
            self.needs.written = [req for req in written if req not in guessed]
            parsed["requirements"] = self.needs.written[:]

            def core(req: str) -> tuple[str, str]:
                head, _, tail = str(req).partition(":")
                return re.split(r"\s*\(", head)[0].strip().lower(), tail.strip().lower()

            removed = set(map(core, guessed))
            for row in parsed.get("items") or []:
                if isinstance(row, dict) and isinstance(row.get("requirements"), list):
                    row["requirements"] = [r for r in row["requirements"] if core(r) not in removed]
            # (code reads this wording back from the hints, so it is kept as it was)
            hints.extend(f"{prop.strip()}: the story suggests {tail.strip()} (not stated as a value; ask)"
                         for prop, _, tail in (req.partition(":") for req in guessed))
            self.protocol.journal.append(dict(kind="story_values", moved=guessed))
        try:
            promoted = self.inscribed_clues(parsed, own_hints)
            hints = [h for h in hints if h not in promoted]
        except Exception as exc:
            self.protocol.journal.append(dict(kind="written_hint_error", error=f"{type(exc).__name__}: {exc}"[:300]))
        parsed["hints"] = hints[:12]

    def inscribed_clues(self, parsed: dict[str, Any], candidates_in: list[str]) -> list[str]:
        """Reader hints that are actually written values (lead_value: 'style (jumpsuit or romper): sexy, for a dinner' when
        the request says 'a sexy jumpsuit or romper'): each turns into the stated requirement 'style (jumpsuit or romper):
        sexy' unless that product already has a requirement of that property (or one carrying that word) in force. Returns
        the hints so turned."""
        rows = [r for r in parsed.get("items") or [] if isinstance(r, dict)]
        promoted: list[str] = []
        for hint in candidates_in:
            head, sep, detail = str(hint).partition(":")
            prop = re.split(r"\s*\(", head)[0].strip()
            if not (sep and prop) or rubbish_slot(prop):
                continue
            tag = re.findall(r"\(([^)]*)\)", head)
            names = [tag[-1]] if tag else ([str(r.get("product") or "") for r in rows] or [str(parsed.get("product") or "")])
            word, product = next(((w, n) for n in names for w in (foremost_setting(detail, self.request, n),) if w), ("", ""))
            mine = category_terms(product) if word else None

            def conflicts(req: str) -> bool:
                req_head, _, req_value = req.partition(":")
                req_tag = re.findall(r"\(([^)]*)\)", req_head)
                if req_tag and mine and not category_terms(req_tag[-1]) & mine:
                    return False
                return bool(identical_trait(req_head, prop) or re.search(r"\b" + re.escape(word) + r"\b", req_value.lower()))

            if word and not any(conflicts(req) for req in self.needs.written):
                new_req = f"{head.strip()}: {word}"
                self.needs.written.append(new_req)
                parsed["requirements"] = self.needs.written[:]
                target = next((row for row in rows if isinstance(row.get("requirements"), list)
                               and (len(rows) == 1 or category_terms(row.get("product") or "") & mine)), None)
                if target is not None:
                    target["requirements"].append(new_req)
                promoted.append(hint)
                self.protocol.journal.append(dict(kind="written_hint", hint=str(hint)[:200], requirement=new_req))
        return promoted

    def examine_clues(self, product_no: int | None = None) -> list[str]:
        """Story hints whose property is still open (not asked, not a requirement in force, not one the shopper does not
        mind); given product_no, only those of that product (or of no product). EC5: a hint whose name lies inside an
        answered property's name, or contains it (roast, coffee roast), counts as asked (asked_inside). EC3: a hint whose
        own property drew no reveal returns once, first, under its '(or X)' alternative (hint_alt)."""
        still_open, fallbacks = [], []
        for hint in self.needs.perusal.get("hints") or []:
            prop = str(hint).split(":", 1)[0].strip()
            several = bool(prop) and bool(self.needs.items())
            j = self.ware_belonging(prop) if several else None
            if not prop or (product_no is not None and j not in (product_no, None)):
                continue
            scope = j if several else None
            # (a hint is the story pointing at a property: a requirement under its name settles it only through a positive
            # value found among its listings' sample values, once its field has been seen; EC5: nor is a hint open whose
            # subject was asked under a name containing it)
            if self.examine_trait(scope, prop, self.trait_lineitem(j, prop), by_samples=True) and not self.inquired_enclosed(scope, prop):
                still_open.append(str(hint))
            else:
                alt = self.clue_variant(scope, str(hint), prop)
                if alt:
                    fallbacks.append(alt)
        return fallbacks + still_open
    def trait_lineitem(self, idx: int | None, pname: str) -> str:
        """Return the menu row (field name, how many listings, example values) whose field matches `pname` under
        strict_prop, taken from product idx's own records (its category's when there are several products, otherwise
        the whole store's); empty text when nothing matches. The menu is cached until a new record comes in."""
        try:
            memoized = self.__dict__.setdefault("menu_memo", {}).get(idx)

            def build() -> str:
                if not self.needs.items():
                    mine = self.native_rack()
                    return (self.vault.trait_picklist(mine) if mine else self.vault.trait_picklist())[0]
                shelf_sets = self.landing_racks()
                wanted = set() if idx is None or idx >= len(shelf_sets) else shelf_sets[idx]
                opened = [s for s in self.vault.racks_peruse() if s.strip().lower() in wanted]
                return "" if not opened else self.vault.trait_picklist(opened[0])[0]

            if memoized and memoized[0] == len(self.vault.items):
                menu_text = memoized[1]
            else:
                menu_text = build()
                self.menu_memo[idx] = (len(self.vault.items), menu_text)
            return next((row for row in menu_text.splitlines()
                         if rigorous_trait(row.strip("- ").split(" (in ")[0].strip(), pname)), "")
        except Exception:
            return ""

    def attribute_settings(self, idx: int | None, pname: str) -> set[str]:
        """Example values of product idx's field matching `pname`, normalised; an empty set when the field was never seen."""
        row = self.trait_lineitem(idx, pname)
        _, marker, rest = row.partition("):")
        if not marker:
            return set()
        # (spacing and hyphens are flattened, so round-neck equals round neck)
        squashed = {" ".join(re.findall(r"[a-z0-9]+", piece.lower())) for piece in rest.split("|")}
        squashed.discard("")
        return squashed

    def identical_attribute(self, idx: int | None, lhs: str, rhs: str) -> bool:
        """Whether two names denote a single field of product idx: strictly the same property name, or loosely related
        names (same_prop: collar / collar type, sleeve / sleeves) whose fields have an example value in common."""
        if rigorous_trait(lhs, rhs):
            return True
        try:
            overlap = identical_trait(lhs, rhs) and self.attribute_settings(idx, lhs) & self.attribute_settings(idx, rhs)
        except Exception:
            overlap = False
        return bool(overlap)

    def clued_trait(self, idx: int | None, pname: str) -> bool:
        """Whether some story hint names this property (strict_prop) for product idx, or for no particular product."""
        base_name = str(pname).split("(")[0].strip()

        def points_here(clue: Any) -> Any:
            lead = str(clue).split(":", 1)[0]
            return rigorous_trait(lead, base_name) and (
                idx is None or not self.needs.items() or self.ware_belonging(lead) in (idx, None))
        return any(points_here(h) for h in self.needs.perusal.get("hints") or [])

    def ware_peruses(self, idx: int | None) -> int:
        """How many records with facts were read on product idx's own category (single product: own category or all)."""
        try:
            if not self.needs.items():
                mine = self.native_rack()
                return self.vault.detailed_peruses(*([mine] if mine else []))
            shelf_sets = self.landing_racks()
            wanted = set() if idx is None or idx >= len(shelf_sets) else shelf_sets[idx]
            return sum(self.vault.detailed_peruses(s) for s in wanted) if wanted else 0
        except Exception:
            return 0

    def simple_clue(self, idx: int | None, pname: str, hinted: bool = False) -> bool:
        """True for a hinted property that no field of product idx's records carries by its strict name, once at least
        two of those records were read: it is then not asked under that bare name (EC1: from 2 reads the planner should
        map the story onto a key it was shown)."""
        try:
            base_name = re.split(r"\s*\(", str(pname))[0].strip()
            if base_name and (hinted or self.clued_trait(idx, base_name))\
                    and not self.trait_lineitem(idx, base_name) and self.ware_peruses(idx) >= 2:
                menu_text = (self.__dict__.get("menu_memo", {}).get(idx) or (0, ""))[1]
                # (a loosely related key - 'closure' beside 'closure type' - is still the hint's own field, just renamed)
                keys = (row.strip("- ").split(" (in ")[0].strip() for row in menu_text.splitlines())
                return all(not identical_trait(ckey, base_name) for ckey in keys)
        except Exception:
            pass
        return False

    def clue_flag(self, clue: str) -> str:
        """The hint as shown to the planner, tagged [no key] when plain_hint is true for it."""
        try:
            lead = str(clue).split(":", 1)[0].strip()
            owner = self.ware_belonging(lead) if self.needs.items() else None
            tagged = self.simple_clue(owner, lead, hinted=True)
        except Exception:
            tagged = False
        return f"{clue} [no key]" if tagged else str(clue)

    @staticmethod
    def clue_variants_belonging(detail: str) -> list[str]:
        """The alternatives written as '(or X, Y)' inside a hint's detail (one_prop_hint puts them there)."""
        found: list[str] = []
        for group in re.findall(r"\(or ([^()]+)\)", str(detail)):
            trimmed = (piece.strip() for piece in re.split(r",|\bor\b|/", group))
            found.extend(piece for piece in trimmed if piece)
        return found

    @staticmethod
    def moniker_enclosed(lhs: Any, rhs: Any) -> bool:
        """EC4/EC5: every meaningful word (3+ letters, not a link word) of one property name occurs in the other name:
        roast inside coffee roast, a bare key and its prefixed form for one subject."""
        def core(name: Any) -> set[str]:
            return {w for w in trait_terms(name) if len(w) >= 3 and w not in BOND_TERMS}
        left, right = core(lhs), core(rhs)
        return bool(left and right) and (left <= right or right <= left)

    def inquired_enclosed(self, idx: int | None, topic: str) -> bool:
        """EC5: some answered question about product idx (by its property tag, or else by the product its asked text
        names) concerned a property whose name contains, or is contained in, the hint's name - so that subject was
        already covered under a different key."""
        try:
            multi = bool(self.needs.items())

            def in_scope(piece: str, quote_m: Any) -> bool:
                if not multi:
                    return True
                owner = self.ware_belonging(piece)
                if owner is None and quote_m:
                    owner = self.ware_belonging(quote_m.group(1))
                return idx is None or owner in (idx, None)
            for log_line in self.enquiries_journal:
                caption = log_line.split(":", 1)[0].strip("- ")
                if "[not answered]" not in caption:
                    quote_m = re.search(r'asked "(.*?)" -> ', log_line)
                    if any(in_scope(piece, quote_m) and self.moniker_enclosed(piece, topic) for piece in caption.split(";")):
                        return True
        except Exception:
            pass
        return False

    def clue_variant(self, idx: int | None, clue: str, topic: str) -> str:
        """EC3: when a hint's own property was asked and got no reveal (a no-preference or muddled answer, recorded in
        quiet_asks), the hint comes back once under the first of its '(or X)' alternatives that is a key in product
        idx's records (prop_line), provided that key is still open and unasked; gives 'X (product): detail', else ''."""
        try:
            selected = self.__dict__.setdefault("hint_alts", {})
            multi = bool(self.needs.items())
            if not selected.get(clue):
                def went_quiet(q: Any) -> Any:
                    return (self.identical_attribute(idx, q, topic) or self.moniker_enclosed(q, topic)) and (
                        not multi or self.ware_belonging(q) in (idx, None))
                if not any(went_quiet(q) for q in self.__dict__.get("quiet_asks") or []):
                    return ""
                options = (o for o in self.clue_variants_belonging(clue.partition(":")[2]) if not rubbish_slot(o) and not rigorous_trait(o, topic))
                for option in options:
                    row = self.trait_lineitem(idx, option)
                    if row:
                        selected[clue] = row.strip("- ").split(" (in ")[0].strip()
                        break
            ckey = selected.get(clue) or ""
            if not ckey or ckey.lower() in self.__dict__.get("spent_asks", set()) or not self.scheme_examine(idx, ckey):
                return ""
            caption, _, rest = clue.partition(":")
            tag = re.search(r"\(([^()]*)\)\s*$", caption)
            detail = re.sub(r"\s*\(or [^()]+\)", "", rest).strip()
            suffix = "" if not tag else " (" + tag.group(1).strip() + ")"
            return f"{ckey}{suffix}: {detail} (asked first as {topic}, it drew no preference)"
        except Exception:
            return ""

    def clue_translated(self, idx: int | None, pname: str) -> bool:
        """EC4/EC5: some story hint for product idx (or for no product) leads to `pname`: its own property or one of its
        '(or X)' alternatives matches it under strict_prop, or the hint's text contains every word of the name."""
        try:
            base_name = re.split(r"\s*\(", str(pname))[0].strip()
            needed = [w for w in trait_terms(base_name) if len(w) >= 3 and w not in BOND_TERMS]
            multi = bool(self.needs.items())

            def leads_here(clue: Any) -> Any:
                lead, _, detail = str(clue).partition(":")
                if multi and idx is not None and self.ware_belonging(lead) not in (idx, None):
                    return False
                candidates = [re.split(r"\s*\(", lead)[0].strip(), *self.clue_variants_belonging(detail)]
                lowered = str(clue).lower()
                return any(rigorous_trait(c, base_name) for c in candidates if c) or (
                    bool(needed) and all(re.search(r"\b" + re.escape(w), lowered) for w in needed))
            return any(leads_here(c) for c in self.needs.perusal.get("hints") or [])
        except Exception:
            return False


class InquireVocabulary:
    """Errand component: the property names a shopper will recognise - twins, alternatives, distinct re-asks, first reads."""

    @staticmethod
    def moniker_sibling(lhs: Any, rhs: Any) -> bool:
        """EC4: two property names that could mean one subject. They must differ under strict_prop, and some word of
        one must match a word of the other (identical, or one a 4+ letter prefix of the other) where the matching word
        leads its own name (any word of a single-word name, otherwise any but the final word). Examples: two names of
        a component's kind, a short name and packaging type, material and clothing material - but plug type and bulb
        type, which only share the last word, are not twins."""
        if rigorous_trait(lhs, rhs):
            return False

        def split_up(name: Any) -> tuple[list[str], list[str]]:
            kept = [w for w in trait_terms(name) if len(w) >= 3 and w not in BOND_TERMS]
            return kept, (kept[:-1] if len(kept) > 1 else kept)

        def akin(x: str, y: str) -> bool:
            return x == y or (min(len(x), len(y)) > 3 and (y.startswith(x) or x.startswith(y)))
        every_a, lead_a = split_up(lhs)
        every_b, lead_b = split_up(rhs)
        return any(akin(x, y) for x in lead_a for y in every_b) or any(akin(x, y) for x in every_a for y in lead_b)

    def sibling_belonging(self, idx: int | None, lhs: str, rhs: str) -> bool:
        """EC4: for product idx, `lhs` may name the same subject as `rhs`: a name_twin, or one field under a related
        name (same_field) - never literally the same name."""
        try:
            distinct = not rigorous_trait(lhs, rhs)
            return distinct and (self.moniker_sibling(lhs, rhs) or self.identical_attribute(idx, lhs, rhs))
        except Exception:
            return False

    def sibling_slot(self, idx: int | None, pname: str) -> str:
        """EC4: the story twin of `pname` for product idx - a key of the product's records (most common in the menu
        first) that is open and not yet asked, linked by twin_of (but not one whose name sits inside the other's,
        which is the same subject), and that the story mentions: a word of its own missing from the property's name
        (3+ letters, not a link word, not a word of the product's name) shows up in one of the product's hints or in
        the request, or a hint maps onto it (hint_mapped). Empty when there is none."""
        try:
            multi = bool(self.needs.items())
            self.trait_lineitem(idx, pname)          # (fills the menu cache read just below)
            menu_text = (self.__dict__.get("menu_memo", {}).get(idx) or (0, ""))[1]
            if multi and idx is not None:
                prod_name = self.needs.items()[idx]["product"]
            else:
                prod_name = str(self.needs.perusal.get("product") or "")
            owned = category_lexemes(prod_name)
            pieces = [self.request.lower()]
            for h in self.needs.perusal.get("hints") or []:
                if not multi or self.ware_belonging(str(h).split(":", 1)[0]) in (idx, None):
                    pieces.append(str(h).lower())
            story = " ".join(pieces)
            used_asks = self.__dict__.get("spent_asks", set())
            mine = set(trait_terms(pname))

            def qualifies(ckey: str) -> Any:
                # (nested names such as roast / coffee roast are one subject: the earlier reply holds)
                if not ckey or rubbish_slot(ckey) or ckey.lower() in used_asks or not self.sibling_belonging(idx, ckey, pname)\
                        or not self.scheme_examine(idx, ckey) or self.moniker_enclosed(ckey, pname):
                    return False
                fresh = [w for w in trait_terms(ckey)
                         if len(w) >= 3 and w not in BOND_TERMS and w not in mine and not category_among(w, owned)]
                return any(re.search(r"\b" + re.escape(w), story) for w in fresh) or self.clue_translated(idx, ckey)
            keys = (row.strip("- ").split(" (in ")[0].strip() for row in menu_text.splitlines())
            return next((ckey for ckey in keys if qualifies(ckey)), "")
        except Exception:
            return ""

    def settings_layout(self, idx: int | None, pname: str, prod_name: str) -> str:
        """EC4: builds 'For the <product>, would you like A, B or C?' out of 2-3 example values of product idx's field
        `pname` that no other key of its records also shows (each distinct, containing a letter, at most 30
        characters); empty when fewer than two qualify."""
        try:
            row = self.trait_lineitem(idx, pname)
            if "):" in row:
                menu_text = (self.__dict__.get("menu_memo", {}).get(idx) or (0, ""))[1]

                def compress(v: str) -> str:
                    return " ".join(re.findall(r"[a-z0-9]+", v.lower()))
                elsewhere = {compress(v) for other in menu_text.splitlines() if "):" in other and other.strip() != row.strip()
                             for v in other.split("):", 1)[1].split("|")}
                selected: list[str] = []
                taken: set[str] = set()
                for v in (raw.strip() for raw in row.split("):", 1)[1].split("|")):
                    if v and len(v) <= 30 and re.search(r"[a-z]", v.lower()) and not re.search(r"[,;/]", v):
                        squashed = compress(v)
                        if squashed not in elsewhere and squashed not in taken:
                            selected.append(v)
                            taken.add(squashed)
                if len(selected) >= 2:
                    *front, last = selected[:3]
                    return f"For the {prod_name}, would you like {', '.join(front)} or {last}?"
        except Exception:
            pass
        return ""

    def unique_inquire(self, pname: str) -> tuple[str, str] | None:
        """EC4 (i): the single re-ask for a property whose plain question got a one-at-a-time reply (its name also
        stands for another subject the shopper has): returns (label, text). Preference order: the property's own values
        form (the clash is symmetric - the twin's name would draw the same answer, but the values point at this
        property alone; in the traced clashes the planner's property was the hidden one, e.g. mints packaging or the
        scales' battery size), then the story twin (twin_key) by its values, then the twin by name. At most once per
        product; None when nothing applies."""
        try:
            multi = bool(self.needs.items())
            idx = self.ware_belonging(pname) if multi else None
            asked_once = self.__dict__.setdefault("twin_asks", set())
            slot = idx if multi else -1
            if (multi and idx is None) or slot in asked_once:
                return None
            prod_name = self.needs.items()[idx]["product"] if multi else str(self.needs.perusal.get("product") or "item")
            caption = re.split(r"\s*\(", str(pname))[0].strip()
            qtext = self.settings_layout(idx, caption, prod_name)
            if not qtext:
                twin = self.sibling_slot(idx, pname)
                if twin:
                    caption = twin
                    qtext = self.settings_layout(idx, twin, prod_name) or f"Which {twin} would you like for the {prod_name}?"
            if qtext:
                asked_once.add(slot)
                return (caption if not multi else f"{caption} ({prod_name})"), qtext
        except Exception:
            pass
        return None

    def inquired_toward(self, idx: int | None, pname: Any) -> bool:
        """(v605f FC) whether any message in the questions log asked about `pname` (strict_prop) for product idx -
        matched by its property tag, or by the product its asked text names (any message when there is one product) -
        whether or not it was answered."""
        try:
            multi = bool(self.needs.items())
            base_name = re.split(r"\s*\(", str(pname or ""))[0].strip()

            def in_scope(piece: str, quote_m: Any) -> bool:
                if not multi:
                    return True
                owner = self.ware_belonging(piece)
                if owner is None and quote_m:
                    owner = self.ware_belonging(quote_m.group(1))
                return idx is None or owner == idx
            for log_line in self.enquiries_journal if base_name else ():
                caption = re.sub(r"\s*\[[^\]]*\]", "", log_line.split(":", 1)[0].strip("- "))
                quote_m = re.search(r'asked "(.*?)" -> ', log_line)
                pieces = [p.strip() for p in re.split(r";|\band\b", caption) if p.strip()]
                if any(in_scope(piece, quote_m) and rigorous_trait(re.split(r"\s*\(", piece)[0].strip(), base_name)
                       for piece in pieces):
                    return True
        except Exception:
            pass
        return False

    def variant_slot(self, idx: int | None, pname: str) -> str:
        """(v605f FC) the first '(or X)' alternative, from a story hint of product idx whose own property is `pname`
        (same_field or name_inside), that is a key of the product's records (prop_line), open (plan_open), not spent
        and never asked; empty otherwise."""
        try:
            base_name = re.split(r"\s*\(", str(pname))[0].strip()
            if not base_name:
                return ""
            multi = bool(self.needs.items())
            used_asks = self.__dict__.get("spent_asks", set())

            def alt_for(option: str, lead: str) -> str:
                if rubbish_slot(option) or rigorous_trait(option, lead) or rigorous_trait(option, base_name)\
                        or self.moniker_enclosed(option, base_name):
                    return ""
                row = self.trait_lineitem(idx, option)
                ckey = row.strip("- ").split(" (in ")[0].strip() if row else ""
                if not ckey or rigorous_trait(ckey, base_name) or ckey.lower() in used_asks\
                        or self.inquired_toward(idx, ckey) or not self.scheme_examine(idx, ckey):
                    return ""
                return ckey
            for clue in self.needs.perusal.get("hints") or []:
                full, _, detail = str(clue).partition(":")
                lead = re.split(r"\s*\(", full)[0].strip()
                relevant = bool(lead) and not (multi and self.ware_belonging(full) not in (idx, None))\
                    and (self.identical_attribute(idx, base_name, lead) or self.moniker_enclosed(base_name, lead))
                if relevant:
                    found = next(filter(None, (alt_for(o, lead) for o in self.clue_variants_belonging(detail))), "")
                    if found:
                        return found
            return ""
        except Exception:
            return ""

    def overlap_variant(self, pname: str) -> str:
        """(v605f FC1, EC3 after a one-at-a-time reply) the asked property got a one-at-a-time answer - its message
        also named another of the shopper's subjects, and the reply type depends on the subjects a message names - so
        the '(or X)' alternative of the hint that owns the property (alt_key) labels the next question as
        'X (product)'. Once per product (twin_asks, shared with distinct_ask and quiet_reask); empty when absent."""
        try:
            multi = bool(self.needs.items())
            idx = self.ware_belonging(pname) if multi else None
            asked_once = self.__dict__.setdefault("twin_asks", set())
            slot = idx if multi else -1
            ckey = "" if (multi and idx is None) or slot in asked_once else self.variant_slot(idx, pname)
            if ckey:
                asked_once.add(slot)
                return f"{ckey} ({self.needs.items()[idx]['product']})" if multi else ckey
        except Exception:
            pass
        return ""

    def keeps_simple(self, qtext: Any, plain_q: str) -> bool:
        """(v605f FC2) whether a question text already contains every word (by stem) of the plain form 'Which X would
        you like for the P?' - the property's and the product's words - in which case the plain form would name the
        same subjects again and is no real re-ask."""
        filler = {"which", "would", "you", "like", "for", "the"}
        present = {self.wordbase(w) for w in re.findall(r"[a-z0-9]+", str(qtext).lower())}
        wanted = [self.wordbase(w) for w in re.findall(r"[a-z0-9]+", str(plain_q).lower())
                  if w not in filler and w not in BOND_TERMS]
        return len(wanted) > 0 and all(w in present for w in wanted)

    def variant_unattempted(self, idx: int) -> bool:
        """(v605f FC4) whether product idx has a story hint whose property was asked and whose '(or X)' alternative
        has not been tried yet."""
        try:
            def open_alt(clue: Any) -> Any:
                full = str(clue).split(":", 1)[0]
                if self.ware_belonging(full) not in (idx, None):
                    return False
                lead = re.split(r"\s*\(", full)[0].strip()
                return lead and self.inquired_toward(idx, lead) and self.variant_slot(idx, lead)
            return any(open_alt(c) for c in self.needs.perusal.get("hints") or [])
        except Exception:
            return False

    def hushed_requery(self, pname: str) -> str:
        """EC4 (ii): after a no-preference or muddled answer about a property some story hint maps to (hint_mapped),
        the product's story twin (twin_key) labels the next question - once per product (shared with distinct_ask) -
        except when one of the product's hints already reopened under its own alternative (EC3), or, with several
        products, when another product has had no message yet (coverage comes first). Empty when none applies."""
        try:
            multi = bool(self.needs.items())
            idx = self.ware_belonging(pname) if multi else None
            asked_once = self.__dict__.setdefault("twin_asks", set())
            slot = idx if multi else -1
            if (multi and idx is None) or slot in asked_once or not self.clue_translated(idx, pname):
                return ""
            if any(" drew no preference)" in h for h in self.examine_clues(idx)):
                return ""
            if multi:
                reached = {self.ware_belonging(log_line.split(":", 1)[0]) for log_line in self.enquiries_journal}
                quotes = (re.search(r'asked "(.*?)" -> ', log_line) for log_line in self.enquiries_journal)
                reached.update(self.ware_belonging(m.group(1)) for m in quotes if m)
                if any(j != idx and j not in reached for j in range(len(self.needs.items()))):
                    return ""
            twin = self.sibling_slot(idx, pname)
            if twin:
                asked_once.add(slot)
                return f"{twin} ({self.needs.items()[idx]['product']})" if multi else twin
        except Exception:
            pass
        return ""

    @staticmethod
    def picklist_prune(menu_text: str, reads: int) -> str:
        """EC7: after 4 or more of a product's records were read, its question menu drops keys seen on just one."""
        if reads >= 4 and menu_text:
            survivors = [row for row in menu_text.splitlines() if re.search(r"\(in [01] listings?\)", row) is None]
            if survivors:
                return "\n".join(survivors)
        return menu_text

    def first_reads(self) -> bool:
        """EC2 (MC6): before the first planned message, in a store with no list-read tool (pre_read handles that case),
        every product whose category rows were seen but that has under 2 fact-bearing records read gets the best unread
        rows of its category opened by the code (menu_rows: in stock, inside the budget; at most 2 per product, least
        read products first). This happens once, and only within the steps left ahead of the order's reserve
        (order_due() + 2 per remaining message + 8). Returns True once the reads are queued."""
        try:
            already = self.__dict__.get("first_reads_done")
            if already or self.bundle_instrument() or not self.role("details"):
                return False
            self.first_reads_done = True
            row_map = self.needs.items()
            shelf_sets = self.landing_racks()
            headcounts = [row["quantity"] for row in row_map] or [self.needs.quantity()]
            left = self.shifts_remaining()
            due = self.purchase_owed()
            unsent = max(0, self.inquire_headroom() - self.letters_transmitted)
            room = left - due - 2 * unsent - 8
            wants = []
            taken: list[str] = []
            for idx in range(len(row_map) or 1):
                home = shelf_sets[idx] if idx < len(shelf_sets) else set()
                have = self.ware_peruses(idx if row_map else None)
                if room > 0 and home and have < 2:
                    picks = self.picklist_tablerows(home, 2 - have, taken, headcounts[min(idx, len(headcounts) - 1)])
                    taken.extend(picks)
                    if picks:
                        wants.append((have, idx, picks))
            ordered = sorted(wants)
            pending_ids = [picks[n] for n in (0, 1) for _, _, picks in ordered if len(picks) > n][: max(0, room)]
            if pending_ids:
                self.open_queue, self.open_recheck = pending_ids, True
                self.open_say = ("First I open a few listings of every requested product, to see their fields before "
                                 "questioning the shopper.")
                self.open_purpose = "menu"
                self.journal.append({"kind": "first_reads", "ids": pending_ids})
            return bool(pending_ids)
        except Exception as exc:
            self.journal.append({"kind": "first_reads_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
            return False

    def scheme_examine(self, idx: int | None, pname: str) -> bool:
        """open_prop for something the planner picked; if a story hint also names the property, a requirement under
        that name settles it only with a value found among the field's example values (hint plus pick are two signs
        that the property is still unknown)."""
        try:
            hinted = self.clued_trait(idx, pname)
            return self.examine_trait(idx, pname, self.trait_lineitem(idx, pname) if hinted else "", by_samples=hinted)
        except Exception:
            return self.examine_trait(idx, pname)


class Throttling:
    """Errand component: timing of the order, steps held in reserve, late-read holds and backup rows."""

    def finalizes(self, demand: str, examples: str = "") -> bool:
        """Whether a requirement in force closes its property for questioning: only a positive stated value does. A
        reference closes it; an exclusion or a description (positive_value) does not, nor does a value that neither
        the request nor the shopper's lines actually write (story_guess). Given `examples` (the field's example values),
        a word value none of whose words those samples carry does not close it either."""
        topic, _, req_val = str(demand).partition(":")
        lowered = req_val.lower()
        if re.search(r"as shown|same as|the page|sold by|redirect|refer", lowered) or reference_passage(lowered):
            return True
        listed = self.needs.perusal.get("references")
        known = [str(r) for r in listed or []] if isinstance(listed, list) else []
        known.extend(self.needs.__dict__.get("resolved") or [])
        heard = self.request + " " + " ".join(self.needs.dialogues)
        if not affirmative_setting(req_val, topic) or narrative_estimate(demand, heard, known):
            return False
        if not examples or re.search(r"\d", req_val):
            return True
        toks = [w for w in re.findall(r"[a-z]{3,}", lowered) if w not in BOND_TERMS]
        vocab = set(re.findall(r"[a-z]+", examples))

        def carried(w: str) -> bool:
            wordbase = w[:-1] if len(w) > 4 else w
            near = len(w) >= 4 and any(len(s) >= 4 and revisions_amid(s, w, 1) for s in vocab)
            return bool(re.search(r"\b" + re.escape(wordbase), examples)) or near
        return not toks or any(carried(w) for w in toks)

    # ------------------------------------------------------------------------------------- which tools are offered
    def purchase_owed(self) -> int:
        """How many steps are held back for the first order try plus the repair a refusal would need. CF: a third of
        the errand's steps (capped at 16: 13 for 40 steps, 12 for 37), never under the base rule (a quarter, no less
        than FIRST_ORDER_BY and no more than FIRST_ORDER_DUE, so small worlds keep their old value) and never over half
        the steps. Once this errand has seen a shopper change, a refusal or a sold-out record, at least 8 plus one per
        product (a repair re-reads, puts and orders each line). A late first finish succeeds much less often."""
        budget = self.stage_lid
        floor = max(INITIAL_PURCHASE_VIA, min(INITIAL_PURCHASE_OWED, budget // 4 + 1))
        due = max(floor, min(16, budget // 3))
        # (v605d M7: counts a refusal, a change or a CART LINE that sold out - search rows that were sold out all along do not)
        gone = self.__dict__.get("dead")
        if self.__dict__.get("event_seen") or (gone and set(self.handcart.measures) & gone):
            try:
                repair = 8 + max(1, len(self.needs.items()))
            except Exception:
                repair = 9
            due = max(due, repair)
        return max(floor, min(due, budget // 2)) + self.stage_buffer()

    def tightening(self) -> bool:
        return bool(self.role("order")) and bool(self.role("cart"))

    def proposed_alt(self) -> list[dict[str, Any]]:
        left = self.shifts_remaining()
        order_tool, decline_tool, cart_tool = self.role("order"), self.role("decline"), self.role("cart")
        allowed: set[str] | None = None
        if left <= 1:
            allowed = {order_tool, decline_tool}
        elif left <= 3:
            allowed = {order_tool, decline_tool, cart_tool}
        elif (left <= self.purchase_owed() and self.purchases_attempted == 0) or (left <= INITIAL_PURCHASE_VIA and self.rejection_num > 0):
            allowed = {order_tool, decline_tool, cart_tool, self.role("cart_view"),
                       self.role("view") or self.role("details"), self.inventory_instrument()}
            if left > 3 and self.substitute_examine():
                allowed.add(self.role("details"))     # (CF7: one read of the unread row a sold-out note pointed at)
            # (still a few late reads: when a check names listings to open, nothing is verified yet, or a change widened things)
            if self.__dict__.get("late_n", 0) < 3:
                if (self.unconfirmed() and left > 5) or\
                        (self.titled_examine() and left >= min(INITIAL_PURCHASE_VIA + 2, self.purchase_owed())):
                    allowed.add(self.role("details"))
                if (self.broadened_belated() or self.__dict__.get("named_pages") or (self.unconfirmed() and left > 5)) and left > 4:
                    allowed.add(self.role("search"))
            if left >= 7 and self.fallback_tablerows():
                allowed.add(self.role("details"))     # (ED3: two reads for the cheapest backups)
        if allowed is None or not order_tool or not cart_tool:
            return self.toolkit
        if self.role("message") and left >= 2 and (self.letter_lid is None or self.letters_transmitted < self.letter_lid):
            allowed.add(self.role("message"))         # (messaging stays open as long as no read allowance rules it out)
        kept = [tdef for tdef in self.toolkit if tdef["function"]["name"] in allowed]
        return kept or self.toolkit

    def among_buffer(self) -> bool:
        """Whether the steps held for the first order (or for the repair after a refusal) have started."""
        left = self.shifts_remaining()
        if not (self.role("order") and self.role("cart")) or left <= 3:
            return False
        first_due = left <= self.purchase_owed() and self.purchases_attempted == 0
        return first_due or (left <= INITIAL_PURCHASE_VIA and self.rejection_num > 0)

    def unconfirmed(self) -> bool:
        """True when no listing has passed the item check under the current requirements (refused cart lines aside);
        with several products, also while a product of known category still has no passed listing of its own."""
        dead_set = self.defunct_lineitems()
        iteration = str(self.needs.iteration)
        live = [rid for rid, _, tag in (entry.rpartition("|") for entry in self.succeeded)
                if tag == iteration and rid not in self.rejected_lineitems and rid not in dead_set]
        if not (live and self.needs.items()):
            return not live
        try:
            covered = {self.rack_familiar(i) for i in live} - {""}
            return bool(covered) and any(home and not home & covered for home in self.landing_racks())
        except Exception:
            return False

    def titled_examine(self) -> set[str]:
        """Listings named by a check that were seen, have not been read, and were not part of a refused cart."""
        seen = {i for i in self.__dict__.get("named", set()) if i in self.vault.ids}
        return seen - self.vault.read - self.rejected_lineitems

    def broadened_belated(self) -> bool:
        """Whether the latest change widened what is acceptable and arrived inside the steps held for the order."""
        needs = self.needs
        return needs.broadened_when == needs.iteration and self.protocol.pulse - needs.broadened_pulse <= self.purchase_owed()

    def commence_bundle(self) -> None:
        self.late_q, self.late_ids = 0, set()

    def belated_keep(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        """Inside the reserve, a details read of a cart line or of a verified fit (the stock re-check) always runs; any
        other read runs only while under 3 late reads were made or queued in this reply, and only for a listing a check
        named, while nothing is verified, or (for a search) just after a widening change."""
        details, search = self.role("details"), self.role("search")
        reads = (details, search)
        if tool_name not in reads and tool_name in self.retrieve_instruments() and details:
            # (CG5: any other single-listing lookup is treated in the reserve like a details read - stock re-reads of
            # a cart line or verified fit always run; absorb only counts details and search reads, so it is counted here)
            # (v605d M8: absorb counts it after it ran - the details screen above queues it for this reply)
            return self.belated_keep(details, tool_args)
        if tool_name not in reads or not self.among_buffer():
            return ""
        cited = {v for v in tool_args.values() if isinstance(v, str)}
        if tool_name == details:
            covered = set(self.handcart.measures) | {log_line.rpartition("|")[0] for log_line in self.succeeded}
            if cited & covered:
                return ""
        queued_n = self.__dict__.get("late_q", 0)
        left = self.shifts_remaining() - queued_n

        def backup_room() -> bool:
            return tool_name == details and left >= 7 and self.__dict__.get("backup_n", 0) < 2

        def spare_backups() -> list[str]:
            retained = self.__dict__.get("late_ids", set())
            fresh = [i for i in self.fallback_tablerows() if i not in retained]
            return fresh[: 2 - self.__dict__.get("backup_n", 0)]
        ok = False
        if self.__dict__.get("late_n", 0) + queued_n < 3:
            if tool_name != details:
                ok = (self.broadened_belated()
                      or json.dumps(tool_args, sort_keys=True, default=str) in self.__dict__.get("named_pages", set())
                      or (self.unconfirmed() and left > 5)) and left > 4
            else:
                ok = (self.unconfirmed() and left > 5) or\
                    (bool(cited & self.titled_examine()) and left >= min(INITIAL_PURCHASE_VIA + 2, self.purchase_owed()))
        if not ok and backup_room():
            # (ED3: past the 3 late reads, the cheapest unread rows under the cheapest verified fit - 2 reads overall)
            if cited & set(spare_backups()):
                ok = True
                self.backup_n = self.__dict__.get("backup_n", 0) + 1
        swap = "" if ok or tool_name != details else self.substitute_examine()
        if swap and swap in cited and left > 2:
            ok = True                     # (CF7: a single details read of the cheapest unread row a sold-out note listed)
            self.swap_done = True
        if ok:
            self.late_q = queued_n + 1
            self.__dict__.setdefault("late_ids", set()).update(cited)
            return ""
        extra = spare_backups() if backup_room() else []
        note = ("Not run (no step used): the remaining steps are reserved for the order. At this point details are read "
                "only for the cart's line, for a listing a check named, or (while nothing is verified) for at most 3 "
                "listings in total: place your best verified fit in the cart, re-check its stock and price, and order.")
        if extra:
            note += " Cheaper unread rows that could back up the pick may still be read: " + ", ".join(extra) + "."
        return note

    def fallback_tablerows(self) -> list[str]:
        """ED3 (v509c: the next-cheapest gets verified before checkout): under a cheapest rule, once a sold-out record
        or event appeared (event_seen, dead) or a check named listings still unread, the cheapest unread in-stock rows
        of each product's category priced under its cheapest verified fit (ed_pool), cheapest first; the reserve keeps
        2 details reads for them on top of its 3 late ones. Empty after those 2 are spent."""
        spent_backups = self.__dict__.get("backup_n", 0) >= 2
        if self.needs.priority() != "cheapest" or spent_backups\
                or not (self.__dict__.get("event_seen") or self.defunct_lineitems() or self.titled_examine()):
            return []
        found: list[tuple[float, str]] = []

        def viable(e: dict[str, Any], ceiling: float) -> bool:
            excluded = e["read"] or e["gone"] or e["over"] or e["pointer"] or e["ruled"] or e["clash"]
            return not excluded and e["price"] is not None and e["price"] < ceiling
        try:
            for idx in self.catalog_wares():
                pool = self.elicited_reservoir(idx)
                verified = [e["price"] for e in pool if e["verified"] and e["fit"] and e["price"] is not None]
                if verified:
                    found.extend((e["price"], e["id"]) for e in pool if viable(e, min(verified)))
        except Exception as exc:
            self.journal.append({"kind": "backup_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
            return []
        return [rid for _, rid in sorted(found)][:6]

    # ------------------------------------------------------------------------------------- the checks


class EntryChecks:
    """Errand component: per-field views, the single-item check, the rank check and the review before a decline."""

    def attribute_monikers(self) -> str:
        """Property field names from the listing details read so far - spellings differing only in separators are
        merged - each with the number of listings that carry it (a property's main field is the most widely carried)."""
        menu_text, _ = self.vault.trait_picklist()
        header = ("\n\nFIELD NAMES ON THIS STORE'S LISTINGS (spellings that differ only by separators are merged; "
                  "count = listings read that carry the field):\n")
        return header + menu_text if menu_text else ""

    def assess_entry(self, lid: str, aim_text: str, second: str = "") -> str:
        if self.protocol.surplus_alt() - WRAPUP < 20:
            return ""
        ckey = f"{lid}|{self.needs.iteration}"
        if ckey in self.__dict__.get("near_ok", set()):
            return ""     # (ED6: under these requirements a meaning check already accepted how it words each value it lacks)
        listed = self.needs.perusal.get("references")
        refs = [str(r) for r in listed or [] if r] if isinstance(listed, list) else []
        tracked_refs = list(self.__dict__.get("refs") or [])
        owned = self.own_redirects(lid)       # (CE6: once such notes show up the check runs afresh)
        memo_key = ckey
        if refs or any(r.get("status") != "resolved" for r in tracked_refs):
            memo_key += f"|{len(self.vault.read)}"
        if second:
            memo_key += "|second"
        if owned:
            memo_key += f"|own{len(owned)}"
        if memo_key in self.verdicts_alt:
            return self.verdicts_alt[memo_key]

        def reject(why: str, problems: list[str], extra: dict[str, Any]) -> str:
            verdict = f"An independent check says {lid} does NOT meet every requirement: {why}. Choose another listing."
            self.verdicts_alt[memo_key] = verdict
            self.journal.append({"kind": "verify", "item": lid,
                             "result": {"item": lid, "fits": False, "problems": problems, **extra}})
            return verdict
        pointer = self.reference_indicator_annotation(lid)
        if pointer:
            # (CE5: with differ, a listing on the reference's chain holds the excluded value by definition - no model needed)
            return reject(pointer, ["reference chain"], {})
        records = self.vault.concerning(lid)
        if not records:
            return ""
        evidence = self.attribute_snapshot(lid)
        top = evidence["fails"][:3]
        if top:
            # (CH2: the store writes the value itself in the field named for the property and this listing's value shares
            # nothing with it, or pairs it with a value that field also carries on its own - decided without a model)
            return reject("; ".join(top), top, {"by": "field evidence"})
        chunks = [ASSESS_ENTRY, "\n\nTHE SHOPPER'S REQUEST:\n", self.request, self.needs.alterations(),
                  "\n\nCURRENT REQUIREMENTS (stated, revealed and changed):\n", self.needs.passage_payload()]
        if self.needs.items():
            retained = self.handcart.retained(lid)
            chunks.append("\n\nThe item below belongs to one of the requested products (its category tells which): measure it "
                          "only against that product's requirements and quantity; the other products' requirements do not "
                          "concern it.")
            if retained > 0:
                chunks.append(f" The cart already holds {int(self.handcart.retained(lid))} units of it.")
        pointer_note = self.reference_annotation(lid)
        if pointer_note:
            chunks.append(pointer_note)
        loose = self.reference_unmonitored(refs, tracked_refs)
        if loose:
            toks = {w for r in loose for w in re.findall(r"[a-z]{4,}", r.lower().split(":", 1)[-1])}

            def mentions(i: str) -> bool:
                return any(w in " ".join(self.vault.concerning(i, 1)).lower() for w in toks)
            pointed = [i for i in sorted(self.vault.read) if i != lid and mentions(i)][:3]
            if len(pointed) > 0:
                chunks.append("\n\nRECORDS OF READ LISTINGS THE REQUEST POINTS AT (a requirement stated by reference takes "
                              "its value from the named field of the listing it means):\n"
                              + "\n".join(rec for i in pointed for rec in self.vault.concerning(i, 1)))
        chunks.append(second)
        if owned:
            notes = []
            for demand, pname, sent in owned:
                notes.append(f"one sentence in its notes might move {pname} to a different listing: \"{sent[:240]}\". If it "
                             f"does (it tells the reader to take {pname} from, refer to, go by or look to another listing), the "
                             f"grader reads {pname} from that other listing, so this item's own {pname} field does not count for "
                             f"'{demand}', which is then met only if that other listing's record shows it; notes naming a "
                             "listing merely as another version sold elsewhere, or as what this item is not, change nothing.")
            chunks.append("\n\nNOTE (the code read this item's own notes): " + " ".join(notes))
        chunks.append(f"\n\nTHE ITEM TO {aim_text}: {lid}\nRECORDS OF THIS ITEM SEEN IN THE STORE (oldest first):\n"
                      + "\n".join(records))
        cart_view = self.handcart.final_snapshot
        if cart_view is not None:
            chunks.append("\n\nLATEST CART VIEW:\n" + into_passage(cart_view)[:3000])
        chunks.append(self.attribute_monikers())
        ev_lines = evidence["lines"]
        if ev_lines:
            chunks.append("\n\nFIELD EVIDENCE (worked out by code from this store's records of the item's category; it shows "
                          "how the store writes each required value and what this item carries):\n" + "\n".join(ev_lines))
        labels = self.vault.tag_lineitems()
        if labels:
            chunks.append("\n\nLABELS THAT RECUR ON THIS STORE'S LISTINGS (with counts; the item's category label has to be "
                          "the one matching the request's wording as a whole, not a neighbouring label):\n" + labels)
        reply = self.protocol.confer("".join(chunks), 3000, ASSESSOR)
        self.journal.append({"kind": "verify", "item": lid, "result": reply})
        if reply and reply.get("fits") is False:
            problems_seen = [str(p) for p in reply.get("problems") or []][:5]
            verdict = (f"An independent check says {lid} does NOT meet every requirement: " + "; ".join(problems_seen)
                       + ". Unless you are sure the check is mistaken, pick another fully verified fit.")
            self.verdicts_alt[memo_key] = verdict
            return verdict
        self.verdicts_alt[memo_key] = ""
        if reply:
            self.succeeded.add(ckey)
            self.__dict__.setdefault("pass_price", {})[lid] = self.vault.price(lid)
        return ""

    def crosscheck_tablerows(self) -> dict[str, dict[str, Any]]:
        """CH: code_rows, cached until a new record arrives."""
        memoized = self.__dict__.get("ch_rows_memo")
        if memoized and memoized[0] == len(self.vault.items):
            return memoized[1]
        fresh = self.engine_tablerows()
        self.ch_rows_memo = (len(self.vault.items), fresh)
        return fresh

    def depot_attributes(self, caption: Any = "") -> dict[str, list[str]]:
        """CJ: property fields of the read listings in the category of the product `caption` names (via its "(product)"
        tag; for a single-product request, its own category), each with the values seen; all read listings' fields
        when no category is known."""
        try:
            row_map = self.crosscheck_tablerows()
            shelf_sets = self.landing_racks()
            multi = bool(self.needs.items())
            if multi:
                idx = self.ware_belonging(str(caption))
                wanted = set() if idx is None or idx >= len(shelf_sets) else set(shelf_sets[idx])
            else:
                mine = self.native_rack()
                wanted = {mine} if mine else (set(shelf_sets[0]) if shelf_sets else set())
            found: dict[str, list[str]] = {}
            for row in row_map.values():
                if row["fields"] and not (wanted and row["shelf"] not in wanted):
                    for name, val in row["fields"].items():
                        if str(val).strip() and not rubbish_slot(name):
                            found.setdefault(str(name), []).append(str(val))
        except Exception:
            found = {}
        return found

    def crosscheck_rerouted(self, lid: str, pname: str) -> bool:
        """CH: whether the listing's own notes might route this property to another listing (via CE's own_redirects when
        it exists; otherwise any of its records with redirect wording plus the property's name), so its own field may
        not settle it."""
        finder = getattr(self, "own_redirects", None)
        if callable(finder):
            try:
                groups = (got if isinstance(got, (list, tuple)) else [got] for got in finder(lid) or [])
                verdict = any(identical_trait(str(x).partition(":")[0], pname) for grp in groups for x in grp[:2])
            except Exception:
                verdict = True
            return verdict
        toks = [w for w in trait_terms(pname) if len(w) >= 3]
        name_pat = r"\b" + r"\w*[\s_-]+".join(re.escape(w) for w in toks)
        cue = r"\b(?:refer|taken|take it|go by|goes by|look to|look at|comes? from|read from)\b"

        def sentences() -> Any:
            for body in self.vault.concerning(lid, 8):
                yield from re.split(r"(?<=[.!?;])\s+", body.lower())
        return bool(toks) and any(re.search(name_pat, s) and re.search(cue, s) for s in sentences())

    def attribute_snapshot(self, lid: str) -> dict[str, list[str]]:
        """CH: one listing's FIELD EVIDENCE (field_facts), cached per listing, record count and requirements."""
        ckey = (lid, len(self.vault.items), self.needs.iteration, "|".join(self.needs.traits_remaining()))
        cache = self.__dict__.setdefault("ch_view_memo", {})
        if ckey not in cache:
            try:
                cache[ckey] = self.attribute_findings(lid)
            except Exception as exc:
                self.journal.append({"kind": "field_facts_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
                cache[ckey] = {"lines": [], "rules": [], "fails": []}
        accepted = (self.__dict__.get("near_said") or {}).get(f"{lid}|{self.needs.iteration}")
        if not accepted:
            return cache[ckey]
        # (v605f FE: a meaning check accepted how this listing words these values - drop its plain failures, say so in its lines)
        try:
            view = cache[ckey]
            props = [p for p, _, _ in accepted]

            def alike(name: str, p: Any) -> Any:
                return identical_trait(name, p) or rigorous_trait(name, p)

            def annotate(row: str) -> str:
                lead = re.match(r"- (.+?) = '", row)
                hit = next(((p, v, s) for p, v, s in accepted if alike(lead.group(1), p)), None) if lead else None
                if not hit:
                    return row
                return row + (f" A meaning check confirmed that its '{hit[2][:80]}' says '{hit[1][:60]}' in other "
                              "words: it meets it.")
            kept_fails = [f for f in view["fails"] if not any(alike(f.split(":", 1)[0].strip(), p) for p in props)]
            return {"lines": [annotate(row) for row in view["lines"]], "rules": list(view["rules"]), "fails": kept_fails}
        except Exception:
            return cache[ckey]

    def attribute_findings(self, lid: str) -> dict[str, list[str]]:
        """CH: for each requirement in force on the listing's product that says what the item is (not a limit with a
        figure, not a reference), worked out from the read listings of its category: the store's field for it (the
        one named like the property, else the one whose values write the value), how many listings carry that field,
        how many other listings write the value itself there, and what this listing shows. "lines" holds those facts;
        "rules" only the category part; "fails" the plain failures the code settles alone - the field named for the
        property shares no word, abbreviation start or figure with a value other listings of the category write there,
        or its only matching value pairs the value with another one that field also carries on its own."""
        result: dict[str, list[str]] = {"lines": [], "rules": [], "fails": []}
        row_map = self.crosscheck_tablerows()
        self_row = row_map.get(lid)
        if not self_row or not self_row["fields"]:
            return result
        multi = bool(self.needs.items())
        my_shelf = self_row["shelf"]
        if multi and not my_shelf:
            return result
        idx = None
        if multi:
            idx = next((n for n, home in enumerate(self.landing_racks()) if my_shelf in home), None)
        kin_rows = {i: r for i, r in row_map.items() if r["fields"] and (not my_shelf or r["shelf"] == my_shelf)}
        if len(kin_rows) < 3:
            return result
        field_count: dict[str, int] = {}
        table: dict[str, list[tuple[str, str, list[str]]]] = {}      # field name -> (listing, element, element words)
        for i, r in kin_rows.items():
            for name, val in r["fields"].items():
                if not str(val).strip() or rubbish_slot(name):
                    continue
                field_count[name] = field_count.get(name, 0) + 1
                val_parts = crosscheck_portions(val)
                if len(val_parts) > 1:
                    val_parts.append(str(val).strip().lower())      # (the full list as well: 'a, b' may be wanted together)
                table.setdefault(name, []).extend((i, p, crosscheck_terms(p)) for p in val_parts)
        line_out, rule_out, fail_out = result["lines"], result["rules"], result["fails"]

        def short(qtext: Any, n: int = 70) -> str:
            squashed = " ".join(str(qtext).split())
            if len(squashed) <= n:
                return squashed
            return squashed[: n - 3] + "..."

        def by_count(keys: Any) -> list[str]:
            return sorted(keys, key=field_count.get, reverse=True)

        def holds(r: dict[str, Any], keys: list[str]) -> bool:
            return any(str(r["fields"].get(x, "")).strip() for x in keys)

        def bound_check(topic: str, pname: str, req_val: str) -> None:
            # (a limit carrying a figure is compared as an amount. v605f FG: when 2+ read listings of the category have a
            # field named for the property, a listing without any field sharing a word of that name does not show it -
            # 'capacity' for a volume; 'calories' for calories per serving may be the same property under a shorter name)
            j = self.ware_belonging(topic) if multi else None
            own_w = set(trait_terms(pname))
            bound_keys = [x for x in field_count if field_count[x] >= 2 and rigorous_trait(x, pname)]
            kin_bound = {w for w in own_w if len(w) >= 3 and w not in BOND_TERMS}
            if not pname or not kin_bound or not bound_keys:
                return
            if multi and j is not None and j != idx:
                return

            def shows(x: str, v: Any) -> Any:
                return str(v).strip() and (rigorous_trait(x, pname) or set(trait_terms(x)) & kin_bound)
            if any(shows(x, v) for x, v in self_row["fields"].items()):
                return
            bound_txt = " / ".join(f"'{x}'" for x in bound_keys[:3])
            on_b = sum(1 for r in kin_rows.values() if holds(r, bound_keys))
            line_out.append(f"- {pname} = '{short(req_val)}': the store keeps it in the field {bound_txt}, present on {on_b} of "
                         f"{len(kin_rows)} read listings of this category. This listing lacks that field, so the limit is not "
                         "shown to be met: it does NOT meet it (a figure from some other field cannot replace it).")
            fail_out.append(f"{pname}: it lacks the {bound_keys[0]} field that this store's listings of the category carry, so "
                         f"'{short(req_val)}' is not shown to be met")

        def judge(demand: Any) -> None:
            topic, _, raw = str(demand).partition(":")
            req_val = " ".join(re.sub(r"\((?:stated|revealed[^)]*)\)", " ", raw).split())
            pname = re.split(r"\s*\(", topic)[0].strip()
            if not pname or not req_val:
                return
            lowered = req_val.lower()
            if reference_passage(req_val) or re.search(r"as shown|same as|the page|sold by|redirect|refer", lowered):
                return
            if re.search(r"\d", req_val) and re.search(CROSSCHECK_BOUNDARY, lowered):
                bound_check(topic, pname, req_val)
                return
            if re.fullmatch(r"\s*\d+(?:\.\d+)?\s+[a-z]+\s*", lowered):
                return        # (a number with its unit, '15 minutes', is compared as an amount rather than by words)
            j = self.ware_belonging(topic) if multi else None
            if multi and j is not None and j != idx:
                return
            affirm = affirmative_setting(req_val, topic)
            if not affirm and not re.search(OMIT_TERMS, lowered):
                return        # (a description: no field carries it)
            value_alts = crosscheck_variants(req_val)
            if not value_alts:
                return
            cited = by_count(x for x in field_count if rigorous_trait(x, pname))
            named_field = bool(cited)
            if named_field and not affirm:
                squashed = " ".join(re.findall(r"[a-z0-9]+", lowered))
                if any(" ".join(toks) == squashed for x in cited for _, _, toks in table.get(x, [])):
                    affirm = True          # (CJ2: the store writes this very value in the property's field: 'not available')
                    VERBATIM_SETTINGS.add(squashed)
            if not cited:
                if not affirm or (multi and j is None):
                    return
                found = [x for x in field_count if field_count[x] >= 2 and any(crosscheck_correspond(value_alts, toks) for _, _, toks in table.get(x, []))]
                if not found:
                    return
                top = max(found, key=field_count.get)
                cited = by_count(x for x in field_count if rigorous_trait(x, top))
            entries = [(i, toks) for x in cited for i, _, toks in table.get(x, [])]
            carriers = sum(1 for r in kin_rows.values() if holds(r, cited))
            writer_ids = sorted({i for i, toks in entries if i != lid and crosscheck_correspond(value_alts, toks)})
            solo_vals = {" ".join(toks) for i, toks in entries if i != lid}
            examples = list(dict.fromkeys(short(r["fields"][x], 40) for r in kin_rows.values() for x in cited
                                         if str(r["fields"].get(x, "")).strip()))[:6]
            field_txt = " / ".join(f"'{x}'" for x in cited[:3])
            seen_vals = " | ".join(examples)
            tail = f"{len(writer_ids)} other listing(s) write '{short(req_val)}' itself there."
            if not affirm:
                # (v605d M9: for an exclusion, count the listings that write the excluded value, its negation word removed)
                neg_val = " ".join(re.sub(OMIT_TERMS, " ", lowered).split())
                neg_alts = crosscheck_variants(neg_val)
                if neg_alts:
                    retained = {i for i, toks in entries if i != lid and crosscheck_correspond(neg_alts, toks)}
                    tail = (f"{len(retained)} other listing(s) write '{short(neg_val)}' there, the excluded value (a listing "
                            "writing it does not meet the requirement).")
            base = (f"- {pname} = '{short(req_val)}': the store keeps it in the field {field_txt}"
                    + (f" (no field is called '{pname}'; this field's values write the value)" if not named_field else "")
                    + f", present on {carriers} of {len(kin_rows)} read listings of this category; " + tail)
            own_kinds = category_terms(req_val)
            # (when no read listing writes it, category listings whose titles or texts name it reveal the store's own word)
            title_namers = []
            if not writer_ids and own_kinds:
                title_namers = [i for i, r in row_map.items()
                          if i != lid and (not my_shelf or r["shelf"] == my_shelf) and own_kinds <= r["words"]][:3]
            if not affirm:
                rule_out.append(base)
            elif writer_ids:
                rule = base + (" Hence only a value that writes those words meets it (containment and a single list "
                               "element both count; a synonym or related word does not).")
                if not named_field:
                    rule += (" Since no field is named for the property, a listing whose title writes the value meets it "
                             "as well, unless one of its own fields sharing the property's name says otherwise.")
                rule_out.append(rule)
            elif title_namers:
                rule_out.append(base + " Values seen: " + seen_vals + f"; listings of this category name '{short(req_val)}' in "
                             f"their title or text ({', '.join(title_namers)}), so the store has a word of its own for it: a "
                             "synonym or related value meets it only once such a listing's record shows this field "
                             "writing it differently.")
            elif carriers < CROSSCHECK_EQUIVALENT_LOWER:
                # (v605e F3: a field few read listings carry cannot yet show how this store writes the value)
                rule_out.append(base + " Values seen: " + seen_vals + f"; none of them writes it, but too few read listings "
                             f"carry this field to tell whether the store writes '{short(req_val)}' itself: a value with the "
                             "same meaning is not yet shown to meet it.")
            else:
                rule_out.append(base + " Values seen: " + seen_vals + "; none of them writes it, so a value with the same "
                             "meaning may meet it.")
            mine = [(x, str(self_row["fields"][x])) for x in cited if str(self_row["fields"].get(x, "")).strip()]
            shown = "; ".join(f"{x}: '{short(v, 90)}'" for x, v in mine)
            # (EA2: a key of this listing equal to the property's words after one extra leading word - a prefixed length
            # key for length - is that property's field here: while it writes or touches the value there is no plain
            # failure, and the item check decides)
            own_words = trait_terms(pname)
            qualified = [(x, str(v)) for x, v in self_row["fields"].items() if own_words and x not in cited and str(v).strip()
                         and not rubbish_slot(x) and len(trait_terms(x)) == len(own_words) + 1 and trait_terms(x)[1:] == own_words]
            qual_writes = [(x, v) for x, v in qualified if crosscheck_inscribes(req_val, v)]
            qual_near = bool(qual_writes) or any(crosscheck_contacts(req_val, v) for _, v in qualified)
            qual_note = ""
            if qual_writes:      # (v605f FB: this settles it, rather than 'weigh both fields')
                qual_note = (f" Its {qual_writes[0][0]} '{short(qual_writes[0][1], 60)}' writes it (a qualified key of "
                             "the same property: a field named for the property with one more leading word is that "
                             "property's field under a fuller name), so it meets it.")
            if self.crosscheck_rerouted(lid, pname):
                line_out.append(base + " This listing's own notes might route this property to another listing, so its "
                             "own field may not settle it.")
                return
            if not affirm:
                line_out.append(base + " This listing: " + (shown or "no such field") + " (an exclusion: the listing must not "
                             "carry the excluded value).")
                return
            # (careful: never over a field of the listing that shares the property's name without writing the value -
            # 'waist' for a waist type may be the property's own field under a shorter name, and the title yields to it)
            kin_words = {w for w in trait_terms(pname) if len(w) >= 3 and w not in BOND_TERMS}
            kin = [x for x, v in self_row["fields"].items() if str(v).strip() and not rubbish_slot(x) and set(trait_terms(x)) & kin_words
                   and not crosscheck_inscribes(req_val, v)]
            title_writes = not named_field and not kin and crosscheck_inscribes(req_val, self_row.get("title") or "")
            if not mine:
                if qual_writes:
                    # (FB: nothing under the plain name, and the prefixed field writes the value)
                    line_out.append(base + " This listing has no such field." + qual_note)
                elif title_writes:
                    # (v605e F2: no field is named for the property and the listing has no stand-in field: its title counts)
                    line_out.append(base + " This listing has no such field, and no field is named for the property: its "
                                 f"title writes '{short(req_val)}': it meets it.")
                else:
                    pw = {w for w in trait_terms(pname) if len(w) >= 3}
                    related = sorted(((x, v) for x, v in self_row["fields"].items()
                                      if x not in cited and str(v).strip() and not rubbish_slot(x)
                                      and (set(trait_terms(x)) & pw or crosscheck_inscribes(req_val, v))),
                                     ckey=lambda kv: not crosscheck_inscribes(req_val, kv[1]))[:4]
                    listing = ""
                    if related:
                        listing = (": fields of it that may name the property: "
                                   + "; ".join(f"{x}: '{short(v, 60)}'" for x, v in related))
                    any_writes = any(str(v).strip() and crosscheck_inscribes(req_val, v) for v in self_row["fields"].values())
                    if named_field and not related and not any_writes and carriers >= 3:
                        # (the category keeps the property in a field of its own name and this record has no field touching
                        # it at all: a title or blurb naming the value does not stand in for the field the grader reads)
                        line_out.append(base + " This listing has no such field and none of its fields (options included) "
                                     "writes the value, so its title or text cannot stand in for it: it does NOT meet it.")
                        fail_out.append(f"{pname}: it lacks the {cited[0]} field that {carriers} read listings of its category "
                                     f"carry, so '{short(req_val)}' is not shown to be met")
                        return
                    line_out.append(base + " This listing has no such field" + listing + ". Another name of the SAME "
                                 "property counts; a field of a different property (a height for a type, a frame material "
                                 "for the material) does not stand in.")
                return
            elements = [p for _, v in mine for p in crosscheck_portions(v)] + [v.strip().lower() for _, v in mine if len(crosscheck_portions(v)) > 1]

            def split_off(elem: str) -> list[str]:
                halves = crosscheck_flanks(elem)
                if not halves or not any(crosscheck_correspond(value_alts, crosscheck_terms(s)) for s in halves):
                    return []
                return [s for s in halves if not crosscheck_correspond(value_alts, crosscheck_terms(s)) and " ".join(crosscheck_terms(s)) in solo_vals]
            plain_hits: list[str] = []
            paired: list[tuple[str, list[str]]] = []
            for p in elements:
                if not crosscheck_correspond(value_alts, crosscheck_terms(p)):
                    continue
                off = split_off(p)
                if off:
                    paired.append((p, off))
                else:
                    plain_hits.append(p)
            if plain_hits:
                line_out.append(base + f" This listing: {shown} - it writes '{short(req_val)}': it meets it.")
            elif qual_writes:
                # (FB: the prefixed field writes the value - met regardless of the plainly named field)
                line_out.append(base + f" This listing: {shown}." + qual_note)
            elif title_writes:
                # (v605e F2: the field stands in for the property only through its values - none is named for it - so
                # where it does not write the value, the title decides, per READ_RULES' title rule)
                line_out.append(base + f" This listing: {shown} - as no field is named for the property, its title writes "
                             f"'{short(req_val)}': it meets it.")
            elif paired:
                elem, others = paired[0]
                verdict = ("this field does not meet it." + qual_note) if qual_writes else "it does NOT meet it."
                line_out.append(base + f" This listing: {shown} - '{short(elem)}' pairs '{short(req_val)}' with '{others[0]}', "
                             "which other listings carry on its own in this field: " + verdict)
                if named_field and not qual_near:
                    fail_out.append(f"{pname}: its {cited[0]} is '{short(elem)}', pairing {short(req_val)} with {others[0]} (a "
                                 f"value this store also writes on its own there), so it is not {short(req_val)}")
            elif writer_ids:
                verdict = ("this field does not meet it." + qual_note) if qual_writes else\
                    "it does NOT meet it (where the store writes the value itself, a synonym or related word does not count)."
                line_out.append(base + f" This listing: {shown} - it does not write '{short(req_val)}', which other listings "
                             "of this category write in this field: " + verdict)
                if named_field and not qual_near and not any(crosscheck_contacts(req_val, v) for _, v in mine):
                    fail_out.append(f"{pname}: its {cited[0]} is '{short(mine[0][1], 60)}', whereas this store writes "
                                 f"'{short(req_val)}' itself in that field on other listings")
            elif title_namers:
                line_out.append(base + f" This listing: {shown} - it does not write '{short(req_val)}'. No read listing writes "
                             f"it in this field so far (its values: {seen_vals}), yet listings of this category name "
                             f"'{short(req_val)}' in their title or text ({', '.join(title_namers)}): the store has a word of its "
                             "own for it, so this listing's different value is not shown to meet it.")
            elif carriers < CROSSCHECK_EQUIVALENT_LOWER:
                # (v605e F3)
                line_out.append(base + f" This listing: {shown}. No read listing of this category writes '{short(req_val)}' "
                             f"in this field (its values: {seen_vals}), but only {carriers} read listing(s) carry it: too few "
                             f"to tell whether the store writes '{short(req_val)}' itself, so a value with the same meaning "
                             "is not yet shown to meet it.")
            else:
                line_out.append(base + f" This listing: {shown}. No read listing of this category writes '{short(req_val)}' "
                             f"in this field (its values: {seen_vals}), so a value with the same meaning may meet it.")

        for demand in self.needs.traits_remaining()[:12]:
            judge(demand)
        return result

    def standing_assess(self, lid: str) -> str:
        """Under a cheapest or best request: listings in the records that might fit and beat the chosen one."""
        prio = self.needs.priority()
        ckey = f"{lid}|{self.needs.iteration}|{self.rejection_num}"
        if prio == "none" or ckey in self.standing_assessed:
            return ""
        if self.protocol.surplus_alt() < WRAPUP + 30 or self.shifts_remaining() < 4:
            return ""
        self.standing_assessed.add(ckey)
        prio_detail = self.needs.perusal.get("priority_detail") or ""
        cheapest = prio == "cheapest"
        if cheapest:
            what = "cheapest (lowest price)"
        elif prio_detail:
            what = f"best ({prio_detail})"
        else:
            what = "best"
        chunks = [STANDING_ASSESS.replace("PRIORITY", what), "\n\nTHE SHOPPER'S REQUEST:\n", self.request, self.needs.alterations(),
                  "\n\nCURRENT REQUIREMENTS:\n", self.needs.passage_payload(), self.reference_annotation()]
        if self.needs.items():
            chunks.append("\n\nSeveral products are requested: only listings of the SAME product (category) as the chosen "
                          "item compete with it; listings of the other products can never beat it.")
        legend = ("one card each; DETAILS = full record read, ROW = only seen in search rows, fields unknown: a ROW in the "
                  "requested category (whatever its title says the object is) that beats the chosen item and is not ruled "
                  "out by its price or stock MAY fit and gets named")
        if cheapest:
            legend += "; only listings cheaper than the chosen one appear"
        profiles = self.vault.profiles(20000, lid if cheapest else "", self.reference_external(), {lid})
        chunks.append(f"\n\nTHE CHOSEN ITEM: {lid}\n\nLISTINGS SEEN ({legend}):\n" + "\n".join(profiles))
        chunks.append(self.attribute_monikers())
        rule_out = self.attribute_snapshot(lid)["rules"]
        if rule_out:
            chunks.append("\n\nFIELD EVIDENCE (worked out by code from this store's records of the chosen item's category: "
                          "how the store writes each required value):\n" + "\n".join(rule_out))
        reply = self.protocol.confer("".join(chunks), 2500, ASSESSOR)
        self.journal.append({"kind": "rank", "result": reply})
        # (a line that sat in the cart when a checkout was refused counts as gone: never put forward as better)
        dead_set = self.defunct_lineitems()
        proposed = [str(x) for x in (reply or {}).get("better") or [] if isinstance(x, (str, int))]
        beats = [x for x in proposed if x != lid and x not in self.rejected_lineitems and x not in dead_set][:4]
        beats = [x for x in beats if not self.attribute_snapshot(x)["fails"]]     # (CH2: the code's own field evidence rules it out)
        self.rank_better = list(beats)       # (v605f FD)
        self.__dict__.setdefault("named", set()).update(beats)
        if not beats:
            return ""
        why = f" ({reply.get('reason')})" if reply.get("reason") else ""
        return (f"A rank check suspects these listings could also fit and beat {lid} on the {what}: " + ", ".join(beats)
                + why + ". Check each of them (details, stock, price) before ordering; if they all fail, order the current pick.")

    def scrutiny_abstain(self, early: bool = False) -> str:
        """The audit before declining; an early one (nothing verified, late in the episode) does not count toward the two
        audits a decline may set off."""
        again = len(self.vault.read) > self.__dict__.get("audit_reads", 10 ** 9) and self.__dict__.get("audits", 0) < 2
        if self.audit_done and not again:
            return ""
        if self.protocol.surplus_alt() < WRAPUP + 30:
            return ""
        self.audit_done = True
        if not early:
            self.audits = self.__dict__.get("audits", 0) + 1
        foreign = self.reference_external()
        prior = self.__dict__.get("audit_ids")
        focus = (set(self.vault.read) - set(prior or ())) if prior is not None else ()
        profiles = self.vault.profiles(16000, "", foreign, focus)
        self.audit_reads = len(self.vault.read)
        if not profiles:
            return ""
        fresh = [] if prior is None else sorted(self.vault.read - prior)
        self.audit_ids = set(self.vault.read)
        chunks = [PRIOR_ABSTAIN, "\n\nTHE SHOPPER'S REQUEST:\n", self.request, self.needs.alterations(),
                  "\n\nCURRENT REQUIREMENTS:\n", self.needs.passage_payload(), self.reference_annotation(),
                  "\n\nLISTINGS SEEN (one card each; DETAILS = full record read, ROW = only seen in search rows):\n",
                  "\n".join(profiles)]
        if fresh:
            chunks.append("\n\nOPENED SINCE THE LAST CHECK (their DETAILS cards above are new: judge each of them first, "
                          "field by field, under the reading rules): " + ", ".join(fresh))
        chunks.append(self.attribute_monikers())
        reply = self.protocol.confer("".join(chunks), 6000, INSPECTOR)
        self.journal.append({"kind": "audit", "result": reply})
        got = reply or {}
        fits_found = [str(x) for x in got.get("possible_fits") or [] if isinstance(x, (str, int))][:5]
        fits_found = [x for x in fits_found if not self.attribute_snapshot(x)["fails"]]     # (CH2)
        self.audit_fits = fits_found
        unopened_ids = [str(x) for x in got.get("unopened") or [] if isinstance(x, (str, int))
                    and str(x) not in self.vault.read and str(x) in self.vault.ids][:6]
        shelf_set = self.rack_tags()
        budget, units_wanted = self.needs.outlay_lid, self.needs.quantity()
        shelf_of_hit = self.__dict__.get("hit_shelf", {})
        scores = self.__dict__.get("hit_score", {})

        def admissible(i: str) -> bool:
            if i in self.vault.read:
                return False
            if shelf_set and shelf_of_hit.get(i) and shelf_of_hit.get(i) not in shelf_set:
                return False
            return not (whether_numeric(budget) and whether_numeric(self.vault.price(i)) and self.vault.price(i) * units_wanted > budget)
        hits = [i for i in sorted(self.__dict__.get("all_hits", []), key=lambda i: scores.get(i, 0), reverse=True)
                if admissible(i)]
        try:
            slack_ids = [i for i in self.margin_tablerows() if i not in self.vault.read]      # (ED5)
        except Exception:
            slack_ids = []
        unopened_ids = list(dict.fromkeys(slack_ids + hits[:4] + [i for i in unopened_ids if i not in hits]))[:6]
        page_calls = self.examine_sheets(2) if self.shifts_remaining() > 5 else []
        if page_calls:
            unopened_ids = unopened_ids[:4]
            self.__dict__.setdefault("named_pages", set()).update(json.dumps(p, sort_keys=True, default=str) for p in page_calls)
        try:
            combo_calls = self.unified_searchterms() if self.shifts_remaining() > 5 else []
        except Exception as exc:
            self.journal.append({"kind": "combo_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
            combo_calls = []
        if combo_calls:
            self.__dict__.setdefault("named_pages", set()).update(json.dumps(p, sort_keys=True, default=str) for p in combo_calls)
        self.__dict__.setdefault("named", set()).update(fits_found + unopened_ids)
        cited_list = self.__dict__.setdefault("named_list", [])
        cited_list += [i for i in fits_found + unopened_ids if i not in cited_list]
        if not (fits_found or unopened_ids or page_calls or combo_calls):
            return ""

        def calls(found: list[Any]) -> str:
            return "; ".join(f"{self.role('search')} {json.dumps(p, ensure_ascii=False)}" for p in found)
        said = []
        if fits_found:
            why = f" ({reply.get('reason')})" if reply.get("reason") else ""
            said.append("A separate check believes these listings may meet every requirement: " + ", ".join(fits_found)
                        + why + ".")
        if page_calls:
            said.append("These searches include required values and matched more listings than were read, so suitable ones "
                        "may be on their unread pages: " + calls(page_calls) + ". Read those pages first (a single step "
                        "shows a full page of rows) and open the in-budget rows of the requested kind they list.")
        if combo_calls:
            said.append("These searches join the required values and have never been run (when every word is required, "
                        "each returns only listings carrying all of them): " + calls(combo_calls) + ". Run them (the search "
                        "tool remains available for them) and open the in-budget rows of the requested kind they list.")
        if unopened_ids:
            said.append("These listings of the requested kind appeared only in search rows and were never opened, so their "
                        "fields are unchecked: " + ", ".join(unopened_ids) + ". Open their details first (several per reply).")
        return " ".join(said) + " Verify them (details, stock, price) and buy the right one; decline only when every one truly fails."

    def titled_unperused(self) -> list[str]:
        """Listings an all-words search or a check named that remain unread: seen, within budget at their last price, in
        the requested category when that is known, not in a refused cart; all-words hits first, then in naming order."""
        budget, units_wanted = self.needs.outlay_lid, self.needs.quantity()
        shelf_set, shelf_of_hit = self.rack_tags(), self.__dict__.get("hit_shelf", {})
        scores = self.__dict__.get("hit_score", {})
        ranked = sorted(self.__dict__.get("all_hits", []), key=lambda i: scores.get(i, 0), reverse=True)

        def wanted(rid: str) -> bool:
            cost = self.vault.price(rid)
            if rid in self.vault.read or rid in self.rejected_lineitems or rid not in self.vault.ids:
                return False
            if whether_numeric(budget) and whether_numeric(cost) and cost * units_wanted > budget:
                return False
            return not (shelf_set and shelf_of_hit.get(rid) and shelf_of_hit[rid] not in shelf_set)
        return [rid for rid in dict.fromkeys(ranked + list(self.__dict__.get("named_list", []))) if wanted(rid)]

    def particulars_invoke(self, rid: str) -> dict[str, Any] | None:
        """Arguments that open one listing: a past details call's arguments with this id substituted, otherwise the
        details tool's single id-like text argument."""
        tname = self.role("details")
        known = self.vault.ids
        for log_line in reversed(self.journal):
            if log_line.get("kind") != "env":
                continue
            for perform in log_line.get("actions") or []:
                if perform.get("name") != tname:
                    continue
                arg_map = perform.get("args")
                if not isinstance(arg_map, dict):
                    continue
                if any(isinstance(v, str) and v in known for v in arg_map.values()):
                    return {name: (rid if isinstance(v, str) and v in known else v) for name, v in arg_map.items()}
        schema = self.structures.get(tname or "") or {}
        fields = schema.get("properties")
        if not isinstance(fields, dict):
            fields = {}
        texts = [name for name, spec in fields.items()
                 if isinstance(spec, dict) and spec.get("type") == "string" and not spec.get("enum")]
        pick = next((name for name in texts if name in (schema.get("required") or [])), None)
        if not pick and len(texts) == 1:
            pick = texts[0]
        if not pick:
            return None
        return {pick: rid}

    def examine_peruses(self, pending_ids: list[str], say: str, batch: tuple[str, str, int] | None, reader: str | None = None) -> list[tuple[str, str]]:
        """A single scripted reply that reads the listings in `pending_ids` (as lists of ids when `batch` is given, else one
        call per id to `reader`, the details tool by default); returns its responses ([] when no call could be made)."""
        calls: list[dict[str, Any]] = []
        if batch:
            tname, arg_name, per_call = batch
            calls = [{arg_name: pending_ids[start: start + per_call]} for start in range(0, len(pending_ids), per_call)]
        else:
            tname = reader or self.role("details")
            if tname:
                for rid in pending_ids:
                    arg_map = self.retrieve_invoke(rid, tname)
                    if arg_map:
                        calls.append(arg_map)
        pulse = self.protocol.pulse
        planned = [{"id": f"open_{n}_{pulse}", "type": "function", "function": {"name": tname, "arguments": json.dumps(arg_map)}}
                     for n, arg_map in enumerate(calls)]
        if not planned:
            return []
        self.converse.append({"role": "assistant", "content": say, "tool_calls": planned})
        start = len(self.vault.items)
        self.covering = True
        try:
            replies = self.route_alt(planned, scripted=True)
        finally:
            self.covering = False
        self.last_reads = self.vault.items[start:]
        return replies

    def examine_titled(self) -> bool:
        """Runs the detail reads that code checks left queued, as one scripted reply (a store tool able to read several
        listings at once gets them as a list of ids, and its first use is checked). Unless the reads were re-reads or
        served the question menu, the decline audit then looks over the fresh records."""
        attrs, protocol = self.__dict__, self.protocol
        pending = attrs.pop("open_queue", None)
        from_put = False
        if not pending and attrs.get("put_queue") and not protocol.done and not protocol.crashed:
            # (CB4: with no other read waiting, the code checks the stock of whatever it has just added to the cart)
            pending, from_put = attrs.pop("put_queue"), True
            put_tool = attrs.pop("put_tool", None)
            self.open_recheck = True
            self.open_purpose = "cart"
            self.open_tool = put_tool
            self.open_say = "I check the stock of what I just added to the cart, since a listing can sell out right after it goes in."
        if protocol.done or protocol.crashed or not pending:
            return False
        ids = list(dict.fromkeys(map(str, pending)))
        recheck = attrs.pop("open_recheck", False)
        purpose = str(attrs.pop("open_purpose", "") or "")
        floor, reader = attrs.pop("open_floor", None), attrs.pop("open_tool", None)
        say = attrs.pop("open_say", "") or ("Before I order, I read the cart lines once more." if recheck
                                            else "Before I decline, I open the listings the checks pointed to.")
        for_cart = purpose == "cart"
        if purpose in ("menu", "quota", "walk") and self.role("order") and self.role("cart") and not protocol.silence:
            # (CF8: reads made for the question menu, for one product's quota or for the cheapest walk must never eat into the
            # steps held back for the order while there is a candidate - a line in the cart or a verified fit)
            try:
                held_back = INITIAL_PURCHASE_VIA if self.purchases_attempted != 0 else self.purchase_owed()
                room = self.shifts_remaining() - held_back - 1
                as_list = bool(self.bundle_instrument()) and len(ids) >= 2
                if room < (1 if as_list else len(ids)) and (self.handcart.measures or not self.unconfirmed()):
                    survivors = [] if as_list else ids[:max(0, room)]
                    self.journal.append({"kind": "reserve_skip", "purpose": purpose, "ids": ids[:6], "kept": survivors})
                    ids = survivors
            except Exception as exc:
                self.journal.append({"kind": "reserve_skip_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
        elif for_cart:
            # (CB3: a cart line shown by a stock-bearing record since the latest cart change is not read again; CB4: neither is
            # a listing queued by the code after putting it in that has since left the cart)
            fresh_needed, in_cart = set(self.outdated_lineitems()), self.handcart.measures
            ids = [i for i in ids if i in fresh_needed or not (from_put or i in in_cart)]
        if not ids:
            return False
        batch = self.bundle_instrument() if len(ids) > 1 else None
        if batch:
            shows_stock = ((attrs.get("tool_caps") or {}).get(batch[0]) or {}).get("stock")
            # (an unproven list tool is never tried first on the cart's own lines - those go one by one; and a stock re-read
            # only goes through a list read whose answers have shown stock)
            if (for_cart and not attrs.get("batch_proven")) or (reader and not shows_stock):
                batch = None
        replies = self.examine_peruses(ids, say, batch, reader)
        already_posted = False
        untested = batch and replies and not attrs.get("batch_proven")
        if untested and self.bundle_approved(attrs.get("last_reads") or [], ids):
            self.batch_proven = True
        elif untested:
            # (the list tool gave back no records for the listings: it is retired and the ids are read singly - at most as many
            # as this queue was worth before list reads existed (none for optional menu reads, 2 for quota and walk opens, 3
            # for the rest), and only while the steps reserved for the order, walk or decline stay free)
            listed_tool = batch[0]
            self.batch_spec = None
            self.journal.append({"tool": listed_tool, "kind": "batch_dropped"})
            limit = {"menu": 0, "quota": 2, "walk": 2}.get(purpose, 3)
            spare_steps = int(floor) if whether_numeric(floor) else (self.purchase_owed() if purpose in ("menu", "quota") else 3)
            count = min(limit, len(ids), self.shifts_remaining() - spare_steps - 1)
            if count > 0 and not protocol.done and (reader or self.role("details")):
                self.later_responses(replies)
                already_posted = True
                replies = self.examine_peruses(ids[:count], say, None, reader)
        if not replies:
            return already_posted
        audit = "" if recheck or protocol.surplus_alt() <= WRAPUP + 30 else self.scrutiny_abstain()
        if audit:
            last_id, last_text = replies[-1]
            replies[-1] = (last_id, last_text + "\n[review after the reads] " + audit)
        self.later_responses(replies)
        return True

    def final_inventory(self, ident: str) -> bool | None:
        """(v605f) the stock reported by the latest record that carries this listing's stock: True available, False gone,
        None when no record says."""
        states: list[bool] = []

        def note(rec: dict[str, Any]) -> None:
            found = self.inventory_condition(rec) if self.tablerow_flag(rec) == ident else None
            if found is not None:
                states.append(found)
        for item in self.vault.items:
            traverse_mappings(item, note)
        return states[-1] if states else None

    def standing_ascertain(self, thing: str) -> str:
        """(v605f FD) under a cheapest rule, once a rank check has named listings that beat `thing`: each of them that was
        already read, is not gone by its last read, was not refused, fits the budget for its units and shows no plain field
        failure is item-checked right away (costs no step; two at most), so a pass feeds the cheaper-fit holds. The result is
        cheaper_fit(thing)."""
        outcome = ""
        try:
            wants = self.needs
            if wants.priority() == "cheapest":
                iteration, budget, checked = str(wants.iteration), wants.outlay_lid, 0

                def worth(other: str) -> bool:
                    if other not in self.vault.read or f"{other}|{iteration}" in self.succeeded or other in self.rejected_lineitems\
                            or other in self.defunct_lineitems() or self.final_inventory(other) is False or self.attribute_snapshot(other)["fails"]:
                        return False
                    idx = self.ware_position(other) if wants.items() else None
                    measures = wants.items()[idx]["quantity"] if idx is not None and idx < len(wants.items()) else wants.quantity()
                    price = self.vault.price(other)
                    return whether_numeric(price) and not (whether_numeric(budget) and whether_numeric(measures) and price * measures > budget)
                for other in self.__dict__.get("rank_better") or []:
                    if checked >= 2 or self.protocol.surplus_alt() < WRAPUP + 30:
                        break
                    if worth(other):
                        checked += 1
                        self.assess_entry(other, "ORDER")
                outcome = self.lowercost_suit(thing)
        except Exception as exc:
            self.journal.append({"kind": "rank_verify_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
            outcome = ""
        return outcome


class RacksAndWares:
    """Errand component: each product's home categories, joined product names, and merging two products into one."""

    def lowercost_suit(self, thing: str) -> str:
        """With a cheapest rule in force: the lowest-priced listing that passed the item check under today's requirements
        and costs less than the chosen one (lines of a refused cart count as gone)."""
        if self.needs.priority() != "cheapest" or not thing or (own_cost := self.vault.price(thing)) is None:
            return ""
        shelf, iteration, gone = self.rack_familiar(thing), str(self.needs.iteration), self.defunct_lineitems()
        best_id, best_cost = "", None
        for passed_key in self.succeeded:
            ident, _, ver = passed_key.rpartition("|")
            viable = ver == iteration and ident != thing and ident not in self.rejected_lineitems and ident not in gone
            if viable and self.needs.items():
                # (with several products only a listing of the same product may stand in for this one)
                viable = bool(shelf) and self.rack_familiar(ident) == shelf
            price = self.vault.price(ident) if viable else None
            cheaper = price is not None and price < own_cost and (best_cost is None or price < best_cost)
            # (v605d M3, CH2: a record read after its check that shows its own field plainly failing rules it out)
            if cheaper and not self.attribute_snapshot(ident)["fails"]:
                best_id, best_cost = ident, price
        return best_id

    def rack_familiar(self, thing: str) -> str:
        """A listing's category label, taken from any record carrying its id ('' when none is known)."""
        label = ""

        def look(rec: dict[str, Any]) -> None:
            nonlocal label
            if any(isinstance(v, str) and v == thing and "id" in str(k).lower() for k, v in rec.items()):
                label = self.rack_belonging(rec) or label
        for item in self.vault.items:
            traverse_mappings(item, look)
        fallback = (self.vault.__dict__.get("label_of") or {}).get(thing)     # (CC)
        return label or str(fallback or "").strip().lower()

    def selected(self) -> str | None:
        """The line an order would buy: the single cart line, or else the most recently added one."""
        lines = list(self.handcart.measures)
        return lines[-1] if lines else None

    # ------------------------------------------------------------------------------------- products and their categories
    def landing_racks(self) -> list[set[str]]:
        """Per requested product (a single entry for a one-product request): the category labels on listings (rows or
        records) whose kind words match that product's best (a word shared by several products counts half); an empty set
        when nothing matches."""
        size = len(self.vault.items)
        memoized = self.__dict__.get("home_memo")
        if memoized and memoized[0] == size:
            return [set(h) for h in memoized[1]]
        try:
            wanted = self.needs.items() or [{"product": str(self.needs.perusal.get("product") or "")}]
            labels: set[str] = set()

            def gather(rec: dict[str, Any]) -> None:
                shelf = self.rack_belonging(rec)
                if shelf and any(isinstance(v, str) and v in self.vault.ids and "id" in str(k).lower() for k, v in rec.items()):
                    labels.add(shelf)
            for item in self.vault.items:
                traverse_mappings(item, gather)
            token_sets = [set(category_lexemes(w["product"])) for w in wanted]
            by_product: list[set[str]] = []
            for pos, mine in enumerate(token_sets):
                others = set().union(*(t for q, t in enumerate(token_sets) if q != pos))
                picked: set[str] = set()
                best_score = 0.0
                for shelf in sorted(labels):
                    score = category_rating(mine, set(category_lexemes(shelf)), others)      # (CC: 'battery' matches 'batteries')
                    if score > best_score:
                        picked, best_score = {shelf}, score
                    elif score > 0 and score == best_score:
                        picked.add(shelf)
                by_product.append(picked)
        except Exception:
            by_product = [set() for _row in self.needs.items() or [0]]
        self.home_memo = (size, [set(h) for h in by_product])
        return by_product

    def native_rack(self) -> str:
        """One-product request only: the product's own category label when listings from some other category were read as
        well (a referenced listing, a stray row); '' when only its own category was read or no label matches."""
        try:
            if not self.needs.items():
                own = self.landing_racks()[0]
                labels = [s.strip().lower() for s in self.vault.racks_peruse()]
                if own and any(s not in own for s in labels):
                    hits = [s for s in labels if s in own]
                    return hits[0] if hits else min(own)
        except Exception:
            pass
        return ""

    # ------------------------------------------------------------------------------- CA: categories that join two kinds
    def fused_expression(self, a: str, b: str, label: str) -> tuple[str, bool] | None:
        """CA: (how the request itself names two products that one store category joins, whether that is the category's
        own name). First the category's name in order, with either joiner ('women's flats and heels'); failing that the
        two names linked by 'and', '&', 'or', '/', '+' or a comma within four words and with no figure between ('flats and
        heels'). The looser form applies only when the two names end in different kinds (a men's and a women's watch, an
        indoor and an outdoor light are two products of one kind) and the request does not name both again outside the
        joined phrase ('a table and a chair: the table in oak, the chair with armrests' describes two). None when neither
        form is written (two products named separately, each with its own quantity, stay two)."""
        text = re.sub(r"\s+", " ", self.request.lower().replace("’", "'"))
        sides = [p for p in re.split(CONNECTORS, str(label).lower()) if p.strip()]
        if len(sides) == 2:
            first, second = (moniker_motif(s) for s in sides)
            whole = first and second and re.search(r"\b" + first + r"\s*(?:\band\b|&|\bor\b|/|\+)\s*" + second, text)
            if whole:
                return whole.group(0).strip(), True
        ka, kb = category_lexemes(a), category_lexemes(b)
        if ka and kb and not identical_category(ka[-1], kb[-1]):
            for lead, tail in ((a, b), (b, a)):
                p_lead, p_tail = moniker_motif(lead), moniker_motif(tail)
                spans = re.finditer(r"\b(?:" + p_lead + r")(?P<mid>[^.;:!?]{0,40}?)(?:" + p_tail + r")", text)\
                    if p_lead and p_tail else ()
                for m in spans:
                    gap = m.group("mid")
                    tight = len(re.findall(r"[a-z0-9]+", gap)) <= 3 and re.search(r"\band\b|&|\bor\b|,|/|\+", gap)\
                        and not re.search(r"\d|\bx\b", gap)
                    outside = text[:m.start()] + " | " + text[m.end():]
                    # (when both are named again separately, the request means two products)
                    if tight and not all(re.search(r"\b" + moniker_motif(z), outside) for z in (a, b)):
                        return m.group(0).strip(), False
        return None

    def ware_enquiry(self, line: str, rows: list[dict[str, Any]]) -> bool:
        """CA: whether a logged question (questions_log) concerned one of the products - its property label carries a tag
        of a product's kind, or the question names a product."""
        m = re.match(r'-\s*(.*?): asked "(.*?)" -> ', str(line), re.S)
        label, question = (m.group(1), m.group(2)) if m else (str(line), str(line))
        product_kinds = [category_lexemes(r["product"]) for r in rows]
        tag_words = (category_lexemes(tag) for tag in re.findall(r"\(([^()]*)\)", label))
        if any(words and any(category_among(w, pk) for w in words for pk in product_kinds) for words in tag_words):
            return True
        lowered = question.lower().replace("’", "'")
        return any(bool(moniker_motif(r["product"])) and bool(re.search(r"\b" + moniker_motif(r["product"]), lowered))
                   for r in rows)

    @staticmethod
    def fused_conflict(a: dict[str, Any], b: dict[str, Any]) -> bool:
        """CA: the two products' requirements give one property two different values (so the request wrote each its own)."""
        def props(row: dict[str, Any]) -> list[tuple[str, str]]:
            split = [str(req).partition(":") for req in row.get("requirements") or []]
            return [(re.split(r"\s*\(", head)[0].strip(), " ".join(re.findall(r"[a-z0-9]+", tail.lower())))
                    for head, _, tail in split]
        theirs = props(b)
        return any(identical_trait(pa, pb) and va != vb for pa, va in props(a) for pb, vb in theirs)

    def merge_joined(self, obs: Any) -> str:
        """CA (F1): undoes a reading that split ONE store category, named as two kinds joined ('A and B', 'A & B', 'A/B',
        'A + B'), into two products. Each search's row labels are collected. As long as the cart was never used, no question
        about a product went out (product_question) and no shopper reply set or freed a requirement, products i and j are
        merged when a label splits at one joiner into sides A and B, both products' kind words lie within the label, the
        words one has and the other lacks fall on opposite sides (each product touching its own side), both quantities are
        equal, their requirements do not give one property two values, and the request writes the label's words in order
        or the two names joined (joined_phrase). Kind words compare by same_kind. Gives the remark for the actor ('' when
        no merge happened)."""
        labels = self.__dict__.setdefault("joined_seen", {})

        def collect(rec: dict[str, Any]) -> None:
            if any(isinstance(v, str) and "id" in str(k).lower() and re.fullmatch(UID_RESEMBLING, v) for k, v in rec.items()):
                name = next((v.strip() for k, v in rec.items() if isinstance(v, str) and 0 < len(v.strip()) <= 60
                             and re.search(r"categor|shelf|department|aisle", str(k).lower()) and "id" not in str(k).lower()), "")
                if name and len([p for p in re.split(CONNECTORS, name.lower()) if p.strip()]) == 2:
                    labels.setdefault(name.lower(), name)
        traverse_mappings(obs, collect)

        def splits_apart(ti: list[str], tj: list[str], x: list[str], y: list[str]) -> bool:
            only_i = [w for w in ti if not category_among(w, tj)]
            only_j = [w for w in tj if not category_among(w, ti)]
            each_own_side = any(category_among(w, x) for w in ti) and any(category_among(w, y) for w in tj)
            return all(category_among(w, x) and not category_among(w, y) for w in only_i)\
                and all(category_among(w, y) and not category_among(w, x) for w in only_j) and each_own_side

        def mergeable(rows: list[dict[str, Any]], i: int, j: int, left: list[str], right: list[str]) -> bool:
            ki, kj = category_lexemes(rows[i]["product"]), category_lexemes(rows[j]["product"])
            if not (ki and kj) or rows[i]["quantity"] != rows[j]["quantity"]:
                return False
            if (categories_amid(ki, kj) and categories_amid(kj, ki)) or not categories_amid(ki + kj, left + right):
                return False
            return (splits_apart(ki, kj, left, right) or splits_apart(ki, kj, right, left)) and not self.fused_conflict(rows[i], rows[j])

        def find_pair(rows: list[dict[str, Any]]) -> tuple[Any, ...] | None:
            for key in sorted(labels):
                halves = [category_lexemes(p) for p in re.split(CONNECTORS, key) if p.strip()]
                if len(halves) == 2 and all(halves):
                    pairs = [(i, j) for i in range(len(rows)) for j in range(i + 1, len(rows))]
                    for i, j in pairs:
                        phrase = mergeable(rows, i, j, halves[0], halves[1]) and\
                            self.fused_expression(rows[i]["product"], rows[j]["product"], labels[key])
                        if phrase:
                            return (i, j, labels[key], *phrase)
            return None

        notes: list[str] = []
        for _round in range(2):
            current = self.needs.items()
            blocked = len(current) < 2 or not labels or self.handcart.employed or self.handcart.measures or self.needs.uncovered\
                or self.needs.indifferent or any(self.ware_enquiry(q, current) for q in self.enquiries_journal)
            pair = None if blocked else find_pair(current)
            if pair is None:
                break
            notes.append(self.combine_entries(*pair))
        return " ".join(n for n in notes if n)

    def combine_entries(self, i: int, j: int, label: str, phrase: str, in_order: bool = True) -> str:
        """CA: turns products i and j of the reading into the single product `phrase` (merge_joined). Its requirements are
        the union of both; their "(product)" tags are rewritten everywhere (to the phrase while several products remain,
        removed once one is left); the per-product caches (home_memo, menu_memo, quota_reads, walked, keyed by product or
        index) are discarded and the requirement version moves on."""
        wants = self.needs
        perusal, listed = wants.perusal, wants.items()
        raw = [entry for entry in perusal.get("items") or [] if isinstance(entry, dict) and str(entry.get("product") or "").strip()]
        if not 0 <= i < j < len(listed) or len(raw) != len(listed):
            return ""
        first, second = listed[i], listed[j]
        both = category_lexemes(first["product"]) + category_lexemes(second["product"])
        label_tokens = category_lexemes(label)
        rest = [category_lexemes(r["product"]) for pos, r in enumerate(listed) if pos not in (i, j)]
        remaining = len(listed) - 1
        new_tag = phrase if remaining > 1 else ""

        def ours(tag: Any) -> bool:
            toks = category_lexemes(tag)
            return bool(toks) and categories_amid(toks, label_tokens) and any(category_among(w, both) for w in toks)\
                and not any(categories_amid(toks, o) and categories_amid(o, toks) for o in rest)

        def swap(m: Any) -> str:
            if ours(m.group(1)):
                return f" ({new_tag})" if new_tag else ""
            return m.group(0)

        def retag(text: Any) -> Any:
            if isinstance(text, str):
                head, sep, tail = text.partition(":")
                return re.sub(r"\s*\(([^()]*)\)", swap, head) + sep + tail
            return text

        def dedup(seq: list[Any]) -> list[Any]:
            kept: list[Any] = []
            for x in seq:
                if x not in kept:
                    kept.append(x)
            return kept
        merged_reqs: list[str] = []
        for req in [*first["requirements"], *second["requirements"]]:
            prefix, _, value = str(req).partition(":")
            bare = re.sub(r"\s*\([^()]*\)", "", prefix).strip() + ":" + value
            if all(bare.lower() != r.lower() for r in merged_reqs):
                merged_reqs.append(bare)
        entry = {"product": phrase, "quantity": first["quantity"], "requirements": merged_reqs}
        survivors = [entry if pos == i else r for pos, r in enumerate(raw) if pos != j]
        for field in ("written", "uncovered", "let_go", "indifferent", "now_reqs"):
            current = getattr(wants, field)
            if current is not None:
                setattr(wants, field, dedup([retag(r) for r in current]))
        wants.resolved = [retag(item) for item in wants.__dict__.get("resolved") or []]
        for field in ("requirements", "hints", "references", "soft_wants"):
            if isinstance(perusal.get(field), list):
                perusal[field] = dedup([retag(r) for r in perusal[field]])
        targets = perusal.get("reference_targets")
        for tgt in targets if isinstance(targets, list) else []:
            if isinstance(tgt, dict) and ours(str(tgt.get("product") or "")):
                tgt["product"] = phrase
        for link in self.__dict__.get("refs") or []:
            for slot in ("req", "head", "now"):
                if isinstance(link.get(slot), str):
                    link[slot] = retag(link[slot])
            if ours(str(link.get("tag") or "")):
                link["tag"] = new_tag
        for attr in ("pending_seconds", "open_pair"):
            retained = self.__dict__.get(attr)
            if isinstance(retained, list):
                self.__dict__[attr] = [retag(p) for p in retained]
        if remaining <= 1:
            perusal.pop("items", None)
            perusal.update(product=phrase, quantity=first["quantity"])
        else:
            perusal.update(items=survivors, product=" and ".join(str(r.get("product")) for r in survivors))
        wants.iteration = wants.iteration + 1
        for cache in ("home_memo", "menu_memo", "quota_reads", "walked"):
            self.__dict__.pop(cache, None)
        product_count = len(wants.items())
        widen = 1.0 + 0.3 * min(2, max(0, product_count - 1))
        squeeze = max(0.5, min(1.0, 40.0 / max(1, self.stage_lid)))
        self.protocol.scale = widen * squeeze
        self.journal.append({"kind": "merge_joined", "label": label, "products": [first["product"], second["product"]],
                         "into": phrase, "label_words": in_order, "products_left": max(1, product_count)})
        lines_kept = "keeps one line for it" if remaining > 1 else "keeps a SINGLE line"
        return (f"The store sells '{label}' as a SINGLE category, and the request wants one product from it ('{phrase}'): the "
                f"order {lines_kept} rather than one line per kind its name combines. Requirements in force: "
                + wants.passage_payload())

    # ------------------------------------------------------------------------------------- CB: lookup tools and stock


class InventoryAndRejections:
    """Errand component: single-listing lookup tools, what their answers carry, gone cart lines and refused orders."""

    def retrieve_slot(self, tool: str | None) -> str | None:
        """CB: the single text argument of a one-listing lookup tool (its only required plain string without choices, else
        its only string); None for a tool needing other arguments or none (searches, carts, list reads, views)."""
        schema = self.structures.get(tool or "") or {}
        fields = schema.get("properties")
        if not isinstance(fields, dict):
            fields = {}
        required = schema.get("required")
        must = [k for k in required if k in fields] if isinstance(required, list) else []
        strings = [k for k, spec in fields.items() if isinstance(spec, dict) and spec.get("type") == "string" and not spec.get("enum")]
        key = next((k for k in strings if k in must), None)
        if not key and len(strings) == 1:
            key = strings[0]
        return None if not key or any(k != key for k in must) else key

    def retrieve_instruments(self) -> list[str]:
        """CB: the store's one-listing lookup tools - anything other than search, message, cart, cart view or a finisher."""
        busy = {self.role(r) for r in ("search", "message", "cart", "cart_view", "order", "decline")}
        busy.discard(None)
        return [name for name in self.structures if name not in busy and self.retrieve_slot(name)]

    def retrieve_invoke(self, ident: str, tool: str | None) -> dict[str, Any] | None:
        """CB: the arguments that read one listing with `tool` (details_call for the details tool): an earlier call's
        arguments with this listing's id put in place, else the tool's single id text argument."""
        known = self.vault.ids

        def reuse() -> dict[str, Any] | None:
            for entry in self.journal[::-1]:
                acts = (entry.get("actions") or []) if entry.get("kind") == "env" else []
                for perform in acts:
                    args = perform.get("args") if perform.get("name") == tool else None
                    if isinstance(args, dict) and any(isinstance(v, str) and v in known for v in args.values()):
                        return {k: ident if isinstance(v, str) and v in known else v for k, v in args.items()}
            return None
        if not tool:
            call = None
        elif tool == self.role("details"):
            call = self.particulars_invoke(ident)
        else:
            call = reuse()
            if call is None:
                only = self.retrieve_slot(tool)
                call = {only: ident} if only else None
        return call

    @staticmethod
    def dossier_uid(d: dict[str, Any]) -> str | None:
        """CB: the listing id of a record (a key called id, else one ending in id) when the value looks like an id."""
        named = [(str(k).lower(), v) for k, v in d.items()]
        hits = (v for strict in (True, False) for name, v in named
                if (name == "id" if strict else name.endswith("id")) and isinstance(v, str) and re.fullmatch(UID_RESEMBLING, v))
        return next(hits, None)

    @staticmethod
    def inquired_uids(call_args: Any) -> set[str]:
        """CB: every id-like text a call names (its text arguments, plus the texts inside its list arguments)."""
        found: set[str] = set()
        for v in (call_args if isinstance(call_args, dict) else {}).values():
            found.update(x for x in (v if isinstance(v, list) else [v]) if isinstance(x, str) and re.fullmatch(UID_RESEMBLING, x))
        return found

    def discover_lids(self, call_name: str, call_args: Any, obs: Any) -> list[str]:
        """CB: learns from use what each tool's answers carry (tool_caps): per tool, how many answers held a record with the
        named listing's id together with a stock field (stock_state), a property map (2+ scalar fields), notes, or a price.
        Notes are a long free text (8+ words) that the listing's own search row lacks or words differently (with no row of
        it seen: under a key no row has); their keys are kept per tool (note_keys), next to every key a tool's answers had
        (tool_keys). Each stock record counts as a row (for a listing the call did not name) or a lookup (one it named, or a
        cart reply). Rows are learned to show live stock (rows_live) once a row shows a listing's stock differing from an
        earlier row of it, and to lag (rows_lag) once a row and a lookup of one listing disagree with no stock-relevant cart
        change in between. dead: listings whose newest stock record has them gone - a lookup's verdict is undone only by a
        later lookup. Gives back the cart lines newly found gone."""
        named = self.inquired_uids(call_args)
        tool = str(call_name)
        attrs = self.__dict__
        tally = attrs.setdefault("tool_caps", {}).setdefault(tool, dict.fromkeys(("calls", "id", "stock", "props", "notes", "price"), 0))
        tally["calls"] = tally["calls"] + 1
        got = dict.fromkeys(("id", "stock", "props", "notes", "price"), False)
        gone_ids = attrs.setdefault("dead", set())
        gone_by = attrs.setdefault("dead_by", {})
        heard = attrs.setdefault("stock_by", {})
        row_text = attrs.setdefault("row_text", {})
        row_keys = attrs.setdefault("row_keys", set())
        tool_keys = attrs.setdefault("tool_keys", {}).setdefault(tool, set())
        note_keys = attrs.setdefault("note_keys", {}).setdefault(tool, set())
        cart_tools = {self.role("cart"), self.role("cart_view")} - {None}
        epoch = attrs.get("cart_seq", 0)
        lost_lines: list[str] = []

        def book_stock(ident: str, stock: bool, looked_up: bool) -> None:
            here, there = ("lookup", "row") if looked_up else ("row", "lookup")
            self.stock_seen = True
            book = heard.setdefault(ident, {})
            rival = book.get(there)
            if rival and rival[0] == epoch and rival[1] != stock:
                self.rows_lag = True
            if not looked_up and book.get("row") and book["row"][1] != stock:
                self.rows_live = True
            book[here] = (epoch, stock)
            if stock is not False:
                if looked_up or gone_by.get(ident) != "lookup":
                    gone_ids.discard(ident)
                return
            if ident in self.handcart.measures and ident not in gone_ids and ident not in lost_lines:
                lost_lines.append(ident)
            gone_ids.add(ident)
            gone_by[ident] = here

        def soakup(rec: dict[str, Any], ident: str) -> None:
            stock = self.inventory_condition(rec)
            if stock is not None:
                book_stock(ident, stock, ident in named or call_name in cart_tools)
            if ident not in named:
                if call_name not in cart_tools:
                    row_keys.update(map(str, rec))
                    row_text.setdefault(ident, {}).update({str(k): v for k, v in rec.items() if isinstance(v, str)})
                return
            got["id"] = True
            attrs.setdefault("read_with", {}).setdefault(tool, set()).add(ident)      # (CE3: which listings each tool read)
            # (CE3: keys that held text or nothing on a record the call named - where a notes field may live)
            attrs.setdefault("text_keys", {}).setdefault(tool, set()).update(
                str(k) for k, v in rec.items() if v is None or isinstance(v, str))
            got["stock"] = got["stock"] or stock is not None
            if not got["props"]:
                got["props"] = any(isinstance(v, dict) and sum(1 for x in v.values() if isinstance(x, (str, int, float))) >= 2
                                   for v in rec.values())
            if not got["price"]:
                got["price"] = any(whether_numeric(v) and re.search(r"price|cost", str(k).lower()) for k, v in rec.items())
            tool_keys.update(map(str, rec))
            own_row = row_text.get(ident)
            for k, v in rec.items():
                long_text = isinstance(v, str) and len(v.split()) >= 8
                if long_text and (str(k) not in row_keys if own_row is None else own_row.get(str(k)) != v):
                    note_keys.add(str(k))
                    got["notes"] = True

        def scan(rec: dict[str, Any]) -> None:
            ident = self.dossier_uid(rec)
            if ident:
                soakup(rec, ident)
        if not isinstance(obs, dict) or not obs.get("error"):
            traverse_mappings(obs, scan)
        for flag_name, flag in got.items():
            tally[flag_name] += int(flag)
        return lost_lines

    def inventory_instrument(self) -> str | None:
        """CB: the tool for re-reading a listing's stock: the one-listing lookup tool whose answers carried stock most often;
        before any has, the view tool, then the code's own guess of it from the tool descriptions, then the details tool -
        each only while it has not answered twice for a listing with no stock field; then a lookup tool that has not yet
        answered for any listing and that the guess gives no cart, message or finishing role (the read after a put tries
        it). None when no lookup tool could still show stock."""
        learned = self.__dict__.get("tool_caps") or {}
        candidates = self.retrieve_instruments()
        proven = [t for t in candidates if (learned.get(t) or {}).get("stock", 0) > 0]
        if proven:
            view, details = self.role("view"), self.role("details")
            return max(proven, key=lambda t: (learned[t]["stock"], t == view, t == details, t))
        if self.__dict__.get("role_guess") is None:
            try:
                fresh_guess = self.engine_duties()
            except Exception:
                fresh_guess = {}
            self.role_guess = fresh_guess
        guess = self.role_guess

        def silent_twice(t: str) -> bool:
            employed = learned.get(t) or {}
            return employed.get("id", 0) >= 2 and employed.get("stock", 0) == 0
        preferred = (self.role("view"), guess.get("view"), self.role("details"))
        pick = next((t for t in preferred if t and t in candidates and not silent_twice(t)), None)
        if pick:
            return pick
        claimed = {guess.get(r) for r in ("cart", "cart_view", "order", "decline", "message")} - {None}
        return next((t for t in candidates if t not in claimed and not (learned.get(t) or {}).get("id", 0)), None)

    def annotations_instrument(self) -> str | None:
        """CB: the lookup tool whose answers hold notes (learn_caps) under a key the details tool's answers never had;
        None when no tool has shown such notes (the details tool shows them itself, or none were seen)."""
        details_tool = self.role("details")
        detail_keys = set((self.__dict__.get("tool_keys") or {}).get(details_tool or "", set()))
        by_tool = self.__dict__.get("note_keys") or {}
        learned = self.__dict__.get("tool_caps") or {}
        shown = [t for t in self.retrieve_instruments() if t != details_tool and set(by_tool.get(t) or set()) - detail_keys]
        return max(shown, key=lambda t: ((learned.get(t) or {}).get("notes", 0), t)) if shown else None

    def defunct_lineitems(self) -> set[str]:
        """CB: the listings whose latest record with a stock field shows them gone."""
        return set(self.__dict__.get("dead") or ())

    def outdated_lineitems(self) -> list[str]:
        """CB: cart lines lacking a stock-bearing record (CB3) since the latest stock-relevant cart change."""
        latest = self.__dict__.get("cart_seq", 0)
        seq_of = (self.__dict__.get("line_seq") or {}).get
        return [i for i in self.handcart.measures if seq_of(i, -1) < latest]

    def dead_swap(self, line: str) -> tuple[str, str]:
        """CB: (listing, '') to put in place of a gone cart line - one that passed the item check under today's requirements,
        of the line's category (known to be of it when there are several products), not gone, refused or already in the
        cart, and keeping the cart total (other lines included) inside the budget; the cheapest first. Otherwise ('', rows):
        up to three of the cheapest in-stock rows of its category that keep the total inside the budget."""
        gone = self.defunct_lineitems()
        iteration = str(self.needs.iteration)
        home = self.rack_familiar(line)
        multi = len(self.needs.items()) >= 2
        budget = self.needs.outlay_lid
        measures = max(1.0, self.handcart.retained(line))
        endless = float("inf")

        def unit_price(ident: str) -> float:
            p = self.vault.price(ident)
            return float(p) if whether_numeric(p) else endless

        def taken(ident: str) -> bool:
            return ident == line or ident in gone or ident in self.rejected_lineitems or ident in self.handcart.measures

        others = 0
        for i in self.handcart.measures:
            if i != line and unit_price(i) != endless:
                others += unit_price(i) * self.handcart.retained(i)

        def over_budget(price: float) -> bool:
            return whether_numeric(budget) and price * measures + others > budget

        def passes_swap(ident: str) -> bool:
            # (v605d M4, CH2: a record read after its check that shows its own field plainly failing rules it out)
            if self.attribute_snapshot(ident)["fails"]:
                return False
            label = self.rack_familiar(ident)
            if home and (label != home if multi else label not in ("", home)):
                return False
            return unit_price(ident) == endless or not over_budget(unit_price(ident))
        split = (key.rpartition("|") for key in self.succeeded)
        verified = [ident for ident, _, ver in split if ver == iteration and not taken(ident) and passes_swap(ident)]
        if verified:
            return min(verified, key=lambda i: (unit_price(i), i)), ""
        if multi and not home:
            return ("", "")
        # (CF7: rows the item check failed, and read rows whose own record lacks or fails a stated value, cannot stand in;
        # read rows showing every stated value lead, unread rows follow, cheapest first within each)
        k = next((n for n, shelves in enumerate(self.landing_racks()) if home and home in shelves), None) if multi else None
        judged = [(prop, simple) for prop, simple in self.engine_settings(k) if self.elicited_simple(simple)]
        unsuccessful = {bits[0] for bits in (key.split("|") for key, val in self.verdicts_alt.items() if val) if bits[1:2] == [iteration]}
        pointed = self.reference_indicators()

        def row_rank(ident: str, row: dict[str, Any], price: float) -> int | None:
            if taken(ident) or row["stock"] is not True or (home and row["shelf"] != home):
                return None
            if price == endless or over_budget(price):
                return None
            # (v605d M4, CE5: a pointer, or a listing showing a value a differ reference rules out)
            if ident in unsuccessful or ident in pointed or self.reference_omitted(row["fields"], k):
                return None
            if ident not in self.vault.read:
                return 1
            # (v605d M4, CH2)
            if self.attribute_snapshot(ident)["fails"] or any(self.engine_conflict(row["fields"], prop, simple) for prop, simple in judged)\
                    or not all(self.engine_displays(row, prop, simple, set(), True) for prop, simple in judged):
                return None
            return 0
        ranked = []
        for ident, row in self.engine_tablerows().items():
            listed = row["price"]
            price = listed if whether_numeric(listed) else unit_price(ident)
            tier = row_rank(ident, row, price)
            if tier is not None:
                ranked.append((tier, price, ident))
        ranked.sort()
        unread_low = min((x for x in ranked if x[0] == 1), key=lambda x: (x[1], x[2]), default=None)
        if unread_low and self.needs.priority() == "cheapest" and all(unread_low[1] < p for r, p, _ in ranked if r == 0):
            # (F4: with a cheapest rule, the cheapest unread row leads when it undercuts every read row listed)
            ranked.remove(unread_low)
            ranked.insert(0, unread_low)
        top = ranked[:3]
        self.swap_unread = [i for r, _, i in top if r == 1]
        self.swap_page = None
        if not any(r == 0 for r, _, _ in ranked):
            more = None
            try:
                more = self.defunct_sheet(k)
            except Exception:
                pass
            self.swap_page = more

        def describe(r: int, p: Any, i: str) -> str:
            amount = int(p) if float(p).is_integer() else p
            extra = ", already read: its own record shows each stated value" if r == 0 else ""
            return f"{i} ({amount}{extra})"
        return "", ", ".join(describe(*x) for x in top)

    def defunct_sheet(self, k: int | None) -> dict[str, Any] | None:
        """CF7: the next unread page, as call arguments, of the search whose leading words carry a stated value of product
        k (the walk carrying most of them); None when no such search has a page left unread."""
        arg = self.sheet_param()
        book = self.__dict__.get("page_book", {}).values() if arg else ()
        open_walks = [w for w in book if self.traverse_conveys(w["args"], k) and self.subsequent_sheet(w) is not None]
        if not open_walks:
            return None
        pick = max(open_walks, key=lambda w: (len(self.traverse_conveys(w["args"], k)), -len(w["read"])))
        call = dict(pick["args"])
        call[arg[0]] = self.subsequent_sheet(pick)
        return call

    def substitute_examine(self) -> str:
        """CF7: the cheapest unread row named by the latest gone-line note, while its single reserved details read is still
        unused."""
        gone = self.defunct_lineitems()
        queue = [] if self.__dict__.get("swap_done") else self.__dict__.get("swap_unread") or []
        return next((i for i in queue if i not in self.vault.read and i not in gone and i not in self.handcart.measures), "")

    def defunct_annotation(self, lines: list[str]) -> str:
        """CB: the remark for cart lines shown gone, naming what should go in instead."""
        self.event_seen = True            # (CF1: this errand has now seen a gone record)
        notes: list[str] = []
        for gone_line in lines:
            self.swap_unread, self.swap_page = [], None
            choice, listing = self.dead_swap(gone_line)
            head = (f"{gone_line} is now OUT OF STOCK (its latest record says so), and an order that includes it will fail. "
                    "Remove it from the cart")
            cheapest = ", and it is the cheapest such listing" if choice and self.needs.priority() == "cheapest" else ""
            if choice:
                notes.append(f"{head} and add {choice} instead (the item check found it meets every requirement in force now"
                             f"{cheapest}); read its stock and price again, then order.")
            elif not listing:
                notes.append(f"{head} and look for another listing that fits.")
            else:
                notes.append(f"{head}. No other verified fit for its product is known, so look at the cheapest in-stock rows "
                             f"of its category that stay inside the budget ({listing}; read rows whose own record is missing "
                             "or fails a stated requirement were dropped, and an unread one needs its details read first); add "
                             "the first one that fits, read its stock and price again, then order.")
            extra_page = self.__dict__.get("swap_page")
            if extra_page and not choice:
                notes.append(f"Further rows of this kind: the next unread page of the search led by the stated value, "
                             f"{json.dumps(extra_page, default=str)}.")
        return " ".join(notes)

    def rejection_reperuse(self, obs: Any = None) -> str:
        """CF4: straight after a refused order the code itself reads the stock of the cart lines (with the stock tool, as the
        cart's re-read, in one scripted reply): a refusal often means some line went out of stock without the text naming
        it, and the next order needs that re-read anyway (CB3). Skipped while the code is finishing alone (muted) or another
        read is queued. ED3: for a refused one-line cart whose refusal text the listener now reads as an item becoming
        unavailable - the needs' version moved while no requirement, revealed need or budget changed - that line is marked
        gone with no re-read (a later lookup can still clear it); the listener's note leads the remark either way."""
        note, gone_remark = "", ""
        try:
            note, gone_remark = self.rejection_vanished(obs)
        except Exception as exc:
            self.journal.append({"kind": "refusal_gone_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
        return note + (gone_remark or self.rejection_backlog())

    def rejection_vanished(self, obs: Any) -> tuple[str, str]:
        """ED3: (the listener's note on the refusal text, the gone-line remark) for a refused one-line cart; ('', '') when
        the refusal has no text or the cart has several lines (some stores sell only the first one)."""
        retained = [i for i in self.handcart.measures if self.handcart.retained(i) > 0]
        quiet = obs is None or len(retained) != 1 or self.protocol.silence or self.__dict__.get("covering")\
            or self.__dict__.get("last_kind") != "order"
        said = [] if quiet else self.customer_lineitems(self.role("order") or "", obs, True)
        if not said:
            return ("", "")
        wants = self.needs

        def snapshot() -> tuple[Any, ...]:
            return wants.iteration, len(wants.let_go), len(wants.uncovered), str(wants.outlay_lid), "|".join(wants.traits_remaining())
        before = snapshot()
        remark = wants.receive(said, None)
        after = snapshot()
        heard = (" " + remark) if remark else ""
        if not remark or after[0] == before[0] or after[1:] != before[1:]:
            return (heard, "")
        only_line = retained[0]
        gone_set = self.__dict__.setdefault("dead", set())
        gone_set.add(only_line)
        verdicts = self.__dict__.setdefault("dead_by", {})
        verdicts[only_line] = "lookup"
        self.journal.append({"line": only_line, "kind": "refusal_dead"})
        return heard, (" The refusal reports that an item became unavailable while the cart held a single line, so the code "
                       "treats " + only_line + " as sold out (without a re-read): " + self.defunct_annotation([only_line]))

    def rejection_backlog(self) -> str:
        """CF4: queues the stock re-read of the cart lines after a refusal ('' when it cannot run)."""
        reader = self.inventory_instrument()
        cart_lines = [*self.handcart.measures][:4]
        blocked = not cart_lines or not reader or self.protocol.silence or self.__dict__.get("covering")\
            or self.__dict__.get("open_queue") or self.shifts_remaining() < len(cart_lines) + 3 or self.protocol.surplus_alt() < WRAPUP
        if blocked:
            return ""
        self.open_queue = cart_lines
        self.open_recheck = True
        self.open_purpose = "cart"
        self.open_tool = reader
        self.open_say = "The order was refused, so I read the stock of the cart lines again (a line may have sold out)."
        return " The stock of the cart lines is being read for you now (" + reader + ")."


class AbstainRepair:
    """Errand component: repairing a planned decline, budget slack rows and the near-basket meaning check."""

    def engine_evaluates_whole(self) -> bool:
        """CF5: every requirement in force is a plain value the code's own evidence can test (code_values, with no bound,
        range or approximation): a read record showing every stated value is then a fit by the code's evidence as well."""
        many = self.needs.items()
        products = range(len(many)) if many else [None]
        plain_values = [v for k in products for _, v in self.engine_settings(k) if self.elicited_simple(v)]
        # (EA4: the requirement of a resolved differ reference, 'P: not V', is tested by the code too - field and value known)
        settled = self.diverge_completed(None)
        covered = sum(1 for req in self.needs.traits_remaining() if any(self.reference_correspond(ref, req) for ref in settled))
        return len(plain_values) + covered >= len(list(self.needs.traits_remaining()))

    def diverge_completed(self, k: int | None) -> list[dict[str, Any]]:
        """EA4: the resolved differ references still in force (some requirement in force is theirs) that concern product k
        (every one when k is None or the reference names no product)."""
        many = self.needs.items()
        left = self.needs.traits_remaining()

        def in_force(ref: dict[str, Any]) -> bool:
            return ref.get("relation") == "differ" and ref.get("status") == "resolved"\
                and bool(str(ref.get("value") or "").strip()) and bool(self.reference_active(ref))\
                and any(self.reference_correspond(ref, req) for req in left)

        def concerns_k(ref: dict[str, Any]) -> bool:
            if not (many and k is not None and ref.get("tag")):
                return True
            return self.ware_belonging(f"x ({ref['tag']})") in (k, None)
        return [ref for ref in self.__dict__.get("refs") or [] if in_force(ref) and concerns_k(ref)]

    def decline_repair(self) -> str:
        """CF5 (v605_plan 3.4): holds a decline - once per refusal count - that comes after a refused order, or inside the
        steps reserved for the order, while the records hold a fitting candidate for every product: one that passed the
        item check under the requirements in force, or a read one whose own record shows every stated value (code_picks,
        code_fit). The code then makes the cart hold exactly those picks (code_put: other lines and gone lines out, a line
        read in stock since the refusal back among the candidates; no search, no other pick), reads the stock of lines not
        read since, and tells the actor to order. '' lets the decline through: TA6's rule stays the single authority
        (nothing that can fit means decline), so a task whose right answer is a decline can still be declined."""
        multi = len(self.needs.items()) >= 2
        early = multi and self.rejection_num == 0 and self.shifts_remaining() > self.purchase_owed()     # (ED4: also before the reserve)
        among_buffer = self.shifts_remaining() <= self.purchase_owed()
        protocol = self.protocol
        if self.rejection_num <= 0 and not among_buffer and not early:
            return ""
        if not self.role("cart") or not self.role("order") or self.shifts_remaining() < 3 or protocol.surplus_alt() <= WRAPUP\
                or protocol.silence or self.__dict__.get("covering") or protocol.done:
            return ""
        attempts = self.__dict__.setdefault("repair_tried", set())
        token: Any = ("early", self.rejection_num) if early else self.rejection_num
        if token in attempts:
            return ""
        attempts.add(token)
        if multi:
            # (ED4, MULTI_ITEM_PLAN 2.6: when the records prove some product has no fit, the decline goes ahead)
            hopeless: list[int] = []
            try:
                hopeless = self.elicited_corroboration()
            except Exception as exc:
                self.journal.append({"kind": "decline_proof_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
            if hopeless:
                self.journal.append({"products": hopeless, "kind": "decline_proof"})
                return ""
        gone = self.defunct_lineitems()
        unseen = set(self.outdated_lineitems())
        alive = {i for i in self.handcart.measures if i not in gone and i not in unseen}
        refused_before = set(self.rejected_lineitems)
        placed = False
        iteration = str(self.needs.iteration)

        own_evidence = self.engine_evaluates_whole()

        def all_fit(chosen_picks: list[tuple[str, int]]) -> bool:
            # (a pick counts once the item check passed it now; the code's own evidence counts only when the code can see
            # every requirement in force - no bound, range or exclusion beyond it - and the pick's record has been read)
            code_fit = self.__dict__.get("code_fit") or {}
            return bool(chosen_picks) and all(
                f"{i}|{iteration}" in self.succeeded or (own_evidence and i in self.vault.read and code_fit.get(i))
                for i, _ in chosen_picks)
        try:
            self.rejected_lineitems -= alive
            avoid = set(gone)
            pool: dict[str, dict[str, Any]] = {}
            if multi:
                pool = {e["id"]: e for k in range(len(self.needs.items())) for e in self.elicited_reservoir(k)}
                avoid.update(i for i, e in pool.items() if e["clash"])
            plan = self.engine_chooses(avoid)
            checked, rounds_left = 0, 30
            while multi and plan and checked < 3 and rounds_left > 0 and protocol.surplus_alt() > WRAPUP + 30:
                # (ED4: an unverified pick first gets the item check, as unrefuted does - one that fails drops out; a read pick
                # the code's own evidence already fits needs none; one whose own record (fields, else title) fails to show
                # every plain value and every need the shopper revealed is passed over without a check)
                pending = [i for i, _ in plan if not all_fit([(i, 1)])]
                if not pending:
                    break
                bare = [i for i in pending if not pool.get(i) or not pool[i]["revealed_ok"] or pool[i]["misses"]]
                if not bare:
                    checked += 1
                    self.assess_entry(pending[0], "ORDER")
                else:
                    # (v605e review: every unread listing that is just as bare - not passed, its row missing a plain value or
                    # a revealed need - goes along with them at once, instead of one per round with code_picks re-run over
                    # every row)
                    avoid.update(bare)
                    avoid.update(i for i, e in pool.items() if not e["read"] and f"{i}|{iteration}" not in self.succeeded
                                 and (e["misses"] or not e["revealed_ok"]))
                    rounds_left -= 1
                plan = self.engine_chooses(avoid)
            if not all_fit(plan) or len(plan) + 2 > self.shifts_remaining():
                entry = {"kind": "decline_repair", "picks": plan, "done": False}
                if all_fit(plan):
                    entry["why"] = "steps"
                self.journal.append(entry)
                return ""
            # (only the picks found here go in - no search, no other pick: the cart calls, then one stock read of the lines
            # no stock-bearing record has shown since, while the order's step stays free)
            prior = protocol.silence
            protocol.silence = True
            try:
                rejected = self.engine_place(plan)
                placed = rejected == [] and all(i in self.handcart.measures for i, _ in plan)
                wanted = {p for p, _ in plan}
                to_read = [i for i in self.outdated_lineitems() if i in wanted]
                if placed and to_read and self.shifts_remaining() >= len(to_read) + 2:
                    placed = not self.engine_reassess(to_read)[1]
            finally:
                self.protocol.silence = prior
            if placed:
                placed = bool(self.handcart.measures) and not set(self.handcart.measures) & self.defunct_lineitems()
            self.journal.append({"kind": "decline_repair", "picks": plan, "done": placed, "cart": dict(self.handcart.measures)})
        except Exception as exc:
            placed = False
            self.journal.append({"kind": "decline_repair_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
        finally:
            if not placed:
                self.rejected_lineitems.clear()
                self.rejected_lineitems.update(refused_before)
        if not placed:
            return ""
        plural = bool(self.needs.items())
        return ("Not run yet (no step used). Declining is right only when nothing can fit, yet the records hold a fitting "
                "candidate " + ("for each product requested" if plural else "") + " (passed by the item check, or read with "
                "each stated value in its own record). The code therefore put " + ("them" if plural else "it") + " in the cart "
                "and read the stock: " + self.handcart.recap_alt() + " Order this cart now; decline only if a record you have read "
                "shows one of these lines failing a requirement now in force.")

    @staticmethod
    def elicited_simple(value: str) -> bool:
        """ED: whether the code's own evidence can test a value word by word (it has no bound, range, '+' join or
        approximation)."""
        lowered = str(value).lower()
        return re.search(r"\b(?:" + SETTING_BONDS + r")\b|\d\s*(?:-|to)\s*\d|[+<>]|\b(?:about|around|approx\w*|roughly)\b",
                         lowered) is None

    @staticmethod
    def elicited_term(value: str) -> bool:
        """ED6: a value made of words - letters, with no figure, bound, exclusion or reference."""
        text = str(value).lower()
        return bool(re.search(r"[a-z]{3,}", text)) and not re.search(r"\d", text) and Errand.elicited_simple(text)\
            and not re.search(OMIT_TERMS, text) and not reference_passage(text)

    def elicited_requisites(self, k: int | None) -> list[tuple[str, str, bool]]:
        """ED6: (property, value, revealed) for each requirement in force on product k (every one for a single product)."""
        many = self.needs.items()
        split = [(req, *str(req).partition(":")) for req in self.needs.traits_remaining()]
        return [(re.split(r"\s*\(", head)[0].strip(), re.sub(r"\((?:stated|revealed[^)]*)\)", "", value).strip(),
                 req not in self.needs.written)
                for req, head, _, value in split if not many or self.ware_belonging(head) in (k, None)]

    def elicited_reservoir(self, k: int | None) -> list[dict[str, Any]]:
        """ED (v605e group D): the listings seen in product k's category (single product: k None), from the records alone,
        read the way code_picks reads them: price, read, verified (passed the item check under the requirements in force),
        fit (verified, or read with its own record showing every plain stated or revealed value - code_shows - and no field
        clashing; never gone, refused, ruled out by a check or its own CH field evidence, a CE5 pointer, or over the budget
        for its units), the requirements its record does not show (misses) and those it clashes on. Memoised per records,
        requirements, checks, stock and budget."""
        many = self.needs.items()
        iteration = str(self.needs.iteration)
        stamp = (k, len(self.vault.items), len(self.vault.read), iteration, "|".join(self.needs.traits_remaining()),
                 len(self.succeeded), sum(map(bool, self.verdicts_alt.values())), tuple(sorted(self.defunct_lineitems())),
                 tuple(sorted(self.rejected_lineitems)), str(self.needs.outlay_lid), len(self.__dict__.get("refs") or []))
        cache = self.__dict__.setdefault("ed_memo", {})
        if stamp in cache:
            return cache[stamp]
        if len(cache) > 64:
            cache.clear()
        shelves = self.landing_racks()
        if many:
            home = set(shelves[k]) if k is not None and k < len(shelves) else set()
        else:
            home = set(self.rack_tags()) or set(shelves[0] if shelves else ())

        def collect() -> list[dict[str, Any]]:
            which = k if many else None
            per_line = self.needs.headcounts()
            qty = per_line[k] if many and k is not None and k < len(per_line) else (per_line[0] if per_line else 1)
            unsuccessful = {bits[0] for bits in (key.split("|") for key, val in self.verdicts_alt.items() if val) if bits[1:2] == [iteration]}
            live = {key.rpartition("|")[0] for key in self.succeeded if key.rpartition("|")[2] == iteration}
            budget = float(self.needs.outlay_lid) if whether_numeric(self.needs.outlay_lid) else None
            values = self.engine_settings(which)
            judged = [(p, v) for p, v in values if self.elicited_simple(v)]
            # (a value joined with '+' names parts that must all be present: a field of the property lacking one clashes)
            joins = [(p, bits) for p, bits in ((p, [x.strip() for x in v.split("+") if x.strip()]) for p, v in values if "+" in v)
                     if len(bits) >= 2 and all(re.search(r"[a-z]{2,}", x.lower()) for x in bits)]
            revealed = [(p, v) for p, v, flag in self.elicited_requisites(which) if flag]
            pointed = self.reference_indicators()
            gone_set = self.defunct_lineitems()

            def missing_part(fields: dict[str, Any], p: str, bits: list[str]) -> bool:
                named = [val for key, val in fields.items() if identical_trait(key, p)]
                return bool(named) and not all(any(self.engine_observed(x, val) for val in named) for x in bits)

            def word_for_word(fields: dict[str, Any], p: str, v: str) -> bool:
                return any(self.engine_observed(v, val) for key, val in fields.items() if identical_trait(key, p))

            def describe(ident: str, row: dict[str, Any]) -> dict[str, Any]:
                fields = row["fields"]
                opened = ident in self.vault.read
                raw_price = row["price"] if row["price"] is not None else self.vault.price(ident)
                price = float(raw_price) if whether_numeric(raw_price) else None
                gone = ident in gone_set or row["stock"] is False or ident in self.rejected_lineitems
                over = budget is not None and price is not None and price * qty > budget + 1e-9
                pointer = ident in pointed or self.reference_omitted(fields, which)
                # (a field that writes the value word for word never clashes - value_seen skips two-letter words)
                clash = [p for p, v in judged if self.engine_conflict(fields, p, v) and not word_for_word(fields, p, v)]
                clash += [p for p, bits in joins if missing_part(fields, p, bits)]
                missing = [(p, v) for p, v in judged if not self.engine_displays(row, p, v, set(), True)]
                ruled = ident in unsuccessful or bool(opened and self.attribute_snapshot(ident)["fails"])
                checked_ok = not gone and ident in live
                fit = (checked_ok or (opened and not missing and not clash)) and not (ruled or pointer or gone or over)
                return dict(id=ident, price=price, read=opened, verified=checked_ok, fit=fit, gone=gone, over=over,
                            pointer=pointer, ruled=ruled, clash=clash, misses=missing, row=row,
                            out=bool(gone or over or pointer or ruled or clash),
                            revealed_ok=all(self.engine_displays(row, p, v, set(), True) for p, v in revealed))
            return [describe(ident, row) for ident, row in self.engine_tablerows().items() if row["shelf"] in home]
        listing = collect() if home else []
        cache[stamp] = listing
        return listing

    def elicited_corroboration(self) -> list[int]:
        """ED4 (MC5, MULTI_ITEM_PLAN 2.6): the products the records prove unfittable - some listing of it was read, and every
        read one fails a firm need on its own record (a value its field contradicts, a check's ruling or CH field evidence,
        a CE5 pointer) or is gone, refused or over the budget for its units."""
        def no_fit(k: int) -> bool:
            opened = [e for e in self.elicited_reservoir(k) if e["read"]]
            return bool(opened) and all(e["out"] for e in opened)
        return [k for k in range(len(self.needs.items())) if no_fit(k)]

    def margin_tablerows(self) -> list[str]:
        """ED5 (v509c: decline_ok false and open these rows, in code): several products, each with a fit in the records
        (ed_pool), whose cheapest fits together go over the budget. Per product: the unread in-stock rows of its category
        priced at or below its slack - the budget minus the other products' cheapest fits, per unit - with rows of searches
        whose leading words carry most of its required values first (all-words hits count one), then the cheapest; three at
        most overall, taken from each product in turn. [] in every other case."""
        many = self.needs.items()
        budget = self.needs.outlay_lid
        if len(many) < 2 or not whether_numeric(budget):
            return []
        try:
            per_line = self.needs.headcounts()

            def measures(k: int) -> int:
                return per_line[k] if k < len(per_line) else 1
            floor_costs: list[float] = []
            for k in range(len(many)):
                fit_prices = [e["price"] for e in self.elicited_reservoir(k) if e["fit"] and e["price"] is not None]
                if not fit_prices:
                    return []
                floor_costs.append(min(fit_prices) * measures(k))
            total = sum(floor_costs)
            if total <= float(budget) + 1e-9:
                return []
            walks = list(self.__dict__.get("page_book", {}).values())
            all_hits = set(self.__dict__.get("all_hits") or [])

            def lead_scores(k: int) -> dict[str, int]:
                wanted = [{w for w in re.findall(r"[a-z0-9]{2,}", v.lower()) if w not in BOND_TERMS} for _, v, _ in self.elicited_requisites(k)]
                lead = dict.fromkeys(all_hits, 1)
                for walk in walks:
                    # (how many of the product's values the search's leading words carry - two-letter words included)
                    tokens = [w for text in self.loose_passage(walk["args"]) for w in re.findall(r"[a-z0-9]{2,}", text.lower())]
                    token_set = set(tokens[: self.foremost_terms] if self.foremost_terms else tokens)
                    score = sum(1 for vals in wanted if vals & token_set)
                    for i in (walk.get("row_shelf") or {}) if score else ():
                        lead[str(i)] = max(lead.get(str(i), 0), score)
                return lead
            queues = []
            for k in range(len(many)):
                slack = (float(budget) - (total - floor_costs[k])) / max(1, measures(k))
                lead = lead_scores(k)
                eligible = [(-lead.get(e["id"], 0), e["price"], e["id"]) for e in self.elicited_reservoir(k)
                            if not (e["read"] or e["gone"] or e["pointer"] or e["ruled"] or e["clash"])
                            and e["price"] is not None and 0 < slack and e["price"] <= slack + 1e-9]
                queues.append([i for _, _, i in sorted(eligible)])
            picked: list[str] = []
            while len(picked) < 3 and any(queues):
                for q in queues:
                    if q and len(picked) < 3:
                        picked.append(q.pop(0))
            if picked:
                self.slack_total = total
            return picked
        except Exception as exc:
            self.journal.append({"kind": "slack_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
            return []

    def ascertain_issues(self, ident: str) -> list[str] | None:
        """v605e review (ED6): the problems named by the latest item check of `ident` when it failed it; None when that
        latest check passed it or none is logged."""
        latest = next((e for e in reversed(self.journal)
                       if isinstance(e, dict) and e.get("kind") == "verify" and e.get("item") == ident), None)
        outcome = latest.get("result") if latest is not None else None
        if isinstance(outcome, dict) and outcome.get("fits") is False:
            return [str(problem) for problem in outcome.get("problems") or [] if str(problem).strip()]
        return None

    def elicited_updates(self, k: int | None) -> bool:
        """ED6: product k saw news during this errand - a refusal, or a requirement revealed or dropped for it (or for no
        product in particular)."""
        many = self.needs.items()
        return self.rejection_num > 0 or any(not many or self.ware_belonging(str(req).partition(":")[0]) in (k, None)
                                         for req in [*self.needs.uncovered, *self.needs.let_go])

    def adjacent_keep(self) -> str:
        """(v605f FE) a decline while near_basket's meaning-confirmed basket still sits in the cart (same requirements, no
        refusal since, no line gone or refused): the code places the order itself (finish_now) instead; '' lets the decline
        proceed."""
        try:
            basket = self.__dict__.get("near_cart")
            protocol = self.protocol
            current = bool(basket) and basket.get("ver") == self.needs.iteration and basket.get("refusals") == self.rejection_num\
                and bool(self.role("order"))
            lines = list(basket.get("lines") or []) if current else []
            intact = bool(lines) and all(i in self.handcart.measures and i not in self.defunct_lineitems() and i not in self.rejected_lineitems
                                         for i in lines)
            if not intact or self.shifts_remaining() < 1 or protocol.done or protocol.crashed or protocol.surplus_alt() < 2:
                return ""
            self.near_cart = None
            self.journal.append({"lines": lines, "kind": "near_order"})
            prior = protocol.silence
            try:
                self.conclude_current()
            finally:
                self.protocol.silence = prior
            return ("Not run (no step used). A meaning check confirmed that the cart lines (" + ", ".join(lines) + ") state "
                    "the requested values in different words, and nothing has changed since, so the code placed the order for "
                    "this cart rather than declining.")
        except Exception as exc:
            self.journal.append({"kind": "near_hold_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
            return ""

    def near_check(self) -> str:
        """ED6 (v509c G5/V3a near_basket): runs once per errand, before a decline that follows news (a reveal, a drop or a
        refusal), after every other hold. For each product without a fit, up to 6 (single product) or 3 read listings - in
        stock, inside the budget, meeting every revealed need on their own fields (code_shows), each of whose unshown
        requirements is a stated word value (no figure, bound, exclusion or reference) that a field of the property writes
        in words - go to ONE cheap meaning call. A basket whose every line is a fit or confirmed, inside the budget
        together, is put in by the code (its confirmed listings count as verified, rulings cleared as arbitrate does) and
        the actor is told to order. '' lets the decline proceed."""
        protocol = self.protocol
        blocked = self.__dict__.get("near_done") or not self.role("cart") or not self.role("order")\
            or self.shifts_remaining() < 4 or protocol.surplus_alt() <= WRAPUP + 30 or protocol.silence or self.__dict__.get("covering")\
            or protocol.done or protocol.expended_depleted() or protocol.without_llm
        outcome = ""
        if not blocked:
            try:
                outcome = self.near_basket()
            except Exception as exc:
                self.journal.append({"kind": "near_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
        return outcome

    def near_basket(self) -> str:
        many = self.needs.items()
        prods: list[int | None] = list(range(len(many))) if many else [None]
        per_line = self.needs.headcounts()

        def measures(n: int) -> int:
            return per_line[n] if n < len(per_line) else 1
        budget = float(self.needs.outlay_lid) if whether_numeric(self.needs.outlay_lid) else None
        basket: dict[int, tuple[str, float, int]] = {}
        options: dict[int, list[tuple[int, float, str, list[tuple[str, str, str]]]]] = {}

        def word_gaps(cand: dict[str, Any], reqs: list[tuple[str, str, bool]]) -> tuple[list[tuple[str, str, str]], bool]:
            gaps: list[tuple[str, str, str]] = []
            for prop, asked, from_shopper in reqs:
                if not self.engine_displays(cand["row"], prop, asked, set(), True):
                    written = [str(v).strip() for key, v in cand["row"]["fields"].items() if identical_trait(key, prop) and str(v).strip()]
                    if from_shopper or not written or not self.elicited_term(asked) or any(re.search(r"\d", w) for w in written):
                        return gaps, False
                    gaps.append((prop, asked, "; ".join(dict.fromkeys(written))[:120]))
            return gaps, True

        def worded_options(entries: list[dict[str, Any]], reqs: list[tuple[str, str, bool]]) -> list[Any]:
            found = []
            for cand in entries:
                if not cand["read"] or cand["gone"] or cand["over"] or cand["pointer"] or cand["price"] is None:
                    continue
                gaps, ok = word_gaps(cand, reqs)
                ruled = cand["ruled"]
                fails = {str(x).split(":", 1)[0].strip() for x in self.attribute_snapshot(cand["id"])["fails"]} if ruled else set()
                problems = (self.ascertain_issues(cand["id"]) or []) if ruled else []
                marks = {w for prop, asked, _ in gaps for w in trait_terms(prop) + re.findall(r"[a-z0-9]+", asked.lower())
                         if len(w) >= 3 and w not in BOND_TERMS}
                # (CH failures are allowed only on the gaps' own properties)
                own_fails = all(any(identical_trait(f, prop) for prop, _, _ in gaps) for f in fails)
                if ok and gaps and own_fails and all(any(re.search(r"\b" + re.escape(w), p.lower()) for w in marks) for p in problems):
                    found.append((len(gaps), cand["price"], cand["id"], gaps))
            return found
        for n, k in enumerate(prods):
            entries = self.elicited_reservoir(k)
            fits = sorted((e["price"], e["id"]) for e in entries if e["fit"] and e["price"] is not None)
            if fits:
                cheapest_price, cheapest_id = fits[0]
                basket[n] = (cheapest_id, cheapest_price, measures(n))
            elif not self.elicited_updates(k):
                return ""
            else:
                found = worded_options(entries, self.elicited_requisites(k))
                if not found:
                    return ""
                options[n] = sorted(found)[: 6 if len(prods) == 1 else 3]
        fixed = sum(price * qty for _, price, qty in basket.values())
        floor_rest = sum(min(c[1] for c in options[n]) * measures(n) for n in options)
        if not options or (budget is not None and fixed + floor_rest > budget + 1e-9):
            return ""
        flat = [(n, ident, prop, asked, says) for n in sorted(options) for _, _, ident, gaps in options[n]
                for prop, asked, says in gaps][:12]
        numbered = "\n".join(f"{pos}. {ident}: {prop} = '{asked[:60]}' | the listing's own field reads: {says}"
                             for pos, (_, ident, prop, asked, says) in enumerate(flat, 1))
        ask = ("A shopping assistant is about to turn down an order because the listings below do not state a requested value "
               "in the request's own words. THE SHOPPER'S REQUEST (for context):\n" + self.request[:1500] + "\n\nEvery line "
               "gives a listing, a requested property with its requested value, and what that listing's own field says about "
               "it.\n" + numbered + "\n\nFor every line: does the listing's value express the requested value in different "
               "words (a synonym, an alternative spelling or phrasing, or a more precise form of the same thing), or is it a "
               "different value of that property? If unsure, answer false.\nReply with ONE JSON object and nothing more: "
               "{\"lines\": [{\"n\": 1, \"same\": true}]}")
        self.near_done = True
        reply = self.protocol.confer(ask, 600, ASSESSOR)
        verdicts = reply.get("lines") if isinstance(reply, dict) and isinstance(reply.get("lines"), list) else []
        agreed: set[int] = {int(v["n"]) - 1 for v in verdicts
                            if isinstance(v, dict) and v.get("same") is True and whether_numeric(v.get("n"))}
        self.journal.append({"kind": "near_check", "same": sorted(agreed),
                         "lines": [f"{ident}:{prop}={asked}|{says}"[:120] for _, ident, prop, asked, says in flat]})
        for n in sorted(options):
            for _, price, ident, gaps in options[n]:
                spots = [pos for pos, row in enumerate(flat) if row[0] == n and row[1] == ident]
                if len(spots) == len(gaps) and all(pos in agreed for pos in spots):
                    basket[n] = (ident, price, measures(n))
                    break
            else:
                return ""
        picks = [(basket[n][0], basket[n][2]) for n in range(len(prods))]
        if (budget is not None and sum(price * qty for _, price, qty in basket.values()) > budget + 1e-9)\
                or len(picks) + 2 > self.shifts_remaining():
            return ""
        iteration = str(self.needs.iteration)
        for n in options:
            ident = basket[n][0]
            stamp = f"{ident}|{iteration}"
            self.verdicts_alt.update(dict.fromkeys([m for m in self.verdicts_alt if m == stamp or m.startswith(stamp + "|")], ""))
            self.succeeded.add(stamp)
            self.__dict__.setdefault("near_ok", set()).add(stamp)
            self.__dict__.setdefault("pass_price", {})[ident] = self.vault.price(ident)
        placed = False
        prior = self.protocol.silence
        self.protocol.silence = True
        try:
            rejected = self.engine_place(picks)
            placed = rejected == [] and all(i in self.handcart.measures for i, _ in picks)
            wanted = {p for p, _ in picks}
            to_read = [i for i in self.outdated_lineitems() if i in wanted]
            if placed and to_read and self.shifts_remaining() >= len(to_read) + 2:
                placed = not self.engine_reassess(to_read)[1]
        finally:
            self.protocol.silence = prior
        if placed:
            placed = bool(self.handcart.measures) and not set(self.handcart.measures) & self.defunct_lineitems()
        self.journal.append({"kind": "near_basket", "picks": picks, "done": placed, "cart": dict(self.handcart.measures)})
        if not placed:
            return ""
        # (v605f FE: the basket stands, and a later decline while it is still in the cart becomes the code's own order)
        self.near_cart = {"lines": [i for i, _ in picks], "ver": self.needs.iteration, "refusals": self.rejection_num}
        wording_log = self.__dict__.setdefault("near_said", {})
        kept = [(ident, prop, asked, says) for n, ident, prop, asked, says in flat if ident == basket.get(n, ("",))[0]]
        for ident, prop, asked, says in kept:
            wording_log.setdefault(f"{ident}|{iteration}", []).append((prop, asked, says))
        said = "; ".join(f"{ident} gives {prop} as '{says}' for '{asked[:60]}'" for ident, prop, asked, says in kept)
        return ("Not run yet (no step used). A meaning check found that " + said + " - the same value in different words - "
                "and every such listing meets each need the shopper revealed, is in stock and inside the budget, so the code "
                "put the basket into the cart and read its stock: " + self.handcart.recap_alt() + " Order this cart now; decline "
                "only if a record you have read shows one of these lines failing a requirement now in force.")


class CarriageAndPicklist:
    """Errand component: cart bookkeeping, list reads, the property menu and calls that message the shopper."""

    def carriage_shifted(self, prior: dict[str, float]) -> None:
        """CB3/CB6, run once a cart call went through. The cart version (cart_ver) goes up and the actor's call count
        since the change (since_cart) starts again at 0. A listing that was not in the cart before puts every line due
        for a fresh stock read (cart_seq); a line whose unit count rose needs a fresh read of its own; dropping a line or
        lowering a count needs none, since taking stock out of the cart cannot make anything sell out."""
        state = vars(self)
        self.cart_ver, self.since_cart = state.get("cart_ver", 0) + 1, 0
        retained = self.handcart.measures
        if not all(key in prior for key in retained):
            self.cart_seq = state.get("absorb_n", 0)
        read_marks = state.setdefault("line_seq", {})
        grown = [key for key, count in retained.items() if key in prior and count > prior[key] + 1e-9]
        for key in grown:
            read_marks.pop(key, None)

    def following_place(self, args_sent: dict[str, Any], reply: Any) -> None:
        """CB4, run after a cart call. A reply that already lists the cart lines makes a cart view straight after it
        pointless (cart_shown). A listing entering the cart for the first time gets its stock read right away by the code
        (one scripted call of the stock tool): it may sell out the moment it is added, and that read doubles as the
        re-read before ordering."""
        state = vars(self)
        listed = self.handcart.snapshot_measures(reply)
        if listed is not None:
            self.cart_shown = state.get("cart_ver", 0)
        seen_before = state.setdefault("first_puts", set())
        added = []
        for key, count in self.handcart.lines(args_sent):
            positive = count is None or count > 0
            if positive and key in self.handcart.measures and key not in seen_before:
                added.append(key)
        if not added:
            return
        if not self.handcart.whether_place(args_sent) or self.engine_defect(reply):
            return
        seen_before.update(added)
        unread = set(self.outdated_lineitems())
        # (when the cart reply itself showed a line's stock, that already was its read)
        added = [key for key in added if key in unread]
        reader = self.inventory_instrument()
        if not added or not reader or self.protocol.silence or state.get("covering"):
            return
        if self.shifts_remaining() < 4 or self.protocol.surplus_alt() < WRAPUP:
            return
        # (a queue of its own, kept apart from open_queue, so no other path that queues reads picks up its purpose or
        # tool; open_named serves it only when no other queued read is pending)
        queued = list(state.get("put_queue") or [])
        extra = [key for key in added if key not in queued]
        self.put_queue = (queued + extra)[:4]
        self.put_tool = reader
        entry = {"kind": "after_put_read", "lines": list(self.put_queue), "tool": reader}
        self.journal.append(entry)

    def flag_recent(self, reply: Any) -> None:
        """CB3: when a cart reply itself carries a line's stock (a record holding the line's id plus a stock field), that
        counts as the line's stock read after this change."""
        state = vars(self)
        if not state.get("stock_seen"):
            return
        read_marks = state.setdefault("line_seq", {})

        def note(rec: dict[str, Any]) -> None:
            key = self.dossier_uid(rec)
            if key not in self.handcart.measures:
                return
            if self.inventory_condition(rec) is None:
                return
            read_marks[key] = vars(self).get("absorb_n", 0)
        traverse_mappings(reply, note)

    def carriage_vanished(self, cart_lines: list[str]) -> set[str]:
        """CB5, ahead of the code's own first order: the cart lines known to be sold out, after a single stock re-read
        (with the stock tool) of the lines not read since the latest cart change, when the steps allow it."""
        dead = set(cart_lines) & self.defunct_lineitems()
        unread = []
        for key in self.outdated_lineitems():
            if key in cart_lines and key not in dead:
                unread.append(key)
        if unread and self.inventory_instrument() and self.shifts_remaining() >= len(unread) + 2:
            sold = self.engine_reassess(unread)[1]
            dead |= set(sold) | (set(cart_lines) & self.defunct_lineitems())
        return dead

    def bundle_instrument(self) -> tuple[str, str, int] | None:
        """(tool, argument, id limit) for a tool that reads several listings in one call, judged by its schema only: a
        single array argument of plain strings allowing 2 or more items, nothing else required, and no other role
        (search, message, cart, cart view, order, decline). None when the store offers none, or after its first use
        failed."""
        if "batch_spec" in vars(self):
            return vars(self)["batch_spec"]

        def list_arg(spec: Any) -> Any:
            return isinstance(spec, dict) and spec.get("type") == "array" and isinstance(spec.get("items"), dict)\
                and spec["items"].get("type") == "string" and not spec["items"].get("enum")

        def scan() -> tuple[str, str, int] | None:
            busy = {self.role(r) for r in ("search", "message", "cart", "cart_view", "order", "decline")} - {None}
            blurb: dict[str, str] = {}
            for entry in self.toolkit or []:
                fn = entry.get("function") if isinstance(entry, dict) else None
                if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                    continue
                spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", fn["name"]).replace("_", " ")
                blurb[fn["name"]] = spaced + " " + str(fn.get("description") or "")
            writes = r"\b(?:remov|delet|cancel|clear|discard|releas|refund|purchas|checkout|wish|basket|cart|bag)"
            for tool_name in list(self.structures):
                # (reads only: a tool whose name or blurb says it alters the cart or a list never counts as a list read)
                if re.search(writes, blurb.get(tool_name, tool_name).lower()):
                    busy.add(tool_name)
            for tool_name, schema in self.structures.items():
                fields = schema.get("properties") if isinstance(schema, dict) else None
                if not isinstance(fields, dict):
                    fields = {}
                arrays = [key for key, spec in fields.items() if list_arg(spec)]
                if len(arrays) != 1 or tool_name in busy:
                    continue
                must = schema.get("required")
                if not isinstance(must, list):
                    must = []
                if any(r not in arrays for r in must):
                    continue
                limit = fields[arrays[0]].get("maxItems")
                if whether_numeric(limit) and limit >= 2:
                    return tool_name, arrays[0], int(min(limit, 10))
            return None
        try:
            found = scan()
        except Exception:
            found = None
        self.batch_spec = found
        return found

    @staticmethod
    def bundle_approved(replies: list[Any], wanted_ids: list[str]) -> bool:
        """Whether a list read gave back the listings' records: at least half of the requested ids appear in a record that
        holds a property map (or, for flat records, 5 or more short text fields)."""
        wanted, matched = set(wanted_ids), set()

        def scalars(inner: dict[str, Any]) -> int:
            return sum(1 for x in inner.values() if isinstance(x, (str, int, float)))

        def scan(rec: dict[str, Any]) -> None:
            fields = list(rec.values())
            mapped = any(isinstance(v, dict) and scalars(v) >= 2 for v in fields)
            shorts = sum(1 for v in fields if isinstance(v, str) and 0 < len(v) <= 70)
            flat = shorts >= 5 and all(not isinstance(v, dict) for v in fields)
            if not (mapped or flat):      # (a flat record with 5+ short fields qualifies too, as rich_reads counts it)
                return
            matched.update(v for v in fields if isinstance(v, str) and v in wanted)
        try:
            for payload in replies:
                traverse_mappings(payload, scan)
        except Exception:
            return False
        return len(matched) > 0 and len(matched) * 2 >= len(wanted)

    @staticmethod
    def heading_belonging(rec: dict[str, Any]) -> str:
        direct = rec.get("title")
        if isinstance(direct, str):
            return direct
        for key, val in rec.items():
            if isinstance(val, str) and re.search(r"title|heading|headline", str(key).lower()):
                return val
        return ""

    def picklist_tablerows(self, shelf_set: set[str], limit: int, skip: list[str], measures: int = 1) -> list[str]:
        """No more than `limit` unread rows from the categories in `shelf_set`: in stock, affordable for `units` units at
        the last price seen, and neither in a refused cart nor in `skip`. Until some record read there shows every stated
        value of the product in its fields, the first such row (result order) of the search whose leading words hold most
        of those values comes first; after it the all-words hits, then rows shown by a search led by a required value (the likely
        fits), then everything else; inside each group, rows whose titles overlap least with one another and with the
        titles already read (their fields cover more of the shelf)."""
        state = vars(self)
        budget_cap = self.needs.outlay_lid
        seq: list[str] = []
        names: dict[str, str] = {}
        gone: set[str] = set()
        seen_words: list[set[str]] = []

        def tokens(text: str) -> set[str]:
            return set(re.findall(r"[a-z0-9]{3,}", text.lower()))

        def collect(record: dict[str, Any]) -> None:
            mark = self.tablerow_flag(record)
            if mark not in self.vault.ids or self.rack_belonging(record) not in shelf_set:
                return
            if mark in self.vault.read:
                seen_words.append(tokens(self.heading_belonging(record)))
                return
            if self.traded_depleted(record):
                gone.add(mark)
            if mark in seq:
                return
            seq.append(mark)
            names[mark] = self.heading_belonging(record)
        for entry in self.vault.items:
            traverse_mappings(entry, collect)
        scores = state.get("hit_score", {})
        hits = sorted(state.get("all_hits", []), key=lambda i: -scores.get(i, 0))

        def affordable(mark: str) -> bool:
            if not whether_numeric(budget_cap):
                return True
            cost = self.vault.price(mark)
            return not (whether_numeric(cost) and cost * measures > budget_cap)
        ordered = [i for i in hits if i in names]
        ordered += [i for i in seq if i not in hits]
        pool_all = [i for i in ordered if i not in gone and i not in skip and i not in self.rejected_lineitems and affordable(i)]
        bag = {i: tokens(names.get(i, "")) for i in pool_all}
        led_rows = state.get("row_score", {})
        selected: list[str] = []
        try:
            # (while no likely fit was read: the search that carries the product's values lists its likeliest fits
            # first, and its first such row is placed ahead of the picks made for title variety)
            multi = bool(self.needs.items())
            slot = None
            if multi:
                slot = next((n for n, h in enumerate(self.landing_racks()) if h == shelf_set), None)
            wanted_vals = []
            for need in self.needs.traits_remaining():
                if multi and self.ware_belonging(need.partition(":")[0]) not in (slot, None):
                    continue
                wanted_vals.append(re.sub(r"\((?:stated|revealed[^)]*)\)", "", need.partition(":")[2]))
            pointer = r"as shown|same as|the page|sold by|redirect|refer"
            wanted_vals = [v for v in wanted_vals if v.strip() and affirmative_setting(v) and not re.search(pointer, v.lower())]
            viable = wanted_vals and limit > 0 and (not multi or slot is not None)
            fact_texts = {}

            def gather(record: dict[str, Any]) -> None:
                mark = self.tablerow_flag(record)
                if mark not in self.vault.read or self.rack_belonging(record) not in shelf_set:
                    return
                for inner in record.values():
                    if isinstance(inner, dict):
                        fact_texts.setdefault(mark, []).extend(str(x) for x in inner.values() if isinstance(x, (str, int, float)))
            if viable:
                for entry in self.vault.items:
                    traverse_mappings(entry, gather)
            fit_seen = any(all(setting_observed(v, " | ".join(texts)) for v in wanted_vals) for texts in fact_texts.values())
            if viable and not fit_seen:
                walks = [w for w in state.get("page_book", {}).values() if self.traverse_conveys(w["args"], slot)]
                walks.sort(key=lambda w: -len(self.traverse_conveys(w["args"], slot)))
                for walk in walks:
                    lead = next((i for i in (walk.get("row_shelf") or {}) if i in bag), None)
                    if not lead:
                        continue
                    selected.append(lead)
                    seen_words.append(bag[lead])
                    break
        except Exception:
            selected = []

        def rank(mark: str) -> tuple[int, float, int]:
            tier = 0 if mark in hits else (1 if led_rows.get(mark) else 2)
            mine = bag[mark]
            overlap = max((len(mine & w) / max(1, len(mine | w)) for w in seen_words), default=0.0)
            return tier, overlap, pool_all.index(mark)
        while len(selected) < limit:
            window = [i for i in pool_all if i not in selected][:16]
            if not window:
                break
            winner = min(window, key=rank)
            selected.append(winner)
            seen_words.append(bag[winner])
        return selected

    def upfront_peruse(self) -> bool:
        """Ahead of the first planned question, when a tool reads several listings at once: one list read of unread rows
        for each product whose own category has too few records read (2 rows for a product with none, 4 for a lone product
        with none, enough to reach 4 records for one with fewer), so that the question menu carries each product's own
        fields; after that, once more while over 25 steps remain, one list read topping up to 4 any product whose
        category still holds under 3 records with facts (its first reads came back thin). True when reads were queued."""
        try:
            state = vars(self)
            lister = self.bundle_instrument()
            second_round = state.get("pre_rounds", 0) == 1
            if not lister:
                return False
            if state.get("pre_read_done") and not second_round:
                return False
            if len(self.enquiries_journal) > 1 or self.shifts_remaining() <= (25 if second_round else 20):
                return False
            if self.protocol.surplus_alt() < WRAPUP + 60:
                return False
            shelf_sets = self.landing_racks()
            headcounts = [row["quantity"] for row in self.needs.items()] or [self.needs.quantity()]

            def measures(n: int) -> Any:
                return headcounts[min(n, len(headcounts) - 1)]

            def stocked(shelf_set: set[str]) -> int:
                return sum(self.vault.detailed_peruses(s) for s in shelf_set) if shelf_set else 4

            def queue_up(rows_in: list[str], say: str) -> None:
                self.open_queue, self.open_recheck = rows_in, True
                self.open_say, self.open_purpose = say, "menu"
            if second_round:
                self.pre_rounds = 2
                lists: list[list[str]] = []
                employed: list[str] = []
                for n, shelf_set in enumerate(shelf_sets):
                    count_read = stocked(shelf_set)
                    if count_read >= 3:
                        continue
                    rows = self.picklist_tablerows(shelf_set, 4 - count_read, employed + list(state.get("pre_picked") or []), measures(n))
                    employed += rows
                    lists.append(rows)
                depth = max((len(rows) for rows in lists), default=0)
                woven = [rows[d] for d in range(depth) for rows in lists if d < len(rows)]
                woven = woven[: lister[2]]
                if not woven:
                    return False
                queue_up(woven, "A few products' opened listings came back with thin details, so I read some more of them "
                                "before asking anything.")
                return True
            blank: list[str] = []
            short: list[tuple[int, int]] = []
            for n, shelf_set in enumerate(shelf_sets):
                count_read = stocked(shelf_set)
                if count_read == 0:
                    blank += self.picklist_tablerows(shelf_set, 2 if self.needs.items() else 4, blank, measures(n))
                elif count_read < 4:
                    short.append((n, 4 - count_read))
            woven = list(blank)
            for n, gap in short:
                woven += self.picklist_tablerows(shelf_sets[n], gap, woven, measures(n))
            factor = 2 if len(blank) > lister[2] and self.shifts_remaining() > 25 else 1
            woven = woven[: lister[2] * factor]
            if not woven:
                return False
            self.pre_read_done, self.pre_rounds = True, 1
            self.pre_picked = list(woven)
            queue_up(woven, "I read a few listings of every product requested first, to learn which fields they carry before "
                            "I ask the shopper anything.")
            return True
        except Exception as fault:
            detail = (type(fault).__name__ + ": " + str(fault))[:300]
            self.journal.append({"kind": "pre_read_error", "error": detail})
            return False

    def absent_tablerows(self) -> list[str]:
        """For a request naming several products: unread rows of each product still lacking a verified fit (no listing
        that passed the item check, and none the latest decline audit named as a possible fit, in its own category),
        in stock and affordable for its units at the last price seen; all-words hits first, then the newest rows, dealt
        out in turn across those products - as many as one list read can take (3 without such a tool)."""
        try:
            products = self.needs.items()
            if not products:
                return []
            shelf_sets = self.landing_racks()
            iteration = str(self.needs.iteration)
            live = {key.rpartition("|")[0] for key in self.succeeded if key.rpartition("|")[2] == iteration}
            live = live - self.rejected_lineitems - self.defunct_lineitems()
            known = live | set(vars(self).get("audit_fits") or [])
            covered = {self.rack_familiar(i) for i in known} - {""}
            short = [n for n, shelf_set in enumerate(shelf_sets) if shelf_set and not shelf_set & covered]
            if not short:
                return []
            lister = self.bundle_instrument()
            limit = lister[2] if lister else 3
            budget_cap = self.needs.outlay_lid
            recent: list[tuple[str, str]] = []
            gone: set[str] = set()

            def collect(record: dict[str, Any]) -> None:
                mark = self.tablerow_flag(record)
                shelf = self.rack_belonging(record)
                if mark not in self.vault.ids or not shelf:
                    return
                recent.append((mark, shelf))
                if self.traded_depleted(record):
                    gone.add(mark)
            for entry in reversed(self.vault.items):
                traverse_mappings(entry, collect)
            place: dict[str, str] = {}
            for mark, shelf in recent:
                if mark not in place:
                    place[mark] = shelf
            scores = vars(self).get("hit_score", {})
            hits = sorted(vars(self).get("all_hits", []), key=lambda i: -scores.get(i, 0))
            buckets: dict[int, list[str]] = {n: [] for n in short}
            listed = [i for i in hits if i in place] + [mark for mark, _ in recent]
            done: set[str] = set()
            for mark in listed:
                if mark in done:
                    continue
                done.add(mark)
                if mark in self.vault.read or mark in self.rejected_lineitems or mark in gone:
                    continue
                n = next((n for n in short if place.get(mark) in shelf_sets[n]), None)
                if n is None:
                    continue
                cost = self.vault.price(mark)
                over = whether_numeric(budget_cap) and whether_numeric(cost) and cost * products[n]["quantity"] > budget_cap
                if not over:
                    buckets[n].append(mark)
            depth = max((len(b) for b in buckets.values()), default=0)
            dealt = [buckets[n][d] for d in range(depth) for n in short if d < len(buckets[n])]
            return dealt[:limit]
        except Exception as fault:
            detail = (type(fault).__name__ + ": " + str(fault))[:300]
            self.journal.append({"kind": "lacking_error", "error": detail})
            return []

    # ---------------------------------------------------------------------------------------------- asking the shopper
    def passage_param(self, tool_name: str) -> str | None:
        schema = self.structures.get(tool_name or "") or {}
        fields = schema.get("properties") or {}
        must = list(schema.get("required") or [])

        def is_text(spec: Any) -> bool:
            if not isinstance(spec, dict):
                return False
            textual = spec.get("type") == "string" or "string" in (spec.get("type") or [])
            return textual and "enum" not in spec
        texts = [key for key, spec in fields.items() if is_text(spec)]
        target = next((key for key in must if key in texts), None)
        if not target and len(texts) == 1:
            target = texts[0]
        if target and all(key == target for key in must):
            return target
        return None

    def missive_invoke(self, body: str, call_id: str) -> dict[str, Any] | None:
        tool_name = self.role("message")
        field = self.passage_param(tool_name) if tool_name else None
        if not field:
            return None
        built, problems = ParamAssess.carryout({field: body}, self.structures[tool_name])
        if problems:
            return None
        payload = json.dumps(built)
        return {"id": call_id, "type": "function", "function": {"name": tool_name, "arguments": payload}}


class EnquiryPlanning:
    """Errand part: picking the product to ask about, keeping the question quota and sending the planned questions."""

    def ware_belonging(self, source_text: str) -> int | None:
        """The requested product a property label or a question points at: the one named by its first "(product)" tag,
        otherwise the one the whole text names, judged by shared kind words (a word several products have counts half);
        None when nothing is named or two products tie (a paired question naming both belongs to neither)."""
        items = self.needs.items()
        if not items:
            return None
        kinds = [set(category_lexemes(item["product"])) for item in items]
        source_text = str(source_text)
        for chunk in re.findall(r"\(([^)]*)\)", source_text)[:1] + [source_text]:
            said = set(category_lexemes(chunk))
            tallies = []
            for n, mine in enumerate(kinds):
                others = set().union(*(other for j, other in enumerate(kinds) if j != n))
                tallies.append(category_rating(mine, said, others))      # (CC: singular and plural count the same)
            best = max(tallies)
            if best > 0 and tallies.count(best) == 1:
                return tallies.index(best)
        return None

    def scheme_ware(self, entry: dict[str, Any]) -> int | None:
        """Which product a planned question concerns: the tag on its property, failing that its question's words."""
        by_prop = self.ware_belonging(str(entry.get("property") or ""))
        if by_prop is None:
            return self.ware_belonging(str(entry.get("question") or ""))
        return by_prop

    def allocation_initial(self, plan_list: list[dict[str, Any]], fresh_ok: Any = None) -> list[dict[str, Any]]:
        """CC: each message goes to a product whose turn it is: fewest answered questions plus revealed needs, then fewest
        revealed needs (a question counts for the product its property's tag names, else the one its sent text names; an
        unanswered one, '[not answered]', counts for none); a product with a revealed need waits until every product
        without one had 2 answered questions (EC5/MC2: rather, a product with a revealed need comes after every product
        with no answered question; among the others a product an open story hint names comes first whatever its reveals,
        then fewest answered plus revealed; EC6/MC1: then fewest unanswered messages sent to it, then fewer revealed).
        For the products whose turn it is, the planner's first question goes out first, else a planned second of one,
        else quota_pick's question (a product an open story hint names first, then request order; its common-field
        fallback only for a product never asked); with nothing, the next products in line. While nothing has been
        asked yet, a first planned question that names no product is left in front."""
        items = self.needs.items()
        if not items:
            return plan_list
        n_items = len(items)
        answered = [0] * n_items
        messaged = [0] * n_items
        for entry in self.enquiries_journal:
            head = entry.split(":", 1)[0]
            j = self.ware_belonging(head)
            if j is None:
                quoted = re.search(r'asked "(.*?)" -> ', entry)
                j = self.ware_belonging(quoted.group(1)) if quoted else None
            if j is None:
                continue
            messaged[j] += 1          # (EC6: every message to the product counts here, unanswered ones as well)
            if "[not answered]" in head:
                continue
            answered[j] += 1
        if plan_list and max(answered) == 0 and self.scheme_ware(plan_list[0]) is None:
            return plan_list
        revealed = [0] * n_items
        for found_need in self.needs.uncovered:
            j = self.ware_belonging(str(found_need).split(":", 1)[0])
            if j is not None:
                revealed[j] += 1
        try:
            self.examine_clues()     # (refreshes the hint menu memo exactly as the original lookup did)
        except Exception:
            pass
        never = [j for j in range(n_items) if not answered[j]]
        try:
            # (EC5: a product counts as story-hinted while one of its open hints can be asked by its own key, or while
            # its open [no key] hints - which the planner asks through a shown key it maps them to - outnumber its
            # answered questions)
            by_key: set[int] = set()
            no_key = [0] * n_items
            for hint in self.examine_clues():
                name = str(hint).split(":", 1)[0]
                j = self.ware_belonging(name)
                if j is None:
                    continue
                if self.simple_clue(j, name.strip(), hinted=True):
                    no_key[j] += 1
                else:
                    by_key.add(j)
            story_hinted = {j for j in range(n_items) if j in by_key or no_key[j] > answered[j]}
        except Exception:
            story_hinted = set()
        untried = {j for j in range(n_items) if messaged[j] > answered[j] and self.variant_unattempted(j)}

        def standing(j: int) -> tuple[int, int, int, int, int]:
            # (EC5: a product with a reveal after those with no answered question; then one an open story hint names;
            # then fewest answered plus revealed; EC6: then fewest unanswered messages - v605f FC4: not counted for a
            # product that still holds an untried hint alternative; then fewer revealed)
            behind = 1 if revealed[j] and any(o != j for o in never) else 0
            silent = 0 if j in untried else messaged[j] - answered[j]
            return behind, 0 if j in story_hinted else 1, answered[j] + revealed[j], silent, revealed[j]

        def plain_for(name: str, product: str) -> dict[str, Any]:
            return {"property": name, "question": f"Which {name.split('(')[0].strip()} would you like for the {product}?"}
        levels = {j: standing(j) for j in range(n_items)}
        for level in sorted(set(levels.values())):
            due_now = [j for j in range(n_items) if levels[j] == level]
            for pos, entry in enumerate(plan_list):
                if self.scheme_ware(entry) in due_now:
                    rest = plan_list[:pos] + plan_list[pos + 1:]
                    return [entry] + rest     # (the planner's own order among the products due)
            for entry in plan_list:
                alt = str(entry.get("second") or "").strip()
                j = self.ware_belonging(alt) if alt else None
                if j not in due_now:
                    continue
                # (a planned second for a product due is sent by itself: it is the planner's next choice for it)
                tag = re.search(r"\(([^)]+)\)", alt)
                return [plain_for(alt, tag.group(1).strip() if tag else items[j]["product"])] + plan_list
            for j in sorted(due_now, key=lambda j: (j not in story_hinted, j)):
                try:
                    choice = self.allocation_choose(j, answered[j] == 0, fresh_ok)
                except Exception:
                    choice = None
                if choice and fresh_ok is not None:
                    label = choice.get("property")
                    if not fresh_ok(label) or self.inquired_toward(j, label):
                        choice = None       # (v605f FC3: the code's own pick must pass the planner's freshness test too)
                if choice:
                    return [choice] + plan_list
                if vars(self).get("quota_wait"):
                    return plan_list        # (its rows get opened first; the question comes after them)
        return plan_list

    def allocation_choose(self, idx: int, field_pick: bool = True, fresh_ok: Any = None) -> dict[str, Any] | None:
        """A question for product idx, which had none so far: when none of its records was read, the code first reads a few
        of its search rows (and asks afterwards); otherwise one planner call limited to that product's fields decides,
        and failing that the product's most common field still open (open_prop)."""
        state = vars(self)
        product = self.needs.items()[idx]["product"]
        shelf_sets = self.landing_racks()
        shelf_set = shelf_sets[idx] if idx < len(shelf_sets) else set()
        shelves = [s for s in self.vault.racks_peruse() if s.strip().lower() in shelf_set]
        if not shelves and shelf_set and state.get("quota_reads", {}).get(idx):
            # (CC: its rows were opened and still no read record of its category counts: use its category menu as is)
            with_menu = [s for s in sorted(shelf_set) if self.vault.trait_picklist(s)[0]]
            shelves = with_menu[:1]
        if not shelves:
            # (nothing of this product read yet: the code opens a few of its search rows before asking about it)
            lister = self.bundle_instrument()
            unread_rows: list[str] = []
            if shelf_set:
                unread_rows = self.picklist_tablerows(shelf_set, min(4, lister[2]) if lister else 2, [], self.needs.items()[idx]["quantity"])
            if unread_rows and (self.role("details") or lister) and not state.get("quota_reads", {}).get(idx)\
                    and self.shifts_remaining() > 12:
                state.setdefault("quota_reads", {})[idx] = True
                self.open_queue, self.open_recheck = unread_rows, True
                self.open_say = f"I first read a few {product} listings, to learn which fields they carry, before asking about it."
                self.open_purpose, self.quota_wait = "quota", True
            return None
        first_shelf = shelves[0]
        menu = self.picklist_prune(self.vault.trait_picklist(first_shelf)[0], self.vault.detailed_peruses(first_shelf))     # (EC7)

        def viable(label: Any) -> bool:
            # (v605f FC3: never a property already spent, already asked for this product (answered or not), or stale)
            if fresh_ok is None:
                return True
            return bool(fresh_ok(label)) and not self.inquired_toward(idx, label)

        def simple(name: str) -> dict[str, Any]:
            return {"property": f"{name} ({product})", "question": f"Which {name} would you like for the {product}?"}
        planned = self.product_plan(idx, menu)
        if planned and viable(planned.get("property")):
            return planned
        try:
            hints = self.examine_clues(idx)
        except Exception:
            hints = []
        for hint in hints:
            # (an open story hint for this product goes before its most common field)
            name = re.split(r"\s*\(", hint.split(":", 1)[0])[0].strip()
            if not name or rubbish_slot(name) or self.simple_clue(idx, name, hinted=True):
                continue
            if viable(f"{name} ({product})"):
                return simple(name)
        if not field_pick:
            return None       # (CC: a product asked before only gets a planned or a hinted question, never the field pick)
        for by_values in (False, True):
            # (first the fields no requirement settles; then, once none is left, one whose like-named requirement holds a
            # value none of its sample values carry: story words put where the property belongs)
            for row_text in menu.splitlines():
                name = row_text.strip("- ").split(" (in ")[0].strip()
                if not name or not self.examine_trait(idx, name, row_text, by_samples=by_values):
                    continue
                if viable(f"{name} ({product})"):
                    return simple(name)
        return None

    def product_plan(self, idx: int, field_menu: str) -> dict[str, Any] | None:
        """A single planner call (the cheap model, no more than once per errand) limited to product idx: the request, the
        ledger, the questions so far and that product's own field list. Its first property still open becomes that
        product's question."""
        try:
            result = self.ware_scheme_invoke(idx, field_menu)
        except Exception:
            result = None
        return result

    def ware_scheme_invoke(self, idx: int, field_menu: str) -> dict[str, Any] | None:
        state = vars(self)
        if not field_menu or state.get("product_plans", 0) >= 1:
            return None
        if self.protocol.surplus_alt() < WRAPUP + 40:
            return None
        self.product_plans = state.get("product_plans", 0) + 1
        item_name = self.needs.items()[idx]["product"]
        shout = item_name.upper()
        left = max(1, self.inquire_headroom() - self.letters_transmitted)
        pieces = [INQUIRE_SCHEME.replace("QUESTION_COUNT", str(min(3, left))), "\n\nTHE SHOPPER'S REQUEST:\n", self.request,
                  "\n\nREQUIREMENTS SO FAR:\n", self.needs.passage_payload()]
        if len(self.enquiries_journal):
            pieces.append("\n\nALREADY ASKED (each with the shopper's reply):\n" + "\n".join(self.enquiries_journal))
        pieces.append(f"\n\nTHIS TIME EVERY QUESTION CONCERNS THE {shout}: each other requested product has had a question "
                      f"already and this one has had none. Pick properties of the {item_name} only, from its own list below, "
                      "starting with the one the story's details point to.")
        try:
            hints = self.examine_clues(idx)
        except Exception:
            hints = []
        if hints:
            listed = "\n".join("- " + self.clue_flag(h) for h in hints[:6])
            pieces.append("\n\nSTORY HINTS (story details from the request with the property each one seems to be about, the "
                          "likeliest first; these are not requirements):\n" + listed)
        pieces.append(self.scheme_annotations(idx))
        pieces.append(f"\n\nPROPERTIES IN THE DETAILS OF THE {shout} LISTINGS SEEN SO FAR (name, how many listings, sample "
                      "values):\n" + field_menu)
        reply = self.protocol.confer("".join(pieces), 1500, STRATEGIST[:1])
        self.journal.append({"kind": "product_plan", "product": item_name, "result": reply})
        kinds = category_terms(item_name)

        def simple(name: str) -> str:
            return f"Which {name} would you like for the {item_name}?"
        for entry in (reply or {}).get("questions") or []:
            if not isinstance(entry, dict):
                continue
            lead = str(entry.get("property") or "").split("(")[0].strip()
            backup = str(entry.get("second") or "").split("(")[0].strip()
            if re.search(r"none|^-$|n/a", backup.lower()):
                backup = ""
            text = str(entry.get("question") or "").strip()
            if self.simple_clue(idx, lead):
                lead = ""
            if backup and self.simple_clue(idx, backup):
                backup = ""
            if lead and self.scheme_examine(idx, lead):
                mentions = bool(text) and bool(kinds & category_terms(text))
                partner = ""
                if mentions and backup and not rigorous_trait(lead, backup) and self.scheme_examine(idx, backup):
                    partner = backup
                return {"property": f"{lead} ({item_name})", "second": f"{partner} ({item_name})" if partner else "",
                        "question": text if mentions else simple(lead)}
            if backup and self.scheme_examine(idx, backup):
                return {"property": f"{backup} ({item_name})", "question": simple(backup)}
        return None

    def examine_trait(self, idx: int | None, asked_prop: str, menu_line: str = "", by_samples: bool = False) -> bool:
        """Whether a property may still hide a need of product idx (None: the single product, or any): not asked yet (a
        paired question that got no answer leaves both open), not settled by a requirement in force for it (one with the
        same strict name and a positive stated value settles it; with by_samples and its menu line, only a value among its
        sample values does; and, given its menu line, one whose positive word value its sample values carry), and not a
        property the shopper does not mind (strict name)."""
        try:
            bare = str(asked_prop).split("(")[0].strip()
            if not bare or rubbish_slot(bare):
                return False
            multi = bool(self.needs.items())

            def ours(text: str, tagged_only: bool = False) -> bool:
                if idx is None or not multi:
                    return True
                which = self.ware_belonging(text)
                return which == idx or (which is None and not tagged_only)
            for entry in self.enquiries_journal:
                need_head = entry.split(":", 1)[0].strip("- ")
                if "[not answered]" in need_head:
                    continue
                if any(ours(part) and self.identical_attribute(idx, part, bare) for part in need_head.split(";")):
                    return False
            samples = ""
            if "):" in menu_line:
                samples = menu_line.split("):", 1)[1].lower()
            for need in self.needs.traits_remaining():
                need_head, _, need_value = str(need).partition(":")
                if not ours(need_head):
                    continue
                if rigorous_trait(need_head, bare):
                    # (only a positive stated value settles it: 'closure: no laces, ties or buckles' keeps it open)
                    if self.finalizes(str(need), samples if by_samples else ""):
                        return False
                elif identical_trait(need_head, bare):
                    # (related names - collar type and collar, sleeve length and sleeves, material and material filter -
                    # mean this property under another key when one of the requirement's values shows among the field's
                    # sample values; display size and display resolution, or protein and protein source, share none)
                    seen_in = samples or " | ".join(self.attribute_settings(idx, bare))
                    if affirmative_setting(need_value, need_head) and setting_observed(need_value, seen_in):
                        return False
                else:
                    vwords = [w for w in re.findall(r"[a-z]{3,}", need_value.lower()) if w not in BOND_TERMS]
                    if not samples or not vwords or re.search(r"\d", need_value) or not affirmative_setting(need_value, need_head):
                        continue
                    if all(re.search(r"\b" + re.escape(w), samples) for w in vwords):
                        return False
            for p in self.needs.indifferent:
                if ours(str(p), tagged_only=True) and self.identical_attribute(idx, str(p), bare):
                    return False
            return True
        except Exception:
            return True

    def scheme_annotations(self, idx: int | None) -> str:
        """Extra planner notes (with idx: only for product idx): the second halves of paired questions whose reply ignored
        them (still open: the shopper answered only the first), and the questions the assistant itself wanted to ask."""
        try:
            multi = bool(self.needs.items())
            asked = set()
            for entry in self.enquiries_journal:
                for part in entry.split(":", 1)[0].strip("- ").split(";"):
                    asked.add(part.strip().lower())
            leftover = []
            for prop in vars(self).get("open_pair") or []:
                j = self.ware_belonging(prop) if multi else None
                if idx is not None and multi and j not in (idx, None):
                    continue
                if prop.lower() in asked or not self.scheme_examine(j, prop):
                    continue
                leftover.append(prop)
            desired = []
            for want in vars(self).get("wanted_asks") or []:
                if idx is None or not multi or self.ware_belonging(want) in (idx, None):
                    desired.append(want)
            notes = []
            if leftover:
                notes.append("\n\nOPEN FROM LAST PAIR (the second property of a planned pair that its reply left unanswered - "
                             "the shopper took up only the first, or only the first was sent - so it is still open; each was "
                             "the plan's next choice):\n" + "\n".join("- " + p for p in leftover[-3:]))
            if desired:
                notes.append("\n\nQUESTIONS THE ASSISTANT WANTED TO ASK (written after reading the listing records; a property "
                             "one of them names that is still open and appears as a key below goes first, asked under that "
                             "key's own name):\n" + "\n".join("- " + w for w in desired[-3:]))
            return "".join(notes)
        except Exception:
            return ""

    def desire_inquire(self, args_in: Any) -> None:
        """Remembers the text of an actor question the code turned down while it plans the questions (last 3 distinct)."""
        try:
            if isinstance(args_in, dict):
                parsed = args_in
            elif isinstance(args_in, str) and args_in.strip():
                parsed = serialized_entity(args_in)
            else:
                parsed = {}
            bits = [str(v).strip() for v in (parsed or {}).values() if isinstance(v, str) and v.strip()]
            joined = " ".join(bits)[:300]
            if not joined:
                return
            low = joined.lower()
            stored = [w for w in vars(self).get("wanted_asks") or [] if w.lower() != low]
            stored.append(joined)
            self.wanted_asks = stored[-3:]
        except Exception:
            pass

    def discard_desired(self, asked: str) -> None:
        """Stops offering a wanted question whose words cover a property that was just asked (every word of its name)."""
        try:
            cleaned = str(asked).replace("[not answered]", "")
            names = [re.split(r"\s*\(", piece)[0] for piece in re.split(r";|\band\b", cleaned)]

            def covered(name: str, stems: set[str]) -> bool:
                needed = trait_terms(name)
                return bool(needed) and all(w in stems or self.wordbase(w) in stems for w in trait_terms(name))
            survivors = []
            for text in vars(self).get("wanted_asks") or []:
                stems = {self.wordbase(w) for w in re.findall(r"[a-z]+", text.lower())}
                if not any(covered(name, stems) for name in names):
                    survivors.append(text)
            self.wanted_asks = survivors
        except Exception:
            pass

    def monikers_peer(self, text_in: str, own_prop: str, field_menu: str, partner: str = "") -> bool:
        """CD: a planned question's wording may name a subject other than its property: all the words of another listed
        field's name (or of the planner's second), or one of another field's sample values offered as an example -
        leaving aside the words of the property and of the products' names."""
        try:
            squashed = " ".join(re.findall(r"[a-z0-9]+", str(text_in).lower()))
            words_said = set(squashed.split())
            singles = {w[:-1] for w in words_said if len(w) > 3 and w.endswith("s") and not w.endswith("ss")}
            words_said |= singles
            own_words = set(trait_terms(own_prop)) | {"which", "would", "you", "like"}
            rows = self.needs.items() or [{"product": str(self.needs.perusal.get("product") or "")}]
            for entry in rows:
                for word in re.findall(r"[a-z0-9]+", str(entry.get("product") or "").lower()):
                    wordbase = word[:-1] if len(word) > 3 and word.endswith("s") else word
                    own_words |= {word, wordbase}

            def names_field(name: str) -> bool:
                core = [w for w in trait_terms(name) if len(w) >= 3 and w not in BOND_TERMS]
                if not core or not all(w in words_said for w in core):
                    return False
                return any(w not in own_words for w in core)
            if partner and not rigorous_trait(partner, own_prop) and names_field(partner):
                return True
            for row_text in str(field_menu or "").splitlines():
                field_key = row_text.strip().lstrip("-").strip().split(" (in ")[0].strip()
                if not field_key or field_key.endswith(":") or rigorous_trait(field_key, own_prop) or rubbish_slot(field_key):
                    continue
                if names_field(field_key):
                    return True
                tail = row_text.split("):", 1)[1] if "):" in row_text else ""
                for sample in tail.split("|"):
                    tokens = re.findall(r"[a-z0-9]+", sample.lower())
                    if not any(t not in own_words and t not in BOND_TERMS for t in tokens) or len("".join(tokens)) < 3:
                        continue
                    phrase = re.escape(" ".join(tokens))
                    if re.search(r"(?<![a-z0-9])" + phrase + r"(?![a-z0-9])", squashed):
                        return True
            return False
        except Exception:
            return False

    def couple_suits(self, lead: str, partner: str) -> bool:
        """CD: whether a second property may ride along with the first in the last message: same product (a single
        product, or the same "(product)"), and not a near-twin of the first - no equal or related name (strict_prop,
        same_prop, same_field), no shared name word, no shared sample values."""
        try:
            multi = bool(self.needs.items())
            idx = self.ware_belonging(lead) if multi else None
            if multi and (idx is None or self.ware_belonging(partner) != idx):
                return False
            if rigorous_trait(lead, partner) or identical_trait(lead, partner) or self.identical_attribute(idx, lead, partner):
                return False

            def core(name: str) -> set[str]:
                return {w for w in trait_terms(name) if len(w) >= 3 and w not in BOND_TERMS}
            if core(lead) & core(partner):
                return False
            if self.attribute_settings(idx, lead) & self.attribute_settings(idx, partner):
                return False
            # (EC4: nor a second that may stand for another open key of the product - twin_of - which the reply would
            # treat as one more subject: a longer name while its shorter twin is still open)
            memo = vars(self).get("menu_memo", {}).get(idx) or (0, "")
            for row_text in memo[1].splitlines():
                key = row_text.strip("- ").split(" (in ")[0].strip()
                if not key or rigorous_trait(key, lead) or rigorous_trait(key, partner):
                    continue
                if self.sibling_belonging(idx, key, partner) and self.scheme_examine(idx, key):
                    return False
            return True
        except Exception:
            return False

    def backlog_secondary(self, name: str) -> None:
        """Puts the unasked second property of a pair in the queue; it is sent alone as the next question."""
        waiting = vars(self).setdefault("pending_seconds", [])
        if not name:
            return
        if name.lower() in [p.lower() for p in waiting]:
            return
        waiting.append(name)

    def present_intended(self, urge: bool = False) -> bool:
        """Sends the next planned question, one per call: each is chosen only after the replies to the earlier ones were
        read, so a 'don't mind' or a revealed need guides the following pick. Asking begins once two listings' details
        were read (or straight away when something has to be learned from the shopper first)."""
        state = vars(self)
        perusal = self.needs.perusal
        channel = self.role("message")
        letter_lid = self.inquire_headroom()
        if not channel or self.letters_transmitted >= letter_lid:
            self.inquiring = False
            return False
        menu_text, _ = self.vault.trait_picklist()
        read_sets = self.vault.detailed_peruses()
        own_cat = self.native_rack()
        if own_cat:
            # (a single product with listings of other categories read too: only its own category's records build the
            # menu and trigger the questions, so a referenced listing of another kind neither lends fields nor fires them)
            menu_text = self.vault.trait_picklist(own_cat)[0]
            read_sets = self.vault.detailed_peruses(own_cat)
        pressing = bool(perusal.get("ask_first")) and not self.enquiries_journal
        if read_sets < INQUIRE_FOLLOWING_GROUPS and not pressing and not urge:
            early = self.protocol.pulse < INQUIRE_FOLLOWING_STAGES
            if early or (self.protocol.pulse < INQUIRE_FOLLOWING_STAGES + INQUIRE_LEEWAY_STAGES and self.shifts_remaining() > 20):
                return False
        reserve_hit = self.shifts_remaining() <= max(10, self.purchase_owed())
        if reserve_hit or self.protocol.surplus_alt() < WRAPUP + 40:
            self.inquiring = False       # (CF2: no more questions once the steps reserved for the order start)
            return False
        retry_at = state.get("plan_retry_at")
        if retry_at is not None and not urge:
            if self.protocol.pulse < retry_at:
                return False
            self.plan_retry_at = None       # (the retry falls due: plan again now, whatever was read meanwhile)
        else:
            hold_at = state.get("plan_wait")
            if hold_at is not None and read_sets <= hold_at and not urge:
                return False
        prior = state.get("asked_at_sets")
        if prior is not None and read_sets <= prior and not urge and not pressing:
            left = self.shifts_remaining()
            spare_steps = self.purchase_owed() + 2 * max(0, letter_lid - self.letters_transmitted) + 14
            # (EC2/MC6: with several products there is no such wait while the steps keep the +8 margin, which the +14
            # test implies; the next question waits for one more listing record, so it is drawn from more of the store)
            recent = self.protocol.pulse - state.get("asked_at_tick", 0) < 4
            if left > spare_steps and recent and not self.needs.items():
                return False
        if self.upfront_peruse():
            return True       # (the reads feeding the question menu go first, right now; the question comes after)
        if not urge and not pressing and not self.letters_transmitted and not self.enquiries_journal and self.first_reads():
            return True       # (EC2: every product's first 2 records are read before the first message; the question follows)
        left_msgs = letter_lid - self.letters_transmitted
        prompt = [INQUIRE_SCHEME.replace("QUESTION_COUNT", str(left_msgs)), SUBSEQUENT_SINGLE.replace("REMAINING", str(left_msgs)),
                 "\n\nTHE SHOPPER'S REQUEST:\n", self.request, "\n\nREQUIREMENTS SO FAR:\n", self.needs.passage_payload()]
        if len(self.enquiries_journal):
            prompt.append("\n\nALREADY ASKED (each with the shopper's reply):\n" + "\n".join(self.enquiries_journal))
        if perusal.get("ask_first"):
            prompt.append("\n\nMUST BE LEARNED FROM THE SHOPPER: " + "; ".join(str(h) for h in perusal.get("ask_first") or []))
        prompt.append(self.scheme_annotations(None))
        try:
            hints = self.examine_clues()
        except Exception:
            hints = []
        if hints:
            prompt.append("\n\nSTORY HINTS (story details from the request with the property each one seems to be about; "
                         "these are not requirements, yet the hidden need tends to sit on one of them):\n"
                         + "\n".join("- " + self.clue_flag(h) for h in hints[:8]))
        shown = self.picklist_prune(menu_text, read_sets)      # (EC7: the planner's menu; menu_text stays complete for the CD2 test)
        items = self.needs.items()
        foreign_only = False
        if items:
            products_line = " | ".join(f"{n + 1}) {row['product']} x{row['quantity']}" for n, row in enumerate(self.needs.items()))
            prompt.append("\n\nTHE PRODUCTS ASKED FOR: " + products_line)
            read_cats = self.vault.racks_peruse()
            shelf_sets = self.landing_racks()
            every_home = set().union(*shelf_sets)
            own_cats = [s for s in read_cats if s.strip().lower() in every_home]
            foreign_only = not own_cats      # (only other categories were read - a pointer's listings: no product field is known)
            # (the products' own categories go first: a referenced listing's category adds no fields while each
            # product has one of its own)
            if all(shelf_sets):
                cat_order = own_cats or read_cats
            else:
                cat_order = own_cats + [s for s in read_cats if s not in own_cats]
            cat_menus = [(shelf, self.vault.trait_picklist(shelf)[0]) for shelf in cat_order[:4]]
            full_menus = [f"CATEGORY {shelf}:\n{menu}" for shelf, menu in cat_menus if menu]
            menu_text = "\n".join(full_menus) or menu_text
            trim_menus = [f"CATEGORY {shelf}:\n{self.picklist_prune(menu, self.vault.detailed_peruses(shelf))}" for shelf, menu in cat_menus if menu]
            shown = "\n".join(trim_menus) or shown
        prompt.append("\n\nPROPERTIES IN THE LISTINGS' DETAILS SEEN SO FAR (name, how many listings, sample values):\n"
                     + (shown or "(none seen yet)"))
        if not shown or foreign_only:
            # (no detail record of the wanted product was read yet - a pointer chain or other categories took the reads:
            # an empty plan would leave the questions unasked, so the planner names likely properties itself)
            prompt.append("\nNo field names of the wanted product are known yet, so the key rule above cannot apply this time: "
                          "name instead a property that listings of this kind of product commonly carry (by its usual plain "
                          "name), the one that a STORY HINT or a "
                          "story detail of the request points to first, and leave the requirements already set alone. Do "
                          "not return an empty list while questions are left.")
        ask_text = "".join(prompt)
        asked_names = set()
        for entry in self.enquiries_journal:
            for part in entry.split(":", 1)[0].strip("- ").split(";"):
                asked_names.add(part.strip().lower())

        def is_fresh(name_in: Any) -> bool:
            # (a property asked once is never asked again: a 'don't mind' or a confused reply stays its answer; nor a
            # requirement in force, nor a property the shopper does not mind, by name for the same product; nor a story
            # hint's plain name once its product's records show no field of that name)
            name_in = str(name_in or "").strip()
            if not name_in or name_in.lower() in asked_names or rubbish_slot(name_in):
                return False
            if name_in.lower() in vars(self).get("spent_asks", set()):
                return False      # (CD: nor one whose plain question drew a one-question-at-a-time reply)
            idx = self.ware_belonging(name_in) if self.needs.items() else None
            return not self.simple_clue(idx, name_in) and self.scheme_examine(idx, name_in)

        def tag_of(raw: str) -> str:
            tag = re.search(r"\(([^)]+)\)", raw)
            return tag.group(1).strip() if tag else str(perusal.get("product") or "item")

        def plain_q(name: str, item_name: str) -> str:
            return f"Which {name.split('(')[0].strip()} would you like for the {item_name}?"

        def is_blank(raw: str) -> Any:
            return re.search(r"none|^-$|n/a", raw.lower())

        def push_front(name_in: str) -> None:
            low = name_in.lower()
            rest = [p for p in vars(self).get("pending_seconds") or [] if p.lower() != low]
            self.pending_seconds = [name_in] + rest

        def remember_open(name_in: str) -> None:
            pairs = vars(self).setdefault("open_pair", [])
            if name_in.lower() not in [p.lower() for p in pairs]:
                pairs.append(name_in)
        # (an earlier pair's second property, left unsettled by its reply, is sent next on its own; it stays queued
        # until sent, and one that is no longer open drops out of the queue)
        waiting = [p for p in state.get("pending_seconds") or [] if is_fresh(p)]
        self.pending_seconds = waiting
        queued = waiting[0] if waiting else ""
        if queued:
            item_name = tag_of(queued)
            own_text = (state.get("pending_text") or {}).pop(queued.lower(), "")      # (EC4: a distinct re-ask keeps its text)
            plan_reply = {"questions": [{"property": queued, "question": own_text or plain_q(queued, item_name)}]}
        else:
            plan_reply = self.protocol.confer(ask_text, 2500, STRATEGIST)
            if plan_reply is None and self.protocol.misses > 0:
                # (TA4: the planner call was lost to load or the network: no plan proves nothing - planned again after
                # another listing is opened, and not counted as an empty plan)
                self.plan_wait = read_sets
                return False
        proposed_alt = [q for q in (plan_reply or {}).get("questions") or [] if isinstance(q, dict) and str(q.get("question") or "").strip()]
        plan_items = []
        for qitem in proposed_alt:
            other = str(qitem.get("second") or "").strip()
            if is_blank(other):
                other = ""
            if not is_fresh(qitem.get("property")):
                if other and is_fresh(other):
                    # (its first property was asked or is settled: the plan's unasked second goes out by itself)
                    plan_items.append({"property": other, "question": plain_q(other, tag_of(other))})
                continue
            plan_items.append(dict(qitem, second=other if other and is_fresh(other) else ""))
            if other and self.simple_clue(self.ware_belonging(other) if self.needs.items() else None, other):
                # (its second is a story hint's plain name that no field carries: the question asks its first alone)
                head = str(qitem.get("property") or "").strip()
                plan_items[-1]["question"] = plain_q(head, tag_of(head))
        if self.needs.items() and not queued:      # (CD: a queued re-ask goes first; the rotation follows it)
            plan_items = self.allocation_initial(plan_items, is_fresh)
            if state.pop("quota_wait", False):
                return True       # (reads queued for an unread product go first, right now; the question comes after)
        ask_call = None
        held_second = ""
        for item in plan_items[:2]:
            multi_now = bool(self.needs.items())
            if multi_now:
                # (several products: a question names the product its property belongs to, or the reply is a confused one)
                name_in = str(item.get("property") or "")
                tag = re.search(r"\(([^)]+)\)", name_in)
                item_name = tag.group(1).strip() if tag else ""
                pwords = re.findall(r"[a-z]{3,}", item_name.lower())
                if item_name and pwords:
                    asked_text = str(item.get("question") or "").lower()
                    if not any(w.rstrip("s") in asked_text for w in pwords):
                        item = dict(item, question=plain_q(name_in, item_name))
            if not queued and not (perusal.get("ask_first") and not self.enquiries_journal):
                if self.monikers_peer(str(item.get("question") or ""), str(item.get("property") or ""), menu_text,
                                    str(item.get("second") or "")):
                    # (CD: the planner's wording may name a second subject - another field's name or one of its sample
                    # values as an example: the property is sent alone in the plain form, property plus product)
                    name_in = str(item.get("property") or "")
                    item = dict(item, question=plain_q(name_in, tag_of(name_in)))
            call_id = f"ask_{self.letters_transmitted + 1}_{self.protocol.pulse}"
            ask_call = self.missive_invoke(str(item.get("question")).strip(), call_id)
            if not ask_call:
                continue
            sent = (str(item.get("property")), str(item.get("question")).strip())
            other = str(item.get("second") or "").strip()
            if other and other.lower() != sent[0].lower() and not is_blank(other):
                # (with few messages left there is one property per question: a refused pair would cost too much)
                single = plain_q(sent[0], tag_of(sent[0]))
                ask_call = self.missive_invoke(single, f"ask_{self.letters_transmitted + 1}_{self.protocol.pulse}")
                sent = (sent[0], single)
                if not ask_call:
                    continue
                held_second = other
            break
        if not ask_call:
            if (not shown or foreign_only) and self.shifts_remaining() > 12 and not urge and state.get("blind_retries", 0) < 4:
                # (no field of the wanted product was known: the wanted product's records may come late - behind a pointer
                # chain - so the plan is drawn again two steps on rather than after the next record of its own category)
                self.blind_retries = state.get("blind_retries", 0) + 1
                self.plan_retry_at = self.protocol.pulse + 2
                return False
            if read_sets < INQUIRE_FOLLOWING_GROUPS and self.shifts_remaining() > 12 and not urge:
                self.plan_wait = read_sets     # (planned again after another listing is opened)
                return False
            try:
                thin_products = [idx for idx in range(len(self.needs.items()) or 1) if self.ware_peruses(idx if self.needs.items() else None) < 2]
            except Exception:
                thin_products = []
            if thin_products and self.shifts_remaining() > 12 and not urge:
                # (an empty plan while some product's own listings are still unread settles nothing: plan again later)
                self.plan_wait = read_sets
                return False
            self.empty_plans = state.get("empty_plans", 0) + (1 if shown and not foreign_only else 0)     # (a plan drawn with no field known yet does not count)
            if self.empty_plans <= 1 and self.shifts_remaining() > 12:
                # (messages and steps remain: plan again once another listing is opened, rather than never)
                self.plan_wait = read_sets
                return False
            self.inquiring = False
            return False
        self.asked_at_sets, self.asked_at_tick = read_sets, self.protocol.pulse
        self.converse.append({"role": "assistant", "content": f"Next I put a question to the shopper about {sent[0]}.",
                          "tool_calls": [ask_call]})
        reveals_before = len(self.needs.uncovered)
        frees_before = len(self.needs.indifferent) + self.needs.__dict__.get("pinned_heard", 0)
        self.asking_label = sent[0]       # (CE2b: the property being asked, while its reply is read)
        try:
            replies = self.route_alt([ask_call], scripted=True)
        finally:
            self.asking_label = ""
        self.later_responses(replies)
        asked_tag = sent[0]
        if queued and sent[0].split(";")[0].strip().lower() == queued.lower():
            low = queued.lower()
            self.pending_seconds = [p for p in state.get("pending_seconds") or [] if p.lower() != low]
        note_tail = ""
        meta_heard = self.needs.__dict__.pop("meta_last", None)

        def compress(raw: Any) -> str:
            return " ".join(re.findall(r"[a-z0-9]+", str(raw).lower()))

        def frees_now() -> int:
            return len(self.needs.indifferent) + self.needs.__dict__.get("pinned_heard", 0)
        paired = ";" in sent[0]
        sent_flat = compress(sent[1])
        if meta_heard is not None and sent_flat and sent_flat in compress(meta_heard):
            # (CD: the reply only asked for one question at a time - nothing got answered: pairs are off; the first
            # property is sent again by itself in the plain form as the next message (once, and not when that very text
            # drew the reply: such a property is not planned again), and a second stays open for the planner)
            self.without_couples = True
            subjects = [p.strip() for p in sent[0].split(";") if p.strip()] or [sent[0]]
            lead = subjects[0]
            first_low = lead.lower()
            item_name = tag_of(lead)
            reasked = state.setdefault("reasked", set())
            plain_text = f"Which {lead.split('(')[0].strip()} would you like for the {item_name}?"
            solo = len(subjects) == 1      # (a pair's one-at-a-time reply may stem from its two subjects: its first goes alone)
            other = self.overlap_variant(lead) if solo else ""
            if other:
                # (v605f FC1: the story hint's '(or X)' alternative goes next, ahead of any re-ask of the same subject)
                reasked.add(first_low)
                state.setdefault("spent_asks", set()).add(first_low)
                push_front(other)
                note_tail = f" [{lead}: not answered; the story hint's alternative {other} is asked next]"
            elif first_low not in reasked and compress(plain_text) != compress(sent[1])\
                    and not (solo and self.keeps_simple(sent[1], plain_text)):
                # (v605f FC2: not when the sent text already used the plain form's words - the same subjects again)
                reasked.add(first_low)
                push_front(lead)
                note_tail = f" [{lead}: not answered; asked again by itself next]"
            else:
                # (EC4/MC3: the plain text itself drew it - the property's name also stands for a second subject: once per
                # product the next message asks the story twin, else the property by its own values; then it is spent)
                re_ask = self.unique_inquire(lead) if first_low not in reasked else None
                if not re_ask:
                    state.setdefault("spent_asks", set()).add(first_low)
                    note_tail = f" [{lead}: not answered; not asked again]"
                else:
                    target, wording = re_ask[0], re_ask[1]
                    reasked.update({first_low, target.lower()})      # (a one-at-a-time reply to the re-ask spends it)
                    state.setdefault("pending_text", {})[target.lower()] = wording
                    push_front(target)
                    same_subject = re.split(r"\s*\(", target)[0].strip().lower() == re.split(r"\s*\(", lead)[0].strip().lower()
                    if not same_subject:
                        state.setdefault("spent_asks", set()).add(first_low)
                    tail = "by its values next]" if same_subject else f"as {target} next]"
                    note_tail = f" [{lead}: not answered; asked again " + tail
            state.setdefault("open_pair", [])
            for extra in subjects[1:] + ([held_second] if held_second else []):
                remember_open(extra)
            sent = (" and ".join(subjects) + " [not answered] [one at a time]", sent[1])
        elif paired and len(self.needs.uncovered) > reveals_before:
            # (a reply reveals one need at most: after a reveal the second property is still unasked)
            other = sent[0].split(";", 1)[1].strip()
            reveal_words = " ".join(self.needs.uncovered[reveals_before:]).lower()
            if not any(w in reveal_words for w in re.findall(r"[a-z]{4,}", other.split("(")[0].lower())):
                self.backlog_secondary(other)
                note_tail = f" [{other}: not answered, still open]"
            sent = (sent[0].split(";")[0].strip(), sent[1])
        elif paired and frees_now() > frees_before:
            # (a "don't mind" to a pair settles only its first property; the reply skipped the second, which stays open
            # and is listed for the planner to rank - a shopper may well have answered it too)
            other = sent[0].split(";", 1)[1].strip()
            sent = (sent[0].split(";")[0].strip(), sent[1])
            freed_names = (other.lower(), other.split("(")[0].strip().lower())
            self.needs.indifferent = [p for p in self.needs.indifferent if p.strip().lower() not in freed_names]
            remember_open(other)
            note_tail = f" [{other}: not answered, still open]"
        elif paired and frees_now() == frees_before:
            # (a paired question the reply left unanswered: pairs are off, and both properties stay open)
            self.without_couples = True
            state.setdefault("open_pair", [])
            for piece in sent[0].split(";"):
                piece = piece.strip()
                if piece:
                    remember_open(piece)        # (CD: both stay open and are listed for the planner)
            sent = (sent[0].replace(";", " and") + " [not answered]", sent[1])
        elif held_second:
            # (the planner's second, split off because few messages remain, stays open and is offered to the planner
            # to rank next, unless the reply settled it)
            reveal_words = " ".join(self.needs.uncovered[reveals_before:]).lower()
            state.setdefault("open_pair", [])
            if not any(w in reveal_words for w in re.findall(r"[a-z]{4,}", held_second.split("(")[0].lower())):
                remember_open(held_second)
        if meta_heard is None and len(self.needs.uncovered) == reveals_before:
            # (EC3: no reveal came from the property asked - a no-preference or confused reply: a story hint of it may
            # come back under its own alternative; a reply revealing nothing means the key was wrong for that product,
            # so the next question moves on)
            state.setdefault("quiet_asks", []).append(asked_tag.split(";")[0].strip())
        if replies:
            reply = replies[0][1].split("\n[status]")[0][:400]
            self.enquiries_journal.append("- " + sent[0] + ': asked "' + sent[1] + '" -> ' + reply + note_tail)
        self.discard_desired(asked_tag)
        self.inquiring = self.letters_transmitted < letter_lid
        return True

    @staticmethod
    def numericval_slots(field_menu: str) -> set[str]:
        """Menu keys whose values are figures carrying a unit, normalised for comparison (prop_key)."""
        found = set()
        for row_text in str(field_menu).splitlines():
            if "[figures with a unit]" not in row_text:
                continue
            match = re.match(r"\s*-\s*(.+?)\s+\(in \d+ listings?\)", row_text)
            if match:
                found.add(trait_slot(match.group(1)))
        return found

    def measured_last(self, plan_list: list[dict[str, Any]], field_menu: str, asked_props: set[str], left: int, has_pending: bool) -> list[dict[str, Any]]:
        """For the final question: when no measured property (a key whose values are figures with a unit) has been asked
        and the plan offers one as its second choice while its first is not one, the measured one is sent first."""
        if has_pending or not plan_list or left != 1:
            return plan_list
        unit_keys = self.numericval_slots(field_menu)
        if not unit_keys:
            return plan_list
        if any(trait_slot(p) in unit_keys for p in asked_props):
            return plan_list
        top = plan_list[0]
        backup = str(top.get("second") or "").strip()
        if trait_slot(top.get("property")) in unit_keys:
            return plan_list
        if not backup or trait_slot(backup) not in unit_keys:
            return plan_list
        lead_prop = str(top.get("property") or "")
        tag = re.search(r"\(([^)]+)\)", backup) or re.search(r"\(([^)]+)\)", lead_prop)
        product = tag.group(1).strip() if tag else str(self.needs.perusal.get("product") or "item")
        self.journal.append({"kind": "measured_last", "asked": backup, "instead_of": lead_prop[:80]})
        swapped = {"property": backup, "second": "", "question": f"Which {backup.split('(')[0].strip()} would you like for the {product}?"}
        return [swapped] + plan_list

    def response_returned(self, body: str) -> bool:
        if not self.role("message") or self.letters_transmitted >= self.inquire_headroom():
            return False
        if self.shifts_remaining() <= 4 or self.protocol.surplus_alt() < WRAPUP:
            return False
        if self.inquiring and self.letters_transmitted >= self.inquire_headroom() - 1 and self.shifts_remaining() > 10\
                and self.protocol.surplus_alt() >= WRAPUP + 40:
            return False
        msg_call = self.missive_invoke(body, f"reply_{self.protocol.pulse}")
        if not msg_call:
            return False
        self.converse.append({"role": "assistant", "content": "Replying to the shopper's question.", "tool_calls": [msg_call]})
        answers = self.route_alt([msg_call], scripted=True)
        self.later_responses(answers)
        return True

    def later_responses(self, replies: list[tuple[str, str]]) -> None:
        final = len(replies) - 1
        for idx, (call_id, body) in enumerate(replies):
            content = body + "\n" + self.progress_lineitem() if idx == final else body
            self.converse.append({"role": "tool", "tool_call_id": call_id, "content": content})

    # ---------------------------------------------------------------------------------------------- closing out


class EnginePilot:
    """Errand part: building arguments and the steps the code drives itself: reading, roles, queries, picks, the cart."""

    def simple_params(self, tool_name: str, body: str, purpose: str = "") -> dict[str, Any]:
        schema = self.structures.get(tool_name) or {}
        fields = schema.get("properties") or {}
        args_out: dict[str, Any] = {}
        if purpose == "decline":
            cue = ABSTAIN_TERMS
        elif purpose == "order":
            cue = PURCHASE_TERMS
        else:
            cue = ""

        def clip(spec: dict[str, Any]) -> str:
            return body[: spec.get("maxLength", len(body))]
        for field in schema.get("required") or []:
            spec = fields.get(field) or {}
            typed = spec.get("type")
            if spec.get("enum"):
                options = spec["enum"]
                matching = [v for v in options if cue and isinstance(v, str) and re.search(cue, v.lower())]
                args_out[field] = matching[0] if matching else options[0]
            elif typed == "string" or "string" in (typed or []):
                args_out[field] = clip(spec)
            elif typed == "boolean":
                args_out[field] = True
            elif typed in ("integer", "number"):
                args_out[field] = max(1, spec.get("minimum", 1))
        if len(fields) == 1:
            only, spec = list(fields.items())[0]
            if only not in args_out and isinstance(spec, dict) and spec.get("type") == "string" and "enum" not in spec:
                args_out[only] = clip(spec)
        return args_out

    def cleanse_abstain(self, args_in: dict[str, Any], tool_name: str) -> dict[str, Any]:
        no_names = self.needs.perusal.get("decline_no_names")
        if no_names is False:
            return args_in
        splitter = r"[\s.,;:!?()\[\]{}<>\"'`]+"
        texts = [v for v in args_in.values() if isinstance(v, str)]
        names_id = any(tok and tok in self.vault.ids for text in texts for tok in re.split(splitter, text))
        if not no_names and not names_id:
            return args_in
        fields = (self.structures.get(tool_name) or {}).get("properties") or {}
        cleaned: dict[str, Any] = {}
        for key, val in args_in.items():
            spec = fields.get(key) or {}
            if isinstance(val, str) and "enum" not in spec:
                cleaned[key] = APOLOGY[: spec.get("maxLength", len(APOLOGY))]
            else:
                cleaned[key] = val
        return cleaned

    # ---------------------------------------------------------------------------------------------- code only (TA)
    def code_reading(self) -> dict[str, Any]:
        """TA5: a plan built from the request's own words and the tools' schemas, for an errand whose request no model
        managed to read: each tool's role, the budget (code_budget), stock, a cheapest rule, limits on messages and
        steps (code_limit), and the products: the shelf labels the request writes with a count when it names two or
        more, otherwise the label sharing most words with the request, each with its units and requirements
        (code_needs)."""
        source = self.request
        lowered = source.lower()
        plan: dict[str, Any] = {
            "code_reading": True, "requirements": [], "items": [], "quantity": 1, "rules": [], "soft_wants": [],
            "ask_first": [], "hints": [], "references": [], "priority_detail": "",
        }
        cap_figure, cap_unit = self.engine_allotment(source)
        plan["price_cap"] = cap_figure
        plan["currency"] = cap_unit
        stock_cue = (r"\bin[- ]stock\b|\bon hand\b|\bon the shelf\b|\bsold[- ]out\b|\bback[- ]?order|"
                     r"\bout of stock\b|\bavailable now\b")
        plan["in_stock_only"] = bool(re.search(stock_cue, lowered))
        cheap_cue = (r"\bcheapest\b|\blowest[- ](?:price|priced|cost)\b|\bleast expensive\b|"
                     r"\blowest\b[^.]{0,20}\bprice\b|\bsmallest (?:total|price)\b")
        plan["priority"] = "cheapest" if re.search(cheap_cue, lowered) else "none"
        letter_lid = self.engine_threshold(lowered, r"messages?|questions?|notes?|repl(?:y|ies)|texts?")
        if letter_lid is not None:
            plan["message_budget"] = letter_lid
        stage_lid = self.engine_threshold(lowered, r"steps?|moves?|actions?|(?:tool )?calls?|turns?")
        if stage_lid is not None and stage_lid >= INITIAL_PURCHASE_VIA + 3:
            plan["step_budget"] = stage_lid
        tool_roles = self.engine_duties()
        plan["roles"] = tool_roles
        found = self.engine_entries(tool_roles.get("search"))
        if len(found) < 2:
            # (a single product: the shelf label written with a count, else the one sharing most words with the request)
            if found:
                plan["product"] = found[0]["label"]
                plan["quantity"] = found[0]["quantity"]
            else:
                plan["product"] = self.engine_tag(tool_roles.get("search"), category_terms(source))
                plan["quantity"] = self.engine_headcount(lowered)
            plan["requirements"] = [f"{prop}: {value}" for prop, value in self.engine_requirements(lowered)]
            return plan
        plan["items"] = [{field: entry[field] for field in ("product", "quantity", "requirements")} for entry in found]
        plan["product"] = " and ".join(entry["product"] for entry in found)
        plan["requirements"] = [req for entry in found for req in entry["requirements"]]
        return plan

    @staticmethod
    def engine_allotment(text: str) -> tuple[Any, Any]:
        """TA5: (figure, unit) of the spending limit a text states, else (None, None). Every figure that is not a count of
        messages, steps or units and not part of a run of figures is scored from the words around it: a spending word or
        a currency sign right before it 3, a limit cue right before it ('under', 'up to', 'at most', 'to', 'at') 2, a
        limit cue right after it ('or lower', 'at most', 'and not a bit more') 2, a sentence about the whole order 1;
        the top score of 2 or more wins (the larger figure on a tie), and its unit is the one or two words after it."""
        top: tuple[int, float] = (0, 0.0)
        result: tuple[Any, Any] = (None, None)
        counted = (r"\s*(?:x\b|×|%|:\s*\d|-\s*\d|/\s*\d|(?:[a-z-]+\s+){0,2}?(?:messages?|questions?|notes?|repl\w*|texts?|"
                   r"steps?|moves?|actions?|(?:tool )?calls?|turns?|units?|items?|pieces?|packs?|pairs?|lots?|sets?|of\b|times?)\b)")
        spend_cue = r"\b(?:spend\w*|budget\w*|afford\w*|pay\w*|price[ds]?|costs?|cash|money)\b(?:\W+[\w']+){0,4}\W*$"
        limit_before = (r"\b(?:under|below|at most|no more than|not more than|up to|within|max\w*|less than|cheaper than|"
                        r"not (?:more|over|above)|limit\w*|reach\w*|go to|to|at|cap\w*|past|exceed\w*|beyond|go over)\W+(?:[\w']+\W+)?$")
        limit_after = (r"\s*(?:[a-z.()]+\s*){0,3}?(?:or (?:lower|less|under|below|cheaper)|at (?:the (?:very )?)?most|max\w*|tops\b|"
                       r",?\s*(?:and )?not (?:one|a) \w+ (?:over|more|above))")
        whole_cue = r"\b(?:order|total|whole|lot|sum|bill|altogether|overall|everything)\b"
        for sentence in re.split(r"(?<=[.!?;])\s+", str(text).lower()):
            base = 1 if re.search(whole_cue, sentence) else 0
            for hit in re.finditer(r"([$€£]\s*)?(\d[\d,]*(?:\.\d+)?)", sentence):
                sign_part = hit.group(1)
                rest = sentence[hit.end():]
                if re.match(counted, rest):
                    continue
                if not sign_part and re.search(r"[\d.,:/-]\s*$|\bx\s*$", sentence[:hit.start()]):
                    continue          # (one figure of a run: 1920 x 1080, 4000:1)
                lead_in = sentence[max(0, hit.start() - 60): hit.start()]
                points = base
                if sign_part or re.search(spend_cue, lead_in):
                    points += 3
                if re.search(limit_before, lead_in):
                    points += 2
                if re.match(limit_after, rest):
                    points += 2
                value = float(hit.group(2).replace(",", ""))
                if points < 2 or (points, value) <= top:
                    continue
                top = (points, value)
                unit_hit = re.match(r"\s*((?:[a-z][a-z.]*\s*){0,2})", rest)
                raw = unit_hit.group(1).split() if unit_hit else []
                unit_words = []
                for w in raw:
                    bare = w.strip(".")
                    if bare not in BOND_TERMS and len(bare) >= 2 and not re.search(r"\d", w):
                        unit_words.append(bare)
                symbol = (sign_part or "").strip()
                result = (int(value) if value.is_integer() else value, " ".join(unit_words) or symbol or None)
        return result

    @staticmethod
    def engine_threshold(lowered: str, noun_rx: str) -> int | None:
        """TA5: a cap the request puts on messages or steps (`noun_rx`): a count of them with a limit cue just before
        it ('at most', 'no more than', 'up to', 'only', 'send', 'use', 'take', 'receive', 'allow', 'inside', 'within',
        'total of', "you've") or just after it ('in total', 'no more', 'at most', 'max'); a count lacking one ('my 3 step
        routine', 'with 2 notes of rose') belongs to the story."""
        figures = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "twelve": 12, "fifteen": 15, "twenty": 20,
        }
        counter = (r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen|twenty)\s+(?:[a-z-]+\s+){0,2}?(?:"
                   + noun_rx + r")\b")
        cue_before = (r"\b(?:at most|no more than|not more than|up to|only|max\w*|use|using|take|send|allow\w*|"
                      r"receive|get|total of|limit\w*|inside|within|under|fewer than|less than|you've|you have|have|"
                      r"keep\w*|stick to|go past|past|exceed\w*|beyond)\W+(?:[\w']+\W+){0,3}$")
        cue_after = r"\W*(?:[\w']+\W+){0,3}?(?:in total|total|no more|at most|max\w*|tops|or fewer|or less|allowed|limit)\b"
        for hit in re.finditer(counter, lowered):
            lead_in = lowered[max(0, hit.start() - 60): hit.start()]
            trail = lowered[hit.end(): hit.end() + 40]
            if not re.search(cue_before, lead_in) and not re.match(cue_after, trail):
                continue
            said = hit.group(1)
            return int(said) if said.isdigit() else figures[said]
        return None

    @staticmethod
    def engine_headcount(lowered: str) -> int:
        """TA5: how many units of one listing a one-product request wants: a count before 'units', 'of them', 'copies';
        otherwise 1."""
        figures = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        }
        hit = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:units?|of (?:them|those|these)|copies)\b", lowered)
        if hit is None:
            return 1
        said = hit.group(1)
        total = int(said) if said.isdigit() else figures[said]
        if total < 1 or total > 99:
            return 1
        return total

    @staticmethod
    def engine_requirements(stretch: str) -> list[tuple[str, str]]:
        """TA5: (property, value) for each requirement a stretch of the request writes as 'where/whose (the) P is V' or
        '(and) the P has to / must / should be V'; the value stops at a comma or a joining word ('so', 'since', 'and
        the') and is four words at most; a property naming a person, or a value that reads as story (opening with a
        preposition, a pronoun or a verb such as 'for', 'my', 'see') or that points at another listing, is skipped."""
        found: list[tuple[str, str]] = []
        linking = r"(?:is|are|has to be|have to be|must be|should be|needs to be|need to be|can be)"
        stop = (r"(?=\s*(?:[,.;:!?()]|$)|\s+(?:so|since|as|because|for|to|which|that|nothing|otherwise|but|when|while|if|then|"
                r"though|please|and (?:the|its|their|a|an|one|two|three|\d))\b)")
        opener = (r"(?:\b(?:where|whose)\s+(?:the\s+|its\s+|their\s+)?|(?:\b(?:and|with|but)\s+|[,;:]\s*|^|[.!?]\s+)"
                  r"(?:the|its|their)\s+)")
        pattern = opener + r"([a-z][a-z'&/ -]{1,32}?)\s+" + linking + r"\s+(?:either\s+|just\s+)?([^,.;:!?()]{1,60}?)" + stop
        person = (r"\b(?:i|you|we|it|he|she|they|my|your|our|there|this|that|which|what|who|me|us|them|one|ones|price|budget|"
                  r"cost|total|order|stock)\b")
        story = (r"(?:a|an|the|my|your|our|his|her|their|me|us|for|to|in|on|at|of|see|go|get|be|have|only|too|very|so|not|no|"
                 r"what|how|it|all|mostly|always|never|just|really|still|also|from|about|like|one|ones)\b")
        for hit in re.finditer(pattern, stretch):
            pname = hit.group(1).strip()
            pvalue = hit.group(2).strip()
            after = stretch[hit.end(): hit.end() + 60]
            if reference_passage(pvalue) or (reference_passage(after) and re.search(r"\b(?:page|listing|shown|sold by)\b", after)):
                continue
            if not pvalue or len(pvalue.split()) > 4 or len(pname.split()) > 4:
                continue
            if re.search(person, pname) or re.match(story, pvalue):
                continue
            found.append((pname, pvalue))
        return found
    def engine_entries(self, tool_arg: str | None) -> list[dict[str, Any]]:
        """TA5: every product the request names together with a count, in the order written. A product is a shelf label
        of the search tool whose words appear consecutively in the request (ignoring link words and a trailing plural s,
        so 'bowls and plates' matches 'Bowl & Plates'), preceded by a count ('3 x', 'two', '2 pairs of'; up to four
        non-unit words in between). Each comes with its units and the requirements written after it, up to the next
        product or the end of the sentence. Empty when no label carries a count."""
        arg_map = (self.structures.get(tool_arg or "") or {}).get("properties")
        enum_lists = []
        if isinstance(arg_map, dict):
            for a_spec in arg_map.values():
                if isinstance(a_spec, dict) and isinstance(a_spec.get("enum"), list) and len(a_spec["enum"]) >= 4:
                    enum_lists.append(a_spec["enum"])
        labels = [str(e) for e in max(enum_lists, key=len)] if enum_lists else []
        said = self.request.lower()
        marks = []
        for mt in re.finditer(r"[a-z]{3,}", said):
            if mt.group(0) in BOND_TERMS:
                continue
            marks.append((mt.group(0).rstrip("s"), mt.start(), mt.end()))
        word_seq = [stem_w for stem_w, _, _ in marks]
        word_counts = dict(zip("one two three four five six seven eight nine ten".split(), range(1, 11)))
        spotted: dict[str, tuple[int, int, int]] = {}
        for lab in labels:
            lab_words = [w.rstrip("s") for w in re.findall(r"[a-z]{3,}", lab.lower()) if w not in BOND_TERMS]
            if not lab_words or lab in spotted:
                continue
            span_len = len(lab_words)
            for at in range(len(word_seq) - span_len + 1):
                if word_seq[at: at + span_len] != lab_words:
                    continue
                begin = marks[at][1]
                finish = marks[at + span_len - 1][2]
                if len(re.findall(r"[a-z0-9]+", said[begin:finish])) > 2 * span_len:
                    continue
                # (a unit standing between count and label voids it: '16 gb of memory' is no count of memory)
                window = said[max(0, begin - 56): begin]
                count_m = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:x\s+|×\s*)?(?:(?:of|[a-z'&-]{3,})\s+){0,4}$",
                                 window)
                if not count_m:
                    continue
                num_txt = count_m.group(1)
                measures = int(num_txt) if num_txt.isdigit() else word_counts[num_txt]
                if 1 <= measures <= 99:
                    spotted[lab] = (begin, finish, measures)
                break
        in_order = sorted(spotted.items(), key=lambda pair: pair[1][0])
        items_out = []
        for j, (lab, (_, seg_from, measures)) in enumerate(in_order):
            final = j == len(in_order) - 1
            seg_to = len(said) if final else in_order[j + 1][1][0]
            segment = said[seg_from: seg_to]
            if final:
                segment = re.split(r"(?<=[.!?])\s", segment)[0]
            prod = lab.lower()
            needs_here = [f"{pname} ({prod}): {pval}" for pname, pval in self.engine_requirements(segment)]
            items_out.append({"product": prod, "label": lab, "quantity": measures, "requirements": needs_here})
        return items_out

    def engine_duties(self) -> dict[str, str]:
        """TA5: which tool plays which role, judged from tool names, descriptions and argument schemas. A tool whose
        required choice offers both an order word and a decline word finishes either way; finishing words in a name
        come next; a cart-like tool taking arguments adds to the cart, one taking none shows it; after that the search
        tool (a search word plus free text or a choice, and not something that talks to the shopper), the message tool
        (a person plus a send/reply word, exactly one required text), the details tool and the view tool. Apart from
        the order/decline pair, a tool never holds two roles."""

        def describe(fdef: dict[str, Any]) -> str:
            split_name = re.sub(r"([a-z])([A-Z])", r"\1 \2", fdef["name"]).replace("_", " ").lower()
            return split_name + " | " + str(fdef.get("description") or "").lower()
        said = {entry["function"]["name"]: describe(entry["function"]) for entry in self.toolkit}

        def params_of(tool_id: str) -> dict[str, Any]:
            props_here = (self.structures.get(tool_id) or {}).get("properties")
            return props_here if isinstance(props_here, dict) else {}

        def musts_of(tool_id: str) -> list[str]:
            found = (self.structures.get(tool_id) or {}).get("required")
            if not isinstance(found, list):
                return []
            return [a_name for a_name in found if a_name in params_of(tool_id)]

        def text_args_of(tool_id: str) -> list[str]:
            return [a_name for a_name, a_spec in params_of(tool_id).items()
                    if isinstance(a_spec, dict) and a_spec.get("type") == "string" and not a_spec.get("enum")]
        jobs: dict[str, str] = {}
        for tool_id in said:
            for a_name in musts_of(tool_id):
                spec_here = params_of(tool_id)[a_name]
                choices = [str(c).lower() for c in (spec_here.get("enum") or [])] if isinstance(spec_here, dict) else []
                if not any(re.search(PURCHASE_TERMS, c) for c in choices):
                    continue
                if any(re.search(ABSTAIN_TERMS, c) for c in choices):
                    jobs.setdefault("order", tool_id)
                    jobs.setdefault("decline", tool_id)
        for tool_id, text_of in said.items():
            if tool_id in jobs.values():
                continue
            name_part = text_of.partition(" | ")[0]
            if "decline" not in jobs and re.search(ABSTAIN_TERMS, name_part):
                jobs["decline"] = tool_id
            elif "order" not in jobs and re.search(PURCHASE_TERMS, name_part):
                jobs["order"] = tool_id
        for tool_id, text_of in said.items():
            if tool_id in jobs.values() or musts_of(tool_id) or not re.search(r"\bfinal\b", text_of):
                continue
            if "decline" not in jobs and re.search(ABSTAIN_TERMS, text_of):
                jobs["decline"] = tool_id
            elif "order" not in jobs and re.search(PURCHASE_TERMS, text_of):
                jobs["order"] = tool_id
        for tool_id, text_of in said.items():
            if tool_id in jobs.values() or not re.search(r"\b(?:cart|basket|bag|trolley)\b", text_of):
                continue
            has_array = any(isinstance(a_spec, dict) and a_spec.get("type") == "array" for a_spec in params_of(tool_id).values())
            jobs.setdefault("cart" if musts_of(tool_id) or has_array else "cart_view", tool_id)
        who_pat = r"\b(?:shoppers?|customers?|buyers?|clients?|users?)\b"
        talk_pat = r"\b(?:send\w*|repl\w*|notes?|tell\w*|messag\w*|chat\w*|writ\w*|talk\w*)\b"

        def fits_role(job: str, tool_id: str, text_of: str, single_text: bool) -> bool:
            to_person = re.search(who_pat, text_of)
            if job == "search" and to_person and re.search(talk_pat, text_of):
                return False        # (whatever it claims to find, a tool that writes to the shopper is not a search)
            if job == "message" and not to_person and not re.search(r"\b(?:messag\w*|chat\w*)\b", text_of):
                return False
            if single_text:
                musts = musts_of(tool_id)
                return len(musts) == 1 and musts[0] in text_args_of(tool_id)
            return bool(text_args_of(tool_id)) or any(isinstance(a_spec, dict) and a_spec.get("enum") for a_spec in params_of(tool_id).values())
        cue_table = (("search", r"\b(?:search\w*|find\w*|look\w*|locat\w*|hunt\w*|brows\w*|quer\w*)\b", False),
                ("message", talk_pat, True),
                ("details", r"\b(?:details?|records?|spec\w*|propert\w*|attribut\w*|facts?|full)\b", True),
                ("view", r"\b(?:offer|price|stock|availab\w*)\b", True))
        for job, cue_pat, single_text in cue_table:
            if job in jobs:
                continue
            for tool_id, text_of in said.items():
                if tool_id in jobs.values() or not re.search(cue_pat, text_of):
                    continue
                if fits_role(job, tool_id, text_of, single_text):
                    jobs[job] = tool_id
                    break
        return jobs

    def engine_tag(self, search_name: str | None, words_wanted: set[str]) -> str:
        """TA: the search tool's shelf label (taken from its longest choice list) that shares the most words with
        `wanted`, ties going to the label whose own words are covered more fully; '' when no label shares a word."""
        arg_map = (self.structures.get(search_name or "") or {}).get("properties")
        spec_vals = arg_map.values() if isinstance(arg_map, dict) else []
        enum_lists = [a_spec["enum"] for a_spec in spec_vals
                        if isinstance(a_spec, dict) and isinstance(a_spec.get("enum"), list) and len(a_spec["enum"]) >= 4]
        if not enum_lists:
            return ""
        winner, best_mark = "", (0, 0.0)
        for lab in max(enum_lists, key=len):
            lab_words = category_terms(lab)
            common = len(lab_words & words_wanted)
            if not common:
                continue
            this_mark = (common, common / max(1, len(lab_words)))
            if this_mark > best_mark:
                winner, best_mark = str(lab), this_mark
        return winner

    def engine_settings(self, k: int | None) -> list[tuple[str, str]]:
        """TA6: (property, value) for each requirement, stated or revealed, that is in force for product k (every one when
        there is a single product) and whose value says what the item is - a positive_value, or a short name of five
        words or fewer with no exclusion in it (a several-word colour, a model name) - with the '(stated)' marks removed."""
        multi = self.needs.items()
        kept = []
        for line_req in self.needs.traits_remaining():
            prop_head, _, raw_value = str(line_req).partition(":")
            if multi and self.ware_belonging(prop_head) not in (k, None):
                continue
            clean_value = re.sub(r"\((?:stated|revealed[^)]*)\)", "", raw_value).strip()
            clean_low = clean_value.lower()
            short_name = not re.search(OMIT_TERMS, clean_low) if len(re.findall(r"[a-z0-9]+", clean_low)) <= 5 else False
            if not (affirmative_setting(raw_value, prop_head) or short_name):
                continue
            if clean_value and not reference_passage(clean_value):
                kept.append((re.split(r"\s*\(", prop_head)[0].strip(), clean_value))
        return kept

    def engine_searchterms(self, k: int | None) -> list[dict[str, Any]]:
        """TA6: search calls derived from nothing but the search tool's schema, strongest first. (1) A keyword search for
        the first stated value plus the product's kind word, value leading since some stores read only the opening
        words; (2) when the tool lists shelves, the shelf whose label overlaps the product most, inside the budget (a
        numeric ceiling argument); (3) the value's longest word plus the kind word, for stores that read two words.
        Without a stated value, a keyword search of the product's first and last words. A choice argument gets the
        value its description ties to the argument in use (otherwise its first value for words, its last for shelves)."""
        search_name = self.role("search")
        schema = self.structures.get(search_name or "") or {}
        arg_map = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        required = [a_name for a_name in schema.get("required") or [] if a_name in arg_map]
        prod_rows = self.needs.items()
        if prod_rows and k is not None and k < len(prod_rows):
            product_name = str(prod_rows[k]["product"])
        else:
            product_name = str(self.needs.perusal.get("product") or "")
        shelf_label = self.engine_tag(search_name, category_terms(product_name) or category_terms(self.request))
        described = ""
        for entry in self.toolkit:
            if entry["function"]["name"] == search_name:
                described = str(entry["function"].get("description") or "")
                break
        sentences = [bit.lower() for bit in re.split(r"[.;]", described)]
        tok_arg = self.lexeme_slot()
        text_args = [a_name for a_name, a_spec in arg_map.items()
                if isinstance(a_spec, dict) and a_spec.get("type") == "string" and not a_spec.get("enum") and a_name != tok_arg]
        passage_param = text_args[0] if text_args else None
        for a_name in text_args:
            if re.search(r"^q$|query|text|keyword|needle|term|word|phrase|string|search", a_name.lower()):
                passage_param = a_name
                break
        choice_map = {a_name: a_spec["enum"] for a_name, a_spec in arg_map.items()
                 if isinstance(a_spec, dict) and isinstance(a_spec.get("enum"), list) and a_spec["enum"]}
        shelf_arg = None
        for a_name, choices in choice_map.items():
            if len(choices) >= 4 and (shelf_arg is None or len(choices) > len(choice_map[shelf_arg])):
                shelf_arg = a_name
        mode_args = [a_name for a_name in choice_map if a_name != shelf_arg and len(choice_map[a_name]) <= 4]
        mode_arg = mode_args[0] if mode_args else None
        for a_name in mode_args:
            if a_name in required:
                mode_arg = a_name
                break

        def mode_value(a_name: str, at_end: bool) -> Any:
            choices = choice_map.get(mode_arg or "", [])
            a_pat = r"\b" + re.escape(a_name.lower()) + r"\b"
            for choice in choices:
                c_pat = r"\b" + re.escape(str(choice).lower()) + r"\b"
                if any(re.search(c_pat, sent) and re.search(a_pat, sent) for sent in sentences):
                    return choice
            return choices[-1] if at_end else choices[0]

        def is_price_arg(a_name: str) -> bool:
            a_low = a_name.lower()
            if re.search(r"ceil|max|price|cap$|budget|upper|most", a_low):
                return True
            return any(a_low in sent and re.search(r"\bpric|\bcost", sent) for sent in sentences)
        cap_arg = None
        for a_name, a_spec in arg_map.items():
            if not isinstance(a_spec, dict) or a_spec.get("type") not in ("integer", "number") or a_spec.get("enum"):
                continue
            if is_price_arg(a_name) and not re.search(r"page|first|offset|skip|count|size|num", a_name.lower()):
                cap_arg = a_name
                break
        ceiling = self.needs.outlay_lid
        if k is not None and k < len(self.needs.headcounts()):
            units_each = self.needs.headcounts()[k]
        else:
            units_each = self.needs.quantity()
        units_each = units_each or 1
        price_bound = {}
        if cap_arg and whether_numeric(ceiling) and ceiling > 0:
            price_bound = {cap_arg: int(ceiling // units_each) if arg_map[cap_arg].get("type") == "integer" else round(ceiling / units_each, 2)}
        len_cap = None
        if passage_param and isinstance(arg_map.get(passage_param), dict):
            len_cap = arg_map.get(passage_param, {}).get("maxLength")
        product_words = [wd for wd in re.findall(r"[a-z0-9]+", product_name.lower()) if len(wd) > 2 and wd not in BOND_TERMS]
        kind_word = product_words[-1] if product_words else ""
        multi_pat = r"\bor\b|/|,|\b(?:" + SETTING_BONDS + r")\b"
        firm_values = [fv for _, fv in self.engine_settings(k) if not re.search(multi_pat, fv.lower())]
        phrases: list[str] = []
        if firm_values:
            vwords = [wd for wd in re.findall(r"[a-z0-9]+", firm_values[0].lower()) if wd not in BOND_TERMS]
            if vwords:
                phrases.append(" ".join(dict.fromkeys(vwords + [kind_word])))
                if len(vwords) > 1:
                    phrases.append(" ".join(dict.fromkeys([max(vwords, key=len), kind_word])))
        if not phrases:
            if product_words:
                phrases.append(" ".join(dict.fromkeys(product_words[:1] + product_words[-1:])))
            else:
                phrases.append(" ".join(sorted(category_terms(shelf_label))[:2]))
        if whether_numeric(len_cap):
            cut_phrases = []
            for phrase in phrases:
                while " " in phrase and len(phrase) > len_cap:
                    phrase = phrase.rsplit(" ", 1)[0]
                cut_phrases.append(phrase[: int(len_cap)])
            phrases = cut_phrases

        def make_call(a_name: str, a_val: Any, at_end: bool) -> dict[str, Any]:
            mode_part = {mode_arg: mode_value(a_name, at_end)} if mode_arg else {}
            return {a_name: a_val, **mode_part, **price_bound}
        pending_calls: list[dict[str, Any]] = []
        if passage_param and phrases[0]:
            pending_calls.append(make_call(passage_param, phrases[0], False))
        if shelf_label and shelf_arg:
            pending_calls.append(make_call(shelf_arg, shelf_label, True))
        if passage_param and len(phrases) > 1:
            pending_calls.append(make_call(passage_param, phrases[1], False))
        accepted: list[dict[str, Any]] = []
        for call_draft in pending_calls:
            for a_name in required:
                if a_name in call_draft:
                    continue
                a_spec = arg_map[a_name] if isinstance(arg_map[a_name], dict) else {}
                if a_spec.get("enum"):
                    call_draft[a_name] = a_spec["enum"][0]
                elif a_spec.get("type") in ("integer", "number"):
                    call_draft[a_name] = a_spec.get("default", a_spec.get("minimum", 1))
                elif a_spec.get("type") == "string":
                    call_draft[a_name] = phrases[0] or shelf_label
            clean, issues = ParamAssess.carryout(call_draft, schema)
            if issues and price_bound:
                clean, issues = ParamAssess.carryout({a_name: av for a_name, av in call_draft.items() if a_name not in price_bound}, schema)
            if issues or clean in accepted:
                continue
            accepted.append(clean)
        return accepted

    def engine_tablerows(self) -> dict[str, dict[str, Any]]:
        """TA6: each listing seen so far, keyed by id: its category label, its title, the latest price and stock the
        records report, the words found in all of its fields, and its property fields."""
        by_id: dict[str, dict[str, Any]] = {}

        def note_row(rec: dict[str, Any]) -> None:
            rid = self.tablerow_flag(rec)
            if rid not in self.vault.ids:
                return
            slot = by_id.get(rid)
            if slot is None:
                slot = by_id[rid] = {"shelf": "", "title": "", "price": None, "stock": None, "words": set(), "fields": {}}
            slot["shelf"] = self.rack_belonging(rec) or slot["shelf"]
            found_title = ""
            for fld, fv in rec.items():
                low_key = str(fld).lower()
                if isinstance(fv, str) and re.search(r"title|name", low_key) and "id" not in low_key:
                    found_title = str(fv)
                    break
            slot["title"] = found_title or slot["title"]
            for fld, fv in rec.items():
                low_key = str(fld).lower()
                if not whether_numeric(fv) or isinstance(fv, bool) or not re.search(r"price|cost|amount", low_key):
                    continue
                if re.search(r"count|qty|quant|total|units|line|due|payable|sum", low_key):
                    continue
                slot["price"] = float(fv)
                break
            stock_notes = [sv for sv in (self.slot_inventory(fld, fv) for fld, fv in rec.items()) if sv is not None]
            if stock_notes:
                slot["stock"] = stock_notes[-1]
            nested = [sv for sv in rec.values() if isinstance(sv, dict)]
            texts = [str(sv) for sv in rec.values() if isinstance(sv, (str, int, float)) and not isinstance(sv, bool)]
            texts.extend(str(item) for part_dict in nested for item in part_dict.values() if isinstance(item, (str, int, float)))
            slot["words"] |= category_terms(" ".join(texts))
            for part_dict in nested:
                slot["fields"].update({str(fld): str(item) for fld, item in part_dict.items()
                                        if isinstance(item, (str, int, float)) and not isinstance(item, bool)})
        for stored in self.vault.items:
            traverse_mappings(stored, note_row)
        return by_id

    @staticmethod
    def engine_displays(rec: dict[str, Any], prop: str, value_text: str, search_words: set[str], opened: bool) -> bool:
        """TA6: whether a listing itself evidences a stated value (code_seen). A field for the property must carry it;
        lacking such a field, the title may; and for a listing whose record is still unread, having been found by a
        keyword search containing every word of the value also counts (the words may live in a differently named field)."""
        field_hits = [fv for fk, fv in rec["fields"].items() if identical_trait(fk, prop)]
        if field_hits:
            return any(Errand.engine_observed(value_text, fv) for fv in field_hits)
        if Errand.engine_observed(value_text, rec.get("title") or ""):
            return True
        if opened:
            return False
        value_words = category_terms(value_text)
        return bool(value_words) and value_words <= search_words

    @staticmethod
    def engine_observed(wanted: str, where: str) -> bool:
        """TA6: `text` holds every word and number of at least one alternative of the value (alternatives split at 'or',
        '/', ','); longer words match by their stem, and a number must not be part of a bigger one."""
        haystack = str(where or "").lower()
        if not haystack.strip():
            return False
        stripped_value = re.sub(r"\b(?:" + SETTING_BONDS + r")\b", " ", str(wanted).lower())
        for alt in re.split(r"\s*(?:\bor\b|/|,|;)\s*", stripped_value):
            alpha = [wd for wd in re.findall(r"[a-z]{2,}", alt) if wd not in BOND_TERMS]
            digits = re.findall(r"\d+(?:\.\d+)?", alt)
            if not alpha and not digits:
                continue
            if not all(re.search(r"\b" + re.escape(word if len(word) <= 4 else word[:-1]), haystack) for word in alpha):
                continue
            if all(re.search(r"(?<![\d.])" + re.escape(fig) + r"(?![\d.])", haystack) for fig in digits):
                return True
        return False

    def engine_chooses(self, forbidden: set[str] | None = None, kept: set[str] | None = None) -> list[tuple[str, int]]:
        """TA6: (listing, units) per requested product, decided from the records alone. Preference order: a line in `keep`;
        a listing the item check passed; one whose own evidence shows every stated value (code_shows); a read listing
        before an unread one (cheaper first under a cheapest rule); shown in stock; most words shared with the
        requirements, its kind and the request; then the lower price. A pick is never shown out of stock, never ruled
        out (by a check, an answered decline audit, a refused checkout, `avoid`, or its own field for a stated property),
        stays within the budget, and sits on the product's category (or, while none is known, names the product's kind).
        If the picks together overrun the budget, the cheapest fitting listing per product is used instead. Returns []
        when a product has no candidate or even the cheapest set overruns: nothing in the records can fit then."""
        product_rows = self.needs.items()
        unit_list = self.needs.headcounts()
        if product_rows:
            home_sets = self.landing_racks()
        else:
            single_home = set(self.rack_tags())
            if not single_home:
                single_home = (self.landing_racks() or [set()])[0]
            home_sets = [single_home]
        seen_rows = self.engine_tablerows()
        version_tag = str(self.needs.iteration)
        failed_ids = set()
        for rule_key, ruling in self.verdicts_alt.items():
            parts = rule_key.split("|")
            if ruling and parts[1:2] == [version_tag]:
                failed_ids.add(parts[0])
        passed_ids = set()
        for pass_key in self.succeeded:
            lead_id, _, ver_part = pass_key.rpartition("|")
            if ver_part == version_tag:
                passed_ids.add(lead_id)
        audit_seen = self.__dict__.get("audit_ids")
        if self.__dict__.get("audit_done"):
            if audit_seen is None or len(self.vault.read) <= self.__dict__.get("audit_reads", -1):
                audit_seen = set(self.vault.read)
        # (only an answered decline audit rules out the listings it looked at without naming them; an audit whose model
        # call failed or that ran muted proves nothing)
        audit_answered = False
        for ev in reversed(self.journal):
            if isinstance(ev, dict) and ev.get("kind") == "audit":
                audit_answered = isinstance(ev.get("result"), dict)
                break
        audit_out: set[str] = set()
        if audit_answered:
            audit_out = set(audit_seen or set()) - set(self.__dict__.get("audit_fits") or []) - passed_ids
        # (EA4: while the code itself judges every requirement in force, a read listing whose own evidence fits them all
        # survives the audit's silence; a failed check still rules it out)
        code_all = self.engine_evaluates_whole() if audit_out else False
        budget = float(self.needs.outlay_lid) if whether_numeric(self.needs.outlay_lid) else None

        def field_grade(rec: dict[str, Any], prop_name: str, value_text: str) -> tuple[int, int]:
            # (EA5: how a read record shows a value. First part: in the field named like the request's property (0),
            # under another name for it (1), or in no field (2). Second part: equal to the whole value (0), equal to a
            # list part (1), only in the title (2), merely contained (3))
            plain_forms = set()
            for piece in re.split(r"\s*(?:\bor\b|/)\s*", value_text.lower()):
                if crosscheck_terms(piece):
                    plain_forms.add(" ".join(crosscheck_terms(piece)))
            best_grade = (2, 3)
            for fname, ftext in rec["fields"].items():
                if not (identical_trait(fname, prop_name) and Errand.engine_observed(value_text, ftext)):
                    continue
                if " ".join(crosscheck_terms(ftext)) in plain_forms:
                    match_tier = 0
                elif any(" ".join(crosscheck_terms(chunk)) in plain_forms for chunk in crosscheck_portions(ftext)):
                    match_tier = 1
                else:
                    match_tier = 3
                best_grade = min(best_grade, (int(trait_slot(fname) != trait_slot(prop_name)), match_tier))
            if best_grade == (2, 3) and Errand.engine_observed(value_text, rec.get("title") or ""):
                return (2, 2)
            return best_grade
        request_words = category_terms(self.request)
        hit_map = self.__dict__.get("code_hits") or {}
        cheap_first = self.needs.priority() == "cheapest"
        forbidden = set(forbidden or ())
        kept = set(kept or ())
        forbidden |= self.defunct_lineitems()       # (CB5: listings whose latest stock record says sold out)
        forbidden |= self.reference_indicators()     # (CE5: listings a differ reference's chain passed through)
        range_pat = r"\b(?:" + SETTING_BONDS + r")\b|\d\s*(?:-|to)\s*\d|[+<>]|\b(?:about|around|approx\w*|roughly)\b"
        candidate_lists: list[list[tuple]] = []
        for pos, home_set in enumerate(home_sets):
            unit_n = unit_list[pos] if pos < len(unit_list) else 1
            prod_slot = pos if product_rows else None
            kind_set = category_terms(product_rows[pos]["product"] if product_rows else self.needs.perusal.get("product") or "")
            value_words: set[str] = set()
            checks: list[tuple[str, str]] = []
            bounds: list[tuple[str, str, float, str]] = []
            for prop_name, value_str in self.engine_settings(prod_slot):
                value_words |= category_terms(value_str)
                if not re.search(range_pat, value_str.lower()):
                    checks.append((prop_name, value_str))      # (bounds and ranges are left out: no single word or figure decides them)
                else:
                    # (a one-sided limit with one figure and its unit - 'at least 11 inch' - is kept for a plain-failure test)
                    low_txt = value_str.lower()
                    side = ("min" if re.search(r"\b(?:at least|no less than|not less than|minimum|or more|or above|over|above|more than)\b", low_txt)
                            else "max" if re.search(r"\b(?:at most|no more than|not more than|maximum|up to|or less|or below|under|below|less than)\b", low_txt)
                            else "")
                    amounts = reference_numerals(low_txt)
                    if side and len(amounts) == 1:
                        amount, unit = next(iter(amounts))
                        bounds.append((prop_name, side, amount, unit))
            # (EA4: the product's settled differ references - a fit shows some value of the property other than the excluded one)
            differs = []
            for dref in self.diverge_completed(prod_slot):
                excluded = [str(rv) for rv in [dref.get("value")] + list(dref.get("goes_with") or []) if str(rv or "").strip()]
                differs.append((dref["prop"], excluded))
            ranked = []
            for lid, cand in seen_rows.items():
                if lid in failed_ids or lid in self.rejected_lineitems or lid in forbidden or cand["stock"] is False:
                    continue
                if home_set:
                    if cand["shelf"] not in home_set:
                        continue
                elif not kind_set & cand["words"]:
                    continue
                if any(self.engine_conflict(cand["fields"], prop_name, value_str) for prop_name, value_str in checks):
                    continue

                def breaks(limit: tuple[str, str, float, str]) -> bool:
                    # (a field named for the property writes figures in the limit's unit and every one of them misses it)
                    prop_name, side, amount, unit = limit
                    for fname, ftext in cand["fields"].items():
                        if not identical_trait(fname, prop_name):
                            continue
                        same_unit = [f for f, u in reference_numerals(ftext) if u == unit]
                        if same_unit and all((f < amount) if side == "min" else (f > amount) for f in same_unit):
                            return True
                    return False
                if any(breaks(limit) for limit in bounds):
                    continue
                if self.reference_omitted(cand["fields"], prod_slot):
                    continue          # (CE5: a field shows the value a differ reference excludes)
                unit_cost = cand["price"] if cand["price"] is not None else self.vault.price(lid)
                if budget is not None and whether_numeric(unit_cost) and unit_cost * unit_n > budget:
                    continue
                was_read = lid in self.vault.read
                found_words = hit_map.get(lid) or set()
                fit_ok = all(self.engine_displays(cand, prop_name, value_str, found_words, was_read) for prop_name, value_str in checks)
                if fit_ok:
                    fit_ok = all(any(identical_trait(fname, prop_name) and str(ftext).strip() and not any(crosscheck_inscribes(bad_val, ftext) for bad_val in excluded)
                                   for fname, ftext in cand["fields"].items()) for prop_name, excluded in differs)
                grades = [field_grade(cand, prop_name, value_str) for prop_name, value_str in checks] if was_read else []
                if lid in audit_out:
                    evidence_wins = code_all and fit_ok and (checks or differs) and all(max(g) <= 1 for g in grades)
                    if not evidence_wins:
                        continue      # (an answered decline audit passed over it in silence; that silence gives way only to the
                        # value itself or a list part sitting in a field of the property - not to a title, a longer word, a
                        # phrase containing it, nor a product with no requirement the code can check)
                overlap = (3 * len(value_words & cand["words"]) + 2 * len(kind_set & cand["words"]) + len(request_words & cand["words"])
                         + 4 * len(value_words & found_words))
                sort_price = unit_cost if whether_numeric(unit_cost) else float("inf")
                grade_sum = (sum(g[0] for g in grades), sum(g[1] for g in grades)) if grades else (9, 9)
                ranked.append((int(lid not in kept), int(lid not in passed_ids), not fit_ok, budget is not None and not whether_numeric(unit_cost),
                             sort_price if cheap_first else 0.0, int(not was_read), cand["stock"] is not True,
                             (grade_sum, -overlap), sort_price, lid))
            if not ranked:
                return []
            candidate_lists.append(sorted(ranked))

        def plain_order(option: tuple) -> tuple:
            return option

        def cheap_order(option: tuple) -> tuple:
            return (option[2], option[3], option[8], option[-1])
        for sorter in (plain_order, cheap_order):
            selection: list[tuple[str, int]] = []
            total_cost = 0.0
            entries = []
            for pos, ranked in enumerate(candidate_lists):
                # (v605d M4, CH2: a read listing whose field evidence shows a plain failure is not picked - checked lazily here)
                pick = None
                for option in sorted(ranked, key=sorter):
                    if any(option[-1] == prev_id for prev_id, _ in selection):
                        continue
                    if option[-1] in self.vault.read and self.attribute_snapshot(option[-1])["fails"]:
                        continue
                    pick = option
                    break
                if pick is None:
                    return []
                unit_n = unit_list[pos] if pos < len(unit_list) else 1
                selection.append((pick[-1], unit_n))
                entries.append(pick)
                each_cost = 0.0 if pick[8] == float("inf") else pick[8]
                total_cost += each_cost * unit_n
            if budget is None or total_cost <= budget:
                self.code_fit = {x[-1]: not x[2] for x in entries}
                return selection
        return []

    @staticmethod
    def engine_conflict(fmap: dict[str, str], prop_name: str, wanted: str) -> bool:
        """TA6: a read record shows a stated requirement failing - it has fields for that property and not one of them
        holds a word or number of the required value (the code's own evidence; with no such field it never clashes)."""
        prop_vals = [fv for fk, fv in fmap.items() if identical_trait(fk, prop_name)]
        if not prop_vals:
            return False
        return not any(setting_observed(wanted, fv) for fv in prop_vals)

    def carriage_layout(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str | None, list[str]]:
        """TA6: the cart tool's schema as the code's own calls use it: (schema, the add value of each required choice,
        its remove value (None where the tool offers none), the id argument, the units arguments). The add value is the
        one the actor's first accepted cart call used, otherwise the value whose words (split at non-letters and capitals)
        contain an add word; the remove value is the one containing a remove word (or, with two values, the other one)."""
        schema = self.structures.get(self.role("cart") or "") or {}
        arg_map = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        prior = serialized_entity(self.handcart.place_layout) if self.handcart.place_layout else None

        def word_set(opt: Any) -> set[str]:
            split_txt = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(opt)).lower()
            return set(re.findall(r"[a-z]+", split_txt))
        put_vals: dict[str, Any] = {}
        take_vals: dict[str, Any] = {}
        for a_name in schema.get("required") or []:
            a_spec = arg_map.get(a_name) if isinstance(arg_map.get(a_name), dict) else {}
            choices = a_spec.get("enum") if isinstance(a_spec.get("enum"), list) else []
            if not choices:
                continue
            put_like = [c for c in choices if word_set(c) & PLACE_TERMS and not word_set(c) & DISCARD_TERMS]
            take_like = [c for c in choices if word_set(c) & DISCARD_TERMS]
            if isinstance(prior, dict) and prior.get(a_name) in choices:
                put_vals[a_name] = prior[a_name]
            else:
                put_vals[a_name] = put_like[0] if put_like else choices[0]
            others = [c for c in choices if c != put_vals[a_name]]
            alt_default = others[0] if len(choices) == 2 else None
            take_vals[a_name] = next((c for c in take_like if c != put_vals[a_name]), alt_default)
        text_args = [a_name for a_name, a_spec in arg_map.items() if isinstance(a_spec, dict) and a_spec.get("type") == "string" and not a_spec.get("enum")]
        id_arg = text_args[0] if text_args else None
        for a_name in text_args:
            if re.search(r"id|item|product|listing|sku|target|which|number", a_name.lower()):
                id_arg = a_name
                break
        num_args = [a_name for a_name, a_spec in arg_map.items()
                     if isinstance(a_spec, dict) and a_spec.get("type") in ("integer", "number") and not a_spec.get("enum")]
        return schema, put_vals, take_vals, id_arg, num_args

    def carriage_params(self, selected: list[tuple[str, int]]) -> list[dict[str, Any]]:
        """TA6: the cart calls that add `picks`, built from the cart tool's schema: a single call listing every line when
        the tool accepts a list of line objects (that call states the whole cart), otherwise one call per line (its id
        argument, its units argument, the add value of each required choice); [] when the schema permits none."""
        schema, put_vals, _, id_arg, num_args = self.carriage_layout()
        line_layout = self.handcart.lineitem_specification()
        if line_layout:
            list_arg, line_id, line_qty = line_layout
            pending_calls = [{**put_vals, list_arg: [{line_id: lid, line_qty: n_units} for lid, n_units in selected]}]
        elif not id_arg:
            return []
        else:
            pending_calls = []
            for lid, n_units in selected:
                call_draft = {**put_vals, id_arg: lid}
                if num_args:
                    call_draft[num_args[0]] = n_units
                pending_calls.append(call_draft)
        accepted = []
        for call_draft in pending_calls:
            clean, issues = ParamAssess.carryout(call_draft, schema)
            if issues:
                return []
            accepted.append(clean)
        return accepted

    def carriage_discards(self, doomed: list[str]) -> list[dict[str, Any]] | None:
        """TA6: the cart calls that remove `lines`: each required choice set to its remove value together with the line's
        units, or (for a tool that sets units) the line at 0 units. [] when the tool takes a list of lines (its add call
        restates the whole cart) or when there is nothing to remove; None when the schema offers no way to remove."""
        if not doomed or self.handcart.lineitem_specification():
            return []
        schema, put_vals, take_vals, id_arg, num_args = self.carriage_layout()
        arg_map = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        if not id_arg:
            return None
        if put_vals:
            if any(take_vals.get(a_name) is None for a_name in put_vals):
                return None
            set_zero = False
        else:
            least = (arg_map.get(num_args[0]) or {}).get("minimum") if num_args else None
            if not num_args or (whether_numeric(least) and not arg_map[num_args[0]]["minimum"] <= 0):
                return None
            set_zero = True
        accepted = []
        for lid in doomed:
            if set_zero:
                call_draft = {id_arg: lid, num_args[0]: 0}
            else:
                call_draft = {**take_vals, id_arg: lid}
                if num_args:
                    call_draft[num_args[0]] = max(1, int(self.handcart.retained(lid) or 1))
            clean, issues = ParamAssess.carryout(call_draft, schema)
            if issues:
                return None
            accepted.append(clean)
        return accepted

    def engine_responses(self, limit: int) -> list[Any]:
        """TA6: the observations returned for the latest world call the code made (one per action)."""
        last_env = None
        for ev in reversed(self.journal):
            if isinstance(ev, dict) and ev.get("kind") == "env":
                last_env = ev
                break
        payload = (last_env or {}).get("reply")
        if not isinstance(payload, dict):
            return []
        obs_list = payload.get("observations")
        series = obs_list if isinstance(obs_list, list) else [payload.get("observation")]
        return series[:limit]

    @staticmethod
    def engine_defect(payload: Any) -> bool:
        """TA6: the reply turns the call down - some key mentioning an error, fault, refusal, rejection or invalid input
        carries a value."""
        hits = []

        def inspect(rec: dict[str, Any]) -> None:
            for fk, fv in rec.items():
                if re.search(DEFECT_SLOTS, str(fk).lower()) and fv not in (None, False, "", [], {}):
                    hits.append(rec)
                    return
        traverse_mappings(payload, inspect, 0, 2)
        return len(hits) > 0

    def engine_seek(self, n_products: int, extra_pages: bool = False) -> None:
        """TA6: runs the code's own searches (code_queries) for every product, each product's best query first, as far
        as the steps allow. Then, while some product still lacks a fitting row or a cheapest rule applies, it follows
        the next pages of each product's best query (at most CODE_PAGES rounds; pages not yet read are kept in
        code_more). With `more`, it runs a single round of those kept pages only (used after a later round found no
        fitting row). Rows returned by a keyword search are noted with its words (code_hits) unless the store is known
        to match on any one word."""
        search_name = self.role("search")
        text_args = [a_name for a_name, a_spec in ((self.structures.get(search_name or "") or {}).get("properties") or {}).items()
                if isinstance(a_spec, dict) and a_spec.get("type") == "string" and not a_spec.get("enum")]
        keep_back = 2 + 2 * n_products
        if not extra_pages:
            query_lists = []
            for kk in range(n_products):
                query_lists.append(self.engine_searchterms(kk if self.needs.items() else None))
            ordered: list[dict[str, Any]] = []
            deepest = max((len(qs) for qs in query_lists), default=0)
            for level in range(deepest):
                fresh = [qs[level] for qs in query_lists if level < len(qs) and qs[level] not in ordered]
                ordered.extend(fresh)
            batch = ordered[: max(0, self.shifts_remaining() - keep_back)]
            leaders = [qs[0] for qs in query_lists if qs]
            passes = 1 + ENGINE_SHEETS + (2 if self.needs.priority() == "cheapest" else 0)
        else:
            leaders = list(self.__dict__.get("code_more") or [])
            batch = leaders[: max(0, self.shifts_remaining() - keep_back - 2 * n_products)]
            passes = 1
        if not batch:
            return
        self.code_searched = True

        def make_booker(hit_map: dict[str, set[str]], wset: set[str]):
            def book_row(rec: dict[str, Any]) -> None:
                rid = self.tablerow_flag(rec)
                if rid in self.vault.ids:
                    hit_map.setdefault(rid, set()).update(wset)
            return book_row
        done_passes = 0
        while True:
            stamp = self.protocol.pulse
            self.transmit_segment([(f"code_search_{i}_{stamp}", search_name, cargs) for i, cargs in enumerate(batch)])
            answers = self.engine_responses(len(batch))
            paired = list(zip(batch, answers))
            known_rule = self.__dict__.get("word_kind")
            if known_rule != "any":
                hit_map = self.__dict__.setdefault("code_hits", {})
                for cargs, ob in paired:
                    typed = " ".join(str(cargs.get(a_name) or "") for a_name in text_args)
                    if typed.strip():
                        traverse_mappings(ob, make_booker(hit_map, category_terms(typed)))
            next_calls = []
            for cargs, ob in paired:
                if cargs not in leaders:
                    continue
                follow = self.engine_subsequent(cargs, ob)
                if follow:
                    next_calls.append(follow)
            self.code_more = next_calls
            done_passes += 1
            if done_passes >= passes:
                break
            selected = self.engine_chooses()
            fit_map = self.__dict__.get("code_fit") or {}
            if self.needs.priority() != "cheapest" and selected and all(fit_map.get(lid) for lid, _ in selected):
                break
            leaders = next_calls
            # (further pages leave room for the steps that reading the picks needs: about three records per product)
            batch = next_calls[: max(0, self.shifts_remaining() - keep_back - 3 * n_products)]
            if not batch:
                break

    def engine_subsequent(self, base_args: dict[str, Any], page_obs: Any) -> dict[str, Any] | None:
        """TA6: the call that fetches the following page of a search, built from the search tool's paging argument and
        what the page reports (a next marker, or else its match count compared with the rows shown so far); None when
        this was the last page."""
        page_spec = self.sheet_param()
        if not page_spec:
            return None
        page_key, whole_pages, start_at, max_at = page_spec
        found_pages: list[dict[str, Any]] = []

        def note_page(rec: dict[str, Any]) -> None:
            for val in rec.values():
                if isinstance(val, list) and val and all(isinstance(item, dict) for item in val):
                    found_pages.append(rec)
                    return
        traverse_mappings(page_obs, note_page, 0, 3)
        if not found_pages:
            return None
        first_page = found_pages[0]
        rows_shown = max(len(val) for val in first_page.values() if isinstance(val, list))
        next_marks = [val for fkey, val in first_page.items() if re.search(r"next|onward|cursor|bookmark|token|continu|more", str(fkey).lower())]
        match_total = None
        for fkey, val in first_page.items():
            if whether_numeric(val) and not isinstance(val, bool) and re.search(r"count|total|matched|found|hits", str(fkey).lower()):
                match_total = val
                break
        if next_marks and all(mk in (None, False, "", 0) for mk in next_marks):
            return None
        if whole_pages == "token":
            next_tok = next((mk for mk in next_marks if isinstance(mk, str) and mk.strip()), None)
            if not next_tok:
                return None
            next_args = {**base_args, page_key: next_tok}
        else:
            at = base_args.get(page_key, start_at)
            if not whether_numeric(at):
                at = start_at
            upcoming = at + 1 if whole_pages else at + rows_shown
            if max_at is not None and upcoming > max_at:
                return None
            if not next_marks:
                seen_so_far = (at - start_at + 1) * rows_shown if whole_pages else at + rows_shown
                if match_total is None or seen_so_far >= match_total:
                    return None
            next_args = {**base_args, page_key: int(upcoming) if float(upcoming).is_integer() else upcoming}
        clean, issues = ParamAssess.carryout(next_args, self.structures.get(self.role("search") or "") or {})
        if issues:
            return None
        return clean

    def engine_peruse(self, selected: list[tuple[str, int]], forbidden: set[str], kept: set[str] | None) -> list[tuple[str, int]]:
        """TA6: reads the picks' own records while steps permit (the list-read tool for two or more, else the details
        tool) and picks again from what they show; at most CODE_READS rounds, always keeping the steps needed for the
        cart, a stock re-check, the order and one repair."""
        tries = ENGINE_PERUSES
        while tries > 0:
            tries -= 1
            pending = [lid for lid, _ in selected if lid not in self.vault.read]
            bulk_tool = self.bundle_instrument()
            use_bulk = bool(bulk_tool) and len(pending) >= 2
            steps_needed = (1 if use_bulk else len(pending)) + 2 * len(selected) + 3
            if not pending:
                break
            if not (bulk_tool or self.role("details")):
                break
            if self.shifts_remaining() < steps_needed or self.protocol.surplus_alt() <= 8:
                break
            if use_bulk:
                wave = [(f"code_read_{self.protocol.pulse}", bulk_tool[0], {bulk_tool[1]: pending[: bulk_tool[2]]})]
            else:
                wave = []
                for i, lid in enumerate(pending):
                    cargs = self.particulars_invoke(lid)
                    if cargs:
                        wave.append((f"code_read_{i}_{self.protocol.pulse}", self.role("details"), cargs))
            if not wave:
                break
            self.transmit_segment(wave)
            redo = self.engine_chooses(forbidden, kept)
            if redo == selected:
                break
            selected = redo
        return selected

    def engine_place(self, selected: list[tuple[str, int]]) -> list[str] | None:
        """TA6: makes the cart hold exactly `picks` - lines that are not picks come out, picks not yet held at their units
        go in - and checks the cart's replies. A pick whose call was refused, or which the replies (or one cart view, when
        the replies show no lines) fail to show, is returned for swapping. None when the calls cannot be made."""
        cart_tool = self.role("cart")
        wanted_ids = {lid for lid, _ in selected}
        line_layout = self.handcart.lineitem_specification()
        surplus = [ln for ln in self.handcart.measures if ln not in wanted_ids]
        instructions = self.needs.informed
        must_ask = any(ins.get("before_change") is True for ins in instructions)
        unknown_rule = any(ins.get("before_change") is None for ins in instructions)
        if surplus and (must_ask or unknown_rule):
            # (a stated step requires checking with the shopper before replacing anything in the cart, or the reading
            # of an instruction is unknown: the code asks first while a message beyond the basket's own is left; with a
            # stated step and no such message, no swap is made)
            approved = self.engine_inquiries_substitute(surplus, [lid for lid, _ in selected])
            if must_ask and not approved:
                return None
        if line_layout:
            to_add = list(selected)
        else:
            to_add = [(lid, n_units) for lid, n_units in selected if abs(self.handcart.retained(lid) - n_units) > 1e-9]
        removals = self.carriage_discards(surplus)
        additions = self.carriage_params(to_add) if to_add else []
        if removals is None or (to_add and not additions):
            return None
        batch = removals + additions
        if not batch:
            return []
        if self.shifts_remaining() < 1 + len(batch):
            return None
        stamp = self.protocol.pulse
        self.transmit_segment([(f"code_cart_{i}_{stamp}", cart_tool, cargs) for i, cargs in enumerate(batch)])
        answers = self.engine_responses(len(batch))
        rejected: set[str] = set()
        visible = None
        for cargs, ob in zip(batch, answers):
            listed = self.handcart.snapshot_measures(ob)
            if listed is not None:
                visible = listed
            if not self.engine_defect(ob):
                continue
            mentioned = [lid for lid in wanted_ids if lid in into_passage(ob)]
            in_args = [ln for ln, _ in self.handcart.lines(cargs) if ln in wanted_ids]
            if mentioned:
                rejected.update(mentioned)
            elif in_args:
                rejected.update(in_args)
            elif line_layout:
                rejected.update(list(wanted_ids))
        if len(answers) < len(batch):
            return None           # (no answer from the world: nothing is known about the picks)
        if visible is None and not rejected and self.role("cart_view") and self.shifts_remaining() >= 3:
            view_call = (f"code_view_{self.protocol.pulse}", self.role("cart_view"), {})
            self.transmit_segment([view_call])
            look = self.engine_responses(1)
            if look:
                visible = self.handcart.snapshot_measures(look[0])
                if visible is None:
                    # (a view naming none of the lines empties the cart as counted: those picks are not in it)
                    rejected.update(lid for lid in wanted_ids if lid not in self.handcart.measures)
            else:
                visible = None
        if visible is not None:
            rejected.update(lid for lid in wanted_ids if lid not in visible)
        for lid in rejected:
            self.handcart.measures.pop(lid, None)
        return sorted(rejected)

    def engine_reassess(self, line_ids: list[str]) -> tuple[bool, set[str]]:
        """TA6: (was read, sold out). When steps permit, each line's stock is read once more (the view tool, else the
        details tool, one call per line), since a listing can sell out while it sits in the cart; returns the lines now
        shown out of stock."""
        reader = self.inventory_instrument() or self.role("view") or self.role("details")
        no_read: tuple[bool, set[str]] = (False, set())
        if not line_ids or not reader:
            return no_read
        if self.shifts_remaining() < len(line_ids) + 2 or self.protocol.surplus_alt() < 8:
            return (False, set())
        wave = []
        for i, lid in enumerate(line_ids):
            cargs = self.retrieve_invoke(lid, reader)
            if not cargs:
                continue
            wave.append((f"code_check_{i}_{self.protocol.pulse}", reader, cargs))
        if not wave:
            return (False, set())
        self.transmit_segment(wave)
        by_id = self.engine_tablerows()
        sold_now = set()
        for lid in line_ids:
            if (by_id.get(lid) or {}).get("stock") is False:
                sold_now.add(lid)
        return True, sold_now

    def code_news(self, fresh_lines: list[str]) -> None:
        """TA6: shopper lines that no model has read (a refusal's, or ones whose listener call failed) are scanned by the
        code for a new budget: a spending figure in a line (code_budget) becomes the budget."""
        queued = []
        for group, _ in self.needs.__dict__.get("pending") or []:
            queued.extend(group)
        handled = self.__dict__.setdefault("news_read", set())
        for text_line in [*fresh_lines, *queued]:
            if text_line in handled:
                continue
            handled.add(text_line)
            amount = self.engine_allotment(text_line)[0]
            if not whether_numeric(amount) or amount <= 0:
                continue
            self.needs.outlay_lid = amount
            self.journal.append({"kind": "code_news", "budget": amount})

    def engine_carriage(self, forbidden: set[str] | None = None, kept: set[str] | None = None) -> bool:
        """TA6: the errand is ending with nothing in the cart to order (models gave out, the clock or the steps ran low,
        or a checkout was refused). Rather than decline, the code shops on its own without any model call: it searches
        when the records hold no fitting candidate, reads its picks while steps permit, makes the cart hold one line per
        product (code_picks; `keep` lines first, never `avoid`), confirms the cart from the replies, re-reads the lines'
        stock, and swaps a refused or sold-out line (up to four rounds). True when the cart holds the picks; False when
        nothing in the records can fit or no call can be made (the caller then declines)."""
        cart_tool = self.role("cart")
        search_tool = self.role("search")
        if not cart_tool or self.protocol.done or self.protocol.crashed:
            return False
        if self.shifts_remaining() < 2 or self.protocol.surplus_alt() < 4:
            return False
        self.protocol.silence = True

        def finish_note(**rest: Any) -> None:
            self.journal.append({"kind": "code_finish", **rest})
        try:
            forbidden = set(forbidden or ())
            n_products = max(1, len(self.needs.items()))
            for _round in range(4):
                selected = self.engine_chooses(forbidden, kept)
                fit_map = self.__dict__.get("code_fit", {})
                lacking = not selected or not all(fit_map.get(lid) for lid, _ in selected)
                if search_tool and not self.__dict__.get("code_searched") and self.shifts_remaining() >= 4 and lacking:
                    self.engine_seek(n_products)
                    selected = self.engine_chooses(forbidden, kept)
                elif search_tool and self.__dict__.get("code_more") and self.shifts_remaining() >= 6 + 2 * n_products and (
                        lacking or self.needs.priority() == "cheapest"):
                    # (nothing fitting is left, or something cheaper may lie further on: follow the best searches' next pages)
                    self.engine_seek(n_products, True)
                    selected = self.engine_chooses(forbidden, kept)
                selected = self.engine_peruse(selected, forbidden, kept)
                if not selected:
                    finish_note(picks=[], result="nothing to order")
                    return False
                rejected = self.engine_place(selected)
                if rejected is None:
                    finish_note(picks=selected, result="no cart call")
                    return False
                if not rejected:
                    sold_now: set[str] = set()
                    if self.shifts_remaining() >= len(selected) + 3:
                        sold_now = self.engine_reassess([lid for lid, _ in selected])[1]
                    rejected = sorted(sold_now)
                    if not rejected:
                        finish_note(picks=selected, cart=dict(self.handcart.measures))
                        return len(self.handcart.measures) > 0
                forbidden.update(rejected)
                self.journal.append({"kind": "code_swap", "lines": rejected})
            return False
        except Exception as exc:
            self.journal.append({"kind": "code_finish_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
            return False


class SeekAndPaging:
    """Errand component: finishing, notes, shelf labels, search keys, sweeps and paging."""

    def conclude_current(self) -> None:
        """Closes the errand: the cart is ordered as it stands, and an empty cart is first filled by the code (TA6,
        code_cart); it declines only when nothing that can fit turns up. A refused finish is booked like any other
        refusal, the lines' stock is read again, a sold-out line is swapped and the order goes again (at most
        CODE_FINISHES calls); when the store never accepts the order, the final word is still the decline."""

        def stopped() -> bool:
            return self.protocol.done or self.protocol.crashed or self.shifts_remaining() <= 0

        def make_order() -> dict[str, Any]:
            return {"name": order_name, "args": self.simple_params(order_name, "Order the cart.", "order")}

        def make_decline() -> dict[str, Any]:
            return {"name": decline_name, "args": self.simple_params(decline_name, APOLOGY, "decline")}
        if stopped() or self.protocol.surplus_alt() < 2:
            return
        order_name = self.role("order")
        decline_name = self.role("decline")
        forbidden: set[str] = set()
        kept: set[str] | None = None
        action: dict[str, Any] | None = None
        act_kind = None
        final_turn = ENGINE_CONCLUDES - 1
        for attempt in range(ENGINE_CONCLUDES):
            if stopped() or self.protocol.surplus_alt() < 2:
                return
            if action is None:
                dead: set[str] = set()
                first_try = bool(order_name) and attempt == 0
                if first_try and self.handcart.measures:
                    # (CB5: cart lines not read since the latest cart change get one fresh stock read; the code swaps a
                    # sold-out line before the first order rather than ordering it)
                    try:
                        self.protocol.silence = True
                        cart_now = list(self.handcart.measures)
                        dead = self.carriage_vanished(cart_now)
                        if dead:
                            forbidden |= dead
                            kept = set(cart_now) - dead
                            self.journal.append({"kind": "code_dead", "lines": sorted(dead)})
                    except Exception as exc:
                        self.journal.append({"kind": "cart_gone_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
                        dead = set()
                if first_try and not dead and self.handcart.employed and (
                        self.handcart.measures or self.handcart.snapshot_measures(self.handcart.final_snapshot) is None):
                    action = make_order()
                elif order_name and (self.engine_carriage(forbidden, kept) or (attempt > 0 and self.anew_approved(forbidden))):
                    # (TA6: the code filled an empty cart or replaced a gone line by itself, so the errand orders; CF6: after
                    # a refusal with no repair possible, no line known sold out and candidates still in the records, the
                    # same cart goes again instead of a decline)
                    action = make_order()
                elif decline_name:
                    action = make_decline()
                else:
                    return
            act_kind = self.invoke_category(action["name"], action["args"])
            try:
                self.conclude_stages(act_kind or "order")
            except Exception as exc:
                self.journal.append({"kind": "finish_steps_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
            if stopped():
                return
            answer = self.protocol.perform([action])
            self.journal.append({"kind": "forced_finish", "action": action, "reply": answer})
            if not answer or answer.get("finished") or self.protocol.done:
                break
            if attempt == final_turn or self.shifts_remaining() <= 1:
                break
            # (TA6: the store refused the finish - book it like any refusal: the lines that sat in the cart, the records,
            # the shopper's lines; the code reads any budget they carry)
            self.protocol.silence = True
            cart_lines = list(self.handcart.measures)
            ob = answer.get("observation")
            self.soakup(action["name"], action["args"], ob, answer, True)
            self.code_news(self.customer_lineitems(action["name"], ob, True))
            action = None
            if act_kind != "order" or not cart_lines:
                continue
            checked, sold_now = self.engine_reassess(cart_lines)
            if not checked:
                continue          # (the lines remain refused: the code fills the cart with other listings)
            self.rejected_lineitems.difference_update(set(cart_lines) - sold_now)
            by_id = self.engine_tablerows()
            cart_total = 0
            for ln in cart_lines:
                cart_total += ((by_id.get(ln) or {}).get("price") or 0.0) * self.handcart.retained(ln)
            budget = self.needs.outlay_lid
            too_dear = whether_numeric(budget) and cart_total > budget
            if too_dear:
                forbidden, kept = set(sold_now), None
            elif sold_now:
                forbidden, kept = set(sold_now), set(cart_lines) - sold_now
            else:
                # (nothing sold out and the cart fits the budget: the refusal was a change of mind the store has now
                # reported - the same cart goes again)
                action = make_order()
        if self.protocol.done or not answer or answer.get("finished") or not decline_name or act_kind == "decline":
            return
        if self.shifts_remaining() <= 0:
            return
        resend = False
        try:
            resend = bool(order_name) and act_kind == "order" and self.anew_approved(forbidden)
        except Exception as exc:
            self.journal.append({"kind": "again_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
        # (CF6: a refused order is sent again as long as no cart line is known sold out and something can fit)
        action = make_order() if resend else make_decline()
        final_reply = self.protocol.perform([action])
        self.journal.append({"kind": "forced_finish", "action": action, "reply": final_reply})

    def anew_approved(self, forbidden: set[str]) -> bool:
        """CF6: a refused cart may be sent again unchanged when it holds lines, none of them known sold out or to be
        avoided, and the records still offer a candidate for every product (code_picks; declining is right only when
        nothing can fit)."""
        cart_ids = set(self.handcart.measures)
        excluded = set(forbidden) | self.defunct_lineitems()
        if not cart_ids or cart_ids & excluded:
            return False
        return bool(self.engine_chooses(excluded))

    def intercept_upward(self) -> None:
        """TA4: when a model answers again after an outage, a plan the code made is replaced (once) by a model reading of
        the request, and shopper lines whose listener call failed are read now; the actor hears about both."""
        try:
            if self.protocol.silence or WRAPUP > self.protocol.surplus_alt():
                return
            added = []
            need_reread = bool(self.needs.perusal.get("code_reading")) and not self.__dict__.get("reread")
            if need_reread and self.protocol.final_response > self.__dict__.get("read_at", self.protocol.final_response)\
                    and self.protocol.surplus_alt() > WRAPUP + 60:
                self.reread = True
                answer = self.protocol.searchterm_llm([{"role": "user", "content": self.peruse_substance}], PERUSER, 3000)
                fresh_read = serialized_entity((answer or {}).get("content") or "") or {}
                has_roles = isinstance(fresh_read.get("roles"), dict)
                self.journal.append({"kind": "re_read", "ok": has_roles})
                if has_roles:
                    self.employ_perusal(fresh_read, True)
                    added.append("[plan updated] the models are responding again and have read the request; this reading "
                                 "takes the place of the first one:\n" + self.needs.synopsis(self.stage_lid, self.letter_lid))
            queued = self.needs.__dict__.get("pending") or []
            if queued and self.protocol.final_response > self.needs.__dict__.get("pending_at", 0.0) and self.protocol.surplus_alt() >= WRAPUP:
                self.needs.pending = []
                for group, question in queued[-3:]:
                    recap_alt = self.needs.receive(group, question)
                    if recap_alt:
                        added.append("[earlier lines from the shopper, read only now] " + recap_alt)
            if not added:
                return
            addendum = "\n".join(added) + "\n" + self.progress_lineitem()
            last_msg = self.converse[-1] if self.converse else None
            if last_msg and last_msg.get("role") in ("user", "tool") and isinstance(last_msg.get("content"), str):
                last_msg["content"] += "\n" + addendum
            else:
                self.converse.append({"role": "user", "content": addendum})
        except Exception as exc:
            self.journal.append({"kind": "catch_up_error", "error": f"{type(exc).__name__}: {exc}"[:300]})

    # ----------------------------------------------------------------------------------------------- observations
    def refused(self, tool_called: str, result: Any, envelope: dict[str, Any]) -> bool:
        """True for an order call that did not end the episode, or for any result carrying a refusal field."""
        is_map = isinstance(result, dict)
        if is_map and "refusal" in result:
            return True
        if tool_called != self.role("order") or envelope.get("finished"):
            return False
        return not (is_map and result.get("error") in ("invalid_args", "unknown_tool"))

    def annotations_toward(self, tool_called: str, result: Any, was_refused: bool) -> str:
        said_notes = []
        if isinstance(result, dict) and result.get("error") == "budget_exceeded":
            if tool_called == self.role("message") or self.letter_lid is not None:
                if self.letter_lid is None:
                    self.letter_lid = self.letters_transmitted
                else:
                    self.letter_lid = min(self.letter_lid, self.letters_transmitted)
            said_notes.append("This allowance is spent; make no further calls to this tool.")
        if was_refused and tool_called == self.role("decline") and getattr(self, "last_kind", None) == "decline":
            # (the store rejected the decline, so it holds that something can still be bought - the audit runs once more)
            self.audit_done = False
            self.abstain_sanctioned = False
            self.event_seen = True        # (CF1)
            said_notes.append("YOUR DECLINE WAS REFUSED and the episode goes on: the store does not agree that nothing fits. "
                           "Read the refusal and do what it says, look once more at the nearest candidates (their own "
                           "properties, stock and price measured against the current requirements), and order the one that "
                           "fits; decline again only if every one of them really fails.")
            return " ".join(said_notes)
        if was_refused:
            self.rejection_num += 1
            self.purchase_sanctioned = False
            self.event_seen = True        # (CF1: the order's reserve now keeps room for a repair)
            reread_note = ""
            try:
                reread_note = self.rejection_reperuse(result)
            except Exception as exc:
                self.journal.append({"kind": "refusal_reread_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
            self.rejected_lineitems.update(self.handcart.measures)
            long_texts: list[str] = []

            def harvest(node: Any, lvl: int = 0) -> None:
                if lvl >= 7:
                    return
                if isinstance(node, dict):
                    kids = list(node.values())
                elif isinstance(node, list):
                    kids = node[:20]
                else:
                    if isinstance(node, str) and len(node.split()) >= 4:
                        long_texts.append(node)
                    return
                for kid in kids:
                    harvest(kid, lvl + 1)
            harvest(result)
            self.refusal_texts = {" ".join(str(txt).lower().split()) for txt in long_texts}       # (v605f FI)
            said_notes.append("YOUR ORDER WAS REFUSED and the episode goes on; the cart has not changed. Carry out every change "
                           "the refusal states, repair the cart (remove any line that is no longer acceptable, such as a "
                           "sold-out item), check stock and price of the new pick again, and order once more." + reread_note)
            return " ".join(said_notes)
        if isinstance(result, dict) and len(result) >= 2 and tool_called != self.role("message"):
            pushed: list[str] = self.alert_lineitems(result)
            if pushed:
                prior_texts = self.__dict__.pop("refusal_texts", None) or set()
                is_echo = bool(prior_texts) and all(any(" ".join(str(one).lower().split()) in pt for pt in prior_texts) for one in pushed)
                if is_echo:
                    # (v605f FI: a push made at finish time appears on the next call as well - the refusal already brought it)
                    said_notes.append("This result only echoes the message that the refused finish already brought (dealt with "
                                   "above); it tells nothing about this result.")
                else:
                    self.event_seen = True    # (CF1)
                    said_notes.append("This result brings a change from the shopper. Act on it: " + " | ".join(pushed[:4]))
        return " ".join(said_notes)

    @staticmethod
    def rack_belonging(rec: dict[str, Any]) -> str:
        """The category label carried by a record ('' if there is none)."""
        for fk, fv in rec.items():
            if not isinstance(fv, str) or not 0 < len(fv.strip()) <= 60:
                continue
            low_key = str(fk).lower()
            if re.search(r"categor|shelf|department|aisle", low_key) and "id" not in low_key:
                return fv.strip().lower()
        return ""

    @staticmethod
    def slot_inventory(field: Any, content: Any) -> bool | None:
        """CB: what a single field says about stock - True (buyable now), False (not buyable), None (not a stock field, or
        empty). For a stock or availability field: one named 'out'/'sold' means False when it is true; any other means
        False when it is false, zero or negative, or text such as out, sold, none, not. An on-hand field that is false or
        zero, and back-order, pre-order or discontinued text, also mean False. This is the code's only stock reader
        (code_rows, sold_out, CB)."""
        fname = str(field).lower()
        if content is None or not re.search(r"stock|avail|on.?hand", fname):
            return None
        nil_qty = content is False or (whether_numeric(content) and content <= 0)
        cannot_buy = False
        if re.search(r"stock|avail", fname):
            if re.search(r"out|sold", fname):
                cannot_buy = content is True
            elif nil_qty:
                cannot_buy = True
            elif isinstance(content, str):
                cannot_buy = bool(re.search(r"\b(?:out|sold|none|unavailable|not|no|false)\b", content.lower()))
        if nil_qty and re.search(r"on.?hand", fname):
            cannot_buy = True
        if isinstance(content, str) and re.search(r"back.?order|pre.?order|discontinu", content.lower()):
            cannot_buy = True
        return not cannot_buy

    @staticmethod
    def inventory_condition(rec: dict[str, Any]) -> bool | None:
        """CB: a record's stock according to its fields (key_stock) - False when any field says it cannot be bought now,
        True when some field says it can, None when there is no stock field."""
        states = []
        for fk, fv in rec.items():
            st = Errand.slot_inventory(fk, fv)
            if st is not None:
                states.append(st)
        if not states:
            return None
        return False not in states

    @staticmethod
    def traded_depleted(rec: dict[str, Any]) -> bool:
        """A record with a stock field saying it cannot be bought now (key_stock)."""
        for fk, fv in rec.items():
            if Errand.slot_inventory(fk, fv) is False:
                return True
        return False

    def rack_tags(self) -> set[str]:
        """The category label(s) on which most listings read so far sit; empty while none is known."""
        per_shelf: dict[str, int] = {}
        already: set[str] = set()
        foreign = self.reference_external()

        def tally_row(rec: dict[str, Any]) -> None:
            lid = None
            for fk, fv in rec.items():
                if isinstance(fv, str) and "id" in str(fk).lower() and fv in self.vault.read:
                    lid = fv
                    break
            if not lid or lid in already or lid in foreign:
                return
            label = self.rack_belonging(rec)
            if label:
                already.add(lid)
                per_shelf[label] = per_shelf.get(label, 0) + 1
        for stored in self.vault.items:
            traverse_mappings(stored, tally_row)
        peak = max(per_shelf.values(), default=0)
        leaders = {label for label, cnt in per_shelf.items() if cnt == peak}
        if not per_shelf or not self.needs.items():
            return leaders
        try:
            # (with several products: per product, its own most-read label - or, when none of its records was read, the
            # label of it seen in rows; for a product that no label matches, the overall most-read labels fill in)
            home_sets = self.landing_racks()
            selected: set[str] = set()
            for hs in home_sets:
                read_here = [lab for lab in hs if lab in per_shelf]
                if not read_here:
                    selected |= set(hs)
                    continue
                peak_here = max(per_shelf[lab] for lab in read_here)
                selected |= {lab for lab in read_here if per_shelf[lab] == peak_here}
            if not all(home_sets):
                for label in sorted(per_shelf, key=lambda lab: -per_shelf[lab]):
                    if len(selected) >= len(home_sets):
                        break
                    selected.add(label)
            return selected or leaders
        except Exception:
            return leaders

    @staticmethod
    def tablerow_flag(rec: dict[str, Any]) -> str:
        """A result row's identity: its id (a key called id, else one ending in id), else the whole row serialised."""

        def id_like(v: Any) -> bool:
            return isinstance(v, (str, int)) and not isinstance(v, bool)
        lowered = [(str(k).lower(), v) for k, v in rec.items()]
        for lk, v in lowered:
            if lk == "id" and id_like(v):
                return str(v)
        for lk, v in lowered:
            if lk.endswith("id") and id_like(v):
                return str(v)
        return json.dumps(rec, sort_keys=True, default=str)

    def seek_slots(self, args_in: dict[str, Any]) -> tuple[list[tuple], tuple, bool, set]:
        """For a search call: (its free-text arguments as (key, words in order with repeats removed), its mode (the enum
        arguments), whether it asks for the first page, its paging keys). A word is a whitespace-separated token with
        outer punctuation stripped; a text is kept only if every word is plain letters or digits, two characters or
        more (stores split hyphens, slashes and decimals and drop one-letter words in differing ways, so other words
        cannot be counted on). A paging argument (a page or an offset - not a page size or a price) asks for the first
        page when it is no higher than its schema minimum (else its default, else 0, or 1 for a page number)."""
        raw = self.structures.get(self.role("search") or "")
        raw = raw.get("properties") if isinstance(raw, dict) else None
        arg_map = raw if isinstance(raw, dict) else {}

        def spec_of(a_name: str) -> dict[str, Any]:
            g = arg_map.get(a_name)
            return g if isinstance(g, dict) else {}
        text_parts: list[tuple] = []
        mode_pairs: list[tuple] = []
        on_first = True
        page_keys: set = set()
        tok_arg = self.lexeme_slot()
        for a_name, v in args_in.items():
            low_name = str(a_name).lower()
            if isinstance(v, str):
                if a_name == tok_arg:
                    page_keys.add(a_name)
                    if on_first:
                        on_first = not v.strip()
                elif spec_of(a_name).get("enum"):
                    mode_pairs.append((a_name, v))
                else:
                    cleaned = (tk.strip("\"'()[]{},.;:!?|/") for tk in v.lower().split())
                    wtuple = tuple(dict.fromkeys(wd for wd in cleaned if re.search(r"[a-z0-9]", wd)))
                    if wtuple and all(re.fullmatch(r"[a-z0-9]{2,}", wd) for wd in wtuple):
                        text_parts.append((a_name, wtuple))
                continue
            if not whether_numeric(v) or not re.search(r"page|offset|first|start|skip|from|cursor", low_name):
                continue
            if re.search(r"size|per_?page|limit|price|cost", low_name):
                continue
            page_keys.add(a_name)
            floor_val = spec_of(a_name).get("minimum", spec_of(a_name).get("default"))
            if not whether_numeric(floor_val):
                floor_val = 1 if "page" in low_name else 0
            if on_first:
                on_first = v <= floor_val
        return text_parts, tuple(sorted(mode_pairs)), on_first, page_keys

    def term_principle(self, args_in: dict[str, Any], result: dict[str, Any], found_rows: list) -> str:
        """'any' when a query extending another (its words first, then more) matched more listings, 'all' when it matched
        fewer, '' while that is unknown; compared within the same mode and other arguments, on first pages, per free-text
        argument. Only extensions count, since a store reading only a query's opening words gives unrelated totals for
        the same words in another order."""
        text_parts, mode_sig, on_first, page_keys = self.seek_slots(args_in)
        if not on_first:
            return self.word_kind
        ints = [val for val in result.values() if isinstance(val, int) and not isinstance(val, bool)]
        match_total = max(ints + [len(found_rows)])
        for a_name, wseq in text_parts:
            context = (a_name, mode_sig)
            rest_args = tuple(sorted((ok, str(ov)) for ok, ov in args_in.items() if ok != a_name and ok not in page_keys))
            subset_hit = False
            for old_ctx, old_rest, old_words, old_total in self.term_observed:
                if old_ctx != context or old_rest != rest_args or old_total == match_total:
                    continue
                len_now, len_then = len(wseq), len(old_words)
                if len_now > len_then and wseq[:len_then] == old_words:
                    self.word_kind = "all" if match_total <= old_total else "any"
                elif len_then > len_now and old_words[:len_now] == wseq:
                    self.word_kind = "all" if match_total >= old_total else "any"
                else:
                    # (EB5: a strict word-subset pair in a different order - over the words the store reads, once those are
                    # known - whose larger side matched fewer listings; it decides 'all' only while no prefix pair has)
                    depth = self.foremost_terms or None
                    now_set, old_set = set(wseq[:depth]), set(old_words[:depth])
                    if (now_set > old_set and match_total < old_total) or (old_set > now_set and old_total < match_total):
                        subset_hit = True
            if subset_hit and not self.word_kind:
                self.word_kind = "all"
            self.term_observed.append((context, rest_args, wseq, match_total))
        return self.word_kind

    def foremost_principle(self, args_in: dict[str, Any], result: dict[str, Any], found_rows: list) -> int:
        """K when the store reads only a query's first K words, 0 while unknown. First-page searches in a single mode that
        agree on every other argument are compared per free-text argument: a pair matching on its first k words shows the
        same total and rows when nothing past word k was read (K <= k), and different ones when something was (K > k). K
        is claimed once these bounds meet, which also proves the argument is read in that mode. A 'same' pair counts only
        when one query extends the other, and one at or below a proven 'read' depth is coincidence (word variants, words
        that every listing on a shelf carries). Only plain-word queries count (see search_keys)."""
        text_parts, mode_sig, on_first, page_keys = self.seek_slots(args_in)
        if not on_first:
            return self.foremost_terms
        ints = [val for val in result.values() if isinstance(val, int) and not isinstance(val, bool)]
        match_total = max(ints + [len(found_rows)])
        row_ids = tuple(self.tablerow_flag(rw) for rw in found_rows)
        for a_name, wseq in text_parts:
            context = (a_name, mode_sig)
            rest_args = tuple(sorted((ok, str(ov)) for ok, ov in args_in.items() if ok != a_name and ok not in page_keys))
            bounds_pair = self.foremost_boundaries.setdefault(context, [-1, set()])
            for old_ctx, old_rest, old_words, old_total, old_marks in self.foremost_observed:
                if old_ctx != context or old_rest != rest_args or old_words == wseq:
                    continue
                common = 0
                for a_w, b_w in zip(wseq, old_words):
                    if a_w != b_w:
                        break
                    common += 1
                if match_total != old_total or row_ids != old_marks:
                    if common > bounds_pair[0]:
                        bounds_pair[0] = common
                elif common and common == min(len(wseq), len(old_words)) and len(set(row_ids)) >= 3:
                    bounds_pair[1].add(common)
            self.foremost_observed.append((context, rest_args, wseq, match_total, row_ids))
        candidates = []
        for context, (lo, hi) in self.foremost_boundaries.items():
            over = [depth_k for depth_k in hi if depth_k > lo]
            if over:
                candidates.append((min(over), context))
        candidates.sort()
        agreed = [(depth_k, context) for depth_k, context in candidates if depth_k == self.foremost_boundaries[context][0] + 1]
        self.foremost_terms, self.foremost_context = agreed[0] if agreed else (0, None)
        return self.foremost_terms

    def lexeme_slot(self) -> str | None:
        """The search tool's string paging argument (a continuation token returned with results), if it has one."""
        arg_map = (self.structures.get(self.role("search") or "") or {}).get("properties")
        if not isinstance(arg_map, dict):
            return None
        for a_name, a_spec in arg_map.items():
            if not isinstance(a_spec, dict) or a_spec.get("type") != "string" or a_spec.get("enum"):
                continue
            if re.search(r"bookmark|cursor|token|resume|continu|after|next|page|offset|start", str(a_name).lower()):
                return a_name
        return None

    @staticmethod
    def subsequent_lexeme(result: Any) -> str | None:
        """The continuation token a result supplies for its next page (None on the final page)."""
        if isinstance(result, dict):
            for fk, fv in result.items():
                if not isinstance(fv, str) or not fv.strip():
                    continue
                if re.search(r"next|onward|cursor|bookmark|token|continu|more", str(fk).lower()):
                    return fv
        return None

    def sheet_param(self) -> tuple[str, Any, Any, float | None] | None:
        """The search tool's paging argument read from its schema: (name, whether it counts whole pages or "token",
        first value, highest value or None). For a token argument the first page is the empty token."""
        arg_map = (self.structures.get(self.role("search") or "") or {}).get("properties")
        for a_name, a_spec in (arg_map.items() if isinstance(arg_map, dict) else ()):
            low_name = str(a_name).lower()
            if not isinstance(a_spec, dict) or a_spec.get("type") not in ("integer", "number"):
                continue
            if not re.search(r"page|offset|first|start|skip|from|cursor", low_name) or re.search(r"size|per_?page|limit|price|cost", low_name):
                continue
            whole = "page" in low_name and not re.search(r"offset|skip|first|start|from", low_name)
            first_val = a_spec.get("minimum", a_spec.get("default"))
            if not whether_numeric(first_val):
                first_val = 1 if whole else 0
            ceiling = a_spec.get("maximum")
            return a_name, whole, first_val, ceiling if whether_numeric(ceiling) else None
        tok_arg = self.lexeme_slot()
        if not tok_arg:
            return None
        return tok_arg, "token", "", None

    @staticmethod
    def wordbase(term: str) -> str:
        if len(term) <= 3 or not term.endswith("s") or term.endswith("ss"):
            return term
        return term[:-1]

    def rack_tablerows(self) -> dict[str, tuple[float, str, str]]:
        """Rows seen on the requested category that are in stock and within the budget: id -> (price, title, maker)."""
        labels = self.rack_tags()
        budget = self.needs.outlay_lid
        measures = self.needs.quantity()
        rows_by_id: dict[str, tuple[float, str, str]] = {}
        if not labels:
            return rows_by_id

        def collect_row(rec: dict[str, Any]) -> None:
            lid = None
            for fk, fv in rec.items():
                if isinstance(fv, str) and "id" in str(fk).lower() and fv in self.vault.ids:
                    lid = fv
                    break
            if not lid or lid in self.rejected_lineitems or self.rack_belonging(rec) not in labels:
                return
            if self.traded_depleted(rec):
                rows_by_id.pop(lid, None)
                return
            price_now = None
            for fk, fv in rec.items():
                if whether_numeric(fv) and not isinstance(fv, bool) and re.search(r"price|cost", str(fk).lower()):
                    price_now = fv
                    break
            raw_title = rec.get("title")
            if isinstance(raw_title, str) and 0 < len(raw_title) <= 300:
                heading = rec["title"]
            else:
                heading = ""
                for fk, fv in rec.items():
                    if not isinstance(fv, str) or not 0 < len(fv) <= 300:
                        continue
                    low_key = str(fk).lower()
                    if not re.search(r"title|heading|headline|name", low_key) or "id" in low_key:
                        continue
                    if self.rack_belonging({fk: fv}) or rubbish_slot(fk):
                        continue
                    heading = fv
                    break
            brand = " ".join(fv for fk, fv in rec.items()
                             if isinstance(fv, str) and re.search(r"brand|make|manufactur|vendor", str(fk).lower()))
            if price_now is not None:
                rows_by_id[lid] = (float(price_now), heading, brand)
        for stored in self.vault.items:
            traverse_mappings(stored, collect_row)
        for lid, (price_now, heading, brand) in list(rows_by_id.items()):
            latest = self.vault.price(lid)
            if latest is not None:
                price_now = latest
            if whether_numeric(budget) and price_now * measures > budget:
                rows_by_id.pop(lid)
                continue
            rows_by_id[lid] = (float(price_now), heading, brand)
        return rows_by_id
    def tag_searchterms(self) -> list[str]:
        """Search words taken from the requested shelf label itself: the stem of the closing word of each part (the kind
        that part names), the label's opening two words as one phrase, those closing words as written, then the rest."""
        gathered: list[str] = []
        for shelf_name in sorted(self.rack_tags()):
            groups: list[list[str]] = []
            run: list[str] = []
            for raw_word in shelf_name.lower().split():
                token = raw_word.strip(".,;:!?'\"()[]")
                if token not in BOND_TERMS and re.fullmatch(r"[a-z0-9][a-z0-9'-]*", token):
                    run.append(token)
                else:
                    if run:
                        groups.append(run)
                    run = []
            if run:
                groups.append(run)
            flat = [tok for grp in groups for tok in grp]
            tails = [grp[-1] for grp in groups]
            gathered.extend(self.wordbase(t) for t in tails)
            if len(flat) > 1:
                gathered.append(" ".join(flat[:2]))
            gathered += tails
            gathered += [tok for tok in flat if tok not in tails]
        return list(dict.fromkeys(gathered))

    def category_searchterms(self, as_pairs: bool) -> list[str]:
        """Terms for the kinds of object the shelf holds, drawn from titles of in-stock shelf rows already seen: adjacent
        word pairs (as_pairs) or lone words, each word 3+ letters, present in at least two different titles, neither a
        function word nor the row's maker, and not made only of label words; ordered by the cheapest row carrying them."""
        label_stems = {self.wordbase(tok) for lab in self.rack_tags() for tok in re.findall(r"[a-z0-9]+", lab)}
        by_title: dict[str, tuple[float, str]] = {}
        for price, name, brand in self.rack_tablerows().values():
            norm = name.lower().strip()
            if norm:
                prior = by_title.get(norm, (price, ""))[0]
                by_title[norm] = (min(price, prior), brand.lower())
        tally: dict[str, int] = {}
        low_cost: dict[str, float] = {}
        display: dict[str, str] = {}
        for name, (price, brand) in by_title.items():
            brand_words = set(re.findall(r"[a-z]+", brand))
            toks = re.findall(r"[a-z]+", name)
            ok = [len(t) > 2 and t not in BOND_TERMS and t not in brand_words for t in toks]
            if as_pairs:
                grams = [(toks[j], toks[j + 1]) for j in range(len(toks) - 1) if ok[j] and ok[j + 1]]
            else:
                grams = [(toks[j],) for j, good in enumerate(ok) if good]
            counted: set[str] = set()
            for gram in grams:
                stemmed = " ".join(self.wordbase(t) for t in gram)
                if stemmed in counted or all(self.wordbase(t) in label_stems for t in gram):
                    continue
                counted.add(stemmed)
                tally[stemmed] = 1 + tally.get(stemmed, 0)
                low_cost[stemmed] = min(low_cost.get(stemmed, price), price)
                display.setdefault(stemmed, " ".join(gram))
        repeated = [s for s, seen in tally.items() if seen > 1]
        repeated.sort(key=lambda s: (low_cost[s], -tally[s], s))
        return [display[s] for s in repeated]

    def survey_subsequent(self, rotation: int) -> tuple[dict[str, Any] | None, bool]:
        """(next shelf slice to read, whether it is a page of the shelf listing): the listing's next unread page while any
        remain, then by rotation a kind pair, a kind word and a label term (the first pages of queries not run yet, in the
        mode the actor's keyword searches used). Offline runs showed listing pages first, then title kind words ordered
        by their cheapest carrier, to beat label-word queries page for page."""
        paging = self.sheet_param()
        wanted = self.rack_tags()
        walks = self.__dict__.get("page_book", {})
        nothing = (None, False)
        if not (paging and wanted and walks):
            return nothing
        page_key, _, opening_at, _ = paging
        already = self.__dict__.get("sweep_tried", set())

        def fresh(call: dict[str, Any]) -> bool:
            return json.dumps(call, sort_keys=True, default=str) not in already

        listing = None
        for trail in walks.values():
            if trail.get("id_order") and trail["shelves"] and trail["shelves"] <= wanted and self.subsequent_sheet(trail) is not None:
                if listing is None or trail["total"] > listing["total"]:
                    listing = trail

        def keyword_walks(flag: Any) -> list[dict[str, Any]]:
            return [t for t in walks.values() if t.get("id_order") is flag and self.loose_passage(t["args"])]
        ranked = keyword_walks(False) or keyword_walks(None)
        base_args: dict[str, Any] | None = None
        passage_param = None
        if ranked:
            schema = (self.structures.get(self.role("search") or "") or {}).get("properties") or {}
            latest = ranked[-1]["args"]

            def anchored(name: str, val: Any) -> bool:
                if not isinstance(val, str) or (schema.get(name) or {}).get("enum"):
                    return True
                return val.strip().lower() in wanted or name == page_key
            free = [name for name, val in latest.items() if not anchored(name, val)]
            if self.foremost_context and self.foremost_context[0] in free:
                passage_param = self.foremost_context[0]
            else:
                passage_param = max(free, key=lambda name: len(latest[name].split()), default=None)
            if passage_param:
                base_args = {name: val for name, val in latest.items() if name != passage_param and anchored(name, val)}

        def lead_sig(text: str) -> tuple:
            return tuple(self.wordbase(t) for t in re.findall(r"[a-z0-9]{2,}", text.lower())[:2])
        covered = {lead_sig(" ".join(self.loose_passage(t["args"]))) for t in walks.values()
                   if opening_at in t["read"] and t.get("id_order") is not True}

        def make_query(text: str) -> dict[str, Any] | None:
            sig = lead_sig(text)
            if base_args is None or not sig or sig in covered:
                return None
            call = {**base_args, passage_param: text}
            return call if fresh(call) else None
        one_word = {self.wordbase(q) for q in self.tag_searchterms() if " " not in q}

        def page_twos() -> list[dict[str, Any]]:
            # a single-word label search whose opening page held rows of the shelf: queue its following page
            found_pages = []
            for trail in walks.values():
                spoken = [self.wordbase(t.strip().lower()) for t in self.loose_passage(trail["args"])]
                single = len(trail["read"]) == 1 and spoken and spoken[0] in one_word
                if single and trail["shelves"] & wanted and self.subsequent_sheet(trail) is not None:
                    found_pages.append({**trail["args"], page_key: self.subsequent_sheet(trail)})
            return found_pages

        def from_listing() -> list[dict[str, Any]]:
            if not listing:
                return []
            return [{**listing["args"], page_key: self.subsequent_sheet(listing)}]

        def from_texts(texts: list[str]) -> list[dict[str, Any]]:
            return [c for c in map(make_query, texts) if c]
        sources = [from_listing,
                   lambda: from_texts(self.category_searchterms(True)[:6]),
                   lambda: from_texts(self.category_searchterms(False)[:6]),
                   lambda: from_texts(self.tag_searchterms())]
        price_call, finished = self.roofline_survey(already)
        if price_call is not None:
            return price_call, True     # CG3: a price-filtered listing under the cheapest in-stock row known goes first
        if finished:
            return nothing              # and once it shows nothing cheaper in stock, the sweep is over
        extra = [c for c in page_twos() if fresh(c)]
        if rotation % 2 == 1 and extra:
            return extra[0], False
        for idx in [0, *(1 + (rotation + s) % 3 for s in range(3))]:
            hit = next((c for c in sources[idx]() if fresh(c)), None)
            if hit is not None:
                return hit, idx == 0
        return nothing

    def survey_stage(self) -> bool:
        """When the cheapest rule holds and a change has removed every property requirement: one more scripted read of a
        shelf slice, keeping free the steps that an order plus one repair need; after the final read, the cheapest rows
        seen are listed."""
        nd = self.needs
        blocked = (nd.priority() != "cheapest" or not nd.let_go or nd.traits_remaining()
                   or not self.role("search") or self.protocol.done)
        if blocked:
            return False
        # steps to keep: drop the old line, add the pick, re-read it, order
        reserve = 4 if self.handcart.measures else 3
        if not self.purchases_attempted:
            reserve = max(3 + reserve, INITIAL_PURCHASE_VIA)     # plus a repair should the first order be refused

        def has_room(count: int) -> bool:
            return count < SURVEY_PERUSES and self.shifts_remaining() > reserve and self.protocol.surplus_alt() >= WRAPUP + 40
        step_no = self.__dict__.get("sweep_turn", 0)
        picked, is_listing = self.survey_subsequent(step_no) if has_room(self.__dict__.get("swept", 0)) else (None, False)
        if picked is None:
            return False
        self.swept = 1 + self.__dict__.get("swept", 0)
        self.sweep_turn = step_no if is_listing else step_no + 1
        encoded = json.dumps(picked, sort_keys=True, default=str)
        self.__dict__.setdefault("sweep_tried", set()).add(encoded)
        call_spec = {"id": f"sweep_{self.swept}_{self.protocol.pulse}", "type": "function",
                     "function": {"name": self.role("search"), "arguments": json.dumps(picked)}}
        intro = ("Every property requirement is gone now, so the lowest-priced in-stock listing anywhere on the shelf is the "
                 "winner: I look at one more part of the shelf before picking.")
        self.converse.append({"role": "assistant", "content": intro, "tool_calls": [call_spec]})
        self.covering = True
        try:
            replies = self.route_alt([call_spec], scripted=True)
        finally:
            self.covering = False
        go_on = has_room(self.swept) and self.survey_subsequent(self.sweep_turn)[0] is not None
        cheapest = sorted(self.rack_tablerows().items(), key=lambda kv: (kv[1][0], kv[0]))[:6] if replies and not go_on else []
        if cheapest:
            listed = "; ".join(f"{rid} '{name[:80]}' {cost:g}" for rid, (cost, name, _) in cheapest)
            last_id, last_text = replies[-1]
            replies[-1] = (last_id, last_text + "\n[shelf sweep done] Of everything seen, the lowest-priced in-stock rows on the "
                           "requested category inside the budget: " + listed + ". Unless a change rules it out, the cheapest "
                           "of them is the pick: make it the only line in the cart, re-check its stock and price, then order it.")
        self.later_responses(replies)
        return True

    def loose_passage(self, call: dict[str, Any]) -> list[str]:
        """Values of a search call's free-text arguments (arguments with an enum, and the paging token, skipped)."""
        schema = (self.structures.get(self.role("search") or "") or {}).get("properties")
        if not isinstance(schema, dict):
            schema = {}
        token = self.lexeme_slot()
        return [val for name, val in call.items()
                if isinstance(val, str) and name != token and not (schema.get(name) or {}).get("enum")]

    def ledger_sheet(self, call: dict[str, Any], found_rows: list, count: int, payload: Any = None) -> None:
        """Records one search page under its walk (all calls equal but for the paging argument)."""
        paging = self.sheet_param()
        if not (paging and found_rows):
            return
        page_key, how, first_at = paging[0], paging[1], paging[2]
        where = call.get(page_key, first_at)
        if how != "token":
            if not whether_numeric(where):
                return
        else:
            where = where.strip() if isinstance(where, str) else ""
        rest = {name: val for name, val in call.items() if name != page_key}
        sig = json.dumps(rest, sort_keys=True, default=str)
        entry = self.__dict__.setdefault("page_book", {}).setdefault(
            sig, {"args": rest, "read": set(), "size": 0, "total": 0, "shelves": set()})
        entry["read"].add(where)
        if how == "token":
            entry.setdefault("tokens", {})[where] = self.subsequent_lexeme(payload)
        entry["size"] = max(entry["size"], len(found_rows))
        entry["total"] = max(entry["total"], count)
        dict_rows = [row for row in found_rows if isinstance(row, dict)]
        entry["shelves"] |= {self.rack_belonging(row) for row in dict_rows} - {""}
        row_shelf = entry.setdefault("row_shelf", {})
        for row in dict_rows:
            row_shelf.setdefault(self.tablerow_flag(row), self.rack_belonging(row))
        marks = [self.tablerow_flag(row) for row in dict_rows]
        if where == first_at:
            entry["first"] = tuple(marks)
            if count <= len(found_rows):
                entry["every"] = dict_rows      # an opening page that holds every match
        if len(marks) > 4:
            # five or more rows sorted by id: the store keeps a fixed listing order rather than ranking
            def natural(mark: str) -> list:
                return [int(bit) if bit.isdigit() else bit for bit in re.split(r"(\d+)", mark)]
            still = entry.get("id_order", True) is not False
            entry["id_order"] = still and (marks == sorted(marks) or marks == sorted(marks, key=natural))

    def subsequent_sheet(self, trail: dict[str, Any]) -> Any:
        """The earliest paging value of a walk not read yet that may still bring unseen rows; None when there is none."""
        _, how, start, top = self.sheet_param()
        done_at = trail["read"]
        if how == "token":
            # the token returned by a read page whose successor has not been read
            tokens = trail.get("tokens") or {}
            return next((t for t in tokens.values() if isinstance(t, str) and t not in done_at), None)
        start = min([start, *(v for v in done_at if whether_numeric(v))])
        per_page = max(1, trail["size"])
        stride = 1 if how else per_page
        for i in range(1, 400):
            cand = start + i * stride
            if i * per_page >= trail["total"] or (top is not None and cand > top):
                break
            if cand not in done_at:
                return cand
        return None

    def setting_rating(self, call: dict[str, Any]) -> int:
        """Count of in-force property requirements whose value has a word among the search's leading query words."""
        def close(a: str, b: str) -> bool:
            return min(len(a), len(b)) > 2 and (a.startswith(b) or b.startswith(a))
        value_sets = []
        for need in self.needs.traits_remaining():
            value = need.partition(":")[2]
            if not reference_passage(value):
                value_sets.append(re.findall(r"[a-z0-9]{3,}", value.lower()))
        lead = [tok for text in self.loose_passage(call) for tok in re.findall(r"[a-z0-9]{2,}", text.lower())]
        if self.foremost_terms:
            lead = lead[: self.foremost_terms]
        return sum(1 for vw in value_sets if any(close(a, b) for a in lead for b in vw))

    def tablerow_strikes(self, found_rows: list, query_words: list[str] | None = None) -> list[str]:
        """Unread search rows that are in stock, inside the budget and (if known) on the requested category. Given the
        query's words and a request that names a value by reference: unread rows whose shop or brand the query names, and
        which the reference or a read record of another seller names as well (pointed_words), are set aside (ref_hits)
        whatever their price, stock or category - such a listing gets read, never bought."""
        limit = self.needs.outlay_lid
        qty = self.needs.quantity()
        wanted = self.rack_tags()
        good_ids: list[str] = []
        named: set[str] = set()
        if self.needs.perusal.get("references"):
            named = {tok for piece in query_words or [] for tok in re.findall(r"[a-z0-9]+", str(piece).lower())}
        ref_list = self.__dict__.setdefault("ref_hits", [])
        for row in (r for r in found_rows if isinstance(r, dict)):
            mark = self.tablerow_flag(row)
            unread = mark in self.vault.ids and mark not in self.vault.read
            if named and unread:
                sellers = [set(re.findall(r"[a-z0-9]+", val.lower())) for fld, val in row.items()
                           if isinstance(val, str) and val.strip() and "id" not in str(fld).lower()
                           and re.search(r"shop|seller|store|vendor|brand|make|manufactur", str(fld).lower())]
                if any(s and s <= named and s <= self.indicated_terms() for s in sellers) and mark not in ref_list:
                    ref_list.append(mark)
            price = next((val for fld, val in row.items() if whether_numeric(val) and re.search(r"price|cost", str(fld).lower())), None)
            shelf = self.rack_belonging(row)
            keep = (unread and not (whether_numeric(limit) and whether_numeric(price) and price * qty > limit)
                    and not self.traded_depleted(row) and not (wanted and shelf and shelf not in wanted))
            if keep:
                good_ids.append(mark)
                self.__dict__.setdefault("hit_shelf", {})[mark] = shelf
        return good_ids

    def indicated_terms(self) -> set[str]:
        """Words naming some other listing's seller: those of the request's references, plus the text fields (titles and
        seller fields excluded) of read records minus that record's own seller words (a redirect chain tends to name the
        next seller there, often inside the seller's notes)."""
        memoized = self.__dict__.get("pointed_memo")
        if memoized and memoized[0] == len(self.vault.items):
            return memoized[1]
        collected: set[str] = set()
        try:
            for ref in self.needs.perusal.get("references") or []:
                collected.update(re.findall(r"[a-z0-9]+", str(ref).lower()))

            def harvest(node: dict[str, Any]) -> None:
                if self.tablerow_flag(node) in self.vault.read:
                    own: set[str] = set()
                    notes: list[str] = []
                    for fld, val in node.items():
                        name = str(fld).lower()
                        if not isinstance(val, str):
                            pass
                        elif (re.search(r"shop|seller|store|vendor|brand|make|manufactur", name) and "id" not in name
                              and len(val) <= 40 and not re.search(r"note|text|desc|comment|detail|info|message|section", name)):
                            own.update(re.findall(r"[a-z0-9]+", val.lower()))     # a seller's name (its notes count as text)
                        elif not re.search(r"title|heading|headline|name|label|caption", name):
                            notes.append(val.lower())
                    for note in notes:
                        collected.update(set(re.findall(r"[a-z0-9]+", note)) - own)
            for record in self.vault.items:
                traverse_mappings(record, harvest)
        except Exception:
            pass
        self.pointed_memo = (len(self.vault.items), collected)
        return collected

    def examine_sheets(self, cap_n: int = 2) -> list[dict[str, Any]]:
        """Call arguments for the next unread page of every search whose leading words hold a required value and whose
        total exceeds the pages read so far: more required values first, then the smaller total."""
        paging = self.sheet_param()
        if not paging:
            return []
        wanted = self.rack_tags()
        ranked = []
        for trail in self.__dict__.get("page_book", {}).values():
            following = self.subsequent_sheet(trail)
            off_shelf = wanted and trail["shelves"] and not trail["shelves"] & wanted
            hits = 0 if following is None or off_shelf else self.setting_rating(trail["args"])
            if hits:
                ranked.append((-hits, trail["total"], {**trail["args"], paging[0]: following}))
        ranked.sort(key=lambda entry: entry[:2])
        return [entry[2] for entry in ranked[:cap_n]]

    def setting_explore(self, typed: list[str]) -> tuple[str, str]:
        """(property, query led by the value) when a keyword query's opening words contain a word of a required property's
        NAME but no word of any required value or of the product: a store's keyword index covers listing values, titles
        and categories, so such a query only keeps listings that repeat the name. The suggested query is the value from
        its first figure on (comparison words gone), else the whole value, followed by the product as the request says it."""
        miss = ("", "")
        if self.word_kind == "any":
            return miss

        def opening(text: str) -> list[str]:
            return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1][: self.foremost_terms or 2]

        def close(a: str, b: str) -> bool:
            if not any(ch.isdigit() for ch in a + b):
                return min(len(a), len(b)) > 2 and (a.startswith(b) or b.startswith(a))
            if a.lstrip("0") == b.lstrip("0"):
                return True
            return a.isdigit() and b.startswith(a) and not b[len(a):len(a) + 1].isdigit()
        lead = opening(" ".join(typed))
        item_words = " ".join(re.findall(r"[a-z0-9]+", str(self.needs.perusal.get("product") or "").lower())[:3])
        pairs = []
        for need in self.needs.traits_remaining():
            label, colon, value = need.partition(":")
            if colon and not reference_passage(value):
                simple = re.sub(r"\((?:stated|revealed[^)]*)\)", "", value)
                pairs.append((label.strip(), re.split(r"[,;]", simple)[0].strip().lower()))
        own_words = [t for _, value in pairs for t in re.findall(r"[a-z0-9]{2,}", value)] + item_words.split()
        if not lead or any(close(a, b) for a in lead for b in own_words):
            return miss
        for label, value in pairs:
            if value and any(close(a, b) for a in lead for b in re.findall(r"[a-z0-9]{2,}", label.lower())):
                first_digit = re.search(r"\d", value)
                core = value[first_digit.start():] if first_digit else value
                probe = f"{core.strip(' .;')} {item_words}".strip()
                head = opening(probe)
                for trail in self.__dict__.get("page_book", {}).values():
                    ran = opening(" ".join(self.loose_passage(trail["args"])))
                    if len(ran) == len(head) and all(close(a, b) for a, b in zip(head, ran)):
                        return miss
                return label, probe
        return miss

    # ------------------------------------------------------------------------------------- walking toward the cheapest
    def executed_searchterm(self, phrase: str) -> bool:
        """True when some search already run led with the same words as `phrase` (as many as the store reads, compared
        as stems in any order)."""
        span_n = self.foremost_terms or 3

        def sig(words: str) -> list[str]:
            return sorted(self.wordbase(t) for t in re.findall(r"[a-z0-9]+", words.lower())[:span_n])
        target = sig(str(phrase))
        return any(sig(" ".join(self.loose_passage(t["args"]))) == target for t in self.__dict__.get("page_book", {}).values())

    def seek_blueprint(self) -> tuple[dict[str, Any], str] | None:
        """For a fresh query: (the remaining arguments, the name of the text argument) of the latest search that has a
        single free-text argument."""
        trails = list(self.__dict__.get("page_book", {}).values())
        for trail in trails[::-1]:
            call = trail["args"]
            # CG: a price-filtered listing call is not a keyword search one can vary
            texts = [] if self.catalog_advert(call) else self.loose_passage(call)
            if len(texts) == 1:
                text_keys = [name for name, val in call.items() if isinstance(val, str) and val in texts and val.strip()]
                if len(text_keys) == 1:
                    return {name: val for name, val in call.items() if name != text_keys[0]}, text_keys[0]
        return None

    def term_snare(self, typed: list[str]) -> tuple[str, str]:
        """(note, query) when an every-word keyword query opens with a word that narrows it the wrong way: a word of a
        required property's NAME carried by no required value, product or category word ("round TOE boot" drops the
        listings whose field holds only the value), or a word only a dropped requirement's value carries. The offered
        query omits that word (adding the product's kind word), provided no such search ran and it still holds a
        required value of its own."""
        miss = ("", "")
        try:
            match_rule = self.word_kind
            if match_rule == "any":
                return miss
            pieces = (t.strip("\"'()[]{},.;!?|").lower() for t in " ".join(typed).split())
            toks = [t for t in pieces if re.search(r"[a-z0-9]", t)]
            # EB6: in an every-word store whose word count is still unknown, all tokens are read
            if self.foremost_terms:
                span_n = self.foremost_terms
            else:
                span_n = max(3, len(toks)) if match_rule == "all" else 3
            del toks[span_n:]

            def close(a: str, b: str) -> bool:
                if any(ch.isdigit() for ch in a + b):
                    return a.lstrip("0") == b.lstrip("0")
                return min(len(a), len(b)) > 2 and (a.startswith(b) or b.startswith(a))

            def bits(tok: str) -> list[str]:
                return [b for b in re.findall(r"[a-z0-9]+", tok) if len(b) > 1]

            def hits(tok_list: list[str], pool: Any) -> bool:
                return any(close(b, w) for tok in tok_list for b in bits(tok) for w in pool)
            triples = []
            for need in self.needs.traits_remaining():
                tag, colon, value = need.partition(":")
                if colon and not reference_passage(value):
                    triples.append((re.split(r"\s*\(", tag)[0].lower(), tag,
                                    re.sub(r"\((?:stated|revealed[^)]*)\)", "", value).lower()))
            value_words = {w for _, _, v in triples for w in re.findall(r"[a-z0-9]{2,}", v)}
            products = [r["product"] for r in self.needs.items()] or [str(self.needs.perusal.get("product") or "")]
            kind_set = set(re.findall(r"[a-z0-9]{3,}", " ".join(products + sorted(self.rack_tags())).lower())) - BOND_TERMS
            name_words = {w for n, _, _ in triples for w in re.findall(r"[a-z]{3,}", n)} - BOND_TERMS
            gone_words = {w for old in self.needs.let_go for w in re.findall(r"[a-z]{3,}", old.partition(":")[2].lower())}
            gone_words -= BOND_TERMS
            flagged: list[str] = []
            reasons: list[str] = []
            for tok in toks:
                if not bits(tok) or hits([tok], value_words | kind_set):
                    why = ""
                elif hits([tok], name_words):
                    why = "name"
                elif hits([tok], gone_words):
                    why = "dropped"
                else:
                    why = ""
                if why:
                    flagged.append(tok)
                    reasons.append(why)
            rest = [tok for tok in toks if tok not in flagged]
            retained = [(tag, v) for _, tag, v in triples if hits(rest, re.findall(r"[a-z0-9]{2,}", v))] if flagged else []
            if not retained:
                return miss
            if not hits(rest, kind_set):
                idx = self.ware_belonging(retained[0][0]) if self.needs.items() else None
                name = products[0] if idx is None or idx >= len(products) else products[idx]
                rest += [w for w in re.findall(r"[a-z]{3,}", name.lower()) if w not in BOND_TERMS][-1:]
            proposed_alt = " ".join(rest[:span_n])
            if not proposed_alt or self.executed_searchterm(proposed_alt):
                return miss
            if reasons[0] == "name":
                note = (f"The word '{flagged[0]}' in this query names a required property; it is not part of a required value. "
                        "Since this store requires every word, the query only keeps listings whose fields or title repeat that "
                        "name, and it loses the ones whose field gives just the value.")
            else:
                note = (f"The word '{flagged[0]}' in this query comes only from a requirement the shopper DROPPED: it still "
                        "restricts the results to listings that carry it, although that is no longer needed.")
            return f"{note} Search '{proposed_alt}' as well and read all of its pages.", proposed_alt
        except Exception as exc:
            self.journal.append({"kind": "trap_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
            return miss

    def unified_searchterms(self) -> list[dict[str, Any]]:
        """At most 2 ready searches, each joining the word values of the in-force requirements of a product without a
        verified fit (a closed list by its first value; figures, references and exclusions skipped), with the product's
        kind word at the end when the store reads that far; only when 2+ values join and no search led by those words ran."""
        tmpl = self.seek_blueprint()
        if not tmpl:
            return []
        base, text_key = tmpl
        basket = self.needs.items()
        span_n = self.foremost_terms or 3
        vkey = str(self.needs.iteration)
        live = {k.rpartition("|")[0] for k in self.succeeded if k.rpartition("|")[2] == vkey} - self.rejected_lineitems
        covered = {self.rack_familiar(i) for i in live} - {""}
        shelf_sets = self.landing_racks() if basket else []

        def word_values(idx: int | None) -> list[list[str]]:
            found: list[list[str]] = []
            for need in self.needs.traits_remaining():
                label, _, raw_value = need.partition(":")
                if basket and self.ware_belonging(label) not in (idx, None):
                    continue
                clean = re.sub(r"\((?:stated|revealed[^)]*)\)", "", raw_value).strip().lower()
                if clean and not re.search(r"\d|as shown|same as|page|redirect|refer|\bnot\b|\bno\b|without|except|anything|other than", clean):
                    lead_alt = re.split(r"[,;/]|\bor\b|\band\b", clean)[0]
                    kept = [w for w in re.findall(r"[a-z]{2,}", lead_alt) if w not in BOND_TERMS]
                    if 0 < len(kept) < 3:
                        found.append(kept)
            return found

        def wanted(idx: int | None) -> bool:
            if basket:
                return idx < len(shelf_sets) and bool(shelf_sets[idx]) and not shelf_sets[idx] & covered
            return bool(self.unconfirmed())
        made: list[dict[str, Any]] = []
        for idx in (range(len(basket)) if basket else [None]):
            vals = word_values(idx) if wanted(idx) else []
            if len(vals) > 1:
                query: list[str] = []
                for kept in vals:
                    if len(kept) + len(query) <= span_n:
                        query.extend(kept)
                product = basket[idx]["product"] if basket else str(self.needs.perusal.get("product") or "")
                tail = [w for w in re.findall(r"[a-z]{3,}", product.lower()) if w not in BOND_TERMS][-1:]
                if tail and tail[0] not in query and len(query) < span_n:
                    query.extend(tail)
                phrase = " ".join(query)
                if len(query) > 1 and not self.executed_searchterm(phrase):
                    made.append({**base, text_key: phrase})
        return made[:2]

    def adjudicate(self, listing_id: str, verdict: str) -> str:
        """The decline audit called a listing a possible fit that the item check rejected: one further item check sees
        both views; if it passes, the listing counts as verified (its earlier failing rulings are wiped)."""
        try:
            audits = (r for r in reversed(self.journal)
                      if isinstance(r, dict) and r.get("kind") == "audit" and isinstance(r.get("result"), dict))
            reason = next((str((r.get("result") or {}).get("reason") or "") for r in audits), "")
            cited = f" ({reason[:400]})" if reason else ""
            note = ("\n\nTWO CHECKS OF THIS ITEM DISAGREE. Reading the same records, the decline audit judged that it meets "
                    f"every requirement{cited}; an earlier check said: {verdict[:500]}"
                    "\nSettle from the records which view holds: a field giving the required value in other words, or a shade, "
                    "qualifier or spelling of that same named value, meets the requirement; a field giving a different value "
                    "fails it.")
            second_view = self.assess_entry(listing_id, "ORDER", second=note)
            if not second_view:
                prefix = f"{listing_id}|{self.needs.iteration}"
                for memo in [m for m in self.verdicts_alt if m == prefix or m.startswith(prefix + "|")]:
                    self.verdicts_alt[memo] = ""
            return second_view or ""
        except Exception as exc:
            self.journal.append({"kind": "arbitrate_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
            return verdict

    def uncontested(self) -> list[str]:
        """For a decline: per requested product (a single one for a one-product request) one read listing on its own
        category, in stock and inside the budget as last seen, that no item check rejected under the requirements in
        force - a verified one first, otherwise the cheapest; [] if a product has none or the picks together exceed
        the budget."""
        try:
            basket = self.needs.items()
            shelf_sets = self.landing_racks() if basket else [set(self.rack_tags())]
            if not shelf_sets or not all(shelf_sets):
                return []
            vkey = str(self.needs.iteration)
            unsuccessful = {memo.split("|")[0] for memo, said in self.verdicts_alt.items() if said and memo.split("|")[1:2] == [vkey]}
            live = {memo.rpartition("|")[0] for memo in self.succeeded if memo.rpartition("|")[2] == vkey}
            # a listing the decline audit looked at without calling it a possible fit is treated as ruled out
            looked = self.__dict__.get("audit_ids")
            whole_audit = looked is None or len(self.vault.read) <= self.__dict__.get("audit_reads", -1)
            if self.__dict__.get("audit_done") and whole_audit:
                looked = set(self.vault.read)     # the latest audit covered every record read so far
            audited = set(looked or set()) - set(self.__dict__.get("audit_fits") or [])
            stock_out: dict[str, bool] = {}

            def note_stock(node: dict[str, Any]) -> None:
                mark = self.tablerow_flag(node)
                if mark in self.vault.ids and any(re.search(r"stock|avail", str(fld).lower()) for fld in node):
                    stock_out[mark] = self.traded_depleted(node)
            for record in self.vault.items:
                traverse_mappings(record, note_stock)
            limit = self.needs.outlay_lid
            pointers = self.reference_indicators()
            selected: list[str] = []
            spend = 0.0
            for idx, shelf_set in enumerate(shelf_sets):
                qty = basket[idx]["quantity"] if basket else self.needs.quantity()
                bucket = []
                for mark in sorted(self.vault.read):
                    excluded = (mark in unsuccessful or mark in audited or mark in self.rejected_lineitems or stock_out.get(mark)
                                or mark in selected or mark in pointers or self.rack_familiar(mark) not in shelf_set)
                    price = None if excluded else self.vault.price(mark)
                    if whether_numeric(price) and not (whether_numeric(limit) and price * qty > limit):
                        bucket.append((mark not in live, price, mark))
                if not bucket:
                    return []
                _, price, mark = min(bucket)
                selected.append(mark)
                spend += price * qty
            return [] if whether_numeric(limit) and spend > limit else selected
        except Exception as exc:
            self.journal.append({"kind": "unrefuted_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
            return []

    def suit_keep(self) -> str:
        """A single time, for a decline with an empty cart once the decline audit listed possible fits: the item check on
        the first two whose details were read; when one passes (for several products: when verified listings then cover
        each product), the decline is held and the actor is told to buy it."""
        try:
            blocked = (self.__dict__.get("fit_held") or self.handcart.measures or self.shifts_remaining() <= 4
                       or self.protocol.surplus_alt() < WRAPUP + 30)
            candidates = [] if blocked else [i for i in self.__dict__.get("audit_fits") or []
                                             if i in self.vault.ids and i not in self.rejected_lineitems]
            if not candidates:
                return ""
            self.fit_held = True
            passing = []
            for cand in candidates[:2]:
                if cand in self.vault.read:     # only judged from its full record, never from a search row
                    verdict = self.assess_entry(cand, "ORDER")
                    if verdict and self.protocol.surplus_alt() > WRAPUP + 30:
                        verdict = self.adjudicate(cand, verdict)
                    if not verdict and f"{cand}|{self.needs.iteration}" in self.succeeded:
                        passing.append(cand)
                        if not self.needs.items():
                            break
            several = bool(self.needs.items()) if passing else False
            if several:
                vkey = str(self.needs.iteration)
                live = {k.rpartition("|")[0] for k in self.succeeded if k.rpartition("|")[2] == vkey} - self.rejected_lineitems
                covered = {self.rack_familiar(i) for i in live} - {""}
                if any(s and not s & covered for s in self.landing_racks()):
                    passing = []
            if not passing:
                return ""
            one = len(passing) == 1
            target = "a verified line of every product" if several else passing[0]
            return (f"The decline audit named {', '.join(passing)} as a possible fit, and the independent item check, reading "
                    f"{'its' if one else 'their'} full record, found {'it meets' if one else 'they meet'} every requirement now "
                    f"in force. Consider {'it' if one else 'them'} before declining: if it fits, put {target} in the cart, "
                    "re-read its stock and price, and order; decline only if you find it fails a requirement in force.")
        except Exception as exc:
            self.journal.append({"kind": "fit_hold_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
            return ""


class RackTraverses:
    """Errand part: walks over shelves and result pages, plus the reads that come before a decline."""

    def uid_depot(self) -> bool:
        """True when search results come in a fixed id order: at least two searches showed 5+ rows sorted by id, and
        none showed a different order."""
        flags = [trail["id_order"] for trail in self.__dict__.get("page_book", {}).values() if "id_order" in trail]
        return False not in flags and flags.count(True) > 1

    def ware_position(self, listing_id: str) -> int | None:
        """Which product a listing is, judged by its category (None for a one-product request, or if unknown)."""
        if self.needs.items():
            shelf = self.rack_familiar(listing_id)
            return next((idx for idx, s in enumerate(self.landing_racks()) if shelf and shelf in s), None)
        return None

    def traverse_landing(self, slot: int | None) -> set[str]:
        shelf_sets = self.landing_racks()
        idx = 0 if slot is None else slot
        selected = set(shelf_sets[idx]) if shelf_sets and idx < len(shelf_sets) else set()
        if selected:
            return selected
        return set(self.rack_tags()) if slot is None else set()

    @staticmethod
    def numeral_magnitude(chunk: Any) -> tuple[str, str] | None:
        """(v605f FA) an amount opened by a whole number with no limit words ('4 pieces', '350 ml'): (the number, its unit
        word); None for anything else (a limit, a decimal, no unit word). Stores index '4-Piece' and '4 pieces' as the
        tokens '4' and 'piece', so a search with the number as a token of its own and a unit word lists exactly the
        records that write it."""
        lowered = str(chunk).lower()
        hit = None if re.search(CROSSCHECK_BOUNDARY, lowered) else re.match(r"\s*(\d+)\s+([a-z]+)", lowered)
        return None if hit is None else (hit.group(1), hit.group(2))

    @staticmethod
    def magnitude_portion(chunk: Any) -> bool:
        """EB4: part of a value that no keyword carries: a number with limit words ('at most 350 ml') or an amount opened
        by its number ('350 ml', '6 cores') - a search word only matches that single written token."""
        lowered = str(chunk).lower()
        has_digit = re.search(r"\d", lowered) is not None
        return has_digit and bool(re.search(CROSSCHECK_BOUNDARY, lowered) or re.match(r"\s*\d+(?:[.,]\d+)?\s+[a-z]", lowered))

    def traverse_conveys(self, call: dict[str, Any], slot: int | None, alts: bool = False) -> set[Any]:
        """The in-force requirements of product slot (None: every one; numbered in props_left order) with a value word
        among a search's leading words; with alts, the (requirement, closed-list alternative) pairs instead. An amount
        bound or a number-led amount never counts as carried (EB4)."""
        def close(a: str, b: str) -> bool:
            return min(len(a), len(b)) > 2 and (a.startswith(b) or b.startswith(a))

        def leading(pattern: str) -> list[str]:
            found = [t for text in self.loose_passage(call) for t in re.findall(pattern, text.lower())]
            return found[: self.foremost_terms] if self.foremost_terms else found
        per_product = slot is not None and bool(self.needs.items())
        words = leading(r"[a-z0-9]{2,}")
        toks = leading(r"[a-z0-9]+")
        carried: set[Any] = set()
        for num, need in enumerate(self.needs.traits_remaining()):
            if per_product and self.ware_belonging(need.partition(":")[0]) not in (slot, None):
                continue
            clean = re.sub(r"\((?:stated|revealed[^)]*)\)", "", need.partition(":")[2]).lower()
            for alt, part in enumerate(re.split(r"[,;/]|\bor\b|\beither\b|\bnor\b", clean)):
                figure = self.numeral_magnitude(part)
                if figure:
                    # v605f FA: '4 pieces' counts as carried when the number is its own token next to a word of its unit
                    hit = figure[0] in toks and any(t == figure[1] or close(t, figure[1]) for t in toks)
                else:
                    hit = not self.magnitude_portion(part) and any(
                        close(w, v) for w in words for v in re.findall(r"[a-z0-9]{3,}", part))
                if hit:
                    carried.add((num, alt) if alts else num)
        return carried

    def traverse_rating(self, call: dict[str, Any], slot: int | None) -> int:
        """Count of product slot's in-force requirements (None: all) with a value word among a search's leading words."""
        return len(self.traverse_conveys(call, slot))

    def traverse_choices(self, slot: int | None) -> list[tuple[int, str, dict[str, Any]]]:
        """(pages left, key, walk) of searches the cheapest walk may keep reading for product slot: id-ordered, leading
        words carrying a value of its requirements, a row of its category among those seen, 1 to 2 x WALK_PAGES pages
        unread, and not covered by a search read to its end that showed every match (one whose words are a subset of
        its own, or are all values of the product and words of its category label while carrying every value - each
        alternative of a closed list - that it carries: all-words results hold nothing more); of twins (same total and
        first page) the one with fewer pages left; in an all-words store none whose words contain another option's
        words that one walk can read (EB4: its rows are among that one's); fewest pages first. Words are compared as
        whole sets (the words the store reads, once that is learned)."""
        shelf_set = self.traverse_landing(slot)
        if not shelf_set:
            return []
        upto = self.foremost_terms or None

        def lead(trail: dict[str, Any]) -> set[str]:
            joined = " ".join(self.loose_passage(trail["args"])).lower()
            return {self.wordbase(t) for t in re.findall(r"[a-z0-9]+", joined)[:upto]}
        label = {self.wordbase(t) for s in shelf_set for t in re.findall(r"[a-z0-9]{2,}", s)}
        per_product = slot is not None and bool(self.needs.items())
        values: set[str] = set()
        for need in self.needs.traits_remaining():
            if not per_product or self.ware_belonging(need.partition(":")[0]) in (slot, None):
                values |= {self.wordbase(t) for t in re.findall(r"[a-z0-9]{2,}", need.partition(":")[2].lower())}
        known = values | label
        ledger = self.__dict__.get("page_book", {})

        def live_walk(trail: dict[str, Any]) -> bool:
            return trail.get("id_order") is True and bool(trail.get("first"))
        finished = [(lead(t), self.traverse_conveys(t["args"], slot, alts=True), self.valsweep_regime(t["args"]))
                    for t in ledger.values()
                    if live_walk(t) and self.subsequent_sheet(t) is None and lead(t) and len(t.get("row_shelf") or {}) >= t["total"]]
        best: dict[tuple, tuple[int, str, dict[str, Any]]] = {}
        for sig, trail in ledger.items():
            if not live_walk(trail) or self.subsequent_sheet(trail) is None:
                continue
            seen_shelves = (trail.get("row_shelf") or {}).values()
            if not any(s in shelf_set for s in seen_shelves) or not self.traverse_conveys(trail["args"], slot):
                continue
            mine = lead(trail)
            alt_pairs = self.traverse_conveys(trail["args"], slot, alts=True)
            style = self.valsweep_regime(trail["args"])
            # a same-mode search read to its end that showed every match already holds every fit this one could show
            if any(mode == style and (old <= mine or (old <= known and alt_pairs <= pairs)) for old, pairs, mode in finished):
                continue
            left = -(-trail["total"] // max(1, trail["size"])) - len(trail["read"])
            sibling_slot = (trail["total"], trail["first"])
            if 0 < left <= 2 * TRAVERSE_SHEETS and (sibling_slot not in best or left < best[sibling_slot][0]):
                best[sibling_slot] = (left, sig, trail)
        options = list(best.values())
        if self.whole_terms():
            # EB4: when two options' words nest, the longer one lists only rows the shorter lists. If its extra words are
            # values of the product or words of its category label nothing is lost and the shorter goes; otherwise the
            # shorter, when a single walk can read it, holds every fit of the longer one, and the longer goes
            def covered_by(p: tuple, o: tuple) -> bool:
                mine_p, mine_o = lead(p[2]), lead(o[2])
                if mine_p < mine_o:
                    return not (mine_o - mine_p) <= known and p[0] <= TRAVERSE_SHEETS
                return mine_o < mine_p and (mine_p - mine_o) <= known
            options = [o for o in options if not any(covered_by(p, o) for p in options if p is not o)]
        options.sort(key=lambda p: (p[0], -self.traverse_rating(p[2]["args"], slot), p[1]))
        return options

    def traverse_scheme(self, allowance: int, subset: set | None = None) -> list[tuple[int | None, str, int]]:
        """(product, walk key, pages) to read under a cheapest rule in an id-ordered store: for each product not walked
        yet, its search with the fewest pages left; products needing fewest first, each capped at WALK_PAGES, within
        `allowance`."""
        viable = (allowance > 0 and self.needs.priority() == "cheapest" and self.role("search") and self.sheet_param()
                  and self.uid_depot())
        if not viable:
            return []
        already = self.__dict__.get("walked", set())
        wants = []
        for slot in list(range(len(self.needs.items()))) or [None]:
            skip = slot in already or (subset is not None and slot not in subset)
            options = [] if skip else self.traverse_choices(slot)
            if options:
                wants.append((options[0][0], slot if slot is not None else -1, slot, options[0][1]))
        wants.sort(key=lambda w: w[:2])
        schedule: list[tuple[int | None, str, int]] = []
        remaining = allowance
        for pages, _, slot, sig in wants:
            take = min(pages, TRAVERSE_SHEETS, remaining)
            if take <= 0:
                break
            schedule.append((slot, sig, take))
            remaining -= take
        return schedule

    def traverse_choose(self, slot: int | None) -> tuple[str, float] | None:
        """(id, price) of product slot's pick: its line in the cart, otherwise its cheapest listing that passed the item check."""
        gone = self.defunct_lineitems()       # v605f FF: a line that sold out is not a pick

        def ours(i: str) -> bool:
            return slot is None or self.ware_position(i) == slot
        cands = [i for i in self.handcart.measures if ours(i) and i not in gone]
        if not cands:
            vkey = str(self.needs.iteration)
            succeeded = [memo.rpartition("|")[0] for memo in self.succeeded if memo.rpartition("|")[2] == vkey]
            cands = [i for i in succeeded if i not in self.rejected_lineitems and i not in gone and ours(i)]
        priced = [(i, float(p)) for i, p in ((i, self.vault.price(i)) for i in cands) if whether_numeric(p)]
        return min(priced, key=lambda ip: (ip[1], ip[0])) if priced else None

    def lineitem_searchterm(self, slot: int | None) -> dict[str, Any] | None:
        """A ready search for the product line of product slot's pick: its maker plus the first title word after the maker
        that is not a maker, kind, link or value word nor a figure (the line's name), when that search has not run."""
        current = self.traverse_choose(slot)
        tmpl = self.seek_blueprint()
        if not (current and tmpl):
            return None
        holder: list[dict[str, Any]] = [{}]

        def grab(node: dict[str, Any]) -> None:
            if self.tablerow_flag(node) == current[0] and self.heading_belonging(node):
                holder[0] = node
        for record in self.vault.items:
            traverse_mappings(record, grab)
        rec = holder[0]
        maker = next((val for fld, val in rec.items() if isinstance(val, str) and val.strip()
                      and re.search(r"brand|make|manufactur", str(fld).lower()) and "id" not in str(fld).lower()), "")
        maker_toks = re.findall(r"[a-z0-9]+", maker.lower())
        if not maker_toks:
            return self.heading_lineitem(slot, rec, tmpl)     # CG5: the pick's record names no maker, so its title's opening words name the line
        products = self.needs.items() or [{"product": str(self.needs.perusal.get("product") or "")}]
        avoid = set(maker_toks) | BOND_TERMS
        avoid.update(w for s in self.traverse_landing(slot) for w in re.findall(r"[a-z]+", s))
        avoid.update(w for need in self.needs.traits_remaining() for w in re.findall(r"[a-z]+", need.lower()))
        avoid.update(w for r in products for w in re.findall(r"[a-z]+", r["product"].lower()))
        in_title = re.findall(r"[a-z][a-z'-]{3,}", self.heading_belonging(rec).lower())
        lead_maker = maker_toks[0]
        after = in_title[in_title.index(lead_maker) + 1:] if lead_maker in in_title else in_title
        line_word = next((w for w in after if "'" not in w and w.strip("'-") not in avoid), None)
        phrase = None if line_word is None else f"{lead_maker} {line_word}"
        if phrase is None or self.executed_searchterm(phrase):
            return None
        return {**tmpl[0], tmpl[1]: phrase}

    def traverse_tablerows(self, slot: int | None, extra_sigs: list[str], cap_n: int) -> list[str]:
        """Unread rows on product slot's category shown by its value-led id-ordered searches (and by the searches
        `extra_sigs`): in stock as last seen, inside the budget for its units as last priced, not in a refused cart,
        priced under its pick (lacking one, under the cheapest of those rows already read); only rows from the searches
        carrying the most of its values (an all-words search shows only rows carrying its words); cheapest first."""
        shelf_set = self.traverse_landing(slot)
        ledger = self.__dict__.get("page_book", {})
        ordered = [t for t in ledger.values() if t.get("id_order") is True]
        top = max((len(self.traverse_conveys(t["args"], slot)) for t in ordered), default=0)
        pool_ids: set[str] = set()
        for sig, trail in ledger.items():
            carried = self.traverse_conveys(trail["args"], slot) if trail.get("id_order") is True else set()
            if sig in extra_sigs or (carried and len(carried) == top):
                pool_ids.update(i for i, s in (trail.get("row_shelf") or {}).items() if s in shelf_set)
        current = self.traverse_choose(slot)
        if not current:
            # no pick yet: the cheapest already-read row among these stands in
            read_rows = sorted((float(self.vault.price(i)), i) for i in pool_ids
                               if i in self.vault.read and whether_numeric(self.vault.price(i)))
            current = (read_rows[0][1], read_rows[0][0]) if read_rows else None
        basket = self.needs.items()
        qty = basket[slot]["quantity"] if slot is not None and slot < len(basket) else self.needs.quantity()
        limit = self.needs.outlay_lid
        stock_out: dict[str, bool] = {}

        def note_stock(node: dict[str, Any]) -> None:
            if any(isinstance(v, str) and v in pool_ids and "id" in str(fld).lower() for fld, v in node.items()):
                stock_out[self.tablerow_flag(node)] = self.traded_depleted(node)
        for record in self.vault.items:
            traverse_mappings(record, note_stock)
        cheaper = []
        for mark in pool_ids:
            price = self.vault.price(mark)
            skip = mark in self.vault.read or mark in self.rejected_lineitems or stock_out.get(mark) or not whether_numeric(price)
            if not skip and not (whether_numeric(limit) and price * qty > limit) and not (current and price >= current[1]):
                cheaper.append((price, mark))
        cheaper.sort()
        return [mark for _, mark in cheaper[:cap_n]]

    def sheet_condense(self, payload: Any) -> str:
        """A short line for every row of a search page: id, price, stock, title, category."""
        found: list[dict[str, Any]] = []

        def gather(node: dict[str, Any]) -> None:
            if len(node) > 3 and any(isinstance(v, str) and v in self.vault.ids and "id" in str(fld).lower() for fld, v in node.items()):
                found.append(node)
        traverse_mappings(payload, gather, ceiling=3)
        lines = []
        for row in found[:12]:
            price = next((v for fld, v in row.items() if whether_numeric(v) and re.search(r"price|cost", str(fld).lower())), None)
            shown = "?" if price is None else price
            flag = " SOLD OUT" if self.traded_depleted(row) else ""
            lines.append(f"{self.tablerow_flag(row)} {shown}{flag} '{self.heading_belonging(row)[:70]}' [{self.rack_belonging(row)}]")
        return "Rows on this page (id, price, title, category): " + ("; ".join(lines) or "none")

    def traverse_invoke(self, call: dict[str, Any], remark: str) -> bool:
        """A single scripted search of the walk; the actor sees a compact digest of its rows (plus any shopper news)."""
        call_spec = {"id": f"walk_{len(self.converse)}_{self.protocol.pulse}", "type": "function",
                     "function": {"name": self.role("search"), "arguments": json.dumps(call)}}
        self.converse.append({"role": "assistant", "content": remark, "tool_calls": [call_spec]})
        count_before = len(self.vault.items)
        self.covering = True
        try:
            replies = self.route_alt([call_spec], scripted=True)
        finally:
            self.covering = False
        if not replies:
            self.converse.append({"role": "tool", "tool_call_id": call_spec["id"], "content": "Not run."})
            return False
        reply_id, reply_text = replies[0]
        grew = len(self.vault.items) > count_before
        content = reply_text
        if grew:
            content = self.sheet_condense(self.vault.items[-1])
            _, marker, note = reply_text.partition("\n[note] ")
            if marker and note and re.search(r"shopper|Meaning:|REFUSED", note):
                content = f"{content}\n[note] {note}"
        self.converse.append({"role": "tool", "tool_call_id": reply_id, "content": content})
        return grew

    def conduct_traverse(self, schedule: list[tuple[int | None, str, int]], reserve: int, ceil_slots: list | None = None, repair: bool = False) -> bool:
        """Reads the planned pages (scripted, one search per step: every page's token comes from the one before), runs one
        line search per pick, then opens the cheapest unread rows under each pick; the walked products' lines get their
        rank check again before the next cart call or order."""
        paging = self.sheet_param()
        if not paging or not (schedule or ceil_slots) or not self.role("search"):
            return False
        page_key = paging[0]
        ledger = self.__dict__.get("page_book", {})
        walked = self.__dict__.setdefault("walked", set())
        notes: list[str] = []
        did_read = False

        def wire_ok() -> bool:
            return not self.protocol.done and self.protocol.surplus_alt() >= WRAPUP + 30
        for slot, sig, count in schedule:
            walked.add(slot)
            trail = ledger.get(sig)
            if trail:
                phrase = " ".join(self.loose_passage(trail["args"]))
                pages_read = 0
                while pages_read < count:
                    upcoming = self.subsequent_sheet(trail)
                    if upcoming is None or self.shifts_remaining() <= reserve + 1 or self.protocol.crashed or not wire_ok():
                        break
                    remark = ("The lowest price is wanted, and this store returns search results in a fixed id order rather "
                              f"than by price: I read the following page of '{phrase}'.")
                    if not self.traverse_invoke({**trail["args"], page_key: upcoming}, remark):
                        break
                    pages_read += 1
                    did_read = True
                seen_n = len(trail.get("row_shelf") or {})
                notes.append(f"'{phrase}': {pages_read} more page(s) read, {seen_n} of its {trail['total']} matches seen")
        line_sigs: dict[Any, str] = {}
        for slot, _, _ in schedule:
            if self.shifts_remaining() <= reserve + 2 or not wire_ok():
                break
            line_call = self.lineitem_searchterm(slot)
            ran_line = bool(line_call) and self.traverse_invoke(
                line_call, "I also search the pick's product line (its maker plus the word that names the line): other "
                "sizes and colours of a line are separate listings and may cost less.")
            if ran_line:
                did_read = True
                line_sigs[slot] = json.dumps({a: v for a, v in line_call.items() if a != page_key}, sort_keys=True, default=str)
                line_trail = self.__dict__.get("page_book", {}).get(line_sigs[slot])
                extra = 0
                # a short line (three pages at most) is read through to its end
                while line_trail and extra < 2 and line_trail["total"] <= 3 * max(1, line_trail["size"])\
                        and self.subsequent_sheet(line_trail) is not None and self.shifts_remaining() > reserve + 3\
                        and not self.protocol.done and self.protocol.surplus_alt() > WRAPUP + 30:
                    if not self.traverse_invoke({**line_trail["args"], page_key: self.subsequent_sheet(line_trail)},
                                          "I read the following page of the line search."):
                        break
                    extra += 1
                notes.append(f"the pick's line '{' '.join(self.loose_passage(line_call))}'")
        ceil_rows: dict[Any, list[str]] = {}
        for slot in dict.fromkeys(ceil_slots or []):
            # CG3: the category under each pick through the store's price filter, once one has been learned
            keep = reserve if repair else max(reserve, self.purchase_owed() + 2)
            if self.shifts_remaining() <= keep + 1 or not wire_ok():
                break
            got_rows = self.ceiling_walk(slot, reserve, repair)
            if got_rows:
                did_read = True
                notes.append(got_rows[0])
                ceil_rows[slot] = list(got_rows[1])
        if not did_read:
            return False
        bulk = self.bundle_instrument()
        cap_n = bulk[2] if bulk else (ROOF_EXAMINES if ceil_rows else 2)
        buckets = [self.traverse_tablerows(slot, [line_sigs.get(slot, "")], cap_n) for slot, _, _ in schedule] + list(ceil_rows.values())
        to_open: list[str] = []
        while len(to_open) < cap_n and any(buckets):
            # a single list read, filled in turn from each walked product's cheapest rows
            for bucket in buckets:
                while bucket and bucket[0] in to_open:
                    del bucket[0]
                if bucket and len(to_open) < cap_n:
                    to_open.append(bucket.pop(0))
        while to_open and (1 if bulk and len(to_open) > 1 else len(to_open)) >= self.shifts_remaining() - reserve:
            del to_open[-1]
        walked_slots = [slot for slot, _, _ in schedule] + list(ceil_rows)
        shelf_union = set().union(*(self.traverse_landing(slot) for slot in walked_slots))
        self.standing_assessed = {m for m in self.standing_assessed if self.rack_familiar(m.split("|")[0]) not in shelf_union}
        self.purchase_sanctioned = False
        if to_open:
            priced = ", ".join(f"{i} ({self.vault.price(i):g})" for i in to_open)
            middle = ("The cheapest unread in-stock rows inside the budget that these searches showed, under each pick's price, "
                      f"are opened next: {priced}. ")
        else:
            middle = "These searches showed no unread in-stock row inside the budget under the pick's price. "
        recap_alt = ("[cheapest walk] " + "; ".join(notes) + ". " + middle
                   + "For every product, choose the lowest-priced listing whose own fields meet each requirement now in force; "
                   "the rank check runs again on the cart call and on the order.")
        if to_open:
            self.open_queue = to_open
            self.open_recheck = True
            self.open_say = "The lowest price is wanted: I open the cheapest unread rows that the walk turned up."
            self.open_purpose, self.open_floor = "walk", reserve
            self.examine_titled()
        last = self.converse[-1] if self.converse else None
        if last is not None and last.get("role") == "tool":
            tail = "" if to_open else "\n" + self.progress_lineitem()
            last["content"] = f"{last['content']}\n{recap_alt}{tail}"
        self.journal.append({"kind": "walk", "plan": [[slot, count] for slot, _, count in schedule], "ceiling": list(ceil_rows),
                         "opened": to_open})
        return True

    def abstain_suits(self) -> set[str]:
        """Listings that count as fits for a decline: passed by the item check under the requirements in force, or named
        by the latest decline audit and not failed by an item check since; none from a refused cart."""
        vkey = str(self.needs.iteration)
        unsuccessful = {memo.split("|")[0] for memo, said in self.verdicts_alt.items() if said and memo.split("|")[1:2] == [vkey]}
        live = {memo.rpartition("|")[0] for memo in self.succeeded if memo.rpartition("|")[2] == vkey}
        named = set(self.__dict__.get("audit_fits") or []) - unsuccessful
        return (live | named) - self.rejected_lineitems

    def abstain_absent(self) -> list[int | None]:
        """Products (None: the single requested product) that have no fit (decline_fits) on their own category."""
        fitted = {self.rack_familiar(i) for i in self.abstain_suits()} - {""}
        if self.needs.items():
            return [idx for idx, s in enumerate(self.landing_racks()) if s and not s & fitted]
        return [None] if not set(self.rack_tags()) & fitted else []

    def abstain_traverses(self) -> list[str]:
        """For every product lacking a fit (2 searches at most overall): the search the cheapest walk could keep reading
        for it (walk_options: id order, leading words carrying its values, rows of its category, unread pages, not
        covered by a search read to its end) that carries the most of its values given in words (a requirement given
        by reference counts for nothing: its words name a listing to read, not the value), then the most values
        overall, then fewest pages left."""
        in_force = self.needs.traits_remaining()

        def stated(call: dict[str, Any], slot: int | None) -> int:
            return sum(1 for num in self.traverse_conveys(call, slot) if num < len(in_force)
                       and not re.search(r"as shown|same as|the page|sold by|redirect|refer", in_force[num].lower()))
        selected: list[str] = []
        for slot in self.abstain_absent():
            options = [o for o in self.traverse_choices(slot) if stated(o[2]["args"], slot)]
            options.sort(key=lambda p: (-stated(p[2]["args"], slot), -self.traverse_rating(p[2]["args"], slot), p[0], p[1]))
            fresh = next((o[1] for o in options if o[1] not in selected), None)
            if fresh is not None:
                selected.append(fresh)
        return selected[:2]

    def tablerow_reservoirs(self, targets: list[tuple[set[str], int]], only_ids: set[str] | None, under: list[float] | None = None) -> list[list[str]]:
        """For each (categories, units): unread rows seen (restricted to `only_ids` when given) on those categories, in
        stock as last seen, inside the budget for the units as last priced, not in a refused cart (under: priced below
        that product's figure); cheapest first."""
        limit = self.needs.outlay_lid
        where_of: dict[str, str] = {}
        stock_out: dict[str, bool] = {}

        def note_row(node: dict[str, Any]) -> None:
            mark = self.tablerow_flag(node)
            if mark in self.vault.ids and self.rack_belonging(node):
                where_of[mark] = self.rack_belonging(node)
                if any(re.search(r"stock|avail", str(fld).lower()) for fld in node):
                    stock_out[mark] = self.traded_depleted(node)
        for record in self.vault.items:
            traverse_mappings(record, note_row)

        def pool_for(num: int, shelf_set: set[str], qty: int) -> list[str]:
            bucket = []
            for mark, where in where_of.items():
                price = self.vault.price(mark)
                blocked = (where not in shelf_set or mark in self.vault.read or mark in self.rejected_lineitems
                           or stock_out.get(mark) or (only_ids is not None and mark not in only_ids))
                too_dear = not whether_numeric(price) or (whether_numeric(limit) and price * qty > limit)\
                    or (under is not None and not price < under[num])
                if not (blocked or too_dear):
                    bucket.append((float(price), mark))
            return [mark for _, mark in sorted(bucket)]
        return [pool_for(num, shelf_set, qty) for num, (shelf_set, qty) in enumerate(targets)]

    def exceeding_allotment_tablerows(self) -> list[list[str]]:
        """Several products under no cheapest rule, each with a fit (decline_fits), whose cheapest fits together exceed the
        budget: per product, the unread rows under its cheapest fit (row_pools); [] otherwise."""
        try:
            basket = self.needs.items()
            limit = self.needs.outlay_lid
            if not basket or not whether_numeric(limit) or self.needs.priority() == "cheapest":
                return []
            shelf_sets = self.landing_racks()
            fit_ids = self.abstain_suits()
            floors: list[float] = []
            shares: list[float] = []
            for idx, shelf_set in enumerate(shelf_sets):
                costs = [float(self.vault.price(i)) for i in fit_ids if self.rack_familiar(i) in shelf_set and whether_numeric(self.vault.price(i))]
                if not (shelf_set and costs):
                    return []
                lowest = min(costs)
                floors.append(lowest)
                shares.append(lowest * basket[idx]["quantity"])
            if sum(shares) <= limit:
                return []
            pools = self.tablerow_reservoirs([(s, basket[idx]["quantity"]) for idx, s in enumerate(shelf_sets)], None, floors)
            # the product whose fit takes the largest share of the order goes first: a cheaper row there helps most
            ranked = [pools[idx] for idx in sorted(range(len(pools)), key=lambda idx: -shares[idx])]
            return ranked if any(ranked) else []
        except Exception as exc:
            self.journal.append({"kind": "over_budget_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
            return []

    def decline_reads(self) -> str:
        """Once per errand, at a decline: queues the code's own reads of unread pages of searches carrying required values
        (only in a store with a fixed id order, where later pages hold fits as often as the first; up to 3 pages, keeping
        6 steps for the order) and, for several products over the budget, of the cheapest rows under each pick; returns
        the hold note ('' when there is nothing to read)."""
        try:
            bulk = self.bundle_instrument()
            idle = (self.__dict__.get("decline_read_done") or self.protocol.surplus_alt() < WRAPUP + 60
                    or not (self.role("details") or bulk) or self.handcart.measures)
            if idle:
                return ""
            opens = 2 - bool(bulk)
            page_room = min(3, self.shifts_remaining() - opens - 6)
            can_walk = page_room > 0 and self.sheet_param() and self.role("search") and self.uid_depot()
            sigs = self.abstain_traverses() if can_walk else []
            cheaper = self.exceeding_allotment_tablerows() if self.shifts_remaining() > opens + 6 else []
            if not (sigs or cheaper):
                return ""
            self.decline_read_done = True
            self.decline_queue = (sigs, page_room, 6)
            parts = []
            if sigs:
                ledger = self.__dict__.get("page_book", {})
                quoted = "; ".join(f"'{' '.join(self.loose_passage(ledger[s]['args']))}'" for s in sigs)
                parts.append(f"searches carrying required values matched more listings than were read ({quoted}), and this "
                             "store returns results in a fixed id order rather than by fit or price, so fitting listings are as "
                             f"likely on unread pages as on read ones: the code now reads up to {page_room} of those pages for "
                             "you and opens the cheapest in-stock rows inside the budget that they show")
            if cheaper:
                parts.append("the listings found for all the products together cost more than the budget: the cheapest unread "
                             "in-stock rows priced under each product's pick are opened for you as well")
            buy = "one line per product, at its quantity" if self.needs.items() else "the item"
            return (f"Before a decline, {'; '.join(parts)}. Judge those records against the requirements now in force by meaning; "
                    f"buy what fits ({buy}), and decline only if every one of them fails.")
        except Exception as exc:
            self.journal.append({"kind": "decline_reads_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
            return ""

    def conduct_abstain_peruses(self) -> bool:
        """Carries out the reads decline_reads queued: the pages (one search per step), then one list read (else up to 2
        details reads) of the cheapest rows, taken in turn from the rows under each pick and the rows the pages showed;
        the decline audit then runs over the new records."""
        sigs, page_cap, reserve = self.__dict__.pop("decline_queue")
        paging = self.sheet_param()
        ledger = self.__dict__.get("page_book", {})
        bulk = self.bundle_instrument()
        opens = 2 - bool(bulk)
        pages_read = 0
        walked: list[str] = []
        for sig in (sigs if paging else []):
            trail = ledger.get(sig)
            if not trail:
                continue
            walked.append(sig)
            phrase = " ".join(self.loose_passage(trail["args"]))
            remark = (f"Before I decline, I read the following page of '{phrase}': results here come in a fixed id order, so "
                      "fitting listings may be on pages not read yet.")
            while pages_read < page_cap and self.shifts_remaining() > reserve + opens and not (self.protocol.done or self.protocol.crashed)\
                    and self.protocol.surplus_alt() > WRAPUP + 30:
                upcoming = self.subsequent_sheet(trail)
                if upcoming is None or not self.traverse_invoke({**trail["args"], paging[0]: upcoming}, remark):
                    break
                pages_read += 1
        seen: set[str] = set()
        for sig in walked:
            seen.update((ledger.get(sig) or {}).get("row_shelf") or {})
        basket = self.needs.items()
        targets = []
        for slot in self.abstain_absent():
            qty = basket[slot]["quantity"] if slot is not None and slot < len(basket) else self.needs.quantity()
            targets.append((self.traverse_landing(slot), qty))
        buckets = self.tablerow_reservoirs(targets, seen) if walked else []
        buckets += self.exceeding_allotment_tablerows()
        cap_n = bulk[2] if bulk else 2
        to_open: list[str] = []
        while any(buckets) and len(to_open) < cap_n:
            for bucket in buckets:
                while bucket and bucket[0] in to_open:
                    del bucket[0]
                if bucket and len(to_open) < cap_n:
                    to_open.append(bucket.pop(0))
        while to_open and (1 if bulk and len(to_open) > 1 else len(to_open)) >= self.shifts_remaining() - reserve:
            del to_open[-1]
        self.journal.append({"kind": "decline_reads", "pages": pages_read, "opened": to_open})
        last = self.converse[-1] if self.converse else None
        if to_open and not self.protocol.done:
            self.open_queue = to_open
            self.open_say = "Before I decline, I open the cheapest unread rows inside the budget that those pages and picks turned up."
            self.open_purpose, self.open_floor = "decline", reserve
            self.examine_titled()
        elif last is not None and last.get("role") == "tool":
            last["content"] = (str(last["content"]) + "\nThose pages showed no unread in-stock row inside the budget.\n"
                               + self.progress_lineitem())
        return bool(to_open) or pages_read > 0

    # ---------------------------------------------------------------------------------- requirements by reference (BT)


class IndicatorRequirements:
    """Errand part: requirements that borrow a value from another listing - targets, hops, checks, repairs."""

    def reference_enroll(self) -> None:
        """BT1: opens one reference for each stated requirement whose value points at another listing; the reading's
        reference_targets supply its seller, kind and details, failing that the request's own words."""
        perusal = self.needs.perusal
        targets = perusal.get("reference_targets")
        infos = [t for t in targets if isinstance(t, dict)] if isinstance(targets, list) else []
        named = perusal.get("references")
        listed = {str(r) for r in named or []} if isinstance(named, list) else set()
        for written in list(self.needs.written):
            text = str(written)
            value = text.partition(":")[2]
            if reference_passage(value) or (text in listed and re.search(DIVERGE_TERMS, value.lower())):
                self.reference_attach(text, infos, self.request, "request")

    def reference_unmonitored(self, wanted_refs: list[str], records: list[dict[str, Any]]) -> list[str]:
        """References of the reading that no reference record follows (BW3: or that a record follows without having found
        the listing it points at - an empty chain and still unresolved - so the records the request points at still
        reach the check)."""
        try:
            def followed(r: str) -> bool:
                return any(self.reference_correspond(t, r) and (t.get("status") == "resolved" or t.get("chain")) for t in records)
            return [r for r in wanted_refs if not followed(r)]
        except Exception:
            return wanted_refs

    def reference_adopt(self) -> None:
        """BT1: a need the shopper revealed by pointing at another listing opens a reference as well (its seller being a
        name the shopper's line writes) and joins the reading's references; a resolved value that a rebuilt requirement
        list lost back to the reference's words is restored."""
        try:
            for record in self.refs:
                stale = record["status"] == "resolved" and any(
                    self.reference_correspond(record, r) and r != record["now"] for r in self.needs.traits_remaining())
                if stale:
                    self.reference_substitute(record, record["now"])
            news = list(self.needs.__dict__.get("ref_news") or [])
            self.needs.ref_news = []
            for item in news:
                opened = self.reference_attach(str(item.get("need") or ""), list(item.get("infos") or []), str(item.get("line") or ""), "reply")
                if opened:
                    known = self.needs.perusal.get("references")
                    if not isinstance(known, list):
                        known = self.needs.perusal["references"] = []
                    if opened["req"] not in known:
                        known.append(opened["req"])
        except Exception as exc:
            self.journal.append({"kind": "ref_take_error", "error": f"{type(exc).__name__}: {exc}"[:300]})

    def reference_attach(self, need: str, infos: list[dict[str, Any]], said: str, how: str) -> dict[str, Any] | None:
        """Opens the reference record for one requirement given by reference (None when its listing lacks a usable seller
        or a reference for that property and product is already open)."""
        label, _, value = str(need).partition(":")
        prop = re.split(r"\s*\(", label)[0].strip()
        if not prop or rubbish_slot(prop):
            return None
        bracketed = re.findall(r"\(([^)]*)\)", label)
        tag = bracketed[-1].strip() if bracketed else ""
        tag_kind = category_terms(tag)
        if any(old["req"] == need or (identical_trait(old["prop"], prop) and category_terms(old["tag"]) == tag_kind) for old in self.refs):
            return None

        def matches(cand: dict[str, Any]) -> Any:
            if not identical_trait(str(cand.get("property") or ""), prop):
                return False
            return not tag or not cand.get("product") or not self.needs.items() or category_terms(cand.get("product")) & category_terms(tag)
        info = next((cand for cand in infos if matches(cand)), None)
        target = self.reference_goal(info, value, said, prop)
        if not target:
            self.journal.append({"kind": "ref_skip", "req": str(need)[:200]})
            return None
        relation, tolerance, unit_word = reference_relationship(value)
        informed = ""
        if isinstance(info, dict):
            informed = str(info.get("relation") or "").lower()
            if relation == "same" and informed == "within" and whether_numeric(info.get("tolerance")):
                relation, tolerance, unit_word = "within", float(info["tolerance"]), str(info.get("unit") or "")
        # CE2: the source's own words only count between where the property is mentioned and the listing's seller name -
        # a differ word after the listing ('...; apart from that I'm easy') concerns something else
        window = reference_stretch(prop, said).lower()
        seller = str(target.get("seller") or "")
        cut = window.find(seller.lower()) if seller.strip() else -1
        if relation == "same":
            differs = re.search(r"differ|not the same|exclu|apart|other than|unlike", informed) or (
                cut > 0 and re.search(DIVERGE_TERMS, window[:cut]))
            if differs:
                relation, tolerance, unit_word = "differ", None, ""   # CE2: a differ from the reader or listener, or in the source's own words
        record = {"req": str(need), "head": label.strip(), "prop": prop, "tag": tag, "relation": relation, "tol": tolerance,
                  "unit": unit_word, "target": target, "first": dict(target), "chain": [], "seen": {self.reference_slot(target)},
                  "judged": set(), "queries": [], "pages": 0, "hops": 0, "tries": 0, "fails": 0, "status": "open",
                  "value": "", "goes_with": [], "now": "", "origin": how, "unindexed": False}
        self.refs.append(record)
        self.journal.append({"kind": "ref_open", "req": str(need)[:200], "target": target, "relation": relation, "origin": how})
        return record

    def reference_goal(self, info: Any, value: str, source: str, prop: str) -> dict[str, Any] | None:
        """Builds {seller, kind, details} for the listing a reference names. The reading's (or listener's) entry is used
        when the source text writes its seller; otherwise the entry is parsed from the source's words that follow the
        property, and failing that from the requirement's own value. Details absent from the source are dropped. Returns
        None when no seller can be found."""
        def writes(candidate: Any) -> bool:
            return self.reference_vendor_approved(candidate, source)

        given = info if isinstance(info, dict) else {}
        who = str(given.get("seller") or "").strip()
        what = str(given.get("kind") or "").strip()
        listed = given.get("details")
        facts = [str(x).strip() for x in (listed if isinstance(listed, list) else [])
                 if isinstance(x, (str, int, float)) and not isinstance(x, bool) and str(x).strip()]
        attempts = (reference_decode(chunk, writes) for chunk in (reference_stretch(prop, source), value))
        found = next((a for a in attempts if a and writes(a["seller"])), None)
        if writes(who):
            same_seller = bool(found) and reference_moniker(found["seller"]) == reference_moniker(who)
            if same_seller and not facts:
                facts = found["details"]
        elif found:
            who, what, facts = found["seller"], what or found["kind"], facts or found["details"]
        else:
            return None
        source_toks = set(reference_pieces(source))
        kept = [f for f in dict.fromkeys(facts) if source_toks & set(reference_pieces(f))]
        label = (what or found["kind"]) if found else what
        return {"seller": who, "kind": label[:80], "details": kept[:6]}

    def reference_vendor_approved(self, seller: Any, source: Any) -> bool:
        """True for a seller name that the source writes as whole words (an 's possessive allowed) and that is more than a
        product or category word."""
        cleaned = str(seller or "").strip()
        bounded = r"(?<![a-z0-9])" + re.escape(cleaned.lower()) + r"(?:'s)?(?![a-z0-9])"
        if len(re.findall(r"[A-Za-z]", cleaned)) < 3 or not re.search(bounded, str(source or "").lower()):
            return False
        generic: set[str] = set()
        products = self.needs.items() or [{"product": str(self.needs.perusal.get("product") or "")}]
        for entry in products:
            generic.update(category_terms(entry["product"]))
        search_shape = self.structures.get(self.role("search") or "") or {}
        enums = [spec["enum"] for spec in (search_shape.get("properties") or {}).values()
                 if isinstance(spec, dict) and isinstance(spec.get("enum"), list)]
        for choices in enums:
            generic.update(w for choice in choices for w in category_terms(choice))
        own_words = category_terms(cleaned)
        return len(own_words) > 0 and not own_words.issubset(generic)

    @staticmethod
    def reference_slot(tgt: dict[str, Any]) -> tuple:
        tokens: set[str] = set()
        for piece in tgt.get("details") or []:
            tokens.update(reference_pieces(piece))
        return reference_moniker(tgt.get("seller") or ""), tuple(sorted(tokens))

    @staticmethod
    def reference_description_belonging(tgt: dict[str, Any]) -> str:
        extras = tgt.get("details")
        tail = " (" + "; ".join(tgt.get("details") or []) + ")" if extras else ""
        noun = tgt.get("kind") or "listing"
        return f"the {noun} that {tgt.get('seller')} sells{tail}"

    def reference_description(self, ref: dict[str, Any]) -> str:
        target = ref["target"]
        return self.reference_description_belonging(target)

    def reference_position(self) -> dict[str, dict[str, Any]]:
        """Maps each listing id to all fields observed for it (search rows and records merged, later sightings
        overwriting earlier ones), ordered by first sighting; ref_seen_at records the latest observation index of each."""
        memoized = self.__dict__.get("ref_memo")
        if memoized and memoized[0] == len(self.vault.items):
            return memoized[1]
        merged: dict[str, dict[str, Any]] = {}
        latest: dict[str, int] = {}
        pos = 0

        def gather(piece: dict[str, Any]) -> None:
            mark = self.tablerow_flag(piece)
            # (CE3: any record holding a property map qualifies regardless of size, since a spec sheet is just id, title and map)
            if mark in self.vault.ids and (len(piece) >= 4 or any(isinstance(sub, dict) and len(sub) >= 2 for sub in piece.values())):
                merged.setdefault(mark, {}).update(piece)
                latest[mark] = pos

        for pos, record in enumerate(self.vault.items):
            traverse_mappings(record, gather)
        self.ref_memo = (len(self.vault.items), merged)
        self.ref_seen_at = latest
        return merged

    def reference_vendors(self, row: dict[str, Any]) -> set[str]:
        """Reduced (ref_name) seller names found on a record or row: fields whose names look like a seller's (SELLER_KEYS:
        shop, brand ...) plus (CE1) the short text fields that were learned to hold a seller (seller_keys)."""
        taught = self.vendor_slots()

        def names_seller(field: Any, entry: Any) -> bool:
            low = str(field).lower()
            short_text = isinstance(entry, str) and bool(entry.strip()) and len(entry) <= 60
            sellerish = bool(re.search(VENDOR_SLOTS, low)) or str(field) in taught
            textual = bool(re.search(r"note|text|desc|comment|detail|info|message|section", low))
            return short_text and sellerish and "id" not in low and not textual

        found = {reference_moniker(entry) for field, entry in row.items() if names_seller(field, entry)}
        found.discard("")
        return found

    def vendor_slots(self) -> set[str]:
        """CE1: learns which record fields hold a listing's seller. A candidate is a short text field (not an id, title,
        name, label, category or text field) whose value is either the seller of some reference seen on a read record
        or row, or a capitalised name that some record's notes mention mid-sentence (one note pointing at another
        listing through its seller) and that never appears as a value in any of the store's property maps."""
        try:
            targets = set()
            for entry in self.__dict__.get("refs") or []:
                for side in (entry.get("target") or {}, entry.get("first") or {}):
                    if isinstance(side, dict):
                        targets.add(reference_moniker(side.get("seller")))
            targets.discard("")
            stamp = (len(self.vault.items), tuple(sorted(targets)))
            memoized = self.__dict__.get("seller_memo")
            if memoized and memoized[0] == stamp:
                return memoized[1]
            index = self.reference_position()
            mentioned: set[str] = set()
            in_maps: set[str] = set()

            def spans(caps: list[str]) -> list[str]:
                return [" ".join(caps[lo:hi]) for lo in range(len(caps)) for hi in range(lo + 1, min(len(caps), lo + 3) + 1)]

            for record in index.values():
                for entry in record.values():
                    if isinstance(entry, dict):
                        in_maps.update(reference_moniker(x) for x in entry.values() if isinstance(x, str))
                for clause in re.split(r"(?<=[.!?;])\s+", self.reference_passages(record, desc=False)):
                    caps: list[str] = []
                    for token in re.findall(r"[A-Za-z][A-Za-z'-]*", clause)[1:]:
                        if token[:1].isupper() and re.search(r"[a-z]", token):
                            caps.append(token)
                        else:
                            mentioned.update(reference_moniker(s) for s in spans(caps))
                            caps = []
                    mentioned.update(reference_moniker(s) for s in spans(caps))
            mentioned = {m for m in mentioned if len(m) >= 4} - in_maps
            barred = (r"title|heading|name|label|caption|categor|shelf|department|aisle|note|text|desc|comment|detail|info|"
                      r"message|section|url|image|summary|overview")

            def holds_seller(field: Any, entry: Any) -> bool:
                low = str(field).lower()
                plain_field = "id" not in low and not re.search(barred, low)
                if isinstance(entry, str) and 0 < len(entry) <= 60 and plain_field:
                    capitalised = entry.strip()[:1].isupper() and len(entry.split()) <= 3
                    return reference_moniker(entry) in targets or (capitalised and reference_moniker(entry) in mentioned)
                return False

            learned = {str(f) for record in index.values() for f, e in record.items() if holds_seller(f, e)}
            self.seller_memo = (stamp, learned)
            return learned
        except Exception:
            return set()

    @staticmethod
    def reference_passages(row: dict[str, Any], desc: bool = True) -> str:
        """The free-text fields of a record: its seller notes, or (with desc and no notes present) its description."""
        skip = r"title|heading|headline|name|label|caption|categor|shelf|department|url|image"
        texts = [(str(f).lower(), e) for f, e in row.items()
                 if isinstance(e, str) and len(e.split()) >= 3 and not str(f).lower().endswith("id") and not re.search(skip, str(f).lower())]
        blurbs = [e for low, e in texts if re.search(r"desc|overview|summary|about", low)]
        own_notes = [e for low, e in texts if not re.search(r"desc|overview|summary|about", low)]
        selected = own_notes if own_notes else (blurbs if desc else [])
        return "\n".join(selected)

    def reference_portions(self, row: dict[str, Any]) -> tuple[set[str], set[str]]:
        """Returns (tokens of the record's field names, tokens of its title, field values and notes). Inside the notes,
        words under a negation ('and not black', 'not the green one') are removed first."""
        simple = self.reference_passages(row, desc=False)
        simple = re.sub(r"\b(?:not|no|never|without|nor)\b(?:\W+[A-Za-z0-9.:/&'-]+){1,3}", " ", simple, flags=re.I)
        values = [self.heading_belonging(row), simple]
        names = []
        def scalar(x: Any) -> bool:
            return isinstance(x, (str, int, float)) and not isinstance(x, bool)

        groups = [entry for entry in row.values() if isinstance(entry, dict)] or [
            {f: e for f, e in row.items()
             if scalar(e) and not re.search(r"id$|title|shop|seller|brand|price|amount|desc|overview|note", str(f).lower())}]
        for group in groups:
            names.extend(re.sub(r"([a-z])([A-Z])", r"\1 \2", str(f)).replace("_", " ") for f in group)
            values.extend(str(e) for e in group.values() if scalar(e))
        return set(reference_pieces(" ".join(names))), set(reference_pieces(" | ".join(values)))

    def reference_displays(self, row: dict[str, Any], detail: str, parts: tuple[set[str], set[str]] | None = None) -> bool:
        """Whether a record can show a detail. At least one of the detail's value words (field names excluded) must be in
        the record's title, values or notes. For each figure tied to a unit word, the record's own title and fields (its
        notes are ignored, as they talk about other listings too) must either write that figure in that unit or carry no
        figure in that unit (the amount may be stated in another unit)."""
        names, values = self.reference_portions(row) if parts is None else parts
        words = [tok for tok in reference_pieces(detail) if len(tok) >= 3 and not re.search(r"\d", tok)]
        wanted = set(words) - names
        if not wanted and words:
            # (every word is also some field's name - 'organic' beside an 'Organic' field: a field named by one of the words
            # that holds a plain negative ('no', 'false', 'none') says the record lacks it)
            for entry in [e for e in row.values() if isinstance(e, dict)] or [row]:
                for field, val in entry.items():
                    fname = set(reference_pieces(re.sub(r"([a-z])([A-Z])", r"\1 \2", str(field)).replace("_", " ")))
                    if fname and fname <= set(words) and str(val).strip().lower() in ("no", "false", "none", "n/a", "without", "0"):
                        return False
        if wanted and wanted.isdisjoint(values):
            return False
        asked = reference_numerals(detail)
        if asked:
            bits = [self.heading_belonging(row)] + [str(x) for entry in row.values() if isinstance(entry, dict)
                                           for x in entry.values() if isinstance(x, (str, int, float))]
            written = reference_numerals(" | ".join(bits))

            def contradicted(amount: float, unit: str) -> bool:
                same_unit = [f for f, u in written if u == unit]
                return bool(same_unit) and all(not abs(f - amount) < 1e-9 for f in same_unit)

            if any(contradicted(amount, unit) for amount, unit in asked):
                return False
        return True

    @staticmethod
    def reference_attribute(row: dict[str, Any], prop: str) -> str:
        """The record's value in the field named for the property, or '' when it has none."""
        pairs = [(field, str(x).strip()) for entry in row.values() if isinstance(entry, dict)
                 for field, x in entry.items() if isinstance(x, (str, int, float)) and not isinstance(x, bool)]
        for test in (rigorous_trait, identical_trait):
            hit = next((text for field, text in pairs if text and test(field, prop)), None)
            if hit is not None:
                return hit
        return ""

    def reference_uids_attach(self, ids: list[str]) -> None:
        earlier = self.__dict__.get("ref_ids") or set()
        self.ref_ids = set(earlier) | set(ids)

    def reference_external(self) -> set[str]:
        """BT8: listings opened only as reference candidates whose category belongs to none of the products."""
        try:
            opened = self.__dict__.get("ref_ids") or set()
            if not opened:
                return set()
            per_product = self.landing_racks()
            if not per_product or not all(per_product):
                return set()      # (BW4: while some product's own category is unknown, nothing counts as foreign)
            homes = set().union(*per_product)
            index = self.reference_position()
            carted = set(self.handcart.measures)

            def elsewhere(ident: str) -> bool:
                shelf = self.rack_belonging(index.get(ident) or {})
                return bool(shelf) and shelf not in homes

            return {ident for ident in opened if ident not in carted and elsewhere(ident)}
        except Exception:
            return set()

    def reference_candidates(self, ref: dict[str, Any]) -> list[str]:
        """The seen listings (rows or records) of the current target's seller, best first: those on a category whose name
        matches the target's kind, then those whose title or overview shows more of its details, then those the most
        recent search returned."""
        target = ref["target"]
        wanted_seller = reference_moniker(target["seller"])
        kind_set = category_terms(target["kind"])
        index = self.reference_position()
        latest = self.__dict__.get("ref_seen_at") or {}
        def score(ident: str, record: dict[str, Any]) -> tuple:
            blurbs = [str(v) for k, v in record.items() if isinstance(v, str) and re.search(r"desc|overview|summary", str(k).lower())]
            visible = set(reference_pieces(self.heading_belonging(record) + " " + " ".join(blurbs)))
            token_sets = [set(reference_pieces(piece)) for piece in target["details"]]
            hits = len([t for t in token_sets if t and t <= visible])
            bonus = 2 if kind_set & category_terms(self.rack_belonging(record)) else 0
            return -bonus - hits, -latest.get(ident, 0), ident, hits

        ranked = sorted(score(i, rec) for i, rec in index.items() if wanted_seller in self.reference_vendors(rec))
        ref["shows"] = {entry[2]: entry[3] for entry in ranked}
        return [entry[2] for entry in ranked]

    def reference_regimes(self, needed: set[str], labels: set[str]) -> tuple[list[tuple[dict[str, Any], str]], set[str]]:
        """BW1: returns (keyword-search templates seen so far, best first; the modes found to ignore their words). Searches
        that carry one free-text argument are grouped by their remaining arguments (minus an optional one holding a
        category label). A group where two searches sharing no word returned the same leading rows (2 or more) ignores
        its words and is dropped; the rest are ordered by how many distinct texts they were used with, earliest first on
        a tie."""
        by_mode: dict[str, dict[str, Any]] = {}
        walks = self.__dict__.get("page_book", {}).values()
        for order, walk in enumerate(walks):
            call_args = walk.get("args") if isinstance(walk.get("args"), dict) else {}
            free = self.loose_passage(call_args)
            text_keys = [k for k, v in call_args.items() if isinstance(v, str) and v in free and v.strip()]
            if len(free) == 1 and len(text_keys) == 1:
                word_key = text_keys[0]
                rest = {k: v for k, v in call_args.items()
                        if k != word_key and (k in needed or not (isinstance(v, str) and v.strip().lower() in labels))}
                slot = by_mode.setdefault(json.dumps(rest, sort_keys=True, default=str), {"tpl": (rest, word_key), "seen": [], "at": order})
                stems = {self.wordbase(w) for w in re.findall(r"[a-z0-9]+", free[0].lower())}
                slot["seen"].append((stems, tuple(walk.get("first") or ())))

        def ignores_words(seen: list) -> bool:
            return any(len(rows_a) >= 2 and rows_a == rows_b and not (words_a & words_b)
                       for i, (words_a, rows_a) in enumerate(seen) for words_b, rows_b in seen[i + 1:])

        deaf = {mode for mode, slot in by_mode.items() if ignores_words(slot["seen"])}
        viable = sorted(((-len({tuple(sorted(w)) for w, _ in slot["seen"]}), slot["at"], slot["tpl"])
                         for mode, slot in by_mode.items() if mode not in deaf), key=lambda item: (item[0], item[1]))
        return [item[2] for item in viable], deaf


    def reference_blueprint(self) -> tuple[dict[str, Any], str] | None:
        """Returns (other arguments, text argument) for the code's own reference searches: the keyword-search mode the
        errand used most (ref_modes; never one shown to ignore its words), minus an optional argument holding a category
        label seen on rows; otherwise, when the search tool has a single free-text argument and requires nothing else,
        just that argument."""
        shape = self.structures.get(self.role("search") or "") or {}
        required = shape.get("required")
        needed = set(required or []) if isinstance(required, list) else set()
        labels = {self.rack_belonging(r) for r in self.reference_position().values()}
        labels.discard("")
        modes, deaf = self.reference_regimes(needed, labels)
        selected = modes[0] if modes else None
        page_spec = self.sheet_param()

        def trim(call_args: dict[str, Any]) -> dict[str, Any]:
            return {k: v for k, v in call_args.items() if k in needed or not (isinstance(v, str) and v.strip().lower() in labels)}

        def from_log() -> tuple[dict[str, Any], str] | None:
            # (an empty search yields no walk, yet its arguments still reveal how a keyword search is made)
            env_entries = (e for e in reversed(self.journal) if e.get("kind") == "env")
            for entry in env_entries:
                for perform in reversed(entry.get("actions") or []):
                    call_args = perform.get("args") if perform.get("name") == self.role("search") else None
                    if isinstance(call_args, dict):
                        text_keys = [k for k, v in call_args.items() if isinstance(v, str) and v.strip() and v in self.loose_passage(call_args)]
                        if len(text_keys) == 1:
                            word_key = text_keys[0]
                            rest = {k: v for k, v in call_args.items() if k != word_key and not (page_spec and k == page_spec[0])}
                            if json.dumps(trim(rest), sort_keys=True, default=str) not in deaf:
                                return rest, word_key
            return None

        if not selected:
            selected = from_log()
        if selected:
            rest, word_key = selected
            return trim(rest), word_key
        fields = shape.get("properties") if isinstance(shape.get("properties"), dict) else {}
        token = self.lexeme_slot()
        free = [k for k, spec in fields.items()
                if isinstance(spec, dict) and spec.get("type") == "string" and not spec.get("enum") and k != token]
        lone = len(free) == 1 and not needed - {free[0]}
        return ({}, free[0]) if lone else None

    def reference_terms(self, ref: dict[str, Any]) -> list[str]:
        """Words from the details that a search for the target may carry, most likely indexed first: figures with their
        unit glued on ('4gb'), then bare figures of two or more digits, then words of three or more letters not found in
        any field name read so far (nor in a requirement's property), longest first."""
        def letters_of(field: Any) -> list[str]:
            return re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", str(field)).lower())

        field_words: set[str] = {w for record in self.reference_position().values() for entry in record.values()
                                 if isinstance(entry, dict) for field in entry for w in letters_of(field)}
        for req in self.needs.written + self.needs.uncovered:
            label = re.split(r"\s*\(", str(req).partition(":")[0])[0]
            field_words.update(re.findall(r"[a-z]+", label.lower()))
        seller_bits = set(re.findall(r"[a-z0-9]+", str(ref["target"]["seller"]).lower()))
        glued, numbers, simple = [], [], []
        for piece in ref["target"]["details"]:
            before = ""
            for chunk in str(piece).lower().split():
                after_figure = re.fullmatch(r"\d+(?:[.,]\d+)?", before) is not None
                before = chunk
                # (CE3: the word that follows a figure is its unit, not a search word on its own: '2.43 inches')
                bare_chunk = chunk.strip(".,;:!?()[]{}\"'")
                for tok in re.findall(r"[a-z0-9]+", chunk):
                    digits = re.search(r"\d", tok)
                    skip = tok in BOND_TERMS or tok in seller_bits or (after_figure and not digits)
                    if skip:
                        pass
                    elif digits and re.search(r"[a-z]", tok):
                        glued.append(tok)
                    elif digits:
                        if len(tok) >= 2:
                            numbers.append(tok)
                    elif len(tok) >= 3 and tok == bare_chunk and tok not in field_words and tok.rstrip("s") not in field_words:
                        simple.append(tok)
        simple = sorted(simple, key=lambda w: -len(w))
        # (CE3: for a detail written as "<field> <value>", the final word is the value and is searched ahead of the rest,
        # provided its leading words name a field found on the store's records, or the shopper wrote each detail of the
        # pointed-at listing in that field-value form)
        depot_attributes = self.depot_slots()
        tokenised = [[w for w in re.findall(r"[a-z0-9]+", str(piece).lower()) if w not in BOND_TERMS]
                     for piece in ref["target"]["details"]]
        all_pairs = ref.get("hops", 0) == 0 and bool(tokenised) and all(len(t) >= 2 for t in tokenised)

        def leads_with_field(toks: list[str]) -> bool:
            return any(" ".join(toks[:j]) in depot_attributes for j in range(1, len(toks)))

        values_first = [toks[-1] for toks in tokenised
                        if len(toks) >= 2 and toks[-1] in simple and (all_pairs or leads_with_field(toks))]
        simple = [w for w in values_first if w in simple] + [w for w in simple if w not in values_first]
        return list(dict.fromkeys(glued + numbers + simple))

    def depot_slots(self) -> set[str]:
        """CE: property field names found on the records read so far, lowercased, separators as spaces, camelCase split."""
        count = len(self.vault.items)
        memoized = self.__dict__.get("store_keys_memo")
        if memoized and memoized[0] == count:
            return memoized[1]
        found: set[str] = set()

        def collect(piece: dict[str, Any]) -> None:
            maps = [e for e in piece.values()
                    if isinstance(e, dict) and len([x for x in e.values() if isinstance(x, (str, int, float))]) >= 2]
            for entry in maps:
                spaced = (re.sub(r"([a-z])([A-Z])", r"\1 \2", str(field)).lower() for field in entry)
                found.update(j for j in (" ".join(re.findall(r"[a-z0-9]+", s)) for s in spaced) if j)

        for record in self.vault.items:
            traverse_mappings(record, collect)
        self.store_keys_memo = (count, found)
        return found

    def reference_schemes(self, ref: dict[str, Any]) -> list[str]:
        """The reference searches left to run for the current target, in order: seller plus one detail word, then the
        seller by itself; where the store does not index sellers, the detail word with the kind word instead."""
        limit = max(2, self.foremost_terms or 3)
        bare = re.sub(r"'s\b", "", str(ref["target"]["seller"]))
        seller = " ".join(re.findall(r"[A-Za-z0-9]+", bare))
        terms = self.reference_terms(ref)
        kind_terms = [w for w in re.findall(r"[a-z]{3,}", str(ref["target"]["kind"]).lower()) if w not in BOND_TERMS]
        last_kind = kind_terms[-1:]
        candidates: list[str] = []
        if not ref.get("unindexed"):
            if terms and len(seller.split()) < limit:
                candidates.append(f"{seller} {terms[0]}")
            candidates.append(seller)
        elif terms or last_kind:
            candidates.append(" ".join((terms[:1] + last_kind)[:limit]))
        done = ref["queries"]
        return [c for c in candidates if c.strip() and c.lower() not in done]

    def reference_sheet(self, ref: dict[str, Any]) -> dict[str, Any] | None:
        """The next unread page of a search that returned the seller's rows (no more than 3 extra pages per target)."""
        page_spec = self.sheet_param()
        wanted = reference_moniker(ref["target"]["seller"]) if page_spec and ref["pages"] < 3 else ""
        result = None
        if wanted:
            history = list(self.__dict__.get("page_book", {}).values())
            for walk in history[::-1]:
                following = self.subsequent_sheet(walk) if wanted in reference_moniker(" ".join(self.loose_passage(walk["args"]))) else None
                if following is not None:
                    result = {**walk["args"], page_spec[0]: following}
                    break
        return result

    def reference_headroom(self) -> bool:
        """BT5: whether steps, time and model budget allow one more scripted reference read or search. CE: a smaller share
        applies while every open reference is a differ (a one-value exclusion that stays soft if unresolved), and only
        REF_DIFFER_BLIND steps while none of them has yet found a listing of its chain."""
        live = [r for r in self.__dict__.get("refs") or [] if r.get("status") == "open"]
        employed = self.__dict__.get("ref_steps", 0)
        only_differ = bool(live) and all(r.get("relation") == "differ" for r in live)
        if only_differ and all(not r.get("chain") for r in live) and employed >= REFERENCE_DIVERGE_UNLOOKED:
            return False      # (CE, review: a soft exclusion whose listing no search turned up does not justify the first add's steps)
        budget = max(8, int(REFERENCE_STAGES * self.stage_lid))
        reserve = self.reference_lowbound()
        if only_differ:
            budget = max(6, int(REFERENCE_DIVERGE_STAGES * self.stage_lid))
        protocol = self.protocol
        within = employed < budget and protocol.surplus_alt() > WRAPUP + 60 and not protocol.expended_depleted() and not protocol.done and not protocol.crashed
        # (ED2: with several products, the chase holds while some other product still lacks a candidate; ED1: the final
        # steps of a pending named hop may still go ahead)
        return within and not self.reference_lingers(reserve) and (self.shifts_remaining() > reserve or self.reference_jump_headroom())

    def reference_lowbound(self) -> int:
        """Steps that the scripted reference reads and searches must leave untouched (v605d M1: the floor of the base
        reserve rule, kept at least 7 steps above the order's reserve, since CF's bigger reserve would otherwise cut the
        last notes read of a differ chain, and a reference chain needs 5 steps of window)."""
        base = min(INITIAL_PURCHASE_OWED, self.stage_lid // 4 + 1)
        base = max(INITIAL_PURCHASE_VIA, base) + 10
        return max(base, self.purchase_owed() + 7)

    def reference_ware(self, ref: dict[str, Any]) -> int | None:
        """The product a reference concerns (by its tag); None when unknown or when there is one product."""
        tag = ref.get("tag")
        several = self.needs.items()
        return self.ware_belonging(f"x ({tag})") if several and tag else None

    def reference_lingers(self, floor: int | None = None) -> bool:
        """ED2 (MC4b): with several products, when every open reference is a differ (a soft one-value exclusion) already
        past its source listing, the scripted chase pauses while some product other than the references' own still lacks
        a candidate (ed_pool: a listing that passed the item check, or a read one whose own record shows every plain
        value), letting that product's value pages and walks run first; the pause ends while the chase still has 4 steps
        above its floor. Never with one product, a same / within reference (which binds its product's search as a chain),
        or a differ whose source is not yet in its chain."""
        products = list(self.needs.items() or [])
        live = [r for r in self.__dict__.get("refs") or [] if r.get("status") == "open"]
        # (same / within references steer their own product's search, and a differ needs its source listing found
        # first - CE5's pointer guard requires it to be in the chain: none of these pause)
        eligible = bool(products) and bool(live) and all(r.get("relation") == "differ" and r.get("chain") for r in live)
        if not eligible or self.shifts_remaining() <= (self.reference_lowbound() if floor is None else floor) + 4:
            return False
        try:
            owners = {self.reference_ware(r) for r in live}
            empty = [k for k in range(len(products)) if not any(e["fit"] for e in self.elicited_reservoir(k))]
        except Exception as exc:
            problem = f"{type(exc).__name__}: {exc}"
            self.journal.append({"kind": "ref_wait_error", "error": problem[:300]})
            return False
        others = [k for k in empty if k not in owners - {None}]
        halt_alt = len(others) > (1 if None in owners else 0)
        if halt_alt and self.__dict__.get("ref_wait_said") != tuple(others):
            self.ref_wait_said = tuple(others)
            self.journal.append({"kind": "ref_wait", "lacking": others})
        return halt_alt

    def reference_jump_headroom(self) -> bool:
        """ED1 (v509c: seller + details): an open reference whose latest hop named the next listing's seller, kind and
        details, and for which no search has run yet, may still take up to 3 scripted steps (search, read, notes) while
        more than order_due() + 2 steps remain - in a one-product errand, or once every other product has a verified
        pick. The 3 steps are counted from the first one granted, across later hops too."""
        if self.shifts_remaining() <= 2 + self.purchase_owed():
            return False
        employed = self.__dict__.get("ref_steps", 0)

        def named_hop(entry: dict[str, Any]) -> bool:
            target = entry.get("target") or {}
            spelled = all(str(target.get(f) or "").strip() for f in ("seller", "kind")) and bool(target.get("details"))
            return entry.get("status") == "open" and entry.get("hops", 0) >= 1 and spelled

        for entry in filter(named_hop, self.__dict__.get("refs") or []):
            began = entry.get("late_from")
            if began is None and not entry.get("queries") and self.elicited_peers_confirmed(entry):
                entry["late_from"] = employed
                self.journal.append({"kind": "ref_late", "prop": entry.get("prop"), "to": str((entry.get("target") or {}).get("seller"))[:80]})
                return True
            if began is not None and employed - began < 3:
                return True
        return False

    def elicited_peers_confirmed(self, ref: dict[str, Any]) -> bool:
        """ED1: true with one product, or when every product except the reference's own has a verified pick (all but one
        when its own product is not known)."""
        products = self.needs.items()
        if products:
            try:
                unpicked = [k for k in range(len(products)) if not any(e["verified"] and e["fit"] for e in self.elicited_reservoir(k))]
            except Exception:
                return False
            mine = self.reference_ware(ref)
            return len(unpicked) <= 1 if mine is None else all(k == mine for k in unpicked)
        return True

    def reference_invokes_approved(self) -> bool:
        # (BW6: hop calls end at four fifths of the errand's spending cap, leaving the actor its budget to finish)
        protocol = self.protocol
        under_count = self.__dict__.get("ref_calls", 0) < REFERENCE_INVOKES
        ceiling = 0.8 * OUTLAY_LID * protocol.__dict__.get("scale", 1.0)
        return under_count and protocol.surplus_alt() > WRAPUP + 40 and not protocol.expended_depleted() and protocol.spend < ceiling

    def reference_correspond(self, ref: dict[str, Any], req: Any) -> bool:
        """Whether `req` is this reference's own requirement: identical text, or a reference-shaped value of the same
        property (and product) - or, after resolution, the resolved text."""
        whole = str(req)
        if whole in (ref["req"], ref.get("now")):
            return True
        head, _, value = whole.partition(":")
        # (CE2: a differ reference's requirement may carry only its differ words)
        shaped = reference_passage(value) or (ref.get("relation") == "differ" and re.search(DIVERGE_TERMS, value.lower()))
        if shaped and identical_trait(re.split(r"\s*\(", head)[0], ref["prop"]):
            labels = re.findall(r"\(([^)]*)\)", head)
            return not labels or not ref["tag"] or bool(category_terms(labels[-1]) & category_terms(ref["tag"]))
        return False

    def reference_active(self, ref: dict[str, Any]) -> bool:
        """The reference still applies unless a requirement of its property was dropped while none remains in force."""
        def label(req: Any) -> str:
            return re.split(r"\s*\(", str(req).partition(":")[0])[0]

        in_force = any(self.reference_correspond(ref, r) or identical_trait(label(r), ref["prop"]) for r in self.needs.traits_remaining())
        return in_force or all(not identical_trait(label(r), ref["prop"]) for r in self.needs.let_go)

    def reference_stage(self) -> bool:
        """BT: a single turn of the scripted reference resolver - hops judged on records already read (costing no step),
        then at most one scripted read or search. True when it did something."""
        if self.protocol.done or self.protocol.crashed or not self.__dict__.get("refs"):
            return False
        try:
            outcome = self.reference_cycle()
        except Exception as exc:
            problem = f"{type(exc).__name__}: {exc}"
            self.journal.append({"kind": "ref_error", "error": problem[:300]})
            for entry in [e for e in self.refs if e["status"] == "open"]:
                entry["status"], entry["why"] = "stuck", "the scripted resolver broke down"
            return False
        return outcome

    def reference_cycle(self) -> bool:
        def still_open(entry: dict[str, Any]) -> bool:
            if entry["status"] == "open" and not self.reference_active(entry):
                entry["status"] = "dropped"
                self.journal.append({"kind": "ref_dropped", "req": entry["req"][:200]})
            return entry["status"] == "open"

        live = [e for e in self.refs if still_open(e)]
        if not live:
            return False
        tick_at_start = self.protocol.pulse
        verdicts = [self.reference_evaluate(e) for e in sorted(live, key=lambda r: r["hops"])]
        did = any(verdicts)
        if self.protocol.pulse > tick_at_start:
            return True         # (CE3: judging sent out a notes read, which is this turn's one scripted step)
        live = [r for r in live if r["status"] == "open"]
        if not (live and self.reference_headroom()):
            return did
        lister = self.bundle_instrument()
        cap = lister[2] if lister else 2
        def pool_of(entry: dict[str, Any]) -> list[str]:
            unread = [i for i in self.reference_candidates(entry) if i not in self.vault.read and i not in entry["judged"]]
            hinted = entry.get("shows") or {}
            # (rows get opened once a detail-led search has run (or none is possible), when few remain, or when a title shows a detail)
            opened = entry["queries"] or not self.reference_schemes(entry) or len(unread) <= cap or any(hinted.get(i) for i in unread[:1])
            return unread if opened else []

        pools = [pool_of(e) for e in live]
        template = self.reference_blueprint()
        starving = [r for r, pool in zip(live, pools) if not pool and self.reference_schemes(r)] if template else []
        batch_ids: list[str] = []
        while len(batch_ids) < cap and any(pools):
            for source in pools:
                lead = 0
                while lead < len(source) and source[lead] in batch_ids:
                    lead += 1
                del source[:lead]
                if source and len(batch_ids) < cap:
                    batch_ids.append(source.pop(0))

        # (a reference with nothing to read runs its search first, so a single list read can serve both)
        hungry = starving if batch_ids and len(batch_ids) < cap else []
        if batch_ids and not hungry:
            return self.ref_read(batch_ids, live) or did
        if not template:
            return did            # (no keyword search to imitate yet: the actor's first search will provide one)
        extra, word_key = template
        for entry in hungry or sorted(live, key=lambda r: (r["hops"], len(r["queries"]))):
            plans = self.reference_schemes(entry)
            if not plans:
                more = self.reference_sheet(entry)
                if more:
                    entry["pages"] += 1
                    return self.ref_search(entry, more) or did
            else:
                query = plans[0]
                entry["queries"].append(query.lower())
                # (a search that already ran counts as done: its rows become the candidates read next)
                return True if self.executed_searchterm(query) else (self.ref_search(entry, {**extra, word_key: query}) or did)

        def exhausted(entry: dict[str, Any]) -> bool:
            remaining = [i for i in self.reference_candidates(entry) if i not in entry["judged"]]
            return not remaining and not self.reference_schemes(entry) and not self.reference_sheet(entry)

        for entry in filter(exhausted, live):
            self.reference_cease(entry, "the store's search never turned up the listing it names")
            did = True
        return did

    def ref_read(self, ids: list[str], refs: list[dict[str, Any]]) -> bool:
        """A single scripted read of the candidate rows (one list read if possible, else detail reads)."""
        lister = self.bundle_instrument() if len(ids) > 1 else None
        start = self.protocol.pulse
        props = ", ".join(dict.fromkeys(r["head"] for r in refs))
        note = f"The shopper's {props} is defined by a different listing, so I open the listings that could be that one."
        replies = self.examine_peruses(ids, note, lister)
        discarded = False
        if lister and replies and not self.__dict__.get("batch_proven"):
            proven = self.bundle_approved(self.__dict__.get("last_reads") or [], ids)
            if proven:
                self.batch_proven = True
            else:
                self.batch_spec, discarded = None, True
                self.journal.append({"kind": "batch_dropped", "tool": lister[0]})
        if replies:
            self.later_responses(replies)
        spent = self.protocol.pulse - start
        self.ref_steps = self.__dict__.get("ref_steps", 0) + spent
        self.reference_uids_attach(ids)
        self.journal.append({"kind": "ref_read", "ids": ids})
        # (a listing that the read did not return is put aside for good)
        missing = [] if discarded else [i for i in ids if i not in self.vault.read]
        for entry in refs if not discarded else []:
            entry["judged"].update(missing)
        return self.protocol.pulse > start

    def ref_search(self, ref: dict[str, Any], args: dict[str, Any]) -> bool:
        """A single scripted search for the reference's current target."""
        start = self.protocol.pulse
        note = f"The shopper's {ref['head']} is defined by a different listing ({self.reference_description(ref)}), so I search for it."
        result = self.traverse_invoke(args, note)
        self.ref_steps = self.__dict__.get("ref_steps", 0) + (self.protocol.pulse - start)
        found: list[dict[str, Any]] = []

        def keep(piece: dict[str, Any]) -> None:
            if self.tablerow_flag(piece) in self.vault.ids and len(piece) >= 4:
                found.append(piece)

        if result:
            traverse_mappings(self.vault.items[-1], keep, ceiling=3)
        wanted = reference_moniker(ref["target"]["seller"])
        query = " ".join(self.loose_passage(args))
        if found and all(wanted not in self.reference_vendors(r) for r in found) and reference_moniker(query) == wanted:
            ref["unindexed"] = True      # (a seller-only search returned none of that seller's rows: sellers are not searchable)
        self.journal.append({"kind": "ref_search", "prop": ref["prop"], "query": query[:120], "rows": len(found)})
        return self.protocol.pulse > start

    def reference_evaluate(self, ref: dict[str, Any]) -> bool:
        """BT3/BT4: judges the read records that may be the current target without spending a step - a plain record by the
        code, otherwise one hop call over the best four. True when it judged."""
        target = ref["target"]
        fresh = [i for i in self.reference_candidates(ref) if i in self.vault.read and i not in ref["judged"]]
        if not fresh:
            return False
        index = self.reference_position()
        by_seller = [i for i in self.reference_candidates(ref) if i in self.vault.read]
        split = {i: self.reference_portions(index.get(i) or {}) for i in set(fresh) | set(by_seller)}
        vague = ref.setdefault("loose", set())
        wanted = target["details"]

        def has(ident: str, piece: str) -> bool:
            return self.reference_displays(index.get(ident) or {}, piece, split[ident])

        if len(by_seller) >= 4:
            # (a detail phrased in words that none of 4+ records from that seller shows cannot be tested by the code -
            # likely a word carried over from the request's phrasing; the hop call still reads it)
            vague.update({p for p in wanted if not any(has(i, p) for i in by_seller)})

        def fits(ident: str) -> bool:
            return all(has(ident, p) for p in wanted if p not in vague)

        def coverage(ident: str) -> int:
            names, values = split[ident]
            return sum(bool(t) and t <= names | values and has(ident, p) for p, t in ((p, set(reference_pieces(p))) for p in wanted))

        def more_to_read() -> bool:
            pending = [i for i in self.reference_candidates(ref) if i not in self.vault.read and i not in ref["judged"]]
            return bool(pending or self.reference_schemes(ref) or self.reference_sheet(ref))

        full = {i: coverage(i) for i in fresh}
        order = {i: n for n, i in enumerate(fresh)}
        ranked = sorted(fresh, key=lambda i: (not fits(i), -full[i], order[i]))
        # (none of these shows the details: the next rows get read first - ED2: once the pause is over)
        if not fits(ranked[0]) and (self.reference_headroom() or self.reference_lingers()) and more_to_read():
            return False
        pick = [i for i in ranked if fits(i)][:4] or ranked[:4]     # (records that could be the one; else the nearest)
        exact = [i for i in pick if wanted and full[i] == len(wanted)]
        single = exact[0] if len(exact) == 1 else None
        probe = [single] if single is not None else ([i for i in pick if fits(i)][:2] or pick[:1])
        unnoted = self.reference_annotations_lost(ref, probe)
        # (CE3: their notes come first, through the tool that shows them, as a note may redirect; ED2: they are read once
        # another product has a candidate, which is never a reason to stop)
        if unnoted and self.reference_headroom() and self.reference_annotations_peruse(ref, unnoted):
            return True
        if unnoted and self.reference_lingers():
            return False
        judged = ref["judged"]
        stop_reason = ""
        if unnoted and not self.reference_headroom() and ref.get("relation") == "differ":
            # (CE3: a listing whose notes the store keeps apart and which were never read might move the property
            # elsewhere, so its own value cannot settle an exclusion - the reference stays unresolved and soft;
            # any other relation still takes the record's own value, since leaving it unresolved would bind nothing)
            stop_reason = "the notes of the listing it names could not be read within the remaining steps"
        elif single is not None and self.reference_simple(ref, single):
            judged.update(pick)
            self.reference_uids_attach(pick)
            self.reference_determine(ref, single, self.reference_attribute(index.get(single) or {}, ref["prop"]), [])
        elif not self.reference_invokes_approved():
            stop_reason = "the hop-check allowance is used up"
        else:
            self.reference_enact(ref, pick, self.reference_inquire(ref, pick, ""))
            if ref["target"] is target and ref["status"] == "open":
                # (records missing a detail given in words are not the one)
                ref["judged"].update(i for i in fresh if not fits(i))
        if stop_reason:
            judged.update(pick)
            self.reference_cease(ref, stop_reason)
        return True

    def reference_simple(self, ref: dict[str, Any], ident: str) -> bool:
        """A record that can be settled without a hop call: it has a field named for the property, and its notes neither
        name the property (such a note might redirect, even to the same seller), nor mention that field's value, nor name
        a different seller (one seen on another row, or a capitalised word after a sentence's first that none of the
        record's other fields holds)."""
        record = self.reference_position().get(ident) or {}
        own_value = self.reference_attribute(record, ref["prop"])
        notes = self.reference_passages(record, desc=False)
        hinted = (not own_value or self.reference_monikers_trait(ref["prop"], notes)
                  or bool(set(reference_pieces(own_value)) & set(reference_pieces(notes))))
        if hinted:
            return False
        mine = self.reference_vendors(record)
        others = {s for other in self.reference_position().values() for s in self.reference_vendors(other)} - mine
        names, _ = self.reference_portions(record)
        bits = [self.heading_belonging(record), self.rack_belonging(record)] + [
            str(x) for entry in record.values() if isinstance(entry, dict) for x in entry.values() if isinstance(x, (str, int, float))]
        known = names | set(reference_pieces(" | ".join(bits)))

        def foreign(token: str) -> bool:
            low = reference_moniker(token)
            return low in others or (token[:1].isupper() and len(low) >= 4 and low not in mine and low not in known)

        clauses = re.split(r"(?<=[.!?;])\s+", notes)
        return not any(foreign(t) for c in clauses for t in re.findall(r"[A-Za-z][A-Za-z'-]*", c)[1:])


    def reference_annotations_instrument(self) -> str | None:
        """CE3: the lookup tool that shows sellers' notes when the details tool's records lack them. The learned notes tool
        if any; None while the details tool's answers have carried notes; otherwise a lookup tool besides the details
        tool whose answers held a text (or empty) field that neither the details answers nor the search rows hold (a notes
        field, empty on listings read so far), most used first; otherwise the view tool, then the stock tool, if it is a
        single-listing lookup tool other than the details tool that has not yet answered for a listing (one that did
        answer showed nothing of the sort)."""
        learned = self.annotations_instrument()
        if learned:
            return learned
        details_tool = self.role("details")
        if (self.__dict__.get("note_keys") or {}).get(details_tool or ""):
            return None
        per_tool = self.__dict__.get("tool_keys") or {}
        text_fields = self.__dict__.get("text_keys") or {}
        details_fields = set(per_tool.get(details_tool or "") or set())
        row_fields = set(self.__dict__.get("row_keys") or set())
        usage = self.__dict__.get("tool_caps") or {}
        others = [t for t in self.retrieve_instruments() if t != details_tool]

        def adds_text(tool: str) -> bool:
            novel = set(per_tool.get(tool) or set()) - details_fields - row_fields
            return bool(novel & set(text_fields.get(tool) or set()))

        richer = [t for t in others if adds_text(t)]
        if richer:
            return max(richer, key=lambda t: ((usage.get(t) or {}).get("calls", 0), t))
        fallbacks = (self.role("view"), self.inventory_instrument())
        unused = [t for t in fallbacks if t in others and not (usage.get(t) or {}).get("id", 0)]
        return unused[0] if unused else None

    def reference_annotations_lost(self, ref: dict[str, Any], ids: list[str]) -> list[str]:
        """CE3: up to two of `ids` whose notes the store keeps apart (ref_notes_tool) and whose record from that tool has
        neither been seen nor requested yet for this reference."""
        via = self.reference_annotations_instrument()
        if via and ids:
            already = set((self.__dict__.get("read_with") or {}).get(via) or set())
            asked = ref.setdefault("noted", set())
            return [i for i in dict.fromkeys(ids) if i not in already and i not in asked][:2]
        return []

    def reference_annotations_peruse(self, ref: dict[str, Any], ids: list[str]) -> bool:
        """CE3: a single scripted read of those listings' notes via ref_notes_tool (each requested once). True when a step
        was spent."""
        via = self.reference_annotations_instrument()
        if not ids or not via:
            return False
        asked = ref.setdefault("noted", set())
        asked.update(ids)
        start = self.protocol.pulse
        note = (f"The shopper's {ref['head']} is defined by a different listing, so I read the notes of the listing that "
                f"could be that one ({via}), as a note may point somewhere else.")
        replies = self.examine_peruses(ids, note, None, via)
        if replies:
            self.later_responses(replies)
        self.ref_steps = self.__dict__.get("ref_steps", 0) + (self.protocol.pulse - start)
        self.reference_uids_attach(ids)
        self.journal.append({"kind": "ref_notes", "ids": ids, "tool": via})
        return self.protocol.pulse > start

    def reference_indicators(self) -> set[str]:
        """CE5: every listing an open differ reference's chain passed through (source included). Each carries the excluded
        value by construction (its own value, or the end of its notes' redirect) - counted for the reference's own
        product only."""
        def headcounts(entry: dict[str, Any]) -> bool:
            kept = entry.get("relation") == "differ" and entry.get("status") in ("open", "stuck", "resolved")
            return kept and self.reference_active(entry)

        found: set[str] = set()
        for entry in filter(headcounts, self.__dict__.get("refs") or []):
            found.update(i for i in entry.get("chain") or [] if self.reference_native_ware(entry, i))
        return found

    def reference_native_ware(self, ref: dict[str, Any], ident: str) -> bool:
        """CE5 (review): a chain listing weighs against the reference's own product only. With several products, one shelved
        in another product's category is that product's candidate rather than a pointer (an unknown shelf still counts)."""
        if self.needs.items():
            known_shelf = self.rack_familiar(ident)
            k = None if not known_shelf else self.ware_belonging(f"x ({known_shelf})")
            if k is not None and ref.get("tag"):
                return self.ware_belonging(f"x ({ref['tag']})") == k
        return True

    def reference_indicator_annotation(self, thing: str) -> str:
        """CE5: explains why a listing on a differ reference's chain cannot satisfy it ('' for any other listing)."""
        def applies(entry: dict[str, Any]) -> bool:
            on_chain = entry.get("relation") == "differ" and thing in (entry.get("chain") or [])
            return on_chain and self.reference_active(entry) and self.reference_native_ware(entry, thing)

        entry = next(filter(applies, self.__dict__.get("refs") or []), None)
        if entry is not None:
            shown = f" ({entry['value']})" if entry.get("status") == "resolved" and entry.get("value") else ""
            return (f"{thing} is the listing that the shopper's {entry['head']} reference names, or one its notes pass "
                    f"through, so its {entry['prop']} is the very value{shown} that the shopper's {entry['prop']} has to "
                    "differ from: it cannot satisfy that requirement")
        return ""

    def reference_omitted(self, fields: dict[str, str], k: int | None) -> bool:
        """CE5: whether a listing's fields carry, in the property's field, the excluded value (or a paired one) of a resolved
        differ reference that applies to product k (any product when the reference has no product)."""
        multi = self.needs.items()

        def bears(entry: dict[str, Any]) -> bool:
            if entry.get("relation") == "differ" and entry.get("status") == "resolved" and self.reference_active(entry):
                return not (multi and entry.get("tag") and self.ware_belonging(f"x ({entry['tag']})") not in (k, None))
            return False

        def shows_banned(entry: dict[str, Any]) -> bool:
            forbidden = {" ".join(reference_pieces(v)) for v in [entry.get("value")] + list(entry.get("goes_with") or []) if str(v or "").strip()}
            return any(identical_trait(key, entry["prop"]) and " ".join(reference_pieces(val)) in forbidden for key, val in (fields or {}).items())

        return any(shows_banned(e) for e in self.__dict__.get("refs") or [] if bears(e))


    def own_redirects(self, thing: str) -> list[tuple[str, str, str]]:
        """CE6: one (requirement, property, sentence) per requirement in force (for the listing's own product) whose
        property the listing's own notes hand off to another listing: a notes sentence naming the property followed by a
        seller name the listing does not carry (another record's seller, or a capitalised word after the sentence's
        first that none of its fields holds), with no negation in front ('it is not the X')."""
        try:
            index = self.reference_position()
            record = index.get(thing) or {}
            notes = self.reference_passages(record, desc=False)
            if notes == "":
                return []
            mine = self.reference_vendors(record)
            others = {s for other in index.values() for s in self.reference_vendors(other)} - mine
            names, _ = self.reference_portions(record)
            shelf = self.rack_belonging(record)
            bits = [self.heading_belonging(record), shelf] + [
                str(x) for entry in record.values() if isinstance(entry, dict) for x in entry.values() if isinstance(x, (str, int, float))]
            known = names | set(reference_pieces(" | ".join(bits)))
            multi = self.needs.items()
            k = self.ware_belonging(f"x ({shelf})") if multi and shelf else None
            clauses = re.split(r"(?<=[.!?;])\s+", notes)

            def hands_off(prop: str, clause: str) -> bool:
                tokens = list(re.finditer(r"[A-Za-z][A-Za-z'-]*", clause))
                # (review: the word just before 'by <Name>' is the other listing's kind ('... the lamp by Name'), not a
                # mention of the property; it is blanked for the property test only)
                masked = re.sub(r"[A-Za-z0-9]+(?=\s+by\s+[A-Z])", lambda h: " " * len(h.group(0)), clause)
                where = next((m.start() for m in tokens if trait_titled(prop, masked[:m.end()])), None)
                if where is None:
                    return False

                def strange(text: str, low: str) -> bool:
                    if low in others:
                        return True
                    capital = text[:1].isupper() and len(low) >= 4
                    return capital and low not in mine and not set(reference_pieces(text)) & known and low not in known

                for n, m in enumerate(tokens):
                    low = reference_moniker(m.group(0))
                    if n == 0 or not low or m.start() <= where:
                        continue
                    negations = {t.group(0).lower() for t in tokens[max(0, n - 5): n]} & {"not", "nor", "never", "no", "neither"}
                    if strange(m.group(0), low) and not negations:
                        return True
                return False

            found = []
            in_force = self.needs.traits_remaining()
            for req in in_force:
                head = str(req).partition(":")[0]
                other_product = multi and k is not None and self.ware_belonging(head) not in (k, None)
                label = "" if other_product else re.split(r"\s*\(", head)[0].strip()
                clause = next((c for c in clauses if hands_off(label, c)), None) if label else None
                if clause is not None:
                    found.append((str(req), label, clause.strip()))
            return found
        except Exception:
            return []

    def annotations_comment(self, call_name: str, call_args: dict[str, Any]) -> str:
        """CE6: after a lookup, the remark for a cart line or verified candidate whose own notes hand a required property to
        another listing (once per listing and requirement version); its verified mark is removed."""
        skipped = (self.role("search"), self.role("cart"), self.role("cart_view"), self.role("message"))
        if call_name in skipped or call_name in self.concluders():
            return ""
        iteration = str(self.needs.iteration)
        remarks = []
        looked_up = sorted(self.inquired_uids(call_args))
        for ident in looked_up:
            stamp = f"{ident}|{iteration}"
            if ident not in self.handcart.measures and stamp not in self.succeeded:
                continue
            reported = self.__dict__.setdefault("own_told", set())
            hits = self.own_redirects(ident) if stamp not in reported else []
            if len(hits) == 0:
                continue
            reported.add(stamp)
            self.succeeded.discard(stamp)
            self.journal.append({"kind": "own_redirect", "id": ident, "props": [p for _, p, _ in hits]})
            which = "; ".join(f"{p} (\"{s[:160]}\")" for _, p, s in hits)
            reqs = "; ".join(f"'{r}'" for r, _, _ in hits)
            pronoun = "them" if len(hits) > 1 else "it"
            ending = (". Should they, remove it from the cart and add a different verified fit instead."
                      if ident in self.handcart.measures else ".")
            remarks.append(f"The notes on {ident} itself may hand {which} over to a different listing: in that case its own "
                           f"field does not count toward {reqs} (the grader uses the other listing's value), so it satisfies "
                           f"{pronoun} only when that listing's record shows it; the item check is repeated before any "
                           f"order{ending}")
        return " ".join(remarks)

    # ----------------------------------------------------------------------------------------------- CE2b lines where the shopper points elsewhere
    def reference_expression(self, line: str) -> tuple[str, int, int, dict[str, Any]] | None:
        """CE2b: (phrase, start, end, listing) for the part of a shopper's line that sets a value by pointing at another
        listing. It runs from the first pointer words ahead of the seller's name within its clause (relation words, then
        words pointing at a listing or page, then the seller) to the end of that listing's description: an '(or ...)'
        aside if one comes before the sentence ends, otherwise the sentence end. The listing itself is parsed from the
        line (ref_parse: seller, kind, details). None when the line points at no listing or page (a bare differ word -
        'anything but <Name>' - excludes a value or a seller rather than pointing) or at one with no seller."""
        text = str(line) if line else ""

        def seller_here(name: Any) -> bool:
            return self.reference_vendor_approved(name, text)

        parsed = reference_decode(text, seller_here) if reference_passage(text) else None
        if not parsed or not seller_here(parsed["seller"]):
            return None
        seller_rx = r"(?<![A-Za-z'])" + re.escape(parsed["seller"]) + r"(?:'s)?(?![A-Za-z])"
        hit = re.search(seller_rx, text)
        if hit is None:
            return None
        before = text[:hit.start()]
        breaks = [before.rfind(c) for c in (",", ";", "!", "?")]
        breaks += [m.start() for m in re.finditer(r"[.](?:\s|$)", before)]
        cut = max(breaks + [-1])
        clause = before[cut + 1:].lower()
        offsets = []
        for pattern in (DIVERGE_TERMS, REFERENCE_TERMS, INDICATE_TERMS):
            offsets.extend(m.start() for m in re.finditer(pattern, clause))
        if not offsets:
            return None       # (a name with no pointer words ahead of it in its clause points at nothing: 'nowt but Nike will do')
        begin = cut + 1 + min(offsets)
        after = text[hit.end():]
        aside = re.search(r"\(\s*or\b[^)]*\)", after)
        stop = re.search(r"[.!?;](?:\s|$)", after)
        if aside is not None and (stop is None or aside.start() < stop.start()):
            finish = hit.end() + aside.end()
        elif stop is not None:
            finish = hit.end() + stop.start()
        else:
            finish = hit.end() + len(after)
        phrase = text[begin:finish].strip(" ,;")
        return (phrase, begin, finish, parsed) if phrase else None


    def reference_lineitem_terms(self, qtext: str | None, outside: str) -> list[str]:
        """CE2b: word runs (three or more letters, no link word, no product kind word) that a shopper's line shares,
        outside its pointing phrase, with the question asked - longest runs first; they name the property the reply
        concerns."""
        if not qtext:
            return []
        product_words: set[str] = set()
        products = self.needs.items() or [{"product": str(self.needs.perusal.get("product") or "")}]
        for entry in products:
            product_words.update(re.findall(r"[a-z]+", str(entry["product"]).lower()))
        asked = re.findall(r"[a-z0-9]+", str(qtext).lower())

        def viable(w: str) -> bool:
            return not (w in BOND_TERMS or len(w) < 3 or w in product_words or w.rstrip("s") in product_words)

        runs = []
        for size in (3, 2, 1):
            for lo in range(len(asked) - size + 1):
                window = asked[lo: lo + size]
                if not all(viable(w) for w in window):
                    continue
                joined = " ".join(window)
                if trait_titled(joined, outside):
                    runs.append(joined)
        return runs

    def reference_remedy(self, answer: dict[str, Any], lines: list[str], qtext: str | None) -> list[str]:
        """CE2b (hook in Needs.hear, run on the listener's answer before it is applied). For each shopper line pointing at
        another listing (ref_phrase): exactly ONE need for the property the line names outside its pointing phrase - the
        listener's reference property, else a property of a pointing need, else of any need, else a store field name or a
        stated property, picking the longest such name - whose value keeps the line's pointing words (unless the
        listener's value already points at the same seller with the same relation), tagged with the product the line
        names outside the phrase (else the listener's tag, else the question's product); a reference entry parsed from the
        line; and the other listing's facts (a need, drop or freed property named only inside the phrase, or a need whose
        value words all sit inside it) removed from the answer. Returns the lines with their pointing phrases blanked."""
        def label_of(req: Any) -> str:
            head = str(req).partition(":")[0]
            return re.split(r"\s*\(", head)[0].strip()

        def tag_list(req: Any) -> list[str]:
            return re.findall(r"\(([^)]*)\)", str(req).partition(":")[0])

        def points(req: str) -> bool:
            tail = req.partition(":")[2]
            return bool(reference_passage(tail) or re.search(DIVERGE_TERMS, tail.lower()))

        def rework(line: str, phrase: str, start: int, end: int, parsed: dict[str, Any]) -> None:
            inner, outer = line[start:end], line[:start] + " " + line[end:]
            raw = answer.get("needs")
            needs = [n for n in raw or [] if isinstance(n, str) and ":" in n] if isinstance(raw, list) else []
            raw = answer.get("references")
            refs = [r for r in raw or [] if isinstance(r, dict)] if isinstance(raw, list) else []

            def only_inside(p: str) -> bool:
                return bool(p) and trait_titled(p, inner) and not trait_titled(p, outer)

            def other_fact(req: str) -> bool:
                if only_inside(label_of(req)):
                    return True
                words = set(reference_pieces(str(req).partition(":")[2]))
                return bool(words) and words <= set(reference_pieces(inner)) and not words & set(reference_pieces(outer))

            def rank(p: str) -> tuple:
                return trait_titled(p, outer, exact=True), len(re.findall(r"[a-z0-9]+", p.lower())), len(p)

            asked = [label_of(a) for a in str(self.__dict__.get("asking_label") or "").split(";") if label_of(a)]
            ref_props = [str(r.get("property") or "") for r in refs]
            pointing_props = [label_of(n) for n in needs if points(n)]
            sources = [ref_props, pointing_props, [label_of(n) for n in needs], asked,
                       sorted(self.depot_slots()) + [label_of(r) for r in self.needs.written],
                       self.reference_lineitem_terms(qtext, outer)]
            hit_lists = ([p for p in source if p and trait_titled(p, outer)] for source in sources)
            prop = next((max(hits, key=rank) for hits in hit_lists if hits), "")
            if not prop:
                surplus_alt = ref_props + pointing_props + (asked if len(asked) == 1 else [])
                prop = next((p for p in surplus_alt if p and not only_inside(p)), "")
            if not prop:
                self.journal.append({"kind": "ref_line_unplaced",
                                 "line": line[:200]})
                return
            multi = self.needs.items()

            def ware_position() -> int | None:
                found = self.ware_belonging(outer)
                if found is None:
                    own_tags = [t for n in needs if identical_trait(label_of(n), prop) for t in tag_list(n)]
                    found = self.ware_belonging(f"x ({own_tags[-1]})") if own_tags else None
                if found is None and qtext:
                    found = self.ware_belonging(qtext)
                return found

            k = ware_position() if multi else None
            tag = "" if k is None else multi[k]["product"]

            def same_product(req: Any) -> bool:
                labels = tag_list(req)
                return not tag or not labels or bool(category_terms(labels[-1]) & category_terms(tag))

            ours = [n for n in needs if identical_trait(label_of(n), prop) and same_product(n)]
            relation, tolerance, unit = reference_relationship(phrase)
            seller_key = reference_moniker(parsed["seller"])

            def reusable(n: str) -> bool:
                tail = n.partition(":")[2]
                return points(n) and reference_relationship(tail)[0] == relation and seller_key in reference_moniker(tail)

            keep = next(filter(reusable, ours), None)
            head = ours[0].partition(":")[0].strip() if ours else (f"{prop} ({tag})" if tag else prop)
            if tag and multi and not re.findall(r"\(([^)]*)\)", head):
                head += f" ({tag})"
            need = keep if keep is not None else f"{head}: {phrase}"
            foreign_facts = [n for n in needs if n not in ours and other_fact(n)]
            spot = needs.index(ours[0]) if ours else len(needs)
            remaining = [n for n in needs if n not in ours and n not in foreign_facts]
            remaining.insert(spot, need)
            answer["needs"] = remaining
            current = answer.get("requirements_now")
            if isinstance(current, list):
                def survives(req: str) -> bool:
                    if req in self.needs.written:
                        return True
                    if req in foreign_facts or other_fact(req):
                        return False      # (facts that belong to the other listing)
                    # (a requirement on this property is superseded by the need built here)
                    return not (identical_trait(label_of(req), prop) and same_product(req))

                kept = [r for r in current if isinstance(r, str) and survives(r)]
                answer["requirements_now"] = kept if need in kept else kept + [need]
            dropped = answer.get("dropped")
            if isinstance(dropped, list):
                answer["dropped"] = [d for d in dropped if not isinstance(d, str) or not (only_inside(label_of(d)) or other_fact(d))]
            free = answer.get("no_preference")
            if isinstance(free, list):
                answer["no_preference"] = [f for f in free
                                           if not isinstance(f, str) or not (only_inside(label_of(f)) or identical_trait(label_of(f), prop))]
            matches = [r for r in refs if identical_trait(str(r.get("property") or ""), prop)]
            if not matches:
                refs.append({"property": prop})
            entry = matches[0] if matches else refs[-1]
            same_seller = self.reference_vendor_approved(entry.get("seller"), line) and reference_moniker(entry.get("seller")) == seller_key
            if not same_seller:
                entry.update({"seller": parsed["seller"], "kind": parsed["kind"], "details": [*parsed["details"]]})
            elif not entry.get("details"):
                entry["details"] = [*parsed["details"]]
            if "kind" not in entry:
                entry["kind"] = parsed["kind"]
            if not (relation == "same" and entry.get("relation")):
                entry.update({"relation": relation, "tolerance": tolerance, "unit": unit})
            answer["references"] = refs
            answer["ref_needs"] = [*(answer.get("ref_needs") or []), need]
            self.journal.append({"kind": "ref_line", "prop": prop, "tag": tag, "relation": relation, "need": need[:200],
                             "kept": keep is not None, "details_out": [f[:80] for f in foreign_facts]})

        result = []
        for text in lines:
            located = self.reference_expression(text)
            if located:
                _, start, end, _ = located
                result.append(text[:start] + " " + text[end:])
                rework(text, *located)
            else:
                result.append(text)
        return result

    @staticmethod
    def reference_monikers_trait(prop: str, text: str) -> bool:
        """Whether a text names the property: some property word of 3+ letters equals a word of the text (or the two
        share their first 4 letters)."""
        terms = [w for w in trait_terms(prop) if len(w) >= 3]
        found = set(re.findall(r"[a-z]+", str(text).lower()))
        stems = {t[:4] for t in found if len(t) >= 4}
        return not terms or any(term in found or (len(term) >= 4 and term[:4] in stems) for term in terms)

    def reference_profile(self, ident: str) -> str:
        record = self.reference_position().get(ident) or {}
        card_facts: dict[str, str] = {}
        for group in (e for e in record.values() if isinstance(e, dict)):
            card_facts.update((str(f), str(x)[:90]) for f, x in list(group.items())[:40]
                              if isinstance(x, (str, int, float)) and not isinstance(x, bool))
        taught = self.vendor_slots()

        def seller_field(field: Any, entry: Any) -> bool:
            low = str(field).lower()
            if isinstance(entry, str) and entry.strip() and len(entry) <= 60 and "id" not in low:
                named = re.search(VENDOR_SLOTS, low) or str(field) in taught
                return bool(named) and not re.search(r"note|text|desc|comment|detail|info|message|section", low)
            return False

        sellers = sorted({str(e) for f, e in record.items() if seller_field(f, e)})
        return into_passage({"id": ident, "sold by": " / ".join(sellers), "category": self.rack_belonging(record),
                        "title": self.heading_belonging(record)[:160], "facts": card_facts, "notes": self.reference_passages(record)[:900]}, 5000)

    def reference_inquire(self, ref: dict[str, Any], pick: list[str], reason: str) -> dict[str, Any] | None:
        """The hop call on the cheap model: which record is the target, and where it leads (go_to) or its value."""
        reply = None
        if self.reference_invokes_approved():
            self.ref_calls = 1 + self.__dict__.get("ref_calls", 0)
            sections = [REFERENCE_JUMP.replace("PROPERTY", ref["prop"]).replace("DESCRIPTION", self.reference_description(ref)),
                        "RECORDS:\n" + "\n".join(self.reference_profile(i) for i in pick)]
            sections += [f"YOUR PREVIOUS ANSWER WAS REJECTED: {reason} Answer once more, using the records."] if reason else []
            reply = self.protocol.confer("\n\n".join(sections), 3000, ASSESSOR[:1])
            self.journal.append({"kind": "ref_hop", "prop": ref["prop"], "ids": pick, "answer": reply})
        return reply


    def reference_enact(self, ref: dict[str, Any], pick: list[str], answer: Any, again: bool = True) -> None:
        """BT4/BT5: applies a checked hop answer (go_to gives the next target; a value resolves it); a rejected answer is
        asked again once, together with the reason."""
        judged = ref["judged"]
        if isinstance(answer, dict):
            named = answer.get("target")
            selected = str(named).strip() if isinstance(named, (str, int)) and not isinstance(named, bool) else ""
            gaps = answer.get("missing")
            lacking = [str(x) for x in gaps or [] if str(x).strip()] if isinstance(gaps, list) else []
        else:
            ref["fails"] = ref["fails"] + 1
            if ref["fails"] > 1:
                judged.update(pick)
                self.reference_cease(ref, "the hop check returned nothing usable")
            return
        if selected not in pick or lacking:
            # (none of them is the listing - or the one named lacks a detail from the description and so is not it:
            # the following rows get read)
            judged.update(pick)
            self.reference_uids_attach(pick)
            if selected in pick:
                self.journal.append({"kind": "ref_partial", "prop": ref["prop"], "target": selected, "missing": lacking[:4]})
            return
        where, problem, go, value, pairs = self.reference_confirm(ref, selected, answer, pick)
        if again and problem and self.reference_invokes_approved():
            second = self.reference_inquire(ref, pick, problem)
            self.reference_enact(ref, pick, second, again=False)
            return
        self.reference_uids_attach(pick)
        if problem != "":
            self.journal.append({"kind": "ref_invalid", "prop": ref["prop"], "target": selected, "problem": problem[:300]})
            if where != "target":
                judged.update(pick)
                ref["chain"].append(selected)      # (the listing was located, but what its notes say could not be read)
                self.reference_cease(ref, problem)
            else:
                judged.add(selected)    # (only that record is put aside; the rest get judged again)
        elif selected in ref["chain"]:
            judged.update(pick)
            self.reference_cease(ref, "its notes loop back to a listing already followed")
        else:
            judged.update(pick)
            if go:
                self.ref_go(ref, selected, go)
            else:
                self.reference_determine(ref, selected, value, pairs)

    def reference_confirm(self, ref: dict[str, Any], target: str, answer: dict[str, Any],
                     pick: list[str] | None = None) -> tuple[str, str, dict[str, Any] | None, str, list[str]]:
        """Checks a hop answer against the target's own record and returns (where, problem, go_to, value, goes_with). The
        target must show every detail given in words, and no other record shown may show more of the details verbatim;
        a go_to seller must be written in the target's notes, in the sentence that names the property, and must not be
        the record itself; go_to details absent from those notes are dropped; a value must be written in the record and
        a goes_with value in its notes."""
        record = self.reference_position().get(target) or {}
        names, values = self.reference_portions(record)
        vague = ref.get("loose") or set()
        wanted = ref["target"]["details"]
        unseen = next((p for p in wanted if p not in vague and not self.reference_displays(record, p, (names, values))), None)
        if unseen is not None:
            return "target", f"{target} does not show '{unseen}', so it is not the listing described.", None, "", []

        def coverage(ident: str) -> int:
            fields, words = self.reference_portions(self.reference_position().get(ident) or {})
            token_sets = (set(reference_pieces(p)) for p in wanted)
            return sum(1 for t in token_sets if t and t <= fields | words)

        def clean(raw: Any) -> list[str]:
            items = raw if isinstance(raw, list) else []
            return [str(x).strip() for x in items if isinstance(x, (str, int, float)) and not isinstance(x, bool) and str(x).strip()]

        rival = max(pick or [target], key=coverage)
        if coverage(rival) > coverage(target):
            return "target", f"{rival} shows more of the named details than {target} does.", None, "", []
        notes = self.reference_passages(record)
        note_toks = set(reference_pieces(notes))
        hop = answer.get("go_to") if isinstance(answer.get("go_to"), dict) else None
        next_seller = str(hop.get("seller") or "").strip() if hop else ""
        if next_seller:
            seller_rx = r"(?<![a-z0-9])" + re.escape(next_seller.lower()) + r"(?:'s)?(?![a-z0-9])"
            low_notes = notes.lower()
            if not re.search(seller_rx, low_notes):
                return "answer", f"the go_to seller '{next_seller}' does not appear in {target}'s notes.", None, "", []
            about_prop = [s for s in re.split(r"(?<=[.!?])\s+", notes)
                          if re.search(seller_rx, s.lower()) and self.reference_monikers_trait(ref["prop"], s)]
            if not about_prop:
                return "answer", f"the note that names {next_seller} concerns a different property, not {ref['prop']}.", None, "", []

            def mostly_noted(piece: str) -> bool:
                toks = set(reference_pieces(piece))
                return bool(toks) and 2 * len(toks & note_toks) >= len(toks)

            following = {"seller": next_seller, "kind": str(hop.get("kind") or "").strip()[:80],
                         "details": [p for p in clean(hop.get("details")) if mostly_noted(p)][:6]}
            itself = reference_moniker(next_seller) in self.reference_vendors(record) and self.reference_slot(following) == self.reference_slot(ref["target"])
            if itself:
                return "answer", f"the go_to points back at {target} itself.", None, "", []
            return "", "", following, "", []
        found = str(answer.get("value") or "").strip()
        found_toks = set(reference_pieces(found))
        if not (found and found_toks and found_toks <= values | names):
            return "answer", f"the value '{found}' does not appear in {target}'s record.", None, "", []

        def partner(piece: str) -> bool:
            toks = set(reference_pieces(piece))
            return bool(toks) and toks <= note_toks and toks != found_toks

        partners = [p for p in dict.fromkeys(clean(answer.get("goes_with"))) if partner(p)]
        return "", "", None, found, partners[:3]

    def ref_go(self, ref: dict[str, Any], target: str, go: dict[str, Any]) -> None:
        """The target's notes redirect the reader: the next target becomes current (a loop or too many hops ends it)."""
        chain = ref["chain"]
        chain.append(target)
        stamp = self.reference_slot(go)
        hops_after = ref["hops"] + 1
        halt = ""
        if stamp in ref["seen"]:
            halt = "its notes loop back to a listing already followed"
        elif hops_after >= REFERENCE_JUMPS:
            halt = f"over {REFERENCE_JUMPS} listings in a row redirect the reader elsewhere"
        if halt:
            self.reference_cease(ref, halt)
            return
        ref["seen"].add(stamp)
        ref["hops"] = hops_after
        fresh = {"target": go, "judged": set(), "queries": [], "pages": 0, "tries": 0, "fails": 0, "unindexed": False, "loose": set()}
        ref.update(fresh)
        self.journal.append({"kind": "ref_go", "prop": ref["prop"],
                         "from": target, "to": go})
        self.reference_inform(f"{target} is the listing named for {ref['head']}, and its notes redirect the reader to "
                      f"{self.reference_description(ref)}: the code is following them.")

    def reference_setting(self, ref: dict[str, Any]) -> str:
        """The requirement value built from what was read: a range for 'within', a bound for 'at most' / 'at least',
        otherwise the value itself (or the closed list of it plus its paired values)."""
        value = str(ref["value"])
        pairs = list(ref.get("goes_with") or [])
        relation = ref["relation"]
        if relation == "within" and whether_numeric(ref.get("tol")):
            m = re.search(r"(-?\d+(?:\.\d+)?)\s*([a-z%]*)", value.lower().replace(",", ""))
            if m:
                centre = float(m.group(1))
                spread = float(ref["tol"])
                unit = m.group(2) or str(ref.get("unit") or "")
                low, high = round(centre - spread, 6), round(centre + spread, 6)
                return f"{low:g} to {high:g} {unit}".strip() + f" (within {spread:g} of {value})"
        if relation in ("at most", "at least"):
            return f"{relation} {value}"
        options = [value] + [p for p in pairs if p.lower() != value.lower()]
        if relation == "differ":
            # (CE2: an exclusion - every other value satisfies it)
            return " and ".join(f"not {v}" for v in options)
        return " or ".join(options)

    def reference_determine(self, ref: dict[str, Any], target: str, value: str, pairs: list[str]) -> None:
        """BT6: the value read becomes the requirement (and needs.version goes up)."""
        text = str(value or "").strip()
        if text == "":
            self.reference_cease(ref, f"{target} carries no value for it")
            return
        chain = ref["chain"]
        chain.append(target)
        ref.update(status="resolved", value=text, goes_with=list(pairs))
        updated = f"{ref['head']}: {self.reference_setting(ref)}"
        ref["now"] = updated
        self.reference_substitute(ref, updated)
        needs = self.needs
        needs.resolved.append(updated)
        needs.iteration = needs.iteration + 1
        self.journal.append({"kind": "ref_resolved", "req": ref["req"][:200],
                         "now": updated, "chain": list(chain)})
        route = f", reached from {chain[0]} by following the listings' notes" if len(chain) > 1 else ""
        joined = " and ".join(pairs)
        exclusion = ref["relation"] == "differ"
        if exclusion:
            extra = (f"; its notes pair {joined} with it" if pairs else "") + (
                ". The shopper's value MUST DIFFER from that one: every other value satisfies it, that value does not")
        elif pairs:
            extra = f"; its notes say {ref['value']} goes together with {joined}, so each of them satisfies it"
        else:
            extra = ""
        self.reference_inform(f"RESOLVED: another listing sets {ref['head']}, and {target} shows {ref['value']}{route}{extra}. The "
                      f"requirement now reads: {updated}.")

    def reference_substitute(self, ref: dict[str, Any], new_req: str) -> None:
        """Replaces the reference requirement with the resolved one everywhere it lives: the stated, revealed and rebuilt
        lists, the reading's requirements, and its product's entry."""
        book = self.needs
        first_seller = reference_moniker(ref["first"]["seller"])
        own_tag = ref["tag"]

        def loosely(req: Any) -> bool:
            label, _, said = str(req).partition(":")
            labels = re.findall(r"\(([^)]*)\)", label)
            if labels and own_tag and not category_terms(labels[-1]) & category_terms(own_tag):
                return False      # (a requirement of a different product)
            names_seller = bool(first_seller) and first_seller in reference_moniker(said)
            return names_seller and identical_trait(re.split(r"\s*\(", label)[0], ref["prop"])

        def replaces(req: Any) -> bool:
            return bool(self.reference_correspond(ref, req)) or bool(loosely(req))


        def swapped(reqs: list) -> list:
            return [new_req if replaces(r) else r for r in reqs]

        changed = []
        for attr in ("written", "uncovered"):
            before = list(getattr(book, attr))
            after = swapped(before)
            changed.append(after != before)
            setattr(book, attr, list(dict.fromkeys(after)))
        if book.now_reqs is not None:
            after = swapped(book.now_reqs)
            book.now_reqs = list(dict.fromkeys(after if new_req in after else after + [new_req]))
        if not any(changed):
            book.uncovered.append(new_req)
        perusal = book.perusal
        if isinstance(perusal.get("requirements"), list):
            perusal["requirements"] = list(book.written)
        settled = new_req.partition(":")[2].strip()
        entries = [e for e in perusal.get("items") or [] if isinstance(e, dict) and isinstance(e.get("requirements"), list)]

        def foreign_entry(entry: dict[str, Any]) -> bool:
            # (a different product's entry: its requirements have no product tag)
            return len(entries) >= 2 and bool(own_tag) and not (category_terms(entry.get("product") or "") & category_terms(own_tag))

        for entry in entries:
            if not foreign_entry(entry):
                entry["requirements"] = [f"{str(r).partition(':')[0].strip()}: {settled}" if replaces(r) else r
                                         for r in entry["requirements"]]

    def reference_cease(self, rf: dict[str, Any], cause: str) -> None:
        """BT7: marks a still-open reference as stuck. Soft by design: an unknown value never vetoes a listing and is
        never grounds for a decline."""
        if rf["status"] == "open":
            rf["status"] = "stuck"
            rf["why"] = cause
            self.journal.append(dict(kind="ref_stuck", req=rf["req"][:200], why=cause[:300], chain=list(rf["chain"])))
            self.reference_inform(f"UNRESOLVED: {rf['head']} ({cause}). Its value remains unknown. Should the notes of a listing you "
                          "have read show where the chain ends, take that value; if not, do not decline or reject any listing "
                          "on its account.")

    def reference_inform(self, txt: str) -> None:
        """Passes `txt` to the actor a single time, appended to the newest result it is about to read."""
        converse = self.converse
        if converse and converse[-1].get("role") in ("tool", "user") and isinstance(converse[-1].get("content"), str) and len(converse) > 2:
            converse[-1]["content"] += "\n[reference] " + txt
            return
        self.ref_told = [*(vars(self).get("ref_told") or []), txt]

    def ref_line(self) -> str:
        """The status-line part about references: where each one stands."""
        try:
            shown: list[str] = []
            for rf in vars(self).get("refs") or []:
                stock_flag = rf["status"]
                if stock_flag == "open":
                    shown.append(f"{rf['head']}: still being traced by the code (so far: {self.reference_description(rf)})")
                elif stock_flag == "resolved":
                    differs = rf.get("relation") == "differ"
                    shown.append(f"{rf['now']} (taken from {rf['chain'][-1]}"
                                 + ("; the value has to differ from it" if differs else "") + ")")
                elif stock_flag == "stuck":
                    shown.append(f"{rf['head']}: unresolved, value unknown (no listing may be declined or rejected because of it)")
            pending = vars(self).pop("ref_told", None)
            line_out = "\n[references] " + "; ".join(shown) if shown else ""
            if pending:
                line_out += "".join("\n[reference] " + t for t in pending)
            return line_out
        except Exception:
            return ""

    def reference_annotation(self, focus: str = "") -> str:
        """BT6/BT7: the reference block for the item check, the decline audit and the rank check. A resolved reference shows
        its value and the listing it was taken from (with any pairing); an unresolved one shows as unknown, together with
        the newest records its chain got to."""
        try:
            entries: list[str] = []
            tails: list[str] = []
            for rf in vars(self).get("refs") or []:
                stock_flag = rf["status"]
                if stock_flag in ("open", "stuck"):
                    chain = rf["chain"]
                    route = ("the chain so far: " + " -> ".join(chain)) if chain else "the listing it names was not found"
                    entries.append(f"- {rf['head']}: the code could NOT resolve it ({route}). If a record below is that listing, "
                                   "or the last listing the chain got to, and states its own value for it without redirecting the "
                                   "reader, that value is the requirement; if not, the value is unknown and no item may fail on "
                                   "it alone.")
                    tails.extend(rid for rid in chain[-2:] if rid != focus)
                    continue
                if stock_flag != "resolved":
                    continue
                chain = rf["chain"]
                hops = f" (got there from {chain[0]} by way of the listings' notes)" if len(chain) > 1 else ""
                partners = rf["goes_with"]
                if rf.get("relation") == "differ":
                    extra = f"; its notes pair {' and '.join(partners)} with it" if partners else ""
                    entries.append(f"- {rf['now']}: the shopper's {rf['prop']} MUST DIFFER from what {chain[-1]} shows "
                                   f"({rf['value']}){hops}{extra}: every other value satisfies it, that one does not; the "
                                   f"listings along its chain ({', '.join(chain)}) all show it.")
                else:
                    extra = (f"; according to its notes {rf['value']} comes with {' and '.join(partners)}, so each of those "
                             "satisfies it" if partners else "")
                    entries.append(f"- {rf['now']}: whatever value {chain[-1]} shows{hops}{extra}.")
            if not entries:
                return ""
            block = ("\n\nREQUIREMENTS SET BY POINTING AT ANOTHER LISTING (the code traced that listing's notes through the "
                     "store's records):\n" + "\n".join(entries))
            records = [line for lid in dict.fromkeys(reversed(tails)) for line in self.vault.concerning(lid, 1)]
            if records:
                block += "\nRECORDS OF THE LISTINGS ITS CHAIN LAST GOT TO:\n" + "\n".join(records[:2])
            return block
        except Exception:
            return ""

    # ============================================================================== CG: covering the cheapest listings


class SettingSeek:
    """Errand part: price-limit listings and searches driven by required values."""

    def roof_discover(self, sent_args: dict[str, Any], rowset: list, n_total: int, reply: Any) -> None:
        """CG1: learns from use what this store's searches reveal about their filters and their depth. Each search's total
        is stored under its arguments (paging left out). A numeric limit argument is taken to HOLD in a mode once every row
        of a call is priced at or under it and every row sits on the single category label that one of the call's text
        arguments names; a mode whose rows ever exceed its limit is never trusted with it again. The learned listing keeps
        the call's other arguments, minus a text value that every row carries in some field (a maker filter); a call with
        fewer free words takes its place. Depth: a full page offering no next page while its total promised more rows
        tells how far one search can reach."""
        pinfo = self.sheet_param()
        pkey = pinfo[0] if pinfo else None
        base = {name: val for name, val in sent_args.items() if name != pkey}
        count = int(n_total)
        vars(self).setdefault("ceil_seen", {})[json.dumps(base, sort_keys=True, default=str)] = count
        dict_rows = [rr for rr in rowset if isinstance(rr, dict)]
        if pinfo and pinfo[1] is True and dict_rows and isinstance(reply, dict):
            start = sent_args.get(pkey, pinfo[2])
            flags = [vv for name, vv in reply.items()
                     if re.search(r"next|more|onward|continu|cursor", str(name).lower()) and not isinstance(vv, (list, dict))]
            per_page = max(len(dict_rows), self.sheet_dimension())
            exhausted = bool(flags) and not any(vv is not None and vv is not False and vv != "" for vv in flags)
            if whether_numeric(start) and exhausted and len(dict_rows) >= per_page:
                reach = int((start - pinfo[2] + 1) * per_page)
                if n_total > reach:
                    self.page_cap = min(vars(self).get("page_cap") or reach, reach)
        _, fmode, _, skip_keys = self.seek_slots(sent_args)

        def is_limit(name: str, val: Any) -> bool:
            if not whether_numeric(val) or isinstance(val, bool) or name == pkey or name in skip_keys or not val > 1:
                return False
            return not re.search(r"page|size|count|result|min|low|from|start|offset", str(name).lower())
        limits = [name for name, val in sent_args.items() if is_limit(name, val)]
        if not dict_rows or len(limits) != 1:
            return
        bkey = limits[0]
        spend_lim = float(sent_args[bkey])
        label_set = {self.rack_belonging(rr) for rr in dict_rows}
        named = [name for name, val in sent_args.items()
                 if isinstance(val, str) and name != pkey and val.strip() and val.strip().lower() in label_set]
        skey = named[0] if len(named) == 1 else None
        fixed = {name: val for name, val in fmode if name != skey}
        msig = json.dumps([sorted(fixed.items()), bkey], default=str)
        price_vals = [self.tablerow_expense(rr) for rr in dict_rows]
        bad = vars(self).setdefault("ceil_bad", set())
        if any(p is not None and p > spend_lim + 1e-9 for p in price_vals):
            bad.add(msig)      # this mode does not honour the limit
            return
        if msig in bad or skey is None or len(label_set) != 1 or "" in label_set or None in price_vals:
            return

        def on_every_row(val: str) -> bool:
            want = val.strip().lower()
            for rr in dict_rows:
                if not any(isinstance(x, str) and x.strip().lower() == want for kk, x in rr.items() if not self.rack_belonging({kk: x})):
                    return False
            return True

        def maker_like(name: str, val: Any) -> Any:
            return isinstance(val, str) and val.strip() and name not in fixed and on_every_row(val)
        shape_tmpl = {name: val for name, val in base.items() if name not in (bkey, skey) and not maker_like(name, val)}
        wlist = sum(len(val.split()) for name, val in shape_tmpl.items() if isinstance(val, str) and name not in fixed)
        schema = self.structures.get(self.role("search") or "") or {}
        props = schema.get("properties") or {}
        prior = vars(self).get("ceil_mode")
        if prior and prior["sig"] not in bad and prior["words"] < wlist:
            return
        limit_spec = (props.get(bkey) or {}) if isinstance(props, dict) else {}
        self.ceil_mode = dict(sig=msig, tmpl=shape_tmpl, mode=fixed, shelf=skey, bound=bkey, words=wlist,
                              int=limit_spec.get("type") == "integer")

    @staticmethod
    def tablerow_expense(rec_row: dict[str, Any]) -> float | None:
        for name, val in rec_row.items():
            if whether_numeric(val) and not isinstance(val, bool) and re.search(r"price|cost", str(name).lower()):
                return float(val)
        return None

    def roof_specification(self) -> dict[str, Any] | None:
        """CG1: the learned price-filter listing, provided its mode has not broken a limit since."""
        learned = vars(self).get("ceil_mode")
        if not learned or learned["sig"] in vars(self).get("ceil_bad", set()):
            return None
        return learned

    def catalog_advert(self, cargs: dict[str, Any]) -> bool:
        """CG: whether a search call uses the learned listing mode (its mode arguments, or its limit argument when the
        mode has none)."""
        cfg = self.roof_specification()
        if cfg and isinstance(cargs, dict):
            wanted = cfg["mode"]
            if not wanted:
                return cfg["bound"] in cargs
            return all(cargs.get(name) == val for name, val in wanted.items())
        return False

    def sheet_dimension(self) -> int:
        sizes = [entry.get("size", 0) for entry in vars(self).get("page_book", {}).values()]
        return max(sizes, default=0)

    def apparent_reach(self) -> int | None:
        """CG1: how many rows one search can reach: from the schema's last page (or offset) and the page size, otherwise the
        depth a capped page revealed; None while that is unknown."""
        pinfo = self.sheet_param()
        psize = self.sheet_dimension()
        if not (pinfo and psize > 0 and pinfo[1] != "token" and pinfo[3] is not None):
            return vars(self).get("page_cap")
        span = pinfo[3] - pinfo[2]
        if pinfo[1]:
            return int((span + 1) * psize)
        return int(span + psize)

    def unprocessed_tag(self, label_set: set[str]) -> str:
        """CG3: among the lower-case category labels `label_set`, the one carried by the most rows seen, spelled the way the
        rows spell it."""
        if not label_set:
            return ""
        headcounts: dict[str, int] = {}

        def note_row(node: dict[str, Any]) -> None:
            if not any(isinstance(vv, str) and vv in self.vault.ids and "id" in str(slot).lower() for slot, vv in node.items()):
                return
            for slot, val in node.items():
                if isinstance(val, str) and 0 < len(val.strip()) <= 60\
                        and re.search(r"categor|shelf|department|aisle", str(slot).lower()) and "id" not in str(slot).lower():
                    spelled = val.strip()
                    break
            else:
                return
            if spelled.lower() in label_set:
                headcounts[spelled] = headcounts.get(spelled, 0) + 1
        for entry in self.vault.items:
            traverse_mappings(entry, note_row)
        best, best_n = "", 0
        for name in sorted(headcounts):
            if headcounts[name] > best_n:
                best, best_n = name, headcounts[name]
        return best

    def beneath(self, amount: Any) -> float | int | None:
        """CG3: the highest limit that still excludes `amount`: one unit lower for whole prices (or a whole-number limit),
        otherwise one hundredth lower; None when nothing would remain."""
        if not whether_numeric(amount):
            return None
        cfg = self.roof_specification() or {}
        val = float(amount)
        whole = val.is_integer()
        if not (cfg.get("int") or whole):
            limit: float | int = round(val - 0.01, 2)
        elif whole:
            limit = int(val) - 1
        else:
            limit = int(val)
        return limit if limit > 0 else None

    def roof_middle(self, lo: float, hi: float) -> float | int:
        cfg = self.roof_specification() or {}
        if not (cfg.get("int") or (float(lo).is_integer() and float(hi).is_integer())):
            return round((float(lo) + float(hi)) / 2, 2)
        return (int(lo) + int(hi)) // 2

    def roof_params(self, cfg: dict[str, Any], shelf_name: str, ceil_val: float | int) -> dict[str, Any]:
        as_int = cfg.get("int") or float(ceil_val).is_integer()
        limit = int(ceil_val) if as_int else ceil_val
        return {**cfg["tmpl"], cfg["shelf"]: shelf_name, cfg["bound"]: limit}

    def catalog_wares(self) -> list[int | None]:
        count = len(self.needs.items())
        return list(range(count)) if count else [None]

    def roofline_prepared(self, pid: int | None) -> bool:
        """CG3: product pid may be listed below its pick: a cheapest rule applies, a price-filter listing was learned, pid's
        category label has been seen on rows, and pid has not been listed this way yet."""
        if pid is None and self.needs.items():
            return False      # several products: a listing for an unknown product names none of them
        if self.needs.priority() != "cheapest" or pid in vars(self).get("ceil_walked", set()):
            return False
        if self.roof_specification() is None or self.sheet_param() is None:
            return False
        return bool(self.unprocessed_tag(self.traverse_landing(pid)))

    def roof_intent(self, pid: int | None) -> float | None:
        """CG3: the price a cheaper listing of product pid has to beat: the lower of its pick and a held put's listing; v605f
        FG: with several products, neither of those, and an amount bound among pid's requirements, the cheapest in-stock row
        inside the budget seen on its category."""
        selected = self.traverse_choose(pid)
        retained = (vars(self).get("ceil_aims") or {}).get(pid)
        price_list = [selected[1]] if selected else []
        if whether_numeric(retained):
            price_list.append(float(retained))
        if not price_list and self.boundary_requirement(pid):
            cats = self.traverse_landing(pid)

            def live_read(lid: str) -> bool:
                return (self.rack_familiar(lid) in cats and whether_numeric(self.vault.price(lid)) and lid not in self.defunct_lineitems()
                        and lid not in self.rejected_lineitems and self.final_inventory(lid) is not False)
            unread = [c for c, _ in self.lowcost_unperused(self.landing_tablerows(cats), pid)]
            seen = unread + [float(self.vault.price(lid)) for lid in self.vault.read if live_read(lid)]
            price_list = [min(seen)] if seen else []
        return min(price_list) if price_list else None

    def boundary_requirement(self, pid: int | None) -> bool:
        """v605f FG, several products: whether a requirement of product pid still in force is an amount bound (a figure with
        the wording of a limit), which no keyword can carry."""
        if pid is None or not self.needs.items():
            return False

        def is_bound(rq: Any) -> bool:
            hd, _, vtext = str(rq).partition(":")
            return self.ware_belonging(hd) == pid and bool(re.search(r"\d", vtext)) and bool(re.search(CROSSCHECK_BOUNDARY, vtext.lower()))
        return any(is_bound(rq) for rq in self.needs.traits_remaining())

    def landing_tablerows(self, cats: set[str]) -> set[str]:
        """v605f FG: the ids of rows seen on the categories `cats`."""
        hits: set[str] = set()

        def grab(node: dict[str, Any]) -> None:
            mark = self.tablerow_flag(node)
            if mark in self.vault.ids and self.rack_belonging(node) in cats:
                hits.add(mark)
        for entry in self.vault.items:
            traverse_mappings(entry, grab)
        return hits

    def roof_conflict(self, cats: set[str], ceil_val: float) -> bool:
        """CG3: True when a row already seen in the categories `cats` can be bought and costs no more than `ceil_val`
        (a listing filtered by that cap would then never come back empty)."""
        hits: list[bool] = []

        def probe(node: dict[str, Any]) -> None:
            if hits or self.rack_belonging(node) not in cats:
                return
            if not any(isinstance(vv, str) and vv in self.vault.ids and "id" in str(slot).lower() for slot, vv in node.items()):
                return
            amt = self.tablerow_expense(node)
            if amt is not None and amt <= ceil_val and self.inventory_condition(node) is True:
                hits.append(True)
        for entry in self.vault.items:
            traverse_mappings(entry, probe)
        return bool(hits)

    def lowcost_unperused(self, pool_ids: Any, pid: int | None, beneath: float | None = None) -> list[tuple[float, str]]:
        """CG: (price, id) pairs for the rows `pool_ids` that are still unread: in stock as last seen, inside the budget for
        product pid's units at the last price, not in a refused cart, and under `below` when given; cheapest first."""
        wanted = set(pool_ids)
        if not wanted:
            return []
        lines = self.needs.items()
        if pid is not None and pid < len(lines):
            qty = lines[pid]["quantity"]
        else:
            qty = self.needs.quantity()
        spend_lim = self.needs.outlay_lid
        soldmap: dict[str, bool] = {}

        def mark_stock(node: dict[str, Any]) -> None:
            if not any(isinstance(vv, str) and vv in wanted and "id" in str(slot).lower() for slot, vv in node.items()):
                return
            stock_flag = self.inventory_condition(node)
            if stock_flag is not None:
                soldmap[self.tablerow_flag(node)] = not stock_flag
        for entry in self.vault.items:
            traverse_mappings(entry, mark_stock)
        budgeted = whether_numeric(spend_lim) and whether_numeric(qty)
        picks = []
        for lid in wanted:
            amount = self.vault.price(lid)
            if lid in self.vault.read or lid in self.rejected_lineitems or soldmap.get(lid) or not whether_numeric(amount):
                continue
            over_budget = budgeted and amount * qty > spend_lim
            if over_budget or (beneath is not None and amount >= beneath):
                continue
            picks.append((float(amount), lid))
        picks.sort()
        return picks

    def whole_terms(self) -> bool:
        """CG: whether this store's keyword search has been learned to need every word (a store seen reading only its
        leading words, and never seen taking any of them, counts too, just as in search_note)."""
        if self.word_kind == "all":
            return True
        return bool(self.foremost_terms) and self.word_kind != "any"

    def appraised_uids(self, pid: int | None) -> set[str]:
        """CG: rows on product pid's category shown by its id-ordered searches led by a required value, in a store whose
        keyword search needs every word (so each such row holds those words somewhere in its record)."""
        if not self.whole_terms():
            return set()
        cats = self.traverse_landing(pid)
        hits: set[str] = set()
        for pager in vars(self).get("page_book", {}).values():
            if pager.get("id_order") is not True or self.catalog_advert(pager["args"]) or not self.traverse_conveys(pager["args"], pid):
                continue
            hits.update(lid for lid, shelf in (pager.get("row_shelf") or {}).items() if shelf in cats)
        return hits

    def appraised_fulfilled(self, pid: int | None) -> bool:
        """v605f FH: in an all-words store, whether some id-ordered search led by a required value of product pid (one whose
        requirement has a single alternative) was read to the end and showed every match, so a row outside it lacks
        that value."""
        try:
            if not self.whole_terms():
                return False
            cats = self.traverse_landing(pid)
            multi = bool(self.needs.items()) and pid is not None
            single: set[int] = set()
            for idx, rq in enumerate(self.needs.traits_remaining()):
                hd, _, rest = rq.partition(":")
                if multi and self.ware_belonging(hd) not in (pid, None):
                    continue
                vtext = re.sub(r"\((?:stated|revealed[^)]*)\)", "", rest).lower()
                options = [p for p in re.split(r"[,;/]|\bor\b|\beither\b|\bnor\b", vtext) if p.strip()]
                if len(options) == 1:
                    single.add(idx)
            for pager in vars(self).get("page_book", {}).values():
                shelf_set = pager.get("row_shelf") or {}
                finished = (pager.get("id_order") is True and not self.catalog_advert(pager["args"]) and pager.get("first")
                            and self.subsequent_sheet(pager) is None and len(shelf_set) >= pager["total"])
                if finished and any(sh in cats for sh in shelf_set.values()) and self.traverse_conveys(pager["args"], pid) & single:
                    return True
            return False
        except Exception:
            return False

    def tablerow_conveys(self, lid: str, pid: int | None) -> bool:
        """CG3: whether a search row of `lid` writes, in its own text, every word of one required value of product pid
        (one alternative of a closed list)."""
        multi = bool(self.needs.items()) and pid is not None
        word_groups: list[list[str]] = []
        figures: list[tuple[str, str]] = []
        for rq in self.needs.traits_remaining():
            hd, _, rest = rq.partition(":")
            if multi and self.ware_belonging(hd) not in (pid, None):
                continue
            vtext = re.sub(r"\((?:stated|revealed[^)]*)\)", "", rest).lower()
            if reference_passage(vtext):
                continue
            for part in re.split(r"[,;/]|\bor\b|\beither\b|\bnor\b", vtext):
                figv = self.numeral_magnitude(part)
                if figv:
                    figures.append(figv)      # v605f FA: the figure written just before its unit ('4-piece', '4 pieces')
                    continue
                if self.magnitude_portion(part):
                    continue
                stems = [self.wordbase(wd) for wd in re.findall(r"[a-z0-9]{3,}", part)]
                if stems:
                    word_groups.append(stems)
        if not (word_groups or figures):
            return False
        chunks: list[str] = []

        def collect(node: dict[str, Any]) -> None:
            if any(isinstance(vv, str) and vv == lid and "id" in str(slot).lower() for slot, vv in node.items()):
                chunks.extend(str(vv).lower() for vv in node.values() if isinstance(vv, str))
        for entry in self.vault.items:
            traverse_mappings(entry, collect)
        body = " ".join(chunks)
        have = {self.wordbase(wd) for wd in re.findall(r"[a-z0-9]{3,}", body)}
        for figv, unit in figures:
            if re.search(r"(?<![\d.,])" + re.escape(figv) + r"\s*-?\s*" + re.escape(self.wordbase(unit)), body):
                return True
        return any(all(wd in have for wd in group) for group in word_groups)

    def valsweep_headroom(self) -> tuple[int, int]:
        """EB: (messages still left during the question burst, steps beyond order_due() after two per such message)."""
        if self.inquiring and self.role("message"):
            msgs_left = max(0, self.inquire_headroom() - self.letters_transmitted)
        else:
            msgs_left = 0
        return msgs_left, self.shifts_remaining() - self.purchase_owed() - 2 * msgs_left

    def valsweep_settings(self, pid: int | None) -> list[list[list[str]]]:
        """EB1: for each firm word value of product pid (code_values), in order, the query words of its alternatives: a closed
        list ('or', 'either', '/') through its first three alternatives, any other value through its first part ('new' out
        of 'new, boxed and never used'). Words of two or more characters are kept, minus link and comparison words and the
        property's own name words; whole figures of a name-like value stay ('windows 10'), a decimal figure is dropped
        (stores split it). No alternative comes from a reference, an exclusion, an amount bound, or an amount led by its
        figure."""

        def viable(wd: str, figv: Any, own: set[str]) -> Any:
            if not (len(wd) >= 2 or (figv and wd == figv[0])):
                return False
            if re.search(r"[.,]", wd) or wd in BOND_TERMS:
                return False
            return wd not in own or (figv and wd == figv[1])
        collected: list[list[list[str]]] = []
        for hd, vtext in self.engine_settings(pid):
            lowtxt = re.sub(r"\([^)]*\)", " ", str(vtext).lower().replace("’", "'")).strip()
            if not lowtxt or reference_passage(lowtxt):
                continue
            if not re.search(r"\bor\b|\beither\b|/", lowtxt):
                segs = [re.split(r"[,;&+]|\band\b", lowtxt)[0]]
            else:
                segs = [p for p in re.split(r"[,;/]|\bor\b|\beither\b", lowtxt) if p.strip()][:3]
            own = set(re.findall(r"[a-z0-9]+", str(hd).lower()))
            choices: list[list[str]] = []
            for piece in segs:
                figv = self.numeral_magnitude(piece)
                if re.search(OMIT_TERMS, piece):
                    continue
                if self.magnitude_portion(piece) and not figv:
                    continue
                stripped = re.sub(r"\b(?:" + SETTING_BONDS + r")\b", " ", re.sub(r"'s\b", "", piece))
                # v605f FA: an amount led by a whole figure keeps both the figure and its unit word ('4 pieces')
                kept = [wd for wd in re.findall(r"\d+(?:[.,]\d+)+|[a-z0-9]+", stripped) if viable(wd, figv, own)]
                if kept and kept not in choices:
                    choices.append(kept)
            if choices and choices not in collected:
                collected.append(choices)
        return collected

    def valsweep_terms(self, pid: int | None, shelf_name: str) -> list[str]:
        """EB1: the words of product pid's category label as rows spell it ('&', link words and a possessive 's dropped). Once a
        composed search has shown the store does not index the label: the label words found in at least half of pid's row
        titles, failing that the request's kind word for pid."""
        cleaned = re.sub(r"'s\b", "", shelf_name.lower().replace("’", "'"))
        wlist = list(dict.fromkeys(wd for wd in re.findall(r"[a-z0-9]+", cleaned) if len(wd) >= 2 and wd not in BOND_TERMS))
        if vars(self).get("label_indexed") is not False:
            return wlist
        cats = self.traverse_landing(pid)
        title_sets: list[set[str]] = []

        def take_title(node: dict[str, Any]) -> None:
            if self.rack_belonging(node) in cats and self.tablerow_flag(node) in self.vault.ids:
                title_sets.append({self.wordbase(wd) for wd in re.findall(r"[a-z0-9]+", self.heading_belonging(node).lower())})
        for entry in self.vault.items:
            traverse_mappings(entry, take_title)
        if title_sets:
            common = [wd for wd in wlist if 2 * sum(1 for t in title_sets if self.wordbase(wd) in t) >= len(title_sets)]
            if common:
                return common
        lines = self.needs.items()
        if lines and pid is not None and pid < len(lines):
            product = str(lines[pid]["product"])
        else:
            product = str(self.needs.perusal.get("product") or "")
        kind = [wd for wd in re.findall(r"[a-z0-9]+", product.lower()) if len(wd) > 2 and wd not in BOND_TERMS]
        return kind[-1:]

    def valsweep_searchterms(self, pid: int | None, shelf_name: str, max_chars: Any) -> tuple[list[str], list[str]]:
        """EB1: product pid's composed searches as (forms with all values, forms with the first value). With two or more
        values: form (i) holds every firm value plus the category label's words, form (ii) the first value plus the label's
        words; with one value only its forms (first list). Under a cheapest rule a closed list yields one form per
        alternative (the first closed list among the values: the fits of every alternative are needed), otherwise its first
        alternative. Values lead, so a store reading only its first words still reads them; trimmed to the words the store
        reads (once learned), else to the schema's length."""
        value_sets = self.valsweep_settings(pid)
        if not value_sets:
            return [], []
        label_words = self.valsweep_terms(pid, shelf_name)
        every_alt = self.needs.priority() == "cheapest"

        def compose(parts: list[list[str]]) -> str:
            read_by: set[str] = set()
            picked: list[str] = []
            for wd in [wd for p in parts for wd in p] + label_words:
                sh = self.wordbase(wd)
                if sh not in read_by:
                    read_by.add(sh)
                    picked.append(wd)
            if self.foremost_terms:
                picked = picked[: self.foremost_terms]
            txt = " ".join(picked)
            if whether_numeric(max_chars):
                while " " in txt and len(txt) > max_chars:
                    txt = txt.rsplit(" ", 1)[0]
                txt = txt[: int(max_chars)]
            return txt

        def options(vv: list[list[str]]) -> list[list[str]]:
            return vv if every_alt else vv[:1]

        def unique(variants: list[str]) -> list[str]:
            return list(dict.fromkeys(f for f in variants if f))
        singles = [compose([lo]) for lo in options(value_sets[0])]
        if len(value_sets) < 2:
            return unique(singles), []
        j = next((pos for pos, vv in enumerate(value_sets) if len(options(vv)) > 1), 0)
        combined = [compose([lo if pos == j else vv[0] for pos, vv in enumerate(value_sets)]) for lo in options(value_sets[j])]
        return unique(combined), unique(singles)

    def valsweep_subsequent(self, full_forms: list[str], lone_forms: list[str], text_key: str, fmode: dict[str, Any]) -> str | None:
        """EB1: the composed form to send next: an all-values form not yet sent; once all of them came back empty, a
        first-value form not yet sent; '' when a sent form returned rows (or no form exists), None when every form came
        back empty."""
        for variants in (full_forms, lone_forms):
            if not variants:
                continue
            totals = [self.valsweep_observed(layout, text_key, fmode) for layout in variants]
            if None in totals:
                return variants[totals.index(None)]
            if any(totals):
                return ""
        if full_forms or lone_forms:
            return None
        return ""

    def valsweep_regime(self, cargs: dict[str, Any]) -> tuple:
        """EB: the mode of a search call: its choice arguments with at most four options (a category or another long
        choice list excluded)."""
        schema = self.structures.get(self.role("search") or "") or {}
        props = schema.get("properties")
        if not isinstance(props, dict):
            props = {}

        def short_choice(name: str) -> bool:
            cfg = props.get(name)
            return isinstance(cfg, dict) and isinstance(cfg.get("enum"), list) and len(cfg["enum"]) <= 4
        return tuple(sorted((name, val) for name, val in cargs.items() if isinstance(val, str) and short_choice(name)))

    def valsweep_group(self, txt: str) -> frozenset:
        """EB: the stems of a query's words as a set (only the words the store reads, once that is learned)."""
        tokens = re.findall(r"[a-z0-9]+", str(txt).lower())
        return frozenset(self.wordbase(wd) for wd in tokens[: self.foremost_terms or None])

    def valsweep_observed(self, txt: str, text_key: str, fmode: dict[str, Any]) -> int | None:
        """EB1: the total of an already-sent search in the same mode whose text argument has the words of `txt`
        (vl_set); None when no such search was sent."""
        target = self.valsweep_group(txt)
        for sent, n_total in (vars(self).get("ceil_seen") or {}).items():
            try:
                cargs = json.loads(sent)
            except Exception:
                continue
            if not isinstance(cargs, dict) or not isinstance(cargs.get(text_key), str):
                continue
            if any(cargs.get(name) != val for name, val in fmode.items()):
                continue
            if self.valsweep_group(cargs[text_key]) == target:
                return int(n_total)
        return None

    def valsweep_catalogued(self, shelf_name: str) -> None:
        """EB1, after a composed search: the store does index the category label when at least two of the rows on it that
        the search returned carry a label word found in none of their other text fields."""
        if vars(self).get("label_indexed") is not None or not self.vault.items:
            return
        own = {self.wordbase(wd) for wd in re.findall(r"[a-z0-9]+", re.sub(r"'s\b", "", shelf_name.lower())) if len(wd) >= 2 and wd not in BOND_TERMS}
        shelf = shelf_name.strip().lower()
        hits = [0]

        def tally(node: dict[str, Any]) -> None:
            if self.tablerow_flag(node) not in self.vault.ids or self.rack_belonging(node) != shelf:
                return
            elsewhere: set[str] = set()
            for slot, vv in node.items():
                if isinstance(vv, str) and not re.search(r"categor|shelf|department|aisle", str(slot).lower()):
                    elsewhere.update(self.wordbase(wd) for wd in re.findall(r"[a-z0-9]+", vv.lower()))
            if own - elsewhere:
                hits[0] += 1
        traverse_mappings(self.vault.items[-1], tally, ceiling=3)
        if hits[0] >= 2:
            self.label_indexed = True

    def valsweep_backlog(self, pid: int | None, pool_ids: Any, keep_steps: int, cap_n: int) -> list[str]:
        """EB2/EB7: the cheapest unread rows among `pool_ids` (rows on product pid's category): in stock as last seen, inside the
        budget, and under a cheapest rule below pid's pick (CEIL_OPENS opens per product overall there); at most `cap_n`, and
        never eating into floor."""
        want_low = self.needs.priority() == "cheapest"
        selected = self.traverse_choose(pid) if want_low else None
        bt = self.bundle_instrument()
        open_count = vars(self).setdefault("valued_opens", {})
        headroom = ROOF_EXAMINES - open_count.get(pid, 0) if want_low else cap_n
        ordered = [lid for _, lid in self.lowcost_unperused(pool_ids, pid, selected[1] if selected else None)]
        to_open = ordered[:max(0, min(bt[2] if bt else cap_n, headroom))]

        def steps(q: list[str]) -> int:
            return 1 if bt and len(q) >= 2 else len(q)
        while to_open and steps(to_open) >= self.shifts_remaining() - keep_steps:
            to_open.pop()
        if want_low:
            open_count[pid] = open_count.get(pid, 0) + len(to_open)
        return to_open

    def valsweep_kickoff(self, to_open: list[str], homes: set[str], keep_steps: int, say_txt: str) -> None:
        """EB: opens the queued rows as scripted reads; the rank check of those categories will run once more."""
        self.standing_assessed = {done for done in self.standing_assessed if self.rack_familiar(done.split("|")[0]) not in homes}
        self.purchase_sanctioned = False
        self.open_queue, self.open_recheck, self.open_say = to_open, True, say_txt
        self.open_purpose = "walk"
        self.open_floor = keep_steps
        self.examine_titled()

    def valsweep_seek(self) -> bool:
        """EB1 (the first branch of value_pages, for every objective): once per set of firm word values of a product whose
        category label the rows show, its composed value + label search (vl_queries: form (i), then form (ii) when (i) came
        back empty or there is one value; when every form came back empty though the value alone had rows on its category,
        once more with the label's words found in the row titles instead), built on the schema-built keyword call
        (code_queries), while more than order_due() + 2 per remaining message + 4 steps are left; LABEL_TRIES per product.
        A search with the same words sent earlier counts. When one page shows every match: under a cheapest rule its
        cheapest unread rows are opened like the value pages' (several products: later, in vl_flush); otherwise (EB7),
        when it holds every value, its unread in-stock rows inside the budget, at most LABEL_OPENS."""
        if self.valsweep_headroom()[1] <= 4:
            return False
        tname = self.role("search")
        sch = self.structures.get(tname or "") or {}
        raw_props = sch.get("properties")
        props = raw_props if isinstance(raw_props, dict) else {}
        pinfo = self.sheet_param()
        tries = vars(self).setdefault("label_tries", {})
        settled = vars(self).setdefault("label_done", {})
        sets = vars(self).setdefault("label_sets", {})
        plist = self.catalog_wares()

        def keyword_call(pid: int | None) -> tuple[dict[str, Any] | None, str]:
            for call in self.engine_searchterms(pid):
                free = self.loose_passage(call)
                texts = [name for name, val in call.items() if isinstance(val, str) and val.strip() and val in free]
                if len(texts) == 1 and not self.catalog_advert(call):
                    return call, texts[0]
            return None, ""

        def ids_on(pager: dict[str, Any], cats: set[str]) -> list[str]:
            return [lid for lid, shelf in (pager.get("row_shelf") or {}).items() if shelf in cats]
        for pid in plist:
            if tries.get(pid, 0) >= TAG_ATTEMPTS:
                continue
            msig = json.dumps(self.valsweep_settings(pid))
            closed = settled.setdefault(str(pid), [])
            if msig in closed:
                continue
            cats = self.traverse_landing(pid)
            shelf_name = self.unprocessed_tag(cats)
            if not shelf_name:
                continue
            shape_tmpl, text_key = keyword_call(pid)
            if not shape_tmpl:
                closed.append(msig)
                continue
            fmode = {name: val for name, val in shape_tmpl.items() if isinstance(props.get(name), dict) and props[name].get("enum")}
            # a bare keyword search, without a price limit or any other filter, so it lines up with the actor's own searches
            needed = sch.get("required") or []
            shape_tmpl = {name: val for name, val in shape_tmpl.items() if name == text_key or name in fmode or name in needed}
            text_spec = props.get(text_key)
            max_chars = text_spec.get("maxLength") if isinstance(text_spec, dict) else None
            full_forms, lone_forms = self.valsweep_searchterms(pid, shelf_name, max_chars)
            bucket = sets.setdefault(str(pid), set())
            bucket.update(self.valsweep_group(q) for q in full_forms + lone_forms)
            qform = self.valsweep_subsequent(full_forms, lone_forms, text_key, fmode)
            if qform is None:
                qform = ""
                carried = any(self.traverse_conveys(wd["args"], pid) and any(sh in cats for sh in (wd.get("row_shelf") or {}).values())
                              for wd in vars(self).get("page_book", {}).values())
                if carried and vars(self).get("label_indexed") is None:
                    # every form came back empty although the value alone had rows on the category, so this store does not
                    # index the category label: its words found in the row titles (or else the kind word) replace it
                    self.label_indexed = False
                    full_forms, lone_forms = self.valsweep_searchterms(pid, shelf_name, max_chars)
                    bucket.update(self.valsweep_group(q) for q in full_forms + lone_forms)
                    qform = self.valsweep_subsequent(full_forms, lone_forms, text_key, fmode) or ""
            if not qform:
                closed.append(msig)
                continue
            cargs, defects = ParamAssess.carryout({**shape_tmpl, text_key: qform}, sch)
            if defects or not isinstance(cargs, dict):
                closed.append(msig)
                continue
            tries[pid] = tries.get(pid, 0) + 1
            plural = "s" if len(self.valsweep_settings(pid)) > 1 else ""
            self.traverse_invoke(cargs, f"Searching for the required value{plural} together with the category name as this store's rows "
                                 f"spell it ('{qform}'): keyword searches match listing values, titles and categories, so the "
                                 "result shows which of the category's listings have the value.")
            n_total = self.valsweep_observed(qform, text_key, fmode)
            self.valsweep_catalogued(shelf_name)
            slot = json.dumps({lo: vv for lo, vv in cargs.items() if not pinfo or lo != pinfo[0]}, sort_keys=True, default=str)
            pager = vars(self).get("page_book", {}).get(slot)
            every = False
            if pager:
                shelf_set = pager.get("row_shelf") or {}
                every = len(shelf_set) >= pager["total"] and any(sh in cats for sh in shelf_set.values())
            keep_steps = self.purchase_owed() + 2
            to_open: list[str] = []
            if n_total is None:
                msg = ""
            elif not n_total:
                msg = f"[value search] '{qform}': nothing matched."
            else:
                msg = f"[value search] '{qform}': {n_total} match(es)" + (" - all of them appear here." if every else ".")
            want_low = self.needs.priority() == "cheapest"
            if every and want_low and pinfo and pager.get("id_order") is not False:
                vars(self).setdefault("valued", set()).add(slot)
                vars(self).setdefault("valued_keys", {}).setdefault(str(pid), []).append(slot)
                if len(plist) > 1:
                    vars(self).setdefault("valued_wait", {})[pid] = True
                else:
                    to_open = self.valsweep_backlog(pid, ids_on(pager, cats), keep_steps, ROOF_EXAMINES)
            elif every and not want_low and qform in (full_forms or lone_forms):
                to_open = self.valsweep_backlog(pid, ids_on(pager, cats), keep_steps, TAG_EXAMINES)
            if to_open:
                under = " and under the pick's price" if self.traverse_choose(pid) and want_low else ""
                listed = ", ".join(f"{rid} ({self.vault.price(rid):g})" for rid in to_open)
                msg += (f" Next, its unread rows that are in stock, within the budget and on the requested category{under} "
                         f"get opened, cheapest first: {listed}.")
            if msg and self.converse and self.converse[-1].get("role") == "tool":
                tail = "" if to_open else "\n" + self.progress_lineitem()
                self.converse[-1]["content"] = str(self.converse[-1]["content"]) + "\n" + msg + tail
            self.journal.append(dict(kind="label_search", product=pid, query=qform, total=n_total, every=every, opened=to_open))
            if to_open:
                self.valsweep_kickoff(to_open, cats, keep_steps, "Opening that search's cheapest unread rows: its matches include every "
                                                   "listing in the category that has the required values.")
            return True
        return False

    def valsweep_drain(self, keep_steps: int) -> bool:
        """EB3: several products under a cheapest rule, once no product's value search is still waiting to be read: opens
        the cheapest unread in-stock rows inside the budget and under each pick that its read value searches turned up,
        one or two per product (two while the steps allow), in turn."""
        wait = vars(self).setdefault("valued_wait", {})
        pids = [pid for pid, flag in wait.items() if flag]
        wait.clear()
        if not pids:
            return False
        pagebook = vars(self).get("page_book", {})
        keys = vars(self).get("valued_keys") or {}
        open_count = vars(self).setdefault("valued_opens", {})
        bt = self.bundle_instrument()
        per_pid = 2 if self.shifts_remaining() - keep_steps - 1 >= 2 * len(pids) else 1

        def pool_for(pid: int | None) -> list[str]:
            cats = self.traverse_landing(pid)
            pool_ids = {lid for slot in keys.get(str(pid), []) if slot in pagebook and self.traverse_conveys(pagebook[slot]["args"], pid)
                   for lid, shelf in (pagebook[slot].get("row_shelf") or {}).items() if shelf in cats}
            selected = self.traverse_choose(pid)
            headroom = max(0, min(per_pid, ROOF_EXAMINES - open_count.get(pid, 0)))
            return [lid for _, lid in self.lowcost_unperused(pool_ids, pid, selected[1] if selected else None)][:headroom]
        pools = [pool_for(pid) for pid in pids]
        owner_of: dict[str, Any] = {}
        to_open: list[str] = []
        cap_n = bt[2] if bt else per_pid * len(pids)
        while any(pools) and len(to_open) < cap_n:
            for pid, pool in zip(pids, pools):
                while pool and pool[0] in to_open:
                    del pool[0]
                if pool and len(to_open) < cap_n:
                    hd = pool.pop(0)
                    owner_of[hd] = pid
                    to_open.append(hd)
        while to_open and (1 if bt and len(to_open) >= 2 else len(to_open)) >= self.shifts_remaining() - keep_steps:
            to_open.pop()
        if not to_open:
            return False
        for lid in to_open:
            open_count[owner_of[lid]] = open_count.get(owner_of[lid], 0) + 1
        homes = set().union(*(self.traverse_landing(pid) for pid in pids))
        self.valsweep_kickoff(to_open, homes, keep_steps, "The shopper wants the cheapest: opening the cheapest unread rows that each "
                                            "product's value search turned up.")
        listed = ", ".join(f"{rid} ({self.vault.price(rid):g})" for rid in to_open)
        recap = ("[value searches read] Opened, per product, the cheapest unread rows its value search turned up that are in "
                   f"stock, within the budget and under its pick: {listed}. Pick for each product the cheapest listing whose "
                   "own fields satisfy every requirement currently in force; choosing a pricier fit requires opening and "
                   "rejecting every cheaper candidate first.")
        if self.converse and self.converse[-1].get("role") == "tool":
            self.converse[-1]["content"] = str(self.converse[-1]["content"]) + "\n" + recap
        self.journal.append(dict(kind="value_flush", products=pids, opened=to_open))
        return True

    def ware_stages(self) -> str:
        """EB3, several products under a cheapest rule: for each product the value-search pages the code read and whether
        it holds a pick (a cart line or a listing that passed the item check), plus the first product lacking one."""
        try:
            rowset = self.needs.items()
            if self.needs.priority() != "cheapest" or len(rowset) < 2:
                return ""
            read_by = vars(self).get("valued_by") or {}
            bits: list[str] = []
            nextp = ""
            for pid, rec_row in enumerate(rowset):
                selected = self.traverse_choose(pid)
                bits.append(f"{rec_row['product']}: value pages read {read_by.get(pid, 0)}, has a pick: {'yes' if selected else 'no'}")
                if not nextp and not selected:
                    nextp = str(rec_row["product"])
            tail = f"; up next: {nextp}" if nextp else ""
            return "\n[product steps] " + "; ".join(bits) + tail + "."
        except Exception:
            return ""

    def value_pages(self) -> bool:
        """CG2 + EB1-EB3: first the composed value + label search (vl_search, for every objective). Then, under a cheapest
        rule in a store that lists results in id order and whose keyword search has not been seen taking any word: a search
        led by a required value of a product, with a next page and a total the visible pages can show (while the AND rule
        is unknown only such a search, so every listing it matched gets seen; a composed search deeper than the depth is
        read to its reachable end), is read to the end by the code (max(VALUE_PAGES, visible pages - 1) more pages per
        product, max(VALUE_TOTAL, VALUE_TOTAL_PER per product) overall; several products: the one with the fewest pages
        read goes first, each inside its share of the steps). Pages wait for the question burst unless two or fewer would
        be read or the steps run down to them. Afterwards the cheapest unread in-stock rows inside the budget on the
        product's category (under its pick) are opened (several products: once no value search waits, vl_flush). Stops
        at order_due() + 2 steps."""
        try:
            if any(rr.get("status") == "open" for rr in vars(self).get("refs") or []) and self.reference_headroom():
                return False      # v605d M5: an open reference's resolver runs first; the pages wait until its room is used up
            if (self.purchases_attempted or self.rejection_num or self.protocol.done or self.protocol.crashed
                    or self.protocol.surplus_alt() < WRAPUP + 60 or not self.role("search")):
                return False
            if self.valsweep_seek():
                return True
            if self.needs.priority() != "cheapest" or self.word_kind == "any":
                return False
            if not self.sheet_param():
                return False
            pagebook = vars(self).get("page_book", {})
            done = vars(self).setdefault("valued", set())
            read_by = vars(self).setdefault("valued_by", {})
            open_count = vars(self).setdefault("valued_opens", {})
            wait = vars(self).setdefault("valued_wait", {})
            sets = vars(self).get("label_sets") or {}
            vis_depth = self.apparent_reach()
            psize = max(1, self.sheet_dimension())
            per_pid = SETTING_SHEETS if not vis_depth else max(SETTING_SHEETS, -(-vis_depth // psize) - 1)
            plist = self.catalog_wares()
            multi = len(plist) > 1
            spend_lim = max(SETTING_AGGREGATE, SETTING_AGGREGATE_EACH * len(plist))
            if self.bundle_instrument():
                open_cost = 1
            else:
                open_cost = 2 if multi else ROOF_EXAMINES
            keep_steps = self.purchase_owed() + 2
            strict = self.whole_terms()
            msgs_left, headroom = self.valsweep_headroom()
            unstarted = [pid for pid in plist if self.traverse_landing(pid) and not read_by.get(pid)]
            share = max(4, headroom // max(1, len(unstarted)))
            options: list[tuple] = []
            pending = False
            for pos, pid in enumerate(plist):
                cats = self.traverse_landing(pid)
                if not cats:
                    continue
                composed_sets = sets.get(str(pid)) or set()
                scoped = bool(self.needs.items()) and pid is not None
                values = {self.wordbase(wd) for rq in self.needs.traits_remaining()
                          if not scoped or self.ware_belonging(rq.partition(":")[0]) in (pid, None)
                          for wd in re.findall(r"[a-z0-9]{2,}", rq.partition(":")[2].lower())}
                shelf_name = {self.wordbase(wd) for sh in cats for wd in re.findall(r"[a-z0-9]{2,}", sh)}
                vocab = values | shelf_name
                finished = [(self.valsweep_group(" ".join(self.loose_passage(wd["args"]))), self.traverse_conveys(wd["args"], pid, alts=True),
                             self.valsweep_regime(wd["args"]))
                            for wd in pagebook.values() if wd.get("id_order") is True and wd.get("first") and self.subsequent_sheet(wd) is None
                            and not self.catalog_advert(wd["args"]) and len(wd.get("row_shelf") or {}) >= wd["total"]]
                shortlist: list[tuple] = []
                for slot, pager in pagebook.items():
                    shelf_set = pager.get("row_shelf") or {}
                    if (slot in done or pager.get("id_order") is not True or not pager.get("first") or self.subsequent_sheet(pager) is None
                            or self.catalog_advert(pager["args"]) or not self.traverse_conveys(pager["args"], pid)
                            or not any(sh in cats for sh in shelf_set.values())):
                        continue
                    wlist = self.valsweep_group(" ".join(self.loose_passage(pager["args"])))
                    composed = wlist in composed_sets
                    too_deep = vis_depth is not None and pager["total"] > vis_depth
                    if strict:
                        if too_deep and not composed:
                            continue
                    elif vis_depth is None or too_deep:
                        continue      # EB2, rule unknown: only a search whose every match fits on the visible pages
                    pairs = self.traverse_conveys(pager["args"], pid, alts=True)
                    how = self.valsweep_regime(pager["args"])
                    if strict and any(m == how and (seen <= wlist or (seen <= vocab and pairs <= n_read)) for seen, n_read, m in finished):
                        continue      # a search already read to its end showing every match holds every fit this one could show
                    reach = pager["total"] if vis_depth is None else min(pager["total"], vis_depth)
                    pages = -(-reach // max(1, pager["size"])) - len(pager["read"])
                    allowed = min(per_pid - read_by.get(pid, 0), spend_lim - sum(read_by.values()))
                    if multi:
                        reserve = 0 if open_count.get(pid) else open_cost
                        allowed = min(allowed, share - read_by.get(pid, 0) - open_count.get(pid, 0) - reserve)
                    if pages <= allowed:
                        take = pages
                    else:
                        take = allowed if composed else 0
                    if take <= 0 or self.shifts_remaining() - take - open_cost <= keep_steps:
                        continue
                    if (multi or take > 2) and msgs_left and headroom > take + open_cost + 4:
                        # EB2: the question burst goes first, since a reveal narrows the search before its pages are
                        # spent; v605f FG: with several products this holds for any number of pages
                        pending = True
                        continue
                    shortlist.append((take, -self.traverse_rating(pager["args"], pid), slot, wlist))
                if strict:
                    # EB4: when two candidates' words nest, the longer one lists only rows the shorter lists: if its extra
                    # words are values of pid or words of pid's category label the shorter is dropped, otherwise the longer
                    def beaten(c: tuple) -> bool:
                        for o in shortlist:
                            if o[3] < c[3] and not (c[3] - o[3]) <= vocab:
                                return True
                            if c[3] < o[3] and (o[3] - c[3]) <= vocab:
                                return True
                        return False
                    shortlist = [c for c in shortlist if not beaten(c)]
                options.extend((read_by.get(pid, 0), take, score, pos, slot, pid) for take, score, slot, _ in shortlist)
            if options:
                best = min(options, key=lambda o: o[:5])
                done.add(best[4])
                return self.peruse_setting_seek(best[5], best[4], keep_steps, best[1], multi)
            if wait and not pending and not (msgs_left and headroom > open_cost * len(wait) + 4):
                return self.valsweep_drain(keep_steps)     # v605f FG: the opens also wait for the question burst
            return False
        except Exception as exc:
            self.journal.append(dict(kind="value_pages_error", error=f"{type(exc).__name__}: {exc}"[:300]))
            return False

    def peruse_setting_seek(self, pid: int | None, slot: str, keep_steps: int, cap_n: int = SETTING_SHEETS, defer: bool = False) -> bool:
        """CG2: reads the remainder of one value-led search (scripted, one page per step, `cap_n` pages at most), then opens
        its cheapest unread rows (EB3 defer, several products: once no product's value search waits - vl_flush)."""
        pinfo = self.sheet_param()
        pager = vars(self).get("page_book", {}).get(slot)
        if not (pinfo and pager):
            return False
        query = " ".join(self.loose_passage(pager["args"]))
        read_by = vars(self).setdefault("valued_by", {})
        strict = self.whole_terms()
        if strict:
            say_txt = ("The shopper wants the cheapest, and this store's keyword search requires every word and returns results in a "
                   f"fixed id order rather than by price: reading the next page of '{query}' so every listing with these words "
                   "gets seen.")
        else:
            say_txt = ("The shopper wants the cheapest, and this store returns search results in a fixed id order rather than by "
                   f"price: reading the next page of '{query}' so every listing it matched gets seen.")
        n_read = 0
        tried = False

        def can_go() -> bool:
            return (self.subsequent_sheet(pager) is not None and n_read < cap_n and self.shifts_remaining() > keep_steps + 1 and not self.protocol.done
                    and not self.protocol.crashed and self.protocol.surplus_alt() >= WRAPUP + 30)
        while can_go():
            tried = True
            read_by[pid] = read_by.get(pid, 0) + 1
            nextp = self.subsequent_sheet(pager)
            if not self.traverse_invoke({**pager["args"], pinfo[0]: nextp}, say_txt):
                break
            n_read += 1
            if self.subsequent_sheet(pager) == nextp:
                break      # an empty page: the search has nothing further
        if n_read == 0:
            return tried
        vars(self).setdefault("valued_keys", {}).setdefault(str(pid), []).append(slot)
        seen = len(pager.get("row_shelf") or {})
        exhaustive = self.subsequent_sheet(pager) is None and seen >= pager["total"]
        cats = self.traverse_landing(pid)
        if exhaustive:
            extra = " - this includes every listing with these words" if strict else " - this includes every listing it matched"
        elif self.subsequent_sheet(pager) is None:
            extra = " - the remainder is out of reach, so a cheaper fit could sit beyond the reachable rows"
        else:
            extra = ""
        hd = f"[value search read] '{query}': {seen} of {pager['total']} matches now seen{extra}. "
        closing = ("Pick the cheapest listing whose own fields satisfy every requirement currently in force; choosing a pricier "
                   "fit requires opening and rejecting every cheaper candidate first.")
        to_open: list[str] = []
        if defer:
            vars(self).setdefault("valued_wait", {})[pid] = True
            recap = (hd + "Its cheapest unread rows that are in stock and within the budget get opened once every product's "
                       "value search has been read. " + closing)
        else:
            selected = self.traverse_choose(pid)
            to_open = self.valsweep_backlog(pid, [rid for rid, sh in (pager.get("row_shelf") or {}).items() if sh in cats], keep_steps, ROOF_EXAMINES)
            if to_open:
                listed = ", ".join(f"{rid} ({self.vault.price(rid):g})" for rid in to_open)
                body = ("Next, its cheapest unread rows that are in stock, within the budget and on the requested category"
                        + (" and under the pick's price" if selected else "") + f" get opened: {listed}. ")
            else:
                body = "None of them is an unread in-stock row within the budget on the requested category. "
            recap = hd + body + closing
            if to_open:
                self.valsweep_kickoff(to_open, cats, keep_steps, "The shopper wants the cheapest: opening that search's cheapest unread rows.")
        if self.converse and self.converse[-1].get("role") == "tool":
            tail = "" if to_open else "\n" + self.progress_lineitem()
            self.converse[-1]["content"] = str(self.converse[-1]["content"]) + "\n" + recap + tail
        self.journal.append(dict(kind="value_pages", product=pid, query=query, pages=n_read, complete=exhaustive, opened=to_open,
                             defer=defer))
        return True

    def ceiling_walk(self, pid: int | None, keep_steps: int, repair: bool = False) -> tuple[str, list[str]] | None:
        """CG3: lists product pid's category below the price of its pick using the learned price filter. The limit starts one
        unit under the pick; while the total exceeds what the remaining calls can read (or the visible depth), the limit
        drops to the midpoint between the cheapest price seen and the limit, or further where an even spread of the total
        across the prices says the remaining pages would hold it (climbing back halfway when nothing is left); a total
        that fits is read to its end; CEIL_CALLS calls overall. Returns (report, rows to open: rows an all-words search
        led by a value also showed, then rows whose own text has a required value, then the rest, cheapest first,
        CEIL_OPENS at most). Marks pid as listed; a zero total while a cheaper in-stock row is known, or a row above the
        limit, retires the mode. Stops at order_due() + 2 steps."""
        cfg, pinfo = self.roof_specification(), self.sheet_param()
        cats = self.traverse_landing(pid)
        shelf_name = self.unprocessed_tag(cats)
        aim_price = self.roof_intent(pid)
        if aim_price is None or not (cfg and pinfo and shelf_name):
            return None
        vars(self).setdefault("ceil_walked", set()).add(pid)
        current = self.traverse_choose(pid)
        aimed_at = (float(aim_price), current[0] if current else "")
        vars(self).setdefault("ceil_aimed", {})[pid] = aimed_at     # v605f FF
        ceil_val = self.beneath(aim_price)
        if ceil_val is None:
            return None
        # v605f FF: a repair walk's listing calls halt two steps above the floor that its opens keep
        if repair:
            keep_steps += 2
        elif not self.purchases_attempted:
            keep_steps = max(keep_steps, self.purchase_owed() + 2)
        vis_depth = self.apparent_reach()
        pagebook = vars(self).setdefault("page_book", {})
        keys: list[str] = []
        n_calls = 0
        ceiling_hi = None
        exhaustive = ran = False
        last = ceil_val
        # v605f FG: with no pick yet (and no held put) the walk aims at the cheapest row seen
        what = "the pick" if current or whether_numeric((vars(self).get("ceil_aims") or {}).get(pid)) else "the cheapest row seen"
        say_txt = f"The shopper wants the cheapest: using the store's price filter to list '{shelf_name}' under {what} ({aim_price:g})."

        def can_step() -> bool:
            return (self.shifts_remaining() > keep_steps + 1 and not self.protocol.done and not self.protocol.crashed
                    and self.protocol.surplus_alt() >= WRAPUP + 30)
        while n_calls < ROOF_INVOKES and can_step():
            cargs = self.roof_params(cfg, shelf_name, ceil_val)
            slot = json.dumps(cargs, sort_keys=True, default=str)
            vars(self).setdefault("ceil_keys", set()).add(slot)
            n_calls += 1
            # v605f FF: an identical call already booked (the EB8 seed or one of the actor's) is reused at no step cost
            enrolled = slot in pagebook and bool(pagebook[slot].get("first")) and slot in (vars(self).get("ceil_seen") or {})
            if not enrolled and not self.traverse_invoke(cargs, say_txt):
                break      # EB8: when the seed listing was exactly this call, its page is reused and no step is spent
            ran, last = True, ceil_val
            keys.append(slot)
            pager = pagebook.get(slot)
            if pager:
                shelf_ids = pager.get("row_shelf") or {}
                if any(whether_numeric(self.vault.price(rid)) and self.vault.price(rid) > ceil_val for rid in shelf_ids):
                    vars(self).setdefault("ceil_bad", set()).add(cfg["sig"])
            if self.roof_specification() is None:
                break      # a row over the limit: this mode ignores it
            n_total = vars(self).get("ceil_seen", {}).get(slot)
            if n_total is None:
                break      # no listing came back (a fault): no conclusion is drawn
            if not n_total or not pager:
                if self.roof_conflict(cats, ceil_val):
                    vars(self).setdefault("ceil_bad", set()).add(cfg["sig"])
                    break
                if ceiling_hi is None:
                    exhaustive = True      # the category has nothing priced under the pick
                    break
                mid = self.roof_middle(ceil_val, ceiling_hi)
                if not ceil_val < mid < ceiling_hi:
                    break
                ceil_val = mid
                continue
            pages = -(-n_total // max(1, pager["size"]))
            if (vis_depth is None or n_total <= vis_depth) and pages - len(pager["read"]) <= ROOF_INVOKES - n_calls:
                while self.subsequent_sheet(pager) is not None and n_calls < ROOF_INVOKES and can_step():
                    n_calls += 1
                    nextp = self.subsequent_sheet(pager)
                    if not self.traverse_invoke({**cargs, pinfo[0]: nextp}, "Reading the following page of that price-filtered listing.")\
                            or self.subsequent_sheet(pager) == nextp:
                        break
                exhaustive = self.subsequent_sheet(pager) is None
                break
            ceiling_hi = ceil_val
            price_vals = [float(c) for c in (self.vault.price(rid) for rid in (pager.get("row_shelf") or {})) if whether_numeric(c)]
            lowtxt = min(price_vals) if price_vals else 0.0
            # lower to the midpoint of the cheapest price seen and the limit, or further: down to where the total, spread
            # evenly over the prices, would fill the pages the remaining calls can read
            fill = max(1, ROOF_INVOKES - n_calls) * max(1, pager["size"])
            spread = lowtxt + (ceil_val - lowtxt) * 1.8 * fill / max(n_total, 1)
            nb = min(self.roof_middle(lowtxt, ceil_val), self.roof_middle(lowtxt, spread))
            if not nb < ceil_val:
                break
            ceil_val = nb
        if not ran:
            return None
        pool_ids: set[str] = set()
        for booked_key in keys:
            pool_ids.update(rid for rid, sh in (pagebook.get(booked_key, {}).get("row_shelf") or {}).items() if sh in cats)
        shortlist = [rid for _, rid in self.lowcost_unperused(pool_ids, pid, aim_price)]
        vset = self.appraised_uids(pid)
        lead_ids: list[str] = []
        rest: list[str] = []
        for rid in shortlist:
            (lead_ids if rid in vset else rest).append(rid)
        text_hits = [rid for rid in rest if self.tablerow_conveys(rid, pid)]
        others = [] if self.appraised_fulfilled(pid) else [rid for rid in rest if rid not in text_hits]     # v605f FH
        ordered = lead_ids + text_hits + others
        if not exhaustive:
            report_txt = (f"the price filter on '{shelf_name}' under {aim_price:g}: {len(pool_ids)} of the cheaper listings seen so far ({len(shortlist)} "
                      "unread, in stock and within the budget)")
        elif pool_ids:
            report_txt = (f"the price filter on '{shelf_name}': all listings priced up to {last:g} seen ({len(pool_ids)}; {len(shortlist)} of them "
                      "unread, in stock and within the budget)")
        else:
            report_txt = f"the price filter on '{shelf_name}' under {aim_price:g}: nothing in it is cheaper than {what}"
        self.journal.append(dict(kind="ceiling_walk", product=pid, label=shelf_name, aim=aim_price, calls=n_calls, complete=exhaustive,
                             rows=len(pool_ids), ranked=ordered[:ROOF_EXAMINES]))
        return report_txt, ordered[:ROOF_EXAMINES]

    def roofline_survey(self, tried: set) -> tuple[dict[str, Any] | None, bool]:
        """CG3, first source of the shelf sweep: (the next price-filter listing call, whether the sweep is proven done). The
        limit is one unit under the cheapest in-stock row inside the budget seen on the requested category; further pages
        are read while it shows nothing cheaper in stock; done once that listing is empty or read to the end. Single
        product only."""
        cfg, pinfo = self.roof_specification(), self.sheet_param()
        if not cfg or not pinfo or self.needs.items():
            return None, False
        rowset = self.rack_tablerows()
        shelf_set = self.rack_tags()
        shelf_name = self.unprocessed_tag(shelf_set)
        if not (rowset and shelf_name):
            return None, False
        ceil_val = self.beneath(min(amt for amt, _, _ in rowset.values()))
        if ceil_val is None:
            return None, True
        cargs = self.roof_params(cfg, shelf_name, ceil_val)
        slot = json.dumps(cargs, sort_keys=True, default=str)
        if slot not in tried:
            vars(self).setdefault("ceil_keys", set()).add(slot)
            return cargs, False
        pager = vars(self).get("page_book", {}).get(slot)
        if not pager:
            empty = vars(self).get("ceil_seen", {}).get(slot) == 0
            return None, empty and not self.roof_conflict(set(shelf_set), ceil_val)
        nextp = self.subsequent_sheet(pager)
        if nextp is None:
            return None, len(pager.get("row_shelf") or {}) >= pager["total"]
        more = {**cargs, pinfo[0]: nextp}
        if json.dumps(more, sort_keys=True, default=str) in tried:
            return None, False
        return more, False

    def setting_sentry(self, tool_called: str, sent_args: dict[str, Any]) -> str:
        """CG4: under a cheapest rule, in a store whose keyword search needs every word: a details read of an unread listing
        that no value-led search showed is held once per product and requirement version while rows those searches showed
        on its category, cheaper than both the pick and that listing, in stock and inside the budget, remain unread (those
        rows hold the value's words somewhere in their records). Not inside the reserve, not for a listing a check named,
        and not when the request names a value by pointing at a listing."""
        try:
            if tool_called != self.role("details") or self.needs.priority() != "cheapest" or not self.whole_terms() or self.among_buffer():
                return ""
            named = [vv for vv in sent_args.values() if isinstance(vv, str) and vv in self.vault.ids]
            if len(named) != 1:
                return ""
            lid = named[0]
            if lid in self.vault.read or lid in self.handcart.measures:
                return ""
            if lid in self.titled_examine() or vars(self).get("refs") or self.needs.perusal.get("references"):
                return ""      # a listing named by a check, or a request giving a value by pointing at a listing: no hold
            pid = self.ware_position(lid)
            if pid is None and self.needs.items():
                return ""
            mark = (pid, self.needs.iteration)
            already = vars(self).setdefault("vguard_told", set())
            if mark in already:
                return ""
            vset = self.appraised_uids(pid)
            if not vset or lid in vset:
                return ""
            selected = self.traverse_choose(pid)
            own_price = self.vault.price(lid)
            price_list = [selected[1]] if selected else []
            if whether_numeric(own_price):
                price_list.append(float(own_price))
            if not price_list:
                return ""
            rowset = self.lowcost_unperused(vset, pid, min(price_list))[:6]
            if not rowset:
                return ""
            already.add(mark)
            than = " than both it and the pick" if selected else " than it"
            listed = ", ".join(f"{rid} ({c:g})" for c, rid in rowset)
            return ("Not run (no step used). The shopper wants the cheapest and this store's keyword search requires every word, "
                    "so each row returned by a search led by a required value has those words somewhere in its record. "
                    f"{lid} was not among those rows, yet the following ones from them are cheaper{than} and still unread: "
                    f"{listed}. Open those first, lowest price first; repeat this call afterwards if you still need it.")
        except Exception as exc:
            self.journal.append(dict(kind="value_guard_error", error=f"{type(exc).__name__}: {exc}"[:300]))
            return ""


class Route:
    """Errand part: walk holds, screening calls, sending batches and taking in their results."""

    def heading_lineitem(self, pid: int | None, rec: dict[str, Any], template: tuple[dict[str, Any], str]) -> dict[str, Any] | None:
        """CG5: a ready search for the product line of a pick whose record names no maker: the first title word of three
        or more letters that is not a kind, link or value word, joined by the next title word when that is not one of
        those either (a model number, say), provided that search has not been run."""
        avoid = set(BOND_TERMS)
        for shelf in self.traverse_landing(pid):
            avoid.update(re.findall(r"[a-z]+", shelf))
        for rq in self.needs.traits_remaining():
            avoid.update(re.findall(r"[a-z0-9]+", rq.lower()))
        kinds = self.needs.items() or [{"product": str(self.needs.perusal.get("product") or "")}]
        for entry in kinds:
            avoid.update(re.findall(r"[a-z]+", str(entry.get("product") or "").lower()))
        stems = {self.wordbase(wd) for wd in avoid}

        def simple(word: str) -> bool:
            return word not in avoid and self.wordbase(word) not in stems
        tokens = re.findall(r"[a-z0-9]+", self.heading_belonging(rec).lower())
        for pos, tok in enumerate(tokens):
            if not re.fullmatch(r"[a-z]{3,}", tok) or not simple(tok):
                continue
            follow = tokens[pos + 1] if pos + 1 < len(tokens) else ""
            query = f"{tok} {follow}" if follow and simple(follow) else tok
            if self.executed_searchterm(query):
                return None
            return {**template[0], template[1]: query}
        return None

    def reprime_traverse(self) -> bool | None:
        """v605f FF, under a cheapest rule: a product whose CG3 ceiling walk aimed at a listing that has died since (sold out
        or refused) and whose new pick is pricier gets the walk one more time, under the new pick, inside the repair
        reserve (down to FIRST_ORDER_BY steps); None when nothing is due."""
        if self.needs.priority() != "cheapest" or self.protocol.done or self.protocol.surplus_alt() < WRAPUP + 60 or not self.role("search"):
            return None
        aimed = vars(self).get("ceil_aimed") or {}
        dead_set = self.defunct_lineitems() | self.rejected_lineitems
        open_cost = 1 if self.bundle_instrument() else 2
        for pid, (aim_price, lid) in list(aimed.items()):
            if not lid or lid not in dead_set:
                continue
            if self.roof_specification() is None or self.sheet_param() is None:
                continue
            selected = self.traverse_choose(pid)
            if not selected or selected[1] <= aim_price + 1e-9:
                continue
            if self.shifts_remaining() < INITIAL_PURCHASE_VIA + 3 + open_cost:
                continue
            del aimed[pid]
            retained = vars(self).get("ceil_aims") or {}
            retained.pop(pid, None)      # a held put's price (the dead listing's) no longer sets the aim
            self.journal.append(dict(kind="ceil_rearm", product=pid, dead=lid, aim=selected[1]))
            return self.conduct_traverse([], INITIAL_PURCHASE_VIA, [pid], repair=True)
        return None

    def traverse_stage(self) -> bool:
        """Runs the cheapest walk queued by a held cart call or order; failing that, the price-filter listing of a product
        with a verified pick (CG3) while the steps allow; failing that, the walk that falls due as the reserve approaches:
        when the remaining steps shrink to the reserve plus the planned pages and reads (CG5: in any loop, booking only a
        plan that fits the steps). (Also runs the reads a held decline queued.)"""
        try:
            if vars(self).get("decline_queue"):
                return self.conduct_abstain_peruses()
            if vars(self).get("deeper_queue") or self.deeper_ready():
                return self.deeper_pages()
            pending_walk = vars(self).pop("walk_queue", None)
            if pending_walk:
                return self.conduct_traverse(*pending_walk)
            rerun = self.reprime_traverse()
            if rerun is not None:
                return rerun
            if self.purchases_attempted or self.needs.priority() != "cheapest" or self.protocol.surplus_alt() < WRAPUP + 60 or self.protocol.done:
                return False
            if self.undiscovered_ware():
                return False      # several products with one still unfound: the steps before the reserve go to its search
            open_cost = 1 if self.bundle_instrument() else 2
            spare_moves = self.shifts_remaining()
            ceil_ready = [pid for pid in self.catalog_wares() if (self.traverse_choose(pid) or self.boundary_requirement(pid)) and self.roofline_prepared(pid)
                     and self.roof_intent(pid) is not None]
            roomy = spare_moves > self.purchase_owed() + 2 + ROOF_INVOKES + open_cost
            if ceil_ready:
                if roomy:
                    # EB3: the product with the fewest value pages read goes first rather than the first in order; v605f FG:
                    # a product without a pick (an amount bound no keyword carries) precedes those that hold one
                    read_by = vars(self).get("valued_by") or {}
                    lead_pid = min(ceil_ready, key=lambda pid: (bool(self.traverse_choose(pid)), read_by.get(pid, 0), -1 if pid is None else pid))
                    return self.conduct_traverse([], self.purchase_owed() + 2, [lead_pid])
            elif roomy and self.ceil_seed():
                return True
            walk_route = self.traverse_scheme(TRAVERSE_AGGREGATE)
            if not walk_route:
                return False
            planned = sum(pos + (1 if self.traverse_choose(pid) else 0) for pid, _, pos in walk_route)
            if spare_moves > self.purchase_owed() + 2 + planned + open_cost:
                return False
            walk_route = self.traverse_scheme(min(TRAVERSE_AGGREGATE, spare_moves - self.purchase_owed() - 1 - open_cost))
            if not walk_route:
                return False      # CG5: no plan fits the steps right now; a smaller one may still run in a later loop
            self.traverse_premature = True
            return self.conduct_traverse(walk_route, self.purchase_owed())
        except Exception as exc:
            self.journal.append(dict(kind="walk_error", error=f"{type(exc).__name__}: {exc}"[:300]))
            return False

    def starter_params(self, shelf_name: str, ceil_val: Any) -> dict[str, Any] | None:
        """EB8: a category listing with a price limit built from nothing but the search schema: the argument the schema
        names for a category (a free-text one, or a choice that holds `shelf_name`), a numeric price argument, and the mode value
        that the tool's description pairs with the category argument; None when the schema offers no such call."""
        tname = self.role("search")
        sch = self.structures.get(tname or "") or {}
        raw_props = sch.get("properties")
        props = raw_props if isinstance(raw_props, dict) else {}
        pinfo = self.sheet_param()
        if not (props and pinfo and whether_numeric(ceil_val)):
            return None
        blurb = next((str(tdef["function"].get("description") or "") for tdef in self.toolkit if tdef["function"]["name"] == tname), "")
        clauses = [c.lower() for c in re.split(r"[.;]", blurb)]
        tok = self.lexeme_slot()
        shelf_key = shelf_val = None
        for name, cfg in props.items():
            if not isinstance(cfg, dict) or name in (tok, pinfo[0]) or not re.search(r"categor|shelf|department|aisle", str(name).lower()):
                continue
            choices = cfg.get("enum")
            if isinstance(choices, list):
                want = shelf_name.strip().lower()
                match = next((vv for vv in choices if isinstance(vv, str) and vv.strip().lower() == want), None)
                if match is None:
                    continue
                shelf_key, shelf_val = name, match
                break
            if cfg.get("type") == "string":
                shelf_key, shelf_val = name, shelf_name
                break

        def is_limit(name: str, cfg: Any) -> bool:
            if not isinstance(cfg, dict) or cfg.get("type") not in ("integer", "number") or cfg.get("enum") or name == pinfo[0]:
                return False
            lowtxt = str(name).lower()
            return bool(re.search(r"ceil|max|price|cap$|budget|upper|most", lowtxt)) and not re.search(r"page|first|offset|skip|count|size|num", lowtxt)
        bound_key = next((name for name, cfg in props.items() if is_limit(name, cfg)), None)
        if not (shelf_key and bound_key):
            return None
        whole = props[bound_key].get("type") == "integer" or float(ceil_val).is_integer()
        cargs: dict[str, Any] = {shelf_key: shelf_val, bound_key: int(ceil_val) if whole else ceil_val}
        shelf_pat = r"\b" + re.escape(shelf_key.lower()) + r"\b"

        def paired(vv: Any) -> bool:
            own = r"\b" + re.escape(str(vv).lower()) + r"\b"
            return any(re.search(own, c) and re.search(shelf_pat, c) for c in clauses)
        for name, cfg in props.items():
            if name == shelf_key or not isinstance(cfg, dict):
                continue
            choices = cfg.get("enum")
            if not isinstance(choices, list) or not 2 <= len(choices) <= 4:
                continue
            match = next((vv for vv in choices if paired(vv)), None)
            if match is None:
                return None      # the description never ties a mode to the category argument: no guessing
            cargs[name] = match
        if any(name not in cargs for name in sch.get("required") or []):
            return None
        fixed, defects = ParamAssess.carryout(cargs, sch)
        if defects or not isinstance(fixed, dict):
            return None
        return fixed

    def ceil_seed(self) -> bool:
        """EB8, under a cheapest rule while no price-filter listing is learned (or retired): once per product with a pick
        and a seen category label, the schema-built listing of that label (seed_args) with its limit one unit under the
        pick, so ceil_learn can learn the filter from the rows; the ceiling walk (CG3) follows and reuses this page."""
        try:
            if self.roof_specification() is not None or vars(self).get("ceil_mode") or self.needs.priority() != "cheapest":
                return False
            seeded = vars(self).setdefault("ceil_seeded", set())
            pinfo = self.sheet_param()
            pkey = pinfo[0] if pinfo else None
            for pid in self.catalog_wares():
                if pid in seeded or (pid is None and self.needs.items()):
                    continue
                selected = self.traverse_choose(pid)
                shelf_name = self.unprocessed_tag(self.traverse_landing(pid))
                if not (selected and shelf_name):
                    continue
                seeded.add(pid)
                cargs = self.starter_params(shelf_name, self.beneath(selected[1]))
                if not cargs:
                    return False
                fmode = dict(self.seek_slots(cargs)[1])
                kept = sorted((m, vv) for m, vv in fmode.items() if not (isinstance(cargs.get(m), str) and cargs[m] == shelf_name))
                limit_key = next((lo for lo in cargs if whether_numeric(cargs[lo]) and lo != pkey), "")
                msig = json.dumps([kept, limit_key], default=str)
                slot = json.dumps({lo: vv for lo, vv in cargs.items() if not pinfo or lo != pinfo[0]}, sort_keys=True, default=str)
                if msig in vars(self).get("ceil_bad", set()) or slot in (vars(self).get("ceil_seen") or {}):
                    continue
                vars(self).setdefault("ceil_seed_keys", set()).add(slot)
                self.traverse_invoke(cargs, f"The shopper wants the cheapest: listing '{shelf_name}' under the pick's price ({selected[1]:g}) to "
                                     "test whether the store's price filter works.")
                self.journal.append(dict(kind="ceil_seed", product=pid, args=cargs, learned=self.roof_specification() is not None))
                if self.converse and self.converse[-1].get("role") == "tool" and self.roof_specification() is None:
                    self.converse[-1]["content"] = str(self.converse[-1]["content"]) + "\n" + self.progress_lineitem()
                return True
            return False
        except Exception as exc:
            self.journal.append(dict(kind="ceil_seed_error", error=f"{type(exc).__name__}: {exc}"[:300]))
            return False

    def undiscovered_ware(self) -> bool:
        """With several products: true when some product has no listing of its own category read (nothing opened, or
        its category never met), or when one product holds a verified fit while another has none."""
        if not self.needs.items():
            return False
        try:
            read_labels = {lab.strip().lower() for lab in self.vault.racks_peruse()}
            for home_set in self.landing_racks():
                if not home_set or not (home_set & read_labels):
                    return True
            ver_tag = str(self.needs.iteration)
            live_any = any(key.rpartition("|")[2] == ver_tag for key in self.succeeded)
            return live_any and self.unconfirmed()
        except Exception:
            return False

    def deeper_pick(self, pids: list[int | None]) -> tuple[int | None, str, int] | None:
        """For deeper paging: the first of `pids` (products, None for a lone one) whose widest search - the one listing
        most rows of its category - still has unread pages, with the pages the steps allow (up to DEEPER_SHEETS, leaving
        the order's reserve, two steps per unsent message and a margin free). Products already paged this way, or covered
        by a cheapest walk or value search, are skipped."""
        prio = self.needs.priority()
        done = vars(self).setdefault("deeper_done", set())
        unsent = max(0, self.inquire_threshold() - self.letters_transmitted)
        room = self.shifts_remaining() - self.purchase_owed() - 2 * unsent - 6
        for pid in pids:
            if pid in done:
                continue
            if prio == "cheapest" and ((vars(self).get("valued_by") or {}).get(pid) or pid in vars(self).get("ceil_walked", set())):
                continue
            cats = self.traverse_landing(pid)
            best_slot, best_rows = None, 0
            for slot, pager in (vars(self).get("page_book") or {}).items():
                if not pager.get("first") or self.subsequent_sheet(pager) is None:
                    continue
                own_rows = sum(1 for sh in (pager.get("row_shelf") or {}).values() if not cats or sh in cats)
                if own_rows > best_rows:
                    best_slot, best_rows = slot, own_rows
            if best_slot is None or best_rows < 3:
                continue
            pager = vars(self)["page_book"][best_slot]
            left_pages = -(-max(0, pager["total"] - len(pager.get("row_shelf") or {})) // max(1, pager["size"]))
            take = min(DEEPER_SHEETS, left_pages, room)
            if take > 0:
                return pid, best_slot, take
        return None

    def deeper_ready(self) -> bool:
        """Under a best or cheapest rule, with nothing in the cart and no order tried: True when deeper paging is still
        possible for some product but the steps would no longer allow it after a couple more moves - it is queued then,
        so the pages are not lost to a late first cart put."""
        try:
            if self.needs.priority() not in ("best", "cheapest") or self.purchases_attempted or self.rejection_num:
                return False
            if self.handcart.measures or vars(self).get("deeper_queue") or not self.sheet_param():
                return False
            if self.protocol.surplus_alt() < WRAPUP + 60:
                return False
            if any(rr.get("status") == "open" for rr in vars(self).get("refs") or []):
                return False      # (an open reference's chain needs the steps first; its value may change what fits)
            unsent = max(0, self.inquire_threshold() - self.letters_transmitted)
            room = self.shifts_remaining() - self.purchase_owed() - 2 * unsent - 6
            if room > DEEPER_SHEETS + 2:
                return False
            picked = self.deeper_pick(self.catalog_wares())
            if not picked:
                return False
            vars(self).setdefault("deeper_done", set()).add(picked[0])
            self.deeper_queue = picked
            self.journal.append({"kind": "deeper_due", "product": picked[0], "pages": picked[2]})
            return True
        except Exception as exc:
            self.journal.append({"kind": "deeper_ready_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
            return False

    def deeper_keep(self, cart_call: dict[str, Any]) -> str:
        """Under a best-by-a-measure or cheapest rule, before a product's first cart put: when that product's widest search
        still has unread pages and the steps allow them (deeper_pick), the put is held once and the pages are queued for
        the code (deeper_pages): a winner on the measure can sit on any page of a store that lists rows in a fixed order,
        and the rank check only weighs rows that were seen."""
        try:
            prio = self.needs.priority()
            if prio not in ("best", "cheapest") or self.purchases_attempted or self.rejection_num or not self.sheet_param():
                return ""
            if vars(self).get("deeper_queue") or self.protocol.surplus_alt() < WRAPUP + 60:
                return ""
            if any(rr.get("status") == "open" for rr in vars(self).get("refs") or []) and self.reference_headroom():
                return ""
            homes = self.landing_racks()
            pids: list[int | None] = []
            for line_id, n_units in self.handcart.lines(cart_call):
                if n_units is not None and n_units <= 0:
                    continue
                if not self.needs.items():
                    pids = [None]
                    break
                shelf = self.rack_familiar(line_id)
                pid = next((n for n, home in enumerate(homes) if shelf and shelf in home), None)
                if pid is not None and pid not in pids:
                    pids.append(pid)
            picked = self.deeper_pick(pids)
            if not picked:
                return ""
            pid, slot, take = picked
            vars(self).setdefault("deeper_done", set()).add(pid)
            self.deeper_queue = picked
            pager = vars(self)["page_book"][slot]
            query = " ".join(self.loose_passage(pager["args"]))
            aim = "the lowest price" if prio == "cheapest" else (self.needs.perusal.get("priority_detail") or "the measure asked for")
            self.journal.append({"kind": "deeper_hold", "product": pid, "query": query, "pages": take})
            return (f"Not run (no step used). The shopper wants the winner on {aim}, and the search '{query}' has pages "
                    f"not read yet ({pager['total']} matches, rows in a fixed order, not ranked): the code reads up to {take} "
                    "more of them now. Then compare every fitting listing seen on that measure and put the winner in the cart.")
        except Exception as exc:
            self.journal.append({"kind": "deeper_keep_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
            return ""

    def deeper_pages(self) -> bool:
        """Reads the pages deeper_keep queued (scripted, one page per step), keeping the order's reserve free."""
        queued = vars(self).pop("deeper_queue", None)
        if not queued:
            return False
        pid, slot, take = queued
        pinfo = self.sheet_param()
        pager = (vars(self).get("page_book") or {}).get(slot)
        if not (pinfo and pager):
            return False
        query = " ".join(self.loose_passage(pager["args"]))
        say_txt = f"Reading the next page of '{query}', so the winner on the measure is picked from every row it lists."
        keep_steps = self.purchase_owed() + 2
        n_read = 0
        while (n_read < take and self.subsequent_sheet(pager) is not None and self.shifts_remaining() > keep_steps + 1
               and not self.protocol.done and not self.protocol.crashed and self.protocol.surplus_alt() >= WRAPUP + 30):
            nextp = self.subsequent_sheet(pager)
            if not self.traverse_invoke({**pager["args"], pinfo[0]: nextp}, say_txt):
                break
            n_read += 1
            if self.subsequent_sheet(pager) == nextp:
                break
        seen = len(pager.get("row_shelf") or {})
        recap = (f"[pages read] '{query}': {seen} of {pager['total']} matches now seen. Compare every fitting listing seen on "
                 "the shopper's measure (open the details of the strongest candidates first) and put the winner in the cart.")
        if self.converse and self.converse[-1].get("role") == "tool":
            self.converse[-1]["content"] = str(self.converse[-1]["content"]) + "\n" + recap + "\n" + self.progress_lineitem()
        self.journal.append({"kind": "deeper_pages", "product": pid, "query": query, "pages": n_read})
        return n_read > 0

    def traverse_keep(self, cart_call: dict[str, Any] | None) -> str:
        """Holds a cart put (cart_call given) or an order (cart_call None) while the cheapest is wanted and a product it
        puts or keeps has not been walked yet: when the steps allow it (keeping the put, one re-read, the order and one
        repair), the walk is queued and the call is held back without spending a step."""
        try:
            if self.needs.priority() != "cheapest":
                return ""
            if self.__dict__.get("walk_queue") or self.protocol.surplus_alt() < WRAPUP + 60:
                return ""
            if self.rejection_num and self.shifts_remaining() <= 15:
                return ""      # (late repair of a refused order is left alone)
            if cart_call is None:
                wanted_ids = list(self.handcart.measures)
            else:
                wanted_ids = [line_id for line_id, units_n in self.handcart.lines(cart_call) if units_n is None or units_n > 0]
            product_set = {self.ware_position(line_id) for line_id in wanted_ids if line_id}
            reserve = 4 if self.purchases_attempted else INITIAL_PURCHASE_VIA + 1
            # (pages, the line search and the opens together get at most half of the steps beyond those kept back and the
            # held call itself: the walk stops at that half by its own floor; pages take the half minus the opens, and the
            # line search whatever remains)
            half_room = (self.shifts_remaining() - reserve - 1) // 2
            spare_calls = half_room - (1 if self.bundle_instrument() else 2)
            ceil_marks: list = []
            aim_book = self.__dict__.setdefault("ceil_aims", {})

            def none_low(pidx: Any) -> Any:
                return -1 if pidx is None else pidx
            for pidx in sorted(product_set, key=none_low):
                # (CG3: a product put in or ordered for the first time also gets listed below its price through the learned
                # price filter, with CEIL_CALLS calls reserved for it ahead of its value searches' pages)
                if spare_calls < 2 or not self.roofline_prepared(pidx):
                    continue
                ceil_marks.append(pidx)
                spare_calls -= ROOF_INVOKES
                quotes = [self.vault.price(line_id) for line_id in wanted_ids if line_id and self.ware_position(line_id) == pidx]
                known_costs = [float(qv) for qv in quotes if whether_numeric(qv)]
                if known_costs:
                    if whether_numeric(aim_book.get(pidx)):
                        known_costs.append(float(aim_book[pidx]))
                    aim_book[pidx] = min(known_costs)
            walk_route = self.traverse_scheme(min(TRAVERSE_AGGREGATE, spare_calls), product_set)
            if not (walk_route or ceil_marks):
                return ""
            self.walk_queue = (walk_route, max(reserve, self.shifts_remaining() - half_room - 1), ceil_marks)
            book_pages = self.__dict__.get("page_book", {})
            phrase_list = [" ".join(self.loose_passage(book_pages[pkey]["args"])) for _, pkey, _ in walk_route if pkey in book_pages]
            duties = []
            if phrase_list:
                duties.append("reads the still unread pages of " + ", ".join(f"'{ph}'" for ph in phrase_list))
            if ceil_marks:
                duties.append("lists the category's listings priced under the pick through the store's price filter")
            return ("Not run yet (no step used). The cheapest fit is wanted, and this store returns search results in a fixed "
                    "id order rather than by price, so a cheaper fit can hide on pages nobody has read: on your behalf the code now "
                    + " and ".join(duties) + ", then opens the cheapest unread rows priced under the pick. After that, send the call "
                    "again, or the same call with a cheaper fit.")
        except Exception as exc:
            self.journal.append({"kind": "walk_hold_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
            return ""

    def seek_annotation(self, args_in: dict[str, Any], result: Any) -> str:
        """Remarks on one search result: its reported total exceeds the rows shown (the rest stay unseen, whatever their
        price or fit), or its rows break a numeric limit the call set (a filter the search did not apply)."""
        if not isinstance(result, dict):
            return ""

        def rows_in(piece: dict[str, Any]) -> list | None:
            lists_found = [item for item in piece.values() if isinstance(item, list) and all(isinstance(row_d, dict) for row_d in item)]
            nonempty = [item for item in lists_found if item]
            if nonempty:
                return nonempty[0]
            return lists_found[0] if lists_found else None
        shown_rows = rows_in(result)
        if shown_rows is None:
            # (the rows can sit one level down, wrapped beside a shopper message or a refusal)
            nested = None
            for item in result.values():
                if isinstance(item, dict) and rows_in(item) is not None:
                    nested = item
                    break
            if nested is None:
                return ""
            result = nested
            shown_rows = rows_in(result) or []
        over_totals = [item for item in result.values() if isinstance(item, int) and not isinstance(item, bool) and item > len(shown_rows)]
        top_total = max(over_totals) if over_totals else len(shown_rows)
        self.ledger_sheet(args_in, shown_rows, top_total, result)
        try:
            self.roof_discover(args_in, shown_rows, top_total, result)
        except Exception as exc:
            self.journal.append({"kind": "ceil_learn_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
        lead_score = self.setting_rating(args_in) if shown_rows else 0
        if lead_score:
            row_best = self.__dict__.setdefault("row_score", {})
            for row_d in shown_rows:
                if not isinstance(row_d, dict):
                    continue
                mark = self.tablerow_flag(row_d)
                row_best[mark] = max(row_best.get(mark, 0), lead_score)
        remarks_out: list[str] = []
        word_mode = self.term_principle(args_in, result, shown_rows)
        lead_n = self.foremost_principle(args_in, result, shown_rows)
        text_fields, mode_pairs, first_page, page_keys = self.seek_slots(args_in)
        mode_names = dict(mode_pairs)
        if lead_n:
            ctx_field = self.foremost_context[0]
            counted_note = ""
            if mode_pairs == self.foremost_context[1]:
                qwords = next((wd for fkey, wd in text_fields if fkey == ctx_field), ())
                if len(qwords) > lead_n:
                    counted_note = f" (of this query only these counted: {' '.join(qwords[:lead_n])})"
            remarks_out.append(f"The keyword search here ({ctx_field}) uses just the FIRST {lead_n} word(s) of a query and drops the "
                         f"rest{counted_note}: lead with the {lead_n} most telling words (a required value together with the item kind, "
                         "value first), and look up other values in short queries of their own.")
        typed = [item for kname, item in args_in.items() if isinstance(item, str) and kname not in mode_names and kname not in page_keys]
        if first_page and typed:
            # (some stores skip one-character words: a query made only of those finds nothing)
            tokens_1 = [tk for tx in typed for tk in re.findall(r"[a-z0-9]+", tx.lower())]
            if tokens_1 and all(len(tk) == 1 for tk in tokens_1) and not shown_rows and not self.__dict__.get("short_told"):
                self.short_told = True
                remarks_out.append("Nothing came back: this store's search skips one-character words, so a figure such as 0.4 or "
                             "5 x 7 searches for nothing (or only for its other words). Search with a word value, or with the unit "
                             "word plus the item kind, and read all of the pages.")
            name_prop, value_query = self.setting_explore(typed)
            named_n = self.__dict__.get("name_told", 0)
            if value_query:
                if named_n < 2 and self.shifts_remaining() > self.purchase_owed() + 3:
                    self.name_told = named_n + 1
                    if named_n:
                        opener_txt = f"Once more the query leads with the NAME of a property ({name_prop})."
                    else:
                        opener_txt = (f"This query opens with words of a property's NAME ({name_prop}) and carries no required value: "
                                   "keyword search matches listing values, titles and categories, and a field's name only "
                                   f"where some value or title repeats it, so listings whose {name_prop} field holds only the value "
                                   "are missed.")
                    remarks_out.append(opener_txt + f" Put the value first and the item kind after it: '{value_query}' (a figure the store "
                                 "cannot read falls away and leaves its unit word), then read every page of it.")
            elif self.__dict__.get("trap_told", 0) < 2 and self.shifts_remaining() > self.purchase_owed() + 3:
                trap_text, _ = self.term_snare(typed)
                if trap_text:
                    self.trap_told = self.__dict__.get("trap_told", 0) + 1
                    remarks_out.append(trap_text)
            row_ids = tuple(self.tablerow_flag(row_d) for row_d in shown_rows)
            other_args = tuple(sorted((fkey, str(fval)) for fkey, fval in args_in.items()
                                if fkey not in page_keys and not (isinstance(fval, str) and fkey not in mode_names)))
            page_sig = (mode_pairs, other_args, top_total, row_ids)
            sig_book = self.__dict__.setdefault("page_sigs", {})
            joined_query = " ".join(typed)
            if len(set(row_ids)) >= 3 and sig_book.get(page_sig, joined_query) != joined_query:
                remarks_out.append(f"Your earlier search '{sig_book[page_sig]}' gave this same total and these same rows: the store reads "
                             "both alike, so spelling it differently shows nothing new. Read its next page, or change the words "
                             "it starts with.")
            sig_book.setdefault(page_sig, joined_query)
            # (a store needing every word and showing all matches: each row holds the words it matched, often in fields
            # the title hides)
            matched: list = []
            strict_all = bool((word_mode == "all" or self.foremost_terms) and not over_totals and shown_rows)
            if strict_all or (word_mode != "any" and not over_totals and shown_rows):
                # (EB5: the rows of a fully shown search are booked whatever rule was learned, unless the store takes any
                # word; the remarks are kept for a store known to read every word)
                kept_refs = None if strict_all else list(self.__dict__.get("ref_hits") or [])
                self.ref_hits = []
                matched = self.tablerow_strikes(shown_rows, typed)
                val_score = self.setting_rating(args_in)
                rank_book = self.__dict__.setdefault("hit_score", {})
                rank_book.update({hid: max(rank_book.get(hid, 0), val_score) for hid in matched})
                hit_pool = self.__dict__.setdefault("all_hits", [])
                hit_pool.extend([hid for hid in matched if hid not in hit_pool])
                if kept_refs is not None:
                    self.ref_hits = kept_refs
                if strict_all and matched and text_fields and len(shown_rows) <= 12:
                    remarks_out.append("Here every word is required and every match is listed: each row holds the words the store "
                                 "matched somewhere in its record, often in fields its title leaves out. Open the ones within "
                                 "the budget before searching further: " + ", ".join(matched[:6]) + ".")
                if strict_all:
                    pointed = [hid for hid in self.__dict__.pop("ref_hits", []) if hid not in matched][:6]
                    if pointed:
                        remarks_out.append("The request states a value by pointing at another listing, and these rows come from the "
                                     "shop or brand this search names: open them whatever their price, stock or category (a "
                                     "listing that supplies a value is read, not bought), and do what its notes say: "
                                     + ", ".join(pointed) + ".")
            if (word_mode == "all" or self.foremost_terms) and not self.__dict__.get("hits_back"):
                # (the rule is only known now: earlier searches that listed every match and carry a required value)
                self.hits_back = True
                own_key = json.dumps({kname: item for kname, item in args_in.items() if kname not in page_keys}, sort_keys=True, default=str)
                backfill: list[str] = []
                for pkey, walk_rec in self.__dict__.get("page_book", {}).items():
                    if pkey == own_key or not walk_rec.get("every") or not self.setting_rating(walk_rec["args"]):
                        continue
                    fresh_ids = [hid for hid in self.tablerow_strikes(walk_rec["every"]) if hid not in backfill and hid not in matched]
                    rank_book = self.__dict__.setdefault("hit_score", {})
                    traverse_rating = self.setting_rating(walk_rec["args"])
                    rank_book.update({hid: max(rank_book.get(hid, 0), traverse_rating) for hid in fresh_ids})
                    backfill += fresh_ids
                hit_pool = self.__dict__.setdefault("all_hits", [])
                hit_pool.extend([hid for hid in backfill if hid not in hit_pool])
                if backfill:
                    remarks_out.append("The keyword search of this store requires every word, so earlier searches that listed every "
                                 "match showed only rows holding all of their words, often in fields the titles leave out. "
                                 "Open the unread ones within the budget before searching further: "
                                 + ", ".join(backfill[:6]) + ".")
        if not shown_rows:
            return " ".join(remarks_out)
        if over_totals:
            if word_mode == "any":
                how_text = ("the keyword search here takes ANY of the words and ranks listings matching more of them higher, "
                          "so extra words widen it: a required value's own distinctive word or figure on its own lists only the "
                          "listings carrying it (often few enough to read every page and compare prices); the item kind plus the "
                          "values puts the closest matches first; use the category mode (or a price filter if one works) for "
                          "the rest.")
            elif word_mode == "all":
                how_text = ("the keyword search here needs EVERY word, so more exact words narrow it, and so do the category "
                          "and price filters with a lower price limit.")
            else:
                how_text = ("then narrow it for the rest: when every word is required, add exact words; when any word matches "
                          "(a longer query matching more listings shows that), put the required values in the query so "
                          "fitting listings rank first; a price filter, when one works, helps too.")
            remarks_out.append(f"This search matched {max(over_totals)} listings yet shows just {len(shown_rows)} on this page, and only a few "
                         "pages can be read: the listings that matter may be among those not shown. Read this same search's "
                         "further pages first (as many as can be read); if the total exceeds those pages, " + how_text)
        caps = [item for kname, item in args_in.items() if whether_numeric(item) and item > 1 and kname not in page_keys
                      and not re.search(r"page|size|count|result|min|low|from|start", str(kname).lower())]
        if caps:
            cap_val = max(caps)

            def above_cap(one_row: dict[str, Any]) -> bool:
                return any(whether_numeric(item) and item > cap_val and "price" in str(kname).lower() for kname, item in one_row.items())
            too_dear = [row_d for row_d in shown_rows if above_cap(row_d)]
            if too_dear:
                remarks_out.append(f"{len(too_dear)} of these rows are priced above the limit {cap_val} this call set: this search mode "
                             "did not apply that filter (and may skip others, such as the category). Use a mode whose filters "
                             "work, or check each row yourself.")
        return " ".join(remarks_out)

    def customer_lineitems(self, tool_used: str, reply_obs: Any, was_refused: bool) -> list[str]:
        lines_out: list[str] = []
        if tool_used != self.role("message"):
            if isinstance(reply_obs, dict):
                lines_out = self.alert_lineitems(reply_obs, was_refused)
        elif isinstance(reply_obs, str):
            lines_out = [reply_obs]
        elif isinstance(reply_obs, dict) and "error" not in reply_obs:
            # (the reply can be wrapped: every sentence-like string of the result, outermost first)
            def harvest(node: dict[str, Any]) -> None:
                lengthy_charseqs([val for val in node.values() if isinstance(val, (str, list))], lines_out)
            traverse_mappings(reply_obs, harvest, ceiling=4)
        return list(dict.fromkeys(lines_out))

    @staticmethod
    def alert_lineitems(payload: Any, refusal_seen: bool = False) -> list[str]:
        """Shopper lines or store notices inside the result of any other tool: lists of sentence-like strings at the top
        level and within small nested wrappers (a dict of at most six keys; listing records hold more); for a refusal,
        its own sentence-like strings as well."""
        sentences: list[str] = []
        if not isinstance(payload, dict):
            return sentences

        def cites_listing(node: dict[str, Any]) -> bool:
            return any(isinstance(fval, str) and "id" in str(fkey).lower() and re.fullmatch(UID_RESEMBLING, fval) for fkey, fval in node.items())

        def harvest(node: dict[str, Any]) -> None:
            if len(node) > 6 and node is not payload:
                return
            for val_piece in node.values():
                if isinstance(val_piece, list):
                    if val_piece and all(isinstance(fval, str) for fval in val_piece):
                        lengthy_charseqs(val_piece, sentences)
                    continue
                if not isinstance(val_piece, str) or len(val_piece.split()) < 4:
                    continue
                # (a short sentence in a small wrapper naming no listing is a notice, not a record's own text)
                if refusal_seen or (len(node) <= 3 and len(val_piece) <= 600 and not cites_listing(node)):
                    sentences.append(val_piece)
        traverse_mappings(payload, harvest, ceiling=4)
        return list(dict.fromkeys(sentences))

    # ----------------------------------------------------------------------------------------------- a single batch
    def sift(self, prop_call: dict[str, Any], open_tools: set[str], by_code: bool, batch_n: int, msgs_waiting: int) -> tuple[str | None, str, dict[str, Any]]:
        """For one proposed call: (the refusal text, or None), the tool's name, and the checked arguments."""
        func_spec = prop_call.get("function") or {}
        tool_name = func_spec.get("name")
        arg_blob = func_spec.get("arguments")
        if tool_name not in self.structures:
            names_list = ", ".join(sorted(open_tools))
            return f"Not run (no step used): there is no tool called '{tool_name}'. Pick one of: {names_list}.", tool_name, {}
        if not by_code and tool_name not in open_tools:
            return ("Not run (no step used): not enough steps remain for that; the only tools open now are: "
                    + ", ".join(sorted(open_tools)) + "."), tool_name, {}
        if isinstance(arg_blob, dict):
            args_ok = arg_blob
        elif (arg_blob or "").strip():
            args_ok = serialized_entity(arg_blob)
        else:
            args_ok = {}
        if args_ok is None:
            return "Not run (no step used): the arguments did not parse as a JSON object.", tool_name, {}
        args_ok, problems = ParamAssess.carryout(args_ok, self.structures[tool_name])
        if problems:
            return "Not run (no step used): the arguments were rejected: " + "; ".join(problems[:6]), tool_name, args_ok
        is_finish = tool_name in self.concluders()
        if is_finish and batch_n > 1:
            return ("Not run (no step used): an order or a decline has to go out on its own, once the results of your "
                    "other calls are in."), tool_name, args_ok
        message_tool = self.role("message")
        if tool_name == message_tool:
            msgs_planned = self.letters_transmitted + msgs_waiting
            if self.letter_lid is not None and msgs_planned >= self.letter_lid:
                return ("Not run (no step used): no messages are left in the budget. Do not write to the shopper "
                        "again."), tool_name, args_ok
            if not by_code and not self.handcart.measures and self.letter_buffer() and msgs_planned >= self.inquire_headroom():
                return ("Not run (no step used): the remaining messages are saved for telling the shopper the basket once "
                        "the cart holds something (and for any finishing step the instructions name). Put your pick in the "
                        "cart first; then the message tool is open to you again."), tool_name, args_ok
            last_one = self.letter_lid is not None and msgs_planned + 1 >= self.letter_lid
            if not by_code and last_one and self.needs.items() and self.handcart.measures and self.shifts_remaining() > 4:
                # (several products: the last message the shopper answers is the one about the final cart, so it waits until
                # the cart holds a line for every product - a cart change after it would leave it describing another cart)
                homes = self.landing_racks()
                covered = {n for line_id in self.handcart.measures for n, home in enumerate(homes)
                           if (self.rack_familiar(line_id) or "") in home}
                missing = [row["product"] for n, row in enumerate(self.needs.items()) if n not in covered]
                told = self.__dict__.setdefault("partial_told", set())
                key = (tuple(sorted(self.handcart.measures)), tuple(missing))
                if missing and key not in told:
                    told.add(key)
                    return ("Not run (no step used): this is the last message the shopper will answer, and the cart has no "
                            "line yet for: " + ", ".join(missing) + ". Put a line for every requested product in the cart "
                            "first, then send this message about the final cart (a cart change after it would leave the "
                            "shopper's last word about another cart)."), tool_name, args_ok
            if not by_code and last_one and self.handcart.measures and self.shifts_remaining() > 4:
                # (the last message allowed, while a cheaper passed fit of a cart line is known: once it is sent the cart
                # can no longer change without leaving that message stale, so the swap comes first)
                told = self.__dict__.setdefault("cheaper_told", {})
                for line_id in list(self.handcart.measures):
                    cheaper_id = self.lowercost_suit(line_id)
                    if cheaper_id and cheaper_id not in self.handcart.measures and not told.get(("last_msg", line_id, cheaper_id)):
                        told[("last_msg", line_id, cheaper_id)] = 1
                        return (f"Not run (no step used): this is the last message the shopper will answer, and the "
                                f"independent item check judged that {cheaper_id} meets every requirement currently in force "
                                f"and is cheaper than {line_id}, which is in the cart. Settle the cart first (put {cheaper_id} "
                                f"in place of {line_id} unless a record shows it failing a requirement), then send the "
                                "message about the final cart."), tool_name, args_ok

        def view_adds_nothing() -> bool:
            # (CB6: the newest cart call's reply already listed the lines and the actor made no call of its own since -
            # the code's scripted reads aside: a view shows nothing new, and no stock either)
            if by_code or tool_name != self.role("cart_view") or tool_name == self.role("cart"):
                return False
            inst = self.__dict__
            if inst.get("batch_cart") or not self.handcart.measures:
                return False
            return inst.get("cart_shown", -1) == inst.get("cart_ver", -2) and not inst.get("since_cart", 1)
        if view_adds_nothing():
            views_told = self.__dict__.setdefault("view_told", set())
            if self.cart_shown not in views_told:
                views_told.add(self.cart_shown)
                stock_reader = self.inventory_instrument() or "the tool that shows stock"
                return ("Not run (no step used): the newest cart result already shows the basket - " + self.handcart.recap_alt()
                        + " Viewing the cart now adds nothing; to check a line's stock again, read its record with "
                        + stock_reader + "."), tool_name, args_ok
        call_sig = tool_name + into_passage(args_ok)
        may_repeat = is_finish or tool_name in (message_tool, self.role("cart"))
        if not may_repeat and call_sig in self.final_invokes:
            return ("Not run (no step used): this exact call was just made and nothing has changed since; its result is "
                    "above. Use other arguments or take a different step."), tool_name, args_ok
        if not by_code:
            held_note = self.belated_keep(tool_name, args_ok) or self.setting_sentry(tool_name, args_ok)
            if held_note:
                return held_note, tool_name, args_ok
        cart_refusal = self.sift_carriage(args_ok, call_sig) if tool_name == self.role("cart") else None
        if not cart_refusal and not by_code and tool_name == self.role("cart"):
            cart_refusal = self.deeper_keep(args_ok)
        return (cart_refusal or None), tool_name, args_ok

    def sift_carriage(self, carriage_params: dict[str, Any], call_sig: str) -> str:
        aim_id = self.handcart.aim(carriage_params)
        is_putting = self.handcart.whether_place(carriage_params)
        ptr_note = self.reference_indicator_annotation(aim_id) if (aim_id and is_putting) else ""
        ptr_told = self.__dict__.setdefault("pointer_told", {})
        if ptr_note and ptr_told.get(aim_id, 0) < 2:
            # (CE5: the reference's source, or a listing on its chain, carries the excluded value)
            ptr_told[aim_id] = ptr_told.get(aim_id, 0) + 1
            return f"Not run (no step used): {ptr_note}. Pick a different listing."
        warn_mark = "{}#{}".format(call_sig, self.__dict__.get("puts", {}).get(aim_id, 0))
        aim_price = self.vault.price(aim_id) if aim_id else None

        def undercut_in_batch() -> bool:
            for peer in self.__dict__.get("batch_puts", []):
                if peer == aim_id:
                    continue
                if whether_numeric(aim_price) and whether_numeric(self.vault.price(peer)) and self.vault.price(peer) < aim_price:
                    return True
            return False
        should_swap = self.needs.priority() == "cheapest" and (undercut_in_batch() or bool(self.lowercost_suit(aim_id)))
        price_moved = aim_id in self.__dict__.get("pass_price", {}) and self.__dict__["pass_price"][aim_id] != aim_price
        was_passed = f"{aim_id}|{self.needs.iteration}" in self.succeeded
        if (aim_id and not is_putting and was_passed and not should_swap and not price_moved
                and self.rejection_num == self.rejections_observed.get(aim_id, self.rejection_num) and warn_mark not in self.cautioned):
            # (a line the check passed is being taken out with no change or refusal since: held once each time it went in)
            self.cautioned.add(warn_mark)
            cheapest_too = self.needs.priority() == "cheapest" and not self.lowercost_suit(aim_id)
            least_note = " It is the cheapest verified fit as well." if cheapest_too else ""
            return (f"Not run (no step used): an independent check judged that {aim_id} meets every requirement currently in "
                    "force, reading each property by the value rules (where this store writes the value itself in a field, "
                    "only those words meet it); only the category label settles the item kind, whatever object its title or "
                    f"description names.{least_note} Nothing has changed since then. If a change or a refusal rules it out, if one "
                    "of its fields plainly fails a requirement, or if you are swapping it for a cheaper listing, send the same "
                    "call again.")
        if not (is_putting and aim_id):
            return ""
        if self.handcart.retained(aim_id) > 0 and self.handcart.attaches is not False and call_sig not in self.cautioned:
            self.cautioned.add(call_sig)
            held_n = int(self.handcart.retained(aim_id))
            return (f"Not run (no step used): {aim_id} x{held_n} is in the cart already, and this cart tool adds units to "
                    "a line it holds, so this call would increase it. To order it, leave the cart alone; to change the units, "
                    "take out or put in the difference. If adding is really what you mean, send the same call again.")
        walk_note = self.traverse_keep(carriage_params) if not ptr_note else ""     # (v605d M2: no cheapest walk under a listing the reference points at)
        if walk_note:
            return walk_note

        def vet_line(ln_id: str) -> str:
            ver_key = f"{ln_id}|{self.needs.iteration}"
            if self.shifts_remaining() <= 3 or ver_key in self.assessed_attaches:
                return ""
            self.assessed_attaches.add(f"{ln_id}|{self.needs.iteration}")
            ruling_text = self.assess_entry(ln_id, "ADD TO THE CART")
            return ("Not run (no step used). " + ruling_text + " If you are sure it fits, send the same call again.") if ruling_text else ""
        for extra_id, n_units in self.handcart.lines(carriage_params)[1:]:
            # (a call naming several lines has each new line checked as well)
            if not (n_units is None or n_units > 0):
                continue
            ruling_text = vet_line(extra_id)
            if ruling_text:
                return ruling_text
        ruling_text = vet_line(aim_id)
        if ruling_text:
            return ruling_text
        rank_targets = self.handcart.lines(carriage_params) if self.shifts_remaining() > 3 else []
        for ln_id, n_units in rank_targets:
            # (each line of the call gets the rank check, once per line and requirement version - again after a cheapest walk)
            if n_units is not None and n_units <= 0:
                continue
            rank_note = self.standing_assess(ln_id)
            if not rank_note:
                continue
            cheaper_id = self.standing_ascertain(ln_id)      # (v605f FD: the read listings it names are checked straight away)
            held_text = rank_note.replace("before ordering", "before putting it in the cart").replace(
                "order the current pick", "send the same call again")
            if cheaper_id:
                held_text += (f" The independent item check judged that {cheaper_id} meets every requirement currently in force and "
                          f"is cheaper than {ln_id}: put {cheaper_id} in instead.")
            return "Not run (no step used). " + held_text
        if self.needs.priority() == "cheapest" and not self.cart_sealed():
            for ln_id, n_units in self.handcart.lines(carriage_params):
                # (v605f FD: a put undercut by a cheaper item-check pass is held once per pair)
                if n_units is not None and n_units <= 0:
                    continue
                cheaper_id = self.lowercost_suit(ln_id)
                told_pairs = self.__dict__.setdefault("cheaper_told", {})
                pair_key = ("cart", ln_id, cheaper_id)
                if not cheaper_id or cheaper_id in self.handcart.measures or told_pairs.get(pair_key):
                    continue
                told_pairs[pair_key] = 1
                return (f"Not run (no step used). The independent item check judged that {cheaper_id} meets every requirement "
                        f"currently in force, and it is cheaper than {ln_id}. The cheapest fit is wanted and the check's "
                        f"reading outranks yours: put {cheaper_id} in instead. If a record shows {cheaper_id} failing a requirement "
                        "currently in force, send the same call again.")
        wanted_n = self.needs.quantity()
        asked_n = self.handcart.headcount_param(carriage_params)
        target_n = wanted_n if self.handcart.attaches is False else wanted_n - self.handcart.retained(aim_id)
        if self.needs.items():
            odd_lines = [(ln, int(n_units)) for ln, n_units in self.handcart.lines(carriage_params)
                   if n_units is not None and n_units > 0 and int(n_units) not in self.needs.headcounts()]
            if not odd_lines or call_sig in self.cautioned:
                return ""
            self.cautioned.add(call_sig)
            return (f"Not run (no step used): the requested products call for {self.needs.headcounts()} units (a line apiece), "
                    f"yet this call asks for {odd_lines}. If you are sure, send the same call again.")
        if asked_n is None or not abs(asked_n - target_n) > 1e-9 or call_sig in self.cautioned:
            return ""
        self.cautioned.add(call_sig)
        return (f"Not run (no step used): the request wants {wanted_n} of a single listing, yet this call asks for {int(asked_n)}. "
                "If you are sure, send the same call again.")

    def sift_conclude(self, call_id: str, finish_tool: str, finish_args: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        """For an order or a decline: the answer that holds it back (None lets it go), and the arguments to send."""
        finish_kind = self.invoke_category(finish_tool, finish_args)
        inst = self.__dict__

        def reads_needed(row_ids: list, bulk_spec: Any) -> int:
            if bulk_spec and len(row_ids) >= 2:
                return (len(row_ids) + bulk_spec[2] - 1) // bulk_spec[2]
            return len(row_ids)

        def order_hold() -> str:
            if self.shifts_remaining() > 2:
                # (CB5: a cart line whose newest record shows it sold out makes the order fail - held twice per cart
                # state, after which the code swaps that line itself, keeping the rest, if the steps allow)
                traded_depleted = sorted(set(self.handcart.measures) & self.defunct_lineitems())
                cart_mark = (tuple(traded_depleted), inst.get("cart_seq", 0))
                dead_warns = inst.setdefault("dead_told", {})
                if traded_depleted and dead_warns.get(cart_mark, 0) < 2:
                    dead_warns[cart_mark] = dead_warns.get(cart_mark, 0) + 1
                    return "Not run yet (no step used). " + self.defunct_annotation(traded_depleted)
                if traded_depleted and dead_warns.get(cart_mark, 0) == 2 and self.role("cart") and self.shifts_remaining() >= 6\
                        and self.protocol.surplus_alt() > WRAPUP:
                    dead_warns[cart_mark] = 3
                    mute_was = self.protocol.silence
                    try:
                        swapped = self.engine_carriage(set(traded_depleted) | self.defunct_lineitems(), set(self.handcart.measures) - set(traded_depleted))
                    except Exception as exc:
                        swapped = False
                        self.journal.append({"kind": "dead_swap_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
                    finally:
                        self.protocol.silence = mute_was
                    self.journal.append({"kind": "dead_swap", "lines": traded_depleted, "done": swapped, "cart": dict(self.handcart.measures)})
                    if swapped and self.handcart.measures and not (set(self.handcart.measures) & self.defunct_lineitems()):
                        return ("Not run yet (no step used). The cart held " + ", ".join(traded_depleted) + ", which is sold out, so "
                                "the code removed it and put a candidate in its place, kept the other lines, and read the "
                                "lines' stock: " + self.handcart.recap_alt() + " Compare this cart with the request, then "
                                "order it.")
            if self.shifts_remaining() > 3 and not self.cart_sealed():
                chosen_id = self.selected()
                cheaper_id = self.lowercost_suit(chosen_id) if chosen_id else ""
                if not cheaper_id:
                    # (v605f FD: every cart line, not just the newest one)
                    for line_alt in [line_id for line_id in self.handcart.measures if line_id != chosen_id]:
                        cheaper_id = self.lowercost_suit(line_alt)
                        if cheaper_id:
                            chosen_id = line_alt
                            break
                cheap_warns = inst.setdefault("cheaper_told", {})
                if cheaper_id and cheap_warns.get((chosen_id, cheaper_id), 0) < 2:
                    cheap_warns[(chosen_id, cheaper_id)] = cheap_warns.get((chosen_id, cheaper_id), 0) + 1
                    return (f"Not run yet (no step used). The independent item check judged that {cheaper_id} meets every "
                            f"requirement, and it is cheaper than {chosen_id}. The cheapest fit is wanted and the check's reading "
                            f"outranks yours: take {chosen_id} out of the cart, put {cheaper_id} in, re-read its stock and price, "
                            "and order it.")
            if self.handcart.measures:
                walk_note = self.traverse_keep(None)
                if walk_note:
                    return walk_note
            stock_reader = self.inventory_instrument()
            if stock_reader and self.shifts_remaining() > len(self.handcart.measures) + 2 and self.protocol.surplus_alt() > WRAPUP:
                # (CB3: the re-read uses the tool whose answers carry the stock; a line counts as read only from a record
                # showing its stock after the newest cart change)
                unread_lines = self.outdated_lineitems()
                cart_mark = (tuple(sorted(self.handcart.measures)), inst.get("cart_seq", 0))
                if unread_lines and inst.get("recheck_mark") != cart_mark:
                    self.recheck_mark = cart_mark
                    self.open_tool = stock_reader
                    self.open_purpose = "cart"
                    self.open_recheck = True
                    self.open_queue = unread_lines[:4]
                    return ("Not run yet (no step used). Stock and prices can change without any notice, and these cart "
                            "lines have not been read since the last cart change: " + ", ".join(unread_lines[:4]) + ". Their stock "
                            f"is being read for you now ({stock_reader}, the tool that shows stock); then order if each is still "
                            "in stock and the total fits the budget, otherwise replace the line.")
            try:
                sweep_note = self.cheap_sweep()
            except Exception as exc:
                sweep_note = ""
                self.journal.append({"kind": "cheap_sweep_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
            if sweep_note:
                return sweep_note
            if self.shifts_remaining() > 5 and self.protocol.surplus_alt() > WRAPUP:
                try:
                    dep_list = self.prerequisite_defects()
                except Exception:
                    dep_list = []
                cart_mark = (tuple(sorted(self.handcart.measures)), tuple(dep_list))
                dep_warns = inst.setdefault("dep_told", {})
                if dep_list and dep_warns.get(cart_mark, 0) < 2 and len(dep_warns) < 4:
                    dep_warns[cart_mark] = dep_warns.get(cart_mark, 0) + 1
                    return ("Not run yet (no step used). The request says one product has to work with another: "
                            + "; ".join(dep_list) + ". Search the dependent product using each identifier as the phrase, open "
                            "the results, and put in one whose page names one of them (or choose another anchor whose "
                            "identifiers a dependent page names); then order.")
            if self.purchase_sanctioned or self.shifts_remaining() <= 2 or self.protocol.surplus_alt() <= WRAPUP:
                return ""
            self.purchase_sanctioned = True
            cart_faults = self.handcart.defects(self.needs.quantity(), self.needs.headcounts())
            if cart_faults and self.shifts_remaining() > 3:
                return "Not run yet (no step used). The cart, as counted, is not ready: " + "; ".join(cart_faults) + ". Fix it, then order."
            multi = bool(self.needs.items())
            if multi:
                check_ids = list(self.handcart.measures)
            else:
                check_ids = [self.selected()] if self.selected() else []
            for line_id in check_ids:
                ruling_text = self.assess_entry(line_id, "ORDER")
                if ruling_text:
                    return "Not run yet (no step used). " + ruling_text
            for line_id in check_ids:
                rank_note = self.standing_assess(line_id)
                if not rank_note:
                    continue
                cheaper_id = self.standing_ascertain(line_id)      # (v605f FD)
                tail_text = (f" The independent item check judged that {cheaper_id} meets every requirement currently in force and "
                        f"is cheaper than {line_id}: take {line_id} out, put {cheaper_id} in, re-read its stock and price, and order."
                        if cheaper_id else "")
                return "Not run yet (no step used). " + rank_note + tail_text
            line_rule = ("per requested product, each at its quantity" if multi
                        else "the chosen item at the required quantity")
            return ("Not run yet (no step used). Check before ordering: (1) the cart holds exactly one line " + line_rule
                    + ", and nothing more; (2) after your last cart change you re-read this item's stock and price with a tool "
                    "that shows them, it is in stock, and price times quantity fits the current budget; (3) its own properties "
                    "meet every stated requirement, every need the shopper revealed and every announced change; (4) if the "
                    "cheapest or best was asked for, no better fitting item exists among all you could find; (5) every "
                    "procedural rule in the request was followed. If something is unchecked, check it now; otherwise call the "
                    "order tool again.")

        def decline_hold() -> str:
            near_text = self.adjacent_keep()
            if near_text:
                return near_text
            ver = self.needs.iteration
            ok_lines = [line_id for line_id in self.handcart.measures if f"{line_id}|{ver}" in self.succeeded and line_id not in self.rejected_lineitems
                        and line_id not in self.defunct_lineitems()]
            if ok_lines and self.shifts_remaining() > 1 and not inst.get("decline_kept"):
                # (declining with a verified line in the cart is almost never right)
                self.decline_kept = True
                all_lines_txt = " with every product's line in the cart" if self.needs.items() else ""
                return ("Not run yet (no step used). The cart holds " + ", ".join(ok_lines) + ", which the independent "
                        "check judged to meet every requirement currently in force. Declining would discard it: re-read its "
                        "stock and price and order it" + all_lines_txt + ". Decline only if a change since then plainly rules "
                        "it out.")
            slack_ids = self.margin_tablerows() if not inst.get("slack_done") and (self.role("details") or self.bundle_instrument()) else []
            bulk = self.bundle_instrument()
            if slack_ids and self.shifts_remaining() > reads_needed(slack_ids, bulk) + 4 and self.protocol.surplus_alt() > WRAPUP + 40:
                # (ED5: each product has a fit, but the cheapest fits together exceed the budget: cheaper rows within what
                # the budget leaves each product are read before any decline)
                self.slack_done = True
                self.open_queue = slack_ids
                self.open_say = "Before declining, I open cheaper rows that could let every product fit the budget together."
                slack_sum = inst.get("slack_total")
                sum_text = f"{slack_sum:g}" if whether_numeric(slack_sum) else "more than the budget"
                return ("Not run yet (no step used). Each product has a fitting listing, yet the cheapest fits together cost "
                        + sum_text + f", above the budget of {self.needs.outlay_lid}. Rows priced within what the budget leaves "
                        "each product after the other products' cheapest fits were never opened, so their records are being "
                        "read for you now: " + ", ".join(slack_ids) + ". Judge each one against its own product's requirements "
                        "by the value rules (where this store writes the value itself in a field, only those words meet it); "
                        "when fits within the budget together cover every product, put every product's line in the cart and "
                        "order; decline only if no such set exists.")
            bulk_tool = self.bundle_instrument()
            fill_ids = self.absent_tablerows() if not inst.get("decline_fill") and (self.role("details") or bulk_tool) else []
            if fill_ids and self.shifts_remaining() > reads_needed(fill_ids, bulk_tool) + 3 and self.protocol.surplus_alt() > WRAPUP + 40:
                # (a product without a verified fit whose in-budget rows were never opened: they get read before any decline)
                self.decline_fill = True
                self.open_queue = fill_ids
                self.open_say = "Before declining, I open unread listings of the products that still lack a verified fit."
                return ("Not run yet (no step used). Some requested products have no verified fit yet, and listings of their "
                        "categories within the budget were never opened, so their records are being read for you now, ahead "
                        "of any decline: " + ", ".join(fill_ids) + ". Judge each one against its own product's requirements by "
                        "the value rules (where this store writes the value itself in a field, only those words meet it); once "
                        "every product has a fitting listing, put every product's line in the cart and order; decline only "
                        "if some product really has none.")
            named_queue = self.titled_unperused()[: bulk_tool[2] if bulk_tool else 3]
            if named_queue and not inst.get("decline_opens") and (self.role("details") or bulk_tool)\
                    and self.shifts_remaining() > reads_needed(named_queue, bulk_tool) + 3 and self.protocol.surplus_alt() > WRAPUP + 40:
                # (listings the checks named but nobody opened: the code reads their records before any decline)
                self.decline_opens = True
                self.open_queue = named_queue
                return ("Not run yet (no step used). Listings named by the checks were never opened, so their records are "
                        "being read for you now, ahead of any decline: " + ", ".join(named_queue) + ". Judge each one against the "
                        "requirements by the value rules (a field carrying the property's own name decides, and where this "
                        "store writes the value itself there only those words meet it; when there is no such field, another "
                        "field, the options list or the title may show the value), buy one that fits, and decline only if "
                        "every one fails.")
            pages_text = self.decline_reads()
            if pages_text:
                return "Not run yet (no step used). " + pages_text
            fresh_reads = len(self.vault.read) > inst.get("audit_reads", 10 ** 9) and inst.get("audits", 0) < 2
            if fresh_reads and self.abstain_sanctioned and self.shifts_remaining() > 3 and self.protocol.surplus_alt() > WRAPUP + 30:
                # (listings were read after the audit, usually the rows it named: their records get checked as well)
                audit_text = self.scrutiny_abstain()
                if audit_text:
                    return "Not run yet (no step used). " + audit_text
            if not self.abstain_sanctioned and self.shifts_remaining() > 3 and self.protocol.surplus_alt() > WRAPUP + 30:
                self.abstain_sanctioned = True
                audit_text = self.scrutiny_abstain()
                if audit_text:
                    return "Not run yet (no step used). " + audit_text
                return ("Not run yet (no step used). Be sure before declining: you searched in several ways (other words, "
                        "filters, pages), you checked each candidate's own properties, and every one fails a stated "
                        "requirement, a revealed need, the budget or stock. Any key spelling of a property's field counts. "
                        "If something fits, buy it instead; if nothing fits, call the decline tool again.")
            fit_text = self.suit_keep()
            if fit_text:
                return "Not run yet (no step used). " + fit_text
            cands = self.uncontested() if not inst.get("best_held") and self.shifts_remaining() > 4 else []
            rounds = 0
            while rounds < 3:
                # (an unverified pick gets the item check first: a failing one drops out and the next is tried)
                rounds += 1
                to_verify = [line_id for line_id in cands if f"{line_id}|{self.needs.iteration}" not in self.succeeded]
                if not to_verify or self.protocol.surplus_alt() < WRAPUP + 30:
                    break
                self.assess_entry(to_verify[0], "ORDER")
                cands = self.uncontested()
            if cands and self.shifts_remaining() > len(cands) + 3 and self.protocol.surplus_alt() > WRAPUP + 20:
                # (declining is right only when nothing can fit, never because some budget ran out)
                self.best_held = True
                multi = bool(self.needs.items())
                return ("Not run yet (no step used). Declining is right only when no listing can meet the requirements, "
                        "and the records hold " + ("a candidate for each requested product" if multi else "a candidate")
                        + " that is in stock, within the budget and not ruled out by any check: " + ", ".join(cands) + ". "
                        "Unless a record shows one of them failing a requirement currently in force, put "
                        + ("one line per product, at its quantity," if multi else "it") + " in the cart, re-read stock "
                        "and price, and order; decline only if every candidate plainly fails.")
            open_refs = [rf for rf in inst.get("refs") or [] if rf.get("relation") == "differ"
                    and rf.get("status") in ("open", "stuck") and self.reference_active(rf)]
            if open_refs and self.shifts_remaining() > 3 and not inst.get("ref_decline_held"):
                # (CE4: the value of an unresolved differ reference is unknown - it never rules a listing out nor justifies
                # a decline; held once, after the decline's other checks)
                self.ref_decline_held = True
                head_names = ", ".join(dict.fromkeys(rf["head"] for rf in open_refs))
                per_prod = ", one line per product" if self.needs.items() else ""
                return ("Not run yet (no step used). " + head_names + " is set by another listing the code could not resolve, so "
                        "its value is unknown: it never rules a listing out and never justifies a decline. Put the listing "
                        "meeting every other requirement (in stock, within the budget) in the cart" + per_prod
                        + ", re-read its stock and price, and order; decline only if no listing meets the other requirements.")
            for hold_fn in (self.decline_repair, self.near_check):       # (ED6: one meaning check before declining after news)
                hold_text = hold_fn()
                if hold_text:
                    return hold_text
            return self.conclude_stages("decline")

        if finish_kind == "order":
            hold_answer = order_hold() or self.conclude_stages("order")
            return (hold_answer or None), finish_args
        if finish_kind == "decline":
            hold_answer = decline_hold()
            if hold_answer:
                return hold_answer, finish_args
            return None, self.cleanse_abstain(finish_args, finish_tool)
        return None, finish_args

    def change_review(self) -> str:
        """Once a store notice, a refusal or an unasked shopper line has changed the requirements or the budget: a single
        cheap model call re-reads the listings opened so far against the conditions now in force and says which fit,
        which to take and whether the cart still holds; the actor is told to act on that first. Three at most per errand."""
        prior_n = self.__dict__.get("reviews", 0)

        def skip_review() -> bool:
            if prior_n >= 3 or not self.vault.read or self.protocol.silence:
                return True
            if self.protocol.surplus_alt() < WRAPUP + 20 or not self.protocol.viable(ASSESSOR):
                return True
            return self.shifts_remaining() < 3
        if skip_review():
            return ""
        self.reviews = prior_n + 1
        card_lines = self.vault.profiles(14000, "", self.reference_external())
        if not card_lines:
            return ""
        prio_detail = self.needs.perusal.get("priority_detail") or ""
        blocks = [
            ALTERATION_SCRUTINY,
            "\n\nTHE SHOPPER'S REQUEST AS WRITTEN:\n" + self.request + self.needs.alterations(),
            "\n\nTHE CHANGE JUST HEARD: " + " / ".join(self.needs.dialogues[-2:]),
            "\n\nREQUIREMENTS IN FORCE: " + self.needs.passage_payload(),
            f"\n\nPRIORITY: {self.needs.priority()} {prio_detail}".rstrip(),
            "\n\nCART AT PRESENT: " + self.handcart.recap_alt(),
            "\n\nLISTINGS SEEN (DETAILS = its full record was read, ROW = met only in search rows):\n" + "\n".join(card_lines),
        ]
        ask = "".join(blocks) + self.attribute_monikers()
        verdict_obj = self.protocol.confer(ask, 2500, ASSESSOR)
        self.journal.append(dict(kind="change_review", answer=verdict_obj))
        if not verdict_obj:
            return ""
        known_ids = self.vault.ids

        def id_of(xv: Any) -> str:
            return str(xv) if isinstance(xv, (str, int)) and str(xv) in known_ids else ""
        fit_ids = [id_of(xv) for xv in verdict_obj.get("fits") or []]
        fit_ids = [fv for fv in fit_ids if fv][:6]
        take_id = id_of(verdict_obj.get("pick"))
        self.__dict__.setdefault("named", set()).update(fit_ids)
        if fit_ids:
            fit_text = "the opened listings meeting every requirement currently in force, best first: " + ", ".join(fit_ids)
        else:
            fit_text = "no opened listing is shown to meet every requirement currently in force"
        pieces = ["[review after the change]", fit_text]
        if take_id:
            pieces.append(f"take this one now: {take_id}")
        cart_ok = verdict_obj.get("cart_still_right")
        if self.handcart.measures and cart_ok is True:
            pieces.append("the cart's line still fits: keep it, so the basket already described is the one ordered")
        elif self.handcart.measures and cart_ok is False:
            pieces.append("under the conditions now the cart's line is no longer the right pick: replace it (when the "
                         "instructions ask you to check with the shopper before replacing anything, ask first)")
        if verdict_obj.get("reason"):
            pieces.append(str(verdict_obj["reason"])[:300])
        pieces.append("Act on this before anything else: choose again among what fits now (the cheapest or best of those "
                     "when the request asks for that), fix the cart, re-read its stock and price, then finish; never decline "
                     "while an opened listing fits.")
        return " ".join(pieces)

    def engine_inquiries_substitute(self, drop_ids: list[str], add_ids: list[str]) -> bool:
        """Puts one question to the shopper about a planned cart replacement (the stated ask-before-replacing step);
        True when it went out and got an answer, False when no message or step remains for it."""
        asker = self.role("message")
        text_field = self.passage_param(asker) if asker else None
        if not text_field:
            return False
        if self.inquire_threshold() - self.letters_transmitted < 1 + self.letter_buffer() or self.shifts_remaining() < 4:
            return False
        ask_text = "Before I change the basket: I would like to take out {} and put in {} instead, since the first no longer fits. Is that all right with you?".format(
            ", ".join(drop_ids[:3]), ", ".join(add_ids[:3]))
        answers_back = self.transmit_segment([(f"code_swap_ask_{self.protocol.pulse}", asker, {text_field: ask_text})])
        self.journal.append({"kind": "code_swap_ask", "old": drop_ids[:3], "new": add_ids[:3]})
        if not answers_back:
            return False
        return not str(answers_back[0][1]).startswith("Not run")

    def lineitem_vendor(self, line_id: str) -> str:
        """The seller a cart line's own record names, as written ('' if none does)."""
        seller_fields = self.vendor_slots()
        sellers: list[str] = []

        def visit_rec(node: dict[str, Any]) -> None:
            ids_here = [fval for fkey, fval in node.items() if isinstance(fval, str) and "id" in str(fkey).lower()]
            if line_id not in ids_here:
                return
            for fkey, fval in node.items():
                if not isinstance(fval, str) or not fval.strip() or len(fval) > 60:
                    continue
                kl = str(fkey).lower()
                if "id" in kl:
                    continue
                if re.search(VENDOR_SLOTS, kl) or str(fkey) in seller_fields:
                    sellers.append(fval.strip())
        for rec in self.vault.items:
            traverse_mappings(rec, visit_rec)
        return sellers[-1] if sellers else ""

    def sheet_passage(self, listing_id: str) -> str:
        """All the text a listing's records hold (title, about text, notes, fact values), joined together."""
        texts_all: list[str] = []

        def visit_rec(node: dict[str, Any]) -> None:
            ids_here = [fval for fkey, fval in node.items() if isinstance(fval, str) and "id" in str(fkey).lower()]
            if listing_id in ids_here:
                lengthy_charseqs_whole(node, texts_all)
        for rec in self.vault.items:
            traverse_mappings(rec, visit_rec)
        return " ".join(dict.fromkeys(texts_all))

    @staticmethod
    def identity_lexemes(blob: str) -> list[str]:
        """The identifier-like tokens of a text, as written: a word followed by a figure (a model name and its number)."""
        tokens = list(dict.fromkeys(re.findall(r"\b([A-Z][A-Za-z]{1,12}[ -]?\d{3,6})\b", blob)))
        return tokens[:8]

    @staticmethod
    def compress(raw_text: str) -> str:
        low = str(raw_text).lower()
        return "".join(re.findall(r"[a-z0-9]", low))

    def prerequisite_defects(self) -> list[str]:
        """For each product the reading marks as having to work with another: when both sit in the cart and the dependent
        line's page names none of the anchor line's identifiers, a sentence saying so ([] when everything agrees)."""
        rows_p = self.needs.items()
        if not rows_p or all(rw.get("depends_on") is None for rw in rows_p):
            return []
        held_by: dict[int, str] = {}
        for line in self.handcart.measures:
            pidx = self.ware_position(line)
            if pidx is not None and pidx not in held_by:
                held_by[pidx] = line
        out_faults = []
        for pidx, prow in enumerate(rows_p):
            anchor_idx = prow.get("depends_on")
            if anchor_idx is None or anchor_idx == pidx or anchor_idx >= len(rows_p) or pidx not in held_by or anchor_idx not in held_by:
                continue
            anchor_id, dep_id = held_by[anchor_idx], held_by[pidx]
            id_tokens = self.identity_lexemes(self.sheet_passage(anchor_id))
            if not id_tokens:
                continue
            dep_flat = self.compress(self.sheet_passage(dep_id))
            if any(self.compress(tok) in dep_flat for tok in id_tokens):
                continue
            out_faults.append(f"the {prow['product']} ({dep_id}) has to work with the {rows_p[anchor_idx]['product']} "
                            f"({anchor_id}), yet its page names none of that listing's identifiers ({', '.join(id_tokens)})")
        return out_faults

    def prerequisite_clue(self) -> str:
        """Status text naming, for each dependent product whose anchor is already a cart line, the anchor identifiers to search by."""
        rows_p = self.needs.items()
        if not rows_p or all(rw.get("depends_on") is None for rw in rows_p):
            return ""
        hints = []
        for pidx, prow in enumerate(rows_p):
            anchor_idx = prow.get("depends_on")
            if anchor_idx is None or anchor_idx == pidx or anchor_idx >= len(rows_p):
                continue
            anchor_id = next((line for line in self.handcart.measures if self.ware_position(line) == anchor_idx), "")
            id_tokens = self.identity_lexemes(self.sheet_passage(anchor_id)) if anchor_id else []
            own_kind, anchor_kind = prow["product"], rows_p[anchor_idx]["product"]
            if id_tokens:
                hints.append(f"the {own_kind} has to work with the {anchor_kind} in the cart ({anchor_id}): search the {own_kind} "
                                 f"using each of its identifiers as the phrase ({', '.join(id_tokens)}), open the results, and "
                                 "pick one whose page names one of them")
            else:
                hints.append(f"the {own_kind} has to work with the {anchor_kind}: settle that one first, then search the "
                                 f"{own_kind} by its model names or numbers and take a page naming one of them")
        if not hints:
            return ""
        return "\n[must work together] " + "; ".join(hints)

    def cheap_sweep(self) -> str:
        """With a cheapest rule, once, before the first order of a one-product request that has a line in the cart: at
        most two short searches the errand never ran (the category's kind words, singular first); the in-stock,
        in-budget rows of the requested category priced under the cart's line and never opened then go to the actor to
        open first. '' when nothing cheaper showed up or steps are short."""
        if self.needs.priority() != "cheapest" or self.needs.items() or self.purchases_attempted or self.__dict__.get("cheap_swept"):
            return ""
        search_tool = self.role("search")
        if not (search_tool and self.handcart.measures) or self.shifts_remaining() < 12 or self.protocol.surplus_alt() < WRAPUP + 40:
            return ""
        self.cheap_swept = True
        schema = self.structures.get(search_tool) or {}
        fields = schema.get("properties") or {}
        needed = set(schema.get("required") or [])
        str_keys = [fk for fk, fv in fields.items() if isinstance(fv, dict) and fv.get("type") == "string" and not fv.get("enum")]
        if len(str_keys) != 1 or needed - {str_keys[0]}:
            return ""
        qkey = str_keys[0]
        ran_before = {" ".join(self.loose_passage(rec["args"])).strip().lower() for rec in self.__dict__.get("page_book", {}).values()}
        new_queries = [qq for qq in self.tag_searchterms() if " " not in qq and qq not in ran_before][:2]
        if not new_queries:
            return ""
        now_tick = self.protocol.pulse
        self.transmit_segment([(f"cheap_{num}_{now_tick}", search_tool, {qkey: qq}) for num, qq in enumerate(new_queries)])
        line0 = next(iter(self.handcart.measures))
        line_price = self.vault.price(line0)
        if line_price is None:
            return ""
        cheaper_rows = [(c_price, rid, rtitle) for rid, (c_price, rtitle, _) in self.rack_tablerows().items()
                 if c_price < line_price and rid not in self.handcart.measures and rid not in self.vault.read]
        cheaper_rows = sorted(cheaper_rows)[:4]
        self.journal.append({"kind": "cheap_sweep", "queries": new_queries, "rows": [rid for _, rid, _ in cheaper_rows]})
        if not cheaper_rows:
            return ""
        self.__dict__.setdefault("named", set()).update(rid for _, rid, _ in cheaper_rows)
        rows_text = "; ".join(f"{rid} '{rtitle[:70]}' {c_price:g}" for c_price, rid, rtitle in cheaper_rows)
        return ("Not run yet (no step used). The cheapest fit is wanted, so the code searched " + ", ".join(new_queries)
                + ": these in-stock rows of the requested category are cheaper than the cart's line and were never opened: "
                + rows_text + ". Open them (several in one reply), and if one meets every requirement, make it the cart's "
                "line, re-read its stock and price, and order; otherwise order the cart as it stands.")

    def boundary_clue(self) -> str:
        """With a cheapest rule, for each requirement in force that sets a limit by a figure with a unit: a reminder to
        search the listings writing that figure (in the store's spellings, with and without a space) together with the
        kind word, two pages each, before the pick is settled. '' otherwise."""
        if self.needs.priority() != "cheapest" or self.protocol.surplus_alt() < WRAPUP:
            return ""
        fig_re = r"(\d+(?:\.\d+)?)\s?([a-z\"']{1,4})\b"
        all_figs = [mm.group(1) + mm.group(2) for rq in self.needs.traits_remaining()
                 for mm in re.finditer(fig_re, str(rq).split(":", 1)[-1].lower())]
        fig_list = list(dict.fromkeys(all_figs))[:3]
        if not fig_list:
            return ""
        return ("\n[cheapest under a measured limit] the cheapest fit often writes the limit's own figure: search "
                + ", ".join(fig_list) + " with the kind word (with and without a space before the unit), read two pages of "
                "each, and open the cheapest in-budget rows before settling the pick.")

    def hamper_passage(self) -> str:
        """A plain account of the cart for the shopper, taken from the records: each line, its count and price, and the total."""
        try:
            rows_info = self.engine_tablerows()
        except Exception:
            rows_info = {}
        line_texts: list[str] = []
        sum_cost = 0.0
        priced_all = True
        for line_id, qty in self.handcart.measures.items():
            info_row = rows_info.get(line_id) or {}
            unit_price = self.vault.price(line_id)
            if unit_price is None:
                unit_price = info_row.get("price")
            shown_n = int(qty) if float(qty).is_integer() else qty
            suffix = f" ({info_row['title'][:60]})" if info_row.get("title") else ""
            try:
                seller_name = self.lineitem_vendor(line_id)
            except Exception:
                seller_name = ""
            if seller_name:
                suffix += f" from {seller_name}"
            if unit_price is not None:
                sum_cost += float(unit_price) * float(qty)
                line_texts.append(f"{shown_n} x {line_id}{suffix} at {unit_price:g} each")
            else:
                priced_all = False
                line_texts.append(f"{shown_n} x {line_id}{suffix}")
        cur = str(self.needs.perusal.get("currency") or "").strip()
        if not line_texts:
            return "Before I finish: the basket is empty, so nothing will be ordered."
        msg_text = "Before I finish, here is the basket: " + "; ".join(line_texts) + "."
        if priced_all:
            msg_text += " The total is {:g}{}.".format(sum_cost, " " + cur if cur else "")
        return msg_text + " I will place the order now."

    def conclude_stages(self, finish_kind: str) -> str:
        """Just before a finish goes out: a cheap model call reads the instructions in force, the newest shopper lines and
        notices, and what was done after the last cart change, and names any stated step not yet done; the code carries
        those out (a message to the shopper, or another listed tool) and the finish then goes ahead. With no instruction
        on the ledger and nothing heard from the shopper or the store, no check runs. One check per cart state and
        ledger version. When no model answers, a stated finishing message not sent since the last cart change is written
        by the code from the records."""
        nothing_heard = not self.needs.informed and not self.needs.dialogues and not self.__dict__.get("event_seen")
        if self.shifts_remaining() < 2 or self.protocol.done or self.protocol.crashed:
            return ""
        cart_changes = self.__dict__.get("cart_edits", 0)
        state_key = (finish_kind, tuple(sorted(self.handcart.measures.items())), cart_changes, self.needs.informed_iteration, self.rejection_num, self.needs.iteration)
        seen_keys = self.__dict__.setdefault("finish_checked", set())
        if state_key in seen_keys:
            return ""
        seen_keys.add(state_key)
        message_tool = self.role("message")
        messages_left = max(0, self.inquire_threshold() - self.letters_transmitted) if message_tool else 0
        after_change = []
        for at_step, text_said, edit_mark in self.__dict__.get("msg_log", []):
            if edit_mark == cart_changes and (cart_changes > 0 or not self.handcart.measures):
                after_change.append(f"- message sent at step {at_step}: {text_said}")
        rules = self.needs.informed
        basket_tool = self.role("cart")
        tool_defs = []
        for td in self.toolkit:
            tool_nm = td["function"]["name"]
            if tool_nm in self.concluders() or tool_nm == basket_tool:
                continue
            if messages_left <= 0 and tool_nm == message_tool:
                continue
            tool_defs.append(td)
        rules_text = "\n".join(f"- {tl['text']} (from the {tl.get('source')})" for tl in rules)
        tools_text = "\n".join(
            f"- {td['function']['name']}: {td['function'].get('description', '')} | arguments: "
            f"{into_passage(td['function'].get('parameters') or {})}" for td in tool_defs)
        ask = "".join([
            CONCLUDE_ASSESS,
            "\n\nINSTRUCTIONS NOW IN FORCE:\n" + (rules_text or "(none stated)"),
            "\n\nTHE SHOPPER'S REQUEST AS WRITTEN:\n" + self.request,
            "\n\nLATEST SHOPPER LINES AND STORE NOTICES (oldest first):\n" + ("\n".join(self.needs.dialogues[-8:]) or "(none)"),
            "\n\nREQUIREMENTS IN FORCE: " + self.needs.passage_payload(),
            "\n\nCART AT PRESENT: " + self.handcart.recap_alt() + " " + self.hamper_passage(),
            "\n\nTHE FINISH ABOUT TO GO OUT: " + finish_kind,
            "\n\nDONE SINCE THE LAST CART CHANGE:\n" + ("\n".join(after_change) or "(nothing was sent or done after the last cart change)"),
            f"\n\nMESSAGES LEFT: {messages_left}\nSTEPS LEFT: {self.shifts_remaining()} (the finish itself uses one)",
            "\n\nTOOLS ABLE TO CARRY OUT A STEP:\n" + tools_text,
        ])
        reply_obj = None
        if not nothing_heard and self.protocol.viable(CONCLUDER) and self.protocol.surplus_alt() > FINAL_INVOKE + 10:
            reply_obj = self.protocol.confer(ask, 2500, CONCLUDER)
        todo: list[tuple[str, dict[str, Any], str]] = []
        if reply_obj is not None:
            names_ok = {td["function"]["name"] for td in tool_defs}
            for entry in (reply_obj.get("pending") or [])[:CONCLUDE_CYCLES]:
                if not isinstance(entry, dict) or entry.get("tool") not in names_ok or not isinstance(entry.get("args"), dict):
                    continue
                clean_args, problems = ParamAssess.carryout(dict(entry["args"]), self.structures[entry["tool"]])
                if not problems:
                    todo.append((entry["tool"], clean_args, str(entry.get("why") or "")[:120]))
        text_field = self.passage_param(message_tool) if message_tool else None
        msg_steps = sum(1 for ent in todo if ent[0] == message_tool)
        if finish_kind == "order" and self.handcart.measures and text_field and not after_change and messages_left > msg_steps and not msg_steps:
            # (the basket is told to the shopper after the last cart change and before the order: a told basket is never
            # wrong, and it costs one step and the message kept back for it)
            todo.append((message_tool, {text_field: self.hamper_passage()}, "the basket told before the order"))
        self.journal.append(dict(kind="finish_check", finish=finish_kind, answer=reply_obj, calls=[ent[0] for ent in todo]))
        todo = todo[: max(0, self.shifts_remaining() - 1)]
        if not todo:
            return ""
        at_step = self.protocol.pulse
        props_list = [{"id": f"finish_step_{at_step}_{idx}", "type": "function",
                      "function": {"name": tool_n, "arguments": json.dumps(step_args)}} for idx, (tool_n, step_args, _) in enumerate(todo)]
        # (the steps run within the finish call's own turn: what they did and what came back goes into that call's result,
        # so each call in the chat keeps its own result right after it)

        def conditions() -> tuple:
            return (self.needs.iteration, self.needs.informed_iteration, self.letter_lid, self.stage_lid, self.rejection_num, len(self.handcart.measures))
        cond_before = conditions()
        flag_was = self.__dict__.get("scripted_batch")
        try:
            results = self.route_alt(props_list, scripted=True)
        finally:
            self.scripted_batch = flag_was
        moved_on = cond_before != conditions()
        status_cut = chr(10) + "[status]"
        done_lines = [f"{tool_n} {json.dumps(step_args, ensure_ascii=False)[:400]} -> {reply_text.split(status_cut)[0][:700]}"
                for (tool_n, step_args, _), (_, reply_text) in zip(todo, results)]
        finish_text = "[the code ran these before this finish: " + "; ".join(ent[2] for ent in todo) + "] " + " | ".join(done_lines)
        self.journal.append(dict(kind="finish_steps_done", note=finish_text[:600]))
        asked_shopper = bool(message_tool) and any(tool_n == message_tool for tool_n, _, _ in todo)
        if asked_shopper and (moved_on or self.needs.response_owed or self.protocol.done):
            # (the shopper's answer changed something or asks for something: the actor reads it before any finish goes)
            return ("Not run yet (no step used). " + finish_text + "\nThe answer changed something: check it against the "
                    "requirements and the instructions in force, do what it asks, and finish again.")
        self.finish_note = finish_text
        return ""

    def route_alt(self, batch_calls: list[dict[str, Any]], scripted: bool = False) -> list[tuple[str, str]]:
        answers: list[tuple[str, str]] = []
        self.scripted_batch = scripted
        self.commence_bundle()
        open_tools = {td["function"]["name"] for td in self.proposed_alt()}
        basket_tool = self.role("cart")
        to_run: list[tuple[str, str, dict[str, Any]]] = []
        msgs_planned = 0
        cart_stopped = False
        self.batch_puts = []
        self.batch_cart = any((pc.get("function") or {}).get("name") == basket_tool for pc in batch_calls if isinstance(pc, dict))
        for call_d in batch_calls:
            func_spec = call_d.get("function") or {}
            if func_spec.get("name") != basket_tool:
                continue
            arg_blob = func_spec.get("arguments")
            if isinstance(arg_blob, dict):
                parsed_args = arg_blob
            elif isinstance(arg_blob, str) and arg_blob.strip():
                parsed_args = serialized_entity(arg_blob)
            else:
                parsed_args = {}
            if isinstance(parsed_args, dict) and self.handcart.whether_place(parsed_args) and self.handcart.aim(parsed_args):
                self.batch_puts.append(self.handcart.aim(parsed_args))
        for call_d in batch_calls:
            call_id = call_d["id"]
            if cart_stopped and call_d["function"]["name"] == basket_tool:
                # (once a cart change was held, the later ones in the same reply are stale)
                answers.append((call_id, "Not run (no step used): an earlier cart change in this reply did not run; read the reason, "
                                     "then resend the cart changes."))
                continue
            why_not, tool_nm, args_ok = self.sift(call_d, open_tools, scripted, len(batch_calls), msgs_planned)
            if not why_not:
                if tool_nm == self.role("message"):
                    msgs_planned += 1
                to_run.append((call_id, tool_nm, args_ok))
                continue
            if not cart_stopped:
                cart_stopped = (tool_nm == basket_tool and isinstance(args_ok, dict) and bool(args_ok)
                                and not self.handcart.whether_place(args_ok))
            answers.append((call_id, why_not))
        if to_run and to_run[0][1] in self.concluders():
            rid, tool_nm, args_ok = to_run[0]
            hold_text, args_ok = self.sift_conclude(rid, tool_nm, args_ok)
            if hold_text:
                answers.append((rid, hold_text))
                return answers
            to_run[0] = (rid, tool_nm, args_ok)
        if to_run and self.concluders() and to_run[0][1] not in self.concluders():
            keep_n = max(0, self.shifts_remaining() - 1)
            answers.extend((rid, "Not run (no step used): just the last step remains, and it has to be the order (or the "
                                 "decline).") for rid, _, _ in to_run[keep_n:])
            to_run = to_run[:keep_n]
        for offset in range(0, len(to_run), 16):
            chunk = to_run[offset: offset + 16]
            if self.protocol.done or self.protocol.crashed or self.shifts_remaining() <= 0:
                answers.extend((rid, "Not run: the episode has ended or no steps remain.") for rid, _, _ in chunk)
                continue
            answers.extend(self.transmit_segment(chunk[: self.shifts_remaining()]))
        finish_text = self.__dict__.pop("finish_note", "")
        if finish_text and to_run and to_run[0][1] in self.concluders():
            first_id = to_run[0][0]
            answers = [(rid, finish_text + "\n" + body if rid == first_id else body) for rid, body in answers]
        order_of = {cd["id"]: pos for pos, cd in enumerate(batch_calls)}
        answers.sort(key=lambda resp_pair: order_of.get(resp_pair[0], 0))
        return answers

    def transmit_segment(self, batch_slab: list[tuple[str, str, dict[str, Any]]]) -> list[tuple[str, str]]:
        actions = [{"name": tool_nm, "args": arg_d} for _, tool_nm, arg_d in batch_slab]
        tick_before = self.protocol.pulse
        env_reply = self.protocol.perform(actions)
        self.journal.append({"kind": "env", "step": tick_before + 1, "actions": actions, "reply": env_reply})
        if env_reply is None:
            return [(rid, "The call failed (network or server trouble). Try it again.") for rid, _, _ in batch_slab]
        obs_seq = env_reply.get("observations")
        if not isinstance(obs_seq, list):
            obs_seq = [env_reply.get("observation")]
        last_idx = len(obs_seq) - 1
        results = []
        for pos, (rid, tool_nm, arg_d) in enumerate(batch_slab):
            if pos > last_idx:
                results.append((rid, "Not run: the episode ended before this call."))
            else:
                results.append((rid, self.soakup(tool_nm, arg_d, obs_seq[pos], env_reply, pos == last_idx)))
        return results

    def soakup(self, tool_nm: str, args_in: dict[str, Any], seen: Any, env_reply: dict[str, Any], is_last: bool) -> str:
        """Books one answered call and gives back the text the actor reads for it."""
        shown = seen
        seen = focused(seen)
        message_tool = self.role("message")
        cart_tool = self.role("cart")
        inst = self.__dict__
        if tool_nm == message_tool:
            self.letters_transmitted += 1
            sent_text = " ".join(str(fv) for fv in args_in.values() if isinstance(fv, str))
            inst.setdefault("msg_log", []).append((self.protocol.pulse, sent_text[:600], inst.get("cart_edits", 0)))
            if not inst.get("scripted_batch"):
                self.enquiries_journal.append(f"- (the assistant's own message): \"{sent_text[:300]}\" -> {into_passage(seen)[:300]}")
        if tool_nm == cart_tool:
            self.cart_edits = inst.get("cart_edits", 0) + 1       # (what 'after the last cart change' counts from)
        self.last_kind = self.invoke_category(tool_nm, args_in)
        if self.last_kind == "order":
            self.purchases_attempted += 1
        if tool_nm in self.concluders() or tool_nm in (cart_tool, message_tool):
            self.final_invokes = []
        else:
            history = self.final_invokes + [tool_nm + into_passage(args_in)]
            self.final_invokes = history[-2:]
        self.vault.add(seen)
        self.absorb_n = inst.get("absorb_n", 0) + 1
        if not inst.get("covering") and tool_nm != cart_tool:
            self.since_cart = inst.get("since_cart", 0) + 1      # (CB6: calls of the actor's own since the newest cart call)
        try:
            sold_now = self.discover_lids(tool_nm, args_in, seen)
        except Exception as exc:
            sold_now = []
            self.journal.append({"kind": "caps_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
        if self.handcart.measures:
            # (when each cart line's own record was last seen: a line unseen since the last cart change is re-read; CB3:
            # once any record has shown a stock field, only a record holding the line's id and a stock field counts - a
            # search row only once rows have shown the current stock and have never been seen lagging the lookups)
            seen_lines: set[str] = set()
            stock_strict = bool(inst.get("stock_seen"))
            asked_set = self.inquired_uids(args_in)
            rows_fresh = bool(inst.get("rows_live")) and not inst.get("rows_lag")
            lookup_only = not rows_fresh and tool_nm not in (cart_tool, self.role("cart_view"))

            def mark_line(node: dict[str, Any]) -> None:
                cart_ids = {fv for fk, fv in node.items() if isinstance(fv, str) and fv in self.handcart.measures and "id" in str(fk).lower()}
                if not cart_ids:
                    return
                if stock_strict:
                    counts_ok = self.inventory_condition(node) is not None and not (lookup_only and not cart_ids & asked_set)
                else:
                    counts_ok = any(isinstance(fv, dict) for fv in node.values())
                if counts_ok:
                    seen_lines.update(fv for fk, fv in node.items() if isinstance(fv, str) and fv in self.handcart.measures)
            traverse_mappings(seen, mark_line)
            if seen_lines:
                read_book = inst.setdefault("line_read", {})
                seq_book = inst.setdefault("line_seq", {})
                for line_id in seen_lines:
                    read_book[line_id] = self.protocol.pulse
                    seq_book[line_id] = self.absorb_n
        detail_tool, find_tool = self.role("details"), self.role("search")
        if tool_nm == detail_tool and not (isinstance(seen, dict) and seen.get("error")):
            self.vault.read.update(fv for fv in args_in.values() if isinstance(fv, str) and fv in self.vault.ids)
        # (v605d M8, CG5: any other single-listing lookup counts like a details read - except the stock re-read of a cart
        # line or of a verified fit)
        other_lookup = bool(detail_tool) and tool_nm not in (detail_tool, find_tool) and tool_nm in self.retrieve_instruments()
        arg_ids = {fv for fv in args_in.values() if isinstance(fv, str)}

        def is_late() -> bool:
            if self.shifts_remaining() >= self.purchase_owed():
                return False
            if tool_nm not in (detail_tool, find_tool) and not other_lookup:
                return False
            if inst.get("covering"):
                return False
            if tool_nm == detail_tool and arg_ids & set(self.handcart.measures):
                return False
            return not (other_lookup and arg_ids & (set(self.handcart.measures) | {fk.rpartition("|")[0] for fk in self.succeeded}))
        if is_late():
            self.late_n = inst.get("late_n", 0) + 1
        notes = []
        if sold_now:
            notes.append("[stock] " + self.defunct_annotation(sold_now))
        try:
            notes_text = self.annotations_comment(tool_nm, args_in)
            if notes_text:
                notes.append("[notes] " + notes_text)
        except Exception as exc:
            self.journal.append({"kind": "notes_remark_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
        if tool_nm == cart_tool:
            self.purchase_sanctioned = False
            self.handcart.altered_when = self.protocol.pulse
            cart_before = dict(self.handcart.measures)
            notes.append(self.handcart.enrolled(args_in, seen))
            try:
                self.carriage_shifted(cart_before)
                self.flag_recent(seen)
                self.following_place(args_in, seen)
            except Exception as exc:
                self.journal.append({"kind": "after_put_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
            put_id = self.handcart.aim(args_in)
            puts_in = bool(put_id) and self.handcart.whether_place(args_in)
            if puts_in:
                self.rejections_observed[put_id] = self.rejection_num
                put_counts = inst.setdefault("puts", {})
                put_counts[put_id] = put_counts.get(put_id, 0) + 1
            # (v605e F5: a removal shares the put's form - so the remark goes only with a call that puts units in, while
            # the line sits in the cart and is not sold out)
            put_n = self.handcart.headcount_param(args_in)
            if puts_in and f"{put_id}|{self.needs.iteration}" in self.succeeded and (put_n is None or put_n > 0)\
                    and self.handcart.retained(put_id) > 0 and put_id not in self.defunct_lineitems():
                notes.append(f"An independent check judged that {put_id} meets every requirement currently in force, "
                               "reading each property by the value rules (where this store writes the value itself in a "
                               "field, only those words meet it); keep it unless a new change rules it out.")
        if tool_nm == self.role("cart_view"):
            self.handcart.inspected(seen, self.protocol.pulse)
            notes.append(self.handcart.recap_alt())
        was_refused = self.refused(tool_nm, seen, env_reply if is_last else {"finished": False})
        if was_refused and self.last_kind == "order":
            self.cart_seq = self.absorb_n      # (CB3: after a refused checkout every line's stock gets read again before the next)
        notes.append(self.annotations_toward(tool_nm, seen, was_refused))
        if tool_nm == find_tool:
            notes.append(self.seek_annotation(args_in, seen))
            inst.get("named_pages", set()).discard(json.dumps(args_in, sort_keys=True, default=str))
            try:
                notes.append(self.merge_joined(seen))
            except Exception as exc:
                self.journal.append({"kind": "merge_joined_error", "error": f"{type(exc).__name__}: {exc}"[:300]})

        def should_audit() -> bool:
            if not is_last or inst.get("early_audit"):
                return False
            if tool_nm in (cart_tool, message_tool) or tool_nm in self.concluders():
                return False
            if not INITIAL_PURCHASE_VIA < self.shifts_remaining() <= self.stage_lid // 3 + 2 or self.handcart.measures:
                return False
            ver_tag = str(self.needs.iteration)
            return not any(pk.rpartition("|")[2] == ver_tag for pk in self.succeeded)
        if should_audit():
            # (nothing verified while a third of the steps remain: the decline audit runs now, while its rows can still be opened)
            self.early_audit = True
            audit_text = self.scrutiny_abstain(early=True)
            if audit_text:
                notes.append("No verified fit yet. " + audit_text)
        ask_text = " ".join(str(fv) for fv in args_in.values() if isinstance(fv, str)) if tool_nm == message_tool else None
        heard = self.customer_lineitems(tool_nm, seen, was_refused)
        if heard:
            ver_was = self.needs.iteration
            loose_was = len(self.needs.let_go)
            cap_was = self.needs.outlay_lid
            notes.append(self.needs.receive(heard, ask_text))
            self.reference_adopt()
            allowance = self.needs.__dict__.get("allowance_news")
            if allowance:
                self.needs.allowance_news = None
                try:
                    self.enact_entitlement(allowance)
                except Exception as exc:
                    self.journal.append({"kind": "allowance_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
            if inst.get("told_seen") != self.needs.informed_iteration:
                self.told_seen = self.needs.informed_iteration
                self.purchase_sanctioned = False       # (a new instruction: the finish gets checked again before it goes)
            unprompted = ask_text is None or len(self.needs.let_go) > loose_was or self.needs.outlay_lid != cap_was
            if self.needs.iteration > ver_was and unprompted:
                # (a notice or a refusal changed the conditions: the opened listings are judged again before the next move)
                try:
                    review_text = self.change_review()
                    if review_text:
                        notes.append(review_text)
                except Exception as exc:
                    self.journal.append({"kind": "change_review_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
        clip_n = SEEK_PASSAGE if tool_nm == self.role("search") else CHRONICLE_PASSAGE
        text_out = into_passage(truncated(shown, clip_n))
        if is_last and env_reply.get("finished"):
            text_out += " [the episode has ended]"
        note_text = " ".join(rm for rm in notes if rm)
        return text_out + "\n[note] " + note_text if note_text else text_out

    # ----------------------------------------------------------------------------------------------- the loop
    def tool_calls(self, reply_msg: dict[str, Any], counter: int) -> tuple[list[dict[str, Any]], int]:
        calls_out = []
        for tc in reply_msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            counter += 1
            func_spec = tc.get("function") or {}
            arg_text = func_spec.get("arguments")
            if not isinstance(arg_text, str):
                arg_text = json.dumps(arg_text or {})
            cid = tc.get("id") or f"call_{counter}"
            calls_out.append({"id": cid, "type": "function", "function": {"name": func_spec.get("name"), "arguments": arg_text}})
        return calls_out, counter

    def carryout(self) -> list[dict[str, Any]]:
        if self.protocol.done:
            return self.journal
        self.peruse_petition()
        self.journal.append({"kind": "plan", "plan": self.needs.perusal, "roles": self.duty_table})
        opening_obs = self.policyview.get("opening")
        first_msg = "THE SHOPPER'S REQUEST:\n" + self.request
        if opening_obs is not None:
            first_msg += "\n\nFIRST OBSERVATION:\n" + into_passage(opening_obs)
            self.vault.add(opening_obs)
        brief = self.needs.synopsis(self.stage_lid, self.letter_lid)
        if brief:
            first_msg += "\n\n" + brief
        first_msg += "\n\n" + self.progress_lineitem()
        self.converse = [{"role": "system", "content": PERFORM_PRINCIPLES}, {"role": "user", "content": first_msg}]
        idle_turns = 0
        hard_fails = 0
        call_seq = 0
        last_ok = _TALLY.now()
        misses_n = 0

        def code_moved() -> bool:
            # (the code's own moves, in priority order; True when one of them took the turn)
            if self.examine_titled():
                return True
            if (not self.inquiring or self.__dict__.get("empty_plans")) and self.survey_stage():
                return True
            if self.value_pages():
                return True
            if self.inquiring and self.present_intended():
                return True
            return bool(self.reference_stage() or self.traverse_stage())
        for _ in range(CYCLE_THRESHOLD):
            if self.protocol.done or self.protocol.crashed or self.shifts_remaining() <= 0 or self.protocol.surplus_alt() < FINAL_INVOKE:
                break
            self.intercept_upward()
            if self.needs.response_owed:
                due_text, self.needs.response_owed = self.needs.response_owed, ""
                if self.response_returned(due_text):
                    continue
            if code_moved():
                continue
            models = PERFORMER[1:] + PERFORMER[:1] if self.stuck >= STALLED_THRESHOLD else PERFORMER
            reply = self.protocol.searchterm_llm(self.converse, models, 2500, self.toolkit)
            if reply is None:
                # (TA4: replies with no content, or calls refused outright, still end the loop at the second; a call load or
                # a lost network does not - the models count as gone only when none can be called, or no actor reply came
                # for DOWN_AFTER seconds across two failed turns or across one that tried two models)
                misses_n += 1
                silent_too_long = _TALLY.now() - last_ok > LOWERED_FOLLOWING and (
                    misses_n >= 2 or (self.protocol.lowered_alt() and len(self.protocol.invoke_llms) >= 2))
                if not self.protocol.viable(models, True) or silent_too_long:
                    break
                if self.protocol.invoke_breakdown in ("empty", "refused"):
                    hard_fails += 1
                    if hard_fails >= 2:
                        break
                continue
            last_ok = _TALLY.now()
            misses_n = 0
            calls, call_seq = self.tool_calls(reply, call_seq)
            text_part = reply.get("content") or ""
            if not isinstance(text_part, str):
                text_part = into_passage(text_part)
            binding = {self.role("cart"), self.role("order"), self.role("decline")} - {None}
            binds = any(cd["function"]["name"] in binding for cd in calls)
            if self.inquiring and binds and self.present_intended(urge=True):
                continue
            if calls:
                idle_turns = 0
                self.journal.append({"kind": "model", "content": text_part[:2000], "calls": [cd["function"] for cd in calls]})
                self.converse.append({"role": "assistant", "content": text_part or None, "tool_calls": calls})
                tick_before = self.protocol.pulse
                self.later_responses(self.route_alt(calls))
                self.stuck = 0 if self.protocol.pulse != tick_before else self.stuck + 1
                continue
            idle_turns += 1
            self.converse.append({"role": "assistant", "content": text_part or "(no tool was called)"})
            self.converse.append({"role": "user", "content": "Each reply has to call at least one of the tools provided. Go on.\n"
                              + self.progress_lineitem()})
            if idle_turns >= INACTIVE_THRESHOLD:
                break
        if not (self.protocol.done or self.protocol.crashed):
            self.conclude_current()
        closing = {"kind": "end", "finished": self.protocol.done, "steps": self.protocol.pulse, "cost": round(self.protocol.spend, 5)}
        closing["seconds"] = round(_TALLY.now() - self.protocol.startmark, 1)
        self.journal.append(closing)
        return self.journal


class Errand(OutingNucleus, NarrativeClues, InquireVocabulary, Throttling, EntryChecks, RacksAndWares, InventoryAndRejections, AbstainRepair, CarriageAndPicklist, EnquiryPlanning, EnginePilot, SeekAndPaging, RackTraverses, IndicatorRequirements, SettingSeek, Route):
    """One shopping errand: the parts above, joined (no method is defined twice, so their order does not matter)."""




def agent_main(pdata: dict[str, Any]) -> list[dict[str, Any]]:
    errand = Errand(pdata)
    try:
        return errand.carryout()
    except Exception as err:
        errand.journal.append({"kind": "crash", "error": f"{type(err).__name__}: {err}"[:500]})
        try:
            if not errand.protocol.done and not errand.protocol.crashed:
                errand.conclude_current()
        except Exception as second_err:
            errand.journal.append({"kind": "crash", "error": f"{type(second_err).__name__}: {second_err}"[:500]})
        return errand.journal


__all__ = ["agent_main"]
