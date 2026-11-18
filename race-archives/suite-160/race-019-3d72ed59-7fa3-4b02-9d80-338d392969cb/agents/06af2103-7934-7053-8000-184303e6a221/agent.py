"""Wren: a buyer that shops on behalf of one shopper inside an ORO store world.

The labour is shared between a model and ordinary code. An inexpensive model walks the store with whichever tools
the world provides; the code wrapped around it keeps the ledger of items and requirements, chooses which questions
reach the shopper, and checks the cart before any order and the evidence before any finish that buys nothing.
What sits on top of that loop:
- if Wren is ready to buy while the request still saves a message for the end, that last message also carries one
  more planned question, so a tight message allowance cannot leave a hidden preference unasked;
- a clock keeps every episode inside the validator's 300-second limit: the interview, a single question, a loop
  turn or a model request begins only while its answer can still be put to use in time, and the model timeout
  shrinks as the time that is left runs out;
- while the shopper keeps answering lists, one message names up to TRAITS_PER_QUESTION planned properties of one
  item, and the properties take turns, so a property the shopper does not care about can never hide the ones
  waiting behind it; once a reply asks for questions one at a time, each message names just one property.
"""
from __future__ import annotations

import json
import random
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

# ----------------------------------------------------------------------------------------- model seats
# The low-cost seat drives the store loop, the strong seat does reviews, and the third seat stands in for the
# store loop and for note reading when the first one fails.
QUICK_SEAT = "anthropic/claude-haiku-5.5"
DEEP_SEAT = "anthropic/claude-sonnet-5.5"
FALLBACK_SEAT = "openai/gpt-6-luna"

# ------------------------------------------------------------------- what one episode may spend on models
TURN_CEILING = 42                 # store-loop turns
MODEL_CALL_CEILING = 60           # model requests of any kind
# money is counted in credit units
SPEND_COMFORT = 0.055             # a review may still open on the strong seat while spending is below this
SPEND_CEILING = 0.078             # hard ceiling for the store loop's spending
DEEP_SEAT_HEADROOM = 0.02         # room a single strong-seat request needs under the ceiling

# ---------------------------------------------------------------- retry pacing for failed model requests
BACKOFF_SECONDS: tuple = (4, 10, 20, 35)   # successive waits, in seconds
PAUSE_BUDGET = 45                          # the most those waits may add up to

# ------------------------------------------------------- character caps on text the store loop is handed
CELL_TEXT_LIMIT = 450             # a single string field of a tool result
REPLY_TEXT_LIMIT = 15000          # a whole rendered tool result
TRANSCRIPT_TEXT_LIMIT = 150000    # the full chat; past this, older results are shortened

# ------------------------------------------------------------ store work the code may take on by itself
PROCEDURE_GATE_ROUNDS = 3         # model rounds for the request's own steps that fall due at an order or an ending
AUTO_PAGE_LIMIT = 6               # result pages fetched without being asked
LOWER_PRICE_OPENS_PER_LINE = 3    # unread cheaper listings opened per cart line when the pick must be the cheapest

# ------------------------------------------------- the wall clock (seconds counted from the episode start)
RUN_TIME_BUDGET = 300.0           # the validator's limit per episode
CLOCK_CUSHION = 20.0              # left unused at the very end
SHORTEST_CALL_SECONDS = 6.0       # a model request needs at least this much time left to be started
ASKING_CUTOFF = 165.0             # the shopper interview may begin only before this
QUESTION_CUTOFF = 210.0           # no new interview question after this
LAST_QUESTION_CUTOFF = 190.0      # the question kept back for checkout goes out only before this
RUSH_SECONDS = 200.0              # from here on the status line presses for a checkout
LAST_TURN_SECONDS = 255.0         # no fresh store-loop turn after this, except ...
BARE_BASKET_SECONDS = 12.0        # ... while the cart is still empty and at least this much model time is left

# ------------------------------------------------------------------------- messages sent to the shopper
LAST_QUESTION_STEP_ROOM = 8       # steps that have to remain for the question kept back for checkout
SCANT_MESSAGE_ALLOWANCE = 2       # an allowance this low is treated as scarce while lists go unanswered
TRAITS_PER_QUESTION = 8           # upper bound on the planned properties named in one message
FIRST_LIST_WIDTH = 2              # properties the first message names on a one-product task (lists then grow)
TURN_TOKEN_ROOM = 8000            # reply room of a store-loop turn: hidden reasoning counts against it
BRISK_AFTER_SECONDS = 90.0        # from here on store-loop turns ask for low reasoning effort
JSON_TOKEN_FLOOR = 3200           # least reply room of a JSON side call: hidden reasoning comes out of it first

# ------------------------------------------------ regexes over tool schemas: parameter names, browse modes
CAP_ARG_RX = r"(?:^|_)(?:cap|max|maximum|ceiling|under|up_?to)(?:_|$)"
NON_CAP_ARG_RX = r"result|row|page|count|size|num|item|step"
CASH_ARG_RX = r"price|cost|budget|spend|money|dollar|cents?\b"
FLOOR_ARG_RX = r"(?:^|_|\s)(?:min|minimum|floor|from|lower|least|starting)(?:_|$|\s)"
SHELF_ARG_RX = r"section|department|categor|label|shelf|aisle|(?<!age)(?:^|_)group(?:_name|_label)?$"
WORDS_ARG_RX = r"(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword"
WORDS_MODE_RX = r"keyword|phrase|free.?text|full.?text|query|words?\b"
LISTING_MODE_RX = r"field|filter|facet|brows|categor|list|structured"
MODE_EXPLAIN_RX = r"(?:^|[\s:;,(])({choice})\s*(?:=|:|-|means|is)\s*([^;.,|)]+)"

# --------------------------------- regexes over requirement values: an amount with a unit, a lower bound before it
QUANTITY_RX = r'''\d[\d.,]*\s*[a-z"']{1,12}'''
MINIMUM_PHRASE_RX = r"\b(?:at least|no (?:less|fewer) than|min(?:imum)?)\s+(" + QUANTITY_RX + r")\s*$"

# ---------------------------------------------------------------------------------------- proxy links
# Built in this order: the main model link, the cache of short-timeout model links (one per timeout), the store link.
_llm_main_client = ProxyClient(max_retries=1, retry_delay=1, timeout=50)
_llm_quick_clients: dict = {}
_shop_client = ProxyClient(max_retries=3, retry_delay=1, timeout=95)

# The store loop's system message (p09 starts the chat with it). Sections are separated by one blank line; inside a
# section the heading comes first and every rule is its own "- " line.
DRIVER_GUIDE = "\n\n".join((
    "You are Wren, a careful buyer working on behalf of one shopper in a web shop, and nothing gets done except "
    "through the shop's tools. Grading is binary: an episode earns 1 when it ends exactly right and 0 in every "
    "other case. Ending right means one of two things. Either an order that contains each requested item at its "
    "requested quantity, where every item meets all the requirements that still apply at checkout and the total "
    "stays within the budget; or, when the shop truly offers nothing that meets everything, a finish through the "
    "tool that ends without buying, carried out in the manner the request asks for.",

    "USING THE SHOP'S TOOLS\n"
    "- Thinking costs nothing, but every tool call spends one step from a fixed allowance. Tool names, argument "
    "names and the shape of results are particular to this shop: read the description and the schema of each "
    "tool closely and send exactly the argument names and allowed values given there.\n"
    "- Search can come with limits: filtering by category, a keyword mode that may read just the first few words "
    "of the query, small result pages, a token or an offset for getting the next page. When the right answer "
    "needs every row, compare the total count the shop gives with how many rows you have seen and fetch more "
    "pages, sending the token or offset exactly as the shop handed it over. Short keyword queries that lead with "
    "the value work best (the wanted value first, the product noun after it). If a filtered search comes back "
    "empty, the filter value or the filter parameter is usually wrong: retry with the label spelled exactly as "
    "result rows show it, or use another filter parameter.\n"
    "- Search rows only summarise. Shops split a listing's data in different ways: one view may show its "
    "attributes (properties), another just the offer and the description, and a comparison view may show "
    "either. Once Wren has learned which tool displays attributes, the ledger says so. Property requirements "
    "are decided from those attributes, so read them for each listing you may buy; when one call covers several "
    "listings and shows what you need, it saves steps. Leave a listing you have already read alone unless "
    "something changed after you read it.\n"
    "- All questions to the shopper are handled by Wren's interviewer, so you never write to the shopper "
    "yourself. The answers it collects appear in the conversation and go into the ledger.\n"
    "- Apart from buying, the request may want things done on the way: getting an approval, sending a message, "
    "checking or reading something first. The ledger lists them. Whatever falls due just before the order or "
    "just before the ending is done by Wren itself when you call the checkout tool or the ending tool, so do not "
    "wait for those and never end without buying because of them; do the remaining ones with the shop's tools "
    "(product questions go through Wren's interviewer).",

    "DECIDING WHETHER A LISTING FITS\n"
    "- A listing's category label decides what product it is: only a label that matches the requested item as a "
    "whole counts, never one that is merely close to it.\n"
    "- A requirement on a property is met when the listing's attribute for that property shows the value. How "
    "the key is spelled does not matter (spaces, underscores, camelCase, prefixes), and matching any single "
    "entry of a list split by commas or slashes is enough. Values are compared word by word, disregarding case, "
    'singular or plural and simple synonyms, and a value that contains the words of the required value counts '
    '("dusty rose" meets rose, "slim tapered cut" meets tapered). A value in other words does not count (a code '
    'or an abbreviation is not the word behind it, and "navy blue" is not "royal blue"). Bring numbers to one '
    "unit before comparing them; a limit includes its boundary, and a range must lie wholly inside the limit.\n"
    "- If an attribute key names the requirement's property (with the same word or a simple synonym), that "
    "attribute settles it: when it shows something else, the listing fails whatever the title or the other keys "
    "say. A key for another property, however close (a lid's colour is not the body's colour), decides nothing "
    "about the requirement. Under opened listings Wren adds [evidence] lines listing the keys whose names match "
    "a requirement.\n"
    "- When the listing has no attribute for that property, the wanted value still counts if it appears under "
    "another attribute (for example a features or styles list that includes it) or in the title or the "
    "description. A number measuring a different quantity does not count (a length is not a volume), nor does a "
    "bare number whose text never says what it measures.\n"
    "- A category label that itself states a requirement's value (an audience or a season named in the label, "
    "say) meets that requirement.\n"
    "- A requirement may point at another listing instead of giving a value (\"the same X as the listing sold by "
    "<shop> with <property>\", \"no more than N away from the X shown on the page of ...\"). Resolve such a "
    "requirement first: find the listing it refers to (search by its category, brand, shop or wording), open its "
    "full page with the single-listing or the comparison tool (a search row is not enough), and if that page "
    "sends you to yet another listing for the property, find and open that one the same way; the last page "
    "gives the value. The order only counts if every page in that chain was opened. When the requirement asks "
    "for anything other than that listing's value, it excludes only the listings showing exactly that value, "
    "and on its own it never justifies ending without a purchase.\n"
    "- When a listing you plan to buy sends you to another page for further details about it, look at that page "
    "before buying.\n"
    "- When one requested item has to fit another (an accessory for a device), choose the two together from what "
    "the listings say (a list of what it fits, a model or series code shown on both pages). If no accessory "
    "meeting its own requirements fits the device you chose, switch to a different device that meets its "
    "requirements and that such an accessory fits; until that switch has been tried, do not end without buying.\n"
    '- Respect exclusions and closed lists. Wishes ("preferably", "if possible") only decide ties.\n'
    "- Never buy what the shop shows as sold out; the total (price times quantity, summed over every line) may "
    "equal the budget but must not exceed it.",

    "CHOOSING AMONG FITTING LISTINGS\n"
    "- For a cheapest request (lowest total, smallest price), the order must hold the lowest-priced listing among "
    "ALL fitting listings. Collect the whole fitting set (page through it: rows are often not sorted by price, "
    "and Wren pulls the remaining pages of a short result on its own), check candidates from the cheapest up, "
    "and take the first one that fits fully. If a search takes a maximum price, use it: list the category below "
    "the price of your current best pick, so only cheaper candidates are left to check.\n"
    "- For a request for the most or the least of some measure, compare every fitting listing on that measure "
    "and take the extreme; a listing that does not state the measure cannot win.\n"
    "- In every other case any listing that meets everything is right: do not spend steps hunting for a better "
    "one.\n"
    "- Add an item to the cart the moment you have opened a listing meeting all of its requirements (for a "
    "cheapest, most or least pick: once the fitting set has been compared), then go on. Lines frequently sell "
    "out soon after they are added, and adding early leaves steps to replace one.\n"
    "- If the request names several different items, give each its own cart line at its requested quantity.",

    "CHANGES ALONG THE WAY\n"
    "- The shopper or the shop can change things: a different budget, a requirement added or dropped, a price "
    "change, a listing selling out. These changes arrive inside tool results, often as a refused checkout; a "
    "note holds no matter which tool's result it comes in, and the ledger shows the requirements that apply. A "
    "refusal is not the end of the episode: take in every change, fix the cart and check out again.\n"
    "- Just before checking out, read the record of every chosen listing again, because price and stock may "
    "change without warning.",

    "FINISHING\n"
    "- Leave exactly the chosen lines in the cart (remove everything else), look at the cart once, and then check "
    "out in a call of its own.\n"
    "- End without buying only after a real search found no listing that meets everything. Running low on steps "
    "proves nothing of the sort: in that case check out the best cart you have. Phrase that ending the way the "
    "request says, and if a remark is needed, make it short and name no listing.\n"
    "- Each reply must include at least one tool call. Keep the text short: what you have found out, then what "
    "you do next.",
))

# The request reader's prompt: one JSON brief per episode. p03 appends "\nTOOLS:\n..." and then the REQUEST block
# straight after the final newline, so each key keeps a line of its own, in this order.
REQUEST_DIGEST_PROMPT = "\n".join((
    "You convert a shopper's request into instructions for an agent who will do the buying. Read the request "
    "together with the store's tools and reply with ONE JSON object (compact, not indented, nothing else around "
    "it) containing the following keys:",
    "",
    '"items": a list with an entry for each distinct product the shopper wants bought, each {"name": what kind '
    "of product it is, in the request's own wording, \"shelf\": the store's category label for it when some tool "
    "lists labels (copied exactly), else the category the request itself puts it in (\"a kettle from the kitchen "
    'section" gives "kitchen"), else "", "quantity": how many units of that one listing (integer; 1 when not '
    'stated; a pair or a set sold as a single listing counts as 1), "musts": a list of the product\'s firm '
    'requirements, each written "property: constraint" with the exact value, limit, exclusion or closed list. '
    'Name the property with the word the request uses for it ("meant for an outdoor setting" gives "setting: '
    'outdoor", "a 12-litre tank" gives "tank capacity: 12 litre"), and never swap it for a neighbouring property. '
    'Write the value in the words the request gives that value, adding none of your own and dropping none, but '
    'without the name of the property itself ("bags with a striped pattern" gives "pattern: striped"). A property '
    'stated together with the product ("a chair in oak", "a red kettle") is still a must of that product, never part '
    'of its name. '
    "A must whose value comes from another listing is ONE must that keeps that listing's full description (\"lid "
    'colour: unlike the lid colour of the kettle Tollmark sells with a glass body"): the words that describe the '
    'other listing (its shop, its kind, a "with <property> <value> and <property> <value>" after it) stay inside '
    "that one must and never turn into musts of this product}. Such a must also keeps, in the request's own words, "
    'how its value relates to the other listing\'s value (equal to it, different from it, its partner, within some '
    "distance of it): never trade one relation for another. Price, budget, stock and quantity are never musts. "
    "The story around the request (who will use the product, where it will be placed, what went wrong before, "
    "the occasion for buying) is background and not a must, unless it states a property's value or limit "
    "outright; and words that are part of the product's own name (an audience or a season given together with "
    'the product) never become musts: they belong in "name". When the request makes a use or an occasion into a '
    'property of the product itself ("sandals for beach wear", "a blazer designed for interviews", "a tie, and '
    'it is for a funeral", "the occasion must be formal"), that IS a must on that property ("occasion: beach"), '
    "unless the store's category label already names it; one particular person's event, a person or a place the "
    "purchase is for (\"to wear at his cousin's graduation\", \"for a colleague\"), and how the product should sit, "
    "feel or look in use, stay background. Products the request names in one phrase joined by \"and\" (a single "
    '"I want ..." with a single quantity, often the name of one store department) make ONE item; make separate '
    "items only when the request gives each product its own quantity or its own requirements.",
    '"budget": the most the shopper is willing to pay in total, as a number in the request\'s own unit, or null.',
    '"budget_unit": what that unit is called (for example "cents"), or null.',
    '"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when no limit '
    "is given.",
    '"in_stock_only": true unless the request clearly allows items that are not available.',
    '"pick": "cheapest" when the request asks for the lowest price or the smallest total, "most" or "least" when '
    'it asks for the highest or lowest value of some measure, else "any". Remarks about money being tight are not '
    "a pick rule.",
    '"pick_measure": with "most"/"least", the measure to maximise or minimise; else "".',
    '"messages": how many messages the shopper allows the agent to send (integer), or null when the request is '
    "silent about it.",
    '"steps": how many steps, moves or tool calls the request allows (integer), or null.',
    '"lookups": the musts (copied exactly) whose value must first be read off another listing the request points '
    "to, else [].",
    '"rules": every instruction the request gives on how to proceed or how to finish, one entry each, 12 words at '
    "most.",
    '"wishes": the things the shopper would only like to have.',
    '"ask_first": the points the request explicitly says to check with the shopper before choosing (empty when '
    "there are none).",
    '"process": the actions beyond finding and buying the right items that the request demands of the agent '
    "(getting the shopper's approval, telling or showing the shopper something, confirming a point with them, "
    'reading something first, using a particular tool), each {"step": what to do, 15 words at most, "when": '
    '"choose" (before the products are chosen), "order" (right before checkout) or "end" (right before ending '
    'without a purchase), "messages": how many messages to the shopper it takes (0 if none)}; [] when the request '
    "asks for no such action. Requirements on the items, the budget, stock, the limits on messages and steps, and "
    "how to end when nothing fits are not steps.",
    '"quiet_decline": true when the request says that, if nothing fits, alternatives may not be named or '
    "suggested; false when it asks for alternatives; null when it says nothing about this.",
    '"fits": a list of {"item": item number, "with": item number}, one entry per requested item that, by the '
    "request, must work with, fit or suit ANOTHER requested item (whichever one the agent chooses); [] when no "
    "item does.",
    '"roles": an object naming the exact tool for each job, or null for a job that no tool does: "search", "open" '
    "(shows one listing's attributes or properties; when two tools show a single listing, the one whose "
    'description mentions attributes, properties or specifications), "offer" (shows one listing\'s current offer '
    'such as its price and stock, when that is a different tool from "open"), "compare" (shows several listings '
    'in full), "ask" (sends the shopper a message), "cart" (changes the cart), "cart_view" (shows the cart), '
    '"buy" (checks out), "refuse" (ends without buying).',
    '"buy_args": the fixed arguments that make the "buy" tool check out (the mode, for a tool with a mode '
    "argument), else {}.",
    '"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.',
    "",
))

# The question planner's prompt. p06 swaps every COUNT for the per-item allowance and then adds the REQUEST, ITEMS,
# ASK FIRST and ATTRIBUTE KEYS ON OPENED LISTINGS blocks, so COUNT stays one upper-case word used once only.
ASKING_PLAN_PROMPT = "\n".join((
    "You choose which questions a shopping agent asks a shopper. The shopper has a few firm preferences that the "
    "written request leaves out, each about one property of one requested product. A preference comes out only "
    'when a question names that exact property; a property the shopper has no view on gets "don\'t mind", and one '
    "the shopper never considers gets a puzzled reply, so every property you name costs a share of a message.",
    "",
    "You are given the request, the items with what the request already fixes for each, and a numbered list of "
    "the attribute keys seen on opened listings in each item's category, with how many of those listings carry "
    "each key and a few sample values. An unwritten preference may be about a property that no listed key names.",
    "",
    "Put the properties in this order:",
    "0. Anything the request itself says to confirm with the shopper (listed under ASK FIRST when there is any), "
    "each named by the property it is about.",
    "1. Hinted properties. The story of the request often points at an unwritten preference without giving its "
    "value: a sentence about something the shopper cannot judge or leaves to you, about someone whose taste "
    "decides, about the product's look or the place it belongs in, about who will use or wear it, or about what "
    "is inside it. For each such sentence, name the one property it points at and the item it concerns, in the "
    "order the story brings them up.",
    "If one of that item's listed keys fits the hint, name the hinted property by that key (for a hint about the "
    "wearer, the gender or audience key; for a hint about looks, whichever listed colour, pattern or finish key "
    "most of its listings carry); a property that none of the item's listings carries goes after the item's three "
    "most widely carried keys.",
    "2. Next, the listed keys by rank: with no hint, put first the properties most particular to this kind of "
    "product (how it is built or fastens, its type or style, the user it is made for, its main technical figure, "
    "what sets one candidate apart from another), then the generic ones nearly every product has (colour, size, "
    "pattern, material), which are more often ones the shopper has no view on; among equals favour keys that "
    "most of the item's listings carry with values that differ between listings; once five or more listings have "
    "been read, avoid keys found on only one or two of them.",
    "Skip a property the request already fixes for that item (a requirement names that property; a key whose "
    "values only repeat a required value, such as a pattern key showing the required motif, is NOT fixed), "
    "another name for a property already chosen, a key whose every value bundles several purchase options, "
    "identifiers, model names, warranty and other logistics. Pack size, count and quantity per pack are normal "
    "properties, and a key marked [not the stated ...] is a different property from that requirement, whatever "
    "words its name shares with it. Rank every item on its own: its hinted properties first, then its listed keys.",
    '"label" is the property\'s plain name: for a listed key, all of the key\'s words written out plainly '
    '("handle material" for handleMaterial, "sole width" for sole_width; never shortened to "material" or '
    "\"width\"); only the property's name, with no sample values, brackets or second property.",
    "Choose no more than COUNT per item, strongest first. Answer with ONE JSON object only: "
    '{"ask": [{"item": item number, "label": "...", "hint": "the story words it comes from, 8 at most, or an '
    'empty string"}]}, listing each item\'s picks in rank order',
))
# ==== tuning numbers and code-shape patterns ===========================================================================
# Question lanes (read by the question planner): at most this many of the most frequent attribute keys may join an
# item's lane; the floor is the smallest share of the item's listings that has to carry such a key; the margin is the
# share by which a common key must lead a weaker, unhinted pick before it pushes that pick out.
SHARED_KEY_COUNT: int = 3
SHARED_KEY_FLOOR: float = 0.0
SHARED_KEY_MARGIN: float = 0.34
# Search groups: how many listing pages get opened per group while the request names no more than two items.
READS_PER_SHELF: int = 6

# Attribute keys that describe no property of the product (ids and codes, pictures and links, warranty, condition, ...).
NOT_A_TRAIT_KEY_RX = r'\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\w*|video|url|links?|images?|photos?|condition)\b'
# Grab-bag keys that lump several features together; keys about type, style, kind or shape never count as such.
BUNDLE_KEY_RX = r'^(?!.*\b(?:types?|styles?|kinds?|shapes?)\b).*\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\w*|accessor\w*|extras?|misc\w*)\b'
# Device model codes and part numbers as titles and descriptions print them.
DEVICE_CODE_RX = r'[A-Z][A-Za-z]{0,5}[ -]?\d{3,6}'
# A whole family of model codes whose trailing digits are written as x letters (e.g. "KQ7xx" stands for every KQ7
# code of that length).
WILDCARD_CODE_RX = r'[A-Z][A-Za-z]{0,5}[ -]?\d{1,5}[xX]{1,4}'
PART_CODE_RX = r'[A-Z]{1,4}\d{3,6}'
# A part numbered within a series (e.g. "TR2-4471" is part 4471 of series TR2), then a bare series code.
RANGE_PART_RX = r'[A-Z]{1,4}\d{1,4}-\d{2,6}'
RANGE_CODE_RX = r'[A-Z]{1,4}\d{1,4}'


# ==== prompts for the single-shot JSON side calls =====================================================================
# Every text below opens a side call whose answer has to be one bare JSON object. Callers tack their own headed blocks
# (REQUEST, REQUIREMENTS IN FORCE, RECORDS, TOOLS and so on) onto the end, which is why each text closes with its JSON
# contract or with a last rule. In the step gate the word STAGE is a slot that the caller fills with the stage's verb
# phrase.

# How to judge a listing record: a lower-case clause list that the two fit-finding prompts splice in after a lead-in.
_READING_CLAUSES = ('a property is decided by an attribute only when the key of that attribute is the property '
                    'itself (the key may be spelled any way, one matching entry of a list is enough, synonyms and '
                    'unit conversions are fine, and so is a value containing the words of the wanted value, e.g. '
                    '"dark navy" where navy is wanted; a bound includes its boundary value); a key for a different '
                    'property decides nothing, no matter how near it is (the colour of a strap is not the colour of '
                    'the main body); when no attribute is keyed by the property, take the wanted value if it is '
                    'written under another attribute, in the description or in the title (a lone number that does '
                    'not say what it measures counts for nothing); a measured limit (a figure with a unit) counts only '
                    'under an attribute whose key shares a word with the property, never from a key for another '
                    'measure or from the title alone; a list under a key named in the plural of the property holds the '
                    "variants on offer, not this listing's own value when its title names another of them; when the "
                    'category label itself spells out a '
                    'requirement, membership of that category meets it; the category label must match the item; a '
                    'rule requiring the value to differ from the value of some other listing is met unless the records '
                    'show that very value; an out-of-stock listing, or one over the budget, never fits.')

# The gate on the cart right before the agent pays.
BASKET_AUDIT_PROMPT = ('You are the final checkpoint before a shopping agent pays. Using the records supplied, decide '
                       'if the cart is correct in every detail.\n'
                       'Reading a listing: a requirement is met when the attribute the listing has for that property '
                       'shows the value (the key may be spelled any way, one matching entry of a list is enough, upper '
                       'or lower case, singular or plural and everyday synonyms are all fine, and so is a value '
                       'containing the words of the wanted value, e.g. "dark navy" where navy is wanted; units get '
                       'converted, a bound includes its boundary value, and a range passes only if it lies wholly '
                       "within); an attribute whose key is the requirement's own property decides it, and the listing "
                       'fails when that attribute shows anything different; a key for a different property decides '
                       'nothing, no matter how near it is (the colour of a strap is not the colour of the main body); '
                       "when none of the listing's attributes covers the property, take the wanted value if it is "
                       'written under another attribute, in the description or in the title (but not a number '
                       'measuring something else: a weight is no capacity; and not a lone number whose text does not '
                       'say what it measures); for a measured limit (a figure with a unit) only an attribute whose key '
                       'shares a word with the property counts: the same figure under a key for another measure, or '
                       'only in the title, leaves the limit unknown, and the listing fails it; a list under a key named '
                       'in the plural of the property holds the variants on offer and does not give this listing\'s '
                       'own value when its title names another of them; when the category label itself spells out a requirement, membership of '
                       'that category meets it; the category label must match the requested item as a whole; a rule '
                       'requiring the value to differ from the value of some other listing is met unless the records '
                       'show that very value. '
                       'Each requested item must have its own cart line at the quantity asked for, nothing else may '
                       'sit in the cart, no line may be marked out of stock, and the sum of price times quantity '
                       'across all lines has to fit within the budget. When one listing has several records, the '
                       'newest one counts.\n'
                       'If the pick rule asks for the cheapest, the most or the least, look through the other listings '
                       'shown as well: if any of them plainly meets every requirement and beats the cart\'s pick under '
                       'that rule, the cart is wrong.\n'
                       'Reply with ONE JSON object, no other text: {"ok": true or false, "problems": ["short, concrete '
                       'problem", ...], "better": [ids of the listings that ought to replace a cart line, best '
                       'first]}. Give ok=false only if the records clearly reveal a problem, or if some requirement is '
                       'addressed by no record at all.')

# Before a purchase-free ending: a second look for listings that might fit after all.
REFUSAL_AUDIT_PROMPT = ('A shopping agent intends to finish with no purchase because it thinks nothing on offer '
                        'fits. Check the listing records supplied against the requirements, applying these rules to '
                        'every record: '
                        + _READING_CLAUSES
                        + '\nA ROW record comes only from one line of search results: its page, which carries the '
                        'attributes, has not been read, so any property absent from the row can neither pass it nor '
                        'fail it.\n'
                        'Reply with ONE JSON object, no other text: {"fits": {"item number": [ids of the listings '
                        'that appear to meet every requirement, cheapest first, no more than 4]}, "open": [ids of the '
                        'ROW listings of a requested kind and inside the budget whose page could reveal a fit, most '
                        'promising first, no more than 6], "reason": "one sentence"}')

# Replacements for a cart line whose listing has sold out.
SUBSTITUTE_PROMPT = ("A shopping agent has a listing in its cart that is now sold out. Look among the other listing "
                     'records supplied for replacements of that cart line, applying these rules to each: '
                     + _READING_CLAUSES
                     + "\nReply with ONE JSON object, no other text: {\"fits\": [{\"id\": the listing's id, "
                     '"price": the price shown in its record, as a number} for every listing that appears to meet '
                     "each requirement of the sold-out line's item, no more than 6], \"reason\": \"one sentence\"}")

# Following requirements whose value sits on other listings, one link after another.
LINK_TRACE_PROMPT = ('A shopping agent is on the point of filling its cart or checking out. Some requirements give '
                     'no value of their own but take it from a different listing, and that listing\'s page may, in its '
                     'turn, send the reader on to one more listing for that same property. The order is valid only once '
                     'the agent has opened in full every listing along each such chain. Begin at the requirement\'s '
                     'own description and follow each chain link by link, relying on what the pages themselves say, '
                     'using the records below (OPENED: the agent opened this page; ATTRIBUTES: only its attributes '
                     'were seen, through a different view, so its page, where a pointer to a further listing would be '
                     'printed, has not been read; ROW: seen only as a line of search results, page not read). An '
                     'ATTRIBUTES or ROW record may also be a link: name its id, and the agent will open that page.\n'
                     'Reply with ONE JSON object, no other text: {"chains": [{"need": the requirement, copied '
                     'verbatim, "links": [{"id": the listing\'s id if this link is one of the records below (the '
                     'record\'s own id, never a code printed in its title), otherwise "", "about": how the '
                     'requirement or the preceding page characterises this listing, "values": the values that '
                     'description gives for this listing, as a few words without the names of the properties or of '
                     'its shop, "find": "for a link that was not found: words to search with (the shop\'s name plus '
                     'the values in its description, leaving out the names of the properties and the kind of '
                     'product, since the store\'s search wants every word to appear in the listing)"}], "done": true '
                     "if the last link is an OPENED page that states the property's value itself and whose own words "
                     'do not send the reader on to yet another listing for that property (a page that leaves the '
                     'value to a different listing is not the last link, even when its attributes list a value; '
                     "done also applies when the requirement asks for a value that differs from or is unlike the "
                     "page's: the page's value is the reference), otherwise false, \"value\": the property's value "
                     'as given on the last page; when the requirement, or the request\'s own words for it, asks for '
                     'what goes with or pairs with that value rather than the value itself, and the page names what '
                     'its value pairs with, that other value; or ""}]}\n'
                     'A record is a link only when it matches the whole description of the link: shop or brand, kind '
                     'of product AND every property value named in the description (so "a 6 bottles gift box" rules '
                     'out a listing from that shop with 6 bottles that is not a gift box); when no record matches all '
                     'of it, the link stays not found, even if that shop has other products of this kind on offer. '
                     'Put a few search words into "find" for such a link, namely the shop\'s name together with the '
                     'values in the description.\n'
                     "Give each chain's links in order and halt at the earliest link that has not been found or has "
                     'no OPENED page, since its contents are not known yet; as long as that link is still an '
                     'ATTRIBUTES or ROW record, "done" remains false and "value" remains "", even if its attributes '
                     'show the property.')

# The check on steps the request wants done before ordering or ending; the caller swaps STAGE for a verb phrase.
PROCEDURE_GATE_PROMPT = ("The shopping agent is ready to STAGE. On the way there, the shopper's request asks for the "
                         'steps listed below. Going by the numbered RECORD of what the agent has done, decide, for '
                         'every step falling due before the agent can STAGE, if it has been completed for the cart '
                         'as it now stands. A step counts as done only when a RECORD line shows it taking place; for '
                         'a step that concerns the cart or the order, that line also has to come after the latest '
                         "cart change, and the reply it got must approve it. Asking about a product's properties "
                         'carries out no step.\n'
                         'Should any step remain undone, name the one store call that would carry it out next: a '
                         'tool from TOOLS with its arguments. A message to the shopper is sent with the messaging '
                         'tool as one brief sentence or question, plainly worded, concerning that step only and '
                         'leaving out any product property or value that the step does not call for. Name no call '
                         'if every due step is done, or if the messages or steps left cannot achieve the missing one.\n'
                         'Reply with ONE JSON object, no other text: {"steps": [{"step": the step as listed, "done": '
                         'true or false, "lines": [the RECORD line numbers that show its completion]}], "call": '
                         '{"tool": "...", "args": {...}} or null, "hold": the change any reply asks for before going '
                         'on, or "", "needs": [{"item": item number, "need": "property: constraint"} per firm '
                         'requirement that such a reply states]}')

# Turning the shopper's replies to the agent's questions into requirements.
REPLY_PARSE_PROMPT = ('A shopping agent put questions to the shopper, and the replies are now in. Read each reply, '
                      'then answer with ONE JSON object and no other text:\n'
                      '{"needs": [{"item": item number, "need": "property: constraint", "points": true when the '
                      'value must be fetched from another listing, else false}], "drop": [{"item": item number, '
                      '"property": "a requirement the shopper now gives up"}], "free": [{"item": item number, '
                      '"property": "..."}], "crowded": [numbers of the questions whose reply complains that the '
                      'message bundled too many things together or asks for them one by one; skip replies that only '
                      'say they did not understand it or want it phrased more simply], "budget": the new total limit '
                      '(a number), or null}\n'
                      '- "needs": every firm wish voiced in a reply, with its precise value or limit and the unit '
                      '("at most 30 cm", "round", "not plastic"). Name the need after the property the reply itself '
                      'talks about, even when the question asked about a different property; if several replies '
                      'touch the same property, they make up one need between them. Copy the value word for word as '
                      'the reply gives it, keeping nouns and units ("at least 3 drawers", not "at least 3"); but keep '
                      'only the portion a listing could state: start at the wanted value or its limit word, go on to '
                      'its end, and drop whatever framing the reply puts around it ("round" from "matching round", '
                      'from "settled, with round", or from "round and nothing else"; "at most 30 cm" from "equal to '
                      'at most 30 cm").\n'
                      '- At times a reply pins the value to another listing (a value similar to, close to or '
                      "different from the one on some shop's product, that product being picked out by a few of its "
                      'properties). Record it as ONE need for the property the reply names, and set "points": true. '
                      "As its value, copy the reply's own words unchanged, from the comparing words up to where that "
                      'description ends, together with whatever it says about any listing named on that page. Never '
                      'shorten or reword it. Properties describing the other product are not needs of this item.\n'
                      '- "free": for each question to which the shopper replied that the property asked about makes '
                      'no difference to them, that property.\n'
                      '- Talkative or roundabout phrasing still gives a value when it names one. Phrasing that '
                      'refuses to bargain over a value (it must not be exchanged, given up or changed) means that '
                      'value is wanted; write "not X" only when the reply rejects X itself.')

# Reading a note the store put up while the agent was working.
SHOP_NOTICE_PROMPT = ("The store showed the note below partway through a shopping agent's work; its source may be the "
                      'shopper or the store. Reply with ONE JSON object, no other text:\n'
                      '{"kind": "change" if the note changes the task (the budget, requirements, a price, what can be '
                      'bought, or whether the order is refused), "error" if it only reports that a call was malformed '
                      'or turned down, "other" for anything else, "budget": the new total limit as a number in the '
                      'request\'s unit, or null, "add": [{"item": item number, "need": "property: constraint"}], '
                      '"drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [ids of '
                      'the listings that the note reports as unavailable], "sold_out": true when it reports that a '
                      'chosen listing is gone, sold out or no longer obtainable (even without naming it), '
                      '"price_moved": true when it reports a price change, "summary": "one short sentence"}\n'
                      'If one requirement takes the place of another, list the old one under drop and the new one '
                      'under add. Cover every change in the note.')

# Telling the words that merely describe a pointed-at listing apart from real requirements.
BORROWED_SPLIT_PROMPT = ("A shopper's request was read for the firm requirements on one product. Some of these take "
                         'their value from another listing the request mentions, and the words picking out that '
                         'listing (its shop, its kind, the properties that follow it in a "with ..." phrase) are '
                         'sometimes wrongly taken as requirements on the product being bought. Judging by the '
                         "request's wording, say which of the entries under OTHER REQUIREMENTS below only describe "
                         'that other listing. Reply with ONE JSON object, no other text: {"describe_other": [every '
                         'such requirement, copied word for word; an empty list if they all really apply to the '
                         'product]}')

# Matching the pieces of a store tool's result to its keys.
REPLY_SHAPE_PROMPT = ("A store tool returned the result below. Name its parts with exactly the keys that occur in the "
                      "result and among the tool's parameters. Reply with ONE JSON object, no other text: {\"rows\": "
                      'the key whose list contains the listings, "count": the key holding how many listings match in '
                      'total (not just how many appear here), or null, "next": the key holding the value that asks for '
                      'the following page, or null, "next_arg": the tool parameter into which that value goes, or '
                      "null, \"price\": the key inside a row for the listing's price, or null, \"stock\": the key "
                      'inside a row saying whether the listing can be bought now, or null}')
# ================================================================ stateless module tools
# Plain functions with no session state, shared by every run class: which model client suits the time that is left,
# the record of one wanted product, fit hints read out of listing text, listings that arrive as one delimited string,
# decoding model replies, squaring tool arguments with their schema, finding ids in store records, and how keys and
# names are spelled.


# ---------------------------------------------------------------- compiled patterns
# Each pattern below keeps the exact text and flags it always had; only the way the text is put together is new.

def _bounded(core: str) -> re.Pattern:
    """`core` as a pattern that must stand as a whole word on both sides."""
    return re.compile(rf'\b{core}\b')


def _bracketed(core: str) -> re.Pattern:
    """`core`, captured, between round or square brackets with an optional leading "part"; case is ignored."""
    return re.compile(r'[\[(]\s*(?:part\s+)?(' + core + r')\s*[\])]', re.I)


_DEVICE_TOKEN = _bounded(DEVICE_CODE_RX)
_FAMILY_TOKEN = _bounded(WILDCARD_CODE_RX)
_BRACKETED_PART = _bracketed(PART_CODE_RX)
_BRACKETED_SERIES_PART = _bracketed(RANGE_PART_RX)
_EXCLUSION_WORD = re.compile(r'\b(?:except(?: for)?|excluding|but not|other than|apart from)\b', re.I)
_MODEL_LABEL = re.compile(rf'\bmodel\s*(?:no\.?|number|#)?\s*[:.]?\s*({DEVICE_CODE_RX})\b', re.I)
_ALIAS_PHRASE = re.compile(r'\b(?:also sold as|also known as|sold also as)\s+([^.;]+)', re.I)
_FITS_PHRASE = re.compile(r'\b(?<!not )(?<!never )(?:works with|compatible with|fits)\s*:?\s+([^.;]+)', re.I)
_ALSO_LEADIN = re.compile(rf'\bpart\s+{PART_CODE_RX}\s+also\s+$', re.I)
_ALSO_FITS_PHRASE = re.compile(rf'\bpart\s+({PART_CODE_RX})\s+also fits\s*:?\s*([^.;]+)', re.I)
_SERIES_PHRASE = re.compile(rf'\bseries\s*:?\s+({RANGE_CODE_RX})(?![\w-])', re.I)
# a list of series after the word ("series AB4, AB8 and AB9"), and a model given more series on another listing
_SERIES_RUN = rf'(?:{RANGE_CODE_RX})(?:\s*(?:,|and|&)\s*{RANGE_CODE_RX})*'
_SERIES_LIST_PHRASE = re.compile(rf'\bseries\s*:?\s+({_SERIES_RUN})(?![\w-])', re.I)
_MODEL_ALSO_SERIES = re.compile(
    rf'\bmodel\s+({DEVICE_CODE_RX})\s+also\s+(?:\w+\s+){{1,2}}series\s*:?\s+({_SERIES_RUN})(?![\w-])', re.I)
_POINTER_PHRASE = re.compile(
    rf'\b(?:more|details|see|read)\s+(?:on|at|under|about)?\s*(?:the\s+)?part\s+({RANGE_PART_RX}|{PART_CODE_RX})', re.I)
_CODE_FENCE = re.compile(r'^```[A-Za-z]*\s*(.*?)\s*```$', re.S)
_HUMP = re.compile(r'([a-z0-9])([A-Z])')
_JOINERS = re.compile(r'[\s_.\-]+')
_SIGNED_NUM = re.compile(r'\s*-?\d+(\.\d+)?\s*')
_WHOLE_NUM = re.compile(r'\s*\d+(\.0+)?\s*')
_ANY_DIGIT = re.compile(r'\d')

# returned by a repair step that has nothing to offer for a value
_UNTOUCHED = object()


# ---------------------------------------------------------------- picking a model client for the time left

def _llm_client_within(seconds_left: float) -> ProxyClient:
    """With 50 s or more to go, the shared client. Below that, a client kept per timeout, the timeout being the
    seconds left cut down to a multiple of 5 and never below 5; it is built on first use."""
    if seconds_left >= 50:
        return _llm_main_client
    seconds = max(5, int(seconds_left) // 5 * 5)
    pooled = _llm_quick_clients.get(seconds)
    if pooled is None:
        pooled = ProxyClient(timeout=seconds, max_retries=1, retry_delay=1)
        _llm_quick_clients[seconds] = pooled
    return pooled


# ---------------------------------------------------------------- the record of one wanted product

class RequestedGood:
    """One product on the shopper's list: what it is called, the shelf label it goes under, the number of units,
    the needs stated up front, and what the shopper's later answers settled, withdrew or left undecided."""

    def __init__(self, name: str, shelf: str, units: int, stated: list[str]):
        self.good_title = name
        self.shelf_name = shelf
        self.count_wanted = units
        self.given_needs = stated
        # the three lists below grow during the interview
        self.told_needs = []
        self.dropped_needs = []
        self.open_traits = []


# ---------------------------------------------------------------- fit hints inside listing text

def _range_code(series_code: str) -> str:
    """How a part series is written among fit codes, so _code_sets_meet can compare it like any other code."""
    return f'series {series_code.lower()}'


def _code_spelling(code: str) -> str:
    """A single way of writing a code: hyphens become spaces, space runs shrink to one, all in lower case."""
    pieces = code.replace('-', ' ').split()
    return ' '.join(pieces).lower()


def _codes_named(snippet: Any) -> list[str]:
    """The model codes a piece of text mentions. Families such as "ab31xx" come first, then exact codes; whatever
    exact codes follow an exclusion word ("except", "other than", ...) are added at the end with a "!" in front."""
    said = str(snippet or '')
    cut = _EXCLUSION_WORD.search(said)
    allowed, barred = (said[:cut.start()], said[cut.end():]) if cut else (said, '')
    mentioned = _FAMILY_TOKEN.findall(allowed) + _DEVICE_TOKEN.findall(allowed)
    excluded = _DEVICE_TOKEN.findall(barred)
    return [_code_spelling(code) for code in mentioned] + ['!' + _code_spelling(code) for code in excluded]


def _family_covers(offered: str, target: str) -> bool:
    """An offered code covers a target when both are equal, or when the offer is a family (trailing x's) of the
    same length whose fixed head starts the target."""
    if offered == target:
        return True
    head = offered.rstrip('x')
    return head != offered and len(offered) == len(target) and target.startswith(head)


def _code_sets_meet(part_codes: Any, device_codes: Any) -> bool:
    """Does any code a part claims cover one of the device's codes? Spaces are ignored on both sides, and a device
    code the claim excludes ("!" entries) is never covered."""
    excluded = {code[1:].replace(' ', '') for code in part_codes if code.startswith('!')}
    offers = [code.replace(' ', '') for code in part_codes if not code.startswith('!')]
    targets = (code.replace(' ', '') for code in device_codes)
    return any(_family_covers(offer, target) for target in targets if target not in excluded for offer in offers)


def _also_entry(body: str, start: int) -> bool:
    # a "fits" right after "part XY123 also" belongs to that part's own "also fits" list, not to this listing
    window = body[max(0, start - 40):start]
    return _ALSO_LEADIN.search(window) is not None


def _fit_signals(blurb: Any) -> dict[str, Any]:
    """Everything a listing tells about fit: the part number in brackets ('own'), its model codes and other names
    ('codes'), the models it works with ('works'), other parts' "also fits" lists ('also'), a part it sends the
    reader to for details ('more'), the series of its own bracketed series part ('in_series'), and every series it
    names ('series')."""
    body = str(blurb or '')

    def first_upper(pattern: Any) -> str:
        hit = pattern.search(body)
        return hit.group(1).upper() if hit else ''

    def captures(pattern: Any) -> list[str]:
        return [hit.group(1) for hit in pattern.finditer(body)]

    names = [code for chunk in captures(_MODEL_LABEL) + captures(_ALIAS_PHRASE) for code in _codes_named(chunk)]
    partners = [code for hit in _FITS_PHRASE.finditer(body) if not _also_entry(body, hit.start())
                for code in _codes_named(hit.group(1))]
    others = [(hit.group(1).upper(), _codes_named(hit.group(2))) for hit in _ALSO_FITS_PHRASE.finditer(body)]
    numbered = first_upper(_BRACKETED_SERIES_PART)
    listed = {code.upper() for run in captures(_SERIES_LIST_PHRASE) for code in re.findall(RANGE_CODE_RX, run, re.I)}
    extended = [(code, sorted({one.upper() for one in re.findall(RANGE_CODE_RX, hit.group(2), re.I)}))
                for hit in _MODEL_ALSO_SERIES.finditer(body) for code in _codes_named(hit.group(1))]
    return {'own': first_upper(_BRACKETED_PART), 'codes': names, 'works': partners, 'also': others,
            'more': first_upper(_POINTER_PHRASE), 'in_series': numbered.split('-')[0], 'own_series_part': numbered,
            'series': sorted({series.upper() for series in captures(_SERIES_PHRASE)} | listed),
            'model_also': extended}


# ---------------------------------------------------------------- listings delivered as one delimited string
# A store may hand over each listing as a single line like "sku: q-7 | name: ... | cost: 9 | specs: x=1; y=2".
# Such a line becomes a dict, so rows, pages, checks and the model's view all see the same named fields.

_PIPE = ' | '
_KEY_HEAD = re.compile(r'^\s*([A-Za-z][A-Za-z0-9_]*(?: [A-Za-z0-9_]+){0,2}):\s?(.*)$', re.S)
_BARE_NUMBER = re.compile(r'^-?\d+(?:\.\d+)?$')
_SPELLED_CONSTANTS = {'True': True, 'true': True, 'False': False, 'false': False, 'None': None, 'null': None, '': None}


def _fold_pieces(pieces: list[str], opener: Any, glue: str, orphan_fails: bool) -> dict[str, str] | None:
    """Gather split pieces under keys. `opener(piece, keys_so_far)` gives (key, value) for a piece that starts a
    new key and None otherwise; such a piece is joined back with `glue` onto the value before it. A keyless piece
    before any key either sinks the whole reading (None, when orphan_fails) or is left out."""
    grouped: dict[str, str] = {}
    current = None
    for chunk in pieces:
        head = opener(chunk, grouped)
        if head is not None:
            current = head[0]
            grouped[current] = head[1]
        elif current is not None:
            grouped[current] += glue + chunk
        elif orphan_fails:
            return None
    return grouped


def _pipe_head(piece: str, seen: dict[str, str]) -> tuple[str, str] | None:
    # "key: value" with a key of one to three words; a key met before counts as plain text
    found = _KEY_HEAD.match(piece)
    if found is None or found.group(1) in seen:
        return None
    return found.group(1), found.group(2)


def _pair_head(piece: str, seen: dict[str, str]) -> tuple[str, str] | None:
    # "key=value" with a fresh, non-blank key of at most 60 characters (counted before trimming)
    name, sign, rest = piece.partition('=')
    label = name.strip()
    if not sign or not label or label in seen or len(name) > 60:
        return None
    return label, rest.strip()


def _semicolon_pairs(text: str) -> dict[str, str] | None:
    """A "k=v; k=v" value as a dict of texts. None unless it has two parts or more and at least two of them, and no
    less than 70% of them, contain '='."""
    pieces = text.split('; ')
    with_sign = sum(1 for piece in pieces if '=' in piece)
    if len(pieces) < 2 or with_sign < max(2, len(pieces) * 0.7):
        return None
    return _fold_pieces(pieces, _pair_head, '; ', False)


def _typed_value(text: str) -> Any:
    """The value a field text stands for: True/False, None for a blank or a null word, an int or float for a plain
    number, otherwise the trimmed text."""
    bare = text.strip()
    if bare in _SPELLED_CONSTANTS:
        return _SPELLED_CONSTANTS[bare]
    if not _BARE_NUMBER.match(bare):
        return bare
    return float(bare) if '.' in bare else int(bare)


def _field_value(raw: str) -> Any:
    pairs = _semicolon_pairs(raw)
    return _typed_value(raw) if pairs is None else pairs


def _pipe_record(text: Any) -> dict[str, Any] | None:
    """The dict a delimited listing line stands for; None when the text is not such a line (fewer than two
    separators, a first part without a key, or fewer than three keys). A part with no key of its own belongs to the
    value before it, since a name may contain the separator itself."""
    if not isinstance(text, str) or text.count(_PIPE) < 2:
        return None
    fields = _fold_pieces(text.split(_PIPE), _pipe_head, _PIPE, True)
    if not fields or len(fields) < 3:
        return None
    return {key: _field_value(raw) for key, raw in fields.items()}


def _expand_text_records(tree: Any, depth: int=0) -> Any:
    """A copy of `tree` in which every delimited listing line found in a list or dict has become its dict. Nesting
    deeper than six levels is returned untouched."""
    if depth > 6 or not isinstance(tree, (list, dict)):
        return tree

    def swapped(child: Any) -> Any:
        return _pipe_record(child) or _expand_text_records(child, depth + 1)

    if isinstance(tree, list):
        return [swapped(child) for child in tree]
    return {key: swapped(child) for key, child in tree.items()}


# ---------------------------------------------------------------- decoding model replies, encoding values

def _jdump(payload: Any) -> str:
    """Compact JSON (no spaces, non-ASCII left readable); str(payload) when json refuses the value."""
    try:
        text = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        text = str(payload)
    return text


def _first_json_object(answer: Any) -> dict[str, Any] | None:
    """The dict a model reply carries, or None. A surrounding code fence is dropped; the whole text is tried first
    and, only when it fails to parse, the slice from the first '{' to the last '}'. A parsed value that is not an
    object also gives None."""
    if not isinstance(answer, str):
        return None
    text = answer.strip()
    fenced = _CODE_FENCE.match(text)
    text = fenced.group(1) if fenced else text
    attempts = [text]
    left, right = text.find('{'), text.rfind('}')
    if 0 <= left < right:
        attempts.append(text[left:right + 1])
    for attempt in attempts:
        try:
            parsed = json.loads(attempt)
        except ValueError:
            continue
        return parsed if isinstance(parsed, dict) else None
    return None


def _fitted_dump(record: Any, cell: int, width: int) -> str:
    """A record as text for a judging model, at most `width` characters. A record that fits is rendered exactly as a
    plain clipped dump; one that does not keeps its own single-value fields (title, prose, remarks, price: where a
    page points elsewhere is written) whole up to `cell` each, and the nested blocks (the attribute map) absorb the cut,
    so the cut never drops the prose that comes after a long attribute block."""
    text = _jdump(_clip_texts(record, cell))
    if len(text) <= width or not isinstance(record, dict):
        return text[:width]
    single = {key: value for key, value in record.items() if not isinstance(value, (dict, list))}
    nested = {key: value for key, value in record.items() if isinstance(value, (dict, list))}
    head = _jdump(_clip_texts(single, cell))
    room = width - len(head) - 1
    if not nested or room < 40:
        return head[:width]
    tail = _jdump(_clip_texts(nested, cell))
    return head + ' ' + (tail if len(tail) <= room else tail[:room - 3] + '...')


def _clip_texts(tree: Any, limit: int, depth: int=0) -> Any:
    """A copy in which each string longer than `limit` is cut to `limit` characters plus '...'; whatever sits more
    than seven levels down turns into '...'."""
    if depth > 7:
        return '...'
    if isinstance(tree, str):
        return tree[:limit] + '...' if len(tree) > limit else tree

    def deeper(child: Any) -> Any:
        return _clip_texts(child, limit, depth + 1)

    if isinstance(tree, dict):
        return {key: deeper(child) for key, child in tree.items()}
    if isinstance(tree, list):
        return list(map(deeper, tree))
    return tree


# ---------------------------------------------------------------- squaring a value with a JSON schema

def _declared_types(schema: dict[str, Any]) -> list[str]:
    """The kinds a schema names under "type": a lone name as a one-item list, or the string entries of a list."""
    named = schema.get('type')
    if isinstance(named, str):
        return [named]
    return list(filter(lambda kind: isinstance(kind, str), named or ()))


def _matches_kind(value: Any, kind: str) -> bool:
    # bools never pass as numbers; a kind this table does not know lets every value through
    if kind in ('integer', 'number'):
        if isinstance(value, bool):
            return False
        return isinstance(value, int) if kind == 'integer' else isinstance(value, (int, float))
    shapes = {'string': str, 'boolean': bool, 'array': list, 'object': dict}
    if kind in shapes:
        return isinstance(value, shapes[kind])
    return value is None if kind == 'null' else True


def _kind_ok(value: Any, kinds: list[str]) -> bool:
    return not kinds or any(_matches_kind(value, kind) for kind in kinds)


def _text_as_number(given: Any, whole: bool) -> Any:
    if not isinstance(given, str) or not _SIGNED_NUM.fullmatch(given):
        return _UNTOUCHED
    number = float(given)
    return int(number) if whole or number.is_integer() else number


def _repaired_as(kind: str, given: Any) -> Any:
    """The cheap fix that turns `given` into `kind`, or _UNTOUCHED: an integral float or a number text for an
    integer, a number text for a number, a number written out for a string, "true"/"false" (any case, padded) for a
    boolean, a lone non-null value wrapped for an array."""
    if kind == 'integer':
        if isinstance(given, float) and given.is_integer():
            return int(given)
        return _text_as_number(given, True)
    if kind == 'number':
        return _text_as_number(given, False)
    if kind == 'string':
        is_number = isinstance(given, (int, float)) and not isinstance(given, bool)
        return str(given) if is_number else _UNTOUCHED
    if kind == 'boolean':
        word = given.strip().lower() if isinstance(given, str) else ''
        return word == 'true' if word in ('true', 'false') else _UNTOUCHED
    if kind == 'array':
        return [given] if given is not None and not isinstance(given, list) else _UNTOUCHED
    return _UNTOUCHED


def _coerce_toward(value: Any, kinds: list[str]) -> Any:
    """The first fix that works, trying the schema's kinds in their own order; the value itself when none does."""
    fixes = (_repaired_as(kind, value) for kind in kinds)
    return next((fixed for fixed in fixes if fixed is not _UNTOUCHED), value)


def _enum_stage(value: Any, schema: dict[str, Any], where: str, notes: list[str]) -> Any:
    choices = schema.get('enum')
    if not isinstance(choices, list) or value in choices:
        return value
    if isinstance(value, str):
        folded = value.strip().lower()
        spelled = [choice for choice in choices if isinstance(choice, str) and choice.lower() == folded]
        if spelled:
            return spelled[0]
    notes.append(f'{where}: must be one of {choices}')
    return value


def _string_stage(value: Any, schema: dict[str, Any], where: str, notes: list[str]) -> Any:
    if isinstance(value, str):
        cap, floor = schema.get('maxLength'), schema.get('minLength')
        if isinstance(cap, int) and len(value) > cap:
            # cut at a word boundary: a word cut in half matches nothing in a store that wants every word
            kept = value[:cap]
            if cap < len(value) and not value[cap].isspace() and ' ' in kept.strip():
                kept = kept.rstrip().rsplit(' ', 1)[0]
            value = kept.rstrip()
        if isinstance(floor, int) and len(value) < floor:
            notes.append(f'{where} is below the minimum length')
    return value


def _bounds_stage(value: Any, schema: dict[str, Any], where: str, notes: list[str]) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    low, high = schema.get('minimum'), schema.get('maximum')
    if isinstance(low, (int, float)) and value < low:
        notes.append(f'{where} is below the minimum {low}')
    if isinstance(high, (int, float)) and value > high:
        notes.append(f'{where} is above the maximum {high}')
    return value


def _list_stage(value: Any, schema: dict[str, Any], where: str, notes: list[str]) -> Any:
    if not isinstance(value, list):
        return value
    most, least = schema.get('maxItems'), schema.get('minItems')
    if isinstance(most, int) and len(value) > most:
        notes.append(f'{where} allows at most {most} entries')
    if isinstance(least, int) and len(value) < least:
        notes.append(f'{where} needs at least {least} entries')
    entry_schema = schema.get('items')
    if not isinstance(entry_schema, dict):
        return value
    return [_fit_to_schema(entry, entry_schema, f'{where}[{index}]', notes) for index, entry in enumerate(value)]


def _object_stage(value: Any, schema: dict[str, Any], where: str, notes: list[str]) -> Any:
    if not isinstance(value, dict):
        return value
    props = schema.get('properties')
    if not isinstance(props, dict):
        props = {}
    if schema.get('additionalProperties') is False:
        notes.extend(f'{where}.{key} is not an argument of this tool' for key in value if key not in props)
    notes.extend(f'{where}.{key} is required but missing' for key in schema.get('required') or () if key not in value)
    fitted = {}
    for key, inner in value.items():
        fitted[key] = _fit_to_schema(inner, props[key], f'{where}.{key}', notes) if key in props else inner
    return fitted


def _fit_to_schema(value: Any, schema: Any, where: str, notes: list[str]) -> Any:
    """`value` adjusted to a JSON schema wherever a cheap fix exists; each rule it still breaks goes into `notes`
    as a short line that starts with its path (`where`). A value of the wrong kind that no fix rescues is noted and
    handed back as it was, without the further checks."""
    if not isinstance(schema, dict):
        return value
    kinds = _declared_types(schema)
    if not _kind_ok(value, kinds):
        value = _coerce_toward(value, kinds)
        if not _kind_ok(value, kinds):
            wanted = '/'.join(kinds)
            notes.append(f'{where} needs type {wanted}')
            return value
    for stage in (_enum_stage, _string_stage, _bounds_stage, _list_stage, _object_stage):
        value = stage(value, schema, where, notes)
    return value


# ---------------------------------------------------------------- loose values: counts and lists

def _coerce_int(given: Any) -> int | None:
    """A whole number read from an int, a float with nothing after the point, or text made of digits (spaces and a
    ".0" tail allowed, e.g. " 42.00"); None for bools and anything else."""
    if isinstance(given, str):
        return int(float(given)) if _WHOLE_NUM.fullmatch(given) else None
    if isinstance(given, float):
        return int(given) if given.is_integer() else None
    if isinstance(given, int) and not isinstance(given, bool):
        return given
    return None


def _listify(given: Any) -> list[Any]:
    """A list as it is, a non-empty string or dict as a one-item list, anything else as an empty list."""
    if isinstance(given, list):
        return given
    return [given] if isinstance(given, (str, dict)) and given else []


# ---------------------------------------------------------------- ids and blocks inside store records

def _id_shaped(candidate: Any, id_column: bool=False) -> bool:
    """Could this be an id? A string that, trimmed, has 3 to 64 characters and no space, and that holds a digit
    unless it sits in a column already known to hold ids."""
    if not isinstance(candidate, str):
        return False
    bare = candidate.strip()
    if ' ' in bare or len(bare) < 3 or len(bare) > 64:
        return False
    return id_column or _ANY_DIGIT.search(bare) is not None


def _common_id_prefix(ids: list[str]) -> str:
    """The opening all ids have in common, trimmed back so it ends on a character that is not a letter or digit
    ("sku-71", "sku-9c" -> "sku-"); '' when nothing is shared or the list is empty."""
    if not ids:
        return ''
    shared = ids[0]
    for other in ids[1:]:
        length = 0
        limit = min(len(shared), len(other))
        while length < limit and shared[length] == other[length]:
            length += 1
        shared = shared[:length]
    while shared and shared[-1].isalnum():
        shared = shared[:-1]
    return shared


def _id_column_scores(rows: list[dict[str, Any]]) -> Any:
    """(key, score) for each key of the first record that could serve as the id column: every record's value there
    looks like an id and, with two records or more, no trimmed value repeats. The score ranks a key with "id" in its
    name first, then the longer average trimmed value."""
    for key in rows[0]:
        named = 'id' in str(key).lower()
        column = [row.get(key) for row in rows]
        if not all(_id_shaped(cell, named) for cell in column):
            continue
        cleaned = [cell.strip() for cell in column]
        if len(rows) > 1 and len(set(cleaned)) != len(cleaned):
            continue
        yield key, (named, sum(len(cell) for cell in cleaned) / len(cleaned))


def _infer_id_key(rows: list[dict[str, Any]]) -> str | None:
    """The key that best holds the rows' ids (best score; on a tie the key that comes first), or None."""
    best = max(_id_column_scores(rows), key=lambda scored: scored[1], default=None)
    return best[0] if best else None


def _id_among_rows(data: Any, target: str, depth: int=0) -> bool:
    """True when some dict that sits inside a list, five levels down at most, has a string field equal (once
    trimmed) to `target`."""
    if depth > 5:
        return False
    if isinstance(data, list):
        if any(_holds_listing_id(row, target) for row in data):
            return True
        inner = data
    elif isinstance(data, dict):
        inner = data.values()
    else:
        return False
    return any(_id_among_rows(child, target, depth + 1) for child in inner)


def _holds_listing_id(row: Any, wanted: Any) -> bool:
    if not isinstance(row, dict):
        return False
    return wanted in (str(cell).strip() for cell in row.values() if isinstance(cell, str))


def _trait_block(row: Any) -> dict[str, Any] | None:
    """The scalar fields (texts and non-bool numbers) of the row's nested dict that has the most of them, when
    that is two or more; the earlier dict wins a tie. None for a non-dict or when no nested dict qualifies."""
    if not isinstance(row, dict):
        return None
    best = None
    for inner in row.values():
        if not isinstance(inner, dict) or len(inner) < 2:
            continue
        flat = {key: cell for key, cell in inner.items()
                if isinstance(cell, (str, int, float)) and not isinstance(cell, bool)}
        if len(flat) >= 2 and (best is None or len(flat) > len(best)):
            best = flat
    return best


# ---------------------------------------------------------------- how keys and names are spelled

def _bare_words(label: Any) -> str:
    """A key as plain lower-case words: camelCase is split at each hump, and every run of whitespace, underscores,
    dots or hyphens becomes a single space."""
    spaced = _HUMP.sub(r'\1 \2', str(label))
    return _JOINERS.sub(' ', spaced).strip().lower()


def _to_snake(label: Any) -> str:
    return _bare_words(label).replace(' ', '_')


def _trim_s(token: str, keep_ss: bool) -> str:
    """The token without its final 's' when it has more than three letters; with keep_ss a final "ss" survives."""
    if len(token) > 3 and token.endswith('s') and not (keep_ss and token.endswith('ss')):
        return token[:-1]
    return token


def _stem_set(label: Any) -> frozenset[str]:
    return frozenset(_trim_s(token, False) for token in _bare_words(label).split())


def _plural_root(token: str) -> str:
    # an "-es" after ch, sh, ss, x or z goes as a whole ("benches" -> "bench"); otherwise one "s" may go
    if token.endswith(('ches', 'shes', 'sses', 'xes', 'zes')):
        return token[:-2]
    return _trim_s(token, True)


def _word_roots(label: Any) -> set[str]:
    """Roots of a label's words, keeping only those of three letters or more that are not and/for/the/with."""
    roots = set()
    for token in _bare_words(label).split():
        root = _plural_root(token)
        if len(root) >= 3 and root not in ('and', 'for', 'the', 'with'):
            roots.add(root)
    return roots


def _canonical_key(label: Any) -> frozenset[str]:
    """A name's words with a plural 's' trimmed and a closing British "our" turned into "or", so "flavours" and
    "flavor" give the same key."""
    stems = (_trim_s(token, True) for token in _bare_words(label).split())
    return frozenset(stem[:-3] + 'or' if stem.endswith('our') else stem for stem in stems)


def _same_trait_name(heard: frozenset[str], candidate: frozenset[str]) -> bool:
    """Both names must have words. They match when the heard name lies wholly inside the candidate, or when the
    candidate, if it has at least two words, lies wholly inside the heard name."""
    if not heard or not candidate:
        return False
    if heard <= candidate:
        return True
    return len(candidate) > 1 and candidate <= heard
class RunBase:
    """Core of a single Wren run.

    Keeps all episode state, watches the step counter, the clock and the money spent, talks to the model seats and
    to the store, reads the shopper's request into a brief during setup, and composes the two texts the driving
    model sees on every turn: the ledger of items and requirements, and the status line on steps and time.
    """

    def __init__(self, problem_data: dict[str, Any]) -> None:
        self._attach(problem_data['environment'])
        self._zero_counters()
        self._clear_flags()
        self._empty_registers()
        self._empty_id_sets()
        self._empty_logs()

    # ================================================================== store link

    def shop_call(self, tool: str, args: dict[str, Any]) -> dict[str, Any] | None:
        """Send one action to the store, at most four times. A step clash is re-synced and sent again, a final
        refusal stops at once; the parsed reply (or None) comes back."""
        packet = {**self._session_binding, 'step': self.moves_made + 1, 'action': {'name': tool, 'args': args}}
        for _round in range(4):
            answer = _shop_client.post_verbose('/environment/call', packet)
            if isinstance(answer.data, dict):
                return self._absorb_store_data(tool, args, answer.data)
            if not self._store_fault_retryable(answer.error or {}, packet):
                return None
        return None

    def _store_fault_retryable(self, problem: dict[str, Any], packet: dict[str, Any]) -> bool:
        """Log a failed store call and tell whether it may go out again: 409 follows the clash rules, a 402 also
        marks the wallet as empty, and the other listed client errors are final."""
        code = problem.get('status')
        said = str(problem.get('body') or '')
        self.journal.append({'kind': 'env_fail', 'status': code, 'body': said[:200]})
        if code == 409:
            return self._resync_after_clash(said, packet)
        final = code in (400, 401, 402, 403, 404, 422)
        if final and code == 402:
            self.wallet_empty = True
        return not final

    def _resync_after_clash(self, said: str, packet: dict[str, Any]) -> bool:
        """A step clash: a closed episode ends the call; otherwise adopt the step the store expects (or skip one
        past a reused number) and stamp the packet afresh."""
        lowered = said.lower()
        closed = any(mark in lowered for mark in ('finish', 'quarantin', 'finaliz'))
        guess = None if closed else self._clash_step_guess(said, lowered)
        self.run_closed = self.run_closed or closed
        if guess is not None:
            self.moves_made = guess
            packet['step'] = guess + 1
        return guess is not None

    def _clash_step_guess(self, said: str, lowered: str) -> int | None:
        expected = re.search('expected (\\d+)', said)
        if expected:
            return int(expected.group(1)) - 1
        reused = 'reuse' in lowered or 'different action' in lowered
        return self.moves_made + 1 if reused else None

    def _absorb_store_data(self, name: str, args: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Take over the store's step number and closing flag, unpack delimited records in the observation, and
        remember which ids a successful page or attribute view showed."""
        step_no = data.get('step')
        self.moves_made = step_no if isinstance(step_no, int) else self.moves_made + 1
        self.run_closed = bool(data.get('finished')) or self.run_closed
        observed = data.get('observation')
        if 'observation' in data:
            observed = data['observation'] = _expand_text_records(observed)
        if not (isinstance(observed, dict) and observed.get('error')):
            self._mark_viewed_ids(name, args)
        return data

    def _mark_viewed_ids(self, name: str, args: dict[str, Any]) -> None:
        """A page view files its string arguments under paged ids, the open view under attribute-read ids."""
        open_tool = self.role_map.get('open')
        if name in self._page_showing_tools():
            bucket = self.paged_ids
        elif open_tool is not None and name == open_tool:
            bucket = self.trait_read_ids
        else:
            bucket = None
        if bucket is not None:
            for value in args.values():
                for piece in value if isinstance(value, list) else (value,):
                    if isinstance(piece, str):
                        bucket.add(piece.strip())

    # ================================================================== model link

    def ask_model(self, dialogue: list[dict[str, Any]], seats: list[str], allowance: int, offered: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        """Walk the seats in order and hand back the first usable message; a lasting credit refusal stops the walk
        for good, a passing one only for this request."""
        for seat in seats:
            if not self._seat_allowed():
                return None
            self.model_calls_made += 1
            message, status = self._try_seat(seat, dialogue, allowance, offered)
            if message is not None:
                return message
            refused = status == 402
            if refused:
                # "retry later" / "too many in flight" refusals leave the credit flag down
                self.credit_gone = not self._last_refusal_is_passing()
                return None
        return None

    def ask_model_json(self, prompt: str, seats: list[str], allowance: int = 1800) -> dict[str, Any] | None:
        """One seat at a time, ask for a JSON object; give up early when money or the request cap is exhausted."""
        allowance = max(allowance, JSON_TOKEN_FLOOR)
        for lone_seat in seats:
            asked = [{'role': 'user', 'content': prompt}]
            found = _first_json_object((self.ask_model(asked, [lone_seat], allowance) or {}).get('content'))
            if found is not None or self._model_budget_gone():
                return found
        return None

    def _model_budget_gone(self) -> bool:
        return any((self.wallet_empty, self.credit_gone, self.model_calls_made >= MODEL_CALL_CEILING))

    def _seat_allowed(self) -> bool:
        """No once the request cap is hit; no (and logged) once too little wall clock remains for a request."""
        return self.model_calls_made < MODEL_CALL_CEILING and self._clock_permits_request()

    def _clock_permits_request(self) -> bool:
        late = self.model_time_allowance() < SHORTEST_CALL_SECONDS
        if late:
            self.journal.append({'kind': 'late_call_skipped', 'at': round(self.seconds_spent(), 1)})
        return not late

    def _last_refusal_is_passing(self) -> bool:
        """Whether the latest logged model failure reads as temporary (retry later, requests in flight) instead of
        an exhausted credit."""
        failures = (entry for entry in reversed(self.journal) if entry.get('kind') == 'llm_fail')
        wording = str(next(failures, {}).get('body') or '').lower()
        return any(mark in wording for mark in ('in_flight', 'in-flight', 'retry'))

    def _try_seat(self, seat: str, chat: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None, attempt: int = 0) -> tuple[dict[str, Any] | None, Any]:
        """Post to one seat; after a failure that the retry rules grant a pause for, sleep it (with jitter) and post
        again as the next attempt."""
        message, status = self._send_completion(seat, chat, max_tokens, tools)
        pause = None if message is not None else self._retry_pause(status, attempt)
        if pause is None:
            return message, status
        self._pause_spent += pause
        time.sleep(pause * (0.8 + 0.4 * random.random()))
        return self._try_seat(seat, chat, max_tokens, tools, attempt + 1)

    def _retry_pause(self, status: Any, attempt: int) -> Any:
        """The pause BACKOFF_SECONDS[attempt] when a failed try earns one more, else None.

        Worth a retry: a missing status, 402, 408, 425, 429 and any 5xx. A missing status and 408 get a single
        retry, the others one per BACKOFF_SECONDS entry; the pauses together stay within PAUSE_BUDGET, and after
        the pause there must still be time for a request.
        """
        transient = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
        tries = 1 if status in (None, 408) else len(BACKOFF_SECONDS)
        if not transient or attempt >= tries:
            return None
        pause = BACKOFF_SECONDS[attempt]
        within_budget = self._pause_spent + pause <= PAUSE_BUDGET
        if within_budget and not self.model_time_allowance() - pause * 1.2 < SHORTEST_CALL_SECONDS:
            return pause
        return None

    def _send_completion(self, seat: str, chat: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        """One completion request: (message, 200) when usable, else (None, status) with the failure logged."""
        payload = self._completion_payload(seat, chat, max_tokens, tools)
        client = _llm_client_within(self.model_time_allowance())
        result = client.post_verbose('/inference/chat/completions', payload)
        body = result.data
        if isinstance(body, dict) and body.get('choices'):
            return self._choice_message(seat, body)
        code, said = self._failure_of(result)
        self.journal.append({'kind': 'llm_fail', 'model': seat, 'status': code, 'body': str(said or '')[:200]})
        if 'reasoning' in payload and code == 400 and not self.effort_refused:
            # the route does not take the effort field: never send it again and resend this request without it
            self.effort_refused = True
            return self._send_completion(seat, chat, max_tokens, tools)
        return None, code

    def _failure_of(self, result: Any) -> tuple[Any, Any]:
        """Status and text of a failed request; an error object inside the reply body outranks the link's own."""
        link_fault = result.error or {}
        code, said = link_fault.get('status'), link_fault.get('body')
        body = result.data
        inline = body.get('error') if isinstance(body, dict) else None
        if isinstance(inline, dict):
            code, said = _coerce_int(inline.get('code')), inline.get('message')
        return code, said

    def _choice_message(self, seat: str, body: dict[str, Any]) -> tuple[dict[str, Any] | None, Any]:
        """Book the reported cost, then accept the first choice only if it carries text or tool calls."""
        charge = (body.get('usage') or {}).get('cost')
        if isinstance(charge, (int, float)):
            self.model_cost += float(charge)
        message = body['choices'][0].get('message') or {}
        if message.get('content') or message.get('tool_calls'):
            return message, 200
        self.journal.append(dict(kind='llm_fail', model=seat, status=0))
        return None, 0

    def _completion_payload(self, seat: str, chat: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> dict[str, Any]:
        """The request body. For Anthropic seats with more than one message, copies of the first and the last
        message get a cache mark on their text."""
        sent = chat
        if len(chat) > 1 and seat.startswith('anthropic/'):
            sent = [dict(entry) for entry in chat]
            for edge in (sent[0], sent[-1]):
                words = edge.get('content')
                if words and isinstance(words, str):
                    edge['content'] = [{'type': 'text', 'text': words, 'cache_control': {'type': 'ephemeral'}}]
        payload: dict[str, Any] = {'model': seat, 'messages': sent, 'max_tokens': max_tokens, 'temperature': 0}
        if tools:
            payload.update(tools=tools, tool_choice='auto')
            if self.brisk_turn and not self.effort_refused and seat.startswith('anthropic/'):
                payload['reasoning'] = {'effort': 'low'}
        return payload

    # ================================================================== budgets: steps, clock, messages, money

    def moves_left(self) -> int:
        return self.step_cap - self.moves_made

    def seconds_spent(self) -> float:
        return time.monotonic() - self._started_at

    def model_time_allowance(self) -> float:
        """Seconds a model request may still use: the run's time limit, less the cushion, less what has passed."""
        return RUN_TIME_BUDGET - CLOCK_CUSHION - self.seconds_spent()

    def free_message_count(self) -> int:
        """Shopper messages still free after setting aside those the request's own steps will need."""
        reserved = self._kept_for_process - self.procedure_messages
        return self.message_cap - self.messages_out - (reserved if reserved > 0 else 0)

    def wrap_up_reserve(self) -> int:
        """Steps to keep for the ending: one, or two while an order-time step still owes the shopper a message,
        plus one spare until the first checkout."""
        closing = 2 if self._closing_message_pending() else 1
        return closing + self._first_checkout_spare()

    def _closing_message_pending(self) -> bool:
        talks_at_order = any(entry['when'] == 'order' and entry['messages'] > 0 for entry in self.procedure_steps)
        return talks_at_order and self.messages_out < self.message_cap and self.procedure_check_due('order')

    def _first_checkout_spare(self) -> int:
        # Until a checkout has been tried, a filled cart keeps one step free so that a refused checkout (a price that
        # moved, a line that sold out, a requirement that changed) can still be dealt with. No cart look is reserved:
        # the checkout reply itself reports such changes.
        if self.purchase_tries:
            return 0
        return 1 if any(isinstance(units, int) and units > 0 for units in self.basket_counts.values()) else 0

    def procedure_check_due(self, phase: str) -> bool:
        """True when some request step belongs to this phase and it has not been passed for this cart and state."""
        belongs = any(entry['when'] == phase for entry in self.procedure_steps)
        return belongs and (phase, self.basket_fingerprint() + f'#{self.revision}') not in self.procedure_passes

    def basket_fingerprint(self) -> str:
        return _jdump(sorted(self.basket_counts.items()))

    def audit_seats(self, ceiling: float = SPEND_COMFORT) -> list[str]:
        """Judge seats the spend still pays for: strong and fast while below the ceiling with room for a strong
        request, only fast up to the hard cap, none beyond it."""
        used = self.model_cost
        if used <= ceiling and used + DEEP_SEAT_HEADROOM <= SPEND_CEILING:
            return [DEEP_SEAT, QUICK_SEAT]
        return [QUICK_SEAT] if used <= SPEND_CEILING else []

    # ================================================================== which tool shows what

    def _page_showing_tools(self) -> set[str]:
        """Tools that display a listing's own page: the offer view (or, lacking it, the open view) and the
        side-by-side view."""
        pair = (self.role_map.get('offer') or self.role_map.get('open'), self.role_map.get('compare'))
        return {tool for tool in pair if tool}

    def trait_viewer(self) -> str | None:
        """The first single-id tool known to show attributes; otherwise the open tool, unless it is known not to."""
        known = next((tool for tool, shows in self.tool_shows_traits.items() if shows and self.lone_id_arg(tool)), None)
        if known:
            return known
        open_tool = self.role_map.get('open')
        if self.tool_shows_traits.get(open_tool or '') is False:
            return None
        return open_tool if self.lone_id_arg(open_tool) else None

    def side_by_side_hides_traits(self) -> bool:
        """Whether the side-by-side view leaves attributes out: as recorded for it, else assumed when another view
        is known to carry them (a proven attribute viewer, or an offer view in the roles)."""
        verdict = self.tool_shows_traits.get(self.role_map.get('compare') or '')
        if verdict is not None:
            return verdict is False
        viewer_shows = self.tool_shows_traits.get(self.trait_viewer() or '') is True
        return viewer_shows or bool(self.role_map.get('offer'))

    # ================================================================== setup: the request becomes a brief

    def digest_request(self) -> None:
        """Read the request into a brief (one or two model passes) and load the brief into the run state."""
        brief = self._gather_brief(self._digest_prompt())
        self.digest = brief
        self._take_roles(brief)
        self._ensure_mode_args(brief)
        self._take_fit_pairs(brief, self._take_goods(brief))
        self._take_lookups(brief)
        self._merge_pointer_fragments()
        self._split_pointer_needs()
        self._scope_shared_needs()
        self._take_limits(brief)
        self._take_item_caps()
        self._take_process(brief)

    def _take_item_caps(self) -> None:
        """A price cap the request sets for one product ("I won't pay more than 6550 cents for an internal ssd") goes to
        the one item whose name shares the most word roots with the words after "for"/"on"; a tie, no shared root or a
        cap above the whole budget is ignored, and the lowest cap per item is kept."""
        pattern = (r"\b(?:won'?t|will not|wouldn'?t|would not|never|not)\s+(?:\w+\s+){0,2}?(?:pay|spend)\s+(?:\w+\s+){0,2}?"
                   r"(?:more than|over|above|beyond)\s+\$?(\d[\d,]*(?:\.\d+)?)[^,.;!?]{0,40}?\s(?:for|on)\s+"
                   r"(?:a|an|the|each|any|one|single)?\s*([^,.;!?]+)")
        whole = self.money_cap if self._is_number(self.money_cap) else None
        for found in re.finditer(pattern, self.shopper_request, re.I):
            amount = float(found.group(1).replace(',', ''))
            roots = _word_roots(' '.join(found.group(2).split()[:6]))
            shares = [len(roots & _word_roots(good.good_title)) for good in self.goods]
            best = max(shares, default=0)
            if not amount or best == 0 or shares.count(best) > 1 or (whole is not None and amount > whole):
                continue
            slot = shares.index(best) + 1
            if slot not in self.item_caps or amount < self.item_caps[slot]:
                self.item_caps[slot] = amount
                self.journal.append({'kind': 'item_cap', 'item': self.goods[slot - 1].good_title, 'cap': amount})

    def _scope_shared_needs(self) -> None:
        """A requirement the brief gave to two or more items stays only with the items named by the request's own
        sentences that state it (the property's words and its figures), when each such sentence names one of them as
        a whole. An item counts as named when every word of its name is there, unless those words stand once and only
        inside the name of a wider item named there too. The requirement comes off only an unnamed item that shares a
        name word with a named one (a requirement stated for "the X and Y" kind otherwise also lands on the "X" item,
        which then has no fitting listing)."""
        if len(self.goods) < 2:
            return
        pointed = {need for _, need in self.borrowed_needs}
        sentences = [part for part in re.split(r'(?<=[.!?;])\s+|\n+', self.shopper_request) if part.strip()]
        said = [[_plural_root(token) for token in _bare_words(part).split()] for part in sentences]
        names = [_word_roots(good.good_title) for good in self.goods]

        def figures(text: str) -> set[str]:
            return set(re.findall(r'\d+(?:\.\d+)?', text))

        def named(words: list[str]) -> set[int]:
            held = set(words)
            found = set()
            for slot, mine in enumerate(names):
                if not mine or not mine <= held:
                    continue
                if any(mine < wide <= held for wide in names) and all(words.count(root) == 1 for root in mine):
                    continue
                found.add(slot)
            return found
        shared = sorted({need for good in self.goods for need in good.given_needs if need not in pointed and ':' in need})
        for need in shared:
            holders = [slot for slot, good in enumerate(self.goods) if need in good.given_needs]
            if len(holders) < 2:
                continue
            prop, _colon, value = need.partition(':')
            wanted = _word_roots(prop)
            stating = [words for part, words in zip(sentences, said)
                       if wanted and wanted <= set(words) and figures(value) <= figures(part)]
            keep: set[int] = set()
            for words in stating:
                hit = named(words) & set(holders)
                if not hit:
                    keep = set()
                    break
                keep |= hit
            if not keep:
                continue
            for slot in holders:
                if slot not in keep and any(names[slot] & names[other] for other in keep):
                    self.goods[slot].given_needs.remove(need)
                    self.journal.append({'kind': 'need_scoped', 'need': need, 'from': self.goods[slot].good_title,
                                         'to': [self.goods[other].good_title for other in sorted(keep)]})

    def _digest_prompt(self) -> str:
        def catalogue_line(fn: dict[str, Any]) -> str:
            schema_text = _jdump(fn.get('parameters') or {})
            return f"- {fn['name']}: {fn.get('description', '')} | parameters: " + schema_text

        sections = [REQUEST_DIGEST_PROMPT, '\nTOOLS:\n', '\n'.join(catalogue_line(spec['function']) for spec in self.tool_list),
                    '\n\nREQUEST:\n', self.shopper_request]
        store_note = self.policy.get('opening')
        if store_note is not None:
            sections += ["\n\nTHE STORE'S OPENING NOTE:\n", _jdump(_clip_texts(store_note, 1500))]
        return ''.join(sections)

    def _gather_brief(self, text: str) -> dict[str, Any]:
        """The fast seat reads first. An incomplete brief is read again by strong, then fast; a complete one in
        which two items share a category label is put to the strong seat once more, and that reading wins only
        if it is complete."""
        draft = self.ask_model_json(text, [QUICK_SEAT], 4000) or {}
        if self.credit_gone:
            return draft
        if not self._brief_complete(draft):
            return self.ask_model_json(text, [DEEP_SEAT, QUICK_SEAT], 2500) or draft
        if not self._brief_has_twin_shelves(draft):
            return draft
        recheck = self.ask_model_json(text, [DEEP_SEAT], 2500) or {}
        return recheck if self._brief_complete(recheck) else draft

    def _brief_complete(self, brief: dict[str, Any]) -> bool:
        roles = brief.get('roles')
        return bool(_listify(brief.get('items'))) and isinstance(roles, dict) and bool(roles.get('buy'))

    def _brief_has_twin_shelves(self, brief: dict[str, Any]) -> bool:
        """Two items of the brief name the same category label."""
        labels = [label for label in (_bare_words(str(entry.get('shelf') or '')) for entry in _listify(brief.get('items')) if isinstance(entry, dict)) if label]
        return len(labels) != len(set(labels))

    def _take_roles(self, brief: dict[str, Any]) -> None:
        given = brief.get('roles')
        pairs = given.items() if isinstance(given, dict) else ()
        self.role_map = {role: name for role, name in pairs if isinstance(name, str) and name in self.schemas}

    def _ensure_mode_args(self, brief: dict[str, Any]) -> None:
        """Both mode-argument slots must exist. If one tool both orders and declines and its two argument sets are
        empty or the same, a seat is asked which fixed arguments select each mode."""
        for key in ('buy_args', 'refuse_args'):
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        tool = self.role_map.get('buy')
        if self._modes_unclear(tool, brief):
            self._ask_mode_args(tool, brief)

    def _modes_unclear(self, tool: Any, brief: dict[str, Any]) -> bool:
        """One tool serves both the order and the decline, and its two argument sets do not already differ."""
        shared = bool(tool) and tool == self.role_map.get('refuse')
        current = (brief['buy_args'], brief['refuse_args'])
        return shared and not (all(current) and current[0] != current[1])

    def _ask_mode_args(self, tool: str, brief: dict[str, Any]) -> None:
        answer = self.ask_model_json(self._mode_args_prompt(tool), [QUICK_SEAT, DEEP_SEAT], 1500) or {}
        to_buy, to_refuse = answer.get('buy_args'), answer.get('refuse_args')
        if isinstance(to_buy, dict) and isinstance(to_refuse, dict) and to_buy != to_refuse:
            brief['buy_args'], brief['refuse_args'] = to_buy, to_refuse

    def _mode_args_prompt(self, tool: str) -> str:
        return (f'The store tool {tool} does two jobs: given some arguments it places the order, given others it '
                'ends the episode without buying anything. The tool schema: ' + _jdump(self.schemas.get(tool))
                + '\nReply with ONE JSON object and no other text: {"buy_args": the fixed arguments that make it '
                'place the order, "refuse_args": the fixed arguments that make it end without a purchase} (only '
                'the arguments that pick the mode).')

    def _wanted_good(self, entry: dict[str, Any]) -> RequestedGood:
        quantity = _coerce_int(entry.get('quantity'))
        conditions = [str(c) for c in _listify(entry.get('musts')) if c]
        title, shelf = str(entry.get('name') or 'the product'), str(entry.get('shelf') or '')
        return RequestedGood(title, shelf, quantity if quantity and quantity > 0 else 1, conditions)

    def _take_goods(self, brief: dict[str, Any]) -> dict[int, int]:
        """Create the wanted products, fold together items on one category label, and return old->new numbers."""
        self.goods.extend(self._wanted_good(entry) for entry in _listify(brief.get('items')) if isinstance(entry, dict))
        self._counts_from_request()
        kept, renumber = self._merge_twins(self.goods)
        self.goods = kept or [RequestedGood('the requested product', '', 1, [])]
        return renumber

    def _counts_from_request(self) -> None:
        """"N x <product>" in the request sets that product's quantity: the one item whose name shares the most word
        roots with the first words after the sign takes N when the brief read another count. A figure right after the
        sign is a size ("3 x 5 inch"), and a tie or no shared root changes nothing."""
        for found in re.finditer(r'(?<![\w.])(\d{1,3})\s*[x\u00d7]\s+([^,;.:!?]+?)(?=$|[,;.:!?]|\s\d{1,3}\s*[x\u00d7]\s)',
                                 self.shopper_request, re.I):
            count, words = int(found.group(1)), found.group(2).strip()
            if not count or not words or words[0].isdigit():
                continue
            roots = _word_roots(' '.join(words.split()[:6]))
            shares = [len(roots & _word_roots(good.good_title)) for good in self.goods]
            best = max(shares, default=0)
            if best == 0 or shares.count(best) > 1:
                continue
            good = self.goods[shares.index(best)]
            if good.count_wanted != count:
                self.journal.append({'kind': 'count_from_request', 'item': good.good_title, 'was': good.count_wanted, 'now': count})
                good.count_wanted = count

    def _merge_twins(self, products: list[RequestedGood]) -> tuple[list[RequestedGood], dict[int, int]]:
        kept: list[RequestedGood] = []
        renumber: dict[int, int] = {}
        for old, good in enumerate(products, 1):
            slot = self._twin_slot(kept, good)
            if slot is None:
                kept.append(good)
                renumber[old] = len(kept)
            else:
                twin = kept[slot - 1]
                renumber[old] = slot
                twin.given_needs += [need for need in good.given_needs if need not in twin.given_needs]
                self.journal.append({'kind': 'items_merged', 'into': twin.good_title, 'dropped': good.good_title})
        return kept, renumber

    def _twin_slot(self, kept: list[RequestedGood], good: RequestedGood) -> int | None:
        """Number of the first kept item on the same category label whose requirements do not clash with these."""
        if not good.shelf_name.strip():
            return None
        label = _bare_words(good.shelf_name)
        return next((slot for slot, held in enumerate(kept, 1)
                     if _bare_words(held.shelf_name) == label and not self._needs_conflict(held.given_needs, good.given_needs)), None)

    def _needs_conflict(self, kept: list[str], incoming: list[str]) -> bool:
        """Both lists pin one property to different values."""
        def split_need(need: str) -> tuple[str, str]:
            prop, _sep, value = need.partition(':')
            return _bare_words(prop), _bare_words(value)
        ours = [split_need(need) for need in kept]
        theirs = [split_need(need) for need in incoming]
        return any(p == q and v != w for p, v in ours for q, w in theirs)

    def _take_fit_pairs(self, brief: dict[str, Any], renumber: dict[int, int]) -> None:
        total = len(self.goods)
        for link in (entry for entry in _listify(brief.get('fits')) if isinstance(entry, dict)):
            accessory, device = (renumber.get(_coerce_int(link.get(key)) or 0) for key in ('item', 'with'))
            fresh = accessory and device and accessory != device and (accessory, device) not in self.fit_links
            if fresh and 1 <= accessory <= total and 1 <= device <= total:
                self.fit_links.append((accessory, device))

    def _pin_lookup(self, text: str) -> str:
        """A lookup copied loosely gives way to the first requirement on the same property."""
        if any(text in good.given_needs for good in self.goods):
            return text
        wanted_prop = _bare_words(text.split(':')[0])
        same_prop = (need for good in self.goods for need in good.given_needs if _bare_words(need.split(':')[0]) == wanted_prop)
        return next(same_prop, text)

    def _take_lookups(self, brief: dict[str, Any]) -> None:
        named = self._request_proper_nouns()
        for raw in _listify(brief.get('lookups')):
            text = self._pin_lookup(str(raw).strip())
            if text and not set(re.findall("[a-z0-9']+", text.lower())) & named:
                # a listing elsewhere is always picked out by its shop or brand; a lookup naming none is a condition
                # between requested items or a detail the shopper holds back, and stays an ordinary requirement
                self.journal.append({'kind': 'lookup_not_pointer', 'need': text[:120]})
                continue
            owner = next((number for number, good in enumerate(self.goods, 1) if text in good.given_needs), 1)
            entry = (owner, text)
            if text and entry not in self.borrowed_needs:
                self.borrowed_needs.append(entry)

    def _request_proper_nouns(self) -> set[str]:
        """Words of three or more letters the request capitalises inside a sentence (not as its first word): the
        names of shops and brands."""
        out: set[str] = set()
        for sentence in re.split(r'(?<=[.!?;:])\s+|\n+', self.shopper_request):
            words = re.findall(r"[A-Za-z][A-Za-z0-9'\-]*", sentence)
            out.update(word.lower() for word in words[1:] if word[0].isupper() and len(word) >= 3)
        return out

    def _merge_pointer_fragments(self) -> None:
        """Two pointer requirements of one item that share the pointed listing's description (a run of four or more
        words) where the request writes one of them as "with <property> <value>": that one only describes the pointed
        listing (the request's "<property> is the one on the page of the X that <shop> carries, with <property>
        <value>, points to"), so it is folded into the other requirement's description instead of standing as a
        requirement of the item bought."""
        said = re.findall('[a-z0-9]+', self.shopper_request.lower())
        text = ' ' + ' '.join(said) + ' '
        for number, good in enumerate(self.goods, 1):
            pointed = [need for owner, need in self.borrowed_needs if owner == number and need in good.given_needs]
            for main in list(pointed):
                for part in list(pointed):
                    if part == main or part not in good.given_needs or main not in good.given_needs:
                        continue
                    a, b = re.findall('[a-z0-9]+', main.lower()), re.findall('[a-z0-9]+', part.lower())
                    run = self._longest_shared_run(a, b)
                    if run < 4:
                        continue
                    prop = re.findall('[a-z0-9]+', part.partition(':')[0].lower())
                    tail = re.findall('[a-z0-9]+', part.partition(':')[2].lower())
                    start = self._run_start(tail, a)
                    value = tail[:start] if start is not None else []
                    while value and len(value[-1]) <= 2 and not value[-1].isdigit():
                        value = value[:-1]
                    if not prop or not value or ' with ' + ' '.join(prop + value) + ' ' not in text:
                        continue
                    merged = main.rstrip(' .') + ', with ' + ' '.join(prop + value)
                    good.given_needs = [merged if need == main else need for need in good.given_needs if need != part]
                    self.borrowed_needs = [(owner, merged if need == main else need) for owner, need in self.borrowed_needs
                                           if not (owner == number and need == part)]
                    self.journal.append({'kind': 'pointer_fragment', 'item': good.good_title, 'folded': part, 'into': merged})
                    pointed = [merged if need == main else need for need in pointed if need != part]
                    main = merged

    @staticmethod
    def _longest_shared_run(a: list[str], b: list[str]) -> int:
        best = 0
        for i in range(len(a)):
            for j in range(len(b)):
                k = 0
                while i + k < len(a) and j + k < len(b) and a[i + k] == b[j + k]:
                    k += 1
                best = max(best, k)
        return best

    @staticmethod
    def _run_start(tail: list[str], other: list[str]) -> int | None:
        """Where, in `tail`, the first run of four or more words that `other` also holds begins."""
        for i in range(len(tail)):
            for j in range(len(other)):
                k = 0
                while i + k < len(tail) and j + k < len(other) and tail[i + k] == other[j + k]:
                    k += 1
                if k >= 4:
                    return i
        return None

    def _split_pointer_needs(self) -> None:
        """Per item, ask once which plain requirements merely describe the listing a pointer requirement names."""
        for number, good in enumerate(self.goods, 1):
            pointed = [need for owner, need in self.borrowed_needs if owner == number and need in good.given_needs]
            rest = [need for need in good.given_needs if need not in pointed] if pointed else []
            if rest:
                self._split_one_good(number, good, pointed, rest)

    def _split_one_good(self, number: int, good: RequestedGood, pointed: list[str], rest: list[str]) -> None:
        sections = [BORROWED_SPLIT_PROMPT, '\n\nREQUEST:\n', self.shopper_request,
                    f'\n\nPRODUCT BEING BOUGHT: {good.good_title}',
                    '\nREQUIREMENTS THAT TAKE THEIR VALUE FROM ANOTHER LISTING:\n', '\n'.join(pointed),
                    '\nOTHER REQUIREMENTS:\n', '\n'.join(rest)]
        answer = self.ask_model_json(''.join(sections), [DEEP_SEAT, QUICK_SEAT], 1500) or {}
        describes = {str(x).strip() for x in _listify(answer.get('describe_other'))}
        shifted = [need for need in rest if need in describes]
        if shifted:
            head = pointed[0]
            merged = head + ' (that listing: ' + '; '.join(shifted) + ')'
            good.given_needs = [merged if need == head else need for need in good.given_needs if need not in shifted]
            self.borrowed_needs = [(owner, merged if (owner, need) == (number, head) else need) for owner, need in self.borrowed_needs]
            self.journal.append(dict(kind='unmix', item=number, moved=shifted))

    def _budget_number(self, figure: Any) -> Any:
        """A positive budget figure (plain digit text is turned into a float), else None."""
        if isinstance(figure, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', figure):
            figure = float(figure)
        if isinstance(figure, bool) or not isinstance(figure, (int, float)):
            return None
        return figure if figure > 0 else None

    def _take_limits(self, brief: dict[str, Any]) -> None:
        cap_value = self._budget_number(brief.get('budget'))
        if cap_value is not None:
            self.money_cap = cap_value
        self.message_cap = self._message_room(_coerce_int(brief.get('messages')))
        steps = _coerce_int(brief.get('steps'))
        if steps is not None and 10 <= steps < self.step_cap:
            self.step_cap = steps

    def _message_room(self, declared: int | None) -> int:
        """The brief's message allowance when it gives one; three with a messaging tool; else unchanged."""
        if declared is not None and declared >= 0:
            return declared
        return 3 if self.role_map.get('ask') else self.message_cap

    def _process_entry(self, raw: Any) -> dict[str, Any] | None:
        if not isinstance(raw, dict) or not str(raw.get('step') or '').strip():
            return None
        phase = str(raw.get('when') or '').strip().lower()
        return {'step': str(raw['step']).strip(), 'when': phase if phase in ('choose', 'order', 'end') else 'order', 'messages': max(0, _coerce_int(raw.get('messages')) or 0)}

    def _take_process(self, brief: dict[str, Any]) -> None:
        """Store the request's own steps; with a messaging tool, set aside the messages the closing ones need."""
        for raw in _listify(brief.get('process')):
            entry = self._process_entry(raw)
            if entry is not None:
                self.procedure_steps.append(entry)
        if self.role_map.get('ask'):
            closing = [entry['messages'] for entry in self.procedure_steps if entry['when'] in ('order', 'end') and entry['messages']]
            self._kept_for_process = min(max(closing), self.message_cap) if closing else 0

    # ================================================================== requirements no attribute can state

    def _refers_elsewhere(self, need: str) -> bool:
        pointers = {held for _, held in self.borrowed_needs}
        pointers.update(str(x) for x in _listify(self.digest.get('lookups')))
        return need in pointers

    def _mentions_other_good(self, good: RequestedGood, value: str) -> bool:
        """The value shares a word of four or more letters with the name of another wanted product."""
        stems = _stem_set(value)
        return any(other is not good and stems & {w for w in _stem_set(other.good_title) if len(w) >= 4} for other in self.goods)

    def _attribute_key_census(self, good: RequestedGood) -> tuple[int, set[str]]:
        """Count the opened records of this item's category that have an attribute block, and gather their keys."""
        label_key = self.shelf_field_name()
        with_block, keys = 0, set()
        for rec in self.page_records.values():
            if label_key and not self.good_has_shelf(good, str(rec.get(label_key) or '')):
                traits = None
            else:
                traits = _trait_block(rec) or {}
            if traits:
                with_block += 1
                keys.update(_bare_words(k) for k in traits)
        return with_block, keys

    def _long_plain_value(self, prop: str, value: str) -> bool:
        """A named property whose value runs to four words or more and does not talk about other listings."""
        wordy = bool(prop.strip()) and len(re.findall('[a-z0-9]+', value.lower())) >= 4
        return wordy and not re.search('\\b(?:listing|page|shown|other than|unlike|same as)\\b', value, re.I)

    def _no_key_names(self, number: int, prop: str, value: str) -> bool:
        good = self.goods[number - 1]
        return not self._mentions_other_good(good, value) and self._keys_miss(good, prop)

    def _keys_miss(self, good: RequestedGood, prop: str) -> bool:
        """Three or more opened records of the item carry attributes, and none of their keys names the property."""
        with_block, keys = self._attribute_key_census(good)
        return with_block >= 3 and not any(_stem_set(k) & _stem_set(prop) for k in keys)

    def _unlisted_wording(self, number: int, need: str) -> bool:
        """A long requirement phrased in the shopper's words whose property matches no attribute key across three
        or more opened records of the item; the model sees it only as background."""
        prop, _sep, value = need.partition(':')
        return not self._refers_elsewhere(need) and self._long_plain_value(prop, value) and self._no_key_names(number, prop, value)

    # ================================================================== what the driving model reads

    def ledger_lines(self) -> str:
        several = len(self.goods) > 1
        rows = [self._ledger_good_row(number, good, several) for number, good in enumerate(self.goods, 1)]
        rows.extend(self._ledger_conditions())
        rows.extend(self._ledger_procedure())
        viewer_note = self._ledger_viewer_note()
        if viewer_note:
            rows.append(viewer_note)
        return '\n'.join(f'- {row}' for row in rows)

    def _ledger_good_row(self, number: int, good: RequestedGood, several: bool) -> str:
        live = [need for need in good.given_needs + good.told_needs if need not in good.dropped_needs]
        loose = [need for need in live if self._unlisted_wording(number, need)]
        firm = [need + self._borrowed_note(need) for need in live if need not in loose]
        pieces = [f'item {number}: {good.good_title}' if several else f'item: {good.good_title}']
        if good.shelf_name:
            pieces.append(f' (category label in the store: {good.shelf_name})')
        pieces.append(f', units wanted: {good.count_wanted}')
        pieces.append('; requirements: ' + ('; '.join(firm) if firm else 'none apart from being this sort of product'))
        if loose:
            pieces.append("; put by the shopper in words that no listing attribute holds (they decide nothing): " + '; '.join(loose))
        if good.dropped_needs:
            pieces.append('; DROPPED by the shopper since (no longer required, whatever the request says): '
                          + '; '.join(good.dropped_needs))
        cap = self.item_caps.get(number)
        if cap is not None:
            pieces.append(f'; price cap for one unit of this item: {cap:g}')
        return ''.join(pieces)

    def budget_in_unit(self, value: float) -> float:
        """A changed budget read from a reply or a store note, kept in the unit of the budget in force: a new figure
        within 30 times of the old one stands; otherwise the figure times or divided by 100 is taken when it lands
        within 3 times of the old one (a budget read in dollars where the request counts cents), else it stands."""
        old = self.money_cap
        if not self._is_number(old) or not old > 0 or not value > 0:
            return value
        if max(value / old, old / value) <= 30:
            return value
        for scaled in (value * 100, value / 100):
            if max(scaled / old, old / scaled) <= 3:
                self.journal.append({'kind': 'budget_unit', 'read': value, 'kept': scaled})
                return scaled
        return value

    def _borrowed_note(self, need: str) -> str:
        """After a chain resolved: the value the pointed page gives, and that the clause's other words only pick out
        that page (its shop, kind and values describe the other listing, not this item)."""
        if need not in self.borrowed_values:
            return ''
        prop = need.partition(':')[0].strip()
        return (f' (resolved: the page it points at gives "{prop}: {self.borrowed_values[need]}"; the other details '
                'in this clause only pick out that page and are not requirements on this item)')

    def _choice_rule_text(self, brief: dict[str, Any]) -> str:
        rule = brief.get('pick')
        if rule in ('most', 'least'):
            return f"selection: the {rule} {brief.get('pick_measure') or ''} among the listings that fit".replace('  ', ' ')
        if rule == 'cheapest':
            return 'selection: the lowest-priced listing that fits'
        return 'selection: any listing meeting every requirement'

    def _ledger_conditions(self) -> list[str]:
        """Budget, choice rule, stock, tie-breaking wishes, the request's instructions, the silent-decline wish."""
        facts = self.digest
        rows: list[str] = []
        if self.money_cap is not None:
            reach = 'for each item' if facts.get('budget_scope') == 'each' else 'across the whole order'
            rows.append(f"budget: at most {self.money_cap} {facts.get('budget_unit') or ''} {reach}".replace('  ', ' '))
        rows.append(self._choice_rule_text(facts))
        if facts.get('in_stock_only', True) is not False:
            rows.append('only listings that are in stock')
        rows += [f'wish (only to settle ties): {wish}' for wish in _listify(facts.get('wishes'))]
        rows += [f'the request instructs: {rule}' for rule in _listify(facts.get('rules'))]
        if facts.get('quiet_decline') is True:
            rows.append('if no listing fits: finish without mentioning or suggesting any listing')
        return rows

    def _ledger_procedure(self) -> list[str]:
        moment = {'choose': 'ahead of making the choice',
                  'order': 'together with the order (Wren performs it when you check out)',
                  'end': 'together with a no-purchase ending (Wren performs it at that point)'}
        return [f"required step from the request, {moment[entry['when']]}: {entry['step']}" for entry in self.procedure_steps]

    def _ledger_viewer_note(self) -> str:
        """Name the tool that shows attributes, the page views that show none, and the tool showing listing pages."""
        viewer = self.trait_viewer()
        named = bool(viewer) and (self.tool_shows_traits.get(viewer) is True or bool(self.role_map.get('offer')))
        return self._viewer_sentence(viewer) if named else ''

    def _viewer_sentence(self, viewer: str) -> str:
        page_tools = self._page_showing_tools()
        blind = sorted(t for t in page_tools | {self.role_map.get('offer') or ''} if t and self.tool_shows_traits.get(t) is False)
        pages = sorted(t for t in page_tools if t != viewer)
        note = f'listing attributes are shown by {viewer}'
        if blind:
            note += '; ' + ', '.join(blind) + ' show no attributes'
        if pages:
            note += "; a listing's page (what a page check or a chain of pages refers to) is shown by " + ' or '.join(pages)
        return note

    def status_line(self) -> str:
        remaining = self.moves_left()
        reserve = self.wrap_up_reserve()
        out = [f'[Wren] Steps used: {self.moves_made} out of {self.step_cap}; {remaining} still available.', self._step_warning(remaining, reserve)]
        if self.seconds_spent() >= RUSH_SECONDS and remaining > reserve + 1:
            out.append(' TIME IS NEARLY UP: cart the lines you picked and check out within the next two turns; a run'
                       ' that reaches the time limit earns no score.')
        else:
            out.append(self._pace_line(remaining, reserve))
        out += [self._status_addenda(), '\n[ledger]\n', self.ledger_lines()]
        return ''.join(out)

    def _pace_line(self, remaining: int, reserve: int) -> str:
        """Once this run's own turn times show the clock, not the step allowance, will end it (the turns that still
        fit before the last-turn mark, at two calls each, cover fewer steps than remain for searching): ask for every
        call already known to be needed in one turn. A turn costs about the same with one call or with four, so
        batching is how the remaining steps get used. Silent before two timed turns."""
        if len(self.turn_seconds) < 2:
            return ''
        per_turn = max(sum(self.turn_seconds) / len(self.turn_seconds), 1.0)
        turns_fit = max(0.0, LAST_TURN_SECONDS - self.seconds_spent()) / per_turn
        room = remaining - reserve - len(self.goods)
        if turns_fit * 2 >= room:
            return ''
        text = (f' TIME BEFORE STEPS: about {int(turns_fit)} more turns fit in the time left (a turn takes about '
                f'{per_turn:.0f} s here, and each further call inside the same turn adds only a second or two), far '
                f'fewer than the {remaining} steps left. Send every search, next page and listing read you already '
                'know you need in this one turn as separate calls, and keep your text short.')
        lacking = self._goods_without_line() if self.basket_counts else list(range(1, len(self.goods) + 1))
        if lacking and len(self.goods) >= 2:
            text += ' No cart line yet for: ' + '; '.join(
                f'item {n} ({self.goods[n - 1].good_title})' for n in lacking[:6]) + '.'
        return text

    def _step_warning(self, remaining: int, reserve: int) -> str:
        """The warning that fits how few steps are left and whether a checkout has been tried."""
        no_checkout_yet = self.purchase_tries == 0
        ladder = (
            (remaining <= reserve + 1,
             ' FINAL STEPS: once every item has its cart line, check out immediately; finish without buying only if'
             ' your searches showed that some item has no fitting listing at all (running short of steps shows no'
             ' such thing).'),
            (remaining <= 6 or (remaining <= len(self.goods) + 5 and no_checkout_yet),
             ' Very few steps are left: cart the lines you picked and check out now.'),
            (no_checkout_yet and remaining <= 12,
             ' Steps are running low: settle on your picks, fill the cart and check out within the coming few steps.'),
            (no_checkout_yet and not self.basket_touched and self.moves_made >= self.step_cap // 2,
             ' Half the step allowance is used and the cart is still empty: cart a fitting listing for each item right'
             ' away (a listing may still sell out).'),
        )
        return next((text for hit, text in ladder if hit), '')

    def _status_addenda(self) -> str:
        """Open pointer look-ups, review leads, the fit plan and the latest changes."""
        in_force = {need for _, need in self.needs_in_force()}
        no_checkout_yet = self.purchase_tries == 0
        open_lookups = [need for _, need in self.borrowed_needs if need in in_force and need not in self.borrowed_values]
        pieces: list[str] = []
        if open_lookups and no_checkout_yet:
            pieces.append('\n[resolve these first: open the listing each one refers to, then every listing that page'
                          ' sends you to; the last page holds the value. The shop, kind and values named in such a'
                          ' clause only pick out the listing to look up: they are not requirements on the item you'
                          ' buy] ' + ' | '.join(open_lookups[:3]))
        if self.audit_leads:
            pieces.append('\n[leads worth checking] ' + ', '.join(self.audit_leads))
        route = self.fit_route_text() if self.fit_links and no_checkout_yet else ''
        if route:
            pieces.append('\n' + route)
        if self.shift_notes:
            pieces.append('\n[changes so far] ' + ' | '.join(self.shift_notes[-4:]))
        return ''.join(pieces)

    # ================================================================== fresh episode state

    def _attach(self, environment: dict[str, Any]) -> None:
        """Bind the session, keep the policy view, the usable tool specs and the step cap; start the clock."""
        binding, self.policy = environment['binding'], environment['policy_view']
        self._session_binding = dict(binding)
        self.shopper_request = str(self.policy.get('query') or '')
        self.tool_list = [spec for spec in self.policy.get('tools') or [] if self._names_a_function(spec)]
        self.schemas = {}
        for spec in self.tool_list:
            params = spec['function'].get('parameters')
            self.schemas[spec['function']['name']] = params if isinstance(params, dict) else {'type': 'object', 'properties': {}}
        self.step_cap = self._step_allowance(self.policy.get('max_steps'))
        self._started_at = time.monotonic()

    def _names_a_function(self, spec: Any) -> bool:
        fn = spec.get('function') if isinstance(spec, dict) else None
        return isinstance(fn, dict) and isinstance(fn.get('name'), str)

    def _step_allowance(self, raw: Any) -> int:
        """The declared step limit; a fractional float is truncated, and anything unusable means 30."""
        value = _coerce_int(raw)
        if value is None:
            value = int(raw) if isinstance(raw, float) else 0
        return value if value > 0 else 30

    def _zero_counters(self) -> None:
        """Integer counters, and the step markers that start at -1 (nothing happened yet)."""
        self.moves_made = self.model_calls_made = self.driver_turns = self.turn_failures = self._pause_spent = 0
        self.model_cost = 0.0
        self.messages_out = self.message_cap = self.procedure_messages = self._kept_for_process = 0
        self.purchase_tries = self.revision = self.auto_pages_done = self.empty_answers = 0
        self.purchase_refused_at = self.basket_changed_at = self.basket_seen_at = -1
        self.fit_pause_count = self.sold_pause_count = self.line_pause_count = 0
        self._decline_attempts = self.refusal_block_count = self.link_query_count = 0
        self.trace_block_count = self.trace_rounds = self.side_fail_width = self.side_ok_width = 0

    def _clear_flags(self) -> None:
        """Yes/no flags, and the slots that hold one value or nothing."""
        self.run_closed = bool(self.policy.get('finished'))
        self.credit_gone = self.wallet_empty = self.basket_touched = self.lead_purchased = False
        self.late_refusal_held = self.refusal_pushed = self.refusal_audited = self.wider_hunt_done = False
        self.notice_says_sold = self.notice_named_id = self.final_question_spent = self.interview_ran = False
        self.lists_ok = True
        # list questions so far, the longest list answered without complaint, and whether a need came from past a
        # list's first place (the shopper reads whole lists)
        self.list_asks, self.list_taken, self.lists_read = 0, 0, False
        self.stock_key = self.price_key = self.row_id_key = self.money_cap = self.backed_lead = self.basket_view = None
        self.last_basket_reply = ''

    def _empty_registers(self) -> None:
        """Keyed stores: the brief, roles, records, tool facts, cart counts, reviews and question lanes."""
        self.digest, self.role_map, self.borrowed_values = {}, {}, {}
        self.page_records, self.row_records, self.read_cache = {}, {}, {}
        self.tool_key_sets, self.tool_shows_traits, self.seen_view_keys, self.reply_shapes = {}, {}, {}, {}
        self.sold_out_steps, self.offer_seen_at, self.basket_counts = {}, {}, {}
        self.audit_verdicts, self.leads_per_good, self.link_query_hits = {}, {}, {}
        self.question_tracks, self.track_asked, self.track_hits = {}, {}, {}
        self.track_messages, self.track_dry_streak, self.adopted_shelves = {}, {}, {}

    def _empty_id_sets(self) -> None:
        """Membership sets: passed process checks, ids seen in each view, notes already taken in, and so on."""
        self.procedure_passes, self.audited_fingerprints, self.unbuyable_ids = set(), set(), set()
        self.trait_read_ids, self.paged_ids, self.id_args = set(), set(), set()
        # dropped_needs: ids of listings the model removed from the cart; the turn loop reads it
        self.dropped_needs, self.lower_price_marks, self.lone_goods = set(), set(), set()
        self.sellout_notice_lines, self.seen_notice_lines = set(), set()
        self.early_trace_marks, self.ref_codes_tried, self.capped_queries = set(), set(), set()

    def _empty_logs(self) -> None:
        """Ordered lists: the journal, products, links, sightings, cart drafts, leads, questions and the trail."""
        self.journal, self.goods, self.fit_links = [], [], []
        self.borrowed_needs, self.procedure_steps, self.seen_sequence = [], [], []
        self.extra_pages, self.composed_lines, self.audit_leads, self.shift_notes = [], [], [], []
        self._question_pairs, self.queued_followups, self.hinted_good_order = [], [], []
        self.convo_log, self.act_steps, self.act_log = [], [], []
        self.pending_notes, self.recent_marks, self.recent_queries = [], [], []
        # per search tool: numeric arguments learned to skip matches (offsets), whatever their names say
        self.offset_args: dict[str, set[str]] = {}
        # long searches that matched nothing and were sent again with their first words; late-decline checkouts
        self.short_retries, self.late_checkout_tried = 0, False
        # seconds each store-loop model turn took, for the pace line
        self.turn_seconds: list[float] = []
        # low reasoning effort on late store-loop turns, and whether the route refused that field
        self.brisk_turn, self.effort_refused = False, False
        # pointer chains: what Wren checked on the records (shown to the tracer), next pages of link searches,
        # whether the chains were followed right after the interview
        self.trail_checks: list[str] = []
        self.link_query_more: dict[str, Any] = {}
        self.trace_again, self.early_trail_done = False, False
        # pages opened when the chains were last followed; whether the decline-time chain round ran
        self.trace_paged_mark, self.refusal_trail_done = 0, False
        self.cut_query_noted = False
        # cheaper read listings the lowest-price hold named per cart line; whether code made an agreed swap
        self.cheaper_named: dict[str, str] = {}
        self.agreed_swap_done = False
        # per-item price caps the request states; the code order of a fully proven set (tried once)
        self.item_caps: dict[int, float] = {}
        self.code_order_tried = False
        self.proven_decline_held = False
        # the attributes of the listing being judged, for the value check's look at the other variants
        self._rv_cells_beside: dict[str, Any] = {}
# ---------------------------------------------------------------------- listing helpers that need no run state

def _real_number(value: Any) -> bool:
    """An int or a float; a bool does not count."""
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _whole_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int)


def _label_shaped(value: Any) -> bool:
    """Text of 1 to 40 characters holding no run of three digits."""
    if not isinstance(value, str) or not value or len(value) > 40:
        return False
    return re.search('\\d{3}', value) is None


def _price_from_cell(cell: Any) -> float | None:
    """A price cell as a float: a nested dict yields its first number, digits-only text is read as a number."""
    if isinstance(cell, dict):
        cell = next((part for part in cell.values() if _real_number(part)), None)
    if isinstance(cell, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', cell):
        cell = float(cell)
    return float(cell) if _real_number(cell) else None


def _lists_of_rows(observed: Any) -> list[str]:
    """Keys of a reply dict whose value is a list of two or more dicts."""
    if not isinstance(observed, dict):
        return []
    return [key for key, value in observed.items()
            if isinstance(value, list) and len(value) >= 2 and all(isinstance(row, dict) for row in value)]


def _layout_prompt(tool: str, schema: dict[str, Any], observed: dict[str, Any], row_lists: list[str]) -> str:
    preview = {key: value[:3] if key in row_lists else value for key, value in observed.items()}
    return ''.join((REPLY_SHAPE_PROMPT, '\n\nTOOL: ', tool, ' | its parameter schema: ', _jdump(schema),
                    '\n\nRESULT (first rows only):\n', _jdump(_clip_texts(preview, 120))[:4000]))


def _stock_flip(before: dict[str, Any], after: dict[str, Any]) -> bool | None:
    """False when some yes/no field went from True to False (the listing left stock), True when one went from False
    to True (it came back), None otherwise; leaving outranks coming back."""
    turned = {now for key, now in after.items() if isinstance(now, bool) and before.get(key) is (not now)}
    if False in turned:
        return False
    return True if True in turned else None


def _shelf_args(props: dict[str, Any]) -> list[str]:
    """Free-text arguments (string, no enum) whose name reads as a category, department or the like."""
    return [name for name, spec in props.items()
            if isinstance(spec, dict) and 'string' in _declared_types(spec) and 'enum' not in spec
            and re.search(SHELF_ARG_RX, _to_snake(name), re.I)]


def _money_not_floor(name: str, spec: dict[str, Any]) -> bool:
    gist = _to_snake(name).replace('_', ' ') + ' ' + str(spec.get('description') or '')
    return bool(re.search(CASH_ARG_RX, gist, re.I)) and not re.search(FLOOR_ARG_RX, gist, re.I)


def _gloss_says_browse(option: Any, sources: list[str]) -> bool:
    """Whether some text glosses this option ("match = filter search") as a listing or filter mode, not a word search."""
    if not isinstance(option, str) or not option.strip():
        return False
    pattern = MODE_EXPLAIN_RX.format(choice=re.escape(option))
    for text in sources:
        for hit in re.finditer(pattern, text, re.I):
            gloss = hit.group(2)
            if re.search(LISTING_MODE_RX, gloss, re.I) and not re.search(WORDS_MODE_RX, gloss, re.I):
                return True
    return False


def _filled_words(past: dict[str, Any], props: dict[str, Any], shelf: str) -> bool:
    """Whether an earlier call filled a free-text word argument (neither the category one nor an enum)."""
    for name, value in past.items():
        spec = props.get(name)
        if name == shelf or (isinstance(spec, dict) and 'enum' in spec):
            continue
        if isinstance(value, str) and value.strip() and re.search(WORDS_ARG_RX, _to_snake(name), re.I):
            return True
    return False


def _param_mentions_money(props: dict[str, Any], name: str) -> bool:
    spec = props.get(name)
    blurb = (spec.get('description') if isinstance(spec, dict) else None) or ''
    return bool(re.search('price|cost|amount|budget|spend|money', name + ' ' + str(blurb), re.I))


def _cheaper_mode_hint(call_args: dict[str, Any], props: dict[str, Any], cap_name: str) -> str:
    swaps: dict[str, list[Any]] = {}
    for name, value in call_args.items():
        if not isinstance(value, str):
            continue
        options = (props.get(name) or {}).get('enum')
        if isinstance(options, list):
            swaps[name] = [option for option in options if option != value]
    if not any(swaps.values()):
        return ''
    modes = ', '.join(f'{name}={_jdump(rest[0])}' for name, rest in swaps.items() if rest)
    return f'; to see nothing but cheaper listings, browse the category with {modes} and leave {cap_name} unchanged'


def _tally_columns(columns: dict[str, dict[str, Any]], ident: str, attrs: dict[str, Any]) -> None:
    """Add one opened listing's attributes to the per-key columns (each plain key name once per listing)."""
    done: set[str] = set()
    for key_text, cell in attrs.items():
        name = _bare_words(key_text)
        if not name or name in done:
            continue
        done.add(name)
        column = columns.setdefault(name, {'hits': 0, 'texts': [], 'by_id': {}})
        column['hits'] += 1
        flat = str(cell).strip().lower()
        column['by_id'][ident] = flat
        if flat not in column['texts']:
            column['texts'].append(flat)


def _column_too_noisy(texts: list[str]) -> bool:
    """Too noisy to offer: values averaging over 40 characters, slash lists on at least half the listings, or four or
    more values, all different and averaging over 12 characters, of which fewer than half share a word with the
    others."""
    count = len(texts)
    avg_len = sum(len(text) for text in texts) / count
    if avg_len > 40 or 2 * sum(' / ' in text for text in texts) >= count:
        return True
    if count < 4 or len(set(texts)) != count or avg_len <= 12:
        return False
    bags = [set(re.findall('[a-z]{3,}', text)) for text in texts]
    echoed = 0
    for pos, bag in enumerate(bags):
        rest = set().union(*(other for at, other in enumerate(bags) if at != pos))
        echoed += bool(bag & rest)
    return 2 * echoed < count


def _fold_column(twin: dict[str, Any], column: dict[str, Any]) -> None:
    for ident, flat in column['by_id'].items():
        twin['by_id'].setdefault(ident, flat)
        if flat not in twin['texts']:
            twin['texts'].append(flat)
    twin['hits'] = len(twin['by_id'])


def _texts_match(first: str, second: str) -> bool:
    """Equal texts, or every word of the shorter found in the longer, which has at most twice as many plus two."""
    if first == second:
        return True
    short, long_ = sorted((re.findall('[a-z0-9]+', first.lower()), re.findall('[a-z0-9]+', second.lower())), key=len)
    return bool(short) and set(short) <= set(long_) and len(long_) <= 2 * len(short) + 2


def _same_attribute_columns(old_name: str, old: dict[str, Any], new_name: str, new: dict[str, Any]) -> bool:
    """Two keys sharing a word name one attribute when their values agree on the listings both carry (on all of at
    least two when neither name holds the other, on a majority otherwise) or, sharing no listing, when they have a
    non-empty value in common."""
    old_words, new_words = set(old_name.split()), set(new_name.split())
    if not old_words & new_words:
        return False
    mine, theirs = old['by_id'], new['by_id']
    common = [ident for ident in mine if ident in theirs and mine[ident].strip() and theirs[ident].strip()]
    agree = sum(1 for ident in common if _texts_match(mine[ident], theirs[ident]))
    if not (old_words <= new_words or new_words <= old_words):
        return len(common) >= 2 and agree == len(common)
    if common:
        return agree * 2 > len(common)
    return bool((set(old['texts']) & set(new['texts'])) - {''})


def _other_trait_named(prop: str, target: str, key_name: str, visible: set[str]) -> bool:
    """A plain required value (no negation, bound or alternative) whose property shares a word stem with the key under
    a different name, and whose words are not all among the key's values."""
    if re.search('\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', target, re.I):
        return False
    if not prop.strip() or _bare_words(prop) == key_name or not _stem_set(prop) & _stem_set(key_name):
        return False
    target_words = set(re.findall('[a-z0-9]{3,}', target.lower()))
    return bool(target_words) and not target_words <= visible


class RunListings:
    """Wren's memory of the store's listings for one run.

    It decides which dicts in a store reply are listings and under which id field, keeps every listing's merged row
    and (once opened) its full page, reads prices and stock, groups the attribute keys of opened pages by category
    label, learns how each tool lays out a search result (row list, match total, next-page token), and makes the
    store calls Wren sends on its own: further result pages, the listing a cart line's page refers to, and category
    listings capped just under a cart line's price.
    """

    # ================================================================== small readings

    def _is_number(self, value: Any) -> bool:
        return _real_number(value)

    def _just_under(self, amount: float) -> int:
        """A whole-number price cap that leaves the given price out: one less when the price is whole, otherwise its
        whole part."""
        floor = int(amount)
        return floor - 1 if float(amount).is_integer() else floor

    def _past_first_page(self, requested: Any) -> bool:
        """Whether a paging argument already points beyond page one: a whole number other than 0 or 1 does, and so
        does any token or cursor that is not blank."""
        if _whole_number(requested):
            return requested not in (0, 1)
        return str(requested or '').strip() != ''

    def price_of_listing(self, ident: str) -> float | None:
        key = self.price_key
        if not key:
            return None
        return _price_from_cell((self.row_records.get(ident) or {}).get(key))

    def is_purchasable(self, ident: str) -> bool:
        if self.sold_out_now(ident):
            return False
        stock_flag = (self.row_records.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        return stock_flag is not False and not (_real_number(stock_flag) and stock_flag <= 0)

    def _rows_cheapest_first(self, observed: Any) -> bool:
        """True when the reply's rows (at least three, every one priced) already run from cheapest to dearest."""
        costs = [self.price_of_listing(ident) for ident, _ in self.records_in_reply(observed)]
        if len(costs) < 3 or None in costs:
            return False
        return all(low <= high for low, high in zip(costs, costs[1:]))

    # ================================================================== r16: listing a category under a price

    def _price_capped_browser(self) -> tuple[str, dict[str, Any], str, str] | None:
        """(tool, mode arguments, category argument, price-cap argument) for the first tool whose parameters hold
        exactly one free-text category argument and exactly one price cap. Its mode arguments repeat the newest
        category listing made without search words, or else take the single option that names a listing or browse
        mode. None when no tool qualifies, or when the first tool that does leaves a mode undecided."""
        for candidate, schema in self.schemas.items():
            props = (schema or {}).get('properties') or {}
            caps, shelves = self._cap_arg_candidates(props), _shelf_args(props)
            if (len(caps), len(shelves)) == (1, 1):
                modes = self._fill_mode_args(props, shelves[0], candidate)
                return None if modes is None else (candidate, modes, shelves[0], caps[0])
        return None

    def _cap_arg_candidates(self, props: dict[str, Any]) -> list[str]:
        """Number arguments able to carry a price ceiling, tried in three tiers and the first tier with a hit wins:
        names that read as a cap; then names or descriptions about money that say nothing of a minimum; then the
        argument an earlier search filled with the request's budget. [] when no tier finds one."""
        numbers = [name for name, spec in props.items()
                   if isinstance(spec, dict) and {'integer', 'number'} & set(_declared_types(spec))
                   and not re.search(NON_CAP_ARG_RX, name, re.I)]
        stated = self.digest.get('budget')
        tiers = (lambda name: bool(re.search(CAP_ARG_RX, _to_snake(name), re.I)),
                 lambda name: _money_not_floor(name, props[name]),
                 lambda name: self._filled_with_budget(name, stated))
        for passes in tiers:
            hits = [name for name in numbers if passes(name)]
            if hits:
                return hits
        return []

    def _filled_with_budget(self, name: str, stated: Any) -> bool:
        if not _real_number(stated) or stated <= 0:
            return False
        return any(past.get(name) == stated for past in self.recent_queries)

    def _fill_mode_args(self, props: dict[str, Any], shelf: str, tool: str) -> dict[str, Any] | None:
        """A value for every enum argument with two or more options, or None as soon as one stays open."""
        chosen: dict[str, Any] = {}
        for name, spec in props.items():
            options = spec.get('enum') if isinstance(spec, dict) else None
            if isinstance(options, list) and len(options) >= 2:
                picked = self._pick_mode(name, options, props, shelf, tool)
                if picked is None:
                    return None
                chosen[name] = picked
        return chosen

    def _pick_mode(self, name: str, options: list[Any], props: dict[str, Any], shelf: str, tool: str) -> Any:
        """One mode argument's value: what the newest word-free category listing used; failing that, the one option
        whose own text names a listing mode; failing that, the one option a description glosses that way."""
        reused = next((past[name] for past in reversed(self.recent_queries)
                       if shelf in past and name in past and not _filled_words(past, props, shelf)), None)
        if reused is not None:
            return reused
        listed = [option for option in options if isinstance(option, str) and re.search(LISTING_MODE_RX, option, re.I)]
        if len(listed) != 1:
            listed = self._glossed_browse_modes(name, options, props, tool)
        return listed[0] if len(listed) == 1 else None

    def _glossed_browse_modes(self, name: str, options: list[Any], props: dict[str, Any], tool: str) -> list[Any]:
        """Options that the argument's own description or the tool's explains as a listing or filter mode rather than
        a word search."""
        own = props.get(name) if isinstance(props.get(name), dict) else {}
        sources = [str(own.get('description') or '')]
        sources += [str(entry['function'].get('description') or '') for entry in self.tool_list
                    if entry['function'].get('name') == tool]
        return [option for option in options if _gloss_says_browse(option, sources)]

    def _browse_under_line_prices(self, line_ids: list[str]) -> None:
        """For each cart line, list its category with the price cap just below the line's price (once per line and
        price); a second page is read when the first shows more than half of the matches, unsorted."""
        capped = self._price_capped_browser()
        label_field = self.shelf_field_name()
        if not capped or not label_field:
            return
        tool, modes, shelf, cap = capped
        for line_id in line_ids:
            ceiling = self.price_of_listing(line_id)
            label = str((self.row_records.get(line_id) or {}).get(label_field) or '')
            if ceiling is None or ceiling < 2 or not label or (line_id, ceiling) in self.capped_queries:
                continue
            if self.moves_left() <= self.wrap_up_reserve() + 3 or self.run_closed:
                return
            if not self._browse_shelf_under(tool, modes, shelf, cap, line_id, label, ceiling):
                return

    def _browse_shelf_under(self, tool: str, modes: dict[str, Any], shelf: str, cap: str, line_id: str, label: str,
                            ceiling: float) -> bool:
        self.capped_queries.add((line_id, ceiling))
        sent, reply = self._send_fitted(tool, {**modes, shelf: label, cap: self._just_under(ceiling)})
        if reply is None:
            return False
        listing_page = reply.get('observation')
        self.digest_reply(tool, listing_page)
        self.journal.append(dict(kind='list_below', line=line_id, price=ceiling))
        self._capped_follow_page(tool, sent, listing_page)
        return True

    def _capped_follow_page(self, tool: str, sent: dict[str, Any], listing_page: Any) -> None:
        """Read one more page of a capped listing when the first showed some but not all matches, when it showed
        over half of them or more than eight steps are spare, when more than five are spare at all, and when the
        rows are not already in price order."""
        total, shown, param, token = self._paging_state(tool, listing_page) or (0, 0, None, None)
        if token is None or shown >= total:
            return
        spare = self.moves_left() - self.wrap_up_reserve()
        if total > 2 * shown and spare <= 8:
            return
        if spare <= 5 or self._rows_cheapest_first(listing_page):
            return
        _, more = self._send_fitted(tool, {**sent, param: token})
        if more is not None:
            self.digest_reply(tool, more.get('observation'))

    # ================================================================== store calls and what they show

    def _send_fitted(self, tool: str, raw_args: dict[str, Any]) -> tuple[Any, dict[str, Any] | None]:
        """Shape the arguments to the tool's schema, then call the store; gives (shaped arguments, reply). The reply
        is None when shaping failed (nothing was sent) or the store returned nothing."""
        issues: list[str] = []
        shaped = _fit_to_schema(raw_args, self.schemas.get(tool), 'args', issues)
        if issues:
            return shaped, None
        return shaped, self.shop_call(tool, shaped)

    def digest_reply(self, tool: str, observed: Any) -> str:
        """Keep the listings a reply carries and pass its loose text to the note reader; when the reader reports a
        change, it is queued for the model with the tool it came from."""
        self.ingest_records(observed)
        loose = self.loose_text_lines(observed)
        if not loose:
            return ''
        change = self.interpret_shop_note(loose)
        if change:
            self.pending_notes.append(f'(seen via {tool}) {change}')
        return change

    def ingest_records(self, observed: Any) -> None:
        self._walk_seen(observed, 0, False)

    def _walk_seen(self, node: Any, depth: int, in_rows: bool) -> None:
        """Depth-first pass over a reply, at most six levels down. A dict that is a listing gets merged and is not
        looked into; other dicts and lists are looked into, and the items of a list that reads as listing rows are
        taken as rows."""
        if depth > 6:
            return
        if isinstance(node, list):
            rowish = self._list_is_rows(node)
            for item in node:
                self._walk_seen(item, depth + 1, rowish)
            return
        if not isinstance(node, dict):
            return
        ident = self.id_of_record(node, in_rows, depth)
        if ident:
            self._absorb_row(ident, node)
            return
        for item in node.values():
            self._walk_seen(item, depth + 1, False)

    def _absorb_row(self, ident: str, row: dict[str, Any]) -> None:
        if ident not in self.row_records:
            self.seen_sequence.append(ident)
        merged = self.row_records.setdefault(ident, {})
        turn = _stock_flip(merged, row)
        if turn is False:
            self.unbuyable_ids.add(ident)
        elif turn:
            self.unbuyable_ids.discard(ident)
        merged.update(row)
        if _trait_block(row):
            self.page_records[ident] = dict(merged)
        elif ident in self.page_records:
            self.page_records[ident].update(row)

    def _list_is_rows(self, items: list[Any]) -> bool:
        """A list reads as listing rows when at least half its entries are dicts of three or more fields that share
        an id field; the first such list settles the id field for the whole run."""
        records = [entry for entry in items if isinstance(entry, dict) and len(entry) >= 3]
        if not records or 2 * len(records) < len(items):
            return False
        known = self.row_id_key
        if known and all(_id_shaped(entry.get(known), True) for entry in records):
            return True
        guessed = _infer_id_key(records)
        if guessed:
            self.row_id_key = guessed
        return bool(guessed)

    def id_of_record(self, record: Any, in_rows: bool = False, depth: int = 0) -> str | None:
        """The listing id of a dict, or None. Below the top level and outside a row list, a dict with no dict inside
        is never a listing; a guessed id key counts when its name says id or its value is an id already known."""
        if not isinstance(record, dict) or len(record) < 3:
            return None
        known = self.row_id_key
        if known and _id_shaped(record.get(known), True):
            return record[known].strip()
        if not (in_rows or depth == 0 or any(isinstance(inner, dict) for inner in record.values())):
            return None
        guessed = _infer_id_key([record])
        if not guessed:
            return None
        value = record[guessed].strip()
        return value if 'id' in str(guessed).lower() or value in self.row_records else None

    def _identified_rows(self, rows: list[Any]) -> list[tuple[str, dict[str, Any]]]:
        return [(ident, row) for row in rows if isinstance(row, dict)
                for ident in [self.id_of_record(row, True)] if ident]

    def records_in_reply(self, observed: Any) -> list[tuple[str, dict[str, Any]]]:
        """(id, record) pairs: the reply itself when it is a single listing, otherwise the rows of the lists it holds
        at the top or one dict down."""
        if isinstance(observed, list):
            return self._identified_rows(observed)
        if not isinstance(observed, dict):
            return []
        single = self.id_of_record(observed)
        if single:
            return [(single, observed)]
        row_lists: list[list[Any]] = []
        for value in observed.values():
            if isinstance(value, list):
                row_lists.append(value)
            elif isinstance(value, dict):
                row_lists.extend(inner for inner in value.values() if isinstance(inner, list))
        return [pair for chunk in row_lists for pair in self._identified_rows(chunk)]

    def note_view_keys(self, tool: str, observed: Any) -> None:
        """Remember the fields this tool showed per listing, tied to the current state revision, so a listing that has
        not changed is not reopened."""
        version = self.revision
        for ident, record in self.records_in_reply(observed):
            shown_keys = set(record)
            self.tool_key_sets.setdefault(tool, set()).update(shown_keys)
            stamp, held = self.seen_view_keys.get(ident, (-1, set()))
            self.seen_view_keys[ident] = (version, held | shown_keys if stamp == version else set(shown_keys))

    def view_same_as_before(self, tool: str, ident: str) -> bool:
        expected = self.tool_key_sets.get(tool)
        if not expected:
            return False
        stamp, held = self.seen_view_keys.get(ident, (-1, set()))
        return stamp == self.revision and expected <= held and ident not in self.sold_out_steps

    # ================================================================== result layouts

    def lone_id_arg(self, tool: str | None) -> str | None:
        """The one argument through which a tool takes a single listing: a free-text string (no enum) that is the only
        required argument or, with nothing required, the only free-text argument. None in every other case."""
        schema = self.schemas.get(tool or '') or {}
        required = schema.get('required') or []
        texts = [name for name, sub in (schema.get('properties') or {}).items()
                 if isinstance(sub, dict) and 'string' in _declared_types(sub) and 'enum' not in sub]
        slot = next((name for name in required if name in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not (tool and slot):
            return None
        return slot if all(name == slot for name in required) else None

    def _layout_of(self, tool: str, observed: Any) -> dict[str, Any] | None:
        """A tool's result layout (rows key, match total, next-page key and parameter, price and stock keys). The model
        is asked once per tool, on the first reply that holds a list of rows."""
        shapes = self.reply_shapes
        if tool in shapes:
            return shapes[tool]
        candidates = _lists_of_rows(observed)
        if not candidates:
            return None
        shapes[tool] = None
        tool_schema = self.schemas.get(tool) or {}
        question = _layout_prompt(tool, tool_schema, observed, candidates)
        verdict = self.ask_model_json(question, [QUICK_SEAT, DEEP_SEAT], 2500) or {}
        rows_key = verdict.get('rows')
        if not rows_key or rows_key not in candidates:
            return None
        shape = self._checked_layout(verdict, rows_key, observed, tool_schema.get('properties') or {})
        shapes[tool] = shape
        if not self.price_key:
            self.price_key = shape['price']
        if not self.stock_key:
            self.stock_key = shape['stock']
        self.journal.append(dict(kind='shape', tool=tool, shape=shape))
        return shape

    def _checked_layout(self, verdict: dict[str, Any], rows_key: str, observed: dict[str, Any], props: Any) -> dict[str, Any]:
        """Each key the model named survives only where it truly sits: total and next-page value in the reply, the
        next-page parameter among the tool's parameters, price and stock in a row; a total must be a whole number."""
        sample_row = observed[rows_key][0]
        homes = {'count': observed, 'next': observed, 'next_arg': props, 'price': sample_row, 'stock': sample_row}
        named = {slot: verdict.get(slot) for slot in homes}
        shape = {'rows': rows_key, **{slot: (key if isinstance(key, str) and key in homes[slot] else None)
                                      for slot, key in named.items()}}
        total_key = shape['count']
        if total_key and not _whole_number(observed[total_key]):
            shape['count'] = None
        if shape['next'] and not shape['next_arg']:
            shape['next_arg'] = self._next_page_param(props, observed.get(shape['next']))
            if shape['next_arg']:
                self.journal.append({'kind': 'next_arg_from_schema', 'arg': shape['next_arg']})
        return shape

    def _next_page_param(self, props: Any, token: Any) -> str | None:
        """The parameter that carries a next-page value back, read from the tool's declared schema when the layout
        reader named none: an optional, non-enumerated parameter of the token's type, other than a free-text words
        field; with several, the one whose name reads like paging. Only a single survivor counts, so an ambiguous
        schema leaves paging as it was."""
        if not isinstance(props, dict) or token is None or isinstance(token, bool):
            return None
        wanted = 'integer' if isinstance(token, int) else 'string' if isinstance(token, str) else None
        if wanted is None:
            return None
        words_slot = self._words_slot_of(props)
        fits = [name for name, spec in props.items()
                if isinstance(spec, dict) and 'enum' not in spec and name != words_slot
                and (wanted in _declared_types(spec) or (wanted == 'integer' and 'number' in _declared_types(spec)))]
        fits = [name for name in fits if not re.search(CASH_ARG_RX, name + ' ' + str(props[name].get('description') or ''), re.I)]
        if len(fits) > 1:
            fits = [name for name in fits if re.search(r'page|offset|cursor|token|next|start|skip|after|from|continu',
                                                      name, re.I)]
        return fits[0] if len(fits) == 1 else None

    def _words_slot_of(self, props: dict[str, Any]) -> str | None:
        """The search's free-text field: the one string parameter without fixed options, when there is exactly one."""
        texts = [name for name, spec in props.items()
                 if isinstance(spec, dict) and 'string' in _declared_types(spec) and 'enum' not in spec]
        return texts[0] if len(texts) == 1 else None

    def _paging_state(self, tool: str, observed: Any) -> tuple[int, int, str, Any] | None:
        """(matches in all, rows on this page, next-page parameter, its value or None) for a paged reply the layout
        explains; None otherwise."""
        shape = self._layout_of(tool, observed)
        if not shape or not all(shape[slot] for slot in ('count', 'next', 'next_arg')) or not isinstance(observed, dict):
            return None
        rows, total, token = (observed.get(shape[slot]) for slot in ('rows', 'count', 'next'))
        if not isinstance(rows, list) or not rows or not _whole_number(total):
            return None
        if token is not None and (isinstance(token, bool) or not isinstance(token, (str, int))):
            return None
        return total, len(rows), shape['next_arg'], token

    # ================================================================== pages Wren fetches by itself

    def fetch_more_pages(self, tool: str, call_args: dict[str, Any], observed: Any) -> str:
        """For a cheapest, most or least pick, fetch up to two more pages (a step each) of a first page that shows only
        part of the matches, and say how to narrow the search or page on when even that falls short. For a cheapest
        pick over a far larger match count, with a price-capped category listing at hand, ask for a cart line first."""
        self.extra_pages = []
        pick = self.digest.get('pick')
        if pick not in ('cheapest', 'most', 'least'):
            return ''
        state = self._paging_state(tool, observed)
        if not state or not self._worth_paging(pick, state, call_args, observed):
            return ''
        if pick == 'cheapest' and self._capped_listing_beats_paging(state):
            total, shown = state[:2]
            return (f'\n[Wren] Of {total} matching listings, {shown} were returned, not ordered by price. Put the '
                    'first listing that satisfies all requirements in the cart; Wren will then list its category under '
                    'that price to show anything cheaper.')
        return self._page_onward(tool, call_args, state)

    def _capped_listing_beats_paging(self, state: tuple[int, int, str, Any]) -> bool:
        # paging through unsorted rows costs a step per page, while one category listing capped below a carted line's
        # price brings up every cheaper row in a single step
        total, shown = state[:2]
        return total > 3 * max(1, shown) and bool(self.shelf_field_name()) and self._price_capped_browser() is not None

    def _worth_paging(self, pick: Any, state: tuple[int, int, str, Any], call_args: dict[str, Any], observed: Any) -> bool:
        """A next page is offered, this call asked for page one, rows are missing and, for the cheapest, the rows shown
        are not already sorted by price."""
        total, shown, param, token = state
        if token is None or total <= shown or self._past_first_page(call_args.get(param)):
            return False
        return not (pick == 'cheapest' and self._rows_cheapest_first(observed))

    def _page_onward(self, tool: str, call_args: dict[str, Any], state: tuple[int, int, str, Any]) -> str:
        total, shown, param, token = state
        count = self._auto_page_budget(total, shown)
        if not count:
            return (f"\n[Wren] This search matched {total} listings yet returned just {shown}, unsorted by price. The "
                    "pick depends on seeing every fitting listing: narrow the search words (with the requirement's "
                    f'value, say) or request the next page with {param}={_jdump(token)}.')
        pages, last_token = self._pull_pages(tool, call_args, param, token, count)
        self.journal.append(dict(kind='pages', tool=tool, read=len(pages)))
        if not pages:
            return ''
        tail = '' if last_token is None else (
            f'\n[Wren] Wren has now seen {shown * (len(pages) + 1)} rows out of {total} matching listings; for the '
            "rest, put the requirement's value into the search words or go on paging.")
        return '\n[Wren paged further through this search by itself, spending one step per page]\n' + '\n'.join(pages) + tail

    def _auto_page_budget(self, total: int, shown: int) -> int:
        """How many missing pages to fetch (two at most); 0 when that would pass the auto-paging limit or leave fewer
        than 12 steps besides the closing ones."""
        wanted = min(2, (total - 1) // shown)
        spare = self.moves_left() - self.wrap_up_reserve()
        if self.auto_pages_done + wanted > AUTO_PAGE_LIMIT or spare - wanted < 12:
            return 0
        return wanted

    def _pull_pages(self, tool: str, call_args: dict[str, Any], param: str, token: Any, count: int) -> tuple[list[str], Any]:
        """One store call per page, stopping early on a failed call, a finished run or a missing next page; gives the
        page texts and the last next-page value seen."""
        pages: list[str] = []
        while len(pages) < count and token is not None:
            sent, answer = self._send_fitted(tool, {**call_args, param: token})
            if answer is None or self.run_closed:
                return pages, token
            self.auto_pages_done += 1
            fetched = answer.get('observation')
            self.digest_reply(tool, fetched)
            self.extra_pages.append(fetched)
            shown_text = self.show_reply(fetched)
            self.read_cache[tool + _jdump(sent)] = (self.revision, self.moves_made, shown_text)
            pages.append(f'{param}={_jdump(token)}: {shown_text}')
            onward = self._paging_state(tool, fetched)
            token = onward[3] if onward else None
        return pages, token

    # ================================================================== notes attached to a search result

    def ignored_cap_note(self, tool: str, call_args: dict[str, Any], observed: Any) -> str:
        """Point out a search that returned rows above the maximum price it was given, and the mode to list in instead."""
        props = (self.schemas.get(tool) or {}).get('properties') or {}
        caps = [name for name, value in call_args.items() if self._money_cap_arg(name, value, props)]
        if not caps or not self.price_key:
            return ''
        cap_name = sorted(caps, key=call_args.get)[0]
        cap = call_args[cap_name]
        over = [ident for ident, _ in self.records_in_reply(observed) if (self.price_of_listing(ident) or 0) > cap]
        if not over:
            return ''
        hint = _cheaper_mode_hint(call_args, props, cap_name)
        return f'\n[Wren] the search ignored {cap_name}={cap:g}: {len(over)} of its returned rows are priced higher{hint}.'

    def _money_cap_arg(self, name: str, value: Any, props: dict[str, Any]) -> bool:
        if name not in props or not _real_number(value):
            return False
        return bool(re.search('max|most|under|below|limit|cap|upto|up_to', name, re.I)) and _param_mentions_money(props, name)

    def lowest_price_rows_note(self, results: list[Any]) -> str:
        if not self.price_key or self.digest.get('pick') != 'cheapest':
            return ''
        idents = {ident for result in results for ident, _ in self.records_in_reply(result)}
        ranked = sorted((cost, ident) for ident in idents if self.is_purchasable(ident)
                        for cost in [self.price_of_listing(ident)] if cost is not None)
        if len(ranked) < 2:
            return ''
        return '\n[Wren] buyable rows shown here, from the lowest price up: ' + ', '.join(f'{ident} ({cost:g})' for cost, ident in ranked[:6])

    def wider_hunt_pause(self) -> str:
        """Hold the first checkout of a cheapest pick once, while steps are plentiful, and ask for broader searches."""
        remaining = self.moves_left()
        if self.wider_hunt_done or self.digest.get('pick') != 'cheapest' or remaining < 14 + 2 * len(self.goods):
            return ''
        self.wider_hunt_done = True
        priced = [f'{ident} at {self.price_of_listing(ident):g}' for ident in self.basket_ids()
                  if self.price_of_listing(ident) is not None]
        carted = ' (' + '; '.join(priced) + ')' if priced else ''
        return (f'Not run yet (no step used). {remaining} steps are left and the request asks for the lowest price. A'
                " search may match on only part of each listing, so looking up a requirement's value finds only some of"
                ' the fitting listings, and the rows come back unsorted by price. Before checking out, run a few more'
                ' searches for each requested item under other names for the product (its plain one-word name,'
                ' synonyms, words lifted from the titles of fitting listings seen so far) and weigh only buyable rows'
                " priced below the cart's line" + carted + '; open the attributes of the cheapest such rows, swap a'
                ' cart line for any one that fits, and only then check out.')

    # ================================================================== cheaper rows for a cheapest pick

    def read_lower_priced_rows(self) -> list[str]:
        """Once per cart state of a cheapest pick: list each line's category under its price, then open up to two
        cheaper buyable rows with the line's label per line (with an overall cap)."""
        if self.digest.get('pick') != 'cheapest':
            return []
        mark = self.basket_fingerprint() + f'#{self.revision}'
        label_field = self.shelf_field_name()
        if not label_field or mark in self.lower_price_marks:
            return []
        self.lower_price_marks.add(mark)
        return self._open_cheaper_rows(label_field)

    def _open_cheaper_rows(self, label_field: str) -> list[str]:
        carted = self.basket_ids()
        self._browse_under_line_prices(carted)
        picks: list[str] = []
        for line_id in carted:
            head = self._cheaper_peers(line_id, carted, label_field)[:2]
            picks += [ident for ident in head if ident not in picks]
        limit = min(6, LOWER_PRICE_OPENS_PER_LINE * max(1, len(carted)))
        return self.read_traits_of(picks[:limit], self.wrap_up_reserve() + 4)

    def _shelf_text(self, ident: str, field: str) -> str:
        row = self.row_records.get(ident) or {}
        return str(row.get(field) or '').strip().lower()

    def _cheaper_peers(self, line_id: str, carted: list[str], label_field: str) -> list[str]:
        """Listings outside the cart, not opened yet, buyable, under the line's label and priced below it, cheapest
        first; [] when the line has no price or no label."""
        ceiling = self.price_of_listing(line_id)
        kind = self._shelf_text(line_id, label_field)
        if ceiling is None or not kind:
            return []

        def below(ident: str) -> bool:
            cost = self.price_of_listing(ident)
            return cost is not None and cost < ceiling
        peers = [ident for ident in self.seen_sequence
                 if ident not in carted and ident not in self.page_records and self.is_purchasable(ident)
                 and self._shelf_text(ident, label_field) == kind and below(ident)]
        return sorted(peers, key=lambda ident: self.price_of_listing(ident) or 0.0)

    # ================================================================== opening listings and following part codes

    def read_traits_of(self, idents: list[str], keep: int) -> list[str]:
        """Open with the attribute tool, one step apiece, each listing of `idents` that has a row, no full page yet and
        is not sold out, stopping when only `keep` steps remain, the run ends or the tool proves to show no
        attributes. One text per listing opened."""
        viewer = self.trait_viewer()
        slot = self.lone_id_arg(viewer)
        if not (viewer and slot):
            return []
        queue = [ident for ident in idents
                 if ident in self.row_records and ident not in self.page_records and not self.sold_out_now(ident)]
        texts = self._open_queue(viewer, slot, queue, keep)
        if texts:
            self.journal.append(dict(kind='fill_facts', n=len(texts)))
        return texts

    def _open_queue(self, viewer: str, slot: str, queue: list[str], keep: int) -> list[str]:
        texts: list[str] = []
        for ident in queue:
            if self.moves_left() <= keep or self.run_closed or self.tool_shows_traits.get(viewer) is False:
                return texts
            opened = self._open_one(viewer, slot, ident)
            if opened is None:
                return texts
            texts.append(f'{ident}: ' + self.show_reply(opened.get('observation'))[:1500])
        return texts

    def _open_one(self, viewer: str, slot: str, ident: str) -> dict[str, Any] | None:
        _, opened = self._send_fitted(viewer, {slot: ident})
        if opened is not None:
            view = opened.get('observation')
            self.digest_reply(viewer, view)
            self.note_view_keys(viewer, view)
            if isinstance(view, dict) and self.id_of_record(view):
                self.tool_shows_traits[viewer] = self.tool_shows_traits.get(viewer, False) or bool(_trait_block(view))
        return opened

    def chase_part_refs(self, line_ids: list[str], keep: int) -> None:
        """With two or more cart lines (or requested items): whenever a line's page sends the reader to another part
        code for the rest of its details, search that code once and open the listings owning it while steps allow.
        It only feeds the ledger; the model is shown nothing."""
        plan = self.values_query_plan()
        if not plan or (len(self.goods) < 2 and len(line_ids) < 2):
            return
        tool, fixed, field = plan
        for line_id in line_ids:
            ref = self._fresh_ref_code(line_id)
            if ref and (self.moves_left() <= keep or self.run_closed
                        or not self._search_ref_code(tool, fixed, field, line_id, ref, keep)):
                return

    def _fresh_ref_code(self, line_id: str) -> Any:
        """The 'see part X' code on a line's page, unless it was searched before or a known listing already owns it."""
        fit_marks = self.fit_signal_index()['marks']
        ref = (fit_marks.get(line_id) or {}).get('more')
        if not ref or ref in self.ref_codes_tried or any(ref in (mark.get('own'), mark.get('own_series_part')) for mark in fit_marks.values()):
            return None
        return ref

    def _search_ref_code(self, tool: str, fixed: dict[str, Any], field: str, line_id: str, ref: Any, keep: int) -> bool:
        self.ref_codes_tried.add(ref)
        _, found = self._send_fitted(tool, {**fixed, field: ref})
        if found is None:
            return False
        self.digest_reply(tool, found.get('observation'))
        self.journal.append(dict(kind='show_link', line=line_id, code=ref))
        owners = [ident for ident, mark in self.fit_signal_index()['marks'].items() if ref in (mark.get('own'), mark.get('own_series_part'))]
        stop_at = max(keep, self.wrap_up_reserve() + 6)
        self.read_traits_of(owners, stop_at)
        return True

    # ================================================================== category labels

    def _schema_options(self) -> set[str]:
        """All string options (lower-cased) of every enum in the tool schemas, searched up to six levels down."""
        options: set[str] = set()
        stack = [(schema, 0) for schema in self.schemas.values()]
        while stack:
            node, depth = stack.pop()
            if depth <= 6 and isinstance(node, dict):
                options.update(choice.strip().lower() for choice in node.get('enum') or [] if isinstance(choice, str))
                stack.extend((inner, depth + 1) for inner in node.values() if isinstance(inner, dict))
        return options

    def _label_like_columns(self) -> tuple[dict[str, set[str]], dict[str, int]]:
        """Per row key: the distinct label-like texts under it, and how many rows hold one."""
        texts: dict[str, set[str]] = {}
        counts: dict[str, int] = {}
        for row in self.row_records.values():
            for key, cell in row.items():
                if _label_shaped(cell):
                    texts.setdefault(key, set()).add(cell)
                    counts[key] = counts.get(key, 0) + 1
        return texts, counts

    def shelf_field_name(self) -> str | None:
        """The row key that holds the category label: label-like text on at least 80% of rows with 40 distinct values
        at most. Among several, prefer the larger share of values the schemas offer as options, then fewer distinct
        values, then presence on more rows, then longer values."""
        texts, counts = self._label_like_columns()
        floor = 0.8 * max(1, len(self.row_records))
        options = self._schema_options()

        def preference(key: str) -> tuple[float, int, int, float]:
            values = texts[key]
            offered = sum(1 for value in values if value.strip().lower() in options) / len(values)
            avg_len = sum(len(value) for value in values) / len(values)
            return -offered, len(values), -counts[key], -avg_len
        eligible = [key for key, values in texts.items() if counts[key] >= floor and len(values) <= 40]
        return min(eligible, key=preference) if eligible else None

    def _requested_kinds(self) -> list[tuple[str, str, set[str]]]:
        """Per requested item: its plain shelf words, its plain name words and the word stems (4+ letters) of both."""
        kinds = []
        for good in self.goods:
            shelf = good.shelf_name or ''
            roots = {stem for stem in _stem_set(shelf) | _stem_set(good.good_title) if len(stem) >= 4}
            kinds.append((_bare_words(shelf), _bare_words(good.good_title), roots))
        return kinds

    def _kind_wanted(self, label: str, kinds: list[tuple[str, str, set[str]]]) -> bool:
        flat = _bare_words(label)
        roots = {stem for stem in _stem_set(label) if len(stem) >= 4}
        if any(flat in (shelf, title) or roots & item_roots for shelf, title, item_roots in kinds):
            return True
        return bool(self.good_for_shelf(label))

    def shelf_groups(self) -> list[list[str]]:
        """Listing ids by category label, keeping only the requested kinds when any label matches one, largest group
        first, as many groups as requested items."""
        label_field = self.shelf_field_name()
        groups: dict[str, list[str]] = {}
        for ident in self.seen_sequence:
            label = str(self.row_records[ident].get(label_field) or '') if label_field else ''
            groups.setdefault(label, []).append(ident)
        kinds = self._requested_kinds()
        chosen = [members for label, members in groups.items() if self._kind_wanted(label, kinds)] or list(groups.values())
        return sorted(chosen, key=lambda members: -len(members))[:max(1, len(self.goods))]

    # ================================================================== attribute keys of opened listings

    def trait_key_table(self) -> list[tuple[str, str, int, list[str]]]:
        """Rows (label, plain key name, how many opened listings carry it, up to 4 sample values cut to 40 chars): up to
        14 attribute keys per category label, the requested kinds only when any label matches, largest label first."""
        groups = self._opened_by_kind(self.shelf_field_name())
        kinds = self._requested_kinds()
        wanted = {label for label in groups if self._kind_wanted(label, kinds)}
        if wanted:
            groups = {label: group for label, group in groups.items() if label in wanted}
        table: list[tuple[str, str, int, list[str]]] = []
        for label, group in sorted(groups.items(), key=lambda pair: -pair[1]['count']):
            table.extend((label, name, column['hits'], [flat[:40] for flat in column['texts'][:4]])
                         for name, column in self._useful_columns(group['columns'])[:14])
        return table

    def _opened_by_kind(self, label_field: str | None) -> dict[str, dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
        for ident in self.seen_sequence:
            page = self.page_records.get(ident)
            if page:
                label = str(page.get(label_field) or 'listings') if label_field else 'listings'
                group = groups.setdefault(label, {'count': 0, 'columns': {}})
                group['count'] += 1
                _tally_columns(group['columns'], ident, _trait_block(page) or {})
        return groups

    def _useful_columns(self, columns: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
        """Noisy keys dropped, a key naming the same attribute as an earlier one folded into it, most carried first."""
        kept: list[tuple[str, dict[str, Any]]] = []
        for name, column in sorted(columns.items(), key=lambda pair: (-pair[1]['hits'], len(pair[0]))):
            values = list(column['by_id'].values())
            if not values or _column_too_noisy(values):
                continue
            twin = next((spec for other, spec in kept if _same_attribute_columns(other, spec, name, column)), None)
            if twin is not None:
                _fold_column(twin, column)
                continue
            kept.append((name, column))
        kept.sort(key=lambda pair: -pair[1]['hits'])
        return kept

    def different_trait_note(self, label: str, key_name: str, examples: list[str]) -> str:
        """Tag a key that shares a word with a required property yet is a different one: another name, and values that
        do not show the required value's words."""
        slot = self.good_for_shelf(label)
        if not slot:
            return ''
        visible = set(re.findall('[a-z0-9]{3,}', ' '.join(examples).lower()))
        for need in self.goods[slot - 1].given_needs:
            prop, _, target = need.partition(':')
            if _other_trait_named(prop, target, key_name, visible):
                return f' [not the stated {prop.strip()}: some other property]'
        return ''
class RunSearching:
    """Everything that surrounds the conversation with the shopper. Before the interview a few listings are opened
    so the interviewer has something to look at; the interview then asks while messages, steps and time last, and
    one held-back question may go out right before checkout. Values an answer disclosed are searched afterwards,
    with a search modelled on the earlier ones. Also here: picking the next property to ask and phrasing it so the
    message tool takes it whole."""

    _TOKEN3_RX = '[a-z0-9]{3,}'
    _QUALIFIER_RX = '\\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b'

    # ================================================================== search plans copied from earlier searches

    def words_query_plan(self) -> tuple[str, dict[str, Any], str] | None:
        """Model a words search on the most recent earlier search that typed words into a text field. The field is a
        query-named one; if the schema has none, it must be a field used like a phrase box (differing values, some of
        two or more words, none equal to a category label). Gives (tool, arguments to carry over, words field), or
        None when no earlier search qualifies."""
        tool, spec, props = self._tool_spec_for('search')
        texty = self._text_slots(props)
        named = self._query_named(texty)
        allowed = None
        if not named and self.recent_queries:
            allowed = self._phrase_fields(texty)
        if not tool:
            return None
        found = self._latest_worded_search(named or texty, allowed)
        if found is None:
            return None
        earlier, box = found
        return tool, self._kept_search_args(earlier, box, spec, props), box

    def values_query_plan(self) -> tuple[str, dict[str, Any], str] | None:
        """The search a value search should copy. words_query_plan() wins when it has an answer. Failing that, if all
        earlier searches only browsed a category, the search tool's one query field is used, with its one mode
        argument flipped to reading that field: the option naming words or a query, else the other value of a
        required two-valued mode. Any ambiguity in the schema gives None."""
        plan = self.words_query_plan()
        if plan is not None:
            return plan
        tool, spec, props = self._tool_spec_for('search')
        if not tool or not self.recent_queries:
            return None
        return self._browse_turned_query(tool, spec, props)

    def _browse_turned_query(self, tool: str, spec: dict[str, Any], props: dict[str, Any]) -> tuple[str, dict[str, Any], str] | None:
        required = spec.get('required') or []
        boxes = self._query_named(self._text_slots(props))
        last = self.recent_queries[-1]
        enums = {key: sub['enum'] for key, sub in props.items() if isinstance(sub, dict) and isinstance(sub.get('enum'), list) and len(sub['enum']) >= 2}
        switches = [hit for hit in (self._switch_to_words(key, opts, last, required) for key, opts in enums.items()) if hit]
        if len(boxes) != 1 or len(switches) > 1 or (enums and not switches):
            return None
        box = boxes[0]
        carried = {key: last[key] for key in required if key in last and key != box}
        carried.update(switches)
        return tool, carried, box

    @staticmethod
    def _switch_to_words(key: str, options: list[Any], last: dict[str, Any], required: list[str]) -> tuple[str, Any] | None:
        """The value a mode argument takes for a words search: its single option that names words or a query, or,
        for a required argument with exactly two options, the one the last search did not use."""
        wordy = [opt for opt in options if isinstance(opt, str) and re.search(WORDS_ARG_RX, opt, re.I)]
        if len(wordy) == 1:
            return key, wordy[0]
        if key not in required or len(options) != 2:
            return None
        others = [opt for opt in options if opt != last.get(key)]
        return (key, others[0]) if len(others) == 1 else None

    @staticmethod
    def _query_named(keys: list[str]) -> list[str]:
        return [key for key in keys if re.search(WORDS_ARG_RX, _to_snake(key), re.I)]

    @staticmethod
    def _typed_letters(query: dict[str, Any], key: str) -> bool:
        value = query.get(key)
        return isinstance(value, str) and re.search('[a-z]', value, re.I) is not None

    def _latest_worded_search(self, candidates: list[str], allowed: set[str] | None) -> tuple[dict[str, Any], str] | None:
        """Walk the earlier searches from the newest back and stop at the first that put letters into one of the
        candidate fields (restricted to `allowed` when given); returns (that search, that field)."""
        for earlier in reversed(self.recent_queries):
            for key in candidates:
                if self._typed_letters(earlier, key) and (allowed is None or key in allowed):
                    return earlier, key
        return None

    def _phrase_fields(self, texty: list[str]) -> set[str]:
        """For a schema without a query-named field: the text fields the earlier searches filled the way one fills a
        phrase box - at least two distinct values, at least one value of several words, and no value that equals a
        category label."""
        label_key = self.shelf_field_name()
        shelf_texts: set[str] = set()
        if label_key:
            shelf_texts = {str(row.get(label_key) or '').strip().lower() for row in self.row_records.values()}
            shelf_texts.discard('')
        return {key for key in texty if self._used_like_phrase([query.get(key) for query in self.recent_queries], shelf_texts)}

    @staticmethod
    def _used_like_phrase(values: list[Any], shelf_texts: set[str]) -> bool:
        if len({str(value) for value in values if value}) < 2:
            return False
        if not any(len(str(value).split()) >= 2 for value in values):
            return False
        return all(str(value or '').strip().lower() not in shelf_texts for value in values)

    @staticmethod
    def _has_enum(props: dict[str, Any], key: str) -> bool:
        return isinstance((props.get(key) or {}).get('enum'), list)

    def _kept_search_args(self, earlier: dict[str, Any], box: str, spec: dict[str, Any], props: dict[str, Any]) -> dict[str, Any]:
        """What a copied search carries over from `earlier`: its required and enumerated arguments except the words
        field. Optional text that merely names a shelf is left behind, and every enumerated argument is set to the
        value the worded searches used most often."""
        required = spec.get('required') or []
        shelf_texts = self._known_shelf_texts()
        carried: dict[str, Any] = {}
        for key, value in earlier.items():
            if key == box or not (self._has_enum(props, key) or key in required):
                continue
            if key not in required and isinstance(value, str) and self._only_names_a_shelf(value, shelf_texts):
                continue
            carried[key] = value
        worded = [query for query in self.recent_queries if self._typed_letters(query, box)]
        for key in [k for k in carried if self._has_enum(props, k)]:
            seen = [(_jdump(query[key]), query[key]) for query in worded if key in query]
            if not seen:
                continue
            dumps = [dumped for dumped, _ in seen]
            top = max(dumps, key=dumps.count)
            carried[key] = next(raw for dumped, raw in seen if dumped == top)
        return carried

    def _known_shelf_texts(self) -> set[str]:
        names: set[str] = set()
        for good in self.goods:
            names.add(_bare_words(good.shelf_name or ''))
            names.add(_bare_words(good.good_title))
        label_key = self.shelf_field_name()
        if label_key:
            names |= {_bare_words(str(row.get(label_key) or '')) for row in self.row_records.values()}
        return names

    @staticmethod
    def _only_names_a_shelf(text: str, shelf_texts: set[str]) -> bool:
        """True when the text is a shelf name, or shares a root word of four or more letters with one."""
        if _bare_words(text) in shelf_texts:
            return True
        long_roots = {root for root in _stem_set(text) if len(root) >= 4}
        for shelf in shelf_texts:
            if shelf and long_roots.intersection(root for root in _stem_set(shelf) if len(root) >= 4):
                return True
        return False

    # ================================================================== searching what the answers disclosed

    def _probe_disclosed_values(self, slots: list[int]) -> str:
        """Each listed item whose answers gave a value gets its requirement values searched together through the
        copied search (values_query_plan); fitting rows are kept and the first few unopened ones read. Spends no more
        than 8 steps and starts only with 22 left. Gives a note for the loop model, or ''."""
        plan = self.values_query_plan()
        if not plan or self.moves_left() < 22:
            return ''
        notes: list[str] = []
        self._value_search_round(plan, slots, notes)
        if not notes:
            return ''
        self.journal.append({'kind': 'value_search', 'seen': [note[:200] for note in notes]})
        return '[Wren searched each such item with all its requirement values in one query and read its first rows of the matching kind] ' + '\n'.join(notes)

    def _value_search_round(self, plan: tuple[str, dict[str, Any], str], slots: list[int], notes: list[str]) -> None:
        tool, carried, box = plan
        budget = 8
        for slot in slots:
            found = self._searchable_values(slot)
            good = self.goods[slot - 1]
            told = [value for value, by_answer in found if by_answer]
            if not told or budget <= 2:
                continue
            tries = self._query_word_sets([value for value, _ in found], told, good.good_title)
            budget = self._run_value_tries(tool, carried, box, tries, good, slot, budget, notes)

    def _searchable_values(self, slot: int) -> list[tuple[str, bool]]:
        """(value, came from an answer) for each requirement of the item a words search could hit. A plain value is
        taken as written. An answered lower bound on a measured quantity contributes its number and unit, because such
        a hidden bound often equals the value of a fitting listing. Values that refer to other listings, or that carry
        a limit, a negation or an alternative, are skipped."""
        told = self.goods[slot - 1].told_needs
        linked = [text for _, text in self.borrowed_needs]
        picked: list[tuple[str, bool]] = []
        for owner, need in self.needs_in_force():
            if owner != slot or need in linked or ':' not in need:
                continue
            value = need.partition(':')[2].strip()
            minimum = re.search(MINIMUM_PHRASE_RX, value, re.I)
            if minimum:
                if need in told:
                    picked.append((minimum.group(1).strip(), True))
                continue
            if value and not re.search(self._QUALIFIER_RX, value, re.I):
                picked.append((value, need in told))
        return picked

    @staticmethod
    def _query_word_sets(values: list[str], told: list[str], title: str) -> list[list[str]]:
        """At most two word lists to try, answered single-word or measured values in front. Several values: all of
        them first, then all without the last value the request stated itself (without the first if it stated none).
        One value in front: that value plus the product name first, then the value on its own."""
        front = [value for value in told if len(value.split()) == 1 or re.fullmatch(QUANTITY_RX, value, re.I)]
        words = front + [value for value in values if value not in front]
        if len(words) <= 1:
            return [[words[0], title], words] if front else [words]
        from_request = [pos for pos, value in enumerate(words) if value not in told]
        drop = from_request[-1] if from_request else 0
        return [words, words[:drop] + words[drop + 1:]]

    def _run_value_tries(self, tool: str, carried: dict[str, Any], box: str, tries: list[list[str]], good: Any, slot: int, budget: int, notes: list[str]) -> int:
        """Send the item's word lists one after another; the first that returns fitting rows gets its first unopened
        ones read and ends the item, as does any call that cannot be made or fails. Returns the remaining budget."""
        for words in tries[:2]:
            if not words:
                continue
            args = self._conform_or_none({**carried, box: ' '.join(words)}, self.schemas.get(tool))
            if args is None or self.run_closed:
                break
            reply = self.shop_call(tool, args)
            budget -= 1
            if reply is None:
                break
            rows = self._rows_worth_reading(tool, reply.get('observation'), good, slot)
            notes.append(f'{_jdump(args.get(box))}: ' + (', '.join(rows[:6]) or 'no rows of that kind'))
            if rows:
                return budget - self._read_value_rows(rows, budget, notes)
        return budget

    def _read_value_rows(self, rows: list[str], budget: int, notes: list[str]) -> int:
        unopened = [lid for lid in rows if lid not in self.page_records][:min(3, budget)]
        texts = self.read_traits_of(unopened, self.wrap_up_reserve() + 16)
        if texts:
            notes.append('opened: ' + ' | '.join(texts)[:2400])
        return len(texts)

    def _rows_worth_reading(self, tool: str, view: Any, good: Any, slot: int) -> list[str]:
        """Take in a value search's reply; return its rows that can be bought, are of the item's own kind and stay
        within the price cap, those on the item's home shelf first."""
        self.digest_reply(tool, view)
        label_key = self.shelf_field_name()
        row_ids = [lid for lid, _ in self.records_in_reply(view)]
        cap = self._budget_cap_for(row_ids)

        def label_of(lid: str) -> str:
            return str((self.row_records.get(lid) or {}).get(label_key) or '')

        def keeps(lid: str) -> bool:
            if not self.is_purchasable(lid):
                return False
            if label_key and not self.good_has_shelf(good, label_of(lid)):
                return False
            return cap is None or (self.price_of_listing(lid) or 0) <= cap

        rows = [lid for lid in row_ids if keeps(lid)]
        home = self._home_shelf_label(slot) if label_key else ''
        if home:
            rows.sort(key=lambda lid: _bare_words(label_of(lid)) != home)
        return rows

    def _budget_cap_for(self, row_ids: list[str]) -> float | None:
        """The budget used as a price cap on value-search rows. None when the budget is not a number, or when each
        row is priced above ten times it (a different price unit, where a cap would hide every row)."""
        cap = self.money_cap
        if not isinstance(cap, (int, float)):
            return None
        prices = [self.price_of_listing(lid) for lid in row_ids]
        if prices and all(price is not None and price > 10 * cap for price in prices):
            return None
        return cap

    def _home_shelf_label(self, slot: int) -> str:
        """The item's own category label in _plain_words form. Opened listings are consulted first, search rows only
        when no opened listing belongs to the item: the request's shelf or product name wins when such a listing
        carries it, else the label most of them carry. '' when nothing is known."""
        label_key = self.shelf_field_name()
        if not label_key:
            return ''
        good = self.goods[slot - 1]
        for pool in (self.page_records, self.row_records):
            counts = self._shelf_counts(good, pool, label_key)
            own = next((_bare_words(name) for name in (good.shelf_name, good.good_title) if _bare_words(name) in counts), None)
            if own is not None:
                return own
            if counts:
                return max(counts, key=counts.get)
        return ''

    def _shelf_counts(self, good: Any, pool: dict[str, Any], label_key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rec in pool.values():
            text = str(rec.get(label_key) or '')
            if text and self.good_has_shelf(good, text):
                flat = _bare_words(text)
                counts[flat] = counts.get(flat, 0) + 1
        return counts

    # ================================================================== listings opened before the interview

    def _spread_before_asking(self) -> None:
        """Give the interviewer a spread of listings first. A side-by-side tool that requires nothing but an id list
        (and is not known to hide attributes) shows a few unopened listings of every category group in one call;
        without such a tool the listings are opened one by one."""
        tool, spec, props = self._tool_spec_for('compare')
        ids_key = next((key for key, sub in props.items() if isinstance(sub, dict) and 'array' in _declared_types(sub)), None)
        workable = bool(tool) and bool(ids_key) and all(key == ids_key for key in spec.get('required') or [])
        if not workable or self.side_by_side_hides_traits():
            self._single_opens_if_possible()
            return
        if self.moves_left() <= 12:
            return
        ceiling = props[ids_key].get('maxItems')
        sample = self._sample_across_groups(ceiling if isinstance(ceiling, int) else 4)
        if len(sample) < 2:
            return
        args = self._conform_or_none({ids_key: sample}, spec)
        if args is None:
            return
        before = len(self.page_records)
        reply = self.shop_call(tool, args)
        if reply is not None:
            self._absorb_side_by_side(tool, spec, ids_key, sample, reply.get('observation'), before)

    def _sample_across_groups(self, cap: int) -> list[str]:
        groups = self.shelf_groups()
        share = max(1, cap // max(1, len(groups)))
        picks = [lid for members in groups for lid in [m for m in members if m not in self.page_records][:share]]
        return picks[:cap]

    def _absorb_side_by_side(self, tool: str, spec: dict[str, Any], ids_key: str, sample: list[str], view: Any, before: int) -> None:
        """Take in a side-by-side reply and settle whether the view shows attributes. If it does not, open the
        listings one by one instead; if it opened nothing new, ask once more for just the first two ids."""
        self.digest_reply(tool, view)
        self.note_view_keys(tool, view)
        rows = self.records_in_reply(view)
        if rows and not self.tool_shows_traits.get(tool, False):
            self.tool_shows_traits[tool] = any(_trait_block(rec) for _, rec in rows)
        gained = len(self.page_records) - before
        self.journal.append({'kind': 'widen', 'ids': sample, 'opened': gained})
        if rows and self.tool_shows_traits.get(tool) is False:
            self._single_opens_if_possible()
            return
        if gained == 0 and len(sample) > 2 and self.moves_left() > 12:
            self._retry_with_pair(tool, spec, ids_key, sample)

    def _retry_with_pair(self, tool: str, spec: dict[str, Any], ids_key: str, sample: list[str]) -> None:
        args = self._conform_or_none({ids_key: sample[:2]}, spec)
        if args is None:
            return
        reply = self.shop_call(tool, args)
        if reply is not None:
            self.digest_reply(tool, reply.get('observation'))

    def _single_opens_if_possible(self) -> None:
        reader = self.trait_viewer()
        if reader and self.lone_id_arg(reader):
            self._read_in_two_waves()

    def _read_in_two_waves(self) -> None:
        """Read listings singly with the one-id tool that shows attributes. Every category group contributes its best
        unopened listings up to a quota (lower when the request has three or more items). Wave one takes at most
        four per group, alternating between groups; wave two takes what is left, and the total stays small."""
        value_words = self._words_of_required_values()
        quota = READS_PER_SHELF if len(self.goods) <= 2 else 4
        queues: list[list[str]] = []
        heads: list[int] = []
        for members in self.shelf_groups():
            opened = len([lid for lid in members if lid in self.page_records])
            queues.append(self._best_unopened(members, value_words)[:max(0, quota - opened)])
            heads.append(max(0, 4 - opened))
        early = self._round_robin(queues, [(0, min(head, len(queue))) for queue, head in zip(queues, heads)], 4)
        late = self._round_robin(queues, [(head, len(queue)) for queue, head in zip(queues, heads)], quota)
        cut = max(5, 2 * len(queues) + 2)
        first_read = self.read_traits_of(early[:min(cut, 8)], 22)
        self.read_traits_of((early[cut:] + late)[:max(0, 10 - len(first_read))], 30)

    @staticmethod
    def _round_robin(queues: list[list[str]], spans: list[tuple[int, int]], rounds: int) -> list[str]:
        """Position k of every queue in turn, for k from 0 up to rounds-1, taking a queue's entry only while k lies
        in that queue's span [low, high)."""
        return [queue[k] for k in range(rounds) for queue, (low, high) in zip(queues, spans) if low <= k < high]

    def _best_unopened(self, members: list[str], value_words: set[str]) -> list[str]:
        """The group's buyable listings not yet opened, ordered by how many words their rows share with the
        requirement values (cheapest first between equals when the request asks for the cheapest)."""
        pool = [lid for lid in members if lid not in self.page_records and self.is_purchasable(lid)]
        passes: list[Any] = [self._price_or_inf] if self.digest.get('pick') == 'cheapest' else []
        passes.append(lambda lid: -len(value_words & self._row_tokens(lid)))
        for order_key in passes:
            pool.sort(key=order_key)
        return pool

    @staticmethod
    def _value_part(need: str) -> str:
        head, colon, tail = need.partition(':')
        return tail if colon else head

    def _words_of_required_values(self) -> set[str]:
        bag: set[str] = set()
        for _, need in self.needs_in_force():
            bag.update(re.findall(self._TOKEN3_RX, self._value_part(need).lower()))
        return bag

    def _row_tokens(self, lid: str) -> set[str]:
        return set(re.findall(self._TOKEN3_RX, _jdump(self.row_records.get(lid) or {}).lower()))

    def _price_or_inf(self, lid: str) -> float:
        price = self.price_of_listing(lid)
        return float('inf') if price is None else price

    # ================================================================== the interview

    def ready_for_questions(self) -> bool:
        """Whether the interview should start now. It needs a message tool, spare messages, room in steps and time,
        and either enough opened listings (over three per item) or a quarter of the step allowance (8 or more)
        already used."""
        if self.interview_ran or not self.role_map.get('ask'):
            return False
        if self.free_message_count() <= 0 or self.moves_left() <= 8:
            return False
        if self.seconds_spent() > ASKING_CUTOFF:
            return False
        enough_open = len(self.page_records) > 3 * len(self.goods)
        return enough_open or self.moves_made >= max(8, self.step_cap // 4)

    def conduct_interview(self) -> str:
        """The one interview of the episode: its report for the loop model, '' when no message can be spared."""
        self.interview_ran = True
        if self.free_message_count() <= 0:
            return ''
        return self._run_interview()

    def _run_interview(self) -> str:
        """Spread some listings out, plan what to ask, ask while messages, steps and time last, then search the
        values that came out of the answers."""
        self._spread_before_asking()
        remaining = self.message_cap - self.messages_out
        self._reset_tracks(min(12, max(4 * remaining, TRAITS_PER_QUESTION)) if self.lists_ok else remaining)
        self.queued_followups = []
        talk: list[tuple[int, str, str]] = []
        gained: list[str] = []
        while self._may_ask_again():
            if not self._one_question_round(gained, talk):
                break
        told_slots = [slot for slot, count in self.track_hits.items() if count > 0]
        found = self._probe_disclosed_values(told_slots) if told_slots else ''
        if gained:
            self.revision += 1
        return self._interview_report(talk, gained, found)

    def _may_ask_again(self) -> bool:
        if self.free_message_count() <= 0 or self.run_closed:
            return False
        return self.moves_left() > 6 and self.seconds_spent() < QUESTION_CUTOFF

    def _one_question_round(self, gained: list[str], talk: list[tuple[int, str, str]]) -> bool:
        """Ask once more. Every free message is spent while steps and time last, even after a reply that brought out
        nothing: the shopper answers only the first property of a list it keeps track of, so a need hidden behind
        properties it does not mind comes out on the second or third message, once the rotation has moved those
        out of the way."""
        count_before, cap_before = len(gained), self.money_cap
        sent = self.ask_next_question(gained)
        talk.extend(sent)
        if not sent:
            return False
        if len(gained) == count_before and self.money_cap == cap_before:
            self.empty_answers += 1
            self.journal.append({'kind': 'ask_fruitless', 'q': sent[-1][1][:120]})
        return True

    def _reset_tracks(self, per_item: int) -> None:
        self.question_tracks = planned = self.plan_question_tracks(per_item)
        self.track_hits = {slot: 0 for slot in planned}
        self.track_messages = {slot: 0 for slot in planned}
        self.track_asked = {slot: set() for slot in planned}

    def _interview_report(self, talk: list[tuple[int, str, str]], gained: list[str], found: str) -> str:
        lines = [f'Q: {asked}\nA: {reply}' for _, asked, reply in talk]
        report = "Wren's interviewer put these questions to the shopper (each one cost a step):\n" + '\n'.join(lines)
        report += '\nNew firm requirements from these answers: ' + ('; '.join(gained) if gained else 'none')
        report += '. Every pick must meet them (search again with the new values where useful).'
        report += self._linked_value_hint(gained, ', and an order counts only after each of them has been opened.')
        if found:
            report += '\n' + found
        waiting = self.pending_notes
        if waiting:
            report += '\n[notes that arrived in the meantime] ' + ' | '.join(waiting)
            self.pending_notes = []
        return report

    def _linked_value_hint(self, gained: list[str], tail: str) -> str:
        linked = [need for _, need in self.borrowed_needs if any(need in line for line in gained)]
        if not linked:
            return ''
        return (' These values come from a different listing: ' + ' | '.join(linked)
                + '. Find it, read its whole page and also open every listing that page refers to; the value sits'
                + ' on the last page in that chain' + tail)

    # ================================================================== the question held back for checkout

    def final_question_pass(self) -> str:
        """Right before checkout, spend one message the order itself does not need on the next planned question.

        While the interview runs, every message a process step of the request might need is held back, for a step
        tied to the order as well as for one tied to ending the episode without a purchase; once Wren is about to
        buy, the messages kept for that ending are free. Race-13 evidence (an allowance of two messages): the base
        sent only one question in 28 of 180 episodes, 12 of those left a hidden need unasked, and none of the
        episodes holding such a need passed. Returns a summary for the loop model when the answer brings a new
        requirement, else ''."""
        if not self._late_question_allowed():
            return ''
        self.final_question_spent = True
        if not self.question_tracks:
            self._reset_tracks(TRAITS_PER_QUESTION if self.lists_ok else 1)
        gained: list[str] = []
        cap_before = self.money_cap
        sent = self.ask_next_question(gained, LAST_QUESTION_STEP_ROOM - 2)
        if not sent:
            self.journal.append({'kind': 'late_ask', 'skipped': 'no question left to ask, or the message was not delivered'})
            return ''
        return self._report_late_answer(sent[0], gained, cap_before)

    def _late_question_allowed(self) -> bool:
        """Checked in this order, each veto ending the check: already spent, an earlier fruitless answer, no message
        tool, no message left, an order-time step still owed a message, too few steps, too late, no judge seat."""
        vetoes = (
            lambda: self.final_question_spent,
            lambda: self.empty_answers > 0,
            lambda: not self.role_map.get('ask'),
            lambda: self.messages_out >= self.message_cap,
            lambda: any(entry['when'] == 'order' and entry['messages'] > 0 for entry in self.procedure_steps),
            lambda: self.moves_left() < LAST_QUESTION_STEP_ROOM,
            lambda: self.seconds_spent() > LAST_QUESTION_CUTOFF,
            lambda: not self.audit_seats(),
        )
        return not any(veto() for veto in vetoes)

    def _report_late_answer(self, first: tuple[int, str, str], gained: list[str], cap_before: Any) -> str:
        if self.money_cap != cap_before:
            gained.append(f'budget now {self.money_cap}')
        slot, asked, reply = first
        self.journal.append({'kind': 'late_ask', 'q': asked, 'a': reply[:200], 'told': gained})
        if not gained:
            return ''
        self.revision += 1
        note = f'Right before checkout Wren sent the shopper one further question (one step):\nQ: {asked}\nA: {reply}\n'
        note += 'New firm requirements from this answer: ' + '; '.join(gained)
        note += ('. Nothing has been checked out yet: hold every cart line against them (search again with the new'
                 ' values where useful), fix the cart, and only then check out.')
        note += self._linked_value_hint(gained, '.')
        found = self._probe_disclosed_values([slot])
        return note + '\n' + found if found else note

    # ================================================================== which property to ask next, and the wording

    def next_traits_to_ask(self, tracks: dict[int, list[str]], hits: dict[int, int], sent: dict[int, int], asked: dict[int, set[str]]) -> tuple[int, list[str]]:
        """Which item to ask about and which labels go into the message. Items for which no answer has yielded a need
        yet come first; then, unless messages are scarce while lists go unanswered, the item asked least so far;
        then the order the request hinted at. Returns (item number, up to one message's worth of labels not yet asked
        or answered), or (0, []) once no lane has anything left."""
        order_pos = {slot: idx for idx, slot in enumerate(self.hinted_good_order)}
        scarce = self.message_cap <= SCANT_MESSAGE_ALLOWANCE and not self.lists_ok
        per_message = self.list_width()

        def urgency(slot: int) -> tuple[bool, int, int]:
            return (hits.get(slot, 0) > 0, 0 if scarce else sent.get(slot, 0), order_pos.get(slot, len(order_pos) + slot))

        for slot in sorted(tracks, key=urgency):
            fresh = [label for label in tracks[slot] if self._label_still_open(slot, label, asked)]
            if fresh:
                return slot, fresh[:per_message]
        return 0, []

    def list_width(self) -> int:
        """How many properties the next message names. Shoppers differ: some answer a list about whichever need it
        names, others turn a message with several subjects away; nothing in a task says which, but the replies do. So
        the first message names a short list (two on a one-product task, three when it is the only message or there
        are several products); a list answered without complaint lets the next be twice as long, and a need told
        from past a list's first place opens the full width (eight, six on several products). A reply asking for one
        subject a message turns lists off (one property a message)."""
        if not self.lists_ok:
            return 1
        one = len(self.goods) == 1
        full = TRAITS_PER_QUESTION if one else TRAITS_PER_QUESTION - 2
        if self.list_asks == 0:
            free = self.message_cap - self.messages_out
            return (3 if free <= 1 else FIRST_LIST_WIDTH) if one else 3
        if self.lists_read:
            return full
        if self.list_taken:
            return min(full, 2 * self.list_taken)
        return 4 if one else 3

    def _label_still_open(self, slot: int, label: str, asked: dict[int, set[str]]) -> bool:
        return _bare_words(label) not in asked[slot] and not self.answered_before(slot, label)

    def answered_before(self, slot: int, label: str) -> bool:
        """Whether an answer about this item already settled the property `label` names. Both names are reduced to
        folded words (a trailing plural -s removed, -our turned into -or, so colour matches color and singular
        matches plural). The label may sit inside the answered name ("finish" inside "surface finish"), or an
        answered name of two or more words may sit inside the label; a single answered word ("size") never covers a
        longer label ("frame size")."""
        target = _canonical_key(label)
        for told in self.goods[slot - 1].told_needs:
            if _same_trait_name(target, _canonical_key(told.partition(':')[0])):
                return True
        return False

    def sized_question(self, slot: int, names: list[str], examples: bool) -> tuple[list[str], str]:
        """The longest front part of `names` whose question the message tool accepts unaltered, with that question.
        Shaping cuts text at the schema's length limit, so names come off the end until the full question fits;
        ([], '') when not even one name fits."""
        size = len(names)
        while size > 0:
            head = names[:size]
            text = self._wording_for(slot, head, examples)
            if self._fits_message_tool(text):
                return head, text
            size -= 1
        return [], ''

    def _wording_for(self, slot: int, names: list[str], examples: bool = False) -> str:
        """The text the shopper reads: one list question for two or more names, otherwise the single-property
        question (carrying two sample values when `examples` is set)."""
        if len(names) >= 2:
            *front, last = names
            head = ', '.join(front)
            return f'What {head} or {last} would you like for the {self.good_phrase(slot)}?'
        name = names[0]
        return self.one_trait_question(slot, name, self.sample_values(slot, name) if examples else None)

    def _fits_message_tool(self, text: str) -> bool:
        packed = self.message_args_for(text)
        return text in packed.values() if packed else False

    def message_args_for(self, text: str) -> dict[str, Any] | None:
        """Message-tool arguments that carry `text`. The text fills the first required free-text field (or the first
        free-text field when none is required); other required fields get their first listed option. None if a
        required field lists no options or the shaped arguments fail the schema."""
        _tool, spec, props = self._tool_spec_for('ask')
        required = spec.get('required') or []
        texty = self._text_slots(props)
        target = next((key for key in required if key in texty), None) or (texty[0] if texty else None)
        if not target:
            return None
        defaults = self._default_enum_args(required, props, target)
        if defaults is None:
            return None
        return self._conform_or_none({target: text, **defaults}, spec)

    @staticmethod
    def _default_enum_args(required: list[str], props: dict[str, Any], text_key: str) -> dict[str, Any] | None:
        """First listed option for each required field other than the text field, in required order; None as soon
        as one of them lists no options."""
        picked: dict[str, Any] = {}
        for key in required:
            if key == text_key or key in picked:
                continue
            options = (props.get(key) or {}).get('enum')
            if not (isinstance(options, list) and options):
                return None
            picked[key] = options[0]
        return picked

    # ================================================================== schema access

    def _tool_spec_for(self, role: str) -> tuple[str | None, dict[str, Any], dict[str, Any]]:
        """(tool playing `role`, its argument schema or {} if unknown, the schema's properties)."""
        name = self.role_map.get(role)
        spec = self.schemas.get(name or '') or {}
        return name, spec, spec.get('properties') or {}

    @staticmethod
    def _text_slots(props: dict[str, Any]) -> list[str]:
        """Properties that accept free text (string type, no enumerated options), kept in schema order."""
        return [key for key, sub in props.items() if isinstance(sub, dict) and 'string' in _declared_types(sub) and 'enum' not in sub]

    @staticmethod
    def _conform_or_none(args: dict[str, Any], spec: Any) -> dict[str, Any] | None:
        issues: list[str] = []
        fitted = _fit_to_schema(args, spec, 'args', issues)
        return None if issues else fitted
class RunInterview:
    """Wren's interviewer. It decides which property of which requested item goes to the shopper next, in what words
    and in what order, and it works every reply back into the plan."""

    # ======================================================== text the shopper reads
    # Every string this block builds is sent to the shopper as it stands. Its fixed words were tuned on race data,
    # so the produced text must not change by a single character.

    def good_phrase(self, slot: int) -> str:
        """The item's name as a question spells it.

        When the name strings several plural products together with "and", the question uses "or" there instead.
        In race 13 (ten agents) such "and" names, like "door handles and locks", met a please-ask-one-thing reply on
        81 of 155 questions; names with "or" met it on none of 50, and "dresser or cabinet" brought out a need in 10
        of 28. Once the shopper has asked for one thing at a time about this item, a single joined product is named
        instead (chosen by _wq_best_known_product)."""
        good = self.goods[slot - 1]
        title = good.good_title
        joined = self._wq_joined_products(slot)
        if not joined:
            return title
        if slot in self.lone_goods:
            return self._wq_best_known_product(good, joined)
        return re.sub('\\band\\b', 'or', title, flags=re.I)

    def _wq_joined_products(self, slot: int) -> list[str]:
        """The distinct products an item name chains together with commas or "and", but only when each piece ends as
        a plural noun ("door handles and locks"). A single product that just has "and" in its name gives []
        ("salt and pepper grinder", "usb and lightning cables", "his and hers towels")."""
        title = self.goods[slot - 1].good_title
        if not re.search('\\band\\b', title, re.I):
            return []
        chunks = [chunk.strip() for chunk in re.split('\\s*(?:,|\\band\\b)\\s*', title, flags=re.I)]
        chunks = [chunk for chunk in chunks if chunk]
        if len(chunks) < 2:
            return []
        every_plural = all(re.search('[a-z]{3}(?<![su])s$', chunk, re.I) for chunk in chunks)
        return chunks if every_plural else []

    def _wq_best_known_product(self, good: RequestedGood, joined: list[str]) -> str:
        """Of the joined products, the one whose final word shows up in the most listing rows of this item (the
        category field is not searched). A tie, or an item without rows, falls to the piece listed first."""
        label_key = self.shelf_field_name()
        bags = []
        for row in self.row_records.values():
            if label_key and not self.good_has_shelf(good, str(row.get(label_key) or '')):
                continue
            prose = ' '.join(str(value) for field, value in row.items() if field != label_key and isinstance(value, str))
            bags.append(_word_roots(prose))
        hits = [sum(1 for bag in bags if _word_roots(name.split()[-1]) & bag) for name in joined]
        return joined[max(range(len(joined)), key=hits.__getitem__)]

    def one_trait_question(self, slot: int, label: str, examples: list[str] | None = None) -> str:
        """The question about a single property; given examples, it offers the first two of them."""
        if examples:
            first, second = examples[0], examples[1]
            tail = f', for example {first} or {second}?'
        else:
            tail = '?'
        return f'What {label} would you like for the {self.good_phrase(slot)}{tail}'

    def sample_values(self, slot: int, label: str) -> list[str]:
        """At most two values seen under the label's key, each 30 characters or shorter, that the item's stated
        needs do not already include."""
        plain = _bare_words(label)
        stated = ' '.join(self.goods[slot - 1].given_needs).lower()
        values = next((row[3] for row in self.trait_key_table() if row[1] == plain), [])
        usable = [value for value in values if value and len(value) <= 30 and value not in stated]
        return usable[:2]

    # ======================================================== a single question round

    def ask_next_question(self, news: list[str], floor: int = 6) -> list[tuple[int, str, str]]:
        """Put one question to the shopper, a queued follow-up taking precedence over the planned labels. Its reply
        is worked into the items' needs (each new need is also added to `news`) and the labels it named are
        settled. Returns the exchanges that went out, [] if no message was sent."""
        choice = self._wq_choose_question()
        if choice is None:
            return []
        slot, labels, with_examples, text = choice
        sent = self.deliver_messages([(slot, labels[0], text)], floor)
        if sent:
            self._wq_absorb_reply(slot, labels, with_examples, sent, news)
        return sent

    def _wq_choose_question(self) -> tuple[int, list[str], bool, str] | None:
        """No more attempts than one list holds labels. Each attempt takes the next candidate and keeps the longest
        leading run of its labels that the message tool carries whole (dropping the example values when only they
        made it too long). A label that does not fit even on its own is booked as asked; then the next one is
        tried."""
        for _attempt in range(TRAITS_PER_QUESTION):
            slot, labels, with_examples = self._wq_next_in_line()
            if not labels:
                break
            fitted, text = self._wq_fit(slot, labels, with_examples)
            if fitted:
                return slot, fitted, with_examples, text
            self.track_asked.setdefault(slot, set()).add(_bare_words(labels[0]))
            self.journal.append({'kind': 'ask_too_long', 'item': slot, 'label': labels[0]})
        return None

    def _wq_fit(self, slot: int, labels: list[str], with_examples: bool) -> tuple[list[str], str]:
        fitted, text = self.sized_question(slot, labels, with_examples)
        if fitted or not with_examples:
            return fitted, text
        return self.sized_question(slot, labels, False)

    def _wq_next_in_line(self) -> tuple[int, list[str], bool]:
        """The first queued follow-up whose leading label is still open (queued ones an answer already covers are
        thrown away); with none left, the next labels of the planned tracks."""
        self.queued_followups = [entry for entry in self.queued_followups if not self.answered_before(entry[0], entry[1][0])]
        if self.queued_followups:
            return self.queued_followups.pop(0)
        slot, labels = self.next_traits_to_ask(self.question_tracks, self.track_hits, self.track_messages, self.track_asked)
        return slot, labels, False

    def _wq_absorb_reply(self, slot: int, labels: list[str], with_examples: bool, sent: list[tuple[int, str, str]], news: list[str]) -> None:
        self.track_messages[slot] = self.track_messages.get(slot, 0) + 1
        self.list_asks += 1
        counts_before = [len(good.told_needs) for good in self.goods]
        reading = self._wq_read_replies(sent)
        if len(labels) > 1:
            reading.pop('free', None)
        self.fold_in_answers(reading, news)
        gained: list[str] = []
        for number, (good, start) in enumerate(zip(self.goods, counts_before), 1):
            new = good.told_needs[start:]
            if new:
                self.track_hits[number] = self.track_hits.get(number, 0) + 1
            gained += new
        crowded = any(_coerce_int(entry) == 1 for entry in _listify(reading.get('crowded')))
        self._wq_settle(slot, labels, with_examples, gained, crowded)

    # ======================================================== messages out, replies in

    def deliver_messages(self, questions: list[tuple[int, str, str]], floor: int = 6) -> list[tuple[int, str, str]]:
        """Send the (item, label, text) questions in turn through the messaging tool while more than `floor` steps
        and at least one message remain. The first refused call or error reply ends the run. Returns (item, text,
        reply words) for each message that actually went out."""
        tool = self.role_map['ask']
        sent: list[tuple[int, str, str]] = []
        for slot, _label, text in questions:
            if not self._wq_send_one(tool, slot, text, floor, sent):
                break
        return sent

    def _wq_send_one(self, tool: str, slot: int, text: str, floor: int, sent: list[tuple[int, str, str]]) -> bool:
        """Send a single message; True when the next one may go after it."""
        if not self._wq_can_message(floor):
            return False
        args = self.message_args_for(text)
        if not args:
            return False
        result = self.shop_call(tool, args)
        if result is None:
            return False
        self.messages_out += 1
        observation = result.get('observation')
        reply = self.message_words(observation)
        sent.append((slot, text, reply))
        self._wq_log_exchange(text, reply)
        failed = isinstance(observation, dict) and observation.get('error')
        return not failed

    def _wq_can_message(self, floor: int) -> bool:
        if self.run_closed or self.moves_left() <= floor:
            return False
        return self.messages_out < self.message_cap

    def _wq_log_exchange(self, text: str, reply: str) -> None:
        at = self.moves_made
        cart = self.basket_fingerprint()
        self.act_log.append(f'step {at}: message to the shopper {text!r} -> reply {reply[:300]!r} (cart then: {cart})')
        self.act_steps.append(at)
        self.journal.append({'kind': 'ask', 'q': text, 'a': reply[:300]})

    def _wq_read_replies(self, sent: list[tuple[int, str, str]]) -> dict[str, Any]:
        items = '\n'.join(f'item {number}: {good.good_title}' for number, good in enumerate(self.goods, 1))
        exchanges = '\n'.join(f'{pos}. (item {number}) Q: {question}\nA: {answer}'
                              for pos, (number, question, answer) in enumerate(sent, 1))
        prompt = '\n\n'.join((REPLY_PARSE_PROMPT, 'ITEMS:\n' + items, 'EACH QUESTION SENT, FOLLOWED BY ITS REPLY:\n' + exchanges))
        return self.ask_model_json(prompt, [QUICK_SEAT, DEEP_SEAT], 2500) or {}

    def fold_in_answers(self, reading: dict[str, Any], news: list[str]) -> None:
        """Apply a parsed reply to the items: needs it adds (pointer needs are noted as well), requirements it takes
        back, properties the shopper is indifferent to, and a new budget above zero. Lines meant for the loop model
        are appended to `news`."""
        handlers = {'needs': self._wq_add_need, 'drop': self._wq_drop_needs, 'free': self._wq_mark_indifferent}
        for field, handle in handlers.items():
            rows = [row for row in _listify(reading.get(field)) if isinstance(row, dict)]
            for row in rows:
                handle(row, news)
        limit = reading.get('budget')
        if isinstance(limit, bool) or not isinstance(limit, (int, float)):
            return
        if limit > 0:
            self.money_cap = self.budget_in_unit(limit)

    def _wq_add_need(self, row: dict[str, Any], news: list[str]) -> None:
        need = str(row.get('need') or '').strip()
        if not need:
            return
        slot = self._wq_clamped_item(row.get('item'))
        good = self.goods[slot - 1]
        if need not in good.told_needs:
            good.told_needs.append(need)
            news.append(need if len(self.goods) <= 1 else f'item {slot}: {need}')
        pointer = (slot, need)
        if row.get('points') is True and pointer not in self.borrowed_needs:
            self.borrowed_needs.append(pointer)

    def _wq_drop_needs(self, row: dict[str, Any], news: list[str]) -> None:
        prop = str(row.get('property') or '')
        if not prop.strip():
            return
        good = self.goods[self._wq_clamped_item(row.get('item')) - 1]
        target = _bare_words(prop.split(':')[0])
        withdrawn = [need for need in good.given_needs + good.told_needs
                     if need not in good.dropped_needs and _bare_words(need.split(':')[0]) == target]
        good.dropped_needs.extend(withdrawn)
        news.extend(f'dropped {need}' for need in withdrawn)

    def _wq_mark_indifferent(self, row: dict[str, Any], news: list[str]) -> None:
        prop = row.get('property')
        if prop:
            self.goods[self._wq_clamped_item(row.get('item')) - 1].open_traits.append(str(prop))

    # ======================================================== booking what a reply settled

    def _wq_settle(self, slot: int, labels: list[str], with_examples: bool, gained: list[str], crowded: bool) -> None:
        """Book what the reply showed about the label, or the list of labels, its question named."""
        asked = self.track_asked.setdefault(slot, set())
        if len(labels) == 1:
            self._wq_settle_single(slot, labels, with_examples, asked, gained, crowded)
        else:
            self._wq_settle_list(slot, labels, asked, gained, crowded)

    def _wq_settle_single(self, slot: int, labels: list[str], with_examples: bool, asked: set[str], gained: list[str], crowded: bool) -> None:
        """A lone label is always booked as asked. A narrower follow-up is queued only after a reply that pleaded
        for one thing at a time and brought out no need."""
        asked.add(_bare_words(labels[0]))
        if not crowded or gained:
            return
        follow = self._wq_narrower_followup(slot, labels, with_examples, asked)
        if follow is not None:
            self.queued_followups.insert(0, follow)

    def _wq_narrower_followup(self, slot: int, labels: list[str], with_examples: bool, asked: set[str]) -> tuple[int, list[str], bool] | None:
        """The follow-up to a one-label question that drew a one-thing-at-a-time plea, in the base agent's order:
        the same label with the item named as just one of its joined products; failing that, the label's last word
        on its own if the label has several words and that word was never asked; failing that, the same label with
        two example values."""
        if slot not in self.lone_goods and self._wq_joined_products(slot):
            self.lone_goods.add(slot)
            self.journal.append({'kind': 'one_product', 'item': slot, 'said': self.good_phrase(slot)})
            return slot, labels, with_examples
        words = labels[0].split()
        last = words[-1]
        if len(words) > 1 and _bare_words(last) not in asked:
            return slot, [last], False
        if with_examples:
            return None
        return (slot, labels, True) if len(self.sample_values(slot, labels[0])) >= 2 else None

    def _wq_settle_list(self, slot: int, labels: list[str], asked: set[str], gained: list[str], crowded: bool) -> None:
        """Book a question that named a list of labels.

        The shopper replies about the first property of a message that it keeps track of, and a message that names
        none of those properties draws the same reply as a property it is indifferent to. So a revealed need settles
        its own label together with every label before it, and the labels after it start the item's next list. A
        reply that reveals nothing settles only the first label (a need there would have surfaced) and queues the
        rest in reverse order, so that labels the shopper is indifferent to stop holding up the others; on every
        second such reply in a row the item's labels not yet named go first instead, otherwise the order would
        return to that of the first list. A plea for one question at a time turns lists off for the rest of the
        episode, and the list's first label is then asked alone."""
        if crowded and not gained:
            self.lists_ok = False
            self.journal.append({'kind': 'lists_off', 'item': slot})
            self.queued_followups.insert(0, (slot, labels[:1], False))
            return
        cut = self._wq_earliest_answered(labels, gained) if gained else -1
        # a list answered without complaint: the shopper takes lists at least this long, and a need told from past
        # the first place shows the whole list was read
        self.list_taken = max(self.list_taken, len(labels))
        if cut >= 1:
            self.lists_read = True
        if cut >= 0:
            done, ahead = labels[:cut + 1], labels[cut + 1:]
            dry = 0
        else:
            done, ahead = [], list(reversed(labels[1:]))
            dry = self.track_dry_streak.get(slot, 0) + 1
        asked.update(_bare_words(label) for label in (done or labels[:1]))
        self.track_dry_streak[slot] = dry
        rest = [label for label in self.question_tracks.get(slot) or [] if label not in labels]
        # after a reply that told nothing, two properties never named yet lead (the likelier home of a need), then
        # the ones the last list named after its first, the last named first
        self.question_tracks[slot] = ahead + rest if cut >= 0 else rest[:2] + ahead + rest[2:]
        self.journal.append({'kind': 'list_settled', 'item': slot, 'done': done, 'next': ahead[:4]})

    def _wq_earliest_answered(self, labels: list[str], gained: list[str]) -> int:
        """Index of the earliest label in the list that one of the new needs answers; -1 if no need maps onto any
        label (that reply is then booked like one that revealed nothing)."""
        value_words: dict[str, set[str]] = {}
        for _shelf, key, _count, seen in self.trait_key_table():
            bucket = value_words.setdefault(_bare_words(key), set())
            for value in seen:
                bucket.update(_word_roots(value))
        hits = [self._wq_label_answered_by(labels, need, value_words) for need in gained]
        return min((pos for pos in hits if pos is not None), default=-1)

    def _wq_label_answered_by(self, labels: list[str], need: str, value_words: dict[str, set[str]]) -> int | None:
        """The label a new need belongs to: first by the most name words shared with the need's property (with
        spellings folded), else by the most words shared between the label's seen values and the need's value;
        None when both come up empty."""
        prop, _colon, value = need.partition(':')
        by_name = [len(_canonical_key(label) & _canonical_key(prop)) for label in labels]
        if max(by_name, default=0) > 0:
            return self._wq_first_top(by_name)
        value_roots = _word_roots(value)
        by_value = [len(value_words.get(_bare_words(label), set()) & value_roots) for label in labels]
        return self._wq_first_top(by_value)

    @staticmethod
    def _wq_first_top(scores: list[int]) -> int | None:
        top = max(scores, default=0)
        return scores.index(top) if top > 0 else None

    # ======================================================== planning the question tracks

    def plan_question_tracks(self, per_good: int) -> dict[int, list[str]]:
        """An ordered list of property labels to ask for every item. The strong model's picks come first (at most
        `per_good` per item, with the request's ASK FIRST entries and the hinted properties at the front), spelled
        as the item's own keys; the most widely carried keys are moved up among them, every other ranked key
        follows, and labels that lump features together always close the list."""
        self.adopt_loose_shelves()
        keyrows = self.trait_key_table()
        prompt = self._wq_plan_prompt(per_good, keyrows)
        answer = self.ask_model_json(prompt, [DEEP_SEAT, QUICK_SEAT], 3500) or {}
        chosen, hinted = self._wq_model_picks(answer)
        tracks: dict[int, list[str]] = {}
        for slot, picks in chosen.items():
            tracks[slot] = self._wq_build_lane(slot, picks, hinted[slot], keyrows)
        self.journal.append({'kind': 'plan', 'ask': _clip_texts(answer.get('ask'), 120), 'lanes': {slot: lane[:8] for slot, lane in tracks.items()}})
        return tracks

    def _wq_plan_prompt(self, per_good: int, keyrows: list[tuple[str, str, int, list[str]]]) -> str:
        roster = []
        for number, good in enumerate(self.goods, 1):
            settled = '; '.join(good.given_needs) or 'nothing'
            roster.append(f'item {number}: {good.good_title} - the request already settles: {settled}')
        blocks = [ASKING_PLAN_PROMPT.replace('COUNT', str(max(1, per_good))),
                  'REQUEST:\n' + self.shopper_request,
                  'ITEMS:\n' + '\n'.join(roster)]
        confirm = [str(entry) for entry in _listify(self.digest.get('ask_first')) if str(entry).strip()]
        if confirm:
            blocks.append('ASK FIRST (the request says to check these with the shopper):\n' + '\n'.join(confirm))
        listing = '\n'.join(self._wq_key_lines(keyrows))
        blocks.append('ATTRIBUTE KEYS ON OPENED LISTINGS:\n' + (listing or '(nothing opened in full so far)'))
        return '\n\n'.join(blocks)

    def _wq_key_lines(self, keyrows: list[tuple[str, str, int, list[str]]]) -> list[str]:
        lines: list[str] = []
        shelf_now = None
        for pos, (shelf, key, many, seen) in enumerate(keyrows, 1):
            if shelf != shelf_now:
                lines.append(f'category {shelf!r}:')
                shelf_now = shelf
            note = self.different_trait_note(shelf, key, seen)
            lumped = ' [bundles several features: ask it last]' if re.search(BUNDLE_KEY_RX, key) else ''
            lines.append(f'  {pos}. {key} (seen on {many} opened): ' + ' | '.join(seen) + note + lumped)
        return lines

    def _wq_model_picks(self, answer: dict[str, Any]) -> tuple[dict[int, list[str]], dict[int, set[str]]]:
        """Per item, the labels the model picked (trimmed by _wq_trimmed_label, five words at most, no repeats), and
        the plain forms of those it tied to a hint; an item with any hint is added to the hinted order."""
        chosen: dict[int, list[str]] = {slot: [] for slot in range(1, len(self.goods) + 1)}
        hinted: dict[int, set[str]] = {slot: set() for slot in chosen}
        entries = [entry for entry in _listify(answer.get('ask')) if isinstance(entry, dict)]
        for entry in entries:
            slot = self._wq_clamped_item(entry.get('item'))
            label = self._wq_trimmed_label(entry.get('label'))
            if not label or len(label.split()) > 5:
                continue
            plain = _bare_words(label)
            if plain not in {_bare_words(prev) for prev in chosen[slot]}:
                chosen[slot].append(label)
            if str(entry.get('hint') or '').strip():
                hinted[slot].add(plain)
                if slot not in self.hinted_good_order:
                    self.hinted_good_order.append(slot)
        return chosen, hinted

    @staticmethod
    def _wq_trimmed_label(raw: Any) -> str:
        """A picked label up to its first joining word ("and", "or") or separator, without a closing ? or ."""
        head = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(raw or ''))[0]
        return head.strip().rstrip('?.')

    def _wq_build_lane(self, slot: int, picks: list[str], hinted: set[str], keyrows: list[tuple[str, str, int, list[str]]]) -> list[str]:
        ranked = self._wq_keys_by_reach(slot, keyrows)
        reach = self._wq_reach_table(slot, keyrows)
        backed = {_bare_words(self._wq_house_spelling(hint, reach)) for hint in hinted}
        lane = self._wq_merge_unique([], [self._wq_house_spelling(pick, reach) for pick in picks])
        widely = [key for key in ranked if reach.get(key, 0.0) >= SHARED_KEY_FLOOR and not re.search(BUNDLE_KEY_RX, key)]
        for key in widely[:SHARED_KEY_COUNT]:
            lane = self._wq_promote_common_key(lane, key, reach, backed) or lane
        self._wq_merge_unique(lane, ranked)
        return self._wq_lumped_to_back(lane, _bare_words)

    def _wq_reach_table(self, slot: int, keyrows: list[tuple[str, str, int, list[str]]]) -> dict[str, float]:
        """Per key of the item's own categories: the best share of its fully opened records (with attributes) that
        carry the key."""
        opened = max(1, self._wq_opened_with_traits(slot))
        reach: dict[str, float] = {}
        for shelf, key, many, _seen in keyrows:
            if self.good_for_shelf(shelf) == slot:
                reach[key] = max(reach.get(key, 0.0), many / opened)
        return reach

    def _wq_house_spelling(self, label: str, reach: dict[str, float]) -> str:
        """The item's own key that the label names once plurals and -our are folded; the label itself otherwise."""
        target = self._wq_folded_spelling(label)
        return next((key for key in reach if self._wq_folded_spelling(key) == target), label)

    @staticmethod
    def _wq_merge_unique(lane: list[str], labels: list[str]) -> list[str]:
        """Append to the lane each label whose plain form it does not hold yet, and hand the lane back."""
        for label in labels:
            if _bare_words(label) not in {_bare_words(held) for held in lane}:
                lane.append(label)
        return lane

    def _wq_promote_common_key(self, lane: list[str], key: str, reach: dict[str, float], backed: set[str]) -> list[str] | None:
        """A copy of the lane with a widely carried key placed among its first three labels, or None to leave the
        lane alone.

        The key takes the place of the first of those three labels that no hint supports and whose share falls
        short of the key's by at least the margin; with more than one item that label moves down to fourth place,
        with a single item the key is simply put in front of it. When no label qualifies, the key can only take a
        free place among the first three. Any change that would push a hinted label out of the first three is
        refused."""
        plain = _bare_words(key)
        front = lane[:3]
        front_plain = [_bare_words(label) for label in front]
        if plain in front_plain:
            return None
        goal = reach.get(key, 0.0)
        weak = next((pos for pos, label in enumerate(front)
                     if front_plain[pos] not in backed and self._wq_reach_of(label, reach) + SHARED_KEY_MARGIN <= goal), None)
        if weak is None and len(front) >= 3:
            return None
        rest = [label for label in lane if _bare_words(label) != plain]
        if weak is None:
            rest.append(key)
        elif len(self.goods) <= 1:
            rest.insert(min(len(rest), weak), key)
        else:
            displaced = rest[weak]
            rest[weak] = key
            rest.insert(min(3, len(rest)), displaced)
        kept_front = {_bare_words(label) for label in rest[:3]}
        if any(name in backed and name not in kept_front for name in front_plain):
            return None
        return rest

    @staticmethod
    def _wq_reach_of(label: str, reach: dict[str, float]) -> float:
        """The share of the item's opened listings carrying the label's key; failing an exact key, the best share
        among keys for the same property; 0.0 if none matches."""
        plain = _bare_words(label)
        if plain in reach:
            return reach[plain]
        mine = _stem_set(label)
        alike = []
        for key, part in reach.items():
            theirs = _stem_set(key)
            if _same_trait_name(mine, theirs) or _same_trait_name(theirs, mine):
                alike.append(part)
        return max(alike, default=0.0)

    @staticmethod
    def _wq_folded_spelling(text: str) -> str:
        """The plain words with a final plural s dropped and a British -our ending turned into -or."""
        singular = re.sub('(?<=[a-z]{3})s\\b', '', _bare_words(text))
        return re.sub('our\\b', 'or', singular)

    def _wq_keys_by_reach(self, slot: int, keyrows: list[tuple[str, str, int, list[str]]]) -> list[str]:
        """The item's attribute keys, the most widely carried first; on a tie a key whose values are mostly numbers
        goes after one that is not, and more distinct values go first. A single item without a category of its own
        takes every row. Keys the request already settles and keys that are no property (codes, links, warranty and
        the like) are dropped, and keys that lump features together end the list."""
        mostly_numbers = {key: self._wq_numeric_majority(seen) for _shelf, key, _count, seen in keyrows}
        own = [row for row in keyrows if self.good_for_shelf(row[0]) == slot]
        if not own and len(self.goods) == 1:
            own = list(keyrows)
        entries = [(key, many, len(seen)) for _shelf, key, many, seen in own]
        entries.sort(key=lambda entry: (-entry[1], mostly_numbers.get(entry[0], False), -entry[2]))
        settled = [_stem_set(need.split(':', 1)[0]) for need in self.goods[slot - 1].given_needs]
        usable = [key for key, _many, _distinct in entries if self._wq_askable_key(key, settled)]
        return self._wq_lumped_to_back(usable)

    @staticmethod
    def _wq_numeric_majority(values: list[str]) -> bool:
        numeric = [value for value in values if re.match('^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$', value)]
        return 2 * len(numeric) > len(values)

    @staticmethod
    def _wq_askable_key(key: str, settled: list[set[str]]) -> bool:
        if re.search(NOT_A_TRAIT_KEY_RX, key):
            return False
        stems = _stem_set(key)
        return not any(_same_trait_name(fixed, stems) for fixed in settled)

    @staticmethod
    def _wq_lumped_to_back(labels: list[str], view: Any = None) -> list[str]:
        """Stable split that moves labels whose text (seen through `view` when given) bundles several features to
        the end."""
        plain: list[str] = []
        lumped: list[str] = []
        for label in labels:
            text = view(label) if view else label
            (lumped if re.search(BUNDLE_KEY_RX, text) else plain).append(label)
        return plain + lumped

    # ======================================================== which requested item a category label stands for

    def good_for_shelf(self, shelf: str) -> int:
        """The 1-based item a listing's category label means, 0 if none. An adopted label stays with the item it was
        given to. Otherwise each item is scored: 3 for a label equal to its shelf, 2 for one equal to its product
        name, else the root words shared over the root words of both together. The earliest item with the top score
        wins, provided that score is above zero."""
        if not shelf:
            return 0
        if shelf in self.adopted_shelves:
            return self.adopted_shelves[shelf]
        plain, stems = _bare_words(shelf), _word_roots(shelf)
        best_slot, best = 0, 0.0
        for slot, good in enumerate(self.goods, 1):
            score = self._wq_shelf_closeness(plain, stems, good)
            if score > best:
                best_slot, best = slot, score
        return best_slot

    @staticmethod
    def _wq_shelf_closeness(plain: str, stems: set[str], good: RequestedGood) -> float:
        for name, exact in ((good.shelf_name or '', 3.0), (good.good_title, 2.0)):
            if plain == _bare_words(name):
                return exact
        pool = _word_roots(good.shelf_name or '') | _word_roots(good.good_title)
        return len(stems & pool) / max(1, len(stems | pool))

    def good_has_shelf(self, wanted: RequestedGood, shelf: str) -> bool:
        if not shelf or wanted not in self.goods:
            return False
        return self.good_for_shelf(shelf) == self.goods.index(wanted) + 1

    def adopt_loose_shelves(self) -> None:
        """With two or more items: the category labels on opened records that no item claims all go to the one item
        without a label of its own (labels it adopted earlier do not count), as long as exactly one such item
        exists."""
        label_key = self.shelf_field_name()
        if not label_key or len(self.goods) < 2:
            return
        labels = {str(page.get(label_key) or '') for page in self.page_records.values()} - {''}
        strays = [label for label in labels if not self.good_for_shelf(label)]
        if not strays:
            return
        unlabelled = [slot for slot in range(1, len(self.goods) + 1) if not self._wq_owns_label(slot, labels)]
        if len(unlabelled) != 1:
            return
        heir = unlabelled[0]
        self.adopted_shelves.update(dict.fromkeys(strays, heir))
        self.journal.append({'kind': 'adopted', 'item': heir, 'labels': strays})

    def _wq_owns_label(self, slot: int, labels: set[str]) -> bool:
        return any(label not in self.adopted_shelves and self.good_for_shelf(label) == slot for label in labels)

    def _wq_opened_with_traits(self, slot: int) -> int:
        """Fully opened records of the item's own categories that show an attribute block."""
        label_key = self.shelf_field_name()
        if not label_key:
            return 0
        good = self.goods[slot - 1]
        return sum(1 for page in self.page_records.values()
                   if self.good_has_shelf(good, str(page.get(label_key) or '')) and _trait_block(page))

    def _wq_clamped_item(self, raw: Any) -> int:
        """An item number read from a model answer and held inside 1..number of items (1 when it cannot be read)."""
        number = _coerce_int(raw) or 1
        return min(max(number, 1), len(self.goods))
class RunVerification:
    """Wren's safety net around the model loop. It reads the cart, runs the steps a request demands before any
    ending, folds store notes into the ledger, hands the model evidence and compatibility hints, checks the cart's
    lines and quantities, and asks for one last close review before money is spent."""

    # Opening words of the evidence block and of the fit plan the model reads (the guide explains the evidence lines).
    _RV_EVIDENCE_HEAD = ("[evidence: attribute keys naming the property of a requirement; a key on a different property "
                         "(some other part or aspect) proves nothing, yet a key with another name may still hold that "
                         "same property] ")
    _RV_FIT_PLAN_HEAD = ("[fit plan: a part suits a device when the part's own text names the device's model code or "
                         "another name the device is sold under, or when the part's own number falls in the part series "
                         "that device takes; a code is not searchable, so browse the device's category and read the "
                         "codes off its result rows] ")
    # A requirement value with one of these words is a bound, a negation or a borrowed value: the plain check skips it.
    _RV_PLAIN_HEDGE_RX = re.compile(r'\b(?:not|no|unlike|different|other than|least|most|under|over|below|above|between|within|up to|same|identical|partner)\b', re.I)

    # ================================================================== requirements still standing

    def _rv_live_needs_of(self, good: RequestedGood) -> list[str]:
        withdrawn = good.dropped_needs
        return [need for need in good.given_needs + good.told_needs if need not in withdrawn]

    def needs_in_force(self) -> list[tuple[int, str]]:
        """Every requirement that still stands, stated or answered and not taken back, as (item number, text) in
        item order."""
        standing: list[tuple[int, str]] = []
        for number, good in enumerate(self.goods, 1):
            standing.extend((number, need) for need in self._rv_live_needs_of(good))
        return standing

    def _rv_word_set(self, text: str) -> set[str]:
        spaced = re.sub('[,/;|]', ' ', _bare_words(text))
        return set(spaced.split())

    def _rv_spells_every_need(self, listing: str, slot: int) -> bool:
        """Whether the read attributes of `listing` repeat each requirement of item `slot` word for word. A value that
        carries a bound or a negation is not judged here, so such a requirement makes the answer False."""
        traits = _trait_block(self.page_records.get(listing) or {}) or {}
        qualifiers = {'not', 'no', 'least', 'most', 'under', 'over', 'unlike', 'different', 'min', 'max', 'minimum', 'maximum'}
        for owner, need in self.needs_in_force():
            if owner != slot:
                continue
            label, _colon, value = need.partition(':')
            words = self._rv_word_set(value)
            if not words or words & qualifiers:
                return False
            key_words = _canonical_key(label)
            if not any(_same_trait_name(key_words, _canonical_key(key)) and words <= self._rv_word_set(str(cell))
                       for key, cell in traits.items()):
                return False
        return True

    # ================================================================== the cart: viewed or counted

    def _rv_view_current(self) -> bool:
        """Only a cart view taken at or after the latest cart change is believed."""
        if self.basket_view is None:
            return False
        return self.basket_seen_at >= self.basket_changed_at

    def _rv_view_nodes(self, top: Any):
        """Yields each dict of a cart view, depth first; among sibling branches the last one is walked first."""
        pending = [top]
        while pending:
            node = pending.pop()
            if isinstance(node, dict):
                yield node
                pending += [child for child in node.values() if isinstance(child, (dict, list))]
            elif isinstance(node, list):
                pending += node

    def _rv_lone_count(self, node: dict[str, Any]) -> int | None:
        """The one whole number a cart node keeps under a quantity-like key, or None when there is not exactly one."""
        found: list[int] = []
        for key, cell in node.items():
            if isinstance(cell, str):
                continue
            number = _coerce_int(cell)
            if number is not None and re.search('\\b(?:qty|quantity|count|units?)\\b', _bare_words(key)):
                found.append(number)
        return found[0] if len(found) == 1 else None

    def _rv_units_held(self, listing: str) -> int:
        """How many units of a listing the cart holds. A current cart view answers when a node naming the listing has
        exactly one quantity-like number; failing that Wren's own count, which takes 1 for a listing it never sent."""
        if self._rv_view_current():
            for node in self._rv_view_nodes(self.basket_view):
                if not any(isinstance(cell, str) and cell.strip() == listing for cell in node.values()):
                    continue
                count = self._rv_lone_count(node)
                if count is not None:
                    return count
        tally = self.basket_counts
        return tally[listing] if listing in tally else 1

    def _rv_held_lines(self) -> list[str]:
        return [listing for listing in self.basket_ids() if (self._rv_units_held(listing) or 0) > 0]

    def basket_ids(self) -> list[str]:
        if not self._rv_view_current():
            return list(self.basket_counts)
        snapshot = _jdump(self.basket_view)
        return [known for known in self.row_records if known in snapshot]

    def basket_brief(self) -> str:
        if self._rv_view_current():
            return _jdump(_clip_texts(self.basket_view, 300))[:3000]
        return "Wren's count from the cart calls it sent itself (listing: units): " + _jdump(self.basket_counts)

    def listing_blurb(self, listing: str) -> str:
        record = self.page_records.get(listing) or self.row_records.get(listing)
        if not record:
            return '(nothing read about it yet)'
        return _fitted_dump(record, 400, 3000)

    def _rv_short_record(self, listing: str) -> str:
        return _fitted_dump(self.page_records.get(listing) or self.row_records.get(listing), 160, 900)

    def _rv_lowest_price_rule(self) -> bool:
        return self.digest.get('pick') == 'cheapest'

    def _rv_price_rank(self, listing: str) -> tuple[bool, float]:
        price = self.price_of_listing(listing)
        return (listing not in self.page_records, float('inf') if price is None else price)

    def _rv_other_seen_rows(self, exclude: list[str], cap: int) -> str:
        """Short records of seen listings that are neither excluded nor sold out, one per line and `cap` at most:
        newest first, or, under a lowest-price rule, opened ones before the rest and cheaper before dearer."""
        order = list(reversed(self.seen_sequence))
        if self._rv_lowest_price_rule():
            order.sort(key=self._rv_price_rank)
        kept = [cand for cand in order if cand not in exclude and not self.sold_out_now(cand)][:cap]
        return '\n'.join(self._rv_short_record(cand) for cand in kept)

    # ================================================================== free text and evidence in store replies

    @classmethod
    def _rv_phrases_in(cls, node: Any, depth: int = 0):
        """Strings of at least two words, no deeper than four levels."""
        if depth > 4:
            return
        if isinstance(node, dict):
            node = list(node.values())
        if isinstance(node, list):
            for child in node:
                yield from cls._rv_phrases_in(child, depth + 1)
        elif isinstance(node, str) and len(node.split()) >= 2:
            yield node

    def message_words(self, payload: Any) -> str:
        """What a reply says in words: a bare string unchanged; else its multi-word strings (four levels deep at
        most) run together, or the reply's JSON when it has none."""
        if not isinstance(payload, str):
            phrases = list(self._rv_phrases_in(payload))
            return ' '.join(phrases)[:800] if phrases else _jdump(payload)[:400]
        return payload

    def _rv_sentences_of_field(self, value: Any) -> list[str]:
        """Sentence-like text inside one top-level field: five words or more as a plain string or in a list made only
        of strings, six or more for the strings of a nested block that is not itself a listing."""
        if isinstance(value, list):
            if not value or not all(isinstance(entry, str) for entry in value):
                return []
            return [entry for entry in value if len(entry.split()) >= 5]
        if isinstance(value, dict) and not self.id_of_record(value, False, 1):
            return [inner for inner in value.values() if isinstance(inner, str) and len(inner.split()) >= 6]
        return [value] if isinstance(value, str) and len(value.split()) >= 5 else []

    def loose_text_lines(self, payload: Any) -> list[str]:
        if not isinstance(payload, dict) or self.id_of_record(payload):
            return []
        found = [sentence for value in payload.values() for sentence in self._rv_sentences_of_field(value)]
        return [sentence for sentence in found if sentence not in self.seen_notice_lines]

    def _rv_collect_trait_holders(self, node: Any, depth: int, found: list[str]) -> None:
        """Appends, in reading order, the ids of listings in a reply that bring an attribute block; a listing so
        taken is not searched any deeper."""
        if depth > 6:
            return
        children: list[Any] = []
        if isinstance(node, dict):
            ident = self.id_of_record(node, False, depth)
            if ident and _trait_block(node) and ident not in found:
                found.append(ident)
                return
            children = list(node.values())
        elif isinstance(node, list):
            children = node
        for child in children:
            self._rv_collect_trait_holders(child, depth + 1, found)

    def _rv_evidence_row(self, listing: str, needs: list[tuple[int, str]], numbered: bool) -> str:
        traits = _trait_block(self.page_records.get(listing) or {}) or {}
        pieces: list[str] = []
        for slot, need in needs:
            prop = need.split(':', 1)[0].strip()
            stems = _stem_set(prop)
            hits = [f'{key}={str(cell)[:50]}' for key, cell in traits.items() if _same_trait_name(stems, _stem_set(key))]
            if not hits:
                continue
            lead = f'(item {slot}) ' if numbered else ''
            pieces.append(f"{lead}{prop} -> {', '.join(hits[:3])}")
        return f"{listing}: {'; '.join(pieces)}" if pieces else ''

    def proof_lines(self, payload: Any) -> str:
        holders: list[str] = []
        self._rv_collect_trait_holders(payload, 0, holders)
        standing = self.needs_in_force()
        if not (holders and standing):
            return ''
        numbered = len(self.goods) > 1
        rows = [self._rv_evidence_row(listing, standing[:8], numbered) for listing in holders[:12]]
        rows = [row for row in rows if row]
        return self._RV_EVIDENCE_HEAD + ' | '.join(rows) if rows else ''

    # ================================================================== notes from the store or the shopper

    def interpret_shop_note(self, lines: list[str]) -> str:
        """A model reads one free-text note (from the store or from the shopper) and Wren applies it: listings that
        went, a new budget, requirements taken back or added, a price change. Returns a short account of the
        change, or '' when there was none."""
        self.seen_notice_lines.update(lines)
        prompt = '\n\n'.join([SHOP_NOTICE_PROMPT,
                              'CURRENT ITEMS AND REQUIREMENTS:\n' + self.ledger_lines(),
                              'NOTE TEXT:\n' + '\n'.join(lines)])
        reading = self.ask_model_json(prompt, [QUICK_SEAT, FALLBACK_SEAT], 2500) or {}
        gone = self._rv_mark_note_sellouts(reading, lines)
        if reading.get('kind') not in (None, 'change'):
            return ''
        return self._rv_fold_note_changes(reading, gone)

    def _rv_mark_note_sellouts(self, reading: dict[str, Any], lines: list[str]) -> list[str]:
        """Flags the listings a note calls unavailable and records whether it speaks of a sell-out at all."""
        gone = [str(entry) for entry in _listify(reading.get('gone')) if isinstance(entry, (str, int))]
        known = [entry for entry in gone if entry in self.row_records]
        self.unbuyable_ids.update(known)
        self.notice_says_sold = bool(gone) or reading.get('sold_out') is True
        self.notice_named_id = bool(known)
        lone_note = len(lines) == 1
        if lone_note and self.notice_says_sold:
            self.sellout_notice_lines.update(lines)
        return gone

    def _rv_take_note_budget(self, value: Any, edits: list[str]) -> bool:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        value = self.budget_in_unit(value)
        if not value > 0 or value == self.money_cap:
            return False
        self.money_cap = value
        edits.append(f'budget changed to {value}')
        return True

    def _rv_need_entries(self, reading: dict[str, Any], field: str) -> list[dict[str, Any]]:
        entries = []
        for entry in _listify(reading.get(field)):
            if isinstance(entry, dict) and str(entry.get('need') or '').strip():
                entries.append(entry)
        return entries

    def _rv_good_for_entry(self, entry: dict[str, Any]) -> RequestedGood:
        number = _coerce_int(entry.get('item')) or 1
        return self.goods[min(max(1, number), len(self.goods)) - 1]

    def _rv_withdraw_from_note(self, entry: dict[str, Any], edits: list[str]) -> None:
        good = self._rv_good_for_entry(entry)
        target = str(entry['need']).strip()
        standing = self._rv_live_needs_of(good)
        matches = [need for need in standing if need == target]
        if not matches:
            head = _bare_words(target.split(':')[0])
            first = next((need for need in standing if _bare_words(need.split(':')[0]) == head), None)
            matches = [] if first is None else [first]
        good.dropped_needs.extend(matches or [target])
        edits.append(f'dropped requirement: {target}')

    def _rv_add_from_note(self, entry: dict[str, Any], edits: list[str]) -> None:
        good = self._rv_good_for_entry(entry)
        fresh = str(entry['need']).strip()
        if fresh in good.dropped_needs:
            good.dropped_needs.remove(fresh)
        if fresh not in good.given_needs and fresh not in good.told_needs:
            good.told_needs.append(fresh)
        edits.append(f'new requirement: {fresh}')

    def _rv_fold_note_changes(self, reading: dict[str, Any], gone: list[str]) -> str:
        edits = []
        budget_moved = self._rv_take_note_budget(reading.get('budget'), edits)
        taken: dict[str, list[dict[str, Any]]] = {}
        for field, apply in (('drop', self._rv_withdraw_from_note), ('add', self._rv_add_from_note)):
            taken[field] = self._rv_need_entries(reading, field)
            for entry in taken[field]:
                apply(entry, edits)
        if gone:
            edits.append('no longer for sale: ' + ', '.join(gone))
        edits += ['a price changed'] if reading.get('price_moved') is True else []
        headline = str(reading.get('summary') or '').strip()
        if self._rv_lowest_price_rule() and (taken['drop'] or gone or self.notice_says_sold or budget_moved):
            # the lowest-price hunt starts over once the field of candidates changed
            self.wider_hunt_done = False
            self.lower_price_marks = set()
        if headline or edits:
            self.revision += 1
            self.recent_marks = []
            self.shift_notes.append(headline or '; '.join(edits))
        detail = f" ({'; '.join(edits)})" if edits else ''
        return (headline + detail).strip()

    # ================================================================== which part goes with which device

    def _rv_fit_wording(self, listing: str) -> str:
        record = self.page_records.get(listing) or self.row_records.get(listing) or {}
        flat: list[str] = []
        nested: list[str] = []
        for cell in record.values():
            if isinstance(cell, str):
                flat.append(cell)
            elif isinstance(cell, dict):
                nested += [f'{key} {text}' for key, text in cell.items() if isinstance(text, str)]
        return ' '.join(flat) + ' ' + '. '.join(nested)

    def fit_signal_index(self) -> dict[str, Any]:
        """For each seen listing the fit marks its own text gives ('marks'), and for each part number the extra devices
        other listings say that part fits ('notes')."""
        marks: dict[str, dict[str, Any]] = {}
        also_fits: dict[str, set[str]] = {}
        model_series: dict[str, set[str]] = {}
        for ident in self.seen_sequence:
            mark = _fit_signals(self._rv_fit_wording(ident))
            marks[ident] = mark
            for number, named in mark['also']:
                also_fits.setdefault(number, set()).update(named)
            for model, series in mark.get('model_also') or []:
                model_series.setdefault(model, set()).update(series)
        return {'marks': marks, 'notes': also_fits, 'model_series': model_series}

    def _rv_part_fit_codes(self, listing: str, index: dict[str, Any]) -> set[str]:
        """The device codes a part is good for: those it names itself, those other listings add for its part number,
        those of the part it refers the reader to for the rest, and the series its own number sits in."""
        mark = index['marks'].get(listing) or {}
        reach = set(mark.get('works') or [])
        own = mark.get('own')
        if own:
            reach |= index['notes'].get(own, set())
        pointed = mark.get('more')
        if pointed and pointed != own:
            for other in index['marks'].values():
                if pointed not in (other.get('own'), other.get('own_series_part')):
                    continue
                reach |= set(other.get('works') or []) | index['notes'].get(pointed, set())
        series = mark.get('in_series')
        if series:
            reach.add(_range_code(series))
        return reach

    def _rv_device_fit_codes(self, listing: str, index: dict[str, Any]) -> set[str]:
        """A device's model codes and alternative names together with the part series it accepts."""
        mark = index['marks'].get(listing) or {}
        takes = {_range_code(series) for series in mark.get('series') or []}
        extra = index.get('model_series') or {}
        takes |= {_range_code(series) for code in mark.get('codes') or [] for series in extra.get(code, ())}
        return set(mark.get('codes') or []) | takes

    def _rv_candidates_for(self, slot: int) -> list[str]:
        good = self.goods[slot - 1]
        label_key = self.shelf_field_name()
        if not label_key:
            return []
        return [ident for ident in self.seen_sequence if self.good_has_shelf(good, self._rv_shelf_of(ident, label_key))]

    def _rv_fitting_spares(self, part_pool: list[str], device_codes: set[str], carted: list[str], index: dict[str, Any]) -> list[str]:
        spares = [cand for cand in part_pool
                  if _code_sets_meet(self._rv_part_fit_codes(cand, index), device_codes) and not self.sold_out_now(cand)
                  and cand not in carted]
        return spares[:4]

    def _rv_misfits_for_pair(self, part_slot: int, device_slot: int, index: dict[str, Any], carted: list[str]) -> list[str]:
        device_pool = self._rv_candidates_for(device_slot)
        part_pool = self._rv_candidates_for(part_slot)
        carted_devices = [ident for ident in carted if ident in device_pool]
        carted_parts = [ident for ident in carted if ident in part_pool]
        if len(carted_devices) != 1 or not carted_parts:
            return []
        device, = carted_devices
        device_codes = self._rv_device_fit_codes(device, index)
        if not device_codes:
            return []
        kind = self.goods[part_slot - 1].good_title
        device_text = ', '.join(sorted(device_codes))
        found: list[str] = []
        for line in carted_parts:
            fits = self._rv_part_fit_codes(line, index)
            if not fits or _code_sets_meet(fits, device_codes):
                continue
            spares = self._rv_fitting_spares(part_pool, device_codes, carted, index)
            advice = (f"; seen parts of this kind that do fit it: {', '.join(spares)}" if spares else
                      '; no part of this kind seen so far fits it, so look for one or switch the device: ' + self.fit_route_text(True))
            found.append(f"{line} ({kind}) is listed for {', '.join(sorted(fits)[:4])}, which does not cover {device} ({device_text})" + advice)
        return found

    def fit_clashes(self) -> list[str]:
        if not self.fit_links:
            return []
        index = self.fit_signal_index()
        carted = self.basket_ids()
        found: list[str] = []
        for part_slot, device_slot in self.fit_links:
            found += self._rv_misfits_for_pair(part_slot, device_slot, index, carted)
        return found

    def _rv_plainly_meets(self, listing: str, slot: int) -> bool:
        """Whether the listing's read attributes spell out every requirement of the item that has no bound, negation or
        borrowed value (the full check judges those); an "or" in a value offers alternatives. False while nothing of
        its attributes was read."""
        traits = _trait_block(self.page_records.get(listing) or {})
        if not traits:
            return False
        borrowed = {need for _, need in self.borrowed_needs}
        for owner, need in self.needs_in_force():
            if owner != slot or need in borrowed:
                continue
            label, _colon, value = need.partition(':')
            if self._RV_PLAIN_HEDGE_RX.search(value):
                continue
            choices = [words for words in map(self._rv_word_set, re.split(r'\s+or\s+|,', value)) if words]
            if not choices:
                return False
            key_words = _canonical_key(label)
            if not any(_same_trait_name(key_words, _canonical_key(key)) and any(choice <= self._rv_word_set(str(cell)) for choice in choices)
                       for key, cell in traits.items()):
                return False
        return True

    _RV_LOWER_RX = re.compile(r'\b(?:at least|minimum|min|no less than|not less than|or more|over|above|more than)\b', re.I)
    _RV_UPPER_RX = re.compile(r'\b(?:at most|maximum|max|no more than|not more than|up to|under|below|less than|or less)\b', re.I)
    _RV_STRICT_RX = re.compile(r'\b(?:over|above|more than|under|below|less than)\b', re.I)
    _RV_EXCLUDE_RX = re.compile(r'^\s*(?:not|no|anything but|anything except|other than|except|never)\b\s*', re.I)
    _RV_UNLIKE_RX = re.compile(r'\b(?:unlike|different from|differs? from|other than|not the same|anything but|except)\b', re.I)

    def _rv_figures(self, text: str) -> list[tuple[float, str]]:
        """(number, unit letters right after it, '' when none) for each figure in a text."""
        return [(float(number), unit.lower()) for number, unit in
                re.findall(r'(\d+(?:\.\d+)?)\s*(?!(?:and|or|to)\b)([a-zA-Z]{1,5}(?![a-zA-Z]))?', str(text).replace(',', ''))]

    def _rv_bound_holds(self, cells: list[str], value: str) -> bool:
        """Whether a figure under a key named for the property meets a measured bound ("at least 65 W", "between 10 and
        20 cm"): same unit letters on both (or none on both); every bound the value states is checked."""
        wanted = self._rv_figures(value)
        if not wanted:
            return False
        unit = next((letters for _, letters in wanted if letters), '')
        between = re.search(r'\bbetween\b', value, re.I) and len(wanted) >= 2
        low = wanted[0][0] if between or self._RV_LOWER_RX.search(value) else None
        high = wanted[1][0] if between else (wanted[0][0] if self._RV_UPPER_RX.search(value) else None)
        if low is None and high is None:
            return False
        strict = bool(self._RV_STRICT_RX.search(value)) and not between
        for cell in cells:
            for number, letters in self._rv_figures(cell):
                if letters != unit and not (letters.rstrip('s') == unit.rstrip('s') and unit):
                    continue
                if low is not None and (number < low or (strict and number == low)):
                    continue
                if high is not None and (number > high or (strict and number == high)):
                    continue
                return True
        return False

    def _rv_proves(self, listing: str, slot: int) -> bool:
        """A strict reading of whether a read listing meets every requirement of item `slot` in force, the way grading
        reads pages: each requirement under an attribute whose key names its property - a value as one run of words
        in one part of that attribute, a measured bound by a figure in the same unit, an exclusion by a named
        attribute free of the excluded words, a borrowed value only once its chain gave it. Anything else (a
        condition, an unresolved pointer, a property no key names) leaves the listing unproven. A listing must also
        be buyable, priced, within the shopper's cap for that item, and its title or attributes must hold the words of
        the item's name that the store label does not."""
        record = self.page_records.get(listing) or {}
        traits = _trait_block(record)
        if not traits or not self.is_purchasable(listing) or self.sold_out_now(listing):
            return False
        price = self.price_of_listing(listing)
        cap = self.item_caps.get(slot)
        if price is None or (cap is not None and price > cap):
            return False
        title = str((self.row_records.get(listing) or {}).get('title') or record.get('title') or '')
        label_key = self.shelf_field_name()
        shelf = self._shelf_text(listing, label_key) if label_key else ''
        spare = _word_roots(self.goods[slot - 1].good_title) - _word_roots(shelf)
        if spare and not spare <= _word_roots(title + ' ' + ' '.join(str(v) for v in traits.values())):
            return False
        kind = {_plural_root(token) for token in _bare_words(self.goods[slot - 1].good_title + ' ' + shelf).split()}
        self._rv_cells_beside = {key: value for key, value in traits.items() if not isinstance(value, (dict, list))}
        borrowed = {need for _, need in self.borrowed_needs}
        for owner, need in self.needs_in_force():
            if owner != slot:
                continue
            label, _colon, value = need.partition(':')
            if not value.strip():
                return False
            named = [key for key in traits if _same_trait_name(_canonical_key(label), _canonical_key(key))]
            if not named:
                return False
            if need in borrowed:
                taken = self.borrowed_values.get(need)
                if not taken or str(taken).startswith('as stated on'):
                    return False
                if self._RV_UNLIKE_RX.search(value):
                    if any(self._rv_names_value(key, traits[key], label, str(taken), title, kind) for key in named):
                        return False
                    continue
                value = str(taken)
            if self._RV_LOWER_RX.search(value) or self._RV_UPPER_RX.search(value) or re.search(r'\bbetween\b', value, re.I):
                if not self._rv_bound_holds([str(traits[key]) for key in named], value):
                    return False
                continue
            excluded = self._RV_EXCLUDE_RX.match(value)
            if excluded:
                rest = value[excluded.end():]
                if any(self._rv_names_value(key, traits[key], label, part, title, kind)
                       for key in named for part in re.split(r'\s+or\s+|,', rest) if part.strip()):
                    return False
                continue
            if self._RV_PLAIN_HEDGE_RX.search(value):
                return False
            if not any(self._rv_names_value(key, traits[key], label, part, title, kind)
                       for key in named for part in re.split(r'\s+or\s+|,', value) if part.strip()):
                return False
        return True

    def proven_set(self) -> dict[str, int] | None:
        """The cheapest whole order made only of read listings that `_rv_proves` passes, one line per item at its
        quantity, inside the budget; None when some item has none, two items share a store label, items must fit
        each other (code fits are judged elsewhere), or a quantity or price is unknown."""
        label_key = self.shelf_field_name()
        if not label_key or self.fit_links or not self.goods:
            return None
        chosen: dict[int, str] = {}
        for slot in range(1, len(self.goods) + 1):
            proven = [lid for lid in self.page_records
                      if self.good_for_shelf(self._shelf_text(lid, label_key)) == slot and self._rv_proves(lid, slot)]
            if not proven:
                return None
            chosen[slot] = min(proven, key=lambda lid: (self.price_of_listing(lid) or 0.0, lid))
        if len(set(chosen.values())) != len(chosen):
            return None
        plan = {lid: max(1, int(self.goods[slot - 1].count_wanted or 1)) for slot, lid in chosen.items()}
        total = sum((self.price_of_listing(lid) or 0.0) * units for lid, units in plan.items())
        cap = self.money_cap if self._is_number(self.money_cap) else None
        scope = self.digest.get('budget_scope')
        if cap is not None and scope != 'each' and total > cap + 1e-09:
            return None
        if cap is not None and scope == 'each' and any((self.price_of_listing(lid) or 0.0) > cap + 1e-09 for lid in plan):
            return None
        return plan

    def _rv_unshown_needs(self, listing: str, slot: int) -> list[str]:
        """The item's plain requirements (no bound, negation or borrowed value) that the listing's read attributes do
        not show: an attribute named for the property gives another value, or no attribute names it and its value
        words appear in none of the attributes. Empty while nothing of its attributes was read."""
        traits = _trait_block(self.page_records.get(listing) or {})
        if not traits:
            return []
        borrowed = {need for _, need in self.borrowed_needs}
        every_cell = [self._rv_word_set(str(cell)) for cell in traits.values()]
        label_key = self.shelf_field_name()
        shelf = self._shelf_text(listing, label_key) if label_key else ''
        kind = {_plural_root(token) for token in _bare_words(self.goods[slot - 1].good_title + ' ' + shelf).split()} if 0 < slot <= len(self.goods) else set()
        self._rv_cells_beside = {key: value for key, value in traits.items() if not isinstance(value, (dict, list))}
        missing: list[str] = []
        for owner, need in self.needs_in_force():
            if owner != slot or need in borrowed:
                continue
            label, _colon, value = need.partition(':')
            unit = self._rv_measure_unit(value)
            if unit is not None:
                if not self._rv_states_measure(traits, label, unit) or self._rv_bare_figure_off(traits, label, value):
                    missing.append(need)
                continue
            if self._RV_PLAIN_HEDGE_RX.search(value):
                continue
            choices = [words for words in map(self._rv_word_set, re.split(r'\s+or\s+|,', value)) if words]
            if not choices:
                continue
            key_words = _canonical_key(label)
            named_keys = [key for key in traits if _same_trait_name(key_words, _canonical_key(key))]
            named = [traits[key] for key in named_keys]
            title = str((self.row_records.get(listing) or {}).get('title') or '')
            if any(self._rv_names_value(key, traits[key], label, part, title, kind)
                   for key in named_keys for part in re.split(r'\s+or\s+|,', value) if part.strip()):
                continue
            if not named and any(choice <= words for words in every_cell + [self._rv_word_set(title)] for choice in choices):
                continue
            missing.append(need)
        return missing

    @staticmethod
    def _rv_value_tokens(text: str, kind: Any) -> list[str]:
        """A value's words as compared: plural folded, without words of the item's own name or shelf (a kind word the
        brief kept in the value, "fluorocarbon line" for a fishing line) and without short joining words; a figure
        always stays."""
        tokens = [_plural_root(token) for token in _bare_words(text).split()]
        return [token for token in tokens if re.search(r'\d', token)
                or (len(token) >= 3 and token not in kind and token not in ('and', 'for', 'the', 'with'))]

    def _rv_names_value(self, key: str, cell: Any, label: str, wanted: str, title: str, kind: Any = frozenset()) -> bool:
        """Whether an attribute named for the property states the wanted value: its words (kind and joining words
        aside) stand together, in order, in one comma/slash part of the attribute (words of the value split by other
        words, such as "10k white gold" for "10k gold", are another value); and a list under the plural of the
        property holds the variants on offer, which state this listing's value only when its title does not name
        another of them instead."""
        want = self._rv_value_tokens(wanted, kind) or [_plural_root(token) for token in _bare_words(wanted).split()]
        if not want:
            return False
        raw_parts = [part for part in re.split(r'[,/;|]', str(cell)) if part.strip()]
        parts = [self._rv_value_tokens(part, kind) or [_plural_root(token) for token in _bare_words(part).split()]
                 for part in raw_parts]
        hits = [index for index, part in enumerate(parts)
                if any(part[start:start + len(want)] == want for start in range(len(part) - len(want) + 1))]
        if not hits:
            return False
        if len(parts) > 1:
            # a list holds the variants on offer: when the title or another attribute names another of them and none
            # names the wanted one, the listing sells that other variant
            elsewhere = [title] + [str(text) for name, text in (self._rv_cells_beside or {}).items() if name != key]
            seen_words = [set(self._rv_value_tokens(text, kind)) for text in elsewhere if text]
            others = [part for index, part in enumerate(parts) if index not in hits]
            named_other = any(part and set(part) <= words for part in others for words in seen_words)
            named_wanted = any(set(parts[index]) <= words for index in hits for words in seen_words)
            if named_other and not named_wanted:
                return False
        return True

    _RV_BOUND_RX = re.compile(r'\b(?:at least|at most|minimum|maximum|min|max|under|over|below|above|between|'
                              r'within|up to|no more than|no less than|less than|more than|or more|or less)\b', re.I)

    def _rv_measure_unit(self, value: str) -> str | None:
        """For a measured limit (a bound word, or a decimal figure, with a number in the value): the unit written
        right after the number ('' when none); None for any other value, such as a count or a word."""
        figure = re.search(r'(\d+(?:[.,]\d+)?)\s*([a-z"\'%]+)?', value.lower())
        if not figure or not (self._RV_BOUND_RX.search(value) or re.search(r'\d\.\d', value)):
            return None
        unit = figure.group(2) or ''
        return unit if re.fullmatch('[a-z]+', unit) else ''

    def _rv_bare_figure_off(self, traits: dict[str, Any], label: str, value: str) -> bool:
        """Whether the only figures stated for a measured limit, under keys sharing a word with the property, are bare
        numbers (no unit) at least ten times away from the limit's own figure ("400" for "at most 1.188 kg"): such a
        figure measures something else, or in another unit, and leaves the limit unproven."""
        figure = re.search(r'\d+(?:\.\d+)?', value.replace(',', ''))
        if not figure or float(figure.group()) <= 0:
            return False
        limit = float(figure.group())
        roots = set(_canonical_key(label))
        cells = [str(cell).strip().lower() for key, cell in traits.items() if roots & set(_canonical_key(key))]
        cells = [cell for cell in cells if re.search(r'\d', cell)]
        if not cells or any(re.search('[a-z]', cell) for cell in cells):
            return False
        numbers = [float(n) for cell in cells for n in re.findall(r'\d+(?:\.\d+)?', cell.replace(',', ''))]
        return bool(numbers) and all(n <= 0 or max(n / limit, limit / n) >= 10 for n in numbers)

    def _rv_states_measure(self, traits: dict[str, Any], label: str, unit: str) -> bool:
        """Whether an attribute whose key shares a word with the property gives a figure (in the limit's unit when
        both carry letters): a figure under a key for another measure, or only in the title, leaves the limit
        unknown to a reader who goes by the page's attributes."""
        roots = set(_canonical_key(label))
        for key, cell in traits.items():
            text = str(cell).lower()
            if not roots & set(_canonical_key(key)) or not re.search(r'\d', text):
                continue
            if not unit or not re.search('[a-z]', text) or re.search(r'\d\s*' + re.escape(unit), text):
                return True
        return False

    def _rv_code_tallies(self, device_slot: int, index: dict[str, Any]) -> tuple[dict[str, dict[int, list[str]]], dict[str, dict[int, list[str]]]]:
        """Per fit code and part item: the buyable candidate parts naming the code, and those of them whose read
        attributes already show their plain requirements."""
        named: dict[str, dict[int, list[str]]] = {}
        backed: dict[str, dict[int, list[str]]] = {}
        for part_slot in (part for part, device in self.fit_links if device == device_slot):
            for ident in self._rv_candidates_for(part_slot):
                if self.sold_out_now(ident):
                    continue
                meets = self._rv_plainly_meets(ident, part_slot)
                for code in self._rv_part_fit_codes(ident, index):
                    if code.startswith('!'):
                        continue
                    named.setdefault(code, {}).setdefault(part_slot, []).append(ident)
                    if meets:
                        backed.setdefault(code, {}).setdefault(part_slot, []).append(ident)
        return named, backed

    def _rv_plan_piece(self, code: str, by_part: dict[int, list[str]], backed_by_part: dict[int, list[str]], device_pool: list[str], device_slot: int, index: dict[str, Any]) -> str:
        hosts = [ident for ident in device_pool if _code_sets_meet({code}, self._rv_device_fit_codes(ident, index))]
        hosts.sort(key=lambda ident: not self._rv_plainly_meets(ident, device_slot))
        counts: list[str] = []
        for part_slot, found in sorted(by_part.items()):
            good_ones = backed_by_part.get(part_slot, [])
            text = f'{len(found)} {self.goods[part_slot - 1].good_title}'
            if good_ones:
                text += f" ({len(good_ones)} with their requirements shown met: {', '.join(good_ones[:2])})"
            counts.append(text)
        shown = [ident + (' (its requirements shown met)' if self._rv_plainly_meets(ident, device_slot) else '') for ident in hosts[:2]]
        where = f" (devices seen carrying it: {', '.join(shown)})" if shown else ' (no device with this code seen yet)'
        return f"{code}: {', '.join(counts)}{where}"

    def _rv_plan_for_device(self, device_slot: int, index: dict[str, Any]) -> str:
        """One device item's line of the plan: the codes its candidate parts name, best first by how many part kinds
        have a part whose read attributes already show its plain requirements, then by how many part kinds and how
        many parts name the code; for each code the parts behind it and the seen devices carrying it, the ones that
        show their requirements met first."""
        named, backed = self._rv_code_tallies(device_slot, index)
        if not named:
            return ''

        def strength(code: str) -> tuple[int, int, int]:
            return (-len(backed.get(code, {})), -len(named[code]), -sum(map(len, named[code].values())))
        device_pool = self._rv_candidates_for(device_slot)
        pieces = [self._rv_plan_piece(code, named[code], backed.get(code, {}), device_pool, device_slot, index)
                  for code in sorted(named, key=strength)[:4]]
        return f'fit codes named by the candidate parts for the {self.goods[device_slot - 1].good_title} -> ' + '; '.join(pieces)

    def fit_route_text(self, brief: bool = False) -> str:
        if not self.fit_links:
            return ''
        index = self.fit_signal_index()
        device_slots = sorted({device for _, device in self.fit_links})
        rows = [row for row in (self._rv_plan_for_device(slot, index) for slot in device_slots) if row]
        plan = ' | '.join(rows)
        return plan if brief or not rows else self._RV_FIT_PLAN_HEAD + plan

    # ================================================================== one cart line per requested good

    def _rv_shelf_of(self, listing: str, label_key: str) -> str:
        row = self.row_records.get(listing) or {}
        return str(row.get(label_key) or '')

    def _rv_shelf_labels(self, label_key: str) -> list[str]:
        return [str(row.get(label_key) or '') for row in self.row_records.values()]

    def _rv_goods_on_shelf(self, label: str) -> list[int]:
        return [slot for slot, good in enumerate(self.goods, 1) if self.good_has_shelf(good, label)]

    def _rv_good_on_shelves(self, good: RequestedGood, labels: list[str]) -> bool:
        return any(self.good_has_shelf(good, label) for label in labels)

    def _rv_budget_each(self) -> bool:
        return self.digest.get('budget_scope') == 'each'

    def _goods_without_line(self) -> list[int]:
        """Items the cart covers with no line, looked for only while it has fewer lines than there are items; an item
        whose label shows on none of the seen listings is never reported."""
        label_key = self.shelf_field_name()
        carted = self._rv_held_lines()
        if not label_key or len(carted) >= len(self.goods):
            return []
        every_label = self._rv_shelf_labels(label_key)
        carted_labels = [self._rv_shelf_of(ident, label_key) for ident in carted]
        return [slot for slot, good in enumerate(self.goods, 1)
                if self._rv_good_on_shelves(good, every_label) and not self._rv_good_on_shelves(good, carted_labels)]

    def _rv_cheapest_filler(self, slot: int, good: RequestedGood, tied: set[int]) -> str | None:
        """The lowest-priced opened, buyable listing for the item whose attributes state all its requirements, as long
        as it keeps the cart inside the budget; never one for an item bound to another by a fit rule."""
        if slot in tied:
            return None
        able = [cand for cand in self._rv_candidates_for(slot)
                if cand in self.page_records and cand not in self.basket_counts and self.is_purchasable(cand)
                and self.price_of_listing(cand) is not None and self._rv_spells_every_need(cand, slot)]
        if not able:
            return None
        pick = sorted(able, key=lambda cand: self.price_of_listing(cand) or 0.0)[0]
        limit = self.money_cap
        if not pick or not isinstance(limit, (int, float)):
            return pick
        each_alone = self._rv_budget_each()
        spent = 0.0 if each_alone else sum(((self.price_of_listing(cand) or 0.0) * (self._rv_units_held(cand) or 0) for cand in self.basket_ids()))
        adds = (self.price_of_listing(pick) or 0.0) * (1 if each_alone else good.count_wanted)
        return None if adds + spent > limit + 1e-09 else pick

    def _rv_tallied_lines(self) -> dict[str, int]:
        counted = self.basket_counts
        return {ident: counted[ident] for ident in self.basket_ids() if ident in counted}

    def _rv_send_cart_edit(self, cart_tool: str, raw: dict[str, Any]) -> bool:
        issues: list[str] = []
        fitted = _fit_to_schema(raw, self.schemas.get(cart_tool), 'args', issues)
        if issues or self.shop_call(cart_tool, fitted) is None:
            return False
        self.note_basket_call(cart_tool, fitted)
        return True

    def _rv_cover_missing_good(self, number: int, cart_tool: str, tied: set[int]) -> bool:
        good = self.goods[number - 1]
        pick = self._rv_cheapest_filler(number, good, tied)
        if not pick:
            return False
        calls = self.basket_edit_calls({**self._rv_tallied_lines(), pick: good.count_wanted}, [])
        if not calls or len(calls) + 1 > self.moves_left():
            return False
        if not all(self._rv_send_cart_edit(cart_tool, raw) for raw in calls):
            return False
        self.journal.append({'kind': 'filled_line', 'item': number, 'line': pick})
        return True

    def top_up_basket(self) -> bool:
        """Ahead of a forced order, puts a line in the cart for each item still without one: the cheapest opened
        listing whose attributes state every requirement of that item. Gives up (False) at the first item it cannot
        cover, so the episode closes with a refusal rather than a partial order."""
        cart_tool = self.role_map.get('cart') or ''
        tied = {number for link in self.fit_links for number in link}
        for number in self._goods_without_line():
            if not self._rv_cover_missing_good(number, cart_tool, tied):
                return False
        return True

    def _rv_fuller_label(self, slot: int, label: str) -> str:
        """A store label naming item `slot` completely, offered only when the line's own label carries just part of
        the item's name."""
        label_key = self.shelf_field_name()
        good = self.goods[slot - 1]
        name_roots = _word_roots(good.good_title)
        if len(name_roots) < 2 or not label_key:
            return ''
        label_roots = _word_roots(label)
        if name_roots <= label_roots:
            return ''
        if good.shelf_name and label_roots == _word_roots(good.shelf_name):
            return ''
        for shelf in sorted({str(row.get(label_key) or '') for row in self.row_records.values()}):
            if shelf and name_roots <= _word_roots(shelf):
                return shelf
        return ''

    def _rv_good_line_faults(self, slot: int, good: RequestedGood, lines: list[str], shelf: dict[str, str], owners: dict[str, list[int]]) -> list[str]:
        tag = f'item {slot} ({good.good_title})'
        own_lines = [ident for ident in lines if slot in owners[ident]]
        if not own_lines:
            return [f'{tag} has nothing in the cart so far']
        faults: list[str] = []
        if sum(1 for ident in own_lines if owners[ident] == [slot]) > 1:
            faults.append(f"{tag} is covered by {len(own_lines)} lines ({', '.join(own_lines)}); keep just one")
        for ident in own_lines:
            units = self._rv_units_held(ident)
            if owners[ident] == [slot] and units != good.count_wanted:
                # the turn loop spots a pure quantity fault by the words ' is held at '
                faults.append(f'{ident} is held at {units} unit(s), but {tag} needs {good.count_wanted}')
            fuller = self._rv_fuller_label(slot, shelf[ident])
            if fuller:
                faults.append(f'{ident} sits under {shelf[ident]}, but the store also has the label {fuller}, which names all of {tag}: take the line from that label instead')
        return faults

    def _rv_budget_faults(self, lines: list[str]) -> list[str]:
        limit = self.money_cap
        priced = {ident: self.price_of_listing(ident) for ident in lines}
        if not isinstance(limit, (int, float)) or any(price is None for price in priced.values()):
            return []
        if self._rv_budget_each():
            return [f'{ident} costs {priced[ident]:g}, over the per-item limit of {limit:g}' for ident in lines if limit < priced[ident] <= 10 * limit]
        total = sum((priced[ident] * self._rv_units_held(ident) for ident in lines))
        return [f'the lines total {total:g}, over the budget of {limit:g}'] if limit + 1e-09 < total <= 10 * limit else []

    def basket_line_faults(self) -> list[str]:
        self.adopt_loose_shelves()
        label_key = self.shelf_field_name()
        lines = self._rv_held_lines()
        if not label_key or not lines:
            return []
        shelf = {ident: self._rv_shelf_of(ident, label_key) for ident in lines}
        every_label = self._rv_shelf_labels(label_key)
        known_goods = {slot for slot, good in enumerate(self.goods, 1) if self._rv_good_on_shelves(good, every_label)}
        owners = {ident: self._rv_goods_on_shelf(shelf[ident]) for ident in lines}
        faults: list[str] = []
        for slot, good in enumerate(self.goods, 1):
            if slot in known_goods:
                faults += self._rv_good_line_faults(slot, good, lines, shelf, owners)
        if len(known_goods) == len(self.goods):
            faults += [f"{ident} ({shelf[ident] or 'unknown kind'}) is not a product the shopper asked for: take it out"
                       for ident in lines if not owners[ident]]
        faults += [f'{ident} is sold out and cannot be bought any more' for ident in lines if self.sold_out_now(ident)]
        return faults + self._rv_budget_faults(lines)

    # ================================================================== quantity fixes Wren sends on its own

    def count_repairs(self) -> list[tuple[str, int, int]]:
        label_key = self.shelf_field_name()
        carted = self.basket_ids() if label_key else []
        fixes: list[tuple[str, int, int]] = []
        for ident in carted:
            units = self._rv_units_held(ident) or 0
            owners = self._rv_goods_on_shelf(self._rv_shelf_of(ident, label_key))
            if units <= 0 or len(owners) != 1 or self.sold_out_now(ident):
                continue
            good = self.goods[owners[0] - 1]
            peers = [other for other in carted
                     if (self.basket_counts.get(other, 1) or 0) > 0 and self.good_has_shelf(good, self._rv_shelf_of(other, label_key))]
            if units != good.count_wanted and len(peers) == 1:
                fixes.append((ident, units, good.count_wanted))
        return fixes

    def _rv_first_param_where(self, fields: dict[str, Any], test) -> str | None:
        return next((key for key, spec in fields.items() if isinstance(spec, dict) and test(spec)), None)

    def _rv_cart_count_params(self, cart_tool: str, schema: dict[str, Any]) -> tuple[str, str, str | None] | None:
        """Names of the cart tool's id, count and mode parameters; None when it lacks an id or a count."""
        fields = schema.get('properties') or {}
        text_keys = [key for key, spec in fields.items()
                     if isinstance(spec, dict) and 'string' in _declared_types(spec) and 'enum' not in spec]
        id_key = (next((key for owner, key in self.id_args if owner == cart_tool and key in text_keys), None)
                  or next((key for key in schema.get('required') or [] if key in text_keys), None)
                  or (text_keys[0] if len(text_keys) == 1 else None))
        count_key = self._rv_first_param_where(fields, lambda spec: 'integer' in _declared_types(spec))
        mode_key = self._rv_first_param_where(fields, lambda spec: isinstance(spec.get('enum'), list) and len(spec['enum']) >= 2)
        return (id_key, count_key, mode_key) if id_key and count_key else None

    def _rv_count_fix_calls(self, cart_tool: str, fixes: list[tuple[str, int, int]], schema: dict[str, Any], params: tuple[str, str, str | None]) -> list[dict[str, Any]] | None:
        """Cart calls bringing each listed line to its wanted count: a single call when the tool sets counts outright,
        otherwise taking off the units held and putting on the wanted ones; None if the tool allows neither."""
        id_key, count_key, mode_key = params
        calls: list[dict[str, Any]] = []
        for ident, units, wanted in fixes:
            if self.basket_sets_count(cart_tool):
                calls.append({id_key: ident, count_key: wanted})
                continue
            if not mode_key:
                return None
            options = [str(option) for option in schema['properties'][mode_key]['enum']]
            removal = next((option for option in options if re.search('remov|delet|drop|clear|take', option, re.I)), options[1])
            adding = next((option for option in options if option != removal), options[0])
            calls += [{id_key: ident, mode_key: removal, count_key: units},
                      {id_key: ident, mode_key: adding, count_key: wanted}]
        return calls

    def _rv_push_count_fixes(self, cart_tool: str, schema: dict[str, Any], calls: list[dict[str, Any]]) -> list[str]:
        done: list[str] = []
        for raw in calls:
            issues: list[str] = []
            fitted = _fit_to_schema(raw, schema, 'args', issues)
            if issues:
                break
            reply = self.shop_call(cart_tool, fitted)
            if reply is None or self.run_closed:
                break
            self.note_basket_call(cart_tool, fitted)
            observation = reply.get('observation')
            self.last_basket_reply = _jdump(observation)
            self.digest_reply(cart_tool, observation)
            done.append(_jdump(fitted) + ' -> ' + self.show_reply(observation)[:300])
        return done

    def fix_line_counts(self, corrections: list[tuple[str, int, int]]) -> list[str]:
        cart_tool = self.role_map.get('cart') or ''
        schema = self.schemas.get(cart_tool) or {}
        params = self._rv_cart_count_params(cart_tool, schema)
        if not cart_tool or params is None:
            return []
        calls = self._rv_count_fix_calls(cart_tool, corrections, schema, params)
        if calls is None or len(calls) + 3 > self.moves_left():
            return []
        done = self._rv_push_count_fixes(cart_tool, schema, calls)
        self.journal.append({'kind': 'count_fix', 'calls': done})
        return done

    # ================================================================== steps the request wants before an ending

    def _rv_gate_tool_lines(self, closers: set[str]) -> list[str]:
        lines: list[str] = []
        for spec in self.tool_list:
            body = spec['function']
            if body['name'] in closers:
                continue
            lines.append(f"- {body['name']}: {body.get('description', '')} | parameters: {_jdump(body.get('parameters') or {})}")
        return lines

    def _rv_gate_prompt(self, moment: str, due: list[dict[str, Any]], tool_lines: list[str]) -> str:
        buying = moment == 'order'
        record = '\n'.join(f'{number}. {line}' for number, line in enumerate(self.act_log, 1)) or '(nothing yet)'
        sections = [PROCEDURE_GATE_PROMPT.replace('STAGE', 'submit the order' if buying else 'stop without buying anything'),
                    'REQUEST:\n' + self.shopper_request,
                    'STEPS DUE AT THIS POINT:\n' + '\n'.join(f"- {entry['step']}" for entry in due),
                    'LEDGER:\n' + self.ledger_lines(),
                    'CART NOW: ' + self.basket_brief(),
                    'RECORD (each message and store step so far, numbered):\n' + record]
        recent_notes = self.shift_notes[-4:]
        if recent_notes:
            sections.append('STORE NOTES SO FAR:\n' + '\n'.join(recent_notes))
        closing = 'order' if buying else 'ending'
        sections.append(f'MESSAGES LEFT: {max(0, self.message_cap - self.messages_out)}   STEPS LEFT: {self.moves_left()} (the {closing} itself takes one of them)')
        sections.append('TOOLS:\n' + '\n'.join(tool_lines))
        return '\n\n'.join(sections)

    def _rv_gate_reply_changes(self, ruling: dict[str, Any], done_lines: list[str]) -> tuple[list[str], str]:
        """New requirements and holds from a shopper's reply; looked at only after Wren acted inside this gate."""
        if not done_lines:
            return [], ''
        news: list[str] = []
        self.fold_in_answers({'needs': ruling.get('needs')}, news)
        return news, str(ruling.get('hold') or '').strip()

    def _rv_gate_open_steps(self, ruling: dict[str, Any], talk_needed: bool) -> list[dict[str, Any]]:
        """The steps the verdict leaves unproven. If the order needs a message, only RECORD lines written after the
        latest cart change can prove a step."""
        total = len(self.act_log)
        valid = set(range(1, total + 1))
        if talk_needed:
            valid = {n for n in valid if n <= len(self.act_steps) and self.act_steps[n - 1] > self.basket_changed_at}

        def shown_done(entry: dict[str, Any]) -> bool:
            if entry.get('done') is not True:
                return False
            return any((_coerce_int(n) or 0) in valid for n in _listify(entry.get('lines')))
        return [entry for entry in _listify(ruling.get('steps')) if isinstance(entry, dict) and not shown_done(entry)]

    def _rv_gate_can_message(self) -> bool:
        if not self.role_map.get('ask') or self.messages_out >= self.message_cap:
            return False
        return self.moves_left() >= 2

    def _rv_gate_push_back(self, unproven: list[dict[str, Any]]) -> None:
        self.journal.append({'kind': 'gate_unproven', 'steps': _clip_texts(unproven, 120)})
        named = '; '.join(str(entry.get('step') or '') for entry in unproven)[:200]
        left_messages = self.message_cap - self.messages_out
        self.act_log.append(f'(Wren: no record line written since the cart last changed (step {self.basket_changed_at}) proves this done: {named}; '
                            f'{left_messages} message(s) and {self.moves_left()} steps are still available, so name the call that does it)')
        self.act_steps.append(-1)

    def _rv_gate_send_message(self, args: dict[str, Any]) -> str | None:
        wording = next((cell for cell in args.values() if isinstance(cell, str)), '')
        replies = self.deliver_messages([(1, '', wording)], 1)
        if not replies:
            return None
        self.procedure_messages += 1
        return f'Sent: {wording}\nAnswer: {replies[0][2]}'

    def _rv_gate_store_action(self, name: str, args: dict[str, Any]) -> str | None:
        reply = self.shop_call(name, args)
        if reply is None:
            return None
        observation = reply.get('observation')
        self.ingest_records(observation)
        shown = _jdump(args)
        rendered = self.show_reply(observation)
        self.act_log.append(f'step {self.moves_made}: {name} {shown} -> {rendered[:300]} (cart then: {self.basket_fingerprint()})')
        self.act_steps.append(self.moves_made)
        return f'{name} {shown} -> {rendered[:400]}'

    def _rv_gate_carry_out(self, call: dict[str, Any], closers: set[str]) -> str | None:
        """Runs the call the gate model asked for and returns the line to report; None ends the gate."""
        name = str(call.get('tool') or '')
        messenger = self.role_map.get('ask')
        if name not in self.schemas or name in closers or self.moves_left() < 2:
            return None
        if name == messenger and self.messages_out >= self.message_cap:
            return None
        issues: list[str] = []
        raw = call.get('args')
        fitted = _fit_to_schema(raw if isinstance(raw, dict) else {}, self.schemas[name], 'args', issues)
        if issues:
            return None
        return self._rv_gate_send_message(fitted) if name == messenger else self._rv_gate_store_action(name, fitted)

    def _rv_gate_result(self, done_lines: list[str]) -> tuple[bool, str]:
        if not done_lines:
            return True, ''
        return True, 'Before going on, Wren carried out the steps the request asks for at this point:\n' + '\n'.join(done_lines)

    def run_procedure_gate(self, moment: str) -> tuple[bool, str]:
        """Carries out the request's own steps that fall due just before an order (moment 'order') or an ending with
        no purchase ('end'). For up to PROCEDURE_GATE_ROUNDS rounds a model reads the numbered record, says which
        steps are done and names the call for one still open. Returns (False, why) when a shopper's reply changes
        the task, otherwise (True, account of what was done)."""
        due = [entry for entry in self.procedure_steps if entry['when'] == moment]
        done_lines: list[str] = []
        if self.run_closed or not due or not self.procedure_check_due(moment):
            return self._rv_gate_result(done_lines)
        closers = {self.role_map.get('buy'), self.role_map.get('refuse')} - {None}
        tool_lines = self._rv_gate_tool_lines(closers)
        talk_needed = moment == 'order' and any(entry['messages'] > 0 for entry in due)
        nudged = False
        for _round in range(PROCEDURE_GATE_ROUNDS):
            seats = [DEEP_SEAT, QUICK_SEAT] if nudged else [QUICK_SEAT, DEEP_SEAT]
            ruling = self.ask_model_json(self._rv_gate_prompt(moment, due, tool_lines), seats, 3500)
            if ruling is None:
                break
            news, hold = self._rv_gate_reply_changes(ruling, done_lines)
            wanted_call = ruling['call'] if isinstance(ruling.get('call'), dict) else None
            self.journal.append({'kind': 'gate', 'stage': moment, 'call': _clip_texts(wanted_call, 200),
                                 'steps': _clip_texts(ruling.get('steps'), 120), 'hold': hold[:200], 'told': news})
            if news or hold:
                self.revision += 1
                change_text = '; '.join(news + ([hold] if hold else []))
                report = self._rv_gate_result(done_lines)[1]
                return False, (report + '\n' + 'The shopper answered with a change to the task: ' + change_text
                               + '. Bring the cart in line with it, then check out again.')
            if wanted_call:
                outcome_line = self._rv_gate_carry_out(wanted_call, closers)
                if outcome_line is None:
                    break
                done_lines.append(outcome_line)
                continue
            unproven = self._rv_gate_open_steps(ruling, talk_needed)
            if not (unproven and not done_lines and not nudged and self._rv_gate_can_message()):
                stamp = f'{self.basket_fingerprint()}#{self.revision}'
                self.procedure_passes.add((moment, stamp))
                break
            nudged = True
            self._rv_gate_push_back(unproven)
        return self._rv_gate_result(done_lines)

    # ================================================================== the final close review

    def audit_fingerprint(self, ids: list[str]) -> str:
        return f"{'|'.join(sorted(ids))}#{self.revision}"

    def _rv_audit_prompt(self, chosen: list[str], planned: bool, rule_ranked: bool) -> str:
        in_force = 'REQUIREMENTS IN FORCE:\n' + self.ledger_lines()
        if any(good.dropped_needs for good in self.goods):
            in_force += ('\nThese hold the shopper\'s later changes and win over the request\'s own words: a requirement '
                         'marked DROPPED no longer counts, even where the request still states it.')
        sections = [BASKET_AUDIT_PROMPT, 'REQUEST:\n' + self.shopper_request, in_force]
        if planned:
            asked = '; '.join(f'item {slot}: quantity {good.count_wanted}' for slot, good in enumerate(self.goods, 1))
            cart_block = ('CART (nothing added yet; judge these listings as if each were in the cart at the requested '
                          'quantity: ' + asked + '):\n' + ', '.join(chosen))
        else:
            cart_block = 'CART:\n' + self.basket_brief()
        sections.append(cart_block)
        records = []
        for ident in chosen:
            flag = 'SOLD OUT (the store says it can no longer be bought). ' if self.sold_out_now(ident) else ''
            records.append(f'{ident}: {flag}{self.listing_blurb(ident)}')
        sections.append('RECORDS OF THE LISTINGS IN THE CART:\n' + '\n'.join(records))
        if rule_ranked:
            sections.append('OTHER LISTINGS SEEN (newest first):\n' + self._rv_other_seen_rows(chosen, 15))
        return '\n\n'.join(sections)

    def _rv_escalation_seats(self, first_look: dict[str, Any], ceiling: float) -> list[str]:
        if first_look.get('ok') is False and DEEP_SEAT in self.audit_seats(ceiling):
            return [DEEP_SEAT]
        return []

    def _rv_audit_verdict(self, prompt: str, seats: list[str], planned: bool, rule_ranked: bool, ceiling: float) -> dict[str, Any] | None:
        """A live, unranked cart first gets a cheap look, and only an objection is passed on to the strong seat; any
        other cart goes straight to the seats the spend allows."""
        first_look = None
        if len(seats) > 1 and not (planned or rule_ranked or self.refusal_audited):
            first_look = self.ask_model_json(prompt, [QUICK_SEAT], 3000)
            if first_look:
                seats = self._rv_escalation_seats(first_look, ceiling)
        if not seats or self.credit_gone:
            return first_look
        return self.ask_model_json(prompt, seats, 3000) or first_look

    def _rv_audit_objection(self, verdict: dict[str, Any], chosen: list[str]) -> str:
        self.audit_leads = [lead for lead in self.audit_leads if lead not in chosen]
        complaints = [str(item) for item in _listify(verdict.get('problems'))][:5]
        upgrades = [str(alt) for alt in _listify(verdict.get('better')) if str(alt) in self.row_records and not self.sold_out_now(str(alt))][:4]
        text = 'A second, closer review says this cart is not right yet: ' + '; '.join(complaints or ['check it against the requirements'])
        text += '. Listings that seem better choices: ' + ', '.join(upgrades) if upgrades else ''
        return text + '. Fix the cart (or, if the review is wrong, prove it by reading the records once more), then check out.'

    def _rv_audit_now(self, stamp: str, chosen: list[str], planned: bool, seats: list[str], ceiling: float) -> str:
        self.audited_fingerprints.add(stamp)
        rule_ranked = self.digest.get('pick') in ('cheapest', 'most', 'least')
        prompt = self._rv_audit_prompt(chosen, planned, rule_ranked)
        verdict = self._rv_audit_verdict(prompt, seats, planned, rule_ranked, ceiling)
        self.journal.append({'kind': 'check', 'cart': chosen, 'got': verdict})
        objects = bool(verdict) and verdict.get('ok') is False
        self.audit_verdicts[stamp] = bool(verdict) and verdict.get('ok') is not False
        return self._rv_audit_objection(verdict, chosen) if objects else ''

    def audit_basket(self, chosen: list[str] | None = None, ceiling: float = SPEND_COMFORT) -> str:
        planned = chosen is not None
        lines = chosen if planned else self.basket_ids()
        stamp = self.audit_fingerprint(lines)
        seats = self.audit_seats(ceiling)
        if stamp in self.audited_fingerprints or not seats or not lines:
            return ''
        return self._rv_audit_now(stamp, lines, planned, seats, ceiling)
class RunBasket:
    """Wren's cart work within one episode.

    Covered here: spotting a checkout or a decline call, trimming the tool menu close to the step limit, the plain
    remark sent with a decline, turning listing text into ids, offers read again and sold-out replacements, the
    second look taken before a purchase-free ending, following requirements that borrow their value from another
    listing, the cart calls themselves, and buying the lead an independent check approved. All of it is read off the
    store's own tool schemas and the records gathered so far; calls to the store and to the model always come in the
    same order, so replaying an episode repeats it call for call.
    """

    _VENDOR_KEY_RX = 'seller|shop|store|vendor|brand|maker|merchant'
    _TRAIL_STOPWORDS = frozenset({'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for', 'of'})
    _PLAIN_DECLINE = 'Sorry, no listing in this store satisfies all of your requirements, so I am not buying anything.'
    _ENDING_PRESETS = {'buy': 'buy_args', 'refuse': 'refuse_args'}

    # ================================================================ which call ends the run, what stays offered

    def _ends_as(self, role: str, name: str, args: dict[str, Any]) -> bool:
        """True for a call to the tool behind this ending role that repeats each fixed argument noted for it."""
        preset = self.digest.get(self._ENDING_PRESETS[role])
        fixed = preset.items() if isinstance(preset, dict) else ()
        if name != self.role_map.get(role):
            return False
        return all(args.get(field) == expected for field, expected in fixed)

    def is_purchase_call(self, name: str, args: dict[str, Any]) -> bool:
        return self._ends_as('buy', name, args)

    def is_refusal_call(self, name: str, args: dict[str, Any]) -> bool:
        return self._ends_as('refuse', name, args)

    def offered_tool_specs(self) -> list[dict[str, Any]]:
        """Tools put before the model: the interviewer's tool never; close to the step limit only those that can still
        finish the run (the full menu again whenever the trimmed one would come out empty)."""
        enders = {self.role_map.get('buy'), self.role_map.get('refuse')} - {None}
        keep = self._closing_menu(self.moves_left(), enders)
        asker = self.role_map.get('ask')
        menu = [spec for spec in self.tool_list if spec['function']['name'] != asker]
        if keep is None or not enders:
            return menu
        return [spec for spec in menu if spec['function']['name'] in keep] or menu

    def _closing_menu(self, left: int, enders: set[str]) -> set[Any] | None:
        """Tool names worth keeping with `left` steps to go; None while the menu needs no trimming."""
        reserve = self.wrap_up_reserve()
        cart_tools = {self.role_map.get('cart'), self.role_map.get('cart_view')}
        uncovered = len(self._goods_without_line())
        # some requested good has no cart line yet, so finishing now could only be a decline: keep every tool while
        # each such good can still get one step to be read and one to be carted, plus a step for the checkout
        if uncovered and left >= 2 * uncovered + 1:
            return None
        if left <= reserve:
            return set(enders)
        extra = cart_tools if left <= reserve + 3 else None
        if extra is None and left <= len(self.goods) + 5 and self.purchase_tries == 0:
            extra = cart_tools | {self.role_map.get('offer')}
        return None if extra is None else enders | extra

    def quiet_refusal_args(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Free-text arguments of the decline call get the fixed plain remark (cut to the field's length) in place of
        the model's words: always when a quiet decline is wanted, else when the text is empty or names a listing."""
        schema = self.schemas.get(name) or {}
        needed = schema.get('required') or []
        hush = self.digest.get('quiet_decline') is not False
        out = dict(args)
        for field, spec in (schema.get('properties') or {}).items():
            written = out.get(field)
            if not self._is_plain_text(spec) or (written is None and field not in needed):
                continue
            text = str(written or '')
            if hush or not text or any(listing in text for listing in self.row_records):
                out[field] = self._PLAIN_DECLINE[:spec.get('maxLength', 200)]
        return out

    # ================================================================ requirements whose value sits on another listing

    def trace_borrowed_values(self) -> tuple[str, bool]:
        """Walk every requirement in force that borrows its value from another listing, one link per round; gives the
        note for the model and whether that note has to hold the order back."""
        in_force = {need for _, need in self.needs_in_force()}
        pending = [need for _, need in self.borrowed_needs if need in in_force and need not in self.borrowed_values]
        viewer = self.offer_viewer()
        slot = self.lone_id_arg(viewer)
        opened: list[str] = []
        unfound: list[str] = []
        while pending and self._may_trace_again():
            self.trace_rounds += 1
            self.trace_again = False
            hops, unfound, gaps = self._sort_chains(self._consult_tracer(pending), pending)
            pending = [need for need in pending if need not in self.borrowed_values]
            if not hops and gaps and viewer and slot:
                hops, unfound = self._search_first_gap(gaps[0], unfound, opened)
            if not (hops and viewer and slot) or self.run_closed:
                # a link Wren just found wrong, or a page that hands the property on, gets the tracer once more
                if self.trace_again and pending and viewer and slot and not self.run_closed:
                    continue
                break
            if len(hops) + 4 > self.moves_left():
                break
            self._open_pointer_pages(viewer, slot, hops[:max(0, min(3, 6 - len(opened)))], opened)
        self.trace_paged_mark = len(self.paged_ids)
        return self._compose_trail_note(opened, unfound)

    def trace_early(self) -> str:
        """Follow the requirements that take their value from another listing once, right after the interview, while
        there are steps and time to act on what the pages say; the rounds spent here do not count against the ones
        kept for the checkout."""
        if self.early_trail_done or not self._unsettled_borrowed() or self.moves_left() < 15:
            return ''
        if self.seconds_spent() > RUSH_SECONDS:
            return ''
        self.early_trail_done = True
        kept = (self.trace_rounds, self.trace_block_count)
        self.trace_rounds = 0
        try:
            report, _ = self.trace_borrowed_values()
        finally:
            self.trace_rounds, self.trace_block_count = kept
        self.journal.append({'kind': 'early_trail', 'left': self.moves_left(), 'values': list(self.borrowed_values.values())[:3]})
        return report

    def _may_trace_again(self) -> bool:
        round_cap = 6 if self.moves_left() > 15 else 4
        return self.trace_rounds < round_cap and self.model_cost < SPEND_CEILING

    def _consult_tracer(self, pending: list[str]) -> dict[str, Any]:
        sections = [LINK_TRACE_PROMPT,
                    'REQUIREMENTS THAT TAKE THEIR VALUE FROM ANOTHER LISTING:\n' + '\n'.join(pending),
                    "THE SHOPPER'S OWN REQUEST (each requirement above is a short reading of it; where the request's "
                    'words say how the wanted value relates to the value found along the chain, those words decide '
                    "the value, not the requirement's wording):\n" + self.shopper_request[:4000]]
        if self.trail_checks:
            sections.append('WHAT WREN CHECKED ON THE RECORDS:\n' + '\n'.join(self.trail_checks[-6:]))
        sections.append('RECORDS (most recent first):\n' + '\n'.join(self._trace_lines()))
        seats = [QUICK_SEAT]
        if self.model_cost + DEEP_SEAT_HEADROOM <= SPEND_CEILING:
            seats.append(DEEP_SEAT)
        answer = self.ask_model_json('\n\n'.join(sections), seats, 4500) or {}
        self.journal.append({'kind': 'trail', 'pending': pending, 'got': _clip_texts(answer, 150)})
        return answer

    @staticmethod
    def _labelled(tag: str, record: Any, cell: int, width: int) -> str:
        return tag + ' ' + _fitted_dump(record, cell, width)

    def _best_record(self, listing: str) -> Any:
        return self.page_records.get(listing) or self.row_records.get(listing)

    def _trace_lines(self) -> list[str]:
        """Records for the tracer, newest first in every group: OPENED pages (45 at most), then ATTRIBUTES records (up
        to 50 lines so far), then ROW entries (up to 60 lines in all)."""
        newest = list(reversed(self.seen_sequence))
        known = self._best_record
        full = [lid for lid in newest if lid in self.paged_ids and known(lid)][:45]
        out = [self._labelled('OPENED', known(lid), 300, 1400) for lid in full]
        traits = [lid for lid in newest if lid not in self.paged_ids and lid in self.trait_read_ids and known(lid)]
        traits = traits[:max(0, 50 - len(out))]
        out.extend(self._labelled('ATTRIBUTES', known(lid), 160, 1000) for lid in traits)
        rest = [lid for lid in newest if not known(lid) or (lid not in self.paged_ids and lid not in traits)]
        out.extend(self._labelled('ROW', self.row_records.get(lid), 80, 300) for lid in rest[:max(0, 60 - len(out))])
        return out

    @staticmethod
    def _need_of_chain(chain: dict[str, Any], pending: list[str]) -> str | None:
        named = str(chain.get('need') or '').strip()
        match = named if named in pending else None
        return match or (pending[0] if len(pending) == 1 else None)

    def _settled_value(self, chain: dict[str, Any], links: list[dict[str, Any]], ids: list[str]) -> str | None:
        """The borrowed value, once each link of the chain is an opened page and the last one asks for no search."""
        if not links or any(lid not in self.paged_ids for lid in ids):
            return None
        stated = str(chain.get('value') or '').strip()
        if chain.get('done') is True and stated:
            return stated
        if str(links[-1].get('find') or '').strip():
            return None
        return stated or f'as stated on {ids[-1]}'

    def _sort_chains(self, answer: dict[str, Any], pending: list[str]) -> tuple[list[str], list[str], list[tuple[str, str]]]:
        """Record each value the tracer settled; for open chains collect (rows to open, notes on links nobody found,
        (description, search words) for each such link)."""
        hops: list[str] = []
        unfound: list[str] = []
        gaps: list[tuple[str, str]] = []
        for chain in (entry for entry in _listify(answer.get('chains')) if isinstance(entry, dict)):
            need = self._need_of_chain(chain, pending)
            if not need:
                continue
            links = [link for link in _listify(chain.get('links')) if isinstance(link, dict)]
            ids = [self._link_listing_id(str(link.get('id') or '').strip()) for link in links]
            links, ids = self._drop_unlike_link(need, links, ids)
            value = self._settled_value(chain, links, ids)
            if value is not None and chain.get('done') is True and self._hands_on(need, ids[-1], value):
                value = None
            if value is None:
                self._queue_chain_gap(need, links, ids, (hops, unfound, gaps))
                continue
            self.borrowed_values[need] = value
            self.revision += 1
        return hops, unfound, gaps

    def _link_listing_id(self, given: str) -> str:
        """A link the tracer named, as the id of a seen listing. The tracer sometimes gives a code the listing's title
        prints (a part or series code) instead of its id: such a code maps to the one seen listing whose title holds
        it, else to the one record holding it anywhere."""
        if not given or given in self.row_records:
            return given
        inside = [lid for lid in self.row_records if len(lid) >= 6 and lid in given]
        if len(inside) == 1:
            return inside[0]
        if not 3 <= len(given) <= 40 or not re.search(r'\d', given):
            return given
        pattern = '(?<![a-z0-9])' + re.escape(given.lower()) + '(?![a-z0-9])'
        for whole in (False, True):
            hits = [lid for lid in self.seen_sequence
                    if re.search(pattern, (_jdump(self._best_record(lid) or {}) if whole
                                           else str((self.row_records.get(lid) or {}).get('title') or '')).lower())]
            if len(hits) == 1:
                self.journal.append({'kind': 'link_id', 'sent': given, 'id': hits[0]})
                return hits[0]
            if hits:
                break
        return given

    def _link_lacks(self, listing: str, words: str) -> list[str]:
        """The value words of a link's description that the listing's attributes and title hold nowhere (spaces and
        signs aside; a word straight after a number is its unit and is skipped; the shop's own name is skipped). The
        store's decoys share the named shop and kind and miss one described value."""
        records = [rec for rec in (self.page_records.get(listing), self.row_records.get(listing)) if isinstance(rec, dict)]
        blocks = [block for block in (_trait_block(rec) for rec in records) if block]
        if not blocks or not words.strip():
            return []
        names = {word for _, value in self._vendor_values(listing) for word in re.findall('[a-z0-9]+', value.lower())}
        title = str((self.row_records.get(listing) or {}).get('title') or '')
        held = re.sub('[^a-z0-9.]+', '', (' '.join(_jdump(block) for block in blocks) + ' ' + title).lower())
        tokens = [token.strip('.') for token in re.findall('[a-z0-9][a-z0-9.]*', words.lower())]
        wanted = [token for n, token in enumerate(tokens)
                  if token and (len(token) >= 3 or token[0].isdigit()) and token not in names
                  and token not in self._TRAIL_STOPWORDS and not (n and tokens[n - 1][:1].isdigit() and token.isalpha())]
        return [token for token in dict.fromkeys(wanted) if token not in held]

    def _drop_unlike_link(self, need: str, links: list[dict[str, Any]], ids: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
        """The first opened link whose attributes and title lack a value its description gives (the tracer's "values",
        which leave out property names, the shop and the kind of product) is not that link: the chain is cut there and
        that link becomes one still to find (search words: the tracer's, else the shop and the values), and the tracer
        hears what Wren found."""
        for place, (link, lid) in enumerate(zip(links, ids)):
            if not lid or lid not in self.paged_ids:
                continue
            missing = self._link_lacks(lid, str(link.get('values') or ''))
            about = str(link.get('about') or '')[:150]
            note = f'{lid} is not the listing described as "{about}": its attributes and title lack ' + ', '.join(missing[:3])
            if not missing or note in self.trail_checks:
                continue
            self.trail_checks.append(note)
            self.trace_again = True
            self.journal.append({'kind': 'link_unlike', 'id': lid, 'lacks': missing[:3]})
            shop = next((name for _, name in self._vendor_values(lid)
                         if self._word_in(name.lower(), (need + ' ' + about).lower())), '')
            words = str(link.get('find') or '').strip() or f"{shop} {link.get('values') or ''}".strip()
            return links[:place] + [dict(link, id='', find=words)], ids[:place] + ['']
        return links, ids

    def _hands_on(self, need: str, listing: str, value: str) -> bool:
        """Whether the page a chain was declared done on names the requirement's property in its own longer words
        without the value taken from it: such a page hands that property on to a different listing. Noted once, and
        the tracer is asked again."""
        prop = ' '.join(re.findall('[a-z0-9]+', need.split(':')[0].lower()))
        want = re.sub('[^a-z0-9]+', '', value.lower())
        if len(prop) < 3 or not want:
            return False
        for record in (self.row_records.get(listing) or {}, self.page_records.get(listing) or {}):
            for key, text in record.items():
                if not isinstance(text, str) or len(text) <= 40 or key == 'title':
                    continue
                spaced = ' '.join(re.findall('[a-z0-9]+', text.lower()))
                if self._word_in(prop, spaced) and want not in re.sub('[^a-z0-9]+', '', text.lower()):
                    note = (f'{listing} names the property of "{need[:80]}" in its own words without the value '
                            f'{value.strip()[:40]}: it hands that property on to a different listing, so the chain '
                            'goes on from what its words name')
                    if note in self.trail_checks:
                        return False
                    self.trail_checks.append(note)
                    self.trace_again = True
                    self.journal.append({'kind': 'trail_onward', 'id': listing, 'value': value[:40]})
                    return True
        return False

    def _queue_chain_gap(self, need: str, links: list[dict[str, Any]], ids: list[str], bins: tuple[list[str], list[str], list[tuple[str, str]]]) -> None:
        """An open chain: its rows not opened yet join the queue; with none, its first link no row matches is noted."""
        hops, unfound, gaps = bins
        unopened = [lid for lid in ids if lid in self.row_records and lid not in self.paged_ids]
        fresh = [lid for lid in unopened if lid not in hops]
        hops += fresh
        if unopened:
            return
        gap = next((link for link, lid in zip(links, ids) if lid not in self.row_records), {})
        about = str(gap.get('about') or '') or 'the listing it points at'
        words = str(gap.get('find') or '').strip()
        unfound.append(f'{need} -> find {about}' + (f' (search for: {words})' if words else ''))
        gaps.append((about, words))

    def _search_first_gap(self, gap: tuple[str, str], unfound: list[str], opened: list[str]) -> tuple[list[str], list[str]]:
        """No chain gave a row to open: hunt for the first missing link (only with steps to spare, three hunts at most)."""
        if self.moves_left() <= 12 or self.link_query_count >= 3:
            return [], unfound
        rows, known = self._locate_link(*gap)
        if known is None:
            return rows, unfound
        opened.append(f'{known}: an already opened page from the shop the chain names, and it shows the values '
                      'the description lists; read the value off it')
        return rows, unfound[1:]

    def _open_pointer_pages(self, viewer: str, slot: str, ids: list[str], opened: list[str]) -> None:
        for listing in ids:
            _, page, alive = self._send_one(viewer, {slot: listing})
            if not alive:
                return
            self.digest_reply(viewer, page)
            opened.append(f'{listing}: ' + self.show_reply(page)[:1500])

    def _trace_should_block(self, unfound: list[str]) -> bool:
        block_cap = 3 if self.moves_left() > 15 else 2
        return bool(unfound) and self.trace_block_count < block_cap and self.moves_left() > 5

    def _compose_trail_note(self, opened: list[str], unfound: list[str]) -> tuple[str, bool]:
        parts: list[str] = []
        if opened:
            parts.append('[Wren opened the pages the requirements point to]\n' + '\n'.join(opened) + '\n')
        taken = '; '.join(f'{need} -> {value}' for need, value in self.borrowed_values.items())
        if taken:
            parts.append('[values read off the pages the requirements point to] ' + taken + '\n')
        holds = self._trace_should_block(unfound)
        if holds:
            self.trace_block_count += 1
            parts.append(
                'An order only counts after every page a requirement points to has been opened in full (a row of '
                'search results is not enough), and so has each further listing such a page sends you on to. Still '
                'to be found and opened: ' + ' | '.join(unfound[:3]) + ". Search with the shop's name and the values "
                "the description gives (that shop's other listings of the same kind are not the one), open the row "
                'from that shop whose page shows those values (with the page view, since an attributes view alone '
                'hides where a page points), and use the value that page gives.')
        return ''.join(parts), holds

    # ---------------------------------------------------------------- finding the listing a chain link describes

    @staticmethod
    def _word_in(word: str, text: str) -> Any:
        return re.search('(?<![a-z0-9])' + re.escape(word) + '(?![a-z0-9])', text)

    def _vendor_values(self, listing: str) -> list[tuple[str, str]]:
        """(key, text) pairs of the listing's row whose key looks like a seller or brand field."""
        row = self.row_records.get(listing) or {}
        return [(key, value) for key, value in row.items()
                if isinstance(value, str) and re.search(self._VENDOR_KEY_RX, str(key), re.I)]

    def _shop_matches(self, about: str, pool: Any) -> list[str]:
        """Ids in pool whose seller-like value appears in the description as a whole word."""
        lowered = about.lower()
        found: list[str] = []
        for listing in pool:
            names = [str(value).strip() for _, value in self._vendor_values(listing) if 2 < len(value.strip()) <= 40]
            if any(self._word_in(name.lower(), lowered) for name in names):
                found.append(listing)
        return found

    def _unread_shop_rows(self, about: str, ids: list[str]) -> list[str]:
        return [lid for lid in self._shop_matches(about, ids) if lid not in self.paged_ids]

    def _shop_own_words(self, about: str) -> set[str]:
        """Words of the shop's own name, kept out of the values a link search is looking for."""
        pool = self._unread_shop_rows(about, list(self.row_records)) + self._shop_matches(about, self.paged_ids)
        return {word for lid in pool for _, value in self._vendor_values(lid) if len(value.strip()) <= 40
                for word in re.findall('[a-z0-9]{3,}', value.lower())}

    @staticmethod
    def _leaf_texts(record: Any) -> list[str]:
        stack, leaves = [record], []
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
            else:
                leaves.append(str(node))
        return leaves

    def _record_names_every(self, listing: str, wanted: set[str]) -> bool:
        blob = ' '.join(self._leaf_texts(self._best_record(listing) or {})).lower()
        return all(self._word_in(word, blob) for word in wanted)

    def _search_for_link(self, about: str, find_words: str) -> list[str] | None:
        """All rows a word search for a chain link has shown. A wording sent before reuses its rows without a step;
        once every row of the named shop it showed has been opened, its next page is fetched while the store offers
        one. None when no search can be sent (no words, no words search, the link-search allowance spent)."""
        plan = self.words_query_plan() or self._lone_text_query_plan()
        memo = ' '.join(find_words.lower().split())
        if not memo or not plan:
            return None
        found = self.link_query_hits.get(memo)
        onward = self.link_query_more.get(memo) if found is not None else None
        if found is not None and (self._unread_shop_rows(about, found) or not onward):
            return found
        if self.link_query_count >= 6 or self.moves_left() <= 12:
            return found
        tool, carried, words_field = plan
        fitted = self._fit_args(tool, {**carried, words_field: find_words, **({onward[0]: onward[1]} if onward else {})})
        if fitted is None:
            return found
        self.link_query_count += 1
        reply = self.shop_call(tool, fitted)
        if reply is None:
            return found
        page = reply.get('observation')
        self.digest_reply(tool, page)
        state = self._paging_state(tool, page)
        self.link_query_more[memo] = (state[2], state[3]) if state and state[3] is not None else None
        found = self.link_query_hits[memo] = list(dict.fromkeys((found or []) + [lid for lid, _ in self.records_in_reply(page)]))
        self.journal.append({'kind': 'link_search', 'words': find_words[:80], 'paged': bool(onward),
                             'rows': self._unread_shop_rows(about, found)[:4]})
        return found

    def _locate_link(self, about: str, find_words: str) -> tuple[list[str], str | None]:
        """Locate what a chain link describes: (rows worth opening, an opened page that already shows it or None)."""
        shop_words = self._shop_own_words(about)
        source = (find_words or about).lower()
        noise = shop_words | self._TRAIL_STOPWORDS
        wanted = {word for word in re.findall('[a-z0-9]+', source) if len(word) >= 3 or word.isdigit()} - noise
        candidates = self._shop_matches(about, self.paged_ids) if wanted else []
        known = next((lid for lid in candidates if self._record_names_every(lid, wanted)), None)
        if known is not None:
            self.journal.append({'kind': 'link_known', 'id': known})
            return [], known
        votes = self._vote_link_rows(about, find_words, shop_words)
        kind = set(re.findall('[a-z0-9]{3,}', f'{about} {find_words}'.lower())) - noise

        def of_kind(ids: list[str]) -> list[str]:
            """Rows sharing a word with the link's description (its kind, its values), those sharing more first."""
            scored = []
            for place, lid in enumerate(ids):
                hit = len(kind & set(re.findall('[a-z0-9]{3,}', _jdump(self.row_records.get(lid) or {}).lower())))
                if hit or not kind:
                    scored.append((-hit, place, lid))
            return [lid for _, _, lid in sorted(scored)]
        voted = sorted(of_kind(list(votes)), key=lambda lid: (-len(votes[lid]), min(votes[lid])))
        picks = voted or of_kind(self._unread_shop_rows(about, list(reversed(self.seen_sequence))))
        return picks[:2], None

    def _vote_link_rows(self, about: str, find_words: str, shop_words: set[str]) -> dict[str, list[int]]:
        """Unopened rows of the named shop that link searches showed, each with the sizes of the searches that showed
        it. The searches go from the whole wording to the shop's name with each value alone (when two or more values
        were given) to the shop's name alone, the last only when no value search showed such a row: a store whose
        search wants every word hides a link whose page spells one value otherwise, while the shop with one value
        shows it."""
        # the tracer's words, before any note it ran on into, without punctuation and repeats
        head = re.split(r'[;|]', find_words)[0]
        parts = list(dict.fromkeys(re.findall(r'[A-Za-z0-9][A-Za-z0-9.\-]*', head)))
        parts = [word.rstrip('.') for word in parts if word.rstrip('.')]
        if not parts:
            return {}
        find_words = ' '.join(parts)
        shop_part = [word for word in parts if word.lower() in shop_words] or parts[:1]
        shop = ' '.join(shop_part)
        values = [word for word in parts if word not in shop_part]
        singles = [f'{shop} {value}' for value in values][:3] if len(values) >= 2 else []
        votes: dict[str, list[int]] = {}
        for query in dict.fromkeys([' '.join(parts)] + singles + [shop]):
            if query == shop and votes:
                break
            found = self._search_for_link(about, query)
            if found is None:
                break
            fresh = self._unread_shop_rows(about, found)
            for lid in fresh:
                votes.setdefault(lid, []).append(len(found))
            if fresh and query not in singles:
                break
        return votes

    # ================================================================ second look before ending without a purchase

    def audit_refusal(self) -> str:
        """Once per run, ahead of a purchase-free ending: a judge rereads the gathered records; listings it names as
        fits become the leads, and the note returned points the model at them."""
        if self.refusal_audited or not self.audit_seats():
            return ''
        self.refusal_audited = True
        verdict = self._review_records()
        leads = self._named_fits(verdict)
        if not leads and self._open_review_rows(verdict):
            verdict = self._review_records()
            leads = self._named_fits(verdict)
        return self._take_up_leads(leads, verdict) if leads else ''

    def _review_line(self, listing: str) -> str:
        page = self.page_records.get(listing)
        if page:
            return self._labelled('OPENED', page, 200, 1200)
        return self._labelled('ROW', self.row_records.get(listing), 120, 500)

    def _review_records(self) -> dict[str, Any] | None:
        """One reading of the records gathered so far: the fast judge first, the strong one too when the fast one
        names no fit."""
        live = [lid for lid in reversed(self.seen_sequence) if not self.sold_out_now(lid)]
        held = [lid for lid in self.basket_ids() if lid in live]
        others = [lid for lid in live if lid not in held]
        # cart lines first, opened pages next (their attributes decide), search rows last; newest first in each
        ordered = (held + [lid for lid in others if lid in self.page_records][:20]
                   + [lid for lid in others if lid not in self.page_records][:15])
        if not ordered:
            return None
        sections = [REFUSAL_AUDIT_PROMPT, 'REQUEST:\n' + self.shopper_request,
                    'REQUIREMENTS IN FORCE:\n' + self.ledger_lines(),
                    'RECORDS OF LISTINGS SEEN (cart lines first, opened pages next, search rows last; most recent '
                    'first within each group):\n' + '\n'.join(self._review_line(lid) for lid in ordered)]
        prompt = '\n\n'.join(sections)
        seats = self.audit_seats()
        verdict = self.ask_model_json(prompt, [QUICK_SEAT] if seats else [], 4000)
        if DEEP_SEAT in seats and not self.credit_gone and not self._named_fits(verdict):
            verdict = self.ask_model_json(prompt, [DEEP_SEAT], 4000) or verdict
        self.journal.append({'kind': 'audit', 'got': verdict})
        return verdict

    def _buyable_ids(self, raw: Any) -> list[str]:
        """Known listing ids from a model answer that can still be bought, in the answer's order."""
        return [str(lid) for lid in _listify(raw) if str(lid) in self.row_records and not self.sold_out_now(str(lid))]

    def _named_fits(self, verdict: dict[str, Any] | None) -> dict[str, list[str]]:
        """item -> up to four buyable ids the judge reports as fitting; items left without any are dropped."""
        fits = (verdict or {}).get('fits')
        if not isinstance(fits, dict):
            return {}
        per_item = {str(item): self._buyable_ids(ids)[:4] for item, ids in fits.items()}
        return {item: ids for item, ids in per_item.items() if ids}

    def _review_read_plan(self, rows: list[str]) -> list[tuple[str, dict[str, Any]]] | None:
        list_arg = self.side_by_side_list_arg()
        compare_tool = self.role_map.get('compare')
        if list_arg and compare_tool and not self.side_by_side_hides_traits():
            return [(compare_tool, {list_arg: rows[:self.side_by_side_width()]})]
        single = self.trait_viewer()
        slot = self.lone_id_arg(single)
        return [(single, {slot: lid}) for lid in rows[:2]] if single and slot else None

    def _carry_out(self, plan: list[tuple[str, dict[str, Any]]]) -> bool:
        """Make the planned reads; False only when a schema turned one down (the caller then logs nothing)."""
        for tool, raw in plan:
            sent, page, alive = self._send_one(tool, raw)
            if not alive:
                return sent is not None
            self.digest_reply(tool, page)
        return True

    def _open_review_rows(self, verdict: dict[str, Any] | None) -> bool:
        """Read the pages of the unread rows the judge wanted to see; True when a new full record came in."""
        rows = [lid for lid in self._buyable_ids((verdict or {}).get('open')) if lid not in self.paged_ids]
        plan = self._review_read_plan(rows) if rows else None
        before = len(self.page_records)
        if plan is None or len(plan) + 5 > self.moves_left() or not self._carry_out(plan):
            return False
        self.journal.append({'kind': 'audit_open', 'rows': rows, 'opened': len(self.page_records) - before})
        return len(self.page_records) > before

    def _take_up_leads(self, leads: dict[str, list[str]], verdict: dict[str, Any]) -> str:
        self.leads_per_good = leads
        self.audit_leads = [lid for ids in leads.values() for lid in ids]
        why = verdict.get('reason')
        per_item = '; '.join(f"item {item}: {', '.join(ids)}" for item, ids in leads.items())
        because = f' ({why})' if why else ''
        return ('On a second look these listings may fit after all: ' + per_item + because
                + '. Open them and buy the one that truly fits; end with no purchase only if all of them fail.')

    # ================================================================ sold-out cart lines and offers read again

    def sold_out_now(self, listing: str) -> bool:
        return listing in self.sold_out_steps or listing in self.unbuyable_ids

    def offer_viewer(self) -> str | None:
        """The tool showing a listing's offer page: the offer view, failing that the plain open view."""
        return self.role_map.get('offer') or self.role_map.get('open')

    def sellout_guidance(self) -> str:
        """Cart lines that just became unbuyable are marked sold out, and the note advises what should replace them."""
        gone = [lid for lid in self.basket_ids() if lid not in self.sold_out_steps and self.sold_out_now(lid)]
        if not gone:
            return ''
        step = self.moves_made
        for lid in gone:
            self.sold_out_steps[lid] = step
        self.audit_leads = [lid for lid in self.audit_leads if lid not in gone]
        self.revision += 1
        notice = (f"[Wren] Sold out now: {', '.join(gone)} (the store reports it cannot be bought any more). It "
                  'will not come back, so do not open it again.')
        return notice + self._replacement_advice(self._stand_ins(gone))

    def find_sold_line(self) -> str:
        """After a refused checkout or a sell-out notice naming no line: several live lines get their offers read
        again; a lone live line is taken to be the one that can no longer be bought."""
        standing = [lid for lid in self.basket_ids() if lid not in self.sold_out_steps]
        if any(self.sold_out_now(lid) for lid in standing):
            return ''
        if len(standing) > 1:
            return self.reread_basket_offers()
        if standing:
            self.unbuyable_ids.add(standing[0])
        return ''

    def reread_basket_offers(self) -> str:
        """Read the offers of cart lines not looked at lately once more; the changes found, one per line."""
        viewer = self.offer_viewer()
        slot = self.lone_id_arg(viewer)
        due = self._lines_due_reread() if viewer and slot else []
        if not due or len(due) + 3 > self.moves_left():
            return ''
        changes = self._refresh_offers(viewer, slot, due)
        self.journal.append({'kind': 'reread', 'ids': due, 'changed': len(changes)})
        if changes:
            self.revision += 1
        return '\n'.join(changes)

    def _lines_due_reread(self) -> list[str]:
        """Live cart lines whose offer was last read before two steps ago or before the last refused checkout."""
        horizon = max(self.moves_made - 2, self.purchase_refused_at)
        return [lid for lid in self.basket_ids()
                if lid not in self.sold_out_steps and self.offer_seen_at.get(lid, -99) < horizon]

    def _refresh_offers(self, viewer: str, slot: str, due: list[str]) -> list[str]:
        changes: list[str] = []
        for listing in due:
            before = dict(self.row_records.get(listing) or {})
            _, page, alive = self._send_one(viewer, {slot: listing})
            if not alive:
                break
            self.offer_seen_at[listing] = self.moves_made
            self.digest_reply(viewer, page)
            change = self._describe_offer_change(listing, before, page)
            if change:
                changes.append(change)
        return changes

    def _describe_offer_change(self, listing: str, before: dict[str, Any], page: Any) -> str:
        after = self.row_records.get(listing) or {}
        moved = [key for key, value in after.items()
                 if not isinstance(value, (dict, list)) and key in before and before[key] != value]
        if not moved:
            return ''
        return f"{listing} has changed ({', '.join(moved[:4])}): {_jdump(_clip_texts(page, 300))[:900]}"

    def _stand_ins(self, gone: list[str]) -> list[str]:
        """A judge picks already-read listings able to replace the sold-out lines (four at most)."""
        seats = self.audit_seats()
        others = [lid for lid in reversed(self.seen_sequence) if lid not in gone and not self.sold_out_now(lid)][:25] if seats else []
        if not others:
            return []
        records = [_fitted_dump(self._best_record(lid), 200, 1200) for lid in others]
        sections = [SUBSTITUTE_PROMPT, 'REQUEST:\n' + self.shopper_request,
                    'REQUIREMENTS IN FORCE:\n' + self.ledger_lines(),
                    'CART LINE NOW SOLD OUT:\n' + '\n'.join(f'{lid}: {self.listing_blurb(lid)}' for lid in gone),
                    'CART NOW:\n' + self.basket_brief(),
                    'RECORDS OF OTHER LISTINGS SEEN (most recent first):\n' + '\n'.join(records)]
        priced = self._usable_fits(self.ask_model_json('\n\n'.join(sections), seats[::-1], 3500), gone)
        order = sorted(priced, key=lambda fit: fit[0]) if self.digest.get('pick') == 'cheapest' else priced
        self.journal.append({'kind': 'swap', 'lost': gone, 'fits': priced})
        return [lid for _, lid in order][:4]

    def _usable_fits(self, answer: dict[str, Any] | None, gone: list[str]) -> list[tuple[float, str]]:
        """(price, id) for each listed fit that is known, buyable, new to the cart and not listed twice; inf when the
        answer gave no numeric price."""
        picked: list[tuple[float, str]] = []
        taken: set[str] = set()
        for entry in _listify((answer or {}).get('fits')):
            is_card = isinstance(entry, dict)
            listing = str(entry.get('id') if is_card else entry)
            if listing not in self.row_records or listing in gone or listing in self.sold_out_steps:
                continue
            if listing in self.basket_ids() or listing in taken:
                continue
            price = entry.get('price') if is_card else None
            if isinstance(price, bool) or not isinstance(price, (int, float)):
                price = float('inf')
            picked.append((float(price), listing))
            taken.add(listing)
        return picked

    def _replacement_advice(self, options: list[str]) -> str:
        if not options:
            return (' Remove that line from the cart and add the best other listing that fits instead; if no listing '
                    'read so far fits, search for one.')
        first = options[0]
        if self.moves_left() <= 6:
            spare = ', '.join(options[1:]) or 'none'
            return (f' Few steps are left: in this turn put {first} in at the requested quantity and remove the '
                    f'sold-out line (two calls), then check out. Other listings already read that look like a fit: '
                    f'{spare}.')
        listed = ', '.join(options)
        if self.digest.get('pick') == 'cheapest':
            return (f' Listings already read that look able to replace that item, cheapest first: {listed}. The pick '
                    f"rule wants the lowest price, so run one search under {first}'s price (passing the search's "
                    'maximum price, if it takes one) before swapping the cheapest fitting listing in for the sold-out '
                    'line, unless every fitting listing has been seen already.')
        return (f' Listings already read that look able to replace that item: {listed}. In this turn take the '
                f'sold-out line out of the cart and put {first} in at the requested quantity, then check out.')

    # ================================================================ listing ids the model wrote as text

    @staticmethod
    def _carries_ids(spec: Any) -> bool:
        if not isinstance(spec, dict) or 'enum' in spec:
            return False
        kinds = _declared_types(spec)
        return 'string' in kinds or 'array' in kinds

    def _id_slot(self, name: str) -> str | None:
        """The lone string or list argument through which a listing tool takes ids, else None."""
        fields = (self.schemas.get(name) or {}).get('properties') or {}
        slots = [field for field, spec in fields.items() if self._carries_ids(spec)]
        if len(slots) != 1:
            return None
        if name == self.role_map.get('cart') or (name, slots[0]) in self.id_args:
            return slots[0]
        return None

    def normalize_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """Swap listing text the model wrote for the listing's id; (arguments, '') or (arguments, the complaint)."""
        listing_tools = {self.role_map.get(role) for role in ('cart', 'open', 'offer', 'compare')} - {None}
        slot = self._id_slot(name) if name in listing_tools and self.row_id_key else None
        if slot is None:
            return args, ''
        if slot not in args:
            return dict(args), ''
        value = args[slot]
        many = isinstance(value, list)
        resolved = [self._resolve_listing_text(item) for item in (value if many else [value])]
        complaint = next((issue for _, issue in resolved if issue), '')
        if complaint:
            return args, complaint
        ids = [item for item, _ in resolved]
        return {**args, slot: ids if many else ids[0]}, ''

    def _rows_opening_with(self, prefix: str) -> list[str]:
        id_key = self.row_id_key
        return [lid for lid, row in self.row_records.items()
                if any(key != id_key and isinstance(text, str) and text.strip().lower().startswith(prefix)
                       for key, text in row.items())]

    def _resolve_listing_text(self, raw: Any) -> tuple[Any, str]:
        """One argument value -> (value to send, complaint). Text that begins exactly one listing becomes its id."""
        if not isinstance(raw, str):
            return raw, ''
        text = raw.strip()
        if text in self.row_records:
            return text, ''
        short = len(text) < 8
        known = list(self.row_records)
        shared = _common_id_prefix(known) if len(known) >= 5 else ''
        if short and len(shared) >= 2 and not text.startswith(shared):
            return raw, f'{raw!r} is not a listing id (all listing ids here begin with {shared!r}); take one from the result rows'
        if short or _id_shaped(raw, True):
            return raw, ''
        hits = self._rows_opening_with(text.lower())
        if len(hits) == 1:
            return hits[0], ''
        if hits:
            return raw, f"the text {raw!r} is how several listings start ({', '.join(hits[:6])}); name one listing id"
        return raw, (f'{raw!r} is not a listing id; send an id as the result rows show it (in their '
                     f'{self.row_id_key} field)')

    # ================================================================ cart calls built from the cart tool's schema

    @staticmethod
    def _is_plain_text(spec: Any) -> bool:
        """A string argument that is not restricted to a list of options."""
        return isinstance(spec, dict) and 'string' in _declared_types(spec) and 'enum' not in spec

    @staticmethod
    def _is_choice(spec: Any) -> bool:
        """An argument picking one of two or more fixed options (add / remove and the like)."""
        return isinstance(spec, dict) and isinstance(spec.get('enum'), list) and len(spec['enum']) >= 2

    @staticmethod
    def _is_count(spec: Any) -> bool:
        return isinstance(spec, dict) and 'integer' in _declared_types(spec)

    @staticmethod
    def _first_field(fields: dict[str, Any], test: Any) -> str | None:
        return next((field for field, spec in fields.items() if test(spec)), None)

    @staticmethod
    def _holds_line_objects(spec: Any) -> bool:
        """A list argument whose items are objects with properties (whole cart lines)."""
        if not isinstance(spec, dict) or 'array' not in _declared_types(spec):
            return False
        member = spec.get('items')
        return isinstance(member, dict) and isinstance(member.get('properties'), dict)

    def _fit_args(self, tool: str, raw: dict[str, Any]) -> dict[str, Any] | None:
        """raw shaped to the tool's schema; None when the schema rejects it."""
        problems: list[str] = []
        fitted = _fit_to_schema(raw, self.schemas.get(tool), 'args', problems)
        return None if problems else fitted

    def _send_one(self, tool: str, raw: dict[str, Any]) -> tuple[dict[str, Any] | None, Any, bool]:
        """Send one schema-shaped call -> (arguments sent, observation, alive). The arguments are None when the schema
        rejected them and nothing went out; alive is False when the store gave no reply or the run has closed, and
        the observation is then of no use."""
        fitted = self._fit_args(tool, raw)
        if fitted is None:
            return None, None, False
        reply = self.shop_call(tool, fitted)
        if reply is None or self.run_closed:
            return fitted, None, False
        return fitted, reply.get('observation'), True

    def basket_sets_count(self, tool: str) -> bool:
        """True for a cart tool that sets a line's units outright: no option switch and an integer allowed to be 0."""
        specs = list((self.schemas.get(tool, {}).get('properties') or {}).values())
        if any(self._is_choice(spec) for spec in specs):
            return False
        return any(self._is_count(spec) and spec.get('minimum') == 0 for spec in specs)

    @staticmethod
    def _is_removal_value(fields: dict[str, Any], field: str, value: Any) -> bool:
        options = (fields.get(field) or {}).get('enum')
        return isinstance(options, list) and value in fields[field]['enum'][1:]

    def whole_line_removal(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        """A removal sent without a count gets every unit the line holds; otherwise the store takes off just one."""
        fields = self.schemas.get(tool, {}).get('properties') or {}
        units_key = self._first_field(fields, self._is_count)
        removing = any(self._is_removal_value(fields, field, value) for field, value in args.items())
        line = next((value for value in args.values() if isinstance(value, str) and value in self.basket_counts), None)
        if not (units_key and removing and line) or units_key in args:
            return args
        held = self.basket_counts[line]
        return {**args, units_key: min(held, fields[units_key].get('maximum', held))}

    def basket_edit_calls(self, wanted: dict[str, int], removals: list[str]) -> list[dict[str, Any]] | None:
        """Cart calls that turn the cart into `wanted` (id -> units) after taking off `removals`; None when the cart
        tool's schema cannot express that."""
        tool = self.role_map.get('cart') or ''
        schema = self.schemas.get(tool) or {}
        fields = schema.get('properties') or {}
        list_key = next((field for field, spec in fields.items() if self._holds_line_objects(spec)), None)
        if list_key is not None:
            return self._whole_basket_call(list_key, fields[list_key]['items']['properties'], wanted)
        id_key = self._line_id_field(tool, schema, fields)
        return self._per_line_calls(tool, fields, id_key, wanted, removals) if id_key else None

    def _whole_basket_call(self, list_key: str, inner: dict[str, Any], wanted: dict[str, int]) -> list[dict[str, Any]] | None:
        """The full cart as one call carrying a list of line objects (id, plus a count when the line takes one)."""
        line_id = self._first_field(inner, self._is_plain_text)
        units_key = self._first_field(inner, self._is_count)
        if not line_id:
            return None
        lines: list[dict[str, Any]] = []
        for listing, units in wanted.items():
            line = {line_id: listing}
            if units_key:
                line[units_key] = units
            lines.append(line)
        return [{list_key: lines}]

    def _line_id_field(self, tool: str, schema: dict[str, Any], fields: dict[str, Any]) -> str | None:
        texts = [field for field, spec in fields.items() if self._is_plain_text(spec)]
        registered = next((field for owner, field in self.id_args if owner == tool and field in texts), None)
        required = next((field for field in schema.get('required') or [] if field in texts), None)
        if registered or required:
            return registered or required
        return texts[0] if len(texts) == 1 else None

    def _per_line_calls(self, tool: str, fields: dict[str, Any], id_key: str, wanted: dict[str, int], removals: list[str]) -> list[dict[str, Any]] | None:
        """One call per line that changes: removals first, then each line whose units differ from the wanted count;
        None as soon as one of them cannot be expressed."""
        units_key = self._first_field(fields, self._is_count)
        mode_key = self._first_field(fields, self._is_choice)
        present = self.basket_ids()
        absolute = self.basket_sets_count(tool)
        shape = (id_key, units_key, mode_key, fields, absolute)
        planned = [self._drop_line_args(listing, shape) for listing in removals]
        for listing, want in wanted.items():
            have = self.basket_counts.get(listing, 0) if listing in present else 0
            if have == want:
                continue
            planned.append(None if have and not absolute else self._add_line_args(listing, want, shape))
        return None if None in planned else planned

    def _drop_line_args(self, listing: str, shape: tuple[Any, ...]) -> dict[str, Any] | None:
        id_key, units_key, mode_key, fields, absolute = shape
        units = self.basket_counts.get(listing, 1)
        if mode_key:
            call = {id_key: listing, mode_key: fields[mode_key]['enum'][1]}
            if units_key:
                call[units_key] = units
            return call
        if units_key and absolute:
            return {id_key: listing, units_key: 0}
        floor = fields[units_key].get('minimum') if units_key else None
        if units_key and isinstance(floor, int) and floor <= -units:
            return {id_key: listing, units_key: -units}
        return None

    @staticmethod
    def _add_line_args(listing: str, want: int, shape: tuple[Any, ...]) -> dict[str, Any]:
        id_key, units_key, mode_key, fields, _ = shape
        call = {id_key: listing}
        if mode_key:
            call[mode_key] = fields[mode_key]['enum'][0]
        if units_key:
            call[units_key] = want
        return call

    # ================================================================ buying the lead the independent check approved

    def purchase_backed_lead(self) -> str:
        """Build the cart around the approved lead and check out on Wren's own; '' when that did not happen."""
        cart_tool = self.role_map.get('cart')
        order_tool = self.role_map.get('buy')
        lead = self.backed_lead
        if self.lead_purchased or not lead or self.sold_out_now(lead) or not cart_tool or not order_tool:
            return ''
        done = self._buy_around_lead(lead, cart_tool, order_tool)
        if done is None:
            return ''
        sent_lines, answers = done
        return ('Not run (Wren acted instead): the separate check backed ' + lead + ', so Wren arranged the cart ('
                + '; '.join(sent_lines) + ') and proceeded to checkout. The store replied: ' + ' | then: '.join(answers))

    def _buy_around_lead(self, lead: str, cart_tool: str, order_tool: str) -> tuple[list[str], list[str]] | None:
        """(cart calls sent, store answers) once the cart was rebuilt around the lead and checked out, else None."""
        self.lead_purchased = True
        shelf_key = self.shelf_field_name()
        chosen = self._lines_from_leads(shelf_key)
        layout = self._cart_after_swap(chosen, shelf_key) if lead in chosen else None
        calls = self.basket_edit_calls(*layout) if layout else None
        if not calls or len(calls) + 1 > self.moves_left():
            return None
        lines, removals = layout
        room = len(calls) + 3
        self.read_traits_of(list(lines), room)
        self.chase_part_refs(list(lines), room)
        if not self._review_passes(lines):
            return None
        sent = self._push_cart_calls(cart_tool, calls)
        if sent is None:
            return None
        answers = self._checkout_twice_at_most(order_tool, lines, removals)
        return (sent, answers) if answers else None

    def _shelf_of(self, listing: str, shelf_key: str | None) -> str:
        """The listing's category text in lower case, or the id itself when rows carry no category field."""
        if not shelf_key:
            return listing
        return str((self.row_records.get(listing) or {}).get(shelf_key) or '').strip().lower()

    @staticmethod
    def _item_slot(label: Any, total: int) -> int | None:
        digits = re.search('\\d+', str(label))
        if digits:
            return int(digits.group())
        return 1 if total == 1 else None

    def _lines_from_leads(self, shelf_key: str | None) -> dict[str, int]:
        """A buyable lead per requested item, one per category -> the units that item wants."""
        chosen: dict[str, int] = {}
        total = len(self.goods)
        for label, ids in self.leads_per_good.items():
            slot = self._item_slot(label, total)
            pick = next((lid for lid in ids if not self.sold_out_now(lid)), None)
            if not (pick and slot and 1 <= slot <= total):
                continue
            if self._shelf_of(pick, shelf_key) in {self._shelf_of(other, shelf_key) for other in chosen}:
                continue
            chosen[pick] = self.goods[slot - 1].count_wanted
        return chosen

    def _cart_after_swap(self, chosen: dict[str, int], shelf_key: str | None) -> tuple[dict[str, int], list[str]] | None:
        """(cart lines after the swap, lines to take off); None when those lines would leave an item uncovered."""
        taken = {self._shelf_of(lid, shelf_key) for lid in chosen}
        current = self.basket_ids()
        kept: dict[str, int] = {}
        for lid in current:
            if lid in chosen or self.sold_out_now(lid) or self._shelf_of(lid, shelf_key) in taken:
                continue
            kept[lid] = self.basket_counts.get(lid, 1)
        removals = [lid for lid in current if lid not in chosen and lid not in kept]
        kept_kinds = {self._shelf_of(lid, shelf_key) for lid in kept}
        if len(chosen) + len(kept_kinds) < len(self.goods):
            return None
        return {**kept, **chosen}, removals

    def _review_passes(self, lines: dict[str, int]) -> bool:
        self.audit_basket(list(lines), SPEND_CEILING)
        verdict = self.audit_verdicts.get(self.audit_fingerprint(list(lines)))
        return bool(verdict)

    def _push_cart_calls(self, cart_tool: str, calls: list[dict[str, Any]]) -> list[str] | None:
        """Send the cart calls in order; the arguments sent, as JSON, or None once one of them fails."""
        shown: list[str] = []
        for raw in calls:
            sent, page, alive = self._send_one(cart_tool, raw)
            if not alive:
                return None
            self.note_basket_call(cart_tool, sent)
            self.digest_reply(cart_tool, page)
            shown.append(_jdump(sent))
        return shown

    def _try_checkout(self, order_tool: str, order_args: dict[str, Any], lines: dict[str, int], removals: list[str], answers: list[str]) -> bool:
        """One checkout attempt, after any process step due; True when the store refused and the run goes on."""
        cleared, remark = self.run_procedure_gate('order')
        if remark:
            answers.append(remark)
        if not cleared:
            return False
        reply = self.shop_call(order_tool, order_args)
        self.journal.append({'kind': 'completed', 'lines': lines, 'drop': removals, 'reply': _clip_texts(reply, 200)})
        if reply is None:
            return False
        self.purchase_tries += 1
        finished = reply.get('finished')
        if not finished:
            self.purchase_refused_at = self.moves_made
        seen = reply.get('observation')
        self.digest_reply(order_tool, seen)
        answers.append(self.show_reply(seen))
        return not self.run_closed and not finished

    def _second_try_ok(self, lines: dict[str, int]) -> bool:
        self.reread_basket_offers()
        if not self._review_passes(lines):
            return False
        return self.moves_left() >= 1 and not self.run_closed and not any(self.sold_out_now(lid) for lid in lines)

    def _checkout_twice_at_most(self, order_tool: str, lines: dict[str, int], removals: list[str]) -> list[str]:
        """At most two checkout attempts; the second only after fresh offer reads and a passing review."""
        order_args = self._fit_args(order_tool, dict(self.digest.get('buy_args') or {}))
        answers: list[str] = []
        if order_args is None:
            return answers
        for attempt in range(2):
            if attempt and not self._second_try_ok(lines):
                break
            if not self._try_checkout(order_tool, order_args, lines, removals, answers):
                break
        return answers
_CLOSED_MARK = ' [episode over]'

# Words that bound, negate or hedge a required value; a value holding one stays out of the capped word search.
_HEDGED_VALUE_RX = re.compile(r'\b(?:not|no|unlike|other than|or|at least|at most|under|over|below|above|between|within|up to|same|identical)\b', re.I)

_OPENING_NUDGE = ("Start by searching the category of every item and reading a few listings completely, so that the "
                  'interviewer finds out which properties such listings carry.')


class _ProposedCall:
    """A tool call the model asked for, carried from check to check until it is held back or sent to the store.

    The checks write down what they work out on it (decoded parameters, a fingerprint of the call, the list parameter
    of a multi-listing view and so on) and may add Wren's remarks to `side_notes`; those remarks travel back to the
    model together with the store's answer.
    """

    __slots__ = ('call_id', 'merged', 'tool', 'arg_text', 'batch_size', 'params', 'list_key', 'id_key', 'finishing',
                 'fingerprint', 'read_tools', 'side_notes', 'refreshed', 'trail')

    def __init__(self, entry: dict[str, Any], batch_size: int) -> None:
        self.call_id = entry['id']
        self.merged = bool(entry.get('merged'))
        fn = entry.get('function') or {}
        self.tool, self.arg_text = fn.get('name'), fn.get('arguments')
        self.batch_size = batch_size
        self.params: dict[str, Any] = {}
        self.list_key: str | None = None
        self.id_key: str | None = None
        self.finishing = False
        self.fingerprint = ''
        self.read_tools: set[str] = set()
        self.side_notes = self.refreshed = self.trail = ''


class RunTurnLoop:
    # Checks every proposed call passes, in this order; the first one that answers keeps the call from the store.
    _VET_ORDER = ('_vet_entry', '_vet_params', '_vet_side_room', '_vet_sold_target', '_vet_ghost_removal',
                  '_vet_finish_alone', '_vet_repeat', '_vet_side_overlap', '_vet_trace_first', '_vet_finish_pauses',
                  '_vet_late_refusal', '_vet_unused_searches', '_vet_backed_lead', '_vet_last_question',
                  '_vet_procedure')
    # Pauses on a checkout call (they may spend steps and model calls), in this order.
    _BUY_PAUSES = ('_pause_wider_hunt', '_pause_cart_look', '_pause_sold_lines', '_pause_borrowed_trail', '_pause_fit',
                   '_pause_counts', '_pause_proven_set', '_pause_unshown_needs', '_pause_cheaper_fit', '_pause_audit',
                   '_pause_reread')
    # Pauses on an ending without purchase, in this order.
    _REFUSAL_PAUSES = ('_pause_proven_decline', '_pause_refusal_trail', '_pause_refusal_audit', '_pause_leads')

    # ================================================================== start to finish

    def run_to_end(self) -> list[dict[str, Any]]:
        """Play the episode out and return the journal; a run the store already reports closed is left as it is."""
        if self.run_closed:
            return self.journal
        self.digest_request()
        self.journal.append({'kind': 'brief', 'brief': self.digest})
        self.convo_log = [{'role': 'system', 'content': DRIVER_GUIDE},
                          {'role': 'user', 'content': self._first_user_text()}]
        self._run_turns(TURN_CEILING, SPEND_CEILING)
        if not self.run_closed:
            self._closing_fallback()
        self.journal.append({'kind': 'end', 'finished': self.run_closed, 'steps': self.moves_made,
                             'cost': round(self.model_cost, 5), 'calls': self.model_calls_made})
        return self.journal

    def _first_user_text(self) -> str:
        shown = self.policy.get('opening')
        if shown is None:
            store_view = []
        else:
            self.ingest_records(shown)
            store_view = ['WHAT THE STORE DISPLAYED FIRST:\n' + self.show_reply(shown)]
        early = self._opening_moves()
        if early:
            store_view.append('WREN SEARCHED EACH ITEM ONCE AND READ A FEW OF ITS LISTINGS BEFORE YOUR FIRST TURN '
                              '(one step per call; build on these results instead of repeating them):\n' + early)
        sections = ['THE REQUEST FROM THE SHOPPER:\n' + self.shopper_request, *store_view, _OPENING_NUDGE,
                    self.status_line()]
        return '\n\n'.join(sections)

    def _opening_moves(self) -> str:
        """Before the model's first turn on a task of two or more items with room to spare: one words search per item
        (its plain values with its name, then its name alone when that finds nothing) and up to two of its leading
        buyable rows opened (four on a task of more than 50 steps), so the first model turns are not spent on calls code can make. A model turn takes many
        seconds while a store call takes well under one, and tasks with several items are the ones the clock ends.
        At most a fifth of the step allowance; '' when nothing was sent."""
        plan = self._lone_text_query_plan()
        if len(self.goods) < 2 or self.step_cap < 20 or not plan or self.run_closed:
            return ''
        tool, base_args, text_field = plan
        longest = ((self.schemas.get(tool) or {}).get('properties') or {}).get(text_field, {}).get('maxLength')
        floor = self.moves_left() - self.step_cap // 5
        cap = self.money_cap if self._is_number(self.money_cap) else None
        # on a long task the clock, not the step allowance, ends the run: code reads more before the first model turn
        per_item = 4 if self.step_cap > 50 else 2
        shown: list[str] = []
        for slot in range(1, len(self.goods) + 1):
            title = self.goods[slot - 1].good_title.strip()
            tries = list(dict.fromkeys(q for q in (self._query_words_for_good(slot), title) if q))
            rows: list[str] = []
            for words in tries:
                if isinstance(longest, int) and len(words) > longest:
                    words = words[:longest].rsplit(' ', 1)[0]
                if self.moves_left() <= floor or self.run_closed:
                    break
                sent, reply = self._send_fitted(tool, {**base_args, text_field: words})
                if reply is None:
                    break
                observed = reply.get('observation')
                self.digest_reply(tool, observed)
                rows = [lid for lid, _ in self.records_in_reply(observed)]
                shown.append(f'search {_jdump(sent)}:\n' + (self.show_reply(observed) if rows else 'no rows'))
                if rows:
                    self.recent_queries = (self.recent_queries + [dict(sent)])[-8:]
                    break
            picks = []
            for lid in rows:
                price = self.price_of_listing(lid)
                if lid in self.page_records or not self.is_purchasable(lid) or (cap is not None and price is not None and price > cap):
                    continue
                picks.append(lid)
                if len(picks) == per_item:
                    break
            if picks and self.moves_left() > floor + len(picks):
                opened = self.read_traits_of(picks, max(floor, self.wrap_up_reserve() + 2))
                shown += [f'read in full: {text}' for text in opened]
        if shown:
            self.journal.append({'kind': 'opening_moves', 'steps': self.moves_made, 'rows': len(self.row_records),
                                 'opened': len(self.page_records)})
        return '\n'.join(shown)

    def _closing_fallback(self) -> None:
        """The model stopped before any ending: Wren ends the run itself, and when the store refuses it gives the model
        a few more turns before trying a last time."""
        refused = self.close_out_run(explain_refusal=True)
        retry = (refused and not self.run_closed and not self.wallet_empty and not self.credit_gone
                 and self.moves_left() > 1)
        if retry:
            self._chat_user('[Wren] Wren attempted to end the episode, but ' + refused
                            + '\nDo what the store asks, then finish.\n' + self.status_line())
            self._run_turns(self.driver_turns + 10, SPEND_CEILING + 0.01)
        if not self.run_closed:
            self.close_out_run()

    # ================================================================== turns

    def _cannot_act(self) -> bool:
        return self.wallet_empty or self._no_more_moves()

    def _turn_allowed(self, turn_cap: int, spend_cap: float) -> bool:
        if self._cannot_act():
            return False
        if self.driver_turns >= turn_cap or self.model_cost >= spend_cap:
            return False
        if self.seconds_spent() < LAST_TURN_SECONDS:
            return True
        # beyond the last-turn mark only an empty cart, with model time still in hand, earns another turn
        return not self.basket_ids() and self.model_time_allowance() > BARE_BASKET_SECONDS

    def _run_turns(self, turn_cap: int, spend_cap: float) -> None:
        idle: int | None = 0   # model replies in a row without a tool call; None means stop now
        while idle is not None and self._turn_allowed(turn_cap, spend_cap):
            idle = self._interview_turn(idle) if self.ready_for_questions() else self._model_turn(idle)

    def _interview_turn(self, idle: int) -> int:
        news = self.conduct_interview()
        chain = self.trace_early() if not self.run_closed else ''
        if chain:
            news = (news + '\n' if news else '') + chain
        if news:
            self._chat_user(news + '\n' + self.status_line())
        return idle

    def _model_turn(self, idle: int) -> int | None:
        if self._code_order_turn():
            return 0
        self._shrink_old_results()
        self.driver_turns += 1
        began = self.seconds_spent()
        # past the opening, the store-loop turns think briefly: a few long-thinking turns cost 30-45 s each late on
        self.brisk_turn = began >= BRISK_AFTER_SECONDS
        try:
            answer = self.ask_model(self.convo_log, [QUICK_SEAT, FALLBACK_SEAT], TURN_TOKEN_ROOM, self.offered_tool_specs())
        finally:
            self.brisk_turn = False
        self.turn_seconds.append(self.seconds_spent() - began)
        if answer is None:
            return self._after_failed_turn(idle)
        offered = answer.get('tool_calls') or []
        calls = [self._clean_call(slot, raw) for slot, raw in enumerate(offered) if isinstance(raw, dict)]
        said = self._reply_words(answer)
        if calls:
            handler = self._ask_before_commit if self._commit_before_interview(calls) else self._relay_batch
            handler(said, calls)
            return 0
        self.convo_log.append({'role': 'assistant', 'content': said or '(no tool used)'})
        self._chat_user('Go on with a call to one of the tools.\n' + self.status_line())
        idle += 1
        return idle if idle < 3 else None

    def _code_order_turn(self) -> bool:
        """Once, on an "any" pick, in place of a model turn: when the read listings make a fully proven order (one line
        per item, every requirement stated under its own key, inside the budget) and the cart is not that order yet,
        Wren sends the cart calls and then the checkout itself, through the same checks as a model call (the
        checkout must still pass every pause before it is sent). Lowest-price and ranked picks are left to the model,
        since an early order there skips cheaper listings not read yet. Whether it acted."""
        if self.code_order_tried or self.digest.get('pick') not in (None, '', 'any') or self.run_closed:
            return False
        if self.purchase_tries or self._unsettled_borrowed():
            return False
        # the shopper may still hold needs back: never order before the interview had its chance
        if self.role_map.get('ask') and not self.interview_ran and self.free_message_count() > 0:
            return False
        plan = self.proven_set()
        if not plan:
            return False
        cart_tool, buyer = self.role_map.get('cart'), self.role_map.get('buy')
        held = {lid: units for lid, units in self.basket_counts.items() if units > 0}
        calls = [] if plan == held else (self.basket_edit_calls(plan, [lid for lid in held if lid not in plan]) or [])
        if not cart_tool or not buyer or (plan != held and not calls):
            return False
        if len(calls) + self.wrap_up_reserve() + 2 > self.moves_left():
            return False
        self.code_order_tried = True
        self.journal.append({'kind': 'code_order', 'plan': plan})
        said = ("Wren's own check: the listings read already make an order in which every requirement of each line is "
                'stated under its own key, so Wren sets the cart to it and checks out.')
        tag = f'w{self.driver_turns}'
        if calls:
            batch = [{'id': f'{tag}_c{n}', 'type': 'function', 'function': {'name': cart_tool, 'arguments': _jdump(raw)}}
                     for n, raw in enumerate(calls)]
            self._relay_batch(said, batch)
            if self.run_closed:
                return True
        buy_args = dict(self.digest.get('buy_args') or {})
        self._relay_batch('', [{'id': f'{tag}_b', 'type': 'function', 'function': {'name': buyer, 'arguments': _jdump(buy_args)}}])
        return True

    def _after_failed_turn(self, idle: int) -> int | None:
        self.turn_failures += 1
        give_up = self.credit_gone or self.model_calls_made >= MODEL_CALL_CEILING or self.turn_failures > 2
        return None if give_up else idle

    def _clean_call(self, slot: int, raw: dict[str, Any]) -> dict[str, Any]:
        """A model tool call brought into one tidy shape: string id, string name, a JSON object as its arguments."""
        fn = raw.get('function')
        if not isinstance(fn, dict):
            fn = {}
        given = fn.get('arguments')
        if isinstance(given, dict):
            decoded = given
        elif isinstance(given, str) and given.strip():
            decoded = _first_json_object(given)
        else:
            decoded = {}
        if not isinstance(decoded, dict):
            decoded = {}
        call_id = raw.get('id') or f'k{self.driver_turns}_{slot}'
        return {'id': str(call_id), 'type': 'function',
                'function': {'name': str(fn.get('name') or ''), 'arguments': _jdump(decoded)}}

    @staticmethod
    def _reply_words(answer: dict[str, Any]) -> str:
        content = answer.get('content')
        if not isinstance(content, str):
            content = _jdump(content) if content else ''
        return content

    def _role_tools(self, *roles: str) -> tuple:
        return tuple(self.role_map.get(role) for role in roles)

    def _commit_before_interview(self, calls: list[dict[str, Any]]) -> bool:
        """Whether the model reaches for a binding call (cart, checkout, ending) while the interviewer has not run yet."""
        if self.interview_ran or not self.role_map.get('ask') or self.free_message_count() <= 0:
            return False
        binding = set(self._role_tools('cart', 'buy', 'refuse')) - {None}
        if not any(entry['function']['name'] in binding for entry in calls):
            return False
        return self.moves_left() > 8 and self.seconds_spent() <= ASKING_CUTOFF

    def _chat_user(self, text: str) -> None:
        self.convo_log.append({'role': 'user', 'content': text})

    def _chat_tool(self, call_id: str, text: str) -> None:
        self.convo_log.append({'role': 'tool', 'tool_call_id': call_id, 'content': text})

    def _chat_calls(self, said: str, calls: list[dict[str, Any]]) -> None:
        self.convo_log.append({'role': 'assistant', 'content': said or None, 'tool_calls': calls})

    def _ask_before_commit(self, said: str, calls: list[dict[str, Any]]) -> None:
        self._chat_calls(said, calls)
        for entry in calls:
            self._chat_tool(entry['id'], 'Not run (no step used): the shopper is asked before this.')
        news = self.conduct_interview()
        self._chat_user((news or 'The interviewer had no question to ask.') + '\n' + self.status_line())

    def _relay_batch(self, said: str, calls: list[dict[str, Any]]) -> None:
        self._chat_calls(said, calls)
        self.journal.append({'kind': 'turn', 'text': said[:300], 'calls': [entry['function'] for entry in calls]})
        *earlier, (last_id, last_text) = self._vet_and_send(calls)
        for call_id, text in earlier:
            self._chat_tool(call_id, text)
        self._chat_tool(last_id, last_text + '\n' + self.status_line())

    def _shrink_old_results(self) -> None:
        """When the chat has grown past its size limit, the long tool results in its older half shrink to their opening
        stretch plus the listings they mentioned."""
        size = sum(len(_jdump(entry.get('content'))) for entry in self.convo_log)
        if size <= TRANSCRIPT_TEXT_LIMIT:
            return
        tool_msgs = [entry for entry in self.convo_log if entry.get('role') == 'tool']
        for entry in tool_msgs[:max(1, len(tool_msgs) // 2)]:
            text = entry.get('content') or ''
            if len(text) > 600:
                entry['content'] = self._shortened_text(text)

    def _shortened_text(self, text: str) -> str:
        cited = dict.fromkeys(re.findall('"([A-Za-z0-9][A-Za-z0-9_\\-:.]{5,40})"', text))
        known = [key for key in cited if key in self.row_records][:20]
        return text[:400] + f"...(older result shortened; listings named in it: {', '.join(known)})"

    # ================================================================== checking a batch of proposed calls

    def _vet_and_send(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        """Check the proposed calls one after another and send those nothing holds back; every call gets one answer."""
        batch = self._merge_single_opens(calls)
        answers: list[tuple[str, str]] = []
        for entry in batch:
            answers.append((entry['id'], self._handle_call(_ProposedCall(entry, len(batch)))))
        return answers

    def _handle_call(self, job: _ProposedCall) -> str:
        return self._first_objection(job, self._VET_ORDER) or self._send_and_answer(job)

    def _first_objection(self, job: _ProposedCall, names: tuple) -> str:
        """Run the named checks in turn. A check answers with the text that holds the call back, or '' to let it
        pass; every hold text is non-empty, so the first non-empty answer stops the call there."""
        for name in names:
            verdict = getattr(self, name)(job)
            if verdict:
                return verdict
        return ''

    def _cart_now(self, joint: str = '\n') -> str:
        return joint + 'Current cart: ' + self.basket_brief()

    # --- checks that cost neither a step nor a model call

    def _vet_entry(self, job: _ProposedCall) -> str:
        if job.merged:
            return 'Answered by the side-by-side result above: Wren read these listings together in one step.'
        if self.run_closed:
            return 'Not run: the episode is already over.'
        if job.tool == self.role_map.get('ask'):
            return "Not run (no step used): Wren's interviewer sends every message to the shopper."
        usable = sorted({spec['function']['name'] for spec in self.offered_tool_specs()})
        if job.tool in self.schemas and job.tool in usable:
            return ''
        menu = ', '.join(usable)
        if job.tool not in self.schemas:
            return f'Not run (no step used): that tool does not exist. Available: {menu}.'
        return f'Not run (no step used): too few steps are left for that; available: {menu}.'

    def _vet_params(self, job: _ProposedCall) -> str:
        raw = job.arg_text
        if isinstance(raw, dict):
            decoded = raw
        else:
            decoded = _first_json_object(raw) if (raw or '').strip() else {}
        if decoded is None:
            return 'Not run (no step used): the arguments were not one JSON object.'
        issues: list[str] = []
        fitted = _fit_to_schema(decoded, self.schemas[job.tool], 'args', issues)
        if issues:
            return 'Not run (no step used): ' + '; '.join(issues[:6])
        fitted, objection = self.normalize_ids(job.tool, fitted)
        if objection:
            return 'Not run (no step used): ' + objection + '.'
        is_cart = job.tool == self.role_map.get('cart')
        job.params = self.whole_line_removal(job.tool, fitted) if is_cart else fitted
        job.params = self._fresh_search_from_start(job)
        self._note_cut_query(job, decoded)
        if is_cart:
            job.params = self._units_not_rows(job)
        return ''

    def _note_cut_query(self, job: _ProposedCall, sent: dict[str, Any]) -> None:
        """Once per episode: a search whose words passed the field's length limit was cut at a word boundary; say so,
        so later queries put the telling words first."""
        if job.tool != self.role_map.get('search') or self.cut_query_noted or not isinstance(sent, dict):
            return
        props = (self.schemas.get(job.tool) or {}).get('properties') or {}
        for key, value in sent.items():
            cap = (props.get(key) or {}).get('maxLength') if isinstance(props.get(key), dict) else None
            if isinstance(value, str) and isinstance(cap, int) and len(value) > cap:
                self.cut_query_noted = True
                job.side_notes += (f'\n[Wren] {key} takes at most {cap} characters, so the search went out as '
                                   f'{_jdump(job.params.get(key))}; put the most telling words first in later searches.')
                return

    def _units_not_rows(self, job: _ProposedCall) -> dict[str, Any]:
        """A cart addition that names a listing not held yet with a count of zero adds nothing: the count argument
        means units. Send the item's requested quantity instead (within the schema's maximum), and say so."""
        params = job.params
        listing = next((value for value in params.values() if isinstance(value, str) and value in self.row_records), None)
        if not listing or (self.basket_counts.get(listing) or 0) > 0:
            return params
        props = (self.schemas.get(job.tool) or {}).get('properties') or {}
        # only where an option field says the call adds: on a tool without one, a zero count may mean "take off"
        kinds = [key for key, spec in props.items() if isinstance(spec, dict) and isinstance(spec.get('enum'), list)]
        if not kinds or any(self._is_removal_value(props, key, params.get(key)) for key in kinds):
            return params
        zeros = [key for key, value in params.items()
                 if self._is_number(value) and not isinstance(value, bool) and value == 0
                 and isinstance(props.get(key), dict) and 'integer' in _declared_types(props[key])]
        if len(zeros) != 1:
            return params
        label_key = self.shelf_field_name()
        slot = self.good_for_shelf(self._shelf_text(listing, label_key)) if label_key else 0
        want = self.goods[slot - 1].count_wanted if slot else 1
        top = props[zeros[0]].get('maximum')
        if self._is_number(top):
            want = min(want, int(top))
        job.side_notes += (f'\n[Wren] {zeros[0]} counts units, so 0 would add nothing; Wren sent {want}, the '
                           'quantity wanted.')
        self.journal.append({'kind': 'units_not_rows', 'listing': listing, 'units': want})
        return {**params, zeros[0]: want}

    def _fresh_search_from_start(self, job: _ProposedCall) -> dict[str, Any]:
        """A search with new words starts at the first match: an argument learned to skip matches is left out (or
        set to 0 when the schema requires it). Paging on through the same search keeps it."""
        known = self.offset_args.get(job.tool)
        if not known or job.tool != self.role_map.get('search'):
            return job.params
        rest = {key: value for key, value in job.params.items() if key not in known}
        if any({key: value for key, value in past.items() if key not in known} == rest for past in self.recent_queries):
            return job.params
        needed = set((self.schemas.get(job.tool) or {}).get('required') or [])
        dropped = [key for key in known if self._is_number(job.params.get(key))
                   and job.params[key] > self._lowest_allowed(job.tool, key)]
        if not dropped:
            return job.params
        trimmed = {key: (self._lowest_allowed(job.tool, key) if key in dropped else value)
                   for key, value in job.params.items() if key not in dropped or key in needed}
        job.side_notes += (f"\n[Wren] {', '.join(dropped)} was reset for this new search: it skips that many matches "
                           'before the first one shown, so a new search starts from the beginning.')
        return trimmed

    def _vet_side_room(self, job: _ProposedCall) -> str:
        """A side-by-side request naming more listings than the store is known to show at once is cut to fit."""
        job.list_key = self.side_by_side_list_arg() if job.tool == self.role_map.get('compare') else None
        names = job.params.get(job.list_key) if job.list_key else None
        if not isinstance(names, list) or not self.side_fail_width:
            return ''
        room = self.side_by_side_width()
        if len(names) > room:
            rest = ', '.join(map(str, names[room:]))
            job.side_notes = (f"\n[Wren] Just the first {room} listings were sent, since the store's side-by-side view "
                              f'takes no more than {room}. Not shown yet: {rest}.')
            job.params = {**job.params, job.list_key: names[:room]}
        return ''

    def _vet_sold_target(self, job: _ProposedCall) -> str:
        job.id_key = self.lone_id_arg(job.tool)
        if not job.id_key or job.tool not in self._role_tools('open', 'offer'):
            return ''
        listing = str(job.params.get(job.id_key))
        if listing not in self.sold_out_steps:
            return ''
        return (f'Not run (no step used): {job.params[job.id_key]} sold out at step {self.sold_out_steps[listing]} and '
                'will not return. Replace its cart line with another listing that fits.')

    def _vet_ghost_removal(self, job: _ProposedCall) -> str:
        """Taking units off a listing the cart does not hold would only burn a step."""
        if job.tool != self.role_map.get('cart'):
            return ''
        counts = [v for v in job.params.values() if not isinstance(v, str)]
        if all((_coerce_int(v) or 0) >= 0 for v in counts):
            return ''
        cart_text = _jdump(self.basket_view) if self.basket_seen_at >= self.basket_changed_at else ''
        ghost = next((v for v in job.params.values()
                      if isinstance(v, str) and v in self.row_records and v not in self.basket_counts
                      and v not in cart_text and v not in self.last_basket_reply), None)
        if ghost is None:
            return ''
        return (f'Not run (no step used): the cart has no line for {ghost}, so nothing can be taken off.'
                + self._cart_now(' '))

    def _vet_finish_alone(self, job: _ProposedCall) -> str:
        job.finishing = bool(self.is_purchase_call(job.tool, job.params) or self.is_refusal_call(job.tool, job.params))
        if job.finishing:
            if job.batch_size > 1:
                return ('Not run (no step used): go through your other results first, then send the finishing call on '
                        'its own.')
            return ''
        if self.moves_left() > self.wrap_up_reserve():
            return ''
        if not (self.role_map.get('buy') or self.role_map.get('refuse')):
            return ''
        return ('Not run (no step used): the last steps are kept for the checkout (once the shopper has approved the '
                'basket, where that is needed) or for an ending without a purchase.')

    def _vet_repeat(self, job: _ProposedCall) -> str:
        job.fingerprint = job.tool + _jdump(job.params)
        repeated = (not job.finishing and job.tool != self.role_map.get('cart')
                    and job.fingerprint in self.recent_marks)
        if repeated:
            return ('Not run (no step used): the very same call was just sent and nothing has changed since; its result '
                    'is above.')
        job.read_tools = set(self._role_tools('search', 'open', 'offer', 'compare')) - {None}
        return self._remembered_view(job) if job.tool in job.read_tools else ''

    def _remembered_view(self, job: _ProposedCall) -> str:
        """A reading call whose answer is already known: the cached result, or the stored record of a listing this view
        can tell nothing new about."""
        hit = self.read_cache.get(job.fingerprint)
        if hit and hit[0] == self.revision:
            return (f'Not run (no step used): the same call at step {hit[1]} gave the result below, and nothing has '
                    'changed after it:\n' + hit[2][:3000])
        if not job.id_key or job.tool == self.role_map.get('compare'):
            return ''
        listing = str(job.params.get(job.id_key))
        if not self.view_same_as_before(job.tool, listing):
            return ''
        return (f'Not run (no step used): since the last change this view has nothing more to show about '
                f'{job.params[job.id_key]}. What is on record for it: ' + self.listing_blurb(listing))

    def _vet_side_overlap(self, job: _ProposedCall) -> str:
        """Leave out the listings a side-by-side view cannot say anything new about; hold the call if none remain."""
        names = job.params.get(job.list_key) if job.list_key else None
        if not names or not isinstance(names, list):
            return ''
        new = [name for name in names if not self.view_same_as_before(job.tool, str(name))]
        if not new:
            records = '\n'.join(f'{name}: {self.listing_blurb(str(name))}' for name in names)[:4000]
            return ('Not run (no step used): since the last change this view has nothing more to show about these '
                    'listings. What is on record:\n' + records)
        if len(new) < len(names):
            left_out = ', '.join(str(name) for name in names if name not in new)
            job.side_notes += f'\n[Wren] Left out {left_out}: shown in full before, with no change since.'
            job.params = {**job.params, job.list_key: new}
            job.fingerprint = job.tool + _jdump(job.params)
        return ''

    def _unsettled_borrowed(self) -> tuple:
        active = {need for _, need in self.needs_in_force()}
        return tuple(sorted({need for _, need in self.borrowed_needs
                             if need in active and need not in self.borrowed_values}))

    def _vet_trace_first(self, job: _ProposedCall) -> str:
        """Ahead of a cart addition, while a requirement in force still takes its value from another listing, follow
        those links (opening the pages they point to) so the line gets chosen against the borrowed value. Twice at
        most, and a second time only after more pages have been opened."""
        if job.tool != self.role_map.get('cart') or job.finishing:
            return ''
        if self.moves_left() <= self.wrap_up_reserve() + 8:
            return ''
        if any(self._is_number(v) and v <= 0 for v in job.params.values()):
            return ''
        pending = self._unsettled_borrowed()
        stamp = (pending, len(self.paged_ids))
        if not pending or stamp in self.early_trace_marks or len(self.early_trace_marks) >= 2:
            return ''
        self.early_trace_marks.add(stamp)
        report, blocking = self.trace_borrowed_values()
        settled = [need for need in pending if need in self.borrowed_values]
        grew = len(self.paged_ids) > stamp[1]
        if not report.strip() or not (blocking or settled or grew):
            return ''
        self.journal.append({'kind': 'pointer_first', 'settled': len(settled), 'blocking': blocking})
        return ('Not run yet (no step used). Ahead of any cart addition, the requirements whose value comes from another '
                'listing have been followed up.\n' + report + '\nChoose the line by those values, then add it.'
                + self._cart_now())

    # --- pauses on a finishing call (these may spend steps and model calls)

    def _vet_finish_pauses(self, job: _ProposedCall) -> str:
        if not job.finishing or self.moves_left() <= 2:
            return ''
        if self.is_purchase_call(job.tool, job.params):
            return self._first_objection(job, self._BUY_PAUSES)
        return self._first_objection(job, self._REFUSAL_PAUSES) if self.moves_left() > 5 else ''

    def _pause_wider_hunt(self, job: _ProposedCall) -> str:
        note = self.wider_hunt_pause()
        return note + self._cart_now() if note else ''

    def _peek_changed_cart(self) -> str:
        """View the cart when it changed after the last view and four steps or more are left (the view, the checkout,
        and one step to answer a refused checkout before trying again). Returns what the view revealed, '' when it
        revealed nothing or no view was taken; nearer the limit the checkout's own answer reports any change."""
        viewer = self.role_map.get('cart_view')
        due = self.basket_touched and self.basket_seen_at < self.basket_changed_at and viewer
        if not due or self.moves_left() < 4:
            return ''
        issues: list[str] = []
        empty_args = _fit_to_schema({}, self.schemas.get(viewer), 'args', issues)
        if issues:
            return ''
        view = self.shop_call(viewer, empty_args)
        if view is None:
            return ''
        self.basket_view = view.get('observation')
        self.basket_seen_at = self.moves_made
        return self.digest_reply(viewer, self.basket_view)

    def _pause_cart_look(self, job: _ProposedCall) -> str:
        """Read the carted offers again, and view the cart itself when it changed since it was last viewed."""
        job.refreshed = self.reread_basket_offers()
        if self.notice_says_sold:
            self.find_sold_line()
        news = self._peek_changed_cart()
        if not news:
            return ''
        self.pending_notes = []
        return ('Not run yet (no step used). Viewing the cart showed a change: ' + news + self._cart_now()
                + '\nRespond to the change, fix the cart where needed, then check out.')

    def _pause_sold_lines(self, job: _ProposedCall) -> str:
        advice = self.sellout_guidance()
        if advice:
            return 'Not run yet (no step used). ' + advice + self._cart_now()
        if self.sold_pause_count >= 2:
            return ''
        gone = [lid for lid in self.basket_ids() if lid in self.sold_out_steps]
        if not gone:
            return ''
        self.sold_pause_count += 1
        return (f"Not run yet (no step used). {', '.join(gone)} in the cart is sold out: replace that line first."
                + self._cart_now())

    def _pause_borrowed_trail(self, job: _ProposedCall) -> str:
        job.trail, blocking = self.trace_borrowed_values()
        if not blocking:
            return ''
        return 'Not run yet (no step used). ' + job.trail + self._cart_now()

    def _pause_fit(self, job: _ProposedCall) -> str:
        no_room = self.moves_left() <= self.wrap_up_reserve() + 3
        clashes = [] if self.fit_pause_count >= 2 or no_room else self.fit_clashes()
        if not clashes:
            return ''
        self.fit_pause_count += 1
        self.journal.append({'kind': 'fit_hold', 'faults': clashes[:4]})
        return ('Not run yet (no step used). A part in the cart does not fit the device in the cart: '
                + '; '.join(clashes[:3]) + '. Swap the part (or else the device), then check out.' + self._cart_now())

    def _repair_counts_alone(self, faults: list[str]) -> tuple[str, list[str]]:
        """When every fault is only a unit count, Wren fixes the counts itself; returns (hold text or '', faults left)."""
        if not faults or not all(' is held at ' in fault for fault in faults):
            return '', faults
        repaired = self.fix_line_counts(self.count_repairs())
        if not repaired:
            return '', faults
        faults = self.basket_line_faults()
        if faults:
            return '', faults
        self.line_pause_count += 1
        return ('Not run yet (no step used). Wren corrected the unit counts in the cart by itself: '
                + '; '.join(repaired) + '. Fix any line the store refused, then check out.' + self._cart_now()), faults

    def _pause_counts(self, job: _ProposedCall) -> str:
        """Lines that differ from the request: unit counts Wren repairs by itself, any other gap goes to the model."""
        if self.line_pause_count >= 2 or self.moves_left() <= self.wrap_up_reserve() + 2:
            return ''
        fixed, faults = self._repair_counts_alone(self.basket_line_faults())
        if fixed or not faults:
            return fixed
        self.line_pause_count += 1
        self.journal.append({'kind': 'line_hold', 'faults': faults[:6]})
        return ('Not run yet (no step used). The cart does not yet match the request: ' + '; '.join(faults[:6])
                + '. Fix those lines (adding or taking off units the way the cart tool does it), then check out.'
                + self._cart_now())

    def _pause_proven_set(self, job: _ProposedCall) -> str:
        """At the model's checkout, once per cart state: when the read listings make a fully proven order (every
        requirement of every line stated under its own key) and the cart is not one, because a line is unproven or,
        on a lowest-price pick, the proven order costs less, Wren sets the cart to that order itself and lets the
        checkout go on through the remaining checks. Nothing changes when the cart is already fully proven."""
        mark = ('proven_set', self.basket_fingerprint())
        if mark in self.capped_queries or self.moves_left() <= self.wrap_up_reserve() + 2:
            return ''
        self.capped_queries.add(mark)
        plan = self.proven_set()
        held = {lid: units for lid, units in self.basket_counts.items() if units > 0}
        if not plan or plan == held:
            return ''
        label_key = self.shelf_field_name()
        slot_of = {lid: self.good_for_shelf(self._shelf_text(lid, label_key)) for lid in held}
        unproven = [lid for lid in held if not slot_of[lid] or not self._rv_proves(lid, slot_of[lid])]
        cost = lambda lines: sum((self.price_of_listing(lid) or 0.0) * units for lid, units in lines.items())
        pick = self.digest.get('pick')
        if pick in ('most', 'least'):
            return ''
        dearer = pick == 'cheapest' and cost(plan) < cost(held) - 1e-09
        # on a lowest-price pick a line the strict reading cannot prove may still be the accepted cheapest (a value
        # stated only in its title): swap only to a cheaper proven order there
        if pick == 'cheapest':
            unproven = []
        if not unproven and not dearer:
            return ''
        cart_tool = self.role_map.get('cart')
        calls = self.basket_edit_calls(plan, [lid for lid in held if lid not in plan]) if cart_tool else None
        if not calls or len(calls) + self.wrap_up_reserve() + 1 > self.moves_left():
            return ''
        sent = self._push_cart_calls(cart_tool, calls)
        self.journal.append({'kind': 'proven_swap', 'from': held, 'to': plan, 'sent': sent is not None})
        if sent is None:
            return 'Not run yet. Wren tried to set the cart to a fully proven order, but a cart call failed.' + self._cart_now()
        why = 'some line did not state every requirement under its own key' if unproven else 'it costs less'
        job.side_notes += (f"\n[Wren] Before this checkout Wren set the cart to {', '.join(f'{lid} x{units}' for lid, units in plan.items())}"
                           f': every requirement of each line is stated under its own key in the listings read, and '
                           f'{why} (one step per cart call).')
        return ''

    def _pause_unshown_needs(self, job: _ProposedCall) -> str:
        """Hold a checkout once per cart state when a cart line's read attributes do not show a plain requirement of
        its item, naming the line and the requirement."""
        if self.moves_left() <= self.wrap_up_reserve() + 3:
            return ''
        label_key = self.shelf_field_name()
        mark = ('unshown', self.basket_fingerprint())
        if not label_key or mark in self.capped_queries:
            return ''
        faults = []
        for line in self.basket_ids():
            slot = self.good_for_shelf(self._shelf_text(line, label_key))
            gaps = self._rv_unshown_needs(line, slot) if slot else []
            if gaps:
                faults.append(f"{line} (item {slot}) does not show " + '; '.join(f'"{need}"' for need in gaps[:3]))
            cap, price = self.item_caps.get(slot), self.price_of_listing(line)
            if slot and cap is not None and price is not None and price > cap:
                faults.append(f"{line} (item {slot}) costs {price:g}, over the shopper's cap of {cap:g} for one unit")
        if not faults:
            return ''
        self.capped_queries.add(mark)
        self.journal.append({'kind': 'unshown_hold', 'faults': faults[:4]})
        return ('Not run yet (no step used). The attributes read for these cart lines do not show a requirement of '
                'their item: ' + ' | '.join(faults[:4]) + '. A requirement holds only when the listing\'s own '
                'attribute gives that value. Replace each such line with a listing whose attributes show it (search '
                'with the value and read the candidates); keep the line only if no listing in the store shows it.'
                + self._cart_now())

    def _pause_cheaper_fit(self, job: _ProposedCall) -> str:
        """A cheapest pick about to check out while a read row on the same shelf costs less than a cart line and its
        attributes already show every plain requirement of that item: hold once per cart state and name the swap."""
        if self.digest.get('pick') != 'cheapest' or self.moves_left() <= self.wrap_up_reserve() + 2:
            return ''
        label_key = self.shelf_field_name()
        mark = ('cheaper_fit', self.basket_fingerprint())
        if not label_key or mark in self.capped_queries:
            return ''
        swaps = []
        for line in self.basket_ids():
            ceiling = self.price_of_listing(line)
            slot = self.good_for_shelf(self._shelf_text(line, label_key))
            if ceiling is None or not slot:
                continue
            better = [(lid, price) for lid, price in self._cheaper_rivals(line, ceiling, label_key)
                      if lid in self.page_records and self._rv_plainly_meets(lid, slot)]
            if better:
                lid, price = better[0]
                swaps.append(f'{lid} ({price:g}) in place of {line} ({ceiling:g}) for item {slot}')
                self.cheaper_named[line] = lid
        if not swaps:
            return ''
        self.capped_queries.add(mark)
        self.journal.append({'kind': 'cheaper_fit_hold', 'swaps': swaps[:4]})
        return ('Not run yet (no step used). The pick rule is the lowest price, and a listing already read costs less '
                'than a cart line while its attributes show every plain requirement of that item: '
                + '; '.join(swaps[:4]) + '. Swap each such line (take the dearer one out, put the cheaper one in at the '
                'same quantity) unless a limit, exclusion or borrowed value in its record rules it out, then check out.'
                + self._cart_now())

    def _pause_audit(self, job: _ProposedCall) -> str:
        lines = self.basket_ids()
        spare = 1 if len(lines) >= 2 else 3
        opened = self.read_traits_of(lines, self.wrap_up_reserve() + spare)
        self.chase_part_refs(lines, self.wrap_up_reserve() + 1)
        opened += self.read_lower_priced_rows()
        verdict = self.audit_basket()
        if not verdict:
            return ''
        swapped = self._agreed_swap()
        if swapped:
            return swapped
        reads = ''
        if opened:
            reads = ('To run the check, Wren read the attributes of these listings (one step each):\n'
                     + '\n'.join(opened) + '\n')
        return 'Not run yet. ' + job.trail + reads + verdict + self._cart_now()

    def _agreed_swap(self) -> str:
        """Once: the cart check rejected the cart and names, as the better choice, the very listing the cheaper-listing
        hold named for a cart line, yet the model came back to checkout without swapping. Wren swaps that line itself
        (same quantity) when the steps allow it and the checkout, and says so; '' when it did nothing."""
        if self.agreed_swap_done or not self.cheaper_named:
            return ''
        last = next((entry for entry in reversed(self.journal) if entry.get('kind') == 'check'), None)
        better = [str(lid) for lid in _listify((last or {}).get('got', {}).get('better') if isinstance((last or {}).get('got'), dict) else [])]
        held = dict(self.basket_counts)
        pairs = [(line, lid) for line, lid in self.cheaper_named.items()
                 if line in held and held[line] > 0 and lid in better and not self.sold_out_now(lid)]
        cart_tool = self.role_map.get('cart')
        if not pairs or not cart_tool:
            return ''
        wanted = {key: units for key, units in held.items() if units > 0}
        for line, lid in pairs:
            wanted[lid] = wanted.pop(line)
        calls = self.basket_edit_calls(wanted, [line for line, _ in pairs])
        if not calls or len(calls) + self.wrap_up_reserve() + 1 > self.moves_left():
            return ''
        self.agreed_swap_done = True
        sent = self._push_cart_calls(cart_tool, calls)
        self.journal.append({'kind': 'agreed_swap', 'pairs': pairs, 'sent': bool(sent)})
        if sent is None:
            return ''
        named = '; '.join(f'{lid} in place of {line}' for line, lid in pairs)
        return ('Not run yet. The lowest-price check and the cart check both named a cheaper listing that states every '
                f'requirement, so Wren swapped the line itself ({named}; one step per cart call). Look at the cart '
                'once if needed, then check out.' + self._cart_now())

    def _pause_reread(self, job: _ProposedCall) -> str:
        if not job.refreshed or self.model_cost <= SPEND_COMFORT:
            return ''
        return ('Not run yet (no step used). Wren has just read the carted listings again:\n' + job.refreshed
                + '\nIf every line still shows stock and a price inside the budget, check out again; otherwise replace '
                'that line first.')

    def _pause_proven_decline(self, job: _ProposedCall) -> str:
        """Once, before an ending without a purchase: the read listings already make an order in which every
        requirement of each line is stated under its own key. Name it; ending without a purchase then earns nothing."""
        if self.proven_decline_held:
            return ''
        plan = self.proven_set()
        if not plan:
            return ''
        self.proven_decline_held = True
        total = sum((self.price_of_listing(lid) or 0.0) * units for lid, units in plan.items())
        self.journal.append({'kind': 'proven_decline_hold', 'plan': plan})
        return ('Not run yet (no step used). The listings read already make a whole order in which every requirement of '
                'each line is stated under its own key: ' + ', '.join(f'{lid} x{units}' for lid, units in plan.items())
                + f' (total {total:g}). Put exactly these lines in the cart and check out; end without a purchase only '
                'if one of their records clearly breaks a requirement, and then say which.' + self._cart_now())

    def _pause_refusal_trail(self, job: _ProposedCall) -> str:
        """Once, before an ending without a purchase: while a requirement still takes its value from another listing
        and pages were opened since the chains were last followed, follow them again, since a chain followed before
        the pointed page was opened never learned where it leads."""
        pending = self._unsettled_borrowed()
        if not pending or self.refusal_trail_done or len(self.paged_ids) <= self.trace_paged_mark:
            return ''
        if self.moves_left() <= self.wrap_up_reserve() + 4:
            return ''
        self.refusal_trail_done = True
        kept = self.trace_rounds
        self.trace_rounds = 0
        try:
            report, _ = self.trace_borrowed_values()
        finally:
            self.trace_rounds = kept
        self.journal.append({'kind': 'refusal_trail', 'settled': [n for n in pending if n in self.borrowed_values]})
        if not report.strip():
            return ''
        return ('Not run yet (no step used). Before ending without a purchase, the requirements that take their value '
                'from another listing were followed again over the pages opened since:\n' + report
                + '\nSearch and buy by those values; end without a purchase only if no listing meets them.' + self._cart_now())

    def _pause_refusal_audit(self, job: _ProposedCall) -> str:
        review = self.audit_refusal()
        return 'Not run yet (no step used). ' + review if review else ''

    def _pause_leads(self, job: _ProposedCall) -> str:
        """Leads from the review hold an ending without purchase twice at most: first by naming them, then by a separate
        check of the first lead, which backs that lead when it passes."""
        if not self.audit_leads or self.moves_left() <= 4:
            return ''
        if self.refusal_block_count == 0:
            return self._name_review_leads()
        if self.refusal_block_count == 1:
            return self._check_first_lead()
        return ''

    def _name_review_leads(self) -> str:
        self.refusal_block_count = 1
        self.journal.append({'kind': 'refuse_blocked', 'leads': list(self.audit_leads)})
        return ('Not run yet (no step used). The review found listings that seem to fit: ' + ', '.join(self.audit_leads)
                + '. Open them where needed, put the best fitting one in the cart (by the pick rule) and check out. End '
                'without a purchase only when each of them plainly fails a requirement; in that case name in your text '
                'the requirement it fails and send the ending call again.')

    def _check_first_lead(self) -> str:
        self.refusal_block_count = 2
        lead = self.audit_leads[0]
        # with several items the lead is judged alongside the lines already carted for the other items
        planned = list(dict.fromkeys(self.basket_ids() + [lead])) if len(self.goods) > 1 else [lead]
        self.audit_basket(planned, SPEND_CEILING)
        passed = self.audit_verdicts.get(self.audit_fingerprint(planned))
        if not passed or lead not in self.audit_leads:
            return ''
        self.backed_lead = lead
        self.journal.append({'kind': 'refuse_blocked', 'confirmed': lead})
        return (f'Not run yet (no step used). An independent check of the records of {lead} finds every requirement '
                'met. Put it in the cart at the requested quantity and check out; end without a purchase only if its '
                'record clearly breaks a requirement.')

    # --- the last checks before the store sees the call

    def _vet_late_refusal(self, job: _ProposedCall) -> str:
        """Close to the end, an ending without purchase goes back once when fitting lines were already put together."""
        if not self.is_refusal_call(job.tool, job.params) or self.moves_left() > 5 or self.late_refusal_held:
            return ''
        built = [lid for lid in self.composed_lines if not self.sold_out_now(lid)]
        trusted = self._cart_view_current()
        carted = self.basket_ids()
        missing = [lid for lid in built if lid not in carted or (lid in self.dropped_needs and not trusted)]
        if len(built) < len(self.goods) or len(missing) + 1 > self.moves_left():
            return ''
        self.late_refusal_held = True
        self.journal.append({'kind': 'late_decline_held', 'lines': built, 'missing': missing})
        bought = '' if missing else self._late_checkout()
        if bought:
            return bought
        if missing:
            fix = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out."
        else:
            fix = 'Check out the cart as it is.'
        return ('Not run yet (no step used). An ending without a purchase earns nothing while a fitting listing exists, '
                'and running short of steps does not show that none does. ' + fix + ' End without a purchase only when '
                'your search found a requested item for which no listing meets the requirements; then send the ending '
                'call again.')

    def _late_checkout(self) -> str:
        """Near the step limit with a line in the cart for every item: Wren checks out itself instead of passing the
        ending without a purchase on, once. Running out of steps shows nothing about whether a fitting listing exists,
        and a whole cart put together on purpose usually is one; an ending without a purchase earns nothing then."""
        buyer = self.role_map.get('buy')
        if self.late_checkout_tried or not buyer or self.moves_left() < 1:
            return ''
        self.late_checkout_tried = True
        stop, said = self._forced_order(buyer, explain_refusal=True)
        self.journal.append({'kind': 'late_checkout', 'closed': self.run_closed, 'lines': self.basket_ids()})
        if self.run_closed:
            return ('Not run (Wren acted instead): this close to the step limit an ending without a purchase earns '
                    'nothing while the cart holds a line for every item, so Wren checked out.' + _CLOSED_MARK)
        if stop and said:
            return ('Not run (Wren tried a checkout instead, since the cart holds a line for every item), but ' + said
                    + self._cart_now())
        return ''

    def _vet_unused_searches(self, job: _ProposedCall) -> str:
        if not self.is_refusal_call(job.tool, job.params) or self.audit_leads or self.refusal_pushed:
            return ''
        left = self.moves_left()
        if left <= self.step_cap // 3:
            return ''
        self.refusal_pushed = True
        return (f"Not run yet (no step used). {left} steps are left, so try searches you have not run yet before ending: "
                "list each requested item's category and keep paging with the token or offset the store returns, search "
                'single value words and other names for the property, and open the listings that might fit. End without '
                'a purchase only after that.')

    def _vet_backed_lead(self, job: _ProposedCall) -> str:
        if not self.is_refusal_call(job.tool, job.params):
            return ''
        outcome = self.purchase_backed_lead()
        if not outcome:
            job.params = self.quiet_refusal_args(job.tool, job.params)
            return ''
        return outcome + (_CLOSED_MARK if self.run_closed else '')

    def _vet_last_question(self, job: _ProposedCall) -> str:
        if not self.is_purchase_call(job.tool, job.params):
            return ''
        asked = self.final_question_pass()
        return 'Not run yet. ' + asked + self._cart_now() if asked else ''

    def _vet_procedure(self, job: _ProposedCall) -> str:
        if not job.finishing or self.moves_left() < 2:
            return ''
        phase = 'order' if self.is_purchase_call(job.tool, job.params) else 'end'
        passed, remark = self.run_procedure_gate(phase)
        if not passed:
            return 'Not run yet. ' + remark + self._cart_now()
        if remark:
            job.side_notes = '\n[Wren] ' + remark + job.side_notes
        return ''

    # ================================================================== sending a call and building the answer

    def _send_and_answer(self, job: _ProposedCall) -> str:
        reply = self.shop_call(job.tool, job.params)
        if reply is None:
            return 'The call did not get through (network or server trouble). It may be sent again.'
        seen = reply.get('observation')
        retried = self._rerun_from_start(job, seen)
        if retried is not None:
            reply, seen = retried, retried.get('observation')
        self._learn_side_width(job, seen)
        self._absorb_call(job, seen)
        role_bound = {tool for role, tool in self.role_map.items() if role != 'search'}
        browsing = not job.finishing and job.tool not in role_bound
        more_pages = self.fetch_more_pages(job.tool, job.params, seen) if browsing else ''
        self._learn_from_single_view(job.tool, seen)
        self._record_store_effects(job, reply, seen)
        job.side_notes += self._cheaper_rows_after_cart(job)
        more_pages += self._shorter_retry(job, seen)
        more_pages += self._open_leading_rows(job, seen)
        return self._build_answer(job, reply, seen, more_pages, browsing)

    def _shorter_retry(self, job: _ProposedCall, seen: Any) -> str:
        """A words search of three or more words that the store matched with nothing at all is sent again with just its
        first two words (one step, no model turn): a store that wants every word in a listing finds nothing for a
        long query far more often than for a short one. Three times an episode at most, never twice for the same
        words, and only while the closing steps stay safe."""
        if job.finishing or job.tool != self.role_map.get('search') or self.run_closed or not isinstance(seen, dict):
            return ''
        if self.short_retries >= 3 or self.records_in_reply(seen) or self.moves_left() <= self.wrap_up_reserve() + 8:
            return ''
        for value in seen.values():
            if isinstance(value, (list, dict, str)) and value:
                return ''
            if self._is_number(value) and not isinstance(value, bool) and value > 0:
                return ''
        plan = self.words_query_plan() or self._lone_text_query_plan()
        if not plan or plan[0] != job.tool:
            return ''
        field = plan[2]
        text = job.params.get(field)
        if not isinstance(text, str) or len(text.split()) < 3:
            return ''
        skips = self.offset_args.get(job.tool, set())
        args = {key: value for key, value in job.params.items() if key not in skips}
        # a model or part code is often printed only where the search does not look (a description): drop codes
        # first, else keep the first two words
        uncoded = re.sub(r'\s+', ' ', re.sub('(?:' + RANGE_PART_RX + '|' + DEVICE_CODE_RX + ')', ' ', text)).strip()
        args[field] = uncoded if uncoded and uncoded != text.strip() else ' '.join(text.split()[:2])
        if self._words_sent_before(job.tool, args):
            return ''
        self.short_retries += 1
        sent, reply = self._send_fitted(job.tool, args)
        if reply is None:
            return ''
        observed = reply.get('observation')
        self.digest_reply(job.tool, observed)
        if self.records_in_reply(observed):
            self.recent_queries = (self.recent_queries + [dict(sent)])[-8:]
        self.journal.append({'kind': 'shorter_search', 'from': text[:80], 'to': args[field],
                             'rows': len(self.records_in_reply(observed))})
        return (f'\n[Wren] No listing holds every one of those words, so Wren sent the same search with just '
                f'{_jdump(args[field])} (one step):\n' + self.show_reply(observed))

    def _open_leading_rows(self, job: _ProposedCall, seen: Any) -> str:
        """Right after a search, while steps are plentiful and time is not yet short: open the first few unread,
        buyable rows that sit on the shelf of a requested item still lacking a cart line and fit the budget (one
        step each), so their attributes arrive without a model turn spent on opening them."""
        if job.finishing or job.tool != self.role_map.get('search') or self.run_closed:
            return ''
        if self.moves_left() <= self.wrap_up_reserve() + 20 or self.seconds_spent() >= RUSH_SECONDS:
            return ''
        label_key = self.shelf_field_name()
        lacking = set(self._goods_without_line()) if self.basket_counts else set(range(1, len(self.goods) + 1))
        if not label_key or not lacking:
            return ''
        cap = self.money_cap if self._is_number(self.money_cap) else None
        picks = []
        for lid, _ in self.records_in_reply(seen):
            if lid in self.page_records or lid in picks or not self.is_purchasable(lid):
                continue
            price = self.price_of_listing(lid)
            if cap is not None and price is not None and price > cap:
                continue
            if self.good_for_shelf(self._shelf_text(lid, label_key)) in lacking:
                picks.append(lid)
            if len(picks) == 3:
                break
        texts = self.read_traits_of(picks, self.wrap_up_reserve() + 18) if picks else []
        if not texts:
            return ''
        self.journal.append({'kind': 'lead_rows_opened', 'ids': picks[:len(texts)]})
        return ('\n[Wren opened the leading rows of this search that could fill a still-missing item, one step each; '
                'judge them from these attributes instead of opening them again]\n' + '\n'.join(texts))

    def _lowest_allowed(self, tool: str, key: str) -> int:
        """The smallest whole value the schema allows for an argument (its minimum, else 0)."""
        spec = ((self.schemas.get(tool) or {}).get('properties') or {}).get(key)
        low = spec.get('minimum') if isinstance(spec, dict) else None
        return int(low) if self._is_number(low) else 0

    def _offset_suspects(self, job: _ProposedCall, seen: Any) -> list[str]:
        """Positive whole-number arguments of a search that counted matches yet showed no row: such an argument
        skips that many matches (an offset), whatever its name suggests. Money arguments are never suspects."""
        if job.tool != self.role_map.get('search') or not isinstance(seen, dict) or self.records_in_reply(seen):
            return []
        if not self._counted_matches(job.tool, seen):
            return []
        props = (self.schemas.get(job.tool) or {}).get('properties') or {}

        def suspect(key: str, value: Any) -> bool:
            spec = props.get(key)
            about = key + ' ' + str(spec.get('description') or '') if isinstance(spec, dict) else ''
            return (isinstance(spec, dict) and 'integer' in _declared_types(spec) and self._is_number(value)
                    and value > self._lowest_allowed(job.tool, key) and not re.search(CASH_ARG_RX, about, re.I))
        return [key for key, value in job.params.items() if suspect(key, value)]

    def _counted_matches(self, tool: str, seen: dict[str, Any]) -> int:
        """The match count a search reply states: the count field of the layout already learned (no model call
        here), else a top-level whole number under a count-like key; 0 when none."""
        shape = self.reply_shapes.get(tool)
        named = shape.get('count') if isinstance(shape, dict) else None
        values = [seen.get(named)] if named else [
            value for key, value in seen.items() if re.search(r'count|total|found|hits?\b|match', str(key), re.I)]
        return next((value for value in values if self._is_number(value) and int(value) == value and value > 0), 0)

    def _words_sent_before(self, tool: str, params: dict[str, Any]) -> bool:
        """Whether an earlier search sent these same arguments, whole numbers aside (paging on through it)."""
        def gist(args: dict[str, Any]) -> dict[str, Any]:
            return {key: value for key, value in args.items() if isinstance(value, bool) or not self._is_number(value)}
        mine = gist(params)
        return any(gist(past) == mine for past in self.recent_queries)

    def _positioned_past_start(self, job: _ProposedCall, seen: Any) -> list[str]:
        """A search with new words that set a whole-number argument above its lowest value, when the reply shows the
        argument placed the page: the next-page value it gives is the argument plus the rows shown (it skips that many
        matches) or plus one (a page number), or, on a last page, the match count is the argument plus the rows shown.
        Such a search began past its best matches. Read from the learned layout, else from next-page-like keys; no
        model call. Money arguments never count."""
        if job.tool != self.role_map.get('search') or not isinstance(seen, dict):
            return []
        shown = len(self.records_in_reply(seen))
        if not shown or self._words_sent_before(job.tool, job.params):
            return []
        props = (self.schemas.get(job.tool) or {}).get('properties') or {}

        def placed(key: str) -> bool:
            spec = props.get(key)
            value = job.params.get(key)
            if not isinstance(spec, dict) or 'integer' not in _declared_types(spec) or isinstance(value, bool):
                return False
            about = key + ' ' + str(spec.get('description') or '')
            return (self._is_number(value) and value > self._lowest_allowed(job.tool, key)
                    and not re.search(CASH_ARG_RX, about, re.I))
        candidates = [key for key in job.params if placed(key)]
        if not candidates:
            return []
        shape = self.reply_shapes.get(job.tool)
        if isinstance(shape, dict) and shape.get('next') and shape.get('next_arg'):
            candidates = [key for key in candidates if key == shape['next_arg']]
            next_keys = [shape['next']]
        else:
            next_keys = [key for key, value in seen.items()
                         if re.search(r'next|page|cursor|offset|more|continu', str(key), re.I)
                         and (value is None or (self._is_number(value) and not isinstance(value, bool)))]
        if len(candidates) != 1 or not next_keys:
            return []
        start = job.params[candidates[0]]
        count = self._counted_matches(job.tool, seen)
        for key in next_keys:
            following = seen.get(key)
            if following is None:
                if count and count == start + shown:
                    return candidates
            elif following in (start + shown, start + 1):
                return candidates
        return []

    def _rerun_from_start(self, job: _ProposedCall, seen: Any) -> dict[str, Any] | None:
        """A search that counted matches but showed none because an argument skipped past them, or a search with new
        words that the reply shows began past its first matches: remember that argument as one that places the page
        and, the first time, send the search again from the start (one more step)."""
        suspects = self._offset_suspects(job, seen)
        emptied = bool(suspects)
        if not suspects:
            suspects = self._positioned_past_start(job, seen)
        if not suspects:
            return None
        known = self.offset_args.setdefault(job.tool, set())
        fresh = [key for key in suspects if key not in known]
        known.update(suspects)
        self.journal.append({'kind': 'offset_arg', 'args': suspects, 'rerun': bool(fresh), 'empty': emptied})
        names = ', '.join(suspects)
        tail = (f' Leave {names} out of a new search; use it only to page further through the same search.')
        if emptied:
            first_try = (f'The first try counted matches but showed none, because {names} skips that many matches '
                         'before the first one shown (an offset, not a size).')
        else:
            first_try = (f'The first try began past the best matches: {names} sets where the page starts (the next '
                         'page the store offered shows it), not how many rows come back.')
        if not fresh or self.moves_left() <= self.wrap_up_reserve() + 2 or self.run_closed:
            if emptied:
                job.side_notes += f'\n[Wren] This search counted matches but showed none: {names} skipped past them.' + tail
            else:
                job.side_notes += f'\n[Wren] {first_try}' + tail
            return None
        needed = set((self.schemas.get(job.tool) or {}).get('required') or [])
        again = {key: (self._lowest_allowed(job.tool, key) if key in suspects else value)
                 for key, value in job.params.items() if key not in suspects or key in needed}
        second = self.shop_call(job.tool, again)
        if second is None:
            return None
        job.params = again
        job.side_notes += (f'\n[Wren] {first_try} Wren ran the search again from the start (one more step); the rows '
                           'below come from that second try.' + tail)
        return second

    def _cheaper_rows_after_cart(self, job: _ProposedCall) -> str:
        """For a cheapest pick, right after a cart addition: list that line's category below its price (one step, two
        when the rows spill onto a second page) and name the cheaper buyable rows found, lowest first, so that the
        cheapest choice is settled while steps remain instead of at checkout."""
        if job.tool != self.role_map.get('cart') or job.finishing or self.digest.get('pick') != 'cheapest':
            return ''
        label_key = self.shelf_field_name()
        line = next((v for v in job.params.values()
                     if isinstance(v, str) and (self.basket_counts.get(v) or 0) > 0), None)
        ceiling = self.price_of_listing(line) if line else None
        if not label_key or ceiling is None or self.moves_left() <= self.wrap_up_reserve() + 4:
            return ''
        before = len(self.capped_queries)
        if self._price_capped_browser() is None:
            if not self._sweep_line_query(line, ceiling, label_key):
                return ''
            self._read_cheapest_rivals(line, ceiling, label_key)
            return self._cheaper_rows_note(line, ceiling, label_key, swept=True)
        worded = self._capped_word_search(line, ceiling, label_key)
        if not worded or self.moves_left() > self.wrap_up_reserve() + 7:
            self._browse_under_line_prices([line])
        if not worded and len(self.capped_queries) == before:
            return ''
        self._read_cheapest_rivals(line, ceiling, label_key)
        return self._cheaper_rows_note(line, ceiling, label_key)

    def _cheaper_rivals(self, line: str, ceiling: float, label_key: str) -> list[tuple[str, float]]:
        """Buyable rows on the line's shelf priced below it, lowest first, at most ten."""
        shelf = self._shelf_text(line, label_key)
        rivals = [lid for lid in self.seen_sequence
                  if lid != line and lid not in self.basket_counts and self.is_purchasable(lid)
                  and self._shelf_text(lid, label_key) == shelf]
        priced = [(lid, self.price_of_listing(lid)) for lid in rivals]
        return sorted(((lid, price) for lid, price in priced if price is not None and price < ceiling),
                      key=lambda pair: pair[1])[:10]

    def _read_cheapest_rivals(self, line: str, ceiling: float, label_key: str) -> None:
        """Read the cheaper rows in price order, one step each and at most four, stopping at the first whose
        attributes show every plain requirement of the item, while the steps for a swap and the checkout stay
        free; the note then says which rows already show the item's needs."""
        slot = self.good_for_shelf(self._shelf_text(line, label_key))
        reads = 0
        for lid, _ in self._cheaper_rivals(line, ceiling, label_key):
            if lid in self.page_records:
                if slot and self._rv_plainly_meets(lid, slot):
                    return
                continue
            if reads >= 4 or self.moves_left() <= self.wrap_up_reserve() + 6 or self.run_closed:
                return
            reads += 1
            self.read_traits_of([lid], self.wrap_up_reserve() + 4)
            if slot and self._rv_plainly_meets(lid, slot):
                return

    def _sweep_line_query(self, line: str, ceiling: float, label_key: str) -> bool:
        """Without a price-capped search: run the line's item query (its plain values and its name) and page through
        it, at most three pages and one step each, so every row priced below the line comes into view; once per line
        and price. Whether any page was read."""
        plan = self.words_query_plan() or self._lone_text_query_plan()
        tag = ('sweep', line, ceiling)
        if not plan or tag in self.capped_queries:
            return False
        tool, base_args, text_field = plan
        slot = self.good_for_shelf(self._shelf_text(line, label_key))
        words = self._query_words_for_good(slot) if slot else ''
        if not words:
            return False
        longest = ((self.schemas.get(tool) or {}).get('properties') or {}).get(text_field, {}).get('maxLength')
        if isinstance(longest, int) and len(words) > longest:
            words = words[:longest].rsplit(' ', 1)[0]
        self.capped_queries.add(tag)
        bag = sorted(words.lower().split())
        if sum(sorted(str(past.get(text_field) or '').lower().split()) == bag for past in self.recent_queries) >= 2:
            # the same words, in any order, were searched and paged already: their rows are known, no step again
            self.journal.append({'kind': 'sweep_known', 'line': line, 'words': words})
            return True
        args, pages = {**base_args, text_field: words}, 0
        while pages < 3 and self.moves_left() > self.wrap_up_reserve() + 5 and not self.run_closed:
            _, reply = self._send_fitted(tool, args)
            if reply is None:
                break
            observed = reply.get('observation')
            self.digest_reply(tool, observed)
            pages += 1
            state = self._paging_state(tool, observed)
            if not state or state[3] is None:
                break
            args = {**args, state[2]: state[3]}
        self.journal.append({'kind': 'sweep_pages', 'line': line, 'pages': pages, 'words': words})
        pages += self._sweep_rungs(tool, base_args, text_field, words, slot, longest)
        return pages > 0

    def _sweep_rungs(self, tool: str, base_args: dict[str, Any], text_field: str, words: str, slot: int,
                     longest: Any) -> int:
        """Further one-page searches after the line's own query, each only when no search sent these words yet and
        steps and time allow: the query with every number-and-unit value spelled the other way ("2 TB" / "2tb"),
        since the store reads the two as different words; and the query with the item's noun in the singular, since
        store labels are plural while many listing titles are not. Pages read."""
        rungs: list[str] = []
        flipped = re.sub(r'\b(\d+(?:\.\d+)?)\s+([a-zA-Z]{1,4})\b', r'\1\2', words)
        if flipped == words:
            flipped = re.sub(r'\b(\d+(?:\.\d+)?)([a-zA-Z]{1,4})\b', r'\1 \2', words)
        if flipped != words:
            rungs.append(flipped)
        title = self.goods[slot - 1].good_title.strip() if slot else ''
        noun = title.split()[-1] if title else ''
        single = _plural_root(noun.lower()) if noun else ''
        if noun and single and single != noun.lower() and re.search(r'\b' + re.escape(noun) + r'\b', words):
            rungs.append(re.sub(r'\b' + re.escape(noun) + r'\b', single, words))
        read = 0
        for query in rungs:
            if isinstance(longest, int) and len(query) > longest:
                query = query[:longest].rsplit(' ', 1)[0]
            bag = sorted(query.lower().split())
            if any(sorted(str(past.get(text_field) or '').lower().split()) == bag for past in self.recent_queries):
                continue
            if self.moves_left() <= self.wrap_up_reserve() + 8 or self.seconds_spent() >= RUSH_SECONDS or self.run_closed:
                break
            sent, reply = self._send_fitted(tool, {**base_args, text_field: query})
            if reply is None:
                break
            observed = reply.get('observation')
            self.digest_reply(tool, observed)
            read += 1
            if self.records_in_reply(observed):
                self.recent_queries = (self.recent_queries + [dict(sent)])[-8:]
            self.journal.append({'kind': 'sweep_rung', 'words': query, 'rows': len(self.records_in_reply(observed))})
        return read

    def _lone_text_query_plan(self) -> tuple[str, dict[str, Any], str] | None:
        """When no earlier search taught which field takes the words: the search tool's one required free-text
        argument (a string without fixed options), with the enumerated arguments the latest search used."""
        tool = self.role_map.get('search')
        schema = self.schemas.get(tool) or {}
        props = schema.get('properties') or {}
        texts = [key for key in schema.get('required') or []
                 if isinstance(props.get(key), dict) and 'string' in _declared_types(props[key]) and 'enum' not in props[key]]
        if not tool or len(texts) != 1:
            return None
        latest = self.recent_queries[-1] if self.recent_queries else {}
        base = {key: value for key, value in latest.items()
                if key != texts[0] and isinstance(props.get(key), dict) and 'enum' in props[key]}
        return tool, base, texts[0]

    def _cheaper_rows_note(self, line: str, ceiling: float, label_key: str, swept: bool = False) -> str:
        shelf = self._shelf_text(line, label_key)
        below = self._cheaper_rivals(line, ceiling, label_key)
        how = 'paged through the search for' if swept else 'listed'
        head = f"\n[Wren {how} {shelf or 'its category'} below the price of {line} ({ceiling:g})"
        if not below:
            return head + ': no cheaper row that can be bought turned up.]'
        slot = self.good_for_shelf(shelf)

        def shown(lid: str, price: float) -> str:
            if lid not in self.page_records:
                return f'{lid} ({price:g})'
            fits = bool(slot) and self._rv_plainly_meets(lid, slot)
            return f"{lid} ({price:g}, attributes read: {'every plain requirement shown' if fits else 'not every requirement shown'})"
        rows = ', '.join(shown(lid, price) for lid, price in below)
        return (head + f'; cheaper rows that can be bought, lowest price first: {rows}. Switch the line to the '
                'cheapest of these that meets every requirement of the item (read the attributes of an unread one '
                'first); a row marked "every plain requirement shown" still needs any bound, exclusion or borrowed '
                'value checked.]')

    def _capped_word_search(self, line: str, ceiling: float, label_key: str) -> bool:
        """One word search for the line's item (its plain required values, then its name) with the search's top price
        set just below the line's price, once per line and price; tells whether it ran."""
        plan, capped = self.words_query_plan(), self._price_capped_browser()
        tag = ('words', line, ceiling)
        if not plan or not capped or plan[0] != capped[0] or tag in self.capped_queries:
            return False
        tool, base_args, text_field = plan
        slot = self.good_for_shelf(self._shelf_text(line, label_key))
        words = self._query_words_for_good(slot) if slot else ''
        if not words:
            return False
        props = (self.schemas.get(tool) or {}).get('properties') or {}
        longest = props.get(text_field, {}).get('maxLength')
        if isinstance(longest, int) and len(words) > longest:
            words = words[:longest].rsplit(' ', 1)[0]
        self.capped_queries.add(tag)
        _, reply = self._send_fitted(tool, {**base_args, text_field: words, capped[3]: self._just_under(ceiling)})
        if reply is None:
            return False
        self.digest_reply(tool, reply.get('observation'))
        self.journal.append({'kind': 'words_below', 'line': line, 'price': ceiling, 'words': words})
        return True

    def _query_words_for_good(self, slot: int) -> str:
        """Search words for one item: at most two of its plain required values (no limits, negations, alternatives or
        borrowed values) followed by its name."""
        good = self.goods[slot - 1]
        borrowed = {need for _, need in self.borrowed_needs}
        named = set(good.good_title.lower().split())
        picked: list[str] = []
        for need in good.given_needs + good.told_needs:
            value = need.partition(':')[2].strip()
            if not value or need in borrowed or need in good.dropped_needs or len(value.split()) > 3:
                continue
            if set(value.lower().split()) <= named:
                continue
            if _HEDGED_VALUE_RX.search(value) is None:
                picked.append(value)
        return ' '.join(picked[:2] + [good.good_title]).strip()

    def _learn_side_width(self, job: _ProposedCall, seen: Any) -> None:
        names = job.params.get(job.list_key) if job.list_key else None
        if not isinstance(names, list) or len(names) < 2:
            return
        dumped = _jdump(seen)
        if any(str(name) in dumped for name in names):
            self.side_ok_width = max(self.side_ok_width, len(names))
            return
        self.side_fail_width = min(self.side_fail_width or 999, len(names))
        job.side_notes += (f'\n[Wren] The side-by-side view returned nothing for {len(names)} listings: put at most '
                           f'{self.side_by_side_width()} in one call.')

    def _only_listing_ids(self, value: Any) -> bool:
        members = value if isinstance(value, list) else [value]
        if not members:
            return False
        return all(isinstance(member, str) and member in self.row_records for member in members)

    def _absorb_call(self, job: _ProposedCall, seen: Any) -> None:
        self.recent_marks = (self.recent_marks + [job.fingerprint])[-3:]
        self.id_args.update((job.tool, key) for key, value in job.params.items() if self._only_listing_ids(value))
        self.ingest_records(seen)
        roles = self.role_map
        if job.tool == roles.get('compare'):
            rows = self.records_in_reply(seen)
            if rows:
                shows = self.tool_shows_traits.get(job.tool, False) or any(_trait_block(rec) for _, rec in rows)
                self.tool_shows_traits[job.tool] = shows
        if job.tool in job.read_tools - {roles.get('search')}:
            self.note_view_keys(job.tool, seen)

    def _move_open_role(self, failed: str) -> None:
        """Hand the open role to another single-listing tool not yet known to lack attributes, if there is one."""
        roles = self.role_map
        excluded = self._role_tools('cart', 'buy', 'refuse', 'ask', 'search')
        for tool in self.schemas:
            if tool != failed and self.lone_id_arg(tool) and self.tool_shows_traits.get(tool) is not False \
                    and tool not in excluded:
                roles['open'] = tool
                return

    def _learn_from_single_view(self, tool: str, seen: Any) -> None:
        """A view of one listing shows whether that tool carries attributes, which may shift the open and offer roles."""
        if not self.lone_id_arg(tool) or tool in self._role_tools('cart', 'buy', 'refuse', 'ask'):
            return
        listing = self.id_of_record(seen) if isinstance(seen, dict) else None
        if not listing:
            return
        roles = self.role_map
        has_facts = self.tool_shows_traits.get(tool, False) or bool(_trait_block(seen))
        self.tool_shows_traits[tool] = has_facts
        if not has_facts:
            if tool == roles.get('open'):
                self._move_open_role(tool)
            if tool != roles.get('open') and not roles.get('offer'):
                roles['offer'] = tool
        if tool == self.offer_viewer():
            self.offer_seen_at[listing] = self.moves_made

    def _record_store_effects(self, job: _ProposedCall, reply: dict[str, Any], seen: Any) -> None:
        roles = self.role_map
        if job.tool == roles.get('cart'):
            self._absorb_cart_answer(job.tool, job.params, seen)
        if job.tool == roles.get('search') and self.records_in_reply(seen):
            self.recent_queries = (self.recent_queries + [dict(job.params)])[-8:]
        if job.tool == roles.get('cart_view'):
            self.basket_view, self.basket_seen_at = seen, self.moves_made
        if not self.is_purchase_call(job.tool, job.params):
            return
        self.purchase_tries += 1
        if not reply.get('finished'):
            self.purchase_refused_at = self.moves_made

    def _absorb_cart_answer(self, tool: str, call_args: dict[str, Any], seen: Any) -> None:
        self.note_basket_call(tool, call_args)
        self.last_basket_reply = _jdump(seen)
        self.recent_marks = []
        counts = [_coerce_int(v) for v in call_args.values() if not isinstance(v, str) and _coerce_int(v) is not None]
        if counts:
            adding = counts[0] > 0
            for lid in [v for v in call_args.values() if isinstance(v, str) and v in self.row_records]:
                # a line brought to zero that the answer no longer shows counts as withdrawn; adding units restores it
                if adding:
                    self.dropped_needs.discard(lid)
                elif not _id_among_rows(seen, lid):
                    self.dropped_needs.add(lid)
        standing = [lid for lid in self.basket_ids() if not self.sold_out_now(lid) and lid not in self.dropped_needs]
        if len(standing) >= len(self.composed_lines):
            self.composed_lines = standing

    def _build_answer(self, job: _ProposedCall, reply: dict[str, Any], seen: Any, more_pages: str,
                      browsing: bool) -> str:
        text = self.show_reply(seen) + more_pages + job.side_notes
        if browsing:
            text += (self.lowest_price_rows_note([seen] + self.extra_pages)
                     + self.ignored_cap_note(job.tool, job.params, seen))
        to_cache = text
        proof = self.proof_lines(seen)
        extras = ['\n' + proof] if proof else []
        extras.append(self._zero_units_hint(job, seen))
        self.notice_says_sold = False
        extras.append(self._store_notice_text(seen))
        extras.append(self._recheck_after_refusal(job, reply, seen))
        self.notice_says_sold = False
        advice = self.sellout_guidance()
        if advice:
            extras.append('\n' + advice)
        if self.pending_notes:
            extras.append("\n[what Wren's own calls noted] " + ' | '.join(self.pending_notes))
            self.pending_notes = []
        if job.tool in job.read_tools:
            self.read_cache[job.fingerprint] = (self.revision, self.moves_made, to_cache)
        if reply.get('finished'):
            extras.append(_CLOSED_MARK)
        return text + ''.join(extras)

    def _zero_units_hint(self, job: _ProposedCall, seen: Any) -> str:
        if job.tool != self.role_map.get('cart'):
            return ''
        values = list(job.params.values())
        if not any(_coerce_int(v) == 0 for v in values if not isinstance(v, str)):
            return ''
        if not any(isinstance(v, str) and v in self.row_records and _id_among_rows(seen, v) for v in values):
            return ''
        return ("\n[Wren] A count of 0 left that line in the cart: this tool adds the count it gets to the line's units, "
                'so take units off with a negative count.')

    def _store_notice_text(self, seen: Any) -> str:
        loose = self.loose_text_lines(seen)
        if not loose:
            return ''
        meaning = self.interpret_shop_note(loose)
        if not meaning:
            return ''
        return '\n[store notice] ' + meaning + '. Adjust your choices to this change before you check out.'

    def _recheck_after_refusal(self, job: _ProposedCall, reply: dict[str, Any], seen: Any) -> str:
        """After a refused checkout or a sell-out notice, read the carted lines again to find the one that went."""
        buying = self.is_purchase_call(job.tool, job.params)
        refused = buying and not reply.get('finished')
        repeated = refused and any(line in self.message_words(seen) for line in self.sellout_notice_lines)
        unnamed = self.notice_says_sold and (refused or (not self.notice_named_id and not buying))
        if not (unnamed or repeated) or not self.basket_ids() or self.run_closed:
            return ''
        found = self.find_sold_line()
        return '\n[Wren checked the cart lines again after the refusal]\n' + found if found else ''

    # ================================================================== several opens as one side-by-side view

    def _merge_single_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Two or more single-listing opens in one batch turn into one side-by-side call when that view shows
        attributes."""
        plan = self._merge_plan(calls)
        return self._apply_merge(calls, *plan) if plan else calls

    def _merge_plan(self, calls: list[dict[str, Any]]) -> tuple:
        """(side-by-side tool, its list parameter, [(batch position, listing)]) when a merge applies, else ()."""
        opener, joiner = self._role_tools('open', 'compare')
        id_field = self.lone_id_arg(opener)
        list_field = self.side_by_side_list_arg()
        if not (opener and joiner and id_field and list_field) or opener == joiner:
            return ()
        if self.side_by_side_hides_traits():
            return ()
        picks = self._mergeable_opens(calls, opener, id_field)[:self.side_by_side_width()]
        return (joiner, list_field, picks) if len(picks) > 1 else ()

    def _apply_merge(self, calls: list[dict[str, Any]], joiner: str, list_field: str,
                     picks: list[tuple[int, str]]) -> list[dict[str, Any]]:
        ids = [lid for _, lid in picks]
        taken = {slot for slot, _ in picks}
        first = picks[0][0]
        out: list[dict[str, Any]] = []
        for slot, entry in enumerate(calls):
            if slot == first:
                # the first merged open carries the side-by-side call; the others only point at its result
                out.append({'id': entry['id'], 'type': 'function',
                            'function': {'name': joiner, 'arguments': _jdump({list_field: ids})}})
            elif slot in taken:
                out.append({'id': entry['id'], 'merged': True})
            else:
                out.append(entry)
        self.journal.append({'kind': 'joined', 'ids': list(ids)})
        return out

    def _mergeable_opens(self, calls: list[dict[str, Any]], opener: str, id_field: str) -> list[tuple[int, str]]:
        found: list[tuple[int, str]] = []
        for slot, entry in enumerate(calls):
            fn = entry.get('function') or {}
            if fn.get('name') != opener:
                continue
            decoded = _first_json_object(fn.get('arguments'))
            if not decoded or set(decoded) != {id_field}:
                continue
            lid = str(decoded.get(id_field) or '').strip()
            if lid in self.row_records and lid not in self.sold_out_steps:
                found.append((slot, lid))
        return found

    def _side_view_schema(self) -> dict[str, Any]:
        return self.schemas.get(self.role_map.get('compare') or '') or {}

    def side_by_side_list_arg(self) -> str | None:
        schema = self._side_view_schema()
        fields = schema.get('properties') or {}
        array_key = next((key for key, spec in fields.items()
                          if isinstance(spec, dict) and 'array' in _declared_types(spec)), None)
        # usable only when nothing but that list is required
        sole = bool(array_key) and all(key == array_key for key in schema.get('required') or [])
        return array_key if sole else None

    def side_by_side_width(self) -> int:
        """The most listings one side-by-side call may name: the schema's own cap, else what the store has shown."""
        fields = self._side_view_schema().get('properties') or {}
        cap = (fields.get(self.side_by_side_list_arg() or '') or {}).get('maxItems')
        if isinstance(cap, int):
            return cap
        ok = self.side_ok_width
        if not self.side_fail_width:
            return max(4, ok)
        return max(2, min(self.side_fail_width - 1, max(ok, 2)))

    def show_reply(self, observation: Any) -> str:
        text = _jdump(_clip_texts(observation, CELL_TEXT_LIMIT))
        return text if len(text) <= REPLY_TEXT_LIMIT else text[:REPLY_TEXT_LIMIT] + '...(cut short)'

    # ================================================================== Wren's own count of cart units

    def note_basket_call(self, tool: str, call_args: dict[str, Any]) -> None:
        """Bring Wren's unit count up to date with a cart call the store has just carried out."""
        self.basket_touched = True
        self.basket_changed_at = self.moves_made
        fields = self.schemas.get(tool, {}).get('properties') or {}
        removing = any([self._picks_later_option(fields.get(key), value) for key, value in call_args.items()])
        plain_units = next((_coerce_int(v) for v in call_args.values()
                            if _coerce_int(v) is not None and not isinstance(v, str)), None)
        named, restates = self._lines_named_in(fields, call_args, plain_units)
        if restates:
            # a call made of whole line objects describes the entire cart
            self.basket_counts = {lid: units for lid, units in named if units and units > 0}
            return
        sets_total = self.basket_sets_count(tool)
        for lid, units in named:
            after = self._units_after(fields, self.basket_counts.get(lid, 0), units, removing, sets_total)
            if after > 0:
                self.basket_counts[lid] = after
            else:
                self.basket_counts.pop(lid, None)

    @staticmethod
    def _units_after(fields: dict[str, Any], held: int, units: int | None, removing: bool, sets_total: Any) -> int:
        if removing:
            if any('integer' in _declared_types(spec) for spec in fields.values() if isinstance(spec, dict)):
                return held - (units or 1)
            return held - units if units else 0
        delta = 1 if units is None else units
        return delta if sets_total else held + delta

    @staticmethod
    def _picks_later_option(spec: Any, value: Any) -> bool:
        # a switch set to any option after its first one (add / remove style) means units come off
        options = (spec or {}).get('enum')
        if not isinstance(options, list) or len(options) < 2:
            return False
        return value in options[1:]

    def _lines_named_in(self, fields: dict[str, Any], call_args: dict[str, Any],
                        plain_units: int | None) -> tuple[list[tuple[str, int | None]], bool]:
        """(listing, units) for each listing the call names, and whether the call restates the whole cart."""
        found: list[tuple[str, int | None]] = []
        restates = False
        for key, given in call_args.items():
            if isinstance(given, list):
                restates = self._gather_list_lines(fields.get(key), given, plain_units, found) or restates
            elif isinstance(given, str) and given in self.row_records:
                found.append((given, plain_units))
        return found, restates

    def _gather_list_lines(self, spec: Any, entries: list[Any], plain_units: int | None,
                           found: list[tuple[str, int | None]]) -> bool:
        """Add the (listing, units) pairs of one list argument to `found`; tells whether the list restates the cart
        (it holds line objects, or it is empty while declared as a list of line objects)."""
        row_shape = (spec or {}).get('items') or {}
        if not entries:
            return isinstance(row_shape, dict) and (row_shape.get('type') == 'object' or 'properties' in row_shape)
        whole = False
        for entry in entries:
            if isinstance(entry, str):
                if entry in self.row_records:
                    found.append((entry, plain_units))
                continue
            if not isinstance(entry, dict):
                continue
            whole = True
            lid = next((v for v in entry.values() if isinstance(v, str) and v in self.row_records), None)
            units = next((_coerce_int(v) for v in entry.values()
                          if _coerce_int(v) is not None and not isinstance(v, str)), 1)
            if lid:
                found.append((lid, units))
        return whole

    # ================================================================== ending the run without the model

    def _cart_view_current(self) -> bool:
        return self.basket_view is not None and self.basket_seen_at >= self.basket_changed_at

    def _no_more_moves(self) -> bool:
        return self.run_closed or self.moves_left() <= 0

    def _gate_if_room(self, phase: str) -> None:
        if self.moves_left() >= 2:
            self.run_procedure_gate(phase)

    def _quiet_refusal_fitted(self, refuser: str, issues: list[str]) -> dict[str, Any]:
        quiet = self.quiet_refusal_args(refuser, dict(self.digest.get('refuse_args') or {}))
        return _fit_to_schema(quiet, self.schemas.get(refuser), 'args', issues)

    def _refusal_args_fit(self) -> bool:
        refuser = self.role_map.get('refuse')
        if not refuser:
            return False
        issues: list[str] = []
        self._quiet_refusal_fitted(refuser, issues)
        return not issues

    def close_out_run(self, explain_refusal: bool = False) -> str:
        """Check out a held cart, otherwise end without a purchase. With explain_refusal a refused checkout comes back
        as text for the model, and a refused ending without purchase always does; '' in every other case."""
        if self._cannot_act():
            return ''
        self._cart_proven_basket()
        trusted = self._cart_view_current()
        holding = bool(self.basket_ids()) if trusted else bool(self.basket_counts)
        complete = self._finish_multi_item_cart() if holding and len(self.goods) >= 2 else True
        if self._no_more_moves():
            return ''
        buyer, refuser = self._role_tools('buy', 'refuse')
        if complete and buyer and (holding or (self.basket_touched and not trusted and self.moves_left() >= 2)):
            stop, said = self._forced_order(buyer, explain_refusal)
            if stop:
                return said
        return self._forced_refusal(refuser) if refuser else ''

    def _cart_proven_basket(self) -> None:
        """At a forced ending: when the cart is short of an item or holds an unproven line while the read listings make
        a fully proven order, set the cart to that order first (one step per cart call, leaving one for the checkout)."""
        plan = self.proven_set()
        cart_tool = self.role_map.get('cart')
        if not plan or not cart_tool or self._no_more_moves():
            return
        held = {lid: units for lid, units in self.basket_counts.items() if units > 0}
        if plan == held:
            return
        label_key = self.shelf_field_name()
        slot_of = {lid: self.good_for_shelf(self._shelf_text(lid, label_key)) for lid in held} if label_key else {}
        short = len(set(slot_of.values()) - {0}) < len(self.goods)
        weak = any(not slot or not self._rv_proves(lid, slot) for lid, slot in slot_of.items())
        if self.digest.get('pick') in ('cheapest', 'most', 'least'):
            weak = False
        if not short and not weak:
            return
        calls = self.basket_edit_calls(plan, [lid for lid in held if lid not in plan])
        if not calls or len(calls) + 1 > self.moves_left():
            return
        sent = self._push_cart_calls(cart_tool, calls)
        self.journal.append({'kind': 'proven_basket', 'plan': plan, 'short': short, 'sent': sent is not None})

    def _finish_multi_item_cart(self) -> Any:
        complete = self.top_up_basket() or not self._refusal_args_fit()
        if not complete:
            self.journal.append({'kind': 'short_cart_refused', 'lines': self.basket_ids()})
            return complete
        if len(self.basket_ids()) >= 2:
            self.read_traits_of(self.basket_ids(), self.wrap_up_reserve())
            if self.moves_left() >= 3 and self.model_time_allowance() > 5:
                self.chase_part_refs(self.basket_ids(), self.wrap_up_reserve())
        return complete

    def _forced_order(self, buyer: str, explain_refusal: bool) -> tuple[bool, str]:
        """Check out without the model. Returns (stop, text): with stop False the caller goes on to the ending without
        purchase; with stop True it hands back text ('' or the refusal to show the model)."""
        issues: list[str] = []
        order_args = _fit_to_schema(dict(self.digest.get('buy_args') or {}), self.schemas.get(buyer), 'args', issues)
        if issues:
            return False, ''
        for prepare in (lambda: self._gate_if_room('order'), self._peek_changed_cart):
            prepare()
            if self._no_more_moves():
                return True, ''
        reply = self.shop_call(buyer, order_args)
        self.journal.append({'kind': 'forced_buy', 'reply': _clip_texts(reply, 200)})
        if self._no_more_moves():
            return True, ''
        if reply is None or not explain_refusal:
            return False, ''
        self.purchase_tries, self.purchase_refused_at = self.purchase_tries + 1, self.moves_made
        page = reply.get('observation')
        self.digest_reply(buyer, page)
        return True, 'the store refused the checkout: ' + self.show_reply(page)

    def _forced_refusal(self, refuser: str) -> str:
        issues: list[str] = []
        end_args = self._quiet_refusal_fitted(refuser, issues)
        if issues:
            return ''
        self._gate_if_room('end')
        if self._no_more_moves():
            return ''
        reply = self.shop_call(refuser, end_args)
        self.journal.append({'kind': 'forced_refuse', 'reply': _clip_texts(reply, 200)})
        if reply is None or self.run_closed:
            return ''
        page = reply.get('observation')
        self.digest_reply(refuser, page)
        return 'the store refused the ending without purchase: ' + self.show_reply(page)


class ShoppingRun(RunBase, RunListings, RunSearching, RunInterview, RunVerification, RunBasket, RunTurnLoop):
    """One whole shopping episode put together from the mixins: store and model links, listing records, searching,
    the shopper interview, the checks, the cart and the turn loop."""


def _crash_entry(error: BaseException) -> dict[str, Any]:
    return {'kind': 'crash', 'error': f'{type(error).__name__}: {error}'[:400]}


def _salvage_ending(run: ShoppingRun) -> None:
    """After a failure in mid-episode, still try to close the run; a failure while closing is only logged."""
    try:
        if not run.run_closed:
            run.close_out_run()
    except Exception as close_failure:
        run.journal.append(_crash_entry(close_failure))


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        run = ShoppingRun(problem_data)
    except Exception as boot_failure:
        return [_crash_entry(boot_failure)]
    try:
        return run.run_to_end()
    except Exception as play_failure:
        run.journal.append(_crash_entry(play_failure))
        _salvage_ending(run)
        return run.journal


__all__ = ['agent_main']
