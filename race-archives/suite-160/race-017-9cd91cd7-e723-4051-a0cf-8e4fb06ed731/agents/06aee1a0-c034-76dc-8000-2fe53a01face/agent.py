"""Corvin: one buyer acting for one shopper inside an ORO store world.

Work is split between a model and plain code. A low-cost model moves through the store using whatever tools the
world exposes; the code around it holds the ledger of items and requirements, decides which questions go to the
shopper, and inspects the cart before any order and the evidence before any ending without a purchase. Additions
on top of that loop:
- when Corvin is about to buy and the request still keeps a message back for the end, that message carries one
  more planned question, so a small message allowance does not leave a hidden preference unasked;
- a clock holds each episode within the validator's 300-second limit: an interview, a question, a loop turn or a
  model request is only started while its result can still be used in time, and the model timeout gets shorter
  as the remaining time runs down;
- for as long as the shopper answers lists, a message names up to PROPERTIES_PER_MESSAGE planned properties of a
  single item, and the properties rotate so that one the shopper is indifferent to cannot keep the ones queued
  behind it hidden; once a reply asks for a single question at a time, every message names one property only.
"""
from __future__ import annotations

import json
import random
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

# --- models: the cheap workhorse, the stronger reviewer, and the fallback for the store loop and note reading
FAST_MODEL: str = "anthropic/claude-haiku-5.5"
STRONG_MODEL: str = "anthropic/claude-sonnet-5.5"
BACKUP_MODEL: str = "openai/gpt-6-luna"

# --- how much model work one episode may buy (cost limits are in credit units)
MAX_LOOP_TURNS: int = 42
MAX_MODEL_REQUESTS: int = 60
SOFT_COST_LIMIT: float = 0.055      # reviews may still start with the strong model up to this spend
HARD_COST_LIMIT: float = 0.078      # the store loop's spending ceiling
STRONG_COST_MARGIN: float = 0.02    # headroom one strong-model request needs below the ceiling

# --- retrying a failed model request: waits in seconds, and their combined maximum
RETRY_PAUSES: tuple[int, ...] = (4, 10, 20, 35)
MAX_TOTAL_PAUSE: int = 45

# --- size caps on text handed to the store loop (characters)
FIELD_CHAR_CAP: int = 450           # one string field inside a tool result
RESULT_CHAR_CAP: int = 15000        # one rendered tool result
HISTORY_CHAR_CAP: int = 150000      # the whole chat before older results get shortened

# --- extra store work the code may do by itself
PROCESS_CHECK_ROUNDS: int = 3       # model rounds for the request's own steps due at an order or an ending
SELF_PAGING_CAP: int = 6            # result pages fetched without the model asking
CHEAPER_READS_PER_LINE: int = 3     # cheaper unread listings opened per cart line when the pick is cheapest

# --- wall clock, in seconds since the episode started
EPISODE_SECONDS: float = 300.0      # the validator's limit
SAFETY_SECONDS: float = 20.0        # kept free at the end of the episode
MIN_REQUEST_SECONDS: float = 6.0    # no model request is started with less time than this left
INTERVIEW_DEADLINE: float = 165.0   # the interview with the shopper starts only before this
QUESTION_DEADLINE: float = 210.0    # no further interview question after this
HURRY_AFTER: float = 200.0          # the status line starts urging a checkout
FINAL_TURN_AFTER: float = 255.0     # the store loop takes no new turn after this ...
EMPTY_CART_SECONDS: float = 12.0    # ... unless the cart is empty and this much model time remains
LATE_QUESTION_DEADLINE: float = 190.0  # the question held back for checkout is asked only before this

# --- messages to the shopper
LATE_QUESTION_STEPS: int = 8        # steps that must remain for the question held back for checkout
FEW_MESSAGES: int = 2               # an allowance this small counts as scarce when lists go unanswered
PROPERTIES_PER_MESSAGE: int = 8     # most planned properties one message may name

# --- patterns over tool schemas (parameter names and allowed modes)
PRICE_CAP_PARAM_RE: str = r"(?:^|_)(?:cap|max|maximum|ceiling|under|up_?to)(?:_|$)"
NOT_PRICE_CAP_RE: str = r"result|row|page|count|size|num|item|step"
CATEGORY_PARAM_RE: str = r"section|department|categor|label|shelf|aisle|(?<!age)(?:^|_)group(?:_name|_label)?$"
BROWSE_MODE_RE: str = r"field|filter|facet|brows|categor|list|structured"
QUERY_PARAM_RE: str = r"(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword"

# --- patterns over requirement values: a measured amount, and a lower bound written before one
MEASURE_RE: str = r'''\d[\d.,]*\s*[a-z"']{1,12}'''
LOWER_BOUND_RE: str = r"\b(?:at least|no (?:less|fewer) than|min(?:imum)?)\s+(" + MEASURE_RE + r")\s*$"

# --- proxy links: model requests, short-timeout model links cached per timeout, and store calls
_model_client = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_short_clients: dict[int, ProxyClient] = {}
_store_client = ProxyClient(timeout=95, max_retries=3, retry_delay=1)

# System message of the tool-using model that walks the store (p09 opens the chat with it).
LOOP_SYSTEM_PROMPT = (
    'You are Corvin, a meticulous purchasing agent acting for a single shopper in an online store, and the '
    "store's tools are your only means of doing anything. Scoring is all or nothing: the episode gets 1 when "
    'its finish is completely correct, and 0 otherwise. A correct finish is either an order that holds every '
    'requested item in the requested quantity, each one satisfying every requirement still in force at '
    'checkout, with the total inside the budget; or, if the store genuinely has nothing that satisfies '
    'everything, an ending through the no-purchase tool carried out the way the request says.\n'
    '\n'
    'THE STORE AND ITS TOOLS\n'
    '- Each tool call uses up one step out of a limited supply, while reasoning is free. This store has its '
    'own tool names, argument names and result layouts, so study the description and schema of every tool and '
    'pass precisely the argument names and permitted values they define.\n'
    '- Searching may be restricted: a category filter, a keyword mode that might look at only the opening '
    'words of a query, pages of just a few rows, a token or offset that fetches the following page. Whenever '
    'the answer depends on having seen every row, check the total the store reports against the rows seen so '
    'far and keep paging, passing the token or offset exactly as the store supplies it. Keyword queries do '
    'best when short with the value up front (the value you want, then the product noun). An empty result '
    'from a filtered search most often means a wrong filter value or a wrong filter parameter: try again with '
    'the label copied exactly as result rows print it, or switch to a different filter parameter.\n'
    "- A search row is only a summary. Stores divide a listing's record in different ways: one view may "
    'expose its attributes (properties), another only its offer and description, and a side-by-side view '
    'could hold either. As soon as Corvin has found out which tool shows attributes, the ledger names it. '
    'Each requirement on a property is judged from those attributes, so read them for every listing you might '
    'buy; a single call covering several listings saves steps when it displays what you need. Do not reopen a '
    'listing you have already read unless something has changed since.\n'
    "- Corvin's interviewer takes care of every question to the shopper, so you never send the shopper a "
    'message yourself. The answers it gets show up in the conversation and are recorded in the ledger.\n'
    '- Besides the purchase, the request may call for actions along the way: an approval, a message, '
    'something to check or to read beforehand. These appear in the ledger. Those due right before the order '
    'or right before the ending are performed by Corvin itself at the moment you call the checkout tool or '
    'the ending tool, so do not hold back waiting for them and never end without a purchase on their account; '
    "carry out any others with the store's tools (questions about the products go through Corvin's "
    'interviewer).\n'
    '\n'
    'WHEN A LISTING FITS\n'
    '- What kind of product a listing is gets settled by its category label: only a label matching the '
    'requested item as a whole qualifies, never one that merely neighbours it.\n'
    "- A property requirement is satisfied when the listing's attribute for that property gives the value. "
    'The way the key is written is irrelevant (spaces, underscores, camel case, prefixes), and a match on one '
    'entry of a comma- or slash-separated list suffices. Values are compared by their words, ignoring case, '
    "singular versus plural and plain synonyms, and a value that includes the required value's words "
    'qualifies ("pale mint" satisfies mint, "loose relaxed fit" satisfies relaxed). A value made of other '
    'words does not qualify (an abbreviation or code differs from the word it abbreviates, and "mint green" '
    'differs from "olive green"). Convert units before comparing numbers; a limit includes its boundary '
    'value, and a range has to sit entirely within the limit.\n'
    "- When an attribute's key names the requirement's property (by that name or by a plain synonym), that "
    'attribute is decisive: if it gives something different, the listing fails no matter what other keys or '
    'the title claim. A key about a different property, even a related one (the colour of a strap is not the '
    'colour of the case), settles nothing about the requirement. After opened listings Corvin appends '
    '[evidence] lines that give the keys whose names match a requirement.\n'
    '- If the listing carries no attribute for that property, the wanted value counts when it is stated under '
    'some other attribute (a list of styles or features that contains it, for instance) or in the title or '
    'description. A number measuring something else does not count (a weight is no capacity), and neither '
    'does a lone figure whose text never says what it measures.\n'
    "- If the category label itself states a requirement's value (for example a season or an audience in the "
    'label), the category satisfies that requirement.\n'
    '- Instead of a value, a requirement may refer to another listing ("the same X as the listing sold by '
    '<shop> with <property>", "within N of the X on the page of ..."). Work such a requirement out first: '
    'locate that listing (search by its category, brand, shop or wording), open its full page with the '
    'single-item or side-by-side tool (a search row will not do), and if that page sends you on to a further '
    'listing for the property, locate and open that one in the same way; the final page supplies the value. '
    'An order counts only if every page along that chain was opened. When the wanted value is anything other '
    "than that listing's value, the requirement rules out only listings carrying exactly that value, and it "
    'is never by itself a reason to end without a purchase.\n'
    '- If a listing you are about to buy refers you to another page for more about it, view that page before '
    'you buy.\n'
    '- If one requested item has to fit another (an accessory for a device), select the pair jointly from '
    'what the listings themselves say (a list of what it fits, a model or series code present on both pages). '
    'Should no accessory that meets its own requirements fit the device you chose, move to another device '
    'that meets its requirements and that such an accessory does fit; as long as that switch is untried, do '
    'not end without a purchase.\n'
    '- Honour exclusions and closed lists. Wishes ("ideally", "would be nice") serve only to break ties.\n'
    '- Do not buy anything the store marks as sold out; the total (price multiplied by quantity, added up '
    'over all lines) must not go above the budget, though matching it exactly is fine.\n'
    '\n'
    'PICKING A LISTING\n'
    '- If the request wants the cheapest (lowest total, smallest price), the order has to contain the '
    'lowest-priced listing out of ALL listings that fit. Gather the complete fitting set (page through it: '
    'result rows frequently ignore price order, and Corvin fetches the further pages of a short result by '
    'itself), examine candidates from the lowest price upward, and take the first one that fits completely. '
    'If a search accepts a maximum price, make use of it: list the category under the price of your best pick '
    'so far, which leaves only cheaper candidates to examine.\n'
    '- If the request wants the most or the least of some measure, compare all fitting listings on that '
    'measure and take the extreme one; a listing that does not give the measure cannot be the winner.\n'
    '- Otherwise, every listing that satisfies everything is correct: spend no steps looking for a better '
    'one.\n'
    '- Put each item in the cart as soon as you have opened a listing that satisfies all of its requirements '
    '(for a cheapest, most or least pick: after comparing the fitting set), then carry on. A carted line '
    'often sells out shortly after it goes in, and carting early leaves steps for a replacement.\n'
    '- When several different items are requested, each needs a cart line of its own at its requested '
    'quantity.\n'
    '\n'
    'WHEN THINGS CHANGE\n'
    '- The shopper or the store may alter things: a new budget, a requirement added or removed, a price that '
    'moves, a listing that sells out. Such changes turn up inside tool results, often as a refusal when you '
    "try to check out; a note is valid whichever tool's result carries it, and the ledger lists the "
    'requirements in force. A refusal does not end the episode: apply all the changes, correct the cart and '
    'check out once more.\n'
    "- Shortly before checkout, read each chosen listing's record again, since price and stock can change "
    'without notice.\n'
    '\n'
    'ENDING THE EPISODE\n'
    '- Make the cart hold exactly the chosen lines (take out anything else), view the cart one time, then '
    'check out in a call by itself.\n'
    '- Ending without a purchase is allowed only after a genuine search has turned up no listing that '
    'satisfies everything. Being short of steps proves no such thing: in that case check out the best cart '
    'you have. Word that ending as the request instructs, and if a remark is needed, keep it brief and '
    'mention no listing.\n'
    '- Every reply has to contain at least one tool call. Keep your text brief: what you have established, '
    'then your next action.'
)

# Request reader: one JSON brief per episode. TOOLS: and REQUEST: are appended right after the
# closing newline, so the key lines stay one per line and in this order.
BRIEF_PROMPT = (
    "Your job is to turn a shopper's request into a work order for a buying agent. Go through the request and "
    "the store's tools, then answer with ONE JSON object (compact, without indentation, no other text) that "
    'holds these keys:\n'
    '\n'
    '"items": a list with one entry for every separate product the shopper wants bought, each {"name": the '
    'sort of product it is, in the request\'s own words, "shelf": the store\'s category label for it if some '
    'tool lists labels (copied exactly), otherwise the category the request itself assigns it ("a lamp from '
    'lighting" gives "lighting"), otherwise "", "quantity": how many units of that single listing (integer; 1 '
    'if not stated; a pair or a set sold as one listing counts as 1), "musts": a list of the product\'s firm '
    'requirements, each written "property: constraint" carrying the exact value, limit, exclusion or closed '
    'list. Call the property by the word the request uses for it ("intended for a garden occasion" gives '
    '"occasion: garden", "a 12-inch blade" gives "blade length: 12 inch"), and never trade it for a nearby '
    "property. A must whose value comes from another listing is ONE must that holds that listing's whole "
    'description ("strap colour: unlike the strap colour of the watch Brisvel sells with a steel case"): the '
    'words describing that other listing (its shop, its kind, a "with <property> <value> and <property> '
    '<value>" following it) remain inside that single must and never become musts of this product}. Price, '
    'budget, stock and quantity never go into musts. The narrative around the request (who is meant to use '
    'the product, where it is going to stand, what went wrong in the past, the occasion behind the purchase) '
    "is background, not a must, except where it spells out a property's value or limit directly; and words "
    "belonging to the product's own name (a season or an audience named together with the product) are never "
    'made into musts: they go into "name". When the request turns a use or occasion into a property of the '
    'product itself ("trousers for business wear", "a skirt designed for parties", "a hat, and it is for a '
    'wedding", "the occasion must be party"), that IS a must on that property ("occasion: business"), unless '
    "the store's category label already names it; a specific person's event, a person or a setting the "
    'purchase is meant for ("to wear at her sister\'s wedding", "for a neighbour"), and how the product should '
    'sit, feel or look while in use, remain background. Products the request names in a single phrase linked '
    'by "and" (one "I want ..." with one quantity, often the name of a single store department) form ONE '
    'item; create separate items only if the request gives each product a quantity of its own or requirements '
    'of its own.\n'
    '"budget": the highest total the shopper will pay, as a number in the unit the request uses, or null.\n'
    '"budget_unit": the name of that unit (such as "cents"), or null.\n'
    '"budget_scope": "total" if the limit applies to the entire order, "each" if it applies per item, null if '
    'there is no limit.\n'
    '"in_stock_only": true, unless the request plainly accepts items that are unavailable.\n'
    '"pick": "cheapest" if the request wants the lowest price or the smallest total, "most" or "least" if it '
    'wants the extreme of some measure, otherwise "any". Comments that money is short do not make a pick '
    'rule.\n'
    '"pick_measure": the measure being maximised or minimised under "most"/"least", otherwise "".\n'
    '"messages": the number of messages the shopper lets the agent send (integer), or null if the request '
    'does not say.\n'
    '"steps": the number of steps, moves or tool calls the request permits (integer), or null.\n'
    '"lookups": a list of those musts (copied exactly) whose value has to be read first from another listing '
    'the request refers to, otherwise [].\n'
    '"rules": a list holding each instruction in the request about how to work or how to finish, each in 12 '
    'words or fewer.\n'
    '"wishes": a list of the things the shopper would merely like to have.\n'
    '"ask_first": a list of the things the request expressly says to confirm with the shopper before choosing '
    '(empty if there are none).\n'
    '"process": a list of the actions, apart from finding and buying the right items, that the request '
    "requires of the agent (obtaining the shopper's approval, telling or showing the shopper something, "
    'confirming something with them, reading something beforehand, using a specific tool), each {"step": the '
    'action to take, in 15 words or fewer, "when": "choose" (before the products are picked), "order" '
    '(immediately before checkout) or "end" (immediately before ending without a purchase), "messages": the '
    'number of messages to the shopper it needs (0 if none)}; [] if the request demands no such action. '
    'Requirements on the items, the budget, stock, the caps on messages and steps, and the way to end when '
    'nothing fits do not count as steps.\n'
    '"quiet_decline": true if the request says alternatives must not be suggested or named when nothing fits, '
    'false if it asks for them, null if it is silent on this.\n'
    '"fits": a list of {"item": item number, "with": item number}, one for every requested item that the '
    'request says has to work with, fit or suit ANOTHER requested item (whichever one the agent picks); [] if '
    'there is none.\n'
    '"roles": an object giving the exact tool for each job, or null for a job no tool does: "search", "open" '
    "(displays a single listing's attributes or properties; if there are two single-listing tools, the one "
    'whose description promises attributes, properties or specifications), "offer" (displays one listing\'s '
    'current offer, such as its price and stock, if that is a separate tool from "open"), "compare" (several '
    'listings in full), "ask" (messages the shopper), "cart" (modifies the cart), "cart_view" (displays the '
    'cart), "buy" (performs the checkout), "refuse" (ends without a purchase).\n'
    '"buy_args": the fixed arguments under which the "buy" tool checks out (for a tool with a mode argument, '
    'that mode), otherwise {}.\n'
    '"refuse_args": the fixed arguments under which the "refuse" tool ends without a purchase, otherwise {}.\n'
)

# Question planner: every COUNT is replaced by the per-item allowance before the REQUEST, ITEMS,
# ASK FIRST and ATTRIBUTE KEYS ON OPENED LISTINGS blocks are appended; keep COUNT a single upper-case word.
QUESTION_PLAN_PROMPT = (
    'Your task is to pick the questions a shopping agent puts to a shopper. A shopper holds a handful of firm '
    'preferences that were left out of the written request, each concerning one property of one requested '
    'product. Such a preference surfaces only when a question names exactly that property; a property the '
    'shopper has no opinion on draws "don\'t mind", and one the shopper never thinks about draws a baffled '
    'answer, so each property you name uses up part of a message.\n'
    '\n'
    'You receive the request, the items together with what the request already settles for each, and a '
    "numbered list of the attribute keys found on opened listings in each item's category, showing how many "
    'of those listings carry each key along with some example values. A preference left unwritten can concern '
    'a property that none of the listed keys names.\n'
    '\n'
    'Take the properties in the following order:\n'
    '0. Whatever the request itself tells you to confirm with the shopper (shown under ASK FIRST when '
    'present), each named by the property it concerns.\n'
    '1. Hinted properties. The narrative of the request frequently alludes to an unwritten preference while '
    'withholding its value: a sentence on something the shopper is unable to judge or hands over to you, on '
    'someone whose taste is decisive, on how the product looks or the surroundings it belongs to, on the '
    'person who will use or wear it, or on what it contains. For every such sentence, give the single '
    'property it alludes to and the item concerned, following the order in which the narrative presents them.\n'
    "When one of that item's listed keys matches the hint, use it as the hinted property's name (a hint about "
    'the wearer: the key for gender or audience; a hint about appearance: the listed key for colour, pattern '
    "or finish carried by most of its listings); a property that none of the item's listings carries is "
    "placed after the three keys that the item's listings carry most.\n"
    '2. After that, the listed keys in ranked order: prefer the properties a buyer of that kind of product '
    'decides on first (frequently its type or style, its shape or fit, its size, capacity or dimensions, its '
    'material, colour or pattern, whichever of these that kind of product is chosen by) and keys that most of '
    "the item's listings carry; when five or more listings have been read, steer clear of keys that appear on "
    'just one or two of them.\n'
    'Leave out a property the request already settles for that item (some requirement names that property; a '
    'key whose values merely echo a required value, for example a pattern key displaying the required '
    'decoration, is NOT settled), an alternative name for a property already picked, a key where each value '
    'lists several purchase options together, identifiers, model names, warranty and similar logistics. Pack '
    'sizes, counts and quantities per pack are ordinary properties, and a key tagged [not the stated ...] is '
    'some other property than that requirement, no matter which words its name has in common with it. Rank '
    'each item separately: its hinted properties first, its listed keys after them.\n'
    '"label" gives the property\'s plain name: for a listed key, every word of the key spelled out plainly '
    '("strap colour" for strapColour, "seat finish" for seat_finish; never cut down to "colour" or "finish"); '
    "the property's name alone, without example values, brackets or a second property.\n"
    'Pick at most COUNT per item, strongest first. Reply with ONE JSON object and nothing else: {"ask": '
    '[{"item": item number, "label": "...", "hint": "up to 8 of the narrative\'s words it is drawn from, or an '
    'empty string"}]}, with each item\'s picks in rank order'
)
# ---- side-call prompts --------------------------------------------------------------------------------------------
# Each prompt below is a one-shot instruction for a JSON-only side call. The calling code appends its own labelled
# sections (REQUEST, REQUIREMENTS IN FORCE, RECORDS, TOOLS, ...) after the text, so every prompt ends on its JSON shape
# or on a closing rule. The placeholder STAGE in the step gate is swapped for the stage's verb phrase by the caller.

# Shared listing-reading rules: a lower-case fragment spliced into the audit prompts after a short lead-in.
MATCH_RULES = (
    'an attribute settles a property only when its key names that very property (any spelling of the key, a single '
    'list element suffices, synonyms and converted units count, so does a value holding the words of the required '
    'value, for example "deep teal" when teal is required; a limit takes in its edge value); a key naming some '
    "other property, however closely related (a handle's colour is not the body colour), settles nothing; where no "
    'attribute names the property, the wanted value written under some other attribute, in the title or in the '
    'description is accepted (a bare figure that leaves unsaid what it measures is worth nothing); a requirement '
    'spelled out by the category label itself is satisfied by that category; the label of the category has to '
    'match the item; a rule that the value differ from the value of another listing counts as satisfied unless '
    'that same value appears in the records; no listing fits that is out of stock or breaks the budget.'
)

# Final check of the cart just before checkout.
CART_REVIEW_PROMPT = (
    'You are the last gate before a shopping agent pays. Judge from the records given whether the cart is exactly '
    'right.\n'
    'How to read a listing: a requirement holds when the attribute of the listing for that property gives the value '
    '(any spelling of the key, a single list element suffices, letter case, singular or plural and ordinary synonyms '
    'all count, so does a value holding the words of the required value, for example "deep teal" when teal is '
    'required; units are converted, a limit takes in its edge value, and a range qualifies only when all of it sits '
    'inside); an attribute keyed by the very property of the requirement settles it, and the listing fails if that '
    "attribute gives anything else; a key naming some other property, however closely related (a handle's colour "
    'is not the body colour), settles nothing; if no attribute of the listing covers the '
    'property, the wanted value written under some other attribute, in the title or in the description is accepted '
    '(a number of a different kind is not: a weight does not count as a capacity; nor is a bare figure whose text '
    'leaves unsaid what it measures); a requirement spelled out by the category label itself is satisfied by that '
    'category; the label of the category has to match the requested item taken as a whole; a rule that the value '
    'differ from the value of another listing counts as satisfied unless that same value appears in the records. '
    'Every requested item needs a cart line of its own at the requested quantity, the cart may hold nothing more, '
    'no line may show as out of stock, and price times quantity summed over all lines must stay inside the budget. '
    'Of several records of one listing, the latest wins.\n'
    'If the pick rule says cheapest, most or least, go through the other listings shown too: should any of them '
    "clearly meet every requirement and beat the cart's choice under that rule, then the cart is wrong.\n"
    'Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], '
    '"better": [ids of listings that should take the place of a cart line, best first]}. Answer ok=false only when '
    'the records plainly show a problem, or when no record speaks to some requirement.'
)

# Audit of a planned purchase-free ending: which listings might fit after all.
DECLINE_REVIEW_PROMPT = (
    'A shopping agent plans to stop without a purchase, believing that no listing fits. Test the listing records '
    'given against the requirements, and read each record by these rules: '
    + MATCH_RULES +
    '\nA record marked ROW is known only from a row of search results: its page (where the attributes are) is still '
    'unread, so a property that the row leaves out can neither pass nor fail it.\n'
    'Return ONE JSON object and nothing else: {"fits": {"item number": [ids of listings that seem to satisfy every '
    'requirement, cheapest first, at most 4]}, "open": [ids of ROW listings that are of a requested kind and within '
    'the budget, whose page might show a fit, likeliest first, at most 6], "reason": "one sentence"}'
)

# Substitutes for a cart line whose listing sold out.
REPLACEMENT_PROMPT = (
    "One listing in a shopping agent's cart has sold out. Find substitutes for that cart line among the other "
    'listing records given, reading each by these rules: '
    + MATCH_RULES +
    '\nReturn ONE JSON object and nothing else: {"fits": [{"id": id of the listing, "price": the price its record '
    'shows, as a number} for each listing that seems to satisfy every requirement of the item on the sold-out line, '
    'at most 6], "reason": "one sentence"}'
)

# Reading the shopper's replies to the agent's questions.
ANSWER_READ_PROMPT = (
    'The shopper has now answered questions put by a shopping agent. Go through every answer, then return ONE JSON '
    'object and nothing else:\n'
    '{"needs": [{"item": item number, "need": "property: constraint", "points": true if the value has to be taken '
    'from a different listing, otherwise false}], "drop": [{"item": item number, "property": "a requirement the '
    'shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [question numbers '
    'where the reply protests that the message covered too many things at once or asks to get them one at a time; '
    'leave out replies that merely say they could not follow it or would like it put more simply], "budget": the '
    'new overall limit as a number, or null}\n'
    '- "needs": each firm wish an answer expresses, carrying its exact value or limit together with the unit '
    '("at least 30 cm", "oval", "not wicker"). Label the need with the property the answer itself mentions, even if '
    'that is not the property the question was about; when several answers concern one property, together they '
    'form a single need. Take the value over verbatim, just as the answer puts it, nouns and units included '
    '("at least 3 shelves", not "at least 3"); yet the value keeps only the part a listing could state, beginning '
    'at the wanted value or at its limit word and running to its end, minus any framing the answer wraps around it '
    '("oval" out of "matching oval", out of "settled, with oval", or out of "oval and nothing else"; "at least 30 '
    'cm" out of "equal to at least 30 cm").\n'
    '- Sometimes an answer fixes the value by reference to another listing (a value that is like, near or unlike '
    "the one on a certain shop's product, with that product singled out through a few of its properties). Treat it "
    'as ONE need, on the property named in the answer, and set "points": true. For its value, copy the wording of '
    'the answer itself without any change, from the comparison words through to the end of that description, '
    'including what it says of any listing that page may name. Do not trim or paraphrase it. Properties that '
    'describe the other product do not count as needs of this item.\n'
    '- "free": the asked-about property of every question where the shopper said it does not matter to them.\n'
    '- Chatty or indirect wording still states a value if it names one. Wording that will not negotiate on a value '
    '(it is not to be swapped, surrendered or altered) counts as wanting that value; use "not X" only if the answer '
    'turns down X itself.'
)

# Reading a note the store showed mid-run.
STORE_NOTE_PROMPT = (
    'While a shopping agent was working, the store displayed the note below; it may have come from the shopper or '
    'from the store itself. Return ONE JSON object and nothing else:\n'
    '{"kind": "change" if the note alters the task (the budget, the requirements, a price, what is available, or '
    'the order being refused), "error" if all it says is that a call was badly formed or was rejected, "other" in '
    'every other case, "budget": the new overall limit in the unit of the request, as a number, or null, "add": '
    '[{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the '
    'requirement no longer wanted"}], "gone": [ids of listings the note calls unavailable], "sold_out": true if it '
    'says a chosen listing has gone, sold out or can no longer be had (even when it does not name it), '
    '"price_moved": true if it says that a price changed, "summary": "one short sentence"}\n'
    'When a requirement is replaced, give the old one as a drop and the new one as an add. Include every change the '
    'note makes.'
)

# Separating words that only describe a pointed-at listing from real requirements.
POINTER_SPLIT_PROMPT = (
    "Firm requirements on a product were extracted from a shopper's request. For some of them the value comes from a "
    'different listing the request refers to, and the words that identify that listing (its kind, its shop, the '
    'properties given after it in a "with ..." phrase) occasionally get mistaken for requirements on the product to '
    'be bought. Going by how the request is worded, tell which entries under OTHER REQUIREMENTS below merely '
    'describe that different listing. Return ONE JSON object and nothing else: {"describe_other": [each such '
    'requirement, copied exactly; leave it empty if all of them truly apply to the product]}'
)

# Tracing requirements whose value lives on other listings, link by link.
CHAIN_FOLLOW_PROMPT = (
    'Checkout is next for a shopping agent. Certain requirements do not state their value but borrow it from '
    'another listing, and the page of that listing may in turn pass the reader on to yet another listing for the '
    'same property. The order only counts if the agent has fully opened every listing on each such chain. Starting '
    "from the requirement's own description, trace every chain one link at a time, going by what the pages "
    'themselves say, with the records below (OPENED: this page was opened by the agent; ROW: known only as a row of '
    'search results, its page unread).\n'
    'Return ONE JSON object and nothing else: {"chains": [{"need": the requirement, copied exactly, "links": [{"id": '
    'the id of the listing if one record below is this link, otherwise "", "about": the way the requirement or the '
    'page before it describes this listing, "find": "for a link not found: search words (its shop and the values '
    'the description gives)"}], "done": true if the final link is an OPENED page stating the value of the property '
    'itself (this holds too when the requirement wants a value different from or unlike that one: the value on the '
    'page is the yardstick), otherwise false, "value": the value of the property as the final page states it, or '
    '""}]}\n'
    "A record counts as a link only if it fits the link's entire description: shop or brand, product kind AND each "
    'property value the description names (so "a 10 pieces multipack" excludes a listing of that shop with 10 '
    'capsules that is not a multipack); if no record fits all of that, that link counts as not found yet, even when '
    'the shop sells other listings of the same kind. For such a link fill in "find" with a few search words: the '
    'name of the shop plus the values from the description.\n'
    'Give the links of each chain in their order, halting at the first link that is either not found or whose page '
    'is not OPENED, because what it says is not known yet.'
)

# Gate for steps the request wants done before ordering or ending; the caller replaces STAGE.
PROCESS_STEP_PROMPT = (
    "Next, a shopping agent will STAGE. Along the way, the shopper's request calls for the steps listed below. Using "
    "the numbered RECORD of the agent's actions, judge for each step that falls due before it may STAGE whether it "
    'has been done for the cart in its present state. A step is done only if some RECORD line shows it happening; '
    'for a step concerning the order or the cart, that line must also come later than the last change to the cart, '
    'and the reply on it must grant it. Asking about the properties of a product performs no step.\n'
    'If a step is still open, name the single store call that would do it next: a tool from TOOLS together with its '
    'arguments. Messages to the shopper go through the messaging tool as one short sentence or question in plain '
    'words about that step only, mentioning no product property or value the step does not need. Name no call when '
    'all due steps are done, or when the remaining messages or steps cannot accomplish the missing one.\n'
    'Return ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [numbers '
    'of the RECORD lines showing it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what any reply '
    'wants changed before continuing, or "", "needs": [{"item": item number, "need": "property: constraint"} for '
    'every firm requirement stated in such a reply]}'
)

# Mapping the parts of a store tool's result onto its keys.
RESULT_LAYOUT_PROMPT = (
    'Below is what a store tool gave back. Identify its pieces using exactly the keys that appear in the result and '
    'in the parameters of the tool. Return ONE JSON object and nothing else: {"rows": the key whose list holds the '
    'listings, "count": the key giving the total number of matching listings (not merely the number shown), or '
    'null, "next": the key whose value asks for the page after this one, or null, "next_arg": the parameter of the '
    'tool that takes this value, or null, "price": the key within a row for the price of the listing, or null, '
    '"stock": the key within a row telling whether the listing is buyable right now, or null}'
)

# ---- attribute-key heuristics and code patterns ---------------------------------------------------------------------
COMMON_KEY_SLOTS: int = 3           # how many of the most common attribute keys may enter an item's question lane
COMMON_KEY_MIN_SHARE: float = 0.0   # least share of the item's listings that must carry such a key
COMMON_KEY_SHARE_GAP: float = 0.34  # share lead a common key needs over a weaker, unhinted pick it displaces
OPENS_PER_GROUP: int = 6            # listing pages opened per search group when the request has at most two items

# Keys that name no product property (identifiers, media, warranty, condition, ...).
NON_PROPERTY_KEY_RE = r'\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\w*|video|url|links?|images?|photos?|condition)\b'
# Catch-all keys that bundle several features (but not type/style/kind/shape keys).
MIXED_FEATURE_KEY_RE = r'^(?!.*\b(?:types?|styles?|kinds?|shapes?)\b).*\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\w*|accessor\w*|extras?|misc\w*)\b'
# Model codes and part numbers as they appear in titles and descriptions.
MODEL_CODE_RE = r'[A-Z][A-Za-z]{0,5}[ -]?\d{3,6}'
# a family of model codes, the last digits written as placeholder letters ("AB31xx" = every AB31 code of that length)
CODE_FAMILY_RE = r'[A-Z][A-Za-z]{0,5}[ -]?\d{1,5}[xX]{1,4}'
PART_NUMBER_RE = r'[A-Z]{1,4}\d{3,6}'
# ---------------------------------------------------------------- shared module helpers
# Pure functions the agent classes lean on: key spelling, value reading, JSON in and out, tool-schema conformance,
# record walking, compatibility notes in listing text, the short-timeout model clients and the wanted-product record.

# Patterns, compiled once (the same pattern text and flags as the calls they replace).
_CODE_TOKEN_RX = re.compile('\\b' + MODEL_CODE_RE + '\\b')
_CODE_FAMILY_RX = re.compile('\\b' + CODE_FAMILY_RE + '\\b')
_CODE_EXCEPT_RX = re.compile(r'\b(?:except(?: for)?|excluding|but not|other than|apart from)\b', re.I)
_OWN_PART_RX = re.compile('[\\[(]\\s*(?:part\\s+)?(' + PART_NUMBER_RE + ')\\s*[\\])]', re.I)
_MODEL_NO_RX = re.compile('\\bmodel\\s*(?:no\\.?|number|#)\\s*[:.]?\\s*(' + MODEL_CODE_RE + ')\\b', re.I)
_ALIAS_RX = re.compile('\\b(?:also sold as|also known as|sold also as)\\s+([^.;]+)', re.I)
_FITS_CLAIM_RX = re.compile('\\b(?<!not )(?<!never )(?:works with|compatible with|fits)\\s*:?\\s+([^.;]+)', re.I)
_PART_ALSO_TAIL_RX = re.compile('\\bpart\\s+' + PART_NUMBER_RE + '\\s+also\\s+$', re.I)
_PART_ALSO_FITS_RX = re.compile('\\bpart\\s+(' + PART_NUMBER_RE + ')\\s+also fits\\s*:?\\s*([^.;]+)', re.I)
_SEE_PART_RX = re.compile('\\b(?:more|details|see|read)\\s+(?:on|at|under|about)?\\s*(?:the\\s+)?part\\s+(' + PART_NUMBER_RE + ')', re.I)
_FENCE_RX = re.compile('^```[A-Za-z]*\\s*(.*?)\\s*```$', re.S)
_CAMEL_RX = re.compile('([a-z0-9])([A-Z])')
_SEPARATORS_RX = re.compile('[\\s_.\\-]+')
_SIBILANT_ES_RX = re.compile('(?:ch|sh|ss|x|z)es$')
_BRITISH_OUR_RX = re.compile('our$')
_SIGNED_NUMBER_RX = re.compile('\\s*-?\\d+(\\.\\d+)?\\s*')
_WHOLE_NUMBER_RX = re.compile('\\s*\\d+(\\.0+)?\\s*')
_DIGIT_RX = re.compile('\\d')

_COMPACT_JSON = {'ensure_ascii': False, 'separators': (',', ':')}
_UNDECODED = object()
_NO_CHANGE = object()


# ---------------------------------------------------------------- spelling of keys and names

def _plain_words(name: str) -> str:
    """Lower-case words of a key: camelCase humps split, runs of spaces, underscores, dots and dashes made one space."""
    humps_split = _CAMEL_RX.sub('\\1 \\2', str(name))
    return _SEPARATORS_RX.sub(' ', humps_split).strip().lower()


def _snake_case(name: str) -> str:
    return '_'.join(_plain_words(name).split(' '))


def _singular(word: str, spare_double_s: bool) -> str:
    """Cut one closing 's' from a word longer than three letters; with spare_double_s a closing 'ss' is left whole."""
    if len(word) <= 3 or not word.endswith('s'):
        return word
    if spare_double_s and word.endswith('ss'):
        return word
    return word[:-1]


def _word_stems(phrase: str) -> frozenset[str]:
    return frozenset(_singular(word, False) for word in _plain_words(phrase).split())


def _root_word(word: str) -> str:
    # "boxes", "dishes", "classes" lose "es"; other plurals lose the single "s"
    return word[:-2] if _SIBILANT_ES_RX.search(word) else _singular(word, True)


def _root_words(phrase: str) -> set[str]:
    return {root for root in map(_root_word, _plain_words(phrase).split()) if len(root) >= 3 and root not in ('and', 'for', 'the', 'with')}


def _folded_spelling(phrase: str) -> frozenset[str]:
    """Words of a name with a plural 's' dropped and a British closing 'our' written 'or': "colours" equals "color"."""
    return frozenset(_BRITISH_OUR_RX.sub('or', _singular(word, True)) for word in _plain_words(phrase).split())


def _names_same_property(told: frozenset[str], other: frozenset[str]) -> bool:
    # neither may be empty: the told name's words all inside the other name, or the other name (two words or more)
    # wholly inside the told one
    if not (told and other):
        return False
    return told <= other or (len(other) >= 2 and other <= told)


# ---------------------------------------------------------------- reading single values

def _as_int(raw: Any) -> int | None:
    """A whole number from an int, an integral float or digit text ("12", " 7.00 "); None for anything else."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw) if raw.is_integer() else None
    digits = isinstance(raw, str) and _WHOLE_NUMBER_RX.fullmatch(raw)
    return int(float(raw)) if digits else None


def _as_list(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    lone = isinstance(raw, (str, dict)) and raw
    return [raw] if lone else []


def _looks_like_id(text: Any, id_field: bool=False) -> bool:
    """Text of 3..64 characters without a space; it also needs a digit unless the field is known to hold ids."""
    if not isinstance(text, str):
        return False
    core = text.strip()
    if ' ' in core or not 3 <= len(core) <= 64:
        return False
    return id_field or bool(_DIGIT_RX.search(text))


def _id_column_rank(records: list[dict[str, Any]], field: Any) -> tuple[bool, float] | None:
    """How well one key serves as the id column (None when it cannot): every value id-like, none repeated."""
    cells = [record.get(field) for record in records]
    id_named = 'id' in str(field).lower()
    if not all(_looks_like_id(cell, id_named) for cell in cells):
        return None
    trimmed = [cell.strip() for cell in cells]
    if len(records) > 1 and len(set(trimmed)) < len(trimmed):
        return None
    return id_named, sum(map(len, trimmed)) / len(trimmed)


def _guess_id_field(records: list[dict[str, Any]]) -> str | None:
    """The first row's key that best holds ids: a key naming "id" first, then the longer average value; the earlier
    key keeps a tie."""
    chosen, chosen_rank = None, None
    for field in records[0]:
        rank = _id_column_rank(records, field)
        if rank is not None and (chosen_rank is None or rank > chosen_rank):
            chosen, chosen_rank = field, rank
    return chosen


# ---------------------------------------------------------------- JSON in and out

def _to_json(obj: Any) -> str:
    try:
        rendered = json.dumps(obj, **_COMPACT_JSON)
    except (TypeError, ValueError):
        rendered = str(obj)
    return rendered


def _decode_or_marker(blob: str) -> Any:
    try:
        return json.loads(blob)
    except ValueError:
        return _UNDECODED


def _json_candidates(body: str):
    """The whole reply first; then, only if that does not decode, the span from its first '{' to its last '}'."""
    yield body
    opening, closing = body.find('{'), body.rfind('}')
    if 0 <= opening < closing:
        yield body[opening:closing + 1]


def _parse_json_object(reply: Any) -> dict[str, Any] | None:
    """The JSON object inside a model reply (a code fence is unwrapped); None when the text holds no object."""
    if not isinstance(reply, str):
        return None
    body = reply.strip()
    fenced = _FENCE_RX.match(body)
    if fenced:
        body = fenced.group(1)
    for candidate in _json_candidates(body):
        decoded = _decode_or_marker(candidate)
        if decoded is not _UNDECODED:
            return decoded if isinstance(decoded, dict) else None
    return None


# ------------------------------------------------------------------ listing records sent as one delimited string
# Some stores send each listing as a single text such as "id_key: p-1 | title: ... | price: 12 | facts: a=b; c=d".
# Such a text is turned into a record so every reader (rows, pages, checks, the model's view) sees its fields.

_PART_SPLIT = ' | '
_KEYED_PART = re.compile(r'^\s*([A-Za-z][A-Za-z0-9_]*(?: [A-Za-z0-9_]+){0,2}):\s?(.*)$', re.S)
_PLAIN_NUMBER = re.compile(r'^-?\d+(?:\.\d+)?$')


def _scalar_from_text(text: str) -> Any:
    """A field value as the store meant it: a flag, nothing, a number, or the text itself."""
    stripped = text.strip()
    if stripped in ('True', 'true'):
        return True
    if stripped in ('False', 'false'):
        return False
    if stripped in ('None', 'null', ''):
        return None
    if _PLAIN_NUMBER.match(stripped):
        return float(stripped) if '.' in stripped else int(stripped)
    return stripped


def _pairs_from_text(text: str) -> dict[str, str] | None:
    """A "k=v; k=v" list as a dict (values kept as text); None unless most parts are such pairs."""
    pieces = [piece for piece in text.split('; ')]
    if len(pieces) < 2 or sum('=' in piece for piece in pieces) < max(2, len(pieces) * 0.7):
        return None
    pairs: dict[str, str] = {}
    last = None
    for piece in pieces:
        key, sign, value = piece.partition('=')
        if sign and key.strip() and key.strip() not in pairs and len(key) <= 60:
            last = key.strip()
            pairs[last] = value.strip()
        elif last is not None:
            pairs[last] += '; ' + piece
    return pairs


def _record_from_text(text: Any) -> dict[str, Any] | None:
    """The record a delimited listing text holds, or None when the text is not one. Parts without a key of their
    own continue the value before them (a title may itself contain the separator)."""
    if not isinstance(text, str) or text.count(_PART_SPLIT) < 2:
        return None
    fields: dict[str, str] = {}
    last = None
    for piece in text.split(_PART_SPLIT):
        keyed = _KEYED_PART.match(piece)
        if keyed and keyed.group(1) not in fields:
            last = keyed.group(1)
            fields[last] = keyed.group(2)
        elif last is None:
            return None
        else:
            fields[last] += _PART_SPLIT + piece
    if len(fields) < 3:
        return None
    record: dict[str, Any] = {}
    for key, value in fields.items():
        pairs = _pairs_from_text(value)
        record[key] = pairs if pairs is not None else _scalar_from_text(value)
    return record


def _records_from_texts(node: Any, level: int=0) -> Any:
    """`node` with every delimited listing text inside lists and dicts replaced by its record (a copy)."""
    if level > 6:
        return node
    if isinstance(node, list):
        return [_record_from_text(item) or _records_from_texts(item, level + 1) for item in node]
    if isinstance(node, dict):
        return {key: _record_from_text(value) or _records_from_texts(value, level + 1) for key, value in node.items()}
    return node


def _truncate_strings(node: Any, cap: int, level: int=0) -> Any:
    """A copy with every string cut to cap characters (plus '...'); below seven levels of nesting all is '...'."""
    if level > 7:
        return '...'
    if isinstance(node, str):
        return node if len(node) <= cap else node[:cap] + '...'
    below = level + 1
    if isinstance(node, dict):
        return {name: _truncate_strings(inner, cap, below) for name, inner in node.items()}
    if isinstance(node, list):
        return [_truncate_strings(inner, cap, below) for inner in node]
    return node


# ---------------------------------------------------------------- tool-schema conformance

def _schema_types(spec: dict[str, Any]) -> list[str]:
    declared = spec.get('type')
    if isinstance(declared, str):
        return [declared]
    return [name for name in declared or [] if isinstance(name, str)]


_KIND_TESTS = {
    'string': lambda v: isinstance(v, str),
    'integer': lambda v: isinstance(v, int) and not isinstance(v, bool),
    'number': lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    'boolean': lambda v: isinstance(v, bool),
    'array': lambda v: isinstance(v, list),
    'object': lambda v: isinstance(v, dict),
    'null': lambda v: v is None,
}


def _fits_some_kind(datum: Any, kinds: list[str]) -> bool:
    # a kind the table does not know accepts every value
    return any(_KIND_TESTS[kind](datum) if kind in _KIND_TESTS else True for kind in kinds)


def _number_from_text(raw: Any, whole: bool) -> Any:
    if not isinstance(raw, str) or not _SIGNED_NUMBER_RX.fullmatch(raw):
        return _NO_CHANGE
    parsed = float(raw)
    return int(parsed) if whole or parsed.is_integer() else parsed


def _toward_integer(raw: Any) -> Any:
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    return _number_from_text(raw, True)


def _toward_number(raw: Any) -> Any:
    return _number_from_text(raw, False)


def _toward_string(raw: Any) -> Any:
    numeric = isinstance(raw, (int, float)) and not isinstance(raw, bool)
    return str(raw) if numeric else _NO_CHANGE


def _toward_boolean(raw: Any) -> Any:
    word = raw.strip().lower() if isinstance(raw, str) else None
    return word == 'true' if word in ('true', 'false') else _NO_CHANGE


def _toward_array(raw: Any) -> Any:
    return _NO_CHANGE if isinstance(raw, list) or raw is None else [raw]


_COERCERS = {'integer': _toward_integer, 'number': _toward_number, 'string': _toward_string,
             'boolean': _toward_boolean, 'array': _toward_array}


def _coerce_toward(datum: Any, kinds: list[str]) -> Any:
    """The first easy repair, in the schema's own order of kinds: integral float to int, number text to a number,
    a number to text, "true"/"false" to a bool, a lone value wrapped in a list. Unchanged when none applies."""
    for kind in kinds:
        repair = _COERCERS.get(kind)
        repaired = _NO_CHANGE if repair is None else repair(datum)
        if repaired is not _NO_CHANGE:
            return repaired
    return datum


def _settle_enum(datum: Any, spec: dict[str, Any], path: str, problems: list[str]) -> Any:
    allowed = spec.get('enum')
    if not isinstance(allowed, list) or datum in allowed:
        return datum
    if isinstance(datum, str):
        wanted = datum.strip().lower()
        for option in allowed:
            if isinstance(option, str) and option.lower() == wanted:
                return option
    problems.append(f'{path}: use one of {allowed}')
    return datum


def _text_limits(datum: Any, spec: dict[str, Any], path: str, problems: list[str]) -> Any:
    if not isinstance(datum, str):
        return datum
    longest, shortest = spec.get('maxLength'), spec.get('minLength')
    if isinstance(longest, int) and len(datum) > longest:
        datum = datum[:longest]
    if isinstance(shortest, int) and len(datum) < shortest:
        problems.append(f'{path} is shorter than allowed')
    return datum


_BOUND_RULES = (('minimum', lambda have, edge: have < edge, 'is under the minimum'),
                ('maximum', lambda have, edge: have > edge, 'is over the maximum'))


def _numeric_limits(datum: Any, spec: dict[str, Any], path: str, problems: list[str]) -> Any:
    if isinstance(datum, (int, float)) and not isinstance(datum, bool):
        for field, beyond, words in _BOUND_RULES:
            edge = spec.get(field)
            if isinstance(edge, (int, float)) and beyond(datum, edge):
                problems.append(f'{path} {words} {edge}')
    return datum


def _sequence_limits(datum: Any, spec: dict[str, Any], path: str, problems: list[str]) -> Any:
    if not isinstance(datum, list):
        return datum
    most, fewest, member = spec.get('maxItems'), spec.get('minItems'), spec.get('items')
    if isinstance(most, int) and len(datum) > most:
        problems.append(f'{path} takes no more than {most} entries')
    if isinstance(fewest, int) and len(datum) < fewest:
        problems.append(f'{path} takes no fewer than {fewest} entries')
    if isinstance(member, dict):
        datum = [_conform_to_schema(entry, member, f'{path}[{n}]', problems) for n, entry in enumerate(datum)]
    return datum


def _mapping_limits(datum: Any, spec: dict[str, Any], path: str, problems: list[str]) -> Any:
    if not isinstance(datum, dict):
        return datum
    declared = spec.get('properties')
    if not isinstance(declared, dict):
        declared = {}
    if spec.get('additionalProperties') is False:
        strangers = [name for name in datum if name not in declared]
        problems.extend(f'{path}.{name} is not an argument this tool takes' for name in strangers)
    missing = [name for name in spec.get('required') or [] if name not in datum]
    problems.extend(f'{path}.{name} is missing and must be given' for name in missing)
    return {name: _conform_to_schema(inner, declared[name], f'{path}.{name}', problems) if name in declared else inner
            for name, inner in datum.items()}


_SCHEMA_STAGES = (_settle_enum, _text_limits, _numeric_limits, _sequence_limits, _mapping_limits)


def _conform_to_schema(datum: Any, spec: Any, path: str, problems: list[str]) -> Any:
    """Bring a value in line with a JSON schema where an easy repair exists and note in problems (path first) every
    rule it still breaks. A value of the wrong kind that cannot be repaired is noted and returned as it is."""
    if not isinstance(spec, dict):
        return datum
    allowed_kinds = _schema_types(spec)
    if allowed_kinds and not _fits_some_kind(datum, allowed_kinds):
        datum = _coerce_toward(datum, allowed_kinds)
        if not _fits_some_kind(datum, allowed_kinds):
            problems.append(f"{path} must have type {'/'.join(allowed_kinds)}")
            return datum
    for stage in _SCHEMA_STAGES:
        datum = stage(datum, spec, path, problems)
    return datum


# ---------------------------------------------------------------- walking store records

def _row_carries_id(row: Any, wanted: Any) -> bool:
    return isinstance(row, dict) and wanted in [str(cell).strip() for cell in row.values() if isinstance(cell, str)]


def _id_in_rows(tree: Any, wanted: str, level: int=0) -> bool:
    """Whether some record (a dict inside a list), at most five levels down, has a text field equal to wanted."""
    if level > 5:
        return False
    if isinstance(tree, dict):
        children = tree.values()
    elif isinstance(tree, list):
        if any(_row_carries_id(row, wanted) for row in tree):
            return True
        children = tree
    else:
        return False
    return any(_id_in_rows(child, wanted, level + 1) for child in children)


def _scalar_fields(block: dict[Any, Any]) -> dict[Any, Any]:
    return {name: cell for name, cell in block.items() if isinstance(cell, (str, int, float)) and not isinstance(cell, bool)}


def _attribute_block(record: Any) -> dict[str, Any] | None:
    """Among a record's nested dicts, the scalar fields of the one with the most of them (two at least; first wins
    a tie); None when there is none."""
    if not isinstance(record, dict):
        return None
    pools = (_scalar_fields(inner) for inner in record.values() if isinstance(inner, dict) and len(inner) >= 2)
    return max((pool for pool in pools if len(pool) >= 2), key=len, default=None)


# ---------------------------------------------------------------- compatibility notes in listing text

def _plain_code(code: str) -> str:
    """One spelling per code: lower case, a hyphen read as a space, runs of spaces collapsed."""
    return ' '.join(code.replace('-', ' ').split()).lower()


def _codes_listed_in(fragment: Any) -> list[str]:
    """Model codes a text names: plain codes, code families ("ab31xx"), and after an "except" the codes it rules
    out, written with a leading "!"."""
    text = str(fragment or '')
    halves = _CODE_EXCEPT_RX.split(text, maxsplit=1)
    kept, ruled_out = (halves[0], halves[1]) if len(halves) == 2 else (text, '')
    named = [_plain_code(code) for code in _CODE_FAMILY_RX.findall(kept) + _CODE_TOKEN_RX.findall(kept)]
    return named + ['!' + _plain_code(code) for code in _CODE_TOKEN_RX.findall(ruled_out)]


def _codes_overlap(claimed: Any, device: Any) -> bool:
    """Whether a part's claimed codes cover one of a device's codes: the same code, or a family whose fixed digits
    lead the device code at the same length; codes the claim rules out never count."""
    squeeze = lambda code: code.replace(' ', '')
    ruled_out = {squeeze(code[1:]) for code in claimed if code.startswith('!')}
    offered = [squeeze(code) for code in claimed if not code.startswith('!')]
    for target in (squeeze(code) for code in device):
        if target in ruled_out:
            continue
        for code in offered:
            stem = code.rstrip('x')
            if code == target or (stem != code and len(code) == len(target) and target.startswith(stem)):
                return True
    return False


def _upper_capture(pattern: Any, body: str) -> str:
    found = pattern.search(body)
    return found.group(1).upper() if found else ''


def _after_part_also(body: str, start: int) -> bool:
    # "part AB123 also fits ..." is read as an 'also' entry, not as a plain fits claim
    return bool(_PART_ALSO_TAIL_RX.search(body[max(0, start - 40):start]))


def _compatibility_marks(listing_text: Any) -> dict[str, Any]:
    """What a listing says about fit: its own part number, its model codes and other names, the models it works
    with, other parts' "also fits" lists and a part it points to for details."""
    body = str(listing_text or '')
    named = [code for pattern in (_MODEL_NO_RX, _ALIAS_RX) for hit in pattern.finditer(body)
             for code in _codes_listed_in(hit.group(1))]
    partners = [code for hit in _FITS_CLAIM_RX.finditer(body) if not _after_part_also(body, hit.start())
                for code in _codes_listed_in(hit.group(1))]
    elsewhere = [(hit.group(1).upper(), _codes_listed_in(hit.group(2))) for hit in _PART_ALSO_FITS_RX.finditer(body)]
    return {'own': _upper_capture(_OWN_PART_RX, body), 'codes': named, 'works': partners, 'also': elsewhere,
            'more': _upper_capture(_SEE_PART_RX, body)}


# ---------------------------------------------------------------- model clients by time left

def _cached_short_client(step: int) -> ProxyClient:
    client = _short_clients.get(step)
    if client is None:
        client = _short_clients[step] = ProxyClient(timeout=step, max_retries=1, retry_delay=1)
    return client


def _client_for_seconds(time_left: float) -> ProxyClient:
    """The shared client when 50 s or more remain; otherwise a cached client whose timeout is the time left rounded
    down to a multiple of 5 s (never under 5 s)."""
    return _model_client if time_left >= 50 else _cached_short_client(max(5, int(time_left) // 5 * 5))


# ---------------------------------------------------------------- one requested product

class WantedProduct:
    """A product the shopper asked for: its name, its shelf label, how many units, the needs the request states,
    and what the shopper's answers later added, took back or left open."""

    def __init__(self, title: str, aisle: str, count: int, firm: list[str]):
        self.product_name, self.category_label = title, aisle
        self.units_wanted, self.stated_needs = count, firm
        # filled while the interview runs
        self.answered_needs, self.withdrawn, self.indifferent_to = [], [], []
class EpisodeCore:
    """Backbone of one Corvin episode.

    Holds the run's state, reads the clock and the spend, owns both network links (the model seats and the store),
    turns the shopper's request into a brief at setup, and writes the two texts the walking model re-reads every
    turn: the ledger of what must be bought and the status line about steps and time.
    """

    def __init__(self, problem_data: dict[str, Any]) -> None:
        self._bind_world(problem_data['environment'])
        self._init_link_state()
        self._init_task_state()
        self._init_listing_state()
        self._init_cart_state()
        self._init_review_state()
        self._init_question_state()
        self._init_trail_state()

    # ------------------------------------------------------------------ state set-up

    def _bind_world(self, environment: dict[str, Any]) -> None:
        """Session binding, the policy view, the usable tool specs and the step cap; starts the episode clock."""
        self._session_binding = dict(environment['binding'])
        view = self.world_view = environment['policy_view']
        self.request_text = str(view.get('query') or '')
        self.tool_specs = [spec for spec in view.get('tools') or [] if self._is_tool_spec(spec)]
        self.tool_schemas = {spec['function']['name']: self._schema_or_blank(spec['function'].get('parameters')) for spec in self.tool_specs}
        self.step_limit = self._declared_step_cap(view.get('max_steps'))
        self._clock_origin = time.monotonic()

    @staticmethod
    def _is_tool_spec(spec: Any) -> bool:
        if not isinstance(spec, dict) or not isinstance(spec.get('function'), dict):
            return False
        return isinstance(spec['function'].get('name'), str)

    @staticmethod
    def _schema_or_blank(params: Any) -> dict[str, Any]:
        if isinstance(params, dict):
            return params
        return {'type': 'object', 'properties': {}}

    @staticmethod
    def _declared_step_cap(raw: Any) -> int:
        """The world's step allowance; a fractional float is cut down, anything unusable falls back to 30."""
        cap = _as_int(raw)
        if cap is None and isinstance(raw, float):
            cap = int(raw)
        return cap if cap and cap > 0 else 30

    def _init_link_state(self) -> None:
        """Counters of the two links: steps, model requests, spend, retry pauses and the trace."""
        self.steps_used: int = 0
        self.episode_over: bool = bool(self.world_view.get('finished'))
        self.out_of_funds = self.credit_exhausted = False
        self._pause_spent = 0
        self.failed_turns = self.model_requests = self.loop_turns = 0
        self.model_spend: float = 0.0
        self.event_log = []

    def _init_task_state(self) -> None:
        """The brief and what it fixes: roles, products, budget, message room, process steps."""
        self.parsed_brief = {}
        self.tool_roles = {}
        self.wanted_products = []
        self.budget_limit = None
        self.message_limit = self.messages_sent = 0
        self.process_steps = []
        self._kept_for_process = self.process_messages_used = 0
        self.process_checks_passed = set()
        self.pointer_musts = []
        self.pointer_values = {}
        self.fit_pairs = []

    def _init_listing_state(self) -> None:
        """What has been seen of the catalogue: rows, full records, result shapes and field guesses."""
        self.listing_rows = {}
        self.full_records = {}
        self.listing_order = []
        self.id_field = self.price_field = self.stock_field = None
        self.shows_attributes = {}
        self.keys_per_tool = {}
        self.viewed_keys = {}
        self.read_results_cache = {}
        self.result_layouts = {}
        self.auto_pages_read: int = 0
        self.paged_results = []
        self.opened_ids = set()
        self.id_params = set()
        self.offer_read_step = {}
        self.sold_out_at = {}
        self.unavailable_ids = set()

    def _init_cart_state(self) -> None:
        """Cart lines and the bookkeeping around cart changes, views and refusals."""
        self.cart_units = {}
        self.cart_modified: bool = False
        self.cart_changed_step: int = -1
        self.cart_viewed_step: int = -1
        self.cart_snapshot = None
        self.last_cart_reply: str = ''
        self.checkout_refused_step: int = -1
        self.checkout_attempts: int = 0
        self.built_lines = []
        self.line_hold_count = self.sold_hold_count = self.fit_hold_count = 0
        self.state_version: int = 0

    def _init_review_state(self) -> None:
        """Independent reviews, leads, decline handling and the store's notes."""
        self.reviewed_signatures = set()
        self.decline_reviewed = self.decline_pushed = self.late_decline_blocked = False
        self.decline_block_count = self._decline_tries = 0
        self.review_leads = []
        self.leads_by_item = {}
        self.review_verdicts = {}
        self.confirmed_lead = None
        self.lead_bought: bool = False
        self.withdrawn = set()  # listing ids the model took out of the cart (read by the turn loop)
        self.seen_note_lines = set()
        self.sellout_note_lines = set()
        self.change_notes = []
        self.note_named_listing = self.note_reports_sellout = False
        self.cheaper_check_marks = set()

    def _init_question_state(self) -> None:
        """The interview: question lanes per item, their hits and messages, follow-ups."""
        self.interview_done = self.late_question_used = False
        self._asked_pairs = []
        self.question_lanes = {}
        self.lane_hits = {}
        self.lane_messages = {}
        self.lane_asked_labels = {}
        self.lane_empty_streak = {}
        self.adopted_labels = {}
        self.hinted_item_order = []
        self.single_product_items = set()
        self.pending_followups = []
        self.lists_allowed: bool = True

    def _init_trail_state(self) -> None:
        """The walking model's conversation, the action trail, searches and page chains."""
        self.chat_history = []
        self.action_record = []
        self.action_steps = []
        self.recent_calls = []
        self.queued_notes = []
        self.recent_searches = []
        self.capped_searches = set()
        self.broader_search_done: bool = False
        self.link_codes_tried = set()
        self.link_search_count: int = 0
        self.link_search_results = {}
        self.chain_rounds = self.chain_block_count = 0
        self.compare_ok_size = self.compare_fail_size = 0

    # ------------------------------------------------------------------ small readers

    def steps_remaining(self):
        used = self.steps_used
        return self.step_limit - used

    def elapsed_seconds(self):
        now = time.monotonic()
        return now - self._clock_origin

    def seconds_for_model(self):
        """Wall-clock room still usable by a model request (the episode limit less the safety margin)."""
        room = EPISODE_SECONDS - SAFETY_SECONDS - self.elapsed_seconds()
        return room

    def spare_messages(self):
        """Messages to the shopper still free once the ones held for process steps are set aside."""
        held_back = max(0, self._kept_for_process - self.process_messages_used)
        return self.message_limit - self.messages_sent - held_back

    def cart_signature(self):
        lines = sorted(self.cart_units.items())
        return _to_json(lines)

    def _page_view_tools(self) -> set[str]:
        """Tools that show a listing's own page: the offer view (else the open view) and the side-by-side view."""
        roles = self.tool_roles
        candidates = (roles.get('offer') or roles.get('open'), roles.get('compare'))
        return {name for name in candidates if name}

    def attribute_tool(self):
        """The single-id tool known to show attributes; failing that the open tool, unless it is known not to."""
        for name, shows in self.shows_attributes.items():
            if shows and self.single_id_param(name):
                if name:
                    return name
                break
        opener = self.tool_roles.get('open')
        usable = self.shows_attributes.get(opener or '') is not False and self.single_id_param(opener)
        return opener if usable else None

    def compare_lacks_attributes(self):
        flag = self.shows_attributes.get(self.tool_roles.get('compare') or '')
        if flag is not None:
            return flag is False
        facts_tool = self.attribute_tool() or ''
        return self.shows_attributes.get(facts_tool) is True or bool(self.tool_roles.get('offer'))

    def process_check_pending(self, phase: str) -> bool:
        """A process step is due at this phase and has not yet been done for the current cart and state."""
        if not any(entry['when'] == phase for entry in self.process_steps):
            return False
        stamp = f'{self.cart_signature()}#{self.state_version}'
        return (phase, stamp) not in self.process_checks_passed

    def closing_steps(self):
        """Steps the ending needs: two when an order-time process step still has to send a message, else one."""
        needs_message = any(entry['when'] == 'order' and entry['messages'] > 0 for entry in self.process_steps)
        if needs_message and self.messages_sent < self.message_limit and self.process_check_pending('order'):
            return 2
        return 1

    def review_models(self, ceiling: float=SOFT_COST_LIMIT) -> list[str]:
        """Judges the spend still allows: strong then fast while under the ceiling with margin, else fast, else none."""
        spent = self.model_spend
        if not spent <= HARD_COST_LIMIT:
            return []
        if spent <= ceiling and spent + STRONG_COST_MARGIN <= HARD_COST_LIMIT:
            return [STRONG_MODEL, FAST_MODEL]
        return [FAST_MODEL]

    # ------------------------------------------------------------------ model link

    def request_model(self, conversation: list[dict[str, Any]], seats: list[str], allowance: int, offered: list[dict[str, Any]] | None=None) -> dict[str, Any] | None:
        """Try each seat in turn and return the first usable reply; a credit refusal ends the whole attempt."""
        answer = None
        for seat in seats:
            if not self._seat_open():
                break
            self.model_requests += 1
            answer, status = self._ride_seat(seat, conversation, allowance, offered)
            if answer is not None:
                break
            if status == 402:
                # a refusal that asks to retry later (too many requests in flight) only ends this request
                self.credit_exhausted = not self._refusal_is_temporary()
                break
        return answer

    def _refusal_is_temporary(self) -> bool:
        """Whether the newest model failure says it will pass (retry later, requests in flight) rather than that
        the credit is gone."""
        failure = next((entry for entry in reversed(self.event_log) if entry.get('kind') == 'llm_fail'), {})
        wording = str(failure.get('body') or '').lower()
        return 'in_flight' in wording or 'in-flight' in wording or 'retry' in wording

    def _seat_open(self) -> bool:
        """False when the request cap is reached or too little wall clock is left (that case is logged)."""
        if self.model_requests >= MAX_MODEL_REQUESTS:
            return False
        if self.seconds_for_model() < MIN_REQUEST_SECONDS:
            moment = round(self.elapsed_seconds(), 1)
            self.event_log.append({'kind': 'late_call_skipped', 'at': moment})
            return False
        return True

    def _ride_seat(self, seat: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        """Post to one seat, waiting a jittered pause between tries while the retry rules allow another."""
        attempt = 0
        while True:
            reply, status = self._post_chat(seat, messages, max_tokens, tools)
            if reply is not None:
                return reply, status
            if not self._may_retry(status, attempt):
                return None, status
            base = RETRY_PAUSES[attempt]
            jittered = base * (0.8 + 0.4 * random.random())
            self._pause_spent += base
            attempt += 1
            time.sleep(jittered)

    def _may_retry(self, status: Any, attempt: int) -> bool:
        """Whether a failed try earns another one after the pause RETRY_PAUSES[attempt].

        Retried: no status, 402, 408, 425, 429 and 5xx. No status and 408 get one retry, the rest one per entry of
        RETRY_PAUSES; the summed pauses stay within MAX_TOTAL_PAUSE and a retry must still leave room for a request.
        """
        retryable = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
        if not retryable or attempt >= (1 if status in (None, 408) else len(RETRY_PAUSES)):
            return False
        base = RETRY_PAUSES[attempt]
        if self._pause_spent + base > MAX_TOTAL_PAUSE:
            return False
        return not self.seconds_for_model() - base * 1.2 < MIN_REQUEST_SECONDS

    def _post_chat(self, seat: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        body = self._chat_body(seat, messages, max_tokens, tools)
        link = _client_for_seconds(self.seconds_for_model())
        return self._read_chat_result(seat, link.post_verbose('/inference/chat/completions', body))

    @staticmethod
    def _chat_body(seat: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> dict[str, Any]:
        """The completion request; Anthropic seats get cache marks on the first and the last text message."""
        outgoing = messages
        if seat.startswith('anthropic/') and len(messages) > 1:
            outgoing = [dict(entry) for entry in messages]
            for spot in (0, len(outgoing) - 1):
                text = outgoing[spot].get('content')
                if isinstance(text, str) and text:
                    outgoing[spot]['content'] = [{'type': 'text', 'text': text, 'cache_control': {'type': 'ephemeral'}}]
        body: dict[str, Any] = {'model': seat, 'messages': outgoing, 'max_tokens': max_tokens, 'temperature': 0}
        if tools:
            body.update(tools=tools, tool_choice='auto')
        return body

    def _read_chat_result(self, seat: str, result: Any) -> tuple[dict[str, Any] | None, Any]:
        """(message, 200) for a usable reply; otherwise (None, status) with the failure logged."""
        data = result.data
        if isinstance(data, dict) and data.get('choices'):
            return self._take_choice(seat, data)
        fault = result.error or {}
        status = fault.get('status')
        if isinstance(data, dict) and isinstance(data.get('error'), dict):
            status = _as_int(data['error'].get('code'))
            fault = {'body': data['error'].get('message')}
        self.event_log.append({'kind': 'llm_fail', 'model': seat, 'status': status, 'body': str(fault.get('body') or '')[:200]})
        return None, status

    def _take_choice(self, seat: str, data: dict[str, Any]) -> tuple[dict[str, Any] | None, Any]:
        usage = data.get('usage') or {}
        cost = usage.get('cost')
        if isinstance(cost, (int, float)):
            self.model_spend += float(cost)
        reply = data['choices'][0].get('message') or {}
        if reply.get('content') or reply.get('tool_calls'):
            return reply, 200
        self.event_log.append({'kind': 'llm_fail', 'model': seat, 'status': 0})
        return None, 0

    def _links_spent(self) -> bool:
        return self.out_of_funds or self.credit_exhausted or self.model_requests >= MAX_MODEL_REQUESTS

    def model_json(self, prompt_text: str, seats: list[str], allowance: int=1800) -> dict[str, Any] | None:
        """Ask the seats one at a time for a JSON object; stop early once money or the request cap runs out."""
        parsed = None
        for seat in seats:
            reply = self.request_model([{'role': 'user', 'content': prompt_text}], [seat], allowance)
            parsed = _parse_json_object((reply or {}).get('content'))
            if parsed is not None or self._links_spent():
                break
        return parsed

    # ------------------------------------------------------------------ store link

    def store_call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        """One store action, sent up to four times; step clashes are re-synced, hard refusals end it at once."""
        envelope = dict(self._session_binding)
        envelope['step'] = self.steps_used + 1
        envelope['action'] = {'name': tool, 'args': arguments}
        outcome = None
        for _attempt in range(4):
            reply = _store_client.post_verbose('/environment/call', envelope)
            if isinstance(reply.data, dict):
                outcome = self._take_store_reply(tool, arguments, reply.data)
                break
            fault = reply.error or {}
            status = fault.get('status')
            text = str(fault.get('body') or '')
            self.event_log.append({'kind': 'env_fail', 'status': status, 'body': text[:200]})
            if not self._store_retry_ok(status, text, envelope):
                break
        return outcome

    def _take_store_reply(self, name: str, args: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Adopt the store's step count and finish flag; a page view that worked marks its ids as opened."""
        stamp = data.get('step')
        self.steps_used = stamp if isinstance(stamp, int) else self.steps_used + 1
        if data.get('finished'):
            self.episode_over = True
        if 'observation' in data:
            data['observation'] = _records_from_texts(data['observation'])
        seen = data.get('observation')
        if name in self._page_view_tools() and not (isinstance(seen, dict) and seen.get('error')):
            for value in args.values():
                batch = value if isinstance(value, list) else [value]
                self.opened_ids.update(v.strip() for v in batch if isinstance(v, str))
        return data

    def _store_retry_ok(self, status: Any, text: str, envelope: dict[str, Any]) -> bool:
        """Whether a failed store call may be sent again (409 has its own rules; listed client errors are final)."""
        if status == 409:
            return self._settle_step_clash(text, envelope)
        if status in (400, 401, 402, 403, 404, 422):
            if status == 402:
                self.out_of_funds = True
            return False
        return True

    def _settle_step_clash(self, text: str, envelope: dict[str, Any]) -> bool:
        lowered = text.lower()
        if any(mark in lowered for mark in ('finish', 'quarantin', 'finaliz')):
            self.episode_over = True
            return False
        wanted = re.search('expected (\\d+)', text)
        if wanted:
            self.steps_used = int(wanted.group(1)) - 1
        elif 'reuse' in lowered or 'different action' in lowered:
            self.steps_used += 1
        else:
            return False
        envelope['step'] = self.steps_used + 1
        return True

    # ------------------------------------------------------------------ setup: reading the request

    def parse_request(self):
        """Turn the request into the brief (one or two model reads), then apply it to the episode state."""
        brief = self._read_brief(self._brief_prompt())
        self._apply_brief(brief)

    @staticmethod
    def _tool_line(fn: dict[str, Any]) -> str:
        return f"- {fn['name']}: {fn.get('description', '')} | parameters: {_to_json(fn.get('parameters') or {})}"

    def _brief_prompt(self) -> str:
        catalogue = '\n'.join(self._tool_line(spec['function']) for spec in self.tool_specs)
        prompt = BRIEF_PROMPT + '\nTOOLS:\n' + catalogue + '\n\nREQUEST:\n' + self.request_text
        opening = self.world_view.get('opening')
        if opening is not None:
            prompt += "\n\nTHE STORE'S OPENING NOTE:\n" + _to_json(_truncate_strings(opening, 1500))
        return prompt

    @staticmethod
    def _brief_usable(brief: dict[str, Any]) -> bool:
        return bool(_as_list(brief.get('items')) and isinstance(brief.get('roles'), dict) and brief['roles'].get('buy'))

    @staticmethod
    def _shelves_repeat(brief: dict[str, Any]) -> bool:
        """True when two items of the brief carry the same category label."""
        labels = [_plain_words(str(entry.get('shelf') or '')) for entry in _as_list(brief.get('items')) if isinstance(entry, dict)]
        named = [label for label in labels if label]
        return len(set(named)) != len(named)

    def _read_brief(self, prompt: str) -> dict[str, Any]:
        """The fast seat reads first; an unusable brief is read again by strong-then-fast, and a brief with a
        repeated category label gets a second opinion from the strong seat, kept only when usable."""
        first = self.model_json(prompt, [FAST_MODEL], 4000) or {}
        if self.credit_exhausted:
            return first
        if not self._brief_usable(first):
            return self.model_json(prompt, [STRONG_MODEL, FAST_MODEL], 2500) or first
        if self._shelves_repeat(first):
            second = self.model_json(prompt, [STRONG_MODEL], 2500) or {}
            if self._brief_usable(second):
                return second
        return first

    def _apply_brief(self, brief: dict[str, Any]) -> None:
        self.parsed_brief = brief
        self._adopt_roles(brief)
        self._settle_order_modes(brief)
        renumbered = self._adopt_products(brief)
        self._adopt_fits(brief, renumbered)
        self._adopt_lookups(brief)
        self._unmix_pointer_needs()
        self._adopt_limits(brief)
        self._adopt_process(brief)

    def _adopt_roles(self, brief: dict[str, Any]) -> None:
        raw = brief.get('roles')
        roles = raw if isinstance(raw, dict) else {}
        self.tool_roles = {job: tool for job, tool in roles.items() if isinstance(tool, str) and tool in self.tool_schemas}

    def _settle_order_modes(self, brief: dict[str, Any]) -> None:
        """Make sure both mode-argument slots exist; when a single tool both orders and declines and its two
        argument sets are missing or equal, ask a seat which fixed arguments pick each mode."""
        for slot in ('buy_args', 'refuse_args'):
            if not isinstance(brief.get(slot), dict):
                brief[slot] = {}
        buy, refuse = self.tool_roles.get('buy'), self.tool_roles.get('refuse')
        if not buy or buy != refuse:
            return
        paid, unpaid = brief['buy_args'], brief['refuse_args']
        if paid and unpaid and paid != unpaid:
            return
        ask = (f'One store tool, {buy}, serves two ends: some arguments make it place the order, others make it '
               'close the episode with nothing bought. Its schema: ' + _to_json(self.tool_schemas.get(buy))
               + '\nAnswer with ONE JSON object and nothing more: {"buy_args": the fixed arguments that place the '
               'order, "refuse_args": the fixed arguments that close without buying} (just the arguments that '
               'select the mode).')
        fixed = self.model_json(ask, [FAST_MODEL, STRONG_MODEL], 1500) or {}
        modes = (fixed.get('buy_args'), fixed.get('refuse_args'))
        if isinstance(modes[0], dict) and isinstance(modes[1], dict) and modes[0] != modes[1]:
            brief['buy_args'], brief['refuse_args'] = modes

    @staticmethod
    def _product_from_entry(entry: dict[str, Any]) -> WantedProduct:
        wanted = _as_int(entry.get('quantity'))
        units = wanted if wanted and wanted > 0 else 1
        musts = [str(m) for m in _as_list(entry.get('musts')) if m]
        return WantedProduct(str(entry.get('name') or 'the product'), str(entry.get('shelf') or ''), units, musts)

    def _adopt_products(self, brief: dict[str, Any]) -> dict[int, int]:
        """Build the wanted products, fold items that share a category label, and return the old->new numbering."""
        for entry in _as_list(brief.get('items')):
            if isinstance(entry, dict):
                self.wanted_products.append(self._product_from_entry(entry))
        merged, renumbered = self._fold_same_shelf(self.wanted_products)
        self.wanted_products = merged or [WantedProduct('the requested product', '', 1, [])]
        return renumbered

    def _fold_same_shelf(self, products: list[WantedProduct]) -> tuple[list[WantedProduct], dict[int, int]]:
        merged: list[WantedProduct] = []
        renumbered: dict[int, int] = {}
        for old, item in enumerate(products, 1):
            host = self._shelf_twin(merged, item)
            if host is None:
                merged.append(item)
                renumbered[old] = len(merged)
                continue
            slot, twin = host
            renumbered[old] = slot
            extra = [need for need in item.stated_needs if need not in twin.stated_needs]
            twin.stated_needs += extra
            self.event_log.append({'kind': 'items_merged', 'into': twin.product_name, 'dropped': item.product_name})
        return merged, renumbered

    def _shelf_twin(self, merged: list[WantedProduct], item: WantedProduct) -> tuple[int, WantedProduct] | None:
        """The first kept item on the same category label whose requirements do not contradict this one's."""
        found = None
        if item.category_label.strip():
            shelf = _plain_words(item.category_label)
            found = next(((slot, held) for slot, held in enumerate(merged, 1) if _plain_words(held.category_label) == shelf and not self._needs_clash(held.stated_needs, item.stated_needs)), None)
        return found

    @staticmethod
    def _needs_clash(kept: list[str], incoming: list[str]) -> bool:
        """True when both lists constrain one property to different values."""
        for a in kept:
            prop_a, _, value_a = a.partition(':')
            for b in incoming:
                prop_b, _, value_b = b.partition(':')
                if _plain_words(prop_a) == _plain_words(prop_b) and _plain_words(value_a) != _plain_words(value_b):
                    return True
        return False

    def _adopt_fits(self, brief: dict[str, Any], renumbered: dict[int, int]) -> None:
        count = len(self.wanted_products)
        for pair in _as_list(brief.get('fits')):
            if not isinstance(pair, dict):
                continue
            part = renumbered.get(_as_int(pair.get('item')) or 0)
            main = renumbered.get(_as_int(pair.get('with')) or 0)
            if not part or not main or part == main:
                continue
            if 1 <= part <= count and 1 <= main <= count and (part, main) not in self.fit_pairs:
                self.fit_pairs.append((part, main))

    def _anchor_lookup(self, text: str) -> str:
        """A lookup copied loosely is replaced by the first requirement on the same property."""
        if any(text in item.stated_needs for item in self.wanted_products):
            return text
        prop = _plain_words(text.split(':')[0])
        return next((need for item in self.wanted_products for need in item.stated_needs if _plain_words(need.split(':')[0]) == prop), text)

    def _adopt_lookups(self, brief: dict[str, Any]) -> None:
        for lookup in _as_list(brief.get('lookups')):
            text = self._anchor_lookup(str(lookup).strip())
            owner = next((number for number, item in enumerate(self.wanted_products, 1) if text in item.stated_needs), 1)
            if text and (owner, text) not in self.pointer_musts:
                self.pointer_musts.append((owner, text))

    def _unmix_pointer_needs(self) -> None:
        """Ask once per item which plain requirements only describe the listing a pointer requirement names."""
        for number, item in enumerate(self.wanted_products, 1):
            pointed = [need for owner, need in self.pointer_musts if owner == number and need in item.stated_needs]
            others = [need for need in item.stated_needs if need not in pointed]
            if pointed and others:
                self._unmix_one(number, item, pointed, others)

    def _unmix_one(self, number: int, item: WantedProduct, pointed: list[str], others: list[str]) -> None:
        prompt = (POINTER_SPLIT_PROMPT + '\n\nREQUEST:\n' + self.request_text
                  + f'\n\nPRODUCT BEING BOUGHT: {item.product_name}'
                  + '\nREQUIREMENTS THAT TAKE THEIR VALUE FROM ANOTHER LISTING:\n' + '\n'.join(pointed)
                  + '\nOTHER REQUIREMENTS:\n' + '\n'.join(others))
        answer = self.model_json(prompt, [STRONG_MODEL, FAST_MODEL], 1500) or {}
        flagged = {str(x).strip() for x in _as_list(answer.get('describe_other'))}
        moved = [need for need in others if need in flagged]
        if not moved:
            return
        anchor = pointed[0]
        joined = anchor + ' (that listing: ' + '; '.join(moved) + ')'
        item.stated_needs = [joined if need == anchor else need for need in item.stated_needs if need not in moved]
        self.pointer_musts = [(owner, joined if (owner, need) == (number, anchor) else need) for owner, need in self.pointer_musts]
        self.event_log.append({'kind': 'unmix', 'item': number, 'moved': moved})

    @staticmethod
    def _positive_amount(raw: Any) -> Any:
        """A budget as a positive number (a plain numeric string is converted), else None."""
        if isinstance(raw, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', raw):
            raw = float(raw)
        usable = isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw > 0
        return raw if usable else None

    def _adopt_limits(self, brief: dict[str, Any]) -> None:
        amount = self._positive_amount(brief.get('budget'))
        if amount is not None:
            self.budget_limit = amount
        allowance = _as_int(brief.get('messages'))
        if allowance is not None and allowance >= 0:
            self.message_limit = allowance
        elif self.tool_roles.get('ask'):
            self.message_limit = 3
        cap = _as_int(brief.get('steps'))
        if cap is not None and 10 <= cap < self.step_limit:
            self.step_limit = cap

    def _adopt_process(self, brief: dict[str, Any]) -> None:
        """Record the request's extra steps and, with a messaging tool, hold back the messages they need."""
        for entry in _as_list(brief.get('process')):
            if not isinstance(entry, dict) or not str(entry.get('step') or '').strip():
                continue
            when = str(entry.get('when') or '').strip().lower()
            self.process_steps.append({'step': str(entry['step']).strip(), 'when': when if when in ('choose', 'order', 'end') else 'order', 'messages': max(0, _as_int(entry.get('messages')) or 0)})
        if not self.tool_roles.get('ask'):
            return
        talk = [entry['messages'] for entry in self.process_steps if entry['when'] in ('order', 'end') and entry['messages']]
        self._kept_for_process = min(max(talk), self.message_limit) if talk else 0

    # ------------------------------------------------------------------ requirements no attribute states

    def _points_elsewhere(self, need: str) -> bool:
        if any(need == held for _, held in self.pointer_musts):
            return True
        return need in {str(x) for x in _as_list(self.parsed_brief.get('lookups'))}

    def _names_other_product(self, item: WantedProduct, value: str) -> bool:
        """True when the value shares a long word with another wanted product's name."""
        value_stems = _word_stems(value)
        for other in self.wanted_products:
            if other is not item and {w for w in _word_stems(other.product_name) if len(w) >= 4} & value_stems:
                return True
        return False

    def _attribute_keys_of(self, item: WantedProduct) -> tuple[int, set[str]]:
        """How many opened records of this item's category carry an attribute block, and their key names."""
        shelf_key = self.category_field()
        keys: set[str] = set()
        counted = 0
        for record in self.full_records.values():
            if shelf_key and not self.item_owns_label(item, str(record.get(shelf_key) or '')):
                continue
            block = _attribute_block(record) or {}
            if not block:
                continue
            counted += 1
            keys |= {_plain_words(k) for k in block}
        return counted, keys

    def _told_not_listed(self, number: int, need: str) -> bool:
        """A long requirement in the shopper's words whose property no attribute key of three or more opened
        records of the item names; it is shown to the model as context only."""
        if self._points_elsewhere(need):
            return False
        prop, _, value = need.partition(':')
        if not prop.strip() or len(re.findall('[a-z0-9]+', value.lower())) < 4:
            return False
        if re.search('\\b(?:listing|page|shown|other than|unlike|same as)\\b', value, re.I):
            return False
        item = self.wanted_products[number - 1]
        if self._names_other_product(item, value):
            return False
        counted, keys = self._attribute_keys_of(item)
        return counted >= 3 and not any(_word_stems(k) & _word_stems(prop) for k in keys)

    # ------------------------------------------------------------------ ledger and status for the model

    def _pointer_suffix(self, need: str) -> str:
        if need not in self.pointer_values:
            return ''
        return f' (value read from the page it points to: {self.pointer_values[need]})'

    def _ledger_item_line(self, number: int, item: WantedProduct, numbered: bool) -> str:
        live = [need for need in item.stated_needs + item.answered_needs if need not in item.withdrawn]
        told = [need for need in live if self._told_not_listed(number, need)]
        firm = [need + self._pointer_suffix(need) for need in live if need not in told]
        line = f'item {number}: {item.product_name}' if numbered else f'item: {item.product_name}'
        if item.category_label:
            line += f' (store category label: {item.category_label})'
        line += f', wanted quantity {item.units_wanted}'
        line += '; must meet: ' + ('; '.join(firm) if firm else 'nothing beyond being this kind of product')
        if told:
            line += "; said in the shopper's own words that no listing attribute carries (these settle nothing): " + '; '.join(told)
        return line

    @staticmethod
    def _pick_line(brief: dict[str, Any]) -> str:
        pick = brief.get('pick')
        if pick == 'cheapest':
            return 'choose: the fitting listing with the lowest price'
        if pick in ('most', 'least'):
            return f"choose: the {pick} {brief.get('pick_measure') or ''} of the fitting listings".replace('  ', ' ')
        return 'choose: any listing that meets everything'

    def _ledger_terms(self) -> list[str]:
        """Budget, the choice rule, stock, wishes, the request's rules and the quiet-decline wish."""
        brief = self.parsed_brief
        terms: list[str] = []
        if self.budget_limit is not None:
            span = 'per item' if brief.get('budget_scope') == 'each' else 'for the whole order'
            terms.append(f"spending cap: no more than {self.budget_limit} {brief.get('budget_unit') or ''} {span}".replace('  ', ' '))
        terms.append(self._pick_line(brief))
        if brief.get('in_stock_only', True) is not False:
            terms.append('in-stock listings only')
        terms.extend(f'nice to have, only to break ties: {wish}' for wish in _as_list(brief.get('wishes')))
        terms.extend(f'instruction in the request: {rule}' for rule in _as_list(brief.get('rules')))
        if brief.get('quiet_decline') is True:
            terms.append('when nothing fits: end without naming or proposing any listing')
        return terms

    def _ledger_process_lines(self) -> list[str]:
        timing = {'choose': 'before the choice is made', 'order': 'with the order (Corvin carries it out as you check out)', 'end': 'with an ending that buys nothing (Corvin carries it out then)'}
        return [f"step the request requires, {timing[entry['when']]}: {entry['step']}" for entry in self.process_steps]

    def _ledger_facts_line(self) -> str:
        """Which tool shows listing attributes, which page views show none, and which tool shows a listing's page."""
        facts = self.attribute_tool()
        if not facts or not (self.shows_attributes.get(facts) is True or self.tool_roles.get('offer')):
            return ''
        page_tools = self._page_view_tools()
        bare = sorted(t for t in page_tools | {self.tool_roles.get('offer') or ''} if t and self.shows_attributes.get(t) is False)
        pages = sorted(t for t in page_tools if t != facts)
        line = f'the tool that shows listing attributes is {facts}'
        if bare:
            line += f"; no attributes appear in {', '.join(bare)}"
        if pages:
            line += f"; {' or '.join(pages)} shows a listing's page (the page meant by a page check or a chain of pages)"
        return line

    def ledger_text(self):
        numbered = len(self.wanted_products) > 1
        lines = [self._ledger_item_line(number, item, numbered) for number, item in enumerate(self.wanted_products, 1)]
        lines += self._ledger_terms()
        lines += self._ledger_process_lines()
        facts_line = self._ledger_facts_line()
        if facts_line:
            lines.append(facts_line)
        return '\n'.join('- ' + line for line in lines)

    def _step_pressure(self, left: int, closing: int) -> str:
        """The step-budget warning matching how few steps remain and whether a checkout was tried."""
        untried = self.checkout_attempts == 0
        if left <= closing + 1:
            return (' LAST STEPS: if the cart has a line for every item, check out right away; end with no purchase'
                    ' only if your searching proved that some item has no listing that fits (a lack of steps proves'
                    ' nothing of the kind).')
        if left <= 6 or (left <= len(self.wanted_products) + 5 and untried):
            return ' Only a few steps remain: put the chosen lines in the cart and check out at once.'
        if untried and left <= 12:
            return ' The step budget is getting short: fix your choices, fill the cart and check out in the next few steps.'
        if untried and not self.cart_modified and self.steps_used >= self.step_limit // 2:
            return (" Half of the steps are gone and nothing is in the cart yet: add a fitting listing for every item"
                    " now (the store can still sell one out).")
        return ''

    def _status_extras(self) -> str:
        """Pending page look-ups, review candidates, the compatibility plan and the latest changes."""
        extra = ''
        live = {need for _, need in self.live_requirements()}
        unresolved = [need for _, need in self.pointer_musts if need in live and need not in self.pointer_values]
        if unresolved and self.checkout_attempts == 0:
            extra += ('\n[look these up before anything else: open the listing each one points to, then every listing'
                      ' that page refers to; the value is on the final page] ' + ' | '.join(unresolved[:3]))
        if self.review_leads:
            extra += '\n[candidates to review] ' + ', '.join(self.review_leads)
        if self.fit_pairs and self.checkout_attempts == 0:
            plan = self.compatibility_plan()
            if plan:
                extra += '\n' + plan
        if self.change_notes:
            extra += '\n[changes up to now] ' + ' | '.join(self.change_notes[-4:])
        return extra

    def status_text(self):
        left = self.steps_remaining()
        closing = self.closing_steps()
        text = f'[Corvin] {self.steps_used} of {self.step_limit} steps spent, {left} remaining.' + self._step_pressure(left, closing)
        if self.elapsed_seconds() >= HURRY_AFTER and left > closing + 1:
            text += (' THE CLOCK IS ALMOST OUT: put the chosen lines in the cart and check out within two more turns;'
                     ' an episode that hits the time limit scores nothing.')
        text += self._status_extras()
        return text + '\n[ledger]\n' + self.ledger_text()
class EpisodeRecords:
    """What the episode knows about the store's listings: which dicts are listings and under which id, the merged
    row and full page of each, prices and stock, the attribute keys per category label, how a search result is
    laid out (rows, total, next-page token), and the extra store calls Corvin makes on its own (further pages,
    pages a cart line points to, cheaper rows below a cart line's price)."""

    # ------------------------------------------------------------------ small value tests

    @staticmethod
    def _numeric_value(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    @staticmethod
    def _true_int(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    @staticmethod
    def _asks_later_page(given: Any) -> bool:
        """True when the call already asked for something past the first page (a page number above 1, or any
        non-blank cursor or token)."""
        if isinstance(given, int) and not isinstance(given, bool):
            return given not in (0, 1)
        return bool(str(given or '').strip())

    @staticmethod
    def _price_below(price: float) -> int:
        whole = int(price)
        return whole - 1 if float(price).is_integer() else whole

    @staticmethod
    def _short_label_text(value: Any) -> bool:
        return isinstance(value, str) and 0 < len(value) <= 40 and not re.search('\\d{3}', value)

    def _send_conformed(self, tool: str, raw: dict[str, Any]) -> tuple[Any, dict[str, Any] | None]:
        """Fit the arguments to the tool's schema and make the store call: (fitted arguments, reply). The reply is
        None when the arguments could not be fitted (no call made) or the store gave nothing back."""
        problems: list[str] = []
        fitted = _conform_to_schema(raw, self.tool_schemas.get(tool), 'args', problems)
        return fitted, (None if problems else self.store_call(tool, fitted))

    # ------------------------------------------------------------------ taking in what the store showed

    def take_observation(self, tool: str, seen: Any) -> str:
        """Store the listings an observation carries; free text in it goes to the note reader, and whatever the
        note changed is queued for the model, tagged with the tool that showed it."""
        self.absorb_records(seen)
        fresh = self.free_text_lines(seen)
        reading = self.read_store_note(fresh) if fresh else ''
        if reading:
            self.queued_notes.append(f'(noted on {tool}) {reading}')
        return reading

    def absorb_records(self, seen: Any) -> None:
        self._soak_node(seen, 0, False)

    def _soak_node(self, node: Any, level: int, row_context: bool) -> None:
        """Walk an observation (six levels at most): a dict that is a listing is merged and not entered; any other
        dict or list is entered, a list's items being treated as rows when the list reads as listing rows."""
        if level <= 6 and isinstance(node, list):
            as_rows = self._rows_hold_listings(node)
            for child in node:
                self._soak_node(child, level + 1, as_rows)
        elif level <= 6 and isinstance(node, dict):
            listing = self.listing_id_of(node, row_context, level)
            if listing:
                self._merge_listing_row(listing, node)
            else:
                for child in node.values():
                    self._soak_node(child, level + 1, False)

    def _merge_listing_row(self, listing: str, fresh: dict[str, Any]) -> None:
        if listing not in self.listing_rows:
            self.listing_order.append(listing)
        known = self.listing_rows.setdefault(listing, {})
        # a boolean field that flipped tells whether the listing went out of (False) or back into (True) stock
        flipped = {value for key, value in fresh.items() if isinstance(value, bool) and known.get(key) is (not value)}
        if False in flipped:
            self.unavailable_ids.add(listing)
        elif True in flipped:
            self.unavailable_ids.discard(listing)
        known.update(fresh)
        if _attribute_block(fresh):
            self.full_records[listing] = dict(known)
        elif listing in self.full_records:
            self.full_records[listing].update(fresh)

    def _rows_hold_listings(self, items: list[Any]) -> bool:
        """Whether a list reads as listing rows (mostly dicts of 3+ fields sharing an id field); the first such list
        also fixes the id field for the episode."""
        dicts = [entry for entry in items if isinstance(entry, dict) and len(entry) >= 3]
        mostly = bool(dicts) and 2 * len(dicts) >= len(items)
        current = self.id_field
        if not mostly or (current and all(_looks_like_id(entry.get(current), True) for entry in dicts)):
            return mostly
        guess = _guess_id_field(dicts)
        if guess:
            self.id_field = guess
        return bool(guess)

    def listing_id_of(self, record: Any, row_context: bool=False, level: int=0) -> str | None:
        """The listing id a dict carries, or None. Off the top level and outside a row list, a dict without nested
        dicts is not taken for a listing; a guessed key counts when it names an id or its value is a known id."""
        found = None
        if isinstance(record, dict) and len(record) >= 3:
            current = self.id_field
            if current and _looks_like_id(record.get(current), True):
                found = record[current].strip()
            elif row_context or level == 0 or any(isinstance(inner, dict) for inner in record.values()):
                guess = _guess_id_field([record])
                if guess:
                    candidate = record[guess].strip()
                    found = candidate if 'id' in str(guess).lower() or candidate in self.listing_rows else None
        return found

    def _rows_with_ids(self, rows: list[Any]) -> list[tuple[str, dict[str, Any]]]:
        pairs = []
        for row in rows:
            listing = self.listing_id_of(row, True) if isinstance(row, dict) else None
            if listing:
                pairs.append((listing, row))
        return pairs

    def listing_records(self, seen: Any) -> list[tuple[str, dict[str, Any]]]:
        """(id, record) for the observation itself when it is one listing, else for the rows of its lists (top
        level, or one dict down)."""
        if isinstance(seen, list):
            return self._rows_with_ids(seen)
        pairs: list[tuple[str, dict[str, Any]]] = []
        if isinstance(seen, dict):
            whole = self.listing_id_of(seen)
            if whole:
                return [(whole, seen)]
            for part in seen.values():
                if isinstance(part, list):
                    pairs.extend(self._rows_with_ids(part))
                elif isinstance(part, dict):
                    for inner in part.values():
                        if isinstance(inner, list):
                            pairs.extend(self._rows_with_ids(inner))
        return pairs

    def remember_view_keys(self, tool: str, seen: Any) -> None:
        """Note which fields this tool showed for each listing, stamped with the state version, so that an
        unchanged listing need not be opened again."""
        for listing, record in self.listing_records(seen):
            fields = set(record)
            self.keys_per_tool.setdefault(tool, set()).update(fields)
            stamp, held = self.viewed_keys.get(listing, (-1, set()))
            now = self.state_version
            self.viewed_keys[listing] = (now, held | fields if stamp == now else set() | fields)

    def view_unchanged(self, tool: str, listing: str) -> bool:
        wanted = self.keys_per_tool.get(tool)
        stamp, held = self.viewed_keys.get(listing, (-1, set()))
        return bool(wanted) and stamp == self.state_version and wanted <= held and listing not in self.sold_out_at

    # ------------------------------------------------------------------ price and stock

    def listing_price(self, listing: str) -> float | None:
        field = self.price_field
        raw = (self.listing_rows.get(listing) or {}).get(field) if field else None
        if isinstance(raw, dict):
            raw = next((inner for inner in raw.values() if self._numeric_value(inner)), None)
        if isinstance(raw, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', raw):
            raw = float(raw)
        return float(raw) if self._numeric_value(raw) else None

    def can_buy(self, listing: str) -> bool:
        flag = (self.listing_rows.get(listing) or {}).get(self.stock_field) if self.stock_field else None
        return not self.is_sold_out(listing) and flag is not False and not (self._numeric_value(flag) and flag <= 0)

    def _prices_climb(self, seen: Any) -> bool:
        """Whether the result's rows (three or more, all priced) already come cheapest first."""
        prices = [self.listing_price(listing) for listing, _ in self.listing_records(seen)]
        return len(prices) >= 3 and None not in prices and all(prices[k] <= prices[k + 1] for k in range(len(prices) - 1))

    # ------------------------------------------------------------------ tools and result layouts

    def single_id_param(self, tool: str | None) -> str | None:
        """The one parameter a tool needs to show one listing: a free-text string that is the only required
        argument (or the only free-text one when nothing is required); None otherwise."""
        spec = self.tool_schemas.get(tool or '') or {}
        needed = spec.get('required') or []
        free_text = [key for key, sub in (spec.get('properties') or {}).items() if isinstance(sub, dict) and 'string' in _schema_types(sub) and 'enum' not in sub]
        choice = next((key for key in needed if key in free_text), None) or (free_text[0] if len(free_text) == 1 else None)
        return choice if tool and choice and all(key == choice for key in needed) else None

    def _result_shape(self, tool: str, seen: Any) -> dict[str, Any] | None:
        """How a tool's result is laid out (rows key, total, next-page key and parameter, price and stock keys),
        asked of the model once per tool on the first result holding a list of rows."""
        if tool in self.result_layouts:
            return self.result_layouts[tool]
        row_lists = [key for key, value in seen.items() if isinstance(value, list) and len(value) >= 2 and all(isinstance(row, dict) for row in value)] if isinstance(seen, dict) else []
        layout = None
        if row_lists:
            self.result_layouts[tool] = None
            schema = self.tool_schemas.get(tool) or {}
            answer = self.model_json(self._shape_prompt(tool, schema, seen, row_lists), [FAST_MODEL, STRONG_MODEL], 2500) or {}
            rows_key = answer.get('rows') if answer.get('rows') in row_lists else None
            if rows_key:
                layout = self._shape_from_answer(answer, rows_key, seen, schema.get('properties') or {})
                self.result_layouts[tool] = layout
                self.price_field = self.price_field or layout['price']
                self.stock_field = self.stock_field or layout['stock']
                self.event_log.append({'kind': 'shape', 'tool': tool, 'shape': layout})
        return layout

    @staticmethod
    def _shape_prompt(tool: str, schema: dict[str, Any], seen: dict[str, Any], row_lists: list[str]) -> str:
        sample = {key: (value[:3] if key in row_lists else value) for key, value in seen.items()}
        return (RESULT_LAYOUT_PROMPT + '\n\nTOOL: ' + tool + ' | parameter schema: ' + _to_json(schema)
                + '\n\nRESULT (only the first rows):\n' + _to_json(_truncate_strings(sample, 120))[:4000])

    def _shape_from_answer(self, answer: dict[str, Any], rows_key: str, seen: dict[str, Any], params: Any) -> dict[str, Any]:
        """Keep each key the model named only when it really exists where it should: the total and the next-page
        value in the result, the next-page parameter among the tool's parameters, price and stock in a row."""
        first_row = seen[rows_key][0]
        places = (('count', seen), ('next', seen), ('next_arg', params), ('price', first_row), ('stock', first_row))
        layout: dict[str, Any] = {'rows': rows_key}
        for slot, where in places:
            named = answer.get(slot)
            layout[slot] = named if isinstance(named, str) and named in where else None
        if layout['count'] and not self._true_int(seen[layout['count']]):
            layout['count'] = None
        return layout

    def _paging_state(self, tool: str, seen: Any) -> tuple[int, int, str, Any] | None:
        """(total matched, rows shown, next-page parameter, next-page value or None) for a paged result."""
        layout = self._result_shape(tool, seen)
        state = None
        if layout and all(layout[slot] for slot in ('count', 'next', 'next_arg')) and isinstance(seen, dict):
            rows, total, token = (seen.get(layout[slot]) for slot in ('rows', 'count', 'next'))
            token_ok = token is None or (isinstance(token, (str, int)) and not isinstance(token, bool))
            if isinstance(rows, list) and rows and self._true_int(total) and token_ok:
                state = (total, len(rows), layout['next_arg'], token)
        return state

    # ------------------------------------------------------------------ further pages of a search

    def read_more_pages(self, tool: str, args: dict[str, Any], seen: Any) -> str:
        """For a cheapest / most / least pick, fetch up to two further pages of a short first-page result (one step
        each); when more would be needed, tell the model how to narrow or page on instead."""
        self.paged_results = []
        goal = self.parsed_brief.get('pick')
        state = self._paging_state(tool, seen) if goal in ('cheapest', 'most', 'least') else None
        note = ''
        if state and self._page_on_wanted(goal, state, args, seen):
            note = self._extend_first_page(tool, args, state)
        return note

    def _page_on_wanted(self, goal: Any, state: tuple[int, int, str, Any], args: dict[str, Any], seen: Any) -> bool:
        """A next page exists, the call asked for the first page, rows are missing, and (for the cheapest) the
        shown rows are not already sorted by price."""
        total, shown, param, token = state
        first_page = not self._asks_later_page(args.get(param))
        return token is not None and first_page and total > shown and not (goal == 'cheapest' and self._prices_climb(seen))

    def _extend_first_page(self, tool: str, args: dict[str, Any], state: tuple[int, int, str, Any]) -> str:
        total, shown, param, token = state
        pages = self._pages_to_fetch(total, shown)
        if not pages:
            return (f'\n[Corvin] {total} listings matched this search but only {shown} came back, and not sorted by price.'
                    ' Every fitting listing matters for the pick: search again using narrower words (such as the'
                    f" requirement's value) or fetch the next page with {param}={_to_json(token)}.")
        texts, token = self._fetch_extra_pages(tool, args, param, token, pages)
        self.event_log.append({'kind': 'pages', 'tool': tool, 'read': len(texts)})
        tail = ''
        if token is not None:
            tail = (f'\n[Corvin] Of the {total} matching listings Corvin has now seen {shown * (len(texts) + 1)} rows;'
                    " to see the remainder, add the requirement's value to the search words or keep paging.")
        return '\n[Corvin fetched further pages of this search on its own, one step per page]\n' + '\n'.join(texts) + tail if texts else ''

    def _pages_to_fetch(self, total: int, shown: int) -> int:
        """Pages still missing, at most two; 0 when even that would break the paging cap or leave fewer than 12
        steps beyond the closing ones."""
        pages = min(2, (total - 1) // shown)
        room = self.steps_remaining() - self.closing_steps()
        fits = self.auto_pages_read + pages <= SELF_PAGING_CAP and room - pages >= 12
        return pages if fits else 0

    def _fetch_extra_pages(self, tool: str, args: dict[str, Any], param: str, token: Any, pages: int) -> tuple[list[str], Any]:
        """Fetch the pages one store call each; stops early when a call fails, the episode ends or no next page
        is offered. Returns the page texts and the last next-page value."""
        texts: list[str] = []
        while token is not None and len(texts) < pages:
            fitted, reply = self._send_conformed(tool, {**args, param: token})
            if reply is None or self.episode_over:
                return texts, token
            self.auto_pages_read += 1
            page = reply.get('observation')
            self.take_observation(tool, page)
            self.paged_results.append(page)
            rendered = self.render_result(page)
            self.read_results_cache[tool + _to_json(fitted)] = (self.state_version, self.steps_used, rendered)
            texts.append(f'{param}={_to_json(token)}: {rendered}')
            following = self._paging_state(tool, page)
            token = following[3] if following else None
        return texts, token

    # ------------------------------------------------------------------ notes added after a search

    def price_cap_note(self, tool: str, args: dict[str, Any], seen: Any) -> str:
        """Flag a search that ignored the maximum-price argument it was given, with the browse mode to retry in."""
        params = (self.tool_schemas.get(tool) or {}).get('properties') or {}
        cap_keys = [key for key, value in args.items() if key in params and self._numeric_value(value) and re.search('max|most|under|below|limit|cap|upto|up_to', key, re.I) and self._param_is_money(params, key)]
        note = ''
        if cap_keys and self.price_field:
            cap_key = sorted(cap_keys, key=lambda key: args[key])[0]
            cap = args[cap_key]
            pricier = [listing for listing, _ in self.listing_records(seen) if (self.listing_price(listing) or 0) > cap]
            if pricier:
                hint = self._cheaper_listing_hint(args, params, cap_key)
                note = f'\n[Corvin] {cap_key}={cap:g} was not applied by this search: {len(pricier)} of the rows it returned cost more{hint}.'
        return note

    @staticmethod
    def _param_is_money(params: dict[str, Any], key: str) -> bool:
        spec = params.get(key)
        about = (spec.get('description') if isinstance(spec, dict) else None) or ''
        return bool(re.search('price|cost|amount|budget|spend|money', key + ' ' + str(about), re.I))

    @staticmethod
    def _cheaper_listing_hint(args: dict[str, Any], params: dict[str, Any], cap_key: str) -> str:
        switches = {key: [option for option in params[key]['enum'] if option != value] for key, value in args.items() if isinstance(value, str) and isinstance((params.get(key) or {}).get('enum'), list)}
        modes = ', '.join(f'{key}={_to_json(options[0])}' for key, options in switches.items() if options)
        return f'; for cheaper listings only, list the category with {modes} while keeping the same {cap_key}' if any(switches.values()) else ''

    def cheapest_rows_note(self, results: list[Any]) -> str:
        priced: list[tuple[float, str]] = []
        if self.price_field and self.parsed_brief.get('pick') == 'cheapest':
            shown = {listing for result in results for listing, _ in self.listing_records(result)}
            priced = sorted((price, listing) for listing in shown if self.can_buy(listing) for price in [self.listing_price(listing)] if price is not None)
        listed = ', '.join(f'{listing} ({price:g})' for price, listing in priced[:6])
        return '\n[Corvin] rows here that can be bought, lowest price first: ' + listed if len(priced) >= 2 else ''

    def broader_search_hold(self) -> str:
        """Once per cheapest pick, with steps to spare, hold the first checkout and ask for wider searches."""
        left = self.steps_remaining()
        text = ''
        if not self.broader_search_done and self.parsed_brief.get('pick') == 'cheapest' and left >= 14 + 2 * len(self.wanted_products):
            self.broader_search_done = True
            lines = [f'{listing} at {self.listing_price(listing):g}' for listing in self.cart_listing_ids() if self.listing_price(listing) is not None]
            current = ' (' + '; '.join(lines) + ')' if lines else ''
            text = (f'Not run yet (no step used). {left} steps remain, and the request wants the lowest price. A search'
                    " may look at only part of each listing, so searching by a requirement's value turns up just some"
                    ' of the listings that fit, and rows arrive in no price order. Before you check out, search a few'
                    ' more times for every requested item using different names for the product (its simple one-word'
                    ' name, synonyms, words taken from titles of fitting listings already seen) and consider only'
                    " buyable rows that cost less than the cart's line" + current + '; open the attributes of the'
                    ' cheapest of those rows, replace a cart line with any that fits, and then check out.')
        return text

    # ------------------------------------------------------------------ opening listings for their attributes

    def open_for_attributes(self, wanted: list[str], keep: int) -> list[str]:
        """Open each known, not yet fully read, buyable listing with the attribute tool (one step each) while more
        than `keep` steps remain; returns one text per listing opened."""
        reader = self.attribute_tool()
        id_param = self.single_id_param(reader)
        queue = [listing for listing in wanted if listing in self.listing_rows and listing not in self.full_records and not self.is_sold_out(listing)] if reader and id_param else []
        notes: list[str] = []
        for listing in queue:
            if self.steps_remaining() <= keep or self.episode_over or self.shows_attributes.get(reader) is False:
                break
            answer = self._open_listing_page(reader, id_param, listing)
            if answer is None:
                break
            notes.append(f'{listing}: ' + self.render_result(answer.get('observation'))[:1500])
        if notes:
            self.event_log.append({'kind': 'fill_facts', 'n': len(notes)})
        return notes

    def _open_listing_page(self, reader: str, id_param: str, listing: str) -> dict[str, Any] | None:
        _, answer = self._send_conformed(reader, {id_param: listing})
        if answer is not None:
            page = answer.get('observation')
            self.take_observation(reader, page)
            self.remember_view_keys(reader, page)
            if isinstance(page, dict) and self.listing_id_of(page):
                self.shows_attributes[reader] = self.shows_attributes.get(reader, False) or bool(_attribute_block(page))
        return answer

    def follow_part_links(self, lines: list[str], keep: int) -> None:
        """For carts of two or more lines (or requests for two or more items): when a cart line's page names a part
        code for more details, search that code once, then open the listings that own it if steps allow. This
        only feeds the ledger; nothing is shown to the model."""
        search = self.value_search_plan()
        if search and (len(self.wanted_products) >= 2 or len(lines) >= 2):
            search_tool, fixed, code_param = search
            for line in lines:
                code = self._unfollowed_link_code(line)
                if not code:
                    continue
                if self.steps_remaining() <= keep or self.episode_over or not self._search_link_code(search_tool, fixed, code_param, line, code, keep):
                    break

    def _unfollowed_link_code(self, line: str) -> Any:
        """The 'more details' part code on a line's page, unless it was searched already or a known listing owns it."""
        marks = self.compatibility_index()['marks']
        code = (marks.get(line) or {}).get('more')
        fresh = code and code not in self.link_codes_tried and not any(mark.get('own') == code for mark in marks.values())
        return code if fresh else None

    def _search_link_code(self, search_tool: str, fixed: dict[str, Any], code_param: str, line: str, code: Any, keep: int) -> bool:
        self.link_codes_tried.add(code)
        _, reply = self._send_conformed(search_tool, {**fixed, code_param: code})
        if reply is not None:
            self.take_observation(search_tool, reply.get('observation'))
            self.event_log.append({'kind': 'show_link', 'line': line, 'code': code})
            owners = [other for other, mark in self.compatibility_index()['marks'].items() if mark.get('own') == code]
            self.open_for_attributes(owners, max(keep, self.closing_steps() + 6))
        return reply is not None

    # ------------------------------------------------------------------ category labels and attribute keys

    def _enum_vocab(self) -> set[str]:
        """Every string option (lower case) of any enum in the tool schemas, up to six levels deep."""
        vocab: set[str] = set()
        pending = [(schema, 0) for schema in self.tool_schemas.values()]
        while pending:
            node, level = pending.pop()
            if level <= 6 and isinstance(node, dict):
                vocab.update(option.strip().lower() for option in node.get('enum') or [] if isinstance(option, str))
                pending.extend((child, level + 1) for child in node.values() if isinstance(child, dict))
        return vocab

    def _short_text_columns(self) -> tuple[dict[str, set[str]], dict[str, int]]:
        texts_by_key: dict[str, set[str]] = {}
        hits: dict[str, int] = {}
        for record in self.listing_rows.values():
            for key, text in record.items():
                if self._short_label_text(text):
                    texts_by_key.setdefault(key, set()).add(text)
                    hits[key] = hits.get(key, 0) + 1
        return texts_by_key, hits

    def category_field(self) -> str | None:
        """The row key that carries the category label: short text on at least 80% of rows with at most 40 values,
        preferring values the schemas list as options, then fewer values, wider presence and longer values."""
        texts_by_key, hits = self._short_text_columns()
        rows = max(1, len(self.listing_rows))
        vocab = self._enum_vocab()

        def rank(key: str) -> tuple[float, int, int, float]:
            distinct = texts_by_key[key]
            listed = sum(1 for text in distinct if text.strip().lower() in vocab) / len(distinct)
            mean = sum(len(text) for text in distinct) / len(distinct)
            return (-listed, len(distinct), -hits[key], -mean)
        fitting = [key for key, distinct in texts_by_key.items() if hits[key] >= 0.8 * rows and len(distinct) <= 40]
        return min(fitting, key=rank) if fitting else None

    def _wanted_label_profiles(self) -> list[tuple[str, str, set[str]]]:
        profiles = []
        for item in self.wanted_products:
            shelf = item.category_label or ''
            stems = _word_stems(shelf) | _word_stems(item.product_name)
            profiles.append((_plain_words(shelf), _plain_words(item.product_name), {stem for stem in stems if len(stem) >= 4}))
        return profiles

    def _label_is_requested(self, label: str, profiles: list[tuple[str, str, set[str]]]) -> bool:
        plain = _plain_words(label)
        stems = {stem for stem in _word_stems(label) if len(stem) >= 4}
        return any(plain in (shelf, name) or stems & long for shelf, name, long in profiles) or bool(self.item_for_label(label))

    def category_groups(self) -> list[list[str]]:
        """Listing ids grouped by category label, the requested kinds only when any match, biggest first, one
        group per requested item."""
        label_key = self.category_field()
        by_label: dict[str, list[str]] = {}
        for listing in self.listing_order:
            shelf = str(self.listing_rows[listing].get(label_key) or '') if label_key else ''
            by_label.setdefault(shelf, []).append(listing)
        profiles = self._wanted_label_profiles()
        pool = [members for shelf, members in by_label.items() if self._label_is_requested(shelf, profiles)] or list(by_label.values())
        pool.sort(key=len, reverse=True)
        return pool[:max(1, len(self.wanted_products))]

    def attribute_key_table(self) -> list[tuple[str, str, int, list[str]]]:
        """Per category label (the requested kinds when any match, biggest first): up to 14 attribute keys seen on
        opened listings, as (label, plain key name, listings carrying it, up to 4 example values of 40 chars)."""
        buckets = self._attribute_groups_by_label(self.category_field())
        profiles = self._wanted_label_profiles()
        chosen = {shelf for shelf in buckets if self._label_is_requested(shelf, profiles)}
        if chosen:
            buckets = {shelf: bucket for shelf, bucket in buckets.items() if shelf in chosen}
        summary: list[tuple[str, str, int, list[str]]] = []
        for shelf, bucket in sorted(buckets.items(), key=lambda pair: -pair[1]['n']):
            for key, column in self._keys_worth_listing(bucket['keys'])[:14]:
                summary.append((shelf, key, column['n'], [text[:40] for text in column['values'][:4]]))
        return summary

    def _attribute_groups_by_label(self, label_key: str | None) -> dict[str, dict[str, Any]]:
        buckets: dict[str, dict[str, Any]] = {}
        for listing in self.listing_order:
            page = self.full_records.get(listing)
            if page:
                attrs = _attribute_block(page) or {}
                shelf = str(page.get(label_key) or 'listings') if label_key else 'listings'
                bucket = buckets.setdefault(shelf, {'n': 0, 'keys': {}})
                bucket['n'] += 1
                self._tally_attribute_keys(bucket['keys'], listing, attrs)
        return buckets

    @staticmethod
    def _tally_attribute_keys(columns: dict[str, dict[str, Any]], listing: str, attrs: dict[str, Any]) -> None:
        named: set[str] = set()
        for raw_key, raw_value in attrs.items():
            key = _plain_words(raw_key)
            if key and key not in named:
                named.add(key)
                column = columns.setdefault(key, {'n': 0, 'values': [], 'per': {}})
                column['n'] += 1
                text = str(raw_value).strip().lower()
                column['per'][listing] = text
                if text not in column['values']:
                    column['values'].append(text)

    def _keys_worth_listing(self, columns: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
        """Drop noisy keys, fold each key into an earlier one that names the same attribute, most carried first."""
        survivors: list[tuple[str, dict[str, Any]]] = []
        for key, column in sorted(columns.items(), key=lambda pair: (-pair[1]['n'], len(pair[0]))):
            texts = list(column['per'].values())
            if texts and not self._column_is_noise(texts):
                twin = next((spec for other, spec in survivors if self._columns_alike(other, spec, key, column)), None)
                if twin is None:
                    survivors.append((key, column))
                else:
                    self._fold_twin_column(twin, column)
        survivors.sort(key=lambda pair: -pair[1]['n'])
        return survivors

    @staticmethod
    def _column_is_noise(texts: list[str]) -> bool:
        """Long free text, mostly slash lists, or four or more distinct, longish values sharing no words."""
        size = len(texts)
        mean = sum(len(text) for text in texts) / size

        def echoed(text: str) -> bool:
            rest = ' '.join(other for other in texts if other is not text)
            return bool(set(re.findall('[a-z]{3,}', text)) & set(re.findall('[a-z]{3,}', rest)))
        slashed = 2 * sum(1 for text in texts if ' / ' in text) >= size
        scattered = size >= 4 and len(set(texts)) == size and mean > 12 and 2 * sum(1 for text in texts if echoed(text)) < size
        return mean > 40 or slashed or scattered

    @staticmethod
    def _fold_twin_column(twin: dict[str, Any], column: dict[str, Any]) -> None:
        merged_per, merged_values = twin['per'], twin['values']
        for listing, text in column['per'].items():
            merged_per.setdefault(listing, text)
            if text not in merged_values:
                merged_values.append(text)
        twin['n'] = len(merged_per)

    @staticmethod
    def _texts_agree(first: str, second: str) -> bool:
        words_a, words_b = re.findall('[a-z0-9]+', first.lower()), re.findall('[a-z0-9]+', second.lower())
        fewer, more = (words_a, words_b) if len(words_a) <= len(words_b) else (words_b, words_a)
        return first == second or (bool(fewer) and set(fewer) <= set(more) and len(more) <= 2 * len(fewer) + 2)

    @staticmethod
    def _columns_alike(kept_key: str, kept: dict[str, Any], new_key: str, column: dict[str, Any]) -> bool:
        """Whether two keys sharing a word name one attribute: their values on the same listings agree (all of
        them, at least two, when neither name contains the other; a majority otherwise), or, with no listing in
        common, they share a non-empty value."""
        left, right = set(kept_key.split()), set(new_key.split())
        verdict = False
        if left & right:
            ours, theirs = kept['per'], column['per']
            shared = [listing for listing in ours if listing in theirs and ours[listing].strip() and theirs[listing].strip()]
            agreeing = sum(1 for listing in shared if EpisodeRecords._texts_agree(ours[listing], theirs[listing]))
            if not (left <= right or right <= left):
                verdict = len(shared) >= 2 and agreeing == len(shared)
            elif shared:
                verdict = agreeing * 2 > len(shared)
            else:
                verdict = bool((set(kept['values']) & set(column['values'])) - {''})
        return verdict

    def other_property_note(self, label: str, key_name: str, examples: list[str]) -> str:
        """Tag a key that shares a word with a requirement's property but is not that property: its name differs
        and its values do not show the required value's words."""
        item_no = self.item_for_label(label)
        tag = ''
        if item_no:
            shown = set(re.findall('[a-z0-9]{3,}', ' '.join(examples).lower()))
            for need in self.wanted_products[item_no - 1].stated_needs:
                prop, _, wanted = need.partition(':')
                if self._clashing_property(prop, wanted, key_name, shown):
                    tag = f' [not the stated {prop.strip()}: a different property]'
                    break
        return tag

    @staticmethod
    def _clashing_property(prop: str, wanted: str, key_name: str, shown: set[str]) -> bool:
        """A plain required value (no negation, bound or alternative) whose property shares a word stem with the
        key, under another name, and whose words the key's values do not all show."""
        hedged = re.search('\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', wanted, re.I)
        needed = set(re.findall('[a-z0-9]{3,}', wanted.lower()))
        return not hedged and bool(prop.strip()) and _plain_words(prop) != key_name and bool(_word_stems(prop) & _word_stems(key_name)) and bool(needed) and not needed <= shown

    # ------------------------------------------------------------------ cheaper rows for a cheapest pick

    def open_cheaper_rows(self) -> list[str]:
        """Once per cart state of a cheapest pick: list each line's category below its price, then open up to two
        cheaper buyable rows of the same label per line (capped in all)."""
        opened: list[str] = []
        if self.parsed_brief.get('pick') == 'cheapest':
            stamp = self.cart_signature() + f'#{self.state_version}'
            label_key = self.category_field()
            if label_key and stamp not in self.cheaper_check_marks:
                self.cheaper_check_marks.add(stamp)
                opened = self._read_cheaper_than_cart(label_key)
        return opened

    def _read_cheaper_than_cart(self, label_key: str) -> list[str]:
        lines = self.cart_listing_ids()
        self._list_below_cart_prices(lines)
        chosen: list[str] = []
        for line in lines:
            head = self._cheaper_same_kind(line, lines, label_key)[:2]
            chosen.extend([listing for listing in head if listing not in chosen])
        allowance = min(6, CHEAPER_READS_PER_LINE * max(1, len(lines)))
        return self.open_for_attributes(chosen[:allowance], self.closing_steps() + 4)

    def _label_text(self, listing: str, label_key: str) -> str:
        return str((self.listing_rows.get(listing) or {}).get(label_key) or '').strip().lower()

    def _cheaper_same_kind(self, line: str, lines: list[str], label_key: str) -> list[str]:
        """Buyable, not yet opened listings with the line's label and a lower known price, cheapest first."""
        ceiling = self.listing_price(line)
        kind = self._label_text(line, label_key)
        pool: list[str] = []
        if ceiling is not None and kind:

            def cheaper(listing: str) -> bool:
                other = self.listing_price(listing)
                return (other if other is not None else float('inf')) < ceiling
            pool = [listing for listing in self.listing_order if listing not in lines and listing not in self.full_records and self.can_buy(listing) and self._label_text(listing, label_key) == kind and cheaper(listing)]
            pool.sort(key=lambda listing: self.listing_price(listing) or 0.0)
        return pool

    def _capped_listing_tool(self) -> tuple[str, dict[str, Any], str, str] | None:
        """The first tool taking exactly one category argument and one maximum-price argument, with its mode
        arguments filled: as the latest word-free listing by category set them, else the single option that names a
        browse or listing mode. None when no such tool exists or a mode stays open."""
        plan = None
        for tool, schema in self.tool_schemas.items():
            params = (schema or {}).get('properties') or {}
            caps = [key for key, spec in params.items() if isinstance(spec, dict) and {'integer', 'number'} & set(_schema_types(spec)) and re.search(PRICE_CAP_PARAM_RE, _snake_case(key), re.I) and not re.search(NOT_PRICE_CAP_RE, key, re.I)]
            shelves = [key for key, spec in params.items() if isinstance(spec, dict) and 'string' in _schema_types(spec) and 'enum' not in spec and re.search(CATEGORY_PARAM_RE, _snake_case(key), re.I)]
            if len(caps) == 1 and len(shelves) == 1:
                fixed = self._browse_mode_args(params, shelves[0])
                plan = (tool, fixed, shelves[0], caps[0]) if fixed is not None else None
                break
        return plan

    def _browse_mode_args(self, params: dict[str, Any], shelf: str) -> dict[str, Any] | None:
        fixed: dict[str, Any] | None = {}
        for key, spec in params.items():
            choices = spec.get('enum') if isinstance(spec, dict) else None
            if isinstance(choices, list) and len(choices) >= 2:
                value = self._mode_value(key, choices, params, shelf)
                if value is None:
                    fixed = None
                    break
                fixed[key] = value
        return fixed

    def _mode_value(self, key: str, choices: list[Any], params: dict[str, Any], shelf: str) -> Any:
        earlier = next((past[key] for past in reversed(self.recent_searches) if shelf in past and key in past and not self._search_had_words(past, params, shelf)), None)
        browse = [choice for choice in choices if isinstance(choice, str) and re.search(BROWSE_MODE_RE, choice, re.I)]
        return earlier if earlier is not None else (browse[0] if len(browse) == 1 else None)

    @staticmethod
    def _search_had_words(past: dict[str, Any], params: dict[str, Any], shelf: str) -> bool:
        def word_query(key: str, value: Any) -> bool:
            spec = params.get(key)
            open_text = key != shelf and not (isinstance(spec, dict) and 'enum' in spec)
            return open_text and isinstance(value, str) and bool(value.strip()) and bool(re.search(QUERY_PARAM_RE, _snake_case(key), re.I))
        return any(word_query(key, value) for key, value in past.items())

    def _list_below_cart_prices(self, lines: list[str]) -> None:
        """List each cart line's category with the maximum price set just under the line's price (once per line
        and price), reading a second page when the first shows over half of the matches unsorted."""
        plan = self._capped_listing_tool()
        label_key = self.category_field()
        if plan and label_key:
            tool, fixed, shelf, cap = plan
            for line in lines:
                ceiling = self.listing_price(line)
                shelf_label = str((self.listing_rows.get(line) or {}).get(label_key) or '')
                if ceiling is None or ceiling < 2 or not shelf_label or (line, ceiling) in self.capped_searches:
                    continue
                if self.steps_remaining() <= self.closing_steps() + 5 or self.episode_over or not self._list_one_shelf_below(tool, fixed, shelf, cap, line, shelf_label, ceiling):
                    break

    def _list_one_shelf_below(self, tool: str, fixed: dict[str, Any], shelf: str, cap: str, line: str, shelf_label: str, ceiling: float) -> bool:
        self.capped_searches.add((line, ceiling))
        sent, reply = self._send_conformed(tool, {**fixed, shelf: shelf_label, cap: self._price_below(ceiling)})
        if reply is not None:
            found = reply.get('observation')
            self.take_observation(tool, found)
            self.event_log.append({'kind': 'list_below', 'line': line, 'price': ceiling})
            self._second_capped_page(tool, sent, found)
        return reply is not None

    def _second_capped_page(self, tool: str, sent: dict[str, Any], found: Any) -> None:
        """One more page of a capped listing when the first showed more than half (but not all) of the matches,
        steps allow, and the rows are not already sorted by price."""
        total, shown, param, token = self._paging_state(tool, found) or (0, 0, None, None)
        worth = token is not None and shown < total <= 2 * shown
        if worth and self.steps_remaining() > self.closing_steps() + 5 and not self._prices_climb(found):
            _, extra = self._send_conformed(tool, {**sent, param: token})
            if extra is not None:
                self.take_observation(tool, extra.get('observation'))
class EpisodeSearch:
    """Talking to the shopper and the extra searches around that talk: which property of which item to ask about
    next and how the question is worded, whether the message tool takes the wording whole, the interview itself and
    the single late question before checkout, the opens that give the interviewer listings to look at first, and the
    searches that try the values an answer disclosed."""

    _WORD3_RE = '[a-z0-9]{3,}'
    _HEDGED_VALUE_RE = '\\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b'

    def _role_schema(self, role: str) -> tuple[str | None, dict[str, Any], dict[str, Any]]:
        """The tool that holds `role`, its argument schema ({} when unknown) and that schema's properties."""
        tool = self.tool_roles.get(role)
        schema = self.tool_schemas.get(tool or '') or {}
        return (tool, schema, schema.get('properties') or {})

    @staticmethod
    def _free_text_fields(fields: dict[str, Any]) -> list[str]:
        """Schema properties that take free text (a string type without enumerated choices), in schema order."""
        return [name for name, spec in fields.items() if isinstance(spec, dict) and 'string' in _schema_types(spec) and ('enum' not in spec)]

    @staticmethod
    def _shaped_or_none(call: dict[str, Any], schema: Any) -> dict[str, Any] | None:
        problems: list[str] = []
        shaped = _conform_to_schema(call, schema, 'args', problems)
        return shaped if not problems else None

    # ------------------------------------------------------------------ choosing and wording a question

    def already_answered(self, item_number: int, label_text: str) -> bool:
        """True once an answer about this item has settled the property `label_text` names. Names are compared by
        their folded words (plural -s dropped, -our spelled -or, so colour/color and singular/plural match): the label
        may lie inside the answered property's name ("colour" is covered by "shade colour"), or an answered name of at
        least two words may lie inside the label. A one-word answered name ("type") never covers a longer label
        ("bulb type")."""
        asked = _folded_spelling(label_text)
        answered = self.wanted_products[item_number - 1].answered_needs
        return any((_names_same_property(asked, _folded_spelling(need.split(':', 1)[0])) for need in answered))

    def _unasked_labels(self, labels: list[str], item_number: int, asked_labels: dict[int, set[str]]) -> list[str]:
        return [label for label in labels if _plain_words(label) not in asked_labels[item_number] and (not self.already_answered(item_number, label))]

    def next_question_labels(self, lanes: dict[int, list[str]], hits: dict[int, int], sent_counts: dict[int, int], asked_labels: dict[int, set[str]]) -> tuple[int, list[str]]:
        """The item to ask about next and the labels for its message. Items no answer has given a need for come
        first, then (unless messages are scarce and lists are off) the item with the fewest messages so far, then the
        order the request hinted at. Returns (item number, at most one message's worth of labels neither asked nor
        answered yet), or (0, []) when every lane is exhausted."""
        rank: dict[int, int] = {}
        for position, item_number in enumerate(self.hinted_item_order):
            rank[item_number] = position
        flat = self.message_limit <= FEW_MESSAGES and (not self.lists_allowed)
        width = PROPERTIES_PER_MESSAGE if self.lists_allowed else 1

        def priority(item_number: int) -> tuple[bool, int, int]:
            asked = 0 if flat else sent_counts.get(item_number, 0)
            return (hits.get(item_number, 0) > 0, asked, rank.get(item_number, len(rank) + item_number))
        candidates = ((n, self._unasked_labels(lanes[n], n, asked_labels)) for n in sorted(lanes, key=priority))
        return next(((n, left[:width]) for n, left in candidates if left), (0, []))

    def _compose_question(self, item_number: int, labels: list[str], with_examples: bool=False) -> str:
        """The shopper-facing wording: a list question for two or more labels, else the single-property question
        (with two example values when `with_examples`)."""
        if len(labels) < 2:
            only = labels[0]
            samples = self.example_values(item_number, only) if with_examples else None
            return self.single_question(item_number, only, samples)
        listed = ', '.join(labels[:-1])
        return f'What {listed} or {labels[-1]} would you like for the {self.item_wording(item_number)}?'

    def fitting_question(self, item_number: int, labels: list[str], with_examples: bool) -> tuple[list[str], str]:
        """The longest leading run of `labels` for which the message tool accepts the built question unchanged,
        together with that question. Shaping cuts a text at the schema's length cap, so labels drop off the end until the whole
        question survives; ([], '') when even a single label does not."""
        attempts = ((labels[:size], self._compose_question(item_number, labels[:size], with_examples)) for size in range(len(labels), 0, -1))
        return next(((head, wording) for head, wording in attempts if self._message_carries(wording)), ([], ''))

    def _message_carries(self, wording: str) -> bool:
        call = self.message_call_args(wording)
        return bool(call) and wording in call.values()

    @staticmethod
    def _fixed_choices(needed: list[str], fields: dict[str, Any], slot: str) -> dict[str, Any] | None:
        """The first enumerated option of every required field besides the text slot, in required order; None as
        soon as one such field offers no options."""
        chosen: dict[str, Any] = {}
        complete = True
        for name in needed:
            if complete and name != slot and (name not in chosen):
                options = (fields.get(name) or {}).get('enum')
                complete = isinstance(options, list) and bool(options)
                if complete:
                    chosen[name] = options[0]
        return chosen if complete else None

    def message_call_args(self, wording: str) -> dict[str, Any] | None:
        """Arguments for the message tool carrying `wording`: the text goes into the first required free-text field
        (else the first free-text field), every other required field takes its first enumerated option. None when a
        required field has no options to choose from or the shaped arguments break the schema."""
        _, schema, fields = self._role_schema('ask')
        needed = schema.get('required') or []
        free = self._free_text_fields(fields)
        slot = next((name for name in needed if name in free), None) or (free[0] if free else None)
        fixed = self._fixed_choices(needed, fields, slot) if slot else None
        return None if fixed is None else self._shaped_or_none({slot: wording, **fixed}, schema)

    # ------------------------------------------------------------------ the interview and the late question

    def interview_due(self) -> bool:
        """Whether to run the interview now: a message tool exists, spare messages remain, there is room in steps and
        time, and either enough listings are open (more than three per item) or a quarter of the steps (at least 8)
        has gone by."""
        closed = self.interview_done or not self.tool_roles.get('ask') or self.spare_messages() <= 0 or (self.steps_remaining() <= 8) or (self.elapsed_seconds() > INTERVIEW_DEADLINE)
        return not closed and (len(self.full_records) > 3 * len(self.wanted_products) or self.steps_used >= max(8, self.step_limit // 4))

    def _fresh_question_lanes(self, count: int) -> None:
        lanes = self.plan_question_lanes(count)
        self.question_lanes = lanes
        self.lane_hits = dict.fromkeys(lanes, 0)
        self.lane_messages = dict.fromkeys(lanes, 0)
        self.lane_asked_labels = {item_number: set() for item_number in lanes}

    def _requirements_changed(self) -> None:
        self.state_version = self.state_version + 1

    def run_interview(self) -> str:
        """Run the interview once: returns its account for the loop model, or '' when no message is spare."""
        self.interview_done = True
        return self._interview() if self.spare_messages() > 0 else ''

    def _interview(self) -> str:
        """Open a spread of listings, plan the question lanes, ask while messages, steps and time allow, then search
        the values the answers disclosed."""
        self._widen_by_compare()
        left = self.message_limit - self.messages_sent
        self._fresh_question_lanes(max(left, PROPERTIES_PER_MESSAGE) if self.lists_allowed else left)
        self.pending_followups = []
        exchanges: list[tuple[int, str, str]] = []
        added: list[str] = []
        asking = True
        while asking and self._interview_may_continue():
            asking = self._ask_one_more(added, exchanges)
        disclosed = [item_number for item_number, count in self.lane_hits.items() if count > 0]
        probe = self._search_told_values(disclosed) if disclosed else ''
        if added:
            self._requirements_changed()
        return self._interview_summary(exchanges, added, probe)

    def _interview_may_continue(self) -> bool:
        return self.spare_messages() > 0 and (not self.episode_over) and (self.steps_remaining() > 6) and (self.elapsed_seconds() < QUESTION_DEADLINE)

    def _ask_one_more(self, added: list[str], exchanges: list[tuple[int, str, str]]) -> bool:
        fresh_batch = self.send_next_question(added)
        exchanges.extend(fresh_batch)
        return bool(fresh_batch)

    def _interview_summary(self, exchanges: list[tuple[int, str, str]], added: list[str], probe: str) -> str:
        transcript = '\n'.join((f'Q: {asked}\nA: {reply}' for _, asked, reply in exchanges))
        text = "Corvin's interviewer asked the shopper these questions (one step each):\n" + transcript
        text += '\nFirm requirements these answers added: ' + ('; '.join(added) if added else 'none')
        text += '. Hold every pick to them (run new searches with the new values where that helps).'
        text += self._pointer_hint(added, ', and the order only counts once every one of them has been opened.')
        if probe:
            text += '\n' + probe
        backlog = self.queued_notes
        if backlog:
            text += '\n[notes received meanwhile] ' + ' | '.join(backlog)
            self.queued_notes = []
        return text

    def _pointer_hint(self, added: list[str], ending: str) -> str:
        linked = [need for _, need in self.pointer_musts if any((need in line for line in added))]
        hint = ' Their value comes from another listing: ' + ' | '.join(linked) + '. Locate that listing, read its full page, and open each listing that page points to as well; the final page holds the value' + ending
        return hint if linked else ''

    def _order_keeps_messages(self) -> bool:
        return any((step['when'] == 'order' and step['messages'] > 0 for step in self.process_steps))

    def _late_question_blocked(self) -> bool:
        return self.late_question_used or not self.tool_roles.get('ask') or self.messages_sent >= self.message_limit or self._order_keeps_messages() or (self.steps_remaining() < LATE_QUESTION_STEPS) or (self.elapsed_seconds() > LATE_QUESTION_DEADLINE) or (not self.review_models())

    def ask_before_checkout(self) -> str:
        """Spend one message the order does not need on the next planned question, right before checkout.

        During the interview every message a process step of the request may need stays reserved, both for a step
        that comes with the order and for one that comes with closing the episode without buying; once a purchase by
        Corvin is imminent, the messages held back for that closing are spare. In race 13 (two messages allowed) the
        base asked a single question in 28 episodes out of 180 and 12 of those left a hidden need unasked; not one
        episode with such a need passed. Gives the loop model a summary when the answer brings a new requirement, otherwise ''."""
        return '' if self._late_question_blocked() else self._late_question()

    def _late_question(self) -> str:
        self.late_question_used = True
        if not self.question_lanes:
            self._fresh_question_lanes(PROPERTIES_PER_MESSAGE if self.lists_allowed else 1)
        added: list[str] = []
        budget_before = self.budget_limit
        exchange = self.send_next_question(added, LATE_QUESTION_STEPS - 2)
        note = ''
        if exchange:
            note = self._after_late_answer(exchange[0], added, budget_before)
        else:
            self.event_log.append({'kind': 'late_ask', 'skipped': 'nothing left to ask, or the message did not go out'})
        return note

    def _after_late_answer(self, first: tuple[int, str, str], added: list[str], budget_before: Any) -> str:
        if self.budget_limit != budget_before:
            added.append(f'budget now {self.budget_limit}')
        item_number, asked, reply = first
        self.event_log.append({'kind': 'late_ask', 'q': asked, 'a': reply[:200], 'told': added})
        note = ''
        if added:
            self._requirements_changed()
            note = f'Just before checkout Corvin put one more question to the shopper (one step):\nQ: {asked}\nA: {reply}\n'
            note += 'Firm requirements this answer added: ' + '; '.join(added)
            note += '. Checkout has not happened yet: test every cart line against them (run new searches with the new values where that helps), repair the cart, then check out.'
            note += self._pointer_hint(added, '.')
            probe = self._search_told_values([item_number])
            if probe:
                note += '\n' + probe
        return note

    # ------------------------------------------------------------------ widening opens before the interview

    def _widen_by_compare(self) -> None:
        """Before the interview, put a spread of listings in front of the interviewer. A side-by-side tool whose only
        required argument is an id list (and that is not known to hide attributes) opens a few unopened listings of
        every category group in one call; without one, listings are opened singly."""
        tool, schema, fields = self._role_schema('compare')
        id_list = next((name for name, spec in fields.items() if isinstance(spec, dict) and 'array' in _schema_types(spec)), None)
        usable = bool(tool) and bool(id_list) and all((name == id_list for name in schema.get('required') or []))
        if not usable or self.compare_lacks_attributes():
            self._open_singly_for_attributes()
        elif self.steps_remaining() > 12:
            most = fields[id_list].get('maxItems')
            picks = self._grouped_compare_sample(most if isinstance(most, int) else 4)
            call = self._shaped_or_none({id_list: picks}, schema) if len(picks) >= 2 else None
            if call is not None:
                self._compare_and_follow_up(tool, schema, id_list, picks, call)

    def _grouped_compare_sample(self, limit: int) -> list[str]:
        groups = self.category_groups()
        per_group = max(1, limit // max(1, len(groups)))
        chosen: list[str] = []
        for group in groups:
            chosen.extend([ident for ident in group if ident not in self.full_records][:per_group])
        return chosen[:limit]

    def _compare_and_follow_up(self, tool: str, schema: dict[str, Any], id_list: str, picks: list[str], call: dict[str, Any]) -> None:
        known_before = len(self.full_records)
        outcome = self.store_call(tool, call)
        if outcome is not None:
            self._learn_from_compare(tool, schema, id_list, picks, outcome.get('observation'), known_before)

    def _learn_from_compare(self, tool: str, schema: dict[str, Any], id_list: str, picks: list[str], view: Any, known_before: int) -> None:
        """Absorb a side-by-side view and learn from it whether this view shows attributes. When it does not, fall
        back to single opens; when it opened nothing new, try once more with only the first two ids."""
        self.take_observation(tool, view)
        self.remember_view_keys(tool, view)
        listed = self.listing_records(view)
        if listed and (not self.shows_attributes.get(tool, False)):
            self.shows_attributes[tool] = any((_attribute_block(record) for _, record in listed))
        self.event_log.append({'kind': 'widen', 'ids': picks, 'opened': len(self.full_records) - known_before})
        if listed and self.shows_attributes.get(tool) is False:
            self._open_singly_for_attributes()
        elif len(self.full_records) == known_before and len(picks) > 2 and (self.steps_remaining() > 12):
            self._compare_first_pair(tool, schema, id_list, picks)

    def _compare_first_pair(self, tool: str, schema: dict[str, Any], id_list: str, picks: list[str]) -> None:
        retry = self._shaped_or_none({id_list: picks[:2]}, schema)
        outcome = self.store_call(tool, retry) if retry is not None else None
        if outcome is not None:
            self.take_observation(tool, outcome.get('observation'))

    def _open_singly_for_attributes(self) -> None:
        viewer = self.attribute_tool()
        if viewer and self.single_id_param(viewer):
            self._open_in_waves()

    def _requirement_value_words(self) -> set[str]:
        return {word for _, need in self.live_requirements() for word in re.findall(self._WORD3_RE, need.split(':', 1)[-1].lower())}

    def _row_words(self, ident: str) -> set[str]:
        return set(re.findall(self._WORD3_RE, _to_json(self.listing_rows.get(ident) or {}).lower()))

    def _price_or_last(self, ident: str) -> float:
        price = self.listing_price(ident)
        return float('inf') if price is None else price

    def _ranked_unopened(self, group: list[str], wanted: set[str]) -> list[str]:
        """The group's unopened, buyable listings, those whose rows share most words with the requirement values
        first (cheapest first among equals when the request wants the cheapest)."""
        pool = [ident for ident in group if ident not in self.full_records and self.can_buy(ident)]
        cheapest_wanted = self.parsed_brief.get('pick') == 'cheapest'
        if cheapest_wanted:
            pool.sort(key=self._price_or_last)
        pool.sort(key=lambda ident: -len(wanted & self._row_words(ident)))
        return pool

    @staticmethod
    def _interleave(queues: list[list[str]], spans: list[tuple[int, int]], rounds: int) -> list[str]:
        """Round-robin over the queues: for k = 0 .. rounds-1, entry k of each queue in turn, kept only when the
        queue's span (low, high) holds k."""
        merged: list[str] = []
        for k in range(rounds):
            for queue, (low, high) in zip(queues, spans):
                if low <= k < high:
                    merged.append(queue[k])
        return merged

    def _open_in_waves(self) -> None:
        """Open listings one at a time with the single-listing tool that shows attributes. Each category group
        offers its best unopened listings up to a quota (smaller with three or more items); a first wave takes up to
        four per group, interleaved across groups, and a second wave the rest, while the total stays small."""
        wanted = self._requirement_value_words()
        quota = OPENS_PER_GROUP if len(self.wanted_products) <= 2 else 4
        queues: list[list[str]] = []
        fronts: list[int] = []
        for group in self.category_groups():
            already = sum((1 for ident in group if ident in self.full_records))
            queues.append(self._ranked_unopened(group, wanted)[:max(0, quota - already)])
            fronts.append(max(0, 4 - already))
        first_wave = self._interleave(queues, [(0, min(front, len(queue))) for queue, front in zip(queues, fronts)], 4)
        second_wave = self._interleave(queues, [(front, len(queue)) for queue, front in zip(queues, fronts)], quota)
        cut = max(5, 2 * len(queues) + 2)
        done = self.open_for_attributes(first_wave[:min(cut, 8)], 22)
        self.open_for_attributes((first_wave[cut:] + second_wave)[:max(0, 10 - len(done))], 30)

    # ------------------------------------------------------------------ searching the values an answer gave

    def value_search_plan(self) -> tuple[str, dict[str, Any], str] | None:
        """The search a value search imitates: word_search_plan() when it finds one; otherwise, when every earlier
        search browsed a category without words, the search tool's single query field, with its single mode argument
        switched to reading it (the choice that mentions words or a query; failing that, the remaining choice of a
        mandatory mode that has exactly two values). None whenever the schema leaves room for doubt."""
        template = self.word_search_plan()
        if template is None:
            tool, schema, fields = self._role_schema('search')
            if tool and self.recent_searches:
                template = self._category_listing_plan(tool, schema, fields)
        return template

    def _category_listing_plan(self, tool: str, schema: dict[str, Any], fields: dict[str, Any]) -> tuple[str, dict[str, Any], str] | None:
        needed = schema.get('required') or []
        query_like = [name for name in self._free_text_fields(fields) if re.search(QUERY_PARAM_RE, _snake_case(name), re.I)]
        latest = self.recent_searches[-1]
        choices = {name: spec['enum'] for name, spec in fields.items() if isinstance(spec, dict) and isinstance(spec.get('enum'), list) and (len(spec['enum']) >= 2)}
        modes = [mode for mode in (self._query_mode(name, options, latest, needed) for name, options in choices.items()) if mode]
        box = query_like[0] if len(query_like) == 1 else None
        template = None
        if box is not None and len(modes) <= 1 and (not choices or modes):
            fixed = {name: latest[name] for name in needed if name in latest and name != box}
            fixed.update(modes)
            template = (tool, fixed, box)
        return template

    @staticmethod
    def _query_mode(name: str, options: list[Any], latest: dict[str, Any], needed: list[str]) -> tuple[str, Any] | None:
        reading = [option for option in options if isinstance(option, str) and re.search(QUERY_PARAM_RE, option, re.I)]
        unused = [option for option in options if option != latest.get(name)]
        mode = None
        if len(reading) == 1:
            mode = (name, reading[0])
        elif name in needed and len(options) == 2 and (len(unused) == 1):
            mode = (name, unused[0])
        return mode

    def word_search_plan(self) -> tuple[str, dict[str, Any], str] | None:
        """The newest earlier search that put words into a text field: a query-named one, or, when the schema names
        none, a field whose earlier values differ, run to two or more words and never equal a category label.
        Returns (tool, the arguments worth carrying over, that words field), or None when there is no such search."""
        tool, schema, fields = self._role_schema('search')
        free = self._free_text_fields(fields)
        query_like = [name for name in free if re.search(QUERY_PARAM_RE, _snake_case(name), re.I)]
        eligible = None if query_like or not self.recent_searches else self._phrase_fields(free)
        hit = next(self._worded_searches(query_like or free, eligible), None) if tool else None
        return hit and (tool, self._kept_search_args(hit[0], hit[1], schema, fields), hit[1])

    def _worded_searches(self, candidates: list[str], eligible: set[str] | None) -> Any:
        """(earlier search, its words field) for each earlier search that typed letters into a usable field, newest
        first."""
        for earlier in reversed(self.recent_searches):
            field = self._worded_field(earlier, candidates, eligible)
            if field is not None:
                yield (earlier, field)

    @staticmethod
    def _worded_field(earlier: dict[str, Any], candidates: list[str], eligible: set[str] | None) -> str | None:
        usable = (name for name in candidates if isinstance(earlier.get(name), str) and re.search('[a-z]', earlier[name], re.I) and (eligible is None or name in eligible))
        return next(usable, None)

    def _phrase_fields(self, free: list[str]) -> set[str]:
        """Without a query-named field: the text fields earlier searches used like a phrase box (two or more
        different values, some value of several words, no value equal to a category label)."""
        shelf_key = self.category_field()
        labels = {str(row.get(shelf_key) or '').strip().lower() for row in self.listing_rows.values()} - {''} if shelf_key else set()
        history = self.recent_searches

        def phrase_like(name: str) -> bool:
            values = [s.get(name) for s in history]
            return len({str(v) for v in values if v}) >= 2 and any((len(str(v).split()) >= 2 for v in values)) and (not any((str(v or '').strip().lower() in labels for v in values)))
        return {name for name in free if phrase_like(name)}

    def _shelf_names(self) -> set[str]:
        shelves = {_plain_words(product.category_label or '') for product in self.wanted_products}
        shelves.update((_plain_words(product.product_name) for product in self.wanted_products))
        shelf_key = self.category_field()
        if shelf_key:
            shelves.update((_plain_words(str(row.get(shelf_key) or '')) for row in self.listing_rows.values()))
        return shelves

    @staticmethod
    def _names_shelf(value: str, shelves: set[str]) -> bool:
        stems = {w for w in _word_stems(value) if len(w) >= 4}
        return _plain_words(value) in shelves or any((stems & {w for w in _word_stems(shelf) if len(w) >= 4} for shelf in shelves if shelf))

    def _kept_search_args(self, earlier: dict[str, Any], field: str, schema: dict[str, Any], fields: dict[str, Any]) -> dict[str, Any]:
        """The earlier search's required and enumerated arguments minus the words field; text that only names a shelf
        is dropped unless required, and each enumerated argument takes the value most worded searches used."""
        needed = schema.get('required') or []

        def enumerated(name: str) -> bool:
            return isinstance((fields.get(name) or {}).get('enum'), list)
        kept = {name: value for name, value in earlier.items() if name != field and (enumerated(name) or name in needed)}
        shelves = self._shelf_names()
        kept = {name: value for name, value in kept.items() if name in needed or not isinstance(value, str) or (not self._names_shelf(value, shelves))}
        worded = [s for s in self.recent_searches if isinstance(s.get(field), str) and re.search('[a-z]', s[field], re.I)]
        for name in [key for key in kept if enumerated(key)]:
            used = [_to_json(s[name]) for s in worded if name in s]
            if used:
                favourite = max(used, key=used.count)
                kept[name] = next((s[name] for s in worded if name in s and _to_json(s[name]) == favourite))
        return kept

    def _search_term(self, need: str, answered: list[str]) -> tuple[str, bool] | None:
        wanted_value = need.partition(':')[2].strip()
        floor = re.search(LOWER_BOUND_RE, wanted_value, re.I)
        if floor:
            return (floor.group(1).strip(), True) if need in answered else None
        hedged = not wanted_value or re.search(self._HEDGED_VALUE_RE, wanted_value, re.I)
        return None if hedged else (wanted_value, need in answered)

    def _value_words_for_item(self, item_number: int) -> list[tuple[str, bool]]:
        """Words to search with, one entry per requirement value of the item that a words search could match, each
        flagged with whether an answer supplied it. A plain value is used as it stands; an answer's lower limit on a measure with
        a unit gives its number and unit, since a hidden limit of that kind is often the value of a fitting listing.
        Values that point at other listings or carry limits, negations or alternatives are left out."""
        answered = self.wanted_products[item_number - 1].answered_needs
        pointer_texts = [text for _, text in self.pointer_musts]
        terms: list[tuple[str, bool]] = []
        for owner, need in self.live_requirements():
            if owner == item_number and need not in pointer_texts and (':' in need):
                term = self._search_term(need, answered)
                if term is not None:
                    terms.append(term)
        return terms

    @staticmethod
    def _value_query_tries(values: list[str], fresh: list[str], product_name: str) -> list[list[str]]:
        """Up to two word lists to search: answered one-word or measure values lead. With several values, all of
        them, then all but the last value the request itself stated (or but the first when none was stated); with a
        single leading value, that value with the product's name, then the value alone."""
        leading = [value for value in fresh if len(value.split()) == 1 or re.fullmatch(MEASURE_RE, value, re.I)]
        ordered = leading + [value for value in values if value not in leading]
        if len(ordered) > 1:
            stated = [index for index, value in enumerate(ordered) if value not in fresh]
            skip = stated[-1] if stated else 0
            return [ordered, ordered[:skip] + ordered[skip + 1:]]
        return [[ordered[0], product_name], ordered] if leading else [ordered]

    def _price_ceiling(self, found: list[str]) -> float | None:
        """The budget as a price cap for value-search rows; None without a numeric budget, or when every row costs
        over ten times the budget (prices in another unit, where the cap would hide everything)."""
        limit = self.budget_limit if isinstance(self.budget_limit, (int, float)) else None
        prices = [self.listing_price(ident) for ident in found]
        all_far_above = bool(prices) and all((price is not None and price > 10 * limit for price in prices)) if limit is not None else False
        return None if all_far_above else limit

    def _fitting_rows(self, tool: str, view: Any, product: Any, item_number: int) -> list[str]:
        """Absorb a value search's result and return its buyable rows of the item's own kind within the price cap,
        rows on the item's home shelf first."""
        self.take_observation(tool, view)
        shelf_key = self.category_field()
        found = [ident for ident, _ in self.listing_records(view)]
        ceiling = self._price_ceiling(found)

        def shelf_of(ident: str) -> str:
            return str((self.listing_rows.get(ident) or {}).get(shelf_key) or '')
        fitting = [ident for ident in found if self.can_buy(ident) and (not shelf_key or self.item_owns_label(product, shelf_of(ident))) and (ceiling is None or (self.listing_price(ident) or 0) <= ceiling)]
        home = self._home_shelf_label(item_number)
        if shelf_key and home:
            fitting.sort(key=lambda ident: _plain_words(shelf_of(ident)) != home)
        return fitting

    def _try_value_queries(self, tool: str, base: dict[str, Any], box: str, tries: list[list[str]], product: Any, item_number: int, allowance: int, notes: list[str]) -> int:
        """Run the item's tries in turn until one finds fitting rows (opening the first few unopened ones) or a call
        fails; returns the step allowance left."""
        for words in [w for w in tries[:2] if w]:
            call = self._shaped_or_none({**base, box: ' '.join(words)}, self.tool_schemas.get(tool))
            if call is None or self.episode_over:
                return allowance
            answer = self.store_call(tool, call)
            allowance -= 1
            if answer is None:
                return allowance
            fitting = self._fitting_rows(tool, answer.get('observation'), product, item_number)
            notes.append(f'{_to_json(call.get(box))}: ' + (', '.join(fitting[:6]) or 'nothing of that kind'))
            if fitting:
                newly_open = self.open_for_attributes([ident for ident in fitting if ident not in self.full_records][:min(3, allowance)], self.closing_steps() + 16)
                allowance -= len(newly_open)
                if newly_open:
                    notes.append('opened: ' + ' | '.join(newly_open)[:2400])
                return allowance
        return allowance

    def _run_value_searches(self, template: tuple[str, dict[str, Any], str], numbers: list[int], notes: list[str]) -> None:
        tool, base, box = template
        allowance = 8
        for item_number in numbers:
            terms = self._value_words_for_item(item_number)
            product = self.wanted_products[item_number - 1]
            fresh = [value for value, told in terms if told]
            if fresh and allowance > 2:
                tries = self._value_query_tries([value for value, _ in terms], fresh, product.product_name)
                allowance = self._try_value_queries(tool, base, box, tries, product, item_number, allowance, notes)

    def _search_told_values(self, numbers: list[int]) -> str:
        """For each listed item with a value an answer disclosed, search its requirement values together through the
        imitated search (value_search_plan), keep the fitting rows and open the first few unopened ones. Uses at
        most 8 steps and needs 22 left. Returns a note for the loop model, or ''."""
        template = self.value_search_plan()
        notes: list[str] = []
        if template and self.steps_remaining() >= 22:
            self._run_value_searches(template, numbers, notes)
        summary = ''
        if notes:
            self.event_log.append({'kind': 'value_search', 'seen': [note[:200] for note in notes]})
            summary = '[Corvin searched all requirement values of each such item at once and read the first rows of the right kind] ' + '\n'.join(notes)
        return summary

    def _owned_label_tally(self, product: Any, records: dict[str, Any], shelf_key: str) -> dict[str, int]:
        tally: dict[str, int] = {}
        for record in records.values():
            label = str(record.get(shelf_key) or '')
            if label and self.item_owns_label(product, label):
                key = _plain_words(label)
                tally[key] = tally.get(key, 0) + 1
        return tally

    def _label_choice(self, product: Any, records: dict[str, Any], shelf_key: str) -> str | None:
        tally = self._owned_label_tally(product, records, shelf_key)
        named = next((_plain_words(name) for name in (product.category_label, product.product_name) if _plain_words(name) in tally), None)
        if named is None and tally:
            named = max(tally, key=tally.get)
        return named

    def _home_shelf_label(self, item_number: int) -> str:
        """The category label (in _plain_words form) the item's own listings carry: the request's shelf or product
        name when a seen listing carries it, otherwise the label found on the largest number of the item's opened
        listings (on its search rows while nothing is opened yet); '' when nothing is known."""
        shelf_key = self.category_field()
        home = ''
        if shelf_key:
            product = self.wanted_products[item_number - 1]
            choices = (self._label_choice(product, records, shelf_key) for records in (self.full_records, self.listing_rows))
            home = next((label for label in choices if label is not None), '')
        return home
class EpisodeQuestions:
    """Corvin's question engine: which properties of which item to put to the shopper, in what order and words, and
    how every reply moves the plan along."""

    # ---------------------------------------------------------------- which wanted item a category label means

    @staticmethod
    def _qe_affinity(flat: str, roots: set[str], product: WantedProduct) -> float:
        if flat == _plain_words(product.category_label or ''):
            return 3.0
        if flat == _plain_words(product.product_name):
            return 2.0
        theirs = _root_words(product.category_label or '') | _root_words(product.product_name)
        return len(roots & theirs) / max(1, len(roots | theirs))

    def item_for_label(self, shelf: str) -> int:
        """The 1-based item a listing category label stands for, 0 when none. An adopted label keeps the item it was
        handed; otherwise a label equal to the item's shelf scores 3, equal to its product name 2, and any other the
        overlap of root words over their union. The earliest item with the best score above zero wins."""
        if shelf and shelf in self.adopted_labels:
            return self.adopted_labels[shelf]
        scores: list[float] = []
        if shelf:
            flat, roots = _plain_words(shelf), _root_words(shelf)
            scores = [self._qe_affinity(flat, roots, product) for product in self.wanted_products]
        best = max(scores, default=0.0)
        return scores.index(best) + 1 if best > 0.0 else 0

    def item_owns_label(self, wanted: WantedProduct, shelf: str) -> bool:
        listed = bool(shelf) and wanted in self.wanted_products
        return listed and self.item_for_label(shelf) == self.wanted_products.index(wanted) + 1

    def _qe_adopt_strays(self, shelf_key: str) -> None:
        present = {str(rec.get(shelf_key) or '') for rec in self.full_records.values()} - {''}
        unclaimed = [lab for lab in present if not self.item_for_label(lab)]
        bare = [n for n in range(1, len(self.wanted_products) + 1)
                if all(lab in self.adopted_labels or self.item_for_label(lab) != n for lab in present)]
        if unclaimed and len(bare) == 1:
            heir = bare[0]
            self.adopted_labels.update(dict.fromkeys(unclaimed, heir))
            self.event_log.append({'kind': 'adopted', 'item': heir, 'labels': unclaimed})

    def assign_stray_labels(self) -> None:
        """Several items, and the opened records carry category labels no item claims: when exactly one item has no
        label of its own (adopted ones do not count), it takes all of those strays."""
        shelf_key = self.category_field()
        if shelf_key and len(self.wanted_products) >= 2:
            self._qe_adopt_strays(shelf_key)

    def _qe_opened_for(self, slot: int) -> int:
        """How many fully opened records of the item's own categories show an attribute block."""
        shelf_key = self.category_field()
        owner = self.wanted_products[slot - 1]
        mine = (rec for rec in self.full_records.values() if self.item_owns_label(owner, str(rec.get(shelf_key) or '')))
        return sum(1 for rec in mine if _attribute_block(rec)) if shelf_key else 0

    def _qe_slot(self, raw: Any) -> int:
        """An item number from a model's answer, clamped into 1..number of items (1 when unreadable)."""
        return min(max(1, _as_int(raw) or 1), len(self.wanted_products))

    @staticmethod
    def _qe_bump(counter: dict[int, int], slot: int) -> None:
        counter[slot] = counter.get(slot, 0) + 1

    # ---------------------------------------------------------------- ordering attribute keys

    @staticmethod
    def _qe_bundled_last(names: list[str], view: Any = None) -> list[str]:
        """Stable split: names whose (viewed) text lumps several features together go to the back."""
        look = view or (lambda text: text)
        lumped = [bool(re.search(MIXED_FEATURE_KEY_RE, look(name))) for name in names]
        return [n for n, mixed in zip(names, lumped) if not mixed] + [n for n, mixed in zip(names, lumped) if mixed]

    def _qe_ranked_keys(self, slot: int, keyrows: list[tuple[str, str, int, list[str]]]) -> list[str]:
        """The item's attribute keys, most carried first; among equals a key with mostly numeric values goes later
        and a key with more distinct values earlier. A lone item with no category of its own uses every row.
        Properties the request already fixes, non-property keys (codes, links, warranty ...) are left out, and
        keys that lump features together close the list."""
        fixed = [_word_stems(need.split(':', 1)[0]) for need in self.wanted_products[slot - 1].stated_needs]
        numeric: dict[str, bool] = {}
        for _shelf, key, _count, seen in keyrows:
            numeric[key] = 2 * sum(1 for v in seen if re.match('^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$', v)) > len(seen)
        rows = [(key, many, len(seen)) for shelf, key, many, seen in keyrows if self.item_for_label(shelf) == slot]
        if not rows and len(self.wanted_products) == 1:
            rows = [(key, many, len(seen)) for _shelf, key, many, seen in keyrows]
        rows.sort(key=lambda row: (-row[1], numeric.get(row[0], False), -row[2]))
        kept = [key for key, _, _ in rows
                if not re.search(NON_PROPERTY_KEY_RE, key)
                and not any(_names_same_property(stems, _word_stems(key)) for stems in fixed)]
        return self._qe_bundled_last(kept)

    @staticmethod
    def _qe_fold(text: str) -> str:
        """Plain words with a trailing plural s and the British -our ending folded away."""
        return re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', _plain_words(text)))

    @staticmethod
    def _qe_share(name: str, share: dict[str, float]) -> float:
        """Share of the item's opened listings carrying the name's key, or the best share of a key naming the same
        property; 0.0 when nothing matches."""
        flat = _plain_words(name)
        if flat in share:
            return share[flat]
        stems = _word_stems(name)
        near = [part for key, part in share.items()
                if _names_same_property(stems, _word_stems(key)) or _names_same_property(_word_stems(key), stems)]
        return max(near, default=0.0)

    # ---------------------------------------------------------------- planning the lanes

    def _qe_key_listing(self, keyrows: list[tuple[str, str, int, list[str]]]) -> list[str]:
        lines: list[str] = []
        current = None
        for pos, (shelf, key, many, seen) in enumerate(keyrows, 1):
            if shelf != current:
                current = shelf
                lines.append(f'category {shelf!r}:')
            tag = ' [mixes several features: ask it last]' if re.search(MIXED_FEATURE_KEY_RE, key) else ''
            lines.append(f'  {pos}. {key} (on {many} opened): ' + ' | '.join(seen) + self.other_property_note(shelf, key, seen) + tag)
        return lines

    def _qe_plan_prompt(self, per_item: int, keyrows: list[tuple[str, str, int, list[str]]]) -> str:
        items = [f'item {n}: {it.product_name} - already fixed by the request: ' + ('; '.join(it.stated_needs) or 'nothing')
                 for n, it in enumerate(self.wanted_products, 1)]
        sections = [QUESTION_PLAN_PROMPT.replace('COUNT', str(max(1, per_item))), 'REQUEST:\n' + self.request_text, 'ITEMS:\n' + '\n'.join(items)]
        upfront = [str(x) for x in _as_list(self.parsed_brief.get('ask_first')) if str(x).strip()]
        if upfront:
            sections.append('ASK FIRST (the request wants these checked with the shopper):\n' + '\n'.join(upfront))
        sections.append('ATTRIBUTE KEYS ON OPENED LISTINGS:\n' + ('\n'.join(self._qe_key_listing(keyrows)) or '(no listing opened in full yet)'))
        return '\n\n'.join(sections)

    def _qe_take_pick(self, entry: dict[str, Any], picks: dict[int, list[str]], hinted: dict[int, set[str]]) -> None:
        slot = self._qe_slot(entry.get('item'))
        name = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(entry.get('label') or ''))[0].strip().rstrip('?.')
        if name and len(name.split()) <= 5:
            flat = _plain_words(name)
            if all(flat != _plain_words(prev) for prev in picks[slot]):
                picks[slot].append(name)
            if str(entry.get('hint') or '').strip():
                hinted[slot].add(flat)
                if slot not in self.hinted_item_order:
                    self.hinted_item_order.append(slot)

    def _qe_planned_picks(self, answer: dict[str, Any]) -> tuple[dict[int, list[str]], dict[int, set[str]]]:
        """The model's picks per item (cut at the first joining word or separator, at most five words, no repeats)
        and the plain names of the picks it tied to a hint; an item with a hint joins the hinted order."""
        picks: dict[int, list[str]] = {n: [] for n in range(1, len(self.wanted_products) + 1)}
        hinted: dict[int, set[str]] = {n: set() for n in picks}
        for entry in _as_list(answer.get('ask')):
            if isinstance(entry, dict):
                self._qe_take_pick(entry, picks, hinted)
        return picks, hinted

    def _qe_with_common_key(self, lane: list[str], key: str, share: dict[str, float], hinted: set[str]) -> list[str] | None:
        """The lane with a widely carried key moved into its first three, or None when it stays as it is.

        The key displaces the first of the three that no hint backs and whose share trails the key's by the gap
        (with several items the displaced label drops to fourth place, with one item the key just goes in front of
        it); with no such label the key only fills a free place among the first three. A change that would push a
        hinted label out of the first three is refused."""
        flat = _plain_words(key)
        head = lane[:3]
        if any(_plain_words(x) == flat for x in head):
            return None
        target = share.get(key, 0.0)
        soft = next((i for i, x in enumerate(head)
                     if _plain_words(x) not in hinted and self._qe_share(x, share) + COMMON_KEY_SHARE_GAP <= target), None)
        if soft is None and len(head) >= 3:
            return None
        trial = [x for x in lane if _plain_words(x) != flat]
        if soft is None:
            trial.append(key)
        elif len(self.wanted_products) > 1:
            bumped = trial[soft]
            trial[soft] = key
            trial.insert(min(3, len(trial)), bumped)
        else:
            trial.insert(min(len(trial), soft), key)
        front = {_plain_words(y) for y in trial[:3]}
        if any(_plain_words(x) in hinted and _plain_words(x) not in front for x in head):
            return None
        return trial

    def _qe_shape_lane(self, slot: int, chosen: list[str], hinted: set[str], keyrows: list[tuple[str, str, int, list[str]]]) -> list[str]:
        keys = self._qe_ranked_keys(slot, keyrows)
        reach = max(1, self._qe_opened_for(slot))
        share: dict[str, float] = {}
        for shelf, key, many, _seen in keyrows:
            if self.item_for_label(shelf) == slot:
                share[key] = max(share.get(key, 0.0), many / reach)

        def spell(text: str) -> str:
            folded = self._qe_fold(text)
            return next((key for key in share if self._qe_fold(key) == folded), text)

        backed = {_plain_words(spell(h)) for h in hinted}
        lane: list[str] = []
        for pick in chosen:
            name = spell(pick)
            if all(_plain_words(name) != _plain_words(prev) for prev in lane):
                lane.append(name)
        common = [k for k in keys if share.get(k, 0.0) >= COMMON_KEY_MIN_SHARE and not re.search(MIXED_FEATURE_KEY_RE, k)]
        for key in common[:COMMON_KEY_SLOTS]:
            lane = self._qe_with_common_key(lane, key, share, backed) or lane
        for extra in keys:
            if all(_plain_words(extra) != _plain_words(prev) for prev in lane):
                lane.append(extra)
        return self._qe_bundled_last(lane, _plain_words)

    def plan_question_lanes(self, per_item: int) -> dict[int, list[str]]:
        """One ordered list of property labels to ask per item: the strong model's picks (up to `per_item` each, the
        request's ASK FIRST entries and hinted properties leading) under the item's own key spellings, the most
        carried keys pulled forward, then every remaining ranked key; lumped-feature labels always come last."""
        self.assign_stray_labels()
        keyrows = self.attribute_key_table()
        answer = self.model_json(self._qe_plan_prompt(per_item, keyrows), [STRONG_MODEL, FAST_MODEL], 3500) or {}
        picks, hinted = self._qe_planned_picks(answer)
        planned = {slot: self._qe_shape_lane(slot, chosen, hinted[slot], keyrows) for slot, chosen in picks.items()}
        self.event_log.append({'kind': 'plan', 'ask': _truncate_strings(answer.get('ask'), 120), 'lanes': {n: lane[:8] for n, lane in planned.items()}})
        return planned

    # ---------------------------------------------------------------- naming the item and wording questions

    def _qe_joined_names(self, slot: int) -> list[str]:
        """The separate products an item name strings together with "and" or commas, provided every piece is a
        plural noun ("door handles and locks"); [] for one product that merely contains "and"
        ("salt and pepper grinder", "usb and lightning cables", "his and hers towels")."""
        title = self.wanted_products[slot - 1].product_name
        pieces = []
        if re.search('\\band\\b', title, re.I):
            pieces = [p.strip() for p in re.split('\\s*(?:,|\\band\\b)\\s*', title, flags=re.I) if p.strip()]
        plural = len(pieces) >= 2 and all(re.search('[a-z]{3}(?<![su])s$', p, re.I) for p in pieces)
        return pieces if plural else []

    def _qe_likeliest_piece(self, product: WantedProduct, pieces: list[str]) -> str:
        """The joined product whose last word turns up in the most of the item's listing rows (category field left
        out); ties and an empty shelf go to the earliest piece."""
        shelf_key = self.category_field()
        texts = [_root_words(' '.join(str(v) for k, v in row.items() if k != shelf_key and isinstance(v, str)))
                 for row in self.listing_rows.values()
                 if not shelf_key or self.item_owns_label(product, str(row.get(shelf_key) or ''))]
        tallies = [sum(1 for words in texts if _root_words(p.split()[-1]) & words) for p in pieces]
        return pieces[tallies.index(max(tallies))]

    def item_wording(self, slot: int) -> str:
        """How a question names the item. Plural products joined by "and" are named with "or" instead: in race 13
        (ten agents) "door handles and locks" got a one-thing-at-a-time reply to 81 of 155 questions, names joined
        by "or" to none of 50, and "dresser or cabinet" drew out a need 10 times in 28. Once the shopper asked for
        one thing at a time about this item, only one of its joined products is named (see _qe_likeliest_piece)."""
        product = self.wanted_products[slot - 1]
        pieces = self._qe_joined_names(slot)
        if not pieces:
            return product.product_name
        if slot in self.single_product_items:
            return self._qe_likeliest_piece(product, pieces)
        return re.sub('\\band\\b', 'or', product.product_name, flags=re.I)

    def single_question(self, slot: int, prop: str, samples: list[str] | None = None) -> str:
        ending = f', for example {samples[0]} or {samples[1]}?' if samples else '?'
        return f'What {prop} would you like for the {self.item_wording(slot)}' + ending

    def example_values(self, slot: int, prop: str) -> list[str]:
        """Up to two values (30 characters at most) seen for the label's key that the item's stated needs do not
        already contain."""
        wanted = _plain_words(prop)
        fixed = ' '.join(self.wanted_products[slot - 1].stated_needs).lower()
        seen = next((vals for _shelf, key, _count, vals in self.attribute_key_table() if key == wanted), [])
        return [v for v in seen if v and len(v) <= 30 and v not in fixed][:2]

    # ---------------------------------------------------------------- sending messages and reading replies

    def _qe_may_message(self, floor: int) -> bool:
        return not (self.episode_over or self.steps_remaining() <= floor or self.messages_sent >= self.message_limit)

    def _qe_note_message(self, wording: str, words: str) -> None:
        step = self.steps_used
        self.action_record.append(f'step {step}: said to the shopper {wording!r} -> answer {words[:300]!r} (cart then: {self.cart_signature()})')
        self.action_steps.append(step)
        self.event_log.append({'kind': 'ask', 'q': wording, 'a': words[:300]})

    def _qe_deliver(self, channel: str, slot: int, wording: str, floor: int, delivered: list[tuple[int, str, str]]) -> bool:
        """Send one question; True when the next one may follow."""
        call = self.message_call_args(wording) if self._qe_may_message(floor) else None
        reply = self.store_call(channel, call) if call else None
        if reply is None:
            return False
        self.messages_sent = self.messages_sent + 1
        seen = reply.get('observation')
        words = self.reply_words(seen)
        delivered.append((slot, wording, words))
        self._qe_note_message(wording, words)
        return not (isinstance(seen, dict) and seen.get('error'))

    def send_messages(self, questions: list[tuple[int, str, str]], floor: int = 6) -> list[tuple[int, str, str]]:
        """Send each (item, label, text) question through the messaging tool while steps above `floor` and messages
        remain; returns (item, text, reply words) for every one that went out. Stops at a refused call or an
        error reply."""
        channel = self.tool_roles['ask']
        delivered: list[tuple[int, str, str]] = []
        queue = list(questions)
        going = True
        while going and queue:
            slot, _, wording = queue.pop(0)
            going = self._qe_deliver(channel, slot, wording, floor, delivered)
        return delivered

    def _qe_read_replies(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]) -> dict[str, Any]:
        roster = '\n'.join(f'item {n}: {it.product_name}' for n, it in enumerate(self.wanted_products, 1))
        talk = '\n'.join(f'{i}. (item {n}) Q: {q}\nA: {a}' for i, (n, q, a) in enumerate(sent, 1))
        request = ANSWER_READ_PROMPT + '\n\nITEMS:\n' + roster + '\n\nQUESTIONS SENT, EACH WITH ITS REPLY:\n' + talk
        return self.model_json(request, list(models), 2500) or {}

    def _qe_take_need(self, row: dict[str, Any], news: list[str]) -> None:
        need = str(row.get('need') or '').strip()
        if need:
            slot = self._qe_slot(row.get('item'))
            product = self.wanted_products[slot - 1]
            if need not in product.answered_needs:
                product.answered_needs.append(need)
                news.append(f'item {slot}: {need}' if len(self.wanted_products) > 1 else need)
            if row.get('points') is True and (slot, need) not in self.pointer_musts:
                self.pointer_musts.append((slot, need))

    def _qe_take_drop(self, row: dict[str, Any], news: list[str]) -> None:
        prop = str(row.get('property') or '')
        if prop.strip():
            product = self.wanted_products[self._qe_slot(row.get('item')) - 1]
            gone = _plain_words(prop.split(':')[0])
            lost = [m for m in product.stated_needs + product.answered_needs
                    if m not in product.withdrawn and _plain_words(m.split(':')[0]) == gone]
            product.withdrawn.extend(lost)
            news.extend(f'gave up {m}' for m in lost)

    def _qe_take_free(self, row: dict[str, Any], news: list[str]) -> None:
        if row.get('property'):
            self.wanted_products[self._qe_slot(row.get('item')) - 1].indifferent_to.append(str(row['property']))

    def apply_answers(self, reading: dict[str, Any], news: list[str]) -> None:
        """Fold a read reply into the items: new needs (and pointer needs), withdrawn requirements, properties the
        shopper does not mind, and a new positive budget. Lines for the loop model go to `news`."""
        for field, take in (('needs', self._qe_take_need), ('drop', self._qe_take_drop), ('free', self._qe_take_free)):
            for row in _as_list(reading.get(field)):
                if isinstance(row, dict):
                    take(row, news)
        budget = reading.get('budget')
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0:
            self.budget_limit = budget

    # ---------------------------------------------------------------- one question round

    def _qe_next_candidate(self) -> tuple[int, list[str], bool]:
        """A waiting follow-up whose first label is still unanswered, else the next planned labels."""
        waiting = [entry for entry in self.pending_followups if not self.already_answered(entry[0], entry[1][0])]
        self.pending_followups = waiting
        if waiting:
            return waiting.pop(0)
        slot, names = self.next_question_labels(self.question_lanes, self.lane_hits, self.lane_messages, self.lane_asked_labels)
        return slot, names, False

    def _qe_pick_question(self) -> tuple[int, list[str], bool, str] | None:
        """Up to one try per list slot: the candidate cut to what the message tool takes whole (without example
        values when those do not fit). A label too long even alone counts as asked and the next one is tried."""
        for _ in range(PROPERTIES_PER_MESSAGE):
            slot, names, shown = self._qe_next_candidate()
            if not names:
                return None
            kept, wording = self.fitting_question(slot, names, shown)
            if shown and not kept:
                kept, wording = self.fitting_question(slot, names, False)
            if kept:
                return slot, kept, shown, wording
            self.lane_asked_labels.setdefault(slot, set()).add(_plain_words(names[0]))
            self.event_log.append({'kind': 'ask_too_long', 'item': slot, 'label': names[0]})
        return None

    def _qe_take_reply(self, slot: int, names: list[str], shown: bool, sent: list[tuple[int, str, str]], news: list[str]) -> None:
        self._qe_bump(self.lane_messages, slot)
        before = [len(product.answered_needs) for product in self.wanted_products]
        reading = self._qe_read_replies(sent, (FAST_MODEL, STRONG_MODEL))
        if len(names) >= 2:
            reading.pop('free', None)
        self.apply_answers(reading, news)
        fresh: list[str] = []
        for n, (product, size) in enumerate(zip(self.wanted_products, before), 1):
            gained = product.answered_needs[size:]
            if gained:
                self._qe_bump(self.lane_hits, n)
            fresh.extend(gained)
        cramped = any(_as_int(x) == 1 for x in _as_list(reading.get('crowded')))
        self._qe_settle(slot, names, shown, fresh, cramped)

    def send_next_question(self, news: list[str], floor: int = 6) -> list[tuple[int, str, str]]:
        """Ask the next question (a waiting follow-up before the planned labels), fold its reply into the items'
        needs (each new one also appended to `news`) and settle the labels it named. Returns what was sent, [] when
        nothing went out."""
        picked = self._qe_pick_question()
        sent = self.send_messages([(picked[0], picked[1][0], picked[3])], floor) if picked else []
        if sent:
            self._qe_take_reply(picked[0], picked[1], picked[2], sent, news)
        return sent

    # ---------------------------------------------------------------- settling the labels a reply covered

    def _qe_label_for_need(self, names: list[str], need: str, vocab: dict[str, set[str]]) -> int | None:
        """The label a new need answers: the one sharing most name words with the need's property (spelling
        folded), else the one whose seen values share most words with the need's value; None for neither."""
        prop, _sep, wanted = need.partition(':')
        scores = [len(_folded_spelling(x) & _folded_spelling(prop)) for x in names]
        if max(scores, default=0) <= 0:
            scores = [len(vocab.get(_plain_words(x), set()) & _root_words(wanted)) for x in names]
        top = max(scores, default=0)
        return scores.index(top) if top > 0 else None

    def _qe_earliest_answered(self, names: list[str], fresh: list[str]) -> int:
        """Position of the first label in a list that some new need answers, -1 when none maps to a label (such a
        reply is settled like one that disclosed nothing)."""
        vocab: dict[str, set[str]] = {}
        for _shelf, key, _count, seen in self.attribute_key_table():
            vocab.setdefault(_plain_words(key), set()).update(w for v in seen for w in _root_words(v))
        found = [i for i in (self._qe_label_for_need(names, need, vocab) for need in fresh) if i is not None]
        return min(found, default=-1)

    def _qe_narrower_followup(self, slot: int, names: list[str], shown: bool, asked: set[str]) -> tuple[int, list[str], bool] | None:
        """What a one-label question that drew 'one thing at a time' is followed by, as in the base agent: the same
        label once the item is named as a single one of its joined products, else the label's last word alone
        when it has two or more words and that word is unasked, else the label again with two example values."""
        if slot not in self.single_product_items and self._qe_joined_names(slot):
            self.single_product_items.add(slot)
            self.event_log.append({'kind': 'one_product', 'item': slot, 'said': self.item_wording(slot)})
            return (slot, names, shown)
        words = names[0].split()
        head = words[-1]
        if len(words) >= 2 and _plain_words(head) not in asked:
            return (slot, [head], False)
        if not shown and len(self.example_values(slot, names[0])) >= 2:
            return (slot, names, True)
        return None

    def _qe_settle_one(self, slot: int, names: list[str], shown: bool, asked: set[str], fresh: list[str], cramped: bool) -> None:
        """A single label always counts as asked; only a 'one thing at a time' reply that disclosed nothing earns
        it a narrower follow-up."""
        asked.add(_plain_words(names[0]))
        follow = self._qe_narrower_followup(slot, names, shown, asked) if cramped and not fresh else None
        if follow is not None:
            self.pending_followups.insert(0, follow)

    def _qe_settle_list(self, slot: int, names: list[str], shown: bool, asked: set[str], fresh: list[str], cramped: bool) -> None:
        """Settle a question that named several labels.

        The shopper answers about the first property in a message it tracks, and a message naming none of them
        gets the same reply as a property it does not mind. A disclosed need therefore settles its own label and
        all before it, and the ones after it open the item's next list. A reply with nothing in it settles only
        the first label (a need there would have come out) and puts the rest back in reverse, so labels the
        shopper does not care about stop blocking the others; on every second such reply in a row the labels not
        yet named go first instead, or the order would cycle back to the first list's. A plea for one question at
        a time switches lists off for the episode and the list's first label is asked by itself."""
        if cramped and not fresh:
            self.lists_allowed = False
            self.event_log.append({'kind': 'lists_off', 'item': slot})
            return self.pending_followups.insert(0, (slot, names[:1], False))
        hit = self._qe_earliest_answered(names, fresh) if fresh else -1
        done = names[:hit + 1] if hit >= 0 else []
        ahead = names[hit + 1:] if hit >= 0 else names[:0:-1]
        asked.update(_plain_words(x) for x in (done or names[:1]))
        self.lane_empty_streak[slot] = 0 if hit >= 0 else self.lane_empty_streak.get(slot, 0) + 1
        others = [x for x in self.question_lanes.get(slot) or [] if x not in names]
        streak = self.lane_empty_streak[slot]
        self.question_lanes[slot] = others + ahead if streak and streak % 2 == 0 else ahead + others
        self.event_log.append({'kind': 'list_settled', 'item': slot, 'done': done, 'next': ahead[:4]})

    def _qe_settle(self, slot: int, names: list[str], shown: bool, fresh: list[str], cramped: bool) -> None:
        """Record what the reply showed about the labels its question named (one label, or a list)."""
        asked = self.lane_asked_labels.setdefault(slot, set())
        settle = self._qe_settle_one if len(names) == 1 else self._qe_settle_list
        settle(slot, names, shown, asked, fresh, cramped)
class EpisodeChecks:
    """Checks Corvin runs around the model loop: what the cart holds, the request's own steps before an ending, store
    notes read into the ledger, evidence and fit hints for the model, the line and count checks on the cart, and the
    last careful review before checking out."""

    # ------------------------------------------------------------------ cart as seen or as tallied

    def _cart_view_trusted(self) -> bool:
        """A cart view is trusted only when it was taken no earlier than the last cart change."""
        return self.cart_snapshot is not None and self.cart_viewed_step >= self.cart_changed_step

    @staticmethod
    def _dicts_in_view(root: Any):
        """Every dict inside a cart view, depth first, with the later siblings of a branch visited first."""
        todo = [root]
        while todo:
            here = todo.pop()
            if isinstance(here, list):
                todo.extend(here)
            elif isinstance(here, dict):
                yield here
                todo.extend((sub for sub in here.values() if isinstance(sub, (dict, list))))

    def _held_count(self, listing: str) -> int:
        """Units of one listing in the cart. A trusted cart view decides when a node naming the listing carries exactly
        one count-like number; otherwise Corvin's own tally, which assumes 1 for a listing it does not know."""
        if self._cart_view_trusted():
            for node in self._dicts_in_view(self.cart_snapshot):
                if listing in [text.strip() for text in node.values() if isinstance(text, str)]:
                    numbers = [_as_int(raw) for key, raw in node.items() if not isinstance(raw, str) and _as_int(raw) is not None and re.search('\\b(?:qty|quantity|count|units?)\\b', _plain_words(key))]
                    if len(numbers) == 1:
                        return numbers[0]
        return self.cart_units.get(listing, 1)

    def cart_listing_ids(self) -> list[str]:
        if not self._cart_view_trusted():
            return [*self.cart_units]
        dump = _to_json(self.cart_snapshot)
        return [known for known in self.listing_rows if known in dump]

    def cart_summary(self) -> str:
        if self._cart_view_trusted():
            view = _truncate_strings(self.cart_snapshot, 300)
            return _to_json(view)[:3000]
        return "Corvin's tally from its own cart calls (listing: units): " + _to_json(self.cart_units)

    def record_text(self, listing: str) -> str:
        seen = self.full_records.get(listing) or self.listing_rows.get(listing)
        return _to_json(_truncate_strings(seen, 400))[:3000] if seen else '(record not seen yet)'

    def _rival_rows(self, skip: list[str], limit: int) -> str:
        """Seen listings outside the cart (none sold out), newest first, or cheapest opened first when the pick rule is
        the lowest price; at most `limit` (at least 1) of them."""
        queue = list(reversed(self.listing_order))
        if self.parsed_brief.get('pick') == 'cheapest':
            unknown_price = float('inf')
            queue.sort(key=lambda cand: (cand not in self.full_records, self.listing_price(cand) if self.listing_price(cand) is not None else unknown_price))
        rows = [_to_json(_truncate_strings(self.full_records.get(cand) or self.listing_rows.get(cand), 160))[:900] for cand in queue if cand not in skip and (not self.is_sold_out(cand))]
        return '\n'.join(rows[:limit])

    # ------------------------------------------------------------------ requirements in force

    def live_requirements(self) -> list[tuple[int, str]]:
        """(item number, requirement) for each stated or answered requirement not withdrawn, item by item."""
        return [(slot, need) for slot, item in enumerate(self.wanted_products, 1) for need in item.stated_needs + item.answered_needs if need not in item.withdrawn]

    def _states_item_needs(self, listing: str, slot: int) -> bool:
        """True when the listing's attributes spell out every requirement of item `slot` word for word. Limits and
        exclusions are not judged here, so any of them makes the answer False."""
        attrs = _attribute_block(self.full_records.get(listing) or {}) or {}
        hedges = {'not', 'no', 'least', 'most', 'under', 'over', 'unlike', 'different', 'min', 'max', 'minimum', 'maximum'}

        def word_bag(text: str) -> set[str]:
            return set(re.sub('[,/;|]', ' ', _plain_words(text)).split())

        def spelled_out(need: str) -> bool:
            prop_name, _sep, wanted_value = need.partition(':')
            wanted = word_bag(wanted_value)
            name = _folded_spelling(prop_name)
            return bool(wanted) and (not wanted & hedges) and any((_names_same_property(name, _folded_spelling(key)) and wanted <= word_bag(str(raw)) for key, raw in attrs.items()))
        return all((spelled_out(need) for owner, need in self.live_requirements() if owner == slot))

    # ------------------------------------------------------------------ reading free text in results

    def _spoken_phrases(self, node: Any, level: int):
        if level <= 4:
            if isinstance(node, str):
                if len(node.split()) >= 2:
                    yield node
            elif isinstance(node, (dict, list)):
                for child in node.values() if isinstance(node, dict) else node:
                    yield from self._spoken_phrases(child, level + 1)

    def reply_words(self, payload: Any) -> str:
        """The readable words of a reply: a plain string as it is; otherwise its strings of two words or more (up to
        four levels down) joined, or the JSON itself when it has none."""
        if isinstance(payload, str):
            return payload
        phrases = list(self._spoken_phrases(payload, 0))
        return ' '.join(phrases)[:800] if phrases else _to_json(payload)[:400]

    def _note_sentences(self, value: Any) -> list[str]:
        """Sentence-like strings in one top-level field of a result: five words or more at the top or in a list of
        strings, six or more one level down in a block that is not a listing."""
        picked: list[str] = []
        if isinstance(value, str):
            picked = [value] if len(value.split()) >= 5 else picked
        elif isinstance(value, list):
            if value and all((isinstance(entry, str) for entry in value)):
                picked = [entry for entry in value if len(entry.split()) >= 5]
        elif isinstance(value, dict) and (not self.listing_id_of(value, False, 1)):
            picked = [inner for inner in value.values() if isinstance(inner, str) and len(inner.split()) >= 6]
        return picked

    def free_text_lines(self, payload: Any) -> list[str]:
        sentences: list[str] = []
        if isinstance(payload, dict) and (not self.listing_id_of(payload)):
            for value in payload.values():
                sentences += self._note_sentences(value)
        return [sentence for sentence in sentences if sentence not in self.seen_note_lines]

    def _attribute_holders(self, node: Any, level: int, into: list[str]) -> None:
        """Collects, in reading order, the ids of listings in a result that carry an attribute block; the inside of a
        listing taken is not searched further."""
        if level <= 6:
            if isinstance(node, list):
                for child in node:
                    self._attribute_holders(child, level + 1, into)
            elif isinstance(node, dict):
                found_id = self.listing_id_of(node, False, level)
                if found_id and _attribute_block(node) and (found_id not in into):
                    into.append(found_id)
                else:
                    for child in node.values():
                        self._attribute_holders(child, level + 1, into)

    def _evidence_for(self, listing: str, in_force: list[tuple[int, str]], tagged: bool) -> str:
        attrs = _attribute_block(self.full_records.get(listing) or {}) or {}
        notes = []
        for slot, need in in_force:
            prop_name = need.split(':', 1)[0].strip()
            stems = _word_stems(prop_name)
            matched = [f'{key}={str(raw)[:50]}' for key, raw in attrs.items() if _names_same_property(stems, _word_stems(key))]
            if matched:
                notes.append(('(item %d) ' % slot if tagged else '') + prop_name + ' -> ' + ', '.join(matched[:3]))
        return listing + ': ' + '; '.join(notes) if notes else ''

    def evidence_text(self, payload: Any) -> str:
        holders: list[str] = []
        self._attribute_holders(payload, 0, holders)
        in_force = self.live_requirements()
        rows: list[str] = []
        if holders and in_force:
            tagged = len(self.wanted_products) > 1
            rows = [row for row in (self._evidence_for(listing, in_force[:8], tagged) for listing in holders[:12]) if row]
        if not rows:
            return ''
        return "[evidence: attribute keys that name a requirement's property; a key about some other property (another part or aspect) settles nothing, while a key under a different name can still carry the same property] " + ' | '.join(rows)

    # ------------------------------------------------------------------ store notes

    def _note_gone(self, reading: dict[str, Any], note_lines: list[str]) -> list[str]:
        """Marks the listings a note calls unavailable and remembers whether it reports a sell-out at all."""
        gone_ids = [str(entry) for entry in _as_list(reading.get('gone')) if isinstance(entry, (str, int))]
        self.unavailable_ids.update((entry for entry in gone_ids if entry in self.listing_rows))
        self.note_reports_sellout = bool(gone_ids) or reading.get('sold_out') is True
        self.note_named_listing = any((entry in self.listing_rows for entry in gone_ids))
        if self.note_reports_sellout and len(note_lines) == 1:
            self.sellout_note_lines.update(note_lines)
        return gone_ids

    def _note_budget(self, value: Any, edits: list[str]) -> bool:
        usable = isinstance(value, (int, float)) and (not isinstance(value, bool)) and (value > 0) and (value != self.budget_limit)
        if usable:
            self.budget_limit = value
            edits.append(f'the budget is now {value}')
        return usable

    @staticmethod
    def _note_requests(reading: dict[str, Any], field: str) -> list[dict[str, Any]]:
        return [entry for entry in _as_list(reading.get(field)) if isinstance(entry, dict) and str(entry.get('need') or '').strip()]

    def _note_target_item(self, entry: dict[str, Any]) -> WantedProduct:
        slot = min(max(1, _as_int(entry.get('item')) or 1), len(self.wanted_products))
        return self.wanted_products[slot - 1]

    def _note_withdraw(self, entry: dict[str, Any], edits: list[str]) -> None:
        item = self._note_target_item(entry)
        gone_need = str(entry['need']).strip()
        in_force = [need for need in item.stated_needs + item.answered_needs if need not in item.withdrawn]
        same = [need for need in in_force if need == gone_need]
        if not same:
            head = _plain_words(gone_need.split(':')[0])
            same = [need for need in in_force if _plain_words(need.split(':')[0]) == head][:1]
        item.withdrawn.extend(same or [gone_need])
        edits.append(f'no longer wanted: {gone_need}')

    def _note_add(self, entry: dict[str, Any], edits: list[str]) -> None:
        item = self._note_target_item(entry)
        fresh_need = str(entry['need']).strip()
        if fresh_need in item.withdrawn:
            item.withdrawn.remove(fresh_need)
        if fresh_need not in item.stated_needs + item.answered_needs:
            item.answered_needs.append(fresh_need)
        edits.append(f'now also wanted: {fresh_need}')

    def _apply_note_change(self, reading: dict[str, Any], gone_ids: list[str]) -> str:
        edits: list[str] = []
        budget_moved = self._note_budget(reading.get('budget'), edits)
        withdrawals = self._note_requests(reading, 'drop')
        for entry in withdrawals:
            self._note_withdraw(entry, edits)
        for entry in self._note_requests(reading, 'add'):
            self._note_add(entry, edits)
        if gone_ids:
            edits.append('no longer available: ' + ', '.join(gone_ids))
        if reading.get('price_moved') is True:
            edits.append('a price has moved')
        summary = str(reading.get('summary') or '').strip()
        wider_hunt = withdrawals or gone_ids or self.note_reports_sellout or budget_moved
        if self.parsed_brief.get('pick') == 'cheapest' and wider_hunt:
            self.broader_search_done, self.cheaper_check_marks = (False, set())
        if edits or summary:
            self.state_version, self.recent_calls = (self.state_version + 1, [])
            self.change_notes.append(summary or '; '.join(edits))
        tail = ' (' + '; '.join(edits) + ')' if edits else ''
        return (summary + tail).strip()

    def read_store_note(self, note_lines: list[str]) -> str:
        """Has a model read a free-text note (from the store or the shopper) and applies it: listings gone, a new budget,
        requirements dropped or added, a price move. Returns a short account, or '' when nothing changed."""
        self.seen_note_lines.update(note_lines)
        ask = STORE_NOTE_PROMPT + '\n\nITEMS AND REQUIREMENTS NOW:\n' + self.ledger_text() + '\n\nTHE NOTE:\n' + '\n'.join(note_lines)
        reading = self.model_json(ask, [FAST_MODEL, BACKUP_MODEL], 2500) or {}
        gone_ids = self._note_gone(reading, note_lines)
        return self._apply_note_change(reading, gone_ids) if reading.get('kind') in (None, 'change') else ''

    # ------------------------------------------------------------------ the request's own steps before an ending

    def _gate_menu(self, finishers: set[str]) -> list[str]:
        menu = []
        for spec in self.tool_specs:
            fn = spec['function']
            if fn['name'] not in finishers:
                menu.append(f"- {fn['name']}: {fn.get('description', '')} | arguments: {_to_json(fn.get('parameters') or {})}")
        return menu

    def _gate_prompt_text(self, phase: str, due: list[dict[str, Any]], menu: list[str]) -> str:
        ordering = phase == 'order'
        history = '\n'.join((f'{number}. {line}' for number, line in enumerate(self.action_record, 1)))
        blocks = [PROCESS_STEP_PROMPT.replace('STAGE', 'place the order' if ordering else 'end without a purchase'), 'REQUEST:\n' + self.request_text, 'STEPS DUE NOW:\n' + '\n'.join((f"- {entry['step']}" for entry in due)), 'LEDGER:\n' + self.ledger_text(), 'CART NOW: ' + self.cart_summary(), 'RECORD (every message and step call up to now):\n' + (history or '(empty so far)')]
        if self.change_notes:
            blocks.append('STORE NOTES SEEN:\n' + '\n'.join(self.change_notes[-4:]))
        blocks.append(f"MESSAGES LEFT: {max(0, self.message_limit - self.messages_sent)}   STEPS LEFT: {self.steps_remaining()} (one of them goes to the {('order' if ordering else 'ending')} itself)")
        blocks.append('TOOLS:\n' + '\n'.join(menu))
        return '\n\n'.join(blocks)

    def _gate_heard(self, verdict: dict[str, Any], acted: list[str]) -> tuple[list[str], str]:
        """Requirements and holds a shopper's reply brought in; only read once Corvin has acted in this gate."""
        heard: list[str] = []
        hold = ''
        if acted:
            self.apply_answers({'needs': verdict.get('needs')}, heard)
            hold = str(verdict.get('hold') or '').strip()
        return (heard, hold)

    def _gate_missing_steps(self, verdict: dict[str, Any], need_talk: bool) -> list[dict[str, Any]]:
        """Steps the verdict does not prove done. When the order needs a message, only record lines made after the
        cart last changed count as proof."""
        numbers = range(1, len(self.action_record) + 1)
        proof = {n for n in numbers if not need_talk or (n <= len(self.action_steps) and self.action_steps[n - 1] > self.cart_changed_step)}

        def proven(entry: dict[str, Any]) -> bool:
            return entry.get('done') is True and any(((_as_int(n) or 0) in proof for n in _as_list(entry.get('lines'))))
        return [entry for entry in _as_list(verdict.get('steps')) if isinstance(entry, dict) and (not proven(entry))]

    def _gate_may_message(self) -> bool:
        return bool(self.tool_roles.get('ask')) and self.messages_sent < self.message_limit and (self.steps_remaining() >= 2)

    def _gate_nudge(self, missing: list[dict[str, Any]]) -> None:
        self.event_log.append({'kind': 'gate_unproven', 'steps': _truncate_strings(missing, 120)})
        named = '; '.join((str(entry.get('step') or '') for entry in missing))[:200]
        self.action_record.append(f'(Corvin: no record line made after the cart\'s latest change (step {self.cart_changed_step}) shows this done: ' + named + f'; {self.message_limit - self.messages_sent} message(s) and {self.steps_remaining()} steps remain, so name the call that does it)')
        self.action_steps += [-1]

    def _gate_ask_shopper(self, args: dict[str, Any]) -> str | None:
        text = next((value for value in args.values() if isinstance(value, str)), '')
        answered = self.send_messages([(1, '', text)], 1)
        if not answered:
            return None
        self.process_messages_used += 1
        return f'Message: {text}\nReply: {answered[0][2]}'

    def _gate_store_step(self, name: str, args: dict[str, Any]) -> str | None:
        outcome = self.store_call(name, args)
        if outcome is None:
            return None
        result_obs = outcome.get('observation')
        self.absorb_records(result_obs)
        shown, result = (_to_json(args), self.render_result(result_obs))
        self.action_record.append(f'step {self.steps_used}: {name} {shown} -> {result[:300]} (cart then: {self.cart_signature()})')
        self.action_steps += [self.steps_used]
        return f'{name} {shown} -> {result[:400]}'

    def _gate_run_call(self, name: str, call: dict[str, Any], finishers: set[str]) -> str | None:
        """Carries out the call the gate model named. Returns the line to report, or None when the gate stops."""
        asker = self.tool_roles.get('ask')
        if name not in self.tool_schemas or name in finishers:
            return None
        if self.steps_remaining() < 2 or (name == asker and self.messages_sent >= self.message_limit):
            return None
        problems: list[str] = []
        given = call.get('args')
        args = _conform_to_schema(given if isinstance(given, dict) else {}, self.tool_schemas[name], 'args', problems)
        if problems:
            return None
        return self._gate_ask_shopper(args) if name == asker else self._gate_store_step(name, args)

    @staticmethod
    def _gate_outcome(acted: list[str]) -> tuple[bool, str]:
        if not acted:
            return (True, '')
        return (True, 'Corvin first carried out what the request wants done at this point:\n' + '\n'.join(acted))

    def run_process_steps(self, phase: str) -> tuple[bool, str]:
        """Runs the request's own steps due right before the order (phase 'order') or a no-purchase ending ('end'): a
        model judges from the numbered record whether they are done and names the call that does a missing one, for up
        to PROCESS_CHECK_ROUNDS rounds. (False, why) when a shopper's reply changes the task, else (True, report)."""
        due = [entry for entry in self.process_steps if entry['when'] == phase]
        acted: list[str] = []
        if self.episode_over or not due or (not self.process_check_pending(phase)):
            return self._gate_outcome(acted)
        finishers = {self.tool_roles.get('buy'), self.tool_roles.get('refuse')} - {None}
        menu = self._gate_menu(finishers)
        need_talk = phase == 'order' and any((entry['messages'] > 0 for entry in due))
        pushed = False
        rounds_left = PROCESS_CHECK_ROUNDS
        while rounds_left > 0:
            rounds_left -= 1
            seats = [STRONG_MODEL, FAST_MODEL] if pushed else [FAST_MODEL, STRONG_MODEL]
            verdict = self.model_json(self._gate_prompt_text(phase, due, menu), seats, 3500)
            if verdict is None:
                return self._gate_outcome(acted)
            heard, hold = self._gate_heard(verdict, acted)
            named_call = verdict['call'] if isinstance(verdict.get('call'), dict) else None
            self.event_log.append({'kind': 'gate', 'stage': phase, 'call': _truncate_strings(named_call, 200), 'steps': _truncate_strings(verdict.get('steps'), 120), 'hold': hold[:200], 'told': heard})
            if heard or hold:
                self.state_version += 1
                changes = '; '.join(heard + ([hold] if hold else []))
                return (False, self._gate_outcome(acted)[1] + '\n' + 'A reply from the shopper changes the task: ' + changes + '. Adjust the cart to it, then check out again.')
            if named_call:
                line = self._gate_run_call(str(named_call.get('tool') or ''), named_call, finishers)
                if line is None:
                    return self._gate_outcome(acted)
                acted.append(line)
            else:
                missing = self._gate_missing_steps(verdict, need_talk)
                if not (missing and (not acted) and (not pushed) and self._gate_may_message()):
                    self.process_checks_passed.add((phase, self.cart_signature() + f'#{self.state_version}'))
                    return self._gate_outcome(acted)
                pushed = True
                self._gate_nudge(missing)
        return self._gate_outcome(acted)

    # ------------------------------------------------------------------ accessory / device fit

    def _fit_text_of(self, listing: str) -> str:
        seen = self.full_records.get(listing) or self.listing_rows.get(listing) or {}
        top = ' '.join((str(value) for value in seen.values() if isinstance(value, str)))
        inner = '. '.join((f'{key} {value}' for block in seen.values() if isinstance(block, dict) for key, value in block.items() if isinstance(value, str)))
        return top + ' ' + inner

    def compatibility_index(self) -> dict[str, Any]:
        """Fit marks of every seen listing, read from its own text, and for each part number the further devices
        some listing says that part also fits."""
        fit_marks: dict[str, dict[str, Any]] = {}
        extra: dict[str, set[str]] = {}
        for listing in self.listing_order:
            mark = fit_marks[listing] = _compatibility_marks(self._fit_text_of(listing))
            for part_no, codes in mark['also']:
                extra.setdefault(part_no, set()).update(codes)
        return {'marks': fit_marks, 'notes': extra}

    @staticmethod
    def _fit_codes_of(listing: str, fit_index: dict[str, Any]) -> set[str]:
        """Device codes a listing says it works with, plus those its own part number is said to fit elsewhere."""
        mark = fit_index['marks'].get(listing) or {}
        codes = set(mark.get('works') or [])
        own_part = mark.get('own')
        if own_part:
            codes |= fit_index['notes'].get(own_part, set())
        return codes

    def _item_pool(self, slot: int) -> list[str]:
        shelf_key = self.category_field()
        item = self.wanted_products[slot - 1]
        return [listing for listing in self.listing_order if self.item_owns_label(item, self._line_shelf(listing, shelf_key))] if shelf_key else []

    def _pair_misfits(self, accessory: int, device_item: int, fit_index: dict[str, Any], held: list[str]) -> list[str]:
        device_pool, part_pool = (self._item_pool(device_item), self._item_pool(accessory))
        devices = [listing for listing in held if listing in device_pool]
        attached = [listing for listing in held if listing in part_pool]
        misfits: list[str] = []
        if len(devices) == 1 and attached:
            device = devices[0]
            device_codes = set(fit_index['marks'].get(device, {}).get('codes') or [])
            part_name = self.wanted_products[accessory - 1].product_name if device_codes else ''
            for line in attached if device_codes else ():
                fits = self._fit_codes_of(line, fit_index)
                if fits and (not _codes_overlap(fits, device_codes)):
                    spares = [cand for cand in part_pool if _codes_overlap(self._fit_codes_of(cand, fit_index), device_codes) and (not self.is_sold_out(cand)) and (cand not in held)][:4]
                    head = f"{line} ({part_name}) fits {', '.join(sorted(fits)[:4])}, which leaves out {device} ({', '.join(sorted(device_codes))})"
                    tail = f"; parts of that kind already seen that fit it: {', '.join(spares)}" if spares else '; no part of that kind seen so far fits it, so search for one or change the device: ' + self.compatibility_plan(short=True)
                    misfits.append(head + tail)
        return misfits

    def compatibility_problems(self) -> list[str]:
        misfits: list[str] = []
        if self.fit_pairs:
            fit_index = self.compatibility_index()
            held = self.cart_listing_ids()
            for accessory, device_item in self.fit_pairs:
                misfits += self._pair_misfits(accessory, device_item, fit_index, held)
        return misfits

    def _fit_plan_row(self, device_item: int, fit_index: dict[str, Any]) -> str:
        """For one device item: the codes its candidate parts name most, how many parts of each kind name them, and
        which seen devices carry them."""
        code_counts: dict[str, dict[int, int]] = {}
        for accessory in (p for p, m in self.fit_pairs if m == device_item):
            for listing in (cand for cand in self._item_pool(accessory) if not self.is_sold_out(cand)):
                for code in (c for c in self._fit_codes_of(listing, fit_index) if not c.startswith('!')):
                    per_part = code_counts.setdefault(code, {})
                    per_part[accessory] = per_part.get(accessory, 0) + 1
        best = sorted(code_counts.items(), key=lambda pair: (-len(pair[1]), -sum(pair[1].values())))[:4]
        device_pool = self._item_pool(device_item)
        bits = []
        for code, per_part in best:
            carriers = [listing for listing in device_pool if _codes_overlap({code}, fit_index['marks'].get(listing, {}).get('codes') or [])][:2]
            counts = ', '.join((f'{n} {self.wanted_products[p - 1].product_name}' for p, n in sorted(per_part.items())))
            bits.append(f'{code}: {counts}' + (f" (device on file: {', '.join(carriers)})" if carriers else ' (no device with it seen yet)'))
        return f'codes that parts for the {self.wanted_products[device_item - 1].product_name} candidates name -> ' + '; '.join(bits) if bits else ''

    def compatibility_plan(self, short: bool=False) -> str:
        rows: list[str] = []
        if self.fit_pairs:
            fit_index = self.compatibility_index()
            rows = [row for row in (self._fit_plan_row(device_item, fit_index) for device_item in sorted({m for _, m in self.fit_pairs})) if row]
        joined = ' | '.join(rows)
        if short or not rows:
            return joined
        return "[fit plan: a part goes with a device when the part's own wording carries that device's model code or a second name the device sells under; codes cannot be searched for, so browse the device's category and take the codes from its result rows] " + joined

    # ------------------------------------------------------------------ lines per requested item

    def _line_shelf(self, listing: str, shelf_key: str) -> str:
        return str((self.listing_rows.get(listing) or {}).get(shelf_key) or '')

    def _owners_of_shelf(self, label: str) -> list[int]:
        return [slot for slot, item in enumerate(self.wanted_products, 1) if self.item_owns_label(item, label)]

    def _item_on_any_shelf(self, item: WantedProduct, labels: list[str]) -> bool:
        return any((self.item_owns_label(item, label) for label in labels))

    def _items_lacking_line(self) -> list[int]:
        """Item numbers the cart has nothing for, checked only while it holds fewer lines than there are items; an
        item whose label appears on none of the seen listings is left out."""
        shelf_key = self.category_field()
        held = [listing for listing in self.cart_listing_ids() if (self._held_count(listing) or 0) > 0]
        lacking: list[int] = []
        if shelf_key and len(held) < len(self.wanted_products):
            every_shelf = [str(row.get(shelf_key) or '') for row in self.listing_rows.values()]
            held_shelves = [self._line_shelf(listing, shelf_key) for listing in held]
            lacking = [slot for slot, item in enumerate(self.wanted_products, 1) if self._item_on_any_shelf(item, every_shelf) and (not self._item_on_any_shelf(item, held_shelves))]
        return lacking

    def _filler_for(self, slot: int, item: WantedProduct, paired: set[int]) -> str | None:
        """The cheapest opened, buyable listing of the item that states all its requirements and still fits the
        budget; None for an item tied to another by a fit rule."""
        able = [] if slot in paired else [cand for cand in self._item_pool(slot) if cand in self.full_records and cand not in self.cart_units and self.can_buy(cand) and self.listing_price(cand) is not None and self._states_item_needs(cand, slot)]
        choice = sorted(able, key=lambda cand: self.listing_price(cand) or 0.0)[0] if able else None
        if choice and isinstance(self.budget_limit, (int, float)):
            per_item = self.parsed_brief.get('budget_scope') == 'each'
            spent = 0.0 if per_item else sum(((self.listing_price(cand) or 0.0) * (self._held_count(cand) or 0) for cand in self.cart_listing_ids()))
            adds = (self.listing_price(choice) or 0.0) * (1 if per_item else item.units_wanted)
            if adds + spent > self.budget_limit + 1e-09:
                choice = None
        return choice

    def _kept_lines(self) -> dict[str, int]:
        return {listing: self.cart_units[listing] for listing in self.cart_listing_ids() if listing in self.cart_units}

    def _push_one_cart_call(self, tool: str, raw: dict[str, Any]) -> bool:
        problems: list[str] = []
        args = _conform_to_schema(raw, self.tool_schemas.get(tool), 'args', problems)
        accepted = not problems and self.store_call(tool, args) is not None
        if accepted:
            self.record_cart_call(tool, args)
        return accepted

    def _cover_item(self, slot: int, tool: str, paired: set[int]) -> bool:
        item = self.wanted_products[slot - 1]
        choice = self._filler_for(slot, item, paired)
        calls = self.cart_change_calls({**self._kept_lines(), choice: item.units_wanted}, []) if choice else None
        covered = bool(calls) and len(calls) + 1 <= self.steps_remaining() and all((self._push_one_cart_call(tool, raw) for raw in calls))
        if covered:
            self.event_log.append({'kind': 'filled_line', 'item': slot, 'line': choice})
        return covered

    def fill_missing_lines(self) -> bool:
        """Before a forced order, gives every item the cart lacks a line: the lowest-priced opened listing whose
        attributes state all of that item's requirements. False as soon as one item cannot be covered, so the episode
        ends with a refusal and not with an incomplete order."""
        tool = self.tool_roles.get('cart') or ''
        paired = {slot for pair in self.fit_pairs for slot in pair}
        return all((self._cover_item(slot, tool, paired) for slot in self._items_lacking_line()))

    def _fuller_shelf_for(self, slot: int, label: str) -> str:
        """A store label that names item `slot` in full when the line's own label covers only part of its name."""
        shelf_key = self.category_field()
        item = self.wanted_products[slot - 1]
        name_words = _root_words(item.product_name)
        settled = len(name_words) < 2 or name_words <= _root_words(label) or not shelf_key or bool(item.category_label and _root_words(label) == _root_words(item.category_label))
        shelves = [] if settled else sorted({str(row.get(shelf_key) or '') for row in self.listing_rows.values()})
        return next((shelf for shelf in shelves if shelf and name_words <= _root_words(shelf)), '')

    def _faults_of_item(self, slot: int, item: WantedProduct, lines: list[str], shelf: dict[str, str], owners: dict[str, list[int]]) -> list[str]:
        tag = f'item {slot} ({item.product_name})'
        mine = [listing for listing in lines if slot in owners[listing]]
        if not mine:
            return [f'{tag} has no line in the cart yet']
        notes: list[str] = []
        if sum((1 for listing in mine if owners[listing] == [slot])) > 1:
            notes.append(f"{tag} has {len(mine)} lines ({', '.join(mine)}); keep one")
        for listing in mine:
            units = self._held_count(listing)
            if owners[listing] == [slot] and units != item.units_wanted:
                notes.append(f'{listing} is held at {units} unit(s), while {tag} calls for {item.units_wanted}')
            better = self._fuller_shelf_for(slot, shelf[listing])
            if better:
                notes.append(f'{listing} comes from {shelf[listing]}, yet the store also has the label {better}, which names {tag} completely: take that line from there instead')
        return notes

    def _line_budget_faults(self, lines: list[str]) -> list[str]:
        cap = self.budget_limit
        prices = {listing: self.listing_price(listing) for listing in lines}
        over: list[str] = []
        if isinstance(cap, (int, float)) and (not any((price is None for price in prices.values()))):
            if self.parsed_brief.get('budget_scope') == 'each':
                over = [f'{listing} costs {prices[listing]:g}, above the per-item limit of {cap:g}' for listing in lines if cap < prices[listing] <= 10 * cap]
            else:
                total = sum((prices[listing] * self._held_count(listing) for listing in lines))
                over = [f'the lines add up to {total:g}, above the budget of {cap:g}'] if cap + 1e-09 < total <= 10 * cap else over
        return over

    def cart_line_problems(self) -> list[str]:
        self.assign_stray_labels()
        shelf_key = self.category_field()
        lines = [listing for listing in self.cart_listing_ids() if (self._held_count(listing) or 0) > 0]
        problems: list[str] = []
        if shelf_key and lines:
            shelf = {listing: self._line_shelf(listing, shelf_key) for listing in lines}
            every_shelf = [str(row.get(shelf_key) or '') for row in self.listing_rows.values()]
            stocked = {slot for slot, item in enumerate(self.wanted_products, 1) if self._item_on_any_shelf(item, every_shelf)}
            owners = {listing: self._owners_of_shelf(shelf[listing]) for listing in lines}
            for slot, item in enumerate(self.wanted_products, 1):
                if slot in stocked:
                    problems += self._faults_of_item(slot, item, lines, shelf, owners)
            if len(stocked) == len(self.wanted_products):
                problems += [f"{listing} ({shelf[listing] or 'unknown kind'}) is no kind of product the shopper wants: remove it" for listing in lines if not owners[listing]]
            problems += [f'{listing} can no longer be bought (sold out)' for listing in lines if self.is_sold_out(listing)]
            problems += self._line_budget_faults(lines)
        return problems

    # ------------------------------------------------------------------ count fixes Corvin makes itself

    def quantity_corrections(self) -> list[tuple[str, int, int]]:
        shelf_key = self.category_field()
        held = self.cart_listing_ids() if shelf_key else []
        wrong = []
        for listing in held:
            units = self._held_count(listing) or 0
            owners = self._owners_of_shelf(self._line_shelf(listing, shelf_key))
            if units > 0 and len(owners) == 1 and (not self.is_sold_out(listing)):
                item = self.wanted_products[owners[0] - 1]
                peers = [other for other in held if (self.cart_units.get(other, 1) or 0) > 0 and self.item_owns_label(item, self._line_shelf(other, shelf_key))]
                if units != item.units_wanted and len(peers) == 1:
                    wrong.append((listing, units, item.units_wanted))
        return wrong

    def _count_slots(self, tool: str, schema: dict[str, Any]) -> tuple[str, str, str | None] | None:
        """The cart tool's id, count and mode parameters; None without an id or a count parameter."""
        params = schema.get('properties') or {}
        free_text = [key for key, spec in params.items() if isinstance(spec, dict) and 'string' in _schema_types(spec) and ('enum' not in spec)]
        id_key = next((key for owner, key in self.id_params if owner == tool and key in free_text), None) or next((key for key in schema.get('required') or [] if key in free_text), None) or (free_text[0] if len(free_text) == 1 else None)
        count_key = next((key for key, spec in params.items() if isinstance(spec, dict) and 'integer' in _schema_types(spec)), None)
        mode_key = next((key for key, spec in params.items() if isinstance(spec, dict) and isinstance(spec.get('enum'), list) and (len(spec['enum']) >= 2)), None)
        return (id_key, count_key, mode_key) if id_key and count_key else None

    def _count_calls_for(self, tool: str, wrong: list[tuple[str, int, int]], schema: dict[str, Any], slots: tuple[str, str, str | None]) -> list[dict[str, Any]] | None:
        """Cart calls that set each wrong line to its wanted count: one call when the tool sets counts, else a removal of
        the held units and an add of the wanted ones; None when the tool can do neither."""
        id_key, count_key, mode_key = slots
        planned_calls: list[dict[str, Any]] = []
        for listing, units, wanted in wrong:
            if self.cart_sets_quantity(tool):
                planned_calls.append({id_key: listing, count_key: wanted})
            elif mode_key:
                modes = [str(option) for option in schema['properties'][mode_key]['enum']]
                take_off = next((mode for mode in modes if re.search('remov|delet|drop|clear|take', mode, re.I)), modes[1])
                put_on = next((mode for mode in modes if mode != take_off), modes[0])
                planned_calls.append({id_key: listing, mode_key: take_off, count_key: units})
                planned_calls.append({id_key: listing, mode_key: put_on, count_key: wanted})
            else:
                return None
        return planned_calls

    def _send_count_calls(self, tool: str, schema: dict[str, Any], planned_calls: list[dict[str, Any]]) -> list[str]:
        sent: list[str] = []
        for raw in planned_calls:
            problems: list[str] = []
            args = _conform_to_schema(raw, schema, 'args', problems)
            answer = None if problems else self.store_call(tool, args)
            if answer is None or self.episode_over:
                return sent
            self.record_cart_call(tool, args)
            result_obs = answer.get('observation')
            self.last_cart_reply = _to_json(result_obs)
            self.take_observation(tool, result_obs)
            sent.append(_to_json(args) + ' -> ' + self.render_result(result_obs)[:300])
        return sent

    def apply_quantity_fixes(self, fixes: list[tuple[str, int, int]]) -> list[str]:
        tool = self.tool_roles.get('cart') or ''
        schema = self.tool_schemas.get(tool) or {}
        slots = self._count_slots(tool, schema)
        planned_calls = self._count_calls_for(tool, fixes, schema, slots) if tool and slots is not None else None
        sent: list[str] = []
        if planned_calls is not None and len(planned_calls) + 3 <= self.steps_remaining():
            sent = self._send_count_calls(tool, schema, planned_calls)
            self.event_log.append({'kind': 'count_fix', 'calls': sent})
        return sent

    # ------------------------------------------------------------------ the last careful review

    def review_signature(self, listing_ids: list[str]) -> str:
        return '|'.join(sorted(listing_ids)) + '#' + str(self.state_version)

    def _review_prompt_text(self, lines: list[str], planned: bool, ranked: bool) -> str:
        blocks = [CART_REVIEW_PROMPT, 'REQUEST:\n' + self.request_text, 'REQUIREMENTS IN FORCE:\n' + self.ledger_text()]
        wanted = '; '.join((f'item {slot}: quantity {item.units_wanted}' for slot, item in enumerate(self.wanted_products, 1))) if planned else ''
        blocks.append('CART (not carted yet; judge these listings as though each sat in the cart at the quantity asked for: ' + wanted + '):\n' + ', '.join(lines) if planned else 'CART:\n' + self.cart_summary())
        records = [f'{listing}: ' + ('SOLD OUT (the store reports it cannot be bought any more). ' if self.is_sold_out(listing) else '') + self.record_text(listing) for listing in lines]
        blocks.append("RECORDS OF THE CART'S LISTINGS:\n" + '\n'.join(records))
        if ranked:
            blocks.append('OTHER LISTINGS SEEN (most recent first):\n' + self._rival_rows(lines, 15))
        return '\n\n'.join(blocks)

    def _review_judgement(self, prompt: str, seats: list[str], planned: bool, ranked: bool, limit: float) -> dict[str, Any] | None:
        """A cheap first look on an unranked live cart, escalated to the strong seat only when it objects; otherwise the
        seats the spend allows."""
        verdict = None
        if len(seats) > 1 and not planned and (not self.decline_reviewed) and (not ranked):
            verdict = self.model_json(prompt, [FAST_MODEL], 3000)
            if verdict:
                seats = [STRONG_MODEL] if verdict.get('ok') is False and STRONG_MODEL in self.review_models(limit) else []
        if seats and (not self.credit_exhausted):
            verdict = self.model_json(prompt, seats, 3000) or verdict
        return verdict

    def _review_rebuttal(self, verdict: dict[str, Any], lines: list[str]) -> str:
        self.review_leads = [lead for lead in self.review_leads if lead not in lines]
        problems = [str(problem) for problem in _as_list(verdict.get('problems'))][:5]
        upgrades = [str(alt) for alt in _as_list(verdict.get('better')) if str(alt) in self.listing_rows and (not self.is_sold_out(str(alt)))][:4]
        note = 'A second, careful check finds this cart not right yet: ' + '; '.join(problems or ['compare it with the requirements'])
        if upgrades:
            note += '. Listings that look like better picks: ' + ', '.join(upgrades)
        return note + '. Correct the cart (or, if the check is mistaken, show it by reading the records again), then check out.'

    def _run_review(self, sig: str, lines: list[str], planned: bool, seats: list[str], limit: float) -> str:
        self.reviewed_signatures.add(sig)
        ranked = self.parsed_brief.get('pick') in ('cheapest', 'most', 'least')
        verdict = self._review_judgement(self._review_prompt_text(lines, planned, ranked), seats, planned, ranked, limit)
        self.event_log.append({'kind': 'check', 'cart': lines, 'got': verdict})
        self.review_verdicts[sig] = bool(verdict) and verdict.get('ok') is not False
        return self._review_rebuttal(verdict, lines) if verdict and verdict.get('ok') is False else ''

    def review_cart(self, planned_ids: list[str] | None=None, limit: float=SOFT_COST_LIMIT) -> str:
        planned = planned_ids is not None
        lines = planned_ids if planned else self.cart_listing_ids()
        sig = self.review_signature(lines)
        seats = self.review_models(limit)
        fresh = sig not in self.reviewed_signatures and bool(seats) and bool(lines)
        return self._run_review(sig, lines, planned, seats, limit) if fresh else ''

class EpisodeCart:
    """Cart side of Corvin's episode: lines in the cart, sold-out swaps, the audit before a decline, pointer trails.

    Every helper works from the store's own tool schemas and the listing records already read; nothing is tuned to
    one world. Store and model calls keep a fixed order, so a replay of the same episode makes the same calls.
    """

    _SHOP_FIELD_RE = 'seller|shop|store|vendor|brand|maker|merchant'
    _LINK_FILLER = frozenset({'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for', 'of'})
    _DECLINE_REMARK = 'Sorry, nothing in this store meets every one of your requirements, so I will not buy anything.'

    # ------------------------------------------------------------------ schema readings shared by the cart helpers

    @staticmethod
    def _free_text_arg(spec: Any) -> bool:
        """A free string argument: typed string and not limited to a list of options."""
        return isinstance(spec, dict) and 'string' in _schema_types(spec) and 'enum' not in spec

    @staticmethod
    def _is_option_switch(spec: Any) -> bool:
        """An argument choosing between two or more fixed options (an add / remove style switch)."""
        return isinstance(spec, dict) and isinstance(spec.get('enum'), list) and len(spec['enum']) >= 2

    @staticmethod
    def _first_count_arg(fields: dict[str, Any]) -> str | None:
        return next((key for key, spec in fields.items() if isinstance(spec, dict) and 'integer' in _schema_types(spec)), None)

    def _first_switch_arg(self, fields: dict[str, Any]) -> str | None:
        return next((key for key, spec in fields.items() if self._is_option_switch(spec)), None)

    def _cart_shape_args(self, tool: str, raw: dict[str, Any]) -> dict[str, Any] | None:
        """Raw arguments shaped to the tool's schema, or None when the schema refuses them."""
        problems: list[str] = []
        shaped = _conform_to_schema(raw, self.tool_schemas.get(tool), 'args', problems)
        return shaped if not problems else None

    def _cart_send_shaped(self, tool: str, raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None] | None:
        """(arguments sent, store reply); None when the schema refused the arguments and nothing was sent."""
        shaped = self._cart_shape_args(tool, raw)
        return None if shaped is None else (shaped, self.store_call(tool, shaped))

    def cart_sets_quantity(self, tool_name: str) -> bool:
        specs = list((self.tool_schemas.get(tool_name, {}).get('properties') or {}).values())
        return not any((self._is_option_switch(spec) for spec in specs)) and any((isinstance(spec, dict) and 'integer' in _schema_types(spec) and spec.get('minimum') == 0 for spec in specs))

    def full_removal_args(self, tool_name: str, call_args: dict[str, Any]) -> dict[str, Any]:
        """Give a removal without a count the full number of units held; the store would otherwise drop only one."""
        fields = self.tool_schemas.get(tool_name, {}).get('properties') or {}
        count = self._first_count_arg(fields)
        removing = any((isinstance((fields.get(key) or {}).get('enum'), list) and value in fields[key]['enum'][1:] for key, value in call_args.items()))
        held = next((value for value in call_args.values() if isinstance(value, str) and value in self.cart_units), None)
        if count and removing and held and count not in call_args:
            units = self.cart_units[held]
            return {**call_args, count: min(units, fields[count].get('maximum', units))}
        return call_args

    # ------------------------------------------------------------------ ending calls and the tool menu

    def _end_call_matches(self, role: str, fixed_key: str, tool_name: str, call_args: dict[str, Any]) -> bool:
        fixed = self.parsed_brief.get(fixed_key)
        pairs = fixed.items() if isinstance(fixed, dict) else ()
        return tool_name == self.tool_roles.get(role) and all((call_args.get(key) == value for key, value in pairs))

    def is_checkout_call(self, tool_name: str, call_args: dict[str, Any]) -> bool:
        return self._end_call_matches('buy', 'buy_args', tool_name, call_args)

    def is_decline_call(self, tool_name: str, call_args: dict[str, Any]) -> bool:
        return self._end_call_matches('refuse', 'refuse_args', tool_name, call_args)

    def _late_tool_names(self, steps_left: int, finals: set[str]) -> set[Any] | None:
        """The tools still worth offering close to the step limit; None while every tool stays on offer."""
        reserve = self.closing_steps()
        cart_side = {self.tool_roles.get('cart'), self.tool_roles.get('cart_view')}
        if steps_left <= reserve:
            return set(finals)
        if steps_left <= reserve + 3:
            return finals | cart_side
        tight = steps_left <= len(self.wanted_products) + 5 and self.checkout_attempts == 0
        return finals | cart_side | {self.tool_roles.get('offer')} if tight else None

    def available_tools(self) -> list[dict[str, Any]]:
        steps_left = self.steps_remaining()
        interviewer_tool = self.tool_roles.get('ask')
        finals = {self.tool_roles.get('buy'), self.tool_roles.get('refuse')} - {None}
        allowed = self._late_tool_names(steps_left, finals)
        offered = [spec for spec in self.tool_specs if spec['function']['name'] != interviewer_tool]
        narrowed = [spec for spec in offered if spec['function']['name'] in allowed] if allowed is not None and finals else []
        return narrowed or offered

    def silent_decline_args(self, tool_name: str, call_args: dict[str, Any]) -> dict[str, Any]:
        schema = self.tool_schemas.get(tool_name) or {}
        required = schema.get('required') or []
        quiet = self.parsed_brief.get('quiet_decline') is not False
        reply_args = dict(call_args)
        for key, spec in (schema.get('properties') or {}).items():
            current = reply_args.get(key)
            if self._free_text_arg(spec) and (current is not None or key in required):
                said = str(current or '')
                if quiet or not said or any((lid in said for lid in self.listing_rows)):
                    reply_args[key] = self._DECLINE_REMARK[:spec.get('maxLength', 200)]
        return reply_args

    # ------------------------------------------------------------------ listing ids written as text

    def _map_listing_ref(self, given: Any) -> tuple[Any, str]:
        """One argument value -> (value to send, problem). A text that starts exactly one listing becomes its id."""
        if not isinstance(given, str):
            return (given, '')
        trimmed = given.strip()
        if trimmed in self.listing_rows:
            return (trimmed, '')
        if _looks_like_id(given, True) or len(trimmed) < 8:
            return (given, '')
        prefix = trimmed.lower()
        matches = [lid for lid, row in self.listing_rows.items() if any((key != self.id_field and isinstance(text, str) and text.strip().lower().startswith(prefix) for key, text in row.items()))]
        if len(matches) == 1:
            return (matches[0], '')
        if matches:
            return (given, f"the text {given!r} begins several listings ({', '.join(matches[:6])}); give a single listing id")
        return (given, f'{given!r} is not a listing id; use the id given in the result rows (their {self.id_field} field)')

    def resolve_listing_ids(self, tool_name: str, call_args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        id_tools = {self.tool_roles.get(role) for role in ('cart', 'open', 'offer', 'compare')} - {None}
        if tool_name not in id_tools or not self.id_field:
            return (call_args, '')
        fields = (self.tool_schemas.get(tool_name) or {}).get('properties') or {}
        slots = [key for key, spec in fields.items() if isinstance(spec, dict) and ('string' in _schema_types(spec) or 'array' in _schema_types(spec)) and ('enum' not in spec)]
        usable = len(slots) == 1 and (tool_name == self.tool_roles.get('cart') or (tool_name, slots[0]) in self.id_params)
        if not usable:
            return (call_args, '')
        slot = slots[0]
        if slot not in call_args:
            return (dict(call_args), '')
        several = isinstance(call_args[slot], list)
        mapped = [self._map_listing_ref(item) for item in (call_args[slot] if several else [call_args[slot]])]
        trouble = next((issue for _, issue in mapped if issue), '')
        if trouble:
            return (call_args, trouble)
        return ({**call_args, slot: [item for item, _ in mapped] if several else mapped[0][0]}, '')

    # ------------------------------------------------------------------ sold-out lines and fresh offers

    def is_sold_out(self, lid: str) -> bool:
        return lid in self.sold_out_at or lid in self.unavailable_ids

    def price_stock_tool(self) -> str | None:
        roles = self.tool_roles
        return roles.get('offer') or roles.get('open')

    def _stale_cart_ids(self) -> list[str]:
        """Cart lines not re-read since two steps ago or since the last refused checkout, sold-out ones left out."""
        cutoff = max(self.steps_used - 2, self.checkout_refused_step)
        return [lid for lid in self.cart_listing_ids() if self.offer_read_step.get(lid, -99) < cutoff and lid not in self.sold_out_at]

    def _offer_shift(self, lid: str, snapshot: dict[str, Any], page: Any) -> str:
        fresh = self.listing_rows.get(lid) or {}
        moved = [key for key, value in fresh.items() if key in snapshot and snapshot[key] != value and (not isinstance(value, (dict, list)))]
        return f"{lid} now differs ({', '.join(moved[:4])}): {_to_json(_truncate_strings(page, 300))[:900]}" if moved else ''

    def _reread_offers(self, viewer: str, slot: str, stale: list[str]) -> list[str]:
        shifts: list[str] = []
        for lid in stale:
            snapshot = dict(self.listing_rows.get(lid) or {})
            sent = self._cart_send_shaped(viewer, {slot: lid})
            if sent is None or sent[1] is None or self.episode_over:
                return shifts
            page = sent[1].get('observation')
            self.offer_read_step[lid] = self.steps_used
            self.take_observation(viewer, page)
            shift = self._offer_shift(lid, snapshot, page)
            if shift:
                shifts.append(shift)
        return shifts

    def refresh_cart_offers(self) -> str:
        viewer = self.price_stock_tool()
        slot = self.single_id_param(viewer)
        stale = self._stale_cart_ids() if viewer and slot else []
        shifts: list[str] = []
        if stale and len(stale) + 3 <= self.steps_remaining():
            shifts = self._reread_offers(viewer, slot, stale)
            self.event_log.append({'kind': 'reread', 'ids': stale, 'changed': len(shifts)})
            if shifts:
                self.state_version += 1
        return '\n'.join(shifts)

    def locate_sold_line(self) -> str:
        live = [lid for lid in self.cart_listing_ids() if lid not in self.sold_out_at]
        if len(live) > 1 and not any((self.is_sold_out(lid) for lid in live)):
            return self.refresh_cart_offers()
        if len(live) == 1 and (not self.is_sold_out(live[0])):
            self.unavailable_ids.add(live[0])
        return ''

    def _swap_hint(self, options: list[str]) -> str:
        if not options:
            return ' Take that line out of the cart and put in the best other listing that fits; search for one if nothing you have read fits.'
        first, listed = (options[0], ', '.join(options))
        if self.steps_remaining() <= 6:
            return f" Only a few steps remain: this turn, add {first} at the requested quantity and take out the sold-out line (two calls), then check out. Other already-read listings that seem to fit: {', '.join(options[1:]) or 'none'}."
        if self.parsed_brief.get('pick') == 'cheapest':
            return f" Already-read listings that seem to fit that item in its place, cheapest first: {listed}. The pick rule is the cheapest, so unless the whole fitting set has been seen already, run one search below {first}'s price (with the search's maximum price, if it offers one) before replacing the sold-out line with the cheapest fitting listing."
        return f' Already-read listings that seem to fit that item in its place: {listed}. This turn, remove the sold-out line from the cart and add {first} at the requested quantity, then check out.'

    def _announce_sold(self, gone: list[str]) -> str:
        self.sold_out_at.update({lid: self.steps_used for lid in gone})
        self.review_leads = [lid for lid in self.review_leads if lid not in gone]
        self.state_version += 1
        notice = f"[Corvin] Now sold out: {', '.join(gone)} (the store shows it can no longer be bought). It will not return, so do not open it again."
        return notice + self._swap_hint(self._swap_candidates(gone))

    def sold_out_advice(self) -> str:
        gone = [lid for lid in self.cart_listing_ids() if lid not in self.sold_out_at and self.is_sold_out(lid)]
        return self._announce_sold(gone) if gone else ''

    def _swap_candidates(self, gone: list[str]) -> list[str]:
        """Ask a review seat which already-read listings could stand in for the sold-out lines."""
        others = [lid for lid in reversed(self.listing_order) if lid not in gone and (not self.is_sold_out(lid))][:25] if self.review_models() else []
        ranked: list[str] = []
        if others:
            records = [_to_json(_truncate_strings(self.full_records.get(lid) or self.listing_rows.get(lid), 200))[:1200] for lid in others]
            prompt = REPLACEMENT_PROMPT + '\n\nREQUEST:\n' + self.request_text + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger_text() + '\n\nCART LINE NOW SOLD OUT:\n' + '\n'.join((f'{lid}: {self.record_text(lid)}' for lid in gone)) + '\n\nCART NOW:\n' + self.cart_summary() + '\n\nRECORDS OF OTHER LISTINGS SEEN (newest first):\n' + '\n'.join(records)
            priced = self._priced_fits(self.model_json(prompt, self.review_models()[::-1], 3500), gone)
            by_price = self.parsed_brief.get('pick') == 'cheapest'
            ranked = [lid for _, lid in (sorted(priced, key=lambda fit: fit[0]) if by_price else priced)]
            self.event_log.append({'kind': 'swap', 'lost': gone, 'fits': priced})
        return ranked[:4]

    def _priced_fits(self, answer: dict[str, Any] | None, gone: list[str]) -> list[tuple[float, str]]:
        priced: list[tuple[float, str]] = []
        for entry in _as_list((answer or {}).get('fits')):
            card = isinstance(entry, dict)
            lid = str(entry.get('id')) if card else str(entry)
            fresh = lid in self.listing_rows and lid not in gone and lid not in self.sold_out_at
            if fresh and lid not in self.cart_listing_ids() and all((lid != seen for _, seen in priced)):
                price = entry.get('price') if card else None
                numeric = isinstance(price, (int, float)) and (not isinstance(price, bool))
                priced.append((float(price) if numeric else float('inf'), lid))
        return priced

    # ------------------------------------------------------------------ audit before ending without a purchase

    def _audit_line(self, lid: str) -> str:
        page = self.full_records.get(lid)
        return 'OPENED ' + _to_json(_truncate_strings(page, 200))[:1200] if page else 'ROW ' + _to_json(_truncate_strings(self.listing_rows.get(lid), 120))[:500]

    def _decline_audit_call(self) -> dict[str, Any] | None:
        """One audit of the records read so far (light seat, then the strong one when the light one names no fit)."""
        candidates = [lid for lid in reversed(self.listing_order) if not self.is_sold_out(lid)][:30]
        verdict = None
        if candidates:
            prompt = DECLINE_REVIEW_PROMPT + '\n\nREQUEST:\n' + self.request_text + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger_text() + '\n\nRECORDS OF LISTINGS SEEN (newest first):\n' + '\n'.join((self._audit_line(lid) for lid in candidates))
            seats = self.review_models()
            verdict = self.model_json(prompt, [FAST_MODEL] if seats else [], 4000)
            if STRONG_MODEL in seats and (not self.credit_exhausted) and (not self._audit_fits(verdict)):
                verdict = self.model_json(prompt, [STRONG_MODEL], 4000) or verdict
            self.event_log.append({'kind': 'audit', 'got': verdict})
        return verdict

    def _audit_fits(self, verdict: dict[str, Any] | None) -> dict[str, list[str]]:
        """item -> up to four buyable ids the audit says fit; items left with none are dropped."""
        fits = (verdict or {}).get('fits')
        per_item = {str(item): [str(lid) for lid in _as_list(ids) if str(lid) in self.listing_rows and (not self.is_sold_out(str(lid)))][:4] for item, ids in fits.items()} if isinstance(fits, dict) else {}
        return {item: ids for item, ids in per_item.items() if ids}

    def _audit_open_plan(self, rows: list[str]) -> list[tuple[str, dict[str, Any]]] | None:
        list_param, compare_tool = (self.compare_list_param(), self.tool_roles.get('compare'))
        single = self.attribute_tool()
        slot = self.single_id_param(single)
        if list_param and compare_tool and (not self.compare_lacks_attributes()):
            return [(compare_tool, {list_param: rows[:self.compare_capacity()]})]
        return [(single, {slot: lid}) for lid in rows[:2]] if single and slot else None

    def _run_audit_plan(self, plan: list[tuple[str, dict[str, Any]]]) -> bool:
        """Send the planned reads; False only when a schema refused one (the caller then logs nothing)."""
        for tool, raw in plan:
            sent = self._cart_send_shaped(tool, raw)
            if sent is None or sent[1] is None or self.episode_over:
                return sent is not None
            self.take_observation(tool, sent[1].get('observation'))
        return True

    def _open_audit_rows(self, verdict: dict[str, Any] | None) -> bool:
        """Read the pages of unread rows the audit asked to see; True when a new full record arrived."""
        rows = [str(lid) for lid in _as_list((verdict or {}).get('open')) if str(lid) in self.listing_rows and str(lid) not in self.opened_ids and (not self.is_sold_out(str(lid)))]
        plan = self._audit_open_plan(rows) if rows else None
        grew = False
        known_before = len(self.full_records)
        if plan is not None and len(plan) + 5 <= self.steps_remaining() and self._run_audit_plan(plan):
            self.event_log.append({'kind': 'audit_open', 'rows': rows, 'opened': len(self.full_records) - known_before})
            grew = len(self.full_records) > known_before
        return grew

    def _adopt_leads(self, leads: dict[str, list[str]], verdict: dict[str, Any]) -> str:
        self.leads_by_item = leads
        self.review_leads = [lid for ids in leads.values() for lid in ids]
        why = verdict.get('reason')
        per_item = '; '.join((f"item {item}: {', '.join(ids)}" for item, ids in leads.items()))
        return 'A closer review suggests these listings could fit: ' + per_item + (f' ({why})' if why else '') + '. Open them and buy whatever really fits; end without buying only if every one of them fails.'

    def review_decline(self) -> str:
        message = ''
        if not self.decline_reviewed and self.review_models():
            self.decline_reviewed = True
            verdict = self._decline_audit_call()
            leads = self._audit_fits(verdict)
            if not leads and self._open_audit_rows(verdict):
                verdict = self._decline_audit_call()
                leads = self._audit_fits(verdict)
            if leads:
                message = self._adopt_leads(leads, verdict)
        return message

    # ------------------------------------------------------------------ pages a requirement points at

    def _seller_labels(self, lid: str) -> list[str]:
        fields = self.listing_rows.get(lid) or {}
        return [str(value).strip() for key, value in fields.items() if isinstance(value, str) and 2 < len(value.strip()) <= 40 and re.search(self._SHOP_FIELD_RE, str(key), re.I)]

    def _same_shop(self, about: str, pool: Any) -> list[str]:
        """Ids from pool whose seller-like field is named, as a whole word, inside the description."""
        lowered = about.lower()
        return [lid for lid in pool if any((re.search('(?<![a-z0-9])' + re.escape(label.lower()) + '(?![a-z0-9])', lowered) for label in self._seller_labels(lid)))]

    def _unopened_shop_rows(self, about: str, ids: list[str]) -> list[str]:
        return [lid for lid in self._same_shop(about, ids) if lid not in self.opened_ids]

    def _shop_name_words(self, about: str) -> set[str]:
        """Words of the shop's own name, so a search for the link does not count them as the values sought."""
        pool = self._unopened_shop_rows(about, list(self.listing_rows)) + self._same_shop(about, self.opened_ids)
        return {word for lid in pool for key, value in (self.listing_rows.get(lid) or {}).items() if isinstance(value, str) and len(value.strip()) <= 40 and re.search(self._SHOP_FIELD_RE, str(key), re.I) for word in re.findall('[a-z0-9]{3,}', value.lower())}

    def _page_mentions_all(self, lid: str, wanted: set[str]) -> bool:
        todo, scalars = ([self.full_records.get(lid) or self.listing_rows.get(lid) or {}], [])
        while todo:
            node = todo.pop()
            if not isinstance(node, (dict, list)):
                scalars.append(str(node))
            elif isinstance(node, dict):
                todo.extend(node.values())
            elif isinstance(node, list):
                todo.extend(node)
        blob = ' '.join(scalars).lower()
        return all((re.search('(?<![a-z0-9])' + re.escape(word) + '(?![a-z0-9])', blob) for word in wanted))

    def _link_search_rows(self, about: str, find_words: str) -> list[str]:
        """Unopened rows of the named shop from a word search for the link (one search per wording, then memo)."""
        search_plan = self.word_search_plan()
        memo_key = ' '.join(find_words.lower().split())
        if memo_key and memo_key in self.link_search_results:
            self.link_search_count += 1
            return self._unopened_shop_rows(about, self.link_search_results[memo_key])
        shaped = self._cart_shape_args(search_plan[0], {**search_plan[1], search_plan[2]: find_words}) if search_plan and find_words else None
        rows: list[str] = []
        if shaped is not None:
            self.link_search_count += 1
            response = self.store_call(search_plan[0], shaped)
            if response is not None:
                page = response.get('observation')
                self.take_observation(search_plan[0], page)
                self.link_search_results[memo_key] = [lid for lid, _ in self.listing_records(page)]
                rows = self._unopened_shop_rows(about, self.link_search_results[memo_key])
                self.event_log.append({'kind': 'link_search', 'words': find_words[:80], 'rows': rows[:4]})
        return rows

    def _seek_link(self, about: str, find_words: str) -> tuple[list[str], str | None]:
        """Find the listing a chain link describes: (rows worth opening, id of an opened page that already shows it)."""
        shop_words = self._shop_name_words(about)
        source = (find_words or about).lower()
        wanted = {word for word in re.findall('[a-z0-9]+', source) if len(word) >= 3 or word.isdigit()} - shop_words - self._LINK_FILLER
        known = next((lid for lid in self._same_shop(about, self.opened_ids) if wanted and self._page_mentions_all(lid, wanted)), None)
        if known is not None:
            self.event_log.append({'kind': 'link_known', 'id': known})
            return ([], known)
        found = self._link_search_rows(about, find_words)
        kind = set(re.findall('[a-z0-9]{3,}', source)) - shop_words - self._LINK_FILLER

        def same_kind(candidates: list[str]) -> list[str]:
            return [lid for lid in candidates if not kind or kind & set(re.findall('[a-z0-9]{3,}', _to_json(self.listing_rows.get(lid) or {}).lower()))]
        return ((same_kind(found) or same_kind(self._unopened_shop_rows(about, list(reversed(self.listing_order)))))[:2], None)

    def _page_or_row(self, lid: str) -> Any:
        return self.full_records.get(lid) or self.listing_rows.get(lid)

    def _trail_records(self) -> list[str]:
        """OPENED pages first (at most 45), then ROW entries, newest first, at most 60 lines in all."""
        newest = list(reversed(self.listing_order))
        read = [lid for lid in newest if lid in self.opened_ids and self._page_or_row(lid)][:45]
        lines = ['OPENED ' + _to_json(_truncate_strings(self._page_or_row(lid), 300))[:1400] for lid in read]
        skimmed = [lid for lid in newest if lid not in self.opened_ids or not self._page_or_row(lid)][:60 - len(lines)]
        return lines + ['ROW ' + _to_json(_truncate_strings(self.listing_rows.get(lid), 80))[:300] for lid in skimmed]

    def _ask_trail_model(self, open_needs: list[str]) -> dict[str, Any]:
        prompt = CHAIN_FOLLOW_PROMPT + '\n\nREQUIREMENTS WHOSE VALUE LIVES ON ANOTHER LISTING:\n' + '\n'.join(open_needs) + '\n\nRECORDS (newest first):\n' + '\n'.join(self._trail_records())
        seats = [FAST_MODEL] + ([STRONG_MODEL] if self.model_spend + STRONG_COST_MARGIN <= HARD_COST_LIMIT else [])
        answer = self.model_json(prompt, seats, 4500) or {}
        self.event_log.append({'kind': 'trail', 'pending': open_needs, 'got': _truncate_strings(answer, 150)})
        return answer

    @staticmethod
    def _chain_need(chain: dict[str, Any], open_needs: list[str]) -> str | None:
        said = str(chain.get('need') or '').strip()
        exact = next((need for need in open_needs if need == said), None)
        return exact or (open_needs[0] if len(open_needs) == 1 else None)

    def _chain_value(self, chain: dict[str, Any], links: list[dict[str, Any]], ids: list[str]) -> str | None:
        """The pointed-at value once every link is an opened page and the chain needs no further search, else None."""
        stated = str(chain.get('value') or '').strip()
        if not links or not all((lid in self.opened_ids for lid in ids)):
            return None
        if chain.get('done') is True and stated:
            return stated
        return None if str(links[-1].get('find') or '').strip() else stated or f'as shown on {ids[-1]}'

    def _note_open_link(self, need: str, links: list[dict[str, Any]], ids: list[str], found: tuple[list[str], list[str], list[tuple[str, str]]]) -> None:
        """A chain not settled yet: queue its unopened rows, or note the first link no row matches."""
        hops, missing, gaps = found
        unopened = [lid for lid in ids if lid in self.listing_rows and lid not in self.opened_ids]
        hops.extend([lid for lid in unopened if lid not in hops])
        if not unopened:
            gap = next((link for link, lid in zip(links, ids) if lid not in self.listing_rows), {})
            described = str(gap.get('about') or '') or 'the listing it points at'
            search_words = str(gap.get('find') or '').strip()
            missing.append(f'{need} -> locate {described}' + (f' (search words: {search_words})' if search_words else ''))
            gaps.append((described, search_words))

    def _read_trail_answer(self, answer: dict[str, Any], open_needs: list[str]) -> tuple[list[str], list[str], list[tuple[str, str]]]:
        """-> (unopened rows to open, notes on links still missing, (description, search words) per missing link)."""
        found: tuple[list[str], list[str], list[tuple[str, str]]] = ([], [], [])
        for chain in [entry for entry in _as_list(answer.get('chains')) if isinstance(entry, dict)]:
            need = self._chain_need(chain, open_needs)
            links = [link for link in _as_list(chain.get('links')) if isinstance(link, dict)] if need else []
            ids = [str(link.get('id') or '').strip() for link in links]
            value = self._chain_value(chain, links, ids) if need else None
            if value is not None:
                self.pointer_values[need] = value
                self.state_version += 1
            elif need:
                self._note_open_link(need, links, ids, found)
        return found

    def _open_trail_pages(self, viewer: str, slot: str, ids: list[str], visited: list[str]) -> None:
        for lid in ids:
            sent = self._cart_send_shaped(viewer, {slot: lid})
            if sent is None or sent[1] is None or self.episode_over:
                return
            page = sent[1].get('observation')
            self.take_observation(viewer, page)
            visited.append(f'{lid}: ' + self.render_result(page)[:1500])

    def _trail_report(self, visited: list[str], missing: list[str]) -> tuple[str, bool]:
        pieces: list[str] = []
        if visited:
            pieces.append('[Corvin opened the pages that requirements refer to]\n' + '\n'.join(visited) + '\n')
        values = '; '.join((f'{need} -> {value}' for need, value in self.pointer_values.items()))
        if values:
            pieces.append('[values taken from the pages requirements refer to] ' + values + '\n')
        blocking = bool(missing) and self.chain_block_count < (3 if self.steps_remaining() > 15 else 2) and (self.steps_remaining() > 5)
        if blocking:
            self.chain_block_count += 1
            pieces.append('Checking out counts only once every page a requirement refers to has been opened in full (a search result row does not count), including each further listing such a page sends you on to. Not yet found and opened: ' + ' | '.join(missing[:3]) + ". Search with the shop's name plus the values the description states (that shop's other listings of the same kind are not the one), open the row from that shop whose page shows those values, use the value given on that page, and then check out.")
        return (''.join(pieces), blocking)

    def _trail_round_allowed(self) -> bool:
        return self.chain_rounds < (6 if self.steps_remaining() > 15 else 4) and self.model_spend < HARD_COST_LIMIT

    def follow_pointer_chains(self) -> tuple[str, bool]:
        in_force = {need for _, need in self.live_requirements()}
        open_needs = [need for _, need in self.pointer_musts if need in in_force and need not in self.pointer_values]
        viewer = self.price_stock_tool()
        slot = self.single_id_param(viewer)
        visited: list[str] = []
        missing: list[str] = []
        going = True
        while going and open_needs and self._trail_round_allowed():
            self.chain_rounds += 1
            hops, missing, gaps = self._read_trail_answer(self._ask_trail_model(open_needs), open_needs)
            open_needs = [need for need in open_needs if need not in self.pointer_values]
            if not hops and gaps and viewer and slot and (self.steps_remaining() > 12) and (self.link_search_count < 3):
                hops, known = self._seek_link(*gaps[0])
                if known is not None:
                    visited.append(f'{known}: a page already opened from the shop this chain names, showing the values its description lists; take the value from it')
                    missing = missing[1:]
            going = bool(hops) and bool(viewer) and bool(slot) and len(hops) + 4 <= self.steps_remaining() and (not self.episode_over)
            if going:
                self._open_trail_pages(viewer, slot, hops[:max(0, min(3, 6 - len(visited)))], visited)
        return self._trail_report(visited, missing)

    # ------------------------------------------------------------------ cart calls and buying a confirmed lead

    @staticmethod
    def _is_basket_spec(spec: Any) -> bool:
        entry = spec.get('items') if isinstance(spec, dict) and 'array' in _schema_types(spec) else None
        return isinstance(entry, dict) and isinstance(entry.get('properties'), dict)

    def _basket_call(self, fields: dict[str, Any], wanted_lines: dict[str, int]) -> list[dict[str, Any]] | None:
        """The whole cart in one call, for a cart tool taking a list of line objects (id + count)."""
        key = next((name for name, spec in fields.items() if self._is_basket_spec(spec)))
        inner = fields[key]['items']['properties']
        line_id = next((name for name, spec in inner.items() if self._free_text_arg(spec)), None)
        line_count = self._first_count_arg(inner)
        return [{key: [{line_id: lid, **({line_count: units} if line_count else {})} for lid, units in wanted_lines.items()]}] if line_id else None

    def _cart_id_key(self, tool: str, schema: dict[str, Any], fields: dict[str, Any]) -> str | None:
        texts = [key for key, spec in fields.items() if self._free_text_arg(spec)]
        registered = next((key for owner, key in self.id_params if owner == tool and key in texts), None)
        needed = next((key for key in schema.get('required') or [] if key in texts), None)
        return registered or needed or (texts[0] if len(texts) == 1 else None)

    def _single_line_calls(self, tool: str, fields: dict[str, Any], id_arg: str, wanted_lines: dict[str, int], removals: list[str]) -> list[dict[str, Any]] | None:
        """One call per changed line: removals first, then the lines whose held units differ from the wanted ones."""
        count, switch = (self._first_count_arg(fields), self._first_switch_arg(fields))
        in_cart = self.cart_listing_ids()

        def held_units(lid: str) -> int:
            return self.cart_units.get(lid, 0) if lid in in_cart else 0

        def take_off(lid: str) -> dict[str, Any] | None:
            units = self.cart_units.get(lid, 1)
            if switch:
                return {id_arg: lid, switch: fields[switch]['enum'][1], **({count: units} if count else {})}
            if count and self.cart_sets_quantity(tool):
                return {id_arg: lid, count: 0}
            floor = fields[count].get('minimum') if count else None
            return {id_arg: lid, count: -units} if count and isinstance(floor, int) and floor <= -units else None

        def put_in(lid: str, want: int) -> dict[str, Any] | None:
            refused = held_units(lid) and (not self.cart_sets_quantity(tool))
            return None if refused else {id_arg: lid, **({switch: fields[switch]['enum'][0]} if switch else {}), **({count: want} if count else {})}
        planned = [take_off(lid) for lid in removals] + [put_in(lid, want) for lid, want in wanted_lines.items() if held_units(lid) != want]
        return None if any((call is None for call in planned)) else planned

    def cart_change_calls(self, wanted_lines: dict[str, int], removals: list[str]) -> list[dict[str, Any]] | None:
        tool = self.tool_roles.get('cart') or ''
        cart_schema = self.tool_schemas.get(tool) or {}
        fields = cart_schema.get('properties') or {}
        if any((self._is_basket_spec(spec) for spec in fields.values())):
            return self._basket_call(fields, wanted_lines)
        id_arg = self._cart_id_key(tool, cart_schema, fields)
        return self._single_line_calls(tool, fields, id_arg, wanted_lines, removals) if id_arg else None

    def _lead_kind(self, lid: str, sort_key: str | None) -> str:
        """The listing's category label (lowered), or the id itself when the rows carry no category field."""
        return str((self.listing_rows.get(lid) or {}).get(sort_key) or '').strip().lower() if sort_key else lid

    def _lead_lines(self, sort_key: str | None) -> dict[str, int]:
        """One buyable lead per requested item and per category -> the units that item asks for."""
        chosen: dict[str, int] = {}
        wanted = len(self.wanted_products)
        for item_no, lead_ids in self.leads_by_item.items():
            digits = re.search('\\d+', str(item_no))
            index = int(digits.group()) if digits else (1 if wanted == 1 else None)
            pick = next((lid for lid in lead_ids if not self.is_sold_out(lid)), None)
            if pick and index and 1 <= index <= wanted and self._lead_kind(pick, sort_key) not in {self._lead_kind(other, sort_key) for other in chosen}:
                chosen[pick] = self.wanted_products[index - 1].units_wanted
        return chosen

    def _lead_layout(self, chosen: dict[str, int], sort_key: str | None) -> tuple[dict[str, int], list[str]] | None:
        """(cart lines after the swap, lines to take off), or None when the lines would not cover every item."""
        taken = {self._lead_kind(lid, sort_key) for lid in chosen}
        current = self.cart_listing_ids()
        kept = {lid: self.cart_units.get(lid, 1) for lid in current if lid not in chosen and (not self.is_sold_out(lid)) and (self._lead_kind(lid, sort_key) not in taken)}
        removals = [lid for lid in current if lid not in chosen and lid not in kept]
        covered = len(chosen) + len({self._lead_kind(lid, sort_key) for lid in kept}) >= len(self.wanted_products)
        return ({**kept, **chosen}, removals) if covered else None

    def _lines_pass_review(self, basket_lines: dict[str, int]) -> bool:
        self.review_cart(list(basket_lines), HARD_COST_LIMIT)
        return bool(self.review_verdicts.get(self.review_signature(list(basket_lines))))

    def _send_cart_lines(self, cart_tool: str, calls: list[dict[str, Any]]) -> list[str] | None:
        """Send every cart call in turn; the sent arguments as JSON, or None as soon as one fails."""
        sent_lines: list[str] = []
        for raw in calls:
            sent = self._cart_send_shaped(cart_tool, raw)
            if sent is None or sent[1] is None or self.episode_over:
                return None
            self.record_cart_call(cart_tool, sent[0])
            self.take_observation(cart_tool, sent[1].get('observation'))
            sent_lines.append(_to_json(sent[0]))
        return sent_lines

    def _checkout_once(self, order_tool: str, order_args: dict[str, Any], basket_lines: dict[str, int], removals: list[str], answers: list[str]) -> bool:
        """One checkout try (process steps first); True when the store left room for a second try."""
        cleared, spoken = self.run_process_steps('order')
        if spoken:
            answers.append(spoken)
        outcome = self.store_call(order_tool, order_args) if cleared else None
        if cleared:
            self.event_log.append({'kind': 'completed', 'lines': basket_lines, 'drop': removals, 'reply': _truncate_strings(outcome, 200)})
        if outcome is not None:
            self.checkout_attempts += 1
            if not outcome.get('finished'):
                self.checkout_refused_step = self.steps_used
            self.take_observation(order_tool, outcome.get('observation'))
            answers.append(self.render_result(outcome.get('observation')))
        return outcome is not None and (not self.episode_over) and (not outcome.get('finished'))

    def _retry_allowed(self, basket_lines: dict[str, int]) -> bool:
        self.refresh_cart_offers()
        passed = self._lines_pass_review(basket_lines)
        return passed and self.steps_remaining() >= 1 and (not self.episode_over) and (not any((self.is_sold_out(lid) for lid in basket_lines)))

    def _lead_checkout(self, order_tool: str, basket_lines: dict[str, int], removals: list[str]) -> list[str]:
        """Up to two checkout tries; the second only after fresh offers and a passing review."""
        order_args = self._cart_shape_args(order_tool, dict(self.parsed_brief.get('buy_args') or {}))
        answers: list[str] = []
        if order_args is not None and self._checkout_once(order_tool, order_args, basket_lines, removals, answers):
            if self._retry_allowed(basket_lines):
                self._checkout_once(order_tool, order_args, basket_lines, removals, answers)
        return answers

    def _swap_in_lead(self, lead: str, cart_tool: str, order_tool: str) -> tuple[list[str], list[str]] | None:
        """Rebuild the cart around the confirmed lead and check out: (cart calls sent, store answers) or None."""
        self.lead_bought = True
        sort_key = self.category_field()
        chosen = self._lead_lines(sort_key)
        layout = self._lead_layout(chosen, sort_key) if lead in chosen else None
        calls = self.cart_change_calls(*layout) if layout else None
        if not calls or len(calls) + 1 > self.steps_remaining():
            return None
        basket_lines, removals = layout
        room = len(calls) + 3
        self.open_for_attributes(list(basket_lines), room)
        self.follow_part_links(list(basket_lines), room)
        sent_lines = self._send_cart_lines(cart_tool, calls) if self._lines_pass_review(basket_lines) else None
        answers = self._lead_checkout(order_tool, basket_lines, removals) if sent_lines is not None else []
        return (sent_lines, answers) if answers else None

    def buy_confirmed_lead(self) -> str:
        cart_tool, order_tool = (self.tool_roles.get('cart'), self.tool_roles.get('buy'))
        lead = self.confirmed_lead
        ready = not self.lead_bought and lead and (not self.is_sold_out(lead)) and cart_tool and order_tool
        outcome = self._swap_in_lead(lead, cart_tool, order_tool) if ready else None
        if outcome is None:
            return ''
        sent_lines, answers = outcome
        return 'Not run (Corvin acted instead): the independent check approved ' + lead + ', so Corvin set up the cart (' + '; '.join(sent_lines) + ') and went to checkout. What the store said: ' + ' | next: '.join(answers)
_ENDED_TAG = ' [episode closed]'


class _ProposedCall:
    """One tool call the model proposed, carried through the screening steps.

    The screens fill in what they learn (parsed arguments, the call's fingerprint, the list argument of a
    side-by-side view, ...) and may append Corvin's side notes to `extra`, which ride along with the store's reply.
    """

    __slots__ = ('ident', 'joined', 'name', 'raw', 'siblings', 'args', 'many', 'target', 'ending', 'mark',
                 'readers', 'extra', 'reread', 'chain')

    def __init__(self, call: dict[str, Any], siblings: int) -> None:
        self.ident = call['id']
        self.joined = bool(call.get('joined'))
        spec = call.get('function') or {}
        self.name = spec.get('name')
        self.raw = spec.get('arguments')
        self.siblings = siblings
        self.args: dict[str, Any] = {}
        self.many: str | None = None
        self.target: str | None = None
        self.ending = False
        self.mark = ''
        self.readers: set[str] = set()
        self.extra = ''
        self.reread = ''
        self.chain = ''


class EpisodeTurns:
    # ------------------------------------------------------------------ the episode from start to end

    def play_episode(self) -> list[dict[str, Any]]:
        """Run the whole episode (unless the store already reports it closed) and hand back the event log."""
        if not self.episode_over:
            self._play_from_request()
        return self.event_log

    def _play_from_request(self) -> None:
        self.parse_request()
        self.event_log.append({'kind': 'brief', 'brief': self.parsed_brief})
        self.chat_history = [{'role': 'system', 'content': LOOP_SYSTEM_PROMPT},
                             {'role': 'user', 'content': self._opening_message()}]
        self.drive_turns(MAX_LOOP_TURNS, HARD_COST_LIMIT)
        if not self.episode_over:
            self._last_chance_finish()
        self.event_log.append({'kind': 'end', 'finished': self.episode_over, 'steps': self.steps_used,
                               'cost': round(self.model_spend, 5), 'calls': self.model_requests})

    def _opening_message(self) -> str:
        first_view = self.world_view.get('opening')
        if first_view is not None:
            self.absorb_records(first_view)
        sections = ["THE SHOPPER'S REQUEST:\n" + self.request_text]
        if first_view is not None:
            sections.append('WHAT THE STORE SHOWED AT THE START:\n' + self.render_result(first_view))
        sections.append("Begin by searching each item's category and opening several listings in full, so the "
                        'interviewer learns which properties these listings have.')
        sections.append(self.status_text())
        return '\n\n'.join(sections)

    def _last_chance_finish(self) -> None:
        """The model stopped without closing: Corvin closes, and on a refusal hands the model a few more turns."""
        refusal = self.wrap_up_episode(report_refusal=True)
        if refusal and not self.episode_over and not self.out_of_funds and not self.credit_exhausted \
                and self.steps_remaining() > 1:
            self.chat_history.append({'role': 'user', 'content': '[Corvin] Corvin tried to close the episode, but '
                                      + refusal + '\nFollow what the store says, then finish.\n' + self.status_text()})
            self.drive_turns(self.loop_turns + 10, HARD_COST_LIMIT + 0.01)
        if not self.episode_over:
            self.wrap_up_episode()

    # ------------------------------------------------------------------ the turn loop

    def _may_take_turn(self, turn_cap: int, spend_cap: float) -> bool:
        if self.episode_over or self.out_of_funds or self.steps_remaining() <= 0:
            return False
        if not (self.loop_turns < turn_cap and self.model_spend < spend_cap):
            return False
        if self.elapsed_seconds() < FINAL_TURN_AFTER:
            return True
        # past the last-turn mark only an empty cart with model time to spare keeps the loop going
        return not self.cart_listing_ids() and self.seconds_for_model() > EMPTY_CART_SECONDS

    def drive_turns(self, turn_cap: int, spend_cap: float) -> None:
        # streak = turns in a row without a tool call; None once the loop has to stop early
        streak: int | None = 0
        while streak is not None and self._may_take_turn(turn_cap, spend_cap):
            streak = self._take_turn(streak)

    def _take_turn(self, streak: int) -> int | None:
        if self.interview_due():
            outcome = self.run_interview()
            if outcome:
                self.chat_history.append({'role': 'user', 'content': outcome + '\n' + self.status_text()})
            return streak
        self.compact_history()
        self.loop_turns += 1
        message = self.request_model(self.chat_history, [FAST_MODEL, BACKUP_MODEL], 3000, self.available_tools())
        if message is None:
            self.failed_turns += 1
            hopeless = self.credit_exhausted or self.model_requests >= MAX_MODEL_REQUESTS or self.failed_turns > 2
            return None if hopeless else streak
        proposed = [self._tidy_call(position, offered)
                    for position, offered in enumerate(message.get('tool_calls') or []) if isinstance(offered, dict)]
        said = self._message_text(message)
        if not proposed:
            self.chat_history.append({'role': 'assistant', 'content': said or '(no tool call)'})
            self.chat_history.append({'role': 'user', 'content': 'Continue by calling one of the tools.\n'
                                      + self.status_text()})
            return streak + 1 if streak + 1 < 3 else None
        (self._defer_for_interview if self._interview_preempts(proposed) else self._play_calls)(said, proposed)
        return 0

    def _tidy_call(self, position: int, offered: dict[str, Any]) -> dict[str, Any]:
        """One model tool call in a clean shape: string id, string name, a JSON object as arguments."""
        body = offered.get('function') if isinstance(offered.get('function'), dict) else {}
        given = body.get('arguments')
        decoded = given if isinstance(given, dict) else (
            _parse_json_object(given) if isinstance(given, str) and given.strip() else {})
        return {'id': str(offered.get('id') or f'k{self.loop_turns}_{position}'), 'type': 'function',
                'function': {'name': str(body.get('name') or ''),
                             'arguments': _to_json(decoded if isinstance(decoded, dict) else {})}}

    @staticmethod
    def _message_text(message: dict[str, Any]) -> str:
        words = message.get('content')
        if isinstance(words, str):
            return words
        return _to_json(words) if words else ''

    def _interview_preempts(self, proposed: list[dict[str, Any]]) -> bool:
        """True when the model wants to commit (cart, checkout, ending) before the interviewer has had its turn."""
        binding = {self.tool_roles.get('cart'), self.tool_roles.get('buy'), self.tool_roles.get('refuse')} - {None}
        return (not self.interview_done and self.tool_roles.get('ask') and self.spare_messages() > 0
                and any(c['function']['name'] in binding for c in proposed)
                and self.steps_remaining() > 8 and self.elapsed_seconds() <= INTERVIEW_DEADLINE)

    def _defer_for_interview(self, said: str, proposed: list[dict[str, Any]]) -> None:
        self.chat_history.append({'role': 'assistant', 'content': said or None, 'tool_calls': proposed})
        for entry in proposed:
            self.chat_history.append({'role': 'tool', 'tool_call_id': entry['id'],
                                      'content': 'Not run (no step used): the shopper gets asked first.'})
        outcome = self.run_interview()
        self.chat_history.append({'role': 'user', 'content': (outcome or 'The interviewer found nothing to ask.')
                                  + '\n' + self.status_text()})

    def _play_calls(self, said: str, proposed: list[dict[str, Any]]) -> None:
        self.chat_history.append({'role': 'assistant', 'content': said or None, 'tool_calls': proposed})
        self.event_log.append({'kind': 'turn', 'text': said[:300], 'calls': [c['function'] for c in proposed]})
        answers = self.dispatch_calls(proposed)
        final = len(answers) - 1
        for position, (call_id, reply_text) in enumerate(answers):
            if position == final:
                reply_text += '\n' + self.status_text()
            self.chat_history.append({'role': 'tool', 'tool_call_id': call_id, 'content': reply_text})

    def compact_history(self) -> None:
        """Past the size limit, cut the older half of the tool results down to a head plus the listings they named."""
        if sum(len(_to_json(m.get('content'))) for m in self.chat_history) <= HISTORY_CHAR_CAP:
            return
        results = [m for m in self.chat_history if m.get('role') == 'tool']
        for entry in results[:max(1, len(results) // 2)]:
            body = entry.get('content') or ''
            if len(body) > 600:
                entry['content'] = self._shortened_result(body)

    def _shortened_result(self, body: str) -> str:
        quoted = re.findall('"([A-Za-z0-9][A-Za-z0-9_\\-:.]{5,40})"', body)
        named = [i for i in dict.fromkeys(quoted) if i in self.listing_rows][:20]
        return body[:400] + f"...(earlier result trimmed; listings it held: {', '.join(named)})"

    # ------------------------------------------------------------------ running one batch of proposed calls

    def dispatch_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        """Screen and run the proposed calls in order; every call gets exactly one reply text."""
        batch = self.fold_detail_opens(calls)
        return [(entry['id'], self._settle_proposed(_ProposedCall(entry, len(batch)))) for entry in batch]

    def _settle_proposed(self, job: _ProposedCall) -> str:
        held = self._run_screens(job, (
            self._screen_admission,
            self._screen_arguments,
            self._screen_compare_room,
            self._screen_gone_listing,
            self._screen_phantom_removal,
            self._screen_finish_timing,
            self._screen_repeats,
            self._screen_stale_compare,
            self._screen_finishing_holds,
            self._screen_late_decline,
            self._screen_untried_searches,
            self._screen_confirmed_lead,
            self._screen_late_question,
            self._screen_process_gate,
        ))
        return held if held is not None else self._send_and_report(job)

    @staticmethod
    def _run_screens(job: _ProposedCall, screens: tuple) -> str | None:
        """Apply the screens in order; the first one that answers stops the call (None lets it through)."""
        for screen in screens:
            answer = screen(job)
            if answer is not None:
                return answer
        return None

    def _cart_tail(self) -> str:
        return '\nCart at present: ' + self.cart_summary()

    # --- screens that need neither a step nor a model call

    def _screen_admission(self, job: _ProposedCall) -> str | None:
        if job.joined:
            return 'See the side-by-side result above: Corvin fetched these listings together as a single step.'
        if self.episode_over:
            return 'Not run: the episode has ended.'
        if job.name == self.tool_roles.get('ask'):
            return "Not run (no step used): every message to the shopper goes through Corvin's interviewer."
        offered = sorted({spec['function']['name'] for spec in self.available_tools()})
        if job.name not in self.tool_schemas:
            return f"Not run (no step used): no such tool. Tools: {', '.join(offered)}."
        if job.name not in offered:
            return 'Not run (no step used): not enough steps remain for that; use: ' + ', '.join(offered) + '.'
        return None

    def _screen_arguments(self, job: _ProposedCall) -> str | None:
        parsed = job.raw if isinstance(job.raw, dict) else (
            _parse_json_object(job.raw) if (job.raw or '').strip() else {})
        if parsed is None:
            return 'Not run (no step used): the arguments did not form a single JSON object.'
        problems: list[str] = []
        shaped = _conform_to_schema(parsed, self.tool_schemas[job.name], 'args', problems)
        if problems:
            return 'Not run (no step used): ' + '; '.join(problems[:6])
        shaped, complaint = self.resolve_listing_ids(job.name, shaped)
        if complaint:
            return 'Not run (no step used): ' + complaint + '.'
        if job.name == self.tool_roles.get('cart'):
            shaped = self.full_removal_args(job.name, shaped)
        job.args = shaped
        return None

    def _screen_compare_room(self, job: _ProposedCall) -> None:
        """A side-by-side call longer than the store is known to show gets cut to the room it has."""
        job.many = self.compare_list_param() if job.name == self.tool_roles.get('compare') else None
        listed = job.args.get(job.many) if job.many else None
        if isinstance(listed, list) and len(listed) > self.compare_capacity() and self.compare_fail_size:
            room = self.compare_capacity()
            job.extra = (f"\n[Corvin] Only the first {room} listings went out: the store's side-by-side view holds at "
                         f"most {room}. Still to show: {', '.join(map(str, listed[room:]))}.")
            job.args = {**job.args, job.many: listed[:room]}
        return None

    def _screen_gone_listing(self, job: _ProposedCall) -> str | None:
        job.target = self.single_id_param(job.name)
        if not job.target or job.name not in (self.tool_roles.get('open'), self.tool_roles.get('offer')):
            return None
        wanted = str(job.args.get(job.target))
        if wanted not in self.sold_out_at:
            return None
        return (f'Not run (no step used): {job.args[job.target]} went out of stock at step {self.sold_out_at[wanted]} '
                'and stays gone. Swap its cart line for a different listing that fits.')

    def _screen_phantom_removal(self, job: _ProposedCall) -> str | None:
        """A negative cart count for a listing the cart does not hold would only waste a step."""
        if job.name != self.tool_roles.get('cart'):
            return None
        if not any((_as_int(v) or 0) < 0 for v in job.args.values() if not isinstance(v, str)):
            return None
        snapshot = _to_json(self.cart_snapshot) if self.cart_viewed_step >= self.cart_changed_step else ''
        absent = [v for v in job.args.values()
                  if isinstance(v, str) and v in self.listing_rows and v not in self.cart_units and v not in snapshot
                  and v not in self.last_cart_reply]
        if not absent:
            return None
        return (f'Not run (no step used): {absent[0]} is not a cart line, so there is nothing to remove. '
                'Cart at present: ' + self.cart_summary())

    def _screen_finish_timing(self, job: _ProposedCall) -> str | None:
        job.ending = self.is_checkout_call(job.name, job.args) or self.is_decline_call(job.name, job.args)
        if job.ending and job.siblings > 1:
            return 'Not run (no step used): read your other results first, then send the finishing call by itself.'
        if job.ending or self.steps_remaining() > self.closing_steps():
            return None
        if self.tool_roles.get('buy') or self.tool_roles.get('refuse'):
            return ("Not run (no step used): the final steps are reserved for the checkout (after the shopper approves "
                    'the basket, when that is required) or for ending with no purchase.')
        return None

    def _screen_repeats(self, job: _ProposedCall) -> str | None:
        job.mark = job.name + _to_json(job.args)
        roles = self.tool_roles
        if not job.ending and job.name != roles.get('cart') and job.mark in self.recent_calls:
            return 'Not run (no step used): this exact call was just made and nothing has changed; see its result above.'
        job.readers = {roles.get(r) for r in ('search', 'open', 'offer', 'compare')} - {None}
        if job.name not in job.readers:
            return None
        cached = self.read_results_cache.get(job.mark)
        if cached and cached[0] == self.state_version:
            return (f'Not run (no step used): an identical call at step {cached[1]} returned this, and nothing has '
                    'changed since:\n' + cached[2][:3000])
        if job.target and job.name != roles.get('compare') \
                and self.view_unchanged(job.name, str(job.args.get(job.target))):
            return (f'Not run (no step used): this view has already shown all it can for {job.args[job.target]} since '
                    'the last change. Its stored record: ' + self.record_text(str(job.args[job.target])))
        return None

    def _screen_stale_compare(self, job: _ProposedCall) -> str | None:
        """Drop the listings a side-by-side view has nothing new to say about; refuse the call when none is left."""
        listed = job.args.get(job.many) if job.many else None
        if not isinstance(listed, list) or not listed:
            return None
        fresh = [i for i in listed if not self.view_unchanged(job.name, str(i))]
        if not fresh:
            return ('Not run (no step used): this view has already shown all it can for these listings since the '
                    'last change. Stored records:\n'
                    + '\n'.join(f'{i}: {self.record_text(str(i))}' for i in listed)[:4000])
        if len(fresh) < len(listed):
            dropped = ', '.join(str(i) for i in listed if i not in fresh)
            job.extra += f'\n[Corvin] Dropped {dropped}: already fully shown, and nothing has changed since.'
            job.args = {**job.args, job.many: fresh}
            job.mark = job.name + _to_json(job.args)
        return None

    # --- holds on a finishing call (these may spend steps and model calls)

    def _screen_finishing_holds(self, job: _ProposedCall) -> str | None:
        if not job.ending or self.steps_remaining() <= 2:
            return None
        if self.is_checkout_call(job.name, job.args):
            return self._run_screens(job, (
                self._hold_wider_search,
                self._hold_cart_glance,
                self._hold_gone_lines,
                self._hold_pointer_trail,
                self._hold_part_fit,
                self._hold_line_counts,
                self._hold_cart_review,
                self._hold_stale_offers,
            ))
        if self.steps_remaining() > 5:
            return self._run_screens(job, (self._hold_decline_review, self._hold_review_leads, self._hold_confirm_lead))
        return None

    def _hold_wider_search(self, job: _ProposedCall) -> str | None:
        wider = self.broader_search_hold()
        return wider + self._cart_tail() if wider else None

    def _hold_cart_glance(self, job: _ProposedCall) -> str | None:
        """Refresh the carted offers, and look at the cart itself when it changed since the last look."""
        job.reread = self.refresh_cart_offers()
        if self.note_reports_sellout:
            self.locate_sold_line()
        if not (self.cart_modified and self.cart_viewed_step < self.cart_changed_step
                and self.tool_roles.get('cart_view') and self.steps_remaining() > 3):
            return None
        viewer = self.tool_roles['cart_view']
        problems: list[str] = []
        view_args = _conform_to_schema({}, self.tool_schemas.get(viewer), 'args', problems)
        glance = None if problems else self.store_call(viewer, view_args)
        if glance is None:
            return None
        self.cart_snapshot = glance.get('observation')
        self.cart_viewed_step = self.steps_used
        change = self.take_observation(viewer, self.cart_snapshot)
        if not change:
            return None
        self.queued_notes = []
        return ('Not run yet (no step used). A look at the cart revealed a change: ' + change + self._cart_tail()
                + '\nAct on the change, correct the cart if needed, then check out.')

    def _hold_gone_lines(self, job: _ProposedCall) -> str | None:
        advice = self.sold_out_advice()
        gone = [i for i in self.cart_listing_ids() if i in self.sold_out_at] if self.sold_hold_count < 2 else []
        if not advice and not gone:
            return None
        if not advice:
            self.sold_hold_count += 1
        return ('Not run yet (no step used). '
                + (advice or f"{', '.join(gone)} in the cart has sold out: swap that line out first.") + self._cart_tail())

    def _hold_pointer_trail(self, job: _ProposedCall) -> str | None:
        job.chain, blocking = self.follow_pointer_chains()
        return 'Not run yet (no step used). ' + job.chain + self._cart_tail() if blocking else None

    def _hold_part_fit(self, job: _ProposedCall) -> str | None:
        if self.fit_hold_count >= 2 or self.steps_remaining() <= self.closing_steps() + 3:
            return None
        clashes = self.compatibility_problems()
        if not clashes:
            return None
        self.fit_hold_count += 1
        self.event_log.append({'kind': 'fit_hold', 'faults': clashes[:4]})
        return ('Not run yet (no step used). A carted part is not compatible with the carted device: '
                + '; '.join(clashes[:3]) + '. Replace the part (or the device), then check out.' + self._cart_tail())

    def _hold_line_counts(self, job: _ProposedCall) -> str | None:
        """Lines that differ from the request: unit counts Corvin fixes itself, anything else goes back to the model."""
        if self.line_hold_count >= 2 or self.steps_remaining() <= self.closing_steps() + 2:
            return None
        problems = self.cart_line_problems()
        if problems and all(' is held at ' in p for p in problems):
            adjusted = self.apply_quantity_fixes(self.quantity_corrections())
            if adjusted:
                problems = self.cart_line_problems()
                if not problems:
                    self.line_hold_count += 1
                    return ("Not run yet (no step used). Corvin adjusted the cart's quantities on its own: "
                            + '; '.join(adjusted) + '. Correct any line the store rejected; then check out.'
                            + self._cart_tail())
        if not problems:
            return None
        self.line_hold_count += 1
        self.event_log.append({'kind': 'line_hold', 'faults': problems[:6]})
        return ('Not run yet (no step used). The cart still differs from the request: ' + '; '.join(problems[:6])
                + ". Correct those lines (using the cart tool's own way of adding or removing units), then check out."
                + self._cart_tail())

    def _hold_cart_review(self, job: _ProposedCall) -> str | None:
        carted = self.cart_listing_ids()
        opened = self.open_for_attributes(carted, self.closing_steps() + (1 if len(carted) >= 2 else 3))
        self.follow_part_links(carted, self.closing_steps() + 1)
        opened += self.open_cheaper_rows()
        verdict = self.review_cart()
        if not verdict:
            return None
        shown = ("For the check Corvin opened these listings' attributes (one step apiece):\n" + '\n'.join(opened) + '\n'
                 if opened else '')
        return 'Not run yet. ' + job.chain + shown + verdict + self._cart_tail()

    def _hold_stale_offers(self, job: _ProposedCall) -> str | None:
        if job.reread and self.model_spend > SOFT_COST_LIMIT:
            return ("Not run yet (no step used). Corvin has just re-read the listings in the cart:\n" + job.reread
                    + '\nCheck out again if each line still shows stock and a price within the budget; if not, replace '
                    'that line first.')
        return None

    def _hold_decline_review(self, job: _ProposedCall) -> str | None:
        review = self.review_decline()
        return 'Not run yet (no step used). ' + review if review else None

    def _hold_review_leads(self, job: _ProposedCall) -> str | None:
        if not self.review_leads or self.steps_remaining() <= 4 or self.decline_block_count != 0:
            return None
        self.decline_block_count = 1
        self.event_log.append({'kind': 'refuse_blocked', 'leads': list(self.review_leads)})
        return ('Not run yet (no step used). The review turned up listings that look like a fit: '
                + ', '.join(self.review_leads) + '. Open them if you need to, cart the best fitting one (following the '
                'pick rule) and check out. End with no purchase only if every one of them clearly breaks a requirement; '
                'then say in your text which one it is and repeat the ending call.')

    def _hold_confirm_lead(self, job: _ProposedCall) -> str | None:
        if not self.review_leads or self.steps_remaining() <= 4 or self.decline_block_count != 1:
            return None
        self.decline_block_count = 2
        lead = self.review_leads[0]
        self.review_cart([lead], HARD_COST_LIMIT)
        if not (self.review_verdicts.get(self.review_signature([lead])) and lead in self.review_leads):
            return None
        self.confirmed_lead = lead
        self.event_log.append({'kind': 'refuse_blocked', 'confirmed': lead})
        return (f"Not run yet (no step used). A separate check of {lead}'s records finds that it satisfies every "
                'requirement. Cart it at the requested quantity and check out; end with no purchase only if its record '
                'plainly violates a requirement.')

    # --- last screens before the store call

    def _screen_late_decline(self, job: _ProposedCall) -> str | None:
        """Near the end, an ending without purchase while fitting lines were already built is sent back once."""
        if not self.is_decline_call(job.name, job.args) or self.steps_remaining() > 5 or self.late_decline_blocked:
            return None
        built = [i for i in self.built_lines if not self.is_sold_out(i)]
        fresh = self._cart_view_is_fresh()
        missing = [i for i in built if i not in self.cart_listing_ids() or (i in self.withdrawn and not fresh)]
        if len(built) < len(self.wanted_products) or len(missing) + 1 > self.steps_remaining():
            return None
        self.late_decline_blocked = True
        self.event_log.append({'kind': 'late_decline_held', 'lines': built, 'missing': missing})
        redo = (f"Return {', '.join(missing)} to the cart at the requested quantities, then check out." if missing
                else 'Check out the cart as it stands.')
        return ('Not run yet (no step used). Ending with no purchase scores nothing when a fitting listing exists, and '
                'being low on steps is no proof that none exists. ' + redo + ' End with no purchase only if your '
                'search found a requested item with no listing that meets its requirements; then repeat the ending '
                'call.')

    def _screen_untried_searches(self, job: _ProposedCall) -> str | None:
        if not self.is_decline_call(job.name, job.args) or self.review_leads or self.decline_pushed:
            return None
        if self.steps_remaining() <= self.step_limit // 3:
            return None
        self.decline_pushed = True
        return (f'Not run yet (no step used). {self.steps_remaining()} steps remain, so before ending try searches you '
                "have not used yet: browse each requested item's category and page on through it using the token "
                'or offset the store hands back, try one-word value queries and other names for the property, and open '
                'the listings that could fit. Only then end with no purchase.')

    def _screen_confirmed_lead(self, job: _ProposedCall) -> str | None:
        if not self.is_decline_call(job.name, job.args):
            return None
        bought = self.buy_confirmed_lead()
        if bought:
            return bought + (_ENDED_TAG if self.episode_over else '')
        job.args = self.silent_decline_args(job.name, job.args)
        return None

    def _screen_late_question(self, job: _ProposedCall) -> str | None:
        if not self.is_checkout_call(job.name, job.args):
            return None
        question = self.ask_before_checkout()
        return 'Not run yet. ' + question + self._cart_tail() if question else None

    def _screen_process_gate(self, job: _ProposedCall) -> str | None:
        if not job.ending or self.steps_remaining() < 2:
            return None
        cleared, remark = self.run_process_steps('order' if self.is_checkout_call(job.name, job.args) else 'end')
        if not cleared:
            return 'Not run yet. ' + remark + self._cart_tail()
        if remark:
            job.extra = '\n[Corvin] ' + remark + job.extra
        return None

    # --- the store call and what Corvin learns from its reply

    def _send_and_report(self, job: _ProposedCall) -> str:
        reply = self.store_call(job.name, job.args)
        if reply is None:
            return 'The call fell through (network or server). You can retry it.'
        seen = reply.get('observation')
        self._learn_compare_width(job, seen)
        self._note_call_effects(job, seen)
        not_browsing = {tool for role, tool in self.tool_roles.items() if role != 'search'}
        browsing = job.name not in not_browsing and not job.ending
        paged = self.read_more_pages(job.name, job.args, seen) if browsing else ''
        self._learn_single_view(job.name, seen)
        self._book_store_state(job, reply, seen)
        return self._compose_reply(job, reply, seen, paged, browsing)

    def _learn_compare_width(self, job: _ProposedCall, seen: Any) -> None:
        listed = job.args.get(job.many) if job.many else None
        if not isinstance(listed, list) or len(listed) <= 1:
            return
        shown = _to_json(seen)
        if any(str(i) in shown for i in listed):
            self.compare_ok_size = max(self.compare_ok_size, len(listed))
            return
        self.compare_fail_size = min(self.compare_fail_size or 999, len(listed))
        job.extra += (f'\n[Corvin] The side-by-side view came back empty for {len(listed)} listings: send no more than '
                      f'{self.compare_capacity()} per call.')

    def _names_only_listings(self, value: Any) -> bool:
        group = value if isinstance(value, list) else [value]
        return bool(group) and all(isinstance(x, str) and x in self.listing_rows for x in group)

    def _note_call_effects(self, job: _ProposedCall, seen: Any) -> None:
        self.recent_calls = (self.recent_calls + [job.mark])[-3:]
        self.id_params.update(((job.name, key) for key, value in job.args.items() if self._names_only_listings(value)))
        self.absorb_records(seen)
        if job.name == self.tool_roles.get('compare'):
            rows = self.listing_records(seen)
            if rows:
                self.shows_attributes[job.name] = (self.shows_attributes.get(job.name, False)
                                                   or any(_attribute_block(r) for _, r in rows))
        if job.name in job.readers - {self.tool_roles.get('search')}:
            self.remember_view_keys(job.name, seen)

    def _learn_single_view(self, name: str, seen: Any) -> None:
        """A one-listing view tells whether it carries attributes; that can move the open and offer roles."""
        if not self.single_id_param(name):
            return
        roles = self.tool_roles
        if name in (roles.get('cart'), roles.get('buy'), roles.get('refuse'), roles.get('ask')):
            return
        listing = self.listing_id_of(seen) if isinstance(seen, dict) else None
        if not listing:
            return
        has_facts = self.shows_attributes.get(name, False) or bool(_attribute_block(seen))
        self.shows_attributes[name] = has_facts
        if name == roles.get('open') and not has_facts:
            skip = (roles.get('cart'), roles.get('buy'), roles.get('refuse'), roles.get('ask'), roles.get('search'))
            substitute = next((tool for tool in self.tool_schemas
                               if tool != name and self.single_id_param(tool)
                               and self.shows_attributes.get(tool) is not False and tool not in skip), None)
            if substitute:
                roles['open'] = substitute
        if not has_facts and name != roles.get('open') and not roles.get('offer'):
            roles['offer'] = name
        if name == self.price_stock_tool():
            self.offer_read_step[listing] = self.steps_used

    def _book_store_state(self, job: _ProposedCall, reply: dict[str, Any], seen: Any) -> None:
        roles = self.tool_roles
        if job.name == roles.get('cart'):
            self._book_cart_reply(job.name, job.args, seen)
        if job.name == roles.get('search') and self.listing_records(seen):
            self.recent_searches = (self.recent_searches + [dict(job.args)])[-8:]
        if job.name == roles.get('cart_view'):
            self.cart_snapshot = seen
            self.cart_viewed_step = self.steps_used
        if self.is_checkout_call(job.name, job.args):
            self.checkout_attempts += 1
            if not reply.get('finished'):
                self.checkout_refused_step = self.steps_used

    def _book_cart_reply(self, name: str, args: dict[str, Any], seen: Any) -> None:
        self.record_cart_call(name, args)
        self.last_cart_reply = _to_json(seen)
        self.recent_calls = []
        numbers = [_as_int(v) for v in args.values() if not isinstance(v, str) and _as_int(v) is not None]
        named = [v for v in args.values() if isinstance(v, str) and v in self.listing_rows] if numbers else []
        for ident in named:
            # a line taken to zero that the reply no longer lists counts as withdrawn; adding units brings it back
            if numbers[0] > 0:
                self.withdrawn.discard(ident)
            elif not _id_in_rows(seen, ident):
                self.withdrawn.add(ident)
        standing = [i for i in self.cart_listing_ids() if not self.is_sold_out(i) and i not in self.withdrawn]
        if len(standing) >= len(self.built_lines):
            self.built_lines = standing

    def _compose_reply(self, job: _ProposedCall, reply: dict[str, Any], seen: Any, paged: str, browsing: bool) -> str:
        text = self.render_result(seen) + paged + job.extra
        if browsing:
            text += self.cheapest_rows_note([seen] + self.paged_results) + self.price_cap_note(job.name, job.args, seen)
        cacheable = text
        proof = self.evidence_text(seen)
        if proof:
            text += '\n' + proof
        text += self._zero_count_hint(job, seen)
        self.note_reports_sellout = False
        text += self._store_note_text(seen)
        text += self._refusal_recheck(job, reply, seen)
        self.note_reports_sellout = False
        advice = self.sold_out_advice()
        if advice:
            text += '\n' + advice
        if self.queued_notes:
            text += "\n[notes from Corvin's own calls] " + ' | '.join(self.queued_notes)
            self.queued_notes = []
        if job.name in job.readers:
            self.read_results_cache[job.mark] = (self.state_version, self.steps_used, cacheable)
        if reply.get('finished'):
            text += _ENDED_TAG
        return text

    def _zero_count_hint(self, job: _ProposedCall, seen: Any) -> str:
        stuck = (job.name == self.tool_roles.get('cart')
                 and any(_as_int(v) == 0 for v in job.args.values() if not isinstance(v, str))
                 and any(isinstance(v, str) and v in self.listing_rows and _id_in_rows(seen, v)
                         for v in job.args.values()))
        return ('\n[Corvin] Sending a count of 0 kept that line in the cart: this tool adds the given count to the '
                "line's units, so use a negative count to remove units.") if stuck else ''

    def _store_note_text(self, seen: Any) -> str:
        loose = self.free_text_lines(seen)
        meaning = self.read_store_note(loose) if loose else ''
        return '\n[store note] ' + meaning + '. Update your picks for this change before checking out.' if meaning else ''

    def _refusal_recheck(self, job: _ProposedCall, reply: dict[str, Any], seen: Any) -> str:
        """After a refused checkout or a sell-out note, re-read the carted lines to find the one that went."""
        refused = self.is_checkout_call(job.name, job.args) and not reply.get('finished')
        repeated = refused and any(line in self.reply_words(seen) for line in self.sellout_note_lines)
        unnamed = self.note_reports_sellout and (
            refused or (not self.note_named_listing and not self.is_checkout_call(job.name, job.args)))
        wanted = (unnamed or repeated) and self.cart_listing_ids() and not self.episode_over
        found = self.locate_sold_line() if wanted else ''
        return "\n[Corvin re-checked the carted lines after the refusal]\n" + found if found else ''

    # ------------------------------------------------------------------ opening several listings as one view

    def fold_detail_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Several single-listing opens in one batch become one side-by-side call when that view shows attributes."""
        plan = self._fold_plan(calls)
        return calls if plan is None else self._apply_fold(calls, *plan)

    def _fold_plan(self, calls: list[dict[str, Any]]) -> tuple[str, str, list[tuple[int, str]]] | None:
        roles = self.tool_roles
        opener, joiner = roles.get('open'), roles.get('compare')
        id_field = self.single_id_param(roles.get('open'))
        list_field = self.compare_list_param()
        usable = opener and joiner and id_field and list_field and opener != joiner
        if not usable or self.compare_lacks_attributes():
            return None
        picks = self._foldable_opens(calls, opener, id_field)[:self.compare_capacity()]
        return (joiner, list_field, picks) if len(picks) >= 2 else None

    def _apply_fold(self, calls: list[dict[str, Any]], joiner: str, list_field: str,
                    picks: list[tuple[int, str]]) -> list[dict[str, Any]]:
        folded = [ident for _, ident in picks]
        merged = list(calls)
        for position, _ in picks:
            merged[position] = {'id': calls[position]['id'], 'joined': True}
        lead = picks[0][0]
        # the first folded open carries the side-by-side call; the others only point at its result
        merged[lead] = {'id': calls[lead]['id'], 'type': 'function',
                        'function': {'name': joiner, 'arguments': _to_json({list_field: folded})}}
        self.event_log.append({'kind': 'joined', 'ids': list(folded)})
        return merged

    def _foldable_opens(self, calls: list[dict[str, Any]], opener: str, id_field: str) -> list[tuple[int, str]]:
        found: list[tuple[int, str]] = []
        for position, entry in enumerate(calls):
            body = entry.get('function') or {}
            given = body.get('arguments')
            parsed = _parse_json_object(given) if isinstance(given, str) else None
            ident = str((parsed or {}).get(id_field) or '').strip()
            bare_open = body.get('name') == opener and parsed and set(parsed) == {id_field}
            if bare_open and ident in self.listing_rows and ident not in self.sold_out_at:
                found.append((position, ident))
        return found

    def _comparison_schema(self) -> dict[str, Any]:
        return self.tool_schemas.get(self.tool_roles.get('compare') or '') or {}

    def compare_list_param(self) -> str | None:
        schema = self._comparison_schema()
        array_field = next((key for key, field in (schema.get('properties') or {}).items()
                            if isinstance(field, dict) and 'array' in _schema_types(field)), None)
        if array_field and all(key == array_field for key in schema.get('required') or []):
            return array_field
        return None

    def compare_capacity(self) -> int:
        """How many listings one side-by-side call may carry: the schema's cap, else what the store has shown."""
        fields = self._comparison_schema().get('properties') or {}
        limit = (fields.get(self.compare_list_param() or '') or {}).get('maxItems')
        if isinstance(limit, int):
            return limit
        if not self.compare_fail_size:
            return max(4, self.compare_ok_size)
        return max(2, min(self.compare_fail_size - 1, max(self.compare_ok_size, 2)))

    def render_result(self, observation: Any) -> str:
        rendered = _to_json(_truncate_strings(observation, FIELD_CHAR_CAP))
        if len(rendered) <= RESULT_CHAR_CAP:
            return rendered
        return rendered[:RESULT_CHAR_CAP] + '...(clipped)'

    # ------------------------------------------------------------------ Corvin's own tally of the cart

    def record_cart_call(self, name: str, args: dict[str, Any]) -> None:
        """Bring the unit tally in line with a cart call the store just carried out."""
        self.cart_modified = True
        self.cart_changed_step = self.steps_used
        fields = self.tool_schemas.get(name, {}).get('properties') or {}
        takes_off = any([self._names_later_choice(fields.get(key), value) for key, value in args.items()])
        default_units = next((_as_int(v) for v in args.values() if _as_int(v) is not None and not isinstance(v, str)),
                             None)
        touched, rebuilds = self._cart_lines_in_args(fields, args, default_units)
        if rebuilds:
            # a call that lists whole line objects describes the full cart
            self.cart_units = {ident: units for ident, units in touched if units and units > 0}
            return
        absolute = self.cart_sets_quantity(name)
        for ident, units in touched:
            held = self._units_after_call(fields, self.cart_units.get(ident, 0), units, takes_off, absolute)
            if held > 0:
                self.cart_units[ident] = held
            elif ident in self.cart_units:
                del self.cart_units[ident]

    def _units_after_call(self, fields: dict[str, Any], held: int, units: int | None, takes_off: bool,
                          absolute: Any) -> int:
        if takes_off:
            return self._units_left_after_removal(fields, held, units)
        step = 1 if units is None else units
        return step if absolute else held + step

    @staticmethod
    def _names_later_choice(field: Any, value: Any) -> bool:
        # an enum switch set to anything past its first option (add/remove style) means taking units off
        options = (field or {}).get('enum')
        return isinstance(options, list) and len(options) >= 2 and value in options[1:]

    def _cart_lines_in_args(self, fields: dict[str, Any], args: dict[str, Any],
                            default_units: int | None) -> tuple[list[tuple[str, int | None]], bool]:
        """(listing, units) for every listing the call names, and whether the call restates the whole cart."""
        found: list[tuple[str, int | None]] = []
        rebuilds = False
        for field_name, given in args.items():
            if isinstance(given, list):
                rebuilds = self._collect_list_lines(fields.get(field_name), given, default_units, found) or rebuilds
            elif isinstance(given, str) and given in self.listing_rows:
                found.append((given, default_units))
        return found, rebuilds

    def _collect_list_lines(self, field: Any, entries: list[Any], default_units: int | None,
                            found: list[tuple[str, int | None]]) -> bool:
        row_shape = (field or {}).get('items') or {}
        # an empty list of line objects still restates the cart (as empty)
        whole = not entries and isinstance(row_shape, dict) and (row_shape.get('type') == 'object' or 'properties' in row_shape)
        for entry in entries:
            if isinstance(entry, dict):
                whole = True
                ident = next((v for v in entry.values() if isinstance(v, str) and v in self.listing_rows), None)
                units = next((_as_int(v) for v in entry.values() if _as_int(v) is not None and not isinstance(v, str)), 1)
                if ident:
                    found.append((ident, units))
            elif isinstance(entry, str) and entry in self.listing_rows:
                found.append((entry, default_units))
        return whole

    @staticmethod
    def _units_left_after_removal(fields: dict[str, Any], held: int, units: int | None) -> int:
        if any('integer' in _schema_types(f) for f in fields.values() if isinstance(f, dict)):
            return held - (units or 1)
        return held - units if units else 0

    # ------------------------------------------------------------------ closing the episode without the model

    def _cart_view_is_fresh(self) -> bool:
        return self.cart_snapshot is not None and self.cart_viewed_step >= self.cart_changed_step

    def _ending_blocked(self) -> bool:
        return self.episode_over or self.steps_remaining() <= 0

    def _shaped_refusal_args(self, refuser: str, problems: list[str]) -> dict[str, Any]:
        quiet = self.silent_decline_args(refuser, dict(self.parsed_brief.get('refuse_args') or {}))
        return _conform_to_schema(quiet, self.tool_schemas.get(refuser), 'args', problems)

    def _refusal_args_valid(self) -> bool:
        refuser = self.tool_roles.get('refuse')
        if not refuser:
            return False
        problems: list[str] = []
        self._shaped_refusal_args(refuser, problems)
        return not problems

    def wrap_up_episode(self, report_refusal: bool = False) -> str:
        """Check out a held cart, or else end without a purchase. With report_refusal, a refused checkout comes back
        as text for the model; a refused no-purchase ending always does. Returns '' otherwise."""
        if self.episode_over or self.out_of_funds or self.steps_remaining() <= 0:
            return ''
        fresh = self._cart_view_is_fresh()
        holding = bool(self.cart_listing_ids()) if fresh else bool(self.cart_units)
        complete = True
        if holding and len(self.wanted_products) >= 2:
            complete = self._complete_multi_item_cart()
        if self._ending_blocked():
            return ''
        buyer, refuser = self.tool_roles.get('buy'), self.tool_roles.get('refuse')
        if complete and buyer and (holding or (self.cart_modified and not fresh and self.steps_remaining() >= 2)):
            outcome = self._forced_checkout(buyer, report_refusal)
            if outcome is not None:
                return outcome
        return self._forced_decline(refuser) if refuser else ''

    def _complete_multi_item_cart(self) -> Any:
        complete = self.fill_missing_lines() or not self._refusal_args_valid()
        if complete and len(self.cart_listing_ids()) >= 2:
            self.open_for_attributes(self.cart_listing_ids(), self.closing_steps())
            if self.steps_remaining() >= 3 and self.seconds_for_model() > 5:
                self.follow_part_links(self.cart_listing_ids(), self.closing_steps())
        if not complete:
            self.event_log.append({'kind': 'short_cart_refused', 'lines': self.cart_listing_ids()})
        return complete

    def _forced_checkout(self, buyer: str, report_refusal: bool) -> str | None:
        """None: carry on to the no-purchase ending; '' stop here; other text: the refusal to show the model."""
        problems: list[str] = []
        args = _conform_to_schema(dict(self.parsed_brief.get('buy_args') or {}), self.tool_schemas.get(buyer), 'args',
                                  problems)
        if problems:
            return None
        if self.steps_remaining() >= 2:
            self.run_process_steps('order')
        if self._ending_blocked():
            return ''
        reply = self.store_call(buyer, args)
        self.event_log.append({'kind': 'forced_buy', 'reply': _truncate_strings(reply, 200)})
        if self._ending_blocked():
            return ''
        if reply is None or not report_refusal:
            return None
        self.checkout_attempts += 1
        self.checkout_refused_step = self.steps_used
        self.take_observation(buyer, reply.get('observation'))
        return 'the checkout was turned down: ' + self.render_result(reply.get('observation'))

    def _forced_decline(self, refuser: str) -> str:
        problems: list[str] = []
        args = self._shaped_refusal_args(refuser, problems)
        if problems:
            return ''
        if self.steps_remaining() >= 2:
            self.run_process_steps('end')
        if self._ending_blocked():
            return ''
        reply = self.store_call(refuser, args)
        self.event_log.append({'kind': 'forced_refuse', 'reply': _truncate_strings(reply, 200)})
        if reply is None or self.episode_over:
            return ''
        self.take_observation(refuser, reply.get('observation'))
        return 'the no-purchase ending was turned down: ' + self.render_result(reply.get('observation'))


class ShopperEpisode(EpisodeCore, EpisodeRecords, EpisodeSearch, EpisodeQuestions, EpisodeChecks, EpisodeCart, EpisodeTurns):
    """A full shopping episode: links to the store and the model, listing records, searches, the shopper interview,
    the checks, cart handling and the turn loop."""


def _crash_record(error: BaseException) -> dict[str, Any]:
    return {'kind': 'crash', 'error': f'{type(error).__name__}: {error}'[:400]}


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = ShopperEpisode(problem_data)
    except Exception as setup_error:
        return [_crash_record(setup_error)]
    try:
        return episode.play_episode()
    except Exception as run_error:
        episode.event_log.append(_crash_record(run_error))
        try:
            if not episode.episode_over:
                episode.wrap_up_episode()
        except Exception as close_error:
            episode.event_log.append(_crash_record(close_error))
        return episode.event_log
__all__ = ['agent_main']
