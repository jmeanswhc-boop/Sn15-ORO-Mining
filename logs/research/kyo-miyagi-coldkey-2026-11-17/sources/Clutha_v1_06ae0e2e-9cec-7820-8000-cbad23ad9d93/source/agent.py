from __future__ import annotations

import json
import re
from typing import Any

from src.agent.proxy_client import ProxyClient

# Seats: who answers which kind of request.
DRIVER_SEAT = "anthropic/claude-haiku-5.5"        # the tool-calling conversation
FALLBACK_SEAT = "openai/gpt-6-luna"               # when a seat gives nothing
READING_SEAT = "anthropic/claude-sonnet-5.5"      # the one-off reading of the request
CHECK_SEAT = "anthropic/claude-sonnet-5.5"        # listing checks and decline reviews
LIGHT_SEAT = "anthropic/claude-haiku-5.5"         # shopper lines, question plans, checks once spending is high

# Episode limits. Seconds are counted, never read from a clock (see Meter).
RUN_SECONDS = 265.0
CLOSE_MARGIN_S = 70.0      # below this many seconds left the episode heads for its finish
STOP_MARGIN_S = 28.0       # below this the code finishes on its own
SOFT_SPEND_USD = 0.06      # past this spend the checks move to the lighter seat first
TEXT_CAP = 16000           # longest JSON text handed to a model
# Seconds an optional action needs before the hard stop (STOP_MARGIN_S), sized by what the action costs (a check
# is one slow-seat reply, a shopper reading one fast reply) rather than by the closing mark: the network allows
# 300 s per task, and long episodes otherwise lose their checks and their shopper reading near the end.
LISTEN_ROOM_S = STOP_MARGIN_S + 10     # read a shopper line
CHECK_ROOM_S = STOP_MARGIN_S + 25      # a listing check, a decline review, the pre-order and decline holds
ANSWER_ROOM_S = STOP_MARGIN_S + 20     # send an owed answer to the shopper
ASK_ROOM_S = STOP_MARGIN_S + 40        # plan and send questions
SEARCH_ROOM_S = STOP_MARGIN_S + 45     # holds that send the driver searching again
TURN_CAP = 80

# The counted clock: each call adds an estimate of its duration, per seat, from replies measured on 2026-10-27
# (arena race-5 run, 2,000+ calls): Haiku 0.62 s + tokens/209, Sonnet 2.35 s + tokens/147, GPT 2.84 s + tokens/78.
# A 10% margin covers slower hours. (A flat 1.7x charge pushed most long episodes past the closing marks.)
_TALLY_SCALE = 1.0
_REPLY_MARGIN = 1.1
_REPLY_PACE = {"anthropic/claude-haiku-5.5": (0.62, 209.0), "anthropic/claude-sonnet-5.5": (2.35, 147.0),
               "openai/gpt-6-luna": (2.84, 78.0)}
_REPLY_PACE_OTHER = (2.84, 78.0)       # an unlisted seat is charged like the slowest one measured
_REPLY_UNSIZED_S = 4.0
_STORE_TRIP_S = 0.5
_FAULT_S = 1.0


def _TEST_FREEZE_CLOCK():
    """Stops the counted clock (the test harness keeps runs off the time-dependent paths)."""
    global _TALLY_SCALE
    _TALLY_SCALE = 0.0


# ---------------------------------------------------------------------------------------------- prompts
DRIVER_BRIEFING = """Your role: a careful, thorough shopping assistant working for a single shopper in a simulated store, where the supplied tools are your only way to act. Scoring is all or nothing, 1 or 0. A 1 needs a completely correct finish: the item you order satisfies each requirement the shopper stated, each hidden need the shopper discloses when you ask, and each change the shopper announces along the way; quantity and budget are correct; when the request calls for the cheapest (or best), that is what you pick; and you obey every rule written in the request. Should no item in the store be able to satisfy all of it, the correct finish is to decline, in precisely the manner the request describes.

HOW THE STORE WORKS
- A tool call costs one step whether it succeeds, fails or returns an error, and you have a limited number of steps. Your own reasoning costs nothing; steps do.
- When no call relies on the outcome of another, one reply may hold several tool calls (opening the details of a few listings together, say); they are executed in sequence. An order or a decline always goes alone, never alongside other calls.
- Should the store offer a tool that returns full details for several listings in a single call, prefer it to opening listings individually, since it spends fewer steps.
- Each store has its own tool names, arguments and result layouts. Study every tool description and JSON schema closely, and pass argument names and allowed values exactly as defined there. Use your first results to work out which tool reveals which facts (price, stock, detailed properties).
- Early on, discover which category labels this store gives the requested kind of product (listings show them). If neighbouring labels exist, the correct one is the label matching the request's full wording; do your search and buying inside it.
- Search may have limits: only so many rows or pages shown, ordering by id instead of relevance or price, all words mandatory, filters that work in one mode only. Check the total the search reports against the rows actually visible to you, and keep narrowing (more precise words, category or price filters, further pages) until the entire relevant set has passed before you.

HIDDEN NEEDS AND THE SHOPPER
- Some of the shopper's needs are not in the written request; a need is revealed only when a message asks about its property. The shopper answers just the FIRST thing a message asks, so each message must cover exactly ONE property, phrased as "Which <property> would you like the <product> to have?": the property named the way the listings name it, the product given by the plain noun the request uses, and no example values. Do not give the shopper instructions, do not mention lists, labels, positions or rules, and do not ask two things in one message.
- If the request is for several different products, name in each question the product it concerns; leaving the product unnamed confuses the shopper. Distribute the questions across the products.
- When the shopper says a message was not understood, or wants it explained, simplified or rephrased, neither explain nor repeat it: most likely they do not care about that property. Move on and ask about another property (with the product named).
- If the reply is that they don't mind, have no preference, or that anything is fine, the property is free. If the reply gives a preference or a limit, treat it as a hard requirement from then on, in its exact wording (numeric limits included).
- Requirements consist solely of what the request states plus what the shopper reveals or changes. Do not invent extra ones from the story (a gender, an age group, a size, the person who will use it): those details are not requirements and no listing fails on them. If one could matter, it might be a hidden need, so ask about it.
- If a planner is present, it picks and sends your questions as soon as two or three listings of the kind requested have had their details opened (of every kind, if several different products are requested); so do this among your first moves. The shopper's answers then show up in the conversation together with their meaning: treat each revealed need as a hard requirement and run a new search that joins it with the stated values.
- Use messages to the shopper solely for questions (and for replying to a question the shopper put to you). Do not send a message reporting your findings, your purchase or your decline; that goes through the finishing tool.
- When the asking falls to you: first open two or three listings' details, then ask about properties hinted at by the story (whom it is for, its required look, where or how it gets used), and after that about properties the listings carry whose values vary between candidates. Do not ask about anything the request has already settled, and stay within the message budget.

PROCEDURAL RULES
- A request can lay down rules for the way you work: the number of listings to view or compare before you choose, how to decline, what to check, what a remark must contain, what to tell the shopper. They are graded just like requirements, so carry out each one to the letter and in full. A rule to view or open N listings means N fully fitting listings, every one opened with the details tool.
- Questions the shopper puts to you (such as a request to confirm a change) are answered on your behalf while messages remain; the change itself binds you.
- A mere wish of the shopper ("ideally", "would be nice", "if possible") is no requirement: let it only break ties among items that already meet everything, and never let it override a cheapest rule.
- When the request leaves open something only the shopper can tell you (no price limit, say, or two wishes that might not both be achievable, so the shopper has to say which one counts more), ask about it before making your choice.

WHEN THINGS CHANGE
- Things may shift as you work: a new budget (up or down), a requirement removed, swapped or added, a price rising, an item selling out. A single shopper line may hold several changes, and shopper lines may turn up inside any tool result (an additional field containing sentences) or come attached to an order attempt. Every change binds from the moment it appears: test your pick and backups against it again, and if the request wants the cheapest or best, redo the choice over the new set of acceptable items.
- Order attempts can be REFUSED. In that case some field of the result gives the reason (a refusal, complaints, objections or the like), even when the inner result seems successful (accepted: true, placed: true, executed: true). A successful order ends the episode, so as long as the episode goes on nothing has been bought, no matter what the result claims. After a refusal the episode goes on with the cart as it was. Read the refusal, apply each change it names, take out of the cart whatever is no longer acceptable (a sold-out item, for instance), put in the next verified fit, check stock and price once more, and then order again.
- Stock can also run out, and prices can move, without notice. For that reason re-check stock and price immediately before each order attempt; once an item's current price times quantity exceeds the current budget, it no longer fits.

JUDGING A LISTING (the score depends on this)
- A listing meets a requirement only if its own property for that requirement states the value. Key spellings differ (spaces, dots, underscores, prefixes), and one property can sit under several keys; every key naming that property counts. Matching is by meaning: a single element of a comma-separated list suffices, singular equals plural, close synonyms count, and case is ignored. A word that only shares its beginning does not count, nor does a broader or different class (for example girls is not women, plain unisex is not female).
- The title or description may serve as evidence only when the listing lacks any property key for that requirement. When the listing's own property gives a different value, it fails, however clearly the title or description states the value.
- A value found under some other, related property is not enough (a style requirement is not satisfied by a length field, nor the material by a lining material).
- Numbers: convert units, then compare numerically; the edge value is inside a limit; a range qualifies only if all of it meets the limit.
- Honour exclusions ("leave out anything with X") and closed lists ("A or B and nothing else").
- The listing's category is what satisfies the requested kind of product; do not require any property to repeat the category name. The item has to be of the requested kind: where the store assigns category labels, it has to be in the category whose label fits the request's full wording (a request for "A or B" fits a category named "A & B", not a narrower neighbour named "B"). If the request asks for things on hand, it has to be in stock, and price times quantity has to stay within the budget (equal counts as within).
- If a listing has options or variants (sizes, colours, versions) carrying their own ids, prices or stock, treat each option as a separate candidate: the option's own value is what meets a requirement, its own price and stock are what count, and the cart line has to identify the chosen option in the form the cart tool's schema requires.
- When a property's value lists several options and the required one is among them (a comma list), it simply matches; this is not mixed evidence. Evidence counts as mixed only if the listing's own keys for that property give different, incompatible values. Only in that case, and only if no cheapest or best rule settles it, favour a candidate with clean evidence.

CHEAPEST AND BEST
- If the request wants the cheapest, you must order the lowest-priced item among ALL the items meeting every requirement; finding a single fitting item does not settle it. Apply price filters where a tool has them, bring the price limit down until all remaining rows are visible, sort the visible candidates by price on your own, and check them from the lowest price upward: the answer is the first that fully fits once every cheaper one has been checked and rejected. Hold the following fitting ones as backups.
- If the request wants the best by some measure (highest, most, largest, longest, etc.), compare ALL fitting items on that measure in the same manner and take the extreme one. An item whose listing gives no value for the measure cannot win.

ENDING THE TASK
- Maintain a ranked list of verified fits that includes at least one backup, since your top pick could sell out at checkout.
- Prior to each order attempt make sure that the cart contains precisely the lines the request calls for (normally one: the item you chose, at the quantity required; one line for each item if several different items are requested) and nothing beyond them (take any other line out), and that since your latest cart change you have read the chosen item's stock and price again and found it in stock and within budget.
- A decline is seldom the correct finish: usually the store does have an item meeting every requirement, and declining wrongly scores 0. Only decline after searching thoroughly, and only if for each candidate you can point to a revealed need, a stated requirement, the budget or stock that the candidate's own records plainly fail. Uncertainty, an absent nice-to-have, or a requirement you merely assumed gives no grounds to decline; order the best verified fit in that case. Apply the request's rule for declining to the letter. If substitutes must not be offered according to the request, the decline remark may not name, describe or refer to any listing: no titles, no ids, no prices.
- Budget your steps: make the first order attempt while seven or more steps are still left, so there is room to deal with a refusal. Once 7 steps remain, only the cart, stock-check and finishing tools can still be used. Make your decision once and avoid swapping the cart back and forth: each cart change spends a step.
- Include at least one tool call in every reply. Keep its text brief: the requirements as they now stand, the verified fits along with price and stock, and what you will do next."""

READING_PROMPT = """Below are the request of the shopper and the tool list of the store; study both. Reply with ONE compact JSON object (no indentation) and no other text, containing these keys:

"product": the requested kind of product, worded as the request words it (name every one if the request asks for several different items).
"quantity": how many units of one listing should be ordered (integer; 1 unless the request asks for several). Anything normally sold as a single item, like a set or a pair of shoes, counts as quantity 1.
"items": if several DIFFERENT products are requested, one object for each: {"product": its short name as the request words it, "quantity": how many units of that product (integer)}; if just one kind of product is requested, an empty list.
"requirements": a list of strings, one for each firm requirement the request states on the product, in the form "property: constraint", carrying the exact value, limit, exclusion or closed list the shopper used. If several different products are requested, prefix each with the short name of the product it belongs to, as "<product> <property>: constraint". Leave out price, stock, quantity, budgets and mere wishes, and never turn the product kind or category into a requirement: any words belonging to the category's name (an occasion, a season or an audience inside the category name) are handled by the category rather than being a separate property requirement.
"price_cap": the highest total price allowed, as a number; null if the request states none.
"currency": the unit of price used in the request, or null.
"in_stock_only": true if the request wants only items that are really on hand or in stock.
"priority": "cheapest" solely if the request explicitly wants the cheapest or lowest-priced option that fits; "best" solely if it explicitly wants the best by some measure; "none" in every other case. Remarks such as money being tight or every penny counting do not ask for the cheapest; such remarks merely explain the price cap.
"priority_detail": with "best", the measure that "best" is judged by; else "".
"message_budget": how many messages the shopper lets you send, as an integer; null if the request does not say.
"step_budget": how many moves or steps the request permits, as an integer; null if not stated.
"rules": a list quoting, word for word, each procedural rule the request sets for the assistant (how to decline, how many listings to look at, what to tell or check, what to include, what to avoid). What the shopper says they will do themselves is story, not a rule.
"soft_wants": a list of what the shopper would like without strictly requiring it (worded as preferably, ideally, if possible, would be nice). These never go in "requirements".
"ask_first": a list of things only the shopper can tell the agent and that should be asked before choosing: e.g. the budget when the request gives no price limit, which of two wishes counts more if the request says both might not be possible (in that case also place both wishes in soft_wants rather than requirements), or anything the request says should be checked with the shopper. Empty if there are none.
Each sentence of the request has to end up somewhere: a requirement, a rule, the priority, a soft want, a budget, a hint, or plain story. If you are unsure whether a sentence is a rule, place it in "rules".
"decline_no_names": true if the request's declining rule forbids naming, offering or suggesting alternatives or substitutes; false if it asks you to name something; null if the request has no such rule.
"roles": an object giving, for each role, the tool's exact name as it appears in the list, or null if no tool suits that role:
  "search", "details", "view", "message", "cart", "cart_view", "order", "decline".
  "message" = the tool for messaging the shopper. "cart" = the tool for adding to or removing from the cart. "order" = the tool for buying or checking out. "decline" = the tool for ending the task without buying.
"""

LISTENER_PROMPT = """Your job is to help a shopping assistant make sense of the shopper's latest words. Reply with ONE JSON object and no other text, holding these keys:
"meaning": a single short sentence on the shopper's meaning.
"needs": a list of the product's new firm requirements, each written as "property: constraint" using the shopper's exact value, limit, exclusion or closed list.
"no_preference": a list of the properties the shopper says do not matter to them.
"dropped": a list of former requirements the shopper no longer needs.
"new_budget": a number giving the new highest total price, or null.
"new_quantity": if the shopper changes the number of units wanted, {"product": the product concerned ("" if just one kind is requested), "quantity": the new count as an integer}; else null.
"unavailable": true if the shopper says an item can't be bought anymore (gone, sold out, button disabled, out of stock); false otherwise.
"requirements_now": the full list of the product's firm requirements as they stand once everything the shopper has said up to now is taken in, assembled from the requirements up to now (shown below) plus whatever the shopper has said explicitly, each as "property: constraint". Do not add requirements drawn from the story or from your own interpretation of the request: the only values that count are values stated as requirements by the request or stated by the shopper in a reply. Retain each earlier requirement the shopper has not dropped or replaced; when a value was replaced, retain just the latest; add each newly revealed need. Leave out price, stock, quantity and the product category.
"unclear": true if the shopper says the assistant's message was not followed or understood, or wants it explained, simplified, reworded or repeated; false otherwise.
"reply": if the shopper asks the assistant something or wants a confirmation, a brief plain answer to send back that responds to it and states the change in substance (such as confirming the new value); else "". It is always "" when "unclear" is true: when the shopper wants the assistant's own question explained, there is nothing to answer.
If several different products are requested, start each requirement in "needs" and in "requirements_now" with its product's short name, as "<product> <property>: constraint" (shoppers often say "<property> of the <product kind>").
A single line may hold several changes; report all of them. When one requirement replaces another, the old one is dropped and the new one is a need. If the shopper says which of two wishes counts more, the preferred wish becomes a need. A stated budget from the shopper goes in new_budget. A need is never a mere wish ("ideally", "would be nice"). Take the shopper's words literally and charitably. Folksy, figurative or indirect wording that names a value signals that the shopper firmly wants it; count a value as excluded only if the shopper plainly asks to avoid it. Any number paired with a comparison (no more than, at most, at least, under, over) is a limit; keep its unit with it. A new budget takes the old one's place whether higher or lower, even if the shopper presents it as good news. If the shopper only says anything works, they have no preference, or they don't mind, the property that was asked about goes in no_preference; such an answer concerns only the property the assistant asked about: it never removes a stated requirement, a rule or a price limit, and it changes nothing when the assistant's message asked about no property."""

CHECKER_PROMPT = """Your role is the last check before a shopping assistant places an order. A score of 1 needs the ordered item to meet each requirement listed below. Base your judgement ONLY on the records shown, reading them by these rules:
- The item meets a requirement only if its own property for that requirement states the value. Key spellings differ (spaces, dots, underscores, prefixes), and one property can sit under several keys; every key naming the property counts. Matching is by meaning: a single element of a comma list suffices, singular/plural and close synonyms count, case is ignored. Neither a word that only looks similar nor a broader/different class counts.
- Title or description count only if the item lacks a property key for that requirement. When its own property gives something else, it fails, even if the title states the value.
- A value found under some other, related property does not count.
- Numbers: convert units, then compare; the edge is inside a limit; a range qualifies only if all of it meets the limit.
- Honour exclusions and closed lists.
- The item has to be the requested kind of product; where listings have a category label, its category must be the one whose label fits the request's full wording, not a narrower or neighbouring one.
- If the item's own property for a requirement gives a different value, that requirement fails, whatever the title suggests.
- Price times quantity has to stay within the budget (equal is fine). An item the records show out of stock fails.
- A wish marked as not required ("ideally", "would be nice") is no requirement.
- Check only the requirements listed below; do not invent extra ones from the story (a gender, an age group, a size, the person who will use it).
- If the request asks for several different products, measure the item only against requirements for its own kind of product; one that begins with another product's name is not for it, and the quantity used is that of the item's kind.
If records conflict (on price or stock, say), go by the newest record.
Reply with ONE JSON object and no other text: {"item": the item's id, "fits": true or false, "problems": [short strings, each naming a requirement not met or not shown]}. Set fits=false only for a requirement the records plainly fail or that no record covers at all; a missing record of stock or price is no reason to fail an item."""

QUESTION_PLAN_PROMPT = """Your task is to plan which questions a shopping assistant sends the shopper. The shopper holds hidden needs, each the value of a single property of the product, and a need comes out only when a message asks about that very property; a message gets an answer only on the first thing it asks, and asking about anything else just brings a polite "don't mind". Pick QUESTION_COUNT different properties (fewer only if there are not enough useful ones), the most promising first:
0. to begin, whatever appears below under FIRST FIND OUT FROM THE SHOPPER and has not been asked yet. Only two kinds are asked in their own words: for the budget, ask the most they want to spend; for a choice between two wishes, give both values with their properties, e.g. "If no single listing can have both, would you prefer the <value> <property> or the <value> <property>?". Anything else listed there (what they need, who it is for, how it will be used) is found out through the property questions of rule 1, one property per question;
1. next, properties really present in the listings' details below, chosen in this order: (a) a property that a reason, story or circumstance in the request makes decisive for this product (how it will be used, seen, matched, carried or fitted, what went wrong before, who it is for), reading figurative wording for the part or aspect of the product it describes; (b) who the product is made for (men, women, children, an age group) when it comes in versions for different people and the request leaves that open, and its condition (new, used, refurbished) when the listings differ on it; (c) the defining choices a buyer of this kind of product makes by preference: how it looks (colour, pattern, style, shape), what it is made of (material), and its main type or configuration (type, form, cut, closure, the key component or technology), preferring those whose values differ among the listings. Sizes and dimensions follow the same rules as any other property. Rank low the fields about the listing, seller, warranty, model code or delivery, fields that lump several properties together (option or variant labels), and a property with the same single value on every listing; never ask about the maker, brand, seller or shop. A property counts as settled only when the request names it or gives its value: a stated finish does not settle the colour, and a stated colour does not settle the finish or pattern.
Do not choose a property the request has already settled (its stated requirements, the price, stock, quantity); do not repeat a property already asked about for the same product (if the shopper did not follow a question, it still counts as asked, so move on, unless that question left the product unnamed); do not use two questions on nearly duplicate properties (one property spelled two ways, two kinds of style); and skip properties that no listing carries. Name each property by the plainest name the listings use for it (the menu has already merged keys that hold the same values), never by a long title or description field. Each question is about exactly ONE property: never two or more properties in one message, and never an open question about the story (who it is for, what it is used for) instead of a property. Something to find out first that is itself a property of the product is asked like any other property, in the form below.
If the request asks for several different products, every question concerns ONE of them and the questions are distributed across the products: favour a product asked about less so far, and draw its properties from listings of that product's kind (when the listings carry labels, the menu below is grouped by them).
Every question takes the form "Which <property> would you like the <product> to have?": the property name given exactly as the listings give it (spaces in place of underscores or dots) and the product called by the plain noun the request uses ("the lamp", "the blanket"; no adjectives, colours, materials or other property values attached). No example values, no extra words, no second question. Anything to find out first (rule 0) is asked plainly in a single short sentence.
Reply with ONE JSON object and no other text: {"questions": [{"product": "...", "property": "...", "question": "..."}, ...]}."""

DECLINE_REVIEW_PROMPT = """A shopping assistant intends to tell the shopper that the store has nothing that fits. Test the listing records below against the requirements by these reading rules: a listing meets a requirement when the value is stated by its own property for it (any key spelling counts, as do a single element of a comma list, close synonyms and singular/plural; a broader or different class does not); title or description count only if the listing lacks a key for that property; numbers are compared once units are converted, and the edge is inside a limit; the price has to fall within the budget and the item may not appear out of stock; only the listed requirements matter, never one guessed from the story; if several different products are requested, check each listing only against requirements for its own product kind. Declining wrongly scores 0, so include every listing that might fit; omit one only if its own records plainly fail a listed requirement.
Reply with ONE JSON object and no other text: {"possible_fits": [ids of listings that seem to meet every requirement, best first, at most 5], "reason": "one sentence"}."""


# ---- section headings appended after the prompts ----

# reading (setup): static part
HEAD_TOOLS = "\n\nAVAILABLE TOOLS:\n"
# reading (setup): the variable part starts with this heading (no blank lines before it)
HEAD_REQUEST = "REQUEST FROM THE SHOPPER:\n"
HEAD_OPENING = "\n\nFIRST OBSERVATION FROM THE STORE:\n"

# checker, planner and decline review: the request
HEAD_SHOPPER_REQUEST = "\n\nREQUEST FROM THE SHOPPER:\n"

# listener (interpret)
HEAD_CONTEXT_REQUEST = "\n\nREQUEST FROM THE SHOPPER (background only):\n"
HEAD_REQS_SO_FAR = "\n\nREQUIREMENTS UP TO NOW:\n"          # listener and planner
HEAD_EARLIER = "\n\nPREVIOUS EXCHANGES WITH THE SHOPPER (oldest first):\n"
HEAD_ASKED = "\n\nWHAT THE ASSISTANT ASKED:\n"
HEAD_SHOPPER_NOW = "\n\nWHAT THE SHOPPER SAYS NOW:\n"

# checker (verify)
HEAD_CURRENT_REQS = "\n\nREQUIREMENTS IN FORCE (stated, revealed and changed):\n"
HEAD_ITEM_RECORDS = "STORE RECORDS FOR THIS ITEM (oldest first):\n"
# format with purpose ("ORDER" / "ADD TO THE CART") and item; already ends with the records heading
HEAD_ITEM_TO = "\n\nITEM THE ASSISTANT IS ABOUT TO {purpose}: {item}\n" + HEAD_ITEM_RECORDS
HEAD_CART_LINE = "\n\nCART LINE FOR THIS ITEM (if it names an option or variant, judge that option by its own values, price and stock count): "
HEAD_CART_VIEW = "\n\nMOST RECENT CART VIEW:\n"
HEAD_LABELS = "\n\nLABELS SEEN REPEATEDLY ON THIS STORE'S LISTINGS (with counts; the item's category label has to be the one fitting the request's full wording, not a neighbouring one):\n"

# planner
HEAD_SEVERAL_PRODUCTS = "\n\nSEVERAL DIFFERENT PRODUCTS ARE REQUESTED (each question names the one it concerns): "
HEAD_ALREADY_ASKED = "\n\nQUESTIONS ALREADY SENT (with the shopper's answers):\n"
HEAD_LEARN_FIRST = "\n\nFIRST FIND OUT FROM THE SHOPPER: "
HEAD_MENU = "\n\nPROPERTY MENU FROM THE LISTINGS' DETAILS SEEN SO FAR (name, number of listings, example values):\n"
MENU_EMPTY = "(nothing seen yet)"

# decline review (audit)
HEAD_AUDIT_REQS = "\n\nREQUIREMENTS IN FORCE:\n"
HEAD_RECORDS = "\n\nLISTING RECORDS ON HAND (newest first):\n"


_model_link = ProxyClient(timeout=70, max_retries=1, retry_delay=1)
_store_link = ProxyClient(timeout=95, max_retries=2, retry_delay=1)


# ---------------------------------------------------------------------------------------------- word patterns
def _any_of(*alternatives: str, flags: int = re.I) -> re.Pattern[str]:
    return re.compile("|".join(alternatives), flags)


# a refusal saying the item can no longer be had, as against one that reports a new price
_SOLD_OUT = _any_of(
    "sold out", "out of stock", "unavailable", "no longer", "zero left", "none left", "no units", "hit zero",
    "all gone", "is gone", "it's gone", "now gone", "gone from stock", "last one", "bought the last",
    "dried up", "ran dry", "run dry", "cleared out", "cleared from", "snapped up", "been sold", "been taken",
    "can't be ordered", "cannot be ordered", "grayed out", "greyed out",
    "none remaining", r"no .{0,12}remaining", r"in stock any ?more", "not in stock", "isn't in stock", "no stock",
    r"stock.{0,20}\bzero\b")
# a line about money (a new price alone leaves an item acceptable)
_MONEY_TALK = _any_of("price", "priced", "cost", "charg", "cents", "dearer", "cheaper",
                      "bumped", "gone up", "went up", "come down", "dropped")
# names that describe the listing rather than the product: never offered as question topics. The first group
# matches anywhere; the second only as the whole name, since those words also end real property names
_META_ANYWHERE = ("brand", "maker", "manufacturer", "made by", "seller", "sold by", "vendor", "merchant",
                  "title", "headline", "heading", "blurb", "tagline", "caption", "teaser", "slogan", "sku", "url")
_META_WHOLE = ("make", "shop", "shop name", "store", "store name", "outlet", "company",
               "category", "department", "shelf", "description", "(overview|description|summary)( text)?",
               "notes?", "image", "images", "rating", "reviews?", "review count", "link",
               "options?", "variants?", "id", "name", "price", "amount", "cost", "stock", "on hand", "in stock")
_LISTING_META = re.compile(r"\b(" + "|".join(_META_ANYWHERE) + r")\b|^(" + "|".join(_META_WHOLE) + r")$", re.I)
# an error saying a tool's allowance is spent (not a bad argument or an over-budget cart)
_ALLOWANCE_GONE = _any_of("budget_exceeded", "quota", "allowance", "used up", "already used", "limit reached",
                          "reached the limit", r"maximum (number|reached)", r"no more\b.{0,40}\b(may|can|allowed|left)")
_PROPERTY_KEYS = _any_of("propert", "attribut", "characteristic", "feature", "spec", "trait", "fact", "detail")
_ID_KEYS = _any_of("id", "ref", "key", "code", "number", "sku", "listing", "product")
_OPTION_KEYS = _any_of("option", "variant", "choice", "version", "edition", "config", "model", "sku")
_KIND_KEYS = _any_of("categor", "department", "^dept$", "shelf", "aisle", "section", "^kind$", "^family$", "^group$",
                     r"product.?(type|family|group|kind)", r"item.?(type|kind)", r"offer.?kind")
_FIRST_PERSON = re.compile(r"\b(I|I'm|I'd|I've|I'll|my|me|mine)\b")
# a shopper line that wants an answer; a request rule that wants the shopper told something
_WANTS_ANSWER = re.compile(r"\?|\b(" + "|".join(("confirm", "let me know", "tell me", "acknowledge", "reply",
                                                   "answer", "get back to me")) + r")\b", re.I)
_TELLING_RULE = re.compile(r"\b(" + "|".join(("tell", "inform", r"let (me|them|the shopper) know", "update",
                                                "confirm", "report", "mention", "explain", "message")) + r")\b", re.I)
# a tool no role covers that may finish a checkout; result keys that mark a waiting checkout step
_CHECKOUT_STEP = _any_of("final", "confirm", "complete", "checkout", "pay", "place", "submit", "commit", "token")
_WAITING_KEYS = _any_of("token", "pending", "await", "confirm", r"next.?step", r"checkout.?id", "session")
_NO_BUY_WORDS = _any_of("refus", "declin", "reject", "abandon", r"give.?up", r"no.?(buy|purchase|order)", "cancel",
                        "quit", "skip")
_ID_SHAPE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}$")
_CODE_LIKE = re.compile(r"[A-Za-z0-9_\-]+")
_DIGIT = re.compile(r"\d")


def _has_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def _looks_like_id(text: Any) -> bool:
    return isinstance(text, str) and bool(_ID_SHAPE.match(text))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# ---------------------------------------------------------------------------------------------- JSON and text
def _dump(value: Any) -> str:
    """Compact JSON text of a value, capped at TEXT_CAP characters."""
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text = str(value)
    return text if len(text) <= TEXT_CAP else text[:TEXT_CAP] + "...(cut)"


def _trim(value: Any, limit: int, depth: int = 0) -> Any:
    """The value with every string cut to `limit` characters (nesting deeper than eight levels is dropped)."""
    if depth > 8:
        return None
    if isinstance(value, str):
        return value[:limit] + "..." if len(value) > limit else value
    if isinstance(value, list):
        return [_trim(v, limit, depth + 1) for v in value]
    if isinstance(value, dict):
        return {k: _trim(v, limit, depth + 1) for k, v in value.items()}
    return value


def _top_object(text: str) -> dict[str, Any] | None:
    """The answer object in a model reply: the whole text, its outermost braces, or else the last complete
    top-level object when the model wrote several with words between them."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z]*", "", text).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    first, last = text.find("{"), text.rfind("}")
    for piece in (text, text[first:last + 1] if 0 <= first < last else None):
        if piece is None:
            continue
        try:
            value = json.loads(piece)
        except ValueError:
            continue
        return value if isinstance(value, dict) else None
    decoder = json.JSONDecoder()
    found = None
    at = first
    while 0 <= at < len(text):
        try:
            value, end = decoder.raw_decode(text, at)
        except ValueError:
            at = text.find("{", at + 1)
            continue
        # a piece nested inside a broken outer object is not an answer
        if isinstance(value, dict) and text.count("{", 0, at) == text.count("}", 0, at):
            found = value
        at = text.find("{", end)
    return found


def _wordy_strings(value: Any, into: list[str], depth: int = 0) -> None:
    """Collect strings of four or more words from a string or (nested) list; dicts are not entered."""
    if depth > 4:
        return
    if isinstance(value, str):
        if len(value.split()) >= 4:
            into.append(value)
    elif isinstance(value, list):
        for part in value[:10]:
            _wordy_strings(part, into, depth + 1)


def _gather_ids(value: Any, into: set[str], depth: int = 0) -> None:
    """Every id-shaped string held under a key containing "id"."""
    if depth > 6:
        return
    if isinstance(value, list):
        for part in value:
            _gather_ids(part, into, depth + 1)
    elif isinstance(value, dict):
        for key, part in value.items():
            if isinstance(part, str) and "id" in key.lower() and _ID_SHAPE.match(part):
                into.add(part)
            else:
                _gather_ids(part, into, depth + 1)


def _has_words(value: Any, depth: int = 0) -> bool:
    """Whether a result holds any word at all (beyond ids, numbers and flags)."""
    if depth > 4:
        return False
    if isinstance(value, str):
        return any(ch.isalpha() for ch in value) and not (_ID_SHAPE.match(value) and _has_digit(value))
    parts = value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()
    return any(_has_words(part, depth + 1) for part in parts)


def _waiting_checkout(observation: dict[str, Any]) -> bool:
    """An order result that waits for another step (a token, a confirmation) and carries no shopper sentence."""
    said: list[str] = []
    _wordy_strings([v for v in observation.values() if isinstance(v, (list, str))], said)
    keys = list(observation) + [k for v in observation.values() if isinstance(v, dict) for k in v]
    return any(_WAITING_KEYS.search(str(k)) for k in keys) and not any(len(s.split()) >= 8 for s in said)


def _id_field(key: Any, value: Any) -> bool:
    return _looks_like_id(value) and _has_digit(value) and bool(_ID_KEYS.search(str(key)))


def _basket_line_ids(args: dict[str, Any]) -> list[str] | None:
    """Listing ids of a cart call that passes its lines as a list (None when the call names a listing directly or
    has no such list). Each line gives its first id-like value, normally the listing."""
    if any(_id_field(k, v) for k, v in args.items()):
        return None
    ids = []
    for value in args.values():
        if isinstance(value, list) and value and all(isinstance(x, dict) for x in value):
            for line in value:
                first = next((v for k, v in line.items() if _id_field(k, v)), None)
                if first:
                    ids.append(first)
    return ids or None


def _line_naming(args: dict[str, Any], ident: str) -> str:
    """Text of the cart line naming this listing, for a cart call whose lines come as a list."""
    for value in args.values():
        for line in value if isinstance(value, list) else ():
            if isinstance(line, dict) and ident in line.values():
                return _dump(line)
    return ""


def _string_values(args: dict[str, Any]) -> str:
    return " ".join(str(v) for v in args.values() if isinstance(v, str))


# ---------------------------------------------------------------------------------------------- argument schemas
_TYPE_TESTS = {
    "string": lambda v: isinstance(v, str),
    "integer": _is_count,
    "number": _is_number,
    "boolean": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "null": lambda v: v is None,
}


def _types_of(schema: dict[str, Any]) -> list[str]:
    kinds = schema.get("type")
    return [kinds] if isinstance(kinds, str) else kinds or []


def _fits_type(value: Any, kind: str) -> bool:
    test = _TYPE_TESTS.get(kind)
    return test(value) if test else True


def _convert(value: Any, kind: str) -> tuple[bool, Any]:
    """Try to turn a value of the wrong type into `kind`; (True, new value) when it worked."""
    if kind == "integer":
        if isinstance(value, float) and value.is_integer():
            return True, int(value)
        if isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.0+)?\s*", value):
            return True, int(float(value))
    elif kind == "number":
        if isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
            number = float(value)
            return True, int(number) if number.is_integer() else number
    elif kind == "string":
        if _is_number(value):
            return True, str(value)
    elif kind == "boolean":
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return True, value.lower() == "true"
    return False, value


def _nudge(value: Any, schema: dict[str, Any]) -> Any:
    kinds = _types_of(schema)
    if not kinds:
        return value
    if any(_fits_type(value, k) for k in kinds):
        if "integer" in kinds and isinstance(value, float) and value.is_integer():
            return int(value)
        return value
    for kind in kinds:
        try:
            done, new = _convert(value, kind)
        except (TypeError, ValueError):
            continue
        if done:
            return new
    return value


def _bound_problems(value: Any, schema: dict[str, Any], path: str) -> list[str]:
    out = []
    if "minimum" in schema and value < schema["minimum"]:
        out.append(f"{path} is under the minimum {schema['minimum']}")
    if "maximum" in schema and value > schema["maximum"]:
        out.append(f"{path} is over the maximum {schema['maximum']}")
    low, high = schema.get("exclusiveMinimum"), schema.get("exclusiveMaximum")
    if "exclusiveMinimum" in schema and isinstance(low, (int, float)) and value <= low:
        out.append(f"{path} must be greater than {low}")
    if "exclusiveMaximum" in schema and isinstance(high, (int, float)) and value >= high:
        out.append(f"{path} must be less than {high}")
    return out


def _conform(value: Any, schema: dict[str, Any], path: str) -> tuple[Any, list[str]]:
    """The value fitted to a JSON schema where that is safe (type nudges, enum case, cut to maxLength), with the
    problems that remain."""
    if not isinstance(schema, dict):
        return value, []
    value = _nudge(value, schema)
    kinds = _types_of(schema)
    if kinds and not any(_fits_type(value, k) for k in kinds):
        return value, [f"{path} should be {' or '.join(kinds)}"]
    problems: list[str] = []
    allowed = schema.get("enum")
    if "enum" in schema and value not in allowed:
        if isinstance(value, str):
            wanted = value.strip().lower()
            value = next((o for o in allowed if isinstance(o, str) and o.lower() == wanted), value)
        if value not in allowed:
            problems.append(f"{path} must be one of {allowed}")
    if "const" in schema and value != schema["const"]:
        problems.append(f"{path} must be exactly {schema['const']!r}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            problems.append(f"{path} needs at least {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            value = value[: schema["maxLength"]]
    elif _is_number(value):
        problems += _bound_problems(value, schema, path)
    elif isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            problems.append(f"{path} needs at least {schema['minItems']} entries")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            problems.append(f"{path} takes at most {schema['maxItems']} entries")
        if isinstance(schema.get("items"), dict):
            fitted = []
            for index, entry in enumerate(value):
                entry, more = _conform(entry, schema["items"], f"{path}[{index}]")
                fitted.append(entry)
                problems += more
            value = fitted
    elif isinstance(value, dict):
        fields = schema.get("properties") or {}
        problems += [f"{path}.{name} is missing" for name in schema.get("required") or [] if name not in value]
        if schema.get("additionalProperties") is False:
            problems += [f"{path}.{name} is not an accepted argument" for name in value if name not in fields]
        fitted_obj = {}
        for name, entry in value.items():
            if name in fields:
                entry, more = _conform(entry, fields[name], f"{path}.{name}")
                problems += more
            fitted_obj[name] = entry
        value = fitted_obj
    return value, problems


def _is_text_field(spec: Any) -> bool:
    """A free-text argument: typed string (alone or among types) and not an enum."""
    if not isinstance(spec, dict) or "enum" in spec:
        return False
    kind = spec.get("type")
    return kind == "string" or "string" in (kind or [])


def _enum_part(args: dict[str, Any], fields: dict[str, Any]) -> str:
    """The call's enum-valued arguments as text: what tells an "add" from a "remove" of one cart tool."""
    return _dump({k: v for k, v in args.items() if isinstance(fields.get(k), dict) and "enum" in fields[k]})


def _sorted_by_order(replies: list[tuple[str, str]], calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
    position = {c["id"]: i for i, c in enumerate(calls)}
    return sorted(replies, key=lambda r: position.get(r[0], 0))


# ---------------------------------------------------------------------------------------------- the counted meter
class Meter:
    """Steps, counted seconds and money of one episode."""

    def __init__(self, step_cap: int):
        self.step = 0
        self.step_cap = step_cap
        self.seconds = 0.0
        self.spend = 0.0

    def steps_left(self) -> int:
        return self.step_cap - self.step

    def seconds_left(self) -> float:
        return RUN_SECONDS - self.seconds

    def charge(self, seconds: float) -> None:
        self.seconds += seconds * _TALLY_SCALE

    def book_reply(self, data: Any, seat: str) -> None:
        """Money and counted time of a model reply that came back with choices."""
        usage = data.get("usage") or {}
        if isinstance(usage.get("cost"), (int, float)):
            self.spend += float(usage["cost"])
        produced = usage.get("completion_tokens")
        if _is_number(produced):
            base, per_second = _REPLY_PACE.get(seat, _REPLY_PACE_OTHER)
            self.charge(_REPLY_MARGIN * (base + produced / per_second))
        else:
            self.charge(_REPLY_UNSIZED_S)

    def book_fault(self, outcome: Any, link: ProxyClient) -> None:
        """A call that brought no data: one with no status is taken to have waited out its timeout."""
        status = (outcome.error or {}).get("status") if isinstance(outcome.error, dict) else None
        self.charge(_FAULT_S if status is not None else float(link.timeout))


def _with_cache_marks(seat: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For an Anthropic seat in a conversation, the first and the newest message carry a cache mark."""
    if not seat.startswith("anthropic/") or len(messages) <= 2:
        return messages
    marked = [dict(m) for m in messages]
    for at in (0, len(marked) - 1):
        text = marked[at].get("content")
        if isinstance(text, str) and text:
            marked[at]["content"] = [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
    return marked


class Wire:
    """Requests to the model seats and to the store."""

    def __init__(self, errand: "Errand"):
        self.e = errand

    def ask(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, seats: list[str],
            max_tokens: int) -> dict[str, Any] | None:
        """The first seat in the list that answers with words or tool calls; None when none does (or time is up)."""
        meter = self.e.meter
        for seat in seats:
            left = meter.seconds_left()
            if left < STOP_MARGIN_S - 10:
                return None
            _model_link.timeout = int(max(8, min(75, left - STOP_MARGIN_S + 15)))
            body: dict[str, Any] = {"model": seat, "messages": _with_cache_marks(seat, messages), "temperature": 0,
                                    "max_tokens": max_tokens}
            if tools:
                body["tools"] = tools
                body["tool_choice"] = "auto"
            outcome = _model_link.post_verbose("/inference/chat/completions", body)
            data = outcome.data
            if isinstance(data, dict) and data.get("choices"):
                meter.book_reply(data, seat)
                message = data["choices"][0].get("message") or {}
                if message.get("content") or message.get("tool_calls"):
                    return message
            else:
                meter.book_fault(outcome, _model_link)
            self.e.log("inference_failure", model=seat, error=str(outcome.error)[:300] if outcome.error else "empty")
        return None

    def ask_json(self, prompt: str, max_tokens: int = 700, seats: list[str] | None = None) -> dict[str, Any] | None:
        """A one-message request whose answer is a JSON object. Checks move to the lighter seat first once the
        episode has spent a lot, so that one task cannot eat the key."""
        seats = seats or [CHECK_SEAT, DRIVER_SEAT]
        if self.e.meter.spend > SOFT_SPEND_USD and seats[0] == CHECK_SEAT:
            seats = [LIGHT_SEAT, CHECK_SEAT]
        message = self.ask([{"role": "user", "content": prompt}], None, seats, max_tokens)
        return _top_object((message or {}).get("content") or "")

    def act(self, actions: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Send one or more actions as the next step(s). A step mismatch is resynced and sent again; other faults
        end the attempt (a 402 ends the episode's store access)."""
        e, meter = self.e, self.e.meter
        envelope: dict[str, Any] = dict(e.binding)
        envelope["step"] = meter.step + 1
        envelope["action" if len(actions) == 1 else "actions"] = actions[0] if len(actions) == 1 else actions
        for _attempt in range(3):
            left = meter.seconds_left()
            if left < 2:
                return None
            _store_link.timeout = int(max(3, min(95, left - 1)))
            outcome = _store_link.post_verbose("/environment/call", envelope)
            meter.charge(_STORE_TRIP_S)
            if isinstance(outcome.data, dict):
                data = outcome.data
                meter.step = data["step"] if isinstance(data.get("step"), int) else meter.step + len(actions)
                if data.get("finished"):
                    e.finished = True
                return data
            meter.book_fault(outcome, _store_link)
            fault = outcome.error or {}
            status, text = fault.get("status"), str(fault.get("body") or "")
            e.log("env_failure", status=status, body=text[:300])
            if status == 402:
                e.dead = True
                return None
            if status == 409:
                if "finished" in text:
                    e.finished = True
                    return None
                expected = re.search(r"expected (\d+)", text)
                if not expected:
                    return None
                meter.step = int(expected.group(1)) - 1
                envelope["step"] = meter.step + 1
                continue
            if status in (403, 422, 500, 503, 504):
                return None
        return None


# ---------------------------------------------------------------------------------------------- reading the request
def _name_words(name: str) -> set[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return {w.lower() for w in re.split(r"[^A-Za-z0-9]+", spaced) if w}


_BASKET_WORDS = {"cart", "basket", "bag", "trolley", "stash", "hold", "line", "lines", "keep", "selection"}
_LOOK_WORDS = {"view", "show", "list", "review", "status", "current", "my", "see", "contents", "summary"}
_EDIT_WORDS = {"add", "put", "set", "remove", "rewrite", "update"}
_STOP_WORDS = {"decline", "refuse", "reject", "abandon", "dismiss", "quit", "giveup"}
_BUY_WORDS = {"checkout", "purchase", "order", "buy", "commit", "confirm", "place", "settle", "pay", "execute",
              "finalize", "submit"}
_TALK_WORDS = {"message", "shopper", "buyer", "customer", "client", "ask", "chat", "note", "contact", "write", "talk"}
_OPEN_WORDS = {"detail", "details", "inspect", "open", "read", "view", "fetch", "get", "datasheet", "info", "spec",
               "specs", "describe", "examine"}
_FIND_WORDS = {"search", "find", "browse", "locate", "hunt", "query", "lookup", "catalog", "seek", "list"}


class Reading:
    """What the reading seat made of the request and the tool list: the plan, and which tool plays which role."""

    def __init__(self, errand: "Errand"):
        self.e = errand
        self.plan: dict[str, Any] = {}
        self.roles: dict[str, str] = {}
        self.budget_unknown = False     # the request names no price limit: the budget is asked before anything else

    def tool(self, role: str) -> str | None:
        return self.roles.get(role)

    def perform(self) -> None:
        e = self.e
        listing = "\n".join(f"- {t['function']['name']}: {t['function'].get('description', '')} | arguments: "
                            f"{_dump(t['function'].get('parameters') or {})}" for t in e.tools)
        fixed_part = READING_PROMPT + HEAD_TOOLS + listing
        task_part = HEAD_REQUEST + e.request
        if e.opening is not None:
            task_part += HEAD_OPENING + _dump(e.opening)
        content = [{"type": "text", "text": fixed_part, "cache_control": {"type": "ephemeral"}},
                   {"type": "text", "text": task_part}]
        reply = e.wire.ask([{"role": "user", "content": content}], None, [READING_SEAT, FALLBACK_SEAT], 1500)
        plan = _top_object((reply or {}).get("content") or "") or {}
        if not isinstance(plan.get("roles"), dict) and e.meter.seconds_left() > 200:
            # every later guard leans on this reading: one more try
            reply = e.wire.ask([{"role": "user", "content": content}], None, [READING_SEAT, DRIVER_SEAT], 2000)
            plan = _top_object((reply or {}).get("content") or "") or plan
        self.plan = plan
        named = plan.get("roles") if isinstance(plan.get("roles"), dict) else {}
        self.roles = {role: tool for role, tool in named.items() if isinstance(tool, str) and tool in e.schemas}
        if not self.roles:
            self.roles = self.guess_roles()
            e.log("guessed_roles", roles=self.roles)
        allowance = plan.get("message_budget")
        if _is_number(allowance) and allowance >= 0:
            e.talk.allowance = int(allowance)
        e.wants.stated = [str(r) for r in plan.get("requirements") or [] if r]
        e.wants.budget = plan.get("price_cap")
        if plan and e.wants.budget is None and self.roles.get("message"):
            plan["ask_first"] = [NOTE_ASK_BUDGET] + [str(a) for a in plan.get("ask_first") or [] if a]
            self.budget_unknown = True
        steps = plan.get("step_budget")
        if _is_number(steps) and 0 < steps < e.meter.step_cap:
            e.meter.step_cap = int(steps)

    def guess_roles(self) -> dict[str, str]:
        """Roles read off the tool names, for when the reading failed."""
        schemas = self.e.schemas

        def single_text_argument(name: str) -> bool:
            fields = (schemas.get(name) or {}).get("properties") or {}
            return sum(1 for spec in fields.values() if isinstance(spec, dict) and spec.get("type") == "string"
                       and "enum" not in spec) == 1

        rules = (
            ("decline", lambda w, n: bool(w & _STOP_WORDS) or any(x.startswith(("declin", "refus")) for x in w)
             or {"give", "up"} <= w or "nothing" in w),
            ("cart_view", lambda w, n: bool(w & (_BASKET_WORDS | {"items", "order"})) and bool(w & _LOOK_WORDS)),
            ("cart", lambda w, n: bool(w & _BASKET_WORDS)
             or bool(w & _EDIT_WORDS) and bool(w & (_BASKET_WORDS | {"items", "order", "all"}))),
            ("order", lambda w, n: bool(w & _BUY_WORDS)),
            ("message", lambda w, n: bool(w & _TALK_WORDS) and single_text_argument(n)),
            ("details", lambda w, n: bool(w & _OPEN_WORDS)),
            ("search", lambda w, n: bool(w & _FIND_WORDS)),
        )
        names = [t["function"]["name"] for t in self.e.tools]
        roles: dict[str, str] = {}
        for role, fits in rules:
            pick = next((n for n in names if n not in roles.values() and fits(_name_words(n), n)), None)
            if pick is not None:
                roles[role] = pick
        return roles

    # ---- the plan's fields
    def products(self) -> list[tuple[str, int]]:
        """The different products of a request that asks for several (name, quantity); empty for one product."""
        found = []
        raw = self.plan.get("items")
        for entry in raw if isinstance(raw, list) else []:
            if isinstance(entry, str) and entry.strip():
                entry = {"product": entry, "quantity": 1}
            if not (isinstance(entry, dict) and str(entry.get("product") or "").strip()):
                continue
            count = entry.get("quantity")
            if isinstance(count, str) and count.strip().isdigit():
                count = int(count.strip())
            count = int(count) if _is_number(count) and count >= 1 else 1
            found.append((str(entry["product"]).strip(), count))
        return found if len(found) >= 2 else []

    def priority(self) -> Any:
        return self.plan.get("priority")

    def currency_suffix(self, amount: Any) -> str:
        return f"{amount} {self.plan.get('currency') or ''}".rstrip()

    def outline(self) -> str:
        """The reading, laid out for the driver's first message."""
        plan = self.plan
        if not plan:
            return ""
        rows = [OUTLINE_HEAD]
        if plan.get("product"):
            rows.append(f"- product: {plan.get('product')}")
        several = self.products()
        if several:
            rows += [f"- buy {name}, quantity {count} (its own cart line)" for name, count in several]
            rows.append("- before choosing, open the details of six or more listings of EACH product (a tool that shows "
                        "several listings at once saves steps): the questions to the shopper are planned from them")
        else:
            rows.append(f"- quantity: {plan.get('quantity') or 1}")
        rows += [f"- requirement stated in the request: {r}" for r in plan.get("requirements") or []]
        if plan.get("price_cap") is not None:
            rows.append(f"- budget for the total price: {self.currency_suffix(plan.get('price_cap'))}")
        if plan.get("in_stock_only"):
            rows.append("- in-stock / on-hand items only")
        if plan.get("priority") in ("cheapest", "best"):
            measure = f" ({plan.get('priority_detail')})" if plan.get("priority_detail") else ""
            rows.append(f"- pick the {plan.get('priority')}{measure} of all the items that fit")
        else:
            rows.append(OUTLINE_NO_PRIORITY)
        rows += [f"- wish only, NOT a requirement (use it just to break ties): {w}" for w in plan.get("soft_wants") or []]
        rows += [f"- to learn from the shopper first: {a}" for a in plan.get("ask_first") or []]
        rows += [f"- rule of the request: {r}" for r in plan.get("rules") or []]
        if self.e.talk.allowance is not None:
            rows.append(f"- messages you may send the shopper: {self.e.talk.allowance}")
        rows.append(f"- steps you may use: {self.e.meter.step_cap}")
        return "\n".join(rows)


# ---------------------------------------------------------------------------------------------- the requirements
class Wants:
    """The requirements as they stand: stated in the request, revealed or changed by the shopper."""

    def __init__(self, errand: "Errand"):
        self.e = errand
        self.stated: list[str] = []
        self.revealed: list[str] = []
        self.free: list[str] = []
        self.dropped: list[str] = []
        self.rewritten: list[str] | None = None     # the listener's full list, once it has given one
        self.budget: Any = None
        self.version = 0                             # bumped whenever the shopper changes something

    def text(self) -> str:
        reading = self.e.reading
        if self.rewritten is not None:
            parts = list(self.rewritten)
        else:
            parts = [f"{r} (stated)" for r in self.stated if r not in self.dropped]
            parts += [f"{r} (revealed by the shopper)" for r in self.revealed if r not in self.dropped]
        out = "; ".join(parts) if parts else "as in the request"
        if self.budget is not None:
            out += "; total price no more than " + reading.currency_suffix(self.budget)
        several = reading.products()
        if several:
            out += "; products and quantities: " + ", ".join(f"{name} x{count}" for name, count in several)
        else:
            out += f"; quantity {reading.plan.get('quantity') or 1}"
        if self.dropped:
            out += "; dropped by the shopper: " + "; ".join(self.dropped)
        if self.free:
            out += "; the shopper has no preference on: " + ", ".join(self.free)
        return out


# ---------------------------------------------------------------------------------------------- what the store showed
def _short_text(value: Any) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 70


_PLAIN_VALUE_WORDS = 6     # a longer value is a title or a blurb, not the value of a property


def _code_value(text: str) -> bool:
    """An id-like token (letters and digits joined by - or _), never a property value."""
    return bool(_DIGIT.search(text)) and bool(_CODE_LIKE.fullmatch(text)) and ("-" in text or "_" in text)


def _holds_property_table(value: Any, depth: int) -> bool:
    if depth > 6:
        return False
    if isinstance(value, list):
        return any(_holds_property_table(v, depth + 1) for v in value)
    if not isinstance(value, dict):
        return False
    return any((isinstance(v, dict) and _PROPERTY_KEYS.search(str(k))
                and sum(1 for x in v.values() if _short_text(x)) >= 2) or _holds_property_table(v, depth + 1)
               for k, v in value.items())


class _MenuTally:
    """Property names seen in listing records, per kind label: how often, which values, a few samples."""

    def __init__(self, flat_rows: bool = True) -> None:
        self.flat_rows = flat_rows      # whether a record's own short values may stand for its properties
        self.count: dict[str, dict[str, int]] = {}
        self.values: dict[str, dict[str, set[str]]] = {}
        self.samples: dict[str, dict[str, list[str]]] = {}
        self.tables: dict[str, int] = {}
        self.fact_sets = 0
        self.table_no = 0
        # (table, value) pairs per name: two names with the same pairs are one property under two keys
        self.pairs: dict[str, dict[str, set[tuple[int, str]]]] = {}

    def add(self, group: str, name: str, value: str) -> None:
        self.pairs.setdefault(group, {}).setdefault(name, set()).add((self.table_no, value))
        per = self.count.setdefault(group, {})
        per[name] = per.get(name, 0) + 1
        self.values.setdefault(group, {}).setdefault(name, set()).add(value)
        kept = self.samples.setdefault(group, {}).setdefault(name, [])
        if value not in kept and len(kept) < 5:
            kept.append(value)

    def visit(self, value: Any, depth: int, key: str = "", group: str = "") -> None:
        if depth > 6:
            return
        if isinstance(value, list):
            for part in value:
                self.visit(part, depth + 1, key, group)
            return
        if not isinstance(value, dict):
            return
        # a kind label (category, department, shelf...) groups the properties under it
        label = next((v.strip().lower() for k, v in value.items() if _short_text(v) and _KIND_KEYS.search(str(k))), "")
        group = label or group
        shorts = [(k, v) for k, v in value.items() if _short_text(v)]
        holds_table = any(isinstance(v, dict) and _PROPERTY_KEYS.search(str(k)) for k, v in value.items())
        # a property table: two short values under a key that names properties or options, or five anywhere in a
        # record that has no such table of its own
        if (len(shorts) >= 2 and (_PROPERTY_KEYS.search(key) or _OPTION_KEYS.search(key))) or (len(shorts) >= 5 and not holds_table and self.flat_rows):
            kept = 0
            self.table_no += 1
            for raw_name, text in shorts:
                if _code_value(text) or len(text.split()) > _PLAIN_VALUE_WORDS:
                    continue
                name = re.sub(r"[_.]+", " ", str(raw_name)).strip().lower()
                if _LISTING_META.search(name) or _KIND_KEYS.search(name):
                    continue
                self.add(group, name, text.lower())
                kept += 1
            # ids, titles and seller names alone are no set of facts; the options of one listing are not listings
            if kept >= 2 and not _OPTION_KEYS.search(key):
                self.fact_sets += 1
                self.tables[group] = self.tables.get(group, 0) + 1
        for name, part in value.items():
            if isinstance(part, (dict, list)):
                self.visit(part, depth + 1, str(name), group)

    def plain_names(self, group: str) -> list[str]:
        """The group's property names with duplicates dropped, keeping the shorter, plainer name: two keys holding the
        same values on the same listings, or a key that only adds a word or two to another one (material / material
        filter, pattern / fa pattern) whose values agree with it on at least half the listings carrying both."""
        pairs = self.pairs.get(group, {})
        kept: list[str] = []
        for name in sorted(self.count.get(group, {}), key=lambda n: (len(n.split()), len(n))):
            if not any(self._same_property(pairs.get(name) or set(), pairs.get(other) or set(), name, other) for other in kept):
                kept.append(name)
        return kept

    @staticmethod
    def _same_property(mine: set, theirs: set, name: str, other: str) -> bool:
        if mine and mine == theirs:
            return True
        words, other_words = set(name.split()), set(other.split())
        if not (other_words < words and len(words - other_words) <= 2):
            return False
        a, b = dict(mine), dict(theirs)
        shared = [t for t in a if t in b]
        return bool(shared) and sum(a[t] == b[t] for t in shared) >= len(shared) / 2

    def lines(self, group: str, limit: int) -> list[str]:
        count, values, samples = self.count.get(group, {}), self.values.get(group, {}), self.samples.get(group, {})
        # properties whose values differ between listings first (only those can change the choice), then the ones
        # most listings carry
        order = sorted(self.plain_names(group), key=lambda k: (len(values.get(k, ())) < 2, -count[k]))[:limit]
        return [f"- {k} (on {count[k]} listings, {len(values.get(k, ()))} distinct values): " + " | ".join(samples.get(k, []))
                for k in order]

    def fold(self) -> None:
        """Merge every group into the unlabelled one ("")."""
        merged: dict[str, int] = {}
        for group in self.count:
            for name, n in self.count[group].items():
                merged[name] = merged.get(name, 0) + n
                self.pairs.setdefault("", {}).setdefault(name, set()).update(self.pairs.get(group, {}).get(name, ()))
                self.values.setdefault("", {}).setdefault(name, set()).update(self.values[group][name])
                kept = self.samples.setdefault("", {}).setdefault(name, [])
                for sample in self.samples[group][name]:
                    if sample not in kept and len(kept) < 5:
                        kept.append(sample)
        self.count[""] = merged


class Archive:
    """Every observation the store returned, and what can be read from them."""

    def __init__(self, errand: "Errand"):
        self.e = errand
        self.records: list[Any] = []
        self.seen_ids: set[str] = set()
        self.opened: set[str] = set()
        self.searched_at = -1

    def keep(self, observation: Any) -> None:
        self.records.append(observation)

    def records_of(self, item: str) -> list[str]:
        """The newest three records that name this id (an option is shown inside the listing that holds it)."""
        found: list[str] = []

        def visit(value: Any, depth: int, holder: Any = None, in_options: bool = False) -> None:
            if depth > 6 or len(found) >= 8:
                return
            if isinstance(value, list):
                for part in value:
                    visit(part, depth + 1, holder, in_options)
                return
            if not isinstance(value, dict):
                return
            if any(isinstance(v, str) and v == item for v in value.values()):
                shown = holder if holder is not None and in_options and holder is not value else value
                text = _dump(_trim(shown, 260))[:3000]
                if text not in found:
                    found.append(text)
                return
            if any(_id_field(k, v) for k, v in value.items()):
                holder, in_options = value, False
            for k, v in value.items():
                visit(v, depth + 1, holder, in_options or bool(_OPTION_KEYS.search(str(k))))

        for record in self.records:
            visit(record, 0)
        return found[-3:]

    def property_menu(self) -> tuple[str, int]:
        """The properties the listings' details carry (for the question planner), and how many fact sets were seen."""
        # once the store has shown a real property table, flat result rows (titles, blurbs, sellers) stay out
        tally = _MenuTally(flat_rows=not any(_holds_property_table(r, 0) for r in self.records))
        for record in self.records:
            tally.visit(record, 0)
        # several products: one menu per kind label, even when a kind has a single listing with details, so that
        # each question can be taken from its own product's properties
        groups = [g for g in sorted(tally.count, key=lambda g: -tally.tables.get(g, 0)) if g and tally.tables.get(g, 0) >= 1]
        if groups and self.e.reading.products():
            text: list[str] = []
            for group in groups[:6]:
                text.append(f"Listings with the label \"{group}\" ({tally.tables[group]} with details):")
                text += tally.lines(group, 14)
            if tally.count.get(""):
                text.append("Listings with no label:")
                text += tally.lines("", 10)
            return "\n".join(text), tally.fact_sets
        tally.fold()
        return "\n".join(tally.lines("", 28)), tally.fact_sets

    def label_counts(self) -> str:
        """Short values that recur under one key across records (likely category labels), with counts."""
        seen: dict[str, dict[str, int]] = {}

        def visit(value: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(value, list):
                for part in value:
                    visit(part, depth + 1)
                return
            if not isinstance(value, dict):
                return
            if len(value) >= 4:
                for key, text in value.items():
                    if isinstance(text, str) and 0 < len(text) <= 40 and not (_DIGIT.search(text) and _CODE_LIKE.fullmatch(text)):
                        per = seen.setdefault(str(key), {})
                        per[text] = per.get(text, 0) + 1
            for part in value.values():
                if isinstance(part, (dict, list)):
                    visit(part, depth + 1)

        for record in self.records:
            visit(record, 0)
        rows = []
        for key, per in seen.items():
            if sum(per.values()) >= 4 and 2 <= len(per) <= 8 and max(per.values()) >= 2:
                ranked = sorted(per.items(), key=lambda kv: -kv[1])
                rows.append(f"- {key}: " + " | ".join(f"{v} (x{n})" for v, n in ranked))
        return "\n".join(rows[:12])

    def unopened_of_kind(self, stems: set[str], limit: int) -> list[str]:
        """Listings seen in results whose kind label shares a word with `stems` and whose details were never opened."""
        found: list[str] = []
        stack = list(self.records)
        while stack and len(found) < limit:
            value = stack.pop(0)
            if isinstance(value, list):
                stack.extend(value)
            elif isinstance(value, dict):
                ident = next((v for k, v in value.items() if _id_field(k, v)), None)
                label = next((v for k, v in value.items() if _short_text(v) and _KIND_KEYS.search(str(k))), "")
                if ident and label and ident not in self.opened and ident not in found and stems & _name_stems(label):
                    found.append(ident)
                stack.extend(v for v in value.values() if isinstance(v, (dict, list)))
        return found

    def opened_of_kind(self, stems: set[str]) -> int:
        return sum(1 for ident in self.opened if stems & _name_stems(self.label_of(ident)))

    def label_of(self, ident: str) -> str:
        """The kind label (category, shelf...) of the record that names this listing ("" when none shows one)."""
        for text in self.records_of(ident):
            try:
                record = json.loads(text)
            except ValueError:
                continue
            stack = [record]
            while stack:
                value = stack.pop()
                if isinstance(value, dict):
                    if ident in value.values():
                        label = next((v for k, v in value.items() if _short_text(v) and _KIND_KEYS.search(str(k))), "")
                        if label:
                            return label.lower()
                    stack.extend(v for v in value.values() if isinstance(v, (dict, list)))
                elif isinstance(value, list):
                    stack.extend(value)
        return ""

    def unopened_cheaper(self) -> list[tuple[str, Any]]:
        """For a cheapest request: rows priced below the newest cart item, in its category, not shown out of stock
        or over the budget, whose details were never opened (cheapest first, at most five)."""
        e = self.e
        if e.reading.priority() != "cheapest" or not e.basket.ids:
            return []
        rows: dict[str, dict[str, Any]] = {}

        def visit(value: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(value, list):
                for part in value:
                    visit(part, depth + 1)
                return
            if not isinstance(value, dict):
                return
            ident = next((v for k, v in value.items() if isinstance(v, str) and "id" in k.lower() and _ID_SHAPE.match(v)), None)
            if ident and _is_number(value.get("price")):
                rows[ident] = {**rows.get(ident, {}), **{k: v for k, v in value.items() if v is not None}}
            for part in value.values():
                if isinstance(part, (dict, list)):
                    visit(part, depth + 1)

        for record in self.records:
            visit(record, 0)
        held = rows.get(e.basket.ids[-1])
        if not held:
            return []
        wanted = e.reading.plan.get("quantity")
        units = wanted if isinstance(wanted, int) and wanted > 0 else 1
        budget = e.wants.budget if _is_number(e.wants.budget) else None
        cheaper = []
        for ident, row in rows.items():
            if ident in self.opened or ident in e.basket.ids or row["price"] >= held["price"]:
                continue
            if row.get("in_stock") is False or (budget is not None and row["price"] * units > budget):
                continue
            if held.get("category") and row.get("category") and row.get("category") != held.get("category"):
                continue
            cheaper.append((row["price"], ident))
        return [(ident, price) for price, ident in sorted(cheaper)[:5]]


# ---------------------------------------------------------------------------------------------- the cart
class Basket:
    """What the cart is believed to hold, from the cart calls sent."""

    def __init__(self, errand: "Errand"):
        self.e = errand
        self.ids: list[str] = []
        self.lines: dict[str, str] = {}
        self.add_mode: str | None = None    # enum arguments of the first cart call: that is how this store adds
        self.used = False
        self.latest_view: Any = None
        self.vetted: set[str] = set()
        self.warned: set[str] = set()

    def fields(self) -> dict[str, Any]:
        return self.e.schemas[self.e.reading.tool("cart")].get("properties") or {}

    def add_args(self, ident: str, units: int) -> dict[str, Any] | None:
        """Arguments that add `units` of a listing with the cart tool, read off its schema: the listing in its id
        field (or as one line of a lines list), the units in its count field, an add-like value for a mode."""
        fields = self.fields()
        args: dict[str, Any] = {}
        for key, spec in fields.items():
            if not isinstance(spec, dict):
                continue
            kind = spec.get("type")
            if "enum" in spec and spec["enum"]:
                adding = [v for v in spec["enum"] if isinstance(v, str) and re.search(r"add|put|hold|set|include|keep|plus|insert", v, re.I)]
                args[key] = (adding or spec["enum"])[0]
            elif kind == "integer":
                args[key] = units
            elif kind == "string" and key not in args:
                if not any(isinstance(v, str) and v == ident for v in args.values()):
                    args[key] = ident
            elif kind == "array" and isinstance(spec.get("items"), dict):
                line_fields = spec["items"].get("properties") or {}
                line = {}
                for lk, ls in line_fields.items():
                    lt = ls.get("type") if isinstance(ls, dict) else None
                    if lt == "string" and ident not in line.values():
                        line[lk] = ident
                    elif lt == "integer":
                        line[lk] = units
                if line:
                    args[key] = self._lines_with(line)
        fitted, problems = _conform(args, self.e.schemas[self.e.reading.tool("cart")], "args")
        return None if problems else fitted

    def _lines_with(self, line: dict[str, Any]) -> list[dict[str, Any]]:
        """The cart's believed lines plus a new one, for a tool that takes the whole basket as a list."""
        kept = []
        for ident in self.ids:
            text = self.lines.get(ident)
            try:
                kept.append(json.loads(text)) if text else None
            except ValueError:
                pass
        return kept + [line]

    def target(self, args: dict[str, Any]) -> str | None:
        """The listing a cart call is about: a one-word free-text argument, else the last listed line."""
        fields = self.fields()
        direct = next((v for k, v in args.items() if isinstance(v, str) and "enum" not in (fields.get(k) or {})
                       and v.strip() and " " not in v.strip()), None)
        if direct is None and _basket_line_ids(args):
            return _basket_line_ids(args)[-1]
        return direct

    def quantity_issue(self, name: str, args: dict[str, Any]) -> str:
        """Why an add call asks for the wrong number of units ("" when it does not)."""
        wanted = self.e.reading.plan.get("quantity")
        if not _is_count(wanted) or wanted < 1:
            return ""
        fields = self.e.schemas[name].get("properties") or {}
        if self.add_mode is not None and _enum_part(args, fields) != self.add_mode:
            return ""
        several = self.e.reading.products()
        for key, spec in fields.items():
            if not (isinstance(spec, dict) and spec.get("type") == "integer" and "enum" not in spec):
                continue
            asked = args.get(key, spec.get("default", 1))
            if several:
                if isinstance(asked, int) and asked not in {count for _, count in several}:
                    listed = ", ".join(f"{n} x{c}" for n, c in several)
                    return f"the request wants these quantities: {listed}, and this call asks for {asked}."
                return ""
            if isinstance(asked, int) and asked != wanted:
                return f"the request wants {wanted} units of one listing, and this call asks for {asked}."
        return ""

    def record(self, name: str, args: dict[str, Any]) -> None:
        """Update the believed contents after a cart call went through."""
        fields = self.e.schemas[name].get("properties") or {}
        mode = _enum_part(args, fields)
        if self.add_mode is None:
            self.add_mode = mode
        # a call that sets a listing's count to zero takes it out, whatever its mode
        zeroed = any(_is_count(v) and v == 0 and isinstance(fields.get(k), dict) and fields[k].get("type") == "integer"
                     for k, v in args.items())
        adding = mode == self.add_mode and not zeroed
        listed = _basket_line_ids(args)
        if listed is not None:
            if adding:
                self.ids = listed    # the tool takes the whole basket as a list of lines
                self.lines = {i: _line_naming(args, i) for i in listed}
            else:
                self.ids = [i for i in self.ids if i not in listed]     # the same tool used to take lines out
            return
        for key, value in args.items():
            spec = fields.get(key) if isinstance(fields.get(key), dict) else {}
            if isinstance(value, str) and "enum" not in spec and value.strip() and " " not in value.strip():
                self.ids = [i for i in self.ids if i != value] + ([value] if adding else [])


# ---------------------------------------------------------------------------------------------- independent checks
class Referee:
    """Second opinions: a listing check before it goes into the cart or is ordered, and a review before a decline."""

    def __init__(self, errand: "Errand"):
        self.e = errand
        self.verdicts: dict[str, str] = {}
        self.last_fits: list[str] = []      # listings the latest decline review thinks may fit

    def check_cart(self) -> str:
        basket = self.e.basket
        if not basket.ids:
            return ""
        # several different products mean one line each: every line is checked
        for item in basket.ids[-3:] if self.e.reading.products() else basket.ids[-1:]:
            verdict = self.check_item(item, PURPOSE_ORDER, basket.lines.get(item, ""))
            if verdict:
                return verdict
        return ""

    def check_item(self, item: str, purpose: str, line: str = "") -> str:
        """A warning when the checker finds the item failing a requirement; "" when it fits or cannot be judged."""
        e = self.e
        if e.meter.seconds_left() < CHECK_ROOM_S:
            return ""
        memo = f"{item}|{line}|{e.wants.version}"
        if memo in self.verdicts:
            return self.verdicts[memo]
        records = e.archive.records_of(item)
        if not records:
            return ""
        prompt = (CHECKER_PROMPT + HEAD_SHOPPER_REQUEST + e.request + HEAD_CURRENT_REQS + e.wants.text()
                  + HEAD_ITEM_TO.format(purpose=purpose, item=item) + "\n".join(records))
        if line:
            prompt += HEAD_CART_LINE + line
        if e.basket.latest_view is not None:
            prompt += HEAD_CART_VIEW + _dump(e.basket.latest_view)[:2000]
        labels = e.archive.label_counts()
        if labels:
            prompt += HEAD_LABELS + labels
        answer = e.wire.ask_json(prompt, 1200)
        e.log("verify", item=item, result=answer)
        if not answer or answer.get("fits") is not False:
            self.verdicts[memo] = ""
            return ""
        problems = "; ".join([str(p) for p in answer.get("problems") or []][:5])
        verdict = (f"According to a separate check, {item} does NOT meet every requirement: {problems}. Unless you "
                   "are sure the check got it wrong, go with another verified listing.")
        self.verdicts[memo] = verdict
        return verdict

    def review_decline(self) -> str:
        """Listings a reviewer thinks may fit after all ("" when it names none)."""
        e = self.e
        if e.meter.seconds_left() < CHECK_ROOM_S:
            return ""
        blocks: list[str] = []
        size = 0
        for record in reversed(e.archive.records):
            text = _dump(_trim(record, 160))
            if size + len(text) <= 16000:
                blocks.append(text)
                size += len(text)
        if not blocks:
            return ""
        prompt = (DECLINE_REVIEW_PROMPT + HEAD_SHOPPER_REQUEST + e.request + HEAD_AUDIT_REQS + e.wants.text()
                  + HEAD_RECORDS + "\n".join(blocks))
        answer = e.wire.ask_json(prompt, 1200)
        e.log("audit", result=answer)
        fits = [str(x) for x in (answer or {}).get("possible_fits") or [] if isinstance(x, (str, int))][:5]
        self.last_fits = fits
        if not fits:
            return ""
        why = f" ({answer.get('reason')})" if answer.get("reason") else ""
        text = (f"A separate review thinks these listings may meet every requirement: {', '.join(fits)}{why}. "
                "Check them (details, stock, price) and order the right one; decline only if every one of them really fails.")
        if e.closer.order_attempts:
            text += (" So far nothing has been bought: a successful order ends the episode and this one is still "
                     "going, so declining now leaves the shopper empty-handed.")
        return text


# ---------------------------------------------------------------------------------------------- the shopper
_FILLER = ("and", "the", "for", "set", "with", "pair", "pack", "of")
# Question pacing taken from T.KROOS: the first round goes out as soon as the first listing details are in, and
# uses every message but one; the held-back message waits for a later round (more listings seen, or a commit).
_FIRST_ROUND_SETS = 2
_HELD_FOR_LATER = 1
# Plain questions: one property per message, in one fixed form
_QUESTION_FORM = re.compile(r"^Which (.+?) would you like the (.+?) to have\?$", re.I)
_SEVERAL_PROPERTIES = re.compile(r",|/|\b(and|or)\b", re.I)
_MENU_NAME = re.compile(r"^- (.+?) \(on \d+ listings", re.M)
# Several products: the shopper's needs are spread over them, so every product is asked about once before any is
# asked about twice, each time with its most widely carried property not yet asked
HEAD_TARGET_PRODUCT = ("\n\nASK THIS ROUND ABOUT: {product} (the product with the fewest questions so far). Take the first "
                       "property in that product's list below (the listings labelled like it) that has not been asked "
                       "about for it and that the request does not already settle.")


def _spoken_property(name: str) -> str:
    """A listing key as a shopper would say it: no trailing unit or store suffix (filter, options), no leading
    audience or footwear word before a two-word name, "closuretype" read as "closure"."""
    n = re.sub(r"\s+(?:inch|inches|mm|cm|kg|gb|tb|ml|hz|mah|oz|lbs?|grams?)$", "", name.strip())
    n = re.sub(r"(?<=\S)\s+(?:filter|options?)$", "", n)
    n = re.sub(r"^(?:(?:kids?|kid's|children's|women's|men's|womens|mens|shoes?)\s+)+(?=\S+\s+\S)", "", n)
    n = re.sub(r"\b([a-z]{4,})type\b", r"\1", n)
    n = re.sub(r"^(?:[a-z]{1,2}\s+)+(?=\S)", "", n)       # a short store prefix ("fa pattern")
    return n.strip() or name


def _spoken_product(name: str) -> str:
    """A product name as one plain noun phrase: "traditional & gaming laptops" -> "laptops", "door handles and
    locks" -> "door handles", "usb and lightning cables" -> "cables"."""
    n = re.split(r"\s+(?:with|in|for|of|that|where|which|whose|from)\s+|[,;:(]", " ".join(name.lower().split()))[0].strip()
    pair = re.match(r"^(.+?)\s+(?:and|&)\s+(.+)$", n)
    if pair and len(n.split()) <= 5:
        first, second = pair.group(1).strip(), pair.group(2).strip()
        n = second.split()[-1] if len(first.split()) == 1 and len(second.split()) >= 2 else first
    return n or name


def _name_stems(text: str) -> set[str]:
    return {w.rstrip("s") for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _FILLER}


def _stem(word: str) -> str:
    return word.rstrip("s")


class Talk:
    """Everything said with the shopper: questions planned and sent, replies read, answers owed."""

    def __init__(self, errand: "Errand"):
        self.e = errand
        self.allowance: int | None = None     # messages the request allows (None: not stated)
        self.used = 0
        self.sent: list[str] = []             # lower-cased texts of messages that went through
        self.asked: list[str] = []            # one line per planned question, with the shopper's answer
        self.planner_on = True
        self.fact_sets_at_ask = -1
        self.last_unclear = False
        self.plan_misses = 0
        self.retry_at = 0
        self.owed_reply = ""
        self.transcript: list[str] = []
        self.heard: set[str] = set()
        self.per_product: dict[str, int] = {}     # questions sent per product of a several-product request
        self.target: str | None = None

    def limit(self) -> int:
        return self.allowance if self.allowance is not None else 3

    def message_args(self, text: str) -> dict[str, Any] | None:
        """Arguments that send `text` with the message tool: its one free-text field, when nothing else is required."""
        schema = self.e.schemas.get(self.e.reading.tool("message") or "") or {}
        fields = schema.get("properties") or {}
        required = list(schema.get("required") or [])
        texts = [k for k, spec in fields.items() if _is_text_field(spec)]
        field = next((k for k in required if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not field or any(k != field for k in required):
            return None
        return {field: text}

    # ---- reading what the shopper said
    def listen(self, lines: list[str], question: str | None) -> str:
        """Read new shopper lines into the requirements; returns a note for the driver ("" when nothing new)."""
        e = self.e
        fresh = [line for line in lines if line not in self.heard]
        if not fresh:
            return ""
        self.heard.update(fresh)
        if e.meter.seconds_left() < LISTEN_ROOM_S:
            return ""
        prompt = LISTENER_PROMPT + HEAD_CONTEXT_REQUEST + e.request + HEAD_REQS_SO_FAR + e.wants.text()
        if self.transcript:
            prompt += HEAD_EARLIER + "\n".join(self.transcript[-12:])
        if question:
            prompt += HEAD_ASKED + question
        prompt += HEAD_SHOPPER_NOW + "\n".join(fresh)
        self.transcript.append((f"asked: {question} -> " if question else "") + "shopper: " + " / ".join(fresh))
        answer = e.wire.ask_json(prompt, 1500, [LIGHT_SEAT, CHECK_SEAT])
        if not answer:
            return ""
        notes = self._apply(answer)
        if notes:
            e.wants.version += 1
        reply = str(answer.get("reply") or "").strip()
        unclear = answer.get("unclear") is True and question is not None
        self.last_unclear = unclear
        if unclear and not self._names_a_product(question):
            # with several products a question that does not say which one is ambiguous: the property may matter
            notes.append("the shopper was lost because the question did not say which product it meant: if messages "
                         "remain, ask about the same property again with the product named")
        elif unclear:
            # explaining our own question rarely helps: the property asked about is most likely not a need
            notes.append("the shopper could not follow that question: neither explain nor repeat it - the property it "
                         "asked about is probably free, so if messages remain, ask about a different one (naming the product)")
        elif reply and e.reading.tool("message") and _WANTS_ANSWER.search(" ".join(fresh)):
            # only a line that asks something gets an answer; a stated need needs no confirmation
            self.owed_reply = reply
        meaning = str(answer.get("meaning") or "")
        return "Meaning: " + meaning + (" (" + "; ".join(notes) + ")" if notes else "")

    def _names_a_product(self, question: str | None) -> bool:
        several = self.e.reading.products()
        if not several:
            return True
        asked = {_stem(w) for w in str(question).lower().replace("?", " ").split()}
        return any(_stem(w) in asked for product, _ in several for w in product.lower().split()
                   if len(w) >= 3 and w not in _FILLER)

    def _apply(self, answer: dict[str, Any]) -> list[str]:
        """Fold the listener's answer into the requirements, budget and quantities; the changes, as notes."""
        wants = self.e.wants
        notes: list[str] = []
        full = answer.get("requirements_now")
        if isinstance(full, list) and all(isinstance(r, str) for r in full):
            wants.rewritten = [r for r in full if r.strip()]
        for need in answer.get("needs") or []:
            if isinstance(need, str) and need and need not in wants.revealed:
                wants.revealed.append(need)
                notes.append(f"new requirement: {need}")
        for name in answer.get("no_preference") or []:
            if isinstance(name, str) and name and name not in wants.free:
                wants.free.append(name)
        for gone in answer.get("dropped") or []:
            if isinstance(gone, str) and gone and gone not in wants.dropped:
                wants.dropped.append(gone)
                notes.append(f"no longer required: {gone}")
        budget = answer.get("new_budget")
        if _is_number(budget) and budget > 0 and budget != wants.budget:
            wants.budget = budget
            notes.append(f"budget is now {budget}")
        change = answer.get("new_quantity")
        if _is_count(change):
            change = {"product": "", "quantity": change}
        if isinstance(change, dict) and _is_count(change.get("quantity")) and change["quantity"] >= 1:
            notes += self._new_quantity(change["quantity"], str(change.get("product") or "").strip().lower())
        if answer.get("unavailable"):
            notes.append("an item is no longer for sale: remove it from the cart and leave it out of any order")
        return notes

    def _new_quantity(self, units: int, product: str) -> list[str]:
        plan = self.e.reading.plan
        if not self.e.reading.products():
            plan["quantity"] = units     # one kind of product: the plan's quantity is the one that counts
            return [f"quantity is now {units}"]
        entries = plan.get("items") if isinstance(plan.get("items"), list) else []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            label = str(entry.get("product") or "").lower()
            if product and (product in label or label in product):
                entry["quantity"] = units
                return [f"quantity is now {entry.get('product')} x{units}"]
        return []

    # ---- planned questions
    def plan_and_ask(self, force: bool = False) -> bool:
        """Plan the next question(s) from the listings seen and send them; True when something was sent."""
        e = self.e
        tool = e.reading.tool("message")
        limit = self.limit()
        if not tool or self.used >= limit:
            self.planner_on = False
            return False
        menu, fact_sets = e.archive.property_menu()
        # only a missing budget is asked before any listing is seen; other things to find out first are covered by
        # the property questions, which still come before anything is chosen
        urgent = e.reading.budget_unknown and not self.asked
        if fact_sets < _FIRST_ROUND_SETS and e.meter.step < 8 and not urgent and not force:
            return False
        if force:
            urgent = False
        # a later round waits for more listings (or a commit attempt), so it knows the earlier answers and more of the store
        if self.asked and not force and not urgent and not self.last_unclear and fact_sets < self.fact_sets_at_ask + 2:
            return False
        if not force and fact_sets < self.retry_at:
            return False
        if e.meter.steps_left() <= (6 if force else 10) or e.meter.seconds_left() < ASK_ROOM_S:
            self.planner_on = False
            return False
        several = e.reading.products()
        # several products: one at a time, each knowing which products were already asked about. One product: the
        # first round holds one message back for later; a later or forced round uses all that remain
        remaining = max(1, limit - self.used)
        if several:
            count = 1
        elif not self.asked and not force and remaining > _HELD_FOR_LATER:
            count = remaining - _HELD_FOR_LATER
        else:
            count = remaining
        self.target = None
        if several and not urgent:
            names = [name for name, _ in several]
            self.target = min(names, key=lambda n: (self.per_product.get(n, 0), names.index(n)))
            if self._top_up(self.target):
                menu, fact_sets = e.archive.property_menu()
        planned = self._plan(count, urgent, menu, fact_sets, several)
        calls, labels = self._question_calls(planned, tool, several, menu, urgent)
        if not calls:
            # a plan with nothing usable is tried once more, after another listing has been opened
            self.plan_misses += 1
            self.retry_at = fact_sets + 1
            if self.plan_misses >= 2:
                self.planner_on = False
            return False
        e.messages.append({"role": "assistant", "content": "Questions for the shopper: " + ", then ".join(l[0] for l in labels) + ".",
                           "tool_calls": calls})
        self.last_unclear = False      # spent on this round; the new answers set it again
        results = e.run_calls(calls, planned=True)
        e.add_tool_results(results)
        for index, (_, text) in enumerate(results):
            if index < len(labels):
                answer = text.split(STATUS_MARK)[0][:400]
                self.asked.append(f"- {labels[index][0]}: asked \"{labels[index][1]}\" -> {answer}")
        self.fact_sets_at_ask = fact_sets
        self.planner_on = self.used < limit
        return True

    def _plan(self, count: int, urgent: bool, menu: str, fact_sets: int, several: list[tuple[str, int]]) -> list[dict[str, Any]]:
        e = self.e
        prompt = QUESTION_PLAN_PROMPT.replace("QUESTION_COUNT", str(count)) + HEAD_SHOPPER_REQUEST + e.request
        prompt += HEAD_REQS_SO_FAR + e.wants.text()
        if several:
            prompt += HEAD_SEVERAL_PRODUCTS + "; ".join(name for name, _ in several)
            if self.target:
                prompt += HEAD_TARGET_PRODUCT.format(product=self.target)
        if self.asked:
            prompt += HEAD_ALREADY_ASKED + "\n".join(self.asked)
        if e.reading.plan.get("ask_first"):
            prompt += HEAD_LEARN_FIRST + "; ".join(str(a) for a in e.reading.plan.get("ask_first") or [])
        prompt += HEAD_MENU + (self._open_menu(menu) or MENU_EMPTY)
        answer = e.wire.ask_json(prompt, 2000, [LIGHT_SEAT, CHECK_SEAT])
        planned = [q for q in (answer or {}).get("questions") or [] if isinstance(q, dict) and str(q.get("question") or "").strip()]
        if urgent and fact_sets < 4:
            planned = planned[:1]
        # a question already sent in the same words is never sent again
        planned = [q for q in planned if str(q.get("question")).strip().lower() not in self.sent]
        return planned[:count]

    def _plain_question(self, entry: dict[str, Any], menu_names: set[str], must_learn: bool) -> str | None:
        """The question to send for one planned entry, or None to drop it. A listing property is asked in the fixed
        form about that one property; a question about several properties at once is dropped; anything else (the
        budget, a choice between two wishes) keeps its own words when it asks one thing."""
        question = str(entry.get("question")).strip()
        prop = re.sub(r"[_.]+", " ", str(entry.get("property") or "")).strip().lower()
        if must_learn:
            return question
        if not prop or _SEVERAL_PROPERTIES.search(prop) or self._settled(prop, str(entry.get("product") or "")):
            return None
        if prop in menu_names:
            form = _QUESTION_FORM.match(question)
            product = form.group(2) if form else str(entry.get("product") or self.e.reading.plan.get("product") or "").strip()
            if product:
                return f"Which {_spoken_property(prop)} would you like the {_spoken_product(product)} to have?"
        return question if question.count("?") <= 1 else None

    def _question_calls(self, planned: list[dict[str, Any]], tool: str, several: list[tuple[str, int]], menu: str,
                        urgent: bool) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
        e = self.e
        calls, labels = [], []
        menu_names = {name.strip().lower() for name in _MENU_NAME.findall(menu or "")}
        for index, entry in enumerate(planned):
            question = self._plain_question(entry, menu_names, urgent and index == 0)
            if not question or question.lower() in self.sent:
                continue
            args = self.message_args(question)
            if not args:
                continue
            args, problems = _conform(args, e.schemas[tool], "args")
            if problems:
                continue
            self._credit_product(str(entry.get("product") or ""), several)
            calls.append({"id": f"q{self.used + index + 1}s{e.meter.step}", "type": "function",
                          "function": {"name": tool, "arguments": json.dumps(args)}})
            product = str(entry.get("product") or "").strip()
            labels.append(((product + " " if product and several else "") + str(entry.get("property")), question))
        return calls, labels

    def _open_menu(self, menu: str) -> str:
        """The menu without the properties a requirement in force already names for that group's product."""
        kept, label = [], ""
        for line in (menu or "").splitlines():
            head = re.match(r'^Listings with the label "(.+?)"', line)
            if head:
                label = head.group(1)
            name = _MENU_NAME.match(line)
            if name and self._settled(name.group(1).strip().lower(), label):
                continue
            kept.append(line)
        return "\n".join(kept)

    def _settled(self, prop: str, product: str) -> bool:
        """Whether a requirement in force already names this property (for this product, when several are asked
        for): asking about it again wastes a message."""
        wants = self.e.wants
        current = wants.rewritten if wants.rewritten is not None else wants.stated + wants.revealed
        stems = _name_stems(product)
        for line in current:
            head = str(line).split(":")[0].lower()
            if re.search(r"\b" + re.escape(prop) + r"\b", head) and (not stems or not self.e.reading.products()
                                                                     or stems & _name_stems(head)):
                return True
        return False

    def _top_up(self, product: str) -> bool:
        """Before asking about a product with fewer than three opened listings, open up to four of its listings seen
        in results (one call to a tool that shows several listings, else up to two detail calls). True when any ran."""
        e = self.e
        stems = _name_stems(product)
        if not stems or e.meter.steps_left() <= 12 or e.archive.opened_of_kind(stems) >= 3:
            return False
        ids = e.archive.unopened_of_kind(stems, 4)
        if not ids:
            return False
        batch = []
        view = e.reading.tool("view")
        fields = (e.schemas.get(view or "") or {}).get("properties") or {}
        list_field = next((k for k, s in fields.items() if isinstance(s, dict) and s.get("type") == "array"), None)
        if view and list_field and len(ids) >= 2:
            args, problems = _conform({list_field: ids}, e.schemas[view], "args")
            if not problems:
                batch.append(("topup0", view, args))
        if not batch:
            details = e.reading.tool("details")
            dfields = (e.schemas.get(details or "") or {}).get("properties") or {}
            id_field = next((k for k, s in dfields.items() if _is_text_field(s)), None)
            for n, ident in enumerate(ids[:2]):
                if details and id_field:
                    args, problems = _conform({id_field: ident}, e.schemas[details], "args")
                    if not problems:
                        batch.append((f"topup{n}", details, args))
        if not batch:
            return False
        e.log("top_up", product=product, ids=ids)
        e._run_batch(batch)
        return True

    def _credit_product(self, product: str, several: list[tuple[str, int]]) -> None:
        """Count a planned question against the request's product it names (else the round's target)."""
        if not several:
            return
        stems = _name_stems(product)
        named = next((n for n, _ in several if stems & _name_stems(n)), None) or self.target
        if named:
            self.per_product[named] = self.per_product.get(named, 0) + 1

    def answer_shopper(self, text: str) -> bool:
        """Send an owed answer to a shopper's question; True when it was sent."""
        e = self.e
        tool = e.reading.tool("message")
        if (not tool or self.used >= self.limit() or e.meter.steps_left() <= 4
                or e.meter.seconds_left() < ANSWER_ROOM_S):
            return False
        args = self.message_args(text)
        if not args:
            return False
        args, problems = _conform(args, e.schemas[tool], "args")
        if problems:
            return False
        call = {"id": f"r{e.meter.step}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
        e.messages.append({"role": "assistant", "content": "Answering the shopper's question.", "tool_calls": [call]})
        e.add_tool_results(e.run_calls([call], planned=True))
        return True


# ---------------------------------------------------------------------------------------------- notes to the driver
NOTE_ASK_BUDGET = ("how much the shopper will spend at most, as the request names no price limit (leave it out if this "
                   "list already holds the same point)")
OUTLINE_HEAD = "First reading of the request (check it against the request itself):"
OUTLINE_NO_PRIORITY = ("- no cheapest/best rule was found in the request: every item meeting all requirements within the "
                       "budget is a correct pick, so spend no steps chasing the lowest price (unless the request says otherwise)")
PURPOSE_ORDER = "ORDER"
PURPOSE_ADD = "ADD TO THE CART"
STATUS_MARK = "\n[state]"
ORDER_REMARK = "Please order what is in the cart."
DECLINE_REMARK = "I could not find anything in this store that meets every one of your requirements, so no order is placed."

HOLD_RESEARCH = ("Held back (no step used). Your previous order was refused, and this request wants the cheapest item "
                 "that fits, so an earlier backup only qualifies if nothing cheaper fits today. Run the search again "
                 "first: the requirement words and the category, a price limit slightly above the refused item's price "
                 "where the search accepts one, and the following pages. Open the details of the cheaper candidates, move "
                 "the cheapest one meeting every requirement into the cart, and then order.")
HOLD_CHEAPER = ("Held back (no step used). This request wants the cheapest fitting item, yet these lower-priced listings "
                "from your results have never been opened with the details tool: {rows}. Open their details together in "
                "one reply. If any of them meets every requirement, switch the cart to the cheapest such listing; if none "
                "does, send the order again.")
HOLD_PRE_ORDER = ("Held back (no step used). One last look before ordering: (1) does the cart contain only the line(s) "
                  "this request calls for - normally a single line, the chosen item at the required quantity? (2) after "
                  "your latest cart change, did a tool that shows stock and price confirm the item is in stock and that "
                  "price x quantity fits the current budget? (3) do the item's own properties satisfy each stated "
                  "requirement, each need the shopper revealed and each announced change? (4) if the request wants the "
                  "cheapest or best, is there truly no better fitting item among all you found? (5) has every procedural "
                  "rule of the request been obeyed? Settle any open point now; if all are settled, send the order call again.")
HOLD_DECLINE = ("Held back (no step used). A decline is almost never the right finish: put the best-matching listing of "
                "each product in the cart and order it instead; decline only if the store has no listing of the "
                "requested kind at all. A decline needs certainty: have you searched several ways (different words, "
                "filters, further pages), read the own properties of each candidate, and found that each fails a stated "
                "requirement, a revealed need, the budget or stock? Close synonyms and any spelling of a key count, while "
                "a requirement you merely inferred from the story does not. Order whatever fits; send the decline tool "
                "again only if nothing does.")
NOTE_ALLOWANCE = ("This tool has no allowance left and is no longer offered; do not call it again. Read prices and stock "
                  "from search results or the cart view, and decide with what you already have.")
NOTE_REFUSED = ("ORDER REFUSED: the episode continues with the cart unchanged. Carry out every change the refusal names, "
                "correct the cart (drop any line that is no longer acceptable, a sold-out item for instance), check stock "
                "and price of the replacement pick, then order once more.")
NOTE_PRICE_ONLY = ("Nothing in this refusal says the item has gone; when all it reports is a new price (or some change the "
                   "item still satisfies), the item remains on sale. Look up its current price, and if price x quantity "
                   "still fits the budget and every requirement still holds, send the order again at once. A price change "
                   "is never a reason to decline.")
NOTE_SHOPPER_IN_RESULT = "The shopper speaks in this result (a change); act on it: "
NOTE_WAITING = ("One more step is needed to complete this order (it issued a token or wants a confirmation): use the "
                "tool that completes the checkout now. Nothing counts as bought until the episode ends.")
NOTE_NOT_ENDED = ("The episode is still running after that order call, so NOTHING HAS BEEN BOUGHT, whatever the result "
                  "claims (a successful order ends the episode). Should the result ask for one more step, such as "
                  "confirming with a token or code it supplied, do that next. Should it have been refused or have failed, "
                  "find the reason, carry out every change it names, correct the cart, check the pick's stock and price "
                  "again and re-send the order; a changed price does not by itself rule an item out while price x "
                  "quantity remains inside the budget.")
NOTE_EARLY_COMMIT = ("Not run (no step used): it is too early to choose. First open the details of two or three further "
                     "listings of the requested kind, in one reply, picking ones that differ from what you have already "
                     "seen; the shopper questions are built from those details.")
NOTE_NO_CALL = "No tool was called. Every reply needs at least one call to the supplied tools; continue.\n"


# ---------------------------------------------------------------------------------------------- finishing
class Closer:
    """Orders and declines: the checks held in front of them, and the finish the code makes itself."""

    def __init__(self, errand: "Errand"):
        self.e = errand
        self.order_attempts = 0
        self.refusals = 0
        self.refused_at = -1          # step of the latest refused order
        self.order_checked = False
        self.decline_checks = 0
        self.cheaper_checks = 0
        self.research_checks = 0

    def note_refusal(self) -> None:
        self.refusals += 1
        self.refused_at = self.e.meter.step
        self.order_checked = False

    def is_decline(self, name: str, args: dict[str, Any]) -> bool:
        """Whether a finish call ends without buying. One tool may serve both finishes, told apart by an enum value
        (for example "refuse" against "commit")."""
        reading = self.e.reading
        if name != reading.tool("decline"):
            return False
        if reading.tool("order") != name:
            return True
        fields = (self.e.schemas.get(name) or {}).get("properties") or {}
        return any(isinstance(v, str) and "enum" in (fields.get(k) or {}) and _NO_BUY_WORDS.search(v) for k, v in args.items())

    def hold(self, name: str, args: dict[str, Any]) -> tuple[str | None, dict[str, Any], str]:
        """A reason to hold this finish call back (None to send it), and the arguments and tool to send."""
        order = self.e.reading.tool("order")
        if self.is_decline(name, args) and order and self.e.basket.ids:
            # a decline is almost never the right finish: with lines in the cart, the cart is ordered instead (a
            # product the store could not supply stays out, as requests that let sold-out items be skipped want)
            self.e.log("decline_to_order", cart=list(self.e.basket.ids))
            return None, self.plain_args(order, ORDER_REMARK), order
        reason, args = self._hold(name, args)
        if reason is None and self.is_decline(name, args) and order and not self.e.basket.ids and self._fill_from_review():
            self.e.log("decline_to_order", cart=list(self.e.basket.ids), filled=True)
            return None, self.plain_args(order, ORDER_REMARK), order
        if reason is None and self.is_decline(name, args) and self._cart_confirmed():
            self.e.log("decline_to_order", cart=list(self.e.basket.ids), fits=list(self.e.referee.last_fits))
            return None, self.plain_args(order, ORDER_REMARK), order
        return reason, args, name

    def _fill_from_review(self) -> bool:
        """An empty cart about to be declined: put the review's first possible fit for each product in the cart
        (matched to products by their listing's kind label) at the requested quantity. True when lines went in."""
        e = self.e
        fits = list(e.referee.last_fits)
        cart = e.reading.tool("cart")
        if not fits or not cart or e.meter.steps_left() < 2:
            return False
        several = e.reading.products()
        wanted = e.reading.plan.get("quantity")
        chosen: dict[int, tuple[str, int]] = {}
        for ident in fits:
            if several:
                label = e.archive.label_of(ident)
                slot = next((i for i, (n, _) in enumerate(several) if label and _name_stems(label) & _name_stems(n)), None)
                if slot is None or slot in chosen:
                    continue
                chosen[slot] = (ident, several[slot][1])
            else:
                chosen[0] = (ident, wanted if _is_count(wanted) and wanted >= 1 else 1)
                break
        batch = []
        for slot, (ident, units) in sorted(chosen.items()):
            args = e.basket.add_args(ident, units)
            if args is not None:
                batch.append((f"fill{slot}", cart, args))
        batch = batch[: max(0, e.meter.steps_left() - 1)]
        if not batch:
            return False
        e._run_batch(batch)
        return bool(e.basket.ids)

    def _cart_confirmed(self) -> bool:
        ids, fits = self.e.basket.ids, set(self.e.referee.last_fits)
        return bool(self.e.reading.tool("order") and ids and fits and set(ids) <= fits)

    def _hold(self, name: str, args: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        e, meter = self.e, self.e.meter
        declining = self.is_decline(name, args)
        buying = not declining and name == e.reading.tool("order")
        cheapest = e.reading.priority() == "cheapest"
        roomy = meter.steps_left() > 10 and meter.seconds_left() > SEARCH_ROOM_S
        # after a refused order a backup picked earlier may no longer be the cheapest fit: search once more first
        if buying and cheapest and self.refused_at > e.archive.searched_at and self.research_checks < 1 and roomy:
            self.research_checks += 1
            return HOLD_RESEARCH, args
        if buying and self.cheaper_checks < 2 and roomy:
            rows = e.archive.unopened_cheaper()
            if rows:
                self.cheaper_checks += 1
                return HOLD_CHEAPER.format(rows=", ".join(f"{i} ({p})" for i, p in rows)), args
        if buying and not self.order_checked and meter.steps_left() > 2 and meter.seconds_left() > CHECK_ROOM_S:
            self.order_checked = True
            verdict = e.referee.check_cart()
            return ("Held back (no step used). " + verdict if verdict else HOLD_PRE_ORDER), args
        if declining:
            # declines are rarely right: each of the first two attempts gets an independent look for fitting listings
            if self.decline_checks < 2 and meter.steps_left() > 4 and meter.seconds_left() > CHECK_ROOM_S:
                self.decline_checks += 1
                review = e.referee.review_decline()
                if review:
                    return "Held back (no step used). " + review, args
                if self.decline_checks == 1:
                    return HOLD_DECLINE, args
            return None, self.scrub_decline(name, args)
        return None, args

    def scrub_decline(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """A decline remark that must not point to any listing is replaced by a neutral one."""
        e = self.e
        if e.reading.plan.get("decline_no_names") is False:
            return args
        names_listing = any(token and token in e.archive.seen_ids for value in args.values() if isinstance(value, str)
                            for token in re.split(r"[\s.,;:!?()\[\]{}<>\"'`]+", value))
        if not (e.reading.plan.get("decline_no_names") or names_listing):
            return args
        fields = (e.schemas.get(name) or {}).get("properties") or {}
        scrubbed = {}
        for key, value in args.items():
            spec = fields.get(key) or {}
            scrubbed[key] = DECLINE_REMARK[: spec.get("maxLength", len(DECLINE_REMARK))] if isinstance(value, str) and "enum" not in spec else value
        return scrubbed

    def plain_args(self, name: str, text: str, declining: bool = False) -> dict[str, Any]:
        """Arguments for a finish the code sends itself: every required field filled, the mode matching the intent."""
        schema = self.e.schemas.get(name) or {}
        fields = schema.get("properties") or {}
        args: dict[str, Any] = {}
        for key in schema.get("required") or []:
            spec = fields.get(key) or {}
            kind = spec.get("type")
            if "enum" in spec and spec["enum"]:
                matching = [v for v in spec["enum"] if isinstance(v, str) and bool(_NO_BUY_WORDS.search(v)) == declining]
                args[key] = (matching or spec["enum"])[0]
            elif kind == "string" or "string" in (kind or []):
                args[key] = text[: spec.get("maxLength", len(text))]
            elif kind == "boolean":
                args[key] = True
            elif kind in ("integer", "number"):
                args[key] = max(1, spec.get("minimum", 1))
        if len(fields) == 1:
            for key, spec in fields.items():
                if key not in args and isinstance(spec, dict) and spec.get("type") == "string" and "enum" not in spec:
                    args[key] = text[: spec.get("maxLength", len(text))]
        return args

    def finish_now(self) -> None:
        """End the episode from code: order a used cart, else decline. A refusal for a new price alone gets one
        more order; any other refusal with a reason ends in a decline."""
        e, meter = self.e, self.e.meter
        if e.finished or e.dead or meter.steps_left() <= 0 or meter.seconds_left() < 2:
            return
        order, decline = e.reading.tool("order"), e.reading.tool("decline")
        if order and e.basket.used:
            action = {"name": order, "args": self.plain_args(order, ORDER_REMARK)}
        elif decline:
            action = {"name": decline, "args": self.plain_args(decline, DECLINE_REMARK, declining=True)}
        else:
            return
        reply = self._send_finish(action)
        seen = (reply or {}).get("observation")
        seen = seen if isinstance(seen, dict) else {}
        parts = [v for v in seen.values() if isinstance(v, (list, str))]
        parts += [x for v in seen.values() if isinstance(v, dict) for x in v.values() if isinstance(x, (list, str))]
        reasons: list[str] = []
        _wordy_strings(parts, reasons)
        joined = " ".join(reasons)
        if (reply and not reply.get("finished") and reasons and order and action["name"] == order and e.basket.used
                and _MONEY_TALK.search(joined) and not _SOLD_OUT.search(joined) and meter.steps_left() > 0):
            reply = self._send_finish(action)     # refused only for a new price: the cart still holds a good item
        if (reply and not reply.get("finished") and isinstance(reply.get("observation"), dict) and reasons and decline
                and meter.steps_left() > 0):
            self._send_finish({"name": decline, "args": self.plain_args(decline, DECLINE_REMARK, declining=True)})

    def _send_finish(self, action: dict[str, Any]) -> dict[str, Any] | None:
        reply = self.e.wire.act([action])
        self.e.log("forced_finish", action=action, reply=reply)
        return reply


# ---------------------------------------------------------------------------------------------- the episode
class Errand:
    """One episode: the parts above, the screening and running of tool calls, and the driver loop."""

    def __init__(self, problem_data: dict[str, Any]):
        environment = problem_data["environment"]
        self.binding = dict(environment["binding"])
        self.policy = environment["policy_view"]
        self.request = str(self.policy.get("query") or "")
        self.opening = self.policy.get("opening")
        self.tools = [t for t in self.policy.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict)]
        self.schemas = {t["function"]["name"]: t["function"].get("parameters") or {"type": "object", "properties": {}}
                        for t in self.tools}
        self.finished = bool(self.policy.get("finished"))
        self.dead = False
        self.trace: list[dict[str, Any]] = []
        self.messages: list[dict[str, Any]] = []
        self.meter = Meter(int(self.policy.get("max_steps") or 30))
        self.wire = Wire(self)
        self.reading = Reading(self)
        self.wants = Wants(self)
        self.archive = Archive(self)
        self.basket = Basket(self)
        self.talk = Talk(self)
        self.referee = Referee(self)
        self.closer = Closer(self)
        self.recent: list[str] = []         # signatures of the last two plain calls (repeat guard)
        self.capped: set[str] = set()       # tools whose allowance is spent
        self.early_bounced = False

    def log(self, kind: str, **fields: Any) -> None:
        self.trace.append({"kind": kind, **fields})

    # ---- what the driver sees
    def status(self) -> str:
        meter, closer, talk = self.meter, self.closer, self.talk
        left = meter.steps_left()
        sent = f"messages sent {talk.used}" + (f" of {talk.allowance}" if talk.allowance is not None else "")
        text = (f"{STATUS_MARK.strip()} {meter.step} of {meter.step_cap} steps used, {left} left; {sent}; "
                f"about {int(max(0, meter.seconds_left() - STOP_MARGIN_S))} s left.")
        text += "\n[requirements now] " + self.wants.text()
        if closer.refusals and not self.finished and meter.step - closer.refused_at >= 2:
            # a refused order is easily misread as a placed one; nothing is bought until the episode ends
            text += (f" NOTHING HAS BEEN BOUGHT YET: the episode kept running after the order attempt {meter.step - closer.refused_at} "
                     "steps back. With fitting items within the current budget in the cart, send the order tool now; "
                     "otherwise repair the cart and then order (decline only when nothing in the store fits any longer).")
        if left <= 2:
            text += " FINAL STEPS: with a verified fit in the cart, order now; without one, decline."
        elif left <= 5 or meter.seconds_left() < CLOSE_MARGIN_S:
            text += (" FINISH NOW: reduce the cart to the verified line(s) the request calls for, check stock once more, "
                     "then order (or decline if nothing fits).")
        elif closer.order_attempts == 0 and left <= 14:
            text += (" Steps are getting scarce: in the next few steps, place your best verified fit in the cart and "
                     "order. Searching and detail views close when 7 steps remain, and fixing a refused order costs about 4 steps.")
        return text

    def offered_tools(self) -> list[dict[str, Any]]:
        """The tools offered this turn: near the end only the finishing ones (and a checkout step no role covers)."""
        role = self.reading.tool
        left = self.meter.steps_left()
        order, decline, cart = role("order"), role("decline"), role("cart")
        if left <= 1:
            keep = {order, decline}
        elif left <= 3:
            keep = {order, decline, cart}
        elif left <= 7 and (self.closer.order_attempts == 0 or self.closer.refusals > 0):
            keep = {order, decline, cart, role("cart_view"), role("view")}
        else:
            keep = None
        tools = self.tools
        if keep is not None and order and cart:
            covered = set(self.reading.roles.values())
            keep |= {t["function"]["name"] for t in self.tools
                     if t["function"]["name"] not in covered and _CHECKOUT_STEP.search(t["function"]["name"])}
            tools = [t for t in self.tools if t["function"]["name"] in keep] or tools
        if self.talk.planner_on and role("message"):
            tools = [t for t in tools if t["function"]["name"] != role("message")]
        if self.capped:
            tools = [t for t in tools if t["function"]["name"] not in self.capped] or tools
        return tools

    def add_tool_results(self, results: list[tuple[str, str]]) -> None:
        for index, (call_id, text) in enumerate(results):
            if index == len(results) - 1:
                text += "\n" + self.status()
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})

    # ---- screening calls before they cost a step
    def _screen(self, call: dict[str, Any], allowed: set[str], planned: bool, batch: int,
                queued_messages: int) -> tuple[str | None, str, dict[str, Any]]:
        """(reason not to run it or None, tool name, arguments)."""
        role = self.reading.tool
        function = call.get("function") or {}
        name, raw = function.get("name"), function.get("arguments")
        finishing = {role("order"), role("decline")} - {None}
        if name not in self.schemas:
            return f"Not run (no step used): there is no tool called '{name}'. Use one of: {', '.join(sorted(allowed))}.", name, {}
        if not planned and name not in allowed:
            if name == role("message") and self.talk.planner_on:
                why = "questions to the shopper are planned and sent for you once you have opened a few listings' details"
            else:
                why = "too few steps remain for this; the tools usable now are: " + ", ".join(sorted(allowed))
            return f"Not run (no step used): {why}.", name, {}
        if isinstance(raw, dict):
            args = raw
        else:
            args = _top_object(raw or "{}") if (raw or "").strip() else {}
            if args is None:
                return "Not run (no step used): the arguments are not a valid JSON object.", name, {}
        args, problems = _conform(args, self.schemas[name], "args")
        if problems:
            return "Not run (no step used): the arguments are invalid: " + "; ".join(problems[:6]), name, args
        if name in finishing and batch > 1:
            return ("Not run (no step used): send an order or a decline on its own, after you have seen the results of "
                    "your other calls."), name, args
        if name == role("message"):
            text = _string_values(args)
            if (not planned and "?" not in text
                    and not any(_TELLING_RULE.search(str(rule)) for rule in self.reading.plan.get("rules") or [])):
                return ("Not run (no step used): this message contains no question, and shopper messages are reserved "
                        "for questions. Findings, choices and plans are not reported to the shopper; the finishing tool "
                        "covers that."), name, args
            if text.strip().lower() in self.talk.sent:
                return ("Not run (no step used): you already sent this very message, and the answer is above. Ask "
                        "about a different property, or move on."), name, args
            if self.talk.allowance is not None and self.talk.used + queued_messages >= self.talk.allowance:
                return "Not run (no step used): the message allowance is spent. Do not message the shopper again.", name, args
        cart = role("cart")
        if name not in finishing and name != role("message") and name != cart and name + _dump(args) in self.recent:
            return ("Not run (no step used): this is the same call you just made, and nothing has changed since; "
                    "the result is above. Use other arguments or take another action."), name, args
        if name == cart:
            held = self._screen_cart(name, args)
            if held:
                return held, name, args
        return None, name, args

    def _screen_cart(self, name: str, args: dict[str, Any]) -> str | None:
        basket = self.basket
        # one product: an add is checked once per requirements version (several products are checked at the order)
        if self.meter.steps_left() > 3 and not self.reading.products():
            fields = self.schemas[name].get("properties") or {}
            if basket.add_mode is None or _enum_part(args, fields) == basket.add_mode:
                target = basket.target(args)
                memo = f"{target}|{self.wants.version}"
                if target and memo not in basket.vetted:
                    basket.vetted.add(memo)
                    verdict = self.referee.check_item(target, PURPOSE_ADD, _line_naming(args, target))
                    if verdict:
                        return "Not run (no step used). " + verdict + " If you are sure it fits, send the same call again."
        issue = basket.quantity_issue(name, args)
        signature = name + _dump(args)
        if issue and signature not in basket.warned:
            basket.warned.add(signature)
            return "Not run (no step used): " + issue + " If you are sure, send the same call again."
        return None

    # ---- running calls
    def run_calls(self, calls: list[dict[str, Any]], planned: bool = False) -> list[tuple[str, str]]:
        """Screen the calls, hold back a finish that needs a check first, run the rest; one reply per call."""
        role = self.reading.tool
        allowed = {t["function"]["name"] for t in self.offered_tools()}
        finishing = {role("order"), role("decline")} - {None}
        replies: list[tuple[str, str]] = []
        runnable: list[tuple[str, str, dict[str, Any]]] = []
        queued_messages = 0
        for call in calls:
            reason, name, args = self._screen(call, allowed, planned, len(calls), queued_messages)
            if reason is not None:
                replies.append((call["id"], reason))
                continue
            if name == role("message"):
                queued_messages += 1
            runnable.append((call["id"], name, args))
        if runnable and runnable[0][1] in finishing:
            call_id, name, args = runnable[0]
            reason, args, name = self.closer.hold(name, args)
            if reason is not None:
                replies.append((call_id, reason))
                return replies
            runnable[0] = (call_id, name, args)
        if runnable and runnable[0][1] not in finishing and finishing:
            room = max(0, self.meter.steps_left() - 1)
            if room < len(runnable):
                replies += [(call_id, "Not run (no step used): a single step is left, and it is reserved for the "
                             "order (or the decline).") for call_id, _, _ in runnable[room:]]
                runnable = runnable[:room]
        for start in range(0, len(runnable), 16):
            replies += self._run_batch(runnable[start:start + 16])
        return _sorted_by_order(replies, calls)

    def _run_batch(self, batch: list[tuple[str, str, dict[str, Any]]]) -> list[tuple[str, str]]:
        if self.finished or self.dead or self.meter.steps_left() <= 0:
            return [(call_id, "Not run: the episode is over or no steps remain.") for call_id, _, _ in batch]
        batch = batch[: self.meter.steps_left()]
        actions = [{"name": name, "args": args} for _, name, args in batch]
        first_step = self.meter.step + 1
        reply = self.wire.act(actions)
        self.log("env", step=first_step, actions=actions, reply=reply)
        if reply is None:
            return [(call_id, "The call failed (network or server). Try it again.") for call_id, _, _ in batch]
        observations = reply.get("observations") if isinstance(reply.get("observations"), list) else [reply.get("observation")]
        out = []
        for index, (call_id, name, args) in enumerate(batch):
            if index >= len(observations):
                out.append((call_id, "Not run: the episode finished before this call."))
                continue
            note = self._absorb(name, args, observations[index], reply)
            text = _dump(observations[index])
            if index == len(observations) - 1 and reply.get("finished"):
                text += " [the episode is finished]"
            out.append((call_id, text + ("\n[note] " + note if note else "")))
        return out

    def _absorb(self, name: str, args: dict[str, Any], observation: Any, reply: dict[str, Any]) -> str:
        """Book one call's result into the episode's state; the note the driver gets with it."""
        role = self.reading.tool
        message_tool, cart, order = role("message"), role("cart"), role("order")
        finishing = {order, role("decline")} - {None}
        is_dict = isinstance(observation, dict)
        if name == message_tool:
            self.talk.used += 1
            text = _string_values(args).strip().lower()
            if text and not (is_dict and observation.get("error")):
                self.talk.sent.append(text)      # a message that failed may be sent again
        if name == order:
            self.closer.order_attempts += 1
        if name == cart:
            self.basket.used = True
            self.closer.order_checked = False
        if name in (cart, message_tool) or name in finishing:
            self.recent = []
        else:
            self.recent = (self.recent + [name + _dump(args)])[-2:]
        self.archive.keep(observation)
        if name == cart:
            self.basket.record(name, args)
        if name == role("search"):
            self.archive.searched_at = self.meter.step
        if name in (role("details"), role("view")):
            self.archive.opened.update(v for v in args.values() if isinstance(v, str))
            self.archive.opened.update(x for v in args.values() if isinstance(v, list) for x in v if isinstance(x, str))
        if name == role("cart_view"):
            self.basket.latest_view = observation
        notes = [self._observe(name, observation)]
        refused = is_dict and "refusal" in observation
        if name == order and name != role("decline") and not reply.get("finished") and not refused:
            if is_dict and _waiting_checkout(observation):
                notes.append(NOTE_WAITING)
            else:
                # the order did not end the episode: refused in other words, failed, or waiting for another step
                self.closer.note_refusal()
                notes.append(NOTE_NOT_ENDED)
        lines, question = self._shopper_lines(name, args, observation)
        if lines:
            notes.append(self.talk.listen(lines, question))
        return " ".join(n for n in notes if n)

    def _observe(self, name: str, observation: Any) -> str:
        """Notes on a result: a spent allowance, a refused order, a shopper line inside the result."""
        if isinstance(observation, list):
            _gather_ids(observation, self.archive.seen_ids)
        if not isinstance(observation, dict):
            return ""
        role = self.reading.tool
        _gather_ids(observation, self.archive.seen_ids)
        notes = []
        problem = " ".join(str(observation.get(k) or "") for k in ("error", "detail", "message", "reason")
                           if isinstance(observation.get(k), str))
        core = {role(r) for r in ("order", "decline", "cart", "cart_view")} - {None}
        if observation.get("error") and _ALLOWANCE_GONE.search(problem) and name not in core:
            if name == role("message"):
                talk = self.talk
                talk.allowance = talk.used if talk.allowance is None else min(talk.allowance, talk.used)
            self.capped.add(name)
            notes.append(NOTE_ALLOWANCE)
        if "refusal" in observation:
            self.closer.note_refusal()
            notes.append(NOTE_REFUSED)
            refusal = observation.get("refusal")
            said = refusal if isinstance(refusal, str) else _dump(refusal)
            if _MONEY_TALK.search(said) and not _SOLD_OUT.search(said):
                notes.append(NOTE_PRICE_ONLY)
        elif len(observation) >= 2 and name != role("message"):
            lines: list[str] = []
            for value in observation.values():
                if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                    _wordy_strings(value, lines)
            if lines:
                notes.append(NOTE_SHOPPER_IN_RESULT + " | ".join(lines[:4]))
        return " ".join(notes)

    def _shopper_lines(self, name: str, args: dict[str, Any], observation: Any) -> tuple[list[str], str | None]:
        """What the shopper said in this result, and the question it answers (None outside a message)."""
        heard: list[str] = []
        if name == self.reading.tool("message"):
            if isinstance(observation, str):
                heard = [observation]
            elif isinstance(observation, (dict, list)) and not (isinstance(observation, dict) and "error" in observation):
                if isinstance(observation, dict):
                    _wordy_strings([v for v in observation.values() if isinstance(v, (str, list))], heard)
                if not heard and _has_words(observation):
                    heard = [_dump(observation)[:600]]     # a short or structured answer is still the answer
            return heard, _string_values(args)
        if isinstance(observation, dict):
            wrapper = len(observation) <= 3 and any(isinstance(v, (dict, list)) for v in observation.values())
            for value in observation.values():
                if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                    _wordy_strings(value, heard)
                elif isinstance(value, str) and len(value.split()) >= 4 and "refusal" in observation:
                    heard.append(value)
                elif (isinstance(value, str) and wrapper and len(value.split()) >= 6
                      and (_FIRST_PERSON.search(value) or name == self.reading.tool("order"))):
                    heard.append(value)     # a line wrapped around the real result, as a plain string
        return heard, None

    # ---- the driver loop
    def opening_message(self) -> str:
        text = "SHOPPER REQUEST:\n" + self.request
        if self.opening is not None:
            text += "\n\nFIRST OBSERVATION FROM THE STORE:\n" + _dump(self.opening)
        outline = self.reading.outline()
        if outline:
            text += "\n\n" + outline
        return text + "\n\n" + self.status()

    def _driver_calls(self, message: dict[str, Any], numbered: int) -> tuple[list[dict[str, Any]], int]:
        calls = []
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            numbered += 1
            function = call.get("function") or {}
            arguments = function.get("arguments")
            calls.append({"id": call.get("id") or f"c{numbered}", "type": "function",
                          "function": {"name": function.get("name"),
                                       "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments or {})}})
        return calls, numbered

    def _bounce_early_commit(self, content: str, calls: list[dict[str, Any]]) -> bool:
        """A commit before enough listings were opened is sent back once: the questions are planned from them."""
        meter = self.meter
        if (self.early_bounced or self.archive.property_menu()[1] >= 3 or meter.steps_left() <= 12
                or meter.seconds_left() <= CLOSE_MARGIN_S + 60):
            return False
        self.early_bounced = True
        self.messages.append({"role": "assistant", "content": content or None, "tool_calls": calls})
        self.add_tool_results([(c["id"], NOTE_EARLY_COMMIT) for c in calls])
        return True

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.reading.perform()
        self.log("plan", plan=self.reading.plan, roles=self.reading.roles)
        first = self.opening_message()
        if self.opening is not None:
            self.archive.keep(self.opening)
            _gather_ids(self.opening, self.archive.seen_ids)
        self.messages = [{"role": "system", "content": DRIVER_BRIEFING}, {"role": "user", "content": first}]
        idle = failures = numbered = 0
        role = self.reading.tool
        for _turn in range(TURN_CAP):
            if self.finished or self.dead or self.meter.steps_left() <= 0:
                break
            if self.meter.seconds_left() < STOP_MARGIN_S:
                self.closer.finish_now()
                break
            if self.talk.owed_reply:
                owed, self.talk.owed_reply = self.talk.owed_reply, ""
                if self.talk.answer_shopper(owed):
                    continue
            if self.talk.planner_on and self.talk.plan_and_ask():
                continue
            message = self.wire.ask(self.messages, self.offered_tools(), [DRIVER_SEAT, FALLBACK_SEAT, DRIVER_SEAT], 2500)
            if message is None:
                failures += 1
                if failures >= 2:
                    self.closer.finish_now()
                    break
                continue
            calls, numbered = self._driver_calls(message, numbered)
            content = message.get("content") or ""
            if not isinstance(content, str):
                content = _dump(content)
            committing = {role("cart"), role("order"), role("decline")} - {None}
            if self.talk.planner_on and any(c["function"]["name"] in committing for c in calls):
                if self._bounce_early_commit(content, calls):
                    continue
                if self.talk.plan_and_ask(force=True):
                    continue
            if not calls:
                idle += 1
                self.messages.append({"role": "assistant", "content": content or "(no tool call)"})
                self.messages.append({"role": "user", "content": NOTE_NO_CALL + self.status()})
                if idle >= 3:
                    self.closer.finish_now()
                    break
                continue
            idle = 0
            self.log("model", content=content[:2000], calls=[c["function"] for c in calls])
            self.messages.append({"role": "assistant", "content": content or None, "tool_calls": calls})
            self.add_tool_results(self.run_calls(calls))
        if not self.finished and not self.dead:
            self.closer.finish_now()
        self.log("end", finished=self.finished, steps=self.meter.step, cost=round(self.meter.spend, 5),
                 seconds=round(self.meter.seconds, 1))
        return self.trace


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    errand = Errand(problem_data)
    try:
        return errand.run()
    except Exception as exc:
        errand.log("crash", error=f"{type(exc).__name__}: {exc}"[:500])
        try:
            if not errand.finished and not errand.dead:
                errand.closer.finish_now()
        except Exception as again:
            errand.log("crash", error=f"{type(again).__name__}: {again}"[:500])
        return errand.trace


__all__ = ["agent_main"]
