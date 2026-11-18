"""Sieve: a shopping agent where code decides every step and models only read.

Four parts work over one shared table of listings:
  Interview  - a belief per (item, property). Each message names several properties in belief order, and
               each reply updates every name by how likely it was to be heard at its place in the message.
  Hunt       - searches and opens listings in the order the item's pick rule needs, until the fitting set
               is good enough or its steps run out.
  Verdicts   - one cached yes / no / unknown per (listing, requirement), each yes backed by a quote that code
               checks against the record. A change in the task only adds requirements to judge.
  Finish     - estimates the chance that ordering or ending without a purchase is right, and takes the better.

Version 6, from version 5's race (where most model requests failed and almost every task ended with nothing) and from
version 3's race read beside the race leader's runs on the same tasks:
  - A model request that fails in a way that passes (a rate limit, an overloaded or failing provider, a timeout) is sent
    again after a short, uneven pause, while the episode's clock and a total pause allowance permit. A request refused
    for lack of credit drops the heavy model, since a cheap request can still fit the credit left; a refusal of the
    light model stops all model requests, and code finishes alone.
  - The first reading of the request is tried again until a set time, and a reading that never comes leaves the store's
    tools found by their names and schemas, so an episode still searches, carts and orders.
  - Every listing an order holds is opened first: orders holding a listing never opened all failed.
  - The interview ranks what the request asks to check with the shopper first, then what the story points at, then the
    properties a buyer of that kind of product chooses between, among the keys most listings carry; a message names at
    most three properties, since the shopper mostly answers the first.
  - A message to the shopper for a procedure goes out only when the request asks for that message.
  - A purpose the request insists on, or pairs with a value it rules out, is a firm requirement, not a soft one.
  - Ending without a purchase needs a longer search first: the leader ordered and passed on a sixth of the tasks this
    agent ended without one.
  - The agent plans to use more of the time a task has, so the search after the interview is not cut short.

Version 5, from version 3's race (two runs of ninety tasks, read beside the leaders' runs on the same tasks):
  - A plus sign that closes a word or a number stays part of it in every comparison: a value with it names
    something other than the same value without it, and a yes never rests on the value without it.
  - A requirement whose value is to be read from another listing describes that listing by some of its values.
    Those values are that listing's, not needs of this product: a need that only repeats them is dropped, so they
    neither steer searches nor rule listings out.

Version 4, from the qualifying runs of version 3 and about three hundred other agents' episodes on the same tasks:
  - A requirement is decided by the listing attributes whose key names its property, alone or with words naming a
    part or kind of the product. When such keys disagree, the one named for this kind of product decides, else the
    bare name. Only a record with no such attribute is decided by its title or description, since stores leave
    attributes out of some records. The variant on offer, a search filter or an attribute of another property never
    decides: orders of listings whose value stood only there failed. Code decides what spelling alone can settle,
    with the words of a value together and in order, and checks every model verdict against what it names.
  - A requirement that rules a value out is not met when any such attribute holds the value.
  - A purpose read from the request is soft: requests tell plans that no attribute settles. It orders the fits and
    is searched for only when the store files such a value, and never blocks an order nor justifies an ending.
  - When the request names the category the product comes from, listings of other categories are never taken.
  - The interview asks about a property it has not asked about before one it has.
  - A line whose pick is the cheapest is not finished before its planned searches have run.

Version 3, from all sixteen races of the current suite (races 8-16 read in full):
  - Questions adapt to the shopper: the first message names two properties of one item; a reply that takes it widens
    later messages to four, and a reply asking for one thing at a time narrows every later one to one. Some races' shoppers answer more for a wider message, others
    refuse it, and the first reply tells which.
  - A store's tool that shows several listings at once reads them in one step, when its records carry the attributes.
  - Ending without a purchase needs evidence: every listing read for some item is contradicted after a full search.
    Across those races, where some leader ended without a purchase, orders passed three times as often.
  - Requirements that point at another listing may ask for a compatible value as well as the same, a different or a
    partner one.

Version 2, from the first race this agent ran:
  - One inference key pays for every task of a run, and tasks it cannot pay for score nothing, so each task spends
    less, and the light model reads listings.
  - Each message asks about one property: shoppers can refuse a message that names several.
  - A store refusing an ending turns the finish to the best cart known: almost no ending without a purchase was right.
  - A cart is read back after it is set, and lines nobody wants are taken out, with a zero quantity when the store's
    only cart tool sets quantities.
  - The wanted values are always searched by keyword, even when a browse of the category returned rows, and a
    keyword search that matches nothing is tried again with fewer words.
  - The first reading of a long request has room for its answer.
  - The heavy model, about twenty times the light one's price, reads only early and only when its likely cost fits;
    exploring stops at a set spend, but light reads go on so the episode can still verify, follow changes and finish.
"""
from __future__ import annotations

import json
import random
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

LIGHT = "anthropic/claude-haiku-5.5"
HEAVY = "anthropic/claude-sonnet-5.5"
# Models to fall back on when the proxy refuses one (a run's list of allowed models can change). A refusal that names the
# allowed models puts those first.
SPARES = {LIGHT: ["openai/gpt-6-luna"], HEAVY: ["anthropic/claude-sonnet-5"]}

# One key pays for all the tasks of a run, about $5 for 90 tasks, so what one task spends is taken from the others: a
# run that runs out leaves its last tasks unscored. The heavy model costs about twenty times the light one per token,
# so the heavy reads are what the budget limits; light reads of replies, notes and listings cost little each, and an
# episode that cannot make them can no longer verify a fit or follow a change, so they go on after exploring stops.
COST_SOFT = 0.045      # a heavy read starts only below this spend, and only if its likely cost stays under COST_STOP
COST_STOP = 0.055     # above this spend, searching and reading more listings stop and the finish begins
COST_HARD = 0.065     # no model call above this spend: code finishes alone
MIN_TOKENS = 2000     # least room a read gets: the models reason first, and what they reason uses this room too
HEAVY_IN = 2e-6       # rough heavy-model price per prompt token, for the estimate made before a heavy read
HEAVY_OUT = 1e-5      # rough heavy-model price per answer token
DEADLINE = 265.0      # seconds this agent plans to use of the 300 s a task has
HURRY = 75.0          # with less time left than this, stop hunting and head for the finish
LAST_MODEL = 25.0     # with less time left than this, no model call starts
# A failed model request is sent again when its failure passes with time: a rate limit, a provider that is overloaded or
# failing, a request that timed out. Pauses grow and are uneven, so many episodes refused at once do not all come back
# at the same moment. An episode pauses at most WAIT_TOTAL seconds in all.
WAITS = (3.0, 8.0, 15.0)
WAIT_TOTAL = 40.0
PASSING = {408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529}
# The first reading of the request is tried again, a pause apart, until this many seconds into the episode.
REQUEST_BY = 100.0
REQUEST_GAP = 8.0

# Rough settings measured on one race: how likely a held preference is to be answered at each place in a
# message. They only set the order of later questions; a different world costs at most some efficiency.
HEARD = (0.9, 0.75, 0.55, 0.35, 0.3, 0.2, 0.2, 0.15)
# How many properties a message names. Across races, a wider message brought a preference to light more often where the
# shopper took it (about twice as often for 5 to 8 names) and much less often where the shopper asked for one thing at a
# time; that answer comes at once, so the first message names a few and a refusal narrows every later one to one. Names
# far back in a message were rarely answered, so a few carry most of the gain.
# Version 6 names at most three: a shopper answering one property per message mostly answers the first, and in later
# races the fourth name of a message was rarely the one answered.
MAX_NAMES = 3
# The first message names two: where shoppers asked for one thing at a time, two names still found preferences about as
# often as one, and where they took more, two already beat one. A reply that takes it widens later messages to MAX_NAMES.
FIRST_NAMES = 2
MIN_BELIEF = 0.02
# A name already asked counts this much less for each time it was: in the qualifying runs a property asked alone was
# answered whenever it held a preference, so one a reply passed over is less likely to hold one than one never asked.
REPEAT = 0.5
# A stated requirement that no listing read can speak to is set aside only after this many records: a store files a
# requirement's attribute on some listings only, and setting a real one aside orders a listing that fails it.
SET_ASIDE_AFTER = 8
# A purpose read from the request is soft: requests tell stories of plans, and a reading of one as a requirement is often
# the story, which no attribute settles. A soft need orders the fits and keeps the hunt going for a listing that meets
# it, but never blocks an order nor justifies ending without one. These are its factors in a listing's score.
SOFT_UNKNOWN = 0.9
SOFT_NO = 0.6

FIT_SCORE = 0.85      # chance an order is right when every requirement of a line has a checked yes
# Rough settings. Across this agent's own practice runs, picks with an open requirement or read only as a row were
# never acceptable, so an order needs a verified fit: these stay below the chance of ending without a purchase.
UNKNOWN_SCORE = 0.3   # one requirement the records leave open: in the races, a best guess beat an ending
                      # without a purchase about three to one on the same tasks
NO_SCORE = 0.01       # one requirement the records contradict
UNREAD_SCORE = 0.02   # a listing seen only as a search row
# Chances that ending without a purchase is right. Rough settings: in one race only about 5% of such endings were.
END_BASE = 0.02       # every line has a fit
END_OPEN = 0.005      # some line has no fit, and nothing shows that none can fit: below a contradicted pick, so only an
                      # exhausted search justifies ending without a purchase (version 6; it was 0.01)
END_STATED = 0.1      # exhausted search, and every listing read is contradicted by the request's own requirements
END_TOLD = 0.2        # exhausted search, and every listing read is contradicted, a revealed requirement among them
# Searches an item needs before its search counts as exhausted. In version 3's race the agent ended without a purchase
# on about a quarter of the tasks, a few dozen steps in, and only 1 in 23 of those endings was right, while the leader
# ordered on the same tasks and passed one in six: those endings came before the fitting listing was found.
END_SEARCHES = 5
# The chance that an order short of a line is right, against which a line's unverified pick is weighed. In a later
# race, every leader order that said it left an item out failed, as did almost all endings without a purchase, so a
# short cart is worth nothing and a line takes its best guess, as before.
LEFT_OUT = 0.0

_llm = ProxyClient(timeout=45, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=50, max_retries=3, retry_delay=1)

READ_REQUEST = """TASK read_request. An agent serves one shopper in an online store and acts only through the store's tools. Read the request and the tools. Return ONE JSON object, nothing else:
{"items": [{"name": "the kind of product in the request's words", "head": "the one word of the name that says what the product is", "shelf": "the store category name the request uses for it, or empty", "quantity": integer, "pick": "any" | "cheapest" | "most" | "least", "measure": "property the pick compares, or empty", "needs": [{"property": "property name in the request's words", "want": "the value, limit or exclusion exactly as stated", "words": "at most 3 words a listing meeting it would show, or empty for a limit or an exclusion, which no keyword can express", "pointer": true when the value must be read from another listing the request describes, "about": "product" when it describes the product itself, "other" when it concerns price, budget, stock, delivery or how to deal with the shopper, "firm": true only for a use or occasion the request insists the product must suit (it must, has to, only) or whose opposite it rules out, else false}]}],
"budget": number or null, "budget_words": "the budget word for word as the request writes it, or empty", "budget_scope": "total" | "each",
"messages": how many messages the request allows the agent to send, or null,
"steps": [{"step": "a procedure the request tells the agent to follow, such as telling or checking something before ordering", "when": "before_order" | "before_end" | "any", "uses_message": true only when the request explicitly asks for a message to be sent}],
"hints": [{"item": item number, "property": "a property the story points at without giving its value", "words": "the story's words"}],
"roles": {"facts": "tool that shows one listing's attributes", "facts_id": "its id argument", "offer": "tool that shows one listing's price and stock if different from facts, else empty", "offer_id": "its id argument", "compare": "tool that shows the records of several listings in one call, given a list of their ids, or empty", "compare_ids": "its argument holding that list", "ask": "tool that messages the shopper", "ask_text": "its text argument", "add": {"tool": "cart tool", "lines": "when the tool sets the whole cart at once from a list of lines, the argument holding that list, else empty", "id": "id argument, or the id key inside a line", "qty": "quantity argument, or the quantity key inside a line, or empty", "fixed": {other arguments that make it add}}, "remove": {"tool": "cart tool, or empty when the cart is only set whole", "id": "id argument", "qty": "quantity argument or empty", "fixed": {other arguments that make it remove}}, "cart": "tool that shows the cart, or empty", "buy": "tool that places the order", "buy_args": {arguments that make it place the order, when one tool both orders and ends}, "end": "tool that ends without a purchase, which may be the same tool", "end_args": {arguments that make it end without a purchase; only what is required}},
"search": {"words_tool": "tool for keyword search", "words_fixed": {arguments that select a keyword search}, "words_arg": "argument holding the keywords", "words_max": max characters of the keywords or null, "words_page_arg": "its argument asking for a later page, or empty", "shelf_tool": "tool for browsing a category, which may be the same tool, or empty if none", "shelf_fixed": {arguments that select browsing a category}, "shelf_arg": "argument naming the category, or empty", "shelf_page_arg": "its argument asking for a later page, or empty", "shelf_with_words": true when the keyword search also accepts the category argument, "price_arg": "the browsing tool's argument limiting the highest price, or empty"}}
Rules: copy the budget's number exactly as the request writes it, with no change of unit. One entry per distinct product wanted. "needs" hold only what the request states about the product's own properties, including limits and exclusions. "want" keeps the words that make it a limit or an exclusion, such as "at most" or "not", and adds no explanation: a value the request rules out is written as ruled out. When the request requires the product to suit an occasion or a use, such as an event or an activity, that is a need under the property "purpose"; how, where, by whom or with what the story says it will be used is not. Other plans, places, activities and people in the story are not needs, nor are words that only ask for a suitable product without naming a value of one of its properties. When the request says the product must come from a named category, range or section of the store, "shelf" holds that name word for word; a style, type or other value the product must have is a need, never the shelf. Never make a need of the price or budget, stock or availability, substitutes, delivery, the message or step limits, or how to report back. Words that only say what kind of product it is are not needs: they go in "name" and "shelf". A value the request requires of one of the product's properties is a need even when the request also puts that value in the product's name, and when the request names the property, its "property" is that name word for word. A value to be read from another listing that the request describes is one need with "pointer": true whose "want" copies the request's words for it in full; the values that describe that other listing are not needs. Ending without a purchase when nothing fits, or saying so, is not a step, and neither is an invitation to ask the shopper questions. A step is an action the request tells the agent to take; what the story says about people, notes, plans, tastes or past purchases is not a step, and a remark that someone has a value in mind or written down, or that the shopper cannot judge or leaves something open, is a hint on that property, never a step. Use exact tool and argument names from the tools below.

REQUEST:
"""

SHAPE = """TASK shape. A store's search returned the result below. Name its parts by exact key. Return ONE JSON object, nothing else: {"rows": "key of the list of listings", "id": "key of a listing's id", "title": "key of its title or empty", "price": "key of its price or empty", "stock": "key saying if it can be bought, or empty", "category": "key of its category label or empty", "seller": "key of its seller or brand or empty", "total": "key of how many matched in all, or empty", "next": "key of the value that asks for the next page, or empty"}

RESULT:
"""

RANK = """TASK rank. A shopper holds a few firm preferences they did not write down, each on one property of one requested product. A question finds one only when it names that exact property, and a shopper mostly answers the first property a question names, so the order decides what the questions find. For each item below, list the properties worth asking about, best first, and estimate for each the chance the shopper holds an unwritten firm preference on it.
Order each item's properties this way:
1. A property the request asks the agent to check or confirm with the shopper.
2. A property the story points at without giving its value. Stories hide a preference in a remark about how the product should look or the place it is for, who will use or wear it, what goes in, on or with it, what the shopper cannot judge or leaves open, or why a past one did not suit. Read each such remark and name the one property of one item it points at, by the listed key that fits it best. Its chance is at least 0.5.
3. The properties a buyer of that kind of product chooses between first (its type or style, its material, its colour or pattern, its size, fit or main measure), among the keys most of its listings carry. Keys few listings carry come last.
Return ONE JSON object, nothing else: {"items": [{"item": item number, "properties": [{"name": "plain property name, using all the words of the key", "keys": ["every listed key that means this same property"], "p": number from 0 to 1, "hint": "the request's words it comes from when it is a property of kind 1 or 2, else empty"}]}]}
Rules: use the listed keys. Add a property no key lists only when the request points at it, and leave its "keys" empty. Each name is one property, with no "or" and no "vs". Leave out a key whose values list several purchase options at once. Merge different spellings of one property into one entry. Leave out properties the item's stated needs already settle, identifiers, model numbers, seller names, and selling terms such as warranty, shipping or returns. Keep at most 14 per item.

REQUEST:
"""

READ_REPLY = """TASK read_reply. An agent asked a shopper about some properties of the products below and got this reply. Return ONE JSON object, nothing else:
{"kind": "need" when the reply states a firm preference, "no_view" when it says the shopper does not mind, "puzzled" when it says the message was not understood, "too_many" when it asks for fewer things at a time, "refused" when the store says the message could not be sent or no messages remain, "other",
"needs": [{"item": item number, "property": "property name as the reply says it", "want": "the value, limit or exclusion, word for word with its units", "words": "at most 3 words a listing meeting it would show, or empty for a limit or an exclusion", "pointer": true when the value must be read from another listing the reply describes}],
"drop": [{"item": item number, "property": "requirement the shopper gives up"}], "budget": a money amount the reply now sets for the shopper to spend or stay within, its number exactly as the reply writes it with no change of unit, or null}
Rules: a value to be read from another listing that the reply describes, by its seller, category or some of its values, is one need with "pointer": true whose "want" copies the reply's words for it in full. The values that describe that other listing are that listing's, not needs of the product asked about.
"want" holds only what a listing's attribute could state: the value, with a limit word or a ruling-out word kept in front of it, and none of the shopper's framing around it. "finish matching brushed steel" wants "brushed steel"; "length equal to at least 2 m" wants "at least 2 m"; "shade should be teal and nothing else" wants "teal".

"""

READ_NOTE = """TASK read_note. While an agent shops, the store showed this text. Return ONE JSON object, nothing else:
{"kind": "change" when it changes the task or says the order cannot go through, "error" when it only reports a malformed call, "other",
"budget": a money amount the text now sets for the shopper to spend or stay within, its number exactly as the text writes it with no change of unit, or null, "needs": [{"item": item number, "property": "...", "want": "...", "words": "at most 3 words a listing meeting it would show, or empty for a limit or an exclusion", "pointer": false}], "drop": [{"item": item number, "property": "..."}],
"price_moved": true when it says in any words that the price of something the agent picked has changed, "gone": [listing ids it says are unavailable], "gone_unnamed": true when it says in any words that something the agent picked or suggested is gone, sold out, used up or cannot be ordered, without giving an id, "summary": "a few words"}

"""

JUDGE = """TASK judge. For each listing record below decide each requirement of the item from the record's attributes.
- Only attributes whose key names the requirement's property decide: the property's name in any case, spacing or punctuation, singular or plural, alone or with words naming a part or kind of the product, or the store's other name for that same property. A key naming a different property, the options or variants on offer, or a search filter does not decide, even when it holds the wanted words. The title and the description never decide.
- Records often repeat a property under several keys. When any key naming the property states the wanted value, the verdict is "yes".
- A value meets a wanted value when it contains the wanted words in the same order: case, spelling, singular or plural and units do not matter, and one element of a list is enough. A different word does not, even a near meaning, and neither do the same words in another order.
- A requirement that rules a value out is not met when any attribute naming the property holds it, and met when such attributes exist and none holds it.
- A limit includes its edge, and a range must lie wholly inside it.
- When keys naming the property disagree, the one named for this kind of product decides, else the property's bare name.
- With no attribute naming the property, the title or the description stating the value decides, since stores leave attributes out of some records; the options or variants on offer and attributes of other properties still do not. If nothing states it, the verdict is "unknown".
- "key" is the deciding attribute's key exactly as the record writes it, "title" or "description" when one of them decides, or empty. "quote" is the words of its value that decide, copied character for character.
- A requirement about availability or price is decided by the record's stock and price fields.
- "kind_ok" is false when the listing's category or title shows another kind of product than the item.
Return ONE JSON object, nothing else: {"listings": [{"id": "...", "kind_ok": boolean, "measure": number or null, "verdicts": [{"n": requirement number, "verdict": "yes" | "no" | "unknown", "key": "...", "quote": "..."}]}]}

"""

QUERIES = """TASK queries. Write short keyword searches that would find listings of this item meeting its requirements. The store's search needs every word to be on the listing, so each extra word can hide the right one: use at most 3 words, the wanted value first, and vary the words a listing might use for it. Leave out anything the listing must not be. Return ONE JSON object, nothing else: {"queries": ["..."]} with at most 3 entries.

"""

RESULT = """TASK result. A store's tool replied as below to a call. Did the call go through? It did not when the reply says it was refused, failed or was not allowed. Return ONE JSON object, nothing else: {"ok": true or false}

"""

STOCK = """TASK stock. A store's listing has this value in its stock field. Can the listing be bought now? Return ONE JSON object, nothing else: {"buyable": true or false}

VALUE: """

OFFER = """TASK offer. A store's page for one listing is below. Name its parts by exact key. Return ONE JSON object, nothing else: {"price": "key of its price, or empty", "stock": "key saying if it can be bought, or empty"}

PAGE:
"""

POINTER = """TASK pointer. A requirement takes its value from another listing that it describes, for example by seller, category and some of that listing's values. That listing's pages may state the value, state a rule such as what goes with it, or point on to a further listing. Using the requirement and the pages opened so far, say what to do next. Return ONE JSON object, nothing else:
{"found": "the value that the described listing, or the listing it points on to, has for the requirement's property, once the pages opened state it, else empty",
"needed": "the value a listing must have to meet the requirement, worked out from the pages: the same value when it must match; the value a rule on the pages pairs with it when it must go with it, be compatible with it or be its partner; empty when the requirement only rules a value out or sets a distance from it, or when the pages do not settle it yet",
"open": "id of a listing shown below to open next, or empty", "search": "keywords, at most 4 words, to find the listing still needed, such as its seller and the product word, or empty", "shelf": "its category if known, or empty"}

"""

CHECK = """TASK check. An agent is about to order this cart. Check it against every requirement, reading values the way a careful person would: spelling, case and units do not matter, a different word does. A requirement is decided only by attributes whose key names its property, in any spelling, alone or with words naming a part or kind of the product, or the store's other name for it; the title, the description, the options or variants on offer, search filters and attributes of other properties do not decide it. Return ONE JSON object, nothing else: {"wrong": [{"id": listing id, "n": requirement number, "key": "the deciding attribute's key", "why": "a few words"}]} with an empty list when all is right. List only what the deciding attribute clearly contradicts.

"""

GATE = """TASK gate. The request asks the agent to follow these procedures before finishing. From the record of what was done, name the next call needed, if any. Return ONE JSON object, nothing else: {"call": {"tool": "exact tool name", "args": {}} or null}
Never call a tool that orders or ends the task. Message the shopper only when a procedure itself asks for that message, and then only about that procedure: never to confirm a product, ask about its properties, or ask about anything else the request's story mentions. A procedure the record shows done needs no call.

"""


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _plain(text: Any) -> str:
    """Lower case words and numbers only; a possessive 's goes with its word, and a plus sign that closes a word or a
    number stays on it, since a value written with it names something other than the same value without it."""
    low = re.sub(r"(\w)['’]s\b", r"\1", str(text).lower())
    low = re.sub(r"(?<![a-z0-9+])\+|\+(?=[a-z0-9])", " ", low)
    return re.sub(r"[^a-z0-9+]+", " ", low).strip()


def _marked(text: Any) -> set[str]:
    """The words of a text that close with a plus sign."""
    return {w for w in _plain(text).split() if w.endswith("+")}


def _stem(word: str) -> str:
    """One spelling for a word's singular and plural, and for its British and American spellings in -our and -or,
    worked out from its ending."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    if len(word) > 5 and word.endswith("our"):
        word = word[:-3] + "or"
    word = re.sub(r"(ch|sh|ss|x|z)e$", r"\1", word)
    if len(word) > 4:
        word = re.sub(r"ie$", "y", word)
    return re.sub(r"(fe|f|ve)$", "v", word)


def _words(text: Any) -> set[str]:
    return {_stem(w) for w in _plain(text).split()}


def _seq(text: Any) -> list[str]:
    return [_stem(w) for w in _plain(text).split()]


def holds_phrase(value: Any, phrase: Any) -> bool:
    """Whether a value holds a phrase's words together and in the phrase's order, as "light brown" holds "brown"
    and "1440 x 3440" does not hold "3440 x 1440"."""
    have, want = _seq(value), _seq(phrase)
    return bool(want) and any(have[k:k + len(want)] == want for k in range(len(have) - len(want) + 1))


# Words a shopper puts in front of a value, which no listing states: read as part of the value ("matching brushed steel"),
# they made every listing that states the value ("brushed steel") read as failing it.
_BE = re.compile(r"^(?:(?:should|must|has to|have to|needs? to|is to|got to)\s+be|to be)\s+(?=\S)", re.I)
_FRAMING = re.compile(r"^(?:matching|matched (?:to|with)|matches|equal to|equals|equalling|set to|pinned to|landing on|land on|landed on|coming in as|comes in as|in the form of)\s+(?=\S)", re.I)


def unframed(want: str) -> str:
    """A told value without the framing words in front of it; a limit or ruling-out word stays."""
    text = str(want or "").strip()
    for _ in range(3):
        cut = _FRAMING.sub("", _BE.sub("", text)).strip()
        if not cut or cut == text:
            break
        text = cut
    return text


def drop_descriptions(raws: list[dict[str, Any]], reply: str = "") -> list[dict[str, Any]]:
    """Needs read from one text, without those that only repeat values a pointer in it uses to describe the other
    listing: those values are the other listing's, not this product's. A reading may move them out of the pointer's
    own words, so in a shopper's reply that holds a pointer, a value the reply writes right after its property's name,
    the way a description lists them, goes too. A request can state its own needs that way, so it gets only the
    first test."""
    told = [" ".join(str(r.get(k) or "") for k in ("property", "want")) for r in raws if r.get("pointer")]
    if not told:
        return raws

    def described(raw: dict[str, Any]) -> bool:
        want = str(raw.get("want") or "").strip()
        if not want:
            return False
        if any(holds_phrase(text, want) for text in told):
            return True
        return bool(reply) and holds_phrase(reply, f"{raw.get('property') or ''} {want}")

    return [r for r in raws if r.get("pointer") or not described(r)]


def _json_of(text: str) -> dict[str, Any] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        found = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        return float(found.group()) if found else None
    return None


def _dicts(value: Any) -> list[dict[str, Any]]:
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def belief_after_silence(p: float, heard: float) -> float:
    """The belief in a property after a message named it, with chance `heard` of an answer, and got none."""
    missed = p * (1.0 - heard)
    return missed / (missed + 1.0 - p) if missed + 1.0 - p > 0 else 0.0


def heard_at(place: int) -> float:
    return HEARD[min(place, len(HEARD) - 1)]


def update_beliefs(beliefs: dict[str, float], names: list[str], hit: int | None) -> None:
    """Update beliefs after a message naming `names` in order. `hit` is the place of the name that was answered.
    Names before it were passed over, so they weaken. Names after it were never reached, so they keep their
    belief. With no hit, every name was passed over."""
    last = len(names) if hit is None else hit
    for place, name in enumerate(names[:last]):
        if name in beliefs:
            beliefs[name] = belief_after_silence(beliefs[name], heard_at(place))
    if hit is not None and names[hit] in beliefs:
        del beliefs[names[hit]]


def line_gain(beliefs: dict[str, float], limit: int = MAX_NAMES) -> float:
    """The chance that a message of the top names of this line brings a preference to light."""
    miss = 1.0
    for place, p in enumerate(sorted(beliefs.values(), reverse=True)[:limit]):
        miss *= 1.0 - p * heard_at(place)
    return 1.0 - miss


def split_notes(observation: Any) -> tuple[Any, list[str]]:
    """Separate a tool result from notes carried beside it: a two-key wrapper holding the result and a list of
    texts, or a bare list of texts."""
    if isinstance(observation, list) and observation and all(isinstance(x, str) for x in observation):
        return None, list(observation)
    if isinstance(observation, dict) and len(observation) == 2:
        first, second = list(observation.values())
        for texts, rest in ((first, second), (second, first)):
            if isinstance(texts, list) and texts and all(isinstance(x, str) for x in texts) and not isinstance(rest, list):
                return rest, list(texts)
    return observation, []


def fit_args(args: dict[str, Any], schema: dict[str, Any] | None) -> dict[str, Any]:
    """Shape arguments to a tool's schema: unknown names go, numbers are coerced and clamped, long text is cut."""
    props = (schema or {}).get("properties") or {}
    out: dict[str, Any] = {}
    for name, value in (args or {}).items():
        spec = props.get(name)
        if spec is None or value is None:
            continue
        kind = spec.get("type")
        if kind in ("integer", "number"):
            number = _number(value)
            if number is None:
                continue
            value = int(number) if kind == "integer" else number
            if isinstance(spec.get("minimum"), (int, float)):
                value = max(value, spec["minimum"])
            if isinstance(spec.get("maximum"), (int, float)):
                value = min(value, spec["maximum"])
        elif kind == "string":
            value = str(value)
            if isinstance(spec.get("maxLength"), int):
                value = value[:spec["maxLength"]]
        if isinstance(spec.get("enum"), list) and value not in spec["enum"]:
            continue
        out[name] = value
    return out


def search_phrase(want: str) -> str:
    """The first wording of a wanted value."""
    return re.split(r"[,;(]|\s+or\s+|\s+and\s+", str(want).strip())[0].strip()


_RULED_OUT = re.compile(r"^\s*(?:not|no|never|anything but|anything other than|other than|except|excluding|without|none of)\b[\s:,-]*", re.I)
_LIMIT = re.compile(r"\b(?:at most|at least|up to|under|over|below|above|less|more|max(?:imum)?|min(?:imum)?|between|within|from)\b|[<>≤≥]", re.I)


def ruled_out_value(want: Any) -> str:
    """The value a requirement rules out, when its wording rules one out; else empty."""
    text = str(want or "")
    found = _RULED_OUT.match(text)
    return text[found.end():].strip() if found else ""


def same_property(key: Any, prop: Any) -> bool:
    """Whether an attribute key spells the requirement's property itself: the same words in any case, spacing,
    punctuation, singular or plural."""
    a, b = _words(key), _words(prop)
    return bool(a) and a == b


_OFFERED = {"option", "available", "variant", "choice", "filter"}


def names_property(key: Any, prop: Any) -> bool:
    """Whether an attribute key names the requirement's property: it holds all the property's words, alone or with
    words naming a part or kind of the product. Stores file a property under a longer key as often as under its bare
    name, and the task's own attribute was found under either. A key about what is on offer or a search filter
    names no value of this listing."""
    a, b = _words(key), _words(prop)
    return bool(b) and b <= a and not (a & _OFFERED)


def holding_families(records: list[dict[str, Any]], prop: str, want: str) -> set[frozenset]:
    """The wordings of key naming the property under which some record holds the wanted value: the attributes the
    store files that value under."""
    options = [o for o in re.split(r"\s+or\s+|\s*/\s*|\s*,\s*", str(want)) if _seq(o)]
    out: set[frozenset] = set()
    for facts in records:
        for key, value in facts.items():
            if names_property(key, prop) and any(holds_phrase(value, o) for o in options):
                out.add(frozenset(_words(key)))
    return out


def decide_by_spelling(facts: dict[str, Any], prop: str, want: str, item: Any = "", title: Any = "", filed: set[frozenset] | None = None) -> tuple[str, str, str] | None:
    """A verdict that spelling alone settles. Attributes whose keys name the property decide. When keys of different
    wording disagree on a wanted value, only a wording under which the store files that value somewhere (`filed`, from
    the records read) can decide, since the task's attribute is one that holds such values; of several, the key
    whose other words all name this kind of product decides (a skirt's "skirt style" over its general "style"), else
    the property's bare name (a top's "material" over its "clothing material"):
    the task's attribute was found under either, and those were the ones that decided. With no attribute naming
    the property, the title holding the value decides, since stores leave attributes out of some records. A
    ruled-out value is not met when any such attribute holds it, and met when they exist and none does. Limits and
    other wordings are left to a model. Returns (verdict, key, value) or None."""
    named = [(k, str(v)) for k, v in facts.items() if names_property(k, prop) and str(v).strip()]
    options = [o for o in re.split(r"\s+or\s+|\s*/\s*|\s*,\s*", str(want)) if _seq(o)]
    banned = ruled_out_value(want)
    if banned:
        if _LIMIT.search(banned) or not _seq(banned):
            return None
        for key, value in named:
            if holds_phrase(value, banned):
                return "no", key, value
        if named:
            return "yes", named[0][0], named[0][1]
        return ("no", "title", str(title)) if holds_phrase(title, banned) else None
    if _LIMIT.search(want) or not options:
        return None
    holds = lambda value: any(holds_phrase(value, o) for o in options)
    if not named:
        return ("yes", "title", str(title)) if title and holds(title) else None
    families: dict[frozenset, list[tuple[str, str]]] = {}
    for key, value in named:
        families.setdefault(frozenset(_words(key)), []).append((key, value))
    verdicts = {fam: any(holds(v) for _, v in pairs) for fam, pairs in families.items()}
    if len(set(verdicts.values())) > 1:
        contenders = [f for f in families if verdicts[f] or f in (filed or set())]
        if all(verdicts[f] for f in contenders):
            key, value = next(pair for f in contenders for pair in families[f] if holds(pair[1]))
            return "yes", key, value
        bare, kind = frozenset(_words(prop)), _words(item)
        own = [f for f in contenders if (f - bare) and (f - bare) <= kind]
        decider = own[0] if own else (bare if bare in contenders else None)
        if decider is None:
            return None
        key, value = families[decider][0]
        return ("yes" if verdicts[decider] else "no"), key, value
    for fam, pairs in families.items():
        for key, value in pairs:
            if holds(value):
                return "yes", key, value
    return None


def phrase_of(need: dict[str, Any]) -> str:
    """What to search for to meet a need: the value a pointer's pages say a listing needs, else the reader's search words,
    which are empty for a limit or an exclusion. With no reading, the wanted value's own words. The other listing's own
    value is no search: when the requirement rules it out or sets a distance from it, listings showing it are the wrong ones."""
    if need.get("needed"):
        return search_phrase(need["needed"])
    if need["value"] or need.get("pointer"):
        return ""
    return search_phrase(need["want"] if need["words"] is None else need["words"])


def query_ladder(name: str, phrases: list[str], head: str = "") -> list[str]:
    """Searches built from the wanted values. An all-words search hides a listing for every extra word, so each value
    goes with the word that says what the product is, then alone; several values together come first, as the sharpest
    try. Without a reading of that word, the name's last word stands in."""
    words = _plain(name).split()
    noun = _plain(head) or (words[-1] if words else "")
    out: list[str] = []
    if len(phrases) > 1:
        out.append(" ".join(phrases + [noun]).strip())
    out += [f"{p} {noun}".strip() for p in phrases]
    out += phrases
    if not phrases and name:
        out.append(name)
    seen: list[str] = []
    for query in out:
        if query and query not in seen:
            seen.append(query)
    return seen[:6]


def as_stated(read: float | None, text: str) -> float | None:
    """A budget read from a shopper's or store's words. A reading that converted the number to another unit is put back:
    when the read number is not in the text and the text holds exactly one number, that number stands."""
    if read is None or not text:
        return read
    found = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", str(text).replace(",", ""))]
    if not found or any(abs(n - read) < 1e-6 for n in found):
        return read
    return found[0] if len(found) == 1 else read


def records_in(value: Any, batch: list[str], out: dict[str, dict[str, Any]]) -> None:
    """Find each asked listing's record in a several-listing result: a dict holding the listing's id as one of its
    values, or a line of text naming it."""
    if isinstance(value, dict):
        hit = next((v for v in value.values() if isinstance(v, str) and v in batch), None)
        if hit is not None and hit not in out:
            out[hit] = value
            return
        for inner in value.values():
            records_in(inner, batch, out)
    elif isinstance(value, list):
        for inner in value:
            records_in(inner, batch, out)
    elif isinstance(value, str):
        for ident in batch:
            if ident in value and ident not in out:
                out[ident] = record_of_text(value)


def record_of_text(text: str) -> dict[str, Any]:
    """A record written as text: "key: value" parts joined by bars or line breaks, a value of "name=value" parts joined
    by semicolons being a table of its own. The whole text is kept too, so a form this does not split is still read."""
    record: dict[str, Any] = {"record": text}
    for part in re.split(r"\s+\|\s+|\n+", text):
        key, sep, value = part.partition(": ")
        if not sep:
            key, sep, value = part.partition("=")
        if not sep or not key.strip() or len(key) > 60:
            continue
        pairs = [p.partition("=") for p in value.split("; ")]
        table = {a.strip(): b.strip() for a, eq, b in pairs if eq and a.strip()}
        record[key.strip()] = table if len(table) >= 2 else value.strip()
    return record


def fewer_words(query: str, head: str = "") -> list[str]:
    """Shorter forms of a keyword search that matched nothing: its first two words, its words without numbers, and its
    first word with the word that says what the product is."""
    words = str(query).split()
    if len(words) <= 2:
        return []
    out = [" ".join(words[:2])]
    plain = [w for w in words if not re.search(r"\d", w)]
    if 2 <= len(plain) <= 3 and len(plain) < len(words):
        out.append(" ".join(plain))
    noun = _plain(head)
    if noun and _plain(words[0]) != noun:
        out.append(f"{words[0]} {noun}")
    return [q for q in dict.fromkeys(out) if q != query]


def question_text(parts: list[tuple[str, list[str]]]) -> str:
    """One sentence per item, each naming that item's properties in order."""
    out = []
    for item, names in parts:
        if len(names) == 1:
            out.append(f"What {names[0]} would you like for the {item}?")
        else:
            out.append(f"For the {item}, what {', '.join(names[:-1])} or {names[-1]} would you like?")
    return " ".join(out)


class Line:
    """One requested product and everything known about it."""

    def __init__(self, number: int, raw: dict[str, Any]):
        self.number = number
        self.name = str(raw.get("name") or "item")
        self.head = str(raw.get("head") or "")
        self.shelf = str(raw.get("shelf") or "")
        self.quantity = max(1, int(_number(raw.get("quantity")) or 1))
        self.pick = str(raw.get("pick") or "any")
        self.measure = str(raw.get("measure") or "")
        self.needs: list[dict[str, Any]] = []
        self.ids: list[str] = []
        self.beliefs: dict[str, float] = {}
        self.keys_of: dict[str, list[str]] = {}
        self.asked = 0
        self.queries: list[str] = []
        self.queries_made = False
        self.searches = 0
        self.more: list[dict[str, Any]] = []
        self.word_hits: dict[str, int] = {}
        self.exact = False
        self.related = False  # some row's label shares a word with the item
        self.wide = False     # the item's own labels gave no fit, so any label may be read
        self.capped_at: float | None = None
        self.last_names: list[str] = []
        self.last_silent = False
        self.browsed: set[str] = set()
        self.tried: list[str] = []
        self.rethought = False
        self.named = bool(self.shelf)   # the request named the category the product comes from
        self.times_asked: dict[str, int] = {}
        for need in drop_descriptions(_dicts(raw.get("needs"))):
            if need.get("about") != "other":
                self.add_need(need, False)

    def add_need(self, raw: dict[str, Any], told: bool) -> dict[str, Any] | None:
        prop, want = str(raw.get("property") or "").strip(), str(raw.get("want") or "").strip()
        if not prop and not want:
            return None
        text = f"{prop}: {want}"
        for need in self.needs:
            if need["text"] == text:
                need["live"] = True
                return need
        words = raw.get("words")
        # a purpose is soft unless the request insists on it or rules its opposite out: then it is a requirement like
        # any other, and an order that ignores it fails
        firm = raw.get("firm") is True or bool(re.search(r"\b(?:not|no|nothing|never|except|without)\b", want, re.I))
        need = {"text": text, "prop": prop, "want": want, "words": None if words is None else str(words), "told": told, "pointer": bool(raw.get("pointer")), "value": "", "needed": "", "live": True,
                "soft": not told and _words(prop) == {"purpose"} and not firm}
        self.needs.append(need)
        return need

    def live(self) -> list[dict[str, Any]]:
        return [n for n in self.needs if n["live"]]

    def label(self) -> str:
        return self.shelf.lower() if self.shelf else self.name


class Sieve:
    def __init__(self, problem: dict[str, Any]):
        environment = problem["environment"]
        self.binding = environment["binding"]
        policy = environment["policy_view"]
        self.request = str(policy.get("query") or "")
        if policy.get("opening") is not None:
            self.request += "\n\n" + _dump(policy["opening"])
        self.tools = policy.get("tools") or []
        self.schemas = {t["function"]["name"]: t["function"].get("parameters") or {} for t in self.tools}
        self.max_steps = int(policy.get("max_steps") or 30)
        self.finished = bool(policy.get("finished"))
        self.t0 = time.monotonic()
        self.step = 0
        self.cost = 0.0
        self.spent: dict[str, float] = {}
        self.banned: set[str] = set()
        self.allowed: list[str] = []
        self.heavy_of: dict[str, str] = {LIGHT: LIGHT, HEAVY: HEAVY}
        self.effort = True
        self.no_credit = False
        self.no_heavy = False
        self.late_asked = False
        self.waited = 0.0
        self.open_tried: set[str] = set()
        self.calls = 0
        self.trace: list[dict[str, Any]] = []
        self.roles: dict[str, Any] = {}
        self.search: dict[str, Any] = {}
        self.shape: dict[str, Any] = {}
        self.lines: list[Line] = []
        self.brief: dict[str, Any] = {}
        self.budget: float | None = None
        self.budget_each = False
        self.msg_limit = 3
        self.msg_used = 0
        self.name_cap = FIRST_NAMES
        self.rows: dict[str, dict[str, Any]] = {}
        self.pages: dict[str, dict[str, Any]] = {}
        self.owner: dict[str, set[int]] = {}
        self.verdicts: dict[tuple[str, str], tuple[str, str]] = {}
        self.kind_bad: set[str] = set()
        self.judged: set[tuple[str, int]] = set()
        self.measures: dict[str, float] = {}
        self.gone: set[str] = set()
        self.cart: dict[str, int] = {}
        self.pending_gone = False
        self.pending_price = False
        self.changed = False
        self.notes_seen = 0
        self.note_readings: dict[str, dict[str, Any]] = {}
        self.searched: set[str] = set()
        self.done_log: list[str] = []
        self.buy_tries = 0
        self.pressed = False
        self.end_refused = False
        self.compare_cap = 4
        self.compare_off = False
        self.offer_read: set[str] = set()
        self.stock_words: dict[str, bool] = {}
        self.result_readings: dict[Any, bool] = {}

    # ---- clock, money, models ----
    def time_left(self) -> float:
        return DEADLINE - (time.monotonic() - self.t0)

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def tail(self) -> int:
        """Steps kept for the finish: the first order (cart lines, a look at the cart, the order) and one swap. An
        order goes out at the first verified fit, so until then steps do more good finding one. A first order is often
        refused for a moved price or a new budget, so two more steps are kept for reading the cart afresh."""
        return 2 * len(self.lines) + 5

    def ladder(self, models: list[str]) -> list[str]:
        """The models to try, a refused one replaced by the first spare the proxy has not refused."""
        out: list[str] = []
        for model in models:
            choice = model
            if model in self.banned:
                choice = next((m for m in self.allowed + SPARES.get(model, []) if m not in self.banned), "")
                if choice:
                    self.heavy_of.setdefault(choice, self.heavy_of.get(model, model))
            if choice and choice not in out:
                out.append(choice)
        return out

    def refuse_model(self, model: str, body: str) -> None:
        """The proxy refused this model for the run: it is not asked again, and models the refusal names as allowed
        become the first spares."""
        self.banned.add(model)
        listed = re.search(r'allowed_models"?\s*:\s*\[([^\]]*)', body)
        if listed:
            for name in re.findall(r'"([^"\s]{3,100})"', listed.group(1)):
                if name not in self.allowed and name not in self.banned:
                    self.allowed.append(name)

    def heavy_fits(self, prompt: str, max_tokens: int) -> bool:
        """Whether a heavy read may start: only early in the spending, and only if its likely cost, from the prompt's
        length at about 3.5 characters a token and a typical answer, keeps the episode under COST_STOP."""
        guess = len(prompt) / 3.5 * HEAVY_IN + min(max_tokens, 1500) * HEAVY_OUT
        return self.cost < COST_SOFT and self.cost + guess <= COST_STOP

    def send(self, body: dict[str, Any], wait: int) -> tuple[dict[str, Any] | None, int | None, str]:
        """One model request, sent again after a pause when its failure is one that passes. Returns the answer, or None
        with the failure's status (None when no reply came at all) and its words."""
        tries = 0
        while True:
            _llm.timeout = max(5, min(wait, int(self.time_left() - 10)))
            self.calls += 1
            started = time.monotonic()
            data, status, said = self.outcome(_llm.post_verbose("/inference/chat/completions", body))
            if status is None and time.monotonic() - started > 10:
                # it timed out: the same model under the same load would likely time out again, so the reading goes on
                # to the next model at once instead
                self.trace.append({"kind": "model_fail", "model": body.get("model"), "status": "timeout", "tries": tries + 1})
                return None, status, said
            if data is not None:
                return data, 200, ""
            if "reasoning" in body and "reasoning" in said.lower() and status not in PASSING:
                # a proxy that does not take the setting: send without it from now on
                self.effort = False
                body.pop("reasoning", None)
                continue
            limit = 1 if status in (None, 200, 408) else len(WAITS)
            passing = status is None or status == 200 or status in PASSING
            pause = WAITS[min(tries, len(WAITS) - 1)] * (0.75 + 0.5 * random.random())
            if not passing or tries >= limit or self.waited + pause > WAIT_TOTAL or self.time_left() - pause < LAST_MODEL + 5:
                self.trace.append({"kind": "model_fail", "model": body.get("model"), "status": status, "tries": tries + 1, "said": said[:120]})
                return None, status, said
            self.waited += pause
            tries += 1
            time.sleep(pause)

    @staticmethod
    def outcome(result: Any) -> tuple[dict[str, Any] | None, int | None, str]:
        """What a request came back with: the answer when it holds choices, else the failure's status and words. A
        provider can report its failure inside an ordinary reply, as an error object with a code."""
        data = getattr(result, "data", None)
        if isinstance(data, dict) and data.get("choices"):
            return data, 200, ""
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            code = _number(data["error"].get("code"))
            return None, int(code) if code is not None else 500, str(data["error"].get("message") or "")[:4000]
        if isinstance(data, dict):
            return None, 200, "a reply without choices"
        error = getattr(result, "error", None)
        error = error if isinstance(error, dict) else {}
        status = error.get("status")
        return None, status if isinstance(status, int) else None, str(error.get("body") or "")[:4000]

    def read(self, prompt: str, models: list[str], max_tokens: int = 1500, wait: int = 45, think: bool = False) -> dict[str, Any] | None:
        """One model reading. Reads ask for low reasoning effort unless `think` lets the model reason at its own length:
        the choice of what to ask the shopper is the judgement the hidden needs hang on."""
        queue, tried = self.ladder(models), set()
        while queue:
            model = queue.pop(0)
            if model in tried:
                continue
            tried.add(model)
            if self.no_credit or self.cost >= COST_HARD or self.time_left() < LAST_MODEL:
                return None
            if self.heavy_of.get(model) == HEAVY and (self.no_heavy or not self.heavy_fits(prompt, max_tokens)):
                continue
            # both models reason before they answer, and the reasoning counts against max_tokens: a tight limit
            # leaves no room for the answer, which then comes back empty. Low effort keeps the reasoning short.
            body = {"model": model, "max_tokens": max(max_tokens, MIN_TOKENS), "temperature": 0,
                    "messages": [{"role": "user", "content": prompt}]}
            if self.effort and not think:
                body["reasoning"] = {"effort": "low"}
            data, status, said = self.send(body, wait)
            if data is None and status == 402:
                # refused for credit. The provider refuses a request whose largest possible cost exceeds the credit
                # left, so the dear model can be refused while the cheap one still fits: the heavy model is dropped
                # first, and only a refusal of the light one stops every model request, leaving code to finish alone
                if self.heavy_of.get(model) == HEAVY:
                    self.no_heavy = True
                    continue
                self.no_credit = True
                self.trace.append({"kind": "no_credit", "model": model})
                return None
            if data is None and status in (401, 403, 404):
                # refused for this run: its spare tries the same reading at once
                self.refuse_model(model, said)
                queue[:0] = [m for m in self.ladder([model]) if m not in tried]
                continue
            if data is None:
                continue
            usage = data.get("usage") or {}
            if isinstance(usage.get("cost"), (int, float)):
                self.cost += float(usage["cost"])
                self.spent[model] = self.spent.get(model, 0.0) + float(usage["cost"])
            parsed = _json_of(str(data["choices"][0].get("message", {}).get("content") or ""))
            if parsed is not None:
                return parsed
            self.trace.append({"kind": "read_fail", "model": model, "task": prompt[:24]})
        return None

    def careful(self) -> list[str]:
        return [HEAVY, LIGHT] if self.cost < COST_SOFT else [LIGHT, HEAVY]

    # ---- the store ----
    def call(self, name: str, args: dict[str, Any]) -> Any:
        """One environment step. Returns the tool's own result; notes beside it are read at once."""
        if self.finished or self.steps_left() <= 0 or name not in self.schemas:
            return None
        self.step += 1
        reply = _env.post("/environment/call", {**self.binding, "step": self.step, "action": {"name": name, "args": fit_args(args, self.schemas[name])}})
        if reply is None:
            self.step -= 1
            self.trace.append({"kind": "env_fail", "tool": name})
            return None
        self.finished = bool(reply.get("finished"))
        result, notes = split_notes(reply.get("observation"))
        self.done_log.append(f"{self.step}. {name} {_dump(args)[:160]} -> {_dump(result)[:160]}")
        for note in notes:
            self.on_note(note)
        return result

    def refused(self, tool: str, reply: Any) -> bool:
        """Whether a tool's reply says the call did not go through. Stores word this differently, so a model reads it,
        once per shape of reply. With no reading the call counts as done, and the order step reports any trouble."""
        if not isinstance(reply, dict) or not reply:
            return False
        shape = (tool, tuple(sorted((key, value is False, not value) for key, value in reply.items())))
        if shape not in self.result_readings:
            got = self.read(RESULT + f"TOOL: {tool}\nREPLY: {_dump(reply)[:600]}", [LIGHT, HEAVY], 60) or {}
            self.result_readings[shape] = got.get("ok") is False
        return self.result_readings[shape]

    def on_note(self, text: str) -> None:
        self.notes_seen += 1
        self.trace.append({"kind": "note", "text": text[:200]})
        got = self.note_readings.get(text)
        if got is None:
            items = "\n".join(f"item {l.number}: {l.name}" for l in self.lines)
            got = self.read(READ_NOTE + f"ITEMS:\n{items}\nCART: {_dump(self.cart)}\n\nTEXT:\n{text[:1500]}", [LIGHT, HEAVY], 500) or {}
            self.note_readings[text] = got
        self.apply_change(got, text)

    def apply_change(self, got: dict[str, Any], text: str = "") -> None:
        # the kind label alone is not trusted: any concrete change counts, and a note that could not be read is
        # treated as possibly saying the pick is gone, which costs a stock check rather than a wrong order
        concrete = [got.get(k) for k in ("budget", "needs", "drop", "gone", "gone_unnamed", "price_moved")]
        if got.get("kind") == "change" or any(concrete):
            self.changed = True
        if not got:
            self.changed = True
            self.pending_gone = True
        budget = as_stated(_number(got.get("budget")), text)
        if budget is not None:
            self.budget = budget
        for raw in _dicts(got.get("needs")):
            if not raw.get("pointer"):
                raw = {**raw, "want": unframed(str(raw.get("want") or ""))}
            self.line_of(raw.get("item")).add_need(raw, True)
        self.drop_needs(got.get("drop"))
        for ident in got.get("gone") or []:
            self.gone.add(str(ident))
        if got.get("gone_unnamed"):
            self.pending_gone = True
        if got.get("price_moved"):
            self.pending_price = True

    def drop_needs(self, raws: Any) -> None:
        """The shopper gave these up. The same property can be named with more or fewer words each time, so a name
        contained in the other either way is a match."""
        for raw in _dicts(raws):
            words = _words(raw.get("property"))
            for need in self.line_of(raw.get("item")).needs:
                prop = _words(need["prop"])
                if words and prop and (words <= prop or prop <= words):
                    need["live"] = False

    def line_of(self, number: Any) -> Line:
        index = int(_number(number) or 1)
        return self.lines[min(max(index, 1), len(self.lines)) - 1]

    def learn_shape(self, result: Any) -> None:
        # an empty result cannot show what a listing looks like, so wait for one that holds records
        if self.shape.get("rows") or not isinstance(result, dict) or not any(_dicts(v) for v in result.values()):
            return
        got = self.read(SHAPE + _dump(result)[:2500], [LIGHT, HEAVY], 300) or {}
        if got.get("rows"):
            self.shape = got
            return
        # no reading available: the first list of records, and its first key holding a different code on every row
        for key, value in result.items():
            rows = _dicts(value)
            if rows:
                codes = [k for k in rows[0] if all(isinstance(r.get(k), str) and " " not in r[k] and re.search(r"\d", r[k]) for r in rows) and len({r[k] for r in rows}) == len(rows)]
                self.shape = {"rows": key, "id": codes[0] if codes else ""}
                return

    def rows_in(self, result: Any) -> list[dict[str, Any]]:
        if not isinstance(result, dict):
            return []
        rows = _dicts(result.get(self.shape.get("rows")))
        return rows or next((_dicts(v) for v in result.values() if _dicts(v)), [])

    def ident(self, row: dict[str, Any]) -> str:
        return str(row.get(self.shape.get("id")) or "")

    def price(self, ident: str) -> float | None:
        for record in (self.pages.get(ident), self.rows.get(ident)):
            if record:
                value = _number(record.get(self.shape.get("price")))
                if value is not None:
                    return value
        return None

    def in_stock(self, ident: str) -> bool:
        if ident in self.gone:
            return False
        key = self.shape.get("stock")
        for record in (self.pages.get(ident), self.rows.get(ident)):
            if record and key in record:
                return self.stock_says(record[key])
        return True

    def stock_says(self, value: Any) -> bool:
        """A flag or count speaks for itself. Words differ between stores, so a model reads each one, once."""
        if isinstance(value, (bool, int, float)):
            return bool(value)
        text = str(value or "").strip()
        if not text:
            return True
        if text not in self.stock_words:
            got = self.read(STOCK + text[:80], [LIGHT, HEAVY], 60) or {}
            self.stock_words[text] = got.get("buyable") is not False
        return self.stock_words[text]

    def category(self, ident: str) -> str:
        return str((self.rows.get(ident) or self.pages.get(ident) or {}).get(self.shape.get("category")) or "")

    def affordable(self, ident: str, line: Line) -> bool:
        price = self.price(ident)
        if self.budget is None or price is None:
            return True
        return (price if self.budget_each else price * line.quantity) <= self.budget

    def over_budget(self, cart: dict[str, int]) -> bool:
        """Whether a cart breaks a total budget. A limit per unit is checked listing by listing instead."""
        if self.budget is None or self.budget_each:
            return False
        return sum((self.price(i) or 0) * count for i, count in cart.items()) > self.budget

    def run_search(self, line: Line, args: dict[str, Any]) -> Any:
        tool = self.tool_of_search(args)
        mark = tool + _dump(sorted(fit_args(args, self.schemas.get(tool)).items()))
        if mark in self.searched:
            return None  # the same search shows the same rows, so it would only spend a step
        self.searched.add(mark)
        result = self.call(tool, args)
        self.learn_shape(result)
        rows = self.rows_in(result)
        for row in rows:
            ident = self.ident(row)
            if ident:
                self.rows[ident] = {**self.rows.get(ident, {}), **row}
                self.owner.setdefault(ident, set()).add(line.number)
                if ident not in line.ids:
                    line.ids.append(ident)
                if self.search.get("words_arg") in args:
                    line.word_hits[ident] = line.word_hits.get(ident, 0) + 1
        line.searches += 1
        words_arg = self.search.get("words_arg")
        if not rows and words_arg in args:
            # every word must be on a listing, so a search that matched nothing is tried again with fewer words
            for query in reversed(fewer_words(str(args[words_arg]), line.head)):
                if query not in line.queries:
                    line.queries.insert(0, query)
        if rows and line.number:
            self.learn_shelf(line)
        self.trace.append({"kind": "search", "args": _dump(args)[:120], "rows": len(rows), "item": line.number, "ids": [self.ident(r) for r in rows]})
        self.queue_next_page(line, args, result, rows)
        return result

    def queue_next_page(self, line: Line, args: dict[str, Any], result: Any, rows: list[dict[str, Any]]) -> None:
        """Remember how to read on: the store's own next value, or an offset or page number from the schema."""
        tool = self.tool_of_search(args)
        arg = self.search.get("words_page_arg") if tool == self.search.get("words_tool") and self.search.get("words_arg") in args else self.search.get("shelf_page_arg")
        if not arg or not rows or not isinstance(result, dict):
            return
        following = result.get(self.shape.get("next")) if self.shape.get("next") else None
        total = _number(result.get(self.shape.get("total"))) if self.shape.get("total") else None
        spec = ((self.schemas.get(tool) or {}).get("properties") or {}).get(arg) or {}
        current = args.get(arg)
        if following is None:
            if self.shape.get("next") and self.shape.get("next") in result:
                return
            if total is None or spec.get("type") != "integer":
                return
            if spec.get("minimum") == 1:
                following = int(current or 1) + 1
            else:
                following = int(current or 0) + len(rows)
            limit = spec.get("maximum")
            if isinstance(limit, int) and following > limit:
                return
            if spec.get("minimum") != 1 and following >= total:
                return
        # later pages of a browse come first; of keyword searches only the first one's are worth a step
        page = {**args, arg: following}
        if self.search.get("words_arg") not in args:
            line.more.insert(0, page)
        elif not any(self.search.get("words_arg") in other for other in line.more):
            line.more.append(page)

    def compare_room(self) -> int:
        """How many listings one call of the store's several-listing tool may show: what its schema allows, at most four,
        less after a refusal; none when the store has no such tool or its records lack attributes."""
        tool, arg = self.roles.get("compare"), self.roles.get("compare_ids")
        if self.compare_off or not tool or tool not in self.schemas or not arg:
            return 0
        spec = ((self.schemas.get(tool) or {}).get("properties") or {}).get(arg)
        if not isinstance(spec, dict) or spec.get("type") not in (None, "array"):
            return 0
        top = spec.get("maxItems")
        room = min(4, int(top)) if isinstance(top, int) and top > 0 else 4
        return min(room, self.compare_cap)

    def read_listings(self, ids: list[str], stop: int | None = None) -> None:
        """Read listings, several in a step when the store allows it, the rest one at a time."""
        stop = self.max_steps if stop is None else stop
        todo = [i for i in dict.fromkeys(ids) if i not in self.pages]
        while len(todo) >= 2 and self.compare_room() >= 2 and self.step < stop and self.steps_left() > self.tail():
            batch = todo[:self.compare_room()]
            shown = self.compare(batch)
            if shown is None:
                # refused: perhaps too many at once, so fewer next time, then none
                if self.compare_cap > 2:
                    self.compare_cap = 2
                else:
                    self.compare_off = True
                continue
            todo = [i for i in todo if i not in shown]
            if batch[0] in todo:
                break  # the tool did not show the first listing it was asked for: read the rest alone
        for ident in todo:
            if self.step < stop and self.steps_left() > self.tail():
                self.open_listing(ident)

    def compare(self, batch: list[str]) -> set[str] | None:
        """One call of the several-listing tool. Records that carry an attribute table count as read pages; records
        without one only add what they show to the rows, and the tool is not used again. None when refused."""
        tool, arg = self.roles["compare"], self.roles["compare_ids"]
        result = self.call(tool, {arg: list(batch)})
        if result is None or not isinstance(result, (dict, list)) or self.refused(tool, result):
            return None
        found: dict[str, dict[str, Any]] = {}
        records_in(result, batch, found)
        if not found:
            return None
        full = any(any(isinstance(v, dict) and len(v) >= 2 for v in rec.values()) for rec in found.values())
        for ident, rec in found.items():
            if full:
                self.pages[ident] = {**self.pages.get(ident, {}), **rec}
            else:
                self.rows[ident] = {**self.rows.get(ident, {}), **{k: v for k, v in rec.items() if not isinstance(v, (dict, list))}}
        self.trace.append({"kind": "compare", "asked": len(batch), "shown": len(found), "full": full})
        if not full:
            self.compare_off = True
            return set()
        return set(found)

    def open_listing(self, ident: str) -> None:
        tool, arg = self.roles.get("facts"), self.roles.get("facts_id")
        if not tool or not arg:
            return
        result = self.call(tool, {arg: ident})
        if isinstance(result, dict) and result:
            self.pages[ident] = {**self.pages.get(ident, {}), **result}
        else:
            self.pages.setdefault(ident, {})
        self.read_offer(ident)

    def read_offer(self, ident: str) -> None:
        """A store may show price and stock only on a second page. It is read only when the price is unknown and a
        budget or a cheapest pick depends on it."""
        tool, arg = self.roles.get("offer"), self.roles.get("offer_id")
        depends = self.budget is not None or any(l.pick == "cheapest" for l in self.lines)
        if not tool or not arg or not depends or self.price(ident) is not None or self.steps_left() <= self.tail():
            return
        self.offer_read.add(ident)
        extra = self.call(tool, {arg: ident})
        if not isinstance(extra, dict) or not extra:
            return
        self.pages[ident] = {**self.pages.get(ident, {}), **extra}
        self.rows[ident] = {**self.rows.get(ident, {}), **extra}
        if not self.shape.get("price"):
            got = self.read(OFFER + _dump(extra)[:1500], [LIGHT, HEAVY], 100) or {}
            if got.get("price"):
                self.shape["price"] = got["price"]
            if got.get("stock") and not self.shape.get("stock"):
                self.shape["stock"] = got["stock"]

    def record_text(self, ident: str) -> str:
        return _dump({**(self.rows.get(ident) or {}), **(self.pages.get(ident) or {})})[:2200]

    def brief_text(self, ident: str, clip: int) -> str:
        """A listing with every text value cut to `clip` characters, so that a long description cannot push its other
        values, such as its id, seller or a remark about another listing, out of a short view."""
        def cut(value: Any) -> Any:
            if isinstance(value, str):
                return value if len(value) <= clip else value[:clip] + "..."
            if isinstance(value, dict):
                return {k: cut(v) for k, v in value.items()}
            if isinstance(value, list):
                return [cut(v) for v in value[:12]]
            return value
        return _dump(cut({**(self.rows.get(ident) or {}), **(self.pages.get(ident) or {})}))

    # ---- reading the task ----
    def read_task(self) -> None:
        tools = "\n".join(f"- {t['function']['name']}: {t['function'].get('description', '')} | {_dump(t['function'].get('parameters') or {})}" for t in self.tools)
        # a request with several items, each with its needs, and a long tool list makes a long answer: a cut answer
        # cannot be read, and without it the episode never starts
        prompt = READ_REQUEST + self.request + "\n\nTOOLS:\n" + tools
        got = self.read(prompt, [HEAVY, LIGHT], 4500, 80) or {}
        # without this reading the episode cannot start: one that never came, or came without the tools to search and
        # order with, is tried again a pause apart while the clock allows, each model first in turn
        tries = 0
        while not self.usable(got) and not self.no_credit and tries < 4 and time.monotonic() - self.t0 + REQUEST_GAP < REQUEST_BY:
            tries += 1
            if not got:
                time.sleep(REQUEST_GAP * (0.75 + 0.5 * random.random()))
            again = self.read(prompt, [LIGHT, HEAVY] if tries % 2 else [HEAVY, LIGHT], 4500, 60) or {}
            if self.usable(again) or len(_dicts(again.get("items"))) > len(_dicts(got.get("items"))):
                got = again
        self.brief = got
        roles = got.get("roles") if isinstance(got.get("roles"), dict) else {}
        # a store may lack a tool for a role, so roles are guessed from the tools' names only when no usable reading came
        self.roles = roles if self.usable(got) else self.roles_by_name(dict(roles))
        self.search = self.normal_search(got.get("search"))
        if self.search.get("words_tool") not in self.schemas and self.search.get("shelf_tool") not in self.schemas:
            self.search = {**self.search, **self.search_by_name()}
        self.find_compare()
        self.lines = [Line(i, raw) for i, raw in enumerate(_dicts(got.get("items")), 1)] or [Line(1, {"name": self.item_in_request()})]
        self.budget = self.as_written(_number(got.get("budget")), got.get("budget_words"))
        self.budget_each = got.get("budget_scope") == "each"
        limit = _number(got.get("messages"))
        self.msg_limit = int(limit) if limit is not None else 3
        self.trace.append({"kind": "brief", "items": [(l.name, l.quantity, l.pick, [n["text"] for n in l.needs]) for l in self.lines], "budget": self.budget, "messages": self.msg_limit})

    def find_compare(self) -> None:
        """When the reading named no several-listing tool, a tool whose one argument is a list of texts and that has no
        other role is taken as one; the first call shows whether it returns records."""
        if self.roles.get("compare") in self.schemas:
            return
        taken = {self.roles.get(k) for k in ("facts", "offer", "ask", "cart", "buy", "end")}
        taken |= {(self.roles.get(k) or {}).get("tool") for k in ("add", "remove") if isinstance(self.roles.get(k), dict)}
        taken |= {self.search.get("words_tool"), self.search.get("shelf_tool")}
        for name, schema in self.schemas.items():
            props = (schema or {}).get("properties") or {}
            if name in taken or len(props) != 1:
                continue
            arg, spec = next(iter(props.items()))
            items = spec.get("items") if isinstance(spec, dict) else None
            if isinstance(spec, dict) and spec.get("type") == "array" and (not isinstance(items, dict) or items.get("type") in (None, "string")):
                self.roles["compare"], self.roles["compare_ids"] = name, arg
                return

    def usable(self, got: dict[str, Any]) -> bool:
        """Whether a reading of the request can start the episode: it names the products and the tools that order."""
        roles = got.get("roles") if isinstance(got.get("roles"), dict) else {}
        search = got.get("search") if isinstance(got.get("search"), dict) else {}
        found = any(search.get(k) in self.schemas for k in ("words_tool", "shelf_tool", "tool"))
        return bool(_dicts(got.get("items"))) and roles.get("buy") in self.schemas and found

    def described(self, name: str) -> str:
        """A tool's name, split into words, with its description: what the fallback reads to tell tools apart."""
        spec = next((t["function"] for t in self.tools if t["function"]["name"] == name), {})
        return (re.sub(r"(?<=[a-z])(?=[A-Z])|[_\-]", " ", name) + " " + str(spec.get("description") or "")).lower()

    def params(self, name: str) -> dict[str, Any]:
        return (self.schemas.get(name) or {}).get("properties") or {}

    def roles_by_name(self, roles: dict[str, Any]) -> dict[str, Any]:
        """Roles the reading left empty, taken from the tools' names, descriptions and schemas. Only a reading that
        failed leaves them empty, so this guesses from the words stores use; a role it cannot place stays empty."""
        def first(pattern: str, avoid: tuple[str, ...] = (), need: str = "") -> str:
            for name in self.schemas:
                if name in avoid or not re.search(pattern, self.described(name)):
                    continue
                if need and not any(re.search(need, p, re.I) for p in self.params(name)):
                    continue
                return name
            return ""

        def param(name: str, pattern: str) -> str:
            return next((p for p in self.params(name) if re.search(pattern, p, re.I)), "")

        ident = r"(?:^|_)id$|id$|^id|ident|sku|code"
        if roles.get("ask") not in self.schemas:
            roles["ask"] = first(r"\b(?:message|ask|shopper|buyer|customer|client|chat|contact)\b")
            roles["ask_text"] = param(roles["ask"], r"text|message|body|question|content|msg") if roles["ask"] else ""
        add = roles.get("add") if isinstance(roles.get("add"), dict) else {}
        if add.get("tool") not in self.schemas:
            tool = first(r"\b(?:cart|bag|basket|trolley)\b", (roles.get("ask", ""),), ident)
            if tool:
                roles["add"] = {"tool": tool, "lines": "", "id": param(tool, ident), "qty": param(tool, r"qty|quant|count|number|units|amount"), "fixed": {}}
        if roles.get("buy") not in self.schemas:
            roles["buy"] = first(r"\b(?:check ?out|purchase|place (?:the |an |my )?order|order|buy|pay)\b", (roles.get("ask", ""), (roles.get("add") or {}).get("tool", "")))
        if roles.get("end") not in self.schemas:
            roles["end"] = first(r"\b(?:decline|refuse|without (?:a )?purchas\w*|no purchase|give up|abandon|pass|end|quit|exit|leave|walk away)\b", (roles.get("ask", ""), roles.get("buy", "")))
        if roles.get("facts") not in self.schemas:
            tool = next((n for n in self.schemas if len(self.params(n)) == 1 and re.search(ident, next(iter(self.params(n))), re.I)
                         and re.search(r"\b(?:detail|view|read|open|show|info|get|inspect|look|page|card|spec)", self.described(n))
                         and n not in ((roles.get("add") or {}).get("tool"), roles.get("buy"), roles.get("end"))), "")
            roles["facts"], roles["facts_id"] = tool, (next(iter(self.params(tool))) if tool else "")
        self.trace.append({"kind": "roles", "roles": {k: v for k, v in roles.items() if k in ("ask", "add", "buy", "end", "facts")}})
        return roles

    def search_by_name(self) -> dict[str, Any]:
        """The keyword search the reading left out: the tool with a free-text argument named for search words."""
        words = r"^(?:q|query|search|keywords?|terms?|text|look ?up|phrase|words|find)$|query|search|keyword|lookup|phrase"
        talk = r"\b(?:message|ask|shopper|buyer|customer|client|chat|contact)\b"
        ranked = sorted(self.schemas, key=lambda n: not re.search(r"\b(?:search|find|look|locate|catalog|browse|query)", self.described(n)))
        for name in ranked:
            if name == self.roles.get("ask") or re.search(talk, self.described(name)):
                continue
            arg = next((p for p, spec in self.params(name).items() if re.search(words, p, re.I) and (spec or {}).get("type") in (None, "string")), "")
            if arg:
                return {"words_tool": name, "words_arg": arg, "words_fixed": {}, "shelf_tool": ""}
        return {}

    def item_in_request(self) -> str:
        """The product a request asks for, by its usual wording ("I want a ...", "It's ... I'm after"), when no reading
        of it came: a search needs some words."""
        text = self.request.lower()
        for pattern in (r"\b(?:it'?s|it is) (?:a |an |some )?([a-z][a-z' &-]{2,40}?) (?:that )?i(?:'m| am)? (?:want|after|need)",
                        r"\b([a-z][a-z' &-]{2,40}?) (?:are|is) what i'?m after",
                        r"\bi(?:'d like| want| need| would like| am looking for|'m looking for)(?: to buy| to get)? (?:a |an |some |the )?([a-z][a-z' &-]{2,40}?)(?:[,.;!]| with| that| for| and| from|$)"):
            found = re.search(pattern, text)
            if found:
                return found.group(1).strip()
        return "item"

    def normal_search(self, raw: Any) -> dict[str, Any]:
        """One view of the search tools: a store may have one tool for both ways of searching, or one for each."""
        search = dict(raw) if isinstance(raw, dict) else {}
        shared = search.get("tool") or ""
        search["words_tool"] = search.get("words_tool") or shared
        search["shelf_tool"] = search.get("shelf_tool") or shared
        for field in ("words_page_arg", "shelf_page_arg"):
            search[field] = search.get(field) or search.get("page_arg") or ""
        return search

    def tool_of_search(self, args: dict[str, Any]) -> str:
        return self.search.get("words_tool") if self.search.get("words_arg") in args else (self.search.get("shelf_tool") or self.search.get("words_tool") or "")

    def as_written(self, budget: float | None, words: Any = "") -> float | None:
        """A budget read smaller than the request writes it was converted to a larger unit: the written number stands.
        A larger reading is kept, since the request's words can multiply the number. The words must really be there."""
        text = str(words or "")
        if budget is None or not text or text.lower() not in self.request.lower():
            return budget
        found = re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))
        return float(found[0]) if len(found) == 1 and float(found[0]) > budget else budget

    def steps_of_procedure(self, stage: str) -> list[str]:
        return [str(s.get("step")) for s in _dicts(self.brief.get("steps")) if s.get("when") in (stage, "any")]

    def messages_free(self) -> int:
        kept = sum(1 for s in _dicts(self.brief.get("steps")) if s.get("uses_message") and s.get("when") != "before_end")
        return self.msg_limit - self.msg_used - kept

    # ---- scouting: learn each item's properties ----
    def shelf_args(self, line: Line, query: str | None = None) -> dict[str, Any]:
        s = self.search
        if query:
            args = dict(s.get("words_fixed") or {})
            args[s.get("words_arg", "")] = query[:int(_number(s.get("words_max")) or 200)]
            if s.get("shelf_with_words") and s.get("shelf_arg") and line.shelf and s.get("words_tool") == s.get("shelf_tool"):
                args[s["shelf_arg"]] = line.shelf
        else:
            args = dict(s.get("shelf_fixed") or {})
            if s.get("shelf_arg"):
                args[s["shelf_arg"]] = line.shelf or line.name
        return args

    def learn_shelf(self, line: Line) -> None:
        """Take the store's own label for this item from the rows seen. The label that matches the item as a whole
        becomes the only one accepted, so a neighbouring category sharing a word is not bought from."""
        mine = _words(line.shelf) or _words(line.name)
        scored: dict[str, float] = {}
        for ident in line.ids:
            label = self.category(ident)
            words = _words(label)
            if label and label not in scored and words & (mine | _words(line.name)):
                scored[label] = len(words & mine) / len(words | mine)
        if not scored:
            return
        line.related = True
        best = max(scored, key=lambda k: scored[k])
        if scored[best] >= 0.5:
            line.shelf, line.exact = best, True
        elif not line.shelf:
            line.shelf = best

    def browse_label(self, line: Line) -> None:
        """Browse the store's own label for the item, once it is known, if that has not been done yet."""
        if self.search.get("shelf_arg") and line.shelf and line.shelf not in line.browsed and line.ids:
            line.browsed.add(line.shelf)
            self.run_search(line, self.shelf_args(line))

    def scout(self) -> None:
        share = max(2, min(5, (self.max_steps // 4) // max(1, len(self.lines))))
        for line in self.lines:
            if self.search.get("shelf_arg"):
                line.browsed.add(line.shelf or line.name)
                self.run_search(line, self.shelf_args(line))
            # a browse shows a category in the store's own order, where the listings with a wanted value can sit far
            # behind the shown pages: the values are searched by keyword whatever the browse returned
            phrases = self.phrases(line)
            for query in query_ladder(line.name, phrases, line.head)[:2]:
                if not line.ids or (phrases and not line.word_hits):
                    self.run_search(line, self.shelf_args(line, query))
            if not line.ids:
                self.run_search(line, self.shelf_args(line, line.name))
            self.browse_label(line)
            # a tool that shows several listings in one step lets the first reads cover more of them
            self.read_listings(self.unread_by_fit(line)[:share * (2 if self.compare_room() >= 2 else 1)])
            self.set_aside_unanswerable(line)

    def want_words(self, line: Line) -> set[str]:
        words: set[str] = set()
        for need in line.live():
            if not need["pointer"] or need.get("needed"):
                words |= _words(phrase_of(need))
        return words

    def set_aside_unanswerable(self, line: Line) -> None:
        """A stated requirement that no listing read can speak to cannot decide between listings: it came from the story,
        not from the shelf, so it stops blocking fits. That takes at least three records judged, all left open, and no
        related key or such value among them: a store can file a requirement under a key of another name, and any
        verdict other than open shows it does. What the shopper revealed when asked is never set aside."""
        opened = [i for i in line.ids if self.pages.get(i)]
        if len(opened) < SET_ASIDE_AFTER:
            return
        keys: set[str] = set()
        values: set[str] = set()
        for ident in opened:
            for key, value in self.facts_of(ident).items():
                keys |= _words(key)
                values |= _words(value)
        if not keys:
            return  # pages whose attributes could not be found say nothing about any requirement
        for need in line.live():
            if need["told"] or need["pointer"] or need.get("checked") or need.get("soft"):
                continue
            said = [self.verdicts[(i, need["text"])][0] for i in opened if (i, need["text"]) in self.verdicts]
            if len(said) < SET_ASIDE_AFTER:
                continue
            need["checked"] = True
            if all(v == "unknown" for v in said) and not (_words(need["prop"]) & keys) and not (_words(phrase_of(need)) & values):
                need["live"] = False
                self.trace.append({"kind": "set_aside", "need": need["text"][:100]})

    def phrases(self, line: Line, soft: bool = False) -> list[str]:
        """Wanted values to search for, what the shopper revealed first. A soft need's words join no other search,
        since an all-words search loses every listing that lacks a story's word; `soft` asks for those alone."""
        ordered = sorted((n for n in line.live() if (not n["pointer"] or n.get("needed")) and n.get("soft", False) == soft), key=lambda n: not n["told"])
        found = [phrase_of(n) for n in ordered]
        return [p for p in found if p]

    def label_fits(self, ident: str, line: Line) -> bool:
        """Whether a listing's label allows it. A store may call an item something the request does not, so sharing no
        word rejects a label only while some other label does share one; otherwise the reading of kind decides."""
        label = _words(self.category(ident))
        if not label or line.wide:
            return True
        if line.exact:
            return label == _words(line.shelf)
        return bool(label & (_words(line.name) | _words(line.shelf))) or not line.related

    def unread_by_fit(self, line: Line) -> list[str]:
        """Unread listings of a line in the order worth reading for its pick rule."""
        wants = self.want_words(line)
        rows = [i for i in line.ids if i not in self.pages and i not in self.kind_bad and self.in_stock(i) and self.affordable(i, line) and self.label_fits(i, line)]

        def overlap(ident: str) -> int:
            return len(wants & _words(self.record_text(ident)))

        # a row that shows the wanted words is far likelier to fit than one that does not, whatever its price, and a
        # row a keyword search returned has matched some of them, so both are read before a row merely browsed
        if line.pick == "cheapest":
            cost = lambda i: self.price(i) if self.price(i) is not None else 1e12
            return sorted(rows, key=lambda i: (-overlap(i), 0 if line.word_hits.get(i) else 1, cost(i)))
        return sorted(rows, key=lambda i: (-overlap(i), -line.word_hits.get(i, 0)))

    def key_table(self, line: Line) -> str:
        seen: dict[str, dict[str, Any]] = {}
        read = [i for i in line.ids if self.pages.get(i)]
        for ident in read:
            for key, value in self.facts_of(ident).items():
                slot = seen.setdefault(key, {"n": 0, "values": []})
                slot["n"] += 1
                if str(value)[:30] not in slot["values"]:
                    slot["values"].append(str(value)[:30])
        ordered = sorted(seen.items(), key=lambda kv: -kv[1]["n"])
        return f"{len(read)} listings read\n" + "\n".join(f"  {k} (on {s['n']}, {len(s['values'])} values): {' | '.join(s['values'][:4])}" for k, s in ordered[:40])

    def facts_of(self, ident: str) -> dict[str, Any]:
        """A record's attributes: its largest dict, else a list of name and value pairs, else the page's own short plain
        values. A lone nested entry may be about something else, so it does not stand for the whole record."""
        page = self.pages.get(ident) or {}
        best: dict[str, Any] = {}
        for value in page.values():
            if isinstance(value, dict) and len(value) > len(best):
                best = value
        if len(best) >= 2:
            return best
        pairs = {}
        for value in page.values():
            for row in _dicts(value):
                parts = list(row.values())
                if len(parts) == 2 and isinstance(parts[0], str):
                    pairs[parts[0]] = parts[1]
        if len(pairs) >= 2:
            return pairs
        known = {self.shape.get(k) for k in ("id", "title", "price", "stock", "category", "seller")}
        plain = {k: v for k, v in page.items() if k not in known and isinstance(v, (str, int, float)) and len(str(v)) <= 80}
        return {**plain, **best}

    def rank_properties(self) -> None:
        body = "\n\n".join(f"ITEM {l.number}: {l.name}\nSTATED NEEDS: {'; '.join(n['text'] for n in l.live()) or 'none'}\nKEYS SEEN: {self.key_table(l)}" for l in self.lines)
        hints = _dump(self.brief.get("hints") or [])
        got = self.read(RANK + self.request + "\n\nHINTS: " + hints + "\n\n" + body, self.careful(), 4000, think=True) or {}
        for entry in _dicts(got.get("items")):
            line = self.line_of(entry.get("item"))
            for raw in _dicts(entry.get("properties")):
                name = re.split(r"\s+(?:or|vs)\s+", _plain(raw.get("name")))[0]
                p = _number(raw.get("p"))
                if p is not None and str(raw.get("hint") or "").strip():
                    # the request points at it: asked ahead of properties only the shelf speaks for
                    p = max(p, 0.5)
                if name and p is not None:
                    line.beliefs[name] = self.weigh(line, list(raw.get("keys") or []), min(max(p, 0.03), 0.9))
                    line.keys_of[name] = [_plain(k) for k in raw.get("keys") or []]
        for line in self.lines:
            if not line.beliefs:
                # the model gave nothing: fall back on the keys read, most carried first
                table = {}
                for ident in line.ids:
                    for key in self.facts_of(ident):
                        table[_plain(key)] = table.get(_plain(key), 0) + 1
                settled = _words(" ".join(n["prop"] for n in line.live()))
                for key, count in sorted(table.items(), key=lambda kv: -kv[1])[:12]:
                    if not _words(key) <= settled:
                        line.beliefs[key] = 0.1 + 0.02 * count
            self.trace.append({"kind": "beliefs", "item": line.number, "top": sorted(((round(p, 2), n) for n, p in line.beliefs.items()), reverse=True)[:10]})

    def weigh(self, line: Line, keys: list[Any], p: float) -> float:
        """Adjust the model's estimate by the shelf's evidence. Over eleven races, of the properties agents asked about,
        those that nearly every listing read carried held the shopper's preference about three times as often as those
        few carried (20% against 7%), and those with several values among the listings read about twice as often as
        those with one (14% against 6.5%). The factors follow those measurements."""
        wanted = {_plain(k) for k in keys}
        read = [i for i in line.ids if self.pages.get(i)]
        if not read:
            return p
        if not wanted:
            return p * 0.25  # no listing read carries it, so only the story speaks for it
        values: set[str] = set()
        carried = 0
        for ident in read:
            hits = [str(v) for k, v in self.facts_of(ident).items() if _plain(k) in wanted]
            carried += bool(hits)
            values.update(hits)
        factor = 0.25 + 0.75 * carried / len(read)
        if carried >= 3:
            factor *= {1: 0.45, 2: 0.75}.get(len(values), 1.0)
        if carried >= 3 and len(values) >= carried and sum(map(len, values)) / len(values) > 20:
            factor *= 0.4  # long text differing on every listing reads as an identifier or an option list
        return p * factor

    # ---- the interview ----
    def next_message(self, late: bool = False) -> list[tuple[Line, list[str]]]:
        # `late`: the procedures have had their messages, so none is kept back for them any more
        free = self.msg_limit - self.msg_used if late else self.messages_free()
        live = [l for l in self.lines if any(p >= MIN_BELIEF for p in l.beliefs.values())]
        if free <= 0 or not live:
            return []
        fresh = [l for l in live if l.asked == 0]
        # one item per message, an item not yet asked about first: a message about several items reads to the shopper
        # as several questions at once. With more items than messages, the likeliest to hold a preference are asked.
        group = [max(fresh or live, key=lambda l: line_gain(l.beliefs, self.name_cap))]
        cap = max(2, self.name_cap // len(group)) if self.name_cap > 1 else 1
        parts = []
        for line in group:
            names = self.names_in_order(line, cap)
            if names:
                parts.append((line, names))
        return self.trim_to_length(parts)

    def names_in_order(self, line: Line, cap: int) -> list[str]:
        """The likeliest names, each counting less for every time it was already asked. A reply that answered nothing
        may mean a name near the front drew it, so what stood behind then goes in front now: untried names first,
        then the last message's names reversed."""
        weight = lambda name: line.beliefs[name] * REPEAT ** line.times_asked.get(name, 0)
        names = [n for n in sorted(line.beliefs, key=lambda n: -weight(n)) if line.beliefs[n] >= MIN_BELIEF][:cap]
        if not line.last_silent:
            return names
        return [n for n in names if n not in line.last_names] + [n for n in reversed(line.last_names) if n in names]

    def trim_to_length(self, parts: list[tuple[Line, list[str]]]) -> list[tuple[Line, list[str]]]:
        spec = ((self.schemas.get(self.roles.get("ask", "")) or {}).get("properties") or {}).get(self.roles.get("ask_text", "")) or {}
        limit = spec.get("maxLength") if isinstance(spec.get("maxLength"), int) else 800
        while len(question_text([(l.label(), n) for l, n in parts])) > limit and any(len(n) > 1 for _, n in parts):
            longest = max(parts, key=lambda part: len(part[1]))
            longest[1].pop()
        return parts

    def interview(self) -> None:
        if not self.roles.get("ask") or not self.roles.get("ask_text"):
            return
        while not self.finished and self.steps_left() > self.tail() and self.time_left() > HURRY:
            parts = self.next_message()
            if not parts:
                return
            text = question_text([(l.label(), n) for l, n in parts])
            self.msg_used += 1
            reply = self.call(self.roles["ask"], {self.roles["ask_text"]: text})
            words = _dump(reply)
            self.trace.append({"kind": "ask", "q": text, "a": words[:300]})
            if self.take_reply(parts, text, words) == "refused":
                # the store refused the message: its real limit is what was used before this one
                self.msg_used -= 1
                self.msg_limit = self.msg_used
                return

    def take_reply(self, parts: list[tuple[Line, list[str]]], text: str, words: str) -> Any:
        """Apply a reply to the beliefs and needs. Returns how the reply was read."""
        items = "\n".join(f"item {l.number}: {l.name}" for l in self.lines)
        got = self.read(READ_REPLY + f"PRODUCTS:\n{items}\n\nMESSAGE: {text}\n\nREPLY: {words[:900]}", [LIGHT, HEAVY], 500) or {}
        kind = got.get("kind")
        if kind == "refused":
            return kind
        for line, _ in parts:
            line.asked += 1
        if kind == "too_many":
            # a shopper who asks for one thing at a time asks it again for any wider message
            self.name_cap = 1
            return kind
        if self.name_cap > 1 and len(parts[0][1]) > 1 and kind in ("need", "no_view", "other"):
            # several names were taken without complaint: later messages may name more
            self.name_cap = MAX_NAMES
        needs = drop_descriptions(_dicts(got.get("needs")), words[:900])
        found: dict[int, list[dict[str, Any]]] = {}
        for raw in needs:
            line = self.line_of(raw.get("item") or parts[0][0].number)
            if not raw.get("pointer"):
                raw = {**raw, "want": unframed(str(raw.get("want") or ""))}
            added = line.add_need(raw, True)
            if added:
                found.setdefault(line.number, []).append(added)
        budget = as_stated(_number(got.get("budget")), words)
        if budget is not None:
            self.budget = budget
        self.drop_needs(got.get("drop"))
        for line, names in parts:
            for name in names:
                line.times_asked[name] = line.times_asked.get(name, 0) + 1
            answered = found.get(line.number, [])
            hit = None
            for need in answered:
                place = self.match_name(line, names, need["prop"])
                hit = place if hit is None else (hit if place is None else min(hit, place))
                self.forget_property(line, need["prop"])
            # a preference on a property this message did not list says nothing about the names it did list
            if hit is not None or not answered:
                update_beliefs(line.beliefs, names, hit)
            line.last_names = list(names)
            line.last_silent = not answered
            # the front name of an unanswered message was almost surely heard and holds nothing, and listed again
            # it could only stand in front of a name that does
            if not answered and names and names[0] in line.beliefs:
                line.beliefs[names[0]] = 0.0
        return kind

    def match_name(self, line: Line, names: list[str], prop: str) -> int | None:
        wanted = _words(prop)
        for place, name in enumerate(names):
            keys = [_words(k) for k in line.keys_of.get(name, [])] + [_words(name)]
            if any(k and (k <= wanted or wanted <= k) for k in keys):
                return place
        return None

    def forget_property(self, line: Line, prop: str) -> None:
        """A property is settled: no spelling of it is asked again."""
        wanted = _words(prop)
        for name in list(line.beliefs):
            keys = [_words(k) for k in line.keys_of.get(name, [])] + [_words(name)]
            if any(k and (k <= wanted or wanted <= k) for k in keys):
                line.beliefs[name] = 0.0

    # ---- requirements that point at other listings ----
    def resolve_pointers(self) -> None:
        for line in self.lines:
            for need in line.live():
                if need["pointer"] and not need["value"]:
                    self.follow(line, need)

    def follow(self, line: Line, need: dict[str, Any]) -> None:
        opened: list[str] = []
        shown: list[str] = []
        for _ in range(6):
            if self.steps_left() <= self.tail() + 2 or self.time_left() < HURRY or self.cost >= COST_STOP:
                return
            # pages and rows lead with long free text, so long values are cut short: a row must still show its id and
            # seller, and a page the remark that names a value, a rule or the next listing
            pages = "\n".join(f"ID {i}: {self.brief_text(i, 400)[:3000]}" for i in opened[-3:]) or "(none)"
            rows = "\n".join(f"ID {i}: {self.brief_text(i, 60)[:500]}" for i in shown[-15:]) or "(none)"
            got = self.read(POINTER + f"REQUIREMENT: {need['text']}\n\nPAGES OPENED:\n{pages}\n\nLISTINGS SHOWN:\n{rows}", [LIGHT, HEAVY], 500) or {}
            found = str(got.get("found") or got.get("value") or "").strip()
            needed = str(got.get("needed") if "needed" in got else found).strip()
            target = str(got.get("open") or "")
            words = str(got.get("search") or "").strip()
            if found or needed:
                need["value"], need["needed"] = found or needed, needed
                # a value with no needed one yet may wait on a rule a further page states: follow on if asked to
                if needed or not (target or words):
                    self.trace.append({"kind": "pointer", "need": need["text"][:120], "found": found[:80], "needed": needed[:80]})
                    return
            if target:
                if target in opened or (target not in self.rows and target not in self.pages):
                    return
                self.open_listing(target)
                if target not in self.offer_read and self.roles.get("offer") and self.roles.get("offer_id") and self.steps_left() > self.tail() + 2:
                    # a store that splits a listing keeps its remarks on the other page
                    self.offer_read.add(target)
                    extra = self.call(self.roles["offer"], {self.roles["offer_id"]: target})
                    if isinstance(extra, dict):
                        self.pages[target] = {**self.pages.get(target, {}), **extra}
                opened.append(target)
                continue
            if not words:
                return
            # a search for the other listing must not add to this item's own pool or page queue
            aside = Line(line.number, {"name": line.name, "shelf": str(got.get("shelf") or "")})
            self.run_search(aside, self.shelf_args(aside, words))
            # a description of the other listing names more than its pages say word for word
            for shorter in aside.queries[:2]:
                if aside.ids or self.steps_left() <= self.tail() + 2:
                    break
                self.run_search(aside, self.shelf_args(aside, shorter))
            shown += aside.ids

    # ---- verdicts ----
    def requirement_text(self, need: dict[str, Any]) -> str:
        if not need["value"]:
            return need["text"]
        extra = f"; a listing meeting it needs: {need['needed']}" if need.get("needed") and need["needed"] != need["value"] else ""
        return need["text"] + f" (the listing it points at gives: {need['value']}{extra})"

    def title_of(self, ident: str) -> str:
        key = self.shape.get("title")
        for record in (self.pages.get(ident), self.rows.get(ident)):
            if record and key and record.get(key):
                return str(record[key])
        return ""

    def has_table(self, ident: str) -> bool:
        """Whether a listing's record shows attributes, which then alone decide its requirements."""
        return len(self.facts_of(ident)) >= 2

    def wanted_of(self, need: dict[str, Any]) -> str | None:
        """The value a listing must hold to meet a need, when there is one to compare: a pointer's only once its pages
        gave one."""
        if need.get("pointer"):
            return need.get("needed") or None
        return need["want"]

    def checked_verdict(self, ident: str, verdict: str, key: Any, quote: Any, prop: str = "", want: Any = "") -> str:
        """A model's verdict stands only on the attribute it names: a yes or a no naming no attribute of the record is
        open, as is one naming another attribute when the record has attributes naming the property, and a yes must
        quote words that attribute holds. A record without attributes keeps the older rule: a yes must quote words the
        record holds. A yes is open too when the wanted value has a word closing with a plus sign that the words
        deciding it lack."""
        if verdict not in ("yes", "no"):
            return "unknown"
        if verdict == "yes" and not _marked(want) <= _marked(quote):
            return "unknown"
        said = _plain(quote)
        if not self.has_table(ident):
            return "unknown" if verdict == "yes" and said and said not in _plain(self.record_text(ident)) else verdict
        named = _plain(key)
        if _words(named) & _OFFERED:
            return "unknown"  # what is on offer or a search filter says nothing of this listing's own value
        facts = self.facts_of(ident)
        naming = [_plain(k) for k in facts if names_property(k, prop)] if prop else []
        held = next((str(v) for k, v in facts.items() if named and _plain(k) == named), None)
        if held is None:
            # with no attribute naming the property, the title or the description may decide: a quote the record holds
            if naming or not said or said not in _plain(self.record_text(ident)):
                return "unknown"
            return verdict
        if naming and named not in naming:
            return "unknown"
        if verdict == "yes" and said and said not in _plain(held):
            return "unknown"
        return verdict

    def judge(self, line: Line) -> None:
        self.set_aside_unanswerable(line)
        needs = line.live()
        ids = [i for i in line.ids if self.pages.get(i) and ((i, line.number) not in self.judged or any((i, n["text"]) not in self.verdicts for n in needs))]
        for start in range(0, len(ids), 5):
            batch = ids[start:start + 5]
            listed = "\n".join(f"{n + 1}. {self.requirement_text(need)}" for n, need in enumerate(needs)) or "(none)"
            extra = f"\n\"measure\" is the listing's value of: {line.measure}" if line.measure else ""
            records = "\n".join(f"ID {i}{'' if self.has_table(i) else ' (NO TABLE)'}: {self.record_text(i)}" for i in batch)
            got = self.read(JUDGE + f"ITEM: {line.name} {('(category ' + line.shelf + ')') if line.shelf else ''}{extra}\n\nREQUIREMENTS:\n{listed}\n\nRECORDS:\n{records}", [LIGHT, HEAVY], 3000) or {}
            for entry in _dicts(got.get("listings")):
                ident = str(entry.get("id"))
                if ident not in batch:
                    continue
                self.judged.add((ident, line.number))
                # once the store's own label for the item is known, that label settles the kind
                if entry.get("kind_ok") is False and not line.exact:
                    self.kind_bad.add(ident)
                measure = _number(entry.get("measure"))
                if measure is not None:
                    self.measures[ident] = measure
                for raw in _dicts(entry.get("verdicts")):
                    place = int(_number(raw.get("n")) or 0) - 1
                    if 0 <= place < len(needs):
                        verdict = self.checked_verdict(ident, str(raw.get("verdict")), raw.get("key"), raw.get("quote"), needs[place]["prop"],
                                                       self.wanted_of(needs[place]) or "")
                        self.verdicts[(ident, needs[place]["text"])] = (verdict, str(raw.get("quote"))[:80])
            for ident in batch:
                self.spell(line, [ident])
                if all((ident, n["text"]) in self.verdicts for n in needs):
                    self.judged.add((ident, line.number))
                self.trace.append({"kind": "verdict", "id": ident, "v": [self.verdicts.get((ident, n["text"]), ("?", ""))[0] for n in needs], "q": [self.verdicts.get((ident, n["text"]), ("", ""))[1][:40] for n in needs]})
        # each listing read widens what the store is seen to file a value under, which can settle a listing read
        # before, so the spelling pass covers every listing read
        self.spell(line, [i for i in line.ids if self.pages.get(i)])

    def filed_under(self, line: Line, need: dict[str, Any]) -> set[frozenset]:
        want = self.wanted_of(need)
        records = [self.facts_of(i) for i in line.ids if self.pages.get(i) and self.has_table(i)]
        return holding_families(records, need["prop"], want) if want else set()

    def spell(self, line: Line, ids: list[str]) -> None:
        """What spelling alone settles stands whatever a reading said, and stands when no reading came back."""
        item = " ".join((line.name, line.head, line.shelf))
        for need in line.live():
            want = self.wanted_of(need)
            if not want:
                continue
            filed = self.filed_under(line, need)
            for ident in ids:
                facts = self.facts_of(ident) if self.has_table(ident) else {}
                settled = decide_by_spelling(facts, need["prop"], want, item, self.title_of(ident), filed)
                if settled:
                    self.verdicts[(ident, need["text"])] = (settled[0], settled[2][:80])

    def verdict_of(self, ident: str, need: dict[str, Any]) -> str:
        """The listing's own category settles a need whose whole value it states, whatever other attributes say."""
        wanted = _words(need.get("needed") or ("" if need.get("pointer") else need["want"]))
        if wanted and wanted <= _words(self.category(ident)):
            return "yes"
        return self.verdicts.get((ident, need["text"]), ("unknown", ""))[0]

    def verdicts_of(self, ident: str, line: Line) -> list[str]:
        return [self.verdict_of(ident, n) for n in line.live()]

    def hard_verdicts(self, ident: str, line: Line) -> list[str]:
        return [self.verdict_of(ident, n) for n in line.live() if not n.get("soft")]

    def soft_misses(self, ident: str, line: Line) -> int:
        return sum(self.verdict_of(ident, n) != "yes" for n in line.live() if n.get("soft"))

    def fits(self, line: Line) -> list[str]:
        out = [i for i in line.ids if self.pages.get(i) and i not in self.kind_bad and self.label_fits(i, line) and self.in_stock(i) and self.affordable(i, line) and all(v == "yes" for v in self.hard_verdicts(i, line))]
        return self.by_pick(out, line)

    def by_pick(self, ids: list[str], line: Line) -> list[str]:
        """Fits in the order the pick rule wants, those meeting the soft needs first."""
        cost = lambda i: self.price(i) if self.price(i) is not None else 1e12
        if line.pick == "most":
            return sorted(ids, key=lambda i: (self.soft_misses(i, line), -self.measures.get(i, -1e12), cost(i)))
        if line.pick == "least":
            return sorted(ids, key=lambda i: (self.soft_misses(i, line), self.measures.get(i, 1e12), cost(i)))
        return sorted(ids, key=lambda i: (self.soft_misses(i, line), cost(i)))

    def filed_soft(self, line: Line) -> list[dict[str, Any]]:
        """The soft needs whose words some listing read carries among its attribute values: the store files such a
        value, so the need can be met and searched for. A purpose the shelf never states is the story's."""
        values: set[str] = set()
        for ident in line.ids:
            if self.pages.get(ident):
                for value in self.facts_of(ident).values():
                    values |= _words(value)
        return [n for n in line.live() if n.get("soft") and _words(phrase_of(n)) and _words(phrase_of(n)) <= values]

    def soft_open(self, line: Line, fits: list[str]) -> bool:
        """Whether the hunt should go on for a fit meeting the soft needs: none does yet, some listing read carries the
        soft need's words among its attribute values, so the store files such a value, and too few listings were
        judged to say that none will. A purpose the shelf never states is the story's, and waiting for it would
        only spend the steps a refused order needs."""
        if not self.filed_soft(line) or any(self.soft_misses(i, line) == 0 for i in fits):
            return False
        return len([i for i in line.ids if self.pages.get(i)]) < SET_ASIDE_AFTER

    def score(self, ident: str, line: Line) -> float:
        """Rough chance that this listing is right for the line."""
        if ident in self.kind_bad or not self.label_fits(ident, line) or not self.in_stock(ident) or not self.affordable(ident, line):
            return 0.0
        if not self.pages.get(ident):
            return UNREAD_SCORE
        value = FIT_SCORE
        for need in line.live():
            verdict = self.verdict_of(ident, need)
            if need.get("soft"):
                value *= {"yes": 1.0, "unknown": SOFT_UNKNOWN}.get(verdict, SOFT_NO)
            else:
                value *= {"yes": 1.0, "unknown": UNKNOWN_SCORE}.get(verdict, NO_SCORE)
        return value

    # ---- the hunt ----
    def line_done(self, line: Line) -> bool:
        fits = self.fits(line)
        if not fits or self.soft_open(line, fits):
            return False
        if line.pick == "any":
            # any verified fit will do. A replacement is worth steps only if this one is refused at checkout, so the
            # steps wait until then instead of being spent on spares that may never be needed.
            return True
        if line.pick == "cheapest":
            best = self.price(fits[0])
            unread = self.unread_by_fit(line)
            cheaper = [i for i in unread if best is not None and (self.price(i) or 0) < best]
            arg = self.search.get("price_arg")
            if arg:
                return line.capped_at == best and not cheaper and not any(arg in page for page in line.more)
            # the searches planned for the line run before its cheapest fit is trusted
            return not cheaper and not line.more and not line.queries and line.queries_made
        return not self.unread_by_fit(line) and not line.more

    def make_queries(self, line: Line) -> None:
        if line.queries_made:
            return
        line.queries_made = True
        filed = self.filed_soft(line)
        needs = "; ".join(self.requirement_text(n) for n in line.live() if not n.get("soft") or n in filed) or "none"
        tried = f"\nTHESE FOUND NOTHING THAT FITS, SO TRY OTHER WORDS: {'; '.join(line.tried)}" if line.tried else ""
        got = self.read(QUERIES + f"ITEM: {line.name}\nREQUIREMENTS: {needs}{tried}", [LIGHT, HEAVY], 400) or {}
        written = [str(q) for q in got.get("queries") or [] if str(q).strip()]
        # searches from code come first and do not depend on a model answering; on a second pass only new words help
        noun = _plain(line.head) or (_plain(line.name).split() or [""])[-1]
        aside = [f"{phrase_of(n)} {noun}".strip() for n in filed if phrase_of(n)]
        planned = (written if line.tried else query_ladder(line.name, self.phrases(line), line.head) + aside + written)
        # shorter forms already queued for a search that matched nothing stay in front
        line.queries = [q for q in dict.fromkeys(line.queries + planned) if q not in line.tried][:8] or ([line.name] if not line.tried else [])
        line.tried += line.queries

    def look_below(self, line: Line) -> bool:
        """On a cheapest line, list the shelf under the best price found, so only cheaper listings are left to read."""
        arg = self.search.get("price_arg")
        fits = self.fits(line)
        if line.pick != "cheapest" or not arg or not fits:
            return False
        best = self.price(fits[0])
        if best is None or line.capped_at == best:
            return False
        line.capped_at = best
        # A store shows only so many pages, so a limit matching more than that hides some of its cheapest rows. Look
        # for the highest limit whose matches can all be shown: that is the widest band that hides nothing.
        pages = self.page_reach()
        low, high, limit = 0.0, best, best
        kept: dict[str, Any] | None = None
        for _ in range(5):
            if self.steps_left() <= self.tail() + 2:
                break
            before = len(line.more)
            result = self.run_search(line, {**self.shelf_args(line), arg: limit})
            if not isinstance(result, dict):
                break
            new_pages = line.more[before:] if len(line.more) > before else [p for p in line.more if p.get(arg) == limit]
            line.more = [p for p in line.more if p.get(arg) is None]
            total = _number(result.get(self.shape.get("total"))) if self.shape.get("total") else None
            shown = len(self.rows_in(result))
            if total is None:
                line.more += new_pages
                return True
            if total > shown * pages:
                high = limit
            else:
                low = limit
                kept = new_pages[0] if new_pages else None
            if limit == best and total <= shown * pages:
                break
            limit = round((low + high) / 2, 2)
            if high - low < max(1.0, best * 0.05):
                break
        if kept:
            line.more.append(kept)
        return True

    def page_reach(self) -> int:
        """How many pages of one search the store's schema allows; three when it does not say."""
        spec = ((self.schemas.get(self.search.get("shelf_tool", "")) or {}).get("properties") or {}).get(self.search.get("shelf_page_arg", "")) or {}
        top = spec.get("maximum")
        return top if isinstance(top, int) and spec.get("minimum") == 1 and top > 0 else 3

    def hunt(self, line: Line, allowance: int, repair: bool = False) -> None:
        """Search and read for one line within `allowance` steps. Exploring stops at COST_STOP, except to `repair` a
        cart a refusal broke, which may spend on to COST_HARD: the episode is nearly done and a replacement decides it."""
        stop = self.step + allowance
        self.make_queries(line)
        while not self.finished and self.step < stop and self.steps_left() > self.tail() and self.time_left() > HURRY and (repair or self.cost < COST_STOP):
            self.judge(line)
            self.browse_label(line)
            # a price limit lists a category without the wanted words, so it waits until the keyword searches, which
            # reach the listings that carry them, are done
            if (not line.queries or not self.phrases(line)) and self.look_below(line):
                continue
            if self.line_done(line):
                return
            # after the first search of this hunt, read a few listings before searching again
            unread = self.unread_by_fit(line)
            # later pages under a price limit hold the cheapest candidates, ahead of any new keyword search
            limited = [page for page in line.more if self.search.get("price_arg") in page]
            if not unread and limited:
                line.more.remove(limited[0])
                self.run_search(line, limited[0])
                continue
            if line.queries and (not unread or len([i for i in line.ids if self.pages.get(i)]) % 4 == 3):
                self.run_search(line, self.shelf_args(line, line.queries.pop(0)))
                continue
            if unread:
                self.read_listings(unread[:max(2, self.compare_room())], stop)
                continue
            if line.more:
                self.run_search(line, line.more.pop(0))
                continue
            if line.queries:
                self.run_search(line, self.shelf_args(line, line.queries.pop(0)))
                continue
            if not self.fits(line) and not line.rethought:
                # steps remain and nothing fits: one fresh set of searches before giving up
                line.rethought = True
                line.queries_made = False
                self.make_queries(line)
                continue
            if not self.fits(line) and not line.wide and not line.exact and not line.named:
                # the labels near the item's words gave no fit and none is its whole name, so any label may be read
                # before giving up, with the reading of kind deciding. A category the request named is never left.
                line.wide = True
                if self.unread_by_fit(line):
                    continue
            return
        self.judge(line)

    def hunt_all(self) -> None:
        rounds = 0
        while rounds < 2 and not self.finished and self.time_left() > HURRY and self.cost < COST_STOP:
            rounds += 1
            open_lines = [l for l in self.lines if not self.line_done(l)]
            if not open_lines:
                return
            # a line the shopper just told a requirement for has the least known fit: it is searched first
            open_lines.sort(key=lambda l: not any(n["told"] for n in l.live()))
            room = self.steps_left() - self.tail()
            if room <= 0:
                return
            for index, line in enumerate(open_lines):
                self.hunt(line, max(2, room // max(1, len(open_lines) - index)))
                room = self.steps_left() - self.tail()

    # ---- the finish ----
    def choose(self, fill: bool = False) -> dict[str, int]:
        """The best listing per line, under the budget, each line's backups kept for later swaps. A line with no
        verified fit is left out, since an unverified pick is worth less than leaving it out, unless `fill` asks
        for the best guess on every line."""
        picks: dict[int, str] = {}
        for line in self.lines:
            taken = set(picks.values())
            fits = [i for i in self.fits(line) if i not in taken]
            if fits:
                picks[line.number] = fits[0]
                continue
            pool = [i for i in line.ids if i not in taken]
            # equal guesses go to the cheaper listing, which keeps a budget and a cheapest pick safest
            best = max(pool, key=lambda i: (self.score(i, line), -(self.price(i) if self.price(i) is not None else 1e12)), default=None)
            if best is not None and self.score(best, line) > (0.0 if fill else LEFT_OUT):
                picks[line.number] = best
        if self.budget is not None and not self.budget_each:
            for _ in range(len(self.lines) * 4):
                total = sum((self.price(i) or 0) * self.line_of(n).quantity for n, i in picks.items())
                if total <= self.budget:
                    break
                swaps = []
                for number, ident in picks.items():
                    line = self.line_of(number)
                    for other in self.fits(line):
                        if other in picks.values():
                            continue
                        saved = ((self.price(ident) or 0) - (self.price(other) or 0)) * line.quantity
                        if saved > 0:
                            swaps.append((saved, number, other))
                if not swaps:
                    break
                _, number, other = max(swaps)
                picks[number] = other
        return {i: self.line_of(n).quantity for n, i in picks.items()}

    def open_picks(self, cart: dict[str, int]) -> bool:
        """Open each listing of a cart about to be ordered that was never opened: in version 3's race every order that
        held a listing never opened failed. Returns whether a page was read, so the cart is chosen again from it."""
        unread = [i for i in cart if not self.pages.get(i) and i not in self.open_tried]
        room = self.steps_left() - (len(cart) + 2)
        if not unread or room <= 0:
            return False
        unread = unread[:room]
        self.open_tried.update(unread)
        if self.roles.get("facts") and self.roles.get("facts_id"):
            for ident in unread:
                self.open_listing(ident)
        elif self.compare_room() >= 1:
            self.compare(unread[:self.compare_room()])
        opened = [i for i in unread if self.pages.get(i)]
        self.trace.append({"kind": "open_picks", "asked": unread, "opened": opened})
        if not opened:
            return False
        for line in self.lines:
            self.judge(line)
        return True

    def press_budget(self) -> None:
        """The cart breaks the total budget. A line that takes any fit stopped at its first, so cheaper fits may be
        unread: those lines now look for their cheapest, dearest first, and spare steps go to them."""
        self.pressed = True
        lines = [l for l in self.lines if l.pick == "any"]
        lines.sort(key=lambda l: -max((self.price(i) or 0 for i in self.fits(l)[:1]), default=0))
        for index, line in enumerate(lines):
            line.pick = "cheapest"
            room = self.steps_left() - self.tail()
            if room > 0 and self.time_left() > HURRY:
                self.hunt(line, max(2, room // (len(lines) - index)))

    def order_chance(self, cart: dict[str, int]) -> float:
        if not cart:
            return 0.0
        chance = NO_SCORE if self.over_budget(cart) else 1.0
        for line in self.lines:
            scores = [self.score(i, line) for i in cart if i in line.ids]
            chance *= max(scores) if scores else LEFT_OUT
        return chance

    def end_chance(self) -> float:
        missing = [l for l in self.lines if not self.fits(l)]
        if not missing:
            return END_BASE
        value = END_OPEN
        for line in missing:
            exhausted = line.searches >= END_SEARCHES and not line.queries and not line.more and not self.unread_by_fit(line)
            if exhausted and self.ruled_out(line):
                told = any(n["told"] for n in line.live())
                value = max(value, END_TOLD if told else END_STATED)
        return value

    def ruled_out(self, line: Line) -> bool:
        """Evidence that nothing fits a line: listings of its kind were read, and each one a record contradicts on some
        requirement. Listings that leave a requirement open are no such evidence: they may meet it."""
        read = [i for i in line.ids if self.pages.get(i) and i not in self.kind_bad and self.label_fits(i, line)]
        if not read:
            return False
        for ident in read:
            usable = self.in_stock(ident) and self.affordable(ident, line)
            if usable and "no" not in self.hard_verdicts(ident, line):
                return False
        return True

    def set_cart(self, wanted: dict[str, int]) -> None:
        add, remove = self.roles.get("add") or {}, self.roles.get("remove") or {}
        if add.get("lines"):
            # this store sets the whole cart in one call
            if add.get("tool") and self.cart != wanted:
                lines = [{add.get("id"): ident, **({add["qty"]: count} if add.get("qty") else {})} for ident, count in wanted.items()]
                reply = self.call(add["tool"], {**(add.get("fixed") or {}), add["lines"]: lines})
                if not self.refused(add["tool"], reply):
                    self.cart = dict(wanted)
            return
        for ident, count in list(self.cart.items()):
            if ident not in wanted:
                self.take_out(ident, count)
        for ident, count in wanted.items():
            have = self.cart.get(ident, 0)
            if have < count and add.get("tool"):
                args = {**(add.get("fixed") or {}), add.get("id"): ident}
                if add.get("qty"):
                    args[add["qty"]] = count - have
                reply = self.call(add["tool"], args)
                if not self.refused(add["tool"], reply):
                    self.cart[ident] = count

    def take_out(self, ident: str, count: int) -> None:
        """Remove a cart line. A store whose only cart tool sets a line's quantity removes a line set to zero: the
        same tool sent with the line's count would only keep it."""
        add, remove = self.roles.get("add") or {}, self.roles.get("remove") or {}
        same = not remove.get("tool") or (remove.get("tool") == add.get("tool") and (remove.get("fixed") or {}) == (add.get("fixed") or {}))
        if same:
            spec = ((self.schemas.get(add.get("tool", "")) or {}).get("properties") or {}).get(add.get("qty", "")) or {}
            low = spec.get("minimum")
            if not add.get("tool") or not add.get("qty") or (isinstance(low, (int, float)) and low > 0):
                return
            self.call(add["tool"], {**(add.get("fixed") or {}), add.get("id"): ident, add["qty"]: 0})
        else:
            args = {**(remove.get("fixed") or {}), remove.get("id"): ident}
            if remove.get("qty"):
                args[remove["qty"]] = count
            self.call(remove["tool"], args)
        self.cart.pop(ident, None)

    def known_in(self, view: Any) -> set[str]:
        """Listing ids a cart view shows: any value that is the id of a listing seen."""
        found: set[str] = set()
        if isinstance(view, dict):
            for value in view.values():
                found |= self.known_in(value)
        elif isinstance(view, list):
            for value in view:
                found |= self.known_in(value)
        elif isinstance(view, str) and (view in self.rows or view in self.pages):
            found.add(view)
        return found

    def check_cart(self, cart: dict[str, int]) -> bool:
        """A last independent read. Returns False when it found a requirement a record contradicts."""
        # a careful read is what makes this check worth its cost: without room for the heavy model it is skipped
        lines = []
        for line in self.lines:
            listed = "; ".join(f"{n + 1}. {self.requirement_text(need)}" for n, need in enumerate(line.live())) or "none"
            for ident in cart:
                if ident in line.ids:
                    lines.append(f"ITEM {line.name}, requirements: {listed}\nRECORD {ident}: {self.record_text(ident)}")
        prompt = CHECK + "\n\n".join(lines)
        if not self.heavy_fits(prompt, 700):
            return True
        got = self.read(prompt, [HEAVY, LIGHT], 700) or {}
        wrong = False
        for raw in _dicts(got.get("wrong")):
            ident = str(raw.get("id"))
            for line in self.lines:
                needs = line.live()
                place = int(_number(raw.get("n")) or 0) - 1
                if ident in line.ids and 0 <= place < len(needs):
                    need = needs[place]
                    if need.get("soft"):
                        continue
                    want = self.wanted_of(need)
                    facts = self.facts_of(ident) if self.has_table(ident) else {}
                    # spelling that settles a yes outranks this reading, and a contradiction must stand on the attribute
                    # that is the property: a title, a variant list or another attribute does not decide
                    settled = decide_by_spelling(facts, need["prop"], want, " ".join((line.name, line.head, line.shelf)), self.title_of(ident), self.filed_under(line, need)) if want else None
                    if settled and settled[0] == "yes":
                        continue
                    if facts and self.checked_verdict(ident, "no", raw.get("key"), "", need["prop"]) != "no":
                        continue
                    self.verdicts[(ident, need["text"])] = ("no", str(raw.get("why"))[:80])
                    wrong = True
        return not wrong

    def gate(self, stage: str) -> None:
        steps = self.steps_of_procedure(stage)
        if not steps:
            return
        ending = {self.roles.get("buy"), self.roles.get("end")}
        tools = "\n".join(f"- {t['function']['name']}: {t['function'].get('description', '')} | {_dump(t['function'].get('parameters') or {})}" for t in self.tools if t["function"]["name"] not in ending)
        for _ in range(3):
            if self.steps_left() <= 2 or self.time_left() < LAST_MODEL + 10:
                return
            got = self.read(GATE + "PROCEDURES:\n" + "\n".join(steps) + f"\n\nCART: {_dump(self.cart)}\nMESSAGES LEFT: {self.msg_limit - self.msg_used}\n\nRECORD:\n" + "\n".join(self.done_log[-30:]) + "\n\nTOOLS:\n" + tools, [LIGHT, HEAVY], 400) or {}
            call = got.get("call")
            if not isinstance(call, dict) or call.get("tool") not in self.schemas or call.get("tool") in ending:
                return
            if call["tool"] == self.roles.get("ask"):
                # a message goes out only for a procedure the request asks a message for: in version 3's race, messages
                # sent to confirm a pick or to ask about something the story mentioned were spent for nothing
                if not any(s.get("uses_message") for s in _dicts(self.brief.get("steps")) if s.get("when") in (stage, "any")):
                    self.trace.append({"kind": "gate_message_skipped", "stage": stage})
                    return
                self.msg_used += 1
            self.call(call["tool"], call.get("args") if isinstance(call.get("args"), dict) else {})

    def find_gone(self) -> None:
        """Something picked changed and the note did not say what: read the cart's lines afresh."""
        gone_note = self.pending_gone
        self.pending_gone = self.pending_price = False
        tool = self.roles.get("offer") or self.roles.get("facts")
        arg = self.roles.get("offer_id") if self.roles.get("offer") else self.roles.get("facts_id")
        for ident in list(self.cart):
            if tool and arg and self.steps_left() > len(self.lines) + 3:
                result = self.call(tool, {arg: ident})
                if isinstance(result, dict) and result:
                    self.pages[ident] = {**self.pages.get(ident, {}), **result}
                    self.rows[ident] = {**self.rows.get(ident, {}), **result}
        # records may not show stock at all; with one line in the cart, that line is the one the note meant
        if gone_note and len(self.cart) == 1 and all(self.in_stock(i) for i in self.cart):
            self.gone.update(self.cart)

    def try_order(self, cart: dict[str, int]) -> bool:
        self.changed = False
        self.set_cart(cart)
        if self.finished:
            return True
        self.gate("before_order")
        if self.late_question():
            # the reply told a requirement: the cart is chosen again before any order
            return False
        if self.roles.get("cart") and self.steps_left() > 4:
            view = self.call(self.roles["cart"], {})
            # a line the agent did not mean to keep, such as one a removal failed to take out, is taken out now
            for ident in sorted(self.known_in(view) - set(cart)):
                if self.steps_left() > 2:
                    self.take_out(ident, self.cart.get(ident, 1))
        if self.changed and self.steps_left() > len(self.lines) + 3:
            # the task changed while the cart was being set: solve it again before ordering, while steps allow
            if self.pending_gone or self.pending_price:
                self.find_gone()
            return False
        self.buy_tries += 1
        notes_before = self.notes_seen
        result = self.call(self.roles.get("buy", ""), dict(self.roles.get("buy_args") or {}))
        if self.finished:
            self.trace.append({"kind": "ordered", "cart": dict(self.cart), "step": self.step})
            return True
        # refused: unless a note already explained it, the reply's own words are read as one
        if result is not None and self.notes_seen == notes_before:
            self.on_note(_dump(result)[:1500])
        if self.pending_gone or self.pending_price:
            self.find_gone()
        return False

    def late_question(self) -> bool:
        """Once, before the first order: a message the procedures turned out not to need is spent on the next planned
        question. The grader passes an order only when every unwritten preference was asked about, however right the
        cart, and with two or three messages a kept-back one can be the item never asked about. Returns whether the
        reply changed a requirement."""
        if self.late_asked or not self.roles.get("ask") or not self.roles.get("ask_text"):
            return False
        if self.msg_limit - self.msg_used < 1 or self.steps_left() <= len(self.cart) + 3 or self.time_left() < HURRY / 2:
            return False
        self.late_asked = True
        parts = self.next_message(late=True)
        if not parts:
            return False
        text = question_text([(l.label(), n) for l, n in parts])
        before = sorted(n["text"] for l in self.lines for n in l.live())
        self.msg_used += 1
        reply = self.call(self.roles["ask"], {self.roles["ask_text"]: text})
        words = _dump(reply)
        self.trace.append({"kind": "ask", "q": text, "a": words[:300], "late": True})
        if self.take_reply(parts, text, words) == "refused":
            self.msg_used -= 1
            self.msg_limit = self.msg_used
            return False
        return sorted(n["text"] for l in self.lines for n in l.live()) != before

    def end_without_order(self) -> None:
        self.gate("before_end")
        notes_before = self.notes_seen
        result = self.call(self.roles.get("end", ""), dict(self.roles.get("end_args") or {}))
        if not self.finished and result is not None and self.notes_seen == notes_before:
            self.on_note(_dump(result)[:1500])
        self.trace.append({"kind": "ended", "step": self.step, "finished": self.finished})

    def finish(self) -> None:
        for _ in range(5):
            if self.finished or self.steps_left() <= 0:
                return
            for line in self.lines:
                self.judge(line)
            # a store that refused an ending wants an order: the best guess on every line
            cart = self.choose(fill=self.end_refused)
            if self.open_picks(cart):
                continue
            if self.over_budget(cart) and not self.pressed and self.steps_left() > self.tail() and self.time_left() > HURRY:
                self.press_budget()
                continue
            # the check costs money and time but no steps, so a short step count is no reason to skip it
            if self.time_left() > HURRY / 2 and not self.check_cart(cart):
                cart = self.choose(fill=self.end_refused)
            order, end = self.order_chance(cart), (0.0 if self.end_refused else self.end_chance())
            self.trace.append({"kind": "finish", "order": round(order, 3), "end": round(end, 3), "cart": cart, "fits": {l.number: len(self.fits(l)) for l in self.lines}})
            if order >= end and cart and self.roles.get("buy"):
                if self.try_order(cart):
                    return
                # a refusal changes the task: spare steps go to a line that lost every fit, or whose best price moved
                if self.steps_left() > self.tail() and self.time_left() > HURRY:
                    for line in self.lines:
                        if not self.fits(line) or line.pick == "cheapest":
                            self.hunt(line, self.steps_left() - self.tail(), repair=True)
                continue
            if self.roles.get("end") and not (self.end_refused and cart):
                # the store can refuse an ending too: then its note is read like any other, and the task solved again
                self.end_without_order()
                if self.finished:
                    return
                self.end_refused = True
                continue
            if cart and self.try_order(cart):
                return
        # no finish was accepted in those rounds: never leave without one
        if not self.finished:
            self.call(self.roles.get("buy", "") if self.cart else self.roles.get("end", ""), dict((self.roles.get("buy_args") if self.cart else self.roles.get("end_args")) or {}))

    def run(self) -> list[dict[str, Any]]:
        if not self.finished:
            self.read_task()
            self.scout()
            self.rank_properties()
            self.interview()
            self.resolve_pointers()
            self.hunt_all()
            self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "spent": {m: round(v, 5) for m, v in self.spent.items()}, "calls": self.calls, "seconds": round(time.monotonic() - self.t0, 1)})
        return self.trace


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Sieve(problem_data)
    except Exception as exc:
        return [{"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]}]
    try:
        return episode.run()
    except Exception as exc:
        episode.trace.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]})
        try:
            if not episode.finished:
                if episode.cart and episode.roles.get("buy"):
                    episode.call(episode.roles["buy"], dict(episode.roles.get("buy_args") or {}))
                elif episode.roles.get("end"):
                    episode.call(episode.roles["end"], dict(episode.roles.get("end_args") or {}))
        except Exception as again:
            episode.trace.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:400]})
        return episode.trace


__all__ = ["agent_main"]
