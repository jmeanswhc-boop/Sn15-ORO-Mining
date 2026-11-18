"""finorok: a shopping agent for ORO store worlds.

Layout, top to bottom: limits and the two proxy clients, the prompt texts, pure
helpers, the episode State, then the parts that act on it.  Every part holds the
same State object and a link to each of the other parts:

  Transport  model calls with their resend rules, and store steps
  Brief      the request read into roles, requested lines, needs and asked-for steps
  Catalog    listings seen so far: ids, fields, stock, and store notes in results
  Pages      result shapes, further pages, cheapest reads and extra opens
  Interview  questions to the shopper and what the answers settle
  Review     the gate before an ending, cart checks, pointer chains and swaps
  Finisher   which tools the loop sees, ending calls and their arguments
  Loop       the model's turns and the handling of each tool call it makes

agent_main(problem_data) builds one Crew per episode and returns its trace.
"""
from __future__ import annotations

import json
import random
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

FAST_MODEL = 'anthropic/claude-haiku-5.5'
STRONG_MODEL = 'anthropic/claude-sonnet-5.5'
BACKUP_MODEL = 'openai/gpt-6-luna'
TURN_CAP = 42
MODEL_CALL_CAP = 60
SPEND_WARN = 0.055
SPEND_CAP = 0.078
RETRY_WAITS = (4, 10, 20, 35)
RETRY_WAIT_BUDGET = 45
SHOWN_CHARS = 450
RESULT_CHAR_CAP = 15000
TRANSCRIPT_CHAR_CAP = 150 * 1000
GATE_CALL_CAP = 3
EXTRA_PAGE_CAP = 6
CHEAPER_READ_CAP = 3
_model_client = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_store_client = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
F_SLOW_GUARD = True
EPISODE_SECONDS = 300
TIME_RESERVE = 18
SHORTEST_CALL = 8
STORE_WAIT_CAP = 30.0
_T0 = time.monotonic()
F_MODEL_RETRY = True
MISS_PAUSE = 15.0
BRIEF_PAUSE = 8.0
BRIEF_RETRY_UNTIL = 90
LADDER_QUIET = 45.0
REFUSE_CUE = re.compile('\\bwithout (buying|purchasing|ordering|a purchase|an order)\\b|\\bdeclin|\\bno purchase\\b', re.I)
BUY_CUE = re.compile('\\b(plac|submit)\\w*\\b[^.;]{0,30}\\border\\b|\\bcheck ?out\\b', re.I)
SPENT_CUE = re.compile('budget|allowance|quota', re.I)
F_DECLINE_REVIEW = True
REVIEW_SPEND_CAP = SPEND_CAP + 0.05
F_AUDIT_ROOM = True
AUDIT_ROOM = 3000
F_TOOL_LIMITS = True
CAP_CUE = re.compile('\\b(?:up to|at most|no more than|(\\d+) or)\\s+(\\d+)\\s+([a-z]+s)\\b', re.I)
F_SHELF_GUARD = True
F_R12_LEAN = True
F_R12_LEAN_ALL = True
F_R12_FAST_RETRY = True
F_R12_NO_SPEND_END = False
F_R12_CHAIN_GATE = True
F_R12_MENU = True
F_R12_TOLD_SEARCH = True
F_R12_BOTH_SPELLINGS = False
F_R12_DROP_UPDATE = True
F_R12_UNSEEN_WORD = True
F_R12_LAST_LINE = True
F_R12_OPEN_TOLD = True
F_R12_ROUND_ASK = True
F_R12_READ_LINES = True
F_R12_WHOLE_PHRASE = True
F_R12_LEAN_CONFIRM = True
F_R12_TOLD_UNSEARCHED = True
F_R12_NO_TOLD_PAGE = True
F_R12_FAST_JUDGE = True
F_R12_ANY = F_R12_LEAN or F_R12_FAST_RETRY or F_R12_NO_SPEND_END or F_R12_CHAIN_GATE or F_R12_MENU or F_R12_TOLD_SEARCH or F_R12_DROP_UPDATE or F_R12_UNSEEN_WORD or F_R12_LAST_LINE or F_R12_OPEN_TOLD or F_R12_ROUND_ASK or F_R12_READ_LINES or F_R12_WHOLE_PHRASE
LEAN_LADDER = ({'enabled': False}, {'effort': 'low'})
RETRY_ROOM = 2500
FAST_JUDGE_ROOM = 4000
FAST_AUDIT_ROOM = 6000
TOLD_SEARCH_CAP = 4
TOLD_PAGE_CAP = 3
OPEN_TOLD_CAP = 3
GATE_VIEW_CAP = 4
_lean_rung = 0
LEAN_CONFIRM_N = 3
_lean_zero = 0

LOOP_PROMPT = """You are Rok, a careful buying assistant acting for a single shopper inside an online store, and calling the store's tools is the only way you can act. Every episode scores either 1 or 0, and only an exactly right finish scores 1. Either the order holds every item the shopper asked for, at the quantity asked for, with each line meeting all of the requirements that apply at checkout and the total within the budget; or, only once a thorough search shows that no listing in the store meets all of them, the episode closes through the no-purchase tool, done in the way the request says.

THE STORE AND ITS TOOLS
- Each tool call uses up one of a limited number of steps; reasoning is free. Every store has its own tool names, argument names and result layouts: read the description and schema of each tool, and send argument names and allowed values spelled exactly as written there.
- Searches may be restricted: a category filter, keyword matching that may only look at the first words of a query, small pages of results, a token or an offset for the next page. When your answer depends on seeing every match, compare the total the store reports with the rows you have and fetch further pages using the exact token or offset the store returned. Keyword queries do best when they are short and lead with the requested value, then the product noun. When a filtered search comes back empty, usually the filter value or the filter argument is wrong: try again with the label spelled as the result rows show it, or with a different filter argument.
- A search row is only a summary. Stores divide a listing's record up in different ways: one view may give its attributes (properties), another only its offer and description, and a side-by-side view may give either one. Once Rok has learned which tool shows attributes, the ledger names that tool. Settle every property requirement from those attributes, which means reading them for each listing you might pick; one call covering several listings saves steps when it shows what you need. Do not open a listing a second time when nothing has changed since you read it.
- Rok's interviewer puts all questions to the shopper for you, so never send the shopper a message yourself. The replies come back in the conversation and are written into the ledger.
- A request may also want things done along the way: an approval, a message, something to look at or confirm first. The ledger lists them. Those due right before the order or right before the ending are carried out by Rok itself at the moment you call the checkout or the ending tool, so do not wait for them, and never end without a purchase on their account; handle any others with the store's tools (questions about products go through Rok's interviewer).

WHEN A LISTING FITS
- What kind of product a listing is gets settled by its category label: only a label matching the requested item as a whole counts, never a nearby one.
- A property requirement holds when the listing's own attribute for that property gives the value. How the key is spelled does not matter (spaces, underscores, camel case, prefixes), and a single entry of a comma or slash list is enough. Values are matched by their words: case, singular or plural and ordinary synonyms are all fine, and a value that includes the words of the requested value counts ("pale amber" meets amber, "super soft weave" meets soft). A value worded differently does not count (an abbreviation or code is not the word behind it, and "stone grey" is not "slate grey"). Convert units before you compare numbers; a limit includes its boundary value, and a range has to sit entirely within a limit.
- An attribute whose key names the requirement's property (by that name or an ordinary synonym) is decisive: if it gives some other value, the listing fails, no matter what its other keys or its title say. A key for some other property, however close (a strap colour is not the case colour), says nothing about the requirement. After each opened listing Rok adds [evidence] lines that list the keys whose names match a requirement.
- When the listing has no attribute for that property, the requested value still counts if another attribute carries it (say a list of styles or features that includes it) or if the title or description states it. A number of a different kind does not count (a length is not a volume), and neither does a bare figure whose text does not say what it measures.
- When the category label itself carries the value of a requirement (a season or an audience in the label), the category meets that requirement.
- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Settle such a requirement first: track that listing down (search its category, brand, shop or words), open its full page with the single-listing or side-by-side tool (a search row will not do), and if that page refers you on to yet another listing for the property, find and open that one in the same way; the last page holds the value. The order only counts when every page in that chain has been opened. When the requested value is anything other than that listing's value, the rule excludes just the listings sharing that exact value, and by itself it is never a reason to end without a purchase.
- If a listing you are about to buy sends you to another page for more about it, read that page before you buy.
- When one requested item has to go with another (an accessory for a device), choose the pair together from the words on the listings (a list of what it fits, a model or series code printed on both pages). If no accessory that meets its own requirements fits the device you chose, move to another device that meets its requirements and that such an accessory fits; do not end without a purchase before that switch has been tried.
- Honour exclusions and closed lists. Wishes ("ideally", "would be nice") only ever break ties.
- Never buy anything the store shows as sold out. The total (price times quantity, added up over all lines) must stay within the budget; being exactly equal is fine.

PICKING ONE
- When the request asks for the cheapest (lowest total, lowest price), the order has to hold the lowest-priced listing among ALL of the fitting listings. Collect the whole fitting set (page through it: rows often come in no order of price, and Rok fetches the next pages of a short result on its own), check the candidates from the lowest price upwards, and take the first one that fits fully. When a search accepts a maximum price, use it: list the category under the price of your current best pick, so that only cheaper candidates are left to check.
- When the request asks for the most or the least of some measure, compare all of the fitting listings on that measure and take the extreme one; a listing that does not state the measure cannot win.
- In every other case any listing that meets everything is correct: do not spend steps looking for something better.
- Move each item into the cart as soon as you have opened a listing that meets all of its requirements (with a cheapest, most or least pick, once you have compared the fitting set), then move to the next item. Stores often sell out a line soon after it goes into the cart, and carting early leaves steps for a replacement.
- A request for several different items needs a separate cart line for each one, at its requested quantity.

WHEN THINGS CHANGE
- The shopper or the store may change things while you work: a new budget, a requirement added or taken away, a price change, a listing that sells out. These show up inside tool results, often at the moment you try to check out (a refusal); a note counts whichever tool's result it appears in, and the ledger shows the requirements now in force. A refusal does not end the episode: apply every change, repair the cart and check out once more.
- Shortly before you check out, read each chosen listing's record again, because price and stock can change without any notice.

ENDING THE EPISODE
- The cart has to hold exactly the chosen lines (take out anything else); look at the cart once, then check out in a call of its own.
- End without a purchase only once a real search has found no listing that meets everything. Being short of steps proves nothing of the sort: in that case check out the best cart you have. Word that ending the way the request asks for, and if a remark is needed, keep it brief and name no listing.
- Every reply must call at least one tool. Keep your text brief: what you know, then your next move."""


BRIEF_PROMPT = """You prepare a shopping task for an agent. Read the shopper's request and the store's tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:

"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request's words, "shelf": the store's category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request's own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property's value or limit outright; and never turn words that are part of what the product is called (a season, an audience or a use named with the product) into musts: those belong to "name". A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.
"budget": the most the shopper will pay in total as a number in the request's own unit, or null.
"budget_unit": that unit in words (for example "cents"), or null.
"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.
"in_stock_only": true unless the request clearly accepts items that are not available.
"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.
"pick_measure": what is maximised or minimised for "most"/"least", else "".
"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.
"steps": how many steps, moves or tool calls the request allows (integer), or null.
"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].
"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.
"wishes": a list of things the shopper would merely like.
"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).
"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper's approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.
"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.
"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing's attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing's current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).
"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.
"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.
"""


PLAN_PROMPT = """You pick the questions a shopping agent will send to a shopper. Every shopper holds a few firm preferences that the request leaves out, each one about a single property of a single requested product. A question uncovers a preference only if it names that exact property, and the answer covers only the first property a question names, so each question spends one property.

You receive the request, the items together with what the request already fixes for each of them, and a numbered list of the attribute keys found on opened listings in each item's category, with the number of those listings carrying each key and some example values. A preference the request leaves out may concern a property that none of the listed keys names.

Take the properties in this order:
0. Whatever the request itself tells you to check with the shopper (shown under ASK FIRST when there is any), each named by the property it concerns.
1. Hinted properties. The story in a request often points to an unstated preference without giving its value: a sentence about something the shopper cannot judge or hands over to you, about someone whose taste is decisive, about the product's appearance or the setting it is meant for, about the person who will use or wear it, or about what it holds. For each such sentence give the one property it points to and the item it concerns, in the order in which the story mentions them.
Call a hinted property by its plain name: a hint about who will wear it points to gender, a hint about its appearance to colour or finish (whichever one the request leaves open).
2. After those, the listed keys in ranked order: prefer the properties that buyers of that kind of product decide on first (often its size, capacity or dimensions, its type, its material or its colour, whichever ones that product is chosen by) and keys that most of the item's listings carry; stay away from keys found on just one or two of them.
Leave out a property the request already fixes for that item, another name for a property you have already chosen, a key that lists buying variants (a mix of sizes, colours and pack counts), identifiers, warranty and other logistics. With more than one item, start with the hinted properties of every item, then alternate between the items.
For "label" give the property's plain name: for a listed key, every word of the key in plain spelling ("strap width" for strapWidth, "frame depth" for frame_depth; never cut down to "width" or "depth"); the name of the property alone, with no example values, no brackets and no second property.
Pick at most COUNT of them, best first. Return ONE JSON object with nothing around it: {"ask": [{"item": item number, "label": "...", "hint": "up to 8 words of the story it rests on, or an empty string"}]}"""


HEAR_PROMPT = """A shopping agent has asked a shopper some questions. Go through every answer and return ONE JSON object with no other text:
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value has to be taken from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper no longer holds"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer objects that the message covered too many things or asks for one question at a time; leave out answers that merely say they did not understand], "budget": new total limit as a number or null}
- "needs": each firm preference an answer gives, with its exact value or limit and its unit ("no wider than 40 cm", "square", "not plastic"). Name the need after the property the answer itself names, even when that differs from the one that was asked about; several answers on one property give one need. Copy the value exactly as the answer words it, nouns and units included ("at least 4 drawers", not "at least 4").
- An answer may give the value by pointing to another listing (a value the same as, different from or close to the one on some shop's product, with that product picked out by a few of its properties): that makes ONE need on the property the answer names, with "points": true. Its value is the answer's own wording copied exactly, from the comparison words through to the end of that description, including whatever it says about a listing that page may name; never cut it short or rephrase it. The properties describing the other product are not needs of this item.
- "free": for each question, the property the shopper said they do not mind about.
- Homely or indirect wording that names a value still gives that value."""


NOTE_PROMPT = """During a shopping agent's work the store displayed the note below (its source may be the shopper or the store). Return ONE JSON object and nothing more:
{"kind": "change" when the note alters the task (the budget, the requirements, a price, availability, a refusal of the order), "error" when it only says a call was badly formed or was not accepted, "other" in every other case, "budget": the total limit now in force as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement that is no longer wanted"}], "gone": [listing ids it calls unavailable], "sold_out": true when it says a chosen listing is gone, sold out or not available any more (even if it names none), "price_moved": true when it says a price has changed, "summary": "one short sentence"}
Treat a replaced requirement as a drop of the old one together with an add of the new one. List every change the note makes."""


CHECK_PROMPT = """You are the final review before a shopping agent checks out. Using only the records shown, decide whether this cart is exactly right.
How to read the records: a requirement holds when the listing's attribute for that property gives the value (key spelling does not matter, a single entry of a list is enough, case, singular or plural and ordinary synonyms count, a value that includes the words of the requested value counts, as "pale amber" does for amber, units get converted, a limit includes its boundary, a range has to sit wholly inside); an attribute whose key names the requirement's own property is decisive, and if it gives something else the listing fails; a key for some other property, however close (a strap colour is not the case colour), decides nothing; when the listing has no attribute for that property, the requested value still counts if another attribute, the title or the description states it (a number of a different kind does not: a length is not a volume; nor does a bare figure whose text does not say what it measures); a requirement that the category label itself states is met by that category; the category label has to match the requested item as a whole; a rule of differing from another listing's value is met unless the records show that very value. Every requested item needs its own cart line at the requested quantity, the cart may hold nothing else, nothing in it may show as out of stock, and price times quantity added up over all lines has to stay within the budget. For each listing, the most recent record is the one that counts.
If the pick rule is cheapest, most or least, look at the other listings shown as well: when one of them plainly fits every requirement and beats the cart's choice under that rule, the cart is wrong.
Return ONE JSON object and nothing besides it: {"ok": true or false, "problems": ["a short, concrete problem", ...], "better": [listing ids that ought to replace a cart line, best first]}. Set ok=false only for a problem the records plainly show, or for a requirement that no record covers."""


READING_RULE = """an attribute decides a property when its key names that very property (any spelling of the key, a single entry of a list is enough, synonyms and converted units count, a value that includes the words of the requested value counts, as "pale amber" does for amber, a limit includes its boundary); a key for some other property, however close (a strap colour is not the case colour), decides nothing; when no attribute names that property, the requested value counts if another attribute, the title or the description states it (a bare figure that does not say what it measures counts for nothing); a requirement that the category label itself states is met by that category; the category label has to match the item; a rule of differing from another listing's value is met unless the records show that very value; listings that are out of stock or that break the budget do not fit."""

AUDIT_PROMPT = """A shopping agent is about to end without buying anything, since it believes that nothing fits. Test the listing records shown against the requirements, using these reading rules: """ + READING_RULE + """
A record marked ROW was only seen as a row of search results: its page (with the attributes) has never been read, so on a property the row does not show it can neither pass nor fail.
Return ONE JSON object with nothing around it: {"fits": {"item number": [listing ids that seem to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind and inside the budget whose page might show that they fit, most promising first, at most 6], "reason": "one sentence"}"""

UNMIX_PROMPT = """A shopper's request has been turned into firm requirements on a product. Some of those requirements borrow their value from another listing that the request refers to, and the words identifying that other listing (its kind, its shop, properties named after it with "with ...") sometimes get taken by mistake for requirements on the product being bought. Going by the request's wording, tell which of the OTHER REQUIREMENTS below do nothing but describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when each one is a genuine requirement on the product]}"""

TRAIL_PROMPT = """A shopping agent is about to check out. Some requirements borrow their value from another listing rather than stating it, and a listing's page may send the reader on to yet another listing for that property. The order only counts once the agent has opened in full every listing along each such chain. Trace the chain of each requirement starting from the requirement's own description, one link at a time, through the words of the pages themselves, using the records below (OPENED: the agent has opened that page; ROW: seen only as a row of search results, its page unread).
Return ONE JSON object with nothing around it: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when one of the records below is that link, else "", "about": how the requirement or the page before describes that listing}], "done": true when the last link is an OPENED page that itself states the property's value, else false, "value": the value of the property as that last page states it, or ""}]}
Give the links of each chain in order and stop at the first link whose page is not OPENED or cannot be found, because its words are not known yet."""

GATE_PROMPT = """A shopping agent is about to STAGE. Along the way, the shopper's request asks for the steps listed below. From the numbered RECORD of what the agent has done, decide for each step that is due at this point whether it has been done for the cart in its present state. A step is done only when some RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart's last change and its reply grants it; questions about a product's properties do not carry out a step.
While a step is still missing, give the one store call that does it next: a tool from TOOLS together with its arguments. A message to the shopper goes through the messaging tool: a single short sentence or question about that step only, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done, or when the messages or steps left are not enough for the missing one.
Return ONE JSON object with nothing around it: {"steps": [{"step": the step, "done": true or false, "lines": [numbers of the RECORD lines that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply wants changed before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for every firm requirement such a reply states]}"""

SWAP_PROMPT = """A listing in a shopping agent's cart has just sold out. Look among the other listing records shown for replacements for that cart line, using these reading rules: """ + READING_RULE + """
Return ONE JSON object with nothing around it: {"fits": [{"id": listing id, "price": its price as a number, as the record gives it} for each listing that seems to meet every requirement of the sold-out line's item, at most 6], "reason": "one sentence"}"""

SHAPE_PROMPT = """A store tool returned the result shown below. Name its parts using the exact keys that the result and the tool's parameters use. Return ONE JSON object with nothing around it: {"rows": the key of the list of listings, "count": the key that holds how many listings matched altogether (not how many are shown), or null, "next": the key that holds the value requesting the next page, or null, "next_arg": the tool parameter that accepts that value, or null, "price": the key inside a row for the listing's price, or null, "stock": the key inside a row that tells whether the listing can be bought now, or null}"""

if F_R12_WHOLE_PHRASE:
    LOOP_PROMPT = LOOP_PROMPT.replace(', and a value that includes the words of the requested value counts ("pale amber" meets amber, "super soft weave" meets soft)', ', and a value counts when the requested words appear in it as one unbroken run of whole words ("pale amber" meets amber, "super soft weave" meets soft weave); a value that puts another word inside the requested words does not count ("soft cotton weave" does not meet soft weave)')
    CHECK_PROMPT = CHECK_PROMPT.replace('a value that includes the words of the requested value counts, as "pale amber" does for amber, ', 'a value counts when the requested words appear in it as one unbroken run of whole words, as "pale amber" does for amber (but "soft cotton weave" does not for soft weave), ')
    READING_RULE = READING_RULE.replace('a value that includes the words of the requested value counts, as "pale amber" does for amber, ', 'a value counts when the requested words appear in it as one unbroken run of whole words, as "pale amber" does for amber (but "soft cotton weave" does not for soft weave), ')
    AUDIT_PROMPT = AUDIT_PROMPT.replace('a value that includes the words of the requested value counts, as "pale amber" does for amber, ', 'a value counts when the requested words appear in it as one unbroken run of whole words, as "pale amber" does for amber (but "soft cotton weave" does not for soft weave), ')
    SWAP_PROMPT = SWAP_PROMPT.replace('a value that includes the words of the requested value counts, as "pale amber" does for amber, ', 'a value counts when the requested words appear in it as one unbroken run of whole words, as "pale amber" does for amber (but "soft cotton weave" does not for soft weave), ')
if F_R12_TOLD_SEARCH:
    LOOP_PROMPT = LOOP_PROMPT.replace('- In every other case any listing that meets everything is correct: do not spend steps looking for something better.', '- In every other case any listing that meets everything is correct. A search that pairs a value the shopper told you with the other values known for the same item is never wasted: it is how a fitting listing is usually found.')
if F_R12_MENU:
    PLAN_PROMPT = PLAN_PROMPT.replace("prefer the properties that buyers of that kind of product decide on first (often its size, capacity or dimensions, its type, its material or its colour, whichever ones that product is chosen by) and keys that most of the item's listings carry; stay away from keys found on just one or two of them.", "take them in the order they are listed for the item's category (ranked by how many of its opened listings carry each key), and stay away from keys found on just one or two of them. A key whose values name a part built into the product (the motor of a fan, the lens of a camera) is a property of that product like any other.")
if F_R12_ROUND_ASK:
    HEAR_PROMPT = HEAR_PROMPT.replace('"budget": new total limit as a number or null}', '"budget": new total limit as a number or null, "unclear": [numbers of the questions whose answer merely says the shopper did not understand which product or property was meant]}')


def inner_of(observation: Any) -> Any:
    """The result inside a store notice: a two-key object whose other value is a non-empty list of text lines."""
    if isinstance(observation, dict) and len(observation) == 2:
        (a, b) = list(observation)
        for (lines, inner) in ((a, b), (b, a)):
            value = observation[lines]
            if isinstance(value, list) and value and all((isinstance(s, str) for s in value)):
                return observation[inner]
    return observation


def bare_stems(text: Any) -> set[str]:
    """Lower-case runs of letters and digits, with a plural s cut as in word_stems."""
    return {w[:-1] if len(w) > 3 and w.endswith('s') else w for w in re.findall('[a-z0-9]+', str(text).lower())}


def told_numbers(text: Any) -> list[tuple[str, str]]:
    """Each number followed by a word in the text: a value together with its unit."""
    return re.findall('(\\d+(?:\\.\\d+)?)\\s*([a-z]+)', str(text).lower())


def word_run(small: Any, big: Any) -> bool:
    """Whether the letter runs and numbers of small appear as one unbroken run among those of big."""
    a = re.findall('[a-z]+|[0-9]+(?:\\.[0-9]+)?', str(small).lower())
    b = re.findall('[a-z]+|[0-9]+(?:\\.[0-9]+)?', str(big).lower())
    return bool(a) and ' ' + ' '.join(a) + ' ' in ' ' + ' '.join(b) + ' '


def to_json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        return str(value)


def clip_value(value: Any, limit: int, depth: int=0) -> Any:
    if depth > 7:
        return '...'
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + '...'
    if isinstance(value, dict):
        return {k: clip_value(v, limit, depth + 1) for (k, v) in value.items()}
    if isinstance(value, list):
        return [clip_value(v, limit, depth + 1) for v in value]
    return value


def parse_json_object(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    text = text.strip()
    fence = re.match('^```[A-Za-z]*\\s*(.*?)\\s*```$', text, re.S)
    if fence:
        text = fence.group(1)
    try:
        got = json.loads(text)
        return got if isinstance(got, dict) else None
    except ValueError:
        pass
    (left, right) = (text.find('{'), text.rfind('}'))
    if 0 <= left < right:
        try:
            got = json.loads(text[left:right + 1])
            return got if isinstance(got, dict) else None
        except ValueError:
            return None
    return None


def flat_text(text: Any) -> str:
    return ' '.join(split_words(text).split()).strip(' .;,')


def near_words(a: Any, b: Any) -> bool:
    """Some word of a and some word of b agree on all but their last letters (bench and benches)."""
    for x in split_words(a).split():
        for y in split_words(b).split():
            n = min(len(x), len(y))
            k = 0
            while k < n and x[k] == y[k]:
                k += 1
            if x == y or (k >= 4 and k >= n - 1):
                return True
    return False


def split_words(key: str) -> str:
    key = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(key))
    return re.sub('[\\s_.\\-]+', ' ', key).strip().lower()


def word_stems(text: str) -> frozenset[str]:
    return frozenset((w[:-1] if len(w) > 3 and w.endswith('s') else w for w in split_words(text).split()))


def names_match(prop: frozenset[str], key: frozenset[str]) -> bool:
    if not prop or not key:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)


def schema_types(schema: dict[str, Any]) -> list[str]:
    kind = schema.get('type')
    if isinstance(kind, str):
        return [kind]
    return [k for k in kind or [] if isinstance(k, str)]


def value_fits_type(value: Any, kind: str) -> bool:
    if kind == 'string':
        return isinstance(value, str)
    if kind == 'integer':
        return isinstance(value, int) and (not isinstance(value, bool))
    if kind == 'number':
        return isinstance(value, (int, float)) and (not isinstance(value, bool))
    if kind == 'boolean':
        return isinstance(value, bool)
    if kind == 'array':
        return isinstance(value, list)
    if kind == 'object':
        return isinstance(value, dict)
    if kind == 'null':
        return value is None
    return True


def convert_kind(value: Any, kinds: list[str]) -> Any:
    """The first conversion that one of the schema's types allows: a whole float to int, a number text to a
    number, a number to text, 'true'/'false' to a bool, a single value to a one-entry list; else the value."""
    for kind in kinds:
        if kind == 'integer' and isinstance(value, float) and value.is_integer():
            return int(value)
        if kind in ('integer', 'number') and isinstance(value, str) and re.fullmatch('\\s*-?\\d+(\\.\\d+)?\\s*', value):
            number = float(value)
            return int(number) if kind == 'integer' or number.is_integer() else number
        if kind == 'string' and isinstance(value, (int, float)) and (not isinstance(value, bool)):
            return str(value)
        if kind == 'boolean' and isinstance(value, str) and (value.strip().lower() in ('true', 'false')):
            return value.strip().lower() == 'true'
        if kind == 'array' and (not isinstance(value, list)) and (value is not None):
            return [value]
    return value


def match_enum(value: Any, schema: dict[str, Any], where: str, faults: list[str]) -> Any:
    """A value outside the schema's enum takes the option it equals ignoring case; otherwise a fault is noted."""
    options = schema.get('enum')
    if isinstance(options, list) and value not in options:
        if isinstance(value, str):
            low = value.strip().lower()
            match = [o for o in options if isinstance(o, str) and o.lower() == low]
            if match:
                value = match[0]
        if value not in options:
            faults.append(f'{where}: allowed values are {options}')
    return value


def check_bounds(value: Any, schema: dict[str, Any], where: str, faults: list[str]) -> Any:
    """Text longer than maxLength is cut; text below minLength and numbers outside minimum/maximum are faults."""
    if isinstance(value, str) and isinstance(schema.get('maxLength'), int) and (len(value) > schema['maxLength']):
        value = value[:schema['maxLength']]
    if isinstance(value, str) and isinstance(schema.get('minLength'), int) and (len(value) < schema['minLength']):
        faults.append(f'{where} is too short')
    if isinstance(value, (int, float)) and (not isinstance(value, bool)):
        if isinstance(schema.get('minimum'), (int, float)) and value < schema['minimum']:
            faults.append(f"{where} is below {schema['minimum']}")
        if isinstance(schema.get('maximum'), (int, float)) and value > schema['maximum']:
            faults.append(f"{where} is above {schema['maximum']}")
    return value


def coerce_list(value: list[Any], schema: dict[str, Any], where: str, faults: list[str]) -> list[Any]:
    """Entry-count faults, then each entry shaped by the schema for entries."""
    if isinstance(schema.get('maxItems'), int) and len(value) > schema['maxItems']:
        faults.append(f"{where} allows at most {schema['maxItems']} entries")
    if isinstance(schema.get('minItems'), int) and len(value) < schema['minItems']:
        faults.append(f"{where} needs at least {schema['minItems']} entries")
    if isinstance(schema.get('items'), dict):
        value = [coerce_to_schema(v, schema['items'], f'{where}[{i}]', faults) for (i, v) in enumerate(value)]
    return value


def coerce_object(value: dict[str, Any], schema: dict[str, Any], where: str, faults: list[str]) -> dict[str, Any]:
    """Unknown keys (when the schema closes them) and missing required keys are faults; known keys are shaped."""
    props = schema.get('properties') if isinstance(schema.get('properties'), dict) else {}
    if schema.get('additionalProperties') is False:
        for name in [n for n in value if n not in props]:
            faults.append(f'{where}.{name} is not an allowed argument')
    for name in schema.get('required') or []:
        if name not in value:
            faults.append(f'{where}.{name} is required')
    return {k: coerce_to_schema(v, props.get(k), f'{where}.{k}', faults) if k in props else v for (k, v) in value.items()}


def coerce_to_schema(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    """Bend a value toward its JSON schema. What still breaks the schema is added to `faults`, in this sequence:
    the type (which stops the walk), the enum, length and bounds, entry counts, then keys, depth first."""
    if not isinstance(schema, dict):
        return value
    kinds = schema_types(schema)
    if kinds and (not any((value_fits_type(value, k) for k in kinds))):
        value = convert_kind(value, kinds)
        if not any((value_fits_type(value, k) for k in kinds)):
            faults.append(f"{where} should be {'/'.join(kinds)}")
            return value
    value = match_enum(value, schema, where, faults)
    value = check_bounds(value, schema, where, faults)
    if isinstance(value, list):
        value = coerce_list(value, schema, where, faults)
    if isinstance(value, dict):
        value = coerce_object(value, schema, where, faults)
    return value


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, (str, dict)) and value:
        return [value]
    return []


def whole_number(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.0+)?\\s*', value):
        return int(float(value))
    return None


def looks_like_id(value: Any, named: bool=False) -> bool:
    if not isinstance(value, str) or not 3 <= len(value.strip()) <= 64 or ' ' in value.strip():
        return False
    return named or bool(re.search('\\d', value))


def guess_id_key(rows: list[dict[str, Any]]) -> str | None:
    best = None
    for key in rows[0]:
        values = [row.get(key) for row in rows]
        if not all((looks_like_id(v, 'id' in str(key).lower()) for v in values)):
            continue
        values = [v.strip() for v in values]
        if len(rows) > 1 and len(set(values)) < len(values):
            continue
        score = ('id' in str(key).lower(), sum((len(v) for v in values)) / len(values))
        if best is None or score > best[0]:
            best = (score, key)
    return best[1] if best else None


def listed_in_rows(node: Any, ident: str, depth: int=0) -> bool:
    if depth > 5:
        return False
    if isinstance(node, list):
        return any((isinstance(v, dict) and ident in [str(x).strip() for x in v.values() if isinstance(x, str)] for v in node)) or any((listed_in_rows(v, ident, depth + 1) for v in node))
    if isinstance(node, dict):
        return any((listed_in_rows(v, ident, depth + 1) for v in node.values()))
    return False


def attribute_block(node: Any) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    best = None
    for value in node.values():
        if isinstance(value, dict) and len(value) >= 2:
            scalars = {k: v for (k, v) in value.items() if isinstance(v, (str, int, float)) and (not isinstance(v, bool))}
            if len(scalars) >= 2 and (best is None or len(scalars) > len(best)):
                best = scalars
    return best


def allowance_spent(observation: Any) -> bool:
    """A store reply that turns a call away because an allowance is used up: an error naming a budget, an
    allowance or a quota (the store answers so for every further call of that kind)."""
    error = observation.get('error') if isinstance(observation, dict) else None
    return isinstance(error, str) and SPENT_CUE.search(error) is not None


def tool_ending_roles(tools: list[dict[str, Any]]) -> dict[str, Any]:
    """The ending roles read from the tools alone, for an episode whose brief never came back.  Two ending tools: a
    role is set only when exactly one tool's description carries its cue and not the other role's cue, and only when
    its required arguments need nothing but a fixed text (a checkout tool with no required argument, a no-purchase
    tool whose required arguments are free strings).  One ending tool that does both by a required choice: both roles
    name it when the description ties exactly one choice to the order cue and exactly one to the no-purchase cue, and
    those choices become buy_args and refuse_args.  Anything else stays unset."""
    found: dict[str, list[str]] = {'buy': [], 'refuse': []}
    both: list[tuple[str, tuple[str, str, str]]] = []
    for tool in tools:
        fn = tool.get('function') if isinstance(tool, dict) else None
        if not isinstance(fn, dict) or not isinstance(fn.get('name'), str):
            continue
        text = str(fn.get('description') or '')
        (buys, refuses) = (BUY_CUE.search(text) is not None, REFUSE_CUE.search(text) is not None)
        schema = fn.get('parameters') if isinstance(fn.get('parameters'), dict) else {}
        props = schema.get('properties') if isinstance(schema.get('properties'), dict) else {}
        required = schema.get('required') or []
        if not isinstance(required, list) or not all((isinstance(k, str) for k in required)):
            continue
        need = [props.get(k) for k in required]
        if buys and (not refuses) and (not need):
            found['buy'].append(fn['name'])
        if refuses and (not buys) and all((isinstance(spec, dict) and spec.get('type') == 'string' and ('enum' not in spec) for spec in need)):
            found['refuse'].append(fn['name'])
        if buys and refuses:
            picked = mode_choices(text, schema)
            if picked:
                both.append((fn['name'], picked))
    out: dict[str, Any] = {k: v[0] for (k, v) in found.items() if len(v) == 1}
    if not out and len(both) == 1:
        (name, (key, buy_value, refuse_value)) = both[0]
        out = {'buy': name, 'refuse': name, 'buy_args': {key: buy_value}, 'refuse_args': {key: refuse_value}}
    return out


def stated_cap(tool: dict[str, Any] | None, param: str | None) -> int | None:
    """A per-call cap on an array argument that the tool states in words, as a number right before a plural noun
    (shaped like 'up to N things', 'M or N things', 'at most N things', 'no more than N things'): first in that
    argument's own description, then in the tool's description when that argument is the tool's only array argument."""
    fn = tool.get('function') if isinstance(tool, dict) and isinstance(tool.get('function'), dict) else {}
    schema = fn.get('parameters') if isinstance(fn.get('parameters'), dict) else {}
    props = schema.get('properties') if isinstance(schema.get('properties'), dict) else {}
    arrays = [k for (k, v) in props.items() if isinstance(v, dict) and 'array' in schema_types(v)]
    if not param or param not in arrays:
        return None
    texts = [str(props[param].get('description') or '')] + ([str(fn.get('description') or '')] if arrays == [param] else [])
    for text in texts:
        for found in CAP_CUE.finditer(text):
            (low, cap) = (int(found.group(1)) if found.group(1) else 0, int(found.group(2)))
            if 1 <= cap and low < cap:
                return cap
    return None


def mode_choices(text: str, schema: dict[str, Any]) -> tuple[str, str, str] | None:
    """For one tool that both orders and declines: its one required choice argument, the choice whose clause of the
    description carries the order cue alone and the choice whose clause carries the no-purchase cue alone."""
    props = schema.get('properties') if isinstance(schema.get('properties'), dict) else {}
    required = schema.get('required') if isinstance(schema.get('required'), list) else []
    need = [k for k in required if isinstance(k, str)]
    modes = [k for k in need if isinstance(props.get(k), dict) and isinstance(props[k].get('enum'), list)]
    rest = [props.get(k) for k in need if k not in modes]
    if len(modes) != 1 or not all((isinstance(spec, dict) and spec.get('type') == 'string' and ('enum' not in spec) for spec in rest)):
        return None
    clauses = re.split('[.;]', text)
    picks: dict[str, list[str]] = {'buy': [], 'refuse': []}
    for value in props[modes[0]]['enum']:
        if not isinstance(value, str) or not value.strip():
            continue
        said = [c for c in clauses if re.search('(?<![\\w-])' + re.escape(value) + '(?![\\w-])', c)]
        (buys, refuses) = (any((BUY_CUE.search(c) for c in said)), any((REFUSE_CUE.search(c) for c in said)))
        if buys and (not refuses):
            picks['buy'].append(value)
        if refuses and (not buys):
            picks['refuse'].append(value)
    if len(picks['buy']) == 1 and len(picks['refuse']) == 1:
        return (modes[0], picks['buy'][0], picks['refuse'][0])
    return None


def mode_question(tool: str, schema: Any) -> str:
    """The question for a store whose one ending tool checks out or declines by a mode argument."""
    return f"Tool {tool} is the store's single ending tool: its arguments decide whether it checks out or ends with no purchase. Its parameters: " + to_json_text(schema) + '\nReturn ONE JSON object only: {"buy_args": the fixed arguments for checking out, "refuse_args": the fixed arguments for ending with no purchase} (just the arguments that select the mode).'


class Want:
    """One line of the request: what it is, its category, its count, and its rules."""

    def __init__(self, label: str, family: str, units: int, rules: list[str]):
        self.label = label
        self.family = family
        self.units = units
        self.rules = rules
        self.answered: list[str] = []
        self.retired: list[str] = []
        self.loose: list[str] = []


class State:
    """Everything one episode knows, shared by all parts: limits, roles, the ledger of wants, listings seen,
    the cart as tracked, holds already used, and the transcript."""

    def __init__(self, problem: dict[str, Any]):
        environment = problem['environment']
        self.bind = dict(environment['binding'])
        self.policy = environment['policy_view']
        self.ask_text = str(self.policy.get('query') or '')
        self.tool_specs = [t for t in self.policy.get('tools') or [] if isinstance(t, dict) and isinstance(t.get('function'), dict) and isinstance(t['function'].get('name'), str)]
        self.arg_schemas = {t['function']['name']: t['function'].get('parameters') if isinstance(t['function'].get('parameters'), dict) else {'type': 'object', 'properties': {}} for t in self.tool_specs}
        steps = whole_number(self.policy.get('max_steps'))
        if steps is None and isinstance(self.policy.get('max_steps'), float):
            steps = int(self.policy['max_steps'])
        self.step_cap = steps if steps and steps > 0 else 30
        self.at_step = 0
        self.over = bool(self.policy.get('finished'))
        self.env_broke = False
        self.model_broke = False
        self.wait_spent = 0
        self.model_misses = 0
        if F_SLOW_GUARD:
            now = time.monotonic()
            self.t0 = _T0 if 0 <= now - _T0 < 30 else now
            self.call_secs: list[float] = []
        if F_MODEL_RETRY and F_SLOW_GUARD:
            self.hard_misses = 0
            self.fail_passing = True
            self.spent_tools: set[str] = set()
            self.replies = 0
            self.answered_at = self.t0
        self.spend = 0.0
        self.model_calls = 0
        self.turns = 0
        self.log: list[dict[str, Any]] = []
        self.sheet: dict[str, Any] = {}
        self.jobs: dict[str, str] = {}
        self.wants: list[Want] = []
        self.budget_cap: Any = None
        self.msg_cap = 0
        self.msg_sent = 0
        self.duties: list[dict[str, Any]] = []
        self.msg_reserve = 0
        self.gate_sent = 0
        self.gate_passed: set[tuple[str, str]] = set()
        self.duty_log: list[str] = []
        self.duty_steps: list[int] = []
        self.interviewed = False
        self.exchanges: list[tuple[str, str]] = []
        self.row_index: dict[str, dict[str, Any]] = {}
        self.full_pages: dict[str, dict[str, Any]] = {}
        self.first_seen: list[str] = []
        self.heard_keys: set[str] = set()
        self.store_notes: list[str] = []
        self.epoch = 0
        self.cart_changed = False
        self.cart_changed_at = -1
        self.cart_viewed_at = -1
        self.cart_seen: Any = None
        self.cart_counts: dict[str, int] = {}
        self.checked_ids: set[str] = set()
        self.reviewed = False
        self.offer_read_at: dict[str, int] = {}
        self.sold_at: dict[str, int] = {}
        self.off_sale: set[str] = set()
        self.sold_phrases: set[str] = set()
        self.note_said_sold = False
        self.sold_holds = 0
        self.id_params: set[tuple[str, str]] = set()
        self.cart_reply = ''
        self.buy_refused_at = -1
        self.hints: list[str] = []
        self.decline_holds = 0
        self.search_pushed = False
        self.best_cart: list[str] = []
        self.taken_off: set[str] = set()
        self.late_hold_done = False
        self.hint_items: dict[str, list[str]] = {}
        self.check_verdicts: dict[str, bool] = {}
        self.confirmed_hint: str | None = None
        self.auto_carted = False
        self.row_id_key: str | None = None
        self.shows_facts: dict[str, bool] = {}
        self.pending_notes: list[str] = []
        self.buy_tries = 0
        self.refusal_count = 0
        self.last_marks: list[str] = []
        self.transcript: list[dict[str, Any]] = []
        self.opened_ids: set[str] = set()
        self.pointers: list[tuple[int, str]] = []
        self.pointer_values: dict[str, str] = {}
        self.chain_runs = 0
        self.chain_holds = 0
        self.side_ok = 0
        self.side_bad = 0
        self.tool_fields: dict[str, set[str]] = {}
        self.view_fields: dict[str, tuple[int, set[str]]] = {}
        self.reads: dict[str, tuple[int, int, str]] = {}
        self.extra_pages = 0
        self.result_shapes: dict[str, dict[str, Any] | None] = {}
        self.price_field: str | None = None
        self.stock_field: str | None = None
        self.cheaper_done: set[str] = set()
        self.wider_done = False
        self.page_results: list[Any] = []
        if F_R12_ANY:
            self.lean_on = False
            self.last_finish: Any = None
            self.row_keys: set[str] = set()
            self.remark_tools: set[str] = set()
            self.gate_views = 0
            self.searches: list[tuple[str, list[str], bool]] = []
            self.page_marks: set[Any] = set()
            self.page_more: dict[str, int] = {}
            self.told_pages = 0
            self.told_runs = 0
            self.drop_updates = 0
            self.drop_searches = 0
            self.retired_before: tuple[int, ...] = ()
            self.turn_calls: list[dict[str, Any]] = []
            self.told_opened = False
            self.fact_read_ids: set[str] = set()


class Part:
    """Base of every part: the shared State and a link to each other part (set by Crew)."""

    def __init__(self, st: State) -> None:
        self.st = st

    def join(self, crew: Crew) -> None:
        self.net = crew.net
        self.brief = crew.brief
        self.catalog = crew.catalog
        self.pages = crew.pages
        self.talk = crew.talk
        self.review = crew.review
        self.fin = crew.fin
        self.loop = crew.loop


class Transport(Part):
    """Model calls (with waits and resends inside a fixed budget) and store steps (kept in step with the store)."""

    def steps_left(self) -> int:
        return self.st.step_cap - self.st.at_step

    def time_left(self) -> float:
        """Seconds before the episode limit, counted from the episode's start (the agent's import, or the call
        when the import happened long before it)."""
        return EPISODE_SECONDS - (time.monotonic() - self.st.t0)

    def call_guess(self) -> float:
        """A cautious length for the next model call: the longest of the last three, never under 4 s."""
        return max([4.0] + self.st.call_secs[-3:])

    def time_fits(self, calls: int) -> bool:
        """Whether `calls` more model calls of the guessed length still leave the reserve for finishing."""
        return self.time_left() >= calls * self.call_guess() + TIME_RESERVE

    def timed_model_post(self, body: dict[str, Any]) -> Any:
        """One model request whose read timeout ends before the reserve starts; its length feeds call_guess."""
        keep = _model_client.timeout
        _model_client.timeout = max(1.0, min(float(keep), self.time_left() - TIME_RESERVE))
        began = time.monotonic()
        try:
            return _model_client.post_verbose('/inference/chat/completions', body)
        finally:
            _model_client.timeout = keep
            self.st.call_secs.append(time.monotonic() - began)

    def time_pause(self, secs: float) -> None:
        """A jittered pause that never takes the room for one more shortest model call and the reserve."""
        pause = min(secs * (0.8 + 0.4 * random.random()), self.time_left() - TIME_RESERVE - SHORTEST_CALL)
        if pause > 0:
            time.sleep(pause)

    def keep_waiting(self, model: str, models: list[str]) -> bool:
        """Past the episode's shared wait allowance, whether a failing model is still worth waiting for: yes when
        no model has answered for LADDER_QUIET seconds (the outage looks general), or when it is the last of
        several models in this call, every one before it having failed just now; otherwise the call moves on to
        the next model at once, so that one model's outage costs no more than the allowance."""
        if time.monotonic() - self.st.answered_at >= LADDER_QUIET:
            return True
        return len(models) > 1 and model == models[-1]

    def timed_store_post(self, payload: dict[str, Any]) -> Any:
        """One store request that ends before the episode limit: the client's own three tries while they fit with
        their waits (up to 15 s after a rate limit), each try waiting at most 30 s for its reply; in the last 35 s a
        single try of at most 8 s, so that a lost reply still leaves room to send again; none in the last 4 s (None)."""
        left = self.time_left()
        if left < 4:
            return None
        keep = (_store_client.timeout, _store_client.max_retries)
        if left >= 35:
            _store_client.timeout = min(float(keep[0]), STORE_WAIT_CAP, (left - 17) / 3)
        else:
            _store_client.timeout, _store_client.max_retries = min(8.0, left - 3), 1
        try:
            return _store_client.post_verbose('/environment/call', payload)
        finally:
            _store_client.timeout, _store_client.max_retries = keep

    def consult(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None=None) -> dict[str, Any] | None:
        for model in models:
            if F_SLOW_GUARD and self.time_left() - TIME_RESERVE < SHORTEST_CALL:
                return None
            if self.st.model_calls >= MODEL_CALL_CAP:
                return None
            self.st.model_calls += 1
            tries = 0
            while True:
                (message, status) = self.complete_once(model, messages, max_tokens, tools)
                if message is not None:
                    if F_MODEL_RETRY and F_SLOW_GUARD:
                        self.st.replies += 1
                        self.st.answered_at = time.monotonic()
                    return message
                passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                if F_MODEL_RETRY and F_SLOW_GUARD:
                    if not passing or tries >= (1 if status in (None, 408) else len(RETRY_WAITS)):
                        break
                    if self.st.wait_spent + RETRY_WAITS[tries] > RETRY_WAIT_BUDGET and (not self.keep_waiting(model, models)):
                        break
                elif not passing or tries >= (1 if status in (None, 408) else len(RETRY_WAITS)) or self.st.wait_spent + RETRY_WAITS[tries] > RETRY_WAIT_BUDGET:
                    break
                pause = RETRY_WAITS[tries] * (0.8 + 0.4 * random.random())
                if F_SLOW_GUARD and self.time_left() - pause - TIME_RESERVE < SHORTEST_CALL:
                    break
                self.st.wait_spent += RETRY_WAITS[tries]
                tries += 1
                time.sleep(pause)
            if F_MODEL_RETRY and F_SLOW_GUARD:
                self.st.fail_passing = passing
            if status == 402:
                self.st.model_broke = True
                return None
        return None

    def complete_once(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        sent = messages
        if model.startswith('anthropic/') and len(messages) > 1:
            sent = [dict(m) for m in messages]
            marks = [0] + ([len(sent) - 1] if len(sent) > 1 else [])
            for index in marks:
                content = sent[index].get('content')
                if isinstance(content, str) and content:
                    sent[index]['content'] = [{'type': 'text', 'text': content, 'cache_control': {'type': 'ephemeral'}}]
        body: dict[str, Any] = {'model': model, 'messages': sent, 'max_tokens': max_tokens, 'temperature': 0}
        if tools:
            body['tools'] = tools
            body['tool_choice'] = 'auto'
        if F_SLOW_GUARD:
            result = self.lean_post(body) if F_R12_LEAN else self.timed_model_post(body)
        else:
            result = _model_client.post_verbose('/inference/chat/completions', body)
        data = result.data
        if isinstance(data, dict) and data.get('choices'):
            usage = data.get('usage') or {}
            if isinstance(usage.get('cost'), (int, float)):
                self.st.spend += float(usage['cost'])
            if F_R12_FAST_RETRY:
                self.st.last_finish = data['choices'][0].get('finish_reason')
            message = data['choices'][0].get('message') or {}
            if message.get('content') or message.get('tool_calls'):
                return (message, 200)
            self.st.log.append({'kind': 'llm_fail', 'model': model, 'status': 0})
            return (None, 0)
        error = result.error or {}
        status = error.get('status')
        if isinstance(data, dict) and isinstance(data.get('error'), dict):
            status = whole_number(data['error'].get('code'))
            error = {'body': data['error'].get('message')}
        self.st.log.append({'kind': 'llm_fail', 'model': model, 'status': status, 'body': str(error.get('body') or '')[:200]})
        return (None, status)

    def ask_structured(self, prompt: str, models: list[str], max_tokens: int=1800) -> dict[str, Any] | None:
        if F_R12_FAST_RETRY:
            return self.ask_retrying(prompt, models, max_tokens)
        for model in models:
            message = self.consult([{'role': 'user', 'content': prompt}], [model], max_tokens)
            got = parse_json_object((message or {}).get('content'))
            if got is not None:
                return got
            if self.st.env_broke or self.st.model_broke or self.st.model_calls >= MODEL_CALL_CAP:
                return None
        return None

    def lean_post(self, body: dict[str, Any]) -> Any:
        """F_R12_LEAN: timed_model_post with the reasoning cut on a STRONG call without tools while lean_on is set
        (on every such call with F_R12_LEAN_ALL): the first rung of LEAN_LADDER that the proxy accepts. A body it
        turns away (400 or 422) goes again one rung down, and below the last rung without a reasoning field; the
        rung reached holds for the rest of the process."""
        global _lean_rung
        while (self.st.lean_on or F_R12_LEAN_ALL) and body.get('model') == STRONG_MODEL and (not body.get('tools')) and (_lean_rung < len(LEAN_LADDER)):
            result = self.timed_model_post({**body, 'reasoning': LEAN_LADDER[_lean_rung]})
            data = result.data
            status = (result.error or {}).get('status')
            if isinstance(data, dict) and isinstance(data.get('error'), dict):
                status = whole_number(data['error'].get('code'))
            if status not in (400, 422):
                if F_R12_LEAN_CONFIRM:
                    self.lean_check(result)
                return result
            _lean_rung += 1
            self.st.log.append({'kind': 'lean_refused', 'rung': _lean_rung})
        return self.timed_model_post(body)

    def lean_check(self, result: Any) -> None:
        """F_R12_LEAN_CONFIRM: reads the reasoning-token count (usage.completion_tokens_details.reasoning_tokens) of
        a 2xx reply to the first rung of LEAN_LADDER, reasoning off. A count above 0 means the field was taken and
        the model reasoned anyway, so later calls go one rung down, and the cut can no longer be confirmed. Each
        count of exactly 0 is one confirmation; the cut counts as confirmed after LEAN_CONFIRM_N of them, because
        without the field sonnet already reported 0 on 168 of 614 local calls without tools (HEAR on 119 of 173),
        so a single 0 proves little. A reply without the count moves nothing and confirms nothing. Replies to the
        effort-low rung are not read: they reason by design, so their count says nothing about the field."""
        global _lean_rung, _lean_zero
        data = result.data
        if _lean_rung != 0 or not isinstance(data, dict) or not data.get('choices'):
            return
        usage = data.get('usage')
        details = usage.get('completion_tokens_details') if isinstance(usage, dict) else None
        spent = details.get('reasoning_tokens') if isinstance(details, dict) else None
        if isinstance(spent, bool) or not isinstance(spent, (int, float)):
            return
        if spent > 0:
            _lean_rung += 1
            self.st.log.append({'kind': 'lean_ignored', 'rung': _lean_rung, 'reasoning': spent})
        else:
            _lean_zero += 1

    def ask_lean(self, prompt: str, models: list[str], max_tokens: int=1800) -> dict[str, Any] | None:
        """F_R12_LEAN: ask_structured with the reasoning cut on its STRONG calls (lean_post)."""
        self.st.lean_on = True
        try:
            return self.ask_structured(prompt, models, max_tokens)
        finally:
            self.st.lean_on = False

    def ask_retrying(self, prompt: str, models: list[str], max_tokens: int) -> dict[str, Any] | None:
        """F_R12_FAST_RETRY: the models in turn as in ask_structured, plus one rule: a FAST reply that does not
        parse, or that stopped at its token limit, is followed by a STRONG call with room for at least RETRY_ROOM
        tokens, unless STRONG already had that room in this call. The parsed part of a cut reply is kept for when
        nothing better comes."""
        order = [(model, max_tokens) for model in models]
        kept = None
        place = 0
        while place < len(order):
            (model, room) = order[place]
            place += 1
            self.st.last_finish = None
            message = self.consult([{'role': 'user', 'content': prompt}], [model], room)
            got = parse_json_object((message or {}).get('content'))
            cut = self.st.last_finish == 'length'
            if got is not None and (not (cut and model == FAST_MODEL)):
                return got
            if got is not None:
                kept = got
            if self.st.env_broke or self.st.model_broke or self.st.model_calls >= MODEL_CALL_CAP:
                return kept
            if model == FAST_MODEL and (cut or (message is not None and got is None)):
                if any((m == STRONG_MODEL and r >= RETRY_ROOM for (m, r) in order[:place])):
                    continue
                later = [i for i in range(place, len(order)) if order[i][0] == STRONG_MODEL]
                if later:
                    order[later[0]] = (STRONG_MODEL, max(order[later[0]][1], RETRY_ROOM))
                else:
                    order.insert(place, (STRONG_MODEL, max(room, RETRY_ROOM)))
                self.st.log.append({'kind': 'fast_retry', 'cut': cut})
        return kept

    def take_step(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(self.st.bind)
        payload['step'] = self.st.at_step + 1
        payload['action'] = {'name': name, 'args': args}
        for _ in range(4):
            if F_SLOW_GUARD:
                result = self.timed_store_post(payload)
                if result is None:
                    return None
            else:
                result = _store_client.post_verbose('/environment/call', payload)
            if isinstance(result.data, dict):
                data = result.data
                self.st.at_step = data['step'] if isinstance(data.get('step'), int) else self.st.at_step + 1
                if data.get('finished'):
                    self.st.over = True
                seen = data.get('observation')
                if F_R12_ANY:
                    self.catalog.note_step(name, args, seen)
                if F_R12_CHAIN_GATE:
                    self.catalog.note_view(name, args, seen)
                elif name in self.catalog.page_views() and (not (isinstance(seen, dict) and seen.get('error'))):
                    for value in args.values():
                        self.st.opened_ids.update((v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)))
                return data
            error = result.error or {}
            status = error.get('status')
            body = str(error.get('body') or '')
            self.st.log.append({'kind': 'env_fail', 'status': status, 'body': body[:200]})
            if status == 409:
                low = body.lower()
                if 'finish' in low or 'quarantin' in low or 'finaliz' in low:
                    self.st.over = True
                    return None
                found = re.search('expected (\\d+)', body)
                if found:
                    self.st.at_step = int(found.group(1)) - 1
                    payload['step'] = self.st.at_step + 1
                    continue
                if 'reuse' in low or 'different action' in low:
                    self.st.at_step += 1
                    payload['step'] = self.st.at_step + 1
                    continue
                return None
            if status in (400, 401, 402, 403, 404, 422):
                if status == 402:
                    self.st.env_broke = True
                return None
        return None


class Brief(Part):
    """The request read once into roles, wants, pointers, limits and duties, and the texts that restate them."""

    def read_brief(self) -> None:
        """Turn the request into the brief, then keep what the episode needs from it, in a fixed sequence."""
        brief = self.ask_brief()
        if F_MODEL_RETRY and F_SLOW_GUARD:
            brief = self.brief_fallback(brief)
        self.st.sheet = brief
        self.take_roles(brief)
        self.take_wants(brief)
        self.take_pointers(brief)
        self.move_lookup_musts()
        self.take_limits(brief)
        self.take_duties(brief)

    def ending_named(self, brief: dict[str, Any]) -> bool:
        roles = brief.get('roles') if isinstance(brief, dict) and isinstance(brief.get('roles'), dict) else {}
        return any((isinstance(roles.get(k), str) and roles[k] in self.st.arg_schemas for k in ('buy', 'refuse')))

    def brief_fallback(self, brief: dict[str, Any]) -> dict[str, Any]:
        """A brief that names no ending tool: one more ask after a pause while the episode is young, but only when
        no model has answered at all yet (the asks failed in transport, never after a 402; a reply that came back in
        an unusable form is not asked for again), then the ending roles read from the tools themselves, so that the
        episode can still be finished."""
        if self.ending_named(brief):
            return brief
        if not self.st.model_broke and self.st.replies == 0 and EPISODE_SECONDS - self.net.time_left() <= BRIEF_RETRY_UNTIL:
            self.net.time_pause(BRIEF_PAUSE)
            again = self.ask_brief()
            if self.ending_named(again):
                self.st.log.append({'kind': 'brief_again'})
                return again
        out = dict(brief) if isinstance(brief, dict) else {}
        roles = dict(out.get('roles')) if isinstance(out.get('roles'), dict) else {}
        read = tool_ending_roles(self.st.tool_specs)
        for key in ('buy', 'refuse'):
            if read.get(key) and (not (isinstance(roles.get(key), str) and roles[key] in self.st.arg_schemas)):
                roles[key] = read[key]
        for key in ('buy_args', 'refuse_args'):
            if key in read and (not (isinstance(out.get(key), dict) and out[key])):
                out[key] = dict(read[key])
        out['roles'] = roles
        self.st.log.append({'kind': 'tool_roles', 'roles': {k: roles.get(k) for k in ('buy', 'refuse')}})
        return out

    def ask_brief(self) -> dict[str, Any]:
        """The BRIEF call on every tool's schema and the request; the strong model again if no buying role came back."""
        lines = []
        for tool in self.st.tool_specs:
            fn = tool['function']
            lines.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {to_json_text(fn.get('parameters') or {})}")
        prompt = BRIEF_PROMPT + '\nTOOLS:\n' + '\n'.join(lines) + '\n\nREQUEST:\n' + self.st.ask_text
        if self.st.policy.get('opening') is not None:
            prompt += '\n\nOPENING NOTE FROM THE STORE:\n' + to_json_text(clip_value(self.st.policy.get('opening'), 1500))
        brief = self.net.ask_structured(prompt, [FAST_MODEL], 4000) or {}
        if not self.st.model_broke and (not as_list(brief.get('items')) or not isinstance(brief.get('roles'), dict) or (not brief['roles'].get('buy'))):
            brief = self.net.ask_structured(prompt, [STRONG_MODEL, FAST_MODEL], 2500) or brief
        return brief

    def take_roles(self, brief: dict[str, Any]) -> None:
        """Roles that name a real tool; when one tool both checks out and declines, the model is questioned on which arguments select which."""
        roles = brief.get('roles') if isinstance(brief.get('roles'), dict) else {}
        self.st.jobs = {k: v for (k, v) in roles.items() if isinstance(v, str) and v in self.st.arg_schemas}
        for key in ('buy_args', 'refuse_args'):
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        (buy, refuse) = (self.st.jobs.get('buy'), self.st.jobs.get('refuse'))
        if buy and buy == refuse and (not brief['buy_args'] or not brief['refuse_args'] or brief['buy_args'] == brief['refuse_args']):
            fixed = self.net.ask_structured(mode_question(buy, self.st.arg_schemas.get(buy)), [FAST_MODEL, STRONG_MODEL], 600) or {}
            if isinstance(fixed.get('buy_args'), dict) and isinstance(fixed.get('refuse_args'), dict) and (fixed['buy_args'] != fixed['refuse_args']):
                (brief['buy_args'], brief['refuse_args']) = (fixed['buy_args'], fixed['refuse_args'])
        if F_MODEL_RETRY and F_SLOW_GUARD and buy and (buy == refuse):
            self.fill_modes(buy, brief)

    def fill_modes(self, tool: str, brief: dict[str, Any]) -> None:
        """One ending tool that checks out or declines by a choice argument: when the two sets of fixed arguments do
        not pick two different allowed choices, both take the choices that the tool's own description ties to checking
        out and to ending with no purchase (any other fixed argument is kept), so that either ending can still be sent."""
        spec = next((t['function'] for t in self.st.tool_specs if t['function']['name'] == tool), {})
        schema = self.st.arg_schemas.get(tool) or {}
        picked = mode_choices(str(spec.get('description') or ''), schema)
        if not picked:
            return
        (key, buy_value, refuse_value) = picked
        allowed = schema['properties'][key]['enum']
        (now_buy, now_refuse) = (brief['buy_args'].get(key), brief['refuse_args'].get(key))
        if now_buy in allowed and now_refuse in allowed and (now_buy != now_refuse):
            return
        brief['buy_args'] = {**brief['buy_args'], key: buy_value}
        brief['refuse_args'] = {**brief['refuse_args'], key: refuse_value}
        self.st.log.append({'kind': 'mode_fill', 'tool': tool, 'was': [now_buy, now_refuse]})

    def take_wants(self, brief: dict[str, Any]) -> None:
        """One Want per requested line (at least one), with its quantity and its rules."""
        for entry in as_list(brief.get('items')):
            if not isinstance(entry, dict):
                continue
            quantity = whole_number(entry.get('quantity'))
            count = quantity if quantity and quantity > 0 else 1
            musts = [str(m) for m in as_list(entry.get('musts')) if m]
            self.st.wants.append(Want(str(entry.get('name') or 'the product'), str(entry.get('shelf') or ''), count, musts))
        if not self.st.wants:
            self.st.wants = [Want('the requested product', '', 1, [])]

    def take_pointers(self, brief: dict[str, Any]) -> None:
        """Requirements whose value sits on another listing, tied to the Want whose rule names them."""
        for lookup in as_list(brief.get('lookups')):
            text = str(lookup).strip()
            if not any((text in item.rules for item in self.st.wants)):
                key = split_words(text.split(':')[0])
                text = next((m for item in self.st.wants for m in item.rules if split_words(m.split(':')[0]) == key), text)
            number = next((n for (n, item) in enumerate(self.st.wants, 1) if text in item.rules), 1)
            if text and (number, text) not in self.st.pointers:
                self.st.pointers.append((number, text))

    def take_limits(self, brief: dict[str, Any]) -> None:
        """Budget, message allowance (3 when the brief names a question tool but no number) and a smaller step limit."""
        budget = brief.get('budget')
        if isinstance(budget, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', budget):
            budget = float(budget)
        if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0):
            self.st.budget_cap = budget
        limit = whole_number(brief.get('messages'))
        if limit is not None and limit >= 0:
            self.st.msg_cap = limit
        elif self.st.jobs.get('ask'):
            self.st.msg_cap = 3
        steps = whole_number(brief.get('steps'))
        if steps is not None and 10 <= steps < self.st.step_cap:
            self.st.step_cap = steps

    def take_duties(self, brief: dict[str, Any]) -> None:
        """Steps the request asks for along the way, and the messages kept back for the ones due at buying time or at the end."""
        for entry in as_list(brief.get('process')):
            if isinstance(entry, dict) and str(entry.get('step') or '').strip():
                when = str(entry.get('when') or '').strip().lower()
                self.st.duties.append({'step': str(entry['step']).strip(), 'when': when if when in ('choose', 'order', 'end') else 'order', 'messages': max(0, whole_number(entry.get('messages')) or 0)})
        if self.st.jobs.get('ask'):
            talk = [p['messages'] for p in self.st.duties if p['when'] in ('order', 'end') and p['messages']]
            self.st.msg_reserve = min(max(talk), self.st.msg_cap) if talk else 0

    def move_lookup_musts(self) -> None:
        for (number, item) in enumerate(self.st.wants, 1):
            pointers = [need for (n, need) in self.st.pointers if n == number and need in item.rules]
            rest = [m for m in item.rules if m not in pointers]
            if not pointers or not rest:
                continue
            prompt = UNMIX_PROMPT + '\n\nREQUEST:\n' + self.st.ask_text + f'\n\nPRODUCT: {item.label}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n' + '\n'.join(pointers) + '\nOTHER REQUIREMENTS:\n' + '\n'.join(rest)
            got = self.net.ask_structured(prompt, [STRONG_MODEL, FAST_MODEL], 500) or {}
            moved = [m for m in rest if m in [str(x).strip() for x in as_list(got.get('describe_other'))]]
            if not moved:
                continue
            item.rules = [m for m in item.rules if m not in moved]
            joined = pointers[0] + ' (that listing: ' + '; '.join(moved) + ')'
            item.rules = [joined if m == pointers[0] else m for m in item.rules]
            self.st.pointers = [(n, joined if (n, need) == (number, pointers[0]) else need) for (n, need) in self.st.pointers]
            self.st.log.append({'kind': 'unmix', 'item': number, 'moved': moved})

    def needs_report(self) -> str:
        parts = []
        many = len(self.st.wants) > 1
        for (number, item) in enumerate(self.st.wants, 1):
            needs = [m + (f' (value on the page it points to: {self.st.pointer_values[m]})' if m in self.st.pointer_values else '') for m in item.rules + item.answered if m not in item.retired]
            head = f'item {number}: {item.label}' if many else f'item: {item.label}'
            if item.family:
                head += f' (store label: {item.family})'
            head += f', quantity {item.units}'
            text = head + '; requirements: ' + ('; '.join(needs) if needs else 'nothing beyond the kind of product itself')
            if item.loose:
                text += '; the shopper does not mind about: ' + ', '.join(item.loose)
            parts.append(text)
        brief = self.st.sheet
        if self.st.budget_cap is not None:
            scope = 'for each item' if brief.get('budget_scope') == 'each' else 'in total'
            parts.append(f"budget: at most {self.st.budget_cap} {brief.get('budget_unit') or ''} {scope}".replace('  ', ' '))
        pick = brief.get('pick')
        if pick == 'cheapest':
            parts.append('pick: the lowest-priced listing that fits')
        elif pick in ('most', 'least'):
            parts.append(f"pick: the {pick} {brief.get('pick_measure') or ''} of all fitting listings".replace('  ', ' '))
        else:
            parts.append('pick: any listing that meets everything')
        if brief.get('in_stock_only', True) is not False:
            parts.append('in stock only')
        for wish in as_list(brief.get('wishes')):
            parts.append(f'wish (only breaks ties): {wish}')
        for rule in as_list(brief.get('rules')):
            parts.append(f'rule the request sets: {rule}')
        if brief.get('quiet_decline') is True:
            parts.append('if nothing fits: end with no purchase, naming or suggesting no listing')
        for entry in self.st.duties:
            when = {'choose': 'before choosing', 'order': 'at checkout (Rok takes care of it when you check out)', 'end': 'at an ending with no purchase (Rok takes care of it then)'}[entry['when']]
            parts.append(f"step asked for by the request, {when}: {entry['step']}")
        facts = self.catalog.facts_view()
        if facts and (self.st.shows_facts.get(facts) is True or self.st.jobs.get('offer')):
            bare = sorted((t for t in self.catalog.page_views() | {self.st.jobs.get('offer') or ''} if t and self.st.shows_facts.get(t) is False))
            pages = sorted((t for t in self.catalog.page_views() if t != facts))
            parts.append(f'listing attributes are shown by {facts}' + (f"; {', '.join(bare)} show no attributes" if bare else '') + (f"; a listing's page (as meant by a page chain or a page check) is shown by {' or '.join(pages)}" if pages else ''))
        return '\n'.join(('- ' + p for p in parts))

    def status_report(self) -> str:
        left = self.net.steps_left()
        text = f'[Rok] steps used: {self.st.at_step} of {self.st.step_cap}, {left} left.'
        if left <= self.review.kept_steps() + 1:
            text += ' LAST STEPS: if the cart has a line for every item, check out now; end with no purchase only if your search showed that some item has no fitting listing (a lack of steps does not show that).'
        elif left <= 6 or (left <= len(self.st.wants) + 5 and self.st.buy_tries == 0):
            text += ' Only a few steps remain: make the cart hold exactly the chosen lines and check out now.'
        elif self.st.buy_tries == 0 and left <= 12:
            text += ' Steps are getting short: decide on your picks, fill the cart and check out within the next few steps.'
        elif self.st.buy_tries == 0 and (not self.st.cart_changed) and (self.st.at_step >= self.st.step_cap // 2):
            text += " Half of the steps are gone and the cart is still empty: get each item's fitting listing into the cart now (the store may yet sell one out)."
        live = {need for (_, need) in self.review.needs_in_force()}
        lookups = [need for (_, need) in self.st.pointers if need in live and need not in self.st.pointer_values]
        if lookups and self.st.buy_tries == 0:
            text += '\n[settle these first: open the listing each one points to, plus every listing its page sends you on to; the final page holds the value] ' + ' | '.join(lookups[:3])
        if self.st.hints:
            text += '\n[leads from the review] ' + ', '.join(self.st.hints)
        if self.st.store_notes:
            text += '\n[changes up to now] ' + ' | '.join(self.st.store_notes[-4:])
        text += '\n[ledger]\n' + self.needs_report()
        return text


class Catalog(Part):
    """What the store has shown: listing records by id, which views show attributes, stock, and store notes."""

    def page_views(self) -> set[str]:
        page = self.st.jobs.get('offer') or self.st.jobs.get('open')
        return {t for t in (page, self.st.jobs.get('compare')) if t}

    def note_view(self, name: str, args: dict[str, Any], seen: Any) -> None:
        """F_R12_CHAIN_GATE: opened-ids bookkeeping by what the view returned. Search rows teach which fields a row
        carries. A listing named in the call counts as opened when the view's record of it holds a field that no
        search row carries and that is not a nested block, so a view of attributes only never counts and a
        side-by-side call that returned no records counts for none; that tool is then known as a page view. Until
        one is known, a page view's record without a nested block counts, as before."""
        jobs = self.st.jobs
        if name in {jobs.get(k) for k in ('cart', 'cart_view', 'buy', 'refuse', 'ask')} - {None}:
            return
        inner = inner_of(seen)
        if not isinstance(inner, (dict, list)) or not inner or (isinstance(inner, dict) and inner.get('error')):
            return
        if name == jobs.get('search'):
            for (_, row) in self.records_of(inner):
                self.st.row_keys.update(row)
            return
        asked = {v.strip() for value in args.values() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)}
        for (ident, record) in self.records_of(inner):
            if ident not in asked:
                continue
            extra = bool(self.st.row_keys) and any((k not in self.st.row_keys and (not isinstance(v, (dict, list))) for (k, v) in record.items()))
            if extra:
                self.st.remark_tools.add(name)
            if extra or (not self.st.remark_tools and name in self.page_views() and (not any((isinstance(v, dict) for v in record.values())))):
                self.st.opened_ids.add(ident)

    def note_step(self, name: str, args: dict[str, Any], seen: Any) -> None:
        """Notes for the race-12 fixes: each search call (its phrase, the ids of its rows in order, and whether it
        asked for a first page, that is no argument holds a paging value an earlier result offered), the paging
        values its result offers, and the listings the attributes view has read."""
        inner = inner_of(seen)
        if not isinstance(inner, dict) or not inner or inner.get('error'):
            return
        if name == self.st.jobs.get('search'):
            later = any((v in self.st.page_marks for v in args.values() if isinstance(v, (str, int)) and (not isinstance(v, bool)) and v not in ('', 0)))
            self.st.page_marks.update((v for v in inner.values() if isinstance(v, str) and v and (' ' not in v)))
            self.st.searches.append((self.search_phrase(name, args), [i for (i, _) in self.records_of(inner)], not later))
        elif name in (self.facts_view(), self.st.jobs.get('open')):
            self.st.fact_read_ids.update((v.strip() for value in args.values() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)))

    def search_phrase(self, name: str, args: dict[str, Any]) -> str:
        """The query text of a search call: its required free-text argument, else its longest text argument."""
        param = self.lone_id_param(name)
        texts = [v for v in args.values() if isinstance(v, str)]
        value = args.get(param) if param else max(texts, key=len, default='')
        return ' '.join(str(value or '').lower().split())

    def next_page_arg(self, name: str, args: dict[str, Any], inner: Any) -> tuple[str, Any] | None:
        """The argument and value that ask a search for its next page, read from the shapes alone: the search
        tool's one optional text argument besides the query, and the result's one top-level text value without
        spaces. None when either is missing or not single."""
        schema = self.st.arg_schemas.get(name) or {}
        props = schema.get('properties') or {}
        query = self.lone_id_param(name)
        optional = [k for (k, v) in props.items() if k != query and k not in (schema.get('required') or []) and isinstance(v, dict) and ('string' in schema_types(v))]
        values = [v for v in inner.values() if isinstance(v, str) and v and (' ' not in v)] if isinstance(inner, dict) else []
        if len(optional) != 1 or len(values) != 1:
            return None
        return (optional[0], values[0])

    def facts_view(self) -> str | None:
        known = next((t for (t, ok) in self.st.shows_facts.items() if ok and self.lone_id_param(t)), None)
        if known:
            return known
        tool = self.st.jobs.get('open')
        return tool if self.st.shows_facts.get(tool or '') is not False and self.lone_id_param(tool) else None

    def side_facts_known(self) -> bool | None:
        return self.st.shows_facts.get(self.st.jobs.get('compare') or '')

    def side_view_bare(self) -> bool:
        seen = self.side_facts_known()
        return seen is False or (seen is None and (self.st.shows_facts.get(self.facts_view() or '') is True or bool(self.st.jobs.get('offer'))))

    def listing_of(self, node: Any, in_list: bool=False, depth: int=0) -> str | None:
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if self.st.row_id_key and looks_like_id(node.get(self.st.row_id_key), True):
            return node[self.st.row_id_key].strip()
        nested = any((isinstance(v, dict) for v in node.values()))
        if not (in_list or depth == 0 or nested):
            return None
        found = guess_id_key([node])
        if found and ('id' in str(found).lower() or node[found].strip() in self.st.row_index):
            return node[found].strip()
        return None

    def learn_id_field(self, node: list[Any]) -> bool:
        rows = [n for n in node if isinstance(n, dict) and len(n) >= 3]
        if not rows or len(rows) < len(node) / 2:
            return False
        if self.st.row_id_key and all((looks_like_id(r.get(self.st.row_id_key), True) for r in rows)):
            return True
        found = guess_id_key(rows)
        if found:
            self.st.row_id_key = found
            return True
        return False

    def take_in(self, observation: Any) -> None:

        def walk(node: Any, depth: int, in_list: bool=False) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                ident = self.listing_of(node, in_list, depth)
                if ident:
                    if ident not in self.st.row_index:
                        self.st.first_seen.append(ident)
                        self.st.row_index[ident] = {}
                    old = self.st.row_index[ident]
                    if any((old.get(k) is True and v is False for (k, v) in node.items())):
                        self.st.off_sale.add(ident)
                    elif any((old.get(k) is False and v is True for (k, v) in node.items())):
                        self.st.off_sale.discard(ident)
                    self.st.row_index[ident].update(node)
                    if attribute_block(node):
                        self.st.full_pages[ident] = dict(self.st.row_index[ident])
                    elif ident in self.st.full_pages:
                        self.st.full_pages[ident].update(node)
                    return
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                rows = self.learn_id_field(node)
                for value in node:
                    walk(value, depth + 1, rows)
        walk(observation, 0)
        if F_SHELF_GUARD:
            self.section_rules()

    def section_rules(self) -> None:
        """A rule whose value is a category label of the rows seen, named in the request but not written there as that
        rule, is the work sheet reading the store's label for a product as its seller or brand: drop the rule and take
        the label as the store label of that want (and as its name when the name says nothing of the product). A value
        another field of a row carries, a value another want uses, or a want under another label is left alone. Nothing
        is done while a single row is in: one row gives every short field one value, so it says nothing of which field
        holds the labels."""
        if len(self.st.row_index) < 2 or not any((want.rules for want in self.st.wants)):
            return
        field = self.category_field()
        if not field:
            return
        said = flat_text(self.st.ask_text)
        labels: dict[str, str] = {}
        elsewhere: set[str] = set()
        for row in self.st.row_index.values():
            for (key, value) in row.items():
                if isinstance(value, str) and value.strip():
                    if key == field:
                        labels.setdefault(flat_text(value), value.strip())
                    else:
                        elsewhere |= {flat_text(value)}
        pointed = {need for (_, need) in self.st.pointers}
        for want in self.st.wants:
            others = [w for w in self.st.wants if w is not want]
            for rule in list(want.rules):
                (prop, sep, value) = rule.partition(':')
                if not sep:
                    (prop, value) = ('', prop)
                found = flat_text(value)
                if rule in pointed or not found or found not in labels or found in elsewhere or found not in said:
                    continue
                if flat_text(prop) and flat_text(prop) + ': ' + found in said:
                    continue
                if flat_text(want.family) not in ('', found) or any((found in (flat_text(w.family), flat_text(w.label)) for w in others)):
                    continue
                if word_stems(prop) and word_stems(prop) <= word_stems(field):
                    continue
                label = labels[found]
                want.rules.remove(rule)
                if not want.family.strip():
                    want.family = label
                if flat_text(want.label) != found and (found in flat_text(want.label) or not (word_stems(want.label) & word_stems(label) or near_words(want.label, label))):
                    want.label = label
                self.st.log.append({'kind': 'store_label_rule', 'rule': rule, 'label': label})

    def enum_labels(self) -> set[str]:
        found: set[str] = set()

        def walk(schema: Any, depth: int) -> None:
            if depth > 6 or not isinstance(schema, dict):
                return
            for option in schema.get('enum') or []:
                if isinstance(option, str):
                    found.add(option.strip().lower())
            for value in schema.values():
                if isinstance(value, dict):
                    walk(value, depth + 1)
        for schema in self.st.arg_schemas.values():
            walk(schema, 0)
        return found

    def category_field(self) -> str | None:
        counts: dict[str, set[str]] = {}
        present: dict[str, int] = {}
        for row in self.st.row_index.values():
            for (key, value) in row.items():
                if isinstance(value, str) and 0 < len(value) <= 40 and (not re.search('\\d{3}', value)):
                    counts.setdefault(key, set()).add(value)
                    present[key] = present.get(key, 0) + 1
        total = max(1, len(self.st.row_index))
        known = self.enum_labels()
        best = None
        for (key, values) in counts.items():
            if present[key] < 0.8 * total or len(values) > 40:
                continue
            listed = sum((1 for v in values if v.strip().lower() in known)) / len(values)
            mean = sum((len(v) for v in values)) / len(values)
            score = (-listed, len(values), -present[key], -mean)
            if best is None or score < best[0]:
                best = (score, key)
        return best[1] if best else None

    def field_table(self) -> list[tuple[str, str, int, list[str]]]:
        cat_key = self.category_field()
        groups: dict[str, dict[str, Any]] = {}
        for ident in self.st.first_seen:
            record = self.st.full_pages.get(ident)
            if not record:
                continue
            block = attribute_block(record) or {}
            label = str(record.get(cat_key) or 'listings') if cat_key else 'listings'
            group = groups.setdefault(label, {'n': 0, 'keys': {}})
            group['n'] += 1
            done: set[str] = set()
            for (key, value) in block.items():
                name = split_words(key)
                if not name or name in done:
                    continue
                done.add(name)
                slot = group['keys'].setdefault(name, {'n': 0, 'values': [], 'per': {}})
                slot['n'] += 1
                text = str(value).strip().lower()
                slot['per'][ident] = text
                if text not in slot['values']:
                    slot['values'].append(text)
        wanted = [(split_words(item.family or ''), split_words(item.label), {w for w in word_stems(item.family or '') | word_stems(item.label) if len(w) >= 4}) for item in self.st.wants]
        matched = {label for label in groups if any((split_words(label) in (catalog, name) or {w for w in word_stems(label) if len(w) >= 4} & long for (catalog, name, long) in wanted))}
        if matched:
            groups = {label: group for (label, group) in groups.items() if label in matched}
        table = []
        for (label, group) in sorted(groups.items(), key=lambda kv: -kv[1]['n']):
            kept: list[tuple[str, dict[str, Any]]] = []
            for (name, slot) in sorted(group['keys'].items(), key=lambda kv: -kv[1]['n']):
                values = list(slot['per'].values())
                if not values:
                    continue
                mean = sum((len(v) for v in values)) / len(values)
                if mean > 40 or sum((1 for v in values if ' / ' in v)) * 2 >= len(values):
                    continue
                if F_R12_MENU:
                    if len(values) >= 4 and len(set(values)) == len(values) and (mean > 12) and (sum((1 for v in values if ' ' not in v)) * 2 >= len(values)):
                        continue
                elif len(values) >= 4 and len(set(values)) == len(values) and (mean > 12):
                    continue
                twin = any((name.startswith(other + ' ') and slot['per'] == spec['per'] for (other, spec) in kept))
                if twin:
                    continue
                kept.append((name, slot))
            for (name, slot) in kept[:14]:
                table.append((label, name, slot['n'], [v[:40] for v in slot['values'][:4]]))
        return table

    def ids_by_category(self) -> list[list[str]]:
        cat_key = self.category_field()
        groups: dict[str, list[str]] = {}
        for ident in self.st.first_seen:
            label = str(self.st.row_index[ident].get(cat_key) or '') if cat_key else ''
            groups.setdefault(label, []).append(ident)
        wanted = [(split_words(item.family or ''), split_words(item.label), {w for w in word_stems(item.family or '') | word_stems(item.label) if len(w) >= 4}) for item in self.st.wants]
        matched = [ids for (label, ids) in groups.items() if any((split_words(label) in (catalog, name) or {w for w in word_stems(label) if len(w) >= 4} & long for (catalog, name, long) in wanted))]
        pool = matched or list(groups.values())
        return sorted(pool, key=lambda ids: -len(ids))[:max(1, len(self.st.wants))]

    def lone_id_param(self, tool: str | None) -> str | None:
        schema = self.st.arg_schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for (k, v) in props.items() if isinstance(v, dict) and 'string' in schema_types(v) and ('enum' not in v)]
        target = next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not tool or not target or any((k != target for k in schema.get('required') or [])):
            return None
        return target

    def price_now(self, ident: str) -> float | None:
        value = (self.st.row_index.get(ident) or {}).get(self.st.price_field) if self.st.price_field else None
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and (not isinstance(v, bool))), None)
        if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', value):
            value = float(value)
        return float(value) if isinstance(value, (int, float)) and (not isinstance(value, bool)) else None

    def in_stock_now(self, ident: str) -> bool:
        if self.review.is_gone(ident):
            return False
        value = (self.st.row_index.get(ident) or {}).get(self.st.stock_field) if self.st.stock_field else None
        return not (value is False or (isinstance(value, (int, float)) and (not isinstance(value, bool)) and (value <= 0)))

    def records_of(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        if isinstance(observation, dict):
            ident = self.listing_of(observation)
            if ident:
                return [(ident, observation)]
            found: list[tuple[str, dict[str, Any]]] = []
            for value in observation.values():
                if isinstance(value, list):
                    found += [(i, r) for r in value if isinstance(r, dict) for i in [self.listing_of(r, True)] if i]
                elif isinstance(value, dict):
                    found += [(i, r) for v in value.values() if isinstance(v, list) for r in v if isinstance(r, dict) for i in [self.listing_of(r, True)] if i]
            return found
        if isinstance(observation, list):
            return [(i, r) for r in observation if isinstance(r, dict) for i in [self.listing_of(r, True)] if i]
        return []

    def note_fields_shown(self, name: str, observation: Any) -> None:
        for (ident, record) in self.records_of(observation):
            keys = set(record)
            self.st.tool_fields.setdefault(name, set()).update(keys)
            (version, have) = self.st.view_fields.get(ident, (-1, set()))
            self.st.view_fields[ident] = (self.st.epoch, (have if version == self.st.epoch else set()) | keys)

    def nothing_new(self, name: str, ident: str) -> bool:
        (version, have) = self.st.view_fields.get(ident, (-1, set()))
        keys = self.st.tool_fields.get(name)
        return bool(keys) and version == self.st.epoch and (keys <= have) and (ident not in self.st.sold_at)

    def visible_text(self, observation: Any) -> str:
        if isinstance(observation, str):
            return observation
        found: list[str] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 4:
                return
            if isinstance(node, str) and len(node.split()) >= 2:
                found.append(node)
            elif isinstance(node, dict):
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1)
        walk(observation, 0)
        return ' '.join(found)[:800] if found else to_json_text(observation)[:400]

    def note_lines(self, observation: Any) -> list[str]:
        lines: list[str] = []
        if isinstance(observation, dict) and (not self.listing_of(observation)):
            for (key, value) in observation.items():
                if isinstance(value, str) and len(value.split()) >= 5:
                    lines.append(value)
                elif isinstance(value, list) and value and all((isinstance(v, str) for v in value)):
                    lines.extend((v for v in value if len(v.split()) >= 5))
                elif isinstance(value, dict) and (not self.listing_of(value, False, 1)):
                    for inner in value.values():
                        if isinstance(inner, str) and len(inner.split()) >= 6:
                            lines.append(inner)
        return [line for line in lines if line not in self.st.heard_keys]

    def read_store_note(self, lines: list[str]) -> str:
        self.st.heard_keys.update(lines)
        prompt = NOTE_PROMPT + '\n\nITEMS AND REQUIREMENTS NOW:\n' + self.brief.needs_report() + '\n\nTHE NOTE:\n' + '\n'.join(lines)
        got = self.net.ask_structured(prompt, [FAST_MODEL, BACKUP_MODEL], 1200) or {}
        gone = [str(g) for g in as_list(got.get('gone')) if isinstance(g, (str, int))]
        self.st.off_sale.update((g for g in gone if g in self.st.row_index))
        self.st.note_said_sold = bool(gone) or got.get('sold_out') is True
        if self.st.note_said_sold and len(lines) == 1:
            self.st.sold_phrases.update(lines)
        if got.get('kind') not in (None, 'change'):
            return ''
        changed = []
        budget = got.get('budget')
        if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0) and (budget != self.st.budget_cap):
            self.st.budget_cap = budget
            changed.append(f'budget now {budget}')
        added: set[str] = set()
        drops = [e for e in as_list(got.get('drop')) if isinstance(e, dict) and str(e.get('need') or '').strip()]
        adds = [e for e in as_list(got.get('add')) if isinstance(e, dict) and str(e.get('need') or '').strip()]
        for entry in drops:
            item = self.st.wants[min(max(1, whole_number(entry.get('item')) or 1), len(self.st.wants)) - 1]
            target = str(entry['need']).strip()
            live = [m for m in item.rules + item.answered if m not in item.retired]
            hit = [m for m in live if m == target]
            if not hit:
                hit = [m for m in live if split_words(m.split(':')[0]) == split_words(target.split(':')[0])][:1]
            item.retired.extend(hit or [target])
            changed.append(f'dropped {target}')
        for entry in adds:
            item = self.st.wants[min(max(1, whole_number(entry.get('item')) or 1), len(self.st.wants)) - 1]
            need = str(entry['need']).strip()
            if need in item.retired:
                item.retired.remove(need)
            if need not in item.rules + item.answered:
                item.answered.append(need)
            added.add(need)
            changed.append(f'added {need}')
        if gone:
            changed.append('unavailable: ' + ', '.join(gone))
        if got.get('price_moved') is True:
            changed.append('a price changed')
        summary = str(got.get('summary') or '').strip()
        if changed or summary:
            self.st.epoch += 1
            self.st.last_marks = []
            self.st.store_notes.append(summary or '; '.join(changed))
        return (summary + (' (' + '; '.join(changed) + ')' if changed else '')).strip()


class Pages(Part):
    """Result shapes and paging, cheapest picks, and the extra reads made before buying."""

    def result_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        if name in self.st.result_shapes:
            return self.st.result_shapes[name]
        if not isinstance(observation, dict):
            return None
        lists = [k for (k, v) in observation.items() if isinstance(v, list) and len(v) >= 2 and all((isinstance(r, dict) for r in v))]
        if not lists:
            return None
        self.st.result_shapes[name] = None
        sample = {k: v[:3] if k in lists else v for (k, v) in observation.items()}
        prompt = SHAPE_PROMPT + '\n\nTOOL: ' + name + ' | parameters: ' + to_json_text(self.st.arg_schemas.get(name) or {}) + '\n\nRESULT (first rows only):\n' + to_json_text(clip_value(sample, 120))[:4000]
        got = self.net.ask_structured(prompt, [FAST_MODEL, STRONG_MODEL], 1500) or {}
        rows = got.get('rows') if got.get('rows') in lists else None
        if not rows:
            return None
        first = observation[rows][0]
        props = (self.st.arg_schemas.get(name) or {}).get('properties') or {}

        def named(key: Any, where: dict[str, Any]) -> str | None:
            return key if isinstance(key, str) and key in where else None
        shape = {'rows': rows, 'count': named(got.get('count'), observation), 'next': named(got.get('next'), observation), 'next_arg': named(got.get('next_arg'), props), 'price': named(got.get('price'), first), 'stock': named(got.get('stock'), first)}
        if shape['count'] and (isinstance(observation[shape['count']], bool) or not isinstance(observation[shape['count']], int)):
            shape['count'] = None
        self.st.result_shapes[name] = shape
        self.st.price_field = self.st.price_field or shape['price']
        self.st.stock_field = self.st.stock_field or shape['stock']
        self.st.log.append({'kind': 'shape', 'tool': name, 'shape': shape})
        return shape

    def page_state(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        shape = self.result_shape(name, observation)
        if not shape or not shape['count'] or (not shape['next']) or (not shape['next_arg']) or (not isinstance(observation, dict)):
            return None
        (rows, count, value) = (observation.get(shape['rows']), observation.get(shape['count']), observation.get(shape['next']))
        if not isinstance(rows, list) or not rows or isinstance(count, bool) or (not isinstance(count, int)):
            return None
        if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))):
            return None
        return (count, len(rows), shape['next_arg'], value)

    def price_ordered(self, observation: Any) -> bool:
        prices = [self.catalog.price_now(i) for (i, _) in self.catalog.records_of(observation)]
        return len(prices) >= 3 and None not in prices and all((a <= b for (a, b) in zip(prices, prices[1:])))

    def read_on_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        self.st.page_results = []
        pick = self.st.sheet.get('pick')
        plan = self.page_state(name, observation) if pick in ('cheapest', 'most', 'least') else None
        if not plan:
            return ''
        (count, shown, param, value) = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and (not isinstance(given, bool)) else bool(str(given or '').strip())
        if value is None or later or count <= shown:
            return ''
        if pick == 'cheapest' and self.price_ordered(observation):
            return ''
        pages = -(-(count - shown) // shown)
        if pages > 2 or self.st.extra_pages + pages > EXTRA_PAGE_CAP or self.net.steps_left() - self.review.kept_steps() - pages < 12:
            return f"\n[Rok] {count} listings matched this search and {shown} were shown, not sorted by price. The pick has to consider every fitting listing: search with narrower words (the requirement's value) or fetch the next page with {param}={to_json_text(value)}."
        texts = []
        for _ in range(pages):
            faults: list[str] = []
            call = coerce_to_schema({**args, param: value}, self.st.arg_schemas.get(name), 'args', faults)
            if faults:
                break
            reply = self.net.take_step(name, call)
            if reply is None or self.st.over:
                break
            self.st.extra_pages += 1
            seen = reply.get('observation')
            self.take_own_result(name, seen)
            self.st.page_results.append(seen)
            shown_text = self.loop.render(seen)
            self.st.reads[name + to_json_text(call)] = (self.st.epoch, self.st.at_step, shown_text)
            texts.append(f'{param}={to_json_text(value)}: {shown_text}')
            after = self.page_state(name, seen)
            value = after[3] if after else None
            if value is None:
                break
        self.st.log.append({'kind': 'pages', 'tool': name, 'read': len(texts)})
        return '\n[Rok fetched the remaining pages of this search itself, one step per page]\n' + '\n'.join(texts) if texts else ''

    def cheapest_hint(self, observations: list[Any]) -> str:
        if self.st.sheet.get('pick') != 'cheapest' or not self.st.price_field:
            return ''
        found = {i for seen in observations for (i, _) in self.catalog.records_of(seen)}
        rows = [(self.catalog.price_now(i), i) for i in found if self.catalog.in_stock_now(i) and self.catalog.price_now(i) is not None]
        if len(rows) < 2:
            return ''
        return '\n[Rok] rows here that can be bought, lowest price first: ' + ', '.join((f'{i} ({p:g})' for (p, i) in sorted(rows)[:6]))

    def wider_search_hold(self) -> str:
        if self.st.wider_done or self.st.sheet.get('pick') != 'cheapest' or self.net.steps_left() < 14 + 2 * len(self.st.wants):
            return ''
        self.st.wider_done = True
        lines = [f'{i} at {self.catalog.price_now(i):g}' for i in self.review.cart_listings() if self.catalog.price_now(i) is not None]
        return f"Not sent yet (no step spent). {self.net.steps_left()} steps remain and the pick is the cheapest. Searches match against a listing's title, so searching for a requirement's value (often found only in the attributes) turns up few of the fitting listings, and rows are not sorted by price. Before you check out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words taken from fitting titles you have seen) and consider only rows that can be bought and cost less than the cart's line" + (f" ({'; '.join(lines)})" if lines else '') + '; read the attributes of the cheapest of those rows, swap in any that fits, then check out.'

    def read_missing_facts(self, ids: list[str], keep: int) -> list[str]:
        tool = self.catalog.facts_view()
        target = self.catalog.lone_id_param(tool)
        if not tool or not target:
            return []
        texts = []
        for ident in [i for i in ids if i in self.st.row_index and i not in self.st.full_pages and (not self.review.is_gone(i))]:
            if self.net.steps_left() <= keep or self.st.over or self.st.shows_facts.get(tool) is False:
                break
            faults: list[str] = []
            args = coerce_to_schema({target: ident}, self.st.arg_schemas.get(tool), 'args', faults)
            if faults:
                break
            reply = self.net.take_step(tool, args)
            if reply is None:
                break
            seen = reply.get('observation')
            self.take_own_result(tool, seen)
            self.catalog.note_fields_shown(tool, seen)
            if isinstance(seen, dict) and self.catalog.listing_of(seen):
                self.st.shows_facts[tool] = self.st.shows_facts.get(tool, False) or bool(attribute_block(seen))
            texts.append(f'{ident}: ' + self.loop.render(seen)[:1500])
        if texts:
            self.st.log.append({'kind': 'fill_facts', 'n': len(texts)})
        return texts

    def read_cheaper_rows(self) -> list[str]:
        if self.st.sheet.get('pick') != 'cheapest':
            return []
        mark = self.review.cart_key() + f'#{self.st.epoch}'
        cat_key = self.catalog.category_field()
        if mark in self.st.cheaper_done or not cat_key:
            return []
        self.st.cheaper_done.add(mark)
        cart = self.review.cart_listings()
        picks: list[str] = []
        for line in cart:
            price = self.catalog.price_now(line)
            kind = str((self.st.row_index.get(line) or {}).get(cat_key) or '').strip().lower()
            if price is None or not kind:
                continue
            pool = [i for i in self.st.first_seen if i not in cart and i not in self.st.full_pages and self.catalog.in_stock_now(i) and (str((self.st.row_index.get(i) or {}).get(cat_key) or '').strip().lower() == kind) and ((self.catalog.price_now(i) if self.catalog.price_now(i) is not None else float('inf')) < price)]
            pool.sort(key=lambda i: self.catalog.price_now(i) or 0.0)
            picks += [i for i in pool[:2] if i not in picks]
        return self.read_missing_facts(picks[:CHEAPER_READ_CAP], self.review.kept_steps() + 4)

    def short_rows(self, observation: Any, limit: int=6) -> str:
        """A compact view of a search result: the match count when the result gives one, then its first rows."""
        inner = inner_of(observation)
        rows = self.catalog.records_of(inner)
        count = next((v for v in inner.values() if isinstance(v, int) and (not isinstance(v, bool))), None) if isinstance(inner, dict) else None
        lines = [f'{ident}: ' + to_json_text(clip_value(row, 60))[:360] for (ident, row) in rows[:limit]]
        return (f'{count} matched; ' if count is not None else '') + (' | '.join(lines) if lines else 'no rows')

    def run_search(self, phrase: str) -> Any:
        """One search that Rok makes itself; the observation, or None when it could not be sent."""
        tool = self.st.jobs.get('search')
        param = self.catalog.lone_id_param(tool)
        if not tool or not param or self.st.over:
            return None
        faults: list[str] = []
        args = coerce_to_schema({param: phrase}, self.st.arg_schemas.get(tool), 'args', faults)
        if faults:
            return None
        reply = self.net.take_step(tool, args)
        if reply is None:
            return None
        seen = reply.get('observation')
        self.take_own_result(tool, seen)
        self.st.reads[tool + to_json_text(args)] = (self.st.epoch, self.st.at_step, self.loop.render(seen))
        return seen

    def told_searches(self) -> str:
        """F_R12_TOLD_SEARCH: for each value the shopper told with a number and a unit, one search that pairs it
        with the other values known for the same line (those a seen listing carries word for word) and then the
        line's product noun; the other spelling of the value (with or without the space) only when no row of the
        result holds it, or always with F_R12_BOTH_SPELLINGS (the store ranks the two spellings differently even
        when both find rows that hold the value). At most TOLD_SEARCH_CAP searches in an episode, while 14 steps
        beyond the kept ones remain."""
        texts: list[str] = []
        pointed = {need for (_, need) in self.st.pointers}
        done = {p for (p, _, _) in self.st.searches}
        if F_R12_TOLD_UNSEARCHED:
            searched = set(re.findall('\\d+(?:\\.\\d+)?', ' '.join(done)))
        for (number, item) in enumerate(self.st.wants, 1):
            for need in [m for m in item.answered if m not in item.retired and m not in pointed]:
                value = need.partition(':')[2] or need
                for (amount, unit) in told_numbers(value)[:1]:
                    if F_R12_TOLD_UNSEARCHED and amount in searched:
                        continue
                    found = re.search(re.escape(amount) + '\\s*' + re.escape(unit), value.lower())
                    told = found.group(0) if found else amount + ' ' + unit
                    others = [v for v in self.review.line_values(number) if v != ' '.join(value.lower().split()) and (not word_run(told, v))][:2]
                    for spelling in (told, told.replace(' ', '') if ' ' in told else amount + ' ' + unit):
                        if self.st.told_runs >= TOLD_SEARCH_CAP or self.net.steps_left() - self.review.kept_steps() < 14 or self.st.over:
                            return self.told_text(texts)
                        phrase = ' '.join([spelling] + others + [self.review.line_noun(number)]).strip()
                        if phrase in done:
                            break
                        seen = self.run_search(phrase)
                        if seen is None:
                            return self.told_text(texts)
                        self.st.told_runs += 1
                        done.add(phrase)
                        texts.append(f'{phrase!r}: ' + self.short_rows(seen))
                        rows = ' '.join((to_json_text(r) for (_, r) in self.catalog.records_of(inner_of(seen)))).lower()
                        if not F_R12_BOTH_SPELLINGS and re.search(re.escape(amount) + '\\s*' + re.escape(unit), rows):
                            break
        return self.told_text(texts)

    def told_text(self, texts: list[str]) -> str:
        if not texts:
            return ''
        self.st.log.append({'kind': 'told_search', 'n': len(texts)})
        return '\n[Rok searched each told value together with the other values known for the same item, one step each]\n' + '\n'.join(texts)

    def drop_search(self, number: int) -> str:
        """F_R12_TOLD_SEARCH: one fresh search after a requirement was dropped, with the values still in force for
        that line and then its product noun; at most two in an episode, and only while the search, one cart call
        and the steps kept for the ending still fit (a store refusal comes late, often with four or five steps left)."""
        if not number or self.st.drop_searches >= 2 or self.net.steps_left() - self.review.kept_steps() < 2 or self.st.over:
            return ''
        phrase = ' '.join(self.review.line_values(number)[:2] + [self.review.line_noun(number)]).strip()
        if not phrase or phrase in {p for (p, _, _) in self.st.searches}:
            return ''
        self.st.drop_searches += 1
        seen = self.run_search(phrase)
        if seen is None:
            return ''
        self.st.log.append({'kind': 'drop_search', 'phrase': phrase})
        return f'\n[Rok searched again with the requirements still in force: {phrase!r}] ' + self.short_rows(seen)

    def valid_rows(self, inner: Any) -> list[str]:
        """Rows of a result that could be picked: of a requested category, in stock and within the budget."""
        cap = self.st.budget_cap if isinstance(self.st.budget_cap, (int, float)) and (not isinstance(self.st.budget_cap, bool)) else None
        return [i for (i, _) in self.catalog.records_of(inner) if self.review.wanted_kind(i) and self.catalog.in_stock_now(i) and (cap is None or (self.catalog.price_now(i) or 0) <= cap)]

    def page_further(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """F_R12_TOLD_SEARCH, with a pick of any fitting listing: a search whose phrase carries a number value told
        for a line gets its next page fetched while the page just read held a valid candidate (valid_rows); at
        most two further pages per phrase and TOLD_PAGE_CAP in an episode, inside the shared page cap."""
        if self.st.sheet.get('pick') in ('cheapest', 'most', 'least'):
            return ''
        phrase = self.catalog.search_phrase(name, args)
        told = {pair for (_, need) in self.review.needs_in_force() for pair in told_numbers(need.partition(':')[2] or need)}
        if not phrase or not set(told_numbers(phrase)) & told:
            return ''
        texts: list[str] = []
        inner = inner_of(observation)
        while self.st.page_more.get(phrase, 0) < 2 and self.st.told_pages < TOLD_PAGE_CAP and self.st.extra_pages < EXTRA_PAGE_CAP and (self.net.steps_left() - self.review.kept_steps() >= 13) and (not self.st.over):
            step = self.catalog.next_page_arg(name, args, inner)
            if not step or not self.valid_rows(inner):
                break
            faults: list[str] = []
            call = coerce_to_schema({**args, step[0]: step[1]}, self.st.arg_schemas.get(name), 'args', faults)
            if faults:
                break
            reply = self.net.take_step(name, call)
            if reply is None:
                break
            self.st.extra_pages += 1
            self.st.told_pages += 1
            self.st.page_more[phrase] = self.st.page_more.get(phrase, 0) + 1
            seen = reply.get('observation')
            self.take_own_result(name, seen)
            self.st.reads[name + to_json_text(call)] = (self.st.epoch, self.st.at_step, self.loop.render(seen))
            texts.append(f'{step[0]}={to_json_text(step[1])}: ' + self.short_rows(seen))
            inner = inner_of(seen)
        if not texts:
            return ''
        self.st.log.append({'kind': 'told_pages', 'phrase': phrase, 'n': len(texts)})
        return '\n[Rok fetched the next page of this search itself, one step per page]\n' + '\n'.join(texts)

    def read_unread_lines(self, ids: list[str], keep: int) -> list[str]:
        """F_R12_READ_LINES: each cart line that the attributes view has not read in this episode is read with it
        now, one step each, while more than keep steps remain (keep = the steps the order itself still needs: a
        read as late as the step before the order still counts)."""
        tool = self.catalog.facts_view() or self.st.jobs.get('open')
        target = self.catalog.lone_id_param(tool)
        if not tool or not target:
            return []
        texts = []
        for ident in [i for i in ids if i in self.st.row_index and i not in self.st.fact_read_ids and (not self.review.is_gone(i))]:
            if self.net.steps_left() <= keep or self.st.over:
                break
            faults: list[str] = []
            args = coerce_to_schema({target: ident}, self.st.arg_schemas.get(tool), 'args', faults)
            if faults:
                break
            reply = self.net.take_step(tool, args)
            if reply is None:
                break
            seen = reply.get('observation')
            self.take_own_result(tool, seen)
            self.catalog.note_fields_shown(tool, seen)
            texts.append(f'{ident}: ' + self.loop.render(seen)[:1500])
        if texts:
            self.st.log.append({'kind': 'read_lines', 'n': len(texts)})
        return texts

    def take_own_result(self, name: str, observation: Any) -> str:
        self.catalog.take_in(observation)
        lines = self.catalog.note_lines(observation)
        if not lines:
            return ''
        meaning = self.catalog.read_store_note(lines)
        if meaning:
            self.st.pending_notes.append(f'(seen on {name}) {meaning}')
        return meaning

    def spread_picks(self, ranked: list[list[str]], room: int) -> list[str]:
        """Unread listings for one side-by-side call of at most room listings: an equal share from each wanted
        category first, then the rest of the room from the categories in the same order."""
        unread = [[i for i in ids if i not in self.st.full_pages] for ids in ranked]
        share = max(1, room // max(1, len(unread)))
        picks = [i for ids in unread for i in ids[:share]] + [i for ids in unread for i in ids[share:]]
        return picks[:room]

    def open_side_batch(self) -> None:
        tool = self.st.jobs.get('compare')
        schema = self.st.arg_schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        target = next((k for (k, v) in props.items() if isinstance(v, dict) and 'array' in schema_types(v)), None)
        if not tool or not target or any((k != target for k in schema.get('required') or [])) or self.catalog.side_view_bare():
            self.open_single_batch()
            return
        if self.net.steps_left() <= 12:
            return
        cap = props[target].get('maxItems') if isinstance(props[target].get('maxItems'), int) else 4
        ranked = self.catalog.ids_by_category()
        share = max(1, cap // max(1, len(ranked)))
        picks: list[str] = []
        for ids in ranked:
            picks += [i for i in ids if i not in self.st.full_pages][:share]
        picks = picks[:cap]
        if F_TOOL_LIMITS and len(picks) > self.loop.side_room():
            picks = self.spread_picks(ranked, self.loop.side_room())
        if len(picks) < 2:
            return
        faults: list[str] = []
        args = coerce_to_schema({target: picks}, schema, 'args', faults)
        if faults:
            return
        before = len(self.st.full_pages)
        reply = self.net.take_step(tool, args)
        if reply is not None:
            seen = reply.get('observation')
            self.take_own_result(tool, seen)
            self.catalog.note_fields_shown(tool, seen)
            rows = self.catalog.records_of(seen)
            if rows:
                self.st.shows_facts[tool] = self.st.shows_facts.get(tool, False) or any((attribute_block(r) for (_, r) in rows))
            self.st.log.append({'kind': 'widen', 'ids': picks, 'opened': len(self.st.full_pages) - before})
            if rows and self.st.shows_facts.get(tool) is False:
                self.open_single_batch()
                return
            if len(self.st.full_pages) == before and len(picks) > 2 and (self.net.steps_left() > 12):
                faults = []
                args = coerce_to_schema({target: picks[:2]}, schema, 'args', faults)
                if not faults:
                    reply = self.net.take_step(tool, args)
                    if reply is not None:
                        self.take_own_result(tool, reply.get('observation'))

    def open_single_batch(self) -> None:
        tool = self.catalog.facts_view()
        target = self.catalog.lone_id_param(tool)
        if not tool or not target:
            return
        lanes: list[list[str]] = []
        for ids in self.catalog.ids_by_category():
            opened = sum((1 for i in ids if i in self.st.full_pages))
            pool = [i for i in ids if i not in self.st.full_pages and self.catalog.in_stock_now(i)]
            if self.st.sheet.get('pick') == 'cheapest':
                pool.sort(key=lambda i: self.catalog.price_now(i) if self.catalog.price_now(i) is not None else float('inf'))
            lanes.append(pool[:max(0, min(2, 8 - opened))])
        queue = [lane[k] for k in range(2) for lane in lanes if k < len(lane)]
        self.read_missing_facts(queue[:5], 22)


class Interview(Part):
    """When to question the shopper, which questions to send in which batch, and what the answers bring to the ledger."""

    def messages_free(self) -> int:
        if F_MODEL_RETRY and F_SLOW_GUARD and self.st.jobs.get('ask') in self.st.spent_tools:
            return 0
        return self.st.msg_cap - self.st.msg_sent - max(0, self.st.msg_reserve - self.st.gate_sent)

    def should_ask(self) -> bool:
        if self.st.interviewed or not self.st.jobs.get('ask') or self.messages_free() <= 0:
            return False
        if self.net.steps_left() <= 8:
            return False
        opened = len(self.st.full_pages)
        if opened >= 3 * len(self.st.wants) + 1:
            return True
        return self.st.at_step >= max(8, self.st.step_cap // 4)

    def message_call_args(self, text: str) -> dict[str, Any] | None:
        name = self.st.jobs.get('ask')
        schema = self.st.arg_schemas.get(name or '') or {}
        props = schema.get('properties') or {}
        texts = [k for (k, v) in props.items() if isinstance(v, dict) and 'string' in schema_types(v) and ('enum' not in v)]
        target = next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if texts else None)
        if not target:
            return None
        args: dict[str, Any] = {target: text}
        for key in schema.get('required') or []:
            if key not in args:
                spec = props.get(key) or {}
                if isinstance(spec.get('enum'), list) and spec['enum']:
                    args[key] = spec['enum'][0]
                else:
                    return None
        faults: list[str] = []
        args = coerce_to_schema(args, schema, 'args', faults)
        return None if faults else args

    def hold_interview(self) -> str:
        self.st.interviewed = True
        count = self.messages_free()
        if count <= 0:
            return ''
        self.pages.open_side_batch()
        table = self.catalog.field_table()
        settled = []
        for (number, item) in enumerate(self.st.wants, 1):
            settled.append(f'item {number}: {item.label} - settled by the request: ' + ('; '.join(item.rules) or 'nothing'))
        listing = []
        last = None
        for (index, (label, name, carried, values)) in enumerate(table, 1):
            if label != last:
                listing.append(f'category {label!r}:')
                last = label
            listing.append(f'  {index}. {name} (on {carried}): ' + ' | '.join(values))
        prompt = PLAN_PROMPT.replace('COUNT', str(count + 1)) + '\n\nREQUEST:\n' + self.st.ask_text + '\n\nITEMS:\n' + '\n'.join(settled)
        first = [str(x) for x in as_list(self.st.sheet.get('ask_first')) if str(x).strip()]
        if first:
            prompt += '\n\nASK FIRST (the request asks to be asked about these):\n' + '\n'.join(first)
        prompt += '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(listing) or '(no full listing record read yet)')
        plan = (self.net.ask_lean(prompt, [STRONG_MODEL, FAST_MODEL], 1800) if F_R12_LEAN else self.net.ask_structured(prompt, [STRONG_MODEL, FAST_MODEL], 1800)) or {}
        ordered: list[tuple[int, str]] = []
        for pick in as_list(plan.get('ask')):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, whole_number(pick.get('item') or 1) or 1), len(self.st.wants))
            label = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(pick.get('label') or ''))[0].strip().rstrip('?.')
            if not label or len(label.split()) > 5:
                continue
            if all((split_words(label) != split_words(other) for (n, other) in ordered if n == number)):
                ordered.append((number, label))
        self.st.log.append({'kind': 'plan', 'ask': clip_value(plan.get('ask'), 120)})
        if F_R12_ROUND_ASK:
            ordered = self.round_robin(ordered)
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        head = ordered[:count - 1] if count >= 2 else ordered[:count]
        batch = self.send_batch([(number, label, self.question_for(number, label)) for (number, label) in head])
        if not batch:
            return ''
        sent += batch
        heard = self.parse_answers(batch)
        self.store_answers(heard, told)
        if count >= 2 and self.messages_free() > 0:
            crowded = [batch[i - 1] for i in (whole_number(x) for x in as_list(heard.get('crowded'))) if i and 1 <= i <= len(batch)]
            again = None
            for (number, text, _) in crowded:
                label = next((lab for (n, lab) in head if n == number and self.question_for(n, lab) == text), None)
                samples = self.sample_pair(number, label) if label else []
                if label and len(samples) >= 2:
                    again = (number, label, self.question_for(number, label, samples))
                    break
            if F_R12_ROUND_ASK and again is None:
                again = self.unclear_again(batch, heard, head, sent)
            if again is None and len(ordered) > len(head):
                (number, label) = ordered[len(head)]
                again = (number, label, self.question_for(number, label))
            if again:
                batch = self.send_batch([again])
                if batch:
                    sent += batch
                    self.store_answers(self.parse_answers(batch, (FAST_MODEL, STRONG_MODEL)), told)
        if told:
            self.st.epoch += 1
        lines = [f'Q: {q}\nA: {a}' for (_, q, a) in sent]
        summary = "Rok's interviewer questioned the shopper (one step per question):\n" + '\n'.join(lines)
        summary += '\nFirm requirements these answers add: ' + ('; '.join(told) if told else 'none')
        summary += '. Apply them to every pick (searching again with the new values where that is needed).'
        if F_R12_TOLD_SEARCH and told:
            summary += self.pages.told_searches()
        pointed = [need for (_, need) in self.st.pointers if any((need in t for t in told))]
        if pointed:
            summary += ' These get their value from another listing: ' + ' | '.join(pointed) + '. Find that listing and open its full page, then open every listing that page sends you on to; the final page holds the value, and the order only counts once all of them are open.'
        if self.st.pending_notes:
            summary += '\n[notes that came in meanwhile] ' + ' | '.join(self.st.pending_notes)
            self.st.pending_notes = []
        return summary

    def round_robin(self, ordered: list[tuple[int, str]]) -> list[tuple[int, str]]:
        """F_R12_ROUND_ASK: the planned questions with every line asked once before any line is asked again; round
        k holds the k-th question of each line, the lines in the order of their first question."""
        lines: dict[int, list[tuple[int, str]]] = {}
        for pick in ordered:
            lines.setdefault(pick[0], []).append(pick)
        rounds = max((len(picks) for picks in lines.values()), default=0)
        return [picks[k] for k in range(rounds) for picks in lines.values() if k < len(picks)]

    def unclear_again(self, batch: list[tuple[int, str, str]], heard: dict[str, Any], head: list[tuple[int, str]], sent: list[tuple[int, str, str]]) -> tuple[int, str, str] | None:
        """F_R12_ROUND_ASK: after an answer that only says the shopper did not understand, the same property asked
        again with the product named by the store's category label for that line, unless the question already
        named the product that way: the same question draws the same answer, so it is never sent twice."""
        asked = {text for (_, text, _) in sent}
        for index in (whole_number(x) for x in as_list(heard.get('unclear'))):
            if not index or not 1 <= index <= len(batch):
                continue
            (number, text, _) = batch[index - 1]
            label = next((lab for (n, lab) in head if n == number and self.question_for(n, lab) == text), None)
            shelf = self.review.line_category(number)
            if not label or not shelf or ({w for w in bare_stems(shelf) if len(w) >= 4} & bare_stems(text)):
                continue
            again = f'What {label} would you like for the {shelf}?'
            if again not in asked:
                self.st.log.append({'kind': 'unclear_again', 'q': again})
                return (number, label, again)
        return None

    def question_for(self, number: int, label: str, samples: list[str] | None=None) -> str:
        text = f'What {label} would you like for the {self.st.wants[number - 1].label}'
        return text + (f', for example {samples[0]} or {samples[1]}?' if samples else '?')

    def sample_pair(self, number: int, label: str) -> list[str]:
        want = split_words(label)
        settled = ' '.join(self.st.wants[number - 1].rules).lower()
        for (_, name, _, values) in self.catalog.field_table():
            if name == want:
                return [v for v in values if v and v not in settled and (len(v) <= 30)][:2]
        return []

    def send_batch(self, questions: list[tuple[int, str, str]], floor: int=6) -> list[tuple[int, str, str]]:
        tool = self.st.jobs['ask']
        sent: list[tuple[int, str, str]] = []
        for (number, _, text) in questions:
            if self.st.over or self.net.steps_left() <= floor or self.st.msg_sent >= self.st.msg_cap:
                break
            if F_MODEL_RETRY and F_SLOW_GUARD and tool in self.st.spent_tools:
                break
            args = self.message_call_args(text)
            if not args:
                break
            reply = self.net.take_step(tool, args)
            if reply is None:
                break
            self.st.msg_sent += 1
            answer = reply.get('observation')
            words = self.catalog.visible_text(answer)
            sent.append((number, text, words))
            self.st.duty_log.append(f'step {self.st.at_step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {self.review.cart_key()})')
            self.st.duty_steps.append(self.st.at_step)
            self.st.log.append({'kind': 'ask', 'q': text, 'a': words[:300]})
            if F_MODEL_RETRY and F_SLOW_GUARD and allowance_spent(answer):
                self.st.spent_tools.add(tool)
            if isinstance(answer, dict) and answer.get('error'):
                break
        return sent

    def store_answers(self, heard: dict[str, Any], told: list[str]) -> None:
        many = len(self.st.wants) > 1
        for entry in as_list(heard.get('needs')):
            if not isinstance(entry, dict) or not str(entry.get('need') or '').strip():
                continue
            number = min(max(1, whole_number(entry.get('item')) or 1), len(self.st.wants))
            need = str(entry['need']).strip()
            if need not in self.st.wants[number - 1].answered:
                self.st.wants[number - 1].answered.append(need)
                told.append(f'item {number}: {need}' if many else need)
            if entry.get('points') is True and (number, need) not in self.st.pointers:
                self.st.pointers.append((number, need))
        for entry in as_list(heard.get('drop')):
            if isinstance(entry, dict) and str(entry.get('property') or '').strip():
                item = self.st.wants[min(max(1, whole_number(entry.get('item')) or 1), len(self.st.wants)) - 1]
                gone = split_words(str(entry['property']).split(':')[0])
                hit = [m for m in item.rules + item.answered if m not in item.retired and split_words(m.split(':')[0]) == gone]
                item.retired.extend(hit)
                told.extend((f'dropped {m}' for m in hit))
        for entry in as_list(heard.get('free')):
            if isinstance(entry, dict) and entry.get('property'):
                number = min(max(1, whole_number(entry.get('item')) or 1), len(self.st.wants))
                self.st.wants[number - 1].loose.append(str(entry['property']))
        if isinstance(heard.get('budget'), (int, float)) and (not isinstance(heard.get('budget'), bool)) and (heard['budget'] > 0):
            self.st.budget_cap = heard['budget']

    def parse_answers(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(STRONG_MODEL, FAST_MODEL)) -> dict[str, Any]:
        prompt = HEAR_PROMPT + '\n\nITEMS:\n' + '\n'.join((f'item {n}: {it.label}' for (n, it) in enumerate(self.st.wants, 1)))
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f'{i}. (about item {n}) Q: {q}\nA: {a}' for (i, (n, q, a)) in enumerate(sent, 1)))
        return self.net.ask_structured(prompt, list(models), 1500) or {}


class Review(Part):
    """Checks around an ending: the gate for asked-for steps, the cart check, pointer chains, sold-out lines, the
    review of an ending without a sale."""

    def cart_key(self) -> str:
        return to_json_text(sorted(self.st.cart_counts.items()))

    def stage_pending(self, stage: str) -> bool:
        return any((p['when'] == stage for p in self.st.duties)) and (stage, self.cart_key() + f'#{self.st.epoch}') not in self.st.gate_passed

    def kept_steps(self) -> int:
        talk = any((p['when'] == 'order' and p['messages'] > 0 for p in self.st.duties))
        return 2 if talk and self.st.msg_sent < self.st.msg_cap and self.stage_pending('order') else 1

    def pass_gate(self, stage: str) -> tuple[bool, str]:
        steps = [p for p in self.st.duties if p['when'] == stage]
        if not steps or self.st.over or (not self.stage_pending(stage)):
            return (True, '')
        ending = {self.st.jobs.get('buy'), self.st.jobs.get('refuse')} - {None}
        tools = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {to_json_text(t['function'].get('parameters') or {})}" for t in self.st.tool_specs if t['function']['name'] not in ending]
        said: list[str] = []
        nudged = False
        for _ in range(GATE_CALL_CAP):
            prompt = GATE_PROMPT.replace('STAGE', 'place the order' if stage == 'order' else 'end without a purchase') + '\n\nREQUEST:\n' + self.st.ask_text + '\n\nSTEPS DUE NOW:\n' + '\n'.join((f"- {p['step']}" for p in steps)) + '\n\nLEDGER:\n' + self.brief.needs_report() + '\n\nCART NOW: ' + self.cart_report() + '\n\nRECORD (messages and step calls so far):\n' + ('\n'.join((f'{i}. {line}' for (i, line) in enumerate(self.st.duty_log, 1))) or '(nothing yet)') + ('\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.st.store_notes[-4:]) if self.st.store_notes else '') + f"\n\nMESSAGES LEFT: {max(0, self.st.msg_cap - self.st.msg_sent)}   STEPS LEFT: {self.net.steps_left()} (the {('order' if stage == 'order' else 'ending')} itself takes one)" + '\n\nTOOLS:\n' + '\n'.join(tools)
            got = self.net.ask_structured(prompt, [STRONG_MODEL, FAST_MODEL] if nudged else [FAST_MODEL, STRONG_MODEL], 2000)
            if got is None:
                break
            told: list[str] = []
            hold = ''
            if said:
                self.talk.store_answers({'needs': got.get('needs')}, told)
                hold = str(got.get('hold') or '').strip()
            call = got.get('call') if isinstance(got.get('call'), dict) else None
            self.st.log.append({'kind': 'gate', 'stage': stage, 'call': clip_value(call, 200), 'steps': clip_value(got.get('steps'), 120), 'hold': hold[:200], 'told': told})
            if told or hold:
                self.st.epoch += 1
                return (False, ('Before continuing, Rok carried out the steps the request asks for:\n' + '\n'.join(said) + '\n' if said else '') + ("The shopper's reply alters the task: " + '; '.join(told + ([hold] if hold else [])) + '. Rework the cart to suit it, then check out again.'))
            name = str((call or {}).get('tool') or '')
            shown = [e for e in as_list(got.get('steps')) if isinstance(e, dict)]
            talk = stage == 'order' and any((p['messages'] > 0 for p in steps))
            lines = range(1, len(self.st.duty_log) + 1)
            proof = {i for i in lines if not talk or (i <= len(self.st.duty_steps) and self.st.duty_steps[i - 1] > self.st.cart_changed_at)}
            unproven = [e for e in shown if e.get('done') is not True or not any(((whole_number(n) or 0) in proof for n in as_list(e.get('lines'))))]
            if not call and unproven and (not said) and (not nudged) and self.st.jobs.get('ask') and (self.st.msg_sent < self.st.msg_cap) and (self.net.steps_left() >= 2):
                nudged = True
                self.st.log.append({'kind': 'gate_unproven', 'steps': clip_value(unproven, 120)})
                self.st.duty_log.append(f'(Rok: since the cart last changed at step {self.st.cart_changed_at}, no line shows ' + '; '.join((str(e.get('step') or '') for e in unproven))[:200] + f' as done, and {self.st.msg_cap - self.st.msg_sent} message(s) and {self.net.steps_left()} steps remain: give the call that does it)')
                self.st.duty_steps.append(-1)
                continue
            if not call or name not in self.st.arg_schemas or name in ending:
                if not call:
                    self.st.gate_passed.add((stage, self.cart_key() + f'#{self.st.epoch}'))
                break
            if self.net.steps_left() < 2 or (name == self.st.jobs.get('ask') and self.st.msg_sent >= self.st.msg_cap):
                break
            faults: list[str] = []
            args = coerce_to_schema(call.get('args') if isinstance(call.get('args'), dict) else {}, self.st.arg_schemas[name], 'args', faults)
            if faults:
                break
            if name == self.st.jobs.get('ask'):
                text = next((v for v in args.values() if isinstance(v, str)), '')
                batch = self.talk.send_batch([(1, '', text)], 1)
                if not batch:
                    break
                self.st.gate_sent += 1
                said.append(f'Q: {text}\nA: {batch[0][2]}')
                continue
            reply = self.net.take_step(name, args)
            if reply is None:
                break
            seen = reply.get('observation')
            self.catalog.take_in(seen)
            self.st.duty_log.append(f'step {self.st.at_step}: {name} {to_json_text(args)} -> {self.loop.render(seen)[:300]} (cart then: {self.cart_key()})')
            self.st.duty_steps.append(self.st.at_step)
            said.append(f'{name} {to_json_text(args)} -> {self.loop.render(seen)[:400]}')
        return (True, 'Before continuing, Rok carried out the steps the request asks for:\n' + '\n'.join(said) if said else '')

    def needs_in_force(self) -> list[tuple[int, str]]:
        out = []
        for (number, item) in enumerate(self.st.wants, 1):
            for need in item.rules + item.answered:
                if need not in item.retired:
                    out.append((number, need))
        return out

    def proof_lines(self, observation: Any) -> str:
        ids: list[str] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                ident = self.catalog.listing_of(node, False, depth)
                if ident and attribute_block(node) and (ident not in ids):
                    ids.append(ident)
                    return
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1)
        walk(observation, 0)
        needs = self.needs_in_force()
        if not ids or not needs:
            return ''
        many = len(self.st.wants) > 1
        lines = []
        for ident in ids[:12]:
            block = attribute_block(self.st.full_pages.get(ident) or {}) or {}
            parts = []
            for (number, need) in needs[:8]:
                prop = need.split(':', 1)[0].strip()
                hits = [f'{k}={str(v)[:50]}' for (k, v) in block.items() if names_match(word_stems(prop), word_stems(k))]
                tag = f'(item {number}) ' if many else ''
                if hits:
                    parts.append(f'{tag}{prop} -> ' + ', '.join(hits[:3]))
            if parts:
                lines.append(f'{ident}: ' + '; '.join(parts))
        if not lines:
            return ''
        return "[evidence: keys whose names include a requirement's property; a key naming some other property (another part or aspect) decides nothing, while a key under a different name may still hold that same property] " + ' | '.join(lines)

    def cart_report(self) -> str:
        if self.st.cart_seen is not None and self.st.cart_viewed_at >= self.st.cart_changed_at:
            return to_json_text(clip_value(self.st.cart_seen, 300))[:3000]
        return "cart lines as Rok recorded them from its cart calls (listing: units): " + to_json_text(self.st.cart_counts)

    def cart_listings(self) -> list[str]:
        if self.st.cart_seen is not None and self.st.cart_viewed_at >= self.st.cart_changed_at:
            text = to_json_text(self.st.cart_seen)
            return [i for i in self.st.row_index if i in text]
        return list(self.st.cart_counts)

    def record_text(self, ident: str) -> str:
        record = self.st.full_pages.get(ident) or self.st.row_index.get(ident)
        return to_json_text(clip_value(record, 400))[:3000] if record else '(no record seen)'

    def other_records(self, skip: list[str], limit: int=30) -> str:
        lines = []
        order = list(reversed(self.st.first_seen))
        if self.st.sheet.get('pick') == 'cheapest':
            order.sort(key=lambda i: (i not in self.st.full_pages, self.catalog.price_now(i) if self.catalog.price_now(i) is not None else float('inf')))
        for ident in order:
            if ident in skip or self.is_gone(ident):
                continue
            record = self.st.full_pages.get(ident) or self.st.row_index.get(ident)
            lines.append(to_json_text(clip_value(record, 160))[:900])
            if len(lines) >= limit:
                break
        return '\n'.join(lines)

    def verdict_key(self, ids: list[str]) -> str:
        return '|'.join(sorted(ids)) + f'#{self.st.epoch}'

    def judge_cart(self, ids: list[str] | None=None, limit: float=SPEND_WARN) -> str:
        lead = ids is not None
        ids = ids if lead else self.cart_listings()
        key = self.verdict_key(ids)
        if key in self.st.checked_ids or self.st.spend > limit or (not ids):
            return ''
        self.st.checked_ids.add(key)
        prompt = CHECK_PROMPT + '\n\nREQUEST:\n' + self.st.ask_text + '\n\nREQUIREMENTS IN FORCE:\n' + self.brief.needs_report()
        if F_R12_UNSEEN_WORD:
            prompt += self.unseen_note()
        if lead:
            quantity = '; '.join((f'item {n}: quantity {it.units}' for (n, it) in enumerate(self.st.wants, 1)))
            prompt += '\n\nCART (planned, judge these listings as if they were carted at the requested quantities: ' + quantity + '):\n' + ', '.join(ids)
        else:
            prompt += '\n\nCART:\n' + self.cart_report()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + '\n'.join((f'{i}: ' + ('SOLD OUT (the store says it can no longer be bought). ' if self.is_gone(i) else '') + self.record_text(i) for i in ids))
        if self.st.sheet.get('pick') in ('cheapest', 'most', 'least'):
            prompt += '\n\nOTHER LISTINGS SEEN (newest first):\n' + self.other_records(ids, 15)
        got = self.net.ask_lean(prompt, [FAST_MODEL, STRONG_MODEL], FAST_JUDGE_ROOM) if F_R12_FAST_JUDGE else self.net.ask_lean(prompt, [STRONG_MODEL, FAST_MODEL], 1500) if F_R12_LEAN else self.net.ask_structured(prompt, [STRONG_MODEL, FAST_MODEL], 1500)
        self.st.log.append({'kind': 'check', 'cart': ids, 'got': got})
        self.st.check_verdicts[key] = bool(got) and got.get('ok') is not False
        if not got or got.get('ok') is not False:
            return ''
        self.st.hints = [i for i in self.st.hints if i not in ids]
        problems = [str(p) for p in as_list(got.get('problems'))][:5]
        better = [str(b) for b in as_list(got.get('better')) if str(b) in self.st.row_index and (not self.is_gone(str(b)))][:4]
        text = 'An independent review finds this cart not right yet: ' + '; '.join(problems or ['check the requirements'])
        if better:
            text += '. Listings that look like better picks: ' + ', '.join(better)
        return text + '. Repair the cart (or prove the review wrong by reading the records again), then check out.'

    def walk_chains(self) -> tuple[str, bool]:
        live = {need for (_, need) in self.needs_in_force()}
        pending = [need for (_, need) in self.st.pointers if need in live and need not in self.st.pointer_values]
        tool = self.st.jobs.get('offer') or self.st.jobs.get('open')
        target = self.catalog.lone_id_param(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.st.chain_runs < 4 and (self.st.spend < SPEND_CAP):
            self.st.chain_runs += 1
            records = []
            for ident in reversed(self.st.first_seen):
                record = self.st.full_pages.get(ident) or self.st.row_index.get(ident) if ident in self.st.opened_ids else None
                if record:
                    records.append('OPENED ' + to_json_text(clip_value(record, 300))[:1400])
                else:
                    records.append('ROW ' + to_json_text(clip_value(self.st.row_index.get(ident), 80))[:300])
                if len(records) >= 60:
                    break
            prompt = TRAIL_PROMPT + '\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n' + '\n'.join(pending)
            prompt += '\n\nRECORDS (newest first):\n' + '\n'.join(records)
            got = self.net.ask_structured(prompt, [FAST_MODEL, STRONG_MODEL], 1800) or {}
            self.st.log.append({'kind': 'trail', 'pending': pending, 'got': clip_value(got, 150)})
            reach: list[str] = []
            lost = []
            for chain in as_list(got.get('chains')):
                if not isinstance(chain, dict):
                    continue
                need = next((p for p in pending if p == str(chain.get('need') or '').strip()), None) or (pending[0] if len(pending) == 1 else None)
                if not need:
                    continue
                links = [l for l in as_list(chain.get('links')) if isinstance(l, dict)]
                ids = [str(l.get('id') or '').strip() for l in links]
                if chain.get('done') is True and links and all((i in self.st.opened_ids for i in ids)) and str(chain.get('value') or '').strip():
                    self.st.pointer_values[need] = str(chain['value']).strip()
                    self.st.epoch += 1
                    continue
                closed = [i for i in ids if i in self.st.row_index and i not in self.st.opened_ids]
                if closed:
                    reach += [i for i in closed if i not in reach]
                else:
                    about = next((str(l.get('about') or '') for (l, i) in zip(links, ids) if i not in self.st.row_index), '') or 'the linked listing'
                    lost.append(f'{need} -> find {about}')
            pending = [p for p in pending if p not in self.st.pointer_values]
            if not reach or not tool or (not target) or (self.net.steps_left() < len(reach) + 4) or self.st.over:
                break
            for ident in reach[:max(0, min(3, 6 - len(seen)))]:
                faults: list[str] = []
                args = coerce_to_schema({target: ident}, self.st.arg_schemas.get(tool), 'args', faults)
                if faults:
                    break
                reply = self.net.take_step(tool, args)
                if reply is None or self.st.over:
                    break
                self.pages.take_own_result(tool, reply.get('observation'))
                seen.append(f'{ident}: ' + self.loop.render(reply.get('observation'))[:1500])
        text = ''
        if seen:
            text += '[Rok opened the pages that requirements point to]\n' + '\n'.join(seen) + '\n'
        if self.st.pointer_values:
            text += '[values taken from the pages requirements point to] ' + '; '.join((f'{k} -> {v}' for (k, v) in self.st.pointer_values.items())) + '\n'
        hold = bool(lost) and self.st.chain_holds < 2 and (self.net.steps_left() > 5)
        if hold:
            self.st.chain_holds += 1
            text += 'The order only counts once every page that a requirement points to has been opened in full (a search row will not do), following on wherever a page refers to a further listing. Still to find and open: ' + ' | '.join(lost[:3]) + '. Look it up by its shop, brand or words, open it, use the value its page states, then check out.'
        return (text, hold)

    def label_fits(self, label: str, want: Want) -> bool:
        """The matching of ids_by_category: a category label that names a requested line."""
        long = {w for w in word_stems(want.family or '') | word_stems(want.label) if len(w) >= 4}
        return split_words(label) in (split_words(want.family or ''), split_words(want.label)) or bool({w for w in word_stems(label) if len(w) >= 4} & long)

    def wanted_kind(self, ident: str) -> bool:
        cat_key = self.catalog.category_field()
        label = str((self.st.row_index.get(ident) or {}).get(cat_key) or '') if cat_key else ''
        return bool(label) and any((self.label_fits(label, want) for want in self.st.wants))

    def line_category(self, number: int) -> str | None:
        """The category label that most rows seen for line number carry, in lower case."""
        cat_key = self.catalog.category_field()
        want = self.st.wants[number - 1]
        counts: dict[str, int] = {}
        for row in self.st.row_index.values():
            label = str(row.get(cat_key) or '').strip() if cat_key else ''
            if label and self.label_fits(label, want):
                counts[label] = counts.get(label, 0) + 1
        return max(counts, key=lambda k: counts[k]).lower() if counts else None

    def line_noun(self, number: int) -> str:
        """A line's product noun for a search phrase: its name in the request's words, at most three words."""
        return ' '.join(' '.join(str(self.st.wants[number - 1].label).lower().split()).split()[-3:])

    def line_values(self, number: int) -> list[str]:
        """The values in force for a line, told ones first, then the request's, leaving out values that point at
        another listing: those of at most four words that some listing seen carries word for word in a text field
        or an attribute."""
        want = self.st.wants[number - 1]
        pointed = {need for (_, need) in self.st.pointers}
        texts = []
        for row in self.st.row_index.values():
            for value in list(row.values()) + list((attribute_block(row) or {}).values()):
                if isinstance(value, str):
                    texts.append(flat_text(value))
        pool = ' | '.join(texts)
        out: list[str] = []
        for need in want.answered + want.rules:
            if need in want.retired or need in pointed:
                continue
            raw = ' '.join((need.partition(':')[2] or need).lower().split())
            value = flat_text(raw)
            if value and raw not in out and len(raw.split()) <= 4 and re.search('(^|[^a-z0-9])' + re.escape(value) + '($|[^a-z0-9])', pool):
                out.append(raw)
        return out

    def told_words(self) -> set[str]:
        """Word stems of the values the shopper told in replies that are still in force (not the request's own
        values), three letters or more, the product names of the lines left out."""
        nouns = {w for want in self.st.wants for w in bare_stems(want.label) | bare_stems(want.family or '')}
        found: set[str] = set()
        for want in self.st.wants:
            for need in want.answered:
                if need not in want.retired:
                    found |= {w for w in bare_stems(need.partition(':')[2] or need) if len(w) >= 3}
        return found - nouns

    def unseen_words(self) -> list[str]:
        """F_R12_UNSEEN_WORD: words of the requirements in force, four letters or more, that two different searches
        named (different sets of words; a repeat or a further page of one search counts once) and that no listing
        of a requested category seen so far carries, in a text field of at most 150 characters (its title, seller,
        category) or in an attribute key or value. A requirement whose value holds a number (a measure, a size, a
        model number) names a property listings are sold by, carried under spellings the searches may not have met
        yet, so none of its words count."""
        listings = [row for (ident, row) in self.st.row_index.items() if self.wanted_kind(ident)]
        if not listings:
            return []
        pool: set[str] = set()
        for row in listings:
            for (key, value) in list(row.items()) + list((attribute_block(row) or {}).items()):
                if isinstance(value, (str, int, float)) and (not isinstance(value, bool)) and len(str(value)) <= 150:
                    pool |= bare_stems(key) | bare_stems(value)
            for key in attribute_block(row) or {}:
                pool |= bare_stems(key)
        searched: dict[str, int] = {}
        for bag in {frozenset(bare_stems(phrase)) for (phrase, _, _) in self.st.searches}:
            for word in bag:
                searched[word] = searched.get(word, 0) + 1
        words: list[str] = []
        for (_, need) in self.needs_in_force():
            if re.search('[0-9]', need.partition(':')[2] or need):
                continue
            for word in sorted(bare_stems(need.partition(':')[2] or need)):
                if len(word) >= 4 and word not in pool and searched.get(word, 0) >= 2 and word not in words:
                    words.append(word)
        return words

    def unseen_note(self) -> str:
        words = self.unseen_words()
        if not words:
            return ''
        self.st.log.append({'kind': 'unseen_words', 'words': words})
        return '\n\nWORDS NO LISTING OF THE REQUESTED KINDS CARRIES: ' + ', '.join(words) + '. Different searches named them, yet no title or attribute of any listing of those categories seen so far has them. A word of a requirement that no listing of the kind carries names no property these listings are sold by, so it is not an unmet requirement: judge the listings on the remaining requirements.'

    def retired_counts(self) -> tuple[int, ...]:
        return tuple((len(want.retired) for want in self.st.wants))

    def dropped_line(self) -> int:
        """The number of the first line that lost a requirement since retired_before was taken, else 0."""
        before = self.st.retired_before
        for (number, count) in enumerate(self.retired_counts(), 1):
            if count > (before[number - 1] if number - 1 < len(before) else 0):
                return number
        return 0

    def drop_update(self) -> str:
        """F_R12_DROP_UPDATE, once: a refused ending without a sale took a requirement away, which updates the
        requirements rather than ending the task. The review of the listings read runs again under the
        requirements now in force; when the check approves its first lead, Rok carts it and checks out. It needs a
        cart call and the order (cart_confirmed counts its own calls against the steps left), so it runs while two
        steps are left: a store refusal comes late, often with four or five steps left."""
        if self.st.drop_updates or self.net.steps_left() < 2 or (not self.net.time_fits(3)) or self.st.over:
            return ''
        self.st.drop_updates += 1
        (self.st.hints, self.st.confirmed_hint, self.st.auto_carted, self.st.decline_holds) = ([], None, False, 0)
        got = self.audit_request()
        self.st.reviewed = True
        named = self.audit_matches(got)
        self.st.log.append({'kind': 'drop_update', 'named': named})
        if not named:
            return ''
        self.st.hint_items = named
        self.st.hints = [i for ids in named.values() for i in ids]
        first = self.st.hints[0]
        self.judge_cart([first], REVIEW_SPEND_CAP)
        if self.st.check_verdicts.get(self.verdict_key([first])):
            self.st.confirmed_hint = first
            done = self.fin.cart_confirmed()
            if done:
                return done + (' [episode finished]' if self.st.over else '')
        return 'The store turned the ending down and took a requirement away, so the task changed rather than ended. Under the requirements now in force the review names these leads: ' + ', '.join(self.st.hints) + '. Open them, get the best fit into the cart and check out.'

    def chain_roots(self) -> list[tuple[str, list[str]]]:
        """F_R12_CHAIN_GATE: for each requirement in force whose value the shopper takes from another listing, the
        listings it most likely means, found in code. Candidates are the listings seen whose category label words
        and one short field value (a seller or a brand) the text names. Each part of the text after that value, up
        to a bracket and split at commas, semicolons, '&' and 'and', is tested: a part that starts with the words of
        one of the candidate's attribute keys supports it when the rest of the part appears in that attribute's
        value and contradicts it otherwise; a part naming no key supports it when its last words (a number with its
        unit, or one longer word) appear in the candidate's text. A contradicted candidate drops out; those with the
        most support stay (all of them on a tie), and a lone candidate stays without support."""
        cat_key = self.catalog.category_field()
        live = {need for (_, need) in self.needs_in_force()}
        out: list[tuple[str, list[str]]] = []

        def key_value(key: str, part: str) -> str | None:
            kw = split_words(key).split()
            pw = re.findall('[a-z]+|[0-9]+(?:\\.[0-9]+)?', part)
            if not kw or len(kw[0]) < 3:
                return None
            for start in range(len(pw)):
                if pw[start] != kw[0]:
                    continue
                n = 1
                while n < len(kw) and start + n < len(pw) and (pw[start + n] == kw[n]):
                    n += 1
                if n == len(kw) or len(kw[0]) >= 4:
                    return ' '.join(pw[start + n:]) or None
            return None
        for need in dict.fromkeys((need for (_, need) in self.st.pointers if need in live)):
            text = ' '.join((need.partition(':')[2] or need).lower().split())
            words = bare_stems(text)
            scored: list[tuple[int, str]] = []
            for (ident, row) in self.st.row_index.items():
                label = str(row.get(cat_key) or '') if cat_key else ''
                long = {w for w in bare_stems(label) if len(w) >= 4}
                if not long or not long <= words:
                    continue
                spot = -1
                for (key, value) in row.items():
                    if key == cat_key or not isinstance(value, str) or not 0 < len(value) <= 30 or len(value.split()) > 3 or looks_like_id(value):
                        continue
                    found = re.search('(^|[^a-z0-9])' + re.escape(' '.join(value.lower().split())) + '($|[^a-z0-9])', text)
                    if found:
                        spot = max(spot, found.end())
                if spot < 0:
                    continue
                parts = [p.strip() for p in re.split('[,;&]|\\band\\b', text[spot:].split('(')[0]) if p.strip()]
                if not parts:
                    continue
                facts = attribute_block(self.st.full_pages.get(ident) or row) or {}
                body = ' '.join((v for v in row.values() if isinstance(v, str)))
                (support, against) = (0, 0)
                for part in parts:
                    said = [(k, v) for k in facts for v in [key_value(k, part)] if v]
                    if said:
                        if any((word_run(v, facts[k]) for (k, v) in said)):
                            support += 1
                        else:
                            against += 1
                        continue
                    tokens = re.findall('[a-z]+|[0-9]+(?:\\.[0-9]+)?', part)
                    tail = ' '.join(tokens[-2:]) if len(tokens) >= 2 and re.fullmatch('[0-9.]+', tokens[-2]) else (tokens[-1] if tokens and len(tokens[-1]) >= 4 else '')
                    if tail and word_run(tail, body):
                        support += 1
                if not against:
                    scored.append((support, ident))
            if not scored:
                continue
            best = max((s for (s, _) in scored))
            tops = [i for (s, i) in scored if s == best] if best > 0 else ([scored[0][1]] if len(scored) == 1 else [])
            if tops:
                out.append((need, tops))
        return out

    def chain_gate(self) -> str:
        """F_R12_CHAIN_GATE, at checkout: for every requirement whose value comes from another listing, each of the
        top listings that chain_roots finds for it (all of them on a tie) must have shown its full page through a
        view that returns more than a row (opened_ids, see Catalog.note_view). Those that have not are opened now
        with that view, at most three per requirement and GATE_VIEW_CAP in an episode, while the steps for them, the
        kept steps and two more remain. Independent of walk_chains and its limits."""
        if self.st.over or self.st.gate_views >= GATE_VIEW_CAP:
            return ''
        tool = next((t for t in sorted(self.st.remark_tools) if self.catalog.lone_id_param(t)), None) or self.live_offer_view()
        target = self.catalog.lone_id_param(tool)
        if not tool or not target:
            return ''
        missing: list[str] = []
        for (_, tops) in self.chain_roots():
            missing += [i for i in tops if i not in self.st.opened_ids and i not in missing and (not self.is_gone(i))][:3]
        missing = missing[:GATE_VIEW_CAP - self.st.gate_views]
        if not missing or self.net.steps_left() < len(missing) + self.kept_steps() + 2:
            return ''
        seen = []
        for ident in missing:
            faults: list[str] = []
            args = coerce_to_schema({target: ident}, self.st.arg_schemas.get(tool), 'args', faults)
            if faults:
                break
            reply = self.net.take_step(tool, args)
            if reply is None or self.st.over:
                break
            self.st.gate_views += 1
            self.pages.take_own_result(tool, reply.get('observation'))
            seen.append(f'{ident}: ' + self.loop.render(reply.get('observation'))[:1500])
        if seen:
            self.st.log.append({'kind': 'chain_gate', 'ids': missing[:len(seen)]})
        return '\n'.join(seen)

    def live_offer_view(self) -> str | None:
        return self.st.jobs.get('offer') or self.st.jobs.get('open')

    def reread_cart(self) -> str:
        tool = self.live_offer_view()
        target = self.catalog.lone_id_param(tool)
        if not tool or not target:
            return ''
        since = max(self.st.at_step - 2, self.st.buy_refused_at)
        stale = [i for i in self.cart_listings() if self.st.offer_read_at.get(i, -99) < since and i not in self.st.sold_at]
        if not stale or self.net.steps_left() < len(stale) + 3:
            return ''
        lines = []
        for ident in stale:
            faults: list[str] = []
            args = coerce_to_schema({target: ident}, self.st.arg_schemas.get(tool), 'args', faults)
            if faults:
                break
            before = dict(self.st.row_index.get(ident) or {})
            reply = self.net.take_step(tool, args)
            if reply is None or self.st.over:
                break
            observation = reply.get('observation')
            self.st.offer_read_at[ident] = self.st.at_step
            self.pages.take_own_result(tool, observation)
            after = self.st.row_index.get(ident) or {}
            moved = [k for (k, v) in after.items() if k in before and before[k] != v and (not isinstance(v, (dict, list)))]
            if moved:
                lines.append(f"{ident} changed ({', '.join(moved[:4])}): {to_json_text(clip_value(observation, 300))[:900]}")
        self.st.log.append({'kind': 'reread', 'ids': stale, 'changed': len(lines)})
        if lines:
            self.st.epoch += 1
        return '\n'.join(lines)

    def is_gone(self, ident: str) -> bool:
        return ident in self.st.off_sale or ident in self.st.sold_at

    def replacement_hint(self) -> str:
        lost = [i for i in self.cart_listings() if i not in self.st.sold_at and self.is_gone(i)]
        if not lost:
            return ''
        for ident in lost:
            self.st.sold_at[ident] = self.st.at_step
        self.st.hints = [i for i in self.st.hints if i not in lost]
        self.st.epoch += 1
        picks = self.replacement_ids(lost)
        text = f"[Rok] Now sold out: {', '.join(lost)} (the store shows it can no longer be bought). It will not return, so do not open it again."
        if picks and self.net.steps_left() <= 6:
            return text + f" Only a few steps remain: in this turn add {picks[0]} at the requested quantity and take out the sold-out line (two calls), then check out. Other listings you have read that seem to fit: {', '.join(picks[1:]) or 'none'}."
        if picks and self.st.sheet.get('pick') == 'cheapest':
            return text + f" Listings you have read that seem to fit that item instead, lowest price first: {', '.join(picks)}. The pick rule is the cheapest, so unless you have seen the whole fitting set already, run one search below {picks[0]}'s price (with the search's maximum price, if it has one) before swapping the sold-out line for the cheapest fitting listing."
        if picks:
            return text + f" Listings you have read that seem to fit that item instead: {', '.join(picks)}. In this turn remove the sold-out line from the cart and add {picks[0]} at the requested quantity, then check out."
        return text + ' Take that line out of the cart and add the best other fitting listing; search for one if none of those you read fits.'

    def find_sold_line(self) -> str:
        ids = [i for i in self.cart_listings() if i not in self.st.sold_at]
        if any((self.is_gone(i) for i in ids)):
            return ''
        if len(ids) == 1:
            self.st.off_sale.add(ids[0])
            return ''
        return self.reread_cart() if ids else ''

    def replacement_ids(self, lost: list[str]) -> list[str]:
        if self.st.spend > SPEND_WARN:
            return []
        records = []
        for ident in reversed(self.st.first_seen):
            if ident in lost or ident in self.st.sold_at or self.is_gone(ident):
                continue
            records.append(to_json_text(clip_value(self.st.full_pages.get(ident) or self.st.row_index.get(ident), 200))[:1200])
            if len(records) >= 25:
                break
        if not records:
            return []
        prompt = SWAP_PROMPT + '\n\nREQUEST:\n' + self.st.ask_text + '\n\nREQUIREMENTS IN FORCE:\n' + self.brief.needs_report()
        prompt += '\n\nSOLD-OUT CART LINE:\n' + '\n'.join((f'{i}: {self.record_text(i)}' for i in lost)) + '\n\nCART NOW:\n' + self.cart_report()
        prompt += '\n\nOTHER LISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
        got = self.net.ask_structured(prompt, [FAST_MODEL, STRONG_MODEL], FAST_AUDIT_ROOM) if F_R12_FAST_JUDGE else self.net.ask_structured(prompt, [STRONG_MODEL, FAST_MODEL], 1200)
        fits: list[tuple[float, str]] = []
        for entry in as_list((got or {}).get('fits')):
            ident = str(entry.get('id') if isinstance(entry, dict) else entry)
            price = entry.get('price') if isinstance(entry, dict) else None
            if ident in self.st.row_index and ident not in lost and (ident not in self.st.sold_at) and (ident not in self.cart_listings()) and (ident not in [i for (_, i) in fits]):
                fits.append((float(price) if isinstance(price, (int, float)) and (not isinstance(price, bool)) else float('inf'), ident))
        ranked = [i for (_, i) in sorted(fits, key=lambda f: f[0])] if self.st.sheet.get('pick') == 'cheapest' else [i for (_, i) in fits]
        self.st.log.append({'kind': 'swap', 'lost': lost, 'fits': fits})
        return ranked[:4]

    def audit_decline(self) -> str:
        if F_DECLINE_REVIEW and F_SLOW_GUARD:
            if self.st.reviewed or self.st.spend > REVIEW_SPEND_CAP or self.net.steps_left() <= 5 or (not self.net.time_fits(2)):
                return ''
        elif self.st.reviewed or self.st.spend > SPEND_WARN:
            return ''
        self.st.reviewed = True
        got = self.audit_request()
        named = self.audit_matches(got)
        if not named:
            rows = [str(i) for i in as_list((got or {}).get('open')) if str(i) in self.st.row_index and str(i) not in self.st.opened_ids and (not self.is_gone(str(i)))]
            if rows and self.fin.open_row_pages(rows):
                got = self.audit_request()
                named = self.audit_matches(got)
        if not named:
            return ''
        self.st.hint_items = named
        self.st.hints = [i for ids in named.values() for i in ids]
        return 'An independent review thinks these listings might fit: ' + '; '.join((f"item {k}: {', '.join(v)}" for (k, v) in named.items())) + (f" ({got.get('reason')})" if got.get('reason') else '') + '. Open them and buy whatever genuinely fits; end with no purchase only if every one of them fails.'

    def opened_first(self) -> list[str]:
        """The listings seen, newest first by first sight, with every listing whose page was read moved ahead of
        the rows only seen in result lists (so that a page read late, of a row seen early, stays in the window)."""
        seen = list(reversed(self.st.first_seen))
        return [i for i in seen if self.st.full_pages.get(i)] + [i for i in seen if not self.st.full_pages.get(i)]

    def audit_request(self) -> dict[str, Any] | None:
        records = []
        for ident in (self.opened_first() if F_DECLINE_REVIEW and F_SLOW_GUARD else reversed(self.st.first_seen)):
            if self.is_gone(ident):
                continue
            record = self.st.full_pages.get(ident) if ident in self.st.full_pages else None
            records.append('OPENED ' + to_json_text(clip_value(record, 200))[:1200] if record else 'ROW ' + to_json_text(clip_value(self.st.row_index.get(ident), 120))[:500])
            if len(records) >= 30:
                break
        if not records:
            return None
        prompt = AUDIT_PROMPT + '\n\nREQUEST:\n' + self.st.ask_text + '\n\nREQUIREMENTS IN FORCE:\n' + self.brief.needs_report()
        if F_R12_UNSEEN_WORD:
            prompt += self.unseen_note()
        prompt += ('\n\nLISTING RECORDS SEEN (listings whose page was read come first, then rows from result lists; newest first within each part):\n' if F_DECLINE_REVIEW and F_SLOW_GUARD else '\n\nLISTING RECORDS SEEN (newest first):\n') + '\n'.join(records)
        got = self.net.ask_lean(prompt, [FAST_MODEL, STRONG_MODEL], max(FAST_AUDIT_ROOM, AUDIT_ROOM)) if F_R12_FAST_JUDGE else self.net.ask_lean(prompt, [STRONG_MODEL, FAST_MODEL], AUDIT_ROOM if F_AUDIT_ROOM and F_DECLINE_REVIEW and F_SLOW_GUARD else 1500) if F_R12_LEAN else self.net.ask_structured(prompt, [STRONG_MODEL, FAST_MODEL], AUDIT_ROOM if F_AUDIT_ROOM and F_DECLINE_REVIEW and F_SLOW_GUARD else 1500)
        self.st.log.append({'kind': 'audit', 'got': got})
        return got

    def audit_matches(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get('fits')
        if not isinstance(fits, dict):
            return {}
        named = {str(k): [str(i) for i in as_list(v) if str(i) in self.st.row_index and (not self.is_gone(str(i)))][:4] for (k, v) in fits.items()}
        return {k: v for (k, v) in named.items() if v}


class Finisher(Part):
    """Which tools the loop is shown, the recognition and filling of ending calls, and the forced ending."""

    def open_row_pages(self, rows: list[str]) -> bool:
        (many, joiner) = (self.loop.side_param(), self.st.jobs.get('compare'))
        tool = self.catalog.facts_view()
        target = self.catalog.lone_id_param(tool)
        if many and joiner and (not self.catalog.side_view_bare()):
            plan = [(joiner, {many: rows[:self.loop.side_room()]})]
        elif tool and target:
            plan = [(tool, {target: i}) for i in rows[:2]]
        else:
            return False
        if self.net.steps_left() < len(plan) + 5:
            return False
        before = len(self.st.full_pages)
        for (name, args) in plan:
            faults: list[str] = []
            args = coerce_to_schema(args, self.st.arg_schemas.get(name), 'args', faults)
            if faults:
                return False
            reply = self.net.take_step(name, args)
            if reply is None or self.st.over:
                break
            self.pages.take_own_result(name, reply.get('observation'))
        self.st.log.append({'kind': 'audit_open', 'rows': rows, 'opened': len(self.st.full_pages) - before})
        return len(self.st.full_pages) > before

    def cart_call_args(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        name = self.st.jobs.get('cart') or ''
        schema = self.st.arg_schemas.get(name) or {}
        props = schema.get('properties') or {}
        for (key, spec) in props.items():
            shape = spec.get('items') if isinstance(spec, dict) and 'array' in schema_types(spec) else None
            if isinstance(shape, dict) and isinstance(shape.get('properties'), dict):
                inner = shape['properties']
                id_key = next((k for (k, s) in inner.items() if isinstance(s, dict) and 'string' in schema_types(s) and ('enum' not in s)), None)
                count_key = next((k for (k, s) in inner.items() if isinstance(s, dict) and 'integer' in schema_types(s)), None)
                if not id_key:
                    return None
                return [{key: [{id_key: i, **({count_key: n} if count_key else {})} for (i, n) in lines.items()]}]
        texts = [k for (k, s) in props.items() if isinstance(s, dict) and 'string' in schema_types(s) and ('enum' not in s)]
        target = next((k for (t, k) in self.st.id_params if t == name and k in texts), None) or next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        count = next((k for (k, s) in props.items() if isinstance(s, dict) and 'integer' in schema_types(s)), None)
        mode = next((k for (k, s) in props.items() if isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) >= 2)), None)
        if not target:
            return None
        calls: list[dict[str, Any]] = []
        for ident in drop:
            have = self.st.cart_counts.get(ident, 1)
            if mode:
                calls.append({target: ident, mode: props[mode]['enum'][1], **({count: have} if count else {})})
            elif count and self.cart_sets_quantity(name):
                calls.append({target: ident, count: 0})
            elif count and isinstance(props[count].get('minimum'), int) and (props[count]['minimum'] <= -have):
                calls.append({target: ident, count: -have})
            else:
                return None
        for (ident, want) in lines.items():
            have = self.st.cart_counts.get(ident, 0) if ident in self.review.cart_listings() else 0
            if have == want:
                continue
            if have and (not self.cart_sets_quantity(name)):
                return None
            calls.append({target: ident, **({mode: props[mode]['enum'][0]} if mode else {}), **({count: want} if count else {})})
        return calls

    def cart_confirmed(self) -> str:
        (cart, buy) = (self.st.jobs.get('cart'), self.st.jobs.get('buy'))
        if self.st.auto_carted or not self.st.confirmed_hint or self.review.is_gone(self.st.confirmed_hint) or (not cart) or (not buy):
            return ''
        self.st.auto_carted = True
        cat_key = self.catalog.category_field()

        def kind(ident: str) -> str:
            return str((self.st.row_index.get(ident) or {}).get(cat_key) or '').strip().lower() if cat_key else ident
        plan: dict[str, int] = {}
        for (number, ids) in self.st.hint_items.items():
            index = whole_number(number)
            pick = next((i for i in ids if not self.review.is_gone(i)), None)
            if pick and index and (1 <= index <= len(self.st.wants)) and (kind(pick) not in {kind(p) for p in plan}):
                plan[pick] = self.st.wants[index - 1].units
        if self.st.confirmed_hint not in plan:
            return ''
        now = self.review.cart_listings()
        keep = {i: self.st.cart_counts.get(i, 1) for i in now if i not in plan and (not self.review.is_gone(i)) and (kind(i) not in {kind(p) for p in plan})}
        drop = [i for i in now if i not in plan and i not in keep]
        if len(plan) + len({kind(i) for i in keep}) < len(self.st.wants):
            return ''
        lines = {**keep, **plan}
        calls = self.cart_call_args(lines, drop)
        if not calls or len(calls) + 1 > self.net.steps_left():
            return ''
        if F_R12_READ_LINES:
            self.pages.read_unread_lines(list(lines), len(calls) + 1)
        self.pages.read_missing_facts(list(lines), len(calls) + 3)
        self.review.judge_cart(list(lines), REVIEW_SPEND_CAP if F_DECLINE_REVIEW and F_SLOW_GUARD else SPEND_CAP)
        if not self.st.check_verdicts.get(self.review.verdict_key(list(lines))):
            return ''
        done = []
        for args in calls:
            faults: list[str] = []
            args = coerce_to_schema(args, self.st.arg_schemas.get(cart), 'args', faults)
            if faults:
                return ''
            reply = self.net.take_step(cart, args)
            if reply is None or self.st.over:
                return ''
            self.loop.track_cart_call(cart, args)
            self.pages.take_own_result(cart, reply.get('observation'))
            done.append(to_json_text(args))
        faults = []
        args = coerce_to_schema(dict(self.st.sheet.get('buy_args') or {}), self.st.arg_schemas.get(buy), 'args', faults)
        if faults:
            return ''
        if F_R12_CHAIN_GATE:
            self.review.chain_gate()
        replies = []
        for attempt in range(2):
            if attempt:
                self.review.reread_cart()
                self.review.judge_cart(list(lines), REVIEW_SPEND_CAP if F_DECLINE_REVIEW and F_SLOW_GUARD else SPEND_CAP)
                if not self.st.check_verdicts.get(self.review.verdict_key(list(lines))) or self.net.steps_left() < 1 or self.st.over or any((self.review.is_gone(i) for i in lines)):
                    break
            (go, said) = self.review.pass_gate('order')
            if said:
                replies.append(said)
            if not go:
                break
            reply = self.net.take_step(buy, args)
            self.st.log.append({'kind': 'completed', 'lines': lines, 'drop': drop, 'reply': clip_value(reply, 200)})
            if reply is None:
                break
            self.st.buy_tries += 1
            if not reply.get('finished'):
                self.st.buy_refused_at = self.st.at_step
            self.pages.take_own_result(buy, reply.get('observation'))
            replies.append(self.loop.render(reply.get('observation')))
            if self.st.over or reply.get('finished'):
                break
        if not replies:
            return ''
        return 'Not sent (Rok acted instead): the independent check approved ' + self.st.confirmed_hint + ', so Rok set up the cart (' + '; '.join(done) + ') and checked out. The store replied: ' + ' | then: '.join(replies)

    def is_checkout(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.st.jobs.get('buy'):
            return False
        fixed = self.st.sheet.get('buy_args') if isinstance(self.st.sheet.get('buy_args'), dict) else {}
        return all((args.get(k) == v for (k, v) in fixed.items()))

    def is_decline(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.st.jobs.get('refuse'):
            return False
        fixed = self.st.sheet.get('refuse_args') if isinstance(self.st.sheet.get('refuse_args'), dict) else {}
        return all((args.get(k) == v for (k, v) in fixed.items()))

    def visible_tools(self) -> list[dict[str, Any]]:
        left = self.net.steps_left()
        ask = self.st.jobs.get('ask')
        keep: set[str] | None = None
        ends = {self.st.jobs.get('buy'), self.st.jobs.get('refuse')} - {None}
        if left <= self.review.kept_steps():
            keep = set(ends)
        elif left <= self.review.kept_steps() + 3:
            keep = set(ends) | {self.st.jobs.get('cart'), self.st.jobs.get('cart_view')}
        elif left <= len(self.st.wants) + 5 and self.st.buy_tries == 0:
            keep = set(ends) | {self.st.jobs.get('cart'), self.st.jobs.get('cart_view'), self.st.jobs.get('offer')}
        tools = [t for t in self.st.tool_specs if t['function']['name'] != ask]
        if keep is not None and ends:
            narrowed = [t for t in tools if t['function']['name'] in keep]
            tools = narrowed or tools
        return tools

    def decline_text_args(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        schema = self.st.arg_schemas.get(name) or {}
        props = schema.get('properties') or {}
        out = dict(args)
        for (key, spec) in props.items():
            if not isinstance(spec, dict) or 'string' not in schema_types(spec) or 'enum' in spec:
                continue
            value = out.get(key)
            if value is None and key not in (schema.get('required') or []):
                continue
            text = str(value or '')
            named = any((ident in text for ident in self.st.row_index)) or self.st.sheet.get('quiet_decline') is not False
            if named or not text:
                out[key] = 'Sorry, nothing in the store meets all of your requirements, so I am not buying anything.'[:spec.get('maxLength', 200)]
        return out

    def resolve_listing_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if name not in {self.st.jobs.get(r) for r in ('cart', 'open', 'offer', 'compare')} - {None} or not self.st.row_id_key:
            return (args, '')
        props = (self.st.arg_schemas.get(name) or {}).get('properties') or {}
        free = [k for (k, v) in props.items() if isinstance(v, dict) and ('string' in schema_types(v) or 'array' in schema_types(v)) and ('enum' not in v)]
        if len(free) != 1 or (name != self.st.jobs.get('cart') and (name, free[0]) not in self.st.id_params):
            return (args, '')

        def fix(value: Any) -> tuple[Any, str]:
            if isinstance(value, str) and value.strip() in self.st.row_index:
                return (value.strip(), '')
            if not isinstance(value, str) or looks_like_id(value, True) or len(value.strip()) < 8:
                return (value, '')
            low = value.strip().lower()
            hits = [i for (i, row) in self.st.row_index.items() if any((k != self.st.row_id_key and isinstance(v, str) and v.strip().lower().startswith(low) for (k, v) in row.items()))]
            if len(hits) == 1:
                return (hits[0], '')
            if hits:
                return (value, f"the text {value!r} appears on several listings ({', '.join(hits[:6])}); pass a single listing id")
            return (value, f'{value!r} is not a listing id; use the id given in the result rows (their {self.st.row_id_key} field)')
        out = dict(args)
        for key in free:
            if key not in args:
                continue
            fixed = [fix(v) for v in args[key]] if isinstance(args[key], list) else [fix(args[key])]
            problem = next((p for (_, p) in fixed if p), '')
            if problem:
                return (args, problem)
            out[key] = [v for (v, _) in fixed] if isinstance(args[key], list) else fixed[0][0]
        return (out, '')

    def cart_sets_quantity(self, name: str) -> bool:
        props = self.st.arg_schemas.get(name, {}).get('properties') or {}
        if any((isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) >= 2) for s in props.values())):
            return False
        return any((isinstance(s, dict) and 'integer' in schema_types(s) and (s.get('minimum') == 0) for s in props.values()))

    def wrap_up(self, again: bool=False) -> str:
        if self.st.over or self.st.env_broke or self.net.steps_left() <= 0:
            return ''
        (buy, refuse) = (self.st.jobs.get('buy'), self.st.jobs.get('refuse'))
        fresh = self.st.cart_seen is not None and self.st.cart_viewed_at >= self.st.cart_changed_at
        holding = bool(self.review.cart_listings()) if fresh else bool(self.st.cart_counts)
        if buy and (holding or (self.st.cart_changed and (not fresh) and (self.net.steps_left() >= 2))):
            faults: list[str] = []
            args = coerce_to_schema(dict(self.st.sheet.get('buy_args') or {}), self.st.arg_schemas.get(buy), 'args', faults)
            if not faults:
                if F_R12_CHAIN_GATE and self.net.time_fits(1):
                    self.review.chain_gate()
                if self.net.steps_left() >= 2:
                    self.review.pass_gate('order')
                if self.st.over or self.net.steps_left() <= 0:
                    return ''
                reply = self.net.take_step(buy, args)
                self.st.log.append({'kind': 'forced_buy', 'reply': clip_value(reply, 200)})
                if self.st.over or self.net.steps_left() <= 0:
                    return ''
                if reply is not None and again:
                    self.st.buy_tries += 1
                    self.st.buy_refused_at = self.st.at_step
                    self.pages.take_own_result(buy, reply.get('observation'))
                    return 'the checkout was turned down: ' + self.loop.render(reply.get('observation'))
        if refuse:
            faults = []
            args = coerce_to_schema(self.decline_text_args(refuse, dict(self.st.sheet.get('refuse_args') or {})), self.st.arg_schemas.get(refuse), 'args', faults)
            if not faults:
                if self.net.steps_left() >= 2:
                    self.review.pass_gate('end')
                if self.st.over or self.net.steps_left() <= 0:
                    return ''
                reply = self.net.take_step(refuse, args)
                self.st.log.append({'kind': 'forced_refuse', 'reply': clip_value(reply, 200)})
                if reply is not None and (not self.st.over):
                    self.pages.take_own_result(refuse, reply.get('observation'))
                    return 'the ending without a purchase was turned down: ' + self.loop.render(reply.get('observation'))
        return ''


class Job:
    """One tool call of the loop model while Loop works through it."""

    def __init__(self, cid: str) -> None:
        self.cid = cid
        self.name: Any = None
        self.args: dict[str, Any] = {}
        self.trimmed = ''
        self.many: str | None = None
        self.target: str | None = None
        self.ending = False
        self.mark = ''
        self.readers: set[str] = set()


class Loop(Part):
    """The episode from the first turn to the end: model turns, and each tool call screened, held or sent."""

    def track_cart_call(self, name: str, args: dict[str, Any]) -> None:
        self.st.cart_changed = True
        self.st.cart_changed_at = self.st.at_step
        spec = self.st.arg_schemas.get(name, {}).get('properties') or {}
        removal = False
        for (key, value) in args.items():
            options = (spec.get(key) or {}).get('enum')
            if isinstance(options, list) and len(options) >= 2 and (value in options[1:]):
                removal = True
        loose_count = next((whole_number(v) for v in args.values() if whole_number(v) is not None and (not isinstance(v, str))), None)
        lines: list[tuple[str, int | None]] = []
        whole = False
        for (key, value) in args.items():
            if isinstance(value, list):
                shape = (spec.get(key) or {}).get('items') or {}
                if not value and isinstance(shape, dict) and (shape.get('type') == 'object' or 'properties' in shape):
                    whole = True
                for line in value:
                    if isinstance(line, dict):
                        whole = True
                        ident = next((v for v in line.values() if isinstance(v, str) and v in self.st.row_index), None)
                        count = next((whole_number(v) for v in line.values() if whole_number(v) is not None and (not isinstance(v, str))), 1)
                        if ident:
                            lines.append((ident, count))
                    elif isinstance(line, str) and line in self.st.row_index:
                        lines.append((line, loose_count))
            elif isinstance(value, str) and value in self.st.row_index:
                lines.append((value, loose_count))
        if whole:
            self.st.cart_counts = {i: c for (i, c) in lines if c and c > 0}
            return
        sets = self.fin.cart_sets_quantity(name)
        for (ident, count) in lines:
            have = self.st.cart_counts.get(ident, 0)
            if removal:
                have = have - count if count else 0
            elif sets:
                have = count if count is not None else 1
            else:
                have = have + (count if count is not None else 1)
            if have > 0:
                self.st.cart_counts[ident] = have
            else:
                self.st.cart_counts.pop(ident, None)

    def removes(self, name: str, args: dict[str, Any]) -> bool:
        """Whether a cart call takes units out: a removal value of an enum argument (any value after the first), a
        negative count, or a count of 0 for a tool that sets quantities."""
        spec = self.st.arg_schemas.get(name, {}).get('properties') or {}
        if any((isinstance((spec.get(k) or {}).get('enum'), list) and len(spec[k]['enum']) >= 2 and (v in spec[k]['enum'][1:]) for (k, v) in args.items())):
            return True
        count = next((whole_number(v) for v in args.values() if whole_number(v) is not None and (not isinstance(v, str))), None)
        return count is not None and (count < 0 or (count == 0 and self.fin.cart_sets_quantity(name)))

    def removed_ids(self, name: str, args: dict[str, Any]) -> list[str]:
        """Cart lines that a removing cart call takes out whole (every unit the line holds)."""
        if not self.removes(name, args):
            return []
        count = next((whole_number(v) for v in args.values() if whole_number(v) is not None and (not isinstance(v, str))), None)
        ids = [v for v in args.values() if isinstance(v, str) and v in self.st.cart_counts]
        return [i for i in ids if count is None or count == 0 or abs(count) >= self.st.cart_counts[i]]

    def last_line_kept(self, job: Job) -> str | None:
        """F_R12_LAST_LINE: in the final steps, a cart call that would take out the only cart line of its
        category, with no line of that category added in the same turn, is not sent."""
        if job.name != self.st.jobs.get('cart') or self.net.steps_left() > len(self.st.wants) + 5:
            return None
        gone = self.removed_ids(job.name, job.args)
        if not gone:
            return None
        cat_key = self.catalog.category_field()

        def kind(ident: str) -> str:
            return str((self.st.row_index.get(ident) or {}).get(cat_key) or '').strip().lower() if cat_key else ident
        added: list[str] = []
        for call in self.st.turn_calls:
            fn = call.get('function') or {}
            got = parse_json_object(fn.get('arguments')) if isinstance(fn.get('arguments'), str) else None
            if fn.get('name') == job.name and got and (not self.removes(job.name, got)):
                added += [v for v in got.values() if isinstance(v, str) and v in self.st.row_index]
        for ident in gone:
            if any((i != ident and kind(i) == kind(ident) for i in list(self.st.cart_counts) + added)):
                continue
            self.st.log.append({'kind': 'last_line_kept', 'id': ident})
            return f'Not sent (no step spent): only a few steps remain and {ident} is the only line in the cart for its item, so taking it out would leave that item unbought. Keep it, or add a fitting replacement in the same turn.'
        return None

    def open_told_rows(self) -> str:
        """F_R12_OPEN_TOLD, once, before an ending without a sale goes out: up to OPEN_TOLD_CAP of the top three rows
        of the searches whose phrase carries a word of a value the shopper told (Review.told_words), never read with
        the attributes view, are read with it whatever their title says: every rank-0 row first, latest search first, then rank 1,
        then rank 2. Their records are shown and the ending waits for the next call; it is never refused."""
        if self.st.told_opened or self.net.steps_left() < OPEN_TOLD_CAP + self.review.kept_steps() + 2 or (not self.net.time_fits(2)):
            return ''
        self.st.told_opened = True
        words = self.review.told_words()
        runs = [ids for (phrase, ids, first) in reversed(self.st.searches) if first and bare_stems(phrase) & words]
        picks: list[str] = []
        for rank in range(3):
            for ids in runs:
                if rank < len(ids) and ids[rank] not in picks and ids[rank] not in self.st.full_pages and (not self.review.is_gone(ids[rank])):
                    picks.append(ids[rank])
        texts = self.pages.read_missing_facts(picks[:OPEN_TOLD_CAP], self.review.kept_steps() + 2)
        if not texts:
            return ''
        self.st.log.append({'kind': 'told_rows', 'ids': picks[:OPEN_TOLD_CAP]})
        return 'Not sent yet. Before the ending, Rok read the attributes of the top rows of your searches for the requested values that had never been opened (one step each):\n' + '\n'.join(texts) + '\nIf one of them meets every requirement in force, get it into the cart and check out; if none does, send the ending again.'

    def drop_follow(self, job: Job, reply: dict[str, Any]) -> str:
        """After a note that took a requirement away: on a refused ending without a sale, the requirements update of
        Review.drop_update (F_R12_DROP_UPDATE); otherwise, or when that names nothing, one fresh search with the
        values still in force (F_R12_TOLD_SEARCH) on a refused ending without a sale, with an empty cart, or with
        a pick by price or measure; a cart that still fits needs no new search."""
        number = self.review.dropped_line()
        refused = self.fin.is_decline(job.name, job.args) and (not reply.get('finished'))
        if F_R12_DROP_UPDATE and refused and (not self.st.over):
            done = self.review.drop_update()
            if done:
                return '\n[Rok] ' + done
        if F_R12_TOLD_SEARCH and (not self.st.over) and (refused or not self.st.cart_counts or self.st.sheet.get('pick') in ('cheapest', 'most', 'least')):
            return self.pages.drop_search(number)
        return ''

    def render(self, observation: Any) -> str:
        text = to_json_text(clip_value(observation, SHOWN_CHARS))
        if len(text) > RESULT_CHAR_CAP:
            text = text[:RESULT_CHAR_CAP] + '...(cut)'
        return text

    def side_param(self) -> str | None:
        schema = self.st.arg_schemas.get(self.st.jobs.get('compare') or '') or {}
        props = schema.get('properties') or {}
        target = next((k for (k, v) in props.items() if isinstance(v, dict) and 'array' in schema_types(v)), None)
        if not target or any((k != target for k in schema.get('required') or [])):
            return None
        return target

    def stated_room(self) -> int:
        """side_room under the cap the side-by-side tool states in words (the smaller of the two), unless the
        store has already taken more listings than that in one call."""
        tool = self.st.jobs.get('compare') or ''
        many = self.side_param()
        spec = ((self.st.arg_schemas.get(tool) or {}).get('properties') or {}).get(many or '') or {}
        if isinstance(spec.get('maxItems'), int):
            base = spec['maxItems']
        elif self.st.side_bad:
            base = max(2, min(self.st.side_bad - 1, max(self.st.side_ok, 2)))
        else:
            base = max(4, self.st.side_ok)
        said = stated_cap(next((t for t in self.st.tool_specs if t['function']['name'] == tool), None), many)
        return min(base, max(said, self.st.side_ok)) if said else base

    def side_room(self) -> int:
        if F_TOOL_LIMITS:
            return self.stated_room()
        spec = ((self.st.arg_schemas.get(self.st.jobs.get('compare') or '') or {}).get('properties') or {}).get(self.side_param() or '') or {}
        if isinstance(spec.get('maxItems'), int):
            return spec['maxItems']
        if self.st.side_bad:
            return max(2, min(self.st.side_bad - 1, max(self.st.side_ok, 2)))
        return max(4, self.st.side_ok)

    def merge_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        (opener, joiner, target) = (self.st.jobs.get('open'), self.st.jobs.get('compare'), self.catalog.lone_id_param(self.st.jobs.get('open')))
        many = self.side_param()
        if not opener or not joiner or (not target) or (not many) or (opener == joiner):
            return calls
        if self.catalog.side_view_bare():
            return calls
        picks: list[tuple[int, str]] = []
        for (index, call) in enumerate(calls):
            fn = call.get('function') or {}
            args = parse_json_object(fn.get('arguments')) if isinstance(fn.get('arguments'), str) else None
            ident = str((args or {}).get(target) or '').strip()
            if fn.get('name') == opener and args and (set(args) == {target}) and (ident in self.st.row_index) and (ident not in self.st.sold_at):
                picks.append((index, ident))
        picks = picks[:self.side_room()]
        if len(picks) < 2:
            return calls
        first = picks[0][0]
        out = []
        for (index, call) in enumerate(calls):
            if index == first:
                out.append({'id': call['id'], 'type': 'function', 'function': {'name': joiner, 'arguments': to_json_text({many: [i for (_, i) in picks]})}})
            elif index in {i for (i, _) in picks}:
                out.append({'id': call['id'], 'joined': True})
            else:
                out.append(call)
        self.st.log.append({'kind': 'joined', 'ids': [i for (_, i) in picks]})
        return out

    def handle_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        """Answer each tool call of one model turn, in sequence: a screen, the holds, or the store's reply."""
        out: list[tuple[str, str]] = []
        calls = self.merge_opens(calls)
        if F_R12_LAST_LINE:
            self.st.turn_calls = calls
        for call in calls:
            allowed = {t['function']['name'] for t in self.fin.visible_tools()}
            job = Job(call['id'])
            said = self.admit(job, call, allowed)
            if said is None:
                self.fit_side(job)
                said = self.screen(job, len(calls))
            if said is None:
                said = self.drop_shown(job)
            if said is None:
                said = self.hold(job)
            if said is None:
                said = self.carry_out(job)
            out.append((job.cid, said))
        return out

    def admit(self, job: Job, call: dict[str, Any], allowed: set[str]) -> str | None:
        """Turn away a call that cannot be sent as it is; otherwise fill in the job's tool and arguments."""
        if call.get('joined'):
            return 'Included in the side-by-side result above: Rok opened these listings together in a single step.'
        fn = call.get('function') or {}
        name = fn.get('name')
        raw = fn.get('arguments')
        if self.st.over:
            return 'Not sent: the episode has ended.'
        if name == self.st.jobs.get('ask'):
            return "Not sent (no step spent): Rok's interviewer sends every message to the shopper."
        if name not in self.st.arg_schemas:
            return f"Not sent (no step spent): there is no such tool. Tools: {', '.join(sorted(allowed))}."
        if name not in allowed:
            return 'Not sent (no step spent): not enough steps remain for that; use: ' + ', '.join(sorted(allowed)) + '.'
        if isinstance(raw, dict):
            args = raw
        else:
            args = parse_json_object(raw) if (raw or '').strip() else {}
            if args is None:
                return 'Not sent (no step spent): the arguments were not a single JSON object.'
        faults: list[str] = []
        args = coerce_to_schema(args, self.st.arg_schemas[name], 'args', faults)
        if faults:
            return 'Not sent (no step spent): ' + '; '.join(faults[:6])
        (args, wrong) = self.fin.resolve_listing_ids(name, args)
        if wrong:
            return 'Not sent (no step spent): ' + wrong + '.'
        job.name = name
        job.args = args
        return None

    def fit_stated_side(self, job: Job) -> None:
        """fit_side, where a cap the tool states in words also counts as shown (the call is cut before it is sent)."""
        job.many = self.side_param() if job.name == self.st.jobs.get('compare') else None
        many = job.many
        tool = next((t for t in self.st.tool_specs if t['function']['name'] == job.name), None)
        known = self.st.side_bad or stated_cap(tool, many)
        if many and isinstance(job.args.get(many), list) and (len(job.args[many]) > self.side_room()) and known:
            room = self.side_room()
            job.trimmed = f"\n[Rok] Only the first {room} listings were sent: this store shows no more than {room} side by side. Still not shown: {', '.join(map(str, job.args[many][room:]))}."
            job.args = {**job.args, many: job.args[many][:room]}

    def fit_side(self, job: Job) -> None:
        """A side-by-side call longer than the store has shown it can take is cut to that length."""
        if F_TOOL_LIMITS:
            self.fit_stated_side(job)
            return
        job.many = self.side_param() if job.name == self.st.jobs.get('compare') else None
        many = job.many
        if many and isinstance(job.args.get(many), list) and (len(job.args[many]) > self.side_room()) and self.st.side_bad:
            room = self.side_room()
            job.trimmed = f"\n[Rok] Only the first {room} listings were sent: this store shows no more than {room} side by side. Still not shown: {', '.join(map(str, job.args[many][room:]))}."
            job.args = {**job.args, many: job.args[many][:room]}

    def screen(self, job: Job, count: int) -> str | None:
        """Calls that would waste a step: a sold-out open, a removal of an absent line, an ending sent with other
        calls, a non-ending call inside the kept steps, a repeat, a cached read, a view with nothing new."""
        name, args = job.name, job.args
        if F_MODEL_RETRY and F_SLOW_GUARD and name in self.st.spent_tools:
            return 'Not sent (no step spent): the store turned this tool away because its allowance is used up, and it takes no more of these calls.'
        job.target = self.catalog.lone_id_param(name)
        target = job.target
        if target and name in (self.st.jobs.get('open'), self.st.jobs.get('offer')) and (str(args.get(target)) in self.st.sold_at):
            return f'Not sent (no step spent): {args[target]} sold out at step {self.st.sold_at[str(args[target])]} and will not return. Replace its cart line with another listing that fits.'
        if name == self.st.jobs.get('cart') and any(((whole_number(v) or 0) < 0 for v in args.values() if not isinstance(v, str))):
            view = to_json_text(self.st.cart_seen) if self.st.cart_viewed_at >= self.st.cart_changed_at else ''
            absent = [v for v in args.values() if isinstance(v, str) and v in self.st.row_index and (v not in self.st.cart_counts) and (v not in view) and (v not in self.st.cart_reply)]
            if absent:
                return f'Not sent (no step spent): {absent[0]} is not in the cart, so there is nothing to remove. Cart now: ' + self.review.cart_report()
        if F_R12_LAST_LINE:
            kept = self.last_line_kept(job)
            if kept:
                return kept
        job.ending = self.fin.is_checkout(name, args) or self.fin.is_decline(name, args)
        if job.ending and count > 1:
            return 'Not sent (no step spent): send the finishing call by itself, once you have read your other results.'
        if not job.ending and self.net.steps_left() <= self.review.kept_steps() and (self.st.jobs.get('buy') or self.st.jobs.get('refuse')):
            return "Not sent (no step spent): the remaining steps are reserved for checking out (with the shopper's approval of the basket when that is due) or for ending with no purchase."
        job.mark = name + to_json_text(args)
        if not job.ending and name != self.st.jobs.get('cart') and (job.mark in self.st.last_marks):
            return 'Not sent (no step spent): you just made this exact call and nothing has changed; its result is above.'
        job.readers = {self.st.jobs.get(r) for r in ('search', 'open', 'offer', 'compare')} - {None}
        cached = self.st.reads.get(job.mark)
        if name in job.readers and cached and (cached[0] == self.st.epoch):
            return f'Not sent (no step spent): the same call at step {cached[1]} gave this result, and nothing has changed since then:\n' + cached[2][:3000]
        if target and name in job.readers and (name != self.st.jobs.get('compare')) and self.catalog.nothing_new(name, str(args.get(target))):
            return f'Not sent (no step spent): {args[target]} has already shown all this view offers since the last change. Its record: ' + self.review.record_text(str(args[target]))
        return None

    def drop_shown(self, job: Job) -> str | None:
        """Listings of a side-by-side call that have nothing new to show are left out; all of them: not run."""
        name, args, many = job.name, job.args, job.many
        if many and isinstance(args.get(many), list) and args[many]:
            keep = [i for i in args[many] if not self.catalog.nothing_new(name, str(i))]
            if not keep:
                return 'Not sent (no step spent): these listings have already shown all this view offers since the last change. Their records:\n' + '\n'.join((f'{i}: {self.review.record_text(str(i))}' for i in args[many]))[:4000]
            if len(keep) < len(args[many]):
                job.trimmed += f"\n[Rok] Skipped {', '.join((str(i) for i in args[many] if i not in keep))}: already shown in full, and nothing has changed since."
                job.args = {**args, many: keep}
                job.mark = name + to_json_text(job.args)
        return None

    def hold(self, job: Job) -> str | None:
        """Holds on an ending call, in sequence: the buying or no-sale checks, the late no-sale and search pushes,
        the confirmed-lead cart, then the gate for steps the request asks for at that point."""
        if F_SLOW_GUARD and job.ending and (not self.net.time_fits(2)):
            self.st.log.append({'kind': 'time_skip', 'what': 'holds', 'left': round(self.net.time_left(), 1)})
            if self.fin.is_decline(job.name, job.args):
                done = self.fin.cart_confirmed()
                if done:
                    return done + (' [episode finished]' if self.st.over else '')
                job.args = self.fin.decline_text_args(job.name, job.args)
            return None
        if job.ending and self.net.steps_left() > 2:
            if self.fin.is_checkout(job.name, job.args):
                said = self.hold_checkout(job)
            elif self.net.steps_left() > 5:
                said = self.hold_decline()
            else:
                said = None
            if said is not None:
                return said
        if self.fin.is_decline(job.name, job.args) and self.net.steps_left() <= 5 and (not self.st.late_hold_done):
            said = self.hold_late_decline()
            if said is not None:
                return said
        if self.fin.is_decline(job.name, job.args) and (not self.st.hints) and (not self.st.search_pushed) and (self.net.steps_left() > self.st.step_cap // 3):
            self.st.search_pushed = True
            return f"Not sent yet (no step spent). {self.net.steps_left()} steps still remain, so before ending, search in ways you have not tried yet: browse each requested item's category and page through it with the token or offset the store returns, try one-word value queries and other names for the property, and open the listings that might fit. Only after that, end with no purchase."
        if F_R12_OPEN_TOLD and self.fin.is_decline(job.name, job.args):
            opened = self.open_told_rows()
            if opened:
                return opened
        if self.fin.is_decline(job.name, job.args):
            done = self.fin.cart_confirmed()
            if done:
                return done + (' [episode finished]' if self.st.over else '')
            job.args = self.fin.decline_text_args(job.name, job.args)
        if job.ending and self.net.steps_left() >= 2:
            (go, said) = self.review.pass_gate('order' if self.fin.is_checkout(job.name, job.args) else 'end')
            if not go:
                return 'Not sent yet. ' + said + '\nCart now: ' + self.review.cart_report()
            if said:
                job.trimmed = '\n[Rok] ' + said + job.trimmed
        return None

    def hold_checkout(self, job: Job) -> str | None:
        """Before buying: a wider search, a fresh cart view, sold-out lines, pointer chains, the cart check."""
        wider = self.pages.wider_search_hold()
        if wider:
            return wider + '\nCart now: ' + self.review.cart_report()
        reread = self.review.reread_cart()
        if self.st.cart_changed and self.st.cart_viewed_at < self.st.cart_changed_at and self.st.jobs.get('cart_view') and (self.net.steps_left() > 3):
            faults: list[str] = []
            view_args = coerce_to_schema({}, self.st.arg_schemas.get(self.st.jobs['cart_view']), 'args', faults)
            view = self.net.take_step(self.st.jobs['cart_view'], view_args) if not faults else None
            if view is not None:
                self.st.cart_seen = view.get('observation')
                self.st.cart_viewed_at = self.st.at_step
                change = self.pages.take_own_result(self.st.jobs['cart_view'], self.st.cart_seen)
                if change:
                    self.st.pending_notes = []
                    return 'Not sent yet (no step spent). Viewing the cart revealed a change: ' + change + '\nCart now: ' + self.review.cart_report() + '\nApply the change, repair the cart where needed, then check out.'
        swap = self.review.replacement_hint()
        held = [i for i in self.review.cart_listings() if i in self.st.sold_at] if self.st.sold_holds < 2 else []
        if swap or held:
            self.st.sold_holds += 0 if swap else 1
            return 'Not sent yet (no step spent). ' + (swap or f"{', '.join(held)} in the cart has sold out: replace that line first.") + '\nCart now: ' + self.review.cart_report()
        (chain, stop) = self.review.walk_chains()
        if stop:
            return 'Not sent yet (no step spent). ' + chain + '\nCart now: ' + self.review.cart_report()
        if F_R12_CHAIN_GATE:
            gate = self.review.chain_gate()
            if gate:
                return 'Not sent yet. ' + chain + '[Rok opened the full pages of the listings a requirement points to, one step each]\n' + gate + '\nCheck the cart against the values these pages give (a page may send you on to another listing: open that one too), then check out.\nCart now: ' + self.review.cart_report()
        if F_R12_READ_LINES:
            read = self.pages.read_unread_lines(self.review.cart_listings(), self.review.kept_steps()) + self.pages.read_missing_facts(self.review.cart_listings(), self.review.kept_steps() + 3) + self.pages.read_cheaper_rows()
        else:
            read = self.pages.read_missing_facts(self.review.cart_listings(), self.review.kept_steps() + 3) + self.pages.read_cheaper_rows()
        verdict = self.review.judge_cart()
        if verdict:
            seen = "For the check, Rok read these listings' attributes (one step each):\n" + '\n'.join(read) + '\n' if read else ''
            return 'Not sent yet. ' + chain + seen + verdict + '\nCart now: ' + self.review.cart_report()
        if reread and self.st.spend > SPEND_WARN:
            return "Not sent yet (no step spent). Rok has just re-read the cart's listings:\n" + reread + '\nCheck out again if every line still shows stock and a price within the budget; if not, replace that line first.'
        return None

    def hold_decline(self) -> str | None:
        """Before an ending without a sale while steps remain: the review, then up to two pushes toward its leads."""
        review = self.review.audit_decline()
        if review:
            return 'Not sent yet (no step spent). ' + review
        if self.st.hints and self.net.steps_left() > 4 and (self.st.decline_holds == 0):
            self.st.decline_holds = 1
            self.st.log.append({'kind': 'refuse_blocked', 'leads': list(self.st.hints)})
            return 'Not sent yet (no step spent). The review found listings that seem to fit: ' + ', '.join(self.st.hints) + '. Open them if you need to, get the best-fitting one into the cart (following the pick rule) and check out. End with no purchase only if every one of them plainly fails a requirement; then name that requirement in your text and send the ending call again.'
        if self.st.hints and self.net.steps_left() > 4 and (self.st.decline_holds == 1):
            self.st.decline_holds = 2
            first = self.st.hints[0]
            self.review.judge_cart([first], REVIEW_SPEND_CAP if F_DECLINE_REVIEW and F_SLOW_GUARD else SPEND_CAP)
            if self.st.check_verdicts.get(self.review.verdict_key([first])) and first in self.st.hints:
                self.st.confirmed_hint = first
                self.st.log.append({'kind': 'refuse_blocked', 'confirmed': first})
                return f"Not sent yet (no step spent). An independent check of the records of {first} says it meets every requirement. Get it into the cart at the requested quantity and check out; end with no purchase only if its record plainly breaks a requirement."
        return None

    def hold_late_decline(self) -> str | None:
        """Once, with five steps or fewer left: an ending without a sale that would throw away a full cart built earlier is held."""
        lines = [i for i in self.st.best_cart if not self.review.is_gone(i)]
        fresh = self.st.cart_seen is not None and self.st.cart_viewed_at >= self.st.cart_changed_at
        missing = [i for i in lines if i not in self.review.cart_listings() or (i in self.st.taken_off and (not fresh))]
        if len(lines) >= len(self.st.wants) and len(missing) + 1 <= self.net.steps_left():
            self.st.late_hold_done = True
            self.st.log.append({'kind': 'late_decline_held', 'lines': lines, 'missing': missing})
            redo = f"Return {', '.join(missing)} to the cart at the requested quantities, then check out." if missing else 'Check out the cart as it stands.'
            return 'Not sent yet (no step spent). Ending with no purchase scores nothing when the store has a fitting listing, and running low on steps does not show that it lacks one. ' + redo + ' End with no purchase only if your search showed that some requested item has no listing meeting its requirements; then send the ending call again.'
        return None

    def carry_out(self, job: Job) -> str:
        """Send the call to the store, learn from the result, and build the text the model sees."""
        name, args = job.name, job.args
        reply = self.net.take_step(name, args)
        if reply is None:
            return 'The call did not go through (network or server problem). You may try it again.'
        observation = reply.get('observation')
        if F_MODEL_RETRY and F_SLOW_GUARD and allowance_spent(observation):
            self.st.spent_tools.add(name)
        self.learn_side_limit(job, observation)
        self.st.last_marks = (self.st.last_marks + [job.mark])[-3:]
        self.st.id_params.update(((name, k) for (k, v) in args.items() if (v if isinstance(v, list) else [v]) and all((isinstance(x, str) and x in self.st.row_index for x in (v if isinstance(v, list) else [v])))))
        self.catalog.take_in(observation)
        if name == self.st.jobs.get('compare'):
            rows = self.catalog.records_of(observation)
            if rows:
                self.st.shows_facts[name] = self.st.shows_facts.get(name, False) or any((attribute_block(r) for (_, r) in rows))
        if name in job.readers - {self.st.jobs.get('search')}:
            self.catalog.note_fields_shown(name, observation)
        plain = {tool for (kind, tool) in self.st.jobs.items() if kind != 'search'}
        paged = self.pages.read_on_pages(name, args, observation) if name not in plain and (not job.ending) else ''
        if F_R12_TOLD_SEARCH and (not F_R12_NO_TOLD_PAGE) and name == self.st.jobs.get('search') and (not paged) and (not job.ending):
            paged = self.pages.page_further(name, args, observation)
        if self.catalog.lone_id_param(name) and name not in (self.st.jobs.get('cart'), self.st.jobs.get('buy'), self.st.jobs.get('refuse'), self.st.jobs.get('ask')):
            self.learn_single_view(name, observation)
        if name == self.st.jobs.get('cart'):
            self.learn_cart_reply(name, args, observation)
        if name == self.st.jobs.get('cart_view'):
            self.st.cart_seen = observation
            self.st.cart_viewed_at = self.st.at_step
        if self.fin.is_checkout(name, args):
            self.st.buy_tries += 1
            if not reply.get('finished'):
                self.st.buy_refused_at = self.st.at_step
        return self.compose(job, reply, observation, paged, plain)

    def learn_side_limit(self, job: Job, observation: Any) -> None:
        """A side-by-side reply that names not one of the listings sent marks that count as too many."""
        many, args = job.many, job.args
        if many and isinstance(args.get(many), list) and (len(args[many]) > 1):
            shown = to_json_text(observation)
            if not any((str(i) in shown for i in args[many])):
                self.st.side_bad = min(self.st.side_bad or 999, len(args[many]))
                job.trimmed += f'\n[Rok] The side-by-side view returned nothing for {len(args[many])} listings: send no more than {self.side_room()} per call.'
            else:
                self.st.side_ok = max(self.st.side_ok, len(args[many]))

    def learn_single_view(self, name: str, observation: Any) -> None:
        """Whether a one-listing view shows attributes; the attributes and offer roles follow what was seen."""
        listing = self.catalog.listing_of(observation) if isinstance(observation, dict) else None
        if listing:
            self.st.shows_facts[name] = self.st.shows_facts.get(name, False) or bool(attribute_block(observation))
            if name == self.st.jobs.get('open') and (not self.st.shows_facts[name]):
                other = next((t for t in self.st.arg_schemas if t != name and self.catalog.lone_id_param(t) and (self.st.shows_facts.get(t) is not False) and (t not in (self.st.jobs.get('cart'), self.st.jobs.get('buy'), self.st.jobs.get('refuse'), self.st.jobs.get('ask'), self.st.jobs.get('search')))), None)
                if other:
                    self.st.jobs['open'] = other
            if not self.st.shows_facts[name] and name != self.st.jobs.get('open') and (not self.st.jobs.get('offer')):
                self.st.jobs['offer'] = name
            if name == self.review.live_offer_view():
                self.st.offer_read_at[listing] = self.st.at_step

    def learn_cart_reply(self, name: str, args: dict[str, Any], observation: Any) -> None:
        """Cart bookkeeping after a cart call: counts, lines taken off, and the fullest cart built so far."""
        self.track_cart_call(name, args)
        self.st.cart_reply = to_json_text(observation)
        self.st.last_marks = []
        counts = [whole_number(v) for v in args.values() if not isinstance(v, str) and whole_number(v) is not None]
        for value in args.values():
            if counts and isinstance(value, str) and (value in self.st.row_index):
                if counts[0] <= 0 and (not listed_in_rows(observation, value)):
                    self.st.taken_off.add(value)
                elif counts[0] > 0:
                    self.st.taken_off.discard(value)
        lines = [i for i in self.review.cart_listings() if not self.review.is_gone(i) and i not in self.st.taken_off]
        if len(lines) >= len(self.st.best_cart):
            self.st.best_cart = lines

    def compose(self, job: Job, reply: dict[str, Any], observation: Any, paged: str, plain: set[str]) -> str:
        """The result text: the shown reply with paging, hints, proof lines, read notes and swap advice."""
        name, args = job.name, job.args
        text = self.render(observation) + paged + job.trimmed
        if name not in plain and (not job.ending):
            text += self.pages.cheapest_hint([observation] + self.st.page_results)
        base = text
        proof = self.review.proof_lines(observation)
        if proof:
            text += '\n' + proof
        if name == self.st.jobs.get('cart') and any((whole_number(v) == 0 for v in args.values() if not isinstance(v, str))):
            if any((isinstance(v, str) and v in self.st.row_index and listed_in_rows(observation, v) for v in args.values())):
                text += '\n[Rok] A count of 0 kept that line in the cart: this tool adds the count to the line, so a negative count removes units.'
        if F_R12_DROP_UPDATE or F_R12_TOLD_SEARCH:
            self.st.retired_before = self.review.retired_counts()
        self.st.note_said_sold = False
        lines = self.catalog.note_lines(observation)
        if lines:
            meaning = self.catalog.read_store_note(lines)
            if meaning:
                text += '\n[note read] ' + meaning + '. Apply this change to your picks before you check out.'
        if (F_R12_DROP_UPDATE or F_R12_TOLD_SEARCH) and self.review.dropped_line():
            text += self.drop_follow(job, reply)
        if self.fin.is_checkout(name, args) and (not reply.get('finished')) and (self.st.note_said_sold or any((line in self.catalog.visible_text(observation) for line in self.st.sold_phrases))):
            seen = self.review.find_sold_line()
            if seen:
                text += "\n[Rok re-read the cart's lines after the refusal]\n" + seen
        swap = self.review.replacement_hint()
        if swap:
            text += '\n' + swap
        if self.st.pending_notes:
            text += "\n[notes seen on Rok's own calls] " + ' | '.join(self.st.pending_notes)
            self.st.pending_notes = []
        if name in job.readers:
            self.st.reads[job.mark] = (self.st.epoch, self.st.at_step, base)
        if reply.get('finished'):
            text += ' [episode finished]'
        return text

    def shorten_transcript(self) -> None:
        size = sum((len(to_json_text(m.get('content'))) for m in self.st.transcript))
        if size <= TRANSCRIPT_CHAR_CAP:
            return
        tools = [i for (i, m) in enumerate(self.st.transcript) if m.get('role') == 'tool']
        for index in tools[:max(1, len(tools) // 2)]:
            content = self.st.transcript[index].get('content') or ''
            if len(content) > 600:
                ids = re.findall('"([A-Za-z0-9][A-Za-z0-9_\\-:.]{5,40})"', content)
                ids = [i for i in dict.fromkeys(ids) if i in self.st.row_index][:20]
                self.st.transcript[index]['content'] = content[:400] + f"...(older result cut short; listings in it: {', '.join(ids)})"

    def review_at_cap(self) -> None:
        """Once, when the turns stopped at the spend cap with nothing in the cart and steps left: the review that an
        ending without a sale gets (the listings seen, then a check of the first lead it names), and the cart of a
        lead that the check approves, before the forced ending without a sale."""
        fresh = self.st.cart_seen is not None and self.st.cart_viewed_at >= self.st.cart_changed_at
        holding = bool(self.review.cart_listings()) if fresh else bool(self.st.cart_counts)
        if self.st.over or self.st.env_broke or self.st.model_broke or (self.st.spend < SPEND_CAP) or holding or (self.st.cart_changed and (not fresh)):
            return
        if not self.st.jobs.get('cart') or not self.st.jobs.get('buy') or self.net.steps_left() <= 5 or (not self.net.time_fits(2)):
            return
        self.st.log.append({'kind': 'cap_review', 'spend': round(self.st.spend, 5), 'left': round(self.net.time_left(), 1)})
        self.review.audit_decline()
        if self.st.hints and (not self.st.confirmed_hint) and (self.net.steps_left() > 4) and self.net.time_fits(2):
            first = self.st.hints[0]
            self.review.judge_cart([first], REVIEW_SPEND_CAP)
            if self.st.check_verdicts.get(self.review.verdict_key([first])) and first in self.st.hints:
                self.st.confirmed_hint = first
                self.st.log.append({'kind': 'refuse_blocked', 'confirmed': first})
        if self.st.confirmed_hint and (not self.st.over) and self.net.time_fits(2):
            done = self.fin.cart_confirmed()
            self.st.log.append({'kind': 'cap_review_cart', 'done': bool(done), 'over': self.st.over})

    def play(self) -> list[dict[str, Any]]:
        """Read the brief, open the transcript, run the turns, then make sure the episode ends; returns the trace."""
        if self.st.over:
            return self.st.log
        self.brief.read_brief()
        self.st.log.append({'kind': 'brief', 'brief': self.st.sheet})
        opening = self.st.policy.get('opening')
        if opening is not None:
            self.catalog.take_in(opening)
        intro = "SHOPPER'S REQUEST:\n" + self.st.ask_text
        if opening is not None:
            intro += '\n\nWHAT THE STORE SHOWS FIRST:\n' + self.render(opening)
        intro += "\n\nBegin by searching each item's category and opening several listings in full, so that the interviewer can see which properties these listings have.\n\n" + self.brief.status_report()
        self.st.transcript = [{'role': 'system', 'content': LOOP_PROMPT}, {'role': 'user', 'content': intro}]
        self.drive(TURN_CAP, SPEND_CAP)
        if F_DECLINE_REVIEW and F_SLOW_GUARD:
            self.review_at_cap()
        if not self.st.over:
            turned = self.fin.wrap_up(again=True)
            if F_SLOW_GUARD and turned and (not self.net.time_fits(2)):
                self.st.log.append({'kind': 'time_skip', 'what': 'second round'})
                turned = ''
            if turned and (not self.st.over) and (not self.st.env_broke) and (not self.st.model_broke) and (self.net.steps_left() > 1):
                self.st.transcript.append({'role': 'user', 'content': '[Rok] Rok tried to end the episode and ' + turned + '\nDo what the store says, then finish.\n' + self.brief.status_report()})
                self.drive(self.st.turns + 10, SPEND_CAP + 0.01)
            if not self.st.over:
                self.fin.wrap_up()
        self.st.log.append({'kind': 'end', 'finished': self.st.over, 'steps': self.st.at_step, 'cost': round(self.st.spend, 5), 'calls': self.st.model_calls})
        return self.st.log

    def drive(self, loop_cap: int, spend_cap: float) -> None:
        """Model turns until the episode ends, steps or turns run out, or spend reaches the cap."""
        idle = 0
        while not self.st.over and (not self.st.env_broke) and (self.net.steps_left() > 0) and (self.st.turns < loop_cap) and (True if F_R12_NO_SPEND_END and (not F_R12_LEAN_CONFIRM or (_lean_rung == 0 and _lean_zero >= LEAN_CONFIRM_N)) else self.st.spend < spend_cap):
            if F_SLOW_GUARD and (not self.net.time_fits(1) or self.net.time_left() - TIME_RESERVE < SHORTEST_CALL):
                self.st.log.append({'kind': 'time_stop', 'left': round(self.net.time_left(), 1), 'guess': round(self.net.call_guess(), 1)})
                break
            if F_SLOW_GUARD and (not self.net.time_fits(3)) and self.talk.should_ask():
                self.st.interviewed = True
                self.st.log.append({'kind': 'time_skip', 'what': 'interview'})
            if self.talk.should_ask():
                summary = self.talk.hold_interview()
                if summary:
                    self.st.transcript.append({'role': 'user', 'content': summary + '\n' + self.brief.status_report()})
                continue
            self.shorten_transcript()
            self.st.turns += 1
            message = self.net.consult(self.st.transcript, [FAST_MODEL, BACKUP_MODEL], 3000, self.fin.visible_tools())
            if message is None:
                self.st.model_misses += 1
                if F_MODEL_RETRY and F_SLOW_GUARD:
                    self.st.hard_misses = 0 if self.st.fail_passing else self.st.hard_misses + 1
                    if self.st.model_broke or self.st.model_calls >= MODEL_CALL_CAP or self.st.hard_misses > 2:
                        break
                    if self.st.model_misses % 3 == 0:
                        self.st.log.append({'kind': 'miss_pause', 'misses': self.st.model_misses})
                        self.net.time_pause(MISS_PAUSE)
                    continue
                if self.st.model_broke or self.st.model_calls >= MODEL_CALL_CAP or self.st.model_misses > 2:
                    break
                continue
            if F_MODEL_RETRY and F_SLOW_GUARD:
                self.st.model_misses = 0
                self.st.hard_misses = 0
            calls = self.tool_calls_of(message)
            content = self.text_of(message)
            if not calls:
                idle += 1
                self.st.transcript.append({'role': 'assistant', 'content': content or '(no call)'})
                self.st.transcript.append({'role': 'user', 'content': 'Call one of the tools to go on.\n' + self.brief.status_report()})
                if idle >= 3:
                    break
                continue
            idle = 0
            if self.ask_before_commit(content, calls):
                continue
            self.post_turn(content, calls)

    def tool_calls_of(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        """The reply's tool calls in one form: an id, the tool name, and the arguments as one JSON object text."""
        calls = []
        for (number, call) in enumerate(message.get('tool_calls') or []):
            if not isinstance(call, dict):
                continue
            fn = call.get('function') if isinstance(call.get('function'), dict) else {}
            raw = fn.get('arguments')
            parsed = raw if isinstance(raw, dict) else parse_json_object(raw) if isinstance(raw, str) and raw.strip() else {}
            calls.append({'id': str(call.get('id') or f'k{self.st.turns}_{number}'), 'type': 'function', 'function': {'name': str(fn.get('name') or ''), 'arguments': to_json_text(parsed if isinstance(parsed, dict) else {})}})
        return calls

    @staticmethod
    def text_of(message: dict[str, Any]) -> str:
        """The reply's text; a non-text content is written out as JSON."""
        content = message.get('content')
        if not isinstance(content, str):
            content = to_json_text(content) if content else ''
        return content

    def ask_before_commit(self, content: str, calls: list[dict[str, Any]]) -> bool:
        """A cart or ending call before any question, while messages and steps remain: the interview goes first."""
        committing = {self.st.jobs.get('cart'), self.st.jobs.get('buy'), self.st.jobs.get('refuse')} - {None}
        if not self.st.interviewed and self.st.jobs.get('ask') and (self.talk.messages_free() > 0) and any((c['function']['name'] in committing for c in calls)) and (self.net.steps_left() > 8):
            self.st.transcript.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
            for call in calls:
                self.st.transcript.append({'role': 'tool', 'tool_call_id': call['id'], 'content': 'Not sent (no step spent): the shopper gets asked first.'})
            summary = self.talk.hold_interview()
            self.st.transcript.append({'role': 'user', 'content': (summary or 'The interviewer found nothing to ask.') + '\n' + self.brief.status_report()})
            return True
        return False

    def post_turn(self, content: str, calls: list[dict[str, Any]]) -> None:
        """Log the turn, answer its calls, and append the answers to the transcript (the status goes on the last one)."""
        self.st.transcript.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
        self.st.log.append({'kind': 'turn', 'text': content[:300], 'calls': [c['function'] for c in calls]})
        replies = self.handle_calls(calls)
        for (index, (cid, text)) in enumerate(replies):
            if index == len(replies) - 1:
                text += '\n' + self.brief.status_report()
            self.st.transcript.append({'role': 'tool', 'tool_call_id': cid, 'content': text})


class Crew:
    """One episode: its State and one of each part, linked to each other."""

    def __init__(self, problem: dict[str, Any]) -> None:
        self.st = State(problem)
        self.net = Transport(self.st)
        self.brief = Brief(self.st)
        self.catalog = Catalog(self.st)
        self.pages = Pages(self.st)
        self.talk = Interview(self.st)
        self.review = Review(self.st)
        self.fin = Finisher(self.st)
        self.loop = Loop(self.st)
        for part in (self.net, self.brief, self.catalog, self.pages, self.talk, self.review, self.fin, self.loop):
            part.join(self)


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        crew = Crew(problem_data)
    except Exception as exc:
        return [{"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]}]
    st = crew.st
    try:
        return crew.loop.play()
    except Exception as exc:
        st.log.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]})
        try:
            if not st.over:
                crew.fin.wrap_up()
        except Exception as again:
            st.log.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:400]})
        return st.log


__all__ = ["agent_main"]
