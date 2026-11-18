"""22-halo: a shopping agent for one shopper inside an ORO store world.

An open fork of Halo v2 (released by its miner after race 15; itself built on Know Thyself), rebuilt in our own
structure. One episode is a Session made of parts that each own their state:

  Models     - model requests, retries, spend and the episode clock
  Store      - tool schemas, the role of each tool, store calls and step accounting
  Memory     - every listing seen (rows, full records, ids, category labels, attribute tables, result shapes)
  Order      - the shopper's request: items, requirements, budget, message and step limits, process steps, pointers
  Basket     - the cart as the agent's own calls and the cart view describe it, and the cart's line checks
  Reader     - code-side reading: attribute pages, extra result pages, cheaper rows, notes from the store
  Interview  - questions to the shopper: when to ask, which properties, list rotation, reading the answers
  Referee    - checks before an ending: cart review, pointer chains, fits between parts, sold-out swaps, audits
  Pilot      - the light model's loop over the store tools and the guard layer around each of its calls

Every rule of the base is kept; the base's own parts are marked in DESIGN.md.
"""
from __future__ import annotations
import json
import random
import re
import time
from typing import Any
from src.agent.proxy_client import ProxyClient

# ---------------------------------------------------------------------------------------------------------- settings
LIGHT = 'anthropic/claude-haiku-5.5'
HEAVY = 'anthropic/claude-sonnet-5.5'
SPARE = 'openai/gpt-6-luna'

PILOT_TURNS = 42            # main-loop model turns in the first round
MODEL_CALLS = 60            # model requests in one episode, all kinds
SPEND_SOFT = 0.055          # above this the optional heavy checks are skipped
SPEND_HARD = 0.078          # the main loop stops here
# b10: room for the light model's own reasoning: at 3000 (main loop) and 4000 (brief) tokens some replies were cut
# before any text or call (arena r11/r14: 9-18 main-loop replies a run, each a lost turn); unused room costs nothing
PILOT_TOKENS = 6000
BRIEF_TOKENS = 6000
HEAVY_ROOM = 0.02           # room a heavy-model request needs under SPEND_HARD
RETRY_WAITS = (4, 10, 20, 35)
RETRY_TOTAL = 45

SHOW_CHARS = 450            # longest string of a result shown to the pilot
SHOW_LIMIT = 15000          # longest result text shown to the pilot
HISTORY_CHARS = 150000      # the pilot's chat is shortened above this
# b4 (27-know b15): one deep cut of the chat's history, then none until it has grown TRIM_GAP characters more; a cut
# message never changes again, so the prompt cache keeps reading it
TRIM_GAP = 60000
TRIM_KEEP = 3               # the newest turns kept as they are
DIGEST_KEEP = 600           # an older result this short stays whole
DIGEST_SAY = 400            # an older pilot text is cut to this
DIGEST_ROWS = 20            # listings named in one digest
DIGEST_ROW = 220            # characters a listing in a digest
DIGEST_FIELD = 50           # characters a field in a digest
GATE_ROUNDS = 3
AUTO_PAGES = 6              # result pages the reader may fetch itself in one episode
CHEAPER_OPENS = 3

TASK_SECONDS = 300.0        # the validator's limit for one episode
TIME_MARGIN = 20.0
CALL_FLOOR = 6.0            # no model request starts with less room than this
ASK_BY = 165.0              # no interview starts after this many seconds
QUESTION_BY = 210.0         # no question is sent after this
WIND_DOWN = 200.0           # the status asks for the checkout from here
LAST_TURN = 255.0           # no pilot turn starts after this (unless the cart is empty)
EMPTY_ROOM = 12.0
LATE_BY = 190.0             # the late question is skipped after this
LATE_STEPS = 8
SCARCE_MESSAGES = 2
LIST_SIZE = 8               # properties named in one question
HEARD_LABELS = 3            # b17: after a reply naming no need, the next list starts at this place of the old one
ASK_STEP = 16               # b3 (27-know b17): the interview starts by this step at the latest
TABLE_KEYS = 24             # b3 (27-know b17): attribute keys kept per category (the planner is shown PLAN_SHOWN)
PLAN_SHOWN = 14
TIGHT_STEPS = 45            # b15: a step budget this small gets the lighter code-side value search

LANE_GUARDS = 3             # most-carried keys moved into an item's first three questions
LANE_GUARD_SHARE = 0.0
LANE_GUARD_GAP = 0.34
OPEN_EACH = 6               # listings read per item before planning (4 with three items or more)

CAP_ARG = '(?:^|_)(?:cap|max|maximum|ceiling|under|up_?to)(?:_|$)'
CAP_NOT = 'result|row|page|count|size|num|item|step'
SHELF_ARG = 'section|department|categor|label|shelf|aisle|(?<!age)(?:^|_)group(?:_name|_label)?$'
LIST_MODE = 'field|filter|facet|brows|categor|list|structured'
AMOUNT = '\\d[\\d.,]*\\s*[a-z"\']{1,12}'
FLOOR = '\\b(?:at least|no (?:less|fewer) than|min(?:imum)?)\\s+(' + AMOUNT + ')\\s*$'
QUERY_WORDS = '(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword'
NOT_PROPERTY = '\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b'
CATCH_ALL = '^(?!.*\\b(?:types?|styles?|kinds?|shapes?)\\b).*\\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\\w*|accessor\\w*|extras?|misc\\w*)\\b'
FIT_CODE = '[A-Z][A-Za-z]{0,5}[ -]?\\d{3,6}'      # b7: also hyphenated model codes (MOZ-7440)
PART_CODE = '[A-Z]{1,4}\\d{3,6}'
SHOP_FIELD = 'seller|shop|store|vendor|brand|maker|merchant'
LIMIT_WORDS = '\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b'

_LLM = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_LLM_SHORT: dict[int, ProxyClient] = {}
_STORE = ProxyClient(timeout=95, max_retries=3, retry_delay=1)

# ----------------------------------------------------------------------------------------------------------- prompts
P_PILOT = """You are Arc. You shop for one shopper in an online store, and the store's tools are your only way to act. An episode scores 1 or 0, and only an exactly right ending scores 1: either a purchase holding every requested item at its requested quantity, each meeting every requirement in force at checkout, inside the budget; or, when the store really has nothing that meets everything, an ending without a purchase done in the way the request asks.

THE STORE
- Each tool call uses one step and steps are limited; thinking is free. Every store has its own tool names, argument names and result fields: read each tool's description and schema and use the exact argument names and allowed values.
- Searches may be narrow: a category filter, a word mode that reads only the first words of a query, a few rows a page, a token or offset for the next page. When the answer depends on seeing every match, compare the reported total with the rows you saw and page on with the exact token or offset the store gives. Word queries work best short and value-first (the wanted value, then the product noun). A filtered search with no rows usually has a wrong filter value or parameter: retry with the label exactly as result rows spell it, or with another filter parameter.
- A search row is a summary. Stores split a listing's record in different ways: one view shows its attributes (properties), another only its offer and description, and a side-by-side view can carry either. Once Arc knows which tool shows attributes, the ledger names it. Judge each requirement on a property from those attributes, so read them for every listing you might buy; one call that shows several listings saves steps when it shows what you need. Do not open a listing again when nothing changed since you read it.
- Arc's interviewer sends every question to the shopper; you never message the shopper yourself. Its answers come into the conversation and go into the ledger.
- The request may also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Arc itself carries out the ones due just before the order or the ending when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do the others with the store's tools (questions about a product go through Arc's interviewer).

WHEN A LISTING FITS
- The listing's category label decides the kind of product: only the label matching the requested item as a whole counts, never a neighbouring label.
- A requirement on a property is met when the listing's attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value containing the required value's words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value in other words does not count (an abbreviation or a code is not the word it stands for, and "teal green" is not "sea green"). Convert units before comparing numbers; limits include the edge value; a range must lie wholly inside a limit.
- An attribute whose key names the requirement's property (by that name or a plain synonym) decides: when it says something else, the listing fails whatever the title or other keys say. A key for another property, even a related one (a handle colour is not the body's colour), decides nothing about the requirement. Arc adds [evidence] lines after opened listings, naming the keys whose names match a requirement.
- When the listing has no attribute for the property, the wanted value stated under another attribute (a list of styles or features that includes it, for example) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor does a bare figure whose text does not say what it measures.
- When the category label itself states a requirement's value (a season or an audience in the label), the category meets that requirement.
- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you onward to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of the chain was opened. When the wanted value is anything but that other listing's value, the rule rules out only listings with that exact value, and never end without a purchase because of it alone.
- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.
- When one requested item has to fit another (an accessory for a device), choose the two together from the listings' own words (a list of what it fits, a model or series code both pages carry). When no accessory meeting its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.
- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") only break ties.
- Never buy a listing the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.

CHOOSING
- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows often come in no price order, and Arc reads the next pages of a short result itself), check candidates from the lowest price up, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates remain to check.
- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.
- Otherwise any listing meeting everything is right: do not spend steps hunting for a better one.
- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.
- A request for several different items needs one cart line per item, each at its requested quantity.

CHANGES ON THE WAY
- The shopper or the store can change things: a new budget, a requirement added or dropped, a new price, a listing sold out. Changes come inside tool results, often when you try to check out (a refusal); a note counts whichever tool's result shows it, and the ledger lists the requirements in force. A refusal does not end the episode: apply every change, fix the cart and check out again.
- Read a chosen listing's record again shortly before checking out, because price and stock can change without notice.

ENDING
- Put exactly the chosen lines in the cart (take out anything else), look at the cart once, then check out with a call on its own.
- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request's wording for that ending, and when a remark is needed, keep it short and name no listing.
- Every reply must call at least one tool. Keep your text short: what you know, then the next move."""

P_BRIEF = """You turn a shopper's request into a task sheet for a shopping agent. Read the request and the store's tools, then return ONE JSON object (compact, no indentation, nothing else) with the keys below.

"items": one entry per distinct product to be bought, each {"name": the kind of product in the request's words, "shelf": the store's category label for it when a tool lists labels (copied exactly), or the category the request itself names for it ("a coat in rainwear" gives "rainwear"), else "", "quantity": units of that one listing (integer; 1 when not stated; a pair or set sold as one listing counts as 1), "musts": the firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list}.
  - Name a property by the request's own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); never swap in a neighbouring property.
  - A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): the words describing that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product.
  - Never put price, budget, stock or quantity into musts.
  - The story around the request (who will use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property's value or limit outright. Words that are part of what the product is called (a season or an audience named with the product) belong to "name", not to musts.
  - A use or occasion the request makes a property of the product itself ("blazers for business wear", "a dress made for parties", "a bag, and it is for a wedding", "the occasion has to be party") IS a must on that property ("occasion: business"), unless the store's category label already names it. A particular person's event, a person or a setting the product is bought for ("to wear at his wedding", "for a colleague"), and how it should sit, feel or look in use, stay context.
  - A product named in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.
"budget": the most the shopper will pay in total, as a number in the request's own unit, or null.
"budget_unit": that unit in words (for example "cents"), or null.
"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.
"in_stock_only": true unless the request clearly accepts items that are not available.
"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.
"pick_measure": what is maximised or minimised for "most"/"least", else "".
"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.
"steps": how many steps, moves or tool calls the request allows (integer), or null.
"lookups": the musts (copied exactly) whose value must first be read from another listing the request points at, else [].
"rules": every instruction the request gives about how to work or how to end, each in at most 12 words.
"wishes": things the shopper would merely like.
"ask_first": things the request explicitly says to check with the shopper before choosing (empty when none).
"process": the steps the request requires besides finding and buying the right items (getting the shopper's approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.
"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.
"fits": {"item": item number, "with": item number} for each requested item the request says must work with, fit or suit ANOTHER requested item (the one the agent picks); [] when none.
"roles": the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing's attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing's current offer, such as its price and stock, when that is another tool than "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).
"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.
"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.
"""

P_MODES = """The store tool <TOOL> both checks out and ends without a purchase, depending on its arguments. Its schema: <SCHEMA>
Return ONE JSON object and nothing else: {"buy_args": the fixed arguments that check out, "refuse_args": the fixed arguments that end without a purchase} (only the arguments that choose the mode)."""

P_PLAN = """You pick the questions a shopping agent will send to a shopper. Each shopper holds a few firm preferences they did not write down, each about one property of one requested product. A question reveals such a preference only when it names that exact property. A property the shopper has no view on gets "don't mind", and one the shopper never thinks about gets a puzzled reply; every property named uses part of a message.

You are given the request, the items with what the request already settles for each, and a numbered list of the attribute keys seen on opened listings of each item's category, with how many of those listings carry the key and some example values. An unwritten preference can also sit on a property no listed key names.

Rank the properties in this order:
0. What the request itself asks the agent to check with the shopper (under ASK FIRST when there is any), each named by the property it concerns.
1. Hinted properties. The request's story often points at an unwritten preference without its value: a sentence about something the shopper cannot judge or leaves to the agent, someone whose taste decides, the product's looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence name the one property it points at and the item it concerns, in the story's order. Use a listed key of that item for it when one fits the hint (a hint about who wears it: the key for gender or audience; about its looks: the listed key for colour, pattern or finish most of its listings carry); a property no listing of the item carries goes after the item's three most carried keys.
2. Then the listed keys: favour the properties a buyer of that kind of product chooses first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and the keys most of the item's listings carry; once five or more listings were read, avoid keys seen on only one or two of them.
   The main specification of that kind of product (what its sellers put in a listing's name and a buyer states first) counts even when no listed key names it or a key names it in other words: name it by its usual name, rank it among the listed keys by how much a buyer cares about it, and mark it "spec": true.

Leave out: a property the request already settles for that item (a requirement names that property; but a key whose values only repeat a required value, such as a pattern key showing the required decoration, is NOT settled), a second name for a property already chosen, a key whose values each list several purchase options at once, identifiers, model names, warranty and other logistics. Pack sizes, counts and quantities per pack are properties like any other, and a key marked [not the stated ...] is a different property from that requirement, whatever words its name shares with it. Rank each item on its own: its hinted properties first, then its listed keys.

"label" is the property's plain name: for a listed key, all of the key's words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); the name only, with no example values, no brackets and no second property.

Pick up to <COUNT> for each item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story's words it comes from, or an empty string", "spec": true only for a main specification no listed key names}]}, with each item's picks in rank order"""

P_HEAR = """A shopping agent asked a shopper some questions. Read every answer and return ONE JSON object and nothing else:
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value must be taken from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants them asked one at a time; not answers that only say they did not follow it or want it in simpler words], "budget": new total limit as a number or null}
- "needs": each firm preference an answer states, with its exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which can differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer gives it, keeping its nouns and units ("at least 3 shelves", not "at least 3"). The value is only what a listing could state, from the wanted value or its limit word to its end, without the answer's framing around it ("oval" for "matching oval", "settled, with oval" or "oval and nothing else"; "at least 30 cm" for "equal to at least 30 cm").
- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop's product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.
- "free": the property of each question the shopper said they do not mind about.
- Folksy or roundabout wording that names a value still states that value. Wording that refuses to bargain over a value (it will not be traded, given up or changed) states that value as wanted; write "not X" only when the answer rejects X itself."""

P_NOTE = """The store showed this note while a shopping agent was working (the shopper or the store may have written it). Return ONE JSON object and nothing else:
{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}
A replaced requirement is a drop of the old one plus an add of the new one. Report every change the note makes."""

P_READING = """a listing's attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value's words counts, such as "deep teal" for teal, limits include the edge); a key for another property, even a related one (a handle colour is not the body's colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing's value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit."""

P_CHECK = """You are the final check before a shopping agent checks out. From the records shown, decide whether this cart is exactly right.
How to read the records: a requirement is met when the listing's attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value's words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and when it says something else the listing fails; a key for another property, even a related one (a handle colour is not the body's colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing's value counts as met unless the records show that same value.
The cart: every requested item must be its own cart line at the requested quantity, nothing else may be in the cart, nothing may be shown out of stock, and the total of price times quantity over all lines must be within the budget. A listing's newest record wins.
When the pick rule is cheapest, most or least, also look at the other listings shown: when one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.
Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Give ok=false only for a problem the records clearly show or a requirement no record addresses."""

P_AUDIT = """A shopping agent wants to end without buying because it believes nothing fits. Check the listing records below against the requirements, reading them like this: """ + P_READING + """
Records marked ROW were only seen as a search result row: their page (the attributes) was not read, so they can be neither passed nor failed on a property the row does not show.
Return ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}"""

P_UNMIX = """A shopper's request was read into firm requirements on a product. Some of them take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. From the request's wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}"""

P_TRAIL = """A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing's page can send the reader on to a further listing for that property. The order counts only once the agent has opened in full every listing along each such chain. Follow each requirement's chain from its own description, link by link, through the pages' own words, using the records below (OPENED: the agent opened that page; ROW: seen only as a search result row, its page unread).
Return ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing, "find": "for a link not found: search words (its shop and the values the description gives)"}], "done": true when the last link is an OPENED page that states the property's value itself (also when the requirement asks for a value other than or unlike that one: the page's value is what it is measured against), else false, "value": the property's value as that last page states it, or ""}]}
A record is a link only when it matches the whole description of that link: the shop or brand, the kind of product AND every property value the description gives (for example "a 10 pieces multipack" rules out the same shop's listing with 10 capsules and no multipack); when no record matches all of it, that link is not found yet, even if the same shop sells other listings of that kind. Give "find" for such a link: the shop's name and the description's values, as a few search words.
List each chain's links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown."""

P_GATE = """A shopping agent is about to <STAGE>. The shopper's request asks for the steps below along the way. From the numbered RECORD of what the agent has done, decide whether each step due before this happens is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product's properties do not do a step.
When a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.
Return ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}"""

P_SWAP = """A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, reading them like this: """ + P_READING + """
Return ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line's item, at most 6], "reason": "one sentence"}"""

P_SHAPE = """A store tool returned the result below. Name its parts by the exact keys the result and the tool's parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing's price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}"""

QUIET_REMARK = 'Sorry, nothing in the store meets all of your requirements, so I am not buying anything.'

# ----------------------------------------------------------------------------------------------- text and json tools

def to_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        return str(value)


def clip(value: Any, limit: int, depth: int = 0) -> Any:
    """The value with every string cut at `limit` characters (nested containers kept, 8 levels deep)."""
    if depth > 7:
        return '...'
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + '...'
    if isinstance(value, dict):
        return {k: clip(v, limit, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [clip(v, limit, depth + 1) for v in value]
    return value


def parse_object(text: Any) -> dict[str, Any] | None:
    """The JSON object a model reply holds: the whole text, a fenced block, or the outermost braces."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    fenced = re.match('^```[A-Za-z]*\\s*(.*?)\\s*```$', text, re.S)
    if fenced:
        text = fenced.group(1)
    for candidate in (text, None):
        if candidate is None:
            left, right = text.find('{'), text.rfind('}')
            if not 0 <= left < right:
                return None
            candidate = text[left:right + 1]
        try:
            got = json.loads(candidate)
        except ValueError:
            continue
        return got if isinstance(got, dict) else None
    return None


def plain(key: Any) -> str:
    """A key or name in lower-case words: camel case, underscores, dots and dashes become spaces."""
    key = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(key))
    return re.sub('[\\s_.\\-]+', ' ', key).strip().lower()


def snake(key: Any) -> str:
    return plain(key).replace(' ', '_')


def stems(text: Any) -> frozenset[str]:
    """Words with a plural -s dropped (words of four letters or more)."""
    return frozenset(w[:-1] if len(w) > 3 and w.endswith('s') else w for w in plain(text).split())


def roots(text: Any) -> set[str]:
    """A name's content words, -es / -s plurals folded. Joining words are left out by their shape, not by a list
    (27-know b2): in a name of three words or more a word under four letters without a digit carries no kind; in a
    short name ("CPU", "Sun Hat") every word counts."""
    words = plain(text).split()
    out = set()
    for w in words:
        if re.search('(?:ch|sh|ss|x|z)es$', w):
            w = w[:-2]
        elif len(w) > 3 and w.endswith('s') and not w.endswith('ss'):
            w = w[:-1]
        if len(w) >= 4 or re.search('\\d', w) or (len(w) == 3 and len(words) <= 2):
            out.add(w)
    return out


def spellings(text: Any) -> frozenset[str]:
    """A name's words with plurals and the British -our folded, so "colours" and "color" compare equal."""
    out = []
    for w in plain(text).split():
        if len(w) > 3 and w.endswith('s') and not w.endswith('ss'):
            w = w[:-1]
        out.append(re.sub('our$', 'or', w))
    return frozenset(out)


def same_property(prop: frozenset[str], key: frozenset[str]) -> bool:
    """Whether a key names a property: equal word sets, the property's words inside the key's, or a key of two words
    or more inside the property."""
    if not prop or not key:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, (str, dict)) and value:
        return [value]
    return []


def as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.0+)?\\s*', value):
        return int(float(value))
    return None


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def as_number(value: Any) -> float | None:
    if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', value):
        return float(value)
    return float(value) if is_number(value) else None


def looks_like_id(value: Any, named: bool = False) -> bool:
    if not isinstance(value, str) or not 3 <= len(value.strip()) <= 64 or ' ' in value.strip():
        return False
    return named or bool(re.search('\\d', value))


def guess_id_key(rows: list[dict[str, Any]]) -> str | None:
    """The key of rows that holds a distinct id on every row; one named "id" and longer values win."""
    best = None
    for key in rows[0]:
        values = [row.get(key) for row in rows]
        if not all(looks_like_id(v, 'id' in str(key).lower()) for v in values):
            continue
        values = [v.strip() for v in values]
        if len(rows) > 1 and len(set(values)) < len(values):
            continue
        score = ('id' in str(key).lower(), sum(len(v) for v in values) / len(values))
        if best is None or score > best[0]:
            best = (score, key)
    return best[1] if best else None


def mentions(node: Any, ident: str, depth: int = 0) -> bool:
    """Whether a list of rows somewhere in the node has a row with `ident` as one of its string values."""
    if depth > 5:
        return False
    if isinstance(node, list):
        if any(isinstance(v, dict) and ident in [str(x).strip() for x in v.values() if isinstance(x, str)] for v in node):
            return True
        return any(mentions(v, ident, depth + 1) for v in node)
    if isinstance(node, dict):
        return any(mentions(v, ident, depth + 1) for v in node.values())
    return False


def attributes(node: Any) -> dict[str, Any] | None:
    """The record's attribute block: its largest nested object with two or more scalar values."""
    if not isinstance(node, dict):
        return None
    best = None
    for value in node.values():
        if isinstance(value, dict) and len(value) >= 2:
            scalars = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
            if len(scalars) >= 2 and (best is None or len(scalars) > len(best)):
                best = scalars
    return best


def leaves(node: Any) -> list[str]:
    stack, out = [node], []
    while stack:
        part = stack.pop()
        if isinstance(part, (dict, list)):
            stack += list(part.values()) if isinstance(part, dict) else part
        else:
            out.append(str(part))
    return out


def word_in(word: str, text: str) -> bool:
    return bool(re.search('(?<![a-z0-9])' + re.escape(word) + '(?![a-z0-9])', text))


def shop_names(row: dict[str, Any], short: int = 40) -> list[str]:
    return [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= short and re.search(SHOP_FIELD, str(k), re.I)]


# b6: some worlds send each listing as one line of text ("id_key: p-.. | title: .. | price: 4998 | facts: colour=black;
# storage=128gb | ..."). Read in code, that line is a record like any other: its fields are split on " | " at a
# "key: " start (a part without one belongs to the field before), and a field of "k=v; k=v" parts is an attribute block.
FIELD_START = re.compile('^([A-Za-z_][A-Za-z0-9_]{0,40}): (.*)$', re.S)


def field_value(text: str) -> Any:
    text = text.strip()
    if text in ('True', 'False', 'true', 'false'):
        return text.lower() == 'true'
    if text in ('None', 'null'):
        return None
    if re.fullmatch('-?\\d+', text):
        return int(text)
    if re.fullmatch('-?\\d+\\.\\d+', text):
        return float(text)
    parts = text.split('; ')
    if len(parts) >= 2 and '=' in parts[0] and sum('=' in p for p in parts) >= 2:
        block: dict[str, str] = {}
        last = None
        for part in parts:
            key, sep, value = part.partition('=')
            if sep and key.strip() and len(key.strip()) <= 40 and key.strip() not in block:
                last = key.strip()
                block[last] = value.strip()
            elif last is not None:
                block[last] += '; ' + part
        return block
    return text


def record_line(text: str) -> dict[str, Any] | None:
    """A listing written as one line of "key: value" fields, as a record; None when the text is not one."""
    parts = text.split(' | ')
    if len(parts) < 3:
        return None
    fields: dict[str, str] = {}
    last = None
    for part in parts:
        start = FIELD_START.match(part)
        if start and start.group(1) not in fields:
            last = start.group(1)
            fields[last] = start.group(2)
        elif last is not None:
            fields[last] += ' | ' + part
        else:
            return None
    if len(fields) < 3:
        return None
    return {k: field_value(v) for k, v in fields.items()}


def unfold(value: Any, depth: int = 0) -> Any:
    """The observation with every listing written as a line of text turned into a record."""
    if depth > 6:
        return value
    if isinstance(value, str):
        record = record_line(value) if ' | ' in value else None
        return record if record is not None else value
    if isinstance(value, dict):
        return {k: unfold(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [unfold(v, depth + 1) for v in value]
    return value


def client_for(seconds: float) -> ProxyClient:
    """The model client whose timeout fits the time left: the usual one, or one timing out in whole 5 s steps."""
    if seconds >= 50:
        return _LLM
    limit = max(5, int(seconds) // 5 * 5)
    if limit not in _LLM_SHORT:
        _LLM_SHORT[limit] = ProxyClient(timeout=limit, max_retries=1, retry_delay=1)
    return _LLM_SHORT[limit]


# --------------------------------------------------------------------------------------------------- tool schemas

def type_list(schema: dict[str, Any]) -> list[str]:
    kind = schema.get('type')
    if isinstance(kind, str):
        return [kind]
    return [k for k in kind or [] if isinstance(k, str)]


def is_type(value: Any, kind: str) -> bool:
    if kind == 'string':
        return isinstance(value, str)
    if kind == 'integer':
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == 'number':
        return is_number(value)
    if kind == 'boolean':
        return isinstance(value, bool)
    if kind == 'array':
        return isinstance(value, list)
    if kind == 'object':
        return isinstance(value, dict)
    if kind == 'null':
        return value is None
    return True


def _coerced(value: Any, kinds: list[str]) -> Any:
    """The value converted to the first of the schema's types it can plainly become (unchanged when none)."""
    for kind in kinds:
        if kind == 'integer' and isinstance(value, float) and value.is_integer():
            return int(value)
        if kind in ('integer', 'number') and isinstance(value, str) and re.fullmatch('\\s*-?\\d+(\\.\\d+)?\\s*', value):
            number = float(value)
            return int(number) if kind == 'integer' or number.is_integer() else number
        if kind == 'string' and is_number(value):
            return str(value)
        if kind == 'boolean' and isinstance(value, str) and value.strip().lower() in ('true', 'false'):
            return value.strip().lower() == 'true'
        if kind == 'array' and not isinstance(value, list) and value is not None:
            return [value]
    return value


def conform(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    """The value fitted to a JSON schema where that is safe (types converted, enum case, long text cut); every
    problem left is added to `faults`."""
    if not isinstance(schema, dict):
        return value
    kinds = type_list(schema)
    if kinds and not any(is_type(value, k) for k in kinds):
        value = _coerced(value, kinds)
        if not any(is_type(value, k) for k in kinds):
            faults.append(f"{where} should be {'/'.join(kinds)}")
            return value
    options = schema.get('enum')
    if isinstance(options, list) and value not in options:
        if isinstance(value, str):
            same = [o for o in options if isinstance(o, str) and o.lower() == value.strip().lower()]
            if same:
                value = same[0]
        if value not in options:
            faults.append(f'{where}: allowed values are {options}')
    if isinstance(value, str):
        if isinstance(schema.get('maxLength'), int) and len(value) > schema['maxLength']:
            value = value[:schema['maxLength']]
        if isinstance(schema.get('minLength'), int) and len(value) < schema['minLength']:
            faults.append(f'{where} is too short')
    if is_number(value):
        if is_number(schema.get('minimum')) and value < schema['minimum']:
            faults.append(f"{where} is below {schema['minimum']}")
        if is_number(schema.get('maximum')) and value > schema['maximum']:
            faults.append(f"{where} is above {schema['maximum']}")
    if isinstance(value, list):
        if isinstance(schema.get('maxItems'), int) and len(value) > schema['maxItems']:
            faults.append(f"{where} allows at most {schema['maxItems']} entries")
        if isinstance(schema.get('minItems'), int) and len(value) < schema['minItems']:
            faults.append(f"{where} needs at least {schema['minItems']} entries")
        if isinstance(schema.get('items'), dict):
            value = [conform(v, schema['items'], f'{where}[{i}]', faults) for i, v in enumerate(value)]
    if isinstance(value, dict):
        props = schema.get('properties') if isinstance(schema.get('properties'), dict) else {}
        if schema.get('additionalProperties') is False:
            faults += [f'{where}.{name} is not an allowed argument' for name in value if name not in props]
        faults += [f'{where}.{name} is required' for name in schema.get('required') or [] if name not in value]
        value = {k: conform(v, props.get(k), f'{where}.{k}', faults) if k in props else v for k, v in value.items()}
    return value


def free_text_keys(schema: dict[str, Any]) -> list[str]:
    """The schema's string parameters without a closed list of values."""
    props = schema.get('properties') or {}
    return [k for k, v in props.items() if isinstance(v, dict) and 'string' in type_list(v) and 'enum' not in v]


# ----------------------------------------------------------------------------------------------- fit codes in text

def model_codes(text: Any) -> list[str]:
    return [re.sub('\\s+', ' ', c).strip().lower() for c in re.findall('\\b' + FIT_CODE + '\\b', str(text or ''))]


def fit_marks(text: Any) -> dict[str, Any]:
    """What a listing's text says about fitting: its own part code, the model codes it carries, the codes it works
    with, the parts it says other codes fit, and a part code it sends the reader to for more."""
    text = str(text or '')
    own = re.search('[\\[(]\\s*(?:part\\s+)?(' + PART_CODE + ')\\s*[\\])]', text, re.I)
    codes: list[str] = []
    for hit in re.finditer('\\bmodel\\s*(?:no\\.?|number|#)\\s*[:.]?\\s*(' + FIT_CODE + ')\\b', text, re.I):
        codes += model_codes(hit.group(1))
    for hit in re.finditer('\\b(?:also sold as|also known as|sold also as)\\s+([^.;]+)', text, re.I):
        codes += model_codes(hit.group(1))
    works: list[str] = []
    for hit in re.finditer('\\b(?<!not )(?<!never )(?:works with|compatible with|fits)\\s*:?\\s+([^.;]+)', text, re.I):
        if not re.search('\\bpart\\s+' + PART_CODE + '\\s+also\\s+$', text[max(0, hit.start() - 40):hit.start()], re.I):
            works += model_codes(hit.group(1))
    also = [(hit.group(1).upper(), model_codes(hit.group(2))) for hit in re.finditer('\\bpart\\s+(' + PART_CODE + ')\\s+also fits\\s*:?\\s*([^.;]+)', text, re.I)]
    more = re.search('\\b(?:more|details|see|read)\\s+(?:on|at|under|about)?\\s*(?:the\\s+)?part\\s+(' + PART_CODE + ')', text, re.I)
    return {'own': own.group(1).upper() if own else '', 'codes': codes, 'works': works, 'also': also, 'more': more.group(1).upper() if more else ''}


class Item:
    """One requested product: its name, store label, quantity, the request's requirements and the shopper's answers."""

    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []

    def live(self) -> list[str]:
        return [m for m in self.musts + self.told if m not in self.dropped]


class Part:
    """A part of a Session; `s` reaches the session and through it every other part."""

    def __init__(self, session: 'Session'):
        self.s = session


# ------------------------------------------------------------------------------------------------------- Models

class Models(Part):
    """Model requests: the light/heavy choice, retries with a time-aware pause, spend and call counts, and the
    episode clock."""

    def __init__(self, session: 'Session'):
        super().__init__(session)
        self.started = time.monotonic()
        self.cost = 0.0
        self.calls = 0
        self.waited = 0
        self.no_credit = False

    def clock(self) -> float:
        return time.monotonic() - self.started

    def room(self) -> float:
        return TASK_SECONDS - TIME_MARGIN - self.clock()

    def judges(self, limit: float = SPEND_SOFT) -> list[str]:
        """Models an optional check may use: heavy first while the spend allows it, light while under SPEND_HARD."""
        if self.cost <= limit and self.cost + HEAVY_ROOM <= SPEND_HARD:
            return [HEAVY, LIGHT]
        return [LIGHT] if self.cost <= SPEND_HARD else []

    def chat(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        s = self.s
        for model in models:
            if self.calls >= MODEL_CALLS:
                return None
            if self.room() < CALL_FLOOR:
                s.log('late_call_skipped', at=round(self.clock(), 1))
                return None
            self.calls += 1
            tries = 0
            while True:
                message, status = self.request(model, messages, max_tokens, tools)
                if message is not None:
                    return message
                retryable = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                limit = 1 if status in (None, 408) else len(RETRY_WAITS)
                if not retryable or tries >= limit or self.waited + RETRY_WAITS[tries] > RETRY_TOTAL:
                    break
                if self.room() - RETRY_WAITS[tries] * 1.2 < CALL_FLOOR:
                    break
                pause = RETRY_WAITS[tries] * (0.8 + 0.4 * random.random())
                self.waited += RETRY_WAITS[tries]
                tries += 1
                time.sleep(pause)
            if status == 402:
                self.no_credit = True
                return None
        return None

    def request(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        sent = messages
        if model.startswith('anthropic/') and len(messages) > 1:
            # cache marks on the first message and the newest one
            sent = [dict(m) for m in messages]
            for index in (0, len(sent) - 1):
                content = sent[index].get('content')
                if isinstance(content, str) and content:
                    sent[index]['content'] = [{'type': 'text', 'text': content, 'cache_control': {'type': 'ephemeral'}}]
        body: dict[str, Any] = {'model': model, 'messages': sent, 'max_tokens': max_tokens, 'temperature': 0}
        if tools:
            body['tools'] = tools
            body['tool_choice'] = 'auto'
        # b13: the model is called through the client's documented post() (the arena pre-check counts no other form as
        # a model call). post() returns no status, so a failure that used up nearly the whole timeout counts as a
        # timeout (408) and any other as an upstream failure (502); an error object in the reply still gives its code
        client = client_for(self.room())
        started = time.monotonic()
        data = client.post('/inference/chat/completions', json_data=body)
        if isinstance(data, dict) and data.get('choices'):
            usage = data.get('usage') or {}
            if is_number(usage.get('cost')):
                self.cost += float(usage['cost'])
            message = data['choices'][0].get('message') or {}
            if message.get('content') or message.get('tool_calls'):
                return message, 200
            self.s.log('llm_fail', model=model, status=0)
            return None, 0
        error: dict[str, Any] = {}
        limit = float(getattr(client, 'timeout', 50) or 50)
        status = 408 if time.monotonic() - started >= 0.9 * limit else 502
        if isinstance(data, dict) and isinstance(data.get('error'), dict):
            status = as_int(data['error'].get('code'))
            error = {'body': data['error'].get('message')}
        self.s.log('llm_fail', model=model, status=status, body=str(error.get('body') or '')[:200])
        return None, status

    def ask(self, prompt: str, models: list[str], max_tokens: int = 1800) -> dict[str, Any] | None:
        """One JSON answer to a one-message prompt, trying the models in turn."""
        for model in models:
            message = self.chat([{'role': 'user', 'content': prompt}], [model], max_tokens)
            got = parse_object((message or {}).get('content'))
            if got is not None:
                return got
            if self.s.store.broke or self.no_credit or self.calls >= MODEL_CALLS:
                return None
        return None


# -------------------------------------------------------------------------------------------------------- Store

class Store(Part):
    """The world's tools: their schemas, the job of each (from the brief), store calls and the step count."""

    def __init__(self, session: 'Session', view: dict[str, Any], binding: dict[str, Any]):
        super().__init__(session)
        self.binding = dict(binding)
        self.tools = [t for t in view.get('tools') or [] if isinstance(t, dict) and isinstance(t.get('function'), dict) and isinstance(t['function'].get('name'), str)]
        self.schemas = {t['function']['name']: t['function'].get('parameters') if isinstance(t['function'].get('parameters'), dict) else {'type': 'object', 'properties': {}} for t in self.tools}
        steps = as_int(view.get('max_steps'))
        if steps is None and isinstance(view.get('max_steps'), float):
            steps = int(view['max_steps'])
        self.max_steps = steps if steps and steps > 0 else 30
        self.step = 0
        self.finished = bool(view.get('finished'))
        self.broke = False
        self.roles: dict[str, str] = {}
        self.buy_args: dict[str, Any] = {}
        self.refuse_args: dict[str, Any] = {}
        self.id_args: set[tuple[str, str]] = set()
        self.compare_ok = 0
        self.compare_bad = 0

    def left(self) -> int:
        return self.max_steps - self.step

    def tool(self, job: str) -> str | None:
        return self.roles.get(job)

    def props(self, name: str | None) -> dict[str, Any]:
        return (self.schemas.get(name or '') or {}).get('properties') or {}

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        """One store call. Step clashes are repaired from the store's own count; None when the call failed."""
        s = self.s
        payload = dict(self.binding)
        payload['step'] = self.step + 1
        payload['action'] = {'name': name, 'args': args}
        for _ in range(4):
            result = _STORE.post_verbose('/environment/call', payload)
            if isinstance(result.data, dict):
                data = result.data
                if 'observation' in data:
                    data = dict(data, observation=unfold(data['observation']))
                self.step = data['step'] if isinstance(data.get('step'), int) else self.step + 1
                if data.get('finished'):
                    self.finished = True
                seen = data.get('observation')
                if name in self.page_tools() and not (isinstance(seen, dict) and seen.get('error')):
                    for value in args.values():
                        s.mem.opened.update(v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str))
                return data
            error = result.error or {}
            status = error.get('status')
            body = str(error.get('body') or '')
            s.log('env_fail', status=status, body=body[:200])
            if status == 409:
                low = body.lower()
                if 'finish' in low or 'quarantin' in low or 'finaliz' in low:
                    self.finished = True
                    return None
                expected = re.search('expected (\\d+)', body)
                if expected:
                    self.step = int(expected.group(1)) - 1
                    payload['step'] = self.step + 1
                    continue
                if 'reuse' in low or 'different action' in low:
                    self.step += 1
                    payload['step'] = self.step + 1
                    continue
                return None
            if status in (400, 401, 402, 403, 404, 422):
                if status == 402:
                    self.broke = True
                return None
        return None

    def run(self, name: str | None, args: dict[str, Any]) -> dict[str, Any] | None:
        """A call fitted to the tool's schema first; None when it does not fit or the call failed."""
        if not name:
            return None
        faults: list[str] = []
        args = conform(args, self.schemas.get(name), 'args', faults)
        if faults:
            return None
        return self.call(name, args)

    # -- what the tools are for
    def page_tools(self) -> set[str]:
        page = self.roles.get('offer') or self.roles.get('open')
        return {t for t in (page, self.roles.get('compare')) if t}

    def offer_tool(self) -> str | None:
        return self.roles.get('offer') or self.roles.get('open')

    def readers(self) -> set[str]:
        return {self.roles.get(r) for r in ('search', 'open', 'offer', 'compare')} - {None}

    def enders(self) -> set[str]:
        return {self.roles.get('buy'), self.roles.get('refuse')} - {None}

    def single_target(self, tool: str | None) -> str | None:
        """The one string parameter a single-listing tool takes, when that is all it requires."""
        schema = self.schemas.get(tool or '') or {}
        texts = free_text_keys(schema)
        target = next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not tool or not target or any(k != target for k in schema.get('required') or []):
            return None
        return target

    def compare_target(self) -> str | None:
        schema = self.schemas.get(self.roles.get('compare') or '') or {}
        props = schema.get('properties') or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in type_list(v)), None)
        if not target or any(k != target for k in schema.get('required') or []):
            return None
        return target

    def compare_room(self) -> int:
        spec = self.props(self.roles.get('compare')).get(self.compare_target() or '') or {}
        if isinstance(spec.get('maxItems'), int):
            return spec['maxItems']
        if self.compare_bad:
            return max(2, min(self.compare_bad - 1, max(self.compare_ok, 2)))
        return max(4, self.compare_ok)

    def cart_sets(self, name: str) -> bool:
        """Whether the cart tool sets a line's count (a count with minimum 0 and no mode) rather than adding to it."""
        props = self.props(name)
        if any(isinstance(s, dict) and isinstance(s.get('enum'), list) and len(s['enum']) >= 2 for s in props.values()):
            return False
        return any(isinstance(s, dict) and 'integer' in type_list(s) and s.get('minimum') == 0 for s in props.values())

    def message_args(self, text: str) -> dict[str, Any] | None:
        name = self.roles.get('ask')
        schema = self.schemas.get(name or '') or {}
        props = schema.get('properties') or {}
        texts = free_text_keys(schema)
        target = next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if texts else None)
        if not target:
            return None
        args: dict[str, Any] = {target: text}
        for key in schema.get('required') or []:
            if key in args:
                continue
            spec = props.get(key) or {}
            if isinstance(spec.get('enum'), list) and spec['enum']:
                args[key] = spec['enum'][0]
            else:
                return None
        faults: list[str] = []
        args = conform(args, schema, 'args', faults)
        return None if faults else args

    def is_buy(self, name: str, args: dict[str, Any]) -> bool:
        return name == self.roles.get('buy') and all(args.get(k) == v for k, v in self.buy_args.items())

    def is_refuse(self, name: str, args: dict[str, Any]) -> bool:
        return name == self.roles.get('refuse') and all(args.get(k) == v for k, v in self.refuse_args.items())

    def labels(self) -> set[str]:
        """Every value the tools' schemas list as an option (category labels among them)."""
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
        for schema in self.schemas.values():
            walk(schema, 0)
        return found


# ------------------------------------------------------------------------------------------------------- Memory

class Memory(Part):
    """Every listing the episode has seen: search rows, full records (rows with an attribute block), ids, category
    labels, which views show attributes, and the shape of each search result."""

    def __init__(self, session: 'Session'):
        super().__init__(session)
        self.rows: dict[str, dict[str, Any]] = {}
        self.full: dict[str, dict[str, Any]] = {}
        self.order: list[str] = []
        self.id_key: str | None = None
        self.unavailable: set[str] = set()
        self.gone: dict[str, int] = {}
        self.opened: set[str] = set()
        self.facts_from: dict[str, bool] = {}
        self.tool_keys: dict[str, set[str]] = {}
        self.view_keys: dict[str, tuple[int, set[str]]] = {}
        self.shapes: dict[str, dict[str, Any] | None] = {}
        self.price_key: str | None = None
        self.stock_key: str | None = None
        self.vocab: tuple[set[str], set[str]] = (set(), set())
        self.vocab_mark: tuple[int, int] = (-1, -1)

    def store_words(self) -> tuple[set[str], set[str]]:
        """The store's own words (27-know b2): the words of attribute values and short fields of the listings seen,
        and the words of category labels. A pointer's description is matched on these (and on numbers), so its
        joining and pointing words never need a list."""
        mark = (len(self.rows), len(self.full))
        if self.vocab_mark != mark:
            values: set[str] = set()
            labels: set[str] = set()
            cat_key = self.category_key()
            for ident, row in self.rows.items():
                record = self.full.get(ident) or row
                for v in record.values():
                    if isinstance(v, str) and len(v.strip()) <= 40:
                        values |= set(re.findall('[a-z0-9]+', v.lower()))
                for v in (attributes(record) or {}).values():
                    values |= set(re.findall('[a-z0-9]+', str(v).lower()))
                if cat_key and isinstance(row.get(cat_key), str):
                    labels |= set(re.findall('[a-z0-9]{3,}', row[cat_key].lower()))
            self.vocab = (values, labels)
            self.vocab_mark = mark
        return self.vocab

    # -- ids and rows
    def ident(self, node: Any, in_list: bool = False, depth: int = 0) -> str | None:
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if self.id_key and looks_like_id(node.get(self.id_key), True):
            return node[self.id_key].strip()
        nested = any(isinstance(v, dict) for v in node.values())
        if not (in_list or depth == 0 or nested):
            return None
        found = guess_id_key([node])
        if found and ('id' in str(found).lower() or node[found].strip() in self.rows):
            return node[found].strip()
        return None

    def learn_rows(self, node: list[Any]) -> bool:
        rows = [n for n in node if isinstance(n, dict) and len(n) >= 3]
        if not rows or len(rows) < len(node) / 2:
            return False
        if self.id_key and all(looks_like_id(r.get(self.id_key), True) for r in rows):
            return True
        found = guess_id_key(rows)
        if found:
            self.id_key = found
            return True
        return False

    def absorb(self, observation: Any) -> None:
        """Merge every listing an observation shows into the rows (and the full records when it shows attributes)."""
        def walk(node: Any, depth: int, in_list: bool = False) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                ident = self.ident(node, in_list, depth)
                if ident:
                    if ident not in self.rows:
                        self.order.append(ident)
                        self.rows[ident] = {}
                    old = self.rows[ident]
                    if any(old.get(k) is True and v is False for k, v in node.items()):
                        self.unavailable.add(ident)
                    elif any(old.get(k) is False and v is True for k, v in node.items()):
                        self.unavailable.discard(ident)
                    self.rows[ident].update(node)
                    if attributes(node):
                        self.full[ident] = dict(self.rows[ident])
                    elif ident in self.full:
                        self.full[ident].update(node)
                    return
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                rows = self.learn_rows(node)
                for value in node:
                    walk(value, depth + 1, rows)
        walk(observation, 0)

    def records_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        if isinstance(observation, dict):
            ident = self.ident(observation)
            if ident:
                return [(ident, observation)]
            found: list[tuple[str, dict[str, Any]]] = []
            for value in observation.values():
                if isinstance(value, list):
                    found += [(i, r) for r in value if isinstance(r, dict) for i in [self.ident(r, True)] if i]
                elif isinstance(value, dict):
                    found += [(i, r) for v in value.values() if isinstance(v, list) for r in v if isinstance(r, dict) for i in [self.ident(r, True)] if i]
            return found
        if isinstance(observation, list):
            return [(i, r) for r in observation if isinstance(r, dict) for i in [self.ident(r, True)] if i]
        return []

    def record(self, ident: str) -> dict[str, Any] | None:
        return self.full.get(ident) or self.rows.get(ident)

    def record_text(self, ident: str) -> str:
        record = self.record(ident)
        return to_json(clip(record, 400))[:3000] if record else '(no record seen)'

    def others(self, skip: list[str], limit: int = 30) -> str:
        """Other listings seen, newest first (cheapest first, opened ones ahead, for a cheapest pick)."""
        lines = []
        order = list(reversed(self.order))
        if self.s.order.pick() == 'cheapest':
            order.sort(key=lambda i: (i not in self.full, self.price_of(i) if self.price_of(i) is not None else float('inf')))
        for ident in order:
            if ident in skip or self.sold_out(ident):
                continue
            lines.append(to_json(clip(self.record(ident), 160))[:900])
            if len(lines) >= limit:
                break
        return '\n'.join(lines)

    # -- price and stock
    def price_of(self, ident: str) -> float | None:
        value = (self.rows.get(ident) or {}).get(self.price_key) if self.price_key else None
        if isinstance(value, dict):
            value = next((v for v in value.values() if is_number(v)), None)
        return as_number(value)

    def sold_out(self, ident: str) -> bool:
        return ident in self.unavailable or ident in self.gone

    def buyable(self, ident: str) -> bool:
        if self.sold_out(ident):
            return False
        value = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        return not (value is False or (is_number(value) and value <= 0))

    # -- category labels and attribute keys
    def category_key(self) -> str | None:
        """The row key that holds the category label: short text on most rows, few distinct values, best when the
        values are options some tool lists."""
        counts: dict[str, set[str]] = {}
        present: dict[str, int] = {}
        for row in self.rows.values():
            for key, value in row.items():
                if isinstance(value, str) and 0 < len(value) <= 40 and not re.search('\\d{3}', value):
                    counts.setdefault(key, set()).add(value)
                    present[key] = present.get(key, 0) + 1
        total = max(1, len(self.rows))
        known = self.s.store.labels()
        best = None
        for key, values in counts.items():
            if present[key] < 0.8 * total or len(values) > 40:
                continue
            listed = sum(1 for v in values if v.strip().lower() in known) / len(values)
            mean = sum(len(v) for v in values) / len(values)
            score = (-listed, len(values), -present[key], -mean)
            if best is None or score < best[0]:
                best = (score, key)
        return best[1] if best else None

    def label_of(self, ident: str, key: str | None = None) -> str:
        key = key if key is not None else self.category_key()
        return str((self.rows.get(ident) or {}).get(key) or '') if key else ''

    def key_table(self) -> list[tuple[str, str, int, list[str]]]:
        """(category label, attribute key, listings carrying it, up to 4 example values) for the requested kinds,
        second names of one property merged, long free-text keys left out, 14 keys a label at most."""
        order = self.s.order
        cat_key = self.category_key()
        groups: dict[str, dict[str, Any]] = {}
        for ident in self.order:
            record = self.full.get(ident)
            if not record:
                continue
            block = attributes(record) or {}
            label = str(record.get(cat_key) or 'listings') if cat_key else 'listings'
            group = groups.setdefault(label, {'n': 0, 'keys': {}})
            group['n'] += 1
            done: set[str] = set()
            for key, value in block.items():
                name = plain(key)
                if not name or name in done:
                    continue
                done.add(name)
                slot = group['keys'].setdefault(name, {'n': 0, 'values': [], 'per': {}})
                slot['n'] += 1
                text = str(value).strip().lower()
                slot['per'][ident] = text
                if text not in slot['values']:
                    slot['values'].append(text)
        matched = {label for label in groups if order.wanted(label) or order.owner_of(label)}
        if matched:
            groups = {label: group for label, group in groups.items() if label in matched}
        table = []
        for label, group in sorted(groups.items(), key=lambda kv: -kv[1]['n']):
            kept: list[tuple[str, dict[str, Any]]] = []
            for name, slot in sorted(group['keys'].items(), key=lambda kv: (-kv[1]['n'], len(kv[0]))):
                values = list(slot['per'].values())
                if not values or self.free_text(values):
                    continue
                twin = next((spec for other, spec in kept if self.twins(other, spec, name, slot)), None)
                if twin is None:
                    kept.append((name, slot))
                    continue
                for ident, text in slot['per'].items():
                    twin['per'].setdefault(ident, text)
                    if text not in twin['values']:
                        twin['values'].append(text)
                twin['n'] = len(twin['per'])
            kept.sort(key=lambda kv: -kv[1]['n'])
            for name, slot in kept[:TABLE_KEYS]:
                table.append((label, name, slot['n'], [v[:40] for v in slot['values'][:4]]))
        return table

    @staticmethod
    def free_text(values: list[str]) -> bool:
        """Values that are no property to ask about: long text, purchase-option lists, or unique long phrases that
        share no words."""
        mean = sum(len(v) for v in values) / len(values)
        if mean > 40 or sum(1 for v in values if ' / ' in v) * 2 >= len(values):
            return True
        recur = sum(len(set(re.findall('[a-z]{3,}', v)) & set(re.findall('[a-z]{3,}', ' '.join(o for o in values if o is not v)))) > 0 for v in values)
        return len(values) >= 4 and len(set(values)) == len(values) and mean > 12 and recur * 2 < len(values)

    @staticmethod
    def twins(name: str, spec: dict[str, Any], other: str, slot: dict[str, Any]) -> bool:
        """Whether two keys are second names of one property: shared words and agreeing values on the same listings."""
        a, b = set(name.split()), set(other.split())
        if not a or not b or not a & b:
            return False

        def agree(x: str, y: str) -> bool:
            wx, wy = re.findall('[a-z0-9]+', x.lower()), re.findall('[a-z0-9]+', y.lower())
            short, long = (wx, wy) if len(wx) <= len(wy) else (wy, wx)
            return x == y or (bool(short) and set(short) <= set(long) and len(long) <= 2 * len(short) + 2)
        both = [i for i in spec['per'] if i in slot['per'] and spec['per'][i].strip() and slot['per'][i].strip()]
        same = sum(1 for i in both if agree(spec['per'][i], slot['per'][i]))
        if not (a <= b or b <= a):
            return len(both) >= 2 and same == len(both)
        if both:
            return same * 2 > len(both)
        return bool((set(spec['values']) & set(slot['values'])) - {''})

    def item_groups(self) -> list[list[str]]:
        """Seen listings grouped by label, the requested kinds' groups only (all when none matches), largest first."""
        order = self.s.order
        cat_key = self.category_key()
        groups: dict[str, list[str]] = {}
        for ident in self.order:
            label = str(self.rows[ident].get(cat_key) or '') if cat_key else ''
            groups.setdefault(label, []).append(ident)
        matched = [ids for label, ids in groups.items() if order.wanted(label) or order.owner_of(label)]
        pool = matched or list(groups.values())
        return sorted(pool, key=lambda ids: -len(ids))[:max(1, len(order.items))]

    # -- which views show what
    def facts_tool(self) -> str | None:
        store = self.s.store
        known = next((t for t, ok in self.facts_from.items() if ok and store.single_target(t)), None)
        if known:
            return known
        tool = store.roles.get('open')
        return tool if self.facts_from.get(tool or '') is not False and store.single_target(tool) else None

    def compare_bare(self) -> bool:
        """Whether the side-by-side view shows no attributes (or likely shows none)."""
        store = self.s.store
        seen = self.facts_from.get(store.roles.get('compare') or '')
        return seen is False or (seen is None and (self.facts_from.get(self.facts_tool() or '') is True or bool(store.roles.get('offer'))))

    def note_view(self, name: str, observation: Any) -> None:
        for ident, record in self.records_in(observation):
            keys = set(record)
            self.tool_keys.setdefault(name, set()).update(keys)
            version, have = self.view_keys.get(ident, (-1, set()))
            self.view_keys[ident] = (self.s.version, (have if version == self.s.version else set()) | keys)

    def shown_already(self, name: str, ident: str) -> bool:
        version, have = self.view_keys.get(ident, (-1, set()))
        keys = self.tool_keys.get(name)
        return bool(keys) and version == self.s.version and keys <= have and ident not in self.gone

    # -- result shapes and paging
    def shape_of(self, name: str, observation: Any) -> dict[str, Any] | None:
        """The parts of a listing result (rows, total, next page, price and stock keys), asked once per tool."""
        if name in self.shapes:
            return self.shapes[name]
        if not isinstance(observation, dict):
            return None
        lists = [k for k, v in observation.items() if isinstance(v, list) and len(v) >= 2 and all(isinstance(r, dict) for r in v)]
        if not lists:
            return None
        self.shapes[name] = None
        sample = {k: v[:3] if k in lists else v for k, v in observation.items()}
        prompt = P_SHAPE + '\n\nTOOL: ' + name + ' | parameters: ' + to_json(self.s.store.schemas.get(name) or {}) + '\n\nRESULT (first rows only):\n' + to_json(clip(sample, 120))[:4000]
        got = self.s.models.ask(prompt, [LIGHT, HEAVY], 2500) or {}
        rows = got.get('rows') if got.get('rows') in lists else None
        if not rows:
            return None
        first = observation[rows][0]
        props = self.s.store.props(name)

        def named(key: Any, where: dict[str, Any]) -> str | None:
            return key if isinstance(key, str) and key in where else None
        shape = {'rows': rows, 'count': named(got.get('count'), observation), 'next': named(got.get('next'), observation),
                 'next_arg': named(got.get('next_arg'), props), 'price': named(got.get('price'), first), 'stock': named(got.get('stock'), first)}
        if shape['count'] and (isinstance(observation[shape['count']], bool) or not isinstance(observation[shape['count']], int)):
            shape['count'] = None
        self.shapes[name] = shape
        self.price_key = self.price_key or shape['price']
        self.stock_key = self.stock_key or shape['stock']
        self.s.log('shape', tool=name, shape=shape)
        return shape

    def paging(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        """(matched in all, rows shown, next-page parameter, its value) for a result that can be paged."""
        shape = self.shape_of(name, observation)
        if not shape or not shape['count'] or not shape['next'] or not shape['next_arg'] or not isinstance(observation, dict):
            return None
        rows, count, value = observation.get(shape['rows']), observation.get(shape['count']), observation.get(shape['next'])
        if not isinstance(rows, list) or not rows or isinstance(count, bool) or not isinstance(count, int):
            return None
        if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))):
            return None
        return count, len(rows), shape['next_arg'], value

    def price_sorted(self, observation: Any) -> bool:
        prices = [self.price_of(i) for i, _ in self.records_in(observation)]
        return len(prices) >= 3 and None not in prices and all(a <= b for a, b in zip(prices, prices[1:]))

    # -- fit codes
    def fit_index(self) -> dict[str, Any]:
        marks: dict[str, dict[str, Any]] = {}
        notes: dict[str, set[str]] = {}
        for ident in self.order:
            record = self.record(ident) or {}
            text = ' '.join(str(v) for v in record.values() if isinstance(v, str))
            text += ' ' + '. '.join(f'{k} {v}' for block in record.values() if isinstance(block, dict) for k, v in block.items() if isinstance(v, str))
            marks[ident] = fit_marks(text)
            for part, codes in marks[ident]['also']:
                notes.setdefault(part, set()).update(codes)
        return {'marks': marks, 'notes': notes}

    @staticmethod
    def fit_set(ident: str, index: dict[str, Any]) -> set[str]:
        mark = index['marks'].get(ident) or {}
        found = set(mark.get('works') or [])
        if mark.get('own'):
            found |= index['notes'].get(mark['own'], set())
        return found


# -------------------------------------------------------------------------------------------------------- Order

class Order(Part):
    """What the shopper asked for, read once by the brief model and kept up to date by answers and store notes."""

    def __init__(self, session: 'Session', view: dict[str, Any]):
        super().__init__(session)
        self.view = view
        self.request = str(view.get('query') or '')
        self.brief: dict[str, Any] = {}
        self.items: list[Item] = []
        self.budget: Any = None
        self.msg_limit = 0
        self.msg_used = 0
        self.reserve = 0
        self.gate_msgs = 0
        self.process: list[dict[str, Any]] = []
        self.refs: list[tuple[int, str]] = []
        self.ref_values: dict[str, str] = {}
        self.bound: list[tuple[int, int]] = []
        self.notes: list[str] = []
        self.adopted: dict[str, int] = {}

    def pick(self) -> Any:
        return self.brief.get('pick')

    def each_budget(self) -> bool:
        return self.brief.get('budget_scope') == 'each'

    def msgs_free(self) -> int:
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)

    def item_index(self, value: Any) -> int:
        return min(max(1, as_int(value) or 1), len(self.items))

    # -- reading the request
    def read(self) -> None:
        s = self.s
        store = s.store
        lines = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {to_json(t['function'].get('parameters') or {})}" for t in store.tools]
        prompt = P_BRIEF + '\nTOOLS:\n' + '\n'.join(lines) + '\n\nREQUEST:\n' + self.request
        if self.view.get('opening') is not None:
            prompt += '\n\nOPENING NOTE FROM THE STORE:\n' + to_json(clip(self.view.get('opening'), 1500))
        brief = s.models.ask(prompt, [LIGHT], BRIEF_TOKENS) or {}
        if not s.models.no_credit and not self.complete(brief):
            brief = s.models.ask(prompt, [HEAVY, LIGHT], 2500) or brief
        elif not s.models.no_credit and self.shared_shelf(brief):
            again = s.models.ask(prompt, [HEAVY], 2500) or {}
            if self.complete(again):
                brief = again
        self.brief = brief
        roles = brief.get('roles') if isinstance(brief.get('roles'), dict) else {}
        store.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in store.schemas}
        for key in ('buy_args', 'refuse_args'):
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        self.split_modes()
        store.buy_args, store.refuse_args = brief['buy_args'], brief['refuse_args']
        self.take_items()
        self.take_limits()

    @staticmethod
    def complete(brief: dict[str, Any]) -> bool:
        return bool(as_list(brief.get('items'))) and isinstance(brief.get('roles'), dict) and bool(brief['roles'].get('buy'))

    @staticmethod
    def shared_shelf(brief: dict[str, Any]) -> bool:
        shelves = [plain(str(e.get('shelf') or '')) for e in as_list(brief.get('items')) if isinstance(e, dict)]
        shelves = [s for s in shelves if s]
        return len(shelves) != len(set(shelves))

    def split_modes(self) -> None:
        """When one tool both checks out and ends without a purchase, make sure the two argument sets differ."""
        brief, store = self.brief, self.s.store
        buy, refuse = store.roles.get('buy'), store.roles.get('refuse')
        if not buy or buy != refuse or (brief['buy_args'] and brief['refuse_args'] and brief['buy_args'] != brief['refuse_args']):
            return
        prompt = P_MODES.replace('<TOOL>', buy).replace('<SCHEMA>', to_json(store.schemas.get(buy)))
        fixed = self.s.models.ask(prompt, [LIGHT, HEAVY], 1500) or {}
        if isinstance(fixed.get('buy_args'), dict) and isinstance(fixed.get('refuse_args'), dict) and fixed['buy_args'] != fixed['refuse_args']:
            brief['buy_args'], brief['refuse_args'] = fixed['buy_args'], fixed['refuse_args']

    def take_items(self) -> None:
        brief = self.brief
        found: list[Item] = []
        for entry in as_list(brief.get('items')):
            if not isinstance(entry, dict):
                continue
            quantity = as_int(entry.get('quantity'))
            found.append(Item(str(entry.get('name') or 'the product'), str(entry.get('shelf') or ''), quantity if quantity and quantity > 0 else 1, [str(m) for m in as_list(entry.get('musts')) if m]))
        # items on the same store label with no clashing requirement are one item
        merged: list[Item] = []
        became: dict[int, int] = {}
        for old, item in enumerate(found, 1):
            twin = next((m for m in merged if item.shelf.strip() and plain(m.shelf) == plain(item.shelf) and not self.clash(m, item)), None)
            if twin is None:
                merged.append(item)
                became[old] = len(merged)
                continue
            became[old] = merged.index(twin) + 1
            twin.musts += [m for m in item.musts if m not in twin.musts]
            self.s.log('items_merged', into=twin.name, dropped=item.name)
        self.items = merged or [Item('the requested product', '', 1, [])]
        for pair in as_list(brief.get('fits')):
            if not isinstance(pair, dict):
                continue
            part, main = became.get(as_int(pair.get('item')) or 0), became.get(as_int(pair.get('with')) or 0)
            if part and main and part != main and 1 <= part <= len(self.items) and 1 <= main <= len(self.items) and (part, main) not in self.bound:
                self.bound.append((part, main))
        for lookup in as_list(brief.get('lookups')):
            text = str(lookup).strip()
            if not any(text in item.musts for item in self.items):
                key = plain(text.split(':')[0])
                text = next((m for item in self.items for m in item.musts if plain(m.split(':')[0]) == key), text)
            number = next((n for n, item in enumerate(self.items, 1) if text in item.musts), 1)
            if text and (number, text) not in self.refs:
                self.refs.append((number, text))
        self.unmix()

    @staticmethod
    def clash(a: Item, b: Item) -> bool:
        return any(plain(x.split(':')[0]) == plain(y.split(':')[0]) and plain(x.partition(':')[2]) != plain(y.partition(':')[2]) for x in a.musts for y in b.musts)

    def take_limits(self) -> None:
        brief, store = self.brief, self.s.store
        budget = as_number(brief.get('budget')) if isinstance(brief.get('budget'), str) else brief.get('budget')
        if is_number(budget) and budget > 0:
            self.budget = budget
        limit = as_int(brief.get('messages'))
        if limit is not None and limit >= 0:
            self.msg_limit = limit
        elif store.roles.get('ask'):
            self.msg_limit = 3
        steps = as_int(brief.get('steps'))
        if steps is not None and 10 <= steps < store.max_steps:
            store.max_steps = steps
        for entry in as_list(brief.get('process')):
            if isinstance(entry, dict) and str(entry.get('step') or '').strip():
                when = str(entry.get('when') or '').strip().lower()
                self.process.append({'step': str(entry['step']).strip(), 'when': when if when in ('choose', 'order', 'end') else 'order', 'messages': max(0, as_int(entry.get('messages')) or 0)})
        if store.roles.get('ask'):
            talk = [p['messages'] for p in self.process if p['when'] in ('order', 'end') and p['messages']]
            self.reserve = min(max(talk), self.msg_limit) if talk else 0

    def unmix(self) -> None:
        """Move words that only describe a pointed-at listing out of the item's requirements and into the pointer."""
        for number, item in enumerate(self.items, 1):
            pointers = [need for n, need in self.refs if n == number and need in item.musts]
            rest = [m for m in item.musts if m not in pointers]
            if not pointers or not rest:
                continue
            prompt = P_UNMIX + '\n\nREQUEST:\n' + self.request + f'\n\nPRODUCT: {item.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n' + '\n'.join(pointers) + '\nOTHER REQUIREMENTS:\n' + '\n'.join(rest)
            got = self.s.models.ask(prompt, [HEAVY, LIGHT], 1500) or {}
            moved = [m for m in rest if m in [str(x).strip() for x in as_list(got.get('describe_other'))]]
            if not moved:
                continue
            item.musts = [m for m in item.musts if m not in moved]
            joined = pointers[0] + ' (that listing: ' + '; '.join(moved) + ')'
            item.musts = [joined if m == pointers[0] else m for m in item.musts]
            self.refs = [(n, joined if (n, need) == (number, pointers[0]) else need) for n, need in self.refs]
            self.s.log('unmix', item=number, moved=moved)

    # -- requirements now
    def needs_now(self) -> list[tuple[int, str]]:
        return [(number, need) for number, item in enumerate(self.items, 1) for need in item.musts + item.told if need not in item.dropped]

    def pointer_needs(self) -> list[str]:
        return [m for _, m in self.refs]

    def described(self, number: int, need: str) -> bool:
        """A long requirement whose property no opened listing of the item carries (three read at least): the shopper's
        description of the product, which decides nothing."""
        if need in self.pointer_needs() or need in [str(x) for x in as_list(self.brief.get('lookups'))]:
            return False
        prop, _, value = need.partition(':')
        if not prop.strip() or len(re.findall('[a-z0-9]+', value.lower())) < 4:
            return False
        if re.search('\\b(?:listing|page|shown|other than|unlike|same as)\\b', value, re.I):
            return False
        item = self.items[number - 1]
        if any({w for w in stems(other.name) if len(w) >= 4} & stems(value) for other in self.items if other is not item):
            return False
        mem = self.s.mem
        cat_key = mem.category_key()
        keys: set[str] = set()
        read = 0
        for record in mem.full.values():
            if cat_key and not self.owns(item, str(record.get(cat_key) or '')):
                continue
            block = attributes(record) or {}
            if block:
                read += 1
                keys |= {plain(k) for k in block}
        return read >= 3 and not any(stems(k) & stems(prop) for k in keys)

    def ledger(self) -> str:
        """The requirements in force, as the pilot and every check read them."""
        s = self.s
        parts = []
        many = len(self.items) > 1
        for number, item in enumerate(self.items, 1):
            live = item.live()
            told = [m for m in live if self.described(number, m)]
            needs = [m + (f' (the page it points at gives: {self.ref_values[m]})' if m in self.ref_values else '') for m in live if m not in told]
            head = f'item {number}: {item.name}' if many else f'item: {item.name}'
            if item.shelf:
                head += f' (store label: {item.shelf})'
            head += f', quantity {item.qty}'
            text = head + '; requirements: ' + ('; '.join(needs) if needs else 'none beyond the kind of product')
            if told:
                text += "; described in the shopper's words, which no listing attribute states (decides nothing): " + '; '.join(told)
            parts.append(text)
        brief = self.brief
        if self.budget is not None:
            scope = 'for each item' if self.each_budget() else 'in total'
            parts.append(f"budget: at most {self.budget} {brief.get('budget_unit') or ''} {scope}".replace('  ', ' '))
        pick = self.pick()
        if pick == 'cheapest':
            parts.append('pick: the cheapest listing that fits (lowest price)')
        elif pick in ('most', 'least'):
            parts.append(f"pick: the {pick} {brief.get('pick_measure') or ''} among fitting listings".replace('  ', ' '))
        else:
            parts.append('pick: any listing that fits everything')
        if brief.get('in_stock_only', True) is not False:
            parts.append('in stock only')
        parts += [f'wish (tie-breaker only): {wish}' for wish in as_list(brief.get('wishes'))]
        parts += [f'rule from the request: {rule}' for rule in as_list(brief.get('rules'))]
        if brief.get('quiet_decline') is True:
            parts.append('if nothing fits: end without naming or suggesting any listing')
        for entry in self.process:
            when = {'choose': 'before choosing', 'order': 'at the order (Arc does it when you check out)', 'end': 'at an ending without a purchase (Arc does it then)'}[entry['when']]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        mem, store = s.mem, s.store
        facts = mem.facts_tool()
        if facts and (mem.facts_from.get(facts) is True or store.roles.get('offer')):
            bare = sorted(t for t in store.page_tools() | {store.roles.get('offer') or ''} if t and mem.facts_from.get(t) is False)
            pages = sorted(t for t in store.page_tools() if t != facts)
            parts.append(f'listing attributes come from {facts}' + (f"; {', '.join(bare)} show none" if bare else '') + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ''))
        return '\n'.join('- ' + p for p in parts)

    # -- which item a category label belongs to
    def wanted(self, label: str) -> bool:
        """Whether a label names a requested item: its shelf or name exactly, or a shared long word."""
        words = plain(label)
        long = {w for w in stems(label) if len(w) >= 4}
        for item in self.items:
            mine = {w for w in stems(item.shelf or '') | stems(item.name) if len(w) >= 4}
            if words in (plain(item.shelf or ''), plain(item.name)) or long & mine:
                return True
        return False

    def owner_of(self, label: str) -> int:
        """The item a category label belongs to: exact shelf, then exact name, then the most shared root words."""
        if not label:
            return 0
        if label in self.adopted:
            return self.adopted[label]
        words = plain(label)
        long = roots(label)
        best, top = 0, 0.0
        for number, item in enumerate(self.items, 1):
            if words == plain(item.shelf or ''):
                score = 3.0
            elif words == plain(item.name):
                score = 2.0
            else:
                mine = roots(item.shelf or '') | roots(item.name)
                score = len(long & mine) / max(1, len(long | mine))
            if score > top:
                best, top = number, score
        return best

    def owns(self, item: Item, label: str) -> bool:
        return bool(label) and item in self.items and self.owner_of(label) == self.items.index(item) + 1

    def adopt_labels(self) -> None:
        """With exactly one item left without a label of its own, the labels no item claims go to it."""
        mem = self.s.mem
        cat_key = mem.category_key()
        if len(self.items) < 2 or not cat_key:
            return
        labels = {str(record.get(cat_key) or '') for record in mem.full.values()} - {''}
        strays = [label for label in labels if not self.owner_of(label)]
        orphans = [n for n in range(1, len(self.items) + 1) if not any(self.owner_of(label) == n and label not in self.adopted for label in labels)]
        if strays and len(orphans) == 1:
            self.adopted.update({label: orphans[0] for label in strays})
            self.s.log('adopted', item=orphans[0], labels=strays)

    def item_ids(self, number: int) -> list[str]:
        mem = self.s.mem
        cat_key = mem.category_key()
        item = self.items[number - 1]
        return [i for i in mem.order if cat_key and self.owns(item, mem.label_of(i, cat_key))]

    def records_of_item(self, number: int) -> int:
        mem = self.s.mem
        cat_key = mem.category_key()
        item = self.items[number - 1]
        return sum(1 for record in mem.full.values() if cat_key and self.owns(item, str(record.get(cat_key) or '')) and attributes(record))

    def home_label(self, number: int) -> str:
        """The item's own category label (plain form): the request's shelf or name when a seen listing carries it,
        else the label most of its opened listings carry (with none opened, its search rows); '' when none is known."""
        mem = self.s.mem
        cat_key = mem.category_key()
        if not cat_key:
            return ''
        item = self.items[number - 1]
        for pool in (mem.full, mem.rows):
            counts: dict[str, int] = {}
            for record in pool.values():
                label = str(record.get(cat_key) or '')
                if label and self.owns(item, label):
                    counts[plain(label)] = counts.get(plain(label), 0) + 1
            for name in (item.shelf, item.name):
                if plain(name) in counts:
                    return plain(name)
            if counts:
                return max(counts, key=lambda label: counts[label])
        return ''

    # -- process steps
    def gate_due(self, stage: str) -> bool:
        return any(p['when'] == stage for p in self.process) and (stage, self.s.basket.mark() + f'#{self.s.version}') not in self.s.referee.gate_ok

    def tail(self) -> int:
        """Steps kept for the ending: two when an order-time message is still due, else one."""
        talk = any(p['when'] == 'order' and p['messages'] > 0 for p in self.process)
        return 2 if talk and self.msg_used < self.msg_limit and self.gate_due('order') else 1


# ------------------------------------------------------------------------------------------------------- Basket

class Basket(Part):
    """The cart as the agent's own cart calls and the last cart view describe it, with the line-by-line checks."""

    def __init__(self, session: 'Session'):
        super().__init__(session)
        self.lines: dict[str, int] = {}
        self.view: Any = None
        self.seen_at = -1
        self.changed_at = -1
        self.touched = False
        self.said = ''
        self.built: list[str] = []
        self.dropped: set[str] = set()
        self.line_holds = 0

    def mark(self) -> str:
        return to_json(sorted(self.lines.items()))

    def fresh(self) -> bool:
        return self.view is not None and self.seen_at >= self.changed_at

    def ids(self) -> list[str]:
        if self.fresh():
            text = to_json(self.view)
            return [i for i in self.s.mem.rows if i in text]
        return list(self.lines)

    def units(self, ident: str) -> int:
        """Units held of a line: the cart view's one count field for it when the view is fresh, else the agent's record."""
        if self.fresh():
            stack: list[Any] = [self.view]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    if ident in [v.strip() for v in node.values() if isinstance(v, str)]:
                        counts = [as_int(v) for k, v in node.items() if not isinstance(v, str) and as_int(v) is not None and re.search('\\b(?:qty|quantity|count|units?)\\b', plain(k))]
                        if len(counts) == 1:
                            return counts[0]
                    stack += [v for v in node.values() if isinstance(v, (dict, list))]
                elif isinstance(node, list):
                    stack += node
        return self.lines.get(ident, 1)

    def text(self) -> str:
        if self.fresh():
            return to_json(clip(self.view, 300))[:3000]
        return "lines by Arc's record of the cart calls (listing: units): " + to_json(self.lines)

    def seen(self, observation: Any, step: int) -> None:
        self.view = observation
        self.seen_at = step

    def note_call(self, name: str, args: dict[str, Any]) -> None:
        """Update the agent's record of the cart after a cart call: a whole-cart list, a set count, an add or a removal."""
        store = self.s.store
        rows = self.s.mem.rows
        self.touched = True
        self.changed_at = store.step
        spec = store.props(name)
        removal = any(isinstance((spec.get(key) or {}).get('enum'), list) and len(spec[key]['enum']) >= 2 and value in spec[key]['enum'][1:] for key, value in args.items())
        loose_count = next((as_int(v) for v in args.values() if as_int(v) is not None and not isinstance(v, str)), None)
        lines: list[tuple[str, int | None]] = []
        whole = False
        for key, value in args.items():
            if isinstance(value, list):
                shape = (spec.get(key) or {}).get('items') or {}
                if not value and isinstance(shape, dict) and (shape.get('type') == 'object' or 'properties' in shape):
                    whole = True
                for line in value:
                    if isinstance(line, dict):
                        whole = True
                        ident = next((v for v in line.values() if isinstance(v, str) and v in rows), None)
                        count = next((as_int(v) for v in line.values() if as_int(v) is not None and not isinstance(v, str)), 1)
                        if ident:
                            lines.append((ident, count))
                    elif isinstance(line, str) and line in rows:
                        lines.append((line, loose_count))
            elif isinstance(value, str) and value in rows:
                lines.append((value, loose_count))
        if whole:
            self.lines = {i: c for i, c in lines if c and c > 0}
            return
        sets = store.cart_sets(name)
        counted = any('integer' in type_list(v) for v in spec.values() if isinstance(v, dict))
        for ident, count in lines:
            have = self.lines.get(ident, 0)
            if removal:
                have = have - (count or 1) if counted else (have - count if count else 0)
            elif sets:
                have = count if count is not None else 1
            else:
                have = have + (count if count is not None else 1)
            if have > 0:
                self.lines[ident] = have
            else:
                self.lines.pop(ident, None)

    def after_call(self, name: str, args: dict[str, Any], observation: Any) -> None:
        """Bookkeeping after the pilot's cart call: record, removed lines, and the fullest cart built so far."""
        mem = self.s.mem
        self.note_call(name, args)
        self.said = to_json(observation)
        counts = [as_int(v) for v in args.values() if not isinstance(v, str) and as_int(v) is not None]
        for value in args.values():
            if counts and isinstance(value, str) and value in mem.rows:
                if counts[0] <= 0 and not mentions(observation, value):
                    self.dropped.add(value)
                elif counts[0] > 0:
                    self.dropped.discard(value)
        lines = [i for i in self.ids() if not mem.sold_out(i) and i not in self.dropped]
        if len(lines) >= len(self.built):
            self.built = lines

    def whole_removal(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """A removal of a held line sent without a count names the units held, since the store takes off only one."""
        spec = self.s.store.props(name)
        count = next((k for k, v in spec.items() if isinstance(v, dict) and 'integer' in type_list(v)), None)
        removal = any(isinstance((spec.get(k) or {}).get('enum'), list) and v in spec[k]['enum'][1:] for k, v in args.items())
        ident = next((v for v in args.values() if isinstance(v, str) and v in self.lines), None)
        if count and removal and ident and count not in args:
            return {**args, count: min(self.lines[ident], spec[count].get('maximum', self.lines[ident]))}
        return args

    def plan_calls(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        """Cart calls that turn the cart into exactly `lines` after taking out `drop`; None when the tool can't."""
        store = self.s.store
        name = store.roles.get('cart') or ''
        schema = store.schemas.get(name) or {}
        props = schema.get('properties') or {}
        for key, spec in props.items():
            shape = spec.get('items') if isinstance(spec, dict) and 'array' in type_list(spec) else None
            if isinstance(shape, dict) and isinstance(shape.get('properties'), dict):
                inner = shape['properties']
                id_key = next((k for k, v in inner.items() if isinstance(v, dict) and 'string' in type_list(v) and 'enum' not in v), None)
                count_key = next((k for k, v in inner.items() if isinstance(v, dict) and 'integer' in type_list(v)), None)
                if not id_key:
                    return None
                return [{key: [{id_key: i, **({count_key: n} if count_key else {})} for i, n in lines.items()]}]
        texts = free_text_keys(schema)
        target = next((k for t, k in store.id_args if t == name and k in texts), None) or next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        count = next((k for k, v in props.items() if isinstance(v, dict) and 'integer' in type_list(v)), None)
        mode = next((k for k, v in props.items() if isinstance(v, dict) and isinstance(v.get('enum'), list) and len(v['enum']) >= 2), None)
        if not target:
            return None
        calls: list[dict[str, Any]] = []
        for ident in drop:
            have = self.lines.get(ident, 1)
            if mode:
                calls.append({target: ident, mode: props[mode]['enum'][1], **({count: have} if count else {})})
            elif count and store.cart_sets(name):
                calls.append({target: ident, count: 0})
            elif count and isinstance(props[count].get('minimum'), int) and props[count]['minimum'] <= -have:
                calls.append({target: ident, count: -have})
            else:
                return None
        for ident, want in lines.items():
            have = self.lines.get(ident, 0) if ident in self.ids() else 0
            if have == want:
                continue
            if have and not store.cart_sets(name):
                return None
            calls.append({target: ident, **({mode: props[mode]['enum'][0]} if mode else {}), **({count: want} if count else {})})
        return calls

    def apply(self, calls: list[dict[str, Any]], observe: bool = True) -> list[str] | None:
        """Send planned cart calls; their argument texts, or None when one did not fit or failed."""
        s = self.s
        cart = s.store.roles.get('cart') or ''
        done = []
        for args in calls:
            faults: list[str] = []
            args = conform(args, s.store.schemas.get(cart), 'args', faults)
            if faults:
                return None
            reply = s.store.call(cart, args)
            if reply is None or (observe and s.store.finished):
                return None
            self.note_call(cart, args)
            if observe:
                s.reader.observe(cart, reply.get('observation'))
            done.append(to_json(args))
        return done

    # -- line checks
    def count_fixes(self) -> list[tuple[str, int, int]]:
        """(line, units held, units wanted) for each line that alone covers its item at the wrong count."""
        s = self.s
        mem, order = s.mem, s.order
        cat_key = mem.category_key()
        if not cat_key:
            return []
        out = []
        for ident in self.ids():
            have = self.units(ident) or 0
            owners = [n for n, item in enumerate(order.items, 1) if order.owns(item, mem.label_of(ident, cat_key))]
            mine = [i for i in self.ids() if (self.lines.get(i, 1) or 0) > 0 and len(owners) == 1 and order.owns(order.items[owners[0] - 1], mem.label_of(i, cat_key))]
            if have > 0 and len(owners) == 1 and len(mine) == 1 and have != order.items[owners[0] - 1].qty and not mem.sold_out(ident):
                out.append((ident, have, order.items[owners[0] - 1].qty))
        return out

    def fix_counts(self, wrong: list[tuple[str, int, int]]) -> list[str]:
        s = self.s
        store = s.store
        name = store.roles.get('cart') or ''
        schema = store.schemas.get(name) or {}
        props = schema.get('properties') or {}
        texts = free_text_keys(schema)
        target = next((k for t, k in store.id_args if t == name and k in texts), None) or next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        count = next((k for k, v in props.items() if isinstance(v, dict) and 'integer' in type_list(v)), None)
        mode = next((k for k, v in props.items() if isinstance(v, dict) and isinstance(v.get('enum'), list) and len(v['enum']) >= 2), None)
        if not name or not target or not count:
            return []
        calls: list[dict[str, Any]] = []
        for ident, have, want in wrong:
            if store.cart_sets(name):
                calls.append({target: ident, count: want})
            elif mode:
                options = [str(o) for o in props[mode]['enum']]
                off = next((o for o in options if re.search('remov|delet|drop|clear|take', o, re.I)), options[1])
                on = next((o for o in options if o != off), options[0])
                calls += [{target: ident, mode: off, count: have}, {target: ident, mode: on, count: want}]
            else:
                return []
        if len(calls) + 3 > store.left():
            return []
        done = []
        for args in calls:
            faults: list[str] = []
            args = conform(args, schema, 'args', faults)
            if faults:
                break
            reply = store.call(name, args)
            if reply is None or store.finished:
                break
            self.note_call(name, args)
            self.said = to_json(reply.get('observation'))
            s.reader.observe(name, reply.get('observation'))
            done.append(to_json(args) + ' -> ' + s.reader.show(reply.get('observation'))[:300])
        s.log('count_fix', calls=done)
        return done

    def missing_items(self) -> list[int]:
        """Items with no line in a cart holding fewer lines than items (items no seen listing is labelled for aside)."""
        s = self.s
        mem, order = s.mem, s.order
        cat_key = mem.category_key()
        ids = [i for i in self.ids() if (self.units(i) or 0) > 0]
        if not cat_key or len(ids) >= len(order.items):
            return []
        labels = [str(r.get(cat_key) or '') for r in mem.rows.values()]
        held = [mem.label_of(i, cat_key) for i in ids]
        return [n for n, item in enumerate(order.items, 1) if any(order.owns(item, label) for label in labels) and not any(order.owns(item, label) for label in held)]

    def states_needs(self, ident: str, number: int) -> bool:
        """Whether a listing's attributes state each of the item's requirements word for word. b5: a requirement that
        points at another listing counts as not stated; a limit or an exclusion fails by itself, since every word of
        the requirement must stand in the attribute's value (Halo's list of limit words removed: a word list)."""
        s = self.s
        block = attributes(s.mem.full.get(ident) or {}) or {}
        split = lambda text: set(re.sub('[,/;|]', ' ', plain(text)).split())
        pointers = s.order.pointer_needs()
        for owner, need in s.order.needs_now():
            if owner != number:
                continue
            prop, _, value = need.partition(':')
            words = split(value)
            if not words or need in pointers:
                return False
            if not any(same_property(spellings(prop), spellings(k)) and words <= split(str(v)) for k, v in block.items()):
                return False
        return True

    def fill_short(self) -> bool:
        """Before a forced order: add, for each item with no line, the cheapest opened listing that states its
        requirements. False when an item stays uncovered, so the episode ends by refusing rather than short."""
        s = self.s
        mem, order, store = s.mem, s.order, s.store
        cart = store.roles.get('cart') or ''
        bound = {n for pair in order.bound for n in pair}
        for number in self.missing_items():
            item = order.items[number - 1]
            fits = [i for i in order.item_ids(number) if i in mem.full and i not in self.lines and mem.buyable(i) and mem.price_of(i) is not None and self.states_needs(i, number)]
            fits.sort(key=lambda i: mem.price_of(i) or 0.0)
            pick = fits[0] if fits and number not in bound else None
            if pick and is_number(order.budget):
                held = sum((mem.price_of(i) or 0.0) * (self.units(i) or 0) for i in self.ids())
                cost = (mem.price_of(pick) or 0.0) * (1 if order.each_budget() else item.qty)
                if cost + (0.0 if order.each_budget() else held) > order.budget + 1e-09:
                    pick = None
            calls = self.plan_calls({**{i: self.lines[i] for i in self.ids() if i in self.lines}, pick: item.qty}, []) if pick else None
            if not calls or len(calls) + 1 > store.left():
                return False
            for args in calls:
                faults: list[str] = []
                args = conform(args, store.schemas.get(cart), 'args', faults)
                if faults or store.call(cart, args) is None:
                    return False
                self.note_call(cart, args)
            s.log('filled_line', item=number, line=pick)
        return True

    def fuller_label(self, number: int, label: str) -> str:
        """A seen label that names the item in full when the line's label names it only in part."""
        s = self.s
        cat_key = s.mem.category_key()
        item = s.order.items[number - 1]
        mine = roots(item.name)
        if len(mine) < 2 or mine <= roots(label) or not cat_key or (item.shelf and roots(label) == roots(item.shelf)):
            return ''
        return next((seen for seen in sorted({str(r.get(cat_key) or '') for r in s.mem.rows.values()}) if seen and mine <= roots(seen)), '')

    def line_faults(self) -> list[str]:
        """What is wrong with the cart line by line: a missing or doubled item, wrong units, an unrequested kind, a
        sold-out line, the budget."""
        s = self.s
        mem, order = s.mem, s.order
        order.adopt_labels()
        cat_key = mem.category_key()
        ids = [i for i in self.ids() if (self.units(i) or 0) > 0]
        if not ids or not cat_key:
            return []
        label = {i: mem.label_of(i, cat_key) for i in ids}
        known = {n for n, item in enumerate(order.items, 1) if any(order.owns(item, str(r.get(cat_key) or '')) for r in mem.rows.values())}
        owner = {i: [n for n, item in enumerate(order.items, 1) if order.owns(item, label[i])] for i in ids}
        faults: list[str] = []
        for n, item in enumerate(order.items, 1):
            if n not in known:
                continue
            lines = [i for i in ids if n in owner[i]]
            if not lines:
                faults.append(f'item {n} ({item.name}) has no line in the cart')
            elif len([i for i in lines if owner[i] == [n]]) > 1:
                faults.append(f"item {n} ({item.name}) has {len(lines)} lines ({', '.join(lines)}); keep one")
            for i in lines:
                have = self.units(i)
                if owner[i] == [n] and have != item.qty:
                    faults.append(f'{i} is held at {have} unit(s); item {n} ({item.name}) wants {item.qty}')
                fuller = self.fuller_label(n, label[i])
                if fuller:
                    faults.append(f'{i} is from {label[i]}, but the store has the label {fuller}, which names item {n} ({item.name}) in full: take the line from there')
        if len(known) == len(order.items):
            faults += [f"{i} ({label[i] or 'unknown kind'}) is not a kind the shopper asked for: take it out" for i in ids if not owner[i]]
        faults += [f'{i} is sold out' for i in ids if mem.sold_out(i)]
        prices = {i: mem.price_of(i) for i in ids}
        if is_number(order.budget) and all(p is not None for p in prices.values()):
            if order.each_budget():
                faults += [f'{i} costs {prices[i]:g}, over the {order.budget:g} limit for each item' for i in ids if order.budget < prices[i] <= 10 * order.budget]
            else:
                total = sum(prices[i] * self.units(i) for i in ids)
                if order.budget + 1e-09 < total <= 10 * order.budget:
                    faults.append(f'the total is {total:g}, over the budget of {order.budget:g}')
        return faults


# ------------------------------------------------------------------------------------------------------- Reader

class Reader(Part):
    """Reading done in code: attribute pages of chosen listings, further result pages, cheaper rows below a cart line,
    pages a part code points at, and the notes the store slips into results."""

    def __init__(self, session: 'Session'):
        super().__init__(session)
        self.heard: set[str] = set()
        self.queued: list[str] = []
        self.note_sold = False
        self.note_named = False
        self.sold_lines: set[str] = set()
        self.pages_read = 0
        self.paged_obs: list[Any] = []
        self.read_cache: dict[str, tuple[int, int, str]] = {}
        self.offer_seen: dict[str, int] = {}
        self.capped: set[tuple[str, float]] = set()
        self.link_tried: set[str] = set()
        self.searched: list[dict[str, Any]] = []

    @staticmethod
    def show(observation: Any) -> str:
        text = to_json(clip(observation, SHOW_CHARS))
        return text[:SHOW_LIMIT] + '...(cut)' if len(text) > SHOW_LIMIT else text

    def observe(self, name: str, observation: Any) -> str:
        """Take in a result: its listings into memory, and any note in it read for changes."""
        self.s.mem.absorb(observation)
        lines = self.loose_lines(observation)
        if not lines:
            return ''
        meaning = self.take_note(lines)
        if meaning:
            self.queued.append(f'(seen on {name}) {meaning}')
        return meaning

    # -- notes
    @staticmethod
    def spoken(observation: Any) -> str:
        """The words in a reply: its strings of two words or more, else the reply itself."""
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
        return ' '.join(found)[:800] if found else to_json(observation)[:400]

    def loose_lines(self, observation: Any) -> list[str]:
        """Sentences in a result outside its listings, not read before."""
        mem = self.s.mem
        lines: list[str] = []
        if isinstance(observation, dict) and not mem.ident(observation):
            for value in observation.values():
                if isinstance(value, str) and len(value.split()) >= 5:
                    lines.append(value)
                elif isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                    lines.extend(v for v in value if len(v.split()) >= 5)
                elif isinstance(value, dict) and not mem.ident(value, False, 1):
                    lines.extend(inner for inner in value.values() if isinstance(inner, str) and len(inner.split()) >= 6)
        return [line for line in lines if line not in self.heard]

    def take_note(self, lines: list[str]) -> str:
        """Read a store note into the order: budget, requirements added or dropped, listings gone, prices moved."""
        s = self.s
        order, mem = s.order, s.mem
        self.heard.update(lines)
        prompt = P_NOTE + '\n\nITEMS AND REQUIREMENTS NOW:\n' + order.ledger() + '\n\nTHE NOTE:\n' + '\n'.join(lines)
        got = s.models.ask(prompt, [LIGHT, SPARE], 2500) or {}
        gone = [str(g) for g in as_list(got.get('gone')) if isinstance(g, (str, int))]
        mem.unavailable.update(g for g in gone if g in mem.rows)
        self.note_sold = bool(gone) or got.get('sold_out') is True
        self.note_named = any(g in mem.rows for g in gone)
        if self.note_sold and len(lines) == 1:
            self.sold_lines.update(lines)
        if got.get('kind') not in (None, 'change'):
            return ''
        changed = []
        budget = got.get('budget')
        if is_number(budget) and budget > 0 and budget != order.budget:
            order.budget = budget
            changed.append(f'budget now {budget}')
        drops = [e for e in as_list(got.get('drop')) if isinstance(e, dict) and str(e.get('need') or '').strip()]
        adds = [e for e in as_list(got.get('add')) if isinstance(e, dict) and str(e.get('need') or '').strip()]
        for entry in drops:
            item = order.items[order.item_index(entry.get('item')) - 1]
            target = str(entry['need']).strip()
            live = item.live()
            hit = [m for m in live if m == target] or [m for m in live if plain(m.split(':')[0]) == plain(target.split(':')[0])][:1]
            item.dropped.extend(hit or [target])
            changed.append(f'dropped {target}')
        for entry in adds:
            item = order.items[order.item_index(entry.get('item')) - 1]
            need = str(entry['need']).strip()
            if need in item.dropped:
                item.dropped.remove(need)
            if need not in item.musts + item.told:
                item.told.append(need)
            changed.append(f'added {need}')
        if gone:
            changed.append('unavailable: ' + ', '.join(gone))
        if got.get('price_moved') is True:
            changed.append('a price changed')
        summary = str(got.get('summary') or '').strip()
        if order.pick() == 'cheapest' and (drops or gone or self.note_sold or any(c.startswith('budget now') for c in changed)):
            s.referee.wider_held = False
            s.referee.cheaper_marks = set()
        if changed or summary:
            s.version += 1
            s.pilot.recent = []
            order.notes.append(summary or '; '.join(changed))
        return (summary + (' (' + '; '.join(changed) + ')' if changed else '')).strip()

    def evidence(self, observation: Any) -> str:
        """[evidence] lines for opened listings: the attribute keys whose names match a requirement's property."""
        s = self.s
        mem = s.mem
        ids: list[str] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                ident = mem.ident(node, False, depth)
                if ident and attributes(node) and ident not in ids:
                    ids.append(ident)
                    return
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1)
        walk(observation, 0)
        needs = s.order.needs_now()
        if not ids or not needs:
            return ''
        many = len(s.order.items) > 1
        lines = []
        for ident in ids[:12]:
            block = attributes(mem.full.get(ident) or {}) or {}
            parts = []
            for number, need in needs[:8]:
                prop = need.split(':', 1)[0].strip()
                hits = [f'{k}={str(v)[:50]}' for k, v in block.items() if same_property(stems(prop), stems(k))]
                if hits:
                    parts.append(f"{f'(item {number}) ' if many else ''}{prop} -> " + ', '.join(hits[:3]))
            if parts:
                lines.append(f'{ident}: ' + '; '.join(parts))
        if not lines:
            return ''
        return "[evidence: keys whose names contain a requirement's property; one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property] " + ' | '.join(lines)

    # -- reading pages
    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        """Open the attribute page of each listing not read in full yet, while more than `keep` steps are left."""
        s = self.s
        mem, store = s.mem, s.store
        tool = mem.facts_tool()
        target = store.single_target(tool)
        if not tool or not target:
            return []
        texts = []
        for ident in [i for i in ids if i in mem.rows and i not in mem.full and not mem.sold_out(i)]:
            if store.left() <= keep or store.finished or mem.facts_from.get(tool) is False:
                break
            faults: list[str] = []
            args = conform({target: ident}, store.schemas.get(tool), 'args', faults)
            if faults:
                break
            reply = store.call(tool, args)
            if reply is None:
                break
            seen = reply.get('observation')
            self.observe(tool, seen)
            mem.note_view(tool, seen)
            if isinstance(seen, dict) and mem.ident(seen):
                mem.facts_from[tool] = mem.facts_from.get(tool, False) or bool(attributes(seen))
            texts.append(f'{ident}: ' + self.show(seen)[:1500])
        if texts:
            s.log('fill_facts', n=len(texts))
        return texts

    def show_links(self, ids: list[str], keep: int) -> None:
        """Search the part code a cart line sends the reader to, once per code, and open what it finds while steps
        allow; the result feeds the records, not the pilot. Multi-line carts only."""
        s = self.s
        plan = self.value_template()
        if not plan or (len(s.order.items) < 2 and len(ids) < 2):
            return
        tool, base, field = plan
        for ident in ids:
            marks = s.mem.fit_index()['marks']
            code = (marks.get(ident) or {}).get('more')
            if not code or code in self.link_tried or any(mark.get('own') == code for mark in marks.values()):
                continue
            if s.store.left() <= keep or s.store.finished:
                return
            self.link_tried.add(code)
            reply = s.store.run(tool, {**base, field: code})
            if reply is None:
                return
            self.observe(tool, reply.get('observation'))
            s.log('show_link', line=ident, code=code)
            marks = s.mem.fit_index()['marks']
            self.fill_facts([i for i, mark in marks.items() if mark.get('own') == code], max(keep, s.order.tail() + 6))

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """For a cheapest/most/least pick, read up to two further pages of a short, unsorted result."""
        s = self.s
        mem, store, order = s.mem, s.store, s.order
        self.paged_obs = []
        pick = order.pick()
        plan = mem.paging(name, observation) if pick in ('cheapest', 'most', 'least') else None
        if not plan:
            return ''
        count, shown, param, value = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and not isinstance(given, bool) else bool(str(given or '').strip())
        if value is None or later or count <= shown:
            return ''
        if pick == 'cheapest' and mem.price_sorted(observation):
            return ''
        pages = -(-(count - shown) // shown)
        spare = store.left() - order.tail()
        # b12: a single item's cheapest pick reads up to 4 further pages of a result (10 an episode), Halo 2 (6):
        # race 14's cheapest misses were single products whose fitting set the pilot never saw in full
        room, most = (4, AUTO_PAGES + 4) if pick == 'cheapest' and len(order.items) == 1 else (2, AUTO_PAGES)
        if pages > room and self.pages_read + room <= most and spare - room >= 12:
            pages = room
        if pages > room or self.pages_read + pages > most or spare - pages < 12:
            return f"\n[Arc] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with {param}={to_json(value)}."
        texts = []
        for _ in range(pages):
            faults: list[str] = []
            call = conform({**args, param: value}, store.schemas.get(name), 'args', faults)
            if faults:
                break
            reply = store.call(name, call)
            if reply is None or store.finished:
                break
            self.pages_read += 1
            seen = reply.get('observation')
            self.observe(name, seen)
            self.paged_obs.append(seen)
            shown_text = self.show(seen)
            self.read_cache[name + to_json(call)] = (s.version, store.step, shown_text)
            texts.append(f'{param}={to_json(value)}: {shown_text}')
            after = mem.paging(name, seen)
            value = after[3] if after else None
            if value is None:
                break
        s.log('pages', tool=name, read=len(texts))
        if not texts:
            return ''
        rest = f"\n[Arc] {count} listings matched and Arc has read {shown * (len(texts) + 1)} rows of them; narrow the search with the requirement's value or page on to see the rest." if value is not None else ''
        return '\n[Arc read the next pages of this search itself, one step a page]\n' + '\n'.join(texts) + rest

    def cap_ignored(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """A search whose maximum price was not applied: how to list the category below that price instead."""
        mem = self.s.mem
        props = self.s.store.props(name)

        def priced(key: str) -> bool:
            spec = props.get(key) if isinstance(props.get(key), dict) else {}
            return bool(re.search('price|cost|amount|budget|spend|money', key + ' ' + str(spec.get('description') or ''), re.I))
        caps = [k for k, v in args.items() if k in props and is_number(v) and re.search('max|most|under|below|limit|cap|upto|up_to', k, re.I) and priced(k)]
        if not caps or not mem.price_key:
            return ''
        caps.sort(key=lambda k: args[k])
        cap = args[caps[0]]
        over = [i for i, _ in mem.records_in(observation) if (mem.price_of(i) or 0) > cap]
        if not over:
            return ''
        modes = {k: v for k, v in args.items() if isinstance(v, str) and isinstance((props.get(k) or {}).get('enum'), list)}
        others = {k: [o for o in props[k]['enum'] if o != v] for k, v in modes.items()}
        hint = '; list the category with ' + ', '.join(f'{k}={to_json(o[0])}' for k, o in others.items() if o) + f' and the same {caps[0]} to see only cheaper listings' if any(others.values()) else ''
        return f'\n[Arc] This search did not apply {caps[0]}={cap:g}: {len(over)} of its rows cost more{hint}.'

    def cheapest_here(self, observations: list[Any]) -> str:
        mem = self.s.mem
        if self.s.order.pick() != 'cheapest' or not mem.price_key:
            return ''
        found = {i for seen in observations for i, _ in mem.records_in(seen)}
        rows = [(mem.price_of(i), i) for i in found if mem.buyable(i) and mem.price_of(i) is not None]
        if len(rows) < 2:
            return ''
        return '\n[Arc] buyable rows here, cheapest first: ' + ', '.join(f'{i} ({p:g})' for p, i in sorted(rows)[:6])

    def cap_template(self) -> tuple[str, dict[str, Any], str, str] | None:
        """The listing tool that takes a shelf and a maximum price, with the mode it needs: the mode an earlier listing
        by shelf used, else the one option naming a listing. None when no tool or mode is clear."""
        for tool, schema in self.s.store.schemas.items():
            props = (schema or {}).get('properties') or {}
            caps = [k for k, v in props.items() if isinstance(v, dict) and {'integer', 'number'} & set(type_list(v)) and re.search(CAP_ARG, snake(k), re.I) and not re.search(CAP_NOT, k, re.I)]
            shelves = [k for k, v in props.items() if isinstance(v, dict) and 'string' in type_list(v) and 'enum' not in v and re.search(SHELF_ARG, snake(k), re.I)]
            if len(caps) != 1 or len(shelves) != 1:
                continue
            base: dict[str, Any] = {}
            for key, spec in props.items():
                options = spec.get('enum') if isinstance(spec, dict) else None
                if not isinstance(options, list) or len(options) < 2:
                    continue
                pick = next((a[key] for a in reversed(self.searched) if shelves[0] in a and key in a and not self.worded(a, shelves[0], props)), None)
                named = [o for o in options if isinstance(o, str) and re.search(LIST_MODE, o, re.I)]
                pick = pick if pick is not None else named[0] if len(named) == 1 else None
                if pick is None:
                    return None
                base[key] = pick
            return tool, base, shelves[0], caps[0]
        return None

    @staticmethod
    def worded(args: dict[str, Any], shelf: str, props: dict[str, Any]) -> bool:
        """Whether a search's arguments carry query words besides the shelf."""
        return any(k != shelf and not (isinstance(props.get(k), dict) and 'enum' in props[k]) and isinstance(v, str) and v.strip() and re.search(QUERY_WORDS, snake(k), re.I) for k, v in args.items())

    def list_below(self, cart: list[str]) -> None:
        """For each cart line, list its category below its price (and one more page when it is short)."""
        s = self.s
        mem, store = s.mem, s.store
        plan = self.cap_template()
        cat_key = mem.category_key()
        if not plan or not cat_key:
            return
        tool, base, shelf, cap = plan
        for line in cart:
            price = mem.price_of(line)
            label = mem.label_of(line, cat_key)
            if price is None or price < 2 or not label or (line, price) in self.capped:
                continue
            if store.left() <= s.order.tail() + 5 or store.finished:
                return
            self.capped.add((line, price))
            faults: list[str] = []
            args = conform({**base, shelf: label, cap: int(price) - 1 if float(price).is_integer() else int(price)}, store.schemas.get(tool), 'args', faults)
            if faults:
                return
            reply = store.call(tool, args)
            if reply is None:
                return
            self.observe(tool, reply.get('observation'))
            s.log('list_below', line=line, price=price)
            page = mem.paging(tool, reply.get('observation'))
            if page and page[3] is not None and page[1] < page[0] <= 2 * page[1] and store.left() > s.order.tail() + 5 and not mem.price_sorted(reply.get('observation')):
                call = conform({**args, page[2]: page[3]}, store.schemas.get(tool), 'args', faults)
                more = None if faults else store.call(tool, call)
                if more is not None:
                    self.observe(tool, more.get('observation'))

    def reread_offers(self) -> str:
        """Read the cart lines' offers again when they were last seen before the latest refusal or two steps ago."""
        s = self.s
        mem, store = s.mem, s.store
        tool = store.offer_tool()
        target = store.single_target(tool)
        if not tool or not target:
            return ''
        since = max(store.step - 2, s.referee.refused_at)
        stale = [i for i in s.basket.ids() if self.offer_seen.get(i, -99) < since and i not in mem.gone]
        if not stale or store.left() < len(stale) + 3:
            return ''
        lines = []
        for ident in stale:
            faults: list[str] = []
            args = conform({target: ident}, store.schemas.get(tool), 'args', faults)
            if faults:
                break
            before = dict(mem.rows.get(ident) or {})
            reply = store.call(tool, args)
            if reply is None or store.finished:
                break
            observation = reply.get('observation')
            self.offer_seen[ident] = store.step
            self.observe(tool, observation)
            after = mem.rows.get(ident) or {}
            moved = [k for k, v in after.items() if k in before and before[k] != v and not isinstance(v, (dict, list))]
            if moved:
                lines.append(f"{ident} changed ({', '.join(moved[:4])}): {to_json(clip(observation, 300))[:900]}")
        s.log('reread', ids=stale, changed=len(lines))
        if lines:
            s.version += 1
        return '\n'.join(lines)

    def open_rows(self, rows: list[str]) -> bool:
        """Open listings seen only as rows (side by side when that view shows attributes); True when one was read."""
        s = self.s
        mem, store = s.mem, s.store
        many, joiner = store.compare_target(), store.roles.get('compare')
        tool = mem.facts_tool()
        target = store.single_target(tool)
        if many and joiner and not mem.compare_bare():
            plan = [(joiner, {many: rows[:store.compare_room()]})]
        elif tool and target:
            plan = [(tool, {target: i}) for i in rows[:2]]
        else:
            return False
        if store.left() < len(plan) + 5:
            return False
        before = len(mem.full)
        for name, args in plan:
            faults: list[str] = []
            args = conform(args, store.schemas.get(name), 'args', faults)
            if faults:
                return False
            reply = store.call(name, args)
            if reply is None or store.finished:
                break
            self.observe(name, reply.get('observation'))
        s.log('audit_open', rows=rows, opened=len(mem.full) - before)
        return len(mem.full) > before

    # -- search templates
    def search_template(self) -> tuple[str, dict[str, Any], str] | None:
        """The search tool, the fixed arguments of the latest worded search, and its query field."""
        s = self.s
        mem, store, order = s.mem, s.store, s.order
        tool = store.roles.get('search')
        schema = store.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = free_text_keys(schema)
        named = [k for k in texts if re.search(QUERY_WORDS, snake(k), re.I)]
        cat_key = mem.category_key()
        for args in reversed(self.searched):
            fields = [k for k in named or texts if isinstance(args.get(k), str) and re.search('[a-z]', args[k], re.I)]
            if not named:
                labels = {str(r.get(cat_key) or '').strip().lower() for r in mem.rows.values()} - {''} if cat_key else set()
                fields = [k for k in fields if len({str(a.get(k)) for a in self.searched if a.get(k)}) >= 2 and any(len(str(a.get(k)).split()) >= 2 for a in self.searched) and not any(str(a.get(k) or '').strip().lower() in labels for a in self.searched)]
            if not fields or not tool:
                continue
            field = fields[0]
            keep = {k: v for k, v in args.items() if k != field and (isinstance((props.get(k) or {}).get('enum'), list) or k in (schema.get('required') or []))}
            shelves = {plain(item.shelf or '') for item in order.items} | {plain(item.name) for item in order.items}
            shelves |= {plain(str(row.get(cat_key) or '')) for row in mem.rows.values()} if cat_key else set()
            keep = {k: v for k, v in keep.items() if k in (schema.get('required') or []) or not isinstance(v, str) or (plain(v) not in shelves and not any({w for w in stems(v) if len(w) >= 4} & {w for w in stems(x) if len(w) >= 4} for x in shelves if x))}
            worded = [a for a in self.searched if isinstance(a.get(field), str) and re.search('[a-z]', a[field], re.I)]
            for k in list(keep):
                if isinstance((props.get(k) or {}).get('enum'), list):
                    used = [to_json(a[k]) for a in worded if k in a]
                    if used:
                        best = max(used, key=used.count)
                        keep[k] = next(a[k] for a in worded if k in a and to_json(a[k]) == best)
            return tool, keep, field
        return None

    def value_template(self) -> tuple[str, dict[str, Any], str] | None:
        """The search a value search copies: search_template(), or, when every search so far listed a category without
        words, the tool's one query field with its one mode argument set to read it. None when in doubt."""
        store = self.s.store
        plan = self.search_template()
        tool = store.roles.get('search')
        if plan or not tool or not self.searched:
            return plan
        schema = store.schemas.get(tool) or {}
        props = schema.get('properties') or {}
        required = schema.get('required') or []
        named = [k for k in free_text_keys(schema) if re.search(QUERY_WORDS, snake(k), re.I)]
        last = self.searched[-1]
        enums = {k: v['enum'] for k, v in props.items() if isinstance(v, dict) and isinstance(v.get('enum'), list) and len(v['enum']) >= 2}
        modes = []
        for key, options in enums.items():
            reads = [o for o in options if isinstance(o, str) and re.search(QUERY_WORDS, o, re.I)]
            others = [o for o in options if o != last.get(key)]
            if len(reads) == 1:
                modes.append((key, reads[0]))
            elif key in required and len(options) == 2 and len(others) == 1:
                modes.append((key, others[0]))
        if len(named) != 1 or len(modes) > 1 or (enums and not modes):
            return None
        base = {k: last[k] for k in required if k in last and k != named[0]}
        base.update(modes)
        return tool, base, named[0]


# ---------------------------------------------------------------------------------------------------- Interview

class Interview(Part):
    """Questions to the shopper. Each item has a lane of properties (the planner's picks, then the keys its listings
    carry); a message names up to LIST_SIZE of one item's properties, and settle() rotates a lane after each answer."""

    def __init__(self, session: 'Session'):
        super().__init__(session)
        self.asked = False
        self.late_asked = False
        self.listing = True                      # lists stay on until a shopper asks for one thing at a time
        self.lanes: dict[int, list[str]] = {}
        self.found: dict[int, int] = {}
        self.spent: dict[int, int] = {}
        self.used: dict[int, set[str]] = {}
        self.misses: dict[int, int] = {}
        self.follow_up: list[tuple[int, list[str], bool]] = []
        self.one_product: set[int] = set()
        self.plan_order: list[int] = []

    # -- when
    def ready(self) -> bool:
        s = self.s
        if self.asked or not s.store.roles.get('ask') or s.order.msgs_free() <= 0:
            return False
        if s.store.left() <= 8 or s.models.clock() > ASK_BY:
            return False
        # b3 (27-know b17): ask once each item has two listings opened (one item: four) or by ASK_STEP. Halo waited for
        # 3 x items + 1 opened listings or a quarter of the steps; in race 15 that left 31 of its 180 episodes with one
        # question or none (all 3-4 product tasks, none passed), and Imperial 32
        items = len(s.order.items)
        if len(s.mem.full) >= (4 if items == 1 else 2 * items + 1):
            return True
        return s.store.step >= max(8, min(ASK_STEP, s.store.max_steps // 4))

    def wants_first(self, calls: list[dict[str, Any]]) -> bool:
        """Whether a pilot turn that commits (cart, buy, refuse) must wait for the interview."""
        s = self.s
        store = s.store
        committing = {store.roles.get('cart'), store.roles.get('buy'), store.roles.get('refuse')} - {None}
        return (not self.asked and bool(store.roles.get('ask')) and s.order.msgs_free() > 0 and any(c['function']['name'] in committing for c in calls)
                and store.left() > 8 and s.models.clock() <= ASK_BY)

    def start_lanes(self, count: int) -> None:
        self.lanes = self.plan_lanes(count)
        self.found = {n: 0 for n in self.lanes}
        self.spent = {n: 0 for n in self.lanes}
        self.used = {n: set() for n in self.lanes}

    # -- the interview
    def run(self) -> str:
        """Read listings, plan the lanes, ask while messages, steps and time allow, then search the told values.
        Returns the summary for the pilot ('' when nothing could be asked)."""
        s = self.s
        order, store = s.order, s.store
        self.asked = True
        if order.msgs_free() <= 0:
            return ''
        self.widen()
        count = order.msg_limit - order.msg_used
        self.start_lanes(max(count, LIST_SIZE) if self.listing else count)
        self.follow_up = []
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        while order.msgs_free() > 0 and not store.finished and store.left() > 6 and s.models.clock() < QUESTION_BY:
            batch = self.ask_next(told)
            if not batch:
                break
            sent += batch
        found_items = [n for n in self.found if self.found[n] > 0]
        # b8: an item the answers told nothing new about is searched by the request's own values (they were
        # never searched together by code: Halo searched only told values)
        stated = [n for n in self.found if self.found[n] == 0 and any(v for v, _ in self.value_words(n))]
        searched = self.value_search(found_items + stated, set(stated)) if found_items or stated else ''
        if told:
            s.version += 1
        summary = "Arc's interviewer asked the shopper (each question used one step):\n" + '\n'.join(f'Q: {q}\nA: {a}' for _, q, a in sent)
        summary += '\nNew firm requirements from these answers: ' + ('; '.join(told) if told else 'none')
        summary += '. Apply them to every pick (search again with the new values where needed).'
        pointed = [need for _, need in order.refs if any(need in t for t in told)]
        if pointed:
            summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.'
        if searched:
            summary += '\n' + searched
        if s.reader.queued:
            summary += '\n[notes seen meanwhile] ' + ' | '.join(s.reader.queued)
            s.reader.queued = []
        return summary

    def late(self) -> str:
        """Before a checkout, ask the next planned question with a message the order itself does not need: the
        interview keeps back every message a process step may use, and when Arc is about to buy, the ones kept for an
        ending without a purchase are free. Returns the summary when the answer adds a requirement, else ''."""
        s = self.s
        order, store = s.order, s.store
        if self.late_asked or not store.roles.get('ask') or order.msg_used >= order.msg_limit:
            return ''
        if any(p['when'] == 'order' and p['messages'] > 0 for p in order.process):
            return ''
        if store.left() < LATE_STEPS or s.models.clock() > LATE_BY or not s.models.judges():
            return ''
        self.late_asked = True
        if not self.lanes:
            self.start_lanes(LIST_SIZE if self.listing else 1)
        told: list[str] = []
        budget = order.budget
        batch = self.ask_next(told, LATE_STEPS - 2)
        if not batch:
            s.log('late_ask', skipped='no property left to ask or no message sent')
            return ''
        if order.budget != budget:
            told.append(f'budget now {order.budget}')
        number, question, answer = batch[0]
        s.log('late_ask', q=question, a=answer[:200], told=told)
        if not told:
            return ''
        s.version += 1
        summary = f'Arc asked the shopper one more question before checking out (one step):\nQ: {question}\nA: {answer}\nNew firm requirements from this answer: ' + '; '.join(told) + '. The cart was not checked out: check each line against them (search again with the new values where needed), fix the cart, then check out.'
        pointed = [need for _, need in order.refs if any(need in t for t in told)]
        if pointed:
            summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value.'
        searched = self.value_search([number])
        return summary + ('\n' + searched if searched else '')

    # -- one question
    def ask_next(self, told: list[str], floor: int = 6) -> list[tuple[int, str, str]]:
        """Send the next question (a pending follow-up first, else the next planned labels), read its answer into
        the items' needs (each new one appended to `told`) and settle the labels it named; [] when nothing was sent."""
        s = self.s
        order = s.order
        for _ in range(LIST_SIZE):
            self.follow_up = [f for f in self.follow_up if not self.told_about(f[0], f[1][0])]
            if self.follow_up:
                number, labels, shown = self.follow_up.pop(0)
            else:
                number, labels = self.next_labels()
                shown = False
            if not labels:
                return []
            fitted, text = self.fit_question(number, labels, shown)
            if not fitted and shown:
                fitted, text = self.fit_question(number, labels, False)
            if fitted:
                break
            self.used.setdefault(number, set()).add(plain(labels[0]))
            s.log('ask_too_long', item=number, label=labels[0])
        else:
            return []
        labels = fitted
        batch = self.send([(number, labels[0], text)], floor)
        if not batch:
            return []
        self.spent[number] = self.spent.get(number, 0) + 1
        had = [len(item.told) for item in order.items]
        heard = self.hear(batch, (LIGHT, HEAVY))
        if len(labels) > 1:
            heard.pop('free', None)
        self.take_heard(heard, told)
        new = [need for item, size in zip(order.items, had) for need in item.told[size:]]
        for n, (item, size) in enumerate(zip(order.items, had), 1):
            if len(item.told) > size:
                self.found[n] = self.found.get(n, 0) + 1
        crowded = any(as_int(x) == 1 for x in as_list(heard.get('crowded')))
        self.settle(number, labels, shown, new, crowded)
        return batch

    def next_labels(self) -> tuple[int, list[str]]:
        """The item to ask about next (no need found yet first, then the fewest messages spent, then the plan's
        order) and the labels for its message."""
        first = {n: i for i, n in enumerate(self.plan_order)}
        scarce = self.s.order.msg_limit <= SCARCE_MESSAGES and not self.listing
        for number in sorted(self.lanes, key=lambda n: (self.found.get(n, 0) > 0, 0 if scarce else self.spent.get(n, 0), first.get(n, len(first) + n))):
            left = [label for label in self.lanes[number] if plain(label) not in self.used[number] and not self.told_about(number, label)]
            if left:
                return number, left[:LIST_SIZE if self.listing else 1]
        return 0, []

    def settle(self, number: int, labels: list[str], shown: bool, new: list[str], crowded: bool) -> None:
        """Mark what an answer showed about the labels its question named.

        One label: it is asked; a reply asking for a narrower question gets a follow-up (one product of an item name
        joining several, the property's head word, or example values). A list: the shopper answers about the first
        property of a message it holds a view on, and a reply naming none is the same as "don't mind" on the first
        one. So a disclosed need settles its label and every label before it, and the labels after it lead the next
        list; a reply that discloses nothing settles only the first label, and the others come next in reverse order;
        after a second such reply in a row the labels not yet named lead. A reply asking for one question at a time
        ends lists for the episode and the list's first label is asked alone."""
        s = self.s
        used = self.used.setdefault(number, set())
        if len(labels) == 1:
            used.add(plain(labels[0]))
            if new or not crowded:
                return
            if number not in self.one_product and self.name_parts(number):
                self.one_product.add(number)
                s.log('one_product', item=number, said=self.item_said(number))
                self.follow_up.insert(0, (number, labels, shown))
                return
            head = labels[0].split()[-1]
            if len(labels[0].split()) >= 2 and plain(head) not in used:
                self.follow_up.insert(0, (number, [head], False))
            elif not shown and len(self.samples(number, labels[0])) >= 2:
                self.follow_up.insert(0, (number, labels, True))
            return
        if crowded and not new:
            self.listing = False
            s.log('lists_off', item=number)
            self.follow_up.insert(0, (number, labels[:1], False))
            return
        hit = self.answered_label(labels, new) if new else -1
        used.update(plain(label) for label in labels[:max(hit, 0) + 1])
        rest = [label for label in self.lanes.get(number) or [] if label not in labels]
        self.misses[number] = 0 if hit >= 0 else self.misses.get(number, 0) + 1
        if hit >= 0:
            ahead = labels[hit + 1:]
            self.lanes[number] = ahead + rest
        else:
            # b17: a reply naming no need shifts the list by HEARD_LABELS: its labels from that place on lead the next
            # list, in the plan's order, and its second and third go last. A shopper tells a need named in the first
            # three places most often (r14: 81/64/59%, then 46-17%; the arena's race-14 shopper: only those three),
            # and the old reversal never put the fourth and fifth labels first
            ahead = labels[HEARD_LABELS:]
            self.lanes[number] = ahead + rest + labels[1:HEARD_LABELS]
        s.log('list_settled', item=number, done=labels[:hit + 1], next=ahead[:4])

    def answered_label(self, labels: list[str], new: list[str]) -> int:
        """Index of the earliest label a list's reply answered: each new need maps to the label sharing most words
        with its property, else to the label whose values on opened listings share most words with its value."""
        seen: dict[str, set[str]] = {}
        for _, key, _, values in self.s.mem.key_table():
            seen.setdefault(plain(key), set()).update(w for v in values for w in roots(v))
        hits = []
        for need in new:
            prop, _, value = need.partition(':')
            scores = [len(spellings(label) & spellings(prop)) for label in labels]
            if max(scores, default=0) == 0:
                scores = [len(seen.get(plain(label), set()) & roots(value)) for label in labels]
            if max(scores, default=0) > 0:
                hits.append(scores.index(max(scores)))
        return min(hits) if hits else -1

    def told_about(self, number: int, label: str) -> bool:
        """Whether an answer already gave a need on this property of the item (colour and color, plurals alike)."""
        mine = spellings(label)
        for need in self.s.order.items[number - 1].told:
            prop = spellings(need.split(':', 1)[0])
            if same_property(mine, prop):
                return True
            # b17: a told property of one long word also covers the labels that contain it ("processor" told: no
            # "processor speed"); such a label drew the told need again and used up a message (arena r14: 3 of 73)
            if prop and prop <= mine and (len(prop) >= 2 or max(len(w) for w in prop) >= 5):
                return True
        return False

    # -- the question's words
    def question_text(self, number: int, labels: list[str], shown: bool = False) -> str:
        if len(labels) >= 2:
            return f"What {', '.join(labels[:-1])} or {labels[-1]} would you like for the {self.item_said(number)}?"
        samples = self.samples(number, labels[0]) if shown else None
        text = f'What {labels[0]} would you like for the {self.item_said(number)}'
        return text + (f', for example {samples[0]} or {samples[1]}?' if samples else '?')

    def fit_question(self, number: int, labels: list[str], shown: bool) -> tuple[list[str], str]:
        """The longest leading part of `labels` whose question the message tool takes whole (a list too long for its
        text cap drops labels from the end), with that question; ([], '') when not even one label fits."""
        for size in range(len(labels), 0, -1):
            text = self.question_text(number, labels[:size], shown)
            args = self.s.store.message_args(text)
            if args and text in args.values():
                return labels[:size], text
        return [], ''

    def name_parts(self, number: int) -> list[str]:
        """The products an item name joins with "and" or commas when each is a plural noun ("door handles and
        locks"); a single product named with "and" ("salt and pepper grinder") gives []."""
        name = self.s.order.items[number - 1].name
        if not re.search('\\band\\b', name, re.I):
            return []
        parts = [p.strip() for p in re.split('\\s*(?:,|\\band\\b)\\s*', name, flags=re.I) if p.strip()]
        return parts if len(parts) >= 2 and all(re.search('[a-z]{3}(?<![su])s$', p, re.I) for p in parts) else []

    def item_said(self, number: int) -> str:
        """The item as a question names it: plural nouns joined with "and" are asked with "or"; after a one-at-a-time
        reply about the item, the one product its own rows name most."""
        s = self.s
        mem, order = s.mem, s.order
        item = order.items[number - 1]
        parts = self.name_parts(number)
        if not parts:
            return item.name
        if number not in self.one_product:
            return re.sub('\\band\\b', 'or', item.name, flags=re.I)
        cat_key = mem.category_key()
        rows = [row for row in mem.rows.values() if not cat_key or order.owns(item, str(row.get(cat_key) or ''))]
        named = [roots(' '.join(str(v) for k, v in row.items() if k != cat_key and isinstance(v, str))) for row in rows]
        hits = {p: sum(1 for words in named if roots(p.split()[-1]) & words) for p in parts}
        return max(parts, key=lambda p: (hits[p], -parts.index(p)))

    def samples(self, number: int, label: str) -> list[str]:
        want = plain(label)
        settled = ' '.join(self.s.order.items[number - 1].musts).lower()
        for _, name, _, values in self.s.mem.key_table():
            if name == want:
                return [v for v in values if v and v not in settled and len(v) <= 30][:2]
        return []

    # -- sending and hearing
    def send(self, questions: list[tuple[int, str, str]], floor: int = 6) -> list[tuple[int, str, str]]:
        s = self.s
        store, order = s.store, s.order
        tool = store.roles['ask']
        sent: list[tuple[int, str, str]] = []
        for number, _, text in questions:
            if store.finished or store.left() <= floor or order.msg_used >= order.msg_limit:
                break
            args = store.message_args(text)
            if not args:
                break
            reply = store.call(tool, args)
            if reply is None:
                break
            order.msg_used += 1
            answer = reply.get('observation')
            words = s.reader.spoken(answer)
            sent.append((number, text, words))
            s.referee.record(f'step {store.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {s.basket.mark()})', store.step)
            s.log('ask', q=text, a=words[:300])
            if isinstance(answer, dict) and answer.get('error'):
                break
        return sent

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...] = (HEAVY, LIGHT)) -> dict[str, Any]:
        items = self.s.order.items
        prompt = P_HEAR + '\n\nITEMS:\n' + '\n'.join(f'item {n}: {it.name}' for n, it in enumerate(items, 1))
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join(f'{i}. (about item {n}) Q: {q}\nA: {a}' for i, (n, q, a) in enumerate(sent, 1))
        return self.s.models.ask(prompt, list(models), 2500) or {}

    def take_heard(self, heard: dict[str, Any], told: list[str]) -> None:
        order = self.s.order
        many = len(order.items) > 1
        for entry in as_list(heard.get('needs')):
            if not isinstance(entry, dict) or not str(entry.get('need') or '').strip():
                continue
            number = order.item_index(entry.get('item'))
            need = str(entry['need']).strip()
            if need not in order.items[number - 1].told:
                order.items[number - 1].told.append(need)
                told.append(f'item {number}: {need}' if many else need)
            if entry.get('points') is True and (number, need) not in order.refs:
                order.refs.append((number, need))
        for entry in as_list(heard.get('drop')):
            if isinstance(entry, dict) and str(entry.get('property') or '').strip():
                item = order.items[order.item_index(entry.get('item')) - 1]
                gone = plain(str(entry['property']).split(':')[0])
                hit = [m for m in item.musts + item.told if m not in item.dropped and plain(m.split(':')[0]) == gone]
                item.dropped.extend(hit)
                told.extend(f'dropped {m}' for m in hit)
        for entry in as_list(heard.get('free')):
            if isinstance(entry, dict) and entry.get('property'):
                order.items[order.item_index(entry.get('item')) - 1].free.append(str(entry['property']))
        if is_number(heard.get('budget')) and heard['budget'] > 0:
            order.budget = heard['budget']

    # -- planning the lanes
    def plan_lanes(self, count: int) -> dict[int, list[str]]:
        """Each item's lane of properties: the planner's ranked picks with up to LANE_GUARDS of the item's most carried
        keys moved forward, then the rest of its keys; catch-all keys go last."""
        s = self.s
        order, mem = s.order, s.mem
        order.adopt_labels()
        table = mem.key_table()
        prompt = self.plan_prompt(count, table)
        plan = s.models.ask(prompt, [HEAVY, LIGHT], 3500) or {}
        lanes: dict[int, list[str]] = {n: [] for n in range(1, len(order.items) + 1)}
        hinted: dict[int, set[str]] = {n: set() for n in lanes}
        for pick in as_list(plan.get('ask')):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, as_int(pick.get('item') or 1) or 1), len(order.items))
            label = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(pick.get('label') or ''))[0].strip().rstrip('?.')
            if not label or len(label.split()) > 5:
                continue
            if all(plain(label) != plain(other) for other in lanes[number]):
                lanes[number].append(label)
            if str(pick.get('hint') or '').strip():
                hinted[number].add(plain(label))
                if number not in self.plan_order:
                    self.plan_order.append(number)
            elif pick.get('spec') is True:
                # b19: the product's main specification named by no listed key keeps its place like a hinted pick (the
                # guard moved such a pick out of the first three, its share of the listings being 0)
                hinted[number].add(plain(label))
        for number in lanes:
            lanes[number] = self.guarded_lane(number, lanes[number], hinted[number], table)
        s.log('plan', ask=clip(plan.get('ask'), 120), lanes={n: v[:8] for n, v in lanes.items()})
        return lanes

    def plan_prompt(self, count: int, table: list[tuple[str, str, int, list[str]]]) -> str:
        order = self.s.order
        settled = [f'item {number}: {item.name} - settled by the request: ' + ('; '.join(item.musts) or 'nothing') for number, item in enumerate(order.items, 1)]
        listing = []
        last = None
        shown: dict[str, int] = {}
        index = 0
        for label, name, carried, values in table:
            shown[label] = shown.get(label, 0) + 1
            if shown[label] > PLAN_SHOWN:
                continue
            index += 1
            if label != last:
                listing.append(f'category {label!r}:')
                last = label
            listing.append(f'  {index}. {name} (on {carried}): ' + ' | '.join(values) + self.not_stated(label, name, values) + (' [several features in one: ask last]' if re.search(CATCH_ALL, name) else ''))
        prompt = P_PLAN.replace('<COUNT>', str(max(1, count))) + '\n\nREQUEST:\n' + order.request + '\n\nITEMS:\n' + '\n'.join(settled)
        first = [str(x) for x in as_list(order.brief.get('ask_first')) if str(x).strip()]
        if first:
            prompt += '\n\nASK FIRST (the request asks to be asked about these):\n' + '\n'.join(first)
        return prompt + '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(listing) or '(no full listing records yet)')

    def guarded_lane(self, number: int, picks: list[str], hinted: set[str], table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        s = self.s
        order = s.order
        keys = self.lane_keys(number, table)
        read = max(1, order.records_of_item(number))
        own: dict[str, float] = {}
        for label, key, carried, _ in table:
            if order.owner_of(label) == number:
                own[key] = max(own.get(key, 0.0), carried / read)

        def spelled(label: str) -> str:
            fold = lambda text: re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', plain(text)))
            return next((k for k in own if fold(k) == fold(label)), label)

        def share_of(label: str) -> float:
            if plain(label) in own:
                return own[plain(label)]
            near = [v for k, v in own.items() if same_property(stems(label), stems(k)) or same_property(stems(k), stems(label))]
            return max(near) if near else 0.0
        hinted = {plain(spelled(h)) for h in hinted}
        lane: list[str] = []
        for pick in picks:
            if all(plain(spelled(pick)) != plain(other) for other in lane):
                lane.append(spelled(pick))
        for key in [k for k in keys if own.get(k, 0.0) >= LANE_GUARD_SHARE and not re.search(CATCH_ALL, k)][:LANE_GUARDS]:
            if any(plain(key) == plain(other) for other in lane[:3]):
                continue
            weak = next((i for i, other in enumerate(lane[:3]) if plain(other) not in hinted and share_of(other) + LANE_GUARD_GAP <= own.get(key, 0.0)), None)
            if weak is None and len(lane) >= 3:
                continue
            placed = [other for other in lane if plain(other) != plain(key)]
            if weak is not None and len(order.items) > 1:
                moved = placed[weak]
                placed[weak] = key
                placed.insert(min(3, len(placed)), moved)
            else:
                placed.insert(min(len(placed), weak if weak is not None else len(placed)), key)
            if any(plain(x) in hinted and all(plain(x) != plain(y) for y in placed[:3]) for x in lane[:3]):
                continue
            lane = placed
        for key in keys:
            if all(plain(key) != plain(other) for other in lane):
                lane.append(key)
        return [x for x in lane if not re.search(CATCH_ALL, plain(x))] + [x for x in lane if re.search(CATCH_ALL, plain(x))]

    def lane_keys(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        """The item's own attribute keys, most carried first and measured keys after the others; settled properties
        and identifier-like keys left out, catch-all keys last."""
        order = self.s.order
        item = order.items[number - 1]
        settled = [stems(m.split(':', 1)[0]) for m in item.musts]
        rows = [(label, key, carried, len(values)) for label, key, carried, values in table if order.owner_of(label) == number]
        measured = {key: sum(1 for v in values if re.match('^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$', v)) * 2 > len(values) for _, key, _, values in table}
        if len(order.items) == 1 and not rows:
            rows = [(label, key, carried, len(values)) for label, key, carried, values in table]
        rows.sort(key=lambda r: (-r[2], measured.get(r[1], False), -r[3]))
        keys = [key for _, key, _, _ in rows if not any(same_property(prop, stems(key)) for prop in settled) and not re.search(NOT_PROPERTY, key)]
        return [key for key in keys if not re.search(CATCH_ALL, key)] + [key for key in keys if re.search(CATCH_ALL, key)]

    def not_stated(self, label: str, name: str, values: list[str]) -> str:
        """A planner mark on a key that shares words with a requirement's property but never shows its value."""
        order = self.s.order
        number = order.owner_of(label)
        if not number:
            return ''
        for must in order.items[number - 1].musts:
            prop, _, value = must.partition(':')
            if re.search(LIMIT_WORDS, value, re.I):
                continue
            words = set(re.findall('[a-z0-9]{3,}', value.lower()))
            shown = set(re.findall('[a-z0-9]{3,}', ' '.join(values).lower()))
            if prop.strip() and plain(prop) != name and stems(prop) & stems(name) and words and not words <= shown:
                return f' [not the stated {prop.strip()}: another property]'
        return ''

    # -- reading before planning
    def widen(self) -> None:
        """Before planning, open listings of each requested kind side by side (or one by one)."""
        s = self.s
        mem, store = s.mem, s.store
        tool = store.roles.get('compare')
        schema = store.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in type_list(v)), None)
        if not tool or not target or any(k != target for k in schema.get('required') or []) or mem.compare_bare():
            self.widen_singly()
            return
        if store.left() <= 12:
            return
        cap = props[target].get('maxItems') if isinstance(props[target].get('maxItems'), int) else 4
        ranked = mem.item_groups()
        share = max(1, cap // max(1, len(ranked)))
        picks: list[str] = []
        for ids in ranked:
            picks += [i for i in ids if i not in mem.full][:share]
        picks = picks[:cap]
        if len(picks) < 2:
            return
        faults: list[str] = []
        args = conform({target: picks}, schema, 'args', faults)
        if faults:
            return
        before = len(mem.full)
        reply = store.call(tool, args)
        if reply is None:
            return
        seen = reply.get('observation')
        s.reader.observe(tool, seen)
        mem.note_view(tool, seen)
        rows = mem.records_in(seen)
        if rows:
            mem.facts_from[tool] = mem.facts_from.get(tool, False) or any(attributes(r) for _, r in rows)
        s.log('widen', ids=picks, opened=len(mem.full) - before)
        if rows and mem.facts_from.get(tool) is False:
            self.widen_singly()
            return
        if len(mem.full) == before and len(picks) > 2 and store.left() > 12:
            faults = []
            args = conform({target: picks[:2]}, schema, 'args', faults)
            if not faults:
                reply = store.call(tool, args)
                if reply is not None:
                    s.reader.observe(tool, reply.get('observation'))

    def widen_singly(self) -> None:
        s = self.s
        mem, store = s.mem, s.store
        tool = mem.facts_tool()
        target = store.single_target(tool)
        if not tool or not target:
            return
        wanted = {w for _, need in s.order.needs_now() for w in re.findall('[a-z0-9]{3,}', need.split(':', 1)[-1].lower())}
        each = OPEN_EACH if len(s.order.items) <= 2 else 4
        lanes: list[list[str]] = []
        heads: list[int] = []
        for ids in mem.item_groups():
            opened = sum(1 for i in ids if i in mem.full)
            pool = [i for i in ids if i not in mem.full and mem.buyable(i)]
            if s.order.pick() == 'cheapest':
                pool.sort(key=lambda i: mem.price_of(i) if mem.price_of(i) is not None else float('inf'))
            pool.sort(key=lambda i: -len(wanted & set(re.findall('[a-z0-9]{3,}', to_json(mem.rows.get(i) or {}).lower()))))
            lanes.append(pool[:max(0, each - opened)])
            heads.append(max(0, 4 - opened))
        first = [lane[k] for k in range(4) for lane, head in zip(lanes, heads) if k < min(head, len(lane))]
        more = [lane[k] for k in range(each) for lane, head in zip(lanes, heads) if head <= k < len(lane)]
        cap = max(5, 2 * len(lanes) + 2)
        read = s.reader.fill_facts(first[:cap][:8], 22)
        s.reader.fill_facts((first[cap:] + more)[:max(0, 10 - len(read))], 30)

    # -- searching the told values
    def value_words(self, number: int) -> list[tuple[str, bool]]:
        """Search words for each requirement value of the item a word search can match, each with whether an answer
        gave it; an answer's lower limit with a unit gives its number and unit."""
        order = self.s.order
        out = []
        told = order.items[number - 1].told
        pointers = order.pointer_needs()
        for n, need in order.needs_now():
            if n != number or need in pointers or ':' not in need:
                continue
            value = need.split(':', 1)[1].strip()
            edge = re.search(FLOOR, value, re.I)
            if edge:
                if need in told:
                    out.append((edge.group(1).strip(), True))
                continue
            if not value or re.search('\\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', value, re.I):
                continue
            out.append((value, need in told))
        return out

    def value_search(self, numbers: list[int], stated: set[int] | None = None) -> str:
        """For each item with a told value, search its values together and read the first rows of its kind; an item in
        `stated` is searched by the values the request states for it."""
        s = self.s
        mem, store, order = s.mem, s.store, s.order
        plan = s.reader.value_template()
        if not plan or store.left() < 22:
            return ''
        tool, base, field = plan
        seen: list[str] = []
        # b15: on a tight step budget (r11/r14: 35-43 calls, of which Arc's own reads took 13.5) the value search gets
        # 4 calls and opens one row an item instead of 8 and 3
        tight = store.max_steps <= TIGHT_STEPS
        budget = 4 if tight else 8
        for number in numbers:
            pairs = self.value_words(number)
            item = order.items[number - 1]
            values = [v for v, _ in pairs]
            fresh = [v for v, told in pairs if told or number in (stated or set())]
            if not fresh or budget <= 2:
                continue
            lead = [v for v in fresh if len(v.split()) == 1 or re.fullmatch(AMOUNT, v, re.I)]
            values = lead + [v for v in values if v not in lead]
            stated = [i for i, v in enumerate(values) if v not in fresh]
            drop = stated[-1] if stated else 0
            if len(values) > 1:
                tries = [values, values[:drop] + values[drop + 1:]]
            elif lead:
                tries = [[values[0], item.name], values]
            else:
                tries = [values]
            for words in tries[:2]:
                if not words:
                    continue
                faults: list[str] = []
                args = conform({**base, field: ' '.join(words)}, store.schemas.get(tool), 'args', faults)
                if faults or store.finished:
                    break
                reply = store.call(tool, args)
                budget -= 1
                if reply is None:
                    break
                observation = reply.get('observation')
                s.reader.observe(tool, observation)
                cat_key = mem.category_key()
                found = [i for i, _ in mem.records_in(observation)]
                cap = order.budget if is_number(order.budget) else None
                prices = [mem.price_of(i) for i in found]
                if cap is not None and prices and all(p is not None and p > 10 * cap for p in prices):
                    cap = None
                rows = [i for i in found if mem.buyable(i) and (not cat_key or order.owns(item, mem.label_of(i, cat_key))) and (cap is None or (mem.price_of(i) or 0) <= cap)]
                home = order.home_label(number)
                if cat_key and home:
                    rows.sort(key=lambda i: plain(mem.label_of(i, cat_key)) != home)
                seen.append(f'{to_json(args.get(field))}: ' + (', '.join(rows[:6]) or 'no row of that kind'))
                if rows:
                    opened = s.reader.fill_facts([i for i in rows if i not in mem.full][:min(1 if tight else 3, budget)], order.tail() + 16)
                    budget -= len(opened)
                    if opened:
                        seen.append('opened: ' + ' | '.join(opened)[:2400])
                    break
        if not seen:
            return ''
        s.log('value_search', seen=[x[:200] for x in seen])
        return "[Arc searched each such item's requirement values together and read the first fitting-kind rows] " + '\n'.join(seen)


# ------------------------------------------------------------------------------------------------------ Referee

class Referee(Part):
    """Everything checked before an ending: process steps the request asks for, the cart review, pointer chains,
    parts that must fit a device, sold-out lines and their replacements, and the evidence before ending unbought."""

    def __init__(self, session: 'Session'):
        super().__init__(session)
        self.gate_ok: set[tuple[str, str]] = set()
        self.done_log: list[str] = []
        self.done_steps: list[int] = []
        self.checked: set[str] = set()
        self.verdicts: dict[str, bool] = {}
        self.audited = False
        self.leads: list[str] = []
        self.lead_items: dict[str, list[str]] = {}
        self.refuse_blocks = 0
        self.pushed = False
        self.late_held = False
        self.confirmed: str | None = None
        self.completed = False
        self.trails = 0
        self.trail_blocks = 0
        self.link_searches = 0
        self.link_seen: dict[str, list[str]] = {}
        self.fit_holds = 0
        self.held = 0
        self.wider_held = False
        self.cheaper_marks: set[str] = set()
        self.refused_at = -1
        self.buys = 0
        self.early_done = False

    def record(self, line: str, step: int) -> None:
        self.done_log.append(line)
        self.done_steps.append(step)

    def bought(self, reply: dict[str, Any]) -> None:
        self.buys += 1
        if not reply.get('finished'):
            self.refused_at = self.s.store.step

    # -- process steps (approval, a message, a check) due before an ending
    def gate(self, stage: str) -> tuple[bool, str]:
        """Carry out the request's steps due before the order or the ending. (go on?, what was done or what changed)."""
        s = self.s
        order, store = s.order, s.store
        steps = [p for p in order.process if p['when'] == stage]
        if not steps or store.finished or not order.gate_due(stage):
            return True, ''
        ending = store.enders()
        tools = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {to_json(t['function'].get('parameters') or {})}" for t in store.tools if t['function']['name'] not in ending]
        said: list[str] = []
        nudged = False
        for _ in range(GATE_ROUNDS):
            prompt = (P_GATE.replace('<STAGE>', 'place the order' if stage == 'order' else 'end without a purchase') + '\n\nREQUEST:\n' + order.request
                      + '\n\nSTEPS DUE NOW:\n' + '\n'.join(f"- {p['step']}" for p in steps) + '\n\nLEDGER:\n' + order.ledger() + '\n\nCART NOW: ' + s.basket.text()
                      + '\n\nRECORD (messages and step calls so far):\n' + ('\n'.join(f'{i}. {line}' for i, line in enumerate(self.done_log, 1)) or '(nothing yet)')
                      + ('\n\nSTORE NOTES SEEN:\n' + '\n'.join(order.notes[-4:]) if order.notes else '')
                      + f"\n\nMESSAGES LEFT: {max(0, order.msg_limit - order.msg_used)}   STEPS LEFT: {store.left()} (the {('order' if stage == 'order' else 'ending')} itself takes one)"
                      + '\n\nTOOLS:\n' + '\n'.join(tools))
            got = s.models.ask(prompt, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 3500)
            if got is None:
                break
            told: list[str] = []
            hold = ''
            if said:
                s.talk.take_heard({'needs': got.get('needs')}, told)
                hold = str(got.get('hold') or '').strip()
            call = got.get('call') if isinstance(got.get('call'), dict) else None
            s.log('gate', stage=stage, call=clip(call, 200), steps=clip(got.get('steps'), 120), hold=hold[:200], told=told)
            if told or hold:
                s.version += 1
                return False, ('Before going on, Arc did the steps the request asks for:\n' + '\n'.join(said) + '\n' if said else '') + ("The shopper's reply changes the task: " + '; '.join(told + ([hold] if hold else [])) + '. Fix the cart for it, then check out again.')
            name = str((call or {}).get('tool') or '')
            shown = [e for e in as_list(got.get('steps')) if isinstance(e, dict)]
            talk = stage == 'order' and any(p['messages'] > 0 for p in steps)
            proof = {i for i in range(1, len(self.done_log) + 1) if not talk or (i <= len(self.done_steps) and self.done_steps[i - 1] > s.basket.changed_at)}
            unproven = [e for e in shown if e.get('done') is not True or not any((as_int(n) or 0) in proof for n in as_list(e.get('lines')))]
            if not call and unproven and not said and not nudged and store.roles.get('ask') and order.msg_used < order.msg_limit and store.left() >= 2:
                nudged = True
                s.log('gate_unproven', steps=clip(unproven, 120))
                self.record(f'(Arc: no line made since the cart last changed at step {s.basket.changed_at} shows ' + '; '.join(str(e.get('step') or '') for e in unproven)[:200] + f' done, and {order.msg_limit - order.msg_used} message(s) and {store.left()} steps are left: give the call that does it)', -1)
                continue
            if not call or name not in store.schemas or name in ending:
                if not call:
                    self.gate_ok.add((stage, s.basket.mark() + f'#{s.version}'))
                break
            if store.left() < 2 or (name == store.roles.get('ask') and order.msg_used >= order.msg_limit):
                break
            faults: list[str] = []
            args = conform(call.get('args') if isinstance(call.get('args'), dict) else {}, store.schemas[name], 'args', faults)
            if faults:
                break
            if name == store.roles.get('ask'):
                text = next((v for v in args.values() if isinstance(v, str)), '')
                batch = s.talk.send([(1, '', text)], 1)
                if not batch:
                    break
                order.gate_msgs += 1
                said.append(f'Q: {text}\nA: {batch[0][2]}')
                continue
            reply = store.call(name, args)
            if reply is None:
                break
            seen = reply.get('observation')
            s.mem.absorb(seen)
            self.record(f'step {store.step}: {name} {to_json(args)} -> {s.reader.show(seen)[:300]} (cart then: {s.basket.mark()})', store.step)
            said.append(f'{name} {to_json(args)} -> {s.reader.show(seen)[:400]}')
        return True, ('Before going on, Arc did the steps the request asks for:\n' + '\n'.join(said) if said else '')

    # -- the cart review
    def check_key(self, ids: list[str]) -> str:
        return '|'.join(sorted(ids)) + f'#{self.s.version}'

    def check_cart(self, ids: list[str] | None = None, limit: float = SPEND_SOFT) -> str:
        """A model review of the cart (or of planned lines); the problems it finds, or '' when it passes or is skipped."""
        s = self.s
        order, mem, models = s.order, s.mem, s.models
        lead = ids is not None
        ids = ids if lead else s.basket.ids()
        key = self.check_key(ids)
        if key in self.checked or not models.judges(limit) or not ids:
            return ''
        self.checked.add(key)
        prompt = P_CHECK + '\n\nREQUEST:\n' + order.request + '\n\nREQUIREMENTS IN FORCE:\n' + order.ledger()
        if lead:
            quantity = '; '.join(f'item {n}: quantity {it.qty}' for n, it in enumerate(order.items, 1))
            prompt += '\n\nCART (planned, judge these listings as if they were carted at the requested quantities: ' + quantity + '):\n' + ', '.join(ids)
        else:
            prompt += '\n\nCART:\n' + s.basket.text()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + '\n'.join(f'{i}: ' + ('SOLD OUT (the store says it can no longer be bought). ' if mem.sold_out(i) else '') + mem.record_text(i) for i in ids)
        ranked = order.pick() in ('cheapest', 'most', 'least')
        if ranked:
            prompt += '\n\nOTHER LISTINGS SEEN (newest first):\n' + mem.others(ids, 15)
        judges = models.judges(limit)
        got = None
        if len(judges) > 1 and not lead and not self.audited and not ranked:
            # a light review first; the heavy one only when it finds a problem
            got = models.ask(prompt, [LIGHT], 3000)
            if got:
                judges = [HEAVY] if got.get('ok') is False and HEAVY in models.judges(limit) else []
        if judges and not models.no_credit:
            got = models.ask(prompt, judges, 3000) or got
        s.log('check', cart=ids, got=got)
        self.verdicts[key] = bool(got) and got.get('ok') is not False
        if not got or got.get('ok') is not False:
            return ''
        self.leads = [i for i in self.leads if i not in ids]
        problems = [str(p) for p in as_list(got.get('problems'))][:5]
        better = [str(b) for b in as_list(got.get('better')) if str(b) in mem.rows and not mem.sold_out(str(b))][:4]
        text = 'A careful review says this cart is not right yet: ' + '; '.join(problems or ['see the requirements'])
        if better:
            text += '. Listings that look better: ' + ', '.join(better)
        return text + '. Fix the cart (or show the review is wrong by re-reading the records), then check out.'

    def wider_hold(self) -> str:
        s = self.s
        mem, store = s.mem, s.store
        if self.wider_held or s.order.pick() != 'cheapest' or store.left() < 14 + 2 * len(s.order.items):
            return ''
        self.wider_held = True
        lines = [f'{i} at {mem.price_of(i):g}' for i in s.basket.ids() if mem.price_of(i) is not None]
        return f"Not run yet (no step used). {store.left()} steps are left and the pick is the cheapest. Searches may read only part of a listing, so a requirement's value finds only some of the fitting listings, and rows come in no price order. Before checking out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line" + (f" ({'; '.join(lines)})" if lines else '') + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'

    def cheaper_rows(self) -> list[str]:
        """For a cheapest pick, once per cart state: list each line's category below its price, then read the two
        cheapest unread rows of its kind under it."""
        s = self.s
        mem, basket = s.mem, s.basket
        if s.order.pick() != 'cheapest':
            return []
        mark = basket.mark() + f'#{s.version}'
        cat_key = mem.category_key()
        if mark in self.cheaper_marks or not cat_key:
            return []
        self.cheaper_marks.add(mark)
        cart = basket.ids()
        s.reader.list_below(cart)
        picks: list[str] = []
        for line in cart:
            price = mem.price_of(line)
            kind = mem.label_of(line, cat_key).strip().lower()
            if price is None or not kind:
                continue
            pool = [i for i in mem.order if i not in cart and i not in mem.full and mem.buyable(i) and mem.label_of(i, cat_key).strip().lower() == kind and (mem.price_of(i) if mem.price_of(i) is not None else float('inf')) < price]
            pool.sort(key=lambda i: mem.price_of(i) or 0.0)
            picks += [i for i in pool[:2] if i not in picks]
        return s.reader.fill_facts(picks[:min(6, CHEAPER_OPENS * max(1, len(cart)))], s.order.tail() + 4)

    # -- pointer chains
    def trail(self) -> tuple[str, bool]:
        """Follow each pointer requirement's chain of pages, opening the links found; (text for the pilot, hold?)."""
        s = self.s
        mem, store, order, models = s.mem, s.store, s.order, s.models
        live = {need for _, need in order.needs_now()}
        pending = [need for _, need in order.refs if need in live and need not in order.ref_values]
        tool = store.offer_tool()
        target = store.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trails < (6 if store.left() > 15 else 4) and models.cost < SPEND_HARD:
            self.trails += 1
            records = []
            for ident in reversed(mem.order):
                record = mem.record(ident) if ident in mem.opened else None
                if record and len(records) < 45:
                    records.append('OPENED ' + to_json(clip(record, 300))[:1400])
            for ident in reversed(mem.order):
                if len(records) >= 60:
                    break
                if ident not in mem.opened or not mem.record(ident):
                    records.append('ROW ' + to_json(clip(mem.rows.get(ident), 80))[:300])
            prompt = P_TRAIL + '\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n' + '\n'.join(pending) + '\n\nRECORDS (newest first):\n' + '\n'.join(records)
            got = models.ask(prompt, [LIGHT] + ([HEAVY] if models.cost + HEAVY_ROOM <= SPEND_HARD else []), 4500) or {}
            s.log('trail', pending=pending, got=clip(got, 150))
            reach: list[str] = []
            lost = []
            gaps: list[tuple[str, str]] = []
            for chain in as_list(got.get('chains')):
                if not isinstance(chain, dict):
                    continue
                need = next((p for p in pending if p == str(chain.get('need') or '').strip()), None) or (pending[0] if len(pending) == 1 else None)
                if not need:
                    continue
                links = [l for l in as_list(chain.get('links')) if isinstance(l, dict)]
                ids = [str(l.get('id') or '').strip() for l in links]
                every_open = bool(links) and all(i in mem.opened for i in ids)
                if chain.get('done') is True and every_open and str(chain.get('value') or '').strip():
                    order.ref_values[need] = str(chain['value']).strip()
                    s.version += 1
                    continue
                if every_open and not str(links[-1].get('find') or '').strip():
                    order.ref_values[need] = str(chain.get('value') or '').strip() or f'as stated on {ids[-1]}'
                    s.version += 1
                    continue
                closed = [i for i in ids if i in mem.rows and i not in mem.opened]
                if closed:
                    reach += [i for i in closed if i not in reach]
                    continue
                gap = next((l for l, i in zip(links, ids) if i not in mem.rows), {})
                about = str(gap.get('about') or '') or 'the listing it points at'
                words = str(gap.get('find') or '').strip()
                lost.append(f'{need} -> find {about}' + (f' (search: {words})' if words else ''))
                gaps.append((about, words))
            pending = [p for p in pending if p not in order.ref_values]
            if not reach and gaps and tool and target and store.left() > 12 and self.link_searches < 3:
                reach = self.find_link(*gaps[0])
                if s.trace and s.trace[-1].get('kind') == 'link_known':
                    seen.append(f"{s.trace[-1]['id']}: an opened page of the shop this chain names that shows the values its description gives; read the value from it")
                    lost = lost[1:]
            if not reach or not tool or not target or store.left() < len(reach) + 4 or store.finished:
                break
            for ident in reach[:max(0, min(3, 6 - len(seen)))]:
                faults: list[str] = []
                args = conform({target: ident}, store.schemas.get(tool), 'args', faults)
                if faults:
                    break
                reply = store.call(tool, args)
                if reply is None or store.finished:
                    break
                s.reader.observe(tool, reply.get('observation'))
                seen.append(f'{ident}: ' + s.reader.show(reply.get('observation'))[:1500])
        text = ''
        if seen:
            text += '[Arc opened the pages a requirement points at]\n' + '\n'.join(seen) + '\n'
        if order.ref_values:
            text += '[values read from the pages requirements point at] ' + '; '.join(f'{k} -> {v}' for k, v in order.ref_values.items()) + '\n'
        hold = bool(lost) and self.trail_blocks < (3 if store.left() > 15 else 2) and store.left() > 5
        if hold:
            self.trail_blocks += 1
            text += 'The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: ' + ' | '.join(lost[:3]) + ". Search for it by its shop's name together with the values its description gives (the shop's other listings of that kind are not it), open the row of that shop whose page shows those values, apply the value its page gives, then check out."
        return text, hold

    def resolve_early(self) -> str:
        """U1 (b14): follow every pointer requirement's chain once, early (after the interview), instead of only when
        the pilot tries to check out. Halo's resolver did it as a checkout hold, after the pilot had spent its steps
        chasing the chain by hand (r14/r11-noask: 24 of 27 and 26 of 29 pointer tasks failed, most ending unbought).
        The early pass gets its own rounds: the checkout check keeps all of its own."""
        s = self.s
        order, store = s.order, s.store
        self.early_done = True
        live = {need for _, need in order.needs_now()}
        if not any(need in live and need not in order.ref_values for _, need in order.refs):
            return ''
        if store.left() <= 15 or s.models.cost >= SPEND_HARD or store.finished:
            return ''
        blocks, before = self.trail_blocks, dict(order.ref_values)
        text, _ = self.trail()
        self.trail_blocks = blocks
        self.trails = 0
        self.link_searches = 0
        s.log('early_trail', found={k: v for k, v in order.ref_values.items() if k not in before}, steps=store.step)
        return text.strip()

    def shop_rows(self, about: str, ids: list[str]) -> list[str]:
        """Unopened listings whose shop or brand the description names."""
        mem = self.s.mem
        low = about.lower()
        return [i for i in ids if i not in mem.opened and any(word_in(n.lower(), low) for n in shop_names(mem.rows.get(i) or {}))]

    def opened_shop(self, about: str) -> list[str]:
        mem = self.s.mem
        low = about.lower()
        return [i for i in mem.opened if any(word_in(n.lower(), low) for n in shop_names(mem.rows.get(i) or {}))]

    def find_link(self, about: str, words: str) -> list[str]:
        """Find a chain link no record matches: an opened page of that shop carrying every value given, else a search
        by the shop's name and the values (at most 3, remembered), keeping that shop's rows of that kind."""
        s = self.s
        mem = s.mem
        known: list[str] = []
        named = {w for ident in self.shop_rows(about, list(mem.rows)) + self.opened_shop(about) for k, v in (mem.rows.get(ident) or {}).items() if isinstance(v, str) and len(v.strip()) <= 40 and re.search(SHOP_FIELD, str(k), re.I) for w in re.findall('[a-z0-9]{3,}', v.lower())}
        store_values, store_labels = mem.store_words()
        values = {w for w in re.findall('[a-z0-9]+', (words or about).lower()) if w.isdigit() or (len(w) >= 3 and w in store_values)} - named
        for ident in self.opened_shop(about):
            text = ' '.join(leaves(mem.record(ident) or {})).lower()
            if values and all(word_in(v, text) for v in values):
                s.log('link_known', id=ident)
                return []
        plan = s.reader.search_template()
        memo = ' '.join(words.lower().split())
        if memo and memo in self.link_seen:
            self.link_searches += 1
            known = self.shop_rows(about, self.link_seen[memo])
        elif plan and words:
            tool, base, field = plan
            faults: list[str] = []
            args = conform({**base, field: words}, s.store.schemas.get(tool), 'args', faults)
            if not faults:
                self.link_searches += 1
                reply = s.store.call(tool, args)
                if reply is not None:
                    observation = reply.get('observation')
                    s.reader.observe(tool, observation)
                    self.link_seen[memo] = [i for i, _ in mem.records_in(observation)]
                    known = self.shop_rows(about, self.link_seen[memo])
                    s.log('link_search', words=words[:80], rows=known[:4])
        kind = {w for w in re.findall('[a-z0-9]{3,}', (words or about).lower()) if w in store_labels} - named

        def of_kind(ids: list[str]) -> list[str]:
            return [i for i in ids if not kind or kind & set(re.findall('[a-z0-9]{3,}', to_json(mem.rows.get(i) or {}).lower()))]
        return (of_kind(known) or of_kind(self.shop_rows(about, list(reversed(mem.order)))))[:2]

    # -- parts that must fit a device
    def fit_faults(self) -> list[str]:
        s = self.s
        mem, order = s.mem, s.order
        if not order.bound:
            return []
        index = mem.fit_index()
        cart = s.basket.ids()
        faults = []
        for part, main in order.bound:
            devices = [i for i in cart if i in order.item_ids(main)]
            parts = [i for i in cart if i in order.item_ids(part)]
            if len(devices) != 1 or not parts:
                continue
            codes = set(index['marks'].get(devices[0], {}).get('codes') or [])
            if not codes:
                continue
            for line in parts:
                fits = mem.fit_set(line, index)
                if fits and not fits & codes:
                    good = [i for i in order.item_ids(part) if mem.fit_set(i, index) & codes and not mem.sold_out(i) and i not in cart][:4]
                    faults.append(f"{line} ({order.items[part - 1].name}) works with {', '.join(sorted(fits)[:4])}, not with {devices[0]} ({', '.join(sorted(codes))})" + (f"; seen parts of that kind that work with it: {', '.join(good)}" if good else '; no seen part of that kind works with it, so look for one or switch the device: ' + self.fit_plan(short=True)))
        return faults

    def fit_plan(self, short: bool = False) -> str:
        s = self.s
        mem, order = s.mem, s.order
        if not order.bound:
            return ''
        index = mem.fit_index()
        lines = []
        for main in sorted({m for _, m in order.bound}):
            parts = [p for p, m in order.bound if m == main]
            tally: dict[str, dict[int, int]] = {}
            for p in parts:
                for i in order.item_ids(p):
                    if mem.sold_out(i):
                        continue
                    for code in mem.fit_set(i, index):
                        tally.setdefault(code, {}).setdefault(p, 0)
                        tally[code][p] += 1
            ranked = sorted(tally.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1].values())))[:4]
            devices = {code: [i for i in order.item_ids(main) if code in (index['marks'].get(i, {}).get('codes') or [])][:2] for code, _ in ranked}
            text = '; '.join(f'{code}: ' + ', '.join(f'{n} {order.items[p - 1].name}' for p, n in sorted(per.items())) + (f" (device seen: {', '.join(devices[code])})" if devices[code] else ' (device not seen yet)') for code, per in ranked)
            if text:
                lines.append(f"codes the {order.items[main - 1].name} candidates' parts name -> {text}")
        if not lines:
            return ''
        return ' | '.join(lines) if short else "[fit plan: a device fits a part when the part's text names the device's model code or another name it is sold as; codes are not searchable, so list the device's category and read the codes on its rows] " + ' | '.join(lines)

    # -- sold-out lines
    def swap_hint(self) -> str:
        s = self.s
        mem, store = s.mem, s.store
        lost = [i for i in s.basket.ids() if i not in mem.gone and mem.sold_out(i)]
        if not lost:
            return ''
        for ident in lost:
            mem.gone[ident] = store.step
        self.leads = [i for i in self.leads if i not in lost]
        s.version += 1
        picks = self.replacements(lost)
        text = f"[Arc] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        if picks and store.left() <= 6:
            return text + f" Few steps are left: in this turn add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks[1:]) or 'none'}."
        if picks and s.order.pick() == 'cheapest':
            return text + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(picks)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {picks[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        if picks:
            return text + f" Listings already read that appear to fit that item instead: {', '.join(picks)}. In this turn take the sold-out line out of the cart and add {picks[0]} at the requested quantity, then check out."
        return text + ' Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits.'

    def find_sold(self) -> str:
        """After a sold-out note that names no line: one line must be it; with several, read their offers again."""
        s = self.s
        mem = s.mem
        ids = [i for i in s.basket.ids() if i not in mem.gone]
        if any(mem.sold_out(i) for i in ids):
            return ''
        if len(ids) == 1:
            mem.unavailable.add(ids[0])
            return ''
        return s.reader.reread_offers() if ids else ''

    def replacements(self, lost: list[str]) -> list[str]:
        s = self.s
        mem, order, models = s.mem, s.order, s.models
        if not models.judges():
            return []
        records = []
        for ident in reversed(mem.order):
            if ident in lost or ident in mem.gone or mem.sold_out(ident):
                continue
            records.append(to_json(clip(mem.record(ident), 200))[:1200])
            if len(records) >= 25:
                break
        if not records:
            return []
        prompt = P_SWAP + '\n\nREQUEST:\n' + order.request + '\n\nREQUIREMENTS IN FORCE:\n' + order.ledger()
        prompt += '\n\nSOLD-OUT CART LINE:\n' + '\n'.join(f'{i}: {mem.record_text(i)}' for i in lost) + '\n\nCART NOW:\n' + s.basket.text()
        prompt += '\n\nOTHER LISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
        got = models.ask(prompt, models.judges()[::-1], 3500)
        fits: list[tuple[float, str]] = []
        cart = s.basket.ids()
        for entry in as_list((got or {}).get('fits')):
            ident = str(entry.get('id') if isinstance(entry, dict) else entry)
            price = entry.get('price') if isinstance(entry, dict) else None
            if ident in mem.rows and ident not in lost and ident not in mem.gone and ident not in cart and ident not in [i for _, i in fits]:
                fits.append((float(price) if is_number(price) else float('inf'), ident))
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if order.pick() == 'cheapest' else [i for _, i in fits]
        s.log('swap', lost=lost, fits=fits)
        return ranked[:4]

    # -- before ending unbought
    def audit(self) -> str:
        """A review of the records seen before ending unbought; its leads when it thinks something fits."""
        s = self.s
        if self.audited or not s.models.judges():
            return ''
        self.audited = True
        got = self.audit_call()
        named = self.audit_fits(got)
        if not named:
            rows = [str(i) for i in as_list((got or {}).get('open')) if str(i) in s.mem.rows and str(i) not in s.mem.opened and not s.mem.sold_out(str(i))]
            if rows and s.reader.open_rows(rows):
                got = self.audit_call()
                named = self.audit_fits(got)
        if not named:
            return ''
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return 'A careful review thinks these listings may fit: ' + '; '.join(f"item {k}: {', '.join(v)}" for k, v in named.items()) + (f" ({got.get('reason')})" if got.get('reason') else '') + '. Open them and buy what truly fits; end without a purchase only if each one fails.'

    def audit_call(self) -> dict[str, Any] | None:
        s = self.s
        mem, models = s.mem, s.models
        records = []
        for ident in reversed(mem.order):
            if mem.sold_out(ident):
                continue
            record = mem.full.get(ident) if ident in mem.full else None
            records.append('OPENED ' + to_json(clip(record, 200))[:1200] if record else 'ROW ' + to_json(clip(mem.rows.get(ident), 120))[:500])
            if len(records) >= 30:
                break
        if not records:
            return None
        prompt = P_AUDIT + '\n\nREQUEST:\n' + s.order.request + '\n\nREQUIREMENTS IN FORCE:\n' + s.order.ledger() + '\n\nLISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
        judges = models.judges()
        got = models.ask(prompt, [LIGHT] if judges else [], 4000)
        if HEAVY in judges and not models.no_credit and not self.audit_fits(got):
            got = models.ask(prompt, [HEAVY], 4000) or got
        s.log('audit', got=got)
        return got

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        mem = self.s.mem
        fits = (got or {}).get('fits')
        if not isinstance(fits, dict):
            return {}
        named = {str(k): [str(i) for i in as_list(v) if str(i) in mem.rows and not mem.sold_out(str(i))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def complete_cart(self) -> str:
        """When a lead was confirmed by an independent check and the pilot still ends unbought: set the cart to the
        leads (one per item, other lines kept), check it, and check out in code."""
        s = self.s
        store, mem, order, basket = s.store, s.mem, s.order, s.basket
        cart, buy = store.roles.get('cart'), store.roles.get('buy')
        if self.completed or not self.confirmed or mem.sold_out(self.confirmed) or not cart or not buy:
            return ''
        self.completed = True
        cat_key = mem.category_key()

        def kind(ident: str) -> str:
            return mem.label_of(ident, cat_key).strip().lower() if cat_key else ident
        plan: dict[str, int] = {}
        for number, ids in self.lead_items.items():
            found = re.search('\\d+', str(number))
            index = int(found.group()) if found else (1 if len(order.items) == 1 else None)
            pick = next((i for i in ids if not mem.sold_out(i)), None)
            if pick and index and 1 <= index <= len(order.items) and kind(pick) not in {kind(p) for p in plan}:
                plan[pick] = order.items[index - 1].qty
        if self.confirmed not in plan:
            return ''
        now = basket.ids()
        keep = {i: basket.lines.get(i, 1) for i in now if i not in plan and not mem.sold_out(i) and kind(i) not in {kind(p) for p in plan}}
        drop = [i for i in now if i not in plan and i not in keep]
        if len(plan) + len({kind(i) for i in keep}) < len(order.items):
            return ''
        lines = {**keep, **plan}
        calls = basket.plan_calls(lines, drop)
        if not calls or len(calls) + 1 > store.left():
            return ''
        s.reader.fill_facts(list(lines), len(calls) + 3)
        s.reader.show_links(list(lines), len(calls) + 3)
        self.check_cart(list(lines), SPEND_HARD)
        if not self.verdicts.get(self.check_key(list(lines))):
            return ''
        done = basket.apply(calls)
        if done is None:
            return ''
        faults: list[str] = []
        args = conform(dict(store.buy_args), store.schemas.get(buy), 'args', faults)
        if faults:
            return ''
        replies = []
        for attempt in range(2):
            if attempt:
                s.reader.reread_offers()
                self.check_cart(list(lines), SPEND_HARD)
                if not self.verdicts.get(self.check_key(list(lines))) or store.left() < 1 or store.finished or any(mem.sold_out(i) for i in lines):
                    break
            go, said = self.gate('order')
            if said:
                replies.append(said)
            if not go:
                break
            reply = store.call(buy, args)
            s.log('completed', lines=lines, drop=drop, reply=clip(reply, 200))
            if reply is None:
                break
            self.bought(reply)
            s.reader.observe(buy, reply.get('observation'))
            replies.append(s.reader.show(reply.get('observation')))
            if store.finished or reply.get('finished'):
                break
        if not replies:
            return ''
        return 'Not run (Arc acted instead): the independent check confirmed ' + self.confirmed + ', so Arc set the cart (' + '; '.join(done) + ') and checked out. Store reply: ' + ' | then: '.join(replies)


# -------------------------------------------------------------------------------------------------------- Pilot

class Pilot(Part):
    """The light model's loop over the store tools. Each call it makes passes the guard layer: argument checks,
    repeats, steps kept for the ending, and the checks before an ending; then its result is shown with Arc's notes."""

    def __init__(self, session: 'Session'):
        super().__init__(session)
        self.chat: list[dict[str, Any]] = []
        self.turns = 0
        self.misses = 0
        self.recent: list[str] = []
        self.cut_to = 2
        self.cut_size = 0

    # -- status shown after each turn
    def status(self) -> str:
        s = self.s
        store, order, ref = s.store, s.order, s.referee
        left = store.left()
        tail = order.tail()
        text = f'[Arc] step {store.step} of {store.max_steps} used, {left} left.'
        if left <= tail + 1:
            text += ' FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that).'
        elif left <= 6 or (left <= len(order.items) + 5 and ref.buys == 0):
            text += ' Few steps left: set the cart to the chosen lines and check out now.'
        elif ref.buys == 0 and left <= 12:
            text += ' Steps are running low: settle on your picks, fill the cart and check out within the next few steps.'
        elif ref.buys == 0 and not s.basket.touched and store.step >= store.max_steps // 2:
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        if s.models.clock() >= WIND_DOWN and left > tail + 1:
            text += ' TIME IS NEARLY UP: set the cart to the chosen lines and check out within the next two turns; an episode that runs out of time earns nothing.'
        live = {need for _, need in order.needs_now()}
        lookups = [need for _, need in order.refs if need in live and need not in order.ref_values]
        if lookups and ref.buys == 0:
            text += '\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] ' + ' | '.join(lookups[:3])
        if ref.leads:
            text += '\n[review candidates] ' + ', '.join(ref.leads)
        plan = ref.fit_plan() if order.bound and ref.buys == 0 else ''
        if plan:
            text += '\n' + plan
        if order.notes:
            text += '\n[changes so far] ' + ' | '.join(order.notes[-4:])
        return text + '\n[ledger]\n' + order.ledger()

    def offered(self) -> list[dict[str, Any]]:
        """Tools the pilot may call now: never the message tool, and only the ending (and cart) tools near the end."""
        s = self.s
        store = s.store
        left, tail = store.left(), s.order.tail()
        ends = store.enders()
        keep: set[str] | None = None
        if left <= tail:
            keep = set(ends)
        elif left <= tail + 3:
            keep = set(ends) | {store.roles.get('cart'), store.roles.get('cart_view')}
        elif left <= len(s.order.items) + 5 and s.referee.buys == 0 and not s.basket.missing_items():
            # b9: only once every item has a cart line; with an item missing the order fails anyway, so its search
            # stays open (race 14: the pilot wanted to search for the missing item, had no search tool and idled)
            keep = set(ends) | {store.roles.get('cart'), store.roles.get('cart_view'), store.roles.get('offer')}
        tools = [t for t in store.tools if t['function']['name'] != store.roles.get('ask')]
        if keep is not None and ends:
            tools = [t for t in tools if t['function']['name'] in keep] or tools
        return tools

    # -- the loop
    def run(self) -> None:
        s = self.s
        store, order = s.store, s.order
        opening = order.view.get('opening')
        if opening is not None:
            s.mem.absorb(opening)
        intro = "SHOPPER'S REQUEST:\n" + order.request
        if opening is not None:
            intro += '\n\nWHAT THE STORE SHOWS FIRST:\n' + s.reader.show(opening)
        intro += '\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n' + self.status()
        self.chat = [{'role': 'system', 'content': P_PILOT}, {'role': 'user', 'content': intro}]
        self.work(PILOT_TURNS, SPEND_HARD)
        if store.finished:
            return
        turned = self.finish(again=True)
        if turned and not store.finished and not store.broke and not s.models.no_credit and store.left() > 1:
            self.chat.append({'role': 'user', 'content': '[Arc] Arc tried to end the episode and ' + turned + '\nApply what the store says, then finish.\n' + self.status()})
            self.work(self.turns + 10, SPEND_HARD + 0.01)
        if not store.finished:
            self.finish()

    def may_turn(self, turn_cap: int, spend_cap: float) -> bool:
        s = self.s
        store, models = s.store, s.models
        if store.finished or store.broke or store.left() <= 0 or self.turns >= turn_cap or models.cost >= spend_cap:
            return False
        return models.clock() < LAST_TURN or (not s.basket.ids() and models.room() > EMPTY_ROOM)

    def work(self, turn_cap: int, spend_cap: float) -> None:
        s = self.s
        idle = 0
        while self.may_turn(turn_cap, spend_cap):
            if not s.referee.early_done and s.order.refs and s.store.step >= 2 and (s.talk.asked or not s.store.roles.get('ask') or s.order.msgs_free() <= 0):
                # b14: pointer chains are followed in code once, right after the interview
                text = s.referee.resolve_early()
                if text:
                    note = text + '\n' + self.status()
                    if self.chat and self.chat[-1].get('role') == 'user' and isinstance(self.chat[-1].get('content'), str):
                        self.chat[-1]['content'] = self.chat[-1]['content'].rsplit('\n[Arc] step ', 1)[0] + '\n' + note
                    else:
                        self.chat.append({'role': 'user', 'content': note})
                continue
            if s.talk.ready():
                summary = s.talk.run()
                if summary:
                    self.chat.append({'role': 'user', 'content': summary + '\n' + self.status()})
                continue
            self.trim()
            self.turns += 1
            message = s.models.chat(self.chat, [LIGHT, SPARE], PILOT_TOKENS, self.offered())
            if message is None:
                self.misses += 1
                if s.models.no_credit or s.models.calls >= MODEL_CALLS or self.misses > 2:
                    break
                continue
            calls = self.calls_of(message)
            content = message.get('content')
            if not isinstance(content, str):
                content = to_json(content) if content else ''
            if not calls:
                idle += 1
                self.chat.append({'role': 'assistant', 'content': content or '(no call)'})
                # b9: name the tools open now (a reply without a call often means the wanted tool is closed)
                names = ', '.join(t['function']['name'] for t in self.offered())
                self.chat.append({'role': 'user', 'content': f'Call one of the tools to continue (open now: {names}).\n' + self.status()})
                if idle >= 3:
                    break
                continue
            idle = 0
            if s.talk.wants_first(calls):
                self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
                for call in calls:
                    self.chat.append({'role': 'tool', 'tool_call_id': call['id'], 'content': 'Not run (no step used): the shopper is asked first.'})
                summary = s.talk.run()
                self.chat.append({'role': 'user', 'content': (summary or 'The interviewer had nothing to ask.') + '\n' + self.status()})
                continue
            self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
            s.log('turn', text=content[:300], calls=[c['function'] for c in calls])
            replies = self.run_calls(calls)
            for index, (cid, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += '\n' + self.status()
                self.chat.append({'role': 'tool', 'tool_call_id': cid, 'content': text})

    def calls_of(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        calls = []
        for number, call in enumerate(message.get('tool_calls') or []):
            if not isinstance(call, dict):
                continue
            fn = call.get('function') if isinstance(call.get('function'), dict) else {}
            raw = fn.get('arguments')
            parsed = raw if isinstance(raw, dict) else parse_object(raw) if isinstance(raw, str) and raw.strip() else {}
            calls.append({'id': str(call.get('id') or f'k{self.turns}_{number}'), 'type': 'function', 'function': {'name': str(fn.get('name') or ''), 'arguments': to_json(parsed if isinstance(parsed, dict) else {})}})
        return calls

    def trim(self) -> None:
        """b4 (27-know b15): the prompt cache reads the history at a tenth of what writing it costs, and a change to an
        older message makes everything after it be written again. Halo shortened half of the older results on every
        call once the chat passed HISTORY_CHARS, so long episodes rewrote the whole history call after call. Here the
        cut is deep (every message older than the newest TRIM_KEEP turns becomes a digest) and the next one waits
        until the chat has grown TRIM_GAP characters more; a cut message is never changed again."""
        size = sum(len(to_json(m.get('content'))) for m in self.chat)
        if size <= max(HISTORY_CHARS, self.cut_size + TRIM_GAP):
            return
        turns = [i for i, m in enumerate(self.chat) if m.get('role') == 'assistant']
        if len(turns) <= TRIM_KEEP or turns[-TRIM_KEEP] <= self.cut_to:
            return
        edge = turns[-TRIM_KEEP]
        for index in range(max(2, self.cut_to), edge):
            message = self.chat[index]
            content = message.get('content')
            if not isinstance(content, str) or not content:
                continue
            if message.get('role') == 'tool':
                message['content'] = self.digest(content)
            elif message.get('role') == 'assistant':
                if len(content) > DIGEST_SAY:
                    message['content'] = content[:DIGEST_SAY] + '...'
            elif message.get('role') == 'user':
                message['content'] = self.drop_status(content)
        self.cut_to = edge
        self.cut_size = sum(len(to_json(m.get('content'))) for m in self.chat)
        self.s.log('cut', before=size, after=self.cut_size, upto=edge)

    @staticmethod
    def drop_status(content: str) -> str:
        """An older copy of the step and ledger block goes; the newest one rides on the latest message."""
        mark = content.rfind('\n[Arc] step ')
        return content[:mark] if mark >= 0 else content

    def digest(self, content: str) -> str:
        """An older result as Arc last saw its listings (id, price, stock, then the row's short fields) plus Arc's own
        notes on it; any page can be opened again."""
        body = self.drop_status(content)
        if len(body) <= DIGEST_KEEP:
            return body
        rows = self.s.mem.rows
        head, _, tail = body.partition('\n')
        notes = [line[:300] for line in tail.split('\n') if line.startswith('[')][:5]
        ids = [i for i in dict.fromkeys(re.findall('"([A-Za-z0-9][A-Za-z0-9_\\-:.]{5,40})"', body)) if i in rows][:DIGEST_ROWS]
        if ids and head and head[0] in '{[':
            lines = ['(older result, shortened by Arc; its listings as last seen:)'] + [self.row_digest(i) for i in ids]
        else:
            lines = [head[:DIGEST_KEEP] + ('...' if len(head) > DIGEST_KEEP else '')]
        return '\n'.join(lines + notes)

    def row_digest(self, ident: str) -> str:
        mem = self.s.mem
        row = mem.rows.get(ident) or {}
        lead = {k: row[k] for k in (mem.price_key, mem.stock_key) if k and k in row}
        rest = {k: v for k, v in row.items() if k not in lead and v != ident and not isinstance(v, (dict, list))}
        text = ident + ' ' + to_json(clip(dict(lead, **rest), DIGEST_FIELD))
        if ident in mem.gone:
            text += ' (sold out)'
        return text[:DIGEST_ROW]

    # -- the guard layer
    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Several single-listing opens in one turn become one side-by-side call when that view shows attributes."""
        s = self.s
        store, mem = s.store, s.mem
        opener, joiner = store.roles.get('open'), store.roles.get('compare')
        target, many = store.single_target(opener), store.compare_target()
        if not opener or not joiner or not target or not many or opener == joiner or mem.compare_bare():
            return calls
        picks: list[tuple[int, str]] = []
        for index, call in enumerate(calls):
            fn = call.get('function') or {}
            args = parse_object(fn.get('arguments')) if isinstance(fn.get('arguments'), str) else None
            ident = str((args or {}).get(target) or '').strip()
            if fn.get('name') == opener and args and set(args) == {target} and ident in mem.rows and ident not in mem.gone:
                picks.append((index, ident))
        picks = picks[:store.compare_room()]
        if len(picks) < 2:
            return calls
        first = picks[0][0]
        joined = {i for i, _ in picks}
        out = []
        for index, call in enumerate(calls):
            if index == first:
                out.append({'id': call['id'], 'type': 'function', 'function': {'name': joiner, 'arguments': to_json({many: [i for _, i in picks]})}})
            elif index in joined:
                out.append({'id': call['id'], 'joined': True})
            else:
                out.append(call)
        s.log('joined', ids=[i for _, i in picks])
        return out

    def resolve_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """A listing argument given as the start of a title is mapped to the one listing whose text starts so."""
        s = self.s
        store, mem = s.store, s.mem
        if name not in {store.roles.get(r) for r in ('cart', 'open', 'offer', 'compare')} - {None} or not mem.id_key:
            return args, ''
        props = store.props(name)
        free = [k for k, v in props.items() if isinstance(v, dict) and ('string' in type_list(v) or 'array' in type_list(v)) and 'enum' not in v]
        if len(free) != 1 or (name != store.roles.get('cart') and (name, free[0]) not in store.id_args):
            return args, ''

        def fix(value: Any) -> tuple[Any, str]:
            if isinstance(value, str) and value.strip() in mem.rows:
                return value.strip(), ''
            if not isinstance(value, str) or looks_like_id(value, True) or len(value.strip()) < 8:
                return value, ''
            low = value.strip().lower()
            hits = [i for i, row in mem.rows.items() if any(k != mem.id_key and isinstance(v, str) and v.strip().lower().startswith(low) for k, v in row.items())]
            if len(hits) == 1:
                return hits[0], ''
            if hits:
                return value, f"several listings carry the text {value!r} ({', '.join(hits[:6])}); pass one listing id"
            return value, f'{value!r} is not a listing id; pass the id from the result rows (their {mem.id_key} field)'
        out = dict(args)
        for key in free:
            if key not in args:
                continue
            fixed = [fix(v) for v in args[key]] if isinstance(args[key], list) else [fix(args[key])]
            problem = next((p for _, p in fixed if p), '')
            if problem:
                return args, problem
            out[key] = [v for v, _ in fixed] if isinstance(args[key], list) else fixed[0][0]
        return out, ''

    def quiet(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """An ending without a purchase says the short remark and names no listing."""
        s = self.s
        schema = s.store.schemas.get(name) or {}
        props = schema.get('properties') or {}
        out = dict(args)
        for key, spec in props.items():
            if not isinstance(spec, dict) or 'string' not in type_list(spec) or 'enum' in spec:
                continue
            value = out.get(key)
            if value is None and key not in (schema.get('required') or []):
                continue
            text = str(value or '')
            named = any(ident in text for ident in s.mem.rows) or s.order.brief.get('quiet_decline') is not False
            if named or not text:
                out[key] = QUIET_REMARK[:spec.get('maxLength', 200)]
        return out

    def checked_args(self, name: str, raw: Any, calls: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
        """The call's arguments after the argument guards, or the reason it is not run."""
        s = self.s
        store = s.store
        if isinstance(raw, dict):
            args = raw
        else:
            args = parse_object(raw) if (raw or '').strip() else {}
            if args is None:
                return None, 'Not run (no step used): the arguments were not one JSON object.'
        faults: list[str] = []
        args = conform(args, store.schemas[name], 'args', faults)
        if faults:
            return None, 'Not run (no step used): ' + '; '.join(faults[:6])
        args, wrong = self.resolve_ids(name, args)
        if wrong:
            return None, 'Not run (no step used): ' + wrong + '.'
        if name == store.roles.get('cart'):
            args = s.basket.whole_removal(name, args)
        return args, ''

    def buy_holds(self) -> str:
        """The checks before a checkout the pilot asked for; the text that holds it, or '' to let it run."""
        s = self.s
        store, ref, basket, reader = s.store, s.referee, s.basket, s.reader
        wider = ref.wider_hold()
        if wider:
            return wider + '\nCart now: ' + basket.text()
        reread = reader.reread_offers()
        if reader.note_sold:
            ref.find_sold()
        view_tool = store.roles.get('cart_view')
        if basket.touched and basket.seen_at < basket.changed_at and view_tool and store.left() > 3:
            faults: list[str] = []
            view_args = conform({}, store.schemas.get(view_tool), 'args', faults)
            view = store.call(view_tool, view_args) if not faults else None
            if view is not None:
                basket.seen(view.get('observation'), store.step)
                change = reader.observe(view_tool, basket.view)
                if change:
                    reader.queued = []
                    return 'Not run yet (no step used). Looking at the cart showed a change: ' + change + '\nCart now: ' + basket.text() + '\nApply the change, fix the cart if needed, then check out.'
        swap = ref.swap_hint()
        held = [i for i in basket.ids() if i in s.mem.gone] if ref.held < 2 else []
        if swap or held:
            ref.held += 0 if swap else 1
            return 'Not run yet (no step used). ' + (swap or f"{', '.join(held)} in the cart is sold out: replace that line first.") + '\nCart now: ' + basket.text()
        chain, hold = ref.trail()
        if hold:
            return 'Not run yet (no step used). ' + chain + '\nCart now: ' + basket.text()
        tail = s.order.tail()
        misfit = ref.fit_faults() if ref.fit_holds < 2 and store.left() > tail + 3 else []
        if misfit:
            ref.fit_holds += 1
            s.log('fit_hold', faults=misfit[:4])
            return 'Not run yet (no step used). A part in the cart does not work with the device in the cart: ' + '; '.join(misfit[:3]) + '. Swap the part (or the device), then check out.\nCart now: ' + basket.text()
        faults = basket.line_faults() if basket.line_holds < 2 and store.left() > tail + 2 else []
        if faults and all(' is held at ' in f for f in faults):
            fixed = basket.fix_counts(basket.count_fixes())
            faults = basket.line_faults() if fixed else faults
            if fixed and not faults:
                basket.line_holds += 1
                return "Not run yet (no step used). Arc set the cart's quantities itself: " + '; '.join(fixed) + '. If the store refused any of these, fix those lines; then check out.\nCart now: ' + basket.text()
        if faults:
            basket.line_holds += 1
            s.log('line_hold', faults=faults[:6])
            return 'Not run yet (no step used). The cart does not match the request yet: ' + '; '.join(faults[:6]) + ". Fix those lines (the cart tool's own way to add or take off units), then check out.\nCart now: " + basket.text()
        lines = basket.ids()
        read = reader.fill_facts(lines, s.order.tail() + (1 if len(lines) >= 2 else 3))
        reader.show_links(lines, s.order.tail() + 1)
        read += ref.cheaper_rows()
        verdict = ref.check_cart()
        if verdict:
            seen = "Arc read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n' if read else ''
            return 'Not run yet. ' + chain + seen + verdict + '\nCart now: ' + basket.text()
        if reread and s.models.cost > SPEND_SOFT:
            return "Not run yet (no step used). Arc re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'
        return ''

    def refuse_holds(self) -> str:
        """The audit before ending unbought, while more than five steps are left."""
        s = self.s
        ref, store = s.referee, s.store
        review = ref.audit()
        if review:
            return 'Not run yet (no step used). ' + review
        if ref.leads and store.left() > 4 and ref.refuse_blocks == 0:
            ref.refuse_blocks = 1
            s.log('refuse_blocked', leads=list(ref.leads))
            return 'Not run yet (no step used). The review found listings that appear to fit: ' + ', '.join(ref.leads) + '. Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again.'
        if ref.leads and store.left() > 4 and ref.refuse_blocks == 1:
            ref.refuse_blocks = 2
            first = ref.leads[0]
            ref.check_cart([first], SPEND_HARD)
            if ref.verdicts.get(ref.check_key([first])) and first in ref.leads:
                ref.confirmed = first
                s.log('refuse_blocked', confirmed=first)
                return f"Not run yet (no step used). An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."
        return ''

    def late_refuse_hold(self) -> str:
        """Near the end, an ending unbought is held once when the fullest cart built so far covers every item."""
        s = self.s
        ref, basket, store = s.referee, s.basket, s.store
        lines = [i for i in basket.built if not s.mem.sold_out(i)]
        missing = [i for i in lines if i not in basket.ids() or (i in basket.dropped and not basket.fresh())]
        if len(lines) < len(s.order.items) or len(missing) + 1 > store.left():
            return ''
        ref.late_held = True
        s.log('late_decline_held', lines=lines, missing=missing)
        redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else 'Check out the cart as it is.'
        return 'Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. ' + redo + ' End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again.'

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        s = self.s
        store, mem, ref, reader, basket = s.store, s.mem, s.referee, s.reader, s.basket
        out: list[tuple[str, str]] = []
        calls = self.join_opens(calls)
        for call in calls:
            allowed = {t['function']['name'] for t in self.offered()}
            cid = call['id']
            if call.get('joined'):
                out.append((cid, 'Shown in the side-by-side result above: Arc opened these listings together in one step.'))
                continue
            fn = call.get('function') or {}
            name = fn.get('name')
            if store.finished:
                out.append((cid, 'Not run: the episode is over.'))
                continue
            if name == store.roles.get('ask'):
                out.append((cid, "Not run (no step used): Arc's interviewer handles every message to the shopper."))
                continue
            if name not in store.schemas:
                out.append((cid, f"Not run (no step used): unknown tool. Tools: {', '.join(sorted(allowed))}."))
                continue
            if name not in allowed:
                out.append((cid, 'Not run (no step used): too few steps are left for that; use: ' + ', '.join(sorted(allowed)) + '.'))
                continue
            args, refusal = self.checked_args(name, fn.get('arguments'), calls)
            if args is None:
                out.append((cid, refusal))
                continue
            trimmed = ''
            many = store.compare_target() if name == store.roles.get('compare') else None
            if many and isinstance(args.get(many), list) and len(args[many]) > store.compare_room() and store.compare_bad:
                room = store.compare_room()
                trimmed = f"\n[Arc] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
                args = {**args, many: args[many][:room]}
            target = store.single_target(name)
            if target and name in (store.roles.get('open'), store.roles.get('offer')) and str(args.get(target)) in mem.gone:
                out.append((cid, f'Not run (no step used): {args[target]} sold out at step {mem.gone[str(args[target])]} and will not come back. Replace its cart line with another fitting listing.'))
                continue
            if name == store.roles.get('cart') and any((as_int(v) or 0) < 0 for v in args.values() if not isinstance(v, str)):
                view = to_json(basket.view) if basket.seen_at >= basket.changed_at else ''
                absent = [v for v in args.values() if isinstance(v, str) and v in mem.rows and v not in basket.lines and v not in view and v not in basket.said]
                if absent:
                    out.append((cid, f'Not run (no step used): {absent[0]} is not in the cart, so there is nothing to take off. Cart now: ' + basket.text()))
                    continue
            buying, refusing = store.is_buy(name, args), store.is_refuse(name, args)
            ending = buying or refusing
            if ending and len(calls) > 1:
                out.append((cid, 'Not run (no step used): send the finishing call on its own, after reading your other results.'))
                continue
            if not ending and store.left() <= s.order.tail() and (store.roles.get('buy') or store.roles.get('refuse')):
                out.append((cid, "Not run (no step used): the last steps are kept for checking out (with the shopper's approval of the basket when it is due) or ending without a purchase."))
                continue
            mark = name + to_json(args)
            if not ending and name != store.roles.get('cart') and mark in self.recent:
                out.append((cid, 'Not run (no step used): you just made this same call and nothing changed; its result is above.'))
                continue
            readers = store.readers()
            cached = reader.read_cache.get(mark)
            if name in readers and cached and cached[0] == s.version:
                out.append((cid, f'Not run (no step used): the same call at step {cached[1]} returned this, and nothing has changed since:\n' + cached[2][:3000]))
                continue
            if target and name in readers and name != store.roles.get('compare') and mem.shown_already(name, str(args.get(target))):
                out.append((cid, f'Not run (no step used): {args[target]} has shown everything this view gives since the last change. Its record: ' + mem.record_text(str(args[target]))))
                continue
            if many and isinstance(args.get(many), list) and args[many]:
                keep = [i for i in args[many] if not mem.shown_already(name, str(i))]
                if not keep:
                    out.append((cid, 'Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n' + '\n'.join(f'{i}: {mem.record_text(str(i))}' for i in args[many])[:4000]))
                    continue
                if len(keep) < len(args[many]):
                    trimmed += f"\n[Arc] Left out {', '.join(str(i) for i in args[many] if i not in keep)}: shown in full already, nothing changed since."
                    args = {**args, many: keep}
                    mark = name + to_json(args)
            if ending and store.left() > 2:
                hold = self.buy_holds() if buying else (self.refuse_holds() if store.left() > 5 else '')
                if hold:
                    out.append((cid, hold))
                    continue
            if refusing and store.left() <= 5 and not ref.late_held:
                hold = self.late_refuse_hold()
                if hold:
                    out.append((cid, hold))
                    continue
            if refusing and not ref.leads and not ref.pushed and store.left() > store.max_steps // 3:
                ref.pushed = True
                out.append((cid, f"Not run yet (no step used). {store.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if refusing:
                done = ref.complete_cart()
                if done:
                    out.append((cid, done + (' [episode finished]' if store.finished else '')))
                    continue
                args = self.quiet(name, args)
            if buying:
                late = s.talk.late()
                if late:
                    out.append((cid, 'Not run yet. ' + late + '\nCart now: ' + basket.text()))
                    continue
            if ending and store.left() >= 2:
                go, said = ref.gate('order' if buying else 'end')
                if not go:
                    out.append((cid, 'Not run yet. ' + said + '\nCart now: ' + basket.text()))
                    continue
                if said:
                    trimmed = '\n[Arc] ' + said + trimmed
            reply = store.call(name, args)
            if reply is None:
                out.append((cid, 'The call failed (network or server). You may try again.'))
                continue
            out.append((cid, self.result_text(name, args, reply, mark, many, buying, ending, trimmed)))
        return out

    def result_text(self, name: str, args: dict[str, Any], reply: dict[str, Any], mark: str, many: str | None, buying: bool, ending: bool, trimmed: str) -> str:
        """Take a store result in (memory, cart, views, notes) and write what the pilot sees of it."""
        s = self.s
        store, mem, ref, reader, basket = s.store, s.mem, s.referee, s.reader, s.basket
        observation = reply.get('observation')
        if many and isinstance(args.get(many), list) and len(args[many]) > 1:
            if not any(str(i) in to_json(observation) for i in args[many]):
                store.compare_bad = min(store.compare_bad or 999, len(args[many]))
                trimmed += f'\n[Arc] The side-by-side view showed nothing for {len(args[many])} listings: send at most {store.compare_room()} per call.'
            else:
                store.compare_ok = max(store.compare_ok, len(args[many]))
        self.recent = (self.recent + [mark])[-3:]
        store.id_args.update((name, k) for k, v in args.items() if (v if isinstance(v, list) else [v]) and all(isinstance(x, str) and x in mem.rows for x in (v if isinstance(v, list) else [v])))
        mem.absorb(observation)
        readers = store.readers()
        if name == store.roles.get('compare'):
            rows = mem.records_in(observation)
            if rows:
                mem.facts_from[name] = mem.facts_from.get(name, False) or any(attributes(r) for _, r in rows)
        if name in readers - {store.roles.get('search')}:
            mem.note_view(name, observation)
        plain_jobs = {tool for job, tool in store.roles.items() if job != 'search'}
        paged = reader.more_pages(name, args, observation) if name not in plain_jobs and not ending else ''
        self.learn_views(name, observation)
        if name == store.roles.get('cart'):
            basket.after_call(name, args, observation)
            self.recent = []
        if name == store.roles.get('search') and mem.records_in(observation):
            reader.searched = (reader.searched + [dict(args)])[-8:]
        if name == store.roles.get('cart_view'):
            basket.seen(observation, store.step)
        if buying:
            ref.bought(reply)
        text = reader.show(observation) + paged + trimmed
        if name not in plain_jobs and not ending:
            text += reader.cheapest_here([observation] + reader.paged_obs) + reader.cap_ignored(name, args, observation)
        base = text
        proof = reader.evidence(observation)
        if proof:
            text += '\n' + proof
        if name == store.roles.get('cart') and any(as_int(v) == 0 for v in args.values() if not isinstance(v, str)):
            if any(isinstance(v, str) and v in mem.rows and mentions(observation, v) for v in args.values()):
                text += '\n[Arc] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'
        reader.note_sold = False
        lines = reader.loose_lines(observation)
        if lines:
            meaning = reader.take_note(lines)
            if meaning:
                text += '\n[note read] ' + meaning + '. Apply this change to your picks before checking out.'
        refused = buying and not reply.get('finished')
        again = refused and any(line in reader.spoken(observation) for line in reader.sold_lines)
        loose = reader.note_sold and (refused or (not reader.note_named and not buying))
        if (loose or again) and basket.ids() and not store.finished:
            seen = ref.find_sold()
            if seen:
                text += "\n[Arc re-read the cart's lines after the refusal]\n" + seen
        reader.note_sold = False
        swap = ref.swap_hint()
        if swap:
            text += '\n' + swap
        if reader.queued:
            text += "\n[notes seen on Arc's own calls] " + ' | '.join(reader.queued)
            reader.queued = []
        if name in readers:
            reader.read_cache[mark] = (s.version, store.step, base)
        if reply.get('finished'):
            text += ' [episode finished]'
        return text

    def learn_views(self, name: str, observation: Any) -> None:
        """After a single-listing view: whether it shows attributes, which tool opens pages and which shows offers."""
        s = self.s
        store, mem, reader = s.store, s.mem, s.reader
        roles = store.roles
        if not store.single_target(name) or name in (roles.get('cart'), roles.get('buy'), roles.get('refuse'), roles.get('ask')):
            return
        listing = mem.ident(observation) if isinstance(observation, dict) else None
        if not listing:
            return
        mem.facts_from[name] = mem.facts_from.get(name, False) or bool(attributes(observation))
        if name == roles.get('open') and not mem.facts_from[name]:
            other = next((t for t in store.schemas if t != name and store.single_target(t) and mem.facts_from.get(t) is not False and t not in (roles.get('cart'), roles.get('buy'), roles.get('refuse'), roles.get('ask'), roles.get('search'))), None)
            if other:
                roles['open'] = other
        if not mem.facts_from[name] and name != roles.get('open') and not roles.get('offer'):
            roles['offer'] = name
        if name == store.offer_tool():
            reader.offer_seen[listing] = store.step

    # -- ending when the loop stops
    def can_refuse(self) -> bool:
        store = self.s.store
        refuse = store.roles.get('refuse')
        if not refuse:
            return False
        faults: list[str] = []
        conform(self.quiet(refuse, dict(store.refuse_args)), store.schemas.get(refuse), 'args', faults)
        return not faults

    def finish(self, again: bool = False) -> str:
        """End the episode in code: check out a cart that covers every item (filled where it can be), else end
        without a purchase. With `again`, a refused checkout returns the store's words for one more round."""
        s = self.s
        store, basket, ref = s.store, s.basket, s.referee
        if store.finished or store.broke or store.left() <= 0:
            return ''
        buy, refuse = store.roles.get('buy'), store.roles.get('refuse')
        fresh = basket.fresh()
        holding = bool(basket.ids()) if fresh else bool(basket.lines)
        whole = True
        if holding and len(s.order.items) >= 2:
            whole = basket.fill_short() or not self.can_refuse()
            if whole and len(basket.ids()) >= 2:
                s.reader.fill_facts(basket.ids(), s.order.tail())
                if store.left() >= 3 and s.models.room() > 5:
                    s.reader.show_links(basket.ids(), s.order.tail())
            if not whole:
                s.log('short_cart_refused', lines=basket.ids())
        if store.finished or store.left() <= 0:
            return ''
        if whole and buy and (holding or (basket.touched and not fresh and store.left() >= 2)):
            faults: list[str] = []
            args = conform(dict(store.buy_args), store.schemas.get(buy), 'args', faults)
            if not faults:
                if store.left() >= 2:
                    ref.gate('order')
                if store.finished or store.left() <= 0:
                    return ''
                reply = store.call(buy, args)
                s.log('forced_buy', reply=clip(reply, 200))
                if store.finished or store.left() <= 0:
                    return ''
                if reply is not None and again:
                    ref.buys += 1
                    ref.refused_at = store.step
                    s.reader.observe(buy, reply.get('observation'))
                    return 'the checkout was refused: ' + s.reader.show(reply.get('observation'))
        if refuse:
            faults = []
            args = conform(self.quiet(refuse, dict(store.refuse_args)), store.schemas.get(refuse), 'args', faults)
            if not faults:
                if store.left() >= 2:
                    ref.gate('end')
                if store.finished or store.left() <= 0:
                    return ''
                reply = store.call(refuse, args)
                s.log('forced_refuse', reply=clip(reply, 200))
                if reply is not None and not store.finished:
                    s.reader.observe(refuse, reply.get('observation'))
                    return 'ending without a purchase was refused: ' + s.reader.show(reply.get('observation'))
        return ''


# ------------------------------------------------------------------------------------------------------ Session

class Session:
    """One episode: the parts, the shared version counter (bumped whenever what is known changes) and the trace."""

    def __init__(self, problem: dict[str, Any]):
        environment = problem['environment']
        view = environment['policy_view']
        self.version = 0
        self.trace: list[dict[str, Any]] = []
        self.models = Models(self)
        self.store = Store(self, view, environment['binding'])
        self.mem = Memory(self)
        self.order = Order(self, view)
        self.basket = Basket(self)
        self.reader = Reader(self)
        self.talk = Interview(self)
        self.referee = Referee(self)
        self.pilot = Pilot(self)

    def log(self, kind: str, **fields: Any) -> None:
        self.trace.append({'kind': kind, **fields})

    def run(self) -> list[dict[str, Any]]:
        if self.store.finished:
            return self.trace
        self.order.read()
        self.log('brief', brief=self.order.brief)
        self.pilot.run()
        self.end()
        return self.trace

    def end(self) -> None:
        self.log('end', finished=self.store.finished, steps=self.store.step, cost=round(self.models.cost, 5), calls=self.models.calls)


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        session = Session(problem_data)
    except Exception as exc:
        return [{'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]}]
    try:
        return session.run()
    except Exception as exc:
        session.log('crash', error=f'{type(exc).__name__}: {exc}'[:400])
        try:
            if not session.store.finished:
                session.pilot.finish()
        except Exception as again:
            session.log('crash', error=f'{type(again).__name__}: {again}'[:400])
        return session.trace


__all__ = ['agent_main']
