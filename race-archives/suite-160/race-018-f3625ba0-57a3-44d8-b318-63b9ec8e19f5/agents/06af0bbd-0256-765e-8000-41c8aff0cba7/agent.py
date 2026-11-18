from __future__ import annotations
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from src.agent.proxy_client import ProxyClient
LIGHT = 'anthropic/claude-haiku-5.5'
HEAVY = 'anthropic/claude-sonnet-5.5'
SPARE = 'openai/gpt-6-luna'
LOOP_CALLS = 42
ALL_CALLS = 60
SPEND_SOFT = 0.055
SPEND_HARD = 0.078
HEAVY_ROOM = 0.02
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 36000
GATE_ROUNDS = 3
AUTO_PAGES = 12
CHEAPER_OPENS = 3
FIRST_READS = 6
# price-capped category listing at the first checkout of a cheapest pick: needs this many steps left, this many
# seconds before the clock stop, and reads at most this many pages
CAPPED_ROOM = 18
CAPPED_CLOCK = 90
CAPPED_PAGES = 5
# the episode lasts 300 s; past CLOCK_STOP no model turn starts and the held cart is checked out (or the episode ended)
CLOCK_STOP = 268
EPISODE_SECONDS = 300
# before this many seconds the two question-plan samples are requested together (no call timeout is cut yet)
SIDE_BY_SIDE_UNTIL = 150
# a search paged this often without a new row of a requested kind holding its own words is not paged further
EMPTY_PAGES = 3
# search arguments filtering by seller or maker (matched on the argument name). Real r8-r14: after such a filter came
# back empty, the same filter with another name found rows 5 times in 581 (category filters 52%)
SELLER_ARG = re.compile('vendor|maker|brand|seller|dealer|merchant|manufactur', re.I)
# once the store refused an unbought close, a further such close is held back (twice at most)
NO_SECOND_END = True
# the engine profile follows the number of requested products (see SizeProfile)
SIZE_PROFILES = True
# question plan: a second PLAN sample from the other model; properties chosen by both go first
P1_ON = True
# question plan: past the first 14 rows the key table keeps every key seen on 2+ read listings (no purchase-option fields)
P2_ON = True
# question plan: with at most 2 messages in all, a hinted key with measure values does not take the first one
P3_ON = True
# adaptive L2 asking (11-08): a list names up to LIST_KEYS keys of one item, the first LIST_FRONT fresh; a grouped
# message names GROUP_KEYS keys per item (slot 3+ of a grouped message drew a need .15, real r15/r17)
LIST_KEYS = 8
LIST_FRONT = 3
GROUP_KEYS = 2
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
GUIDE = (
    'You buy on behalf of a single shopper in an online store; the store\'s tools are the only things you can do. The result is '
    'pass or fail, and only two outcomes pass: (a) a checkout where every requested item is present at its requested quantity, '
    'each line satisfies every requirement in force at checkout time, and the order total is within budget; or (b) when the store '
    'really has no listing satisfying everything, an ending without a purchase, done in the manner the request asks.\n\n'
    'PART 1 - STEPS, TOOLS, STORE\n'
    '- Reasoning is free; each tool call uses up one of a limited number of steps. Every store names its tools, arguments and '
    'result shapes differently, so read each tool\'s description and schema before using it and give argument names and allowed '
    'values exactly.\n'
    '- Searches may be narrow: a category filter, keywords matched only against the first words, a few rows per page, a token or '
    'offset for the next page. When the choice depends on every match, compare the stated total with the rows you have seen and '
    'keep paging with the exact token or offset you got back. Short queries work best, wanted value first, product noun last. A '
    'filtered search that returns nothing usually has a wrong filter value or parameter: retry with the label spelled exactly as '
    'the rows spell it, or filter on a different parameter.\n'
    '- A search row is just a summary. Stores split a listing\'s record differently: one view may hold the attributes '
    '(properties), another only the offer and the description, and a side-by-side view may hold either. When Desk has found the '
    'tool showing attributes, the ledger names it. Property requirements are judged on those attributes, so read them for each '
    'listing you might buy; one call showing several listings saves steps when it shows what you need. Do not reread a listing '
    'that has not changed since you read it.\n'
    '- You never message the shopper yourself: every question goes through Desk\'s interviewer. Answers appear in the conversation '
    'and are copied into the ledger.\n'
    '- A request may require steps on the way (an approval, a message, a check or a reading beforehand); the ledger lists them. The '
    'ones due just before the order or the ending Desk performs itself when you call the checkout or ending tool, so do not wait '
    'for them or drop the purchase because of them; any other such step you do with the store\'s tools (product questions go '
    'through Desk\'s interviewer).\n\n'
    'PART 2 - DOES A LISTING FIT?\n'
    '- The category label sets the product kind, and only a label matching the requested item as a whole counts; a similar label '
    'never does. A label that itself holds a requirement\'s value (an audience or season written into it) satisfies that '
    'requirement.\n'
    '- A property requirement is satisfied when the listing\'s attribute for that property gives the value. How the key is written '
    'does not matter (spaces, underscores, camelCase, prefixes), and one member of a comma or slash list is enough. Compare values '
    'word by word: case, plural/singular and ordinary synonyms are irrelevant, and a value containing the required words still '
    'matches ("deep teal" is teal, "extra roomy fit" is roomy). Other words do not match: a code or abbreviation is not the word it '
    'stands for, and "teal green" is not "sea green". Convert units before comparing numbers; limits include their end point; a '
    'range has to sit entirely inside a limit.\n'
    '- The deciding attribute is the one whose key names the requirement\'s property (that name or an ordinary synonym). If it '
    'says something else, the listing fails, whatever the title or other keys say. Keys about another property decide nothing, '
    'however close (a handle colour says nothing about the body colour). After opened listings Desk appends [evidence] lines '
    'naming the keys that match a requirement.\n'
    '- If no attribute covers the property, the wanted value under some other attribute (a features or styles list containing it, '
    'for example), in the title or in the description is accepted. A figure measuring something different never is (weight is not '
    'capacity), and neither is a bare number whose text does not say what it measures.\n'
    '- Some requirements state no value but refer to another listing ("the same X as the listing that <shop> sells with '
    '<property>", "within N of the X shown on the page of ..."). Settle these first: locate that listing (search by its category, '
    'brand, shop or words), open its full page with the single-listing or side-by-side tool (a search row will not do), and if that '
    'page refers on to a different listing for the property, locate and open that one likewise. The value comes from the final page; '
    'if that page gives a rule pairing values of the property (one value goes with another), the requirement takes the value '
    'paired with that page\'s own value. An order counts only if every page on the chain was opened. When the wanted value is '
    'anything but that listing\'s value, the rule excludes only listings with exactly that value, and on its own it never '
    'justifies ending without a purchase.\n'
    '- When a listing you mean to buy refers you to another page for more about itself, read that page first.\n'
    '- When one requested item has to fit another (an accessory for a device), pick both together from the listings\' own words (a '
    'list of compatible models, a model or series code found on both pages). If no accessory that meets its own requirements fits '
    'the device you chose, change to another device meeting its requirements that such an accessory fits; do this before any '
    'ending without a purchase.\n'
    '- Exclusions and closed lists are binding. Wishes ("ideally", "would be nice") only settle ties.\n'
    '- Never buy a listing shown as sold out. The order total, price times quantity summed over all lines, must stay within the '
    'budget (equal to it is allowed).\n\n'
    'PART 3 - CHOOSING AMONG FITTING LISTINGS\n'
    '- Cheapest (lowest total, smallest price): the correct line is the lowest-priced listing of the WHOLE fitting set. Gather that '
    'set by paging through it (rows rarely come sorted by price, and Desk fetches further pages of a short result on its own), test '
    'candidates from the lowest price upward, and take the first that fits fully. If a search accepts a maximum price, use it: list '
    'the category below your current best pick\'s price so only cheaper candidates are left.\n'
    '- Most or least of a measure: compare all fitting listings on it and take the extreme; a listing not stating the measure '
    'cannot win.\n'
    '- Any other request: the first listing satisfying everything is right; do not spend steps looking for a better one.\n'
    '- As soon as an opened listing satisfies every requirement of an item (for cheapest, most or least: after the fitting set has '
    'been compared), put it in the cart and continue. Lines in the cart often sell out shortly after, and carting early leaves '
    'steps for a replacement.\n'
    '- Every distinct requested item has its own cart line at its own quantity.\n\n'
    'PART 4 - CHANGES MID-RUN\n'
    '- The shopper or the store may change the task: a new budget, a requirement added or removed, a new price, a listing sold '
    'out. Changes arrive inside tool results, often as a refused checkout; a note counts no matter which tool\'s result carries '
    'it, and the ledger holds the requirements in force. A refusal does not end the run: apply every change, correct the cart and '
    'check out again.\n'
    '- Right before checkout, reread the record of each chosen listing, because price and stock can change without notice.\n\n'
    'PART 5 - FINISHING\n'
    '- Leave exactly the chosen lines in the cart (take out anything else), view the cart once, then check out in a call of its own.\n'
    '- End without buying only after a genuine search found no listing satisfying everything. Running low on steps proves nothing '
    'of the kind; in that case check out the best cart you have. Word that ending as the request asks, and if a remark is needed '
    'keep it short and name no listing.\n'
    '- Every reply must contain at least one tool call. Say little in your own words: what you know, then your next move.'
)
BRIEF = (
    'Turn the shopper\'s request and the store\'s tool list below into the task sheet a shopping agent will work from. Reply '
    'with one compact JSON object and nothing else (no indentation, no text around it), holding the fields that follow.\n\n'
    'PRODUCTS\n'
    '"items" - an entry for every distinct product: {"name": the kind of product in the request\'s words, "shelf": the exact '
    'category label when a tool lists labels, else "", "quantity": whole units of that one listing (1 if not said; a pair or set '
    'sold as a single listing is 1), "musts": the firm requirements, each as "property: constraint" with the exact value, limit, '
    'exclusion or closed list}.\n'
    'Writing musts:\n'
    '- Use the request\'s own word for the property ("a 12-inch blade" -> "blade length: 12 inch"; "meant for a lakeside occasion" '
    '-> "occasion: lakeside"); never swap in a neighbouring property.\n'
    '- A value that must be read from another listing is exactly ONE must holding that listing\'s full description ("rim colour: '
    'unlike the rim colour of the kettle Ovarno sells with a glass lid"); that listing\'s shop, kind and any trailing "with '
    '<property> <value> and <property> <value>" stay inside it and are never musts by themselves.\n'
    '- Price, budget, stock and quantity are not musts.\n'
    '- Back-story (who uses it, where, past problems, why it is being bought) is no must unless it clearly gives a property value '
    'or limit; words that are part of the product\'s name (a season or audience said along with the product) belong in "name".\n'
    '- A use or occasion made into a property of the product ("blazers for business wear", "a dress made for parties", "the '
    'occasion has to be party") is a must on that property ("occasion: business") unless the store\'s category label already '
    'carries it; the event, person or place it is for ("to wear at his wedding", "for a colleague") and how it should sit, feel or '
    'look in use are context.\n'
    '- One "and"-joined phrase naming products (a single "I want ..." with one quantity, often a store department\'s name) is ONE '
    'item; make several items only when each has its own quantity or its own requirements.\n\n'
    'MONEY AND CHOICE\n'
    '"budget" - the maximum total spend as a number in the request\'s unit; null if there is none.\n'
    '"budget_unit" - that unit as words (for example "cents") or null.\n'
    '"budget_scope" - "total" (whole order), "each" (per item), null (no limit).\n'
    '"in_stock_only" - true unless unavailable items are plainly acceptable.\n'
    '"pick" - "cheapest" (lowest price or smallest total wanted), "most" / "least" (an extreme of some measure), else "any"; '
    'comments that money is tight set no pick rule.\n'
    '"pick_measure" - for most/least the measure pushed up or down, else "".\n\n'
    'LIMITS AND CONDUCT\n'
    '"messages" - whole number of messages the agent may send the shopper, or null.\n'
    '"steps" - whole number of steps / moves / tool calls allowed, or null.\n'
    '"lookups" - the musts (copied exactly) whose value must first be read from another listing the request refers to; else [].\n'
    '"rules" - each instruction on how to work or how to finish, 12 words at most apiece.\n'
    '"wishes" - things that are only nice to have.\n'
    '"ask_first" - points to confirm with the shopper before choosing, only if the request explicitly asks for that; else [].\n'
    '"process" - actions demanded besides finding and buying the right products (the shopper\'s approval, telling or showing the '
    'shopper something, checking something with them, reading something first, using a certain tool), each {"step": the action in '
    'at most 15 words, "when": "choose" (before products are chosen) | "order" (right before checkout) | "end" (right before '
    'ending without a purchase), "messages": shopper messages it needs (0 if none)}; else []. Item requirements, budget, stock, '
    'message or step limits and the nothing-fits ending are not steps.\n'
    '"quiet_decline" - true if alternatives must not be named or proposed when nothing fits, false if they are wanted, null if '
    'the request is silent.\n'
    '"fits" - [{"item": item number, "with": item number}] for every requested item said to fit, suit or work with ANOTHER '
    'requested item (the agent\'s choice); else [].\n\n'
    'TOOLS\n'
    '"roles" - each job mapped to the exact tool name, null if no tool does it: "search"; "open" (attributes/properties of a single '
    'listing; with two single-listing tools, the one promising attributes, properties or specifications); "offer" (a single '
    'listing\'s current offer - price, stock - when some tool other than "open" gives it); "compare" (several listings in full); '
    '"ask" (message to the shopper); "cart" (changes the cart); "cart_view" (shows the cart); "buy" (checkout); "refuse" (ends '
    'without a purchase).\n'
    '"buy_args" - fixed arguments that make the "buy" tool perform the checkout (its mode, when the tool takes a mode argument), '
    'else {}.\n'
    '"refuse_args" - fixed arguments that make the "refuse" tool end without a purchase, else {}.\n'
)
PLAN = (
    'Your job: pick the properties a shopping agent should ask the shopper about.\n'
    'Background on shoppers: each one holds a handful of firm wishes that the request never mentions, every wish tied to a single '
    'property of a single requested product. A wish is revealed only when a question names that exact property. A property the '
    'shopper is indifferent to earns "don\'t mind", a property they never considered earns puzzlement, and every named property '
    'uses up some of a message.\n'
    'Input: the request; the items, each listed with what the request already fixes for it; a numbered list of the attribute keys '
    'seen on opened listings of each item\'s category, with a count of listings per key and some example values. A hidden wish may '
    'also concern a property absent from that list.\n\n'
    'Priority:\n'
    '1. Anything the request tells the agent to confirm with the shopper (the ASK FIRST section, if there is one), expressed as '
    'the property it concerns.\n'
    '2. Story clues. Stories often hint at a hidden wish without revealing its value - a remark about something the shopper cannot '
    'assess or leaves to you, another person whose taste rules, the look of the product or the place it goes, who will use or wear '
    'it, or what is inside it. Per clue, take the single property it hints at and its item, following the order of the story. '
    'Prefer a listed key of that item that suits the clue (wearer -> a gender/audience key; look -> the listed colour, pattern or '
    'finish key that most of its listings have). A clued property that none of the item\'s listings carry ranks below the three '
    'keys most carried for that item.\n'
    '3. Then listed keys in order: the properties buyers of this product compare first (commonly type/style, form/fit, '
    'size/capacity/dimensions, material, colour/pattern - whatever drives the purchase) and the keys carried by most of the '
    'item\'s listings; when 5 or more listings have been read, leave out keys present on just 1-2 of them.\n'
    'Exclude: properties the request already fixes for that item (named in a requirement - but a key whose values merely repeat a '
    'required value, like a pattern key showing the required decoration, is NOT fixed), other names for a property already '
    'picked, keys whose every value lists several purchase options, identifiers, model names, warranty and other logistics. Pack '
    'sizes, counts and quantities per pack are ordinary properties; a key marked [not the stated ...] is not that requirement\'s '
    'property, however much the names overlap. Rank every item separately: clues first, then listed keys.\n'
    '"label" holds the bare property name; for a listed key write out all its words in plain form ("lid colour" for lidColour, '
    '"rim finish" for rim_finish - never cut down to "colour"/"finish"); only the name, no example values, brackets or second '
    'property.\n'
    'Pick no more than @N@ per item, best first. Reply with a single JSON object and nothing else: {"ask": [{"item": item number, '
    '"label": "...", "hint": "up to 8 story words it was drawn from, or an empty string"}]}, each item\'s picks in ranked order'
)
HEAR = (
    'Below are the questions a shopping agent asked and the shopper\'s replies to them. Turn the replies into ONE JSON object, '
    'with nothing else in the output:\n'
    '{"needs": [{"item": item number, "need": "property: constraint", "points": true if the value has to be looked up on a '
    'different listing, otherwise false}],\n'
    ' "drop": [{"item": item number, "property": "a requirement the shopper abandons"}],\n'
    ' "free": [{"item": item number, "property": "..."}],\n'
    ' "crowded": [numbers of questions whose reply objected that too much was asked at once or asked for one question at a time '
    '- a mere "I do not understand" does not count as one],\n'
    ' "budget": the revised overall limit as a number, else null}\n'
    'Filling rules:\n'
    '- needs: one entry for every firm wish, carrying its exact value or limit with the unit ("at least 30 cm", "oval", "not '
    'wicker"), placed under the property the reply itself names (possibly not the one asked about); a property gets a single '
    'need even when several replies touch it. Copy values verbatim, nouns and units included ("at least 3 shelves", never '
    '"at least 3"). When a value is stated plainly, keep just the wanted part and drop emphasis or rule-out-the-rest wording '
    '("it must be oval, oval only" becomes "oval"); keep limits and explicit exclusions ("at least 30 cm", "not wicker").\n'
    '- When a reply fixes a value by pointing at another listing (like, unlike or near the value of some shop\'s product, '
    'identified by a few of its properties), write ONE need under the property the reply names, with "points": true and the '
    'reply\'s own words as the value, from the comparison word to the end of that description, including whatever it says about '
    'a listing named on that page; no trimming, no paraphrase. Properties that only describe the other product are not needs '
    'of this item.\n'
    '- free: per question, the property the shopper is indifferent to.\n'
    '- A value phrased in a homely or indirect way still counts as stated.'
)
NOTE = (
    'Partway through the task the store displayed the message below. It could have been written by the shopper or by the '
    'store itself. Classify it and pull out every change it brings.\n'
    'Reply with a single JSON object and nothing else:\n'
    '{"kind": one of "change" (something about the task moved - the budget, a requirement, a price, stock, or an order that was '
    'turned down), "error" (only a bad or rejected call) or "other",\n'
    ' "budget": the revised overall limit as a number in the unit of the request, else null,\n'
    ' "add": [{"item": item number, "need": "property: constraint"}],\n'
    ' "drop": [{"item": item number, "need": "the requirement that no longer applies"}],\n'
    ' "gone": [ids the message says cannot be had],\n'
    ' "sold_out": true when the message says a listing already picked (by name or not) is sold out, gone or unavailable,\n'
    ' "price_moved": true when the message says a price is different now,\n'
    ' "summary": "a brief sentence"}\n'
    'Leave no change out. A requirement that is swapped for another shows up twice: the old one under drop and the new one under add.'
)
CHECK = (
    'A shopping agent is about to pay, and you are the last check. Using only the records shown, judge whether the cart is '
    'exactly right.\n'
    'A correct cart has one separate line for each requested item at the requested quantity, nothing more, no line shown as out '
    'of stock, and a total of price x quantity within the budget. When a listing has several records, the most recent one '
    'counts.\n'
    'How to read the records:\n'
    '- A requirement is satisfied when the attribute for its property gives the value: how the key is spelled is irrelevant, one '
    'list member is enough, case, singular/plural and ordinary synonyms are accepted, so is a value containing the required words '
    '("deep teal" for teal); units get converted, limits include their end points, and a range has to lie entirely inside.\n'
    '- The attribute keyed by the requirement\'s property is final - any other value there fails the listing; keys for other '
    'properties, related or not, decide nothing (handle colour vs body colour). If several keys spell the same property (ignoring '
    'case, spaces, underscores, punctuation), one giving the value settles it regardless of the rest; a key about a different '
    'aspect of the same thing never overrides it. A bare figure in that key does not supply a word value, whatever the title or '
    'description say.\n'
    '- If no attribute covers the property, the wanted value in another attribute, the title or the description counts; a figure '
    'of a different kind does not (weight vs capacity), nor does a bare number without a label.\n'
    '- A category label that states a requirement satisfies it; the category label must match the requested item as a whole.\n'
    '- "unlike another listing\'s value" is satisfied unless the records show that exact value; a value taken from another '
    'listing\'s page is the value that page\'s pairing rule sets against the page\'s own value, when the page has such a rule for '
    'the property.\n'
    'With pick rule cheapest / most / least, also weigh the other listings shown: if one plainly satisfies everything and beats '
    'the cart\'s choice on that rule, the cart is wrong.\n'
    'The page of a cart listing may send the reader to another listing, named by its code, for the rest of its own details; put '
    'those codes in "refs" (not codes of devices or parts it only fits, works with or replaces).\n'
    'Reply with a single JSON object and nothing else: {"ok": true or false, "problems": ["brief concrete problem", ...], '
    '"better": [ids of listings that should take a cart line\'s place, best first], "refs": [codes]}. Use ok=false only for a '
    'problem the records clearly show, or for a requirement that no record covers.'
)
READING = (
    'a key that names the property decides it - spelling of the key is irrelevant, a single member of a list is enough, '
    'synonyms and unit conversions are accepted, and so is a value that contains the required words ("deep teal" for teal); '
    'limits include their end points; if several keys are spellings of the same property, one of them giving the value decides; '
    'a bare figure in such a key never supplies a word value. Keys about other properties decide nothing, even related ones '
    '(handle colour vs body colour). Only when no such key exists does the wanted value count from another attribute, the title '
    'or the description (an unlabelled bare number never counts). A category label that states a requirement satisfies it, and '
    'the category label must match the item. "unlike another listing\'s value" is satisfied unless the records show exactly that '
    'value. A value borrowed from another listing\'s page means the value that page\'s pairing rule matches with the page\'s own '
    'value, when the page has such a rule for the property. A listing shown as sold out or above budget never fits.'
)
AUDIT = (
    'The shopping agent is on the point of ending with no purchase, convinced that nothing fits. Check the listing records below '
    'against the requirements once more. Rules for reading the records: ' + READING + '\nA ROW record is only a row from search '
    'results whose page (attributes) nobody has read yet; a property the row does not show is neither passed nor failed.\n'
    'Reply with a single JSON object and nothing else: {"fits": {"item number": [ids that seem to satisfy every requirement, '
    'lowest price first, 4 at most]}, "open": [ids of ROW listings that are a requested kind, within budget, and whose page could '
    'turn out to fit, likeliest first, 6 at most], "reason": "a single sentence"}'
)
CLOSEST = (
    'Nothing on offer satisfies every requirement, so the shopping agent will end without a purchase, and the request wants the '
    'closest candidate pointed out. Go through each listing record shown: say which requested item it is an instance of, and which '
    'of that item\'s current requirements the record fails to show as satisfied. Rules for reading the records: ' + READING +
    '\nIgnore stock and price; the agent handles them. Reply with a single JSON object and nothing else: {"listings": [{"id": '
    'listing id, "item": the number of the requested item it is an instance of (0 when none), "misses": [the property name of '
    'every requirement not shown satisfied]}]}'
)
UNMIX = (
    'These requirements were extracted from a shopper\'s request. Some of them borrow their value from another listing that the '
    'request points to, and the words picking out that other listing (what kind it is, which shop sells it, the "with ..." '
    'properties that follow it) are at times wrongly extracted as requirements of the product being bought.\n'
    'Using the wording of the request, list which of the OTHER REQUIREMENTS below merely describe that other listing.\n'
    'Reply with a single JSON object and nothing else: {"describe_other": [such requirements, copied character for character; '
    'an empty list when all of them really belong to the product]}'
)
TRAIL = (
    'Certain requirements give no value of their own; they borrow it from another listing, whose page may in turn point to yet '
    'a different listing for that property. The shopping agent is ready to check out, but its order counts only after every listing '
    'on each such chain has been fully opened. Walk each requirement\'s chain link by link - beginning with the requirement\'s '
    'own description, then the words of the pages - using the records below (OPENED = a page the agent has opened; ROW = only a '
    'search-result row, page not read).\n'
    'A record is a link only when it matches the ENTIRE description: shop or brand, kind of product AND each stated property '
    'value (a described two-pack, say, excludes the same shop\'s single unit). Short of a full match the link counts as not found, '
    'even when that shop carries other listings of the kind; give "find" for it: the shop name and the description\'s values as a '
    'few search words.\n'
    'List each chain\'s links in order, ending at the first link that is not OPENED or not found - its words are not known yet.\n'
    'A page whose own words direct the reader to some other listing for the property (its value is to be read elsewhere, not '
    'from this item) is never the last link, even if it also states a value for that property: the listing it names is the next '
    'link. OPENED records tagged [sends on] are such pages.\n'
    'Copy the value in full, exactly as the last page states it. It is that page\'s own value, except where the request\'s own '
    'words for the requirement (REQUEST\'S WORDS below) ask for a counterpart to it - a value that pairs with or complements it '
    'rather than the same one: only then, if the page gives a rule pairing values of the property, is it the value that rule '
    'pairs with the page\'s own value. Words asking for the same, an identical or a matching value, or for simply what that page '
    'states, take the page\'s own value.\n'
    'Reply with a single JSON object and nothing else: {"chains": [{"need": the requirement, word for word, "links": [{"id": the '
    'listing id when one of the records below is this link, else "", "about": how the requirement or the previous page describes '
    'this listing, "find": "for a link not found: search words (its shop and the description\'s values)"}], "done": true when the '
    'last link is an OPENED page that itself gives the property\'s value (also when the requirement wants a value different from '
    'or unlike it - that page\'s value is the reference), else false, "value": the value the last page gives, as described '
    'above, or ""}]}'
)
GATE = (
    'The shopper\'s request requires the steps listed below before a shopping agent is allowed to @ACT@. For every step that '
    'must happen before it can @ACT@, use the numbered RECORD of the agent\'s actions to judge whether the step is complete for '
    'the cart in its present state. A step is complete when a RECORD line shows it complete; when the step is about the cart or '
    'the order, that line also has to come later than the last cart change, and the reply on it has to grant it. A question '
    'about product properties never completes a step.\n'
    'For a step still missing, supply the one store call that performs it next: a tool named in TOOLS plus its arguments. '
    'Anything addressed to the shopper uses the messaging tool, as one brief sentence or question about that step alone, in plain '
    'language, leaving out product properties or values the step does not need. Supply no call if all due steps are complete, or '
    'if the remaining messages or steps cannot perform the missing one.\n'
    'Reply with a single JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [numbers of '
    'the RECORD lines showing it complete]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply wants changed '
    'before proceeding, or "", "needs": [{"item": item number, "need": "property: constraint"} for every firm requirement such a '
    'reply gives]}'
)
SWAP = (
    'A listing in the shopping agent\'s cart is now sold out. Search the remaining listing records for stand-ins. '
    'Rules for reading the records: ' + READING + '\nReply with a single JSON object and nothing else: {"fits": [{"id": listing '
    'id, "price": the price as a number, copied from the record} - one entry per listing that seems to satisfy every requirement '
    'of the item the sold-out line was for, 6 at most], "reason": "a single sentence"}'
)
SHAPE = (
    'The text below came back from a store tool. Map its structure, naming each part by a key that literally occurs in that '
    'text or among the tool\'s parameters.\n'
    'Reply with a single JSON object and nothing else:\n'
    '{"rows": key of the list that holds the listings,\n'
    ' "count": key giving how many matches exist in total (not how many are on this page), else null,\n'
    ' "next": key whose value gets the following page, else null,\n'
    ' "next_arg": the tool parameter that accepts that value, else null,\n'
    ' "price": key inside a row carrying the price, else null,\n'
    ' "stock": key inside a row saying whether the listing is buyable right now, else null}'
)
GUARD_KEYS = 3
GUARD_SHARE = 0.0
GUARD_GAP = 0.34
# a key whose shown values are mostly lists of several phrases (a features or contents field), judged by shape alone
LIST_PARTS, LIST_PART_CHARS, LIST_VALUE_CHARS = 3, 10, 60
NOT_PROPERTY = '\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b'
# NOT_PROPERTY's purchase-option and identifier part (model, condition and video stay in the table)
TABLE_SKIP = '\\b(?:options?|variants?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|url|links?|images?|photos?)\\b'
TABLE_FIRST = 14
# a key naming the purchase options or variants a listing comes in (fit reviews leave such keys out)
CHOICE_KEY = '\\b(?:options?|variants?)\\b'
# a number followed by at most a short unit word (shape only)
MEASURE_VALUE = '^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$'
WIDEN_EACH = 6
FIT_CODE = '[A-Z][A-Za-z]{0,5}[ -]?\\d{3,6}'
# a model family written with trailing x's ("every QX12xx unit"): any code of that prefix and length
FIT_WILD = '[A-Z][A-Za-z]{0,5}[ -]?\\d{1,5}[xX]{1,4}'
PART_CODE = '[A-Z]{1,4}\\d{3,6}'
BOUND_WORDS = '\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or|same|than|like|either|unless|if|only)\\b'
# grammar endings stripped when two spellings of one word are compared
INFLECTIONS = ('s', 'es', 'ed', 'd', 'ing', 'er', 'ers', 'est', 'ies', 'ied', 'ly')
# inflection endings (grammar, not vocabulary): a word and its -ing/-ed/-s forms are one word
FORM_ENDS = ('ing', 'ies', 'ied', 'ed', 'es', 's', 'e', 'd')
NUMERAL = '\\s*-?\\d+(\\.\\d+)?\\s*'
SHOP_FIELD = 'seller|shop|store|vendor|brand|maker|merchant'
LIMIT_ARG = 'max|most|under|below|limit|cap|upto|up_to'
TITLE_FIELD = 'title|name|heading'
_NO = object()

def model_codes(text: Any, wild: bool = False) -> list[str]:
    """Model codes in a text, normalised."""
    found = re.findall('\\b(?:' + (FIT_WILD + '|' + FIT_CODE if wild else FIT_CODE) + ')\\b', str(text or ''))
    return [re.sub('[\\s-]+', '', hit).lower() for hit in found]

def codes_named(marks: list[str], known: set[str]) -> set[str]:
    """Known codes a set of fit marks names."""
    barred = {m[1:] for m in marks if m.startswith('!')}
    allowed = [m for m in marks if not m.startswith('!')]
    family = lambda mark, code: len(code) == len(mark) and code.startswith(mark.rstrip('x')) and code[len(mark.rstrip('x')):].isdigit()
    hits = {code for m in allowed for code in ([c for c in known if family(m, c)] if m.endswith('x') else [m])}
    return hits - barred

def fit_marks(text: Any) -> dict[str, Any]:
    text = str(text or '')
    first_upper = lambda hit: hit.group(1).upper() if hit else ''
    model_rx = '\\bmodel\\s*(?:(?:no\\.?|number|#)\\s*[:.]?|:)\\s*(' + FIT_CODE + ')\\b'
    sold_as = '\\b(?:also sold as|also known as|sold also as)\\s+([^.;]+)'
    codes = []
    for rx in (model_rx, sold_as):
        for hit in re.finditer(rx, text, re.I):
            codes.extend(model_codes(hit.group(1)))
    works: list[str] = []
    for hit in re.finditer('\\b(?<!not )(?<!never )(?:works with|compatible with|fits|goes with|for use with)\\s*:?\\s+([^.;|]+)', text, re.I):
        if re.search('\\bpart\\s+' + PART_CODE + '\\s+also\\s+$', text[max(0, hit.start() - 40):hit.start()], re.I):
            continue
        named, *left_out = re.split('\\b(?:except|excluding|but not)\\b', hit.group(1), maxsplit=1, flags=re.I)
        works.extend(model_codes(named, True))
        works.extend('!' + code for code in model_codes(''.join(left_out)))
    also = [(part.upper(), model_codes(listed))
            for part, listed in re.findall('\\bpart\\s+(' + PART_CODE + ')\\s+also fits\\s*:?\\s*([^.;]+)', text, re.I)]
    # a device taking a series of parts ("Takes part series XY5") fits each part whose own code is that series, a dash and
    # a number ("[XY5-1234]"); both sides carry the series as one shared mark (Arena r16: sofa / cover pairs)
    series_rx = ('\\btakes\\s+(?:parts?\\s+)?(?:of\\s+)?(?:the\\s+)?series\\s+([A-Z]{1,5}\\d{1,3})\\b',
                 '[\\[(]\\s*([A-Z]{1,5}\\d{1,3})-\\d{3,6}\\s*[\\])]')
    takes = ['series:' + hit.lower() for hit in re.findall(series_rx[0], text, re.I)]
    works.extend('series:' + hit.lower() for hit in re.findall(series_rx[1], text))
    own = first_upper(re.search('[\\[(]\\s*(?:part\\s+)?(' + PART_CODE + ')\\s*[\\])]', text, re.I))
    more = first_upper(re.search('\\b(?:more|details|see|read)\\s+(?:on|at|under|about)?\\s*(?:the\\s+)?part\\s+(' + PART_CODE + ')', text, re.I))
    return {'takes': takes, 'own': own, 'codes': codes, 'works': works, 'also': also, 'more': more}

def dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        return str(value)

def clip(value: Any, limit: int, depth: int = 0) -> Any:
    if depth > 7:
        return '...'
    inner = lambda v: clip(v, limit, depth + 1)
    if isinstance(value, dict):
        return {k: inner(v) for k, v in value.items()}
    if isinstance(value, list):
        return list(map(inner, value))
    return value[:limit] + '...' if isinstance(value, str) and len(value) > limit else value

def parse_object(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    body = text.strip()
    fenced = re.match('^```[A-Za-z]*\\s*(.*?)\\s*```$', body, re.S)
    body = fenced.group(1) if fenced else body
    start, end = body.find('{'), body.rfind('}')
    for piece in [body] + ([body[start:end + 1]] if 0 <= start < end else []):
        try:
            got = json.loads(piece)
        except ValueError:
            continue
        return got if isinstance(got, dict) else None
    return None

def plain_key(key: str) -> str:
    spaced = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(key))
    return re.sub('[\\s_.\\-]+', ' ', spaced).strip().lower()

def stems(text: str) -> frozenset[str]:
    trim = lambda w: w[:-1] if len(w) > 3 and w.endswith('s') else w
    return frozenset(map(trim, plain_key(text).split()))

def roots(text: str) -> set[str]:
    trim = lambda w: w[:-2] if re.search('(?:ch|sh|ss|x|z)es$', w) else w[:-1] if len(w) > 3 and w.endswith('s') and not w.endswith('ss') else w
    return {w for w in map(trim, plain_key(text).split()) if len(w) >= 3 and w not in ('and', 'for', 'the', 'with')}

def names_same(prop: frozenset[str], key: frozenset[str]) -> bool:
    return bool(prop and key) and (prop <= key or (len(key) >= 2 and key <= prop))

def worded(text: str) -> bool:
    """Search text holds words (not a bare code)."""
    text = str(text or '')
    if not re.search('[a-z]', text, re.I):
        return False
    return not re.search('\\d', text) or any(re.fullmatch('[a-z]+', token.strip('.,;:!?"\'()[]'), re.I) for token in text.split())

def plain_words(text: Any) -> list[str]:
    trim = lambda w: w[:-1] if len(w) > 3 and w.endswith('s') and not w.endswith('ss') else w
    return list(map(trim, re.findall('[a-z0-9]+(?:\\.[0-9]+)?', str(text).lower())))

def spread(value: str, stated: Any) -> str:
    """'phrase', 'apart' or '' for a multi-word value."""
    want = plain_words(value)
    if len(want) < 2 or re.search(BOUND_WORDS, value, re.I):
        return ''
    width, wanted = len(want), set(want)
    parts = [plain_words(p) for p in re.split('[,/;|]', str(stated))]
    windows = lambda words: (words[i:i + width] for i in range(len(words) - width + 1))
    if any(want in windows(words) for words in parts + [plain_words(stated)]):
        return 'phrase'
    return 'apart' if any(wanted <= set(words) for words in parts) else ''

def word_bases(word: str) -> dict[str, str]:
    bases = {word: ''}
    trimmed = (end for end in FORM_ENDS if word.endswith(end) and len(word) - len(end) >= 3 and not (end == 's' and word.endswith('ss')))
    for end in trimmed:
        bases.setdefault(word[:-len(end)] + ('y' if end in ('ies', 'ied') else ''), end)
    return bases

def consonants(word: str) -> str:
    return re.sub('(.)\\1', '\\1', re.sub('[aeiou]', '', word))

def form_kind(a: str, b: str) -> str:
    """'same', 'form', 'vowel' or '' for two words."""
    if a == b:
        return 'same'
    if re.search('\\d', a + b):
        return ''
    left, right = word_bases(a), word_bases(b)
    # a bare final e is no inflection by itself (plan / plane): one side must carry a real ending
    if any(len(base) >= 4 and {left[base], right[base]} - {'', 'e'} for base in left.keys() & right.keys()):
        return 'form'
    shifted = lambda x, y: (x.endswith('en') and not y.endswith('en') and min(len(x), len(y)) >= 5 and x[0] == y[0]
                            and len(consonants(x[:-2])) >= 2 and consonants(x[:-2]) == consonants(y[:-1] if y.endswith('e') else y))
    return 'vowel' if shifted(a, b) or shifted(b, a) else ''

def joined_forms(want: list[str], got: list[str], kinds: tuple[str, ...] = ()) -> tuple[str, ...] | None:
    """Word-by-word match allowing joins and forms."""
    if not (want and got):
        return None if want or got else kinds
    head, other = want[0], got[0]
    kind = form_kind(head, other)
    found = joined_forms(want[1:], got[1:], kinds + (kind,)) if kind else None
    if found is not None or re.search('\\d', head + other):
        return found
    joined = kinds + ('join',)
    if len(other) >= 3 and len(head) - len(other) >= 3 and head.startswith(other):
        return joined_forms([head[len(other):]] + want[1:], got[1:], joined)
    if len(head) >= 3 and len(other) - len(head) >= 3 and other.startswith(head):
        return joined_forms(want[1:], [other[len(head):]] + got[1:], joined)
    return None

def same_in_other_form(value: str, stated: Any) -> bool:
    """Value stated in another spelling of its words."""
    if re.search(BOUND_WORDS, value, re.I):
        return False
    pieces = '[a-z]+|[0-9]+(?:\\.[0-9]+)?'
    want = re.findall(pieces, value.lower())
    if len(''.join(want)) < 4 or want in (re.findall(pieces, part.lower()) for part in re.split('[,/;|]', str(stated))):
        return False
    # only the whole attribute: a list of several values does not state one of them
    kinds = set(joined_forms(want, re.findall(pieces, str(stated).lower())) or ())
    return bool(kinds - {'same'}) and ('vowel' not in kinds or bool(kinds & {'same', 'join'}))

def restore_value(request: str, must: str) -> str:
    """Give back a value word the reader cut short."""
    prop, sep, value = must.partition(':')
    words = re.findall('[a-z0-9]+', value.lower())
    named = re.findall('[a-z0-9]+', prop.lower())
    if not (sep and named and words) or len(words) > 3:
        return must
    if re.search('\\d', value) or re.search(BOUND_WORDS, value, re.I):
        return must
    width = len(named)
    tokens = re.findall('[a-z0-9]+', request.lower())
    # the three request words after each place the property's own words appear
    windows = [tokens[i + width:i + width + 3] for i in range(len(tokens) - width + 1) if tokens[i:i + width] == named]
    completions = lambda word, near: {t for t in near if t.startswith(word) and t[len(word):] not in INFLECTIONS}
    fixed = value
    for word in words:
        if len(word) < 3:
            continue
        for near in windows:
            if word in near:
                break
            grown = completions(word, near)
            if len(grown) == 1:
                fixed = re.sub('\\b' + re.escape(word) + '\\b', grown.pop(), fixed, count=1, flags=re.I)
                break
    return prop + sep + fixed if fixed != value else must

def type_names(schema: dict[str, Any]) -> list[str]:
    kind = schema.get('type')
    listed = [kind] if isinstance(kind, str) else kind or []
    return [k for k in listed if isinstance(k, str)]

def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)

TYPE_TESTS = {'string': lambda v: isinstance(v, str), 'integer': lambda v: isinstance(v, int) and not isinstance(v, bool),
              'number': is_number, 'boolean': lambda v: isinstance(v, bool), 'array': lambda v: isinstance(v, list),
              'object': lambda v: isinstance(v, dict), 'null': lambda v: v is None}

def to_integer(v: Any) -> Any:
    numeral = isinstance(v, str) and re.fullmatch(NUMERAL, v)
    return int(float(v)) if numeral or (isinstance(v, float) and v.is_integer()) else _NO

def to_number(v: Any) -> Any:
    if isinstance(v, str) and re.fullmatch(NUMERAL, v):
        figure = float(v)
        return figure if not figure.is_integer() else int(figure)
    return _NO

def to_boolean(v: Any) -> Any:
    word = v.strip().lower() if isinstance(v, str) else ''
    return word == 'true' if word in ('true', 'false') else _NO

# JSON-schema type -> how a near miss becomes that type
COERCE = {'integer': to_integer, 'number': to_number, 'string': lambda v: str(v) if is_number(v) else _NO,
          'boolean': to_boolean, 'array': lambda v: _NO if isinstance(v, list) or v is None else [v]}

def int_field(schema: dict[str, Any], key: str) -> Any:
    got = schema.get(key)
    return got if isinstance(got, int) else None

def conform_text(value: str, schema: dict[str, Any], where: str, faults: list[str]) -> str:
    longest, shortest = int_field(schema, 'maxLength'), int_field(schema, 'minLength')
    cut = value if longest is None else value[:longest]
    if shortest is not None and len(cut) < shortest:
        faults.append(f'{where} is too short')
    return cut

def conform_number(value: Any, schema: dict[str, Any], where: str, faults: list[str]) -> Any:
    edges = (('below', schema.get('minimum'), lambda edge: value < edge), ('above', schema.get('maximum'), lambda edge: value > edge))
    faults.extend(f'{where} is {word} {edge}' for word, edge, beyond in edges if is_number(edge) and beyond(edge))
    return value

def conform_list(value: list[Any], schema: dict[str, Any], where: str, faults: list[str]) -> list[Any]:
    most, least = int_field(schema, 'maxItems'), int_field(schema, 'minItems')
    if not (most is None or len(value) <= most):
        faults.append(f'{where} allows at most {most} entries')
    if not (least is None or len(value) >= least):
        faults.append(f'{where} needs at least {least} entries')
    inner = schema.get('items')
    return [conform(v, inner, f'{where}[{i}]', faults) for i, v in enumerate(value)] if isinstance(inner, dict) else value

def conform_record(value: dict[str, Any], schema: dict[str, Any], where: str, faults: list[str]) -> dict[str, Any]:
    props = schema.get('properties')
    props = props if isinstance(props, dict) else {}
    if schema.get('additionalProperties') is False:
        faults.extend(f'{where}.{name} is not an allowed argument' for name in value if name not in props)
    missing = [name for name in schema.get('required') or [] if name not in value]
    faults.extend(f'{where}.{name} is required' for name in missing)
    return {k: v if k not in props else conform(v, props[k], f'{where}.{k}', faults) for k, v in value.items()}

def conform(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    """Fit a value to a JSON schema, noting faults."""
    if not isinstance(schema, dict):
        return value
    kinds = type_names(schema)
    typed = lambda v: any(TYPE_TESTS.get(k, lambda _: True)(v) for k in kinds)
    if kinds and not typed(value):
        for kind in kinds:
            turned = COERCE[kind](value) if kind in COERCE else _NO
            if turned is not _NO:
                value = turned
                break
        if not typed(value):
            faults.append(f"{where} should be {'/'.join(kinds)}")
            return value
    allowed = schema.get('enum')
    if isinstance(allowed, list) and value not in allowed:
        if isinstance(value, str):
            lowered = value.strip().lower()
            value = next((o for o in allowed if isinstance(o, str) and o.lower() == lowered), value)
        if value not in allowed:
            faults.append(f'{where}: allowed values are {allowed}')
    if isinstance(value, str):
        return conform_text(value, schema, where, faults)
    if is_number(value):
        return conform_number(value, schema, where, faults)
    if isinstance(value, list):
        return conform_list(value, schema, where, faults)
    if isinstance(value, dict):
        return conform_record(value, schema, where, faults)
    return value

def shaped(args: Any, schema: Any) -> dict[str, Any] | None:
    """Schema-fitted args, or None on any fault."""
    faults: list[str] = []
    fitted = conform(args, schema, 'args', faults)
    return fitted if not faults else None

def units_after(have: int, count: int | None, removal: bool, sets: bool) -> int:
    """Line units after a cart call names it."""
    if not removal:
        step = count if count is not None else 1
        return step if sets else have + step
    return have - count if count else 0

def free_text(values: list[str]) -> bool:
    """Values too free-form to ask about."""
    if not values:
        return True
    count = len(values)
    mean = sum(map(len, values)) / count
    if mean > 40 or 2 * len([v for v in values if ' / ' in v]) >= count:
        return True
    vocab = lambda text: set(re.findall('[a-z]{3,}', text))
    overlapping = [v for v in values if vocab(v) & vocab(' '.join(o for o in values if o is not v))]
    return count >= 4 and len(set(values)) == count and mean > 12 and 2 * len(overlapping) < count

def cache_marked(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chat copy with cache marks on the ends."""
    if len(messages) < 2:
        return messages
    marked = [dict(m) for m in messages]
    ends = [m for m in (marked[0], marked[-1]) if isinstance(m.get('content'), str) and m.get('content')]
    for message in ends:
        message['content'] = [{'type': 'text', 'text': message['content'], 'cache_control': {'type': 'ephemeral'}}]
    return marked

def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value] if value and isinstance(value, (str, dict)) else []

def whole(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    integral = (isinstance(value, float) and value.is_integer()) or (isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.0+)?\\s*', value))
    return int(float(value)) if integral else None

def pick_number(value: Any, size: int) -> int:
    """An entry's item number, clamped."""
    return min(max(1, whole(value) or 1), size)

def as_stated(record: Any) -> Any:
    """Record without purchase-option keys."""
    # real race 15: lines whose required value sat only in such a key (or there and as bare title
    # figures) failed 36 of 36
    if not isinstance(record, dict):
        return record
    offered = lambda name, value: isinstance(value, str) and re.search(CHOICE_KEY, plain_key(name))
    return {key: {a: b for a, b in value.items() if not offered(a, b)} if isinstance(value, dict) else value for key, value in record.items()}

def id_like(value: Any, named: bool = False) -> bool:
    bare = value.strip() if isinstance(value, str) else ''
    if not (3 <= len(bare) <= 64 and ' ' not in bare):
        return False
    return named or bool(re.search('\\d', value))

def has_id_text(values: Any) -> bool:
    return any(id_like(v, True) and re.search('\\d', v) for v in values)

def pairs_block(value: str) -> dict[str, Any] | None:
    """A "name=value; ..." text as a block, or None."""
    pairs = [p.partition('=') for p in value.split('; ')]
    keyed = [bool(eq and name.strip()) for name, eq, _ in pairs]
    if len(pairs) < 2 or not pairs[0][1] or 2 * sum(keyed) <= len(pairs):
        return None
    block: dict[str, Any] = {}
    last = None
    for (name, _, rest), starts in zip(pairs, keyed):
        if starts:
            last = name.strip()
            block[last] = rest.strip()
        elif last is not None:
            block[last] += '; ' + name
    # a purchase-options key lists choices, not what this listing states (real race 15: order lines whose required
    # value sat only there failed 36 of 36)
    offered = [n for n, v in block.items() if isinstance(v, str) and re.search(CHOICE_KEY, plain_key(n))]
    block.update({n: 'choices offered, not stated specs: ' + block[n] for n in offered})
    return block

def text_listing(text: str) -> dict[str, Any] | None:
    """A " | "-joined key: value line as a record."""
    if not isinstance(text, str) or text.count(' | ') < 2:
        return None
    record: dict[str, Any] = {}
    last = None
    for part in text.split(' | '):
        hit = re.match('^\\s*([A-Za-z_][A-Za-z0-9_ ]{0,29}):(?:\\s+|$)(.*)$', part, re.S)
        fresh = hit.group(1).strip() if hit else None
        if fresh is not None and fresh not in record:
            last = fresh
            record[last] = hit.group(2).strip()
            continue
        if last is None:
            return None
        record[last] += ' | ' + part
    if len(record) < 3 or not has_id_text(record.values()):
        return None
    scalars = (('-?\\d{1,12}', int), ('-?\\d{1,12}\\.\\d+', float))
    fixed = {'True': True, 'False': False, 'None': None}
    for key, value in list(record.items()):
        block = pairs_block(value)
        if block is not None:
            record[key] = block
        elif value in fixed:
            record[key] = fixed[value]
        else:
            turn = next((kind for pattern, kind in scalars if re.fullmatch(pattern, value)), None)
            if turn is not None:
                record[key] = turn(value)
    return record

def as_records(node: Any, depth: int = 0) -> Any:
    """Observation with text listings parsed."""
    if depth > 3:
        return node
    if isinstance(node, str):
        return text_listing(node) or node
    if isinstance(node, list):
        return [as_records(v, depth + 1) for v in node]
    plain = isinstance(node, dict) and not has_id_text(v for v in node.values() if isinstance(v, str))
    return {k: as_records(v, depth + 1) for k, v in node.items()} if plain else node

def guess_id_key(rows: list[dict[str, Any]]) -> str | None:
    scored = []
    for key in rows[0]:
        named = 'id' in str(key).lower()
        values = [row.get(key) for row in rows]
        if all(id_like(v, named) for v in values):
            values = [v.strip() for v in values]
            if len(rows) == 1 or len(set(values)) == len(values):
                scored.append(((named, sum(map(len, values)) / len(values)), key))
    best = max(scored, key=lambda pair: pair[0], default=None)
    return None if best is None else best[1]

def mentions_row(node: Any, ident: str, depth: int = 0) -> bool:
    if depth > 5 or not isinstance(node, (list, dict)):
        return False
    if isinstance(node, dict):
        return any(mentions_row(v, ident, depth + 1) for v in node.values())
    holds = lambda row: isinstance(row, dict) and ident in [str(x).strip() for x in row.values() if isinstance(x, str)]
    return any(map(holds, node)) or any(mentions_row(v, ident, depth + 1) for v in node)

def listy(values: list[str]) -> bool:
    """Values shaped like feature lists."""
    def shaped_as_list(value: Any) -> bool:
        parts = [x.strip() for x in re.split('[,;|]', str(value)) if x.strip()]
        many = len(parts) >= LIST_PARTS and sum(map(len, parts)) >= LIST_PART_CHARS * len(parts)
        return len(str(value)) >= LIST_VALUE_CHARS or many
    return bool(values) and 2 * sum(map(shaped_as_list, values)) >= len(values)

def fact_block(node: Any) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    scalar = lambda v: isinstance(v, (str, int, float)) and not isinstance(v, bool)
    nested = (value for value in node.values() if isinstance(value, dict) and len(value) >= 2)
    blocks = [block for block in ({k: v for k, v in value.items() if scalar(v)} for value in nested) if len(block) >= 2]
    return max(blocks, key=len) if blocks else None

def has_bound(value: str, extra: str = '') -> bool:
    return bool(re.search('\\b(?:not|no|other than|unlike|' + extra + 'at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', value, re.I))

class ItemSpec:
    """A requested product and its requirements."""

    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name, self.shelf, self.qty, self.musts = name, shelf, qty, musts
        self.added: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []

    def live(self) -> list[str]:
        return [m for m in self.musts + self.added if m not in self.dropped]

class SizeProfile:
    """Message policy for one task size (1, 2, 3, 4+ items)."""
    # real races 10-13: one question plan wins at every size; the size only decides whether a message may
    # wait for a step due at an unbought ending (each product keeps one question and one more). An unsure
    # item count keeps the earlier behaviour

    def __init__(self, count: int, sure: bool):
        self.count = max(1, count)
        self.size = str(self.count) if self.count <= 3 else '4+'
        self.active = SIZE_PROFILES and sure

    def end_room(self, limit: int, held: int) -> bool:
        """Whether held messages may wait for an ending step."""
        return not self.active or limit - held >= self.count + 1

class Call:
    """One model tool call on its way through the desk."""

    def __init__(self, name: str, args: dict[str, Any], ending: bool, many: str | None):
        self.name, self.args, self.ending, self.many = name, args, ending, many
        self.searching = False
        self.extra = self.chain = self.reread = ''

    def mark(self) -> str:
        return self.name + dumps(self.args)

    def listed(self) -> list[Any] | None:
        """Ids a side-by-side call names, if it is one."""
        ids = self.args.get(self.many) if self.many else None
        return ids if isinstance(ids, list) else None

class Episode:
    """Shared episode record plus its services."""

    def __init__(self, problem: dict[str, Any]):
        environment = problem['environment']
        self.binding = dict(environment['binding'])
        self.view = environment['policy_view']
        self.request = str(self.view.get('query') or '')
        self.tools = [t for t in self.view.get('tools') or []
                      if isinstance(t, dict) and isinstance(t.get('function'), dict) and isinstance(t['function'].get('name'), str)]
        self.schemas = {}
        for spec in (t['function'] for t in self.tools):
            params = spec.get('parameters')
            self.schemas[spec['name']] = params if isinstance(params, dict) else {'type': 'object', 'properties': {}}
        cap = self.view.get('max_steps')
        steps = int(cap) if whole(cap) is None and isinstance(cap, float) else whole(cap)
        self.max_steps = steps if steps and steps > 0 else 30
        self.finished = bool(self.view.get('finished'))
        self.step = self.version = 0
        self.msg_cap = self.msg_sent = self.msg_reserve = 0
        self.cap_stated = False
        self.checkouts = self.end_refusals = 0
        self.side_ok = self.side_bad = 0
        self.refused_at = -1
        self.budget: Any = None
        self.confirmed: str | None = None
        self.items: list[ItemSpec] = []
        self.brief: dict[str, Any] = {}
        self.roles: dict[str, str] = {}
        # the trace and the running notes, leads, queued note readings and recent results
        self.trace, self.process, self.notes, self.leads, self.queued, self.recent = [], [], [], [], [], []
        self.refs: list[tuple[int, str]] = []
        self.bound: list[tuple[int, int]] = []
        self.lead_items, self.fact_tools, self.ref_values, self.ref_pages = {}, {}, {}, {}
        self.cache: dict[str, tuple[int, int, str]] = {}
        self.offer_seen: dict[str, int] = {}
        self.added_at: dict[str, int] = {}
        self.page_seen: set[str] = set()
        self.gone: dict[str, int] = {}
        self.unavailable, self.opened = set(), set()
        self.id_args: set[tuple[str, str]] = set()
        self.repick: set[int] = set()
        self.began = time.monotonic()
        self.profile = SizeProfile(1, False)
        self.llm = ModelDesk(self)
        self.store = StoreGateway(self)
        self.mem = ListingMemory()
        self.scout = SearchScout()
        self.req = RequestReader()
        self.talk = Interviewer()
        self.news = ChangeWatcher()
        self.judge = Checker()
        self.chains = ChainWalker()
        self.basket = CartDesk()
        self.gates = Gatekeeper()
        self.close = Closer()
        self.screen = Screener()
        self.loop = TurnLoop(self)

    # -- clock, steps and messages --
    def elapsed(self) -> float:
        return time.monotonic() - self.began

    def past_stop(self) -> bool:
        """Past the clock stop: no new turns."""
        return self.elapsed() > CLOCK_STOP

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def clock_room(self) -> int:
        """Steps the clock still allows at this pace."""
        if self.step < 6:
            return 10 ** 6
        pace = max(3.0, self.elapsed() / self.step)
        return max(0, int((CLOCK_STOP - self.elapsed()) / pace))

    def free_messages(self) -> int:
        return self.msg_cap - self.msg_sent - max(0, self.msg_reserve - self.gates.sent)

    def stage_open(self, stage: str) -> bool:
        if not any(p['when'] == stage for p in self.process):
            return False
        return (stage, self.cart_stamp() + f'#{self.version}') not in self.gates.cleared

    def reserve(self) -> int:
        """Steps held back for the close."""
        talk = any(p['when'] == 'order' and p['messages'] > 0 for p in self.process)
        return (2 if talk and self.msg_sent < self.msg_cap and self.stage_open('order') else 1) + self.offers_due() + self.refusal_room()

    def refusal_room(self) -> int:
        """Steps kept free to answer a refused first ending."""
        # Arena r16 noask b02/b03: 11-13 of 36-44 first checkouts were refused by a store change (sold out, new price or
        # budget); 4 came with no step left to answer. Two steps stay free for that answer until the first ending of
        # either kind was refused: real races 12-16, after a refused unbought ending, episodes that went on to act passed
        # 16-52% vs 4-17% for a second ending (ours ended again 13 times, 0 passed; the kept room had left only the ending)
        return 2 if self.checkouts == 0 and self.end_refusals == 0 and self.max_steps >= 30 else 0

    def page_cap(self) -> int:
        """Search pages Desk may read on its own in this episode."""
        # Arena r16 noask b10: in a 35-step four-item task Desk read 12 later pages of two searches, so the first listing
        # was opened at step 22 and later reads were refused for want of steps; real races 12-16: our episodes spending 30%+
        # of their steps on later pages passed 1 of 17
        return min(AUTO_PAGES, max(2, self.max_steps // 6))

    def offer_view(self) -> str | None:
        """A single-listing offer tool apart from the attribute view, or None."""
        tool = self.roles.get('offer')
        if tool and tool != self.roles.get('open') and self.id_named(tool):
            return tool
        # the brief can leave the offer role empty (Arena r12 b08: 36 of 66 briefs; 2 of 10 'process requirement missed'
        # orders never read a line's own page, KING v11 passed them reading it after carting): a single-listing tool no
        # role took, taking one id-named argument, is that page
        taken = {t for t in self.roles.values() if t}
        spare = [t for t in self.schemas if t not in taken and t != self.attr_tool() and self.fact_tools.get(t) is not True and re.search('id|code|number|sku|listing|item|product', self.id_param(t) or '', re.I)]
        return spare[0] if len(spare) == 1 else None

    def unoffered(self) -> list[str]:
        """Cart lines whose offer page was not read since they were carted."""
        if not self.offer_view():
            return []
        return [i for i in basket_ids(self) if i not in self.gone and self.offer_seen.get(i, -1) < self.added_at.get(i, 0)]

    def offers_due(self) -> int:
        """Offer reads still owed before checkout: one per uncarted item and per unread cart line."""
        # Arena r16 noask b01: 15/15 'process requirement missed' orders checked out with a cart line whose offer page was
        # never read (tight step ends); every paid order had read each line's offer page after carting it
        if not self.offer_view():
            return 0
        held = basket_ids(self)
        return min(4, len(self.unoffered()) + max(0, len(self.items) - len(held)))

    def cart_stamp(self) -> str:
        return dumps(sorted(self.basket.lines.items()))

    # -- tools --
    def id_param(self, tool: str | None) -> str | None:
        """A single-listing tool's id argument."""
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        needed = schema.get('required') or []
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in type_names(v) and 'enum' not in v]
        target = next((k for k in needed if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not tool or not target or any(k != target for k in needed):
            return None
        return target

    def side_param(self) -> str | None:
        schema = self.schemas.get(self.roles.get('compare') or '') or {}
        props = schema.get('properties') or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in type_names(v)), None)
        if not target or any(k != target for k in schema.get('required') or []):
            return None
        return target

    def side_room(self) -> int:
        spec = ((self.schemas.get(self.roles.get('compare') or '') or {}).get('properties') or {}).get(self.side_param() or '') or {}
        # stores refuse fewer ids than their schema allows (real r8: every call of 6+ ids, r9/r12: 3+), so a size
        # once refused caps the schema's own limit
        if isinstance(spec.get('maxItems'), int):
            return min(spec['maxItems'], max(1, self.side_bad - 1)) if self.side_bad else spec['maxItems']
        if self.side_bad:
            return max(2, min(self.side_bad - 1, max(self.side_ok, 2)))
        return max(4, self.side_ok)

    def page_tools(self) -> set[str]:
        return {t for t in (self.roles.get('offer') or self.roles.get('open'), self.roles.get('compare')) if t}

    def id_named(self, tool: str | None) -> bool:
        """A single-listing tool whose one argument is named for an id, and not the search tool."""
        return bool(tool) and tool != self.roles.get('search') and bool(re.search('id|code|number|sku|listing|item|product', self.id_param(tool) or '', re.I))

    def offer_tool(self) -> str | None:
        # Arena r13 b09: the brief set offer to the search tool; Desk's offer reads then searched listing ids as text
        offer = self.roles.get('offer')
        return offer if self.id_named(offer) else self.roles.get('open')

    def attr_tool(self) -> str | None:
        known = next((t for t, ok in self.fact_tools.items() if ok and self.id_param(t)), None)
        if known:
            return known
        tool = self.roles.get('open')
        return tool if self.fact_tools.get(tool or '') is not False and self.id_param(tool) else None

    def side_view_bare(self) -> bool:
        seen = self.fact_tools.get(self.roles.get('compare') or '')
        if seen is False:
            return True
        return seen is None and (self.fact_tools.get(self.attr_tool() or '') is True or bool(self.roles.get('offer')))

    def reading_tools(self) -> set[str]:
        return {self.roles.get(r) for r in ('search', 'open', 'offer', 'compare')} - {None}

    def closing_tools(self) -> set[str]:
        return {tool for tool in (self.roles.get('buy'), self.roles.get('refuse')) if tool}

    def role_tools(self, *skip: str) -> set[Any]:
        """Tools of all roles except the skipped ones."""
        return {tool for role, tool in self.roles.items() if role not in skip}

    def mode_ok(self, role: str, name: str, args: dict[str, Any]) -> bool:
        """Call is the role's tool in the brief's mode."""
        if name != self.roles.get(role):
            return False
        fixed = self.brief.get(role + '_args')
        return all(args.get(k) == v for k, v in (fixed if isinstance(fixed, dict) else {}).items())

    # -- requirements and records --
    def live_needs(self) -> list[tuple[int, str]]:
        return [(number, need) for number, item in enumerate(self.items, 1) for need in item.live()]

    def open_pointers(self) -> list[str]:
        """Pointer requirements still unresolved."""
        live = {need for _, need in self.live_needs()}
        return [need for _, need in self.refs if need in live and need not in self.ref_values]

    def is_gone(self, ident: str) -> bool:
        return ident in self.unavailable or ident in self.gone

    def render(self, observation: Any) -> str:
        text = dumps(clip(observation, SHOW_CHARS))
        return text[:OBS_LIMIT] + '...(cut)' if len(text) > OBS_LIMIT else text

    def log(self, kind: str, **fields: Any) -> dict[str, Any]:
        entry = {'kind': kind, **fields}
        self.trace.append(entry)
        return entry

    def ingest(self, name: str, observation: Any) -> str:
        """Absorb a result and read any note in it."""
        mem_absorb(self, observation)
        lines = news_free_lines(self, observation)
        if not lines:
            return ''
        meaning = news_read_note(self, lines)
        if meaning:
            self.queued.append(f'(seen on {name}) {meaning}')
        return meaning

    def cheapest(self) -> bool:
        return self.brief.get('pick') == 'cheapest'

    def price_rank(self, ident: str) -> float:
        price = mem_price(self, ident)
        return price if price is not None else float('inf')

    def numeric_budget(self) -> Any:
        return self.budget if is_number(self.budget) else None

def chat_body(model: str, messages: list[dict[str, Any]], max_tokens: int,
              tools: list[dict[str, Any]] | None) -> dict[str, Any]:
    """The proxy request body: cache marks for Anthropic models, tools in loop turns."""
    anthropic = model.startswith('anthropic/')
    body: dict[str, Any] = {'model': model, 'messages': cache_marked(messages) if anthropic else messages,
                            'max_tokens': max_tokens, 'temperature': 0}
    if not tools:
        return body
    body.update(tools=tools, tool_choice='auto')
    # Arena r16 noask b04: tool-loop turns ran ~1.2-2.5k completion tokens for ~150 visible (default thinking),
    # 7.5-12 s each; 27/81 episodes hit the clock with ~205 s of loop calls. Loop turns think at low effort (as
    # terrya2 r7-r9 and YesYesYes r16 send it through the race proxy); readings and checks keep full thinking
    if anthropic:
        body['reasoning'] = {'effort': 'low'}
    return body

def timed_post(client: Any, path: str, body: dict[str, Any], room: float, floor: int) -> Any:
    """Post with the client's wait cut so all its tries end within room seconds (never below floor)."""
    usual = getattr(client, 'timeout', None)
    if not isinstance(usual, (int, float)):
        return client.post_verbose(path, body)
    tries = 1 + max(0, int(getattr(client, 'max_retries', 0) or 0))
    client.timeout = max(floor, int(min(usual, room / tries)))
    try:
        return client.post_verbose(path, body)
    finally:
        client.timeout = usual

class ModelDesk:
    """Model calls: choice, retries, time cuts, JSON."""

    def __init__(self, ep: Episode):
        self.ep = ep
        self.calls = 0
        self.waited = 0
        self.cost = 0.0
        self.out_of_credit = False

    def models_for(self, limit: float = SPEND_SOFT) -> list[str]:
        if self.cost <= limit and self.cost + HEAVY_ROOM <= SPEND_HARD:
            return [HEAVY, LIGHT]
        return [LIGHT] if self.cost <= SPEND_HARD else []

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int,
              tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        for model in models:
            # past the clock stop a fallback model would start a call the 300 s limit cuts off
            if self.calls >= ALL_CALLS or (model != models[0] and self.ep.past_stop()):
                return None
            self.calls += 1
            for tries in range(len(WAITS) + 1):
                message, status = self.post(model, messages, max_tokens, tools)
                if message is not None:
                    return message
                transient = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                if not transient or tries >= (1 if status in (None, 408) else len(WAITS)):
                    break
                if self.waited + WAITS[tries] > WAIT_TOTAL or self.ep.past_stop():
                    break
                self.waited += WAITS[tries]
                time.sleep(WAITS[tries] * (0.75 + 0.05 * (self.calls % 11)))
            if status == 402:
                self.out_of_credit = True
                return None
        return None

    def post(self, model: str, messages: list[dict[str, Any]], max_tokens: int,
             tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        """One proxy chat call: (message, 200) or (None, status)."""
        ep = self.ep
        body = chat_body(model, messages, max_tokens, tools)
        # each call must end inside the episode's 300 s, so its wait is cut to the seconds left (Arena run 13021: a call
        # started near the clock stop ran past the limit and the episode ended unfinished)
        room = EPISODE_SECONDS - ep.elapsed() - 6
        if room < 4:
            ep.log('llm_skip', model=model, room=round(room, 1))
            return None, 0
        result = timed_post(_llm, '/inference/chat/completions', body, room, 4)
        data = result.data if isinstance(result.data, dict) else None
        if not data or not data.get('choices'):
            failed = (data or {}).get('error')
            if isinstance(failed, dict):
                status, text = whole(failed.get('code')), failed.get('message')
            else:
                error = result.error or {}
                status, text = error.get('status'), error.get('body')
            ep.log('llm_fail', model=model, status=status, body=str(text or '')[:200])
            return None, status
        spent = (data.get('usage') or {}).get('cost')
        if isinstance(spent, (int, float)):
            self.cost += float(spent)
        message = data['choices'][0].get('message') or {}
        if not (message.get('content') or message.get('tool_calls')):
            ep.log('llm_fail', model=model, status=0)
            return None, 0
        return message, 200

    def ask(self, prompt: str, models: list[str], max_tokens: int = 1800) -> dict[str, Any] | None:
        """First JSON object any of the models returns."""
        for model in models:
            message = self.think([{'role': 'user', 'content': prompt}], [model], max_tokens)
            got = parse_object((message or {}).get('content'))
            if got is not None:
                return got
            if self.stopped():
                return None
        return None

    def stopped(self) -> bool:
        return self.ep.store.blocked or self.out_of_credit or self.calls >= ALL_CALLS

class StoreGateway:
    """Store actions with step bookkeeping."""

    def __init__(self, ep: Episode):
        self.ep = ep
        self.blocked = False

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        ep = self.ep
        payload = {**ep.binding, 'step': ep.step + 1, 'action': {'name': name, 'args': args}}
        for _ in range(4):
            # each store call waits at most for the episode's remaining time shared over the client's retries: a slow
            # store near the end must not hold the episode past its limit without a finish
            result = timed_post(_env, '/environment/call', payload, EPISODE_SECONDS - ep.elapsed() - 3, 3)
            if isinstance(result.data, dict):
                return self.accept(name, args, result.data)
            error = result.error or {}
            status, body = error.get('status'), str(error.get('body') or '')
            ep.log('env_fail', status=status, body=body[:200])
            if status in (400, 401, 402, 403, 404, 422):
                self.blocked = self.blocked or status == 402
                return None
            if status == 409:
                if not self.step_resync(body):
                    return None
                payload['step'] = ep.step + 1
        return None

    def step_resync(self, body: str) -> bool:
        """After a step conflict, take the store's step count; False when the episode is over or the cause unknown."""
        ep = self.ep
        low = body.lower()
        if any(word in low for word in ('finish', 'quarantin', 'finaliz')):
            ep.finished = True
            return False
        expected = re.search('expected (\\d+)', body)
        if expected:
            ep.step = int(expected.group(1)) - 1
            return True
        if 'reuse' in low or 'different action' in low:
            ep.step += 1
            return True
        return False

    def accept(self, name: str, args: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        ep = self.ep
        ep.step = data['step'] if isinstance(data.get('step'), int) else ep.step + 1
        if data.get('finished'):
            ep.finished = True
        if 'observation' in data:
            data['observation'] = as_records(data['observation'])
        seen = data.get('observation')
        if (name in ep.page_tools() or name == ep.offer_view()) and not (isinstance(seen, dict) and seen.get('error')):
            # a view that came back without the listing (a refused side-by-side size) has not opened it
            shown = dumps(seen)
            for value in args.values():
                ep.opened.update(v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str) and v.strip() in shown)
        return data

class RequestReader:
    """Request reading and the ledger text."""

    def __init__(self) -> None:
        self.routed: tuple[int, bool] | None = None

def req_read(ep) -> None:
    """Brief, roles, items, limits, message reserve."""
    ep.brief = req_first_reading(ep)
    named = ep.brief.get('roles')
    pairs = named.items() if isinstance(named, dict) else ()
    ep.roles = {job: tool for job, tool in pairs if isinstance(tool, str) and tool in ep.schemas}
    req_settle_end_modes(ep)
    req_collect_links(ep, req_collect_items(ep, as_list(ep.brief.get('items'))))
    req_collect_limits(ep)
    ep.msg_reserve = req_close_reserve(ep, req_profile_now(ep))

def req_profile_now(ep) -> SizeProfile:
    """Profile for the item count (re-chosen on change)."""
    sure = not req_shared_shelf(ep.brief) and all(t.get('kind') != 'items_merged' for t in ep.trace)
    stamp = (len(ep.items), sure)
    if ep.req.routed == stamp:
        return ep.profile
    ep.req.routed = stamp
    ep.profile = SizeProfile(*stamp)
    ep.log('profile', size=ep.profile.size, sure=sure)
    return ep.profile

def req_tool_digest(ep, skip: set[str] | None = None) -> str:
    hidden = skip or set()
    line = lambda spec: f"- {spec['name']}: {spec.get('description', '')} | parameters: {dumps(spec.get('parameters') or {})}"
    return '\n'.join(line(t['function']) for t in ep.tools if t['function']['name'] not in hidden)

def req_first_reading(ep) -> dict[str, Any]:
    opening = ep.view.get('opening')
    note = [] if opening is None else ['\n\nOPENING NOTE FROM THE STORE:\n', dumps(clip(opening, 1500))]
    prompt = ''.join([BRIEF, '\nTOOLS:\n', req_tool_digest(ep), '\n\nREQUEST:\n', ep.request] + note)
    usable = lambda got: bool(as_list(got.get('items'))) and isinstance(got.get('roles'), dict) and bool(got['roles'].get('buy'))
    first = ep.llm.ask(prompt, [LIGHT], 4000) or {}
    if ep.llm.out_of_credit:
        return first
    if not usable(first):
        return ep.llm.ask(prompt, [HEAVY, LIGHT], 2500) or first
    if not req_shared_shelf(first):
        return first
    second = ep.llm.ask(prompt, [HEAVY], 2500) or {}
    return second if usable(second) else first

def req_settle_end_modes(ep) -> None:
    """Separate buy and refuse modes of one tool."""
    ep.brief.update({key: {} for key in ('buy_args', 'refuse_args') if not isinstance(ep.brief.get(key), dict)})
    tool = ep.roles.get('buy')
    buying, ending = ep.brief['buy_args'], ep.brief['refuse_args']
    distinct = buying and ending and buying != ending
    if tool and tool == ep.roles.get('refuse') and not distinct:
        prompt = (f'Store tool {tool} can either check out or end without a purchase, depending on its arguments. Schema: '
                  + dumps(ep.schemas.get(tool)) + '\nReply with ONE JSON object and nothing else: {"buy_args": the fixed arguments that make it check out, '
                  '"refuse_args": the fixed arguments that make it end without a purchase} (only the arguments that select the mode).')
        modes = ep.llm.ask(prompt, [LIGHT, HEAVY], 1500) or {}
        found = modes.get('buy_args'), modes.get('refuse_args')
        if all(isinstance(m, dict) for m in found) and found[0] != found[1]:
            ep.brief['buy_args'], ep.brief['refuse_args'] = found

def musts_clash(one: ItemSpec, other: ItemSpec) -> bool:
    """Two items set one property to different values."""
    halves = lambda must: (plain_key(must.split(':')[0]), plain_key(must.partition(':')[2]))
    theirs = [halves(m) for m in other.musts]
    return any(key == k and value != v for key, value in map(halves, one.musts) for k, v in theirs)

def req_item_of(ep, entry: dict[str, Any]) -> ItemSpec:
    """One brief entry as an item, its values restored to the request's wording."""
    written = [str(m) for m in as_list(entry.get('musts')) if m]
    restored = [restore_value(ep.request, m) for m in written]
    if restored != written:
        before, after = ([m for m in one if m not in other] for one, other in ((written, restored), (restored, written)))
        ep.log('request_value', read=before, now=after)
    units = max(whole(entry.get('quantity')) or 0, 1)
    name = str(entry.get('name') or 'the product')
    # the brief counts "a pair or set sold as one listing" as 1, and with a size value beside it also turned an
    # explicit "N x <item>" into 1 (Arena r17 b11 and KING; real r17: orders holding fewer units than the request's
    # written count passed 0/19, matching ones about a third): a count the request writes before the item's name wins
    # over a brief quantity of 1; the pack size stays a must
    written = req_written_count(ep.request, name)
    if units == 1 and written and written > 1:
        ep.log('request_count', item=name, brief=units, now=written)
        units = written
    return ItemSpec(name, str(entry.get('shelf') or ''), units, restored)

def req_words(text: str) -> set[str]:
    """Word roots without possessive endings or punctuation ("the lamps's colour," -> lamp, colour)."""
    return roots(re.sub("[^a-z0-9]+", ' ', re.sub("['\u2019]s?(?![a-z])", '', str(text).lower())))

def req_written_count(request: str, name: str) -> int | None:
    """The one count the request writes directly before the item's name ("N x <item>", "N <item>"), else None."""
    tokens = [m.group(0) for m in re.finditer("[a-z0-9]+(?:['\u2019][a-z]+)?|[^\\sa-z0-9]", str(request).lower())]
    sequence = lambda words: [r for word in words for r in sorted(req_words(word))]
    want = sequence(re.findall('[a-z0-9]+', str(name).lower()))
    if not want:
        return None
    # English words that tie a number to a property or a bound rather than to the item ("the set size is 6 cups",
    # "a box of 10 forks", "at most 3 rugs"): grammar, not product vocabulary. A number written that way anywhere
    # in the request is a value, not the item's count (real r17: pack-size numbers repeat as "30 hangers")
    linking = {'is', 'are', 'was', 'be', 'of', 'per', 'with', 'has', 'have', 'holds', 'contains', 'includes', 'in', 'than', 'to',
               'by', 'from', 'least', 'most', 'up', 'over', 'under', 'about', 'around', 'between', 'each', 'every'}
    numbers = [(at, re.fullmatch('(\\d{1,3})(x|\u00d7)?', token)) for at, token in enumerate(tokens)]
    valued = {int(hit.group(1)) for at, hit in numbers if hit and at and (tokens[at - 1] in linking or tokens[at - 1] in ('.', '$'))}
    found = set()
    for at, hit in numbers:
        if not hit or int(hit.group(1)) in valued or not 1 <= int(hit.group(1)) <= 99:
            continue
        rest = tokens[at + 1:]
        rest = rest[1:] if not hit.group(2) and rest and rest[0] in ('x', '\u00d7', '*') else rest
        phrase = []
        for word in rest[:len(want) + 3]:
            if not re.match('[a-z0-9&]', word):
                break
            phrase.append(word)
        # the item's own words follow the count, in their order ("3 x tea cups where ...")
        if sequence(phrase)[:len(want)] == want:
            found.add(int(hit.group(1)))
    return found.pop() if len(found) == 1 else None

def req_collect_items(ep, entries: list[Any]) -> dict[int, int]:
    """Items from the brief; entry number -> item number."""
    became: dict[int, int] = {}
    for number, entry in enumerate([e for e in entries if isinstance(e, dict)], 1):
        item = req_item_of(ep, entry)
        shelf = plain_key(item.shelf) if item.shelf.strip() else None
        home = next((n for n, kept in enumerate(ep.items, 1)
                     if shelf is not None and plain_key(kept.shelf) == shelf and not musts_clash(kept, item)), 0)
        if home:
            kept = ep.items[home - 1]
            kept.musts.extend(m for m in item.musts if m not in kept.musts)
            ep.log('items_merged', into=kept.name, dropped=item.name)
        else:
            ep.items.append(item)
            home = len(ep.items)
        became[number] = home
    if not ep.items:
        ep.items.append(ItemSpec('the requested product', '', 1, []))
    return became

def req_collect_links(ep, became: dict[int, int]) -> None:
    """Part-device pairs and pointer requirements."""
    number_of = lambda raw: became.get(whole(raw) or 0, 0)
    for entry in as_list(ep.brief.get('fits')):
        if not isinstance(entry, dict):
            continue
        pair = (number_of(entry.get('item')), number_of(entry.get('with')))
        if pair[0] != pair[1] and all(0 < n <= len(ep.items) for n in pair) and pair not in ep.bound:
            ep.bound.append(pair)
    musts = [(n, m) for n, item in enumerate(ep.items, 1) for m in item.musts]
    item_roots = [roots(item.name) for item in ep.items]
    names_other = lambda number, value: any(words and words <= roots(value) for n, words in enumerate(item_roots, 1) if n != number)
    # the same test on words without possessive endings or punctuation: "unless the floor lamp's colour is white"
    # names the floor lamp (Arena r17 b11: 12 tasks carry such a clause, ours 0/12; read as a chain, it spent TRAIL
    # rounds, searches and checkout holds hunting for a listing that is the agent's own pick)
    plain_names = [(n, req_words(item.name)) for n, item in enumerate(ep.items, 1)]
    names_another = lambda number, value: any(n != number and words and words <= req_words(value) for n, words in plain_names)
    lookups = [str(x).strip() for x in as_list(ep.brief.get('lookups'))]
    for text in (line for line in lookups if line):
        head = plain_key(text.split(':')[0])
        owner = next(iter([pair for pair in musts if pair[1] == text] or [pair for pair in musts if plain_key(pair[1].split(':')[0]) == head]), (1, text))
        value = text.partition(':')[2]
        # a value that depends on what is picked for ANOTHER requested item and names no shop or listing is a
        # condition inside this order, not a page to find: it stays a requirement and no chain is held for it
        # (Arena r15: such a hold spent three checkouts)
        if names_another(owner[0], value) and not re.search('\\b[A-Z][a-z]', value):
            ep.log('condition_not_lookup', need=text[:120])
        elif owner not in ep.refs:
            ep.refs.append(owner)
    # a must whose own value describes another listing by a capitalised shop or maker name ("same as the X of the
    # Name's <kind> listing") is a pointer even when the brief left it out of its lookups (Arena r15: 1 of 33
    # pointer tasks per full run had no lookup, and that episode never opened the pointed page); a value naming
    # ANOTHER requested item is a fit or condition inside this order, not a page to find
    for pair in musts:
        value = pair[1].partition(':')[2]
        if pair in ep.refs or names_other(pair[0], value):
            continue
        if re.search('\\b(?:listing|page)\\b', value, re.I) and re.search('\\b[A-Z][A-Za-z]{3,}', value):
            ep.refs.append(pair)
            ep.log('pointer_from_must', need=pair[1][:120])
    req_untangle(ep)

def req_collect_limits(ep) -> None:
    """Budget, caps and the required process steps."""
    brief = ep.brief
    budget = brief.get('budget')
    if isinstance(budget, str) and budget.strip().replace('.', '', 1).isdecimal():
        budget = float(budget)
    if is_number(budget) and budget > 0:
        ep.budget = budget
    cap, steps = whole(brief.get('messages')), whole(brief.get('steps'))
    if cap is not None and cap >= 0:
        ep.msg_cap = cap
        ep.cap_stated = True
    elif ep.roles.get('ask'):
        ep.msg_cap = 3
    if steps is not None and 10 <= steps < ep.max_steps:
        ep.max_steps = steps
    stage = lambda raw: raw if raw in ('choose', 'order', 'end') else 'order'
    listed = ((e, str(e.get('step') or '').strip()) for e in as_list(brief.get('process')) if isinstance(e, dict))
    ep.process += [{'step': step, 'when': stage(str(e.get('when') or '').strip().lower()), 'messages': max(0, whole(e.get('messages')) or 0)}
                   for e, step in listed if step]

def req_close_reserve(ep, profile: SizeProfile) -> int:
    """Messages reserved for closing steps."""
    if not ep.roles.get('ask'):
        return 0
    at_order, at_end = (max((p['messages'] for p in ep.process if p['when'] == stage), default=0) for stage in ('order', 'end'))
    held = at_end if at_end > at_order and profile.end_room(ep.msg_cap, at_end) else at_order
    return min(held, ep.msg_cap)

def req_shared_shelf(brief: dict[str, Any]) -> bool:
    labels = (plain_key(str(e.get('shelf') or '')) for e in as_list(brief.get('items')) if isinstance(e, dict))
    shelves = [s for s in labels if s]
    return len(set(shelves)) < len(shelves)

def req_untangle(ep) -> None:
    """Fold a pointer's description back into it."""
    for number, item in enumerate(ep.items, 1):
        pointers = [need for n, need in ep.refs if n == number and need in item.musts]
        rest = [m for m in item.musts if m not in pointers]
        if not (pointers and rest):
            continue
        prompt = (UNMIX + '\n\nREQUEST:\n' + ep.request + f'\n\nPRODUCT: {item.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n'
                  + '\n'.join(pointers) + '\nOTHER REQUIREMENTS:\n' + '\n'.join(rest))
        named = {str(x).strip() for x in as_list((ep.llm.ask(prompt, [HEAVY, LIGHT], 1500) or {}).get('describe_other'))}
        moved = [m for m in rest if m in named]
        if moved:
            lead = pointers[0]
            joined = lead + ' (that listing: ' + '; '.join(moved) + ')'
            item.musts = [m if m != lead else joined for m in item.musts if m not in moved]
            ep.refs = [(n, need if (n, need) != (number, lead) else joined) for n, need in ep.refs]
            ep.log('unmix', item=number, moved=moved)

def req_is_description(ep, number: int, need: str) -> bool:
    """Shopper's own wording no read key carries."""
    if any(need == m for _, m in ep.refs) or need in [str(x) for x in as_list(ep.brief.get('lookups'))]:
        return False
    prop, _, value = need.partition(':')
    if not prop.strip() or len(re.findall('[a-z0-9]+', value.lower())) < 4:
        return False
    if re.search('\\b(?:other than|unlike)\\b', value, re.I):
        return False
    item = ep.items[number - 1]
    # a likeness to something no listing was found for (a request requirement, not a lookup) is the shopper's
    # own description when no listing states that property
    likened = need in item.musts and need not in item.added and re.search('\\bsame as\\b', value, re.I)
    if not likened and re.search('\\b(?:listing|page|shown|same as)\\b', value, re.I):
        return False
    value_stems = stems(value)
    for other in ep.items:
        if other is not item and any(len(w) >= 4 and w in value_stems for w in stems(other.name)):
            return False
    cat_key = mem_kind_key(ep)
    blocks = []
    for record in ep.mem.pages.values():
        if cat_key and not mem_belongs(ep, item, str(record.get(cat_key) or '')):
            continue
        block = fact_block(record)
        if block:
            blocks.append(block)
    if len(blocks) < 3:
        return False
    prop_stems = stems(prop)
    return not any(stems(k) & prop_stems for k in {plain_key(k) for block in blocks for k in block})

def req_ledger(ep) -> str:
    """Requirements in force, as the model reads them."""
    many = len(ep.items) > 1
    lines = [req_item_line(ep, n, item, many) for n, item in enumerate(ep.items, 1)] + req_terms_lines(ep)
    return '\n'.join('- ' + line for line in lines)

def req_item_line(ep, number: int, item: ItemSpec, many: bool) -> str:
    live = item.live()
    own_words = [m for m in live if req_is_description(ep, number, m)]
    sourced = lambda m: m if m not in ep.ref_values else m + f' (value from the page it points at: {ep.ref_values[m]})'
    needs = '; '.join(sourced(m) for m in live if m not in own_words) or 'nothing beyond being this kind of product'
    label = f' (store label: {item.shelf})' if item.shelf else ''
    lead = f'item {number}: ' if many else 'item: '
    extras = (("the shopper's own description, stated by no listing attribute (it decides nothing): ", own_words, '; '),
              ('no preference on: ', item.free, ', '))
    parts = [f'{lead}{item.name}{label}, quantity {item.qty}', 'must have: ' + needs]
    return '; '.join(parts + [title + sep.join(values) for title, values, sep in extras if values])

def req_terms_lines(ep) -> list[str]:
    brief = ep.brief
    pick = brief.get('pick')
    scope = 'for the whole order' if brief.get('budget_scope') != 'each' else 'per item'
    measured = f"pick: among fitting listings, the {pick} {brief.get('pick_measure') or ''}".replace('  ', ' ').rstrip()
    plain_pick = 'pick: the lowest-priced listing that fits' if pick == 'cheapest' else 'pick: any listing that meets everything'
    head = [(ep.budget is not None, f"budget: no more than {ep.budget} {brief.get('budget_unit') or ''} {scope}".replace('  ', ' ')),
            (True, measured if pick in ('most', 'least') else plain_pick),
            (brief.get('in_stock_only', True) is not False, 'only listings in stock')]
    lines = [text for keep, text in head if keep]
    lines += [f'wish, only to break ties: {w}' for w in as_list(brief.get('wishes'))] + [f'request rule: {r}' for r in as_list(brief.get('rules'))]
    if brief.get('quiet_decline') is True:
        lines.append('when nothing fits: end naming or suggesting no listing')
    stages = {'choose': 'before choosing', 'order': 'at the order (Desk does it when you check out)', 'end': 'at an unbought ending (Desk does it then)'}
    lines += [f"required step, {stages[p['when']]}: {p['step']}" for p in ep.process]
    facts = ep.attr_tool()
    if facts and (ep.fact_tools.get(facts) is True or ep.roles.get('offer')):
        lines.append(req_attr_line(ep, facts))
    return lines

def req_attr_line(ep, facts: str) -> str:
    bare = sorted(t for t in ep.page_tools() | {ep.roles.get('offer') or ''} if t and ep.fact_tools.get(t) is False)
    pages = sorted(t for t in ep.page_tools() if t != facts)
    notes = ([f"; {', '.join(bare)} show no attributes"] if bare else []) + (
        [f"; {' or '.join(pages)} shows a listing's page (what a chain of pages or a page check refers to)"] if pages else [])
    return f'attributes are shown by {facts}' + ''.join(notes)

class ListingMemory:
    """Rows and read pages of every listing seen."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.pages: dict[str, dict[str, Any]] = {}
        self.order: list[str] = []
        self.id_key: str | None = None
        self.price_key: str | None = None
        self.stock_key: str | None = None
        self.tool_keys: dict[str, set[str]] = {}
        self.view_keys: dict[str, tuple[int, set[str]]] = {}
        self.catch_stamp: tuple[int, int] | None = None
        self.catch_keys: set[str] = set()

def mem_identify(ep, node: Any, in_list: bool = False, depth: int = 0) -> str | None:
    if not isinstance(node, dict) or len(node) < 3:
        return None
    known = ep.mem.id_key
    if known and id_like(node.get(known), True):
        return node[known].strip()
    nested = any(isinstance(v, dict) for v in node.values())
    found = guess_id_key([node]) if in_list or depth == 0 or nested else None
    if not found:
        return None
    return node[found].strip() if 'id' in str(found).lower() or node[found].strip() in ep.mem.rows else None

def mem_rows_keyed(ep, node: list[Any]) -> bool:
    rows = [n for n in node if isinstance(n, dict) and len(n) >= 3]
    if not rows or 2 * len(rows) < len(node):
        return False
    known = ep.mem.id_key
    if known and all(id_like(r.get(known), True) for r in rows):
        return True
    found = guess_id_key(rows)
    ep.mem.id_key = found or ep.mem.id_key
    return bool(found)

def mem_keep_row(ep, ident: str, node: dict[str, Any]) -> None:
    """Merge one listing row, tracking stock flips and read pages."""
    mem = ep.mem
    if ident not in mem.rows:
        mem.order.append(ident)
        mem.rows[ident] = {}
    old = mem.rows[ident]
    flips = lambda was, now: any(old.get(k) is was and v is now for k, v in node.items())
    if flips(True, False):
        ep.unavailable.add(ident)
    elif flips(False, True):
        ep.unavailable.discard(ident)
    old.update(node)
    if fact_block(node):
        mem.pages[ident] = dict(old)
    elif ident in mem.pages:
        mem.pages[ident].update(node)

def mem_walk(ep, node: Any, depth: int, in_list: bool = False) -> None:
    if depth > 6:
        return
    if isinstance(node, dict):
        ident = mem_identify(ep, node, in_list, depth)
        if ident:
            mem_keep_row(ep, ident, node)
            return
        for value in node.values():
            mem_walk(ep, value, depth + 1)
    elif isinstance(node, list):
        keyed = mem_rows_keyed(ep, node)
        for value in node:
            mem_walk(ep, value, depth + 1, keyed)

def mem_absorb(ep, observation: Any) -> None:
    mem_walk(ep, observation, 0)

def mem_enum_labels(ep) -> set[str]:
    found: set[str] = set()
    pending = [(schema, 0) for schema in ep.schemas.values()]
    while pending:
        schema, depth = pending.pop()
        if depth <= 6 and isinstance(schema, dict):
            found.update(o.strip().lower() for o in schema.get('enum') or [] if isinstance(o, str))
            pending += [(v, depth + 1) for v in schema.values() if isinstance(v, dict)]
    return found

def mem_kind_key(ep) -> str | None:
    rows = ep.mem.rows
    seen: dict[str, list[str]] = {}
    for key, value in ((k, v) for row in rows.values() for k, v in row.items()):
        if isinstance(value, str) and 0 < len(value) <= 40 and not re.search('\\d{3}', value):
            seen.setdefault(key, []).append(value)
    total = max(1, len(rows))
    known = mem_enum_labels(ep)
    scored = []
    for key, texts in seen.items():
        values = set(texts)
        if len(texts) >= 0.8 * total and len(values) <= 40:
            listed = len([v for v in values if v.strip().lower() in known]) / len(values)
            scored.append(((-listed, len(values), -len(texts), -sum(map(len, values)) / len(values)), key))
    best = min(scored, key=lambda pair: pair[0], default=None)
    return None if best is None else best[1]

def mem_label(ep, ident: str, key: str | None) -> str:
    return str((ep.mem.rows.get(ident) or {}).get(key) or '')

def mem_is_catch_all(ep, key: str) -> bool:
    """Key whose values look like feature bundles."""
    stamp = (len(ep.mem.pages), ep.version)
    if ep.mem.catch_stamp != stamp:
        ep.mem.catch_stamp = stamp
        ep.mem.catch_keys = {plain_key(name) for _, name, _, values in mem_key_table(ep) if listy(values)}
    return plain_key(key) in ep.mem.catch_keys

def mem_key_table(ep) -> list[tuple[str, str, int, list[str]]]:
    """Planner rows: (category, key, carriers, samples)."""
    table: list[tuple[str, str, int, list[str]]] = []
    groups = mem_wanted_groups(ep, mem_key_groups(ep))
    for label, group in sorted(groups.items(), key=lambda kv: -kv[1]['n']):
        kept = mem_kept_keys(group)
        head, tail = kept[:TABLE_FIRST], kept[TABLE_FIRST:]
        # a need can sit on a key few listings carry: past the first rows every key seen on two or more stays
        later = [(name, slot) for name, slot in tail if slot['n'] >= 2 and not re.search(TABLE_SKIP, name)] if P2_ON else []
        table.extend((label, name, slot['n'], [v[:40] for v in slot['values'][:4]]) for name, slot in head + later)
    return table

def mem_key_groups(ep) -> dict[str, dict[str, Any]]:
    """Per category: pages read and keys with values."""
    cat_key = mem_kind_key(ep)
    groups: dict[str, dict[str, Any]] = {}
    read = [(ident, ep.mem.pages.get(ident)) for ident in ep.mem.order]
    for ident, record in ((i, r) for i, r in read if r):
        label = str(record.get(cat_key) or 'listings') if cat_key else 'listings'
        group = groups.setdefault(label, {'n': 0, 'keys': {}})
        group['n'] += 1
        first: dict[str, Any] = {}
        for key, value in (fact_block(record) or {}).items():
            first.setdefault(plain_key(key), value)
        for name, value in ((n, v) for n, v in first.items() if n):
            text = str(value).strip().lower()
            slot = group['keys'].setdefault(name, {'n': 0, 'values': [], 'per': {}})
            slot['n'] += 1
            slot['per'][ident] = text
            if text not in slot['values']:
                slot['values'].append(text)
    return groups

def mem_asked_label(ep, label: str) -> bool:
    """Category label of a requested kind."""
    long = lambda text: {w for w in stems(text) if len(w) >= 4}
    key, words = plain_key(label), long(label)
    named = lambda item: key in (plain_key(item.shelf or ''), plain_key(item.name)) or words & (long(item.shelf or '') | long(item.name))
    return any(named(item) for item in ep.items) or bool(mem_owner(ep, label))

def mem_wanted_groups(ep, groups: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Groups of requested kinds (all if none match)."""
    matched = [label for label in groups if mem_asked_label(ep, label)]
    return {label: groups[label] for label in matched} if matched else groups

def mem_kept_keys(group: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    kept: list[tuple[str, dict[str, Any]]] = []
    ranked = sorted(group['keys'].items(), key=lambda kv: (-kv[1]['n'], len(kv[0])))
    for name, slot in ((n, s) for n, s in ranked if not free_text(list(s['per'].values()))):
        twin = next((spec for other, spec in kept if mem_twins(other, spec, name, slot)), None)
        if twin is None:
            kept.append((name, slot))
        else:
            twin['per'].update({ident: text for ident, text in slot['per'].items() if ident not in twin['per']})
            twin['values'].extend(dict.fromkeys(text for text in slot['per'].values() if text not in twin['values']))
            twin['n'] = len(twin['per'])
    kept.sort(key=lambda kv: -kv[1]['n'])
    return kept

def mem_stated_clash(ep, label: str, name: str, values: list[str]) -> str:
    """Mark keys that are not a stated property."""
    number = mem_owner(ep, label)
    if not number:
        return ''
    words = lambda text: set(re.findall('[a-z0-9]{3,}', text.lower()))
    shown = words(' '.join(values))
    parts = (must.partition(':') for must in ep.items[number - 1].musts)
    clashing = (prop.strip() for prop, _, value in parts if not has_bound(value) and prop.strip() and plain_key(prop) != name
                and stems(prop) & stems(name) and words(value) and not words(value) <= shown)
    prop = next(clashing, None)
    return '' if prop is None else f' [not the stated {prop}: another property]'

def mem_twins(name: str, spec: dict[str, Any], other: str, slot: dict[str, Any]) -> bool:
    a, b = set(name.split()), set(other.split())
    if not (a and b and a & b):
        return False

    def agree(x: str, y: str) -> bool:
        if x == y:
            return True
        short, longer = sorted((re.findall('[a-z0-9]+', t.lower()) for t in (x, y)), key=len)
        return bool(short) and set(short) <= set(longer) and len(longer) <= 2 * len(short) + 2
    mine, theirs = spec['per'], slot['per']
    both = [i for i in mine if i in theirs and mine[i].strip() and theirs[i].strip()]
    agreeing = [i for i in both if agree(mine[i], theirs[i])]
    if a <= b or b <= a:
        return 2 * len(agreeing) > len(both) if both else bool((set(spec['values']) & set(slot['values'])) - {''})
    return len(both) >= 2 and len(agreeing) == len(both)

def mem_kind_pools(ep) -> list[list[str]]:
    """Seen ids per requested kind, largest first."""
    cat_key = mem_kind_key(ep)
    groups: dict[str, list[str]] = {}
    for ident in ep.mem.order:
        groups.setdefault(str(ep.mem.rows[ident].get(cat_key) or '') if cat_key else '', []).append(ident)
    matched = [ids for label, ids in groups.items() if mem_asked_label(ep, label)]
    return sorted(matched or list(groups.values()), key=lambda ids: -len(ids))[:max(1, len(ep.items))]

def mem_price(ep, ident: str) -> float | None:
    key = ep.mem.price_key
    value = (ep.mem.rows.get(ident) or {}).get(key) if key else None
    if isinstance(value, dict):
        value = next((v for v in value.values() if is_number(v)), None)
    elif isinstance(value, str):
        value = float(value) if re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', value) else None
    return float(value) if is_number(value) else None

def mem_can_buy(ep, ident: str) -> bool:
    if ep.is_gone(ident):
        return False
    value = (ep.mem.rows.get(ident) or {}).get(ep.mem.stock_key) if ep.mem.stock_key else None
    return not (value is False or (is_number(value) and value <= 0))

def mem_listings_in(ep, observation: Any) -> list[tuple[str, dict[str, Any]]]:
    keyed = lambda rows: [(i, r) for r in rows if isinstance(r, dict) for i in [mem_identify(ep, r, True)] if i]
    if isinstance(observation, list):
        return keyed(observation)
    if not isinstance(observation, dict):
        return []
    ident = mem_identify(ep, observation)
    if ident:
        return [(ident, observation)]
    spread_out = lambda value: [value] if isinstance(value, list) else list(value.values()) if isinstance(value, dict) else []
    lists = [inner for value in observation.values() for inner in spread_out(value) if isinstance(inner, list)]
    return [pair for rows in lists for pair in keyed(rows)]

def mem_note_keys(ep, name: str, observation: Any) -> None:
    for ident, record in mem_listings_in(ep, observation):
        keys = set(record)
        ep.mem.tool_keys.setdefault(name, set()).update(keys)
        seen_at, have = ep.mem.view_keys.get(ident, (-1, set()))
        ep.mem.view_keys[ident] = (ep.version, have | keys if seen_at == ep.version else set(keys))

def mem_fully_shown(ep, name: str, ident: str) -> bool:
    seen_at, have = ep.mem.view_keys.get(ident, (-1, set()))
    keys = ep.mem.tool_keys.get(name)
    return bool(keys) and seen_at == ep.version and keys <= have and ident not in ep.gone

def mem_owner(ep, label: str) -> int:
    """Item number a category label belongs to (0: none)."""
    if not label:
        return 0
    words, long = plain_key(label), roots(label)
    share = lambda mine: len(long & mine) / max(1, len(long | mine))
    scores = [3.0 if words == plain_key(item.shelf or '') else 2.0 if words == plain_key(item.name)
              else share(roots(item.shelf or '') | roots(item.name)) for item in ep.items]
    top = max(scores, default=0.0)
    return scores.index(top) + 1 if top > 0 else 0

def mem_belongs(ep, item: ItemSpec, label: str) -> bool:
    items = ep.items
    return bool(label) and item in items and mem_owner(ep, label) == items.index(item) + 1

def mem_read_count(ep, number: int) -> int:
    cat_key = mem_kind_key(ep)
    item = ep.items[number - 1]
    return sum(1 for record in ep.mem.pages.values() if cat_key and mem_belongs(ep, item, str(record.get(cat_key) or '')) and fact_block(record))

def mem_ids_of_item(ep, number: int) -> list[str]:
    cat_key = mem_kind_key(ep)
    item = ep.items[number - 1]
    return [i for i in ep.mem.order if cat_key and mem_belongs(ep, item, mem_label(ep, i, cat_key))]

def mem_record_text(ep, ident: str, review: bool = False) -> str:
    record = ep.mem.pages.get(ident) or ep.mem.rows.get(ident)
    if not record:
        return '(no record seen)'
    return dumps(clip(as_stated(record) if review else record, 400))[:3000]

def mem_other_records(ep, skip: list[str], limit: int = 30) -> str:
    order = ep.mem.order[::-1]
    if ep.cheapest():
        order.sort(key=lambda i: (i not in ep.mem.pages, ep.price_rank(i)))
    chosen = [i for i in order if i not in skip and not ep.is_gone(i)][:max(1, limit)]
    return '\n'.join(dumps(clip(as_stated(ep.mem.pages.get(i) or ep.mem.rows.get(i)), 160))[:900] for i in chosen)

class SearchScout:
    """Searching and paging done on the desk's own."""

    def __init__(self) -> None:
        self.word_rule: str | None = None
        self.cheaper_marks: set[str] = set()
        self.capped_done = self.capped_more = self.wider_done = False
        self.pages_read = 0
        (self.rule_log, self.search_log, self.searched, self.widened,
         self.extra_pages, self.kind_calls, self.cap_calls) = ([] for _ in range(7))
        self.shapes: dict[str, dict[str, Any] | None] = {}
        self.first_reads = 0
        # paging learned in code from results: tool -> (arg that moves through the rows, result key of the next value)
        self.page_ways: dict[str, tuple[str, str | None]] = {}
        # next values the store handed back, per tool and query (the arguments without the paging one)
        self.next_given: dict[str, list[Any]] = {}
        self.loose_notes = 0

def scout_values_shown(ep, ident: str, needs: list[str]) -> int:
    """Required plain values (not limits) whose words all appear in a listing's row or page."""
    words = set(plain_words(dumps(ep.mem.pages.get(ident) or ep.mem.rows.get(ident) or {})))
    wanted = [set(plain_words(n.split(':', 1)[1])) for n in needs if ':' in n and not has_bound(n.split(':', 1)[1])]
    return sum(1 for w in wanted if w and w <= words)

def scout_open_items(ep) -> list[int]:
    """Item numbers with no cart line yet."""
    cat_key = mem_kind_key(ep)
    held = {mem_owner(ep, mem_label(ep, i, cat_key)) for i in basket_ids(ep)} if cat_key else set()
    return [n for n in range(1, len(ep.items) + 1) if n not in held]

def scout_first_reads(ep, seen: Any) -> str:
    """After a search of the model, open the most promising unread rows of items still without a cart line, in one go."""
    # Arena noask b10: the model asked for 30 store calls per episode against the king's 20 at the same step count (the
    # king's own code made about half its store calls), and 158 of 393 episodes ran past 255 s; a Desk read costs a step
    # but no model turn. Not before the interview starts, so the question timing stays as it is
    scout = ep.scout
    if (ep.roles.get('ask') and not ep.talk.started) or scout.first_reads >= FIRST_READS or ep.past_stop():
        return ''
    cat_key = mem_kind_key(ep)
    missing = scout_open_items(ep)
    room = min(ep.steps_left(), ep.clock_room()) - ep.reserve()
    if not cat_key or not missing or room < 12 + 4 * len(missing):
        return ''
    cap = ep.numeric_budget()
    picks: list[str] = []
    for number in missing:
        item = ep.items[number - 1]
        needs = item.live()
        rows = [i for i, _ in mem_listings_in(ep, seen) if i not in ep.mem.pages and mem_can_buy(ep, i) and not ep.is_gone(i)
                and mem_belongs(ep, item, mem_label(ep, i, cat_key)) and (cap is None or (mem_price(ep, i) or 0) <= cap)]
        cheap = ep.cheapest()
        ranked = sorted(rows, key=lambda i: (-scout_values_shown(ep, i, needs), ep.price_rank(i) if cheap else 0))
        picks += [i for i in ranked if cheap or scout_values_shown(ep, i, needs) > 0][:2]
    picks = list(dict.fromkeys(picks))[:max(0, FIRST_READS - scout.first_reads)][:2]
    if not picks:
        return ''
    texts = scout_read_pages(ep, picks, ep.reserve() + 10)
    scout.first_reads += len(texts)
    if not texts:
        return ''
    ep.log('first_reads', ids=picks[:len(texts)])
    return ('\n[Desk opened the most promising unread rows of this result for items without a cart line (one step each)]\n'
            + '\n'.join(texts))

def scout_paging_arg(ep, tool: str) -> Any:
    return (ep.scout.shapes.get(tool) or {}).get('next_arg') or (ep.scout.page_ways.get(tool) or (None, None))[0]

def scout_next_key(ep, tool: str) -> Any:
    """Result key holding the next page's value."""
    shape = ep.scout.shapes.get(tool) or {}
    return shape.get('next') if shape.get('next_arg') else (ep.scout.page_ways.get(tool) or (None, None))[1]

def scout_loose_numbers(ep, tool: str) -> list[str]:
    """Optional number arguments a tool's schema leaves unexplained: no description, no choices, not tied to price."""
    schema = ep.schemas.get(tool) or {}
    props = schema.get('properties') or {}
    required = schema.get('required') or []
    priced = lambda key: re.search('price|cost|amount|budget|spend|money', key, re.I)
    # a number named as a cap or limit bounds the rows (real r13 listing filter: a cap beside a start row); never a start
    return [k for k, spec in props.items() if isinstance(spec, dict) and k not in required and 'enum' not in spec
            and {'integer', 'number'} & set(type_names(spec)) and not str(spec.get('description') or '').strip() and not priced(k)
            and not re.search(LIMIT_ARG, k, re.I)]

def scout_learn_paging(ep, name: str, args: dict[str, Any], observation: Any) -> str:
    """Learn from a search result which unexplained number moves through its rows (a start position). Evidence is
    structural: a next value equal to the start sent plus the rows shown, no rows on a search whose match count lies
    at or below the start sent, or (nothing sent, one such argument) a next value equal to the rows shown while more
    matched. Real race 17: episodes whose new queries carried a start position passed .16 against .37 (all agents)."""
    if name != ep.roles.get('search') or not isinstance(observation, dict) or scout_paging_arg(ep, name):
        return ''
    loose = scout_loose_numbers(ep, name)
    lists = [k for k, v in observation.items() if isinstance(v, list)]
    if not loose or len(lists) != 1:
        return ''
    shown = len(observation[lists[0]])
    ints = {k: v for k, v in observation.items() if isinstance(v, int) and not isinstance(v, bool)}
    sent = {k: args[k] for k in loose if isinstance(args.get(k), int) and not isinstance(args.get(k), bool) and args[k] > 0}
    found = None
    for key, start in sent.items():
        nxt = [k for k, v in ints.items() if shown and v == start + shown]
        if len(nxt) == 1:
            found = (key, nxt[0], start)
        elif not shown and any(0 < v <= start for v in ints.values()):
            found = (key, None, start)
        if found:
            break
    if not found and not sent and len(loose) == 1 and shown:
        nxt = [k for k, v in ints.items() if v == shown]
        if len(nxt) == 1 and any(v > shown for v in ints.values()):
            found = (loose[0], nxt[0], 0)
    if not found:
        return ''
    key, nxt, start = found
    ep.scout.page_ways[name] = (key, nxt)
    ep.log('paging_learned', tool=name, arg=key, next=nxt, start=start)
    if not start:
        return ''
    return (f"\n[Desk] {key} is the start position in this search's results: this page began after row {start}, so the rows "
            f'before it were not shown. A new query starts at its first row with {key} left out; page on only with the value the store '
            'hands back.')

def scout_note_next(ep, name: str, args: dict[str, Any], observation: Any) -> None:
    """Keep the next value a search handed back, per query."""
    paging, key = scout_paging_arg(ep, name), scout_next_key(ep, name)
    if not paging or not key or not isinstance(observation, dict) or key not in observation:
        return
    base = name + dumps({k: v for k, v in args.items() if k != paging})
    ep.scout.next_given[base] = (ep.scout.next_given.get(base, []) + [observation[key]])[-6:]

def scout_reopen_cheapest(ep) -> None:
    """Reopen the cheaper-fit hunt after a change."""
    ep.scout.wider_done = False
    ep.scout.cheaper_marks = set()

def scout_shape_of(got: dict[str, Any], observation: dict[str, Any], lists: list[str], params: dict[str, Any]) -> dict[str, Any] | None:
    """The result shape a SHAPE reply names, each named key checked where it has to occur."""
    rows = got.get('rows')
    if not rows or rows not in lists:
        return None
    first = observation[rows][0]
    shape: dict[str, Any] = {'rows': rows}
    for part, where in zip(('count', 'next', 'next_arg', 'price', 'stock'), (observation, observation, params, first, first)):
        said = got.get(part)
        shape[part] = said if isinstance(said, str) and said in where else None
    total = None if not shape['count'] else observation.get(shape['count'])
    if not isinstance(total, int) or isinstance(total, bool):
        shape['count'] = None
    return shape

def scout_learn_shape(ep, name: str, observation: Any) -> dict[str, Any] | None:
    """Learn a list result's keys once per tool."""
    shapes = ep.scout.shapes
    if name in shapes:
        return shapes[name]
    lists = []
    if isinstance(observation, dict):
        for k, v in observation.items():
            if isinstance(v, list) and len(v) > 1 and all(isinstance(r, dict) for r in v):
                lists.append(k)
    if not lists:
        return None
    shapes[name] = None
    sample = {k: v[:3] if k in lists else v for k, v in observation.items()}
    schema = ep.schemas.get(name) or {}
    prompt = ''.join([SHAPE, '\n\nTOOL: ', name, ' | parameters: ', dumps(schema), '\n\nRESULT (first rows only):\n',
                      dumps(clip(sample, 120))[:4000]])
    got = ep.llm.ask(prompt, [LIGHT, HEAVY], 2500) or {}
    shape = scout_shape_of(got, observation, lists, schema.get('properties') or {})
    if shape is not None:
        shapes[name] = shape
        mem = ep.mem
        mem.price_key = mem.price_key or shape['price']
        mem.stock_key = mem.stock_key or shape['stock']
        ep.log('shape', tool=name, shape=shape)
    return shape

def scout_page_plan(ep, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
    """(matched, shown, paging arg, next value) or None."""
    shape = scout_learn_shape(ep, name, observation)
    if not shape or not isinstance(observation, dict):
        return None
    if not (shape['count'] and shape['next'] and shape['next_arg']):
        return None
    rows = observation.get(shape['rows'])
    count = observation.get(shape['count'])
    value = observation.get(shape['next'])
    integral = lambda v: not isinstance(v, bool) and isinstance(v, int)
    if not isinstance(rows, list) or not rows or not integral(count):
        return None
    if value is None or integral(value) or isinstance(value, str):
        return count, len(rows), shape['next_arg'], value
    return None

def scout_price_sorted(ep, observation: Any) -> bool:
    prices = [mem_price(ep, i) for i, _ in mem_listings_in(ep, observation)]
    return len(prices) >= 3 and None not in prices and all(a <= b for a, b in zip(prices, prices[1:]))

def scout_auto_pages(ep, name: str, args: dict[str, Any], observation: Any) -> str:
    """Read more pages for extreme picks, or advise."""
    ep.scout.extra_pages = []
    pick = ep.brief.get('pick')
    extreme = pick in ('cheapest', 'most', 'least')
    pointed = False if extreme else scout_names_pointed_seller(ep, name, args)
    plan = None
    if extreme or pointed:
        plan = scout_page_plan(ep, name, observation)
    scout_remember_kind_search(ep, name, args, observation)
    if extreme:
        due = scout_pages_due(ep, plan, args, observation, pick, worded(scout_search_words(ep, name, args)))
    else:
        due = scout_pointed_pages(plan, args)
    if not due or isinstance(due, str):
        return due or ''
    count, shown, param, value = plan
    texts, value = scout_read_more(ep, name, args, param, value, due)
    ep.log('pages', tool=name, read=len(texts))
    if not texts:
        return ''
    out = '\n[Desk read the following pages of this search itself, one step per page]\n' + '\n'.join(texts)
    if value is not None:
        out += (f"\n[Desk] {count} listings matched and Desk has now read {shown * (len(texts) + 1)} rows of them; "
                'search again with other words, or page on for the rest.')
    return out

def scout_names_pointed_seller(ep, name: str, args: dict[str, Any]) -> bool:
    """Whether a search's words hold a capitalised name that a pointer requirement uses (the shop or maker of the
    listing the request points at)."""
    texts = ' '.join([need for _, need in ep.refs] + [str(x) for x in as_list(ep.brief.get('lookups'))])
    names = {w for w in re.findall('[A-Z][A-Za-z]{3,}', texts)}
    return bool(names) and any(w in names for w in re.findall('[A-Z][A-Za-z]{3,}', scout_search_words(ep, name, args)))

def scout_pointed_pages(plan: Any, args: dict[str, Any]) -> int:
    """How many more pages to read after a first page that searched a pointed listing's shop; every one if two or
    fewer are left. Real race 15: the pointed listing was on page 2 of an 8-row shop search; 21 of 22 passers read and
    opened it, and 0 of 56 agents that missed it passed."""
    if not plan:
        return 0
    count, shown, param, value = plan
    if value is None or shown <= 0 or count <= shown or args.get(param) not in (None, '', 0, 1):
        return 0
    rest = -(-(count - shown) // shown)
    return 0 if rest > 2 else rest

def scout_search_words(ep, name: str, args: dict[str, Any]) -> str:
    """Text arguments that are not labels or choices."""
    props = (ep.schemas.get(name) or {}).get('properties') or {}
    cat_key = mem_kind_key(ep)
    labels = set()
    if cat_key:
        labels = {str(r.get(cat_key) or '').strip().lower() for r in ep.mem.rows.values()}
        labels.discard('')
    kept = []
    for key, value in args.items():
        if not isinstance(value, str) or not value.strip():
            continue
        spec = props.get(key)
        if isinstance(spec, dict) and 'enum' in spec:
            continue
        if value.strip().lower() not in labels:
            kept.append(value)
    return ' '.join(kept)

def scout_pages_due(ep, plan: Any, args: dict[str, Any], observation: Any, pick: Any, has_words: bool = True) -> int | str:
    """Pages to read here, or a note instead."""
    if not plan:
        return ''
    count, shown, param, value = plan
    given = args.get(param)
    if isinstance(given, int) and not isinstance(given, bool):
        later = given not in (None, 0, 1)
    else:
        later = bool(str(given or '').strip())
    if value is None or later or shown >= count:
        return ''
    if pick == 'cheapest' and scout_price_sorted(ep, observation):
        return ''
    # cheapest picks, real races 13-15: a later page of a category listing or of a number-led search rarely showed
    # a paying listing (0.4-4.7%, r15 .008), a later page of a worded search 4-9% (r15 short results of at most
    # 5 pages .23, a new search's first page .16). Searches led by a number and a noun count as worded: r15 later
    # pages of results up to 56 rows showed a paying listing .356 of calls for them, .145 for plain words, and 30
    # of 49 first sights of a paid listing by passers came on a later page (one task: page 3 of 54 rows for all)
    rest = (count - 1) // shown
    # a store reaching 7 pages of 7 rows leaves at most 6 to read, whatever count it reports
    pages = 0 if not has_words else (1 if rest > 7 else min(rest, 6))
    cap = ep.page_cap()
    # the page budget is shared by the items: one item's search may not use up another's
    if pages:
        pages = min(pages, max(1, cap // max(1, len(ep.items))))
    if pages and ep.scout.pages_read + pages <= cap and ep.steps_left() - ep.reserve() - pages >= 12:
        return pages
    return (f"\n[Desk] Shown: {shown} of the {count} listings this search matched, in no price order; every fitting "
            "one counts for the pick. Fresh words (a synonym, the product's plain name, title words of fitting listings seen so far) turn up "
            f"fitting listings far more often than extra pages; when no fresh words remain, go on paging with {param}={dumps(value)}.")

def scout_read_more(ep, name: str, args: dict[str, Any], param: str, value: Any, pages: int) -> tuple[list[str], Any]:
    texts: list[str] = []
    schema = ep.schemas.get(name)
    while len(texts) < pages and value is not None:
        call = shaped(dict(args, **{param: value}), schema)
        reply = None if call is None else ep.store.call(name, call)
        if reply is None or ep.finished:
            break
        ep.scout.pages_read += 1
        seen = reply.get('observation')
        ep.ingest(name, seen)
        scout_note_next(ep, name, call, seen)
        ep.scout.extra_pages.append(seen)
        rendered = ep.render(seen)
        ep.cache[name + dumps(call)] = (ep.version, ep.step, rendered)
        # the first extra page is shown in full, later ones as compact rows
        body = rendered if not texts else scout_short_rows(ep, seen)
        texts.append(param + '=' + dumps(value) + ': ' + body)
        after = scout_page_plan(ep, name, seen)
        value = None if not after else after[3]
    return texts, value

def scout_short_rows(ep, observation: Any) -> str:
    """Compact one-line-per-row page text."""
    rows = []
    for ident, record in mem_listings_in(ep, observation):
        texts = [v for v in record.values() if isinstance(v, str) and v.strip() != ident and len(v.split()) >= 3]
        blurb = (texts[0] if texts else '')[:90]
        price = mem_price(ep, ident)
        flag = ' | SOLD OUT' if ep.is_gone(ident) else ''
        label = '?' if price is None else price
        rows.append((price if price is not None else float('inf'), f'{ident} | price {label}{flag} | {blurb}'))
    if not rows:
        return ep.render(observation)
    ordered = sorted(rows, key=lambda row: row[0])
    return '(compact rows, cheapest first) ' + '; '.join(text for _, text in ordered)

def scout_price_caps(args: dict[str, Any], props: dict[str, Any]) -> list[str]:
    """Numeric limit arguments the schema ties to price, lowest value first."""
    picked = []
    for key, value in args.items():
        if key not in props or not is_number(value) or not re.search(LIMIT_ARG, key, re.I):
            continue
        about = str(props[key].get('description') or '') if isinstance(props[key], dict) else ''
        if re.search('price|cost|amount|budget|spend|money', f'{key} {about}', re.I):
            picked.append(key)
    picked.sort(key=args.get)
    return picked

def scout_ignored_cap(ep, name: str, args: dict[str, Any], observation: Any) -> str:
    """Note for a price cap the search ignored."""
    props = (ep.schemas.get(name) or {}).get('properties') or {}
    caps = scout_price_caps(args, props)
    if not ep.mem.price_key or not caps:
        return ''
    head = caps[0]
    cap = args[head]
    over = len([i for i, _ in mem_listings_in(ep, observation) if cap < (mem_price(ep, i) or 0)])
    if over == 0:
        return ''
    others = {}
    for k, v in args.items():
        choices = (props.get(k) or {}).get('enum')
        if isinstance(v, str) and isinstance(choices, list):
            others[k] = [o for o in props[k]['enum'] if o != v]
    hint = ''
    if any(others.values()):
        named = ', '.join(f'{k}={dumps(rest[0])}' for k, rest in others.items() if rest)
        hint = '; to see only cheaper listings, list the category with ' + named + f' and the same {head}'
    return f'\n[Desk] {head}={cap:g} was not applied by this search: {over} of its rows cost more{hint}.'

def scout_cheapest_note(ep, observations: list[Any]) -> str:
    mem = ep.mem
    if not ep.cheapest() or not mem.price_key:
        return ''
    found = {i for seen in observations for i, _ in mem_listings_in(ep, seen)}
    rows = []
    for ident in found:
        cost = mem_price(ep, ident) if mem_can_buy(ep, ident) else None
        if cost is not None:
            rows.append((cost, ident))
    return '' if len(rows) < 2 else '\n[Desk] buyable rows here, cheapest first: ' + ', '.join(f'{i} ({p:g})' for p, i in sorted(rows)[:6])

def scout_remember_kind_search(ep, name: str, args: dict[str, Any], observation: Any) -> None:
    """Keep a kind search as a capped-listing base."""
    paging = scout_paging_arg(ep, name)
    cat_key = mem_kind_key(ep)
    rows = mem_listings_in(ep, observation)
    if not rows or not cat_key:
        return
    if paging and args.get(paging) not in (None, 0, ''):
        return
    kinds = {str(record.get(cat_key) or '').strip().lower() for _, record in rows}
    kinds.discard('')
    if not kinds or any(call[0] == name and call[1] == args for call in ep.scout.kind_calls):
        return
    ep.scout.kind_calls = [*ep.scout.kind_calls, (name, dict(args), kinds)][-12:]

def scout_cap_guesses(ep) -> list[tuple[str, str, dict[str, Any], set[str]]]:
    """Calls that may list a category under a price."""
    guesses = list(ep.scout.cap_calls)
    for name, args, kinds in ep.scout.kind_calls:
        props = (ep.schemas.get(name) or {}).get('properties') or {}
        skip = {scout_paging_arg(ep, name), *args}
        for key, spec in props.items():
            if key in skip or not isinstance(spec, dict) or not {'number', 'integer'} & set(type_names(spec)):
                continue
            if re.search(LIMIT_ARG, key, re.I) and not re.search('min|least|page|size|offset|start|row|count', key, re.I):
                guesses.append((name, key, dict(args), kinds))
    return guesses

def scout_capped_listing(ep) -> str:
    """List each cart line's category under its price."""
    if ep.scout.capped_done or ep.finished or not ep.cheapest():
        return ''
    ep.scout.capped_done = True
    late = lambda: ep.elapsed() > CLOCK_STOP - CAPPED_CLOCK
    if late() or ep.steps_left() < CAPPED_ROOM:
        ep.log('capped_skip', left=ep.steps_left(), spent=round(ep.elapsed()))
        return ''
    cat_key = mem_kind_key(ep)
    texts: list[str] = []
    ep.scout.capped_more = False
    for line in basket_ids(ep):
        price = mem_price(ep, line)
        kind = '' if not cat_key else mem_label(ep, line, cat_key).strip().lower()
        if not kind or price is None or late() or ep.steps_left() < ep.reserve() + 8:
            continue
        # the guess that lists only this kind, with the fewest other arguments, goes first
        calls = [guess for guess in scout_cap_guesses(ep) if kind in guess[3]]
        if not calls:
            continue
        best = min(calls, key=lambda guess: (guess[3] != {kind}, len(guess[2])))
        texts += scout_list_under(ep, line, price, best)
    return '\n'.join(texts)

def scout_capped_pages(ep, tool: str, call: dict[str, Any], paging: str | None) -> tuple[list[Any], list[str], Any]:
    """Read the capped listing page by page: the pages, their shown texts, the last paging plan."""
    pages: list[Any] = []
    shown: list[str] = []
    plan = None
    late = CLOCK_STOP - CAPPED_CLOCK / 2
    while len(pages) < CAPPED_PAGES:
        fitted = shaped(call, ep.schemas.get(tool))
        if fitted is None or ep.steps_left() < ep.reserve() + 6 or ep.elapsed() > late:
            break
        reply = ep.store.call(tool, fitted)
        if reply is None:
            break
        seen = reply.get('observation')
        ep.ingest(tool, seen)
        pages.append(seen)
        text = ep.render(seen)
        stamp = tool + dumps(fitted)
        ep.cache[stamp] = (ep.version, ep.step, text)
        shown.append('[' + tool + ' ' + dumps(fitted) + ', one step]\n' + text)
        plan = None if not paging else scout_page_plan(ep, tool, seen)
        if not plan or plan[3] is None or ep.finished or not mem_listings_in(ep, seen):
            break
        call = dict(call)
        call[plan[2]] = plan[3]
    return pages, shown, plan

def scout_list_under(ep, line: str, price: float, guess: tuple[str, str, dict[str, Any], set[str]]) -> list[str]:
    """One cart line's capped listing."""
    tool, key, args, _ = guess
    paging = scout_paging_arg(ep, tool)
    first = {k: v for k, v in args.items() if k != key and k != paging}
    first[key] = int(price) if float(price).is_integer() else price
    pages, shown, plan = scout_capped_pages(ep, tool, first, paging)
    if not pages:
        return []
    scout_learn_cap(ep, tool, first, pages[0])
    found = {ident for page in pages for ident, _ in mem_listings_in(ep, page)}
    cart = basket_ids(ep)
    cheaper = []
    for ident in found:
        if ident in cart or ident in ep.mem.pages:
            continue
        cost = mem_price(ep, ident)
        if cost is not None and cost < price and mem_can_buy(ep, ident):
            cheaper.append((cost, ident))
    cheaper.sort()
    # the cheapest rows under the line are opened here, so the pick and the review read their attributes
    roomy = ep.steps_left() >= ep.reserve() + 8
    if cheaper and ep.elapsed() <= CLOCK_STOP - CAPPED_CLOCK / 2 and roomy:
        tried = [ident for _, ident in cheaper[:CHEAPER_OPENS]]
        opened = scout_read_pages(ep, tried, ep.reserve() + 6)
        if opened:
            proof = judge_evidence(ep, [ep.mem.pages[i] for i in tried if i in ep.mem.pages])
            shown.append('[Desk opened the cheapest of these rows itself, one step each]\n' + '\n'.join(opened)
                         + ('\n' + proof if proof else ''))
            ep.log('capped_open', line=line, opened=[i for i in tried if i in ep.mem.pages])
        cheaper = [pair for pair in cheaper if pair[1] not in ep.mem.pages]
    more = 0
    if plan and plan[3] is not None:
        more = plan[0] - sum(map(len, (mem_listings_in(ep, page) for page in pages)))
    ep.log('capped_listing', line=line, tool=tool, pages=len(pages), rows=len(found), unread=len(cheaper), more=max(0, more))
    text = '\n'.join(shown)
    if cheaper:
        listed = ', '.join(f'{i} ({p:g})' for p, i in cheaper[:12])
        tail = '' if len(cheaper) <= 12 else f' and {len(cheaper) - 12} more'
        text += (f'\n[Desk] Unopened rows priced under the cart line ({price:g}), cheapest first: {listed}{tail}.'
                 ' The pick is the cheapest of them whose attributes meet every requirement, if any.')
    if more > 0:
        ep.scout.capped_more = True
        text += f'\n[Desk] {more} more rows of this listing are unread: page on with {plan[2]}={dumps(plan[3])}.'
    return [text]

def scout_read_pages(ep, ids: list[str], keep: int) -> list[str]:
    """Open listings while more than keep steps remain."""
    tool = ep.attr_tool()
    target = ep.id_param(tool)
    if not tool or not target:
        return []
    mem = ep.mem
    texts = []
    for ident in [i for i in ids if i in mem.rows and i not in mem.pages and not ep.is_gone(i)]:
        if ep.steps_left() <= keep or ep.finished or ep.fact_tools.get(tool) is False:
            break
        args = shaped({target: ident}, ep.schemas.get(tool))
        reply = None if args is None else ep.store.call(tool, args)
        if reply is None:
            break
        seen = reply.get('observation')
        ep.ingest(tool, seen)
        mem_note_keys(ep, tool, seen)
        if isinstance(seen, dict) and mem_identify(ep, seen):
            ep.fact_tools[tool] = ep.fact_tools.get(tool, False) or bool(fact_block(seen))
        texts.append(ident + ': ' + ep.render(seen)[:1500])
    if texts:
        ep.log('fill_facts', n=len(texts))
    return texts

def scout_cheaper_rows(ep) -> list[str]:
    """Open unread rows cheaper than cart lines."""
    mem = ep.mem
    if not ep.cheapest():
        return []
    mark = f'{ep.cart_stamp()}#{ep.version}'
    cat_key = mem_kind_key(ep)
    if not cat_key or mark in ep.scout.cheaper_marks:
        return []
    ep.scout.cheaper_marks.add(mark)
    cart = basket_ids(ep)
    kind_of = lambda ident: mem_label(ep, ident, cat_key).strip().lower()
    picks: list[str] = []
    for line in cart:
        price, kind = mem_price(ep, line), kind_of(line)
        if not kind or price is None:
            continue
        owner = mem_owner(ep, kind)
        needs = ep.items[owner - 1].live() if 0 < owner <= len(ep.items) else []
        pool = sorted((i for i in mem.order if i not in cart and i not in mem.pages and mem_can_buy(ep, i)
                       and kind_of(i) == kind and ep.price_rank(i) < price),
                      key=lambda i: (-scout_values_shown(ep, i, needs), mem_price(ep, i) or 0.0))
        picks.extend(i for i in pool[:3] if i not in picks)
    # Desk reads cost a step but no model turn: with room left, read more of the cheaper rows (Arena b10: 9 of 34
    # 'not the best option' orders ended with 6+ steps and 70+ s unused while a cheaper valid line sat unread; real
    # r16 cheapest episodes using a price cap passed 18% vs 9%)
    room = min(ep.steps_left(), ep.clock_room()) - ep.reserve() - 6
    return scout_read_pages(ep, picks[:max(CHEAPER_OPENS, min(6, room))], ep.reserve() + 4)

def scout_learn_cap(ep, name: str, args: dict[str, Any], observation: Any) -> None:
    """Remember a numeric argument acting as price cap."""
    rows = mem_listings_in(ep, observation)
    prices = [mem_price(ep, ident) for ident, _ in rows]
    if None in prices or len(prices) < 3:
        return
    paging = scout_paging_arg(ep, name)
    cat_key = mem_kind_key(ep)
    kinds = set()
    if cat_key:
        kinds = {str(record.get(cat_key) or '').strip().lower() for _, record in rows}
        kinds.discard('')
    top = max(prices)
    # an argument at or above every shown price may be a cap; one call per tool, argument and other arguments is kept
    for key, value in args.items():
        if key == paging or isinstance(value, bool) or not isinstance(value, (int, float)) or value < top:
            continue
        others = lambda call_args: {k: v for k, v in call_args.items() if k != key and k != paging}
        rest = others(args)
        if all(c[0] != name or c[1] != key or others(c[2]) != rest for c in ep.scout.cap_calls):
            ep.scout.cap_calls = (ep.scout.cap_calls + [(name, key, dict(args), kinds)])[-12:]

def scout_cheaper_plan(ep, number: int, cat_key: str | None) -> tuple[str, str, dict[str, Any], float] | None:
    """(tool, cap argument, call, floor): the item's category listed at its lowest affordable price seen."""
    qty = max(1, ep.items[number - 1].qty)
    ids = mem_ids_of_item(ep, number)
    kinds = set()
    if cat_key:
        kinds = {mem_label(ep, i, cat_key).strip().lower() for i in ids}
        kinds.discard('')
    prices = []
    for ident in ids:
        if not mem_can_buy(ep, ident):
            continue
        cost = mem_price(ep, ident)
        if cost is not None and (ep.budget is None or cost * qty <= ep.budget):
            prices.append(cost)
    calls = [c for c in ep.scout.cap_calls if kinds and c[3] & kinds]
    if not prices or not calls:
        return None
    tool, key, args, _ = min(calls, key=lambda c: len(c[2]))
    paging = scout_paging_arg(ep, tool)
    floor = min(prices)
    call = {k: v for k, v in args.items() if k != paging}
    call[key] = int(floor) if float(floor).is_integer() else floor
    fitted = shaped(call, ep.schemas.get(tool))
    return (tool, key, fitted, floor) if fitted is not None else None

def scout_cheaper_search(ep) -> str:
    """Capped listing for items a note re-opened."""
    # a listing that failed only the dropped requirement may never have been looked at,
    # and rows come in no price order
    due = sorted(ep.repick)
    ep.repick = set()
    if ep.finished or not due or not ep.cheapest():
        return ''
    if not ep.scout.cap_calls:
        return scout_widened_search(ep, due)
    cat_key = mem_kind_key(ep)
    texts = []
    for number in due:
        if number > len(ep.items) or ep.finished or ep.steps_left() < ep.reserve() + 3:
            break
        planned = scout_cheaper_plan(ep, number, cat_key)
        if not planned:
            continue
        tool, key, call, floor = planned
        mark = tool + dumps(call)
        stamp = ep.cache.get(mark)
        if stamp and stamp[0] == ep.version:
            continue
        reply = ep.store.call(tool, call)
        if reply is None:
            break
        seen = reply.get('observation')
        ep.ingest(tool, seen)
        shown = ep.render(seen)
        ep.cache[mark] = (ep.version, ep.step, shown)
        listed = mem_listings_in(ep, seen)
        if any(floor < (mem_price(ep, i) or 0) for i, _ in listed):
            ep.scout.cap_calls = [c for c in ep.scout.cap_calls if c[0] != tool or c[1] != key]
        ep.log('cheaper_search', item=number, tool=tool, args=call, rows=len(mem_listings_in(ep, seen)))
        head = (f'[Desk listed this category at or under {floor:g}, the lowest price seen for item {number}: after a requirement was dropped, '
                f'a listing never looked at can be the cheapest fit ({tool} {dumps(call)}, one step)]\n')
        texts.append(head + shown + scout_cheapest_note(ep, [seen]))
    return '\n'.join(texts)

def scout_widened_search(ep, due: list[int]) -> str:
    """Without a price-capped listing: search each re-opened item once more with the values still in force."""
    # Arena r15 noask b04: 18 'not the best option' orders, 17 under a cheapest pick; after a refusal dropped a
    # requirement the model checked the old pick out again, never looking at listings that only the dropped value had kept out
    plan = scout_search_template(ep)
    if plan is None:
        return ''
    tool, base, field = plan
    texts = []
    for number in due:
        stop = ep.finished or ep.steps_left() < ep.reserve() + 4 or number > len(ep.items) or ep.past_stop()
        if stop:
            break
        values = scout_value_words(ep, number)
        noun = scout_title_noun(ep, number) or ep.items[number - 1].name
        words = noun if not values else scout_value_tries(values, values[0], noun)[0]
        call = dict(base)
        call[field] = words[:30].strip()
        args = shaped(call, ep.schemas.get(tool))
        if args is None:
            continue
        reply = ep.store.call(tool, args)
        if reply is None:
            break
        seen = reply.get('observation')
        ep.ingest(tool, seen)
        ep.log('widened_search', item=number, words=words)
        head = (f'[Desk searched item {number} again with only the requirements still in force ({tool} {dumps(args)}, one step): a requirement '
                'was dropped, so listings it kept out can now be the cheapest fit. Read the cheapest rows of the right kind before checking out]\n')
        texts.append(head + ep.render(seen) + scout_cheapest_note(ep, [seen]))
    return '\n'.join(texts)

def scout_wider_hint(ep) -> str:
    """Capped listing or a wider-search push, once."""
    if ep.scout.wider_done or not ep.cheapest() or ep.steps_left() < 14 + 2 * len(ep.items):
        return ''
    ep.scout.wider_done = True
    listed = scout_capped_listing(ep)
    if not listed:
        priced = '; '.join(f'{i} at {mem_price(ep, i):g}' for i in basket_ids(ep) if mem_price(ep, i) is not None)
        return (f"Not run yet (no step used). The pick is the cheapest and {ep.steps_left()} steps are left. Rows arrive in no price order, "
                "and this store's search reads titles only, while a required value is often written just in the attributes, so searching by value "
                "misses many fitting listings. Ahead of checkout, search every item once more under other product names (synonyms, the bare one-word "
                "noun, title words of fitting listings seen so far), looking only at buyable rows priced under the cart's line" + (f' ({priced})' if priced else '')
                + '; open the cheapest such row, swap any that fits in for the line, then check out.')
    rest = ' and page on through the unread rows while steps allow' if ep.scout.capped_more else ''
    return ("Not run yet. Because the pick is the cheapest, Desk listed the cart's category up to its line's price on its own:\n" + listed
            + f'\nOpen the cheapest of these rows that could fit{rest}; swap in the cheapest one that meets every requirement for the line, then check out.')

def scout_widen(ep) -> None:
    """Read listings of each kind before planning."""
    tool, target = ep.roles.get('compare'), ep.side_param()
    if not tool or not target or ep.side_view_bare():
        return scout_widen_singly(ep)
    if ep.steps_left() <= 12:
        return None
    schema = ep.schemas.get(tool) or {}
    spec = schema['properties'][target]
    limit = spec.get('maxItems')
    cap = limit if isinstance(limit, int) else 4
    pools = mem_kind_pools(ep)
    share = max(1, cap // max(1, len(pools)))
    picks: list[str] = []
    for ids in pools:
        picks.extend([x for x in ids if x not in ep.mem.pages][:share])
    picks = picks[:cap]
    args = None if len(picks) < 2 else shaped({target: picks}, schema)
    reply = None if args is None else (len(ep.mem.pages), ep.store.call(tool, args))
    if reply is None or reply[1] is None:
        return None
    before, answer = reply
    seen = answer.get('observation')
    ep.ingest(tool, seen)
    mem_note_keys(ep, tool, seen)
    rows = mem_listings_in(ep, seen)
    if rows:
        ep.fact_tools[tool] = ep.fact_tools.get(tool, False) or any(fact_block(r) for _, r in rows)
    pages = ep.mem.pages
    ep.scout.widened += list(filter(pages.__contains__, picks))
    ep.log('widen', ids=picks, opened=len(ep.mem.pages) - before)
    if rows and ep.fact_tools.get(tool) is False:
        return scout_widen_singly(ep)
    if len(ep.mem.pages) == before and len(picks) >= 3 and ep.steps_left() > 12:
        pair = picks[:2]
        args = shaped({target: pair}, schema)
        answer = ep.store.call(tool, args) if args is not None else None
        if answer is not None:
            ep.ingest(tool, answer.get('observation'))
            ep.scout.widened += [i for i in pair if i in ep.mem.pages]
    return None

def scout_widen_singly(ep) -> None:
    tool = ep.attr_tool()
    if not (tool and ep.id_param(tool)):
        return
    pages = ep.mem.pages
    words_of = lambda text: set(re.findall('[a-z0-9]{3,}', text))
    wanted: set[str] = set()
    for _, need in ep.live_needs():
        wanted |= words_of(need.split(':', 1)[-1].lower())
    each = 4 if len(ep.items) > 2 else WIDEN_EACH
    built = []
    for ids in mem_kind_pools(ep):
        # unread buyable listings of one kind, best first, and how many may go in the first round
        opened = len([i for i in ids if i in pages])
        pool = [i for i in ids if i not in pages and mem_can_buy(ep, i)]
        if ep.cheapest():
            pool.sort(key=ep.price_rank)
        pool.sort(key=lambda i: -len(wanted & words_of(dumps(ep.mem.rows.get(i) or {}).lower())))
        built.append((pool[:max(0, each - opened)], max(0, 4 - opened)))
    first, more = [], []
    for k in range(4):
        for lane, head in built:
            if k < min(head, len(lane)):
                first.append(lane[k])
    for k in range(each):
        for lane, head in built:
            if head <= k < len(lane):
                more.append(lane[k])
    cap = max(5, 2 * len(built) + 2)
    before = set(pages)
    read = scout_read_pages(ep, first[:min(cap, 8)], 22)
    scout_read_pages(ep, (first[cap:] + more)[:max(0, 10 - len(read))], 30)
    ep.scout.widened.extend(i for i in pages if i not in before)

def scout_search_template(ep) -> tuple[str, dict[str, Any], str] | None:
    """Tool, kept args and text field for new searches."""
    tool = ep.roles.get('search')
    schema = ep.schemas.get(tool or '') or {}
    props = schema.get('properties') or {}
    texts = []
    for k, v in props.items():
        if isinstance(v, dict) and 'enum' not in v and 'string' in type_names(v):
            texts.append(k)
    word_field = re.compile('(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword', re.I)
    named = list(filter(word_field.search, texts))

    def worded(args: dict[str, Any], key: str) -> bool:
        value = args.get(key)
        return isinstance(value, str) and re.search('[a-z]', value, re.I) is not None
    if not tool:
        return None
    for args in reversed(ep.scout.searched):
        fields = [k for k in (named or texts) if worded(args, k)]
        # without a field named for words, only one filled with varied phrases counts as the text field
        if not named:
            fields = [k for k in fields if scout_free_field(ep, k)]
        if fields:
            field = fields[0]
            similar = [a for a in ep.scout.searched if worded(a, field)]
            return tool, scout_kept_args(ep, args, field, schema, props, similar), field
    return None

def scout_free_field(ep, key: str) -> bool:
    """Text field filled with varied phrases."""
    cat_key = mem_kind_key(ep)
    labels = set()
    if cat_key:
        labels = {str(r.get(cat_key) or '').strip().lower() for r in ep.mem.rows.values()} - {''}
    sent = [a.get(key) for a in ep.scout.searched]
    distinct = {str(v) for v in sent if v}
    if len(distinct) < 2 or all(len(str(v).split()) < 2 for v in sent):
        return False
    return not any(str(v or '').strip().lower() in labels for v in sent)

def scout_kept_args(ep, args: dict[str, Any], field: str, schema: dict[str, Any], props: dict[str, Any], worded_args: list[dict[str, Any]]) -> dict[str, Any]:
    """Search arguments worth keeping."""
    required = schema.get('required') or []
    choice = lambda key: isinstance((props.get(key) or {}).get('enum'), list)
    cat_key = mem_kind_key(ep)
    shelves = {plain_key(item.shelf or '') for item in ep.items} | {plain_key(item.name) for item in ep.items}
    if cat_key:
        shelves |= {plain_key(str(row.get(cat_key) or '')) for row in ep.mem.rows.values()}
    long = lambda text: {w for w in stems(text) if len(w) >= 4}

    def names_kind(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        return plain_key(value) in shelves or any(long(value) & long(s) for s in shelves if s)
    # required arguments stay; an optional choice stays unless its value only names the product kind
    keep = {}
    for k, v in args.items():
        if k == field:
            continue
        if k in required or (choice(k) and not names_kind(v)):
            keep[k] = v
    for key in list(keep):
        if not choice(key):
            continue
        values = [a[key] for a in worded_args if key in a]
        if values:
            marks = list(map(dumps, values))
            keep[key] = values[marks.index(max(marks, key=marks.count))]
    return keep

def scout_value_words(ep, number: int) -> list[str]:
    pointers = [m for _, m in ep.refs]
    values = [need.split(':', 1)[1].strip() for n, need in ep.live_needs() if n == number and need not in pointers and ':' in need]
    return [v for v in values if v and not has_bound(v, 'alongside|listing|page|shown|')]

def scout_title_noun(ep, number: int) -> str:
    """Word of the item name titles carry most."""
    item = ep.items[number - 1]
    words = [w for w in re.findall('[a-z]{3,}', item.name.lower()) if w not in ('and', 'the', 'for', 'with')]
    if not words:
        return ''
    cat_key = mem_kind_key(ep)
    titles = []
    for record in ep.mem.rows.values():
        if cat_key and not mem_belongs(ep, item, str(record.get(cat_key) or '')):
            continue
        titles.append(' '.join(str(v) for k, v in record.items() if isinstance(v, str) and re.search(TITLE_FIELD, k, re.I)).lower())

    def score(w: str) -> tuple[int, int]:
        stem = w[:-1] if w.endswith('s') and len(w) > 4 else w
        return len([t for t in titles if stem in t]), words.index(w)
    return max(words, key=score)

WORD_RULE_ADVICE = {
    'first2': 'This store reads just the first two words of a query and ignores any after them. Start with the wanted value and then the '
              'product noun, the two words a fitting title would carry.',
    'and': 'Each extra query word cuts rows here, since a row has to contain all of them. Use two or three words a fitting '
           'title would carry: the product noun and the wanted value.',
    'or': 'A single query word is enough for a row to match here, so every extra word pulls in rows of other kinds. Put the wanted '
          'value first and use two or three words a fitting title would carry.'}

def scout_rule_votes(log: list[tuple[str, tuple[str, ...], tuple[str, ...], Any]],
                     query: tuple[str, tuple[str, ...], tuple[str, ...], Any], heads: list[str]) -> dict[str, int]:
    """Evidence on the query-word rule from comparing this search with earlier ones."""
    other, words, ids, hits = query
    trim = lambda word: word[:-1] if len(word) > 4 and word.endswith('s') else word
    # a word every shown title carries leaves the rows unchanged under any rule: no evidence either way
    # (an every-word store answers "round toe wedge" and "round toe wedge pumps" alike when all rows are pumps)
    apart = lambda extra: any(trim(w) not in head for head in heads for w in extra)
    votes: dict[str, int] = {}
    cast = lambda key: votes.__setitem__(key, votes.get(key, 0) + 1)
    for entry in log:
        o_other, o_words, o_ids, o_hits = entry
        if o_words == words or o_other != other:
            continue
        if min(len(words), len(o_words)) > 2 and words[:2] == o_words[:2] and words[2:] != o_words[2:]:
            if o_ids != ids:
                cast('not first2')
            elif apart(set(words[2:]) ^ set(o_words[2:])):
                cast('first2')
        # one query extends the other at its end: fewer hits -> words narrow, more -> they widen; past the second
        # word, the same rows and count -> only the first two words count
        if words[:len(o_words)] == o_words:
            short, short_hits, short_ids, longer, long_hits, long_ids = o_words, o_hits, o_ids, words, hits, ids
        elif o_words[:len(words)] == words:
            short, short_hits, short_ids, longer, long_hits, long_ids = words, hits, ids, o_words, o_hits, o_ids
        else:
            continue
        if short_hits is None or long_hits is None:
            continue
        if long_hits != short_hits:
            cast('and' if long_hits < short_hits else 'or')
        changed = long_hits != short_hits or long_ids != short_ids
        if len(short) >= 2 and (changed or apart(longer[len(short):])):
            cast('first2' if not changed else 'not first2')
    return votes

def scout_learn_rule(ep, name: str, args: dict[str, Any], observation: Any) -> str:
    """Infer the store's query-word rule."""
    plan = scout_search_template(ep)
    paging = scout_paging_arg(ep, name)
    if not plan or plan[0] != name or (paging and args.get(paging) not in (None, 0, '', 1)):
        return ''
    field = plan[2]
    words = tuple(str(args.get(field) or '').lower().split())
    if not words:
        return ''
    hits = None
    if isinstance(observation, dict):
        for k, v in observation.items():
            if isinstance(v, int) and not isinstance(v, bool) and re.search('hit|count|total|found|match|tally', k, re.I):
                hits = v
                break
    found = mem_listings_in(ep, observation)
    rest = sorted((k, dumps(v)) for k, v in args.items() if k != field and k != paging)
    query = (dumps(rest), words, tuple(ident for ident, _ in found), hits)
    heads = []
    for _, record in found:
        heads.append(' '.join(str(v) for k, v in record.items() if isinstance(v, str) and re.search(TITLE_FIELD, k, re.I)).lower())
    votes = scout_rule_votes(ep.scout.rule_log, query, heads)
    ep.scout.rule_log = (ep.scout.rule_log + [query])[-16:]
    # an extra word that changes nothing is also what an any-word store shows, so a first-two-words reading also
    # needs a narrowing seen; one pair sharing its first two words with other rows rules it out
    narrow, wide = (votes.get(k, 0) for k in ('and', 'or'))
    if wide > narrow:
        rule = 'or'
    elif narrow and votes.get('first2', 0) and not votes.get('not first2', 0):
        rule = 'first2'
    else:
        rule = 'and' if narrow > wide else None
    if rule in (None, ep.scout.word_rule):
        return ''
    ep.scout.word_rule = rule
    ep.log('word_rule', rule=rule, votes=votes)
    if len([t for t in ep.trace if t.get('kind') == 'word_rule']) > 3:
        return ''
    return '\n[Desk, from the searches so far] ' + WORD_RULE_ADVICE[rule]

def scout_value_tries(values: list[str], lead: str, noun: str) -> list[str]:
    """Queries for one new value: at most two, distinct and non-empty."""
    # real races 11-14: 2-3 word searches led by words a fitting title carries surfaced the paying listing
    # first most often (.21-.39 per call vs .06-.18 for 4+ words): one value, then the product noun
    head = ' '.join(lead.split()[:2])
    with_noun = head if not noun or noun in head.lower() else head + ' ' + noun
    # within a task (all agents, r11/r13/r14), a page 1 led by two required values listed a paid listing
    # .51/.49/.42 vs .47/.43/.32 for one value and the noun; one value alone did worse than either
    partner = next((v for v in values if v != lead and len(v.split()) == 1 and v.lower() not in head.lower()
                    and not re.search('\\d', v)), None)
    if partner is not None and len(head.split()) == 1:
        tries = [head + ' ' + partner, with_noun]
    else:
        tries = [with_noun, head]
    out: list[str] = []
    for w in tries:
        if w not in out:
            out.append(w)
    return [w for w in out[:2] if w]

def scout_value_rows(ep, item: ItemSpec, found: list[str]) -> list[str]:
    """Buyable rows of the item's kind within the budget (ignored when every row is far above it)."""
    cat_key = mem_kind_key(ep)
    cap = ep.numeric_budget()
    far = cap is not None and found and not any(p is None or p <= 10 * cap for p in (mem_price(ep, i) for i in found))
    def wanted(ident: str) -> bool:
        if not mem_can_buy(ep, ident) or (cat_key and not mem_belongs(ep, item, mem_label(ep, ident, cat_key))):
            return False
        return far or cap is None or (mem_price(ep, ident) or 0) <= cap
    return list(filter(wanted, found))

def scout_value_search(ep, numbers: list[int]) -> str:
    """Search the new values with the product noun."""
    plan = scout_search_template(ep)
    if ep.steps_left() < 22 or not plan:
        return ''
    tool, base, field = plan
    seen: list[str] = []
    budget = 8
    for number in numbers:
        values = scout_value_words(ep, number)
        item = ep.items[number - 1]
        settled = {t.split(':', 1)[1].strip() for t in item.added if ':' in t}
        first = next((v for v in values if v in settled), None)
        if first is None or budget <= 2:
            continue
        for words in scout_value_tries(values, first, scout_title_noun(ep, number)):
            args = shaped({**base, field: words[:30].strip()}, ep.schemas.get(tool))
            if ep.finished or args is None:
                break
            budget -= 1
            reply = ep.store.call(tool, args)
            if reply is None:
                break
            observation = reply.get('observation')
            ep.ingest(tool, observation)
            rows = scout_value_rows(ep, item, [ident for ident, _ in mem_listings_in(ep, observation)])
            seen.append(f"{dumps(args.get(field))}: {', '.join(rows[:6]) or 'no row of that kind'}")
            if rows:
                opened = scout_read_pages(ep, [i for i in rows if i not in ep.mem.pages][:min(3, budget)], ep.reserve() + 16)
                budget -= len(opened)
                if opened:
                    seen.append('opened: ' + ' | '.join(opened)[:2400])
                break
    if seen:
        ep.log('value_search', seen=[s[:200] for s in seen])
        return '[For each new required value Desk ran a search with the product noun and opened the top rows of the right kind] ' + '\n'.join(seen)
    return ''

class Interviewer:
    """Questions to the shopper and their answers."""

    def __init__(self) -> None:
        self.started = False
        self.plan_order: list[int] = []

def talk_due(ep) -> bool:
    blocked = ep.talk.started or not ep.roles.get('ask') or ep.free_messages() <= 0 or ep.steps_left() <= 8
    return not blocked and (len(ep.mem.pages) >= 3 * len(ep.items) + 1 or ep.step >= max(8, ep.max_steps // 4))

def talk_msg_args(ep, text: str) -> dict[str, Any] | None:
    schema = ep.schemas.get(ep.roles.get('ask') or '') or {}
    props = schema.get('properties') or {}
    needed = schema.get('required') or []
    texts = [k for k, v in props.items() if isinstance(v, dict) and 'enum' not in v and 'string' in type_names(v)]
    target = next(filter(texts.__contains__, needed), None)
    if not target:
        target = texts[0] if texts else None
    if not target:
        return None
    args: dict[str, Any] = {target: text}
    # every other required argument takes its first allowed value; one without a value list leaves no valid call
    for key in (k for k in needed if k not in args):
        options = (props.get(key) or {}).get('enum')
        if not isinstance(options, list) or not options:
            return None
        args[key] = options[0]
    return shaped(args, schema)

def talk_run(ep) -> str:
    """Plan, ask and read the answers; returns a summary."""
    # adaptive L2 (user OK 11-08): lists at every cap; on 1 product a single first (strict-safe), then 5 or 3 keys, then
    # 8; on 2+ products an 8-key list per product, or grouped messages of 2 keys per product when products outnumber the
    # messages; any refusal turns the rest into single questions. Real r17 (lenient, cap 2): list askers missed a need in
    # .21-.27 of episodes vs .47-.51 for single askers (ours .47); the same agent passed .36 with lists vs .28 with singles.
    # Strict r16: single askers .15-.17 vs list openers .10-.12 (review 2026-11-08-1720-asking-engine-review.md)
    ep.talk.started = True
    if ep.free_messages() <= 0:
        return ''
    req_profile_now(ep)
    scout_widen(ep)
    lanes = talk_plan(ep, ep.free_messages())
    found = {n: 0 for n in lanes}
    spent = dict(found)
    used, answered, fronted = ({n: set() for n in lanes} for _ in range(3))
    # keys a list named without drawing a need lead the item's next list from place 1 (re-listed at place 1 they drew
    # the need .95-1.00: real r15 18/19 and 16/16, r17 8/8; asklab: lenient caps 3/4 +7/+11 of 141 tasks)
    missed: dict[int, list[str]] = {n: [] for n in lanes}
    follow: list[tuple[int, list[str]]] = []
    sent: list[tuple[int, str, str]] = []
    told: list[str] = []
    many, rested = len(ep.items) > 1, 0
    # an unknown cap is planned as 2 messages: a third may not exist
    plan_cap = ep.free_messages() if ep.cap_stated else min(ep.free_messages(), 2)
    split_asked = lambda heard: 1 in [whole(x) for x in as_list(heard.get('crowded'))]
    unused = lambda n: [x for x in lanes[n] if plain_key(x) not in used[n]]
    listing = True
    # a list names the item's own listing keys after the planned picks, most carried first, in the store's spelling
    # (real r15: "packaging type" drew nothing where the need was "pack type"; both keys were listed)
    table = mem_key_table(ep)
    for n in lanes:
        have = set(map(plain_key, lanes[n]))
        lanes[n] = lanes[n] + [k for k in talk_own_keys(ep, n, table) if plain_key(k) not in have]
    rounds = talk_spread_rounds(ep, lanes, plan_cap) if many else []
    # a late regroup only when messages outnumber the products at the start; products = messages: one product each
    spare = many and len([v for v in lanes.values() if v]) < plan_cap
    form = 'grouped' if rounds else 'list'
    ep.log('ask_form', form=form, messages=ep.free_messages(), plan_cap=plan_cap, items=len(lanes))

    def tally(batch: list[tuple[int, str, str]], heard: dict[str, Any]) -> list[int]:
        had = [len(item.added) for item in ep.items]
        talk_apply_answers(ep, heard, told, ' '.join(reply for *_, reply in batch))
        fresh = [n for n, (item, before) in enumerate(zip(ep.items, had), 1) if len(item.added) > before]
        for n in fresh:
            found[n] = found.get(n, 0) + 1
        return fresh

    def regroup() -> list[list[int]]:
        """Products still without a need outnumbering the messages left are asked together, the least asked first."""
        waiting = [n for n in lanes if not found.get(n, 0) and unused(n)]
        left = ep.free_messages()
        if not (listing and spare and left >= 1 and len(waiting) > left):
            return []
        rank = dict(zip(ep.talk.plan_order, range(len(ep.talk.plan_order))))
        waiting.sort(key=lambda n: (spent.get(n, 0), rank.get(n, len(rank) + n)))
        return [waiting[:-(-len(waiting) // left)]]
    while ep.free_messages() > 0 and not ep.finished and ep.steps_left() > 6:
        if not rounds and not follow:
            rounds = regroup()
        groups = []
        if rounds and not follow:
            groups = [(n, labels) for n, labels in ((n, unused(n)[:GROUP_KEYS]) for n in rounds.pop(0)) if labels]
        if len(groups) >= 2:
            text = talk_group_text(ep, groups)
            if text not in (talk_msg_args(ep, text) or {}).values():
                # the message tool caps its text: one key per product
                groups = [(n, labels[:1]) for n, labels in groups]
                text = talk_group_text(ep, groups)
            for n, labels in groups:
                spent[n] += 1
                used[n].update(map(plain_key, labels))
            batch = talk_send(ep, [(0, text)])
            if not batch:
                break
            sent += batch
            # a reply answers one of the items at most, so a "don't mind" cannot be pinned to an item
            heard = {**talk_read_answers(ep, batch), 'free': []}
            split = split_asked(heard)
            fresh = tally(batch, heard)
            if not split:
                continue
            # the shopper wants one thing per message: single questions from now on, the first unanswered item first
            rounds, listing = [], False
            first, labels = next(((n, x) for n, x in groups if n not in fresh), (0, []))
            if first:
                used[first].discard(plain_key(labels[0]))
                follow[:0] = [(first, labels[:1])]
            ep.log('items_split', items=[n for n, _ in groups], found=fresh)
            continue
        if follow:
            number, labels = follow.pop(0)
        elif groups:
            # a lone product in its slice: one list of its best keys
            [(number, _)] = groups
            labels = talk_list_pick(lanes[number], used[number], answered[number], fronted[number], missed[number])
        else:
            number, labels = talk_choose_next(ep, lanes, found, spent, used, answered, fronted, rested, listing, missed)
            if not labels:
                break
            if listing:
                labels = labels[:talk_list_size(spent.get(number, 0), plan_cap, many)]
        while True:
            text = talk_text_for(ep, number, labels)
            if len(labels) <= 3 or text in (talk_msg_args(ep, text) or {}).values():
                break
            # the message tool caps its text: the longest list it takes, best-ranked properties kept
            labels = labels[:-1]
        rested = 0
        keys = list(map(plain_key, labels))
        used[number].update(keys)
        fronted[number].update(keys[:LIST_FRONT])
        batch = talk_send(ep, [(number, text)])
        if not batch:
            break
        sent += batch
        spent[number] += 1
        heard = talk_read_answers(ep, batch)
        # the refusal is read before the need is counted: a crowded reply that also gives a need still ends the lists
        split = split_asked(heard)
        if len(labels) > 2:
            # a reply to a list answers one of its properties at most: a "don't mind" names none of them
            heard = {**heard, 'free': []}
            for entry in as_list(heard.get('needs')):
                prop = stems(str(entry.get('need') or '').split(':', 1)[0]) if isinstance(entry, dict) else None
                if prop:
                    answered[number].update(plain_key(x) for x in lanes[number]
                                            if names_same(prop, stems(x)) or names_same(stems(x), prop))
        if len(labels) > 1 and not split:
            # the keys after the one that drew a need, then those before it, lead the next list; with no need the first
            # key is settled and the others lead
            hit = next((i for i, x in enumerate(labels) if plain_key(x) in answered[number]), 0)
            missed[number] = labels[hit + 1:] + labels[:hit] + [x for x in missed[number] if x not in labels]
        fresh = tally(batch, heard)
        if split and len(labels) > 1 and listing:
            # the shopper wants one property per message: single questions from now on, the list's head first
            listing = False
            used[number].difference_update(keys[1:])
            if number not in fresh:
                follow[:0] = [(number, labels[:1])]
            ep.log('list_split', item=number, found=bool(fresh))
            continue
        if fresh:
            continue
        words = labels[0].split()
        empty = bool(heard) and heard.get('budget') is None and all(not as_list(heard.get(k)) for k in ('needs', 'drop', 'free'))
        # after a reply wanting one question per message only the head word of a longer label earns a second try
        # (.28 in real races 10-13); the same label again, with or without examples, almost never draws a value
        if split and words[1:] and plain_key(words[-1]) not in used[number]:
            follow[:0] = [(number, words[-1:])]
        elif many and (split or empty):
            # a reply with nothing in it, or wanting the message split, seldom turns into a value when the same item
            # is asked again: the next message goes to another item still without a value, and this one waits a turn
            rested = number
            ep.log('ask_rest', item=number, split=split)
    return talk_summary(ep, found, sent, told)

def talk_list_size(asked: int, plan_cap: int, many: bool) -> int:
    """Keys in the next list for an item asked `asked` times: 8 on 2+ products; on 1 product a single first (a strict
    reader refuses an opening list: r16 3-4-property first lists drew a need .08 vs singles .28), then 5 keys with 2
    messages or 3 with more, then 8; with one message in all, one 8-key list."""
    if many or plan_cap <= 1:
        return LIST_KEYS
    if asked == 0:
        return 1
    return (5 if plan_cap <= 2 else 3) if asked == 1 else LIST_KEYS

def talk_choose_next(ep, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]],
                answered: dict[int, set[str]], fronted: dict[int, set[str]], rested: int, listing: bool,
                missed: dict[int, list[str]] | None = None) -> tuple[int, list[str]]:
    """Item and labels for the next question."""
    # a resting item gives way only to items with nothing drawn yet: real r8-r14 multi-product
    # tasks, asking a product that already gave a value drew another .003-.043, one still
    # without a value after an empty reply .11-.42
    waiting = {}
    for n, labels in lanes.items():
        if n != rested and not found.get(n, 0):
            waiting[n] = labels
    number, labels = (0, [])
    if waiting:
        number, labels = talk_next_single(ep, waiting, found, spent, used)
    if not labels:
        number, labels = talk_next_single(ep, lanes, found, spent, used)
    if not listing:
        return number, labels
    pick = lambda n: talk_list_pick(lanes[n], used[n], answered[n], fronted[n], None if missed is None else missed.get(n, []))
    if not labels:
        # every planned property went out once: list again the ones that drew no answer
        ranked = sorted(lanes, key=lambda n: (found.get(n, 0) > 0, spent.get(n, 0)))
        number = next(filter(pick, ranked), 0)
    if not number:
        return number, []
    return number, pick(number)

def talk_summary(ep, found: dict[int, int], sent: list[tuple[int, str, str]], told: list[str]) -> str:
    drawn = [n for n, hits in found.items() if hits > 0]
    read_before = list(ep.mem.pages)
    searched = '' if not drawn else scout_value_search(ep, drawn)
    if told:
        ep.version += 1
    talk = '\n'.join('Q: ' + q + '\nA: ' + a for _, q, a in sent)
    out = ("Desk's interviewer asked the shopper (one step per question):\n" + talk
           + '\nFirm requirements these answers add: ' + ('; '.join(told) or 'none')
           + '. Every pick must meet them (where needed, search again with the new values).')
    pointed = [need for _, need in ep.refs if any(need in t for t in told)]
    if pointed:
        out += (' Their value comes from another listing: ' + ' | '.join(pointed) + '. The order counts only once all of these are open: '
                'locate that listing, open its full page and every listing it refers on to; the value comes from the last page.')
    widened = set(ep.scout.widened)
    order = list(dict.fromkeys(ep.scout.widened))
    ep.scout.widened = []
    pages = [ep.mem.pages[i] for i in order if i in ep.mem.pages]
    if pages:
        # pages read so the questions fit the store: the model has not seen them in its own results
        body = '\n'.join(ep.render(page)[:1500] for page in pages)
        out += '\n[pages Desk read while preparing the questions, not shown to you before (one step each)]\n' + body[:8000]
        proof = judge_evidence(ep, pages)
        out += '\n' + proof if proof else ''
    again = '' if not told else talk_recheck_read(ep, [i for i in read_before if i not in widened])
    for extra in (again, searched):
        if extra:
            out += '\n' + extra
    if ep.queued:
        out += '\n[notes seen meanwhile] ' + ' | '.join(ep.queued)
        ep.queued = []
    return out

def talk_recheck_read(ep, ids: list[str]) -> str:
    """Evidence on pages read before the answers."""
    # real races 5-14: of failed picks that had opened the paid listing, .35-.57 opened it
    # only before the hidden need was drawn and never judged it against that need
    kinds = set()
    for n in range(1, len(ep.items) + 1):
        kinds.update(mem_ids_of_item(ep, n))
    wanted = set(ids)
    pool = [i for i in reversed(ep.mem.order) if i in wanted and (i in kinds or not kinds) and mem_can_buy(ep, i) and i not in ep.gone]
    cheapest = ep.cheapest()
    if cheapest:
        pool = sorted(pool, key=ep.price_rank)
    proof = judge_evidence(ep, [ep.mem.pages[i] for i in pool[:8]], False)
    if not proof:
        return ''
    order = ', cheapest first' if cheapest else ''
    return (f'[listings you opened before these answers{order}, with their keys named for a requirement '
            'in force: judge them again against the new requirements before searching further] ' + proof)

def talk_next_single(ep, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> tuple[int, list[str]]:
    rank = dict(zip(ep.talk.plan_order, range(len(ep.talk.plan_order))))
    for n in sorted(lanes, key=lambda n: (found.get(n, 0) > 0, spent.get(n, 0), rank.get(n, len(rank) + n))):
        left = [label for label in lanes[n] if plain_key(label) not in used[n]]
        if left:
            return n, left[:1]
    return 0, []

def talk_list_pick(lane: list[str], used: set[str], answered: set[str], fronted: set[str], ahead: list[str] | None = None) -> list[str]:
    """Labels for one list question: keys an earlier list missed first when given (ahead), then fresh keys, then the rest."""
    # real race 14 lists: a property in places 1-3 drew its need .79, in 4+ .43; one named only
    # further back before drew .58 when moved to the front, .33 when left behind
    pool: dict[str, str] = {}
    for label in lane:
        word = plain_key(label)
        if word in answered or word in pool or re.fullmatch('options?', word):
            continue
        pool[word] = label
    if ahead is not None:
        kept = list(pool.values())
        lead = [x for x in ahead if x in kept]
        fresh = [x for x in kept if plain_key(x) not in used and x not in lead]
        return (lead + fresh + [x for x in kept if x not in lead and x not in fresh])[:LIST_KEYS]
    held = used - fronted
    fresh = [label for word, label in pool.items() if word not in used]
    again = [label for word, label in pool.items() if word in held]
    front = (fresh + again)[:LIST_FRONT]
    rest = [label for label in pool.values() if label not in front]
    return (front + rest)[:LIST_KEYS]

def talk_spread_rounds(ep, lanes: dict[int, list[str]], cap: int | None = None) -> list[list[int]]:
    """Spread items over too few messages."""
    items = [n for n in sorted(lanes) if lanes[n]]
    free = ep.free_messages() if cap is None else cap
    if free <= 0 or len(items) <= free:
        return []
    size, extra = divmod(len(items), free)
    cuts = [0]
    for k in range(free):
        cuts.append(cuts[-1] + size + (k < extra))
    return [items[a:b] for a, b in zip(cuts, cuts[1:]) if b > a]

def talk_group_text(ep, groups: list[tuple[int, list[str]]]) -> str:
    """One message asking several items, each with its own keys."""
    said = lambda labels: ' or '.join(re.sub('[_]+', ' ', label).strip() for label in labels)
    return ' '.join(f"On the {ep.items[n - 1].name}: which {said(labels)} should it have?" for n, labels in groups)

def talk_text_for(ep, number: int, labels: list[str]) -> str:
    item = ep.items[number - 1]
    if len(labels) > 2:
        *head, last = [re.sub('[_]+', ' ', label).strip() for label in labels]
        where = f'{item.name} ({item.shelf})' if item.shelf and plain_key(item.shelf) != plain_key(item.name) else item.name
        return f"For the {where}, what {', '.join(head)} or {last} do you want?"
    return f"Which {' or '.join(re.sub('[_]+', ' ', label).strip() for label in labels)} should the {item.name} have?"

def talk_own_keys(ep, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
    """Item's unsettled listed keys, best first."""
    settled = [stems(m.split(':', 1)[0]) for m in ep.items[number - 1].musts]
    mine = list(filter(lambda row: mem_owner(ep, row[0]) == number, table))
    measured = {}
    for _, key, _, values in table:
        measured[key] = 2 * len([v for v in values if re.match(MEASURE_VALUE, v)]) > len(values)
    if len(ep.items) == 1 and not mine:
        mine = list(table)
    ranked = sorted(mine, key=lambda r: (-r[2], measured.get(r[1], False), -len(r[3])))
    keys = []
    for _, key, _, _ in ranked:
        if re.search(NOT_PROPERTY, key):
            continue
        if not any(names_same(prop, stems(key)) for prop in settled):
            keys.append(key)
    keys.sort(key=lambda k: mem_is_catch_all(ep, k))
    return keys

def talk_plan_prompt(ep, count: int, table: list[tuple[str, str, int, list[str]]]) -> str:
    settled = '\n'.join(f'item {n}: {item.name} - settled by the request: ' + ('; '.join(item.musts) or 'nothing')
                        for n, item in enumerate(ep.items, 1))
    listing = []
    last_label = None
    for index, row in enumerate(table, 1):
        label, name, carried, values = row
        if index == 1 or last_label != label:
            listing.append(f'category {label!r}:')
        last_label = label
        tags = mem_stated_clash(ep, label, name, values)
        if listy(values):
            tags += ' [several features in one: ask last]'
        listing.append(f'  {index}. {name} (on {carried}): ' + ' | '.join(values) + tags)
    text = PLAN.replace('@N@', str(max(1, count))) + '\n\nREQUEST:\n' + ep.request + '\n\nITEMS:\n' + settled
    asks = [str(x) for x in as_list(ep.brief.get('ask_first')) if str(x).strip()]
    if asks:
        text += '\n\nASK FIRST (the request asks to be asked about these):\n' + '\n'.join(asks)
    return text + '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(listing) or '(no full listing records yet)')

def talk_read_picks(ep, got: dict[str, Any]) -> tuple[dict[int, list[str]], dict[int, set[str]], list[int]]:
    size = len(ep.items)
    picked: dict[int, list[str]] = {}
    hints: dict[int, set[str]] = {}
    for n in range(1, size + 1):
        picked[n], hints[n] = [], set()
    order: list[int] = []
    for pick in filter(lambda p: isinstance(p, dict), as_list(got.get('ask'))):
        number = pick_number(pick.get('item') or 1, size)
        raw = str(pick.get('label') or '')
        label = re.split('\\s+(?:and|or)\\s+|[/,;(]', raw)[0].strip().rstrip('?.')
        # a listing's bare purchase-option field is no property of the product: the question would earn a shrug
        bare_option = P2_ON and re.fullmatch('options?', plain_key(label))
        if not label or bare_option or len(label.split()) > 5:
            continue
        key = plain_key(label)
        if key not in map(plain_key, picked[number]):
            picked[number].append(label)
        if not str(pick.get('hint') or '').strip():
            continue
        hints[number].add(key)
        if number not in order:
            order.append(number)
    return picked, hints, order

def talk_plan_samples(ep, prompt: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The question plan and, when affordable, a second sample from the other model."""
    llm = ep.llm
    ask = lambda models: llm.ask(prompt, models, 3500)
    if not P1_ON:
        return ask([HEAVY, LIGHT]) or {}, None
    # the same PLAN prompt to both models: a property both choose is a surer first question than one only one picks
    if ep.elapsed() >= SIDE_BY_SIDE_UNTIL:
        first = ask([HEAVY])
        if first is not None:
            return first, (ask([LIGHT]) if llm.cost <= SPEND_SOFT else None)
        if llm.stopped():
            return {}, None
        return ask([LIGHT]) or {}, None
    # both samples are requested together, so the second model costs no wait of its own (r15 Arena logs:
    # the two ran one after the other on every task, roughly 8 s per episode)
    with ThreadPoolExecutor(max_workers=2) as workers:
        heavy, light = [workers.submit(llm.ask, prompt, [model], 3500) for model in (HEAVY, LIGHT)]
        first, other = heavy.result(), light.result()
    if first is None:
        return other or {}, None
    return first, other

def talk_carried(ep, number: int, table: list[tuple[str, str, int, list[str]]]) -> dict[str, float]:
    """Share of the item's read listings carrying each of its keys."""
    read = max(1, mem_read_count(ep, number))
    shares = [(key, carried / read) for label, key, carried, _ in table if mem_owner(ep, label) == number]
    own: dict[str, float] = {}
    for key, share in shares:
        own[key] = max(own.get(key, 0.0), share)
    return own

def talk_add_unique(lane: list[str], picks: list[str]) -> list[str]:
    """The lane followed by each pick whose plain spelling it does not hold yet."""
    out = list(lane)
    known = {plain_key(x) for x in out}
    for pick in picks:
        if plain_key(pick) not in known:
            known.add(plain_key(pick))
            out.append(pick)
    return out

def talk_plan(ep, count: int) -> dict[int, list[str]]:
    """Ordered properties to ask about, per item."""
    table = mem_key_table(ep)
    plan, second = talk_plan_samples(ep, talk_plan_prompt(ep, count, table))
    lanes, hinted, order = talk_read_picks(ep, plan)
    known_order = ep.talk.plan_order
    known_order.extend(n for n in order if n not in known_order)
    if second:
        lanes2, hinted2, order2 = talk_read_picks(ep, second)
    else:
        lanes2, hinted2, order2 = {}, {}, []
    known_order.extend(n for n in order2 if n not in known_order)
    for number, picks in list(lanes.items()):
        keys = talk_own_keys(ep, number, table)
        own = talk_carried(ep, number, table)
        hints = set()
        for h in hinted[number] | hinted2.get(number, set()):
            hints.add(plain_key(talk_spelled(h, own)))
        hinted[number] = hints
        lane = talk_add_unique([], [talk_spelled(pick, own) for pick in picks])
        other = lanes2.get(number)
        if other:
            lane = talk_merge_samples(lane, other, own)
        lane = sorted(talk_add_unique(talk_guard_lane(ep, lane, keys, own, hints), keys), key=lambda x: mem_is_catch_all(ep, x))
        lanes[number] = lane
        if P3_ON and lane and count <= 2 and plain_key(lane[0]) in hints:
            talk_measure_swap(ep, number, lane, hints, keys, table, max(1, count // len(ep.items)))
    extra = {} if not second else {'ask2': clip(second.get('ask'), 120)}
    ep.log('plan', ask=clip(plan.get('ask'), 120), lanes={n: v[:8] for n, v in lanes.items()}, **extra)
    return lanes

def talk_spelled(label: str, own: dict[str, float]) -> str:
    """Item's own key for an alternate spelling."""
    plain = lambda text: re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', plain_key(text)))
    wanted = plain(label)
    return next((k for k in own if plain(k) == wanted), label)

def talk_share_of(label: str, own: dict[str, float]) -> float:
    key = plain_key(label)
    if key in own:
        return own[key]
    return max((s for k, s in own.items() if names_same(stems(label), stems(k)) or names_same(stems(k), stems(label))), default=0.0)

def talk_merge_samples(lane: list[str], other: list[str], own: dict[str, float]) -> list[str]:
    """Merge the two planner samples."""
    spelled = [talk_spelled(pick, own) for pick in other]
    theirs = list(map(plain_key, spelled))
    both = [x for x in lane if plain_key(x) in theirs]
    extra: list[str] = []
    for pick in other:
        key = plain_key(talk_spelled(pick, own))
        if key not in [plain_key(x) for x in lane + extra]:
            extra.append(talk_spelled(pick, own))
    extra.sort(key=lambda x: -talk_share_of(x, own))
    return both + [x for x in lane if x not in both] + extra

def talk_guard_place(lane: list[str], key: str, weak: int | None, many: bool) -> list[str]:
    """The lane with the key put in: in the weak place (that pick moved to fourth on a multi-item task), else last."""
    placed = [other for other in lane if plain_key(other) != plain_key(key)]
    if weak is None:
        placed.append(key)
    elif many:
        moved = placed[weak]
        placed[weak] = key
        placed.insert(min(3, len(placed)), moved)
    else:
        placed.insert(min(len(placed), weak), key)
    return placed

def talk_guard_lane(ep, lane: list[str], keys: list[str], own: dict[str, float], hinted: set[str]) -> list[str]:
    """Let widely carried keys into the top three."""
    strong = []
    for k in keys:
        if len(strong) == GUARD_KEYS:
            break
        if own.get(k, 0.0) >= GUARD_SHARE and not mem_is_catch_all(ep, k):
            strong.append(k)
    for key in strong:
        top = lane[:3]
        if plain_key(key) in set(map(plain_key, top)):
            continue
        need = own.get(key, 0.0)
        weak = None
        for place, other in enumerate(top):
            if plain_key(other) not in hinted and talk_share_of(other, own) + GUARD_GAP <= need:
                weak = place
                break
        if len(lane) > 2 and weak is None:
            continue
        placed = talk_guard_place(lane, key, weak, len(ep.items) > 1)
        kept = set(map(plain_key, placed[:3]))
        if all(plain_key(x) not in hinted or plain_key(x) in kept for x in top):
            lane = placed
    return lane

def talk_measure_swap(ep, number: int, lane: list[str], hinted: set[str], keys: list[str], table: list[tuple[str, str, int, list[str]]], room: int) -> None:
    """Move a measure-valued hinted key off first place."""
    def measure(vals: list[str]) -> bool:
        return len(vals) > 0 and 2 * len([v for v in vals if re.match(MEASURE_VALUE, v)]) > len(vals)
    mine = [(key, values) for label, key, _, values in table if mem_owner(ep, label) == number]
    heads = [values for key, values in mine if key == plain_key(lane[0])]
    if not measure(heads[0] if heads else []):
        return
    shown = dict(mine)
    going = set(map(plain_key, lane[:room]))
    for k in keys:
        if plain_key(k) in hinted or plain_key(k) in going or measure(shown.get(k, [])) or mem_is_catch_all(ep, k):
            continue
        lane[:] = [k] + [x for x in lane if plain_key(x) != plain_key(k)]
        ep.log('measure_first', item=number, first=k, moved=lane[1])
        return

def talk_send(ep, questions: list[tuple[int, str]], floor: int = 6) -> list[tuple[int, str, str]]:
    tool = ep.roles['ask']
    sent: list[tuple[int, str, str]] = []
    for number, text in questions:
        if ep.finished or ep.steps_left() <= floor or ep.msg_sent >= ep.msg_cap:
            break
        args = talk_msg_args(ep, text)
        if not args:
            break
        reply = ep.store.call(tool, args)
        if reply is None:
            break
        ep.msg_sent += 1
        answer = reply.get('observation')
        words = talk_words_of(answer)
        sent.append((number, text, words))
        note = f'step {ep.step}: asked the shopper {text!r}, reply {words[:300]!r} (cart at that time: {ep.cart_stamp()})'
        gates_note_line(ep, note, ep.step)
        ep.log('ask', q=text, a=words[:300])
        # a reply carrying an error ends the round of questions
        if isinstance(answer, dict) and answer.get('error'):
            break
    return sent

def talk_budget_as_written(ep, value: float, said: str) -> Any:
    """Budget as the answers actually write it."""
    figures = set()
    for raw in re.findall('\\d[\\d,]*(?:\\.\\d+)?', said):
        figures.add(float(raw.replace(',', '')))
    figures.discard(0.0)
    if any(abs(value - f) <= 1e-09 * max(1.0, f) for f in figures):
        return value
    # a figure off by a power of ten is the same budget read in another unit
    shifted = []
    for f in figures:
        if any(abs(f / (value * 10 ** k) - 1) < 0.002 for k in (-3, -2, -1, 1, 2, 3)):
            shifted.append(f)
    old = ep.budget if is_number(ep.budget) and ep.budget > 0 else None
    if not shifted:
        ep.log('budget_unwritten', read=value)
        return ep.budget
    if len(shifted) != 1:
        return value
    written = shifted[0]
    if old:
        gap = lambda x: max(x / old, old / x)
        if gap(written) >= gap(value):
            return value
    ep.log('budget_written', read=value, written=written)
    return int(written) if written.is_integer() else written

def talk_apply_answers(ep, heard: dict[str, Any], told: list[str], said: str = '') -> None:
    size = len(ep.items)
    item_of = lambda entry: ep.items[pick_number(entry.get('item'), size) - 1]
    for field in ('needs', 'drop', 'free'):
        for entry in as_list(heard.get(field)):
            if not isinstance(entry, dict):
                continue
            if field == 'needs':
                need = str(entry.get('need') or '').strip()
                if need:
                    number = pick_number(entry.get('item'), size)
                    added = ep.items[number - 1].added
                    if need not in added:
                        added.append(need)
                        told.append(need if size <= 1 else f'item {number}: {need}')
                    if (number, need) not in ep.refs and entry.get('points') is True:
                        ep.refs.append((number, need))
            elif field == 'drop':
                if str(entry.get('property') or '').strip():
                    item = item_of(entry)
                    gone = plain_key(str(entry['property']).split(':')[0])
                    hit = list(filter(lambda m: plain_key(m.split(':')[0]) == gone, item.live()))
                    item.dropped += hit
                    told.extend('dropped ' + m for m in hit)
            elif entry.get('property'):
                item_of(entry).free.append(str(entry['property']))
    budget = heard.get('budget')
    if is_number(budget) and budget > 0:
        ep.budget = talk_budget_as_written(ep, budget, said)

def talk_read_answers(ep, sent: list[tuple[int, str, str]]) -> dict[str, Any]:
    items = '\n'.join('item ' + str(n) + ': ' + it.name for n, it in enumerate(ep.items, 1))
    asked = []
    for i, (n, q, a) in enumerate(sent, 1):
        about = f'about item {n}' if n else 'about the items it names'
        asked.append(f'{i}. ({about}) Q: {q}\nA: {a}')
    prompt = HEAR + '\n\nITEMS:\n' + items + '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join(asked)
    return ep.llm.ask(prompt, [LIGHT, HEAVY], 2500) or {}

def talk_texts_in(node: Any, depth: int = 0) -> list[str]:
    """Multi-word strings of a nested reply, depth first, at most four levels down."""
    if depth > 4:
        return []
    if isinstance(node, str):
        return [node] if len(node.split()) >= 2 else []
    children = node.values() if isinstance(node, dict) else node if isinstance(node, list) else ()
    return [text for child in children for text in talk_texts_in(child, depth + 1)]

def talk_words_of(observation: Any) -> str:
    """Multi-word texts of a reply, in order."""
    if isinstance(observation, str):
        return observation
    found = talk_texts_in(observation)
    return ' '.join(found)[:800] if found else dumps(observation)[:400]

class ChangeWatcher:
    """Notes inside results and what they change."""

    def __init__(self) -> None:
        self.heard: set[str] = set()
        self.sold_lines: set[str] = set()
        self.sold_flag = False
        self.named_gone = False

def news_free_lines(ep, observation: Any) -> list[str]:
    """Unread sentences of a result."""
    if not isinstance(observation, dict) or mem_identify(ep, observation):
        return []
    long = lambda text, words: len(text.split()) >= words
    lines = []
    for value in observation.values():
        if isinstance(value, str):
            found = [value] if long(value, 5) else []
        elif isinstance(value, list):
            texts_only = bool(value) and all(isinstance(v, str) for v in value)
            found = [v for v in value if long(v, 5)] if texts_only else []
        elif isinstance(value, dict) and not mem_identify(ep, value, False, 1):
            found = [v for v in value.values() if isinstance(v, str) and long(v, 6)]
        else:
            found = []
        lines.extend(line for line in found if line not in ep.news.heard)
    return lines

def news_read_note(ep, lines: list[str]) -> str:
    """Read a note with a model and apply it."""
    news = ep.news
    news.heard.update(lines)
    sections = (NOTE, 'ITEMS AND REQUIREMENTS NOW:\n' + req_ledger(ep), 'THE NOTE:\n' + '\n'.join(lines))
    got = ep.llm.ask('\n\n'.join(sections), [LIGHT, SPARE], 2500) or {}
    gone = list(map(str, filter(lambda g: isinstance(g, (str, int)), as_list(got.get('gone')))))
    known = list(filter(ep.mem.rows.__contains__, gone))
    ep.unavailable.update(known)
    news.sold_flag = got.get('sold_out') is True if not gone else True
    news.named_gone = len(known) > 0
    if len(lines) == 1 and news.sold_flag:
        news.sold_lines.update(lines)
    if got.get('kind') is not None and got.get('kind') != 'change':
        return ''
    changed = news_new_budget(ep, got.get('budget')) + news_new_needs(ep, got)
    if gone:
        changed.append('unavailable: ' + ', '.join(gone))
    if got.get('price_moved') is True:
        changed.append('a price changed')
    summary = str(got.get('summary') or '').strip()
    cheapest = ep.cheapest()
    if cheapest and ep.items:
        for entry in as_list(got.get('drop')):
            if isinstance(entry, dict) and str(entry.get('need') or '').strip():
                ep.repick.add(pick_number(entry.get('item'), len(ep.items)))
        # a sold-out line re-opens its item the same way: the next cheapest fit may be a row never read (r15 b05: 3 of 18
        # not-best orders followed a sold-out refusal)
        cat_key = mem_kind_key(ep)
        if cat_key and (gone or news.sold_flag):
            owners = (mem_owner(ep, mem_label(ep, i, cat_key)) for i in [i for i in basket_ids(ep) if ep.is_gone(i)] or known)
            ep.repick.update(filter(None, owners))
    # a lower budget keeps the cheapest fit cheapest while its line stays inside it (real races 5-14: a still-fitting
    # pick kept after a budget note passed .93, dropped .67), so only a line it breaks reopens the hunt
    held = basket_ids(ep)
    cap = ep.numeric_budget()
    prices = list(map(lambda i: mem_price(ep, i), held))
    broken = not held or cap is None or None in prices or sum(p * max(1, basket_units(ep, i)) for i, p in zip(held, prices)) > cap
    reasons = gone or news.sold_flag or any(c.startswith('dropped ') for c in changed)
    if ep.cheapest() and (reasons or (broken and any(c.startswith('budget now') for c in changed))):
        scout_reopen_cheapest(ep)
    if changed or summary:
        ep.version += 1
        ep.recent = []
        ep.notes.append(summary or '; '.join(changed))
    return f"{summary} ({'; '.join(changed)})".strip() if changed else summary

def news_new_budget(ep, budget: Any) -> list[str]:
    if not is_number(budget) or budget <= 0 or budget == ep.budget:
        return []
    ep.budget = budget
    return [f'budget now {budget}']

def drop_targets(live: list[str], need: str) -> list[str]:
    """Live requirements a dropped one removes: the same text, else the first on the same property, else itself."""
    exact = [m for m in live if m == need]
    if exact:
        return exact
    head = plain_key(need.split(':')[0])
    return [m for m in live if plain_key(m.split(':')[0]) == head][:1] or [need]

def news_new_needs(ep, got: dict[str, Any]) -> list[str]:
    """Apply dropped and added requirements."""
    count = len(ep.items)

    def entries(key: str) -> list[tuple[Any, str]]:
        out = []
        for e in as_list(got.get(key)):
            if isinstance(e, dict) and str(e.get('need') or '').strip():
                out.append((ep.items[pick_number(e.get('item'), count) - 1], str(e['need']).strip()))
        return out
    drops, adds = entries('drop'), entries('add')
    for item, need in drops:
        item.dropped.extend(drop_targets(item.live(), need))
    for item, need in adds:
        if need in item.dropped:
            item.dropped.remove(need)
        if need not in item.musts and need not in item.added:
            item.added.append(need)
    return ['dropped ' + need for _, need in drops] + ['added ' + need for _, need in adds]

def news_store_note(ep, seen: Any) -> str:
    ep.news.sold_flag = False
    lines = news_free_lines(ep, seen)
    if not lines:
        return ''
    meaning = news_read_note(ep, lines)
    return f'[store note] {meaning}. Apply this change to your picks before you check out.' if meaning else ''

def news_after_refusal(ep, job: Call, reply: dict[str, Any]) -> str:
    """Re-read cart lines after a refusal."""
    news = ep.news
    buying = ep.mode_ok('buy', job.name, job.args)
    refused = buying and not reply.get('finished')
    if news.sold_flag:
        due = refused or not (news.named_gone or buying)
    else:
        due = refused and any(line in talk_words_of(reply.get('observation')) for line in news.sold_lines)
    found = news_locate_sold(ep) if due and basket_ids(ep) and not ep.finished else ''
    news.sold_flag = False
    return "[Desk re-read the cart's lines after the refusal]\n" + found if found else ''

def news_flush_queue(ep) -> str:
    if not ep.queued:
        return ''
    text = "[notes seen on Desk's own calls] " + ' | '.join(ep.queued)
    ep.queued = []
    return text

def news_swap_hint(ep) -> str:
    """Mark sold-out lines and suggest swaps."""
    lost = [i for i in basket_ids(ep) if ep.is_gone(i) and i not in ep.gone]
    if not lost:
        return ''
    for ident in lost:
        ep.gone[ident] = ep.step
    ep.leads = [i for i in ep.leads if i not in lost]
    ep.version += 1
    picks = news_replacements(ep, lost)
    text = f"[Desk] Sold out now: {', '.join(lost)} (the store marks it unbuyable). It stays gone, so do not open it again."
    if not picks:
        return text + ' Take that line out of the cart and add the best other fitting listing instead; search for one when nothing read so far fits.'
    first, *rest = picks
    listed = ', '.join(picks)
    if ep.steps_left() <= 6:
        advice = (f' Steps are short: this turn, cart {first} at the requested quantity and drop the sold-out line (two calls), then check out. '
                  f"Other read listings that look like fits: {', '.join(rest) or 'none'}.")
    elif ep.cheapest():
        advice = (f" Read listings that look like fits for that item, cheapest first: {listed}. Since the pick is the cheapest, unless the "
                  f"whole fitting set is already seen, search once below the price of {first} (with the search's maximum price, when it has one) "
                  'before replacing the sold-out line with the cheapest fitting listing.')
    else:
        advice = (f" Read listings that appear to fit that item instead: {listed}. In this turn take the sold-out line out of the cart "
                  f'and add {first} at the requested quantity, then check out.')
    return text + advice

def news_locate_sold(ep) -> str:
    ids = [i for i in basket_ids(ep) if i not in ep.gone]
    if not ids or any(ep.is_gone(i) for i in ids):
        return ''
    if len(ids) > 1:
        return news_refresh_offers(ep)
    ep.unavailable.add(ids[0])
    return ''

def news_replacements(ep, lost: list[str]) -> list[str]:
    """Replacement ids for sold-out lines."""
    mem = ep.mem
    if not ep.llm.models_for():
        return []
    pool = []
    for ident in reversed(mem.order):
        if len(pool) == 25:
            break
        if ident not in lost and ident not in ep.gone and not ep.is_gone(ident):
            pool.append(ident)
    if not pool:
        return []
    records = '\n'.join(dumps(clip(as_stated(mem.pages.get(i) or mem.rows.get(i)), 200))[:1200] for i in pool)
    blocks = {SWAP: '', 'REQUEST:\n': ep.request, 'REQUIREMENTS IN FORCE:\n': req_ledger(ep), 'SOLD-OUT CART LINE:\n': '\n'.join(f'{i}: {mem_record_text(ep, i, True)}' for i in lost),
              'CART NOW:\n': basket_describe(ep), 'OTHER LISTING RECORDS SEEN (newest first):\n': records}
    got = ep.llm.ask('\n\n'.join(head + body for head, body in blocks.items()), ep.llm.models_for(), 3500) or {}
    taken = {*lost, *ep.gone, *basket_ids(ep)}
    first: dict[str, Any] = {}
    for entry in as_list(got.get('fits')):
        ident, price = (str(entry.get('id')), entry.get('price')) if isinstance(entry, dict) else (str(entry), None)
        if ident not in taken and ident in mem.rows and ident not in first:
            first[ident] = price
    fits = [(float(price) if is_number(price) else float('inf'), ident) for ident, price in first.items()]
    ranked = sorted(fits, key=lambda f: f[0]) if ep.cheapest() else fits
    ep.log('swap', lost=lost, fits=fits)
    return [ident for _, ident in ranked[:4]]

def page_reads(ep, tool: str, target: str, ids: list[str]) -> Any:
    """Open each id with a single-listing tool, yielding (id, observation) until a call fails or the episode ends."""
    for ident in ids:
        args = shaped({target: ident}, ep.schemas.get(tool))
        reply = None if args is None else ep.store.call(tool, args)
        if reply is None or ep.finished:
            return
        yield ident, reply.get('observation')

def news_refresh_offers(ep) -> str:
    """Re-read stale cart lines; changed ones."""
    rows = ep.mem.rows
    tool = ep.offer_tool()
    target = ep.id_param(tool)
    if not tool or not target:
        return ''
    # a line read since its own carting and the last refusal is current: b09 re-read every line at each close
    # attempt (Arena r12: 3 extra steps per attempt in clock-bound episodes, a last line then bought unread at the clock)
    def stale_since(ident: str) -> bool:
        carted = min(ep.step - 2, ep.added_at.get(ident, ep.basket.changed_at))
        return ident not in ep.gone and ep.offer_seen.get(ident, -99) < max(carted, ep.refused_at)
    stale = list(filter(stale_since, basket_ids(ep)))
    if not stale or len(stale) + 3 > ep.steps_left():
        return ''
    lines = []
    for ident, observation in page_reads(ep, tool, target, stale):
        # the store call itself leaves the rows alone, so this is still the row as it was before the re-read
        before = dict(rows.get(ident) or {})
        ep.offer_seen[ident] = ep.step
        ep.ingest(tool, observation)
        after = rows.get(ident) or {}
        moved = [k for k in after if k in before and before[k] != after[k] and not isinstance(after[k], (dict, list))]
        if moved:
            lines.append(f"{ident} changed ({', '.join(moved[:4])}): " + dumps(clip(observation, 300))[:900])
    ep.log('reread', ids=stale, changed=len(lines))
    if lines:
        ep.version += 1
    return '\n'.join(lines)

class Checker:
    """Judgements on listings and carts."""

    def __init__(self) -> None:
        self.checked: set[str] = set()
        self.verdicts: dict[str, bool] = {}
        self.form_seen: set[tuple[str, str]] = set()
        self.audited = False
        self.hunted = False
        self.closest: dict[int, list[tuple[int, float, str, list[str]]]] | None = None
        self.closest_ids: set[str] = set()
        self.refs: list[str] = []
        self.refs_read: set[str] = set()

def judge_evidence(ep, observation: Any, head: bool = True) -> str:
    """Keys matching requirements, per read listing."""
    ids: list[str] = []

    def visit(node: Any, depth: int) -> None:
        if depth > 6:
            return
        if isinstance(node, dict):
            ident = mem_identify(ep, node, False, depth)
            if ident and fact_block(node) and ident not in ids:
                ids.append(ident)
                return
            node = list(node.values())
        for child in node if isinstance(node, list) else ():
            visit(child, depth + 1)
    visit(observation, 0)
    needs = ep.live_needs()
    if not (ids and needs):
        return ''
    many = len(ep.items) > 1
    tagged = [(f'(item {number}) ' * many, need) for number, need in needs[:8]]
    lines = list(filter(None, (judge_evidence_line(ep, ident, tagged) for ident in ids[:12])))
    if not lines:
        return ''
    head_text = ("[evidence: keys whose name holds a requirement's property. A key about some other property (another part or aspect) decides "
                 'nothing; a key named differently may still hold that same property] ') if head else ''
    return head_text + ' | '.join(lines)

def judge_evidence_line(ep, ident: str, tagged: list[tuple[str, str]]) -> str:
    """One read listing's keys for each tagged requirement, with values said in another form; '' when none."""
    block = fact_block(ep.mem.pages.get(ident) or {}) or {}
    parts = []
    for tag, need in tagged:
        prop = need.split(':', 1)[0].strip()
        hits = [f'{k}={str(v)[:50]}' for k, v in block.items() if names_same(stems(prop), stems(k))]
        if hits:
            warn = ' [not the required value: its words appear separated or reordered there]' if judge_apart_needs(ep, ident, [need]) else ''
            parts.append(f'{tag}{prop} -> ' + ', '.join(hits[:3]) + warn)
        parts += [tag + said for said in judge_form_said(ep, ident, need)]
    return f'{ident}: ' + '; '.join(parts) if parts else ''

def judge_form_said(ep, ident: str, need: str) -> list[str]:
    """Attributes stating the value in another form."""
    prop, _, raw = need.partition(':')
    value = raw.strip()
    block = fact_block(ep.mem.pages.get(ident) or {}) or {}
    if not (value and block) or judge_apart_needs(ep, ident, [need]):
        return []
    want = plain_words(value)
    for v in block.values():
        if any(plain_words(part) == want for part in [str(v), *re.split('[,/;|]', str(v))]):
            return []
    hits = [f'{k}={str(v)[:50]}' for k, v in block.items() if same_in_other_form(value, v)]
    if hits:
        judge_form_logged(ep, 'form_match', ident, need, hits)
        note = ' [this attribute states the required value: the words are the same, just joined, hyphenated or inflected another way]'
        return [f'{prop.strip()} "{value}" -> ' + ', '.join(hits[:2]) + note]
    within = judge_form_within(prop, value, block)
    if not within:
        return []
    judge_form_logged(ep, 'form_within', ident, need, within)
    return [f'{prop.strip()} "{value}" -> ' + ', '.join(within[:2]) + ' [holds the required words whole within a longer value: this '
            'attribute states that value]']

def judge_form_logged(ep, kind: str, ident: str, need: str, hits: list[str]) -> None:
    """Trace a form finding once per listing and requirement."""
    seen = ep.judge.form_seen
    if (ident, need) not in seen:
        seen.add((ident, need))
        ep.log(kind, id=ident, need=need, hits=hits[:2])

def judge_form_within(prop: str, value: str, block: dict[str, Any]) -> list[str]:
    """Values of the property's own key holding the required words whole and in order inside a longer value."""
    # the required words whole and in order inside a longer value of the property's own key (real r16: 6 of 6 paid
    # pages for a required "sd" said "micro sd"); a hyphen joins one word, so a part of a compound never counts
    # (real r12-r16: 1,100+ fact values such as "x-free" or "non-x" would flip meaning)
    whole = lambda text: re.findall('[a-z0-9]+(?:-[a-z0-9]+)*', str(text).lower())
    wanted = whole(value)
    if not wanted or re.search('\\d', value) or has_bound(value):
        return []
    width = len(wanted)
    out = []
    for k, v in block.items():
        if not names_same(stems(prop), stems(k)):
            continue
        words = whole(v)
        if len(words) > width and any(words[i:i + width] == wanted for i in range(len(words) - width + 1)):
            out.append(f'{k}={str(v)[:50]}')
    return out

def judge_form_lines(ep, ids: list[str]) -> list[str]:
    tag = (lambda n: f'(item {n}) ') if len(ep.items) > 1 else (lambda n: '')
    said = [(ident, [tag(n) + line for n, need in ep.live_needs()[:8] for line in judge_form_said(ep, ident, need)]) for ident in ids]
    return [f'{ident}: ' + '; '.join(found) for ident, found in said if found]

def judge_apart_needs(ep, ident: str, needs: list[str] | None = None) -> list[str]:
    """Values held only separated or reordered."""
    block = fact_block(ep.mem.pages.get(ident) or {}) or {}
    found = []
    for need in [n for _, n in ep.live_needs()] if needs is None else needs:
        prop, _, value = need.partition(':')
        shapes = {spread(value.strip(), v) for k, v in block.items() if names_same(stems(prop), stems(k))}
        if 'apart' in shapes and 'phrase' not in shapes:
            found.append(need)
    return found

def judge_against(ep, ident: str, needs: list[str]) -> bool:
    """A plain requirement the page contradicts."""
    block = fact_block(ep.mem.pages.get(ident) or {}) or {}
    for need in needs:
        prop, _, value = need.partition(':')
        if re.search('\\d|\\b(?:not|no|other than|unlike|same|than|like|only)\\b', value, re.I):
            continue
        words = set(plain_words(value)) - {'or', 'either', 'and'}
        held = [v for k, v in block.items() if names_same(stems(prop), stems(k))]
        if words and held and all(not words & set(plain_words(v)) for v in held):
            return True
    return False

def judge_fit_text(ep, ident: str) -> str:
    """A record's text for fit codes."""
    record = ep.mem.pages.get(ident) or ep.mem.rows.get(ident) or {}
    loose = ' '.join(str(v) for v in record.values() if isinstance(v, str))
    inner = '. '.join(f'{k} {v}' for sub in record.values() if isinstance(sub, dict) for k, v in sub.items() if isinstance(v, str))
    return loose + ' ' + inner

def judge_fit_index(ep) -> dict[str, Any]:
    """Fit marks of all seen listings."""
    marks = {ident: fit_marks(judge_fit_text(ep, ident)) for ident in ep.mem.order}
    notes: dict[str, set[str]] = {}
    for part, codes in (pair for mark in marks.values() for pair in mark['also']):
        notes.setdefault(part, set()).update(codes)
    # one device sold under several names ("Model number: X. Also sold as Y"): each name maps to all of them
    names: dict[str, set[str]] = {}
    for mark in marks.values():
        group = set(mark['codes']).union(*(names.get(code, set()) for code in mark['codes']))
        names.update(dict.fromkeys(group, group))
    known = {code for mark in marks.values() for code in mark['codes']}
    return {'marks': marks, 'notes': notes, 'known': known, 'names': names}

def judge_fit_set(ident: str, index: dict[str, Any]) -> set[str]:
    mark = index['marks'].get(ident) or {}
    named = codes_named(mark.get('works') or [], index.get('known') or set())
    return named | index['notes'].get(mark['own'], set()) if mark.get('own') else named

def judge_fit_faults(ep) -> list[str]:
    if not ep.bound:
        return []
    index = judge_fit_index(ep)
    cart = basket_ids(ep)
    faults = []
    for part, main in ep.bound:
        devices = [i for i in cart if i in mem_ids_of_item(ep, main)]
        lines = [i for i in cart if i in mem_ids_of_item(ep, part)]
        if len(devices) != 1 or not lines:
            continue
        codes = judge_fit_codes(index, devices[0])
        for line in lines if codes else []:
            text = judge_fit_line(ep, index, cart, (part, main), (devices[0], codes), line)
            if text:
                faults.append(text)
    return faults

def judge_fit_codes(index: dict[str, Any], ident: str) -> set[str]:
    """Codes a listing carries or takes, by the fit index."""
    mark = index['marks'].get(ident, {})
    return set(mark.get('codes') or []) | set(mark.get('takes') or [])

def judge_fit_line(ep, index: dict[str, Any], cart: list[str], pair: tuple[int, int], device: tuple[str, set[str]], line: str) -> str:
    """Why one part line does not work with the carted device, with the seen alternatives; '' when it does."""
    fits = judge_fit_set(line, index)
    host, codes = device
    if not fits or fits & codes:
        return ''
    part, main = pair
    item = ep.items[part - 1]
    named_codes, listed = ', '.join(sorted(codes)), ', '.join(sorted(fits)[:4])
    rest = (index['marks'].get(line) or {}).get('more')
    if rest and rest not in {m.get('own') for m in index['marks'].values()}:
        # its own page leaves the device out and sends the rest of its list to a page not seen: a seen part
        # naming the device on its own page is the surer pick (real race 13: orders whose part fit only
        # through such a page paid 3 of 86, parts naming the device 13 of 94)
        needs = item.live()

        def surer(i: str) -> bool:
            if i not in ep.mem.pages or not judge_fit_set(i, index) & codes or not mem_can_buy(ep, i):
                return False
            if i in ep.gone or i in ep.basket.dropped:
                return False
            return not (judge_apart_needs(ep, i, needs) or judge_against(ep, i, needs)) and i not in cart
        named = list(filter(surer, mem_ids_of_item(ep, part)))[:4]
        return (f"{line} ({item.name}) does not name {host} ({named_codes}) on its own page, which lists "
                f"{listed} and sends the rest to part {rest}; seen parts of that kind naming it on their own page: "
                f"{', '.join(named)}. If one of them meets the item's other requirements, keep the device and swap this part for it") if named else ''
    good = [i for i in mem_ids_of_item(ep, part) if judge_fit_set(i, index) & codes and not ep.is_gone(i) and i not in cart][:4]
    hosts = [i for i in mem_ids_of_item(ep, main) if judge_fit_codes(index, i) & fits and not ep.is_gone(i)][:4]
    text = f"{line} ({item.name}) works with {listed}, not with {host} ({named_codes})"
    if good:
        text += f"; seen parts of that kind that work with it: {', '.join(good)}"
    else:
        text += '; none of the parts of that kind seen so far works with it, so either find one or change the device: ' + judge_fit_plan(ep, short=True)
    return text + (f"; seen {ep.items[main - 1].name} listings this part works with: {', '.join(hosts)}" if hosts else '')

def judge_fit_plan(ep, short: bool = False) -> str:
    """Devices named by part listings, ranked."""
    if not ep.bound:
        return ''
    index = judge_fit_index(ep)
    lines = list(filter(None, (judge_fit_plan_line(ep, index, main, short) for main in sorted({m for _, m in ep.bound}))))
    if lines and not short:
        return ("[fit plan: the device is chosen by whichever part item has the fewest listings left under its own requirements. A part "
                "works with a device if the part's text names the device's model code or one of its sale names (names joined by = are one "
                "device). Searches miss codes, so list the device category and check every row for codes] " + ' | '.join(lines))
    return ' | '.join(lines)

def judge_fit_plan_line(ep, index: dict[str, Any], main: int, short: bool) -> str:
    """Codes the parts of one device item name, the most-named first, with the devices seen for each."""
    names = index.get('names') or {}
    tally: dict[str, dict[int, list[str]]] = {}
    for p in (p for p, m in ep.bound if m == main):
        needs = ep.items[p - 1].live()
        for i in mem_ids_of_item(ep, p):
            if ep.is_gone(i) or judge_against(ep, i, needs):
                continue
            for key in {' = '.join(sorted(names.get(code) or {code})) for code in judge_fit_set(i, index)}:
                tally.setdefault(key, {}).setdefault(p, []).append(i)
    ranked = sorted(tally.items(), key=lambda kv: (-len(kv[1]), -sum(len(v) for v in kv[1].values())))[:4]
    if not ranked:
        return ''
    chunks = []
    for key, per in ranked:
        wanted = set(key.split(' = '))
        hosts = [i for i in mem_ids_of_item(ep, main) if wanted & judge_fit_codes(index, i)][:2]
        served = ', '.join(f'{len(ids)} {ep.items[p - 1].name}' + ('' if short else f" ({', '.join(ids[-2:])})") for p, ids in sorted(per.items()))
        chunks.append(f"{key}: {served} " + (f"(device seen: {', '.join(hosts)})" if hosts else '(device not seen yet)'))
    return f"codes the {ep.items[main - 1].name} candidates' parts name -> " + '; '.join(chunks)

def judge_line_faults(ep) -> list[str]:
    """Cart line problems found in code."""
    cat_key = mem_kind_key(ep)
    held = [i for i in basket_ids(ep) if (basket_units(ep, i) or 0) > 0]
    if not (held and cat_key):
        return []
    kind_of = {i: mem_label(ep, i, cat_key) for i in held}
    claims = {i: [n for n, item in enumerate(ep.items, 1) if mem_belongs(ep, item, kind_of[i])] for i in held}
    labels = {str(r.get(cat_key) or '') for r in ep.mem.rows.values()}
    listed = [n for n, item in enumerate(ep.items, 1) if any(mem_belongs(ep, item, label) for label in labels)]
    out = [fault for n in listed for fault in judge_item_faults(ep, n, held, claims)]
    if len(listed) == len(ep.items):
        out.extend(f"{i} ({kind_of[i] or 'unknown kind'}) is no kind the shopper asked for: take it out" for i in held if not claims[i])
    out += judge_shelf_faults(ep, held, claims, kind_of, labels)
    out.extend(f'{i} is sold out' for i in held if ep.is_gone(i))
    return out + judge_budget_faults(ep, held)

def judge_item_faults(ep, n: int, held: list[str], claims: dict[str, list[int]]) -> list[str]:
    """Item n's cart faults: no line, more than one line of its kind alone, a line at the wrong count."""
    item = ep.items[n - 1]
    mine = [i for i in held if n in claims[i]]
    if not mine:
        return [f'item {n} ({item.name}) has no cart line']
    only = [i for i in mine if claims[i] == [n]]
    out = [f"item {n} ({item.name}) has {len(mine)} lines ({', '.join(mine)}); keep one"] if len(only) > 1 else []
    return out + [f'{i} is held at {basket_units(ep, i)} unit(s); item {n} ({item.name}) wants {item.qty}' for i in only if basket_units(ep, i) != item.qty]

def judge_shelf_faults(ep, held: list[str], claims: dict[str, list[int]], kind_of: dict[str, str], labels: set[str]) -> list[str]:
    """Lines of one item taken from a label other than the store's own category for it."""
    # real race 15: in all 39 tasks with a pass, every passing cart took each item from one same category label; 51
    # orders took an item from a neighbouring label (a wider one holding the same words) and all failed
    spelled = {plain_key(label): label for label in labels if label}
    out = []
    for i in held:
        if len(claims[i]) != 1:
            continue
        n = claims[i][0]
        item = ep.items[n - 1]
        shelf = plain_key(item.shelf or '')
        if not shelf or shelf not in spelled or plain_key(kind_of[i]) in (shelf, plain_key(item.name)):
            continue
        out.append(f"{i} is listed under {kind_of[i]!r}, not under {spelled[shelf]!r}, the store's category for item {n} "
                   f'({item.name}): replace it with a listing of that category')
    return out

def judge_budget_faults(ep, held: list[str]) -> list[str]:
    prices = [mem_price(ep, i) for i in held]
    cap = ep.budget
    if not isinstance(cap, (int, float)) or None in prices:
        return []
    if ep.brief.get('budget_scope') == 'each':
        return [f'{i} costs {p:g}, over the per-item limit of {cap:g}' for i, p in zip(held, prices) if cap < p <= 10 * cap]
    total = sum(p * basket_units(ep, i) for i, p in zip(held, prices))
    return [f'the total of {total:g} is over the budget of {cap:g}'] if cap + 1e-09 < total <= 10 * cap else []

def judge_check_key(ep, ids: list[str]) -> str:
    return '|'.join(sorted(ids)) + f'#{ep.version}'

def judge_check_cart(ep, ids: list[str] | None = None, limit: float = SPEND_SOFT) -> str:
    """Cart review, once per cart and version."""
    judge = ep.judge
    lead = ids is not None
    if not lead:
        ids = basket_ids(ep)
    key = judge_check_key(ep, ids)
    if not ids or key in judge.checked:
        return ''
    apart = [(i, need) for i in ids for need in judge_apart_needs(ep, i)] if ep.steps_left() > ep.reserve() + 4 else []
    if not (apart or ep.llm.models_for(limit)):
        return ''
    judge.checked.add(key)
    if apart:
        judge.verdicts[key] = False
        ep.leads = [i for i in ep.leads if i not in ids]
        ep.log('check_apart', cart=ids, needs=apart[:4])
        said = ['{}: its {} holds the words of "{}" only separated or reordered, so it is a different value'.format(ident, *map(str.strip, need.split(':', 1)))
                for ident, need in apart[:3]]
        return ('A close review finds this cart still wrong: ' + '; '.join(said) + '. Find a listing whose attribute states the value exactly as '
                'written (search the value with the product noun), swap it in for that line, then check out.')
    got = ep.llm.ask(judge_check_prompt(ep, ids, lead), ep.llm.models_for(limit), 3000)
    ep.log('check', cart=ids, got=got)
    if got and not lead:
        found = (re.search('[A-Za-z]{1,6}[- ]?\\d{2,7}', str(r)) for r in as_list(got.get('refs')))
        judge.refs += [c for c in (m.group(0) for m in found if m) if c not in judge.refs][:4]
    passed = not got or got.get('ok') is not False
    judge.verdicts[key] = bool(got) and passed
    return '' if passed else judge_check_refusal(ep, ids, got)

def judge_check_refusal(ep, ids: list[str], got: dict[str, Any]) -> str:
    """The review's refusal text; the cart's listings stop being leads."""
    ep.leads = [i for i in ep.leads if i not in ids]
    problems = [str(p) for p in as_list(got.get('problems'))][:5] or ['see the requirements']
    better = [b for b in map(str, as_list(got.get('better'))) if b in ep.mem.rows and not ep.is_gone(b)][:4]
    shown = '. Listings that look better: ' + ', '.join(better) if better else ''
    return ('A careful review says this cart is not right yet: ' + '; '.join(problems) + shown
            + '. Correct the cart, or re-read the records to prove the review mistaken, and then check out.')

def judge_check_prompt(ep, ids: list[str], lead: bool) -> str:
    pages = ep.mem.pages
    blocks = [CHECK, f'REQUEST:\n{ep.request}', f'REQUIREMENTS IN FORCE:\n{req_ledger(ep)}']
    if not lead:
        blocks.append(f'CART:\n{basket_describe(ep)}')
    else:
        wanted = '; '.join(f'item {n}: quantity {it.qty}' for n, it in enumerate(ep.items, 1))
        blocks.append(f'CART (planned: assess these listings as if already carted in the requested quantities: {wanted}):\n' + ', '.join(ids))
    sold = 'SOLD OUT (the store says it can no longer be bought). '
    own = [f'{i}: ' + (sold if ep.is_gone(i) else '') + mem_record_text(ep, i, True) for i in ids]
    blocks.append("RECORDS OF THE CART'S LISTINGS:\n" + '\n'.join(own))
    forms = judge_form_lines(ep, ids)
    # a requirement read from other pages can ask for a value related to the one there (by a rule those pages
    # state): the review sees those pages, not only the value copied from the last one. The last page of a chain gives
    # the value; earlier pages only name the next (Arena b10 r14: the review saw only a start page marked 'not this
    # item' and passed its value; a line offering "A or B" passed for the chain's single value B and failed. Real races
    # 12-16: ordered pointer carts whose line offered alternatives or a list on the chain's property passed 5/122,
    # single values 349/2793)
    live = {need for _, need in ep.live_needs()}
    chains = [(need, [i for i in chain if i in pages]) for need, chain in ep.ref_pages.items() if need in live]
    chains = [(need, chain) for need, chain in chains if chain]
    order = [chain[-1] for _, chain in chains] + [i for _, chain in chains for i in chain[-2::-1]]
    linked = [i for i in dict.fromkeys(order) if i not in ids][:4]
    if linked:
        ends = '; '.join(f'{need} -> ' + ' -> '.join(chain) + f' (the value comes from {chain[-1]}: {ep.ref_values.get(need) or "see its page"})' for need, chain in chains)
        shown = '\n'.join(f'{i}: ' + mem_record_text(ep, i, True) for i in linked)
        blocks.append('RECORDS OF THE PAGES THOSE REQUIREMENTS POINT AT. Every chain ends on the page that gives the value; a page that '
                      'sends the reader on to another gives none: ' + ends + ". How the cart's value has to relate to that value follows "
                      "from the requirement's own wording. Where it wants that value, the cart line has to state it whole: an attribute "
                      'offering alternatives ("A or B") or a list holding it among others does not give the single value B. A pairing rule '
                      "on that page sets the value only when the request's own words ask for a counterpart to the page's value (one that "
                      'pairs with or complements it); for the same, an identical or a matching value, or simply what that page states, its own '
                      'value counts.\n' + shown)
    if ep.brief.get('pick') in ('cheapest', 'most', 'least'):
        blocks.append('OTHER LISTINGS SEEN (newest first):\n' + mem_other_records(ep, ids, 15))
        forms += judge_form_lines(ep, [i for i in pages if i not in ids and not ep.is_gone(i)])
    if forms:
        blocks.append('WORD FORMS (attributes Desk read as stating a required value written another way):\n' + '\n'.join(forms[:8]))
    return '\n\n'.join(blocks)

def judge_audit(ep, keep: int | None = None) -> str:
    """Look for fits before an unbought ending."""
    if ep.judge.audited or not ep.llm.models_for():
        return ''
    ep.judge.audited = True
    got = judge_audit_call(ep)
    named = judge_audit_fits(ep, got)
    if not named:
        unread = [i for i in map(str, as_list((got or {}).get('open'))) if i in ep.mem.rows and i not in ep.opened and not ep.is_gone(i)]
        if unread and judge_open_rows(ep, unread, keep):
            got = judge_audit_call(ep)
            named = judge_audit_fits(ep, got)
    if not named:
        return ''
    ep.lead_items = named
    ep.leads = [i for ids in named.values() for i in ids]
    listed = '; '.join(f"item {k}: {', '.join(v)}" for k, v in named.items())
    reason = f" ({got.get('reason')})" if got.get('reason') else ''
    return ('A careful review thinks these listings may fit: ' + listed + reason
            + '. Open each one and buy whatever really fits; finish without a purchase only when none of them fits.')

def judge_audit_call(ep) -> dict[str, Any] | None:
    # read pages are the only records the review can pass and unread rows the only ones it can send the desk to
    # open, so each kind keeps its own room
    mem = ep.mem
    live = [i for i in mem.order[::-1] if not ep.is_gone(i)]
    opened = [f'OPENED {dumps(clip(as_stated(mem.pages[i]), 200))[:1200]}' for i in [i for i in live if i in mem.pages][:20]]
    # unread rows of a requested kind within budget first, by required values shown, then price; not the 30 newest (real
    # r12-r16: 114 paid rows seen and never opened; Arena b10: in-category rows titled as another kind were seen 5-12
    # times and never opened before a decline)
    cat_key, cap = mem_kind_key(ep), ep.numeric_budget()
    owner = lambda i: mem_owner(ep, mem_label(ep, i, cat_key)) if cat_key else 0
    fits_budget = lambda i: cap is None or (mem_price(ep, i) or 0) <= cap
    shown = lambda i: scout_values_shown(ep, i, ep.items[owner(i) - 1].live()) if 0 < owner(i) <= len(ep.items) else 0
    unread = [i for i in live if i not in mem.pages]
    unread.sort(key=lambda i: (not (0 < owner(i) <= len(ep.items)), not fits_budget(i), -shown(i), mem_price(ep, i) or 0.0))
    bare = [f'ROW {dumps(clip(as_stated(mem.rows.get(i)), 120))[:500]}' for i in unread[:30]]
    records = opened + bare
    if not records:
        return None
    prompt = '\n\n'.join([AUDIT, 'REQUEST:\n' + ep.request, 'REQUIREMENTS IN FORCE:\n' + req_ledger(ep),
                          'LISTING RECORDS SEEN (read pages first, then rows; newest first in each):\n' + '\n'.join(records)])
    got = ep.llm.ask(prompt, ep.llm.models_for(), 4000)
    ep.log('audit', got=got)
    return got

def judge_audit_fits(ep, got: dict[str, Any] | None) -> dict[str, list[str]]:
    """Audit fits per item number."""
    fits = (got or {}).get('fits')
    if not isinstance(fits, dict):
        return {}
    named: dict[str, list[str]] = {}
    for key, ids in fits.items():
        number = judge_item_number(ep, key)
        slot = named.setdefault(str(number or key), [])
        fresh = [i for i in map(str, as_list(ids)) if i in ep.mem.rows and not ep.is_gone(i) and i not in slot]
        slot[:] = (slot + fresh)[:4]
    return {k: v for k, v in named.items() if v}

def judge_item_number(ep, key: Any) -> int:
    """Item number a key names (0: none)."""
    count = len(ep.items)
    text = str(key)
    number = whole(key)
    if number is None:
        digits = re.findall('\\d+', text)
        if len(digits) == 1 and re.fullmatch('\\s*(?:item\\s*)?#?\\s*\\d+\\s*', text, re.I):
            number = int(digits[0])
    if number is not None:
        return number if 1 <= number <= count else 0
    said = plain_key(text)
    hit = next((n for n, item in enumerate(ep.items, 1)
                if said and (said in (plain_key(item.name), plain_key(item.shelf or '')) or mem_belongs(ep, item, text))), 0)
    return hit or (1 if count == 1 else 0)

def judge_row_plan(ep, rows: list[str]) -> list[tuple[str, dict[str, Any]]]:
    """Calls that would read these rows' pages ([] when no tool can)."""
    many, joiner = ep.side_param(), ep.roles.get('compare')
    if many and joiner and not ep.side_view_bare():
        return [(joiner, {many: rows[:ep.side_room()]})]
    tool = ep.attr_tool()
    target = ep.id_param(tool)
    if not (tool and target):
        return []
    # one at a time the review reads 4 rows while steps and the clock are plentiful (an unread row may be the fit
    # the ending would forgo), else 2
    room = 4 if ep.steps_left() >= 15 and ep.elapsed() <= CLOCK_STOP - 60 else 2
    return [(tool, {target: i}) for i in rows[:room]]

def judge_open_rows(ep, rows: list[str], keep: int | None = None) -> bool:
    plan = judge_row_plan(ep, rows)
    # a late review opens only what still leaves room for its cart lines and the order
    plan = plan if keep is None else plan[:max(0, ep.steps_left() - keep)]
    if not plan or ep.steps_left() < len(plan) + (5 if keep is None else keep):
        return False
    pages = ep.mem.pages
    before = len(pages)
    for name, args in plan:
        call = shaped(args, ep.schemas.get(name))
        if call is None:
            return False
        reply = ep.store.call(name, call)
        if reply is None or ep.finished:
            break
        ep.ingest(name, reply.get('observation'))
    ep.log('audit_open', rows=rows, opened=len(pages) - before)
    return len(pages) > before

def judge_fit_picks(ep) -> dict[int, str]:
    """Per item, a lead or cart line passing checks."""
    cat_key = mem_kind_key(ep)
    carted = basket_ids(ep) if cat_key else []
    usable = lambda i: i in ep.mem.rows and not ep.is_gone(i) and not judge_apart_needs(ep, i)
    picks: dict[int, str] = {}
    for number, item in enumerate(ep.items, 1):
        mine = [i for i in carted if mem_belongs(ep, item, mem_label(ep, i, cat_key))]
        pick = next(filter(usable, ep.lead_items.get(str(number), []) + mine), None)
        if pick:
            picks[number] = pick
    return picks

def judge_offer_note(ep, room: int) -> str:
    """Ending remark naming the nearest matches."""
    if ep.brief.get('quiet_decline') is not False:
        return ''
    picks = judge_fit_picks(ep)
    named = [f'{picks[n]} for the {item.name}' for n, item in enumerate(ep.items, 1) if n in picks]
    if not named:
        return judge_closest_note(ep, room)
    missing = [item.name for n, item in enumerate(ep.items, 1) if n not in picks]
    more = ' No ' + ' or '.join(missing) + ' listing I read satisfies all of its requirements.' if missing else ''
    for size in range(len(named), 0, -1):
        text = 'Ending with no purchase: no listing here satisfies the full order. Nearest matches: ' + '; '.join(named[:size]) + ' (each one satisfies its own item).'
        fitting = [t for t in (text + more, text) if len(t) <= room]
        if fitting:
            return fitting[0]
    return ''

def closest_target(entry: Any, pages: list[str], count: int) -> tuple[str, int] | None:
    """(id, item number) a CLOSEST reply entry names, when both are valid."""
    if not isinstance(entry, dict) or str(entry.get('id')) not in pages:
        return None
    number = whole(entry.get('item')) or (1 if entry.get('item') is None and count == 1 else 0)
    return (str(entry.get('id')), number) if 1 <= number <= count else None

def judge_rank_closest(ep, only: list[str] | None = None) -> dict[int, list[tuple[int, float, str, list[str]]]]:
    """Read listings per item by misses, then price."""
    mem, judge = ep.mem, ep.judge
    cat_key = mem_kind_key(ep)
    ranked = {} if only is None else {n: list(rows) for n, rows in (judge.closest or {}).items()}
    pool = mem.pages if only is None else only
    pages = sorted(filter(lambda i: i in mem.pages and i in mem.rows, pool), key=ep.price_rank)[:16]
    models = ep.llm.models_for(SPEND_HARD)
    if not (pages and models and ep.items):
        return judge.closest or {}
    records = '\n'.join(('' if mem_can_buy(ep, i) else 'SOLD OUT ') + dumps(clip(as_stated(mem.pages[i]), 200))[:1000] for i in pages)
    sections = (CLOSEST, 'REQUEST:\n' + ep.request, 'REQUIREMENTS IN FORCE:\n' + req_ledger(ep), 'LISTING RECORDS:\n' + records)
    got = ep.llm.ask('\n\n'.join(sections), models, 8000)
    judge.closest_ids.update(pages)
    for entry in as_list((got or {}).get('listings')):
        found = judge_closest_row(ep, entry, pages, cat_key)
        if found:
            number, row = found
            ranked[number] = [r for r in ranked.get(number, []) if r[2] != row[2]] + [row]
    judge.closest = ranked = {n: sorted(rows, key=lambda r: r[:2]) for n, rows in ranked.items()}
    ep.log('closest', got={n: [r[:3] for r in rows[:4]] for n, rows in ranked.items()})
    return ranked

def judge_closest_row(ep, entry: Any, pages: list[str], cat_key: str | None) -> tuple[int, tuple[int, float, str, list[str]]] | None:
    """One ranked listing: its item number and (miss count, price, id, misses); None when it names no item's listing."""
    target = closest_target(entry, pages, len(ep.items))
    if not target:
        return None
    ident, number = target
    label = mem_label(ep, ident, cat_key) if cat_key else ''
    if label and mem_owner(ep, label) not in (0, number):
        return None
    price = mem_price(ep, ident)
    misses = [str(m) for m in as_list(entry.get('misses'))]
    if not mem_can_buy(ep, ident):
        misses.append('in stock')
    if price is not None and is_number(ep.budget) and price * max(1, ep.items[number - 1].qty) > ep.budget:
        misses.append('budget')
    return number, (len(misses), float('inf') if price is None else price, ident, misses)

def closest_phrase(name: str, row: tuple[int, float, str, list[str]], unit: str) -> str:
    short, price, ident, misses = row
    cost = '' if price == float('inf') else f'{price:g}{unit}, '
    if not short:
        return f'{ident} for the {name} ({cost}the closest I found)'
    return f"{ident} for the {name} ({cost}it misses only: {'; '.join(misses)[:120]})"

def judge_closest_note(ep, room: int) -> str:
    judge = ep.judge
    # the ranking is a long model call: near the clock it could outlast the episode and lose the ending itself
    if ep.past_stop() or ep.elapsed() > CLOCK_STOP - 40:
        ranked = judge.closest or {}
    elif judge.closest is None:
        ranked = judge_rank_closest(ep)
    else:
        fresh = [i for i in ep.mem.pages if i not in judge.closest_ids]
        ranked = judge_rank_closest(ep, fresh) if fresh else judge.closest
    unit = f" {ep.brief.get('budget_unit')}" if ep.brief.get('budget_unit') else ''
    named = [closest_phrase(item.name, ranked[n][0], unit) for n, item in enumerate(ep.items, 1) if ranked.get(n)]
    while named:
        text = 'Ending with no purchase: no listing satisfies every requirement. Nearest matches: ' + '; '.join(named) + '.'
        if len(text) <= room:
            return text
        named = [n.split(' (', 1)[0] for n in named] if any(' (' in n for n in named) else named[:-1]
    return ''

def judge_capped_try(ep, number: int, kind: str, floor: float) -> bool:
    """Search the item's kind capped at the floor price; False when the store gave no reply."""
    calls = [c for c in ep.scout.cap_calls if kind and kind in c[3]]
    if not calls:
        return True
    tool, key, args, _ = min(calls, key=lambda c: len(c[2]))
    paging = scout_paging_arg(ep, tool)
    capped = {k: v for k, v in args.items() if k != paging}
    capped[key] = int(floor) if float(floor).is_integer() else floor
    call = shaped(capped, ep.schemas.get(tool))
    if call is None or tool + dumps(call) in ep.cache:
        return True
    reply = ep.store.call(tool, call)
    if reply is None:
        return False
    seen = reply.get('observation')
    ep.ingest(tool, seen)
    ep.cache[tool + dumps(call)] = (ep.version, ep.step, ep.render(seen))
    ep.log('closest_search', item=number, args=call, rows=len(mem_listings_in(ep, seen)))
    return True

def judge_closest_hunt(ep) -> None:
    """Look for cheaper near misses before naming one."""
    # rows come in no price order and the offer is judged by price among the closest
    mem, log = ep.mem, ep.scout.search_log
    if ep.judge.hunted or ep.brief.get('quiet_decline') is not False or judge_fit_picks(ep) or ep.finished or ep.elapsed() > CLOCK_STOP - 60:
        return
    ep.judge.hunted = True
    if not mem.price_key:
        # the offer is judged by price and stock among the closest: learn where rows keep them, and which searches
        # capped the price, from the searches made so far
        for name, _, seen in log:
            scout_learn_shape(ep, name, seen)
        for name, args, seen in log:
            scout_learn_cap(ep, name, args, seen)
    ranked = judge_rank_closest(ep)
    cat_key = mem_kind_key(ep)
    read: list[str] = []
    for number in sorted(ranked):
        got = judge_hunt_item(ep, number, ranked[number], cat_key)
        if got is None:
            return
        read.extend(got)
    if read and not ep.past_stop():
        judge_rank_closest(ep, read)

def judge_hunt_item(ep, number: int, rows: list[tuple[int, float, str, list[str]]], cat_key: str | None) -> list[str] | None:
    """For one item, a price-capped search and reads of cheaper unread rows of the nearest kind; None when the store failed."""
    mem = ep.mem
    best = [r for r in rows if r[0] == rows[0][0] and r[1] != float('inf')]
    if not (best and cat_key) or ep.steps_left() < ep.reserve() + 5 or ep.elapsed() > CLOCK_STOP - 30:
        return []
    floor = min(price for _, price, _, _ in best)
    kind = mem_label(ep, best[0][2], cat_key).strip().lower()
    if not judge_capped_try(ep, number, kind, floor):
        return None
    pool = [i for i in mem.order if i not in mem.pages and mem_can_buy(ep, i)
            and mem_label(ep, i, cat_key).strip().lower() == kind and ep.price_rank(i) < floor]
    cheaper = sorted(pool, key=lambda i: mem_price(ep, i) or 0.0)[:CHEAPER_OPENS]
    scout_read_pages(ep, cheaper, ep.reserve() + 2)
    return [i for i in cheaper if i in mem.pages]

class ChainWalker:
    """Requirements whose value sits on another page."""

    def __init__(self) -> None:
        self.rounds = 0
        self.holds = 0
        self.searches = 0
        self.link_seen: dict[str, list[str]] = {}
        # pages already opened when the last trail round was asked, and cart adds held back for an open pointer
        self.read_at: set[str] = set()
        self.cart_holds = 0
        # the last answered trail round and what it saw (pending needs, pages read, rows seen), reused while unchanged
        self.last_got: dict[str, Any] | None = None
        self.last_pending: tuple[str, ...] = ()
        self.last_pages: set[str] = set()
        self.last_rows: set[str] = set()
        self.reused = 0
        self.hold_marks: set[str] = set()

def chains_trace(ep, pending: list[str]) -> tuple[list[str], list[str], list[tuple[str, str, tuple[str, ...]]]]:
    """One TRAIL round: ask how each pending chain runs and settle what it can."""
    walker = ep.chains
    walker.rounds += 1
    # Arena b10 noask r12-r16: 63 of 461 trail rounds (482 s) saw no new page or row of a chain's shops since the round
    # before; such a round repeats the last answer, read against the pages as they are now
    if walker.last_got is not None and not chains_changed(ep, pending):
        walker.reused += 1
        ep.log('trail_reuse', pending=pending)
        return chains_read_chains(ep, walker.last_got, pending)
    models = [LIGHT, HEAVY] if ep.llm.cost + HEAVY_ROOM <= SPEND_HARD else [LIGHT]
    walker.read_at = set(ep.opened)
    seen_pages, seen_rows = ep.opened | ep.page_seen, set(ep.mem.rows)
    got = ep.llm.ask(chains_prompt(ep, pending), models, 4500) or {}
    ep.log('trail', pending=pending, got=clip(got, 150))
    answered = [c for c in as_list(got.get('chains')) if isinstance(c, dict)]
    walker.last_got = got if answered else None
    walker.last_pending, walker.last_pages, walker.last_rows = tuple(pending), seen_pages, seen_rows
    return chains_read_chains(ep, got, pending)

def chains_changed(ep, pending: list[str]) -> bool:
    """Whether a trail round would see new input: other pending needs, or a page read / row seen of a chain's listings."""
    walker = ep.chains
    if not set(pending) <= set(walker.last_pending):
        return True
    # the words naming a chain's listings and shops: the pending needs and the last answer's link descriptions
    links = [link for chain in as_list((walker.last_got or {}).get('chains')) if isinstance(chain, dict)
             for link in as_list(chain.get('links')) if isinstance(link, dict)]
    ids = {code_listing(str(link.get('id') or '').strip(), ep.mem.rows) for link in links}
    # plus what a read link's own page says about the next one
    sends = [chains_onward(ep, i, need) for i in ids if i in ep.mem.pages for need in pending]
    text = '\n'.join(pending + [f"{link.get('about') or ''} {link.get('find') or ''}" for link in links] + sends)
    on_chain = lambda ident: ident in ids or chains_names_shop(ep, text, ident)
    fresh_pages = (ep.opened | ep.page_seen) - walker.last_pages
    fresh_rows = set(ep.mem.rows) - walker.last_rows
    return any(map(on_chain, fresh_pages | fresh_rows))

def walk_report(pages: list[str], values: dict[str, str], lost: list[str], hold: bool) -> str:
    """Desk text after a walk: pages opened, values read and, when held, what is still to open."""
    text = '[Desk opened the pages a requirement points at]\n' + '\n'.join(pages) + '\n' if pages else ''
    if values:
        text += '[values read from the pages requirements point at] ' + '; '.join(f'{k} -> {v}' for k, v in values.items()) + '\n'
    if not hold:
        return text
    return (text + 'The order counts only once every page a requirement names, and every page that one leads to, has been opened in full (a '
            'search row is not enough). Still not found and opened: ' + ' | '.join(lost[:3]) + ". Search the shop's name together with the "
            "values its description lists (that shop's other listings of the kind are not it), open the shop row whose page carries those "
            'values, apply the value given there, then check out.')

def chains_walk(ep, room: int = 6) -> tuple[str, bool]:
    """Resolve pointers; returns (text, hold order)."""
    pending = ep.open_pointers()
    tool = ep.offer_view() or ep.offer_tool()
    target = ep.id_param(tool)
    pages, lost = [], []
    # chains of 3-4 hops (Arena r13: Voshibel -> Motormo -> Morisano -> Rarilen) need a round and a search per hop
    left = ep.steps_left()
    limit = min(room + 2, 8) if left > 25 else min(room, 6 if left > 15 else 4)
    while pending and ep.llm.cost < SPEND_HARD and ep.chains.rounds < limit and not ep.past_stop():
        reach, lost, gaps = chains_trace(ep, pending)
        pending = [need for need in pending if need not in ep.ref_values]
        left = ep.steps_left()
        searchable = tool and target and ep.chains.searches < (5 if left > 25 else 3) and left > 12
        if gaps and not reach and searchable:
            reach = chains_find_link(ep, *gaps[0])
            newest = ep.trace[-1]
            if newest.get('kind') == 'link_known':
                pages.append(f"{newest['id']}: take the value from an opened page of the shop named in this chain, one showing the values its description lists")
                lost = lost[1:]
        # the round's candidates open together: in one side-by-side step where that view shows pages, else one read each
        side = chains_side_view(ep) is not None
        batch = reach[:chains_batch_room(ep)] if side else reach[:max(0, min(3, 6 - len(pages)))]
        if ep.finished or not (batch and tool and target) or ep.steps_left() < (1 if side else len(batch)) + 4:
            break
        pages += chains_open_links(ep, tool, target, batch)
    left = ep.steps_left()
    hold = bool(lost) and left > 5 and ep.chains.holds < (3 if left > 15 else 2)
    # a hold naming exactly the links an earlier hold named says nothing new (the trail round repeats its answer while
    # no page or row of a chain's shops came in)
    mark = '|'.join(lost)
    if hold and mark in ep.chains.hold_marks:
        ep.log('chain_hold_repeat', lost=lost[:3])
        hold = False
    if hold:
        ep.chains.hold_marks.add(mark)
    ep.chains.holds += hold
    return walk_report(pages, ep.ref_values, lost, hold), hold

def chains_prompt(ep, pending: list[str]) -> str:
    rows, pages = ep.mem.rows, ep.mem.pages
    newest = ep.mem.order[::-1]
    full = [i for i in newest if i in ep.opened and (pages.get(i) or rows.get(i))]
    # a page whose own words send the reader on is marked with that sentence (Arena b10 r14/r13: TRAIL ended chains on
    # such pages; 3% of opened records are cut at 1400 characters, the note last)
    sends = lambda i: next((f'[sends on for {need.split(":", 1)[0].strip()}: "{said}"] ' for need in pending for said in [chains_onward(ep, i, need)] if said), '')
    records = ['OPENED ' + sends(i) + dumps(clip(pages.get(i) or rows.get(i), 300))[:1400] for i in full][:45]
    bare = ['ROW ' + dumps(clip(rows.get(i), 80))[:300] for i in newest if i not in ep.opened or not (pages.get(i) or rows.get(i))]
    records += bare[:60 - len(records)]
    said = [f'{need}\n  request: "{words}"' for need in pending for words in [chains_request_words(ep, need)] if words]
    parts = [TRAIL, 'REQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n' + '\n'.join(pending)]
    parts += ["REQUEST'S WORDS FOR THEM (the shopper's own sentence for each):\n" + '\n'.join(said)] if said else []
    return '\n\n'.join(parts + ['RECORDS (newest first):\n' + '\n'.join(records)])

def chains_request_words(ep, need: str) -> str:
    """The request's own sentence for a pointer requirement: the one naming the shops its description names, else ''."""
    # Arena b10: the request reader rewrote 'partner' wording as 'same as' (2302, ece5) and TRAIL paired a value the
    # request wanted as given (0e86); whether a pairing rule applies is read from the shopper's own words
    sentences = [s.strip() for s in re.split('(?<=[.!?;])\\s+|\\n+', ep.request) if s.strip()]
    proper = lambda text: {w.lower() for w in re.findall("(?<![A-Za-z])[A-Z][a-z]{3,}", text)}
    names = proper(need) | {n.lower() for row in ep.mem.rows.values() for n in shop_values(row) if len(n) > 3 and n.lower() in need.lower()}
    prop = roots(re.sub('[^a-z0-9]+', ' ', need.split(':', 1)[0].lower()))
    scored = [(len(names & proper(s)) * 3 + len(prop & roots(re.sub('[^a-z0-9]+', ' ', s.lower()))), s) for s in sentences if names & proper(s)]
    best = max(scored, default=(0, ''))[1]
    return ' '.join(best.split())[:500]

def code_pattern(code: str) -> Any:
    """A matcher for a part or model code written with or without a dash or space ("AB1234", "AB-1234"), or None."""
    hit = re.fullmatch('([A-Za-z]{1,6})[- ]?(\\d{2,7})', code.strip())
    return re.compile('(?<![A-Za-z0-9])' + hit.group(1) + '[- ]?' + hit.group(2) + '(?![0-9])', re.I) if hit else None

def code_listing(code: str, rows: dict[str, Any]) -> str:
    """The listing id a chain link names by its own part or model code, or the code itself when no seen row carries it."""
    # Arena r13 noask b07: TRAIL wrote links as the codes the pages name each other by ("LU5557") where the records show
    # them in titles ("... [LU5557]"); such a link never counted as found or opened, and 15/77 orders missed the process
    if code in rows:
        return code
    rx = code_pattern(code)
    # Arena r16 b10 (f788): codes with a digit in the letter part ("AB2-3456") matched no pattern, so a chain the model
    # had walked to its end stayed open and Desk held the cart; such a code counts as a whole token, parts joined by a
    # dash, a space or nothing
    parts = re.findall('[A-Za-z0-9]+', code)
    id_shaped = any(len(i) == len(code) and i[:2] == code[:2] for i in rows)
    if rx is None and 4 <= len(code) <= 16 and re.search('\\d', code) and re.search('[A-Za-z]', code) and len(parts) <= 3 and not id_shaped:
        rx = re.compile('(?<![A-Za-z0-9])' + '[- ]?'.join(map(re.escape, parts)) + '(?![A-Za-z0-9])', re.I)
    if rx is None:
        return code
    titled = [i for i, row in rows.items() if any(rx.search(str(v)) for k, v in row.items() if isinstance(v, str) and re.search(TITLE_FIELD, k, re.I))]
    anywhere = titled or [i for i, row in rows.items() if rx.search(dumps(row))]
    return anywhere[0] if len(anywhere) == 1 else code

def chain_outcome(chain: Any, pending: list[str], opened: set[str], rows: dict[str, Any], onward: Any = None) -> tuple[Any, ...]:
    """What one traced chain says: ('value', need, value, ids), ('open', ids), ('find', need, about, words, ids) or ().
    onward(id, need) gives the sentence of an opened page that sends the reader on for the need's property, or ''."""
    if not isinstance(chain, dict):
        return ()
    named = str(chain.get('need') or '').strip()
    # Arena r16 b10 (2302): TRAIL echoed two needs without their 'property:' head, and both chains were dropped
    near = [p for p in pending if named and (named in p or p.split(':', 1)[-1].strip() in named)]
    if named in pending:
        need = named
    elif len(pending) == 1:
        need = pending[0]
    elif len(near) == 1:
        need = near[0]
    else:
        return ()
    if not need:
        return ()
    links = [step for step in as_list(chain.get('links')) if isinstance(step, dict)]
    ids = [code_listing(str(step.get('id') or '').strip(), rows) for step in links]
    value = str(chain.get('value') or '').strip()
    settled = bool(links) and (chain.get('done') is True and bool(value) or not str(links[-1].get('find') or '').strip())
    if settled and not [i for i in ids if i not in opened]:
        sentence = onward(ids[-1], need) if onward else ''
        if not sentence:
            return 'value', need, value or f'as stated on {ids[-1]}', ids
        # the last page read sends the reader on for this property: its own words describe the next link
        return 'find', need, sentence, chains_onward_words(sentence), tuple(ids)
    closed = [i for i in ids if i in rows and i not in opened]
    if closed:
        return 'open', closed
    missing = [n for n, i in enumerate(ids) if i not in rows]
    gap, code = {}, ''
    if missing:
        gap = links[missing[0]]
        code = ids[missing[0]] if code_pattern(ids[missing[0]]) else ''
    told = str(gap.get('about') or '')
    about = told or 'the listing its value comes from'
    words = str(gap.get('find') or '').strip()
    if code:
        # a link named only by its code: its description (shop, kind, values) is the search, and the code picks the row
        about = f'{about} [{code}]'
        words = words or re.sub('\\([^)]*\\)', ' ', told).strip()
    return 'find', need, about, words, tuple(i for i in ids if i and i in rows)

def chains_read_chains(ep, got: dict[str, Any], pending: list[str]) -> tuple[list[str], list[str], list[tuple[str, str, tuple[str, ...]]]]:
    """Values, ids to open and missing links."""
    reach: list[str] = []
    lost: list[str] = []
    gaps: list[tuple[str, str, tuple[str, ...]]] = []
    # where the store has a page view apart from the attributes, a page's notes ("for X see Shop's listing, not this
    # one") show only there: a link read through the attributes alone is opened again (Arena r12/r16 b09: chains closed
    # on the first page's own fact while its notes sent on)
    read = ep.page_seen if ep.offer_view() else ep.opened
    onward = lambda ident, need: chains_onward(ep, ident, need)
    for outcome in (chain_outcome(chain, pending, read, ep.mem.rows, onward) for chain in as_list(got.get('chains'))):
        tag = outcome[0] if outcome else ''
        if tag == 'find' and outcome[4] and outcome[2] == chains_onward(ep, outcome[4][-1], outcome[1]):
            ep.log('trail_onward', need=outcome[1], page=outcome[4][-1], words=outcome[3])
        if tag == 'value':
            _, need, value, ids = outcome
            ep.ref_values[need] = value
            ep.ref_pages[need] = ids
            ep.version += 1
        elif tag == 'open':
            reach += [i for i in dict.fromkeys(outcome[1]) if i not in reach]
        elif tag == 'find':
            _, need, about, words, ids = outcome
            lost.append(f'{need} -> find {about}' + (f' (search: {words})' if words else ''))
            gaps.append((about, words, ids))
    return reach, lost, gaps

def chains_open_links(ep, tool: str, target: str, ids: list[str]) -> list[str]:
    """Open listings: several in one side-by-side step where that view may read chain pages, else with the page tool."""
    shown: list[str] = []
    side = chains_side_view(ep) if len(ids) > 1 else None
    if side:
        name, many = side
        batch = ids[:ep.side_room()]
        args = shaped({many: batch}, ep.schemas.get(name))
        reply = None if args is None else ep.store.call(name, args)
        if reply is None or ep.finished:
            return shown
        seen = reply.get('observation')
        ep.ingest(name, seen)
        listed = dict(mem_listings_in(ep, seen))
        got = [i for i in batch if i in listed]
        if got:
            ep.side_ok = max(ep.side_ok, len(batch))
        else:
            ep.side_bad = min(ep.side_bad or 999, len(batch))
        ep.log('chain_batch', ids=batch, shown=len(got))
        if got:
            return [f'{i}: ' + ep.render(listed[i])[:1500] for i in got]
        ids = ids[:2] if ep.steps_left() > 8 else []
    for ident, observation in page_reads(ep, tool, target, ids):
        if tool == ep.offer_view():
            ep.page_seen.add(ident)
        ep.ingest(tool, observation)
        shown.append(f'{ident}: ' + ep.render(observation)[:1500])
    return shown

def shop_values(row: dict[str, Any]) -> list[str]:
    """Stripped short text values a row files under a shop-like key."""
    return [str(v).strip() for k, v in row.items() if isinstance(v, str) and len(v.strip()) <= 40 and re.search(SHOP_FIELD, str(k), re.I)]

def chains_names_shop(ep, about: str, ident: str) -> bool:
    low = about.lower()
    names = [n for n in shop_values(ep.mem.rows.get(ident) or {}) if len(n) > 2]
    return any(re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)

def chains_shop_rows(ep, about: str, ids: list[str]) -> list[str]:
    return [i for i in ids if i not in ep.opened and chains_names_shop(ep, about, i)]

def chains_opened_shop(ep, about: str) -> list[str]:
    return [i for i in ep.opened if chains_names_shop(ep, about, i)]

def leaf_texts(node: Any) -> list[str]:
    """Every scalar of a nested record as text."""
    if isinstance(node, dict):
        return [t for value in node.values() for t in leaf_texts(value)]
    if isinstance(node, list):
        return [t for value in node for t in leaf_texts(value)]
    return [str(node)]

def chains_link_search(ep, plan: tuple[str, dict[str, Any], str], words: str, memo: str, about: str, shop: set[str]) -> list[str]:
    """Search the store for a chain's missing link; shop rows found."""
    tool, base, field = plan
    args = shaped({**base, field: chains_kind_last(ep, tool, field, words, shop)}, ep.schemas.get(tool))
    if args is None:
        return []
    ep.chains.searches += 1
    reply = ep.store.call(tool, args)
    if reply is None:
        return []
    observation = reply.get('observation')
    ep.ingest(tool, observation)
    ep.chains.link_seen[memo] = [i for i, _ in mem_listings_in(ep, observation)]
    found = chains_shop_rows(ep, about, ep.chains.link_seen[memo])
    ep.log('link_search', words=words[:80], rows=found[:4])
    return found

def chains_find_link(ep, about: str, words: str, before: tuple[str, ...] = ()) -> list[str]:
    """Candidate rows for a chain's missing link."""
    mem = ep.mem
    shop_ids = chains_shop_rows(ep, about, list(mem.rows)) + chains_opened_shop(ep, about)
    named = {w for ident in shop_ids for v in shop_values(mem.rows.get(ident) or {}) for w in re.findall('[a-z0-9]{3,}', v.lower())}
    # English function words and the words any pointer phrase uses for a page (grammar, not product vocabulary)
    common = named | {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for'}
    described = (words or about).lower()
    values = {w for w in re.findall('[a-z0-9]+', described) if len(w) >= 3 or w.isdigit()} - common
    if chains_link_opened(ep, about, before, values):
        return []
    memo = ' '.join(words.lower().split())
    plan = scout_search_template(ep)
    known: list[str] = []
    if memo and memo in ep.chains.link_seen:
        ep.chains.searches += 1
        known = chains_shop_rows(ep, about, ep.chains.link_seen[memo])
    elif plan and words:
        known = chains_link_search(ep, plan, words, memo, about, named)
    of_kind = chains_link_kind(ep, about, words, set(re.findall('[a-z0-9]{3,}', described)) - common)
    # the seen rows of the named shop and kind, the search's hits first, each group led by the rows showing more of the
    # described values (Arena b10: 11af had the chain's end among its rows unopened, 4c08 the 'used' part as the third
    # of seven shop rows while the first two were read)
    shown = lambda ident: sum(bool(re.search('(?<![a-z0-9])' + re.escape(v) + '(?![a-z0-9])', dumps(mem.rows.get(ident) or {}).lower())) for v in values)
    found = list(filter(of_kind, known))
    seen = sorted((i for i in chains_shop_rows(ep, about, mem.order[::-1]) if of_kind(i) and i not in found), key=shown, reverse=True)
    picks = sorted(found, key=shown, reverse=True) + seen
    patterns = [code_pattern(c) for c in re.findall('\\[([A-Za-z]{1,6}[- ]?\\d{2,7})\\]', about)]
    carry = [i for i in known + mem.order[::-1] if any(rx and rx.search(dumps(mem.rows.get(i) or {})) for rx in patterns)]
    return list(dict.fromkeys(carry + picks))[:chains_batch_room(ep)]

def chains_link_opened(ep, about: str, before: tuple[str, ...], values: set[str]) -> bool:
    """An opened shop page (not an earlier link) whose text carries every described value: the link is already known."""
    # an earlier link of the chain is the page naming this one: its words describe the link, it is not it
    if not values:
        return False
    mem = ep.mem
    for ident in chains_opened_shop(ep, about):
        if ident in before:
            continue
        text = ' '.join(leaf_texts(mem.pages.get(ident) or mem.rows.get(ident) or {})).lower()
        if all(re.search('(?<![a-z0-9])' + re.escape(v) + '(?![a-z0-9])', text) for v in values):
            ep.log('link_known', id=ident)
            return True
    return False

def chains_link_kind(ep, about: str, words: str, kind: set[str]):
    """Test for rows of the link's kind: a kind word in the row, else a category label the description names."""
    # a row whose summary leaves the values out is still of the link's kind when its category label is one the
    # link's description names; the search's own order then ranks it
    cat_key = mem_kind_key(ep)
    told = roots(re.sub('[^a-z0-9]+', ' ', (about + ' ' + words).lower()))

    def test(ident: str) -> bool:
        if not kind or kind & set(re.findall('[a-z0-9]{3,}', dumps(ep.mem.rows.get(ident) or {}).lower())):
            return True
        label = roots(re.sub('[^a-z0-9]+', ' ', mem_label(ep, ident, cat_key).lower())) if cat_key else set()
        return bool(label) and label <= told
    return test

def chains_side_view(ep) -> tuple[str, str] | None:
    """The side-by-side tool and its id list argument when it may read chain pages (no separate offer view), else None."""
    # a store with an offer view apart from the attributes shows a page's notes only there (chains_read_chains)
    tool, many = ep.roles.get('compare'), ep.side_param()
    if not (tool and many) or ep.offer_view() or ep.side_view_bare():
        return None
    return tool, many

def chains_batch_room(ep) -> int:
    """Listings a chain search hands on: what one side-by-side step shows, else two (one read a step)."""
    # real races 12-16 (r15 text listings left out), pointer needs: episodes that opened none of the named shop's seen
    # rows passed 0/306, one 125/830, two or more 336/3135; reading more of them one step each bought nothing, so only
    # the one-step side view opens them all
    return max(2, ep.side_room()) if chains_side_view(ep) else 2

def chains_kind_last(ep, tool: str, field: str, words: str, shop: set[str]) -> str:
    """Kind-only words last for short text fields."""
    spec = ((ep.schemas.get(tool) or {}).get('properties') or {}).get(field)
    cap = spec.get('maxLength') if isinstance(spec, dict) else None
    cat_key = mem_kind_key(ep)
    if not (isinstance(cap, int) and len(words) > cap and cat_key):
        return words
    labels = (re.sub('[^a-z0-9]+', ' ', str(row.get(cat_key) or '').lower()) for row in ep.mem.rows.values())
    kinds = set().union(*map(roots, labels))

    def kind_only(token: str) -> bool:
        low = re.sub('[^a-z0-9]+', ' ', token.lower())
        base = roots(low)
        return bool(base) and base <= kinds and not shop.intersection(re.findall('[a-z0-9]{3,}', low))
    # a stable sort moves the words naming only a category (and no shop) to the end, each group in its own order
    return ' '.join(sorted(words.split(), key=kind_only))

def chains_recheck(ep) -> str:
    """One trail round over pages opened since the last round, outside the checkout's own budget of rounds and holds."""
    if not (ep.opened - ep.chains.read_at) or ep.llm.cost >= SPEND_HARD or ep.past_stop():
        return ''
    rounds, holds = ep.chains.rounds, ep.chains.holds
    ep.chains.rounds = 0
    try:
        text = chains_walk(ep, 1)[0]
    finally:
        ep.chains.rounds, ep.chains.holds = rounds, holds
    ep.log('cart_trail', step=ep.step, values=dict(ep.ref_values))
    return text

def chains_early(ep, fresh: set[str]) -> str:
    """Walk a chain when its shop page gets opened."""
    if ep.chains.rounds or ep.finished or ep.past_stop() or ep.steps_left() <= 15:
        return ''
    pending = ep.open_pointers()
    # Arena r16 noask b01: pointer tasks 3/36 (hayate b128 11/36); the pointed page sat in a search's rows unopened while
    # no chain round ran before the close. Once a worded search shows the store's query field, the first round searches
    # the pointed shop with the described values itself
    searched = bool(pending) and ep.step >= 3 and scout_search_template(ep) is not None
    if not searched and (not fresh or all(fresh.isdisjoint(chains_opened_shop(ep, need)) for need in pending)):
        return ''
    ep.log('early_trail', step=ep.step, pending=pending)
    return chains_walk(ep, 4)[0]

def chains_onward(ep, ident: str, need: str) -> str:
    """The sentence of a listing's own text that sends the reader to a different listing for the need's property, or ''."""
    # Arena b10 r14/r13: TRAIL closed chains on a start page whose note takes the property from another listing ("...,
    # not this item"; 80e1 and 6be5 ordered the start page's value and failed). Such a sentence names the property, then
    # a name other than the page's own shop, and refers back to this page with a negation or calls the other listing
    # another or a different one. Arena r12-r16 chain-property sentences (1,000+): no false hit, about 80% of pointers
    record = ep.mem.pages.get(ident) or ep.mem.rows.get(ident) or {}
    prop = roots(need.split(':', 1)[0])
    if not (prop and record):
        return ''
    own = {w.lower() for n in shop_values(record) for w in re.findall('[A-Za-z]+', n)}
    plain = lambda text: roots(re.sub('[^a-z0-9]+', ' ', text.lower()))
    for text in leaf_texts(record):
        for sentence in re.split('(?<=[.!?;])\\s+', text):
            if len(sentence) > 300 or not prop <= plain(sentence):
                continue
            sends = re.search("\\b(?:not|n't|instead|rather)\\b(?:\\W+\\w+){0,2}?\\W+(?:this|here)\\b|\\b(?:another|different)\\b", sentence.lower())
            names = [m.start() for m in re.finditer('(?<![A-Za-z])[A-Z][a-z]{3,}', sentence) if m.start() > 0 and m.group(0).lower() not in own]
            if sends and names and prop <= plain(sentence[:names[0]]):
                return sentence.strip()
    return ''

def chains_onward_words(sentence: str) -> str:
    """Search words for the listing an onward sentence names: from its first name on, without the words sending back."""
    names = [m.start() for m in re.finditer('(?<![A-Za-z])[A-Z][a-z]{3,}', sentence) if m.start() > 0]
    part = re.split("[,;]?\\s+\\b(?:not|instead|rather)\\b", sentence[names[0]:] if names else sentence)[0]
    # English function words a pointer phrase uses around the description (grammar, not product vocabulary)
    filler = {'the', 'and', 'with', 'from', 'that', 'comes', 'which', 'has', 'its', 'for', 'this', 'item', 'product', 'listing'}
    words = [w for w in re.findall("[A-Za-z0-9][A-Za-z0-9.:/-]*", re.sub("'s\\b", '', part)) if len(w) > 2 and w.lower() not in filler]
    return ' '.join(words[:12])

class CartDesk:
    """Cart record, cart calls and own purchases."""

    def __init__(self) -> None:
        self.lines: dict[str, int] = {}
        self.view: Any = None
        self.viewed_at = -1
        self.changed_at = -1
        self.reply = ''
        self.touched = False
        self.completed = False
        self.built: list[str] = []
        self.built_needs: set[str] = set()
        self.dropped: set[str] = set()

def basket_fresh_view(ep) -> bool:
    return ep.basket.view is not None and ep.basket.viewed_at >= ep.basket.changed_at

def basket_describe(ep) -> str:
    if basket_fresh_view(ep):
        return dumps(clip(ep.basket.view, 300))[:3000]
    return "lines per Desk's record of the cart calls (listing: units): " + dumps(ep.basket.lines)

def basket_units(ep, ident: str) -> int:
    if basket_fresh_view(ep):
        stack: list[Any] = [ep.basket.view]
        while stack:
            node = stack.pop()
            if isinstance(node, list):
                stack += node
                continue
            if not isinstance(node, dict):
                continue
            if ident in [v.strip() for v in node.values() if isinstance(v, str)]:
                counts = [whole(v) for k, v in node.items() if not isinstance(v, str) and whole(v) is not None
                          and re.search('\\b(?:qty|quantity|count|units?)\\b', plain_key(k))]
                if len(counts) == 1:
                    return counts[0]
            stack += [v for v in node.values() if isinstance(v, (dict, list))]
    return ep.basket.lines.get(ident, 1)

def basket_ids(ep) -> list[str]:
    if basket_fresh_view(ep):
        text = dumps(ep.basket.view)
        return [i for i in ep.mem.rows if i in text]
    return list(ep.basket.lines)

def basket_count_fixes(ep) -> list[tuple[str, int, int]]:
    """(line, held, wanted) for wrong quantities."""
    cat_key = mem_kind_key(ep)
    if not cat_key:
        return []
    out = []
    for ident in basket_ids(ep):
        have = basket_units(ep, ident) or 0
        owners = [n for n, item in enumerate(ep.items, 1) if mem_belongs(ep, item, mem_label(ep, ident, cat_key))]
        if len(owners) != 1:
            continue
        item = ep.items[owners[0] - 1]
        mine = [i for i in basket_ids(ep) if (ep.basket.lines.get(i, 1) or 0) > 0 and mem_belongs(ep, item, mem_label(ep, i, cat_key))]
        if have > 0 and len(mine) == 1 and have != item.qty and not ep.is_gone(ident):
            out.append((ident, have, item.qty))
    return out

def basket_fields(ep, name: str) -> tuple[dict[str, Any], dict[str, Any], str | None, str | None, str | None]:
    """Cart tool schema and its argument names."""
    schema = ep.schemas.get(name) or {}
    props = schema.get('properties') or {}
    specs = [(k, v) for k, v in props.items() if isinstance(v, dict)]
    texts = [k for k, v in specs if 'string' in type_names(v) and 'enum' not in v]
    target = (next((k for t, k in ep.id_args if t == name and k in texts), None)
              or next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None))
    count = next((k for k, v in specs if 'integer' in type_names(v)), None)
    mode = next((k for k, v in specs if isinstance(v.get('enum'), list) and len(v['enum']) >= 2), None)
    return schema, props, target, count, mode

def basket_fix_counts(ep, wrong: list[tuple[str, int, int]]) -> list[str]:
    """Set wrong quantities with the cart tool."""
    tool = ep.roles.get('cart') or ''
    schema, props, target, count, mode = basket_fields(ep, tool)
    if not (tool and target and count):
        return []
    if basket_sets_units(ep, tool):
        plan = [{target: ident, count: want} for ident, _, want in wrong]
    elif not wrong:
        plan = []
    elif not mode:
        return []
    else:
        # a tool without a set-count mode takes the line out and adds it again at the wanted count
        choices = list(map(str, props[mode]['enum']))
        drop = next((c for c in choices if re.search('remov|delet|drop|clear|take', c, re.I)), choices[1])
        add = next((c for c in choices if c != drop), choices[0])
        plan = []
        for ident, have, want in wrong:
            plan += [{target: ident, mode: drop, count: have}, {target: ident, mode: add, count: want}]
    if ep.steps_left() < len(plan) + 3:
        return []
    done: list[str] = []
    for args in plan:
        fitted = shaped(args, schema)
        if fitted is None:
            break
        reply = ep.store.call(tool, fitted)
        if reply is None or ep.finished:
            break
        seen = reply.get('observation')
        basket_record(ep, tool, fitted)
        ep.basket.reply = dumps(seen)
        ep.ingest(tool, seen)
        done.append(f'{dumps(fitted)} -> {ep.render(seen)[:300]}')
    ep.log('count_fix', calls=done)
    return done

def basket_calls_for(ep, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
    """Cart calls for exactly these lines, or None."""
    tool = ep.roles.get('cart') or ''
    _, props, target, count, mode = basket_fields(ep, tool)
    whole_call = basket_whole_cart_call(props, lines)
    if whole_call is not _NO:
        return whole_call
    if not target:
        return None
    held = ep.basket.lines
    plan: list[dict[str, Any]] = []
    for ident in drop:
        have = held.get(ident, 1)
        if mode:
            step = {target: ident, mode: props[mode]['enum'][1]}
            if count:
                step[count] = have
        elif count and basket_sets_units(ep, tool):
            step = {target: ident, count: 0}
        elif count and isinstance(props[count].get('minimum'), int) and props[count]['minimum'] <= -have:
            step = {target: ident, count: -have}
        else:
            return None
        plan.append(step)
    in_cart = basket_ids(ep) if lines else []
    for ident, want in lines.items():
        have = held.get(ident, 0) if ident in in_cart else 0
        if have == want:
            continue
        if have and not basket_sets_units(ep, tool):
            return None
        step = {target: ident}
        if mode:
            step[mode] = props[mode]['enum'][0]
        if count:
            step[count] = want
        plan.append(step)
    return plan

def basket_whole_cart_call(props: dict[str, Any], lines: dict[str, int]) -> Any:
    """Whole-cart call for list-taking tools."""
    for key, spec in props.items():
        shape = spec.get('items') if isinstance(spec, dict) and 'array' in type_names(spec) else None
        if not (isinstance(shape, dict) and isinstance(shape.get('properties'), dict)):
            continue
        inner = shape['properties']
        id_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'string' in type_names(s) and 'enum' not in s), None)
        count_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'integer' in type_names(s)), None)
        if not id_key:
            return None
        return [{key: [{id_key: i, **({count_key: n} if count_key else {})} for i, n in lines.items()]}]
    return _NO

def basket_complete_cart(ep) -> str:
    """Buy a confirmed lead's cart directly."""
    cart, buy = ep.roles.get('cart'), ep.roles.get('buy')
    if ep.basket.completed or not ep.confirmed or ep.is_gone(ep.confirmed) or not cart or not buy:
        return ''
    ep.basket.completed = True
    lines, drop = basket_lead_cart(ep)
    calls = basket_calls_for(ep, lines, drop) if lines else None
    if not calls or len(calls) + 1 > ep.steps_left():
        return ''
    ids = list(lines)
    scout_read_pages(ep, ids, len(calls) + 3)
    if not basket_passes(ep, ids):
        return ''
    placed = basket_place(ep, cart, calls)
    replies = basket_buy_confirmed(ep, buy, ids) if placed else []
    if not replies:
        return ''
    return ('Not run (Desk acted instead): the independent check confirmed ' + ep.confirmed + ', so Desk set the cart ('
            + '; '.join(placed) + ') and checked out. Store reply: ' + ' | then: '.join(replies))

def basket_lead_cart(ep) -> tuple[dict[str, int], list[str]]:
    """Lines to set and remove for a lead."""
    key = mem_kind_key(ep)

    kind = lambda ident: mem_label(ep, ident, key).strip().lower() if key else ident
    picks: dict[str, int] = {}
    for number, ids in ep.lead_items.items():
        index = whole(number) or 0
        pick = next((i for i in ids if not ep.is_gone(i)), None)
        if pick and 1 <= index <= len(ep.items) and kind(pick) not in {kind(p) for p in picks}:
            picks[pick] = ep.items[index - 1].qty
    if ep.confirmed not in picks:
        return {}, []
    covered = {kind(p) for p in picks}
    now = basket_ids(ep)
    keep = {i: ep.basket.lines.get(i, 1) for i in now if i not in picks and not ep.is_gone(i) and kind(i) not in covered}
    if len(picks) + len({kind(i) for i in keep}) < len(ep.items):
        return {}, []
    return {**keep, **picks}, [i for i in now if i not in picks and i not in keep]

def basket_passes(ep, ids: list[str]) -> bool:
    """Independent check of these lines as the cart."""
    judge = ep.judge
    judge_check_cart(ep, ids, SPEND_HARD)
    return bool(judge.verdicts.get(judge_check_key(ep, ids)))

def basket_place(ep, cart: str, calls: list[dict[str, Any]]) -> list[str]:
    """Run cart calls; [] when one fails."""
    placed: list[str] = []
    for args in calls:
        fitted = shaped(args, ep.schemas.get(cart))
        reply = None if fitted is None else ep.store.call(cart, fitted)
        if reply is None or ep.finished:
            return []
        basket_record(ep, cart, fitted)
        ep.ingest(cart, reply.get('observation'))
        placed.append(dumps(fitted))
    return placed

def basket_buy_confirmed(ep, buy: str, ids: list[str]) -> list[str]:
    """Check out, once more after a refusal."""
    args = shaped(dict(ep.brief.get('buy_args') or {}), ep.schemas.get(buy))
    if args is None:
        return []
    replies: list[str] = []
    for attempt in range(2):
        if attempt:
            news_refresh_offers(ep)
            if not basket_passes(ep, ids) or ep.steps_left() < 1 or ep.finished or any(ep.is_gone(i) for i in ids):
                break
        go, said = gates_run(ep, 'order')
        if said:
            replies.append(said)
        if not go:
            break
        reply = ep.store.call(buy, args)
        ep.log('completed', lines=ids, reply=clip(reply, 200))
        if reply is None:
            break
        ep.checkouts += 1
        if not reply.get('finished'):
            ep.refused_at = ep.step
        ep.ingest(buy, reply.get('observation'))
        replies.append(ep.render(reply.get('observation')))
        if ep.finished or reply.get('finished'):
            break
    return replies

def basket_sets_units(ep, name: str) -> bool:
    """Cart tool sets units rather than adding."""
    props = ep.schemas.get(name, {}).get('properties') or {}
    specs = [s for s in props.values() if isinstance(s, dict)]
    if any(isinstance(s.get('enum'), list) and len(s['enum']) >= 2 for s in specs):
        return False
    return any('integer' in type_names(s) and s.get('minimum') == 0 for s in specs)

def basket_record(ep, name: str, args: dict[str, Any]) -> None:
    """Track cart lines after a cart call."""
    ep.basket.touched = True
    ep.basket.changed_at = ep.step
    spec = ep.schemas.get(name, {}).get('properties') or {}

    modes = lambda key: (spec.get(key) or {}).get('enum')
    removal = any(isinstance(modes(k), list) and len(modes(k)) >= 2 and v in modes(k)[1:] for k, v in args.items())
    named, whole_cart = basket_lines_named(ep, spec, args)
    if whole_cart:
        ep.basket.lines = {i: c for i, c in named if c and c > 0}
        ep.added_at = {i: ep.added_at.get(i, ep.step) for i in ep.basket.lines}
        return
    sets = basket_sets_units(ep, name)
    for ident, count in named:
        have = units_after(ep.basket.lines.get(ident, 0), count, removal, sets)
        if have > 0:
            ep.basket.lines[ident] = have
            ep.added_at.setdefault(ident, ep.step)
        else:
            ep.basket.lines.pop(ident, None)
            ep.added_at.pop(ident, None)

def basket_lines_named(ep, spec: dict[str, Any], args: dict[str, Any]) -> tuple[list[tuple[str, int | None]], bool]:
    """(id, units) pairs a cart call names."""
    rows = ep.mem.rows
    loose = next((whole(v) for v in args.values() if whole(v) is not None and not isinstance(v, str)), None)
    found: list[tuple[str, int | None]] = []
    whole_cart = False
    for key, value in args.items():
        if isinstance(value, list):
            shape = (spec.get(key) or {}).get('items') or {}
            whole_cart = whole_cart or (not value and isinstance(shape, dict) and (shape.get('type') == 'object' or 'properties' in shape))
            entries = value
        else:
            entries = [value] if isinstance(value, str) else []
        for line in entries:
            if isinstance(line, str):
                if line in rows:
                    found.append((line, loose))
                continue
            if not isinstance(line, dict):
                continue
            whole_cart = True
            ident = next((v for v in line.values() if isinstance(v, str) and v in rows), None)
            if not ident:
                continue
            got = next((whole(v) for v in line.values() if whole(v) is not None and not isinstance(v, str)), None)
            found.append((ident, 1 if got is None else got))
    return found, whole_cart

def basket_late_pick(ep, number: int, held: list[str], index: dict[str, Any]) -> str | None:
    """A read page for an item passing code checks."""
    kind, marks = mem_kind_key(ep), index['marks']
    needs = ep.items[number - 1].live()
    codes = {c for need in needs for c in model_codes(need)}
    for part, main in ep.bound:
        if part != number:
            continue
        for i in held:
            if mem_owner(ep, mem_label(ep, i, kind)) == main:
                codes.update(marks.get(i, {}).get('codes') or [])
    owns = {m.get('own') for m in marks.values()}
    picks = []
    for ident in reversed(list(ep.mem.pages)):
        if mem_owner(ep, mem_label(ep, ident, kind)) != number or ident in held:
            continue
        if not mem_can_buy(ep, ident) or ident in ep.gone or ident in ep.basket.dropped or judge_apart_needs(ep, ident, needs):
            continue
        fits = judge_fit_set(ident, index)
        more = (marks.get(ident) or {}).get('more')
        clash = codes and fits and not fits & codes and (not more or more in owns)
        if not clash:
            picks.append(ident)
    if ep.cheapest():
        picks.sort(key=ep.price_rank)
    return next(iter(picks), None)

def basket_late_add(ep) -> None:
    """P8: fill missing items of a cart at the clock stop."""
    cart, buy, refuse = (ep.roles.get(role) for role in ('cart', 'buy', 'refuse'))
    cat_key = mem_kind_key(ep)
    if ep.finished or ep.store.blocked or ep.llm.out_of_credit or not ep.past_stop() or not all((cart, buy, refuse, cat_key)):
        return
    held = [i for i in basket_ids(ep) if (basket_units(ep, i) or 0) > 0 and not ep.is_gone(i)]
    owners = [mem_owner(ep, mem_label(ep, i, cat_key)) for i in held]
    missing = [n for n in range(1, len(ep.items) + 1) if n not in owners]
    if not held or not missing or 0 in owners or len(set(owners)) < len(held):
        return
    index = judge_fit_index(ep)
    picks = {n: basket_late_pick(ep, n, held, index) for n in missing}
    entry = ep.log('late_add', step=ep.step, cart=held, picks=picks)
    if not all(picks.values()):
        entry['why'] = 'no page passes'
        return
    lines = {**{i: basket_units(ep, i) for i in held}, **{str(p): ep.items[n - 1].qty for n, p in picks.items()}}
    why, calls, room = basket_late_room(ep, lines)
    if why:
        entry['why'] = why
        return
    entry['checked'] = ok = basket_late_check(ep, list(lines), room)
    if ok and EPISODE_SECONDS - ep.elapsed() >= 6:
        entry['placed'] = basket_place(ep, cart, calls)

def basket_late_room(ep, lines: dict[str, Any]) -> tuple[str, list[Any], float]:
    """Why the filled cart cannot be placed now ('' when it can), its cart calls and the seconds left."""
    prices = [mem_price(ep, i) for i in lines]
    cap = ep.budget
    if isinstance(cap, (int, float)) and None not in prices:
        if ep.brief.get('budget_scope') == 'each':
            over = max(prices) > cap
        else:
            over = sum(p * lines[i] for p, i in zip(prices, lines)) > cap + 1e-09
        if over:
            return 'budget', [], 0
    calls = basket_calls_for(ep, lines, [])
    room = EPISODE_SECONDS - ep.elapsed() - 8
    fits = calls and len(calls) + 3 <= ep.steps_left() and room >= 6
    return ('' if fits else 'no room'), calls, room

def basket_late_check(ep, ids: list[str], room: float) -> bool:
    """One model check at most, its timeout cut to the time left; the call count is restored after."""
    llm = ep.llm
    before, timeout = llm.calls, getattr(_llm, 'timeout', None)
    floor = llm.calls = max(before, ALL_CALLS - 1)
    timed = isinstance(timeout, (int, float))
    if timed:
        _llm.timeout = max(4, int(min(timeout, room)))
    try:
        return basket_passes(ep, ids)
    finally:
        llm.calls = before + max(0, llm.calls - floor)
        if timed:
            _llm.timeout = timeout

class Gatekeeper:
    """Required steps at the order or the ending."""

    def __init__(self) -> None:
        self.cleared: set[tuple[str, str]] = set()
        self.record: list[str] = []
        self.record_steps: list[int] = []
        self.sent = 0

def gates_note_line(ep, line: str, step: int) -> None:
    ep.gates.record.append(line)
    ep.gates.record_steps.append(step)

def gates_run(ep, stage: str) -> tuple[bool, str]:
    """Do the steps due at a stage: (go on, report)."""
    due = [p for p in ep.process if p['when'] == stage]
    if not due or ep.finished or not ep.stage_open(stage):
        return True, ''
    done: list[str] = []
    nudged = False
    for _ in range(GATE_ROUNDS):
        prompt = gates_prompt(ep, stage, due)
        got = ep.llm.ask(prompt, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 3500)
        if got is None:
            break
        told: list[str] = []
        hold = ''
        if done:
            talk_apply_answers(ep, {'needs': got.get('needs')}, told)
            hold = str(got.get('hold') or '').strip()
        call = got.get('call')
        if not isinstance(call, dict):
            call = None
        ep.log('gate', stage=stage, call=clip(call, 200), steps=clip(got.get('steps'), 120), hold=hold[:200], told=told)
        if told or hold:
            ep.version += 1
            changes = '; '.join([*told, hold] if hold else told)
            lead = gates_said(done) + '\n' if done else ''
            return False, f"{lead}The shopper's reply changes the task: {changes}. Fix the cart for it, then check out again."
        if call or done or nudged or not gates_unproven(ep, stage, due, got):
            line = gates_do_call(ep, stage, call)
            if line is None:
                break
            done.append(line)
        else:
            nudged = True
    return True, gates_said(done) if done else ''

def gates_said(done: list[str]) -> str:
    return 'Before going on, Desk carried out the steps the request requires:\n' + '\n'.join(done)

def gates_prompt(ep, stage: str, due: list[dict[str, Any]]) -> str:
    buying = stage == 'order'
    record = '\n'.join(f'{n}. {line}' for n, line in enumerate(ep.gates.record, 1)) or '(nothing yet)'
    notes = '\n\nSTORE NOTES SEEN:\n' + '\n'.join(ep.notes[-4:]) if ep.notes else ''
    parts = [GATE.replace('@ACT@', 'place the order' if buying else 'end without a purchase'),
             'REQUEST:\n' + ep.request,
             'STEPS DUE NOW:\n' + '\n'.join(f"- {p['step']}" for p in due),
             'LEDGER:\n' + req_ledger(ep),
             'CART NOW: ' + basket_describe(ep),
             'RECORD (messages and step calls so far):\n' + record + notes,
             f"MESSAGES LEFT: {max(0, ep.msg_cap - ep.msg_sent)}   STEPS LEFT: {ep.steps_left()} (the {'order' if buying else 'ending'} itself takes one)",
             'TOOLS:\n' + req_tool_digest(ep, ep.closing_tools())]
    return '\n\n'.join(parts)

def gates_unproven(ep, stage: str, due: list[dict[str, Any]], got: dict[str, Any]) -> bool:
    """Flag claimed steps no record line backs."""
    if not ep.roles.get('ask') or ep.msg_sent >= ep.msg_cap or ep.steps_left() < 2:
        return False
    record, changed = ep.gates.record, ep.basket.changed_at
    if stage == 'order' and any(p['messages'] > 0 for p in due):
        backed = {n for n, at in enumerate(ep.gates.record_steps[:len(record)], 1) if at > changed}
    else:
        backed = set(range(1, len(record) + 1))
    unproven = []
    for claim in as_list(got.get('steps')):
        if isinstance(claim, dict) and (claim.get('done') is not True or not backed & {whole(n) or 0 for n in as_list(claim.get('lines'))}):
            unproven.append(claim)
    if not unproven:
        return False
    ep.log('gate_unproven', steps=clip(unproven, 120))
    names = '; '.join(str(e.get('step') or '') for e in unproven)[:200]
    gates_note_line(ep, f'(Desk: no line made since the cart last changed at step {changed} shows {names} done, and {ep.msg_cap - ep.msg_sent} '
                   f'message(s) and {ep.steps_left()} steps remain: give the call that does it)', -1)
    return True

def gates_do_call(ep, stage: str, call: dict[str, Any] | None) -> str | None:
    """Run the gate model's call; record line or None."""
    if not call:
        ep.gates.cleared.add((stage, ep.cart_stamp() + f'#{ep.version}'))
        return None
    tool = str(call.get('tool') or '')
    asking = tool == ep.roles.get('ask')
    if tool not in ep.schemas or tool in ep.closing_tools() or ep.steps_left() < 2 or asking and ep.msg_sent >= ep.msg_cap:
        return None
    given = call.get('args')
    args = shaped(given if isinstance(given, dict) else {}, ep.schemas[tool])
    if args is None:
        return None
    if not asking:
        reply = ep.store.call(tool, args)
        if reply is None:
            return None
        seen = reply.get('observation')
        mem_absorb(ep, seen)
        shown, sent_args = ep.render(seen), dumps(args)
        gates_note_line(ep, f'step {ep.step}: {tool} {sent_args} -> {shown[:300]} (cart at that time: {ep.cart_stamp()})', ep.step)
        return f'{tool} {sent_args} -> {shown[:400]}'
    text = next((v for v in args.values() if isinstance(v, str)), '')
    sent = talk_send(ep, [(1, text)], 1)
    if not sent:
        return None
    ep.gates.sent += 1
    return f'Q: {text}\nA: {sent[0][2]}'

class Closer:
    """Holds on endings and the desk's own endings."""

    def __init__(self) -> None:
        self.sold_holds = 0
        self.lead_holds = 0
        self.end_holds = 0
        self.line_holds = 0
        self.fit_holds = 0
        self.pushed = False
        self.late_held = False

def close_policy(ep, job: Call) -> str:
    """Planner of the ending holds: the first step whose executor answers holds the call; '' lets it run."""
    if not job.ending:
        return ''
    return next(filter(None, (CLOSE_STEPS[name](ep, job) for name in close_plan(ep, job))), '')

def close_plan(ep, job: Call):
    """Hold steps for an ending call, in order; the buy branch looks at the room again after the offer pass."""
    roomy = ep.steps_left() > 2
    if ep.mode_ok('buy', job.name, job.args):
        yield 'offer'
        if ep.steps_left() > 2:
            yield from ('wider', 'changed', 'sold', 'chain', 'misfit', 'lines', 'review', 'stale')
    else:
        early = roomy and ep.steps_left() > 5
        yield from ('audit', 'leads', 'second_end', 'late_end') if early else ('second_end', 'late_end', 'late_review')
        yield from ('unsearched', 'closest', 'by_lead')
    yield 'gate'

def close_offer_pass(ep) -> str:
    """Read the offer page of every cart line not read since it was carted; hold only when one changed and room is left."""
    due = ep.unoffered()
    tool = ep.offer_view()
    target = ep.id_param(tool) if tool else None
    if not (due and target) or ep.steps_left() < len(due) + 1:
        return ''
    rows, lines = ep.mem.rows, []
    for ident, observation in page_reads(ep, tool, target, due):
        before = dict(rows.get(ident) or {})
        ep.offer_seen[ident] = ep.step
        ep.page_seen.add(ident)
        ep.ingest(tool, observation)
        moved = [k for k, v in (rows.get(ident) or {}).items() if k in before and before[k] != v and not isinstance(v, (dict, list))]
        if moved or ep.is_gone(ident):
            lines.append(f"{ident} changed ({', '.join(moved[:4]) or 'availability'}): {dumps(clip(observation, 300))[:900]}")
    ep.log('offer_pass', ids=due, changed=len(lines))
    if not lines:
        return ''
    ep.version += 1
    if ep.steps_left() <= 2 and not any(ep.is_gone(i) for i in due):
        return ''
    return 'Desk read the offer page of each cart line before checkout; these changed:\n' + '\n'.join(lines) + close_cart_now(ep)

def close_cart_now(ep) -> str:
    return '\nCart now: ' + basket_describe(ep)

def close_hold_wider(ep) -> str:
    wider = scout_wider_hint(ep)
    return wider + close_cart_now(ep) if wider else ''

def close_hold_changed(ep, job: Call) -> str:
    """Re-read lines, view the cart, hold on change."""
    basket = ep.basket
    job.reread = news_refresh_offers(ep)
    if ep.news.sold_flag:
        news_locate_sold(ep)
    tool = ep.roles.get('cart_view')
    if not (basket.touched and basket.viewed_at < basket.changed_at and tool and ep.steps_left() > 3):
        return ''
    args = shaped({}, ep.schemas.get(tool))
    view = None if args is None else ep.store.call(tool, args)
    if view is None:
        return ''
    basket.view, basket.viewed_at = view.get('observation'), ep.step
    change = ep.ingest(tool, basket.view)
    if not change:
        return ''
    ep.queued = []
    return 'Not run yet (no step used). Viewing the cart showed a change: ' + change + close_cart_now(ep) + '\nApply the change, fix the cart if needed, then check out.'

def close_hold_sold(ep) -> str:
    swap = news_swap_hint(ep)
    held = [i for i in basket_ids(ep) if i in ep.gone] if ep.close.sold_holds < 2 and not swap else []
    if not swap and not held:
        return ''
    ep.close.sold_holds += 1 if held else 0
    return 'Not run yet (no step used). ' + (swap or f"{', '.join(held)} in the cart is sold out: replace that line first.") + close_cart_now(ep)

def close_hold_chain(ep, job: Call) -> str:
    job.chain, hold = chains_walk(ep)
    return 'Not run yet (no step used). ' + job.chain + close_cart_now(ep) if hold else ''

def close_hold_misfit(ep) -> str:
    misfit = judge_fit_faults(ep) if ep.close.fit_holds < 2 and ep.steps_left() > ep.reserve() + 3 else []
    if not misfit:
        return ''
    ep.close.fit_holds += 1
    ep.log('fit_hold', faults=misfit[:4])
    return ('Not run yet (no step used). The cart holds a part that is not compatible with the carted device: ' + '; '.join(misfit[:3])
            + '. Swap the part (or the device), then check out.' + close_cart_now(ep))

def close_hold_lines(ep) -> str:
    """Hold for wrong lines; fix quantities here."""
    faults = judge_line_faults(ep) if ep.close.line_holds < 2 and ep.steps_left() > ep.reserve() + 2 else []
    if not faults:
        return ''
    ep.close.line_holds += 1
    if all(' is held at ' in f for f in faults):
        fixed = basket_fix_counts(ep, basket_count_fixes(ep))
        faults = judge_line_faults(ep) if fixed else faults
        if not faults:
            return ("Not run yet (no step used). Desk set the cart's quantities itself: " + '; '.join(fixed)
                    + '. If the store refused any of these, fix those lines; then check out.' + close_cart_now(ep))
    ep.log('line_hold', faults=faults[:6])
    return ('Not run yet (no step used). The cart does not match the request yet: ' + '; '.join(faults[:6])
            + ". Correct those lines, adding or removing units the way the cart tool provides, and then check out." + close_cart_now(ep))

def close_hold_review(ep, job: Call) -> str:
    read = scout_read_pages(ep, basket_ids(ep), ep.reserve() + 3) + scout_cheaper_rows(ep)
    verdict = judge_check_cart(ep)
    if not verdict:
        close_ref_pass(ep)
        return ''
    seen = "Desk read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n' if read else ''
    return 'Not run yet. ' + job.chain + seen + verdict + close_cart_now(ep)

def close_ref_pass(ep) -> list[str]:
    """Open, one hop, the listings that the cart lines' own pages send the reader to for the rest of their details."""
    # real races 12-16 (all agents, same lines seen both ways): a final cart line whose page sent the reader on to another
    # listing passed 35/53 when that listing was opened vs 42/201 when not (r14 17/17 vs 38/160, r13 17/32 vs 22/102); the
    # passers searched the named code and opened the hit, one hop (Arena Cairn traces). At most two such reads per checkout
    judge = ep.judge
    held = set(basket_ids(ep))
    shown: list[str] = []
    for code in [c for c in judge.refs if c not in judge.refs_read][:2]:
        if ep.finished or ep.past_stop() or ep.steps_left() < ep.reserve() + 3:
            break
        judge.refs_read.add(code)
        ident = code_listing(code, ep.mem.rows)
        if ident not in ep.mem.rows:
            ident = scout_code_row(ep, code)
        if not ident or ident in held or ident in ep.mem.pages or ep.steps_left() < ep.reserve() + 2:
            continue
        shown += scout_read_pages(ep, [ident], ep.reserve() + 1)
    if shown:
        ep.log('ref_pass', refs=judge.refs, read=len(shown))
    return shown

def scout_code_row(ep, code: str) -> str:
    """Search a listing code once in the free-text field; the row whose title (else record) carries it, or ''."""
    plan = scout_search_template(ep)
    rx = code_pattern(code)
    if not plan or rx is None:
        return ''
    tool, base, field = plan
    args = shaped({**base, field: code}, ep.schemas.get(tool))
    reply = None if args is None else ep.store.call(tool, args)
    if reply is None:
        return ''
    seen = reply.get('observation')
    ep.ingest(tool, seen)
    found = [i for i, _ in mem_listings_in(ep, seen)]
    return code_listing(code, {i: ep.mem.rows[i] for i in found if i in ep.mem.rows}) if found else ''

def close_hold_stale(ep, job: Call) -> str:
    if not job.reread or ep.llm.cost <= SPEND_SOFT:
        return ''
    return ("Not run yet (no step used). Desk re-read the cart's listings just now:\n" + job.reread
            + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.')

def close_hold_audit(ep) -> str:
    review = judge_audit(ep)
    return 'Not run yet (no step used). ' + review if review else ''

def close_hold_leads(ep) -> str:
    """Hold an ending while review leads exist."""
    if not ep.leads or ep.steps_left() <= 4 or ep.close.lead_holds > 1:
        return ''
    ep.close.lead_holds += 1
    if ep.close.lead_holds == 1:
        ep.log('refuse_blocked', leads=list(ep.leads))
        return ('Not run yet (no step used). Listings that look like fits turned up in the review: ' + ', '.join(ep.leads) + '. Read any you '
                'need to, cart the best fit by the pick rule and check out. Ending without a purchase is allowed only if each of them clearly '
                'fails a requirement: name the failing one in your text and send the ending call again.')
    first = ep.leads[0]
    if not basket_passes(ep, [first]) or first not in ep.leads:
        return ''
    ep.confirmed = first
    ep.log('refuse_blocked', confirmed=first)
    return (f"Not run yet (no step used). A separate check of the records of {first} finds every requirement met. Cart it at the "
            'requested quantity and check out; end without a purchase only when its record plainly breaks a requirement.')

def close_hold_second_end(ep) -> str:
    if not NO_SECOND_END or not ep.end_refusals or ep.close.end_holds >= 2:
        return ''
    ep.close.end_holds += 1
    ep.log('second_end_held', refused=ep.end_refusals)
    return ('Not run (no step used): the store already turned down an ending without a purchase in this episode, so another would fail too. '
            "Act on the store's reply, put the best fitting listing read so far for every item in the cart at its requested quantity, "
            'and check out.' + close_cart_now(ep))

def close_hold_late_end(ep) -> str:
    """Put a once-complete cart back, once."""
    # real races 5-14: 118 failed episodes dropped a carted paid listing still in stock and in
    # budget, then ended unbought, against at most 37 right endings after a cart
    basket = ep.basket
    if ep.close.late_held:
        return ''
    lines = [i for i in basket.built if not ep.is_gone(i)]
    if ep.steps_left() > 5:
        # before the last steps only a cart no requirement was added to since, inside the budget in force
        cap = ep.numeric_budget()
        if {need for _, need in ep.live_needs()} - basket.built_needs:
            return ''
        if cap is not None and sum((mem_price(ep, i) or 0) * max(1, basket_units(ep, i)) for i in lines) > cap:
            return ''
    fresh = basket_fresh_view(ep)
    missing = [i for i in lines if i not in basket_ids(ep) or (i in basket.dropped and not fresh)]
    if len(lines) < len(ep.items) or len(missing) + 1 > ep.steps_left():
        return ''
    ep.close.late_held = True
    ep.log('late_decline_held', lines=lines, missing=missing)
    redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else 'Check out the cart as it is.'
    return ('Not run yet (no step used). Ending unbought scores nothing if a fitting listing exists, and being short of steps is no proof '
            'that none does. ' + redo + ' End without a purchase only if your search found a requested item with no listing meeting its '
            'requirements; in that case send the ending call again.')

def close_hold_unsearched(ep) -> str:
    room = min(ep.steps_left(), ep.clock_room())
    if ep.leads or ep.close.pushed or room <= ep.max_steps // 3:
        return ''
    ep.close.pushed = True
    return (f'Not run yet (no step used). {room} steps remain; use them on searches still untried: open the category list of each '
            "requested item and page further with the store's offset or token, query lone value words and alternative names of the "
            'property, and open every listing that might fit. End without a purchase only after that.')

def close_hold_late_review(ep) -> str:
    done = close_decline_review(ep)
    return done + (' [episode finished]' if ep.finished else '') if done else ''

def close_decline_review(ep) -> str:
    """One review before an unreviewed ending."""
    # real races 11-14: most unbought endings of the know line on payable tasks came with 6 steps
    # left or fewer; love's own study put 7 of 9 at exactly 5
    if ep.judge.audited or ep.basket.completed or ep.finished or not ep.roles.get('cart') or not ep.roles.get('buy'):
        return ''
    if ep.steps_left() < len(ep.items) + 2 or ep.elapsed() > CLOCK_STOP - 30 or ep.llm.cost > SPEND_HARD:
        return ''
    judge_audit(ep, len(ep.items) + 1)
    ep.log('decline_review', left=ep.steps_left(), leads=list(ep.leads))
    if not ep.leads:
        return ''
    ep.confirmed = ep.leads[0]
    return basket_complete_cart(ep)

def close_hold_closest(ep) -> str:
    judge_closest_hunt(ep)
    return ''

def close_end_by_lead(ep, job: Call) -> str:
    """Buy a confirmed lead, else quiet the ending."""
    done = basket_complete_cart(ep)
    if done:
        return done + (' [episode finished]' if ep.finished else '')
    job.args = close_quiet(ep, job.name, job.args)
    return ''

def close_hold_gate(ep, job: Call) -> str:
    if ep.steps_left() < 2:
        return ''
    go, said = gates_run(ep, 'order' if ep.mode_ok('buy', job.name, job.args) else 'end')
    if not go:
        return 'Not run yet. ' + said + close_cart_now(ep)
    if said:
        job.extra = '\n[Desk] ' + said + job.extra
    return ''

def close_quiet(ep, name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Free-text ending arguments."""
    schema = ep.schemas.get(name) or {}
    out = dict(args)
    for key, spec in (schema.get('properties') or {}).items():
        if not isinstance(spec, dict) or 'string' not in type_names(spec) or 'enum' in spec:
            continue
        value = out.get(key)
        offer = judge_offer_note(ep, spec.get('maxLength', 400) if isinstance(spec.get('maxLength'), int) else 400)
        if offer:
            out[key] = offer
            ep.log('offer_note', text=offer)
            continue
        if value is None and key not in (schema.get('required') or []):
            continue
        text = str(value or '')
        named = any(ident in text for ident in ep.mem.rows) or ep.brief.get('quiet_decline') is not False
        if named or not text:
            out[key] = 'Ending with no purchase: no listing here satisfies every requirement you gave.'[:spec.get('maxLength', 200)]
    return out

CLOSE_STEPS = {
    'offer': lambda ep, job: close_offer_pass(ep), 'wider': lambda ep, job: close_hold_wider(ep), 'changed': close_hold_changed,
    'sold': lambda ep, job: close_hold_sold(ep), 'chain': close_hold_chain, 'misfit': lambda ep, job: close_hold_misfit(ep),
    'lines': lambda ep, job: close_hold_lines(ep), 'review': close_hold_review, 'stale': close_hold_stale,
    'audit': lambda ep, job: close_hold_audit(ep), 'leads': lambda ep, job: close_hold_leads(ep),
    'second_end': lambda ep, job: close_hold_second_end(ep), 'late_end': lambda ep, job: close_hold_late_end(ep),
    'late_review': lambda ep, job: close_hold_late_review(ep), 'unsearched': lambda ep, job: close_hold_unsearched(ep),
    'closest': lambda ep, job: close_hold_closest(ep), 'by_lead': close_end_by_lead, 'gate': close_hold_gate,
}

def close_finish(ep, again: bool = False) -> str:
    """The desk's own ending of the episode."""
    if ep.finished or ep.store.blocked or ep.steps_left() <= 0:
        return ''
    basket = ep.basket
    buy, refuse = ep.roles.get('buy'), ep.roles.get('refuse')
    fresh = basket_fresh_view(ep)
    holding = bool(basket_ids(ep) if fresh else basket.lines)
    # a purchase missing a requested item earns nothing: such a cart ends unbought instead
    short = holding and refuse is not None and len(basket_ids(ep)) < len(ep.items)
    if short:
        ep.log('short_cart_end', cart=basket_ids(ep))
    orderable = holding or (basket.touched and not fresh and ep.steps_left() >= 2)
    args = shaped(dict(ep.brief.get('buy_args') or {}), ep.schemas.get(buy)) if buy and not short and orderable else None
    if args is not None:
        broken = close_blind_broken(ep)
        if broken and refuse:
            ep.log('forced_buy_blocked', ids=broken)
            short = True
        else:
            said = close_forced_buy(ep, buy, args, again)
            if said is not None:
                return said
    if refuse and (short or not holding):
        done = close_decline_review(ep)
        if ep.finished or done:
            return '' if ep.finished else done
    return close_forced_refuse(ep, refuse) if refuse else ''

def close_blind_broken(ep) -> list[str]:
    """Read the cart lines never opened; those whose page breaks their item's requirements."""
    # the desk's own checkout (clock or step stop) first reads lines never opened and each unread line's offer
    # page: store reads are quick, and an order with an unread line misses the process
    blind = [i for i in basket_ids(ep) if i not in ep.mem.pages]
    if not blind or ep.steps_left() < len(blind) + 2:
        return []
    scout_read_pages(ep, blind, 2)
    cat_key = mem_kind_key(ep)
    if not cat_key:
        return []

    def breaks(i: str) -> bool:
        owner = mem_owner(ep, mem_label(ep, i, cat_key))
        return 0 < owner <= len(ep.items) and bool(judge_against(ep, i, ep.items[owner - 1].live()))
    return [i for i in blind if breaks(i)]

def close_forced_buy(ep, buy: str, args: dict[str, Any], again: bool) -> str | None:
    """Offer pass, order gate and the desk's checkout; None hands the ending on to the decline."""
    over = lambda: ep.finished or ep.steps_left() <= 0
    close_offer_pass(ep)
    if over():
        return ''
    if ep.steps_left() >= 2:
        gates_run(ep, 'order')
    if over():
        return ''
    reply = ep.store.call(buy, args)
    ep.log('forced_buy', reply=clip(reply, 200))
    if over():
        return ''
    if reply is None or not again:
        return None
    ep.checkouts += 1
    ep.refused_at = ep.step
    ep.ingest(buy, reply.get('observation'))
    return 'the checkout was refused: ' + ep.render(reply.get('observation'))

def close_forced_refuse(ep, refuse: str) -> str:
    args = shaped(close_quiet(ep, refuse, dict(ep.brief.get('refuse_args') or {})), ep.schemas.get(refuse))
    if args is None:
        return ''
    if ep.steps_left() >= 2:
        gates_run(ep, 'end')
    if ep.finished or ep.steps_left() <= 0:
        return ''
    reply = ep.store.call(refuse, args)
    ep.log('forced_refuse', reply=clip(reply, 200))
    if reply is None or ep.finished:
        return ''
    ep.end_refusals += 1
    ep.refused_at = ep.step
    ep.ingest(refuse, reply.get('observation'))
    return 'ending without a purchase was refused: ' + ep.render(reply.get('observation'))

class Screener:
    """Screens model calls before they run."""

    def __init__(self) -> None:
        self.page_log: list[tuple[str, dict[str, Any], set[str], bool]] = []
        self.dead: dict[tuple[str, str], tuple[str, int] | None] = {}
        self.code_misses = 0
        self.code_found = False
        self.cheaper_held: set[str] = set()
        self.blind_held: set[str] = set()

def screen_screen(ep, fn: dict[str, Any], batch: int) -> Call | str:
    """A Call, or the answer instead of running it."""
    if ep.finished:
        return 'Not run: the episode is over.'
    name = fn.get('name')
    allowed = sorted(t['function']['name'] for t in ep.loop.offered())
    listed = ', '.join(allowed)
    gates = ((name == ep.roles.get('ask'), "Not run (no step used): Desk's interviewer sends every message to the shopper."),
             (name not in ep.schemas, f'Not run (no step used): no such tool. Tools: {listed}.'),
             # Arena b10: the short form made the model believe its tools were gone (a decline with steps left; turns
             # without a call); say what the remaining steps are kept for and which calls work now
             (name not in allowed, f'Not run (no step used): {ep.steps_left()} steps are left and Desk keeps them for finishing the '
                                   f'order, so this tool is paused (nothing is broken). Calls that work now: {listed}. Use them: cart a '
                                   'line that meets every requirement and check out, or end the way the request asks when nothing fits.'))
    stop = next((text for shut, text in gates if shut), '')
    if stop:
        return stop
    raw = fn.get('arguments')
    if isinstance(raw, dict):
        args = raw
    elif (raw or '').strip():
        args = parse_object(raw)
        if args is None:
            return 'Not run (no step used): the arguments were not a single JSON object.'
    else:
        args = {}
    faults: list[str] = []
    args = conform(args, ep.schemas[name], 'args', faults)
    if faults:
        return 'Not run (no step used): ' + '; '.join(faults[:6])
    args, wrong = screen_resolve_ids(ep, name, args)
    if wrong:
        return f'Not run (no step used): {wrong}.'
    ending = ep.mode_ok('buy', name, args) or ep.mode_ok('refuse', name, args)
    job = Call(name, args, ending, ep.side_param() if name == ep.roles.get('compare') else None)
    for rule in (screen_fit_room, screen_first_page, screen_page_value, screen_removal_count, screen_gone_target, screen_absent_line):
        said = rule(ep, job)
        if said:
            return said
    if job.ending and batch > 1:
        return 'Not run (no step used): read your other results first, then send the finishing call on its own.'
    for rule in (screen_tail_kept, screen_pointer_hold, screen_unread_line, screen_cheaper_seen, screen_repeated, screen_cached, screen_shown, screen_page_stop, screen_dead_filter, screen_dead_code):
        said = rule(ep, job)
        if said:
            return said
    return job

def screen_fit_room(ep, job: Call) -> str:
    """Trim side-by-side calls to the known room."""
    ids, room = job.listed(), ep.side_room()
    if ids is not None and len(ids) > room and ep.side_bad:
        job.extra = f"\n[Desk] Sent only the first {room} listings: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, ids[room:]))}."
        job.args = {**job.args, job.many: ids[:room]}
    return ''

def screen_first_page(ep, job: Call) -> str:
    """Start a new query from its first page."""
    paging = scout_paging_arg(ep, job.name) if job.name == ep.roles.get('search') else None
    if not paging and job.name == ep.roles.get('search'):
        return screen_loose_numbers(ep, job)
    value = job.args.get(paging) if paging else None
    if value in (None, '', 0, '0', 1, '1') or isinstance(value, bool):
        return ''
    base = {k: v for k, v in job.args.items() if k != paging}
    if any({k: v for k, v in s.items() if k != paging} == base for s in ep.scout.searched):
        return ''
    schema = ep.schemas.get(job.name) or {}
    if paging in (schema.get('required') or []):
        spec = (schema.get('properties') or {}).get(paging) or {}
        if not {'integer', 'number'} & set(type_names(spec)):
            return ''
        base[paging] = spec.get('minimum', 0) if isinstance(spec.get('minimum'), int) else 0
    job.args = base
    job.extra += f'\n[Desk] Sent from the first page: {paging} {value!r} belonged to an earlier query, and a new query starts at its best rows.'
    return ''

def screen_loose_numbers(ep, job: Call) -> str:
    """Before a result has shown what they do, send a new query without the numbers its schema leaves unexplained: one
    may be a start position that skips the best rows (Arena race 17: 274 searches showed an empty page of a query that
    matched listings, 35 of 47 failed tasks had one; real race 17: such episodes passed .16 against .37)."""
    loose = [k for k in scout_loose_numbers(ep, job.name) if job.args.get(k) not in (None, 0) and not isinstance(job.args.get(k), bool)]
    if not loose:
        return ''
    base = {k: v for k, v in job.args.items() if k not in loose}
    if any({k: v for k, v in s.items() if k not in loose} == base for s in ep.scout.searched):
        return ''
    dropped = {k: job.args[k] for k in loose}
    job.args = base
    ep.log('loose_dropped', step=ep.step, args=dropped)
    if ep.scout.loose_notes < 2:
        ep.scout.loose_notes += 1
        job.extra += (f"\n[Desk] Sent without {', '.join(f'{k}={dumps(v)}' for k, v in dropped.items())}: the tool does not say what "
                      'that number does, and a new query starts at its first row.')
    return ''

def screen_page_value(ep, job: Call) -> str:
    """Page on with the value the store handed back for this query."""
    paging = scout_paging_arg(ep, job.name) if job.name == ep.roles.get('search') else None
    value = job.args.get(paging) if paging else None
    if value in (None, '', 0, '0', 1, '1') or isinstance(value, bool):
        return ''
    given = [v for v in ep.scout.next_given.get(job.name + dumps({k: v for k, v in job.args.items() if k != paging}), []) if v is not None]
    if not given or any(dumps(v) == dumps(value) for v in given):
        return ''
    job.args = {**job.args, paging: given[-1]}
    ep.log('page_value', step=ep.step, sent=value, used=given[-1])
    job.extra += f'\n[Desk] Paged with {paging}={dumps(given[-1])}, the value this search handed back ({dumps(value)} would skip rows).'
    return ''

def screen_removal_count(ep, job: Call) -> str:
    """A removal naming no count takes the whole line: send the units Desk holds (Arena r17 b11: the store took one unit,
    Desk recorded the line as gone and the model removed it again, 12 tasks, 9 failed; real r17: such episodes passed
    4/23 against .33)."""
    if job.name != ep.roles.get('cart'):
        return ''
    spec = (ep.schemas.get(job.name) or {}).get('properties') or {}
    modes = lambda key: (spec.get(key) or {}).get('enum')
    removal = any(isinstance(modes(k), list) and len(modes(k)) >= 2 and v in modes(k)[1:] for k, v in job.args.items())
    counts = [k for k, v in spec.items() if isinstance(v, dict) and {'integer', 'number'} & set(type_names(v)) and 'enum' not in v]
    ids = [v for v in job.args.values() if isinstance(v, str) and v in ep.basket.lines]
    if not removal or len(counts) != 1 or job.args.get(counts[0]) is not None or len(ids) != 1:
        return ''
    have = ep.basket.lines.get(ids[0]) or 0
    if have <= 1:
        return ''
    job.args = {**job.args, counts[0]: have}
    ep.log('removal_count', step=ep.step, line=ids[0], units=have)
    return ''

def screen_gone_target(ep, job: Call) -> str:
    target = ep.id_param(job.name)
    ident = str(job.args.get(target)) if target else ''
    if not target or job.name not in (ep.roles.get('open'), ep.roles.get('offer')) or ident not in ep.gone:
        return ''
    return f'Not run (no step used): {job.args[target]} sold out at step {ep.gone[ident]} and will not come back. Replace its cart line with another fitting listing.'

def screen_absent_line(ep, job: Call) -> str:
    """Units taken off a line not in the cart."""
    basket = ep.basket
    if job.name != ep.roles.get('cart') or not any((whole(v) or 0) < 0 for v in job.args.values() if not isinstance(v, str)):
        return ''
    view = dumps(basket.view) if basket.viewed_at >= basket.changed_at else ''
    held = (basket.lines, view, basket.reply)
    absent = next((v for v in job.args.values() if isinstance(v, str) and v in ep.mem.rows and not any(v in place for place in held)), None)
    return f'Not run (no step used): {absent} is not in the cart, so nothing can be taken off. Cart now: ' + basket_describe(ep) if absent else ''

def screen_tail_kept(ep, job: Call) -> str:
    if job.ending or ep.steps_left() > ep.reserve() or not ep.closing_tools():
        return ''
    return "Not run (no step used): the steps left are held back for checking out (and the shopper's basket approval if it is due) or for ending without a purchase."

def screen_pointer_hold(ep, job: Call) -> str:
    """Hold back a new cart line for an item whose requirement still waits on a pointed listing."""
    # real race 15, a pointer task: a first cart add made after the pointed listing was opened passed 7 of 7, one made
    # before it 15 of 33, and only after re-carting; never reaching the chain's last page passed 0 of 66 (Arena: an
    # early add led the model to treat the chain as settled and check out the early line)
    if job.ending or job.name != ep.roles.get('cart') or ep.checkouts or not ep.refs:
        return ''
    pending = ep.open_pointers()
    waiting = {n for n, need in ep.refs if need in pending}
    if not waiting:
        return ''
    in_cart = set(basket_ids(ep))
    fields = (ep.schemas.get(job.name) or {}).get('properties') or {}
    lines = basket_lines_named(ep, fields, job.args)[0]
    kind = mem_kind_key(ep)
    whole_order = all(k in waiting for k in range(1, len(ep.items) + 1))
    held = []
    for ident, units in lines:
        if ident in in_cart or not (units is None or units > 0):
            continue
        owner = mem_owner(ep, mem_label(ep, ident, kind)) if kind else 0
        if (owner in waiting) if owner else whole_order:
            held.append(ident)
    if not held:
        return ''
    room = min(ep.steps_left(), ep.clock_room())
    if room <= len(ep.items) + 6 or ep.chains.cart_holds >= 2 or ep.past_stop():
        return ''
    ep.chains.cart_holds += 1
    found = chains_recheck(ep)
    now_open = ep.open_pointers()
    still = [need for n, need in ep.refs if n in waiting and need in now_open]
    ep.log('pointer_hold', ids=held, pending=still, step=ep.step)
    ids = ', '.join(held)
    if still:
        return (f"Not run (no step used): {ids} belongs to an item whose requirement takes its value from another listing, and "
                'that chain is still open: ' + ' | '.join(still[:3]) + '. Follow it: open the named listing and every listing its page leads '
                'to, take the value from the final page, and cart a listing carrying it (other items may be carted already now).\n' + found)
    known = '; '.join(f'{need} -> {ep.ref_values[need]}' for need in pending if need in ep.ref_values)
    return (f"Not run (no step used): {ids} was chosen before the pointed value was known. It is known now: {known}. "
            'Cart a listing whose own page shows that value (send the same call again if this one does).\n' + found)

def screen_unread_line(ep, job: Call) -> str:
    """Hold once a new cart line whose listing's attributes were never read, while a read still fits."""
    # Arena r12-r16 b09 vs passing runs: lines carted straight from a search row (one by the fallback model) broke a
    # requirement or missed the process; the passing runs read every line before carting it
    tool = ep.attr_tool()
    if job.ending or job.name != ep.roles.get('cart') or not tool or ep.steps_left() <= ep.reserve() + 2:
        return ''
    held_now = set(basket_ids(ep))
    named, _ = basket_lines_named(ep, (ep.schemas.get(job.name) or {}).get('properties') or {}, job.args)
    blind = [i for i, units in named if (units is None or units > 0) and i not in held_now and i not in ep.mem.pages
             and i not in ep.screen.blind_held]
    if not blind:
        return ''
    ep.screen.blind_held.update(blind)
    ep.log('unread_line_hold', ids=blind)
    return (f"Not run (no step used): {', '.join(blind)} was never opened, so its attributes are unchecked. Open it with {tool} first "
            'and cart it only if it meets every requirement of its item (send the same call again if it does).')

def screen_cheaper_seen(ep, job: Call) -> str:
    """Under a lowest-price pick, hold a new cart line once when a cheaper opened listing of its kind shows no clash."""
    # Arena r16 noask b02: 6 'not the best option' orders; in tight 35-step episodes the checkout came with 1-2 steps left,
    # so no close review ran, and a cheaper opened listing meeting every requirement stayed out of the cart
    held = ep.screen.cheaper_held
    if job.ending or job.name != ep.roles.get('cart') or not ep.cheapest() or len(held) >= 2:
        return ''
    cat_key = mem_kind_key(ep)
    if not cat_key:
        return ''
    carted = set(basket_ids(ep))
    named, _ = basket_lines_named(ep, (ep.schemas.get(job.name) or {}).get('properties') or {}, job.args)
    notes = []
    for ident, units in named:
        price = mem_price(ep, ident)
        skip = (units is not None and units <= 0) or ident in carted or price is None or ident in held
        note = '' if skip else screen_cheaper_line(ep, ident, price, cat_key, carted)
        if note:
            held.add(ident)
            notes.append(note)
    if not notes:
        return ''
    ep.log('cheaper_hold', notes=notes)
    return ('Not run (no step used): the lowest total wins here. ' + ' | '.join(notes) + '. Open any of those pages not yet read and test '
            'each against all of the item\'s requirements; cart the cheapest that passes every one. If none passes, name the broken '
            'requirement for each in your text and repeat this same call.')

def screen_cheaper_line(ep, ident: str, price: float, cat_key: str, carted: set[str]) -> str:
    """The note for one new line with cheaper listings of its kind, opened or only seen in results; '' when none."""
    kind = mem_label(ep, ident, cat_key).strip().lower()
    owner = mem_owner(ep, kind) if kind else 0
    needs = ep.items[owner - 1].live() if 0 < owner <= len(ep.items) else [n for _, n in ep.live_needs()]
    pointers = {m for _, m in ep.refs}
    stated = [n.split(':', 1) for n in needs if ':' in n and n not in pointers]
    props = [stems(key) for key, _ in stated]
    same_kind = lambda i: i != ident and i not in carted and mem_can_buy(ep, i) and not ep.is_gone(i)

    def clean(i: str) -> bool:
        if i in ep.basket.dropped or mem_label(ep, i, cat_key).strip().lower() != kind or not ep.price_rank(i) < price:
            return False
        if judge_apart_needs(ep, i, needs) or judge_against(ep, i, needs):
            return False
        # a page that leaves a required property out is turned down by the model anyway (Arena r14 b06: holds named
        # cheaper pages without the required depth, the model kept its pick and the paid cheaper fit went unnamed)
        block = fact_block(ep.mem.pages.get(i) or {}) or {}
        return all(any(names_same(prop, stems(k)) for k in block) for prop in props)
    opened = sorted((i for i in ep.mem.pages if same_kind(i) and clean(i)), key=ep.price_rank)
    # rows of its kind seen in results but never read, cheaper than the line: the best fit under a lowest-price
    # rule sat unread in an earlier result in 6 of 42 traced failures (Arena r12-r16 b09 vs passing runs)
    # most required values shown in the row first, then price (real r12-r16: among rows seen but never opened,
    # cheapest-first put the paid row first in 9% and in the top 3 in 30% of 114 misses)
    wanted = [set(plain_words(value)) for _, value in stated if not has_bound(value)]

    def rank(i: str) -> tuple[int, float]:
        words = set(plain_words(dumps(ep.mem.rows.get(i) or {})))
        return -sum(1 for w in wanted if w and w <= words), ep.price_rank(i)
    seen = [i for i in ep.mem.order if i not in ep.mem.pages and same_kind(i)
            and mem_label(ep, i, cat_key).strip().lower() == kind and ep.price_rank(i) < price]
    unread = sorted(seen, key=rank)[:3]
    if ep.steps_left() <= ep.reserve() + len(unread) + 2:
        unread = []
    if not (opened or unread):
        return ''
    priced = lambda ids: ', '.join(f'{i} ({ep.price_rank(i):g})' for i in ids)
    parts = [f'opened listings of its kind cost less: {priced(opened[:3])}'] * bool(opened)
    if unread:
        parts.append(f'rows of its kind not read yet cost less: {priced(unread)} (those showing more required values first)')
    return f"{ident} costs {price:g}, yet " + '; '.join(parts)

def screen_repeated(ep, job: Call) -> str:
    if job.ending or job.name == ep.roles.get('cart') or job.mark() not in ep.recent:
        return ''
    return 'Not run (no step used): this exact call was just made and nothing has changed since; see its result above.'

def screen_cached(ep, job: Call) -> str:
    hit = ep.cache.get(job.mark()) if job.name in ep.reading_tools() else None
    if not hit or hit[0] != ep.version:
        return ''
    return f'Not run (no step used): the same call at step {hit[1]} returned this, and nothing has changed since:\n' + hit[2][:3000]

def screen_shown(ep, job: Call) -> str:
    """Skip views showing nothing new."""
    target = ep.id_param(job.name)
    if target and job.name in ep.reading_tools() and job.name != ep.roles.get('compare') and mem_fully_shown(ep, job.name, str(job.args.get(target))):
        return f'Not run (no step used): {job.args[target]} has shown everything this view gives since the last change. Its record: ' + mem_record_text(ep, str(job.args[target]))
    ids = job.listed()
    if not ids:
        return ''
    fresh = [i for i in ids if not mem_fully_shown(ep, job.name, str(i))]
    if not fresh:
        return ('Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n'
                + '\n'.join(f'{i}: {mem_record_text(ep, str(i))}' for i in ids)[:4000])
    if len(fresh) < len(ids):
        job.extra += f"\n[Desk] Left out {', '.join(str(i) for i in ids if i not in fresh)}: already shown in full, nothing changed since."
        job.args = {**job.args, job.many: fresh}
    return ''

def screen_page_stop(ep, job: Call) -> str:
    """P7: stop paging a search gone dry."""
    if job.ending or job.name in ep.role_tools('search'):
        return ''
    paged = screen_page_of(ep, job)
    if not paged:
        return ''
    key, base = paged
    run = [found for name, args, _, found in ep.screen.page_log if name == job.name and dumps({k: v for k, v in args.items() if k != key}) == base]
    empty = 0
    for found in reversed(run):
        if found:
            break
        empty += 1
    if empty < EMPTY_PAGES:
        return ''
    ep.log('page_stop', step=ep.step, tool=job.name, args=clip(job.args, 80), empty=empty)
    return (f"Not run (no step used): beyond this point the search only pages through the store's unrelated rows; none of its last {empty} "
            'pages brought a listing of a requested kind with these words. Use other words: search the values the listing ought to show, '
            "alone or with its seller's name.")

def screen_dead_filter(ep, job: Call) -> str:
    """Refuse a seller filter that never found rows."""
    for key, value in job.args.items():
        dead = ep.screen.dead.get((job.name, key))
        if dead and isinstance(value, str) and value.strip():
            ep.log('dead_filter', step=ep.step, tool=job.name, key=key)
            return (f'Not run (no step used): the {key} filter of {job.name} returned no rows for {dead[0]!r} at step {dead[1]}; in a store '
                    f"where it finds nothing for one name it finds nothing for others either. Search without the {key} filter: the seller's or "
                    "maker's name as words with the product's kind, or the category.")
    return ''

def screen_code_words(ep, job: Call) -> set[str]:
    """Known model codes in a search."""
    if job.name != ep.roles.get('search'):
        return set()
    said = {c for v in job.args.values() if isinstance(v, str) for c in model_codes(v)}
    if not said:
        return set()
    index = judge_fit_index(ep)
    works = [w for m in index['marks'].values() for w in m['works'] if not w.startswith('!')]
    return said & (set(index['known']) | set(model_codes(ep.request)) | codes_named(works, said))

def screen_dead_code(ep, job: Call) -> str:
    """Refuse code searches this store cannot answer."""
    if ep.screen.code_found or ep.screen.code_misses < 2:
        return ''
    codes = screen_code_words(ep, job)
    if not codes:
        return ''
    ep.log('dead_code', step=ep.step, codes=sorted(codes))
    plan = judge_fit_plan(ep, short=True)
    return (f"Not run (no step used): model codes are not indexed by this store's search ({ep.screen.code_misses} code searches came back "
            "empty). Query the product's kind with plain values, or list its category and page on, then read the codes in each row's text; "
            "Desk matches row codes itself." + (f' Codes the seen parts name: {plan}' if plan else ''))

def screen_own_words(ep, args: dict[str, Any], skip: str | None) -> set[str]:
    """Search words not naming a requested kind."""
    mem = ep.mem
    kind: set[str] = set()
    for item in ep.items:
        kind |= roots(item.name) | roots(item.shelf or '')
    cat_key = mem_kind_key(ep)
    for label in {str(r.get(cat_key) or '') for r in mem.rows.values()} if cat_key else set():
        if mem_owner(ep, label):
            kind |= roots(label)
    text = ' '.join(str(v) for k, v in args.items() if k != skip and isinstance(v, str))
    return roots(re.sub('[^A-Za-z0-9]+', ' ', text)) - kind

def screen_note_page(ep, job: Call, before: set[str]) -> None:
    """Record what a search brought."""
    seen, screen = ep.loop.last_obs, ep.screen
    shown: set[str] = set()
    if isinstance(seen, dict):
        shown = {dumps(v) for v in seen.values() if isinstance(v, (str, int)) and not isinstance(v, bool)}
    paged = screen_page_of(ep, job)
    words = screen_own_words(ep, job.args, paged[0] if paged else None)
    kind = mem_kind_key(ep)
    listed = mem_listings_in(ep, seen)
    fresh = (ep.mem.rows.get(ident) or record for ident, record in listed if ident not in before)
    owned = (row for row in fresh if not kind or mem_owner(ep, str(row.get(kind) or '')))
    found = any(not words or words & roots(re.sub('[^A-Za-z0-9]+', ' ', dumps(row))) for row in owned)
    screen.page_log = [*screen.page_log, (job.name, dict(job.args), shown, found)][-40:]
    if screen_code_words(ep, job):
        # a search holding a model code read on a fit text or in the request: real r11 and r15 stores never index such
        # codes (after two came back empty, later ones listed rows 5 times in 2,496), r8, r12 and r13 stores do (.67-.94)
        if listed:
            screen.code_found = True
        else:
            screen.code_misses += 1
    fields = (ep.schemas.get(job.name) or {}).get('properties') or {}
    for key, value in job.args.items():
        if not (isinstance(value, str) and value.strip() and SELLER_ARG.search(key)) or (fields.get(key) or {}).get('enum'):
            continue
        slot = (job.name, key)
        if listed:
            screen.dead[slot] = None
        elif slot not in screen.dead:
            screen.dead[slot] = (value, ep.step)

def screen_page_of(ep, job: Call) -> tuple[str, str] | None:
    """(paging arg, base) if this pages a search."""
    known = scout_paging_arg(ep, job.name)
    for name, args, offered, _ in reversed(ep.screen.page_log):
        if name != job.name:
            continue
        diff = [k for k in set(args) | set(job.args) if args.get(k) != job.args.get(k)]
        if len(diff) != 1 or job.args.get(diff[0]) in (None, '', 0, '0', 1, '1') or isinstance(job.args.get(diff[0]), bool):
            continue
        if diff[0] == known or dumps(job.args[diff[0]]) in offered:
            return diff[0], dumps({k: v for k, v in job.args.items() if k != diff[0]})
    return None

def screen_resolve_ids(ep, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Map titles given as ids to listing ids."""
    mem = ep.mem
    takers = {ep.roles.get(r) for r in ('cart', 'open', 'offer', 'compare')} - {None}
    if name not in takers or not mem.id_key:
        return args, ''
    props = (ep.schemas.get(name) or {}).get('properties') or {}
    free = [k for k, v in props.items() if isinstance(v, dict) and {'string', 'array'} & set(type_names(v)) and 'enum' not in v]
    if len(free) != 1 or (name != ep.roles.get('cart') and (name, free[0]) not in ep.id_args):
        return args, ''
    key = free[0]
    if key not in args:
        return dict(args), ''
    many = isinstance(args[key], list)
    fixed = [screen_listing_named(ep, v) for v in (args[key] if many else [args[key]])]
    problem = next((p for _, p in fixed if p), '')
    if problem:
        return args, problem
    return {**args, key: [v for v, _ in fixed] if many else fixed[0][0]}, ''

def screen_listing_named(ep, value: Any) -> tuple[Any, str]:
    mem = ep.mem
    if isinstance(value, str) and value.strip() in mem.rows:
        return value.strip(), ''
    if not isinstance(value, str) or id_like(value, True) or len(value.strip()) < 8:
        return value, ''
    start = value.strip().lower()
    hits = [i for i, row in mem.rows.items() if any(k != mem.id_key and isinstance(v, str) and v.strip().lower().startswith(start) for k, v in row.items())]
    if len(hits) == 1:
        return hits[0], ''
    if hits:
        return value, f"several listings carry the text {value!r} ({', '.join(hits[:6])}); pass one listing id"
    return value, f'{value!r} is not a listing id; pass the id from the result rows (their {mem.id_key} field)'

def status_unfound(ep) -> str:
    """A hint for items no row of whose kind turned up after several searches."""
    # Arena b10: items with no candidate after repeated searches (the same words resent); real r12-r16: the paid row was
    # first seen through a query led by the request's values 61% of the time, a kind-only query 15%
    cat_key = mem_kind_key(ep)
    if not cat_key or len(ep.scout.searched) < 2 or ep.checkouts:
        return ''
    labels = [mem_label(ep, i, cat_key) for i in ep.mem.rows]
    empty = [n for n in scout_open_items(ep) if not any(mem_belongs(ep, ep.items[n - 1], label) for label in labels if label)]
    if not empty:
        return ''
    return (f"[no listing of item {', '.join(map(str, empty))}'s kind seen yet] search its stated values together with the product noun; "
            'write a number both with and without a space before its unit; or use a noun that rows of that category carry in their titles.')

class TurnLoop:
    """The model's turns and the calls they make."""

    def __init__(self, ep: Episode):
        self.ep = ep
        self.chat: list[dict[str, Any]] = []
        self.turns = 0
        self.misses = 0
        self.last_obs: Any = None

    def status(self) -> str:
        # the step count is not the binding limit on long tasks: real r15 multi-product episodes of the whole KT line
        # ended near step 48 of 100+, and Arena r15 had 24 of 49 failures ending at the clock with steps unused
        ep = self.ep
        steps = ep.steps_left()
        left = min(steps, ep.clock_room())
        unbought = not ep.checkouts
        text = f'[Desk] step {ep.step} of {ep.max_steps} used, {steps} left.'
        if left < steps:
            text += f' At this pace the clock allows only about {left} more steps, so plan for {left}.'
        if left <= ep.reserve() + 1:
            text += (' FINAL STEPS: if every item has a line in the cart, check out now. Ending without a purchase needs your search to have shown '
                     'an item with no fitting listing; a lack of steps proves nothing of the kind.')
        elif left <= 6 or unbought and left <= len(ep.items) + 5:
            text += ' Few steps left: set the cart to the chosen lines and check out now.'
            if self.read_room():
                gap = 'An item has no cart line yet' if basket_ids(ep) else 'No line is in the cart yet'
                which = 'the cheapest such row' if ep.cheapest() else 'one whose row shows the wanted values'
                text += f' {gap}: read the most promising unread listing ({which}) with {ep.attr_tool()}, cart it if it meets every requirement, then check out.'
        elif unbought and left <= 12:
            text += ' Only a few steps remain: decide each pick, put it in the cart and finish the checkout before they run out.'
        elif unbought and not ep.basket.touched and (ep.step >= ep.max_steps // 2 or steps > left <= ep.step):
            text += " The cart is still empty with half the steps or time gone: add the fitting listing for each item now, since the store can still sell one out."
        lookups = ep.open_pointers() if unbought else []
        if lookups:
            text += ('\n[resolve first: for each of these, open the listing it points at, then every page it points on to; the value comes '
                     'from the final page, or from what a pairing rule there sets against that page\'s own value] ' + ' | '.join(lookups[:3]))
        if ep.leads:
            text += '\n[review candidates] ' + ', '.join(ep.leads)
        bare = status_unfound(ep)
        if bare:
            text += '\n' + bare
        plan = judge_fit_plan(ep) if ep.bound and unbought else ''
        if plan:
            text += '\n' + plan
        if ep.notes:
            text += '\n[changes so far] ' + ' | '.join(ep.notes[-4:])
        return text + '\n[ledger]\n' + req_ledger(ep)

    def offered(self) -> list[dict[str, Any]]:
        """Tools offered now (narrowed near the end)."""
        ep, roles = self.ep, self.ep.roles
        left, reserve = ep.steps_left(), ep.reserve()
        ends = {roles.get('buy'), roles.get('refuse')} - {None}
        tools = [t for t in ep.tools if t['function']['name'] != roles.get('ask')]
        if left <= reserve:
            # Arena r17 b11: at 3 left (1 + the 2-step refusal room) a missing line just read, or the swap a final review
            # named, could not be carted (11 refused cart calls, 9 failed); real r17: first endings refused with <= 5 steps
            # left passed 1/8, so the refusal room yields the cart tool while no ending was tried
            also: tuple[str, ...] = ('cart',) if ep.refusal_room() and left >= 2 else ()
        elif left <= reserve + 3:
            also = ('cart', 'cart_view')
        elif left <= len(ep.items) + 5 and ep.checkouts == 0:
            also = ('cart', 'cart_view', 'offer')
        else:
            return tools
        if not ends:
            return tools
        keep = ends | {roles.get(role) for role in also}
        if ep.unoffered():
            keep.add(ep.offer_view())
        if (ep.end_refusals or ep.refused_at >= 0) and left >= 2:
            # a refused ending is answered by a cart change (Arena r13 noask b10: refused decline at step 48 of 50, the
            # store dropping a requirement; only the ending was offered and the model wrote that it had no cart tool)
            keep |= {roles.get('cart'), roles.get('cart_view')}
            if left >= 3:
                keep.add(ep.attr_tool())
        room = self.read_room()
        if room:
            # an item without a cart line may need one listing read before it is carted; the narrowed tools left only the
            # ending (Arena r11/r14: the model stalled there and the episode ended unbought)
            keep.add(ep.attr_tool())
            if left >= reserve + 2 + room:
                # Arena b27-b32: 8 of 17 unbought endings on order tasks came after turns without a call while the
                # narrowed tools held no search
                keep.add(roles.get('search'))
        return [t for t in tools if t['function']['name'] in keep] or tools

    def read_room(self) -> int:
        """Missing items one read each can still fill."""
        ep = self.ep
        missing = max(0, len(ep.items) - len(basket_ids(ep)))
        # reads for items without a line may use the refusal room: without a line for every item no checkout can pass
        # (Arena r16 noask b10: 3 items uncarted at 10 steps left got only the cart and ending tools)
        return missing if missing and ep.attr_tool() and ep.steps_left() >= ep.reserve() - ep.refusal_room() + 1 + missing else 0

    def trim(self) -> None:
        if sum(len(dumps(m.get('content'))) for m in self.chat) <= HISTORY_LIMIT:
            return
        # Arena r16 noask b01: loop prompts ran a median 20.9k tokens per call (hayate 3.0k), 7.2 s per call against 4.6 s;
        # 25 of 81 episodes ran into the clock stop. Older results go to stubs; the last few stay whole
        results = [i for i, m in enumerate(self.chat) if m.get('role') == 'tool']
        for index in results[:max(1, len(results) - 6)]:
            content = self.chat[index].get('content') or ''
            # a stub is not shortened again (its id list sits past the first 400 characters)
            if len(content) <= 600 or '...(older result shortened;' in content:
                continue
            ids = re.findall('"([A-Za-z0-9][A-Za-z0-9_\\-:.]{5,40})"', content)
            ids = [i for i in dict.fromkeys(ids) if i in self.ep.mem.rows][:20]
            # a shortened page read keeps the values it showed for the requirements (Arena r17 b11: in 19 tasks the model
            # took a checked page for one without the value once its result was cut; 14 failed)
            pages = [self.ep.mem.pages[i] for i in ids[:2] if i in self.ep.mem.pages] if len(ids) <= 2 else []
            said = judge_evidence(self.ep, pages, False)[:300] if pages else ''
            self.chat[index]['content'] = (content[:400] + f"...(older result shortened; listings in it: {', '.join(ids)}"
                                           + (f'; values it showed: {said}' if said else '') + ')')
        # b02 still ran 16-30k prompt tokens a call: Desk's older notes stay as stubs too (the request, the newest turns and
        # the status line each turn carries keep what is current)
        notes = [i for i, m in enumerate(self.chat) if i > 1 and m.get('role') == 'user' and isinstance(m.get('content'), str)]
        for index in notes[:max(0, len(notes) - 3)]:
            content = self.chat[index]['content']
            if len(content) > 700:
                self.chat[index]['content'] = content[:500] + '...(older note shortened)'

    def run(self) -> list[dict[str, Any]]:
        ep = self.ep
        if ep.finished:
            return ep.trace
        req_read(ep)
        ep.log('brief', brief=ep.brief)
        opening = ep.view.get('opening')
        intro = "SHOPPER'S REQUEST:\n" + ep.request
        if opening is not None:
            mem_absorb(ep, opening)
            intro += '\n\nWHAT THE STORE SHOWS FIRST:\n' + ep.render(opening)
        intro += ('\n\nFirst, for each item, run a free-text search with the product noun after one or two of its stated values that a '
                  'matching title would show (use a category filter only with a label the store has already shown). Next, open a few '
                  'listings of the right kind in full, so the interviewer learns which properties these listings carry.\n\n' + self.status())
        self.chat = [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': intro}]
        self.work(LOOP_CALLS, SPEND_HARD)
        if not ep.finished:
            basket_late_add(ep)
            turned = close_finish(ep, again=True)
            if turned and not (ep.finished or ep.store.blocked or ep.llm.out_of_credit) and ep.steps_left() > 1 and not ep.past_stop():
                listed = scout_cheaper_search(ep)
                self.chat.append({'role': 'user', 'content': '[Desk] Desk tried to end the episode and ' + turned + ('\n' + listed if listed else '')
                                  + '\nApply what the store says, then finish.\n' + self.status()})
                self.work(self.turns + 10, SPEND_HARD + 0.01)
            if not ep.finished:
                close_finish(ep)
        ep.log('end', finished=ep.finished, steps=ep.step, cost=round(ep.llm.cost, 5), calls=ep.llm.calls)
        return ep.trace

    def parse_calls(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        calls = []
        for number, call in enumerate(message.get('tool_calls') or []):
            if not isinstance(call, dict):
                continue
            fn = call.get('function') if isinstance(call.get('function'), dict) else {}
            raw = fn.get('arguments')
            parsed = raw if isinstance(raw, dict) else parse_object(raw) if isinstance(raw, str) and raw.strip() else {}
            calls.append({'id': str(call.get('id') or f'k{self.turns}_{number}'), 'type': 'function',
                          'function': {'name': str(fn.get('name') or ''), 'arguments': dumps(parsed if isinstance(parsed, dict) else {})}})
        return calls

    def work(self, loop_cap: int, spend_cap: float) -> None:
        """Planner and executor: plan a step from the state, carry it out, plan again until a step says stop."""
        pace = {'idle': 0, 'retried': False, 'nudged': False}
        going = True
        while going:
            step = self.plan_step(loop_cap, spend_cap)
            if step['do'] == 'think':
                step = self.plan_reply(self.think_once())
            going = getattr(self, 'do_' + step['do'])(step, pace)

    def plan_step(self, loop_cap: int, spend_cap: float) -> dict[str, Any]:
        """Next step before any model turn: stop, clock stop, interview, or a model turn."""
        ep = self.ep
        if ep.finished or ep.store.blocked or ep.steps_left() <= 0 or self.turns >= loop_cap or not ep.llm.cost < spend_cap:
            return {'do': 'halt'}
        if ep.past_stop():
            return {'do': 'clock'}
        return {'do': 'ask'} if talk_due(ep) else {'do': 'think'}

    def think_once(self) -> dict[str, Any] | None:
        self.trim()
        self.turns += 1
        return self.ep.llm.think(self.chat, [LIGHT, SPARE], 3000, self.offered())

    def plan_reply(self, message: dict[str, Any] | None) -> dict[str, Any]:
        """Step from the model's reply: a miss, an idle turn, the shopper first, or the calls."""
        if message is None:
            return {'do': 'miss'}
        ep = self.ep
        calls, content = self.parse_calls(message), message.get('content')
        text = content if isinstance(content, str) else dumps(content) if content else ''
        if not calls:
            return {'do': 'idle', 'text': text}
        committing = {ep.roles.get(role) for role in ('cart', 'buy', 'refuse')} - {None}
        asks_first = (not ep.talk.started and ep.roles.get('ask') and ep.free_messages() > 0 and ep.steps_left() > 8
                      and any(c['function']['name'] in committing for c in calls))
        return {'do': 'ask_first' if asks_first else 'calls', 'text': text, 'calls': calls}

    def do_halt(self, step: dict[str, Any], pace: dict[str, Any]) -> bool:
        return False

    def do_clock(self, step: dict[str, Any], pace: dict[str, Any]) -> bool:
        self.ep.log('clock_stop', step=self.ep.step)
        return False

    def do_ask(self, step: dict[str, Any], pace: dict[str, Any]) -> bool:
        summary = talk_run(self.ep)
        if summary:
            self.say(summary)
        return True

    def do_miss(self, step: dict[str, Any], pace: dict[str, Any]) -> bool:
        self.misses += 1
        llm = self.ep.llm
        return not (llm.out_of_credit or llm.calls >= ALL_CALLS or self.misses > 2)

    def say(self, text: str) -> None:
        self.chat.append({'role': 'user', 'content': text + '\n' + self.status()})

    def do_idle(self, step: dict[str, Any], pace: dict[str, Any]) -> bool:
        ep = self.ep
        pace['idle'] += 1
        self.chat.append({'role': 'assistant', 'content': step['text'] or '(no call)'})
        self.say('Call one of the tools to continue.')
        if pace['idle'] < 3:
            return True
        names = [t['function']['name'] for t in self.offered()]
        whole_set = [t for t in ep.tools if t['function']['name'] != ep.roles.get('ask')]
        if not pace['nudged'] and self.read_room() and len(names) < len(whole_set):
            # turns without a call on the last steps' narrowed tools while an item has no cart line (Arena b27-b32: 8
            # of 17 unbought endings on order tasks): once, the open tools are named
            pace.update(nudged=True, idle=0)
            self.chat.pop()
            self.say(f"[Desk] Your last turns made no tool call. Open to you now: {', '.join(names)}. An item still "
                     'has no cart line: use them to find and read a fitting listing, cart it and check out; end without a purchase only '
                     'if your search showed that nothing fits.')
            ep.log('idle_tail_nudge', step=ep.step, tools=names)
            return True
        # turns without a call this early (no page read, under half the steps used) come from a model that wrongly
        # believes the tools are gone: one more round with the tool names before the ending
        late = pace['retried'] or ep.mem.pages or ep.step * 2 >= ep.max_steps or ep.steps_left() <= 2
        if late:
            return False
        pace.update(retried=True, idle=0)
        listed = ', '.join(t['function']['name'] for t in self.offered())
        self.chat.pop()
        self.say(f'[Desk] Your last turns made no tool call. The store tools work and are available now: {listed}. '
                 'Use one of them this turn; no listing has been read in full so far, so open listings in every category.')
        ep.log('idle_retry', step=ep.step)
        return True

    def do_ask_first(self, step: dict[str, Any], pace: dict[str, Any]) -> bool:
        pace['idle'] = 0
        calls = step['calls']
        self.chat.append({'role': 'assistant', 'content': step['text'] or None, 'tool_calls': calls})
        self.chat += [{'role': 'tool', 'tool_call_id': c['id'], 'content': 'Not run (no step used): the shopper is asked first.'} for c in calls]
        self.say(talk_run(self.ep) or 'The interviewer had nothing to ask.')
        return True

    def do_calls(self, step: dict[str, Any], pace: dict[str, Any]) -> bool:
        pace['idle'] = 0
        text, calls = step['text'], step['calls']
        self.chat.append({'role': 'assistant', 'content': text or None, 'tool_calls': calls})
        self.ep.log('turn', text=text[:300], calls=[c['function'] for c in calls])
        replies = self.run_calls(calls)
        tail = len(replies)
        for n, (cid, said) in enumerate(replies, 1):
            self.chat.append({'role': 'tool', 'tool_call_id': cid, 'content': said + '\n' + self.status() if n == tail else said})
        return True

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        """Answer every tool call of one turn."""
        ep = self.ep
        batch = self.join_opens(calls)
        before = set(ep.opened)
        replies = [(call['id'], self.handle(call, len(batch))) for call in batch]
        early = chains_early(ep, ep.opened - before) if replies else ''
        if early:
            replies[-1] = (replies[-1][0], replies[-1][1] + '\n' + early)
        return replies

    def handle(self, call: dict[str, Any], batch: int) -> str:
        ep = self.ep
        if call.get('joined'):
            return 'See the side-by-side result above: these listings were opened together by Desk in a single step.'
        job = screen_screen(ep, call.get('function') or {}, batch)
        if isinstance(job, str):
            return job
        before = set(ep.mem.rows)
        self.last_obs = None
        said = close_policy(ep, job) or self.perform(job)
        if job.searching:
            screen_note_page(ep, job, before)
        return said

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge single opens into one side view."""
        ep = self.ep
        opener, joiner = ep.roles.get('open'), ep.roles.get('compare')
        target, many = ep.id_param(opener), ep.side_param()
        if opener == joiner or not (opener and joiner and target and many) or ep.side_view_bare():
            return calls
        picked: list[tuple[int, str]] = []
        for at, call in enumerate(calls):
            fn = call.get('function') or {}
            text = fn.get('arguments')
            if not isinstance(text, str) or fn.get('name') != opener:
                continue
            args = parse_object(text)
            if not args or set(args) != {target}:
                continue
            ident = str(args[target] or '').strip()
            if ident in ep.mem.rows and ident not in ep.gone:
                picked.append((at, ident))
        picked = picked[:ep.side_room()]
        if len(picked) < 2:
            return calls
        ids = [ident for _, ident in picked]
        out = list(calls)
        (lead, _), *others = picked
        out[lead] = {'id': calls[lead]['id'], 'type': 'function', 'function': {'name': joiner, 'arguments': dumps({many: ids})}}
        for at, _ in others:
            out[at] = {'id': calls[at]['id'], 'joined': True}
        ep.log('joined', ids=list(ids))
        return out

    def perform(self, job: Call) -> str:
        """Run the call and write its answer."""
        ep = self.ep
        reply = ep.store.call(job.name, job.args)
        if reply is None:
            return 'The call failed (network or server). You may try again.'
        seen = self.last_obs = reply.get('observation')
        self.side_view_seen(job, seen)
        ep.recent = (ep.recent + [job.mark()])[-3:]
        listed_values = {key: value if isinstance(value, list) else [value] for key, value in job.args.items()}
        ep.id_args.update((job.name, key) for key, values in listed_values.items()
                          if values and all(isinstance(v, str) and v in ep.mem.rows for v in values))
        mem_absorb(ep, seen)
        paged = self.learn_view(job, seen)
        self.learn_cart(job, seen)
        closed = bool(reply.get('finished'))
        if ep.mode_ok('refuse', job.name, job.args) and not closed and not ep.finished:
            ep.end_refusals += 1
        if ep.mode_ok('buy', job.name, job.args):
            ep.checkouts += 1
            if not closed:
                ep.refused_at = ep.step
        return self.answer(job, reply, paged)

    def side_view_seen(self, job: Call, seen: Any) -> None:
        """Learn the side view room from its result."""
        ep = self.ep
        ids = job.listed()
        if not ids or len(ids) < 2:
            return
        if any(str(i) in dumps(seen) for i in ids):
            ep.side_ok = max(ep.side_ok, len(ids))
            return
        ep.side_bad = min(ep.side_bad or 999, len(ids))
        job.extra += f'\n[Desk] The side-by-side view showed nothing for {len(ids)} listings: send at most {ep.side_room()} per call.'

    def learn_view(self, job: Call, seen: Any) -> str:
        """Learn tool roles and search facts from a call."""
        ep = self.ep
        roles, scout, tool = ep.roles, ep.scout, job.name
        if tool == roles.get('compare'):
            rows = mem_listings_in(ep, seen)
            if rows:
                ep.fact_tools[tool] = ep.fact_tools.get(tool, False) or any(fact_block(r) for _, r in rows)
        if tool in ep.reading_tools() - {roles.get('search')}:
            mem_note_keys(ep, tool, seen)
        job.searching = not (job.ending or tool in ep.role_tools('search'))
        extra = ''
        if job.searching:
            extra = scout_learn_paging(ep, tool, job.args, seen)
            scout_note_next(ep, tool, job.args, seen)
            extra += scout_auto_pages(ep, tool, job.args, seen)
        if job.searching:
            scout_learn_cap(ep, tool, job.args, seen)
            if ep.brief.get('quiet_decline') is False:
                scout.search_log = list(scout.search_log)[-11:] + [(tool, dict(job.args), seen)]
        fixed = {roles.get(r) for r in ('cart', 'buy', 'refuse', 'ask')}
        found = None
        if isinstance(seen, dict) and ep.id_param(tool) and tool not in fixed:
            found = mem_identify(ep, seen)
        if found:
            ep.fact_tools[tool] = shows = ep.fact_tools.get(tool, False) or bool(fact_block(seen))
            # a single-listing view without attributes: another id tool may show them; this one still shows the offer
            if not shows and tool == roles.get('open'):
                skip = fixed | {roles.get('search')}
                others = (t for t in ep.schemas if t != tool and ep.id_param(t) and ep.fact_tools.get(t) is not False and t not in skip)
                roles['open'] = next(others, None) or roles['open']
            if not (shows or tool == roles.get('open') or roles.get('offer')):
                roles['offer'] = tool
            if tool == ep.offer_tool():
                ep.offer_seen[found] = ep.step
            if tool == ep.offer_view():
                ep.page_seen.add(found)
        if tool != roles.get('search') or not mem_listings_in(ep, seen):
            return extra
        extra += scout_learn_rule(ep, tool, job.args, seen)
        scout.searched = list(scout.searched)[-7:] + [dict(job.args)]
        extra += scout_first_reads(ep, [seen] + scout.extra_pages)
        return extra

    def learn_cart(self, job: Call, seen: Any) -> None:
        """Cart bookkeeping after a call."""
        ep, basket = self.ep, self.ep.basket
        if job.name == ep.roles.get('cart_view'):
            basket.view, basket.viewed_at = seen, ep.step
        if job.name != ep.roles.get('cart'):
            return
        basket_record(ep, job.name, job.args)
        basket.reply = dumps(seen)
        ep.recent = []
        values = list(job.args.values())
        counts = [whole(v) for v in values if not isinstance(v, str) and whole(v) is not None]
        named = [v for v in values if isinstance(v, str) and v in ep.mem.rows] if counts else []
        if counts and counts[0] > 0:
            basket.dropped.difference_update(named)
        else:
            # a zero count the store's reply does not show as kept takes the line out
            basket.dropped.update(i for i in named if not mentions_row(seen, i))
        lines = [i for i in basket_ids(ep) if not (ep.is_gone(i) or i in basket.dropped)]
        if len(lines) >= len(basket.built):
            basket.built, basket.built_needs = lines, {need for _, need in ep.live_needs()}

    def answer(self, job: Call, reply: dict[str, Any], paged: str) -> str:
        """Result text with the notes below it."""
        ep, scout = self.ep, self.ep.scout
        seen = reply.get('observation')
        plain = ep.render(seen) + paged + job.extra
        if job.searching:
            plain += scout_cheapest_note(ep, [seen] + scout.extra_pages) + scout_ignored_cap(ep, job.name, job.args, seen)
        notes = [judge_evidence(ep, seen), self.zero_count_hint(job, seen), news_store_note(ep, seen), news_after_refusal(ep, job, reply),
                 news_swap_hint(ep), news_flush_queue(ep), scout_cheaper_search(ep)]
        if job.name in ep.reading_tools():
            ep.cache[job.mark()] = (ep.version, ep.step, plain)
        return plain + ''.join('\n' + note for note in notes if note) + (' [episode finished]' if reply.get('finished') else '')

    def zero_count_hint(self, job: Call, seen: Any) -> str:
        ep = self.ep
        if job.name != ep.roles.get('cart') or not any(whole(v) == 0 for v in job.args.values() if not isinstance(v, str)):
            return ''
        if not any(isinstance(v, str) and v in ep.mem.rows and mentions_row(seen, v) for v in job.args.values()):
            return ''
        return '[Desk] That line stayed in the cart with a count of 0: the tool adds the count to the line, and a negative count removes units.'

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Episode(problem_data)
    except Exception as exc:
        return [{'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]}]
    try:
        return episode.loop.run()
    except Exception as exc:
        episode.trace.append({'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]})
        try:
            if not episode.finished:
                close_finish(episode)
        except Exception as again:
            episode.trace.append({'kind': 'crash', 'error': f'{type(again).__name__}: {again}'[:400]})
        return episode.trace
__all__ = ['agent_main']
