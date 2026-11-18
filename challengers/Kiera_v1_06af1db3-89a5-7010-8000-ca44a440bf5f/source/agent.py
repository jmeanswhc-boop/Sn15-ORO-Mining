"""NVIDIA: buys for one shopper inside an ORO store world.

Lineage: forked from Cairn v2 (06aef3d7, the public race-17 winner and /top agent), itself built on Halo v3. The
fork keeps Cairn's strategy (planned interview, ledger of requirements, chain resolution, fitting-set hunts,
cart audit and gated finish) under a rewritten code base, and adds:
- S3L: with two messages or fewer, every question names three planned properties from the first one on (Cairn
  opened a one-item interview with a single property to probe the shopper; race 18 live, a hidden property in
  slot 1 / 2 / 3 of a question was revealed 96% / 83% / 50% of the time, so the probe cost half the interview).
- P1: once the store or the shopper changes the task, the best option is the cheapest one that fits. A sold-out line's
  replacements are ranked cheapest first (unless the pick is a most / least measure); after a change note the ledger
  asks for the lowest-priced of the fitting listings already read; and notes that ride on the shopper's reply go to
  the change reader instead of being read as the answer. Race 18 live: not_the_best_option was 9% of episodes and
  clustered after sold-out lines and dropped requirements, and in 14 of 14 one-line differences between a failed and
  a passing cart of the same task the passing line was the cheaper one; a dropped requirement announced inside the
  reply left the Cairn family at 0.55 pass (27 of 71 not the best option).
- P2: once a note drops a requirement, takes a listing away or changes the budget, an "any" pick becomes the
  cheapest pick (a price move alone keeps the pick: after a price rise the line mostly stays the best one), which turns on Cairn's own cheaper hunt (value searches, paged cheapest rows, wider hold).
  Race 17 and 18 live: in 283 of 283 failed carts that differ from a passing cart by one line the passing line is the
  cheaper one, and in 101 of 103 such race-18 failures the cheaper listing never appeared in what the agent saw.
- L6: on a one-item request with two messages or fewer the last question names six labels (three, then six). Race 18
  live, last message: late slots of a long list still reveal about half the time (0.55 / 0.40 / 0.57 / 0.50 at slots
  4-7), where a first message's late slots reveal 0.22; one-item reveal: Mahanadi (1-2 then 5) 0.86, K2 (3 and 3) 0.76.

- E1: on a several-item request, an item that no read listing describes yet still gets planned labels (one short model
  call on its name), so the interview never spends every message on the other items.

Entry point: agent_main(input) -> list of trace entries."""
from __future__ import annotations
import json
import random
import re
import time
from typing import Any
from src.agent.proxy_client import ProxyClient
LIGHTN = 'anthropic/claude-haiku-5.5'
HEAVYN = 'anthropic/claude-sonnet-5.5'
SPAREN = 'openai/gpt-6-luna'
CYCLE_INVOCATIONS = 42
ALL_INVOCATIONS = 60
SPEND_MILD = 0.055
SPEND_HARDN = 0.078
HEAVY_SPACE = 0.02
WAITSN = (4, 10, 20, 35)
WAIT_SUM = 45
DIGEST_BY = 120.0
DIGEST_MARGIN = 10.0
OUTAGE_BYN = 150.0
DISPLAY_CHARS = 450
CHASE_MOVES = 24
WORLD_MARGIN = 0.25
OPEN_MARGIN = 0.5
GRID_FIELDS = 18
LOWER_PRICED_DISPLAYED = 4
QUIET_TERM = 4
OPEN_LATERN = 2
SHEET_ATTR_CHARS = 700
SHEET_REMAINDER_CHARS = 300
NONE_NAME = ('none', 'null', 'n/a', 'unknown')
QUERY_TURNS = 2
NOTE_SLOT = '(?:^|_)(?:texts?|body|messages?|notes?|contents?|questions?|msg|say)(?:_|$)'
NOT_NOTE = '(?:^|_)(?:id|ids|type|format|code|lang|language|kind)$'
PATH_SPACE = 15
MATCH_SHEETS = 2
CRASH_TURNSN = 3
OPEN_MIN_MOVES = 20
OPEN_LINES = 5
OPEN_WARES = 2
OPEN_PORTION = 5
OPEN_EACHN = 6
OPEN_NOTN = '\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or|same|listing)\\b'
BRIEF_RETAIN = 8
SET_DISPLAYED = 2
STOCK_OUTN = '^(?:no|none|false|0|n)$|out[ _-]?of[ _-]?stock|sold[ _-]?out|unavailable|not[ _-]?available'
CHASE_ENTIRE = 49
CHASE_GRACE = 25.0
CHASE_VAL_CHARS = 30
QUERY_TERMS = '\\b(?:one|each|single|first|time|many|much|several|list|once|separate\\w*|split|short\\w*|questions?|ask\\w*|turn|lot|too|all)\\b'
CHASE_ROUNDS = 2
CHASE_RETAIN = 10
CHASE_INACTIVE = 4
CHASE_SHEET_FRESH = 5
CHASE_MATCH_TERMS = 4
CHASE_MATCH_OPENS = 4
CHASE_COMMON = 0.6
CHASE_BY = 200.0
HEADING_CHARS = 200
MIX_TAGS = 3
TEXT_CHARSN = 3000
OBS_BOUND = 15000
HISTORY_BOUND = 150000
BARRIER_ROUNDS = 3
AUTO_SHEETS = 6
LOWER_PRICED_OPENS = 3
TASK_SECONDSN = 300.0
MOMENT_MARGIN = 20.0
INVOKE_BASE = 6.0
QUERY_BY = 165.0
INQUIRY_BY = 210.0
WIND_DOWNN = 200.0
FINAL_TURN = 255.0
EMPTY_SPACE = 12.0
TARDY_BY = 190.0
TARDY_MOVES = 8
SCARCE_MESSAGESN = 2
LIST_MAGNITUDE = 12
LIST_MIXEDN = 8
LIST_MOVES = (1, 3)
LIST_MOVES_MIXED = (3, 3)
CEILING_ARG = '(?:^|_)(?:cap|max|maximum|ceiling|under|up_?to)(?:_|$)'
CEILING_NOT = 'result|row|page|count|size|num|item|step'
CEILING_PRICED = 'price|cost|budget|spend|money|amount'
CEILING_SMALL = '(?:^|_)(?:min|minimum|low|lowest|least|from|start|offset|floor)(?:_|$)'
EARLY_TRAILSN = 3
CEILING_CHANCE = 0.05
STROLL_SPACE = 10
STROLL_INVOCATIONS = 4
AISLE_ARG = 'section|department|categor|label|shelf|aisle|(?<!age)(?:^|_)group(?:_name|_label)?$'
LIST_MANNER = 'field|filter|facet|brows|categor|list|structured'
AMOUNTN = '\\d[\\d.,]*\\s*[a-z"\']{1,12}'
BASE = '\\b(?:at least|no (?:less|fewer) than|min(?:imum)?)\\s+(' + AMOUNTN + ')\\s*$'
PHRASE_TERMS = '(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword'
_lm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_lm_brief: dict[int, ProxyClient] = {}
_world = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
GUIDEN = '\n'.join(("You are NVIDIA, a careful buyer working for one shopper in an online store. You can act only by calling the store's tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.", '', 'HOW THE STORE WORKS', "- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool's description and schema and use the exact argument names and allowed values.", "- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun); a store's search can read attribute values and shop names as well as titles and may want every word, so a few required values plus the noun can list nearly the whole fitting set, and a model or part code may find nothing (one try tells). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.", "- Search rows are summaries. Stores split a listing's record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once NVIDIA knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.", "- Questions to the shopper are handled for you by NVIDIA's interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.", "- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. NVIDIA carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store's tools (product questions go through NVIDIA's interviewer).", '', 'WHAT COUNTS AS A FIT', "- The kind of product is decided by the listing's category label: only the label that matches the requested item as a whole counts, never a neighbouring one.", '- A requirement on a property is met when the listing\'s attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value\'s words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.', '- An attribute whose key names the requirement\'s property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A value under a key that adds the item\'s own name to the property (a bag\'s "bag shape") counts even when the bare key ("shape") says something else. A key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing about the requirement. NVIDIA adds [evidence] lines after opened listings with the keys whose names match a requirement.', '- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.', "- When the category label itself states a requirement's value (a season or audience in the label), that requirement is met by the category. A listing under the requested label is that kind of product even when its title or description names something else, so judge it on the requirements, price and stock.", '- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value; when the requirement asks for what goes with that listing\'s value (its partner, its counterpart) and the page names what its value pairs with, the requirement is that other value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing\'s value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.', '- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.', "- When one requested item must fit another (an accessory for a device), choose the two together from the listings' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.", '- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.', '- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.', '', 'CHOOSING', '- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and NVIDIA reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.', '- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.', '- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.', '- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.', '- A request for several different items needs one cart line per item, each at its requested quantity.', '', 'CHANGES WHILE YOU WORK', "- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool's result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.", "- Price and stock can change silently: NVIDIA re-reads the cart's lines and looks at the cart when you call the checkout, so do not spend a turn on either.", '', 'FINISHING', '- Put exactly the chosen lines in the cart (remove anything else), then check out with a call on its own.', "- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request's wording for that ending, and when a remark is needed, keep it short and name no listing.", '- Every reply must call at least one tool. Keep your text short: what you know, then the next move.'))
DIGEST = '\n'.join(("You prepare a shopping task for an agent. Read the shopper's request and the store's tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:", '', '"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request\'s words, "shelf": the store\'s category label for it when a tool lists labels (copy it exactly), or the category the request itself names for it ("a coat in rainwear" gives "rainwear"), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request\'s own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property\'s value or limit outright; and never turn words that are part of what the product is called (a season or an audience named with the product) into musts: those belong to "name". A use or occasion the request makes a property of the product itself ("blazers for business wear", "a dress made for parties", "a bag, and it is for a wedding", "the occasion has to be party") IS a must on that property ("occasion: business"), unless the store\'s category label already names it; a particular person\'s event, a person or a setting the product is bought for ("to wear at his wedding", "for a colleague"), and how it should sit, feel or look in use, stay context. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.', '"budget": the most the shopper will pay in total as a number in the request\'s own unit, or null.', '"budget_unit": that unit in words (for example "cents"), or null.', '"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.', '"in_stock_only": true unless the request clearly accepts items that are not available.', '"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.', '"pick_measure": what is maximised or minimised for "most"/"least", else "".', '"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.', '"steps": how many steps, moves or tool calls the request allows (integer), or null.', '"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].', '"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.', '"wishes": a list of things the shopper would merely like.', '"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).', '"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper\'s approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.', '"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.', '"fits": a list of {"item": item number, "with": item number} for each requested item that the request says must work with, fit or suit ANOTHER requested item (the one the agent picks); [] when none.', '"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing\'s attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing\'s current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).', '"search_words": {"tool": the tool that searches listings by free words (when one tool lists a category and another takes words, the one that takes words), "field": its parameter that takes the words, "args": the fixed arguments that make it search by words (for a tool with a mode argument, that mode), else {}}.', '"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.', '"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.', ''))
SCHEME = '\n'.join(('You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property; a property the shopper has no view on gets "don\'t mind", and one the shopper does not think about at all gets a puzzled reply, so every property named spends part of a message.', '', "You get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item's category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.", '', 'Choose the properties in this order:', '0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.', "1. Hinted properties. The request's story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product's looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.", "Name a hinted property by a listed key of that item when one fits the hint (a hint about who will wear it: the key for gender or audience; about its looks: the listed key for colour, pattern or finish that most of its listings carry); a property no listing of the item carries comes after the item's three most carried keys.", "2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item's listings; once five or more listings were read, avoid keys seen on only one or two of them.", 'Skip a property the request already settles for that item (a requirement names that property; a key whose values only repeat a required value, such as a pattern key showing the required decoration, is NOT settled), a second name for a property already chosen, a key whose values each list several purchase options at once, identifiers, model names, warranty and other logistics. Pack sizes, counts and quantities per pack are properties like any other, and a key marked [not the stated ...] is another property than that requirement, whatever words its name shares with it. Rank every item on its own: its hinted properties first, then its listed keys.', '"label" is the property\'s plain name: for a listed key, all of the key\'s words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property\'s name, no example values, no brackets, no second property.', 'Choose up to COUNT for each item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story\'s words it comes from, or an empty string"}]}, each item\'s picks in rank order'))
HEARN = '\n'.join(('A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:', '{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants them asked one at a time; not answers that only say they did not follow it or want it in simpler words], "budget": new total limit as a number or null}', '- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3"); the value is only what a listing could state, from the wanted value or its limit word to its end, without the framing the answer puts around it ("oval" for "matching oval", "settled, with oval" or "oval and nothing else", "at least 30 cm" for "equal to at least 30 cm").', '- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop\'s product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer\'s own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.', '- "free": the property of each question the shopper said they do not mind about.', '- Folksy or roundabout wording that names a value still states that value. Wording that refuses to bargain over a value (it will not be traded, given up or changed) states that value as wanted; write "not X" only when the answer rejects X itself.'))
MEMO = '\n'.join(('While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:', '{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request\'s unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}', 'A replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note.'))
VERIFY = '\n'.join(('You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.', 'Reading rules: a requirement is met when the listing\'s attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value\'s words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails, though a value under a key that adds the item\'s own name to the property (a bag\'s "bag shape") counts even when the bare key ("shape") says something else; a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing\'s value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.', "When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.", 'Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses.'))
READINGN = '; '.join(('the listing\'s attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value\'s words counts, such as "deep teal" for teal, limits include the edge)', 'a value under a key that adds the item\'s own name to the property (a bag\'s "bag shape") counts even when the bare key ("shape") says something else', "a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing", 'when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing)', 'a requirement the category label itself states is met by that category', 'the category label must match the item', "a rule of being unlike another listing's value counts as met unless the records show that same value", 'out-of-stock listings and listings that break the budget do not fit.'))
REVIEW = '. '.join(('A shopping agent wants to end without buying because it thinks nothing fits', 'Check the listing records shown against the requirements, with these reading rules: ')) + READINGN + '\n'.join(('', 'Records marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.', 'Return ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}'))
UNMIXN = '. '.join(("A shopper's request was read into firm requirements on a product", 'Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought', "Using the request's wording, say which of the OTHER REQUIREMENTS below only describe that other listing", 'Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}'))
PATH = '\n'.join(("A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing's page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement's chain from the requirement's own description, link by link, through the pages' own words, using the records below (OPENED: the agent has opened that page; ROW: its page not read, seen only as a search result row or by its attributes).", 'Return ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing, "find": "for a link not found: search words (its shop and the values the description gives)"}], "done": true when the last link is an OPENED page that states the property\'s value itself (also when the requirement asks for a value other than or unlike that one: the page\'s value is what it is measured against), else false, "value": the property\'s value as stated on that last page; when the requirement asks for what goes with that value and the page names what its value pairs with, that other value; or ""}]}', 'A record is a link only when it matches the whole description of that link: the shop or brand, the kind of product AND every property value the description gives (for example "a 10 pieces multipack" rules out the same shop\'s listing with 10 capsules and no multipack); when no record matches all of it, that link is not found yet, even if the same shop sells other listings of that kind. Give "find" for such a link: the shop\'s name and the description\'s values, as a few search words.', "List each chain's links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown."))
BARRIER = '\n'.join(("A shopping agent is about to STAGE. The shopper's request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product's properties do not do a step.", 'When a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.', 'Return ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}'))
EXCHANGE = '. '.join(("A listing in a shopping agent's cart has just sold out", 'Find replacements for that cart line among the other listing records shown, with these reading rules: ')) + READINGN + '\n'.join(('', 'Return ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line\'s item, at most 6], "reason": "one sentence"}'))
FORM = '. '.join(('A store tool returned the result below', "Name its parts by the exact keys the result and the tool's parameters use", 'Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing\'s price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}'))
GUARD_FIELDS = 3
GUARD_PORTION = 0.0
GUARD_MARGIN = 0.34
NOT_PROPERTYN = '\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b'
CATCH_ALLN = '^(?!.*\\b(?:types?|styles?|kinds?|shapes?)\\b).*\\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\\w*|accessor\\w*|extras?|misc\\w*)\\b'
WIDEN_EACHN = 6
MATCH_TAG_CODE = '[A-Z][A-Za-z]{0,5}[ -]?(?:\\d{3,6}|\\d{1,5}[xX]{1,3})'
PIECE_TAG_CODE = '[A-Z]{1,5}\\d{0,3}-?\\d{3,7}'
VAL_SPACE = 12
VAL_RETAIN = 8
LOOK_ALIKESN = 2
TIE_SEARCHES = 3
TIE_LINES = 3
TIE_KINS = 12
_g1_SORT_TESTS = {'string': lambda v: isinstance(v, str), 'integer': lambda v: isinstance(v, int) and (not isinstance(v, bool)), 'number': lambda v: isinstance(v, (int, float)) and (not isinstance(v, bool)), 'boolean': lambda v: isinstance(v, bool), 'array': lambda v: isinstance(v, list), 'object': lambda v: isinstance(v, dict), 'null': lambda v: v is None}
_g5_KEPT_STEPS = ' '.join(('Not', 'run', '(no', 'step', 'used):', 'the', 'last', 'steps', 'are', 'kept', 'for', 'checking', 'out', '(with', 'the', "shopper's", 'approval', 'of', 'the', 'basket', 'when', 'it', 'is', 'due)', 'or', 'ending', 'without', 'a', 'purchase.'))
_g5_DEAD_CODES = '. '.join(("Not run (no step used): this store's search found nothing for a model or part code", 'Search by words and values, list the category, and read the codes on the rows.'))
_g5_CLUE_TAIL = '. '.join(('', 'Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out', 'End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again.'))
_g5_LATE_HEAD = '. '.join(('Not run yet (no step used)', 'Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none', ''))
_g5_LATE_TAIL = '; '.join((' End without a purchase only if your search showed that some requested item has no listing that meets its requirements', 'then send the ending call again.'))
_g5_ZERO_NOTE = '\n'.join(('', '[NVIDIA] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'))
_g5_START_NOTE = '\n'.join(('', '', 'Start by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.', '', ''))
_g5_MODE_ASK = '\n'.join(('', 'Return ONE JSON object and nothing else: {"buy_args": the fixed arguments that check out, "refuse_args": the fixed arguments that end without a purchase} (only the arguments that choose the mode).'))
_g6_NEGATORS = frozenset({'not', 'no', 'least', 'most', 'under', 'over', 'unlike', 'different', 'min', 'max', 'minimum', 'maximum'})

def _tag_codes(text_n: Any) -> list[str]:
    out = []
    for c in re.findall('\\b' + MATCH_TAG_CODE + '\\b', str(text_n or '')):
        if re.fullmatch('[A-Z][a-z]+ (?:19|20)\\d\\d', c):
            continue
        out.append(re.sub('[\\s-]+', ' ', c).strip().lower())
    return out

def _tag_code_strike(pattern: str, tag_code: str) -> bool:
    wide = re.sub('[\\s-]+', '', pattern)
    bare = re.sub('[\\s-]+', '', tag_code)
    root = wide.rstrip('x')
    if wide == root or not root[-1:].isdigit():
        return bare == wide
    return len(bare) == len(wide) and bare.startswith(root)

def _match_signs(text_n: Any) -> dict[str, Any]:
    text_n = str(text_n or '')
    own = re.search('[\\[(]\\s*(?:part\\s+)?(' + PIECE_TAG_CODE + ')\\s*[\\])]', text_n, re.I)
    codes = []
    for pat in ('\\bmodel\\s*(?:no\\.?|number|#)?\\s*[:.]?\\s*(?!(?:no|number)\\b)(' + MATCH_TAG_CODE + ')\\b', '\\b(?:also sold as|also known as|sold also as)\\s+([^.;]+)'):
        codes.extend((c for m in re.finditer(pat, text_n, re.I) for c in _tag_codes(m.group(1))))
    works, bars = _g1_fit_lists(text_n)
    also = []
    for m in re.finditer('\\bpart\\s+(' + PIECE_TAG_CODE + ')\\s+also fits\\s*:?\\s*([^.;]+)', text_n, re.I):
        also.append((m.group(1).upper(), _tag_codes(m.group(2))))
    more = re.search('\\b(?:more|details|see|read)\\s+(?:on|at|under|about)?\\s*(?:the\\s+)?part\\s+(' + PIECE_TAG_CODE + ')', text_n, re.I)
    fam, named = _series_signs(text_n)
    signs = {'own': own.group(1).upper() if own else '', 'codes': codes, 'works': works, 'bars': bars, 'also': also}
    signs.update({'more': more.group(1).upper() if more else '', 'fam': fam, 'named': named})
    return signs

def _series_signs(text_n: str) -> tuple[list[str], list[str]]:
    fam = sorted({m.upper() for m in re.findall('\\b([A-Za-z]{1,5}\\d{1,3})-\\d{2,7}\\b', text_n)})
    rest = re.sub('\\b[A-Za-z]{1,5}\\d{1,3}-\\d{2,7}\\b', ' ', text_n)
    rest = re.sub('(?i)\\b(?:except|excluding|but not|not for|not compatible with|does not (?:fit|take))\\b[^.;]*', ' ', rest)
    named = set()
    for m in re.findall('\\b[A-Za-z]{1,5}\\d{1,3}\\b', rest):
        named.add(m.upper())
    return (fam, sorted(named))
COUPLE_SIGNS = (': ', ' = ', '=', ' -> ', ' - ')
SLOT_SIGNS = (' | ', '\n', ' || ', ' ; ', '; ', '\t', ' // ', ' · ', ' • ', '|', ';', ', ')
SLOT_NAME = "[A-Za-z][A-Za-z0-9_ ./&'()-]{0,38}"
LIST_BULLETN = '^\\s*(?:[-*•·]|\\d{1,3}[.)])\\s+'
ENTRY_SLOTS = 3
MEMO_SLOTS = 5
SLOT_PORTION = 0.15
COUPLE_PORTION = 0.7
CHUNK_NAME = 60
CHUNK_VAL = 40
TAG_TEXTN = 20000
COUPLE_KEY = 'name|key|label|attr|prop|field|spec|title|^k$'
ID_SLOT = ' '.join(('(?:^|[_', '-])(?:id|idx|sku|pid|ref|asin|ident\\w*)(?:$|[_', '-])|^id(?-i:[A-Z0-9])|(?-i:Ident)|(?<![a-z])(?!(?:paid|valid|grid|solid|rigid|fluid|liquid|hybrid|rapid|void|acid|humid|android|said|laid)$)[a-z]*id$'))
FAIL_KEYN = 'error|fail|(?<!de)fault|trouble(?!shoot)|denied|refus|invalid'

def _failed_n(obs: Any) -> bool:
    if isinstance(obs, dict) and 0 < len(obs) <= 3:
        for key, val in obs.items():
            if val and re.search(FAIL_KEYN, str(key), re.I):
                return True
        if len(obs) != 1:
            return False
        sole = next(iter(obs.values()))
        return isinstance(sole, str) and 0 < len(sole) <= 40 and (' ' not in sole)
    return False

def _bare(text_n: str) -> Any:
    plain = text_n.strip()
    small = plain.lower()
    if small in ('true', 'false'):
        return small == 'true'
    if small in ('none', 'null', ''):
        return None
    money = re.fullmatch('[$€£]?\\s?((?:\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.\\d+)?)', plain)
    if not money or re.match('0\\d', money.group(1)):
        return plain
    digits = money.group(1)
    num = float(digits.replace(',', ''))
    if num.is_integer() and '.' not in digits:
        return int(num)
    return num

def _chunk_val(text_n: str) -> Any:
    found = _bare(text_n)
    if isinstance(found, bool) or (found is None and text_n.strip()):
        return text_n.strip()
    return found

def _couple(text_n: str, sign: str) -> tuple[str, str] | None:
    head, found, rest = text_n.partition(sign)
    tip = sign.strip()
    if not found and text_n.endswith(tip):
        head, found, rest = (text_n[:-len(tip)], sign, '')
    if not found:
        return None
    name = re.sub(LIST_BULLETN, '', head).strip()
    if re.fullmatch(SLOT_NAME, name) and len(name.split()) <= 5:
        return (name, rest.strip())
    return None

def _split_slots(text_n: str, amid: str, sign: str) -> tuple[list[tuple[str, str]], int]:
    pieces = [p for p in text_n.split(amid) if p.strip()]
    if pieces and (not _couple(pieces[0].strip(), sign)):
        return ([], 0)
    slots = []
    called = 0
    heading = None
    for piece in pieces:
        plain = piece.strip()
        if heading is not None and '\n' == amid and (re.match(LIST_BULLETN, piece) or piece[:1] in ' \t'):
            name, val = slots[heading]
            slots[heading] = (name, (val + '\n' if val else '') + plain)
            continue
        heading = None
        if '\n' == amid and plain.endswith(':') and re.fullmatch(SLOT_NAME, plain[:-1].strip()):
            heading = len(slots)
            slots.append((plain[:-1].strip(), ''))
        else:
            couple = _couple(plain, sign)
            if not couple:
                if slots:
                    slots[-1] = (slots[-1][0], slots[-1][1] + amid + plain)
                continue
            slots.append(couple)
        called += 1
    return (slots, called)

def _attr_chunk(text_n: str) -> dict[str, Any] | None:
    plain = text_n.strip()
    if len(plain) >= 5:
        if plain[0] in '{[' and plain[-1] in '}]':
            return _g1_braced(plain)
        if '=' in plain:
            return _g1_assigns(plain)
        return _g1_listed(plain)
    return None

def _entry_text(text_n: str) -> dict[str, Any] | None:
    for amid in SLOT_SIGNS:
        if amid in text_n:
            best = _g1_best_split(text_n, amid)
            if best is not None:
                break
    else:
        return None
    entry = {}
    for name, val in best[1]:
        if name in entry:
            continue
        inner = _attr_chunk(val)
        if inner:
            entry[name] = inner
        elif re.search(ID_SLOT, name, re.I):
            entry[name] = val.strip()
        else:
            entry[name] = _bare(val)
    return entry

def _grid_text(text_n: str) -> list[dict[str, Any]] | None:
    rows = []
    for row in text_n.splitlines():
        if row.strip() and (not re.fullmatch('[\\s|:+=-]+', row)):
            rows.append(row.strip().strip('|').strip())
    if len(rows) >= 2:
        for amid in ('|', '\t', ';', ','):
            boxes = [[c.strip() for c in row.split(amid)] for row in rows]
            head = boxes[0]
            width = len(head)
            if width < 3 or any((len(line) != width for line in boxes)):
                continue
            if not all((re.fullmatch(SLOT_NAME, c) for c in head)) or any((re.fullmatch('[\\d.,$]+', c) for c in head)):
                continue
            return [dict(zip(head, map(_bare, line))) for line in boxes[1:]]
    return None

def _entries_text(text_n: str) -> Any:
    plain = text_n.strip()
    if len(plain) < 12:
        return None
    if plain[0] in '{[':
        try:
            return _recover(json.loads(plain))
        except (ValueError, RecursionError):
            pass
    stacked = _g1_stacked(plain)
    if stacked is not None:
        return stacked
    found = _entry_text(plain)
    if found is None and '\n' in plain:
        found = _grid_text(plain)
    if found is not None:
        return found
    if len(plain) <= TAG_TEXTN and re.search('<[a-zA-Z/][^<>]{0,40}>', plain):
        bare = re.sub('</?(?:b|i|em|strong|span|a|u|small)\\b[^<>]{0,40}>', '', plain, flags=re.I)
        bare = re.sub('</t[dh]>\\s*<t[dh][^<>]{0,40}>', ': ', bare, flags=re.I)
        bare = re.sub('\\s*(?:<[^<>]{1,40}>\\s*)+', '\n', bare).strip()
        if plain != bare and len(bare) >= 12:
            return _entries_text(bare)
    if '\n' in plain:
        spaced = '\n'.join((re.sub('(?:[ ]{2,}|\t)', ': ', row.strip(), count=1) for row in plain.splitlines() if row.strip()))
        if plain != spaced:
            return _entry_text(spaced)
    return None

def _chunk_text(text_n: str) -> Any:
    if len(text_n) >= 5 and '=' in text_n and re.search('; | \\| |\\n', text_n):
        pieces = [p.strip() for p in re.split(';\\s+|;$|\\s\\|\\s|\\n', text_n.strip()) if p.strip()]
        if len(pieces) >= 2 and all((re.fullmatch('[^=:]{1,40}=[^=]{0,200}', p) for p in pieces)):
            return _attr_chunk(text_n) or text_n
    return text_n

def _couples_chunk(lines: list[Any], in_entry: bool) -> dict[str, Any] | None:
    if len(lines) < 2:
        return None
    first = lines[0]
    if isinstance(first, dict):
        return _g1_dict_pairs(lines, first)
    if isinstance(first, (list, tuple)):
        return _g1_seq_pairs(lines)
    if in_entry and all((isinstance(row, str) for row in lines)):
        return _g1_text_pairs(lines)
    return None

def _column_lines(leaf: dict[str, Any]) -> dict[str, Any]:
    heads = [k for k in leaf if _g1_is_header(leaf[k])]
    for head in heads:
        names = leaf[head]
        width = len(names)
        for key, lines in leaf.items():
            if key == head or not isinstance(lines, list) or (not lines):
                continue
            if all((isinstance(row, (list, tuple)) and len(row) == width for row in lines)):
                out = dict(leaf)
                del out[head]
                out[key] = [dict(zip(names, line)) for line in lines]
                return out
    return leaf

def _id_strings(entry: dict[str, Any]) -> dict[str, Any]:
    for v in entry.values():
        if isinstance(v, str) and ' ' in v:
            break
    else:
        return entry
    out = {}
    for k, v in entry.items():
        if isinstance(v, int) and (not isinstance(v, bool)) and re.search(ID_SLOT, k, re.I) and (abs(v) >= 100):
            v = str(v)
        out[k] = v
    return out

def _keyed_lines(leaf: dict[str, Any]) -> list[dict[str, Any]] | None:
    lines = list(leaf.values())
    if len(lines) < 2 or not all((isinstance(row, dict) and len(row) >= ENTRY_SLOTS for row in lines)):
        return None
    if any((not re.search('\\d', str(k)) or ' ' in str(k) for k in leaf)):
        return None
    if len(set(lines[0]).intersection(*lines[1:])) < ENTRY_SLOTS:
        return None
    out = []
    for k, row in leaf.items():
        if any((re.search(ID_SLOT, str(x), re.I) for x in row)):
            out.append(row)
        else:
            out.append({'id': str(k), **row})
    return out

def _contains_id(leaf: dict[str, Any]) -> bool:
    for k in leaf:
        if re.search(ID_SLOT, str(k), re.I):
            return True
    return False

def _flat_chunk(leaf: dict[str, Any]) -> dict[str, Any]:
    if all((not isinstance(v, dict) for v in leaf.values())):
        return leaf
    out = {}
    for key, val in leaf.items():
        if not isinstance(val, dict):
            out[key] = val
            continue
        for inner, v in val.items():
            if not isinstance(v, (dict, list)):
                out[f'{key} {inner}'] = v
    return out

def _recover_slot(val: Any, level: int, carded_n: bool, page: bool=True) -> Any:
    found = _recover(val, level + 1, carded_n, page)
    if not isinstance(val, str) or not isinstance(found, dict) or len(found) >= MEMO_SLOTS:
        return found
    if _contains_id(found) or any((_id_akin(v) for v in found.values())):
        return found
    return val

def _recover(val: Any, level: int=0, in_entry: bool=False, page: bool=True) -> Any:
    if level > 7:
        return val
    if isinstance(val, str):
        found = _entries_text(val)
        return val if found is None else found
    if isinstance(val, list):
        return _couples_chunk(val, in_entry and page) or [_recover_slot(v, level, False, page) for v in val]
    if isinstance(val, dict):
        return _g1_recover_map(val, level, page)
    return val
_INITIAL_SPELLING: dict[str, str] = {}

def _join_doubles(kept: str, val: str) -> str:
    amid = '(?<!\\d),|,(?!\\d)'
    has = {piece.strip().lower() for piece in re.split(amid, kept)}
    additional = []
    for piece in re.split(amid, val):
        word = piece.strip()
        low = word.lower()
        if word and low not in has:
            has.add(low)
            additional.append(word)
    if not additional:
        return kept
    joined = (f'{kept}, ' if kept.strip() else '') + ', '.join(additional)
    _INITIAL_SPELLING[joined] = _INITIAL_SPELLING.get(kept, kept)
    return joined

def _glued_n(text_n: str) -> str:
    low = text_n.lower()
    return re.sub('(\\d)\\s+([a-z]{1,3})(?![a-z0-9])', '\\1\\2', low)

def _text_terms(text_n: str) -> set[str]:
    small = text_n.lower()
    glued = _glued_n(small)
    return set(re.findall('[a-z0-9]+', small)) | set(re.findall('[a-z0-9]+', glued))

def _val_pieces(val: str) -> list[str]:
    small = _glued_n(val)
    for pat in ('[a-z0-9]{3,}', '[0-9]+[a-z]|[0-9]*[a-z][a-z0-9]+'):
        found = re.findall(pat, small)
        if found:
            return found
    return re.findall('[0-9]+', small)

def _fold_doubles(leaf: Any, level: int=0) -> Any:
    if level > 5:
        return leaf
    deeper = level + 1
    if isinstance(leaf, list):
        return [_fold_doubles(v, deeper) for v in leaf]
    if not isinstance(leaf, dict):
        return leaf
    out = {}
    seen = {}
    for key, val in leaf.items():
        name = _terms(key)
        first = seen.get(name)
        if first is None:
            seen[name] = key
            out[key] = _fold_doubles(val, deeper)
            continue
        held = out[first]
        if held == val:
            continue
        if isinstance(held, str) and isinstance(val, str) and (not re.search(ID_SLOT, name, re.I)):
            out[first] = _join_doubles(held, val)
        else:
            out[key] = _fold_doubles(val, deeper)
    if len(out) < 2 and len(leaf) >= 2:
        return {k: _fold_doubles(v, deeper) for k, v in leaf.items()}
    return out

def _serial(val: Any) -> str:
    try:
        text = json.dumps(val, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        text = str(val)
    return text

def _trim(val: Any, bound: int, level: int=0) -> Any:
    if level > 7:
        return '...'
    if isinstance(val, str):
        if len(val) > bound:
            return val[:bound] + '...'
        return val
    deeper = level + 1
    if isinstance(val, dict):
        return {k: _trim(v, bound, deeper) for k, v in val.items()}
    if isinstance(val, list):
        return [_trim(v, bound, deeper) for v in val]
    return val

def _json_obj_n(text_n: Any) -> dict[str, Any] | None:
    if not isinstance(text_n, str):
        return None
    text_n = text_n.strip()
    fence = re.match('^```[A-Za-z]*\\s*(.*?)\\s*```$', text_n, re.S)
    if fence:
        text_n = fence.group(1)
    try:
        got = json.loads(text_n)
    except ValueError:
        start, end = (text_n.find('{'), text_n.rfind('}'))
        if not 0 <= start < end:
            return None
        try:
            got = json.loads(text_n[start:end + 1])
        except ValueError:
            return None
    return got if isinstance(got, dict) else None

def _terms(key: str) -> str:
    spaced = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(key))
    return re.sub('[\\s_.\\-]+', ' ', spaced).strip().lower()

def _snaky(key: str) -> str:
    return '_'.join(_terms(key).split(' '))

def _root(text_n: str) -> frozenset[str]:
    return frozenset(map(_g1_unplural, _terms(text_n).split()))

def _bases(text_n: str) -> set[str]:
    out = set()
    for w in _terms(text_n).split():
        if re.search('(?:ch|sh|ss|x|z)es$', w):
            w = w[:-2]
        elif len(w) > 3 and w.endswith('s') and (not w.endswith('ss')):
            w = w[:-1]
        if len(w) >= 3 and w not in ('and', 'for', 'the', 'with'):
            out.add(w)
    return out

def _written(text_n: str) -> frozenset[str]:
    words = []
    for w in _terms(text_n).split():
        if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')):
            w = w[:-1]
        words.append(re.sub('our$', 'or', w))
    return frozenset(words)

def _names_n(trait: frozenset[str], key: frozenset[str]) -> bool:
    if trait and key:
        if key == trait or key >= trait:
            return True
        return len(key) >= 2 and trait >= key
    return False

def _sorts(spec: dict[str, Any]) -> list[str]:
    kind = spec.get('type')
    if isinstance(kind, str):
        return [kind]
    out = []
    for k in kind or []:
        if isinstance(k, str):
            out.append(k)
    return out

def _matches_sort(val: Any, sort2: str) -> bool:
    test = _g1_SORT_TESTS.get(sort2)
    return True if test is None else test(val)

def _form(val: Any, spec: Any, locus: str, issues: list[str]) -> Any:
    if not isinstance(spec, dict):
        return val
    sorts = _sorts(spec)
    if sorts and (not _g1_fits(val, sorts)):
        for sort2 in sorts:
            done, val = _g1_coerce(val, sort2)
            if done:
                break
        if not _g1_fits(val, sorts):
            issues.append(f"{locus} should be {'/'.join(sorts)}")
            return val
    choices2 = spec.get('enum')
    if isinstance(choices2, list) and val not in choices2:
        val = _g1_enum_fix(val, choices2)
        if val not in choices2:
            issues.append(f'{locus}: allowed values are {choices2}')
    if isinstance(val, str):
        if isinstance(spec.get('maxLength'), int) and spec['maxLength'] < len(val):
            val = val[:spec['maxLength']]
        if isinstance(spec.get('minLength'), int) and spec['minLength'] > len(val):
            issues.append(f'{locus} is too short')
    elif isinstance(val, (int, float)) and (not isinstance(val, bool)):
        if isinstance(spec.get('minimum'), (int, float)) and spec['minimum'] > val:
            issues.append(f"{locus} is below {spec['minimum']}")
        if isinstance(spec.get('maximum'), (int, float)) and spec['maximum'] < val:
            issues.append(f"{locus} is above {spec['maximum']}")
    elif isinstance(val, list):
        val = _g1_form_list(val, spec, locus, issues)
    if isinstance(val, dict):
        val = _g1_form_map(val, spec, locus, issues)
    return val

def _wares(val: Any) -> list[Any]:
    if isinstance(val, list):
        return val
    return [val] if isinstance(val, (str, dict)) and val else []

def _tally(val: Any) -> int | None:
    if isinstance(val, int):
        return None if isinstance(val, bool) else val
    if isinstance(val, float):
        return int(val) if val.is_integer() else None
    if isinstance(val, str) and re.fullmatch('\\s*\\d+(\\.0+)?\\s*', val):
        return int(float(val))
    return None

def _id_akin(val: Any, called: bool=False) -> bool:
    if isinstance(val, str):
        core = val.strip()
        if 3 <= len(core) <= 64 and ' ' not in core:
            return called or bool(re.search('\\d', val))
    return False

def _learn_id_n(lines: list[dict[str, Any]]) -> str | None:
    best_key = best_rate = None
    for key in lines[0]:
        values = [line.get(key) for line in lines]
        called = bool(re.search(ID_SLOT, str(key), re.I))
        if not all((_id_akin(v, called) for v in values)):
            continue
        values = [v.strip() for v in values]
        if len(lines) > 1 and len(set(values)) < len(values):
            continue
        rating = (called, sum(map(len, values)) / len(values))
        if best_rate is None or rating > best_rate:
            best_rate, best_key = (rating, key)
    return best_key

def _in_lines(leaf: Any, iid: str, level: int=0) -> bool:
    if level > 5:
        return False
    if isinstance(leaf, dict):
        return any((_in_lines(v, iid, level + 1) for v in leaf.values()))
    if not isinstance(leaf, list):
        return False
    if any((_g1_holds(row, iid) for row in leaf)):
        return True
    return any((_in_lines(v, iid, level + 1) for v in leaf))

def _lm_within(seconds_n: float) -> ProxyClient:
    if seconds_n >= 50:
        return _lm
    bound = max(5, int(seconds_n) // 5 * 5)
    client = _lm_brief.get(bound)
    if client is None:
        client = _lm_brief[bound] = ProxyClient(timeout=bound, max_retries=1, retry_delay=1)
    return client

def _detail_chunk(leaf: Any) -> dict[str, Any] | None:
    if not isinstance(leaf, dict):
        return None
    top = None
    for val in leaf.values():
        if not isinstance(val, dict) or len(val) < 2:
            continue
        scalars = {}
        for k, v in val.items():
            if isinstance(v, (str, int, float)) and (not isinstance(v, bool)):
                scalars[k] = v
        if len(scalars) >= 2 and (top is None or len(top) < len(scalars)):
            top = scalars
    return top

def _mild(fallback_n: Any) -> Any:

    def wrap_n(method_n: Any) -> Any:

        def guarded_n(self: Any, *args_n: Any, **kwargs_n: Any) -> Any:
            try:
                return method_n(self, *args_n, **kwargs_n)
            except Exception as err:
                return _g1_soft_fail(self, method_n, err, fallback_n)
        return guarded_n
    return wrap_n

def _g1_fit_lists(text_n):
    works, bars = ([], [])
    for m in re.finditer('\\b(?<!not )(?<!never )(?:works with|compatible with|fits)\\s*:?\\s+([^.;]+)', text_n, re.I):
        lead = text_n[max(0, m.start() - 40):m.start()]
        if re.search('\\bpart\\s+' + PIECE_TAG_CODE + '\\s+also\\s+$', lead, re.I):
            continue
        head, *tail = re.split('(?i)\\b(?:except|excluding|but not)\\b', m.group(1), maxsplit=1)
        works.extend(_tag_codes(head))
        if tail:
            bars.extend(_tag_codes(tail[0]))
    return (works, bars)

def _g1_braced(plain):
    try:
        parsed = json.loads(plain)
    except (ValueError, RecursionError):
        pairs = dict(re.findall('[\'"]?([A-Za-z][\\w ./-]{0,38})[\'"]?\\s*[:=]\\s*[\'"]?([^,\'"}\\]]*)', plain))
        if len(pairs) < 2:
            return None
        return {k.strip(): _bare(v) for k, v in pairs.items()}
    return parsed if isinstance(parsed, dict) else None

def _g1_assigns(plain):
    out = {}
    last = ''
    for piece in re.split(';\\s+|;$|\\s\\|\\s|\\n', plain):
        if not piece.strip():
            continue
        name, eq, val = piece.partition('=')
        key = name.strip()
        if eq and 0 < len(key) <= CHUNK_NAME and (key not in out):
            last = key
            out[last] = _chunk_val(val)
        elif last:
            out[last] = f'{out[last]}; {piece.strip()}'
        else:
            return None
    if len(out) >= 2 or (out and ':' not in last and (len(last) <= 30) and (len(plain) <= 90)):
        return out
    return None

def _g1_named(couples):
    return {name: _bare(val) for name, val in couples}

def _g1_listed(plain):
    rows = list(filter(str.strip, plain.splitlines()))
    if len(rows) >= 2 and all((re.match(LIST_BULLETN, row) for row in rows)):
        couples = [_couple(row.strip(), ': ') for row in rows]
        if all(couples):
            return _g1_named(couples)
    for amid in (', ', '; '):
        if amid not in plain or ': ' not in plain:
            continue
        couples = [_couple(piece.strip(), ': ') for piece in plain.split(amid)]
        if len(couples) < 2 or not all(couples):
            continue
        if any(('.' in name or len(val) > CHUNK_VAL for name, val in couples)):
            continue
        return _g1_named(couples)
    return None

def _g1_best_split(text_n, amid):
    total = sum((1 for p in text_n.split(amid) if p.strip()))
    floor = (COUPLE_PORTION if ', ' == amid else SLOT_PORTION) * total
    best = None
    for sign in COUPLE_SIGNS:
        slots, called = _split_slots(text_n, amid, sign)
        if called < ENTRY_SLOTS or called < floor:
            continue
        if 2 * len({name for name, _ in slots}) <= len(slots):
            continue
        if best is None or best[0] < called:
            best = (called, slots)
    return best

def _g1_stacked(plain):
    for pieces in (re.split('\\n\\s*\\n', plain), plain.splitlines()):
        if len(pieces) < 2:
            continue
        found = [r for r in map(_entry_text, pieces) if r]
        if len(found) < 2 or 5 * len(found) < 4 * len(pieces):
            continue
        if len({next(iter(r)) for r in found}) == 1:
            return found
    return None

def _g1_dict_pairs(lines, first):
    shape = set(first)
    for row in lines:
        if not isinstance(row, dict) or set(row) != shape or len(row) != 2:
            return None
    k0, k1 = list(first)
    if not re.search(COUPLE_KEY, str(k0), re.I):
        return None
    out = {}
    for row in lines:
        name, val = (row[k0], row[k1])
        if isinstance(name, str) and 0 < len(name) <= CHUNK_NAME and (not isinstance(val, (dict, list))):
            out[name.strip()] = val
        else:
            return None
    return out

def _g1_seq_pairs(lines):
    out = {}
    for row in lines:
        good = isinstance(row, (list, tuple)) and len(row) == 2 and isinstance(row[0], str)
        if not good or isinstance(row[1], (dict, list)) or (not 0 < len(row[0]) <= CHUNK_NAME):
            return None
        out[row[0].strip()] = row[1]
    return out

def _g1_text_pairs(lines):
    out = {}
    for row in lines:
        couple = _couple(row.strip(), ': ') or _couple(row.strip(), '=')
        if not couple:
            return None
        name, val = couple
        if len(val) > CHUNK_VAL or len(name.split()) > 3:
            return None
        out[name] = _bare(val)
    return out

def _g1_is_header(v):
    return isinstance(v, list) and len(v) >= 2 and all((isinstance(c, str) and re.fullmatch(SLOT_NAME, c) for c in v))

def _g1_keyed_or(v):
    if isinstance(v, dict):
        return _keyed_lines(v) or v
    return v

def _g1_nested(v, wide):
    if not isinstance(v, dict):
        return False
    inner = list(v.values())
    if not any((isinstance(x, dict) for x in inner)) or any((isinstance(x, list) for x in inner)):
        return False
    return isinstance(next(iter(v.values()), None), (dict, str, int, float)) and wide and (not _contains_id(v))

def _g1_recover_map(val, level, page):
    leaf = _id_strings(_column_lines(val))
    carded = len(leaf) >= ENTRY_SLOTS or _contains_id(leaf)
    leaf = {k: _g1_keyed_or(v) for k, v in leaf.items()}
    offer = _contains_id(leaf) or any((_id_akin(v) for v in leaf.values()))
    out = {}
    for k, v in leaf.items():
        if not isinstance(v, str) or len(leaf) <= 2:
            out[k] = _recover_slot(v, level, carded, page)
        elif offer:
            out[k] = _chunk_text(v)
        else:
            out[k] = v
    if level <= 3:
        wide = len(leaf) >= ENTRY_SLOTS
        out = {k: _flat_chunk(v) if _g1_nested(v, wide) else v for k, v in out.items()}
    if len(out) == 1:
        sole = next(iter(out.values()))
        if isinstance(sole, dict) and len(sole) >= ENTRY_SLOTS and isinstance(next(iter(val.values())), str):
            return sole
    return out

def _g1_unplural(w):
    return w[:-1] if len(w) > 3 and w.endswith('s') else w

def _g1_fits(val, sorts):
    return any((_matches_sort(val, k) for k in sorts))

def _g1_coerce(val, sort2):
    if 'integer' == sort2 and isinstance(val, float) and val.is_integer():
        return (True, int(val))
    if sort2 in ('integer', 'number') and isinstance(val, str) and re.fullmatch('\\s*-?\\d+(\\.\\d+)?\\s*', val):
        num = float(val)
        return (True, int(num) if 'integer' == sort2 or num.is_integer() else num)
    if 'string' == sort2 and isinstance(val, (int, float)) and (not isinstance(val, bool)):
        return (True, str(val))
    if 'boolean' == sort2 and isinstance(val, str) and (val.strip().lower() in ('true', 'false')):
        return (True, val.strip().lower() == 'true')
    if 'array' == sort2 and (not isinstance(val, list)) and (val is not None):
        return (True, [val])
    return (False, val)

def _g1_enum_fix(val, choices):
    if isinstance(val, str):
        small = val.strip().lower()
        for option in choices:
            if isinstance(option, str) and option.lower() == small:
                return option
    return val

def _g1_form_list(val, spec, locus, issues):
    if isinstance(spec.get('maxItems'), int) and spec['maxItems'] < len(val):
        issues.append(f"{locus} allows at most {spec['maxItems']} entries")
    if isinstance(spec.get('minItems'), int) and spec['minItems'] > len(val):
        issues.append(f"{locus} needs at least {spec['minItems']} entries")
    if not isinstance(spec.get('items'), dict):
        return val
    shaped = []
    for i, v in enumerate(val):
        shaped.append(_form(v, spec['items'], f'{locus}[{i}]', issues))
    return shaped

def _g1_form_map(val, spec, locus, issues):
    traits = spec.get('properties') if isinstance(spec.get('properties'), dict) else {}
    if spec.get('additionalProperties') is False:
        for name in val:
            if name not in traits:
                issues.append(f'{locus}.{name} is not an allowed argument')
    for name in spec.get('required') or []:
        if name not in val:
            issues.append(f'{locus}.{name} is required')
    shaped = {}
    for k, v in val.items():
        shaped[k] = _form(v, traits.get(k), f'{locus}.{k}', issues) if k in traits else v
    return shaped

def _g1_holds(row, iid):
    return isinstance(row, dict) and any((isinstance(x, str) and str(x).strip() == iid for x in row.values()))

def _g1_soft_fail(owner, method_n, err, fallback_n):
    owner.journal.append({'kind': 'soft_crash', 'where': method_n.__name__, 'error': f'{type(err).__name__}: {err}'[:300]})
    if callable(fallback_n):
        return fallback_n()
    return fallback_n

def _g1_flip(row, leaf, was, now):
    return any((row.get(k) is was and v is now for k, v in leaf.items()))

def _g1_is_tool(t):
    return isinstance(t, dict) and isinstance(t.get('function'), dict) and isinstance(t['function'].get('name'), str)

def _g1_params(t):
    return t['function'].get('parameters') if isinstance(t['function'].get('parameters'), dict) else {'type': 'object', 'properties': {}}

def _g1_step_cap(pane):
    moves = _tally(pane.get('max_steps'))
    if moves is None and isinstance(pane.get('max_steps'), float):
        moves = int(pane['max_steps'])
    return moves if moves and moves > 0 else 30

def _g2_is_num(val: Any) -> bool:
    if isinstance(val, bool):
        return False
    return isinstance(val, (int, float))

def _g2_kind(s_n: Any, kind: str, without: Any=None) -> bool:
    if not isinstance(s_n, dict) or kind not in _sorts(s_n):
        return False
    return without is None or without not in s_n

def _g2_first_key(props: dict[str, Any], kind: str, without: Any=None) -> str | None:
    for k_n, s_n in props.items():
        if _g2_kind(s_n, kind, without):
            return k_n
    return None

def _g2_shape(payload: Any, spec: Any, label: str) -> tuple[Any, list[str]]:
    found: list[str] = []
    return (_form(payload, spec, label, found), found)

def _g2_text_at(row: dict[str, Any], key: Any, blank: str) -> str:
    return str(row.get(key) or blank)

def _g2_field(spec: Any, key: str) -> Any:
    return (spec or {}).get(key) or {}

def _g2_agree(x_n: str, y_n: str) -> bool:
    left_n = re.findall('[a-z0-9]+', x_n.lower())
    right_n = re.findall('[a-z0-9]+', y_n.lower())
    if y_n == x_n:
        return True
    if len(left_n) <= len(right_n):
        short, lengthy = (left_n, right_n)
    else:
        short, lengthy = (right_n, left_n)
    if not short:
        return False
    return set(short) <= set(lengthy) and len(lengthy) <= 2 * len(short) + 2

def _g6_bag(text_n: str) -> set[str]:
    return set(re.sub('[,/;|]', ' ', _terms(text_n)).split())

def _g6_demand_shown(demand: str, chunk: dict[str, Any]) -> bool:
    trait, __n, val = demand.partition(':')
    terms = _g6_bag(val)
    if not terms or not terms.isdisjoint(_g6_NEGATORS):
        return False
    label = _written(trait)
    for k_n, v_n in chunk.items():
        if _names_n(label, _written(k_n)) and terms <= _g6_bag(str(v_n)):
            return True
    return False

def _g6_counts_as_hit(val: Any) -> bool:
    if isinstance(val, (list, str)):
        return bool(val)
    return isinstance(val, (int, float)) and (not isinstance(val, bool)) and (val > 0)

def _g6_phrases(leaf: Any, level: int) -> Any:
    if level > 4:
        return
    if isinstance(leaf, str):
        if len(leaf.split()) >= 2:
            yield leaf
        return
    kids = leaf.values() if isinstance(leaf, dict) else leaf if isinstance(leaf, list) else ()
    for kid in kids:
        yield from _g6_phrases(kid, level + 1)

def _g6_enum_words(spec: Any, level: int) -> Any:
    if level > 6 or not isinstance(spec, dict):
        return
    yield from (option_n.strip().lower() for option_n in spec.get('enum') or [] if isinstance(option_n, str))
    for val in spec.values():
        yield from _g6_enum_words(val, level + 1)

def _g6_spoke_at_order(p_n: dict[str, Any]) -> bool:
    return p_n['when'] == 'order' and p_n['messages'] > 0

def _g6_bad_page(rows: Any, count: Any, cursor: Any) -> bool:
    if not isinstance(rows, list) or not rows:
        return True
    if isinstance(count, bool) or not isinstance(count, int):
        return True
    if cursor is None:
        return False
    return isinstance(cursor, bool) or not isinstance(cursor, (str, int))

def _g6_said(substance: Any) -> str:
    if isinstance(substance, str):
        return substance
    return _serial(substance) if substance else ''

def _g6_value_tries(values: list[str], novel: list[str], name_n: str) -> list[list[str]]:
    clue = []
    for v_n in novel:
        if len(v_n.split()) == 1 or re.fullmatch(AMOUNTN, v_n, re.I):
            clue.append(v_n)
    ordered = clue + [v_n for v_n in values if v_n not in clue]
    if len(ordered) > 1:
        stated_n = [i_n for i_n, v_n in enumerate(ordered) if v_n not in novel]
        cut = stated_n[-1] if stated_n else 0
        return [ordered, ordered[:cut] + ordered[cut + 1:]]
    if clue:
        return [[ordered[0], name_n], ordered]
    return [ordered]

def _g6_enum_pick(key: str, choices2: list[Any], required_n: list[str], final: dict[str, Any]) -> tuple[str, Any] | None:
    reads_n = []
    rest = []
    for o_n in choices2:
        if isinstance(o_n, str) and re.search(PHRASE_TERMS, o_n, re.I):
            reads_n.append(o_n)
        if o_n != final.get(key):
            rest.append(o_n)
    if len(reads_n) == 1:
        return (key, reads_n[0])
    if key in required_n and len(choices2) == 2 and (len(rest) == 1):
        return (key, rest[0])
    return None

def _g6_long_words(words: Any) -> set[str]:
    return {w_n for w_n in words if len(w_n) >= 4}

def _g6_round_robin(tracks: list[list[str]], ceiling: int) -> list[str]:
    picked: list[str] = []
    for depth in range(ceiling):
        picked.extend((track[depth] for track in tracks if depth < len(track)))
    return picked[:ceiling]

def _g6_layered(tracks: list[list[str]], heads: list[int], depth: int, upper: bool) -> list[str]:
    out_n: list[str] = []
    for k_n in range(depth):
        for track, head in zip(tracks, heads):
            if k_n < len(track) and (k_n < head) == upper:
                out_n.append(track[k_n])
    return out_n

def _g6_shape(fields: Any, spec: Any, locus: str) -> tuple[Any, bool]:
    issues: list[str] = []
    shaped = _form(fields, spec, locus, issues)
    return (shaped, not issues)

def _g6_item_marks(ware2: Any) -> tuple[str, str, set[str]]:
    return (_terms(ware2.aisle or ''), _terms(ware2.name_n), _g6_long_words(_root(ware2.aisle or '') | _root(ware2.name_n)))

def _g6_array_slot(traits: dict[str, Any]) -> str | None:
    for k_n, v_n in traits.items():
        if isinstance(v_n, dict) and 'array' in _sorts(v_n):
            return k_n
    return None

def _g6_arg_terms(args_n: dict[str, Any]) -> set[str]:
    said = [v_n for v_n in args_n.values() if isinstance(v_n, str)]
    return _text_terms(' '.join(said).lower())

def _g6_line_text(line: dict[str, Any], group_key: str | None) -> str:
    said = [str(v_n) for k_n, v_n in line.items() if isinstance(v_n, str) and k_n != group_key]
    return ' '.join(said)

def _g6_item_cap(schema: dict[str, Any]) -> int:
    return schema.get('maxItems') if isinstance(schema.get('maxItems'), int) else 4

def _g6_wind_up(session: Any) -> list[dict[str, Any]]:
    try:
        if not session.closed:
            session.conclude()
    except Exception as again:
        told = f'{type(again).__name__}: {again}'
        session.journal.append({'kind': 'crash', 'error': told[:400]})
    return list(session.journal)

class _Stamped_n(list):

    def __init__(self, timer: Any):
        list.__init__(self)
        self.timer = timer

    def append(self, record: dict[str, Any]) -> None:
        stamp = round(self.timer(), 1)
        record.setdefault('at', stamp)
        list.append(self, record)

def _filler_n(schema: dict[str, Any]) -> Any:
    if isinstance(schema.get('enum'), list) and schema['enum']:
        return schema['enum'][0]
    for term2 in ('const', 'default'):
        if term2 in schema:
            return schema[term2]
    sorts = _sorts(schema)
    if 'integer' in sorts or 'number' in sorts:
        return schema['minimum'] if isinstance(schema.get('minimum'), (int, float)) else 1
    if 'boolean' in sorts:
        return False
    if 'string' in sorts:
        return 'question'[:schema.get('maxLength') or 8]
    return None

def _strings_n(leaf: Any) -> list[str]:
    if isinstance(leaf, str):
        return [leaf]
    if isinstance(leaf, dict):
        leaf = list(leaf.values())
    elif not isinstance(leaf, list):
        return []
    out = []
    for v in leaf:
        out.extend(_strings_n(v))
    return out

def _says_n(leaf: Any, text_n: str) -> bool:
    if isinstance(leaf, (dict, list)):
        kids = leaf.values() if isinstance(leaf, dict) else leaf
        return any((_says_n(v, text_n) for v in kids))
    return text_n == leaf

class ware:

    def __init__(self, name_n: str, aisle: str, amount: int, needs: list[str]):
        vars(self).update(name_n=name_n, aisle=aisle, amount=amount, needs=needs, heard=[], lifted=[], open2=[])

class NvidiaBuyer:

    def __init__(self, trouble: dict[str, Any]):
        environment = trouble['environment']
        self.binding_n = dict(environment['binding'])
        pane = self.pane = environment['policy_view']
        self.demand2 = str(pane.get('query') or '')
        self.ops = list(filter(_g1_is_tool, pane.get('tools') or []))
        self.specs = {t['function']['name']: _g1_params(t) for t in self.ops}
        self.max_moves = _g1_step_cap(pane)
        vars(self).update(started_n=time.monotonic(), move=0, closed=bool(self.pane.get('finished')), broke_n=False, none_merit=False, waited_n=0, failures=0, expense=0.0, invocations=0, cycle_invocations=0, journal=_Stamped_n(self.timer), digest={}, duties={}, items=[], allowance=None, text_bound=0, text_spent=0, procedure=[], reserve_n=0, barrier_texts=0, barrier_fine=set(), completed_log=[], completed_moves=[], queried=False, qa_n=[], lines={}, whole={}, viewed_sequence=[], learnt=set(), memos=[], edition=0, basket_touched=False, basket_at=-1, basket_viewed_at=-1, basket_pane=None, memo_called=False, basket_rows={}, checked_n=set(), audited_n=False, deal_viewed={}, vanished={}, unavailable_n=set(), vended_rows=set(), memo_vended=False, kept=0, id_args_n=set(), basket_stated='', declined_at=-1, clues=[], decline_chunks=0, pushed_n=False, built_n=[], lifted=set(), tardy_kept=False, clue_wares={}, verdicts_n={}, confirmed_n=None, completed_n=False, tie_attempted=set(), track_failures={}, adopted_n={}, capped_n=set(), factless_n=set(), path_space=None, id_key_n=None, details_from={}, pending=[], purchases=0, refusals_n=0, recent_n=[], dialogue=[], unfolded=set(), panes={}, match_stored=None, holders_viewed={}, pointers=[], pointer_vals={}, trails_n=0, path_chunks=0, contrast_fine=0, contrast_poor=0, op_fields={}, pane_fields={}, scan_cache={}, sheets_scan=0, shapes_n={}, cost_key=None, stock_key_n=None, lower_priced_signs=set(), broader_kept=False, paginated_obs=[], scheme_sequence=[], tracks={}, track_located={}, track_used_up={}, track_spent={}, tardy_queried=False, one_product_n=set(), list_asks_n=0, apart_n={}, trail_upward=[], offer=True, row_contains=0, limited=[], match_contains=0, looked=[], tie_searches=0, tie_viewed={}, hunts_n=0, world_margin=WORLD_MARGIN, world_used_up=0.0, world_tally=0, scan_invocations={}, scan_args={}, val_lines=set(), line_fields=set(), decline_trailed=False, world_at=0.0, chase_root={}, chase_none_term=set(), plain_sheets={}, tag_code_attempts=[], tag_codes_dead=False, hear_reads_n=0, match_hunted=False, chase_sign='', chase_attempted={}, sheet_args={}, sheet_queries={})

    def absent_wares(self) -> list[int]:
        group_key = self.section_key()
        iids = [i for i in self.basket_iids() if (self.kept_measures(i) or 0) > 0]
        if not group_key or len(iids) >= len(self.items):
            return []
        tags = [str(row.get(group_key) or '') for row in self.lines.values()]
        kept = [str((self.lines.get(i) or {}).get(group_key) or '') for i in iids]
        out = []
        for n, ware2 in enumerate(self.items, 1):
            if self._g1_owns_any(ware2, tags) and (not self._g1_owns_any(ware2, kept)):
                out.append(n)
        return out

    def absorb_n(self, obs: Any) -> None:
        self._g1_walk(obs, 0, False)

    def adopt_tags(self) -> None:
        group_key = self.section_key()
        if len(self.items) < 2 or not group_key:
            return
        self.piece_aisles()
        tags = {str(entry.get(group_key) or '') for entry in self.whole.values()} - {''}
        strays = [tag for tag in tags if not self.holder_of(tag)]
        orphans = []
        for n in range(1, len(self.items) + 1):
            if not any((self.holder_of(tag) == n and tag not in self.adopted_n for tag in tags)):
                orphans.append(n)
        if not strays or len(orphans) != 1:
            return
        home = orphans[0]
        for tag in strays:
            self.adopted_n[tag] = home
        self.journal.append({'kind': 'adopted', 'item': home, 'labels': strays})

    def assesses(self, bound: float=SPEND_MILD) -> list[str]:
        spent = self.expense
        if not SPEND_HARDN >= spent:
            return []
        picks = [LIGHTN]
        if spent <= bound and spent + HEAVY_SPACE <= SPEND_HARDN:
            picks.insert(0, HEAVYN)
        return picks

    def barrier(self, phase: str) -> tuple[bool, str]:
        moves = [p_n for p_n in self.procedure if p_n['when'] == phase]
        if not moves or self.closed or (not self.barrier_due(phase)):
            return (True, '')
        closing = {self.duties.get('buy'), self.duties.get('refuse')} - {None}
        ops = self._g2_tool_lines(closing)
        stated: list[str] = []
        nudged_n = False
        rounds_left = BARRIER_ROUNDS
        while rounds_left > 0:
            rounds_left -= 1
            prompt_n = self._g2_gate_prompt(phase, moves, ops)
            obtained = self.query_json(prompt_n, [HEAVYN, LIGHTN] if nudged_n else [LIGHTN, HEAVYN], 3500)
            if obtained is None:
                break
            heard: list[str] = []
            grip = ''
            if stated:
                self.grab_learnt({'needs': obtained.get('needs')}, heard)
                grip = str(obtained.get('hold') or '').strip()
            invoke = None
            if isinstance(obtained.get('call'), dict):
                invoke = obtained.get('call')
            cut_call, cut_steps = (_trim(invoke, 200), _trim(obtained.get('steps'), 120))
            self._g2_log({'kind': 'gate', 'stage': phase, 'call': cut_call, 'steps': cut_steps, 'hold': grip[:200], 'told': heard})
            if grip or heard:
                self.edition += 1
                return (False, self._g2_gate_redo(stated, heard, grip))
            name_n = str((invoke or {}).get('tool') or '')
            talks = any((p_n['messages'] > 0 for p_n in moves)) if 'order' == phase else False
            unproven_n = self._g2_unproven(obtained.get('steps'), self._g2_gate_evidence(talks))
            if not (invoke or stated or nudged_n) and unproven_n and self._g2_gate_may_ask():
                nudged_n = True
                self._g2_gate_nudge(unproven_n)
                continue
            if not invoke:
                self.barrier_fine.add((phase, self.basket_sign() + f'#{self.edition}'))
                break
            if name_n in closing or name_n not in self.specs:
                break
            asking = name_n == self.duties.get('ask')
            if self.remaining() < 2 or (asking and self.text_spent >= self.text_bound):
                break
            given = invoke.get('args')
            args_n, issues = _g2_shape(given if isinstance(invoke.get('args'), dict) else {}, self.specs[name_n], 'args')
            if issues:
                break
            if name_n == self.duties.get('ask'):
                if self._g2_gate_ask(args_n, stated):
                    continue
                break
            if not self._g2_gate_step(name_n, args_n, stated):
                break
        if not stated:
            return (True, '')
        return (True, 'Before going on, NVIDIA did the steps the request asks for:\n' + '\n'.join(stated))

    def barrier_due(self, phase: str) -> bool:
        for p_n in self.procedure:
            if p_n['when'] == phase:
                break
        else:
            return False
        return (phase, self.basket_sign() + f'#{self.edition}') not in self.barrier_fine

    def basket_collections(self, name_n: str) -> bool:
        traits = self.specs.get(name_n, {}).get('properties') or {}
        schemas = [s_n for s_n in traits.values() if isinstance(s_n, dict)]
        for s_n in schemas:
            if isinstance(s_n.get('enum'), list) and len(s_n['enum']) >= 2:
                return False
        for s_n in schemas:
            if 'integer' in _sorts(s_n) and s_n.get('minimum') == 0:
                return True
        return False

    def basket_iids(self) -> list[str]:
        if not self._g2_pane_fresh():
            return list(self.basket_rows)
        text_n = _serial(self.basket_pane)
        found: list[str] = []
        for i_n in self.lines:
            if i_n in text_n:
                found.append(i_n)
        return found

    def basket_invocations(self, rows: dict[str, int], discard2: list[str]) -> list[dict[str, Any]] | None:
        name_n = self.duties.get('cart') or ''
        spec = self.specs.get(name_n) or {}
        traits = spec.get('properties') or {}
        nested, made = self._g2_nested_calls(traits, rows)
        if nested:
            return made
        strings = [k_n for k_n, s_n in traits.items() if _g2_kind(s_n, 'string', 'enum')]
        goal = self._g2_cart_goal(name_n, spec, strings)
        count = _g2_first_key(traits, 'integer')
        manner = None
        for k_n, s_n in traits.items():
            if isinstance(s_n, dict) and isinstance(s_n.get('enum'), list) and (len(s_n['enum']) >= 2):
                manner = k_n
                break
        if not goal:
            return None
        invocations: list[dict[str, Any]] = []
        for iid in discard2:
            call = self._g2_drop_call(name_n, traits, goal, manner, count, iid)
            if call is None:
                return None
            invocations.append(call)
        for iid, desire in rows.items():
            has = self.basket_rows.get(iid, 0) if iid in self.basket_iids() else 0
            if desire == has:
                continue
            if has and (not self.basket_collections(name_n)):
                return None
            call = {goal: iid}
            if manner:
                call[manner] = traits[manner]['enum'][0]
            if count:
                call[count] = desire
            invocations.append(call)
        return invocations

    def basket_sign(self) -> str:
        pairs = list(self.basket_rows.items())
        pairs.sort()
        return _serial(pairs)

    def basket_text(self) -> str:
        if not self._g2_pane_fresh():
            return "lines by NVIDIA's record of the cart calls (listing: units): " + _serial(self.basket_rows)
        shown = _trim(self.basket_pane, 300)
        return _serial(shown)[:3000]

    @_mild('')
    def below_basket(self) -> str:
        group_key = self.section_key()
        basket = self.basket_iids()
        rows = []
        for row in basket:
            cost = self.cost_of(row)
            sort2 = self._g2_low_tag(row, group_key, '') if group_key else ''
            if cost is None or not sort2:
                continue
            reserve = []
            for i_n in self.viewed_sequence:
                if i_n in basket or i_n in self.whole or (not self.purchasable(i_n)):
                    continue
                if _g2_text_at(self.lines.get(i_n) or {}, group_key, '').strip().lower() != sort2:
                    continue
                other_n = self.cost_of(i_n)
                if not (float('inf') if other_n is None else other_n) < cost:
                    continue
                reserve.append(i_n)
            reserve.sort(key=lambda i_n: self.cost_of(i_n) or 0.0)
            for i_n in reserve[:LOWER_PRICED_DISPLAYED]:
                rows.append(f'{i_n} at {self.cost_of(i_n):g}')
        if not rows:
            return ''
        return 'Still unread and below a cart line: ' + ', '.join(rows) + '.'

    def broader_grip(self) -> str:
        done_n = self.broader_kept or self.digest.get('pick') != 'cheapest'
        if done_n or 14 + 2 * len(self.items) > self.remaining():
            return ''
        self.broader_kept = True
        rows = []
        for i_n in self.basket_iids():
            if self.cost_of(i_n) is not None:
                rows.append(f'{i_n} at {self.cost_of(i_n):g}')
        tail = ''
        if rows:
            tail = f" ({'; '.join(rows)})"
        return f"Not run yet (no step used). {self.remaining()} steps are left and the pick is the cheapest. Searches may read only part of a listing, so a requirement's value finds only some of the fitting listings, and rows come in no price order. Before checking out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line" + tail + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'

    def called_tag(self, num: int, tag: str) -> str:
        desired = _terms(self.items[num - 1].aisle or '')
        group_key = self.section_key()
        if desired and group_key and (_terms(tag) != desired):
            for r_n in self.lines.values():
                if _terms(_g2_text_at(r_n, group_key, '')) == desired:
                    return str(r_n.get(group_key))
            return ''
        return ''

    def can_decline(self) -> bool:
        decline = self.duties.get('refuse')
        if not decline:
            return False
        quiet = self.quiet_n(decline, dict(self.digest.get('refuse_args') or {}))
        return not _g2_shape(quiet, self.specs.get(decline), 'args')[1]

    def ceiling_fields(self, op: str) -> list[str]:
        traits = _g2_field(self.specs.get(op), 'properties')
        nums: list[str] = []
        for k_n, v_n in traits.items():
            if not isinstance(v_n, dict) or not {'integer', 'number'} & set(_sorts(v_n)):
                continue
            if re.search(CEILING_NOT, k_n, re.I) or re.search(CEILING_SMALL, _snaky(k_n), re.I):
                continue
            nums.append(k_n)
        kept = set()
        for (key, __n), fine in self.ceiling_manners(op).items():
            if fine:
                kept.add(key)
        snakes = {k_n: _snaky(k_n) for k_n in nums}
        called = [k_n for k_n in nums if re.search(CEILING_ARG, snakes[k_n], re.I)]
        priced_n = [k_n for k_n in nums if re.search(CEILING_PRICED, snakes[k_n], re.I)]
        first = [k_n for k_n in nums if k_n in kept]
        if first:
            return first
        both = [k_n for k_n in called if k_n in priced_n]
        if both:
            return both
        return priced_n or called or []

    def ceiling_ignored(self, name_n: str, args_n: dict[str, Any], obs: Any) -> str:
        traits = _g2_field(self.specs.get(name_n), 'properties')
        limits_n = self.ceiling_fields(name_n)
        ceilings = []
        for k_n, v_n in args_n.items():
            if k_n not in traits or not _g2_is_num(v_n):
                continue
            if re.search('max|most|under|below|limit|cap|upto|up_to', k_n, re.I) and self._g2_money_slot(traits, k_n):
                ceilings.append(k_n)
            elif k_n in limits_n[:1] and len(limits_n) == 1:
                ceilings.append(k_n)
        if not ceilings or not self.cost_key:
            return ''
        slot = sorted(ceilings, key=args_n.get)[0]
        ceiling = args_n[slot]
        over = 0
        for i_n, __n in self.entries_in(obs):
            if (self.cost_of(i_n) or 0) > ceiling:
                over += 1
        if not over:
            return ''
        rest = {}
        for k_n, v_n in args_n.items():
            if isinstance(v_n, str) and isinstance((traits.get(k_n) or {}).get('enum'), list):
                rest[k_n] = [o_n for o_n in traits[k_n]['enum'] if v_n != o_n]
        hint_n = ''
        if any(rest.values()):
            shown = [f'{k_n}={_serial(o_n[0])}' for k_n, o_n in rest.items() if o_n]
            hint_n = '; list the category with ' + ', '.join(shown) + f' and the same {slot} to see only cheaper listings'
        return f'\n[NVIDIA] This search did not apply {slot}={ceiling:g}: {over} of its rows cost more{hint_n}.'

    def ceiling_manners(self, op: str) -> dict[tuple[str, tuple[tuple[str, Any], ...]], bool]:
        traits = _g2_field(self.specs.get(op), 'properties')
        group_key = self.section_key()
        out_n: dict[tuple[str, tuple[tuple[str, Any], ...]], bool] = {}
        costs = self._g2_costs_by_tag(group_key)
        for sign, (name_n, args_n) in self.scan_args.items():
            if name_n != op:
                continue
            lines = [i_n for i_n, __n in self.entries_in(self.scan_invocations.get(sign))]
            displayed = [self.cost_of(i_n) for i_n in lines]
            if not lines or None in displayed:
                continue
            pairs = []
            for k_n, v_n in args_n.items():
                if isinstance(traits.get(k_n), dict) and isinstance(traits[k_n].get('enum'), list) and isinstance(v_n, (str, int, float)):
                    pairs.append((k_n, v_n))
            pairs.sort()
            manner = tuple(pairs)
            tags = {self._g2_low_tag(i_n, group_key, '') for i_n in lines} if group_key else set()
            reserve = []
            for tag in tags:
                reserve.extend(costs.get(tag, []))
            top = max(displayed)
            for key, val in args_n.items():
                if key not in traits or not _g2_is_num(val):
                    continue
                if top > val:
                    out_n[key, manner] = False
                    continue
                if out_n.get((key, manner)) is False or not reserve:
                    continue
                share = sum((val >= p_n for p_n in reserve)) / len(reserve)
                if share ** len(lines) < CEILING_CHANCE:
                    out_n[key, manner] = True
        return out_n

    def ceiling_pattern(self) -> tuple[str, dict[str, Any], str, str] | None:
        for op, spec in self.specs.items():
            traits = _g2_field(spec, 'properties')
            ceilings = self.ceiling_fields(op)
            aisles = [k_n for k_n, v_n in traits.items() if _g2_kind(v_n, 'string', 'enum') and re.search(AISLE_ARG, _snaky(k_n), re.I)]
            if len(aisles) == 1 and len(ceilings) == 1:
                return self._g2_ceiling_origin(op, spec, traits, ceilings[0], aisles[0])
        return None

    @_mild('')
    def chase_lower_priced(self) -> str:
        sign = _serial([self.demands_now(), self.allowance, sorted(self.vanished)])
        if self.digest.get('pick') != 'cheapest' or self.hunts_n >= CHASE_ROUNDS or self.chase_sign == sign:
            return ''
        scheme = self.val_pattern()
        if not (scheme and self.cost_key) or self.timer() > CHASE_BY:
            return ''
        op, origin, slot = scheme
        space = self._g2_hunt_room(2 * len(self.items))
        if space < 3:
            return ''
        self.hunts_n += 1
        self.chase_sign = sign
        basket = self.basket_iids()
        total = len(self.items)
        portion = max(3, space // max(1, total))
        used_up = novel_sum = 0
        num = 1
        while num <= total and used_up < space and (not self.closed):
            spent, gained_n = self.chase_ware(num, op, origin, slot, min(portion, space - used_up))
            used_up += spent
            novel_sum += gained_n
            num += 1
        self._g2_log({'kind': 'hunt', 'steps': used_up, 'new_rows': novel_sum, 'cart': basket})
        if not used_up:
            return ''
        self.broader_kept = True
        return f'NVIDIA ran {used_up} more searches for the lowest price, one step each, and they showed {novel_sum} more rows.'

    @_mild(list)
    def chase_matches(self) -> list[str]:
        if self.match_hunted or self.timer() > CHASE_BY:
            return []
        scheme = self.val_pattern()
        if not scheme:
            return []
        op, origin, slot = scheme
        retain = self.end2() + len(self.items) + 1
        spare_n = self.remaining() - retain
        if spare_n < 3:
            return []
        self.match_hunted = True
        group_key = self.section_key()
        nums = self._g2_match_targets(group_key)
        desired = {}
        for num in nums:
            pieces_n = set()
            for val, __n in self.val_terms(num):
                pieces_n.update(_val_pieces(val))
            desired[num] = pieces_n
        paginated = self.unread_sheets(op, desired, min(MATCH_SHEETS, spare_n // 2))
        space = self._g2_hunt_room(0)
        used_up = self._g2_match_search(nums, op, origin, slot, space) if space >= 3 else 0
        if used_up is None:
            return []
        ceiling = self.allowance if isinstance(self.allowance, (int, float)) else None
        choices: list[str] = []
        for num in nums:
            choices.extend(self._g2_match_pool(num, group_key, ceiling, desired.get(num) or set()))
        scan: list[str] = []
        start = 0
        while start < len(choices) and (not self._g2_overtime()):
            scan.extend(self.populate_details(choices[start:start + CHASE_MATCH_OPENS], retain))
            start += CHASE_MATCH_OPENS
        self._g2_log({'kind': 'hunt_fits', 'pages': paginated, 'searches': used_up, 'opened': len(scan)})
        return scan

    def chase_terms(self, num: int) -> tuple[str, list[str]]:
        group_key = self.section_key()
        ware2 = self.items[num - 1]
        title = self.heading_key()
        tags = []
        if group_key:
            for line in self.lines.values():
                if self.possesses(ware2, _g2_text_at(line, group_key, '')):
                    tags.append(_g2_text_at(line, group_key, ''))
        names_n = []
        for line in self.lines.values():
            if title and group_key and self.possesses(ware2, _g2_text_at(line, group_key, '')):
                names_n.append(_g2_text_at(line, title, ''))
        for entry in self.whole.values():
            if not group_key or not self.possesses(ware2, _g2_text_at(entry, group_key, '')):
                continue
            chunk = _detail_chunk(entry) or {}
            names_n.append(' '.join([str(v_n) for v_n in chunk.values() if isinstance(v_n, str) and len(v_n) <= CHASE_VAL_CHARS]))
        rest = set()
        for n_n, other_n in enumerate(self.items, 1):
            if n_n != num:
                rest.update(re.findall('[a-z]{4,}', ((other_n.aisle or '') + ' ' + other_n.name_n).lower()))
        if tags:
            tag = tags[0]
            peak = -1
            for t_n in tags:
                seen_n = tags.count(t_n)
                if seen_n > peak:
                    tag, peak = (t_n, seen_n)
            tag = tag.lower()
        else:
            tag = (ware2.aisle or ware2.name_n).lower()
        nouns_n = [w_n for w_n in re.findall('[a-z]{4,}', tag) if w_n not in rest]
        if num in self.chase_none_term:
            term = ''
        else:
            term = max(nouns_n, key=len) if nouns_n else ''
        if not names_n:
            return (term, [])
        tallies: dict[str, int] = {}
        for name_n in names_n:
            for term2 in {w_n.lower() for w_n in re.findall("[A-Za-z][A-Za-z'-]{2,}", name_n)}:
                tallies[term2] = tallies.get(term2, 0) + 1
        self2 = _bases(ware2.aisle or '') | _bases(ware2.name_n)
        ceiling_n = max(2, int(len(names_n) * CHASE_COMMON))
        terms = []
        for w_n, n_n in tallies.items():
            if not 2 <= n_n <= ceiling_n or w_n == term:
                continue
            if _bases(w_n) and (not _bases(w_n) & self2):
                terms.append(w_n)
        terms.sort(key=lambda w_n: (-tallies[w_n], w_n))
        return (term, terms)

    def chase_ware(self, num: int, op: str, origin: dict[str, Any], slot: str, space: int) -> tuple[int, int]:
        attempted = self.chase_attempted.setdefault(num, set())
        term, __n = self.chase_terms(num)
        values = [val for val, __n in self.val_terms(num)]
        tally: list[Any] = [0, 0, 0, False, False]

        def run_n(phrase: str) -> bool:
            return self._g2_chase_run(num, op, origin, slot, space, phrase, values, attempted, tally)
        sign = (num, tuple(values))
        root = self.chase_root.get(sign)
        if root is None:
            root = self._g2_chase_seed(num, values, term, run_n, tally)
            if root is None:
                return (tally[0], tally[1])
            self.chase_root[sign] = root
        while tally[0] < space and tally[2] < CHASE_INACTIVE and (not tally[4]) and (not self.closed) and (not self._g2_overtime()):
            __n, terms = self.chase_terms(num)
            words = root.lower().split()
            pick = next((w_n for w_n in terms if f'{w_n} {root}'.strip().lower() not in attempted and w_n not in words), None)
            if pick is None:
                break
            run_n(f'{pick} {root}'.strip())
        return (tally[0], tally[1])

    @_mild('')
    def cheapest_collections(self, primary: int, pieces: list[int], index: dict[str, Any]) -> str:
        choices2 = []
        systems_n = {}
        for piece in pieces:
            systems_n[piece] = self.tie_scheme(piece, primary, index)
        aisles = {}
        for piece in pieces:
            pool_n = []
            for i_n in self.ware_iids(piece):
                if self.cost_of(i_n) is None or not self.purchasable(i_n):
                    continue
                pool_n.append((self.cost_of(i_n), i_n, self.match_set(i_n, index), index['marks'].get(i_n, {}).get('bars') or []))
            pool_n.sort(key=lambda o_n: o_n[:2])
            aisles[piece] = pool_n
        for unit in self.ware_iids(primary):
            tag_codes = index['marks'].get(unit, {}).get('codes') or []
            cost = self.cost_of(unit)
            if not tag_codes and (not index['marks'].get(unit, {}).get('named')):
                continue
            if cost is None or not self.purchasable(unit):
                continue
            sum2 = cost * self.items[primary - 1].amount
            choices = [unit]
            for piece in pieces:
                fitting_n = self._g2_set_partner(aisles[piece], tag_codes, systems_n[piece], piece, unit, index)
                if not fitting_n:
                    choices = []
                    break
                choices.append(fitting_n)
                sum2 += (self.cost_of(fitting_n) or 0.0) * self.items[piece - 1].amount
            if choices:
                choices2.append((sum2, choices))
        if not choices2:
            return ''
        choices2.sort(key=lambda o_n: o_n[0])
        shown = [f"{' + '.join(p_n)} = {t_n:g}" for t_n, p_n in choices2[:SET_DISPLAYED]]
        return 'lowest whole sets seen, requirements not yet checked: ' + '; '.join(shown)

    def cheapest_here_n(self, observations_n: list[Any]) -> str:
        if self.digest.get('pick') != 'cheapest' or not self.cost_key:
            return ''
        located = set()
        for viewed in observations_n:
            located.update([i_n for i_n, __n in self.entries_in(viewed)])
        lines = []
        for i_n in located:
            if self.purchasable(i_n) and self.cost_of(i_n) is not None:
                lines.append((self.cost_of(i_n), i_n))
        if len(lines) < 2:
            return ''
        lines.sort()
        return '\n[NVIDIA] buyable rows here, cheapest first: ' + ', '.join([f'{i_n} ({p_n:g})' for p_n, i_n in lines[:6]])

    def clusters_for(self, open2: int) -> list[list[int]]:
        ordered = [n_n for n_n in self.scheme_sequence if n_n in self.tracks]
        ordered.extend([n_n for n_n in self.tracks if n_n not in self.scheme_sequence])
        nums = [n_n for n_n in ordered if self.tracks.get(n_n)]
        if open2 <= 0 or open2 >= len(nums):
            return []
        out_n: list[list[int]] = []
        start = 0
        for parts_left in range(open2, 0, -1):
            magnitude = -((start - len(nums)) // parts_left)
            out_n.append(nums[start:start + magnitude])
            start += magnitude
        return out_n

    def complete_basket(self) -> str:
        basket = self.duties.get('cart')
        purchase = self.duties.get('buy')
        if self.completed_n or not self.confirmed_n:
            return ''
        if self.vended_out(self.confirmed_n) or not (basket and purchase):
            return ''
        self.completed_n = True
        group_key = self.section_key()

        def sort2(iid: str) -> str:
            if group_key:
                return _g2_text_at(self.lines.get(iid) or {}, group_key, '').strip().lower()
            return iid
        scheme = self._g2_clue_scheme(sort2)
        if self.confirmed_n not in scheme:
            return ''
        taken = {sort2(p_n) for p_n in scheme}
        retain = {}
        discard2 = []
        for i_n in self.basket_iids():
            if i_n in scheme:
                continue
            if not self.vended_out(i_n) and sort2(i_n) not in taken:
                retain[i_n] = self.basket_rows.get(i_n, 1)
            else:
                discard2.append(i_n)
        if len(self.items) > len(scheme) + len({sort2(i_n) for i_n in retain}):
            return ''
        rows = dict(retain)
        rows.update(scheme)
        invocations = self.basket_invocations(rows, discard2)
        if not invocations or len(invocations) >= self.remaining():
            return ''
        room = len(invocations) + 3
        self.populate_details(list(rows), room)
        self.display_ties(list(rows), room)
        self.verify_basket(list(rows), SPEND_HARDN)
        if not self.verdicts_n.get(self.verify_key(list(rows))):
            return ''
        completed = self._g2_set_cart(basket, invocations)
        if completed is None:
            return ''
        args_n, issues = _g2_shape(dict(self.digest.get('buy_args') or {}), self.specs.get(purchase), 'args')
        if issues:
            return ''
        answers = self._g2_checkout(purchase, args_n, rows, discard2)
        if not answers:
            return ''
        return 'Not run (NVIDIA acted instead): the independent check confirmed ' + self.confirmed_n + ', so NVIDIA set the cart (' + '; '.join(completed) + ') and checked out. Store reply: ' + ' | then: '.join(answers)

    def conclude(self, again: bool=False) -> str:
        if self.closed or self.broke_n or self.remaining() <= 0:
            return ''
        purchase = self.duties.get('buy')
        decline = self.duties.get('refuse')
        novel = self._g2_pane_fresh()
        if novel:
            holding_n = bool(self.basket_iids())
        else:
            holding_n = bool(self.basket_rows)
        entire = self._g2_whole_cart() if holding_n and len(self.items) > 1 else True
        if self._g2_halted():
            return ''
        empty_ok = self.basket_touched and (not novel) and (self.remaining() >= 2)
        if entire and purchase and (holding_n or empty_ok):
            said = self._g2_forced_buy(purchase, again)
            if said is not None:
                return said
        if decline:
            said = self._g2_forced_refuse(decline)
            if said is not None:
                return said
        return ''

    def conclude_query(self, num: int, tags: list[str], displayed: bool, bundle: list[tuple[int, str, str]], heard: list[str]) -> None:
        self.track_used_up[num] = 1 + self.track_used_up.get(num, 0)
        self.list_asks_n += 1
        held = [len(ware2.heard) for ware2 in self.items]
        learnt = self.hear_n(bundle, (LIGHTN, HEAVYN))
        if len(tags) > 1:
            learnt.pop('free', None)
        self.grab_learnt(learnt, heard)
        fresh = []
        for n_n, ware2 in enumerate(self.items, 1):
            if n_n > len(held):
                break
            gained = ware2.heard[held[n_n - 1]:]
            fresh.extend(gained)
            if gained:
                self.track_located[n_n] = 1 + self.track_located.get(n_n, 0)
        packed = False
        for x_n in _wares(learnt.get('crowded')):
            if _tally(x_n) == 1:
                packed = True
                break
        self.settle_n(num, tags, displayed, fresh, packed)

    def contrast_details(self) -> bool | None:
        key = self.duties.get('compare') or ''
        return self.details_from.get(key)

    def contrast_goal(self) -> str | None:
        spec = self.specs.get(self.duties.get('compare') or '') or {}
        goal = _g2_first_key(spec.get('properties') or {}, 'array')
        if goal and all((k_n == goal for k_n in spec.get('required') or [])):
            return goal
        return None

    def contrast_plain(self) -> bool:
        seen = self.contrast_details()
        if seen is False:
            return True
        if seen is not None:
            return False
        if self.details_from.get(self.details_op() or '') is True:
            return True
        return bool(self.duties.get('offer'))

    def contrast_space(self) -> int:
        spec = self.specs.get(self.duties.get('compare') or '') or {}
        props = spec.get('properties') or {}
        schema = props.get(self.contrast_goal() or '') or {}
        cap = schema.get('maxItems')
        if isinstance(cap, int):
            return schema['maxItems']
        poor = self.contrast_poor
        if not poor:
            return max(4, self.contrast_fine)
        upper = min(poor - 1, max(self.contrast_fine, 2))
        return max(2, upper)

    def cost_of(self, iid: str) -> float | None:
        if not self.cost_key:
            return None
        val = (self.lines.get(iid) or {}).get(self.cost_key)
        if isinstance(val, dict):
            val = next((v_n for v_n in val.values() if _g2_is_num(v_n)), None)
        if isinstance(val, str):
            figure_n = re.fullmatch('\\s*[^\\d\\s]{0,3}\\s?(\\d[\\d,]*(?:\\.\\d+)?)\\s*[A-Za-z]{0,6}\\.?\\s*', val)
            if figure_n is None:
                return None
            val = float(figure_n.group(1).replace(',', ''))
        if _g2_is_num(val):
            return float(val)
        return None

    def cost_sorted(self, obs: Any) -> bool:
        costs = [self.cost_of(i_n) for i_n, __n in self.entries_in(obs)]
        if len(costs) < 3 or None in costs:
            return False
        for k_n in range(1, len(costs)):
            if not costs[k_n] >= costs[k_n - 1]:
                return False
        return True

    def deal_op(self) -> str | None:
        found = self.duties.get('offer')
        if not found:
            found = self.duties.get('open')
        return found

    def decline_path(self) -> str:
        if self.decline_trailed or PATH_SPACE > self.remaining() or self.timer() > CHASE_BY or (not self._g2_open_pointer()):
            return ''
        self.decline_trailed = True
        text_n, __n = self.path(CHASE_BY)
        if not text_n:
            return ''
        return text_n + 'Search for those values now and buy what fits; end without a purchase only if nothing does.'

    def demands_now(self) -> list[tuple[int, str]]:
        return [(num, demand) for num, ware2 in enumerate(self.items, 1) for demand in ware2.needs + ware2.heard if demand not in ware2.lifted]

    def described_n(self, num: int, demand: str) -> bool:
        if self._g2_vague_demand(demand):
            return False
        trait, __n, val = demand.partition(':')
        if not trait.strip():
            return False
        if 4 > len(re.findall('[a-z0-9]+', val.lower())):
            return False
        if re.search('\\b(?:listing|page|shown|other than|unlike|same as)\\b', val, re.I):
            return False
        ware2 = self.items[num - 1]
        val_root = _root(val)
        for other_n in self.items:
            if other_n is not ware2 and {w_n for w_n in _root(other_n.name_n) if len(w_n) >= 4} & val_root:
                return False
        group_key = self.section_key()
        keys: set[str] = set()
        scan = 0
        for entry in self.whole.values():
            if group_key and (not self.possesses(ware2, _g2_text_at(entry, group_key, ''))):
                continue
            chunk = _detail_chunk(entry) or {}
            if not chunk:
                continue
            scan += 1
            keys.update([_terms(k_n) for k_n in chunk])
        if scan < 3:
            return False
        trait_root = _root(trait)
        return not any((_root(k_n) & trait_root for k_n in keys))

    def details_op(self) -> str | None:
        for t_n, fine in self.details_from.items():
            if fine and self.lone_goal(t_n):
                if t_n:
                    return t_n
                break
        op = self.duties.get('open')
        if self.details_from.get(op or '') is False:
            return None
        if not self.lone_goal(op):
            return None
        return op

    def display(self, obs: Any) -> str:
        bound = DISPLAY_CHARS if self.entries_in(obs) else TEXT_CHARSN
        text_n = _serial(_trim(obs, bound))
        if len(text_n) <= OBS_BOUND:
            return text_n
        return text_n[:OBS_BOUND] + '...(cut)'

    def display_ties(self, iids: list[str], retain: int) -> None:
        scheme = self.val_pattern()
        if not scheme:
            return
        if len(iids) < 2 and len(self.items) < 2:
            return
        op, origin, slot = scheme
        for iid in iids:
            signs = self.match_position()['marks']
            tag_code = (signs.get(iid) or {}).get('more')
            if not tag_code or tag_code in self.tie_attempted:
                continue
            if any((sign.get('own') == tag_code for sign in signs.values())):
                continue
            if self.remaining() <= retain or self.closed or self.hurried():
                return
            self.tie_attempted.add(tag_code)
            if not self._g2_probe(op, {**origin, slot: tag_code}):
                return
            self._g2_log({'kind': 'show_link', 'line': iid, 'code': tag_code})
            owners = [i_n for i_n, sign in self.match_position()['marks'].items() if sign.get('own') == tag_code]
            self.populate_details(owners, max(retain, self.end2() + 6))

    def displayed_already(self, name_n: str, iid: str) -> bool:
        edition, has = self.pane_fields.get(iid, (-1, set()))
        keys = self.op_fields.get(name_n)
        if not keys or edition != self.edition:
            return False
        return has >= keys and iid not in self.vanished

    @staticmethod
    def doubles(name_n: str, schema: dict[str, Any], other_n: str, cell: dict[str, Any]) -> bool:
        a_n = set(name_n.split())
        b_n = set(other_n.split())
        if not (a_n and b_n and a_n & b_n):
            return False
        both_n = []
        for i_n in schema['plain']:
            if i_n in cell['plain'] and schema['plain'][i_n].strip() and cell['plain'][i_n].strip():
                both_n.append(i_n)
        same_n = 0
        for i_n in both_n:
            if _g2_agree(schema['plain'][i_n], cell['plain'][i_n]):
                same_n += 1
        if b_n >= a_n or a_n >= b_n:
            if both_n:
                return 2 * same_n > len(both_n)
            return bool((set(schema['values']) & set(cell['values'])) - {''})
        return len(both_n) >= 2 and len(both_n) == same_n

    def early_path(self) -> str:
        if PATH_SPACE > self.remaining() or self.timer() > CHASE_BY or (not self._g2_open_pointer()):
            return ''
        saved_trails, saved_chunks = (self.trails_n, self.path_chunks)
        self.trails_n = 0
        self.path_space = EARLY_TRAILSN
        try:
            text_n, __n = self.path(CHASE_BY)
        finally:
            self.trails_n = saved_trails
            self.path_chunks = saved_chunks
            self.path_space = None
        self._g2_log({'kind': 'early_trail', 'left': self.remaining(), 'values': list(self.pointer_vals.values())[:3]})
        return text_n

    def end2(self) -> int:
        for p_n in self.procedure:
            if p_n['when'] == 'order' and p_n['messages'] > 0:
                break
        else:
            return 1
        if self.text_spent < self.text_bound and self.barrier_due('order'):
            return 2
        return 1

    def engine_space(self) -> float:
        left = TASK_SECONDSN - MOMENT_MARGIN
        return left - self.timer()

    def entire_deletion(self, name_n: str, args_n: dict[str, Any]) -> dict[str, Any]:
        schema = self.specs.get(name_n, {}).get('properties') or {}
        count = _g2_first_key(schema, 'integer')
        deletion = False
        for k_n, v_n in args_n.items():
            if isinstance((schema.get(k_n) or {}).get('enum'), list) and v_n in schema[k_n]['enum'][1:]:
                deletion = True
                break
        iid = None
        for v_n in args_n.values():
            if isinstance(v_n, str) and v_n in self.basket_rows:
                iid = v_n
                break
        if not (count and deletion and iid) or count in args_n:
            return args_n
        have = self.basket_rows[iid]
        widened = dict(args_n)
        widened[count] = min(have, schema[count].get('maximum', have))
        return widened

    def entries_in(self, obs: Any) -> list[tuple[str, dict[str, Any]]]:
        if isinstance(obs, list):
            return self._g2_listed(obs)
        if not isinstance(obs, dict):
            return []
        own = self.iid(obs)
        if own:
            return [(own, obs)]
        located: list[tuple[str, dict[str, Any]]] = []
        for val in obs.values():
            if isinstance(val, list):
                located.extend(self._g2_listed(val))
                continue
            if not isinstance(val, dict):
                continue
            interior = self.iid(val, False, 1)
            if interior:
                located.append((interior, val))
                continue
            for v_n in val.values():
                if isinstance(v_n, list):
                    located.extend(self._g2_listed(v_n))
        return located

    def entries_of_ware(self, num: int) -> int:
        group_key = self.section_key()
        ware2 = self.items[num - 1]
        if not group_key:
            return 0
        found = 0
        for entry in self.whole.values():
            if self.possesses(ware2, str(entry.get(group_key) or '')) and _detail_chunk(entry):
                found += 1
        return found

    def entry_of(self, iid: str) -> str:
        entry = self.whole.get(iid) or self.lines.get(iid)
        if entry:
            return _serial(_trim(entry, 400))[:3000]
        return '(no record seen)'

    def evidence_n(self, obs: Any) -> str:
        iids = self._g3_record_ids(obs)
        demands = self.demands_now()
        if not (iids and demands):
            return ''
        several = len(self.items) > 1
        rows = []
        for iid in iids[:12]:
            chunk = _detail_chunk(self.whole.get(iid) or {}) or {}
            pieces = [p_n for p_n in (self._g3_evidence_piece(chunk, num, demand, several) for num, demand in demands[:8]) if p_n]
            if pieces:
                rows.append(f'{iid}: ' + '; '.join(pieces))
        return '; '.join(("[evidence: keys whose names contain a requirement's property", 'one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property] ')) + ' | '.join(rows) if rows else ''

    def exchange_hint(self) -> str:
        missed = []
        for i_n in self.basket_iids():
            if i_n in self.vanished:
                continue
            if self.vended_out(i_n):
                missed.append(i_n)
        if not missed:
            return ''
        self.vanished.update({iid: self.move for iid in missed})
        self.clues = list(filter(lambda c_n: c_n not in missed, self.clues))
        self.edition += 1
        choices = self.replacements_n(missed)
        text_n = f"[NVIDIA] Sold out now: {', '.join(missed)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        if not choices:
            return text_n + ' Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits.'
        if self.remaining() <= 6:
            return text_n + f" Few steps are left: in this turn add {choices[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(choices[1:]) or 'none'}."
        if self.digest.get('pick') == 'cheapest':
            return text_n + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(choices)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {choices[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        return text_n + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(choices)}. In this turn take the sold-out line out of the cart and add {choices[0]} at the requested quantity, then check out."

    def extra_lines_for_fields(self, op: str, args_n: dict[str, Any], viewed: Any, halt_at: int, num: int) -> list[str]:
        page = self.turning(op, viewed)
        blocked = not page or page[3] is None or self.lines_of_ware(num) >= OPEN_EACHN
        if blocked or self.closed or self.move >= halt_at:
            return []
        invoke, issues = self._g3_fit(self.specs.get(op), self._g3_with(args_n, page[2], page[3]), 'args')
        answer = None if issues else self.perform(op, invoke)
        if answer is None:
            return []
        self.perceive(op, answer.get('observation'))
        return [f'next page: ' + self.display(answer.get('observation'))]

    def extra_sheets(self, name_n: str, args_n: dict[str, Any], obs: Any) -> str:
        self.paginated_obs = []
        choice = self.digest.get('pick')
        if choice not in ('cheapest', 'most', 'least') or not (scheme := self.turning(name_n, obs)):
            return ''
        count, displayed, param_n, val = scheme
        given_n = args_n.get(param_n)
        if self._g3_int(given_n):
            later_n = given_n not in (None, 0, 1)
        else:
            later_n = bool(str(given_n or '').strip())
        if val is None or later_n or displayed >= count:
            return ''
        if choice == 'cheapest' and self.cost_sorted(obs):
            return ''
        sheets = -((displayed - count) // displayed)
        if sheets > 2 and self._g3_sheet_room(2):
            sheets = 2
        if sheets > 2 or not self._g3_sheet_room(sheets):
            return f"\n[NVIDIA] This search matched {count} listings and showed {displayed}, in no price order. The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with {param_n}={_serial(val)}."
        strings = []
        while len(strings) < sheets:
            invoke, issues = self._g3_fit(self.specs.get(name_n), self._g3_with(args_n, param_n, val), 'args')
            if issues:
                break
            answer = self.perform(name_n, invoke)
            if answer is None or self.closed:
                break
            self.sheets_scan += 1
            viewed = answer.get('observation')
            self.perceive(name_n, viewed)
            self.paginated_obs.append(viewed)
            displayed_text = self.display(viewed)
            self.scan_cache[name_n + _serial(invoke)] = (self.edition, self.move, displayed_text)
            strings.append(f'{param_n}={_serial(val)}: {displayed_text}')
            after_n = self.turning(name_n, viewed)
            val = after_n[3] if after_n else None
            if val is None:
                break
        self.journal.append({'kind': 'pages', 'tool': name_n, 'read': len(strings)})
        if not strings:
            return ''
        remainder = f"\n[NVIDIA] {count} listings matched and NVIDIA has read {displayed * (len(strings) + 1)} rows of them; narrow the search with the requirement's value or page on to see the rest." if val is not None else ''
        return '\n[NVIDIA read the next pages of this search itself, one step a page]\n' + '\n'.join(strings) + remainder

    @_mild('')
    def final_sheet_initial(self, name_n: str, args_n: dict[str, Any], obs: Any) -> str:
        if name_n in self.sheet_args or not isinstance(obs, dict):
            return ''
        traits = (self.specs.get(name_n) or {}).get('properties') or {}
        numeric_n = []
        for k_n, v_n in args_n.items():
            if not self._g3_int(v_n) or k_n not in traits:
                continue
            if v_n > (traits[k_n].get('minimum') if isinstance(traits[k_n].get('minimum'), int) else 1):
                numeric_n.append(k_n)
        if len(numeric_n) != 1 or self.remaining() <= self.end2() + 4:
            return ''
        form = self.scan_form(name_n, obs)
        if not self._g3_last_page(form, obs):
            return ''
        lines = obs.get(form['rows'])
        count = obs.get(form['count'])
        if not (isinstance(lines, list) and isinstance(count, int) and (count > len(lines))):
            return ''
        [key] = numeric_n
        small = traits[key].get('minimum') if isinstance(traits[key].get('minimum'), int) else 1
        invoke, issues = self._g3_fit(self.specs.get(name_n), self._g3_with(args_n, key, small), 'args')
        if issues:
            return ''
        answer = self.perform(name_n, invoke)
        if answer is None:
            return ''
        viewed = answer.get('observation')
        self.perceive(name_n, viewed)
        fresh = viewed.get(form['rows']) if isinstance(viewed, dict) else None
        if not isinstance(fresh, list) or len(fresh) < len(lines) or set((i_n for i_n, __n in self.entries_in(viewed))) & set((i_n for i_n, __n in self.entries_in(obs))):
            return ''
        self.sheet_args[name_n] = key
        self.journal.append({'kind': 'page_arg', 'tool': name_n, 'arg': key})
        return f'\n[NVIDIA] {key} is the page number of this search: the call above showed its last page. Page {small} of the same search:\n' + self.display(viewed)

    def find_tie(self, about_n: str, terms: str) -> list[str]:
        not_names_n = self.not_maker_fields()
        called = set()
        for iid in self.store_lines(about_n, list(self.lines)) + self.unfolded_store(about_n):
            for k_n, v_n in (self.lines.get(iid) or {}).items():
                if not isinstance(v_n, str) or len(v_n.strip()) > 40 or k_n in not_names_n:
                    continue
                if v_n.strip().lower() in NONE_NAME:
                    continue
                called.update(re.findall('[a-z0-9]{3,}', v_n.lower()))
        values = {w_n for w_n in re.findall('[a-z0-9]+', (terms or about_n).lower()) if w_n.isdigit() or len(w_n) >= 3}
        values = values - called - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for', 'of'}
        for iid in self.unfolded_store(about_n):
            if not values:
                continue
            text_n = self._g3_flat_text(self.whole.get(iid) or self.lines.get(iid) or {})
            if all((re.search('(?<![a-z0-9])' + re.escape(v_n) + '(?![a-z0-9])', text_n) for v_n in values)):
                self.journal.append({'kind': 'link_known', 'id': iid})
                return []
        scheme = self.val_pattern()
        jot = ' '.join(terms.lower().split())
        learned = []
        if jot and jot in self.tie_viewed:
            learned = self.store_lines(about_n, self.tie_viewed[jot])
        elif scheme and terms:
            learned = self._g3_tie_search(about_n, terms, jot, scheme)
        sort2 = set(re.findall('[a-z0-9]{3,}', (terms or about_n).lower())) - called - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for'}
        picked = self._g3_of_sort(learned, sort2)
        if not picked:
            picked = self._g3_of_sort(self.store_lines(about_n, list(reversed(self.viewed_sequence))), sort2)
        return picked[:2]

    def find_vended(self) -> str:
        iids = list(filter(lambda i_n: i_n not in self.vanished, self.basket_iids()))
        if any(map(self.vended_out, iids)):
            return ''
        if not iids:
            return ''
        if len(iids) > 1:
            return self.rescan_offers()
        self.unavailable_n.add(iids[0])
        return ''

    def fix_tallies(self, wrong_n: list[tuple[str, int, int]]) -> list[str]:
        name_n = self.duties.get('cart') or ''
        spec = self.specs.get(name_n) or {}
        traits = spec.get('properties') or {}
        strings = []
        count = manner = None
        for k_n, v_n in traits.items():
            if not isinstance(v_n, dict):
                continue
            if 'string' in _sorts(v_n) and 'enum' not in v_n:
                strings.append(k_n)
        goal = None
        for t_n, k_n in self.id_args_n:
            if t_n == name_n and k_n in strings:
                goal = k_n
                break
        if not goal:
            goal = next(filter(lambda k_n: k_n in strings, spec.get('required') or []), None)
        if not goal:
            goal = strings[0] if len(strings) == 1 else None
        for k_n, v_n in traits.items():
            if isinstance(v_n, dict) and 'integer' in _sorts(v_n):
                count = k_n
                break
        for k_n, v_n in traits.items():
            if isinstance(v_n, dict) and isinstance(v_n.get('enum'), list) and (len(v_n['enum']) >= 2):
                manner = k_n
                break
        if not (name_n and goal and count):
            return []
        invocations = []
        for iid, has, desire in wrong_n:
            if not self.basket_collections(name_n):
                if not manner:
                    return []
                invocations.extend(self._g3_swap_calls(goal, manner, count, traits[manner]['enum'], iid, has, desire))
                continue
            invocations.append({goal: iid, count: desire})
        if self.remaining() < len(invocations) + 3:
            return []
        completed = []
        for args_n in invocations:
            args_n, issues = self._g3_fit(spec, args_n, 'args')
            if issues:
                break
            answer = self.perform(name_n, args_n)
            if answer is None or self.closed:
                break
            self.memo_basket(name_n, args_n)
            seen_n = answer.get('observation')
            self.basket_stated = _serial(seen_n)
            self.perceive(name_n, answer.get('observation'))
            completed.append(_serial(args_n) + ' -> ' + self.display(answer.get('observation'))[:300])
        self.journal.append({'kind': 'count_fix', 'calls': completed})
        return completed

    def following_query(self, tracks: dict[int, list[str]], located: dict[int, int], used_up: dict[int, int], spent: dict[int, set[str]]) -> tuple[int, list[str]]:
        rank = {}
        for i_n, n_n in enumerate(self.scheme_sequence):
            rank[n_n] = i_n
        tail = len(rank)
        flat = self.text_bound <= SCARCE_MESSAGESN and (not self.offer)

        def order(n_n: int) -> tuple[bool, int, int]:
            return (located.get(n_n, 0) > 0, 0 if flat else used_up.get(n_n, 0), rank.get(n_n, tail + n_n))
        for num in sorted(tracks, key=order):
            open_tags = [tag for tag in tracks[num] if not (_terms(tag) in spent[num] or self.heard_about(num, tag))]
            if not open_tags:
                continue
            width = self.list_now_n() if self.offer else 1
            return (num, open_tags[:width])
        return (0, [])

    def fuller_tag(self, num: int, tag: str) -> str:
        group_key = self.section_key()
        ware2 = self.items[num - 1]
        ours = _bases(ware2.name_n)
        theirs = _bases(tag)
        if len(ours) < 2 or ours <= theirs or (not group_key) or (ware2.aisle and theirs == _bases(ware2.aisle)):
            return ''
        for viewed in sorted(set((str(r_n.get(group_key) or '') for r_n in self.lines.values()))):
            if viewed and ours <= _bases(viewed):
                return viewed
        return ''

    def grab_learnt(self, learnt: dict[str, Any], heard: list[str]) -> None:
        several = len(self.items) > 1
        for record in _wares(learnt.get('needs')):
            if not (isinstance(record, dict) and str(record.get('need') or '').strip()):
                continue
            self._g3_learn_need(record, several, heard)
        for record in _wares(learnt.get('drop')):
            if not (isinstance(record, dict) and str(record.get('property') or '').strip()):
                continue
            ware2 = self.items[self._g3_slot(record.get('item')) - 1]
            vanished = _terms(str(record['property']).split(':')[0])
            strike = list(filter(lambda m_n: m_n not in ware2.lifted and _terms(m_n.split(':')[0]) == vanished, ware2.needs + ware2.heard))
            ware2.lifted.extend(strike)
            heard.extend([f'dropped {m_n}' for m_n in strike])
        for record in _wares(learnt.get('free')):
            if isinstance(record, dict) and record.get('property'):
                self.items[self._g3_slot(record.get('item')) - 1].open2.append(str(record['property']))
        if not isinstance(learnt.get('budget'), bool) and isinstance(learnt.get('budget'), (int, float)) and (learnt['budget'] > 0):
            self.allowance = learnt['budget']

    def grab_memo(self, rows: list[str]) -> str:
        self.learnt.update(rows)
        head = MEMO + '\n\nITEMS AND REQUIREMENTS NOW:\n' + self.register()
        obtained = self.query_json(head + '\n\nTHE NOTE:\n' + '\n'.join(rows), [LIGHTN, SPAREN], 2500) or {}
        vanished = list(map(str, filter(lambda g_n: isinstance(g_n, (str, int)), _wares(obtained.get('gone')))))
        self.unavailable_n.update([g_n for g_n in vanished if g_n in self.lines])
        self.memo_vended = True if vanished else obtained.get('sold_out') is True
        self.memo_called = bool(set(vanished) & self.lines.keys())
        if len(rows) == 1 and self.memo_vended:
            self.vended_rows.update(rows)
        if obtained.get('kind') not in (None, 'change'):
            return ''
        altered = []
        allowance = obtained.get('budget')
        if self._g3_num(allowance) and allowance > 0 and (allowance != self.allowance):
            self.allowance = allowance
            altered.append(f'budget now {allowance}')
        drops_n = list(filter(lambda e_n: isinstance(e_n, dict) and str(e_n.get('need') or '').strip(), _wares(obtained.get('drop'))))
        adds_n = list(filter(lambda e_n: isinstance(e_n, dict) and str(e_n.get('need') or '').strip(), _wares(obtained.get('add'))))
        altered += [f'dropped {self._g3_memo_drop(record)}' for record in drops_n]
        altered += [f'added {self._g3_memo_add(record)}' for record in adds_n]
        if vanished:
            altered.append('unavailable: ' + ', '.join(vanished))
        if obtained.get('price_moved') is True:
            altered.append('a price changed')
        recap = str(obtained.get('summary') or '').strip()
        widened = drops_n or vanished or self.memo_vended or any((c_n.startswith('budget now') for c_n in altered))
        if widened and self.digest.get('pick') not in ('cheapest', 'most', 'least'):
            self.digest['pick'] = 'cheapest'
            self.journal.append({'kind': 'p2_cheapest', 'why': '; '.join(altered)[:160]})
        if self.digest.get('pick') == 'cheapest':
            if drops_n or vanished or self.memo_vended or any((c_n.startswith('budget now') for c_n in altered)):
                self.broader_kept = False
                self.lower_priced_signs = set()
        if altered:
            self.p1_changed = True
        if recap or altered:
            self.edition += 1
            self.recent_n = []
            self.memos.append(recap or '; '.join(altered))
        tail_n = ' (' + '; '.join(altered) + ')' if altered else ''
        return (recap + tail_n).strip()

    def heading_key(self) -> str | None:
        group_key = self.section_key()
        hits, words, chars, kinds = ({}, {}, {}, {})
        for line in self.lines.values():
            for key, val in line.items():
                if key == group_key or not isinstance(val, str) or len(val) > HEADING_CHARS:
                    continue
                hits[key] = hits.get(key, 0) + 1
                words[key] = words.get(key, 0) + len(val.split())
                chars[key] = chars.get(key, 0) + len(val)
                kinds.setdefault(key, set()).add(val)
        floor_n = len(self.lines) * 4
        retained = []
        for key, seen_n in hits.items():
            spread = len(kinds[key])
            if seen_n * 5 >= floor_n and words[key] >= seen_n * 2 and (spread * 2 >= seen_n):
                retained.append((chars[key] / seen_n, key, spread))
        most_n = max([row[2] for row in retained], default=0)
        graded = [(row[0], row[1]) for row in retained if row[2] * 10 >= most_n * 9]
        if graded:
            return min(graded)[1]
        return None

    def hear_n(self, dispatched: list[tuple[int, str, str]], models_n: tuple[str, ...]=(HEAVYN, LIGHTN)) -> dict[str, Any]:
        if self.quiet_answer(dispatched):
            self.journal.append({'kind': 'quiet_reply', 'a': dispatched[0][2][:80]})
            return {}
        self.hear_reads_n += 1
        listed = []
        for n_n, it_n in enumerate(self.items, 1):
            listed.append(f'item {n_n}: {it_n.name_n}')
        asked = []
        for i_n, (n_n, q_n, a_n) in enumerate(dispatched, 1):
            about = f'about item {n_n}' if n_n else 'about several items'
            asked.append(f'{i_n}. ({about}) Q: {q_n}\nA: {a_n}')
        instruction = HEARN + '\n\nITEMS:\n' + '\n'.join(listed) + '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join(asked)
        return self.query_json(instruction, list(models_n), 2500) or {}

    def heard_about(self, num: int, tag: str) -> bool:
        ours = _written(tag)
        for t_n in self.items[num - 1].heard:
            if _names_n(ours, _written(t_n.split(':', 1)[0])):
                return True
        return False

    def holder_of(self, tag: str) -> int:
        if not tag:
            return 0
        if tag in self.adopted_n:
            return self.adopted_n[tag]
        memo_key = (tag, len(self.items))
        learned = self.holders_viewed.get(memo_key)
        if learned is not None:
            return learned
        terms, lengthy = (_terms(tag), _bases(tag))
        scores = [self._g3_holder_score(terms, lengthy, ware2) for ware2 in self.items]
        peak = max(scores, default=0.0)
        top = scores.index(peak) + 1 if peak > 0.0 else 0
        self.holders_viewed[memo_key] = top
        return top

    def home_tag(self, num: int) -> str:
        group_key = self.section_key()
        if not group_key:
            return ''
        ware2 = self.items[num - 1]
        for reserve in (self.whole, self.lines):
            tallies = self._g3_tag_counts(reserve, group_key, ware2)
            hit = next((_terms(name_n) for name_n in (ware2.aisle, ware2.name_n) if _terms(name_n) in tallies), None)
            if hit is not None:
                return hit
            if tallies:
                return max(tallies, key=tallies.get)
        return ''

    def hurried(self) -> bool:
        pace_n = self.world_used_up / self.world_tally if self.world_tally else 1.0
        return 2 * (pace_n if pace_n > 1.0 else 1.0) >= self.engine_space()

    def iid(self, leaf: Any, in_list_n: bool=False, level: int=0) -> str | None:
        if isinstance(leaf, dict) and len(leaf) >= 3:
            known = self.id_key_n
            if known and _id_akin(leaf.get(known), True):
                return leaf[known].strip()
            if in_list_n or level == 0 or any(map(lambda v_n: isinstance(v_n, dict), leaf.values())):
                located = _learn_id_n([leaf])
                if located:
                    found = leaf[located].strip()
                    if 'id' in str(located).lower() or found in self.lines:
                        return found
        return None

    def initial_look(self, name_n: str, args_n: dict[str, Any], novel: bool) -> dict[str, Any]:
        key = self.sheet_args.get(name_n)
        if key and novel and isinstance(args_n.get(key), int):
            traits = (self.specs.get(name_n) or {}).get('properties') or {}
            small = traits.get(key, {}).get('minimum') if isinstance(traits.get(key, {}).get('minimum'), int) else 1
            return self._g3_with(args_n, key, small)
        return args_n

    @_mild('')
    def initial_moves(self) -> str:
        scheme = self.val_pattern()
        if not scheme or self.max_moves < OPEN_MIN_MOVES or len(self.items) < OPEN_WARES:
            return ''
        op, origin, slot = scheme
        self.world_margin = OPEN_MARGIN
        displayed = []
        spare = self.remaining() - self.end2() - self.texts_open()
        halt_at = self.move + spare - OPEN_LATERN * len(self.items) - CHASE_RETAIN // 2
        for num, ware2 in enumerate(self.items, 1):
            if not self._g3_open_item(op, origin, slot, ware2, num, halt_at, displayed):
                return '\n'.join(displayed)
        if self.lines and self.duties.get('ask') and (self.texts_open() > 0):
            self._g3_open_widen(halt_at)
        self.world_margin = WORLD_MARGIN
        self.journal.append({'kind': 'first_moves', 'rows': len(self.lines), 'opened': len(self.whole)})
        for iid, entry in self.whole.items():
            displayed.append('read in full: ' + self.sheet_text(iid, entry))
        evidence = self.evidence_n(list(self.whole.values()))
        if evidence:
            displayed.append(evidence)
        return '\n'.join(displayed)

    def inquiry(self, num: int, tag: str, samples_n: list[str] | None=None) -> str:
        text_n = f'What {tag} would you like for the {self.ware_stated(num)}'
        if samples_n:
            return text_n + f', for example {samples_n[0]} or {samples_n[1]}?'
        return text_n + '?'

    def interview_n(self) -> str:
        self.queried = True
        if self.texts_open() <= 0:
            return ''
        self.widen_n()
        count = self.text_bound - self.text_spent
        if self.offer:
            ceiling_n = self.list_ceiling()
            count = ceiling_n if ceiling_n > count else count
        self.tracks = self.scheme_tracks(count)
        self.track_located = dict.fromkeys(self.tracks, 0)
        self.track_used_up = dict.fromkeys(self.tracks, 0)
        self.track_spent = {n_n: set() for n_n in self.tracks}
        self.trail_upward = []
        dispatched = []
        heard = []
        groups = self.clusters_for(self.texts_open()) if self.offer else []
        while self._g3_may_ask():
            bundle = self._g3_next_bundle(groups, heard)
            if not bundle:
                break
            dispatched += bundle
        located_wares = [n_n for n_n, hits_n in self.track_located.items() if hits_n > 0]
        series = self.early_path()
        looked = self.val_lookup(located_wares) if located_wares else ''
        if heard:
            self.edition += 1
        recap = "NVIDIA's interviewer asked the shopper (each question used one step):\n" + '\n'.join((f'Q: {q_n}\nA: {a_n}' for __n, q_n, a_n in dispatched))
        recap += '\nNew firm requirements from these answers: ' + ('; '.join(heard) if heard else 'none')
        recap += '. Apply them to every pick (search again with the new values where needed).'
        aimed = [demand for __n, demand in self.pointers if any(map(lambda t_n: demand in t_n, heard))]
        if aimed:
            recap += ' These take their value from another listing: ' + ' | '.join(aimed) + '. '.join(('', 'Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.'))
        if series:
            recap += '\n' + series
        if looked:
            recap += '\n' + looked
        if self.pending:
            recap += '\n[notes seen meanwhile] ' + ' | '.join(self.pending)
            self.pending = []
        return recap

    def invoke_engine(self, engine: str, messages_n: list[dict[str, Any]], max_pieces: int, ops: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        heavy = len(messages_n) > 1 and engine.startswith('anthropic/')
        core = {'model': engine}
        core['messages'] = self._g3_cache_marks(messages_n) if heavy else messages_n
        core['max_tokens'] = max_pieces
        core['temperature'] = 0
        if ops:
            core['tools'], core['tool_choice'] = (ops, 'auto')
        outcome = _lm_within(self.engine_space()).post_verbose('/inference/chat/completions', core)
        payload = outcome.data
        framed = isinstance(payload, dict)
        if framed and payload.get('choices'):
            return self._g3_engine_ok(engine, payload)
        failure = outcome.error or {}
        state = failure.get('status')
        if framed and isinstance(payload.get('error'), dict):
            state = _tally(payload['error'].get('code'))
            failure = {'body': payload['error'].get('message')}
        detail = str(failure.get('body') or '')
        self.journal.append({'kind': 'llm_fail', 'model': engine, 'status': state, 'body': detail[:200]})
        return (None, state)

    def invoke_sign(self, name_n: str, args_n: dict[str, Any]) -> str:
        traits = (self.specs.get(name_n) or {}).get('properties') or {}
        retained = {}
        for key, val in sorted(args_n.items()):
            schema = traits.get(key) if isinstance(traits.get(key), dict) else {}
            if self._g3_int(val) and self._g3_plain_default(name_n, key, val, schema):
                continue
            retained[key] = ' '.join(val.lower().split()) if isinstance(val, str) else val
        return name_n + _serial(retained)

    def is_decline(self, name_n: str, args_n: dict[str, Any]) -> bool:
        if self.duties.get('refuse') == name_n:
            return self._g3_holds(args_n, self.digest.get('refuse_args') if isinstance(self.digest.get('refuse_args'), dict) else {})
        return False

    def is_purchase(self, name_n: str, args_n: dict[str, Any]) -> bool:
        if self.duties.get('buy') == name_n:
            return self._g3_holds(args_n, self.digest.get('buy_args') if isinstance(self.digest.get('buy_args'), dict) else {})
        return False

    def join_opens_n(self, invocations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        opener_n = self.duties.get('open')
        joiner_n = self.duties.get('compare')
        goal = self.lone_goal(self.duties.get('open'))
        several = self.contrast_goal()
        if not (opener_n and joiner_n and goal and several) or opener_n == joiner_n or self.contrast_plain():
            return invocations
        choices = []
        for index, invoke in enumerate(invocations):
            target = self._g3_open_target(invoke, opener_n, goal)
            if target is not None:
                choices.append((index, target))
        choices = choices[:self.contrast_space()]
        if len(choices) < 2:
            return invocations
        joined_ids = [i_n for __n, i_n in choices]
        merged_at = {i_n for i_n, __n in choices}
        first = choices[0][0]
        out_n = []
        for index, invoke in enumerate(invocations):
            if index == first:
                out_n.append({'id': invoke['id'], 'type': 'function', 'function': {'name': joiner_n, 'arguments': _serial({several: list(joined_ids)})}})
            elif index in merged_at:
                out_n.append({'id': invoke['id'], 'joined': True})
            else:
                out_n.append(invoke)
        self.journal.append({'kind': 'joined', 'ids': joined_ids})
        return out_n

    def kept_measures(self, iid: str) -> int:
        if self.basket_pane is None or self.basket_at > self.basket_viewed_at:
            return self.basket_rows.get(iid, 1)
        found = self._g3_pane_count(iid)
        return self.basket_rows.get(iid, 1) if found is None else found

    def key_grid(self) -> list[tuple[str, str, int, list[str]]]:
        group_key = self.section_key()
        groups = self._g3_grid_collect(group_key)
        desired = []
        for ware2 in self.items:
            lengthy = {w_n for w_n in _root(ware2.aisle or '') | _root(ware2.name_n) if len(w_n) >= 4}
            desired.append((_terms(ware2.aisle or ''), _terms(ware2.name_n), lengthy))
        matched_n = []
        for tag in groups:
            near = _terms(tag)
            long_n = {w_n for w_n in _root(tag) if len(w_n) >= 4}
            if any((near in (aisle, name_n) or long_n & lengthy for aisle, name_n, lengthy in desired)) or self.holder_of(tag):
                matched_n.append(tag)
        if matched_n:
            groups = {tag: groups[tag] for tag in matched_n}
        grid = []
        for tag, group in sorted(groups.items(), key=lambda kv_n: kv_n[1]['n'], reverse=True):
            for name_n, cell in self._g3_grid_fields(group)[:GRID_FIELDS]:
                grid.append((tag, name_n, cell['n'], [v_n[:40] for v_n in cell['values'][:4]]))
        return grid

    def learn_lines(self, leaf: list[Any]) -> bool:
        lines = list(filter(lambda n_n: isinstance(n_n, dict) and len(n_n) >= 3, leaf))
        if lines and len(lines) * 2 >= len(leaf):
            known = self.id_key_n
            if known and all(map(lambda r_n: _id_akin(r_n.get(known), True), lines)):
                return True
            located = _learn_id_n(lines)
            if located:
                self.id_key_n = located
                return True
        return False

    def line_iids(self, name_n: str, args_n: dict[str, Any]) -> dict[str, Any]:
        if all((self.duties.get(job_n) != name_n for job_n in ('open', 'offer', 'compare', 'cart'))) or not self.lines:
            return args_n
        title = self.heading_key()
        if not title:
            return args_n
        traits = (self.specs.get(name_n) or {}).get('properties') or {}
        out_n = {}
        for key, val in args_n.items():
            if isinstance(val, list):
                out_n[key] = [self._g3_row_id(v_n, key, title, traits) for v_n in val]
            else:
                out_n[key] = self._g3_row_id(val, key, title, traits)
        return out_n

    def lines_of_ware(self, num: int) -> int:
        group_key = self.section_key()
        ware2 = self.items[num - 1]
        if not group_key:
            return len(self.lines)
        return len([line for line in self.lines.values() if self.possesses(ware2, str(line.get(group_key) or ''))])

    def list_below_n(self, basket: list[str]) -> None:
        scheme = self.ceiling_pattern()
        group_key = self.section_key()
        if not (scheme and group_key):
            return
        op, origin, aisle, ceiling = scheme
        for row in basket:
            cost = self.cost_of(row)
            tag = str((self.lines.get(row) or {}).get(group_key) or '')
            if cost is None or cost < 2 or (not tag) or ((row, cost) in self.capped_n):
                continue
            self.capped_n.add((row, cost))
            if self._g3_stroll(op, origin, aisle, ceiling, row, cost, tag):
                return

    def list_ceiling(self) -> int:
        return LIST_MAGNITUDE if len(self.items) == 1 else LIST_MIXEDN

    def list_now_n(self) -> int:
        """Labels the next list names. On one item the first questions name few (nothing shows yet how this shopper
    takes a list) unless the request allows SCARCE_MESSAGESN messages or fewer: then every question is a full step (S3L)."""
        scarce = self.text_bound <= SCARCE_MESSAGESN
        moves = (3, 6) if scarce and len(self.items) == 1 else LIST_MOVES if len(self.items) == 1 and (not scarce) else LIST_MOVES_MIXED
        if self.list_asks_n >= len(moves):
            return self.list_ceiling()
        return min(moves[self.list_asks_n], self.list_ceiling())

    def lists_things_n(self, op: str) -> bool:
        goal = _snaky(self.lone_goal(op) or '')
        if re.search(AISLE_ARG, goal, re.I) or re.search(PHRASE_TERMS, goal, re.I):
            return True
        return op in self.shapes_n

    def lone_goal(self, op: str | None) -> str | None:
        spec = self.specs.get(op or '') or {}
        traits = spec.get('properties') or {}
        strings = []
        for k_n, v_n in traits.items():
            if isinstance(v_n, dict) and 'enum' not in v_n and ('string' in _sorts(v_n)):
                strings.append(k_n)
        goal = None
        for k_n in spec.get('required') or []:
            if k_n in strings:
                goal = k_n
                break
        if not goal and len(strings) == 1:
            goal = strings[0]
        if op and goal and all((k_n == goal for k_n in spec.get('required') or [])):
            return goal
        return None

    def look_alikes_n(self, num: int, tag: str) -> list[str]:
        ours = _terms(tag)
        viewed = {}
        for iid in self.ware_iids(num):
            chunk = _detail_chunk(self.whole.get(iid) or {}) or {}
            for key, val in chunk.items():
                viewed.setdefault(_terms(key), set()).update([w_n for w_n in _bases(str(val)) if len(w_n) > 1])
        bare = set()
        for w_n in ours.split():
            if len([n_n for n_n in viewed if w_n in n_n.split()]) <= 2:
                bare.add(w_n)
        mine = viewed.get(ours, set())
        scored_n = []
        for sequence, (name_n, values) in enumerate(viewed.items()):
            if not name_n or name_n == ours:
                continue
            shared_n = bare & set(name_n.split())
            common_n = mine & values
            if not (shared_n or common_n):
                continue
            capped = len(common_n) if len(common_n) < 3 else 3
            scored_n.append((-(3 * len(shared_n) + capped), sequence, name_n))
        scored_n.sort()
        return [row[2] for row in scored_n[:LOOK_ALIKESN]]

    def lookup_pattern(self):
        op = self.duties.get('search')
        spec = self.specs.get(op or '') or {}
        traits = spec.get('properties') or {}
        textual = []
        for key, schema in traits.items():
            if isinstance(schema, dict) and 'string' in _sorts(schema) and ('enum' not in schema):
                textual.append(key)
        phrased = [key for key in textual if re.search(PHRASE_TERMS, _snaky(key), re.I)]
        candidates = phrased or textual
        labels = None
        for sent in self.looked[::-1]:
            usable = [key for key in candidates if self._g4_worded(sent, key)]
            if not phrased:
                if labels is None:
                    labels = self._g4_section_labels()
                usable = [key for key in usable if self._g4_free_slot(key, labels)]
            if not usable or not op:
                continue
            return (op, self._g4_kept_args(sent, usable[0], traits, spec), usable[0])
        return None

    def lower_priced_lines(self):
        if self.digest.get('pick') != 'cheapest':
            return []
        sign = self.basket_sign() + f'#{self.edition}'
        group_key = self.section_key()
        if not group_key or sign in self.lower_priced_signs:
            return []
        self.lower_priced_signs.add(sign)
        basket = self.basket_iids()
        self.list_below_n(basket)
        span = self.contrast_space() if self.contrast_details() is True else 0
        picks = []
        for row in basket:
            price = self.cost_of(row)
            label = str((self.lines.get(row) or {}).get(group_key) or '').strip().lower()
            if price is not None and label:
                cheaper = self._g4_cheaper(basket, group_key, label, price)
                picks.extend([iid for iid in cheaper[:span or 2] if iid not in picks])
        if span:
            cap = span * len(basket)
        else:
            cap = min(6, LOWER_PRICED_OPENS * max(1, len(basket)))
        return self.populate_details(picks[:cap], self.end2() + 6)

    def match_issues(self):
        if not self.limited:
            return []
        index = self.match_position()
        basket = self.basket_iids()
        found = []
        for piece, primary in self.limited:
            found.extend(self._g4_pair_issues(piece, primary, basket, index))
        return found

    def match_position(self):
        state_n = (len(self.viewed_sequence), len(self.whole), sum((len(v_n) for v_n in self.panes.values())))
        cached = self.match_stored
        if cached and cached[0] == state_n:
            return cached[1]
        marks = {}
        notes = {}
        for iid in self.viewed_sequence:
            sign = _match_signs(self._g4_flat_text(self.whole.get(iid) or self.lines.get(iid) or {}))
            marks[iid] = sign
            for piece, codes in sign['also']:
                if piece not in notes:
                    notes[piece] = set()
                notes[piece].update(codes)
        result = {'marks': marks, 'notes': notes}
        self.match_stored = (state_n, result)
        return result

    def match_query(self, num, tags, displayed):
        size = len(tags)
        while size > 0:
            text_n = self.query_text(num, tags[:size], displayed)
            if self._g4_note_fits(text_n):
                return (tags[:size], text_n)
            size -= 1
        return ([], '')

    def match_scheme(self, short=False):
        if not self.limited:
            return ''
        index = self.match_position()
        rows = []
        for primary in sorted({m_n for __n, m_n in self.limited}):
            rows.extend(self._g4_scheme_rows(primary, index))
        if not rows:
            return ''
        if short:
            return ' | '.join(rows)
        return '; '.join(("[fit plan: a device fits a part when the part's text names the device's model code or another name it is sold as, or when the device's text names the series of the part's code", "try one search of a code or series with the item's words, and if it finds nothing list the device's category and read the codes on its rows] ")) + ' | '.join(rows)

    def match_set(self, iid, index):
        sign = index['marks'].get(iid) or {}
        found = set(sign.get('works') or [])
        if not sign.get('own'):
            return found
        found.update(index['notes'].get(sign['own'], set()))
        return found

    def memo_basket(self, name_n, args_n):
        self.basket_touched = True
        self.basket_at = self.move
        schema = self.specs.get(name_n, {}).get('properties') or {}
        removing = self._g4_removes(schema, args_n)
        rows, entire = self._g4_cart_rows(schema, args_n)
        if entire:
            self.basket_rows = {iid: count for iid, count in rows if count and count > 0}
            return
        collections = self.basket_collections(name_n)
        book = self.basket_rows
        for iid, count in rows:
            has = self._g4_new_count(book.get(iid, 0), count, removing, collections, schema)
            if has <= 0:
                book.pop(iid, None)
                continue
            book[iid] = has

    def memo_details(self, op, displayed):
        if displayed:
            self.details_from[op] = True
            return
        if self.details_from.get(op) is True:
            return
        seen = self.plain_sheets.get(op, 0) + 1
        self.plain_sheets[op] = seen
        if seen >= 2:
            self.details_from[op] = False

    def memo_pane(self, name_n, obs):
        for iid, entry in self.entries_in(obs):
            keys = set(entry)
            self.op_fields.setdefault(name_n, set()).update(keys)
            prior = self.pane_fields.get(iid)
            kept = prior[1] if prior is not None and prior[0] == self.edition else set()
            self.pane_fields[iid] = (self.edition, kept | keys)

    def memo_scan(self, name_n, args_n, viewed):
        if self._g4_unscanned(name_n):
            return
        self.phrase_is_fresh(name_n, args_n)
        if not isinstance(viewed, (dict, list)) or _failed_n(viewed):
            return
        sign = self.invoke_sign(name_n, args_n)
        self.scan_invocations[sign] = viewed
        self.scan_args[sign] = (name_n, dict(args_n))

    def name_pieces(self, num):
        name_n = self.items[num - 1].name_n
        if not re.search('\\band\\b', name_n, re.I):
            return []
        pieces = []
        for chunk in re.split('\\s*(?:,|\\band\\b)\\s*', name_n, flags=re.I):
            if chunk.strip():
                pieces.append(chunk.strip())
        if len(pieces) < 2:
            return []
        for chunk in pieces:
            if not re.search('[a-z]{3}(?<![su])s$', chunk, re.I):
                return []
        return pieces

    @_mild(False)
    def names_tag_code(self, args_n):
        stated = set()
        for val in args_n.values():
            if isinstance(val, str):
                stated.update(_tag_codes(val))
        if not stated:
            return False
        for sign in self.match_position()['marks'].values():
            if stated & (set(sign.get('codes') or []) | set(sign.get('works') or [])):
                return True
        return False

    def not_maker_fields(self):
        fields = [self.id_key_n, self.section_key()]
        fields.append(self.heading_key())
        return tuple(fields)

    def not_stated_n(self, tag, name_n, values):
        num = self.holder_of(tag)
        if not num:
            return ''
        shown = None
        for need in self.items[num - 1].needs:
            trait, __n, val = need.partition(':')
            if re.search('\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', val, re.I):
                continue
            terms = set(re.findall('[a-z0-9]{3,}', val.lower()))
            if shown is None:
                shown = set(re.findall('[a-z0-9]{3,}', ' '.join(values).lower()))
            if not trait.strip() or _terms(trait) == name_n or (not _root(trait) & _root(name_n)):
                continue
            if terms and (not shown >= terms):
                return f' [not the stated {trait.strip()}: another property]'
        return ''

    def note_args(self, text_n):
        name_n = self.duties.get('ask')
        spec = self.specs.get(name_n or '') or {}
        traits = spec.get('properties') or {}
        required_n = spec.get('required') or []
        strings = []
        for key, schema in traits.items():
            if isinstance(schema, dict) and 'string' in _sorts(schema) and ('enum' not in schema) and ('const' not in schema):
                strings.append(key)
        reserve = [key for key in required_n if key in strings] or strings
        args_n = self._g4_note_slot(reserve, traits, text_n) if reserve else self._g4_note_nested(traits, text_n)
        if not args_n:
            return None
        for key in required_n:
            if key in args_n:
                continue
            filled = _filler_n(traits.get(key) or {})
            args_n[key] = filled
            if filled is None:
                return None
        issues = []
        args_n = _form(args_n, spec, 'args', issues)
        return None if issues else args_n

    def offered_n(self):
        remaining = self.remaining()
        query = self.duties.get('ask')
        ends_n = {self.duties.get('buy'), self.duties.get('refuse')} - {None}
        allowed = self._g4_offer_scope(remaining, ends_n)
        blocked = None
        ops = []
        for tool in self.ops:
            if tool['function']['name'] == query:
                if blocked is None:
                    blocked = self.note_args('x' * 200) is not None
                if blocked:
                    continue
            ops.append(tool)
        if allowed is None or not ends_n:
            return ops
        return [tool for tool in ops if tool['function']['name'] in allowed] or ops

    def open_lines(self, lines):
        scheme = self._g4_open_plan(lines)
        if scheme is None:
            return False
        moves = self.basket_invocations({f'#{n_n}': 1 for n_n, __n in enumerate(self.items)}, [])
        if self.remaining() < len(scheme) + max(5, self.end2() + len(moves or self.items)):
            return False
        before = len(self.whole)
        if not self._g4_walk_plan(scheme):
            return False
        self.journal.append({'kind': 'audit_open', 'rows': lines, 'opened': len(self.whole) - before})
        return len(self.whole) > before

    def path(self, halt=None):
        found_vals = self.pointer_vals
        active = set()
        for __n, demand in self.demands_now():
            active.add(demand)
        waiting = []
        for __n, demand in self.pointers:
            if demand in active and demand not in found_vals:
                waiting.append(demand)
        duties = self.duties
        op = duties.get('offer') or duties.get('open')
        goal = self.lone_goal(op)
        ready = bool(op and goal)
        viewed, missed = ([], [])
        while waiting and self._g4_trail_budget(halt):
            self.trails_n += 1
            reach_n, missed, gaps_n = self._g4_trail_round(waiting)
            waiting = [p_n for p_n in waiting if p_n not in found_vals]
            if ready and gaps_n and (not reach_n) and (self.remaining() > 12) and (self.tie_searches < 3):
                reach_n = self.find_tie(*gaps_n[0])
                log = self.journal
                if log and log[-1].get('kind') == 'link_known':
                    viewed.append(f"{log[-1]['id']}: an opened page of the shop this chain names that shows the values its description gives; read the value from it")
                    missed = missed[1:]
            if not (ready and reach_n) or self.remaining() < len(reach_n) + 4 or self.closed:
                break
            room = min(3, 6 - len(viewed))
            self._g4_follow(op, goal, reach_n[:max(0, room)], viewed)
        return self._g4_path_report(viewed, missed)

    def perceive(self, name_n, obs):
        self.absorb_n(obs)
        rows = self.slack_rows(obs)
        sense = self.grab_memo(rows) if rows else ''
        if sense:
            self.pending.append(f'(seen on {name_n}) {sense}')
        return sense

    def perform(self, name_n, args_n):
        payload_n = {**self.binding_n, 'step': self.move + 1, 'action': {'name': name_n, 'args': args_n}}
        self._g4_pace()
        tries = 4
        while tries:
            tries -= 1
            outcome = self._g4_timed_post(payload_n)
            data = outcome.data
            if isinstance(data, dict):
                return self._g4_landed(name_n, args_n, data)
            if not self._g4_retry_after(outcome.error or {}, payload_n):
                break
        return None

    def phrase_is_fresh(self, name_n, args_n):
        kept = {}
        for key, val in args_n.items():
            if isinstance(val, bool) or not isinstance(val, int):
                kept[key] = val
        sign = _serial(kept)
        seen = self.sheet_queries.setdefault(name_n, set())
        if sign in seen:
            return False
        seen.add(sign)
        return True

    def piece_aisles(self):
        group_key = self.section_key()
        tags = sorted({str(line.get(group_key) or '').strip() for line in self.lines.values()} - {''}) if group_key else []
        moved = False
        for ware2 in self.items:
            target = self._g4_aisle_move(ware2, tags)
            if target is None:
                continue
            self.journal.append({'kind': 'shelf_parted', 'item': ware2.name_n, 'from': ware2.aisle, 'to': target})
            ware2.aisle = target
            moved = True
        if not moved:
            return
        self.holders_viewed = {}
        self.match_stored = None

    def populate_brief_basket(self):
        basket = self.duties.get('cart') or ''
        limited = set()
        for couple in self.limited:
            limited.update(couple)
        items = self.items
        for num in self.absent_wares():
            ware2 = items[num - 1]
            choice = self._g4_brief_pick(num, ware2, limited)
            plan = self._g4_brief_plan(choice, ware2.amount) if choice else None
            if not plan or len(plan) + 1 > self.remaining():
                return False
            if not self._g4_cart_calls(basket, plan):
                return False
            self.journal.append({'kind': 'filled_line', 'item': num, 'line': choice})
        return True

    def populate_details(self, iids, retain):
        op = self.details_op()
        goal = self.lone_goal(op)
        reader = bool(op and goal)
        if not (reader or self.contrast_details() is True):
            return []
        lines, whole, gone = (self.lines, self.whole, self.vended_out)
        desired = [i_n for i_n in iids if i_n in lines and i_n not in whole and (not gone(i_n))]
        strings = []
        desired, unread_n = self._g4_compare_pass(self.duties.get('compare'), self.contrast_goal(), desired, retain, strings)
        if reader:
            self._g4_detail_pass(op, goal, unread_n + desired, retain, strings)
            if strings:
                self.journal.append({'kind': 'fill_facts', 'n': len(strings)})
        return strings

    def possesses(self, ware2, tag):
        if not tag or ware2 not in self.items:
            return False
        if self.holder_of(tag) == self.items.index(ware2) + 1:
            return True
        return _terms(tag) == _terms(ware2.aisle or '')

    def purchasable(self, iid):
        if self.vended_out(iid):
            return False
        val = (self.lines.get(iid) or {}).get(self.stock_key_n) if self.stock_key_n else None
        if isinstance(val, str):
            return re.search(STOCK_OUTN, val.strip(), re.I) is None
        if val is False:
            return False
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            return True
        return not val <= 0

    def query_cluster(self, group, heard, base=6):
        portion = self._g4_portion(len(group))
        pieces = self._g4_open_tracks(group, portion)
        if len(pieces) < 2:
            if not pieces:
                return []
            num, tags = pieces[0]
            matched, text_n = self.match_query(num, tags, False)
            if not matched:
                return []
            return self._g4_ask_and_close(num, matched, text_n, False, heard, base)
        pieces, text_n = self._g4_fit_cluster(pieces, portion)
        if not pieces:
            return []
        lead_num, lead_tags = pieces[0]
        bundle = self.send_n([(lead_num, lead_tags[0], text_n)], base)
        if not bundle:
            return []
        used = self.track_used_up
        for num, __n in pieces:
            used[num] = used.get(num, 0) + 1
        self.list_asks_n += 1
        held = list(map(len, (ware2.heard for ware2 in self.items)))
        learnt = self.hear_n([(0, text_n, bundle[0][2])], (LIGHTN, HEAVYN))
        learnt.pop('free', None)
        self.grab_learnt(learnt, heard)
        responded = self._g4_settle_cluster(pieces, held)
        if not responded and self._g4_crowded(learnt):
            self.offer = False
            self.journal.append({'kind': 'lists_off', 'item': lead_num})
            return bundle
        if not responded:
            self.track_spent.setdefault(lead_num, set()).add(_terms(lead_tags[0]))
        self.journal.append({'kind': 'mixed_ask', 'parts': pieces, 'answered': responded})
        return bundle

    def query_following(self, heard, base=6):
        found = self._g4_next_query()
        if found is None:
            return []
        num, matched, text_n, displayed = found
        return self._g4_ask_and_close(num, matched, text_n, displayed, heard, base)

    def query_json(self, instruction, models_n, max_pieces=1800):
        for engine in models_n:
            reply = self.think_n([{'role': 'user', 'content': instruction}], [engine], max_pieces) or {}
            parsed = _json_obj_n(reply.get('content'))
            if parsed is not None or self._g4_spent_out():
                return parsed
        return None

    def query_text(self, num, tags, displayed=False):
        if len(tags) < 2:
            similar = self.apart_n.get(tags[0], {}).get(num)
            if not similar:
                return self.inquiry(num, tags[0], self.samples_n(num, tags[0]) if displayed else None)
            return f"Apart from the {' and the '.join(similar)}, which {tags[0]} would you like for the {self.ware_stated(num)}?"
        return f"What {', '.join(tags[:-1])} or {tags[-1]} would you like for the {self.ware_stated(num)}?"

    @_mild(False)
    def quiet_answer(self, dispatched):
        if self.hear_reads_n < 1:
            return False
        known = self._g4_known_words()
        for __n, inquiry, response in dispatched:
            if self._g4_grounded(inquiry, response, known):
                return False
        return True

    def quiet_n(self, name_n, args_n):
        spec = self.specs.get(name_n) or {}
        out_n = dict(args_n)
        for key, schema in (spec.get('properties') or {}).items():
            if isinstance(schema, dict) and 'string' in _sorts(schema) and ('enum' not in schema) and self._g4_needs_quiet(key, out_n.get(key), spec):
                out_n[key] = 'Sorry, nothing in the store meets all of your requirements, so I am not buying anything.'[:schema.get('maxLength', 200)]
        return out_n

    def ready_to_query(self):
        if self.queried or not self.duties.get('ask') or self.texts_open() <= 0 or (self.remaining() <= 8) or (self.timer() > QUERY_BY):
            return False
        if self.cycle_invocations < QUERY_TURNS and (not self._g4_each_ware_seen()):
            return False
        return len(self.whole) > 3 * len(self.items) or self.move >= max(8, self.max_moves // 4)

    def register(self):
        several = len(self.items) > 1
        pieces = []
        for num, ware2 in enumerate(self.items, 1):
            pieces.append(self._g4_item_line(num, ware2, several))
        pieces += self._g4_order_terms()
        return '\n'.join(['- ' + p_n for p_n in pieces])

    def remaining(self):
        return -(self.move - self.max_moves)

    def replacements_n(self, missed):
        if not self.assesses():
            return []
        entries = self._g4_other_records(missed)
        if not entries:
            return []
        sections = (('\n\nREQUEST:\n', self.demand2), ('\n\nREQUIREMENTS IN FORCE:\n', self.register()), ('\n\nSOLD-OUT CART LINE:\n', '\n'.join((f'{i_n}: {self.entry_of(i_n)}' for i_n in missed))), ('\n\nCART NOW:\n', self.basket_text()), ('\n\nOTHER LISTING RECORDS SEEN (newest first):\n', '\n'.join(entries)))
        instruction = EXCHANGE
        for label, body in sections:
            instruction += label + body
        obtained = self.query_json(instruction, self.assesses()[::-1], 3500)
        matches = []
        taken = set()
        for record in _wares((obtained or {}).get('fits')):
            iid = str(record.get('id')) if isinstance(record, dict) else str(record)
            cost = record.get('price') if isinstance(record, dict) else None
            if iid not in self.lines or iid in missed or iid in self.vanished or (iid in self.basket_iids()) or (iid in taken):
                continue
            taken.add(iid)
            matches.append((float(cost) if isinstance(cost, (int, float)) and (not isinstance(cost, bool)) else float('inf'), iid))
        if self.digest.get('pick') in ('most', 'least'):
            graded = [i_n for __n, i_n in matches]
        else:
            graded = [i_n for __n, i_n in sorted(matches, key=lambda f_n: f_n[0])]
        self.journal.append({'kind': 'swap', 'lost': missed, 'fits': matches})
        return graded[:4]

    def rescan_offers(self):
        op = self.deal_op()
        goal = self.lone_goal(op)
        if not (op and goal):
            return ''
        floor = self.move - 2
        if not floor >= self.declined_at:
            floor = self.declined_at
        stale = []
        for iid in self.basket_iids():
            if self.deal_viewed.get(iid, -99) < floor and iid not in self.vanished:
                stale.append(iid)
        if not stale or len(stale) + 5 > self.remaining():
            return ''
        notes = []
        for iid in stale:
            if not self._g5_reread(op, goal, iid, notes):
                break
        self.journal.append({'kind': 'reread', 'ids': stale, 'changed': len(notes)})
        if notes:
            self.edition += 1
        return '\n'.join(notes)

    def resolve_iids(self, name_n, args_n):
        watched = set(self._g5_duties('cart', 'open', 'offer', 'compare'))
        watched.discard(None)
        if name_n not in watched or not self.id_key_n:
            return (args_n, '')
        props = (self.specs.get(name_n) or {}).get('properties') or {}
        free = [field for field, spec in props.items() if self._g5_textlike(spec)]
        if len(free) != 1:
            return (args_n, '')
        field = free[0]
        if name_n != self._g5_duty('cart') and (name_n, field) not in self.id_args_n:
            return (args_n, '')
        fixed_args = dict(args_n)
        if field in args_n:
            raw = args_n[field]
            many = isinstance(raw, list)
            mended = [self._g5_repair_id(one) for one in raw] if many else [self._g5_repair_id(raw)]
            for __n, why in mended:
                if why:
                    return (args_n, why)
            fixed_args[field] = [one for one, __n in mended] if many else mended[0][0]
        return (fixed_args, '')

    def responded_tag(self, tags, fresh):
        viewed = {}
        for __n, key, __n, values in self.key_grid():
            bucket = viewed.setdefault(_terms(key), set())
            for one in values:
                bucket.update(_bases(one))
        lead = None
        for demand in fresh:
            trait, __n, val = demand.partition(':')
            at = self._g5_peak([len(_written(tag) & _written(trait)) for tag in tags])
            if at is None:
                at = self._g5_peak([len(viewed.get(_terms(tag), set()) & _bases(val)) for tag in tags])
            if at is None:
                continue
            if lead is None or at < lead:
                lead = at
        if lead is None:
            return -1
        return lead

    def response_open_invocations(self):
        last = None
        for pos in reversed(range(len(self.dialogue))):
            if self.dialogue[pos].get('tool_calls'):
                last = pos
                break
        if last is None:
            return
        answered = set((m_n.get('tool_call_id') for m_n in self.dialogue[last + 1:]))
        for invoke in self.dialogue[last]['tool_calls']:
            if invoke['id'] in answered:
                continue
            self.dialogue.append({'role': 'tool', 'tool_call_id': invoke['id'], 'content': 'The call may not have run: something went wrong on the way. Look at the state before you repeat it.'})
        self.dialogue.append({'role': 'user', 'content': self.state()})

    def rest(self, skip_n, bound=30):
        order = list(reversed(self.viewed_sequence))
        if self.digest.get('pick') == 'cheapest':
            order.sort(key=self._g5_cheap_key)
        shown = []
        for iid in (one for one in order if one not in skip_n and (not self.vended_out(one))):
            shown.append(_serial(_trim(self.whole.get(iid) or self.lines.get(iid), 160))[:900])
            if len(shown) >= bound:
                break
        return '\n'.join(shown)

    def review(self):
        if self.audited_n:
            return ''
        if not self.assesses():
            return ''
        self.audited_n = True
        obtained, called = self._g5_review_once()
        if not called:
            unopened = []
            for raw in _wares((obtained or {}).get('open')):
                iid = str(raw)
                if iid in self.lines and iid not in self.unfolded and (not self.vended_out(iid)):
                    unopened.append(iid)
            if unopened and self.open_lines(unopened):
                obtained, called = self._g5_review_once()
        if not called:
            return ''
        self.clue_wares = called
        clues = []
        for iids in called.values():
            clues.extend(iids)
        self.clues = clues
        listing = '; '.join((f"item {k_n}: {', '.join(v_n)}" for k_n, v_n in called.items()))
        why = f" ({obtained.get('reason')})" if obtained.get('reason') else ''
        return self._g5_glue(['A careful review thinks these listings may fit: ', listing, why, '. Open them and buy what truly fits; end without a purchase only if each one fails.'])

    def review_invoke(self):
        records = []
        for text in self._g5_records():
            records.append(text)
            if len(records) >= 30:
                break
        if not records:
            return None
        prompt = self._g5_glue([REVIEW, '\n\nREQUEST:\n', self.demand2, '\n\nREQUIREMENTS IN FORCE:\n', self.register(), '\n\nLISTING RECORDS SEEN (newest first):\n', '\n'.join(records)])
        judges = self.assesses()
        got = self.query_json(prompt, [LIGHTN] if judges else [], 4000)
        if HEAVYN in judges and (not self.none_merit) and (not self.review_matches(got)):
            got = self.query_json(prompt, [HEAVYN], 4000) or got
        self.journal.append({'kind': 'audit', 'got': got})
        return got

    def review_matches(self, obtained):
        fits = (obtained or {}).get('fits')
        if not isinstance(fits, dict):
            return {}
        found = {}
        for item, raw in fits.items():
            found[str(item)] = self._g5_live_ids(raw)
        for item in [k_n for k_n, v_n in found.items() if not v_n]:
            del found[item]
        return found

    def row_issues(self):
        self.adopt_tags()
        group_key = self.section_key()
        iids = []
        for iid in self.basket_iids():
            if (self.kept_measures(iid) or 0) > 0:
                iids.append(iid)
        if not (iids and group_key):
            return []
        tag = {}
        for iid in iids:
            tag[iid] = str((self.lines.get(iid) or {}).get(group_key) or '')
        known = set()
        for num, wanted in enumerate(self.items, 1):
            if any((self.possesses(wanted, str(row.get(group_key) or '')) for row in self.lines.values())):
                known.add(num)
        holder = {iid: [num for num, wanted in enumerate(self.items, 1) if self.possesses(wanted, tag[iid])] for iid in iids}
        faults = []
        for num, wanted in enumerate(self.items, 1):
            if num in known:
                self._g5_item_faults(num, wanted, iids, tag, holder, faults)
        if len(known) == len(self.items):
            faults.extend((f"{iid} ({tag[iid] or 'unknown kind'}) is not a kind the shopper asked for: take it out" for iid in iids if not holder[iid]))
        faults.extend((f'{iid} is sold out' for iid in iids if self.vended_out(iid)))
        self._g5_budget_faults(iids, faults)
        return faults

    def run_invocations(self, invocations):
        batch = self.join_opens_n(invocations)
        return [self._g5_invocation(batch, invoke) for invoke in batch]

    def run_n(self):
        if self.closed:
            return self.journal
        self.scan_demand()
        self.journal.append({'kind': 'brief', 'brief': self.digest})
        unfolding = self.pane.get('opening')
        if unfolding is not None:
            unfolding = _recover(unfolding)
            self.absorb_n(unfolding)
        shifted = self.initial_moves() if not self.closed else ''
        self.world_margin = WORLD_MARGIN
        opening = self._g5_intro(unfolding, shifted)
        self.dialogue = [dict(zip(('role', 'content'), ('system', GUIDEN)))]
        self.dialogue.append(dict(zip(('role', 'content'), ('user', opening))))
        self.work_n(CYCLE_INVOCATIONS, SPEND_HARDN)
        if not self.closed:
            self._g5_second_round()
            if not self.closed:
                self.conclude()
        self.journal.append(dict(zip(('kind', 'finished', 'steps', 'cost', 'calls'), ('end', self.closed, self.move, round(self.expense, 5), self.invocations))))
        return self.journal

    def samples_n(self, num, tag):
        desire = _terms(tag)
        resolved = ' '.join(self.items[num - 1].needs).lower()
        for __n, name_n, __n, values in self.key_grid():
            if name_n != desire:
                continue
            picked = []
            for val in values:
                if val and val not in resolved and (len(val) <= 30):
                    picked.append(val)
            return picked[:2]
        return []

    def scan_demand(self):
        rows = [f"- {fn_n['name']}: {fn_n.get('description', '')} | parameters: {_serial(fn_n.get('parameters') or {})}" for fn_n in (op['function'] for op in self.ops)]
        instruction = self._g5_glue([DIGEST, '\nTOOLS:\n', '\n'.join(rows), '\n\nREQUEST:\n', self.demand2])
        if self.pane.get('opening') is not None:
            instruction += '\n\nOPENING NOTE FROM THE STORE:\n' + _serial(_trim(self.pane.get('opening'), 1500))
        digest = self._g5_first_digest(instruction)
        if digest:
            self.waited_n = 0
        self.digest = digest
        roles = digest.get('roles') if isinstance(digest.get('roles'), dict) else {}
        self.duties = {k_n: v_n for k_n, v_n in roles.items() if isinstance(v_n, str) and v_n in self.specs}
        for key in ('buy_args', 'refuse_args'):
            if not isinstance(digest.get(key), dict):
                digest[key] = {}
        self._g5_split_modes(digest)
        for record in _wares(digest.get('items')):
            if not isinstance(record, dict):
                continue
            quantity_n = _tally(record.get('quantity'))
            count = quantity_n if quantity_n and quantity_n > 0 else 1
            needs = [str(m_n) for m_n in _wares(record.get('musts')) if m_n]
            label = str(record.get('name') or 'the product')
            shelf = str(record.get('shelf') or '')
            self.items.append(ware(label, shelf, count, needs))
        became = self._g5_merge_items()
        if not self.items:
            self.items = [ware('the requested product', '', 1, [])]
        self._g5_pairs(digest, became)
        self._g5_pointers(digest)
        self.unmix_n()
        self._g5_limits(digest)
        self._g5_procedure(digest)

    def scan_form(self, name_n, obs):
        if name_n in self.shapes_n:
            return self.shapes_n[name_n]
        if not isinstance(obs, dict):
            return None
        tables = []
        for key, val in obs.items():
            if isinstance(val, list) and len(val) >= 2 and all((isinstance(r_n, dict) for r_n in val)):
                tables.append(key)
        if not tables:
            return None
        self.shapes_n[name_n] = None
        sample_n = {}
        for key, val in obs.items():
            sample_n[key] = val[:3] if key in tables else val
        instruction = self._g5_glue([FORM, '\n\nTOOL: ', name_n, ' | parameters: ', _serial(self.specs.get(name_n) or {}), '\n\nRESULT (first rows only):\n', _serial(_trim(sample_n, 120))[:4000]])
        obtained = self.query_json(instruction, [LIGHTN, HEAVYN], 2500) or {}
        lines = obtained.get('rows') if obtained.get('rows') in tables else None
        if not lines:
            return None
        initial = obs[lines][0]
        traits = (self.specs.get(name_n) or {}).get('properties') or {}
        named = [self._g5_named(obtained.get(slot), locus) for slot, locus in (('count', obs), ('next', obs), ('next_arg', traits), ('price', initial), ('stock', initial))]
        form = {'rows': lines, 'count': named[0], 'next': named[1], 'next_arg': named[2], 'price': named[3], 'stock': named[4]}
        if form['count']:
            counted = obs[form['count']]
            if not isinstance(counted, int) or isinstance(obs[form['count']], bool):
                form['count'] = None
        self.shapes_n[name_n] = form
        self.cost_key = self.cost_key or form['price']
        self.stock_key_n = self.stock_key_n or form['stock']
        self.journal.append({'kind': 'shape', 'tool': name_n, 'shape': form})
        return form

    def e1_fill(self, num):
        """Labels for an item no read listing describes yet: the properties a shopper buying it would most likely still
        have a fixed wish about, from one short model call."""
        ware2 = self.items[num - 1]
        prompt = '\n'.join(('A shopper wants to buy the item below in an online store. List the product properties (attribute names as a store would label them, two or three words at most each) that such a shopper most often has a firm wish about and has not stated yet, most likely first.', 'Return ONE JSON object and nothing else: {"labels": [up to 6 strings]}', '', 'ITEM: ' + ware2.name_n, 'STATED: ' + ('; '.join(ware2.needs) or 'nothing')))
        got = self.query_json(prompt, [LIGHTN, HEAVYN], 600) or {}
        stated = {_terms(n.split(':')[0]) for n in ware2.needs}
        out = []
        for lab in _wares(got.get('labels')):
            lab = str(lab).strip().lower().rstrip('?.')
            if lab and len(lab.split()) <= 4 and (not re.search('price|cost|budget|quantity|warranty|shipping|delivery|stock|seller|rating|review', lab)) and (_terms(lab) not in stated) and all((_terms(lab) != _terms(o) for o in out)):
                out.append(lab)
        self.journal.append({'kind': 'e1_fill', 'item': num, 'labels': out[:6]})
        return out[:6]

    def scheme_tracks(self, count):
        self.adopt_tags()
        grid = self.key_grid()
        resolved = [f'item {num}: {ware2.name_n} - settled by the request: ' + ('; '.join(ware2.needs) or 'nothing') for num, ware2 in enumerate(self.items, 1)]
        offer = list(self._g5_offer_lines(grid))
        instruction = self._g5_glue([SCHEME.replace('COUNT', str(max(1, count))), '\n\nREQUEST:\n', self.demand2, '\n\nITEMS:\n', '\n'.join(resolved)])
        initial = [str(x_n) for x_n in _wares(self.digest.get('ask_first')) if str(x_n).strip()]
        if initial:
            instruction += '\n\nASK FIRST (the request asks to be asked about these):\n' + '\n'.join(initial)
        instruction += '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(offer) or '(no full listing records yet)')
        scheme = self.query_json(instruction, [HEAVYN, LIGHTN], 3500) or {}
        tracks = {}
        for num in range(1, len(self.items) + 1):
            tracks[num] = []
        hinted_n = {num: set() for num in tracks}
        self._g5_seed_tracks(scheme, tracks, hinted_n)
        for num in tracks:
            tracks[num] = self._g5_lane(num, grid, tracks[num], hinted_n)
        if len(self.items) > 1:
            for num in tracks:
                if not tracks[num]:
                    tracks[num] = self.e1_fill(num)
        lanes = {}
        for num, lane in tracks.items():
            lanes[num] = lane[:8]
        self.journal.append(dict(zip(('kind', 'ask', 'lanes'), ('plan', _trim(scheme.get('ask'), 120), lanes))))
        return tracks

    def section_key(self):
        tallies = {}
        present_n = {}
        for key, val in self._g5_short_texts():
            tallies.setdefault(key, set()).add(val)
            present_n[key] = present_n.get(key, 0) + 1
        sum2 = max(1, len(self.lines))
        learned = self.tags()
        best_rating = None
        best_key = None
        for key, values in tallies.items():
            if present_n[key] < 0.8 * sum2:
                continue
            if len(values) > 40:
                continue
            width = len(values)
            listed_n = sum((1 for v_n in values if v_n.strip().lower() in learned)) / width
            mean_n = sum((len(v_n) for v_n in values)) / width
            rating = (-listed_n, width, -present_n[key], -mean_n)
            if best_rating is not None and (not rating < best_rating):
                continue
            best_rating, best_key = (rating, key)
        if best_rating is None:
            return None
        return best_key

    def send_n(self, questions_n, base=6):
        op = self.duties['ask']
        dispatched = []
        for question in questions_n:
            if not self._g5_deliver(op, question, base, dispatched):
                break
        return dispatched

    def series_for_n(self, num: int) -> list[str]:
        index = self.match_position()
        signs = index['marks']
        basket = self.basket_iids()
        tallies: dict[str, int] = {}
        kept: list[str] = []
        for i_n, family_n in self._g6_series_hits(num, index, signs):
            tallies[family_n] = tallies.get(family_n, 0) + 1
            if family_n not in kept and i_n in basket:
                kept.append(family_n)
        kept += [f_n for f_n in sorted(tallies, key=lambda f_n: (-tallies[f_n], f_n)) if f_n not in kept]
        return kept[:TIE_SEARCHES]

    def settle_n(self, num: int, tags: list[str], displayed: bool, fresh: list[str], packed: bool) -> None:
        spent = self.track_spent.setdefault(num, set())
        if len(tags) != 1:
            self._g6_settle_many(num, tags, fresh, packed, spent)
            return
        label = tags[0]
        spent.add(_terms(label))
        if packed and self.offer:
            self.offer = False
            self.journal.append({'kind': 'lists_off', 'item': num})
        if not packed or fresh:
            return
        if num not in self.one_product_n and self.name_pieces(num):
            self.one_product_n.add(num)
            said = self.ware_stated(num)
            self.journal.append({'kind': 'one_product', 'item': num, 'said': said})
            self.trail_upward.insert(0, (num, tags, displayed))
            return
        if self.apart_n:
            return
        similar = self.look_alikes_n(num, label)
        if not similar:
            return
        self.apart_n.setdefault(label, {})[num] = similar
        self.journal.append({'kind': 'apart', 'item': num, 'label': label, 'from': similar})
        self.trail_upward.insert(0, (num, tags, False))

    @staticmethod
    def shared_aisle(digest: dict[str, Any]) -> bool:
        seen: set[str] = set()
        for e_n in _wares(digest.get('items')):
            if not isinstance(e_n, dict):
                continue
            shelf = _terms(str(e_n.get('shelf') or ''))
            if shelf in seen:
                return True
            if shelf:
                seen.add(shelf)
        return False

    def sheet_ops(self) -> set[str]:
        found: set[str] = set()
        for t_n in (self.duties.get('offer') or self.duties.get('open'), self.duties.get('compare')):
            if t_n:
                found.add(t_n)
        return found

    def sheet_scan(self, iid: str) -> bool:
        page = self.duties.get('offer')
        if iid not in self.unfolded:
            return False
        if page and self.lone_goal(page):
            return page in self.panes.get(iid, ())
        return True

    def sheet_text(self, iid: str, entry: dict[str, Any]) -> str:
        chunk = _detail_chunk(entry) or {}
        remainder: dict[str, Any] = {}
        for k_n, v_n in entry.items():
            if isinstance(v_n, (dict, list)) or v_n is chunk:
                continue
            remainder[k_n] = v_n
        head = _serial(_trim(chunk, 120))[:SHEET_ATTR_CHARS]
        tail = _serial(_trim(remainder, 200))[:SHEET_REMAINDER_CHARS]
        return f'{iid}: attributes ' + head + ' | ' + tail

    @_mild('')
    def shorter_lookup(self, name_n: str, args_n: dict[str, Any], obs: Any) -> str:
        scheme = self.val_pattern()
        if not self._g6_short_ready(scheme, name_n, obs):
            return ''
        slot = scheme[2]
        text_n = args_n.get(slot)
        if not isinstance(text_n, str) or self.entries_in(obs) or len(text_n.split()) < 3:
            return ''
        if any(map(_g6_counts_as_hit, obs.values())):
            return ''
        words = [w_n for w_n in text_n.split() if not _tag_codes(w_n)]
        short = ' '.join(words[:2])
        if not short:
            return ''
        merged = dict(args_n)
        merged[slot] = short
        for k_n in [k_n for k_n, v_n in merged.items() if isinstance(v_n, int) and (not isinstance(v_n, bool))]:
            del merged[k_n]
        invoke, fine = _g6_shape(merged, self.specs.get(name_n), 'args')
        if not fine:
            return ''
        answer = self.perform(name_n, invoke)
        if answer is None:
            return ''
        viewed = answer.get('observation')
        self.perceive(name_n, viewed)
        found = self.entries_in(viewed)
        self.journal.append({'kind': 'shorter', 'from': text_n, 'to': short, 'rows': len(found)})
        return f'\n[NVIDIA] Nothing has all those words; the same search with {short!r} shows:\n' + self.display(viewed)

    def slack_rows(self, obs: Any) -> list[str]:
        if not isinstance(obs, dict) or self.iid(obs):
            return []
        return [row for row in self._g6_wordy(obs) if row not in self.learnt]

    def p1_side_notes(self, op, response):
        """A reply that comes with a list of notes beside its own text: the notes go to the change reader, the text is the answer."""
        if not isinstance(response, dict) or response.get('error'):
            return response
        said = {k: v for k, v in response.items() if isinstance(v, str) and len(v.split()) >= 2}
        notes = [n for k, v in response.items() if isinstance(v, list) for n in v if isinstance(n, str) and len(n.split()) >= 4]
        if not said or not notes:
            return response
        fresh = [n for n in notes if n not in self.learnt]
        sense = self.grab_memo(fresh) if fresh else ''
        if sense:
            self.pending.append(f'(seen on {op}) {sense}')
        self.journal.append({'kind': 'p1_reply_notes', 'notes': [n[:120] for n in notes], 'sense': sense[:160]})
        return said

    def spoken_n(self, obs: Any) -> str:
        if isinstance(obs, str):
            return obs
        located = list(_g6_phrases(obs, 0))
        if located:
            return ' '.join(located)[:800]
        return _serial(obs)[:400]

    def state(self) -> str:
        remaining = self.remaining()
        text_n = f'[NVIDIA] step {self.move} of {self.max_moves} used, {remaining} left.'
        pace = self._g6_pace_note(remaining)
        if pace:
            text_n += pace
        if self.timer() >= WIND_DOWNN and remaining > self.end2() + 1:
            text_n += '; '.join((' TIME IS NEARLY UP: set the cart to the chosen lines and check out within the next two turns', 'an episode that runs out of time earns nothing.'))
        pending = self._g6_unresolved()
        idle = 0 == self.purchases
        if pending and idle:
            text_n += '\n'.join(('', '[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] ')) + ' | '.join(pending[:3])
        if self.clues:
            text_n += '\n[review candidates] ' + ', '.join(self.clues)
        scheme = self.match_scheme() if self.limited and idle else ''
        if scheme:
            text_n += '\n' + scheme
        if self.memos:
            text_n += '\n[changes so far] ' + ' | '.join(self.memos[-4:])
        return text_n + '\n[ledger]\n' + self.register()

    def states_demands(self, iid: str, num: int) -> bool:
        chunk = _detail_chunk(self.whole.get(iid) or {}) or {}
        for holder, demand in self.demands_now():
            if num == holder and (not _g6_demand_shown(demand, chunk)):
                return False
        return True

    def store_lines(self, about_n: str, iids: list[str]) -> list[str]:
        barred = self.not_maker_fields()
        small = about_n.lower()
        found = lambda n_n: re.search('(?<![a-z0-9])' + re.escape(n_n.lower()) + '(?![a-z0-9])', small)
        return [iid for iid in iids if iid not in self.unfolded and any(map(found, self._g6_line_names(iid, barred)))]

    def tags(self) -> set[str]:
        located: set[str] = set()
        for spec in self.specs.values():
            located.update(_g6_enum_words(spec, 0))
        return located

    def tally_fixes(self) -> list[tuple[str, int, int]]:
        group_key = self.section_key()
        if not group_key:
            return []
        fixes: list[tuple[str, int, int]] = []
        for iid in self.basket_iids():
            has = self.kept_measures(iid) or 0
            shelf = str((self.lines.get(iid) or {}).get(group_key) or '')
            holders = [n_n + 1 for n_n in range(len(self.items)) if self.possesses(self.items[n_n], shelf)]
            alone = self._g6_alone_in_basket(holders, group_key)
            if not 0 < has or len(holders) != 1 or (not alone):
                continue
            wanted = self.items[holders[0] - 1].amount
            if has != wanted and (not self.vended_out(iid)):
                fixes.append((iid, has, wanted))
        return fixes

    def tardy_inquiry(self) -> str:
        gate = self._g6_tardy_gate()
        if gate is not None:
            return gate
        self.tardy_queried = True
        if not self.tracks:
            self._g6_seed_tracks()
        heard: list[str] = []
        allowance = self.allowance
        bundle = self.query_following(heard, TARDY_MOVES - 2)
        if not bundle:
            self.journal.append({'kind': 'late_ask', 'skipped': 'no property left to ask or no message sent'})
            return ''
        if allowance != self.allowance:
            heard.append(f'budget now {self.allowance}')
        num, inquiry, response = bundle[0]
        self.journal.append({'kind': 'late_ask', 'q': inquiry, 'a': response[:200], 'told': heard})
        if not heard:
            return ''
        self.edition += 1
        recap = self._g6_late_recap(inquiry, response, heard)
        looked = self.val_lookup([num])
        return recap + ('' if not looked else '\n' + looked)

    def texts_open(self) -> int:
        held = self.reserve_n - self.barrier_texts
        if not held > 0:
            held = 0
        return self.text_bound - self.text_spent - held

    def think_n(self, messages_n: list[dict[str, Any]], models_n: list[str], max_pieces: int, ops: list[dict[str, Any]] | None=None) -> dict[str, Any] | None:
        for engine in models_n:
            if self.invocations >= ALL_INVOCATIONS:
                return None
            if self.engine_space() < INVOKE_BASE:
                moment = round(self.timer(), 1)
                self.journal.append({'kind': 'late_call_skipped', 'at': moment})
                return None
            self.invocations += 1
            note, state = self._g6_engine_tries(engine, messages_n, max_pieces, ops)
            if note is not None:
                return note
            if state == 402:
                self.none_merit = True
                return None
        return None

    def tie_contains(self, stocked: str, labeler: str, kins: set[str], index: dict[str, Any]) -> bool | None:
        own = kins.intersection((index['marks'].get(stocked) or {}).get('fam') or [])
        wanted = kins.intersection((index['marks'].get(labeler) or {}).get('named') or [])
        if own and wanted:
            return not own.isdisjoint(wanted)
        return None

    def tie_issues(self, piece: int, primary: int, unit: str, rows: list[str], index: dict[str, Any]) -> list[str]:
        scheme2 = self.tie_scheme(piece, primary, index)
        if not scheme2:
            return []
        carrier_n, __n, kins = scheme2
        signs = index['marks']
        swap = piece == carrier_n
        pairs = ((row, unit) if swap else (unit, row) for row in rows)
        issues = []
        for stocked, labeler in pairs:
            if self.tie_contains(stocked, labeler, kins, index) is False:
                issues.append(self._g6_tie_line(stocked, labeler, kins, signs, carrier_n, primary))
        return issues

    def tie_scheme(self, piece: int, primary: int, index: dict[str, Any]) -> tuple[int, int, set[str]] | None:
        signs = index['marks']
        for carrier_n, labeler in ((piece, primary), (primary, piece)):
            lines = self.ware_iids(carrier_n)
            tagged = 0
            kins: set[str] = set()
            for i_n in lines:
                if not (signs.get(i_n) or {}).get('fam'):
                    continue
                tagged += 1
                for f_n in signs[i_n]['fam']:
                    kins.add(f_n)
            if tagged >= TIE_LINES and 2 * tagged >= len(lines) and (len(kins) <= TIE_KINS) and self._g6_names_kin(labeler, kins, signs):
                return (carrier_n, labeler, kins)
        return None

    def timer(self) -> float:
        began = self.started_n
        return time.monotonic() - began

    def track_fields(self, num: int, grid: list[tuple[str, str, int, list[str]]]) -> list[str]:
        ware2 = self.items[num - 1]
        resolved = [_root(m_n.split(':', 1)[0]) for m_n in ware2.needs]
        rows = [(tag, key, stocked, len(values)) for tag, key, stocked, values in grid]
        lines = [r_n for r_n in rows if self.holder_of(r_n[0]) == num]
        measured_n: dict[str, bool] = {}
        for __n, key, __n, values in grid:
            hits = len([v_n for v_n in values if re.match('^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$', v_n)])
            measured_n[key] = 2 * hits > len(values)
        if not lines and len(self.items) == 1:
            lines = rows

        def rank(r_n: tuple[str, str, int, int]) -> tuple[Any, bool, int]:
            return (-r_n[2], measured_n.get(r_n[1], False), -r_n[3])
        lines.sort(key=rank)
        every_n: list[str] = []
        for r_n in lines:
            if not any((_names_n(trait, _root(r_n[1])) for trait in resolved)):
                every_n.append(r_n[1])
        keys = [key for key in every_n if not re.search(NOT_PROPERTYN, key) or re.search('\\w[ _]models?\\b', key)]
        plain: list[str] = []
        broad: list[str] = []
        for key in keys:
            (broad if re.search(CATCH_ALLN, key) else plain).append(key)
        return plain + broad + [key for key in every_n if key not in keys]

    def trim_n(self) -> None:
        magnitude = 0
        for m_n in self.dialogue:
            magnitude += len(_serial(m_n.get('content')))
        if magnitude <= HISTORY_BOUND:
            return
        ops = [i_n for i_n in range(len(self.dialogue)) if self.dialogue[i_n].get('role') == 'tool']
        for index in ops[:len(ops) // 2 or 1]:
            entry = self.dialogue[index]
            substance = entry.get('content') or ''
            if len(substance) <= 600:
                continue
            iids = [i_n for i_n in dict.fromkeys(re.findall('"([A-Za-z0-9][A-Za-z0-9_\\-:.]{5,40})"', substance)) if i_n in self.lines][:20]
            entry['content'] = substance[:400] + f"...(older result shortened; listings in it: {', '.join(iids)})"

    def turning(self, name_n: str, obs: Any) -> tuple[int, int, str, Any] | None:
        form = self.scan_form(name_n, obs)
        ready = form and form['count'] and form['next'] and form['next_arg'] and isinstance(obs, dict)
        if not ready:
            return None
        rows = obs.get(form['rows'])
        count = obs.get(form['count'])
        cursor = obs.get(form['next'])
        if _g6_bad_page(rows, count, cursor):
            return None
        return (count, len(rows), form['next_arg'], cursor)

    def turns_n(self, cycle_ceiling: int, spend_ceiling: float) -> None:
        inactive = 0
        while self._g6_may_turn(cycle_ceiling, spend_ceiling):
            if self.ready_to_query():
                recap = self.interview_n()
                if recap:
                    self.dialogue.append({'role': 'user', 'content': recap + '\n' + self.state()})
                continue
            self.trim_n()
            self.cycle_invocations += 1
            note = self.think_n(self.dialogue, [LIGHTN, SPAREN], 3000, self.offered_n())
            if note is None:
                if self._g6_silence_ends():
                    break
                continue
            invocations = self._g6_calls_of(note)
            substance = _g6_said(note.get('content'))
            if invocations:
                inactive = 0
                if not self._g6_ask_before(substance, invocations):
                    self._g6_play(substance, invocations)
                continue
            inactive += 1
            self.dialogue.append({'role': 'assistant', 'content': substance or '(no call)'})
            self.dialogue.append({'role': 'user', 'content': 'Call one of the tools to continue.\n' + self.state()})
            if inactive >= 3:
                break

    def unfolded_store(self, about_n: str) -> list[str]:
        barred = self.not_maker_fields()
        small = about_n.lower()
        out_n: list[str] = []
        for iid in self.unfolded:
            for n_n in self._g6_line_names(iid, barred):
                if re.search('(?<![a-z0-9])' + re.escape(n_n.lower()) + '(?![a-z0-9])', small):
                    out_n.append(iid)
                    break
        return out_n

    def unmix_n(self) -> None:
        for num, ware2 in enumerate(self.items, 1):
            pointers_n = [demand for n_n, demand in self.pointers if num == n_n and demand in ware2.needs]
            if not pointers_n:
                continue
            remainder = [m_n for m_n in ware2.needs if m_n not in pointers_n]
            if remainder:
                self._g6_unmix_item(num, ware2, pointers_n, remainder)

    def unread_sheets(self, op: str, desired: dict[int, set[str]], space: int) -> int:
        used_up = 0
        group_key = self.section_key()
        portion = max(1, space // max(1, len(desired)))
        for bound in (portion, space):
            for num, terms in desired.items():
                halt = min(space, used_up + bound)
                for sign, (name_n, args_n) in list(self.scan_args.items()):
                    if op != name_n or terms.isdisjoint(_g6_arg_terms(args_n)):
                        continue
                    viewed = self.scan_invocations[sign]
                    if not self._g6_sheet_for(num, viewed, group_key):
                        continue
                    made, aborted = self._g6_follow_pages(op, sign, args_n, viewed, halt - used_up)
                    used_up += made
                    if aborted:
                        return used_up
        return used_up

    def unspelled_n(self, name_n: str, obs: Any) -> Any:
        panes = {self.duties.get(d_n) for d_n in ('open', 'offer', 'compare')}
        paged = name_n in panes or name_n != self.duties.get('search')
        out_n = _fold_doubles(_recover(obs, page=paged))
        if name_n not in panes:
            return out_n
        for iid, line in self.lines.items():
            if iid in self.unfolded:
                continue
            for k_n in line:
                self.line_fields.add(k_n)
        learned = self.line_fields
        if not learned or not isinstance(out_n, dict):
            return out_n
        if self.iid(out_n):
            return self._g6_fold_extra(out_n, learned)
        return {k_n: [self._g6_fold_extra(r_n, learned) for r_n in v_n] if isinstance(v_n, list) else v_n for k_n, v_n in out_n.items()}

    def val_lookup(self, nums: list[int]) -> str:
        scheme = self.val_pattern()
        if not scheme or self.remaining() < self.end2() + VAL_SPACE:
            return ''
        op, origin, slot = scheme
        viewed: list[str] = []
        allowance = min(8, self.remaining() - self.end2() - VAL_RETAIN)
        for num in nums:
            couples = self.val_terms(num)
            ware2 = self.items[num - 1]
            novel = [v_n for v_n, heard in couples if heard]
            if novel and allowance > 2:
                tries = _g6_value_tries([v_n for v_n, __n in couples], novel, ware2.name_n)
                allowance = self._g6_value_probe(op, origin, slot, num, ware2, tries, viewed, allowance)
        if not viewed:
            return ''
        seen = [s_n[:200] for s_n in viewed]
        self.journal.append({'kind': 'value_search', 'seen': seen})
        return "[NVIDIA searched each such item's requirement values together and read the first fitting-kind rows] " + '\n'.join(viewed)

    def val_pattern(self) -> tuple[str, dict[str, Any], str] | None:
        scheme = self.lookup_pattern()
        op = self.duties.get('search')
        if scheme or not op:
            return scheme
        spec = self.specs.get(op) or {}
        traits = spec.get('properties') or {}
        required_n = spec.get('required') or []
        told = self._g6_told_field(op)
        if told is not None:
            return told
        called: list[str] = []
        enums_n: dict[str, list[Any]] = {}
        for k_n, v_n in traits.items():
            if not isinstance(v_n, dict):
                continue
            if 'string' in _sorts(v_n) and 'enum' not in v_n and re.search(PHRASE_TERMS, _snaky(k_n), re.I):
                called.append(k_n)
            if isinstance(v_n.get('enum'), list) and len(v_n['enum']) >= 2:
                enums_n[k_n] = v_n['enum']
        final = self.looked[-1] if self.looked else {}
        manners = []
        for key, choices2 in enums_n.items():
            picked = _g6_enum_pick(key, choices2, required_n, final)
            if picked:
                manners.append(picked)
        if len(called) != 1:
            return None
        if len(manners) > 1 or (enums_n and (not manners)):
            return None
        origin = {k_n: final[k_n] for k_n in required_n if k_n in final and k_n != called[0]}
        for key, choice in manners:
            origin[key] = choice
        return (op, origin, called[0])

    def val_terms(self, num: int) -> list[tuple[str, bool]]:
        heard = self.items[num - 1].heard
        return list(self._g6_term_pairs(num, heard))

    def vended_out(self, iid: str) -> bool:
        if iid in self.unavailable_n:
            return True
        return iid in self.vanished

    def verify_basket(self, iids: list[str] | None=None, bound: float=SPEND_MILD) -> str:
        clue = iids is not None
        if not clue:
            iids = self.basket_iids()
        key = self.verify_key(iids)
        if key in self.checked_n or not self.assesses(bound) or (not iids):
            return ''
        self.checked_n.add(key)
        instruction = self._g6_review_brief(iids, clue)
        obtained = self._g6_review_run(instruction, bound, clue)
        self.journal.append({'kind': 'check', 'cart': iids, 'got': obtained})
        self.verdicts_n[key] = bool(obtained) and obtained.get('ok') is not False
        if not obtained or obtained.get('ok') is not False:
            return ''
        self.clues = [i_n for i_n in self.clues if i_n not in iids]
        return self._g6_review_note(obtained)

    def verify_key(self, iids: list[str]) -> str:
        ordered = sorted(iids)
        return '|'.join(ordered) + f'#{self.edition}'

    def ware_clusters(self) -> list[list[str]]:
        group_key = self.section_key()
        groups: dict[str, list[str]] = {}
        for iid in self.viewed_sequence:
            tag = str(self.lines[iid].get(group_key) or '') if group_key else ''
            if tag not in groups:
                groups[tag] = []
            groups[tag].append(iid)
        desired = list(map(_g6_item_marks, self.items))
        reserve = [iids for tag, iids in groups.items() if self._g6_cluster_fits(tag, desired)]
        if not reserve:
            reserve = list(groups.values())
        return sorted(reserve, key=len, reverse=True)[:max(1, len(self.items))]

    def ware_iids(self, num: int) -> list[str]:
        group_key = self.section_key()
        ware2 = self.items[num - 1]
        found: list[str] = []
        if not group_key:
            return found
        for i_n in self.viewed_sequence:
            line = self.lines.get(i_n) or {}
            if self.possesses(ware2, str(line.get(group_key) or '')):
                found.append(i_n)
        return found

    def ware_stated(self, num: int) -> str:
        ware2 = self.items[num - 1]
        pieces = self.name_pieces(num)
        if not pieces:
            return ware2.name_n
        if num not in self.one_product_n:
            return re.sub('\\band\\b', 'or', ware2.name_n, flags=re.I)
        group_key = self.section_key()
        called: list[set[str]] = []
        for line in self.lines.values():
            if group_key and (not self.possesses(ware2, str(line.get(group_key) or ''))):
                continue
            called.append(_bases(_g6_line_text(line, group_key)))
        strikes: dict[str, int] = {}
        for p_n in pieces:
            last = _bases(p_n.split()[-1])
            strikes[p_n] = len([terms for terms in called if last & terms])
        best = pieces[0]
        for p_n in pieces[1:]:
            if strikes[p_n] > strikes[best]:
                best = p_n
        return best

    def widen_n(self) -> None:
        op = self.duties.get('compare')
        spec = self.specs.get(op or '') or {}
        traits = spec.get('properties') or {}
        goal = _g6_array_slot(traits)
        if not self._g6_compare_ready(op, goal, spec):
            self.widen_singly_n()
            return
        if self.remaining() <= 12:
            return
        ceiling = min(_g6_item_cap(traits[goal]), self.contrast_space())
        graded = sorted(self.ware_clusters(), key=self._g6_opened_count)
        choices = _g6_round_robin([[i_n for i_n in iids if i_n not in self.whole] for iids in graded], ceiling)
        if len(choices) < 2:
            self.populate_details(choices, 12)
            return
        issues: list[str] = []
        args_n = _form({goal: choices}, spec, 'args', issues)
        if issues:
            return
        prior = len(self.whole)
        answer = self.perform(op, args_n)
        if answer is not None:
            self._g6_widen_seen(op, spec, goal, choices, prior, answer)

    def widen_singly_n(self) -> None:
        op = self.details_op()
        goal = self.lone_goal(op)
        if not op or not goal:
            return
        desired: set[str] = set()
        for __n, demand in self.demands_now():
            desired.update(re.findall('[a-z0-9]{3,}', demand.split(':', 1)[-1].lower()))
        each_n = WIDEN_EACHN if len(self.items) <= 2 else 4
        tracks: list[list[str]] = []
        heads_n: list[int] = []
        for iids in self.ware_clusters():
            track, head = self._g6_single_track(iids, desired, each_n)
            tracks.append(track)
            heads_n.append(head)
        initial = _g6_layered(tracks, heads_n, 4, True)
        extra = _g6_layered(tracks, heads_n, each_n, False)
        ceiling = max(5, 2 * len(tracks) + 2)
        scan = self.populate_details(initial[:min(ceiling, 8)], 22)
        later = initial[ceiling:] + extra
        self.populate_details(later[:max(0, 10 - len(scan))], 30)

    def work_n(self, cycle_ceiling: int, spend_ceiling: float) -> None:
        rounds = 0
        while rounds < CRASH_TURNSN:
            rounds += 1
            if self._g6_guarded_turns(cycle_ceiling, spend_ceiling):
                return

    def _g1_owns_any(self, ware2, tags):
        return any((self.possesses(ware2, tag) for tag in tags))

    def _g1_walk(self, leaf, level, in_list):
        if level > 6:
            return
        if isinstance(leaf, list):
            listed = self.learn_lines(leaf)
            for val in leaf:
                self._g1_walk(val, level + 1, listed)
        elif isinstance(leaf, dict):
            iid = self.iid(leaf, in_list, level)
            if iid:
                self._g1_take(iid, leaf)
            else:
                for val in leaf.values():
                    self._g1_walk(val, level + 1, False)

    def _g1_take(self, iid, leaf):
        row = self.lines.get(iid)
        if row is None:
            self.viewed_sequence.append(iid)
            row = self.lines[iid] = {}
        if _g1_flip(row, leaf, True, False):
            self.unavailable_n.add(iid)
        elif _g1_flip(row, leaf, False, True):
            self.unavailable_n.discard(iid)
        row.update(leaf)
        if _detail_chunk(leaf):
            self.whole[iid] = dict(row)
        elif iid in self.whole:
            self.whole[iid].update(leaf)

    def _g2_tool_lines(self, closing: set[str]) -> list[str]:
        listed: list[str] = []
        for t_n in self.ops:
            if t_n['function']['name'] in closing:
                continue
            listed.append(f"- {t_n['function']['name']}: {t_n['function'].get('description', '')} | parameters: {_serial(t_n['function'].get('parameters') or {})}")
        return listed

    def _g2_gate_prompt(self, phase: str, moves: list[dict[str, Any]], ops: list[str]) -> str:
        wanted = 'end without a purchase'
        if phase == 'order':
            wanted = 'place the order'
        text_n = BARRIER.replace('STAGE', wanted)
        text_n += '\n\nREQUEST:\n' + self.demand2
        text_n += '\n\nSTEPS DUE NOW:\n' + '\n'.join([f"- {p_n['step']}" for p_n in moves])
        text_n += '\n\nLEDGER:\n' + self.register()
        text_n += '\n\nCART NOW: ' + self.basket_text()
        record = [f'{i_n}. {row}' for i_n, row in enumerate(self.completed_log, 1)]
        text_n += '\n\nRECORD (messages and step calls so far):\n' + ('\n'.join(record) or '(nothing yet)')
        notes = ''
        if self.memos:
            notes = '\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.memos[-4:])
        text_n += notes
        closer = 'order' if 'order' == phase else 'ending'
        text_n += f'\n\nMESSAGES LEFT: {max(0, self.text_bound - self.text_spent)}   STEPS LEFT: {self.remaining()} (the {closer} itself takes one)'
        return text_n + '\n\nTOOLS:\n' + '\n'.join(ops)

    def _g2_gate_evidence(self, converse: bool) -> set[int]:
        seen: set[int] = set()
        moves_n = self.completed_moves
        for i_n in range(1, len(self.completed_log) + 1):
            if converse and (i_n > len(moves_n) or not moves_n[i_n - 1] > self.basket_at):
                continue
            seen.add(i_n)
        return seen

    @staticmethod
    def _g2_unproven(steps: Any, evidence: set[int]) -> list[dict[str, Any]]:
        out_n: list[dict[str, Any]] = []
        for e_n in _wares(steps):
            if not isinstance(e_n, dict):
                continue
            if e_n.get('done') is True:
                cited = (_tally(n_n) or 0 for n_n in _wares(e_n.get('lines')))
                if any((c_n in evidence for c_n in cited)):
                    continue
            out_n.append(e_n)
        return out_n

    def _g2_gate_ask(self, args_n: Any, stated: list[str]) -> bool:
        longest = None
        for s_n in _strings_n(args_n):
            if longest is None or len(s_n) > len(longest):
                longest = s_n
        text_n = '' if longest is None else longest
        if not text_n:
            return False
        bundle = self.send_n([(1, '', text_n)], 1)
        if not bundle:
            return False
        self.barrier_texts += 1
        stated.append(f'Q: {text_n}\nA: {bundle[0][2]}')
        return True

    @staticmethod
    def _g2_gate_redo(stated: list[str], heard: list[str], grip: str) -> str:
        lead = ''
        if stated:
            lead = 'Before going on, NVIDIA did the steps the request asks for:\n' + '\n'.join(stated) + '\n'
        told = list(heard)
        if grip:
            told.append(grip)
        return lead + ("The shopper's reply changes the task: " + '; '.join(told) + '. Fix the cart for it, then check out again.')

    def _g2_gate_may_ask(self) -> bool:
        if not self.duties.get('ask'):
            return False
        return self.text_spent < self.text_bound and self.remaining() >= 2

    def _g2_gate_nudge(self, unproven_n: list[dict[str, Any]]) -> None:
        self._g2_log({'kind': 'gate_unproven', 'steps': _trim(unproven_n, 120)})
        names = []
        for e_n in unproven_n:
            names.append(str(e_n.get('step') or ''))
        left_texts = self.text_bound - self.text_spent
        note = f'(NVIDIA: no line made since the cart last changed at step {self.basket_at} shows ' + '; '.join(names)[:200]
        self.completed_log.append(note + f' done, and {left_texts} message(s) and {self.remaining()} steps are left: give the call that does it)')
        self.completed_moves.append(-1)

    def _g2_gate_step(self, name_n: str, args_n: Any, stated: list[str]) -> bool:
        reply = self.perform(name_n, args_n)
        if reply is None:
            return False
        seen = reply.get('observation')
        self.absorb_n(seen)
        at = self.move
        call_txt = _serial(args_n)
        shown = self.display(seen)
        self.completed_log.append(f'step {at}: {name_n} {call_txt} -> {shown[:300]} (cart then: {self.basket_sign()})')
        self.completed_moves.append(at)
        stated.append(f'{name_n} {call_txt} -> {shown[:400]}')
        return True

    def _g2_log(self, entry: dict[str, Any]) -> None:
        self.journal.append(entry)

    def _g2_halted(self) -> bool:
        return self.closed or self.remaining() <= 0

    def _g2_pane_fresh(self) -> bool:
        return self.basket_pane is not None and self.basket_at <= self.basket_viewed_at

    def _g2_nested_calls(self, traits: dict[str, Any], rows: dict[str, int]) -> tuple[bool, list[dict[str, Any]] | None]:
        for key, schema in traits.items():
            if not isinstance(schema, dict) or 'array' not in _sorts(schema):
                continue
            form = schema.get('items')
            if not isinstance(form, dict) or not isinstance(form.get('properties'), dict):
                continue
            interior = form['properties']
            id_key_n = _g2_first_key(interior, 'string', 'enum')
            tally_key = _g2_first_key(interior, 'integer')
            if not id_key_n:
                return (True, None)
            entries: list[dict[str, Any]] = []
            for i_n, n_n in rows.items():
                entry = {id_key_n: i_n}
                if tally_key:
                    entry[tally_key] = n_n
                entries.append(entry)
            return (True, [{key: entries}])
        return (False, None)

    def _g2_cart_goal(self, name_n: str, spec: dict[str, Any], strings: list[str]) -> str | None:
        for t_n, k_n in self.id_args_n:
            if name_n == t_n and k_n in strings:
                if k_n:
                    return k_n
                break
        for k_n in spec.get('required') or []:
            if k_n in strings:
                if k_n:
                    return k_n
                break
        if len(strings) == 1:
            return strings[0]
        return None

    def _g2_drop_call(self, name_n: str, traits: dict[str, Any], goal: str, manner: str | None, count: str | None, iid: str) -> dict[str, Any] | None:
        has = self.basket_rows.get(iid, 1)
        if manner:
            call = {goal: iid, manner: traits[manner]['enum'][1]}
            if count:
                call[count] = has
            return call
        if count and self.basket_collections(name_n):
            return {goal: iid, count: 0}
        if not count:
            return None
        if isinstance(traits[count].get('minimum'), int) and traits[count]['minimum'] <= -has:
            return {goal: iid, count: -has}
        return None

    def _g2_low_tag(self, iid: str, group_key: str, blank: str) -> str:
        return str((self.lines.get(iid) or {}).get(group_key) or blank).strip().lower()

    @staticmethod
    def _g2_money_slot(traits: dict[str, Any], key: str) -> bool:
        schema = traits.get(key) if isinstance(traits.get(key), dict) else {}
        blob = key + ' ' + str(schema.get('description') or '')
        return re.search('price|cost|amount|budget|spend|money', blob, re.I) is not None

    def _g2_costs_by_tag(self, group_key: str | None) -> dict[str, list[float]]:
        costs: dict[str, list[float]] = {}
        for iid, line in self.lines.items():
            cost = self.cost_of(iid)
            if cost is None:
                continue
            bucket = _g2_text_at(line, group_key, '').strip().lower() if group_key else ''
            costs.setdefault(bucket, []).append(cost)
        return costs

    def _g2_looked_choice(self, key: str, aisle: str, traits: dict[str, Any]) -> Any:
        for a_n in reversed(self.looked):
            if aisle in a_n and key in a_n and (not self._g2_phrased(a_n, traits, aisle)):
                return a_n[key]
        return None

    @staticmethod
    def _g2_phrased(a_n: dict[str, Any], traits: dict[str, Any], aisle: str | None) -> bool:
        for k_n, v_n in a_n.items():
            if aisle is not None:
                if k_n == aisle:
                    continue
                if isinstance(traits.get(k_n), dict) and 'enum' in traits[k_n]:
                    continue
            if isinstance(v_n, str) and v_n.strip() and re.search(PHRASE_TERMS, _snaky(k_n), re.I):
                return True
        return False

    def _g2_enum_choice(self, key: str, choices2: list[Any], learned_n: list[tuple[dict[str, Any], bool]], aisle: str, traits: dict[str, Any], lone: bool) -> Any:
        choice = None
        for manner, kept in learned_n:
            if kept and key in manner:
                choice = manner[key]
                break
        if choice is None:
            choice = self._g2_looked_choice(key, aisle, traits)
            called = [o_n for o_n in choices2 if isinstance(o_n, str) and re.search(LIST_MANNER, o_n, re.I)]
            worded_n = {a_n[key] for a_n in self.looked if key in a_n and self._g2_phrased(a_n, traits, None)}
            unworded_n = [o_n for o_n in choices2 if o_n not in worded_n]
            if choice is None:
                if len(called) == 1:
                    choice = called[0]
                elif lone and worded_n and (len(unworded_n) == 1):
                    choice = unworded_n[0]
        spoiled_n = {manner[key] for manner, kept in learned_n if not kept and key in manner} if lone else set()
        if spoiled_n and (choice is None or choice in spoiled_n):
            remainder = [o_n for o_n in choices2 if o_n not in spoiled_n]
            choice = remainder[0] if len(remainder) == 1 else None
        return choice

    def _g2_ceiling_origin(self, op: str, spec: Any, traits: dict[str, Any], ceiling: str, aisle: str) -> tuple[str, dict[str, Any], str, str] | None:
        learned_n = [(dict(manner), kept) for (key, manner), kept in self.ceiling_manners(op).items() if key == ceiling]
        choices_args = []
        for k_n, v_n in traits.items():
            if isinstance(v_n, dict) and isinstance(v_n.get('enum'), list) and (len(v_n['enum']) >= 2):
                choices_args.append(k_n)
        needed = (spec or {}).get('required') or []
        lone = choices_args[0] in needed if len(choices_args) == 1 else False
        origin = {}
        for key, schema in traits.items():
            if not isinstance(schema, dict):
                continue
            choices2 = schema.get('enum')
            if isinstance(choices2, list) and len(choices2) >= 2:
                picked = self._g2_enum_choice(key, choices2, learned_n, aisle, traits, lone)
                if picked is None:
                    return None
                origin[key] = picked
        return (op, origin, aisle, ceiling)

    def _g2_hunt_room(self, extra: int) -> int:
        return min(CHASE_MOVES, self.remaining() - self.end2() - CHASE_RETAIN - extra)

    def _g2_overtime(self) -> bool:
        return self.timer() > CHASE_BY + CHASE_GRACE

    def _g2_match_phrases(self, num: int) -> list[str]:
        couples = self.val_terms(num)
        term, terms = self.chase_terms(num)
        vals = [v_n for v_n, __n in couples]
        attempts = []
        for family_n in self.series_for_n(num):
            for val in vals[:1] or ['']:
                attempts.append(f'{family_n} {val} {term}'.strip())
        attempts.extend([f'{val} {term}'.strip() for val in vals])
        if term:
            attempts.extend(vals)
        attempts.extend([f'{term2} {term}'.strip() for term2 in terms[:CHASE_MATCH_TERMS]])
        if term:
            attempts.append(term)
        return attempts

    def _g2_match_pool(self, num: int, group_key: str | None, ceiling: Any, terms: set[str]) -> list[str]:
        ware2 = self.items[num - 1]
        reserve = []
        for i_n in self.viewed_sequence:
            if i_n in self.whole or not self.purchasable(i_n):
                continue
            if group_key and (not self.possesses(ware2, _g2_text_at(self.lines.get(i_n) or {}, group_key, ''))):
                continue
            reserve.append(i_n)
        if ceiling is not None:
            reserve = [i_n for i_n in reserve if (self.cost_of(i_n) or 0) <= ceiling]

        def rank_n(i_n: str) -> tuple[int, float]:
            cost = self.cost_of(i_n)
            return (-len(terms & _text_terms(_serial(self.lines.get(i_n) or {}))), float('inf') if cost is None else cost)
        reserve.sort(key=rank_n)
        return reserve[:CHASE_MATCH_OPENS]

    def _g2_match_targets(self, group_key: str | None) -> list[int]:
        kept = []
        if group_key:
            kept = [_g2_text_at(self.lines.get(i_n) or {}, group_key, '') for i_n in self.basket_iids() if (self.kept_measures(i_n) or 0) > 0]
        nums = [n_n for n_n, ware2 in enumerate(self.items, 1) if all((not self.possesses(ware2, tag) for tag in kept))]
        return nums or list(range(1, len(self.items) + 1))

    def _g2_match_search(self, nums: list[int], op: str, origin: dict[str, Any], slot: str, space: int) -> int | None:
        used_up = 0
        portion = max(2, space // max(1, len(nums)))
        for num in nums:
            spent = 0
            for phrase in dict.fromkeys(self._g2_match_phrases(num)):
                if used_up >= space or spent >= portion or self.closed or self._g2_overtime():
                    break
                args_n, issues = _g2_shape({**origin, slot: phrase}, self.specs.get(op), 'args')
                if issues:
                    continue
                reply = self.perform(op, args_n)
                used_up, spent = (used_up + 1, spent + 1)
                if reply is None:
                    return None
                self.perceive(op, reply.get('observation'))
        return used_up

    def _g2_chase_run(self, num: int, op: str, origin: dict[str, Any], slot: str, space: int, phrase: str, values: list[str], attempted: set[str], tally: list[Any]) -> bool:
        attempted.add(phrase.lower())
        args_n, issues = _g2_shape({**origin, slot: phrase}, self.specs.get(op), 'args')
        if issues:
            tally[4] = True
            return False
        if tally[0] >= space or self.closed:
            return False
        tally[3] = True
        low = phrase.lower()
        contains_vals = bool(values) and all((v_n.lower() in low for v_n in values))
        located = False
        walked_n: set[str] = set()
        while True:
            if self.closed or self._g2_overtime():
                return located
            sign = self.invoke_sign(op, args_n)
            if sign in walked_n:
                return located
            walked_n.add(sign)
            if sign in self.scan_invocations:
                viewed = self.scan_invocations[sign]
                fresh = CHASE_SHEET_FRESH
            elif tally[0] < space:
                prior = self.lines_of_ware(num)
                answer = self.perform(op, args_n)
                tally[0] += 1
                if answer is None:
                    return located
                viewed = answer.get('observation')
                self.perceive(op, viewed)
                fresh = self.lines_of_ware(num) - prior
                tally[1] += fresh
                tally[2] = 0 if fresh else tally[2] + 1
            else:
                return located
            lines = self.entries_in(viewed)
            if lines:
                located = True
            if contains_vals:
                self.val_lines.update([i_n for i_n, __n in lines])
            page = self.turning(op, viewed)
            if not page or page[3] is None:
                return located
            if fresh < CHASE_SHEET_FRESH and page[0] > CHASE_ENTIRE:
                return located
            args_n, issues = _g2_shape({**args_n, page[2]: page[3]}, self.specs.get(op), 'args')
            if issues:
                return located

    def _g2_chase_seed(self, num: int, values: list[str], term: str, run_n: Any, tally: list[Any]) -> str | None:
        attempts = []
        if values:
            attempts.append(' '.join(values + [term]).strip())
            attempts.append(' '.join(values))
        if len(values) >= 2:
            attempts.append(' '.join(values[:-1] + [term]).strip())
        if term:
            attempts.append(term)
        root = ''
        for phrase in dict.fromkeys(attempts):
            tally[3] = False
            if run_n(phrase):
                return phrase
            if not tally[3]:
                return None
            if phrase == term:
                self.chase_none_term.add(num)
                term = ''
        return root

    def _g2_set_partner(self, rows: list[tuple[Any, str, set[str], list[str]]], tag_codes: list[str], scheme2: Any, piece: int, unit: str, index: dict[str, Any]) -> str | None:
        for __n, i_n, matches, bars_n in rows:
            struck = any((_tag_code_strike(p_n, c_n) for p_n in matches for c_n in tag_codes))
            if struck and (not any((_tag_code_strike(p_n, c_n) for p_n in bars_n for c_n in tag_codes))):
                return i_n
            if scheme2 and scheme2[0] == piece and (self.tie_contains(i_n, unit, scheme2[2], index) is True):
                return i_n
        return None

    def _g2_clue_scheme(self, sort2: Any) -> dict[str, int]:
        scheme: dict[str, int] = {}
        taken = set()
        for num, iids in self.clue_wares.items():
            located = re.search('\\d+', str(num))
            if located:
                index = int(located.group())
            else:
                index = 1 if len(self.items) == 1 else None
            choice = None
            for i_n in iids:
                if not self.vended_out(i_n):
                    choice = i_n
                    break
            if not (choice and index and (1 <= index <= len(self.items))):
                continue
            label = sort2(choice)
            if label in taken:
                continue
            scheme[choice] = self.items[index - 1].amount
            taken.add(label)
        return scheme

    def _g2_set_cart(self, basket: str, invocations: list[dict[str, Any]]) -> list[str] | None:
        completed = []
        for args_n in invocations:
            args_n, issues = _g2_shape(args_n, self.specs.get(basket), 'args')
            if issues:
                return None
            answer = self.perform(basket, args_n)
            if answer is None or self.closed:
                return None
            self.memo_basket(basket, args_n)
            self.perceive(basket, answer.get('observation'))
            completed.append(_serial(args_n))
        return completed

    def _g2_order_once(self, purchase: str, args_n: Any, rows: dict[str, int], discard2: list[str], answers: list[str]) -> bool:
        go_n, stated = self.barrier('order')
        if stated:
            answers.append(stated)
        if not go_n:
            return False
        reply = self.perform(purchase, args_n)
        self._g2_log({'kind': 'completed', 'lines': rows, 'drop': discard2, 'reply': _trim(reply, 200)})
        if reply is None:
            return False
        self.purchases += 1
        if not reply.get('finished'):
            self.declined_at = self.move
        self.perceive(purchase, reply.get('observation'))
        answers.append(self.display(reply.get('observation')))
        return not (self.closed or reply.get('finished'))

    def _g2_checkout(self, purchase: str, args_n: Any, rows: dict[str, int], discard2: list[str]) -> list[str]:
        answers: list[str] = []
        if not self._g2_order_once(purchase, args_n, rows, discard2, answers):
            return answers
        self.rescan_offers()
        self.verify_basket(list(rows), SPEND_HARDN)
        if self.verdicts_n.get(self.verify_key(list(rows))) and self.remaining() >= 1 and (not self.closed):
            if not any((self.vended_out(i_n) for i_n in rows)):
                self._g2_order_once(purchase, args_n, rows, discard2, answers)
        return answers

    def _g2_forced_buy(self, purchase: str, again: bool) -> str | None:
        args_n, issues = _g2_shape(dict(self.digest.get('buy_args') or {}), self.specs.get(purchase), 'args')
        if issues:
            return None
        if self.remaining() >= 2:
            self.barrier('order')
        if self._g2_halted():
            return ''
        answer = self.perform(purchase, args_n)
        self._g2_log({'kind': 'forced_buy', 'reply': _trim(answer, 200)})
        if self._g2_halted():
            return ''
        if answer is None or not again:
            return None
        self.purchases += 1
        self.declined_at = self.move
        self.perceive(purchase, answer.get('observation'))
        return 'the checkout was refused: ' + self.display(answer.get('observation'))

    def _g2_forced_refuse(self, decline: str) -> str | None:
        args_n, issues = _g2_shape(self.quiet_n(decline, dict(self.digest.get('refuse_args') or {})), self.specs.get(decline), 'args')
        if issues:
            return None
        if self.remaining() >= 2:
            self.barrier('end')
        if self._g2_halted():
            return ''
        answer = self.perform(decline, args_n)
        self._g2_log({'kind': 'forced_refuse', 'reply': _trim(answer, 200)})
        if answer is None or self.closed:
            return None
        self.perceive(decline, answer.get('observation'))
        return 'ending without a purchase was refused: ' + self.display(answer.get('observation'))

    def _g2_whole_cart(self) -> Any:
        whole_n = self.populate_brief_basket() or not self.can_decline()
        if not whole_n:
            self._g2_log({'kind': 'short_cart_refused', 'lines': self.basket_iids()})
            return whole_n
        if len(self.basket_iids()) < 2:
            return whole_n
        self.populate_details(self.basket_iids(), self.end2())
        if self.remaining() >= 3 and self.engine_space() > 5:
            self.display_ties(self.basket_iids(), self.end2())
        return whole_n

    def _g2_open_pointer(self) -> bool:
        active = set()
        for __n, demand in self.demands_now():
            active.add(demand)
        for __n, demand in self.pointers:
            if demand in active and demand not in self.pointer_vals:
                return True
        return False

    def _g2_vague_demand(self, demand: str) -> bool:
        if demand in (m_n for __n, m_n in self.pointers):
            return True
        return demand in (str(x_n) for x_n in _wares(self.digest.get('lookups')))

    def _g2_probe(self, op: str, ask_args: dict[str, Any]) -> bool:
        args_n, issues = _g2_shape(ask_args, self.specs.get(op), 'args')
        if issues:
            return False
        answer = self.perform(op, args_n)
        if answer is None:
            return False
        self.perceive(op, answer.get('observation'))
        return True

    def _g2_listed(self, rows: list[Any]) -> list[tuple[str, dict[str, Any]]]:
        out_n: list[tuple[str, dict[str, Any]]] = []
        for r_n in rows:
            if not isinstance(r_n, dict):
                continue
            i_n = self.iid(r_n, True)
            if i_n:
                out_n.append((i_n, r_n))
        return out_n

    @staticmethod
    def _g3_int(val: Any) -> bool:
        if isinstance(val, bool):
            return False
        return isinstance(val, int)

    @staticmethod
    def _g3_num(val: Any) -> bool:
        if isinstance(val, bool):
            return False
        return isinstance(val, (int, float))

    @staticmethod
    def _g3_fit(spec: Any, wanted: dict[str, Any], locus: str) -> tuple[Any, list[str]]:
        faults = []
        shaped = _form(wanted, spec, locus, faults)
        if faults:
            return (None, faults)
        return (shaped, faults)

    @staticmethod
    def _g3_with(base: dict[str, Any], slot: Any, val: Any) -> dict[str, Any]:
        merged = dict(base)
        merged[slot] = val
        return merged

    def _g3_slot(self, raw: Any) -> int:
        pick = _tally(raw) or 1
        if pick < 1:
            pick = 1
        top = len(self.items)
        return top if pick > top else pick

    def _g3_record_ids(self, obs: Any) -> list[str]:
        found = []
        stack = [(obs, 0)]
        while stack:
            leaf, level = stack.pop()
            if level > 6:
                continue
            if isinstance(leaf, list):
                kids = leaf
            elif isinstance(leaf, dict):
                iid = self.iid(leaf, False, level)
                if iid and _detail_chunk(leaf) and (iid not in found):
                    found.append(iid)
                    continue
                kids = list(leaf.values())
            else:
                continue
            stack.extend(((kid, level + 1) for kid in reversed(kids)))
        return found

    def _g3_evidence_piece(self, chunk: dict[str, Any], num: int, demand: str, several: bool) -> str | None:
        trait = demand.split(':', 1)[0].strip()
        wanted = _root(trait)
        strikes = []
        for k_n, v_n in chunk.items():
            if _names_n(wanted, _root(k_n)):
                strikes.append(f'{k_n}={str(v_n)[:80]}')
        if not strikes:
            return None
        head = f'(item {num}) ' if several else ''
        return f'{head}{trait} -> ' + ', '.join(strikes[:3])

    def _g3_sheet_room(self, steps: int) -> bool:
        if self.sheets_scan + steps > AUTO_SHEETS:
            return False
        return self.remaining() - self.end2() - steps >= 12

    @staticmethod
    def _g3_last_page(form: Any, obs: dict[str, Any]) -> bool:
        if not form or not form.get('count'):
            return False
        if form.get('next') is None:
            return False
        return obs.get(form['next']) is None

    @staticmethod
    def _g3_flat_text(root: Any) -> str:
        bits = []
        pile = [root]
        while pile:
            leaf = pile.pop()
            if isinstance(leaf, dict):
                pile.extend(leaf.values())
            elif isinstance(leaf, list):
                pile.extend(leaf)
            else:
                bits.append(str(leaf))
        return ' '.join(bits).lower()

    def _g3_tie_search(self, about_n: str, terms: str, jot: str, scheme: Any) -> list[str]:
        learned = []
        op, origin, slot = scheme
        for phrase in dict.fromkeys((terms, ' '.join(terms.split()[:2]))):
            args_n, issues = self._g3_fit(self.specs.get(op), self._g3_with(origin, slot, phrase), 'args')
            if issues:
                break
            self.tie_searches += 1
            answer = self.perform(op, args_n)
            if answer is None:
                break
            obs = answer.get('observation')
            self.perceive(op, obs)
            self.tie_viewed[jot] = [i_n for i_n, __n in self.entries_in(obs)]
            learned = self.store_lines(about_n, self.tie_viewed[jot])
            self.journal.append({'kind': 'link_search', 'words': phrase[:80], 'rows': learned[:4]})
            if self.tie_viewed[jot]:
                break
        return learned

    def _g3_of_sort(self, iids: list[str], sort2: set[str]) -> list[str]:
        if not sort2:
            return list(iids)
        kept = []
        for i_n in iids:
            if sort2 & set(re.findall('[a-z0-9]{3,}', _serial(self.lines.get(i_n) or {}).lower())):
                kept.append(i_n)
        return kept

    @staticmethod
    def _g3_swap_calls(goal: str, manner: str, count: str, options: list[Any], iid: str, has: int, desire: int) -> list[dict[str, Any]]:
        choices2 = list(map(str, options))
        off_n = choices2[1]
        for o_n in choices2:
            if re.search('remov|delet|drop|clear|take', o_n, re.I):
                off_n = o_n
                break
        on_n = choices2[0]
        for o_n in choices2:
            if o_n != off_n:
                on_n = o_n
                break
        return [{goal: iid, manner: off_n, count: has}, {goal: iid, manner: on_n, count: desire}]

    def _g3_learn_need(self, record: dict[str, Any], several: bool, heard: list[str]) -> None:
        num = self._g3_slot(record.get('item'))
        ware2 = self.items[num - 1]
        demand = str(record['need']).strip()
        if demand not in ware2.heard:
            ware2.heard.append(demand)
            heard.append(f'item {num}: {demand}' if several else demand)
        if record.get('points') is not True:
            return
        if (num, demand) not in self.pointers:
            self.pointers.append((num, demand))

    def _g3_memo_drop(self, record: dict[str, Any]) -> str:
        ware2 = self.items[self._g3_slot(record.get('item')) - 1]
        goal = str(record['need']).strip()
        active = [m_n for m_n in ware2.needs + ware2.heard if m_n not in ware2.lifted]
        strike = [goal] * active.count(goal)
        if not strike:
            head_n = _terms(goal.split(':')[0])
            strike = next(([m_n] for m_n in active if _terms(m_n.split(':')[0]) == head_n), [])
        ware2.lifted.extend(strike or [goal])
        return goal

    def _g3_memo_add(self, record: dict[str, Any]) -> str:
        ware2 = self.items[self._g3_slot(record.get('item')) - 1]
        demand = str(record['need']).strip()
        if demand in ware2.lifted:
            ware2.lifted.remove(demand)
        if demand not in ware2.needs + ware2.heard:
            ware2.heard.append(demand)
        return demand

    @staticmethod
    def _g3_holder_score(terms: str, lengthy: set[str], ware2: Any) -> float:
        if _terms(ware2.aisle or '') == terms:
            return 3.0
        if _terms(ware2.name_n) == terms:
            return 2.0
        ours = _bases(ware2.aisle or '') | _bases(ware2.name_n)
        return len(lengthy & ours) / max(1, len(lengthy | ours))

    def _g3_tag_counts(self, reserve: dict[str, Any], group_key: str, ware2: Any) -> dict[str, int]:
        tallies = {}
        for entry in reserve.values():
            tag = str(entry.get(group_key) or '')
            if not tag or not self.possesses(ware2, tag):
                continue
            label = _terms(tag)
            tallies[label] = 1 + tallies.get(label, 0)
        return tallies

    def _g3_open_phrases(self, ware2: Any) -> list[str]:
        pointed = [demand for __n, demand in self.pointers]
        desired = []
        for m_n in ware2.needs:
            if ':' not in m_n or m_n in pointed:
                continue
            w_n = m_n.split(':', 1)[1].strip()
            if w_n and len(w_n.split()) <= 3 and (not re.search(OPEN_NOTN, w_n, re.I)):
                desired.append(w_n)
        words = ware2.name_n.split()
        term = ware2.name_n if len(words) <= 3 else ' '.join(ware2.name_n.split()[-2:])
        lead = [' '.join(desired[:2] + [term])] if desired else []
        return list(dict.fromkeys(lead + [term]))

    def _g3_open_item(self, op: str, origin: dict[str, Any], slot: str, ware2: Any, num: int, halt_at: int, displayed: list[str]) -> bool:
        for phrase in self._g3_open_phrases(ware2):
            args_n, issues = self._g3_fit(self.specs.get(op), self._g3_with(origin, slot, phrase), 'args')
            if issues or self.closed or self.move >= halt_at:
                return True
            answer = self.perform(op, args_n)
            if answer is None:
                return False
            viewed = answer.get('observation')
            self.perceive(op, viewed)
            if self.entries_in(viewed):
                self.looked = (self.looked + [dict(args_n)])[-8:]
                displayed.append(f'{phrase}: ' + self.display(viewed))
                displayed.extend(self.extra_lines_for_fields(op, args_n, viewed, halt_at, num))
                if self.lines_of_ware(num) >= OPEN_EACHN:
                    return True
                continue
            note = 'the store refused the call: ' + self.display(viewed) if _failed_n(viewed) else 'no rows'
            displayed.append(f'{phrase}: {note}')
        return True

    def _g3_open_widen(self, halt_at: int) -> None:
        start = self.move
        located = [n_n for n_n in range(1, len(self.items) + 1) if self.lines_of_ware(n_n)]
        while self.move - start < self.max_moves // OPEN_PORTION and self.move < halt_at and (not self.closed):
            counts = [self.entries_of_ware(n_n) for n_n in located]
            if all((c_n >= OPEN_EACHN for c_n in counts)):
                return
            held = len(self.whole)
            self.widen_n()
            if held == len(self.whole):
                return

    def _g3_may_ask(self) -> bool:
        return self.texts_open() > 0 and (not self.closed) and (self.remaining() > 6) and (self.timer() < INQUIRY_BY)

    def _g3_next_bundle(self, groups: list[Any], heard: list[str]) -> list[tuple[int, str, str]]:
        bundle = []
        while not bundle and groups and self.offer:
            bundle = self.query_cluster(groups.pop(0), heard)
        return bundle or self.query_following(heard)

    @staticmethod
    def _g3_cache_marks(messages_n: list[dict[str, Any]]) -> list[dict[str, Any]]:
        copied = list(map(dict, messages_n))
        for index in (0, len(copied) - 1):
            substance = copied[index].get('content')
            if isinstance(substance, str) and substance:
                copied[index]['content'] = [{'type': 'text', 'text': substance, 'cache_control': {'type': 'ephemeral'}}]
        return copied

    def _g3_engine_ok(self, engine: str, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, Any]:
        usage_n = payload.get('usage') or {}
        if isinstance(usage_n.get('cost'), (int, float)):
            self.expense += float(usage_n['cost'])
        note = payload['choices'][0].get('message') or {}
        if not (note.get('content') or note.get('tool_calls')):
            self.journal.append({'kind': 'llm_fail', 'model': engine, 'status': 0})
            return (None, 0)
        return (note, 200)

    def _g3_plain_default(self, name_n: str, key: str, val: Any, schema: dict[str, Any]) -> bool:
        if val in (schema.get('minimum'), schema.get('default')):
            return True
        if val != 1 or key != self.sheet_args.get(name_n):
            return False
        return schema.get('minimum') is None and schema.get('default') is None

    @staticmethod
    def _g3_holds(args_n: dict[str, Any], fixed: dict[str, Any]) -> bool:
        for k_n, v_n in fixed.items():
            if args_n.get(k_n) != v_n:
                return False
        return True

    def _g3_open_target(self, invoke: dict[str, Any], opener_n: str, goal: str) -> str | None:
        fn_n = invoke.get('function') or {}
        args_n = _json_obj_n(fn_n.get('arguments')) if isinstance(fn_n.get('arguments'), str) else None
        iid = str((args_n or {}).get(goal) or '').strip()
        if fn_n.get('name') != opener_n or not args_n or set(args_n) != {goal}:
            return None
        if iid not in self.lines or iid in self.vanished:
            return None
        return iid

    def _g3_pane_count(self, iid: str) -> int | None:
        pile = [self.basket_pane]
        while pile:
            leaf = pile.pop()
            if isinstance(leaf, list):
                pile.extend(leaf)
                continue
            if not isinstance(leaf, dict):
                continue
            if any((isinstance(v_n, str) and v_n.strip() == iid for v_n in leaf.values())):
                tallies = [_tally(v_n) for k_n, v_n in leaf.items() if not isinstance(v_n, str) and _tally(v_n) is not None and re.search('\\b(?:qty|quantity|count|units?)\\b', _terms(k_n))]
                if len(tallies) == 1:
                    return tallies[0]
            pile.extend([v_n for v_n in leaf.values() if isinstance(v_n, (dict, list))])
        return None

    @staticmethod
    def _g3_grid_note(cell: dict[str, Any], iid: str, val: Any) -> None:
        cell['n'] += 1
        raw = str(val).strip()
        text_n = raw.lower()
        cell['per'][iid] = text_n
        cell['plain'][iid] = _INITIAL_SPELLING.get(raw, text_n).strip().lower()
        if text_n not in cell['values']:
            cell['values'].append(text_n)

    def _g3_grid_collect(self, group_key: str | None) -> dict[str, dict[str, Any]]:
        groups = {}
        for iid in self.viewed_sequence:
            entry = self.whole.get(iid)
            if not entry:
                continue
            chunk = _detail_chunk(entry) or {}
            tag = str(entry.get(group_key) or 'listings') if group_key else 'listings'
            group = groups.setdefault(tag, {'n': 0, 'keys': {}})
            group['n'] += 1
            completed = set()
            for key, val in chunk.items():
                name_n = _terms(key)
                if name_n and name_n not in completed:
                    completed.add(name_n)
                    self._g3_grid_note(group['keys'].setdefault(name_n, {'n': 0, 'values': [], 'per': {}, 'plain': {}}), iid, val)
        return groups

    @staticmethod
    def _g3_recurring(values: list[str]) -> int:
        total = 0
        for v_n in values:
            others = ' '.join((o_n for o_n in values if o_n is not v_n))
            if set(re.findall('[a-z]{3,}', v_n)) & set(re.findall('[a-z]{3,}', others)):
                total += 1
        return total

    def _g3_grid_fields(self, group: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        retained = []
        for name_n, cell in sorted(group['keys'].items(), key=lambda kv_n: (-kv_n[1]['n'], len(kv_n[0]))):
            values = list(cell['plain'].values())
            if not values:
                continue
            span = len(values)
            mean_n = sum(map(len, values)) / span
            if mean_n > 40 or len([v_n for v_n in values if ' / ' in v_n]) * 2 >= span:
                continue
            if span >= 4 and len(set(values)) == span and (mean_n > 12) and (self._g3_recurring(values) * 2 < span):
                continue
            double = None
            for other_n, schema in retained:
                if self.doubles(other_n, schema, name_n, cell):
                    double = schema
                    break
            if double is None:
                retained.append((name_n, cell))
                continue
            for iid, text_n in cell['per'].items():
                double['per'].setdefault(iid, text_n)
                double['plain'].setdefault(iid, cell['plain'][iid])
                if text_n not in double['values']:
                    double['values'].append(text_n)
            double['n'] = len(double['per'])
        retained.sort(key=lambda kv_n: kv_n[1]['n'], reverse=True)
        return retained

    def _g3_row_id(self, val: Any, key: str, title: str, traits: dict[str, Any]) -> Any:
        if not isinstance(val, str):
            return val
        bare = val.strip()
        if bare in self.lines or len(bare) < 3 or len(bare) > 30:
            return val
        if not re.search('\\d', val) or 'enum' in (traits.get(key) or {}):
            return val
        pattern = '(?<![a-z0-9])' + re.escape(bare.lower()) + '(?![a-z0-9])'
        strikes = [i_n for i_n, line in self.lines.items() if re.search(pattern, str(line.get(title) or '').lower())]
        if len(strikes) != 1:
            return val
        self.journal.append({'kind': 'row_id', 'sent': val, 'id': strikes[0]})
        return strikes[0]

    @staticmethod
    def _g3_below(cost: Any) -> int:
        if float(cost).is_integer():
            return int(cost) - 1
        return int(cost)

    def _g3_tail_page(self, op: str, args_n: dict[str, Any], page: Any) -> None:
        if page[3] is None:
            return
        invoke, issues = self._g3_fit(self.specs.get(op), self._g3_with(args_n, page[2], page[3]), 'args')
        if issues or self.remaining() <= self.end2() + 7 or self.hurried():
            return
        extra = self.perform(op, invoke)
        if extra is not None:
            self.perceive(op, extra.get('observation'))

    def _g3_stroll(self, op: str, origin: dict[str, Any], aisle: str, ceiling: str, row: str, cost: Any, tag: str) -> bool:
        bound = self._g3_below(cost)
        attempted = set()
        while len(attempted) < STROLL_INVOCATIONS:
            if self.remaining() <= self.end2() + (STROLL_SPACE if attempted else 7) or self.closed or self.hurried():
                return True
            if bound < 1 or bound in attempted:
                return False
            attempted.add(bound)
            wanted = dict(origin)
            wanted[aisle] = tag
            wanted[ceiling] = bound
            args_n, issues = self._g3_fit(self.specs.get(op), wanted, 'args')
            if issues:
                return True
            answer = self.perform(op, args_n)
            if answer is None:
                return True
            viewed = answer.get('observation')
            self.perceive(op, viewed)
            lines = self.entries_in(viewed)
            self.journal.append({'kind': 'list_below', 'line': row, 'price': cost, 'limit': bound, 'rows': len(lines)})
            page = self.turning(op, viewed)
            if not lines or any(((self.cost_of(i_n) or 0) > bound for i_n, __n in lines)):
                return False
            if not page or page[0] <= page[1] or self.cost_sorted(viewed):
                return False
            if page[0] <= 2 * page[1]:
                self._g3_tail_page(op, args_n, page)
                return False
            prices = [p_n for p_n in map(self.cost_of, [i_n for i_n, __n in lines]) if p_n is not None]
            if not prices:
                return False
            bound = self._g3_below(min(prices))
        return False

    def _g4_worded(self, sent, key):
        value = sent.get(key)
        return isinstance(value, str) and re.search('[a-z]', value, re.I) is not None

    def _g4_section_labels(self):
        key = self.section_key()
        if not key:
            return set()
        labels = {str(row.get(key) or '').strip().lower() for row in self.lines.values()}
        return labels - {''}

    def _g4_free_slot(self, key, labels):
        values = [sent.get(key) for sent in self.looked]
        if len({str(value) for value in values if value}) < 2:
            return False
        if not any((len(str(value).split()) >= 2 for value in values)):
            return False
        for value in values:
            if str(value or '').strip().lower() in labels:
                return False
        return True

    def _g4_place_terms(self):
        places = {_terms(thing.aisle or '') for thing in self.items}
        places.update((_terms(thing.name_n) for thing in self.items))
        key = self.section_key()
        if key:
            places.update((_terms(str(row.get(key) or '')) for row in self.lines.values()))
        return places

    def _g4_unplaced(self, value, places):
        if _terms(value) in places:
            return False
        mine = {word for word in _root(value) if len(word) >= 4}
        for other in places:
            if other and mine & {word for word in _root(other) if len(word) >= 4}:
                return False
        return True

    def _g4_kept_args(self, sent, slot, traits, spec):
        kept = {}
        for key, value in sent.items():
            if key == slot:
                continue
            if isinstance((traits.get(key) or {}).get('enum'), list) or key in (spec.get('required') or []):
                kept[key] = value
        places = self._g4_place_terms()
        kept = {key: value for key, value in kept.items() if key in (spec.get('required') or []) or not isinstance(value, str) or self._g4_unplaced(value, places)}
        worded = [past for past in self.looked if isinstance(past.get(slot), str) and re.search('[a-z]', past[slot], re.I)]
        for key in list(kept):
            if not isinstance((traits.get(key) or {}).get('enum'), list):
                continue
            votes = {}
            earliest = {}
            for past in worded:
                if key not in past:
                    continue
                mark = _serial(past[key])
                if mark not in votes:
                    votes[mark] = 0
                    earliest[mark] = past[key]
                votes[mark] += 1
            if votes:
                best = max(votes.values())
                kept[key] = next((earliest[mark] for mark in votes if votes[mark] == best))
        return kept

    def _g4_cheaper(self, basket, group_key, label, price):
        found = []
        for iid in self.viewed_sequence:
            if iid in basket or iid in self.whole or (not self.purchasable(iid)):
                continue
            if str((self.lines.get(iid) or {}).get(group_key) or '').strip().lower() != label:
                continue
            other = self.cost_of(iid)
            if (float('inf') if other is None else other) < price:
                found.append(iid)
        found.sort(key=lambda iid: self.cost_of(iid) or 0.0)
        head = next((iid for iid in found if iid in self.val_lines), None)
        if head is None:
            return found
        return [head] + [iid for iid in found if iid != head]

    def _g4_fits(self, iid, index, codes):
        if not any((_tag_code_strike(named, code) for named in self.match_set(iid, index) for code in codes)):
            return False
        barred = set(index['marks'].get(iid, {}).get('bars') or [])
        return not any((_tag_code_strike(named, code) for named in barred for code in codes))

    def _g4_pair_issues(self, piece, primary, basket, index):
        owned = self.ware_iids(primary) if basket else []
        owners = [iid for iid in basket if iid in owned]
        fitted = self.ware_iids(piece) if basket else []
        parts = [iid for iid in basket if iid in fitted]
        if len(owners) != 1 or not parts:
            return []
        unit = owners[0]
        linked = self.tie_issues(piece, primary, unit, parts, index)
        if linked:
            return linked
        codes = set(index['marks'].get(unit, {}).get('codes') or [])
        if not codes:
            return []
        out = []
        for row in parts:
            matches = self.match_set(row, index)
            if not matches or self._g4_fits(row, index, codes):
                continue
            fine2 = []
            for iid in self.ware_iids(piece):
                if len(fine2) == 4:
                    break
                if self._g4_fits(iid, index, codes) and (not self.vended_out(iid)) and (iid not in basket):
                    fine2.append(iid)
            head = f"{row} ({self.items[piece - 1].name_n}) works with {', '.join(sorted(matches)[:4])}, not with {unit} ({', '.join(sorted(codes))})"
            if fine2:
                head += f"; seen parts of that kind that work with it: {', '.join(fine2)}"
            else:
                head += '; no seen part of that kind works with it, so look for one or switch the device: ' + self.match_scheme(short=True)
            out.append(head)
        return out

    @staticmethod
    def _g4_flat_text(entry):
        plain = [str(value) for value in entry.values() if isinstance(value, str)]
        keyed = []
        for chunk in entry.values():
            if isinstance(chunk, dict):
                keyed.extend((f'{k_n} {v_n}' for k_n, v_n in chunk.items() if isinstance(v_n, str)))
        return ' '.join(plain) + ' ' + '. '.join(keyed)

    def _g4_note_fits(self, text_n):
        found = self.note_args(text_n)
        return bool(found) and _says_n(found, text_n)

    @staticmethod
    def _g4_bump(tally, code, part):
        per = tally.setdefault(code, {})
        per[part] = per.get(part, 0) + 1

    def _g4_devices(self, primary, code, index):
        hits = []
        for iid in self.ware_iids(primary):
            if len(hits) == 2:
                break
            if code in ((index['marks'].get(iid) or {}).get('named') or []) or any((_tag_code_strike(code, c_n) for c_n in index['marks'].get(iid, {}).get('codes') or [])):
                hits.append(iid)
        return hits

    def _g4_scheme_rows(self, primary, index):
        parts = [p_n for p_n, m_n in self.limited if m_n == primary]
        stock, gone = (self.ware_iids, self.vended_out)
        tally = {}
        for part in parts:
            for iid in [i_n for i_n in stock(part) if not gone(i_n)]:
                for code in self.match_set(iid, index):
                    self._g4_bump(tally, code, part)
        marks = index['marks']
        for part in parts:
            plan = self.tie_scheme(part, primary, index)
            if plan and plan[0] == part:
                for iid in [i_n for i_n in stock(part) if not gone(i_n)]:
                    for code in set((marks.get(iid) or {}).get('fam') or []) & plan[2]:
                        self._g4_bump(tally, code, part)
        ranked = sorted(tally.items(), key=lambda kv_n: (-len(kv_n[1]), -sum(kv_n[1].values())))[:6]
        seen = {code: self._g4_devices(primary, code, index) for code, __n in ranked}
        names = self.items
        pieces = []
        for code, per in ranked:
            counts = ', '.join([f'{n_n} {names[p_n - 1].name_n}' for p_n, n_n in sorted(per.items())])
            where = f" (device seen: {', '.join(seen[code])})" if seen[code] else ' (device not seen yet)'
            pieces.append(f'{code}: ' + counts + where)
        out = []
        text_n = '; '.join(pieces)
        if text_n:
            out.append(f"codes the {names[primary - 1].name_n} candidates' parts name -> {text_n}")
        collections = self.cheapest_collections(primary, parts, index) if self.digest.get('pick') == 'cheapest' else ''
        if collections:
            out.append(collections)
        return out

    @staticmethod
    def _g4_removes(schema, args_n):
        for key, val in args_n.items():
            options = (schema.get(key) or {}).get('enum')
            if isinstance(options, list) and len(options) >= 2 and (val in options[1:]):
                return True
        return False

    def _g4_cart_line(self, row):
        iid = next((v_n for v_n in row.values() if isinstance(v_n, str) and v_n in self.lines), None)
        count = next((_tally(v_n) for v_n in row.values() if _tally(v_n) is not None and (not isinstance(v_n, str))), 1)
        return (iid, count)

    def _g4_cart_rows(self, schema, args_n):
        lines = self.lines
        loose = next((_tally(v_n) for v_n in args_n.values() if _tally(v_n) is not None and (not isinstance(v_n, str))), None)
        rows = []
        entire = False
        for key, val in args_n.items():
            if isinstance(val, str):
                if val in lines:
                    rows.append((val, loose))
            elif isinstance(val, list):
                form = (schema.get(key) or {}).get('items') or {}
                if not val and isinstance(form, dict) and (form.get('type') == 'object' or 'properties' in form):
                    entire = True
                for row in val:
                    if isinstance(row, dict):
                        entire = True
                        line = self._g4_cart_line(row)
                        if line[0]:
                            rows.append(line)
                    elif isinstance(row, str) and row in lines:
                        rows.append((row, loose))
        return (rows, entire)

    @staticmethod
    def _g4_new_count(has, count, removing, collections, schema):
        if removing:
            if any(('integer' in _sorts(v_n) for v_n in schema.values() if isinstance(v_n, dict))):
                return has - (count or 1)
            return has - count if count else 0
        if collections:
            return count if count is not None else 1
        return has + (count if count is not None else 1)

    def _g4_unscanned(self, name_n):
        if name_n in self.sheet_ops():
            return True
        return any((self.duties.get(r_n) == name_n for r_n in ('ask', 'cart', 'cart_view', 'buy', 'refuse')))

    @staticmethod
    def _g4_note_slot(reserve, traits, text_n):
        best = None
        rank_best = None
        for key in reserve:
            rank = (not re.search(NOT_NOTE, _snaky(key), re.I), bool(re.search(NOTE_SLOT, _snaky(key), re.I)), traits[key].get('maxLength') or 10 ** 6)
            if rank_best is None or rank > rank_best:
                best, rank_best = (key, rank)
        return {best: text_n}

    @staticmethod
    def _g4_note_nested(traits, text_n):
        for key, schema in traits.items():
            inner = None
            if isinstance(schema, dict):
                inner = next((k_n for k_n, v_n in (schema.get('properties') or {}).items() if isinstance(v_n, dict) and 'string' in _sorts(v_n)), None)
                if 'array' in _sorts(schema):
                    return {key: [text_n]}
            if inner:
                return {key: {inner: text_n}}
        return {}

    def _g4_offer_scope(self, remaining, ends_n):
        if remaining <= self.end2():
            return set(ends_n)
        if remaining <= self.end2() + 3:
            return set(ends_n) | {self.duties.get('cart'), self.duties.get('cart_view')}
        if remaining <= len(self.items) + 5 and self.purchases == 0:
            return set(ends_n) | {self.duties.get('cart'), self.duties.get('cart_view'), self.duties.get('offer')}
        return None

    def _g4_open_plan(self, lines):
        several = self.contrast_goal()
        joiner_n = self.duties.get('compare')
        op = self.details_op()
        goal = self.lone_goal(op)
        if several and joiner_n and (not self.contrast_plain()):
            return [(joiner_n, {several: lines[:self.contrast_space()]})]
        if op and goal:
            return [(op, {goal: i_n}) for i_n in lines[:2]]
        return None

    def _g4_walk_plan(self, scheme):
        for name_n, args_n in scheme:
            issues = []
            shaped = _form(args_n, self.specs.get(name_n), 'args', issues)
            if issues:
                return False
            answer = self.perform(name_n, shaped)
            if answer is None or self.closed:
                return True
            self.perceive(name_n, answer.get('observation'))
        return True

    def _g4_trail_budget(self, halt):
        limit = self.path_space or 6
        rounds = 6 if self.remaining() > 15 else 4
        if self.trails_n >= min(rounds, limit) or not SPEND_HARDN > self.expense:
            return False
        return True if halt is None else self.timer() <= halt

    def _g4_trail_records(self):
        seq = self.viewed_sequence[::-1]
        whole, lines, scan = (self.whole, self.lines, self.sheet_scan)
        opened = [whole.get(iid) or lines.get(iid) for iid in seq if scan(iid)]
        records = ['OPENED ' + _serial(_trim(entry, 300))[:1400] for entry in opened if entry][:45]
        for iid in seq:
            if len(records) >= 60:
                break
            if scan(iid) and (whole.get(iid) or lines.get(iid)):
                continue
            records.append('ROW ' + (_serial(_trim(whole[iid], 120))[:700] if iid in whole else _serial(_trim(lines.get(iid), 80))[:300]))
        return records

    def _g4_trail_round(self, waiting):
        records = self._g4_trail_records()
        heavy = self.expense + HEAVY_SPACE <= SPEND_HARDN
        prompt = PATH + '\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n' + '\n'.join(waiting) + ('\n\nRECORDS (newest first):\n' + '\n'.join(records))
        obtained = self.query_json(prompt, [LIGHTN, HEAVYN] if heavy else [LIGHTN], 4500) or {}
        self.journal.append({'kind': 'trail', 'pending': waiting, 'got': _trim(obtained, 150)})
        reach_n, missed, gaps_n = ([], [], [])
        for series in [s_n for s_n in _wares(obtained.get('chains')) if isinstance(s_n, dict)]:
            demand = self._g4_chain_demand(series, waiting)
            if demand:
                self._g4_take_chain(series, demand, reach_n, missed, gaps_n)
        return (reach_n, missed, gaps_n)

    @staticmethod
    def _g4_chain_demand(series, waiting):
        raw = series.get('need')
        need = str(raw if raw else '').strip()
        if need in waiting:
            return need
        return waiting[0] if len(waiting) == 1 else None

    @staticmethod
    def _g4_chain_value(series, last, last_id):
        if series.get('done') is True and str(series.get('value') or '').strip():
            return str(series['value']).strip()
        if str(last.get('find') or '').strip():
            return None
        return str(series.get('value') or '').strip() or f'as stated on {last_id}'

    def _g4_take_chain(self, series, demand, reach_n, missed, gaps_n):
        ties = [l_n for l_n in _wares(series.get('links')) if isinstance(l_n, dict)]
        iids = []
        for tie in ties:
            raw = tie.get('id')
            iids.append(str(raw if raw else '').strip())
        scan = self.sheet_scan
        if len(ties) > 0 and all(map(scan, iids)):
            found = self._g4_chain_value(series, ties[-1], iids[-1])
            if found is not None:
                self.pointer_vals[demand] = found
                self.edition += 1
                return
        lines = self.lines
        closed_n = [i_n for i_n in iids if i_n in lines and (not scan(i_n))]
        if closed_n:
            reach_n += [i_n for i_n in closed_n if i_n not in reach_n]
            return
        margin = {}
        for tie, iid in zip(ties, iids):
            if iid not in lines:
                margin = tie
                break
        about_n = str(margin.get('about') or '') or 'the listing it points at'
        terms = str(margin.get('find') or '').strip()
        hint = f' (search: {terms})' if terms else ''
        missed.append(f'{demand} -> find {about_n}' + hint)
        gaps_n.append((about_n, terms))

    def _g4_follow(self, op, goal, targets, viewed):
        spec = self.specs.get(op)
        for iid in targets:
            problems = []
            shaped = _form({goal: iid}, spec, 'args', problems)
            if problems or self.hurried():
                return
            reply = self.perform(op, shaped)
            if reply is None or self.closed:
                return
            self.perceive(op, reply.get('observation'))
            viewed.append(f'{iid}: ' + self.display(reply.get('observation'))[:1500])

    def _g4_path_report(self, viewed, missed):
        parts = []
        if viewed:
            parts += ['[NVIDIA opened the pages a requirement points at]\n', '\n'.join(viewed), '\n']
        vals = self.pointer_vals
        if vals:
            parts += ['[values read from the pages requirements point at] ', '; '.join([f'{k_n} -> {v_n}' for k_n, v_n in vals.items()]), '\n']
        left = self.remaining()
        grip = bool(missed) and self.path_chunks < (3 if left > 15 else 2) and (left > 5)
        if grip:
            self.path_chunks += 1
            parts += ['. '.join(('The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing', 'Still to find and open: ')), ' | '.join(missed[:3]), '. '.join(('', "Search for it by its shop's name together with the values its description gives (the shop's other listings of that kind are not it), open the row of that shop whose page shows those values, apply the value its page gives, then check out."))]
        return (''.join(parts), grip)

    def _g4_pace(self):
        wait_n = self.world_at + self.world_margin - time.monotonic()
        if wait_n > 0:
            time.sleep(wait_n)
        self.world_at = time.monotonic()

    def _g4_timed_post(self, payload_n):
        began_n = time.monotonic()
        outcome = _world.post_verbose('/environment/call', payload_n)
        self.world_used_up += time.monotonic() - began_n
        self.world_tally += 1
        return outcome

    def _g4_retry_after(self, failure, payload_n):
        state = failure.get('status')
        core = str(failure.get('body') or '')
        self.journal.append({'kind': 'env_fail', 'status': state, 'body': core[:200]})
        if state in (400, 401, 402, 403, 404, 422):
            if state == 402:
                self.broke_n = True
            return False
        if state != 409:
            return True
        small = core.lower()
        if 'finish' in small or 'quarantin' in small or 'finaliz' in small:
            self.closed = True
            return False
        located = re.search('expected (\\d+)', core)
        if located:
            self.move = int(located.group(1)) - 1
            payload_n['step'] = self.move + 1
            return True
        if 'reuse' in small or 'different action' in small:
            self.move += 1
            payload_n['step'] = self.move + 1
            return True
        return False

    def _g4_landed(self, name_n, args_n, payload):
        self.move = payload['step'] if isinstance(payload.get('step'), int) else self.move + 1
        if payload.get('finished'):
            self.closed = True
        if name_n != self.duties.get('ask'):
            payload['observation'] = self.unspelled_n(name_n, payload.get('observation'))
        viewed = payload.get('observation')
        self.memo_scan(name_n, args_n, viewed)
        self._g4_note_factless(name_n, viewed)
        if self._g4_sheet_ok(name_n, viewed):
            self._g4_mark_unfolded(name_n, args_n, _serial(viewed))
        return payload

    def _g4_sheet_ok(self, name_n, viewed):
        if name_n not in self.sheet_ops():
            return False
        if isinstance(viewed, dict) and viewed.get('error'):
            return False
        return not _failed_n(viewed)

    def _g4_note_factless(self, name_n, viewed):
        if name_n != self.details_op() or not isinstance(viewed, dict) or _failed_n(viewed):
            return
        ident = self.iid(viewed)
        if ident and (not _detail_chunk(viewed)):
            self.factless_n.add((name_n, ident))

    def _g4_mark_unfolded(self, name_n, args_n, displayed):
        unfolded, panes = (self.unfolded, self.panes)
        for val in args_n.values():
            if isinstance(val, list):
                called = [v_n.strip() for v_n in val if isinstance(v_n, str) and v_n.strip() in displayed]
            elif isinstance(val, str):
                called = [val.strip()]
            else:
                called = []
            unfolded.update(called)
            for iid in called:
                panes.setdefault(iid, set()).add(name_n)

    def _g4_aisle_move(self, ware2, tags):
        base = _bases(ware2.name_n)
        if not base or not ware2.aisle.strip():
            return None
        if not any((other_n is not ware2 and _terms(other_n.aisle) == _terms(ware2.aisle) for other_n in self.items)):
            return None
        if any((other_n is not ware2 and _bases(other_n.name_n) == base for other_n in self.items)):
            return None
        options = [tag for tag in tags if _bases(tag) == base and _terms(tag) != _terms(ware2.aisle)]
        if len(options) != 1:
            return None
        if any((_terms(other_n.aisle) == _terms(options[0]) for other_n in self.items)):
            return None
        return options[0]

    def _g4_brief_plan(self, choice, amount):
        rows = self.basket_rows
        keep = {i_n: rows[i_n] for i_n in self.basket_iids() if i_n in rows}
        keep[choice] = amount
        return self.basket_invocations(keep, [])

    def _g4_brief_pick(self, num, ware2, limited):
        options = [i_n for i_n in self.ware_iids(num) if i_n in self.whole and i_n not in self.basket_rows and self.purchasable(i_n) and (self.cost_of(i_n) is not None) and self.states_demands(i_n, num)]
        if not options or num in limited:
            return None
        options.sort(key=lambda i_n: self.cost_of(i_n) or 0.0)
        choice = options[0]
        if not choice or not isinstance(self.allowance, (int, float)):
            return choice
        kept = sum(((self.cost_of(i_n) or 0.0) * (self.kept_measures(i_n) or 0) for i_n in self.basket_iids()))
        expense = (self.cost_of(choice) or 0.0) * (1 if self.digest.get('budget_scope') == 'each' else ware2.amount)
        held = 0.0 if self.digest.get('budget_scope') == 'each' else kept
        return None if expense + held > self.allowance + 1e-09 else choice

    def _g4_cart_calls(self, basket, invocations):
        spec = self.specs.get(basket)
        for args_n in invocations:
            trouble = []
            shaped = _form(args_n, spec, 'args', trouble)
            if trouble or self.perform(basket, shaped) is None:
                return False
            self.memo_basket(basket, shaped)
        return True

    def _g4_compare_open(self, flank, several, desired, retain):
        if not (flank and several) or self.contrast_details() is not True or len(desired) < 2:
            return False
        return self.remaining() > retain and (not self.closed) and (not self.hurried())

    def _g4_compare_pass(self, flank, several, desired, retain, strings):
        unread_n = []
        spec = self.specs.get(flank)
        while self._g4_compare_open(flank, several, desired, retain):
            space = self.contrast_space()
            bundle = desired[:space]
            trouble = []
            request = _form({several: bundle}, spec, 'args', trouble)
            reply = None if trouble else self.perform(flank, request)
            if reply is None:
                break
            viewed = reply.get('observation')
            self.perceive(flank, viewed)
            self.memo_pane(flank, viewed)
            found = self.entries_in(viewed)
            strings += [f'{iid}: ' + self.display(entry)[:1500] for iid, entry in found]
            if found:
                self.contrast_fine = max(self.contrast_fine, len(found))
                whole = self.whole
                unread_n += [i_n for i_n in bundle if i_n not in whole]
                desired = desired[space:]
                continue
            self.contrast_poor = min(self.contrast_poor or 999, len(bundle))
            if self.contrast_space() >= len(bundle):
                break
        return (desired, unread_n)

    def _g4_reading_stops(self, op, retain):
        if self.remaining() <= retain or self.closed:
            return True
        return self.details_from.get(op) is False or self.hurried()

    def _g4_detail_pass(self, op, goal, queue, retain, strings):
        spec = self.specs.get(op)
        shown = self.display
        for iid in queue:
            if self._g4_reading_stops(op, retain):
                return
            if (op, iid) in self.factless_n:
                continue
            trouble = []
            request = _form({goal: iid}, spec, 'args', trouble)
            reply = None if trouble else self.perform(op, request)
            if reply is None:
                return
            viewed = reply.get('observation')
            self.perceive(op, viewed)
            self.memo_pane(op, viewed)
            if isinstance(viewed, dict) and self.iid(viewed):
                self.memo_details(op, bool(_detail_chunk(viewed)))
            strings.append(f'{iid}: ' + shown(viewed)[:1500])

    @staticmethod
    def _g4_crowded(learnt):
        return any((_tally(x_n) == 1 for x_n in _wares(learnt.get('crowded'))))

    def _g4_portion(self, size):
        now_n = self.list_now_n()
        ceiling = self.list_ceiling()
        if now_n < ceiling:
            return max(1, now_n // size)
        return max(MIX_TAGS, ceiling // size)

    def _g4_open_tracks(self, group, portion):
        tracks, spent_by = (self.tracks, self.track_spent)
        out = []
        for num in group:
            options = tracks.get(num) or []
            if not options:
                continue
            spent = spent_by.setdefault(num, set())
            left = [tag for tag in options if not (_terms(tag) in spent or self.heard_about(num, tag))]
            if left:
                out.append((num, left[:portion]))
        return out

    @staticmethod
    def _g4_tag_phrase(tags, size):
        head = tags[:size]
        if len(head) < 2:
            return tags[0]
        return f"{', '.join(head[:-1])} or {head[-1]}"

    def _g4_cluster_text(self, pieces, size):
        phrases = [self._g4_tag_phrase(tags, size) for __n, tags in pieces]
        asks = [f'what {piece_n} for the {self.ware_stated(num)}' for piece_n, (num, __n) in zip(phrases, pieces)]
        text_n = ', and '.join(asks)
        return 'W' + text_n[1:].replace(' for the ', ' would you like for the ', 1) + '?'

    def _g4_fit_cluster(self, pieces, portion):
        while pieces:
            for size in range(portion, 0, -1):
                text_n = self._g4_cluster_text(pieces, size)
                if self._g4_note_fits(text_n):
                    return ([(num, tags[:size]) for num, tags in pieces], text_n)
            pieces = pieces[:-1]
        return ([], '')

    def _g4_settle_cluster(self, pieces, held):
        items = self.items
        responded = False
        for num, tags in pieces:
            fresh = items[num - 1].heard[held[num - 1]:]
            if fresh:
                responded = True
                located = self.track_located
                located[num] = located.get(num, 0) + 1
                self.settle_n(num, tags, False, list(fresh), False)
        return responded

    def _g4_ask_and_close(self, num, matched, text_n, displayed, heard, base):
        bundle = self.send_n([(num, matched[0], text_n)], base)
        if not bundle:
            return []
        self.conclude_query(num, matched, displayed, bundle, heard)
        return bundle

    def _g4_next_query(self):
        for __n in range(LIST_MAGNITUDE):
            self.trail_upward = [f_n for f_n in self.trail_upward if not self.heard_about(f_n[0], f_n[1][0])]
            if self.trail_upward:
                num, tags, displayed = self.trail_upward.pop(0)
            else:
                num, tags = self.following_query(self.tracks, self.track_located, self.track_used_up, self.track_spent)
                displayed = False
            if not tags:
                return None
            matched, text_n = self.match_query(num, tags, displayed)
            if displayed and (not matched):
                matched, text_n = self.match_query(num, tags, False)
            if matched:
                return (num, matched, text_n, displayed)
            self.track_spent.setdefault(num, set()).add(_terms(tags[0]))
            self.journal.append({'kind': 'ask_too_long', 'item': num, 'label': tags[0]})
        return None

    def _g4_spent_out(self):
        return self.broke_n or self.none_merit or self.invocations >= ALL_INVOCATIONS

    def _g4_known_words(self):
        texts = []
        for ware2 in self.items:
            texts.append(ware2.name_n)
            texts.append(ware2.aisle or '')
            texts.extend((' '.join(re.findall('[A-Za-z0-9]+', demand.split(':', 1)[0])) for demand in ware2.needs + ware2.heard))
        for tag, name_n, __n, __n in self.key_grid():
            texts.append(name_n)
            texts.append(tag)
        known = set()
        for text_n in texts:
            known.update(_written(text_n))
        for entry in self.whole.values():
            for val in (_detail_chunk(entry) or {}).values():
                known.update((w_n for w_n in _written(' '.join(re.findall('[A-Za-z0-9]+', str(val)))) if len(w_n) >= QUIET_TERM))
        return known

    @staticmethod
    def _g4_grounded(inquiry, response, known):
        asked = re.sub('\\b(?:what|which|would|you|like|for|the|please|tell|me|and|or)\\b', ' ', inquiry, flags=re.I)
        ours = known | _written(' '.join(re.findall('[A-Za-z0-9]+', asked)))
        stated = _written(' '.join(re.findall('[A-Za-z0-9]+', response)))
        return bool(re.search('\\d', response) or stated & ours or re.search(QUERY_TERMS, response, re.I))

    def _g4_needs_quiet(self, key, val, spec):
        if val is None and key not in (spec.get('required') or []):
            return False
        text_n = str(val or '')
        if not text_n:
            return True
        return any((iid in text_n for iid in self.lines)) or self.digest.get('quiet_decline') is not False

    def _g4_each_ware_seen(self):
        for num in range(1, len(self.items) + 1):
            if not self.entries_of_ware(num):
                return False
        return True

    def _g4_item_line(self, num, ware2, several):
        vals = self.pointer_vals
        active = [m_n for m_n in ware2.needs + ware2.heard if m_n not in ware2.lifted]
        described = self.described_n
        heard = [m_n for m_n in active if described(num, m_n)]
        demands = []
        for m_n in active:
            if m_n not in heard:
                suffix = '' if m_n not in vals else f' (the page it points at gives: {vals[m_n]})'
                demands.append(m_n + suffix)
        text_n = f'item {num}: {ware2.name_n}' if several else f'item: {ware2.name_n}'
        if ware2.aisle:
            text_n += f' (store label: {ware2.aisle})'
        text_n += f', quantity {ware2.amount}' + '; requirements: '
        text_n += '; '.join(demands) if demands else 'none beyond the kind of product'
        if heard:
            text_n += "; described in the shopper's words, which no listing attribute states (decides nothing): " + '; '.join(heard)
        return text_n

    def _g4_order_terms(self):
        get = self.digest.get
        out = []
        if self.allowance is not None:
            scope_n = 'for each item' if get('budget_scope') == 'each' else 'in total'
            out.append(f"budget: at most {self.allowance} {get('budget_unit') or ''} {scope_n}".replace('  ', ' '))
        choice = get('pick')
        if choice == 'cheapest':
            out.append('pick: the cheapest listing that fits (lowest price)')
        elif choice in ('most', 'least'):
            out.append(f"pick: the {choice} {get('pick_measure') or ''} among fitting listings".replace('  ', ' '))
        elif getattr(self, 'p1_changed', False):
            out.append('pick: any listing that fits everything; the task has changed since the start, so when several listings you have read fit, take the lowest-priced (no extra search for it)')
        else:
            out.append('pick: any listing that fits everything')
        if get('in_stock_only', True) is not False:
            out.append('in stock only')
        out += [f'wish (tie-breaker only): {wish_n}' for wish_n in _wares(get('wishes'))]
        out += [f'rule from the request: {rule_n}' for rule_n in _wares(get('rules'))]
        if get('quiet_decline') is True:
            out.append('if nothing fits: end without naming or suggesting any listing')
        out += self._g4_step_lines()
        tail = self._g4_details_line()
        if tail is not None:
            out.append(tail)
        return out

    def _g4_step_lines(self):
        if not self.procedure:
            return []
        stages = {'choose': 'before choosing', 'order': 'at the order (NVIDIA does it when you check out)', 'end': 'at an ending without a purchase (NVIDIA does it then)'}
        return [f"step the request asks for, {stages[record['when']]}: {record['step']}" for record in self.procedure]

    def _g4_details_line(self):
        details = self.details_op()
        if not details:
            return None
        known = self.details_from
        if known.get(details) is not True and (not self.duties.get('offer')):
            return None
        ops = self.sheet_ops()
        plain = sorted((t_n for t_n in ops | {self.duties.get('offer') or ''} if t_n and known.get(t_n) is False))
        sheets = sorted((t_n for t_n in ops if t_n != details))
        line = f'listing attributes come from {details}'
        line += '' if not plain else f"; {', '.join(plain)} show none"
        line += '' if not sheets else f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(sheets)}"
        return line

    def _g4_other_records(self, missed):
        pool = (iid for iid in self.viewed_sequence[::-1] if not (iid in missed or iid in self.vanished or self.vended_out(iid)))
        records = []
        for iid in pool:
            if len(records) == 25:
                break
            records.append(_serial(_trim(self.whole.get(iid) or self.lines.get(iid), 200))[:1200])
        return records

    @staticmethod
    def _g5_glue(parts):
        return ''.join(parts)

    def _g5_duty(self, role):
        return self.duties.get(role)

    def _g5_duties(self, *roles):
        return tuple((self.duties.get(role) for role in roles))

    @staticmethod
    def _g5_multi(args_n, several):
        if several:
            picked = args_n.get(several)
            if isinstance(picked, list):
                return picked
        return None

    def _g5_reread(self, op, goal, iid, rows):
        trouble = []
        sent = _form({goal: iid}, self.specs.get(op), 'args', trouble)
        if trouble or self.hurried():
            return False
        before = dict(self.lines.get(iid) or {})
        reply = self.perform(op, sent)
        if reply is None or self.closed:
            return False
        seen = reply.get('observation')
        self.deal_viewed[iid] = self.move
        self.perceive(op, seen)
        moved = []
        for field, now in (self.lines.get(iid) or {}).items():
            if field not in before or before[field] == now:
                continue
            if isinstance(now, (dict, list)):
                continue
            moved.append(field)
        if not moved:
            return True
        names = ', '.join(moved[:4])
        dump = _serial(_trim(seen, 300))[:900]
        rows.append(f'{iid} changed ({names}): {dump}')
        return True

    def _g5_textlike(self, spec):
        if not isinstance(spec, dict):
            return False
        if 'string' not in _sorts(spec) and 'array' not in _sorts(spec):
            return False
        return 'enum' not in spec

    def _g5_prefixed(self, line, small):
        for field, text in line.items():
            if self.id_key_n == field or not isinstance(text, str):
                continue
            if text.strip().lower().startswith(small):
                return True
        return False

    def _g5_repair_id(self, val):
        trimmed = val.strip() if isinstance(val, str) else None
        if trimmed is not None and trimmed in self.lines:
            return (trimmed, '')
        if trimmed is None or _id_akin(val, True) or len(trimmed) < 8:
            return (val, '')
        small = trimmed.lower()
        strikes = [iid for iid, line in self.lines.items() if self._g5_prefixed(line, small)]
        if not strikes:
            return (val, f'{val!r} is not a listing id; pass the id from the result rows (their {self.id_key_n} field)')
        if len(strikes) > 1:
            return (val, f"several listings carry the text {val!r} ({', '.join(strikes[:6])}); pass one listing id")
        return (strikes[0], '')

    @staticmethod
    def _g5_peak(scores):
        at, best = (None, 0)
        for pos, score in enumerate(scores):
            if score > best:
                at, best = (pos, score)
        return at

    def _g5_cheap_key(self, iid):
        return (iid not in self.whole, self.cost_of(iid) if self.cost_of(iid) is not None else float('inf'))

    def _g5_review_once(self):
        got = self.review_invoke()
        return (got, self.review_matches(got))

    def _g5_records(self):
        for iid in sorted(reversed(self.viewed_sequence), key=lambda one: one not in self.whole):
            if self.vended_out(iid):
                continue
            full = self.whole.get(iid) if iid in self.whole else None
            if full:
                yield ('OPENED ' + _serial(_trim(full, 200))[:1200])
            else:
                yield ('ROW ' + _serial(_trim(self.lines.get(iid), 120))[:500])

    def _g5_live_ids(self, raw):
        kept = []
        for one in _wares(raw):
            if str(one) in self.lines and (not self.vended_out(str(one))):
                kept.append(str(one))
        return kept[:4]

    def _g5_item_faults(self, num, wanted, iids, tag, holder, faults):
        label = wanted.name_n
        held = [iid for iid in iids if num in holder[iid]]
        if not held:
            faults.append(f'item {num} ({label}) has no line in the cart')
        else:
            lone = [iid for iid in held if holder[iid] == [num]]
            if len(lone) > 1:
                width, listed = (len(held), ', '.join(held))
                faults.append(f'item {num} ({label}) has {width} lines ({listed}); keep one')
        for iid in held:
            units, kind = (self.kept_measures(iid), tag[iid])
            if holder[iid] == [num] and units != wanted.amount:
                need = wanted.amount
                faults.append(f'{iid} is held at {units} unit(s); item {num} ({label}) wants {need}')
            fuller_n = self.fuller_tag(num, kind) or self.called_tag(num, kind)
            if fuller_n:
                faults.append(f'{iid} is from {kind}, but the store has the label {fuller_n}, which names item {num} ({label}) in full: take the line from there')

    def _g5_budget_faults(self, iids, faults):
        prices = {iid: self.cost_of(iid) for iid in iids}
        if not isinstance(self.allowance, (int, float)):
            return
        if any((cost is None for cost in prices.values())):
            return
        cap = self.allowance
        if self.digest.get('budget_scope') == 'each':
            for iid in iids:
                if cap < prices[iid] <= 10 * cap:
                    faults.append(f'{iid} costs {prices[iid]:g}, over the {self.allowance:g} limit for each item')
            return
        whole = sum((prices[iid] * self.kept_measures(iid) for iid in iids))
        if cap + 1e-09 < whole <= 10 * cap:
            faults.append(f'the total is {whole:g}, over the budget of {self.allowance:g}')

    def _g5_invocation(self, batch, invoke):
        allowed_n = {t_n['function']['name'] for t_n in self.offered_n()}
        cid_n = invoke['id']
        return (cid_n, self._g5_handle(batch, invoke, allowed_n))

    def _g5_screen(self, name_n, allowed_n):
        if name_n == self._g5_duty('ask') and self.note_args('x' * 200) is not None:
            return "Not run (no step used): NVIDIA's interviewer handles every message to the shopper."
        if name_n not in self.specs:
            return f"Not run (no step used): unknown tool. Tools: {', '.join(sorted(allowed_n))}."
        if name_n in allowed_n:
            return None
        return 'Not run (no step used): too few steps are left for that; use: ' + ', '.join(sorted(allowed_n)) + '.'

    def _g5_bare(self):
        return {op for job_n, op in self.duties.items() if 'search' != job_n}

    def _g5_lone_removal(self, args_n):
        if not any(((_tally(v_n) or 0) < 0 for v_n in args_n.values() if not isinstance(v_n, str))):
            return None
        pane = _serial(self.basket_pane) if self.basket_at <= self.basket_viewed_at else ''
        for v_n in args_n.values():
            if not isinstance(v_n, str) or v_n not in self.lines:
                continue
            if v_n in self.basket_rows or v_n in pane or v_n in self.basket_stated:
                continue
            return f'Not run (no step used): {v_n} is not in the cart, so there is nothing to take off. Cart now: ' + self.basket_text()
        return None

    def _g5_handle(self, batch, invoke, allowed_n):
        if invoke.get('joined'):
            return 'Shown in the side-by-side result above: NVIDIA opened these listings together in one step.'
        fn_n = invoke.get('function') or {}
        name_n = fn_n.get('name')
        crude = fn_n.get('arguments')
        if self.closed:
            return 'Not run: the episode is over.'
        refused = self._g5_screen(name_n, allowed_n)
        if refused is not None:
            return refused
        if isinstance(crude, dict):
            args_n = crude
        else:
            args_n = _json_obj_n(crude) if (crude or '').strip() else {}
            if args_n is None:
                return 'Not run (no step used): the arguments were not one JSON object.'
        faults = []
        args_n = _form(args_n, self.specs[name_n], 'args', faults)
        if faults:
            return 'Not run (no step used): ' + '; '.join(faults[:6])
        args_n, wrong_n = self.resolve_iids(name_n, args_n)
        if wrong_n:
            return 'Not run (no step used): ' + wrong_n + '.'
        if name_n == self._g5_duty('cart'):
            args_n = self.entire_deletion(name_n, args_n)
        pruned = ''
        several = self.contrast_goal() if name_n == self._g5_duty('compare') else None
        picked = self._g5_multi(args_n, several)
        if picked is not None and len(picked) > self.contrast_space() and self.contrast_poor:
            space = self.contrast_space()
            pruned = f"\n[NVIDIA] Sent the first {space} listings only: this store shows at most {space} side by side. Not shown yet: {', '.join(map(str, picked[space:]))}."
            args_n = {**args_n, several: picked[:space]}
        goal = self.lone_goal(name_n)
        if goal and name_n in self._g5_duties('open', 'offer') and (str(args_n.get(goal)) in self.vanished):
            return f'Not run (no step used): {args_n[goal]} sold out at step {self.vanished[str(args_n[goal])]} and will not come back. Replace its cart line with another fitting listing.'
        if name_n == self._g5_duty('cart'):
            refused = self._g5_lone_removal(args_n)
            if refused is not None:
                return refused
        closing = self.is_purchase(name_n, args_n) or self.is_decline(name_n, args_n)
        bare_jobs = self._g5_bare()
        if closing:
            if len(batch) > 1:
                return 'Not run (no step used): send the finishing call on its own, after reading your other results.'
        elif self.remaining() <= self.end2() and (self._g5_duty('buy') or self._g5_duty('refuse')):
            return _g5_KEPT_STEPS
        novel = name_n not in bare_jobs and self.phrase_is_fresh(name_n, args_n)
        sent = args_n
        args_n = self.initial_look(name_n, args_n, novel)
        if sent != args_n:
            pruned += f'\n[NVIDIA] This is a new search, so NVIDIA asked for its first page ({_serial(sent)} was the call you sent).'
        args_n = self.line_iids(name_n, args_n)
        if name_n == self._g5_duty('search') and self.tag_codes_dead and self.names_tag_code(args_n):
            return _g5_DEAD_CODES
        sign = name_n + _serial(args_n)
        if not closing and name_n != self._g5_duty('cart') and (sign in self.recent_n):
            return 'Not run (no step used): you just made this same call and nothing changed; its result is above.'
        readers_n = set(self._g5_duties('search', 'open', 'offer', 'compare')) - {None}
        reading = name_n in readers_n
        stored = self.scan_cache.get(sign)
        if reading and stored and (stored[0] == self.edition):
            return f'Not run (no step used): the same call at step {stored[1]} returned this, and nothing has changed since:\n' + stored[2][:3000]
        if goal and reading and (name_n != self._g5_duty('compare')):
            target = str(args_n.get(goal))
            if self.displayed_already(name_n, target) or (name_n, target) in self.factless_n:
                return f'Not run (no step used): {args_n[goal]} has shown everything this view gives since the last change. Its record: ' + self.entry_of(str(args_n[goal]))
        picked = self._g5_multi(args_n, several)
        if picked:
            retain = [i_n for i_n in picked if not self.displayed_already(name_n, str(i_n))]
            if not retain:
                return 'Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n' + '\n'.join((f'{i_n}: {self.entry_of(str(i_n))}' for i_n in picked))[:4000]
            if len(retain) < len(picked):
                pruned += f"\n[NVIDIA] Left out {', '.join((str(i_n) for i_n in picked if i_n not in retain))}: shown in full already, nothing changed since."
                args_n = {**args_n, several: retain}
                sign = name_n + _serial(args_n)
        if closing and self.remaining() > 2:
            if self.is_purchase(name_n, args_n):
                held = self._g5_hold_purchase()
            elif self.remaining() > 5:
                held = self._g5_hold_decline()
            else:
                held = None
            if held is not None:
                return held
        if self.is_decline(name_n, args_n) and self.remaining() <= 5 and (not self.tardy_kept):
            novel = self.basket_pane is not None and self.basket_at <= self.basket_viewed_at
            held = self._g5_hold_late(novel)
            if held is not None:
                return held
        if self.is_decline(name_n, args_n) and (not self.clues) and (not self.pushed_n) and (self.remaining() > self.max_moves // 3):
            self.pushed_n = True
            return f"Not run yet (no step used). {self.remaining()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."
        if self.is_decline(name_n, args_n):
            completed = self.complete_basket()
            if completed:
                return completed + (' [episode finished]' if self.closed else '')
            args_n = self.quiet_n(name_n, args_n)
        if self.is_purchase(name_n, args_n):
            tardy = self.tardy_inquiry()
            if tardy:
                return 'Not run yet. ' + tardy + '\nCart now: ' + self.basket_text()
        if closing and self.remaining() >= 2:
            go_n, stated = self.barrier('order' if self.is_purchase(name_n, args_n) else 'end')
            if not go_n:
                return 'Not run yet. ' + stated + '\nCart now: ' + self.basket_text()
            if stated:
                pruned = '\n[NVIDIA] ' + stated + pruned
        answer = self.perform(name_n, args_n)
        if answer is None:
            return 'The call failed (network or server). You may try again.'
        return self._g5_settle(name_n, args_n, answer, (several, sign, readers_n, closing, novel, pruned))

    def _g5_peek_cart(self):
        if not (self.basket_touched and self.basket_at > self.basket_viewed_at and self._g5_duty('cart_view')):
            return None
        if self.remaining() <= 5 or self.hurried():
            return None
        trouble = []
        view_args = _form({}, self.specs.get(self.duties['cart_view']), 'args', trouble)
        if trouble:
            return None
        pane = self.perform(self.duties['cart_view'], view_args)
        if pane is None:
            return None
        self.basket_pane = pane.get('observation')
        self.basket_viewed_at = self.move
        change_n = self.perceive(self.duties['cart_view'], self.basket_pane)
        if not change_n:
            return None
        self.pending = []
        return 'Not run yet (no step used). Looking at the cart showed a change: ' + change_n + '\nCart now: ' + self.basket_text() + '\nApply the change, fix the cart if needed, then check out.'

    def _g5_hold_purchase(self):
        hunted_n = self.chase_lower_priced()
        broader = self.broader_grip()
        if broader:
            return broader + '\nCart now: ' + self.basket_text()
        rescan = self.rescan_offers()
        if self.memo_vended:
            self.find_vended()
        changed = self._g5_peek_cart()
        if changed is not None:
            return changed
        exchange = self.exchange_hint()
        gone = [i_n for i_n in self.basket_iids() if i_n in self.vanished] if self.kept < 2 else []
        if exchange or gone:
            self.kept += 0 if exchange else 1
            return 'Not run yet (no step used). ' + (exchange or f"{', '.join(gone)} in the cart is sold out: replace that line first.") + '\nCart now: ' + self.basket_text()
        series, grip = self.path()
        if grip:
            return 'Not run yet (no step used). ' + series + '\nCart now: ' + self.basket_text()
        misfit_n = self.match_issues() if self.match_contains < 2 and self.remaining() > self.end2() + 3 else []
        if misfit_n:
            self.match_contains += 1
            self.journal.append({'kind': 'fit_hold', 'faults': misfit_n[:4]})
            return 'Not run yet (no step used). A part in the cart does not work with the device in the cart: ' + '; '.join(misfit_n[:3]) + '. Swap the part (or the device), then check out.\nCart now: ' + self.basket_text()
        faults = self.row_issues() if self.row_contains < 2 and self.remaining() > self.end2() + 2 else []
        if faults and all((' is held at ' in f_n for f_n in faults)):
            mended = self.fix_tallies(self.tally_fixes())
            if mended:
                faults = self.row_issues()
                if not faults:
                    self.row_contains += 1
                    return "Not run yet (no step used). NVIDIA set the cart's quantities itself: " + '; '.join(mended) + '. If the store refused any of these, fix those lines; then check out.\nCart now: ' + self.basket_text()
        if faults:
            self.row_contains += 1
            self.journal.append({'kind': 'line_hold', 'faults': faults[:6]})
            return 'Not run yet (no step used). The cart does not match the request yet: ' + '; '.join(faults[:6]) + ". Fix those lines (the cart tool's own way to add or take off units), then check out.\nCart now: " + self.basket_text()
        rows = self.basket_iids()
        scan = self.populate_details(rows, self.end2() + (1 if len(rows) >= 2 else 3))
        self.display_ties(rows, self.end2() + 1)
        scan += self.lower_priced_lines()
        verdict_n = self.verify_basket()
        if verdict_n:
            viewed = "NVIDIA read these listings' attributes for the check (one step each):\n" + '\n'.join(scan) + '\n' if scan else ''
            if hunted_n:
                viewed = hunted_n + ' ' + self.below_basket() + '\n' + viewed
            return 'Not run yet. ' + series + viewed + verdict_n + '\nCart now: ' + self.basket_text()
        if rescan and self.expense > SPEND_MILD:
            return "Not run yet (no step used). NVIDIA re-read the cart's listings just now:\n" + rescan + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'
        return None

    def _g5_hold_decline(self):
        aimed = self.decline_path()
        if aimed:
            return 'Not run yet (no step used). ' + aimed
        self.chase_matches()
        review_n = self.review()
        if review_n:
            return 'Not run yet (no step used). ' + review_n
        if not self.clues or self.remaining() <= 4:
            return None
        if self.decline_chunks == 0:
            self.decline_chunks = 1
            self.journal.append({'kind': 'refuse_blocked', 'leads': list(self.clues)})
            return 'Not run yet (no step used). The review found listings that appear to fit: ' + ', '.join(self.clues) + _g5_CLUE_TAIL
        if self.decline_chunks != 1:
            return None
        self.decline_chunks = 2
        initial = self.clues[0]
        self.verify_basket([initial], SPEND_HARDN)
        if not (self.verdicts_n.get(self.verify_key([initial])) and initial in self.clues):
            return None
        self.confirmed_n = initial
        self.journal.append({'kind': 'refuse_blocked', 'confirmed': initial})
        return f"Not run yet (no step used). An independent check of {initial}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."

    def _g5_hold_late(self, novel):
        rows = [i_n for i_n in self.built_n if not self.vended_out(i_n)]
        absent = [i_n for i_n in rows if i_n not in self.basket_iids() or (i_n in self.lifted and (not novel))]
        if len(rows) < len(self.items) or len(absent) + 1 > self.remaining():
            return None
        self.tardy_kept = True
        self.journal.append({'kind': 'late_decline_held', 'lines': rows, 'missing': absent})
        redo_n = f"Put {', '.join(absent)} back in the cart at the requested quantities, then check out." if absent else 'Check out the cart as it is.'
        return _g5_LATE_HEAD + redo_n + _g5_LATE_TAIL

    def _g5_id_pairs(self, name_n, args_n):
        for k_n, v_n in args_n.items():
            seq = v_n if isinstance(v_n, list) else [v_n]
            if seq and all((isinstance(x_n, str) and x_n in self.lines for x_n in seq)):
                yield (name_n, k_n)

    def _g5_open_swap(self, name_n):
        for t_n in self.specs:
            if name_n == t_n or not self.lone_goal(t_n):
                continue
            if self.details_from.get(t_n) is False:
                continue
            if t_n in self._g5_duties('cart', 'buy', 'refuse', 'ask', 'search'):
                continue
            if not self.lists_things_n(t_n):
                return t_n
        return None

    def _g5_single_view(self, name_n, obs):
        offer = self.iid(obs) if isinstance(obs, dict) else None
        if not offer:
            return
        self.memo_details(name_n, bool(_detail_chunk(obs)))
        if name_n == self._g5_duty('open') and self.details_from.get(name_n) is False:
            swap = self._g5_open_swap(name_n)
            if swap:
                self.duties['open'] = swap
        if self.details_from.get(name_n) is False and name_n != self._g5_duty('open') and (not self._g5_duty('offer')):
            self.duties['offer'] = name_n
        if name_n == self.deal_op():
            self.deal_viewed[offer] = self.move

    def _g5_cart_after(self, name_n, args_n, obs):
        self.memo_basket(name_n, args_n)
        self.basket_stated = _serial(obs)
        self.recent_n = []
        lead = None
        for v_n in args_n.values():
            if not isinstance(v_n, str) and _tally(v_n) is not None:
                lead = _tally(v_n)
                break
        if lead is not None:
            for val in args_n.values():
                if not isinstance(val, str) or val not in self.lines:
                    continue
                if lead > 0:
                    self.lifted.discard(val)
                elif not _in_lines(obs, val):
                    self.lifted.add(val)
        rows = [i_n for i_n in self.basket_iids() if not self.vended_out(i_n) and i_n not in self.lifted]
        if len(rows) >= len(self.built_n):
            self.built_n = rows

    def _g5_settle(self, name_n, args_n, answer, carried):
        several, sign, readers_n, closing, novel, pruned = carried
        obs = answer.get('observation')
        picked = self._g5_multi(args_n, several)
        if picked is not None and len(picked) > 1:
            displayed = _serial(obs)
            if not any((str(i_n) in displayed for i_n in picked)):
                self.contrast_poor = min(self.contrast_poor or 999, len(picked))
                pruned += f'\n[NVIDIA] The side-by-side view showed nothing for {len(picked)} listings: send at most {self.contrast_space()} per call.'
            else:
                self.contrast_fine = max(self.contrast_fine, len(picked))
        self.recent_n = (self.recent_n + [sign])[-3:]
        self.id_args_n.update(self._g5_id_pairs(name_n, args_n))
        self.absorb_n(obs)
        if name_n == self._g5_duty('compare'):
            lines = self.entries_in(obs)
            if lines:
                self.details_from[name_n] = self.details_from.get(name_n, False) or any((_detail_chunk(r_n) for __n, r_n in lines))
        if name_n in readers_n - {self._g5_duty('search')}:
            self.memo_pane(name_n, obs)
        bare = {op for job_n, op in self.duties.items() if 'search' != job_n}
        advancing = name_n not in bare and (not closing)
        paginated = self.extra_sheets(name_n, args_n, obs) if advancing else ''
        if novel and (not closing):
            paginated += self.final_sheet_initial(name_n, args_n, obs)
        paginated += self.shorter_lookup(name_n, args_n, obs) if advancing else ''
        if self.lone_goal(name_n) and name_n not in self._g5_duties('cart', 'buy', 'refuse', 'ask'):
            self._g5_single_view(name_n, obs)
        if name_n == self._g5_duty('cart'):
            self._g5_cart_after(name_n, args_n, obs)
        if name_n == self._g5_duty('search') and self.names_tag_code(args_n):
            self.tag_code_attempts.append(bool(self.entries_in(obs)))
            self.tag_codes_dead = len(self.tag_code_attempts) >= 2 and (not any(self.tag_code_attempts))
        if name_n == self._g5_duty('search') and self.entries_in(obs):
            self.looked = (self.looked + [dict(args_n)])[-8:]
        if name_n == self._g5_duty('cart_view'):
            self.basket_pane = obs
            self.basket_viewed_at = self.move
        bought = self.is_purchase(name_n, args_n)
        if bought:
            self.purchases += 1
            if not answer.get('finished'):
                self.declined_at = self.move
        parts = [self.display(obs), paginated, pruned]
        if advancing:
            parts += [self.cheapest_here_n([obs] + self.paginated_obs), self.ceiling_ignored(name_n, args_n, obs)]
        origin = self._g5_glue(parts)
        evidence = self.evidence_n(obs)
        if evidence:
            parts += ['\n', evidence]
        if name_n == self._g5_duty('cart') and any((_tally(v_n) == 0 for v_n in args_n.values() if not isinstance(v_n, str))):
            if any((isinstance(v_n, str) and v_n in self.lines and _in_lines(obs, v_n) for v_n in args_n.values())):
                parts.append(_g5_ZERO_NOTE)
        self.memo_vended = False
        slack_rows = self.slack_rows(obs)
        if slack_rows:
            sense = self.grab_memo(slack_rows)
            if sense:
                parts += ['\n[note read] ', sense, '. Apply this change to your picks before checking out.']
        self._g5_after_refusal(name_n, args_n, answer, obs, parts)
        exchange = self.exchange_hint()
        if exchange:
            parts += ['\n', exchange]
        if name_n == self._g5_duty('cart') and self.limited and (not self.closed):
            misfit_n = self.match_issues()
            if misfit_n:
                parts += ['\n[NVIDIA] These lines do not work together yet: ', '; '.join(misfit_n)]
        if self.pending:
            parts += ["\n[notes seen on NVIDIA's own calls] ", ' | '.join(self.pending)]
            self.pending = []
        if name_n in readers_n:
            self.scan_cache[sign] = (self.edition, self.move, origin)
        if answer.get('finished'):
            parts.append(' [episode finished]')
        return self._g5_glue(parts)

    def _g5_after_refusal(self, name_n, args_n, answer, obs, parts):
        declined = self.is_purchase(name_n, args_n) and (not answer.get('finished'))
        again = declined and any((row in self.spoken_n(obs) for row in self.vended_rows))
        slack = self.memo_vended and (declined or (not self.memo_called and (not self.is_purchase(name_n, args_n))))
        if (slack or again) and self.basket_iids() and (not self.closed):
            viewed = self.find_vended()
            if viewed:
                parts += ["\n[NVIDIA re-read the cart's lines after the refusal]\n", viewed]
        self.memo_vended = False

    def _g5_intro(self, unfolding, shifted):
        parts = ["SHOPPER'S REQUEST:\n" + self.demand2]
        if unfolding is not None:
            parts += ['\n\nWHAT THE STORE SHOWS FIRST:\n', self.display(unfolding)]
        if shifted:
            parts += ['\n\nNVIDIA has searched for each item and read some of its listings (one step each):\n', shifted, '\nJudge these against the requirements and search further only for what is still missing.\n\n']
        else:
            parts.append(_g5_START_NOTE)
        parts.append(self.state())
        return self._g5_glue(parts)

    def _g5_second_round(self):
        turned_n = self.conclude(again=True)
        if not turned_n or self.closed or self.broke_n or self.none_merit:
            return
        if self.remaining() <= 1:
            return
        self.dialogue.append({'role': 'user', 'content': '[NVIDIA] NVIDIA tried to end the episode and ' + turned_n + '\nApply what the store says, then finish.\n' + self.state()})
        self.work_n(self.cycle_invocations + 10, SPEND_HARDN + 0.01)

    def _g5_first_digest(self, instruction):
        digest = self.query_json(instruction, [LIGHTN], 4000) or {}
        if self.none_merit:
            pass
        elif not (_wares(digest.get('items')) and isinstance(digest.get('roles'), dict) and digest['roles'].get('buy')):
            digest = self.query_json(instruction, [HEAVYN, LIGHTN], 2500) or digest
        elif self.shared_aisle(digest):
            again = self.query_json(instruction, [HEAVYN], 2500) or {}
            if _wares(again.get('items')) and isinstance(again.get('roles'), dict) and again['roles'].get('buy'):
                digest = again
        while True:
            if digest or self.none_merit:
                break
            if self.invocations >= ALL_INVOCATIONS or self.timer() >= DIGEST_BY:
                break
            time.sleep(DIGEST_MARGIN)
            digest = self.query_json(instruction, [LIGHTN, HEAVYN], 2500) or {}
        return digest

    def _g5_split_modes(self, digest):
        purchase = self.duties.get('buy')
        decline = self.duties.get('refuse')
        if not purchase or decline != purchase:
            return
        if digest['buy_args'] and digest['refuse_args'] and (digest['buy_args'] != digest['refuse_args']):
            return
        query = f'The store tool {purchase} both checks out and ends without a purchase, depending on its arguments. Its schema: ' + _serial(self.specs.get(purchase)) + _g5_MODE_ASK
        repaired = self.query_json(query, [LIGHTN, HEAVYN], 1500) or {}
        if not isinstance(repaired.get('buy_args'), dict) or not isinstance(repaired.get('refuse_args'), dict):
            return
        if repaired['buy_args'] == repaired['refuse_args']:
            return
        digest['buy_args'], digest['refuse_args'] = (repaired['buy_args'], repaired['refuse_args'])

    @staticmethod
    def _g5_clash(left, right):
        for a_n in left:
            for b_n in right:
                if _terms(a_n.partition(':')[0]) != _terms(b_n.partition(':')[0]):
                    continue
                if _terms(a_n.partition(':')[2]) != _terms(b_n.partition(':')[2]):
                    return True
        return False

    def _g5_merge_items(self):
        merged = []
        became = {}
        for old_n, piece in enumerate(self.items, 1):
            slot = None
            for pos, kept in enumerate(merged, 1):
                if not piece.aisle.strip() or kept.amount != piece.amount:
                    continue
                if _terms(kept.aisle) != _terms(piece.aisle) or self._g5_clash(kept.needs, piece.needs):
                    continue
                slot = pos
                break
            if slot is None:
                merged.append(piece)
                became[old_n] = len(merged)
                continue
            host = merged[slot - 1]
            became[old_n] = slot
            host.needs += [m_n for m_n in piece.needs if m_n not in host.needs]
            self.journal.append({'kind': 'items_merged', 'into': host.name_n, 'dropped': piece.name_n})
        self.items = merged
        return became

    def _g5_pairs(self, digest, became):
        span = len(self.items)
        for couple in _wares(digest.get('fits')):
            if not isinstance(couple, dict):
                continue
            piece = became.get(_tally(couple.get('item')) or 0)
            primary = became.get(_tally(couple.get('with')) or 0)
            if not (piece and primary) or primary == piece:
                continue
            if not (1 <= piece <= span and 1 <= primary <= span):
                continue
            if (piece, primary) not in self.limited:
                self.limited.append((piece, primary))

    def _g5_pointers(self, digest):
        for lookup_n in _wares(digest.get('lookups')):
            text_n = str(lookup_n).strip()
            if all((text_n not in piece.needs for piece in self.items)):
                key = _terms(text_n.partition(':')[0])
                for piece in self.items:
                    hit = next((m_n for m_n in piece.needs if _terms(m_n.partition(':')[0]) == key), None)
                    if hit is not None:
                        text_n = hit
                        break
            num = 1
            for pos, piece in enumerate(self.items, 1):
                if text_n in piece.needs:
                    num = pos
                    break
            if text_n and (num, text_n) not in self.pointers:
                self.pointers.append((num, text_n))

    def _g5_limits(self, digest):
        allowance = digest.get('budget')
        if isinstance(allowance, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', allowance):
            allowance = float(allowance)
        if isinstance(allowance, (int, float)) and (not isinstance(allowance, bool)) and (allowance > 0):
            self.allowance = allowance
        bound = _tally(digest.get('messages'))
        if bound is None or bound < 0:
            if self.duties.get('ask'):
                self.text_bound = 3
        else:
            self.text_bound = bound
        moves = _tally(digest.get('steps'))
        if moves is not None and 10 <= moves < self.max_moves:
            self.max_moves = moves

    def _g5_procedure(self, digest):
        for record in _wares(digest.get('process')):
            if not isinstance(record, dict) or not str(record.get('step') or '').strip():
                continue
            when_n = str(record.get('when') or '').strip().lower()
            self.procedure.append(dict(zip(('step', 'when', 'messages'), (str(record['step']).strip(), when_n if when_n in ('choose', 'order', 'end') else 'order', max(0, _tally(record.get('messages')) or 0)))))
        if not self.duties.get('ask'):
            return
        converse = [p_n['messages'] for p_n in self.procedure if p_n['when'] in ('order', 'end') and p_n['messages']]
        self.reserve_n = min(max(converse), self.text_bound) if converse else 0

    @staticmethod
    def _g5_named(key, locus):
        if isinstance(key, str) and key in locus:
            return key
        return None

    @staticmethod
    def _g5_stem(text_n):
        return re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', _terms(text_n)))

    @classmethod
    def _g5_alias(cls, shares, tag):
        want = cls._g5_stem(tag)
        for k_n in shares:
            if cls._g5_stem(k_n) == want:
                return k_n
        return tag

    @staticmethod
    def _g5_portion(shares, tag):
        if _terms(tag) in shares:
            return shares[_terms(tag)]
        near = [s_n for k_n, s_n in shares.items() if _names_n(_root(tag), _root(k_n)) or _names_n(_root(k_n), _root(tag))]
        if near:
            return max(near)
        return 0.0

    def _g5_offer_lines(self, grid):
        final = None
        for index, (tag, name_n, stocked, values) in enumerate(grid, 1):
            if tag != final:
                yield f'category {tag!r}:'
                final = tag
            yield (f'  {index}. {name_n} (on {stocked}): ' + ' | '.join(values) + self.not_stated_n(tag, name_n, values) + (' [several features in one: ask last]' if re.search(CATCH_ALLN, name_n) else ''))

    def _g5_seed_tracks(self, scheme, tracks, hinted_n):
        span = len(self.items)
        for choice in _wares(scheme.get('ask')):
            if not isinstance(choice, dict):
                continue
            raw = _tally(choice.get('item') or 1) or 1
            floor = raw if raw > 1 else 1
            num = span if span < floor else floor
            tag = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(choice.get('label') or ''))[0].strip().rstrip('?.')
            if not tag or len(tag.split()) > 5:
                continue
            if not any((_terms(tag) == _terms(other_n) for other_n in tracks[num])):
                tracks[num].append(tag)
            if not str(choice.get('hint') or '').strip():
                continue
            hinted_n[num].add(_terms(tag))
            if num not in self.scheme_sequence:
                self.scheme_sequence.append(num)

    @staticmethod
    def _g5_drops_hint(track, put, hints):
        for x_n in track[:3]:
            if _terms(x_n) in hints and (not any((_terms(x_n) == _terms(y_n) for y_n in put[:3]))):
                return True
        return False

    def _g5_lane(self, num, grid, lane, hinted_n):
        keys = self.track_fields(num, grid)
        scan = max(1, self.entries_of_ware(num))
        shares = {}
        for tag, key, stocked, __n in grid:
            if self.holder_of(tag) == num:
                shares[key] = max(shares.get(key, 0.0), stocked / scan)
        hints = {_terms(self._g5_alias(shares, h_n)) for h_n in hinted_n[num]}
        hinted_n[num] = hints
        track = []
        for choice in lane:
            if not any((_terms(self._g5_alias(shares, choice)) == _terms(other_n) for other_n in track)):
                track.append(self._g5_alias(shares, choice))
        strong = [k_n for k_n in keys if shares.get(k_n, 0.0) >= GUARD_PORTION and (not re.search(CATCH_ALLN, k_n))]
        for key in strong[:GUARD_FIELDS]:
            if any((_terms(key) == _terms(other_n) for other_n in track[:3])):
                continue
            weak_n = None
            for pos, other_n in enumerate(track[:3]):
                if _terms(other_n) not in hints and self._g5_portion(shares, other_n) + GUARD_MARGIN <= shares.get(key, 0.0):
                    weak_n = pos
                    break
            if weak_n is None and len(track) >= 3:
                continue
            put = [other_n for other_n in track if _terms(other_n) != _terms(key)]
            if weak_n is not None and len(self.items) > 1:
                shifted = put[weak_n]
                put[weak_n] = key
                put.insert(min(3, len(put)), shifted)
            else:
                put.insert(len(put) if weak_n is None else min(len(put), weak_n), key)
            if not self._g5_drops_hint(track, put, hints):
                track = put
        for key in keys:
            if not any((_terms(key) == _terms(other_n) for other_n in track)):
                track.append(key)
        plain, mixed = ([], [])
        for x_n in track:
            if re.search(CATCH_ALLN, _terms(x_n)):
                mixed.append(x_n)
            else:
                plain.append(x_n)
        return plain + mixed

    def _g5_short_texts(self):
        for line in self.lines.values():
            for key, val in line.items():
                if isinstance(val, str) and 0 < len(val) <= 40 and (not re.search('\\d{3}', val)):
                    yield (key, val)

    def _g5_deliver(self, op, question, base, dispatched):
        num, __n, text_n = question
        if self.closed or self.remaining() <= base or self.text_spent >= self.text_bound:
            return False
        args_n = self.note_args(text_n)
        if not args_n:
            return False
        answer = self.perform(op, args_n)
        if answer is None:
            return False
        self.text_spent += 1
        response = answer.get('observation')
        response = self.p1_side_notes(op, response)
        terms = self.spoken_n(response)
        dispatched.append((num, text_n, terms))
        step, said = (self.move, terms[:300])
        cart = self.basket_sign()
        self.completed_log.append(f'step {step}: message to the shopper {text_n!r} -> reply {said!r} (cart then: {cart})')
        self.completed_moves.append(self.move)
        self.journal.append(dict(zip(('kind', 'q', 'a'), ('ask', text_n, said))))
        return not (isinstance(response, dict) and response.get('error'))

    def _g6_series_hits(self, num: int, index: dict[str, Any], signs: dict[str, Any]) -> Any:
        for piece, primary in self.limited:
            if num not in (piece, primary):
                continue
            scheme2 = self.tie_scheme(piece, primary, index)
            if not scheme2:
                continue
            kins = scheme2[2]
            for member_n in (piece, primary):
                for i_n in self.ware_iids(member_n):
                    mark = signs.get(i_n) or {}
                    for family_n in (set(mark.get('fam') or []) | set(mark.get('named') or [])) & kins:
                        yield (i_n, family_n)

    def _g6_settle_many(self, num: int, tags: list[str], fresh: list[str], packed: bool, spent: set[str]) -> None:
        if packed and (not fresh):
            self.offer = False
            self.journal.append({'kind': 'lists_off', 'item': num})
            self.trail_upward.insert(0, (num, tags[:1], False))
            return
        strike = self.responded_tag(tags, fresh) if fresh else -1
        hit = 0 <= strike
        onward = tags[strike + 1:] if hit else tags[1:][::-1]
        spent.update(map(_terms, tags[:strike + 1 if hit else 1]))
        remainder = [tag for tag in self.tracks.get(num) or [] if tag not in tags]
        misses = 0 if hit else self.track_failures.get(num, 0) + 1
        self.track_failures[num] = misses
        if misses and misses % 2 == 0:
            middle = (len(onward) - 1) / 2
            order_n = sorted(range(len(onward)), key=lambda i_n: (abs(i_n - middle), i_n))
            onward = [onward[i_n] for i_n in order_n]
            self.tracks[num] = remainder + onward
        else:
            self.tracks[num] = onward + remainder
        done = tags[:strike + 1]
        upcoming = onward[:4]
        self.journal.append({'kind': 'list_settled', 'item': num, 'done': done, 'next': upcoming})

    def _g6_short_ready(self, scheme: Any, name_n: str, obs: Any) -> bool:
        return bool(scheme) and scheme[0] == name_n and (self.remaining() > self.end2() + BRIEF_RETAIN) and (not self.closed) and isinstance(obs, dict)

    def _g6_wordy(self, obs: dict[str, Any]) -> Any:
        for val in obs.values():
            if isinstance(val, str):
                if len(val.split()) >= 5:
                    yield val
            elif isinstance(val, list):
                if val and all((isinstance(v_n, str) for v_n in val)):
                    yield from (v_n for v_n in val if len(v_n.split()) >= 5)
            elif isinstance(val, dict) and (not self.iid(val, False, 1)):
                yield from (x_n for x_n in val.values() if isinstance(x_n, str) and len(x_n.split()) >= 6)

    def _g6_pace_note(self, remaining: int) -> str | None:
        if remaining <= self.end2() + 1:
            return '; '.join((' FINAL STEPS: check out now when the cart holds a line for each item', 'end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that).'))
        idle = 0 == self.purchases
        if remaining <= 6 or (idle and remaining <= len(self.items) + 5):
            return ' Few steps left: set the cart to the chosen lines and check out now.'
        if not idle:
            return None
        if remaining <= 12:
            return ' Steps are running low: settle on your picks, fill the cart and check out within the next few steps.'
        if self.basket_touched or self.move < self.max_moves // 2:
            return None
        return ' '.join(('', 'Half', 'the', 'steps', 'are', 'used', 'and', 'the', 'cart', 'is', 'empty:', 'cart', 'each', "item's", 'fitting', 'listing', 'now', '(the', 'store', 'may', 'still', 'sell', 'one', 'out).'))

    def _g6_unresolved(self) -> list[str]:
        active: set[str] = set()
        for __n, demand in self.demands_now():
            active.add(demand)
        return [demand for __n, demand in self.pointers if demand in active and demand not in self.pointer_vals]

    def _g6_line_names(self, iid: str, barred: Any) -> list[str]:
        names_n: list[str] = []
        for k_n, v_n in (self.lines.get(iid) or {}).items():
            if not isinstance(v_n, str):
                continue
            bare = v_n.strip()
            if 2 < len(bare) <= 40 and k_n not in barred and (bare.lower() not in NONE_NAME):
                names_n.append(bare)
        return names_n

    def _g6_alone_in_basket(self, holders: list[int], group_key: str) -> bool:
        ware2 = self.items[holders[0] - 1] if len(holders) == 1 else None
        count = 0
        for i_n in self.basket_iids():
            if not (self.basket_rows.get(i_n, 1) or 0) > 0 or ware2 is None:
                continue
            line = self.lines.get(i_n) or {}
            if self.possesses(ware2, str(line.get(group_key) or '')):
                count += 1
        return count == 1

    def _g6_tardy_gate(self) -> str | None:
        if self.tardy_queried or not self.duties.get('ask'):
            return ''
        if self.text_bound <= self.text_spent or any(map(_g6_spoke_at_order, self.procedure)):
            return ''
        if not self.remaining() < TARDY_MOVES and (not self.timer() > TARDY_BY) and self.assesses():
            return None
        return ''

    def _g6_seed_tracks(self) -> None:
        self.tracks = self.scheme_tracks(self.list_ceiling() if self.offer else 1)
        self.track_located = dict.fromkeys(self.tracks, 0)
        self.track_used_up = dict.fromkeys(self.tracks, 0)
        self.track_spent = {n_n: set() for n_n in self.tracks}

    def _g6_late_recap(self, inquiry: str, response: str, heard: list[str]) -> str:
        recap = f'NVIDIA asked the shopper one more question before checking out (one step):\nQ: {inquiry}\nA: {response}\nNew firm requirements from this answer: ' + '; '.join(heard) + '. '.join(('', 'The cart was not checked out: check each line against them (search again with the new values where needed), fix the cart, then check out.'))
        aimed = [demand for __n, demand in self.pointers if any((demand in t_n for t_n in heard))]
        if not aimed:
            return recap
        return recap + ' These take their value from another listing: ' + ' | '.join(aimed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value.'

    def _g6_engine_tries(self, engine: str, messages_n: list[dict[str, Any]], max_pieces: int, ops: Any) -> tuple[Any, Any]:
        attempts = 0
        while True:
            note, state = self.invoke_engine(engine, messages_n, max_pieces, ops)
            if note is not None or not self._g6_may_retry(state, attempts):
                return (note, state)
            wait = WAITSN[attempts]
            pause_n = wait * (0.8 + 0.4 * random.random())
            self.waited_n += wait
            attempts += 1
            time.sleep(pause_n)

    def _g6_may_retry(self, state: Any, attempts: int) -> bool:
        if not (state is None or state in (402, 408, 425, 429) or (isinstance(state, int) and 500 <= state)):
            return False
        if attempts >= (1 if state in (None, 408) else len(WAITSN)):
            return False
        wait = WAITSN[attempts]
        if self.waited_n + wait > WAIT_SUM:
            return False
        return not self.engine_space() - wait * 1.2 < INVOKE_BASE

    def _g6_tie_line(self, stocked: str, labeler: str, kins: set[str], signs: dict[str, Any], carrier_n: int, primary: int) -> str:
        desired = set(signs[labeler].get('named') or []) & kins
        fine2: list[str] = []
        for i_n in self.ware_iids(carrier_n):
            if not desired.isdisjoint(signs[i_n].get('fam') or []) and (not self.vended_out(i_n)) and (i_n != stocked):
                fine2.append(i_n)
        fine2 = fine2[:4]
        series = ', '.join(sorted(set(signs[stocked].get('fam') or []) & kins))
        head = f"{stocked} is of series {series}, while {labeler} takes {', '.join(sorted(desired))}"
        if fine2:
            return head + f"; seen listings of that series: {', '.join(fine2)}"
        return head + f"; search that series with the item's words, or take another {self.items[primary - 1].name_n} whose series the wanted lines carry"

    def _g6_names_kin(self, labeler: int, kins: set[str], signs: dict[str, Any]) -> bool:
        for i_n in self.ware_iids(labeler):
            for f_n in (signs.get(i_n) or {}).get('named') or []:
                if f_n in kins:
                    return True
        return False

    def _g6_may_turn(self, cycle_ceiling: int, spend_ceiling: float) -> bool:
        if self.closed or self.broke_n or (not self.remaining() > 0):
            return False
        if not cycle_ceiling > self.cycle_invocations or not spend_ceiling > self.expense:
            return False
        return self.timer() < FINAL_TURN or (not self.basket_iids() and self.engine_space() > EMPTY_SPACE)

    def _g6_silence_ends(self) -> bool:
        self.failures += 1
        if self.none_merit or self.invocations >= ALL_INVOCATIONS:
            return True
        if self.failures <= 2:
            return False
        if self.timer() >= OUTAGE_BYN:
            return True
        time.sleep(DIGEST_MARGIN)
        return False

    def _g6_calls_of(self, note: dict[str, Any]) -> list[dict[str, Any]]:
        out_n: list[dict[str, Any]] = []
        for num, invoke in enumerate(note.get('tool_calls') or []):
            if isinstance(invoke, dict):
                out_n.append(self._g6_call_record(num, invoke))
        return out_n

    def _g6_call_record(self, num: int, invoke: dict[str, Any]) -> dict[str, Any]:
        fn_n = invoke.get('function') if isinstance(invoke.get('function'), dict) else {}
        crude = fn_n.get('arguments')
        if isinstance(crude, dict):
            parsed_n = crude
        elif isinstance(crude, str) and crude.strip():
            parsed_n = _json_obj_n(crude)
        else:
            parsed_n = {}
        ident = str(invoke.get('id') or f'k{self.cycle_invocations}_{num}')
        called = str(fn_n.get('name') or '')
        packed = _serial(parsed_n if isinstance(parsed_n, dict) else {})
        return {'id': ident, 'type': 'function', 'function': {'name': called, 'arguments': packed}}

    def _g6_ask_before(self, substance: str, invocations: list[dict[str, Any]]) -> bool:
        committing_n = {self.duties.get(d_n) for d_n in ('cart', 'buy', 'refuse')} - {None}
        if self.queried or not self.duties.get('ask') or (not self.texts_open() > 0):
            return False
        if not any((c_n['function']['name'] in committing_n for c_n in invocations)):
            return False
        if not self.remaining() > 8 or not self.timer() <= QUERY_BY:
            return False
        self.dialogue.append({'role': 'assistant', 'content': substance or None, 'tool_calls': invocations})
        for invoke in invocations:
            self.dialogue.append({'role': 'tool', 'tool_call_id': invoke['id'], 'content': 'Not run (no step used): the shopper is asked first.'})
        recap = self.interview_n()
        self.dialogue.append({'role': 'user', 'content': (recap or 'The interviewer had nothing to ask.') + '\n' + self.state()})
        return True

    def _g6_play(self, substance: str, invocations: list[dict[str, Any]]) -> None:
        self.dialogue.append({'role': 'assistant', 'content': substance or None, 'tool_calls': invocations})
        calls = [c_n['function'] for c_n in invocations]
        self.journal.append({'kind': 'turn', 'text': substance[:300], 'calls': calls})
        answers = self.run_invocations(invocations)
        tail = len(answers) - 1
        for index, (cid_n, text_n) in enumerate(answers):
            body = text_n if index != tail else text_n + '\n' + self.state()
            self.dialogue.append({'role': 'tool', 'tool_call_id': cid_n, 'content': body})

    def _g6_unmix_item(self, num: int, ware2: Any, pointers_n: list[str], remainder: list[str]) -> None:
        product = f'\n\nPRODUCT: {ware2.name_n}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n'
        asks = '\n'.join(pointers_n)
        others = '\n'.join(remainder)
        obtained = self.query_json(UNMIXN + '\n\nREQUEST:\n' + self.demand2 + product + asks + '\nOTHER REQUIREMENTS:\n' + others, [HEAVYN, LIGHTN], 1500) or {}
        offered = [str(x_n).strip() for x_n in _wares(obtained.get('describe_other'))]
        shifted = [m_n for m_n in remainder if m_n in offered]
        if not shifted:
            return
        head = pointers_n[0]
        joined_n = head + ' (that listing: ' + '; '.join(shifted) + ')'
        ware2.needs = [joined_n if m_n == head else m_n for m_n in ware2.needs if m_n not in shifted]
        self.pointers = [(n_n, joined_n if n_n == num and demand == head else demand) for n_n, demand in self.pointers]
        self.journal.append({'kind': 'unmix', 'item': num, 'moved': shifted})

    def _g6_sheet_for(self, num: int, viewed: Any, group_key: str | None) -> bool:
        if not group_key:
            return True
        holders = {self.holder_of(str(entry.get(group_key) or '')) for __n, entry in self.entries_in(viewed)}
        return not holders or 0 in holders or num in holders

    def _g6_follow_pages(self, op: str, sign: str, invoke: dict[str, Any], viewed: Any, room: int) -> tuple[int, bool]:
        made = 0
        walked_n = {sign}
        while made < room and (not self.closed):
            page = self.turning(op, viewed)
            if not page or page[3] is None:
                break
            step = dict(invoke)
            step[page[2]] = page[3]
            invoke, fine = _g6_shape(step, self.specs.get(op), 'args')
            now_n = self.invoke_sign(op, invoke) if fine else ''
            if not fine or now_n in walked_n:
                break
            walked_n.add(now_n)
            if now_n in self.scan_invocations:
                viewed = self.scan_invocations[now_n]
                continue
            answer = self.perform(op, invoke)
            made += 1
            if answer is None:
                return (made, True)
            viewed = answer.get('observation')
            self.perceive(op, viewed)
        return (made, False)

    def _g6_fold_extra(self, entry: Any, learned: set[str]) -> Any:
        if not isinstance(entry, dict) or _detail_chunk(entry) or (not self.iid(entry, True)):
            return entry
        kept: dict[str, Any] = {}
        additional: dict[str, Any] = {}
        for k_n, v_n in entry.items():
            if isinstance(v_n, (str, int, float)) and (not isinstance(v_n, bool)) and (k_n not in learned):
                additional[k_n] = v_n
            else:
                kept[k_n] = v_n
        if len(additional) < ENTRY_SLOTS:
            return entry
        kept['attributes'] = additional
        return kept

    def _g6_value_probe(self, op: str, origin: dict[str, Any], slot: str, num: int, ware2: Any, tries: list[list[str]], viewed: list[str], allowance: int) -> int:
        for terms in filter(None, tries[:2]):
            query = dict(origin)
            query[slot] = ' '.join(terms)
            args_n, fine = _g6_shape(query, self.specs.get(op), 'args')
            if not fine or self.closed:
                return allowance
            answer = self.perform(op, args_n)
            allowance -= 1
            if answer is None:
                return allowance
            obs = answer.get('observation')
            self.perceive(op, obs)
            lines = self._g6_value_rows(num, ware2, obs)
            label = f'{_serial(args_n.get(slot))}: '
            viewed.append(label + (', '.join(lines[:6]) or 'no row of that kind'))
            if lines:
                fresh = [i_n for i_n in lines if i_n not in self.whole]
                unfolded = self.populate_details(fresh[:min(3, allowance)], self.end2() + VAL_RETAIN)
                allowance -= len(unfolded)
                if unfolded:
                    viewed.append('opened: ' + ' | '.join(unfolded)[:2400])
                return allowance
        return allowance

    def _g6_value_rows(self, num: int, ware2: Any, obs: Any) -> list[str]:
        group_key = self.section_key()
        located = [i_n for i_n, __n in self.entries_in(obs)]
        cap = None
        if isinstance(self.allowance, (int, float)):
            cap = self.allowance
            prices = [self.cost_of(i_n) for i_n in located]
            if prices and None not in prices and all((p_n > 10 * cap for p_n in prices)):
                cap = None

        def fits(i_n: str) -> bool:
            if not self.purchasable(i_n):
                return False
            if group_key:
                record = self.lines.get(i_n) or {}
                if not self.possesses(ware2, str(record.get(group_key) or '')):
                    return False
            return cap is None or (self.cost_of(i_n) or 0) <= cap
        lines = list(filter(fits, located))
        home_n = self.home_tag(num)
        if not (group_key and home_n):
            return lines
        near = [_terms(str((self.lines.get(i_n) or {}).get(group_key) or '')) == home_n for i_n in lines]
        return [i_n for i_n, ok in zip(lines, near) if ok] + [i_n for i_n, ok in zip(lines, near) if not ok]

    def _g6_told_field(self, op: str) -> tuple[str, dict[str, Any], str] | None:
        if isinstance(self.digest.get('search_words'), dict):
            heard = self.digest.get('search_words')
        else:
            heard = {}
        stated = heard.get('field') if isinstance(heard.get('field'), str) else None
        repaired = heard.get('args') if isinstance(heard.get('args'), dict) else {}
        if isinstance(heard.get('tool'), str) and heard.get('tool') in self.specs:
            called_op = heard.get('tool')
        else:
            called_op = op
        schema = (self.specs.get(called_op) or {}).get('properties') or {}
        if stated not in schema or 'string' not in _sorts(schema[stated]) or 'enum' in schema[stated]:
            return None
        issues: list[str] = []
        origin = _form({**repaired, stated: 'x'}, self.specs.get(called_op), 'args', issues)
        if issues:
            return None
        origin.pop(stated, None)
        return (called_op, origin, stated)

    def _g6_term_pairs(self, num: int, heard: list[str]) -> Any:
        pointed = [m_n for __n, m_n in self.pointers]
        for n_n, demand in self.demands_now():
            if num != n_n or demand in pointed or ':' not in demand:
                continue
            val = demand.split(':', 1)[1].strip()
            edge_n = re.search(BASE, val, re.I)
            if edge_n:
                if demand in heard:
                    yield (edge_n.group(1).strip(), True)
            elif val and (not re.search(' '.join(('\\b(?:not|no|other', 'than|unlike|alongside|listing|page|shown|at', 'most|at', 'least|under|over|below|above|less|more|between|within|up', 'to|or)\\b')), val, re.I)):
                yield (val, demand in heard)

    def _g6_review_brief(self, iids: list[str], clue: bool) -> str:
        instruction = VERIFY + '\n\nREQUEST:\n' + self.demand2 + '\n\nREQUIREMENTS IN FORCE:\n' + self.register()
        if clue:
            amounts = []
            for n_n, it_n in enumerate(self.items, 1):
                amounts.append(f'item {n_n}: quantity {it_n.amount}')
            instruction += '\n\nCART (planned, judge these listings as if they were carted at the requested quantities: ' + '; '.join(amounts) + '):\n' + ', '.join(iids)
        else:
            instruction += '\n\nCART:\n' + self.basket_text()
        records = []
        for i_n in iids:
            flag = 'SOLD OUT (the store says it can no longer be bought). ' if self.vended_out(i_n) else ''
            records.append(f'{i_n}: ' + flag + self.entry_of(i_n))
        instruction += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + '\n'.join(records)
        if self.digest.get('pick') in ('cheapest', 'most', 'least'):
            instruction += '\n\nOTHER LISTINGS SEEN (newest first):\n' + self.rest(iids, 15)
        return instruction

    def _g6_review_run(self, instruction: str, bound: float, clue: bool) -> Any:
        assesses = self.assesses(bound)
        obtained = None
        if len(assesses) > 1 and (not clue):
            if not self.audited_n and self.digest.get('pick') not in ('cheapest', 'most', 'least'):
                obtained = self.query_json(instruction, [LIGHTN], 3000)
                if obtained:
                    escalate = obtained.get('ok') is False and HEAVYN in self.assesses(bound)
                    assesses = [HEAVYN] if escalate else []
        if assesses and (not self.none_merit):
            obtained = self.query_json(instruction, assesses, 3000) or obtained
        return obtained

    def _g6_review_note(self, obtained: dict[str, Any]) -> str:
        problems_n = [str(p_n) for p_n in _wares(obtained.get('problems'))][:5]
        better_n: list[str] = []
        for b_n in _wares(obtained.get('better')):
            if str(b_n) in self.lines and (not self.vended_out(str(b_n))):
                better_n.append(str(b_n))
        text_n = 'A careful review says this cart is not right yet: ' + '; '.join(problems_n or ['see the requirements'])
        if better_n:
            text_n += '. Listings that look better: ' + ', '.join(better_n[:4])
        return text_n + '. Fix the cart (or show the review is wrong by re-reading the records), then check out.'

    def _g6_cluster_fits(self, tag: str, desired: list[tuple[str, str, set[str]]]) -> bool:
        if desired:
            here = _terms(tag)
            mine = _g6_long_words(_root(tag))
            for aisle, name_n, lengthy in desired:
                if here in (aisle, name_n) or mine & lengthy:
                    return True
        return bool(self.holder_of(tag))

    def _g6_widen_seen(self, op: str, spec: dict[str, Any], goal: str, choices: list[str], prior: int, answer: dict[str, Any]) -> None:
        viewed = answer.get('observation')
        self.perceive(op, viewed)
        self.memo_pane(op, viewed)
        lines = self.entries_in(viewed)
        if lines:
            if not self.details_from.get(op, False):
                self.details_from[op] = any((_detail_chunk(r_n) for __n, r_n in lines))
            self.contrast_fine = max(self.contrast_fine, len(lines))
        else:
            self.contrast_poor = min(self.contrast_poor or 999, len(choices))
        opened = len(self.whole) - prior
        self.journal.append({'kind': 'widen', 'ids': choices, 'opened': opened})
        if lines and self.details_from.get(op) is False:
            self.widen_singly_n()
            return
        if len(self.whole) != prior or len(choices) <= 2 or (not self.remaining() > 12):
            return
        issues: list[str] = []
        args_n = _form({goal: choices[:2]}, spec, 'args', issues)
        if issues:
            return
        answer = self.perform(op, args_n)
        if answer is None:
            return
        self.perceive(op, answer.get('observation'))
        seen_rows = self.entries_in(answer.get('observation'))
        self.contrast_fine = max(self.contrast_fine, len(seen_rows))

    def _g6_compare_ready(self, op: str | None, goal: str | None, spec: dict[str, Any]) -> bool:
        if not op or not goal:
            return False
        for k_n in spec.get('required') or []:
            if goal != k_n:
                return False
        return not self.contrast_plain()

    def _g6_opened_count(self, iids: list[str]) -> int:
        return len([i_n for i_n in iids if i_n in self.whole])

    def _g6_single_track(self, iids: list[str], desired: set[str], each_n: int) -> tuple[list[str], int]:
        unfolded = self._g6_opened_count(iids)
        reserve = [i_n for i_n in iids if i_n not in self.whole and self.purchasable(i_n)]
        if self.digest.get('pick') == 'cheapest':
            reserve.sort(key=self._g6_cost_rank)
        weight = {}
        for i_n in reserve:
            words = re.findall('[a-z0-9]{3,}', _serial(self.lines.get(i_n) or {}).lower())
            weight[i_n] = len(desired.intersection(words))
        reserve.sort(key=weight.__getitem__, reverse=True)
        room = each_n - unfolded
        return (reserve[:room if room > 0 else 0], max(0, 4 - unfolded))

    def _g6_cost_rank(self, iid: str) -> float:
        cost = self.cost_of(iid)
        return float('inf') if cost is None else cost

    def _g6_guarded_turns(self, cycle_ceiling: int, spend_ceiling: float) -> bool:
        try:
            self.turns_n(cycle_ceiling, spend_ceiling)
        except Exception as err:
            told = f'{type(err).__name__}: {err}'
            self.journal.append({'kind': 'crash', 'error': told[:400]})
            self.response_open_invocations()
            return False
        return True

def agent_main(trouble_payload: dict[str, Any]) -> list[dict[str, Any]]:
    session = None
    try:
        session = NvidiaBuyer(trouble_payload)
        return list(session.run_n())
    except Exception as err:
        if session is None:
            return [{'kind': 'crash', 'error': f'{type(err).__name__}: {err}'[:400]}]
        session.journal.append({'kind': 'crash', 'error': f'{type(err).__name__}: {err}'[:400]})
    return _g6_wind_up(session)
__all__ = ['agent_main']
