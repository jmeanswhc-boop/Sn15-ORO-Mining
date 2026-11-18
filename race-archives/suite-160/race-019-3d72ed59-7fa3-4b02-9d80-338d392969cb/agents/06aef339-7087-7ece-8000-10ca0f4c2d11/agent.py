"""Buys for one shopper inside an ORO store world.

Built on the public agents Halo v2 and YesYesYes v1, both forks of Know Thyself v1, itself built on KYO v6. From them come
the working shape: a light model walks the store with the world's own tools, code keeps the ledger and questions the
shopper, a heavy model reads the request and judges the hard calls.

New in this agent:
- A count of moves in place of a clock: model calls, store steps and stalls bound an episode, and no choice reads the time.
- Chains in code: a requirement whose value sits on another listing is a record with a relation and a walk. The walk runs
  early, seller first, reads each page with the tool that shows the listing's notes, and holds the ending while a page is unread.
- Questions that widen only while the shopper takes them: short lists first, one label after any reply asking for one thing
  at a time, every line asked about before any is asked twice, items named in the store's own shelf words.
- Line evidence in code: each requirement on each read line is proven, weak, silent or contradicted, and that decides the
  ending. A full cart orders, a line that is missing at the end declines, and a cheaper proven line replaces a dearer one.
- Fit read by the shape of codes rather than by sentences, checked on the cart, with parts searched by their series.
- A reader that turns rows and pages sent as text into the same records as rows sent as objects.
"""
from __future__ import annotations
import json
import random
import re
import time
from functools import lru_cache
from typing import Any
from src.agent.proxy_client import ProxyClient
LIGHT = 'anthropic/claude-haiku-5.5'
HEAVY = 'anthropic/claude-sonnet-5.5'
SPARE = 'openai/gpt-6-luna'
HEAVY_KIN = (HEAVY, 'anthropic/claude-sonnet-5', 'anthropic/claude-sonnet-4.6')
LOOP_CALLS = 42
ALL_CALLS = 54
SPEND_SOFT = 0.049
SPEND_HARD = 0.074
HEAVY_ROOM = 0.02
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3
AUTO_PAGES = 6
CHEAPER_OPENS = 3
LATE_STEPS = 8
CHAIN_PAGES = 9
CHAIN_READS = 18
CHAIN_TWINS = 8
CHAIN_HOLDS = 4
CODE_ROWS = 5
LIST_SIZE = 8
AMOUNT = '\\d[\\d.,]*\\s*[a-z"\']{1,12}'
FLOOR = '\\b(?:at least|no (?:less|fewer) than|min(?:imum)?)\\s+(' + AMOUNT + ')\\s*$'
QUERY_WORDS = '(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword'
TIGHT_MESSAGES = 2
SHORT_RUN = 45
RECOVERY_STEPS = 8
SWEEP_STEPS = 6
WALK_STEPS = 45
TWIN_OPENS = 3
ASK_BY = 72
LATE_BY = 83
WIND_DOWN = 87
QUESTION_BY = 91
SEARCH_BY = 98
HEAVY_STOP = 102
LAST_TURN = 111
CHECK_BY = 113
EMPTY_STOP = 117
CALL_STOP = 119
STORE_STOP = 122
STALL_MOVES = 20
STALL_LIMIT = 2
BUSY_CALLS = 4
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=60, max_retries=1, retry_delay=1)
GUIDE = 'You are Kyo, a careful buyer working for one shopper in an online store. You can act only by calling the store\'s tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.\n\nHOW THE STORE WORKS\n- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool\'s description and schema and use the exact argument names and allowed values.\n- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.\n- Search rows are summaries. Stores split a listing\'s record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Kyo knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.\n- Questions to the shopper are handled for you by Kyo\'s interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.\n- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Kyo carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store\'s tools (product questions go through Kyo\'s interviewer).\n\nWHAT COUNTS AS A FIT\n- The kind of product is decided by the listing\'s category label: only the label that matches the requested item as a whole counts, never a neighbouring one.\n- A requirement on a property is met when the listing\'s attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value\'s words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.\n- An attribute whose key names the requirement\'s property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing about the requirement. Kyo adds [evidence] lines after opened listings with the keys whose names match a requirement.\n- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.\n- When the category label itself states a requirement\'s value (a season or audience in the label), that requirement is met by the category.\n- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing\'s value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.\n- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.\n- When one requested item must fit another (an accessory for a device), choose the two together from the listings\' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.\n- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.\n- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.\n\nCHOOSING\n- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Kyo reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.\n- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.\n- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.\n- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.\n- A request for several different items needs one cart line per item, each at its requested quantity.\n\nCHANGES WHILE YOU WORK\n- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool\'s result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.\n- Re-read a chosen listing\'s record shortly before checking out, because price and stock can change silently.\n\nFINISHING\n- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.\n- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request\'s wording for that ending, and when a remark is needed, keep it short and name no listing.\n- Every reply must call at least one tool. Keep your text short: what you know, then the next move.'
BRIEF = 'You prepare a shopping task for an agent. Read the shopper\'s request and the store\'s tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:\n\n"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request\'s words, "shelf": the store\'s category label for it when a tool lists labels (copy it exactly), or the category the request itself names for it ("a coat in rainwear" gives "rainwear"), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request\'s own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property\'s value or limit outright; and never turn words that are part of what the product is called (a season or an audience named with the product) into musts: those belong to "name". A use or occasion the request makes a property of the product itself ("blazers for business wear", "a dress made for parties", "a bag, and it is for a wedding", "the occasion has to be party") IS a must on that property ("occasion: business"), unless the store\'s category label already names it; a particular person\'s event, a person or a setting the product is bought for ("to wear at his wedding", "for a colleague"), and how it should sit, feel or look in use, stay context. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.\n"budget": the most the shopper will pay in total as a number in the request\'s own unit, or null.\n"budget_unit": that unit in words (for example "cents"), or null.\n"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.\n"in_stock_only": true unless the request clearly accepts items that are not available.\n"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.\n"pick_measure": what is maximised or minimised for "most"/"least", else "".\n"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.\n"steps": how many steps, moves or tool calls the request allows (integer), or null.\n"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].\n"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.\n"wishes": a list of things the shopper would merely like.\n"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).\n"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper\'s approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.\n"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.\n"fits": a list of {"item": item number, "with": item number} for each requested item that the request says must work with, fit or suit ANOTHER requested item (the one the agent picks); [] when none.\n"browse": how a tool lists one store category, when one can: {"tool": exact tool name, "category": the argument that takes the category label, "max_price": the argument that takes a maximum price, or null, "fixed": the other arguments that put the tool in that mode, else {}}; null when no tool lists a category.\n"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing\'s attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing\'s current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).\n"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.\n"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.\n'
PLAN = 'You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property; a property the shopper has no view on gets "don\'t mind", and one the shopper does not think about at all gets a puzzled reply, so every property named spends part of a message.\n\nYou get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item\'s category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.\n\nChoose the properties in this order:\n0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.\n1. Hinted properties. The request\'s story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product\'s looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.\nName a hinted property by a listed key of that item when one fits the hint (a hint about who will wear it: the key for gender or audience; about its looks: the listed key for colour, pattern or finish that most of its listings carry); a property no listing of the item carries comes after the item\'s three most carried keys.\n2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item\'s listings; once five or more listings were read, avoid keys seen on only one or two of them.\nSkip a property the request already settles for that item (a requirement names that property; a key whose values only repeat a required value, such as a pattern key showing the required decoration, is NOT settled), a second name for a property already chosen, a key whose values each list several purchase options at once, identifiers, model names, warranty and other logistics. Pack sizes, counts and quantities per pack are properties like any other, and a key marked [not the stated ...] is another property than that requirement, whatever words its name shares with it. Rank every item on its own: its hinted properties first, then its listed keys.\n"label" is the property\'s plain name: for a listed key, all of the key\'s words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property\'s name, no example values, no brackets, no second property.\nChoose up to COUNT for each item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story\'s words it comes from, or an empty string"}]}, each item\'s picks in rank order'
HEAR = 'A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants them asked one at a time; not answers that only say they did not follow it or want it in simpler words], "budget": new total limit as a number or null}\n- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3"); the value is only what a listing could state, from the wanted value or its limit word to its end, without the framing the answer puts around it ("oval" for "matching oval", "settled, with oval" or "oval and nothing else", "at least 30 cm" for "equal to at least 30 cm").\n- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop\'s product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer\'s own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.\n- "free": the property of each question the shopper said they do not mind about.\n- Folksy or roundabout wording that names a value still states that value. Wording that refuses to bargain over a value (it will not be traded, given up or changed) states that value as wanted; write \"not X\" only when the answer rejects X itself.\n- The value is only what the property should be. Words that merely join a property to its value (such as matching, set to or coming in as) are not part of it: "size set to large" states large. This does not touch a value pointed at on another listing, which is copied whole.'
NOTE = 'While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:\n{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request\'s unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}\nA replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note.'
CHECK = 'You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.\nReading rules: a requirement is met when the listing\'s attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value\'s words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing\'s value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.\nWhen the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart\'s choice on that rule, the cart is wrong.\nReturn ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses.'
READING = 'the listing\'s attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value\'s words counts, such as "deep teal" for teal, limits include the edge); a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing\'s value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit.'
AUDIT = 'A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: ' + READING + '\nRecords marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.\nThe requirements were written down by a quick reader, so first compare each one with the REQUEST and the ANSWERS FROM THE SHOPPER. A requirement that only repeats the story (who it is for, when, where or why it will be used, what went wrong before, how it should feel) and that the request does not demand of the product goes under "not_firm". A value that picked up words which only join a property to its value goes under "corrected" with the plain value. Judge the listings without the not_firm requirements and with the corrected values. A requirement the request or an answer states outright stays, even when no listing meets it: ending without a purchase is then right.\nReturn ONE JSON object and nothing else: {"not_firm": [{"item": item number, "need": the requirement copied exactly}], "corrected": [{"item": item number, "need": the requirement copied exactly, "value": "property: plain value"}], "fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "search": [at most 4 of {"item": item number, "query": "a keyword search of two or three words, not tried yet, that could surface a fitting listing: another name for the product with one required value (long queries often match nothing)"}], "reason": "one sentence"}'
UNMIX = 'A shopper\'s request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request\'s wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}'
WALK = 'A shopping agent follows a requirement whose value sits on another listing. A page can state that value or send the reader on to a further listing for that property; the chain ends at the first page that does not send the reader on for that property. THE PAGE shows a listing\'s note and its record. Read the sections below and return ONE JSON object and nothing else:\n{"property": the property the requirement is about, "relation": how the wanted value stands to the value at the end of the chain: "same", "different", "near" (within a stated distance of it), "at_least", "at_most", or "partner" (the counterpart stated for it), "distance": the distance for "near" with its unit, else "", "is_target": true when THE PAGE is the listing under LOOKING FOR, false when its shop, its kind or a value of that description differs from what the page states (a value the page does not mention, or states in other words or units, counts as fitting) or the page itself says it is not that one, null when there is no page, "next": with no page, the listing the requirement points at; with a page, the listing it sends the reader on to for this property; else null. Give it as {"shop": the shop selling it, "kind": the kind of product, "values": [each value the description gives, bare, without the name of its property], "about": the description copied whole} ("a kettle from Ovarno with lid glass and capacity 2 litres" gives "Ovarno", "kettle", ["glass", "2 litres"]), "value": the property\'s value as THE PAGE states it, else "", "partner": the value THE PAGE or a rule under RULES SEEN says belongs with that value, read in the direction stated, else "", "rule": a sentence on THE PAGE that lists value pairs for this property, copied whole, else ""}\n"different" is only for a wanted value that must not be the one found: a firm refusal of anything else is "same". Only sending the reader on for this property counts: a sentence about another property, about other versions of the listing or about what this listing is not sends no one on. A page can point at a listing of any kind, and the property need not make sense for it.'
GATE = 'A shopping agent is about to STAGE. The shopper\'s request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product\'s properties do not do a step.\nWhen a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.\nReturn ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}'
SWAP = "A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: " + READING + '\nReturn ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line\'s item, at most 6], "reason": "one sentence"}'
SHAPE = 'A store tool returned the result below. Name its parts by the exact keys the result and the tool\'s parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing\'s price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}'
SAME = 'A shopper wants a property value, and a store listing words that property differently. For each numbered pair decide whether what the listing says means the wanted value. Spelling slips, word forms, word order and plain synonyms mean the same; one element of a list that means the wanted value is enough. An abbreviation or a code is not the word it stands for, and a different, narrower or wider value is not the same. Return ONE JSON object and nothing else: {"same": [numbers of the pairs that mean the same], "differ": [numbers of the other pairs]}'
SAME_PAIRS = 12
HUNTS = 3
HUNT_ROOM = 8
GUARD_KEYS = 3
GUARD_SHARE = 0.0
GUARD_GAP = 0.34
NOT_PROPERTY = '\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b'
WIDEN_EACH = 6
FIT_CODE = '[A-Z][A-Za-z]{0,7}(?:[- ]?\\d{3,6}|[- ]?\\d{1,5}[xX]{1,4})'
PART_CODE = '[A-Z]{1,5}\\d{0,3}-?\\d{3,7}'
PART_SERIES = '[A-Z]{1,5}\\d{1,3}'
FIT_PLAIN_WORDS = 10
NEED_SKIP = frozenset('a an the of in on at to for with and or only just up it its is be must has have that this'.split())
NOT_HEAD = "^\\s*(?:not|no|never|nothing|without|except|excluding|excludes?|avoid|none of|other than|anything (?:but|except|other than)|must not be|cannot be|can't be)\\b(?!-)"
EDGE_LOW = '\\b(?:at least|no (?:less|fewer) than|not (?:less|fewer) than|or (?:more|over|above|higher|greater|bigger|larger|longer|taller|wider)|min(?:imum)?|(?<!no )(?<!not )(?:more|greater) than|over|above)\\b|>'
EDGE_HIGH = '\\b(?:at most|no (?:more|greater) than|not (?:more|greater) than|or (?:less|fewer|under|below|lower|smaller|shorter)|max(?:imum)?|(?<!no )(?<!not )(?:less|fewer) than|under|below)\\b|<'
WORD_ENDS = ('', 's', 'es', 'e', 'y', 'ies', 'd', 'ed', 'ing', 'er', 'ers', 'ery')

def _code_names(stated: str, code: str) -> bool:
    """Whether a stated code or series names a model code: a series ends in x's in place of digits."""
    tail = re.search('(?<=\\d)x+$', stated)
    if not tail:
        return stated == code
    return code.startswith(stated[:tail.start()]) and len(code) == len(stated)

def _code_family(code: str) -> str:
    found = re.match('[a-z]+', code)
    return found.group(0) if found else ''

def _code_key(code: str) -> str:
    return re.sub('[\\s-]+', '', code).lower()

def _fit_tokens(sentence: str) -> list[str]:
    """The code-shaped tokens of a sentence, in order, as written."""
    return [hit.group(0) for hit in re.finditer('\\b(?:' + PART_CODE + '|' + FIT_CODE + '|' + PART_SERIES + ')\\b', sentence)]

def _fit_short(token: str) -> bool:
    """Whether a token has the shape of a series: letters and a few digits, too short for a model code."""
    return re.fullmatch(PART_SERIES, token) is not None and re.fullmatch(FIT_CODE, token) is None

def _fit_statements(text: str, own: str) -> list[list[str]]:
    """The sentences of a listing that state codes rather than describe the product: a sentence counts when it holds a
    code-shaped token and few other words. The field holding the listing's own bracket code is its name, not a statement."""
    found = []
    for line in text.split('\n'):
        if own and re.search('[\\[(]\\s*(?:[A-Za-z]+\\s+)?' + re.escape(own) + '\\s*[\\])]', line):
            continue
        for sentence in re.split('(?<=[.;!?])\\s+', line):
            tokens = _fit_tokens(sentence)
            rest = re.sub('\\b(?:' + '|'.join((re.escape(t) for t in tokens)) + ')\\b', ' ', sentence) if tokens else ''
            if tokens and len(re.findall("[^\\W\\d_][\\w'-]*", rest)) <= FIT_PLAIN_WORDS:
                found.append(tokens)
    return found

@lru_cache(maxsize=4096)
def _fit_marks(text: str) -> dict[str, Any]:
    """What a listing's text says about fit, by the shape of its tokens alone. `own`: the code in a bracket closing or opening
    a field. `works`: the codes it states, whole series (x's for digits) included.
    `deny`: codes a whole series named in the same sentence covers, which can only be its exceptions. `series`: short codes,
    letters then a few digits. A sentence holding a code of the listing's own family speaks about that other listing, not about
    this one: `also` when it pairs that code with others, `more` when the code stands alone (the page to read for the rest).
    `loose`: tokens written as a word and a number, which may be a second name but may as well be plain text."""
    own = re.search('[\\[(]\\s*(?:[A-Za-z]+\\s+)?(' + PART_CODE + ')\\s*[\\])]\\s*$', text, re.M) or re.search('^\\s*[\\[(]\\s*(?:[A-Za-z]+\\s+)?(' + PART_CODE + ')\\s*[\\])]', text, re.M)
    mine = own.group(1) if own else ''
    said = _fit_statements(text, mine)
    kin = {_code_family(_code_key(mine))} - {''}
    if not kin:
        kin = {_code_family(_code_key(t)) for tokens in said for t in tokens if _fit_short(t) and len(_code_family(_code_key(t))) > 1}
    works: list[str] = []
    deny: list[str] = []
    also: list[tuple[str, list[str]]] = []
    series: list[str] = []
    more = ''
    for tokens in said:
        if {_code_key(t) for t in tokens} == {_code_key(mine)}:
            works.append(_code_key(mine))
        tokens = [t for t in tokens if _code_key(t) != _code_key(mine)]
        near = [t for t in tokens if _code_family(_code_key(t)) in kin and (mine or _fit_short(t) or re.match(PART_SERIES + '-', t))]
        far = [t for t in tokens if t not in near]
        if near and far and mine:
            for token in near:
                also += [(_code_key(f), [_code_key(token)]) for f in far] if _fit_short(token) else [(_code_key(token), [_code_key(f) for f in far])]
            continue
        series += [t.upper() for t in near if _fit_short(t)]
        more = more or next((t.upper() for t in near if not _fit_short(t)), '')
        series += [t.upper() for t in far if _fit_short(t) and len(_code_family(_code_key(t))) > 1]
        stated = [_code_key(t) for t in far if not _fit_short(t)]
        whole = [c for c in stated if re.search('(?<=\\d)x+$', c)]
        deny += [c for c in stated if c not in whole and any((_code_names(w, c) for w in whole))]
        works += [c for c in stated if c not in deny]
    loose = [_code_key(t) for tokens in said for t in tokens if re.search('[a-z]', t.rstrip('xX'))]
    return {'own': mine.upper(), 'works': works, 'deny': deny, 'also': also, 'series': series, 'more': more, 'loose': loose}

def _fit_holds(names: set[str], deny: list[str], code: str) -> bool:
    """Whether a line fits a code: it names the code or a whole series holding it, and does not except it."""
    return code not in deny and any((_code_names(n, code) for n in names))

def _sentence_only(seen: Any) -> bool:
    """A store reply whose one value, notes aside, is a sentence."""
    if not isinstance(seen, dict):
        return False
    kept = [v for v in seen.values() if not isinstance(v, list)]
    if len(kept) != 1:
        return False
    return _sentence_only(kept[0]) if isinstance(kept[0], dict) else isinstance(kept[0], str) and ' ' in kept[0].strip()


def _fit_odd(names: dict[str, set[str]], deny: dict[str, list[str]], lead: list[str], loose: set[str]=frozenset()) -> list[tuple[str, list[str]]]:
    """Lines that do not share a code with the others of their family: (the code most of them share, the lines without it).
    Lines of no common family say nothing about each other. A line in `lead` is a device: all its codes are names of one thing,
    so a line naming one of them holds them all, and on a tie its code stands. Codes in `loose` can agree but start no family."""
    names = dict(names)
    for hub in lead:
        for ident in names:
            if ident not in lead and hub in names and any((_fit_holds(names[ident], deny.get(ident) or [], c) for c in names[hub])):
                names[ident] = names[ident] | names[hub]
    families: dict[str, set[str]] = {}
    for ident, codes in names.items():
        for code in codes - loose:
            families.setdefault(_code_family(code), set()).add(ident)
    found: list[tuple[str, list[str]]] = []
    for family, ids in sorted(families.items()):
        if len(ids) < 2 or not family:
            continue
        ranked = []
        for code in sorted({c for i in ids for c in names[i]}):
            holders = {i for i in ids if _fit_holds(names[i], deny.get(i) or [], code)}
            ranked.append((len(holders), len(holders & set(lead)), _code_family(code) == family, code, holders))
        best = max(ranked, key=lambda r: r[:3])
        odd = sorted(ids - best[4])
        if odd and not any((best[3] == code and odd == lines for code, lines in found)):
            found.append((best[3], odd))
    return found

def _dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        return str(value)

def _clip(value: Any, limit: int, depth: int=0) -> Any:
    if depth > 7:
        return '...'
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + '...'
    if isinstance(value, dict):
        return {k: _clip(v, limit, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip(v, limit, depth + 1) for v in value]
    return value

def _json_obj(text: Any) -> dict[str, Any] | None:
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
    left, right = (text.find('{'), text.rfind('}'))
    if 0 <= left < right:
        try:
            got = json.loads(text[left:right + 1])
            return got if isinstance(got, dict) else None
        except ValueError:
            return None
    return None

def _words(key: str) -> str:
    key = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(key))
    return re.sub('[\\s_.\\-]+', ' ', key).strip().lower()

def _snake(key: str) -> str:
    return _words(key).replace(' ', '_')

def _stem(text: str) -> frozenset[str]:
    return frozenset((w[:-1] if len(w) > 3 and w.endswith('s') else w for w in _words(text).split()))

def _roots(text: str) -> set[str]:
    out = set()
    for w in _words(text).split():
        w = w[:-2] if re.search('(?:ch|sh|ss|x|z)es$', w) else w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w
        if len(w) >= 3 and w not in ('and', 'for', 'the', 'with'):
            out.add(w)
    return out

def _spelled(text: str) -> frozenset[str]:
    """A name's words with plural -s and British -our folded, so "colours" and "color" compare equal."""
    return frozenset((re.sub('our$', 'or', w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w) for w in _words(text).split()))

def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    if not prop or not key:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)

def _tokens(text: Any) -> list[str]:
    """The words and numbers of a text in lower case; a thousands comma does not split a number."""
    return re.findall('\\d+(?:\\.\\d+)?|[^\\W\\d_]+', re.sub('(?<=\\d),(?=\\d{3}(?:\\D|$))', '', str(text).lower()))

def _alike(a: str, b: str) -> bool:
    """Whether two tokens are one word: a number by its value, a word up to a plural or other plain ending."""
    if a == b:
        return True
    if a[0].isdigit() or b[0].isdigit():
        return a[0].isdigit() and b[0].isdigit() and float(a) == float(b)
    shared = next((n for n, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return shared >= 3 and shared >= 0.7 * max(len(a), len(b)) and a[shared:] in WORD_ENDS and b[shared:] in WORD_ENDS

def _holds(words: tuple[str, ...], text: Any) -> bool:
    """Whether a text states every one of these words. A word right after a number is its unit: a short and a full form agree."""
    have = _tokens(text)
    for at, word in enumerate(words):
        if any((_alike(word, h) for h in have)):
            continue
        if at and words[at - 1][0].isdigit() and not word[0].isdigit():
            after = [have[n + 1] for n in range(len(have) - 1) if _alike(words[at - 1], have[n])]
            if not after or any((not h[0].isdigit() and (h.startswith(word) or word.startswith(h)) for h in after)):
                continue
        return len(words) >= 2 and ''.join(words) in ''.join(have)
    return bool(words)

def _number_off(words: tuple[str, ...], text: Any) -> bool:
    """Whether a text lacks a number these words give, in their own unit or with none: code decides that without a reader.
    The same amount in another unit is left to a reader."""
    have = _tokens(text)
    units = [have[n + 1] if n + 1 < len(have) and (not have[n + 1][0].isdigit()) else '' for n, h in enumerate(have) if h[0].isdigit()]
    for at, word in enumerate(words):
        if word[0].isdigit() and (not any((_alike(word, h) for h in have))):
            unit = words[at + 1] if at + 1 < len(words) and (not words[at + 1][0].isdigit()) else ''
            if not units or any((not unit or not u or unit.startswith(u) or u.startswith(unit) for u in units)):
                return True
    return False

@lru_cache(maxsize=1024)
def _need_form(value: str, own: str) -> tuple[str, tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...], float, str]:
    """A requirement's value by its shape: (kind, wanted, barred, number, unit). Kind is "low" or "high" for one number
    with a limit word, else "words": the alternatives wanted and the alternatives barred, each without the words in `own`
    (the property's and product's own names). A value is a limit only when it says so."""
    text = re.sub('(?<=\\d),(?=\\d{3}(?:\\D|$))', '', value.lower())
    text = re.sub('\\([^()]*\\)|\\band\\s+(?:nothing|no|none)\\b[^,;|]*$', ' ', text).strip(' ,;') or text
    numbers = re.findall('\\d+(?:\\.\\d+)?', text)
    low, high = (bool(re.search(EDGE_LOW, text)), bool(re.search(EDGE_HIGH, text)))
    if len(numbers) == 1 and low != high and not re.match(NOT_HEAD, re.sub('^\\s*not? (?:more|less|fewer|greater) than', '', text)):
        after = re.search(re.escape(numbers[0]) + '\\s*([^\\W\\d_]*)', text)
        return ('low' if low else 'high', (), (), float(numbers[0]), after.group(1) if after else '')
    skip = set(_tokens(own)) | NEED_SKIP
    wanted: list[tuple[str, ...]] = []
    barred: list[tuple[str, ...]] = []
    negative = False
    for part in re.split('[,;|]|\\bn?or\\b|(?<=[^\\W\\d_])\\s*/\\s*(?=[^\\W\\d_])', text):
        head = re.match(NOT_HEAD, part)
        if head:
            negative, part = (True, part[head.end():])
        elif re.search('\\bonly\\b', part):
            negative = False
        words = [w for w in _tokens(part) if w not in NEED_SKIP]
        words = [w for w in words if w not in skip] or words
        if words:
            (barred if negative else wanted).append(tuple(words))
    return ('words', tuple(wanted) or (() if barred else (tuple(_tokens(text)),)), tuple(barred), 0.0, '')

def _within(kind: str, number: float, unit: str, text: Any) -> bool | None:
    """Whether a text's numbers keep a limit; a range must lie wholly inside it, and one element of a list is enough. None
    when the text gives no number in the limit's own unit: conversions stay with the reader of the records."""
    answers = []
    for piece in re.split('[,;|]', re.sub('(?<=\\d),(?=\\d{3}(?:\\D|$))', '', str(text).lower())):
        found = re.findall('(\\d+(?:\\.\\d+)?)\\s*([^\\W\\d_]*)', piece)
        mine = [float(n) for n, u in found if not unit or not u or unit.startswith(u) or u.startswith(unit)]
        if mine:
            answers.append(min(mine) >= number if kind == 'low' else max(mine) <= number)
    return any(answers) if answers else None

def _key_rank(prop: str, key: str) -> int:
    """How a listing key names a requirement's property: 2 by the same name in any spelling, 1 by a longer name, 0 not."""
    a, b = (_spelled(prop), _spelled(key))
    if not a or not b:
        return 0
    x, y = sorted((''.join(_tokens(_words(prop))), ''.join(_tokens(_words(key)))), key=len)
    shared = next((n for n, (p, q) in enumerate(zip(x, y)) if p != q), len(x))
    if a == b or x == y or (shared >= 3 and len(x) >= 5 and len(y) == len(x) + 1 and y[shared + 1:] == x[shared:]) or (shared >= 3 and len(x) >= 6 and len(y) == len(x) and y[shared + 1:] == x[shared + 1:]):
        return 2
    return 1 if _names(a, b) else 0

def _kinds(schema: dict[str, Any]) -> list[str]:
    kind = schema.get('type')
    if isinstance(kind, str):
        return [kind]
    return [k for k in kind or [] if isinstance(k, str)]

def _fits_kind(value: Any, kind: str) -> bool:
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

def _shape(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    if not isinstance(schema, dict):
        return value
    kinds = _kinds(schema)
    if kinds and (not any((_fits_kind(value, k) for k in kinds))):
        for kind in kinds:
            if kind == 'integer' and isinstance(value, float) and value.is_integer():
                value = int(value)
                break
            if kind in ('integer', 'number') and isinstance(value, str) and re.fullmatch('\\s*-?\\d+(\\.\\d+)?\\s*', value):
                number = float(value)
                value = int(number) if kind == 'integer' or number.is_integer() else number
                break
            if kind == 'string' and isinstance(value, (int, float)) and (not isinstance(value, bool)):
                value = str(value)
                break
            if kind == 'boolean' and isinstance(value, str) and (value.strip().lower() in ('true', 'false')):
                value = value.strip().lower() == 'true'
                break
            if kind == 'array' and (not isinstance(value, list)) and (value is not None):
                value = [value]
                break
        if not any((_fits_kind(value, k) for k in kinds)):
            faults.append(f"{where} should be {'/'.join(kinds)}")
            return value
    options = schema.get('enum')
    if isinstance(options, list) and value not in options:
        if isinstance(value, str):
            low = value.strip().lower()
            match = [o for o in options if isinstance(o, str) and o.lower() == low]
            if match:
                value = match[0]
        if value not in options:
            faults.append(f'{where}: allowed values are {options}')
    if isinstance(value, str) and isinstance(schema.get('maxLength'), int) and (len(value) > schema['maxLength']):
        value = value[:schema['maxLength']]
    if isinstance(value, str) and isinstance(schema.get('minLength'), int) and (len(value) < schema['minLength']):
        faults.append(f'{where} is too short')
    if isinstance(value, (int, float)) and (not isinstance(value, bool)):
        if isinstance(schema.get('minimum'), (int, float)) and value < schema['minimum']:
            faults.append(f"{where} is below {schema['minimum']}")
        if isinstance(schema.get('maximum'), (int, float)) and value > schema['maximum']:
            faults.append(f"{where} is above {schema['maximum']}")
    if isinstance(value, list):
        if isinstance(schema.get('maxItems'), int) and len(value) > schema['maxItems']:
            faults.append(f"{where} allows at most {schema['maxItems']} entries")
        if isinstance(schema.get('minItems'), int) and len(value) < schema['minItems']:
            faults.append(f"{where} needs at least {schema['minItems']} entries")
        if isinstance(schema.get('items'), dict):
            value = [_shape(v, schema['items'], f'{where}[{i}]', faults) for i, v in enumerate(value)]
    if isinstance(value, dict):
        props = schema.get('properties') if isinstance(schema.get('properties'), dict) else {}
        if schema.get('additionalProperties') is False:
            for name in [n for n in value if n not in props]:
                faults.append(f'{where}.{name} is not an allowed argument')
        for name in schema.get('required') or []:
            if name not in value:
                faults.append(f'{where}.{name} is required')
        value = {k: _shape(v, props.get(k), f'{where}.{k}', faults) if k in props else v for k, v in value.items()}
    return value

def _items(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, (str, dict)) and value:
        return [value]
    return []

def _count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.0+)?\\s*', value):
        return int(float(value))
    return None

def _id_like(value: Any, named: bool=False) -> bool:
    if not isinstance(value, str) or not 3 <= len(value.strip()) <= 64 or ' ' in value.strip():
        return False
    return named or bool(re.search('\\d', value))

def _learn_id(rows: list[dict[str, Any]]) -> str | None:
    best = None
    for key in rows[0]:
        values = [row.get(key) for row in rows]
        if not all((_id_like(v, 'id' in str(key).lower()) for v in values)):
            continue
        values = [v.strip() for v in values]
        if len(rows) > 1 and len(set(values)) < len(values):
            continue
        score = ('id' in str(key).lower(), sum((len(v) for v in values)) / len(values))
        if best is None or score > best[0]:
            best = (score, key)
    return best[1] if best else None

def _in_rows(node: Any, ident: str, depth: int=0) -> bool:
    if depth > 5:
        return False
    if isinstance(node, list):
        return any((isinstance(v, dict) and ident in [str(x).strip() for x in v.values() if isinstance(x, str)] for v in node)) or any((_in_rows(v, ident, depth + 1) for v in node))
    if isinstance(node, dict):
        return any((_in_rows(v, ident, depth + 1) for v in node.values()))
    return False

def _fact_block(node: Any) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    best = None
    for value in node.values():
        if isinstance(value, dict) and (len(value) >= 2 or isinstance(value, _Facts)):
            scalars = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and (not isinstance(v, bool))}
            if (len(scalars) >= 2 or (scalars and isinstance(value, _Facts))) and (best is None or len(scalars) > len(best)):
                best = scalars
    return best

class _Facts(dict):
    """The pairs of a fact block read out of a listing string: a real block, even with a single pair."""

PART_MARKS = ((' | ', ': '), ('\n', ': '), ('\n', ' -> '), (' | ', ' -> '), (' | ', '='))
PAIR_MARKS = (('; ', '='), (', ', ': '), ('; ', ': '), (', ', '='))
BARE_WORDS = {'true': True, 'yes': True, 'false': False, 'no': False}

@lru_cache(maxsize=None)
def _part_head(marks: tuple[str, str]) -> Any:
    return re.compile('(?:^|' + re.escape(marks[0]) + ')([^\\W\\d][\\w .()/\\-]{0,39}?)' + re.escape(marks[1]))

def _label_style(label: str) -> set[str]:
    return {'a' if c.islower() else 'A' if c.isupper() else '9' if c.isdigit() else c for c in label}

def _bare(text: str) -> tuple[bool, Any]:
    if re.fullmatch('-?\\d{1,18}', text):
        return (True, int(text))
    if re.fullmatch('-?\\d{1,18}\\.\\d{1,18}', text):
        return (True, float(text))
    low = text.lower()
    return (True, BARE_WORDS[low]) if low in BARE_WORDS else (False, text)

def _pairs(text: str) -> dict[str, str] | None:
    """{key: value} of a fact block, by shape. Nothing is escaped: a piece with no key belongs to the value before it. A colon is common in prose
    and an equals sign is not, so colon pairs must fill the whole text. None when the text is no block."""
    single = None
    for marks in PAIR_MARKS:
        out: dict[str, str] = {}
        last = None
        loose = 0
        for piece in text.split(marks[0]):
            key, mark, value = piece.partition(marks[1])
            if mark and key.strip() and key not in out and len(key) <= 100 and '. ' not in key:
                out[key] = value
                last = key
            elif last is None:
                out = {}
                break
            else:
                out[last] += marks[0] + piece
                loose += 1
        tight = marks[1] == '='
        if len(out) >= (2 if tight else 3) and loose <= (len(out) if tight else 0):
            return out
        single = single or (out if tight and len(out) == 1 and not loose else None)
    return single

class TextReader:
    """Reads listings that a store sends as strings into the records a store of dicts would send.

    Nothing in such a string is escaped, so a title or a blurb can hold a separator and a word that looks like a label. A part
    therefore starts only at a label this store really uses: one that most strings of a result share, or, on a lone string, one
    written like the labels already learnt. A label whose values are mostly pairs is a fact block. Any string that fits no shape
    stays as it came, and a result with no such string comes back as the same object.
    """

    def __init__(self) -> None:
        self.marks: tuple[str, str] | None = None
        self.labels: set[str] = set()
        self.style: set[str] = set()
        self.wordy: set[str] = set()
        self.seen: dict[str, int] = {}
        self.paired: dict[str, int] = {}
        self.id_label: str | None = None

    def read(self, node: Any, depth: int=0) -> Any:
        if isinstance(node, str):
            return self.cards([node])[0]
        if isinstance(node, list):
            return self.cards(node) if node and all((isinstance(v, str) for v in node)) else node
        if not isinstance(node, dict) or depth > 3:
            return node
        small = len(node) <= 3
        out = {k: self.read(v, depth + 1) if isinstance(v, list) or (small and isinstance(v, (str, dict))) else v for k, v in node.items()}
        if all((out[k] is node[k] for k in node)):
            return node
        only = next(iter(out.values()))
        return only if len(out) == 1 and isinstance(only, dict) else out

    def spots(self, text: str) -> list[tuple[str, int, int]]:
        """Where a part could start, by shape alone: (label, start of its mark, start of its value). Empty for a string that is no listing."""
        for marks in [self.marks] if self.marks else PART_MARKS:
            found = [(m.group(1), m.start(), m.end()) for m in _part_head(marks).finditer(text)]
            if len(found) < 3 or found[0][1] != 0:
                continue
            ends = [spot[1] for spot in found[1:]] + [len(text)]
            if found[0][0] == self.id_label or any((_id_like(text[spot[2]:end]) and (not _bare(text[spot[2]:end])[0]) for spot, end in zip(found, ends))):
                self.marks = marks
                return found
        return []

    def learn(self, listings: list[list[tuple[str, int, int]]]) -> None:
        votes: dict[str, int] = {}
        for spots in listings:
            for label in {spot[0] for spot in spots}:
                votes[label] = votes.get(label, 0) + 1
        for label, count in votes.items():
            if len(listings) >= 3 and count * 2 > len(listings):
                self.labels.add(label)
                self.style |= _label_style(label)
        self.labels.update((k for k in votes if not self.style or _label_style(k) <= self.style))

    def split(self, text: str, spots: list[tuple[str, int, int]]) -> list[tuple[str, str]]:
        kept: dict[str, tuple[str, int, int]] = {}
        for spot in spots:
            if spot[0] in self.labels and spot[0] not in kept:
                kept[spot[0]] = spot
        parts = list(kept.values())
        if len(parts) < 3 or parts[0][1] != 0:
            return []
        ends = [spot[1] for spot in parts[1:]] + [len(text)]
        return [(spot[0], text[spot[2]:end]) for spot, end in zip(parts, ends)]

    def record(self, parts: list[tuple[str, str]]) -> dict[str, Any]:
        """A listing's parts as a record. Prose can read like pairs, so at most one part is the fact block: the label that is most often pairs, then the part with the most of them."""
        blocks = []
        for label, text in parts:
            pairs = _pairs(text)
            if pairs and self.paired.get(label, 0) * 2 >= self.seen[label]:
                blocks.append((self.paired[label] / self.seen[label], len(pairs), label, pairs))
        block = max(blocks, key=lambda found: found[:2]) if blocks else None
        out: dict[str, Any] = {}
        for label, text in parts:
            if block and label == block[2]:
                out[label] = _Facts(block[3])
            elif text.lower() in ('none', 'null'):
                out[label] = None
            else:
                out[label] = text if label in self.wordy else _bare(text)[1]
        return out

    def cards(self, texts: list[str]) -> list[Any]:
        """Each string of one result as a record, or as it came when it is no listing."""
        spots = [self.spots(text) for text in texts]
        if not any(spots):
            return texts
        self.learn([s for s in spots if s])
        cut = [self.split(text, s) for text, s in zip(texts, spots)]
        before = set(self.seen)
        for parts in cut:
            for label, text in parts:
                self.seen[label] = self.seen.get(label, 0) + 1
                if len(_pairs(text) or ()) >= (1 if before and label not in before else 2):
                    self.paired[label] = self.paired.get(label, 0) + 1
                if text and text.lower() not in ('none', 'null') and (not _bare(text)[0]):
                    self.wordy.add(label)
        records = [self.record(parts) for parts in cut if parts]
        if records and not self.id_label:
            self.id_label = _learn_id(records)
        made = iter(records)
        return [next(made) if parts else text for text, parts in zip(texts, cut)]

class Item:

    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []

class Chain:

    def __init__(self, number: int, need: str):
        self.number = number
        self.need = need
        self.prop = ''
        self.relation = ''
        self.distance = ''
        self.target: dict[str, Any] = {}
        self.walked: list[str] = []
        self.skipped: set[str] = set()
        self.fresh: list[str] = []
        self.queries: list[str] = []
        self.more: dict[str, Any] | None = None
        self.rules: list[str] = []
        self.status = 'open'
        self.value = ''
        self.partner = ''
        self.reads = 0
        self.tried = 0
        self.misses = 0
        self.spare: tuple[str, str, dict[str, Any]] | None = None
        self.deep = 0

    def wanted(self) -> str:
        return self.partner if self.relation == 'partner' else self.value if self.relation == 'same' else ''

class Kyo:

    def __init__(self, problem: dict[str, Any]):
        environment = problem['environment']
        self.binding = dict(environment['binding'])
        self.view = environment['policy_view']
        self.request = str(self.view.get('query') or '')
        self.tools = [t for t in self.view.get('tools') or [] if isinstance(t, dict) and isinstance(t.get('function'), dict) and isinstance(t['function'].get('name'), str)]
        self.schemas = {t['function']['name']: t['function'].get('parameters') if isinstance(t['function'].get('parameters'), dict) else {'type': 'object', 'properties': {}} for t in self.tools}
        steps = _count(self.view.get('max_steps'))
        if steps is None and isinstance(self.view.get('max_steps'), float):
            steps = int(self.view['max_steps'])
        self.max_steps = steps if steps and steps > 0 else 30
        self.step = 0
        self.finished = bool(self.view.get('finished'))
        self.broke = False
        self.no_credit = False
        self.waited = 0
        self.sent = 0
        self.stalls = 0
        self.env_stalls = 0
        self.misses = 0
        self.cost = 0.0
        self.calls = 0
        self.loop_calls = 0
        self.trace: list[dict[str, Any]] = []
        self.brief: dict[str, Any] = {}
        self.roles: dict[str, str] = {}
        self.items: list[Item] = []
        self.budget: Any = None
        self.msg_limit = 0
        self.msg_used = 0
        self.process: list[dict[str, Any]] = []
        self.reserve = 0
        self.gate_msgs = 0
        self.gate_ok: set[tuple[str, str]] = set()
        self.done_log: list[str] = []
        self.done_steps: list[int] = []
        self.asked = False
        self.qa: list[tuple[str, str]] = []
        self.rows: dict[str, dict[str, Any]] = {}
        self.full: dict[str, dict[str, Any]] = {}
        self.seen_order: list[str] = []
        self.heard: set[str] = set()
        self.text_reader = TextReader()
        self.notes: list[str] = []
        self.version = 0
        self.cart_touched = False
        self.cart_at = -1
        self.cart_seen_at = -1
        self.cart_view: Any = None
        self.note_named = False
        self.cart_lines: dict[str, int] = {}
        self.checked: set[str] = set()
        self.audited = False
        self.offer_seen: dict[str, int] = {}
        self.gone: dict[str, int] = {}
        self.unavailable: set[str] = set()
        self.sold_lines: set[str] = set()
        self.note_sold = False
        self.held = 0
        self.id_args: set[tuple[str, str]] = set()
        self.cart_said = ''
        self.cart_plain = False
        self.looked: set[int] = set()
        self.refused_at = -1
        self.leads: list[str] = []
        self.refuse_blocks = 0
        self.pushed = False
        self.built: list[str] = []
        self.dropped: set[str] = set()
        self.late_held = False
        self.lead_items: dict[str, list[str]] = {}
        self.verdicts: dict[str, bool] = {}
        self.confirmed: str | None = None
        self.completed = False
        self.link_tried: set[str] = set()
        self.adopted: dict[str, int] = {}
        self.id_key: str | None = None
        self.facts_from: dict[str, bool] = {}
        self.queued: list[str] = []
        self.buys = 0
        self.refusals = 0
        self.recent: list[str] = []
        self.chat: list[dict[str, Any]] = []
        self.opened: set[str] = set()
        self.refs: list[tuple[int, str]] = []
        self.ref_values: dict[str, str] = {}
        self.chains: list[Chain] = []
        self.row_keys: dict[str, list[int]] = {}
        self.note_fields: dict[str, dict[str, bool]] = {}
        self.page_notes: dict[str, dict[str, str]] = {}
        self.walk_steps = 0
        self.last_found: list[str] = []
        self.dead_forms: set[str] = set()
        self.dead_codes: set[str] = set()
        self.dead_filters: set[str] = set()
        self.trail_blocks = 0
        self.compare_ok = 0
        self.compare_bad = 0
        self.tool_keys: dict[str, set[str]] = {}
        self.view_keys: dict[str, tuple[int, set[str]]] = {}
        self.read_cache: dict[str, tuple[int, int, str]] = {}
        self.pages_read = 0
        self.shapes: dict[str, dict[str, Any] | None] = {}
        self.price_key: str | None = None
        self.stock_key: str | None = None
        self.cheaper_marks: set[str] = set()
        self.wider_held = False
        self.paged_obs: list[Any] = []
        self.browse: dict[str, Any] | None = None
        self.browse_misses = 0
        self.model_busy: dict[str, float] = {}
        self.swept: set[str] = set()
        self.sweep_spent = 0
        self.plan_order: list[int] = []
        self.lanes: dict[int, list[str]] = {}
        self.lane_found: dict[int, int] = {}
        self.lane_spent: dict[int, int] = {}
        self.lane_used: dict[int, set[str]] = {}
        self.late_asked = False
        self.follow_up: list[tuple[int, list[str]]] = []
        self.listing = True
        self.line_holds = 0
        self.bound: list[tuple[int, int]] = []
        self.fit_holds = 0
        self.searched: list[dict[str, Any]] = []
        self.same_pairs: dict[tuple[str, str], bool] = {}
        self.last_rows: Any = None
        self.hunted: set[tuple[int, str]] = set()
        self.swaps: dict[int, int] = {}
        self.short_holds = 0
        self.swap_holds = 0
        self.end_tried = False
        self.early_done = False
        self.key_memo: dict[tuple[int, str, int, int], set[str]] = {}

    def left(self) -> int:
        return self.max_steps - self.step

    def moves(self) -> int:
        return self.sent + self.step + STALL_MOVES * (self.stalls + self.env_stalls) + self.waited // 2

    def msgs_free(self) -> int:
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)

    def cart_mark(self) -> str:
        return _dump(sorted(self.cart_lines.items()))

    def page_tools(self) -> set[str]:
        page = self.roles.get('offer') or self.roles.get('open')
        return {t for t in (page, self.roles.get('compare')) if t}

    def facts_tool(self) -> str | None:
        known = next((t for t, ok in self.facts_from.items() if ok and self.single_target(t)), None)
        if known:
            return known
        tool = self.roles.get('open')
        return tool if self.facts_from.get(tool or '') is not False and self.single_target(tool) else None

    def compare_facts(self) -> bool | None:
        return self.facts_from.get(self.roles.get('compare') or '')

    def compare_bare(self) -> bool:
        seen = self.compare_facts()
        return seen is False or (seen is None and (self.facts_from.get(self.facts_tool() or '') is True or bool(self.roles.get('offer'))))

    def gate_due(self, stage: str) -> bool:
        return any((p['when'] == stage for p in self.process)) and (stage, self.cart_mark() + f'#{self.version}') not in self.gate_ok

    def tail(self) -> int:
        talk = any((p['when'] == 'order' and p['messages'] > 0 for p in self.process))
        spare = 1 if self.buys == 0 else 0
        return spare + (2 if talk and self.msg_used < self.msg_limit and self.gate_due('order') else 1)

    def judges(self, limit: float=SPEND_SOFT) -> list[str]:
        if self.cost <= limit and self.cost + HEAVY_ROOM <= SPEND_HARD:
            return [HEAVY, LIGHT]
        return [LIGHT] if self.cost <= SPEND_HARD else []

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None=None, deep: bool=False) -> dict[str, Any] | None:
        """One model reply. A heavy model at its request limit hands the call to the next of its kin at once and rests for
        some model calls. Kin answer at low reasoning effort unless the call is `deep`."""
        if self.moves() >= HEAVY_STOP:
            models = list(dict.fromkeys((LIGHT if m == HEAVY else m for m in models)))
        for model in models:
            for choice in (HEAVY_KIN if model == HEAVY else (model,)):
                if self.calls >= ALL_CALLS or self.moves() >= CALL_STOP or self.stalls >= STALL_LIMIT:
                    self.trace.append({'kind': 'late_call_skipped', 'moves': self.moves()})
                    return None
                if self.sent < self.model_busy.get(choice, 0):
                    continue
                self.calls += 1
                tries = 0
                while True:
                    effort = 'low' if model == HEAVY and choice != HEAVY and not deep else None
                    message, status = self.call_model(choice, messages, max_tokens, tools, effort)
                    if message is not None:
                        return message
                    if status in (None, 408):
                        self.stalls += 1
                    if status == 429 and model == HEAVY:
                        self.model_busy[choice] = self.sent + BUSY_CALLS
                        self.calls -= 1
                        break
                    passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                    if not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL:
                        break
                    if self.moves() + WAITS[tries] // 2 >= CALL_STOP:
                        break
                    pause = WAITS[tries] * (0.8 + 0.4 * random.random())
                    self.waited += WAITS[tries]
                    tries += 1
                    time.sleep(pause)
                if status == 402:
                    self.no_credit = True
                    return None
                if status != 429:
                    break
        return None

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None, effort: str | None=None) -> tuple[dict[str, Any] | None, Any]:
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
        if effort:
            body['reasoning'] = {'effort': effort}
        self.sent += 1
        result = _llm.post_verbose('/inference/chat/completions', body)
        data = result.data
        if isinstance(data, dict) and data.get('choices'):
            usage = data.get('usage') or {}
            if isinstance(usage.get('cost'), (int, float)):
                self.cost += float(usage['cost'])
            message = data['choices'][0].get('message') or {}
            if message.get('content') or message.get('tool_calls'):
                return (message, 200)
            self.trace.append({'kind': 'llm_fail', 'model': model, 'status': 0})
            return (None, 0)
        error = result.error or {}
        status = error.get('status')
        if isinstance(data, dict) and isinstance(data.get('error'), dict):
            status = _count(data['error'].get('code'))
            error = {'body': data['error'].get('message')}
        self.trace.append({'kind': 'llm_fail', 'model': model, 'status': status, 'body': str(error.get('body') or '')[:200]})
        return (None, status)

    def ask_json(self, prompt: str, models: list[str], max_tokens: int=1800, deep: bool=False) -> dict[str, Any] | None:
        for model in models:
            message = self.think([{'role': 'user', 'content': prompt}], [model], max_tokens, deep=deep)
            got = _json_obj((message or {}).get('content'))
            if got is not None:
                return got
            if self.broke or self.no_credit or self.calls >= ALL_CALLS:
                return None
        return None

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(self.binding)
        payload['step'] = self.step + 1
        payload['action'] = {'name': name, 'args': args}
        for attempt in range(4):
            result = _env.post_verbose('/environment/call', payload)
            if isinstance(result.data, dict):
                data = result.data
                self.step = data['step'] if isinstance(data.get('step'), int) else self.step + 1
                if data.get('finished'):
                    self.finished = True
                data['observation'] = self.read_text(data.get('observation'))
                seen = data.get('observation')
                self.saw_listing(name, args, seen)
                if name in self.page_tools() and (not (isinstance(seen, dict) and seen.get('error'))):
                    for value in args.values():
                        self.opened.update((v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)))
                return data
            error = result.error or {}
            status = error.get('status')
            body = str(error.get('body') or '')
            self.trace.append({'kind': 'env_fail', 'status': status, 'body': body[:200]})
            if status in (None, 408):
                self.env_stalls += 1
                if attempt >= 1 or self.env_stalls >= STALL_LIMIT:
                    return None
            elif status == 429 or (isinstance(status, int) and status >= 500):
                time.sleep(1 + attempt)
            if status == 409:
                low = body.lower()
                if 'finish' in low or 'quarantin' in low or 'finaliz' in low:
                    self.finished = True
                    return None
                found = re.search('expected (\\d+)', body)
                if found:
                    self.step = int(found.group(1)) - 1
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

    def read_request(self) -> None:
        lines = []
        for tool in self.tools:
            fn = tool['function']
            lines.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {_dump(fn.get('parameters') or {})}")
        prompt = BRIEF + '\nTOOLS:\n' + '\n'.join(lines) + '\n\nREQUEST:\n' + self.request
        if self.view.get('opening') is not None:
            prompt += '\n\nOPENING NOTE FROM THE STORE:\n' + _dump(_clip(self.view.get('opening'), 1500))
        brief = self.ask_json(prompt, [LIGHT], 4000) or {}
        if not self.no_credit and (not _items(brief.get('items')) or not isinstance(brief.get('roles'), dict) or (not brief['roles'].get('buy'))):
            brief = self.ask_json(prompt, [HEAVY, LIGHT], 2500) or brief
        elif not self.no_credit and self.shared_shelf(brief):
            again = self.ask_json(prompt, [HEAVY], 2500) or {}
            if _items(again.get('items')) and isinstance(again.get('roles'), dict) and again['roles'].get('buy'):
                brief = again
        self.brief = brief
        self.read_browse(brief.get('browse'))
        roles = brief.get('roles') if isinstance(brief.get('roles'), dict) else {}
        self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.schemas}
        for key in ('buy_args', 'refuse_args'):
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        buy, refuse = (self.roles.get('buy'), self.roles.get('refuse'))
        if buy and buy == refuse and (not brief['buy_args'] or not brief['refuse_args'] or brief['buy_args'] == brief['refuse_args']):
            ask = f'The store tool {buy} both checks out and ends without a purchase, depending on its arguments. Its schema: ' + _dump(self.schemas.get(buy)) + '\nReturn ONE JSON object and nothing else: {"buy_args": the fixed arguments that check out, "refuse_args": the fixed arguments that end without a purchase} (only the arguments that choose the mode).'
            fixed = self.ask_json(ask, [LIGHT, HEAVY], 1500) or {}
            if isinstance(fixed.get('buy_args'), dict) and isinstance(fixed.get('refuse_args'), dict) and (fixed['buy_args'] != fixed['refuse_args']):
                brief['buy_args'], brief['refuse_args'] = (fixed['buy_args'], fixed['refuse_args'])
        for entry in _items(brief.get('items')):
            if not isinstance(entry, dict):
                continue
            quantity = _count(entry.get('quantity'))
            count = quantity if quantity and quantity > 0 else 1
            musts = [str(m) for m in _items(entry.get('musts')) if m]
            self.items.append(Item(str(entry.get('name') or 'the product'), str(entry.get('shelf') or ''), count, musts))
        merged: list[Item] = []
        became: dict[int, int] = {}
        for old, item in enumerate(self.items, 1):
            twin = next((m for m in merged if item.shelf.strip() and m.qty == item.qty and _words(m.shelf) == _words(item.shelf) and (not any((_words(a.split(':')[0]) == _words(b.split(':')[0]) and _words(a.partition(':')[2]) != _words(b.partition(':')[2]) for a in m.musts for b in item.musts)))), None)
            if twin is None:
                merged.append(item)
                became[old] = len(merged)
            else:
                became[old] = merged.index(twin) + 1
                twin.musts += [m for m in item.musts if m not in twin.musts]
                self.trace.append({'kind': 'items_merged', 'into': twin.name, 'dropped': item.name})
        self.items = merged
        if not self.items:
            self.items = [Item('the requested product', '', 1, [])]
        for pair in _items(brief.get('fits')):
            if isinstance(pair, dict):
                part, main = (became.get(_count(pair.get('item')) or 0), became.get(_count(pair.get('with')) or 0))
                if part and main and (part != main) and (1 <= part <= len(self.items)) and (1 <= main <= len(self.items)) and ((part, main) not in self.bound):
                    self.bound.append((part, main))
        for lookup in _items(brief.get('lookups')):
            text = str(lookup).strip()
            if not any((text in item.musts for item in self.items)):
                key = _words(text.split(':')[0])
                text = next((m for item in self.items for m in item.musts if _words(m.split(':')[0]) == key), text)
            number = next((n for n, item in enumerate(self.items, 1) if text in item.musts), 1)
            if text and (number, text) not in self.refs:
                self.refs.append((number, text))
        self.unmix()
        budget = brief.get('budget')
        if isinstance(budget, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', budget):
            budget = float(budget)
        if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0):
            self.budget = budget
        limit = _count(brief.get('messages'))
        if limit is not None and limit >= 0:
            self.msg_limit = limit
        elif self.roles.get('ask'):
            self.msg_limit = 3
        steps = _count(brief.get('steps'))
        if steps is not None and 10 <= steps < self.max_steps:
            self.max_steps = steps
        for entry in _items(brief.get('process')):
            if isinstance(entry, dict) and str(entry.get('step') or '').strip():
                when = str(entry.get('when') or '').strip().lower()
                self.process.append({'step': str(entry['step']).strip(), 'when': when if when in ('choose', 'order', 'end') else 'order', 'messages': max(0, _count(entry.get('messages')) or 0)})
        if self.roles.get('ask'):
            talk = [p['messages'] for p in self.process if p['when'] in ('order', 'end') and p['messages']]
            self.reserve = min(max(talk), self.msg_limit) if talk else 0

    def read_browse(self, spec: Any) -> None:
        if not isinstance(spec, dict) or spec.get('tool') not in self.schemas:
            return
        props = self.schemas[spec['tool']].get('properties') or {}
        category, price = (spec.get('category'), spec.get('max_price'))
        if category not in props:
            return
        fixed = spec.get('fixed') if isinstance(spec.get('fixed'), dict) else {}
        self.browse = {'tool': spec['tool'], 'category': category, 'price': price if price in props else None, 'fixed': {k: v for k, v in fixed.items() if k in props and k not in (category, price)}}

    @staticmethod
    def shared_shelf(brief: dict[str, Any]) -> bool:
        shelves = [_words(str(e.get('shelf') or '')) for e in _items(brief.get('items')) if isinstance(e, dict)]
        shelves = [s for s in shelves if s]
        return len(shelves) != len(set(shelves))

    def unmix(self) -> None:
        for number, item in enumerate(self.items, 1):
            pointers = [need for n, need in self.refs if n == number and need in item.musts]
            rest = [m for m in item.musts if m not in pointers]
            if not pointers or not rest:
                continue
            prompt = UNMIX + '\n\nREQUEST:\n' + self.request + f'\n\nPRODUCT: {item.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n' + '\n'.join(pointers) + '\nOTHER REQUIREMENTS:\n' + '\n'.join(rest)
            got = self.ask_json(prompt, [HEAVY, LIGHT], 1500) or {}
            moved = [m for m in rest if m in [str(x).strip() for x in _items(got.get('describe_other'))]]
            if not moved:
                continue
            item.musts = [m for m in item.musts if m not in moved]
            joined = pointers[0] + ' (that listing: ' + '; '.join(moved) + ')'
            item.musts = [joined if m == pointers[0] else m for m in item.musts]
            self.refs = [(n, joined if (n, need) == (number, pointers[0]) else need) for n, need in self.refs]
            self.trace.append({'kind': 'unmix', 'item': number, 'moved': moved})

    def described(self, number: int, need: str) -> bool:
        if need in [m for _, m in self.refs] or need in [str(x) for x in _items(self.brief.get('lookups'))]:
            return False
        prop, _, value = need.partition(':')
        if not prop.strip() or len(re.findall('[a-z0-9]+', value.lower())) < 4:
            return False
        if re.search('\\b(?:listing|page|shown|other than|unlike|same as)\\b', value, re.I):
            return False
        item = self.items[number - 1]
        if any(({w for w in _stem(other.name) if len(w) >= 4} & _stem(value) for other in self.items if other is not item)):
            return False
        cat_key = self.category_key()
        keys: set[str] = set()
        read = 0
        for record in self.full.values():
            if cat_key and (not self.owns(item, str(record.get(cat_key) or ''))):
                continue
            block = _fact_block(record) or {}
            if block:
                read += 1
                keys |= {_words(k) for k in block}
        return read >= 3 and (not any((_stem(k) & _stem(prop) for k in keys)))

    def ledger(self) -> str:
        parts = []
        many = len(self.items) > 1
        for number, item in enumerate(self.items, 1):
            live = [m for m in item.musts + item.told if m not in item.dropped]
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
            scope = 'for each item' if brief.get('budget_scope') == 'each' else 'in total'
            parts.append(f"budget: at most {self.budget} {brief.get('budget_unit') or ''} {scope}".replace('  ', ' '))
        pick = brief.get('pick')
        if pick == 'cheapest':
            parts.append('pick: the cheapest listing that fits (lowest price)')
        elif pick in ('most', 'least'):
            parts.append(f"pick: the {pick} {brief.get('pick_measure') or ''} among fitting listings".replace('  ', ' '))
        else:
            parts.append('pick: any listing that fits everything')
        if brief.get('in_stock_only', True) is not False:
            parts.append('in stock only')
        for wish in _items(brief.get('wishes')):
            parts.append(f'wish (tie-breaker only): {wish}')
        for rule in _items(brief.get('rules')):
            parts.append(f'rule from the request: {rule}')
        if brief.get('quiet_decline') is True:
            parts.append('if nothing fits: end without naming or suggesting any listing')
        for entry in self.process:
            when = {'choose': 'before choosing', 'order': 'at the order (Kyo does it when you check out)', 'end': 'at an ending without a purchase (Kyo does it then)'}[entry['when']]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        facts = self.facts_tool()
        if facts and (self.facts_from.get(facts) is True or self.roles.get('offer')):
            bare = sorted((t for t in self.page_tools() | {self.roles.get('offer') or ''} if t and self.facts_from.get(t) is False))
            pages = sorted((t for t in self.page_tools() if t != facts))
            parts.append(f'listing attributes come from {facts}' + (f"; {', '.join(bare)} show none" if bare else '') + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ''))
        return '\n'.join(('- ' + p for p in parts))

    def status(self) -> str:
        left = self.left()
        text = f'[Kyo] step {self.step} of {self.max_steps} used, {left} left.'
        if left <= self.tail() + 1:
            text += ' FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that).'
        elif left <= 6 or (left <= len(self.items) + 5 and self.buys == 0):
            text += ' Few steps left: set the cart to the chosen lines and check out now.'
        elif self.buys == 0 and left <= 12:
            text += ' Steps are running low: settle on your picks, fill the cart and check out within the next few steps.'
        elif self.buys == 0 and (not self.cart_touched) and (self.step >= self.max_steps // 2):
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        if self.moves() >= WIND_DOWN and left > self.tail() + 1:
            text += ' THE WORK BUDGET IS NEARLY USED: set the cart to the chosen lines and check out within the next two turns.'
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values and (need in [c.need for c in self.chains if c.status == 'lost'] or not self.searched)]
        if lookups and self.buys == 0:
            text += '\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] ' + ' | '.join(lookups[:3])
        if self.leads:
            text += '\n[review candidates] ' + ', '.join(self.leads)
        plan = self.fit_plan() if self.buys == 0 else ''
        if plan:
            text += '\n' + plan
        if self.notes:
            text += '\n[changes so far] ' + ' | '.join(self.notes[-4:])
        text += '\n[ledger]\n' + self.ledger()
        return text

    def ident(self, node: Any, in_list: bool=False, depth: int=0) -> str | None:
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if self.id_key and _id_like(node.get(self.id_key), True):
            return node[self.id_key].strip()
        nested = any((isinstance(v, dict) for v in node.values()))
        if not (in_list or depth == 0 or nested):
            return None
        found = _learn_id([node])
        if found and ('id' in str(found).lower() or node[found].strip() in self.rows):
            return node[found].strip()
        return None

    def learn_rows(self, node: list[Any]) -> bool:
        rows = [n for n in node if isinstance(n, dict) and len(n) >= 3]
        if not rows or len(rows) < len(node) / 2:
            return False
        if self.id_key and all((_id_like(r.get(self.id_key), True) for r in rows)):
            return True
        found = _learn_id(rows)
        if found:
            self.id_key = found
            return True
        return False

    def absorb(self, observation: Any) -> None:

        def walk(node: Any, depth: int, in_list: bool=False) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                ident = self.ident(node, in_list, depth)
                if ident:
                    if ident not in self.rows:
                        self.seen_order.append(ident)
                        self.rows[ident] = {}
                    old = self.rows[ident]
                    if any((old.get(k) is True and v is False for k, v in node.items())):
                        self.unavailable.add(ident)
                    elif any((old.get(k) is False and v is True for k, v in node.items())):
                        self.unavailable.discard(ident)
                    self.rows[ident].update(node)
                    if _fact_block(node):
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

    def read_text(self, observation: Any) -> Any:
        seen = self.text_reader.read(observation)
        self.id_key = self.id_key or self.text_reader.id_label
        return seen

    def labels(self) -> set[str]:
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

    def category_key(self) -> str | None:
        counts: dict[str, set[str]] = {}
        present: dict[str, int] = {}
        for row in self.rows.values():
            for key, value in row.items():
                if isinstance(value, str) and 0 < len(value) <= 40 and (not re.search('\\d{3}', value)):
                    counts.setdefault(key, set()).add(value)
                    present[key] = present.get(key, 0) + 1
        total = max(1, len(self.rows))
        known = self.labels()
        best = None
        for key, values in counts.items():
            if present[key] < 0.8 * total or len(values) > 40:
                continue
            listed = sum((1 for v in values if v.strip().lower() in known)) / len(values)
            mean = sum((len(v) for v in values)) / len(values)
            score = (-listed, len(values), -present[key], -mean)
            if best is None or score < best[0]:
                best = (score, key)
        return best[1] if best else None

    def key_table(self) -> list[tuple[str, str, int, list[str]]]:
        cat_key = self.category_key()
        groups: dict[str, dict[str, Any]] = {}
        for ident in self.seen_order:
            record = self.full.get(ident)
            if not record:
                continue
            block = _fact_block(record) or {}
            label = str(record.get(cat_key) or 'listings') if cat_key else 'listings'
            group = groups.setdefault(label, {'n': 0, 'keys': {}})
            group['n'] += 1
            done: set[str] = set()
            for key, value in block.items():
                name = _words(key)
                if not name or name in done:
                    continue
                done.add(name)
                slot = group['keys'].setdefault(name, {'n': 0, 'values': [], 'per': {}})
                slot['n'] += 1
                text = str(value).strip().lower()
                slot['per'][ident] = text
                if text not in slot['values']:
                    slot['values'].append(text)
        wanted = [(_words(item.shelf or ''), _words(item.name), {w for w in _stem(item.shelf or '') | _stem(item.name) if len(w) >= 4}) for item in self.items]
        matched = {label for label in groups if any((_words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long for shelf, name, long in wanted)) or self.owner_of(label)}
        if matched:
            groups = {label: group for label, group in groups.items() if label in matched}
        table = []
        for label, group in sorted(groups.items(), key=lambda kv: -kv[1]['n']):
            kept: list[tuple[str, dict[str, Any]]] = []
            for name, slot in sorted(group['keys'].items(), key=lambda kv: (-kv[1]['n'], len(kv[0]))):
                values = list(slot['per'].values())
                if not values:
                    continue
                mean = sum((len(v) for v in values)) / len(values)
                if mean > 40 or sum((1 for v in values if ' / ' in v)) * 2 >= len(values):
                    continue
                recur = sum((len(set(re.findall('[a-z]{3,}', v)) & set(re.findall('[a-z]{3,}', ' '.join((o for o in values if o is not v))))) > 0 for v in values))
                if len(values) >= 4 and len(set(values)) == len(values) and (mean > 12) and (recur * 2 < len(values)):
                    continue
                twin = next((spec for other, spec in kept if self.twins(other, spec, name, slot)), None)
                if twin is not None:
                    for ident, text in slot['per'].items():
                        twin['per'].setdefault(ident, text)
                        if text not in twin['values']:
                            twin['values'].append(text)
                    twin['n'] = len(twin['per'])
                    continue
                kept.append((name, slot))
            kept.sort(key=lambda kv: -kv[1]['n'])
            for name, slot in kept[:14]:
                table.append((label, name, slot['n'], [v[:40] for v in slot['values'][:4]]))
        return table

    def not_stated(self, label: str, name: str, values: list[str]) -> str:
        number = self.owner_of(label)
        if not number:
            return ''
        for must in self.items[number - 1].musts:
            prop, _, value = must.partition(':')
            if re.search('\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', value, re.I):
                continue
            words = set(re.findall('[a-z0-9]{3,}', value.lower()))
            shown = set(re.findall('[a-z0-9]{3,}', ' '.join(values).lower()))
            if prop.strip() and _words(prop) != name and _stem(prop) & _stem(name) and words and (not words <= shown):
                return f' [not the stated {prop.strip()}: another property]'
        return ''

    @staticmethod
    def twins(name: str, spec: dict[str, Any], other: str, slot: dict[str, Any]) -> bool:
        a, b = (set(name.split()), set(other.split()))
        if not a or not b or (not a & b):
            return False

        def agree(x: str, y: str) -> bool:
            wx, wy = (re.findall('[a-z0-9]+', x.lower()), re.findall('[a-z0-9]+', y.lower()))
            short, long = (wx, wy) if len(wx) <= len(wy) else (wy, wx)
            return x == y or (bool(short) and set(short) <= set(long) and (len(long) <= 2 * len(short) + 2))
        both = [i for i in spec['per'] if i in slot['per'] and spec['per'][i].strip() and slot['per'][i].strip()]
        same = sum((1 for i in both if agree(spec['per'][i], slot['per'][i])))
        if not (a <= b or b <= a):
            return len(both) >= 2 and same == len(both)
        if both:
            return same * 2 > len(both)
        return bool((set(spec['values']) & set(slot['values'])) - {''})

    def item_groups(self) -> list[list[str]]:
        cat_key = self.category_key()
        groups: dict[str, list[str]] = {}
        for ident in self.seen_order:
            label = str(self.rows[ident].get(cat_key) or '') if cat_key else ''
            groups.setdefault(label, []).append(ident)
        wanted = [(_words(item.shelf or ''), _words(item.name), {w for w in _stem(item.shelf or '') | _stem(item.name) if len(w) >= 4}) for item in self.items]
        matched = [ids for label, ids in groups.items() if any((_words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long for shelf, name, long in wanted)) or self.owner_of(label)]
        pool = matched or list(groups.values())
        return sorted(pool, key=lambda ids: -len(ids))[:max(1, len(self.items))]

    def single_target(self, tool: str | None) -> str | None:
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        target = next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not tool or not target or any((k != target for k in schema.get('required') or [])):
            return None
        return target

    def price_of(self, ident: str) -> float | None:
        value = (self.rows.get(ident) or {}).get(self.price_key) if self.price_key else None
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and (not isinstance(v, bool))), None)
        if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', value):
            value = float(value)
        return float(value) if isinstance(value, (int, float)) and (not isinstance(value, bool)) else None

    def buyable(self, ident: str) -> bool:
        if self.sold_out(ident):
            return False
        value = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        return not (value is False or (isinstance(value, (int, float)) and (not isinstance(value, bool)) and (value <= 0)))

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

    def note_view(self, name: str, observation: Any) -> None:
        for ident, record in self.records_in(observation):
            keys = set(record)
            self.tool_keys.setdefault(name, set()).update(keys)
            version, have = self.view_keys.get(ident, (-1, set()))
            self.view_keys[ident] = (self.version, (have if version == self.version else set()) | keys)

    def shown_already(self, name: str, ident: str) -> bool:
        version, have = self.view_keys.get(ident, (-1, set()))
        keys = self.tool_keys.get(name)
        return bool(keys) and version == self.version and (keys <= have) and (ident not in self.gone)

    def read_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        if name in self.shapes:
            return self.shapes[name]
        if not isinstance(observation, dict):
            return None
        lists = [k for k, v in observation.items() if isinstance(v, list) and len(v) >= 2 and all((isinstance(r, dict) for r in v))]
        if not lists:
            return None
        self.shapes[name] = None
        sample = {k: v[:3] if k in lists else v for k, v in observation.items()}
        prompt = SHAPE + '\n\nTOOL: ' + name + ' | parameters: ' + _dump(self.schemas.get(name) or {}) + '\n\nRESULT (first rows only):\n' + _dump(_clip(sample, 120))[:4000]
        got = self.ask_json(prompt, [LIGHT, HEAVY], 2500) or {}
        rows = got.get('rows') if got.get('rows') in lists else None
        if not rows:
            return None
        first = observation[rows][0]
        props = (self.schemas.get(name) or {}).get('properties') or {}

        def named(key: Any, where: dict[str, Any]) -> str | None:
            return key if isinstance(key, str) and key in where else None
        shape = {'rows': rows, 'count': named(got.get('count'), observation), 'next': named(got.get('next'), observation), 'next_arg': named(got.get('next_arg'), props), 'price': named(got.get('price'), first), 'stock': named(got.get('stock'), first)}
        if shape['count'] and (isinstance(observation[shape['count']], bool) or not isinstance(observation[shape['count']], int)):
            shape['count'] = None
        self.shapes[name] = shape
        self.price_key = self.price_key or shape['price']
        self.stock_key = self.stock_key or shape['stock']
        self.trace.append({'kind': 'shape', 'tool': name, 'shape': shape})
        return shape

    def paging(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        shape = self.read_shape(name, observation)
        if not shape or not shape['count'] or (not shape['next']) or (not shape['next_arg']) or (not isinstance(observation, dict)):
            return None
        rows, count, value = (observation.get(shape['rows']), observation.get(shape['count']), observation.get(shape['next']))
        if not isinstance(rows, list) or not rows or isinstance(count, bool) or (not isinstance(count, int)):
            return None
        if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))):
            return None
        return (count, len(rows), shape['next_arg'], value)

    def price_sorted(self, observation: Any) -> bool:
        prices = [self.price_of(i) for i, _ in self.records_in(observation)]
        return len(prices) >= 3 and None not in prices and all((a <= b for a, b in zip(prices, prices[1:])))

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        self.paged_obs = []
        pick = self.brief.get('pick')
        plan = self.paging(name, observation) if pick in ('cheapest', 'most', 'least') else None
        if not plan:
            return ''
        count, shown, param, value = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and (not isinstance(given, bool)) else bool(str(given or '').strip())
        if value is None or later or count <= shown:
            return ''
        if pick == 'cheapest' and self.price_sorted(observation):
            return ''
        pages = -(-(count - shown) // shown)
        if pages > 2 and self.pages_read + 2 <= AUTO_PAGES and (self.left() - self.tail() - 2 >= 12):
            pages = 2
        if pages > 2 or self.pages_read + pages > AUTO_PAGES or self.left() - self.tail() - pages < 12:
            return f"\n[Kyo] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with {param}={_dump(value)}."
        texts = []
        for _ in range(pages):
            faults: list[str] = []
            call = _shape({**args, param: value}, self.schemas.get(name), 'args', faults)
            if faults:
                break
            reply = self.act(name, call)
            if reply is None or self.finished:
                break
            self.pages_read += 1
            seen = reply.get('observation')
            self.observe(name, seen)
            self.paged_obs.append(seen)
            shown_text = self.show(seen)
            self.read_cache[name + _dump(call)] = (self.version, self.step, shown_text)
            texts.append(f'{param}={_dump(value)}: {shown_text}')
            after = self.paging(name, seen)
            value = after[3] if after else None
            if value is None:
                break
        self.trace.append({'kind': 'pages', 'tool': name, 'read': len(texts)})
        rest = f"\n[Kyo] {count} listings matched and Kyo has read {shown * (len(texts) + 1)} rows of them; narrow the search with the requirement's value or page on to see the rest." if value is not None and texts else ''
        return '\n[Kyo read the next pages of this search itself, one step a page]\n' + '\n'.join(texts) + rest if texts else ''

    def cap_ignored(self, name: str, args: dict[str, Any], observation: Any) -> str:
        props = (self.schemas.get(name) or {}).get('properties') or {}

        def priced(key: str) -> bool:
            spec = props.get(key) if isinstance(props.get(key), dict) else {}
            return bool(re.search('price|cost|amount|budget|spend|money', key + ' ' + str(spec.get('description') or ''), re.I))
        caps = [k for k, v in args.items() if k in props and isinstance(v, (int, float)) and (not isinstance(v, bool)) and re.search('max|most|under|below|limit|cap|upto|up_to', k, re.I) and priced(k)]
        if not caps or not self.price_key:
            return ''
        caps.sort(key=lambda k: args[k])
        cap = args[caps[0]]
        over = [i for i, _ in self.records_in(observation) if (self.price_of(i) or 0) > cap]
        if not over:
            return ''
        modes = {k: v for k, v in args.items() if isinstance(v, str) and isinstance((props.get(k) or {}).get('enum'), list)}
        others = {k: [o for o in props[k]['enum'] if o != v] for k, v in modes.items()}
        hint = '; list the category with ' + ', '.join((f'{k}={_dump(o[0])}' for k, o in others.items() if o)) + f' and the same {caps[0]} to see only cheaper listings' if any(others.values()) else ''
        return f'\n[Kyo] This search did not apply {caps[0]}={cap:g}: {len(over)} of its rows cost more{hint}.'

    def cheapest_here(self, observations: list[Any]) -> str:
        if self.brief.get('pick') != 'cheapest' or not self.price_key:
            return ''
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ''
        return '\n[Kyo] buyable rows here, cheapest first: ' + ', '.join((f'{i} ({p:g})' for p, i in sorted(rows)[:6]))

    def wider_hold(self) -> str:
        if self.wider_held or self.brief.get('pick') != 'cheapest':
            return ''
        self.wider_held = True
        if self.cheaper_sweep() or self.left() < 14 + 2 * len(self.items):
            return ''
        lines = [f'{i} at {self.price_of(i):g}' for i in self.cart_ids() if self.price_of(i) is not None]
        return f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest. Searches may read only part of a listing, so a requirement's value finds only some of the fitting listings, and rows come in no price order. Before checking out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line" + (f" ({'; '.join(lines)})" if lines else '') + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'

    def other_values(self, number: int) -> list[str]:
        """Values listings of this item show for properties the request leaves open, taken in turn from its most carried
        keys: searched together with the required values, they bring up rows the plain search does not reach."""
        table = self.key_table()
        keys = self.lane_keys(number, table)[:3]
        said = ' '.join(self.value_texts(number)).lower() + ' ' + self.items[number - 1].name.lower()
        lanes = []
        for key in keys:
            values = next((v for label, name, _, v in table if name == key and (len(self.items) == 1 or self.owner_of(label) == number)), [])
            lanes.append([v for v in values if v and len(v) <= 20 and v not in said and re.search('[a-z0-9]', v)][:2])
        return [lane[k] for k in range(2) for lane in lanes if k < len(lane)]

    def cheaper_sweep(self) -> bool:
        """For a cheapest pick, once the cart's fit holds: bring up rows priced below each cart line, for the reading and the
        check before checkout. Lists the line's category under its price where the store can; on a long task or a line held by a
        fit, then under the lowest listing that could take the line's place, down to the cheap end. Else searches the required
        values with other values listings show. Steps for reading, the cart changes and the checkout stay free. Returns whether
        it could look or has to wait."""
        cat_key = self.category_key()
        plan = self.value_template()
        mark = self.cart_mark() + f'#{self.version}'
        room = self.left() - self.tail() - len(self.items) - 3
        if mark in self.swept or room < 1 or self.moves() >= SEARCH_BY or (not cat_key):
            return False
        index = self.fit_index()
        if self.fit_holds < 2 and self.fit_clashes(index)[0]:
            return True
        self.swept.add(mark)
        lines = self.cart_ids()
        start = self.step
        limit = self.step + min(SWEEP_STEPS * max(1, len(lines)), room, max(SWEEP_STEPS, self.max_steps // 3) - self.sweep_spent)
        ran = False
        cheaper: list[str] = []
        for place, line in enumerate(lines):
            price = self.price_of(line)
            number = self.line_item(line)
            if price is None or not number or self.step >= limit:
                continue
            until = self.step + max(2, (limit - self.step) // (len(lines) - place))
            walk = self.max_steps > WALK_STEPS or bool(self.fit_shared(line, index))
            cap: float | None = price
            found: list[str] | None = None
            while cap is not None and self.step < until:
                found = self.list_category(number, (int(cap) - 1 if float(cap).is_integer() else int(cap)) if cap > 1 else cap * 0.9, 1 if until - self.step >= 2 else 0)
                if found is None:
                    break
                ran = True
                rows = self.fit_pool(line, [i for i in self.kind_rows(number, found, cap) if i != line])
                cheaper += [i for i in rows if i not in self.full and i not in cheaper]
                low = min((self.price_of(i) for i in rows if self.price_of(i) is not None), default=None)
                cap = low if walk and low is not None and low < cap else None
            if found is None and plan:
                base = self.value_texts(number)[:2]
                for other in self.other_values(number)[:3]:
                    if self.step >= limit:
                        break
                    found = (found or []) + self.search_words(plan[0], plan[1], plan[2], ' '.join(base + [other, self.items[number - 1].name]))
                    ran = True
                cheaper += [i for i in self.kind_rows(number, found or [], price) if i != line and i not in self.full]
        self.sweep_spent += self.step - start
        self.trace.append({'kind': 'sweep', 'ran': ran, 'cheaper': cheaper[:6]})
        return ran

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return []
        texts = []
        for ident in [i for i in ids if i in self.rows and i not in self.full and (not self.sold_out(i))]:
            if self.left() <= keep or self.finished or self.facts_from.get(tool) is False or self.moves() >= STORE_STOP:
                break
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
            if faults:
                break
            reply = self.act(tool, args)
            if reply is None:
                break
            seen = reply.get('observation')
            self.observe(tool, seen)
            self.note_view(tool, seen)
            if isinstance(seen, dict) and self.ident(seen):
                self.facts_from[tool] = self.facts_from.get(tool, False) or bool(_fact_block(seen))
            texts.append(f'{ident}: ' + self.show(seen)[:1500])
        if texts:
            self.trace.append({'kind': 'fill_facts', 'n': len(texts)})
        return texts

    def show_links(self, ids: list[str], keep: int) -> None:
        """Run the part code a cart line sends the reader to through the search, once per code, and open
        the page it finds when steps allow; the result feeds the ledger, not the model. Multi-line carts only."""
        plan = self.value_template()
        if not plan or (len(self.items) < 2 and len(ids) < 2):
            return
        tool, base, field = plan
        for ident in ids:
            marks = self.fit_index()['marks']
            code = (marks.get(ident) or {}).get('more')
            if not code or code in self.link_tried or any((mark.get('own') == code for mark in marks.values())):
                continue
            if self.left() <= keep or self.finished:
                return
            self.link_tried.add(code)
            faults: list[str] = []
            args = _shape({**base, field: code}, self.schemas.get(tool), 'args', faults)
            if faults:
                return
            reply = self.act(tool, args)
            if reply is None:
                return
            self.observe(tool, reply.get('observation'))
            self.trace.append({'kind': 'show_link', 'line': ident, 'code': code})
            marks = self.fit_index()['marks']
            self.fill_facts([i for i, mark in marks.items() if mark.get('own') == code], max(keep, self.tail() + 6))

    def cheaper_rows(self) -> list[str]:
        if self.brief.get('pick') != 'cheapest':
            return []
        mark = self.cart_mark() + f'#{self.version}'
        cat_key = self.category_key()
        if mark in self.cheaper_marks or not cat_key:
            return []
        self.cheaper_marks.add(mark)
        cart = self.cart_ids()
        self.cheaper_sweep()
        picks: list[str] = []
        for line in cart:
            price = self.price_of(line)
            kind = str((self.rows.get(line) or {}).get(cat_key) or '').strip().lower()
            if price is None or not kind:
                continue
            pool = [i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i) and (str((self.rows.get(i) or {}).get(cat_key) or '').strip().lower() == kind) and ((self.price_of(i) if self.price_of(i) is not None else float('inf')) < price)]
            pool.sort(key=lambda i: self.price_of(i) or 0.0)
            pool = self.fit_pool(line, pool)
            picks += [i for i in pool[:CHEAPER_OPENS if len(cart) == 1 else 2] if i not in picks]
        return self.fill_facts(picks[:min(6, CHEAPER_OPENS * max(1, len(cart)))], self.tail() + 2)

    def observe(self, name: str, observation: Any) -> str:
        self.absorb(observation)
        lines = self.loose_lines(observation)
        if not lines:
            return ''
        meaning = self.take_note(lines)
        if meaning:
            self.queued.append(f'(seen on {name}) {meaning}')
        return meaning

    def widen(self) -> None:
        tool = self.roles.get('compare')
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in _kinds(v)), None)
        if not tool or not target or any((k != target for k in schema.get('required') or [])) or self.compare_bare():
            self.widen_singly()
            return
        if self.left() <= 12:
            return
        cap = props[target].get('maxItems') if isinstance(props[target].get('maxItems'), int) else 4
        ranked = self.item_groups()
        share = max(1, cap // max(1, len(ranked)))
        picks: list[str] = []
        for ids in ranked:
            picks += [i for i in ids if i not in self.full][:share]
        picks = picks[:cap]
        if len(picks) < 2:
            return
        faults: list[str] = []
        args = _shape({target: picks}, schema, 'args', faults)
        if faults:
            return
        before = len(self.full)
        reply = self.act(tool, args)
        if reply is not None:
            seen = reply.get('observation')
            self.observe(tool, seen)
            self.note_view(tool, seen)
            rows = self.records_in(seen)
            if rows:
                self.facts_from[tool] = self.facts_from.get(tool, False) or any((_fact_block(r) for _, r in rows))
            self.trace.append({'kind': 'widen', 'ids': picks, 'opened': len(self.full) - before})
            if rows and self.facts_from.get(tool) is False:
                self.widen_singly()
                return
            if len(self.full) == before and len(picks) > 2 and (self.left() > 12):
                faults = []
                args = _shape({target: picks[:2]}, schema, 'args', faults)
                if not faults:
                    reply = self.act(tool, args)
                    if reply is not None:
                        self.observe(tool, reply.get('observation'))

    def widen_singly(self) -> None:
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return
        wanted = {w for _, need in self.needs_now() for w in re.findall('[a-z0-9]{3,}', need.split(':', 1)[-1].lower())}
        each = WIDEN_EACH if len(self.items) <= 2 else 4
        lanes: list[list[str]] = []
        heads: list[int] = []
        for ids in self.item_groups():
            opened = sum((1 for i in ids if i in self.full))
            pool = [i for i in ids if i not in self.full and self.buyable(i)]
            if self.brief.get('pick') == 'cheapest':
                pool.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float('inf'))
            pool.sort(key=lambda i: -len(wanted & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower()))))
            lanes.append(pool[:max(0, each - opened)])
            heads.append(max(0, 4 - opened))
        first = [lane[k] for k in range(4) for lane, head in zip(lanes, heads) if k < min(head, len(lane))]
        more = [lane[k] for k in range(each) for lane, head in zip(lanes, heads) if head <= k < len(lane)]
        cap = max(5, 2 * len(lanes) + 2)
        read = self.fill_facts(first[:cap][:8], 22)
        self.fill_facts((first[cap:] + more)[:max(0, 10 - len(read))], 30)

    def ready_to_ask(self) -> bool:
        if self.asked or not self.roles.get('ask') or self.msgs_free() <= 0:
            return False
        if self.left() <= 8 or self.moves() > ASK_BY:
            return False
        opened = len(self.full)
        if opened >= (2 if len(self.items) >= 2 and self.max_steps <= SHORT_RUN else 3) * len(self.items) + 1:
            return True
        return self.step - self.walk_steps >= max(8, self.max_steps // 4)

    def message_args(self, text: str) -> dict[str, Any] | None:
        name = self.roles.get('ask')
        schema = self.schemas.get(name or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
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
        args = _shape(args, schema, 'args', faults)
        return None if faults else args

    def interview(self) -> str:
        self.asked = True
        if self.msgs_free() <= 0:
            return ''
        self.widen()
        count = self.msg_limit - self.msg_used
        self.lanes = self.plan_lanes(max(count, LIST_SIZE) if self.listing else count)
        self.lane_found = {n: 0 for n in self.lanes}
        self.lane_spent = {n: 0 for n in self.lanes}
        self.lane_used = {n: set() for n in self.lanes}
        self.follow_up = []
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        while self.msgs_free() > 0 and (not self.finished) and (self.left() > 6) and (self.moves() < QUESTION_BY):
            batch = self.ask_next(told)
            if not batch:
                break
            sent += batch
        found_items = [n for n in self.lane_found if self.lane_found[n] > 0]
        searched = self.value_search(found_items) if found_items else ''
        if told:
            self.version += 1
        lines = [f'Q: {q}\nA: {a}' for _, q, a in sent]
        summary = "Kyo's interviewer asked the shopper (each question used one step):\n" + '\n'.join(lines)
        summary += '\nNew firm requirements from these answers: ' + ('; '.join(told) if told else 'none')
        summary += '. Apply them to every pick (search again with the new values where needed).'
        pointed = [need for _, need in self.refs if any((need in t for t in told))]
        if pointed:
            summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.'
        if searched:
            summary += '\n' + searched
        if self.queued:
            summary += '\n[notes seen meanwhile] ' + ' | '.join(self.queued)
            self.queued = []
        return summary

    def late_question(self) -> str:
        """Before a checkout, ask the next planned question with a message the order itself does not need.

        The interview leaves out every message the request keeps for a process step, whether the step comes with the
        order or with an ending without a purchase. When Kyo is about to buy, the messages kept for the ending are
        free. Returns the summary for the loop model when the answer adds a requirement, else ''."""
        # Race 13 allowed two messages: the base asked one question in 28 of 180 episodes and left a hidden need
        # undisclosed in 12 of them (no episode with an undisclosed need passed).
        if self.late_asked or not self.roles.get('ask') or self.msg_used >= self.msg_limit:
            return ''
        if any((p['when'] == 'order' and p['messages'] > 0 for p in self.process)):
            return ''
        if self.left() < LATE_STEPS or self.moves() > LATE_BY or not self.judges():
            return ''
        self.late_asked = True
        if not self.lanes:
            self.lanes = self.plan_lanes(LIST_SIZE if self.listing else 1)
            self.lane_found = {n: 0 for n in self.lanes}
            self.lane_spent = {n: 0 for n in self.lanes}
            self.lane_used = {n: set() for n in self.lanes}
        told: list[str] = []
        budget = self.budget
        batch = self.ask_next(told, LATE_STEPS - 2)
        if not batch:
            self.trace.append({'kind': 'late_ask', 'skipped': 'no property left to ask or no message sent'})
            return ''
        if self.budget != budget:
            told.append(f'budget now {self.budget}')
        number, question, answer = batch[0]
        self.trace.append({'kind': 'late_ask', 'q': question, 'a': answer[:200], 'told': told})
        if not told:
            return ''
        self.version += 1
        summary = f'Kyo asked the shopper one more question before checking out (one step):\nQ: {question}\nA: {answer}\nNew firm requirements from this answer: ' + '; '.join(told) + '. The cart was not checked out: check each line against them (search again with the new values where needed), fix the cart, then check out.'
        pointed = [need for _, need in self.refs if any((need in t for t in told))]
        if pointed:
            summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value.'
        searched = self.value_search([number])
        return summary + ('\n' + searched if searched else '')

    def search_template(self) -> tuple[str, dict[str, Any], str] | None:
        tool = self.roles.get('search')
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        named = [k for k in texts if re.search(QUERY_WORDS, _snake(k), re.I)]
        for args in reversed(self.searched):
            fields = [k for k in named or texts if isinstance(args.get(k), str) and re.search('[a-z]', args[k], re.I)]
            if not named:
                labels = {str(r.get(self.category_key()) or '').strip().lower() for r in self.rows.values()} - {''} if self.category_key() else set()
                fields = [k for k in fields if len({str(a.get(k)) for a in self.searched if a.get(k)}) >= 2 and any((len(str(a.get(k)).split()) >= 2 for a in self.searched)) and (not any((str(a.get(k) or '').strip().lower() in labels for a in self.searched)))]
            if fields and tool:
                field = fields[0]
                keep = {k: v for k, v in args.items() if k != field and (isinstance((props.get(k) or {}).get('enum'), list) or k in (schema.get('required') or []))}
                cat_key = self.category_key()
                shelves = {_words(item.shelf or '') for item in self.items} | {_words(item.name) for item in self.items}
                shelves |= {_words(str(row.get(cat_key) or '')) for row in self.rows.values()} if cat_key else set()
                keep = {k: v for k, v in keep.items() if k in (schema.get('required') or []) or not isinstance(v, str) or (_words(v) not in shelves and (not any(({w for w in _stem(v) if len(w) >= 4} & {w for w in _stem(s) if len(w) >= 4} for s in shelves if s))))}
                worded = [a for a in self.searched if isinstance(a.get(field), str) and re.search('[a-z]', a[field], re.I)]
                for k in list(keep):
                    if isinstance((props.get(k) or {}).get('enum'), list):
                        reads = [o for o in props[k]['enum'] if isinstance(o, str) and re.search(QUERY_WORDS, o, re.I)]
                        if len(reads) == 1:
                            keep[k] = reads[0]
                            continue
                        used = [_dump(a[k]) for a in worded if k in a]
                        if used:
                            best = max(used, key=used.count)
                            keep[k] = next((a[k] for a in worded if k in a and _dump(a[k]) == best))
                return (tool, keep, field)
        return None

    def value_words(self, number: int) -> list[tuple[str, bool]]:
        """Search words for each requirement value of the item a words search can match, each with whether an answer
        gave it: the value itself, or for an answer's lower numeric limit with a unit its number and
        unit, since such a hidden limit often sits on the value of a listing that fits."""
        out = [(c.wanted(), True) for c in self.live_chains() if c.number == number and c.wanted()]
        told = self.items[number - 1].told
        for n, need in self.needs_now():
            if n != number or need in [m for _, m in self.refs] or ':' not in need:
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

    def value_texts(self, number: int) -> list[str]:
        return [value for value, _ in self.value_words(number)]

    def value_template(self) -> tuple[str, dict[str, Any], str] | None:
        """The search a value search copies: search_template(), or, when every search so far listed a category without
        words, the tool's one query field with its one mode argument set to read it (the option naming words or a query,
        else the other option of a required two-option mode). None when the schema leaves a doubt."""
        plan = self.search_template()
        tool = self.roles.get('search')
        if plan or not tool or not self.searched:
            return plan
        schema = self.schemas.get(tool) or {}
        props = schema.get('properties') or {}
        required = schema.get('required') or []
        named = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and 'enum' not in v and re.search(QUERY_WORDS, _snake(k), re.I)]
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
        return (tool, base, named[0])

    def value_search(self, numbers: list[int]) -> str:
        plan = self.value_template()
        if not plan or self.left() < 22 or self.moves() >= SEARCH_BY:
            return ''
        tool, base, field = plan
        seen: list[str] = []
        budget = 8
        for number in numbers:
            pairs = self.value_words(number)
            item = self.items[number - 1]
            values = [v for v, _ in pairs]
            fresh = [v for v, told in pairs if told]
            if not fresh or budget <= 2:
                continue
            lead = [v for v in fresh if len(v.split()) == 1 or re.fullmatch(AMOUNT, v, re.I)]
            values = lead + [v for v in values if v not in lead]
            stated = [i for i, v in enumerate(values) if v not in fresh]
            drop = stated[-1] if stated else 0
            tries = [values, values[:drop] + values[drop + 1:]] if len(values) > 1 else [[values[0], item.name], values] if lead else [values]
            for words in tries[:2]:
                if not words:
                    continue
                faults: list[str] = []
                args = _shape({**base, field: ' '.join(words)}, self.schemas.get(tool), 'args', faults)
                if faults or self.finished:
                    break
                reply = self.act(tool, args)
                budget -= 1
                if reply is None:
                    break
                observation = reply.get('observation')
                self.observe(tool, observation)
                cat_key = self.category_key()
                found = [i for i, _ in self.records_in(observation)]
                cap = self.budget if isinstance(self.budget, (int, float)) else None
                prices = [self.price_of(i) for i in found]
                if cap is not None and prices and all((p is not None and p > 10 * cap for p in prices)):
                    cap = None
                rows = [i for i in found if self.buyable(i) and (not cat_key or self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))) and (cap is None or (self.price_of(i) or 0) <= cap)]
                home = self.home_label(number)
                if cat_key and home:
                    rows.sort(key=lambda i: _words(str((self.rows.get(i) or {}).get(cat_key) or '')) != home)
                seen.append(f'{_dump(args.get(field))}: ' + (', '.join(rows[:6]) or 'no row of that kind'))
                if rows:
                    opened = self.fill_facts([i for i in rows if i not in self.full][:min(3, budget)], self.tail() + 16)
                    budget -= len(opened)
                    if opened:
                        seen.append('opened: ' + ' | '.join(opened)[:2400])
                    break
        if not seen:
            return ''
        self.trace.append({'kind': 'value_search', 'seen': [s[:200] for s in seen]})
        return "[Kyo searched each such item's requirement values together and read the first fitting-kind rows] " + '\n'.join(seen)

    def home_label(self, number: int) -> str:
        """The category label (in _words form) of the item's own listings: the request's shelf name when a listing
        seen carries it, else the label most of the item's opened listings (or, with none opened, its search rows)
        carry; '' when none is known."""
        cat_key = self.category_key()
        if not cat_key:
            return ''
        item = self.items[number - 1]
        for pool in (self.full, self.rows):
            counts: dict[str, int] = {}
            for record in pool.values():
                label = str(record.get(cat_key) or '')
                if label and self.owns(item, label):
                    counts[_words(label)] = counts.get(_words(label), 0) + 1
            for name in (item.shelf, item.name):
                if _words(name) in counts:
                    return _words(name)
            if counts:
                return max(counts, key=lambda label: counts[label])
        return ''

    def next_ask(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> tuple[int, list[str]]:
        first = {n: i for i, n in enumerate(self.plan_order)}
        for number in sorted(lanes, key=lambda n: (found.get(n, 0) > 0, spent.get(n, 0), first.get(n, len(first) + n))):
            left = [label for label in lanes[number] if _words(label) not in used[number] and (not self.told_about(number, label))]
            if left:
                return (number, self.list_of(number, left))
        return (0, [])

    def list_of(self, number: int, left: list[str]) -> list[str]:
        """The labels of the next message, in queue order: as many as `list_size` allows and at most one key that another
        requested shelf also carries; a second such key waits for a later message."""
        shared = self.shared_keys(number, self.key_table())
        size = self.list_size()
        chosen: list[str] = []
        for label in left:
            if len(chosen) >= size:
                break
            if _spelled(label) not in shared or not any((_spelled(other) in shared for other in chosen)):
                chosen.append(label)
        return chosen

    def list_size(self) -> int:
        """How many labels the next planned message may name. One after any one-at-a-time reply, for the rest of the
        episode. Before that it goes by the message's place in the episode, whatever the item: an order of three or more
        items starts at one label and widens by one; a smaller order starts at two, keeps two for its second message and
        then widens by one, or widens at once when two messages are all it has."""
        if not self.listing:
            return 1
        place = self.msg_used
        if len(self.items) >= 3:
            return min(LIST_SIZE, place + 1)
        if self.msg_used + self.msgs_free() <= TIGHT_MESSAGES:
            return min(LIST_SIZE, place + 2)
        return min(LIST_SIZE, max(2, place + 1))

    def told_about(self, number: int, label: str) -> bool:
        """Whether the request or an answer already gave a need on this property of the item. The request: the label is
        the need's property. An answer: the label's words are held by the told property's name or hold a told name of two
        or more words. Spelling and plural are folded; a one-word told name hides no longer label."""
        mine = _spelled(label)
        item = self.items[number - 1]
        return any((mine == _spelled(m.split(':', 1)[0]) for m in item.musts)) or any((_names(mine, _spelled(t.split(':', 1)[0])) for t in item.told))

    def ask_text(self, number: int, labels: list[str]) -> str:
        if len(labels) >= 2:
            return f"What {', '.join(labels[:-1])} or {labels[-1]} would you like for the {self.item_said(number)}?"
        return self.question(number, labels[0])

    def fit_ask(self, number: int, labels: list[str]) -> tuple[list[str], str]:
        """The longest leading part of `labels` whose question the message tool's schema takes whole (the schema shaper
        cuts a text past a length cap, so a list too long drops labels from the end instead), with that question;
        ([], '') when not even one label fits."""
        for size in range(len(labels), 0, -1):
            text = self.ask_text(number, labels[:size])
            args = self.message_args(text)
            if args and text in args.values():
                return (labels[:size], text)
        return ([], '')

    def ask_next(self, told: list[str], floor: int=6) -> list[tuple[int, str, str]]:
        """Send the next question (a pending follow-up first, else the next planned labels), read its answer into the
        items' needs (appending each new one to `told`) and settle the labels it named. Returns the sent batch, or []
        when nothing was sent."""
        for _ in range(LIST_SIZE):
            self.follow_up = [f for f in self.follow_up if not self.told_about(f[0], f[1][0])]
            if self.follow_up:
                number, labels = self.follow_up.pop(0)
            else:
                number, labels = self.next_ask(self.lanes, self.lane_found, self.lane_spent, self.lane_used)
            if not labels:
                return []
            fitted, text = self.fit_ask(number, labels)
            if fitted and all((e.get('q') != text for e in self.trace if e.get('kind') == 'ask')):
                break
            self.lane_used.setdefault(number, set()).add(_words(labels[0]))
            self.trace.append({'kind': 'ask_repeat' if fitted else 'ask_too_long', 'item': number, 'label': labels[0]})
        else:
            return []
        labels = fitted
        batch = self.send([(number, labels[0], text)], floor)
        if not batch:
            return []
        self.lane_spent[number] = self.lane_spent.get(number, 0) + 1
        had = [len(item.told) for item in self.items]
        heard = self.hear(batch, (LIGHT, HEAVY))
        if self.judges()[:1] == [HEAVY] and any((self.doubtful(entry) for entry in _items(heard.get('needs')))):
            heard = self.hear(batch, (HEAVY,)) or heard
        if len(labels) > 1:
            heard.pop('free', None)
        named = self.reply_item(batch[0][2])
        for entry in _items(heard.get('needs')) if named else []:
            if isinstance(entry, dict) and entry.get('points') is not True:
                entry['item'] = named
        self.take_heard(heard, told, number)
        new = [need for item, size in zip(self.items, had) for need in item.told[size:]]
        for n, (item, size) in enumerate(zip(self.items, had), 1):
            if len(item.told) > size:
                self.lane_found[n] = self.lane_found.get(n, 0) + 1
        crowded = any((_count(x) == 1 for x in _items(heard.get('crowded'))))
        self.settle(number, labels, new, crowded)
        return batch

    def reply_item(self, answer: str) -> int:
        """The item a reply names in the words the questions use, when it names exactly one item of an order of several,
        else 0. A reply may state a need of another item than the one asked about; a need that points at another listing
        may name a shelf only as that listing's, so it stays with the reader's item."""
        if len(self.items) < 2:
            return 0
        low = answer.lower()
        named = [n for n in range(1, len(self.items) + 1) if re.search('\\b' + re.escape(self.item_said(n).lower()) + '\\b', low)]
        return named[0] if len(named) == 1 else 0

    def settle(self, number: int, labels: list[str], new: list[str], crowded: bool) -> None:
        """Mark what an answer showed about the labels its question named.

        Any reply asking for one question at a time ends lists for the episode. One label: it is asked, whatever came
        back, and a refused label is not sent again in any form. A list: a disclosed need settles its label and every
        label before it, and the labels after it lead the item's next list. A refused list settles nothing: its first
        label goes alone next and the others stay where they are. A reply that discloses nothing settles the first label
        only: labels never named go ahead of the list's others, which follow in reverse order."""
        used = self.lane_used.setdefault(number, set())
        if crowded and self.listing:
            self.listing = False
            self.trace.append({'kind': 'lists_off', 'item': number})
        if len(labels) == 1:
            used.add(_words(labels[0]))
            return
        if crowded and not new:
            self.follow_up.insert(0, (number, labels[:1]))
            return
        hit = self.answered_label(labels, new) if new else -1
        ahead = labels[hit + 1:] if hit >= 0 else labels[:0:-1]
        used.update((_words(label) for label in labels[:max(hit, 0) + 1]))
        rest = [label for label in self.lanes.get(number) or [] if label not in labels]
        self.lanes[number] = ahead + rest if hit >= 0 else rest + ahead
        self.trace.append({'kind': 'list_settled', 'item': number, 'done': labels[:hit + 1], 'next': ahead[:4]})

    def answered_label(self, labels: list[str], new: list[str]) -> int:
        """Index of the earliest label a list's reply answered. Each new need maps to the label sharing most words with
        its property (colour and color alike), else to the label whose values on opened listings share most words with
        its value; -1 when no new need maps to a label (the reply is then settled like one that disclosed nothing)."""
        seen: dict[str, set[str]] = {}
        for _, key, _, values in self.key_table():
            seen.setdefault(_words(key), set()).update((w for v in values for w in _roots(v)))
        hits = []
        for need in new:
            prop, _, value = need.partition(':')
            scores = [len(_spelled(label) & _spelled(prop)) for label in labels]
            if max(scores, default=0) == 0:
                scores = [len(seen.get(_words(label), set()) & _roots(value)) for label in labels]
            if max(scores, default=0) > 0:
                hits.append(scores.index(max(scores)))
        return min(hits) if hits else -1

    def lane_keys(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        item = self.items[number - 1]
        settled = [_stem(m.split(':', 1)[0]) for m in item.musts]
        rows = [(label, key, carried, len(values)) for label, key, carried, values in table if self.owner_of(label) == number]
        measured = {key: sum((1 for v in values if re.match('^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$', v))) * 2 > len(values) for _, key, _, values in table}
        if len(self.items) == 1 and (not rows):
            rows = [(label, key, carried, len(values)) for label, key, carried, values in table]
        shared = self.shared_keys(number, table)
        rows.sort(key=lambda r: (-r[2], _spelled(r[1]) in shared, measured.get(r[1], False), -r[3]))
        keys = [key for _, key, _, _ in rows if not any((_names(prop, _stem(key)) for prop in settled)) and (not re.search(NOT_PROPERTY, key))]
        lists = self.list_keys(number, table)
        return [key for key in keys if key not in lists] + [key for key in keys if key in lists]

    def shared_keys(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> set[frozenset[str]]:
        """Keys of this item that a listing of another requested item also carries, in spelled form: a question on one
        may touch both lines of the order."""
        own = {_spelled(key) for label, key, _, _ in table if self.owner_of(label) == number}
        return own & {_spelled(key) for label, key, _, _ in table if self.owner_of(label) not in (0, number)}

    @staticmethod
    def many_in_one(values: list[str]) -> bool:
        """Whether a key's values list several things at once: in at least two listings, and in at least half of those
        seen, a value holds three or more parts between list separators."""
        lists = sum((1 for value in values if len(re.split('[,;|]', value)) >= 3))
        return lists >= 2 and lists * 2 >= len(values)

    def list_keys(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> set[str]:
        """Keys of this item whose values are such lists: they are asked last, one property hides in many."""
        rows = [row for row in table if self.owner_of(row[0]) == number] or (table if len(self.items) == 1 else [])
        return {key for _, key, _, values in rows if self.many_in_one(values)}

    def adopt_labels(self) -> None:
        cat_key = self.category_key()
        if len(self.items) < 2 or not cat_key:
            return
        labels = {str(record.get(cat_key) or '') for record in self.full.values()} - {''}
        strays = [label for label in labels if not self.owner_of(label)]
        orphans = [n for n in range(1, len(self.items) + 1) if not any((self.owner_of(label) == n and label not in self.adopted for label in labels))]
        if strays and len(orphans) == 1:
            self.adopted.update({label: orphans[0] for label in strays})
            self.trace.append({'kind': 'adopted', 'item': orphans[0], 'labels': strays})

    def owner_of(self, label: str) -> int:
        if not label:
            return 0
        if label in self.adopted:
            return self.adopted[label]
        words = _words(label)
        long = _roots(label)
        best, top = (0, 0.0)
        for number, item in enumerate(self.items, 1):
            if words == _words(item.shelf or ''):
                score = 3.0
            elif words == _words(item.name):
                score = 2.0
            else:
                mine = _roots(item.shelf or '') | _roots(item.name)
                score = len(long & mine) / max(1, len(long | mine))
            if score > top:
                best, top = (number, score)
        return best

    def owns(self, item: Item, label: str) -> bool:
        return bool(label) and item in self.items and (self.owner_of(label) == self.items.index(item) + 1)

    def records_of_item(self, number: int) -> int:
        cat_key = self.category_key()
        item = self.items[number - 1]
        return sum((1 for record in self.full.values() if cat_key and self.owns(item, str(record.get(cat_key) or '')) and _fact_block(record)))

    def plan_lanes(self, count: int) -> dict[int, list[str]]:
        self.adopt_labels()
        table = self.key_table()
        settled = []
        for number, item in enumerate(self.items, 1):
            settled.append(f'item {number}: {item.name} - settled by the request: ' + ('; '.join(item.musts) or 'nothing'))
        listing = []
        last = None
        for index, (label, name, carried, values) in enumerate(table, 1):
            if label != last:
                listing.append(f'category {label!r}:')
                last = label
            listing.append(f'  {index}. {name} (on {carried}): ' + ' | '.join(values) + self.not_stated(label, name, values) + (' [several features in one: ask last]' if self.many_in_one(values) else ''))
        prompt = PLAN.replace('COUNT', str(max(1, count))) + '\n\nREQUEST:\n' + self.request + '\n\nITEMS:\n' + '\n'.join(settled)
        first = [str(x) for x in _items(self.brief.get('ask_first')) if str(x).strip()]
        if first:
            prompt += '\n\nASK FIRST (the request asks to be asked about these):\n' + '\n'.join(first)
        prompt += '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(listing) or '(no full listing records yet)')
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 3500, deep=True) or {}
        lanes: dict[int, list[str]] = {n: [] for n in range(1, len(self.items) + 1)}
        hinted: dict[int, set[str]] = {n: set() for n in lanes}
        for pick in _items(plan.get('ask')):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, _count(pick.get('item') or 1) or 1), len(self.items))
            label = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(pick.get('label') or ''))[0].strip().rstrip('?.')
            if not label or len(label.split()) > 5:
                continue
            if all((_words(label) != _words(other) for other in lanes[number])):
                lanes[number].append(label)
            if str(pick.get('hint') or '').strip():
                hinted[number].add(_words(label))
                if number not in self.plan_order:
                    self.plan_order.append(number)
        for number in lanes:
            keys = self.lane_keys(number, table)
            read = max(1, self.records_of_item(number))
            own: dict[str, float] = {}
            for label, key, carried, _ in table:
                if self.owner_of(label) == number:
                    own[key] = max(own.get(key, 0.0), carried / read)

            def spelled(label: str) -> str:
                plain = lambda text: re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', _words(text)))
                heads = [k for k in own if len(k.split()) >= 2 and plain(k.split()[-1]) == plain(label)]
                return next((k for k in own if plain(k) == plain(label)), heads[0] if len(heads) == 1 else label)

            def share_of(label: str) -> float:
                if _words(label) in own:
                    return own[_words(label)]
                near = [s for k, s in own.items() if _names(_stem(label), _stem(k)) or _names(_stem(k), _stem(label))]
                return max(near) if near else 0.0
            hinted[number] = {_words(spelled(h)) for h in hinted[number]}
            lane = []
            for pick in lanes[number]:
                if all((_words(spelled(pick)) != _words(other) for other in lane)):
                    lane.append(spelled(pick))
            lists = self.list_keys(number, table)
            for key in [k for k in keys if own.get(k, 0.0) >= GUARD_SHARE and k not in lists][:GUARD_KEYS]:
                if any((_words(key) == _words(other) for other in lane[:3])):
                    continue
                weak = next((i for i, other in enumerate(lane[:3]) if _words(other) not in hinted[number] and share_of(other) + GUARD_GAP <= own.get(key, 0.0)), None)
                if weak is None and len(lane) >= 3:
                    continue
                placed = [other for other in lane if _words(other) != _words(key)]
                if weak is not None and len(self.items) > 1:
                    moved = placed[weak]
                    placed[weak] = key
                    placed.insert(min(3, len(placed)), moved)
                else:
                    placed.insert(min(len(placed), weak if weak is not None else len(placed)), key)
                if any((_words(x) in hinted[number] and all((_words(x) != _words(y) for y in placed[:3])) for x in lane[:3])):
                    continue
                lane = placed
            lanes[number] = lane
            for key in keys:
                if all((_words(key) != _words(other) for other in lanes[number])):
                    lanes[number].append(key)
            lanes[number] = [x for x in lanes[number] if not self.bare_label(x, own)]
            lanes[number] = [x for x in lanes[number] if _words(x) not in lists] + [x for x in lanes[number] if _words(x) in lists]
            if self.msgs_free() <= TIGHT_MESSAGES and read >= 3:
                sure = [x for x in lanes[number] if _words(x) in hinted[number] or share_of(x) >= 0.5]
                lanes[number] = sure + [x for x in lanes[number] if x not in sure]
        self.trace.append({'kind': 'plan', 'ask': _clip(plan.get('ask'), 120), 'lanes': {n: v[:8] for n, v in lanes.items()}})
        return lanes

    @staticmethod
    def bare_label(label: str, keys: dict[str, float]) -> bool:
        """Whether a label is a bare head word: one word that no read listing of the item has as a key, while several
        longer keys end in it. It names no property of the item, and the longer keys stand in the queue themselves."""
        word = _spelled(label)
        if len(word) != 1 or any((_spelled(key) == word for key in keys)):
            return False
        return any((len(key.split()) >= 2 and _spelled(key.split()[-1]) == word for key in keys))

    def item_said(self, number: int) -> str:
        """The item as a question names it: the store's own shelf label in lower case, exactly, else the item's name."""
        return self.shelf_said(number).lower() or self.items[number - 1].name

    def shelf_said(self, number: int) -> str:
        """The store's category label for naming an item: of the labels its rows show, the brief's own, else the one
        closest to the item's words, then the one most rows show (near twins of a shelf can outnumber it among search
        results); with no such row, the brief's label if a tool lists it. '' when nothing shows one, or when the label
        stands for two items of the order, which it could not tell apart."""
        item = self.items[number - 1]
        cat_key = self.category_key()
        counts: dict[str, int] = {}
        for row in self.rows.values() if cat_key else []:
            label = str(row.get(cat_key) or '')
            if label and self.owns(item, label):
                counts[label] = counts.get(label, 0) + 1
        mine = _roots(item.shelf) | _roots(item.name)
        best = max(counts, key=lambda x: (_words(x) == _words(item.shelf), len(_roots(x) & mine) / max(1, len(_roots(x) | mine)), counts[x]), default='')
        if not best and item.shelf.strip().lower() in self.labels():
            best = item.shelf.strip()
        return '' if sum((1 for other in self.items if _words(other.shelf) == _words(best))) >= 2 else best

    def question(self, number: int, label: str) -> str:
        return f'What {label} would you like for the {self.item_said(number)}?'

    def doubtful(self, entry: Any) -> bool:
        """Whether a stated answer deserves a second reading: it points at another listing, or its value has words that
        no opened listing shows for that property, which is how words around a value creep into it."""
        if not isinstance(entry, dict):
            return False
        prop, _, value = str(entry.get('need') or '').partition(':')
        words = set(re.findall('[a-z0-9]+', value.lower()))
        if entry.get('points') is True:
            return True
        if len(words) < 2:
            return False
        seen: set[str] = set()
        for record in self.full.values():
            for key, shown in (_fact_block(record) or {}).items():
                if _names(_stem(prop), _stem(key)) or _names(_stem(key), _stem(prop)):
                    seen |= set(re.findall('[a-z0-9]+', str(shown).lower()))
        return bool(words - seen)

    def send(self, questions: list[tuple[int, str, str]], floor: int=6) -> list[tuple[int, str, str]]:
        tool = self.roles['ask']
        sent: list[tuple[int, str, str]] = []
        for number, _, text in questions:
            if self.finished or self.left() <= floor or self.msg_used >= self.msg_limit:
                break
            args = self.message_args(text)
            if not args:
                break
            reply = self.act(tool, args)
            if reply is None:
                break
            self.msg_used += 1
            answer = reply.get('observation')
            words = self.spoken(answer)
            sent.append((number, text, words))
            self.done_log.append(f'step {self.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {self.cart_mark()})')
            self.done_steps.append(self.step)
            self.trace.append({'kind': 'ask', 'q': text, 'a': words[:300]})
            if isinstance(answer, dict) and answer.get('error'):
                break
        return sent

    def take_heard(self, heard: dict[str, Any], told: list[str], about: int=1) -> None:
        many = len(self.items) > 1
        for entry in _items(heard.get('needs')):
            if not isinstance(entry, dict) or not str(entry.get('need') or '').strip():
                continue
            number = min(max(1, _count(entry.get('item')) or about), len(self.items))
            need = str(entry['need']).strip()
            if need not in self.items[number - 1].told:
                self.items[number - 1].told.append(need)
                told.append(f'item {number}: {need}' if many else need)
            if entry.get('points') is True and (number, need) not in self.refs:
                self.refs.append((number, need))
        for entry in _items(heard.get('drop')):
            if isinstance(entry, dict) and str(entry.get('property') or '').strip():
                item = self.items[min(max(1, _count(entry.get('item')) or about), len(self.items)) - 1]
                gone = _words(str(entry['property']).split(':')[0])
                hit = [m for m in item.musts + item.told if m not in item.dropped and _words(m.split(':')[0]) == gone]
                item.dropped.extend(hit)
                told.extend((f'dropped {m}' for m in hit))
        for entry in _items(heard.get('free')):
            if isinstance(entry, dict) and entry.get('property'):
                number = min(max(1, _count(entry.get('item')) or about), len(self.items))
                self.items[number - 1].free.append(str(entry['property']))
        if isinstance(heard.get('budget'), (int, float)) and (not isinstance(heard.get('budget'), bool)) and (heard['budget'] > 0):
            self.budget = heard['budget']

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(HEAVY, LIGHT)) -> dict[str, Any]:
        prompt = HEAR + '\n\nITEMS:\n' + '\n'.join((f'item {n}: {it.name}' for n, it in enumerate(self.items, 1)))
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f'{i}. (about item {n}) Q: {q}\nA: {a}' for i, (n, q, a) in enumerate(sent, 1)))
        return self.ask_json(prompt, list(models), 2500) or {}

    def gate(self, stage: str) -> tuple[bool, str]:
        steps = [p for p in self.process if p['when'] == stage]
        if not steps or self.finished or (not self.gate_due(stage)):
            return (True, '')
        ending = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        tools = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_dump(t['function'].get('parameters') or {})}" for t in self.tools if t['function']['name'] not in ending]
        said: list[str] = []
        nudged = False
        for _ in range(GATE_ROUNDS):
            prompt = GATE.replace('STAGE', 'place the order' if stage == 'order' else 'end without a purchase') + '\n\nREQUEST:\n' + self.request + '\n\nSTEPS DUE NOW:\n' + '\n'.join((f"- {p['step']}" for p in steps)) + '\n\nLEDGER:\n' + self.ledger() + '\n\nCART NOW: ' + self.cart_text() + '\n\nRECORD (messages and step calls so far):\n' + ('\n'.join((f'{i}. {line}' for i, line in enumerate(self.done_log, 1))) or '(nothing yet)') + ('\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.notes[-4:]) if self.notes else '') + f"\n\nMESSAGES LEFT: {max(0, self.msg_limit - self.msg_used)}   STEPS LEFT: {self.left()} (the {('order' if stage == 'order' else 'ending')} itself takes one)" + '\n\nTOOLS:\n' + '\n'.join(tools)
            got = self.ask_json(prompt, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 3500)
            if got is None:
                break
            told: list[str] = []
            hold = ''
            if said:
                self.take_heard({'needs': got.get('needs')}, told)
                hold = str(got.get('hold') or '').strip()
            call = got.get('call') if isinstance(got.get('call'), dict) else None
            self.trace.append({'kind': 'gate', 'stage': stage, 'call': _clip(call, 200), 'steps': _clip(got.get('steps'), 120), 'hold': hold[:200], 'told': told})
            if told or hold:
                self.version += 1
                return (False, ('Before going on, Kyo did the steps the request asks for:\n' + '\n'.join(said) + '\n' if said else '') + ("The shopper's reply changes the task: " + '; '.join(told + ([hold] if hold else [])) + '. Fix the cart for it, then check out again.'))
            name = str((call or {}).get('tool') or '')
            shown = [e for e in _items(got.get('steps')) if isinstance(e, dict)]
            talk = stage == 'order' and any((p['messages'] > 0 for p in steps))
            lines = range(1, len(self.done_log) + 1)
            proof = {i for i in lines if not talk or (i <= len(self.done_steps) and self.done_steps[i - 1] > self.cart_at)}
            unproven = [e for e in shown if e.get('done') is not True or not any(((_count(n) or 0) in proof for n in _items(e.get('lines'))))]
            if not call and unproven and (not said) and (not nudged) and self.roles.get('ask') and (self.msg_used < self.msg_limit) and (self.left() >= 2):
                nudged = True
                self.trace.append({'kind': 'gate_unproven', 'steps': _clip(unproven, 120)})
                self.done_log.append(f'(Kyo: no line made since the cart last changed at step {self.cart_at} shows ' + '; '.join((str(e.get('step') or '') for e in unproven))[:200] + f' done, and {self.msg_limit - self.msg_used} message(s) and {self.left()} steps are left: give the call that does it)')
                self.done_steps.append(-1)
                continue
            if not call or name not in self.schemas or name in ending:
                if not call:
                    self.gate_ok.add((stage, self.cart_mark() + f'#{self.version}'))
                break
            if self.left() < 2 or (name == self.roles.get('ask') and self.msg_used >= self.msg_limit):
                break
            faults: list[str] = []
            args = _shape(call.get('args') if isinstance(call.get('args'), dict) else {}, self.schemas[name], 'args', faults)
            if faults:
                break
            if name == self.roles.get('ask'):
                text = next((v for v in args.values() if isinstance(v, str)), '')
                batch = self.send([(1, '', text)], 1)
                if not batch:
                    break
                self.gate_msgs += 1
                said.append(f'Q: {text}\nA: {batch[0][2]}')
                continue
            reply = self.act(name, args)
            if reply is None:
                break
            seen = reply.get('observation')
            self.absorb(seen)
            self.done_log.append(f'step {self.step}: {name} {_dump(args)} -> {self.show(seen)[:300]} (cart then: {self.cart_mark()})')
            self.done_steps.append(self.step)
            said.append(f'{name} {_dump(args)} -> {self.show(seen)[:400]}')
        return (True, 'Before going on, Kyo did the steps the request asks for:\n' + '\n'.join(said) if said else '')

    def spoken(self, observation: Any) -> str:
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
        return ' '.join(found)[:800] if found else _dump(observation)[:400]

    def loose_lines(self, observation: Any) -> list[str]:
        lines: list[str] = []
        if isinstance(observation, dict) and (not self.ident(observation)):
            for key, value in observation.items():
                if isinstance(value, str) and len(value.split()) >= 5:
                    lines.append(value)
                elif isinstance(value, list) and value and all((isinstance(v, str) for v in value)):
                    lines.extend((v for v in value if len(v.split()) >= 5))
                elif isinstance(value, dict) and (not self.ident(value, False, 1)):
                    for inner in value.values():
                        if isinstance(inner, str) and len(inner.split()) >= 6:
                            lines.append(inner)
        return [line for line in lines if line not in self.heard]

    def take_note(self, lines: list[str]) -> str:
        self.heard.update(lines)
        prompt = NOTE + '\n\nITEMS AND REQUIREMENTS NOW:\n' + self.ledger() + '\n\nTHE NOTE:\n' + '\n'.join(lines)
        got = self.ask_json(prompt, [LIGHT, SPARE], 2500) or {}
        gone = [str(g) for g in _items(got.get('gone')) if isinstance(g, (str, int))]
        self.unavailable.update((g for g in gone if g in self.rows))
        self.note_sold = bool(gone) or got.get('sold_out') is True
        self.note_named = any((g in self.rows for g in gone))
        if self.note_sold and len(lines) == 1:
            self.sold_lines.update(lines)
        if got.get('kind') not in (None, 'change'):
            return ''
        changed = []
        budget = got.get('budget')
        if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0) and (budget != self.budget):
            self.budget = budget
            changed.append(f'budget now {budget}')
        added: set[str] = set()
        drops = [e for e in _items(got.get('drop')) if isinstance(e, dict) and str(e.get('need') or '').strip()]
        adds = [e for e in _items(got.get('add')) if isinstance(e, dict) and str(e.get('need') or '').strip()]
        for entry in drops:
            item = self.items[min(max(1, _count(entry.get('item')) or 1), len(self.items)) - 1]
            target = str(entry['need']).strip()
            live = [m for m in item.musts + item.told if m not in item.dropped]
            hit = [m for m in live if m == target]
            if not hit:
                hit = [m for m in live if _words(m.split(':')[0]) == _words(target.split(':')[0])][:1]
            item.dropped.extend(hit or [target])
            changed.append(f'dropped {target}')
        for entry in adds:
            item = self.items[min(max(1, _count(entry.get('item')) or 1), len(self.items)) - 1]
            need = str(entry['need']).strip()
            if need in item.dropped:
                item.dropped.remove(need)
            if need not in item.musts + item.told:
                item.told.append(need)
            added.add(need)
            changed.append(f'added {need}')
        if gone:
            changed.append('unavailable: ' + ', '.join(gone))
        if got.get('price_moved') is True:
            changed.append('a price changed')
        summary = str(got.get('summary') or '').strip()
        if self.brief.get('pick') == 'cheapest' and (drops or gone or self.note_sold or any((c.startswith('budget now') for c in changed))):
            self.wider_held = False
            self.cheaper_marks = set()
        if changed or summary:
            self.version += 1
            self.recent = []
            self.notes.append(summary or '; '.join(changed))
        return (summary + (' (' + '; '.join(changed) + ')' if changed else '')).strip()

    def needs_now(self) -> list[tuple[int, str]]:
        out = []
        for number, item in enumerate(self.items, 1):
            for need in item.musts + item.told:
                if need not in item.dropped:
                    out.append((number, need))
        return out

    def evidence(self, observation: Any) -> str:
        ids: list[str] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                ident = self.ident(node, False, depth)
                if ident and _fact_block(node) and (ident not in ids):
                    ids.append(ident)
                    return
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1)
        walk(observation, 0)
        needs = self.needs_now()
        if not ids or not needs:
            return ''
        many = len(self.items) > 1
        lines = []
        for ident in ids[:12]:
            block = _fact_block(self.full.get(ident) or {}) or {}
            parts = []
            for number, need in needs[:8]:
                prop = need.split(':', 1)[0].strip()
                hits = [f'{k}={str(v)[:50]}' for k, v in block.items() if _names(_stem(prop), _stem(k))]
                tag = f'(item {number}) ' if many else ''
                if hits:
                    parts.append(f'{tag}{prop} -> ' + ', '.join(hits[:3]))
            if parts:
                lines.append(f'{ident}: ' + '; '.join(parts))
        if not lines:
            return ''
        return "[evidence: keys whose names contain a requirement's property; one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property] " + ' | '.join(lines)

    def cart_text(self) -> str:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            return _dump(_clip(self.cart_view, 300))[:3000]
        return "lines by Kyo's record of the cart calls (listing: units): " + _dump(self.cart_lines)

    def held_units(self, ident: str) -> int:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            stack: list[Any] = [self.cart_view]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    if ident in [v.strip() for v in node.values() if isinstance(v, str)]:
                        counts = [_count(v) for k, v in node.items() if not isinstance(v, str) and _count(v) is not None and re.search('\\b(?:qty|quantity|count|units?)\\b', _words(k))]
                        if len(counts) == 1:
                            return counts[0]
                    stack += [v for v in node.values() if isinstance(v, (dict, list))]
                elif isinstance(node, list):
                    stack += node
        return self.cart_lines.get(ident, 1)

    def cart_ids(self) -> list[str]:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            text = _dump(self.cart_view)
            return [i for i in self.rows if i in text]
        return list(self.cart_lines)

    def record_of(self, ident: str) -> str:
        record = self.full.get(ident) or self.rows.get(ident)
        return _dump(_clip(record, 400))[:3000] if record else '(no record seen)'

    def others(self, skip: list[str], limit: int=30) -> str:
        lines = []
        order = list(reversed(self.seen_order))
        if self.brief.get('pick') == 'cheapest':
            order.sort(key=lambda i: (i not in self.full, self.price_of(i) if self.price_of(i) is not None else float('inf')))
        for ident in order:
            if ident in skip or self.sold_out(ident):
                continue
            record = self.full.get(ident) or self.rows.get(ident)
            lines.append(_dump(_clip(record, 160))[:900])
            if len(lines) >= limit:
                break
        return '\n'.join(lines)

    def fit_index(self) -> dict[str, Any]:
        marks: dict[str, dict[str, Any]] = {}
        notes: dict[str, set[str]] = {}
        for ident in self.seen_order:
            lines: list[str] = []
            for record in (self.rows.get(ident) or {}, self.full.get(ident) or {}):
                lines += [v for v in record.values() if isinstance(v, str) and v not in lines]
                lines += [f'{k} {v}' for block in record.values() if isinstance(block, dict) for k, v in block.items() if isinstance(v, str)]
            marks[ident] = _fit_marks('\n'.join(lines))
            for subject, codes in marks[ident]['also']:
                notes.setdefault(subject, set()).update(codes)
        kin = {_code_family(_code_key(m['own'])) for m in marks.values() if re.match(PART_SERIES + '-', m['own'])}
        return {'marks': marks, 'notes': notes, 'owns': {m['own'] for m in marks.values()}, 'kin': kin}

    def fit_set(self, ident: str, index: dict[str, Any]) -> set[str]:
        mark = index['marks'].get(ident) or {}
        found = set(mark.get('works') or [])
        for key in [_code_key(mark.get('own') or '')] + sorted(found):
            found |= index['notes'].get(key, set())
        return found

    def fit_names(self, ident: str, index: dict[str, Any]) -> set[str]:
        """Every code a line fits: what it states, what other pages state about it, the series it takes or belongs to.
        A short code counts as a series only when listings carry that family before the hyphen of their own codes."""
        mark = index['marks'].get(ident) or {}
        names = self.fit_set(ident, index) | {_code_key(s) for s in mark.get('series') or []}
        names = {c for c in names if not _fit_short(c.upper()) or _code_family(c) in index['kin']}
        if re.match(PART_SERIES + '-', mark.get('own') or ''):
            names.add(_code_key(mark['own'].split('-')[0]))
        return names

    def item_ids(self, number: int) -> list[str]:
        cat_key = self.category_key()
        item = self.items[number - 1]
        return [i for i in self.seen_order if cat_key and self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))]

    def fit_groups(self) -> list[set[int]]:
        """The requested items the request ties together, each set holding every item joined by its fit rules."""
        groups: list[set[int]] = []
        for pair in self.bound:
            linked = set(pair)
            for group in [g for g in groups if g & linked]:
                groups.remove(group)
                linked |= group
            groups.append(linked)
        return groups

    def fit_mates(self, number: int, codes: list[str], index: dict[str, Any]) -> list[str]:
        """Buyable listings of a requested item, outside the cart, that fit one of the codes: cheapest first."""
        cart = self.cart_ids()
        rows = [i for i in self.item_ids(number) if i not in cart and self.buyable(i)]
        mates = [i for i in rows if any((_fit_holds(self.fit_names(i, index), (index['marks'].get(i) or {}).get('deny') or [], c) for c in codes))]
        return sorted(mates, key=lambda i: self.price_of(i) if self.price_of(i) is not None else float('inf'))[:4]

    def fit_seek(self, number: int, code: str, index: dict[str, Any]) -> bool:
        """One look for listings of a requested item that fit a code, once per item and code. A series its rows carry in
        their own codes sits in titles, so it goes in the search words; any other code sits in row text, so the shelf is listed."""
        key = f'{code}@{number}'
        tries = sum(('@' in k for k in self.link_tried))
        if key in self.link_tried or tries >= len(self.items) + 2 or self.left() <= self.tail() + 6 or self.moves() >= SEARCH_BY or self.finished:
            return False
        self.link_tried.add(key)
        plan = self.value_template()
        owns = [(index['marks'].get(i) or {}).get('own') or '' for i in self.item_ids(number)]
        titled = any((re.match(PART_SERIES + '-', own) and _code_family(_code_key(own)) == _code_family(code) for own in owns))
        found: list[str] | None = None
        if plan and titled:
            shelf = {self.browse['category']: self.shelf_of(number)} if self.browse and self.browse['tool'] == plan[0] else {}
            found = self.search_words(plan[0], {**plan[1], **shelf}, plan[2], code.upper() if shelf else f'{code.upper()} {self.items[number - 1].name}')
        if not found:
            found = self.list_category(number, None, 1)
        self.trace.append({'kind': 'fit_seek', 'item': number, 'code': code, 'rows': len(found or [])})
        return bool(found)

    def fit_open_end(self, ident: str, index: dict[str, Any]) -> bool:
        """A listing sending the reader to a page not seen has more codes than it shows, so nothing can be held against it."""
        more = (index['marks'].get(ident) or {}).get('more')
        return bool(more) and more not in index['owns']

    def fit_clashes(self, index: dict[str, Any], ids: list[str] | None=None) -> tuple[list[tuple[str, list[str]]], dict[str, set[str]]]:
        """What breaks a fit in the cart, or in the lines given: lines of items the request ties together must share a code, and
        in any cart, lines of one part series family must share a series. Returns (code, lines without it) and the codes of the
        lines judged."""
        marks = index['marks']
        names = {i: self.fit_names(i, index) for i in (self.cart_ids() if ids is None else ids) if i in marks and not self.fit_open_end(i, index)}
        deny = {i: marks[i]['deny'] for i in names}
        devices = [i for i in names if not marks[i]['own']]
        loose = {c for m in marks.values() for c in m['loose']}
        found: list[tuple[str, list[str]]] = []
        for group in self.fit_groups():
            lines = [i for i in names if self.line_item(i) in group]
            found += _fit_odd({i: names[i] for i in lines}, deny, [i for i in lines if i in devices], loose)
        kin = {_code_family(_code_key(marks[i]['own'])) for i in names if re.match(PART_SERIES + '-', marks[i]['own'])}
        if not self.bound:
            kin &= {_code_family(_code_key(s)) for m in marks.values() for s in m['series']}
        takes = {i: {c for c in codes if _code_family(c) in kin and re.fullmatch(PART_SERIES, c.upper())} for i, codes in names.items()}
        found += [(code, [i for i in lines if i not in devices]) for code, lines in _fit_odd({i: c for i, c in takes.items() if c}, {}, devices)]
        return ([(code, lines) for code, lines in found if lines], names)

    def fit_faults(self) -> list[str]:
        """Each cart line that breaks a fit, with the code it lacks, the lines that carry it and what could replace it."""
        cart = self.cart_ids()
        if len(cart) < 2:
            return []
        if self.bound:
            self.show_links(cart, self.tail() + 3)
        index = self.fit_index()
        found, names = self.fit_clashes(index)
        faults: list[str] = []
        done: set[str] = set()
        for code, lines in found:
            holders = [i for i in names if i not in lines and _fit_holds(names[i], (index['marks'].get(i) or {}).get('deny') or [], code)]
            for line in lines:
                if line in done:
                    continue
                done.add(line)
                number = self.line_item(line)
                mates = self.fit_mates(number, [code], index) if number else []
                if number and (not mates) and self.fit_seek(number, code, index):
                    index = self.fit_index()
                    mates = self.fit_mates(number, [code], index)
                name = self.items[number - 1].name if number else 'that kind'
                faults.append(f"{line} ({name}) does not fit {code.upper()}, which {', '.join(holders) or 'the other lines'} carries" + (f"; seen listings of that kind that do: {', '.join(mates)}" if mates else ''))
        return faults

    def line_item(self, ident: str) -> int:
        """The requested item a listing's shelf belongs to, 0 when none."""
        cat_key = self.category_key()
        return self.owner_of(str((self.rows.get(ident) or {}).get(cat_key) or '')) if cat_key else 0

    def fit_shared(self, line: str, index: dict[str, Any]) -> set[str]:
        """The codes a cart line has to keep: those it shares with the lines the request ties it to, and any part series it
        shares with another line. Empty when nothing holds it to the rest of the cart."""
        marks = index['marks']
        mine = self.fit_names(line, index)
        tied = next((g for g in self.fit_groups() if self.line_item(line) in g), set())
        shared: set[str] = set()
        for other in self.cart_ids():
            if other == line or other not in marks:
                continue
            theirs = self.fit_names(other, index)
            both = {c for c in mine | theirs if _fit_holds(mine, marks[line]['deny'] if line in marks else [], c) and _fit_holds(theirs, marks[other]['deny'], c)}
            shared |= both if self.line_item(other) in tied else {c for c in both if re.fullmatch(PART_SERIES, c.upper())}
        return shared

    def fit_pool(self, line: str, rows: list[str]) -> list[str]:
        """The rows that can take a cart line's place without breaking what the line shares with the cart."""
        index = self.fit_index()
        shared = self.fit_shared(line, index)
        return [i for i in rows if not shared or self.fit_open_end(i, index) or any((_fit_holds(self.fit_names(i, index), (index['marks'].get(i) or {}).get('deny') or [], c) for c in shared))]

    def fit_wants(self, index: dict[str, Any]) -> list[tuple[int, list[str]]]:
        """Requested items with no cart line yet and the codes the cart asks of them: any code of a line the request ties the
        item to, or a series when the item's own rows belong to that series family."""
        marks = index['marks']
        cart = self.cart_ids()
        held = {self.line_item(i) for i in cart}
        loose = {c for m in marks.values() for c in m['loose']}
        wants: list[tuple[int, list[str]]] = []
        for number in range(1, len(self.items) + 1):
            if number in held:
                continue
            tied = next((g for g in self.fit_groups() if number in g), set())
            kin = {_code_family(_code_key(marks[i]['own'])) for i in self.item_ids(number) if i in marks and re.match(PART_SERIES + '-', marks[i]['own'])}
            codes: list[str] = []
            for line in cart:
                for code in sorted(self.fit_names(line, index)):
                    series = re.fullmatch(PART_SERIES, code.upper()) and _code_family(code) in kin
                    if code not in codes and (series or (self.line_item(line) in tied and code not in loose)):
                        codes.append(code)
            if codes:
                wants.append((number, sorted(codes, key=lambda c: not re.fullmatch(PART_SERIES, c.upper()))))
        return wants

    def fit_walk(self) -> None:
        """Device and parts chosen together: once a cart line carries a code, look for each open item's listings that fit it."""
        if self.buys or self.finished or len(self.items) < 2 or not self.cart_ids():
            return
        index = self.fit_index()
        for number, codes in self.fit_wants(index):
            if not self.fit_mates(number, codes, index):
                self.fit_seek(number, codes[0], index)

    def fit_plan(self) -> str:
        index = self.fit_index()
        wanted = [f"{self.items[n - 1].name}: {', '.join(mates)} fit {' or '.join((c.upper() for c in codes[:3]))}" for n, codes in self.fit_wants(index) for mates in [self.fit_mates(n, codes, index)] if mates]
        ask = '[fit: seen listings that fit the cart, cheapest first] ' + ' | '.join(wanted) if wanted else ''
        clashes = '; '.join((f"{', '.join(odd)} lack {code.upper()}" for code, odd in self.fit_clashes(index)[0]))
        ask += ('\n' if ask and clashes else '') + (f'[fit: cart lines that do not go together] {clashes}' if clashes else '')
        if not self.bound:
            return ask
        lines = []
        for main in sorted({m for _, m in self.bound}):
            parts = [p for p, m in self.bound if m == main]
            tally: dict[str, dict[int, int]] = {}
            for p in parts:
                for i in self.item_ids(p):
                    if self.sold_out(i):
                        continue
                    for code in self.fit_names(i, index):
                        tally.setdefault(code, {}).setdefault(p, 0)
                        tally[code][p] += 1
            ranked = sorted(tally.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1].values())))[:4]
            devices = {}
            for code, _ in ranked:
                devices[code] = [i for i in self.item_ids(main) if any((_code_names(code, c) or _code_names(c, code) for c in self.fit_names(i, index)))][:2]
            text = '; '.join((f'{code.upper()}: ' + ', '.join((f'{n} {self.items[p - 1].name}' for p, n in sorted(per.items()))) + (f" (device seen: {', '.join(devices[code])})" if devices[code] else ' (device not seen yet)') for code, per in ranked))
            if text:
                lines.append(f"codes the {self.items[main - 1].name} candidates' parts name -> {text}")
        if not lines:
            return ask
        text = "[fit plan: a device fits a part when the part's text names the device's model code, its series or another name it is sold as, or when the device's text names the part's series; list the device's category and read the codes on its rows] " + ' | '.join(lines)
        return text + ('\n' + ask if ask else '')

    def missing_items(self) -> list[int]:
        """Items with no line in a cart holding fewer lines than items; items no seen listing is labelled for are not counted."""
        cat_key = self.category_key()
        ids = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        if not cat_key or len(ids) >= len(self.items):
            return []
        labels = [str(r.get(cat_key) or '') for r in self.rows.values()]
        held = [str((self.rows.get(i) or {}).get(cat_key) or '') for i in ids]
        return [n for n, item in enumerate(self.items, 1) if any((self.owns(item, label) for label in labels)) and (not any((self.owns(item, label) for label in held)))]

    def states_needs(self, ident: str, number: int) -> bool:
        """Whether a listing's own keys state each of the item's requirements."""
        return all((state == 'proven' for state in self.line_needs(ident, number).values()))

    def need_value(self, need: str) -> tuple[str, str] | None:
        """A requirement as (property, value). A value read from another listing is the chain's resolved value, kept when the
        requirement asks for the same, barred when it asks to be unlike it; None while that value is open, or for any other
        relation (a partner, a distance): that needs the page itself, so such a requirement proves and refuses nothing here."""
        prop, colon, value = need.partition(':')
        if not colon:
            return ('', need.strip())
        if need in {m for _, m in self.refs}:
            chain = next((c for c in self.chains if c.need == need and c.status == 'done'), None)
            if chain is not None and chain.wanted():
                return (prop.strip(), chain.wanted())
            if chain is not None and chain.relation == 'different' and chain.value:
                return (prop.strip(), 'not ' + chain.value)
            known = self.ref_values.get(need) or ''
            unlike = re.search("\\b(?:unlike|differ\\w*|other than|anything (?:but|except)|not (?:the same|match\\w*)|doesn't match)\\b", value, re.I)
            if not known or known.startswith('as stated on') or (not unlike and (not re.search('\\b(?:same|identical|match\\w*|equal\\w*)\\b', value, re.I))):
                return None
            value = ('not ' if unlike else '') + known
        return (prop.strip(), value.strip())

    def need_shape(self, number: int, asked: tuple[str, str]) -> tuple[str, tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...], float, str]:
        """The shape of a requirement's value, read without the words that only repeat the property or the product's noun."""
        item = self.items[number - 1]
        return _need_form(asked[1], ' '.join((asked[0], (_tokens(item.name) or [''])[-1], (_tokens(item.shelf) or [''])[-1])))

    def read_line(self, ident: str) -> bool:
        return ident in self.full or ident in self.opened or ident in self.view_keys

    def need_state(self, number: int, need: str, ident: str) -> str:
        """How a read listing stands against one requirement of an item, decided from its record alone. "proven": a key
        naming the property states the value. "weak": no such key decides, but the value stands elsewhere in the record.
        "silent": no such key, no value. "other": a key naming the property states something else. "doubt": the same in
        words only a reader can settle (same_meaning): its yes keeps the line in play as weak, and proves nothing. A longer
        key decides only when it states the value on some read listing of the item, so a neighbouring property's key never
        counts against a line."""
        record = self.full.get(ident) or (self.rows.get(ident) if self.read_line(ident) else None)
        asked = self.need_value(need)
        if not record or not asked or not asked[1]:
            return 'weak' if record else 'silent'
        prop, value = asked
        kind, wanted, barred, limit, unit = self.need_shape(number, asked)
        facts = {k: v for k, v in record.items() if k != self.id_key and isinstance(v, (str, int, float)) and (not isinstance(v, bool))}
        facts.update(_fact_block(record) or {})
        ranks = {k: _key_rank(prop, k) for k in facts}
        exact = [str(facts[k]) for k in facts if ranks[k] == 2]
        near = [k for k in facts if ranks[k] == 1]
        sure = self.own_keys(number, need)
        deciding = exact + [str(facts[k]) for k in near if k in sure]
        rest = ' '.join((str(v) for k, v in facts.items() if not ranks[k]))
        loose = ' '.join([rest] + [str(facts[k]) for k in near])
        if kind != 'words':
            said = [_within(kind, limit, unit, text) for text in exact + [str(facts[k]) for k in near]]
            if True in said:
                return 'proven'
            if False in [_within(kind, limit, unit, text) for text in deciding]:
                return 'other'
            return 'weak' if _holds((f'{limit:g}',) + ((unit,) if unit else ()), loose) else 'silent'
        hit = [text for text in exact if any((_holds(alt, text) for alt in barred))]
        if any((all((any((_holds(alt, piece) for alt in barred)) for piece in re.split('[,;|/]', text))) for text in hit)):
            return 'other'
        if not wanted:
            return 'proven' if exact and (not hit) else 'weak'
        cat_key = self.category_key()
        label = str(record.get(cat_key) or '') if cat_key else ''
        if any((_holds(alt, text) for alt in wanted for text in exact + [label] + [str(facts[k]) for k in near])):
            return 'proven'
        if deciding:
            pairs = [(value.lower(), text.lower()[:160]) for text in deciding if any((not _number_off(alt, text) for alt in wanted))]
            told = [self.same_pairs.get(pair) for pair in pairs]
            return 'weak' if True in told else 'other' if all((t is False for t in told)) else 'doubt'
        return 'weak' if any((_holds(alt, loose) for alt in wanted)) else 'silent'

    def own_keys(self, number: int, need: str) -> set[str]:
        """Longer keys that state this requirement's value on a read listing of the item: such a key is the property's."""
        mark = (number, need, len(self.full), self.version)
        if mark in self.key_memo:
            return self.key_memo[mark]
        found: set[str] = set()
        asked = self.need_value(need)
        if asked:
            kind, wanted, _, limit, unit = self.need_shape(number, asked)
            for ident in self.item_ids(number) or list(self.full):
                for key, text in (_fact_block(self.full.get(ident) or {}) or {}).items():
                    if _key_rank(asked[0], key) == 1 and (_within(kind, limit, unit, text) is True if kind != 'words' else any((_holds(alt, text) for alt in wanted))):
                        found.add(key)
        self.key_memo[mark] = found
        return found

    def live_needs(self, number: int) -> list[str]:
        return [need for owner, need in self.needs_now() if owner == number and (not self.described(number, need))]

    def line_needs(self, ident: str, number: int) -> dict[str, str]:
        return {need: self.need_state(number, need, ident) for need in self.live_needs(number)}

    def same_meaning(self, ids: list[str]) -> None:
        """Settle the word differences left on these lines with one short question. Without an answer the pair stays a
        doubt, which does not hold a line back."""
        pairs: list[tuple[str, str]] = []
        for number in range(1, len(self.items) + 1):
            for need in self.live_needs(number):
                asked = self.need_value(need)
                if not asked:
                    continue
                wanted = self.need_shape(number, asked)[1]
                keys = self.own_keys(number, need)
                for ident in ids:
                    if self.on_shelf(ident, number) and self.need_state(number, need, ident) == 'doubt':
                        block = {**self.full.get(ident, {}), **(_fact_block(self.full.get(ident) or {}) or {})}
                        for key, text in block.items():
                            pair = (asked[1].lower(), str(text).lower()[:160])
                            if isinstance(text, (str, int, float)) and (_key_rank(asked[0], key) == 2 or key in keys) and pair not in self.same_pairs and pair not in pairs and any((not _number_off(alt, text) for alt in wanted)):
                                pairs.append(pair)
        pairs = pairs[:SAME_PAIRS]
        if not pairs or self.cost > SPEND_HARD or self.no_credit:
            return
        got = self.ask_json(SAME + '\n\nPAIRS:\n' + '\n'.join((f'{n}. wanted: {a} | listing says: {b}' for n, (a, b) in enumerate(pairs, 1))), [LIGHT], 400) or {}
        same = {_count(n) for n in _items(got.get('same'))}
        differ = {_count(n) for n in _items(got.get('differ'))}
        for n, pair in enumerate(pairs, 1):
            if n in same or n in differ:
                self.same_pairs[pair] = n in same
        self.trace.append({'kind': 'same_meaning', 'pairs': pairs, 'same': sorted((n for n in same if n)), 'differ': sorted((n for n in differ if n))})

    def on_shelf(self, ident: str, number: int) -> bool:
        """Whether a listing sits on the shelf an item takes. A label no item names is fine only for an item with no named shelf."""
        cat_key = self.category_key()
        label = str((self.rows.get(ident) or {}).get(cat_key) or '') if cat_key else ''
        owner = self.owner_of(label)
        if owner or not label:
            return owner == number or not label and len(self.items) == 1
        return not any((self.owner_of(str(r.get(cat_key) or '')) == number for r in self.rows.values()))

    def line_counts(self, ident: str, number: int) -> bool:
        """The line test: on the item's shelf, its attributes read, the last read shows it can be bought, nothing says other."""
        return self.on_shelf(ident, number) and self.read_line(ident) and self.buyable(ident) and ('other' not in self.line_needs(ident, number).values())

    def short_items(self) -> list[int]:
        """Items with no counted line in the cart, whose own shelf has been seen in rows. An item no seen label names is
        left out, and so is everything before any page was read, so a miscounted item cannot decline a right cart."""
        if not self.full:
            return []
        cat_key = self.category_key()
        ids = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        labels = {str(r.get(cat_key) or '') for r in self.rows.values()} if cat_key else set()
        return [n for n, item in enumerate(self.items, 1) if (not cat_key and len(self.items) == 1 or any((self.owns(item, label) for label in labels))) and (not any((self.line_counts(i, n) for i in ids)))]

    def best_line(self, number: int, counted: bool=False) -> str | None:
        """The cheapest read, buyable line of an item inside the budget with every requirement proven (or, with `counted`,
        one that counts and has each value stated somewhere in its record). The other cart lines' total stays inside the
        budget too."""
        item = self.items[number - 1]
        others = sum(((self.price_of(i) or 0.0) * (self.held_units(i) or 0) for i in self.cart_ids() if not self.on_shelf(i, number)))
        each = self.brief.get('budget_scope') == 'each'
        pool = []
        for ident in self.item_ids(number) or [i for i in self.seen_order if len(self.items) == 1]:
            price = self.price_of(ident)
            if price is None or not self.read_line(ident) or (not self.buyable(ident)) or (not self.on_shelf(ident, number)):
                continue
            if isinstance(self.budget, (int, float)) and (price if each else price * item.qty + others) > self.budget + 1e-09:
                continue
            if (self.line_counts(ident, number) and set(self.line_needs(ident, number).values()) <= {'proven', 'weak'}) if counted else self.states_needs(ident, number):
                pool.append((price, ident))
        return next((ident for _, ident in sorted(pool) if not self.fit_breaks(ident, number)), None)

    def fit_breaks(self, ident: str, number: int) -> bool:
        """Whether a line brings a clash with the cart lines of the other items, by what the records show."""
        others = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0 and (not self.on_shelf(i, number))]
        if not others:
            return False
        index = self.fit_index()
        return len(self.fit_clashes(index, others + [ident])[0]) > len(self.fit_clashes(index, others)[0])

    def learn_prices(self) -> None:
        """The rows' price and stock keys, for the line test on an order whose pick rule never needed them."""
        tool = self.roles.get('search')
        if not self.price_key and tool and self.last_rows is not None:
            self.read_shape(tool, self.last_rows)

    def shelf_seen(self, number: int) -> bool:
        cat_key = self.category_key()
        if not cat_key:
            return len(self.items) == 1
        return any((self.owns(self.items[number - 1], str(r.get(cat_key) or '')) for r in self.rows.values()))

    def set_cart(self, lines: dict[str, int], drop: list[str]) -> bool:
        """Set the cart in code. False when the tool's schema leaves no way, or a call did not go through."""
        cart = self.roles.get('cart') or ''
        calls = self.cart_calls(lines, drop)
        if calls is None:
            return False
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), 'args', faults)
            if faults:
                return False
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return False
            took = self.cart_took(reply.get('observation'))
            if took:
                self.note_cart(cart, args)
            self.cart_said = _dump(reply.get('observation'))
            self.observe(cart, reply.get('observation'))
            if not took:
                self.unavailable.update((v for v in args.values() if isinstance(v, str) and v in lines))
                self.trace.append({'kind': 'cart_refused', 'args': args})
                return False
        return True

    def cart_took(self, seen: Any) -> bool:
        """Whether the store took a cart call. Stores differ, so this is learnt in the run: once a cart call has answered with
        a flag or the line itself, an answer of one sentence is a refusal."""
        plain = _sentence_only(seen)
        self.cart_plain = self.cart_plain or not plain
        return not (plain and self.cart_plain)

    def open_need(self, number: int, line: str | None) -> str:
        """The requirement a value search should go after: the one the line stands worst on, the item's first without a line."""
        states = self.line_needs(line, number) if line else {need: 'silent' for need in self.live_needs(number)}
        states = {need: state for need, state in states.items() if state != 'proven' and self.need_value(need)}
        return min(states, key=lambda need: ('other', 'silent', 'doubt', 'weak').index(states[need])) if states else ''

    def hunt(self, number: int, need: str) -> bool:
        """One value-first search of two or three words for a requirement no read line proves, and a read of the new rows.
        Only while steps stay clear for the cart, the checkout and a refusal."""
        plan = self.value_template()
        asked = self.need_value(need) if need else None
        if not plan or not asked or (number, need) in self.hunted or len(self.hunted) >= HUNTS or self.left() < self.tail() + HUNT_ROOM or self.moves() >= SEARCH_BY:
            return False
        self.hunted.add((number, need))
        words = [w for w in re.split('[,;|]|\\bor\\b', re.sub('\\([^()]*\\)', ' ', asked[1]))[0].split() if not set(_tokens(w)) <= NEED_SKIP][:2] if self.need_shape(number, asked)[1] else []
        if not words:
            return False
        noun = (_tokens(self.items[number - 1].name) or [''])[-1]
        found = self.search_words(plan[0], plan[1], plan[2], ' '.join(words + ([noun] if noun not in _tokens(' '.join(words)) else [])))
        found = found or self.search_words(plan[0], plan[1], plan[2], ' '.join(words))
        rows = [i for i in self.kind_rows(number, found) if i not in self.full][:3]
        opened = bool(rows) and self.open_rows(rows)
        self.trace.append({'kind': 'hunt', 'item': number, 'need': need, 'words': words, 'rows': rows, 'opened': opened})
        return opened

    def last_look(self, number: int) -> bool:
        """Before an ending with no line on file for an item: a read of the seen rows of its shelf, the ones nearest its values
        first, when steps stay clear for the line and the checkout. Once per item. A row read here still has to prove itself."""
        room = min(3, self.left() - self.tail() - 2)
        if room < 1 or number in self.looked:
            return False
        self.looked.add(number)
        cap = self.budget + 1e-09 if isinstance(self.budget, (int, float)) else None
        rows = [i for i in self.kind_rows(number, self.item_ids(number), cap) if i not in self.full][:room]
        opened = bool(rows) and self.open_rows(rows, self.tail() + 2)
        self.trace.append({'kind': 'last_look', 'item': number, 'rows': rows, 'opened': opened})
        return opened

    def better_line(self, number: int, line: str | None) -> str | None:
        """The line an item should hold, from the read records. A proven line stays, unless the pick is the cheapest and a
        cheaper line is proven. Any other line gives way to a proven one, after one value search if none is on file; a
        line that says other, or no line, gives way to the cheapest that counts. An item with no requirement gets no line
        here: with nothing to prove, that would be a guess. Title words play no part."""
        if not line and not self.live_needs(number):
            return None
        best = self.best_line(number)
        if line and self.line_counts(line, number) and self.states_needs(line, number):
            cheaper = best and self.brief.get('pick') == 'cheapest' and (self.price_of(best) or 0.0) < (self.price_of(line) or 0.0)
            return best if cheaper else line
        if self.brief.get('pick') in ('most', 'least') and line and self.line_counts(line, number):
            return line
        if not best and (self.hunt(number, self.open_need(number, line)) or (not line and self.last_look(number))):
            best = self.best_line(number)
        if best:
            return best
        if line and self.line_counts(line, number):
            return line
        self.same_meaning([i for i in self.item_ids(number) if i in self.full])
        return self.best_line(number, counted=True) or line

    def swap_line(self, number: int, line: str | None, new: str) -> str:
        """Put an item's chosen line in the cart in place of its old one, when steps allow. Returns what was done."""
        lines = {i: self.cart_lines[i] for i in self.cart_ids() if i in self.cart_lines and i != line}
        lines[new] = self.items[number - 1].qty
        drop = [line] if line else []
        calls = self.cart_calls(lines, drop)
        if not calls or len(calls) + self.tail() > self.left() or not self.set_cart(lines, drop):
            return ''
        self.swaps[number] = self.swaps.get(number, 0) + 1
        self.trace.append({'kind': 'line_swap', 'item': number, 'old': line, 'new': new})
        return f'Kyo put {new} in the cart for item {number}' + (f' in place of {line}' if line else '') + ': the records show it fits' + (' at a lower price' if line and self.line_counts(line, number) and self.states_needs(line, number) else '')

    def settle_lines(self) -> tuple[str, str]:
        """The cart against the read records before a checkout. Returns (what Kyo did or wants, verdict): verdict "hold" keeps
        the checkout back for a search, "decline" ends without a purchase because an item still has no line that counts."""
        if not self.full or not self.roles.get('cart'):
            return ('', '')
        self.learn_prices()
        self.same_meaning(self.cart_ids())
        said: list[str] = []
        hints: list[str] = []
        bound = {n for pair in self.bound for n in pair}
        for number in range(1, len(self.items) + 1):
            lines = [i for i in self.cart_ids() if self.on_shelf(i, number) and (self.held_units(i) or 0) > 0]
            if len(lines) > 1 or number in bound or self.swaps.get(number, 0) >= 2 or (not self.shelf_seen(number)):
                continue
            line = lines[0] if lines else None
            new = self.better_line(number, line)
            if new and new != line:
                said.append(self.swap_line(number, line, new))
                if not said[-1] and self.buyable(new):
                    hints.append((f'take {line} out of the cart and ' if line else '') + f'add {new} at quantity {self.items[number - 1].qty} for item {number}')
            if self.finished:
                return ('', '')
        if hints and self.swap_holds < 2 and self.left() > self.tail() + 1:
            self.swap_holds += 1
            return ('The records show a better line: ' + '; '.join(hints) + ', with the cart tool, then check out.', 'hold')
        said.append(self.drop_strays())
        short = self.short_items()
        if not short or not self.can_refuse():
            return ('. '.join((t for t in said if t)), '')
        why = '; '.join((f'item {n} ({self.items[n - 1].name}): ' + self.short_why(n) for n in short[:3]))
        self.trace.append({'kind': 'short_items', 'why': why})
        if self.left() > self.tail() + 3 and self.short_holds + self.line_holds < (3 if self.left() > 12 else 2):
            self.short_holds += 1
            return ('The cart does not hold a line that counts for every item: ' + why + '. Keep searching for that item only: other words for its value, its shelf, the rows you have not opened. An order without a right line for each item earns nothing, so if none is found Kyo ends without a purchase.', 'hold')
        if not self.audited and self.judges():
            self.audited = True
            self.take_review(self.audit_call())
            if not self.short_items():
                return ('. '.join((t for t in said if t)), '')
        return (why, 'decline')

    def drop_strays(self) -> str:
        """With a line on every item's shelf, a line on no requested shelf is not wanted: it comes out, in code."""
        numbers = range(1, len(self.items) + 1)
        cart = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        strays = [i for i in cart if not any((self.on_shelf(i, n) for n in numbers))]
        if not strays or not all((self.shelf_seen(n) and any((self.on_shelf(i, n) for i in cart)) for n in numbers)):
            return ''
        keep = {i: self.cart_lines[i] for i in cart if i in self.cart_lines and i not in strays}
        calls = self.cart_calls(keep, strays)
        if not calls or len(calls) + self.tail() > self.left() or not self.set_cart(keep, strays):
            return ''
        self.trace.append({'kind': 'strays_dropped', 'lines': strays})
        return 'Kyo took ' + ', '.join(strays) + ' out of the cart: no requested item takes that kind'

    def short_why(self, number: int) -> str:
        """Why an item has no counted line, for the model."""
        lines = [i for i in self.cart_ids() if self.on_shelf(i, number)]
        if not lines:
            return 'no line in the cart'
        line = lines[0]
        if not self.read_line(line):
            return f'{line} has not been opened'
        if not self.buyable(line):
            return f'{line} cannot be bought'
        return f'{line} states something else for ' + ', '.join((need for need, state in self.line_needs(line, number).items() if state == 'other'))

    def decline_now(self) -> str:
        """Ending rule C: a checkout with an item that has no counted line ends without a purchase, in code."""
        refuse = self.roles.get('refuse') or ''
        faults: list[str] = []
        args = _shape(self.quiet(refuse, dict(self.brief.get('refuse_args') or {})), self.schemas.get(refuse), 'args', faults)
        if faults or self.left() <= 0:
            return ''
        if self.left() >= 2:
            go, said = self.gate('end')
            if not go:
                return 'Not run yet. ' + said + '\nCart now: ' + self.cart_text()
            if self.finished or self.left() <= 0:
                return ''
        self.end_tried = True
        reply = self.act(refuse, args)
        self.trace.append({'kind': 'short_declined', 'lines': self.cart_ids(), 'reply': _clip(reply, 200)})
        if reply is None:
            return ''
        change = self.observe(refuse, reply.get('observation'))
        if change and (not self.finished):
            self.short_holds = 0
            self.pushed = False
            self.queued = []
            return 'Kyo tried to end without a purchase, since an item has no line that counts, and the store answered with a change: ' + change + '. Apply it to your picks, then finish.'
        return 'Not run (Kyo acted instead): an item has no line that counts, so Kyo ended without a purchase. Store reply: ' + self.show(reply.get('observation'))

    def set_aside(self, number: int) -> None:
        """A requirement no read line of the item has a key for and whose value search found nothing cannot be decided: it
        stays a tie-break and stops counting. A real need on a rare key shows up in that search, so it stays in force."""
        item = self.items[number - 1]
        ids = [i for i in self.item_ids(number) if i in self.full]
        lookups = {need for _, need in self.refs}
        for need in self.live_needs(number):
            if len(ids) < 3 or need in lookups or any((self.need_state(number, need, i) != 'silent' for i in ids)):
                continue
            if (number, need) not in self.hunted:
                self.hunt(number, need)
            ids = [i for i in self.item_ids(number) if i in self.full]
            if (number, need) in self.hunted and all((self.need_state(number, need, i) == 'silent' for i in ids)):
                item.dropped.append(need)
                self.brief['wishes'] = _items(self.brief.get('wishes')) + [need]
                self.version += 1
                self.trace.append({'kind': 'set_aside', 'item': number, 'need': need})
                self.notes.append(f'no listing of item {number} can show {need}: it stays a tie-break')

    def order_instead(self) -> tuple[str, dict[str, Any]] | None:
        """Ending rules A and D: a decline call turns into a checkout, in code, when every item has a line that counts in the
        cart, a proven line on file (not for an item tied to another by a fit, which records alone cannot show), or, near the
        end, a counted line taken out without a cause, and the total fits the latest budget. An item whose shelf never showed
        keeps the decline. Returns the checkout call, with the cart already set."""
        buy = self.roles.get('buy')
        if not buy or not self.full or not self.roles.get('cart'):
            return None
        self.learn_prices()
        cart = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        self.same_meaning(cart + [i for i in self.built if i not in cart])
        lines: dict[str, int] = {}
        bound = {n for pair in self.bound for n in pair}
        for number, item in enumerate(self.items, 1):
            if not self.shelf_seen(number):
                return None
            held = next((i for i in cart if self.line_counts(i, number)), None)
            if not held:
                self.set_aside(number)
                cart = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
            best = self.best_line(number) if number not in bound and (held or self.live_needs(number)) else None
            if not best and not held and number not in bound and self.live_needs(number) and (self.hunt(number, self.open_need(number, None)) or self.last_look(number)):
                best = self.best_line(number)
                cart = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
            sure = bool(held) and self.states_needs(held, number)
            if best and (not held or not sure or (self.brief.get('pick') == 'cheapest' and (self.price_of(best) or 0.0) < (self.price_of(held) or 0.0))):
                pick = best
            else:
                pick = held or (next((i for i in self.built if i not in cart and self.line_counts(i, number)), None) if self.left() <= 5 else None)
            if not pick:
                return None
            lines[pick] = self.held_units(pick) if pick in cart else item.qty
        if not lines or (len(lines) > 1 and self.fit_clashes(self.fit_index(), list(lines))[0]):
            return None
        if isinstance(self.budget, (int, float)):
            prices = {i: self.price_of(i) for i in lines}
            each = self.brief.get('budget_scope') == 'each'
            if None in prices.values() or (max(prices.values()) if each else sum((prices[i] * lines[i] for i in lines))) > self.budget + 1e-09:
                return None
        drop = [i for i in cart if i not in lines]
        calls = self.cart_calls(lines, drop)
        if calls is None:
            self.leads = list(lines)
        if calls is None or len(calls) + 1 > self.left() or (calls and (not self.set_cart(lines, drop))) or self.finished:
            return None
        faults: list[str] = []
        args = _shape(self.order_args(), self.schemas.get(buy) or {}, 'args', faults)
        if faults:
            return None
        self.trace.append({'kind': 'decline_to_order', 'lines': lines, 'drop': drop})
        return (buy, args)

    def order_args(self) -> dict[str, Any]:
        """The checkout call's arguments: the brief's, with a plain line in any required text it left out."""
        spec = self.schemas.get(self.roles.get('buy') or '') or {}
        props = spec.get('properties') or {}
        args = dict(self.brief.get('buy_args') or {})
        args.update({k: 'Every requested item is in the cart.' for k in spec.get('required') or [] if k not in args and 'string' in _kinds(props.get(k) or {}) and ('enum' not in (props.get(k) or {}))})
        return args

    def early_cart(self) -> str:
        """Ending rule G: at the middle of the step limit each item with a proven line holds it, so a stock note still finds
        room. Only a proven line, and none for an item tied to another by a fit, which records alone cannot show: anything
        less in the cart would turn a right decline into an order by rule A."""
        if self.early_done or self.step < self.max_steps // 2 or self.buys or not self.full or not self.roles.get('cart') or (self.roles.get('ask') and (not self.asked) and self.msgs_free() > 0):
            return ''
        self.early_done = True
        self.learn_prices()
        said = []
        bound = {n for pair in self.bound for n in pair}
        for number in range(1, len(self.items) + 1):
            if number not in bound and self.shelf_seen(number) and (not any((self.on_shelf(i, number) for i in self.cart_ids()))):
                pick = self.best_line(number)
                if pick:
                    said.append(self.swap_line(number, None, pick))
        said = [t for t in said if t]
        return '\n[Kyo] The step limit is half used. ' + '. '.join(said) + '. Swap a line if you find a better one.' if said else ''

    def end_due(self) -> bool:
        """Ending rule F: a cart with a line for every item goes to the checkout with at least 3 steps left, so that a refusal
        still leaves room for remove, add, checkout. A short cart keeps those steps for its missing line."""
        return self.left() <= 3 and (not self.end_tried) and (self.buys == 0) and bool(self.roles.get('buy')) and bool(self.cart_ids()) and (not self.short_items())

    def fill_short_cart(self) -> bool:
        """Before a forced order: add, for each item with no line, the cheapest opened listing that states its requirements.
        False when an item stays uncovered, so the episode ends by refusing rather than with a short order."""
        cart = self.roles.get('cart') or ''
        bound = {n for pair in self.bound for n in pair}
        for number in self.missing_items():
            item = self.items[number - 1]
            fits = [i for i in self.item_ids(number) if i in self.full and i not in self.cart_lines and self.buyable(i) and self.price_of(i) is not None and self.live_needs(number) and self.states_needs(i, number)]
            fits.sort(key=lambda i: self.price_of(i) or 0.0)
            pick = fits[0] if fits and number not in bound else None
            if pick and isinstance(self.budget, (int, float)):
                held = sum(((self.price_of(i) or 0.0) * (self.held_units(i) or 0) for i in self.cart_ids()))
                cost = (self.price_of(pick) or 0.0) * (1 if self.brief.get('budget_scope') == 'each' else item.qty)
                if cost + (0.0 if self.brief.get('budget_scope') == 'each' else held) > self.budget + 1e-09:
                    pick = None
            calls = self.cart_calls({**{i: self.cart_lines[i] for i in self.cart_ids() if i in self.cart_lines}, pick: item.qty}, []) if pick else None
            if not calls or len(calls) + 1 > self.left():
                return False
            for args in calls:
                faults: list[str] = []
                args = _shape(args, self.schemas.get(cart), 'args', faults)
                if faults or self.act(cart, args) is None:
                    return False
                self.note_cart(cart, args)
            self.trace.append({'kind': 'filled_line', 'item': number, 'line': pick})
        return True

    def fuller_label(self, number: int, label: str) -> str:
        cat_key = self.category_key()
        mine = _roots(self.items[number - 1].name)
        if len(mine) < 2 or mine <= _roots(label) or not cat_key or (self.items[number - 1].shelf and _roots(label) == _roots(self.items[number - 1].shelf)):
            return ''
        return next((seen for seen in sorted({str(r.get(cat_key) or '') for r in self.rows.values()}) if seen and mine <= _roots(seen)), '')

    def line_faults(self) -> list[str]:
        self.adopt_labels()
        cat_key = self.category_key()
        ids = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        if not ids or not cat_key:
            return []
        label = {i: str((self.rows.get(i) or {}).get(cat_key) or '') for i in ids}
        known = {n for n, item in enumerate(self.items, 1) if any((self.owns(item, str(r.get(cat_key) or '')) for r in self.rows.values()))}
        owner = {i: [n for n, item in enumerate(self.items, 1) if self.owns(item, label[i])] for i in ids}
        faults: list[str] = []
        for n, item in enumerate(self.items, 1):
            if n not in known:
                continue
            lines = [i for i in ids if n in owner[i]]
            if not lines:
                faults.append(f'item {n} ({item.name}) has no line in the cart')
            elif len([i for i in lines if owner[i] == [n]]) > 1:
                faults.append(f"item {n} ({item.name}) has {len(lines)} lines ({', '.join(lines)}); keep one")
            for i in lines:
                have = self.held_units(i)
                if owner[i] == [n] and have != item.qty:
                    faults.append(f'{i} is held at {have} unit(s); item {n} ({item.name}) wants {item.qty}')
                fuller = self.fuller_label(n, label[i])
                if fuller:
                    faults.append(f'{i} is from {label[i]}, but the store has the label {fuller}, which names item {n} ({item.name}) in full: take the line from there')
        if len(known) == len(self.items):
            for i in ids:
                if not owner[i]:
                    faults.append(f"{i} ({label[i] or 'unknown kind'}) is not a kind the shopper asked for: take it out")
        for i in ids:
            if self.sold_out(i):
                faults.append(f'{i} is sold out')
        prices = {i: self.price_of(i) for i in ids}
        if isinstance(self.budget, (int, float)) and all((p is not None for p in prices.values())):
            if self.brief.get('budget_scope') == 'each':
                over = [i for i in ids if self.budget < prices[i] <= 10 * self.budget]
                faults += [f'{i} costs {prices[i]:g}, over the {self.budget:g} limit for each item' for i in over]
            else:
                total = sum((prices[i] * self.held_units(i) for i in ids))
                if self.budget + 1e-09 < total <= 10 * self.budget:
                    faults.append(f'the total is {total:g}, over the budget of {self.budget:g}')
        return faults

    def count_fixes(self) -> list[tuple[str, int, int]]:
        cat_key = self.category_key()
        if not cat_key:
            return []
        out = []
        for ident in self.cart_ids():
            have = self.held_units(ident) or 0
            owners = [n for n, item in enumerate(self.items, 1) if self.owns(item, str((self.rows.get(ident) or {}).get(cat_key) or ''))]
            mine = [i for i in self.cart_ids() if (self.cart_lines.get(i, 1) or 0) > 0 and len(owners) == 1 and self.owns(self.items[owners[0] - 1], str((self.rows.get(i) or {}).get(cat_key) or ''))]
            if have > 0 and len(owners) == 1 and (len(mine) == 1) and (have != self.items[owners[0] - 1].qty) and (not self.sold_out(ident)):
                out.append((ident, have, self.items[owners[0] - 1].qty))
        return out

    def fix_counts(self, wrong: list[tuple[str, int, int]]) -> list[str]:
        name = self.roles.get('cart') or ''
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        target = next((k for t, k in self.id_args if t == name and k in texts), None) or next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        count = next((k for k, v in props.items() if isinstance(v, dict) and 'integer' in _kinds(v)), None)
        mode = next((k for k, v in props.items() if isinstance(v, dict) and isinstance(v.get('enum'), list) and (len(v['enum']) >= 2)), None)
        if not name or not target or (not count):
            return []
        calls: list[dict[str, Any]] = []
        for ident, have, want in wrong:
            if self.cart_sets(name):
                calls.append({target: ident, count: want})
            elif mode:
                options = [str(o) for o in props[mode]['enum']]
                off = next((o for o in options if re.search('remov|delet|drop|clear|take', o, re.I)), options[1])
                on = next((o for o in options if o != off), options[0])
                calls += [{target: ident, mode: off, count: have}, {target: ident, mode: on, count: want}]
            else:
                return []
        if len(calls) + 3 > self.left():
            return []
        done = []
        for args in calls:
            faults: list[str] = []
            args = _shape(args, schema, 'args', faults)
            if faults:
                break
            reply = self.act(name, args)
            if reply is None or self.finished:
                break
            self.note_cart(name, args)
            self.cart_said = _dump(reply.get('observation'))
            self.observe(name, reply.get('observation'))
            done.append(_dump(args) + ' -> ' + self.show(reply.get('observation'))[:300])
        self.trace.append({'kind': 'count_fix', 'calls': done})
        return done

    def check_key(self, ids: list[str]) -> str:
        return '|'.join(sorted(ids)) + f'#{self.version}'

    def check_cart(self, ids: list[str] | None=None, limit: float=SPEND_SOFT) -> str:
        lead = ids is not None
        ids = ids if lead else self.cart_ids()
        key = self.check_key(ids)
        if key in self.checked or not self.judges(limit) or (not ids):
            return ''
        self.checked.add(key)
        prompt = CHECK + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
        if lead:
            quantity = '; '.join((f'item {n}: quantity {it.qty}' for n, it in enumerate(self.items, 1)))
            prompt += '\n\nCART (planned, judge these listings as if they were carted at the requested quantities: ' + quantity + '):\n' + ', '.join(ids)
        else:
            prompt += '\n\nCART:\n' + self.cart_text()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + '\n'.join((f'{i}: ' + ('SOLD OUT (the store says it can no longer be bought). ' if self.sold_out(i) else '') + self.record_of(i) for i in ids))
        if self.brief.get('pick') in ('cheapest', 'most', 'least'):
            prompt += '\n\nOTHER LISTINGS SEEN (newest first):\n' + self.others(ids, 15)
        judges = self.judges(limit)
        got = None
        if len(judges) > 1 and not lead and not self.audited and self.brief.get('pick') not in ('cheapest', 'most', 'least'):
            got = self.ask_json(prompt, [LIGHT], 3000)
            if got:
                judges = [HEAVY] if got.get('ok') is False and HEAVY in self.judges(limit) else []
        if judges and not self.no_credit:
            got = self.ask_json(prompt, judges, 3000) or got
        self.trace.append({'kind': 'check', 'cart': ids, 'got': got})
        self.verdicts[key] = bool(got) and got.get('ok') is not False
        if not got or got.get('ok') is not False:
            return ''
        self.leads = [i for i in self.leads if i not in ids]
        problems = [str(p) for p in _items(got.get('problems'))][:5]
        better = [str(b) for b in _items(got.get('better')) if str(b) in self.rows and (not self.sold_out(str(b)))][:4]
        text = 'A careful review says this cart is not right yet: ' + '; '.join(problems or ['see the requirements'])
        if better:
            text += '. Listings that look better: ' + ', '.join(better)
        return text + '. Fix the cart (or show the review is wrong by re-reading the records), then check out.'

    def trail(self) -> str:
        """The order hold: an ending waits while a chain page is unread by the notes tool and the walk can still go on. Returns what the walk
        found, for the loop model to act on. Holds are counted, and a walk with nothing left to do lets the ending through, so an episode always ends."""
        if self.trail_blocks >= CHAIN_HOLDS or not self.open_chains():
            return ''
        said = self.walk(True)
        if said:
            self.trail_blocks += 1
            self.trace.append({'kind': 'chain_hold', 'left': self.left(), 'open': len(self.open_chains())})
        return said

    def live_chains(self) -> list[Chain]:
        for number, need in self.refs:
            item = self.items[number - 1]
            if need not in item.dropped and (not any((c.need == need for c in self.chains))):
                self.chains.append(Chain(number, need))
                if need not in item.musts + item.told:
                    item.musts.append(need)
        return [c for c in self.chains if c.need not in self.items[c.number - 1].dropped]

    def open_chains(self) -> list[Chain]:
        return [c for c in self.live_chains() if c.status == 'open']

    def listing_tools(self) -> list[str]:
        others = {self.roles.get(r) for r in ('search', 'compare', 'ask', 'cart', 'cart_view', 'buy', 'refuse')}
        return [t for t in dict.fromkeys((self.roles.get('offer'), self.roles.get('open'), *self.schemas)) if t and t not in others and self.single_target(t)]

    def notes_tool(self) -> str | None:
        """The listing tool that shows a listing's free notes, learnt from the pages seen: one that showed text under a field search rows do not
        have, else one that showed such a field empty, else one not tried yet, the page roles first. With a single page tool there is nothing to choose."""

        def rank(tool: str) -> int:
            fields = self.note_fields.get(tool)
            if fields is None:
                return 2 if tool in (self.roles.get('offer'), self.roles.get('open')) else 1
            return 4 if any(fields.values()) else 3 if fields else 0
        return max(self.listing_tools(), key=rank, default=None)

    def saw_listing(self, name: str, args: dict[str, Any], seen: Any) -> None:
        """What a chain needs from any store result: the fields search rows carry, on one listing's page the free text those rows lack, and which
        search forms found nothing."""
        if name != self.roles.get('search') and name not in self.listing_tools():
            return
        self.absorb(seen)
        found = self.records_in(seen)
        if name == self.roles.get('search'):
            parts = list(seen.values()) if isinstance(seen, dict) else [seen]
            if not any((isinstance(v, list) and v for p in parts for v in (p.values() if isinstance(p, dict) else [p]))):
                self.dead_forms.add(self.search_form(args))
                self.dead_codes.update((_code_family(c) for c in self.code_tokens(self.query_of(args))))
                self.dead_filters.update((k for k in args if self.filter_keys(k, args[k])))
            self.last_found = [i for i, _ in found]
            for _, row in found:
                for key, value in row.items():
                    size = self.row_keys.setdefault(key, [0, 0])
                    size[0] += len(value) if isinstance(value, str) else 0
                    size[1] += 1
            return
        if not self.row_keys:
            return
        page = found[0][1] if found else next((v for v in (seen.values() if isinstance(seen, dict) else ()) if isinstance(v, dict) and self.ident(v)), None)
        if page is None:
            return
        fields = self.note_fields.setdefault(name, {})
        texts = []
        for key, value in page.items():
            if key not in self.row_keys and (value in (None, '') or (isinstance(value, str) and len(value.split()) >= 3)):
                fields[key] = fields.get(key, False) or bool(value)
                texts += [value] if value else []
        ident = str(args.get(self.single_target(name) or '') or self.ident(page) or '').strip()
        self.page_notes.setdefault(name, {})[ident] = ' '.join(texts)

    @staticmethod
    def code_tokens(text: Any) -> list[str]:
        return [re.sub('[\\s-]+', '', c).lower() for c in re.findall('\\b' + FIT_CODE + '\\b', str(text or ''))]

    def query_of(self, args: dict[str, Any]) -> str:
        return ' '.join((v for k, v in args.items() if isinstance(v, str) and re.search(QUERY_WORDS, _snake(k), re.I)))

    def search_form(self, args: dict[str, Any]) -> str:
        words = ' '.join(sorted(_words(self.query_of(args)).split()))
        return words + '|' + _dump(sorted(((k, v) for k, v in args.items() if not (isinstance(v, str) and re.search(QUERY_WORDS, _snake(k), re.I))), key=str))

    def dead_search(self, args: dict[str, Any]) -> str:
        """Why a search cannot find anything, from what rows have shown, or ''. A code of a family many row descriptions state and no title shows, or
        of a family that already found nothing; a filter on a row field for a shop's name or any name, once the filter found nothing, that field never held;
        a form that already found nothing."""
        if self.search_form(args) in self.dead_forms:
            return 'this search found nothing before, and nothing has changed. Try other words.'
        rows = self.rows.values()
        codes = self.code_tokens(self.query_of(args))
        if codes and self.row_keys:
            long_key = max(self.row_keys, key=lambda k: self.row_keys[k][0] / max(1, self.row_keys[k][1]))
            titles = {_code_family(c) for row in rows for k, v in row.items() if k in self.row_keys and k != long_key and isinstance(v, str) for c in self.code_tokens(v)}
            stated = [_code_family(c) for row in rows for c in set(self.code_tokens(row.get(long_key)))]
            for code in codes:
                family = _code_family(code)
                if family not in titles and (stated.count(family) >= CODE_ROWS or family in self.dead_codes):
                    return f'{code} is a code from listing descriptions, and no title so far shows a code of that form, so a search cannot find it. Search by the words of the product instead.'
        for arg, value in args.items():
            keys = self.filter_keys(arg, value)
            name = _words(str(value))
            if not keys or any((_words(str(row.get(k) or '')) == name for row in rows for k in keys)):
                continue
            shops = {_words(c.target.get('shop') or '') for c in self.chains}
            if arg in self.dead_filters or name in shops or any((isinstance(v, str) and _words(v) == name for row in rows for v in row.values())):
                return f'no row so far shows {value!r} under {keys[0]}, and this filter reads only that field. Put the name in the search words instead.'
        return ''

    def filter_keys(self, arg: str, value: Any) -> list[str]:
        """The row fields a search filter reads, by its name, when it is a free text filter other than the query and the category."""
        spec = ((self.schemas.get(self.roles.get('search') or '') or {}).get('properties') or {}).get(arg) or {}
        if not isinstance(value, str) or not re.search('[a-z]', value, re.I) or 'enum' in spec or re.search(QUERY_WORDS, _snake(arg), re.I):
            return []
        words = set(_words(arg).split())
        cat_key = self.category_key()
        return [k for k in self.row_keys if k != cat_key and (set(_words(k).split()) <= words or words <= set(_words(k).split()))]

    def chain_due(self) -> bool:
        return bool(self.searched) and any((c.tried <= CHAIN_TWINS for c in self.open_chains())) and self.left() - self.walk_floor(False) >= 2 and (self.moves() < CHECK_BY)

    def walk_floor(self, hold: bool) -> int:
        """Steps a walk leaves alone. At the order hold, only the checkout. Before that also the messages not sent yet, and for each item without a
        line a cart call, with a search and a page when none of its listings was read yet, so a long chain cannot starve the rest of the order."""
        if hold:
            return self.tail() - (1 if self.buys == 0 else 0)
        waiting = 0 if self.asked else max(0, self.msgs_free())
        cat_key = self.category_key()
        held = {str((self.rows.get(i) or {}).get(cat_key) or '') for i in self.cart_ids()} if cat_key else set()
        short = [n for n, item in enumerate(self.items, 1) if not any((self.owns(item, label) for label in held))]
        return self.tail() + waiting + sum((1 if self.records_of_item(n) else 3 for n in short))

    def walk(self, hold: bool=False) -> str:
        """Follows every open chain in code, near one search and one page read a link, down to the step floor. Returns a report for the loop model,
        or '' when nothing moved."""
        started = self.step
        lines: list[str] = []
        finished: list[int] = []
        for chain in self.open_chains():
            pages = len(chain.walked)
            try:
                self.walk_chain(chain, hold)
            except Exception as exc:
                chain.status = 'lost'
                self.trace.append({'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]})
            pages = chain.walked[pages:]
            if chain.status == 'done':
                lines.append(f'{chain.need} -> {self.ref_values[chain.need]}')
                finished += [chain.number] if chain.wanted() else []
            elif chain.status == 'lost':
                lines.append(f"{chain.need} -> not found: {chain.target.get('about') or 'the listing it points at'}")
            elif pages or self.step > started:
                lines.append(f"{chain.need} -> still to find: {chain.target.get('about') or 'the listing it points at'}")
            self.trace.append({'kind': 'chain', 'cid': chain.need[:80], 'prop': chain.prop, 'shop': chain.target.get('shop', ''), 'state': chain.status, 'why': chain.relation, 'links': len(chain.walked)})
            self.trace += [{'kind': 'link', 'cid': chain.need[:80], 'match': ident} for ident in pages]
        if not lines:
            return ''
        said = f"[Kyo followed the pages these requirements point at, in {self.step - started} steps]\n" + '\n'.join(lines)
        said += '\nApply these values to every pick.' + ('' if any((c.status == 'lost' for c in self.chains)) else ' Kyo reads those pages itself: do not search for them.')
        searched = self.value_search(sorted(set(finished))) if finished and (not hold) else ''
        if searched:
            said += '\n' + searched
        if self.queued:
            said += '\n[notes seen meanwhile] ' + ' | '.join(self.queued)
            self.queued = []
        return said

    def walk_chain(self, chain: Chain, hold: bool) -> None:
        if not chain.target:
            got = self.chain_seat(chain)
            chain.prop = str(got.get('property') or '').strip()
            chain.relation = str(got.get('relation') or 'same').strip().lower()
            chain.distance = str(got.get('distance') or '').strip()
            chain.target = self.chain_target(got.get('next'))
            if not chain.target:
                chain.status = 'lost'
                return
            sold = [i for i in self.last_found if self.sold_by(self.rows.get(i) or {}, chain.target['shop'])]
            chain.fresh = sold if len(sold) == 1 else []
        while chain.status == 'open' and (not self.finished) and (not self.broke) and (self.moves() < CHECK_BY) and (self.cost < SPEND_HARD):
            ident = self.chain_next(chain, hold)
            tool = self.notes_tool()
            if ident is None or tool is None:
                return
            chain.tried += 1
            if chain.tried > CHAIN_TWINS and (not hold):
                return
            if ident not in self.page_notes.get(tool, {}):
                if not self.read_page(ident, tool):
                    chain.status = 'lost'
                    return
                if self.notes_tool() != tool:
                    continue
            note = self.page_notes.get(tool, {}).get(ident, '')
            if not note and self.note_fields.get(tool):
                chain.skipped.add(ident)
                continue
            chain.reads += 1
            self.chain_take(chain, ident, note, self.chain_seat(chain, ident, note), hold)
            if chain.reads >= CHAIN_READS and chain.status == 'open':
                chain.status = 'lost'

    def chain_seat(self, chain: Chain, ident: str='', note: str='') -> dict[str, Any]:
        prompt = WALK + '\n\nREQUIREMENT:\n' + chain.need
        if chain.target:
            prompt += '\n\nLOOKING FOR:\n' + (chain.target['about'] or _dump(chain.target))
        if chain.rules:
            prompt += '\n\nRULES SEEN:\n' + '\n'.join(chain.rules)
        if ident:
            prompt += '\n\nTHE PAGE:\n' + (f'note: {note}\n' if note else '') + 'record: ' + _dump(_clip(self.rows.get(ident) or {}, 300))[:2500]
        got = self.ask_json(prompt, [LIGHT, SPARE], 900) or {}
        self.trace.append({'kind': 'chain_read', 'page': ident, 'got': _clip(got, 150)})
        return got

    @staticmethod
    def chain_target(spec: Any) -> dict[str, Any]:
        if not isinstance(spec, dict):
            return {}
        shop, kind = (re.sub("['\u2019]s$", '', str(spec.get('shop') or '').strip()), str(spec.get('kind') or '').strip())
        values = [str(v).strip() for v in _items(spec.get('values')) if str(v).strip()]
        if not shop and (not values):
            return {}
        return {'shop': shop, 'kind': kind, 'values': values, 'about': str(spec.get('about') or '').strip()}

    def sold_by(self, row: dict[str, Any], shop: str) -> bool:
        """Whether a row's own short text names this shop: found by value, under whatever field this store keeps it."""
        name = _words(shop)
        return any((isinstance(v, str) and _words(v) == name for k, v in row.items() if k != self.id_key))

    def chain_rows(self, chain: Chain) -> list[str]:
        """Listings that may be the one looked for, best first: the shop must fit, the last search's rows lead, and the description's values rank them.
        In a store whose rows never show that shop, the rows of a search for the shop alone stand in."""
        target = chain.target
        values = {w for v in target['values'] for w in re.findall('[a-z0-9]+', v.lower())}
        kind = set(re.findall('[a-z0-9]+', target['kind'].lower()))
        ranked = []
        shown = not target['shop'] or any((self.sold_by(r, target['shop']) for r in self.rows.values()))
        alone = chain.queries[-1:] == [target['shop'].lower()]
        for place, ident in enumerate(self.seen_order):
            row = self.rows.get(ident) or {}
            fits = self.sold_by(row, target['shop']) if target['shop'] and shown else shown or (alone and ident in chain.fresh)
            if ident in chain.walked or ident in chain.skipped or (not fits):
                continue
            words = set(re.findall('[a-z0-9]+', _dump(row).lower()))
            if target['shop'] or values & words or (not values):
                ranked.append((ident not in chain.fresh, -len(values & words), -len(kind & words), place, ident))
        return [r[-1] for r in sorted(ranked)]

    def chain_next(self, chain: Chain, hold: bool) -> str | None:
        """The next listing to read for this link, searching when no row is a good guess yet. None when the steps run out or, with the chain then
        lost, when every form of search is used up."""
        floor = self.walk_floor(hold)
        values = {w for v in chain.target['values'] for w in re.findall('[a-z0-9]+', v.lower())}
        while not self.finished:
            rows = self.chain_rows(chain)
            room = self.left() - floor
            if rows:
                record = set(re.findall('[a-z0-9]+', _dump(self.rows.get(rows[0]) or {}).lower()))
                if rows[0] in chain.fresh or (values and values <= record):
                    return rows[0] if room >= 1 or rows[0] in self.page_notes.get(self.notes_tool() or '', {}) else None
            if room < 2:
                return None
            if not self.chain_search(chain):
                if rows:
                    return rows[0]
                if chain.spare:
                    ident, note, got = chain.spare
                    chain.spare = None
                    self.chain_take(chain, ident, note, {**got, 'is_target': True}, hold)
                    if chain.status == 'open':
                        continue
                    return None
                chain.status = 'lost'
                return None
        return None

    def chain_search(self, chain: Chain) -> bool:
        """One more search for the listing looked for. Seller first, then kind and every value; then without values holding digits, which a store may
        state another way; then fewer words; before each new form, the next page of what was last found. Words that found nothing are not sent again."""
        plan = self.value_template()
        if not plan:
            return False
        tool, base, field = plan
        target = chain.target
        args = more = chain.more
        chain.more = None
        if args is None:
            lead = [target['shop'], target['kind']]
            plain = [v for v in target['values'] if not re.search('\\d', v)]
            forms = [' '.join(w for w in lead + target['values'] if w), ' '.join(w for w in lead + plain if w), ' '.join(w for w in lead if w), target['shop'], ' '.join(w for w in [target['kind']] + target['values'] if w)]
            words = next((f for f in forms if f and f.lower() not in chain.queries), None)
            faults: list[str] = []
            args = _shape({**base, field: words}, self.schemas.get(tool), 'args', faults) if words else None
            if args is None or faults:
                return False
            chain.queries.append(words.lower())
            if self.search_form(args) in self.dead_forms:
                return self.chain_search(chain)
        reply = self.act(tool, args)
        self.walk_steps += 1
        if reply is None or self.finished:
            return False
        observation = reply.get('observation')
        self.observe(tool, observation)
        chain.fresh = [i for i, _ in self.records_in(observation)]
        paged = self.paging(tool, observation)
        chain.deep = chain.deep + 1 if args is more else 0
        if paged and paged[3] is not None and chain.deep < 1:
            chain.more = {**args, paged[2]: paged[3]}
        self.trace.append({'kind': 'chain_search', 'words': args.get(field), 'rows': chain.fresh[:6]})
        return True

    def read_page(self, ident: str, tool: str) -> bool:
        target = self.single_target(tool)
        faults: list[str] = []
        args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
        if faults:
            return False
        reply = self.act(tool, args)
        self.walk_steps += 1
        self.note_fields.setdefault(tool, {})
        if reply is None or self.finished:
            return False
        observation = reply.get('observation')
        self.observe(tool, observation)
        self.note_view(tool, observation)
        self.facts_from[tool] = self.facts_from.get(tool, False) or bool(_fact_block(observation))
        if tool == self.offer_tool():
            self.offer_seen[ident] = self.step
        self.page_notes.setdefault(tool, {}).setdefault(ident, self.show(observation))
        return True

    def chain_facts(self, ident: str) -> bool:
        """Reads a page's attributes when its note left the property's value unsaid: with the facts tool, else side by side with the compare tool."""
        tool = self.facts_tool()
        if tool and self.facts_from.get(tool) is not False:
            return self.read_page(ident, tool)
        many = self.compare_target()
        compare = self.roles.get('compare')
        if not many or not compare:
            return False
        faults: list[str] = []
        args = _shape({many: [ident]}, self.schemas.get(compare), 'args', faults)
        reply = self.act(compare, args) if not faults else None
        self.walk_steps += 1 if reply is not None else 0
        if reply is None or self.finished:
            return False
        self.observe(compare, reply.get('observation'))
        return True

    def chain_take(self, chain: Chain, ident: str, note: str, got: dict[str, Any], hold: bool) -> None:
        """What one page means for its chain: a twin to pass, a further listing to find, or the chain's end, whose value becomes the requirement."""
        rule = str(got.get('rule') or '').strip()
        if rule and rule not in chain.rules:
            chain.rules.append(rule)
        if not got or got.get('is_target') is False:
            chain.skipped.add(ident)
            chain.misses += 0 if got else 1
            chain.status = 'lost' if chain.misses >= 2 else chain.status
            if chain.spare is None and (got.get('next') or got.get('value')):
                chain.spare = (ident, note, got)
            return
        chain.walked.append(ident)
        onward = self.chain_target(got.get('next'))
        same = onward and _words(onward['shop']) == _words(chain.target['shop']) and {_words(v) for v in onward['values']} == {_words(v) for v in chain.target['values']}
        if onward and (not same) and len(chain.walked) < CHAIN_PAGES:
            chain.target, chain.fresh, chain.queries, chain.more, chain.skipped, chain.tried, chain.spare = (onward, [], [], None, set(), 0, None)
            return
        if not str(got.get('value') or '').strip() and ident not in self.full and (self.left() - self.walk_floor(hold) >= 1) and self.chain_facts(ident):
            got = {**got, **self.chain_seat(chain, ident, note)}
            if got.get('is_target') is False:
                chain.walked.pop()
                chain.skipped.add(ident)
                return
        chain.value = str(got.get('value') or '').strip()
        chain.partner = str(got.get('partner') or '').strip()
        chain.status = 'done'
        value = chain.value or f'as stated on {ident}'
        if chain.partner and chain.relation == 'partner':
            value = f'{chain.partner}, the counterpart of {value}, the value on the last page'
        elif chain.distance:
            value += f'; the requirement allows {chain.distance} either side of it'
        self.ref_values[chain.need] = value
        self.version += 1
        self.recent = []

    def offer_tool(self) -> str | None:
        return self.roles.get('offer') or self.roles.get('open')

    def reread_offers(self) -> str:
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return ''
        since = max(self.step - 2, self.refused_at)
        stale = [i for i in self.cart_ids() if self.offer_seen.get(i, -99) < since and i not in self.gone]
        if not stale or self.left() < len(stale) + 3:
            return ''
        lines = []
        for ident in stale:
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
            if faults:
                break
            before = dict(self.rows.get(ident) or {})
            reply = self.act(tool, args)
            if reply is None or self.finished:
                break
            observation = reply.get('observation')
            self.offer_seen[ident] = self.step
            self.observe(tool, observation)
            after = self.rows.get(ident) or {}
            moved = [k for k, v in after.items() if k in before and before[k] != v and (not isinstance(v, (dict, list)))]
            if moved:
                lines.append(f"{ident} changed ({', '.join(moved[:4])}): {_dump(_clip(observation, 300))[:900]}")
        self.trace.append({'kind': 'reread', 'ids': stale, 'changed': len(lines)})
        if lines:
            self.version += 1
        return '\n'.join(lines)

    def sold_out(self, ident: str) -> bool:
        return ident in self.unavailable or ident in self.gone

    def swap_hint(self) -> str:
        lost = [i for i in self.cart_ids() if i not in self.gone and self.sold_out(i)]
        if not lost:
            return ''
        for ident in lost:
            self.gone[ident] = self.step
        self.leads = [i for i in self.leads if i not in lost]
        self.version += 1
        picks = self.replacements(lost, self.twin_rows(lost))
        text = f"[Kyo] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        if picks and self.left() <= 6:
            return text + f" Few steps are left: in this turn add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks[1:]) or 'none'}."
        if picks and self.brief.get('pick') == 'cheapest':
            return text + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(picks)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {picks[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        if picks:
            return text + f" Listings already read that appear to fit that item instead: {', '.join(picks)}. In this turn take the sold-out line out of the cart and add {picks[0]} at the requested quantity, then check out."
        return text + ' Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits.'

    def twin_rows(self, lost: list[str]) -> list[str]:
        """After a cheapest pick sells out: rows at or near its price, read before any dearer one is taken, since listings of one
        price can stand side by side. Lists the shelf up to that price, then to twice it when nothing unread and buyable shows."""
        if self.brief.get('pick') != 'cheapest':
            return []
        plan = self.value_template()
        keep = self.tail() + len(self.items) + 2
        read: list[str] = []
        for line in lost:
            price = self.price_of(line)
            number = self.line_item(line)
            if price is None or not number or self.moves() >= SEARCH_BY:
                continue
            for cap in (price, price * 2):
                if self.left() <= keep + 1:
                    break
                found = self.list_category(number, int(cap) if float(cap).is_integer() else cap, 1 if self.left() > keep + 4 else 0)
                if found is None and plan:
                    found = self.search_words(plan[0], plan[1], plan[2], ' '.join(self.value_texts(number)[:2] + [self.items[number - 1].name]))
                near = self.fit_pool(line, [i for i in self.kind_rows(number, found or [], cap + 1) if i != line])
                fresh = [i for i in near if i not in self.full][:TWIN_OPENS]
                self.fill_facts(fresh, keep)
                read += [i for i in dict.fromkeys(fresh + near) if i not in read][:TWIN_OPENS + 5]
                if fresh:
                    break
        self.trace.append({'kind': 'twins', 'lost': lost, 'read': read})
        return read

    def find_sold(self) -> str:
        ids = [i for i in self.cart_ids() if i not in self.gone]
        if any((self.sold_out(i) for i in ids)):
            return ''
        if len(ids) == 1:
            self.unavailable.add(ids[0])
            return ''
        return self.reread_offers() if ids else ''

    def replacements(self, lost: list[str], first: list[str]) -> list[str]:
        if not self.judges():
            return []
        records = []
        for ident in first + [i for i in reversed(self.seen_order) if i not in first]:
            if ident in lost or ident in self.gone or self.sold_out(ident):
                continue
            records.append(_dump(_clip(self.full.get(ident) or self.rows.get(ident), 200))[:1200])
            if len(records) >= 25:
                break
        if not records:
            return []
        prompt = SWAP + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
        prompt += '\n\nSOLD-OUT CART LINE:\n' + '\n'.join((f'{i}: {self.record_of(i)}' for i in lost)) + '\n\nCART NOW:\n' + self.cart_text()
        prompt += '\n\nOTHER LISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
        got = self.ask_json(prompt, self.judges()[::-1], 3500)
        fits: list[tuple[float, str]] = []
        for entry in _items((got or {}).get('fits')):
            ident = str(entry.get('id') if isinstance(entry, dict) else entry)
            price = entry.get('price') if isinstance(entry, dict) else None
            if ident in self.rows and ident not in lost and (ident not in self.gone) and (ident not in self.cart_ids()) and (ident not in [i for _, i in fits]):
                fits.append((float(price) if isinstance(price, (int, float)) and (not isinstance(price, bool)) else float('inf'), ident))
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.brief.get('pick') == 'cheapest' else [i for _, i in fits]
        self.trace.append({'kind': 'swap', 'lost': lost, 'fits': fits})
        return ranked[:4]

    def audit(self) -> str:
        if self.audited or not self.judges():
            return ''
        self.audited = True
        got = self.audit_call()
        self.take_review(got)
        named = self.audit_fits(got)
        missing = [n for n in range(1, len(self.items) + 1) if not named.get(str(n))]
        if missing:
            rows = [str(i) for i in _items((got or {}).get('open')) if str(i) in self.rows and str(i) not in self.opened and (not self.sold_out(str(i)))]
            opened = bool(rows) and self.open_rows(rows)
            if self.recovery_search(got, missing) or opened:
                got = self.audit_call()
                self.take_review(got)
                named = self.audit_fits(got) or named
        if not named:
            return ''
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return 'A careful review thinks these listings may fit: ' + '; '.join((f"item {k}: {', '.join(v)}" for k, v in named.items())) + (f" ({got.get('reason')})" if got and got.get('reason') else '') + '. Open them and buy what truly fits; end without a purchase only if each one fails.'

    def audit_call(self) -> dict[str, Any] | None:
        cap = self.budget if isinstance(self.budget, (int, float)) else None
        cat_key = self.category_key()
        opened: list[str] = []
        rows: list[str] = []
        for strict in (True, False):
            for ident in reversed(self.seen_order):
                label = str((self.rows.get(ident) or {}).get(cat_key) or '') if cat_key else ''
                if self.sold_out(ident) or (strict and ((cap is not None and (self.price_of(ident) or 0) > cap) or (label and (not any((self.owns(item, label) for item in self.items)))))):
                    continue
                record = self.full.get(ident) if ident in self.full else None
                if record and len(opened) < 20:
                    opened.append('OPENED ' + _dump(_clip(record, 200))[:1000])
                elif not record and len(rows) < 15:
                    rows.append('ROW ' + _dump(_clip(self.rows.get(ident), 120))[:500])
            if opened or rows:
                break
        if not opened and not rows:
            return None
        prompt = AUDIT + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
        answers = [line for line in self.done_log if 'message to the shopper' in line]
        if answers:
            prompt += '\n\nANSWERS FROM THE SHOPPER:\n' + '\n'.join(answers)
        tried = [str(v) for args in self.searched for v in args.values() if isinstance(v, str)]
        if tried:
            prompt += '\n\nSEARCHES ALREADY TRIED:\n' + ' | '.join(dict.fromkeys(tried))
        prompt += '\n\nLISTING RECORDS SEEN (newest first):\n' + '\n'.join(opened + rows)
        got = self.ask_json(prompt, self.judges(), 4000)
        self.trace.append({'kind': 'audit', 'got': got})
        return got

    def take_review(self, got: dict[str, Any] | None) -> None:
        """Apply what the decline review found wrong with the ledger: story set aside, values corrected. A requirement
        the shopper stated in an answer can be corrected but not set aside, and a lookup is left alone."""
        lookups = {need for _, need in self.refs}
        changed = []
        aside: set[int] = set()
        for entry in _items((got or {}).get('not_firm')):
            if isinstance(entry, dict):
                number, need = (_count(entry.get('item')) or 0, str(entry.get('need') or '').strip())
                if 1 <= number <= len(self.items) and number not in aside and need in self.items[number - 1].musts and need not in self.items[number - 1].dropped and need not in lookups:
                    aside.add(number)
                    self.items[number - 1].dropped.append(need)
                    changed.append(f'set aside {need}')
        for entry in _items((got or {}).get('corrected')):
            if isinstance(entry, dict):
                number, need, value = (_count(entry.get('item')) or 0, str(entry.get('need') or '').strip(), str(entry.get('value') or '').strip())
                if 1 <= number <= len(self.items) and need in self.items[number - 1].musts + self.items[number - 1].told and need not in lookups and ':' in value:
                    self.items[number - 1].dropped.append(need)
                    self.items[number - 1].told.append(value)
                    changed.append(f'{need} read as {value}')
        if changed:
            self.version += 1
            self.trace.append({'kind': 'review', 'changed': changed})
            self.notes.append('a review of the request: ' + '; '.join(changed))

    def shelf_of(self, number: int) -> str:
        """The category label of a requested item: the one most of its rows show, else the brief's."""
        cat_key = self.category_key()
        item = self.items[number - 1]
        labels = [str(r.get(cat_key) or '') for r in self.rows.values()] if cat_key else []
        labels = [label for label in labels if label and self.owns(item, label)]
        return max(set(labels), key=labels.count) if labels else item.shelf

    def list_category(self, number: int, cap: float | None, more: int=0) -> list[str] | None:
        """The rows the store's category listing shows for an item, under a price limit when one is given, with up to
        `more` further pages. None when the store has no such listing or it does not answer."""
        spec = self.browse
        label = self.shelf_of(number)
        if not spec or not label or self.browse_misses >= 2 or (cap is not None and (not spec['price'])):
            return None
        args = {**spec['fixed'], spec['category']: label}
        if cap is not None:
            args[spec['price']] = cap
        faults: list[str] = []
        args = _shape(args, self.schemas.get(spec['tool']), 'args', faults)
        if faults or self.finished:
            self.browse_misses += 1
            return None
        reply = self.act(spec['tool'], args)
        if reply is None:
            self.browse_misses += 1
            return None
        seen = reply.get('observation')
        self.observe(spec['tool'], seen)
        found = [i for i, _ in self.records_in(seen)]
        if isinstance(seen, dict) and seen.get('error'):
            self.browse_misses += 1
            return None
        for _ in range(more):
            page = self.paging(spec['tool'], seen)
            if not page or page[3] is None or self.finished:
                break
            faults = []
            call = _shape({**args, page[2]: page[3]}, self.schemas.get(spec['tool']), 'args', faults)
            reply = self.act(spec['tool'], call) if not faults else None
            if reply is None:
                break
            seen = reply.get('observation')
            self.observe(spec['tool'], seen)
            found += [i for i, _ in self.records_in(seen)]
        return found

    def search_words(self, tool: str, base: dict[str, Any], field: str, words: str) -> list[str]:
        faults: list[str] = []
        args = _shape({**base, field: words}, self.schemas.get(tool), 'args', faults)
        if faults or self.finished:
            return []
        reply = self.act(tool, args)
        if reply is None:
            return []
        self.observe(tool, reply.get('observation'))
        self.searched = (self.searched + [dict(args)])[-8:]
        return [i for i, _ in self.records_in(reply.get('observation'))]

    def kind_rows(self, number: int, found: list[str], below: float | None=None) -> list[str]:
        """Buyable rows of the item's kind, unread first, most of the requirement's words first, then cheapest."""
        cat_key = self.category_key()
        item = self.items[number - 1]
        wanted = {w for w in re.findall('[a-z0-9]{3,}', ' '.join(self.value_texts(number)).lower())}
        rows = [i for i in dict.fromkeys(found) if self.buyable(i) and (not cat_key or self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or '')))]
        if below is not None:
            rows = [i for i in rows if (self.price_of(i) if self.price_of(i) is not None else below) < below]
        return sorted(rows, key=lambda i: (i in self.full, -len(wanted & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower()))), self.price_of(i) or 0))

    def recovery_search(self, got: dict[str, Any] | None, missing: list[int]) -> bool:
        """Before a decline stands: list each unfitted item's category and run the searches the review proposes, then
        read the most promising new rows, so that the second look at the evidence has something new to judge. Half of
        the steps it may use are kept for reading; the cart and the checkout stay possible."""
        keep = self.tail() + len(self.items) + 2
        room = min(RECOVERY_STEPS, self.left() - keep)
        if room < 3 or self.moves() >= SEARCH_BY:
            return False
        before = len(self.full)
        last = self.step + room
        searching = self.step + max(1, room // 2)
        plan = self.value_template()
        queries: list[tuple[int, str]] = []
        for entry in _items((got or {}).get('search')):
            if isinstance(entry, dict) and str(entry.get('query') or '').strip():
                queries.append((min(max(1, _count(entry.get('item')) or 1), len(self.items)), str(entry['query']).strip()))
        queries = [q for q in queries if q[0] in missing] or [(n, ' '.join(self.value_texts(n)[:2] + [self.items[n - 1].name])) for n in missing]
        found: dict[int, list[str]] = {n: [] for n in missing}
        for number in missing[:3]:
            if self.step < searching:
                cap = self.budget if isinstance(self.budget, (int, float)) and self.browse and self.browse['price'] else None
                found[number] += self.list_category(number, cap) or []
        for number, words in queries[:4]:
            if plan and self.step < searching:
                found.setdefault(number, [])
                found[number] += self.search_words(plan[0], plan[1], plan[2], words)
        for number, rows in found.items():
            picks = [i for i in self.kind_rows(number, rows) if i not in self.full][:2]
            self.fill_facts(picks[:max(0, last - self.step)], keep)
        self.trace.append({'kind': 'recovery', 'missing': missing, 'queries': queries[:4], 'opened': len(self.full) - before})
        return len(self.full) > before

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get('fits')
        if not isinstance(fits, dict):
            return {}
        named = {}
        for key, listed in fits.items():
            found = re.search('\\d+', str(key))
            number = found.group() if found else '1' if len(self.items) == 1 else str(key)
            named[number] = [str(i) for i in _items(listed) if str(i) in self.rows and (not self.sold_out(str(i)))][:4]
        return {k: v for k, v in named.items() if v}

    def open_rows(self, rows: list[str], spare: int=5) -> bool:
        many, joiner = (self.compare_target(), self.roles.get('compare'))
        tool = self.facts_tool()
        target = self.single_target(tool)
        if many and joiner and (not self.compare_bare()):
            plan = [(joiner, {many: rows[:self.compare_room()]})]
        elif tool and target:
            plan = [(tool, {target: i}) for i in rows[:3]]
        else:
            return False
        if self.left() < len(plan) + spare:
            return False
        before = len(self.full)
        for name, args in plan:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(name), 'args', faults)
            if faults:
                return False
            reply = self.act(name, args)
            if reply is None or self.finished:
                break
            self.observe(name, reply.get('observation'))
        self.trace.append({'kind': 'audit_open', 'rows': rows, 'opened': len(self.full) - before})
        return len(self.full) > before

    def cart_calls(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        name = self.roles.get('cart') or ''
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        for key, spec in props.items():
            shape = spec.get('items') if isinstance(spec, dict) and 'array' in _kinds(spec) else None
            if isinstance(shape, dict) and isinstance(shape.get('properties'), dict):
                inner = shape['properties']
                id_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'string' in _kinds(s) and ('enum' not in s)), None)
                count_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'integer' in _kinds(s)), None)
                if not id_key:
                    return None
                return [{key: [{id_key: i, **({count_key: n} if count_key else {})} for i, n in lines.items()]}]
        texts = [k for k, s in props.items() if isinstance(s, dict) and 'string' in _kinds(s) and ('enum' not in s)]
        target = next((k for t, k in self.id_args if t == name and k in texts), None) or next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        count = next((k for k, s in props.items() if isinstance(s, dict) and 'integer' in _kinds(s)), None)
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) >= 2)), None)
        if not target:
            return None
        calls: list[dict[str, Any]] = []
        out: list[dict[str, Any]] = []
        for ident in drop:
            have = self.cart_lines.get(ident, 1)
            if mode:
                out.append({target: ident, mode: props[mode]['enum'][1], **({count: have} if count else {})})
            elif count and self.cart_sets(name):
                out.append({target: ident, count: 0})
            elif count and isinstance(props[count].get('minimum'), int) and (props[count]['minimum'] <= -have):
                out.append({target: ident, count: -have})
            else:
                return None
        for ident, want in lines.items():
            have = self.cart_lines.get(ident, 0) if ident in self.cart_ids() else 0
            if have == want:
                continue
            if have and (not self.cart_sets(name)):
                return None
            calls.append({target: ident, **({mode: props[mode]['enum'][0]} if mode else {}), **({count: want} if count else {})})
        return calls + out

    def complete_cart(self) -> str:
        cart, buy = (self.roles.get('cart'), self.roles.get('buy'))
        if self.completed or not self.confirmed or self.sold_out(self.confirmed) or (not cart) or (not buy):
            return ''
        self.completed = True
        cat_key = self.category_key()

        def kind(ident: str) -> str:
            return str((self.rows.get(ident) or {}).get(cat_key) or '').strip().lower() if cat_key else ident
        plan: dict[str, int] = {}
        for number, ids in self.lead_items.items():
            found = re.search('\\d+', str(number))
            index = int(found.group()) if found else (1 if len(self.items) == 1 else None)
            pick = next((i for i in ids if not self.sold_out(i)), None)
            if pick and index and (1 <= index <= len(self.items)) and (kind(pick) not in {kind(p) for p in plan}):
                plan[pick] = self.items[index - 1].qty
        if self.confirmed not in plan:
            return ''
        now = self.cart_ids()
        keep = {i: self.cart_lines.get(i, 1) for i in now if i not in plan and (not self.sold_out(i)) and (kind(i) not in {kind(p) for p in plan})}
        drop = [i for i in now if i not in plan and i not in keep]
        if len(plan) + len({kind(i) for i in keep}) < len(self.items):
            return ''
        lines = {**keep, **plan}
        calls = self.cart_calls(lines, drop)
        if not calls or len(calls) + 1 > self.left():
            return ''
        self.fill_facts(list(lines), len(calls) + 3)
        self.show_links(list(lines), len(calls) + 3)
        self.check_cart(list(lines), SPEND_HARD)
        if not self.verdicts.get(self.check_key(list(lines))) or self.fit_clashes(self.fit_index(), list(lines))[0]:
            return ''
        done = []
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), 'args', faults)
            if faults:
                return ''
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return ''
            self.note_cart(cart, args)
            self.observe(cart, reply.get('observation'))
            done.append(_dump(args))
        faults = []
        args = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), 'args', faults)
        if faults:
            return ''
        replies = []
        for attempt in range(2):
            if attempt:
                self.reread_offers()
                self.check_cart(list(lines), SPEND_HARD)
                if not self.verdicts.get(self.check_key(list(lines))) or self.left() < 1 or self.finished or any((self.sold_out(i) for i in lines)):
                    break
            go, said = self.gate('order')
            if said:
                replies.append(said)
            if not go:
                break
            reply = self.act(buy, args)
            self.trace.append({'kind': 'completed', 'lines': lines, 'drop': drop, 'reply': _clip(reply, 200)})
            if reply is None:
                break
            self.buys += 1
            if not reply.get('finished'):
                self.refused_at = self.step
            self.observe(buy, reply.get('observation'))
            replies.append(self.show(reply.get('observation')))
            if self.finished or reply.get('finished'):
                break
        if not replies:
            return ''
        return 'Not run (Kyo acted instead): the independent check confirmed ' + self.confirmed + ', so Kyo set the cart (' + '; '.join(done) + ') and checked out. Store reply: ' + ' | then: '.join(replies)

    def is_buy(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.roles.get('buy'):
            return False
        fixed = self.brief.get('buy_args') if isinstance(self.brief.get('buy_args'), dict) else {}
        return all((args.get(k) == v for k, v in fixed.items()))

    def is_refuse(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.roles.get('refuse'):
            return False
        fixed = self.brief.get('refuse_args') if isinstance(self.brief.get('refuse_args'), dict) else {}
        return all((args.get(k) == v for k, v in fixed.items()))

    def offered(self) -> list[dict[str, Any]]:
        left = self.left()
        ask = self.roles.get('ask')
        keep: set[str] | None = None
        ends = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        if left <= self.tail():
            keep = set(ends) | ({self.roles.get('cart')} if left >= 2 and self.missing_items() else set())
        elif left <= self.tail() + 3:
            keep = set(ends) | {self.roles.get('cart'), self.roles.get('cart_view')}
        elif left <= len(self.items) + 5 and self.buys == 0:
            keep = set(ends) | {self.roles.get('cart'), self.roles.get('cart_view'), self.roles.get('offer')}
        tools = [t for t in self.tools if t['function']['name'] != ask]
        if keep is not None and ends:
            narrowed = [t for t in tools if t['function']['name'] in keep]
            tools = narrowed or tools
        return tools

    def quiet(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        out = dict(args)
        for key, spec in props.items():
            if not isinstance(spec, dict) or 'string' not in _kinds(spec) or 'enum' in spec:
                continue
            value = out.get(key)
            if value is None and key not in (schema.get('required') or []):
                continue
            text = str(value or '')
            named = any((ident in text for ident in self.rows)) or self.brief.get('quiet_decline') is not False
            if named or not text:
                out[key] = 'Sorry, nothing in the store meets all of your requirements, so I am not buying anything.'[:spec.get('maxLength', 200)]
        return out

    def resolve_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if name not in {self.roles.get(r) for r in ('cart', 'open', 'offer', 'compare')} - {None} or not self.id_key:
            return (args, '')
        props = (self.schemas.get(name) or {}).get('properties') or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and ('string' in _kinds(v) or 'array' in _kinds(v)) and ('enum' not in v)]
        if len(free) != 1 or (name != self.roles.get('cart') and (name, free[0]) not in self.id_args):
            return (args, '')

        def fix(value: Any) -> tuple[Any, str]:
            if isinstance(value, str) and value.strip() in self.rows:
                return (value.strip(), '')
            if not isinstance(value, str) or _id_like(value, True) or len(value.strip()) < 8:
                return (value, '')
            low = value.strip().lower()
            hits = [i for i, row in self.rows.items() if any((k != self.id_key and isinstance(v, str) and v.strip().lower().startswith(low) for k, v in row.items()))]
            if len(hits) == 1:
                return (hits[0], '')
            if hits:
                return (value, f"several listings carry the text {value!r} ({', '.join(hits[:6])}); pass one listing id")
            return (value, f'{value!r} is not a listing id; pass the id from the result rows (their {self.id_key} field)')
        out = dict(args)
        for key in free:
            if key not in args:
                continue
            fixed = [fix(v) for v in args[key]] if isinstance(args[key], list) else [fix(args[key])]
            problem = next((p for _, p in fixed if p), '')
            if problem:
                return (args, problem)
            out[key] = [v for v, _ in fixed] if isinstance(args[key], list) else fixed[0][0]
        return (out, '')

    def cart_sets(self, name: str) -> bool:
        props = self.schemas.get(name, {}).get('properties') or {}
        if any((isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) >= 2) for s in props.values())):
            return False
        return any((isinstance(s, dict) and 'integer' in _kinds(s) and (s.get('minimum') == 0) for s in props.values()))

    def whole_removal(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """A removal of a held line sent without a count names the units held, since the store would take off only one."""
        spec = self.schemas.get(name, {}).get('properties') or {}
        count = next((k for k, v in spec.items() if isinstance(v, dict) and 'integer' in _kinds(v)), None)
        removal = any((isinstance((spec.get(k) or {}).get('enum'), list) and v in spec[k]['enum'][1:] for k, v in args.items()))
        ident = next((v for v in args.values() if isinstance(v, str) and v in self.cart_lines), None)
        return {**args, count: min(self.cart_lines[ident], spec[count].get('maximum', self.cart_lines[ident]))} if count and removal and ident and count not in args else args

    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        self.cart_touched = True
        self.cart_at = self.step
        spec = self.schemas.get(name, {}).get('properties') or {}
        removal = False
        for key, value in args.items():
            options = (spec.get(key) or {}).get('enum')
            if isinstance(options, list) and len(options) >= 2 and (value in options[1:]):
                removal = True
        loose_count = next((_count(v) for v in args.values() if _count(v) is not None and (not isinstance(v, str))), None)
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
                        ident = next((v for v in line.values() if isinstance(v, str) and v in self.rows), None)
                        count = next((_count(v) for v in line.values() if _count(v) is not None and (not isinstance(v, str))), 1)
                        if ident:
                            lines.append((ident, count))
                    elif isinstance(line, str) and line in self.rows:
                        lines.append((line, loose_count))
            elif isinstance(value, str) and value in self.rows:
                lines.append((value, loose_count))
        if whole:
            self.cart_lines = {i: c for i, c in lines if c and c > 0}
            return
        sets = self.cart_sets(name)
        for ident, count in lines:
            have = self.cart_lines.get(ident, 0)
            if removal:
                have = have - (count or 1) if any(('integer' in _kinds(v) for v in spec.values() if isinstance(v, dict))) else have - count if count else 0
            elif sets or count == 0:
                have = count if count is not None else 1
            else:
                have = have + (count if count is not None else 1)
            if have > 0:
                self.cart_lines[ident] = have
            else:
                self.cart_lines.pop(ident, None)

    def show(self, observation: Any) -> str:
        text = _dump(_clip(observation, SHOW_CHARS))
        if len(text) > OBS_LIMIT:
            text = text[:OBS_LIMIT] + '...(cut)'
        return text

    def compare_target(self) -> str | None:
        schema = self.schemas.get(self.roles.get('compare') or '') or {}
        props = schema.get('properties') or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in _kinds(v)), None)
        if not target or any((k != target for k in schema.get('required') or [])):
            return None
        return target

    def compare_room(self) -> int:
        spec = ((self.schemas.get(self.roles.get('compare') or '') or {}).get('properties') or {}).get(self.compare_target() or '') or {}
        if isinstance(spec.get('maxItems'), int):
            return spec['maxItems']
        if self.compare_bad:
            return max(2, min(self.compare_bad - 1, self.compare_ok or self.compare_bad // 2))
        return max(4, self.compare_ok)

    def compare_known(self) -> bool:
        spec = ((self.schemas.get(self.roles.get('compare') or '') or {}).get('properties') or {}).get(self.compare_target() or '') or {}
        return bool(self.compare_bad) or isinstance(spec.get('maxItems'), int)

    def compare_rest(self, name: str, many: str, args: dict[str, Any], ids: list[Any]) -> str:
        """Send listings a side-by-side call could not take, in parts of the size the store has shown it takes. The limit is
        learnt from its replies. Returns what the store showed, for the same turn."""
        shown = []
        while ids and self.left() > self.tail() + 3 and (not self.finished):
            part, ids = (ids[:self.compare_room()], ids[self.compare_room():])
            reply = self.act(name, {**args, many: part})
            seen = reply.get('observation') if reply else None
            if reply is None or not any((str(i) in _dump(seen) for i in part)):
                self.compare_bad = min(self.compare_bad or 999, len(part))
                ids = part + ids
                break
            self.compare_ok = max(self.compare_ok, len(part))
            self.absorb(seen)
            self.note_view(name, seen)
            shown.append(self.show(seen))
        text = ''.join(('\n[Kyo] The side-by-side view sent again in a smaller call:\n' + part for part in shown))
        return text + (f"\n[Kyo] Not shown yet: {', '.join(map(str, ids))}." if ids else '')

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        opener, joiner, target = (self.roles.get('open'), self.roles.get('compare'), self.single_target(self.roles.get('open')))
        many = self.compare_target()
        if not opener or not joiner or (not target) or (not many) or (opener == joiner):
            return calls
        if self.compare_bare():
            return calls
        picks: list[tuple[int, str]] = []
        for index, call in enumerate(calls):
            fn = call.get('function') or {}
            args = _json_obj(fn.get('arguments')) if isinstance(fn.get('arguments'), str) else None
            ident = str((args or {}).get(target) or '').strip()
            if fn.get('name') == opener and args and (set(args) == {target}) and (ident in self.rows) and (ident not in self.gone):
                picks.append((index, ident))
        picks = picks[:self.compare_room()]
        if len(picks) < 2:
            return calls
        first = picks[0][0]
        out = []
        for index, call in enumerate(calls):
            if index == first:
                out.append({'id': call['id'], 'type': 'function', 'function': {'name': joiner, 'arguments': _dump({many: [i for _, i in picks]})}})
            elif index in {i for i, _ in picks}:
                out.append({'id': call['id'], 'joined': True})
            else:
                out.append(call)
        self.trace.append({'kind': 'joined', 'ids': [i for _, i in picks]})
        return out

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        calls = self.join_opens(calls)
        for index, call in enumerate(calls):
            allowed = {t['function']['name'] for t in self.offered()}
            cid = call['id']
            if call.get('joined'):
                out.append((cid, 'Shown in the side-by-side result above: Kyo opened these listings together in one step.'))
                continue
            fn = call.get('function') or {}
            name = fn.get('name')
            raw = fn.get('arguments')
            if self.finished:
                out.append((cid, 'Not run: the episode is over.'))
                continue
            if name == self.roles.get('ask'):
                out.append((cid, "Not run (no step used): Kyo's interviewer handles every message to the shopper."))
                continue
            if name not in self.schemas:
                out.append((cid, f"Not run (no step used): unknown tool. Tools: {', '.join(sorted(allowed))}."))
                continue
            if name not in allowed:
                out.append((cid, 'Not run (no step used): too few steps are left for that; use: ' + ', '.join(sorted(allowed)) + '.'))
                continue
            if isinstance(raw, dict):
                args = raw
            else:
                args = _json_obj(raw) if (raw or '').strip() else {}
                if args is None:
                    out.append((cid, 'Not run (no step used): the arguments were not one JSON object.'))
                    continue
            faults: list[str] = []
            args = _shape(args, self.schemas[name], 'args', faults)
            if faults:
                out.append((cid, 'Not run (no step used): ' + '; '.join(faults[:6])))
                continue
            args, wrong = self.resolve_ids(name, args)
            if wrong:
                out.append((cid, 'Not run (no step used): ' + wrong + '.'))
                continue
            if name == self.roles.get('cart'):
                args = self.whole_removal(name, args)
            trimmed = ''
            many = self.compare_target() if name == self.roles.get('compare') else None
            rest: list[Any] = []
            if many and isinstance(args.get(many), list) and (len(args[many]) > self.compare_room()) and self.compare_known():
                room = self.compare_room()
                trimmed = f'\n[Kyo] Sent the first {room} listings only: this store shows at most {room} side by side.'
                rest = args[many][room:]
                args = {**args, many: args[many][:room]}
            target = self.single_target(name)
            if target and name in (self.roles.get('open'), self.roles.get('offer')) and (str(args.get(target)) in self.gone):
                out.append((cid, f'Not run (no step used): {args[target]} sold out at step {self.gone[str(args[target])]} and will not come back. Replace its cart line with another fitting listing.'))
                continue
            lost = [v for v in args.values() if isinstance(v, str) and v in self.gone and v not in self.cart_lines]
            if name == self.roles.get('cart') and lost:
                out.append((cid, f'Not run (no step used): {lost[0]} sold out at step {self.gone[lost[0]]} and will not come back. Add another fitting listing.'))
                continue
            if name == self.roles.get('cart') and any(((_count(v) or 0) < 0 for v in args.values() if not isinstance(v, str))):
                view = _dump(self.cart_view) if self.cart_seen_at >= self.cart_at else ''
                absent = [v for v in args.values() if isinstance(v, str) and v in self.rows and (v not in self.cart_lines) and (v not in view) and (v not in self.cart_said)]
                if absent:
                    out.append((cid, f'Not run (no step used): {absent[0]} is not in the cart, so there is nothing to take off. Cart now: ' + self.cart_text()))
                    continue
            ending = self.is_buy(name, args) or self.is_refuse(name, args)
            if ending and len(calls) > 1:
                out.append((cid, 'Not run (no step used): send the finishing call on its own, after reading your other results.'))
                continue
            if self.is_refuse(name, args):
                turned = self.order_instead()
                if turned:
                    name, args = turned
                    trimmed += '\n[Kyo] Every item has a line that counts, so ending without a purchase would earn nothing: Kyo checked out instead.'
                if self.finished:
                    out.append((cid, 'Not run: the episode is over.'))
                    continue
            if not ending and self.left() <= self.tail() and (self.roles.get('buy') or self.roles.get('refuse')) and (not (name == self.roles.get('cart') and self.left() >= 2 and self.missing_items())):
                out.append((cid, "Not run (no step used): the last steps are kept for checking out (with the shopper's approval of the basket when it is due) or ending without a purchase."))
                continue
            mark = name + _dump(args)
            if not ending and name != self.roles.get('cart') and (mark in self.recent):
                out.append((cid, 'Not run (no step used): you just made this same call and nothing changed; its result is above.'))
                continue
            readers = {self.roles.get(r) for r in ('search', 'open', 'offer', 'compare')} - {None}
            cached = self.read_cache.get(mark)
            if name in readers and cached and (cached[0] == self.version):
                out.append((cid, f'Not run (no step used): the same call at step {cached[1]} returned this, and nothing has changed since:\n' + cached[2][:3000]))
                continue
            dead = self.dead_search(args) if name == self.roles.get('search') else ''
            if dead:
                out.append((cid, 'Not run (no step used): ' + dead))
                continue
            if target and name in readers and (name != self.roles.get('compare')) and self.shown_already(name, str(args.get(target))):
                out.append((cid, f'Not run (no step used): {args[target]} has shown everything this view gives since the last change. Its record: ' + self.record_of(str(args[target]))))
                continue
            if many and isinstance(args.get(many), list) and args[many]:
                keep = [i for i in args[many] if not self.shown_already(name, str(i))]
                if not keep:
                    out.append((cid, 'Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n' + '\n'.join((f'{i}: {self.record_of(str(i))}' for i in args[many]))[:4000]))
                    continue
                if len(keep) < len(args[many]):
                    trimmed += f"\n[Kyo] Left out {', '.join((str(i) for i in args[many] if i not in keep))}: shown in full already, nothing changed since."
                    args = {**args, many: keep}
                    mark = name + _dump(args)
            walked = self.trail() if ending and self.left() >= 2 else ''
            if walked:
                out.append((cid, 'Not run yet (no step used). ' + walked + '\nCart now: ' + self.cart_text()))
                continue
            if ending and self.left() > 2 and self.moves() < CHECK_BY:
                if self.is_buy(name, args):
                    wider = self.wider_hold()
                    if wider:
                        out.append((cid, wider + '\nCart now: ' + self.cart_text()))
                        continue
                    reread = self.reread_offers()
                    if self.note_sold:
                        self.find_sold()
                    if self.cart_touched and self.cart_seen_at < self.cart_at and self.roles.get('cart_view') and (self.left() > 3):
                        faults = []
                        view_args = _shape({}, self.schemas.get(self.roles['cart_view']), 'args', faults)
                        view = self.act(self.roles['cart_view'], view_args) if not faults else None
                        if view is not None:
                            self.cart_view = view.get('observation')
                            self.cart_seen_at = self.step
                            change = self.observe(self.roles['cart_view'], self.cart_view)
                            if change:
                                self.queued = []
                                out.append((cid, 'Not run yet (no step used). Looking at the cart showed a change: ' + change + '\nCart now: ' + self.cart_text() + '\nApply the change, fix the cart if needed, then check out.'))
                                continue
                    swap = self.swap_hint()
                    held = [i for i in self.cart_ids() if i in self.gone] if self.held < 2 else []
                    if swap or held:
                        self.held += 0 if swap else 1
                        out.append((cid, 'Not run yet (no step used). ' + (swap or f"{', '.join(held)} in the cart is sold out: replace that line first.") + '\nCart now: ' + self.cart_text()))
                        continue
                    misfit = self.fit_faults() if self.fit_holds < 2 and self.left() > self.tail() + 3 else []
                    if misfit:
                        self.fit_holds += 1
                        self.trace.append({'kind': 'fit_hold', 'faults': misfit[:4]})
                        out.append((cid, 'Not run yet (no step used). A part in the cart does not work with the device in the cart: ' + '; '.join(misfit[:3]) + '. Swap the part (or the device), then check out.\nCart now: ' + self.cart_text()))
                        continue
                    faults = self.line_faults() if self.line_holds < 2 and self.left() > self.tail() + 2 else []
                    fixed = []
                    if faults and all((' is held at ' in f for f in faults)):
                        fixed = self.fix_counts(self.count_fixes())
                        faults = self.line_faults() if fixed else faults
                        if fixed and (not faults):
                            self.line_holds += 1
                            out.append((cid, "Not run yet (no step used). Kyo set the cart's quantities itself: " + '; '.join(fixed) + '. If the store refused any of these, fix those lines; then check out.\nCart now: ' + self.cart_text()))
                            continue
                    if faults:
                        self.line_holds += 1
                        self.trace.append({'kind': 'line_hold', 'faults': faults[:6]})
                        out.append((cid, 'Not run yet (no step used). The cart does not match the request yet: ' + '; '.join(faults[:6]) + ". Fix those lines (the cart tool's own way to add or take off units), then check out.\nCart now: " + self.cart_text()))
                        continue
                    lines = self.cart_ids()
                    read = self.fill_facts(lines, self.tail() + (1 if len(lines) >= 2 else 3))
                    self.show_links(lines, self.tail() + 1)
                    read += self.cheaper_rows()
                    settled, stop = self.settle_lines()
                    if self.finished:
                        out.append((cid, 'Not run: the episode is over.'))
                        continue
                    if stop == 'hold':
                        out.append((cid, 'Not run yet (no step used). ' + settled + '\nCart now: ' + self.cart_text()))
                        continue
                    if stop == 'decline':
                        done = self.decline_now()
                        if done:
                            out.append((cid, done + (' [episode finished]' if self.finished else '')))
                            continue
                    elif settled:
                        trimmed += '\n[Kyo] ' + settled + '.'
                        misfit = self.fit_faults() if self.fit_holds < 2 and self.left() > self.tail() + 3 else []
                        if misfit:
                            self.fit_holds += 1
                            self.trace.append({'kind': 'fit_hold', 'faults': misfit[:4]})
                            out.append((cid, 'Not run yet (no step used). ' + settled + '. A part in the cart does not work with the device in the cart: ' + '; '.join(misfit[:3]) + '. Swap the part (or the device), then check out.\nCart now: ' + self.cart_text()))
                            continue
                    verdict = self.check_cart()
                    if verdict:
                        seen = "Kyo read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n' if read else ''
                        out.append((cid, 'Not run yet. ' + seen + verdict + '\nCart now: ' + self.cart_text()))
                        continue
                    if reread and self.cost > SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). Kyo re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'))
                        continue
                elif self.left() > 5:
                    review = self.audit()
                    if review:
                        out.append((cid, 'Not run yet (no step used). ' + review))
                        continue
                    if self.leads and self.left() > 4 and (self.refuse_blocks == 0):
                        self.refuse_blocks = 1
                        self.trace.append({'kind': 'refuse_blocked', 'leads': list(self.leads)})
                        out.append((cid, 'Not run yet (no step used). The review found listings that appear to fit: ' + ', '.join(self.leads) + '. Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again.'))
                        continue
                    if self.leads and self.left() > 4 and (self.refuse_blocks == 1):
                        self.refuse_blocks = 2
                        first = self.leads[0]
                        self.check_cart([first], SPEND_HARD)
                        if self.verdicts.get(self.check_key([first])) and first in self.leads:
                            self.confirmed = first
                            self.trace.append({'kind': 'refuse_blocked', 'confirmed': first})
                            out.append((cid, f"Not run yet (no step used). An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."))
                            continue
            if self.is_buy(name, args) and (self.left() <= 2 or self.moves() >= CHECK_BY) and self.can_refuse():
                self.same_meaning(self.cart_ids())
                done = self.decline_now() if self.short_items() else ''
                if done:
                    out.append((cid, done + (' [episode finished]' if self.finished else '')))
                    continue
            if self.is_refuse(name, args) and self.left() <= 5 and (not self.late_held):
                lines = [i for i in self.built if not self.sold_out(i) and (not any(('other' in self.line_needs(i, n).values() for n in range(1, len(self.items) + 1) if self.on_shelf(i, n))))]
                fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
                missing = [i for i in lines if i not in self.cart_ids() or (i in self.dropped and (not fresh))]
                if len(lines) >= len(self.items) and len(missing) + 1 <= self.left():
                    self.late_held = True
                    self.trace.append({'kind': 'late_decline_held', 'lines': lines, 'missing': missing})
                    redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else 'Check out the cart as it is.'
                    out.append((cid, 'Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. ' + redo + ' End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again.'))
                    continue
            if self.is_refuse(name, args) and (not self.leads) and (not self.pushed) and (self.left() > self.max_steps // 3) and (self.moves() < SEARCH_BY):
                self.pushed = True
                out.append((cid, f"Not run yet (no step used). {self.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if self.is_refuse(name, args):
                done = self.complete_cart()
                if done:
                    out.append((cid, done + (' [episode finished]' if self.finished else '')))
                    continue
                args = self.quiet(name, args)
            if self.is_buy(name, args):
                late = self.late_question()
                if late:
                    out.append((cid, 'Not run yet. ' + late + '\nCart now: ' + self.cart_text()))
                    continue
            if ending and self.left() >= 2:
                go, said = self.gate('order' if self.is_buy(name, args) else 'end')
                if not go:
                    out.append((cid, 'Not run yet. ' + said + '\nCart now: ' + self.cart_text()))
                    continue
                if said:
                    trimmed = '\n[Kyo] ' + said + trimmed
            version = self.version
            self.end_tried = self.end_tried or ending
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, 'The call failed (network or server). You may try again.'))
                continue
            observation = reply.get('observation')
            if many and isinstance(args.get(many), list) and (len(args[many]) > 1):
                shown = _dump(observation)
                if not any((str(i) in shown for i in args[many])):
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    trimmed += f'\n[Kyo] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_room()} per call.'
                    rest = list(args[many]) + rest
                else:
                    self.compare_ok = max(self.compare_ok, len(args[many]))
            if many and rest:
                trimmed += self.compare_rest(name, many, args, rest)
            self.recent = (self.recent + [mark])[-3:]
            self.id_args.update(((name, k) for k, v in args.items() if (v if isinstance(v, list) else [v]) and all((isinstance(x, str) and x in self.rows for x in (v if isinstance(v, list) else [v])))))
            self.absorb(observation)
            if name == self.roles.get('compare'):
                rows = self.records_in(observation)
                if rows:
                    self.facts_from[name] = self.facts_from.get(name, False) or any((_fact_block(r) for _, r in rows))
            if name in readers - {self.roles.get('search')}:
                self.note_view(name, observation)
            plain = {tool for job, tool in self.roles.items() if job != 'search'}
            paged = self.more_pages(name, args, observation) if name not in plain and (not ending) else ''
            if self.single_target(name) and name not in (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask')):
                listing = self.ident(observation) if isinstance(observation, dict) else None
                if listing:
                    self.facts_from[name] = self.facts_from.get(name, False) or bool(_fact_block(observation))
                    if name == self.roles.get('open') and (not self.facts_from[name]):
                        other = next((t for t in self.schemas if t != name and self.single_target(t) and (self.facts_from.get(t) is not False) and (t not in (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask'), self.roles.get('search')))), None)
                        if other:
                            self.roles['open'] = other
                    if not self.facts_from[name] and name != self.roles.get('open') and (not self.roles.get('offer')):
                        self.roles['offer'] = name
                    if name == self.offer_tool():
                        self.offer_seen[listing] = self.step
            if name == self.roles.get('cart'):
                if self.cart_took(observation):
                    self.note_cart(name, args)
                self.cart_said = _dump(observation)
                self.recent = []
                counts = [_count(v) for v in args.values() if not isinstance(v, str) and _count(v) is not None]
                for value in args.values():
                    if counts and isinstance(value, str) and (value in self.rows):
                        if counts[0] <= 0 and (not _in_rows(observation, value)):
                            self.dropped.add(value)
                        elif counts[0] > 0:
                            self.dropped.discard(value)
                lines = [i for i in self.cart_ids() if not self.sold_out(i) and i not in self.dropped]
                if len(lines) >= len(self.built):
                    self.built = lines
            if name == self.roles.get('search') and self.records_in(observation):
                self.searched = (self.searched + [dict(args)])[-8:]
                self.last_rows = observation
            if name == self.roles.get('cart_view'):
                self.cart_view = observation
                self.cart_seen_at = self.step
            if self.is_buy(name, args):
                self.buys += 1
                if not reply.get('finished'):
                    self.refused_at = self.step
            text = self.show(observation) + paged + trimmed
            if name not in plain and (not ending):
                text += self.cheapest_here([observation] + self.paged_obs) + self.cap_ignored(name, args, observation)
            base = text
            proof = self.evidence(observation)
            if proof:
                text += '\n' + proof
            if name == self.roles.get('cart') and any((_count(v) == 0 for v in args.values() if not isinstance(v, str))):
                if any((isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values())):
                    text += '\n[Kyo] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'
            self.note_sold = False
            lines = self.loose_lines(observation)
            if lines:
                meaning = self.take_note(lines)
                if meaning:
                    text += '\n[note read] ' + meaning + '. Apply this change to your picks before checking out.'
            if ending and (not reply.get('finished')) and self.version != version:
                self.pushed = False
                self.short_holds = 0
            refused = self.is_buy(name, args) and (not reply.get('finished'))
            again = refused and any((line in self.spoken(observation) for line in self.sold_lines))
            loose = self.note_sold and (refused or (not self.note_named and (not self.is_buy(name, args))))
            if (loose or again) and self.cart_ids() and (not self.finished):
                seen = self.find_sold()
                if seen:
                    text += "\n[Kyo re-read the cart's lines after the refusal]\n" + seen
            self.note_sold = False
            swap = self.swap_hint()
            if swap:
                text += '\n' + swap
            if self.queued:
                text += "\n[notes seen on Kyo's own calls] " + ' | '.join(self.queued)
                self.queued = []
            if name in readers:
                self.read_cache[mark] = (self.version, self.step, base)
            if reply.get('finished'):
                text += ' [episode finished]'
            elif not ending and index == len(calls) - 1:
                text += self.early_cart()
            out.append((cid, text))
        return out

    def can_refuse(self) -> bool:
        refuse = self.roles.get('refuse')
        if not refuse:
            return False
        faults: list[str] = []
        _shape(self.quiet(refuse, dict(self.brief.get('refuse_args') or {})), self.schemas.get(refuse), 'args', faults)
        return not faults

    def finish(self, again: bool=False) -> str:
        if self.finished or self.broke or self.left() <= 0:
            return ''
        self.trail()
        buy, refuse = (self.roles.get('buy'), self.roles.get('refuse'))
        self.end_tried = True
        try:
            self.order_instead()
        except Exception as exc:
            self.trace.append({'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]})
        if self.finished or self.left() <= 0:
            return ''
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        whole = True
        if holding and len(self.items) >= 2:
            whole = self.fill_short_cart() or not self.can_refuse()
            if whole and len(self.cart_ids()) >= 2:
                self.fill_facts(self.cart_ids(), self.tail())
                if self.left() >= 3:
                    self.show_links(self.cart_ids(), self.tail())
            if not whole:
                self.trace.append({'kind': 'short_cart_refused', 'lines': self.cart_ids()})
        if self.finished or self.left() <= 0:
            return ''
        if holding and whole and self.can_refuse():
            try:
                self.same_meaning(self.cart_ids())
                whole = not self.short_items()
            except Exception as exc:
                self.trace.append({'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]})
            if not whole:
                self.trace.append({'kind': 'short_cart_refused', 'lines': self.cart_ids()})
        if whole and buy and (holding or (self.cart_touched and (not fresh) and (self.left() >= 2))):
            faults: list[str] = []
            args = _shape(self.order_args(), self.schemas.get(buy), 'args', faults)
            if not faults:
                if self.left() >= 2:
                    self.gate('order')
                if self.finished or self.left() <= 0:
                    return ''
                reply = self.act(buy, args)
                self.trace.append({'kind': 'forced_buy', 'reply': _clip(reply, 200)})
                if self.finished or self.left() <= 0:
                    return ''
                if reply is not None and again:
                    self.buys += 1
                    self.refused_at = self.step
                    self.observe(buy, reply.get('observation'))
                    return 'the checkout was refused: ' + self.show(reply.get('observation'))
        if refuse:
            faults = []
            args = _shape(self.quiet(refuse, dict(self.brief.get('refuse_args') or {})), self.schemas.get(refuse), 'args', faults)
            if not faults:
                if self.left() >= 2:
                    self.gate('end')
                if self.finished or self.left() <= 0:
                    return ''
                reply = self.act(refuse, args)
                self.trace.append({'kind': 'forced_refuse', 'reply': _clip(reply, 200)})
                if reply is not None and (not self.finished):
                    self.observe(refuse, reply.get('observation'))
                    return 'ending without a purchase was refused: ' + self.show(reply.get('observation'))
        return ''

    def trim(self) -> None:
        size = sum((len(_dump(m.get('content'))) for m in self.chat))
        if size <= HISTORY_LIMIT:
            return
        tools = [i for i, m in enumerate(self.chat) if m.get('role') == 'tool']
        for index in tools[:max(1, len(tools) // 2)]:
            content = self.chat[index].get('content') or ''
            if len(content) > 600:
                ids = re.findall('"([A-Za-z0-9][A-Za-z0-9_\\-:.]{5,40})"', content)
                ids = [i for i in dict.fromkeys(ids) if i in self.rows][:20]
                self.chat[index]['content'] = content[:400] + f"...(older result shortened; listings in it: {', '.join(ids)})"

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.read_request()
        self.trace.append({'kind': 'brief', 'brief': self.brief})
        opening = self.read_text(self.view.get('opening'))
        if opening is not None:
            self.absorb(opening)
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            intro += '\n\nWHAT THE STORE SHOWS FIRST:\n' + self.show(opening)
        intro += '\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n' + self.status()
        self.chat = [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': intro}]
        self.work(LOOP_CALLS, SPEND_HARD)
        if not self.finished:
            turned = self.finish(again=True)
            if turned and (not self.finished) and (not self.broke) and (not self.no_credit) and (self.left() > 1) and (self.moves() < LAST_TURN):
                self.chat.append({'role': 'user', 'content': '[Kyo] Kyo tried to end the episode and ' + turned + '\nApply what the store says, then finish.\n' + self.status()})
                self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
            if not self.finished:
                self.finish()
        self.trace.append({'kind': 'end', 'finished': self.finished, 'steps': self.step, 'cost': round(self.cost, 5), 'calls': self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.finished and (not self.broke) and (self.left() > 0) and (self.loop_calls < loop_cap) and (self.cost < spend_cap) and (self.env_stalls < STALL_LIMIT) and (self.moves() < LAST_TURN or (not self.cart_ids() and self.moves() < EMPTY_STOP)) and (not self.end_due()):
            if self.chain_due():
                said = self.walk()
                if said:
                    self.chat.append({'role': 'user', 'content': said + '\n' + self.status()})
                    continue
            if self.ready_to_ask():
                summary = self.interview()
                if summary:
                    self.chat.append({'role': 'user', 'content': summary + '\n' + self.status()})
                continue
            self.trim()
            self.loop_calls += 1
            message = self.think(self.chat, [LIGHT, SPARE], 3000, self.offered())
            if message is None:
                self.misses += 1
                if self.no_credit or self.calls >= ALL_CALLS or self.misses > 2:
                    break
                continue
            calls = []
            for number, call in enumerate(message.get('tool_calls') or []):
                if not isinstance(call, dict):
                    continue
                fn = call.get('function') if isinstance(call.get('function'), dict) else {}
                raw = fn.get('arguments')
                parsed = raw if isinstance(raw, dict) else _json_obj(raw) if isinstance(raw, str) and raw.strip() else {}
                calls.append({'id': str(call.get('id') or f'k{self.loop_calls}_{number}'), 'type': 'function', 'function': {'name': str(fn.get('name') or ''), 'arguments': _dump(parsed if isinstance(parsed, dict) else {})}})
            content = message.get('content')
            if not isinstance(content, str):
                content = _dump(content) if content else ''
            if not calls:
                idle += 1
                self.chat.append({'role': 'assistant', 'content': content or '(no call)'})
                self.chat.append({'role': 'user', 'content': 'Call one of the tools to continue.\n' + self.status()})
                if idle >= 3:
                    break
                continue
            idle = 0
            committing = {self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse')} - {None}
            if not self.asked and self.roles.get('ask') and (self.msgs_free() > 0) and any((c['function']['name'] in committing for c in calls)) and (self.left() > 8) and (self.moves() <= ASK_BY):
                self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
                for call in calls:
                    self.chat.append({'role': 'tool', 'tool_call_id': call['id'], 'content': 'Not run (no step used): the shopper is asked first.'})
                summary = self.interview()
                self.chat.append({'role': 'user', 'content': (summary or 'The interviewer had nothing to ask.') + '\n' + self.status()})
                continue
            self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
            self.trace.append({'kind': 'turn', 'text': content[:300], 'calls': [c['function'] for c in calls]})
            replies = self.run_calls(calls)
            self.fit_walk()
            for index, (cid, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += '\n' + self.status()
                self.chat.append({'role': 'tool', 'tool_call_id': cid, 'content': text})

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Kyo(problem_data)
    except Exception as exc:
        return [{'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]}]
    try:
        return episode.run()
    except Exception as exc:
        episode.trace.append({'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]})
        try:
            if not episode.finished:
                episode.finish()
        except Exception as again:
            episode.trace.append({'kind': 'crash', 'error': f'{type(again).__name__}: {again}'[:400]})
        return episode.trace
__all__ = ['agent_main']
