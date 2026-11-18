"""11-yes: buys for one shopper inside an ORO store world.

Base: the public YesYesYes agent (race-16 winner). Added from our 11-halo line: listings sent as one line of
"key: value | ..." text are read as records; a reply without a cost is priced from its tokens so the spend lines
hold on every route; a lighter walker history; a status line naming read listings that fit an item not carted yet.
YesYesYes notes follow.

Built on the public Know Thyself v1 agent (itself built on KYO v6). A light model walks the store with the world's
own tools; code keeps the ledger, questions the shopper and stops wasted steps; a heavy model reads the request and
checks the cart before an order and the evidence before a decline.

Changes from Know Thyself v1:
- A wall-clock budget per episode: exploring stops in time to finish, and timeouts shrink as the budget runs down.
- The heavy model stops a little earlier in an episode's spending, to stay clear of the per-run inference cap.
- A heavy model at its request limit hands the call to the next of its kin at once instead of waiting.
- A decline is reviewed against the request and the shopper's answers: requirements that only repeat the story are
  set aside, values that picked up connecting words are corrected, and new searches run before the second look.
- A stated answer is read again by the heavy model, so its value reaches the ledger exactly.
- No re-asking with example values; with very few messages, only properties most listings carry are asked about.
- A cheapest pick lists cheaper rows itself before checkout: through a category listing under a price limit where the
  store has one, else through keyword searches that combine the required values with other values listings show.
- Compatibility statements are read in more forms: hyphenated codes, whole series, exceptions, part series.
- Two requested products on one shelf are merged only when their quantities agree.
- One more step is kept back until the first checkout was tried, since a refused checkout needs another.
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
ALL_CALLS = 60
SPEND_SOFT = 0.045
SPEND_HARD = 0.078
HEAVY_ROOM = 0.02
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
ROW_CHARS = 160
SHOW_CHARS = 300
OBS_LIMIT = 8000
HISTORY_LIMIT = 65000
GATE_ROUNDS = 3
AUTO_PAGES = 6
TIGHT_STEPS = 40
CHEAPER_OPENS = 3
EPISODE_SECONDS = 300
EXPLORE_RESERVE = 50
CALL_RESERVE = 15
TIGHT_MESSAGES = 2
RECOVERY_STEPS = 8
BUSY_SECONDS = 20
SWEEP_STEPS = 6
LONG_STEPS = 75
LONG_PAGES = 16
FIRST_LOOK = 3
LONG_READS = 4
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
GUIDE = 'You are Kyo, a careful buyer working for one shopper in an online store. You can act only by calling the store\'s tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.\n\nHOW THE STORE WORKS\n- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool\'s description and schema and use the exact argument names and allowed values.\n- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.\n- Search rows are summaries. Stores split a listing\'s record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Kyo knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.\n- Questions to the shopper are handled for you by Kyo\'s interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.\n- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Kyo carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store\'s tools (product questions go through Kyo\'s interviewer).\n\nWHAT COUNTS AS A FIT\n- The kind of product is decided by the listing\'s category label: only the label that matches the requested item as a whole counts, never a neighbouring one.\n- A requirement on a property is met when the listing\'s attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value\'s words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.\n- An attribute whose key names the requirement\'s property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing about the requirement. Kyo adds [evidence] lines after opened listings with the keys whose names match a requirement.\n- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.\n- When the category label itself states a requirement\'s value (a season or audience in the label), that requirement is met by the category.\n- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing\'s value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.\n- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.\n- When one requested item must fit another (an accessory for a device), choose the two together from the listings\' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.\n- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.\n- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.\n\nCHOOSING\n- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Kyo reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.\n- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.\n- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.\n- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.\n- A request for several different items needs one cart line per item, each at its requested quantity.\n\nCHANGES WHILE YOU WORK\n- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool\'s result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.\n- Re-read a chosen listing\'s record shortly before checking out, because price and stock can change silently.\n\nFINISHING\n- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.\n- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request\'s wording for that ending, and when a remark is needed, keep it short and name no listing.\n- Every reply must call at least one tool. Keep your text short: what you know, then the next move.'
BRIEF = 'You prepare a shopping task for an agent. Read the shopper\'s request and the store\'s tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:\n\n"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request\'s words, "shelf": the store\'s category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request\'s own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property\'s value or limit outright; and never turn words that are part of what the product is called (a season or an audience named with the product) into musts: those belong to "name". A use or occasion the request makes a property of the product itself ("blazers for business wear", "a dress made for parties", "the occasion has to be party") IS a must on that property ("occasion: business"), unless the store\'s category label already names it; an event, person or setting the product is bought for ("to wear at his wedding", "for a colleague"), and how it should sit, feel or look in use, stay context. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.\n"budget": the most the shopper will pay in total as a number in the request\'s own unit, or null.\n"budget_unit": that unit in words (for example "cents"), or null.\n"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.\n"in_stock_only": true unless the request clearly accepts items that are not available.\n"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.\n"pick_measure": what is maximised or minimised for "most"/"least", else "".\n"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.\n"steps": how many steps, moves or tool calls the request allows (integer), or null.\n"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].\n"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.\n"wishes": a list of things the shopper would merely like.\n"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).\n"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper\'s approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.\n"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.\n"fits": a list of {"item": item number, "with": item number} for each requested item that the request says must work with, fit or suit ANOTHER requested item (the one the agent picks); [] when none.\n"browse": how a tool lists one store category, when one can: {"tool": exact tool name, "category": the argument that takes the category label, "max_price": the argument that takes a maximum price, or null, "fixed": the other arguments that put the tool in that mode, else {}}; null when no tool lists a category.\n"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing\'s attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing\'s current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).\n"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.\n"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.\n'
PLAN = 'You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property; a property the shopper has no view on gets "don\'t mind", and one the shopper does not think about at all gets a puzzled reply, so every property named spends part of a message.\n\nYou get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item\'s category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.\n\nChoose the properties in this order:\n0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.\n1. Hinted properties. The request\'s story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product\'s looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.\nName a hinted property by a listed key of that item when one fits the hint (a hint about who will wear it: the key for gender or audience; about its looks: the listed key for colour, pattern or finish that most of its listings carry); a property no listing of the item carries comes after the item\'s three most carried keys.\n2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item\'s listings; once five or more listings were read, avoid keys seen on only one or two of them.\nSkip a property the request already settles for that item (a requirement names that property; a key whose values only repeat a required value, such as a pattern key showing the required decoration, is NOT settled), a second name for a property already chosen, a key whose values each list several purchase options at once, identifiers, model names, warranty and other logistics. Pack sizes, counts and quantities per pack are properties like any other, and a key marked [not the stated ...] is another property than that requirement, whatever words its name shares with it. Rank every item on its own: its hinted properties first, then its listed keys.\n"label" is the property\'s plain name: for a listed key, all of the key\'s words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property\'s name, no example values, no brackets, no second property.\nChoose up to COUNT for each item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story\'s words it comes from, or an empty string"}]}, each item\'s picks in rank order'
HEAR = 'A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "budget": new total limit as a number or null}\n- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3").\n- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop\'s product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer\'s own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.\n- "free": the property of each question the shopper said they do not mind about.\n- Folksy or roundabout wording that names a value still states that value.\n- The value is only what the property should be. Words that merely join a property to its value (such as matching, set to or coming in as) are not part of it: "size set to large" states large. This does not touch a value pointed at on another listing, which is copied whole.'
NOTE = 'While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:\n{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request\'s unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}\nA replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note.'
CHECK = 'You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.\nReading rules: a requirement is met when the listing\'s attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value\'s words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing\'s value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.\nWhen the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart\'s choice on that rule, the cart is wrong.\nReturn ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses.'
READING = 'the listing\'s attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value\'s words counts, such as "deep teal" for teal, limits include the edge); a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing\'s value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit.'
AUDIT = 'A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: ' + READING + '\nRecords marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.\nThe requirements were written down by a quick reader, so first compare each one with the REQUEST and the ANSWERS FROM THE SHOPPER. A requirement that only repeats the story (who it is for, when, where or why it will be used, what went wrong before, how it should feel) and that the request does not demand of the product goes under "not_firm". A value that picked up words which only join a property to its value goes under "corrected" with the plain value. Judge the listings without the not_firm requirements and with the corrected values. A requirement the request or an answer states outright stays, even when no listing meets it: ending without a purchase is then right.\nReturn ONE JSON object and nothing else: {"not_firm": [{"item": item number, "need": the requirement copied exactly}], "corrected": [{"item": item number, "need": the requirement copied exactly, "value": "property: plain value"}], "fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "search": [at most 4 of {"item": item number, "query": "a keyword search of two or three words, not tried yet, that could surface a fitting listing: another name for the product with one required value (long queries often match nothing)"}], "reason": "one sentence"}'
UNMIX = 'A shopper\'s request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request\'s wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}'
TRAIL = 'A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing\'s page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement\'s chain from the requirement\'s own description, link by link, through the pages\' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).\nReturn ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing, "find": "for a link not found: search words (its shop and the values the description gives)"}], "done": true when the last link is an OPENED page that states the property\'s value itself (also when the requirement asks for a value other than or unlike that one: the page\'s value is what it is measured against), else false, "value": the property\'s value as stated on that last page (when that page states a pairing rule for the property, such as "our X pairings are A with B, ..." or "A goes with B", the value is the partner of the own X of that page: a page whose own X is A gives B), or ""}]}\nA record is a link only when it matches the whole description of that link: the shop or brand, the kind of product AND every property value the description gives (for example "a 10 pieces multipack" rules out the same shop\'s listing with 10 capsules and no multipack); when no record matches all of it, that link is not found yet, even if the same shop sells other listings of that kind. Give "find" for such a link: the shop\'s name and the description\'s values, as a few search words.\nList each chain\'s links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown.'
GATE = 'A shopping agent is about to STAGE. The shopper\'s request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product\'s properties do not do a step.\nWhen a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.\nReturn ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}'
SWAP = "A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: " + READING + '\nReturn ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line\'s item, at most 6], "reason": "one sentence"}'
SHAPE = 'A store tool returned the result below. Name its parts by the exact keys the result and the tool\'s parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing\'s price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}'
GUARD_KEYS = 3
GUARD_SHARE = 0.0
GUARD_GAP = 0.34
STOCK_PICK = '^\\s*(?:availability|available|in[ -]stock|stock status|stock level)\\s*$'
NOT_PROPERTY = '\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b'
CATCH_ALL = '^(?!.*\\b(?:types?|styles?|kinds?|shapes?)\\b).*\\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\\w*|accessor\\w*|extras?|misc\\w*)\\b'
WIDEN_EACH = 6
FIT_CODE = '[A-Z][A-Za-z]{0,7}(?:[- ]?\\d{3,6}|[- ]?\\d{1,5}[xX]{1,4})'
PART_CODE = '[A-Z]{1,5}\\d{0,3}-?\\d{3,7}'
PART_SERIES = '[A-Z]{1,5}\\d{1,3}'
HEDGE = re.compile('not|no|least|most|under|over|unlike|different|min|max|minimum|maximum')
FILLER = re.compile('the|and|with|from|that|comes|listing|page|its|for|of')
M_FIRST = 2
M_LIST = 3
M_ONE = 3
M_ROTATE = 0
EARLY_ROUNDS = 3
EARLY_STEPS = 45
M_LAST = 6
M_REFUSED = 'one by one|one at a time|at a time|one subject|one topic|one query|one question|one prompt|trickle|smaller bites|split (?:them|it)|first thing|separately|in one go|too many|bit much|a pile|flood|one thing'
M_NO_VIEW = 'don.t mind|no preference|any of them|any option|happily take|haven.t (?:any|an|got)|no opinion|whatever|not fussed|nothing comes to mind|pick (?:any|whatever|what|one)|choose|surprise me|not attached|not bothered|either|leave (?:that|it) open|no rule|freely|not got a side|pot luck|owt goes|up to you|your call|no strong|all the same|no view'
LOW_WINS = '\\b(?:cheap\\w*|low\\w*|small\\w*|less|least)\\b(?:[\\s,-]+\\w+){0,3}?[\\s,-]+(?:total|price|cost|bill|spend)\\w*(?:[\\s,-]+\\w+){0,3}?[\\s,-]+(?:wins?|prevails?)\\b|\\bcheapest\\b(?:[\\s,-]+\\w+){0,3}?[\\s,-]+wins?\\b'
FIT_WORDS = '(?:works? (?:with|on)|compatible with|compatibility|fitment|fits?|to fit|for use with|usable with|designed for|made for|built for|intended for|suits?|goes with|pairs? with|replacement for|replaces|supports?)'
REFERS = '\\b(?:more on|see|details? (?:on|in)|rest (?:on|in)|continues? on)\\s+part\\s+(' + PART_CODE + ')'
FIT_EXCEPT = '\\b(?:except(?:\\s+for)?|excluding|other than|but not|apart from|not including|not for)\\b'

def _codes(text: Any) -> list[str]:
    return [re.sub('[\\s-]+', '', c).lower() for c in re.findall('\\b' + FIT_CODE + '\\b', str(text or ''))]

def _code_names(stated: str, code: str) -> bool:
    """Whether a stated code or series names a model code: a series ends in x's in place of digits."""
    tail = re.search('(?<=\\d)x+$', stated)
    if not tail:
        return stated == code
    return code.startswith(stated[:tail.start()]) and len(code) == len(stated)

def _code_family(code: str) -> str:
    found = re.match('[a-z]+', code)
    return found.group(0) if found else ''

@lru_cache(maxsize=4096)
def _fit_marks(text: str) -> dict[str, Any]:
    own = re.search('[\\[(]\\s*(?:part\\s+)?(' + PART_CODE + ')\\s*[\\])]', text, re.I)
    codes: list[str] = []
    for hit in re.finditer('\\bmodel\\s*(?:no\\.?|number|#)?\\s*[:.]?\\s*(' + FIT_CODE + ')\\b', text, re.I):
        codes += _codes(hit.group(1))
    for hit in re.finditer('\\b(?:also sold as|also known as|sold also as|sold as)\\s+([^.;]+)', text, re.I):
        codes += _codes(hit.group(1))
    works: list[str] = []
    deny: list[str] = []
    for hit in re.finditer('\\b(?<!not )(?<!never )' + FIT_WORDS + '\\s*:?\\s+([^.;]+)', text, re.I):
        if re.search('\\bpart\\s+' + PART_CODE + '\\s+(?:also\\s+)?$', text[max(0, hit.start() - 40):hit.start()], re.I) or re.match('part\\s+series', hit.group(1), re.I):
            continue
        sides = re.split(FIT_EXCEPT, hit.group(1), maxsplit=1, flags=re.I)
        works += _codes(sides[0])
        if len(sides) > 1:
            deny += _codes(sides[1])
    also = [(hit.group(1).upper(), _codes(hit.group(2))) for hit in re.finditer('\\bpart\\s+(' + PART_CODE + ')\\s+also fits\\s*:?\\s*([^.;]+)', text, re.I)]
    series: list[str] = []
    for hit in re.finditer('\\bfits?\\s+part\\s+series\\s*:?\\s*([^.;]+)', text, re.I):
        series += [c.upper() for c in re.findall('\\b' + PART_SERIES + '\\b', hit.group(1))]
    return {'own': own.group(1).upper() if own else '', 'codes': codes, 'works': works, 'deny': deny, 'also': also, 'series': series}

def _fit_verdict(part: dict[str, Any], device: dict[str, Any], allowed: set[str]) -> bool | None:
    """True or False when a part's statements can be held against this device, None when they say nothing about it.
    A part is judged only against a device whose code is of the family its statements name. A part whose own text
    names the device's model code fits it whatever series its own code suggests (a stated code beats an inferred
    series)."""
    stated = [c for c in device.get('codes') or [] if any((_code_names(a, c) for a in allowed))]
    if stated and (not any((_code_names(d, c) for d in part.get('deny') or [] for c in stated))):
        return True
    if device.get('series') and re.match(PART_SERIES + '-', part.get('own') or ''):
        return any(((part['own'] == s or part['own'].startswith(s + '-')) for s in device['series']))
    if not allowed or (not device.get('codes')):
        return None
    families = {_code_family(c) for c in allowed}
    codes = [c for c in device['codes'] if _code_family(c) in families]
    if not codes:
        return None
    if any((_code_names(d, c) for d in part.get('deny') or [] for c in codes)):
        return False
    return any((_code_names(a, c) for a in allowed for c in codes))

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

def _clip_rows(value: Any, limit: int, depth: int=0) -> Any:
    """Like _clip, but the strings of each row in a list of two or more listing rows are cut at ROW_CHARS: used only in
    stores that send listings as text lines, whose search rows carry long descriptions that crowd out the rest."""
    if depth > 7:
        return '...'
    if isinstance(value, list) and len(value) >= 2 and all((isinstance(v, dict) for v in value)):
        return [_clip(v, min(limit, ROW_CHARS), depth + 1) for v in value]
    if isinstance(value, dict):
        return {k: _clip_rows(v, limit, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip_rows(v, limit, depth + 1) for v in value]
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + '...'
    return value

PLAN_AGAIN = '\n\nEvery item needs its own properties: name them for item ITEMS as well, best first, and return the JSON object.'


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

def _stem(text: str) -> frozenset[str]:
    return frozenset((w[:-1] if len(w) > 3 and w.endswith('s') else w for w in _words(text).split()))

def _roots(text: str) -> set[str]:
    out = set()
    for w in _words(text).split():
        w = w[:-2] if re.search('(?:ch|sh|ss|x|z)es$', w) else w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w
        if len(w) >= 3 and w not in ('and', 'for', 'the', 'with'):
            out.add(w)
    return out

def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    if not prop or not key:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)

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
    """A tool argument fitted to its JSON schema: loose types converted, enum spelling fixed, long text cut; what
    cannot fit goes to faults (one line each, named by where)."""
    if not isinstance(schema, dict):
        return value
    kinds = _kinds(schema)
    if kinds and not _rw_fits_any(value, kinds):
        value = _rw_converted(value, kinds)
        if not _rw_fits_any(value, kinds):
            faults.append(f"{where} should be {'/'.join(kinds)}")
            return value
    options = schema.get('enum')
    if isinstance(options, list) and value not in options:
        value = _rw_enum_spelling(value, options)
        if value not in options:
            faults.append(f'{where}: allowed values are {options}')
    if isinstance(value, str):
        return _rw_shape_text(value, schema, where, faults)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        _rw_check_number(value, schema, where, faults)
        return value
    if isinstance(value, list):
        return _rw_shape_list(value, schema, where, faults)
    if isinstance(value, dict):
        return _rw_shape_object(value, schema, where, faults)
    return value


def _rw_fits_any(value: Any, kinds: list[str]) -> bool:
    return any((_fits_kind(value, kind) for kind in kinds))


def _rw_converted(value: Any, kinds: list[str]) -> Any:
    """The value turned into the first schema type it reads as (number text, whole float, true/false text, a single
    value for an array), else unchanged."""
    for kind in kinds:
        if kind == 'integer' and isinstance(value, float) and value.is_integer():
            return int(value)
        if kind in ('integer', 'number') and isinstance(value, str) and re.fullmatch('\\s*-?\\d+(\\.\\d+)?\\s*', value):
            number = float(value)
            return int(number) if kind == 'integer' or number.is_integer() else number
        if kind == 'string' and isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        if kind == 'boolean' and isinstance(value, str) and value.strip().lower() in ('true', 'false'):
            return value.strip().lower() == 'true'
        if kind == 'array' and value is not None and not isinstance(value, list):
            return [value]
    return value


def _rw_enum_spelling(value: Any, options: list[Any]) -> Any:
    if not isinstance(value, str):
        return value
    low = value.strip().lower()
    return next((o for o in options if isinstance(o, str) and o.lower() == low), value)


def _rw_shape_text(value: str, schema: dict[str, Any], where: str, faults: list[str]) -> str:
    longest, shortest = schema.get('maxLength'), schema.get('minLength')
    if isinstance(longest, int) and len(value) > longest:
        cut = value[:longest]
        if longest > 0 and (not value[longest].isspace()) and (not cut[-1].isspace()) and ' ' in cut.strip():
            cut = cut[:cut.rstrip().rfind(' ')]
        value = cut.rstrip(' ,;') or value[:longest]
    if isinstance(shortest, int) and len(value) < shortest:
        faults.append(f'{where} is too short')
    return value


def _rw_check_number(value: Any, schema: dict[str, Any], where: str, faults: list[str]) -> None:
    low, high = schema.get('minimum'), schema.get('maximum')
    if isinstance(low, (int, float)) and value < low:
        faults.append(f'{where} is below {low}')
    if isinstance(high, (int, float)) and value > high:
        faults.append(f'{where} is above {high}')


def _rw_shape_list(value: list[Any], schema: dict[str, Any], where: str, faults: list[str]) -> list[Any]:
    most, least, each = schema.get('maxItems'), schema.get('minItems'), schema.get('items')
    if isinstance(most, int) and len(value) > most:
        faults.append(f'{where} allows at most {most} entries')
    if isinstance(least, int) and len(value) < least:
        faults.append(f'{where} needs at least {least} entries')
    if not isinstance(each, dict):
        return value
    return [_shape(v, each, f'{where}[{i}]', faults) for i, v in enumerate(value)]


def _rw_shape_object(value: dict[str, Any], schema: dict[str, Any], where: str, faults: list[str]) -> dict[str, Any]:
    props = schema.get('properties')
    if not isinstance(props, dict):
        props = {}
    if schema.get('additionalProperties') is False:
        faults.extend((f'{where}.{name} is not an allowed argument' for name in [n for n in value if n not in props]))
    faults.extend((f'{where}.{name} is required' for name in schema.get('required') or [] if name not in value))
    shaped = {}
    for key, item in value.items():
        shaped[key] = _shape(item, props.get(key), f'{where}.{key}', faults) if key in props else item
    return shaped

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

def _pair_rules(text: str) -> list[tuple[str, list[tuple[str, str]]]]:
    """Pairing rules a page states for one property, as (property, [(value, partner), ...]): "our X pairings are
    A with B, C with D", "our rule for X is A goes with B, C with D", "our X pairing rule: A goes
    together with B". A sentence needs two pairs at least."""
    found: list[tuple[str, list[tuple[str, str]]]] = []
    for sentence in re.split('(?<=[.!?])\\s+', str(text or '')):
        key = re.search('\\b(?:rule|pairings?)\\b', sentence, re.I)
        if not key:
            continue
        cut = re.search(':|\\bis that\\b|\\bare\\b|\\bis\\b', sentence[key.start():], re.I)
        if not cut:
            continue
        head = sentence[:key.start() + cut.start()]
        body = sentence[key.start() + cut.end():].strip(' :.')
        named = re.search('\\brule for (.+)$', head, re.I)
        prop = named.group(1) if named else re.sub("^\\s*(?:our|the store'?s|the shop'?s|this shop'?s)\\s+", '', head, flags=re.I)
        prop = re.sub('\\s*\\b(?:pairing rule|pairings? rule|pairings?|rule)(?:\\s+(?:runs|reads|says|goes|works|states))?\\s*$', '', prop.strip(), flags=re.I).strip(' :')
        pairs = []
        for part in re.split(',\\s*(?:and\\s+)?|\\s+and\\s+', body):
            hit = re.match('(.+?)\\s+(?:goes\\s+|go\\s+|pairs\\s+)?(?:together\\s+)?with\\s+(.+)$', part.strip(), re.I)
            if hit:
                pairs.append((hit.group(1).strip(), hit.group(2).strip()))
        if prop and len(prop.split()) <= 5 and len(pairs) >= 2:
            found.append((prop, pairs))
    return found


def _fact_block(node: Any) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    best = None
    for value in node.values():
        if isinstance(value, dict) and len(value) >= 2:
            scalars = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and (not isinstance(v, bool))}
            if len(scalars) >= 2 and (best is None or len(scalars) > len(best)):
                best = scalars
    return best

_LINE_FIELD = re.compile('\\s*([A-Za-z_][A-Za-z0-9_ \\-]{0,30}?)\\s*:\\s(.*)\\Z', re.S)

def _line_value(text: str) -> Any:
    word = text.strip()
    if word in ('True', 'true'):
        return True
    if word in ('False', 'false'):
        return False
    if word in ('None', 'null', ''):
        return None
    if re.fullmatch('-?\\d+', word):
        return int(word)
    if re.fullmatch('-?\\d+\\.\\d+', word):
        return float(word)
    return word

def _line_pairs(text: str) -> dict[str, Any] | None:
    """'a=1; b=two' as {'a': '1', 'b': 'two'} (2+ pairs, most parts holding '='), else None."""
    if '=' not in text or ';' not in text:
        return None
    pairs: dict[str, Any] = {}
    last = None
    loose = 0
    for part in text.split(';'):
        key, sign, value = part.partition('=')
        if sign and key.strip() and len(key.strip()) <= 40:
            last = key.strip()
            pairs[last] = value.strip() if last not in pairs else pairs[last] + ', ' + value.strip()
        elif last is not None:
            pairs[last] += ';' + part
            loose += 1
        else:
            loose += 1
    return pairs if len(pairs) >= 2 and loose * 2 < len(pairs) + loose else None

def _line_record(text: Any) -> dict[str, Any] | None:
    """A listing written as one line of 'key: value | key: value' text, as a dict (3+ fields, one id-like); a field
    holding 'k=v; k=v' becomes a nested attribute block. None for any other text."""
    if not isinstance(text, str) or text.count(' | ') < 2:
        return None
    raw: dict[str, str] = {}
    last = None
    for chunk in text.split(' | '):
        found = _LINE_FIELD.match(chunk)
        if found and found.group(1).strip() and found.group(1).strip() not in raw:
            last = found.group(1).strip()
            raw[last] = found.group(2)
        elif last is None:
            return None
        else:
            raw[last] += ' | ' + chunk
    if len(raw) < 3:
        return None
    record = {key: _line_pairs(value) or _line_value(value) for key, value in raw.items()}
    if any((isinstance(v, str) and _id_like(v, 'id' in k.lower() or 'ref' in k.lower()) for k, v in record.items())):
        return record
    return None

def _unpack(node: Any, depth: int=0) -> Any:
    """An observation with listings written as text lines turned into dicts: lines in a list, and a dict whose only
    value is such a line (that dict becomes the record)."""
    if depth > 6:
        return node
    if isinstance(node, str):
        return _line_record(node) or node
    if isinstance(node, list):
        return [_unpack(v, depth + 1) for v in node]
    if isinstance(node, dict):
        if len(node) == 1:
            only = next(iter(node.values()))
            got = _line_record(only) if isinstance(only, str) else None
            if got:
                return got
        return {k: _unpack(v, depth + 1) for k, v in node.items()}
    return node

class Item:

    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []

class _RwCall:
    """One walker tool call on its way through Kyo.run_calls: the tool, its checked arguments, how many calls came in
    the same turn, the compare-list and single-id argument names, whether it ends the episode, its repeat mark, the
    reading tools at screening time and the [Kyo] lines added to its result."""

    def __init__(self, name: Any, together: int):
        self.name = name
        self.together = together
        self.args: dict[str, Any] = {}
        self.many: str | None = None
        self.target: str | None = None
        self.ending = False
        self.mark = ''
        self.readers: set[Any] = set()
        self.extra = ''

    def group(self) -> Any:
        """The compare-list argument's value (None when the call has none)."""
        return self.args.get(self.many) if self.many else None

class Kyo:

    def __init__(self, problem: dict[str, Any]):
        self.started = time.monotonic()
        environment = problem['environment']
        self.binding = dict(environment['binding'])
        self.view = environment['policy_view']
        self.request = str(self.view.get('query') or '')
        self.tools = [t for t in self.view.get('tools') or [] if self._rw_named_tool(t)]
        self.schemas = {}
        for tool in self.tools:
            params = tool['function'].get('parameters')
            self.schemas[tool['function']['name']] = params if isinstance(params, dict) else {'type': 'object', 'properties': {}}
        self.max_steps = self._rw_step_limit(self.view.get('max_steps'))
        self.finished = bool(self.view.get('finished'))
        self.step = self.waited = self.misses = self.calls = self.loop_calls = 0
        self.broke = self.no_credit = False
        self.cost = 0.0
        self.trace: list[dict[str, Any]] = []
        self.brief, self.roles = {}, {}
        self.items: list[Item] = []
        self.budget: Any = None
        self.msg_limit = self.msg_used = self.reserve = self.gate_msgs = 0
        self.process: list[dict[str, Any]] = []
        self.gate_ok: set[tuple[str, str]] = set()
        self.done_log, self.done_steps = [], []
        self.asked = False
        self.qa: list[tuple[str, str]] = []
        self.rows, self.full = {}, {}
        self.seen_order: list[str] = []
        self.heard: set[str] = set()
        self.notes: list[str] = []
        self.version = 0
        self.cart_touched = self.note_named = False
        self.cart_at = self.cart_seen_at = -1
        self.cart_view: Any = None
        self.cart_lines: dict[str, int] = {}
        self.checked: set[str] = set()
        self.audited = self.note_sold = False
        self.offer_seen, self.gone = {}, {}
        self.unavailable, self.sold_lines = set(), set()
        self.held = 0
        self.id_args: set[tuple[str, str]] = set()
        self.cart_said = ''
        self.refused_at = -1
        self.leads: list[str] = []
        self.refuse_blocks = 0
        self.pushed = self.late_held = self.completed = False
        self.built: list[str] = []
        self.dropped: set[str] = set()
        self.lead_items, self.verdicts = {}, {}
        self.confirmed: str | None = None
        self.id_key: str | None = None
        self.facts_from: dict[str, bool] = {}
        self.queued, self.recent, self.chat = [], [], []
        self.buys = self.refusals = 0
        self.opened: set[str] = set()
        self.refs: list[tuple[int, str]] = []
        self.ref_values: dict[str, str] = {}
        self.trails = self.trail_blocks = self.compare_ok = self.compare_bad = self.pages_read = 0
        self.tool_keys, self.view_keys, self.read_cache, self.shapes = {}, {}, {}, {}
        self.price_key: str | None = None
        self.stock_key: str | None = None
        self.cheaper_marks: set[str] = set()
        self.wider_held = False
        self.paged_obs: list[Any] = []
        self.browse: dict[str, Any] | None = None
        self.browse_misses = self.line_holds = self.fit_holds = self.link_searches = 0
        self.model_busy: dict[str, float] = {}
        self.swept: set[str] = set()
        self.plan_order: list[int] = []
        self.bound: list[tuple[int, int]] = []
        self.searched: list[dict[str, Any]] = []
        self.link_seen: dict[str, list[str]] = {}

    @staticmethod
    def _rw_named_tool(tool: Any) -> bool:
        """A tool entry with a function block that has a name."""
        return isinstance(tool, dict) and isinstance(tool.get('function'), dict) and isinstance(tool['function'].get('name'), str)

    @staticmethod
    def _rw_step_limit(raw: Any) -> int:
        steps = _count(raw)
        if steps is None and isinstance(raw, float):
            steps = int(raw)
        return steps if steps and steps > 0 else 30

    def left(self) -> int:
        return self.max_steps - self.step

    def time_left(self) -> float:
        return EPISODE_SECONDS - (time.monotonic() - self.started)

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
        """One model reply. Providers limit requests per minute for each model, so a limited heavy model hands the call
        to the next of its kin at once, and rests a while; what is left of a limit does not count against the calls.
        Kin answer at low reasoning effort, which keeps a handed-over call as cheap and quick as a normal one, unless the
        call is `deep`."""
        if self.time_left() < EXPLORE_RESERVE + 15:
            models = list(dict.fromkeys((LIGHT if m == HEAVY else m for m in models)))
        for model in models:
            for choice in (HEAVY_KIN if model == HEAVY else (model,)):
                if self.calls >= ALL_CALLS or self.time_left() < CALL_RESERVE:
                    return None
                if time.monotonic() < self.model_busy.get(choice, 0):
                    continue
                self.calls += 1
                tries = 0
                while True:
                    _llm.timeout = max(8, min(50, int(self.time_left() - CALL_RESERVE)))
                    effort = 'low' if model == HEAVY and choice != HEAVY and not deep else None
                    message, status = self.call_model(choice, messages, max_tokens, tools, effort)
                    if message is not None:
                        return message
                    if status == 429 and model == HEAVY:
                        self.model_busy[choice] = time.monotonic() + BUSY_SECONDS
                        self.calls -= 1
                        break
                    passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                    if not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL:
                        break
                    if self.time_left() < WAITS[tries] + CALL_RESERVE + 10:
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
        result = self.post('/inference/chat/completions', body)
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

    @staticmethod
    def post(path: str, body: dict[str, Any]) -> Any:
        return _llm.post_verbose(path, body)

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
        _env.timeout = max(10, min(95, int(self.time_left() - 5)))
        _env.max_retries = 3 if self.time_left() > EXPLORE_RESERVE else 1
        for _ in range(4):
            result = _env.post_verbose('/environment/call', payload)
            if isinstance(result.data, dict):
                data = result.data
                if 'observation' in data:
                    seen = data['observation']
                    data['observation'] = _unpack(seen)
                    if not getattr(self, 'line_rows', False) and isinstance(seen, (dict, list)) and _dump(seen) != _dump(data['observation']):
                        self.line_rows = True
                self.step = data['step'] if isinstance(data.get('step'), int) else self.step + 1
                if data.get('finished'):
                    self.finished = True
                seen = data.get('observation')
                if name in self.page_tools() and (not (isinstance(seen, dict) and seen.get('error'))):
                    for value in args.values():
                        self.opened.update((v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)))
                return data
            error = result.error or {}
            status = error.get('status')
            body = str(error.get('body') or '')
            self.trace.append({'kind': 'env_fail', 'status': status, 'body': body[:200]})
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
        tools = '\n'.join((self._rw_tool_line(t['function']) for t in self.tools))
        prompt = ''.join((BRIEF, '\nTOOLS:\n', tools, '\n\nREQUEST:\n', self.request))
        if self.view.get('opening') is not None:
            prompt += '\n\nOPENING NOTE FROM THE STORE:\n' + _dump(_clip(self.view.get('opening'), 1500))
        brief = self._rw_ask_brief(prompt)
        if brief.get('pick') not in ('cheapest', 'most', 'least') and re.search(LOW_WINS, self.request, re.I):
            brief['pick'] = 'cheapest'
            self.trace.append({'kind': 'pick_read', 'pick': 'cheapest'})
        self.brief = brief
        self.read_browse(brief.get('browse'))
        self._rw_set_roles(brief)
        became = self._rw_set_items(_items(brief.get('items')))
        self._rw_set_bounds(_items(brief.get('fits')), became)
        for lookup in _items(brief.get('lookups')):
            number, text = self._rw_lookup(lookup)
            if text and (number, text) not in self.refs:
                self.refs.append((number, text))
        self.unmix()
        self._rw_set_limits(brief)

    @staticmethod
    def _rw_tool_line(fn: dict[str, Any]) -> str:
        """One store tool as the brief and gate prompts list it."""
        return f"- {fn['name']}: {fn.get('description', '')} | parameters: {_dump(fn.get('parameters') or {})}"

    @staticmethod
    def _rw_full_brief(brief: dict[str, Any]) -> bool:
        """A brief with items and a buy tool among its roles."""
        roles = brief.get('roles')
        return bool(_items(brief.get('items'))) and isinstance(roles, dict) and bool(roles.get('buy'))

    def _rw_ask_brief(self, prompt: str) -> dict[str, Any]:
        """The light model's brief; the heavy model's when that lacks items or a buy tool, or puts two items on one
        shelf (kept only if complete)."""
        brief = self.ask_json(prompt, [LIGHT], 4000) or {}
        if self.no_credit:
            return brief
        if not self._rw_full_brief(brief):
            return self.ask_json(prompt, [HEAVY, LIGHT], 2500) or brief
        if self.shared_shelf(brief):
            again = self.ask_json(prompt, [HEAVY], 2500) or {}
            if self._rw_full_brief(again):
                return again
        return brief

    def _rw_set_roles(self, brief: dict[str, Any]) -> None:
        given = brief.get('roles')
        self.roles = {job: tool for job, tool in (given.items() if isinstance(given, dict) else ()) if isinstance(tool, str) and tool in self.schemas}
        brief.update({key: {} for key in ('buy_args', 'refuse_args') if not isinstance(brief.get(key), dict)})
        buy = self.roles.get('buy')
        if not buy or buy != self.roles.get('refuse'):
            return
        if brief['buy_args'] and brief['refuse_args'] and brief['buy_args'] != brief['refuse_args']:
            return
        ask = f'The store tool {buy} both checks out and ends without a purchase, depending on its arguments. Its schema: ' + _dump(self.schemas.get(buy)) + '\nReturn ONE JSON object and nothing else: {"buy_args": the fixed arguments that check out, "refuse_args": the fixed arguments that end without a purchase} (only the arguments that choose the mode).'
        fixed = self.ask_json(ask, [LIGHT, HEAVY], 1500) or {}
        modes = (fixed.get('buy_args'), fixed.get('refuse_args'))
        if isinstance(modes[0], dict) and isinstance(modes[1], dict) and modes[0] != modes[1]:
            brief['buy_args'], brief['refuse_args'] = modes

    @staticmethod
    def _rw_same_product(kept: Item, item: Item) -> bool:
        """Two brief items for one product: one shelf, one quantity, no property asked with two values."""
        if not item.shelf.strip() or kept.qty != item.qty or _words(kept.shelf) != _words(item.shelf):
            return False
        for a in kept.musts:
            for b in item.musts:
                if _words(a.split(':')[0]) == _words(b.split(':')[0]) and _words(a.partition(':')[2]) != _words(b.partition(':')[2]):
                    return False
        return True

    def _rw_set_items(self, entries: list[Any]) -> dict[int, int]:
        """Reads the brief's items, merges items for one product; returns each brief item's number after merging."""
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            quantity = _count(entry.get('quantity'))
            musts = [str(m) for m in _items(entry.get('musts')) if m]
            self.items.append(Item(str(entry.get('name') or 'the product'), str(entry.get('shelf') or ''), quantity if quantity and quantity > 0 else 1, musts))
        merged: list[Item] = []
        became: dict[int, int] = {}
        for old, item in enumerate(self.items, 1):
            into = next((at for at, kept in enumerate(merged) if self._rw_same_product(kept, item)), None)
            if into is None:
                merged.append(item)
                became[old] = len(merged)
                continue
            twin = merged[into]
            became[old] = into + 1
            twin.musts += [m for m in item.musts if m not in twin.musts]
            self.trace.append({'kind': 'items_merged', 'into': twin.name, 'dropped': item.name})
        self.items = merged or [Item('the requested product', '', 1, [])]
        return became

    def _rw_set_bounds(self, pairs: list[Any], became: dict[int, int]) -> None:
        size = len(self.items)
        for pair in pairs:
            if not isinstance(pair, dict):
                continue
            part, main = (became.get(_count(pair.get(side)) or 0) for side in ('item', 'with'))
            if part and main and part != main and 1 <= part <= size and 1 <= main <= size and (part, main) not in self.bound:
                self.bound.append((part, main))

    def _rw_lookup(self, lookup: Any) -> tuple[int, str]:
        """A requirement whose value sits on another listing, as (item number, the item's own wording of it)."""
        text = str(lookup).strip()
        musts = [(number, must) for number, item in enumerate(self.items, 1) for must in item.musts]
        if all((must != text for _, must in musts)):
            key = _words(text.split(':')[0])
            text = next((must for _, must in musts if _words(must.split(':')[0]) == key), text)
        return (next((number for number, must in musts if must == text), 1), text)

    def _rw_set_limits(self, brief: dict[str, Any]) -> None:
        """Budget, message limit, step limit, process steps and the messages kept for them."""
        budget = brief.get('budget')
        if isinstance(budget, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', budget):
            budget = float(budget)
        if not isinstance(budget, bool) and isinstance(budget, (int, float)) and budget > 0:
            self.budget = budget
        limit = _count(brief.get('messages'))
        if limit is None or limit < 0:
            if self.roles.get('ask'):
                self.msg_limit = 3
        else:
            self.msg_limit = limit
        steps = _count(brief.get('steps'))
        if isinstance(steps, int) and 10 <= steps < self.max_steps:
            self.max_steps = steps
        for entry in _items(brief.get('process')):
            if not isinstance(entry, dict) or not str(entry.get('step') or '').strip():
                continue
            when = str(entry.get('when') or '').strip().lower()
            if when not in ('choose', 'order', 'end'):
                when = 'order'
            self.process.append({'step': str(entry['step']).strip(), 'when': when, 'messages': max(0, _count(entry.get('messages')) or 0)})
        if self.roles.get('ask'):
            talk = max((p['messages'] for p in self.process if p['when'] in ('order', 'end') and p['messages']), default=0)
            self.reserve = min(talk, self.msg_limit) if talk else 0

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
            if item.free:
                text += '; shopper does not mind: ' + ', '.join(item.free)
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
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if lookups and self.buys == 0:
            text += '\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] ' + ' | '.join(lookups[:3])
        if self.leads:
            text += '\n[review candidates] ' + ', '.join(self.leads)
        plan = self.fit_plan() if self.bound and self.buys == 0 else ''
        if plan:
            text += '\n' + plan
        if self.buys == 0:
            text += self.seen_fits()
        if self.notes:
            text += '\n[changes so far] ' + ' | '.join(self.notes[-4:])
        if self.time_left() < EXPLORE_RESERVE + 45:
            text += f'\n[time] about {max(0, int(self.time_left()))} s of this episode are left: settle on your picks and check out within the next few calls.'
        text += '\n[ledger]\n' + self.ledger()
        return text

    def states_needs(self, ident: str, number: int) -> bool:
        """Whether a read listing's attributes state each of the item's requirement values word for word (limits and
        exclusions count as not stated)."""
        block = _fact_block(self.full.get(ident) or {}) or {}

        def words(text: str) -> set[str]:
            return set(_words(re.sub('[,/;|]', ' ', _words(text))).split())
        for owner, need in self.needs_now():
            if owner != number:
                continue
            prop, _, value = need.partition(':')
            want = words(value)
            if not want or any((HEDGE.fullmatch(w) for w in want)):
                return False
            if not any((_names(_stem(prop), _stem(k)) and want <= words(str(v)) for k, v in block.items())):
                return False
        return True

    def seen_fits(self) -> str:
        """Per item still without a cart line: read buyable listings whose attributes state every requirement."""
        cat_key = self.category_key()
        if not cat_key:
            return ''
        held = self.cart_ids()
        parts = []
        for number, item in enumerate(self.items, 1):
            if any((self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or '')) for i in held)):
                continue
            fits = [i for i in self.item_ids(number) if i in self.full and self.buyable(i) and self.states_needs(i, number)]
            if fits:
                parts.append(f'item {number} ({item.name}): ' + ', '.join(fits[:3]))
        if not parts:
            return ''
        return '\n[fits seen, not carted: their attributes state every requirement word for word; cart one per item now (cheapest pick: compare prices first)] ' + ' | '.join(parts)

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
        if shape['next'] and (not shape['next_arg']):
            value = observation[shape['next']]
            picks = {self.page_arg(name, 1), self.page_arg(name, 'x')} - {None} if value is None else {self.page_arg(name, value)} - {None}
            shape['next_arg'] = picks.pop() if len(picks) == 1 else None
            if shape['next_arg']:
                self.trace.append({'kind': 'page_arg', 'tool': name, 'arg': shape['next_arg']})
        self.shapes[name] = shape
        self.price_key = self.price_key or shape['price']
        self.stock_key = self.stock_key or shape['stock']
        self.trace.append({'kind': 'shape', 'tool': name, 'shape': shape})
        return shape

    def page_arg(self, name: str, value: Any) -> str | None:
        """The argument a next-page value goes back into, from the tool's schema when the shape reader named none:
        the optional, non-enum arguments of the value's type; a paging word in the name decides among them, else the
        offset rule for whole numbers. Only a single survivor counts (ambiguity keeps the reader's answer)."""
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        need = {'integer', 'number'} if isinstance(value, int) and (not isinstance(value, bool)) else ({'string'} if isinstance(value, str) and value.strip() else set())
        if not need:
            return None
        loose = [k for k, v in props.items() if isinstance(v, dict) and need & set(_kinds(v)) and ('enum' not in v) and k not in (schema.get('required') or [])]
        named = [k for k in loose if re.search('page|cursor|token|offset|start|after|next', k, re.I)]
        if len(named) == 1:
            return named[0]
        if isinstance(value, int) and (not named):
            return self.offset_arg(name)
        return None

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

    def long(self) -> bool:
        """Whether the request allows a long run (LONG_STEPS steps or more). Such runs end on the 300 s clock with half
        their steps unused, while a store call costs almost no time next to a walker turn, so Kyo spends steps on
        reading for the walker: a first look at every item, the likeliest rows of later searches, more pages."""
        return self.max_steps >= LONG_STEPS

    def long_reads(self, name: str, observation: Any, browsing: bool) -> str:
        """On a long run, after a walker search: the first time, each item's category listing (or a search for the
        item) and its FIRST_LOOK likeliest listings read in full; later, up to 2 likeliest unread rows of the search
        for items with no cart line and fewer than LONG_READS listings read. A third of the steps stay untouched."""
        if not browsing or not self.long() or self.finished or self.time_left() < EXPLORE_RESERVE + 40:
            return ''
        found = [i for i, _ in self.records_in(observation)]
        cat_key = self.category_key()
        if not found or not cat_key:
            return ''
        keep = self.tail() + self.max_steps // 3
        if self.left() - keep < 3:
            return ''
        held = {self.owner_of(str((self.rows.get(i) or {}).get(cat_key) or '')) for i in self.cart_ids()}
        texts: list[str] = []
        if not getattr(self, 'first_looked', False):
            self.first_looked = True
            plan = self.search_template()
            for number, item in enumerate(self.items, 1):
                if self.left() - keep < 3 or self.time_left() < EXPLORE_RESERVE + 40 or self.finished:
                    break
                rows = list(found)
                if self.records_of_item(number) < 2:
                    if self.browse and self.browse_misses < 2:
                        rows += self.list_category(number, None) or []
                    if plan and (not self.kind_rows(number, rows)) and self.left() - keep >= 3:
                        rows += self.search_words(plan[0], plan[1], plan[2], item.name)
                want = max(0, FIRST_LOOK - self.records_of_item(number))
                picks = [i for i in self.kind_rows(number, rows) if i not in self.full][:want]
                opened = self.fill_facts(picks, keep)
                if opened:
                    texts.append(f'item {number} ({item.name}): ' + ' || '.join((self.read_line(i, number) for i in picks[:len(opened)])))
            head = "[Kyo's first look: this request allows a long run, so Kyo listed or searched every item and read its likeliest listings (one step each); per listing: price, title and its attributes for the requirements]"
        else:
            numbers = sorted({self.owner_of(str((self.rows.get(i) or {}).get(cat_key) or '')) for i in found} - held - {0})
            for number in numbers:
                if texts or self.records_of_item(number) >= LONG_READS:
                    continue
                picks = [i for i in self.kind_rows(number, found) if i not in self.full][:1]
                texts += [self.read_line(i, number) for i in picks[:len(self.fill_facts(picks, keep))]]
            head = '[Kyo read the likeliest unread row of this search for an item with no cart line yet (one step): price, title and its attributes for the requirements]'
        if not texts:
            return ''
        self.trace.append({'kind': 'long_reads', 'n': len(texts)})
        return '\n' + head + '\n' + '\n'.join(texts)

    def read_line(self, ident: str, number: int) -> str:
        """One short line for a listing Kyo read for the walker (the whole record stays in Kyo's ledger): price, title
        and the attributes whose key names one of the item's requirement properties (only what the listing states),
        else its first attributes."""
        record = self.full.get(ident) or self.rows.get(ident) or {}
        block = _fact_block(record) or {}
        title = next((str(v) for k, v in record.items() if isinstance(v, str) and re.search('title|name', str(k), re.I)), '')[:70]
        price = self.price_seen(ident)
        parts: list[str] = []
        for n, need in self.needs_now():
            if n == number and ':' in need:
                prop = need.split(':', 1)[0].strip()
                parts += [f'{k}={str(v)[:40]}' for k, v in block.items() if _names(_stem(prop), _stem(k)) and f'{k}={str(v)[:40]}' not in parts][:2]
        if not parts:
            parts = [f'{k}={str(v)[:30]}' for k, v in list(block.items())[:5]]
        head = ident + (f' ({price:g})' if price is not None else '') + ('' if self.buyable(ident) else ' SOLD OUT')
        return f'{head} {title}' + (' | ' + '; '.join(parts[:6]) if parts else '')

    def states(self, ident: str, need: str) -> bool:
        """Whether a read listing states a requirement under a key naming its property, as written: the value's words
        as one run, or a figure inside a stated limit in the same unit. Exclusions and alternatives are not proven."""
        prop, _, value = need.partition(':')
        value = ' '.join(re.findall('[a-z0-9.]+', value.lower()))
        record = self.full.get(ident) or {}
        if not prop.strip() or not value or re.search('\\b(?:not|no|other than|except|unlike|between|or|any)\\b', value):
            return False
        pairs = list((_fact_block(record) or {}).items()) + list(record.items())
        hits = [' '.join(re.findall('[a-z0-9.]+', str(v).lower())) for k, v in pairs if not isinstance(v, (dict, list)) and _names(_stem(prop), _stem(k))]
        if not hits:
            return False
        bound = re.search('\\b(at most|at least|up to|under|over|below|above|max|min|maximum|minimum)\\b', value)
        if bound:
            nums = re.findall('\\d+(?:\\.\\d+)?', value)
            if len(nums) != 1:
                return False
            unit = value.split(nums[0], 1)[1].split()[:1]
            low = bound.group(1) in ('at least', 'over', 'above', 'min', 'minimum')
            for hit in hits:
                got = re.findall('\\d+(?:\\.\\d+)?', hit)
                if len(got) == 1 and (not unit or unit[0] in hit.split()):
                    if (float(got[0]) >= float(nums[0])) if low else (float(got[0]) <= float(nums[0])):
                        return True
            return False
        return any((re.search('(?<![a-z0-9.])' + re.escape(value) + '(?![a-z0-9.])', hit) for hit in hits))

    def price_seen(self, ident: str) -> float | None:
        """The listing's price: by the learned price key, else from its own row's single price-named number."""
        known = self.price_of(ident)
        if known is not None:
            return known
        row = {**(self.rows.get(ident) or {}), **(self.full.get(ident) or {})}
        found = [v for k, v in row.items() if re.search('price|cost', str(k), re.I) and isinstance(v, (int, float)) and (not isinstance(v, bool))]
        return float(found[0]) if len(found) == 1 else None

    def proven_fill(self) -> None:
        """At the forced end of a long run whose cart lacks some requested item: each missing item gets the cheapest
        read, buyable listing of its kind that states every one of its requirements as written (no requirement taken
        from another listing, no fit between items), within the budget. If any missing item has no such listing, the
        cart stays as it is."""
        cart = self.roles.get('cart')
        cat_key = self.category_key()
        if not self.long() or not cart or not cat_key or getattr(self, 'filled', False) or self.left() < 3 or self.finished:
            return
        self.filled = True
        now = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        held = {self.owner_of(str((self.rows.get(i) or {}).get(cat_key) or '')) for i in now}
        missing = [n for n in range(1, len(self.items) + 1) if n not in held]
        tied = {p for p, _ in self.bound} | {m for _, m in self.bound}
        pointed = {need for _, need in self.refs}
        if not missing or tied & set(missing):
            return
        plan: dict[str, int] = {}
        for number in missing:
            needs = [need for n, need in self.needs_now() if n == number and (not self.described(number, need))]
            if not needs or any((need in pointed for need in needs)):
                return
            ready = [i for i in self.item_ids(number) if i in self.full and i not in now and self.buyable(i) and all((self.states(i, need) for need in needs))]
            if not ready:
                return
            plan[min(ready, key=lambda i: self.price_seen(i) if self.price_seen(i) is not None else float('inf'))] = self.items[number - 1].qty
        lines = {**{i: self.cart_lines.get(i, 1) for i in now}, **plan}
        prices = {i: self.price_seen(i) for i in lines}
        if isinstance(self.budget, (int, float)):
            if None in prices.values():
                return
            over = any((prices[i] > self.budget for i in lines)) if self.brief.get('budget_scope') == 'each' else sum((prices[i] * n for i, n in lines.items())) > self.budget
            if over:
                return
        calls = self.cart_calls(lines, [])
        if not calls or len(calls) + 2 > self.left():
            return
        done = []
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), 'args', faults)
            if faults:
                break
            reply = self.act(cart, args)
            if reply is None or self.finished:
                break
            self.note_cart(cart, args)
            self.observe(cart, reply.get('observation'))
            done.append(_dump(args))
        self.trace.append({'kind': 'proven_fill', 'plan': plan, 'calls': done})

    def show_lines(self) -> int:
        """Right before a checkout Kyo makes itself: each cart line whose offer page was never shown is opened with the
        offer tool (the store counts a page the order relies on only when it was viewed). A store call takes well
        under a second, so only the steps and the last seconds limit it."""
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not tool or not target or self.finished:
            return 0
        never = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0 and i not in self.offer_seen and i not in self.gone]
        shown = 0
        for ident in never:
            if self.left() < 2 or self.time_left() < CALL_RESERVE + 3 or self.finished:
                break
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
            if faults:
                break
            reply = self.act(tool, args)
            if reply is None:
                break
            self.offer_seen[ident] = self.step
            self.observe(tool, reply.get('observation'))
            shown += 1
        if never:
            self.trace.append({'kind': 'show_lines', 'never': never, 'shown': shown})
        return shown

    def code_rows(self, name: str, observation: Any, browsing: bool) -> str:
        """The request names the shopper's own device or vehicle by a model or fit code: after a search, the rows whose
        text names that code (they fit it) are listed for the walker, and the cheapest unread one of an item with no
        cart line is read in full (one step, while steps allow)."""
        if not browsing or self.finished:
            return ''
        asked = _fit_marks(self.request)
        wanted = set(asked.get('codes') or []) | set(asked.get('works') or [])
        found = [i for i, _ in self.records_in(observation)]
        if not wanted or not found:
            return ''
        index = self.fit_index()
        hits = [i for i in found if set((index['marks'].get(i) or {}).get('works') or []) & wanted or set((index['marks'].get(i) or {}).get('codes') or []) & wanted]
        if not hits:
            return ''
        cat_key = self.category_key()
        held = {self.owner_of(str((self.rows.get(i) or {}).get(cat_key) or '')) for i in self.cart_ids()} if cat_key else set()

        def title(ident: str) -> str:
            row = self.rows.get(ident) or {}
            return next((str(v) for k, v in row.items() if isinstance(v, str) and re.search('title|name', str(k), re.I)), '')[:50]
        named = ', '.join((i + (f' ({self.price_seen(i):g})' if self.price_seen(i) is not None else '') + ' ' + title(i) for i in hits[:6]))
        text = f"\n[Kyo] rows of this search whose text names the code the request gives ({', '.join(sorted(c.upper() for c in wanted))[:40]}): {named}"
        fresh = [i for i in hits if i not in self.full and self.buyable(i) and (not cat_key or self.owner_of(str((self.rows.get(i) or {}).get(cat_key) or '')) not in held)]
        fresh.sort(key=lambda i: self.price_seen(i) if self.price_seen(i) is not None else float('inf'))
        if fresh and self.left() > self.tail() + 8 and self.time_left() > EXPLORE_RESERVE + 30:
            read = self.fill_facts(fresh[:1], self.tail() + 8)
            if read:
                number = self.owner_of(str((self.rows.get(fresh[0]) or {}).get(cat_key) or '')) if cat_key else 0
                text += '\n[Kyo read the cheapest of them for an item with no cart line yet (one step)] ' + (self.read_line(fresh[0], number) if number else read[0][:300])
        self.trace.append({'kind': 'code_rows', 'hits': hits[:6]})
        return text

    def price_recheck(self) -> str:
        """After a note that a price moved (often with a refused checkout): the cart's lines are read again on their
        offer pages (one step each) and the walker gets the new prices against the budget. A price change alone is no
        reason to give up a line that still meets the requirements and keeps the cart within the budget."""
        tool = self.offer_tool()
        target = self.single_target(tool)
        lines = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0 and i not in self.gone]
        if not tool or not target or not lines or self.left() < len(lines) + 3 or self.finished:
            return ''
        for ident in lines:
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
            if faults:
                return ''
            reply = self.act(tool, args)
            if reply is None or self.finished:
                return ''
            self.offer_seen[ident] = self.step
            self.observe(tool, reply.get('observation'))
        prices = {i: self.price_seen(i) for i in lines}
        units = {i: self.held_units(i) or 1 for i in lines}
        shown = ', '.join((f'{i} now {prices[i]:g}' if prices[i] is not None else f'{i} (price not shown)' for i in lines))
        text = "\n[Kyo re-read the cart's lines after the price change] " + shown
        if isinstance(self.budget, (int, float)) and None not in prices.values():
            each = self.brief.get('budget_scope') == 'each'
            total = sum((prices[i] * units[i] for i in lines))
            over = any((prices[i] > self.budget for i in lines)) if each else total > self.budget
            text += (f'; total {total:g} against the budget of {self.budget:g}' if not each else f'; limit {self.budget:g} for each item')
            if over:
                text += '. The cart is now over the budget: replace a line with one that still meets its requirements and fits, then check out.'
            else:
                text += '. The cart still fits the budget, and a price change alone is no reason to switch: keep the lines that still meet the requirements in force and check out again.'
        self.trace.append({'kind': 'price_recheck', 'lines': lines, 'prices': prices})
        return text

    def page_allowance(self) -> int:
        """Pages Kyo reads on its own: none when the request allows fewer than TIGHT_STEPS steps for two or more
        items, where every step is wanted for searching each item and reading its listings."""
        if self.long():
            return LONG_PAGES
        return AUTO_PAGES if self.max_steps >= TIGHT_STEPS or len(self.items) < 2 else 0

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
        per = 3 if self.long() else 2
        if pages > per and self.pages_read + per <= self.page_allowance() and (self.left() - self.tail() - per >= 12):
            pages = per
        if pages > per or self.pages_read + pages > self.page_allowance() or self.left() - self.tail() - pages < 12:
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
        return f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest. Searches match a listing's title, so a requirement's value (often only in the attributes) finds few of the fitting listings, and rows come in no price order. Before checking out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line" + (f" ({'; '.join(lines)})" if lines else '') + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'

    def other_values(self, number: int) -> list[str]:
        """Values listings of this item show for properties the request leaves open, taken in turn from its most carried
        keys: searched together with the required values, they bring up rows the plain search does not reach."""
        table = self.key_table()
        keys = self.lane_keys(number, table)[:3]
        said = ' '.join(self.value_words(number)).lower() + ' ' + self.items[number - 1].name.lower()
        lanes = []
        for key in keys:
            values = next((v for label, name, _, v in table if name == key and (len(self.items) == 1 or self.owner_of(label) == number)), [])
            lanes.append([v for v in values if v and len(v) <= 20 and v not in said and re.search('[a-z0-9]', v)][:2])
        return [lane[k] for k in range(2) for lane in lanes if k < len(lane)]

    def cheaper_sweep(self) -> bool:
        """For a cheapest pick: bring up rows priced below each cart line, for the reading and the check before checkout.
        Lists the line's category under its price where the store can, else searches the required values with other values
        listings show. Steps for reading, the cart changes and the checkout stay free. Returns whether it could look."""
        cat_key = self.category_key()
        plan = self.search_template()
        mark = self.cart_mark() + f'#{self.version}'
        room = self.left() - self.tail() - len(self.items) - 3
        if mark in self.swept or room < 1 or self.time_left() < EXPLORE_RESERVE + 25 or (not cat_key):
            return False
        self.swept.add(mark)
        limit = self.step + min(SWEEP_STEPS, room)
        ran = False
        cheaper: list[str] = []
        for line in self.cart_ids():
            price = self.price_of(line)
            label = str((self.rows.get(line) or {}).get(cat_key) or '')
            number = next((n for n, item in enumerate(self.items, 1) if label and self.owns(item, label)), 0)
            if price is None or not number or self.step >= limit:
                continue
            found = self.list_category(number, int(price) - 1 if price > 1 else price * 0.9, 1 if limit - self.step >= 2 else 0)
            if found is not None:
                ran = True
            elif plan:
                base = self.value_words(number)[:2]
                for other in self.other_values(number)[:3]:
                    if self.step >= limit:
                        break
                    found = (found or []) + self.search_words(plan[0], plan[1], plan[2], ' '.join(base + [other, self.items[number - 1].name]))
                    ran = True
            cheaper += [i for i in self.kind_rows(number, found or [], price) if i != line and i not in self.full]
        self.trace.append({'kind': 'sweep', 'ran': ran, 'cheaper': cheaper[:6]})
        return ran

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return []
        texts = []
        for ident in [i for i in ids if i in self.rows and i not in self.full and (not self.sold_out(i))]:
            if self.left() <= keep or self.finished or self.facts_from.get(tool) is False or self.time_left() < CALL_RESERVE + 10:
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

    def cheaper_rows(self) -> list[str]:
        if self.brief.get('pick') != 'cheapest':
            return []
        mark = self.cart_mark() + f'#{self.version}'
        cat_key = self.category_key()
        if mark in self.cheaper_marks or not cat_key:
            return []
        self.cheaper_marks.add(mark)
        cart = self.cart_ids()
        picks: list[str] = []
        for line in cart:
            price = self.price_of(line)
            kind = str((self.rows.get(line) or {}).get(cat_key) or '').strip().lower()
            if price is None or not kind:
                continue
            pool = [i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i) and (str((self.rows.get(i) or {}).get(cat_key) or '').strip().lower() == kind) and ((self.price_of(i) if self.price_of(i) is not None else float('inf')) < price)]
            pool.sort(key=lambda i: self.price_of(i) or 0.0)
            extra = 2 if self.long() else 0
            picks += [i for i in pool[:CHEAPER_OPENS + extra if len(cart) == 1 else 2 + extra // 2] if i not in picks]
        return self.fill_facts(picks[:CHEAPER_OPENS + (2 if self.long() else 0)], self.tail() + 2)

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
        if self.left() <= 8 or self.time_left() < EXPLORE_RESERVE + 40:
            return False
        opened = len(self.full)
        if opened >= 3 * len(self.items) + 1:
            return True
        return self.step >= max(8, self.max_steps // 4)

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

    def early_chains(self) -> str:
        """Once, right after the interview: Kyo follows the pages that requirements take their value from (up to
        EARLY_ROUNDS rounds of the chain check: find the pointed listing, open it and every listing its page names, read
        the value), so the walker shops with those values instead of meeting the chain at checkout. Skipped with 12
        or fewer steps left, too little time, or a request allowing fewer than EARLY_STEPS steps (there the steps and
        seconds it takes cost as much as it saves); the checkout's own chain rounds and link searches stay as they were."""
        if getattr(self, 'early_done', False) or self.finished:
            return ''
        self.early_done = True
        live = {need for _, need in self.needs_now()}
        if not any((need in live and need not in self.ref_values for _, need in self.refs)):
            return ''
        if self.left() <= 12 or self.time_left() < EXPLORE_RESERVE + 60 or self.max_steps < EARLY_STEPS:
            return ''
        kept = (self.trails, self.trail_blocks, self.link_searches)
        self.trails = max(0, (6 if self.left() > 15 else 4) - EARLY_ROUNDS)
        text, _ = self.trail()
        self.trails, self.trail_blocks, self.link_searches = kept
        self.trace.append({'kind': 'early_chains', 'values': dict(self.ref_values)})
        if not text:
            return ''
        return "[Kyo followed the pages the requirements point at before shopping]\n" + text.replace(', then check out.', '.') + '\n'

    def interview(self) -> str:
        """Engine M: the first question names M_FIRST properties, later ones a list of M_LIST (M_ONE on a one-item task)
        for the next product while the shopper has never refused; any refusal turns the rest of the episode to single
        questions, the refused list's first property asked alone next. A product whose reply is confused (it holds no
        need) is left; with M_ROTATE a list that drew nothing is re-asked in reverse order with new keys; the same text
        is never sent twice."""
        self.asked = True
        if self.msgs_free() <= 0:
            return ''
        if self.time_left() > EXPLORE_RESERVE + 30:
            self.widen()
        lanes = self.plan_lanes(self.msgs_free())
        found = {n: 0 for n in lanes}
        spent = {n: 0 for n in lanes}
        used: dict[int, set[str]] = {n: set() for n in lanes}
        follow: list[tuple[int, list[str]]] = []
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        closed: set[int] = set()
        texts: set[str] = set()
        lists_ok = True
        while self.msgs_free() > 0 and (not self.finished) and (self.left() > 6) and (self.time_left() > EXPLORE_RESERVE + 20):
            if follow:
                number, labels = follow.pop(0)
            else:
                size = 1
                if lists_ok:
                    size = M_FIRST if not sent else (M_ONE if len(self.items) == 1 else M_LIST)
                    if self.msgs_free() == 1:
                        size = max(size, M_LAST)
                number, labels = self.next_ask(lanes, found, spent, used, closed, size)
                if not labels:
                    break
            labels = self.fit_labels(number, labels)
            text = self.question(number, labels[0]) if len(labels) == 1 else self.list_question(number, labels)
            used[number].update((_words(label) for label in labels))
            if text in texts:
                continue
            batch = self.send([(number, labels[0], text)])
            if not batch:
                break
            texts.add(text)
            sent += batch
            spent[number] += 1
            had = [len(item.told) for item in self.items]
            heard = self.hear(batch, (LIGHT, HEAVY))
            if self.judges()[:1] == [HEAVY] and (any((self.doubtful(entry) for entry in _items(heard.get('needs')))) or self.misheard(heard, labels, batch[0][2])):
                heard = self.hear(batch, (HEAVY,)) or heard
            heard = self.whole_pointers(heard, batch[0][2])
            self.take_heard(heard, told)
            fresh = [n for n, item in enumerate(self.items, 1) if len(item.told) > had[n - 1]]
            for n in fresh:
                found[n] = found.get(n, 0) + 1
            if fresh:
                continue
            reply = batch[0][2]
            if 1 in [_count(x) for x in _items(heard.get('crowded'))] or re.search(M_REFUSED, reply, re.I):
                lists_ok = False
                self.trace.append({'kind': 'm_refused', 'item': number, 'labels': labels})
                if len(labels) > 1:
                    for label in labels[1:]:
                        used[number].discard(_words(label))
                    follow.insert(0, (number, labels[:1]))
                continue
            free = [entry for entry in _items(heard.get('free')) if isinstance(entry, dict)]
            if not free and not re.search(M_NO_VIEW, reply, re.I):
                closed.add(number)
                self.trace.append({'kind': 'm_confused', 'item': number})
                continue
            if len(labels) > 1 and lists_ok and M_ROTATE:
                rest = labels[1:][::-1]
                for label in rest:
                    used[number].discard(_words(label))
                lanes[number] = rest + [x for x in lanes[number] if x not in rest]
        found_items = [n for n in found if found[n] > 0]
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

    def ask_room(self) -> int:
        """The longest text the message tool takes (its text field's maxLength), 0 when it states none."""
        props = (self.schemas.get(self.roles.get('ask') or '') or {}).get('properties') or {}
        sizes = [v['maxLength'] for v in props.values() if isinstance(v, dict) and isinstance(v.get('maxLength'), int) and v['maxLength'] > 0]
        return min(sizes) if sizes else 0

    def fit_labels(self, number: int, labels: list[str]) -> list[str]:
        """A list cut from its end until its question fits the message tool's length limit (one label at least); the
        labels cut stay unasked."""
        room = self.ask_room()
        while room and len(labels) > 1 and len(self.list_question(number, labels)) > room:
            labels = labels[:-1]
        return labels

    def list_question(self, number: int, labels: list[str]) -> str:
        named = labels[0] if len(labels) == 1 else ', '.join(labels[:-1]) + ' or ' + labels[-1]
        return f'What {named} would you like for the {self.items[number - 1].name}?'

    def search_template(self) -> tuple[str, dict[str, Any], str] | None:
        tool = self.roles.get('search')
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        named = [k for k in texts if re.search('(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword', k, re.I)]
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
                        used = [_dump(a[k]) for a in worded if k in a]
                        if used:
                            best = max(used, key=used.count)
                            keep[k] = next((a[k] for a in worded if k in a and _dump(a[k]) == best))
                return (tool, keep, field)
        return None

    def value_words(self, number: int) -> list[str]:
        out = []
        for n, need in self.needs_now():
            if n != number or need in [m for _, m in self.refs] or ':' not in need:
                continue
            value = need.split(':', 1)[1].strip()
            if not value or re.search('\\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', value, re.I):
                continue
            out.append(value)
        return out

    def value_search(self, numbers: list[int]) -> str:
        plan = self.search_template()
        if not plan or self.left() < 22 or self.time_left() < EXPLORE_RESERVE + 25:
            return ''
        tool, base, field = plan
        seen: list[str] = []
        budget = 8
        for number in numbers:
            values = self.value_words(number)
            item = self.items[number - 1]
            fresh = [v for v in values if any((':' in t and t.split(':', 1)[1].strip() == v for t in item.told))]
            if not fresh or budget <= 2:
                continue
            stated = [i for i, v in enumerate(values) if v not in fresh]
            drop = stated[-1] if stated else 0
            tries = [values, values[:drop] + values[drop + 1:]] if len(values) > 1 else [values]
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

    def next_ask(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]], closed: set[int] | None=None, size: int=1) -> tuple[int, list[str]]:
        """The next product and its next `size` labels: products with no need heard yet first, then the least asked,
        then the planner's story order; products whose reply was confused only when no other product has labels."""
        first = {n: i for i, n in enumerate(self.plan_order)}
        if len(self.items) >= 3 and getattr(self, 'lane_count', 9) <= 2:
            first = {n: (len(self.items[n - 1].musts), n) for n in lanes}
        order = sorted(lanes, key=lambda n: (found.get(n, 0) > 0, spent.get(n, 0), first.get(n, len(first) + n)))
        shut = closed or set()
        for pool in ([n for n in order if n not in shut], [n for n in order if n in shut]):
            for number in pool:
                left = [label for label in lanes[number] if _words(label) not in used[number]]
                if left:
                    return (number, left[:max(1, size)])
        return (0, [])

    def lane_keys(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        item = self.items[number - 1]
        settled = [_stem(m.split(':', 1)[0]) for m in item.musts]
        rows = [(label, key, carried, len(values)) for label, key, carried, values in table if self.owner_of(label) == number]
        measured = {key: sum((1 for v in values if re.match('^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$', v))) * 2 > len(values) for _, key, _, values in table}
        if len(self.items) == 1 and (not rows):
            rows = [(label, key, carried, len(values)) for label, key, carried, values in table]
        rows.sort(key=lambda r: (-r[2], measured.get(r[1], False), -r[3]))
        keys = [key for _, key, _, _ in rows if not any((_names(prop, _stem(key)) for prop in settled)) and (not re.search(NOT_PROPERTY, key))]
        return [key for key in keys if not re.search(CATCH_ALL, key)] + [key for key in keys if re.search(CATCH_ALL, key)]

    def owner_of(self, label: str) -> int:
        if not label:
            return 0
        words = _words(label)
        long = _roots(label)
        flat = re.sub('[^a-z0-9]', '', label.lower())
        best, top = (0, 0.0)
        for number, item in enumerate(self.items, 1):
            if words == _words(item.shelf or '') or (flat and flat == re.sub('[^a-z0-9]', '', (item.shelf or '').lower())):
                score = 3.0
            elif words == _words(item.name) or (flat and flat == re.sub('[^a-z0-9]', '', item.name.lower())):
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
        self.lane_count = count
        table = self.key_table()
        prompt = self._rw_plan_prompt(self.picks_wanted(count, table), table)
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 3500, deep=True) or {}
        lanes, hinted = self._rw_plan_picks(_items(plan.get('ask')))
        stock = self.brief.get('in_stock_only', True) is not False
        for number in lanes:
            lanes[number] = [label for label in lanes[number] if not (stock and re.search(STOCK_PICK, label, re.I))]
        bare = list(lanes) if not _items(plan.get('ask')) else [n for n in lanes if not lanes[n] and (not self.lane_keys(n, table))]
        if not bare and sum((len(v) for v in lanes.values())) + len(table) < (M_FIRST + (count - 2) * M_LIST + 1 if count > 1 else 1):
            bare = list(lanes)
        if bare and (not self.broke) and (not self.no_credit) and (self.time_left() > EXPLORE_RESERVE + 40):
            again = self.ask_json(prompt + PLAN_AGAIN.replace('ITEMS', ', '.join((str(n) for n in bare))), [HEAVY, LIGHT], 3500, deep=True) or {}
            more, seen = self._rw_plan_picks(_items(again.get('ask')))
            for number in bare:
                for label in more.get(number) or []:
                    if not (stock and re.search(STOCK_PICK, label, re.I)) and all((_words(label) != _words(other) for other in lanes[number])):
                        lanes[number].append(label)
                hinted[number] |= seen.get(number) or set()
        for number in lanes:
            lanes[number] = self._rw_lane(number, table, lanes[number], hinted[number], count)
        self.trace.append({'kind': 'plan', 'ask': _clip(plan.get('ask'), 120), 'lanes': {n: v[:8] for n, v in lanes.items()}})
        return lanes

    def picks_wanted(self, count: int, table: list[tuple[str, str, int, list[str]]]) -> int:
        """How many properties per product the planner names: one per free message, or, when those picks and every
        key the listings read gave cannot reach the last free message (first question, middle lists, then at least
        one label), as many as the free messages carry, so the questions do not run dry."""
        if count <= 1:
            return count
        reach = M_FIRST + (count - 2) * M_LIST + 1
        return M_FIRST + (count - 2) * M_LIST + M_LAST if len(self.items) * count + len(table) < reach else count

    def _rw_plan_prompt(self, count: int, table: list[tuple[str, str, int, list[str]]]) -> str:
        settled = [f'item {number}: {item.name} - settled by the request: ' + ('; '.join(item.musts) or 'nothing') for number, item in enumerate(self.items, 1)]
        listing: list[str] = []
        for index, row in enumerate(table, 1):
            label, name, carried, values = row
            if index == 1 or table[index - 2][0] != label:
                listing.append(f'category {label!r}:')
            tag = ' [several features in one: ask last]' if re.search(CATCH_ALL, name) else ''
            listing.append(f'  {index}. {name} (on {carried}): ' + ' | '.join(values) + self.not_stated(label, name, values) + tag)
        parts = [PLAN.replace('COUNT', str(max(1, count))), '\n\nREQUEST:\n', self.request, '\n\nITEMS:\n', '\n'.join(settled)]
        first = [str(x) for x in _items(self.brief.get('ask_first')) if str(x).strip()]
        if first:
            parts += ['\n\nASK FIRST (the request asks to be asked about these):\n', '\n'.join(first)]
        parts += ['\n\nATTRIBUTE KEYS SEEN:\n', '\n'.join(listing) or '(no full listing records yet)']
        return ''.join(parts)

    @staticmethod
    def _rw_plan_label(raw: Any) -> str:
        """The first property a plan pick names (before 'and'/'or'/punctuation), '' when none or too long."""
        label = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(raw or ''))[0].strip().rstrip('?.')
        return label if label and len(label.split()) <= 5 else ''

    def _rw_plan_picks(self, picks: list[Any]) -> tuple[dict[int, list[str]], dict[int, set[str]]]:
        """The planner's questions per item (in its order, one per property) and the properties it saw hinted."""
        lanes: dict[int, list[str]] = {n: [] for n in range(1, len(self.items) + 1)}
        hinted: dict[int, set[str]] = {n: set() for n in lanes}
        for pick in picks:
            if not isinstance(pick, dict):
                continue
            number = min(max(1, _count(pick.get('item') or 1) or 1), len(self.items))
            label = self._rw_plan_label(pick.get('label'))
            if not label:
                continue
            if not any((_words(label) == _words(other) for other in lanes[number])):
                lanes[number].append(label)
            if not str(pick.get('hint') or '').strip():
                continue
            hinted[number].add(_words(label))
            if number not in self.plan_order:
                self.plan_order.append(number)
        return (lanes, hinted)

    @staticmethod
    def _rw_plain(text: str) -> str:
        return re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', _words(text)))

    def _rw_spelled(self, own: dict[str, float], label: str) -> str:
        """The label as the listings spell the key (plural and colour/color forms aside)."""
        wanted = self._rw_plain(label)
        return next((k for k in own if self._rw_plain(k) == wanted), label)

    @staticmethod
    def _rw_share(own: dict[str, float], label: str) -> float:
        """The share of the item's read listings that carry the property (a close key name counts)."""
        if _words(label) in own:
            return own[_words(label)]
        return max((s for k, s in own.items() if _names(_stem(label), _stem(k)) or _names(_stem(k), _stem(label))), default=0.0)

    def _rw_lane(self, number: int, table: list[tuple[str, str, int, list[str]]], picks: list[str], hints: set[str], count: int) -> list[str]:
        """The question order for one item: the planner's picks in the listings' spelling, the keys most of its
        listings carry moved up (GUARD_*), other keys after, catch-all keys last; with few messages, sure ones first."""
        keys = self.lane_keys(number, table)
        read = max(1, self.records_of_item(number))
        own: dict[str, float] = {}
        for label, key, carried, _ in table:
            if self.owner_of(label) == number:
                own[key] = max(own.get(key, 0.0), carried / read)
        hints = {_words(self._rw_spelled(own, h)) for h in hints}
        lane: list[str] = []
        for pick in picks:
            if all((_words(self._rw_spelled(own, pick)) != _words(other) for other in lane)):
                lane.append(self._rw_spelled(own, pick))
        guards = [k for k in keys if own.get(k, 0.0) >= GUARD_SHARE and not re.search(CATCH_ALL, k)]
        for key in guards[:GUARD_KEYS]:
            lane = self._rw_guarded(lane, key, own, hints)
        lane += [k for k in self._rw_new_keys(keys, lane)]
        broad = [bool(re.search(CATCH_ALL, _words(x))) for x in lane]
        lane = [x for x, wide in zip(lane, broad) if not wide] + [x for x, wide in zip(lane, broad) if wide]
        if count > TIGHT_MESSAGES or read < 3:
            return lane
        sure = [x for x in lane if _words(x) in hints or self._rw_share(own, x) >= 0.5]
        return sure + [x for x in lane if x not in sure]

    @staticmethod
    def _rw_new_keys(keys: list[str], lane: list[str]) -> list[str]:
        """Keys not yet in the lane, in order (each added key counts for the next)."""
        added: list[str] = []
        for key in keys:
            if all((_words(key) != _words(other) for other in lane + added)):
                added.append(key)
        return added

    def _rw_guarded(self, lane: list[str], key: str, own: dict[str, float], hints: set[str]) -> list[str]:
        """The lane with a key most listings carry placed among the first three questions, over a weaker unhinted
        pick; the lane unchanged when that would push a hinted pick out of the first three."""
        top = lane[:3]
        if any((_words(key) == _words(other) for other in top)):
            return lane
        weak = None
        for at, other in enumerate(top):
            if _words(other) not in hints and self._rw_share(own, other) + GUARD_GAP <= own.get(key, 0.0):
                weak = at
                break
        if weak is None and len(lane) >= 3:
            return lane
        placed = [other for other in lane if _words(other) != _words(key)]
        if weak is None:
            placed.append(key)
        elif len(self.items) > 1:
            moved = placed[weak]
            placed[weak] = key
            placed.insert(min(3, len(placed)), moved)
        else:
            placed.insert(min(len(placed), weak), key)
        head = placed[:3]
        if any((_words(x) in hints and all((_words(x) != _words(y) for y in head)) for x in top)):
            return lane
        return placed

    def whole_pointers(self, heard: dict[str, Any], reply: str) -> dict[str, Any]:
        """The reading with each pointing need valued by the answer's own words (two words before the value as read, to
        the end of that sentence) and without the needs read from inside such a description."""
        needs = [entry for entry in _items(heard.get('needs')) if isinstance(entry, dict)]
        if not reply or not any((entry.get('points') is True for entry in needs)):
            return heard
        spans: list[str] = []
        for entry in needs:
            if entry.get('points') is not True:
                continue
            prop, _, value = str(entry.get('need') or '').partition(':')
            words = value.split()
            if len(words) < 3:
                continue
            found = re.search('\\s+'.join((re.escape(w) for w in words[:3])), reply, re.I)
            if not found:
                continue
            start = re.search('(?:\\S+\\s+){0,2}$', reply[:found.start()])
            begin = start.start() if start else found.start()
            stop = re.search('[.!?](?:\\s|$)', reply[found.start():])
            end = found.start() + stop.start() if stop else len(reply)
            whole = reply[begin:end].strip()
            if len(whole) > len(value.strip()):
                entry['need'] = prop.strip() + ': ' + whole
            spans.append(' '.join(_words(str(entry['need']).partition(':')[2]).split()))
        if spans:
            kept = []
            for entry in needs:
                prop, _, value = str(entry.get('need') or '').partition(':')
                said = ' '.join((_words(prop) + ' ' + _words(value)).split())
                if entry.get('points') is not True and said and any((said in span for span in spans)):
                    self.trace.append({'kind': 'pointer_part_dropped', 'need': str(entry.get('need'))[:80]})
                    continue
                kept.append(entry)
            heard = dict(heard, needs=kept)
        return heard

    def misheard(self, heard: dict[str, Any], labels: list[str], reply: str) -> bool:
        """Whether an answer deserves a second reading because the reader missed or misplaced what it says: no need, no
        indifference and no complaint read from an answer that names an asked property, or a need read on a property
        sharing no word with the asked ones while the answer names one of them."""
        said = set(re.findall('[a-z0-9]+', _words(reply)))
        asked = [set(_words(label).split()) for label in labels if _words(label)]
        named = any((words <= said for words in asked if words))
        if not named:
            return False
        needs = [entry for entry in _items(heard.get('needs')) if isinstance(entry, dict) and str(entry.get('need') or '').strip()]
        if not needs:
            return not _items(heard.get('free')) and (not _items(heard.get('crowded'))) and (not re.search(M_NO_VIEW, reply, re.I)) and (not re.search(M_REFUSED, reply, re.I))
        mentioned = set().union(*asked) if asked else set()
        return any((not set(_words(str(entry['need']).partition(':')[0]).split()) & mentioned for entry in needs))

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

    def question(self, number: int, label: str) -> str:
        return f'What {label} would you like for the {self.items[number - 1].name}?'

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

    def sane_budget(self, new: float) -> float:
        """A new budget read from a reply or a note, in the request's unit: a figure about a hundredth (or a hundred
        times) of the budget in force is the same sum read in the other unit (dollars for cents), so it is scaled back."""
        old = self.budget
        if not isinstance(old, (int, float)) or old <= 0:
            return new
        if new * 20 < old and 0.2 <= new * 100 / old <= 5:
            self.trace.append({'kind': 'budget_unit', 'read': new, 'kept': new * 100})
            return round(new * 100, 2)
        if new > old * 20 and 0.2 <= new / 100 / old <= 5:
            self.trace.append({'kind': 'budget_unit', 'read': new, 'kept': new / 100})
            return round(new / 100, 2)
        return new

    def take_heard(self, heard: dict[str, Any], told: list[str]) -> None:
        many = len(self.items) > 1
        for entry in _items(heard.get('needs')):
            if not isinstance(entry, dict) or not str(entry.get('need') or '').strip():
                continue
            number = min(max(1, _count(entry.get('item')) or 1), len(self.items))
            need = str(entry['need']).strip()
            if need not in self.items[number - 1].told:
                self.items[number - 1].told.append(need)
                told.append(f'item {number}: {need}' if many else need)
            if entry.get('points') is True and (number, need) not in self.refs:
                self.refs.append((number, need))
        for entry in _items(heard.get('drop')):
            if isinstance(entry, dict) and str(entry.get('property') or '').strip():
                item = self.items[min(max(1, _count(entry.get('item')) or 1), len(self.items)) - 1]
                gone = _words(str(entry['property']).split(':')[0])
                hit = [m for m in item.musts + item.told if m not in item.dropped and _words(m.split(':')[0]) == gone]
                item.dropped.extend(hit)
                told.extend((f'dropped {m}' for m in hit))
        for entry in _items(heard.get('free')):
            if isinstance(entry, dict) and entry.get('property'):
                number = min(max(1, _count(entry.get('item')) or 1), len(self.items))
                self.items[number - 1].free.append(str(entry['property']))
        if isinstance(heard.get('budget'), (int, float)) and (not isinstance(heard.get('budget'), bool)) and (heard['budget'] > 0):
            self.budget = self.sane_budget(heard['budget'])

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(HEAVY, LIGHT)) -> dict[str, Any]:
        prompt = HEAR + '\n\nITEMS:\n' + '\n'.join((f'item {n}: {it.name}' for n, it in enumerate(self.items, 1)))
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f'{i}. (about item {n}) Q: {q}\nA: {a}' for i, (n, q, a) in enumerate(sent, 1)))
        return self.ask_json(prompt, list(models), 2500) or {}

    def gate(self, stage: str) -> tuple[bool, str]:
        steps = [p for p in self.process if p['when'] == stage]
        if not steps or self.finished or not self.gate_due(stage):
            return (True, '')
        ending = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        tools = [self._rw_tool_line(t['function']) for t in self.tools if t['function']['name'] not in ending]
        said: list[str] = []
        nudged = False
        rounds = 0
        while rounds < GATE_ROUNDS:
            rounds += 1
            got = self.ask_json(self._rw_gate_prompt(stage, steps, tools), [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 3500)
            if got is None:
                break
            call, change = self._rw_gate_heard(stage, got, said)
            if change:
                self.version += 1
                return (False, change)
            if not call and not said and not nudged:
                unproven = self._rw_unproven(stage, steps, got)
                if unproven and self.roles.get('ask') and self.msg_used < self.msg_limit and self.left() >= 2:
                    nudged = True
                    self._rw_gate_nudge(unproven)
                    continue
            if not self._rw_gate_call(stage, call, ending, said):
                break
        return (True, 'Before going on, Kyo did the steps the request asks for:\n' + '\n'.join(said) if said else '')

    def _rw_gate_prompt(self, stage: str, steps: list[dict[str, Any]], tools: list[str]) -> str:
        parts = [GATE.replace('STAGE', 'place the order' if stage == 'order' else 'end without a purchase'),
                 '\n\nREQUEST:\n', self.request,
                 '\n\nSTEPS DUE NOW:\n', '\n'.join((f"- {p['step']}" for p in steps)),
                 '\n\nLEDGER:\n', self.ledger(),
                 '\n\nCART NOW: ', self.cart_text(),
                 '\n\nRECORD (messages and step calls so far):\n', '\n'.join((f'{i}. {line}' for i, line in enumerate(self.done_log, 1))) or '(nothing yet)']
        if self.notes:
            parts += ['\n\nSTORE NOTES SEEN:\n', '\n'.join(self.notes[-4:])]
        parts.append(f"\n\nMESSAGES LEFT: {max(0, self.msg_limit - self.msg_used)}   STEPS LEFT: {self.left()} (the {('order' if stage == 'order' else 'ending')} itself takes one)")
        parts += ['\n\nTOOLS:\n', '\n'.join(tools)]
        return ''.join(parts)

    def _rw_gate_heard(self, stage: str, got: dict[str, Any], said: list[str]) -> tuple[dict[str, Any] | None, str]:
        """The gate reply's call, and the text to stop on when the shopper's replies so far change the task."""
        told: list[str] = []
        hold = ''
        if said:
            self.take_heard({'needs': got.get('needs')}, told)
            hold = str(got.get('hold') or '').strip()
        call = got.get('call') if isinstance(got.get('call'), dict) else None
        self.trace.append({'kind': 'gate', 'stage': stage, 'call': _clip(call, 200), 'steps': _clip(got.get('steps'), 120), 'hold': hold[:200], 'told': told})
        if not told and not hold:
            return (call, '')
        done = 'Before going on, Kyo did the steps the request asks for:\n' + '\n'.join(said) + '\n' if said else ''
        return (call, done + "The shopper's reply changes the task: " + '; '.join(told + ([hold] if hold else [])) + '. Fix the cart for it, then check out again.')

    def _rw_unproven(self, stage: str, steps: list[dict[str, Any]], got: dict[str, Any]) -> list[dict[str, Any]]:
        """The steps the gate reply shows that no record line proves done (after the last cart change, when the step
        talks to the shopper before an order)."""
        talk = stage == 'order' and any((p['messages'] > 0 for p in steps))
        proof = set()
        for line in range(1, len(self.done_log) + 1):
            if not talk or (line <= len(self.done_steps) and self.done_steps[line - 1] > self.cart_at):
                proof.add(line)
        unproven = []
        for entry in _items(got.get('steps')):
            if isinstance(entry, dict) and (entry.get('done') is not True or not any(((_count(n) or 0) in proof for n in _items(entry.get('lines'))))):
                unproven.append(entry)
        return unproven

    def _rw_gate_nudge(self, unproven: list[dict[str, Any]]) -> None:
        self.trace.append({'kind': 'gate_unproven', 'steps': _clip(unproven, 120)})
        self.done_log.append(f'(Kyo: no line made since the cart last changed at step {self.cart_at} shows ' + '; '.join((str(e.get('step') or '') for e in unproven))[:200] + f' done, and {self.msg_limit - self.msg_used} message(s) and {self.left()} steps are left: give the call that does it)')
        self.done_steps.append(-1)

    def _rw_gate_call(self, stage: str, call: dict[str, Any] | None, ending: set[str], said: list[str]) -> bool:
        """Runs the call the gate asked for (a message or a store step); False when the gate should stop asking."""
        name = str((call or {}).get('tool') or '')
        if not call or name not in self.schemas or name in ending:
            if not call:
                self.gate_ok.add((stage, self.cart_mark() + f'#{self.version}'))
            return False
        asking = name == self.roles.get('ask')
        if self.left() < 2 or (asking and self.msg_used >= self.msg_limit):
            return False
        faults: list[str] = []
        args = _shape(call.get('args') if isinstance(call.get('args'), dict) else {}, self.schemas[name], 'args', faults)
        if faults:
            return False
        if asking:
            text = next((v for v in args.values() if isinstance(v, str)), '')
            batch = self.send([(1, '', text)], 1)
            if not batch:
                return False
            self.gate_msgs += 1
            said.append(f'Q: {text}\nA: {batch[0][2]}')
            return True
        reply = self.act(name, args)
        if reply is None:
            return False
        seen = reply.get('observation')
        self.absorb(seen)
        self.done_log.append(f'step {self.step}: {name} {_dump(args)} -> {self.show(seen)[:300]} (cart then: {self.cart_mark()})')
        self.done_steps.append(self.step)
        said.append(f'{name} {_dump(args)} -> {self.show(seen)[:400]}')
        return True

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
            budget = self.sane_budget(budget)
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
            self.price_note = True
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
        return _dump(_clip(record, 250))[:1800] if record else '(no record seen)'

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
            record = self.full.get(ident) or self.rows.get(ident) or {}
            text = ' '.join((str(v) for v in record.values() if isinstance(v, str)))
            text += ' ' + '. '.join((f'{k} {v}' for block in record.values() if isinstance(block, dict) for k, v in block.items() if isinstance(v, str)))
            marks[ident] = _fit_marks(text)
            for part, codes in marks[ident]['also']:
                notes.setdefault(part, set()).update(codes)
        return {'marks': marks, 'notes': notes}

    def fit_set(self, ident: str, index: dict[str, Any]) -> set[str]:
        mark = index['marks'].get(ident) or {}
        found = set(mark.get('works') or [])
        if mark.get('own'):
            found |= index['notes'].get(mark['own'], set())
        return found

    def item_ids(self, number: int) -> list[str]:
        cat_key = self.category_key()
        item = self.items[number - 1]
        return [i for i in self.seen_order if cat_key and self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))]

    def fit_faults(self) -> list[str]:
        if not self.bound:
            return []
        index = self.fit_index()
        cart = self.cart_ids()
        faults = []
        for part, main in self.bound:
            devices = [i for i in cart if i in self.item_ids(main)]
            parts = [i for i in cart if i in self.item_ids(part)]
            if len(devices) != 1 or not parts:
                continue
            device = index['marks'].get(devices[0]) or {}
            for line in parts:
                if self.fits_device(line, device, index) is False:
                    good = [i for i in self.item_ids(part) if self.fits_device(i, device, index) and (not self.sold_out(i)) and (i not in cart)][:4]
                    named = sorted(self.fit_set(line, index))[:4] or [(index['marks'].get(line) or {}).get('own') or 'other models']
                    faults.append(f"{line} ({self.items[part - 1].name}) works with {', '.join(named)}, not with {devices[0]} ({', '.join(sorted(device.get('codes') or device.get('series') or []))})" + (f"; seen parts of that kind that work with it: {', '.join(good)}" if good else ''))
        return faults

    def fits_device(self, ident: str, device: dict[str, Any], index: dict[str, Any]) -> bool | None:
        return _fit_verdict(index['marks'].get(ident) or {}, device, self.fit_set(ident, index))

    def holds(self, ident: str, number: int) -> bool | None:
        """Whether a listing shows every plain required value of its item (limits, exclusions and values taken from
        another listing are skipped): under a key naming the property when it has one, else in its title or text.
        None when the item has no plain value."""
        record = self.full.get(ident) or self.rows.get(ident) or {}
        block = _fact_block(record) or {}
        text = ' '.join((str(v) for v in record.values() if isinstance(v, str))).lower()
        pointed = {need for _, need in self.refs}
        plain = [need for n, need in self.needs_now() if n == number and ':' in need and need not in pointed and need.split(':', 1)[1].strip() in self.value_words(number)]
        if not plain:
            return None
        for need in plain:
            prop, value = (part.strip() for part in need.split(':', 1))
            words = re.findall('[a-z0-9.]+', value.lower())
            keyed = [str(v).lower() for k, v in block.items() if _names(_stem(prop), _stem(k))]

            def has(hay: str) -> bool:
                return all((re.search('(?<![a-z0-9.])' + re.escape(w) + '(?![a-z0-9.])', hay) for w in words))
            if keyed:
                if not any((has(v) for v in keyed)):
                    return False
            elif not has(text):
                flag = bool(words) and all((w in ('yes', 'true') for w in words))
                named = re.findall('[a-z0-9.]+', prop.lower())
                if not (flag and named and all((re.search('(?<![a-z0-9.])' + re.escape(w) + '(?![a-z0-9.])', text) for w in named))):
                    return False
        return True

    def fit_plan(self, short: bool=False) -> str:
        if not self.bound:
            return ''
        index = self.fit_index()
        lines = []
        for main in sorted({m for _, m in self.bound}):
            parts = [p for p, m in self.bound if m == main]
            tally: dict[str, dict[int, int]] = {}
            good: dict[str, dict[int, list[str]]] = {}
            for p in parts:
                for i in self.item_ids(p):
                    if self.sold_out(i):
                        continue
                    own = (index['marks'].get(i) or {}).get('own') or ''
                    named = self.fit_set(i, index) | ({own.split('-')[0]} if re.match(PART_SERIES + '-', own) else set())
                    fits = self.holds(i, p)
                    for code in named:
                        tally.setdefault(code, {}).setdefault(p, 0)
                        tally[code][p] += 1
                        if fits:
                            good.setdefault(code, {}).setdefault(p, []).append(i)
            marks = {i: index['marks'].get(i) or {} for i in self.item_ids(main)}
            devices: dict[str, list[str]] = {}
            for code in tally:
                named = [i for i, m in marks.items() if any((_code_names(code, c) for c in m.get('codes') or [])) or code in (m.get('series') or [])]
                devices[code] = sorted(named, key=lambda i: self.holds(i, main) is not True)

            def rank(kv: tuple[str, dict[int, int]]) -> tuple[int, int, int, int]:
                code, per = kv
                return (-len(good.get(code) or {}), -int(any((self.holds(i, main) for i in devices[code]))), -len(per), -sum(per.values()))
            ranked = sorted(tally.items(), key=rank)[:4]

            def part_text(code: str, p: int, n: int) -> str:
                shown = (good.get(code) or {}).get(p) or []
                plain = self.holds(shown[0], p) if shown else next((self.holds(i, p) for i in self.item_ids(p)), None)
                tail = f" ({len(shown)} showing every plain required value: {', '.join(shown[:3])})" if shown else (' (none showing every plain required value)' if plain is not None else '')
                return f'{n} {self.items[p - 1].name}' + tail

            def device_text(code: str) -> str:
                seen = devices[code][:3]
                if not seen:
                    return ' (device not seen yet)'
                return ' (device seen: ' + ', '.join((i + ('*' if self.holds(i, main) else '') for i in seen)) + ')'
            text = '; '.join((f'{code}: ' + ', '.join((part_text(code, p, n) for p, n in sorted(per.items()))) + device_text(code) for code, per in ranked))
            if text:
                lines.append(f"codes the {self.items[main - 1].name} candidates' parts name -> {text}")
        if not lines:
            return ''
        return ' | '.join(lines) if short else "[fit plan: a device fits a part when the part's text names the device's model code, its series or another name it is sold as, or when the device's text names the part's series; codes are not searchable, so list the device's category and read the codes on its rows. Series are ranked by the parts whose own text shows every plain required value of their item; a device marked * shows every plain value of its own (limits and values taken from another listing are not read here and still need checking). Build the order on such a series; when the chosen device takes no series with such a part, switch the device instead of searching on] " + ' | '.join(lines)

    def line_faults(self) -> list[str]:
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
            prompt += '\n\nOTHER LISTINGS SEEN (newest first):\n' + self.others(ids, 10)
        got = self.ask_json(prompt, self.judges(limit), 3000)
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

    def trail(self) -> tuple[str, bool]:
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        tool = self.roles.get('offer') or self.roles.get('open')
        target = self.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        while self._rw_trail_room(pending):
            self.trails += 1
            prompt = TRAIL + '\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n' + '\n'.join(pending) + '\n\nRECORDS (newest first):\n' + '\n'.join(self._rw_trail_records())
            got = self.ask_json(prompt, [LIGHT] + ([HEAVY] if self.cost + HEAVY_ROOM <= SPEND_HARD else []), 4500) or {}
            self.trace.append({'kind': 'trail', 'pending': pending, 'got': _clip(got, 150)})
            reach: list[str] = []
            lost = []
            gaps: list[tuple[str, str]] = []
            for chain in _items(got.get('chains')):
                if isinstance(chain, dict):
                    self._rw_follow(chain, pending, reach, lost, gaps)
            pending = [p for p in pending if p not in self.ref_values]
            if not reach and gaps and tool and target and self.left() > 12 and self.link_searches < 3:
                reach = self.find_link(*gaps[0])
                if self.trace and self.trace[-1].get('kind') == 'link_known':
                    seen.append(f"{self.trace[-1]['id']}: an opened page of the shop this chain names that shows the values its description gives; read the value from it")
                    lost = lost[1:]
            if not reach or not tool or not target or self.left() < len(reach) + 4 or self.finished:
                break
            self._rw_trail_open(tool, target, reach[:max(0, min(3, 6 - len(seen)))], seen)
        parts = []
        if seen:
            parts.append('[Kyo opened the pages a requirement points at]\n' + '\n'.join(seen) + '\n')
        if self.ref_values:
            parts.append('[values read from the pages requirements point at] ' + '; '.join((f'{k} -> {v}' for k, v in self.ref_values.items())) + '\n')
        hold = bool(lost) and self.trail_blocks < (3 if self.left() > 15 else 2) and self.left() > 5
        if hold:
            self.trail_blocks += 1
            parts.append('The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: ' + ' | '.join(lost[:3]) + ". Search for it by its shop's name together with the values its description gives (the shop's other listings of that kind are not it), open the row of that shop whose page shows those values, apply the value its page gives, then check out.")
        return (''.join(parts), hold)

    def _rw_trail_room(self, pending: list[str]) -> bool:
        return bool(pending) and self.trails < (6 if self.left() > 15 else 4) and self.cost < SPEND_HARD

    def _rw_trail_records(self) -> list[str]:
        """Opened records (45 at most), then search rows of listings not opened, newest first (60 lines at most)."""
        records = []
        newest = list(reversed(self.seen_order))
        for ident in newest:
            record = (self.full.get(ident) or self.rows.get(ident)) if ident in self.opened else None
            if record and len(records) < 45:
                records.append('OPENED ' + _dump(_clip(record, 300))[:1400])
        rows = [i for i in newest if i not in self.opened or not (self.full.get(i) or self.rows.get(i))]
        records += ['ROW ' + _dump(_clip(self.rows.get(i), 80))[:300] for i in rows[:max(0, 60 - len(records))]]
        return records

    def page_rules(self, ident: str) -> list[tuple[str, list[tuple[str, str]]]]:
        """The pairing rules a listing's own words state (its text fields)."""
        record = self.full.get(ident) or self.rows.get(ident) or {}
        return [rule for value in record.values() if isinstance(value, str) for rule in _pair_rules(value)]

    def own_value(self, ident: str, prop: str, pairs: list[tuple[str, str]] | None=None) -> str:
        """The listing's own attribute value for a property: the key named the same first, then a key that names it;
        when the rule's values are given and that value is not one of them, a key whose value is one of them (a key
        sharing a word with the property first)."""
        block = _fact_block(self.full.get(ident) or {}) or {}
        want = _stem(prop)
        ranked = sorted(block.items(), key=lambda kv: (_words(kv[0]) != _words(prop), not _names(want, _stem(kv[0]))))
        named = next((str(value).strip() for key, value in ranked if _words(key) == _words(prop) or _names(want, _stem(key))), '')
        lefts = {_words(a) for a, _ in pairs or []}
        if not pairs or _words(named) in lefts:
            return named
        others = sorted(block.items(), key=lambda kv: not (_stem(kv[0]) & want))
        return next((str(value).strip() for _, value in others if _words(str(value)) in lefts), named)

    @staticmethod
    def rule_partner(own: str, pairs: list[tuple[str, str]]) -> str:
        """The partner a pairing rule gives a value: the same words first, else the one pair whose value the own value
        starts with or contains."""
        same = [b for a, b in pairs if _words(a) == _words(own)]
        if same:
            return same[0]
        near = [b for a, b in pairs if re.search('\\b%s\\b' % re.escape(_words(a)), _words(own))]
        return near[0] if len(set(near)) == 1 else ''

    def pairing_notes(self, observation: Any) -> str:
        """A line for each opened listing whose words state a pairing rule: its own value and the partner the rule gives
        it, which is what a requirement taking its value from this page wants."""
        seen = getattr(self, 'pair_seen', None)
        if seen is None:
            seen = self.pair_seen = set()
        lines = []
        for ident, _ in self.records_in(observation):
            if ident in seen or not _fact_block(self.full.get(ident) or {}):
                continue
            for prop, pairs in self.page_rules(ident):
                seen.add(ident)
                own = self.own_value(ident, prop, pairs)
                partner = self.rule_partner(own, pairs) if own else ''
                if partner:
                    lines.append(f'[pairing rule on {ident}: its own {prop} is {own}, and its rule pairs {own} with {partner}. A requirement that takes its {prop} from this page wants {partner}, not {own}.]')
                else:
                    lines.append(f'[pairing rule on {ident} for {prop}: ' + '; '.join((f'{a} -> {b}' for a, b in pairs[:6])) + f'. A requirement that takes its {prop} from this page wants the partner of the own {prop} of the listing the chain ends on.]')
        return '\n'.join(lines)

    def paired_value(self, ident: str, need: str, value: str) -> str:
        """A chain's value read through the pairing rule of the page it ends on, when that page states one for the
        requirement's property and its own value is in the rule."""
        prop = _stem(need.split(':', 1)[0])
        for rule_prop, pairs in self.page_rules(ident):
            if not (_stem(rule_prop) & prop):
                continue
            own = self.own_value(ident, rule_prop, pairs)
            partner = self.rule_partner(own, pairs) if own else ''
            if partner:
                self.trace.append({'kind': 'paired', 'need': need, 'page': ident, 'own': own, 'partner': partner})
                return f'{partner} (its own {rule_prop} {own} pairs with {partner})'
        return value

    def _rw_follow(self, chain: dict[str, Any], pending: list[str], reach: list[str], lost: list[str], gaps: list[tuple[str, str]]) -> None:
        """One chain of the trail reply: a value read, listings still to open (reach), or a listing to find."""
        named = str(chain.get('need') or '').strip()
        need = (named if named in pending else None) or (pending[0] if len(pending) == 1 else None)
        if not need:
            return
        links = [l for l in _items(chain.get('links')) if isinstance(l, dict)]
        ids = [str(l.get('id') or '').strip() for l in links]
        walked = bool(links) and all((i in self.opened for i in ids))
        if walked and chain.get('done') is True and str(chain.get('value') or '').strip():
            self.ref_values[need] = self.paired_value(ids[-1], need, str(chain['value']).strip())
            self.version += 1
            return
        if walked and not str(links[-1].get('find') or '').strip():
            self.ref_values[need] = self.paired_value(ids[-1], need, str(chain.get('value') or '').strip() or f'as stated on {ids[-1]}')
            self.version += 1
            return
        closed = [i for i in ids if i in self.rows and i not in self.opened]
        if closed:
            reach += [i for i in closed if i not in reach]
            return
        gap = next((l for l, i in zip(links, ids) if i not in self.rows), {})
        about = str(gap.get('about') or '') or 'the listing it points at'
        words = str(gap.get('find') or '').strip()
        lost.append(f'{need} -> find {about}' + (f' (search: {words})' if words else ''))
        gaps.append((about, words))

    def _rw_trail_open(self, tool: str, target: str, idents: list[str], seen: list[str]) -> None:
        for ident in idents:
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
            if faults:
                return
            reply = self.act(tool, args)
            if reply is None or self.finished:
                return
            self.observe(tool, reply.get('observation'))
            seen.append(f'{ident}: ' + self.show(reply.get('observation'))[:1500])

    def shop_rows(self, about: str, ids: list[str]) -> list[str]:
        low = about.lower()
        out = []
        for ident in ids:
            row = self.rows.get(ident) or {}
            names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I)]
            if ident not in self.opened and any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)):
                out.append(ident)
        return out

    def opened_shop(self, about: str) -> list[str]:
        low = about.lower()
        out = []
        for ident in self.opened:
            row = self.rows.get(ident) or {}
            names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I)]
            if any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)):
                out.append(ident)
        return out

    def find_link(self, about: str, words: str) -> list[str]:
        known: list[str] = []
        named = {w for ident in self.shop_rows(about, list(self.rows)) + self.opened_shop(about) for k, v in (self.rows.get(ident) or {}).items() if isinstance(v, str) and len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I) for w in re.findall('[a-z0-9]{3,}', v.lower())}
        values = {w for w in re.findall('[a-z0-9]+', (words or about).lower()) if len(w) >= 3 or w.isdigit()} - named - {w for w in re.findall('[a-z]+', (words or about).lower()) if FILLER.fullmatch(w)}
        for ident in self.opened_shop(about):
            stack, leaves = ([self.full.get(ident) or self.rows.get(ident) or {}], [])
            while stack:
                node = stack.pop()
                if isinstance(node, (dict, list)):
                    stack += list(node.values()) if isinstance(node, dict) else node
                else:
                    leaves.append(str(node))
            text = ' '.join(leaves).lower()
            if values and all((re.search('(?<![a-z0-9])' + re.escape(v) + '(?![a-z0-9])', text) for v in values)):
                self.trace.append({'kind': 'link_known', 'id': ident})
                return []
        plan = self.search_template()
        memo = ' '.join(words.lower().split())
        if memo and memo in self.link_seen:
            self.link_searches += 1
            known = self.shop_rows(about, self.link_seen[memo])
        elif plan and words:
            tool, base, field = plan
            faults: list[str] = []
            args = _shape({**base, field: words}, self.schemas.get(tool), 'args', faults)
            if not faults:
                self.link_searches += 1
                reply = self.act(tool, args)
                if reply is not None:
                    observation = reply.get('observation')
                    self.observe(tool, observation)
                    self.link_seen[memo] = [i for i, _ in self.records_in(observation)]
                    known = self.shop_rows(about, self.link_seen[memo])
                    self.trace.append({'kind': 'link_search', 'words': words[:80], 'rows': known[:4]})
        kind = {w for w in re.findall('[a-z0-9]{3,}', (words or about).lower())} - named - {w for w in re.findall('[a-z]+', (words or about).lower()) if FILLER.fullmatch(w)}

        def of_kind(ids: list[str]) -> list[str]:
            return [i for i in ids if not kind or kind & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower()))]
        return (of_kind(known) or of_kind(self.shop_rows(about, list(reversed(self.seen_order)))))[:2]

    def chain_offers(self, name: str, observation: Any) -> str:
        """A store may split a listing over two pages: attributes on one, and price, stock and the pointers to other
        listings on the other. When the walker reads the attributes of listings from a shop a pending lookup names,
        Kyo opens their offer page too (where such a pointer would show), at most 2 per lookup, while steps allow."""
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not tool or not target or name == tool or name not in {self.roles.get('open'), self.roles.get('compare'), self.facts_tool()} - {None}:
            return ''
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if not pending:
            return ''
        spent = getattr(self, 'chain_opened', {})
        texts = []
        for ident, record in self.records_in(observation):
            if ident in self.offer_seen or not _fact_block(record) or self.left() <= self.tail() + 6 or self.finished or self.time_left() < EXPLORE_RESERVE:
                continue
            row = self.rows.get(ident) or {}
            names = [str(v).strip().lower() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I)]
            text = _dump(record).lower()
            about = None
            for need in pending:
                low = need.lower()
                shop = [n for n in names if re.search('(?<![a-z0-9])' + re.escape(n) + '(?![a-z0-9])', low)]
                if not shop or spent.get(need, 0) >= 2:
                    continue
                words = {w for w in re.findall('[a-z0-9]{3,}', low) if not FILLER.fullmatch(w)} - {w for n in shop for w in re.findall('[a-z0-9]{3,}', n)}
                if any((re.search('(?<![a-z0-9])' + re.escape(w) + '(?![a-z0-9])', text) for w in words)):
                    about = need
                    break
            if about is None:
                continue
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
            if faults:
                break
            reply = self.act(tool, args)
            if reply is None or self.finished:
                break
            spent = {**spent, about: spent.get(about, 0) + 1}
            self.chain_opened = spent
            self.offer_seen[ident] = self.step
            seen = reply.get('observation')
            self.observe(tool, seen)
            texts.append(f'{ident}: ' + self.show(seen)[:1200])
        if not texts:
            return ''
        self.trace.append({'kind': 'chain_offers', 'n': len(texts)})
        return "\n[Kyo also opened the offer page of these listings from a shop a requirement names (one step each): pointers to other pages show there]\n" + '\n'.join(texts)

    def part_page(self, code: str, among: list[str] | None=None) -> str | None:
        """The listing whose own part code (in brackets in its title) is `code`, among the given or all seen rows."""
        for ident in among if among is not None else list(self.seen_order):
            row = self.full.get(ident) or self.rows.get(ident) or {}
            text = ' '.join((str(v) for v in row.values() if isinstance(v, str)))
            if _fit_marks(text)['own'] == code:
                return ident
        return None

    def referred_pages(self) -> list[str]:
        """Pages a cart line sends the reader to for the rest of its details ("more on part X"): the order relies on
        them and the store counts a page only when it was opened, so Kyo opens each one once before the order (found
        among the rows seen, else by searching its code), while the steps allow."""
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return []
        done = getattr(self, 'referred', set())
        codes: list[str] = []
        for line in self.cart_ids():
            record = self.full.get(line) or self.rows.get(line) or {}
            text = ' '.join((str(v) for v in record.values() if isinstance(v, str)))
            text += ' ' + ' '.join((f'{k} {v}' for block in record.values() if isinstance(block, dict) for k, v in block.items() if isinstance(v, str)))
            own = _fit_marks(text)['own']
            codes += [c.upper() for c in re.findall(REFERS, text, re.I) if c.upper() != own and c.upper() not in done and c.upper() not in codes]
        texts: list[str] = []
        for code in codes:
            if self.left() - self.tail() < 3 or self.finished or self.time_left() < CALL_RESERVE + 20:
                break
            done = done | {code}
            self.referred = done
            page = self.part_page(code)
            if page is None and self.left() - self.tail() >= 4:
                plan = self.search_template()
                if plan:
                    found = self.search_words(plan[0], plan[1], plan[2], code)
                    page = self.part_page(code, found)
            if page is None or page in self.opened or self.offer_seen.get(page) is not None:
                continue
            faults: list[str] = []
            args = _shape({target: page}, self.schemas.get(tool), 'args', faults)
            if faults:
                break
            reply = self.act(tool, args)
            if reply is None or self.finished:
                break
            self.offer_seen[page] = self.step
            self.observe(tool, reply.get('observation'))
            texts.append(f'{page} (part {code})')
        if codes:
            self.trace.append({'kind': 'referred_pages', 'codes': codes, 'opened': texts})
        return texts

    def offer_tool(self) -> str | None:
        return self.roles.get('offer') or self.roles.get('open')

    def reread_offers(self, only_new: bool=False) -> str:
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return ''
        since = max(self.step - 2, self.refused_at)
        stale = [i for i in self.cart_ids() if self.offer_seen.get(i, -99) < since and i not in self.gone]
        if stale and (only_new or self.left() < len(stale) + 3):
            stale = [i for i in stale if i not in self.offer_seen]
            stale = stale if self.left() >= len(stale) + 2 else []
        if not stale:
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
        picks = self.replacements(lost)
        text = f"[Kyo] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        if picks and self.left() <= 6:
            return text + f" Few steps are left: in this turn add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks[1:]) or 'none'}."
        if picks and self.brief.get('pick') == 'cheapest':
            return text + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(picks)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {picks[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        if picks:
            return text + f" Listings already read that appear to fit that item instead: {', '.join(picks)}. In this turn take the sold-out line out of the cart and add {picks[0]} at the requested quantity, then check out."
        return text + ' Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits.'

    def find_sold(self) -> str:
        ids = [i for i in self.cart_ids() if i not in self.gone]
        if any((self.sold_out(i) for i in ids)):
            return ''
        if len(ids) == 1:
            self.unavailable.add(ids[0])
            return ''
        return self.reread_offers() if ids else ''

    def replacements(self, lost: list[str]) -> list[str]:
        if not self.judges():
            return []
        records = []
        for ident in reversed(self.seen_order):
            if ident in lost or ident in self.gone or self.sold_out(ident):
                continue
            records.append(_dump(_clip(self.full.get(ident) or self.rows.get(ident), 150))[:700])
            if len(records) >= 15:
                break
        if not records:
            return []
        prompt = SWAP + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
        prompt += '\n\nSOLD-OUT CART LINE:\n' + '\n'.join((f'{i}: {self.record_of(i)}' for i in lost)) + '\n\nCART NOW:\n' + self.cart_text()
        prompt += '\n\nOTHER LISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
        got = self.ask_json(prompt, self.light_first(), 3500)
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

    def light_first(self) -> list[str]:
        """The judges in force with the light model first: the heavy one answers only when the light reply is empty."""
        models = self.judges()
        return [LIGHT] + [m for m in models if m != LIGHT] if models else []

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
                if record and len(opened) < 12:
                    opened.append('OPENED ' + _dump(_clip(record, 150))[:700])
                elif not record and len(rows) < 10:
                    rows.append('ROW ' + _dump(_clip(self.rows.get(ident), 100))[:400])
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
        got = self.ask_json(prompt, self.light_first(), 4000)
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
        wanted = {w for w in re.findall('[a-z0-9]{3,}', ' '.join(self.value_words(number)).lower())}
        rows = [i for i in dict.fromkeys(found) if self.buyable(i) and (not cat_key or self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or '')))]
        if below is not None:
            rows = [i for i in rows if (self.price_of(i) if self.price_of(i) is not None else below) < below]
        return sorted(rows, key=lambda i: (i in self.full, -len(wanted & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower()))), self.price_of(i) or 0))

    def recovery_search(self, got: dict[str, Any] | None, missing: list[int]) -> bool:
        """Before a decline stands: list each unfitted item's category and run the searches the review proposes, then
        read the most promising new rows, so that the second look at the evidence has something new to judge. Half of
        the steps it may use are kept for reading; the cart and the checkout stay possible."""
        keep = self.tail() + len(self.items) + 2
        room = min(RECOVERY_STEPS * (2 if self.long() else 1), self.left() - keep)
        if room < 3 or self.time_left() < EXPLORE_RESERVE + 25:
            return False
        before = len(self.full)
        last = self.step + room
        searching = self.step + max(1, room // 2)
        plan = self.search_template()
        queries: list[tuple[int, str]] = []
        for entry in _items((got or {}).get('search')):
            if isinstance(entry, dict) and str(entry.get('query') or '').strip():
                queries.append((min(max(1, _count(entry.get('item')) or 1), len(self.items)), str(entry['query']).strip()))
        queries = [q for q in queries if q[0] in missing] or [(n, ' '.join(self.value_words(n)[:2] + [self.items[n - 1].name])) for n in missing]
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
        named = {str(k): [str(i) for i in _items(v) if str(i) in self.rows and (not self.sold_out(str(i)))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def open_rows(self, rows: list[str]) -> bool:
        many, joiner = (self.compare_target(), self.roles.get('compare'))
        tool = self.facts_tool()
        target = self.single_target(tool)
        if many and joiner and (not self.compare_bare()):
            plan = [(joiner, {many: rows[:self.compare_room()]})]
        elif tool and target:
            plan = [(tool, {target: i}) for i in rows[:3]]
        else:
            return False
        if self.left() < len(plan) + 5:
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
        for ident in drop:
            have = self.cart_lines.get(ident, 1)
            if mode:
                calls.append({target: ident, mode: props[mode]['enum'][1], **({count: have} if count else {})})
            elif count and self.cart_sets(name):
                calls.append({target: ident, count: 0})
            elif count and isinstance(props[count].get('minimum'), int) and (props[count]['minimum'] <= -have):
                calls.append({target: ident, count: -have})
            else:
                return None
        for ident, want in lines.items():
            have = self.cart_lines.get(ident, 0) if ident in self.cart_ids() else 0
            if have == want:
                continue
            if have and (not self.cart_sets(name)):
                return None
            calls.append({target: ident, **({mode: props[mode]['enum'][0]} if mode else {}), **({count: want} if count else {})})
        return calls

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
            index = _count(number)
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
        self.check_cart(list(lines), SPEND_HARD)
        if not self.verdicts.get(self.check_key(list(lines))):
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
            keep = set(ends)
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
                have = have - count if count else 0
            elif sets:
                have = count if count is not None else 1
            else:
                have = have + (count if count is not None else 1)
            if have > 0:
                self.cart_lines[ident] = have
            else:
                self.cart_lines.pop(ident, None)

    def show(self, observation: Any) -> str:
        text = _dump(_clip_rows(observation, SHOW_CHARS) if getattr(self, 'line_rows', False) else _clip(observation, SHOW_CHARS))
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
            return max(2, min(self.compare_bad - 1, max(self.compare_ok, 2)))
        return max(4, self.compare_ok)

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
        calls = self.join_opens(calls)
        replies: list[tuple[str, str]] = []
        for call in calls:
            usable = set((t['function']['name'] for t in self.offered()))
            replies.append((call['id'], self._rw_reply(call, usable, len(calls))))
        return replies

    def fit_codes(self, text: str) -> list[str]:
        """Code-like tokens of a text that the request or a seen listing names as a model or fit code (in a fit or model phrase)."""
        asked = _fit_marks(self.request)
        known = set(asked.get('codes') or []) | set(asked.get('works') or [])
        for mark in self.fit_index()['marks'].values():
            known |= set(mark.get('codes') or []) | set(mark.get('works') or [])
        return [m.group(0) for m in re.finditer('\\b' + FIT_CODE + '\\b', text or '') if (_codes(m.group(0)) or [''])[0] in known]

    def code_free(self, turn: _RwCall) -> str:
        """After a search holding a model or fit code found nothing (the store does not index the text those codes sit
        in), later searches run without such codes; a search left with no words is refused without a step."""
        if not getattr(self, 'code_blind', False) or turn.name != self.roles.get('search'):
            return ''
        for key, value in turn.args.items():
            hits = self.fit_codes(value) if isinstance(value, str) else []
            if not hits:
                continue
            rest = value
            for hit in hits:
                rest = rest.replace(hit, ' ')
            rest = ' '.join(rest.split())
            if len(re.sub('[^A-Za-z]', '', rest)) < 3:
                return f"Not run (no step used): this store's search does not find model or fit codes ({', '.join(hits)} found nothing before); search the item's kind and values, then read the codes on its rows."
            turn.args = {**turn.args, key: rest}
            self.trace.append({'kind': 'code_free', 'dropped': hits})
            turn.extra = f"\n[Kyo] This store's search does not find model or fit codes (an earlier search with one found nothing), so Kyo searched {_dump(rest)} without {', '.join(hits)}; the codes are on the listings' pages." + turn.extra
            return ''
        return ''

    def offset_arg(self, name: str) -> str | None:
        """The search tool's start argument: the one the store's next-page value goes into (learned from a result's
        shape), else its only optional whole-number argument that does not name a size or a limit."""
        props = (self.schemas.get(name) or {}).get('properties') or {}
        shape = self.shapes.get(name) or {}
        if shape.get('next_arg') in props:
            return shape['next_arg']
        loose = [k for k, v in props.items() if isinstance(v, dict) and 'integer' in _kinds(v) and k not in ((self.schemas.get(name) or {}).get('required') or []) and (not re.search('size|limit|count|per|max|num|top|rows|results|amount|quantity|budget|price', k, re.I))]
        return loose[0] if len(loose) == 1 else None

    def fresh_offset(self, turn: _RwCall) -> None:
        """A new search starts at the first result: a start value the walker gives on a query not searched before
        would skip the first rows (it often reads that argument as a result count), so Kyo leaves it out; paging on
        an earlier query keeps it."""
        if turn.name not in {self.roles.get('search'), (self.browse or {}).get('tool')} - {None}:
            return
        key = self.offset_arg(turn.name)
        rest = {k: v for k, v in turn.args.items() if k != key}
        seen = getattr(self, 'query_seen', None)
        if seen is None:
            seen = self.query_seen = set()
        mark = turn.name + _dump(rest)
        value = turn.args.get(key) if key else None
        fresh = mark not in seen
        seen.add(mark)
        if not key or not fresh or isinstance(value, bool) or (not isinstance(value, int)) or value <= 0 or key in ((self.schemas.get(turn.name) or {}).get('required') or []):
            return
        turn.args = rest
        self.trace.append({'kind': 'fresh_offset', 'arg': key, 'dropped': value})
        turn.extra = f'\n[Kyo] {key} is where the results start (the next page value goes there), so a new search starts at the first result: Kyo left out {key}={value}.' + turn.extra

    def _rw_reply(self, call: dict[str, Any], usable: set[str], together: int) -> str:
        """What the walker reads back for one call: why Kyo did not run it, or what the store returned."""
        if call.get('joined'):
            return 'Shown in the side-by-side result above: Kyo opened these listings together in one step.'
        fn = call.get('function') or {}
        turn = _RwCall(fn.get('name'), together)
        why = self._rw_unrunnable(turn.name, usable) or self._rw_take_args(turn, fn.get('arguments'))
        if why:
            return why
        self._rw_trim_compare(turn)
        self.fresh_offset(turn)
        why = self.code_free(turn)
        if why:
            return why
        for stage in (self._rw_screen, self._rw_hold_ending, self._rw_hold_decline, self._rw_gate_ending):
            why = stage(turn)
            if why:
                return why
        reply = self.act(turn.name, turn.args)
        if reply is None:
            return 'The call failed (network or server). You may try again.'
        return self._rw_result(turn, reply)

    def _rw_unrunnable(self, name: Any, usable: set[str]) -> str:
        if self.finished:
            return 'Not run: the episode is over.'
        if name == self.roles.get('ask'):
            return "Not run (no step used): Kyo's interviewer handles every message to the shopper."
        listed = ', '.join(sorted(usable))
        if name not in self.schemas:
            return f'Not run (no step used): unknown tool. Tools: {listed}.'
        if name not in usable:
            return 'Not run (no step used): too few steps are left for that; use: ' + listed + '.'
        return ''

    def _rw_take_args(self, turn: _RwCall, raw: Any) -> str:
        """Parses and checks the call's arguments into turn.args; a reason when they cannot be used."""
        given = raw if isinstance(raw, dict) else (_json_obj(raw) if (raw or '').strip() else {})
        if given is None:
            return 'Not run (no step used): the arguments were not one JSON object.'
        faults: list[str] = []
        props = (self.schemas[turn.name] or {}).get('properties') or {}
        long = {k: props[k]['maxLength'] for k, v in given.items() if isinstance(v, str) and isinstance(props.get(k), dict) and isinstance(props[k].get('maxLength'), int) and len(v) > props[k]['maxLength']} if isinstance(given, dict) else {}
        given = _shape(given, self.schemas[turn.name], 'args', faults)
        if faults:
            return 'Not run (no step used): ' + '; '.join(faults[:6])
        for key, size in long.items():
            told = getattr(self, 'cut_told', set())
            if (turn.name, key) in told or not isinstance(given.get(key), str):
                continue
            self.cut_told = told | {(turn.name, key)}
            self.trace.append({'kind': 'cut_text', 'arg': key, 'size': size})
            turn.extra = f"\n[Kyo] {key} takes at most {size} characters, so Kyo sent {_dump(given[key])} (cut at a whole word); put the words that pick out the listing first." + turn.extra
        turn.args, wrong = self.resolve_ids(turn.name, given)
        return 'Not run (no step used): ' + wrong + '.' if wrong else ''

    def _rw_trim_compare(self, turn: _RwCall) -> None:
        if turn.name == self.roles.get('compare'):
            turn.many = self.compare_target()
        ids = turn.group()
        if not isinstance(ids, list) or len(ids) <= self.compare_room() or not self.compare_bad:
            return
        room = self.compare_room()
        turn.extra = f"\n[Kyo] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, ids[room:]))}."
        turn.args = {**turn.args, turn.many: ids[:room]}

    def _rw_screen(self, turn: _RwCall) -> str:
        """Calls Kyo answers without a step: sold-out listings, nothing to take off, finishing too early or alongside
        other calls, repeats, and views that show nothing new."""
        name = turn.name
        turn.target = self.single_target(name)
        why = self._rw_gone_or_absent(turn)
        if why:
            return why
        turn.ending = self.is_buy(name, turn.args) or self.is_refuse(name, turn.args)
        if turn.ending:
            if turn.together > 1:
                return 'Not run (no step used): send the finishing call on its own, after reading your other results.'
        elif self.left() <= self.tail() and (self.roles.get('buy') or self.roles.get('refuse')):
            return "Not run (no step used): the last steps are kept for checking out (with the shopper's approval of the basket when it is due) or ending without a purchase."
        turn.mark = name + _dump(turn.args)
        if not turn.ending and name != self.roles.get('cart') and turn.mark in self.recent:
            return 'Not run (no step used): you just made this same call and nothing changed; its result is above.'
        turn.readers = set((self.roles.get(job) for job in ('search', 'open', 'offer', 'compare'))) - {None}
        if name not in turn.readers:
            return ''
        cached = self.read_cache.get(turn.mark)
        if cached and cached[0] == self.version:
            return f'Not run (no step used): the same call at step {cached[1]} returned this, and nothing has changed since:\n' + cached[2][:3000]
        return self._rw_nothing_new(turn)

    def _rw_gone_or_absent(self, turn: _RwCall) -> str:
        name, args, target = turn.name, turn.args, turn.target
        if target and name in (self.roles.get('open'), self.roles.get('offer')) and str(args.get(target)) in self.gone:
            return f'Not run (no step used): {args[target]} sold out at step {self.gone[str(args[target])]} and will not come back. Replace its cart line with another fitting listing.'
        if name != self.roles.get('cart') or not any(((_count(v) or 0) < 0 for v in args.values() if not isinstance(v, str))):
            return ''
        view = _dump(self.cart_view) if self.cart_seen_at >= self.cart_at else ''
        for value in args.values():
            if isinstance(value, str) and value in self.rows and value not in self.cart_lines and value not in view and value not in self.cart_said:
                return f'Not run (no step used): {value} is not in the cart, so there is nothing to take off. Cart now: ' + self.cart_text()
        return ''

    def _rw_nothing_new(self, turn: _RwCall) -> str:
        """For a reading tool: Kyo's answer when the listings asked for showed everything already; drops such
        listings from a compare call."""
        name, args = turn.name, turn.args
        if turn.target and name != self.roles.get('compare') and self.shown_already(name, str(args.get(turn.target))):
            return f'Not run (no step used): {args[turn.target]} has shown everything this view gives since the last change. Its record: ' + self.record_of(str(args[turn.target]))
        ids = turn.group()
        if not isinstance(ids, list) or not ids:
            return ''
        keep = [i for i in ids if not self.shown_already(name, str(i))]
        if not keep:
            return 'Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n' + '\n'.join((f'{i}: {self.record_of(str(i))}' for i in ids))[:4000]
        if len(keep) < len(ids):
            turn.extra += f"\n[Kyo] Left out {', '.join((str(i) for i in ids if i not in keep))}: shown in full already, nothing changed since."
            turn.args = {**args, turn.many: keep}
            turn.mark = name + _dump(turn.args)
        return ''

    def _rw_hold_ending(self, turn: _RwCall) -> str:
        if not turn.ending or self.left() <= 2 or self.time_left() <= CALL_RESERVE + 25:
            return ''
        if self.is_buy(turn.name, turn.args):
            return self._rw_order_holds()
        return self._rw_decline_review() if self.left() > 5 else ''

    def _rw_wait(self, why: str) -> str:
        return 'Not run yet (no step used). ' + why + '\nCart now: ' + self.cart_text()

    def _rw_order_holds(self) -> str:
        """Checks before an order goes out; the first one that holds it back gives the walker's reply."""
        wider = self.wider_hold()
        if wider:
            return wider + '\nCart now: ' + self.cart_text()
        reread = self.reread_offers()
        if self.note_sold:
            self.find_sold()
        why = self._rw_cart_changed() or self._rw_sold_lines()
        if why:
            return why
        chain, hold = self.trail()
        if hold:
            return self._rw_wait(chain)
        self.referred_pages()
        why = self._rw_misfit() or self._rw_counts()
        if why:
            return why
        read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()
        verdict = self.check_cart()
        if verdict:
            seen = "Kyo read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n' if read else ''
            return 'Not run yet. ' + chain + seen + verdict + '\nCart now: ' + self.cart_text()
        if reread and self.cost > SPEND_SOFT:
            return "Not run yet (no step used). Kyo re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'
        return ''

    def _rw_cart_changed(self) -> str:
        viewer = self.roles.get('cart_view')
        if not (self.cart_touched and self.cart_seen_at < self.cart_at and viewer and self.left() > 3):
            return ''
        faults: list[str] = []
        view_args = _shape({}, self.schemas.get(viewer), 'args', faults)
        if faults:
            return ''
        view = self.act(viewer, view_args)
        if view is None:
            return ''
        self.cart_view = view.get('observation')
        self.cart_seen_at = self.step
        change = self.observe(viewer, self.cart_view)
        if not change:
            return ''
        self.queued = []
        return 'Not run yet (no step used). Looking at the cart showed a change: ' + change + '\nCart now: ' + self.cart_text() + '\nApply the change, fix the cart if needed, then check out.'

    def _rw_sold_lines(self) -> str:
        swap = self.swap_hint()
        held = [i for i in self.cart_ids() if i in self.gone] if self.held < 2 else []
        if not swap and not held:
            return ''
        if not swap:
            self.held += 1
        return self._rw_wait(swap or f"{', '.join(held)} in the cart is sold out: replace that line first.")

    def _rw_misfit(self) -> str:
        misfit = self.fit_faults() if self.fit_holds < 2 and self.left() > self.tail() + 3 else []
        if not misfit:
            return ''
        self.fit_holds += 1
        self.trace.append({'kind': 'fit_hold', 'faults': misfit[:4]})
        return self._rw_wait('A part in the cart does not work with the device in the cart: ' + '; '.join(misfit[:3]) + '. Swap the part (or the device), then check out.')

    def _rw_counts(self) -> str:
        faults = self.line_faults() if self.line_holds < 2 and self.left() > self.tail() + 2 else []
        if faults and all((' is held at ' in f for f in faults)):
            fixed = self.fix_counts(self.count_fixes())
            if fixed:
                faults = self.line_faults()
                if not faults:
                    self.line_holds += 1
                    return self._rw_wait("Kyo set the cart's quantities itself: " + '; '.join(fixed) + '. If the store refused any of these, fix those lines; then check out.')
        if not faults:
            return ''
        self.line_holds += 1
        self.trace.append({'kind': 'line_hold', 'faults': faults[:6]})
        return self._rw_wait('The cart does not match the request yet: ' + '; '.join(faults[:6]) + ". Fix those lines (the cart tool's own way to add or take off units), then check out.")

    def _rw_decline_review(self) -> str:
        review = self.audit()
        if review:
            return 'Not run yet (no step used). ' + review
        if not self.leads or self.left() <= 4:
            return ''
        if self.refuse_blocks == 0:
            self.refuse_blocks = 1
            self.trace.append({'kind': 'refuse_blocked', 'leads': list(self.leads)})
            return 'Not run yet (no step used). The review found listings that appear to fit: ' + ', '.join(self.leads) + '. Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again.'
        if self.refuse_blocks != 1:
            return ''
        self.refuse_blocks = 2
        first = self.leads[0]
        self.check_cart([first], SPEND_HARD)
        if not self.verdicts.get(self.check_key([first])) or first not in self.leads:
            return ''
        self.confirmed = first
        self.trace.append({'kind': 'refuse_blocked', 'confirmed': first})
        return f"Not run yet (no step used). An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."

    def _rw_hold_decline(self, turn: _RwCall) -> str:
        """A decline late in the episode, with steps to spare, or one the confirmed lead can replace."""
        if not self.is_refuse(turn.name, turn.args):
            return ''
        why = self._rw_late_decline() or self._rw_search_more()
        if why:
            return why
        done = self.complete_cart()
        if done:
            return done + (' [episode finished]' if self.finished else '')
        turn.args = self.quiet(turn.name, turn.args)
        return ''

    def _rw_late_decline(self) -> str:
        if self.late_held:
            return ''
        lines = [i for i in self.built if not self.sold_out(i)]
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        missing = [i for i in lines if i not in self.cart_ids() or (i in self.dropped and (not fresh))]
        if len(lines) < len(self.items) or len(missing) + 1 > self.left():
            return ''
        self.late_held = True
        self.trace.append({'kind': 'late_decline_held', 'lines': lines, 'missing': missing})
        redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else 'Check out the cart as it is.'
        why = 'running short of steps does not show that it has none' if self.left() <= 5 else 'every requested item already has a line you built'
        return 'Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and ' + why + '. ' + redo + ' End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again.'

    def _rw_search_more(self) -> str:
        if self.leads or self.pushed or self.left() <= self.max_steps // 3 or self.time_left() <= EXPLORE_RESERVE + 30:
            return ''
        self.pushed = True
        return f"Not run yet (no step used). {self.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."

    def _rw_gate_ending(self, turn: _RwCall) -> str:
        if not turn.ending or self.left() < 2 or self.time_left() <= CALL_RESERVE + 15:
            return ''
        go, said = self.gate('order' if self.is_buy(turn.name, turn.args) else 'end')
        if not go:
            return 'Not run yet. ' + said + '\nCart now: ' + self.cart_text()
        if said:
            turn.extra = '\n[Kyo] ' + said + turn.extra
        return ''

    def _rw_result(self, turn: _RwCall, reply: dict[str, Any]) -> str:
        """Learns from a call the store ran and writes what the walker reads about it."""
        name, args = turn.name, turn.args
        observation = reply.get('observation')
        self._rw_compare_seen(turn, observation)
        self.recent = (self.recent + [turn.mark])[-3:]
        for key, value in args.items():
            values = value if isinstance(value, list) else [value]
            if values and all((isinstance(x, str) and x in self.rows for x in values)):
                self.id_args.add((name, key))
        self.absorb(observation)
        if name == self.roles.get('search') and (not self.records_in(observation)) and any((isinstance(v, str) and self.fit_codes(v) for v in args.values())):
            self.code_blind = True
        if name == self.roles.get('compare'):
            rows = self.records_in(observation)
            if rows:
                self.facts_from[name] = self.facts_from.get(name, False) or any((_fact_block(r) for _, r in rows))
        if name in turn.readers - {self.roles.get('search')}:
            self.note_view(name, observation)
        plain = {tool for job, tool in self.roles.items() if job != 'search'}
        browsing = name not in plain and not turn.ending
        paged = self.more_pages(name, args, observation) if browsing else ''
        self._rw_learn_listing(name, observation)
        turn.extra += self.chain_offers(name, observation)
        if name == self.roles.get('cart'):
            self._rw_learn_cart(name, args, observation)
        if name == self.roles.get('search') and self.records_in(observation):
            self.searched = (self.searched + [dict(args)])[-8:]
        turn.extra += self.long_reads(name, observation, browsing)
        turn.extra += self.code_rows(name, observation, browsing)
        if name == self.roles.get('cart_view'):
            self.cart_view = observation
            self.cart_seen_at = self.step
        if self.is_buy(name, args):
            self.buys += 1
            if not reply.get('finished'):
                self.refused_at = self.step
        parts = [self.show(observation), paged, turn.extra]
        if browsing:
            parts.append(self.cheapest_here([observation] + self.paged_obs))
            parts.append(self.cap_ignored(name, args, observation))
        base = ''.join(parts)
        parts = [base]
        proof = self.evidence(observation)
        if proof:
            parts.append('\n' + proof)
        paired = self.pairing_notes(observation)
        if paired:
            parts.append('\n' + paired)
        if name == self.roles.get('cart') and self._rw_zero_kept(args, observation):
            parts.append('\n[Kyo] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.')
        self.note_sold = False
        notes = self.loose_lines(observation)
        meaning = self.take_note(notes) if notes else ''
        if meaning:
            parts.append('\n[note read] ' + meaning + '. Apply this change to your picks before checking out.')
        if getattr(self, 'price_note', False):
            self.price_note = False
            parts.append(self.price_recheck())
        parts.append(self._rw_resold(name, args, reply, observation))
        self.note_sold = False
        swap = self.swap_hint()
        if swap:
            parts.append('\n' + swap)
        if self.queued:
            parts.append("\n[notes seen on Kyo's own calls] " + ' | '.join(self.queued))
            self.queued = []
        if name in turn.readers:
            self.read_cache[turn.mark] = (self.version, self.step, base)
        if reply.get('finished'):
            parts.append(' [episode finished]')
        return ''.join(parts)

    def _rw_compare_seen(self, turn: _RwCall, observation: Any) -> None:
        ids = turn.group()
        if not isinstance(ids, list) or len(ids) <= 1:
            return
        shown = _dump(observation)
        if any((str(i) in shown for i in ids)):
            self.compare_ok = max(self.compare_ok, len(ids))
            return
        self.compare_bad = min(self.compare_bad or 999, len(ids))
        turn.extra += f'\n[Kyo] The side-by-side view showed nothing for {len(ids)} listings: send at most {self.compare_room()} per call.'

    def _rw_learn_listing(self, name: str, observation: Any) -> None:
        """A single-listing view: whether it shows attributes, which tool opens listings and which shows offers."""
        if not self.single_target(name) or name in (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask')):
            return
        listing = self.ident(observation) if isinstance(observation, dict) else None
        if not listing:
            return
        facts = self.facts_from.get(name, False) or bool(_fact_block(observation))
        self.facts_from[name] = facts
        if name == self.roles.get('open') and not facts:
            busy = (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask'), self.roles.get('search'))
            other = next((t for t in self.schemas if t != name and self.single_target(t) and self.facts_from.get(t) is not False and t not in busy), None)
            if other:
                self.roles['open'] = other
        if not facts and name != self.roles.get('open') and not self.roles.get('offer'):
            self.roles['offer'] = name
        if name == self.offer_tool():
            self.offer_seen[listing] = self.step

    def _rw_learn_cart(self, name: str, args: dict[str, Any], observation: Any) -> None:
        self.note_cart(name, args)
        self.cart_said = _dump(observation)
        self.recent = []
        counts = [c for c in (_count(v) for v in args.values() if not isinstance(v, str)) if c is not None]
        for value in args.values() if counts else ():
            if not isinstance(value, str) or value not in self.rows:
                continue
            if counts[0] > 0:
                self.dropped.discard(value)
            elif not _in_rows(observation, value):
                self.dropped.add(value)
        lines = [i for i in self.cart_ids() if not self.sold_out(i) and i not in self.dropped]
        if len(lines) >= len(self.built):
            self.built = lines

    def _rw_zero_kept(self, args: dict[str, Any], observation: Any) -> bool:
        """A count of 0 sent for a listing the cart still shows."""
        if not any((_count(v) == 0 for v in args.values() if not isinstance(v, str))):
            return False
        return any((isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values()))

    def _rw_resold(self, name: str, args: dict[str, Any], reply: dict[str, Any], observation: Any) -> str:
        """After a refused order or a sold-out note: the cart's lines read again."""
        refused = self.is_buy(name, args) and (not reply.get('finished'))
        again = refused and any((line in self.spoken(observation) for line in self.sold_lines))
        loose = self.note_sold and (refused or (not self.note_named and (not self.is_buy(name, args))))
        if not (loose or again) or not self.cart_ids() or self.finished:
            return ''
        seen = self.find_sold()
        return "\n[Kyo re-read the cart's lines after the refusal]\n" + seen if seen else ''

    def finish(self, again: bool=False) -> str:
        if self.finished or self.broke or self.left() <= 0:
            return ''
        buy, refuse = (self.roles.get('buy'), self.roles.get('refuse'))
        self.proven_fill()
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if buy and (holding or (self.cart_touched and (not fresh) and (self.left() >= 2))):
            faults: list[str] = []
            args = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), 'args', faults)
            if not faults:
                self.referred_pages()
                if self.left() >= 3 and self.time_left() > CALL_RESERVE + 20:
                    self.reread_offers(True)
                if self.left() >= 2 and self.time_left() > CALL_RESERVE + 15:
                    self.gate('order')
                if self.finished or self.left() <= 0:
                    return ''
                self.show_lines()
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
                if self.left() >= 2 and self.time_left() > CALL_RESERVE + 15:
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
        opening = _unpack(self.view.get('opening'))
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
            if turned and (not self.finished) and (not self.broke) and (not self.no_credit) and (self.left() > 1) and (self.time_left() > EXPLORE_RESERVE):
                self.chat.append({'role': 'user', 'content': '[Kyo] Kyo tried to end the episode and ' + turned + '\nApply what the store says, then finish.\n' + self.status()})
                self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
            if not self.finished:
                self.finish()
        self.trace.append({'kind': 'end', 'finished': self.finished, 'steps': self.step, 'cost': round(self.cost, 5), 'calls': self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.finished and (not self.broke) and (self.left() > 0) and (self.loop_calls < loop_cap) and (self.cost < spend_cap) and (self.time_left() > EXPLORE_RESERVE):
            if self.ready_to_ask():
                summary = self.interview()
                early = self.early_chains()
                if summary or early:
                    self.chat.append({'role': 'user', 'content': (summary + '\n' if summary else '') + early + self.status()})
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
            if not self.asked and self.roles.get('ask') and (self.msgs_free() > 0) and any((c['function']['name'] in committing for c in calls)) and (self.left() > 8):
                self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
                for call in calls:
                    self.chat.append({'role': 'tool', 'tool_call_id': call['id'], 'content': 'Not run (no step used): the shopper is asked first.'})
                summary = self.interview()
                early = self.early_chains()
                self.chat.append({'role': 'user', 'content': (summary or 'The interviewer had nothing to ask.') + '\n' + early + self.status()})
                continue
            self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
            self.trace.append({'kind': 'turn', 'text': content[:300], 'calls': [c['function'] for c in calls]})
            replies = self.run_calls(calls)
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
