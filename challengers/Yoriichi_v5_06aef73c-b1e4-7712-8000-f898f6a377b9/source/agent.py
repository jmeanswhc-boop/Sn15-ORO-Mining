from __future__ import annotations
import concurrent.futures
import json
import random
import re
import time
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
NEGATED = re.compile('^\\s*(definitely\\s+)?(anything\\s+)?(not\\s+other\\s+than|not|other than|apart from|except|but|never|without)\\s+', re.I)
POLARITY = 'Which does the shopper mean? Read the answer to its end: the words that close the sentence say whether the value is wanted or turned down (turning down everything other than a value means that value is wanted; "never" or "not once" means it is turned down). Reply with one letter only.\nA: the PROP must be VALUE (every other value is turned down).\nB: the PROP must not be VALUE (any other value is fine).'
BARE_VALUE = 'A shopper answered:\n"ANSWERS"\n\nA reader noted this need: "NEED".\nCopy the value itself, word for word from the noted value, and nothing else. Words that only join the property to its value are not part of it ("color showing red", "width reading at most 40 cm" and "material coming in as oak" give "red", "at most 40 cm" and "oak"), nor is a figure of speech or a remark that only stresses how firm the shopper is. Keep every word of the value\'s own name and every comparison, limit, negation, number and unit.\nReturn ONE JSON object and nothing else: {"value": the value, copied word for word from the noted value}'
VALUE_KEEP = re.compile('\\d|\\b(?:not|no|never|other|than|unlike|least|most|under|over|below|above|less|more|between|within|up|or|only|except|without|any|either|exactly|max|min|minimum|maximum)\\b', re.I)
EMPTY_RETRY_TOKENS = 8000
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
SHORTENED = '...(older result shortened; listings in it: '
GATE_ROUNDS = 3
AUTO_PAGES = 6
REF_TURNS = 40
REF_CALLS = 45
REF_SPEND = 0.08
CHEAPER_OPENS = 3
SCOUT_TURNS = 16
PREOPEN = "You are Kyo's scout in an online store. Before a shopper is asked about what the request below leaves open, you search the store for each requested item and open listings in full, so the interviewer can see which properties these listings carry and which values they take, and so the buyer starts from good candidates. You only search and open: you never fill a cart, buy, end the episode or message the shopper.\n\n- Tool names, argument names and result shapes are particular to this store: read each tool's description and schema and use the exact argument names and allowed values.\n- Search each requested item in its category with short queries (the product noun, or one value the request states plus the noun). A filtered search that returns no rows usually means the filter value is wrong: retry with the label exactly as result rows show it.\n- Open listings of the requested kind (the category label that matches the item as a whole) whose rows do not break what the request already states: several for each item, varied in shop, price and the properties the request leaves open, not near copies. A side-by-side tool, when there is one, opens several listings in one step.\n- Every call is one step of the shopper's limited episode: never repeat a call, and do not open a listing whose row shows another kind of product.\n- Kyo stops you when enough listings are open. Keep your text short."
FIT_CHECKS = 30
FIT_SECONDS = 150
ROWS_SHOWN = 120
SEEK_SECONDS = FIT_SECONDS - 20
SEEK_SPEND = 0.7 * SPEND_SOFT
SEEK_HOLDS = 6
SPARE_ROUNDS = 3
SPARE_OPENS = 3
SPARE_SECONDS = 220
FIT_RULES = 'RULES\n- The kind of product is decided by the listing\'s category label: only the label that matches the requested item as a whole counts, never a neighbouring one.\n- A requirement on a property is met when the listing\'s attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value\'s words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.\n- An attribute whose key names the requirement\'s property (under that name or a plain synonym) decides: if it says something else, the requirement is not met whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing about the requirement. Two keys with the same words in the same order, differing only in spaces, underscores, hyphens or letter case, are one attribute: the requirement is met when either of them states the value. Keys whose words differ are separate keys, and each decides as above.\n- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures. When nothing in the record states the property, the requirement is not shown.\n- A requirement that the product fits or works with a model named by a code (letters and digits) is decided by the listing\'s fitment statement (fits, compatible with, works with; in any field), which names the codes the store says it fits: an attribute that names a vehicle or device in other words does not fail it. A part also fits a device when the device\'s page names a series it belongs to (for example "takes part in series AB3") and the part\'s own code tag starts with that series code (for example "[AB3-1234]"); a part whose tag carries a different series code does not fit it. A code written with placeholder characters (every ABC 12xx model) covers every code it matches (ABC 1200 to ABC 1299), except a code the statement excludes.\n- When the category label itself states a requirement\'s value (a season or audience in the label), that requirement is met by the category.\n- A requirement that excludes a value ("not V", "anything but V") is met by any stated value other than V. A closed list ("one of A, B or C") is met only by a listed value.\n- A required value that names an absence (none, no such part, not applicable) is a value like any other, never met by default: it is met only when the record states that value for the property, by the word rules above (another word for nothing is a different value); when nothing in the record states the property, the requirement is not shown.'
FIT_PROMPT = 'You check one store listing against a shopper\'s firm requirements for one item. Judge only from the listing\'s record below, never from what such products usually are.\n\nITEM: {item} (store category label: {label})\nFIRM REQUIREMENTS:\n{reqs}\n\nLISTING RECORD:\n{record}\n\n{rules}\n\nFor each requirement give its status (met, not met or not shown) and the record\'s own words that decide it (the key and its value, or the phrase).\nReply with ONE JSON object and nothing else: {{"kind": "yes" or "no", "checks": [{{"n": 1, "status": "met", "evidence": "..."}}]}}'
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
GUIDE = 'You are Kyo, a careful buyer working for one shopper in an online store. You can act only by calling the store\'s tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.\n\nHOW THE STORE WORKS\n- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool\'s description and schema and use the exact argument names and allowed values.\n- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.\n- Search rows are summaries. Stores split a listing\'s record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Kyo knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.\n- Questions to the shopper are handled for you by Kyo\'s interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.\n- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Kyo carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store\'s tools (product questions go through Kyo\'s interviewer).\n\nWHAT COUNTS AS A FIT\n- The kind of product is decided by the listing\'s category label: only the label that matches the requested item as a whole counts, never a neighbouring one.\n- A requirement on a property is met when the listing\'s attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value\'s words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.\n- An attribute whose key names the requirement\'s property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing about the requirement. Kyo adds [evidence] lines after opened listings with the keys whose names match a requirement.\n- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.\n- A requirement that the product fits or works with a model named by a code (letters and digits) is decided by the listing\'s fitment statement (fits, compatible with, works with; in any field), which names the codes the store says it fits: an attribute that names a vehicle or device in other words does not fail it. A part also fits a device when the device\'s page names a series it belongs to (for example "takes part in series AB3") and the part\'s own code tag starts with that series code (for example "[AB3-1234]"); a part whose tag carries a different series code does not fit it. A code written with placeholder characters (every ABC 12xx model) covers every code it matches (ABC 1200 to ABC 1299), except a code the statement excludes.\n- When the category label itself states a requirement\'s value (a season or audience in the label), that requirement is met by the category.\n- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Kyo resolves such a requirement itself as soon as it arrives, and the ledger then shows its value instead; when the ledger still shows one that points at another listing, resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing\'s value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.\n- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.\n- When one requested item must fit another (an accessory for a device), choose the two together from the listings\' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.\n- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.\n- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.\n\nCHOOSING\n- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Kyo reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.\n- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.\n- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.\n- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.\n- A request for several different items needs one cart line per item, each at its requested quantity.\n\nCHANGES WHILE YOU WORK\n- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool\'s result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.\n- Re-read a chosen listing\'s record shortly before checking out, because price and stock can change silently.\n\nFINISHING\n- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.\n- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request\'s wording for that ending, and when a remark is needed, keep it short and name no listing.\n- Every reply must call at least one tool. Keep your text short: what you know, then the next move.'
BRIEF = 'You prepare a shopping task for an agent. Read the shopper\'s request and the store\'s tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:\n\n"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request\'s words, "shelf": the store\'s category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request\'s own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property\'s value or limit outright; and never turn words that are part of what the product is called (a season or an audience named with the product) into musts: those belong to "name". A use or occasion the request makes a property of the product itself ("blazers for business wear", "a dress made for parties", "the occasion has to be party") IS a must on that property ("occasion: business"), unless the store\'s category label already names it; an event, person or setting the product is bought for ("to wear at his wedding", "for a colleague"), and how it should sit, feel or look in use, stay context. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.\n"budget": the most the shopper will pay in total as a number in the request\'s own unit, or null.\n"budget_unit": that unit in words (for example "cents"), or null.\n"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.\n"in_stock_only": true unless the request clearly accepts items that are not available.\n"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.\n"pick_measure": what is maximised or minimised for "most"/"least", else "".\n"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.\n"steps": how many steps, moves or tool calls the request allows (integer), or null.\n"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else []. That other listing is one the request itself picks out by its shop, its kind or its values. A must that only ties this product to another product of this same order, one the agent still has to choose (it must fit, work with or suit it), is never a lookup: it stays in musts, with the pair in "fits"; when a product has both kinds, only the must that points at the other listing is a lookup.\n"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.\n"wishes": a list of things the shopper would merely like.\n"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).\n"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper\'s approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.\n"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.\n"fits": a list of {"item": item number, "with": item number} for each requested item that the request says must work with, fit or suit ANOTHER requested item (the one the agent picks); [] when none.\n"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing\'s attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing\'s current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).\n"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.\n"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.\n'
PLAN = 'You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property; a property the shopper has no view on gets "don\'t mind", and one the shopper does not think about at all gets a puzzled reply, so every property named spends part of a message.\n\nYou get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item\'s category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.\n\nChoose the properties in this order:\n0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.\n1. Hinted properties. The request\'s story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product\'s looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.\nName a hinted property by a listed key of that item when one fits the hint (a hint about who will wear it: the key for gender or audience; about its looks: the listed key for colour, pattern or finish that most of its listings carry); a property no listing of the item carries comes after the item\'s three most carried keys.\n2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item\'s listings; once five or more listings were read, avoid keys seen on only one or two of them.\nSkip a property the request already settles for that item (a requirement names that property; a key whose values only repeat a required value, such as a pattern key showing the required decoration, is NOT settled), a second name for a property already chosen, a key whose values each list several purchase options at once, identifiers, model names, warranty and other logistics. Pack sizes, counts and quantities per pack are properties like any other, and a key marked [not the stated ...] is another property than that requirement, whatever words its name shares with it. Rank every item on its own: its hinted properties first, then its listed keys.\n"label" is the property\'s plain name: for a listed key, all of the key\'s words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property\'s name, no example values, no brackets, no second property.\nChoose up to COUNT for each item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story\'s words it comes from, or an empty string"}]}, each item\'s picks in rank order'
HEAR = 'A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "budget": new total limit as a number or null}\n- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3"). Words that only join the property to its value are not part of the value: "color showing red", "width reading at most 40 cm" and "material coming in as oak" give "red", "at most 40 cm" and "oak". A figure of speech that likens the answer to an everyday thing rather than to a product in the store ("like a kettle on the boil") is not part of the value and points at no listing.\n- The "item" of each entry is the item whose property the answer sets. When the answer names one of the listed items as the owner of that property ("the colour of the lamps"), use the number of that item, even when the question was about another item. A product named only inside the value (a listing the value points at, or a product the item must fit) does not change the item.\n- An answer can give the value by pointing at another listing (a value like, unlike, near, at least or at most the one on some shop\'s product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. The description of that listing can stand where the value would go, even worded as a request or a question of its own, so the sentence reads oddly ("at <the description>? or above"); it still states this need, so never skip it as unclear. Its value is the answer\'s own words copied exactly: the comparison words before the description, the whole description including what it says about a listing that page may name, and any comparison words after it (such as "or above"); never shorten or reword it. The properties describing the other product are not needs of this item.\n- Interpret the complete statement, including what the shopper rejects. State the accepted requirement: rejecting values that do not meet a condition means that condition is required. Keep every comparison, exclusion or limit inside that condition, including when it uses a reference listing.\n- "free": the property of each question the shopper said they do not mind about.\n- Folksy or roundabout wording that names a value still states that value.\n- An answer that only complains about the message (too many questions at once, take them one at a time, do the first and then the next) or asks to hear it again states no need, even when it says "the first" or "the second": those words point at the questions, never at the examples a question offered. The examples in a question are the agent\'s words; a need\'s value must be stated in the answer itself.'
NOTE = 'While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:\n{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request\'s unit, or null, "add": [{"item": item number, "need": "property: constraint", "points": true when its value is to be read from another listing (then the need keeps the note words for it exactly), else false}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}\nA replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note.'
CHECK = 'You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.\nReading rules: a requirement is met when the listing\'s attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value\'s words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement that the product fits a model named by a code is decided by the listing\'s fitment statement (fits, compatible with, works with; in any field), and an attribute naming a vehicle or device in other words does not fail it; a part also fits a device when the device\'s page names a series it belongs to ("takes part in series AB3") and the part\'s own code tag starts with that series code ("[AB3-1234]"), and a part tagged with a different series code does not fit it; a code written with placeholder characters (every ABC 12xx model) covers every code it matches, except a code the statement excludes; a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing\'s value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.\nWhen the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart\'s choice on that rule, the cart is wrong.\nReturn ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses.'
READING = 'the listing\'s attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value\'s words counts, such as "deep teal" for teal, limits include the edge); a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement that the product fits a model named by a code is decided by the listing\'s fitment statement (fits, compatible with, works with; in any field), and an attribute naming a vehicle or device in other words does not fail it; a part also fits a device when the device\'s page names a series it belongs to ("takes part in series AB3") and the part\'s own code tag starts with that series code ("[AB3-1234]"), and a part tagged with a different series code does not fit it; a code written with placeholder characters (every ABC 12xx model) covers every code it matches, except a code the statement excludes; a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing\'s value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit.'
AUDIT = 'A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: ' + READING + '\nRecords marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.\nReturn ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}'
UNMIX = 'A shopper\'s request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request\'s wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}'
TRAIL = 'A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing\'s page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement\'s chain from the requirement\'s own description, link by link, through the pages\' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).\nReturn ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing, "find": "for a link not found: search words (its shop and the values the description gives)"}], "done": true when the last link is an OPENED page that states the property\'s value itself (also when the requirement asks for a value other than or unlike that one: the page\'s value is what it is measured against), else false, "value": the property\'s value as stated on that last page, or ""}]}\nA record is a link only when it matches the whole description of that link: the shop or brand, the kind of product AND every property value the description gives (for example "a 10 pieces multipack" rules out the same shop\'s listing with 10 capsules and no multipack); when no record matches all of it, that link is not found yet, even if the same shop sells other listings of that kind. Give "find" for such a link: the shop\'s name and the description\'s values, as a few search words.\nList each chain\'s links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown.'
RESOLVE = 'You are Kyo\'s reference finder, working in an online store for a shopping agent. One firm requirement on the product being bought does not state its value: it points at another listing (for example "<property> no more than whatever the page of the <kind> sold by <shop> with <property> <value> gives, or wherever that page points"). Find that listing, open its page, follow the page wherever it sends the reader for this requirement\'s property, and report the value. You never buy, fill a cart or message the shopper; another agent does the shopping with what you report.\n\nFINDING A LISTING\n- The description names the listing by a few things: usually its shop or seller, its kind of product and some property values. Only a listing that matches ALL of them is the one: the shop must be the listing\'s shop or seller field (a word in a title is not its shop), the kind must match, and every value given must match. The same shop often sells look-alike listings that differ in one value: those are decoys.\n- Each listing of the chain has its own description: the requirement describes the first one, and each further one is described only by the page that sends the reader to it. Do not carry the values that identified an earlier listing, or the shopper\'s purchase constraints, into the next description unless that page itself carries them over.\n- The listing pointed at can be of any kind, often not the kind being bought: do not limit searches to the shopper\'s category.\n- Search with a few short words. When the current description gives values, search the shop name, the kind of product and the one value of that description that best sets this listing apart from the shop\'s look-alikes, not the shop and kind alone: a shop often sells many listings of one kind, and checking them can take many steps. Use only words of the current description; drop the value only when that search finds no fitting row. Check the shop field and title of every result row. When no row fits, page on with the exact token or offset the store gives, or search again with other words (shop and kind, kind and values, shop alone). A category filter only takes a label exactly as rows show it.\n- A listing\'s page often states the values that describe it. When the rows leave more than one candidate, open the page of the likeliest one first (a side-by-side tool, when there is one, opens several pages in one step); read a candidate\'s attributes only when its row and page do not show a value you must check.\n- After each call Kyo lists the rows seen from the shops named so far and which of them are not checked yet. Judge candidates only against the current link\'s description. A detail a row does not show (a title is the seller\'s own text) is unverified, not a mismatch. Open a plausible candidate\'s page, and read any identifying attributes it leaves out, before searching again or going back to less relevant rows. Reject a candidate only for evidence that contradicts this link; the shopper\'s purchase constraints do not apply to the reference listing. When no shown candidate is left, page on or change the query.\n- Every listing of the chain must be opened with a page tool; a search row or an attribute view does not count.\n\nFOLLOWING THE PAGE\n- Read the whole page. When it sends the reader to a further listing for THIS requirement\'s property ("for <property>, refer to ...", "<property> should be taken from ...", "rather than this item"), the value comes from that further listing instead: find it by the description the page gives (check the rows you already have first) and open its page the same way. Repeat until a page gives the value without sending you on.\n- A remark about a different property, about other versions or about where else the product is sold does not send you on for this property.\n- The value is what the last listing states for the requirement\'s property: on its page, or in its attributes when the page does not say it (then read its attributes too). Use a rule the page gives (which value goes with which) only where the requirement calls for it.\n\nSTEPS\nEvery store call is one step of the shopper\'s episode, and the shopping still needs steps afterwards. Use the fewest calls, never repeat a call, and do not open listings whose row contradicts the description.\n\nREPORT\nWhen done, call REPORT_NAME once:\n- found true: chain = the ids of the listings you opened, in order (the first is the listing the requirement describes, the last is the one that gives the value); value = what that last listing itself states for the property, in its own words; condition = the requirement restated as "property: constraint" on the product being bought: derive the complete condition from the full requirement and that value, applying each requested relation once and using a rule the page gives only where the requirement calls for it; state the allowed value or range itself, not a relation word that is already applied (same as gives the value; unlike or other than gives "not <value>"; no higher than gives "at most <value>"; within N of gives the range; goes with or compatible with, under the page\'s rule, gives the value the rule pairs with it).\n- found false: while store steps for this reference are left, Kyo asks you to keep looking first: check the rows already shown that could fit the description of the listing you are looking for now, then search with other words or page on. Report found false when the steps are used up, when that search finds no listing matching the whole description, when the matching listings or pages contradict each other on the value, when the evidence cannot determine the condition the full requirement asks for (same as, unlike and limits need no page rule), or when no steps are left; say in why what you tried, give in chain the listings of the chain you did confirm (in order) and in next where the last of them sends the reader for this property, in its words, so the shopping agent can go on from there.'
RCHECK = 'A helper searched a store for the listing a shopping requirement points at, opened a chain of listings and read a value. Check its report against the records below. Return ONE JSON object and nothing else: {"ok": true or false, "problem": "when not ok: what is wrong and what to look for or read instead, in one or two sentences", "value": "what the last listing itself states for the property", "condition": "the requirement restated as property: constraint on the product being bought: the complete condition derived from the full requirement and that value, each requested relation applied once, a rule the page gives used only where the requirement calls for it, stating the allowed value itself (same as gives the value, unlike gives not <value>, no higher than gives at most <value>)"}\nThe report is ok only when: (1) listing 1 matches the whole description the requirement gives: the shop or seller, the kind of product and every property value it names; (2) each further listing matches the whole description that the previous listing\'s page gives where it sends the reader on for this requirement\'s property; (3) the last listing\'s page does not send the reader on for this property; (4) the value is what the last listing states for this property (on its page or in its attributes) and the condition follows from the full requirement and that value with each requested relation applied once; a condition that skips, adds or repeats a relation, or keeps a relation word that is already applied, fails. A remark about a different property, about other versions or about another place to buy does not send the reader on. A described value that the records contradict fails the link; when the records do not show a described value at all, ok is false and the problem asks to read that listing\'s attributes. Judge only from the records.'
GATE = 'A shopping agent is about to STAGE. The shopper\'s request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product\'s properties do not do a step.\nWhen a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.\nReturn ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint", "points": true when its value is to be read from another listing (then the need keeps the reply words for it exactly), else false} for each firm requirement such a reply states]}'
SWAP = "A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: " + READING + '\nReturn ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line\'s item, at most 6], "reason": "one sentence"}'
SHAPE = 'A store tool returned the result below. Name its parts by the exact keys the result and the tool\'s parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing\'s price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}'
GUARD_KEYS = 3
GUARD_SHARE = 0.0
GUARD_GAP = 0.34
NOT_PROPERTY = '\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b'
CATCH_ALL = '^(?!.*\\b(?:types?|styles?|kinds?|shapes?)\\b).*\\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\\w*|accessor\\w*|extras?|misc\\w*)\\b'
WIDEN_EACH = 6
MULTI = 8
FIT_CODE = '[A-Z][A-Za-z]{0,5}[ -]?\\d{3,6}'
FIT_FAMILY = '[A-Z][A-Za-z]{0,5}[ -]?\\d{1,5}[xX]{1,4}'
PART_CODE = '[A-Z]{1,4}\\d{3,6}'

def _codes(text: Any) -> list[str]:
    return [re.sub('\\s+', ' ', c).strip().lower() for c in re.findall('\\b' + FIT_CODE + '\\b', str(text or ''))]

def _fit_marks(text: Any) -> dict[str, Any]:
    text = str(text or '')
    own = re.search('[\\[(]\\s*part\\s+(' + PART_CODE + ')\\s*[\\])]', text, re.I) or re.search('[\\[(]\\s*(' + PART_CODE + ')\\s*[\\])]', text, re.I)
    owns = list(dict.fromkeys(([own.group(1).upper()] if own else []) + [c.upper() for c in re.findall('[\\[(]\\s*(?:part\\s+)?(' + PART_CODE + ')\\s*[\\])]', text, re.I)]))
    codes: list[str] = []
    for hit in re.finditer('\\bmodel\\s*(?:no\\.?|nr\\.?|number|ref\\.?|reference|code|#)?\\s*[:.#]?\\s*(' + FIT_CODE + ')\\b', text, re.I):
        codes += _codes(hit.group(1))
    for hit in re.finditer('\\b(?:also sold as|also known as|sold also as)\\s+([^.;]+)', text, re.I):
        codes += _codes(hit.group(1))
    works: list[str] = []
    families: list[str] = []
    nots: list[str] = []
    for hit in re.finditer('\\b(?<!not )(?<!never )(?:works with|compatible with|fits|goes with|pairs with|for use with)\\s*:?\\s+([^.;]+)', text, re.I):
        if not re.search('\\bpart\\s+' + PART_CODE + '\\s+also\\s+$', text[max(0, hit.start() - 40):hit.start()], re.I):
            head = re.split('\\b(?:except|excluding|but not)\\b', hit.group(1), maxsplit=1, flags=re.I)
            works += _codes(head[0])
            families += [re.sub('\\s+', ' ', c).strip().lower() for c in re.findall('\\b' + FIT_FAMILY + '\\b', head[0])]
            if len(head) > 1:
                nots += _codes(head[1])
    also = [(hit.group(1).upper(), _codes(hit.group(2))) for hit in re.finditer('\\bpart\\s+(' + PART_CODE + ')\\s+also fits\\s*:?\\s*([^.;]+)', text, re.I)]
    more = re.search('\\b(?:more|details|see|read)\\s+(?:on|at|under|about)?\\s*(?:the\\s+)?part\\s+(' + PART_CODE + ')', text, re.I)
    for hit in re.finditer('\\b(?:(?:part|member)\\s+(?:in\\s+|of\\s+)?|belongs\\s+to\\s+)(?:the\\s+)?series\\s+([A-Z]{1,4}\\d{1,3})\\b', text, re.I):
        codes.append('series ' + hit.group(1).upper())
    for hit in re.finditer('\\[\\s*([A-Z]{1,4}\\d{1,3})-\\d{2,6}\\s*\\]', text):
        works.append('series ' + hit.group(1).upper())
    return {'own': own.group(1).upper() if own else '', 'owns': owns, 'codes': codes, 'works': works, 'families': families, 'nots': nots, 'also': also, 'more': more.group(1).upper() if more else ''}

def _family_has(family: str, code: str) -> bool:
    return len(family) == len(code) and family != code and all((a == b or (a == 'x' and b.isdigit()) for a, b in zip(family, code)))

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

def _plain_need(need: str) -> str:
    prop, sep, value = need.partition(': ')
    rest = re.sub('^matching\\s+', '', value.strip(), flags=re.I)
    return prop + ': ' + rest if sep and rest and rest != value.strip() else need


def _words(key: str) -> str:
    key = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(key))
    return re.sub('[\\s_.\\-]+', ' ', key).strip().lower()

def _same_label(a: str, b: str) -> bool:
    return re.sub('[^a-z0-9]+', '', str(a).lower()) == re.sub('[^a-z0-9]+', '', str(b).lower())

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

def _prop(need: str) -> str:
    return need.split(':', 1)[0].strip().lower() if ':' in need else need


def _seller_names(row: dict[str, Any]) -> list[str]:
    names = [[str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search(pat, str(k), re.I)] for pat in ('seller|shop|store|vendor|merchant', 'brand|maker')]
    return names[0] or names[1]


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
        if isinstance(value, dict) and len(value) >= 2:
            scalars = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and (not isinstance(v, bool))}
            if len(scalars) >= 2 and (best is None or len(scalars) > len(best)):
                best = scalars
    return best

_REC_FIELDS = re.compile('\\s+\\|\\s+|\\s*\\n\\s*')
_REC_KEY = re.compile('^([A-Za-z][A-Za-z0-9_ ./-]{0,39}?)\\s*:\\s+(.*)$', re.S)
_SUB_KEY = re.compile('^([A-Za-z][A-Za-z0-9_ ./-]{0,39}?)\\s*(?:=|:\\s)\\s*(.*)$', re.S)
_NUMBER = re.compile('-?(?:0|[1-9]\\d*)(?:\\.\\d+)?(?:[eE][-+]?\\d+)?')

def _literal(text: str) -> Any:
    word = text.strip()
    if word in ('True', 'true'):
        return True
    if word in ('False', 'false'):
        return False
    if word in ('None', 'null'):
        return None
    if _NUMBER.fullmatch(word):
        return json.loads(word)
    return word

def _keyed(text: str, split: Any, key: Any, least: int, glue: str, share: float=0.0) -> dict[str, Any] | None:
    out: dict[str, Any] = {}
    last = None
    keyed = loose = 0
    for part in split(text):
        found = key.match(part.strip())
        if found and len(found.group(1).split()) <= 4 and found.group(1).strip() not in out:
            last = found.group(1).strip()
            out[last] = found.group(2).strip()
            keyed += 1
        elif last is None:
            return None
        else:
            out[last] = out[last] + glue + part.strip()
            loose += 1
    return out if keyed >= least and keyed >= share * (keyed + loose) else None

def _records(value: Any, depth: int=0) -> Any:
    if depth > 8:
        return value
    if isinstance(value, dict):
        whole = len(value) >= 3 and _learn_id([value]) is not None
        return {k: v if whole and isinstance(v, str) else _records(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_records(v, depth + 1) for v in value]
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text[:1] in '{[' and text[-1:] in '}]':
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        if isinstance(parsed, (dict, list)):
            return _records(parsed, depth + 1)
    record = _keyed(text, _REC_FIELDS.split, _REC_KEY, 3, ' | ')
    if record is None:
        return value
    for k, v in record.items():
        block = _keyed(v, lambda s: re.split(';\\s*', s), _SUB_KEY, 2, '; ', 0.8)
        record[k] = {a: _literal(b) for a, b in block.items()} if block else _literal(v)
    return record

class Item:

    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []

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
        self.refused_at = -1
        self.leads: list[str] = []
        self.refuse_blocks = 0
        self.scout_chat: list[dict[str, Any]] = []
        self.scout_opened: set[str] = set()
        self.search_log: list[tuple[str, dict[str, Any], Any]] = []
        self.pushed = False
        self.built: list[str] = []
        self.dropped: set[str] = set()
        self.late_held = False
        self.spare_rounds = 0
        self.trimmed_size = 0
        self.lead_items: dict[str, list[str]] = {}
        self.verdicts: dict[str, bool] = {}
        self.confirmed: str | None = None
        self.completed = False
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
        self.trails = 0
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
        self.fit_seen: dict[str, dict[str, Any]] = {}
        self.fit_calls = 0
        self.fit_began = time.time()
        self.fit_base: tuple[dict[int, list[str]], Any] | None = None
        self.fit_holds_done: set[str] = set()
        self.seek_holds = 0
        self.seek_reads = 0
        self.seek_idle = False
        self.seek_gained = ''
        self.scouting = False
        self.main_started = False
        self.paged_obs: list[Any] = []
        self.plan_order: list[int] = []
        self.one_key = False
        self.line_holds = 0
        self.bound: list[tuple[int, int]] = []
        self.fit_holds = 0
        self.pointer_codes: list[str] = []
        self.searched: list[dict[str, Any]] = []
        self.sought: dict[int, frozenset[str]] = {}
        self.alloc_held: set[tuple[int, frozenset[str]]] = set()
        self.link_searches = 0
        self.link_seen: dict[str, list[str]] = {}
        self.ref_cost = 0.0
        self.ref_calls = 0
        self.resolving = False
        self.ref_conds: dict[str, str] = {}
        self.ref_chain: dict[str, list[str]] = {}
        self.ref_tried: dict[tuple[int, str], tuple[int, int]] = {}
        self.ref_prev: dict[tuple[int, str], dict[str, Any]] = {}
        self.ref_steps = 0
        self.ref_touched: set[str] = set()
        self.page_obs: dict[str, Any] = {}
        self.report_name = next((n for n in ('report_reference', 'report_reference_result', 'kyo_reference_report') if n not in self.schemas))

    def left(self) -> int:
        return self.max_steps - self.step

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
        return 2 if talk and self.msg_used < self.msg_limit and self.gate_due('order') else 1

    def judges(self, limit: float=SPEND_SOFT, heavy_first: bool=False) -> list[str]:
        if self.cost > SPEND_HARD:
            return []
        if self.cost <= limit and self.cost + HEAVY_ROOM <= SPEND_HARD:
            return [HEAVY, LIGHT] if heavy_first else [LIGHT, HEAVY]
        return [LIGHT]

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None=None) -> dict[str, Any] | None:
        for model in models:
            if self.resolving:
                if self.ref_calls >= REF_CALLS or self.ref_cost >= REF_SPEND:
                    return None
                self.ref_calls += 1
            elif self.calls >= ALL_CALLS:
                return None
            else:
                self.calls += 1
            tries = 0
            while True:
                message, status = self.call_model(model, messages, max_tokens, tools)
                if message is not None:
                    return message
                passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                if not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL:
                    break
                pause = WAITS[tries] * (0.8 + 0.4 * random.random())
                self.waited += WAITS[tries]
                tries += 1
                time.sleep(pause)
            if status == 402:
                self.no_credit = True
                return None
        return None

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
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
        result = _llm.post_verbose('/inference/chat/completions', body)
        data = result.data
        if isinstance(data, dict) and data.get('choices'):
            usage = data.get('usage') or {}
            if isinstance(usage.get('cost'), (int, float)) and self.resolving:
                self.ref_cost += float(usage['cost'])
            elif isinstance(usage.get('cost'), (int, float)):
                self.cost += float(usage['cost'])
            message = data['choices'][0].get('message') or {}
            if message.get('content') or message.get('tool_calls'):
                return (message, 200)
            if data['choices'][0].get('finish_reason') == 'length' and max_tokens < EMPTY_RETRY_TOKENS:
                # all of max_tokens went to reasoning and nothing was said: ask once more with room for the answer
                self.trace.append({'kind': 'llm_empty_retry', 'model': model, 'max_tokens': max_tokens})
                return self.call_model(model, messages, EMPTY_RETRY_TOKENS, tools)
            self.trace.append({'kind': 'llm_fail', 'model': model, 'status': 0})
            return (None, 0)
        error = result.error or {}
        status = error.get('status')
        if isinstance(data, dict) and isinstance(data.get('error'), dict):
            status = _count(data['error'].get('code'))
            error = {'body': data['error'].get('message')}
        self.trace.append({'kind': 'llm_fail', 'model': model, 'status': status, 'body': str(error.get('body') or '')[:200]})
        return (None, status)

    def ask_json(self, prompt: str, models: list[str], max_tokens: int=1800) -> dict[str, Any] | None:
        for model in models:
            message = self.think([{'role': 'user', 'content': prompt}], [model], max_tokens)
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
        for _ in range(4):
            result = _env.post_verbose('/environment/call', payload)
            if isinstance(result.data, dict):
                data = result.data
                if 'observation' in data:
                    data['observation'] = _records(data['observation'])
                self.step = data['step'] if isinstance(data.get('step'), int) else self.step + 1
                if data.get('finished'):
                    self.finished = True
                seen = data.get('observation')
                self.note_pages(name, seen)
                if name == self.roles.get('search') and isinstance(args, dict):
                    self.search_log.append((name, dict(args), seen))
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
            twin = next((m for m in merged if item.shelf.strip() and _words(m.shelf) == _words(item.shelf) and (not any((_words(a.split(':')[0]) == _words(b.split(':')[0]) and _words(a.partition(':')[2]) != _words(b.partition(':')[2]) for a in m.musts for b in item.musts)))), None)
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
            got = self.ask_json(prompt, [LIGHT, HEAVY], 4000) or {}
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
            needs = [self.ref_conds[m] + f' (Kyo read it on the page of {self.ref_chain[m][-1]}, already opened)' if m in self.ref_conds else m + (f' (the page it points at gives: {self.ref_values[m]})' if m in self.ref_values else '') for m in live if m not in told]
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
        if self.notes:
            text += '\n[changes so far] ' + ' | '.join(self.notes[-4:])
        board = self.fit_board()
        if board:
            text += '\n' + board
        seek = self.seek_line()
        if seek:
            text += '\n' + seek
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
                    inner = self.ident(value, True)
                    if inner:
                        found.append((inner, value))
                        continue
                    found += [(i, r) for v in value.values() if isinstance(v, list) for r in v if isinstance(r, dict) for i in [self.ident(r, True)] if i]
            return found
        if isinstance(observation, list):
            return [(i, r) for r in observation if isinstance(r, dict) for i in [self.ident(r, True)] if i]
        return []

    def page_of(self, observation: Any) -> tuple[str | None, Any]:
        if not isinstance(observation, dict):
            return (None, observation)
        ident = self.ident(observation)
        if ident:
            return (ident, observation)
        inner = [(i, v) for v in observation.values() if isinstance(v, dict) for i in [self.ident(v, True)] if i]
        return inner[0] if len(inner) == 1 else (None, observation)

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
        found = {i for seen in observations for i, _ in self.records_in(seen) if not (self.main_started and i in self.full)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ''
        return '\n[Kyo] buyable rows here' + (' not opened yet' if self.main_started else '') + ', cheapest first: ' + ', '.join((f'{i} ({p:g})' for p, i in sorted(rows)[:6]))

    def firm_needs(self, number: int) -> list[str]:
        item = self.items[number - 1]
        live = [m for m in item.musts + item.told if m not in item.dropped]
        needs = [self.ref_conds[m] if m in self.ref_conds else m + (f' (the page it points at gives: {self.ref_values[m]})' if m in self.ref_values else '') for m in live if not self.described(number, m) and (not self.tie_must(number, m))]
        return list(dict.fromkeys(needs))

    def tie_of(self, number: int) -> int:
        return next((m for p, m in self.bound if p == number and 1 <= m <= len(self.items)), 0)

    def tie_must(self, number: int, need: str) -> bool:
        main = self.tie_of(number)
        if not main:
            return False
        value = need.partition(':')[2] or need
        if not re.search('\\b(?:pick|picks|picked|choose|chose|chosen|select|selected)\\b', value, re.I):
            return False
        return bool(re.search(f'\\bitem {main}\\b', value, re.I)) or bool({w for w in _stem(self.items[main - 1].name) if len(w) >= 4} & _stem(value))

    def pair_mode(self, main: int) -> bool:
        parts = [p for p, m in self.bound if m == main]
        if not parts or not 1 <= main <= len(self.items):
            return False
        index = self.fit_index()
        codes = {c for i in self.item_ids(main) for c in (index['marks'].get(i) or {}).get('codes') or []}
        return bool(codes) and all((any((self.fit_hits(i, codes, index) for i in self.item_ids(p))) for p in parts))

    def pair_best(self, main: int) -> tuple[float, list[str]] | None:
        if not self.pair_mode(main):
            return None
        index = self.fit_index()
        parts = [(p, self.fit_fits(p)) for p, m in self.bound if m == main]
        best = None
        for host in self.fit_fits(main):
            codes = set((index['marks'].get(host) or {}).get('codes') or [])
            price = self.price_of(host)
            if not codes or price is None:
                continue
            total, pick = (price * self.items[main - 1].qty, [host])
            for p, fits in parts:
                part = next((i for i in fits if self.price_of(i) is not None and self.fit_hits(i, codes, index)), None)
                if part is None:
                    break
                total += self.price_of(part) * self.items[p - 1].qty
                pick.append(part)
            else:
                if best is None or total < best[0] - 1e-09:
                    best = (total, pick)
        return best

    def pair_host(self, number: int) -> str:
        main = self.tie_of(number)
        if not main or not self.pair_mode(main):
            return ''
        index = self.fit_index()
        hosts = [i for i in self.cart_ids() if (self.fit_seen.get(i) or {}).get('item') == main and (not self.sold_out(i)) and (index['marks'].get(i) or {}).get('codes')]
        if len(hosts) == 1:
            return hosts[0]
        best = self.pair_best(main)
        return best[1][0] if best else ''

    def pair_text(self, best: tuple[float, list[str]]) -> str:
        return ' + '.join((f'{i} (price {self.price_of(i):g})' for i in best[1])) + f' = {best[0]:g}'

    def pair_picks(self) -> dict[int, str | None]:
        out: dict[int, str | None] = {}
        for main in sorted({m for _, m in self.bound}):
            if not self.pair_mode(main):
                continue
            group = [main] + [p for p, m in self.bound if m == main]
            best = self.pair_best(main)
            for k, number in enumerate(group):
                out[number] = best[1][k] if best else None
        return out

    def cheaper_fits(self, ident: str, number: int) -> list[str]:
        price = self.price_of(ident)
        if price is None:
            return []
        fits = [i for i in self.fit_fits(number) if i != ident and self.price_of(i) is not None and self.price_of(i) < price - 1e-09]
        if any((m == number for _, m in self.bound)) and self.pair_mode(number):
            return []
        main = self.tie_of(number)
        if main and self.pair_mode(main):
            index = self.fit_index()
            hosts = [i for i in self.cart_ids() if (self.fit_seen.get(i) or {}).get('item') == main and (not self.sold_out(i))]
            codes = set((index['marks'].get(hosts[0]) or {}).get('codes') or []) if len(hosts) == 1 else set()
            fits = [i for i in fits if codes and self.fit_hits(i, codes, index)]
        return fits

    def pair_fault(self, main: int) -> str:
        best = self.pair_best(main)
        if not best:
            return ''
        group = {main} | {p for p, m in self.bound if m == main}
        lines = [i for i in self.cart_ids() if (self.fit_seen.get(i) or {}).get('item') in group and (not self.sold_out(i))]
        if len(lines) != len(group) or len({self.fit_seen[i]['item'] for i in lines}) != len(group) or any((self.price_of(i) is None for i in lines)):
            return ''
        total = sum((self.price_of(i) * self.items[self.fit_seen[i]['item'] - 1].qty for i in lines))
        if best[0] < total - 1e-09:
            name = self.items[main - 1].name
            return f"the cart's {name} and the parts that must fit it cost {total:g} together, but this set was read as fitting every requirement in force, with each part's fit codes naming that {name}'s codes, and is cheaper: {self.pair_text(best)}"
        return ''

    def fit_label(self, ident: str) -> str:
        cat_key = self.category_key()
        if not cat_key:
            return ''
        return str((self.full.get(ident) or {}).get(cat_key) or (self.rows.get(ident) or {}).get(cat_key) or '').strip()

    def fit_item(self, ident: str) -> int:
        label = self.fit_label(ident)
        if not label or ident in self.ref_touched or (not _fact_block(self.full.get(ident) or {})):
            return 0
        if any((item.shelf for item in self.items)):
            same = [n for n, item in enumerate(self.items, 1) if item.shelf and _same_label(item.shelf, label)]
            return same[0] if len(same) == 1 else 0
        return self.owner_of(label)

    def fit_due(self) -> bool:
        return not (not self.asked and self.roles.get('ask') and self.msgs_free() > 0 and self.left() > 8)

    def fact_ids(self, observation: Any) -> list[str]:
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
        return ids

    def fit_post(self, prompt: str) -> tuple[str, float]:
        body = {'model': LIGHT, 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 1500, 'temperature': 0}
        try:
            data = _llm.post_verbose('/inference/chat/completions', body).data
        except Exception:
            return ('', 0.0)
        if not isinstance(data, dict) or not data.get('choices'):
            return ('', 0.0)
        usage = data.get('usage') or {}
        cost = float(usage['cost']) if isinstance(usage.get('cost'), (int, float)) else 0.0
        content = (data['choices'][0].get('message') or {}).get('content')
        return (content if isinstance(content, str) else '', cost)

    def label_same(self, ident: str, number: int) -> bool:
        shelf = self.items[number - 1].shelf if 0 < number <= len(self.items) else ''
        label = self.fit_label(ident)
        return bool(shelf) and bool(label) and _same_label(shelf, label)

    def fit_read(self, text: str, count: int, by_label: bool=False) -> tuple[str, list[dict[str, Any]]]:
        got = _json_obj(text)
        if not isinstance(got, dict):
            return ('error', [])
        checks = [c for c in got.get('checks') or [] if isinstance(c, dict)]
        if (not by_label) and str(got.get('kind') or '').strip().lower() != 'yes':
            return ('kind', checks)
        if len(checks) < count:
            return ('error', checks)
        status = [str(c.get('status') or '').strip().lower() for c in checks]
        ordered = [next((c for c in checks if _count(c.get('n')) == number), checks[number - 1]) for number in range(1, count + 1)]
        if any((s == 'not met' for s in status)):
            return ('unfit', ordered)
        if all((s == 'met' for s in status)):
            return ('fit', ordered)
        return ('unclear', ordered)

    def fit_state(self, ident: str, needs: list[str] | None=None) -> str:
        seen = self.fit_seen.get(ident)
        if not seen:
            return ''
        if seen['kind'] is False:
            return 'kind'
        status = [seen['needs'].get(need, ('', ''))[0] for need in (self.firm_needs(seen['item']) if needs is None else needs)]
        if 'not met' in status:
            return 'unfit'
        if not all(status):
            return 'stale'
        return 'fit' if all((s == 'met' for s in status)) else 'unclear'

    def fit_line(self, ident: str) -> str:
        seen = self.fit_seen[ident]
        number = seen['item']
        name = f' (item {number}: {self.items[number - 1].name})' if len(self.items) > 1 else ''
        state = self.fit_state(ident)
        if state == 'kind':
            return f'{ident}{name}: does NOT fit - not the requested kind of product'
        head = {'fit': 'FITS every requirement in force', 'unfit': 'does NOT fit', 'unclear': 'unclear - the page does not show every requirement', 'stale': 'not yet checked against every requirement in force'}[state]
        parts = []
        for need in self.firm_needs(number):
            status, evidence = seen['needs'].get(need, ('not checked', ''))
            parts.append(f'{need}: {status}' + (f' ({evidence})' if evidence else ''))
        return f'{ident}{name}: {head}' + (' - ' + '; '.join(parts) if parts else '')

    def fit_checks(self, idents: list[str]) -> str:
        if self.resolving or self.no_credit or (not self.fit_due()):
            return ''
        jobs: list[tuple[str, int, list[str], str]] = []
        lines = []
        for ident in dict.fromkeys(idents):
            number = self.fit_item(ident)
            if not number or self.sold_out(ident):
                continue
            needs = self.firm_needs(number)
            seen = self.fit_seen.get(ident)
            if seen and seen['item'] != number:
                seen = None
            known = seen['needs'] if seen else {}
            if seen and (seen['kind'] is False or any((known.get(need, ('', ''))[0] == 'not met' for need in needs))):
                lines.append(self.fit_line(ident))
                continue
            missing = [need for need in needs if need not in known]
            if seen and (not missing):
                lines.append(self.fit_line(ident))
                continue
            if self.fit_calls >= FIT_CHECKS or self.cost >= SPEND_SOFT or time.time() - self.fit_began > FIT_SECONDS:
                if seen:
                    lines.append(self.fit_line(ident))
                continue
            self.fit_calls += 1
            item = self.items[number - 1]
            reqs = '\n'.join((f'{n}. {r}' for n, r in enumerate(missing, 1))) or '(none beyond the kind of product)'
            jobs.append((ident, number, missing, FIT_PROMPT.format(item=item.name, label=item.shelf or self.fit_label(ident), reqs=reqs, record=json.dumps(self.full[ident], ensure_ascii=True)[:4000], rules=FIT_RULES)))
        if jobs:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(jobs))) as pool:
                answers = list(pool.map(self.fit_post, [job[3] for job in jobs]))
            for (ident, number, missing, _), (text, cost) in zip(jobs, answers):
                self.cost += cost
                verdict, checks = self.fit_read(text, len(missing), self.label_same(ident, number))
                self.trace.append({'kind': 'fit_check', 'step': self.step, 'id': ident, 'item': number, 'verdict': verdict, 'needs': len(missing), 'cost': round(cost, 5)})
                if verdict == 'error':
                    continue
                seen = self.fit_seen.get(ident)
                if not seen or seen['item'] != number:
                    seen = {'item': number, 'kind': True, 'needs': {}}
                if verdict == 'kind':
                    seen['kind'] = False
                else:
                    for need, check in zip(missing, checks):
                        seen['needs'][need] = (str(check.get('status') or '').strip().lower() or '?', ' '.join(str(check.get('evidence') or '').split())[:90])
                self.fit_seen[ident] = seen
                lines.append(self.fit_line(ident))
        if not lines:
            return ''
        return "[fit check: Kyo's checker read each page against the item's requirements in force (price, stock and budget are not part of it)]\n" + '\n'.join(lines)

    def fit_snapshot(self) -> tuple[dict[int, list[str]], Any]:
        return ({n: self.firm_needs(n) for n in range(1, len(self.items) + 1)}, self.budget)

    def fit_opened(self) -> str:
        text = self.fit_checks([i for i in self.full if i not in self.ref_touched])
        if self.fit_due():
            self.fit_base = self.fit_snapshot()
        return text

    def fit_refresh(self) -> str:
        if self.fit_base is None or self.finished or (not self.fit_due()):
            return ''
        now = self.fit_snapshot()
        if now == self.fit_base:
            return ''
        before, budget = self.fit_base
        changes = []
        for number, needs in now[0].items():
            old = before.get(number, [])
            tag = f'item {number}: ' if len(self.items) > 1 else ''
            added = [need for need in needs if need not in old]
            gone = [need for need in old if need not in needs]
            if added:
                changes.append(tag + 'now required: ' + '; '.join(added))
            if gone:
                changes.append(tag + 'no longer required: ' + '; '.join(gone))
        if now[1] != budget:
            changes.append(f'budget now {now[1]}')
        prior = {ident: self.fit_state(ident, before.get(seen['item'], [])) for ident, seen in self.fit_seen.items()}
        self.fit_checks([i for i in self.full if i not in self.ref_touched])
        self.fit_base = self.fit_snapshot()
        names = {'fit': 'fits', 'unfit': 'does not fit', 'kind': 'does not fit', 'unclear': 'unclear', 'stale': 'not checked'}
        moved = []
        for ident in self.fit_seen:
            state = self.fit_state(ident)
            if not self.sold_out(ident) and state != prior.get(ident):
                moved.append(f"{ident} {names.get(prior.get(ident) or '', 'not checked')} -> {names[state]}")
        self.trace.append({'kind': 'fit_refresh', 'step': self.step, 'changes': changes, 'moved': moved})
        text = '[requirements changed] ' + (' | '.join(changes) or 'the wording of a requirement changed')
        return text + '\n[fit checks redone for the requirements in force] ' + ('; '.join(moved[:12]) or 'no verdict changed')

    def fit_board(self) -> str:
        parts = []
        for number, item in enumerate(self.items, 1):
            got: dict[str, list[str]] = {'fit': [], 'unclear': [], 'no': [], 'stale': []}
            for ident, seen in self.fit_seen.items():
                if seen['item'] != number or self.sold_out(ident):
                    continue
                state = self.fit_state(ident)
                if state == 'fit':
                    continue
                else:
                    got[state if state in ('unclear', 'stale') else 'no'].append(ident)
            got['fit'] = [i + (f' (price {self.price_of(i):g})' if self.price_of(i) is not None else '') for i in self.fit_fits(number)]
            if not any(got.values()):
                continue
            text = (f'item {number}: ' if len(self.items) > 1 else '') + 'fit ' + (', '.join(got['fit']) or 'none yet')
            if got['unclear']:
                text += '; unclear ' + ', '.join(got['unclear'][:6])
            if got['no']:
                text += '; not fitting ' + ', '.join(got['no'][:8])
            if got['stale']:
                text += '; not yet checked against every requirement in force ' + ', '.join(got['stale'][:6])
            parts.append(text)
        for main in sorted({m for _, m in self.bound}):
            best = self.pair_best(main)
            if best:
                parts.append(f"cheapest set read as fitting whose parts' fit codes name the {self.items[main - 1].name}'s codes: {self.pair_text(best)}")
        return '[fit checks so far; sold-out listings left out] ' + ' | '.join(parts) if parts else ''

    def fit_fits(self, number: int) -> list[str]:
        fits = [i for i, seen in self.fit_seen.items() if seen['item'] == number and (not self.sold_out(i)) and self.fit_state(i) == 'fit']
        return sorted(fits, key=lambda i: (self.price_of(i) is None, self.price_of(i) or 0.0))

    def fit_named(self, number: int, limit: int=5) -> str:
        fits = self.fit_fits(number)
        host = self.pair_host(number)
        if host:
            index = self.fit_index()
            codes = set((index['marks'].get(host) or {}).get('codes') or [])
            fits = [i for i in fits if self.fit_hits(i, codes, index)]
        names = [i + (f' (price {self.price_of(i):g})' if self.price_of(i) is not None else '') for i in fits[:limit]]
        tag = f'item {number} ({self.items[number - 1].name}' + (f', fit codes naming {host}' if host else '') + '): ' if len(self.items) > 1 else ''
        return tag + (', '.join(names) if names else 'none read as fitting yet')

    def fit_cheapest(self) -> list[str] | None:
        picks = []
        paired = self.pair_picks()
        for number in range(1, len(self.items) + 1):
            if number in paired:
                if paired[number] is None:
                    return None
                picks.append(paired[number])
                continue
            fits = self.fit_fits(number)
            if not fits:
                return None
            picks.append(fits[0])
        prices = [self.price_of(i) for i in picks]
        if isinstance(self.budget, (int, float)) and (not isinstance(self.budget, bool)):
            if any((p is None or p > 10 * self.budget for p in prices)):
                return None
            if self.brief.get('budget_scope') == 'each':
                if any((p > self.budget + 1e-09 for p in prices)):
                    return None
            elif sum((p * item.qty for p, item in zip(prices, self.items))) > self.budget + 1e-09:
                return None
        return picks

    def fit_cart_note(self, args: dict[str, Any]) -> str:
        counts = [_count(v) for v in args.values() if not isinstance(v, str) and _count(v) is not None]
        if not counts or counts[0] <= 0:
            return ''
        notes = []
        for ident in [v for v in args.values() if isinstance(v, str) and v in self.fit_seen]:
            number = self.fit_seen[ident]['item']
            state = self.fit_state(ident)
            fits = self.fit_fits(number)
            price = self.price_of(ident)
            cheaper = self.cheaper_fits(ident, number)
            if state in ('unfit', 'kind'):
                notes.append(f"{ident} was read as NOT fitting: {self.fit_line(ident).split(': ', 1)[1]}. Listings read as fitting, cheapest first: {self.fit_named(number)}")
            elif self.brief.get('pick') == 'cheapest' and cheaper:
                notes.append(f'a cheaper listing of that item was read as fitting every requirement in force: {self.fit_named(number)}')
        return '[fit list] ' + ' | '.join(notes) if notes else ''

    def fit_buy_hold(self) -> str:
        if self.left() <= self.tail() + 3:
            return ''
        key = self.cart_mark() + f'#{self.version}#' + _dump(self.fit_snapshot()[0])
        if key in self.fit_holds_done:
            return ''
        faults = []
        for ident in self.cart_ids():
            seen = self.fit_seen.get(ident)
            if not seen or self.sold_out(ident):
                continue
            number = seen['item']
            state = self.fit_state(ident)
            price = self.price_of(ident)
            if state in ('unfit', 'kind'):
                faults.append(f"{ident} was read as NOT fitting: {self.fit_line(ident).split(': ', 1)[1]}. Read as fitting, cheapest first: {self.fit_named(number)}")
            elif self.brief.get('pick') == 'cheapest' and price is not None:
                cheaper = self.cheaper_fits(ident, number)
                if cheaper:
                    faults.append(f'{ident} costs {price:g}, but {cheaper[0]} (price {self.price_of(cheaper[0]):g}) was read as fitting every requirement in force and is cheaper. Read as fitting, cheapest first: {self.fit_named(number)}')
        if self.brief.get('pick') == 'cheapest':
            for main in sorted({m for _, m in self.bound}):
                fault = self.pair_fault(main)
                if fault:
                    faults.append(fault)
        if not faults:
            return ''
        self.fit_holds_done.add(key)
        self.trace.append({'kind': 'fit_buy_hold', 'step': self.step, 'faults': [f[:160] for f in faults[:4]]})
        return "Not run yet (no step used). Kyo's fit checker read the opened listings against the requirements in force: " + ' | '.join(faults[:4]) + '. Fix those cart lines, then check out; if you are sure a line fits, check out again as it is.'

    def fit_decline_hold(self) -> str:
        key = f'decline#{self.version}#' + _dump(self.fit_snapshot()[0])
        if key in self.fit_holds_done:
            return ''
        picks = self.fit_cheapest()
        if not picks:
            return self.pair_gap_hold(key)
        if len(picks) + 1 > self.left():
            return ''
        self.fit_holds_done.add(key)
        self.trace.append({'kind': 'fit_decline_hold', 'step': self.step, 'set': picks})
        many = len(self.items) > 1
        named = '; '.join((self.fit_named(number, 3) for number in range(1, len(self.items) + 1)))
        return "Not run yet (no step used). Kyo's fit checker read " + ('a listing for every requested item' if many else 'a listing') + ' as meeting every requirement in force, in stock' + (', and the cheapest such set is inside the budget' if self.budget is not None else '') + f": {', '.join(picks)}. Read as fitting, cheapest first: {named}. Put the right one{'s' if many else ''} in the cart at the requested quantit{'ies' if many else 'y'} (with a most or least pick: the one that wins on it) and check out. End without a purchase only if you see that one of them breaks a requirement; then say which in your text and send the ending call again."

    def pair_leads(self, limit: int=4) -> list[tuple[str, str]]:
        index = self.fit_index()
        found: list[tuple[str, str]] = []

        def unopened(number: int) -> list[str]:
            return [i for i in self.item_ids(number) if i in self.rows and i not in self.full and self.buyable(i)]
        for main in sorted({m for _, m in self.bound}):
            if not self.pair_mode(main) or self.pair_best(main):
                continue
            parts = [p for p, m in self.bound if m == main]
            for host in self.fit_fits(main):
                codes = set((index['marks'].get(host) or {}).get('codes') or [])
                if not codes:
                    continue
                for p in parts:
                    if any((self.fit_hits(i, codes, index) for i in self.fit_fits(p))):
                        continue
                    for i in unopened(p):
                        hit = self.fit_hits(i, codes, index)
                        if hit:
                            found.append((i, f"its fit codes name {host}'s code {', '.join(sorted(hit)).upper()}"))
            for row in unopened(main):
                codes = set((index['marks'].get(row) or {}).get('codes') or [])
                if not codes:
                    continue
                named = [[i for i in self.fit_fits(p) if self.fit_hits(i, codes, index)] for p in parts]
                if all(named):
                    found.append((row, f"its code {', '.join(sorted(codes)).upper()} is named by {', '.join(n[0] for n in named)}"))
        out: dict[str, str] = {}
        for ident, why in found:
            out.setdefault(ident, why)
        ids = list(out)
        if self.brief.get('pick') == 'cheapest':
            ids.sort(key=lambda i: (self.price_of(i) is None, self.price_of(i) or 0.0))
        return [(i, out[i]) for i in ids[:limit]]

    def pair_gap_hold(self, key: str) -> str:
        leads = self.pair_leads()
        if not leads or len(leads) + self.tail() + 2 > self.left():
            return ''
        self.fit_holds_done.add(key)
        self.trace.append({'kind': 'pair_gap_hold', 'step': self.step, 'leads': [i for i, _ in leads]})
        named = '; '.join((i + (f' (price {self.price_of(i):g})' if self.price_of(i) is not None else '') + f': {why}' for i, why in leads))
        return "Not run yet (no step used). Kyo's fit checker has no set yet whose parts' fit codes name the device's codes, but these listings were seen only as search rows, never opened, and their fit codes would complete a set with a listing read as fitting: " + named + '. Open them and check their other requirements; if one fits, put the set in the cart at the requested quantities and check out. End without a purchase only if each of them breaks a requirement; then say which in your text and send the ending call again.'

    def seek_reserve(self) -> int:
        return self.tail() + 2 * len(self.items) + 1

    def seek_room(self) -> int:
        if self.finished or self.cost >= SEEK_SPEND or time.time() - self.fit_began > SEEK_SECONDS or self.loop_calls >= LOOP_CALLS - 8:
            return 0
        return max(0, self.left() - self.seek_reserve())

    def seek_gain(self) -> str:
        lines = [i for i in self.cart_ids() if not self.sold_out(i)]
        if len(lines) < len(self.items) or any((self.fit_state(i) in ('unfit', 'kind') for i in lines)):
            return ''
        pick = self.brief.get('pick')
        if pick == 'cheapest':
            return 'cheaper and still fitting every requirement'
        if pick in ('most', 'least'):
            return f"fitting every requirement with {('more' if pick == 'most' else 'less')} {self.brief.get('pick_measure') or 'of the measure'}".rstrip()
        if any((self.fit_state(i) != 'fit' for i in lines)):
            return 'read as fitting every requirement (a cart line is not read as fitting yet)'
        return ''

    def seek_hold(self) -> str:
        room = self.seek_room()
        gain = self.seek_gain()
        if room < 2 or not gain or self.seek_holds >= SEEK_HOLDS:
            return ''
        idle = self.seek_holds > 0 and self.seek_reads == 0
        if idle and self.seek_idle:
            return ''
        self.seek_idle = idle
        self.seek_holds += 1
        self.seek_reads = 0
        self.seek_gained = gain
        self.trace.append({'kind': 'seek_hold', 'step': self.step, 'room': room, 'idle': idle})
        if idle:
            return f'Not run yet (no step used). Nothing was searched or opened since the last note, and {room} free steps are still left: search and open more listings that may be {gain} first. If you are sure no better listing exists, say why in your text and check out again.'
        return f"Not run yet (no step used). {room} more steps are free for searching (the last {self.seek_reserve()} are kept for fixing the cart and checking out), and time and budget are left, so do not stop at the first set that fits: use those steps to search with other words (each requirement's value, the store's category label, other names for the product, further result pages) and open listings not opened yet that may be {gain}. Kyo's fit checker reads every page you open, and the fit list in the status shows what was read as fitting. When Kyo says the search is over, or nothing better turns up, put the best set in the cart and check out."

    def seek_line(self) -> str:
        if not self.seek_holds or self.finished:
            return ''
        room = self.seek_room()
        if room > 0:
            return f'[keep searching] {room} free steps left for searching and opening listings that may be {self.seek_gained} (the last {self.seek_reserve()} steps are kept for the cart and checkout).'
        return '[search over] put the best set read as fitting in the cart and check out now.'

    def wider_hold(self) -> str:
        if self.wider_held or self.brief.get('pick') != 'cheapest' or self.left() < 14 + 2 * len(self.items):
            return ''
        self.wider_held = True
        lines = [f'{i} at {self.price_of(i):g}' for i in self.cart_ids() if self.price_of(i) is not None]
        return f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest. Searches match a listing's title, so a requirement's value (often only in the attributes) finds few of the fitting listings, and rows come in no price order. Before checking out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line" + (f" ({'; '.join(lines)})" if lines else '') + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return []
        texts = []
        for ident in [i for i in ids if i in self.rows and i not in self.full and (not self.sold_out(i))]:
            if self.left() <= keep or self.finished or self.facts_from.get(tool) is False:
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
            listing, record = self.page_of(seen)
            if listing:
                self.facts_from[tool] = self.facts_from.get(tool, False) or bool(_fact_block(record))
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
            picks += [i for i in pool[:2] if i not in picks]
        return self.fill_facts(picks[:CHEAPER_OPENS], self.tail() + 4)

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
        if self.left() <= 8:
            return False
        opened = len([i for i in self.full if i not in self.ref_touched])
        if opened >= 3 * len(self.items) + 1:
            return True
        return self.step - self.ref_steps >= max(8, self.max_steps // 4)

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
        lanes = self.plan_lanes(min(8, self.msgs_free() * MULTI))
        found = {n: 0 for n in lanes}
        spent = {n: 0 for n in lanes}
        used: dict[int, set[str]] = {n: set() for n in lanes}
        follow: list[tuple[int, list[str], bool]] = []
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        pointers: dict[int, list[str]] = {}
        while self.msgs_free() > 0 and (not self.finished) and (self.left() > 6):
            if follow:
                number, labels, shown = follow.pop(0)
            else:
                number, labels = self.next_ask(lanes, found, spent, used)
                shown = False
                if not labels:
                    break
            used[number].update((_words(label) for label in labels))
            batch = self.send([(number, labels[0], self.ask_text(number, labels, shown))])
            if not batch:
                break
            sent += batch
            spent[number] += 1
            had = [len(item.told) for item in self.items]
            known = list(self.refs)
            heard = self.hear(batch, (LIGHT, HEAVY))
            self.polarity(heard, batch)
            self.bare_values(heard, batch)
            heard = self.named_drops(heard, batch)
            self.take_heard(heard, told)
            for index in range(len(sent) - len(batch), len(sent)):
                pointers[index] = [pair[1] for pair in self.refs if pair not in known]
            fresh = [n for n, item in enumerate(self.items, 1) if len(item.told) > had[n - 1]]
            for n in fresh:
                found[n] = found.get(n, 0) + 1
            if len(labels) >= 2 and (not self.one_key):
                props = [_stem(t.split(':', 1)[0]) for t in self.items[number - 1].told[had[number - 1]:]]
                hit = next((j for j, label in enumerate(labels) if any((_names(p, _stem(label)) or _names(_stem(label), p) for p in props))), 0 if props else 1)
                used[number] -= {_words(label) for label in labels[hit + 1:]}
            if fresh or 1 not in [_count(x) for x in _items(heard.get('crowded'))]:
                continue
            if len(labels) >= 2 and (not self.one_key):
                self.one_key = True
                used[number] -= {_words(label) for label in labels[1:]}
                follow.insert(0, (number, labels[:1], False))
                continue
            head = labels[0].split()[-1]
            if len(labels[0].split()) >= 2 and _words(head) not in used[number]:
                follow.insert(0, (number, [head], False))
            elif not shown and len(self.samples(number, labels[0])) >= 2:
                follow.insert(0, (number, labels, True))
        resolved = self.resolve_pending()
        found_items = [n for n in found if found[n] > 0]
        searched = self.value_search(found_items) if found_items and self.main_started else ''
        if told:
            self.version += 1
        lines = [f'Q: {q}\nA: ' + (self.answer_view(pointers[i]) if pointers.get(i) and all((p in self.ref_conds for p in pointers[i])) else a) for i, (_, q, a) in enumerate(sent)]
        summary = "Kyo's interviewer asked the shopper (each question used one step):\n" + '\n'.join(lines)
        summary += '\nNew firm requirements from these answers: ' + ('; '.join((self.told_view(t) for t in told)) if told else 'none')
        summary += '. Apply them to every pick (search again with the new values where needed).'
        pointed = [need for _, need in self.refs if any((need in t for t in told)) and need not in self.ref_conds]
        if pointed:
            summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.'
        if resolved:
            summary += '\n' + resolved
        if searched:
            summary += '\n' + searched
        if self.queued:
            summary += '\n[notes seen meanwhile] ' + ' | '.join(self.queued)
            self.queued = []
        return summary

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
        if not plan or self.left() < 22:
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
                self.stamp_search(number, found)
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

    def next_ask(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> tuple[int, list[str]]:
        first = {n: i for i, n in enumerate(self.plan_order)}
        for number in sorted(lanes, key=lambda n: (found.get(n, 0) > 0, spent.get(n, 0), first.get(n, len(first) + n))):
            props = [_stem(t.split(':', 1)[0]) for t in self.items[number - 1].told]
            left = [label for label in lanes[number] if _words(label) not in used[number] and (not any((_names(p, _stem(label)) or _names(_stem(label), p) for p in props)))]
            if left:
                return (number, left[:1] if self.one_key else left[:MULTI])
        return (0, [])

    def ask_text(self, number: int, labels: list[str], shown: bool=False) -> str:
        if len(labels) >= 3:
            return f"For the {self.items[number - 1].name}, what {', '.join(labels[:-1])} or {labels[-1]} would you like?"
        if len(labels) >= 2:
            return f'What {labels[0]} or {labels[1]} would you like for the {self.items[number - 1].name}?'
        return self.question(number, labels[0], self.samples(number, labels[0]) if shown else None)

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

    def label_owner(self, label: str) -> int:
        same = [n for n, item in enumerate(self.items, 1) if label and item.shelf and _same_label(item.shelf, label)]
        return same[0] if same else self.owner_of(label)

    def owner_of(self, label: str) -> int:
        if not label:
            return 0
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

    def search_owner(self, args: dict[str, Any]) -> int:
        text = ' '.join((v for v in args.values() if isinstance(v, str) and re.search('[a-z]{3}', v, re.I)))
        long = _roots(text)
        if not long or len(self.items) < 2:
            return 0
        scores = sorted(((len(long & (_roots(item.shelf or '') | _roots(item.name))) / max(1, len(long | _roots(item.shelf or '') | _roots(item.name))), n) for n, item in enumerate(self.items, 1)), reverse=True)
        return scores[0][1] if scores[0][0] > 0 and scores[0][0] > scores[1][0] else 0

    def told_live(self, number: int) -> frozenset[str]:
        item = self.items[number - 1]
        return frozenset((m for m in item.told if m not in item.dropped))

    def stamp_search(self, number: int, ids: list[str]) -> None:
        cat_key = self.category_key()
        item = self.items[number - 1]
        if ids and cat_key and any((self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or '')) for i in ids)):
            self.sought[number] = self.told_live(number)

    def alloc_hold(self, name: str, args: dict[str, Any]) -> str:
        if name != self.roles.get('search') or self.brief.get('pick') != 'cheapest' or len(self.items) < 2:
            return ''
        if self.left() <= self.tail() + len(self.items) + 3:
            return ''
        number = self.search_owner(args)
        if not number or number not in self.sought or self.told_live(number) - self.sought[number]:
            return ''
        newer = {n: self.told_live(n) - self.sought.get(n, frozenset()) for n in range(1, len(self.items) + 1) if n != number}
        stale = [n for n, new in newer.items() if new and (n, new) not in self.alloc_held]
        if not stale:
            return ''
        for n in stale:
            self.alloc_held.add((n, newer[n]))
        self.trace.append({'kind': 'alloc_hold', 'search': number, 'stale': stale, 'step': self.step})
        named = '; '.join((f"{self.items[n - 1].name} (new: {', '.join((m for m in self.items[n - 1].told if m in newer[n]))})" for n in stale))
        mine = self.items[number - 1].name
        return f"Not run (no step used). Requirements the shopper newly gave for {named} arrived after Kyo's last tracked search for it (your searches and Kyo's value searches that returned that kind of listing), or it has had none, so its fitting listings may not have been searched under them. {mine} has no newly given requirement since its last tracked search (that search may have been for a listing its requirement refers to). The pick is the cheapest, so spend this search on that item first: its product words plus a concrete value that meets the new requirement (for a limit, a value inside it that fitting rows show). Then search {mine} again if steps allow."

    def owns(self, item: Item, label: str) -> bool:
        return bool(label) and item in self.items and (self.owner_of(label) == self.items.index(item) + 1)

    def records_of_item(self, number: int) -> int:
        cat_key = self.category_key()
        item = self.items[number - 1]
        return sum((1 for record in self.full.values() if cat_key and self.owns(item, str(record.get(cat_key) or '')) and _fact_block(record)))

    def plan_lanes(self, count: int) -> dict[int, list[str]]:
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
            listing.append(f'  {index}. {name} (on {carried}): ' + ' | '.join(values) + self.not_stated(label, name, values) + (' [several features in one: ask last]' if re.search(CATCH_ALL, name) else ''))
        prompt = PLAN.replace('COUNT', str(max(1, count))) + '\n\nREQUEST:\n' + self.request + '\n\nITEMS:\n' + '\n'.join(settled)
        first = [str(x) for x in _items(self.brief.get('ask_first')) if str(x).strip()]
        if first:
            prompt += '\n\nASK FIRST (the request asks to be asked about these):\n' + '\n'.join(first)
        prompt += '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(listing) or '(no full listing records yet)')
        plan = self.ask_json(prompt, [LIGHT, HEAVY], 6000) or {}
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
                return next((k for k in own if plain(k) == plain(label)), label)

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
            for key in [k for k in keys if own.get(k, 0.0) >= GUARD_SHARE and (not re.search(CATCH_ALL, k))][:GUARD_KEYS]:
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
            lanes[number] = [x for x in lanes[number] if not re.search(CATCH_ALL, _words(x))] + [x for x in lanes[number] if re.search(CATCH_ALL, _words(x))]
        self.trace.append({'kind': 'plan', 'ask': _clip(plan.get('ask'), 120), 'lanes': {n: v[:8] for n, v in lanes.items()}})
        return lanes

    def question(self, number: int, label: str, samples: list[str] | None=None) -> str:
        text = f'What {label} would you like for the {self.items[number - 1].name}'
        return text + (f', for example {samples[0]} or {samples[1]}?' if samples else '?')

    def samples(self, number: int, label: str) -> list[str]:
        want = _words(label)
        settled = ' '.join(self.items[number - 1].musts).lower()
        for _, name, _, values in self.key_table():
            if name == want:
                return [v for v in values if v and v not in settled and (len(v) <= 30)][:2]
        return []

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

    def named_drops(self, heard: dict[str, Any], sent: list[tuple[int, str, str]]) -> dict[str, Any]:
        answers = ' '.join((_words(a) for _, _, a in sent))
        asked = ' '.join((_words(q) for _, q, _ in sent))
        moved = []
        for entry in _items(heard.get('free')):
            if not isinstance(entry, dict) or not str(entry.get('property') or '').strip():
                continue
            item = self.items[min(max(1, _count(entry.get('item')) or 1), len(self.items)) - 1]
            gone = _words(str(entry['property']).split(':')[0])
            named = '\\b' + re.escape(gone) + '\\b'
            live = [m for m in item.musts + item.told if m not in item.dropped and _words(m.split(':')[0]) == gone]
            if gone and live and re.search(named, answers) and (not re.search(named, asked)):
                moved.append({'item': entry.get('item'), 'property': str(entry['property'])})
        if not moved:
            return heard
        self.trace.append({'kind': 'named_drop', 'drop': moved})
        return {**heard, 'drop': list(_items(heard.get('drop'))) + moved}

    def take_heard(self, heard: dict[str, Any], told: list[str]) -> None:
        many = len(self.items) > 1
        for entry in _items(heard.get('needs')):
            if not isinstance(entry, dict) or not str(entry.get('need') or '').strip():
                continue
            number = min(max(1, _count(entry.get('item')) or 1), len(self.items))
            need = str(entry['need']).strip()
            need = need if entry.get('points') is True else _plain_need(need)
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
            self.budget = heard['budget']

    def bare_values(self, heard: dict[str, Any], sent: list[tuple[int, str, str]]) -> None:
        answers = ' '.join((str(answer) for _, _, answer in sent))
        for need in _items(heard.get('needs')):
            if not isinstance(need, dict) or need.get('points') is True:
                continue
            text = str(need.get('need') or '').strip()
            prop, sep, value = text.partition(': ')
            if not sep or len(value.split()) < 2 or self.cost + HEAVY_ROOM > SPEND_HARD:
                continue
            got = self.ask_json(BARE_VALUE.replace('ANSWERS', answers).replace('NEED', text), [HEAVY], 800) or {}
            out = str(got.get('value') or '').strip().strip('"\'`*').strip().rstrip('.').strip()
            found = re.search('(?<![a-z0-9])' + re.escape(out.lower()) + '(?![a-z0-9])', value.lower()) if out and out.lower() != value.lower() else None
            bare = value[found.start():found.end()] if found and (not VALUE_KEEP.search(value[:found.start()] + ' ' + value[found.end():])) else ''
            self.trace.append({'kind': 'bare_value', 'need': text[:160], 'got': out[:80], 'kept': bare or value})
            if bare:
                need['need'] = prop + ': ' + bare

    def polarity(self, heard: dict[str, Any], sent: list[tuple[int, str, str]]) -> None:
        answers = ' '.join((str(answer) for _, _, answer in sent))
        for need in _items(heard.get('needs')):
            text = str(need.get('need') or '') if isinstance(need, dict) else ''
            if ':' not in text or self.cost + HEAVY_ROOM > SPEND_HARD:
                continue
            prop, value = [part.strip() for part in text.split(':', 1)]
            plain = NEGATED.sub('', value).strip()
            if not plain or plain == value:
                continue
            prompt = 'A shopper was asked about a product and answered:\n"' + answers + '"\n\nA reader noted this need: "' + text + '".\n' + POLARITY.replace('PROP', prop).replace('VALUE', plain)
            message = self.think([{'role': 'user', 'content': prompt}], [LIGHT], 800)
            got = re.search('\\b([AB])\\b', str((message or {}).get('content') or ''))
            self.trace.append({'kind': 'polarity', 'need': text[:160], 'got': got.group(1) if got else None})
            if got and got.group(1) == 'A':
                need['need'] = prop + ': ' + plain

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(HEAVY, LIGHT)) -> dict[str, Any]:
        prompt = HEAR + '\n\nITEMS:\n' + '\n'.join((f'item {n}: {it.name}' for n, it in enumerate(self.items, 1)))
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f'{i}. (about item {n}) Q: {q}\nA: {a}' for i, (n, q, a) in enumerate(sent, 1)))
        return self.ask_json(prompt, list(models), 2500) or {}

    def heard_here(self, needs: Any, answers: list[str]) -> list[Any]:
        text = ' '.join(answers).lower()
        kept = []
        for entry in _items(needs):
            value = str(entry.get('need') or '') if isinstance(entry, dict) else ''
            words = [w for w in re.findall('[a-z0-9]+', (value.partition(':')[2] or value).lower()) if len(w) > 2]
            if value and answers and (not words or sum((1 for w in words if w in text)) >= 0.6 * len(words)):
                kept.append(entry)
            elif value:
                self.trace.append({'kind': 'gate_old_need', 'need': value[:200]})
        return kept

    def gate(self, stage: str) -> tuple[bool, str]:
        steps = [p for p in self.process if p['when'] == stage]
        if not steps or self.finished or (not self.gate_due(stage)):
            return (True, '')
        ending = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        tools = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_dump(t['function'].get('parameters') or {})}" for t in self.tools if t['function']['name'] not in ending]
        said: list[str] = []
        answers: list[str] = []
        nudged = False
        for _ in range(GATE_ROUNDS):
            prompt = GATE.replace('STAGE', 'place the order' if stage == 'order' else 'end without a purchase') + '\n\nREQUEST:\n' + self.request + '\n\nSTEPS DUE NOW:\n' + '\n'.join((f"- {p['step']}" for p in steps)) + '\n\nLEDGER:\n' + self.ledger() + '\n\nCART NOW: ' + self.cart_text() + '\n\nRECORD (messages and step calls so far):\n' + ('\n'.join((f'{i}. {line}' for i, line in enumerate(self.done_log, 1))) or '(nothing yet)') + ('\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.notes[-4:]) if self.notes else '') + f"\n\nMESSAGES LEFT: {max(0, self.msg_limit - self.msg_used)}   STEPS LEFT: {self.left()} (the {('order' if stage == 'order' else 'ending')} itself takes one)" + '\n\nTOOLS:\n' + '\n'.join(tools)
            got = self.ask_json(prompt, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 3500)
            if got is None:
                break
            told: list[str] = []
            hold = ''
            if said:
                self.take_heard({'needs': self.heard_here(got.get('needs'), answers)}, told)
                hold = str(got.get('hold') or '').strip()
                if hold and (not answers):
                    self.trace.append({'kind': 'gate_old_hold', 'hold': hold[:200]})
                    hold = ''
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
                answers.append(said[-1])
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
            need = need if entry.get('points') is True else _plain_need(need)
            if need in item.dropped:
                item.dropped.remove(need)
            if need not in item.musts + item.told:
                item.told.append(need)
            added.add(need)
            number = self.items.index(item) + 1
            if entry.get('points') is True and (number, need) not in self.refs:
                self.refs.append((number, need))
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
            record = self.full.get(ident) or self.rows.get(ident) or {}
            text = ' '.join((str(v) for v in record.values() if isinstance(v, str)))
            text += ' ' + '. '.join((f'{k} {v}' for block in record.values() if isinstance(block, dict) for k, v in block.items() if isinstance(v, str)))
            marks[ident] = _fit_marks(text)
            for part, codes in marks[ident]['also']:
                notes.setdefault(part, set()).update(codes)
        return {'marks': marks, 'notes': notes}

    def fit_set(self, ident: str, index: dict[str, Any]) -> set[str]:
        mark = index['marks'].get(ident) or {}
        found = set(mark.get('works') or []) | set(mark.get('families') or [])
        for own in mark.get('owns') or []:
            found |= index['notes'].get(own, set())
        return found

    def fit_hits(self, ident: str, codes: Any, index: dict[str, Any]) -> set[str]:
        mark = index['marks'].get(ident) or {}
        found = self.fit_set(ident, index)
        nots = set(mark.get('nots') or [])
        return {c for c in codes if c not in nots and any((t == c or _family_has(t, c) for t in found))}

    def fit_whole(self, ident: str, index: dict[str, Any]) -> bool:
        mark = index['marks'].get(ident) or {}
        more = mark.get('more')
        return bool(self.fit_set(ident, index)) and (not more or any((more in (m.get('owns') or []) for m in index['marks'].values())))

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
            codes = set(index['marks'].get(devices[0], {}).get('codes') or [])
            if not codes:
                continue
            for line in parts:
                fits = self.fit_set(line, index)
                if self.fit_whole(line, index) and (not self.fit_hits(line, codes, index)):
                    good = [i for i in self.item_ids(part) if self.fit_hits(i, codes, index) and (not self.sold_out(i)) and (i not in cart)][:4]
                    nots = (index['marks'].get(line) or {}).get('nots') or []
                    faults.append(f"{line} ({self.items[part - 1].name}) works with {', '.join(sorted(fits)[:4])}" + (f" except {', '.join(nots[:4])}" if nots else '') + f", not with {devices[0]} ({', '.join(sorted(codes))})" + (f"; seen parts of that kind that work with it: {', '.join(good)}" if good else '; no seen part of that kind works with it, so look for one or switch the device: ' + self.fit_plan(short=True)))
        return faults

    def fit_plan(self, short: bool=False) -> str:
        if not self.bound:
            return ''
        index = self.fit_index()
        lines = []
        for main in sorted({m for _, m in self.bound}):
            parts = [p for p, m in self.bound if m == main]
            tally: dict[str, dict[int, int]] = {}
            for p in parts:
                for i in self.item_ids(p):
                    if self.sold_out(i):
                        continue
                    for code in self.fit_set(i, index):
                        tally.setdefault(code, {}).setdefault(p, 0)
                        tally[code][p] += 1
            ranked = sorted(tally.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1].values())))[:4]
            devices = {code: [i for i in self.item_ids(main) if any((code == c or _family_has(code, c) for c in index['marks'].get(i, {}).get('codes') or []))][:2] for code, _ in ranked}
            text = '; '.join((f'{code}: ' + ', '.join((f'{n} {self.items[p - 1].name}' for p, n in sorted(per.items()))) + (f" (device seen: {', '.join(devices[code])})" if devices[code] else ' (device not seen yet)') for code, per in ranked))
            if text:
                lines.append(f"codes the {self.items[main - 1].name} candidates' parts name -> {text}")
        if not lines:
            return ''
        return ' | '.join(lines) if short else "[fit plan: a device fits a part when the part's text names the device's model code or another name it is sold as; codes are not searchable, so list the device's category and read the codes on its rows] " + ' | '.join(lines)

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
            prompt += '\n\nOTHER LISTINGS SEEN (newest first):\n' + self.others(ids, 15)
        got = self.ask_json(prompt, self.judges(limit, True), 5000)
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

    def note_pages(self, name: str, observation: Any) -> None:
        if name not in self.page_tools():
            return
        for ident, record in self.records_in(observation):
            if isinstance(record, dict) and (not record.get('error')) and (len(record) >= 3):
                self.page_obs[ident] = record

    def ref_names(self, text: str, ids: list[str]) -> list[str]:
        low = text.lower()
        out = []
        for ident in ids:
            row = self.rows.get(ident) or {}
            names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I)]
            if any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)):
                out.append(ident)
        return out

    def ref_tools(self) -> list[dict[str, Any]]:
        skip = {self.roles.get(r) for r in ('ask', 'cart', 'cart_view', 'buy', 'refuse')} - {None}
        report = {'type': 'function', 'function': {'name': self.report_name, 'description': 'Report the result of the reference search. Call it once, when done.', 'parameters': {'type': 'object', 'properties': {'found': {'type': 'boolean', 'description': 'true when the last listing of the chain gives the value'}, 'chain': {'type': 'array', 'items': {'type': 'string'}, 'description': 'ids of the listings opened along the chain, in order (with found false: the links you confirmed)'}, 'next': {'type': 'string', 'description': 'with found false: where the last confirmed page sends the reader for this property'}, 'value': {'type': 'string', 'description': 'the property value as the last listing states it'}, 'condition': {'type': 'string', 'description': 'the requirement restated with that value, as property: constraint'}, 'why': {'type': 'string', 'description': 'one short sentence'}}, 'required': ['found']}}}
        return [t for t in self.tools if t['function']['name'] not in skip] + [report]

    def ref_show(self, observation: Any) -> str:
        single = self.page_of(observation)[0]
        text = _dump(_clip(observation, 1500 if single else 220))
        return text if len(text) <= 9000 else text[:9000] + '...(cut)'

    def answer_view(self, needs: list[str]) -> str:
        return '(this answer gave a requirement that points at another listing; Kyo looked it up: ' + '; '.join((self.ref_conds[n] for n in needs)) + ')'

    def told_view(self, text: str) -> str:
        for need, cond in self.ref_conds.items():
            if text.endswith(need):
                return text[:len(text) - len(need)] + cond
        return text

    def resolve_pending(self) -> str:
        if self.resolving or self.finished or self.broke or self.no_credit:
            return ''
        notes: list[str] = []
        for _ in range(4):
            live = set(self.needs_now())
            todo = []
            for n, need in self.refs:
                tried = self.ref_tried.get((n, need))
                if (n, need) in live and need not in self.ref_values and (tried is None or (tried[0] < 2 and len(self.ref_names(need, self.seen_order)) > tried[1])):
                    todo.append((n, need))
            if not todo or self.finished:
                break
            number, need = todo[0]
            before, spent, asked = (self.step, self.ref_cost, self.ref_calls)
            self.resolving = True
            try:
                got = self.resolve(number, need)
            finally:
                self.resolving = False
            self.ref_steps += self.step - before
            self.ref_tried[(number, need)] = (self.ref_tried.get((number, need), (0, 0))[0] + 1, len(self.ref_names(need, self.seen_order)))
            if (number, need) not in set(self.needs_now()):
                self.trace.append({'kind': 'resolve_stale', 'item': number, 'need': need[:300], 'steps': self.step - before})
                continue
            chain = [i for i in got.get('chain') or [] if isinstance(i, str)]
            self.trace.append({'kind': 'resolve', 'item': number, 'need': need[:300], 'found': bool(got.get('found')), 'chain': chain, 'value': str(got.get('value') or '')[:200], 'condition': str(got.get('condition') or '')[:300], 'why': str(got.get('why') or '')[:400], 'next': str(got.get('next') or '')[:200], 'steps': self.step - before, 'cost': round(self.ref_cost - spent, 5), 'calls': self.ref_calls - asked})
            label = f'item {number}: ' if len(self.items) > 1 else ''
            value = str(got.get('value') or '').strip()
            cond = str(got.get('condition') or '').strip()
            if got.get('found') is True and chain and value and cond:
                self.ref_values[need] = value
                self.ref_conds[need] = cond
                self.ref_chain[need] = chain
                self.keep_prev(number, need, chain, '')
                self.ref_touched.update(chain)
                self.version += 1
                self.recent = []
                self.notes.append(f'requirement resolved: {label}{cond}')
                notes.append(f"[Kyo resolved a requirement that pointed at another listing] {label}{cond} (Kyo found and opened {' -> '.join(chain)}; the value comes from the last one. It is now a firm requirement in the ledger: apply it to every pick, and do not look those listings up again.)")
            else:
                part = [i for i in chain if i in self.page_obs]
                ahead = str(got.get('next') or '').strip()
                self.keep_prev(number, need, part, ahead)
                extra = "the finder's unverified notes: " + (f"pages it opened for this chain, in its order: {', '.join(part)}; " if part else '') + (f'its lead for the next listing: {ahead[:300]}; ' if ahead else '') + 'these links were not checked; ' if part or ahead else ''
                notes.append(f"[Kyo could not resolve a requirement that points at another listing] {label}{need} ({extra}{str(got.get('why') or 'not found')[:300]})")
        if notes and self.queued:
            notes.append('[notes seen meanwhile] ' + ' | '.join(self.queued))
            self.queued = []
        return '\n'.join(notes)

    def keep_prev(self, number: int, need: str, chain: list[str], ahead: str) -> None:
        part = [i for i in chain if i in self.page_obs]
        if part and len(part) >= len((self.ref_prev.get((number, _prop(need))) or {}).get('chain') or []):
            self.ref_prev[(number, _prop(need))] = {'need': need, 'chain': part, 'next': ahead}

    def resolve(self, number: int, need: str) -> dict[str, Any]:
        item = self.items[number - 1]
        page = self.roles.get('offer') or self.roles.get('open')
        search = self.roles.get('search')
        if not page or page not in self.schemas or not self.single_target(page):
            return {'found': False, 'why': 'no single-listing page tool'}
        reserve = len(self.items) + self.tail() + 1
        budget = self.left() - reserve
        if budget < 2:
            return {'found': False, 'why': f'too few steps left ({self.left()})'}
        tools = self.ref_tools()
        names = {t['function']['name'] for t in tools}
        compare = self.roles.get('compare') if self.roles.get('compare') in names else None
        facts = self.facts_tool() or self.roles.get('open')
        lines = [f'REQUIREMENT on the product being bought ({item.name}): {need}', f"TOOLS: {page} opens one listing's page" + (f'; {compare} shows several listings side by side (that also counts as opening their pages)' if compare else '') + (f"; {facts} shows a listing's attributes" if facts and facts != page and facts in names else '') + (f'; {search} searches' if search in names else '') + '.', f'STORE STEPS YOU MAY USE: {budget}']
        named = self.ref_names(need, list(reversed(self.seen_order)))
        prev = self.ref_prev.get((number, _prop(need))) or {}
        done = [i for i in prev.get('chain') or [] if i in self.page_obs][:5]
        if done:
            lines.append(f"AN EARLIER SEARCH ON THE SAME PROPERTY OF THIS PRODUCT (its requirement: {str(prev.get('need') or '')[:300]}) ALREADY OPENED THESE PAGES, IN ORDER (no need to open or search for them again; check that each one fits the description in THIS requirement or on the page that sends the reader to it before you build on it, then go on from the last one that fits):\n" + '\n'.join((f'{i}: ' + _dump(_clip(self.page_obs[i], 600))[:1800] for i in done)))
            if prev.get('next'):
                lines.append(f"ITS UNCHECKED NOTE ON WHERE THE LAST PAGE SENDS THE READER: {prev['next'][:300]}")
        pages = [i for i in named if i in self.page_obs and i not in done][:4]
        if pages:
            lines.append('PAGES ALREADY SHOWN FROM SHOPS THE REQUIREMENT NAMES (already opened, no need to open them again):\n' + '\n'.join((f'{i}: ' + _dump(_clip(self.page_obs[i], 600))[:1800] for i in pages)))
        rows = [i for i in named if i not in pages and i not in done][:12]
        if rows:
            lines.append('ROWS ALREADY SEEN FROM SHOPS THE REQUIREMENT NAMES:\n' + '\n'.join((_dump(_clip(self.rows.get(i), 120))[:500] for i in rows)))
        lines.append("THE SHOPPER'S REQUEST (context only; buying is not your job):\n" + self.request[:2500])
        chat: list[dict[str, Any]] = [{'role': 'system', 'content': RESOLVE.replace('REPORT_NAME', self.report_name)}, {'role': 'user', 'content': '\n\n'.join(lines)}]
        state: dict[str, Any] = {'need': need, 'used': 0, 'budget': budget, 'reserve': reserve, 'searches': 0, 'pushed': False, 'rejects': 0, 'made': set(), 'texts': [need] + [_dump(self.page_obs[i]) for i in done]}
        idle = 0
        for turn in range(REF_TURNS):
            if self.finished or self.broke or self.no_credit:
                break
            message = self.think(chat, [LIGHT, SPARE], 2500, tools)
            if message is None:
                break
            calls = []
            for index, call in enumerate(message.get('tool_calls') or []):
                if not isinstance(call, dict):
                    continue
                fn = call.get('function') if isinstance(call.get('function'), dict) else {}
                raw = fn.get('arguments')
                parsed = raw if isinstance(raw, dict) else _json_obj(raw) if isinstance(raw, str) and raw.strip() else {}
                calls.append({'id': str(call.get('id') or f'r{turn}_{index}'), 'type': 'function', 'function': {'name': str(fn.get('name') or ''), 'arguments': _dump(parsed if isinstance(parsed, dict) else {})}})
            content = message.get('content')
            if not isinstance(content, str):
                content = _dump(content) if content else ''
            if not calls:
                idle += 1
                chat.append({'role': 'assistant', 'content': content or '(no call)'})
                if idle >= 2:
                    break
                chat.append({'role': 'user', 'content': f'Call a tool: a store tool to keep looking, or {self.report_name} when done.'})
                continue
            idle = 0
            calls = calls[:4]
            chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
            final = None
            for call in calls:
                name = call['function']['name']
                args = _json_obj(call['function']['arguments']) or {}
                if final is not None:
                    reply = 'Not run: the report was accepted.'
                elif name == self.report_name:
                    reply, final = self.ref_report(need, args, state, turn)
                else:
                    reply = self.ref_call(name, args, state)
                chat.append({'role': 'tool', 'tool_call_id': call['id'], 'content': reply})
            if final is not None:
                return final
            chat[-1]['content'] += f"\n[steps used for this reference: {state['used']} of {state['budget']}; turns left: {REF_TURNS - turn - 1}]"
        return {'found': False, 'why': 'stopped before a report'}

    def ref_call(self, name: str, args: dict[str, Any], state: dict[str, Any]) -> str:
        names = {t['function']['name'] for t in self.ref_tools()} - {self.report_name}
        if name not in names:
            return 'Not run (no step used): not a tool for this job. Use: ' + ', '.join(sorted(names)) + f', or {self.report_name}.'
        faults: list[str] = []
        args = _shape(args, self.schemas[name], 'args', faults)
        if faults:
            return 'Not run (no step used): ' + '; '.join(faults[:4])
        args, wrong = self.resolve_ids(name, args)
        if wrong:
            return 'Not run (no step used): ' + wrong + '.'
        mark = name + _dump(args)
        if mark in state['made']:
            return 'Not run (no step used): you made this same call already; its result is above.'
        if name in self.page_tools():
            many = self.compare_target() if name == self.roles.get('compare') else None
            want = [str(i) for i in _items(args.get(many))] if many else [str(args.get(self.single_target(name) or '') or '')]
            want = [i for i in want if i]
            if want and all((i in self.page_obs for i in want)):
                state['texts'] += [_dump(self.page_obs[i]) for i in want]
                return 'Not run (no step used): these pages were shown already:\n' + '\n'.join((f'{i}: ' + _dump(_clip(self.page_obs[i], 600))[:1800] for i in want)) + self.ref_candidates(state)
        if state['used'] >= state['budget'] or self.left() - 1 < state['reserve']:
            return f'Not run: no store steps are left for the reference. Call {self.report_name} now.'
        if name == self.roles.get('search'):
            held = self.ref_same_shop(args, state)
            if held:
                return held
        reply = self.act(name, args)
        state['used'] += 1
        state['made'].add(mark)
        if reply is None:
            return 'The store did not accept that call. Check the tool and its arguments.'
        observation = reply.get('observation')
        self.trace.append({'kind': 'ref_call', 'name': name, 'args': _clip(args, 100), 'step': self.step})
        self.observe(name, observation)
        self.note_pages(name, observation)
        ids = [i for i, _ in self.records_in(observation)]
        self.ref_touched.update(ids)
        listing, record = self.page_of(observation)
        if listing and self.single_target(name):
            self.facts_from[name] = self.facts_from.get(name, False) or bool(_fact_block(record))
        shown = self.ref_show(observation)
        if name in self.page_tools():
            state['texts'] += [_dump(self.page_obs[i]) for i in ids if i in self.page_obs]
        if name == self.roles.get('search'):
            state['searches'] += 1
            state['last_rows'] = ids
            state['checked'] = None
            if ids:
                self.searched = (self.searched + [dict(args)])[-8:]
        elif name not in self.page_tools() and self.single_target(name) and _fact_block(record):
            state.setdefault('facts', set()).update(ids)
        if name != self.roles.get('search') and self.single_target(name):
            state['checked'] = str(args.get(self.single_target(name)) or '') or None
        return shown + self.ref_candidates(state)

    def ref_same_shop(self, args: dict[str, Any], state: dict[str, Any]) -> str:
        last, rows = state.get('checked'), state.get('last_rows') or []
        if not last or last not in rows or last not in self.page_obs or state.get('held') == rows:
            return ''
        query = _dump(args).lower()
        shops = [s for s in _seller_names(self.rows.get(last) or {}) if re.search('(?<![a-z0-9])' + re.escape(s.lower()) + '(?![a-z0-9])', query)]
        facts = state.get('facts') or set()
        same = [i for i in rows if i != last and i not in self.page_obs and i not in facts and set(_seller_names(self.rows.get(i) or {})) & set(shops)]
        if not shops or not same:
            return ''
        state['held'] = rows
        return f"Not run (no step used): the last search also showed {', '.join(same[:6])} from {shops[0]}, like {last}, the listing you just checked, and they are not opened yet. Check any that could match the listing you are looking for now (a row's title may not show every value); otherwise send this same search again and it will run."

    def ref_candidates(self, state: dict[str, Any]) -> str:
        named = self.ref_names(' '.join(state['texts']), list(reversed(self.seen_order)))
        facts = state.get('facts') or set()
        fresh = [i for i in named if i not in self.page_obs and i not in facts]
        paged = [i for i in named if i in self.page_obs]
        read = [i for i in named if i in facts and i not in self.page_obs]
        parts = ['not checked yet: ' + (', '.join(fresh[:12]) if fresh else 'none')] + (['page shown: ' + ', '.join(paged[:12])] if paged else []) + (['attributes read: ' + ', '.join(read[:12])] if read else [])
        return '\n[rows seen whose shop or brand is named in the requirement or on the pages opened so far: ' + '; '.join(parts) + ']'

    def ref_report(self, need: str, args: dict[str, Any], state: dict[str, Any], turn: int) -> tuple[str, dict[str, Any] | None]:
        if args.get('found') is not True:
            room = min(state['budget'] - state['used'], self.left() - state['reserve'])
            if room >= 1 and state.get('nofound', 0) < 2 and turn < REF_TURNS - 2:
                state['nofound'] = state.get('nofound', 0) + 1
                named = self.ref_names(' '.join(state['texts']), list(reversed(self.seen_order)))
                fresh = [i for i in named if i not in self.page_obs and i not in (state.get('facts') or set())]
                hint = f"Rows seen from the shops named so far that are not checked yet: {', '.join(fresh[:8])} (some may belong to an earlier link; check the ones that could fit the description of the listing you are looking for now). " if fresh else ''
                return (f'Not accepted yet: {room} store step(s) are left for this reference, so keep looking before reporting it unresolved. ' + hint + 'Otherwise try other words (the shop with the kind, the kind with one value, the shop alone) or the next result page.', None)
            return ('Noted.', {'found': False, 'why': str(args.get('why') or '')[:400], 'chain': [str(i).strip() for i in _items(args.get('chain')) if str(i).strip()][:6], 'next': str(args.get('next') or '').strip()[:300]})
        chain: list[str] = []
        for entry in _items(args.get('chain')):
            ident = str(entry).strip()
            if ident and ident not in chain:
                chain.append(ident)
        value = str(args.get('value') or '').strip()
        if not chain or not value:
            return ('Not accepted: give chain (the listing ids in order) and value.', None)
        page = self.roles.get('offer') or self.roles.get('open')
        target = self.single_target(page)
        missing = [i for i in chain if i not in self.page_obs]
        shown = []
        if missing and target:
            state['budget'] = max(state['budget'], state['used'] + min(2, len(missing)))
            for ident in missing:
                reply = self.ref_call(page, {target: ident}, state)
                if ident in self.page_obs:
                    shown.append(f'{ident}: ' + reply[:2500])
        still = [i for i in chain if i not in self.page_obs]
        if still:
            if state['used'] >= state['budget'] or self.left() - 1 < state['reserve']:
                return ('Noted.', {'found': False, 'why': 'chain pages not shown: ' + ', '.join(still)[:150]})
            return (('\n'.join(shown) + '\n' if shown else '') + f"Not accepted: no page tool has shown {', '.join(still)} (a search row or an attribute view does not count, and the id must be one the store returns). Open it with {page}, or report found false.", None)
        if shown:
            return ('Kyo opened the pages of these chain listings with the page tool, since you had not:\n' + '\n'.join(shown) + '\nRead them: when one sends the reader on for this property, follow it; otherwise report again.', None)
        verdict = self.ref_check(need, chain, value, str(args.get('condition') or ''))
        if not verdict:
            return ('Noted.', {'found': False, 'why': 'the check could not run'})
        cond = str(verdict.get('condition') or args.get('condition') or '').strip()
        if verdict.get('ok') is not True or not cond:
            room = min(state['budget'] - state['used'], self.left() - state['reserve'])
            if state['rejects'] < (3 if room >= 1 else 2):
                state['rejects'] += 1
                problem = str(verdict.get('problem') or '').strip() if verdict.get('ok') is not True else 'no condition was given.'
                return ('Not accepted: ' + (problem or 'the chain does not match the descriptions')[:400] + (' Fix it (search, read, open) and report again.' if room >= 1 else ' Fix it and report again, or report found false.'), None)
            return ('Noted.', {'found': False, 'why': 'check failed: ' + str(verdict.get('problem') or 'no condition')[:160]})
        return ('Accepted.', {'found': True, 'chain': chain, 'value': str(verdict.get('value') or value).strip(), 'condition': cond})

    def ref_check(self, need: str, chain: list[str], value: str, condition: str) -> dict[str, Any]:
        records = []
        for number, ident in enumerate(chain, 1):
            entry = f'LISTING {number} ({ident}) PAGE: ' + _dump(_clip(self.page_obs.get(ident), 700))[:3000]
            block = _fact_block(self.full.get(ident))
            if block:
                entry += '\nATTRIBUTES: ' + _dump(_clip(block, 200))[:1500]
            records.append(entry)
        prompt = RCHECK + '\n\nREQUIREMENT: ' + need + '\nREPORTED VALUE: ' + value + '\nREPORTED CONDITION: ' + (condition or '(none)') + '\n\nTHE CHAIN, IN ORDER:\n' + '\n\n'.join(records)
        got = self.ask_json(prompt, [LIGHT, HEAVY], 5000)
        self.trace.append({'kind': 'ref_check', 'chain': chain, 'got': _clip(got, 200)})
        return got or {}

    def trail(self) -> tuple[str, bool]:
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        tool = self.roles.get('offer') or self.roles.get('open')
        target = self.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trails < (6 if self.left() > 15 else 4) and (self.cost < SPEND_HARD):
            self.trails += 1
            records = []
            for ident in reversed(self.seen_order):
                record = self.full.get(ident) or self.rows.get(ident) if ident in self.opened else None
                if record and len(records) < 45:
                    records.append('OPENED ' + _dump(_clip(record, 300))[:1400])
            for ident in reversed(self.seen_order):
                if len(records) >= 60:
                    break
                if ident not in self.opened or not (self.full.get(ident) or self.rows.get(ident)):
                    records.append('ROW ' + _dump(_clip(self.rows.get(ident), 80))[:300])
            prompt = TRAIL + '\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n' + '\n'.join(pending)
            prompt += '\n\nRECORDS (newest first):\n' + '\n'.join(records)
            got = self.ask_json(prompt, [LIGHT] + ([HEAVY] if self.cost + HEAVY_ROOM <= SPEND_HARD else []), 4500) or {}
            self.trace.append({'kind': 'trail', 'pending': pending, 'got': _clip(got, 150)})
            reach: list[str] = []
            lost = []
            gaps: list[tuple[str, str]] = []
            for chain in _items(got.get('chains')):
                if not isinstance(chain, dict):
                    continue
                need = next((p for p in pending if p == str(chain.get('need') or '').strip()), None) or (pending[0] if len(pending) == 1 else None)
                if not need:
                    continue
                links = [l for l in _items(chain.get('links')) if isinstance(l, dict)]
                ids = [str(l.get('id') or '').strip() for l in links]
                if chain.get('done') is True and links and all((i in self.opened for i in ids)) and str(chain.get('value') or '').strip():
                    self.ref_values[need] = str(chain['value']).strip()
                    self.version += 1
                    continue
                if links and all((i in self.opened for i in ids)) and (not str(links[-1].get('find') or '').strip()) and str(chain.get('value') or '').strip():
                    self.ref_values[need] = str(chain.get('value') or '').strip()
                    self.version += 1
                    continue
                closed = [i for i in ids if i in self.rows and i not in self.opened]
                if closed:
                    reach += [i for i in closed if i not in reach]
                else:
                    gap = next((l for l, i in zip(links, ids) if i not in self.rows), {})
                    about = str(gap.get('about') or '') or 'the listing it points at'
                    words = str(gap.get('find') or '').strip()
                    lost.append(f'{need} -> find {about}' + (f' (search: {words})' if words else ''))
                    gaps.append((about, words))
            pending = [p for p in pending if p not in self.ref_values]
            if not reach and gaps and tool and target and (self.left() > 12) and (self.link_searches < 3):
                reach = self.find_link(*gaps[0])
                if self.trace and self.trace[-1].get('kind') == 'link_known':
                    seen.append(f"{self.trace[-1]['id']}: an opened page of the shop this chain names that shows the values its description gives; read the value from it")
                    lost = lost[1:]
            if not reach or not tool or (not target) or (self.left() < len(reach) + 4) or self.finished:
                break
            for ident in reach[:max(0, min(3, 6 - len(seen)))]:
                faults: list[str] = []
                args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
                if faults:
                    break
                reply = self.act(tool, args)
                if reply is None or self.finished:
                    break
                self.observe(tool, reply.get('observation'))
                seen.append(f'{ident}: ' + self.show(reply.get('observation'))[:1500])
        text = ''
        if seen:
            text += '[Kyo opened the pages a requirement points at]\n' + '\n'.join(seen) + '\n'
        if self.ref_values:
            text += '[values read from the pages requirements point at] ' + '; '.join((f'{k} -> {v}' for k, v in self.ref_values.items())) + '\n'
        hold = bool(lost) and self.trail_blocks < (3 if self.left() > 15 else 2) and (self.left() > 5)
        if hold:
            self.trail_blocks += 1
            text += 'The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: ' + ' | '.join(lost[:3]) + ". Search for it by its shop's name together with the values its description gives (the shop's other listings of that kind are not it), open the row of that shop whose page shows those values, apply the value its page gives, then check out."
        return (text, hold)

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
        values = {w for w in re.findall('[a-z0-9]+', (words or about).lower()) if len(w) >= 3 or w.isdigit()} - named - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for', 'of'}
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
        kind = {w for w in re.findall('[a-z0-9]{3,}', (words or about).lower())} - named - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for'}

        def of_kind(ids: list[str]) -> list[str]:
            return [i for i in ids if not kind or kind & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower()))]
        return (of_kind(known) or of_kind(self.shop_rows(about, list(reversed(self.seen_order)))))[:2]

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
            records.append(_dump(_clip(self.full.get(ident) or self.rows.get(ident), 200))[:1200])
            if len(records) >= 25:
                break
        if not records:
            return []
        prompt = SWAP + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
        prompt += '\n\nSOLD-OUT CART LINE:\n' + '\n'.join((f'{i}: {self.record_of(i)}' for i in lost)) + '\n\nCART NOW:\n' + self.cart_text()
        prompt += '\n\nOTHER LISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
        got = self.ask_json(prompt, self.judges(), 8000)
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
        named = self.audit_fits(got)
        for _ in range(2):
            if named:
                break
            rows = [str(i) for i in _items((got or {}).get('open')) if str(i) in self.rows and str(i) not in self.opened and (not self.sold_out(str(i)))]
            if self.brief.get('pick') == 'cheapest':
                rows.sort(key=lambda i: (self.price_of(i) is None, self.price_of(i) or 0.0))
            if not (rows and self.open_rows(rows)):
                break
            got = self.audit_call()
            named = self.audit_fits(got)
        if not named:
            return ''
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return 'A careful review thinks these listings may fit: ' + '; '.join((f"item {k}: {', '.join(v)}" for k, v in named.items())) + (f" ({got.get('reason')})" if got.get('reason') else '') + '. Open them and buy what truly fits; end without a purchase only if each one fails.'

    def pointer_rows(self, ids: list[str] | None=None, keep: int | None=None) -> list[str]:
        keep = self.tail() + 3 if keep is None else keep
        if self.finished or self.left() <= keep:
            return []
        index = self.fit_index()
        own = {mark.get('own') for mark in index['marks'].values() if mark.get('own')}
        want = []
        for ident in self.cart_ids() if ids is None else ids:
            code = (index['marks'].get(ident) or {}).get('more')
            if code and code not in own and code not in want and code not in self.pointer_codes:
                want.append(code)
        plan = self.search_template() if want else None
        if not plan:
            return []
        tool, base, field = plan
        seen = []
        for code in want:
            faults: list[str] = []
            args = _shape({**base, field: code}, self.schemas.get(tool), 'args', faults)
            if faults or self.finished or self.left() <= keep:
                break
            self.pointer_codes.append(code)
            reply = self.act(tool, args)
            if reply is None:
                break
            observation = reply.get('observation')
            self.observe(tool, observation)
            found = [i for i, _ in self.records_in(observation)]
            seen.append(f"{code} (named by a cart page; shown for reference only): " + (', '.join(found[:4]) or 'no row'))
        if seen:
            self.trace.append({'kind': 'pointer_rows', 'seen': seen})
        return seen

    def row_item(self, ident: str) -> int:
        cat_key = self.category_key()
        label = str((self.rows.get(ident) or {}).get(cat_key) or '').strip() if cat_key else ''
        if not label:
            return 0
        if any((item.shelf for item in self.items)):
            same = [n for n, item in enumerate(self.items, 1) if item.shelf and _same_label(item.shelf, label)]
            return same[0] if len(same) == 1 else 0
        return self.owner_of(label)

    def spare_picks(self) -> list[str]:
        if not self.price_key:
            for name, _, seen in self.search_log:
                if name not in self.shapes:
                    self.read_shape(name, seen)
                if self.price_key:
                    break
        cap = self.budget if isinstance(self.budget, (int, float)) and (not isinstance(self.budget, bool)) else None
        each = self.brief.get('budget_scope') == 'each'
        ranked: list[tuple[tuple[Any, ...], str]] = []
        for number, item in enumerate(self.items, 1):
            if self.fit_fits(number):
                continue
            values = [_roots(m.partition(':')[2]) for m in item.musts + item.told if m not in item.dropped]
            values = [v for v in values if v]
            for place, (name, args, seen) in enumerate(self.search_log):
                props = (self.schemas.get(name) or {}).get('properties') or {}
                said = _roots(' '.join((str(v) for k, v in args.items() if isinstance(v, str) and (not re.search('page|token|cursor|offset|after|next|start|begin|limit|size|sort|order', k, re.I)) and (not (props.get(k) or {}).get('enum')))))
                carried = sum((1 for v in values if v <= said))
                if not carried:
                    continue
                rows = [i for i, _ in self.records_in(seen)]
                count = self.reply_count(name, seen, len(rows))
                for row, ident in enumerate(rows):
                    if ident in self.full or ident not in self.rows or ident in self.ref_touched or (not self.buyable(ident)) or self.row_item(ident) != number:
                        continue
                    price = self.price_of(ident)
                    if cap is not None and price is not None and (price if each else price * item.qty) > cap:
                        continue
                    ranked.append(((-carried, count if count is not None else 10 ** 9, row, place), ident))
        out: list[str] = []
        for _, ident in sorted(ranked):
            if ident not in out:
                out.append(ident)
        return out

    def spare_rows(self) -> str:
        keep = self.tail() + 3
        if self.finished or self.spare_rounds >= SPARE_ROUNDS or self.left() <= keep or self.cost >= SPEND_SOFT or time.time() - self.fit_began > SPARE_SECONDS:
            return ''
        picks = self.spare_picks()[:min(SPARE_OPENS, self.left() - keep)]
        if not picks:
            return ''
        self.spare_rounds += 1
        pages = self.fill_facts(picks, keep)
        opened = [i for i in picks if i in self.full]
        if not opened:
            return ''
        checked = self.fit_checks(opened)
        self.trace.append({'kind': 'spare_rows', 'step': self.step, 'opened': opened, 'fits': [i for i in opened if self.fit_state(i) == 'fit']})
        shown = checked or '\n'.join(pages)
        return 'Not run yet. Before an ending without a purchase, Kyo used spare steps to open rows not opened yet that a store search for the requirement values returned (one step each):\n' + shown + '\nIf one of them fits every requirement in force, put it in the cart at the requested quantity (with a most or least pick: the one that wins on it) and check out; otherwise send the ending call again.'

    def audit_order(self) -> list[str]:
        newest = [i for i in reversed(self.seen_order) if not self.sold_out(i)]
        cat_key = self.category_key()
        kind = [i for i in newest if not cat_key or self.label_owner(str((self.rows.get(i) or {}).get(cat_key) or ''))]
        cap = self.budget if isinstance(self.budget, (int, float)) else None
        fit = [i for i in kind if cap is None or (self.price_of(i) or 0) <= cap] or kind
        first = [i for i in fit if i in self.full] + [i for i in fit if i not in self.full]
        return first + [i for i in newest if i not in first]

    def audit_call(self) -> dict[str, Any] | None:
        records = []
        for ident in self.audit_order():
            record = self.full.get(ident) if ident in self.full else None
            records.append('OPENED ' + _dump(_clip(record, 200))[:1200] if record else 'ROW ' + _dump(_clip(self.rows.get(ident), 120))[:500])
            if len(records) >= 30:
                break
        if not records:
            return None
        prompt = AUDIT + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
        prompt += '\n\nLISTING RECORDS SEEN (a requested kind inside the budget first, read pages before rows, then newest first):\n' + '\n'.join(records)
        got = self.ask_json(prompt, self.judges(), 9000)
        self.trace.append({'kind': 'audit', 'got': got})
        return got

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get('fits')
        if not isinstance(fits, dict):
            return {}
        named: dict[str, list[str]] = {}
        for k, v in fits.items():
            ids = [str(i) for i in _items(v) if str(i) in self.rows and (not self.sold_out(str(i)))]
            key = self.item_key(k)
            named[key] = (named.get(key, []) + [i for i in ids if i not in named.get(key, [])])[:4]
        return {k: v for k, v in named.items() if v}

    def item_key(self, key: Any) -> str:
        number = _count(key)
        if number is None:
            digits = re.findall('\\d+', str(key))
            number = int(digits[0]) if len(digits) == 1 else None
        if number is not None and 1 <= number <= len(self.items):
            return str(number)
        if len(self.items) == 1:
            return '1'
        words = _stem(str(key))
        scores = [len(words & _stem(item.name)) for item in self.items]
        best = max(scores) if scores else 0
        if best and scores.count(best) == 1:
            return str(scores.index(best) + 1)
        return str(key)

    def open_rows(self, rows: list[str]) -> bool:
        many, joiner = (self.compare_target(), self.roles.get('compare'))
        tool = self.facts_tool()
        target = self.single_target(tool)
        if many and joiner and (not self.compare_bare()):
            plan = [(joiner, {many: rows[:self.compare_room()]})]
        elif tool and target:
            plan = [(tool, {target: i}) for i in rows[:min(3, max(2, self.left() - 5))]]
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
        self.pointer_rows(list(lines), len(calls) + 3)
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
            if any(i not in self.opened for i in self.cart_ids()):
                keep.add(self.roles.get('offer') or self.roles.get('open'))
        elif left <= len(self.items) + 5 and self.buys == 0:
            keep = set(ends) | {self.roles.get('cart'), self.roles.get('cart_view'), self.roles.get('offer')}
            if left - 1 >= max(0, len(self.items) - len(set(self.cart_ids()))) + self.tail():
                keep.add(self.roles.get('offer') or self.roles.get('open'))
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
            trimmed = ''
            many = self.compare_target() if name == self.roles.get('compare') else None
            if many and isinstance(args.get(many), list) and (len(args[many]) > self.compare_room()) and self.compare_bad:
                room = self.compare_room()
                trimmed = f"\n[Kyo] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
                args = {**args, many: args[many][:room]}
            target = self.single_target(name)
            if target and name in (self.roles.get('open'), self.roles.get('offer')) and (str(args.get(target)) in self.gone):
                out.append((cid, f'Not run (no step used): {args[target]} sold out at step {self.gone[str(args[target])]} and will not come back. Replace its cart line with another fitting listing.'))
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
            if not ending and self.left() <= self.tail() and (self.roles.get('buy') or self.roles.get('refuse')):
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
            held = '' if ending else self.alloc_hold(name, args)
            if held:
                out.append((cid, held + '\nCart now: ' + self.cart_text()))
                continue
            if ending and self.left() > 2:
                if self.is_buy(name, args):
                    wider = self.wider_hold()
                    if wider:
                        out.append((cid, wider + '\nCart now: ' + self.cart_text()))
                        continue
                    held = self.fit_buy_hold() or self.seek_hold()
                    if held:
                        out.append((cid, held + '\nCart now: ' + self.cart_text()))
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
                    chain, hold = self.trail()
                    if hold:
                        out.append((cid, 'Not run yet (no step used). ' + chain + '\nCart now: ' + self.cart_text()))
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
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()
                    read += self.pointer_rows()
                    verdict = self.check_cart()
                    if verdict:
                        seen = "Kyo read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n' if read else ''
                        out.append((cid, 'Not run yet. ' + chain + seen + verdict + '\nCart now: ' + self.cart_text()))
                        continue
                    if reread and self.cost > SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). Kyo re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'))
                        continue
                elif self.left() > 5:
                    held = self.fit_decline_hold()
                    if held:
                        out.append((cid, held))
                        continue
                    spare = self.spare_rows()
                    if spare:
                        out.append((cid, spare))
                        continue
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
            if self.is_refuse(name, args) and self.left() <= 5 and (not self.late_held):
                lines = [i for i in self.built if not self.sold_out(i)]
                fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
                missing = [i for i in lines if i not in self.cart_ids() or (i in self.dropped and (not fresh))]
                if len(lines) >= len(self.items) and len(missing) + 1 <= self.left():
                    self.late_held = True
                    self.trace.append({'kind': 'late_decline_held', 'lines': lines, 'missing': missing})
                    redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else 'Check out the cart as it is.'
                    out.append((cid, 'Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. ' + redo + ' End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again.'))
                    continue
            if self.is_refuse(name, args) and (not self.leads) and (not self.pushed) and (self.left() > self.max_steps // 3):
                self.pushed = True
                out.append((cid, f"Not run yet (no step used). {self.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if self.is_refuse(name, args):
                done = self.complete_cart()
                if done:
                    out.append((cid, done + (' [episode finished]' if self.finished else '')))
                    continue
                args = self.quiet(name, args)
            if ending and self.left() >= 2:
                go, said = self.gate('order' if self.is_buy(name, args) else 'end')
                if not go:
                    out.append((cid, 'Not run yet. ' + said + '\nCart now: ' + self.cart_text()))
                    continue
                if said:
                    trimmed = '\n[Kyo] ' + said + trimmed
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
                else:
                    self.compare_ok = max(self.compare_ok, len(args[many]))
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
                listing, record = self.page_of(observation)
                if listing:
                    self.facts_from[name] = self.facts_from.get(name, False) or bool(_fact_block(record))
                    if name == self.roles.get('open') and (not self.facts_from[name]):
                        other = next((t for t in self.schemas if t != name and self.single_target(t) and (self.facts_from.get(t) is not False) and (t not in (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask'), self.roles.get('search')))), None)
                        if other:
                            self.roles['open'] = other
                    if not self.facts_from[name] and name != self.roles.get('open') and (not self.roles.get('offer')):
                        self.roles['offer'] = name
                    if name == self.offer_tool():
                        self.offer_seen[listing] = self.step
            if name == self.roles.get('cart'):
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
                owner = self.search_owner(args)
                if owner:
                    self.stamp_search(owner, [i for i, _ in self.records_in(observation)])
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
            if name in {self.roles.get(r) for r in ('search', 'open', 'offer', 'compare')} - {None}:
                self.seek_reads += 1
            if name in {self.roles.get(r) for r in ('open', 'offer', 'compare')} - {None} and (not ending):
                checked = self.fit_checks(self.fact_ids(observation))
                if checked:
                    text += '\n' + checked
            if name == self.roles.get('cart'):
                listed = self.fit_cart_note(args)
                if listed:
                    text += '\n' + listed
            if name == self.roles.get('cart') and any((_count(v) == 0 for v in args.values() if not isinstance(v, str))):
                if any((isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values())):
                    text += '\n[Kyo] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'
            self.note_sold = False
            lines = self.loose_lines(observation)
            if lines:
                meaning = self.take_note(lines)
                if meaning:
                    text += '\n[note read] ' + meaning + '. Apply this change to your picks before checking out.'
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
            out.append((cid, text))
        return out

    def finish(self, again: bool=False) -> str:
        if self.finished or self.broke or self.left() <= 0:
            return ''
        buy, refuse = (self.roles.get('buy'), self.roles.get('refuse'))
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if buy and (holding or (self.cart_touched and (not fresh) and (self.left() >= 2))):
            faults: list[str] = []
            args = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), 'args', faults)
            if not faults:
                self.pointer_rows(keep=self.tail() + 2)
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
        if size <= HISTORY_LIMIT or size < self.trimmed_size + HISTORY_LIMIT:
            return
        tools = [i for i, m in enumerate(self.chat) if m.get('role') == 'tool']
        for index in tools[:max(1, len(tools) // 2)]:
            content = self.chat[index].get('content') or ''
            if len(content) > 600 and SHORTENED not in content:
                ids = re.findall('"([A-Za-z0-9][A-Za-z0-9_\\-:.]{5,40})"', content)
                ids = [i for i in dict.fromkeys(ids) if i in self.rows][:20]
                self.chat[index]['content'] = content[:400] + SHORTENED + ', '.join(ids) + ')'
        self.trimmed_size = sum((len(_dump(m.get('content'))) for m in self.chat))

    def scout_status(self) -> str:
        opened = len([i for i in self.full if i not in self.ref_touched])
        return f'[Kyo] step {self.step} of {self.max_steps} used; {opened} listings opened in full so far, about {3 * len(self.items) + 1} wanted.\n[ledger]\n' + self.ledger()

    def preopen(self, intro: str) -> bool:
        if not self.roles.get('search') or not self.roles.get('ask') or self.msgs_free() <= 0 or self.left() <= 8:
            return False
        committing = {self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse')} - {None}
        start = intro + '\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n' + self.status()
        chat = [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': start}]
        turns = idle = 0
        self.scouting = True
        while turns < SCOUT_TURNS and (not self.finished) and (not self.broke) and (self.left() > 8) and (self.cost < SPEND_SOFT) and (not self.ready_to_ask()):
            turns += 1
            message = self.think(chat, [LIGHT, SPARE], 3000, self.offered())
            if message is None:
                break
            calls = []
            for number, call in enumerate(message.get('tool_calls') or []):
                if not isinstance(call, dict):
                    continue
                fn = call.get('function') if isinstance(call.get('function'), dict) else {}
                raw = fn.get('arguments')
                parsed = raw if isinstance(raw, dict) else _json_obj(raw) if isinstance(raw, str) and raw.strip() else {}
                calls.append({'id': str(call.get('id') or f's{turns}_{number}'), 'type': 'function', 'function': {'name': str(fn.get('name') or ''), 'arguments': _dump(parsed if isinstance(parsed, dict) else {})}})
            content = message.get('content')
            if not isinstance(content, str):
                content = _dump(content) if content else ''
            if not calls:
                idle += 1
                chat.append({'role': 'assistant', 'content': content or '(no call)'})
                chat.append({'role': 'user', 'content': 'Call one of the tools to continue.\n' + self.status()})
                if idle >= 3:
                    break
                continue
            idle = 0
            self.trace.append({'kind': 'scout', 'text': content[:300], 'calls': [c['function'] for c in calls]})
            if any((c['function']['name'] in committing for c in calls)):
                self.trace.append({'kind': 'scout_commit', 'step': self.step})
                chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
                chat += [{'role': 'tool', 'tool_call_id': c['id'], 'content': 'Not run (no step used): the shopper is asked first.'} for c in calls]
                break
            chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
            replies = self.run_calls(calls)
            resolved = self.resolve_pending() if replies and (not self.finished) else ''
            for index, (cid, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += ('\n' + resolved if resolved else '') + '\n' + self.status()
                chat.append({'role': 'tool', 'tool_call_id': cid, 'content': text})
        self.scouting = False
        self.scout_chat = chat
        self.scout_opened = set(self.full)
        self.trace.append({'kind': 'scout_end', 'turns': turns, 'step': self.step, 'opened': len([i for i in self.full if i not in self.ref_touched]), 'ready': self.ready_to_ask()})
        return any((i not in self.ref_touched for i in self.full))

    def interview_opened(self) -> str:
        later = [i for i in self.full if i not in self.scout_opened and i not in self.ref_touched][:16]
        if not later:
            return ''
        text = 'LISTINGS KYO OPENED IN FULL DURING THE INTERVIEW (you have not seen them yet):\n' + '\n'.join((f'{i}: {self.record_of(i)}' for i in later))
        proof = self.evidence([self.full[i] for i in later])
        return text + ('\n' + proof if proof else '')

    def reply_count(self, name: str, seen: Any, shown: int) -> int | None:
        if not isinstance(seen, dict):
            return None
        shape = self.shapes.get(name)
        keys = [shape['count']] if shape and shape.get('count') else [k for k, v in seen.items() if isinstance(v, int) and (not isinstance(v, bool)) and v >= shown and (not re.search('page|token|cursor|offset|after|next|start|begin|limit|size|sort|order', k, re.I))]
        value = seen.get(keys[0]) if len(keys) == 1 else None
        return value if isinstance(value, int) and (not isinstance(value, bool)) else None

    def found_rows(self) -> dict[str, tuple[str, int | None, int]]:
        best: dict[str, tuple[tuple[Any, ...], str, int | None]] = {}
        labels: dict[str, set[str]] = {}
        for number, (name, args, seen) in enumerate(self.search_log):
            ids = [i for i, _ in self.records_in(seen)]
            props = (self.schemas.get(name) or {}).get('properties') or {}
            words = [(k, str(v).strip()) for k, v in args.items() if isinstance(v, str) and v.strip() and (not re.search('page|token|cursor|offset|after|next|start|begin|limit|size|sort|order', k, re.I)) and (not (props.get(k) or {}).get('enum'))]
            if not ids or not words:
                continue
            label = ', '.join((f'{k} "{v}"' for k, v in words))[:120]
            count = self.reply_count(name, seen, len(ids))
            rank = (count is None, count or 0, -len(' '.join((v for _, v in words)).split()), -number)
            for i in ids:
                labels.setdefault(i, set()).add(label)
                if i not in best or rank < best[i][0]:
                    best[i] = (rank, label, count)
        return {i: (b[1], b[2], len(labels[i])) for i, b in best.items()}

    def start_board(self) -> str:
        opened = [i for i in self.full if i not in self.ref_touched]
        cat_key = self.category_key()
        cap = self.budget if isinstance(self.budget, (int, float)) else None
        order = {i: n for n, i in enumerate(self.seen_order)}
        rest = [i for i in self.seen_order if i not in self.full and i not in self.ref_touched and (not self.sold_out(i))]
        cut: dict[int, float] = {}
        if self.brief.get('pick') == 'cheapest':
            for number in range(1, len(self.items) + 1):
                priced = [self.price_of(i) for i in self.fit_fits(number) if self.price_of(i) is not None]
                if priced:
                    cut[number] = min(priced)

        def item_of(ident: str) -> int:
            return self.label_owner(str((self.rows.get(ident) or {}).get(cat_key) or '')) if cat_key else (1 if len(self.items) == 1 else 0)
        dear = [i for i in rest if item_of(i) in cut and self.price_of(i) is not None and self.price_of(i) >= cut[item_of(i)] - 1e-09]
        rest = [i for i in rest if i not in dear]
        found = self.found_rows()

        def source(ident: str) -> str:
            if ident not in found:
                return ''
            label, count, many = found[ident]
            return f' <- found by {label}' + (f' ({count} matches)' if count is not None else '') + (f' and {many - 1} other searches' if many > 1 else '')

        def rank(ident: str) -> tuple[bool, bool, float]:
            kind = not cat_key or bool(self.label_owner(str((self.rows.get(ident) or {}).get(cat_key) or '')))
            price = self.price_of(ident)
            inside = cap is None or (price is not None and price <= cap)
            second = (price if price is not None else float('inf')) if self.brief.get('pick') == 'cheapest' else -order[ident]
            return (not kind, not inside, second)
        rest.sort(key=rank)
        parts = []
        if rest:
            head = f"SEARCH ROWS SEEN AND NOT OPENED YET ({len(rest)}; rows of a requested kind inside the budget first, {'cheapest' if self.brief.get('pick') == 'cheapest' else 'newest'} first; sold-out rows left out). A row shows only part of a listing: open it to read its properties."
            head += " After each row: of the searches that returned it, the one with the fewest matches, and how many other searches returned it. A word search can match words inside a listing's attributes, not only its title, so a row that a search for a value returned can have that value even when the row does not show it."
            lines = [f'{i}: ' + _dump(_clip(self.rows.get(i), 70))[:260] + source(i) for i in rest[:ROWS_SHOWN]]
            if len(rest) > ROWS_SHOWN:
                lines.append(f'(and {len(rest) - ROWS_SHOWN} more rows further down the same order)')
            parts.append(head + '\n' + '\n'.join(lines))
        if dear:
            parts.append(f'({len(dear)} more unopened rows cost at least as much as the cheapest listing of their item already read as fitting every requirement in force; they are left out.)')
        bare = [i for i in opened if i not in self.fit_seen and (not self.sold_out(i))]
        if bare:
            parts.append('OPENED, NOT READ BY THE FIT CHECKER: ' + ', '.join(bare))
        return '\n\n'.join(parts)

    def scouted_view(self) -> str:
        opened = [i for i in self.full if i not in self.ref_touched][:16]
        text = "LISTINGS OPENED IN FULL BEFORE THE SHOPPER WAS ASKED (Kyo's scout opened these; you have not seen them yet):\n" + '\n'.join((f'{i}: {self.record_of(i)}' for i in opened))
        proof = self.evidence([self.full[i] for i in opened])
        if proof:
            text += '\n' + proof
        rest = self.others(opened, 12)
        if rest:
            text += '\n\nOTHER ROWS SEEN IN SEARCHES (not opened):\n' + rest
        if self.searched:
            text += '\n\nSEARCHES MADE: ' + ' | '.join((_dump(a) for a in self.searched))
        return text + '\n\nJudge these against the requirements in force (the ledger below), then search further and open listings that may fit.'

    def can_ask(self) -> bool:
        return not self.asked and bool(self.roles.get('ask')) and self.msgs_free() > 0 and self.left() > 8

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.read_request()
        self.trace.append({'kind': 'brief', 'brief': self.brief})
        opening = self.view.get('opening')
        if opening is not None:
            self.absorb(opening)
        resolved = self.resolve_pending()
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            intro += '\n\nWHAT THE STORE SHOWS FIRST:\n' + self.show(opening)
        if resolved:
            intro += '\n\n' + resolved
        scouted = self.preopen(intro)
        summary = self.interview() if scouted and self.can_ask() else ''
        checked = self.fit_opened() if scouted else ''
        if scouted and self.scout_chat:
            note = (summary or 'The interviewer had nothing to ask.') + ('\n\n' + self.interview_opened() if self.interview_opened() else '') + ('\n\n' + checked if checked else '') + ('\n\n' + self.start_board() if self.start_board() else '') + '\n' + self.status()
        elif scouted:
            intro += '\n\n' + self.scouted_view() + ('\n\n' + summary if summary else '') + ('\n\n' + checked if checked else '') + '\n\n' + self.status()
        else:
            intro += '\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n' + self.status()
        self.main_started = True
        if scouted and self.scout_chat:
            self.chat = list(self.scout_chat)
            if self.chat[-1].get('role') == 'user':
                self.chat[-1] = {'role': 'user', 'content': self.chat[-1]['content'] + '\n\n' + note}
            else:
                self.chat.append({'role': 'user', 'content': note})
        else:
            self.chat = [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': intro}]
        self.work(LOOP_CALLS, SPEND_HARD)
        if not self.finished:
            turned = self.finish(again=True)
            if turned and (not self.finished) and (not self.broke) and (not self.no_credit) and (self.left() > 1):
                extra = self.resolve_pending()
                self.chat.append({'role': 'user', 'content': '[Kyo] Kyo tried to end the episode and ' + turned + ('\n' + extra if extra else '') + '\nApply what the store says, then finish.\n' + self.status()})
                self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
            if not self.finished:
                self.finish()
        self.trace.append({'kind': 'end', 'finished': self.finished, 'steps': self.step, 'cost': round(self.cost, 5), 'calls': self.calls, 'ref_cost': round(self.ref_cost, 5), 'ref_calls': self.ref_calls, 'total_cost': round(self.cost + self.ref_cost, 5)})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.finished and (not self.broke) and (self.left() > 0) and (self.loop_calls < loop_cap) and (self.cost < spend_cap):
            if self.ready_to_ask():
                summary = self.interview()
                checked = self.fit_opened()
                if summary or checked:
                    self.chat.append({'role': 'user', 'content': '\n'.join((p for p in (summary, checked) if p)) + '\n' + self.status()})
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
                checked = self.fit_opened()
                self.chat.append({'role': 'user', 'content': (summary or 'The interviewer had nothing to ask.') + ('\n' + checked if checked else '') + '\n' + self.status()})
                continue
            self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
            self.trace.append({'kind': 'turn', 'text': content[:300], 'calls': [c['function'] for c in calls]})
            replies = self.run_calls(calls)
            resolved = self.resolve_pending() if replies and (not self.finished) else ''
            refreshed = self.fit_refresh() if replies else ''
            for index, (cid, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += ('\n' + resolved if resolved else '') + ('\n' + refreshed if refreshed else '') + '\n' + self.status()
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
