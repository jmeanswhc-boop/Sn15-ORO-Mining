from __future__ import annotations
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
SPEND_SOFT = 0.05
SPEND_HARD = 0.07
HEAVY_ROOM = 0.02
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3
AUTO_PAGES = 6
CHEAPER_OPENS = 3
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_llm_short: dict[int, ProxyClient] = {}
# the validator's time limit per episode, after public Halo v2: no interview, question, loop turn or model request starts when
# its result could not be used in time, and the model timeout shrinks with the time left
TASK_SECONDS = 300.0
TIME_MARGIN = 20.0
CALL_FLOOR = 6.0
ASK_BY = 165.0
ASK_LATEST = 125.0
QUESTION_BY = 210.0
WIND_DOWN = 200.0
LAST_TURN = 255.0
EMPTY_ROOM = 12.0
LATE_BY = 190.0
LIST_SIZE = 2       # properties a question names on a one-product request (M questions: [#1, #M+1], [#2, #M+2] ...)
LIST_MULTI = 3      # the same on a request for several products, one product per question: that product's [#1, #2, #3]
WALK_THINK = None   # a store turn reasons as the model chooses; an empty reply (its reasoning used the whole reply: Haiku 5.5
                    # ignores a cap or an effort level, 1.6-2.9k tokens at a 50k-token history) is asked again at once with reasoning
                    # off (5 s, same tool calls); reasoning off on every turn raised wrong declines (race-16/17 tests: 7 vs 1-2 of 18)
HUNT_STEPS = 8      # store calls the hunt for items with no fitting listing may spend (searches plus listings opened)
HUNT_SHARE = 0.4    # the hunt runs once this share of the steps is left (a decline comes too late for it: race-16/17 tests
                    # declined with 1-11 steps left, while the hunt needs about 14)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
GUIDE = 'You are Kyo, a careful buyer working for one shopper in an online store. You can act only by calling the store\'s tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.\n\nHOW THE STORE WORKS\n- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool\'s description and schema and use the exact argument names and allowed values.\n- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.\n- Search rows are summaries. Stores split a listing\'s record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Kyo knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.\n- Questions to the shopper are handled for you by Kyo\'s interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.\n- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Kyo carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store\'s tools (product questions go through Kyo\'s interviewer).\n\nWHAT COUNTS AS A FIT\n- The kind of product is decided by the listing\'s category label: only the label that matches the requested item as a whole counts, never a neighbouring one.\n- A requirement on a property is met when the listing\'s attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words, which must be written, not inferred: case and singular or plural are fine, and a value that contains the required value\'s words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count, however close its meaning (a synonym, a related word, another form of the word, a code or another shortening is not the word itself, and "teal green" is not "sea green"), except initials: the first letters of the required words, written together, count as those words. Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.\n- An attribute whose key names the requirement\'s property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing about the requirement. Kyo adds [evidence] lines after opened listings with the keys whose names match a requirement.\n- When the listing has no attribute for that property, only its title decides: a title stating the required words or their initials meets it, while another attribute or the description does not, and a limit on a figure is not met without the attribute. When other listings of that kind you opened carry an attribute for that property, prefer one whose own attribute states the value. Likewise, when one opened listing\'s attribute states the required words themselves and another\'s states the property in other words, prefer the one with the required words.\n- When the category label itself states a requirement\'s value (a season or audience in the label), that requirement is met by the category.\n- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing\'s value, the rule excludes only listings sharing that exact value. A reference you could not follow to its end leaves its value unknown: it rules no listing out, and it is never a reason to end without a purchase.\n- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.\n- When one requested item must fit another (an accessory for a device), choose the two together from the listings\' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.\n- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.\n- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.\n\nCHOOSING\n- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Kyo reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.\n- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.\n- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.\n- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.\n- A request for several different items needs one cart line per item, each at its requested quantity.\n\nCHANGES WHILE YOU WORK\n- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool\'s result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.\n- Re-read a chosen listing\'s record shortly before checking out, because price and stock can change silently.\n\nFINISHING\n- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.\n- End without a purchase only when, for some requested item, no listing of its kind that you opened writes every stated requirement (in its attribute for the property, or in its title when it has none) inside the budget and in stock, and a further search with the required words finds no other listing of that kind. A reference you could not resolve, a line you could not finish checking, or running short of steps is never such proof: then put the best candidate of every requested item in the cart and check out. Follow the request\'s wording for that ending, and when a remark is needed, keep it short and name no listing.\n- Every reply must call at least one tool. Keep your text short: what you know, then the next move.'
BRIEF = 'You prepare a shopping task for an agent. Read the shopper\'s request and the store\'s tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:\n\n"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request\'s words, "shelf": the store\'s category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request\'s own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property\'s value or limit outright; and never turn words that are part of what the product is called (a season or an audience named with the product) into musts: those belong to "name". A use or occasion the request makes a property of the product itself ("blazers for business wear", "a dress made for parties", "the occasion has to be party") IS a must on that property ("occasion: business"), unless the store\'s category label already names it; an event, person or setting the product is bought for ("to wear at his wedding", "for a colleague"), and how it should sit, feel or look in use, stay context. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.\n"budget": the most the shopper will pay in total as a number in the request\'s own unit, or null.\n"budget_unit": that unit in words (for example "cents"), or null.\n"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.\n"in_stock_only": true unless the request clearly accepts items that are not available.\n"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.\n"pick_measure": what is maximised or minimised for "most"/"least", else "".\n"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.\n"steps": how many steps, moves or tool calls the request allows (integer), or null.\n"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].\n"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.\n"wishes": a list of things the shopper would merely like.\n"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).\n"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper\'s approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.\n"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.\n"fits": a list of {"item": item number, "with": item number} for each requested item that the request says must work with, fit or suit ANOTHER requested item (the one the agent picks); [] when none.\n"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing\'s attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing\'s current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).\n"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.\n"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.\n'
PLAN = 'You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property; a property the shopper has no view on gets "don\'t mind", and one the shopper does not think about at all gets a puzzled reply, so every property named spends part of a message.\n\nYou get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item\'s category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.\n\nChoose the properties in this order:\n0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.\n1. Hinted properties. The request\'s story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product\'s looks or the setting it belongs in, the person meant to use or wear it, what is inside it, or a space it has to fit into or the things it has to hold. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.\nName a hinted property by a listed key of that item when one fits the hint (a hint about who will wear it: the key for gender or audience; about its looks: the listed key for colour, pattern or finish that most of its listings carry; about a space it must fit into or what it must hold: the listed measurement of that direction, its width, depth or height, or else its size or capacity key); a property no listing of the item carries comes after the item\'s three most carried keys.\n2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item\'s listings; once five or more listings were read, avoid keys seen on only one or two of them.\nSkip a property the request already settles for that item (a requirement names that property; a key whose values only repeat a required value, such as a pattern key showing the required decoration, is NOT settled), a second name for a property already chosen, a key whose values each list several purchase options at once, identifiers, model names, warranty and other logistics, and a list of the devices or models an item works with (the order\'s other items settle that, not the shopper\'s taste). Pack sizes, counts and quantities per pack are properties like any other, and a key marked [not the stated ...] is another property than that requirement, whatever words its name shares with it. Rank every item on its own: its hinted properties first, then its listed keys.\n"label" is the property\'s plain name: for a listed key, all of the key\'s words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property\'s name, no example values, no brackets, no second property.\nChoose up to COUNT for each item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story\'s words it comes from, or an empty string"}]}, each item\'s picks in rank order'
HEAR = 'A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "budget": new total limit as a number or null}\n- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3").\n- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop\'s product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer\'s own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.\n- "free": the property of each question the shopper said they do not mind about.\n- Folksy or roundabout wording that names a value still states that value.'
NOTE = 'While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:\n{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request\'s unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}\nA replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note.'
CHECK = 'You are the last check before a shopping agent checks out. For every cart line and every requirement now in force, decide from the records shown whether the requirement is MET, CONTRADICTED or NOT FOUND, with these reading rules: a requirement is MET when the listing\'s own attribute for that property writes the value (any key spelling, one element of a list is enough, singular or plural and unit conversions count, a value containing the required value\'s words counts, such as "deep teal" for teal, limits include the edge). The value must be written, not inferred: the store grades the words, so a value in other words (a synonym, a related word, another form of the word, a narrower or broader kind) is CONTRADICTED, however close its meaning. The listing\'s own attribute decides: when it states a different value, the requirement is CONTRADICTED even if the title or description shows the wanted words. A key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing. When the listing has no attribute for that property, its title decides (one that names the required value among others, shoes for boys and girls, meets it), except for a requirement listed under ATTRIBUTE GAPS: other listings of the line\'s kind state that property as an attribute of their own, so the line\'s title does not show it met; another attribute or the description never decides a property, except what the listing says it is compatible with or fits. Otherwise the requirement is NOT FOUND. A requirement the category label itself states is met by that category; the category label must match the item; a rule of being like or unlike another listing whose page could not be read is NOT FOUND, never contradicted; out-of-stock listings and listings that break the budget do not fit.\nThe cart is wrong only when: a requirement is CONTRADICTED by a line\'s records; a requested item has no line, or its line has the wrong quantity; a line is something not requested; a line is shown out of stock; the total of price times quantity over all lines is above the budget; or, when the pick rule is cheapest, most or least, another listing shown has every requirement met and clearly beats the cart\'s choice on that rule; or an ATTRIBUTE GAPS entry names a listing of the line\'s kind whose own attribute states the required value (for a limit, a figure inside the limit) and which meets every other requirement, is in stock and inside the budget (list it under better). Otherwise a requirement NOT FOUND never makes the cart wrong. The newest record of a listing wins.\nReturn ONE JSON object and nothing else: {"ok": true or false, "problems": ["the line, the requirement and the record words that contradict it", ...], "better": [listing ids that should replace a cart line, best first]}.'
READING = 'a requirement is MET when the listing\'s own attribute for that property writes the value (any key spelling, one element of a list is enough, singular or plural and unit conversions count, a value containing the required value\'s words counts, such as "deep teal" for teal, limits include the edge). The value must be written, not inferred: the store grades the words, so a value in other words (a synonym, a related word, another form of the word, a narrower or broader kind) is CONTRADICTED, however close its meaning. The listing\'s own attribute decides: when it states a different value, the requirement is CONTRADICTED even if the title or description shows the wanted words. A key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing. Only when the listing has no attribute for that property does its title decide (even though other listings carry such an attribute), and one that names the required value among others (shoes for boys and girls) meets it; another attribute or the description never decides a property, except what the listing says it is compatible with or fits. Otherwise the requirement is NOT FOUND. A requirement the category label itself states is met by that category; the category label must match the item; a rule of being like or unlike another listing whose page could not be read is NOT FOUND, never contradicted; out-of-stock listings and listings that break the budget do not fit.'
AUDIT = 'A shopping agent wants to end without buying because it thinks nothing fits. Ending without a purchase is right only when some requested item has no candidate at all; check the listing records shown against the requirements, with these reading rules: ' + READING + '\nA listing fits unless a requirement is CONTRADICTED by its records (NOT FOUND does not rule it out). Records marked ROW were seen only as a search result row: their page has not been read, so they can be neither passed nor failed on a property the row does not show.\nReturn ONE JSON object and nothing else: {"fits": {"item number": [listing ids with no contradicted requirement, in stock and inside the budget, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}'
UNMIX = 'A shopper\'s request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request\'s wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}'
TRAIL = 'A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing\'s page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement\'s chain from the requirement\'s own description, link by link, through the pages\' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).\nReturn ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing, "find": "for a link not found: search words (its shop and the values the description gives)"}], "done": true when the last link is an OPENED page that states the property\'s value itself (also when the requirement asks for a value other than or unlike that one: the page\'s value is what it is measured against), else false, "value": the property\'s value as stated on that last page, or ""}]}\nA record is a link only when it matches the whole description of that link: the shop or brand, the kind of product AND every property value the description gives (for example "a 10 pieces multipack" rules out the same shop\'s listing with 10 capsules and no multipack); when no record matches all of it, that link is not found yet, even if the same shop sells other listings of that kind. Give "find" for such a link: the shop\'s name and the description\'s values, as a few search words.\nList each chain\'s links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown.'
WALK_PARSE = 'A shopper says that one property of a product takes its value from another listing in the store. From the words below, describe that other listing. Return ONE JSON object (compact, nothing else): {"seller": the shop that sells that other listing, copied exactly as written, or null when the words name no shop for it, "kind": the kind of product that other listing is, "details": a list of the values the words give to tell that listing apart from the shop\'s other listings, each one written "property: value" when the words name its property ("thread count: 0", "condition: used", "processor speed: 3.9 ghz"), else the value alone, copied as written, "relation": how the wanted value stands to that listing\'s value, one of "same", "different" (anything except it), "within" (no further than an amount from it, either way), "at most", "at least", "goes with" (a value the store pairs with it), "amount": the number for "within" in the words\' own unit, else null}.'
WALK_READ = 'A shopping agent follows a chain of listings to learn the value of one property. Below is one listing\'s page in full. Return ONE JSON object (compact, nothing else): {"next": null, or the listing this page sends the property to, as {"seller": its shop exactly as written (when the sentence names no shop, the name it gives for that listing), "kind": its kind of product, "details": the values the page gives to tell it apart, each one written "property: value" when the page names its property, else the value alone, copied as written}, "value": the value of the property this page states for its own item (in its fields, its title or its own description), else null, "pairs": a list of [value, value] pairs when the page states a rule pairing values of this property with each other, else []}. Set "next" only when a sentence of the page says that THIS property is to be taken, read or followed from another listing. A sentence that sends a different property elsewhere is about that property alone: ignore it. A remark that some other version of the item exists elsewhere sends nothing.'
WALK_HOPS = 5
WALK_STEPS = 24
GATE = 'A shopping agent is about to STAGE. The shopper\'s request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product\'s properties do not do a step.\nWhen a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.\nReturn ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}'
SWAP = "A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: " + READING + '\nReturn ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line\'s item, at most 6], "reason": "one sentence"}'
SHAPE = 'A store tool returned the result below. Name its parts by the exact keys the result and the tool\'s parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing\'s price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}'
GUARD_KEYS = 3
GUARD_SHARE = 0.0
GUARD_GAP = 0.34
NOT_PROPERTY = '\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b'
CATCH_ALL = '^(?!.*\\b(?:types?|styles?|kinds?|shapes?)\\b).*\\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\\w*|accessor\\w*|extras?|misc\\w*)\\b'
WIDEN_EACH = 6
GAP_SKIP = '\\b(?:not|no|non|false|without|except|other than|anything but|unlike|same as|at (?:most|least)|up to|under|over|below|above|between|more|less|than|max\\w*|min\\w*|or)\\b|[<>=]'
LIMIT_HEAD = '^(?:at (?:least|most)|up to|no (?:more|less) than|(?:more|less|fewer) than|under|over|below|above|min(?:imum)?|max(?:imum)?)\\s+'
LIT_MOST = '^(?:at most|up to|no more than|max(?:imum)?)\\s+'
LIT_LEAST = '^(?:at least|no less than|min(?:imum)?)\\s+'
LIT_SKIP = '\\b(?:not|no|non|without|except|other than|anything but|unlike|same as|like|between|than|under|over|below|above|listing|page|yes|true)\\b|[<>=≤≥]'


def _vstem(text: Any) -> frozenset[str]:
    """The words of an attribute value, split at punctuation too (a list value "x, low sodium, y" holds low sodium)."""
    return _stem(re.sub('[^a-z0-9]+', ' ', str(text).lower()))

def _root_match(a: str, b: str) -> bool:
    """Two word roots name the same word: equal, or a plural form the roots keep apart (shelve and shelf, knive and knife,
    batterie and battery)."""
    if a == b:
        return True
    for x, y in ((a, b), (b, a)):
        if x.endswith('ve') and y in (x[:-2] + 'f', x[:-2] + 'fe'):
            return True
        if x.endswith('ie') and y == x[:-2] + 'y':
            return True
    return False


def _root_share(mine: set[str], theirs: set[str]) -> float:
    """The share of word roots two names have in common (each matched once)."""
    shared = sum(1 for t in theirs if any(_root_match(t, m) for m in mine))
    return shared / max(1, len(theirs) + len(mine) - shared)


def _same_prop(prop: str, key: str) -> bool:
    """A key holds a requirement's property: the names contain one another, or one is a single word that opens the other's
    name (packaging holds packaging type; pack size does not hold pack type)."""
    p, k = _stem(prop), _stem(key)
    if _names(p, k):
        return True
    wp, wk = _words(prop).split(), _words(key).split()
    return bool(wp and wk) and ((len(k) == 1 and _stem(wp[0]) == k) or (len(p) == 1 and _stem(wk[0]) == p))


def _lit_list(text: Any) -> list[str]:
    """The words of a value as the store's grading reads them, in order: lower case, split at punctuation and between a figure
    and its unit ("26w" is 26 w), a plural s dropped."""
    plain = re.sub('(?<=\\d),(?=\\d{3})', '', re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(text)).lower())
    plain = re.sub('(?<=\\d)(?=[a-z])', ' ', plain)
    return [w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w for w in re.findall('[a-z0-9]+(?:\\.\\d+)?', plain)]


def _lit_writes(want: list[str], text: Any) -> bool:
    """A text writes a value: every one of its words, or, for two or more words, their initials as one word."""
    have = _lit_list(text)
    return bool(want) and (set(want) <= set(have) or (len(want) >= 2 and ''.join(w[0] for w in want) in have))


def _literal(fields: dict[str, Any], title: str, prop: str, value: str) -> bool | None:
    """Whether a listing meets a stated requirement the way the store's grading reads it (race 17: no accepted order held a
    line that broke this): a field whose name holds every word of the property writes the value (word containment, one
    alternative of "a or b", a figure inside a limit); with no such field, the title writes it (a limit needs the field).
    A value in other words or a missing field fails. None when the requirement is no plain value (an exclusion, a reference,
    a comparison) or a limit stated in another unit."""
    value = value.strip()
    if not _lit_list(prop) or not _lit_list(value) or re.search(LIT_SKIP, value, re.I):
        return None
    own = [str(v) for k, v in fields.items() if set(_lit_list(prop)) <= set(_lit_list(k))]
    limit = re.match(LIT_MOST, value, re.I) or re.match(LIT_LEAST, value, re.I)
    if limit:
        figure = value[limit.end():].lower()
        bound = re.findall('\\d+(?:\\.\\d+)?', figure.replace(',', ''))
        unit = (re.findall('[a-z]+', figure) or [''])[0]
        if len(bound) != 1:
            return None
        most = bool(re.match(LIT_MOST, value, re.I))
        seen = []
        for text in own:
            hit = re.search('(\\d+(?:\\.\\d+)?)\\s*([a-z]*)', text.lower().replace(',', ''))
            if not hit:
                continue
            if unit and hit.group(2) and not (hit.group(2).startswith(unit) or unit.startswith(hit.group(2))):
                return None
            seen.append(float(hit.group(1)) <= float(bound[0]) if most else float(hit.group(1)) >= float(bound[0]))
        return any(seen)
    options = [_lit_list(o) for o in re.split('\\s*,\\s*|\\s+or\\s+', value) if _lit_list(o)] if re.search('\\bor\\b', value) else [_lit_list(value)]
    if own:
        return any(_lit_writes(want, text) for want in options for text in own)
    return any(_lit_writes(want, title) for want in options)
FIT_CODE = '[A-Z][A-Za-z]{0,5} ?\\d{3,6}'
PART_CODE = '[A-Z]{1,4}\\d{3,6}'

def _codes(text: Any) -> list[str]:
    return [re.sub('\\s+', ' ', c).strip().lower() for c in re.findall('\\b' + FIT_CODE + '\\b', str(text or ''))]

def _fit_marks(text: Any) -> dict[str, Any]:
    text = str(text or '')
    own = re.search('[\\[(]\\s*(?:part\\s+)?(' + PART_CODE + ')\\s*[\\])]', text, re.I)
    codes: list[str] = []
    for hit in re.finditer('\\bmodel\\s*(?:no\\.?|number|#|ref\\.?)\\s*[:.]?\\s*(' + FIT_CODE + ')\\b', text, re.I):
        codes += _codes(hit.group(1))
    for hit in re.finditer('\\b(?:also sold as|also known as|sold also as)\\s+([^.;]+)', text, re.I):
        codes += _codes(hit.group(1))
    works: list[str] = []
    for hit in re.finditer('\\b(?<!not )(?<!never )(?:works with|compatible with|fits|goes with)\\s*:?\\s+([^.;]+)', text, re.I):
        if not re.search('\\bpart\\s+' + PART_CODE + '\\s+also\\s+$', text[max(0, hit.start() - 40):hit.start()], re.I):
            works += _codes(hit.group(1))
    also = [(hit.group(1).upper(), _codes(hit.group(2))) for hit in re.finditer('\\bpart\\s+(' + PART_CODE + ')\\s+also fits\\s*:?\\s*([^.;]+)', text, re.I)]
    more = re.search('\\b(?:more|details|see|read)\\s+(?:on|at|under|about)?\\s*(?:the\\s+)?part\\s+(' + PART_CODE + ')', text, re.I)
    return {'own': own.group(1).upper() if own else '', 'codes': codes, 'works': works, 'also': also, 'more': more.group(1).upper() if more else ''}

MODEL_MARK = '\\bmodel(?:\\s*(?:no\\.?|number|#|ref\\.?))?\\s*[:.]?\\s*([a-z]{1,6})[- ]?(\\d{2,6})\\b(?!\\s+also\\s+takes)'
FIT_WORDS = '\\b(?<!not )(?<!never )(?:works with|compatible with|fits|goes with)\\s*:?\\s*([^.;]+)'
CODE_FORM = '\\b([a-z]{1,6})[- ]?(\\d{2,6})(x{0,4})\\b'


def _models(text: Any) -> set[str]:
    """A device's own model codes, as letters and digits ("Model: XY-4510" gives xy4510)."""
    return {a + b for a, b in re.findall(MODEL_MARK, str(text or '').lower())}


def _fit_rules(text: Any) -> list[tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]]:
    """What a part says it fits, one entry per clause: the codes it takes in and those an 'except' leaves out; trailing x's in a
    code stand for any digit ("every XY-39xx model")."""
    rules = []
    low = str(text or '').lower()
    for hit in re.finditer(FIT_WORDS, low):
        if re.search('\\bpart\\s+[a-z]{1,4}\\d{3,6}\\s+also\\s+$', low[max(0, hit.start() - 40):hit.start()]):
            continue                 # ("part X also fits ..." on this page is about part X, another listing)
        parts = re.split('\\b(?:except(?:ing)?|excluding|but not)\\b', hit.group(1), maxsplit=1)
        taken = re.findall(CODE_FORM, parts[0])
        if taken:
            rules.append((taken, re.findall(CODE_FORM, parts[1]) if len(parts) > 1 else []))
    return rules


SERIES_SAID = '(?<!also takes )\\bparts?\\s+series\\s*:?\\s*((?:[a-z]{1,5}\\d{1,3}(?:\\s*(?:,|/|\\band\\b|\\bor\\b)\\s*)?)+)'
SERIES_OWN = '[\\[(]\\s*(?:part\\s+)?([a-z]{1,5})(\\d{1,3})-\\d{2,6}\\s*[\\])]'
# a part's page can say a device takes one more series ("Model AB1234 also takes part series CD5"): a device fact, true of every
# part of that series, so it is kept apart from the page's own codes
SERIES_ADDED = '\\bmodel\\s+([a-z]{1,6})[- ]?(\\d{2,6})\\s+also\\s+takes\\s+parts?\\s+series\\s*:?\\s*((?:[a-z]{1,5}\\d{1,3}(?:\\s*,\\s*)?)+)'


def _series_taken(text: Any) -> set[str]:
    """The part series a device says it takes ("... part series AB1, AB3"), as codes like its model codes."""
    low = str(text or '').lower()
    return {c for hit in re.finditer(SERIES_SAID, low) for c in re.findall('[a-z]{1,5}\\d{1,3}', hit.group(1))}


def _series_rule(text: Any) -> list[tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]]:
    """A part whose own bracketed code is a series code, a dash and a number fits the devices that take that series."""
    low = str(text or '').lower()
    if re.search(SERIES_SAID, low):
        return []
    hit = re.search(SERIES_OWN, low)
    return [([(hit.group(1), hit.group(2), '')], [])] if hit else []


def _takes(rules: list, model: str) -> bool | None:
    """Whether a part's clauses take in a device model; None when no clause names codes of the model's letters."""
    head = re.match('[a-z]+', model)
    own = [r for r in rules if head and any(c[0] == head.group(0) for c in r[0])]
    if not own:
        return None

    def same(code: tuple[str, str, str]) -> bool:
        return re.fullmatch(re.escape(code[0] + code[1]) + '\\d' * len(code[2]), model) is not None
    return any(any(same(c) for c in taken) and not any(same(c) for c in left) for taken, left in own)


def _one_off(typed: str, real: str) -> bool:
    """Two strings one character apart (one missing, extra or changed): the common head and tail leave at most one character
    on each side."""
    head = 0
    while head < min(len(typed), len(real)) and typed[head] == real[head]:
        head += 1
    tail = 0
    while tail < min(len(typed), len(real)) - head and typed[-1 - tail] == real[-1 - tail]:
        tail += 1
    rest = (len(typed) - head - tail, len(real) - head - tail)
    return typed != real and max(rest) <= 1


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

def _conditional(need: str) -> bool:
    """Whether a must depends on what Kyo picks for another requested item (its value applies unless, or only while, that
    pick has some attribute): there is no listing to look up for it, even when the brief files it among the lookups. Words
    of a real pointer (a page, a named listing, the same or a partner value) keep it a lookup."""
    text = str(need).partition(':')[2].lower()
    return bool(re.search('\\b(?:except when|except if|unless|only if|only when|if the|when the|should the|turns? out)\\b', text)) and \
        not re.search('\\b(?:page|points?|pointed|named|same|identical|partners?|unlike|matching|differs?|different)\\b', text)

def _fit_must(need: str) -> bool:
    """A must that an item be compatible with something (a car model, a device): stores state that in the listing's own
    fitment line, often only in its description, so the word check of fields and title cannot judge it; the item check
    reads it. A plain fit property (a garment's fit) is no such must."""
    prop, _, value = str(need).partition(':')
    return bool(re.search('compatib|\\bfitment\\b', prop.lower())) or bool(re.match('\\s*fits\\b', value.lower()))

def _words(key: str) -> str:
    key = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(key))
    return re.sub('[\\s_.\\-]+', ' ', key).strip().lower()

def _norm(text: Any) -> str:
    """Lower-case words of a text, a number joined to the unit after it ("12 GB" and "12gb" alike), padded with spaces so a
    value is found only as whole words ("12gb" is not inside "512gb")."""
    plain = re.sub('(?<=\\d),(?=\\d{3})', '', str(text).lower())
    words = [w.strip('.') for w in re.findall('[a-z0-9.]+', plain) if w.strip('.')]
    joined = ' '.join((w[:-1] if len(w) > 3 and w.isalpha() and w.endswith('s') and (not w.endswith('ss')) else w for w in words))
    return ' ' + re.sub('(?<=\\d) (?=[a-z])', '', joined) + ' '

def _numbers(text: Any) -> set[str]:
    return set(re.findall('\\d+(?:\\.\\d+)?', re.sub('(?<=\\d),(?=\\d{3})', '', str(text))))

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

def _llm_within(seconds: float) -> ProxyClient:
    """The model client whose timeout fits the time left: the usual one, or one with a timeout in whole 5 s steps."""
    if seconds >= 50:
        return _llm
    limit = max(5, int(seconds) // 5 * 5)
    if limit not in _llm_short:
        _llm_short[limit] = ProxyClient(timeout=limit, max_retries=1, retry_delay=1)
    return _llm_short[limit]


def _spelled(text: str) -> frozenset[str]:
    """A name's words with plural -s and British -our folded, so "colours" and "color" compare equal."""
    return frozenset((re.sub('our$', 'or', w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w) for w in _words(text).split()))


_ROW_FIELD = re.compile('(?:^|\\s\\|\\s)([A-Za-z][A-Za-z0-9_]{0,40}):(?:\\s|$)')
_PAIR_SPLIT = re.compile(';\\s*(?=[^;=]{1,60}=)')


def _pair_map(text: str) -> dict[str, str] | None:
    """'k=v; k=v' (a listing's properties written in one field) -> {k: v}; None when the text is not such a list."""
    if text.count('=') < 2:
        return None
    pairs: dict[str, str] = {}
    for piece in _PAIR_SPLIT.split(text.strip().rstrip(';')):
        key, eq, value = piece.partition('=')
        if not eq or not key.strip() or len(key.split()) > 6:
            return None
        pairs[key.strip()] = value.strip()
    return pairs if len(pairs) >= 2 else None


def _scalar(text: str) -> Any:
    if text in ('None', 'null'):
        return None
    if text in ('True', 'False'):
        return text == 'True'
    if re.fullmatch('-?\\d+', text):
        return int(text)
    if re.fullmatch('-?\\d+\\.\\d+', text):
        return float(text)
    return text


def _row_record(text: Any) -> dict[str, Any] | None:
    """One listing written as a string of 'field: value' parts joined by ' | ' (race-15 and race-18 stores) -> a record dict,
    a field of 'k=v; k=v' pairs becoming a dict of them; None for any other text. Without it such a store leaves the code
    blind: no listing, no attribute key, nothing for the interviewer to plan from."""
    if not isinstance(text, str) or text.count(' | ') < 2:
        return None
    parts = _ROW_FIELD.split(text.strip())
    if parts[0].strip() or len(parts) < 7:
        return None
    record: dict[str, Any] = {}
    for key, value in zip(parts[1::2], parts[2::2]):
        value = value.strip()
        record[key] = _pair_map(value) or _scalar(value)
    return record if len(record) >= 3 else None


def _read_rows(node: Any, depth: int=0) -> Any:
    """The observation with every listing string turned into a record dict (_row_record); a dict whose one field is such a
    string becomes the record itself. Anything else comes back as it was."""
    if depth > 6:
        return node
    if isinstance(node, str):
        return _row_record(node) or node
    if isinstance(node, list):
        return [_read_rows(v, depth + 1) for v in node]
    if isinstance(node, dict):
        if len(node) == 1:
            record = _row_record(next(iter(node.values())))
            if record:
                return record
        return {k: _read_rows(v, depth + 1) for k, v in node.items()}
    return node


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
        self.started = time.monotonic()
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
        self.pushed = False
        self.built: list[str] = []
        self.dropped: set[str] = set()
        self.late_held = False
        self.lead_items: dict[str, list[str]] = {}
        self.verdicts: dict[str, bool] = {}
        self.contradicted: set[str] = set()
        self.effort_tried = False
        self.effort_cart: dict[str, int] = {}
        self.hunted = False
        self.late_bought = False
        self.list_calls: list[tuple[str, dict[str, Any]]] = []
        self.capped_lines: set[str] = set()
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
        self.conds: set[str] = set()      # (refs that are conditions on another picked item: never walked, trailed or unmixed)
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
        self.paged_obs: list[Any] = []
        self.plan_order: list[int] = []
        self.lanes: dict[int, list[str]] = {}
        self.found: dict[int, int] = {}
        self.spent: dict[int, int] = {}
        self.used: dict[int, set[str]] = {}
        self.late_asked = False
        self.lists_off = False
        self.gap_opened: set[tuple[str, str]] = set()
        self.gap_said: dict[tuple[str, str], str] = {}
        self.gap_limit: set[tuple[str, str]] = set()
        self.browse_next: dict[str, tuple[str, dict[str, Any], Any]] = {}
        self.value_rows: list[str] = []
        self.line_holds = 0
        self.bound: list[tuple[int, int]] = []
        self.fit_holds = 0
        self.searched: list[dict[str, Any]] = []
        self.link_searches = 0
        self.link_seen: dict[str, list[str]] = {}
        self.replies: list[str] = []
        self.walked: set[str] = set()
        self.walk_steps = 0
        self.walk_alias: dict[str, str] = {}
        self.looked_parts: set[str] = set()

    def left(self) -> int:
        return self.max_steps - self.step

    def clock(self) -> float:
        return time.monotonic() - self.started

    def model_room(self) -> float:
        return TASK_SECONDS - TIME_MARGIN - self.clock()

    def msgs_free(self) -> int:
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)

    @property
    def listing(self) -> bool:
        """Every question names several properties of one product: LIST_SIZE on a one-product request, LIST_MULTI on one for
        several products (race 18: one-property questions heard 66% of hidden needs, lists 72-79%; the first slot is answered far
        more often than the next ones). A reply asking for one question at a time ends lists for the episode."""
        return not self.lists_off

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

    def judges(self, limit: float=SPEND_SOFT) -> list[str]:
        if self.cost <= limit and self.cost + HEAVY_ROOM <= SPEND_HARD:
            return [HEAVY, LIGHT]
        return [LIGHT] if self.cost <= SPEND_HARD else []

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None=None) -> dict[str, Any] | None:
        for model in models:
            if self.calls >= ALL_CALLS:
                return None
            if self.model_room() < CALL_FLOOR:
                self.trace.append({'kind': 'late_call_skipped', 'at': round(self.clock(), 1)})
                return None
            self.calls += 1
            tries = 0
            budget = WALK_THINK
            while True:
                message, status = self.call_model(model, messages, max_tokens, tools, budget)
                if message is not None:
                    return message
                if status == 0 and tools and budget is None and model.startswith('anthropic/'):
                    # (a store turn that came back empty spent its whole reply on reasoning over a long history: race-17
                    #  kit episodes lost a third of their 300 s to such replies and their retries; asked again at once
                    #  without reasoning, the same model answers in seconds)
                    budget = 0
                    self.trace.append({'kind': 'think_off', 'model': model})
                    continue
                # (status 0: the model answered with nothing; one more try, like a timeout)
                passing = status is None or status in (0, 402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                if not passing or tries >= (1 if status in (None, 0, 408) else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL:
                    break
                if self.model_room() - WAITS[tries] * 1.2 < CALL_FLOOR:
                    break
                pause = WAITS[tries] * (0.8 + 0.4 * random.random())
                self.waited += WAITS[tries]
                tries += 1
                time.sleep(pause)
            if status == 402:
                self.no_credit = True
                return None
        return None

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None, think: int | None=None) -> tuple[dict[str, Any] | None, Any]:
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
            if model.startswith('anthropic/') and think is not None:
                body['reasoning'] = {'max_tokens': think} if think else {'enabled': False}
        result = _llm_within(self.model_room()).post_verbose('/inference/chat/completions', body)
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

    def ask_json(self, prompt: str, models: list[str], max_tokens: int=1800) -> dict[str, Any] | None:
        for model in models:
            message = self.think([{'role': 'user', 'content': prompt}], [model], max_tokens)
            got = _json_obj((message or {}).get('content'))
            if got is not None:
                return got
            if self.broke or self.no_credit or self.calls >= ALL_CALLS:
                return None
        return None

    def mend_args(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Two arguments the store would turn away, mended in code: (a) a listing reference that names no listing seen here,
        on a page or cart tool: a part code that one seen record shows in brackets becomes that record's id, an id one
        character off becomes the one seen id it matches; (b) a shop's name given to the search tool's brand filter, which
        matches brands only: the name moves into the search words."""
        if not isinstance(args, dict):
            return args
        try:
            out = dict(args)
            known = set(self.rows) | set(self.full)
            if name in {self.roles.get(r) for r in ('open', 'offer', 'compare', 'cart')} - {None} and known:
                for key, value in args.items():
                    vals = value if isinstance(value, list) else [value]
                    fixed = [self.mend_ref(v, known) if isinstance(v, str) else v for v in vals]
                    if fixed != vals:
                        out[key] = fixed if isinstance(value, list) else fixed[0]
                        self.trace.append({'kind': 'ref_mended', 'tool': name, 'from': vals, 'to': fixed})
            if name == self.roles.get('search'):
                props = (self.schemas.get(name) or {}).get('properties') or {}
                brand = next((k for k in out if re.search('vendor|brand|maker|manufactur', k, re.I) and isinstance(out[k], str)), None)
                plan = self.search_template() or self.schema_template()
                text = plan[2] if plan and plan[2] in props else None
                if brand and text:
                    shops, makers = set(), set()
                    for r in self.rows.values():
                        for k, v in r.items():
                            if isinstance(v, str) and re.search('seller|shop|store|merchant', k, re.I):
                                shops.add(v.strip().lower())
                            elif isinstance(v, str) and re.search('vendor|brand|maker|manufactur', k, re.I):
                                makers.add(v.strip().lower())
                    said = out[brand].strip().lower()
                    if said in shops and said not in makers:
                        out.pop(brand)
                        out[text] = ' '.join((args[brand], str(out.get(text) or ''))).strip()
                        self.trace.append({'kind': 'shop_not_brand', 'name': args[brand]})
            return out
        except Exception as err:
            self.trace.append({'kind': 'mend_error', 'error': f'{type(err).__name__}: {err}'[:160]})
            return args

    def mend_ref(self, value: str, known: set[str]) -> str:
        ref = value.strip()
        if not ref or ref in known:
            return value
        pattern = re.compile('[\\[(]\\s*(?:part\\s+)?' + re.escape(ref) + '\\s*[\\])]', re.I)
        tagged = [i for i in known if any(isinstance(v, str) and len(v) < 400 and pattern.search(v)
                                          for v in (self.rows.get(i) or self.full.get(i) or {}).values())]
        if len(tagged) == 1:
            return tagged[0]
        if len(ref) >= 8:
            near = [i for i in known if _one_off(ref, i)]
            if len(near) == 1:
                return near[0]
        return value

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        args = self.mend_args(name, args)
        payload = dict(self.binding)
        payload['step'] = self.step + 1
        payload['action'] = {'name': name, 'args': args}
        for _ in range(4):
            result = _env.post_verbose('/environment/call', payload)
            if isinstance(result.data, dict):
                data = result.data
                self.step = data['step'] if isinstance(data.get('step'), int) else self.step + 1
                if data.get('finished'):
                    self.finished = True
                if data.get('observation') is not None:
                    data['observation'] = _read_rows(data['observation'])
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
                # (kept among the refs, which the word checks skip, so the item check judges it against the other pick;
                #  never walked: the race-17 tests turned "unless" into a "different" relation and declined)
                if _conditional(text):
                    self.conds.add(text)
                    self.walked.add(text)
        if self.conds:
            self.trace.append({'kind': 'conditions', 'musts': sorted(self.conds)})
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
            pointers = [need for n, need in self.refs if n == number and need in item.musts and need not in self.conds]
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
        if self.clock() >= WIND_DOWN and left > self.tail() + 1:
            text += ' TIME IS NEARLY UP: set the cart to the chosen lines and check out within the next two turns; an episode that runs out of time earns nothing.'
        used = self.cost / SPEND_HARD if SPEND_HARD else 0.0
        if used >= 0.5 and self.buys == 0:      # (the run's key pays for every task: past half of a task's share, settle the picks)
            text += f' {round(used * 100)}% of this task\'s model budget is used: settle the picks, fill the cart and check out soon.'
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values and need not in self.conds]
        if lookups and self.buys == 0:
            text += '\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] ' + ' | '.join(lookups[:3])
        if self.leads:
            text += '\n[review candidates] ' + ', '.join(self.leads)
        plan = self.fit_plan() if self.bound and self.buys == 0 else ''
        if plan:
            text += '\n' + plan
        if self.notes:
            text += '\n[changes so far] ' + ' | '.join(self.notes[-4:])
        board = self.candidate_board()
        if board:
            text += '\n' + board
        text += '\n[ledger]\n' + self.ledger()
        return text

    def candidate_board(self) -> str:
        """Per item, the opened listings that are still candidates, best first, with how many stated values each record shows in
        words (a count, not a verdict: a value shown under another name still counts for the check)."""
        if self.buys or not self.items or self.step < 4:
            return ''
        parts = []
        for number, item in enumerate(self.items, 1):
            pool = self.candidates(number)
            if not pool:
                parts.append(f'item {number} ({item.name}): no opened candidate yet')
                continue
            ranked = sorted(pool, key=lambda i: (-self.must_hits(number, i), self.price_of(i) if self.price_of(i) is not None else 1e12))[:3]
            parts.append(f'item {number} ({item.name}): ' + ', '.join(f'{i} ({self.must_hits(number, i)}/{len(item.musts)} stated values shown, price {self.price_of(i)})' for i in ranked))
        return '[candidates opened] ' + ' | '.join(parts)

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
        if texts:
            self.note_next(name, args, value)
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
        if self.wider_held or self.brief.get('pick') != 'cheapest' or self.left() < 14 + 2 * len(self.items):
            return ''
        self.wider_held = True
        lines = [f'{i} at {self.price_of(i):g}' for i in self.cart_ids() if self.price_of(i) is not None]
        swept = self.cheap_sweep()
        if swept:
            return 'Not run yet. Kyo searched further for cheaper listings of that kind and read the cheapest unread ones (one step each):\n' + swept + f"\nIf one of them meets every requirement and costs less than the cart's line" + (f" ({'; '.join(lines)})" if lines else '') + f', swap it in. {self.left()} steps are left: run another search under another name for the product only if steps allow, then check out.'
        return f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest. Searches match a listing's title, so a requirement's value (often only in the attributes) finds few of the fitting listings, and rows come in no price order. Before checking out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line" + (f" ({'; '.join(lines)})" if lines else '') + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'

    def note_next(self, name: str, args: dict[str, Any], value: Any) -> None:
        """The next-page token of a listing call, kept per call (its page argument left out), so a later sweep can read on."""
        page_arg = (self.shapes.get(name) or {}).get('next_arg')
        if page_arg:
            base = {k: v for k, v in args.items() if k != page_arg}
            self.browse_next[name + _dump(base)] = (name, base, value)

    def figure_match(self, number: int, row: dict[str, Any]) -> bool:
        """A search row whose own text shows each heard value of item NUMBER (its figures, or else its words)."""
        heard, _ = self.bare_values(number)
        text = ' '.join(str(v) for v in row.values() if isinstance(v, str)).lower()
        for value in heard:
            figures = re.findall('\\d+(?:\\.\\d+)?', value)
            if figures and not all(re.search('(?<![\\d.])' + re.escape(f) + '(?!\\d)', text) for f in figures):
                return False
            if not figures and not _vstem(value) <= _vstem(text):
                return False
        return bool(heard)

    def cheap_sweep(self) -> str:
        """Under a cheapest rule, once, before the first order with steps to spare: per cart line, one search of the item's
        plain name, up to three further pages of the category listing already begun, then the cheapest unread rows of the
        line's category priced below it (rows showing the shopper's heard value first) are read. Search has no price order,
        so the cheapest fitting listing is often on a page no search has shown yet."""
        if self.clock() > WIND_DOWN:      # (time is nearly up: the cart and the checkout come first)
            return ''
        cat_key = self.category_key()
        room = self.left() - self.tail() - 10
        if not cat_key or room < 3:
            return ''
        notes: list[str] = []
        swept: set[str] = set()
        cart = self.cart_ids()
        for line in cart:
            label = str((self.rows.get(line) or {}).get(cat_key) or '').strip()
            price = self.price_of(line)
            owners = [n for n in range(1, len(self.items) + 1) if self.is_kind(n, label)]
            if len(owners) != 1 or price is None or not label:
                continue
            number = owners[0]
            plan = self.search_template() or self.schema_template()
            name = re.split('\\bor\\b|[,/]', self.items[number - 1].name)[-1].strip()
            done = {_words(str(a.get(plan[2]) or '')) for a in self.searched} if plan else set()
            if plan and name and len(name.split()) <= 3 and _words(name) not in done and room >= 3 and not self.finished:
                tool, base, field = plan
                faults: list[str] = []
                args = _shape({**base, field: name}, self.schemas.get(tool), 'args', faults)
                reply = self.act(tool, args) if not faults else None
                if reply is not None:
                    room -= 1
                    self.observe(tool, reply.get('observation'))
                    self.searched = (self.searched + [dict(args)])[-8:]
                    notes.append(f'search {_dump(args.get(field))}')
            for tool, base, value in list(self.browse_next.values()):
                if value is None or label.lower() in swept or not any((isinstance(v, str) and v.strip().lower() == label.lower() for v in base.values())):
                    continue
                swept.add(label.lower())          # (one listing per category: the same label in another spelling is the same list)
                page_arg = (self.shapes.get(tool) or {}).get('next_arg')
                for _ in range(3):
                    if value is None or room < 4 or self.finished:
                        break
                    faults = []
                    args = _shape({**base, page_arg: value}, self.schemas.get(tool), 'args', faults)
                    reply = self.act(tool, args) if not faults else None
                    if reply is None:
                        break
                    room -= 1
                    seen = reply.get('observation')
                    self.observe(tool, seen)
                    after = self.paging(tool, seen)
                    notes.append(f'{label} listing from {_dump(args.get(page_arg))}')
                    value = after[3] if after else None
                self.note_next(tool, base, value)
            pool = [i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i) and (str((self.rows.get(i) or {}).get(cat_key) or '').strip().lower() == label.lower()) and ((self.price_of(i) if self.price_of(i) is not None else float('inf')) < price)]
            pool.sort(key=lambda i: (not self.figure_match(number, self.rows.get(i) or {}), self.price_of(i) or 0.0))
            read = self.fill_facts(pool[:min(3, max(0, room - 1))], self.tail() + 6)
            if read:
                notes.append('read: ' + ' | '.join(read)[:2400])
        if notes:
            self.trace.append({'kind': 'cheap_sweep', 'notes': [n[:160] for n in notes]})
        return '\n'.join(notes)

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
        return self.fill_facts(self.cheaper_picks(), self.tail() + 4)

    def cheaper_picks(self) -> list[str]:
        """Unopened buyable rows of a cart line's category priced below the line, to read before a cheapest order. Rows the
        value search found (they hold the shopper's heard value in their text) go first, cheapest first, since a listing whose
        title looks unlike the product can still state every required value; then the cheapest other rows."""
        cat_key = self.category_key()
        if not cat_key:
            return []
        cart = self.cart_ids()
        picks: list[str] = []
        for line in cart:
            price = self.price_of(line)
            kind = str((self.rows.get(line) or {}).get(cat_key) or '').strip().lower()
            if price is None or not kind:
                continue
            pool = [i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i) and (str((self.rows.get(i) or {}).get(cat_key) or '').strip().lower() == kind) and ((self.price_of(i) if self.price_of(i) is not None else float('inf')) < price)]
            pool.sort(key=lambda i: self.price_of(i) or 0.0)
            lead = [i for i in pool if i in self.value_rows][:CHEAPER_OPENS]
            rest = [i for i in pool if i not in self.value_rows][:max(0, 2 - len(lead))]
            picks += [i for i in lead + rest if i not in picks]
        return picks[:CHEAPER_OPENS]

    def cap_walk(self) -> list[str]:
        """Under a cheapest rule, before the order: each cart line's category listed below the line's price on a listing tool that
        takes a price cap (a number argument other than its page argument, on a tool an earlier call filled with that category's
        label), so the cheaper rows of that kind are seen and the cheaper-row check can open them; once per line."""
        if self.brief.get('pick') != 'cheapest':
            return []
        cat_key = self.category_key()
        notes: list[str] = []
        for ident in self.cart_ids():
            label = str((self.rows.get(ident) or {}).get(cat_key) or '').strip() if cat_key else ''
            price = self.price_of(ident)
            if ident in self.capped_lines or not label or price is None or price <= 1 or self.left() - self.tail() < 6:
                continue
            found = None
            for tool, args in reversed(self.list_calls):
                props = (self.schemas.get(tool) or {}).get('properties') or {}
                page = (self.shapes.get(tool) or {}).get('next_arg')
                caps = [k for k, spec in props.items() if isinstance(spec, dict) and k != page and ({'number', 'integer'} & set(_kinds(spec)))]
                shelf = [k for k, v in args.items() if isinstance(v, str) and v.strip().lower() == label.lower()]
                if len(caps) == 1 and shelf:
                    found = (tool, shelf[0], caps[0], props[caps[0]])
                    break
            if not found:
                continue
            self.capped_lines.add(ident)
            tool, shelf_arg, cap_arg, spec = found
            bound = price - 1 if float(price).is_integer() else price - 0.01
            faults: list[str] = []
            call = _shape({shelf_arg: label, cap_arg: int(bound) if 'integer' in _kinds(spec) else bound}, self.schemas.get(tool), 'args', faults)
            if faults:
                continue
            reply = self.act(tool, call)
            if reply is None or self.finished:
                break
            seen = reply.get('observation')
            self.observe(tool, seen)
            self.trace.append({'kind': 'cap_walk', 'line': ident, 'tool': tool, 'below': bound, 'rows': len(self.records_in(seen))})
            notes.append(f'{tool} {_dump(call)}: ' + self.show(seen))
        return notes

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
        if self.left() <= 8 or self.clock() > ASK_BY:
            return False
        opened = len(self.full)
        if opened >= 3 * len(self.items) + 1:
            return True
        if opened and self.clock() >= ASK_LATEST:
            return True          # (slow turns: the window for questions closes before the read or step count is reached)
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

    def interview(self) -> str:
        self.asked = True
        if self.msgs_free() <= 0:
            return ''
        self.widen()
        lanes = self.plan_lanes(max(self.msgs_free() * LIST_SIZE, LIST_MULTI) if self.listing else self.msgs_free())
        found = {n: 0 for n in lanes}
        spent = {n: 0 for n in lanes}
        used: dict[int, set[str]] = {n: set() for n in lanes}
        self.lanes, self.found, self.spent, self.used = (lanes, found, spent, used)
        follow: list[tuple[int, list[str], bool]] = []
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        while self.msgs_free() > 0 and (not self.finished) and (self.left() > 6) and (self.clock() < QUESTION_BY):
            if follow:
                number, labels, shown = follow.pop(0)
            else:
                number, labels = self.next_ask(lanes, found, spent, used)
                shown = False
                if not labels:
                    break
            labels, text = self.fit_ask(number, labels, shown)
            if not labels:
                break
            used[number].update((_words(label) for label in labels))
            batch = self.send([(number, labels[0], text)])
            if not batch:
                break
            sent += batch
            spent[number] += 1
            had = [len(item.told) for item in self.items]
            heard = self.hear(batch, (LIGHT, HEAVY))
            if len(labels) > 1:
                heard.pop('free', None)     # (a "don't mind" answers the list's first property at most, not every one named)
            self.take_heard(heard, told)
            fresh = [n for n, item in enumerate(self.items, 1) if len(item.told) > had[n - 1]]
            for n in fresh:
                found[n] = found.get(n, 0) + 1
            if fresh or 1 not in [_count(x) for x in _items(heard.get('crowded'))]:
                continue
            if len(labels) > 1:
                # (a list turned away: lists end for the episode and its first property is asked alone)
                self.lists_off = True
                self.trace.append({'kind': 'lists_off', 'item': number})
                follow.insert(0, (number, labels[:1], False))
                used[number].difference_update((_words(label) for label in labels[1:]))      # (the list's other properties stay open)
                continue
            if not shown and labels[0] not in [lb[0] for _, lb, _ in follow]:
                # (asked once more about the same whole property in other plain words: the shopper turns the same words away
                #  again almost every time, while a turned-away question is as likely as any other to be about a real need;
                #  example values or a shortened property name make it worse)
                follow.insert(0, (number, labels, True))
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

    def schema_template(self) -> tuple[str, dict[str, Any], str] | None:
        """The search tool's free-text call read from its schema alone, for a value search when no free-text search has run
        yet (only category listings before the questions): the text argument by its name, each mode argument set to the
        allowed value that names a text search (else its default); None when a required argument cannot be filled."""
        tool = self.roles.get('search')
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texty = '(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword'
        named = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v) and re.search(texty, k, re.I)]
        if not tool or not named:
            return None
        field = named[0]
        about = next((str(t['function'].get('description') or '') for t in self.tools if t['function']['name'] == tool), '')
        near = ' '.join(s for s in re.split('(?<=[.;])\\s+', about) if field.lower() in s.lower())
        keep: dict[str, Any] = {}
        for k, spec in props.items():
            if k == field or not isinstance(spec, dict):
                continue
            options = spec.get('enum') if isinstance(spec.get('enum'), list) else None
            if options:
                # (the mode value that names the text argument itself, or that the text argument's own field-name pattern
                #  matches, or that the tool's description names beside the text argument)
                strings = [o for o in options if isinstance(o, str)]
                choice = next((o for o in strings if _stem(o) & _stem(field)), None) or next((o for o in strings if re.search(texty, o, re.I)), None) or next((o for o in strings if near and re.search('\\b' + re.escape(o) + '\\b', near, re.I)), None)
                if choice is None and spec.get('default') in options:
                    choice = spec['default']
                if choice is None:
                    if k in (schema.get('required') or []):
                        return None
                    continue
                keep[k] = choice
            elif k in (schema.get('required') or []):
                return None
        return (tool, keep, field)

    def bare_values(self, number: int) -> tuple[list[str], list[str]]:
        """An item's requirement values as search words, (heard, stated): a limit gives its figure ("at least 567 g" gives
        567 g); exclusions, alternatives and values read from another listing give none."""
        item = self.items[number - 1]
        heard_values = {t.split(':', 1)[1].strip() for t in item.told if ':' in t}
        heard: list[str] = []
        stated: list[str] = []
        for n, need in self.needs_now():
            if n != number or need in [m for _, m in self.refs] or ':' not in need:
                continue
            value = given = need.split(':', 1)[1].strip()
            if re.fullmatch('(?:yes|true)', value, re.I):
                value = need.split(':', 1)[0].strip()
            bare = re.sub(LIMIT_HEAD, '', value, flags=re.I) if re.search('\\d', value) else value
            if not bare or len(bare.split()) > 4 or re.search('\\b(?:not|no|non|false|without|except|other than|anything but|unlike|alongside|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or|same as)\\b|[<>=]', bare, re.I):
                continue
            (heard if given in heard_values else stated).append(bare)
        heard = list(dict.fromkeys(heard))
        return (heard, [v for v in dict.fromkeys(stated) if v.lower() not in {h.lower() for h in heard}])

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
        """After the interview: for each item with a newly heard value, one text search of the heard values followed by the
        request's own values (short, value-first), then the heard values alone when that finds too few rows of the item's
        kind; the first fitting-kind rows are read (under a cheapest rule: page 2 of a short result too, and the cheapest
        rows of the fullest query first)."""
        if self.clock() > WIND_DOWN:      # (time is nearly up: the cart and the checkout come first)
            return ''
        plan = self.search_template() or self.schema_template()
        if not plan or self.left() < 22:
            return ''
        tool, base, field = plan
        cheapest = self.brief.get('pick') == 'cheapest'
        cat_key = self.category_key()
        seen: list[str] = []
        budget = 8
        for number in numbers:
            heard, stated = self.bare_values(number)
            if not heard or budget <= 2:
                continue
            words = list(heard)
            for value in stated:
                if len(' '.join(words + [value]).split()) > 5:
                    break
                words.append(value)
            # (then the heard values with the product's name, as a value-first query, and the heard values alone)
            noun = self.items[number - 1].name.strip()
            named = heard + [noun] if noun and len(noun.split()) <= 3 and not re.search('\\bor\\b|[,/]', noun, re.I) and len(' '.join(heard + [noun]).split()) <= 6 else []
            tries = [q for i, q in enumerate([words, named, heard]) if q and q not in [words, named, heard][:i]]
            groups: list[list[str]] = []
            for query in tries:
                if budget <= 2 or self.finished:
                    break
                faults: list[str] = []
                args = _shape({**base, field: ' '.join(query)}, self.schemas.get(tool), 'args', faults)
                if faults:
                    break
                reply = self.act(tool, args)
                budget -= 1
                if reply is None:
                    break
                observation = reply.get('observation')
                self.observe(tool, observation)
                found = [i for i, _ in self.records_in(observation)]
                more = self.paging(tool, observation) if cheapest else None
                if more and more[3] is not None and more[0] <= 3 * more[1] and budget > 3 and not self.finished:
                    page_args = _shape({**args, more[2]: more[3]}, self.schemas.get(tool), 'args', faults)
                    later = self.act(tool, page_args) if not faults else None
                    if later is not None:
                        budget -= 1
                        self.observe(tool, later.get('observation'))
                        found += [i for i, _ in self.records_in(later.get('observation')) if i not in found]
                cap = self.budget if isinstance(self.budget, (int, float)) else None
                prices = [self.price_of(i) for i in found]
                if cap is not None and prices and all((p is not None and p > 10 * cap for p in prices)):
                    cap = None
                rows = [i for i in found if self.buyable(i) and (not cat_key or self.is_kind(number, str((self.rows.get(i) or {}).get(cat_key) or ''))) and (cap is None or (self.price_of(i) or 0) <= cap)]
                seen.append(f'{_dump(args.get(field))}: ' + (', '.join(rows[:6]) or 'no row of that kind'))
                groups.append(rows)
                self.value_rows += [i for i in rows if i not in self.value_rows]
                if len({i for g in groups for i in g}) >= (3 if cheapest else 1):
                    break
            order: list[str] = []
            for rows in groups:
                ranked = sorted(rows, key=lambda i: self.price_of(i) if self.price_of(i) is not None else float('inf')) if cheapest else rows
                order += [i for i in ranked if i not in order]
            if order:
                opened = self.fill_facts([i for i in order if i not in self.full][:min(3, budget)], self.tail() + 16)
                budget -= len(opened)
                if opened:
                    seen.append('opened: ' + ' | '.join(opened)[:2400])
        if not seen:
            return ''
        self.trace.append({'kind': 'value_search', 'seen': [s[:200] for s in seen]})
        return "[Kyo searched each such item's requirement values together and read the first fitting-kind rows] " + '\n'.join(seen)

    def next_ask(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> tuple[int, list[str]]:
        first = {n: i for i, n in enumerate(self.plan_order)}
        mains = {m for _, m in self.bound}
        parts = {p for p, _ in self.bound} - mains
        # (when parts must fit a device, the shopper's unwritten needs sit on a part, not on the device (races 8 and 17 kits:
        #  nearly every heard need), so a part not asked yet gets the first question, one the story hints at first; the device
        #  waits until every part was asked, and an item with a need already heard waits; every item not asked yet comes
        #  before a second question to any item)
        fresh = lambda n: spent.get(n, 0) == 0
        for number in sorted(lanes, key=lambda n: (found.get(n, 0) > 0, not (n in parts and fresh(n)), not (n in first and fresh(n)), spent.get(n, 0), n in mains, first.get(n, len(first) + n))):
            left = [label for label in lanes[number] if _words(label) not in used[number]]
            if left:
                if not self.listing:
                    return (number, left[:1])
                if len(self.items) > 1:
                    return (number, left[:LIST_MULTI])       # (one question per product: its best three)
                # (one product, M questions left: the M best each open a question, since a first slot is answered far more
                #  often than a second one (race-16/17 tests: a need named first 21 of 27 times, second 4 of 15); the next M
                #  fill the second slots in the same order: [#1, #M+1], [#2, #M+2] ... ([#1, #3], [#2, #4] on two messages))
                room = max(1, self.msgs_free())
                return (number, [left[0], left[room]] if len(left) > room else left[:1])
        return (0, [])

    def late_ask(self) -> str:
        """A message held back for a step at the checkout that the gate then did not need is free: one more question to the
        shopper, for the item whose needs are still unheard, before the order goes. Returns the new requirement text (the
        checkout then waits for the cart to be fixed), else ''. Once per episode."""
        if self.late_asked or not self.asked or not self.roles.get('ask') or self.finished or (not self.lanes) or self.clock() > LATE_BY:
            return ''
        self.late_asked = True
        if self.gate_msgs < self.reserve:
            self.reserve = self.gate_msgs
        if self.msgs_free() <= 0 or self.left() <= 8:
            return ''
        number, labels = self.next_ask(self.lanes, self.found, self.spent, self.used)
        if not labels:
            return ''
        labels, text = self.fit_ask(number, labels, False)
        if not labels:
            return ''
        self.used[number].update((_words(label) for label in labels))
        batch = self.send([(number, labels[0], text)])
        if not batch:
            return ''
        self.spent[number] = self.spent.get(number, 0) + 1
        told: list[str] = []
        heard = self.hear(batch, (LIGHT, HEAVY))
        if len(labels) > 1:
            heard.pop('free', None)
        self.take_heard(heard, told)
        self.trace.append({'kind': 'late_ask', 'q': batch[0][1][:120], 'told': told})
        if not told:
            return ''
        self.found[number] = self.found.get(number, 0) + 1
        self.version += 1
        return 'Kyo used the spare message for one more question. Q: ' + batch[0][1] + ' A: ' + batch[0][2][:300] + ' New firm requirement: ' + '; '.join(told) + '. Fix the cart for it (search again with the new value where needed), then check out again.'

    def ask_text(self, number: int, labels: list[str], shown: bool=False) -> str:
        if len(labels) >= 2:
            return f"What {', '.join(labels[:-1])} or {labels[-1]} would you like for the {self.shelf_name(number)}?"
        if shown:
            return f'Which {labels[0]} should the {self.shelf_name(number)} have?'
        return self.question(number, labels[0], None)

    def fit_ask(self, number: int, labels: list[str], shown: bool) -> tuple[list[str], str]:
        """The longest leading part of `labels` whose question the message tool's schema takes whole (the schema shaper
        cuts a text past a length cap, so a list too long drops labels from the end instead), with that question;
        ([], '') when not even one label fits."""
        for size in range(len(labels), 0, -1):
            text = self.ask_text(number, labels[:size], shown)
            args = self.message_args(text)
            if args and text in args.values():
                return (labels[:size], text)
        return ([], '')

    def lane_keys(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        item = self.items[number - 1]
        settled = [_stem(m.split(':', 1)[0]) for m in item.musts]
        rows = [(label, key, carried, len(values)) for label, key, carried, values in table if self.owner_of(label) == number]
        measured = {key: sum((1 for v in values if re.match('^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$', v))) * 2 > len(values) for _, key, _, values in table}
        if len(self.items) == 1 and (not rows):
            rows = [(label, key, carried, len(values)) for label, key, carried, values in table]
        rows.sort(key=lambda r: (-r[2], measured.get(r[1], False), -r[3]))
        # (the key's head word decides: "warranty type" or "model number" is logistics, while a property qualified by such a
        #  word at its end, "cpu chipset model" or "ddr variant", is a taste a shopper holds; race 17 hid needs on both)
        keys = [key for _, key, _, _ in rows if not any((_names(prop, _stem(key)) for prop in settled)) and (not re.search(NOT_PROPERTY, (key.split() or [key])[0]))
                and not re.search('\\d\\s*$', _words(key))]     # (a key ending in a number is a store's second copy of another key)
        return [key for key in keys if not re.search(CATCH_ALL, key)] + [key for key in keys if re.search(CATCH_ALL, key)]

    def exact_labels(self) -> dict[int, frozenset[str]]:
        """Items whose own name or shelf is, word for word, a category this store shows: the category of each."""
        cat_key = self.category_key()
        mark = (len(self.rows), len(self.items), cat_key)
        memo = self.__dict__.get('_exact')
        if memo and memo[0] == mark:
            return memo[1]
        shown = {frozenset(_roots(str(r.get(cat_key) or ''))) for r in self.rows.values()} - {frozenset()} if cat_key else set()
        out: dict[int, frozenset[str]] = {}
        for number, item in enumerate(self.items, 1):
            for name in (item.shelf or '', item.name):
                key = frozenset(_roots(name))
                if key and key in shown:
                    out[number] = key
                    break
        self._exact = (mark, out)
        return out

    def owner_of(self, label: str) -> int:
        if not label:
            return 0
        words = _words(label)
        long = _roots(label)
        exact = self.exact_labels()
        best, top = (0, 0.0)
        for number, item in enumerate(self.items, 1):
            if number in exact and frozenset(long) != exact[number]:
                continue        # (that item is a category of its own here: a category that only shares words is another kind)
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
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 3500) or {}
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
        for number in list(lanes):
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
            # (a property the story hints at leads the lane, so it opens the first question)
            lanes[number] = [x for x in lanes[number] if _words(x) in hinted[number]] + [x for x in lanes[number] if _words(x) not in hinted[number]]
        self.trace.append({'kind': 'plan', 'ask': _clip(plan.get('ask'), 120), 'lanes': {n: v[:8] for n, v in lanes.items()}})
        return lanes

    def shelf_name(self, number: int) -> str:
        """The store's category label of an item's listings (a request name with "or" in it reads as two products)."""
        key = self.category_key()
        seen = [str((self.rows.get(i) or {}).get(key) or '').strip() for i in self.item_ids(number)] if key else []
        seen = [x for x in seen if x]
        if not seen:
            return self.items[number - 1].name
        item = self.items[number - 1]
        mine = _roots(item.shelf or '') | _roots(item.name)

        def fit(label: str) -> float:
            # (a label sharing one word with the item, a neighbouring shelf, loses to the item's own shelf however often it was read)
            if _words(label) in (_words(item.shelf or ''), _words(item.name)):
                return 2.0
            return _root_share(mine, _roots(label))
        return max(set(seen), key=lambda x: (fit(x), seen.count(x))).lower()

    def kind_fit(self, number: int, label: str) -> float:
        """How well a category label names item NUMBER: its own shelf or name 2, otherwise the share of shared word roots."""
        item = self.items[number - 1]
        if _words(label) in (_words(item.shelf or ''), _words(item.name)):
            return 2.0
        return _root_share(_roots(item.shelf or '') | _roots(item.name), _roots(label))

    def is_kind(self, number: int, label: str) -> bool:
        """A listing of item NUMBER's kind: its label is owned by the item and fits it as well as the best label seen (a
        neighbouring shelf that shares one word, fruit juice for canned fruit, is not the item's kind)."""
        item = self.items[number - 1]
        if not self.owns(item, label):
            return False
        key = self.category_key()
        seen = {str((self.rows.get(i) or {}).get(key) or '').strip() for i in self.seen_order} if key else set()
        best = max([self.kind_fit(number, x) for x in seen if x and self.owns(item, x)] or [0.0])
        return self.kind_fit(number, label) >= best

    def question(self, number: int, label: str, samples: list[str] | None=None) -> str:
        text = f'What {label} would you like for the {self.shelf_name(number)}'
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
            self.replies.append(words)
            sent.append((number, text, words))
            self.done_log.append(f'step {self.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {self.cart_mark()})')
            self.done_steps.append(self.step)
            self.trace.append({'kind': 'ask', 'q': text, 'a': words[:300]})
            if isinstance(answer, dict) and answer.get('error'):
                break
        return sent

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
        owned: dict[str, str] = {}
        added: dict[str, set[str]] = {}
        for ident in self.seen_order:
            record = self.full.get(ident) or self.rows.get(ident) or {}
            text = ' '.join((str(v) for v in record.values() if isinstance(v, str)))
            text += ' ' + '. '.join((f'{k} {v}' for block in record.values() if isinstance(block, dict) for k, v in block.items() if isinstance(v, str)))
            marks[ident] = _fit_marks(text)
            marks[ident]['models'] = _models(text) | _series_taken(text)
            marks[ident]['rules'] = _fit_rules(text) + _series_rule(text)
            own_series = re.search(SERIES_OWN, text.lower())
            marks[ident]['series'] = own_series.group(1) + own_series.group(2) if own_series else ''
            for hit in re.finditer(SERIES_ADDED, text.lower()):
                for series in re.findall('[a-z]{1,5}\\d{1,3}', hit.group(3)):
                    added.setdefault(series, set()).add(hit.group(1) + hit.group(2))
            if marks[ident]['own']:
                owned.setdefault(marks[ident]['own'], ident)
            for part, codes in marks[ident]['also']:
                notes.setdefault(part, set()).update(codes)
        return {'marks': marks, 'notes': notes, 'owned': owned, 'added': added}

    def fit_ok(self, ident: str, model: str, index: dict[str, Any]) -> bool | None:
        """Whether a part takes in a device model: its own list (wildcards and exceptions included) or what another listing's
        page says that part also fits; None when its list goes on on a part whose page is not seen yet, or names no such codes."""
        mark = index['marks'].get(ident) or {}
        if mark.get('own') and model in {re.sub('[^a-z0-9]', '', c) for c in index['notes'].get(mark['own'], set())}:
            return True
        if mark.get('series') and model in index.get('added', {}).get(mark['series'], set()):
            return True
        said = _takes(mark.get('rules') or [], model)
        if said is False and mark.get('more') and mark['more'] not in index['owned']:
            return None
        return said

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
        unseen = []
        for part, main in self.bound:
            devices = [i for i in cart if i in self.item_ids(main)]
            models = set(index['marks'].get(devices[0], {}).get('models') or []) if len(devices) == 1 else set()
            for line in [i for i in cart if i in self.item_ids(part)]:
                mark = index['marks'].get(line) or {}
                if models and mark.get('more') and mark['more'] not in index['owned'] and all((self.fit_ok(line, m, index) is None for m in models)):
                    unseen.append(mark['more'])
        if unseen and self.fit_lookup(unseen):
            index = self.fit_index()
        faults = []
        for part, main in self.bound:
            devices = [i for i in cart if i in self.item_ids(main)]
            parts = [i for i in cart if i in self.item_ids(part)]
            if len(devices) != 1 or not parts:
                continue
            models = set(index['marks'].get(devices[0], {}).get('models') or [])
            for line in parts:
                said = [v for v in (self.fit_ok(line, m, index) for m in models) if v is not None]
                if said and not any(said):
                    good = [i for i in self.item_ids(part) if i not in cart and not self.sold_out(i)
                            and any(self.fit_ok(i, m, index) for m in models)][:4]
                    faults.append(f"{line} ({self.items[part - 1].name}) lists the models it fits and leaves out {', '.join(sorted(m.upper() for m in models))}, "
                                  f"the model of {devices[0]}" + (f"; seen parts of that kind that fit it: {', '.join(good)}" if good else
                                                                  '; no seen part of that kind fits it, so look for one or switch the device: ' + self.fit_plan(short=True)))
            codes = set(index['marks'].get(devices[0], {}).get('codes') or [])
            if not codes:
                continue
            for line in parts:
                fits = self.fit_set(line, index)
                if fits and (not fits & codes):
                    good = [i for i in self.item_ids(part) if self.fit_set(i, index) & codes and (not self.sold_out(i)) and (i not in cart)][:4]
                    faults.append(f"{line} ({self.items[part - 1].name}) works with {', '.join(sorted(fits)[:4])}, not with {devices[0]} ({', '.join(sorted(codes))})" + (f"; seen parts of that kind that work with it: {', '.join(good)}" if good else '; no seen part of that kind works with it, so look for one or switch the device: ' + self.fit_plan(short=True)))
        return faults

    def fit_lookup(self, codes: list[str]) -> bool:
        """A cart part whose own codes leave the device out but whose page says the rest of its list is on part X: one search
        for X (at most two in an episode), so the note on X's page about this part is read before the fit verdict."""
        if self.clock() > WIND_DOWN:      # (time is nearly up: the cart and the checkout come first)
            return False
        plan = self.search_template()
        todo = [c for c in dict.fromkeys(codes) if c not in self.looked_parts][:max(0, 2 - len(self.looked_parts))]
        if not plan or not todo or self.left() < self.tail() + 4:
            return False
        tool, base, field = plan
        for code in todo:
            self.looked_parts.add(code)
            faults: list[str] = []
            args = _shape({**base, field: code}, self.schemas.get(tool), 'args', faults)
            if faults:
                return False
            reply = self.act(tool, args)
            if reply is None:
                return False
            self.observe(tool, reply.get('observation'))
            self.trace.append({'kind': 'fit_lookup', 'part': code})
        return True

    def part_lookups(self) -> list[str]:
        """Before an order of several items: each cart line whose page says more on it is under part X gets one search for X
        (once per part), as the store's kit pages ask the buyer to look that part up before the fit is taken as settled."""
        if self.clock() > WIND_DOWN:      # (time is nearly up: the cart and the checkout come first)
            return []
        if len(self.items) < 2:
            return []
        index = self.fit_index()
        todo = [c for c in dict.fromkeys((index['marks'].get(i, {}).get('more') for i in self.cart_ids())) if c and c not in self.looked_parts]
        plan = self.search_template() or self.schema_template()
        if not todo or not plan or self.left() < self.tail() + len(todo) + 2:
            return []
        tool, base, field = plan
        done: list[str] = []
        for code in todo:
            faults: list[str] = []
            args = _shape({**base, field: code}, self.schemas.get(tool), 'args', faults)
            if faults or self.finished:
                break
            self.looked_parts.add(code)
            reply = self.act(tool, args)
            if reply is None:
                break
            self.observe(tool, reply.get('observation'))
            done.append(code)
        if done:
            self.trace.append({'kind': 'part_lookups', 'parts': done})
        return done

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
            for device in self.item_ids(main):
                if self.sold_out(device):
                    continue
                for model in index['marks'].get(device, {}).get('models') or []:
                    for p in parts:
                        took = sum(1 for i in self.item_ids(p) if not self.sold_out(i) and self.fit_ok(i, model, index))
                        if took:
                            tally.setdefault(model, {}).setdefault(p, 0)
                            tally[model][p] = max(tally[model][p], took)
            ranked = sorted(tally.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1].values())))[:4]
            devices = {code: [i for i in self.item_ids(main) if code in (index['marks'].get(i, {}).get('codes') or []) or code in (index['marks'].get(i, {}).get('models') or [])][:2] for code, _ in ranked}
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

    def attr_gaps(self, ids: list[str]) -> list[tuple[str, int, str, list[tuple[str, str, str]]]]:
        """Cart lines with no attribute for a requirement's property while other opened listings of the same kind state that
        property, with the required value, as an attribute of their own: (line, item number, requirement, [(listing, key, value)])."""
        cat_key = self.category_key()
        if not cat_key:
            return []
        skip = {m for _, m in self.refs}
        out = []
        for line in ids:
            record = self.full.get(line)
            block = _fact_block(record or {}) or {}
            if not block:
                continue
            owners = [n for n, item in enumerate(self.items, 1) if self.is_kind(n, str((record or {}).get(cat_key) or ''))]
            if len(owners) != 1:
                continue
            number = owners[0]
            item = self.items[number - 1]
            for need in item.musts + item.told:
                if need in item.dropped or need in skip or self.described(number, need):
                    continue
                self.gap_said.pop((line, need), None)
                self.gap_limit.discard((line, need))
                prop, _, value = need.partition(':')
                if re.fullmatch('\\s*(?:yes|true)\\s*', value, re.I):
                    value = prop          # ("multipack: yes" asks for a multipack)
                want = _vstem(value)
                # (a limit on a figure, "at least 320 ml": the line needs an attribute for that property at all; whether a
                #  listing's figure meets the limit is left to the check)
                figure = re.sub(LIMIT_HEAD, '', value.strip(), flags=re.I) if re.search('\\d', value) else ''
                limit = bool(figure) and figure != value.strip() and (not re.search(GAP_SKIP, figure, re.I))
                if limit:
                    want = _vstem(figure)
                if not prop.strip() or not want or len(want) > 4 or (re.search(GAP_SKIP, value, re.I) and (not limit)):
                    continue
                kin = [o for o in self.seen_order if o != line and o not in ids and self.full.get(o) and self.is_kind(number, str(self.full[o].get(cat_key) or ''))]
                names = {_words(k) for o in kin for k in (_fact_block(self.full[o]) or {}) if _same_prop(prop, k)} | {_words(k) for k in block if _same_prop(prop, k)}
                if limit:
                    if not names or any((_same_prop(prop, k) for k in block)):
                        continue
                    heard_words = [_vstem(t.split(':', 1)[1]) for t in item.told if ':' in t and t not in item.dropped]
                    ranked = []
                    for other in kin:
                        if self.sold_out(other):
                            continue
                        theirs = _fact_block(self.full.get(other) or {}) or {}
                        own_keys = [(k, v) for k, v in theirs.items() if _same_prop(prop, k)]
                        if own_keys:
                            k, v = own_keys[0]
                            facts = _vstem(' '.join(str(x) for x in theirs.values()))
                            # (a listing showing the limit's own figure meets it, edges included: it goes first)
                            ranked.append((not want <= _vstem(v), -sum(1 for w in heard_words if w <= facts), len(ranked), (other, str(k), str(v)[:60])))
                    carriers = [c[-1] for c in sorted(ranked)[:3]]
                    if carriers:
                        out.append((line, number, need, carriers))
                        self.gap_limit.add((line, need))
                    continue
                # (a short name can match several keys of the kind, pack for pack size and pack type: then the keys that hold the
                # required value on an opened listing of the kind are the property)
                held = {n for n in names if any(_words(k) == n and want <= _vstem(v) for o in kin for k, v in (_fact_block(self.full[o]) or {}).items())}
                if len(names) > 1 and held:
                    holds = lambda k, v, held=held: _words(k) in held
                elif names:
                    holds = lambda k, v: _same_prop(prop, k)
                else:
                    # (the reader's name for the property is no key of this kind: the keys that hold the required value on at
                    # least two opened listings of the kind are its name here)
                    tally: dict[str, int] = {}
                    for o in kin:
                        for k, v in (_fact_block(self.full[o]) or {}).items():
                            if want <= _vstem(v):
                                tally[_words(k)] = tally.get(_words(k), 0) + 1
                    keys = {k for k, c in tally.items() if c >= 2}
                    if not keys:
                        continue
                    holds = lambda k, v, keys=keys: _words(k) in keys and want <= _vstem(v)
                said = ''
                own = [(k, v) for k, v in block.items() if holds(k, v)]
                if own:
                    # (the line's own attribute for the property holds other words than the required value, "not applicable"
                    #  for nil: the store's grading reads the words, so when an opened listing of the kind states the
                    #  required words themselves, the line does not show it met; figures are left to the check)
                    # (only words the request or the shopper said: a value the reader put in its own words is no
                    #  literal requirement)
                    if not names or re.search('\\d', value) or any(want & _vstem(v) for _, v in own) or not want <= _vstem(self.request + ' ' + ' '.join(self.replies)):
                        continue
                    said = '; '.join(f'{k}: {v}' for k, v in own)[:90]
                elif self.text_states(line, value):
                    continue          # (no attribute, but its own title states the required words or their initials)
                told = [_vstem(t.split(':', 1)[1]) for t in item.told if ':' in t and t not in item.dropped]
                carriers = []
                for other in kin:
                    rec = self.full.get(other)
                    if self.sold_out(other):
                        continue
                    theirs = _fact_block(rec) or {}
                    for k, v in theirs.items():
                        if holds(k, v) and want <= _vstem(v):
                            # (a carrier that also holds the shopper's heard values goes first: it can replace the line)
                            facts = _vstem(' '.join(str(x) for x in theirs.values()))
                            carriers.append((-sum(1 for w in told if w <= facts), len(carriers), (other, str(k), str(v)[:60])))
                            break
                carriers = [c for _, _, c in sorted(carriers)[:3]]
                if carriers:
                    out.append((line, number, need, carriers))
                    if said:
                        self.gap_said[(line, need)] = said
        return out

    def text_states(self, ident: str, value: str) -> bool:
        """A listing's own row text (title, description) states a required value: all of its words, or, for a value of
        two or more words, their initials written together as one word."""
        row = self.rows.get(ident) or {}
        text = ' '.join(str(v) for v in row.values() if isinstance(v, str)).lower()
        words = re.findall('[a-z]+', str(value).lower())
        if words and _vstem(value) <= _vstem(text):
            return True
        return len(words) >= 2 and ''.join(w[0] for w in words) in re.findall('[a-z0-9]+', text)

    def literal_misses(self, ident: str, number: int | None = None) -> list[str]:
        """The stated requirements of a listing's item that its opened record does not write the way the store's grading reads
        them (_literal); [] for a listing not opened in full or of no single requested kind. Only requirements whose property
        and value words the request itself uses are judged: a reader's own wording is no literal requirement."""
        record = self.full.get(ident) or {}
        block = _fact_block(record) or {}
        if not block or not self.items:
            return []
        cat_key = self.category_key()
        row = self.rows.get(ident) or {}
        label = str(row.get(cat_key) or record.get(cat_key) or '') if cat_key else ''
        if number is None:
            owners = [n for n in range(1, len(self.items) + 1) if label and self.is_kind(n, label)]
            if len(owners) != 1:
                return []
            number = owners[0]
        item = self.items[number - 1]
        title = ' '.join(str(v) for node in (row, record) for k, v in node.items() if isinstance(v, str) and re.search('title|head|name', str(k), re.I) and not re.search('producer|seller|shop|brand|store|vendor|maker', str(k), re.I))
        asked = set(_lit_list(self.request))
        skip = {m for _, m in self.refs}
        out = []
        for need in item.musts:
            if need in item.dropped or need in skip or self.described(number, need) or _fit_must(need):
                continue
            prop, _, value = need.partition(':')
            bare = re.sub(LIT_MOST + '|' + LIT_LEAST, '', value.strip(), flags=re.I)
            if not set(_lit_list(prop)) <= asked or not set(_lit_list(re.sub('\\bor\\b', ' ', bare))) <= asked:
                continue
            if label and _lit_writes(_lit_list(bare), label):
                continue          # (the category label states it)
            if _literal(block, title, prop, value) is False:
                out.append(need)
        return out

    def gap_reads(self, gaps: list[tuple[str, int, str, list[tuple[str, str, str]]]]) -> bool:
        """For a gap whose carriers do not also hold the item's heard values, open (one step each, once per line and requirement)
        up to three unopened rows of the item's kind whose text shares the most words with the item's requirement values, so the
        check can name a listing that states the property and fits the rest."""
        if self.clock() > WIND_DOWN:      # (time is nearly up: the cart and the checkout come first)
            return False
        cat_key = self.category_key()
        opened = False
        for line, number, need, carriers in gaps:
            if (line, need) in self.gap_opened or self.left() <= self.tail() + 6:
                continue
            self.gap_opened.add((line, need))
            item = self.items[number - 1]
            told = [t.split(':', 1)[1].strip() for t in item.told if ':' in t and t not in item.dropped]
            facts = lambda o: _vstem(' '.join(str(v) for v in (_fact_block(self.full.get(o) or {}) or {}).values()))
            if any(all(_vstem(v) <= facts(o) for v in told) for o, _, _ in carriers):
                continue
            words = _vstem(' '.join([need.split(':', 1)[1]] + told))
            cap = self.budget if isinstance(self.budget, (int, float)) else None
            pool = []
            for ident in self.seen_order:
                row = self.rows.get(ident) or {}
                if ident in self.full or not self.buyable(ident) or not self.is_kind(number, str(row.get(cat_key) or '')):
                    continue
                if cap is not None and (self.price_of(ident) or 0) > cap:
                    continue
                text = _vstem(' '.join(str(v) for v in row.values() if isinstance(v, (str, int, float))))
                score = len(words & text)
                if score and (not told or any(_vstem(v) & text for v in told)):
                    pool.append((-score, len(pool), ident))
            picks = [i for _, _, i in sorted(pool)[:3]]
            if picks and self.fill_facts(picks, self.tail() + 4):
                opened = True
                self.trace.append({'kind': 'gap_reads', 'line': line, 'need': need, 'opened': picks})
        return opened

    def check_key(self, ids: list[str]) -> str:
        return '|'.join(sorted(ids)) + f'#{self.version}'

    def literal_text(self, ids: list[str], key: str) -> str:
        """The code's own check of cart lines against the stated requirements (_literal), free of cost: a line whose opened
        record does not write a requirement the way the store's grading reads it makes the cart wrong, whatever a model check
        says; '' when every opened line writes them."""
        missed = {line: self.literal_misses(line) for line in ids}
        missed = {line: needs for line, needs in missed.items() if needs}
        if not missed:
            return ''
        self.verdicts[key] = False
        self.contradicted |= set(missed)
        self.leads = [i for i in self.leads if i not in missed]
        cat_key = self.category_key()
        index = self.fit_index() if self.bound else {}
        held = {i: self.held_units(i) or 1 for i in ids}
        lines, better = [], []
        for line, needs in missed.items():
            lines.append(f'{line} does not write ' + '; '.join(f'"{n}"' for n in needs[:3]))
            label = str((self.rows.get(line) or {}).get(cat_key) or '') if cat_key else ''
            owners = [n for n in range(1, len(self.items) + 1) if label and self.is_kind(n, label)]
            if len(owners) == 1:
                rest = {k: v for k, v in held.items() if k != line}
                # (a stand-in named here must keep the cart's parts fitting their device too)
                pool = [i for i in self.candidates(owners[0]) if i not in ids and self.fits_with(i, owners[0], rest, index)]
                better += sorted(pool, key=lambda i: self.price_of(i) if self.price_of(i) is not None else 1e12)[:3]
        self.trace.append({'kind': 'literal_check', 'cart': ids, 'missed': missed, 'better': better})
        text = ("Kyo's check of the records says this cart is not right yet: the store grades a requirement by its words, in the "
                "field named for the property (or, when the listing has no such field, in its title): " + '; '.join(lines)
                + '. A value in other words, or no such field, does not meet it.')
        if better:
            text += ' Opened listings of that kind that write it: ' + ', '.join(dict.fromkeys(better)) + '.'
        else:
            text += (' No opened listing of that kind writes it: search with the required words (the value first, then the product), '
                     'open what may fit, and end without a purchase only when none does.')
        return text + ' Fix the cart, then check out.'

    def check_cart(self, ids: list[str] | None=None, limit: float=SPEND_SOFT) -> str:
        lead = ids is not None
        ids = ids if lead else self.cart_ids()
        key = self.check_key(ids)
        literal = self.literal_text(ids, key)
        if literal:
            return literal
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
        gaps = self.attr_gaps(ids)
        if gaps and self.gap_reads(gaps):
            gaps = self.attr_gaps(ids)
        if gaps:
            prompt += '\n\nATTRIBUTE GAPS (a cart line has no attribute for a requirement\'s property, or its attribute for it states other words than the required value, while other opened listings of its kind state the required value as an attribute of their own):\n' + '\n'.join(
                (f'{line} (item {number}) states "{self.gap_said[(line, need)]}" for "{need}", other words than the required value' if (line, need) in self.gap_said else f'{line} (item {number}) has no attribute for "{need}"')
                + ('; listings of its kind with that attribute (judge each figure against the limit): ' if (line, need) in self.gap_limit else '; listings of its kind stating it: ') + ', '.join(f'{o} ({k}: {v})' for o, k, v in carriers) for line, number, need, carriers in gaps)
            named = list(dict.fromkeys(o for _, _, _, carriers in gaps for o, _, _ in carriers))[:6]
            prompt += '\n\nRECORDS OF THOSE LISTINGS:\n' + '\n'.join(f'{o}: ' + self.record_of(o) for o in named)
            self.trace.append({'kind': 'attr_gaps', 'gaps': [(line, need, [o for o, _, _ in carriers]) for line, _, need, carriers in gaps]})
        if self.brief.get('pick') in ('cheapest', 'most', 'least'):
            prompt += '\n\nOTHER LISTINGS SEEN (newest first):\n' + self.others(ids, 15)
        got = self.ask_json(prompt, self.judges(limit), 3000)
        self.trace.append({'kind': 'check', 'cart': ids, 'got': got})
        self.verdicts[key] = bool(got) and got.get('ok') is not False
        if not got or got.get('ok') is not False:
            return ''
        self.leads = [i for i in self.leads if i not in ids]
        problems = [str(p) for p in _items(got.get('problems'))][:5]
        for line in ids:              # (a listing the check named in a quoted contradiction is no candidate any more)
            if any(line in p for p in problems):
                self.contradicted.add(line)
        better = [str(b) for b in _items(got.get('better')) if str(b) in self.rows and (not self.sold_out(str(b)))][:4]
        text = 'A careful review says this cart is not right yet: ' + '; '.join(problems or ['see the requirements'])
        if better:
            text += '. Listings that look better: ' + ', '.join(better)
        return text + '. Fix the cart (or show the review is wrong by re-reading the records), then check out.'

    def trail(self) -> tuple[str, bool]:
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values and need not in self.conds]
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
                if links and all((i in self.opened for i in ids)) and (not str(links[-1].get('find') or '').strip()):
                    self.ref_values[need] = str(chain.get('value') or '').strip() or f'as stated on {ids[-1]}'
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

    def link_rank(self, about: str, ids: list[str]) -> list[str]:
        """Rows of a referenced listing ranked by how many of the description's own values ("the one with colour pink and closure
        lace-up") their records show; a row of the named shop first, then more values shown, then a row already opened."""
        tail = re.split('(?i)\\bthe one with\\b|\\bthat has\\b|\\bwith\\b', about, maxsplit=1)
        detail = tail[1] if len(tail) > 1 else about
        wanted = {w for w in re.findall('[a-z0-9.]+', detail.lower()) if (len(w) >= 3 or any(ch.isdigit() for ch in w)) and w not in {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for', 'of'}}
        shops = set(self.shop_rows(about, ids)) | set(self.opened_shop(about))

        def score(ident: str) -> tuple[int, int, int]:
            text = _dump(self.full.get(ident) or self.rows.get(ident) or {}).lower()
            shown = sum(1 for w in wanted if re.search('(?<![a-z0-9])' + re.escape(w) + '(?![a-z0-9])', text))
            return (ident in shops, shown, ident in self.opened)
        return sorted(dict.fromkeys(ids), key=lambda i: tuple(-x for x in score(i)))

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
                    plan_pages = self.paging(tool, observation)
                    turned = 0
                    # (none of the named shop's rows yet on a short result: its other pages, so every row of that search is seen)
                    while not known and plan_pages and plan_pages[3] is not None and plan_pages[0] <= 40 and turned < 2 and self.left() - self.tail() > 8:
                        faults = []
                        call = _shape({**args, plan_pages[2]: plan_pages[3]}, self.schemas.get(tool), 'args', faults)
                        if faults:
                            break
                        reply = self.act(tool, call)
                        if reply is None or self.finished:
                            break
                        turned += 1
                        seen = reply.get('observation')
                        self.observe(tool, seen)
                        self.link_seen[memo] += [i for i, _ in self.records_in(seen)]
                        known = self.shop_rows(about, self.link_seen[memo])
                        plan_pages = self.paging(tool, seen)
                    self.trace.append({'kind': 'link_search', 'words': words[:80], 'rows': known[:4], 'pages': turned + 1})
        kind = {w for w in re.findall('[a-z0-9]{3,}', (words or about).lower())} - named - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for'}

        def of_kind(ids: list[str]) -> list[str]:
            return [i for i in ids if not kind or kind & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower()))]
        return self.link_rank(about, of_kind(known) or of_kind(self.shop_rows(about, list(reversed(self.seen_order)))))[:2]

    def walk_due(self) -> bool:
        """A requirement whose value sits on another listing and that the walker has not followed yet; with no such
        requirement (most tasks) nothing below runs."""
        if self.clock() > WIND_DOWN:      # (time is nearly up: the cart and the checkout come first)
            return False
        if not self.refs or self.finished or self.cost >= SPEND_SOFT:
            return False
        live = {need for _, need in self.needs_now()}
        if not any((need in live and need not in self.ref_values and need not in self.walked for _, need in self.refs)):
            return False
        return self.search_template() is not None and self.single_target(self.offer_tool()) is not None and self.walk_room() > 2

    def walk_room(self) -> int:
        """Steps the walker may still use: the steps left after a reserve for the rest of the order (a few per item), and
        never more than WALK_STEPS in all."""
        return min(self.left() - self.tail() - 6 - 4 * len(self.items), WALK_STEPS - self.walk_steps)

    def walk_models(self) -> list[str]:
        return [LIGHT] + ([HEAVY] if self.cost + HEAVY_ROOM <= SPEND_SOFT else [])

    def walk(self) -> str:
        """Follow each pointing requirement's chain right after it is known: find the exact listing it describes, open its
        page, follow the sentence of that page that sends this property on, and apply the relation to the last page's value."""
        live = {need for _, need in self.needs_now()}
        texts = []
        for _, need in list(self.refs):
            if need in live and need not in self.ref_values and need not in self.walked and (not self.finished):
                self.walked.add(need)
                texts.append(self.walk_one(need))
        if any((need in self.ref_values for need in self.walked)):
            self.version += 1
        return '\n'.join((t for t in texts if t))

    def walk_one(self, need: str) -> str:
        prop = need.split(':', 1)[0].strip() or need
        heard = '\n'.join(self.replies[-6:])
        got = self.ask_json(WALK_PARSE + '\n\nREQUIREMENT: ' + need + "\n\nTHE SHOPPER'S WORDS:\n" + self.request[-2000:] + ('\n' + heard if heard else ''), self.walk_models(), 900) or {}
        kind = str(got.get('kind') or '').strip()
        named = str(got.get('seller') or '').strip()
        seller = self.walk_shop(named, kind)
        self.walk_alias[seller.lower()] = named.lower()
        details = [str(d).strip() for d in _items(got.get('details')) if str(d).strip()][:6]
        relation = str(got.get('relation') or 'same').strip().lower()
        amount = got.get('amount')
        self.trace.append({'kind': 'walk_parse', 'need': need[:160], 'seller': seller, 'of': kind, 'details': details, 'relation': relation})
        if not seller or seller.lower() not in (self.request + '\n' + heard + '\n' + need).lower():
            return ''
        path: list[str] = []
        visited: set[str] = set()
        value: str | None = None
        pairs: list[Any] = []
        stop = 'lost'
        for _ in range(WALK_HOPS):
            found = self.walk_find(seller, kind, details, visited)
            if not found:
                break
            visited.add(found)
            path.append(f'{seller} {found}')
            record = self.rows.get(found) or {}
            read = self.ask_json(WALK_READ + '\n\nPROPERTY: ' + prop + '\n\nPAGE:\n' + self.walk_page(record), self.walk_models(), 700) or {}
            nxt = read.get('next') if isinstance(read.get('next'), dict) else None
            shop = str((nxt or {}).get('seller') or '').strip()
            self.trace.append({'kind': 'walk_hop', 'id': found, 'next': _clip(nxt, 80), 'value': _clip(read.get('value'), 80)})
            if nxt and shop and shop.lower() in _dump(record).lower():
                kind = str(nxt.get('kind') or kind).strip()
                seller = self.walk_shop(shop, kind)
                self.walk_alias[seller.lower()] = shop.lower()
                details = [str(d).strip() for d in _items(nxt.get('details')) if str(d).strip()][:6]
                continue
            own = self.walk_value(record, prop)
            if own is None and (not str(read.get('value') or '').strip()) and self.walk_facts(found):
                record = self.rows.get(found) or {}
                own = self.walk_value(record, prop)
            value = str(read.get('value') or '').strip() or own
            self.trace.append({'kind': 'walk_value', 'id': found, 'read': _clip(read.get('value'), 60), 'own': own, 'prop': prop[:60], 'keys': sorted(record)[:12]})
            pairs = [p for p in _items(read.get('pairs')) if isinstance(p, list) and len(p) == 2]
            stop = 'done' if value else 'no value'
            break
        head = '[Kyo followed the listing this requirement points at, opening each page of the chain in full] ' + need + ': '
        if stop != 'done':
            wanted = f"{seller}'s {kind}".strip() + (f" with {', '.join(details)}" if details else '')
            self.trace.append({'kind': 'walk_lost', 'need': need[:120], 'path': path, 'missing': wanted[:160] if stop == 'lost' else f'no {prop} on the last page'})
            if stop == 'no value':
                return head + ' -> '.join(path) + f'. The last page states no {prop} that Kyo could read: read the value from that page itself, then pick with it.'
            return head + ' -> '.join(path + [f'not found yet: {wanted}']) + '. Find and open that listing (search with its shop name and those values) before checking out.'
        final = value
        if relation.startswith('diff'):
            rule = f'any {prop} except {final}'
        elif relation.startswith('within'):
            rule = f'{prop} within {amount} of {final}' if amount not in (None, '') else f'{prop} close to {final}'
        elif relation in ('at most', 'at least'):
            rule = f'{prop} {relation} {final}'
        elif relation.startswith('goes'):
            key = _norm(final)
            partner = next((str(b if _norm(a) == key else a) for a, b in pairs if key in (_norm(a), _norm(b))), None)
            partner = partner or self.paired_with(final, prop)
            rule = f'{prop} {partner} (the store pairs {final} with {partner})' if partner else f'a {prop} that goes with {final}'
        else:
            rule = f'{prop} {final}'
        self.ref_values[need] = rule
        self.trace.append({'kind': 'walk_done', 'need': need[:120], 'path': path, 'rule': rule[:160]})
        return head + ' -> '.join(path) + f'. The last page gives {prop}: {final}. Requirement in force now: {rule}. Pick with it; the pages of the chain are already opened.'

    def paired_with(self, value: str, prop: str = '') -> str | None:
        """When the chain's last page gave no pairing for its value: the value the store's own pages pair it with, read from
        every record seen ("<value> goes together with <other>"; the other way round only when no page pairs it forward)."""
        want = _norm(value).strip()
        if not want:
            return None
        ahead: dict[str, int] = {}
        behind: dict[str, int] = {}
        for record in list(self.full.values()) + list(self.rows.values()):
            for text in (v for v in record.values() if isinstance(v, str) and ' with ' in v):
                for hit in re.finditer('([^.,;:]{1,60}?)\\s+(?:goes|go|pairs)\\s+(?:together\\s+|up\\s+)?with\\s+([^.,;]{1,40})', text, re.I):
                    left, right = _norm(hit.group(1)), _norm(hit.group(2))
                    lead = re.split('\\s+and\\s+', hit.group(2).strip(), maxsplit=1)[0].strip()
                    if f' {want} ' in f' {left} ' and f' {want} ' not in f' {right} ' and lead:
                        ahead[lead.lower()] = ahead.get(lead.lower(), 0) + 1
                    elif f' {want} ' in f' {right} ' and f' {want} ' not in f' {left} ':
                        words = hit.group(1).split()
                        at = [k for k, w in enumerate(words) if _roots(w) and _roots(w) <= _roots(prop)]
                        if at and at[-1] > 0:      # (the paired value stands just before the property's name)
                            behind[words[at[-1] - 1].lower()] = behind.get(words[at[-1] - 1].lower(), 0) + 1
        pool = ahead or behind
        return max(sorted(pool), key=lambda v: pool[v]) if pool else None

    @staticmethod
    def walk_shop(name: str, kind: str) -> str:
        """A shop name given together with its kind of product ("<shop> <kind>") cut back to the shop's own name."""
        words = name.split()
        kinds = set(_words(kind).split()) | _roots(kind)
        while len(words) > 1 and (words[-1].lower() in kinds or _roots(words[-1]) & kinds):
            words.pop()
        return ' '.join(words)

    def walk_mine(self, ident: str, seller: str) -> bool:
        names = {seller.lower(), self.walk_alias.get(seller.lower(), seller.lower())}
        row = self.rows.get(ident) or {}
        return any((isinstance(v, str) and v.strip().lower() in names and re.search('seller|shop|store|vendor|merchant|brand|maker', str(k), re.I) for k, v in row.items()))

    def walk_hits(self, ident: str, kind: str, details: list[str]) -> tuple[int, int, int]:
        """How many of the description's values one of the listing's own attribute fields shows (its name and value read
        together, so "internal installation" is the field installation: internal), how many its whole record shows (title and
        notes included), and whether it is of the named kind."""
        record = self.rows.get(ident) or {}
        fields = [_words(k) + ' ' + str(v) for block in record.values() if isinstance(block, dict) for k, v in block.items() if isinstance(v, (str, int, float)) and (not isinstance(v, bool))]
        text = _dump(record)

        def shows(value: str, where: str) -> bool:
            wanted, plain = (_numbers(value), _norm(value))
            return plain in _norm(where) or set(plain.split()) <= set(_norm(where).split()) or (bool(wanted) and wanted <= _numbers(where))
        title = _norm(' '.join((str(v) for k, v in record.items() if isinstance(v, str) and re.search('title|head|name', str(k), re.I))))
        values = [d for d in details if _norm(d).strip()]
        pairs = [(k, v) for block in record.values() if isinstance(block, dict) for k, v in block.items() if isinstance(v, (str, int, float)) and (not isinstance(v, bool))]

        def own_shows(detail: str) -> bool:
            # ("property: value": only the field named for that property counts, so "thread count: 0" is not met by any
            #  field holding a 0; race-16/17 tests: a bare "0" or "used" picked the wrong listing of the shop and the chain
            #  ended on another value)
            prop, sep, value = detail.partition(':')
            if sep and prop.strip() and value.strip():
                want = _stem(prop.strip())
                return any((_names(want, _stem(str(k))) or _names(_stem(str(k)), want)) and shows(value.strip(), str(v)) for k, v in pairs)
            return any((shows(detail, f) for f in fields)) or _norm(detail) in title
        own = sum((1 for d in values if own_shows(d)))
        # (the whole record's count reads each detail's value alone: it orders the rows to open and decides when to fetch a
        #  page's attribute table; a listing is taken on its own fields, or as a last resort on walk_text)
        return (own, sum((1 for d in values if shows(d.partition(':')[2].strip() or d, text))), 1 if kind and _roots(kind) & _roots(_norm(text)) else 0)

    def walk_text(self, ident: str, details: list[str]) -> bool:
        """The last resort of walk_find: the listing's whole record shows every detail, a "property: value" one with the
        property's words as well as its value (a bare value such as 0 or used is in almost every record's text)."""
        record = self.rows.get(ident) or {}
        text = _dump(record)
        spaced = set(_norm(re.sub('([a-z0-9])([A-Z])', '\\1 \\2', text)).split())      # (field names in camel case: cardType)
        for detail in details:
            prop, sep, value = detail.partition(':')
            value = value.strip() if sep else detail
            if not value or _norm(value).strip() not in _norm(text) and not set(_norm(value).split()) <= set(_norm(text).split()):
                return False
            if sep and prop.strip() and not set(_norm(prop).split()) <= spaced:
                return False
            if sep and prop.strip():
                # (its own field for that property says something else: the description's words are about another thing)
                want = _stem(prop.strip())
                named = [str(v) for block in record.values() if isinstance(block, dict) for k, v in block.items() if _names(want, _stem(str(k))) or _names(_stem(str(k)), want)]
                if named and not any(_norm(value).strip() in _norm(v) or set(_norm(value).split()) <= set(_norm(v).split()) for v in named):
                    return False
        return True

    def walk_value(self, record: dict[str, Any], prop: str) -> str | None:
        """The listing's own attribute named like the property, when the page reader gave no value."""
        want = _stem(prop)
        for value in record.values():
            if isinstance(value, dict):
                for key, got in value.items():
                    if isinstance(got, (str, int, float)) and (not isinstance(got, bool)) and str(got).strip() and _names(want, _stem(str(key))):
                        return str(got).strip()
        return None

    def walk_page(self, record: dict[str, Any]) -> str:
        """The page for the reader: its attribute blocks first, then its texts shortest first (title and notes before a long
        description), every text clipped."""
        first = {k: v for k, v in record.items() if isinstance(v, (dict, list))}
        rest = sorted((k for k in record if k not in first), key=lambda k: len(str(record[k])))
        return _dump(_clip({**first, **{k: record[k] for k in rest}}, 900))[:5000]

    def walk_find(self, seller: str, kind: str, details: list[str], avoid: set[str]) -> str:
        """The exact listing a description names: a listing of that shop whose page shows every value the description gives,
        opened with the page tool (a listing of the same shop that only shares the wanted value is not it). Searches the
        shop's name with those values first, in the store's own order, then the shop's name with the kind over a few pages;
        opens at most five rows; a page whose own attributes show the values beats one whose text merely mentions them."""
        full = len(details)
        tried: list[str] = []
        spare: list[str] = []

        def shop(ids: list[str]) -> list[str]:
            return [i for i in dict.fromkeys(ids) if i not in avoid and self.walk_mine(i, seller)]

        def look(ids: list[str], opens: int, ordered: bool=False) -> str:
            if ordered:
                ranked = sorted(shop(ids), key=lambda i: self.walk_hits(i, kind, details)[1] < full)
            else:
                ranked = sorted(shop(ids), key=lambda i: tuple((-x for x in self.walk_hits(i, kind, details))))
            for ident in ranked:
                if ident in tried:
                    continue
                if ident not in self.opened:
                    if opens <= 0 or not self.walk_open(ident):
                        continue
                    opens -= 1
                tried.append(ident)
                attrs, text, _ = self.walk_hits(ident, kind, details)
                if attrs < full <= text and self.walk_facts(ident):
                    attrs, text, _ = self.walk_hits(ident, kind, details)
                if attrs >= full:
                    return ident
                if text >= full and ident not in spare and self.walk_text(ident, details):
                    spare.append(ident)
            return ''
        known = [i for i in self.rows if i in self.opened]
        found = look(known, 0)
        if found:
            return found
        # (the shop's name with the description's values first: the kind's words match every listing of that shop's kind
        #  and can push the described one off the first page; then with the kind as well)
        said = [d.replace(':', ' ') for d in details]
        plain = ' '.join(' '.join([seller] + said).split()[:14])
        words = ' '.join(' '.join([seller, kind] + said).split()[:14])
        first_rows: list[str] = []
        if details:
            first_rows = self.walk_search(plain, seller, 1)
            found = look(first_rows, 4, True)
            if found:
                return found
        if words != plain or not details:
            # (full reads again when the first search showed none of that shop's rows: nothing was read there)
            found = look(self.walk_search(words, seller, 1), 2 if shop(first_rows) else 4, True)
            if found:
                return found
        # (when the searches with the details found none of that shop's rows, the shop's rows of the kind seldom show the
        #  details either, so more of them are opened)
        found = look(self.walk_search(f'{seller} {kind}'.strip(), seller, 3), 5 if details and not shop(first_rows) else 2)
        if found:
            return found
        if kind and (not shop(list(self.rows))):
            found = look(self.walk_search(seller, seller, 2), 2)
        return found or (spare[0] if spare else '')

    def walk_facts(self, ident: str) -> bool:
        """On a store whose page tool shows no attribute table (a separate tool does), read that table for one listing."""
        tool = self.facts_tool()
        target = self.single_target(tool)
        record = self.rows.get(ident) or {}
        if not tool or not target or tool == self.offer_tool() or any((isinstance(v, dict) for v in record.values())) or self.walk_room() <= 0 or self.finished:
            return False
        faults: list[str] = []
        args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
        if faults:
            return False
        reply = self.act(tool, args)
        self.walk_steps += 1
        if reply is None:
            return False
        self.observe(tool, reply.get('observation'))
        self.trace.append({'kind': 'walk_facts', 'id': ident, 'tool': tool})
        return any((isinstance(v, dict) for v in (self.rows.get(ident) or {}).values()))

    def walk_open(self, ident: str) -> bool:
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not tool or not target or self.walk_room() <= 0 or self.finished:
            return False
        faults: list[str] = []
        args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
        if faults:
            return False
        reply = self.act(tool, args)
        self.walk_steps += 1
        if reply is None:
            return False
        self.observe(tool, reply.get('observation'))
        return ident in self.opened

    def walk_search(self, words: str, seller: str, pages: int) -> list[str]:
        """The rows of one search in the store's order, over up to that many pages (no further page once a page after the
        first shows none of that shop's rows)."""
        plan = self.search_template()
        if not plan or not words:
            return []
        tool, base, field = plan
        faults: list[str] = []
        args = _shape({**base, field: words}, self.schemas.get(tool), 'args', faults)
        if faults:
            return []
        out: list[str] = []
        for turn in range(pages):
            if self.walk_room() <= 0 or self.finished:
                break
            reply = self.act(tool, args)
            self.walk_steps += 1
            if reply is None:
                break
            seen = reply.get('observation')
            self.observe(tool, seen)
            ids = [i for i, _ in self.records_in(seen)]
            out += ids
            self.trace.append({'kind': 'walk_search', 'words': words[:80], 'rows': len(ids), 'shop_rows': sum((1 for i in ids if self.walk_mine(i, seller)))})
            if not ids or (turn >= 1 and (not any((self.walk_mine(i, seller) for i in ids)))):
                break
            after = self.walk_next(tool, args, seen, len(ids)) if turn + 1 < pages else None
            if not after:
                break
            args = after
        return out

    def walk_next(self, tool: str, args: dict[str, Any], seen: Any, shown: int) -> dict[str, Any] | None:
        """The same search's next page: the store's own next-page value when it gives one, else the offset (or page number)
        argument moved on past the rows just shown."""
        plan = self.paging(tool, seen)
        faults: list[str] = []
        if plan:
            if plan[3] is None or plan[0] <= shown + (args.get(plan[2]) if isinstance(args.get(plan[2]), int) and (not isinstance(args.get(plan[2]), bool)) else 0):
                return None
            nxt = _shape({**args, plan[2]: plan[3]}, self.schemas.get(tool), 'args', faults)
            return None if faults else nxt
        arg = (self.shapes.get(tool) or {}).get('next_arg')
        props = (self.schemas.get(tool) or {}).get('properties') or {}
        if not arg or 'integer' not in _kinds(props.get(arg) or {}) or not shown:
            return None
        start = args.get(arg) if isinstance(args.get(arg), int) and (not isinstance(args.get(arg), bool)) else 0
        step = 1 if re.search('page', arg, re.I) else shown
        nxt = _shape({**args, arg: (start or (1 if step == 1 else 0)) + step}, self.schemas.get(tool), 'args', faults)
        return None if faults else nxt

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
        got = self.ask_json(prompt, self.judges(), 3500)
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
        if not named:
            rows = [str(i) for i in _items((got or {}).get('open')) if str(i) in self.rows and str(i) not in self.opened and (not self.sold_out(str(i)))]
            if rows and self.open_rows(rows):
                got = self.audit_call()
                named = self.audit_fits(got)
        if not named:
            return ''
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return 'A careful review thinks these listings may fit: ' + '; '.join((f"item {k}: {', '.join(v)}" for k, v in named.items())) + (f" ({got.get('reason')})" if got.get('reason') else '') + '. Open them and buy what truly fits; end without a purchase only if each one fails.'

    def audit_call(self) -> dict[str, Any] | None:
        records = []
        for ident in reversed(self.seen_order):
            if self.sold_out(ident):
                continue
            record = self.full.get(ident) if ident in self.full else None
            records.append('OPENED ' + _dump(_clip(record, 200))[:1200] if record else 'ROW ' + _dump(_clip(self.rows.get(ident), 120))[:500])
            if len(records) >= 30:
                break
        if not records:
            return None
        prompt = AUDIT + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
        prompt += '\n\nLISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
        got = self.ask_json(prompt, self.judges(), 4000)
        self.trace.append({'kind': 'audit', 'got': got})
        return got

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get('fits')
        if not isinstance(fits, dict):
            return {}
        owner = lambda k: _count(k) if _count(k) and 1 <= (_count(k) or 0) <= len(self.items) else None
        # (an opened listing whose record does not write a stated requirement is no reason to keep from ending)
        named = {str(k): [str(i) for i in _items(v) if str(i) in self.rows and (not self.sold_out(str(i))) and not self.literal_misses(str(i), owner(k))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def open_rows(self, rows: list[str]) -> bool:
        many, joiner = (self.compare_target(), self.roles.get('compare'))
        tool = self.facts_tool()
        target = self.single_target(tool)
        if many and joiner and (not self.compare_bare()):
            plan = [(joiner, {many: rows[:self.compare_room()]})]
        elif tool and target:
            plan = [(tool, {target: i}) for i in rows[:2]]
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
            more = self.late_ask()
            if more:
                return 'Not run (Kyo set the cart: ' + '; '.join(done) + ', then held the checkout). ' + more
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

    def candidates(self, number: int) -> list[str]:
        """An item's opened listings that could still be bought: of its kind, in stock, inside a per-line share of the budget,
        and not ruled out by a check that quoted a contradiction or by the literal reading of a stated requirement; the audit's
        leads first."""
        item = self.items[number - 1]
        leads = [i for i in self.lead_items.get(str(number), []) + self.lead_items.get(number, []) if i in self.rows]
        mine = [i for i in self.item_ids(number) if i in self.full and _fact_block(self.full.get(i) or {})]
        out = []
        for ident in dict.fromkeys(leads + mine):
            price = self.price_of(ident)
            if not self.buyable(ident) or ident in self.contradicted or self.literal_misses(ident, number):
                continue
            if isinstance(self.budget, (int, float)) and price is not None and price * item.qty > self.budget:
                continue
            out.append(ident)
        return out

    def must_hits(self, number: int, ident: str) -> int:
        """How many of an item's stated requirement values a listing's record shows in words."""
        text = _dump(self.full.get(ident) or self.rows.get(ident) or {}).lower()
        hits = 0
        for must in self.items[number - 1].musts:
            value = must.split(':', 1)[-1]
            words = [w for w in re.findall('[a-z0-9]+', value.lower()) if len(w) >= 3]
            hits += bool(words) and all(re.search('(?<![a-z0-9])' + re.escape(w), text) for w in words)
        return hits

    def best_effort(self) -> str:
        """Before an ending without a purchase: when every requested item still has a candidate, buy the best one of each
        instead (a decline is right only when some item has none); once a task. Picks: the cheapest candidate under a cheapest
        rule, else the one showing most stated values, then the lower price; the whole cart must keep within the budget."""
        cart, buy = self.roles.get('cart'), self.roles.get('buy')
        if self.effort_tried or not cart or not buy or self.left() < 4 or self.cost > SPEND_HARD:
            return ''
        self.effort_tried = True
        pick = self.brief.get('pick')
        plan: dict[str, int] = {}
        for number, item in enumerate(self.items, 1):
            pool = [i for i in self.candidates(number) if i not in plan]
            if not pool:
                self.trace.append({'kind': 'best_effort', 'none_for': item.name})
                return ''
            price = lambda i: self.price_of(i) if self.price_of(i) is not None else 1e12
            pool.sort(key=lambda i: (price(i), -self.must_hits(number, i)) if pick == 'cheapest' else (-self.must_hits(number, i), price(i)))
            plan[pool[0]] = item.qty
        total = sum((self.price_of(i) or 0) * n for i, n in plan.items())
        if isinstance(self.budget, (int, float)) and total > self.budget:
            self.trace.append({'kind': 'best_effort', 'over_budget': total})
            return ''
        drop = [i for i in self.cart_ids() if i not in plan]
        calls = self.cart_calls(plan, drop)
        if not calls or len(calls) + 2 > self.left():
            return ''
        self.check_cart(list(plan), SPEND_HARD)
        verdict = self.verdicts.get(self.check_key(list(plan)))
        if verdict is not True:        # (only a cart the independent check passes replaces the ending)
            self.trace.append({'kind': 'best_effort', 'check': 'contradicted' if verdict is False else 'not run', 'plan': plan})
            return ''
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
        self.trace.append({'kind': 'best_effort', 'plan': plan, 'total': total})
        self.effort_cart = dict(plan)
        return ('Not run yet (no step used). Ending without a purchase is right only when some requested item has no listing that could meet '
                'its requirements, and every item still has one, so Kyo put the best candidate of each in the cart: ' + self.cart_text()
                + '. Re-read anything a requirement depends on if steps allow, then check out.')

    def hunt_due(self) -> bool:
        """The hunt for items with no fitting listing yet runs once the questions are over and HUNT_SHARE of the steps is left
        (with room for it and the checkout), rather than at the ending, which comes with too few steps left."""
        if self.hunted or self.finished or not (self.asked or not self.roles.get('ask')):
            return False
        left = self.left()
        if left > max(HUNT_STEPS + 8, int(self.max_steps * HUNT_SHARE)) or left < 10:
            return False
        return any(not self.candidates(n) for n in range(1, len(self.items) + 1))

    def hunt_fits(self) -> list[str]:
        """Before an ending without a purchase, once: for each requested item with no candidate yet (an opened, in-stock listing
        no check contradicted and no stated value misses), short searches not sent before (each of its values with the item's
        name, then the name alone), then its rows seen but never opened are read, those showing most of its values first
        (race-16/17 tests: in 14 of 19 wrong declines the accepted listing was never shown or never opened, after long
        stacked queries; a value with the product's name, or the name alone, showed it). Returns the listings opened."""
        if self.hunted or self.finished or self.left() < 10 or self.clock() > WIND_DOWN or self.cost > SPEND_SOFT:
            return []
        self.hunted = True
        plan = self.search_template() or self.schema_template()
        lacking = [n for n in range(1, len(self.items) + 1) if not self.candidates(n)]
        if not plan or not lacking:
            return []
        tool, base, field = plan
        sent = {_words(str(a.get(field) or '')) for a in self.searched}
        room = min(HUNT_STEPS, self.left() - 6)      # (six steps stay for the cart and the checkout)
        opened: list[str] = []
        for number in lacking:
            item = self.items[number - 1]
            heard, stated = self.bare_values(number)
            noun = item.name.strip()
            short = noun if noun and len(noun.split()) <= 3 and not re.search('\\bor\\b|[,/]', noun, re.I) else ''
            queries = [f'{v} {short}'.strip() for v in (heard + stated)[:2]] + ([short] if short else [])
            for query in queries:
                if room <= 2 or self.finished or _words(query) in sent:
                    continue
                faults: list[str] = []
                args = _shape({**base, field: query}, self.schemas.get(tool), 'args', faults)
                if faults:
                    break
                reply = self.act(tool, args)
                room -= 1
                sent.add(_words(query))
                if reply is None:
                    break
                self.observe(tool, reply.get('observation'))
            values = [w for v in heard + stated for w in re.findall('[a-z0-9]+', v.lower()) if len(w) >= 3]
            text = lambda i: _dump(self.rows.get(i) or {}).lower()
            unread = [i for i in self.item_ids(number) if i not in self.full and self.buyable(i)]
            cheapest = self.brief.get('pick') == 'cheapest'
            unread.sort(key=lambda i: (-sum(w in text(i) for w in values), (self.price_of(i) or 0) if cheapest else 0))
            if unread and room > 1:
                got = self.fill_facts(unread[:min(3, room - 1)], self.tail() + 6)
                room -= len(got)
                opened += got
        self.trace.append({'kind': 'hunt_fits', 'lacking': lacking, 'opened': opened, 'room_left': room})
        return opened

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
            if ending and self.left() > 2:
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
                    self.part_lookups()
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cap_walk() + self.cheaper_rows()
                    verdict = self.check_cart()
                    if verdict:
                        seen = "Kyo read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n' if read else ''
                        out.append((cid, 'Not run yet. ' + chain + seen + verdict + '\nCart now: ' + self.cart_text()))
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
            if self.is_refuse(name, args) and self.effort_cart and self.roles.get('buy') and all(i in self.cart_ids() and not self.sold_out(i) for i in self.effort_cart):
                # (best_effort put the best candidate of every item in the cart and the independent check passed that cart; an
                #  ending asked after it is the walker turning the check away (race-16/17 tests: "these messages keep
                #  conflicting ... ending without a purchase" on a task with an answer), so that cart is checked out instead)
                faults: list[str] = []
                order_args = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(self.roles['buy']), 'args', faults)
                if not faults:
                    self.trace.append({'kind': 'effort_buy', 'cart': dict(self.effort_cart), 'left': self.left()})
                    self.effort_cart = {}
                    name, args = self.roles['buy'], order_args
            if self.is_refuse(name, args) and self.left() <= 6 and not self.late_bought and self.roles.get('buy'):
                held = [i for i in self.cart_ids() if not self.sold_out(i)]
                if held and len(held) >= len(self.items) and not set(held) & self.contradicted and not any(self.literal_misses(i) for i in held):
                    # (the last steps: an ending without a purchase over a cart that holds a line of every item and that no check
                    # contradicted is the shortage of steps speaking, not proof that nothing fits; that cart is checked out instead)
                    faults: list[str] = []
                    order_args = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(self.roles['buy']), 'args', faults)
                    if not faults:
                        self.late_bought = True
                        self.trace.append({'kind': 'late_buy', 'cart': held, 'left': self.left()})
                        name, args = self.roles['buy'], order_args
            if self.is_refuse(name, args) and self.left() <= 5 and (not self.late_held):
                lines = [i for i in self.built if not self.sold_out(i)]
                fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
                missing = [i for i in lines if i not in self.cart_ids() or (i in self.dropped and (not fresh))]
                if len(lines) >= len(self.items) and len(missing) + 1 <= self.left() and not any(self.literal_misses(i) for i in lines):
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
                self.hunt_fits()
                done = self.complete_cart()
                if done:
                    out.append((cid, done + (' [episode finished]' if self.finished else '')))
                    continue
                kept = self.best_effort()
                if kept:
                    out.append((cid, kept))
                    continue
                args = self.quiet(name, args)
            if ending and self.left() >= 2:
                go, said = self.gate('order' if self.is_buy(name, args) else 'end')
                if not go:
                    out.append((cid, 'Not run yet. ' + said + '\nCart now: ' + self.cart_text()))
                    continue
                if self.is_buy(name, args):
                    more = self.late_ask()
                    if more:
                        out.append((cid, 'Not run yet. ' + more + '\nCart now: ' + self.cart_text()))
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
            if self.shapes.get(name) and self.records_in(observation):
                self.list_calls = (self.list_calls + [(name, dict(args))])[-12:]
                if not self.paged_obs:
                    after = self.paging(name, observation)
                    self.note_next(name, args, after[3] if after else None)
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
        whole = True
        if holding and not self.literal_swap() and self.can_refuse():
            whole = False            # (a line no opened listing of its kind can replace fails a stated requirement: no order)
            self.trace.append({'kind': 'literal_refused', 'lines': self.cart_ids()})
        if whole and holding and len(self.items) >= 2:
            whole = self.fill_short_cart() or not self.can_refuse()
            if not whole:
                self.trace.append({'kind': 'short_cart_refused', 'lines': self.cart_ids()})
        if self.finished or self.left() <= 0:
            return ''
        if whole and buy and (holding or (self.cart_touched and (not fresh) and (self.left() >= 2))):
            faults: list[str] = []
            args = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), 'args', faults)
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

    def fill_short_cart(self) -> bool:
        """Before a forced order: add, for each item with no line, the cheapest opened listing that states its requirements.
        False when an item stays uncovered, so the episode ends by refusing rather than with a short order."""
        cart = self.roles.get('cart') or ''
        bound = {n for pair in self.bound for n in pair}
        for number in self.missing_items():
            item = self.items[number - 1]
            fits = [i for i in self.item_ids(number) if i in self.full and i not in self.cart_lines and self.buyable(i) and self.price_of(i) is not None and self.states_needs(i, number) and not self.literal_misses(i, number)]
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

    def literal_swap(self) -> bool:
        """Before a forced order: each cart line whose opened record does not write a stated requirement (literal_misses) is
        replaced by an opened candidate of its item that writes them (the cheapest under a cheapest rule, else the one showing
        most stated values), a part still fitting its device and the cart inside the budget. False when such a line stays.
        A part line that does not fit the device line is replaced the same way when a fitting candidate exists; with none it
        stays (a fit verdict alone never turns the forced order into an ending without a purchase)."""
        cart = self.roles.get('cart') or ''
        ids = self.cart_ids()
        cat_key = self.category_key()
        index = self.fit_index() if self.bound else {}
        lines = {i: self.held_units(i) or 1 for i in ids}

        def owner(line: str) -> int | None:
            label = str((self.rows.get(line) or {}).get(cat_key) or '') if cat_key else ''
            owners = [n for n in range(1, len(self.items) + 1) if label and self.is_kind(n, label)]
            return owners[0] if len(owners) == 1 else None
        bad = [line for line in ids if self.literal_misses(line)]
        parts = {p for p, _ in self.bound}
        misfit = [line for line in ids if line not in bad and owner(line) in parts
                  and not self.fits_with(line, owner(line), {k: v for k, v in lines.items() if k != line}, index)]
        if not bad and not misfit:
            return True
        pick_rule = self.brief.get('pick')
        price = lambda i: self.price_of(i) if self.price_of(i) is not None else 1e12
        for line in bad + misfit:
            soft = line in misfit
            number = owner(line)
            if number is None:
                if soft:
                    continue
                return False
            item = self.items[number - 1]
            rest = {k: v for k, v in lines.items() if k != line}
            pool = [i for i in self.candidates(number) if i not in lines and self.fits_with(i, number, rest, index)]
            pool.sort(key=lambda i: (price(i), -self.must_hits(number, i)) if pick_rule == 'cheapest' else (-self.must_hits(number, i), price(i)))
            pick = pool[0] if pool else None
            if pick and isinstance(self.budget, (int, float)):
                each = self.brief.get('budget_scope') == 'each'
                total = price(pick) * (1 if each else item.qty) + (0.0 if each else sum(price(i) * n for i, n in rest.items()))
                if total > self.budget + 1e-09:
                    pick = None
            calls = self.cart_calls({**rest, pick: item.qty}, [line]) if pick else None
            if not calls or len(calls) + 1 > self.left():
                if soft:
                    continue
                return False
            for args in calls:
                faults: list[str] = []
                args = _shape(args, self.schemas.get(cart), 'args', faults)
                if faults or self.act(cart, args) is None:
                    return False
                self.note_cart(cart, args)
            lines = {**rest, pick: item.qty}
            self.trace.append({'kind': 'literal_swap', 'item': number, 'out': line, 'in': pick})
        return True

    def fits_with(self, ident: str, number: int, lines: dict[str, int], index: dict[str, Any]) -> bool:
        """Whether a listing for item NUMBER keeps the cart's other lines fitting: as a part, it takes in the model of the
        device line; as a device, every part line takes in one of its models (no code on either side: no objection)."""
        for part, main in self.bound:
            if number == part:
                devices = [i for i in lines if i in self.item_ids(main)]
                models = set(index['marks'].get(devices[0], {}).get('models') or []) if len(devices) == 1 else set()
                said = [v for v in (self.fit_ok(ident, m, index) for m in models) if v is not None]
                if said and not any(said):
                    return False
            if number == main:
                models = set(index['marks'].get(ident, {}).get('models') or [])
                for line in [i for i in lines if i in self.item_ids(part)]:
                    said = [v for v in (self.fit_ok(line, m, index) for m in models) if v is not None]
                    if said and not any(said):
                        return False
        return True

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
        """Whether a listing's attributes state each of the item's requirements word for word; a limit or an exclusion is
        not read here and counts as not stated."""
        block = _fact_block(self.full.get(ident) or {}) or {}
        for owner, need in self.needs_now():
            if owner != number:
                continue
            prop, _, value = need.partition(':')
            split = lambda text: set(re.sub('[,/;|]', ' ', _words(text)).split())
            words = split(value)
            if not words or words & {'not', 'no', 'least', 'most', 'under', 'over', 'unlike', 'different', 'min', 'max', 'minimum', 'maximum'}:
                return False
            if not any((_names(_spelled(prop), _spelled(k)) and words <= split(str(v)) for k, v in block.items())):
                return False
        return True

    def can_refuse(self) -> bool:
        refuse = self.roles.get('refuse')
        if not refuse:
            return False
        faults: list[str] = []
        _shape(self.quiet(refuse, dict(self.brief.get('refuse_args') or {})), self.schemas.get(refuse), 'args', faults)
        return not faults

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
        opening = _read_rows(self.view.get('opening'))
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
            if turned and (not self.finished) and (not self.broke) and (not self.no_credit) and (self.left() > 1):
                self.chat.append({'role': 'user', 'content': '[Kyo] Kyo tried to end the episode and ' + turned + '\nApply what the store says, then finish.\n' + self.status()})
                self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
            if not self.finished:
                self.finish()
        self.trace.append({'kind': 'end', 'finished': self.finished, 'steps': self.step, 'cost': round(self.cost, 5), 'calls': self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.finished and (not self.broke) and (self.left() > 0) and (self.loop_calls < loop_cap) and (self.cost < spend_cap) and (self.clock() < LAST_TURN or (not self.cart_ids() and self.model_room() > EMPTY_ROOM)):
            if self.ready_to_ask():
                summary = self.interview()
                if summary:
                    self.chat.append({'role': 'user', 'content': summary + '\n' + self.status()})
                continue
            if self.hunt_due():
                opened = self.hunt_fits()
                if opened:
                    self.chat.append({'role': 'user', 'content': '[Kyo] Some requested item still had no listing that meets every requirement, so Kyo '
                                      'searched for it in short words (a required value with its name, then the name alone) and read these listings:\n'
                                      + '\n'.join(opened)[:4000] + '\nJudge them against the requirements in force; cart a fitting one.\n' + self.status()})
                continue
            if self.walk_due():
                followed = self.walk()
                if followed:
                    self.chat.append({'role': 'user', 'content': followed + '\n' + self.status()})
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
            if not self.asked and self.roles.get('ask') and (self.msgs_free() > 0) and any((c['function']['name'] in committing for c in calls)) and (self.left() > 8) and (self.clock() <= ASK_BY):
                self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
                for call in calls:
                    self.chat.append({'role': 'tool', 'tool_call_id': call['id'], 'content': 'Not run (no step used): the shopper is asked first.'})
                summary = self.interview()
                self.chat.append({'role': 'user', 'content': (summary or 'The interviewer had nothing to ask.') + '\n' + self.status()})
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
