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
SPEND_SOFT = 0.048
SPEND_HARD = 0.06
HEAVY_ROOM = 0.02
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
WRAP_AT = 230
SHOP_KEYS = 'seller|shop|store|vendor|brand|maker'
PLAN_SIZE = 6
CALL_STOP = 255
LATE_TIMEOUT = 20
SHOW_CHARS = 450
LOOSE_CHARS = 4000
NOTE_CHARS = 600
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3
AUTO_PAGES = 6
CHEAPER_OPENS = 3
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
_late = ProxyClient(timeout=LATE_TIMEOUT, max_retries=1, retry_delay=1)
_env_late = ProxyClient(timeout=LATE_TIMEOUT + 5, max_retries=1, retry_delay=1)
GUIDE = 'You are Sable, a careful buyer working for one shopper in an online store. You can act only by calling the store\'s tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.\n\nHOW THE STORE WORKS\n- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool\'s description and schema and use the exact argument names and allowed values.\n- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query or may need every word to appear on a listing, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.\n- Search rows are summaries. Stores split a listing\'s record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Sable knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.\n- Questions to the shopper are handled for you by Sable\'s interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.\n- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Sable carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store\'s tools (product questions go through Sable\'s interviewer).\n\nWHAT COUNTS AS A FIT\n- The kind of product is decided by the listing\'s category label: only the label that matches the requested item as a whole counts, never a neighbouring one.\n- A requirement on a property is met when the listing\'s attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value\'s words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.\n- An attribute whose key names the requirement\'s property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing about the requirement. Sable adds [evidence] lines after opened listings with the keys whose names match a requirement.\n- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts only as weaker evidence: prefer a listing whose own attribute for that property states it, look for one (another search, a further page, more listings opened) while steps allow, and settle on the weaker one when none turns up. With a lowest-price pick, a cheaper listing whose title or description states a plain value the request itself asks for beats a dearer one that shows it as an attribute. A requirement marked as a limit that allows only some named values is met by a listing that states one of them (in that attribute, or in the title or description when the attribute is missing); a listing that says nothing on that property does not meet it. A limit that rules a value out is broken only by a listing stating that value. A number bound, and a value such as nil, none or no, must be stated like any other value, and only that same word meets nil, none or no. A number of another kind never counts (a weight is not a capacity), nor a bare figure whose text does not say what it measures.\n- When the category label itself states a requirement\'s value (a season or audience in the label), that requirement is met by the category.\n- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing\'s value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.\n- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.\n- When one requested item must fit another (an accessory for a device), choose the two together from the listings\' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.\n- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.\n- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.\n\nCHOOSING\n- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Sable reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.\n- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.\n- For a cheapest, most or least pick, when the category name joins several kinds of product ("X and Y", "X, Y & Z"), search each kind by its own name as well: a keyword search for one kind ranks the other kinds low.\n- Otherwise any listing that meets everything is right: do not waste steps hunting for something better, but open candidates from the cheapest up, and when several listings you have opened meet everything, choose the lowest-priced of them (a cheaper choice also survives a cut in the budget or a rise in price).\n- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.\n- A request for several different items needs one cart line per item, each at its requested quantity.\n\nCHANGES WHILE YOU WORK\n- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool\'s result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.\n- Re-read a chosen listing\'s record shortly before checking out, because price and stock can change silently.\n\nFINISHING\n- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.\n- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request\'s wording for that ending, and when a remark is needed, keep it short and name no listing.\n- Every reply must call at least one tool. Keep your text short: what you know, then the next move.'
BRIEF = 'Turn a customer\'s shopping request, plus the list of shop tools, into a task sheet for a buying assistant. Reply with exactly one compact JSON object (single line, no other text) using the keys below; the order of keys does not matter.\n\n"roles": an object mapping each job to the exact tool that performs it, null where no tool does: "search" (looks up listings), "open" (shows one listing\'s specifications or attributes; if two single-listing tools exist, take the one whose description mentions specifications, attributes or properties), "offer" (shows one listing\'s current price and availability, when that is a separate tool from "open"), "compare" (shows several listings in full), "ask" (messages the customer), "cart" (changes the cart), "cart_view" (displays the cart), "buy" (places the order), "refuse" (ends the visit with no purchase).\n"buy_args": fixed arguments that make the "buy" tool place the order (the mode value, for a tool with a mode argument), else {}.\n"refuse_args": fixed arguments that make the "refuse" tool end with no purchase, else {}.\n"items": one entry for every distinct product to be bought. Each entry is {"name": what kind of product it is, in the request\'s wording, "shelf": the shop\'s department label for it if some tool lists labels (copied exactly), else "", "quantity": how many units of that single listing (an integer; 1 if unstated; a pair or a set sold as one listing is 1), "musts": the firm requirements on that product, each written "property: constraint" with the precise value, bound, exclusion or closed list}. A property stated beside the product, such as its style or type, is a must too, even when the name keeps it. Copy each stated value in the wording of the request: only explicit comparison words make a bound, and a value that reads as a band label stays as written. Name each property the way the request names it ("fit for a garden party" becomes "occasion: garden party", "a 30-litre tank" becomes "tank capacity: 30 litre") and never substitute a neighbouring property. When a must takes its value from another listing, keep it as a single must containing that listing\'s entire description and, word for word, how the request ties the two values (the same as, unlike, goes with, pairs with), since each asks for a different value ("lid colour: the same as the lid colour of the teapot that Marvelo sells with a bamboo handle"); the words describing that other listing (its seller, its kind, a trailing "with <property> <value> ...") stay inside that must and are never musts of this product. Price, budget, stock and quantity never go into musts.\nHow to tell musts from context: background in a request (the eventual user, the spot it will occupy, an earlier disappointment, the reason for buying) is context unless it states a property\'s value or bound outright, and words forming part of how the product is called (a season or an audience attached to the product word) go into "name", never into musts. A use or occasion that the request makes a property of the product itself ("trousers made for hiking", "a jacket intended for skiing", "its use has to be travel") is a must on that property ("use: hiking"), unless the shop\'s department label already says it; the event, person or setting it is bought for ("for my sister\'s graduation", "as a present for a neighbour") and how it should feel, sit or look in use remain context. A phrase joining products with "and" under a single quantity (often one shop department\'s name) is ONE item; only separate quantities or separate requirements per product make separate items. A sentence saying that some property of a product matters, or that the product has to suit something the listings cannot show (the rest of a room, a person\'s taste, who is going to use it), or that the customer is wary or unsure about some property, without giving that property\'s value or bound, is never a must: put it in "hints".\n"budget": the most the customer will spend in total, as a number in the request\'s own unit, or null.\n"budget_unit": that unit in words (for instance "cents"), or null.\n"budget_scope": "total" if the limit covers the whole order, "each" if it applies per item, null if there is no limit.\n"in_stock_only": true unless the request plainly accepts products that cannot be bought now.\n"pick": "cheapest" if the request wants the lowest price or smallest total, "most" or "least" if it wants the extreme of some measure, otherwise "any". Saying money is tight is not a pick rule. What to do when nothing qualifies is never a pick rule.\n"pick_measure": the measure maximised or minimised for "most" or "least", else "".\n"messages": how many messages the customer lets the assistant send (an integer), or null if not stated.\n"steps": how many steps, moves or tool calls the request allows (an integer), or null.\n"lookups": the musts (copied exactly) whose value must first be read off another listing the request refers to, else []. A rule tying one product\'s property to another requested product, so that it binds only under some condition on that other product, is no lookup: keep it as a single must with its condition in full.\n"limits": the musts (copied exactly) that only narrow a property to a closed set of named values or rule a named value out (the request says to pick it from some values only, or anything but a value), not musts demanding that the product has a stated value and never a number bound (at most, at least, under, over, between), else [].\n"rules": every instruction the request gives on how to work or how to finish, each in 12 words or fewer.\n"wishes": things the customer would merely like.\n"ask_first": things the request explicitly says to confirm with the customer before choosing, else [].\n"hints": one entry per sentence of that kind, {"item": item number, "property": the single property it points at, named the way listings name attributes (colour, material, size, gender, style and the like)}, else [].\n"process": the steps the request demands beyond finding and buying the right products (getting the customer\'s approval, telling or showing them something, confirming something with them, reading something first, using a particular tool), each {"step": what has to be done, in 15 words or fewer, "when": "choose" (before products are chosen), "order" (just before checkout) or "end" (just before ending with no purchase), "messages": how many messages to the customer it needs (0 if none)}; [] when none is asked for. Requirements on the products, the budget, stock, the message and step limits, and what to do when nothing qualifies are not process steps. Nor is a remark that the customer, or someone they mention, wants to know, check or settle something about the product itself (what it is made of, how it was made, how it fits, what it is like): that is a hint about a held-back requirement, so put it in "hints".\n"quiet_decline": true if the request says not to suggest or name alternatives when nothing qualifies, false if it asks for them, null if it says nothing.\n"fits": {"item": item number, "with": item number} for each requested product that must work with, fit or suit ANOTHER requested product (whichever one the assistant chooses), else [].'
PLAN = 'Pick the questions a buying assistant will put to its customer. The customer holds back a few firm wishes, each about a single property of a single requested product, and a question brings one out only when it names that very property. A property the customer has no view on gets a shrug ("don\'t mind"), and one they never think about gets a confused reply, so every property a question names spends part of a message.\n\nInputs: the request; every requested item together with what the request already fixes for it; and, per item\'s department, a numbered list of the attribute keys found on its opened listings, with how many of those listings carry each key and some of its values. A held-back wish may be about a property that none of the listed keys names.\n\nWork through these in order:\n0. Whatever the request itself says to confirm with the customer (shown under ASK FIRST when there is any), each named by the property it concerns.\n1. Properties the story hints at. Requests often allude to a held-back wish without giving its value: a line about something the customer cannot judge or leaves to the assistant, a person whose taste is the deciding one, how the thing looks or where it will go, who is going to use or wear it, or what it holds. For every such line, name the single property it points to and the item it concerns, in story order. Where a listed key of the item matches the hint, use that key (who will wear it: the gender or audience key; how it looks: the colour, pattern or finish key carried by most of its listings). A hinted property that none of the item\'s listings carry should come after the three keys most of that item\'s listings carry.\n2. After those, the listed keys: first the properties shoppers for that kind of product usually choose on (commonly type or style, form or fit, size, capacity or dimensions, material, colour or pattern, as suits the product), weighted by how many of the item\'s listings carry the key. Once five or more listings have been read, drop keys seen on just one or two of them.\nSkip a property the request already fixes for that item (some requirement names it; but a key whose values just repeat something required, say a pattern key that shows the requested decoration, still counts as open), a second name for a property already picked, keys whose every value bundles several buying options, identifiers, the product\'s own model names, and warranty or other logistics. What a part or accessory fits (the make, model or year it is meant for) is a property, and so are pack sizes, counts and quantities per pack. A key tagged [not the stated ...] names some other property than that requirement even if the two names share words. Every item gets its own ranking, story hints ahead of listed keys.\nFor "label" write the property name plainly. When it comes from a listed key, keep all of the key\'s words, spelled out ("lid colour" from lidColour, "rim finish" from rim_finish), never trimmed to "colour" or "finish"; a label holds the name alone, without example values, brackets or a second property.\nAt most COUNT picks per item, strongest first. Your whole reply is one JSON object shaped like {"ask": [{"item": <its number>, "label": "<property name>", "hint": "<the story words behind it, eight at most, or empty>"}]}, keeping each item\'s picks in ranked order.'
WANTS = 'A shopping assistant asked a shopper about a product. Read the shopper\'s reply and return ONE JSON object and nothing else: {"wants": true or false}\n- true when the reply names a value or limit the shopper wants for some property of the product, whatever property it names and however folksy the wording.\n- false when it only says they have no preference or do not mind, says they did not follow the message, or remarks on something other than the product itself (their plans, their day, their home life).'
HEAR = 'A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "budget": new total limit as a number or null}\n- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3"). An answer that names a property together with what it must be is always a need, however it is phrased and whichever properties the question listed; it is never an answer that says nothing. The constraint is the value alone: words that tie the property to its value or press the point stay out ("lid colour kept to teal, mind" gives "lid colour: teal").\n- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop\'s product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer\'s own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.\n- "free": the property of each question the shopper said they do not mind about.\n- Folksy or roundabout wording that names a value still states that value.\n- An answer that rejects every value except one is asking for that one value: write the need as that value on its own, with no negation. A need reads \"not X\" only when the answer rejects X itself and accepts the rest.'
NOTE = 'While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:\n{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request\'s unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}\nA replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note.'
CHECK = 'You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.\nReading rules: a requirement is met when the listing\'s attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value\'s words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when the listing has no attribute for that property, a requirement from the request that names a plain value is met when the title or description states that value; any other requirement naming a value the listing must have (one marked heard from the shopper, a number bound such as at most 40 cm, and a value such as nil, none or no, which only that same word meets, included) is unconfirmed even if the title or description mentions it: report it as a problem, and name in "better" any listing shown whose attribute for that property states the value and that meets every other requirement; a requirement marked as a limit that allows only some named values is met when the listing states one of them, in its attribute for that property or, when it has no such attribute, in its title or description, and a listing that states no value for that property leaves it unconfirmed; a limit that rules a value out is broken only by a stated value it rules out (a number of another kind never counts: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing\'s value counts as met unless the records show that same value; a requirement bound to a condition on another requested product applies only when that product\'s cart line meets the condition. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.\nWhen the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart\'s choice on that rule, the cart is wrong.\nReturn ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses.'

AUDIT = 'A buying assistant is about to finish without buying, believing nothing qualifies. Test the listing records below against the requirements, judging each requirement this way: decide a property requirement from the listing\'s specification whose key names that same property (key spelling is irrelevant, a list qualifies through any one member, synonyms and converted units are accepted, a longer value containing the requested wording qualifies, as "satin brushed nickel" does for brushed nickel, and bounds are inclusive); a key about a different property, however related (a strap colour is not a dial colour), settles nothing; only when no key covers the property may the value come from another key or from the title or blurb (a figure that does not say what it measures counts for nothing); a requirement the department name itself states is satisfied by that department; the department must name the requested product; a rule of being unlike another listing\'s value holds unless the records show that very value; a requirement bound to a condition on another requested product applies only when that condition is met; sold-out listings and listings that would break the budget do not qualify.\nA record tagged ROW is only a search preview: its full page (the specifications) has not been read, so it can neither pass nor fail on a property the preview does not show.\nAnswer with exactly one JSON object and nothing more: {"fits": {"1": [ids of listings that seem to meet every requirement of item 1, cheapest first, no more than 4], "2": [the same for item 2], ...} with one key per requested item, written as its item number from the requirements list}, "open": [ids of ROW listings of a requested kind and within budget whose full page might show that they qualify, most promising first, no more than 6], "nearest": {"1": {"id": the in-stock listing within budget whose full page was read and that falls short of item 1\'s requirements on the fewest counts (its first fit, when it has one), "short": [each requirement of item 1 it misses or that no record confirms; [] for a fit], "contradicted": true when a record states a value that breaks one of those requirements, false when they are only unconfirmed}, "2": {...}, ...} with one key per requested item, "reason": "one sentence"}'
UNMIX = 'A shopper\'s request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request\'s wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}'
TRAIL = 'Checkout is next. Certain requirements do not state their value; they name another listing whose page holds it, and that page can in turn hand the reader on to a further listing. The order is valid only once every listing along such a route has been opened in full. For each such requirement, walk its route from the requirement\'s own wording, hop by hop, using only what each page itself says, and the records below (OPENED: its full page was read; SPECS: only its attribute view was read, so its page still counts as unopened; ROW: only its search preview was seen).\nReply with exactly one JSON object and no extra text:\n{"chains": [{"need": that requirement word for word, "value": the property\'s value as printed on the final hop\'s page, or "", "done": true only if the final hop is an OPENED page that prints the value itself (this holds even when the requirement asks for something other than or unlike that value, since it is measured against it), "links": [{"about": the description of this hop given by the requirement or by the page before it, "id": the record\'s listing id if a record below is this hop, else "", "find": "for a hop still unlocated: a handful of search words, namely its seller and the values its description gives"}]}]}\nCount a record as a hop only when it matches that hop\'s full description: seller or brand, product kind, and every value mentioned (a description of "a multipack of 10 pieces" is not met by the same seller\'s 10-capsule single pack). Where no record matches everything, the hop stays unlocated even if the seller has other products of that kind. A hop described as the product chosen or bought in this order ("the lamp chosen for this order") is the CART listing of that kind: give its id.\nList the hops of each route in order and stop at the first one that is unlocated or whose page is not OPENED, because its contents are not known yet.'
GATE = 'A shopping agent is about to STAGE. The shopper\'s request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product\'s properties do not do a step.\nWhen a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.\nReturn ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}'
SWAP = 'One cart line can no longer be bought: its listing has sold out. Look through the other records below for listings that could take its place, judging each requirement this way: decide a property requirement from the listing\'s specification whose key names that same property (key spelling is irrelevant, a list qualifies through any one member, synonyms and converted units are accepted, a longer value containing the requested wording qualifies, as "satin brushed nickel" does for brushed nickel, and bounds are inclusive); a key about a different property, however related (a strap colour is not a dial colour), settles nothing; only when no key covers the property may the value come from another key or from the title or blurb (a figure that does not say what it measures counts for nothing); a requirement the department name itself states is satisfied by that department; the department must name the requested product; a rule of being unlike another listing\'s value holds unless the records show that very value; a requirement bound to a condition on another requested product applies only when that condition is met; sold-out listings and listings that would break the budget do not qualify.\nReply with exactly one JSON object and no extra text: {"reason": "one sentence", "fits": [{"price": the price as a number, exactly as its record gives it, "id": listing id} for each listing that seems to satisfy every requirement on the sold-out line\'s product, at most 6]}'
CHASE = 'One requirement on a product does not state its value: it names another listing whose page holds the value, and that page may in turn send the reader on to a further listing. You get the requirement and, once a listing on the route has been read, that listing\'s record. Reply with exactly one JSON object and no other text:\n{"final": true when the RECORD is the last stop for the requirement\'s property, else false, "value": when final, the value the record gives for that property, else "". When the record pairs its own value of that property with another value (a sentence saying it goes together with, goes with or pairs with another value, or a store rule listing such pairs), the value it gives is the one paired with its own value, unless the requirement explicitly asks for the same, identical or very same value, which is the record\'s own value. Write the value meant first, then the other one ("navy, which this record pairs with its own value, teal" or "teal, the record\'s own value, which it pairs with navy"); with no such sentence, give its attribute or its description\'s value, "shop": when not final, the seller, shop or brand of the listing to read next, exactly as written, "kind": when not final, that listing\'s kind of product or department, as written, "values": when not final, every attribute value the text gives to recognise that listing, each copied as written together with its property word when the text gives one ("lace-up closure", "plain pattern"), "terms": when not final, the same values without their property words ("lace-up", "plain")}\nRules:\n- With no record yet, the requirement itself names the first listing: describe that one (final false). Once a RECORD is shown, it already stands on the route from that listing (it is that listing or one the route sent the reader to, even when its shop or kind differs from the requirement\'s), so never describe the requirement\'s listing again.\n- A record sends the reader on only when it says, for the requirement\'s property, to refer to, look to, go by, use or take the value from a different product, and names that product by its shop, kind or values. Then describe that product (final false), whatever the record says about itself.\n- Otherwise the record is the last stop (final true). A sentence saying what its value goes together with, pairs with or is compatible with belongs to the value and is never a pointer.\n- A pointer about some other property is ignored. "The other" listing of a shop is a second listing of that shop and kind, never the record being read.'
SIFT = 'One product of a shopping request is described below, with records of its lowest-priced listings. Decide which of them qualify, judging each requirement this way: ' + SWAP.split('judging each requirement this way: ', 1)[1].split('\nReply with', 1)[0] + '\nReply with exactly one JSON object and no extra text: {"reason": "one sentence", "fits": [{"price": the price as a number, exactly as its record gives it, "id": listing id} for each listing that seems to satisfy every requirement on this product, lowest price first, at most 6]}'
PICK = 'A requirement points at one listing of a shop, described by a few traits. Below are listings of that shop with their records. Decide which of them is the listing described. Reply with exactly one JSON object and no other text: {"id": the id of that listing exactly as written, or "" when none of them fits the description, "all": true when its record shows every trait of the description, false when it shows only some}. A trait counts when the record states it anywhere (attributes, title, description or remarks), in any wording or unit ("2 m" and "2 meters"). A different value for one of the traits rules a listing out (a 1 m cable is not a 2 m one), and so does a record saying it is not the one described or that the described version is found elsewhere ("not the 2 m one", "it is not red", "a wired version is sold elsewhere"); a trait the record says nothing about leaves "all" false.'
SHAPE = 'A store tool returned the result below. Name its parts by the exact keys the result and the tool\'s parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing\'s price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}'
GUARD_KEYS = 3
GUARD_SHARE = 0.0
GUARD_GAP = 0.34
NOT_PROPERTY = '\\b(?:options?|variants?|sku|ids?|codes?|asin|upc|ean|barcodes?|urls?|links?|images?|photos?)\\b'
LOGISTICS = '\\b(?:warrant\\w*|guarantee\\w*|deliver\\w*|shipping|shipment|dispatch\\w*|returns?|refunds?|sellers?|vendors?|merchants?|stock|availability|dangerous goods)\\b'
CATCH_ALL = '^(?!.*\\b(?:types?|styles?|kinds?|shapes?)\\b).*\\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\\w*|accessor\\w*|extras?|misc\\w*)\\b'
NUMERIC_VALUE = '^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$'
BOUND_WORDS = '\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b'
NUMBER_TEXT = '\\s*-?\\d+(\\.\\d+)?\\s*'
PLAIN_NUMBER = '\\s*\\d+(\\.\\d+)?\\s*'
MODE_ASK = '\nAnswer with a single JSON object and no other text: {"buy_args": the fixed arguments that make it place the order, "refuse_args": the fixed arguments that make it end with no purchase} (only the arguments that select the mode).'
QUERY_FIELD = '(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword'
PAGE_WORDS = '(?:^|_)(?:page|pg|leaf|sheet)(?:s?_|s?$)'
OFFSET_WORDS = '(?:^|_)(?:offset|start|begin|skip|from|after|row|rows)(?:_|$)'
VALUE_SKIP = '\\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b'
BOUND_HINT = '\\b(?:at most|at least|under|over|below|above|less|fewer|more|greater|within|up to)\\b'
BOUND_CUT = '\\b(?:no|not|at most|at least|under|over|below|above|less|fewer|more|greater|within|up to|than|or|and)\\b'
UNIT_VALUE = '\\d[\\d.,/]*\\s*[a-z%"\'][a-z%"\'. -]*'
WIDEN_EACH = 6
FIT_CODE = '[A-Z][A-Za-z]{0,5}[ -]?\\d{3,6}'
PART_CODE = '[A-Z]{1,4}\\d*-?\\d{3,6}'
LINK_STEPS = 8
SHOP_PAGES = 2
SWEEP_UNTIL = 130
SWEEP_GAP_UNTIL = 100
SWEEP_PAGES = 4
SWEEP_OPEN = 12
SWEEP_KEEP = 10
SWEEP_RUNS = 2
SPOT_ROUNDS = 5
CODE_WORD = '\\b' + FIT_CODE + '\\b'
OWN_PART = '[\\[(]\\s*(?:part\\s+|(?![A-Z]+-))(' + PART_CODE + ')\\s*[\\])]'
MODEL_NO = '\\bmodel\\s*(?:no\\.?|number|#)?\\s*[:.]?\\s*(' + FIT_CODE + ')\\b'
ALIASES = '\\b(?:also sold as|also known as|sold also as)\\s+([^.;]+)'
WORKS_WITH = '\\b(?<!not )(?<!never )(?:works with|compatible with|fits|goes with)\\s*:?\\s+([^.;]+)'
ALSO_TAIL = '\\bpart\\s+' + PART_CODE + '\\s+also\\s+$'
ALSO_FITS = '\\bpart\\s+(' + PART_CODE + ')\\s+also (?:fits|goes with)\\s*:?\\s*([^.;]+)'
EXCEPT_WORDS = '\\b(?:except|excluding|but not|other than|apart from|save)\\b'
FIT_FAMILY = '(?<![A-Za-z0-9-])([A-Z]{2,6}[ -]?[0-9]*)(x{1,4}|[*#?]{1,4})(?![A-Za-z0-9])'
SEE_PART = '\\b(?:more|details|see|read)\\s+(?:on|at|under|about)?\\s*(?:the\\s+)?part\\s+(' + PART_CODE + ')'
EVIDENCE_HEAD = "[evidence: keys whose names include a requirement's property; a key that names some other property (a different part or aspect) settles nothing, while a key under another name can still carry the same property] "
SERIES_CODE = '[A-Z]{1,4}\\d{0,3}'
SERIES_LIST = '(' + SERIES_CODE + '(?:\\s*(?:,|/|or|and)\\s*' + SERIES_CODE + ')*)(?![A-Za-z0-9-])'
PART_SERIES = '(?i:\\b(?:parts?|spares?|accessor(?:y|ies)|attachments?|add-?ons?)\\s+(?:from\\s+|of\\s+|in\\s+)?(?:the\\s+)?series|\\b(?:compatible|matching|supported)\\s+series)\\s*:?\\s*' + SERIES_LIST
SERIES_PARTS = '(?i:\\bseries)\\s+' + SERIES_LIST + '\\s+(?i:parts?|spares?|accessor(?:y|ies)|attachments?|add-?ons?)\\b'
FIT_HEAD = "[fit plan: a part suits a device when the part's text gives the device's model code or another name the device is sold under, or when the device's page names a part series that starts the part's own code; a model code that sits only in descriptions may not be searchable, so when a search for one comes back empty, list the device's department and read the codes off its rows] "

def _codes(text: Any) -> list[str]:
    return [' '.join(code.split()).lower() for code in re.findall(CODE_WORD, str(text or ''))]

def _kin(own: Any) -> str:
    hit = re.fullmatch('(' + SERIES_CODE + ')-\\d{3,6}', str(own or ''))
    return hit.group(1) if hit else ''

def _fit_marks(text: Any) -> dict[str, Any]:
    text = str(text or '')
    series: list[str] = []
    lent: list[tuple[str, list[str]]] = []
    for line in re.split('(?<=[.;!])\\s+|\\n+', text):
        for hit in list(re.finditer(PART_SERIES, line)) + list(re.finditer(SERIES_PARTS, line)):
            found = re.findall(SERIES_CODE + '(?![A-Za-z0-9-])', hit.group(1))
            head = line[:hit.start()]
            model = re.search(MODEL_NO, head, re.I)
            if model and re.search('\\balso\\b', head, re.I):
                lent.append((' '.join(model.group(1).split()).lower(), found))
            else:
                series += found
    own = re.search(OWN_PART, text, re.I)
    named = [c for hit in re.finditer(MODEL_NO, text, re.I) for c in _codes(hit.group(1))]
    named += [c for hit in re.finditer(ALIASES, text, re.I) for c in _codes(hit.group(1))]
    works = [c for hit in re.finditer(WORKS_WITH, text, re.I) if not re.search(ALSO_TAIL, text[max(0, hit.start() - 40):hit.start()], re.I) for c in _codes(re.split(EXCEPT_WORDS, hit.group(1), maxsplit=1, flags=re.I)[0])]
    also = [(hit.group(1).upper(), _codes(hit.group(2))) for hit in re.finditer(ALSO_FITS, text, re.I)]
    more = re.search(SEE_PART, text, re.I)
    wild: list[tuple[str, int]] = []
    skip: list[str] = []
    for hit in re.finditer(WORKS_WITH, text, re.I):
        parts = re.split(EXCEPT_WORDS, hit.group(1), maxsplit=1, flags=re.I)
        found = [(' '.join(m.group(1).split()).lower(), 0 if m.group(2)[0] == '*' else len(m.group(2))) for m in re.finditer(FIT_FAMILY, parts[0])]
        wild += found
        if found and len(parts) > 1:
            skip += _codes(parts[1])
    return {'own': own.group(1).upper() if own else '', 'codes': named, 'works': works, 'also': also, 'more': more.group(1).upper() if more else '', 'wild': wild, 'skip': skip, 'series': list(dict.fromkeys(series)), 'lent': lent}

def _dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        return str(value)

def _payload(seen: Any) -> tuple[Any, list[str]]:
    if isinstance(seen, dict) and len(seen) == 2:
        for key, value in seen.items():
            if isinstance(value, list) and 0 < len(value) <= 4 and all((isinstance(v, str) and len(v.split()) >= 3 for v in value)):
                other = next((v for k, v in seen.items() if k != key))
                if not (isinstance(other, list) and all((isinstance(v, str) for v in other))):
                    return (other, value)
    return (seen, [])

def _merge_twins(record: dict[str, Any]) -> None:
    for name, block in list(record.items()):
        if not isinstance(block, dict) or len(block) < 2:
            continue
        groups: dict[str, list[str]] = {}
        for key in block:
            groups.setdefault(re.sub('[^a-z0-9]+', '', _split_name(key)), []).append(key)
        if all((len(keys) == 1 for keys in groups.values())):
            continue
        merged: dict[str, Any] = {}
        for keys in groups.values():
            keep = next((k for k in keys if ' ' in str(k)), keys[0])
            said: list[str] = []
            for key in keys:
                for text in re.split(',\\s+', str(block[key]).strip()):
                    if text and text.lower() not in [s.lower() for s in said]:
                        said.append(text)
            merged[keep] = block[keys[0]] if len(said) <= 1 else ', '.join(said)
        record[name] = merged

def _trouble(seen: Any) -> Any:
    if not isinstance(seen, dict) or not 1 <= len(seen) <= 2:
        return None
    return next((v for k, v in seen.items() if re.search('error|fail|fault|trouble|problem|invalid|denied', str(k), re.I) and v not in (None, False, '', [], {})), None)

def _split_name(name: Any) -> str:
    return re.sub('(?<=[a-z0-9])(?=[A-Z])', '_', str(name)).lower()

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

def _squash(text: Any) -> str:
    flat = re.sub('[^a-z0-9]+', '', re.sub('\\band\\b|&', ' ', str(text or '').lower()))
    return flat[:-1] if len(flat) > 3 and flat.endswith('s') else flat

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
    declared = schema.get('type')
    if isinstance(declared, str):
        return [declared]
    return [k for k in declared if isinstance(k, str)] if declared else []

KIND_TESTS = {'string': lambda v: isinstance(v, str), 'integer': lambda v: isinstance(v, int) and (not isinstance(v, bool)), 'number': lambda v: isinstance(v, (int, float)) and (not isinstance(v, bool)), 'boolean': lambda v: isinstance(v, bool), 'array': lambda v: isinstance(v, list), 'object': lambda v: isinstance(v, dict), 'null': lambda v: v is None}

def _fits_kind(value: Any, kind: str) -> bool:
    test = KIND_TESTS.get(kind)
    return True if test is None else test(value)

def _coerce(value: Any, kinds: list[str]) -> Any:
    for kind in kinds:
        if kind == 'integer' and isinstance(value, float) and value.is_integer():
            return int(value)
        if kind in ('integer', 'number') and isinstance(value, str) and re.fullmatch(NUMBER_TEXT, value):
            number = float(value)
            return int(number) if kind == 'integer' or number.is_integer() else number
        if kind == 'string' and isinstance(value, (int, float)) and (not isinstance(value, bool)):
            return str(value)
        if kind == 'boolean' and isinstance(value, str) and (value.strip().lower() in ('true', 'false')):
            return value.strip().lower() == 'true'
        if kind == 'array' and (not isinstance(value, list)) and (value is not None):
            return [value]
    return value

def _text_limits(value: str, schema: dict[str, Any], where: str, faults: list[str]) -> str:
    longest, shortest = (schema.get('maxLength'), schema.get('minLength'))
    if isinstance(longest, int) and len(value) > longest:
        value = value[:longest]
    if isinstance(shortest, int) and len(value) < shortest:
        faults.append(f'{where} is too short')
    return value

def _number_limits(value: Any, schema: dict[str, Any], where: str, faults: list[str]) -> Any:
    low, high = (schema.get('minimum'), schema.get('maximum'))
    if isinstance(low, (int, float)) and value < low:
        faults.append(f'{where} is below {low}')
    if isinstance(high, (int, float)) and value > high:
        faults.append(f'{where} is above {high}')
    return value

def _list_limits(value: list[Any], schema: dict[str, Any], where: str, faults: list[str]) -> list[Any]:
    most, least = (schema.get('maxItems'), schema.get('minItems'))
    if isinstance(most, int) and len(value) > most:
        faults.append(f'{where} allows at most {most} entries')
    if isinstance(least, int) and len(value) < least:
        faults.append(f'{where} needs at least {least} entries')
    inner = schema.get('items')
    if isinstance(inner, dict):
        return [_shape(v, inner, f'{where}[{i}]', faults) for i, v in enumerate(value)]
    return value

def _dict_fields(value: dict[str, Any], schema: dict[str, Any], where: str, faults: list[str]) -> dict[str, Any]:
    props = schema.get('properties') if isinstance(schema.get('properties'), dict) else {}
    if schema.get('additionalProperties') is False:
        faults.extend((f'{where}.{name} is not an allowed argument' for name in value if name not in props))
    faults.extend((f'{where}.{name} is required' for name in schema.get('required') or [] if name not in value))
    return {k: _shape(v, props[k], f'{where}.{k}', faults) if k in props else v for k, v in value.items()}

def _shape(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    if not isinstance(schema, dict):
        return value
    kinds = _kinds(schema)
    if kinds and (not any((_fits_kind(value, k) for k in kinds))):
        value = _coerce(value, kinds)
        if not any((_fits_kind(value, k) for k in kinds)):
            faults.append(f"{where} should be {'/'.join(kinds)}")
            return value
    options = schema.get('enum')
    if isinstance(options, list) and value not in options:
        if isinstance(value, str):
            value = next((o for o in options if isinstance(o, str) and o.lower() == value.strip().lower()), value)
        if value not in options:
            faults.append(f'{where}: allowed values are {options}')
    if isinstance(value, str):
        return _text_limits(value, schema, where, faults)
    if isinstance(value, (int, float)) and (not isinstance(value, bool)):
        return _number_limits(value, schema, where, faults)
    if isinstance(value, list):
        return _list_limits(value, schema, where, faults)
    if isinstance(value, dict):
        return _dict_fields(value, schema, where, faults)
    return value

def _bound_value(value: str) -> str:
    if not re.search(BOUND_HINT, value, re.I):
        return value
    rest = ' '.join(re.sub(BOUND_CUT, ' ', value, flags=re.I).split())
    return rest if re.fullmatch(UNIT_VALUE, rest, re.I) else value

def _unglue(text: str) -> str:
    return re.sub('(?<=[0-9])(?=[a-z])', ' ', text)

DENIED = '(?:\\b(?:not|isn\'t|aren\'t|never)\\b|\\b(?:look|refer)s? to\\b|\\bgo(?:es)? by\\b|\\b(?:check|see|use)s? (?:the|a|an|another|its)\\b|\\btaken? (?:it |its [a-z ]{1,30})?from\\b|\\bcomes? from\\b)[^.;!?"\\n|{}\\[\\]]*|[^.;:!?"\\n|{}\\[\\]]*\\bexists? elsewhere\\b[^.;:!?"\\n|{}\\[\\]]*'

def _kept(text: str) -> str:
    return re.sub(DENIED, ' ', text)

def _unit_clash(num: str, unit: str, text: str) -> bool:
    after = re.findall('(?<![a-z0-9.])' + re.escape(num) + '(?![0-9])\\s*([a-z]*)', text)
    return bool(after) and all((a and len(a) <= 3 and a not in ('and', 'the', 'for', 'with', 'or', 'of', 'to', 'an', 'at', 'by', 'on', 'is', 'per', 'but', 'nor', 'so', 'are', 'was') and a != unit and a.rstrip('s') != unit for a in after))


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

def _money_like(value: str) -> bool:
    return bool(re.fullmatch('[$\u20ac\u00a3\u00a5]\\s*[\\d.,]+|[\\d.,]+\\s*(?:usd|eur|gbp|cents?|dollars?)|\\d+[.,]\\d{1,2}', value.strip(), re.I))

def _ref_name(key: Any) -> bool:
    return 'id' in str(key).lower() or bool(re.search('(?:^|[^a-z0-9])(?:ids?|refs?|sku)(?:$|[^a-z0-9])', _split_name(key).lower()))

def _learn_id(rows: list[dict[str, Any]]) -> str | None:
    best = None
    for key in rows[0]:
        values = [row.get(key) for row in rows]
        if not all((_id_like(v, 'id' in str(key).lower()) for v in values)):
            continue
        values = [v.strip() for v in values]
        if len(rows) > 1 and len(set(values)) < len(values):
            continue
        score = (_ref_name(key), not any((_money_like(v) for v in values)), sum((len(v) for v in values)) / len(values))
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

READ_KEY = "[A-Za-z_][A-Za-z0-9_ ./#&'()%-]{0,40}?"
READ_MARK = '[:=]'
TAG_NODE = re.compile('<([A-Za-z][\\w:.-]*)((?:\\s+[\\w:.-]+\\s*=\\s*(?:"[^"]*"|\'[^\']*\'))*)\\s*(?:/>|>(.*?)</\\1\\s*>)', re.S)
ID_TOKEN = re.compile('(?<![\\w-])(?=[\\w-]*\\d)(?=[\\w-]*[A-Za-z])[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+(?![\\w-])|(?<![\\w-])[A-Z]{2,}\\d{3,}(?![\\w-])')


def _unmark(text: str) -> str:
    text = re.sub('\\[([^\\]]*)\\]\\([^)]*\\)', '\\1', text)
    text = re.sub('\\*\\*|__|`', '', text)
    for code, plain in (('&amp;', '&'), ('&quot;', '"'), ('&#39;', "'"), ('&apos;', "'"), ('&lt;', '<'), ('&gt;', '>'), ('&nbsp;', ' ')):
        text = text.replace(code, plain)
    return text


def _typed(key: str, text: Any) -> Any:
    if not isinstance(text, str):
        return text
    value = text.strip()
    low = value.lower()
    if low in ('true', 'false'):
        return low == 'true'
    if low in ('none', 'null', ''):
        return None
    if re.fullmatch('-?(?:0|[1-9]\\d*)(?:\\.\\d+)?', value) and (not re.search('id|code|sku|ref', str(key), re.I)):
        return float(value) if '.' in value else int(value)
    return value


def _pairs(text: Any, marks: tuple[tuple[str, int], ...]=(('=', 2), (':', 3)), wide: bool=True) -> dict[str, str] | None:
    if not isinstance(text, str) or len(text) < 5:
        return None
    body = text.strip()
    for mark, need in marks:
        for cut in (';', '\n', '|', ','):
            key = READ_KEY if mark == ':' or (not wide) else '[^=;|\n]{1,60}?'
            parts = re.split('\\s*' + re.escape(cut) + '\\s*(?=' + key + '\\s*' + mark + ')', body)
            if len(parts) < need:
                continue
            pairs = [re.match('(' + key + ')\\s*' + mark + '\\s*(.*)$', part, re.S) for part in parts]
            if not all(pairs):
                continue
            if mark == ':' and any((len(m.group(1).split()) > 4 or len(m.group(2)) > 60 or '. ' in m.group(2) for m in pairs)):
                continue
            return {re.sub('<[^>]+>', '', m.group(1)).strip(): m.group(2).strip() for m in pairs}
    return None


def _fielded(fields: list[tuple[str, str]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    first = fields[0][0] if fields else ''
    for key, value in fields:
        if key in current and key == first:
            records.append(current)
            current = {}
        block = _pairs(value)
        current[key] = block if block else _typed(key, value)
    if current:
        records.append(current)
    return records


def _flat_fields(text: str) -> list[dict[str, Any]]:
    best: list[str] = []
    for cut in ('\\s+\\|\\s+', '\\s*;\\s+', '\\s+[•·]\\s+', '\\t+', ',\\s+'):
        parts = re.split(cut + '(?=' + READ_KEY + '\\s*' + READ_MARK + '(?:\\s|$))', text.strip())
        keyed = [p for p in parts if re.match(READ_KEY + '\\s*' + READ_MARK + '(?:\\s|$)', p)]
        if len(keyed) >= 3:
            best = parts
            break
    keyed = [p for p in best if re.match(READ_KEY + '\\s*' + READ_MARK + '(?:\\s|$)', p)]
    plain = [p for p in keyed if re.match('[a-z0-9_ ]+\\s*' + READ_MARK + '(?:\\s|$)', p)]
    if len(plain) >= 3 and len(plain) >= 0.6 * len(keyed):
        keyed = plain
    if len(keyed) < 3:
        quoted = re.findall('([A-Za-z_][\\w.-]*)=("[^"]*"|\'[^\']*\'|\\S+)', text)
        if len(quoted) < 3 or sum((len(k) + len(v) + 1 for k, v in quoted)) < 0.7 * len(text.strip()):
            return []
        return _fielded([(k, v.strip('"\'')) for k, v in quoted])
    fields = []
    for part in best:
        found = re.match('(' + READ_KEY + ')\\s*' + READ_MARK + '(?:\\s(.*))?$', part, re.S) if part in keyed else None
        if found:
            fields.append((found.group(1).strip(), (found.group(2) or '').strip()))
        elif not fields:
            fields.append(('title', part.strip()))
        else:
            fields[-1] = (fields[-1][0], fields[-1][1] + ' | ' + part.strip())
    return _fielded(fields)


def _cells(line: str, mark: str) -> list[str]:
    if mark == '|':
        return [cell.strip() for cell in line.strip().strip('|').split('|')]
    if mark == '\t':
        return [cell.strip() for cell in line.split('\t')]
    cells: list[str] = []
    cell = ''
    quoted = False
    for char in line:
        if char == '"' and (quoted or not cell.strip()):
            quoted = not quoted
        elif char == mark and (not quoted):
            cells.append(cell.strip())
            cell = ''
        else:
            cell += char
    return cells + [cell.strip()]


def _grid(lines: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    for mark in ('|', '\t', ',', ';'):
        for start in range(min(len(lines), 6)):
            head = _cells(lines[start], mark)
            if len(head) < 3 or not all((re.fullmatch("[A-Za-z_][\\w ./#&'()%-]{0,40}", cell) for cell in head)):
                continue
            rows: list[dict[str, Any]] = []
            for line in lines[start + 1:]:
                if re.fullmatch('[\\s|:+-]{3,}', line):
                    continue
                cells = _cells(line, mark)
                if len(head) - 2 <= len(cells) < len(head):
                    cells += [''] * (len(head) - len(cells))
                if len(cells) != len(head):
                    break
                row: dict[str, Any] = {}
                for key, cell in zip(head, cells):
                    block = _pairs(cell)
                    row[key] = block if block else _typed(key, cell)
                rows.append(row)
            if rows:
                used = start + 1 + len(rows) + sum((1 for line in lines[start + 1:start + 2 + len(rows)] if re.fullmatch('[\\s|:+-]{3,}', line)))
                return (rows, lines[:start] + lines[used:])
    return ([], lines)


def _blocks(lines: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    extra: list[str] = []
    current: dict[str, Any] = {}
    section: dict[str, Any] | None = None
    section_at = -1
    level = 0
    def close() -> None:
        nonlocal current, section, section_at
        if len(current) >= 2:
            records.append(current)
        elif current:
            extra.extend((str(v) for v in current.values()))
        current, section, section_at = ({}, None, -1)
    solid: list[str] = []
    gaps: list[bool] = []
    for at, raw in enumerate(lines):
        if raw.strip():
            gaps.append(not solid or not lines[at - 1].strip())
            solid.append(raw)
    for spot, raw in enumerate(solid):
        if re.fullmatch('\\s*[-=*_]{3,}\\s*', raw):
            close()
            continue
        indent = len(raw) - len(raw.lstrip(' \t'))
        line = _unmark(raw.strip())
        head = re.match('(#{1,6})\\s+(.*)$', line)
        if head:
            depth = len(head.group(1))
            text = head.group(2).strip().rstrip(':')
            if len(current) >= 2 and level and depth > level:
                section = current.setdefault(text, {})
                section_at = -2
                continue
            close()
            level = depth
            current['title'] = text
            continue
        bare = re.sub('^(?:[-*+•]|\\d+[.)])\\s+', '', line)
        pair = re.match('(' + READ_KEY + ')\\s*' + READ_MARK + '(?:\\s+(.*))?$', bare, re.S)
        if pair and (not (pair.group(2) or '').strip()):
            key = pair.group(1).strip()
            after = solid[spot + 1] if spot + 1 < len(solid) else ''
            if section is not None and (indent > section_at or section_at == -2) and len(after) - len(after.lstrip(' \t')) <= indent:
                section[key] = ''
                continue
            ahead = solid[spot + 1:spot + 3]
            if len(ahead) == 2 and re.match('\\s*(?:[-*+•]|\\d+[.)])\\s+\\S', ahead[0]) and len(ahead[1]) - len(ahead[1].lstrip(' \t')) > len(ahead[0]) - len(ahead[0].lstrip(' \t')) and (not re.match('\\s*(?:[-*+•]|\\d+[.)])\\s', ahead[1])):
                close()
                continue
            section = current.setdefault(key, {})
            section_at = indent
            continue
        if pair:
            key, value = (pair.group(1).strip(), pair.group(2).strip())
            if section is not None and (indent > section_at or section_at == -2):
                section[key] = value
                continue
            section, section_at = (None, -1)
            if key in current and len(current) >= 2:
                close()
            block = _pairs(value)
            current[key] = block if block else _typed(key, value)
            continue
        follows = solid[spot + 1] if spot + 1 < len(solid) else ''
        if not pair and gaps[spot] and len(ID_TOKEN.findall(bare)) == 1 and re.match('(?:[-*+\u2022]\\s+)?(' + READ_KEY + ')\\s*' + READ_MARK + '\\s+\\S', _unmark(follows.strip())):
            close()
            current['header'] = bare
            continue
        if current and bare == line and indent == 0 and (not re.match('\\d+[.)]\\s', line)) and ('text' in current or len(current) >= 2):
            current['text'] = (str(current.get('text') or '') + ' ' + bare).strip()
            continue
        if re.match('(?:[-*+•]|\\d+[.)])\\s+', line) and indent == 0:
            close()
            current['title'] = bare
            continue
        if current:
            current['text'] = (str(current.get('text') or '') + ' ' + bare).strip()
        else:
            extra.append(bare)
    close()
    return (records, extra)


def _tag_tree(text: str, depth: int=0) -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    for match in TAG_NODE.finditer(text):
        name, attrs, body = (match.group(1), match.group(2), match.group(3) or '')
        node: Any = dict(re.findall('([\\w:.-]+)\\s*=\\s*["\']([^"\']*)["\']', attrs))
        inner = _tag_tree(body, depth + 1) if depth < 6 and '<' in body else []
        if inner:
            for tag, value in inner:
                if tag in node:
                    node[tag] = (node[tag] if isinstance(node[tag], list) else [node[tag]]) + [value]
                else:
                    node[tag] = value
        else:
            words = _unmark(body).strip()
            if node and words:
                node['text'] = words
            elif not node:
                node = _typed(name, words)
        found.append((name, node))
    return found


def _markup(text: str) -> Any:
    if re.search('<(?:table|tr|td|th|ul|ol|li|dl|dt|dd|div|span|p|br|h[1-6]|b|strong)\\b', text, re.I):
        plain = re.sub('</t[dh]>\\s*<t[dh][^>]*>', ' | ', text, flags=re.I)
        plain = re.sub('</dt>\\s*<dd[^>]*>', ': ', plain, flags=re.I)
        plain = re.sub('<h([1-6])[^>]*>', lambda m: '\n' + '#' * int(m.group(1)) + ' ', plain, flags=re.I)
        plain = re.sub('<(?:/tr|/li|/p|/div|/dd|/h[1-6]|br\\s*/?)>', '\n', plain, flags=re.I)
        plain = re.sub('<tr[^>]*>|<li[^>]*>', '\n', plain, flags=re.I)
        plain = re.sub('<[^>]+>', '', plain).strip()
        return _decode(plain) if plain and plain[0] != '<' and plain != text.strip() else None
    tree = _tag_tree(text)
    if not tree:
        return None
    if len(tree) == 1:
        return tree[0][1]
    return [node for _, node in tree]


def _token_shape(token: str) -> tuple[str, int]:
    return (re.split('[-_]', token)[0] if re.search('[-_]', token) else token[:2], len(token))


def _anchored(lines: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    shapes: dict[tuple[str, int], int] = {}
    for raw in lines:
        for shape in {_token_shape(t) for t in ID_TOKEN.findall(raw)}:
            shapes[shape] = shapes.get(shape, 0) + 1
    if not shapes:
        return rows
    best = max(shapes, key=lambda shape: (shapes[shape], shape[1]))
    if len(lines) >= 3 and shapes[best] < 0.5 * len(lines):
        return rows
    for raw in lines:
        line = _unmark(raw.strip())
        marks = [t for t in ID_TOKEN.findall(line) if _token_shape(t) == best]
        if not marks:
            continue
        ident = marks[0]
        text = re.sub('^(?:[-*+•]|\\d+[.)])\\s+', '', line).replace(ident, ' ')
        text = re.sub('[\\s()\\[\\]#]+$', '', text).strip(' -—–|:,')
        parts = [part.strip() for part in re.split('\\s[-—–|·•]\\s', text)]
        record: dict[str, Any] = {'id': ident, 'title': parts[0], 'text': text}
        money = next((part for part in parts[1:] if re.fullmatch('[^\\w\\s]{0,2}\\s?\\d[\\d,]*(?:\\.\\d+)?\\s?[A-Za-z]{0,8}', part)), None)
        if money:
            record['price'] = money
        rows.append(record)
    return rows


def _decode(text: str) -> Any:
    body = text.strip()
    if len(body) < 12:
        return None
    if body[0] in '{[':
        try:
            return json.loads(body)
        except ValueError:
            pass
    if body[0] == '<':
        got = _markup(body)
        if got:
            return got
    lines = [line for line in body.splitlines() if line.strip()]
    if len(lines) > 1 and all((line.strip()[:1] == '{' for line in lines)):
        try:
            return [json.loads(line) for line in lines]
        except ValueError:
            pass
    if len(lines) == 1:
        rows = _flat_fields(_unmark(body))
        if rows:
            return rows[0] if len(rows) == 1 else rows
        return None
    first = next((i for i, line in enumerate(lines[:4]) if _flat_fields(_unmark(line))), None)
    if first is not None:
        head = next(iter(_flat_fields(_unmark(lines[first]))[0]))
        starts = [i for i, line in enumerate(lines) if i >= first and _unmark(line).strip().startswith(head)]
        chunks = ['\n'.join(lines[a:b]) for a, b in zip(starts, starts[1:] + [len(lines)])]
        flat = [row for chunk in chunks for row in _flat_fields(_unmark(chunk))]
        if flat:
            extra = [line.strip() for line in lines[:first]]
            if len(flat) == 1 and (not extra):
                return flat[0]
            return {'records': flat, 'text': extra} if extra else flat
    rows, extra = _grid(lines)
    if rows:
        if len(rows) == 1 and (not extra):
            return rows[0]
        return {'records': rows, 'text': extra} if extra else rows
    marked = sum((1 for line in lines if ID_TOKEN.search(line)))
    rows = _anchored(lines) if marked >= 0.6 * len(lines) else []
    extra = []
    if not rows:
        rows, extra = _blocks(body.splitlines())
        rows = [row for row in rows if len(row) >= 3]
        for row in rows:
            marks = ID_TOKEN.findall(str(row.get('header') or row.get('title') or ''))
            if not _has_id(row) and len(marks) == 1:
                row['id'] = marks[0]
    if not rows:
        return None
    if len(rows) == 1 and (not extra):
        return rows[0]
    return {'records': rows, 'text': extra} if extra else rows


def _has_id(record: Any) -> bool:
    return isinstance(record, dict) and len(record) >= 3 and any((_id_like(v, 'id' in str(k).lower()) for k, v in record.items()))


def _hoist(node: dict[str, Any]) -> dict[str, Any]:
    if _has_id(node):
        return node
    blocks = [k for k, v in node.items() if isinstance(v, dict)]
    owner = next((k for k in blocks if _has_id(node[k])), None) or next((k for k in blocks if any((_ref_name(n) and _id_like(v, True) and (not _money_like(v)) for n, v in node[k].items()))), None)
    if owner is None or len(blocks) < 2:
        return node
    out = {**node[owner], **{k: v for k, v in node.items() if k != owner}}
    top = max((len(v) for v in out.values() if isinstance(v, dict)), default=0)
    for key in [k for k, v in out.items() if isinstance(v, dict) and len(v) <= 4 and (len(v) < top or any((re.search('price|cost|amount|stock|avail', str(n), re.I) for n in v)))]:
        block = out.pop(key)
        numbers = [v for v in block.values() if isinstance(v, (int, float)) and (not isinstance(v, bool))]
        tags = len(numbers) == 1 and all((isinstance(v, (int, float)) or (isinstance(v, str) and len(v) <= 5) for v in block.values()))
        out.update({k: v for k, v in block.items() if k not in out and (not tags or not isinstance(v, str))})
    return out


def _read_any(value: Any, depth: int=0, inside: bool=False) -> Any:
    if depth > 6:
        return value
    if isinstance(value, str):
        if inside or len(value) < 24 or ('\n' not in value and not re.search(READ_KEY + '\\s*' + READ_MARK + '|<[A-Za-z]|^\\s*[\\[{]', value)):
            return value
        got = _decode(value)
        if got is None or isinstance(got, str):
            return value
        return _read_any(got, depth + 1)
    if isinstance(value, list):
        if value and all((isinstance(v, str) for v in value)):
            if inside:
                pairs = [re.match('(' + READ_KEY + ')\\s*' + READ_MARK + '\\s+(.*)$', v.strip(), re.S) for v in value]
                return {m.group(1).strip(): m.group(2).strip() for m in pairs if m} if len(value) >= 2 and sum((1 for m in pairs if m)) >= 0.8 * len(value) else value
            singles = [_read_any(v, depth + 1) for v in value]
            if sum((1 for v in singles if not isinstance(v, str))) >= 0.6 * len(value):
                rows: list[Any] = []
                for single in singles:
                    if isinstance(single, dict) and isinstance(single.get('records'), list):
                        rows.extend(single['records'])
                    elif isinstance(single, list):
                        rows.extend(single)
                    else:
                        rows.append(single)
                return rows
            pairs = [re.match('(' + READ_KEY + ')\\s*' + READ_MARK + '\\s*(.*)$', v.strip(), re.S) for v in value]
            if len(value) >= 2 and sum((1 for m in pairs if m)) >= 0.8 * len(value):
                return {m.group(1).strip(): m.group(2).strip() for m in pairs if m}
            return value
        if len(value) >= 2 and all((isinstance(v, dict) and len(v) == 1 for v in value)) and all((isinstance(next(iter(v.values())), (str, int, float)) for v in value)):
            return {next(iter(v)): next(iter(v.values())) for v in value}
        if len(value) >= 2 and all((isinstance(v, dict) and len(v) == 2 for v in value)):
            names = [list(v) for v in value]
            if all((n == names[0] for n in names)):
                first, second = names[0]
                keys = [v[first] for v in value]
                said = [', '.join((str(x) for x in v[second])) if isinstance(v[second], list) and all((isinstance(x, (str, int, float)) for x in v[second])) else v[second] for v in value]
                if all((isinstance(k, str) and 0 < len(k) <= 40 for k in keys)) and len(set(keys)) == len(keys) and all((isinstance(v, (str, int, float)) for v in said)):
                    return dict(zip(keys, said))
        return [_read_any(v, depth + 1, inside) for v in value]
    if isinstance(value, dict):
        texted = any((isinstance(v, str) and _id_like(v, 'id' in str(k).lower()) for k, v in value.items()))
        value = _hoist({k: str(v) if isinstance(v, int) and (not isinstance(v, bool)) and (re.search('(?:^|[_-])(?:id|sku)$|[a-z]Id$', str(k)) or (not texted and re.search('(?:^|[^a-z0-9])(?:ids?|sku|no|num|number|code|ref|key)$', _split_name(k).lower()))) else v for k, v in value.items()})
        record = _has_id(value)
        out: dict[str, Any] = {}
        for key, item in value.items():
            if record and isinstance(item, str):
                block = _pairs(item, (('=', 2),), False) if not any((isinstance(v, dict) for v in value.values())) else None
                out[key] = block if block else item
            else:
                out[key] = _read_any(item, depth + 1, inside or record)
            if isinstance(out[key], list) and 0 < len(out[key]) <= 12 and all((isinstance(v, (str, int, float)) and (not isinstance(v, bool)) for v in out[key])) and (record or inside):
                out[key] = ', '.join((str(v) for v in out[key]))
        return out
    return value


def _amount(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    found = re.search('\\d[\\d,]*(?:\\.\\d+)?', value)
    if not found:
        return None
    digits = found.group(0)
    if re.fullmatch('\\d+,\\d{2}', digits):
        digits = digits.replace(',', '.')
    try:
        return float(digits.replace(',', ''))
    except ValueError:
        return None


class Item:

    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []

class Sable:

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
        self.started = time.time()
        self.late_audit = False
        self.after_refusal = 0
        self.chased: set[str] = set()
        self.gap_filled = False
        self.viewed: set[tuple[str, str]] = set()
        self.routes: dict[str, list[str]] = {}
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
        self.flags: dict[str, tuple[list[str], list[str]]] = {}
        self.audited = False
        self.last_audit: dict[str, Any] = {}
        self.near_tried = False
        self.effort = ''
        self.no_effort = False
        self.carted_ever: list[str] = []
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
        self.confirmed: str | None = None
        self.completed = False
        self.id_key: str | None = None
        self.facts_from: dict[str, bool] = {}
        self.bare_views: dict[str, int] = {}
        self.page_size: dict[str, int] = {}
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
        self.row_keys: set[str] = set()
        self.long_empty = False
        self.read_at: dict[str, int] = {}
        self.stock_key: str | None = None
        self.cheaper_marks: set[str] = set()
        self.swept: set[int] = set()
        self.dept_cache: tuple[Any, set[str]] = (None, set())
        self.home_cache: tuple[Any, dict[str, str]] = (None, {})
        self.filter_warned: set[tuple[str, str]] = set()
        self.sweep_fits: dict[int, list[str]] = {}
        self.swap_fits: list[str] = []
        self.sweeps = 0
        self.sweep_tried: set[str] = set()
        self.decline_swept = False
        self.wider_held = False
        self.paged_obs: list[Any] = []
        self.plan_order: list[int] = []
        self.line_holds = 0
        self.bound: list[tuple[int, int]] = []
        self.fit_holds = 0
        self.link_steps = 0
        self.link_marks: set[str] = set()
        self.searched: list[dict[str, Any]] = []
        self.link_searches = 0
        self.link_seen: dict[str, list[str]] = {}

    def left(self) -> int:
        return self.max_steps - self.step

    def elapsed(self) -> float:
        return time.time() - self.started

    def msgs_free(self) -> int:
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)

    def cart_mark(self) -> str:
        return _dump(sorted(self.cart_lines.items()))

    def read_any(self, name: str, args: dict[str, Any], seen: Any) -> Any:
        try:
            return self.read_shop(name, args, seen)
        except Exception as exc:
            self.trace.append({'kind': 'read_fail', 'tool': name, 'error': str(exc)[:160]})
            return seen

    def read_shop(self, name: str, args: dict[str, Any], seen: Any) -> Any:
        if seen is None or name == self.roles.get('ask'):
            return seen
        out = _read_any(seen)
        if isinstance(seen, dict) and len(seen) == 1 and isinstance(next(iter(seen.values())), str) and isinstance(next(iter(out.values())), (dict, list)):
            out = next(iter(out.values()))
        if isinstance(out, list) and out != seen:
            out = {'records': out}
        self.learn_lists(out, 0)
        out = self.lift_facts(name, args, out)
        if name == self.roles.get('search') and (not self.price_key or not self.stock_key):
            self.read_shape(name, out)
        return out

    def learn_lists(self, node: Any, depth: int) -> None:
        if depth > 6:
            return
        if isinstance(node, dict):
            for value in node.values():
                self.learn_lists(value, depth + 1)
        elif isinstance(node, list):
            self.learn_rows(node)
            for value in node:
                if isinstance(value, (dict, list)):
                    self.learn_lists(value, depth + 1)

    def lift_facts(self, name: str, args: dict[str, Any], out: Any) -> Any:
        single = name == self.roles.get('open') and (not self.roles.get('offer'))
        page = single or name == self.roles.get('compare')
        target = next((v for v in args.values() if isinstance(v, str) and v in self.rows), None)
        if single and target:
            text = out if isinstance(out, str) else next(iter(out.values())) if isinstance(out, dict) and len(out) == 1 else None
            if isinstance(text, str) and len(text) >= 80:
                out = {self.id_key or 'id': target, 'title': text[:80], 'text': text}
            elif isinstance(out, dict) and len(out) >= 3 and (not self.ident(out)) and (not _trouble(out)) and (not any((isinstance(v, list) for v in out.values()))):
                out = {self.id_key or 'id': target, **out}
        if single and isinstance(out, dict) and (not self.id_key):
            self.id_key = _learn_id([out]) if len(out) >= 3 else None
        for _, record in self.records_in(out):
            _merge_twins(record)
            self.learn_fields(record)
            if not _fact_block(record):
                facts = self.flat_facts(record, page)
                if facts:
                    record['facts'] = facts
            if not page:
                self.row_keys.update(record)
        return out

    def learn_fields(self, record: dict[str, Any]) -> None:
        if len(record) < 4:
            return
        if not self.price_key:
            for word in ('price', 'cost', 'amount'):
                found = next((k for k, v in record.items() if word in str(k).lower() and (not re.search('total|due|held|qty|count|old|was', str(k), re.I)) and (not isinstance(v, (dict, list))) and (_amount(v) is not None)), None)
                if found:
                    self.price_key = found
                    break
        if not self.stock_key:
            flags = [k for k, v in record.items() if isinstance(v, bool)]
            self.stock_key = next((k for k, v in record.items() if re.search('stock|avail|on.?hand|buyable|purchasable', _split_name(k), re.I) and (not isinstance(v, (dict, list)))), None) or (flags[0] if len(flags) == 1 else None)

    def flat_facts(self, record: dict[str, Any], page: bool) -> dict[str, Any]:
        skip = {self.id_key, self.price_key, self.stock_key, self.category_key()}
        facts: dict[str, Any] = {}
        for key, value in record.items():
            if key in skip or value is None or isinstance(value, (bool, dict, list)) or len(str(value)) > 80:
                continue
            if isinstance(value, (int, float)) and re.search('price|cost|amount|rank|score|count', str(key), re.I):
                continue
            facts[key] = value
        if page:
            fresh = {k: v for k, v in facts.items() if k not in self.row_keys}
            return fresh if len(fresh) >= 3 else {}
        return facts if len(facts) >= 6 else {}

    def page_tools(self) -> set[str]:
        page = self.roles.get('offer') or self.roles.get('open')
        return {t for t in (page, self.roles.get('compare')) if t}

    def note_facts(self, tool: str, record: Any) -> None:
        has = bool(_fact_block(record))
        self.bare_views[tool] = 0 if has else self.bare_views.get(tool, 0) + 1
        rival = any((t != tool and re.search('id|code|sku|ref|number', self.single_target(t) or '', re.I) and t not in self.roles.values() for t in self.schemas)) or bool(self.roles.get('offer'))
        if has or rival or self.bare_views[tool] >= 2:
            self.facts_from[tool] = self.facts_from.get(tool, False) or has

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
        return 3 if talk and self.msg_used < self.msg_limit and self.gate_due('order') else 2

    def judges(self, limit: float=SPEND_SOFT) -> list[str]:
        if self.cost <= limit and self.cost + HEAVY_ROOM <= SPEND_HARD:
            return [HEAVY, LIGHT]
        return [LIGHT] if self.cost <= SPEND_HARD else []

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None=None) -> dict[str, Any] | None:
        for model in models:
            if self.calls >= ALL_CALLS or self.elapsed() > CALL_STOP:
                return None
            self.calls += 1
            tries = 0
            while True:
                message, status = self.call_model(model, messages, max_tokens, tools)
                if message is not None:
                    return message
                passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                if not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL or self.elapsed() + WAITS[tries] > CALL_STOP:
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
        if self.effort and model == LIGHT and (not self.no_effort):
            body['reasoning'] = {'effort': self.effort}
        if tools:
            body['tools'] = tools
            body['tool_choice'] = 'auto'
        result = (_late if self.elapsed() > WRAP_AT else _llm).post_verbose('/inference/chat/completions', body)
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
        if 'reasoning' in body and isinstance(status, int) and 400 <= status < 500 and status not in (402, 408, 429):
            self.no_effort = True
            return self.call_model(model, messages, max_tokens, tools)
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
        late = self.elapsed() > WRAP_AT
        for _ in range(2 if late else 4):
            result = (_env_late if late else _env).post_verbose('/environment/call', payload)
            if isinstance(result.data, dict):
                data = result.data
                self.step = data['step'] if isinstance(data.get('step'), int) else self.step + 1
                if data.get('finished'):
                    self.finished = True
                seen = data['observation'] = self.read_any(name, args, data.get('observation'))
                if not _trouble(seen):
                    tag = '*compare*' if name == self.roles.get('compare') else name
                    for value in args.values():
                        self.viewed.update(((v.strip(), tag) for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)))
                if name in self.page_tools() and (not _trouble(seen)):
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
        self.brief = self.fetch_brief()
        self.take_roles(self.brief)
        self.take_items(self.brief)
        self.take_limits(self.brief)

    @staticmethod
    def usable_brief(brief: dict[str, Any]) -> bool:
        return bool(_items(brief.get('items'))) and isinstance(brief.get('roles'), dict) and bool(brief['roles'].get('buy'))

    def fetch_brief(self) -> dict[str, Any]:
        menu = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_dump(t['function'].get('parameters') or {})}" for t in self.tools]
        prompt = BRIEF + '\nTOOLS:\n' + '\n'.join(menu) + '\n\nREQUEST:\n' + self.request
        if self.view.get('opening') is not None:
            prompt += '\n\nOPENING NOTE FROM THE STORE:\n' + _dump(_clip(self.view.get('opening'), 1500))
        brief = self.ask_json(prompt, [LIGHT], 8000) or {}
        if self.no_credit:
            return brief
        if not self.usable_brief(brief):
            return self.ask_json(prompt, [HEAVY, LIGHT], 5000) or brief
        if self.shared_shelf(brief):
            second = self.ask_json(prompt, [HEAVY], 5000) or {}
            if self.usable_brief(second):
                return second
        return brief

    def take_roles(self, brief: dict[str, Any]) -> None:
        named = brief.get('roles') if isinstance(brief.get('roles'), dict) else {}
        self.roles = {job: tool for job, tool in named.items() if isinstance(tool, str) and tool in self.schemas}
        if not all((self.roles.get(job) for job in ('buy', 'refuse', 'search'))):
            self.guess_roles()
        for key in ('buy_args', 'refuse_args'):
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        buy = self.roles.get('buy')
        if not buy or buy != self.roles.get('refuse'):
            return
        if brief['buy_args'] and brief['refuse_args'] and brief['buy_args'] != brief['refuse_args']:
            return
        ask = f'The store tool {buy} both checks out and ends without a purchase, depending on its arguments. Its schema: ' + _dump(self.schemas.get(buy)) + MODE_ASK
        modes = self.ask_json(ask, [LIGHT, HEAVY], 1500) or {}
        yes, no = (modes.get('buy_args'), modes.get('refuse_args'))
        if isinstance(yes, dict) and isinstance(no, dict) and yes != no:
            brief['buy_args'], brief['refuse_args'] = (yes, no)

    def take_items(self, brief: dict[str, Any]) -> None:
        for entry in _items(brief.get('items')):
            if isinstance(entry, dict):
                wanted = _count(entry.get('quantity'))
                musts = [str(m) for m in _items(entry.get('musts')) if m]
                self.items.append(Item(str(entry.get('name') or 'the product'), str(entry.get('shelf') or ''), wanted if wanted and wanted > 0 else 1, musts))

        def clash(first: Item, second: Item) -> bool:
            return any((_words(a.split(':')[0]) == _words(b.split(':')[0]) and _words(a.partition(':')[2]) != _words(b.partition(':')[2]) for a in first.musts for b in second.musts))
        kept: list[Item] = []
        moved: dict[int, int] = {}
        for number, item in enumerate(self.items, 1):
            host = None
            if item.shelf.strip():
                host = next((other for other in kept if _words(other.shelf) == _words(item.shelf) and (not clash(other, item))), None)
            if host is None:
                kept.append(item)
                moved[number] = len(kept)
                continue
            moved[number] = kept.index(host) + 1
            host.musts += [m for m in item.musts if m not in host.musts]
            self.trace.append({'kind': 'items_merged', 'into': host.name, 'dropped': item.name})
        self.items = kept or [Item('the requested product', '', 1, [])]
        size = len(self.items)
        for pair in _items(brief.get('fits')):
            if not isinstance(pair, dict):
                continue
            part, main = (moved.get(_count(pair.get('item')) or 0), moved.get(_count(pair.get('with')) or 0))
            if part and main and part != main and 1 <= part <= size and 1 <= main <= size and ((part, main) not in self.bound):
                self.bound.append((part, main))
        for lookup in _items(brief.get('lookups')):
            text = str(lookup).strip()
            if not any((text in item.musts for item in self.items)):
                head = _words(text.split(':')[0])
                text = next((m for item in self.items for m in item.musts if _words(m.split(':')[0]) == head), text)
            owner = next((n for n, item in enumerate(self.items, 1) if text in item.musts), 1)
            if text and (owner, text) not in self.refs:
                self.refs.append((owner, text))
        self.unmix()

    def take_limits(self, brief: dict[str, Any]) -> None:
        budget = brief.get('budget')
        if isinstance(budget, str) and re.fullmatch(PLAIN_NUMBER, budget):
            budget = float(budget)
        if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and budget > 0:
            self.budget = budget
        cap = _count(brief.get('messages'))
        if cap is not None and cap >= 0:
            self.msg_limit = cap
        elif self.roles.get('ask'):
            self.msg_limit = 3
        steps = _count(brief.get('steps'))
        if steps is not None and 10 <= steps < self.max_steps:
            self.max_steps = steps
        for entry in _items(brief.get('process')):
            if not isinstance(entry, dict) or not str(entry.get('step') or '').strip():
                continue
            when = str(entry.get('when') or '').strip().lower()
            self.process.append({'step': str(entry['step']).strip(), 'when': when if when in ('choose', 'order', 'end') else 'order', 'messages': max(0, _count(entry.get('messages')) or 0)})
        if self.roles.get('ask'):
            talk = [step['messages'] for step in self.process if step['when'] == 'order' and step['messages']]
            self.reserve = min(max(talk), self.msg_limit) if talk else 0

    def guess_roles(self) -> None:
        for name, schema in self.schemas.items():
            props = schema.get('properties') or {}
            required = [k for k in schema.get('required') or [] if k in props]
            text = (name + ' ' + next((str(t['function'].get('description') or '') for t in self.tools if t['function']['name'] == name), '')).lower()
            texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
            numbers = [k for k, v in props.items() if isinstance(v, dict) and {'integer', 'number'} & set(_kinds(v))]
            modes = [k for k, v in props.items() if isinstance(v, dict) and isinstance(v.get('enum'), list)]
            ending = re.search('\\bfinal\\b|no (?:more|further) actions|\\bends? (?:the|this) (?:task|episode|visit|session|chat)', text)
            if not required and (not texts or ending) or (ending and all((k in texts or k in modes for k in required))):
                pair = None
                for key in modes:
                    values = [v for v in props[key].get('enum') or [] if isinstance(v, str)]
                    no = next((v for v in values if re.search('nothing|no purchase|no[ _-]?(?:buy|order|purchase)|without|none|declin|cancel|abstain|\\bpass\\b|\\bskip\\b|refus', v.lower())), None)
                    yes = next((v for v in values if v != no and re.search('buy|order|purchase|check ?out|basket|cart|trolley|commit|confirm|complete|place', v.lower())), None)
                    if no and yes:
                        pair = (key, yes, no)
                        break
                if pair and 'refuse' not in self.roles:
                    self.roles['buy'] = name
                    self.roles['refuse'] = name
                    self.brief['buy_args'] = {**(self.brief.get('buy_args') if isinstance(self.brief.get('buy_args'), dict) else {}), pair[0]: pair[1]}
                    self.brief['refuse_args'] = {**(self.brief.get('refuse_args') if isinstance(self.brief.get('refuse_args'), dict) else {}), pair[0]: pair[2]}
                    continue
                if 'refuse' not in self.roles and re.search('without|abstain|declin|refus|cancel|give up|leave|walk away|no purchase', text):
                    self.roles['refuse'] = name
                elif 'buy' not in self.roles and re.search('order|check ?out|purchase|buy|finali[sz]|place|pay|confirm|complete', text):
                    self.roles['buy'] = name
                continue
            if len(texts) == 1 and len(props) == 1 and re.search('shopper|customer|user|buyer|message|ask|say|tell|chat', text):
                self.roles.setdefault('ask', name)
            elif texts and (numbers or modes) and re.search('cart|basket|trolley|bag|hold|add|remove', text):
                self.roles.setdefault('cart', name)
            elif texts and len(props) >= 2 and re.search('search|seek|find|look|browse|list|query', text):
                self.roles.setdefault('search', name)
            elif len(required) == 1 and required[0] in texts and re.search('offer|listing|item|product|detail|record|page|examine|open|view|describe', text):
                self.roles.setdefault('open', name)
        buy = self.roles.get('buy') or self.roles.get('refuse')
        if buy and (not (self.roles.get('buy') and self.roles.get('refuse'))):
            words = str(next((t['function'].get('description') or '' for t in self.tools if t['function']['name'] == buy), '')).lower()
            for key, spec in ((self.schemas.get(buy) or {}).get('properties') or {}).items():
                values = [v for v in spec.get('enum') or [] if isinstance(v, str)] if isinstance(spec, dict) else []
                said = {v: v.lower() + ' ' + re.split('[,.;]| or ', words.split(v.lower(), 1)[1], maxsplit=1)[0] if v.lower() in words else v.lower() for v in values}
                refusing = 'nothing|no purchase|no[ _-]?(?:buy|order|purchase)|without|none|declin|cancel|abstain|\\bpass\\b|\\bskip\\b|refus'
                buying = 'buy|order|purchase|check ?out|basket|cart|trolley|commit|confirm|complete|place'
                no = next((v for v in values if re.search(refusing, v.lower())), None) or next((v for v in values if re.search(refusing, said[v])), None)
                yes = next((v for v in values if v != no and re.search(buying, v.lower())), None) or next((v for v in values if v != no and re.search(buying, said[v])), None)
                if no and yes:
                    self.roles['buy'] = buy
                    self.roles['refuse'] = buy
                    self.brief['buy_args'] = {**(self.brief.get('buy_args') if isinstance(self.brief.get('buy_args'), dict) else {}), key: yes}
                    self.brief['refuse_args'] = {**(self.brief.get('refuse_args') if isinstance(self.brief.get('refuse_args'), dict) else {}), key: no}
                    break
        self.trace.append({'kind': 'guessed_roles', 'roles': dict(self.roles)})

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

    def dropped_note(self) -> str:
        gone = [m for item in self.items for m in item.dropped]
        return '\n\nNO LONGER REQUIRED (the shopper has given these up; ignore them wherever the request still states them):\n' + '\n'.join(('- ' + m for m in gone)) if gone else ''

    def ledger(self) -> str:
        lines = [self.item_line(number, item) for number, item in enumerate(self.items, 1)]
        return '\n'.join(('- ' + line for line in lines + self.rule_lines()))

    def item_line(self, number: int, item: Item) -> str:
        live = [m for m in item.musts + item.told if m not in item.dropped]
        loose = [m for m in live if self.described(number, m)]
        firm = [m + (f' (value from the page it points to: {self.ref_values[m]})' if m in self.ref_values else '') for m in live if m not in loose]
        text = f'item {number}: {item.name}' if len(self.items) > 1 else f'item: {item.name}'
        if item.shelf or self.home_of(item):
            text += f' (shop department: {item.shelf or self.home_of(item)})'
        limits = {str(x).strip() for x in _items(self.brief.get('limits'))}
        firm = [m + (' (a limit)' if m.strip() in limits else '') + (' (heard from the shopper)' if m in item.told else '') for m in firm]
        text += f', quantity {item.qty}; requirements: ' + ('; '.join(firm) if firm else 'nothing beyond being that kind of product')
        if loose:
            text += "; worded the shopper's way, with no listing attribute stating it (it settles nothing): " + '; '.join(loose)
        if item.free:
            text += '; shopper is indifferent to: ' + ', '.join(item.free)
        return text

    def rule_lines(self) -> list[str]:
        brief = self.brief
        out: list[str] = []
        if self.budget is not None:
            reach = 'per item' if brief.get('budget_scope') == 'each' else 'overall'
            out.append(f"spending limit: {self.budget} {brief.get('budget_unit') or ''} {reach} at most".replace('  ', ' '))
        pick = brief.get('pick')
        if pick == 'cheapest':
            out.append('pick: the lowest-priced listing that qualifies')
        elif pick in ('most', 'least'):
            out.append(f"pick: the qualifying listing with the {pick} {brief.get('pick_measure') or ''}".replace('  ', ' ').rstrip())
        else:
            out.append('pick: any listing meeting every requirement')
        if brief.get('in_stock_only', True) is not False:
            out.append('only listings in stock')
        out += [f'fit rule: {self.items[p - 1].name} must work with {self.items[m - 1].name} in this order; read it from the two pages, where a code on one is named on the other (a part series named on the device page covers each part whose own code starts with that series), ranges and exceptions included' for p, m in self.bound]
        out += [f'nice to have (only breaks ties): {wish}' for wish in _items(brief.get('wishes'))]
        out += [f'the request says: {rule}' for rule in _items(brief.get('rules'))]
        if brief.get('quiet_decline') is True:
            out.append('when nothing qualifies: end with no purchase, naming or suggesting no listing')
        moments = {'choose': 'before choosing', 'order': 'at checkout (Sable handles it when you call checkout)', 'end': 'at a no-purchase ending (Sable handles it then)'}
        out += [f"step the request requires, {moments[entry['when']]}: {entry['step']}" for entry in self.process]
        facts = self.facts_tool()
        if facts and (self.facts_from.get(facts) is True or self.roles.get('offer')):
            silent = sorted((t for t in self.page_tools() | {self.roles.get('offer') or ''} if t and self.facts_from.get(t) is False))
            pages = sorted((t for t in self.page_tools() if t != facts))
            line = f'specifications come from {facts}'
            if silent:
                line += f"; {', '.join(silent)} show no specifications"
            if pages:
                line += f"; a listing's full page (meant by any chain of pages or page check) comes from {' or '.join(pages)}"
            out.append(line)
        return out

    def status(self) -> str:
        steps = self.left()
        pace = self.elapsed() / max(1, self.step)
        clock = int(max(0.0, WRAP_AT - self.elapsed()) / max(pace, 1.0)) if self.step >= 3 else steps
        room = min(steps, clock)
        text = f'[Sable] {self.step} of {self.max_steps} steps used, {steps} remaining.'
        if clock < steps:
            text += f' At the current pace the clock allows about {clock} more, so plan for {room}.'
        text += self.pace_hint(room)
        live = {need for _, need in self.needs_now()}
        waiting = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if waiting and self.buys == 0:
            text += '\n[settle these first: open the listing each one names, then every listing that page sends you to; the value comes from the final page] ' + ' | '.join(waiting[:3])
        if self.leads:
            text += '\n[candidates to review] ' + ', '.join(self.leads)
        plan = self.fit_plan() if self.bound and self.buys == 0 else ''
        if plan:
            text += '\n' + plan
        if self.notes:
            text += '\n[changes up to now] ' + ' | '.join(self.notes[-4:])
        return text + '\n[ledger]\n' + self.ledger() + self.cart_by_item()

    def cart_by_item(self) -> str:
        cat_key = self.category_key()
        cart = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        if not cat_key or not cart:
            return ''

        def label(ident: str) -> str:
            return str((self.rows.get(ident) or {}).get(cat_key) or '')
        parts = []
        for number, item in enumerate(self.items, 1):
            mine = [i for i in cart if self.owns(item, label(i))]
            parts.append(f'item {number}: ' + (', '.join((f'{i} x{self.held_units(i)}' for i in mine)) if mine else 'nothing yet'))
        stray = [i for i in cart if not any((self.owns(item, label(i)) for item in self.items))]
        text = "\n[cart by item, read from each listing's department] " + '; '.join(parts)
        if stray:
            text += '; in no requested department: ' + ', '.join((f'{i} ({label(i) or "unknown"})' for i in stray))
        return text

    def pace_hint(self, room: int) -> str:
        if room <= self.tail() + 1:
            return ' LAST STEPS: if the cart has a line for every item, check out now; end with no purchase only if your searching proved some item has no qualifying listing (being short of steps proves nothing of the kind).'
        if room <= 6 or (room <= len(self.items) + 5 and self.buys == 0):
            return ' Only a few steps remain: make the cart hold exactly your chosen lines and check out right away.'
        if self.buys:
            return ''
        if room <= 12:
            return ' Steps are getting short: decide your picks, fill the cart and check out within the next few steps.'
        if (self.step >= self.max_steps // 2 or self.elapsed() >= WRAP_AT / 2) and self.uncarted():
            return ' Half the steps or time are gone and these products still have no cart line: ' + ', '.join(self.uncarted()) + '. Cart the best qualifying listing you have for each of them now and refine afterwards (a carted line can still sell out).'
        return ''

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
        stack = [(schema, 0) for schema in self.schemas.values()]
        while stack:
            schema, depth = stack.pop()
            if depth > 6 or not isinstance(schema, dict):
                continue
            found |= {option.strip().lower() for option in schema.get('enum') or [] if isinstance(option, str)}
            stack += [(value, depth + 1) for value in schema.values() if isinstance(value, dict)]
        return found

    def category_key(self) -> str | None:
        values: dict[str, set[str]] = {}
        seen: dict[str, int] = {}
        for row in self.rows.values():
            for key, value in row.items():
                if isinstance(value, str) and 0 < len(value) <= 40 and (not re.search('\\d{3}', value)):
                    values.setdefault(key, set()).add(value)
                    seen[key] = seen.get(key, 0) + 1
        floor = 0.8 * max(1, len(self.rows))
        known = self.labels()
        ranked = []
        for key, options in values.items():
            if seen[key] < floor or len(options) > 40 or key in (self.id_key, self.price_key, self.stock_key):
                continue
            listed = len([v for v in options if v.strip().lower() in known]) / len(options)
            mean = sum((len(v) for v in options)) / len(options)
            ranked.append(((-listed, len(options), -seen[key], -mean), key))
        if not ranked:
            return None
        ranked.sort(key=lambda entry: entry[0])
        best = ranked[0]
        named = set().union(*(_roots(item.name) | _roots(item.shelf or '') for item in self.items)) if self.items else set()

        def touches(key: str) -> bool:
            return any((_roots(v) & named for v in values[key]))
        if named and len(values[best[1]]) == 1 and (not touches(best[1])):
            return next((key for _, key in ranked if touches(key)), best[1])
        return best[1]


    def key_table(self) -> list[tuple[str, str, int, list[str]]]:
        table: list[tuple[str, str, int, list[str]]] = []
        groups = self.key_groups()
        for label, group in sorted(groups.items(), key=lambda kv: -kv[1]['n']):
            for name, slot in self.merge_keys(group['keys'])[:14]:
                table.append((label, name, slot['n'], [v[:40] for v in slot['values'][:4]]))
        return table

    def key_groups(self) -> dict[str, dict[str, Any]]:
        cat_key = self.category_key()
        groups: dict[str, dict[str, Any]] = {}
        for ident in self.seen_order:
            record = self.full.get(ident)
            if not record:
                continue
            label = str(record.get(cat_key) or 'listings') if cat_key else 'listings'
            group = groups.setdefault(label, {'n': 0, 'keys': {}})
            group['n'] += 1
            met: set[str] = set()
            for key, value in (_fact_block(record) or {}).items():
                name = _words(key)
                if not name or name in met:
                    continue
                met.add(name)
                slot = group['keys'].setdefault(name, {'n': 0, 'values': [], 'per': {}})
                slot['n'] += 1
                text = str(value).strip().lower()
                slot['per'][ident] = text
                if text not in slot['values']:
                    slot['values'].append(text)
        wanted = [(_words(item.shelf or ''), _words(item.name), {w for w in _stem(item.shelf or '') | _stem(item.name) if len(w) >= 4}) for item in self.items]

        def relevant(label: str) -> bool:
            long = {w for w in _stem(label) if len(w) >= 4}
            return any((_words(label) in (shelf, name) or bool(long & words) for shelf, name, words in wanted)) or bool(self.owner_of(label))
        chosen = [label for label in groups if relevant(label)]
        return {label: groups[label] for label in groups if label in chosen} if chosen else groups

    def merge_keys(self, keys: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
        kept: list[tuple[str, dict[str, Any]]] = []
        for name, slot in sorted(keys.items(), key=lambda kv: (-kv[1]['n'], len(kv[0]))):
            values = list(slot['per'].values())
            if not values or self.noisy(values):
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
        return kept

    @staticmethod
    def noisy(values: list[str]) -> bool:
        mean = sum((len(v) for v in values)) / len(values)
        if mean > 40 or 2 * len([v for v in values if ' / ' in v]) >= len(values):
            return True
        if len(values) < 4 or len(set(values)) != len(values) or mean <= 12:
            return False
        echoed = 0
        for spot, value in enumerate(values):
            rest = ' '.join((v for j, v in enumerate(values) if j != spot))
            echoed += bool(set(re.findall('[a-z]{3,}', value)) & set(re.findall('[a-z]{3,}', rest)))
        return 2 * echoed < len(values)

    def not_stated(self, label: str, name: str, values: list[str]) -> str:
        number = self.owner_of(label)
        if not number:
            return ''
        shown = set(re.findall('[a-z0-9]{3,}', ' '.join(values).lower()))
        for must in self.items[number - 1].musts:
            prop, _, value = must.partition(':')
            if re.search(BOUND_WORDS, value, re.I):
                mine, theirs = (_stem(name), _stem(prop))
                if prop.strip() and _words(prop) != name and mine & theirs and mine - theirs and theirs - mine:
                    return f' [not the stated {prop.strip()}: another property]'
                continue
            words = set(re.findall('[a-z0-9]{3,}', value.lower()))
            if words and prop.strip() and _words(prop) != name and _stem(prop) & _stem(name) and (not words <= shown):
                return f' [not the stated {prop.strip()}: another property]'
        return ''

    @staticmethod
    def twins(name: str, spec: dict[str, Any], other: str, slot: dict[str, Any]) -> bool:
        left, right = (set(name.split()), set(other.split()))
        if not (left and right and left & right):
            return False

        def close(x: str, y: str) -> bool:
            if x == y:
                return True
            wx, wy = (re.findall('[a-z0-9]+', x.lower()), re.findall('[a-z0-9]+', y.lower()))
            short, long = (wx, wy) if len(wx) <= len(wy) else (wy, wx)
            return bool(short) and set(short) <= set(long) and (len(long) <= 2 * len(short) + 2)
        shared = [i for i in spec['per'] if i in slot['per'] and spec['per'][i].strip() and slot['per'][i].strip()]
        alike = len([i for i in shared if close(spec['per'][i], slot['per'][i])])
        if left <= right or right <= left:
            if shared:
                return 2 * alike > len(shared)
            return bool((set(spec['values']) & set(slot['values'])) - {''})
        return len(shared) >= 2 and alike == len(shared)

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
        if isinstance(value, str):
            value = _amount(value)
        return float(value) if isinstance(value, (int, float)) and (not isinstance(value, bool)) else None

    def buyable(self, ident: str) -> bool:
        if self.sold_out(ident):
            return False
        value = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        if isinstance(value, str) and re.match('\\s*(?:no\\b|false\\b|0\\b|out\\b|sold\\b|unavailable\\b|none\\b)', value, re.I):
            return False
        return not (value is False or (isinstance(value, (int, float)) and (not isinstance(value, bool)) and (value <= 0)))

    def records_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        observation = _payload(observation)[0]
        if isinstance(observation, dict):
            ident = self.ident(observation)
            if ident:
                return [(ident, observation)]
        found: list[tuple[str, dict[str, Any]]] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 5:
                return
            if isinstance(node, dict):
                inner = self.ident(node, True) if depth else None
                if inner:
                    found.append((inner, node))
                    return
                for value in node.values():
                    if isinstance(value, (dict, list)):
                        walk(value, depth + 1)
            elif isinstance(node, list):
                for value in node:
                    if isinstance(value, (dict, list)):
                        walk(value, depth + 1)
        walk(observation, 0)
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
        observation = _payload(observation)[0]
        if name in self.shapes:
            return self.shapes[name]
        if not isinstance(observation, dict):
            return None
        lists = [k for k, v in observation.items() if isinstance(v, list) and len(v) >= 2 and all((isinstance(r, dict) for r in v))]
        if not lists:
            return None
        self.shapes[name] = None
        sample = {k: v[:3] if k in lists else v for k, v in observation.items()}
        got = self.ask_json(SHAPE + '\n\nTOOL: ' + name + ' | parameters: ' + _dump(self.schemas.get(name) or {}) + '\n\nRESULT (first rows only):\n' + _dump(_clip(sample, 120))[:4000], [LIGHT, HEAVY], 2500) or {}
        rows = got.get('rows') if got.get('rows') in lists else None
        if not rows:
            return None
        first = observation[rows][0]
        props = (self.schemas.get(name) or {}).get('properties') or {}

        def named(key: Any, where: dict[str, Any]) -> str | None:
            return key if isinstance(key, str) and key in where else None
        shape = {'rows': rows, 'count': named(got.get('count'), observation), 'next': named(got.get('next'), observation), 'next_arg': named(got.get('next_arg'), props), 'price': named(got.get('price'), first), 'stock': named(got.get('stock'), first)}
        total = observation[shape['count']] if shape['count'] else None
        if shape['count'] and (isinstance(total, bool) or not isinstance(total, int)):
            shape['count'] = None
        self.shapes[name] = shape
        self.price_key = self.price_key or shape['price']
        self.stock_key = self.stock_key or shape['stock']
        self.trace.append({'kind': 'shape', 'tool': name, 'shape': shape})
        return shape

    def page_param(self, name: str) -> str | None:
        props = (self.schemas.get(name) or {}).get('properties') or {}
        numbers = [k for k, v in props.items() if isinstance(v, dict) and set(_kinds(v)) & {'integer', 'number'}]
        return next((k for k in numbers if re.search(PAGE_WORDS + '|' + OFFSET_WORDS, _split_name(k))), None)

    def next_start(self, name: str, arg: str, args: dict[str, Any], shown: int, count: int) -> int | None:
        given = args.get(arg)
        at = given if isinstance(given, int) and (not isinstance(given, bool)) else None
        self.page_size[name] = max(self.page_size.get(name, 0), shown)
        if re.search(OFFSET_WORDS, _split_name(arg)) and (not re.search(PAGE_WORDS, _split_name(arg))):
            later = (at or 0) + shown
            return later if later < count else None
        page = at or 1
        return page + 1 if page * self.page_size[name] < count and shown == self.page_size[name] else None

    def paging(self, name: str, observation: Any, args: dict[str, Any] | None=None) -> tuple[int, int, str, Any] | None:
        observation = _payload(observation)[0]
        shape = self.read_shape(name, observation)
        if not shape or not shape['count'] or (not isinstance(observation, dict)):
            return None
        param = shape['next_arg'] or (None if shape['next'] else self.page_param(name))
        rows, count = (observation.get(shape['rows']), observation.get(shape['count']))
        if not param or not isinstance(rows, list) or (not rows) or isinstance(count, bool) or (not isinstance(count, int)):
            return None
        value = observation.get(shape['next']) if shape['next'] else self.next_start(name, param, args or {}, len(rows), count)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))):
            return None
        return (count, len(rows), param, value)

    def price_sorted(self, observation: Any) -> bool:
        prices = [self.price_of(i) for i, _ in self.records_in(observation)]
        if len(prices) < 3 or None in prices:
            return False
        return all((low <= high for low, high in zip(prices, prices[1:])))

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        self.paged_obs = []
        pick = self.brief.get('pick')
        plan = self.paging(name, observation, args) if pick in ('cheapest', 'most', 'least') else None
        if not plan:
            return ''
        count, shown, param, value = plan
        given = args.get(param)
        if isinstance(given, int) and (not isinstance(given, bool)):
            later = given not in (None, 0, 1)
        else:
            later = bool(str(given or '').strip())
        if value is None or later or count <= shown or (pick == 'cheapest' and self.price_sorted(observation)):
            return ''
        pages = -(-(count - shown) // shown)
        spare = self.left() - self.tail()
        reach = AUTO_PAGES * max(1, len(self.items))
        if pages > 2 and self.pages_read + 2 <= reach and spare - 2 >= 12:
            pages = 2
        if pages > 2 or self.pages_read + pages > reach or spare - pages < 12:
            return f"\n[Sable] {count} listings match this search but only {shown} were shown, in no price order. The pick weighs every qualifying listing: narrow the words (use the requirement's value) or page on with {param}={_dump(value)}."
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
            after = self.paging(name, seen, call)
            value = after[3] if after else None
            if value is None:
                break
        self.trace.append({'kind': 'pages', 'tool': name, 'read': len(texts)})
        if not texts:
            return ''
        rest = f"\n[Sable] {count} listings match; Sable has now read {shown * (len(texts) + 1)} rows of them. Narrow the search with the requirement's value or page on for the rest." if value is not None else ''
        return '\n[Sable fetched the following pages of this search itself, one step per page]\n' + '\n'.join(texts) + rest

    def filter_note(self, args: dict[str, Any], observation: Any) -> str:
        cat_key = self.category_key()
        rows = [r for _, r in self.records_in(observation)]
        if not cat_key or len(rows) < 2:
            return ''
        known = self.departments()
        props = (self.schemas.get(self.roles.get('search') or '') or {}).get('properties') or {}
        for key, value in args.items():
            if not isinstance(value, str) or _squash(value) not in known or re.search(QUERY_FIELD, _split_name(key), re.I):
                continue
            others = sorted({str(r.get(cat_key) or '') for r in rows if _squash(r.get(cat_key)) != _squash(value)} - {''})
            modes = ', '.join((f'{k}={v}' for k, v in sorted(args.items()) if k != key and isinstance(v, str) and isinstance((props.get(k) or {}).get('enum'), list)))
            if not others or (key, modes) in self.filter_warned:
                return ''
            self.filter_warned.add((key, modes))
            return f"\n[Sable] {key}={value!r} did not keep this search inside that department: rows from {', '.join(others[:3])} came back as well" + (f' (with {modes})' if modes else '') + f', so here {key} does not filter; a different search mode may apply it.'
        return ''

    def cap_ignored(self, name: str, args: dict[str, Any], observation: Any) -> str:
        props = (self.schemas.get(name) or {}).get('properties') or {}

        def priced(key: str) -> bool:
            spec = props.get(key) if isinstance(props.get(key), dict) else {}
            return bool(re.search('price|cost|amount|budget|spend|money', key + ' ' + str(spec.get('description') or ''), re.I))
        caps = sorted((k for k, v in args.items() if k in props and isinstance(v, (int, float)) and (not isinstance(v, bool)) and re.search('max|most|under|below|limit|cap|upto|up_to', k, re.I) and priced(k)), key=lambda k: args[k])
        if not caps or not self.price_key:
            return ''
        cap = args[caps[0]]
        over = [i for i, _ in self.records_in(observation) if (self.price_of(i) or 0) > cap]
        if not over:
            return ''
        others = {k: [o for o in props[k]['enum'] if o != v] for k, v in args.items() if isinstance(v, str) and isinstance((props.get(k) or {}).get('enum'), list)}
        hint = ''
        if any(others.values()):
            hint = '; to see only cheaper listings, list the department with ' + ', '.join((f'{k}={_dump(o[0])}' for k, o in others.items() if o)) + f' and the same {caps[0]}'
        return f'\n[Sable] The search ignored {caps[0]}={cap:g}: {len(over)} of its rows are pricier{hint}.'

    def cheapest_here(self, observations: list[Any]) -> str:
        if self.brief.get('pick') != 'cheapest' or not self.price_key:
            return ''
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = sorted(((self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None))
        if len(rows) < 2:
            return ''
        return '\n[Sable] rows here that can be bought, lowest price first: ' + ', '.join((f'{i} ({p:g})' for p, i in rows[:6]))

    def wider_hold(self) -> str:
        if self.wider_held or self.brief.get('pick') != 'cheapest' or self.left() < 14 + 2 * len(self.items):
            return ''
        self.wider_held = True
        lines = [f'{i} at {self.price_of(i):g}' for i in self.cart_ids() if self.price_of(i) is not None]
        held = f" ({'; '.join(lines)})" if lines else ''
        return f"Held back (no step spent). {self.left()} steps remain and the pick is the lowest price. A search can miss qualifying listings, and rows arrive in no price order. Before checking out, run a few more searches for each item under other names for it (its plain one-word name, synonyms, words from qualifying titles you have seen) and consider only buyable rows priced below the cart's line{held}; read the specifications of the cheapest of those rows, swap in any one that qualifies, then check out."







    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return []
        texts = []
        for ident in [i for i in ids if i in self.rows and i not in self.full and (not self.sold_out(i)) and (self.read_at.get(i) != self.version)]:
            if self.left() <= keep or self.finished or self.facts_from.get(tool) is False:
                break
            self.read_at[ident] = self.version
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
            reply = None if faults else self.act(tool, args)
            if reply is None:
                break
            seen = reply.get('observation')
            self.observe(tool, seen)
            self.note_view(tool, seen)
            if isinstance(seen, dict) and self.ident(seen):
                self.note_facts(tool, seen)
            texts.append(f'{ident}: ' + self.show(seen)[:1500])
        if texts:
            self.trace.append({'kind': 'fill_facts', 'n': len(texts)})
        return texts

    def title_of(self, ident: str) -> str:
        row = self.rows.get(ident) or {}
        named = next((v for k, v in row.items() if isinstance(v, str) and re.search('title|name|heading', _split_name(k)) and len(v.split()) >= 2), None)
        if named:
            return named
        return max((v for v in row.values() if isinstance(v, str) and 2 <= len(v.split()) <= 30), key=len, default='')

    def sweep_keep(self) -> int:
        return max(SWEEP_KEEP, self.tail() + 6) + 2 * len(self.items)

    def sweep_ready(self) -> bool:
        if self.finished or self.broke or self.no_credit or self.sweeps >= SWEEP_RUNS or (not self.price_key) or (not self.category_key()):
            return False
        return self.search_template() is not None and self.left() > self.sweep_keep() + 5

    def sweep_due(self) -> bool:
        if self.elapsed() > SWEEP_UNTIL or self.cost > SPEND_SOFT or (not self.sweep_ready()):
            return False
        if self.roles.get('ask') and (not self.asked) and self.msgs_free() > 0:
            return False
        if self.brief.get('pick') == 'cheapest':
            return any((n not in self.swept for n in range(1, len(self.items) + 1)))
        return bool(self.sweep_targets())

    def pointed(self, number: int) -> bool:
        live = {need for n, need in self.needs_now() if n == number}
        return any((n == number and need in live and need not in self.ref_values for n, need in self.refs))

    def sweep_targets(self) -> list[int]:
        late = (self.step >= self.max_steps // 2 or self.elapsed() >= WRAP_AT / 2) and self.elapsed() < SWEEP_GAP_UNTIL
        held = self.cart_ids()
        lost = [n for n in range(1, len(self.items) + 1) if [i for i in held if i in self.item_ids(n)] and all((self.sold_out(i) for i in held if i in self.item_ids(n))) and (not [i for i in self.swap_fits if i in self.item_ids(n)])]
        wanted = (self.gaps() if late else []) + lost
        return [n for n in dict.fromkeys(wanted) if n not in self.swept and (not self.pointed(n))]

    def item_cap(self, number: int) -> float | None:
        item = self.items[number - 1]
        for need in item.musts + item.told:
            if need in item.dropped or ':' not in need:
                continue
            prop, value = need.split(':', 1)
            if re.search('price|cost|pay|spend', prop, re.I) and re.search('at most|under|below|no more|not over|up to|max|or less|top', value, re.I):
                found = re.search('\\d[\\d,]*(?:\\.\\d+)?', value)
                if found:
                    return float(found.group(0).replace(',', ''))
        return None

    def within_budget(self, number: int) -> list[str]:
        quantity = max(1, self.items[number - 1].qty or 1)
        each = self.brief.get('budget_scope') == 'each'
        out = []
        for ident in self.item_ids(number):
            price = self.price_of(ident)
            if price is None or not self.buyable(ident) or self.sold_out(ident):
                continue
            if isinstance(self.budget, (int, float)) and (price if each else price * quantity) > self.budget + 1e-09:
                continue
            out.append(ident)
        cap = self.item_cap(number)
        under = [i for i in out if (self.price_of(i) or 0.0) <= cap + 1e-09] if cap is not None else out
        return sorted(under if under or not out or cap is None or cap * 20 >= min((self.price_of(i) or 0.0 for i in out)) else out, key=lambda i: self.price_of(i) or 0.0)

    def sweep_read(self, words: str, room: int) -> int:
        plan = self.search_template()
        if not plan or room <= 0:
            return 0
        tool, base, field = plan
        faults: list[str] = []
        args = _shape({**base, field: words}, self.schemas.get(tool), 'args', faults)
        used = 0
        while not faults and used < room and self.elapsed() < SWEEP_UNTIL + 30 and (not self.finished):
            reply = self.act(tool, args)
            if reply is None:
                break
            used += 1
            seen = reply.get('observation')
            self.observe(tool, seen)
            if not self.records_in(seen) or self.price_sorted(seen):
                break
            after = self.paging(tool, seen, args)
            if not after or after[3] is None or used >= (2 * SWEEP_PAGES if after[0] <= 7 * after[1] else SWEEP_PAGES):
                break
            args = _shape({**args, after[2]: after[3]}, self.schemas.get(tool), 'args', faults)
        self.trace.append({'kind': 'sweep_read', 'words': words[:60], 'calls': used})
        return used

    def row_hits(self, number: int, ident: str) -> int:
        text = ' ' + ' '.join(re.findall('[a-z0-9]+(?:\\.[0-9]+)?', _dump(self.rows.get(ident) or {}).lower())) + ' '
        hits = 0
        for value in self.value_words(number):
            words = [w for w in re.findall('[a-z0-9]+(?:\\.[0-9]+)?', value.lower()) if len(w) >= 3 or w.isdigit()]
            if words and all((f' {w}' in text for w in words)):
                hits += 1
        return hits

    def cheap_words(self, number: int) -> list[str]:
        item = self.items[number - 1]
        rows = self.within_budget(number)
        noun = set(_stem(item.name)) | set(_stem(item.shelf or ''))
        prices: dict[str, list[float]] = {}
        for ident in rows:
            for word in set(re.findall('[a-z][a-z-]{3,}', self.title_of(ident).lower())):
                if word not in noun and not set(_stem(word)) & noun:
                    prices.setdefault(word, []).append(self.price_of(ident) or 0.0)
        limit = max(3, int(0.4 * len(rows)))
        ranked = sorted((w for w, p in prices.items() if len(p) <= limit), key=lambda w: sum(prices[w]) / len(prices[w]))
        return [f'{w} {item.name}' for w in ranked[:8]]

    def read_lowest(self, ids: list[str], keep: int, until: float=SWEEP_UNTIL + 30) -> int:
        tool, many = (self.roles.get('compare'), self.compare_target())
        done = 0
        if tool and many and self.compare_facts() is not False:
            room = max(1, self.compare_room())
            for start in range(0, len(ids), room):
                if self.left() <= keep or self.finished or self.elapsed() > until:
                    break
                faults: list[str] = []
                args = _shape({many: ids[start:start + room]}, self.schemas.get(tool), 'args', faults)
                reply = None if faults else self.act(tool, args)
                if reply is None:
                    break
                seen = reply.get('observation')
                self.observe(tool, seen)
                self.note_view(tool, seen)
                found = self.records_in(seen)
                if found:
                    self.facts_from[tool] = self.facts_from.get(tool, False) or any((_fact_block(r) for _, r in found))
                    done += len(found)
                if not found or self.facts_from.get(tool) is False:
                    break
            if done:
                return done
        return len(self.fill_facts(ids, keep))

    def sift(self, number: int, ids: list[str]) -> list[str]:
        if not ids or self.cost > SPEND_HARD:
            return []
        item = self.items[number - 1]
        records = '\n'.join((f'{i}: ' + _dump(_clip(self.full.get(i) or self.rows.get(i) or {}, 300))[:1600] for i in ids))
        prompt = SIFT + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger() + self.dropped_note() + f'\n\nTHE PRODUCT: item {number}, {item.name}\n\nRECORDS, LOWEST PRICE FIRST:\n' + records
        got = self.ask_json(prompt, [LIGHT], 2500) or {}
        fits = [str(e.get('id') if isinstance(e, dict) else e).strip() for e in _items(got.get('fits'))]
        return sorted((i for i in dict.fromkeys(fits) if i in ids), key=lambda i: self.price_of(i) or 0.0)

    def sweep_item(self, number: int, room: int, until: float=SWEEP_UNTIL + 30) -> str:
        item = self.items[number - 1]
        start = self.step
        reading = max(2, room * 2 // 3)
        queue = [f'{value} {item.name}' for value in self.value_words(number)[:3] if len(value.split()) <= 3] + [item.name]
        tried = 0
        while self.step - start < reading and self.elapsed() < min(SWEEP_UNTIL, until - 12) and (not self.finished):
            if not queue:
                queue = [w for w in self.cheap_words(number) if ' '.join(w.lower().split()) not in self.sweep_tried]
                if not queue or tried >= 12:
                    break
            words = ' '.join(queue.pop(0).split())
            if words.lower() in self.sweep_tried:
                continue
            self.sweep_tried.add(words.lower())
            tried += 1
            self.sweep_read(words, reading - (self.step - start))
        keep = max(self.sweep_keep(), self.left() - (room - (self.step - start)))
        fits: list[str] = []
        judged = 0
        cheap = self.brief.get('pick') == 'cheapest'
        for _ in range(2):
            pool = self.within_budget(number)
            if not cheap:
                pool.sort(key=lambda i: -self.row_hits(number, i))
            lowest = pool[:SWEEP_OPEN * (judged // SWEEP_OPEN + 1)]
            fresh = lowest[judged:]
            if not fresh:
                break
            self.read_lowest([i for i in fresh if i not in self.full], keep, until)
            fits = list(dict.fromkeys(fits + self.sift(number, [i for i in fresh if i in self.full])))
            judged += len(fresh)
            if len(fits) >= 2 or self.left() <= keep or self.elapsed() > until or self.cost > SPEND_SOFT or self.finished:
                break
        self.trace.append({'kind': 'sweep', 'item': number, 'steps': self.step - start, 'rows': len(self.item_ids(number)), 'read': judged, 'fits': fits[:6]})
        if fits:
            self.sweep_fits[number] = fits
        known = len(self.within_budget(number))
        if fits:
            return f'- item {number} ({item.name}): {known} rows within budget known; of the {judged} lowest-priced read, these fit, lowest first: ' + ', '.join((f'{i} at {self.price_of(i):g}' for i in fits[:5]))
        return ''

    def sweep(self, numbers: list[int] | None=None) -> str:
        todo = [n for n in (numbers or range(1, len(self.items) + 1)) if n not in self.swept and (not self.pointed(n))]
        if not todo:
            return ''
        self.sweeps += 1
        lines = []
        for spot, number in enumerate(todo):
            self.swept.add(number)
            room = (self.left() - self.sweep_keep()) // (len(todo) - spot)
            until = self.elapsed() + max(0.0, SWEEP_UNTIL + 30 - self.elapsed()) / (len(todo) - spot)
            if room >= 4 and self.elapsed() < SWEEP_UNTIL + 30 and (not self.finished):
                lines.append(self.sweep_item(number, room, until))
        if self.brief.get('pick') == 'cheapest':
            self.wider_held = True
        lines = [line for line in lines if line]
        if not lines:
            return ''
        return '[Sable read further searches and the lowest-priced listings itself, one step each]\n' + '\n'.join(lines) + '\nThese come from a first reading of the records: confirm a listing on its record before carting it.'

    def cheaper_rows(self) -> list[str]:
        if self.brief.get('pick') != 'cheapest':
            return []
        mark = self.cart_mark() + f'#{self.version}'
        cat_key = self.category_key()
        if mark in self.cheaper_marks or not cat_key:
            return []
        self.cheaper_marks.add(mark)
        cart = self.cart_ids()

        def shelf(ident: str) -> str:
            return str((self.rows.get(ident) or {}).get(cat_key) or '').strip().lower()

        def cost(ident: str) -> float:
            price = self.price_of(ident)
            return price if price is not None else float('inf')
        picks: list[str] = []
        for line in cart:
            price, kind = (self.price_of(line), shelf(line))
            if price is None or not kind:
                continue
            pool = sorted((i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i) and shelf(i) == kind and cost(i) < price), key=lambda i: self.price_of(i) or 0.0)
            picks += [i for i in pool[:3 if self.elapsed() < WRAP_AT - 30 else 2] if i not in picks]
        return self.fill_facts(picks[:2 * CHEAPER_OPENS if self.elapsed() < WRAP_AT - 30 else CHEAPER_OPENS], self.tail() + 4)

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
        limit = props[target].get('maxItems')
        cap = limit if isinstance(limit, int) else 4
        groups = self.item_groups()
        share = max(1, cap // max(1, len(groups)))
        picks = [i for ids in groups for i in [x for x in ids if x not in self.full][:share]][:cap]
        if len(picks) < 2:
            return
        faults: list[str] = []
        args = _shape({target: picks}, schema, 'args', faults)
        if faults:
            return
        before = len(self.full)
        reply = self.act(tool, args)
        if reply is None:
            return
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
            again = None if faults else self.act(tool, args)
            if again is not None:
                self.observe(tool, again.get('observation'))
        if len(self.full) == before and self.left() > 12 and (not self.finished):
            self.widen_singly()

    def widen_singly(self) -> None:
        tool = self.facts_tool()
        if not tool or not self.single_target(tool):
            return
        wanted = {w for _, need in self.needs_now() for w in re.findall('[a-z0-9]{3,}', need.split(':', 1)[-1].lower())}
        each = WIDEN_EACH if len(self.items) <= 2 else 4
        lanes: list[list[str]] = []
        heads: list[int] = []
        for ids in self.item_groups():
            opened = len([i for i in ids if i in self.full])
            pool = [i for i in ids if i not in self.full and self.buyable(i)]
            if self.brief.get('pick') == 'cheapest':
                pool.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float('inf'))
            pool.sort(key=lambda i: -len(wanted & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower()))))
            lanes.append(pool[:max(0, each - opened)])
            heads.append(max(0, 4 - opened))
        pairs = list(zip(lanes, heads))
        first = [lane[k] for k in range(4) for lane, head in pairs if k < min(head, len(lane))]
        more = [lane[k] for k in range(each) for lane, head in pairs if head <= k < len(lane)]
        cap = max(5, 2 * len(lanes) + 2)
        read = self.fill_facts(first[:cap][:8], 22)
        self.fill_facts((first[cap:] + more)[:max(0, 10 - len(read))], 30)


    def ready_to_ask(self) -> bool:
        if self.asked or not self.roles.get('ask') or self.msgs_free() <= 0 or self.left() <= 8:
            return False
        return len(self.full) >= 3 * len(self.items) + 1 or self.step >= max(8, self.max_steps // 4)

    def message_args(self, text: str) -> dict[str, Any] | None:
        schema = self.schemas.get(self.roles.get('ask') or '') or {}
        props = schema.get('properties') or {}
        needed = list(schema.get('required') or [])
        free = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        target = next((k for k in needed if k in free), None) or (free[0] if free else None)
        if not target:
            return None
        args: dict[str, Any] = {target: text}
        for key in needed:
            if key in args:
                continue
            options = (props.get(key) or {}).get('enum')
            if not (isinstance(options, list) and options):
                return None
            args[key] = options[0]
        faults: list[str] = []
        args = _shape(args, schema, 'args', faults)
        return None if faults else args

    def interview(self) -> str:
        self.asked = True
        if self.msgs_free() <= 0:
            return ''
        self.widen()
        lanes = self.plan_lanes(max(self.msgs_free(), PLAN_SIZE))
        tally = {n: {'found': 0, 'spent': 0, 'used': set()} for n in lanes}
        refilled = False
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        self.trace.append({'kind': 'ask_mode', 'mode': 'single', 'free': self.msgs_free()})
        while self.msgs_free() > 0 and (not self.finished) and (self.left() > 6):
            number, labels = self.next_ask(lanes, tally)
            if not labels and (not refilled) and self.elapsed() < CALL_STOP - 30:
                refilled = True
                self.refill(lanes, tally)
                number, labels = self.next_ask(lanes, tally)
            if not labels:
                break
            tally[number]['used'].update((_words(label) for label in labels))
            batch = self.send([(number, labels[0], self.question(number, labels[0]))])
            if not batch:
                break
            sent += batch
            tally[number]['spent'] += 1
            before = [len(item.told) for item in self.items]
            marks = (len(told), len(self.refs))
            heard = self.hear(batch, (LIGHT, HEAVY))
            self.take_heard(heard, told)
            gained = [n for n, item in enumerate(self.items, 1) if len(item.told) > before[n - 1]]
            fresh = self.items[number - 1].told[before[number - 1]:]
            odd = bool(gained) and (not any((self.same_key(label, str(x).split(':', 1)[0]) for x in fresh for label in labels)))
            if odd and self.elapsed() < CALL_STOP - 20 and self.wants(batch[0][1], batch[0][2]) is False:
                for n, item in enumerate(self.items, 1):
                    del item.told[before[n - 1]:]
                del told[marks[0]:]
                del self.refs[marks[1]:]
                gained = []
            for n in gained:
                tally[n]['found'] += 1
        routed = self.chase()
        hits = [n for n in tally if tally[n]['found'] > 0]
        searched = self.value_search(hits) if hits else ''
        if told:
            self.version += 1
        return self.interview_summary(sent, told, routed, searched)

    def refill(self, lanes: dict[int, list[str]], tally: dict[int, dict[str, Any]]) -> None:
        asked = sorted({label for n in tally for label in tally[n]['used'] | set(tally[n].get('sent', {}))})
        fresh = self.plan_lanes(PLAN_SIZE, asked)
        for number, labels in fresh.items():
            for label in labels:
                if _words(label) not in tally[number]['used'] and _words(label) not in {_words(x) for x in lanes[number]}:
                    lanes[number].append(label)
        self.trace.append({'kind': 'refill', 'asked': asked[:12], 'lanes': {n: v[-6:] for n, v in lanes.items()}})

    def interview_summary(self, sent: list[tuple[int, str, str]], told: list[str], routed: str, searched: str) -> str:
        text = "Sable's interviewer put these questions to the shopper (one step each):\n" + '\n'.join((f'Q: {q}\nA: {a}' for _, q, a in sent))
        text += '\nFirm requirements added by these replies: ' + ('; '.join(told) if told else 'none')
        text += '. Hold every pick to them (run new searches with these values when needed).'
        pointed = [need for _, need in self.refs if any((need in t for t in told)) and need not in self.ref_values]
        if routed:
            text += '\n' + routed
        if pointed:
            text += ' These get their value from some other listing: ' + ' | '.join(pointed) + '. Locate that listing, open its full page, and open each listing that page sends you to; the final page holds the value, and the order only counts once every one of them has been opened.'
        if searched:
            text += '\n' + searched
        if self.queued:
            text += '\n[notes seen meanwhile] ' + ' | '.join(self.queued)
            self.queued = []
        return text

    def search_template(self) -> tuple[str, dict[str, Any], str] | None:
        tool = self.roles.get('search')
        if not tool:
            return None
        schema = self.schemas.get(tool) or {}
        props = schema.get('properties') or {}
        required = schema.get('required') or []
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        named = [k for k in texts if re.search(QUERY_FIELD, _split_name(k), re.I)]
        for args in reversed(self.searched):
            field = self.query_field(args, named, texts)
            if field:
                return (tool, self.kept_filters(args, field, props, required), field)
        if len(named) == 1 and (not self.searched) and set(required) <= set(named):
            return (tool, {}, named[0])
        return self.browse_plan(tool, props, required, named, texts)

    def browse_plan(self, tool: str, props: dict[str, Any], required: list[str], named: list[str], texts: list[str]) -> tuple[str, dict[str, Any], str] | None:
        spare = [k for k in named or texts if not any((a.get(k) for a in self.searched))]
        if not self.searched or not spare:
            return None
        last = self.searched[-1]
        modes: dict[str, Any] = {}
        for k in last:
            options = (props.get(k) or {}).get('enum')
            unused = [o for o in options if all((_dump(a.get(k)) != _dump(o) for a in self.searched))] if isinstance(options, list) else []
            if unused and (k in required or len(unused) == 1):
                modes[k] = unused[0]
        if not named and (len(spare) > 1 or not modes):
            return None
        return (tool, {**self.kept_filters(last, spare[0], props, required), **modes}, spare[0])

    def query_field(self, args: dict[str, Any], named: list[str], texts: list[str]) -> str | None:
        fields = [k for k in named or texts if isinstance(args.get(k), str) and re.search('[a-z]', args[k], re.I)]
        if not named and fields:
            cat_key = self.category_key()
            labels = {str(r.get(cat_key) or '').strip().lower() for r in self.rows.values()} - {''} if cat_key else set()

            def free_text(k: str) -> bool:
                said = [a.get(k) for a in self.searched]
                return len({str(v) for v in said if v}) >= 2 and any((len(str(v).split()) >= 2 for v in said)) and (not any((str(v or '').strip().lower() in labels for v in said)))
            fields = [k for k in fields if free_text(k)]
        return fields[0] if fields else None

    def kept_filters(self, args: dict[str, Any], field: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
        def menu(k: str) -> bool:
            return isinstance((props.get(k) or {}).get('enum'), list)
        keep = {k: v for k, v in args.items() if k != field and (menu(k) or k in required)}
        cat_key = self.category_key()
        shelves = {_words(item.shelf or '') for item in self.items} | {_words(item.name) for item in self.items}
        if cat_key:
            shelves |= {_words(str(row.get(cat_key) or '')) for row in self.rows.values()}
        long_shelves = [{w for w in _stem(s) if len(w) >= 4} for s in shelves if s]

        def shelf_like(value: str) -> bool:
            mine = {w for w in _stem(value) if len(w) >= 4}
            return _words(value) in shelves or any((mine & other for other in long_shelves))
        keep = {k: v for k, v in keep.items() if k in required or not isinstance(v, str) or (not shelf_like(v))}
        worded = [a for a in self.searched if isinstance(a.get(field), str) and re.search('[a-z]', a[field], re.I)]
        for k in [k for k in keep if menu(k)]:
            used = [_dump(a[k]) for a in worded if k in a]
            if used:
                best = max(used, key=used.count)
                keep[k] = next((a[k] for a in worded if k in a and _dump(a[k]) == best))
        return keep

    def value_words(self, number: int) -> list[str]:
        pointed = {m for _, m in self.refs}
        out = []
        for n, need in self.needs_now():
            if n != number or need in pointed or ':' not in need:
                continue
            value = _bound_value(need.split(':', 1)[1].strip())
            if value and value not in out and (not re.search(VALUE_SKIP, value, re.I)):
                out.append(value)
        return out

    def value_search(self, numbers: list[int]) -> str:
        plan = self.search_template()
        room = self.left() - self.tail()
        if not plan or room < 8:
            return ''
        tool, base, field = plan
        notes: list[str] = []
        allowance = min(8, room // 2)
        keep = self.tail() + min(16, room - allowance)
        for number in numbers:
            values = self.value_words(number)
            item = self.items[number - 1]
            heard = [v for v in values if any((':' in t and _bound_value(t.split(':', 1)[1].strip()) == v for t in item.told))]
            if not heard or allowance <= 2:
                continue
            stated = [v for v in values if v not in heard]
            attempts = [heard + stated + [item.name], heard + [item.name]] if stated else [heard + [item.name]]
            if self.long_empty:
                attempts.reverse()
            for words in attempts[:2]:
                if not words:
                    continue
                faults: list[str] = []
                args = _shape({**base, field: ' '.join(words)}, self.schemas.get(tool), 'args', faults)
                if faults or self.finished:
                    break
                reply = self.act(tool, args)
                allowance -= 1
                if reply is None:
                    break
                rows = self.kind_rows(item, tool, reply.get('observation'))
                self.long_empty = self.long_empty or (not rows and len(words) >= 3)
                notes.append(f'{_dump(args.get(field))}: ' + (', '.join(rows[:6]) or 'no row of that kind'))
                if rows:
                    opened = self.fill_facts([i for i in rows if i not in self.full][:min(3, allowance)], keep)
                    allowance -= len(opened)
                    if opened:
                        notes.append('opened: ' + ' | '.join(opened)[:2400])
                    break
        if not notes:
            return ''
        self.trace.append({'kind': 'value_search', 'seen': [s[:200] for s in notes]})
        return "[Sable ran a search on each such item's requirement values and read the first rows of the right kind] " + '\n'.join(notes)

    def kind_rows(self, item: Item, tool: str, observation: Any) -> list[str]:
        self.observe(tool, observation)
        cat_key = self.category_key()
        found = [i for i, _ in self.records_in(observation)]
        cap = self.budget if isinstance(self.budget, (int, float)) else None
        prices = [self.price_of(i) for i in found]
        if cap is not None and prices and all((p is not None and p > 10 * cap for p in prices)):
            cap = None
        rows = [i for i in found if self.buyable(i) and (not cat_key or self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))) and (cap is None or (self.price_of(i) or 0) <= cap)]
        return sorted(rows, key=lambda i: self.price_of(i) or 0) if self.brief.get('pick') == 'cheapest' else rows



    def covers(self, label: str, heard: str) -> bool:
        a, b = _words(label).split(), _words(heard).split()
        return bool(b) and all((any((x == y or (min(len(x), len(y)) >= 4 and (x.startswith(y) or y.startswith(x))) for x in a)) for y in b))

    def same_key(self, label: str, name: str) -> bool:
        a, b = _words(label).split(), _words(name).split()
        if not a or not b:
            return False
        short, long = (a, b) if len(a) <= len(b) else (b, a)
        return all((any((x == y or (min(len(x), len(y)) >= 4 and (x.startswith(y) or y.startswith(x))) for y in long)) for x in short))

    def wants(self, question: str, answer: str) -> bool | None:
        self.effort = 'low'
        try:
            got = self.ask_json(WANTS + '\n\nMESSAGE: ' + question + '\n\nREPLY: ' + str(answer), [LIGHT, HEAVY], 1500)
        finally:
            self.effort = ''
        if not isinstance(got, dict) or not isinstance(got.get('wants'), bool):
            return None
        self.trace.append({'kind': 'wants_check', 'wants': got['wants'], 'reply': str(answer)[:120]})
        return got['wants']

    def next_ask(self, lanes: dict[int, list[str]], tally: dict[int, dict[str, Any]]) -> tuple[int, list[str]]:
        rank = {n: spot for spot, n in enumerate(self.plan_order)}
        for number in sorted(lanes, key=lambda n: (tally[n]['found'] > 0, tally[n]['spent'], rank.get(n, len(rank) + n))):
            used = {_squash(label) for label in tally[number]['used']}
            heard = [str(x).split(':', 1)[0] for x in self.items[number - 1].told]
            open_labels = [label for label in lanes[number] if _words(label) not in tally[number]['used'] and _squash(label) not in used and (not any((self.covers(label, h) for h in heard)))]
            if open_labels:
                return (number, open_labels[:1])
        return (0, [])

    def lane_keys(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        fixed = [_stem(must.split(':', 1)[0]) for must in self.items[number - 1].musts]
        numeric: dict[str, bool] = {}
        for _, key, _, values in table:
            numeric[key] = 2 * len([v for v in values if re.match(NUMERIC_VALUE, v)]) > len(values)
        mine = [entry for entry in table if self.owner_of(entry[0]) == number]
        if not mine and len(self.items) == 1:
            mine = list(table)
        ranked = sorted(mine, key=lambda entry: (-entry[2], numeric.get(entry[1], False), -len(entry[3])))
        usable = []
        for _, key, _, _ in ranked:
            if re.search(NOT_PROPERTY, key) or re.search(LOGISTICS, key) or any((_names(prop, _stem(key)) for prop in fixed)):
                continue
            usable.append(key)
        broad = [key for key in usable if re.search(CATCH_ALL, key)]
        return [key for key in usable if not re.search(CATCH_ALL, key)] + broad

    def owner_of(self, label: str) -> int:
        if not label:
            return 0
        words = _words(label)
        long = _roots(label)
        best, top = (0, 0.0)
        flat = _squash(label)
        for number, item in enumerate(self.items, 1):
            if words == _words(item.shelf or '') or (item.shelf and flat == _squash(item.shelf)):
                score = 3.0
            elif words == _words(item.name) or flat == _squash(item.name):
                score = 2.0
            else:
                mine = _roots(item.shelf or '') | _roots(item.name)
                score = len(long & mine) / max(1, len(long | mine))
            if score > top:
                best, top = (number, score)
        return best

    def owns(self, item: Item, label: str) -> bool:
        if not label or item not in self.items or self.owner_of(label) != self.items.index(item) + 1:
            return False
        shelf = _squash(item.shelf)
        if shelf:
            return _squash(label) == shelf or shelf not in self.departments()
        home = self.home_of(item)
        return not home or _squash(label) == _squash(home)

    def home_of(self, item: Item) -> str:
        cat_key = self.category_key()
        mark = (len(self.rows), cat_key, len(self.items))
        if self.home_cache[0] != mark:
            labels: dict[str, str] = {}
            for row in self.rows.values() if cat_key else []:
                label = ' '.join(str(row.get(cat_key) or '').split())
                if label:
                    labels.setdefault(_squash(label), label)
            homes: dict[str, str] = {}
            for other in self.items:
                roots = _roots(other.name)
                found = sorted({label for label in labels.values() if roots and _roots(label) == roots})
                if len(found) == 1:
                    homes[other.name] = found[0]
            self.home_cache = (mark, homes)
        return '' if item.shelf else self.home_cache[1].get(item.name, '')

    def departments(self) -> set[str]:
        cat_key = self.category_key()
        mark = (len(self.rows), cat_key)
        if self.dept_cache[0] != mark:
            found = {_squash(r.get(cat_key)) for r in self.rows.values()} if cat_key else set()
            for schema in self.schemas.values():
                for spec in ((schema or {}).get('properties') or {}).values():
                    options = spec.get('enum') if isinstance(spec, dict) else None
                    if isinstance(options, list) and len(options) >= 3 and all((isinstance(o, str) for o in options)):
                        found |= {_squash(o) for o in options}
            self.dept_cache = (mark, found - {''})
        return self.dept_cache[1]

    def records_of_item(self, number: int) -> int:
        cat_key = self.category_key()
        item = self.items[number - 1]
        return sum((1 for record in self.full.values() if cat_key and self.owns(item, str(record.get(cat_key) or '')) and _fact_block(record)))

    def plan_lanes(self, count: int, asked: list[str] | None=None) -> dict[int, list[str]]:
        table = self.key_table()
        prompt = self.plan_prompt(count, table)
        if asked:
            prompt += '\n\nALREADY ASKED, so pick other properties than these:\n' + ', '.join(asked)
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 6000) or {}
        lanes, hinted = self.read_picks(plan)
        for number in lanes:
            keys = self.lane_keys(number, table)
            share = self.key_shares(number, table)
            lane = self.guarded_lane(number, lanes[number], hinted[number], keys, share)
            known = {_words(label) for label in lane}
            for key in keys:
                if _words(key) not in known:
                    lane.append(key)
                    known.add(_words(key))
            lane = [re.sub('(?<=[a-z])\\s+\\d+$', '', x) for x in lane]
            lane = [x for spot, x in enumerate(lane) if _squash(x) not in {_squash(y) for y in lane[:spot]}]
            named = [x for x in lane if self.named_in_story(x)]
            lanes[number] = [x for x in lane if not re.search(CATCH_ALL, _words(x)) and x not in named] + [x for x in named if not re.search(CATCH_ALL, _words(x))] + [x for x in lane if re.search(CATCH_ALL, _words(x))]
        self.trace.append({'kind': 'plan', 'ask': _clip(plan.get('ask'), 120), 'lanes': {n: v[:8] for n, v in lanes.items()}})
        return lanes

    def named_in_story(self, label: str) -> bool:
        words = _words(label)
        text = f' {_words(self.request)} '
        for item in self.items:
            for name in (item.name, item.shelf or ''):
                base = _words(name)
                for form in {base, base + 's', base[:-1] if base.endswith('s') else base}:
                    if form.strip():
                        text = text.replace(f' {form} ', ' ')
        if not words or f' {words} ' not in text:
            return False
        stated = [_words(m.split(':', 1)[0]) for item in self.items for m in item.musts + item.told]
        early = ' '.join((_words(str(x)) for x in _items(self.brief.get('ask_first'))))
        return not any((s and (words == s or f' {words} ' in f' {s} ' or f' {s} ' in f' {words} ') for s in stated)) and f' {words} ' not in f' {early} '

    def plan_prompt(self, count: int, table: list[tuple[str, str, int, list[str]]]) -> str:
        fixed = [f'item {n}: {item.name} - settled by the request: ' + ('; '.join(item.musts) or 'nothing') for n, item in enumerate(self.items, 1)]
        shown: list[str] = []
        group = None
        for position, (label, name, carried, values) in enumerate(table, 1):
            if label != group:
                group = label
                shown.append(f'category {label!r}:')
            tag = self.not_stated(label, name, values)
            if re.search(CATCH_ALL, name):
                tag += ' [several features in one: ask last]'
            shown.append(f'  {position}. {name} (on {carried}): ' + ' | '.join(values) + tag)
        text = PLAN.replace('COUNT', str(max(1, count))) + '\n\nREQUEST:\n' + self.request + '\n\nITEMS:\n' + '\n'.join(fixed)
        early = [str(x) for x in _items(self.brief.get('ask_first')) if str(x).strip()]
        if early:
            text += '\n\nASK FIRST (the request wants the customer consulted on these):\n' + '\n'.join(early)
        return text + '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(shown) if shown else '(no full listing records yet, so after any hinted properties list the ones a buyer of each kind of product usually chooses between, up to the limit for every item)')

    def read_picks(self, plan: dict[str, Any]) -> tuple[dict[int, list[str]], dict[int, set[str]]]:
        size = len(self.items)
        lanes: dict[int, list[str]] = {n: [] for n in range(1, size + 1)}
        hinted: dict[int, set[str]] = {n: set() for n in lanes}
        for pick in _items(plan.get('ask')):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, _count(pick.get('item') or 1) or 1), size)
            label = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(pick.get('label') or ''))[0].strip().rstrip('?.')
            label = re.sub('(?<=[a-z])\\s+\\d+$', '', label)
            if not label or len(label.split()) > 5 or re.search(LOGISTICS, _words(label)) or re.search(NOT_PROPERTY, _words(label)):
                continue
            if _words(label) not in {_words(other) for other in lanes[number]}:
                lanes[number].append(label)
            if str(pick.get('hint') or '').strip():
                hinted[number].add(_words(label))
                if number not in self.plan_order:
                    self.plan_order.append(number)
        return (lanes, hinted)

    def key_shares(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> dict[str, float]:
        read = max(1, self.records_of_item(number))
        share: dict[str, float] = {}
        for label, key, carried, _ in table:
            if self.owner_of(label) == number:
                share[key] = max(share.get(key, 0.0), carried / read)
        return share

    def guarded_lane(self, number: int, picks: list[str], hints: set[str], keys: list[str], share: dict[str, float]) -> list[str]:
        def plain(text: str) -> str:
            return re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', _words(text)))

        def spelled(label: str) -> str:
            return next((key for key in share if plain(key) == plain(label)), label)

        def carried(label: str) -> float:
            if _words(label) in share:
                return share[_words(label)]
            close = [s for key, s in share.items() if _names(_stem(label), _stem(key)) or _names(_stem(key), _stem(label))]
            return max(close) if close else 0.0
        hints = {_words(spelled(h)) for h in hints}
        lane: list[str] = []
        for pick in picks:
            name = spelled(pick)
            if _words(name) not in {_words(other) for other in lane}:
                lane.append(name)
        guards = [key for key in keys if share.get(key, 0.0) >= GUARD_SHARE and (not re.search(CATCH_ALL, key))][:GUARD_KEYS]
        for key in guards:
            top = lane[:3]
            if _words(key) in {_words(other) for other in top}:
                continue
            weak = next((spot for spot, other in enumerate(top) if _words(other) not in hints and carried(other) + GUARD_GAP <= share.get(key, 0.0)), None)
            if weak is None and len(lane) >= 3:
                continue
            trial = [other for other in lane if _words(other) != _words(key)]
            if weak is None:
                trial.append(key)
            elif len(self.items) > 1:
                bumped = trial[weak]
                trial[weak] = key
                trial.insert(min(3, len(trial)), bumped)
            else:
                trial.insert(min(len(trial), weak), key)
            kept = {_words(y) for y in trial[:3]}
            if any((_words(x) in hints and _words(x) not in kept for x in top)):
                continue
            lane = trial
        return lane

    def question(self, number: int, label: str) -> str:
        name = re.sub(',\\s*(?:for example|for instance|e\\.g\\.|such as)\\b.*$', '', self.items[number - 1].name, flags=re.I).strip()
        name = re.sub('\\s+(?:and|or)\\s+', ' & ', name) or self.items[number - 1].name
        return f'What {label} would you like for the {name}?'


    def send(self, questions: list[tuple[int, str, str]], floor: int=6) -> list[tuple[int, str, str]]:
        tool = self.roles['ask']
        sent: list[tuple[int, str, str]] = []
        for number, _, text in questions:
            if self.finished or self.left() <= floor or self.msg_used >= self.msg_limit:
                break
            args = self.message_args(text)
            reply = self.act(tool, args) if args else None
            if reply is None:
                break
            self.msg_used += 1
            answer, told = _payload(reply.get('observation'))
            if len(self.spoken(answer).split()) < 3:
                answer, told = (reply.get('observation'), [])
            words = self.spoken(answer)
            sent.append((number, text, words))
            fresh = [line for line in told if line not in self.heard]
            meaning = self.take_note(fresh) if fresh else ''
            if meaning:
                self.queued.append(f'(seen on {tool}) {meaning}')
            self.done_log.append(f'step {self.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {self.cart_mark()})')
            self.done_steps.append(self.step)
            self.trace.append({'kind': 'ask', 'q': text, 'a': words[:300]})
            if _trouble(answer):
                break
        return sent

    def slot_of(self, entry: dict[str, Any]) -> int:
        return min(max(1, _count(entry.get('item')) or 1), len(self.items))

    def item_named_in(self, text: str) -> int:
        def stems(words: str) -> str:
            return ' ' + ' '.join((w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w for w in re.findall('[a-z0-9]+', words.lower()))) + ' '
        said = stems(text)
        hits = sorted({(said.find(stems(phrase)), -len(stems(phrase).split()), number) for number, item in enumerate(self.items, 1) for phrase in (item.shelf, item.name) if phrase and stems(phrase).strip() and stems(phrase) in said})
        if not hits or (len(hits) > 1 and hits[1][:2] == hits[0][:2] and hits[1][2] != hits[0][2]):
            return 0
        return hits[0][2]

    def take_heard(self, heard: dict[str, Any], told: list[str]) -> None:
        many = len(self.items) > 1
        for entry in _items(heard.get('needs')):
            if not isinstance(entry, dict) or not str(entry.get('need') or '').strip():
                continue
            number = self.slot_of(entry)
            need = str(entry['need']).strip()
            words = re.findall('[a-z0-9]+', need.split(':', 1)[0].lower())
            source = [a for a in getattr(self, 'heard_answers', []) if words and all((w in a.lower() for w in words))]
            if many and len(source) == 1:
                named = self.item_named_in(source[0])
                if named and named != number:
                    self.trace.append({'kind': 'need_moved', 'need': need[:80], 'from': number, 'to': named})
                    number = named
            item = self.items[number - 1]
            if need not in item.told:
                item.told.append(need)
                told.append(f'item {number}: {need}' if many else need)
            if entry.get('points') is True and (number, need) not in self.refs:
                self.refs.append((number, need))
        for entry in _items(heard.get('drop')):
            if not isinstance(entry, dict) or not str(entry.get('property') or '').strip():
                continue
            item = self.items[self.slot_of(entry) - 1]
            gone = _words(str(entry['property']).split(':')[0])
            hit = [m for m in item.musts + item.told if m not in item.dropped and _words(m.split(':')[0]) == gone]
            item.dropped.extend(hit)
            told.extend((f'dropped {m}' for m in hit))
        for entry in _items(heard.get('free')):
            if isinstance(entry, dict) and entry.get('property'):
                self.items[self.slot_of(entry) - 1].free.append(str(entry['property']))
        budget = heard.get('budget')
        if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and budget > 0:
            self.budget = self.budget_as_written(budget, getattr(self, 'heard_text', '') + '\n' + '\n'.join((str(x) for x in self.heard)))

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(HEAVY, LIGHT)) -> dict[str, Any]:
        roster = '\n'.join((f'item {n}: {item.name}' for n, item in enumerate(self.items, 1)))
        pairs = '\n'.join((f'{i}. (about item {n}) Q: {q}\nA: {a}' for i, (n, q, a) in enumerate(sent, 1)))
        self.heard_text = '\n'.join((str(a) for _, _, a in sent))
        self.heard_answers = [str(a) for _, _, a in sent]
        prompt = HEAR + '\n\nITEMS:\n' + roster + '\n\nQUESTIONS AND ANSWERS:\n' + pairs + '\n\nPRICES SEEN: ' + self.prices_seen()
        return self.ask_json(prompt, list(models), 2500) or {}

    def gate(self, stage: str) -> tuple[bool, str]:
        steps = [p for p in self.process if p['when'] == stage]
        if not steps or self.finished or (not self.gate_due(stage)):
            return (True, '')
        ending = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        menu = '\n'.join((f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_dump(t['function'].get('parameters') or {})}" for t in self.tools if t['function']['name'] not in ending))
        said: list[str] = []
        nudged = False
        for _ in range(GATE_ROUNDS):
            got = self.ask_json(self.gate_prompt(stage, steps, menu), [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 3500)
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
                done = 'Before going on, Sable carried out the steps the request asks for:\n' + '\n'.join(said) + '\n' if said else ''
                return (False, done + "The shopper's reply alters the task: " + '; '.join(told + ([hold] if hold else [])) + '. Adjust the cart to match, then check out again.')
            if not call and (not said) and (not nudged) and self.gate_unproven(stage, steps, got):
                nudged = True
                continue
            name = str((call or {}).get('tool') or '')
            if not call or name not in self.schemas or name in ending:
                if not call:
                    self.gate_ok.add((stage, self.cart_mark() + f'#{self.version}'))
                break
            if self.left() < 2 or (name == self.roles.get('ask') and self.msg_used >= self.msg_limit):
                break
            faults: list[str] = []
            args = _shape(call.get('args') if isinstance(call.get('args'), dict) else {}, self.schemas[name], 'args', faults)
            if faults or not self.gate_step(name, args, said):
                break
        return (True, 'Before going on, Sable carried out the steps the request asks for:\n' + '\n'.join(said) if said else '')

    def gate_prompt(self, stage: str, steps: list[dict[str, Any]], menu: str) -> str:
        record = '\n'.join((f'{i}. {line}' for i, line in enumerate(self.done_log, 1))) or '(nothing yet)'
        notes = '\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.notes[-4:]) if self.notes else ''
        moment = 'order' if stage == 'order' else 'ending'
        return (GATE.replace('STAGE', 'place the order' if stage == 'order' else 'end without a purchase') + '\n\nREQUEST:\n' + self.request + '\n\nSTEPS DUE NOW:\n' + '\n'.join((f"- {p['step']}" for p in steps))
                + '\n\nLEDGER:\n' + self.ledger() + '\n\nCART NOW: ' + self.cart_text() + '\n\nRECORD (messages and step calls up to now):\n' + record + notes
                + f'\n\nMESSAGES LEFT: {max(0, self.msg_limit - self.msg_used)}   STEPS LEFT: {self.left()} (the {moment} itself takes one)' + '\n\nTOOLS:\n' + menu)

    def gate_unproven(self, stage: str, steps: list[dict[str, Any]], got: dict[str, Any]) -> bool:
        shown = [e for e in _items(got.get('steps')) if isinstance(e, dict)]
        talk = stage == 'order' and any((p['messages'] > 0 for p in steps))
        proof = {i for i in range(1, len(self.done_log) + 1) if not talk or (i <= len(self.done_steps) and self.done_steps[i - 1] > self.cart_at)}
        unproven = [e for e in shown if e.get('done') is not True or not any(((_count(n) or 0) in proof for n in _items(e.get('lines'))))]
        if not unproven or not self.roles.get('ask') or self.msg_used >= self.msg_limit or self.left() < 2:
            return False
        self.trace.append({'kind': 'gate_unproven', 'steps': _clip(unproven, 120)})
        self.done_log.append(f'(Sable: no line made since the cart last changed at step {self.cart_at} shows ' + '; '.join((str(e.get('step') or '') for e in unproven))[:200] + f' done, and {self.msg_limit - self.msg_used} message(s) and {self.left()} steps are left: give the call that does it)')
        self.done_steps.append(-1)
        return True

    def gate_step(self, name: str, args: dict[str, Any], said: list[str]) -> bool:
        if name == self.roles.get('ask'):
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
        observation, told = _payload(observation)
        lines: list[str] = [line for line in told if len(line) <= NOTE_CHARS]
        if isinstance(observation, dict) and (not self.ident(observation)):
            for key, value in observation.items():
                if isinstance(value, str) and len(value.split()) >= 5 and len(value) <= NOTE_CHARS:
                    lines.append(value)
                elif isinstance(value, list) and 0 < len(value) <= 4 and all((isinstance(v, str) for v in value)):
                    lines.extend((v for v in value if len(v.split()) >= 5 and len(v) <= NOTE_CHARS))
                elif isinstance(value, dict) and (not self.ident(value, False, 1)):
                    for inner in value.values():
                        if isinstance(inner, str) and len(inner.split()) >= 6:
                            lines.append(inner)
        return [line for line in lines if line not in self.heard]

    def prices_seen(self) -> str:
        prices = sorted({p for p in (self.price_of(i) for i in self.seen_order[-40:]) if p})
        return ', '.join((f'{p:g}' for p in prices[:3] + prices[3:][-2:])) if prices else 'none yet'

    def budget_as_written(self, value: float, text: str) -> float:
        prices = [p for p in (self.price_of(i) for i in self.seen_order[-60:]) if p]
        if prices and value * 20 >= min(prices):
            return value
        for found in re.finditer('(\\d[\\d,]*(?:\\.\\d+)?)\\s*([A-Za-z]*)', text or ''):
            number = float(found.group(1).replace(',', ''))
            if abs(number - value * 100) < 0.6 and (prices or found.group(2).lower().startswith('cent')):
                return number
        return value

    def take_note(self, lines: list[str]) -> str:
        self.heard.update(lines)
        prompt = NOTE + '\n\nITEMS AND REQUIREMENTS NOW:\n' + self.ledger() + '\n\nPRICES SEEN: ' + self.prices_seen() + '\n\nTHE NOTE:\n' + '\n'.join(lines)
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
        if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0):
            budget = self.budget_as_written(budget, '\n'.join(lines))
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
        if gone or self.note_sold:
            self.swept = set()
        if got.get('price_moved') is True:
            changed.append('a price changed')
        summary = str(got.get('summary') or '').strip()
        if self.brief.get('pick') == 'cheapest' and (drops or gone or self.note_sold or any((c.startswith('budget now') for c in changed))):
            self.wider_held = False
            self.cheaper_marks = set()
            self.swept = set()
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
        ids = self.fact_ids(observation)
        needs = self.needs_now()
        if not ids or not needs:
            return ''
        tagged = len(self.items) > 1
        lines = []
        for ident in ids[:12]:
            block = _fact_block(self.full.get(ident) or {}) or {}
            parts = []
            for number, need in needs[:8]:
                prop = need.split(':', 1)[0].strip()
                hits = [f'{k}={str(v)[:50]}' for k, v in block.items() if _names(_stem(prop), _stem(k))]
                if hits:
                    parts.append((f'(item {number}) ' if tagged else '') + f'{prop} -> ' + ', '.join(hits[:3]))
            if parts:
                lines.append(f'{ident}: ' + '; '.join(parts))
        return EVIDENCE_HEAD + ' | '.join(lines) if lines else ''

    def fact_ids(self, observation: Any) -> list[str]:
        found: list[str] = []
        pending = [(observation, 0)]
        while pending:
            node, depth = pending.pop()
            if depth > 6:
                continue
            if isinstance(node, dict):
                ident = self.ident(node, False, depth)
                if ident and _fact_block(node) and ident not in found:
                    found.append(ident)
                    continue
                children = list(node.values())
            elif isinstance(node, list):
                children = node
            else:
                continue
            pending.extend(((child, depth + 1) for child in reversed(children)))
        return found

    def cart_text(self) -> str:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            return _dump(_clip(self.cart_view, 300))[:3000]
        return "lines by Sable's record of the cart calls (listing: units): " + _dump(self.cart_lines)

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
            lines.append(_dump(_clip(record, 160))[:1300])
            if len(lines) >= limit:
                break
        return '\n'.join(lines)

    def fit_index(self) -> dict[str, Any]:
        marks: dict[str, dict[str, Any]] = {}
        notes: dict[str, set[str]] = {}
        owners: dict[str, str] = {}
        lent: dict[str, set[str]] = {}
        for ident in self.seen_order:
            record = self.full.get(ident) or self.rows.get(ident) or {}
            flat = ' '.join((str(v) for v in record.values() if isinstance(v, str)))
            nested = '. '.join((f'{k} {v}' for block in record.values() if isinstance(block, dict) for k, v in block.items() if isinstance(v, str)))
            mark = marks[ident] = _fit_marks(flat + ' ' + nested)
            if mark['own']:
                owners.setdefault(mark['own'], ident)
            for part, codes in mark['also']:
                notes.setdefault(part, set()).update(codes)
            for code, found in mark['lent']:
                lent.setdefault(code, set()).update(found)
        return {'marks': marks, 'notes': notes, 'owners': owners, 'codes': {c for m in marks.values() for c in m['codes']}, 'lent': lent}

    def fit_set(self, ident: str, index: dict[str, Any]) -> set[str]:
        mark = index['marks'].get(ident) or {}
        own, more = (mark.get('own'), mark.get('more'))
        found = set(mark.get('works') or [])
        for stem, width in mark.get('wild') or []:
            found |= {c for c in index['codes'] if c.startswith(stem) and c != stem and (not width or len(c) == len(stem) + width)} - set(mark.get('skip') or [])
        if own:
            found |= index['notes'].get(own, set())
        if more and more != own:
            holder = index['owners'].get(more)
            if holder:
                found |= set((index['marks'].get(holder) or {}).get('works') or [])
            found |= index['notes'].get(more, set())
        return found

    def host_series(self, ident: str, index: dict[str, Any]) -> set[str]:
        mark = index['marks'].get(ident) or {}
        found = set(mark.get('series') or [])
        for code in mark.get('codes') or []:
            found |= index.get('lent', {}).get(code, set())
        return found

    def fit_verdict(self, line: str, host: str, index: dict[str, Any], linked: bool=True) -> bool | None:
        mark = index['marks'].get(line) or {}
        codes = set((index['marks'].get(host) or {}).get('codes') or [])
        series = self.host_series(host, index)
        kin = _kin(mark.get('own'))
        known = False
        if codes and (mark.get('works') or mark.get('wild')):
            known = True
            if self.fit_set(line, index) & codes:
                return True
        if series and kin and linked:
            known = True
            if kin in series:
                return True
        return False if known else None

    def series_linked(self, main: int, part: int, index: dict[str, Any]) -> bool:
        kinds = {_kin((index['marks'].get(i) or {}).get('own')) for i in self.item_ids(part)} - {''}
        named = {kin for host in self.item_ids(main) for kin in self.host_series(host, index)}
        return bool(kinds & named)

    def item_ids(self, number: int) -> list[str]:
        cat_key = self.category_key()
        if not cat_key:
            return []
        item = self.items[number - 1]
        return [i for i in self.seen_order if self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))]

    def fit_faults(self) -> list[str]:
        if not self.bound:
            return []
        index = self.fit_index()
        cart = self.cart_ids()
        out = []
        for part, main in self.bound:
            main_ids, part_ids = (set(self.item_ids(main)), self.item_ids(part))
            hosts = [i for i in cart if i in main_ids]
            lines = [i for i in cart if i in part_ids]
            if len(hosts) != 1 or not lines:
                continue
            codes = set(index['marks'].get(hosts[0], {}).get('codes') or [])
            series = self.host_series(hosts[0], index)
            if not codes and (not series):
                continue
            linked = self.series_linked(main, part, index)
            for line in lines:
                if self.fit_verdict(line, hosts[0], index, linked) is not False:
                    continue
                fits = self.fit_set(line, index)
                mark = index['marks'].get(line) or {}
                better = [i for i in part_ids if self.fit_verdict(i, hosts[0], index, linked) and (not self.sold_out(i)) and (i not in cart)][:4]
                said = sorted(fits)[:4] or [stem + 'x' * (width or 1) for stem, width in mark.get('wild') or []] or [f'part series {_kin(mark.get("own"))}']
                says = sorted(codes) + [f'part series {x}' for x in sorted(series)]
                text = f"{line} ({self.items[part - 1].name}) suits {', '.join(said)}, not {hosts[0]} ({', '.join(says)})"
                text += f"; parts of that kind already seen that suit it: {', '.join(better)}" if better else '; none of the parts of that kind seen so far work with it, so find one or change the device: ' + self.fit_plan(short=True)
                out.append(text)
        return out

    def link_room(self, cost: int) -> bool:
        return not self.finished and self.left() > self.tail() + 2 + cost and self.elapsed() < WRAP_AT - 15 and self.link_steps + cost <= LINK_STEPS

    def link_search(self, code: str) -> list[str]:
        plan = self.search_template()
        if not plan or not self.link_room(1):
            return []
        tool, base, field = plan
        faults: list[str] = []
        args = _shape({**base, field: code}, self.schemas.get(tool), 'args', faults)
        if faults:
            return []
        before = self.step
        reply = self.act(tool, args)
        self.link_steps += self.step - before
        if reply is None:
            return []
        seen = reply.get('observation')
        self.observe(tool, seen)
        return [i for i, _ in self.records_in(seen)]

    def link_holder(self, code: str, ids: list[str]) -> str:
        index = self.fit_index()
        holder = index['owners'].get(code)
        if holder:
            return holder
        word = '(?<![a-z0-9-])' + re.escape(code.lower()) + '(?![a-z0-9-])'
        for ident in ids:
            mark = index['marks'].get(ident) or {}
            if mark.get('more') != code and all((part != code for part, _ in mark.get('also') or [])) and re.search(word, _dump(self.rows.get(ident) or {}).lower()):
                return ident
        return ''

    def fit_links(self) -> None:
        mark = self.cart_mark()
        if not self.bound or mark in self.link_marks:
            return
        self.link_marks.add(mark)
        index = self.fit_index()
        codes = list(dict.fromkeys((m['more'] for m in (index['marks'].get(i) or {} for i in self.cart_ids()) if m.get('more') and m['more'] != m.get('own'))))
        for code in codes:
            holder = index['owners'].get(code) or self.link_holder(code, self.link_search(code))
            if not holder or holder in self.opened:
                continue
            cost = sum(1 for t in self.views() if (holder, t) not in self.viewed)
            if cost and self.link_room(cost):
                before = self.step
                self.page_text(holder)
                self.link_steps += self.step - before
                self.trace.append({'kind': 'fit_link', 'code': code, 'id': holder, 'opened': holder in self.opened})

    def price_limit(self, number: int) -> float | None:
        limits = []
        cap = self.item_cap(number)
        if cap is not None:
            prices = [x for x in (self.price_of(i) for i in self.item_ids(number)) if x is not None]
            if not prices or min(prices) <= cap * 20:
                limits.append(cap)
        if isinstance(self.budget, (int, float)):
            units = 1 if self.brief.get('budget_scope') == 'each' else max(1, self.items[number - 1].qty or 1)
            limits.append(self.budget / units)
        return min(limits) if limits else None

    def over_cap(self, number: int, ident: str, limit: float | None=None) -> bool:
        price = self.price_of(ident)
        return price is not None and limit is not None and price > limit + 1e-09

    def shows_values(self, number: int, ident: str) -> bool:
        words = [v.lower() for v in self.value_words(number) if not re.search('\\d', v)]
        text = _dump(self.rows.get(ident) or {}).lower()
        return bool(words) and all((re.search('(?<![a-z0-9])' + re.escape(w) + '(?![a-z0-9])', text) for w in words))

    def series_plan(self, main: int, parts: list[int], index: dict[str, Any]) -> list[str]:
        kinds: dict[str, dict[int, list[str]]] = {}
        for p in parts:
            limit = self.price_limit(p)
            for i in self.item_ids(p):
                kin = _kin((index['marks'].get(i) or {}).get('own'))
                if kin and (not self.sold_out(i)) and (not self.over_cap(p, i, limit)):
                    kinds.setdefault(kin, {}).setdefault(p, []).append(i)
        homes: dict[str, list[str]] = {}
        for i in self.item_ids(main):
            if not self.sold_out(i):
                for kin in self.host_series(i, index):
                    homes.setdefault(kin, []).append(i)
        if not homes or not set(kinds) & set(homes):
            return []

        def cost(ident: str) -> float:
            price = self.price_of(ident)
            return price if price is not None else float('inf')

        cart = set(self.cart_ids())

        def tag(ident: str) -> str:
            price = self.price_of(ident)
            return (f'{ident} at {price:g}' if price is not None else ident) + (' (in the cart)' if ident in cart else '')
        order = sorted(set(kinds) | set(homes), key=lambda k: (not set(homes.get(k, [])) & cart, k not in homes, -len(kinds.get(k, {})), -sum((len(v) for v in kinds.get(k, {}).values()))))[:5]
        said = []
        for kin in order:
            have = kinds.get(kin, {})
            bits = [f"{self.items[p - 1].name} seen: {', '.join((tag(i) for i in sorted(have[p], key=lambda i: (i not in cart, cost(i)))[:3]))}" for p in parts if p in have]
            gaps = [self.items[p - 1].name for p in parts if p not in have]
            text = f'series {kin}: ' + ('; '.join(bits) if bits else 'no part seen')
            if gaps and bits:
                text += f"; none seen yet for {', '.join(gaps)}"
            devices = sorted(homes.get(kin, []), key=lambda i: (i not in cart, cost(i)))[:3]
            text += f" (devices taking it: {', '.join((tag(i) for i in devices))})" if devices else ' (no device seen takes it)'
            said.append(text)
        return [f'part series named by the {self.items[main - 1].name} candidates (a part fits when its own code starts with a series named on the device page; a series code printed in part titles can be added to the search words to list that series) -> ' + ' | '.join(said)]

    def fit_plan(self, short: bool=False) -> str:
        if not self.bound:
            return ''
        index = self.fit_index()
        lines = []
        for main in sorted({m for _, m in self.bound}):
            parts = [p for p, m in self.bound if m == main]
            counts: dict[str, dict[int, int]] = {}
            for p in parts:
                limit = self.price_limit(p)
                pool = [i for i in self.item_ids(p) if not self.sold_out(i) and (not self.over_cap(p, i, limit))]
                for i in [i for i in pool if self.shows_values(p, i)] or pool:
                    for code in self.fit_set(i, index):
                        per = counts.setdefault(code, {})
                        per[p] = per.get(p, 0) + 1
            main_ids = self.item_ids(main)
            hosts = [i for i in main_ids if not self.sold_out(i)]
            liked = {code for i in hosts if self.shows_values(main, i) for code in index['marks'].get(i, {}).get('codes') or []}
            carried = {code for i in hosts for code in index['marks'].get(i, {}).get('codes') or []}
            ranked = sorted(counts.items(), key=lambda kv: (-len(kv[1]), kv[0] not in liked, kv[0] not in carried, -sum(kv[1].values())))[:6]
            shown = []
            cheap = self.brief.get('pick') == 'cheapest'
            for code, per in ranked:
                devices = [i for i in main_ids if code in (index['marks'].get(i, {}).get('codes') or [])]
                devices = (sorted(devices, key=lambda i: self.price_of(i) or 0.0) if cheap else devices)[:2]
                named = [f'{i} at {self.price_of(i):g}' if cheap and self.price_of(i) is not None else i for i in devices]
                tail = f" (matching device seen: {', '.join(named)})" if devices else ' (no matching device seen yet)'
                least = ''
                if cheap:
                    costs = sorted(((self.price_of(i), i) for p in per for i in self.item_ids(p) if code in self.fit_set(i, index) and self.price_of(i) is not None and (not self.sold_out(i))))
                    least = f', lowest {costs[0][1]} at {costs[0][0]:g}' if costs else ''
                shown.append(f'{code}: ' + ', '.join((f'{n} {self.items[p - 1].name}' for p, n in sorted(per.items()))) + least + tail)
            if shown:
                lines.append(f'codes named by the parts for the {self.items[main - 1].name} candidates -> ' + '; '.join(shown))
            likely = [h for h in hosts if self.shows_values(main, h)] or hosts
            for p in parts:
                limit = self.price_limit(p)
                pool = [i for i in self.item_ids(p) if not self.sold_out(i) and (not self.over_cap(p, i, limit))]
                pairs = []
                for i in sorted(pool, key=lambda i: (not self.shows_values(p, i), self.price_of(i) if self.price_of(i) is not None else float('inf'))):
                    fits = self.fit_set(i, index)
                    good = [h for h in likely if set(index['marks'].get(h, {}).get('codes') or []) & fits]
                    if good:
                        price = self.price_of(i)
                        pairs.append(f"{i}{f' at {price:g}' if price is not None else ''} fits {', '.join(good[:2])}")
                    if len(pairs) >= 6:
                        break
                if pairs:
                    lines.append(f'{self.items[p - 1].name} within the price limits that fit a {self.items[main - 1].name} candidate -> ' + '; '.join(pairs))
            lines += self.series_plan(main, parts, index)
            pending = sorted({(index['marks'][i]['more'], i) for p in parts for i in self.item_ids(p) if i in index['marks'] and index['marks'][i]['more'] and index['marks'][i]['more'] not in index['owners'] and (not self.sold_out(i))})
            if pending:
                lines.append('fit lists that continue elsewhere -> ' + '; '.join((f'{i} puts the rest of its fit list under part {x}: search for that part code and open it' for x, i in pending[:3])))
        if not lines:
            return ''
        return ' | '.join(lines) if short else FIT_HEAD + ' | '.join(lines)





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
        if key in self.flags and (not lead):
            return self.insist(key)
        if key in self.checked or not self.judges(limit) or (not ids):
            return ''
        self.checked.add(key)
        prompt = CHECK + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger() + self.dropped_note()
        if lead:
            quantity = '; '.join((f'item {n}: quantity {it.qty}' for n, it in enumerate(self.items, 1)))
            prompt += '\n\nCART (planned; treat these listings as carted at the requested quantities: ' + quantity + '):\n' + ', '.join(ids)
        else:
            prompt += '\n\nCART:\n' + self.cart_text()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + '\n'.join((f'{i}: ' + ('SOLD OUT (the store reports it can no longer be bought). ' if self.sold_out(i) else '') + self.record_of(i) for i in ids))
        if self.brief.get('pick') in ('cheapest', 'most', 'least'):
            prompt += '\n\nOTHER LISTINGS SEEN (latest first):\n' + self.others(ids, 15)
        got = self.ask_json(prompt, self.judges(limit), 3000)
        self.trace.append({'kind': 'check', 'cart': ids, 'got': got})
        self.verdicts[key] = bool(got) and got.get('ok') is not False
        if not got or got.get('ok') is not False:
            return ''
        self.leads = [i for i in self.leads if i not in ids]
        problems = [str(p) for p in _items(got.get('problems'))][:5]
        better = [k for k in (self.known_id(b) for b in _items(got.get('better'))) if k and (not self.sold_out(k))][:4]
        if not lead:
            self.flags[key] = (problems, better)
        text = 'A close review finds this cart not right yet: ' + '; '.join(problems or ['see the requirements'])
        if better:
            text += '. Listings that look better: ' + ', '.join(better)
        return text + '. Repair the cart (or re-read the records to show the review is mistaken), then check out.'

    def insist(self, key: str) -> str:
        problems, better = self.flags.pop(key)
        if self.finished or self.left() < 5 or self.elapsed() > WRAP_AT - 20:
            return ''
        swapped = self.adopt(better) if better else ''
        self.trace.append({'kind': 'insist', 'swapped': bool(swapped), 'better': better[:3]})
        if swapped:
            return swapped
        return "The review's finding still stands and the cart is unchanged: " + '; '.join(problems or ['see the requirements']) + ('. Listings that look better: ' + ', '.join(better) if better else '') + '. Repair the cart now, then check out; order it unchanged only if a record you quote plainly contradicts the finding.'

    def adopt(self, better: list[str]) -> str:
        cart = self.roles.get('cart')
        cat_key = self.category_key()
        now = self.cart_ids()
        pick = next((b for b in better if b in self.rows and b not in now and (not self.sold_out(b))), None)
        if not cart or not cat_key or not pick:
            return ''

        def kind(ident: str) -> str:
            return str((self.rows.get(ident) or {}).get(cat_key) or '').strip().lower()
        same = [i for i in now if kind(i) == kind(pick)]
        if len(same) != 1 or not kind(pick):
            return ''
        lines = {i: max(1, self.cart_lines.get(i, 1)) for i in now if i != same[0]}
        lines[pick] = max(1, self.cart_lines.get(same[0], 1))
        calls = self.cart_calls(lines, [same[0]])
        if not calls or len(calls) + 3 > self.left():
            return ''
        self.fill_facts([pick], len(calls) + 3)
        self.check_cart(list(lines), SPEND_HARD)
        if not self.verdicts.get(self.check_key(list(lines))):
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
        self.trace.append({'kind': 'adopted', 'out': same[0], 'in': pick})
        return f'Sable swapped {same[0]} for {pick} in the cart: the review found that {pick} meets every requirement and suits the request better, and a second review of the new cart agrees. Check out now.'

    def trail(self) -> tuple[str, bool]:
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        tool = self.offer_tool()
        target = self.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trails < (6 if self.left() > 15 else 4) and self.cost < SPEND_HARD:
            self.trails += 1
            held = [i for i in self.cart_ids() if i in self.rows]
            prompt = TRAIL + '\n\nREQUIREMENTS WHOSE VALUE SITS ON ANOTHER LISTING:\n' + '\n'.join(pending) + '\n\nCART (the listings chosen for this order):\n' + ('\n'.join((f'{i}: ' + _dump(_clip(self.rows.get(i), 80))[:300] for i in held)) or '(empty)') + '\n\nRECORDS (newest first):\n' + '\n'.join(self.trail_records())
            got = self.ask_json(prompt, [LIGHT] + ([HEAVY] if self.cost + HEAVY_ROOM <= SPEND_HARD else []), 4500) or {}
            self.trace.append({'kind': 'trail', 'pending': pending, 'got': _clip(got, 150)})
            reach, lost, gaps = self.trail_links(got, pending)
            pending = [p for p in pending if p not in self.ref_values]
            if not reach and gaps and tool and target and self.left() > 12 and self.link_searches < 3:
                reach = self.find_link(*gaps[0])
                if self.trace and self.trace[-1].get('kind') == 'link_known':
                    seen.append(f"{self.trace[-1]['id']}: an opened page of the shop this route names, showing the values its description gives; take the value from it")
                    lost = lost[1:]
            if not reach or not tool or not target or self.left() < len(reach) + 4 or self.finished:
                break
            for ident in reach[:max(0, min(3, 6 - len(seen)))]:
                if self.finished or self.left() <= 4:
                    break
                seen.append(f'{ident}: ' + self.page_text(ident)[:1500])
        text = '[Sable opened the pages a requirement points to]\n' + '\n'.join(seen) + '\n' if seen else ''
        if self.ref_values:
            text += '[values taken from the pages requirements point to] ' + '; '.join((f'{k} -> {v}' for k, v in self.ref_values.items())) + '\n'
        hold = bool(lost) and self.trail_blocks < (3 if self.left() > 15 else 2) and self.left() > 5
        if hold:
            self.trail_blocks += 1
            text += 'The order only counts once every page a requirement points to has been opened in full (a search row does not count), including each further listing a page sends you to. Not yet found and opened: ' + ' | '.join(lost[:3]) + ". Look it up by its shop's name plus the values its description mentions (other listings of that shop and kind are not it), open the row of that shop whose page shows those values, apply the value that page gives, then check out."
        return (text, hold)

    def trail_records(self) -> list[str]:
        records = []
        for ident in reversed(self.seen_order):
            record = self.full.get(ident) or self.rows.get(ident) if self.seen_whole(ident) else None
            if record and len(records) < 45:
                records.append('OPENED ' + _dump(_clip(record, 300))[:1400])
        for ident in reversed(self.seen_order):
            if len(records) >= 60:
                break
            if self.seen_whole(ident) and (self.full.get(ident) or self.rows.get(ident)):
                continue
            if ident in self.full:
                records.append('SPECS ' + _dump(_clip(self.full[ident], 200))[:1400])
            else:
                records.append('ROW ' + _dump(_clip(self.rows.get(ident), 80))[:300])
        return records

    def trail_links(self, got: dict[str, Any], pending: list[str]) -> tuple[list[str], list[str], list[tuple[str, str]]]:
        reach: list[str] = []
        lost: list[str] = []
        gaps: list[tuple[str, str]] = []
        for chain in _items(got.get('chains')):
            if not isinstance(chain, dict):
                continue
            need = next((p for p in pending if p == str(chain.get('need') or '').strip()), None) or (pending[0] if len(pending) == 1 else None)
            if not need:
                continue
            links = [l for l in _items(chain.get('links')) if isinstance(l, dict)]
            ids = [str(l.get('id') or '').strip() for l in links]
            opened = bool(links) and all((self.seen_whole(i) for i in ids))
            value = str(chain.get('value') or '').strip()
            if opened and chain.get('done') is True and value:
                self.ref_values[need] = value
                self.version += 1
                continue
            if opened and (not str(links[-1].get('find') or '').strip()):
                self.ref_values[need] = value or f'as stated on {ids[-1]}'
                self.version += 1
                continue
            closed = [i for i in ids if i in self.rows and (not self.seen_whole(i))]
            if closed:
                reach += [i for i in closed if i not in reach]
                continue
            gap = next((l for l, i in zip(links, ids) if i not in self.rows), {})
            about = str(gap.get('about') or '') or 'the listing it points at'
            words = str(gap.get('find') or '').strip()
            lost.append(f'{need} -> find {about}' + (f' (search: {words})' if words else ''))
            gaps.append((about, words))
        return (reach, lost, gaps)

    def chase_due(self) -> bool:
        live = {need for _, need in self.needs_now()}
        waiting = [need for _, need in self.refs if need in live and need not in self.ref_values and need not in self.chased]
        return bool(waiting) and self.left() > 8 + 2 * len(self.items) and self.elapsed() < WRAP_AT - 30 and self.search_template() is not None

    def tell(self, text: str) -> None:
        if self.chat and self.chat[-1].get('role') == 'user' and isinstance(self.chat[-1].get('content'), str):
            self.chat[-1]['content'] += '\n\n' + text
        else:
            self.chat.append({'role': 'user', 'content': text})

    def sellers(self, ident: str) -> list[str]:
        names = []
        for source in (self.rows.get(ident) or {}, self.full.get(ident) or {}):
            for key, value in source.items():
                if isinstance(value, str) and 2 < len(value.strip()) <= 40 and (re.search(SHOP_KEYS, str(key), re.I) or (len(value.split()) <= 3 and (not re.search('[0-9]', value)))):
                    names.append(' '.join(value.lower().split()))
        return names

    def keyed_sellers(self, ident: str) -> list[str]:
        return [' '.join(value.lower().split()) for source in (self.rows.get(ident) or {}, self.full.get(ident) or {}) for key, value in source.items() if isinstance(value, str) and 2 < len(value.strip()) <= 40 and re.search(SHOP_KEYS, str(key), re.I)]

    def chase(self) -> str:
        live = {need for _, need in self.needs_now()}
        waiting = [(n, need) for n, need in self.refs if need in live and need not in self.ref_values and need not in self.chased]
        page = self.offer_tool()
        if not waiting or not page or not self.single_target(page) or self.search_template() is None:
            return ''
        lines = []
        for _, need in waiting[:3]:
            self.chased.add(need)
            route: list[str] = []
            sure = True
            record = ''
            ended = 'stopped to keep steps for the order'
            for _ in range(7):
                if self.finished or self.left() < 8 + 2 * len(self.items) or self.elapsed() > WRAP_AT - 30 or self.cost > SPEND_SOFT:
                    break
                prompt = CHASE + '\n\nREQUIREMENT:\n' + need + (f'\n\nRECORD OF {route[-1]}:\n' + record if route else '')
                got = self.ask_json(prompt, self.judges() if route else [LIGHT, HEAVY], 3000) or {}
                if route and got.get('final') is True:
                    value = str(got.get('value') or '').strip()
                    if value and sure:
                        said = self.page_words(route[-1])
                        self.ref_values[need] = value + (f' (page {route[-1]} says: "{said}")' if said and said.lower() not in value.lower() else '')
                        self.version += 1
                    ended = f'the last page gives: {value}' if value else 'the last page gives no value for it'
                    break
                shop = ' '.join(str(got.get('shop') or '').split())
                if not shop:
                    ended = 'no further listing is named'
                    break
                wanted = [' '.join(str(v).split()) for v in _items(got.get('values')) if str(v).strip()]
                found, exact = self.spot(shop, str(got.get('kind') or ''), wanted, route, [' '.join(str(v).split()) for v in _items(got.get('terms')) if str(v).strip()])
                if not found:
                    ended = f'no listing of {shop} ({", ".join(wanted[:4])}) turned up'
                    sure = False
                    break
                sure = sure and exact
                route.append(found)
                record = self.page_text(found)
                if not exact:
                    ended = f'{found} matched only part of the description of the listing to read, so Sable stopped there'
                    break
            self.routes[need] = route
            self.trace.append({'kind': 'chase', 'need': need[:160], 'route': route, 'end': ended[:200], 'sure': sure})
            settled = need in self.ref_values
            tail = '' if settled else '. The value is not settled yet: read the last page opened (its description and attributes; a goes-together-with or compatible-with sentence is part of its value), open any listing it sends you to, and apply the value before choosing'
            lines.append(f'- {need}: pages opened in full ' + (' -> '.join(route) or 'none') + f'; {ended}' + tail)
        return "[Sable followed the listings these requirements point at]\n" + '\n'.join(lines) if lines else ''

    def spot(self, shop: str, kind: str, wanted: list[str], route: list[str], terms: list[str] | None=None) -> tuple[str | None, bool]:
        terms = terms or []
        want = shop.lower()
        plain = re.sub("['\u2019]s$", '', want)
        named = shop[:len(plain)] if plain != want else shop
        kinds = {w for w in _stem(kind) if len(w) >= 4} - {'listing', 'product', 'item', 'other'}
        cat_key = self.category_key()

        def ours(ids: list[str]) -> list[str]:
            out = []
            for ident in ids:
                if ident in route or ident in out:
                    continue
                if any((n == want or n == plain or ((n in self.keyed_sellers(ident)) and (re.search('(?<![a-z0-9])' + re.escape(want) + '(?![a-z0-9])', n) or re.search('(?<![a-z0-9])' + re.escape(n) + '(?![a-z0-9])', want))) for n in self.sellers(ident))):
                    out.append(ident)
            if kinds and cat_key:
                near = [i for i in out if kinds & {w for w in _stem(str((self.rows.get(i) or {}).get(cat_key) or '')) if len(w) >= 4}]
                out = near or out
            return out

        def grade(ident: str) -> tuple[int, float, int]:
            record = self.full.get(ident) or self.rows.get(ident) or {}
            block = _fact_block(record) or {}
            pairs = [_unglue((_words(k) + ' ' + str(v)).lower()) for k, v in block.items()]
            blob = _kept(_unglue((_dump(record) + ' ' + ' '.join((_words(k) for k in block))).lower()))

            def has(word: str, text: str) -> bool:
                return bool(re.search('(?<![a-z0-9])' + re.escape(word) + '(?![a-z0-9])', text))
            found = strong = 0
            share = 0.0
            for value in (wanted if ident in self.full or len(bare) != len(wanted) else bare):
                words = [w for w in re.findall('[a-z0-9]+(?:\\.[0-9]+)?', _unglue(value.lower())) if w not in ('the', 'and', 'with', 'a', 'of')]
                units = {words[n - 1]: w for n, w in enumerate(words) if n and re.fullmatch('[0-9]+(?:\\.[0-9]+)?', words[n - 1]) and re.fullmatch('[a-z]{1,3}', w)}
                if any((re.fullmatch('[0-9]+(?:\\.[0-9]+)?', w) for w in words)):
                    words = [w for n, w in enumerate(words) if not (n and re.fullmatch('[0-9]+(?:\\.[0-9]+)?', words[n - 1]) and re.fullmatch('[a-z]{1,3}', w))] or words
                if not words:
                    continue
                if any((all((has(w, pair) and (not (w in units and _unit_clash(w, units[w], pair))) for w in words)) for pair in pairs)):
                    strong += 1
                seen = len([w for w in words if has(w, blob) and (not (w in units and _unit_clash(w, units[w], blob)))])
                share += seen / len(words)
                found += seen == len(words)
            return (found, share, strong)

        def hits(ident: str) -> int:
            return grade(ident)[0]
        def settled(pool: list[str]) -> bool:
            if not pool:
                return False
            if wanted and max((hits(i) for i in pool)) < len(wanted):
                for ident in [i for i in sorted(pool, key=lambda i: -hits(i)) if i not in self.full][:3]:
                    self.fill_facts([ident], 6)
                    if ident in self.full and hits(ident) >= len(wanted):
                        break
            return not wanted or max((hits(i) for i in pool)) >= len(wanted)
        plan = self.search_template()
        known = {_words(k) for record in self.full.values() for k in _fact_block(record) or {}}

        def split(value: str) -> tuple[str, str]:
            words = _words(value).split()
            for cut in range(len(words) - 1, 0, -1):
                if ' '.join(words[:cut]) in known:
                    return (' '.join(words[:cut]), ' '.join(words[cut:]))
            return ('', value)
        bare = [t for t in terms if t.strip()] or [split(v)[1] for v in wanted]
        tries = [' '.join([named] + bare[:4]), f'{named} {kind}'.strip(), named, ' '.join([kind] + bare[:4])] if wanted else [f'{named} {kind}'.strip(), named]
        order: list[str] = []

        def look(room: int) -> bool:
            while plan and tries and self.left() > room and (not self.finished):
                words = ' '.join(tries.pop(0).split())
                if not words:
                    continue
                tool, base, field = plan
                faults: list[str] = []
                args = _shape({**base, field: words}, self.schemas.get(tool), 'args', faults)
                reply = None if faults else self.act(tool, args)
                if reply is None:
                    return False
                seen = reply.get('observation')
                self.observe(tool, seen)
                order.extend((i for i, _ in self.records_in(seen) if i not in order))
                self.trace.append({'kind': 'chase_search', 'words': words[:80]})
                for _ in range(SHOP_PAGES):
                    after = self.paging(tool, seen, args) if plain in words.lower() else None
                    if not after or after[3] is None or after[0] > (SHOP_PAGES + 1) * after[1] or self.left() <= room + 1 or self.finished:
                        break
                    args = _shape({**args, after[2]: after[3]}, self.schemas.get(tool), 'args', faults)
                    reply = None if faults else self.act(tool, args)
                    if reply is None:
                        break
                    seen = reply.get('observation')
                    self.observe(tool, seen)
                    order.extend((i for i, _ in self.records_in(seen) if i not in order))
                return True
            return False

        def tokens(text: str) -> set[str]:
            return set(re.findall('[0-9]+(?:\\.[0-9]+)?|[a-z]+', text.lower()))

        def clash(ident: str) -> bool:
            block = {_words(k): str(v) for k, v in (_fact_block(self.full.get(ident) or {}) or {}).items()}
            text = _unglue(_dump(self.full.get(ident) or self.rows.get(ident) or {}).lower())
            kept = _kept(text)
            for value in wanted:
                said = [w for w in re.findall('[a-z0-9]+(?:\\.[0-9]+)?', _unglue(value.lower())) if w not in ('the', 'and', 'with', 'a', 'of')]
                if said and all((re.search('(?<![a-z0-9])' + re.escape(w) + '(?![a-z0-9])', text) for w in said)) and (not all((re.search('(?<![a-z0-9])' + re.escape(w) + '(?![a-z0-9])', kept) for w in said))):
                    return True
            for value in wanted:
                words = _words(value).split()
                key = next((' '.join(words[:cut]) for cut in range(len(words) - 1, 0, -1) if ' '.join(words[:cut]) in block), '')
                rest = words[len(key.split()):]
                rest = [w for n, w in enumerate(rest) if not (n and re.fullmatch('[0-9]+(?:\\.[0-9]+)?', rest[n - 1]) and re.fullmatch('[a-z]{1,3}', w))]
                if key and (not tokens(' '.join(rest)) - {'the', 'and', 'with', 'a', 'of'} <= tokens(_words(block[key]))):
                    return True
            return False

        def judge(group: list[str]) -> tuple[str, bool] | None:
            records = '\n'.join((f'{i}: ' + _dump(_clip(self.full.get(i) or self.rows.get(i) or {}, 300))[:1500] for i in group))
            got = self.ask_json(PICK + f'\n\nDESCRIPTION: the {kind or "product"} listing of {shop} with ' + '; '.join(wanted) + '\n\nLISTINGS:\n' + records, [LIGHT], 3000)
            if not isinstance(got, dict):
                return None
            ident = str(got.get('id') or '').strip()
            whole = got.get('all') is True and ident in group and (not clash(ident))
            self.trace.append({'kind': 'chase_pick', 'id': ident[:40], 'all': whole, 'of': len(group)})
            return (ident, whole) if ident in group else ('', False)
        pool = ours(list(reversed(self.seen_order)))
        if not wanted:
            while not pool and look(6):
                pool = ours(list(reversed(self.seen_order)))
            return (pool[0], True) if pool else (None, False)
        judged: set[str] = set()
        partial = ''
        ran = False
        for rounds in range(SPOT_ROUNDS):
            room = 6 if rounds == 0 else 16
            if (rounds or not pool or max((hits(i) for i in pool)) < len(wanted)) and look(room):
                pool = ours(list(reversed(self.seen_order)))
            mine = ours(order)
            pool.sort(key=lambda i: (-grade(i)[1], -grade(i)[2]))
            mine.sort(key=lambda i: (-grade(i)[1], -grade(i)[2]))
            ranked = mine + [i for i in pool if i not in mine]
            fresh = [i for i in ranked if i not in judged][:4]
            if not fresh:
                if not tries:
                    break
                continue
            for ident in [i for i in fresh if i not in self.full]:
                self.fill_facts([ident], room)
                if ident in self.full and max(grade(ident)[0], grade(ident)[2]) >= len(wanted) and (not clash(ident)):
                    break
            solid = [i for i in ranked if i in self.full and max(grade(i)[0], grade(i)[2]) >= len(wanted) and (not clash(i))]
            if solid:
                self.trace.append({'kind': 'chase_pick', 'id': solid[0][:40], 'all': True, 'of': 0})
                return (solid[0], True)
            verdict = judge(fresh + ([partial] if partial and partial not in fresh else []))
            judged |= set(fresh)
            if verdict is None:
                break
            ran = True
            if verdict[0] and verdict[1]:
                return verdict
            partial = verdict[0] or partial
            if not tries and (not [i for i in ranked if i not in judged]):
                break
        if partial:
            return (partial, False)
        while (not ran) and (not settled(pool)) and look(6):
            pool = ours(list(reversed(self.seen_order)))
        if not pool:
            return (None, False)
        pool.sort(key=lambda i: (-grade(i)[1], -grade(i)[2]))
        return (pool[0], (not ran) and hits(pool[0]) >= len(wanted))

    def page_words(self, ident: str) -> str:
        record = self.full.get(ident) or self.rows.get(ident) or {}
        texts = []
        for value in record.values():
            if isinstance(value, str) and len(value.split()) >= 6 and value.strip() not in texts:
                texts.append(value.strip())
        return ' / '.join(texts)[:400]

    def views(self) -> list[str]:
        busy = {self.roles.get(job) for job in ('cart', 'buy', 'refuse', 'ask', 'search', 'cart_view', 'compare')} - {None}
        out: list[str] = []
        for tool in [self.roles.get('offer'), self.roles.get('open')] + list(self.schemas):
            if tool and tool not in busy and tool not in out:
                target = self.single_target(tool)
                if target and re.search('id|code|sku|ref', target, re.I):
                    out.append(tool)
        return out

    def seen_whole(self, ident: str) -> bool:
        if (ident, '*compare*') in self.viewed:
            return True
        tools = self.views()
        if not tools:
            return ident in self.opened
        return all(((ident, t) in self.viewed for t in tools))

    def page_text(self, ident: str) -> str:
        for tool in [] if (ident, '*compare*') in self.viewed else self.views():
            if (ident, tool) in self.viewed:
                continue
            target = self.single_target(tool)
            if not target or self.left() <= 4 or self.finished:
                continue
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
            if faults:
                continue
            reply = self.act(tool, args)
            if reply is None:
                break
            seen = reply.get('observation')
            self.observe(tool, seen)
            self.note_view(tool, seen)
            if tool == self.offer_tool():
                self.offer_seen[ident] = self.step
        return _dump(_clip(self.full.get(ident) or self.rows.get(ident) or {}, 600))[:3000]

    def shop_rows(self, about: str, ids: list[str]) -> list[str]:
        out = []
        for ident in ids:
            row = self.rows.get(ident) or {}
            if ident not in self.opened and self.shop_named(row, about):
                out.append(ident)
        return out

    def shop_names_in(self, row: dict[str, Any], about: str) -> list[str]:
        out = []
        for key, value in row.items():
            if not isinstance(value, str) or not 2 < len(value.strip()) <= 40:
                continue
            name = ' '.join(value.split())
            keyed = bool(re.search(SHOP_KEYS, str(key), re.I))
            if not keyed and (len(name.split()) > 3 or re.search('[0-9]', name)):
                continue
            for hit in re.finditer('(?<![A-Za-z0-9])' + re.escape(name) + '(?![A-Za-z0-9])', about, re.I):
                if keyed or hit.group(0)[:1].isupper():
                    out.append(name)
                    break
        return out

    def shop_named(self, row: dict[str, Any], about: str) -> bool:
        return bool(self.shop_names_in(row, about))

    def opened_shop(self, about: str) -> list[str]:
        out = []
        for ident in self.opened:
            row = self.rows.get(ident) or {}
            if self.shop_named(row, about):
                out.append(ident)
        return out

    def find_link(self, about: str, words: str) -> list[str]:
        known: list[str] = []
        named = {w for ident in self.shop_rows(about, list(self.rows)) + self.opened_shop(about) for v in self.shop_names_in(self.rows.get(ident) or {}, about) for w in re.findall('[a-z0-9]{3,}', v.lower())}
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
        text = f"[Sable] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        if picks and self.left() <= 6:
            return text + f" Few steps are left: in this turn add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks[1:]) or 'none'}."
        if picks and self.brief.get('pick') == 'cheapest':
            return text + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(picks)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {picks[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        if picks:
            return text + f" Listings already read that appear to fit that item instead: {', '.join(picks)}. In this turn take the sold-out line out of the cart and add {picks[0]} at the requested quantity, then check out."
        return text + ' Take that line out of the cart and add the best other qualifying listing; search for one if none you have read qualifies.'

    def find_sold(self) -> str:
        ids = [i for i in self.cart_ids() if i not in self.gone]
        if any((self.sold_out(i) for i in ids)):
            return ''
        if len(ids) == 1:
            self.unavailable.add(ids[0])
            return ''
        return self.reread_offers() if ids else ''

    def sweep_backup(self, lost: list[str]) -> list[str]:
        cat_key = self.category_key()
        kinds = {str((self.rows.get(i) or {}).get(cat_key) or '') for i in lost} if cat_key else set()
        return [i for ids in self.sweep_fits.values() for i in ids if i not in lost and (not self.sold_out(i)) and i not in self.cart_ids() and str((self.rows.get(i) or {}).get(cat_key) or '') in kinds][:4]

    def replacements(self, lost: list[str]) -> list[str]:
        if not self.judges():
            self.swap_fits = self.sweep_backup(lost)
            return self.swap_fits
        records = []
        for ident in reversed(self.seen_order):
            if ident in lost or ident in self.gone or self.sold_out(ident):
                continue
            records.append(_dump(_clip(self.full.get(ident) or self.rows.get(ident), 200))[:1600])
            if len(records) >= 25:
                break
        if not records:
            self.swap_fits = self.sweep_backup(lost)
            return self.swap_fits
        prompt = SWAP + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
        prompt += '\n\nSOLD-OUT CART LINE:\n' + '\n'.join((f'{i}: {self.record_of(i)}' for i in lost)) + '\n\nCART NOW:\n' + self.cart_text()
        prompt += '\n\nOTHER LISTING RECORDS SEEN (latest first):\n' + '\n'.join(records)
        got = self.ask_json(prompt, self.judges(), 3500)
        fits: list[tuple[float, str]] = []
        for entry in _items((got or {}).get('fits')):
            ident = str(entry.get('id') if isinstance(entry, dict) else entry)
            price = entry.get('price') if isinstance(entry, dict) else None
            if ident in self.rows and ident not in lost and (ident not in self.gone) and (ident not in self.cart_ids()) and (ident not in [i for _, i in fits]):
                fits.append((float(price) if isinstance(price, (int, float)) and (not isinstance(price, bool)) else float('inf'), ident))
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.brief.get('pick') == 'cheapest' else [i for _, i in fits]
        cat_key = self.category_key()
        kinds = {str((self.rows.get(i) or {}).get(cat_key) or '') for i in lost} if cat_key else set()
        backup = [i for n, ids in self.sweep_fits.items() for i in ids if i not in lost and (not self.sold_out(i)) and i not in self.cart_ids() and str((self.rows.get(i) or {}).get(cat_key) or '') in kinds]
        ranked = list(dict.fromkeys(backup + ranked)) if self.brief.get('pick') == 'cheapest' else list(dict.fromkeys(ranked + backup))
        self.trace.append({'kind': 'swap', 'lost': lost, 'fits': fits, 'backup': backup[:4]})
        self.swap_fits = ranked[:4]
        return ranked[:4]

    def audit(self) -> str:
        if self.audited or not self.judges():
            return ''
        self.audited = True
        got = self.audit_call(previews=True)
        named = self.audit_fits(got)
        if not named:
            rows = [k for k in (self.known_id(i) for i in _items((got or {}).get('open'))) if k and k not in self.opened and (not self.sold_out(k))]
            valued = [i for n in range(1, len(self.items) + 1) for i in self.value_rows(n)[:2]]
            room = max(2, min(6, len(rows) + (1 if valued else 0)))
            rows = list(dict.fromkeys(rows + valued))
            if rows and self.open_rows(rows, room):
                got = self.audit_call(previews=True)
                named = self.audit_fits(got)
        if not named:
            return ''
        named = {k: self.own_first(v) for k, v in named.items()}
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return 'A close review suggests these listings may qualify: ' + '; '.join((f"item {k}: {', '.join(v)}" for k, v in named.items())) + (f" ({got.get('reason')})" if got.get('reason') else '') + '. Open them and buy whichever really qualifies; end with no purchase only if every one of them fails.'

    def value_rows(self, number: int) -> list[str]:
        item = self.items[number - 1]
        own = set(re.findall('[a-z0-9]+', _words(item.name + ' ' + (item.shelf or ''))))
        wanted: list[set[str]] = []
        for owner, need in self.needs_now():
            if owner != number:
                continue
            text = self.ref_values.get(need, '').split(' (page ')[0] if (number, need) in self.refs else need.split(':', 1)[-1]
            words = {w for w in re.findall('[a-z]+', text.lower()) if len(w) >= 4 and w not in own}
            if words:
                wanted.append(words)
        if not wanted:
            return []
        scored = []
        for ident in self.item_ids(number) if len(self.items) > 1 else list(self.seen_order):
            if ident in self.full or ident in self.opened or self.sold_out(ident):
                continue
            price = self.price_of(ident)
            if isinstance(self.budget, (int, float)) and price is not None and self.brief.get('budget_scope') != 'each' and price > self.budget:
                continue
            row = _dump(self.rows.get(ident) or {}).lower()
            hits = sum((1 for words in wanted if any((re.search('(?<![a-z])' + re.escape(w) + '(?![a-z])', row) for w in words))))
            if hits:
                scored.append((-hits, price if price is not None else float('inf'), ident))
        return [i for _, _, i in sorted(scored)][:3]

    def audit_call(self, models: list[str] | None=None, previews: bool=False) -> dict[str, Any] | None:
        held = [i for i in dict.fromkeys(self.cart_ids() or list(self.cart_lines)) if i in self.rows and (not self.sold_out(i))]
        latest = [i for i in reversed(self.seen_order) if not self.sold_out(i) and i not in held]
        opened = [i for i in latest if i in self.full]
        if len(self.items) > 1:
            lanes = [[i for i in opened if i in set(self.item_ids(n))] for n in range(1, len(self.items) + 1)]
            mixed = [lane[k] for k in range(max([len(lane) for lane in lanes] + [0])) for lane in lanes if k < len(lane)]
        else:
            mixed = list(opened)
        picked = held + mixed
        picked += [i for i in latest if i not in picked]
        records = []
        for ident in picked[:24]:
            record = self.full.get(ident)
            tag = 'OPENED (in the cart) ' if ident in held else 'OPENED '
            records.append(tag + _dump(_clip(record, 200))[:1600] if record else 'ROW ' + _dump(_clip(self.rows.get(ident), 120))[:500])
        if not records:
            return None
        prompt = AUDIT + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger() + self.dropped_note()
        prompt += '\n\nLISTING RECORDS SEEN (cart lines first, then by item, latest first):\n' + '\n'.join(records)
        shown = set(picked[:24])
        unread = [i for i in latest if previews and i not in shown and i not in self.full and i not in self.opened and self.rows.get(i)]
        if unread and len(self.items) > 1:
            mine = {i for n in range(1, len(self.items) + 1) for i in self.item_ids(n)}
            unread = [i for i in unread if i in mine]
        if isinstance(self.budget, (int, float)) and self.brief.get('budget_scope') != 'each':
            unread = [i for i in unread if self.price_of(i) is None or self.price_of(i) <= self.budget]
        if unread:
            prompt += '\n\nMORE SEARCH PREVIEWS, NOT OPENED YET (latest first):\n' + '\n'.join(('ROW ' + _dump(_clip(self.rows.get(i), 90))[:260] for i in unread[:40]))
        plan = self.fit_plan(short=True) if self.bound else ''
        if plan:
            prompt += '\n\nFIT RELATIONS READ FROM THE LISTINGS (a part fits only the device whose code or series it names; a device can be swapped for another one a fitting part names):\n' + plan[:2500]
        if self.brief.get('quiet_decline') is False:
            prompt += '\n\nFor nearest only: a listing that is sold out or over the limit may be named, each of those counting as one miss, and among the listings with the fewest misses take the lowest priced.'
        late = self.elapsed() > WRAP_AT - 30
        self.effort = 'low'
        try:
            got = self.ask_json(prompt, models or [LIGHT], 8000)
            if (not models) and (not self.audit_fits(got)) and (not self.no_credit) and (not late) and self.elapsed() < CALL_STOP - 10 and HEAVY in self.judges():
                got = self.ask_json(prompt, self.judges(), 8000) or got
        finally:
            self.effort = ''
        self.trace.append({'kind': 'audit', 'got': got})
        if isinstance(got, dict):
            self.last_audit = got
        return got

    def known_id(self, token: Any) -> str:
        text = str(token or '').strip()
        if text in self.rows:
            return text
        if len(text) < 4:
            return ''
        pattern = re.compile('(?<![A-Za-z0-9])' + re.escape(text) + '(?![A-Za-z0-9])', re.I)
        hits = list(dict.fromkeys((i for i in self.seen_order if pattern.search(_dump(self.full.get(i) or self.rows.get(i) or {})))))
        return hits[0] if len(hits) == 1 else ''

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get('fits')
        if not isinstance(fits, dict):
            return {}
        cat_key = self.category_key()
        named: dict[str, list[str]] = {}
        for key, value in fits.items():
            digits = [int(d) for d in re.findall('\\d+', str(key)) if 1 <= int(d) <= len(self.items)]
            for ident in [k for k in (self.known_id(i) for i in _items(value)) if k and (not self.sold_out(k))]:
                if digits:
                    number = digits[0]
                elif len(self.items) == 1:
                    number = 1
                else:
                    number = self.owner_of(str((self.rows.get(ident) or {}).get(cat_key) or '')) if cat_key else 0
                slot = named.setdefault(str(number), []) if number else None
                if slot is not None and ident not in slot and len(slot) < 4:
                    slot.append(ident)
        return named


    def open_rows(self, rows: list[str], room: int=2) -> bool:
        many, joiner = (self.compare_target(), self.roles.get('compare'))
        tool = self.facts_tool()
        target = self.single_target(tool)
        if many and joiner and (not self.compare_bare()):
            size = max(1, self.compare_room())
            plan = [(joiner, {many: rows[i:i + size]}) for i in range(0, min(len(rows), max(room, size)), size)]
        elif tool and target:
            plan = [(tool, {target: i}) for i in rows[:room]]
        else:
            return False
        plan = plan[:max(0, self.left() - 5)]
        if not plan:
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
                whole = {i: n for i, n in self.cart_lines.items() if i not in drop and n and n > 0}
                whole.update(lines)
                return [{key: [{id_key: i, **({count_key: n} if count_key else {})} for i, n in whole.items()]}]
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
        args = _shape(self.closing_args('buy'), self.schemas.get(buy), 'args', faults)
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
        return 'Not run (Sable acted instead): the independent check confirmed ' + self.confirmed + ', so Sable set the cart (' + '; '.join(done) + ') and checked out. Store reply: ' + ' | then: '.join(replies)

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

    def closing_args(self, role: str) -> dict[str, Any]:
        tool = self.roles.get(role) or ''
        schema = self.schemas.get(tool) or {}
        props = schema.get('properties') or {}
        args = dict(self.brief.get(f'{role}_args') or {})
        for key in schema.get('required') or []:
            spec = props.get(key)
            if key not in args and isinstance(spec, dict) and 'string' in _kinds(spec) and ('enum' not in spec):
                most = spec.get('maxLength') if isinstance(spec.get('maxLength'), int) else 200
                args[key] = 'Ordered the cart, whose listings meet the requirements of the request.'[:most]
        return args

    def closest(self) -> str:
        if self.brief.get('quiet_decline') is not False:
            return ''
        if not self.last_audit and (not self.no_credit) and self.elapsed() < CALL_STOP - 10:
            self.audit_call([LIGHT])
        near = self.last_audit.get('nearest') if isinstance(self.last_audit.get('nearest'), dict) else {}
        scored = []
        for number, item in enumerate(self.items, 1):
            entry = near.get(str(number))
            ident = self.known_id(entry.get('id')) if isinstance(entry, dict) else None
            if ident and ident in self.rows and (not self.sold_out(ident)):
                misses = len([s for s in _items(entry.get('short')) if str(s).strip()])
                scored.append((misses, -(self.price_of(ident) or 0) * item.qty, ident))
        return min(scored)[2] if scored else ''

    def quiet(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        out = dict(args)
        nearest = self.closest()
        for key, spec in props.items():
            if not isinstance(spec, dict) or 'string' not in _kinds(spec) or 'enum' in spec:
                continue
            value = out.get(key)
            if nearest:
                out[key] = f'No listing meets every requirement. Closest listing: {nearest}.'[:spec.get('maxLength', 200)]
                continue
            if value is None and key not in (schema.get('required') or []):
                continue
            text = str(value or '')
            named = any((ident in text for ident in self.rows)) or self.brief.get('quiet_decline') is not False
            if named or not text:
                out[key] = 'Nothing in this store meets all of your requirements, so I am ending without buying anything.'[:spec.get('maxLength', 200)]
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
        if not removal:
            self.carted_ever += [i for i, c in lines if i not in self.carted_ever and (c is None or c > 0)]
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
        found = self.records_in(observation)
        prose = len(found) == 1 and set(found[0][1]) <= {self.id_key or 'id', 'title', 'text'}
        text = _dump(_clip(observation, SHOW_CHARS if found and (not prose) else LOOSE_CHARS))
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
        for call in calls:
            allowed = {t['function']['name'] for t in self.offered()}
            ctx = self.vet_call(call, len(calls), allowed)
            if isinstance(ctx, str):
                out.append((call['id'], ctx))
                continue
            held = self.hold_finish(ctx)
            out.append((call['id'], held if held is not None else self.perform(ctx)))
        return out

    def vet_call(self, call: dict[str, Any], batch: int, allowed: set[str]) -> dict[str, Any] | str:
        if call.get('joined'):
            return 'Included in the side-by-side result above: Sable opened these listings together in a single step.'
        fn = call.get('function') or {}
        name, raw = (fn.get('name'), fn.get('arguments'))
        if self.finished:
            return 'Skipped: the episode has ended.'
        if name == self.roles.get('ask'):
            return "Skipped (no step spent): every message to the shopper goes through Sable's interviewer."
        if name not in self.schemas:
            return f"Skipped (no step spent): no such tool. Available: {', '.join(sorted(allowed))}."
        if name not in allowed:
            return 'Skipped (no step spent): not enough steps remain for that; use: ' + ', '.join(sorted(allowed)) + '.'
        args = raw if isinstance(raw, dict) else _json_obj(raw) if (raw or '').strip() else {}
        if args is None:
            return 'Skipped (no step spent): the arguments were not a single JSON object.'
        faults: list[str] = []
        args = _shape(args, self.schemas[name], 'args', faults)
        if faults:
            return 'Skipped (no step spent): ' + '; '.join(faults[:6])
        args, wrong = self.resolve_ids(name, args)
        if wrong:
            return 'Skipped (no step spent): ' + wrong + '.'
        trimmed = ''
        many = self.compare_target() if name == self.roles.get('compare') else None
        if many and isinstance(args.get(many), list) and len(args[many]) > self.compare_room() and self.compare_bad:
            room = self.compare_room()
            trimmed = f"\n[Sable] Only the first {room} listings were sent: this store compares at most {room} at a time. Still to show: {', '.join(map(str, args[many][room:]))}."
            args = {**args, many: args[many][:room]}
        target = self.single_target(name)
        if target and name in (self.roles.get('open'), self.roles.get('offer')) and str(args.get(target)) in self.gone:
            return f'Skipped (no step spent): {args[target]} sold out at step {self.gone[str(args[target])]} and will not return. Replace its cart line with another qualifying listing.'
        if name == self.roles.get('cart') and any(((_count(v) or 0) < 0 for v in args.values() if not isinstance(v, str))):
            view = _dump(self.cart_view) if self.cart_seen_at >= self.cart_at else ''
            absent = [v for v in args.values() if isinstance(v, str) and v in self.rows and v not in self.cart_lines and v not in view and v not in self.cart_said]
            if absent:
                return f'Skipped (no step spent): {absent[0]} is not in the cart, so there is nothing to remove. Cart now: ' + self.cart_text()
        ending = self.is_buy(name, args) or self.is_refuse(name, args)
        if ending and batch > 1:
            return 'Skipped (no step spent): send the finishing call by itself, once you have read your other results.'
        if not ending and self.left() <= self.tail() and (self.roles.get('buy') or self.roles.get('refuse')):
            return "Skipped (no step spent): the final steps are reserved for checking out (with the shopper's approval of the basket when that is required) or for ending with no purchase."
        mark = name + _dump(args)
        if not ending and name != self.roles.get('cart') and mark in self.recent:
            return 'Skipped (no step spent): this exact call was just made and nothing has changed since; its result is above.'
        readers = {self.roles.get(job) for job in ('search', 'open', 'offer', 'compare')} - {None}
        cached = self.read_cache.get(mark)
        if name in readers and cached and cached[0] == self.version:
            return f'Skipped (no step spent): this same call returned the following at step {cached[1]}, and nothing has changed since:\n' + cached[2][:3000]
        if target and name in readers and name != self.roles.get('compare') and self.shown_already(name, str(args.get(target))):
            return f'Skipped (no step spent): {args[target]} already showed all this view offers since the last change. Its record: ' + self.record_of(str(args[target]))
        if many and isinstance(args.get(many), list) and args[many]:
            keep = [i for i in args[many] if not self.shown_already(name, str(i))]
            if not keep:
                return 'Skipped (no step spent): these listings already showed all this view offers since the last change. Their records:\n' + '\n'.join((f'{i}: {self.record_of(str(i))}' for i in args[many]))[:4000]
            if len(keep) < len(args[many]):
                trimmed += f"\n[Sable] Dropped {', '.join((str(i) for i in args[many] if i not in keep))}: already shown in full, unchanged since."
                args = {**args, many: keep}
                mark = name + _dump(args)
        return {'name': name, 'args': args, 'mark': mark, 'many': many, 'trimmed': trimmed, 'ending': ending, 'readers': readers}

    def hold_finish(self, ctx: dict[str, Any]) -> str | None:
        name, args = (ctx['name'], ctx['args'])
        if ctx['ending'] and self.left() > 2:
            if self.is_buy(name, args):
                held = self.hold_order()
            elif self.left() > 5:
                held = self.hold_pass()
            else:
                held = None
            if held is not None:
                return held
        if self.is_refuse(name, args):
            held = self.hold_decline()
            if held is not None:
                return held
            done = self.complete_cart()
            if done:
                return done + (' [episode finished]' if self.finished else '')
            ctx['args'] = args = self.quiet(name, args)
        if ctx['ending'] and self.left() >= 2:
            go, said = self.gate('order' if self.is_buy(name, args) else 'end')
            if not go:
                return 'Held back. ' + said + '\nCart now: ' + self.cart_text()
            if said:
                ctx['trimmed'] = '\n[Sable] ' + said + ctx['trimmed']
        return None

    def hold_order(self) -> str | None:
        wider = self.wider_hold()
        if wider:
            return wider + '\nCart now: ' + self.cart_text()
        reread = self.reread_offers()
        if self.note_sold:
            self.find_sold()
        viewer = self.roles.get('cart_view')
        if self.cart_touched and self.cart_seen_at < self.cart_at and viewer and self.left() > 3:
            faults: list[str] = []
            view_args = _shape({}, self.schemas.get(viewer), 'args', faults)
            view = None if faults else self.act(viewer, view_args)
            if view is not None:
                self.cart_view = view.get('observation')
                self.cart_seen_at = self.step
                change = self.observe(viewer, self.cart_view)
                if change:
                    self.queued = []
                    return 'Held back (no step spent). Viewing the cart revealed a change: ' + change + '\nCart now: ' + self.cart_text() + '\nAbsorb the change, repair the cart if needed, then check out.'
        swap = self.swap_hint()
        gone = [i for i in self.cart_ids() if i in self.gone] if self.held < 4 else []
        if swap or gone:
            self.held += 0 if swap else 1
            tail = '' if self.held < 3 else ' An order that still holds a sold-out line fails for certain: if no other listing qualifies, end without a purchase instead.'
            return 'Held back (no step spent). ' + (swap or f"{', '.join(gone)} in the cart has sold out: replace that line first.") + tail + '\nCart now: ' + self.cart_text()
        chain, hold = self.trail()
        if hold:
            return 'Held back (no step spent). ' + chain + '\nCart now: ' + self.cart_text()
        self.fit_links()
        misfit = self.fit_faults() if self.fit_holds < 2 and self.left() > self.tail() + 3 else []
        if misfit:
            self.fit_holds += 1
            self.trace.append({'kind': 'fit_hold', 'faults': misfit[:4]})
            return 'Held back (no step spent). A carted part is not compatible with the carted device: ' + '; '.join(misfit[:3]) + '. Change the part (or the device), then check out.\nCart now: ' + self.cart_text()
        faults = self.line_faults() if self.line_holds < 2 and self.left() > self.tail() + 2 else []
        if faults and all((' is held at ' in f for f in faults)):
            fixed = self.fix_counts(self.count_fixes())
            faults = self.line_faults() if fixed else faults
            if fixed and (not faults):
                self.line_holds += 1
                return "Held back (no step spent). Sable corrected the cart's quantities itself: " + '; '.join(fixed) + '. Where the store turned any of these down, repair those lines; then check out.\nCart now: ' + self.cart_text()
        if faults:
            self.line_holds += 1
            self.trace.append({'kind': 'line_hold', 'faults': faults[:6]})
            return 'Held back (no step spent). The cart does not yet match the request: ' + '; '.join(faults[:6]) + ". Repair those lines (using the cart tool's own way of adding or removing units), then check out.\nCart now: " + self.cart_text()
        read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()
        verdict = self.check_cart()
        if verdict:
            seen = "For the check Sable read these listings' specifications (one step each):\n" + '\n'.join(read) + '\n' if read else ''
            return 'Held back. ' + chain + seen + verdict + '\nCart now: ' + self.cart_text()
        if reread and self.cost > SPEND_SOFT:
            return "Held back (no step spent). Sable has just re-read the cart's listings:\n" + reread + '\nCheck out again once every line still shows stock and a price within the budget; otherwise replace that line first.'
        return None

    def hold_pass(self) -> str | None:
        review = self.audit()
        if review:
            return 'Held back (no step spent). ' + review
        if not self.leads:
            offer = self.near_offer()
            if offer:
                return offer
        if not self.leads or self.left() <= 4:
            return None
        if self.refuse_blocks == 0:
            self.refuse_blocks = 1
            self.trace.append({'kind': 'refuse_blocked', 'leads': list(self.leads)})
            return 'Held back (no step spent). The review turned up listings that seem to qualify: ' + ', '.join(self.leads) + '. Open them if needed, cart the best qualifying one (following the pick rule) and check out. End with no purchase only if each of them plainly fails some requirement; then name that requirement in your text and send the ending call again.'
        if self.refuse_blocks == 1:
            self.refuse_blocks = 2
            first = self.leads[0]
            self.check_cart([first], SPEND_HARD)
            if self.verdicts.get(self.check_key([first])) and first in self.leads:
                self.confirmed = first
                self.trace.append({'kind': 'refuse_blocked', 'confirmed': first})
                return f"Held back (no step spent). A separate check of {first}'s records finds it meets every requirement. Cart it at the requested quantity and check out; end with no purchase only if its record plainly breaks a requirement."
        return None

    def own_first(self, ids: list[str]) -> list[str]:
        return [i for i in ids if i in self.carted_ever] + [i for i in ids if i not in self.carted_ever]

    def near_offer(self) -> str:
        near = self.last_audit.get('nearest') if isinstance(self.last_audit.get('nearest'), dict) else {}
        cart, buy = (self.roles.get('cart'), self.roles.get('buy'))
        if self.near_tried or not near or not cart or not buy or self.brief.get('quiet_decline') is True or self.left() < 4 or self.elapsed() > CALL_STOP - 20:
            return ''
        cat_key = self.category_key()
        named = self.audit_fits(self.last_audit)
        lines: dict[str, int] = {}
        short: list[str] = []
        for number, item in enumerate(self.items, 1):
            entry = near.get(str(number))
            if not isinstance(entry, dict):
                return ''
            ident = self.known_id(entry.get('id'))
            fits = [i for i in self.own_first(named.get(str(number), [])) if i in self.full]
            if fits and (ident not in fits or fits[0] in self.carted_ever):
                entry, ident = ({'short': [], 'contradicted': False}, fits[0])
            misses = [str(s) for s in _items(entry.get('short')) if str(s).strip()]
            if ident not in self.rows or ident not in self.full or self.sold_out(ident) or entry.get('contradicted') is not False or len(misses) > 1 or ident in lines:
                return ''
            if cat_key and (not self.owns(item, str((self.rows.get(ident) or {}).get(cat_key) or ''))):
                return ''
            lines[ident] = item.qty
            short += misses
        prices = [self.price_of(i) for i in lines]
        if isinstance(self.budget, (int, float)) and self.brief.get('budget_scope') != 'each' and all((p is not None for p in prices)) and sum(((self.price_of(i) or 0) * q for i, q in lines.items())) > self.budget + 1e-09:
            return ''
        if isinstance(self.budget, (int, float)) and self.brief.get('budget_scope') == 'each' and any((p is not None and p > self.budget + 1e-09 for p in prices)):
            return ''
        for number, item in enumerate(self.items, 1):
            cap = self.item_cap(number)
            label_of = {i: str((self.rows.get(i) or {}).get(cat_key) or '') for i in lines} if cat_key else {}
            mine = [i for i in lines if (not cat_key) or len(self.items) == 1 or self.owns(item, label_of.get(i, ''))]
            if cap is not None and any(((self.price_of(i) or 0.0) > cap + 1e-09 for i in mine)):
                return ''
        drop = [i for i in self.cart_ids() if i not in lines]
        calls = self.cart_calls(lines, drop)
        if not calls or len(calls) + 2 > self.left():
            return ''
        self.near_tried = True
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
        if self.fit_faults():
            self.trace.append({'kind': 'near_offer_misfit', 'lines': lines})
            return ''
        go, said = self.gate('order') if self.left() >= 2 else (True, '')
        if not go:
            return 'Held back. ' + said + '\nCart now: ' + self.cart_text()
        faults = []
        args = _shape(self.closing_args('buy'), self.schemas.get(buy), 'args', faults)
        if faults or self.left() < 1:
            return ''
        reply = self.act(buy, args)
        self.trace.append({'kind': 'near_offer', 'lines': lines, 'short': short[:4], 'reply': _clip(reply, 200)})
        if reply is None:
            return ''
        if not self.finished:
            self.buys += 1
            self.refused_at = self.step
            self.observe(buy, reply.get('observation'))
            return 'Sable carted and ordered the nearest offer instead of ending without a purchase, but the checkout was refused: ' + self.show(reply.get('observation'))
        return 'Nothing fit every requirement cleanly, so as the request allows, Sable ordered the listings that miss the fewest requirements (' + ', '.join(lines) + '; unconfirmed: ' + ('; '.join(short) or 'nothing') + ').' + (' [episode finished]' if self.finished else '')

    def hold_decline(self) -> str | None:
        held = self.cart_ids() or list(self.cart_lines)
        prices = [self.price_of(i) for i in held]
        total = sum(((p or 0) * max(1, self.held_units(i) or 1) for p, i in zip(prices, held)))
        within = not isinstance(self.budget, (int, float)) or self.brief.get('budget_scope') == 'each' or None in prices or total <= self.budget + 1e-09
        if (self.elapsed() >= WRAP_AT - 15 or self.left() <= 3) and (self.buys == 0 or self.refused_at >= 0) and held and within and (not self.gaps()) and (not any((self.sold_out(i) for i in held))) and self.roles.get('buy'):
            self.trace.append({'kind': 'late_decline_ordered', 'lines': list(held)})
            note = self.finish(again=True)
            return 'Sable checked out the complete cart instead of ending without a purchase, since time or steps are nearly out' + (f': {note}' if note else '') + (' [episode finished]' if self.finished else '.')
        if self.refused_at >= 0 and self.after_refusal < 2 and self.left() > 2:
            self.after_refusal += 1
            self.trace.append({'kind': 'decline_after_refusal_held', 'step': self.step})
            return 'Held back (no step spent). An earlier finish was turned down because something changed, and giving up after such a change does not pay: apply the change (a new budget, a requirement added or dropped, a line sold out), put the best qualifying listing for every product in the cart and check out.'
        if self.left() <= 5 and (not self.late_held):
            lines = [i for i in self.built if not self.sold_out(i)]
            if len(lines) < len(self.items) and self.left() >= 3 and (not self.late_audit) and self.elapsed() < CALL_STOP:
                self.late_audit = True
                named = self.audit_fits(self.audit_call())
                picks = [next((i for i in named.get(str(n), []) if not self.sold_out(i)), None) for n in range(1, len(self.items) + 1)]
                if all(picks) and len(set(picks)) == len(picks):
                    lines = picks
                    self.trace.append({'kind': 'late_audit_fits', 'lines': lines})
            fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
            missing = [i for i in lines if i not in self.cart_ids() or (i in self.dropped and (not fresh))]
            if len(lines) >= len(self.items) and len(missing) + 1 <= self.left():
                self.late_held = True
                self.trace.append({'kind': 'late_decline_held', 'lines': lines, 'missing': missing})
                redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else 'Check out the cart as it is.'
                return 'Held back (no step spent). An ending with no purchase scores nothing when a qualifying listing exists, and being short of steps is no evidence that none exists. ' + redo + ' End with no purchase only if your searching showed that some requested item has no listing meeting its requirements; then send the ending call again.'
        if not self.decline_swept and self.buys == 0 and self.elapsed() < SWEEP_UNTIL + 30 and self.sweep_ready():
            self.decline_swept = True
            self.swept = set()
            found = self.sweep(self.gaps() or None)
            if found:
                return 'Held back. An ending with no purchase is rarely right while listings remain unread.\n' + found + '\n' + self.status()
        if self.brief.get('quiet_decline') is not True and (not self.near_tried) and self.last_audit.get('nearest'):
            offer = self.near_offer()
            if offer:
                return offer + (' [episode finished]' if self.finished else '')
        if not self.leads and (not self.pushed) and self.left() > self.max_steps // 3:
            self.pushed = True
            return f'Held back (no step spent). {self.left()} steps remain, so before ending try searches you have not run yet: list each requested item\'s department and page through it with the token or offset the store hands back, try one-word value queries and other names for the property, and open the listings that could qualify. Only then end with no purchase.'
        return None

    def perform(self, ctx: dict[str, Any]) -> str:
        name, args, mark, many, ending = (ctx['name'], ctx['args'], ctx['mark'], ctx['many'], ctx['ending'])
        trimmed = ctx['trimmed']
        reply = self.act(name, args)
        if reply is None:
            return 'That call failed (network or server trouble). You can try it again.'
        observation = reply.get('observation')
        if many and isinstance(args.get(many), list) and len(args[many]) > 1:
            shown = _dump(observation)
            if not any((str(i) in shown for i in args[many])):
                self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                trimmed += f'\n[Sable] The comparison view returned nothing for {len(args[many])} listings: send no more than {self.compare_room()} per call.'
            else:
                self.compare_ok = max(self.compare_ok, len(args[many]))
        self.recent = (self.recent + [mark])[-3:]
        for key, value in args.items():
            values = value if isinstance(value, list) else [value]
            if values and all((isinstance(x, str) and x in self.rows for x in values)):
                self.id_args.add((name, key))
        self.absorb(observation)
        if name == self.roles.get('compare'):
            rows = self.records_in(observation)
            if rows:
                self.facts_from[name] = self.facts_from.get(name, False) or any((_fact_block(r) for _, r in rows))
        if name in ctx['readers'] - {self.roles.get('search')}:
            self.note_view(name, observation)
        plain = {tool for job, tool in self.roles.items() if job != 'search'}
        paged = self.more_pages(name, args, observation) if name not in plain and (not ending) else ''
        if name == self.roles.get('search') and (not ending):
            paged += self.filter_note(args, observation)
        self.learn_view(name, observation)
        if name == self.roles.get('cart'):
            self.after_cart(name, args, observation)
        if name == self.roles.get('search') and self.records_in(observation):
            self.searched = (self.searched + [dict(args)])[-8:]
        if name == self.roles.get('cart_view'):
            self.cart_view = observation
            self.cart_seen_at = self.step
        if self.is_buy(name, args):
            self.buys += 1
            if not reply.get('finished'):
                self.refused_at = self.step
        if ending and (not reply.get('finished')) and (not self.finished):
            if self.is_refuse(name, args):
                self.refused_at = self.step
            trimmed += '\n[Sable] The store did not end the episode with this call: its reply carries a change from the shopper or the store. Apply it (a new budget changes which listings are affordable, so look again at the listings you set aside on price), then finish again.'
        return self.report(ctx, reply, observation, paged + trimmed, plain)

    def learn_view(self, name: str, observation: Any) -> None:
        busy = (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask'))
        if not self.single_target(name) or name in busy:
            return
        observation = _payload(observation)[0]
        listing = self.ident(observation) if isinstance(observation, dict) else None
        if not listing:
            return
        self.note_facts(name, observation)
        if name == self.roles.get('open') and self.facts_from.get(name) is False:
            other = next((t for t in self.schemas if t != name and self.single_target(t) and self.facts_from.get(t) is not False and (t not in busy + (self.roles.get('search'),))), None)
            if other:
                self.roles['open'] = other
        if self.facts_from.get(name) is False and name != self.roles.get('open') and (not self.roles.get('offer')):
            self.roles['offer'] = name
        if name == self.offer_tool():
            self.offer_seen[listing] = self.step

    def after_cart(self, name: str, args: dict[str, Any], observation: Any) -> None:
        self.note_cart(name, args)
        self.cart_said = _dump(observation)
        self.recent = []
        counts = [_count(v) for v in args.values() if not isinstance(v, str) and _count(v) is not None]
        if counts:
            for value in args.values():
                if not (isinstance(value, str) and value in self.rows):
                    continue
                if counts[0] <= 0 and (not _in_rows(observation, value)):
                    self.dropped.add(value)
                elif counts[0] > 0:
                    self.dropped.discard(value)
        lines = [i for i in self.cart_ids() if not self.sold_out(i) and i not in self.dropped]
        if len(lines) >= len(self.built):
            self.built = lines

    def empty_hint(self, name: str, args: dict[str, Any], observation: Any) -> str:
        words = max((len(v.split()) for v in args.values() if isinstance(v, str)), default=0)
        if name != self.roles.get('search') or words < 3 or self.records_in(observation) or _trouble(observation):
            return ''
        self.long_empty = True
        return '\n[Sable] No rows came back. This store may need every word to appear on a listing, so try fewer words: one wanted value with the product noun.'

    def report(self, ctx: dict[str, Any], reply: dict[str, Any], observation: Any, extra: str, plain: set[str]) -> str:
        name, args = (ctx['name'], ctx['args'])
        text = self.show(observation) + extra + self.empty_hint(name, args, observation)
        if name not in plain and (not ctx['ending']):
            text += self.cheapest_here([observation] + self.paged_obs) + self.cap_ignored(name, args, observation)
        base = text
        proof = self.evidence(observation)
        if proof:
            text += '\n' + proof
        if name == self.roles.get('cart') and any((_count(v) == 0 for v in args.values() if not isinstance(v, str))):
            if any((isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values())):
                text += '\n[Sable] A count of 0 left that line in the cart: this tool adds the count to the existing line, so removing units takes a negative count.'
        self.note_sold = False
        lines = self.loose_lines(observation)
        if lines:
            meaning = self.take_note(lines)
            if meaning:
                text += '\n[note understood] ' + meaning + '. Carry this change into your picks before checking out.'
        refused = self.is_buy(name, args) and (not reply.get('finished'))
        again = refused and any((line in self.spoken(observation) for line in self.sold_lines))
        loose = self.note_sold and (refused or (not self.note_named and (not self.is_buy(name, args))))
        if (loose or again) and self.cart_ids() and (not self.finished):
            seen = self.find_sold()
            if seen:
                text += "\n[after the refusal Sable re-read the cart's lines]\n" + seen
        self.note_sold = False
        swap = self.swap_hint()
        if swap:
            text += '\n' + swap
        if self.queued:
            text += "\n[notes that appeared on Sable's own calls] " + ' | '.join(self.queued)
            self.queued = []
        if name in ctx['readers']:
            self.read_cache[ctx['mark']] = (self.version, self.step, base)
        if reply.get('finished'):
            text += ' [episode finished]'
        return text

    def uncarted(self) -> list[str]:
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        held = set(self.cart_ids() if fresh else list(self.cart_lines))
        if len(self.items) <= 1:
            return [] if held else [item.name for item in self.items]
        missing = [item.name for number, item in enumerate(self.items, 1) if not held & set(self.item_ids(number))]
        return [] if missing and len(held) >= len(self.items) else missing

    def gaps(self) -> list[int]:
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        held = set(self.cart_ids() if fresh else list(self.cart_lines))
        if len(self.items) <= 1:
            return [] if held else [1]
        missing = [n for n in range(1, len(self.items) + 1) if not held & set(self.item_ids(n))]
        return [] if missing and len(held) >= len(self.items) else missing

    def fill_due(self) -> bool:
        if self.gap_filled or self.buys or not self.roles.get('cart') or not self.roles.get('buy'):
            return False
        if self.roles.get('ask') and (not self.asked) and self.msgs_free() > 0:
            return False
        open_items = self.gaps()
        if not open_items or self.left() < len(open_items) + 1 or self.elapsed() > CALL_STOP - 15:
            return False
        return self.left() <= len(open_items) + 2 or (self.elapsed() >= WRAP_AT - 45 and all((self.records_of_item(n) for n in open_items)))

    def fill_gaps(self) -> str:
        open_items = self.gaps()
        if self.gap_filled or not open_items or self.no_credit or self.broke or self.elapsed() > CALL_STOP - 15:
            return ''
        self.gap_filled = True
        named = self.audit_fits(self.audit_call([LIGHT] if self.cost > SPEND_SOFT else None))
        cat_key = self.category_key()
        held = set(self.cart_ids()) | set(self.cart_lines)
        lines: dict[str, int] = {}
        for number in open_items:
            item = self.items[number - 1]
            pick = next((i for i in self.own_first(named.get(str(number), [])) if i not in held and i not in lines and not self.sold_out(i) and (not cat_key or len(self.items) == 1 or self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or '')))), None)
            if not pick:
                self.trace.append({'kind': 'gap_fill_none', 'item': number})
                offer = self.near_offer()
                return offer if offer else ''
            lines[pick] = item.qty
        prices = [self.price_of(i) for i in list(held) + list(lines)]
        if isinstance(self.budget, (int, float)) and self.brief.get('budget_scope') != 'each' and all((p is not None for p in prices)):
            total = sum(((self.price_of(i) or 0) * self.held_units(i) for i in held)) + sum(((self.price_of(i) or 0) * q for i, q in lines.items()))
            if total > self.budget + 1e-09:
                self.trace.append({'kind': 'gap_fill_budget', 'total': total})
                return ''
        calls = self.cart_calls(lines, [])
        if not calls or len(calls) + 1 > self.left():
            return ''
        cart = self.roles['cart']
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
        self.trace.append({'kind': 'gap_fill', 'lines': lines, 'calls': len(done)})
        if not done:
            return ''
        names = '; '.join((f'item {n}: {i}' for n, i in zip(open_items, lines)))
        return f'[Sable] Steps or time are nearly gone and these products had no cart line, so Sable carted the listing a close review found to meet their requirements ({names}). The cart now has a line for every product: check out now. Swap a line only for a listing you have read that fits better, keeping one line per product.'

    def cart_short(self) -> bool:
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        lines = self.cart_ids() if fresh else list(self.cart_lines)
        return len(self.items) > 1 and 0 < len(set(lines)) < len(self.items)

    def finish(self, again: bool=False) -> str:
        if self.finished or self.broke or self.left() <= 0:
            return ''
        if self.buys == 0 and self.gaps():
            self.fill_gaps()
        if self.buys == 0 and self.gaps() and (not self.finished):
            self.near_offer()
        if self.finished or self.left() <= 0:
            return ''
        buy, refuse = (self.roles.get('buy'), self.roles.get('refuse'))
        self.swap_gone_lines()
        if self.finished or self.left() <= 0:
            return ''
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if buy and any((self.sold_out(i) for i in self.cart_ids())):
            self.trace.append({'kind': 'sold_out_decline', 'lines': [i for i in self.cart_ids() if self.sold_out(i)]})
        elif buy and (holding or (self.cart_touched and (not fresh) and (self.left() >= 2))) and self.cart_short():
            self.trace.append({'kind': 'short_cart_decline', 'items': len(self.items)})
        elif buy and (holding or (self.cart_touched and (not fresh) and (self.left() >= 2))):
            faults: list[str] = []
            args = _shape(self.closing_args('buy'), self.schemas.get(buy), 'args', faults)
            if not faults:
                if self.left() >= 2 and self.elapsed() < CALL_STOP:
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
                if self.left() >= 2 and self.elapsed() < CALL_STOP:
                    self.gate('end')
                if self.finished or self.left() <= 0:
                    return ''
                reply = self.act(refuse, args)
                self.trace.append({'kind': 'forced_refuse', 'reply': _clip(reply, 200)})
                if reply is not None and (not self.finished):
                    self.observe(refuse, reply.get('observation'))
                    return 'the no-purchase ending was refused: ' + self.show(reply.get('observation'))
        return ''

    def swap_gone_lines(self) -> None:
        held = self.cart_ids()
        lost = [i for i in held if self.sold_out(i)]
        cart = self.roles.get('cart')
        cat_key = self.category_key()
        if not lost or not cart or not cat_key or self.left() < 2:
            return
        lines = {i: max(1, self.held_units(i) or 1) for i in held if i not in lost}
        for ident in lost:
            label = str((self.rows.get(ident) or {}).get(cat_key) or '')
            number = next((n for n, item in enumerate(self.items, 1) if self.owns(item, label)), 0)
            if not number:
                return
            item = self.items[number - 1]
            pool = self.sweep_fits.get(number, []) + self.swap_fits + self.leads
            pick = next((p for p in pool if p in self.rows and p not in lines and (not self.sold_out(p)) and self.owns(item, str((self.rows.get(p) or {}).get(cat_key) or ''))), None)
            if not pick:
                return
            lines[pick] = item.qty
        prices = [self.price_of(i) for i in lines]
        if isinstance(self.budget, (int, float)) and self.brief.get('budget_scope') != 'each' and all((p is not None for p in prices)) and sum(((self.price_of(i) or 0.0) * q for i, q in lines.items())) > self.budget + 1e-09:
            return
        calls = self.cart_calls(lines, lost)
        if not calls or len(calls) + 1 > self.left():
            return
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), 'args', faults)
            if faults:
                return
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return
            self.note_cart(cart, args)
            self.observe(cart, reply.get('observation'))
        self.trace.append({'kind': 'swap_gone', 'lost': lost, 'lines': lines})

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
        opening = self.read_any('', {}, self.view.get('opening'))
        if opening is not None:
            self.absorb(opening)
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            intro += '\n\nWHAT THE STORE SHOWS FIRST:\n' + self.show(opening)
        intro += '\n\nBegin by searching each item in its department and opening several of its listings in full, so the interviewer learns which properties those listings carry.\n\n' + self.status()
        self.chat = [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': intro}]
        self.work(LOOP_CALLS, SPEND_HARD)
        if not self.finished:
            turned = self.finish(again=True)
            if turned and (not self.finished) and (not self.broke) and (not self.no_credit) and (self.left() > 1):
                self.chat.append({'role': 'user', 'content': '[Sable] Sable tried to end the episode and ' + turned + '\nAct on what the store says, then finish.\n' + self.status()})
                self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
            if not self.finished:
                self.finish()
        self.trace.append({'kind': 'end', 'finished': self.finished, 'steps': self.step, 'cost': round(self.cost, 5), 'calls': self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.finished and (not self.broke) and (self.left() > 0) and (self.loop_calls < loop_cap) and (self.cost < spend_cap) and (self.elapsed() < WRAP_AT):
            if self.fill_due():
                filled = self.fill_gaps()
                if filled:
                    self.tell(filled + '\n' + self.status())
                    continue
            if (self.asked or not self.roles.get('ask') or self.msgs_free() <= 0) and self.chase_due():
                routed = self.chase()
                if routed:
                    self.tell(routed + '\n' + self.status())
                    continue
            if self.sweep_due():
                swept = self.sweep(None if self.brief.get('pick') == 'cheapest' else self.sweep_targets())
                if swept:
                    self.tell(swept + '\n' + self.status())
                    continue
            if self.ready_to_ask():
                summary = self.interview()
                if summary:
                    self.chat.append({'role': 'user', 'content': summary + '\n' + self.status()})
                continue
            self.trim()
            self.loop_calls += 1
            message = self.think(self.chat, [LIGHT, SPARE], 5000, self.offered())
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
                    self.chat.append({'role': 'tool', 'tool_call_id': call['id'], 'content': 'Skipped (no step spent): the shopper gets asked first.'})
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
        episode = Sable(problem_data)
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
