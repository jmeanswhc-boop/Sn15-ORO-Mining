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
SPEND_SOFT = 0.052
SPEND_HARD = 0.078
HEAVY_ROOM = 0.02
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 320
ROW_CUT = 150
OBS_LIMIT = 9000
HISTORY_LIMIT = 70000
GATE_ROUNDS = 3
AUTO_PAGES = 6
CHEAPER_OPENS = 3
EPISODE_SECONDS = 300
EXPLORE_RESERVE = 50
CALL_RESERVE = 15
TIGHT_MESSAGES = 2
RECOVERY_STEPS = 8
BUSY_SECONDS = 20
SWEEP_STEPS = 6
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
GUIDE = 'You are Kyo, a careful buyer working for one shopper in an online store. You can act only by calling the store\'s tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.\n\nHOW THE STORE WORKS\n- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool\'s description and schema and use the exact argument names and allowed values.\n- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.\n- Search rows are summaries. Stores split a listing\'s record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Kyo knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.\n- Questions to the shopper are handled for you by Kyo\'s interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.\n- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Kyo carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store\'s tools (product questions go through Kyo\'s interviewer).\n\nWHAT COUNTS AS A FIT\n- The kind of product is decided by the listing\'s category label: only the label that matches the requested item as a whole counts, never a neighbouring one.\n- A requirement on a property is met when the listing\'s attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value\'s words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.\n- An attribute whose key names the requirement\'s property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing about the requirement. Kyo adds [evidence] lines after opened listings with the keys whose names match a requirement.\n- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.\n- When the category label itself states a requirement\'s value (a season or audience in the label), that requirement is met by the category.\n- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing\'s value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.\n- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.\n- When one requested item must fit another (an accessory for a device), choose the two together from the listings\' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.\n- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.\n- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.\n\nCHOOSING\n- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Kyo reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.\n- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.\n- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.\n- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.\n- A request for several different items needs one cart line per item, each at its requested quantity.\n\nCHANGES WHILE YOU WORK\n- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool\'s result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.\n- Re-read a chosen listing\'s record shortly before checking out, because price and stock can change silently.\n\nFINISHING\n- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.\n- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request\'s wording for that ending, and when a remark is needed, keep it short and name no listing.\n- Every reply must call at least one tool. Keep your text short: what you know, then the next move.'
BRIEF = 'You prepare a shopping task for an agent. Read the shopper\'s request and the store\'s tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:\n\n"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request\'s words, "shelf": the store\'s category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request\'s own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A clause that gives one of the product\'s properties a value ("where the <property> is <value>") is always a must of that product, never its shelf. Copy only the value itself: a verb or filler word that merely closes the sentence after it is not part of the value. A value that reads like an absence ("<property> is not available") is the literal value the listing must show, not an exclusion. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product, and the words the request uses for how the value relates to that listing\'s value (matching it, unlike it, going with it) stay as the request wrote them, never reduced to \"same as\"}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property\'s value or limit outright; and never turn words that are part of what the product is called (a season or an audience named with the product) into musts: those belong to "name". A use or occasion the request makes a property of the product itself ("blazers for business wear", "a dress made for parties", "the occasion has to be party") IS a must on that property ("occasion: business"), unless the store\'s category label already names it; an event, person or setting the product is bought for ("to wear at his wedding", "for a colleague"), and how it should sit, feel or look in use, stay context. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.\n"budget": the most the shopper will pay in total as a number in the request\'s own unit, or null.\n"budget_unit": that unit in words (for example "cents"), or null.\n"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.\n"caps": a list of {"item": item number, "max_price": number in the budget\'s unit} for each price limit the request sets on ONE product only (not the whole order), else [].\n"in_stock_only": true unless the request clearly accepts items that are not available.\n"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.\n"pick_measure": what is maximised or minimised for "most"/"least", else "".\n"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.\n"steps": how many steps, moves or tool calls the request allows (integer), or null.\n"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].\n"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.\n"wishes": a list of things the shopper would merely like.\n"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).\n"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper\'s approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.\n"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.\n"fits": a list of {"item": item number, "with": item number} for each requested item that the request says must work with, fit or suit ANOTHER requested item (the one the agent picks); [] when none.\n"browse": how a tool lists one store category, when one can: {"tool": exact tool name, "category": the argument that takes the category label, "max_price": the argument that takes a maximum price, or null, "fixed": the other arguments that put the tool in that mode, else {}}; null when no tool lists a category.\n"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing\'s attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing\'s current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).\n"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.\n"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.\n'
PLAN = 'You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property; a property the shopper has no view on gets "don\'t mind", and one the shopper does not think about at all gets a puzzled reply, so every property named spends part of a message.\n\nYou get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item\'s category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.\n\nChoose the properties in this order:\n0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.\n1. Hinted properties. The request\'s story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product\'s looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.\nName a hinted property by a listed key of that item when one fits the hint (a hint about who will wear it: the key for gender or audience; about its looks: the listed key for colour, pattern or finish that most of its listings carry); a property no listing of the item carries comes after the item\'s three most carried keys.\n2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item\'s listings; once five or more listings were read, avoid keys seen on only one or two of them.\nSkip a property the request already settles for that item (a requirement names that property; a key whose values only repeat a required value, such as a pattern key showing the required decoration, is NOT settled), a second name for a property already chosen, a key whose values each list several purchase options at once, identifiers, model names, warranty and other logistics. Pack sizes, counts and quantities per pack are properties like any other, and a key marked [not the stated ...] is another property than that requirement, whatever words its name shares with it. Rank every item on its own: its hinted properties first, then its listed keys.\n"label" is the property\'s plain name: for a listed key, all of the key\'s words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property\'s name, no example values, no brackets, no second property.\nChoose up to COUNT for each item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story\'s words it comes from, or an empty string"}]}, each item\'s picks in rank order'
HEAR = 'A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "budget": new total limit as a number or null}\n- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3").\n- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop\'s product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer\'s own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.\n- "free": the property of each question the shopper said they do not mind about.\n- Folksy or roundabout wording that names a value still states that value.\n- The value is only what the property should be. Words that merely join a property to its value (such as matching, set to or coming in as) are not part of it: "size set to large" states large. This does not touch a value pointed at on another listing, which is copied whole.'
NOTE = 'While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:\n{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request\'s unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "price_fell": true when it says a price fell, dropped or got lower (whichever listing), "summary": "one short sentence"}\nA replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note.'
CHECK = 'You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right. The REQUIREMENTS IN FORCE carry the shopper\'s later changes and outrank the request text: a requirement marked DROPPED is no longer needed, even if the request still asks for it.\nReading rules: a requirement is met when the listing\'s attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value\'s words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a measured limit (a number with a unit) is met only under a key that shares a word with the property: the same number under a key for a different measure, or only in the title, leaves it unproven and the listing fails it; a list under the plural of the property\'s key shows the variants on sale and does not give this listing\'s own value when the title names a different one of them; a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing\'s value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.\nWhen the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart\'s choice on that rule, the cart is wrong.\nReturn ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses.'
READING = 'the listing\'s attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value\'s words counts, such as "deep teal" for teal, limits include the edge); a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a measured limit (a number with a unit) counts only under a key sharing a word with the property, never from a key for a different measure or from the title alone; a list under the plural of the property\'s key shows the variants on sale, not this listing\'s own value, when the title names a different one; a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing\'s value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit; the requirements in force outrank the request text, and one marked DROPPED no longer counts.'
FILL = 'The time for this shopping task is almost over and some requested items still have no line in the cart. For each item listed under ITEMS WITH NO CART LINE, choose from its opened listings the one whose attributes meet every requirement in force for that item, with these reading rules: ' + READING + '\nA listing whose page leaves a requirement unstated or states another value does not qualify. When the pick rule asks for the lowest price, take the cheapest listing that qualifies. Keep every choice inside the budget together with the lines already in the cart, and keep any fit the request asks for (part codes or series named by the device in the cart or the request). Give null for an item when none of its listings clearly qualifies.\nReturn ONE JSON object and nothing else: {"pick": {"item number": "listing id or null"}, "reason": "one sentence"}'
NEAR = 'A shopping agent is about to end without buying, because no listing it read meets every requirement. First read the REQUEST alone: does the shopper say what to do when nothing fits exactly, for example to buy whichever listing comes nearest to the requirements, perhaps with a limit for that case such as a share over the budget? Answer true only when the request itself says so; a request that only says to stop or tell them when nothing fits is false. If it is true, pick for each item, from that item\'s opened listings below, the one that misses the fewest requirements in force, following any limit the request sets for that case, and say what each pick misses.\nReturn ONE JSON object and nothing else: {"wants_closest": true or false, "rule": "the request\'s own words for the no-fit case, or empty", "pick": {"item number": "listing id or null"}, "missed": "one sentence"}'
AUDIT = 'A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: ' + READING + '\nRecords marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.\nThe requirements were written down by a quick reader, so first compare each one with the REQUEST and the ANSWERS FROM THE SHOPPER. A requirement that only repeats the story (who it is for, when, where or why it will be used, what went wrong before, how it should feel) and that the request does not demand of the product goes under "not_firm". A value that picked up words which only join a property to its value goes under "corrected" with the plain value. Judge the listings without the not_firm requirements and with the corrected values. A requirement the request or an answer states outright stays, even when no listing meets it: ending without a purchase is then right.\nReturn ONE JSON object and nothing else: {"not_firm": [{"item": item number, "need": the requirement copied exactly}], "corrected": [{"item": item number, "need": the requirement copied exactly, "value": "property: plain value"}], "fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "search": [at most 4 of {"item": item number, "query": "a keyword search of two or three words, not tried yet, that could surface a fitting listing: another name for the product with one required value (long queries often match nothing)"}], "reason": "one sentence"}'
UNMIX = 'A shopper\'s request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request\'s wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}'
TRAIL = 'A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing\'s page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement\'s chain from the requirement\'s own description, link by link, through the pages\' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).\nReturn ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link (the record\'s own id, never a code printed in its title), else "", "about": how the requirement or the previous page describes that listing, "find": "for a link not found: search words (its shop\'s name and the values the description gives, leaving out property names and the kind of product)"}], "done": true when the last link is an OPENED page that states the property\'s value itself and whose own words do not pass the reader on to another listing for that property (a page that hands the value to a different listing is not the last link, even if its attributes list a value) (also when the requirement asks for a value other than or unlike that one: the page\'s value is what it is measured against), else false, "value": the property\'s value as stated on that last page; but when the request\'s own words ask for a value that pairs with, goes with or partners that one (a counterpart, not that value itself) and the page gives a pairing rule saying what its value goes with, the value that rule pairs with it; words that only ask to match it or be the same as it ask for that value itself; or ""}]}\nA record is a link only when it matches the whole description of that link: the shop or brand, the kind of product AND every property value the description gives (for example "a 10 pieces multipack" rules out the same shop\'s listing with 10 capsules and no multipack); when no record matches all of it, that link is not found yet, even if the same shop sells other listings of that kind. Give "find" for such a link: the shop\'s name and the description\'s values, as a few search words, leaving out property names and the kind of product (a description like "Brandt\'s lamps with shade linen and base oak" gives "Brandt linen oak"): the store\'s search needs every word to be in the listing, and property names or kind words are often absent from it.\nList each chain\'s links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown.'
GATE = 'A shopping agent is about to STAGE. The shopper\'s request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product\'s properties do not do a step.\nWhen a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.\nReturn ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}'
SWAP = "A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: " + READING + '\nReturn ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line\'s item, at most 6], "reason": "one sentence"}'
SHAPE = 'A store tool returned the result below. Name its parts by the exact keys the result and the tool\'s parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing\'s price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}'
GUARD_KEYS = 3
GUARD_SHARE = 0.0
GUARD_GAP = 0.34
NOT_PROPERTY = '\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b'
CATCH_ALL = '^(?!.*\\b(?:types?|styles?|kinds?|shapes?)\\b).*\\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\\w*|accessor\\w*|extras?|misc\\w*)\\b'
WIDEN_EACH = 6
FIT_CODE = '[A-Z][A-Za-z]{0,7}(?:[- ]?\\d{3,6}|[- ]?\\d{1,5}[xX]{1,4})'
PART_CODE = '[A-Z]{1,5}\\d{0,3}-?\\d{3,7}'
PART_SERIES = '[A-Z]{1,5}\\d{1,3}'
FIT_WORDS = '(?:works? (?:with|on)|compatible with|compatibility|fitment|fits?|to fit|for use with|usable with|designed for|made for|built for|intended for|suits?|goes with|pairs? with|replacement for|replaces|supports?)'
FIT_EXCEPT = '\\b(?:except(?:\\s+for)?|excluding|other than|but not|apart from|not including|not for)\\b'

REACT_STEPS = 3
LATE_EFFORT = 90
LATE_ORDER = 215
SWEEP_BIG = 60
SWEEP_SPARE = 20
SERIES_PINNED = 2
SERIES_PAGES = 6
PIPE_CUT = re.compile('\\s\\|\\s(?=[A-Za-z_][A-Za-z0-9_]{0,40}:(?:\\s|$))')

def _plain_value(text):
    word = text.strip()
    if word in ('True', 'False'):
        return word == 'True'
    if word in ('None', 'null'):
        return None
    if re.fullmatch('-?\\d+', word):
        return int(word)
    if re.fullmatch('-?\\d+\\.\\d+', word):
        return float(word)
    pairs = [chunk.split('=', 1) for chunk in word.split(';') if chunk.strip()]
    if len(pairs) > 1 and all((len(p) == 2 and p[0].strip() for p in pairs)):
        return {p[0].strip(): _plain_value(p[1]) for p in pairs}
    return word

def _readable(node: Any, depth: int=0) -> Any:
    if depth > 8:
        return node
    if isinstance(node, dict):
        return {k: _readable(v, depth + 1) for k, v in node.items()}
    if isinstance(node, list):
        return [_readable(v, depth + 1) for v in node]
    if not isinstance(node, str) or not re.match('\\s*[A-Za-z_][A-Za-z0-9_]{0,40}:', node):
        return node
    pieces = PIPE_CUT.split(node.strip())
    if len(pieces) < 3:
        return node
    record = {}
    for piece in pieces:
        key, _, value = piece.partition(':')
        record[key.strip()] = _plain_value(value)
    return record

def _codes(text: Any) -> list[str]:
    return [re.sub('[\\s-]+', '', c).lower() for c in re.findall('\\b' + FIT_CODE + '\\b', str(text or ''))]

def _code_names(stated, code):
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
    codes = []
    for hit in re.finditer('\\bmodel\\s*(?:no\\.?|number|#)?\\s*[:.]?\\s*(' + FIT_CODE + ')\\b', text, re.I):
        codes += _codes(hit.group(1))
    for hit in re.finditer('\\b(?:also sold as|also known as|sold also as|sold as)\\s+([^.;]+)', text, re.I):
        codes += _codes(hit.group(1))
    works = []
    deny = []
    for hit in re.finditer('\\b(?<!not )(?<!never )' + FIT_WORDS + '\\s*:?\\s+([^.;]+)', text, re.I):
        if re.search('\\bpart\\s+' + PART_CODE + '\\s+(?:also\\s+)?$', text[max(0, hit.start() - 40):hit.start()], re.I) or re.match('part\\s+series', hit.group(1), re.I):
            continue
        sides = re.split(FIT_EXCEPT, hit.group(1), maxsplit=1, flags=re.I)
        works += _codes(sides[0])
        if len(sides) > 1:
            deny += _codes(sides[1])
    also = [(hit.group(1).upper(), _codes(hit.group(2))) for hit in re.finditer('\\bpart\\s+(' + PART_CODE + ')\\s+also fits\\s*:?\\s*([^.;]+)', text, re.I)]
    series = []
    for hit in re.finditer('\\bfits?\\s+part\\s+series\\s*:?\\s*([^.;]+)', text, re.I):
        series += [c.upper() for c in re.findall('\\b' + PART_SERIES + '\\b', hit.group(1))]
    return {'own': own.group(1).upper() if own else '', 'codes': codes, 'works': works, 'deny': deny, 'also': also, 'series': series}

def _fit_verdict(part, device, allowed):
    if allowed and device.get('codes') and any((_code_names(a, c) for a in allowed for c in device['codes'])) and (not any((_code_names(d, c) for d in part.get('deny') or [] for c in device['codes']))):
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

def _record_line(text: str) -> bool:
    fields = [part for part in text.split(' | ') if re.match('\\s*[\\w .-]{1,40}:\\s', part)]
    return len(fields) >= 3


def _split_value(want: str, have: Any) -> bool:
    if not isinstance(have, str) or re.search('\\b(?:least|most|not|without|than|between|except)\\b', want, re.I):
        return False
    wanted = [w.rstrip('s') for w in re.findall('[a-z0-9]+', want.lower())]
    if len(wanted) < 2:
        return False
    parts = [[w.rstrip('s') for w in re.findall('[a-z0-9]+', part)] for part in re.split('[,;/|]', have.lower())]
    if any((part[i:i + len(wanted)] == wanted for part in parts for i in range(len(part)))):
        return False
    for part in parts:
        place = 0
        for word in part:
            if place < len(wanted) and word == wanted[place]:
                place += 1
        if place == len(wanted):
            return True
    return False


def _choice_key(key):
    return bool(re.search('option|variant|choice|choose|select', str(key), re.I))


def _flip_units(text: str) -> str:
    joined = re.sub('\\b(\\d+(?:\\.\\d+)?) ([a-z]{1,3})\\b', '\\1\\2', text, flags=re.I)
    if joined != text:
        return joined
    return re.sub('\\b(\\d+(?:\\.\\d+)?)([a-z]{1,3})\\b', '\\1 \\2', text, flags=re.I)


def _judge_view(record: Any, room: int, cut: int) -> str:
    plain = _dump(_clip(record, cut))
    if len(plain) <= room or not isinstance(record, dict):
        return plain[:room]
    flat = {k: v[:400] if isinstance(v, str) else v for k, v in record.items() if not isinstance(v, (dict, list))}
    nested = {k: v for k, v in record.items() if isinstance(v, (dict, list))}
    head = _dump(flat)
    return (head[:room] + ' ' + _dump(_clip(nested, 60))[:max(0, room - len(head))]).strip()


def _fold(word):
    word = word.lower()
    for tail in ('ing', 'able', 'ible', 'ed'):
        if len(word) - len(tail) >= 4 and word.endswith(tail):
            return word[:-len(tail)]
    return word.rstrip('s')


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

def _compact_row(row):
    out = {}
    for key, value in row.items():
        if value is None or value == '' or value == [] or value == {}:
            continue
        if isinstance(value, str) and re.search('desc|detail|text|summary|about', str(key), re.I):
            value = value[:90]
        elif isinstance(value, str) and re.search('title|heading|name', str(key), re.I):
            value = value[:110]
        out[key] = value
    return out


def _row_cut(value: Any, limit: int, depth: int=0) -> Any:
    if depth > 7:
        return '...'
    if isinstance(value, list):
        rows = len(value) >= 2 and all((isinstance(v, dict) for v in value))
        return [_clip(_compact_row(v), min(limit, ROW_CUT), depth + 1) if rows else _row_cut(v, limit, depth + 1) for v in value]
    if isinstance(value, dict):
        return {k: _row_cut(v, limit, depth + 1) for k, v in value.items()}
    return _clip(value, limit, depth)

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
            pass
    found = None
    reader = json.JSONDecoder()
    at = text.find('{')
    while at >= 0:
        try:
            got, end = reader.raw_decode(text, at)
        except ValueError:
            at = text.find('{', at + 1)
            continue
        if isinstance(got, dict):
            found = got
        at = text.find('{', end)
    return found

def _words(key):
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

def _names(prop, key):
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

def _shape(value, schema, where, faults):
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
        room = schema['maxLength']
        head = value[:room]
        if not value[room].isspace() and ' ' in head.strip():
            head = head[:head.rstrip().rfind(' ')]
        value = head.rstrip() or value[:room]
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

def _id_like(value, named=False):
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

def _fact_block(node):
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
        self.told = []
        self.dropped = []
        self.free = []

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
        self.trace = []
        self.brief = {}
        self.roles = {}
        self.items = []
        self.budget = None
        self.msg_limit = 0
        self.msg_used = 0
        self.process = []
        self.reserve = 0
        self.gate_msgs = 0
        self.gate_ok = set()
        self.done_log = []
        self.done_steps = []
        self.asked = False
        self.qa = []
        self.rows = {}
        self.full = {}
        self.seen_order = []
        self.heard = set()
        self.notes = []
        self.version = 0
        self.cart_touched = False
        self.cart_at = -1
        self.cart_seen_at = -1
        self.cart_view = None
        self.note_named = False
        self.cart_lines = {}
        self.checked = set()
        self.audited = False
        self.offer_seen = {}
        self.gone = {}
        self.unavailable = set()
        self.sold_lines = set()
        self.note_sold = False
        self.held = 0
        self.id_args = set()
        self.cart_said = ''
        self.refused_at = -1
        self.leads = []
        self.refuse_blocks = 0
        self.pushed = False
        self.built = []
        self.dropped = set()
        self.late_held = False
        self.lead_items = {}
        self.verdicts = {}
        self.confirmed = None
        self.completed = False
        self.filled = False
        self.id_key = None
        self.facts_from = {}
        self.queued = []
        self.buys = 0
        self.refusals = 0
        self.recent = []
        self.chat = []
        self.opened = set()
        self.refs = []
        self.ref_values = {}
        self.trails = 0
        self.trail_blocks = 0
        self.compare_ok = 0
        self.compare_bad = 0
        self.tool_keys = {}
        self.view_keys = {}
        self.read_cache = {}
        self.pages_read = 0
        self.shapes = {}
        self.price_key = None
        self.stock_key = None
        self.cheaper_marks = set()
        self.wider_held = False
        self.paged_obs = []
        self.browse = None
        self.browse_misses = 0
        self.model_busy = {}
        self.swept = set()
        self.plan_order = []
        self.line_holds = 0
        self.bound = []
        self.fit_holds = 0
        self.searched = []
        self.link_searches = 0
        self.link_seen = {}
        self.fresh_changes = []
        self.looked_again = 0
        self.auto_reads = 0
        self.last_look = False
        self.near_tried = False
        self.near_held = False
        self.referred = set()
        self.said_clash = False
        self.said_gap = False
        self.wipes = {}
        self.end_read = set()
        self.decline_proven = False
        self.chain_retried = False
        self.dear_done = False
        self.dead_caps = set()
        self.effort_off = False
        self.weak_said = False
        self.dead_codes = set()
        self.fit_reads = 0
        self.fit_pages = 0
        self.swept_items = set()
        self.code_held = False
        self.named_refs = set()
        self.read_by = {}
        self.end_tried = False
        self.series_swept = set()
        self.search_seen = []
        self.keys_out_n = 0
        self.chain_hunted = set()
        self.pending_held = []
        self.caps = {}
        self.gap_alt = {}
        self.swapped = False
        self.set_done = False
        self.chain_ids = []
        self.skip_args = {}
        self.query_seen = {}

    def left(self):
        return self.max_steps - self.step

    def time_left(self) -> float:
        return EPISODE_SECONDS - (time.monotonic() - self.started)

    def msgs_free(self) -> int:
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)

    def cart_mark(self):
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

    def compare_facts(self):
        return self.facts_from.get(self.roles.get('compare') or '')

    def compare_bare(self) -> bool:
        seen = self.compare_facts()
        return seen is False or (seen is None and (self.facts_from.get(self.facts_tool() or '') is True or bool(self.roles.get('offer'))))

    def gate_due(self, stage: str) -> bool:
        return any((p['when'] == stage for p in self.process)) and (stage, self.cart_mark() + f'#{self.version}') not in self.gate_ok

    def tail(self):
        talk = any((p['when'] == 'order' and p['messages'] > 0 for p in self.process))
        spare = 1 + (REACT_STEPS if self.cart_complete() else 0) if self.buys == 0 else 0
        return spare + (2 if talk and self.msg_used < self.msg_limit and self.gate_due('order') else 1)

    def judges(self, limit: float=SPEND_SOFT) -> list[str]:
        if self.cost <= limit and self.cost + HEAVY_ROOM <= SPEND_HARD:
            return [HEAVY, LIGHT]
        return [LIGHT] if self.cost <= SPEND_HARD else []

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None=None, deep: bool=False) -> dict[str, Any] | None:
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
                    if tools and effort is None and (not self.effort_off) and EPISODE_SECONDS - self.time_left() > LATE_EFFORT:
                        effort = 'low'
                    message, status = self.call_model(choice, messages, max_tokens, tools, effort)
                    if message is None and effort and status == 400 and tools:
                        self.effort_off = True
                        message, status = self.call_model(choice, messages, max_tokens, tools, None)
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

    def call_model(self, model, messages, max_tokens, tools, effort=None):
        sent = messages
        if model.startswith('anthropic/') and len(messages) > 1:
            sent = [dict(m) for m in messages]
            marks = [0] + ([len(sent) - 1] if len(sent) > 1 else [])
            for index in marks:
                content = sent[index].get('content')
                if isinstance(content, str) and content:
                    sent[index]['content'] = [{'type': 'text', 'text': content, 'cache_control': {'type': 'ephemeral'}}]
        body = {'model': model, 'messages': sent, 'max_tokens': max_tokens, 'temperature': 0}
        if tools:
            body['tools'] = tools
            body['tool_choice'] = 'auto'
        if effort:
            body['reasoning'] = {'effort': effort}
        result = _llm.post_verbose('/inference/chat/completions', body)
        data = result.data
        if isinstance(data, dict) and data.get('choices'):
            usage = data.get('usage') or {}
            if isinstance(usage.get('cost'), (int, float)):
                self.cost += float(usage['cost'])
            message = data['choices'][0].get('message') or {}
            if tools and (not message.get('tool_calls')) and data['choices'][0].get('finish_reason') == 'length':
                self.trace.append({'kind': 'llm_fail', 'model': model, 'status': 0, 'body': 'cut at max_tokens with no call'})
                return (None, 0)
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
        if not models or self.calls >= ALL_CALLS or self.time_left() < CALL_RESERVE + 20:
            return None
        self.calls += 1
        body = {'model': models[-1], 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': max_tokens, 'temperature': 0}
        answer = _llm.post('/inference/chat/completions', body) or {}
        choices = answer.get('choices') if isinstance(answer, dict) else None
        spent = (answer.get('usage') or {}).get('cost') if isinstance(answer, dict) else None
        if isinstance(spent, (int, float)) and (not isinstance(spent, bool)):
            self.cost += float(spent)
        return _json_obj(((choices or [{}])[0].get('message') or {}).get('content')) if choices else None

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
                self.step = data['step'] if isinstance(data.get('step'), int) else self.step + 1
                if data.get('finished'):
                    self.finished = True
                seen = _readable(data.get('observation'))
                data['observation'] = seen
                if not (isinstance(seen, dict) and seen.get('error')):
                    shown = self.read_by.setdefault(name, set())
                    for value in args.values():
                        shown.update((v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)))
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

    def read_request(self):
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
        self.written_counts()
        merged = []
        became = {}
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
        self.scope_musts()
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
            value = text.partition(':')[2]
            others = [item for n, item in enumerate(self.items, 1) if n != number]
            if value and (not re.search('\\b[A-Z][a-z]{2,}', value)) and any((_roots(item.name) and _roots(item.name) <= _roots(value) for item in others)):
                self.trace.append({'kind': 'relation_not_pointer', 'need': text})
                continue
            if text and re.search('\\bsame\\b', text, re.I):
                text = self.relation_words(text)
            if text and (number, text) not in self.refs:
                self.refs.append((number, text))
        self.unmix()
        for entry in _items(brief.get('caps')):
            if isinstance(entry, dict):
                number, cap = (_count(entry.get('item')) or 0, entry.get('max_price'))
                if 1 <= number <= len(self.items) and isinstance(cap, (int, float)) and (not isinstance(cap, bool)) and cap > 0:
                    self.caps[number] = min(cap, self.caps.get(number, cap))
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

    def unmix(self):
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
        if re.search('\\d', value):
            return False
        if re.search('\\b(?:listing|page|shown|other than|unlike|same as)\\b', value, re.I):
            return False
        item = self.items[number - 1]
        if any(({w for w in _stem(other.name) if len(w) >= 4} & _stem(value) for other in self.items if other is not item)):
            return False
        cat_key = self.category_key()
        keys = set()
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
            needs = [m + (f' (the page it points at gives: {self.ref_values[m]}; the other details in this requirement only pick out that page and are not requirements on this item)' if m in self.ref_values else '') for m in live if m not in told]
            head = f'item {number}: {item.name}' if many else f'item: {item.name}'
            if item.shelf:
                head += f' (store label: {item.shelf})'
            head += f', quantity {item.qty}'
            if number in self.caps:
                head += f', price cap {self.caps[number]:g} for one'
            text = head + '; requirements: ' + ('; '.join(needs) if needs else 'none beyond the kind of product')
            if told:
                text += "; described in the shopper's words, which no listing attribute states (decides nothing): " + '; '.join(told)
            if item.free:
                text += '; shopper does not mind: ' + ', '.join(item.free)
            gone = [m for m in dict.fromkeys(item.dropped) if m and m not in live]
            if gone:
                text += '; DROPPED by the shopper after the request (no longer required, even where the request still says it): ' + '; '.join(gone)
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

    def status(self):
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
        held = self.cart_ids() if self.cart_touched else []
        cond = '\\b(?:unless|only if|except when|provided|as long as)\\b'
        terms = [str(r) for r in _items(self.brief.get('rules')) if re.search(cond, str(r), re.I)] + [need for _, need in self.needs_now() if re.search(cond, need, re.I)]
        if held and terms:
            seen = []
            for term in list(dict.fromkeys(terms))[:3]:
                clause = re.split(cond, term, maxsplit=1, flags=re.I)[-1].lower()
                words = {w for w in re.findall('[a-z]{4,}', clause)}
                facts = [f'{i}: {k}={str(v)[:40]}' for i in held for k, v in (_fact_block(self.full.get(i) or {}) or {}).items() if words & set(re.findall('[a-z]{4,}', str(k).lower()))]
                seen.append(term + (' -> cart shows ' + '; '.join(facts[:3]) if facts else ' -> no carted line states it (count the condition as not met)'))
            text += '\n[conditions read from the cart lines now held] ' + ' | '.join(seen)
        if self.time_left() < EXPLORE_RESERVE + 45:
            text += f'\n[time] about {max(0, int(self.time_left()))} s of this episode are left: settle on your picks and check out within the next few calls.'
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

    def absorb(self, observation):

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
        found = set()

        def walk(schema, depth):
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
        counts = {}
        present = {}
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
        groups = {}
        for ident in self.seen_order:
            record = self.full.get(ident)
            if not record:
                continue
            block = _fact_block(record) or {}
            label = str(record.get(cat_key) or 'listings') if cat_key else 'listings'
            group = groups.setdefault(label, {'n': 0, 'keys': {}})
            group['n'] += 1
            done = set()
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
            kept = []
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

    def not_stated(self, label, name, values):
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

    def item_groups(self):
        cat_key = self.category_key()
        groups = {}
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
        if value is None:
            value = next((v for k, v in (self.rows.get(ident) or {}).items() if re.search('price|cost', str(k), re.I) and isinstance(v, (int, float)) and (not isinstance(v, bool))), None)
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and (not isinstance(v, bool))), None)
        if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', value):
            value = float(value)
        return float(value) if isinstance(value, (int, float)) and (not isinstance(value, bool)) else None

    def buyable(self, ident):
        if self.sold_out(ident):
            return False
        value = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        return not (value is False or (isinstance(value, (int, float)) and (not isinstance(value, bool)) and (value <= 0)))

    def records_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        if isinstance(observation, dict):
            ident = self.ident(observation)
            if ident:
                return [(ident, observation)]
            found = []
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

    def shown_already(self, name, ident):
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
            shape['next_arg'] = self.next_arg_of(name, observation[shape['next']])
        self.shapes[name] = shape
        self.price_key = self.price_key or shape['price']
        self.stock_key = self.stock_key or shape['stock']
        self.trace.append({'kind': 'shape', 'tool': name, 'shape': shape})
        return shape

    def paging(self, name, observation):
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
        if self.max_steps < 40 and len(self.items) >= 2:
            return ''
        pick = self.brief.get('pick')
        if pick not in ('cheapest', 'most', 'least'):
            return self.next_page_big(name, args, observation)
        plan = self.paging(name, observation)
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
        if pages > 2 or self.pages_read + pages > AUTO_PAGES or self.left() - self.tail() - pages < (6 if pages == 1 else 12):
            return f"\n[Kyo] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with {param}={_dump(value)}."
        texts = []
        for _ in range(pages):
            faults = []
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
        if value is None and texts and pick == 'cheapest':
            texts += self.end_reads([observation] + self.paged_obs)
        rest = f"\n[Kyo] {count} listings matched and Kyo has read {shown * (len(texts) + 1)} rows of them; narrow the search with the requirement's value or page on to see the rest." if value is not None and texts else ''
        return '\n[Kyo read the next pages of this search itself, one step a page]\n' + '\n'.join(texts) + rest if texts else ''

    def cap_ignored(self, name, args, observation):
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
        self.dead_caps.add((name, caps[0]))
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

    def wider_hold(self):
        if self.wider_held or self.brief.get('pick') != 'cheapest':
            return ''
        self.wider_held = True
        if self.cheaper_sweep() or self.left() < 14 + 2 * len(self.items):
            return ''
        lines = [f'{i} at {self.price_of(i):g}' for i in self.cart_ids() if self.price_of(i) is not None]
        return f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest. Searches match a listing's title, so a requirement's value (often only in the attributes) finds few of the fitting listings, and rows come in no price order. Before checking out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line" + (f" ({'; '.join(lines)})" if lines else '') + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'

    def other_values(self, number: int) -> list[str]:
        table = self.key_table()
        keys = self.lane_keys(number, table)[:3]
        said = ' '.join(self.value_words(number)).lower() + ' ' + self.items[number - 1].name.lower()
        lanes = []
        for key in keys:
            values = next((v for label, name, _, v in table if name == key and (len(self.items) == 1 or self.owner_of(label) == number)), [])
            lanes.append([v for v in values if v and len(v) <= 20 and v not in said and re.search('[a-z0-9]', v)][:2])
        return [lane[k] for k in range(2) for lane in lanes if k < len(lane)]

    def cheaper_sweep(self) -> bool:
        cat_key = self.category_key()
        plan = self.search_template()
        mark = self.cart_mark() + f'#{self.version}'
        room = self.left() - self.tail() - len(self.items) - 3
        if mark in self.swept or room < 1 or self.time_left() < EXPLORE_RESERVE + 25 or (not cat_key):
            return False
        self.swept.add(mark)
        limit = self.step + min(SWEEP_STEPS, room)
        ran = False
        cheaper = []
        for line in self.cart_ids():
            price = self.price_of(line)
            label = str((self.rows.get(line) or {}).get(cat_key) or '')
            number = next((n for n, item in enumerate(self.items, 1) if label and self.owns(item, label)), 0)
            if price is None or not number or self.step >= limit:
                continue
            found = self.descent(number, price, limit)
            if found is not None:
                ran = True
            elif plan:
                base = self.value_words(number)[:2]
                for other in self.other_values(number)[:3]:
                    if self.step >= limit:
                        break
                    found = (found or []) + self.search_words(plan[0], plan[1], plan[2], ' '.join(base + [other, self.items[number - 1].name]))
                    ran = True
                flipped = ' '.join((_flip_units(b) for b in base))
                if base and flipped != ' '.join(base) and self.step < limit:
                    found = (found or []) + self.search_words(plan[0], plan[1], plan[2], ' '.join([flipped, self.items[number - 1].name]))
                    ran = True
            cheaper += [i for i in self.kind_rows(number, found or [], price) if i != line and i not in self.full]
        self.trace.append({'kind': 'sweep', 'ran': ran, 'cheaper': cheaper[:6]})
        return ran

    def fill_facts(self, ids, keep):
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return []
        texts = []
        for ident in [i for i in ids if i in self.rows and i not in self.full and (not self.sold_out(i))]:
            if self.left() <= keep or self.finished or self.facts_from.get(tool) is False or self.time_left() < CALL_RESERVE + 10:
                break
            faults = []
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
        picks = []
        for line in cart:
            price = self.price_of(line)
            kind = str((self.rows.get(line) or {}).get(cat_key) or '').strip().lower()
            if price is None or not kind:
                continue
            pool = [i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i) and (str((self.rows.get(i) or {}).get(cat_key) or '').strip().lower() == kind) and ((self.price_of(i) if self.price_of(i) is not None else float('inf')) < price)]
            pool.sort(key=lambda i: self.price_of(i) or 0.0)
            picks += [i for i in pool[:CHEAPER_OPENS if len(cart) == 1 else 2] if i not in picks]
        return self.fill_facts(picks[:CHEAPER_OPENS], self.tail() + 2)

    def observe(self, name: str, observation: Any) -> str:
        self.absorb(observation)
        lines = self.loose_lines(observation)
        if not lines:
            return ''
        meaning = self.take_note(lines)
        if meaning:
            self.queued.append(f'(seen on {name}) {meaning}')
        return meaning

    def widen(self):
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
        picks = []
        for ids in ranked:
            picks += [i for i in ids if i not in self.full][:share]
        picks = picks[:cap]
        if len(picks) < 2:
            return
        faults = []
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
        lanes = []
        heads = []
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

    def message_args(self, text):
        name = self.roles.get('ask')
        schema = self.schemas.get(name or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        target = next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if texts else None)
        if not target:
            return None
        args = {target: text}
        for key in schema.get('required') or []:
            if key not in args:
                spec = props.get(key) or {}
                if isinstance(spec.get('enum'), list) and spec['enum']:
                    args[key] = spec['enum'][0]
                else:
                    return None
        faults = []
        args = _shape(args, schema, 'args', faults)
        return None if faults else args

    def interview(self) -> str:
        self.asked = True
        if self.msgs_free() <= 0:
            return ''
        if self.time_left() > EXPLORE_RESERVE + 30:
            self.widen()
        lanes = self.plan_lanes(self.msgs_free())
        found = {n: 0 for n in lanes}
        spent = {n: 0 for n in lanes}
        used = {n: set() for n in lanes}
        follow = []
        sent = []
        told = []
        while self.msgs_free() > 0 and (not self.finished) and (self.left() > 6) and (self.time_left() > EXPLORE_RESERVE + 20):
            if follow:
                number, labels = follow.pop(0)
            else:
                number, labels = self.next_ask(lanes, found, spent, used)
                if not labels:
                    break
            used[number].update((_words(label) for label in labels))
            batch = self.send([(number, labels[0], self.question(number, labels[0]))])
            if not batch:
                break
            sent += batch
            spent[number] += 1
            had = [len(item.told) for item in self.items]
            heard = self.hear(batch, (LIGHT, HEAVY))
            if self.judges()[:1] == [HEAVY] and any((self.doubtful(entry) for entry in _items(heard.get('needs')))):
                heard = self.hear(batch, (HEAVY,)) or heard
            self.take_heard(heard, told)
            fresh = [n for n, item in enumerate(self.items, 1) if len(item.told) > had[n - 1]]
            for n in fresh:
                found[n] = found.get(n, 0) + 1
            if fresh or 1 not in [_count(x) for x in _items(heard.get('crowded'))]:
                continue
            head = labels[0].split()[-1]
            if len(labels[0].split()) >= 2 and _words(head) not in used[number]:
                follow.insert(0, (number, [head]))
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

    def value_words(self, number):
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
        seen = []
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
                faults = []
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

    def next_ask(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> tuple[int, list[str]]:
        first = {n: i for i, n in enumerate(self.plan_order)}
        for number in sorted(lanes, key=lambda n: (found.get(n, 0) > 0, spent.get(n, 0), first.get(n, len(first) + n))):
            left = [label for label in lanes[number] if _words(label) not in used[number]]
            if left:
                return (number, left[:1])
        return (0, [])

    def lane_keys(self, number, table):
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

    def records_of_item(self, number):
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
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 3500, deep=True) or {}
        lanes = {n: [] for n in range(1, len(self.items) + 1)}
        hinted = {n: set() for n in lanes}
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
            own = {}
            for label, key, carried, _ in table:
                if self.owner_of(label) == number:
                    own[key] = max(own.get(key, 0.0), carried / read)

            def spelled(label: str) -> str:
                plain = lambda text: re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', _words(text)))
                return next((k for k in own if plain(k) == plain(label)), label)

            def share_of(label):
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
            if count <= TIGHT_MESSAGES and read >= 3:
                sure = [x for x in lanes[number] if _words(x) in hinted[number] or share_of(x) >= 0.5]
                lanes[number] = sure + [x for x in lanes[number] if x not in sure]
        self.trace.append({'kind': 'plan', 'ask': _clip(plan.get('ask'), 120), 'lanes': {n: v[:8] for n, v in lanes.items()}})
        return lanes

    def doubtful(self, entry: Any) -> bool:
        if not isinstance(entry, dict):
            return False
        prop, _, value = str(entry.get('need') or '').partition(':')
        words = set(re.findall('[a-z0-9]+', value.lower()))
        if entry.get('points') is True:
            return True
        if len(words) < 2:
            return False
        seen = set()
        for record in self.full.values():
            for key, shown in (_fact_block(record) or {}).items():
                if _names(_stem(prop), _stem(key)) or _names(_stem(key), _stem(prop)):
                    seen |= set(re.findall('[a-z0-9]+', str(shown).lower()))
        return bool(words - seen)

    def question(self, number: int, label: str) -> str:
        return f'What {label} would you like for the {self.items[number - 1].name}?'

    def send(self, questions, floor=6):
        tool = self.roles['ask']
        sent = []
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
            self.budget = self.budget_unit(heard['budget'])

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(HEAVY, LIGHT)) -> dict[str, Any]:
        prompt = HEAR + '\n\nITEMS:\n' + '\n'.join((f'item {n}: {it.name}' for n, it in enumerate(self.items, 1)))
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f'{i}. (about item {n}) Q: {q}\nA: {a}' for i, (n, q, a) in enumerate(sent, 1)))
        return self.ask_json(prompt, list(models), 2500) or {}

    def gate(self, stage):
        steps = [p for p in self.process if p['when'] == stage]
        if not steps or self.finished or (not self.gate_due(stage)):
            return (True, '')
        ending = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        tools = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_dump(t['function'].get('parameters') or {})}" for t in self.tools if t['function']['name'] not in ending]
        said = []
        nudged = False
        for _ in range(GATE_ROUNDS):
            prompt = GATE.replace('STAGE', 'place the order' if stage == 'order' else 'end without a purchase') + '\n\nREQUEST:\n' + self.request + '\n\nSTEPS DUE NOW:\n' + '\n'.join((f"- {p['step']}" for p in steps)) + '\n\nLEDGER:\n' + self.ledger() + '\n\nCART NOW: ' + self.cart_text() + '\n\nRECORD (messages and step calls so far):\n' + ('\n'.join((f'{i}. {line}' for i, line in enumerate(self.done_log, 1))) or '(nothing yet)') + ('\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.notes[-4:]) if self.notes else '') + f"\n\nMESSAGES LEFT: {max(0, self.msg_limit - self.msg_used)}   STEPS LEFT: {self.left()} (the {('order' if stage == 'order' else 'ending')} itself takes one)" + '\n\nTOOLS:\n' + '\n'.join(tools)
            got = self.ask_json(prompt, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 3500)
            if got is None:
                break
            told = []
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
            faults = []
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
        found = []

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

    def loose_lines(self, observation):
        lines = []
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
        return [line for line in lines if line not in self.heard and (not _record_line(line))]

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
        if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0):
            budget = self.budget_unit(budget)
        if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0) and (budget != self.budget):
            self.budget = budget
            changed.append(f'budget now {budget}')
            self.fresh_changes.append(('budget', f'{budget:g}'))
        added = set()
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
            self.fresh_changes.append(('drop', target))
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
            self.fresh_changes.append(('price', ''))
        if got.get('price_fell') is True:
            changed.append('a price fell')
            self.fresh_changes.append(('cheaper', ''))
        if gone or self.note_sold:
            self.fresh_changes.append(('gone', ', '.join(gone)))
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

    def evidence(self, observation):
        ids = []

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
                hits = [f'{k}={str(v)[:50]}' + (f' [does not state "{need.partition(":")[2].strip()}": other words stand between its words, so this is another value]' if _split_value(need.partition(':')[2], v) else '') + (' [a list of choices on offer, not a stated value]' if _choice_key(k) else '') for k, v in block.items() if _names(_stem(prop), _stem(k))]
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

    def held_units(self, ident):
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            stack = [self.cart_view]
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

    def others(self, skip, limit=30):
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
        marks = {}
        notes = {}
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

    def item_ids(self, number):
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

    def fit_plan(self, short=False):
        if not self.bound:
            return ''
        index = self.fit_index()
        lines = []
        for main in sorted({m for _, m in self.bound}):
            parts = [p for p, m in self.bound if m == main]
            tally = {}
            sure = {}
            for p in parts:
                for i in self.item_ids(p):
                    if self.sold_out(i):
                        continue
                    own = (index['marks'].get(i) or {}).get('own') or ''
                    named = self.fit_set(i, index) | ({own.split('-')[0]} if re.match(PART_SERIES + '-', own) else set())
                    good = i in self.full and self.proves(i, p)
                    for code in named:
                        tally.setdefault(code, {}).setdefault(p, 0)
                        tally[code][p] += 1
                        if good:
                            sure.setdefault(code, {}).setdefault(p, []).append(i)
            ranked = sorted(tally.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1].values()), -len(sure.get(kv[0], {})), kv[0]))[:4]
            devices = {}
            for code, _ in ranked:
                marks = {i: index['marks'].get(i) or {} for i in self.item_ids(main)}
                devices[code] = [i for i, m in marks.items() if any((_code_names(code, c) for c in m.get('codes') or [])) or code in (m.get('series') or [])][:2]
            text = '; '.join((f'{code}: ' + ', '.join((f'{n} {self.items[p - 1].name}' + '' for p, n in sorted(per.items()))) + (f" (device seen: {', '.join(devices[code])})" if devices[code] else ' (device not seen yet)') for code, per in ranked))
            if text:
                lines.append(f"codes the {self.items[main - 1].name} candidates' parts name -> {text}")
        if not lines:
            return ''
        return ' | '.join(lines) if short else "[fit plan: a device fits a part when the part's text names the device's model code, its series or another name it is sold as, or when the device's text names the part's series; codes are not searchable, so list the device's category and read the codes on its rows] " + ' | '.join(lines)

    def line_faults(self) -> list[str]:
        cat_key = self.category_key()
        ids = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        if not ids or not cat_key:
            return []
        label = {i: str((self.rows.get(i) or {}).get(cat_key) or '') for i in ids}
        known = {n for n, item in enumerate(self.items, 1) if any((self.owns(item, str(r.get(cat_key) or '')) for r in self.rows.values()))}
        owner = {i: [n for n, item in enumerate(self.items, 1) if self.owns(item, label[i])] for i in ids}
        faults = []
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
        for i in ids:
            number = self.owner_of(label[i])
            if number in self.caps and (self.price_of(i) or 0) > self.caps[number]:
                faults.append(f'{i} costs {self.price_of(i):g}, over the cap of {self.caps[number]:g} for one {self.items[number - 1].name}')
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

    def fix_counts(self, wrong):
        name = self.roles.get('cart') or ''
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        target = next((k for t, k in self.id_args if t == name and k in texts), None) or next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        count = next((k for k, v in props.items() if isinstance(v, dict) and 'integer' in _kinds(v)), None)
        mode = next((k for k, v in props.items() if isinstance(v, dict) and isinstance(v.get('enum'), list) and (len(v['enum']) >= 2)), None)
        if not name or not target or (not count):
            return []
        calls = []
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
            faults = []
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
        weak = [] if lead else [w for w in self.weak_figures() if w.split(' ', 1)[0] in ids]
        if weak:
            prompt += '\n\nNOT STATED UNDER THE PROPERTY\'S OWN KEY (a measured limit counts only under a key sharing the property\'s name; such a line fails that requirement unless its record states it there in other words):\n' + '\n'.join(weak)
        if self.brief.get('pick') in ('cheapest', 'most', 'least'):
            prompt += '\n\nOTHER LISTINGS SEEN (newest first):\n' + self.others(ids, 15)
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

    def trail(self):
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        tool = self.roles.get('offer') or self.roles.get('open')
        target = self.single_target(tool)
        seen = []
        lost = []
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
            prompt += "\n\nTHE SHOPPER'S REQUEST (each requirement above shortens it; where its words say how the wanted value relates to the value along the chain, those words decide the value, not the requirement's wording):\n" + self.request[:3000]
            prompt += '\n\nRECORDS (newest first):\n' + '\n'.join(records)
            got = self.ask_json(prompt, [LIGHT] + ([HEAVY] if self.cost + HEAVY_ROOM <= SPEND_HARD else []), 4500) or {}
            self.trace.append({'kind': 'trail', 'pending': pending, 'got': _clip(got, 150)})
            reach = []
            lost = []
            gaps = []
            for chain in _items(got.get('chains')):
                if not isinstance(chain, dict):
                    continue
                need = next((p for p in pending if p == str(chain.get('need') or '').strip()), None) or (pending[0] if len(pending) == 1 else None)
                if not need:
                    continue
                links = [l for l in _items(chain.get('links')) if isinstance(l, dict)]
                ids = [str(l.get('id') or '').strip() for l in links]
                self.chain_ids += [i for i in ids if i in self.rows and i not in self.chain_ids]
                stranger = self.not_their_shop(need, ids[0]) if ids else ''
                if stranger:
                    lost.append(f'{need} -> {ids[0]} is not a listing of {stranger}')
                    gaps.append((stranger, stranger))
                    self.trace.append({'kind': 'trail_shop', 'need': need, 'id': ids[0], 'shop': stranger})
                    continue
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
                faults = []
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

    def find_link(self, about, words):
        known = []
        named = {w for ident in self.shop_rows(about, list(self.rows)) + self.opened_shop(about) for k, v in (self.rows.get(ident) or {}).items() if isinstance(v, str) and len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I) for w in re.findall('[a-z0-9]{3,}', v.lower())}
        values = {w for w in re.findall('[a-z0-9]+', (words or about).lower()) if len(w) >= 3 or w.isdigit()} - named - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for', 'of'}
        for ident in self.opened_shop(about):
            stack, leaves = ([self.full.get(ident) or self.rows.get(ident) or {}], [])
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    stack += [v for k, v in node.items() if not re.search('remark|note', str(k), re.I)]
                elif isinstance(node, list):
                    stack += node
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
            faults = []
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
                    if not known and about and self.left() > self.tail() + 8:
                        known = self.link_rungs(tool, base, field, about, sorted(values, key=len, reverse=True))
        kind = {w for w in re.findall('[a-z0-9]{3,}', (words or about).lower())} - named - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for'}

        def of_kind(ids: list[str]) -> list[str]:
            return [i for i in ids if not kind or kind & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower()))]
        return (of_kind(known) or of_kind(self.shop_rows(about, list(reversed(self.seen_order)))))[:2]

    def offer_tool(self) -> str | None:
        return self.roles.get('offer') or self.roles.get('open')

    def reread_offers(self):
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return ''
        since = max(self.step - 2, self.refused_at)
        stale = [i for i in self.cart_ids() if self.offer_seen.get(i, -99) < since and i not in self.gone]
        if stale and self.left() < len(stale) + 3:
            unseen = [i for i in stale if i not in self.offer_seen]
            stale = unseen if self.left() >= len(unseen) + self.tail() else []
        if not stale:
            return ''
        lines = []
        for ident in stale:
            faults = []
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

    def find_sold(self):
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
        fits = []
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
            if self.widen_search(got, missing) or opened:
                got = self.audit_call()
                self.take_review(got)
                named = self.audit_fits(got) or named
        if not named:
            return ''
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return 'A careful review thinks these listings may fit: ' + '; '.join((f"item {k}: {', '.join(v)}" for k, v in named.items())) + (f" ({got.get('reason')})" if got and got.get('reason') else '') + '. Open them and buy what truly fits; end without a purchase only if each one fails.'

    def audit_call(self):
        cap = self.budget if isinstance(self.budget, (int, float)) else None
        cat_key = self.category_key()
        opened = []
        rows = []
        for strict in (True, False):
            for ident in reversed(self.seen_order):
                label = str((self.rows.get(ident) or {}).get(cat_key) or '') if cat_key else ''
                if self.sold_out(ident) or (strict and ((cap is not None and (self.price_of(ident) or 0) > cap) or (label and (not any((self.owns(item, label) for item in self.items)))))):
                    continue
                record = self.full.get(ident) if ident in self.full else None
                if record and len(opened) < 20:
                    opened.append('OPENED ' + _judge_view(record, 1000, 200))
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
        lookups = {need for _, need in self.refs}
        changed = []
        aside = set()
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
        cat_key = self.category_key()
        item = self.items[number - 1]
        labels = [str(r.get(cat_key) or '') for r in self.rows.values()] if cat_key else []
        labels = [label for label in labels if label and self.owns(item, label)]
        return max(set(labels), key=labels.count) if labels else item.shelf

    def list_category(self, number, cap, more=0):
        spec = self.browse
        label = self.shelf_of(number)
        if not spec or not label or self.browse_misses >= 2 or (cap is not None and (not spec['price'])):
            return None
        args = {**spec['fixed'], spec['category']: label}
        if cap is not None:
            args[spec['price']] = cap
        faults = []
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
        faults = []
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
        cat_key = self.category_key()
        item = self.items[number - 1]
        wanted = {w for w in re.findall('[a-z0-9]{3,}', ' '.join(self.value_words(number)).lower())}
        rows = [i for i in dict.fromkeys(found) if self.buyable(i) and (not cat_key or self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or '')))]
        if below is not None:
            rows = [i for i in rows if (self.price_of(i) if self.price_of(i) is not None else below) < below]
        return sorted(rows, key=lambda i: (i in self.full, -len(wanted & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower()))), self.price_of(i) or 0))

    def widen_search(self, got, missing):
        keep = self.tail() + len(self.items) + 2
        room = min(RECOVERY_STEPS, self.left() - keep)
        if room < 3 or self.time_left() < EXPLORE_RESERVE + 25:
            return False
        before = len(self.full)
        last = self.step + room
        searching = self.step + max(1, room // 2)
        plan = self.search_template()
        queries = []
        for entry in _items((got or {}).get('search')):
            if isinstance(entry, dict) and str(entry.get('query') or '').strip():
                queries.append((min(max(1, _count(entry.get('item')) or 1), len(self.items)), str(entry['query']).strip()))
        queries = [q for q in queries if q[0] in missing] or [(n, ' '.join(self.value_words(n)[:2] + [self.items[n - 1].name])) for n in missing]
        found = {n: [] for n in missing}
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
            faults = []
            args = _shape(args, self.schemas.get(name), 'args', faults)
            if faults:
                return False
            reply = self.act(name, args)
            if reply is None or self.finished:
                break
            self.observe(name, reply.get('observation'))
        self.trace.append({'kind': 'audit_open', 'rows': rows, 'opened': len(self.full) - before})
        return len(self.full) > before

    def cart_calls(self, lines, drop):
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
        calls = []
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
        plan = {}
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
            faults = []
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
        shown, held = self.offer_proof()
        if held or self.finished or self.left() < 1:
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

    def is_buy(self, name, args):
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
        keep = None
        ends = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        if left <= self.tail():
            keep = set(ends)
        elif left <= self.tail() + 3:
            keep = set(ends) | {self.roles.get('cart'), self.roles.get('cart_view')}
        elif left <= len(self.items) + 5 and self.buys == 0 and self.cart_complete():
            keep = set(ends) | {self.roles.get('cart'), self.roles.get('cart_view'), self.roles.get('offer')}
        tools = [t for t in self.tools if t['function']['name'] != ask]
        if keep is not None and ends:
            narrowed = [t for t in tools if t['function']['name'] in keep]
            tools = narrowed or tools
        return tools

    def quiet(self, name, args):
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

    def cart_sets(self, name):
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
        lines = []
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
        text = _dump(_row_cut(observation, SHOW_CHARS))
        if len(text) > OBS_LIMIT:
            text = text[:OBS_LIMIT] + '...(cut)'
        return text

    def compare_target(self):
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
        picks = []
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

    def run_calls(self, calls):
        out = []
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
            faults = []
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
            if name and name == self.roles.get('buy') == self.roles.get('refuse') and isinstance(args, dict):
                mode = dict(self.brief.get('buy_args') or {}) if self.cart_ids() else dict(self.brief.get('refuse_args') or {})
                lacking = {k: v for k, v in mode.items() if k not in args}
                if lacking:
                    args = {**args, **lacking}
                    self.trace.append({'kind': 'mode_filled', 'args': lacking})
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
            loose = name + _dump({k: v for k, v in args.items() if (name, k) not in self.dead_caps})
            if not cached and loose != mark and self.read_cache.get(loose) and self.read_cache[loose][0] == self.version:
                cached = self.read_cache[loose]
                self.trace.append({'kind': 'cap_cached', 'tool': name})
                out.append((cid, f'Not run (no step used): this store ignores that price cap, so the same search without it (step {cached[1]}) already gave these rows:\n' + cached[2][:3000]))
                continue
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
            if ending and self.left() > 2 and self.time_left() > CALL_RESERVE + 25:
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
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()
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
                        basket = self.planned_basket(first) if len(self.items) > 1 else None
                        judged = sorted(basket) if basket else [first]
                        self.check_cart(judged, SPEND_HARD)
                        if self.verdicts.get(self.check_key(judged)) and first in self.leads:
                            self.confirmed = first
                            self.trace.append({'kind': 'refuse_blocked', 'confirmed': first, 'basket': judged})
                            if basket:
                                out.append((cid, f"Not run yet (no step used). A separate check of the whole basket {', '.join(judged)} finds that it meets every requirement with one line per item. Make the cart exactly these lines at the requested quantities and check out; end without a purchase only if a page plainly breaks a requirement."))
                                continue
                            out.append((cid, f"Not run yet (no step used). An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."))
                            continue
            if self.is_refuse(name, args) and self.left() <= 5 and (not self.late_held):
                done = self.late_finish()
                if done or self.finished:
                    self.late_held = True
                    out.append((cid, (done or 'Not run: the episode already ended.') + (' [episode finished]' if self.finished else '')))
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
            if self.is_refuse(name, args) and (not self.leads) and (not self.pushed) and (self.left() > self.max_steps // 3) and (self.left() > self.tail() + len(self.items) + 2) and (self.time_left() > EXPLORE_RESERVE + 30):
                self.pushed = True
                out.append((cid, f"Not run yet (no step used). {self.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if self.is_refuse(name, args):
                done = self.complete_cart()
                if done:
                    out.append((cid, done + (' [episode finished]' if self.finished else '')))
                    continue
                args = self.quiet(name, args)
            if ending and self.left() >= 2 and self.time_left() > CALL_RESERVE + 15:
                go, said = self.gate('order' if self.is_buy(name, args) else 'end')
                if not go:
                    out.append((cid, 'Not run yet. ' + said + '\nCart now: ' + self.cart_text()))
                    continue
                if said:
                    trimmed = '\n[Kyo] ' + said + trimmed
            if self.is_buy(name, args) and (not self.said_clash) and self.left() > self.tail() + 2:
                clash = self.clashes()
                if clash:
                    self.said_clash = True
                    self.trace.append({'kind': 'clash_hold', 'lines': clash[:4]})
                    out.append((cid, 'Not run yet (no step used). A cart line states another value than a requirement under its own key: ' + '; '.join(clash[:3]) + '. Replace that line with a listing that states the required value; if you are sure the line is right, check out again.\nCart now: ' + self.cart_text()))
                    continue
            if self.is_buy(name, args) and (not self.dear_done) and self.brief.get('pick') == 'cheapest' and self.left() > self.tail() + 6:
                self.dear_done = True
                trimmed += self.dear_lines()
            if self.is_buy(name, args) and self.said_gap and self.gap_alt and (not self.swapped) and self.left() > self.tail() + 3:
                self.swapped = True
                held = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
                pairs = {line: alt for line, alt in self.gap_alt.items() if line in held and alt not in held}
                if pairs:
                    lines = {pairs.get(i, i): self.held_units(i) or 1 for i in held}
                    if self.cart_to(lines, 2):
                        self.trace.append({'kind': 'agreed_swap', 'pairs': pairs})
                        trimmed += '\n[Kyo swapped ' + ', '.join((f'{a} -> {b}' for a, b in pairs.items())) + ': the cheaper line states every requirement too]'
            if self.is_buy(name, args) and (not self.weak_said) and self.left() > self.tail() + 3:
                weak = self.weak_figures()
                if weak:
                    self.weak_said = True
                    self.trace.append({'kind': 'weak_hold', 'lines': weak[:4]})
                    out.append((cid, 'Not run yet (no step used). ' + ' '.join(weak[:3]) + ' Read every figure exactly as written. Swap such a line if a listing named here states the figure; otherwise check out again.\nCart now: ' + self.cart_text()))
                    continue
            if self.is_buy(name, args) and (not self.code_held) and self.left() > self.tail() + 3:
                missed = self.code_misses()
                if missed:
                    self.code_held = True
                    self.trace.append({'kind': 'code_hold', 'lines': missed[:4]})
                    out.append((cid, 'Not run yet (no step used). ' + ' '.join(missed[:3]) + ' Swap the line for a listing whose text names the code, then check out.\nCart now: ' + self.cart_text()))
                    continue
            if self.is_buy(name, args) and (not self.said_gap) and self.left() > self.tail() + 3:
                gaps = self.proof_gaps()
                if gaps:
                    self.said_gap = True
                    self.trace.append({'kind': 'proof_hold', 'gaps': gaps[:4]})
                    out.append((cid, 'Not run yet (no step used). ' + ' '.join(gaps[:3]) + ' Swap to it if you agree; otherwise check out again.\nCart now: ' + self.cart_text()))
                    continue
            if self.is_buy(name, args) and self.left() > self.tail() + 2:
                trimmed += self.referred_pages()
                trimmed += self.referred_names()
            if self.is_buy(name, args) and (not self.end_tried) and self.left() <= 4 and (not self.cart_complete()):
                self.end_tried = True
                self.filled = self.filled and bool(self.buys)
                before = set(self.cart_ids())
                self.clock_fill(1)
                if self.filled and set(self.cart_ids()) != before:
                    self.trace.append({'kind': 'end_fill', 'cart': self.cart_ids()})
                    trimmed += '\n[Kyo] The cart lacked a line for an item with the last steps running out, so Kyo carted the opened listing that comes closest to its requirements before this checkout.'
            if self.is_buy(name, args):
                shown, held = self.offer_proof()
                if held:
                    out.append((cid, 'Not run yet. ' + shown + '\nCart now: ' + self.cart_text()))
                    continue
                trimmed += shown
            if self.is_refuse(name, args) and not self.last_look:
                self.last_look = True
                unread = self.unread_leads()
                if unread:
                    out.append((cid, 'Not run yet. Before ending without a purchase, Kyo opened rows of the requested items that were seen in searches but never read:\n' + '\n'.join(unread) + '\nIf one of them meets every requirement in force, put it in the cart and check out; if none does, send the ending again.\nCart now: ' + self.cart_text()))
                    continue
            if self.is_refuse(name, args) and (not self.filled) and len(self.items) > 1 and self.cart_ids() and EPISODE_SECONDS - self.time_left() >= LATE_ORDER:
                self.clock_fill()
                if self.filled and self.cart_complete():
                    self.trace.append({'kind': 'late_order'})
                    out.append((cid, 'Not run yet. The clock is nearly out and the cart already held lines, so Kyo filled the missing items with the opened listings that come closest to their requirements. Ending without a purchase earns nothing here: check out now.\nCart now: ' + self.cart_text()))
                    continue
            if self.is_refuse(name, args) and (not self.chain_retried) and self.left() > self.tail() + 3:
                live = {need for _, need in self.needs_now()}
                if any((need in live and need not in self.ref_values for _, need in self.refs)):
                    self.chain_retried = True
                    before = dict(self.ref_values)
                    chain, hold = self.trail()
                    if self.ref_values != before or hold:
                        self.trace.append({'kind': 'chain_retry', 'resolved': [n for n in self.ref_values if n not in before]})
                        out.append((cid, 'Not run yet (no step used). Before ending, Kyo followed the pointed pages again. ' + chain + '\nJudge the listings again with what the chain gives, and end without a purchase only if none fits.\nCart now: ' + self.cart_text()))
                        continue
            if self.is_refuse(name, args) and (not self.decline_proven) and self.left() > self.tail() + len(self.items) + 1:
                self.decline_proven = True
                proven = self.proven_set()
                if proven:
                    self.trace.append({'kind': 'decline_proven', 'lines': proven})
                    out.append((cid, 'Not run yet (no step used). These opened listings state every requirement in force under their own keys, figures read as written: ' + ', '.join((f'{i} x{q}' for i, q in proven.items())) + '. A purchase of them meets the request, so put them in the cart and check out instead of ending without a purchase.\nCart now: ' + self.cart_text()))
                    continue
            if self.is_refuse(name, args) and (not self.near_tried) and self.left() > self.tail() + 1:
                near = self.nearest_order(True)
                if near:
                    out.append((cid, 'Not run yet. ' + near + '\nCart now: ' + self.cart_text()))
                    continue
            args, moved = self.start_from(name, args)
            trimmed += moved
            args, moved = self.units_not_rows(name, args)
            trimmed += moved
            args, moved = self.keep_lines(name, args)
            trimmed += moved
            args, moved = self.without_dead_codes(name, args)
            if moved == 'EMPTY':
                out.append((cid, 'Not run (no step used): the fit code is not searchable in this store, and nothing was left of the query without it. Search the kind of part (and a required value) instead; Kyo marks the rows whose text names the code.'))
                continue
            trimmed += moved
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, 'The call failed (network or server). You may try again.'))
                continue
            observation = reply.get('observation')
            if name == self.roles.get('search') and (not self.records_in(observation)) and self.keys_out_n < 4 and self.left() > self.tail() + 4:
                bare_args = self.keys_out(name, args)
                if bare_args:
                    again = self.act(name, bare_args)
                    if again is not None:
                        self.keys_out_n += 1
                        args, reply, observation = (bare_args, again, again.get('observation'))
                        trimmed += f"\n[Kyo] The store does not index property names, so Kyo ran the search again without them: {_dump(bare_args)[:120]}"
            if name == self.roles.get('cart'):
                trimmed += self.note_wipe(name, args, observation)
            skipped = self.past_the_end(name, args, observation)
            if skipped and self.left() > self.tail() + 1 and not self.finished:
                start = self.skip_args[(name, skipped)]
                again = self.act(name, {**args, skipped: start})
                if again is not None:
                    args, reply = ({**args, skipped: start}, again)
                    observation = reply.get('observation')
                    mark = name + _dump(args)
                    trimmed += f'\n[Kyo] {skipped} skips that many results in this store (it is where the list starts, not how many rows to show): the hits before it were skipped, so Kyo ran the same search again from {skipped}={start}. Leave {skipped} out or at {start} for a new search; raise it only to page past rows already shown.'
            self.count_rows(name, args, observation)
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
                self.search_seen = (self.search_seen + [(dict(args), observation)])[-8:]
            if name == self.roles.get('cart_view'):
                self.cart_view = observation
                self.cart_seen_at = self.step
            if self.is_buy(name, args):
                self.buys += 1
                if not reply.get('finished'):
                    self.refused_at = self.step
            ahead = self.open_found(name, observation) if name == self.roles.get('search') and (not ending) else ''
            ahead += self.fit_scan(name, args, observation) if name == self.roles.get('search') and (not ending) else ''
            ahead += self.series_sweep() if name in readers and (not ending) else ''
            text = self.show(observation) + paged + ahead + trimmed
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
            text += self.after_refusal(name, args, reply)
            if name in readers:
                self.read_cache[mark] = (self.version, self.step, base)
                self.read_cache[name + _dump({k: v for k, v in args.items() if (name, k) not in self.dead_caps})] = (self.version, self.step, base)
            if reply.get('finished'):
                text += ' [episode finished]'
            out.append((cid, text))
        return out

    def after_refusal(self, name: str, args: dict[str, Any], reply: dict[str, Any]) -> str:
        events, self.fresh_changes = (self.fresh_changes, [])
        buying, ending = (self.is_buy(name, args), self.is_refuse(name, args))
        if reply.get('finished') or self.finished or not events or not (buying or ending):
            return ''
        waived = [what for kind, what in events if kind == 'drop' and what]
        text = ''
        if ending and (waived or any((kind == 'budget' for kind, _ in events))) and (not self.cart_ids()):
            lines = self.proven_set()
            if lines and self.cart_to(lines, 1):
                self.trace.append({'kind': 'late_set', 'lines': lines})
                return '\n[Kyo] After that change these opened listings state every requirement still in force, so Kyo set the cart to them: ' + ', '.join((f'{i} x{q}' for i, q in lines.items())) + '. Check out now.'
        if ending and waived and self.cart_ids():
            text += '\n[Kyo] The shopper has just let go of ' + '; '.join(waived) + '. The cart still holds ' + ', '.join(self.cart_ids()) + ': judge those lines against the requirements still in force (the ledger), and check out if they meet them instead of ending again.'
        kinds = {kind for kind, _ in events}
        spend = sum(((self.price_of(i) or 0.0) * self.held_units(i) for i in self.cart_ids()))
        limit = self.budget if isinstance(self.budget, (int, float)) and (not isinstance(self.budget, bool)) else None
        over = bool(limit) and spend > limit
        if 'budget' in kinds and ending:
            text += '\n[Kyo] The budget is now ' + ', '.join((what for kind, what in events if kind == 'budget')) + ': listings passed over as too dear may fit now, so look at them again before ending.'
        if 'gone' in kinds and self.brief.get('pick') == 'cheapest' and self.left() > self.tail() + 6:
            text += self.sold_peers(events)
        if waived or kinds & {'gone', 'cheaper', 'budget'} or ('price' in kinds and over):
            text += self.look_again('cheaper' in kinds, buying or self.left() >= 7)
        elif 'price' in kinds and buying and self.cart_ids():
            text += '\n[Kyo] The note only says the carted listing now costs more. That alone is no reason to swap it: if the cart still meets every requirement in force and the budget, send the same checkout again.'
        self.trace.append({'kind': 'after_refusal', 'events': events, 'said': text[:300]})
        return text

    def live_values(self, number: int) -> list[str]:
        out = []
        for owner, need in self.needs_now():
            prop, sep, value = need.partition(':')
            value = value.strip()
            if owner == number and sep and value and need not in [m for _, m in self.refs] and (not re.search('\\d|\\b(?:not|no|other than|unlike|same|within|at most|at least|under|over|less|more|than|listing|page|or)\\b', value, re.I)):
                out.append(value)
        return out

    def query_key(self, name, args):
        return name + _dump({k: v for k, v in args.items() if (name, k) not in self.skip_args})

    def past_the_end(self, name: str, args: dict[str, Any], observation: Any) -> str | None:
        if name != self.roles.get('search') or not isinstance(observation, dict):
            return None
        rows = len(self.records_in(observation))
        totals = [v for k, v in observation.items() if isinstance(v, int) and (not isinstance(v, bool)) and v > 0 and re.search('hit|total|count|match|found|result', k, re.I)]
        props = (self.schemas.get(name) or {}).get('properties') or {}
        moved = [k for k, v in args.items() if k in props and isinstance(v, int) and (not isinstance(v, bool)) and v > 0 and (not re.search('price|cost|budget|amount|spend|max|min', k, re.I))]
        if not totals or len(moved) != 1:
            return None
        key, value = (moved[0], args[moved[0]])
        rest = {k: v for k, v in args.items() if k != key}
        first = not any(({k: v for k, v in old.items() if k != key} == rest for old in self.searched))
        onward = [v for k, v in observation.items() if isinstance(v, int) and (not isinstance(v, bool)) and v not in totals and re.search('next|page|after|cursor|offset|start', k, re.I)]
        if not (rows == 0 and value >= totals[0] or (rows and first and (rows == totals[0] - value or value + rows in onward))):
            return None
        low = (props.get(key) or {}).get('minimum')
        self.skip_args[name, key] = low if isinstance(low, int) and (not isinstance(low, bool)) and low > 0 else 0
        self.trace.append({'kind': 'skip_arg', 'tool': name, 'arg': key, 'sent': value, 'hits': totals[0], 'rows': rows})
        return key

    def start_from(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        notes = []
        for (tool, key), start in self.skip_args.items():
            value = args.get(key)
            if tool != name or not isinstance(value, int) or isinstance(value, bool) or value <= start:
                continue
            room = max(start, self.query_seen.get(self.query_key(name, args), 0))
            if value > room:
                args = {**args, key: room}
                notes.append(f'{key} {value} -> {room}')
        return (args, '\n[Kyo] Set ' + ', '.join(notes) + ': that argument is where the list starts, and no rows that far in have been shown for this search.' if notes else '')

    def count_rows(self, name, args, observation):
        if name != self.roles.get('search') or not self.skip_args:
            return
        offset = max([v for (tool, key), _ in self.skip_args.items() if tool == name for v in [args.get(key)] if isinstance(v, int) and (not isinstance(v, bool))] or [0])
        key = self.query_key(name, args)
        self.query_seen[key] = max(self.query_seen.get(key, 0), offset + len(self.records_in(observation)))

    def sheet_tools(self) -> list[str]:
        busy = {self.roles.get(r) for r in ('cart', 'buy', 'refuse', 'ask', 'search', 'cart_view', 'offer', 'compare')} - {None}
        return [t for t in self.schemas if t not in busy and self.single_target(t) and (not re.search('add|remove|cart|basket|bag|buy|order|checkout|purchase|stash|hold|conclude|wrap|finish|end|message|ask|chat|say|tell|send|note', t, re.I))]

    def sheet_reads(self) -> list[str]:
        if self.finished:
            return []
        lines = [i for i in self.cart_ids() if i not in self.gone]
        pointed = [i for _, need in self.refs for i in self.full if i not in lines and i not in self.gone and any((re.search('(?<![a-z0-9])' + re.escape(str(v).strip().lower()) + '(?![a-z0-9])', need.lower()) for k, v in (self.rows.get(i) or {}).items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|merchant', str(k), re.I)))]
        lines += list(dict.fromkeys(pointed))[:2]
        todo = [(t, i) for t in self.sheet_tools() for i in lines if i not in self.read_by.get(t, set())]
        unseen_offers = [i for i in lines if i not in self.offer_seen] if self.roles.get('offer') and self.roles.get('offer') != self.roles.get('open') else []
        if not todo or self.left() < len(todo) + len(unseen_offers) + 1:
            return []
        done = []
        for tool, ident in todo:
            if self.finished or self.left() <= 1 + len(unseen_offers):
                break
            faults = []
            args = _shape({self.single_target(tool): ident}, self.schemas.get(tool), 'args', faults)
            if faults:
                continue
            reply = self.act(tool, args)
            if reply is None:
                break
            self.observe(tool, reply.get('observation'))
            done.append(f'{tool}({ident})')
        if done:
            self.trace.append({'kind': 'sheet_reads', 'reads': done})
        return done

    def offer_proof(self):
        sheets = self.sheet_reads()
        tool = self.roles.get('offer')
        target = self.single_target(tool) if tool else None
        if not tool or not target or tool == self.roles.get('open'):
            return ('', False)
        unseen = [i for i in self.cart_ids() + [c for c in self.chain_ids if c in self.opened] if i not in self.offer_seen and i not in self.gone]
        pointed = [i for _, need in self.refs for i in dict.fromkeys(self.opened_shop(need) + [x for x in self.full if any((re.search('(?<![a-z0-9])' + re.escape(str(v).strip().lower()) + '(?![a-z0-9])', need.lower()) for k, v in (self.rows.get(x) or {}).items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|merchant', str(k), re.I)))]) if i not in self.offer_seen and i not in self.gone and i not in unseen]
        unseen = list(dict.fromkeys(unseen + pointed[:2]))
        read = []
        for ident in unseen:
            if self.left() <= 1 or self.finished:
                break
            faults = []
            args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
            if faults:
                break
            reply = self.act(tool, args)
            if reply is None:
                break
            seen = reply.get('observation')
            self.offer_seen[ident] = self.step
            self.observe(tool, seen)
            read.append(ident)
        if not read:
            return ('', False)
        self.trace.append({'kind': 'offer_proof', 'ids': read})
        lost = [i for i in read if self.sold_out(i) and i in self.cart_ids()]
        text = f"\n[Kyo opened the offer page of {', '.join(read)} before checking out (one step each)]"
        if lost and self.left() > 2:
            return (text + f" {', '.join(lost)} now shows no stock: replace that line, then check out.", True)
        return (text, False)

    def unread_leads(self) -> list[str]:
        if self.left() < 3 or self.time_left() < CALL_RESERVE + 10:
            return []
        carted = {str((self.rows.get(i) or {}).get(self.category_key() or '') or '').strip().lower() for i in self.cart_ids()}
        picks = []
        for number, item in enumerate(self.items, 1):
            rows = [i for i in self.kind_rows(number, list(self.seen_order)) if i not in self.full and i not in self.cart_ids()]
            if rows and str((self.rows.get(rows[0]) or {}).get(self.category_key() or '') or '').strip().lower() not in carted:
                picks += rows[:2]
        room = 1 if self.left() < 5 else 2
        read = self.fill_facts(picks[:room], 2)
        self.trace.append({'kind': 'last_look', 'picks': picks[:room], 'read': len(read)})
        return read

    def not_their_shop(self, need: str, ident: str) -> str:
        record = self.full.get(ident) or self.rows.get(ident)
        if not record:
            return ''
        names = [w for w in re.findall("\\b[A-Z][a-z]{3,}\\b", need.split(':', 1)[-1])]
        if not names:
            return ''
        shown = _dump(record).lower()
        missing = [w for w in names if w.lower() not in shown]
        return missing[0] if len(missing) == len(names) else ''

    def open_found(self, name, observation):
        if self.finished or self.time_left() < EXPLORE_RESERVE + 10:
            return ''
        found = [i for i, _ in self.records_in(observation)]
        for extra in self.paged_obs:
            found += [i for i, _ in self.records_in(extra)]
        if not found:
            return ''
        cat_key = self.category_key() or ''
        kinds = [str((self.rows.get(i) or {}).get(cat_key) or '') for i in self.cart_ids()]
        missing = sum((1 for item in self.items if not any((self.owns(item, k) for k in kinds))))
        room = self.left() - self.tail() - 3 * max(1, missing)
        if room < 4:
            return ''
        picks = []
        for number in range(1, len(self.items) + 1):
            fresh = [i for i in self.kind_rows(number, found) if i not in self.full and i not in picks and (not self.sold_out(i))]
            picks += fresh[:2]
        allowed = max(0, self.max_steps // 6 - self.auto_reads)
        picks = picks[:min(2, allowed)]
        if not picks:
            return ''
        many, joiner = (self.compare_target(), self.roles.get('compare'))
        texts = []
        if many and joiner and (not self.compare_bare()) and len(picks) > 1:
            faults = []
            args = _shape({many: picks[:self.compare_room()]}, self.schemas.get(joiner), 'args', faults)
            reply = self.act(joiner, args) if not faults else None
            if reply is not None:
                seen = reply.get('observation')
                self.observe(joiner, seen)
                self.note_view(joiner, seen)
                texts.append(self.show(seen)[:3000])
        else:
            texts = self.fill_facts(picks, self.tail() + 3 * max(1, missing))
        self.auto_reads += len(picks) if texts else 0
        self.trace.append({'kind': 'open_found', 'picks': picks, 'read': len(texts)})
        if not texts:
            return ''
        return '\n[Kyo opened the most promising unread rows of this search for you; no need to open them again]\n' + '\n'.join(texts)

    def phrase_plan(self) -> tuple[str, dict[str, Any], str] | None:
        tool = self.roles.get('search')
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and 'enum' not in v]
        named = [k for k in free if re.search('phrase|query|search|lookup|keyword|text|term|words|string|^q$', k, re.I)]
        field = named[0] if named else (free[0] if len(free) == 1 else None)
        if not tool or not field:
            return None
        base = {}
        for k, v in props.items():
            options = v.get('enum') if isinstance(v, dict) else None
            if isinstance(options, list) and options:
                fit = [o for o in options if isinstance(o, str) and re.search('phrase|keyword|text|search|query|word', o, re.I)]
                if fit or k in (schema.get('required') or []):
                    base[k] = (fit or options)[0]
        return (tool, base, field)

    def first_look(self) -> str:
        plan = self.phrase_plan()
        if not plan or self.max_steps < 30 or self.time_left() < EXPLORE_RESERVE + 60:
            return ''
        tool, base, field = plan
        notes = []
        for number, item in list(enumerate(self.items, 1))[:4]:
            if self.left() - self.tail() < 3 * (len(self.items) - number + 1) + 8 or self.finished:
                break
            words = ' '.join(sorted(self.live_values(number), key=len)[:2] + [item.name])
            faults = []
            args = _shape({**base, field: words}, self.schemas.get(tool), 'args', faults)
            if faults:
                break
            reply = self.act(tool, args)
            if reply is None:
                break
            seen = reply.get('observation')
            self.observe(tool, seen)
            if not self.records_in(seen) and re.search('[a-z]{3}s$', item.name.lower()) and self.left() - self.tail() > 3 * (len(self.items) - number + 1) + 9:
                single = ' '.join(sorted(self.live_values(number), key=len)[:2] + [re.sub('(?:es|s)$', '', item.name.strip(), flags=re.I) if not item.name.lower().endswith('ss') else item.name])
                args_one = _shape({**base, field: single}, self.schemas.get(tool), 'args', [])
                if args_one != args:
                    again = self.act(tool, args_one)
                    if again is not None:
                        args, seen = (args_one, again.get('observation'))
                        self.observe(tool, seen)
                        words = single
                        self.trace.append({'kind': 'singular', 'item': number})
            self.searched = (self.searched + [dict(args)])[-8:] if self.records_in(seen) else self.searched
            found = [i for i, _ in self.records_in(seen)]
            cheap = self.brief.get('pick') == 'cheapest' and self.max_steps <= 75
            info = self.paging(tool, seen) if cheap else None
            pages = 0
            while info and 7 <= info[0] <= 48 and info[3] is not None and pages < 2 and (self.left() - self.tail() > 3 * (len(self.items) - number + 1) + 10) and (not self.finished):
                more_args = _shape({**args, info[2]: info[3]}, self.schemas.get(tool), 'args', [])
                again = self.act(tool, more_args)
                if again is None:
                    break
                more = again.get('observation')
                self.observe(tool, more)
                rows = [i for i, _ in self.records_in(more)]
                if not rows:
                    break
                found += [i for i in rows if i not in found]
                info = self.paging(tool, more)
                pages += 1
            picks = [i for i in self.kind_rows(number, found) if i not in self.full and (not self.sold_out(i))][:2]
            if cheap and pages:
                ranked = [i for i in self.kind_rows(number, found) if i not in self.full and (not self.sold_out(i))][:8]
                picks = sorted(ranked, key=lambda i: self.price_of(i) if self.price_of(i) is not None else float('inf'))[:2]
                self.trace.append({'kind': 'cheap_pages', 'item': number, 'pages': pages, 'rows': len(found)})
            read = self.fill_facts(picks, self.tail() + 8)
            notes.append(f'item {number} ({item.name}), searched {words!r}: ' + (', '.join(found[:8]) or 'no rows') + ('\n' + '\n'.join(read) if read else ''))
        self.trace.append({'kind': 'first_look', 'items': len(notes)})
        if not notes:
            return ''
        return "KYO'S FIRST LOOK (value-first searches and the top rows read, already done):\n" + '\n'.join(notes)[:9000]

    def cart_complete(self):
        cart = [i for i in self.cart_ids() if not self.sold_out(i)]
        cat_key = self.category_key()
        if not cat_key:
            return len(cart) >= len(self.items)
        kinds = [str((self.rows.get(i) or {}).get(cat_key) or '') for i in cart]
        return all((any((self.owns(item, kind) for kind in kinds)) for item in self.items))

    def clock_fill(self, room: int=2) -> None:
        cart, buy = (self.roles.get('cart'), self.roles.get('buy'))
        cat_key = self.category_key()
        if self.filled or self.finished or self.buys or not cart or (not buy) or (not cat_key) or (not self.judges()):
            return
        held = [i for i in self.cart_ids() if not self.sold_out(i) and (self.held_units(i) or 0) > 0]
        kinds = [str((self.rows.get(i) or {}).get(cat_key) or '') for i in held]
        missing = [n for n, item in enumerate(self.items, 1) if not any((self.owns(item, k) for k in kinds))]
        if not missing or self.left() < len(missing) + room or self.time_left() < CALL_RESERVE + 20:
            return
        self.filled = True
        pool = {}
        for n in missing:
            item = self.items[n - 1]
            ids = [i for i in reversed(self.seen_order) if i in self.full and i not in held and self.buyable(i) and (not self.sold_out(i)) and self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))]
            pool[n] = list(dict.fromkeys(ids))[:6]
        if not all((pool[n] for n in missing)) and (not held):
            return
        shown = []
        for n in missing:
            shown.append(f'item {n} ({self.items[n - 1].name}):')
            shown += ['  ' + _judge_view(self.full.get(i), 800, 160) for i in pool[n]] or ['  (no opened listing)']
        prompt = FILL + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger() + '\n\nCART NOW:\n' + (self.cart_text() if held else 'empty')
        plan = self.fit_plan(short=True) if self.bound else ''
        if plan:
            prompt += '\n\nFIT NOTES:\n' + plan
        prompt += '\n\nITEMS WITH NO CART LINE AND THEIR OPENED LISTINGS:\n' + '\n'.join(shown)
        got = self.ask_json(prompt, [LIGHT], 6000) if not all((self.proven_lines(n) and self.proven_lines(n)[0] in pool[n] for n in missing)) else {}
        picks = (got or {}).get('pick') if isinstance((got or {}).get('pick'), dict) else {}
        add = {}
        for n in missing:
            sure = self.proven_lines(n)
            if sure and sure[0] in pool[n] and sure[0] not in add:
                add[sure[0]] = self.items[n - 1].qty
                continue
            ident = str(picks.get(str(n)) or picks.get(n) or '').strip()
            if room == 1 and ident not in pool[n]:
                free = [i for i in pool[n] if i not in add]
                ident = max(free, key=lambda i: (self.proves(i, n, True), -(self.price_of(i) or 0)), default='')
            if ident in pool[n] and ident not in add:
                add[ident] = self.items[n - 1].qty
        self.trace.append({'kind': 'clock_fill', 'missing': missing, 'pool': pool, 'got': got, 'add': add})
        if not add or (not held and len(add) < len(missing)):
            return
        lines = {**{i: self.held_units(i) or 1 for i in held}, **add}
        prices = [self.price_of(i) for i in lines]
        if isinstance(self.budget, (int, float)) and self.brief.get('budget_scope') != 'each' and all((p is not None for p in prices)):
            if sum((p * lines[i] for p, i in zip(prices, lines))) > self.budget + 1e-09:
                return
        calls = self.cart_calls(lines, [])
        if not calls or len(calls) + 1 > self.left():
            return
        for args in calls:
            faults = []
            args = _shape(args, self.schemas.get(cart), 'args', faults)
            if faults or self.finished or self.left() <= 1:
                return
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return
            self.note_cart(cart, args)
            self.observe(cart, reply.get('observation'))

    def next_arg_of(self, name: str, token: Any) -> str | None:
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        need = set(schema.get('required') or [])
        texts = {k for k in props if re.search('phrase|query|search|lookup|keyword|text|term|words|string|^q$', k, re.I)}
        if isinstance(token, bool):
            return None
        fits = {'integer', 'number'} if isinstance(token, int) else {'string'} if isinstance(token, str) else {'integer', 'string'}
        free = [k for k, v in props.items() if isinstance(v, dict) and k not in need and k not in texts and 'enum' not in v and fits & set(_kinds(v))]
        if len(free) > 1:
            free = [k for k in free if re.search('page|offset|cursor|start|skip|after|token|next|lead|from', k, re.I)]
        if len(free) != 1:
            return None
        self.trace.append({'kind': 'next_arg', 'tool': name, 'arg': free[0]})
        return free[0]

    def units_not_rows(self, name, args):
        if name != self.roles.get('cart'):
            return (args, '')
        props = (self.schemas.get(name) or {}).get('properties') or {}
        if any((isinstance(v, dict) and isinstance(v.get('enum'), list) and args.get(k) in v['enum'][1:] for k, v in props.items())):
            return (args, '')
        counts = [k for k, v in args.items() if isinstance(v, int) and (not isinstance(v, bool)) and v == 0 and 'integer' in _kinds(props.get(k) or {})]
        idents = [v for v in args.values() if isinstance(v, str) and v in self.rows]
        if len(counts) != 1 or len(idents) != 1 or idents[0] in self.cart_ids():
            return (args, '')
        cat_key = self.category_key()
        owner = self.owner_of(str((self.rows.get(idents[0]) or {}).get(cat_key) or '')) if cat_key else 0
        want = self.items[owner - 1].qty if 1 <= owner <= len(self.items) else 1
        top = (props.get(counts[0]) or {}).get('maximum')
        if isinstance(top, int) and (not isinstance(top, bool)):
            want = min(want, top)
        self.trace.append({'kind': 'units', 'id': idents[0], 'units': want})
        return ({**args, counts[0]: want}, f'\n[Kyo] {counts[0]} counts units to add, not a row number: sent {want} for {idents[0]}.')

    def nearest_order(self, hold: bool=False) -> str:
        cart = self.roles.get('cart')
        cat_key = self.category_key()
        if self.near_tried or self.finished or (not cart) or (not cat_key) or (not self.judges()):
            return ''
        if self.left() < len(self.items) + 2 or self.time_left() < CALL_RESERVE + 20:
            return ''
        self.near_tried = True
        pool = {}
        for n, item in enumerate(self.items, 1):
            ids = [i for i in reversed(self.seen_order) if i in self.full and self.buyable(i) and (not self.sold_out(i)) and self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))]
            pool[n] = list(dict.fromkeys(ids))[:6]
        if not all(pool.values()):
            return ''
        shown = []
        for n, ids in pool.items():
            shown.append(f'item {n} ({self.items[n - 1].name}):')
            shown += ['  ' + _judge_view(self.full.get(i), 800, 160) for i in ids]
        prompt = NEAR + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger() + '\n\nOPENED LISTINGS PER ITEM:\n' + '\n'.join(shown)
        got = self.ask_json(prompt, self.judges(), 6000) or {}
        picks = got.get('pick') if isinstance(got.get('pick'), dict) else {}
        lines = {}
        for n in pool:
            ident = str(picks.get(str(n)) or picks.get(n) or '').strip()
            if ident in pool[n] and ident not in lines:
                lines[ident] = self.items[n - 1].qty
        self.trace.append({'kind': 'nearest', 'got': got, 'lines': lines})
        if got.get('wants_closest') is not True:
            return ''
        rule = str(got.get('rule') or '').strip()[:200]
        if hold and (not self.near_held) and self.left() > max(12, self.max_steps // 4) and (self.time_left() > 90):
            self.near_held = True
            self.near_tried = False
            return f'The request says what to do when nothing fits ({rule}), so ending without a purchase fails it. First finish any lookups the requirements point at; then put in the cart, for each item, the listing nearest to its requirements as that rule allows, and check out.'
        if len(lines) < len(self.items):
            return ''
        held = list(self.cart_ids())
        calls = self.cart_calls(lines, [i for i in held if i not in lines])
        if not calls or len(calls) + 2 > self.left():
            return ''
        missed = str(got.get('missed') or '').strip()[:300]
        if self.roles.get('ask') and self.msg_used < self.msg_limit and (self.left() > len(calls) + 3):
            self.send([(0, '', 'No listing meets every requirement, so I am ordering the closest match: ' + ', '.join(lines) + (f' (it misses: {missed})' if missed else '') + '.')], len(calls) + 2)
        for args in calls:
            faults = []
            args = _shape(args, self.schemas.get(cart), 'args', faults)
            if faults or self.finished:
                return ''
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return ''
            self.note_cart(cart, args)
            self.observe(cart, reply.get('observation'))
        return f"The request says what to do when nothing fits ({rule}). Kyo put the closest listings in the cart: {', '.join(lines)}" + (f' (they miss: {missed})' if missed else '') + '. Check out now; end without a purchase only if the request clearly forbids these.'

    def budget_unit(self, new: float) -> float:
        old = self.budget
        if not isinstance(old, (int, float)) or isinstance(old, bool) or old <= 0 or new <= 0:
            return new
        if max(new, old) / min(new, old) <= 30:
            return new
        for factor in (100, 0.01):
            moved = new * factor
            if max(moved, old) / min(moved, old) <= 3:
                self.trace.append({'kind': 'budget_unit', 'read': new, 'kept': moved})
                return moved
        return new

    def written_counts(self):
        for item in self.items:
            name = item.name.strip()
            if item.qty != 1 or len(name) < 4:
                continue
            hits = {int(h.group(1)) for h in re.finditer('(?<![\\w.])(\\d{1,3})\\s+' + re.escape(name) + '(?![\\w])', self.request or '', re.I) if not re.search('\\bof\\s*$', (self.request or '')[:h.start()], re.I)}
            if len(hits) == 1 and min(hits) > 1:
                item.qty = min(hits)
                self.trace.append({'kind': 'written_count', 'item': name, 'was': 1, 'now': item.qty})
        for hit in re.finditer('(?<![\\w.])(\\d{1,3})\\s*[x\u00d7]\\s+([^,;.:!?]+?)(?=\\s*(?:[,;.:!?]|\\d{1,3}\\s*[x\u00d7]\\s|$))', self.request or ''):
            words = hit.group(2).split()
            want = int(hit.group(1))
            if not words or words[0][:1].isdigit() or want <= 0:
                continue
            roots = _roots(' '.join(words[:6]))
            scores = [len(roots & (_roots(item.name) | _roots(item.shelf or ''))) for item in self.items]
            best = max(scores, default=0)
            if best == 0 or scores.count(best) > 1:
                continue
            item = self.items[scores.index(best)]
            if item.qty != want:
                self.trace.append({'kind': 'written_count', 'item': item.name, 'was': item.qty, 'now': want})
                item.qty = want

    def clashes(self) -> list[str]:
        cat_key = self.category_key()
        refs = {m for _, m in self.refs}
        out = []
        for ident in self.cart_ids():
            block = _fact_block(self.full.get(ident) or {}) or {}
            if not block:
                continue
            number = self.owner_of(str((self.rows.get(ident) or {}).get(cat_key) or '')) if cat_key else 1 if len(self.items) == 1 else 0
            if not 1 <= number <= len(self.items):
                continue
            item = self.items[number - 1]
            for need in item.musts + item.told:
                if need in item.dropped or need in refs or self.described(number, need):
                    continue
                prop, _, value = need.partition(':')
                value = value.strip().lower()
                figure = re.search('(\\d+(?:\\.\\d+)?)\\s*[a-z]', value)
                if prop.strip() and figure:
                    want = float(figure.group(1))
                    for key, have in block.items():
                        if _names(_stem(prop), _stem(key)) and (not _choice_key(key)) and isinstance(have, (int, float, str)) and (not isinstance(have, bool)) and re.fullmatch('\\s*\\d+(?:\\.\\d+)?\\s*', str(have)):
                            bare = float(str(have))
                            if want > 0 and bare > 0 and max(bare, want) / min(bare, want) >= 10:
                                out.append(f'{ident} states {key} = {have} with no unit, far from the {figure.group(0)} the requirement names ({need})')
                                break
                if not prop.strip() or not value or re.search('\\d|\\b(?:least|most|not|without|than|between|except)\\b', value):
                    continue
                keys = [(k, str(v).lower()) for k, v in block.items() if _names(_stem(prop), _stem(k)) and (not _choice_key(k)) and isinstance(v, (str, int, float)) and (not isinstance(v, bool))]
                if not keys:
                    continue
                options = [o.strip() for o in re.split('\\bor\\b|,|/', value) if o.strip()]

                def holds(option: str, have: str) -> bool:
                    want = {w.rstrip('s') for w in re.findall('[a-z0-9]+', option)}
                    got = {w.rstrip('s') for w in re.findall('[a-z0-9]+', have)}
                    return bool(want & got)
                if any((holds(o, have) for o in options for _, have in keys)):
                    continue
                key, have = keys[0]
                out.append(f'{ident} states {key} = {have[:60]}, not "{value}" ({need})')
        return out

    def referred_pages(self):
        cat_key = self.category_key()
        tool = self.facts_tool()
        target = self.single_target(tool) if tool else None
        if not cat_key or not tool or not target or len(self.referred) >= 6:
            return ''
        opened = []
        for ident in self.cart_ids():
            record = self.full.get(ident)
            if not record or ident in self.referred:
                continue
            self.referred.add(ident)
            label = str((self.rows.get(ident) or {}).get(cat_key) or '')
            own = set(_codes(_dump(self.rows.get(ident) or {})))
            for code in [c for c in dict.fromkeys(_codes(_dump(record))) if c not in own][:3]:
                if len(opened) >= 2 or self.left() <= self.tail() + 2 or self.finished:
                    break
                holders = [i for i, row in self.rows.items() if i != ident and str(row.get(cat_key) or '') == label and code in _codes(_dump(row))]
                if len(holders) != 1 or holders[0] in opened:
                    continue
                faults = []
                args = _shape({target: holders[0]}, self.schemas.get(tool), 'args', faults)
                if faults:
                    continue
                reply = self.act(tool, args)
                if reply is None:
                    break
                self.observe(tool, reply.get('observation'))
                opened.append(holders[0])
        if not opened:
            return ''
        self.trace.append({'kind': 'referred_pages', 'ids': opened})
        return f"\n[Kyo opened the page a cart line refers to: {', '.join(opened)} (one step each)]"

    def proves(self, ident: str, number: int, loose: bool=False) -> bool:
        block = _fact_block(self.full.get(ident) or {}) or {}
        if not block or not 1 <= number <= len(self.items):
            return False
        item = self.items[number - 1]
        refs = {m for _, m in self.refs}
        checked = 0
        for need in item.musts + item.told:
            if need in item.dropped or self.described(number, need):
                continue
            prop, _, value = need.partition(':')
            value = value.strip().lower()
            if need in refs:
                if need not in self.ref_values:
                    return False
                value = str(self.ref_values[need]).strip().lower()
            keys = [str(v).lower() for k, v in block.items() if _names(_stem(prop), _stem(k)) and (not _choice_key(k)) and isinstance(v, (str, int, float)) and (not isinstance(v, bool))]
            if loose and prop.strip() and value and (not keys):
                record = self.full.get(ident) or {}
                stack, leaves = ([record], [])
                while stack:
                    node = stack.pop()
                    if isinstance(node, dict):
                        stack += list(node.values())
                    elif isinstance(node, list):
                        stack += node
                    elif isinstance(node, str):
                        leaves.append(node.lower())
                if leaves:
                    keys = [' '.join(leaves)]
            if not prop.strip() or not value or not keys:
                return False
            bound = re.search('\\b(least|most|under|over|up to|minimum|maximum)\\b\\s*([\\d.]+)\\s*([a-z%"]*)', value)
            after = None if bound else re.search('([\\d.]+)\\s*([a-z%"]*)\\s*(?:or\\s+)?(at most|at least|less|more|maximum|minimum)\\b', value)
            if bound or after:
                try:
                    limit = float((bound or after).group(2 if bound else 1))
                except ValueError:
                    return False
                unit = bound.group(3) if bound else after.group(2)
                word = bound.group(1) if bound else after.group(3)
                low = word in ('least', 'over', 'minimum', 'at least', 'more')
                met = False
                for have in keys:
                    hit = re.search('(\\d+(?:\\.\\d+)?)\\s*([a-z%"]*)', have)
                    if not hit or (unit and hit.group(2) and hit.group(2)[:2] != unit[:2]):
                        continue
                    number_seen = float(hit.group(1))
                    if (number_seen >= limit) if low else (number_seen <= limit):
                        met = True
                        break
                if not met:
                    return False
            elif re.search('\\b(?:not|without|except|than|between)\\b', value):
                return False
            else:
                options = [o.strip() for o in re.split('\\bor\\b|,|/', value) if o.strip()]

                def has(option: str, have: str) -> bool:
                    want = [_fold(w) for w in re.findall('[a-z0-9.]+', option)]
                    got = {_fold(w) for w in re.findall('[a-z0-9.]+', have)}
                    return bool(want) and all((w in got for w in want)) and (not _split_value(option, have))
                if not any((has(o, h) for o in options for h in keys)):
                    return False
            checked += 1
        return checked > 0

    def proven_lines(self, number):
        cap = self.caps.get(number)
        pool = list(self.full) if len(self.items) == 1 else self.item_ids(number)
        ids = [i for i in pool if i in self.full and self.buyable(i) and (not self.sold_out(i)) and (cap is None or (self.price_of(i) or 0) <= cap) and self.proves(i, number) and self.name_shown(i, number)]
        return sorted(dict.fromkeys(ids), key=lambda i: self.price_of(i) if self.price_of(i) is not None else float('inf'))

    def proven_set(self) -> dict[str, int] | None:
        if not self.category_key():
            return None
        choices = {n: self.proven_lines(n) for n in range(1, len(self.items) + 1)}
        if not all(choices.values()):
            return None
        index = self.fit_index() if self.bound else None
        mains = sorted({main for _, main in self.bound})
        best, best_total = None, None
        for device in (choices[mains[0]][:6] if mains else [None]):
            lines = {}
            for n in choices:
                pool = choices[n]
                if device is not None and n == mains[0]:
                    pool = [device]
                elif device is not None and any((part == n and main == mains[0] for part, main in self.bound)):
                    mark = index['marks'].get(device) or {}
                    pool = [i for i in pool if self.fits_device(i, mark, index)]
                pick = next((i for i in pool if i not in lines), None)
                if pick is None:
                    lines = None
                    break
                lines[pick] = self.items[n - 1].qty
            if not lines:
                continue
            prices = [self.price_of(i) for i in lines]
            if any((p is None for p in prices)):
                continue
            total = sum((p * lines[i] for p, i in zip(prices, lines)))
            if isinstance(self.budget, (int, float)):
                if self.brief.get('budget_scope') == 'each' and any((p > self.budget for p in prices)):
                    continue
                if self.brief.get('budget_scope') != 'each' and total > self.budget + 1e-09:
                    continue
            if best_total is None or total < best_total:
                best, best_total = (lines, total)
        return best

    def set_ready(self) -> str:
        if self.set_done or self.buys or self.finished or self.brief.get('pick') in ('cheapest', 'most', 'least'):
            return ''
        cart = self.roles.get('cart')
        lines = self.proven_set() if cart else None
        if not lines:
            return ''
        held = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        if set(held) == set(lines):
            self.set_done = True
            return ''
        calls = self.cart_calls(lines, [i for i in held if i not in lines])
        if not calls or len(calls) + self.tail() + 2 > self.left():
            return ''
        self.set_done = True
        for args in calls:
            faults = []
            args = _shape(args, self.schemas.get(cart), 'args', faults)
            if faults or self.finished:
                return ''
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return ''
            self.note_cart(cart, args)
            self.observe(cart, reply.get('observation'))
        self.trace.append({'kind': 'proven_set', 'lines': lines})
        return '[Kyo] The pages of these listings state every requirement in force under their own keys, so Kyo set the cart to them: ' + ', '.join((f'{i} x{q}' for i, q in lines.items())) + '. Check out now, unless one of them breaks a rule of the request that its page shows.'

    def proof_gaps(self):
        cat_key = self.category_key()
        if not cat_key:
            return []
        cheapest = self.brief.get('pick') == 'cheapest'
        cart = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        out = []
        for line in cart:
            number = self.owner_of(str((self.rows.get(line) or {}).get(cat_key) or ''))
            if not 1 <= number <= len(self.items):
                continue
            better = [i for i in self.proven_lines(number) if i not in cart]
            if not better:
                continue
            alt = better[0]
            if not self.proves(line, number, True):
                out.append(f"{line} does not state every requirement of item {number} ({self.items[number - 1].name}) under its own keys, while {alt}'s page does.")
            elif cheapest and (not any((number in pair for pair in self.bound))) and (self.price_of(alt) or float('inf')) < (self.price_of(line) or 0):
                self.gap_alt[line] = alt
                out.append(f'{alt} also states every requirement of item {number} and costs {self.price_of(alt):g}, less than {line} ({self.price_of(line):g}).')
        return out

    def cart_to(self, lines: dict[str, int], spare: int) -> bool:
        cart = self.roles.get('cart')
        held = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        calls = self.cart_calls(lines, [i for i in held if i not in lines]) if cart else None
        if not calls or len(calls) + spare > self.left():
            return False
        for args in calls:
            faults = []
            args = _shape(args, self.schemas.get(cart), 'args', faults)
            if faults or self.finished:
                return False
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return False
            self.note_cart(cart, args)
            self.observe(cart, reply.get('observation'))
        return True

    def keep_lines(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if name != self.roles.get('cart'):
            return (args, '')
        props = (self.schemas.get(name) or {}).get('properties') or {}
        key = next((k for k, v in args.items() if isinstance(v, list) and v and all((isinstance(x, dict) for x in v)) and 'array' in _kinds(props.get(k) or {})), None)
        cat_key = self.category_key()
        if not key or not cat_key:
            return (args, '')
        sent = [x for x in args[key] if any((isinstance(v, str) and v in self.rows for v in x.values()))]
        if len(sent) != len(args[key]):
            return (args, '')
        ids = [next((v for v in x.values() if isinstance(v, str) and v in self.rows)) for x in sent]
        owners = {self.owner_of(str((self.rows.get(i) or {}).get(cat_key) or '')) for i in ids}
        held = [i for i in self.cart_ids() if i not in ids and (self.held_units(i) or 0) > 0 and (not self.sold_out(i))]
        keep = [i for i in held if self.owner_of(str((self.rows.get(i) or {}).get(cat_key) or '')) not in owners]
        self.pending_held = keep
        if not keep or not self.wipes.get(name):
            return (args, '')
        shape = sent[0]
        id_key = next((k for k, v in shape.items() if isinstance(v, str) and v in self.rows))
        count_key = next((k for k, v in shape.items() if isinstance(v, int) and (not isinstance(v, bool))), None)
        extra = [{id_key: i, **({count_key: self.held_units(i) or 1} if count_key else {})} for i in keep]
        self.trace.append({'kind': 'keep_lines', 'kept': keep})
        return ({**args, key: list(args[key]) + extra}, f"\n[Kyo] This cart call sets the whole basket, so Kyo kept the lines of the other items in it: {', '.join(keep)}.")

    def note_wipe(self, name, args, observation):
        held, self.pending_held = (self.pending_held, [])
        if not held or self.wipes.get(name) or self.finished or self.left() <= self.tail() + 1:
            return ''
        text = _dump(observation)
        if not any((i in text for i in self.rows)) or any((i in text for i in held)):
            return ''
        self.wipes[name] = True
        fixed, _ = self.keep_lines(name, args)
        if fixed is args:
            return ''
        reply = self.act(name, fixed)
        if reply is None:
            return ''
        self.note_cart(name, fixed)
        self.observe(name, reply.get('observation'))
        self.trace.append({'kind': 'wipe_fixed', 'held': held})
        return f"\n[Kyo] That cart call replaced the whole basket and dropped {', '.join(held)}; Kyo put them back (one step). From now on Kyo keeps the other items' lines in such calls."

    def plausible(self, ident: str, number: int) -> bool:
        cat_key = self.category_key()
        row = self.rows.get(ident) or {}
        if not cat_key or not 1 <= number <= len(self.items) or not self.owns(self.items[number - 1], str(row.get(cat_key) or '')):
            return False
        if not self.buyable(ident) or self.sold_out(ident):
            return False
        cap = self.caps.get(number)
        if cap is not None and (self.price_of(ident) or 0) > cap:
            return False
        have = {w.rstrip('s') for w in re.findall('[a-z0-9.]+', _dump(self.full.get(ident) or row).lower())}
        want = {w.rstrip('s') for v in self.value_words(number) for w in re.findall('[a-z0-9.]+', v.lower()) if len(w) >= 3 or w[:1].isdigit()}
        return want <= have

    def descent(self, number: int, price: float, limit: int) -> list[str] | None:
        seen = sorted([i for i in self.item_ids(number) if self.plausible(i, number) and self.price_of(i) is not None and self.price_of(i) < price], key=lambda i: self.price_of(i))
        unread = [i for i in seen if i not in self.full]
        if unread:
            self.fill_facts(unread[:2], self.tail() + 3)
        cap = self.price_of(seen[0]) if seen else price
        found = []
        rounds = 0
        while rounds < 4 and self.step < limit - 1 and self.time_left() > EXPLORE_RESERVE + 20 and (not self.finished):
            rounds += 1
            got = self.list_category(number, cap - 1 if cap > 1 else cap * 0.9, 1 if limit - self.step >= 3 else 0)
            if got is None:
                break
            found += [i for i in got if i not in found]
            cheaper = sorted([i for i in found if i not in self.full and self.plausible(i, number) and self.price_of(i) is not None and self.price_of(i) < cap], key=lambda i: self.price_of(i))
            if not cheaper:
                break
            self.fill_facts(cheaper[:2], self.tail() + 3)
            cap = self.price_of(cheaper[0])
        self.trace.append({'kind': 'descent', 'item': number, 'rounds': rounds, 'cap': cap, 'rows': len(found)})
        if not rounds and not seen:
            return None
        return found

    def end_reads(self, observations):
        cat_key = self.category_key()
        if not cat_key:
            return []
        found = [i for seen in observations for i, _ in self.records_in(seen)]
        held = [str((self.rows.get(i) or {}).get(cat_key) or '') for i in self.cart_ids()]
        out = []
        for number, item in enumerate(self.items, 1):
            if number in self.end_read or any((self.owns(item, k) for k in held)):
                continue
            rows = sorted([i for i in dict.fromkeys(found) if i not in self.full and self.plausible(i, number) and self.price_of(i) is not None], key=lambda i: self.price_of(i))[:2]
            if not rows or self.left() - self.tail() < 6:
                continue
            self.end_read.add(number)
            out += self.fill_facts(rows, self.tail() + 4)
        if out:
            self.trace.append({'kind': 'end_reads', 'read': len(out)})
            return ['[Kyo read the cheapest unread rows of this search that name every required value]'] + out
        return []

    def sold_peers(self, events: list[tuple[str, str]]) -> str:
        cat_key = self.category_key()
        gone = [i for kind, what in events if kind == 'gone' for i in re.split('[,\\s]+', what or '') if i in self.rows]
        reads = []
        for line in gone[:1]:
            price = self.price_of(line)
            number = self.owner_of(str((self.rows.get(line) or {}).get(cat_key) or '')) if cat_key else 0
            if price is None or not 1 <= number <= len(self.items):
                continue
            got = self.list_category(number, price, 0) or []
            pool = sorted([i for i in dict.fromkeys(got + self.item_ids(number)) if i != line and i not in self.full and self.plausible(i, number) and self.price_of(i) is not None and self.price_of(i) <= price], key=lambda i: self.price_of(i))[:2]
            reads += self.fill_facts(pool, self.tail() + 4)
        if not reads:
            return ''
        self.trace.append({'kind': 'sold_peers', 'read': len(reads)})
        return '\n[Kyo read the cheapest listings of that kind priced at or below the sold-out line]\n' + '\n'.join(reads)

    def chain_hunt(self) -> str:
        if self.brief.get('pick') != 'cheapest' or self.buys or self.finished:
            return ''
        plan = self.phrase_plan()
        if not plan:
            return ''
        tool, base, field = plan
        for number, need in self.refs:
            value = self.ref_values.get(need)
            if not value or need in self.chain_hunted or value.startswith('as stated on') or self.left() - self.tail() < 3 * len(self.items) + 4:
                continue
            self.chain_hunted.add(need)
            words = ' '.join([value, self.items[number - 1].name])
            faults = []
            args = _shape({**base, field: words}, self.schemas.get(tool), 'args', faults)
            if faults:
                continue
            reply = self.act(tool, args)
            if reply is None:
                return ''
            seen = reply.get('observation')
            self.observe(tool, seen)
            found = [i for i, _ in self.records_in(seen)]
            rows = sorted([i for i in found if i not in self.full and self.plausible(i, number) and self.price_of(i) is not None], key=lambda i: self.price_of(i))[:2]
            reads = self.fill_facts(rows, self.tail() + 4)
            self.trace.append({'kind': 'chain_hunt', 'need': need, 'rows': len(found), 'read': len(reads)})
            return f"[Kyo searched {words!r} for the cheapest listing that carries the value the chain gave ({value}): " + (', '.join(found[:8]) or 'no rows') + ']' + ('\n' + '\n'.join(reads) if reads else '')
        return ''

    def keys_out(self, name, args):
        plan = self.phrase_plan()
        if not plan or plan[0] != name:
            return None
        field = plan[2]
        query = args.get(field)
        if not isinstance(query, str):
            return None
        props = {w for _, need in self.needs_now() for w in re.findall('[a-z]{3,}', need.partition(':')[0].lower())}
        words = query.split()
        kept = [w for w in words if w.lower().strip('.,;:') not in props]
        if not kept or len(kept) == len(words):
            return None
        return {**args, field: ' '.join(kept)}

    def dear_lines(self) -> str:
        cat_key = self.category_key()
        if not cat_key:
            return ''
        reads = []
        for line in self.cart_ids():
            price = self.price_of(line)
            number = self.owner_of(str((self.rows.get(line) or {}).get(cat_key) or ''))
            if price is None or not 1 <= number <= len(self.items):
                continue
            prices = sorted((self.price_of(i) for i in self.item_ids(number) if self.price_of(i) is not None and self.buyable(i)))
            if len(prices) < 3 or price <= prices[int(len(prices) * 2 / 3) - 1]:
                continue
            below = sorted([i for i in self.item_ids(number) if i != line and i not in self.full and self.plausible(i, number) and self.price_of(i) is not None and self.price_of(i) < price], key=lambda i: self.price_of(i))[:2]
            reads += self.fill_facts(below, self.tail() + 4)
        if not reads:
            return ''
        self.trace.append({'kind': 'dear_line', 'read': len(reads)})
        return '\n[Kyo read cheaper listings of the same kind that name every required value, since the cart line costs more than most rows seen]\n' + '\n'.join(reads)

    def link_rungs(self, tool: str, base: dict[str, Any], field: str, shop: str, values: list[str]) -> list[str]:
        rungs = ([f'{shop} {values[0]}'] if values else []) + [shop]
        for words in rungs:
            memo = ' '.join(words.lower().split())
            if memo in self.link_seen or self.left() <= self.tail() + 6:
                continue
            faults = []
            args = _shape({**base, field: words}, self.schemas.get(tool), 'args', faults)
            if faults:
                continue
            reply = self.act(tool, args)
            if reply is None:
                return []
            seen = reply.get('observation')
            self.observe(tool, seen)
            rows = [i for i, _ in self.records_in(seen)]
            known = self.shop_rows(shop, rows)
            if words == shop and rows and len(known) == len(rows):
                page = self.paging(tool, seen)
                if page and page[3] is not None and self.left() > self.tail() + 6:
                    more = self.act(tool, _shape({**args, page[2]: page[3]}, self.schemas.get(tool), 'args', []))
                    if more is not None:
                        self.observe(tool, more.get('observation'))
                        extra = [i for i, _ in self.records_in(more.get('observation'))]
                        rows += extra
                        known = self.shop_rows(shop, rows)
            self.link_seen[memo] = rows
            self.trace.append({'kind': 'link_rung', 'words': words[:80], 'rows': known[:4]})
            if known:
                return known
        return []

    def relation_words(self, need):
        prop, _, value = need.partition(':')
        shops = re.findall('\\b[A-Z][a-z]{3,}\\b', value)
        for sentence in re.split('(?<=[.!?;])\\s+', self.request or ''):
            if not shops or not all((shop in sentence for shop in shops[:1])):
                continue
            if re.search('\\b(?:same|identical|match\\w*|equal)\\b', sentence, re.I):
                return need
            clause = sentence.strip()[:220]
            self.trace.append({'kind': 'relation_words', 'need': need[:120]})
            return f'{need} (the request says: "{clause}")'
        return need

    def need_codes(self, number: int) -> list[str]:
        refs = {m for _, m in self.refs}
        out = []
        for owner, need in self.needs_now():
            if owner != number or need in refs:
                continue
            out += [c for c in _codes(need.partition(':')[2]) if c not in out]
        return out

    def row_text(self, ident: str) -> str:
        return _dump(self.full.get(ident) or self.rows.get(ident) or {})

    def fit_rows(self, number, code):
        cat_key = self.category_key()
        item = self.items[number - 1]
        rows = [i for i in self.seen_order if self.buyable(i) and (not self.sold_out(i)) and code in _codes(self.row_text(i)) and ((not cat_key) or self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or '')))]
        return sorted(dict.fromkeys(rows), key=lambda i: self.price_of(i) if self.price_of(i) is not None else float('inf'))

    def fit_verdict(self, ident: str, number: int, code: str) -> str:
        if ident not in self.full:
            return 'not opened'
        block = _fact_block(self.full.get(ident) or {}) or {}
        item = self.items[number - 1]
        for need in item.musts + item.told:
            if need in item.dropped or code in _codes(need):
                continue
            prop, _, value = need.partition(':')
            keys = [str(v).lower() for k, v in block.items() if _names(_stem(prop), _stem(k))]
            if not keys:
                return f'opened, {prop.strip()} not on page'
            want = {_fold(w) for w in re.findall('[a-z0-9.]+', value.lower()) if not re.fullmatch('(?:at|least|most|or|and|the|a)', w)}
            got = {_fold(w) for k in keys for w in re.findall('[a-z0-9.]+', k)}
            if want and not (want & got):
                return f'opened, {prop.strip()} {keys[0][:40]} (wanted {value.strip()[:40]})'
        return 'opened, meets the other needs as written'

    def without_dead_codes(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        plan = self.phrase_plan()
        if not self.dead_codes or not plan or plan[0] != name or not isinstance(args.get(plan[2]), str):
            return (args, '')
        query = args[plan[2]]
        kept = query
        for code in self.dead_codes:
            kept = re.sub('(?i)\\b' + '[-\\s]?'.join((re.escape(p) for p in re.findall('[a-z]+|\\d+', code))) + '\\b', ' ', kept)
        kept = ' '.join(kept.split())
        if kept == query:
            return (args, '')
        if not kept:
            return (args, 'EMPTY')
        return ({**args, plan[2]: kept}, f'\n[Kyo] Left the fit code out of this search (it found nothing before; codes sit in the listing text, which the search does not index): {kept!r}.')

    def fit_scan(self, name, args, observation):
        coded = [(n, c) for n in range(1, len(self.items) + 1) for c in self.need_codes(n)]
        if not coded or self.buys or self.time_left() < EXPLORE_RESERVE + 60:
            return ''
        query = ' '.join((str(v) for v in args.values() if isinstance(v, str)))
        if not self.records_in(observation):
            for code in set(_codes(query)):
                if any((code == c for _, c in coded)):
                    self.dead_codes.add(code)
            return ''
        held = [str((self.rows.get(i) or {}).get(self.category_key()) or '') for i in self.cart_ids()] if self.category_key() else []
        lines = []
        for number, code in coded:
            item = self.items[number - 1]
            if any((self.owns(item, k) for k in held)):
                continue
            spare = self.left() - self.tail() - 3 * len(self.items)
            page = self.paging(name, observation)
            seen_now = observation
            while len(self.fit_rows(number, code)) < 3 and page and page[3] is not None and self.fit_pages < 6 and spare > 6 and (not self.finished):
                more = self.act(name, _shape({**args, page[2]: page[3]}, self.schemas.get(name), 'args', []))
                if more is None:
                    break
                self.fit_pages += 1
                spare -= 1
                seen_now = more.get('observation')
                self.observe(name, seen_now)
                page = self.paging(name, seen_now)
            fitting = self.fit_rows(number, code)
            unread = [i for i in fitting if i not in self.full][:3]
            if unread and self.fit_reads < 6 and spare > 6:
                take = unread[:min(3, 6 - self.fit_reads)]
                self.fill_facts(take, self.tail() + 3 * len(self.items) + 3)
                self.fit_reads += len(take)
            fitting = self.fit_rows(number, code)
            if not fitting and number not in self.swept_items and spare > 8:
                self.swept_items.add(number)
                self.shop_sweep(number, code)
                fitting = self.fit_rows(number, code)
            if fitting:
                lines.append(f'item {number} ({item.name}), rows naming {code.upper()}: ' + '; '.join((f'{i} ({self.price_of(i):g}) {self.fit_verdict(i, number, code)}' if self.price_of(i) is not None else f'{i} {self.fit_verdict(i, number, code)}' for i in fitting[:5])))
            else:
                lines.append(f'item {number} ({item.name}): no row seen yet names {code.upper()}')
        if not lines:
            return ''
        self.trace.append({'kind': 'fit_scan', 'lines': len(lines), 'pages': self.fit_pages, 'reads': self.fit_reads})
        return '\n[fit scan: a part fits only when its own text names the code]\n' + '\n'.join(lines)

    def shop_sweep(self, number: int, code: str) -> None:
        plan = self.phrase_plan()
        cat_key = self.category_key()
        if not plan or not cat_key:
            return
        tool, base, field = plan
        item = self.items[number - 1]
        family = _code_family(code)
        shops = []
        for ident in self.seen_order:
            row = self.rows.get(ident) or {}
            if not self.owns(item, str(row.get(cat_key) or '')) or not any((_code_family(c) == family for c in _codes(self.row_text(ident)))):
                continue
            shop = next((str(v).strip() for k, v in row.items() if isinstance(v, str) and len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant|producer', str(k), re.I)), '')
            if shop and shop not in shops:
                shops.append(shop)
        kind = ' '.join([w for w in re.findall('[A-Za-z]+', item.name) if len(w) > 2][:3])
        for shop in shops[:3]:
            if self.fit_rows(number, code) or self.left() - self.tail() < 3 * len(self.items) + 6 or self.finished:
                break
            words = f'{shop} {kind}'.strip()
            args = _shape({**base, field: words}, self.schemas.get(tool), 'args', [])
            reply = self.act(tool, args)
            if reply is None:
                return
            self.observe(tool, reply.get('observation'))
            self.trace.append({'kind': 'shop_sweep', 'words': words, 'rows': len(self.records_in(reply.get('observation')))})
        fresh = [i for i in self.fit_rows(number, code) if i not in self.full][:2]
        if fresh:
            self.fill_facts(fresh, self.tail() + 3 * len(self.items) + 3)

    def code_misses(self) -> list[str]:
        cat_key = self.category_key()
        out = []
        for ident in self.cart_ids():
            number = self.owner_of(str((self.rows.get(ident) or {}).get(cat_key) or '')) if cat_key else (1 if len(self.items) == 1 else 0)
            if not 1 <= number <= len(self.items):
                continue
            for code in self.need_codes(number):
                if code not in _codes(self.row_text(ident)):
                    others = [i for i in self.fit_rows(number, code) if i != ident][:3]
                    out.append(f"{ident} does not name {code.upper()} in its text." + (f" Rows that do: {', '.join(others)}." if others else ''))
        return out

    def weak_figures(self):
        cat_key = self.category_key()
        refs = {m for _, m in self.refs}
        out = []
        for ident in self.cart_ids():
            block = _fact_block(self.full.get(ident) or {}) or {}
            number = self.owner_of(str((self.rows.get(ident) or {}).get(cat_key) or '')) if cat_key else (1 if len(self.items) == 1 else 0)
            if not block or not 1 <= number <= len(self.items):
                continue
            item = self.items[number - 1]
            for need in item.musts + item.told:
                if need in item.dropped or need in refs or _codes(need):
                    continue
                prop, _, value = need.partition(':')
                figure = re.search('\\b(?:least|most|under|over|up to|minimum|maximum)\\b\\s*(\\d+(?:\\.\\d+)?)\\s*([a-z%"]+)', value.lower())
                if not prop.strip() or not figure:
                    continue
                own = [str(v) for k, v in block.items() if _names(_stem(prop), _stem(k)) and (not _choice_key(k)) and re.search('\\d', str(v))]
                if own:
                    continue
                unit = figure.group(2)[:2]
                same = [f'{k} = {str(v)[:30]}' for k, v in block.items() if (not _names(_stem(prop), _stem(k))) and re.search('\\d\\s*' + re.escape(unit), str(v).lower())]
                alts = [i for i in self.proven_lines(number) if i != ident][:3]
                text = f'{ident} does not state "{need}" under a key named for {prop.strip()}.'
                if same:
                    text += f' It does give {same[0]}: if that is the same measure under another name, the line stays.'
                if alts:
                    text += f" Listings that state it under that key: {', '.join(alts)}."
                out.append(text)
                break
        return out

    def referred_names(self) -> str:
        plan = self.phrase_plan()
        tool = self.facts_tool()
        target = self.single_target(tool) if tool else None
        if not plan or not tool or not target or len(self.named_refs) >= 2:
            return ''
        opened = []
        for ident in self.cart_ids():
            record = self.full.get(ident)
            if not record:
                continue
            prose = ' '.join((v for v in record.values() if isinstance(v, str)))
            for hit in re.finditer('\\brefer(?:s|red)?\\s+to\\s+(?:the\\s+)?([A-Za-z0-9][\\w \'&-]{2,60}?)\\s+(?:listing|product|page|item)\\b', prose, re.I):
                phrase = hit.group(1).strip()
                if phrase.lower() in self.named_refs or len(self.named_refs) >= 2 or self.left() <= self.tail() + 3:
                    continue
                self.named_refs.add(phrase.lower())
                words = {w for w in re.findall('[a-z0-9]{3,}', phrase.lower())}
                found = self.search_words(plan[0], plan[1], plan[2], phrase)
                pick = next((i for i in found if i != ident and i not in self.cart_ids() and len(words & set(re.findall('[a-z0-9]{3,}', self.row_text(i).lower()))) >= min(2, len(words))), None)
                if not pick or pick in self.full:
                    continue
                faults = []
                args = _shape({target: pick}, self.schemas.get(tool), 'args', faults)
                if faults:
                    continue
                reply = self.act(tool, args)
                if reply is None:
                    break
                self.observe(tool, reply.get('observation'))
                opened.append(pick)
        if not opened:
            return ''
        self.trace.append({'kind': 'referred_names', 'ids': opened})
        return f"\n[Kyo opened the listing a cart line's page refers to by name: {', '.join(opened)}]"

    def name_shown(self, ident: str, number: int) -> bool:
        cat_key = self.category_key()
        if not cat_key or not 1 <= number <= len(self.items):
            return True
        label = str((self.rows.get(ident) or {}).get(cat_key) or '').lower()
        covered = {_fold(w) for w in re.findall('[a-z0-9]+', label)}
        want = [w for w in re.findall('[a-z]{3,}', self.items[number - 1].name.lower()) if _fold(w) not in covered]
        if not want:
            return True
        record = self.full.get(ident) or self.rows.get(ident) or {}
        stack, leaves = ([record], [])
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                stack += [v for k, v in node.items() if not re.search('remark|note', str(k), re.I)]
            elif isinstance(node, list):
                stack += node
            elif isinstance(node, str):
                leaves.append(node.lower())
        got = {_fold(w) for w in re.findall('[a-z0-9]+', ' '.join(leaves))}
        return all((_fold(w) in got for w in want))

    def planned_basket(self, lead):
        cat_key = self.category_key()

        def kind(ident: str) -> str:
            return str((self.rows.get(ident) or {}).get(cat_key) or '').strip().lower() if cat_key else ident
        plan = {}
        for number, ids in self.lead_items.items():
            index = _count(number)
            pick = next((i for i in ids if not self.sold_out(i)), None)
            if pick and index and 1 <= index <= len(self.items) and kind(pick) not in {kind(p) for p in plan}:
                plan[pick] = self.items[index - 1].qty
        if lead not in plan:
            return None
        keep = {i: self.cart_lines.get(i, 1) for i in self.cart_ids() if i not in plan and (not self.sold_out(i)) and kind(i) not in {kind(p) for p in plan}}
        if len(plan) + len({kind(i) for i in keep}) < len(self.items):
            return None
        lines = {**keep, **plan}
        if self.bound:
            index = self.fit_index()
            for part, main in self.bound:
                devices = [i for i in lines if i in self.item_ids(main)]
                mark = index['marks'].get(devices[0]) or {} if len(devices) == 1 else None
                if mark is not None and any((self.fits_device(i, mark, index) is False for i in lines if i in self.item_ids(part))):
                    self.trace.append({'kind': 'basket_misfit', 'lines': sorted(lines)})
                    return None
        return lines

    def buy_now(self, tag: str) -> list[str]:
        buy = self.roles.get('buy')
        faults = []
        args = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), 'args', faults) if buy else None
        if not buy or faults or self.finished:
            return []
        if self.left() >= 2:
            self.reread_offers()
        shown, held = self.offer_proof()
        if held or self.finished or self.left() < 1:
            return []
        replies = []
        for attempt in range(2):
            if self.finished or self.left() < 1 or (attempt and any((self.sold_out(i) for i in self.cart_ids()))):
                break
            if self.left() >= 2 and self.time_left() > CALL_RESERVE + 15:
                go, said = self.gate('order')
                if said:
                    replies.append(said)
                if not go:
                    break
            if self.finished or self.left() < 1:
                break
            reply = self.act(buy, args)
            self.trace.append({'kind': tag, 'reply': _clip(reply, 200)})
            if reply is None:
                break
            self.buys += 1
            if not reply.get('finished'):
                self.refused_at = self.step
            self.observe(buy, reply.get('observation'))
            replies.append(self.show(reply.get('observation')))
            if self.finished or reply.get('finished'):
                break
        return replies

    def late_finish(self):
        cart, buy = (self.roles.get('cart'), self.roles.get('buy'))
        if not cart or not buy or self.finished:
            return ''
        cat_key = self.category_key()
        lines = [i for i in self.built if not self.sold_out(i)]
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        now = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        missing = []
        if len(lines) >= len(self.items):
            missing = [i for i in lines if i not in now or (i in self.dropped and (not fresh))]
            if missing:
                want = {i: self.cart_lines.get(i) or 1 for i in now if i not in missing}
                for i in missing:
                    owner = self.owner_of(str((self.rows.get(i) or {}).get(cat_key) or '')) if cat_key else 0
                    want[i] = self.items[owner - 1].qty if 1 <= owner <= len(self.items) else 1
                if not self.cart_to(want, 1):
                    return ''
        elif len(self.items) >= 2 and now:
            self.clock_fill()
            if not self.cart_complete():
                return ''
        else:
            return ''
        if self.finished or self.left() < 1:
            return ''
        if any((' sold out' in f or ' over the ' in f for f in self.line_faults())):
            return ''
        replies = self.buy_now('late_finish')
        if not replies:
            return ''
        back = f"put {', '.join(missing)} back and " if missing else ''
        return 'Not run (Kyo acted instead): with this few steps left an ending without a purchase scores nothing, and the cart covers every item, so Kyo ' + back + 'placed the order. Store reply: ' + ' | then: '.join(replies)

    def next_page_big(self, name: str, args: dict[str, Any], observation: Any) -> str:
        cat_key = self.category_key()
        if self.max_steps < SWEEP_BIG or not cat_key or self.left() - self.tail() < SWEEP_SPARE or self.time_left() < EXPLORE_RESERVE + 40:
            return ''
        plan = self.paging(name, observation)
        if not plan:
            return ''
        count, shown, param, value = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and (not isinstance(given, bool)) else bool(str(given or '').strip())
        if value is None or later or count <= shown:
            return ''
        owners = {self.owner_of(str(record.get(cat_key) or '')) for _, record in self.records_in(observation)} - {0}
        held = {self.owner_of(str((self.rows.get(i) or {}).get(cat_key) or '')) for i in self.cart_ids()}
        if not owners or owners <= held:
            return ''
        faults = []
        call = _shape({**args, param: value}, self.schemas.get(name), 'args', faults)
        if faults:
            return ''
        reply = self.act(name, call)
        if reply is None or self.finished:
            return ''
        self.pages_read += 1
        seen = reply.get('observation')
        self.observe(name, seen)
        self.paged_obs.append(seen)
        shown_text = self.show(seen)
        self.read_cache[name + _dump(call)] = (self.version, self.step, shown_text)
        self.trace.append({'kind': 'pages', 'tool': name, 'read': 1, 'big': True})
        return f'\n[Kyo read the following page of this search as well, one step]\n{param}={_dump(value)}: {shown_text}'

    def series_sweep(self) -> str:
        tool, cat_key = (self.roles.get('search'), self.category_key())
        if self.max_steps < SWEEP_BIG or not self.bound or not tool or not cat_key or self.finished:
            return ''
        if self.left() - self.tail() < SWEEP_SPARE or self.time_left() < EXPLORE_RESERVE + 40:
            return ''
        index = self.fit_index()
        texts = []
        for part, main in self.bound:
            proven = [i for i in self.item_ids(part) if i in self.full and self.buyable(i) and (not self.sold_out(i)) and self.proves(i, part)]
            wanted = set()
            for i in proven:
                hit = re.match('(' + PART_SERIES + ')-', (index['marks'].get(i) or {}).get('own') or '')
                if hit:
                    wanted.add(hit.group(1))
            if not proven or len(proven) > SERIES_PINNED or not wanted or (main, frozenset(wanted)) in self.series_swept:
                continue

            def named(i, marks):
                return bool(set((marks.get(i) or {}).get('series') or []) & wanted) and self.buyable(i) and (not self.sold_out(i))
            if any((named(i, index['marks']) for i in self.item_ids(main))):
                continue
            self.series_swept.add((main, frozenset(wanted)))
            words = set(re.findall('[a-z0-9]{3,}', _words(self.items[main - 1].name)))
            for _, need in self.needs_now():
                words |= set(re.findall('[a-z0-9]{3,}', _words(need.split(':', 1)[-1])))
            spent, found, walked = (0, False, set())
            for past, seen in list(self.search_seen):
                query = ' '.join((str(v) for v in past.values() if isinstance(v, str))).lower()
                owners = {self.owner_of(str(r.get(cat_key) or '')) for _, r in self.records_in(seen)}
                if main not in owners or not words & set(re.findall('[a-z0-9]{3,}', query)):
                    continue
                call = dict(past)
                while spent < SERIES_PAGES and self.left() - self.tail() >= SWEEP_SPARE and (not self.finished) and (not found):
                    page = self.paging(tool, seen)
                    if not page or page[3] is None:
                        break
                    faults = []
                    call = _shape({**call, page[2]: page[3]}, self.schemas.get(tool), 'args', faults)
                    mark = _dump(call)
                    if faults or mark in walked:
                        break
                    walked.add(mark)
                    reply = self.act(tool, call)
                    spent += 1
                    if reply is None:
                        break
                    seen = reply.get('observation')
                    self.observe(tool, seen)
                    self.pages_read += 1
                    texts.append(f'{page[2]}={_dump(page[3])} of {_dump(past)[:120]}: ' + self.show(seen))
                    index = self.fit_index()
                    found = any((named(i, index['marks']) for i in self.item_ids(main)))
                if found or spent >= SERIES_PAGES:
                    break
            hits = [i for i in self.item_ids(main) if named(i, index['marks']) and i not in self.full][:2]
            read = self.fill_facts(hits, self.tail() + SWEEP_SPARE // 2) if hits else []
            self.trace.append({'kind': 'series_sweep', 'item': main, 'series': sorted(wanted), 'pages': spent, 'hits': hits})
            if texts or read:
                texts.append(f"[{self.items[main - 1].name} rows that name series {', '.join(sorted(wanted))} (the series of the only {self.items[part - 1].name} listings whose pages state every requirement): {', '.join(hits) or 'none found'}]")
                texts += read
        if not texts:
            return ''
        return '\n[Kyo paged the earlier searches for the item the parts must fit, one step a page]\n' + '\n'.join(texts)

    def scope_musts(self) -> None:
        pointed = {str(x).strip() for x in _items(self.brief.get('lookups'))}
        sentences = [x for x in re.split('(?<=[.!?;])\\s+|\\n+', self.request) if x.strip()]
        said = [[r for w in re.findall('[a-z0-9]+', x.lower()) for r in _roots(w)] for x in sentences]
        names = [_roots(item.name) for item in self.items]

        def named(words: list[str]) -> set[int]:
            held, out = (set(words), set())
            for i, mine in enumerate(names):
                if not mine or not mine <= held:
                    continue
                if any((mine < wide <= held for wide in names)) and all((words.count(r) == 1 for r in mine)):
                    continue
                out.add(i)
            return out
        musts = sorted({m for item in self.items for m in item.musts if m not in pointed and ':' in m})
        for must in musts:
            holders = [i for i, item in enumerate(self.items) if must in item.musts]
            if len(holders) < 2:
                continue
            prop, _, value = must.partition(':')
            need = _roots(prop)
            figures = set(re.findall('\\d+(?:\\.\\d+)?', value))
            keep = set()
            for x, words in zip(sentences, said):
                if not need or not need <= set(words) or not figures <= set(re.findall('\\d+(?:\\.\\d+)?', x)):
                    continue
                hit = named(words) & set(holders)
                if not hit:
                    keep = set()
                    break
                keep |= hit
            if not keep:
                continue
            for i in holders:
                if i not in keep and any((names[i] & names[k] for k in keep)):
                    self.items[i].musts.remove(must)
                    self.trace.append({'kind': 'must_scoped', 'must': must, 'from': self.items[i].name, 'to': [self.items[k].name for k in sorted(keep)]})

    def look_again(self, cheaper_named=False, spend=True):
        cat_key = self.category_key()
        if not cat_key or self.looked_again >= 2 or self.time_left() < CALL_RESERVE + 25:
            return ''
        self.looked_again += 1
        plan = self.search_template()
        cart = self.cart_ids()
        cheapest = cheaper_named or self.brief.get('pick') == 'cheapest'
        found = []
        lost = lambda pair: -sum((1 for i in cart if self.sold_out(i) and self.owns(pair[1], str((self.rows.get(i) or {}).get(cat_key) or ''))))
        for turn, (number, item) in enumerate(sorted(enumerate(self.items, 1), key=lost)):
            if len(found) >= 2:
                break
            words = ' '.join(sorted(self.live_values(number), key=len)[:2] + [item.name])
            fresh = plan is not None and spend and self.left() >= 3 + turn
            rows = self.search_words(*plan, words) if fresh else []
            held = [self.price_of(i) for i in cart if not self.sold_out(i) and self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or '')) and self.price_of(i) is not None]
            ceiling = min(held) if held else None
            pool = [i for i in self.kind_rows(number, rows or list(self.seen_order)) if i not in cart and self.price_of(i) is not None and (cheaper_named or not cheapest or ceiling is None or self.price_of(i) < ceiling)]
            pool.sort(key=lambda i: self.price_of(i) or 0.0)
            where = f'searched {words!r}' if rows or fresh else 'among listings already seen'
            if not pool:
                found.append(f'item {number} ({item.name}), {where}: no other buyable row of its kind' + (' below the carted price' if cheapest and ceiling is not None else ''))
                continue
            opened = self.fill_facts(pool[:1], 3) if spend and self.left() >= 5 else []
            found.append(f'item {number} ({item.name}), {where}: ' + ', '.join((f'{i} at {self.price_of(i):g}' for i in pool[:6])) + ('; read ' + ' | '.join(opened)[:1500] if opened else ''))
        self.trace.append({'kind': 'look_again', 'left': self.left(), 'found': len(found)})
        if not found:
            return ''
        return '\n[Kyo looked again after the change, with the requirements now in force] ' + ' || '.join(found) + ('. The note says a price has fallen, so the shopper wants the cheapest listing that still fits' if cheaper_named else '') + ('. The pick is the cheapest listing that fully fits: if one of these meets every requirement still in force, put it in the cart in place of the old line, then check out.' if cheapest else '. If the old line no longer works, put a listing that meets every requirement still in force in the cart instead, then check out.')

    def finish(self, again: bool=False) -> str:
        if self.finished or self.broke or self.left() <= 0:
            return ''
        buy, refuse = (self.roles.get('buy'), self.roles.get('refuse'))
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if buy and holding and (not self.finished) and (self.line_faults() or self.fit_faults()):
            lines = self.proven_set()
            if lines and set(lines) != set(self.cart_ids()) and self.cart_to(lines, 1):
                self.trace.append({'kind': 'finish_repair', 'lines': lines})
        if buy and (not holding) and (not self.finished) and self.left() >= 3:
            lines = self.proven_set()
            if lines and self.cart_to(lines, 1):
                holding = True
                self.trace.append({'kind': 'proven_basket', 'lines': lines})
        if buy and (holding or (self.cart_touched and (not fresh) and (self.left() >= 2) and (not self.dropped) and (self.buys == 0 or self.refused_at < self.cart_at))):
            faults = []
            args = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), 'args', faults)
            if not faults:
                if self.left() >= 2 and self.time_left() > CALL_RESERVE + 15:
                    self.gate('order')
                if self.finished or self.left() <= 0:
                    return ''
                if self.time_left() > 12:
                    self.offer_proof()
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
        if refuse and (not holding) and self.nearest_order():
            return self.finish(again)
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

    def run(self):
        if self.finished:
            return self.trace
        self.read_request()
        self.trace.append({'kind': 'brief', 'brief': self.brief})
        opening = self.view.get('opening')
        if opening is not None:
            self.absorb(opening)
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            intro += '\n\nWHAT THE STORE SHOWS FIRST:\n' + self.show(opening)
        intro += '\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n' + self.status()
        early = self.first_look()
        if early:
            intro += '\n\n' + early
        self.chat = [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': intro}]
        self.work(LOOP_CALLS, SPEND_HARD)
        if not self.finished:
            self.clock_fill()
        if not self.finished:
            turned = self.finish(again=True)
            if turned and (not self.finished) and (not self.broke) and (not self.no_credit) and (self.left() > 1) and (self.time_left() > EXPLORE_RESERVE):
                self.chat.append({'role': 'user', 'content': '[Kyo] Kyo tried to end the episode and ' + turned + '\nApply what the store says, then finish.\n' + self.status()})
                self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
            for round_no in range(3):
                if self.finished:
                    break
                again = self.finish()
                if round_no == 2 or not again or self.finished or self.broke or self.no_credit or self.left() <= 1 or self.time_left() <= EXPLORE_RESERVE:
                    break
                self.trace.append({'kind': 'end_round', 'round': round_no + 1})
                self.chat.append({'role': 'user', 'content': '[Kyo] Kyo tried to close the episode again and ' + again + '\nFollow what the store says, then finish.\n' + self.status()})
                self.work(self.loop_calls + 6, SPEND_HARD + 0.02)
        self.trace.append({'kind': 'end', 'finished': self.finished, 'steps': self.step, 'cost': round(self.cost, 5), 'calls': self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.finished and (not self.broke) and (self.left() > 0) and (self.loop_calls < loop_cap) and (self.cost < spend_cap) and (self.time_left() > EXPLORE_RESERVE):
            if self.ready_to_ask():
                summary = self.interview()
                if summary:
                    self.chat.append({'role': 'user', 'content': summary + '\n' + self.status()})
                continue
            self.trim()
            if (not self.buys) and (not self.filled) and self.left() <= len(self.items) + 3 and (not self.cart_complete()):
                before = set(self.cart_ids())
                self.clock_fill()
                if set(self.cart_ids()) != before:
                    self.trace.append({'kind': 'step_fill', 'cart': self.cart_ids()})
                    self.chat.append({'role': 'user', 'content': '[Kyo] Few steps are left and an item had no cart line, so Kyo carted the opened listing that comes closest to its requirements. Check out now: a first checkout can be refused with a change, and answering it needs a step.\n' + self.status()})
            hunted = self.chain_hunt()
            if hunted:
                self.chat.append({'role': 'user', 'content': hunted + '\n' + self.status()})
            ready = self.set_ready()
            if ready:
                self.chat.append({'role': 'user', 'content': ready + '\n' + self.status()})
            self.loop_calls += 1
            message = self.think(self.chat, [LIGHT, SPARE], 8000, self.offered())
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
