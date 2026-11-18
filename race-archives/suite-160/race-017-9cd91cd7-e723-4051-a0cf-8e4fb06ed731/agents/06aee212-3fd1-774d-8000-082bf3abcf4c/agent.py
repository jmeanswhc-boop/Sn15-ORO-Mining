from __future__ import annotations
import json
import random
import re
import signal
import time
from contextlib import suppress
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
TIME_LIMIT = 300
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3
CHEAPER_OPENS = 3
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
GUIDE = 'You are Kyo, a careful buyer working for one shopper in an online store. You can act only by calling the store\'s tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.\n\nHOW THE STORE WORKS\n- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool\'s description and schema and use the exact argument names and allowed values.\n- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.\n- Search rows are summaries. Stores split a listing\'s record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Kyo knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. A row whose title lacks a wanted value can still carry it in its attributes (the store\'s search often matched it there), so before choosing for an item or running yet another search for it, open the in-budget rows of its kind you have not read, cheapest first when the pick is the cheapest, instead of judging rows by their titles. Never open a listing again when nothing has changed since you read it.\n- Questions to the shopper are handled for you by Kyo\'s interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.\n- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Kyo carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store\'s tools (product questions go through Kyo\'s interviewer).\n\nWHAT COUNTS AS A FIT\n- The kind of product is decided by the listing\'s category label: only the label that matches the requested item as a whole counts, never a neighbouring one.\n- A requirement on a property is met when the listing\'s attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural, spacing and hyphens do not matter, and a value that contains the required value\'s words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value in other words does not count, even when it means much the same: a synonym, a related or broader term, another form of the word, or a figure for a word ("tartan" is not "checked", "micro-thin" is not "ultra-thin", "2 units" is not "twin"); nor does an abbreviation or code, and "teal green" is not "sea green". Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.\n- An attribute whose key names the requirement\'s property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing about the requirement. Kyo adds [evidence] lines after opened listings with the keys whose names match a requirement.\n- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.\n- When the category label itself states a requirement\'s value (a season or audience in the label), that requirement is met by the category.\n- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing\'s value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.\n- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.\n- When one requested item must fit another (an accessory for a device), choose the two together from the listings\' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.\n- Respect exclusions and closed lists. An exclusion (not X, anything but X) is met for certain by a listing that states its own value for that property and it is not X; a listing that says nothing about that property is only a fallback, so take a fitting listing that states an allowed value whenever you have seen one. Wishes ("ideally", "would be nice") are tie-breakers only.\n- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.\n\nCHOOSING\n- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Kyo reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.\n- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.\n- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.\n- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.\n- A request for several different items needs one cart line per item, each at its requested quantity.\n\nCHANGES WHILE YOU WORK\n- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool\'s result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.\n- Re-read a chosen listing\'s record shortly before checking out, because price and stock can change silently.\n\nFINISHING\n- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.\n- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request\'s wording for that ending, and when a remark is needed, keep it short and name no listing.\n- Every reply must call at least one tool. Keep your text short: what you know, then the next move.'
BRIEF = 'You prepare a shopping task for an agent. Read the shopper\'s request and the store\'s tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:\n\n"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request\'s words, "shelf": the store\'s category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "cap": the most one unit of this product may cost when the request sets a price limit on this product alone, apart from any total ("I won\'t pay over 650 for a lamp", "keep each charger at or under 430", "the rug can\'t cost more than 900 apiece"), as a number in the budget\'s unit, else null; such a limit belongs only to the product or store department it names, never to another requested product whose name merely shares a word with it, "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list, the value in the request\'s own words (a band the request names, such as "200-400" or "800 and up", stays as written). Name the property with the request\'s own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product. A must the request makes depend on what another requested item turns out to be ("should the X\'s colour turn out to be red, I need ...", "... except when ...") is ONE must that keeps its condition ("lid: glass, only if the chosen pot\'s material is steel")}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property\'s value or limit outright; and never turn words that are part of what the product is called (a season or an audience named with the product) into musts: those belong to "name". A use or occasion the request makes a property of the product itself ("blazers for business wear", "a dress made for parties", "the occasion has to be party") IS a must on that property ("occasion: business"), unless the store\'s category label already names it; an event, person or setting the product is bought for ("to wear at his wedding", "for a colleague"), and how it should sit, feel or look in use, stay context. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.\n"budget": the most the shopper will pay in total as a number in the request\'s own unit, or null.\n"budget_unit": that unit in words (for example "cents"), or null.\n"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.\n"in_stock_only": true unless the request clearly accepts items that are not available.\n"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.\n"pick_measure": what is maximised or minimised for "most"/"least", else "".\n"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.\n"steps": how many steps, moves or tool calls the request allows (integer), or null.\n"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].\n"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.\n"wishes": a list of things the shopper would merely like.\n"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).\n"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper\'s approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.\n"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.\n"fits": a list of {"item": item number, "with": item number} for each requested item that the request says must work with, fit or suit ANOTHER requested item (the one the agent picks); [] when none.\n"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing\'s attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing\'s current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).\n"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.\n"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.\n'
PLAN = 'You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property; a property the shopper has no view on gets "don\'t mind", and one the shopper does not think about at all gets a puzzled reply, so every property named spends part of a message.\n\nYou get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item\'s category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.\n\nChoose the properties in this order:\n0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.\n1. Hinted properties. The request\'s story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product\'s looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.\nName a hinted property by a listed key of that item when one fits the hint (a hint about who will wear it: the key for gender or audience; about its looks: the listed key for colour, pattern or finish that most of its listings carry); a property no listing of the item carries comes after the item\'s three most carried keys.\n2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item\'s listings; once five or more listings were read, avoid keys seen on only one or two of them.\nSkip a property the request already settles for that item (a requirement names that property; a key whose values only repeat a required value, such as a pattern key showing the required decoration, is NOT settled), a second name for a property already chosen, a key whose values each list several purchase options at once, identifiers, model names, warranty and other logistics. Pack sizes, counts and quantities per pack are properties like any other, and a key marked [not the stated ...] is another property than that requirement, whatever words its name shares with it. Rank every item on its own: its hinted properties first, then its listed keys.\n"label" is the property\'s plain name: for a listed key, all of the key\'s words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property\'s name, no example values, no brackets, no second property.\nChoose up to COUNT for each item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story\'s words it comes from, or an empty string"}]}, each item\'s picks in rank order'
HEAR = 'A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question or the questions taken one at a time; not answers that only say they did not follow it], "budget": a new total limit an answer states, as a number in the unit of BUDGET NOW (the number unchanged when the answer uses that unit: "2500 cents" gives 2500 when the budget is in cents), or null}\n- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3").\n- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop\'s product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value starts with plain words for what the whole answer means about that listing\'s value ("the same as", "other than", "at least", "at most", "within N of"): read it from what the answer accepts: "anything other than" or "anything but" that value accepts every value except it, so it means "other than" however firmly or oddly the answer frames it; only an answer that accepts that value alone ("nothing other than", "nothing but", "only that value") means "the same as". The rest of the value is the answer\'s own words for that listing copied exactly, to the end of its description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.\n- A value that names no other listing is the wanted value alone, without the shopper\'s own words for wanting it ("has to match navy", "should read navy", "matching navy", "pinned to navy", "holding out for navy" and "nothing else will do except navy" each give "navy"), keeping the words that set a limit or rule a value out ("at least", "at most", "not", "or"). This never applies to a value read from another listing, which is copied as the rule above says.\n- "free": the property of each question the shopper said they do not mind about.\n- Folksy or roundabout wording that names a value still states that value.'
NOTE = 'While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:\n{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the unit of the ledger\'s budget (the number unchanged when the note uses that unit: "2500 cents" gives 2500 when the budget is in cents), or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}\nA replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note.'
CHECK = 'You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.\nReading rules: a requirement is met when the listing\'s attribute for that property states the value (a key in that property\'s own words, whatever its spelling, case, spacing or plural, never a synonym or a related word; the value must be the required value itself or one element of a list or of a combined value split by commas or slashes, case, singular/plural, spacing and hyphens aside: a longer value that only contains the required words, such as "dark teal" for teal, does not count, nor does a synonym, a related term, another form of the word or a figure for a word; units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when the listing has no attribute for that property, only a figure with its unit stated in the title counts for a measured property (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a word value under another attribute, in the title or in the description does not count; a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing\'s value counts as met unless the records show that same value; a requirement that rules a value out (not X, anything but X) is met when the listing states its own value for that property, it is not X and no attribute of that property names X even inside a longer value, and a listing that says nothing about that property is a problem when another listing shown states an allowed value, meets every other requirement and is no worse on the pick rule; Requirements the shopper dropped no longer count, even where the request still states them. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget, and a line whose price is above its item\'s own limit on each unit is wrong. The newest record of a listing wins.\nWhen the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart\'s choice on that rule, the cart is wrong.\nReturn ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses.'
READING = 'the listing\'s attribute for a property decides when its key names that same property (a key in that property\'s own words, whatever its spelling, case, spacing or plural, never a synonym or a related word; the value must be the required value itself or one element of a list or of a combined value split by commas or slashes, case, plural, spacing, hyphens and unit conversions aside: a longer value that only contains the required words, such as "dark teal" for teal, does not count, nor does a synonym, a related term or another form of the word; limits include the edge); a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when no attribute names that property, only a figure with its unit stated in the title counts for a measured property (a bare figure that does not say what it measures counts for nothing); a word value under another attribute, in the title or in the description does not count; a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing\'s value counts as met unless the records show that same value; a requirement that rules a value out (not X, anything but X) is met for certain when the listing states its own value for that property, it is not X and no attribute of that property names X even inside a longer value, and a listing that says nothing about that property fits only when no listing shown states an allowed value; out-of-stock listings and listings that break the budget or their item\'s own limit on each unit do not fit.'
AUDIT = 'A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: ' + READING + '\nRecords marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.\nReturn ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "near": {"item number": [for an item with no fitting listing: the OPENED listings of its kind that fall short on the fewest requirements (stock and budget count among them), fewest first, at most 3]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}'
UNMIX = 'A shopper\'s request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request\'s wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}'
TRAIL = 'A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing\'s page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement\'s chain from the requirement\'s own description, link by link, through the pages\' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).\nReturn ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing, "find": "for a link not found: search words (its shop and the values the description gives)"}], "done": true when the last link is an OPENED page that states the property\'s value itself (also when the requirement asks for a value other than or unlike that one: the page\'s value is what it is measured against), else false, "value": the property\'s value as stated on that last page, or ""}]}\nA record is a link only when it matches the whole description of that link: the shop or brand, the kind of product AND every property value the description gives (for example "a 10 pieces multipack" rules out the same shop\'s listing with 10 capsules and no multipack); when no record matches all of it, that link is not found yet, even if the same shop sells other listings of that kind. Give "find" for such a link: the shop\'s name and the description\'s values, as a few search words.\nList each chain\'s links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown.'
POINT = 'Each numbered requirement below takes the value of one property from another listing of the store instead of stating it. Return ONE JSON object and nothing else: {"refs": [{"number": the requirement\'s number, "attribute": the property whose value is wanted, "relation": what the requirement as a whole asks for relative to that listing\'s value, read by what it accepts ("anything other than" or "anything but" that value accepts every other value, so "different" however it is framed; only "nothing other than", "nothing but" or "only" that value means "same"): "same" (the same value, or one that matches, goes with or is compatible with it), "different" (anything but that value), "at_least", "at_most", or "within" (differing from it by at most a margin), "margin": that margin as a number for "within", else null, "unit": the margin\'s unit, else "", "listing": {"id": the listing id or part code the requirement gives for that listing, else "", "seller": the shop or seller named for that listing (a name before its product or in the possessive), "product": that listing\'s kind of product alone, "facts": {"property": "value" for every detail the requirement gives about that listing, such as the ones after "with" or in brackets, each value copied with all its words}}}]}'
REFER = 'A shopping agent follows a requirement that takes the value of one PROPERTY from one particular listing of the store: the WANTED listing, given by its seller, its kind of product and some facts about it. The facts only tell which listing is meant; the property is the one whose value is wanted. The input holds OPENED listings with all their data, including their free text, and may hold CANDIDATE rows: search results whose page has not been read yet.\nReturn ONE JSON object and nothing else: {"match": null, "exact": false, "redirect": null, "value": [], "pairs_with": [], "check": [], "open": []}\n- "match": the id of the opened listing that is the wanted one: from that seller, of that kind of product (a listing of another seller or another kind of product is never the match, whatever its notes say), and its attributes or its own notes (a short remark about this very listing, apart from its title and sales description) state every given fact in its own words (at most in another unit, spelling or word order; a different term for a similar thing is another value). The title and the sales description are loose seller text: they never prove a fact and never rule a listing out, even when they seem to name another value. A listing whose attributes or notes give another value for a given fact, or say it is not the described one, is not it; nor is one whose attributes and notes leave a given fact unstated or state only part of it (it goes under "check"). When several opened listings fit, the one that states the facts most exactly. When the wanted listing gives an id, the opened listing with that id is the wanted one. null when none fits.\n- "exact": true when the matched listing\'s data states every given fact as given (the same value, at most in another unit or spelling); false when a fact is met only by a near, broader or related value.\n- "redirect": when the matched listing\'s text says to take this very property from another listing instead of itself: {"id": the listing id or part code the text gives (copied exactly), else "", "seller": the shop or maker named for it (a name before the product or in the possessive), "product": its kind of product, "facts": {"property": "value" for each detail the text gives about it}}. A pointer for any other property does not count: then "redirect" is null and the matched listing\'s own data gives the value.\n- "value": the matched listing\'s own value of exactly this property, as its data writes it (an attribute, or its name or free text when they state it); never the value of another property; [] when it states none.\n- "pairs_with": only when the matched listing\'s notes state a rule or list that pairs values of this very property (one value going with, matching or combined with another; a list may name the link once and then give only the pairs): for each pair that holds the value above, its other value. Never a value of another property and never words of the title or description that pair nothing; else [].\n- "check": only when nothing matches: the opened listings that may still be the wanted one because their data leaves a given fact unstated, most likely first.\n- "open": only when nothing matches: every candidate row of that seller and that kind of product, most likely first (a row whose words state more of the given facts, in any wording, short form or language, goes earlier). A row whose title or description seems to name another value may still be the wanted listing, so keep it, after the likelier rows. Leave out only rows of another seller or another kind of product.'
GATE = 'A shopping agent is about to STAGE. The shopper\'s request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product\'s properties do not do a step.\nWhen a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.\nReturn ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}'
SWAP = "A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: " + READING + '\nReturn ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line\'s item, at most 6], "reason": "one sentence"}'
MAP = 'A tool of an online store returned the text below to a shopping agent. It holds product records written as text, not as JSON objects. Say how the agent\'s code can read this text and every later result of the same tool. Return ONE JSON object and nothing else: {"records": the exact characters that separate two records inside one text, or "" when the text is one record, "split": the exact characters between two fields of a record, "pair": the exact characters between a field\'s name and its value, "nested": {"<field>": {"split": "...", "pair": "..."}} for each field whose value holds its own names and values, "id": the field holding the product id (the code the store\'s other tools take), "price": the field holding the price, "stock": the field saying whether it can be bought now, "title": the title field, "category": the category field, "seller": the seller field, "facts": the field holding the product\'s attributes (properties), "" for a part the records do not have}. Copy every separator exactly, spaces included. REJECTED, when given, is an earlier answer that did not read the text, with the reason.'
READ = 'A store\'s tools returned the numbered texts below. Some describe listings (products on sale), one or several in a text; others are messages, notes or errors. Turn every listing described into one JSON record, copying names and values exactly as the text writes them. Return one JSON object per line, one line per listing, and nothing else (no list around them, no code fence): {"from": the number of the text it comes from, "id": the listing\'s id or code exactly as written (the value the store\'s tools take), "title": its name or title, "price": its price as a number as written (no unit change), "stock": true when the text says it can be bought now, false when it says it cannot, the number of units when it gives one, "category": its category label word for word, "seller": the shop or seller that sells it, "notes": the shop\'s own comments on it word for word (not its general product description), "attributes": {"property": "value"} for every attribute, property or specification the text lists for it, each key and value copied exactly}\nLeave out every field the text does not give, and empty ones (no null, "" or {}). A text that describes no listing gives no record.'
SHAPE = 'A store tool returned the result below. Name its parts by the exact keys the result and the tool\'s parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing\'s price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}'
WORDING = 'A shopping agent searches a store for listings that state given values. The store\'s keyword search matches the words of a query against the words of its listings, so a value finds its listings only when the query writes it the way those listings do. For each numbered product below you get the values its listing must state (NEW: the ones to search for now) and what this store\'s listings of that product, seen so far, say.\nWrite short keyword queries for each product, the most promising first:\n- Every query holds a NEW value. Put the values first; add the product\'s plain noun only when the values alone would also match many other kinds of product.\n- Write each value the way these listings write it: their spelling, a compound joined, split or hyphenated as they show it, a number with its unit as they print it.\n- When the listings seen do not show how this store writes a value, add queries with the other ways listings commonly write it: another spelling or a common misspelling, joined, split or hyphenated words, singular or plural.\n- Never swap a value for a synonym, a broader word or another value.\nReturn ONE JSON object: {"queries": {"<product number>": ["query", ...]}}, with no more queries for a product than its line allows.'
GUARD_KEYS = 3
GUARD_SHARE = 0.0
GUARD_GAP = 0.34
NOT_PROPERTY = '\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b'
CATCH_ALL = '^(?!.*\\b(?:types?|styles?|kinds?|shapes?)\\b).*\\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\\w*|accessor\\w*|extras?|misc\\w*)\\b'
WIDEN_EACH = 6
ROW_LIMIT = 30
FIT_CODE = '[A-Z][A-Za-z]{0,5}[ -]?(?:\\d{3,6}|\\d{1,5}[xX]{1,4})'
PART_CODE = '[A-Z]{1,4}\\d{0,3}-?\\d{3,6}'
SERIES_CODE = '[A-Z]{1,4}\\d{1,2}'

def _codes(text: Any) -> list[str]:
    return [re.sub('\\s+', ' ', c).strip().lower() for c in re.findall('\\b' + FIT_CODE + '\\b', str(text or ''))]

def _listed(text: Any) -> list[str]:
    return _codes(text) + [c.lower() for c in re.findall('\\b' + SERIES_CODE + '\\b', str(text or ''))]

def _code_key(code: str) -> str:
    return re.sub('[\\s-]+', '', code).upper()

def _fits_code(pattern: str, code: str) -> bool:
    if re.fullmatch(SERIES_CODE, pattern.strip(), re.I):
        return bool(re.match(re.escape(pattern.strip()) + '[\\s-]\\d', code.strip(), re.I))
    pattern, code = (re.sub('[\\s-]+', '', pattern).lower(), re.sub('[\\s-]+', '', code).lower())
    stem = pattern.rstrip('x')
    return pattern == code or (stem != pattern and len(code) == len(pattern) and code.startswith(stem))

def _fits_any(fits: set[str], codes: set[str]) -> bool:
    return any((_fits_code(f, c) for f in fits for c in codes))

def _fit_marks(text: Any) -> dict[str, Any]:
    text = str(text or '')
    subject = '(' + PART_CODE + '|' + FIT_CODE + ')\\s+also\\s+'
    own = re.search('[\\[(]\\s*(?:part\\s+)?(' + PART_CODE + ')\\s*[\\])]', text, re.I)
    codes: list[str] = []
    for hit in re.finditer('\\bmodel\\s*(?:no\\.?|number|#)?\\s*[:.]?\\s*(' + FIT_CODE + ')\\b(?!\\s+also\\b)', text, re.I):
        codes += _codes(hit.group(1))
    for hit in re.finditer('\\b(?:also sold as|also known as|sold also as)\\s+([^.;]+)', text, re.I):
        codes += _codes(hit.group(1))
    works: list[str] = []
    for hit in re.finditer('\\b(?<!not )(?<!never )(?:works with|compatible with|fits)\\s*:?\\s+([^.;]+)', text, re.I):
        if not re.search('\\b' + subject + '$', text[max(0, hit.start() - 40):hit.start()], re.I):
            works += _listed(hit.group(1))
    also = [(hit.group(1).upper(), _listed(hit.group(2))) for hit in re.finditer('\\b' + subject + '[a-z][a-z\\s]{0,30}?:?\\s+([^.;]+)', text, re.I)]
    more = re.search('\\b(?:more|details|see|read)\\s+(?:on|at|under|about)?\\s*(?:the\\s+)?part\\s+(' + PART_CODE + ')', text, re.I)
    return {'own': own.group(1).upper() if own else '', 'codes': codes, 'works': works, 'also': also, 'more': more.group(1).upper() if more else ''}

def _gives(out: dict[str, Any]) -> bool:
    return bool(out.get('redirect') or _items(out.get('value')) or _items(out.get('pairs_with')))

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

def _flat(text: str) -> str:
    return re.sub('[^a-z0-9.]+', '', str(text).lower())

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
    if kind in ('integer', 'number'):
        return isinstance(value, int if kind == 'integer' else (int, float)) and (not isinstance(value, bool))
    return value is None if kind == 'null' else isinstance(value, {'string': str, 'boolean': bool, 'array': list, 'object': dict}.get(kind, object))

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

def _amount(value: Any) -> float | int | None:
    if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', value):
        value = float(value)
    return value if isinstance(value, (int, float)) and (not isinstance(value, bool)) and (value > 0) else None

def _count(value: Any) -> int | None:
    if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.0+)?\\s*', value):
        return int(float(value))
    return None if isinstance(value, bool) else value if isinstance(value, int) else int(value) if isinstance(value, float) and value.is_integer() else None

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

def _tokens(text: str) -> set[str]:
    return set(re.findall('[a-z0-9]+(?:\\.\\d+)?', text.lower()))

def _leaves(node: Any, depth: int=0) -> str:
    if depth > 6:
        return ''
    if isinstance(node, dict):
        return ' '.join((_leaves(v, depth + 1) for v in node.values()))
    if isinstance(node, list):
        return ' '.join((_leaves(v, depth + 1) for v in node))
    return '' if node is None or isinstance(node, bool) else str(node)

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

def _texts(node: Any, depth: int=0) -> list[str]:
    if isinstance(node, str):
        return [node] if node.strip() else []
    if depth > 6 or not isinstance(node, (dict, list)) or (isinstance(node, dict) and sum((isinstance(v, str) for v in node.values())) > 1):
        return []
    return [t for v in (node.values() if isinstance(node, dict) else node) for t in _texts(v, depth + 1)]

def _lay(text: str, how: dict[str, Any]) -> list[dict[str, Any]]:
    split, pair, parts = (str(how.get('split') or ''), str(how.get('pair') or ''), ('id', 'title', 'price', 'stock', 'category', 'seller'))
    out = []
    for record in (text.split(str(how['records'])) if how.get('records') else [text]) if split and pair else []:
        row, key = ({}, None)
        for piece in record.split(split):
            name, has, value = piece.partition(pair)
            if has and name.strip() and name.strip() not in row:
                key, row[name.strip()] = (name.strip(), value.strip())
            elif key:
                row[key] += split + piece
        for name, sub in (how.get('nested') or {}).items():
            if isinstance(row.get(name), str) and isinstance(sub, dict) and sub.get('split') and sub.get('pair'):
                row[name] = {a.strip(): b.strip() for a, has, b in (x.partition(sub['pair']) for x in row[name].split(sub['split'])) if has and a.strip()}
        named = {k: row.pop(str(how.get(k) or ''), None) for k in parts + ('facts',)}
        rec = {k: v for k, v in named.items() if k in parts and v not in (None, '')} | {'notes': row} if row else {k: v for k, v in named.items() if k in parts and v not in (None, '')}
        rec.update({'price': _amount(rec.get('price')), 'stock': {'true': True, 'false': False}.get(str(rec.get('stock')).lower(), rec.get('stock'))} if 'price' in rec else {})
        rec.update({'attributes': named['facts']} if isinstance(named['facts'], dict) else {})
        out.append(rec)
    return out if out and all((_id_like(r.get('id')) and isinstance(r.get('price'), (int, float)) for r in out)) else []

class Item:

    def __init__(self, name: str, shelf: str, qty: int, musts: list[str], cap: float | int | None=None):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.cap = cap
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
        self.finished = bool(self.view.get('finished'))
        self.step = self.waited = self.misses = self.calls = self.loop_calls = self.buys = self.version = 0
        self.msg_limit = self.msg_used = self.reserve = self.gate_msgs = self.held = self.refuse_blocks = 0
        self.trails = self.trail_blocks = self.compare_ok = self.compare_bad = self.pages_read = self.line_holds = self.fit_holds = 0
        self.broke = self.no_credit = self.asked = self.cart_touched = self.note_named = self.audited = self.note_sold = self.pushed = self.wider_held = self.chase_broke = False
        self.cart_at = self.cart_seen_at = self.refused_at = self.chased_at = -1
        self.cost = self.turn_cost = 0.0
        self.trace, self.brief, self.roles, self.items, self.budget, self.process = ([], {}, {}, [], None, [])
        self.gate_ok, self.done_log, self.done_steps, self.qa, self.rows, self.full = (set(), [], [], [], {}, {})
        self.seen_order, self.heard, self.notes, self.cart_view, self.cart_lines, self.checked = ([], set(), [], None, {}, set())
        self.offer_seen, self.gone, self.unavailable, self.sold_lines, self.id_args, self.cart_said = ({}, {}, set(), set(), set(), '')
        self.leads, self.built, self.lead_items, self.near_items, self.price_key, self.stock_key = ([], [], {}, {}, None, None)
        self.verdicts, self.loop_cap, self.spend_cap, self.id_key, self.facts_from, self.queued = ({}, LOOP_CALLS, SPEND_HARD, None, {}, [])
        self.refused, self.recent, self.chat, self.opened, self.refs, self.ref_values = ({}, [], [], set(), [], {})
        self.value_done, self.tool_keys, self.view_keys, self.read_cache, self.shapes = (set(), {}, {}, {}, {})
        self.cheaper_marks, self.paged_obs, self.plan_order, self.bound, self.pointed, self.kept_held = (set(), [], [], [], set(), set())
        self.scanned_caps, self.searched, self.chains, self.not_pages, self.found, self.swept = (set(), [], {}, set(), {}, set())
        self.pending, self.texts, self.json_reads, self.maps = ([], {}, False, {})
        self.started, self.slowest, self.halted, self.closing, self.last_cart, self.listing = (time.monotonic(), 0.0, False, False, {}, True)

    def left(self) -> int:
        return self.max_steps - self.step

    def msgs_free(self) -> int:
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)

    def cart_mark(self) -> str:
        return _dump(sorted(self.cart_lines.items()))

    def page_tools(self) -> set[str]:
        return {t for t in (self.page_tool(), None if self.compare_pages() is False else self.roles.get('compare')) if t}

    def page_tool(self) -> str | None:
        if self.roles.get('offer'):
            return self.roles['offer']
        jobs = {self.roles.get(r) for r in ('search', 'open', 'compare', 'ask', 'cart', 'cart_view', 'buy', 'refuse')}
        other = [t for t in self.schemas if t not in jobs and t not in self.not_pages and self.single_target(t)]
        return other[0] if len(other) == 1 else self.roles.get('open')

    def compare_pages(self) -> bool | None:
        page, joiner = (self.page_tool(), self.roles.get('compare'))
        if not page or not joiner or page not in self.tool_keys or (joiner not in self.tool_keys):
            return None
        return self.tool_keys[joiner] >= self.tool_keys[page]

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
        return 2 if any((p['when'] == 'order' and p['messages'] > 0 for p in self.process)) and self.msg_used < self.msg_limit and self.gate_due('order') else 1

    def judges(self, limit: float=SPEND_SOFT) -> list[str]:
        if self.cost <= limit and self.cost + HEAVY_ROOM <= SPEND_HARD:
            return [HEAVY, LIGHT]
        return [LIGHT] if self.cost <= SPEND_HARD else []

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None=None) -> dict[str, Any] | None:
        for model in models:
            if self.calls >= ALL_CALLS - (0 if self.closing else self.spare()):
                return None
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
            paid = usage.get('cost') or (usage.get('cost_details') or {}).get('upstream_inference_cost')
            if isinstance(paid, (int, float)):
                self.cost += float(paid)
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

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
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
            self.items.append(Item(str(entry.get('name') or 'the product'), str(entry.get('shelf') or ''), count, musts, _amount(entry.get('cap'))))
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
                twin.cap = min((c for c in (twin.cap, item.cap) if c is not None), default=None)
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
        self.budget = _amount(brief.get('budget'))
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
            talk = [p['messages'] for p in self.process if p['when'] == 'order' and p['messages']]
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
            if item.cap is not None:
                head += f", each unit at most {item.cap:g} {self.brief.get('budget_unit') or ''}".rstrip()
            text = head + '; requirements: ' + ('; '.join(needs) if needs else 'none beyond the kind of product')
            if told:
                text += "; described in the shopper's words, which no listing attribute states (decides nothing): " + '; '.join(told)
            if item.dropped:
                text += '; dropped by the shopper, no longer required: ' + '; '.join(dict.fromkeys(item.dropped))
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
        short = self.uncarted() if self.buys == 0 else []
        if short and self.turns_left() <= len(short) + 2:
            text += ' Kyo can pay for only a few more turns and never ends without a purchase: put the best listing you have read for ' + ', '.join((self.items[n - 1].name for n in short)) + ' in the cart now (the nearest one where none meets every requirement; one line per item at its quantity) and check out.'
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values]
        later = set(self.chase_new()) if not self.chase_broke and (not self.talked()) else set()
        if later and self.buys == 0:
            text += "\n[Kyo itself opens the pages these point at once the shopper has been asked; until then search and open the requested items] " + ' | '.join([need for need in lookups if need in later][:3])
        lookups = [need for need in lookups if need not in later]
        if lookups and self.buys == 0:
            text += '\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] ' + ' | '.join(lookups[:3])
        if self.leads:
            text += '\n[review candidates] ' + ', '.join(self.leads)
        plan = self.fit_plan() if self.bound and self.buys == 0 else ''
        if plan:
            text += '\n' + plan
        if self.notes:
            text += '\n[changes so far] ' + ' | '.join(self.notes[-4:])
        text += '\n[ledger]\n' + self.ledger()
        return text

    def turns_left(self) -> int:
        spend = int(max(0.0, self.spend_cap - self.cost) / self.turn_cost) if self.turn_cost > 0 else ALL_CALLS
        return min(self.loop_cap - self.loop_calls, ALL_CALLS - self.calls, spend, self.clock())

    def clock(self) -> int:
        return int((TIME_LIMIT + self.started - time.monotonic()) / self.slowest) - 1 if self.slowest else ALL_CALLS

    def spare(self) -> int:
        return 1 if self.roles.get('cart') and self.uncarted() else 0

    def owners(self) -> dict[str, int]:
        cat_key = self.category_key()
        if not cat_key:
            return {i: 1 if len(self.items) == 1 else 0 for i in self.rows}
        labels: dict[str, int] = {}
        for ident, row in self.rows.items():
            label = str(row.get(cat_key) or '')
            if label not in labels:
                labels[label] = self.owner_of(label)
        return {i: labels[str(row.get(cat_key) or '')] for i, row in self.rows.items()}

    def fill_room(self, slack: int=2) -> int:
        if self.buys or not self.roles.get('cart'):
            return self.tail()
        short = len(self.uncarted())
        props = (self.schemas.get(self.roles.get('cart') or '') or {}).get('properties') or {}
        swap = 1 if any((isinstance(spec, dict) and 'array' in _kinds(spec) for spec in props.values())) else 2
        return self.tail() + (min(1, short) if swap == 1 else short) + swap + slack

    def uncarted(self) -> list[int]:
        owner = self.owners()
        held = [i for i in self.cart_ids() if not self.sold_out(i)]
        if held and (not any(owner.values())):
            return []
        return [n for n in range(1, len(self.items) + 1) if n not in {owner.get(i, 0) for i in held}]

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

    def absorb(self, observation: Any, name: str='') -> bool:
        found: list[str] = []

        def walk(node: Any, depth: int, in_list: bool=False) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                ident = self.ident(node, in_list, depth)
                if ident:
                    found.append(ident)
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
        fixed = name in {self.roles.get(r) for r in ('ask', 'cart', 'cart_view', 'buy', 'refuse')}
        self.json_reads = self.json_reads or bool(found and name and (not fixed))
        if found or not name or fixed or self.json_reads:
            return False
        units = [t for t in _texts(observation) if len(t.split()) >= 5]
        self.pending += [(name, t) for t in units if t not in self.texts and (name, t) not in self.pending]
        return bool(units)

    def read_texts(self) -> None:
        batch = {t: tool for tool, t in self.pending if not self.json_reads}
        self.pending = []
        for text, tool in list(batch.items()):
            rows = self.mapped(tool, text)
            if rows:
                self.texts[text] = [r['id'] for r in rows]
                self.price_key, self.stock_key = (self.price_key or 'price', self.stock_key or 'stock')
                self.absorb({'records': rows})
                del batch[text]
        chunks: list[list[str]] = []
        for text in batch:
            if not chunks or sum(map(len, chunks[-1])) + len(text) > OBS_LIMIT:
                chunks.append([])
            chunks[-1].append(text)
        for texts in chunks:
            try:
                self.read_chunk(texts)
            except Exception as exc:
                self.trace.append({'kind': 'crash', 'where': 'read_texts', 'error': f'{type(exc).__name__}: {exc}'[:400]})
        notes = [t for t in batch if not self.texts.get(t) and t not in self.heard]
        self.texts.update({t: self.texts.get(t) or [] for t in batch})
        meaning = self.take_note(notes) if notes else ''
        if meaning:
            self.queued.append(f'(seen on {batch[notes[0]]}) {meaning}')

    def mapped(self, tool: str, text: str) -> list[dict[str, Any]]:
        if tool in self.maps:
            return _lay(text, self.maps[tool]) if self.maps[tool] else []
        why = ''
        for _ in range(2):
            how = self.ask_json(MAP + '\n\nTOOL: ' + tool + '\n\nTEXT:\n' + text[:4000] + (f'\n\nREJECTED: {why}' if why else ''), self.judges()[-1:], 1500) or {}
            rows = _lay(text, how) if how else []
            if rows or not how:
                break
            why = _dump(how)[:600] + ' read no record with an id and a number price from this text'
        self.maps[tool] = how if rows else {}
        return rows

    def read_chunk(self, texts: list[str]) -> None:
        message = self.think([{'role': 'user', 'content': READ + '\n\nTEXTS:\n' + '\n\n'.join((f'TEXT {n}:\n{t}' for n, t in enumerate(texts, 1)))}], [LIGHT], 16000) if self.judges() else None
        said = str((message or {}).get('content') or '')
        found = [got for got in map(_json_obj, said.splitlines()) if got] or [_json_obj(said) or {}]
        records = [{k: v.strip() if isinstance(v, str) else v for k, v in r.items() if v not in (None, '', {}, [])} for got in found for r in (_items(got.get('records')) if 'records' in got else [got]) if isinstance(r, dict) and _id_like(r.get('id'), True)]
        places = [_count(r.pop('from', None)) or 0 for r in records]
        merged: dict[str, dict[str, Any]] = {}
        for n, record in zip(places, records):
            merged.setdefault(record['id'], {}).update(record, **({'text': texts[n - 1]} if places.count(n) == 1 and 0 < n <= len(texts) else {}))
        self.texts.update({t: [r['id'] for r, n in zip(records, places) if n == i] for i, t in enumerate(texts, 1)})
        self.trace.append({'kind': 'read_texts', 'texts': len(texts), 'records': len(merged), 'facts': sum((1 for r in merged.values() if _fact_block(r)))})
        if merged:
            self.price_key, self.stock_key = (self.price_key or 'price', self.stock_key or 'stock')
            self.absorb({'records': list(merged.values())})

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

    def kind_key(self) -> str | None:
        key = self.category_key()
        values: dict[str, set[str]] = {}
        for row in self.rows.values():
            for k, v in row.items():
                if isinstance(v, str) and 0 < len(v.strip()) <= 40:
                    values.setdefault(k, set()).add(v.strip())
        owned = {k: sum((1 for v in vs if self.owner_of(v))) for k, vs in values.items()}
        if not key or owned.get(key):
            return key
        best = max(values, key=lambda k: (owned[k] / len(values[k]), -len(values[k])), default=None)
        return best if best and owned.get(best) else key

    def kinds(self) -> dict[str, int]:
        key = self.kind_key()
        if not key:
            return {i: 1 if len(self.items) == 1 else 0 for i in self.rows}
        labels: dict[str, int] = {}
        out = {}
        for ident, row in self.rows.items():
            label = str(row.get(key) or '')
            if label not in labels:
                labels[label] = self.owner_of(label)
            out[ident] = labels[label]
        return out

    def price_rank(self, ident: str) -> float:
        price = self.price_of(ident)
        return price if price is not None else float('inf')

    def priced(self, ids: list[str]) -> str:
        return ', '.join((i if self.price_of(i) is None else f'{i} ({self.price_of(i):g})' for i in ids))

    def buyable(self, ident: str) -> bool:
        if self.sold_out(ident):
            return False
        value = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        return not (value is False or (isinstance(value, (int, float)) and (not isinstance(value, bool)) and (value <= 0)))

    def records_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        ident = self.ident(observation) if isinstance(observation, dict) else None
        if ident:
            return [(ident, observation)]
        lists = [observation] if isinstance(observation, list) else [w for v in observation.values() for w in ([v] if isinstance(v, list) else [x for x in v.values() if isinstance(x, list)] if isinstance(v, dict) else [])] if isinstance(observation, dict) else []
        found = [(i, r) for rows in lists for r in rows if isinstance(r, dict) for i in [self.ident(r, True)] if i]
        if found or not self.texts:
            return found
        return [(i, self.rows[i]) for i in dict.fromkeys((i for t in _texts(observation) for i in self.texts.get(t, []))) if i in self.rows]

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

    def free_steps(self) -> int:
        return self.left() - self.fill_room() - (0 if self.asked else max(0, self.msgs_free())) - 2 * len(self.items)

    def window(self, name: str, shown: int, value: Any) -> int | None:
        shape = self.shapes.get(name) or {}
        spec = ((self.schemas.get(name) or {}).get('properties') or {}).get(shape.get('next_arg') or '')
        top = spec.get('maximum') if isinstance(spec, dict) else None
        if shape.get('window') or type(top) is not int or type(value) is not int:
            return shape.get('window')
        return top + shown if value >= shown > 2 else (top + 2 - value) * shown

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        self.paged_obs = []
        pick = self.brief.get('pick')
        plan = self.paging(name, observation) if pick in ('cheapest', 'most', 'least') else None
        given = args.get(plan[2]) if plan else None
        later = given not in (None, 0, 1) if isinstance(given, int) and (not isinstance(given, bool)) else bool(str(given or '').strip())
        if not plan or plan[0] <= plan[1] or (later and (not isinstance(plan[3], int) or plan[3] < plan[1])) or (pick == 'cheapest' and self.price_sorted(observation)):
            return ''
        count, shown, param, value = plan
        rows = value if later else shown
        top = self.window(name, shown, value)
        rest = -(-(min(count, top or count) - rows) // shown)
        room = self.free_steps() // (1 + len(self.uncarted()))
        whole = rest <= room and (later or not top or count <= top)
        pages = rest if whole else 0 if later else min(rest, room // 8, self.max_steps // 8 - self.pages_read)
        if rest <= 0 or (later and pages <= 0):
            return ''
        texts = []
        while value is not None and len(texts) < pages:
            faults: list[str] = []
            call = _shape({**args, param: value}, self.schemas.get(name), 'args', faults)
            reply = None if faults else self.act(name, call)
            if reply is None or self.finished:
                break
            seen = reply.get('observation')
            self.observe(name, seen)
            self.paged_obs.append(seen)
            self.read_cache[name + _dump(call)] = (self.version, self.step, _dump(_clip(seen, SHOW_CHARS // 3))[:OBS_LIMIT])
            texts.append(f'{param}={_dump(value)}: ' + self.read_cache[name + _dump(call)][2])
            rows += len(self.records_in(seen))
            self.pages_read += 0 if whole else 1
            value = (self.paging(name, seen) or [None] * 4)[3]
        if value is None and rows < count:
            self.shapes[name]['window'] = top = rows
        self.trace.append({'kind': 'pages', 'tool': name, 'read': len(texts), 'rows': rows, 'count': count})
        note = f'the store shows only the first {top} rows of a result' if top and top < count and (value is None or rows >= top) else f'{rows} of them have been shown; the rest would take {rest - len(texts)} more steps' if value is not None else ''
        note = f"\n[Kyo] This search matched {count} listings and {note}, in no price order. The pick counts every fitting listing: narrow the search with more words (the requirement's value, a word the fitting titles share){', or lower its maximum price,' if self.cap_keys(name, False) else ''} until all its matches can be read" + (f' or page on with {param}={_dump(value)}.' if value is not None else '.') if note else ''
        return ('\n[Kyo read the next pages of this search itself, one step a page]\n' + '\n'.join(texts) if texts else '') + note

    def cap_keys(self, name: str, priced: bool=True) -> list[str]:
        props = (self.schemas.get(name) or {}).get('properties') or {}
        return [k for k, v in props.items() if isinstance(v, dict) and ('integer' in _kinds(v) or 'number' in _kinds(v)) and re.search('max|most|under|below|limit|cap|upto|up_to', k, re.I) and (not priced or re.search('price|cost|amount|budget|spend|money', k + ' ' + str(v.get('description') or ''), re.I))]

    def cap_ignored(self, name: str, args: dict[str, Any], observation: Any) -> str:
        props = (self.schemas.get(name) or {}).get('properties') or {}
        caps = sorted((k for k in self.cap_keys(name) if type(args.get(k)) in (int, float)), key=lambda k: args[k])
        over = [i for i, _ in self.records_in(observation) if caps and self.price_key and (self.price_of(i) or 0) > args[caps[0]]]
        if not over:
            return ''
        others = {k: [o for o in props[k]['enum'] if o != v] for k, v in args.items() if isinstance(v, str) and isinstance((props.get(k) or {}).get('enum'), list)}
        hint = '; list the category with ' + ', '.join((f'{k}={_dump(o[0])}' for k, o in others.items() if o)) + f' and the same {caps[0]} to see only cheaper listings' if any(others.values()) else ''
        return f'\n[Kyo] This search did not apply {caps[0]}={args[caps[0]]:g}: {len(over)} of its rows cost more{hint}.'

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
        return f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest. Searches match a listing's title, so a requirement's value (often only in the attributes) finds few of the fitting listings, and rows come in no price order. Before checking out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line" + (f" ({'; '.join(lines)})" if lines else '') + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        tool = self.facts_tool()
        target = self.single_target(tool)
        texts = []
        for ident in [i for i in ids if target and i in self.rows and i not in self.full and (not self.sold_out(i))]:
            if self.left() <= keep or self.finished or self.facts_from.get(tool) is False or (self.texts and (not self.judges() or self.calls >= ALL_CALLS)):
                break
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
            reply = None if faults else self.act(tool, args)
            if reply is None:
                break
            seen = reply.get('observation')
            self.observe(tool, seen, True)
            self.note_view(tool, seen)
            if isinstance(seen, dict) and self.ident(seen):
                self.facts_from[tool] = self.facts_from.get(tool, False) or bool(_fact_block(seen))
            texts.append(f'{ident}: ' + self.show(seen)[:1500])
        self.read_texts()
        self.trace += [{'kind': 'fill_facts', 'n': len(texts)}] if texts else []
        return texts

    def ranked(self, ids: list[str]) -> list[str]:
        cheapest = self.brief.get('pick') == 'cheapest'
        return sorted(ids, key=lambda i: (-self.found.get(i, (0, 0))[0], self.price_rank(i) if cheapest else -self.found.get(i, (0, 0))[1]))

    def proven(self, ident: str, number: int) -> bool:
        return ident in self.full and all((self.meets(ident, m) > 0 for k, m in self.needs_now() if k == number and m not in {need for _, need in self.refs}))

    def lane(self, number: int, below: float) -> list[str]:
        kind = self.kinds()
        pool = [i for i in self.found if i not in self.full and kind.get(i) == number and self.buyable(i) and self.price_rank(i) < below]
        limits = [x for x in (self.items[number - 1].cap, self.budget) if isinstance(x, (int, float)) and any((self.price_rank(i) <= x for i in pool))]
        return self.ranked([i for i in pool if all((self.price_rank(i) <= x for x in limits))])[:max(1, self.max_steps // (2 * WIDEN_EACH * len(self.items)))]

    def sweep(self, plan: list[tuple[int, float]], room: int) -> list[str]:
        lanes = [self.lane(n, below) for n, below in plan]
        self.swept.update(((n, self.version) for n, _ in plan))
        return self.fill_facts([lane[k] for k in range(max(map(len, lanes), default=0)) for lane in lanes if k < len(lane)][:max(0, room)], self.fill_room())

    def sweep_due(self, now: bool=False) -> list[tuple[int, float]]:
        kind = self.kinds()
        proven = {n: [self.price_rank(i) for i in self.full if kind.get(i) == n and self.buyable(i) and self.proven(i, n)] for n in ([] if self.buys or self.finished else self.uncarted())}
        plan = [(n, min(got, default=float('inf'))) for n, got in proven.items() if (n, self.version) not in self.swept and (self.brief.get('pick') == 'cheapest' or not got)]
        lanes = [self.lane(n, below) for n, below in plan]
        return plan if any(lanes) and (now or self.free_steps() <= sum(map(len, lanes))) else []

    def cart_sweep(self, args: dict[str, Any]) -> str:
        ident = next((v for v in args.values() if isinstance(v, str) and v in self.rows and v not in self.cart_lines), '')
        number = self.kinds().get(ident, 0)
        cheapest = self.brief.get('pick') == 'cheapest'
        if not number or number not in self.uncarted():
            return ''
        read = self.fill_facts([ident], self.fill_room(0))
        if (number, self.version) not in self.swept and (cheapest or not self.proven(ident, number)):
            read += self.sweep([(number, self.price_rank(ident) if cheapest else float('inf'))], self.free_steps())
        return "\n[Kyo first read this listing's attributes and the rows of its kind that searches showed but no one had opened, one step each; if one of the others meets every requirement and this one does not" + (', or fits for less' if cheapest else '') + ', swap it in]\n' + '\n'.join(read) if read else ''

    def capped_call(self) -> tuple[str, dict[str, Any], str, str] | None:
        labels = {str(r.get(k)) for k in {self.category_key(), self.kind_key()} - {None} for r in self.rows.values() if isinstance(r.get(k), str)}
        for entry in reversed(self.trace):
            for call in entry.get('calls') or [] if entry.get('kind') == 'turn' else []:
                name = str(call.get('name') or '')
                args = _json_obj(call.get('arguments')) or {}
                page = (self.shapes.get(name) or {}).get('next_arg')
                caps = [k for k in self.cap_keys(name, False) if k in args and k != page]
                cats = [k for k, v in args.items() if isinstance(v, str) and v.strip() in labels]
                if caps and cats:
                    return (name, {k: v for k, v in args.items() if k not in (caps[0], page)}, cats[0], caps[0])
        return None

    def price_scan(self, line: str) -> None:
        learned = self.capped_call()
        price = self.price_of(line)
        key = next((k for k in (self.category_key(), self.kind_key()) if k and isinstance((self.rows.get(line) or {}).get(k), str)), None)
        if not learned or price is None or not key:
            return
        tool, base, cat_arg, cap_arg = learned
        cap = price - 1 if float(price).is_integer() else price * 0.999
        room = self.free_steps() - len(self.cart_ids())
        while room > 0 and cap > 0 and (tool, line, cap) not in self.scanned_caps and not self.finished:
            self.scanned_caps.add((tool, line, cap))
            faults: list[str] = []
            args = _shape({**base, cat_arg: self.rows[line][key], cap_arg: int(cap) if float(cap).is_integer() else cap}, self.schemas.get(tool), 'args', faults)
            reply = None if faults else self.act(tool, args)
            room -= 1
            if reply is None:
                return
            seen = reply.get('observation')
            self.observe(tool, seen)
            prices = sorted((self.price_of(i) or 0 for i, _ in self.records_in(seen)))
            if not prices or prices[-1] > cap:
                return
            count, shown, param, value = self.paging(tool, seen) or (0, 1, '', None)
            fit = max(0, min(count, self.window(tool, shown, value) or count, (room - 1) * shown))
            if value is not None and fit < count and fit * len(prices) // count:
                lower = prices[fit * len(prices) // count]
                cap = lower - 1 if float(lower).is_integer() else lower * 0.999
                continue
            while value is not None and room > 0 and not self.finished:
                faults = []
                more = _shape({**args, param: value}, self.schemas.get(tool), 'args', faults)
                reply = None if faults else self.act(tool, more)
                room -= 1
                if reply is None:
                    break
                self.observe(tool, reply.get('observation'))
                value = (self.paging(tool, reply.get('observation')) or (None,) * 4)[3]
            self.trace.append({'kind': 'price_scan', 'line': line, 'cap': cap})
            return

    def cheaper_rows(self) -> list[str]:
        if self.brief.get('pick') != 'cheapest':
            return []
        mark = self.cart_mark() + f'#{self.version}'
        kind = self.kinds()
        if mark in self.cheaper_marks or not any(kind.values()):
            return []
        self.cheaper_marks.add(mark)
        cart = self.cart_ids()
        picks: list[str] = []
        for line in cart:
            price = self.price_of(line)
            if price is None or not kind.get(line):
                continue
            self.price_scan(line)
            if not (kind := self.kinds()).get(line):
                continue
            needs = [need.split(':', 1)[0] + ': ' + self.plain_value(need) for n, need in self.needs_now() if n == kind[line]]
            pool = [i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i) and kind.get(i) == kind[line] and self.price_rank(i) < price and self.plausible(i, kind[line])]
            pool.sort(key=lambda i: (-sum((self.meets(i, m) for m in needs)), self.price_rank(i)))
            picks += [i for i in pool[:2] if i not in picks]
        return self.fill_facts(picks[:CHEAPER_OPENS], self.tail() + 4)

    def note_found(self, name: str, observation: Any) -> None:
        listing = name in {t for job, t in self.roles.items() if job != 'search'} or (isinstance(observation, dict) and self.ident(observation))
        self.found.update({} if listing else {i: (self.found.get(i, (0, 0))[0] + 1, self.step) for i, _ in self.records_in(observation)})

    def observe(self, name: str, observation: Any, hold: bool=False) -> str:
        if self.absorb(observation, name) and (not hold):
            self.read_texts()
        self.note_found(name, observation)
        lines = self.loose_lines(observation)
        if not lines:
            return ''
        meaning = self.take_note(lines)
        if meaning:
            self.queued.append(f'(seen on {name}) {meaning}')
        return meaning

    def widen(self) -> None:
        tool, target = (self.roles.get('compare'), self.compare_target())
        if not tool or not target or self.compare_bare():
            self.widen_singly()
            return
        cap = self.compare_room()
        ranked = self.item_groups()
        picks = [i for ids in ranked for i in [j for j in ids if j not in self.full][:max(1, cap // max(1, len(ranked)))]][:cap]
        before = len(self.full)
        for group in [picks, picks[:2]][:1 + (len(picks) > 2)] if len(picks) >= 2 else []:
            faults: list[str] = []
            args = _shape({target: group}, self.schemas.get(tool), 'args', faults)
            reply = None if faults or self.left() <= 12 or len(self.full) > before else self.act(tool, args)
            if reply is None:
                return
            seen = reply.get('observation')
            self.observe(tool, seen)
            self.note_view(tool, seen)
            rows = self.records_in(seen)
            if rows:
                self.facts_from[tool] = self.facts_from.get(tool, False) or any((_fact_block(r) for _, r in rows))
            self.trace.append({'kind': 'widen', 'ids': group, 'opened': len(self.full) - before})
            if rows and self.facts_from.get(tool) is False:
                self.widen_singly()
                return

    def widen_singly(self) -> None:
        each = WIDEN_EACH if len(self.items) <= 2 else 4
        lanes = [(self.ranked([i for i in ids if i not in self.full and self.buyable(i)])[:max(0, each - opened)], max(0, 4 - opened)) for ids in self.item_groups() for opened in [sum((1 for i in ids if i in self.full))]]
        first = [lane[k] for k in range(4) for lane, head in lanes if k < min(head, len(lane))]
        more = [lane[k] for k in range(each) for lane, head in lanes if head <= k < len(lane)]
        cap = max(5, 2 * len(lanes) + 2)
        read = self.fill_facts(first[:cap][:8], 22)
        self.fill_facts((first[cap:] + more)[:max(0, 10 - len(read))], 30)

    def ready_to_ask(self) -> bool:
        if self.asked or not self.roles.get('ask') or self.msgs_free() <= 0:
            return False
        if self.left() <= 8:
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

    def interview(self) -> str:
        self.asked = True
        if self.msgs_free() <= 0:
            return ''
        self.read_texts()
        self.widen()
        lanes = self.plan_lanes(max(self.msgs_free(), -(-len(self.key_table()) // len(self.items))) if self.listing else self.msgs_free())
        found, spent, used, misses = ({n: 0 for n in lanes}, {n: 0 for n in lanes}, {n: set() for n in lanes}, {n: 0 for n in lanes})
        follow, sent, told = ([], [], [])
        while self.msgs_free() > 0 and (not self.finished) and (self.left() > 6):
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
            used[number].update([_words(labels[0])] if len(labels) == 1 else [])
            batch = self.send([(number, labels[0], text)])
            if not batch:
                break
            sent += batch
            spent[number] += 1
            had = [len(item.told) for item in self.items]
            heard = self.hear(batch, (LIGHT, HEAVY))
            heard.pop('free', None) if len(labels) > 1 else None
            self.take_heard(heard, told)
            fresh = [n for n, item in enumerate(self.items, 1) if len(item.told) > had[n - 1]]
            for n in fresh:
                found[n] = found.get(n, 0) + 1
            crowded = 1 in [_count(x) for x in _items(heard.get('crowded'))]
            if len(labels) > 1 and crowded and not fresh:
                self.listing = False
                follow.insert(0, (number, labels[:1], False))
                continue
            if len(labels) > 1:
                new = [need.split(':', 1)[0] for item, size in zip(self.items, had) for need in item.told[size:]]
                hit = max([i for i, label in enumerate(labels) if any((_names(_stem(label), _stem(n)) or _names(_stem(n), _stem(label)) for n in new))], default=-1)
                used[number].update((_words(label) for label in labels[:max(hit, 0) + 1]))
                misses[number] = 0 if hit >= 0 else misses[number] + 1
                ahead, rest = (labels[hit + 1:] if hit >= 0 else labels[:0:-1], [label for label in lanes[number] if label not in labels])
                lanes[number] = rest + ahead if misses[number] and misses[number] % 2 == 0 else ahead + rest
                continue
            if fresh or not crowded:
                continue
            head = labels[0].split()[-1]
            if len(labels[0].split()) >= 2 and _words(head) not in used[number]:
                follow.insert(0, (number, [head], False))
            elif not shown and len(self.samples(number, labels[0])) >= 2:
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
        tool, cat_key = (self.roles.get('search'), self.category_key())
        schema = self.schemas.get(tool or '') or {}
        props, need = (schema.get('properties') or {}, schema.get('required') or [])
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        named = [k for k in texts if re.search('(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword', k, re.I)]
        for args in reversed(self.searched):
            fields = [k for k in named or texts if isinstance(args.get(k), str) and re.search('[a-z]', args[k], re.I)]
            if not named:
                labels = {str(r.get(cat_key) or '').strip().lower() for r in self.rows.values()} - {''} if cat_key else set()
                fields = [k for k in fields if len({str(a.get(k)) for a in self.searched if a.get(k)}) >= 2 and any((len(str(a.get(k)).split()) >= 2 for a in self.searched)) and (not any((str(a.get(k) or '').strip().lower() in labels for a in self.searched)))]
            if fields and tool:
                field = fields[0]
                shelves = {_words(item.shelf or '') for item in self.items} | {_words(item.name) for item in self.items} | ({_words(str(row.get(cat_key) or '')) for row in self.rows.values()} if cat_key else set())
                keep = {k: v for k, v in args.items() if k != field and (isinstance((props.get(k) or {}).get('enum'), list) or k in need)}
                keep = {k: v for k, v in keep.items() if k in need or not isinstance(v, str) or (_words(v) not in shelves and (not any(({w for w in _stem(v) if len(w) >= 4} & {w for w in _stem(s) if len(w) >= 4} for s in shelves if s))))}
                worded = [a for a in self.searched if isinstance(a.get(field), str) and re.search('[a-z]', a[field], re.I)]
                for k in [k for k in keep if isinstance((props.get(k) or {}).get('enum'), list) and any((k in a for a in worded))]:
                    used = [_dump(a[k]) for a in worded if k in a]
                    keep[k] = next((a[k] for a in worded if k in a and _dump(a[k]) == max(used, key=used.count)))
                return (tool, keep, field)
        return None

    def plain_value(self, need: str) -> str:
        value = need.split(':', 1)[1].strip() if ':' in need else ''
        limit = re.search('\\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', value, re.I)
        if not limit:
            return value
        facts = [(k, v) for record in self.full.values() for k, v in (_fact_block(record) or {}).items()]
        named = [v for k, v in facts if _names(_stem(need.split(':', 1)[0]), _stem(k))] or [v for _, v in facts]
        vocab = {w for v in named for w in re.findall('[a-z0-9]+', str(v).lower())}
        return ' '.join((w for w in re.findall('[A-Za-z0-9]+', value[:limit.start()]) if w.lower() in vocab))

    def chain_need(self, need: str) -> str:
        return next((str(c.get('result') or '') for c in [self.chains.get(need) or {}] if c.get('state') == 'done' and c.get('relation') == 'same'), '')

    def wanted_words(self, number: int) -> set[str]:
        return {w for v in self.value_words(number) for w in re.findall('[a-z0-9]{3,}', v.lower())}

    def value_words(self, number: int, fresh: bool=False) -> list[str]:
        refs, told = ({m for _, m in self.refs}, self.items[number - 1].told)
        needs = [need for n, need in self.needs_now() if n == number and (not fresh or need in told or need in refs)]
        return list(dict.fromkeys((v for need in needs for v in [self.plain_value(self.chain_need(need) if need in refs else need)] if v)))

    def search_words(self, todo: dict[int, list[str]]) -> dict[int, list[str]]:
        kind, blocks, models = (self.kinds(), [], self.judges()[-1:])
        for number, values in todo.items():
            props = [_stem(need.split(':', 1)[0]) for n, need in self.needs_now() if n == number]
            ids = [i for i in reversed(self.seen_order) if kind.get(i) == number or not any(kind.values())][:ROW_LIMIT]
            pairs = [(_words(k), str(v)) for i in ids for k, v in (_fact_block(self.full.get(i)) or {}).items() if any((_names(p, _stem(k)) for p in props))]
            facts = {key: dict.fromkeys((v for k, v in pairs if k == key)) for key in dict.fromkeys((k for k, _ in pairs))}
            rows = '\n'.join((_dump(_clip({k: v for k, v in (self.rows.get(i) or {}).items() if isinstance(v, str) and v.strip()}, 100)) for i in ids))
            blocks.append(f'PRODUCT {number}: {self.items[number - 1].name} (at most {len(values) + 1} queries)\nVALUES: ' + '; '.join(values) + '\nNEW: ' + '; '.join(self.value_words(number, True)) + '\nVALUES THESE LISTINGS GIVE THOSE PROPERTIES: ' + ('; '.join((k + ': ' + ' | '.join(v) for k, v in facts.items())) or 'none seen') + '\nLISTINGS SEEN:\n' + (rows or 'none'))
        table = ((self.ask_json(WORDING + '\n\n' + '\n\n'.join(blocks), models, 1500) if models else None) or {}).get('queries')
        out = {n: [str(q).strip() for q in _items(table.get(str(n)) if isinstance(table, dict) else table if len(todo) == 1 else None) if str(q).strip()][:len(v) + 1] for n, v in todo.items()}
        self.trace.append({'kind': 'search_words', 'queries': out})
        return out

    def value_search(self, numbers: list[int]) -> str:
        try:
            plan, cheap = (self.search_template(), self.brief.get('pick') == 'cheapest')
            todo = {n: self.value_words(n) for n in numbers if any(((n, v) not in self.value_done for v in self.value_words(n, True)))}
            room = self.free_steps() * len(todo) // max(1, len(todo), len(self.uncarted()))
            if not plan or room < 2:
                return ''
            tool, base, field = plan
            self.value_done.update(((n, v) for n in todo for v in self.value_words(n, True)))
            words, done, seen = (self.search_words(todo), {str(a.get(field) or '').strip().lower() for a in self.searched}, [])
            for number, values in todo.items():
                share, picks = (room // len(todo), [])
                for query in [q for q in dict.fromkeys(words.get(number) or [' '.join(values), ' '.join(self.value_words(number, True))]) if q and q.lower() not in done]:
                    faults: list[str] = []
                    args = _shape({**base, field: query}, self.schemas.get(tool), 'args', faults)
                    reply = None if faults or share < 2 or self.finished else self.act(tool, args)
                    if reply is None:
                        break
                    done.add(query.lower())
                    self.observe(tool, reply.get('observation'))
                    self.more_pages(tool, args, reply.get('observation'))
                    share -= 1 + len(self.paged_obs)
                    kind, found = (self.kinds(), list(dict.fromkeys((i for page in [reply.get('observation')] + self.paged_obs for i, _ in self.records_in(page)))))
                    cap = self.budget if isinstance(self.budget, (int, float)) and (not all((self.price_of(i) is not None and self.price_of(i) > 10 * self.budget for i in found))) else None
                    rows = sorted((i for i in found if self.buyable(i) and (not any(kind.values()) or kind.get(i) == number) and (cap is None or (self.price_of(i) or 0) <= cap)), key=lambda i: self.price_rank(i) if cheap else 0)
                    seen.append(_dump(query) + (f' ({len(self.paged_obs) + 1} pages)' if self.paged_obs else '') + ': ' + (self.priced(rows[:6]) or 'no row of that kind'))
                    picks += [i for i in rows if i not in picks]
                    if rows and self.brief.get('pick') not in ('cheapest', 'most', 'least'):
                        break
                opened = self.fill_facts(sorted((i for i in picks if i not in self.full), key=lambda i: self.price_rank(i) if cheap else 0)[:max(0, min(CHEAPER_OPENS, share))], self.fill_room())
                seen += ['opened: ' + ' | '.join(opened)[:2400]] if opened else []
            self.trace += [{'kind': 'value_search', 'seen': [s[:200] for s in seen]}] if seen else []
            return "[Kyo searched each such item's requirement values the way the store's listings write them and read the " + ('cheapest' if cheap else 'first') + ' fitting-kind rows] ' + '\n'.join(seen) if seen else ''
        except Exception as exc:
            self.trace.append({'kind': 'crash', 'where': 'value_search', 'error': f'{type(exc).__name__}: {exc}'[:400]})
            return ''

    def next_ask(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> tuple[int, list[str]]:
        first, left = ({n: i for i, n in enumerate(self.plan_order)}, {n: [label for label in lanes[n] if _words(label) not in used[n]] for n in lanes})
        size = -(-sum(map(len, left.values())) // max(1, self.msgs_free())) if self.listing else 1
        for number in sorted(lanes, key=lambda n: (found.get(n, 0) > 0, spent.get(n, 0), first.get(n, len(first) + n))):
            if left[number]:
                return (number, left[number][:size])
        return (0, [])

    def ask_text(self, number: int, labels: list[str], shown: bool=False) -> str:
        samples = self.samples(number, labels[0]) if shown and len(labels) == 1 else None
        said = f"{', '.join(labels[:-1])} or {labels[-1]}" if len(labels) >= 2 else labels[0]
        return f'What {said} would you like for the {self.items[number - 1].name}' + (f', for example {samples[0]} or {samples[1]}?' if samples else '?')

    def fit_ask(self, number: int, labels: list[str], shown: bool) -> tuple[list[str], str]:
        for size in range(len(labels), 0, -1):
            text = self.ask_text(number, labels[:size], shown)
            if text in (self.message_args(text) or {}).values():
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

    def records_of_item(self, number: int) -> int:
        cat_key = self.category_key()
        item = self.items[number - 1]
        return sum((1 for record in self.full.values() if cat_key and self.owns(item, str(record.get(cat_key) or '')) and _fact_block(record)))

    def ask_setup(self) -> str:
        first = '\n'.join((str(x) for x in _items(self.brief.get('ask_first')) if str(x).strip()))
        return '\n\nREQUEST:\n' + self.request + '\n\nITEMS:\n' + '\n'.join((f'item {number}: {item.name} - settled by the request: ' + ('; '.join(item.musts) or 'nothing') for number, item in enumerate(self.items, 1))) + ('\n\nASK FIRST (the request asks to be asked about these):\n' + first if first else '')

    def plan_lanes(self, count: int) -> dict[int, list[str]]:
        table = self.key_table()
        listing, last = ([], None)
        for index, (label, name, carried, values) in enumerate(table, 1):
            if label != last:
                listing.append(f'category {label!r}:')
                last = label
            listing.append(f'  {index}. {name} (on {carried}): ' + ' | '.join(values) + self.not_stated(label, name, values) + (' [several features in one: ask last]' if re.search(CATCH_ALL, name) else ''))
        prompt = PLAN.replace('COUNT', str(max(1, count))) + self.ask_setup() + '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(listing) or '(no full listing records yet)')
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 3500) or {}
        lanes, hinted = ({n: [] for n in range(1, len(self.items) + 1)}, {n: set() for n in range(1, len(self.items) + 1)})
        for pick in [p for p in _items(plan.get('ask')) if isinstance(p, dict)]:
            number = min(max(1, _count(pick.get('item') or 1) or 1), len(self.items))
            label = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(pick.get('label') or ''))[0].strip().rstrip('?.')
            if not label or len(label.split()) > 5:
                continue
            if all((_words(label) != _words(other) for other in lanes[number])):
                lanes[number].append(label)
            if str(pick.get('hint') or '').strip():
                hinted[number].add(_words(label))
                self.plan_order = list(dict.fromkeys(self.plan_order + [number]))
        for number in lanes:
            keys, read, own = (self.lane_keys(number, table), max(1, self.records_of_item(number)), {})
            for label, key, carried, _ in table:
                if self.owner_of(label) == number:
                    own[key] = max(own.get(key, 0.0), carried / read)

            def spelled(label: str) -> str:
                plain = lambda text: re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', _words(text)))
                return next((k for k in own if plain(k) == plain(label)), label)

            def share_of(label: str) -> float:
                return own.get(_words(label), max([s for k, s in own.items() if _names(_stem(label), _stem(k)) or _names(_stem(k), _stem(label))], default=0.0))
            hinted[number], lane = ({_words(spelled(h)) for h in hinted[number]}, [])
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
                    placed[weak], moved = (key, placed[weak])
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
        prompt += f"\n\nBUDGET NOW: {self.budget} {self.brief.get('budget_unit') or ''}".rstrip() if self.budget is not None else '\n\nBUDGET NOW: none'
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
            for value in observation.values():
                if isinstance(value, str) or (isinstance(value, list) and value and all((isinstance(v, str) for v in value))):
                    lines += [v for v in _items(value) if len(v.split()) >= 5]
                elif isinstance(value, dict) and (not self.ident(value, False, 1)):
                    lines += [v for v in value.values() if isinstance(v, str) and len(v.split()) >= 6]
        waiting = {t for _, t in self.pending}
        return [line for line in lines if line not in self.heard and line not in self.texts and (line not in waiting)]

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
        lines = self.proof_lines(ids[:12])
        if not lines:
            return ''
        return "[evidence: keys whose names contain a requirement's property; one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property] " + ' | '.join(lines)

    def proof_lines(self, ids: list[str]) -> list[str]:
        needs = self.needs_now()
        many = len(self.items) > 1
        lines = []
        for ident in ids:
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
        return lines

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
                notes.setdefault(_code_key(part), set()).update(codes)
        return {'marks': marks, 'notes': notes}

    def fit_set(self, ident: str, index: dict[str, Any]) -> set[str]:
        mark = index['marks'].get(ident) or {}
        found = set(mark.get('works') or [])
        for code in self.fit_codes(ident, index):
            found |= index['notes'].get(_code_key(code), set())
        return found

    def fit_codes(self, ident: str, index: dict[str, Any]) -> set[str]:
        mark = index['marks'].get(ident) or {}
        return set(mark.get('codes') or []) | ({mark['own'].lower()} if mark.get('own') else set())

    def fit_pages(self, ids: list[str]) -> list[str]:
        if not self.bound:
            return []
        index = self.fit_index()
        lines = []
        for ident in ids:
            mark = index['marks'].get(ident) or {}
            own = mark.get('own') or ''
            for other in self.seen_order:
                seen = index['marks'].get(other) or {}
                if other in ids or not own or not (any((part == own for part, _ in seen.get('also') or [])) or (mark.get('more') and seen.get('own') == mark['more'])):
                    continue
                lines.append(f'{other} (about part {own}, which is {ident}): ' + self.record_of(other)[:900])
        return lines[:6]

    def item_ids(self, number: int) -> list[str]:
        cat_key = self.category_key()
        item = self.items[number - 1]
        return [i for i in self.seen_order if cat_key and self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))]

    def fit_pointers(self, ids: list[str], keep: int) -> None:
        index = self.fit_index()
        owned = {m['own'] for m in index['marks'].values() if m.get('own')}
        wanted = list(dict.fromkeys((m['more'] for i in ids for m in [index['marks'].get(i) or {}] if m.get('more') and m['more'] not in owned and m['more'] not in self.pointed)))[:max(0, self.left() - keep)]
        plan = self.search_template()
        if not wanted or not plan:
            return
        tool, base, field = plan
        for code in wanted:
            self.pointed.add(code)
            faults: list[str] = []
            args = _shape({**base, field: code}, self.schemas.get(tool), 'args', faults)
            if faults or self.finished:
                break
            reply = self.act(tool, args)
            if reply is None:
                break
            self.observe(tool, reply.get('observation'))
        self.trace.append({'kind': 'fit_pointers', 'codes': wanted})

    def plausible(self, ident: str, number: int) -> bool:
        refs = [m for _, m in self.refs]
        return all((self.meets(ident, need.split(':', 1)[0] + ': ' + value) >= 0 for n, need in self.needs_now() if n == number and need not in refs for value in [self.plain_value(need)] if value))

    def fit_devices(self, main: int, cart: list[str], index: dict[str, Any]) -> list[str]:
        parts = [i for p, m in self.bound if m == main for i in cart if i in self.item_ids(p)]
        pool = [i for i in self.item_ids(main) if i not in cart and self.buyable(i) and parts and all((self.fits_with(p, i, index) for p in parts)) and self.plausible(i, main)]
        return sorted(pool, key=self.price_rank)[:4]

    def basket_total(self, lines: dict[str, int]) -> float | None:
        prices = [(self.price_of(i), n) for i, n in lines.items()]
        return None if any((p is None for p, _ in prices)) else sum((p * (n or 0) for p, n in prices))

    def total_with(self, swap: dict[str, str]) -> str:
        total = self.basket_total({swap.get(i, i): self.held_units(i) for i in self.cart_ids()})
        return '' if total is None else f" (order total with {', '.join(swap.values())}: {total:g})"

    def fits_with(self, part: str, device: str, index: dict[str, Any]) -> bool | None:
        known = [(f, c) for f, c in ((self.fit_set(a, index), self.fit_codes(b, index)) for a, b in ((part, device), (device, part))) if f and c]
        return any((_fits_any(f, c) for f, c in known)) if known else None

    def basket_fit(self, lines: dict[str, int]) -> tuple[list[str], float | None]:
        try:
            index = self.fit_index()
            faults = []
            for part, main in self.bound:
                devices = [i for i in lines if i in self.item_ids(main)]
                for line in [i for i in lines if i in self.item_ids(part)]:
                    known = [self.fits_with(line, d, index) for d in devices]
                    if False in known and True not in known:
                        faults.append(f"{line} ({self.items[part - 1].name}) does not work with {', '.join(devices)}")
            return (faults, self.basket_total(lines))
        except Exception as exc:
            self.trace.append({'kind': 'crash', 'where': 'basket_fit', 'error': f'{type(exc).__name__}: {exc}'[:200]})
            return ([], None)

    def pricier_kept(self) -> str:
        if self.brief.get('pick') != 'cheapest':
            return ''
        cart = self.cart_ids()
        owner = self.kinds()
        for entry in reversed(self.trace):
            got = entry.get('got') if entry.get('kind') == 'check' else None
            if not isinstance(got, dict) or got.get('ok') is not False:
                continue
            for better in [str(b) for b in _items(got.get('better'))]:
                if better in cart or better in self.kept_held or better not in self.rows or (not self.buyable(better)):
                    continue
                rivals = [i for i in cart if owner.get(i) and owner.get(i) == owner.get(better) and self.price_rank(i) > self.price_rank(better)]
                if rivals and (not self.basket_fit({better if i == rivals[0] else i: self.held_units(i) for i in cart})[0]):
                    self.kept_held.add(better)
                    self.trace.append({'kind': 'pricier_kept', 'better': better, 'cart': rivals[0]})
                    return f"An earlier review found that {self.priced([better])} meets every requirement and costs less than {self.priced(rivals[:1])}, which is in the cart now. The store's records decide the fit (the category label and the attributes, not what such products are usually used for). Put {better} in its place, or name the requirement its record breaks, then check out."
        return ''

    def fit_faults(self) -> list[str]:
        self.fit_pointers(self.cart_ids(), self.tail() + 3)
        if not self.bound:
            return []
        cart = self.cart_ids()
        index = self.fit_index()
        faults = []
        for part, main in self.bound:
            devices = [i for i in cart if i in self.item_ids(main)]
            parts = [i for i in cart if i in self.item_ids(part)]
            if len(devices) != 1 or not parts:
                continue
            takes = self.fit_set(devices[0], index)
            for line in parts:
                fits = self.fit_set(line, index)
                if self.fits_with(line, devices[0], index) is False:
                    good = [i for i in self.item_ids(part) if self.fits_with(i, devices[0], index) and (not self.sold_out(i)) and (i not in cart) and self.plausible(i, part)]
                    if self.brief.get('pick') == 'cheapest':
                        good.sort(key=self.price_rank)
                    good = good[:4]
                    switch = self.fit_devices(main, cart, index)
                    text = f"{line} ({self.items[part - 1].name}, {', '.join(sorted(self.fit_codes(line, index))) or 'no code'}) does not work with {devices[0]} ({', '.join(sorted(self.fit_codes(devices[0], index))) or 'no code'})" + (f": {line} names {', '.join(sorted(fits)[:6])}" if fits else '') + (f"; {devices[0]} takes {', '.join(sorted(takes)[:6])}" if takes else '')
                    text += f'; seen parts of that kind that work with it: {self.priced(good)}' + self.total_with({line: good[0]}) if good else '; no seen part of that kind works with it, so look for one or switch the device: ' + self.fit_plan(short=True)
                    if switch:
                        text += f'; seen {self.items[main - 1].name} listings that every carted part works with: {self.priced(switch)}' + self.total_with({devices[0]: switch[0]})
                    faults.append(text)
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
            devices = {code: [i for i in self.item_ids(main) if _fits_any({code}, set(index['marks'].get(i, {}).get('codes') or []))][:2] for code, _ in ranked}
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
                price = self.price_of(i)
                if owner[i] == [n] and item.cap is not None and price is not None and (item.cap < price <= 10 * item.cap):
                    faults.append(f'{i} costs {price:g}, over the {item.cap:g} limit on each unit of item {n} ({item.name})')
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
        fit = self.fit_pages(ids)
        if fit:
            prompt += "\n\nOTHER PAGES ABOUT A CART LISTING'S FIT (the page it sends you to, or a note on another listing that its part also fits a model: such a note adds that model to the part's list, so the part's own page can show fewer):\n" + '\n'.join(fit)
        proof = self.proof_lines(ids)
        if proof:
            prompt += "\n\nKEYS WHOSE NAMES CONTAIN A REQUIREMENT'S PROPERTY (a key for that same property decides the requirement, and a value in other words fails it; a key for another part or aspect decides nothing):\n" + '\n'.join(proof)
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

    def trail(self) -> tuple[str, bool]:
        settled = len(self.ref_values)
        text = self.debrief('')
        fresh = len(self.ref_values) > settled
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        tool = self.page_tool()
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
            pending = [p for p in pending if p not in self.ref_values]
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
        if text:
            text += '\n'
        if seen:
            text += '[Kyo opened the pages a requirement points at]\n' + '\n'.join(seen) + '\n'
        if self.ref_values:
            text += '[values read from the pages requirements point at] ' + '; '.join((f'{k} -> {v}' for k, v in self.ref_values.items())) + '\n'
        hold = bool(lost) and self.trail_blocks < (3 if self.left() > 15 else 2) and (self.left() > 5)
        if hold:
            self.trail_blocks += 1
            text += 'The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: ' + ' | '.join(lost[:3]) + ". Search for it by its shop's name together with the values its description gives (the shop's other listings of that kind are not it), open the row of that shop whose page shows those values, apply the value its page gives, then check out."
        elif fresh:
            text += 'A requirement changed with the value just read: fix the cart for it, then check out again.'
        return (text, hold or fresh)

    def light_first(self) -> list[str]:
        return self.judges()[::-1]

    def chase_room(self) -> int:
        return self.tail() + 3 * max(0, len(self.items) - len(self.cart_ids()))

    def talked(self) -> bool:
        return self.asked or ((not self.roles.get('ask') or self.msgs_free() <= 0) and self.loop_calls > 0)

    def chase_new(self) -> list[str]:
        live = {need for _, need in self.needs_now()}
        return [need for _, need in self.refs if need in live and need not in self.ref_values and self.chains.get(need, {}).get('state', 'new') == 'new']

    def chase_open(self) -> bool:
        spare = min(LOOP_CALLS - self.loop_calls, self.left()) // 2
        return not self.finished and self.left() > self.chase_room() + 1 and bool(self.judges()) and (self.calls < ALL_CALLS - spare)

    def chase_due(self) -> bool:
        return not self.chase_broke and self.talked() and self.chased_at != self.step and bool(self.chase_new()) and self.chase_open() and (self.search_template() is not None or not self.roles.get('search'))

    def debrief(self, summary: str) -> str:
        chased = ''
        if self.chase_due():
            try:
                chased = self.chase()
            except Exception as exc:
                self.chase_broke = True
                self.trace.append({'kind': 'crash', 'where': 'chase', 'error': f'{type(exc).__name__}: {exc}'[:400]})
        found = self.value_search(list(dict.fromkeys((n for n, need in self.needs_now() if self.chain_need(need))))) if chased else ''
        return '\n\n'.join((t for t in (summary, chased, found) if t))

    def chase(self) -> str:
        self.chased_at = self.step
        self.read_texts()
        self.point(self.chase_new())
        lines = []
        for need in self.chase_new():
            ref = self.chains.get(need)
            if not ref or not self.chase_open():
                continue
            ref['tries'] += 1
            self.follow(ref)
            if ref['state'] == 'new' and ref['tries'] >= 2 and self.chase_open():
                ref['state'] = 'lost'
            pages = ' -> '.join(ref['pages']) or 'none'
            if ref['state'] == 'done':
                lines.append(f"- {need}: pages opened {pages}; the last page gives {ref['attribute']} {ref['value']}, so the requirement in force is: {ref['result']}")
            elif ref['state'] == 'lost':
                at = ref['at']
                lines.append(f"- {need}: no listing was found that is {at['seller'] or 'the'} {at['product'] or 'listing'} with {_dump(at['facts'])} (pages opened: {pages}); find it yourself, open its page and every listing that page sends the property to")
            else:
                at = ref['at']
                ahead = [i for i in (ref.get('plan') or {}).get('queue') or [] if i not in self.opened]
                lines.append(f"- {need}: not finished, too few steps (pages opened: {pages}); next: the {at['seller'] or ''} {at['product'] or 'listing'} with {_dump(at['facts'])}" + (f", likely {', '.join(ahead[:3])}" if ahead else '') + f'; open its page with {self.page_tool()}')
        self.trace.append({'kind': 'chase', 'refs': [_clip({k: r.get(k) for k in ('need', 'state', 'pages', 'at', 'result')}, 80) for r in self.chains.values()]})
        if not lines:
            return ''
        text = 'Kyo followed the requirements that point at another listing (each search and page used one step):\n' + '\n'.join(lines) + '\nApply the new requirements to every pick (search again with the new values where needed).'
        if self.queued:
            text += '\n[notes seen meanwhile] ' + ' | '.join(self.queued)
            self.queued = []
        return text

    def point(self, needs: list[str]) -> None:
        todo = [need for need in needs if need not in self.chains]
        if not todo:
            return
        prompt = POINT + '\n\nREQUIREMENTS:\n' + '\n'.join((f'{n}. {need}' for n, need in enumerate(todo, 1)))
        got = self.ask_json(prompt, self.light_first(), 1500) or {}
        parsed = {_count(e.get('number')): e for e in _items(got.get('refs')) if isinstance(e, dict)}
        for n, need in enumerate(todo, 1):
            entry = parsed.get(n) or next((e for e in _items(got.get('refs')) if isinstance(e, dict) and len(todo) == 1), {})
            target = self.target_of(entry.get('listing'))
            relation = str(entry.get('relation') or '').strip().lower()
            margin = entry.get('margin')
            if isinstance(margin, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', margin):
                margin = float(margin)
            self.chains[need] = {'need': need, 'attribute': str(entry.get('attribute') or need.split(':')[0]).strip(), 'relation': relation if relation in ('same', 'different', 'at_least', 'at_most', 'within') else 'same', 'margin': margin if isinstance(margin, (int, float)) and (not isinstance(margin, bool)) else None, 'unit': str(entry.get('unit') or '').strip(), 'at': target, 'pages': [], 'tries': 0, 'state': 'new' if target['id'] or target['seller'] else 'skip'}
        self.trace.append({'kind': 'point', 'refs': [_clip(self.chains[need], 80) for need in todo]})

    @staticmethod
    def target_of(listing: Any) -> dict[str, Any]:
        listing = listing if isinstance(listing, dict) else {}
        facts = listing.get('facts') if isinstance(listing.get('facts'), dict) else {}
        seller = str(listing.get('seller') or '').strip()
        return {'id': str(listing.get('id') or '').strip(), 'seller': seller.removesuffix("'s").removesuffix('’s').strip(), 'product': str(listing.get('product') or '').strip(), 'facts': {str(k).strip(): str(v).strip() for k, v in facts.items() if str(k).strip() and str(v).strip()}}

    def follow(self, ref: dict[str, Any]) -> None:
        while not self.finished:
            pid, out = self.find(ref)
            if pid is None:
                if out is not None:
                    ref['state'] = 'lost'
                return
            ref['pages'].append(pid)
            after = self.target_of(out.get('redirect'))
            if (after['id'] or after['seller'] or after['facts']) and after['id'] not in ref['pages']:
                ref['at'] = after
                ref.pop('plan', None)
                continue
            pairs = _items(out.get('pairs_with')) if ref['relation'] == 'same' else []
            values = [str(v).strip() for v in pairs if str(v).strip()] or [str(v).strip() for v in _items(out.get('value')) if str(v).strip()]
            if values:
                self.settle(ref, values)
            else:
                ref['state'] = 'lost'
            return

    def find(self, ref: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
        target = ref['at']
        tool = self.page_tool()
        if not self.single_target(tool):
            return (None, {})
        plan: dict[str, Any] = ref.setdefault('plan', {'texts': self.seek_texts(target), 'judged': set(), 'spare': [], 'unsure': [], 'doubt': None, 'fresh': [], 'view': None, 'queue': []})
        judged, spare, unsure = (plan['judged'], plan['spare'], plan['unsure'])
        while not self.finished:
            pool = sorted((i for i in self.likely(target, 'loose' in plan, ref['pages']) if i not in judged), key=lambda i: i not in plan['fresh'])
            opened = [i for i in pool if i in self.opened]
            rows = [i for i in pool if i not in self.opened][:ROW_LIMIT]
            picks: list[str] = []
            view = (set(opened), set(rows))
            if plan['doubt'] and plan['view'] == view:
                return plan['doubt']
            if (opened or rows) and plan['view'] != view:
                new = [i for i in opened if not plan['view'] or i not in plan['view'][0]]
                queue = [i for i in plan['queue'] if i in rows]
                plan['view'] = view
                if queue and (not any((self.states(target, i) for i in new))):
                    picks = queue
                else:
                    out = self.refer(ref, opened, rows)
                    if out is None:
                        plan['view'] = None
                        return (None, None)
                    if out['match']:
                        pid, got = self.read_value(ref, tool, out['match'], out)
                        if _gives(got) and got.get('exact') is not False:
                            return (pid, got)
                        if not plan['doubt'] or (_gives(got), self.score(target, pid)) > (_gives(plan['doubt'][1]), self.score(target, plan['doubt'][0])):
                            plan['doubt'] = (pid, got)
                        judged.add(pid)
                        continue
                    judged.update(opened)
                    picks = plan['queue'] = out['open']
                    spare += [i for i in rows if i not in picks and i not in spare]
                    unsure += [i for i in out['check'] if i not in unsure and i not in self.full]
            if not self.chase_open():
                return plan['doubt'] or (None, None)
            facts = self.facts_tool()
            if not picks and unsure and facts and (facts != tool) and self.fill_facts(unsure[:1], self.chase_room() + 1):
                judged.discard(unsure.pop(0))
                continue
            rest = [i for i in spare if i not in self.opened and i not in judged and (i not in picks)]
            if not picks and (not rest or plan['doubt'] or len(rest) > self.page_room(tool)) and self.seek(plan, target, ref['pages']):
                continue
            picks = picks or ([] if plan['doubt'] else rest)
            if picks:
                tried = self.open_pages(tool, picks + [i for i in rest if i not in picks])
                spare[:] = [i for i in spare if i not in tried]
                judged.update((i for i in tried if i not in self.opened))
                continue
            if plan['doubt'] or 'loose' in plan or not target['seller']:
                return plan['doubt'] or (None, {})
            plan['loose'] = True
        return (None, None)

    def states(self, target: dict[str, Any], ident: str) -> bool:
        words = set(re.findall('[a-z0-9]+', _dump(self.full.get(ident) or self.rows.get(ident) or {}).lower()))
        for value in target['facts'].values():
            parts = re.findall('[a-z0-9]+', value.lower())
            alpha = {p for p in parts if not p.isdigit() and len(p) >= 3}
            if not {p for p in parts if p.isdigit()} <= words or (alpha and (not alpha & words)):
                return False
        return True

    def read_value(self, ref: dict[str, Any], tool: str, pid: str, out: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if _gives(out):
            return (pid, out)
        facts = self.facts_tool()
        if facts and facts != tool and (pid not in self.full) and self.fill_facts([pid], self.chase_room() + 1):
            again = self.refer(ref, [pid], [])
            if again and again['match'] == pid:
                return (pid, again)
        return (pid, out)

    def seek_texts(self, target: dict[str, Any]) -> list[str]:
        values = sorted(dict.fromkeys(target['facts'].values()), key=lambda v: bool(re.search('\\d', v)))
        lead = target['seller'] or target['product']
        texts = [target['id']] if target['id'] and target['id'] not in self.rows else []
        if values:
            texts.append(' '.join([lead] + values))
        texts += [' '.join((t for t in (target['seller'], target['product']) if t))] + [f'{lead} {v}' for v in values] + [target['seller']]
        return list(dict.fromkeys((t.strip() for t in texts if t.strip())))

    def seek_args(self, text: str) -> tuple[str, dict[str, Any]] | None:
        plan = self.search_template()
        if not plan:
            return None
        tool, base, field = plan
        faults: list[str] = []
        args = _shape({**base, field: text}, self.schemas.get(tool), 'args', faults)
        return None if faults else (tool, args)

    def seek(self, plan: dict[str, Any], target: dict[str, Any], ref_pages: list[str]) -> bool:
        call = plan.pop('next', None)
        while call is None and plan['texts']:
            call = self.seek_args(plan['texts'].pop(0))
        paged = call is None
        if paged:
            call = plan.pop('page', None)
        if call is None:
            return False
        name, args = call
        reply = self.act(name, args)
        if reply is None:
            return bool(plan['texts'] or 'page' in plan)
        seen = reply.get('observation')
        self.observe(name, seen)
        self.read_cache[name + _dump(args)] = (self.version, self.step, self.show(seen))
        shown = [i for i, _ in self.records_in(seen)]
        plan['fresh'] = shown
        after = self.paging(name, seen) if name in self.shapes else None
        if after and after[3] is not None and after[0] > after[1]:
            faults: list[str] = []
            more = _shape({**args, after[2]: after[3]}, self.schemas.get(name), 'args', faults)
            mine = set(self.likely(target, 'loose' in plan, ref_pages)) & set(shown)
            if not faults and shown and mine == set(shown):
                plan['next'] = (name, more)
            elif not faults and mine and ('page' not in plan or paged):
                plan['page'] = (name, more)
        return True

    def likely(self, target: dict[str, Any], loose: bool=False, skip: list[str] | None=None) -> list[str]:
        if target['id'] in self.rows:
            return [target['id']]
        seller = '' if loose else _words(target['seller'])
        scored = [(self.score(target, i), i) for i in self.seen_order if i not in (skip or []) and (not seller or any((isinstance(v, str) and _words(v) == seller for v in self.rows[i].values())))]
        if not seller:
            best = max((s for s, _ in scored), default=None)
            scored = [(s, i) for s, i in scored if s == best and any(s)]
        return [i for _, i in sorted(scored, key=lambda x: x[0], reverse=True)]

    def score(self, target: dict[str, Any], ident: str) -> tuple[bool, int, int]:
        plain = lambda text: _stem(re.sub('[\\W_]+', ' ', str(text).lower()))
        text = _dump(self.rows.get(ident) or {}).lower()
        words = plain(text)
        facts = [f for f in (plain(v) for v in target['facts'].values()) if f]
        return (bool(target['id']) and target['id'].lower() in text, sum((1 for f in facts if f <= words)), len({w for w in plain(target['product']) if len(w) >= 4} & words))

    def page_room(self, tool: str) -> int:
        if not self.compare_target() or tool not in self.tool_keys or self.compare_pages() is False:
            return 1
        return self.compare_room()

    def open_pages(self, tool: str, ids: list[str]) -> list[str]:
        many, joiner = (self.compare_target(), self.roles.get('compare'))
        if len(ids) > 1 and self.page_room(tool) > 1:
            batch = ids[:self.compare_room()]
            faults: list[str] = []
            args = _shape({many: batch}, self.schemas.get(joiner), 'args', faults)
            before = set(self.opened)
            reply = self.act(joiner, args) if len(batch) > 1 and (not faults) else None
            if reply is not None:
                seen = reply.get('observation')
                self.observe(joiner, seen)
                self.note_view(joiner, seen)
                rows = self.records_in(seen)
                if rows:
                    self.facts_from[joiner] = self.facts_from.get(joiner, False) or any((_fact_block(r) for _, r in rows))
                pages = self.compare_pages() is not False
                shown = {i for i, _ in rows} & set(batch) if pages else set()
                self.opened = before | shown
                if shown:
                    self.compare_ok = max(self.compare_ok, len(batch))
                    self.read_cache[joiner + _dump(args)] = (self.version, self.step, self.show(seen))
                    return batch
                if pages:
                    self.compare_bad = min(self.compare_bad or 999, len(batch))
                if not self.chase_open():
                    return []
        faults = []
        args = _shape({self.single_target(tool): ids[0]}, self.schemas.get(tool), 'args', faults)
        if faults:
            return ids[:1]
        reply = self.act(tool, args)
        if reply is None:
            return ids[:1]
        seen = reply.get('observation')
        self.observe(tool, seen)
        self.note_view(tool, seen)
        if isinstance(seen, dict) and self.ident(seen):
            self.facts_from[tool] = self.facts_from.get(tool, False) or bool(_fact_block(seen))
        elif tool != self.roles.get('open'):
            self.not_pages.add(tool)
            self.opened.discard(ids[0])
        if tool == self.offer_tool():
            self.offer_seen[ids[0]] = self.step
        self.read_cache[tool + _dump(args)] = (self.version, self.step, self.show(seen))
        return ids[:1]

    def refer(self, ref: dict[str, Any], opened: list[str], rows: list[str]) -> dict[str, Any] | None:
        prompt = REFER + f"\n\nREQUIREMENT (where the chain starts): {ref['need']}\nPROPERTY: {ref['attribute']}\nWANTED LISTING: " + _dump(ref['at'])
        if opened:
            prompt += '\n\nOPENED LISTINGS:\n' + '\n'.join((f'{i}: ' + _dump(_clip(self.rows.get(i), 600))[:4000] for i in opened))
        if rows:
            prompt += '\n\nCANDIDATE ROWS (page not read yet):\n' + '\n'.join((f'{i}: ' + _dump(_clip(self.rows.get(i), 160))[:900] for i in rows))
        got = self.ask_json(prompt, self.light_first(), 3500)
        if got is None:
            return None
        match = str(got.get('match') or '').strip()
        got['match'] = match if match in opened else None
        picks = [match] + [str(x).strip() for x in _items(got.get('open'))]
        got['open'] = [i for i in dict.fromkeys(picks) if i in rows]
        got['check'] = [i for i in dict.fromkeys((str(x).strip() for x in _items(got.get('check')))) if i in opened]
        self.trace.append({'kind': 'refer', 'need': ref['need'], 'ids': opened, 'rows': len(rows), 'got': _clip(got, 150)})
        return got

    def settle(self, ref: dict[str, Any], values: list[str]) -> None:
        attribute, relation, first = (ref['attribute'], ref['relation'], values[0])
        value = ' or '.join(dict.fromkeys(values))
        found = re.search('-?\\d+(?:\\.\\d+)?', first)
        unit = first[found.end():].strip() if found else ''
        if relation == 'different':
            need = f'{attribute}: anything but {value}'
        elif relation == 'at_least':
            need = f'{attribute}: at least {first}'
        elif relation == 'at_most':
            need = f'{attribute}: at most {first}'
        elif relation == 'within' and found and (ref['margin'] is not None) and (not ref['unit'] or not unit or _words(ref['unit']) == _words(unit)):
            base = float(found.group())
            need = f"{attribute}: between {base - ref['margin']:g} and {base + ref['margin']:g} {unit or ref['unit']}".strip()
        elif relation == 'within':
            need = f"{attribute}: within {(ref['margin'] if ref['margin'] is not None else '')} {ref['unit']} of {first}".replace('  ', ' ')
        else:
            need = f'{attribute}: {value}'
        self.ref_values[ref['need']] = f'{value}, so: {need}'
        ref.update(state='done', value=value, result=need)
        self.version += 1

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

    def spare_rows(self, lost: list[str]) -> list[str]:
        owner = self.kinds()
        kinds = {owner.get(i) for i in lost} - {0, None}
        wanted = {w for n in kinds for w in self.wanted_words(n)}
        cap = self.budget if isinstance(self.budget, (int, float)) and self.brief.get('budget_scope') != 'each' else None
        pool = [i for i in self.seen_order if i not in self.full and i not in lost and self.buyable(i) and owner.get(i) in kinds and (cap is None or self.price_rank(i) <= cap) and self.plausible(i, owner[i])]
        pool.sort(key=self.price_rank)
        pool.sort(key=lambda i: -len(wanted & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower()))))
        return pool

    def replacements(self, lost: list[str]) -> list[str]:
        if not self.judges():
            return []
        spares = self.spare_rows(lost)[:CHEAPER_OPENS]
        self.fill_facts(spares, self.tail() + 2 * len(lost) + 1)
        spares = [i for i in spares if i in self.full]
        records = []
        for ident in spares + [i for i in reversed(self.seen_order) if i not in spares]:
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
        self.near_items = self.audit_fits(got, 'near')
        if not named:
            return ''
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return 'A careful review thinks these listings may fit: ' + '; '.join((f"item {k}: {', '.join(v)}" for k, v in named.items())) + (f" ({got.get('reason')})" if got.get('reason') else '') + '. Open them and buy what truly fits. Kyo never ends without a purchase: sent again, the ending call makes Kyo cart the nearest listings itself and check out.'

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

    def audit_fits(self, got: dict[str, Any] | None, key: str='fits') -> dict[str, list[str]]:
        fits = (got or {}).get(key)
        if not isinstance(fits, dict):
            return {}
        named: dict[str, list[str]] = {}
        for k, v in fits.items():
            number = next(iter(re.findall('\\d+', str(k))), '1' if len(self.items) == 1 else str(k))
            named.setdefault(number, []).extend((str(i) for i in _items(v) if str(i) in self.rows and (not self.sold_out(str(i)))))
        return {k: v[:4] for k, v in named.items() if v}

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
                low = props[count].get('minimum') if count else None
                if mode and count:
                    calls.append({target: ident, mode: props[mode]['enum'][1], count: have})
                elif count and (want > have or (isinstance(low, int) and low <= want - have)):
                    calls.append({target: ident, count: want - have})
                    continue
                else:
                    return None
            calls.append({target: ident, **({mode: props[mode]['enum'][0]} if mode else {}), **({count: want} if count else {})})
        return calls

    def meets(self, ident: str, need: str) -> int:
        record, (prop, _, value) = (self.full.get(ident) or self.rows.get(ident) or {}, need.partition(':'))
        want = _tokens(value)
        block = _fact_block(record) or {}
        keyed = [str(v) for k, v in block.items() if _names(_stem(prop), _stem(k))] or [str(v) for k, v in block.items() if _stem(k) <= _stem(prop) and _tokens(str(v)) & want]
        if not want or not keyed:
            return 0
        return 1 if _flat(value) in {_flat(e) for v in keyed for e in [v] + re.split('[,;/|]', v)} else 0 if want <= _tokens(' '.join(keyed)) else -1

    def basket(self, named: list[str], trust: bool, prefer: list[str]) -> tuple[dict[str, int], list[str], list[str]]:
        pointing = {m for _, m in self.refs}
        held = [i for i in self.cart_ids() if not self.sold_out(i)]
        pool = [i for i in self.seen_order if self.buyable(i)]
        owner = self.owners()
        cheapest = self.brief.get('pick') == 'cheapest'
        tiers: dict[int, list[str]] = {}
        scores: dict[int, dict[str, int]] = {}
        tops: dict[int, int] = {}
        for n in range(1, len(self.items) + 1):
            mine = [i for i in pool if owner.get(i) == n]
            needs = [m for k, m in self.needs_now() if k == n and m not in pointing]
            table = {i: [self.meets(i, m) for m in needs] for i in mine}
            live = [j for j in range(len(needs)) if any((row[j] > 0 for row in table.values()))]
            scores[n] = {i: sum((table[i][j] for j in live)) for i in mine}
            tops[n] = len(live)
            mine.sort(key=lambda i: (-scores[n][i], (self.price_rank(i), i not in self.full) if cheapest else (i not in self.full, self.price_rank(i))))
            fits, near = ([i for i in found.get(str(n), []) if i in pool and owner.get(i) in (0, n)] for found in (self.lead_items, self.near_items))
            ours, told, before = ([i for i in ids if i in mine] for ids in (held, named, self.built))
            vetted = list(dict.fromkeys(ours + fits if trust else fits + ours))
            if cheapest:
                vetted.sort(key=self.price_rank)
            tiers[n] = sorted(dict.fromkeys([i for i in prefer if i in mine] + vetted + told + before + near + mine), key=lambda i: -scores[n].get(i, 0))
        index = self.fit_index()
        known: dict[tuple[str, str], int] = {}

        def works(i: str, j: str) -> int:
            if (i, j) not in known:
                got = self.fits_with(i, j, index)
                known[i, j] = 0 if got is None else 1 if got else -1
            return known[i, j]
        price = lambda i: self.price_of(i) or 0.0
        within = {n: [i for i in ids if self.items[n - 1].cap is None or not self.items[n - 1].cap < price(i) <= 10 * self.items[n - 1].cap] or ids for n, ids in tiers.items()}
        pairs = {frozenset(pair) for pair in self.bound} | {frozenset((x, y)) for x in tiers for y in tiers if x < y and any((works(i, j) > 0 for i in within[x] for j in within[y]))}
        mates_of = {n: [m for pair in pairs if n in pair for m in pair if m != n] for n in tiers}
        order = sorted(tiers, key=lambda n: -len(mates_of[n]))
        each = self.brief.get('budget_scope') == 'each'
        cap = self.budget if isinstance(self.budget, (int, float)) else None
        if cap is not None and (not any((self.price_of(i) is not None and self.price_of(i) <= 10 * cap for ids in tiers.values() for i in ids))):
            cap = None
        plan: dict[str, int] = {}
        chose: dict[int, str] = {}
        reads: list[str] = []
        for pos, n in enumerate(order):
            options = [i for i in within[n] if i not in plan]
            mates = [chose[m] for m in mates_of[n] if m in chose]
            ahead = [m for m in mates_of[n] if m not in chose]
            options.sort(key=lambda i: (-sum((works(i, j) for j in mates)), -sum((any((works(j, i) > 0 for j in within[m])) for m in ahead))))
            if not options:
                continue
            qty = self.items[n - 1].qty
            rest = sum((min((price(i) for i in within[m])) * self.items[m - 1].qty for m in order[pos + 1:] if within[m]))
            spent = sum((price(i) * q for i, q in plan.items()))
            pick = next((i for i in options if cap is None or (price(i) <= cap if each else spent + price(i) * qty + rest <= cap)), None) or min(options, key=price)
            plan[pick] = qty
            chose[n] = pick
            got = scores[n].get(pick, tops[n])
            if got < tops[n] or any((works(pick, j) < 1 for j in mates)):
                reads += [i for i in options if i != pick and i not in self.full and scores[n].get(i, got) >= got and (cap is None or each or price(i) <= cap)]
        spare = len(self.items) - len(chose)
        keep: dict[str, int] = {}
        for i in held:
            if i not in plan and owner.get(i, 0) not in chose and (owner.get(i, 0) or len(keep) < spare):
                keep[i] = self.cart_lines.get(i, 1)
        lines = {**keep, **plan}
        self.trace.append({'kind': 'basket', 'tiers': {n: ids[:4] for n, ids in tiers.items()}, 'links': sorted((sorted(pair) for pair in pairs)), 'plan': plan, 'keep': keep, 'reads': reads[:4], 'fit': self.basket_fit(lines)})
        return (lines, [i for i in self.cart_ids() if i not in lines], reads)

    def settle_cart(self, named: list[str], final: bool) -> str:
        buy, cart = (self.roles.get('buy'), self.roles.get('cart'))
        faults: list[str] = []
        args = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy or ''), 'args', faults)
        if not buy or faults or self.finished or self.left() <= 0:
            return ''
        late = self.halted or self.clock() < 1
        if self.refs and (not late) and (self.left() > len(self.items) + self.tail() + 1):
            self.trail()
        lines, drop, reads = self.basket(named, final, [])
        unread = list(dict.fromkeys([i for i in lines if i not in self.full] + reads))
        if unread and cart and (not late) and self.fill_facts(unread, len(lines) + self.fill_room()):
            lines, drop, reads = self.basket(named, final, [])
        verdict = '' if late else self.check_cart(list(lines), SPEND_HARD)
        better = [i for i in self.rows if i in verdict and i not in lines]
        if better:
            lines, drop, reads = self.basket(named, final, better)
        lines = lines or {i: q for i, q in self.last_cart.items() if not self.sold_out(i)}
        calls = (self.cart_calls(lines, drop) or self.cart_calls(lines, []) or []) if cart and lines else []
        if calls and len(calls) >= self.left():
            calls = self.cart_calls({i: self.cart_lines.get(i, q) for i, q in lines.items()}, []) or []
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        if not calls and (not (self.cart_ids() if fresh else self.cart_lines)) and (not (self.cart_touched and (not fresh))):
            return ''
        done = []
        for call in calls[:max(0, self.left() - 1)]:
            faults = []
            call = _shape(call, self.schemas.get(cart or ''), 'args', faults)
            if faults:
                break
            reply = self.act(cart or '', call)
            if reply is None or self.finished:
                break
            self.note_cart(cart or '', call)
            self.cart_said = _dump(reply.get('observation'))
            self.observe(cart or '', reply.get('observation'))
            done.append(_dump(call) + ' -> ' + self.show(reply.get('observation'))[:200])
        head = 'Kyo never ends without a purchase, so it set the cart to the nearest listings it has read (' + ('; '.join(done) or 'no change needed') + ')'
        self.fit_pointers(list(lines), self.left() if late else self.tail() + 1)
        if self.left() >= 2 and (not late) and self.judges(SPEND_HARD):
            go, said = self.gate('order')
            if not go and (not final):
                return head + '. ' + said
        mark = self.cart_mark()
        if self.finished or self.left() <= 0 or self.refused.get(mark, 0) > 1:
            return '' if self.finished or self.left() <= 0 else head + '. The store turned this same basket down twice, so Kyo did not send it again: change the basket as its replies say, then check out'
        reply = self.act(buy, args)
        self.trace.append({'kind': 'settle', 'lines': lines, 'drop': drop, 'carted': len(done), 'reply': _clip(reply, 200)})
        if reply is None:
            return ''
        self.buys += 1
        if not reply.get('finished'):
            self.refused_at = self.step
            self.refused[mark] = self.refused.get(mark, 0) + 1
        seen = reply.get('observation')
        self.absorb(seen)
        return head + ' and checked out. Store reply: ' + self.show(seen) + ('' if late else self.read_reply(buy, args, seen, reply))

    def read_reply(self, name: str, args: dict[str, Any], observation: Any, reply: dict[str, Any]) -> str:
        text = ''
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
        return text

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
        ends = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        cart, view, offer, facts = (self.roles.get('cart'), self.roles.get('cart_view'), self.roles.get('offer'), self.facts_tool())
        tiers = [(self.tail(), set()), (self.fill_room(), {cart, facts if left > self.fill_room(0) else None}), (self.tail() + 3, {cart, view}), (len(self.items) + 5 if self.buys == 0 else 0, {cart, view, offer, facts})]
        keep = next((ends | more for edge, more in tiers if left <= edge), None)
        tools = [t for t in self.tools if t['function']['name'] != self.roles.get('ask')]
        return [t for t in tools if keep is not None and ends and t['function']['name'] in keep] or tools

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
        self.last_cart = dict(self.cart_lines) if self.cart_lines else self.last_cart
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
        top = spec['maxItems'] if isinstance(spec.get('maxItems'), int) else None
        if self.compare_bad:
            room = max(2, min(self.compare_bad - 1, max(self.compare_ok, 2)))
            return min(room, top) if top else room
        return top or max(4, self.compare_ok)

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
            verify = name == self.facts_tool() and str(args.get(target)) in self.rows and (str(args.get(target)) not in self.full) and (self.left() > self.fill_room(0))
            if not ending and self.roles.get('buy') and (self.left() <= self.fill_room()) and (not verify) and (name != self.roles.get('cart') or self.left() <= self.tail()):
                out.append((cid, 'Not run (no step used): the last steps are kept for ' + ("checking out (with the shopper's approval of the basket when it is due) or ending without a purchase." if name == self.roles.get('cart') else 'putting one line for each item in the cart (the nearest listing where none meets every requirement) and checking out, with room to fix the cart if the store turns the order down.')))
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
                self.read_texts()
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
                    kept = self.pricier_kept() if self.left() > self.tail() + 3 else ''
                    if kept:
                        out.append((cid, 'Not run yet (no step used). ' + kept + '\nCart now: ' + self.cart_text()))
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
                    read = self.sweep(self.sweep_due(True), self.free_steps())
                    review = "Kyo first opened rows of the items not in the cart that searches showed but no one had opened (one step each); if one meets every requirement, cart it and check out instead:\n" + '\n'.join(read) if read else self.audit()
                    if review:
                        out.append((cid, 'Not run yet (no step used). ' + review))
                        continue
                    if self.leads and self.left() > 4 and (self.refuse_blocks == 0):
                        self.refuse_blocks = 1
                        self.trace.append({'kind': 'refuse_blocked', 'leads': list(self.leads)})
                        out.append((cid, 'Not run yet (no step used). The review found listings that appear to fit: ' + ', '.join(self.leads) + '. Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. Kyo never ends without a purchase: sent again, the ending call makes Kyo cart the nearest listings itself and check out.'))
                        continue
                    if self.leads and self.left() > 4 and (self.refuse_blocks == 1):
                        self.refuse_blocks = 2
                        first = self.leads[0]
                        self.check_cart([first], SPEND_HARD)
                        if self.verdicts.get(self.check_key([first])) and first in self.leads:
                            self.trace.append({'kind': 'refuse_blocked', 'confirmed': first})
                            out.append((cid, f"Not run yet (no step used). An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out."))
                            continue
            if self.is_refuse(name, args) and (not self.leads) and (not self.pushed) and (self.left() > self.max_steps // 3):
                self.pushed = True
                out.append((cid, f"Not run yet (no step used). {self.left()} steps are still left, so search in ways you have not tried: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. Kyo never ends without a purchase: then cart the best listing for each item and check out (sent again, the ending call makes Kyo cart the nearest listings itself)."))
                continue
            if self.is_refuse(name, args):
                done = self.close([i for i in self.rows if i in _dump(args)], False)
                if done or not self.pushed:
                    self.pushed = True
                    out.append((cid, done + (' [episode finished]' if self.finished else '') if done else "Not run (no step used): Kyo never ends without a purchase. Search each item's category, open the listings that might fit, cart the best one for each item and check out."))
                    continue
                args = self.quiet(name, args)
            if ending and self.left() >= 2:
                go, said = self.gate('order' if self.is_buy(name, args) else 'end')
                if not go:
                    out.append((cid, 'Not run yet. ' + said + '\nCart now: ' + self.cart_text()))
                    continue
                if said:
                    trimmed = '\n[Kyo] ' + said + trimmed
            trimmed += self.cart_sweep(args) if name == self.roles.get('cart') and (not ending) else ''
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
            listed = self.absorb(observation, name)
            self.note_found(name, observation)
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
                self.built += [i for i in self.cart_lines if i not in self.built]
            if name == self.roles.get('search') and (self.records_in(observation) or listed):
                self.searched = (self.searched + [dict(args)])[-8:]
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
            text += self.read_reply(name, args, observation, reply)
            if self.queued:
                text += "\n[notes seen on Kyo's own calls] " + ' | '.join(self.queued)
                self.queued = []
            if name in readers:
                self.read_cache[mark] = (self.version, self.step, base)
            if reply.get('finished'):
                text += ' [episode finished]'
            out.append((cid, text))
        return out

    def close(self, named: list[str], final: bool, again: bool=False) -> str:
        said = ''
        self.read_texts()
        while not (self.finished or self.broke or self.left() <= 0):
            buys = self.buys
            said = self.settle_cart(named, final)
            if not said or self.finished or again or self.buys == buys:
                break
            named, final = ([], True)
        return said

    def finish(self, again: bool=False) -> str:
        try:
            said = self.close([], True, again)
            if said and again and (not self.finished):
                return said
        except Exception as exc:
            self.trace.append({'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]})
            buy = self.roles.get('buy')
            faults: list[str] = []
            args = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy or ''), 'args', faults)
            if buy and (not faults) and (not self.finished) and (self.left() > 0) and (self.cart_lines or self.cart_touched) and (self.refused.get(self.cart_mark(), 0) < 2):
                self.act(buy, args)
        refuse = self.roles.get('refuse')
        if self.finished or self.broke or self.left() <= 0 or self.cart_touched or self.buys or (not refuse):
            return ''
        faults = []
        args = _shape(self.quiet(refuse, dict(self.brief.get('refuse_args') or {})), self.schemas.get(refuse), 'args', faults)
        if self.left() >= 2 and (not faults) and (not (self.halted or self.clock() < 1)):
            self.gate('end')
        if not faults and (not self.finished) and (self.left() > 0):
            reply = self.act(refuse, args)
            self.trace.append({'kind': 'forced_refuse', 'reply': _clip(reply, 200)})
        return ''

    def trim(self) -> None:
        if sum((len(_dump(m.get('content'))) for m in self.chat)) <= HISTORY_LIMIT:
            return
        tools = [i for i, m in enumerate(self.chat) if m.get('role') == 'tool']
        for index in tools[:max(1, len(tools) // 2)]:
            content = self.chat[index].get('content') or ''
            if len(content) > 600:
                ids = [i for i in dict.fromkeys(re.findall('"([A-Za-z0-9][A-Za-z0-9_\\-:.]{5,40})"', content)) if i in self.rows][:20]
                self.chat[index]['content'] = content[:400] + f"...(older result shortened; listings in it: {', '.join(ids)})"

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.read_request()
        self.trace.append({'kind': 'brief', 'brief': self.brief})
        opening = self.view.get('opening')
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            self.absorb(opening)
            intro += '\n\nWHAT THE STORE SHOWS FIRST:\n' + self.show(opening)
        intro += '\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n' + self.status()
        self.chat = [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': intro}]
        self.work(LOOP_CALLS, SPEND_HARD)
        turned = self.finish(again=True)
        if turned and (not self.finished) and (not self.broke) and (not self.no_credit) and (self.left() > 1):
            self.chat.append({'role': 'user', 'content': '[Kyo] ' + turned + '\nThe store did not accept the order: apply what it says, then check out again.\n' + self.status()})
            self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
        self.finish()
        self.trace.append({'kind': 'end', 'finished': self.finished, 'steps': self.step, 'cost': round(self.cost, 5), 'calls': self.calls, 'secs': round(time.monotonic() - self.started, 1)})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        self.loop_cap, self.spend_cap = (loop_cap, spend_cap)
        while not self.finished and (not self.broke) and (self.left() > 0) and (self.loop_calls < loop_cap) and (self.cost < spend_cap) and (self.clock() > 0):
            self.closing = bool(self.spare()) and self.turns_left() <= 1
            if not self.closing and (self.ready_to_ask() or self.chase_due()):
                summary = self.debrief(self.interview() if self.ready_to_ask() else '')
                if summary:
                    self.chat.append({'role': 'user', 'content': summary + '\n' + self.status()})
                continue
            read = [] if self.closing else self.sweep(self.sweep_due(), self.free_steps())
            if read:
                self.chat[-1]['content'] += "\n[Kyo opened rows of the items not in the cart yet that searches showed but no one had opened, one step each; compare them with your picks before carting]\n" + '\n'.join(read) + '\n' + self.status()
            self.trim()
            if self.closing:
                self.chat[-1]['content'] += '\n[Kyo] This is the last turn Kyo can pay for, and Kyo never ends without a purchase: call the cart tool now with the best listing you have read for each item that has no cart line, at its quantity (the nearest one where none meets every requirement). Kyo checks out right after.'
            self.loop_calls += 1
            before, start = (self.cost, time.monotonic())
            message = self.think(self.chat, [LIGHT, SPARE], 3000, [t for t in self.offered() if not self.closing or t['function']['name'] == self.roles['cart']])
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
            content = message.get('content') if isinstance(message.get('content'), str) else _dump(message.get('content')) if message.get('content') else ''
            if not calls:
                idle += 1
                self.chat += [{'role': 'assistant', 'content': content or '(no call)'}, {'role': 'user', 'content': 'Call one of the tools to continue.\n' + self.status()}]
                if idle >= 3:
                    break
                continue
            idle = 0
            committing = {self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse')} - {None}
            if not self.asked and (not self.closing) and self.roles.get('ask') and (self.msgs_free() > 0) and any((c['function']['name'] in committing for c in calls)) and (self.left() > 8):
                self.chat += [{'role': 'assistant', 'content': content or None, 'tool_calls': calls}] + [{'role': 'tool', 'tool_call_id': call['id'], 'content': 'Not run (no step used): the shopper is asked first.'} for call in calls]
                summary = self.debrief(self.interview())
                self.chat.append({'role': 'user', 'content': (summary or 'The interviewer had nothing to ask.') + '\n' + self.status()})
                continue
            self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
            self.trace.append({'kind': 'turn', 'text': content[:300], 'calls': [c['function'] for c in calls]})
            replies = self.run_calls(calls)
            self.turn_cost = self.cost - before
            self.slowest = max(self.slowest, time.monotonic() - start)
            self.chat += [{'role': 'tool', 'tool_call_id': cid, 'content': text + ('\n' + self.status() if index == len(replies) - 1 else '')} for index, (cid, text) in enumerate(replies)]

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Kyo(problem_data)
    except Exception as exc:
        return [{'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]}]
    with suppress(ValueError):
        signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        return episode.run()
    except (Exception, KeyboardInterrupt) as exc:
        episode.halted = isinstance(exc, KeyboardInterrupt)
        episode.trace.append({'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]})
        try:
            if not episode.finished:
                episode.finish()
        except Exception as again:
            episode.trace.append({'kind': 'crash', 'error': f'{type(again).__name__}: {again}'[:400]})
        return episode.trace
__all__ = ['agent_main']
