"""ORO SN15 miner for validator v3 worlds: v11 line.

Provenance: v11.15 = v11.14 + spend counted at its true price when a call is billed to the user's own provider key (BYOK:
usage.cost reads 0, the price is cost_details.upstream_inference_cost), so the spend limits act locally as they do live
(v11.14 spent $7.30 on the r12 replica, live runs of the same file stop near the race key's $5). v11.14 = v11.13 + before a decline, the rows the audit says could fit a product it found no fit for are opened
(up to three) and the audit runs again, also when other products have fits (6 of v11.10's 20 wrong declines had a
product's fit never opened). v11.13 = v11.12 + two readings of the shopper: an answer that rejects everything but one value requires that
value (not its exclusion), and "not fussed" about a property the request requires drops that requirement
(docs/v11.6-loss-table.md section 6). v11.12 = v11.11 + a decline the store refuses with a change re-arms every check before a refusal, so the next
decline is judged against the changed task (12 of 56 in-scope wrong declines re-sent a refused decline at once:
docs/v11.6-loss-table.md section 6). v11.11 = v11.10 + the code's own buy (complete_and_buy) opens the cart lines with every page tool first, as
every other checkout does since v11.8 (r12 t-ef62e7b57f lost on it). v11.10 = v11.9 + a refusal whose audit names a fit for every product is bought at once (after the independent
check of that basket); v11.9's audits named fits in 21 declines and none was acted on. v11.9 = v11.8 + of equal fits the cheapest opened one (a world may grade the cheapest fitting basket even when
the request sets no price rule: r11, 20 of 90 tasks; docs/v11.6-loss-table.md RC3). v11.8 = v11.7 + every cart line opened once with each single-listing page tool before the first checkout (a store
may count a line as opened only through one of them: docs/v11.6-loss-table.md RC4). v11.7 = v11.6 + memory of opened listings: the decline audit, the cart review and the replacement search read
every opened listing of a requested kind first (they read the newest 30, 15 and 25 records), and every status carries a
per-product list of opened candidates (docs/v11.6-loss-table.md RC2). v11.6 = v11.5 + a message the store answers with an error is not counted and not read as the shopper's
answer; one refused for a spent budget ends the messages (docs/plan-v11.md fix 5). v11.5 = v11.4 + a decline keeps its named alternatives when the request asks for them (docs/plan-v11.md
fix 4). v11.4 = v11.3 + pointer pages: a refusal with an unresolved pointer first follows its chain, and a cart
line whose page says "see part X" has X's page opened before the checkout (docs/plan-v11.md fix 3). v11.3 = v11.2 + a late refusal still audited, and a lead the audit names bought by the code after the
independent check (docs/plan-v11.md fix 2). v11.2 = v11.1 + a wall clock (the platform stops an episode at 300 s with no order sent: 4 of 90 r11
replica episodes of v11.0; docs/plan-v11.md fix 1). v11.1 = v11.0 renamed into our vocabulary, logic unchanged: every class, method, attribute, module constant and
function, and the distinctive parameter and local names, carry the names in tools/v11/rename_map.py (applied by
tools/v11/rename.py, which proves the result equal to v11.0 by inverting the tables and comparing syntax trees); the
persona "Kyo" in the prompts became "Oradon". v11.0 = the released code of Know Thyself v1 (agent version
06ae7239-14de-74d9-8000-263e78622c43, winner of race 13, released 2026-11-04; sha256 6a4f17a791625c63...), itself a fork
of KYO v6 (the race-10 winner). Differences from our retired v10.33: docs/v10.33-vs-know-thyself.md; the plan for the
v11 line: docs/plan-v11.md.
"""

from __future__ import annotations

import json
import random
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

FAST_MODEL = 'anthropic/claude-haiku-5.5'
STRONG_MODEL = 'anthropic/claude-sonnet-5.5'
BACKUP_MODEL = 'openai/gpt-6-luna'
NAVIGATOR_TURNS = 42
MODEL_CALLS = 60
SPEND_SOFT_CAP = 0.055
SPEND_HARD_CAP = 0.078
STRONG_MODEL_ROOM = 0.02
RETRY_WAITS = (4, 10, 20, 35)
RETRY_WAIT_TOTAL = 45
VALUE_CHARS = 450
RESULT_CHARS = 15000
HISTORY_CHARS = 150000
PROCESS_ROUNDS = 3
PAGES_READ_BY_CODE = 6
CHEAPER_ROWS_OPENED = 3
DEADLINE_SECONDS = 290.0
NAVIGATOR_SECONDS = 25.0
WRAP_UP_SECONDS = 60.0
PROCESS_SECONDS = 15.0
_models = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_store = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
SYSTEM_PROMPT = 'You are Oradon, a careful buyer working for one shopper in an online store. You can act only by calling the store\'s tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.\n\nHOW THE STORE WORKS\n- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool\'s description and schema and use the exact argument names and allowed values.\n- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.\n- Search rows are summaries. Stores split a listing\'s record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Oradon knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.\n- Questions to the shopper are handled for you by Oradon\'s interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.\n- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Oradon carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store\'s tools (product questions go through Oradon\'s interviewer).\n\nWHAT COUNTS AS A FIT\n- The kind of product is decided by the listing\'s category label: only the label that matches the requested item as a whole counts, never a neighbouring one.\n- A requirement on a property is met when the listing\'s attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value\'s words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.\n- An attribute whose key names the requirement\'s property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing about the requirement. Oradon adds [evidence] lines after opened listings with the keys whose names match a requirement.\n- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.\n- When the category label itself states a requirement\'s value (a season or audience in the label), that requirement is met by the category.\n- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing\'s value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.\n- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.\n- When one requested item must fit another (an accessory for a device), choose the two together from the listings\' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.\n- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.\n- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.\n\nCHOOSING\n- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Oradon reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.\n- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.\n- Otherwise any listing that meets everything is right: do not waste steps hunting for something better, but when several listings you have already opened meet everything, take the cheapest of them (stores often grade the cheapest fitting choice).\n- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.\n- A request for several different items needs one cart line per item, each at its requested quantity.\n\nCHANGES WHILE YOU WORK\n- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool\'s result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.\n- Re-read a chosen listing\'s record shortly before checking out, because price and stock can change silently.\n\nFINISHING\n- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.\n- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request\'s wording for that ending, and when a remark is needed, keep it short and name no listing.\n- Every reply must call at least one tool. Keep your text short: what you know, then the next move.'
SETUP_PROMPT = 'You prepare a shopping task for an agent. Read the shopper\'s request and the store\'s tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:\n\n"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request\'s words, "shelf": the store\'s category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request\'s own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property\'s value or limit outright; and never turn words that are part of what the product is called (a season or an audience named with the product) into musts: those belong to "name". A use or occasion the request makes a property of the product itself ("blazers for business wear", "a dress made for parties", "the occasion has to be party") IS a must on that property ("occasion: business"), unless the store\'s category label already names it; an event, person or setting the product is bought for ("to wear at his wedding", "for a colleague"), and how it should sit, feel or look in use, stay context. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.\n"budget": the most the shopper will pay in total as a number in the request\'s own unit, or null.\n"budget_unit": that unit in words (for example "cents"), or null.\n"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.\n"in_stock_only": true unless the request clearly accepts items that are not available.\n"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.\n"pick_measure": what is maximised or minimised for "most"/"least", else "".\n"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.\n"steps": how many steps, moves or tool calls the request allows (integer), or null.\n"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].\n"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.\n"wishes": a list of things the shopper would merely like.\n"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).\n"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper\'s approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.\n"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.\n"fits": a list of {"item": item number, "with": item number} for each requested item that the request says must work with, fit or suit ANOTHER requested item (the one the agent picks); [] when none.\n"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing\'s attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing\'s current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).\n"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.\n"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.\n'
PLAN_PROMPT = 'You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property; a property the shopper has no view on gets "don\'t mind", and one the shopper does not think about at all gets a puzzled reply, so every property named spends part of a message.\n\nYou get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item\'s category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.\n\nChoose the properties in this order:\n0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.\n1. Hinted properties. The request\'s story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product\'s looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.\nName a hinted property by a listed key of that item when one fits the hint (a hint about who will wear it: the key for gender or audience; about its looks: the listed key for colour, pattern or finish that most of its listings carry); a property no listing of the item carries comes after the item\'s three most carried keys.\n2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item\'s listings; once five or more listings were read, avoid keys seen on only one or two of them.\nSkip a property the request already settles for that item (a requirement names that property; a key whose values only repeat a required value, such as a pattern key showing the required decoration, is NOT settled), a second name for a property already chosen, a key whose values each list several purchase options at once, identifiers, model names, warranty and other logistics. Pack sizes, counts and quantities per pack are properties like any other, and a key marked [not the stated ...] is another property than that requirement, whatever words its name shares with it. Rank every item on its own: its hinted properties first, then its listed keys.\n"label" is the property\'s plain name: for a listed key, all of the key\'s words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property\'s name, no example values, no brackets, no second property.\nChoose up to COUNT for each item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story\'s words it comes from, or an empty string"}]}, each item\'s picks in rank order'
ANSWERS_PROMPT = 'A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "budget": new total limit as a number or null}\n- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3").\n- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop\'s product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer\'s own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.\n- "free": the property of each question the shopper said they do not mind about.\n- Folksy or roundabout wording that names a value still states that value.\n- An answer that rejects everything except one value states that value itself: \"anything with P apart from V gets binned\", \"don\'t show me anything where P comes in other than V\", \"nothing else will do except V\", \"nothing short of V\" all give \"P: V\", never \"P: apart from V\" or \"P: other than V\". Only words that ACCEPT everything but V (\"anything bar V is fine\", \"any P except V\") exclude V.\n- \"drop\": also every property a line says the shopper is no longer fussed about or can be left off (\"you can leave shape off\"), even when it rides along with an answer to another question.'
NOTICE_PROMPT = 'While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:\n{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request\'s unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}\nA replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note.'
CHECK_PROMPT = 'You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.\nReading rules: a requirement is met when the listing\'s attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value\'s words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing\'s value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.\nWhen the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart\'s choice on that rule, the cart is wrong.\nReturn ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses.'
READING_RULES = 'the listing\'s attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value\'s words counts, such as "deep teal" for teal, limits include the edge); a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing\'s value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit.'
AUDIT_PROMPT = 'A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: ' + READING_RULES + '\nRecords marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.\nReturn ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}'
UNMIX_PROMPT = 'A shopper\'s request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request\'s wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}'
CHAIN_PROMPT = 'A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing\'s page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement\'s chain from the requirement\'s own description, link by link, through the pages\' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).\nReturn ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing, "find": "for a link not found: search words (its shop and the values the description gives)"}], "done": true when the last link is an OPENED page that states the property\'s value itself (also when the requirement asks for a value other than or unlike that one: the page\'s value is what it is measured against), else false, "value": the property\'s value as stated on that last page, or ""}]}\nA record is a link only when it matches the whole description of that link: the shop or brand, the kind of product AND every property value the description gives (for example "a 10 pieces multipack" rules out the same shop\'s listing with 10 capsules and no multipack); when no record matches all of it, that link is not found yet, even if the same shop sells other listings of that kind. Give "find" for such a link: the shop\'s name and the description\'s values, as a few search words.\nList each chain\'s links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown.'
PROCESS_PROMPT = 'A shopping agent is about to STAGE. The shopper\'s request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product\'s properties do not do a step.\nWhen a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.\nReturn ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}'
REPLACEMENT_PROMPT = "A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: " + READING_RULES + '\nReturn ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line\'s item, at most 6], "reason": "one sentence"}'
SHAPE_PROMPT = 'A store tool returned the result below. Name its parts by the exact keys the result and the tool\'s parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing\'s price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}'
CARRIED_KEYS_KEPT = 3
CARRIED_SHARE_MIN = 0.0
CARRIED_SHARE_GAP = 0.34
NOT_A_PROPERTY = '\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b'
BUNDLE_KEY = '^(?!.*\\b(?:types?|styles?|kinds?|shapes?)\\b).*\\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\\w*|accessor\\w*|extras?|misc\\w*)\\b'
OPENS_PER_PRODUCT = 6
MODEL_CODE = '[A-Z][A-Za-z]{0,5} ?\\d{3,6}'
PART_NUMBER = '[A-Z]{1,4}\\d{3,6}'

def _model_codes(text: Any) -> list[str]:
    return [re.sub('\\s+', ' ', c).strip().lower() for c in re.findall('\\b' + MODEL_CODE + '\\b', str(text or ''))]

def _compatibility_marks(text: Any) -> dict[str, Any]:
    text = str(text or '')
    own = re.search('[\\[(]\\s*(?:part\\s+)?(' + PART_NUMBER + ')\\s*[\\])]', text, re.I)
    codes: list[str] = []
    for hit in re.finditer('\\bmodel\\s*(?:no\\.?|number|#)\\s*[:.]?\\s*(' + MODEL_CODE + ')\\b', text, re.I):
        codes += _model_codes(hit.group(1))
    for hit in re.finditer('\\b(?:also sold as|also known as|sold also as)\\s+([^.;]+)', text, re.I):
        codes += _model_codes(hit.group(1))
    works: list[str] = []
    for hit in re.finditer('\\b(?<!not )(?<!never )(?:works with|compatible with|fits)\\s*:?\\s+([^.;]+)', text, re.I):
        if not re.search('\\bpart\\s+' + PART_NUMBER + '\\s+also\\s+$', text[max(0, hit.start() - 40):hit.start()], re.I):
            works += _model_codes(hit.group(1))
    also = [(hit.group(1).upper(), _model_codes(hit.group(2))) for hit in re.finditer('\\bpart\\s+(' + PART_NUMBER + ')\\s+also fits\\s*:?\\s*([^.;]+)', text, re.I)]
    more = re.search('\\b(?:more|details|see|read)\\s+(?:on|at|under|about)?\\s*(?:the\\s+)?part\\s+(' + PART_NUMBER + ')', text, re.I)
    return {'own': own.group(1).upper() if own else '', 'codes': codes, 'works': works, 'also': also, 'more': more.group(1).upper() if more else ''}

def _call_cost(usage: Any) -> float:
    """One answered call's price: the reported cost, or, for a call billed to the user's own provider key (the reported
    cost then reads 0), the provider's upstream cost."""
    if not isinstance(usage, dict):
        return 0.0
    number = lambda v: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0
    cost = number(usage.get('cost'))
    details = usage.get('cost_details') if isinstance(usage.get('cost_details'), dict) else {}
    upstream = number(details.get('upstream_inference_cost'))
    return upstream if (usage.get('is_byok') or cost == 0) and upstream > cost else cost


def _compact(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        return str(value)

def _clipped(value: Any, limit: int, depth: int=0) -> Any:
    if depth > 7:
        return '...'
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + '...'
    if isinstance(value, dict):
        return {k: _clipped(v, limit, depth + 1) for (k, v) in value.items()}
    if isinstance(value, list):
        return [_clipped(v, limit, depth + 1) for v in value]
    return value

def _json_object(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    text = text.strip()
    fence = re.match('^```[A-Za-z]*\\s*(.*?)\\s*```$', text, re.S)
    if fence:
        text = fence.group(1)
    try:
        json_reply = json.loads(text)
        return json_reply if isinstance(json_reply, dict) else None
    except ValueError:
        pass
    (left, right) = (text.find('{'), text.rfind('}'))
    if 0 <= left < right:
        try:
            json_reply = json.loads(text[left:right + 1])
            return json_reply if isinstance(json_reply, dict) else None
        except ValueError:
            return None
    return None

def _plain_words(key: str) -> str:
    key = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(key))
    return re.sub('[\\s_.\\-]+', ' ', key).strip().lower()

def _stems(text: str) -> frozenset[str]:
    return frozenset(w[:-1] if len(w) > 3 and w.endswith('s') else w for w in _plain_words(text).split())

def _root_words(text: str) -> set[str]:
    out = set()
    for w in _plain_words(text).split():
        w = w[:-2] if re.search('(?:ch|sh|ss|x|z)es$', w) else w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w
        if len(w) >= 3 and w not in ('and', 'for', 'the', 'with'):
            out.add(w)
    return out

def _names_property(property_name: frozenset[str], key: frozenset[str]) -> bool:
    if not property_name or not key:
        return False
    return property_name == key or property_name <= key or (len(key) >= 2 and key <= property_name)

def _schema_types(schema: dict[str, Any]) -> list[str]:
    kind = schema.get('type')
    if isinstance(kind, str):
        return [kind]
    return [k for k in kind or [] if isinstance(k, str)]

def _is_type(value: Any, kind: str) -> bool:
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

def _coerce_arguments(value: Any, schema: Any, where: str, shape_problems: list[str]) -> Any:
    if not isinstance(schema, dict):
        return value
    kinds = _schema_types(schema)
    if kinds and (not any(_is_type(value, k) for k in kinds)):
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
        if not any(_is_type(value, k) for k in kinds):
            shape_problems.append(f"{where} should be {'/'.join(kinds)}")
            return value
    options = schema.get('enum')
    if isinstance(options, list) and value not in options:
        if isinstance(value, str):
            low = value.strip().lower()
            match = [o for o in options if isinstance(o, str) and o.lower() == low]
            if match:
                value = match[0]
        if value not in options:
            shape_problems.append(f'{where}: allowed values are {options}')
    if isinstance(value, str) and isinstance(schema.get('maxLength'), int) and (len(value) > schema['maxLength']):
        value = value[:schema['maxLength']]
    if isinstance(value, str) and isinstance(schema.get('minLength'), int) and (len(value) < schema['minLength']):
        shape_problems.append(f'{where} is too short')
    if isinstance(value, (int, float)) and (not isinstance(value, bool)):
        if isinstance(schema.get('minimum'), (int, float)) and value < schema['minimum']:
            shape_problems.append(f"{where} is below {schema['minimum']}")
        if isinstance(schema.get('maximum'), (int, float)) and value > schema['maximum']:
            shape_problems.append(f"{where} is above {schema['maximum']}")
    if isinstance(value, list):
        if isinstance(schema.get('maxItems'), int) and len(value) > schema['maxItems']:
            shape_problems.append(f"{where} allows at most {schema['maxItems']} entries")
        if isinstance(schema.get('minItems'), int) and len(value) < schema['minItems']:
            shape_problems.append(f"{where} needs at least {schema['minItems']} entries")
        if isinstance(schema.get('items'), dict):
            value = [_coerce_arguments(v, schema['items'], f'{where}[{i}]', shape_problems) for (i, v) in enumerate(value)]
    if isinstance(value, dict):
        props = schema.get('properties') if isinstance(schema.get('properties'), dict) else {}
        if schema.get('additionalProperties') is False:
            for name in [n for n in value if n not in props]:
                shape_problems.append(f'{where}.{name} is not an allowed argument')
        for name in schema.get('required') or []:
            if name not in value:
                shape_problems.append(f'{where}.{name} is required')
        value = {k: _coerce_arguments(v, props.get(k), f'{where}.{k}', shape_problems) if k in props else v for (k, v) in value.items()}
    return value

def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, (str, dict)) and value:
        return [value]
    return []

def _as_count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.0+)?\\s*', value):
        return int(float(value))
    return None

def _looks_like_id(value: Any, named: bool=False) -> bool:
    if not isinstance(value, str) or not 3 <= len(value.strip()) <= 64 or ' ' in value.strip():
        return False
    return named or bool(re.search('\\d', value))

def _id_column(rows: list[dict[str, Any]]) -> str | None:
    best = None
    for key in rows[0]:
        values = [row.get(key) for row in rows]
        if not all(_looks_like_id(v, 'id' in str(key).lower()) for v in values):
            continue
        values = [v.strip() for v in values]
        if len(rows) > 1 and len(set(values)) < len(values):
            continue
        score = ('id' in str(key).lower(), sum(len(v) for v in values) / len(values))
        if best is None or score > best[0]:
            best = (score, key)
    return best[1] if best else None

def _id_in_rows(node: Any, listing_id: str, depth: int=0) -> bool:
    if depth > 5:
        return False
    if isinstance(node, list):
        return any(isinstance(v, dict) and listing_id in [str(x).strip() for x in v.values() if isinstance(x, str)] for v in node) or any(_id_in_rows(v, listing_id, depth + 1) for v in node)
    if isinstance(node, dict):
        return any(_id_in_rows(v, listing_id, depth + 1) for v in node.values())
    return False

def _attribute_block(node: Any) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    best = None
    for value in node.values():
        if isinstance(value, dict) and len(value) >= 2:
            scalars = {k: v for (k, v) in value.items() if isinstance(v, (str, int, float)) and (not isinstance(v, bool))}
            if len(scalars) >= 2 and (best is None or len(scalars) > len(best)):
                best = scalars
    return best

class Product:

    def __init__(self, kind: str, category: str, quantity: int, stated: list[str]):
        self.kind = kind
        self.category = category
        self.quantity = quantity
        self.stated = stated
        self.revealed: list[str] = []
        self.waived: list[str] = []
        self.indifferent: list[str] = []

class Agent:

    def __init__(self, problem: dict[str, Any]):
        self.started = time.monotonic()
        environment = problem['environment']
        self.binding = dict(environment['binding'])
        self.policy_view = environment['policy_view']
        self.request = str(self.policy_view.get('query') or '')
        self.tools = [t for t in self.policy_view.get('tools') or [] if isinstance(t, dict) and isinstance(t.get('function'), dict) and isinstance(t['function'].get('name'), str)]
        self.schemas = {t['function']['name']: t['function'].get('parameters') if isinstance(t['function'].get('parameters'), dict) else {'type': 'object', 'properties': {}} for t in self.tools}
        steps = _as_count(self.policy_view.get('max_steps'))
        if steps is None and isinstance(self.policy_view.get('max_steps'), float):
            steps = int(self.policy_view['max_steps'])
        self.max_steps = steps if steps and steps > 0 else 30
        self.step = 0
        self.finished = bool(self.policy_view.get('finished'))
        self.store_credit_gone = False
        self.model_credit_gone = False
        self.wait_seconds = 0
        self.model_misses = 0
        self.spend = 0.0
        self.model_calls_made = 0
        self.navigator_turns = 0
        self.trace: list[dict[str, Any]] = []
        self.reading: dict[str, Any] = {}
        self.roles: dict[str, str] = {}
        self.products: list[Product] = []
        self.budget: Any = None
        self.message_limit = 0
        self.messages_sent = 0
        self.process_steps: list[dict[str, Any]] = []
        self.messages_reserved = 0
        self.process_messages = 0
        self.process_cleared: set[tuple[str, str]] = set()
        self.action_record: list[str] = []
        self.action_record_steps: list[int] = []
        self.interviewed = False
        self.questions_answered: list[tuple[str, str]] = []
        self.listings: dict[str, dict[str, Any]] = {}
        self.opened_records: dict[str, dict[str, Any]] = {}
        self.listing_order: list[str] = []
        self.notices_read: set[str] = set()
        self.changes: list[str] = []
        self.task_version = 0
        self.cart_changed = False
        self.cart_changed_step = -1
        self.cart_viewed_step = -1
        self.cart_view_result: Any = None
        self.notice_named_listing = False
        self.cart_counts: dict[str, int] = {}
        self.reviewed_carts: set[str] = set()
        self.refusal_audited = False
        self.offer_read_step: dict[str, int] = {}
        self.sold_out_step: dict[str, int] = {}
        self.unavailable: set[str] = set()
        self.sold_out_notices: set[str] = set()
        self.notice_sold_out = False
        self.sold_out_holds = 0
        self.id_arguments: set[tuple[str, str]] = set()
        self.last_cart_reply = ''
        self.refused_step = -1
        self.review_candidates: list[str] = []
        self.refusal_blocks = 0
        self.search_more_told = False
        self.built_lines: list[str] = []
        self.removed_lines: set[str] = set()
        self.late_refusal_held = False
        self.review_candidates_by_product: dict[str, list[str]] = {}
        self.review_verdicts: dict[str, bool] = {}
        self.confirmed_candidate: str | None = None
        self.completed_by_code = False
        self.id_field: str | None = None
        self.attributes_shown_by: dict[str, bool] = {}
        self.queued_notes: list[str] = []
        self.checkouts = 0
        self.refusals = 0
        self.recent_calls: list[str] = []
        self.conversation: list[dict[str, Any]] = []
        self.opened_pages: set[str] = set()
        self.pointer_requirements: list[tuple[int, str]] = []
        self.pointer_values: dict[str, str] = {}
        self.chain_rounds = 0
        self.chain_holds = 0
        self.compare_worked = 0
        self.compare_failed = 0
        self.fields_by_tool: dict[str, set[str]] = {}
        self.fields_seen: dict[str, tuple[int, set[str]]] = {}
        self.result_cache: dict[str, tuple[int, int, str]] = {}
        self.pages_read_by_code = 0
        self.result_shapes: dict[str, dict[str, Any] | None] = {}
        self.price_field: str | None = None
        self.stock_field: str | None = None
        self.cheaper_rows_done: set[str] = set()
        self.wider_search_held = False
        self.paged_results: list[Any] = []
        self.hinted_products: list[int] = []
        self.cart_line_holds = 0
        self.compatibility_pairs: list[tuple[int, int]] = []
        self.compatibility_holds = 0
        self.searches_made: list[dict[str, Any]] = []
        self.chain_searches = 0
        self.chain_search_rows: dict[str, list[str]] = {}
        self.part_pages_held = False
        self.decline_rearms = 0
        self.listing_tool_calls: set[tuple[str, str]] = set()
        self.lines_opened_for: set[str] = set()
        self.refusal_chain_followed = False

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def time_left(self) -> float:
        return DEADLINE_SECONDS - (time.monotonic() - self.started)

    def wrapping_up(self) -> bool:
        return self.time_left() < WRAP_UP_SECONDS

    def messages_free(self) -> int:
        return self.message_limit - self.messages_sent - max(0, self.messages_reserved - self.process_messages)

    def cart_signature(self) -> str:
        return _compact(sorted(self.cart_counts.items()))

    def page_tool_names(self) -> set[str]:
        page = self.roles.get('offer') or self.roles.get('open')
        return {t for t in (page, self.roles.get('compare')) if t}

    def attributes_tool(self) -> str | None:
        known = next((t for (t, ok) in self.attributes_shown_by.items() if ok and self.sole_id_argument(t)), None)
        if known:
            return known
        tool = self.roles.get('open')
        return tool if self.attributes_shown_by.get(tool or '') is not False and self.sole_id_argument(tool) else None

    def compare_shows_attributes(self) -> bool | None:
        return self.attributes_shown_by.get(self.roles.get('compare') or '')

    def compare_is_bare(self) -> bool:
        seen = self.compare_shows_attributes()
        return seen is False or (seen is None and (self.attributes_shown_by.get(self.attributes_tool() or '') is True or bool(self.roles.get('offer'))))

    def process_due(self, process_stage: str) -> bool:
        return any(p['when'] == process_stage for p in self.process_steps) and (process_stage, self.cart_signature() + f'#{self.task_version}') not in self.process_cleared

    def final_steps(self) -> int:
        needs_message = any(p['when'] == 'order' and p['messages'] > 0 for p in self.process_steps)
        return 2 if needs_message and self.messages_sent < self.message_limit and self.process_due('order') else 1

    def reviewer_models(self, limit: float=SPEND_SOFT_CAP) -> list[str]:
        if self.spend <= limit and self.spend + STRONG_MODEL_ROOM <= SPEND_HARD_CAP:
            return [STRONG_MODEL, FAST_MODEL]
        return [FAST_MODEL] if self.spend <= SPEND_HARD_CAP else []

    def model_turn(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None=None) -> dict[str, Any] | None:
        for model in models:
            if self.model_calls_made >= MODEL_CALLS:
                return None
            self.model_calls_made += 1
            tries = 0
            while True:
                (message, status) = self.post_model(model, messages, max_tokens, tools)
                if message is not None:
                    return message
                passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                if not passing or tries >= (1 if status in (None, 408) else len(RETRY_WAITS)) or self.wait_seconds + RETRY_WAITS[tries] > RETRY_WAIT_TOTAL or RETRY_WAITS[tries] * 1.2 > self.time_left() - NAVIGATOR_SECONDS:
                    break
                pause = RETRY_WAITS[tries] * (0.8 + 0.4 * random.random())
                self.wait_seconds += RETRY_WAITS[tries]
                tries += 1
                time.sleep(pause)
            if status == 402:
                self.model_credit_gone = True
                return None
        return None

    def post_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
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
        room = self.time_left() - PROCESS_SECONDS / 2
        if room < 3:
            self.trace.append({'kind': 'llm_skipped', 'model': model, 'seconds_left': round(self.time_left(), 1)})
            return (None, 'clock')
        _models.timeout = int(max(3, min(50, room)))
        result = _models.post_verbose('/inference/chat/completions', body)
        data = result.data
        if isinstance(data, dict) and data.get('choices'):
            usage = data.get('usage') or {}
            self.spend += _call_cost(usage)
            message = data['choices'][0].get('message') or {}
            if message.get('content') or message.get('tool_calls'):
                return (message, 200)
            self.trace.append({'kind': 'llm_fail', 'model': model, 'status': 0})
            return (None, 0)
        error = result.error or {}
        status = error.get('status')
        if isinstance(data, dict) and isinstance(data.get('error'), dict):
            status = _as_count(data['error'].get('code'))
            error = {'body': data['error'].get('message')}
        self.trace.append({'kind': 'llm_fail', 'model': model, 'status': status, 'body': str(error.get('body') or '')[:200]})
        return (None, status)

    def json_answer(self, prompt: str, models: list[str], max_tokens: int=1800) -> dict[str, Any] | None:
        for model in models:
            message = self.model_turn([{'role': 'user', 'content': prompt}], [model], max_tokens)
            json_reply = _json_object((message or {}).get('content'))
            if json_reply is not None:
                return json_reply
            if self.store_credit_gone or self.model_credit_gone or self.model_calls_made >= MODEL_CALLS:
                return None
        return None

    def store_call(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(self.binding)
        payload['step'] = self.step + 1
        payload['action'] = {'name': name, 'args': args}
        for _ in range(4):
            if self.time_left() < 1:
                self.trace.append({'kind': 'env_skipped', 'seconds_left': round(self.time_left(), 1)})
                return None
            _store.timeout = int(max(1, min(95, self.time_left())))
            result = _store.post_verbose('/environment/call', payload)
            if isinstance(result.data, dict):
                data = result.data
                self.step = data['step'] if isinstance(data.get('step'), int) else self.step + 1
                if data.get('finished'):
                    self.finished = True
                seen = data.get('observation')
                if not (isinstance(seen, dict) and seen.get('error')):
                    for value in args.values():
                        self.listing_tool_calls.update((name, v.strip()) for v in (value if isinstance(value, list) else [value]) if isinstance(v, str))
                if name in self.page_tool_names() and (not (isinstance(seen, dict) and seen.get('error'))):
                    for value in args.values():
                        self.opened_pages.update(v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str))
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
                    self.store_credit_gone = True
                return None
        return None

    def read_task(self) -> None:
        lines = []
        for tool in self.tools:
            function = tool['function']
            lines.append(f"- {function['name']}: {function.get('description', '')} | parameters: {_compact(function.get('parameters') or {})}")
        prompt = SETUP_PROMPT + '\nTOOLS:\n' + '\n'.join(lines) + '\n\nREQUEST:\n' + self.request
        if self.policy_view.get('opening') is not None:
            prompt += '\n\nOPENING NOTE FROM THE STORE:\n' + _compact(_clipped(self.policy_view.get('opening'), 1500))
        brief = self.json_answer(prompt, [FAST_MODEL], 4000) or {}
        if not self.model_credit_gone and (not _as_list(brief.get('items')) or not isinstance(brief.get('roles'), dict) or (not brief['roles'].get('buy'))):
            brief = self.json_answer(prompt, [STRONG_MODEL, FAST_MODEL], 2500) or brief
        elif not self.model_credit_gone and self.shares_category(brief):
            again = self.json_answer(prompt, [STRONG_MODEL], 2500) or {}
            if _as_list(again.get('items')) and isinstance(again.get('roles'), dict) and again['roles'].get('buy'):
                brief = again
        self.reading = brief
        roles = brief.get('roles') if isinstance(brief.get('roles'), dict) else {}
        self.roles = {k: v for (k, v) in roles.items() if isinstance(v, str) and v in self.schemas}
        for key in ('buy_args', 'refuse_args'):
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        (buy, refuse) = (self.roles.get('buy'), self.roles.get('refuse'))
        if buy and buy == refuse and (not brief['buy_args'] or not brief['refuse_args'] or brief['buy_args'] == brief['refuse_args']):
            ask = f'The store tool {buy} both checks out and ends without a purchase, depending on its arguments. Its schema: ' + _compact(self.schemas.get(buy)) + '\nReturn ONE JSON object and nothing else: {"buy_args": the fixed arguments that check out, "refuse_args": the fixed arguments that end without a purchase} (only the arguments that choose the mode).'
            fixed = self.json_answer(ask, [FAST_MODEL, STRONG_MODEL], 1500) or {}
            if isinstance(fixed.get('buy_args'), dict) and isinstance(fixed.get('refuse_args'), dict) and (fixed['buy_args'] != fixed['refuse_args']):
                (brief['buy_args'], brief['refuse_args']) = (fixed['buy_args'], fixed['refuse_args'])
        for entry in _as_list(brief.get('items')):
            if not isinstance(entry, dict):
                continue
            quantity = _as_count(entry.get('quantity'))
            count = quantity if quantity and quantity > 0 else 1
            musts = [str(m) for m in _as_list(entry.get('musts')) if m]
            self.products.append(Product(str(entry.get('name') or 'the product'), str(entry.get('shelf') or ''), count, musts))
        merged: list[Product] = []
        became: dict[int, int] = {}
        for (old, item) in enumerate(self.products, 1):
            twin = next((m for m in merged if item.category.strip() and _plain_words(m.category) == _plain_words(item.category) and (not any((_plain_words(a.split(':')[0]) == _plain_words(b.split(':')[0]) and _plain_words(a.partition(':')[2]) != _plain_words(b.partition(':')[2]) for a in m.stated for b in item.stated)))), None)
            if twin is None:
                merged.append(item)
                became[old] = len(merged)
            else:
                became[old] = merged.index(twin) + 1
                twin.stated += [m for m in item.stated if m not in twin.stated]
                self.trace.append({'kind': 'items_merged', 'into': twin.kind, 'dropped': item.kind})
        self.products = merged
        if not self.products:
            self.products = [Product('the requested product', '', 1, [])]
        for pair in _as_list(brief.get('fits')):
            if isinstance(pair, dict):
                (part, main) = (became.get(_as_count(pair.get('item')) or 0), became.get(_as_count(pair.get('with')) or 0))
                if part and main and (part != main) and (1 <= part <= len(self.products)) and (1 <= main <= len(self.products)) and ((part, main) not in self.compatibility_pairs):
                    self.compatibility_pairs.append((part, main))
        for lookup in _as_list(brief.get('lookups')):
            text = str(lookup).strip()
            if not any(text in item.stated for item in self.products):
                key = _plain_words(text.split(':')[0])
                text = next((m for item in self.products for m in item.stated if _plain_words(m.split(':')[0]) == key), text)
            product_number = next((n for (n, item) in enumerate(self.products, 1) if text in item.stated), 1)
            if text and (product_number, text) not in self.pointer_requirements:
                self.pointer_requirements.append((product_number, text))
        self.separate_pointer_details()
        budget = brief.get('budget')
        if isinstance(budget, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', budget):
            budget = float(budget)
        if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0):
            self.budget = budget
        limit = _as_count(brief.get('messages'))
        if limit is not None and limit >= 0:
            self.message_limit = limit
        elif self.roles.get('ask'):
            self.message_limit = 3
        steps = _as_count(brief.get('steps'))
        if steps is not None and 10 <= steps < self.max_steps:
            self.max_steps = steps
        for entry in _as_list(brief.get('process')):
            if isinstance(entry, dict) and str(entry.get('step') or '').strip():
                when = str(entry.get('when') or '').strip().lower()
                self.process_steps.append({'step': str(entry['step']).strip(), 'when': when if when in ('choose', 'order', 'end') else 'order', 'messages': max(0, _as_count(entry.get('messages')) or 0)})
        if self.roles.get('ask'):
            needs_message = [p['messages'] for p in self.process_steps if p['when'] in ('order', 'end') and p['messages']]
            self.messages_reserved = min(max(needs_message), self.message_limit) if needs_message else 0

    @staticmethod
    def shares_category(brief: dict[str, Any]) -> bool:
        shelves = [_plain_words(str(e.get('shelf') or '')) for e in _as_list(brief.get('items')) if isinstance(e, dict)]
        shelves = [s for s in shelves if s]
        return len(shelves) != len(set(shelves))

    def separate_pointer_details(self) -> None:
        for (product_number, item) in enumerate(self.products, 1):
            pointers = [requirement for (n, requirement) in self.pointer_requirements if n == product_number and requirement in item.stated]
            rest = [m for m in item.stated if m not in pointers]
            if not pointers or not rest:
                continue
            prompt = UNMIX_PROMPT + '\n\nREQUEST:\n' + self.request + f'\n\nPRODUCT: {item.kind}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n' + '\n'.join(pointers) + '\nOTHER REQUIREMENTS:\n' + '\n'.join(rest)
            json_reply = self.json_answer(prompt, [STRONG_MODEL, FAST_MODEL], 1500) or {}
            moved = [m for m in rest if m in [str(x).strip() for x in _as_list(json_reply.get('describe_other'))]]
            if not moved:
                continue
            item.stated = [m for m in item.stated if m not in moved]
            joined = pointers[0] + ' (that listing: ' + '; '.join(moved) + ')'
            item.stated = [joined if m == pointers[0] else m for m in item.stated]
            self.pointer_requirements = [(n, joined if (n, requirement) == (product_number, pointers[0]) else requirement) for (n, requirement) in self.pointer_requirements]
            self.trace.append({'kind': 'unmix', 'item': product_number, 'moved': moved})

    def is_described_only(self, product_number: int, requirement: str) -> bool:
        if requirement in [m for (_, m) in self.pointer_requirements] or requirement in [str(x) for x in _as_list(self.reading.get('lookups'))]:
            return False
        (property_name, _, value) = requirement.partition(':')
        if not property_name.strip() or len(re.findall('[a-z0-9]+', value.lower())) < 4:
            return False
        if re.search('\\b(?:listing|page|shown|other than|unlike|same as)\\b', value, re.I):
            return False
        item = self.products[product_number - 1]
        if any({w for w in _stems(other.kind) if len(w) >= 4} & _stems(value) for other in self.products if other is not item):
            return False
        category_column = self.category_field()
        keys: set[str] = set()
        read = 0
        for record in self.opened_records.values():
            if category_column and (not self.label_belongs_to(item, str(record.get(category_column) or ''))):
                continue
            block = _attribute_block(record) or {}
            if block:
                read += 1
                keys |= {_plain_words(k) for k in block}
        return read >= 3 and (not any(_stems(k) & _stems(property_name) for k in keys))

    def requirements_text(self) -> str:
        parts = []
        many = len(self.products) > 1
        for (product_number, item) in enumerate(self.products, 1):
            live = [m for m in item.stated + item.revealed if m not in item.waived]
            told = [m for m in live if self.is_described_only(product_number, m)]
            needs = [m + (f' (the page it points at gives: {self.pointer_values[m]})' if m in self.pointer_values else '') for m in live if m not in told]
            head = f'item {product_number}: {item.kind}' if many else f'item: {item.kind}'
            if item.category:
                head += f' (store label: {item.category})'
            head += f', quantity {item.quantity}'
            text = head + '; requirements: ' + ('; '.join(needs) if needs else 'none beyond the kind of product')
            if told:
                text += "; described in the shopper's words, which no listing attribute states (decides nothing): " + '; '.join(told)
            if item.indifferent:
                text += '; shopper does not mind: ' + ', '.join(item.indifferent)
            parts.append(text)
        brief = self.reading
        if self.budget is not None:
            scope = 'for each item' if brief.get('budget_scope') == 'each' else 'in total'
            parts.append(f"budget: at most {self.budget} {brief.get('budget_unit') or ''} {scope}".replace('  ', ' '))
        pick = brief.get('pick')
        if pick == 'cheapest':
            parts.append('pick: the cheapest listing that fits (lowest price)')
        elif pick in ('most', 'least'):
            parts.append(f"pick: the {pick} {brief.get('pick_measure') or ''} among fitting listings".replace('  ', ' '))
        else:
            parts.append('pick: any listing that fits everything; of the fitting listings already opened, the cheapest')
        if brief.get('in_stock_only', True) is not False:
            parts.append('in stock only')
        for wish in _as_list(brief.get('wishes')):
            parts.append(f'wish (tie-breaker only): {wish}')
        for rule in _as_list(brief.get('rules')):
            parts.append(f'rule from the request: {rule}')
        if brief.get('quiet_decline') is True:
            parts.append('if nothing fits: end without naming or suggesting any listing')
        elif self.reading.get('quiet_decline') is False:
            parts.append('if nothing fits: the request asks for alternatives, so the ending text names the listing that comes closest and says what it misses')
        for entry in self.process_steps:
            when = {'choose': 'before choosing', 'order': 'at the order (Oradon does it when you check out)', 'end': 'at an ending without a purchase (Oradon does it then)'}[entry['when']]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        facts = self.attributes_tool()
        if facts and (self.attributes_shown_by.get(facts) is True or self.roles.get('offer')):
            bare = sorted(t for t in self.page_tool_names() | {self.roles.get('offer') or ''} if t and self.attributes_shown_by.get(t) is False)
            pages = sorted(t for t in self.page_tool_names() if t != facts)
            parts.append(f'listing attributes come from {facts}' + (f"; {', '.join(bare)} show none" if bare else '') + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ''))
        return '\n'.join('- ' + p for p in parts)

    def status_text(self) -> str:
        left = self.steps_left()
        text = f'[Oradon] step {self.step} of {self.max_steps} used, {left} left.'
        if self.wrapping_up():
            text += ' TIME IS RUNNING OUT: put the chosen lines in the cart and check out now; this is your last turn or the one before it.'
        elif left <= self.final_steps() + 1:
            text += ' FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that).'
        elif left <= 6 or (left <= len(self.products) + 5 and self.checkouts == 0):
            text += ' Few steps left: set the cart to the chosen lines and check out now.'
        elif self.checkouts == 0 and left <= 12:
            text += ' Steps are running low: settle on your picks, fill the cart and check out within the next few steps.'
        elif self.checkouts == 0 and (not self.cart_changed) and (self.step >= self.max_steps // 2):
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        live = {requirement for (_, requirement) in self.live_requirements()}
        lookups = [requirement for (_, requirement) in self.pointer_requirements if requirement in live and requirement not in self.pointer_values]
        if lookups and self.checkouts == 0:
            text += '\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] ' + ' | '.join(lookups[:3])
        if self.review_candidates:
            text += '\n[review candidates] ' + ', '.join(self.review_candidates)
        plan = self.compatibility_plan() if self.compatibility_pairs and self.checkouts == 0 else ''
        if plan:
            text += '\n' + plan
        if self.changes:
            text += '\n[changes so far] ' + ' | '.join(self.changes[-4:])
        text += self.opened_ledger()
        text += '\n[ledger]\n' + self.requirements_text()
        return text

    def listing_id_of(self, node: Any, inside_list: bool=False, depth: int=0) -> str | None:
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if self.id_field and _looks_like_id(node.get(self.id_field), True):
            return node[self.id_field].strip()
        nested = any(isinstance(v, dict) for v in node.values())
        if not (inside_list or depth == 0 or nested):
            return None
        found = _id_column([node])
        if found and ('id' in str(found).lower() or node[found].strip() in self.listings):
            return node[found].strip()
        return None

    def learn_id_column(self, node: list[Any]) -> bool:
        rows = [n for n in node if isinstance(n, dict) and len(n) >= 3]
        if not rows or len(rows) < len(node) / 2:
            return False
        if self.id_field and all(_looks_like_id(r.get(self.id_field), True) for r in rows):
            return True
        found = _id_column(rows)
        if found:
            self.id_field = found
            return True
        return False

    def take_in(self, observation: Any) -> None:

        def walk(node: Any, depth: int, inside_list: bool=False) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                listing_id = self.listing_id_of(node, inside_list, depth)
                if listing_id:
                    if listing_id not in self.listings:
                        self.listing_order.append(listing_id)
                        self.listings[listing_id] = {}
                    old = self.listings[listing_id]
                    if any(old.get(k) is True and v is False for (k, v) in node.items()):
                        self.unavailable.add(listing_id)
                    elif any(old.get(k) is False and v is True for (k, v) in node.items()):
                        self.unavailable.discard(listing_id)
                    self.listings[listing_id].update(node)
                    if _attribute_block(node):
                        self.opened_records[listing_id] = dict(self.listings[listing_id])
                    elif listing_id in self.opened_records:
                        self.opened_records[listing_id].update(node)
                    return
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                rows = self.learn_id_column(node)
                for value in node:
                    walk(value, depth + 1, rows)
        walk(observation, 0)

    def schema_labels(self) -> set[str]:
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

    def category_field(self) -> str | None:
        counts: dict[str, set[str]] = {}
        present: dict[str, int] = {}
        for row in self.listings.values():
            for (key, value) in row.items():
                if isinstance(value, str) and 0 < len(value) <= 40 and (not re.search('\\d{3}', value)):
                    counts.setdefault(key, set()).add(value)
                    present[key] = present.get(key, 0) + 1
        total = max(1, len(self.listings))
        known = self.schema_labels()
        best = None
        for (key, values) in counts.items():
            if present[key] < 0.8 * total or len(values) > 40:
                continue
            listed = sum(1 for v in values if v.strip().lower() in known) / len(values)
            mean = sum(len(v) for v in values) / len(values)
            score = (-listed, len(values), -present[key], -mean)
            if best is None or score < best[0]:
                best = (score, key)
        return best[1] if best else None

    def attribute_table(self) -> list[tuple[str, str, int, list[str]]]:
        category_column = self.category_field()
        groups: dict[str, dict[str, Any]] = {}
        for listing_id in self.listing_order:
            record = self.opened_records.get(listing_id)
            if not record:
                continue
            block = _attribute_block(record) or {}
            label = str(record.get(category_column) or 'listings') if category_column else 'listings'
            group = groups.setdefault(label, {'n': 0, 'keys': {}})
            group['n'] += 1
            done: set[str] = set()
            for (key, value) in block.items():
                name = _plain_words(key)
                if not name or name in done:
                    continue
                done.add(name)
                slot = group['keys'].setdefault(name, {'n': 0, 'values': [], 'per': {}})
                slot['n'] += 1
                text = str(value).strip().lower()
                slot['per'][listing_id] = text
                if text not in slot['values']:
                    slot['values'].append(text)
        wanted = [(_plain_words(item.category or ''), _plain_words(item.kind), {w for w in _stems(item.category or '') | _stems(item.kind) if len(w) >= 4}) for item in self.products]
        matched = {label for label in groups if any(_plain_words(label) in (shelf, name) or {w for w in _stems(label) if len(w) >= 4} & long for (shelf, name, long) in wanted) or self.product_of_label(label)}
        if matched:
            groups = {label: group for (label, group) in groups.items() if label in matched}
        table = []
        for (label, group) in sorted(groups.items(), key=lambda kv: -kv[1]['n']):
            kept: list[tuple[str, dict[str, Any]]] = []
            for (name, slot) in sorted(group['keys'].items(), key=lambda kv: (-kv[1]['n'], len(kv[0]))):
                values = list(slot['per'].values())
                if not values:
                    continue
                mean = sum(len(v) for v in values) / len(values)
                if mean > 40 or sum(1 for v in values if ' / ' in v) * 2 >= len(values):
                    continue
                recur = sum(len(set(re.findall('[a-z]{3,}', v)) & set(re.findall('[a-z]{3,}', ' '.join(o for o in values if o is not v)))) > 0 for v in values)
                if len(values) >= 4 and len(set(values)) == len(values) and (mean > 12) and (recur * 2 < len(values)):
                    continue
                twin = next((spec for (other, spec) in kept if self.same_attribute(other, spec, name, slot)), None)
                if twin is not None:
                    for (listing_id, text) in slot['per'].items():
                        twin['per'].setdefault(listing_id, text)
                        if text not in twin['values']:
                            twin['values'].append(text)
                    twin['n'] = len(twin['per'])
                    continue
                kept.append((name, slot))
            kept.sort(key=lambda kv: -kv[1]['n'])
            for (name, slot) in kept[:14]:
                table.append((label, name, slot['n'], [v[:40] for v in slot['values'][:4]]))
        return table

    def other_property_mark(self, label: str, name: str, values: list[str]) -> str:
        product_number = self.product_of_label(label)
        if not product_number:
            return ''
        for must in self.products[product_number - 1].stated:
            (property_name, _, value) = must.partition(':')
            if re.search('\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', value, re.I):
                continue
            words = set(re.findall('[a-z0-9]{3,}', value.lower()))
            shown = set(re.findall('[a-z0-9]{3,}', ' '.join(values).lower()))
            if property_name.strip() and _plain_words(property_name) != name and _stems(property_name) & _stems(name) and words and (not words <= shown):
                return f' [not the stated {property_name.strip()}: another property]'
        return ''

    @staticmethod
    def same_attribute(name: str, spec: dict[str, Any], other: str, slot: dict[str, Any]) -> bool:
        (a, b) = (set(name.split()), set(other.split()))
        if not a or not b or (not a & b):
            return False

        def agree(x: str, y: str) -> bool:
            (wx, wy) = (re.findall('[a-z0-9]+', x.lower()), re.findall('[a-z0-9]+', y.lower()))
            (short, long) = (wx, wy) if len(wx) <= len(wy) else (wy, wx)
            return x == y or (bool(short) and set(short) <= set(long) and (len(long) <= 2 * len(short) + 2))
        both = [i for i in spec['per'] if i in slot['per'] and spec['per'][i].strip() and slot['per'][i].strip()]
        same = sum(1 for i in both if agree(spec['per'][i], slot['per'][i]))
        if not (a <= b or b <= a):
            return len(both) >= 2 and same == len(both)
        if both:
            return same * 2 > len(both)
        return bool((set(spec['values']) & set(slot['values'])) - {''})

    def listings_by_product(self) -> list[list[str]]:
        category_column = self.category_field()
        groups: dict[str, list[str]] = {}
        for listing_id in self.listing_order:
            label = str(self.listings[listing_id].get(category_column) or '') if category_column else ''
            groups.setdefault(label, []).append(listing_id)
        wanted = [(_plain_words(item.category or ''), _plain_words(item.kind), {w for w in _stems(item.category or '') | _stems(item.kind) if len(w) >= 4}) for item in self.products]
        matched = [ids for (label, ids) in groups.items() if any(_plain_words(label) in (shelf, name) or {w for w in _stems(label) if len(w) >= 4} & long for (shelf, name, long) in wanted) or self.product_of_label(label)]
        pool = matched or list(groups.values())
        return sorted(pool, key=lambda ids: -len(ids))[:max(1, len(self.products))]

    def sole_id_argument(self, tool: str | None) -> str | None:
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for (k, v) in props.items() if isinstance(v, dict) and 'string' in _schema_types(v) and ('enum' not in v)]
        target = next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not tool or not target or any(k != target for k in schema.get('required') or []):
            return None
        return target

    def price(self, listing_id: str) -> float | None:
        value = (self.listings.get(listing_id) or {}).get(self.price_field) if self.price_field else None
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and (not isinstance(v, bool))), None)
        if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', value):
            value = float(value)
        return float(value) if isinstance(value, (int, float)) and (not isinstance(value, bool)) else None

    def purchasable(self, listing_id: str) -> bool:
        if self.is_sold_out(listing_id):
            return False
        value = (self.listings.get(listing_id) or {}).get(self.stock_field) if self.stock_field else None
        return not (value is False or (isinstance(value, (int, float)) and (not isinstance(value, bool)) and (value <= 0)))

    def listings_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        if isinstance(observation, dict):
            listing_id = self.listing_id_of(observation)
            if listing_id:
                return [(listing_id, observation)]
            found: list[tuple[str, dict[str, Any]]] = []
            for value in observation.values():
                if isinstance(value, list):
                    found += [(i, r) for r in value if isinstance(r, dict) for i in [self.listing_id_of(r, True)] if i]
                elif isinstance(value, dict):
                    found += [(i, r) for v in value.values() if isinstance(v, list) for r in v if isinstance(r, dict) for i in [self.listing_id_of(r, True)] if i]
            return found
        if isinstance(observation, list):
            return [(i, r) for r in observation if isinstance(r, dict) for i in [self.listing_id_of(r, True)] if i]
        return []

    def note_fields_shown(self, name: str, observation: Any) -> None:
        for (listing_id, record) in self.listings_in(observation):
            keys = set(record)
            self.fields_by_tool.setdefault(name, set()).update(keys)
            (version, have) = self.fields_seen.get(listing_id, (-1, set()))
            self.fields_seen[listing_id] = (self.task_version, (have if version == self.task_version else set()) | keys)

    def fully_shown(self, name: str, listing_id: str) -> bool:
        (version, have) = self.fields_seen.get(listing_id, (-1, set()))
        keys = self.fields_by_tool.get(name)
        return bool(keys) and version == self.task_version and (keys <= have) and (listing_id not in self.sold_out_step)

    def learn_result_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        if name in self.result_shapes:
            return self.result_shapes[name]
        if not isinstance(observation, dict):
            return None
        lists = [k for (k, v) in observation.items() if isinstance(v, list) and len(v) >= 2 and all(isinstance(r, dict) for r in v)]
        if not lists:
            return None
        self.result_shapes[name] = None
        sample = {k: v[:3] if k in lists else v for (k, v) in observation.items()}
        prompt = SHAPE_PROMPT + '\n\nTOOL: ' + name + ' | parameters: ' + _compact(self.schemas.get(name) or {}) + '\n\nRESULT (first rows only):\n' + _compact(_clipped(sample, 120))[:4000]
        json_reply = self.json_answer(prompt, [FAST_MODEL, STRONG_MODEL], 2500) or {}
        rows = json_reply.get('rows') if json_reply.get('rows') in lists else None
        if not rows:
            return None
        first = observation[rows][0]
        props = (self.schemas.get(name) or {}).get('properties') or {}

        def named(key: Any, where: dict[str, Any]) -> str | None:
            return key if isinstance(key, str) and key in where else None
        shape = {'rows': rows, 'count': named(json_reply.get('count'), observation), 'next': named(json_reply.get('next'), observation), 'next_arg': named(json_reply.get('next_arg'), props), 'price': named(json_reply.get('price'), first), 'stock': named(json_reply.get('stock'), first)}
        if shape['count'] and (isinstance(observation[shape['count']], bool) or not isinstance(observation[shape['count']], int)):
            shape['count'] = None
        self.result_shapes[name] = shape
        self.price_field = self.price_field or shape['price']
        self.stock_field = self.stock_field or shape['stock']
        self.trace.append({'kind': 'shape', 'tool': name, 'shape': shape})
        return shape

    def page_plan(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        shape = self.learn_result_shape(name, observation)
        if not shape or not shape['count'] or (not shape['next']) or (not shape['next_arg']) or (not isinstance(observation, dict)):
            return None
        (rows, count, value) = (observation.get(shape['rows']), observation.get(shape['count']), observation.get(shape['next']))
        if not isinstance(rows, list) or not rows or isinstance(count, bool) or (not isinstance(count, int)):
            return None
        if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))):
            return None
        return (count, len(rows), shape['next_arg'], value)

    def rows_by_price(self, observation: Any) -> bool:
        prices = [self.price(i) for (i, _) in self.listings_in(observation)]
        return len(prices) >= 3 and None not in prices and all(a <= b for (a, b) in zip(prices, prices[1:]))

    def read_further_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        self.paged_results = []
        pick = self.reading.get('pick')
        plan = self.page_plan(name, observation) if pick in ('cheapest', 'most', 'least') else None
        if not plan:
            return ''
        (count, shown, param, value) = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and (not isinstance(given, bool)) else bool(str(given or '').strip())
        if value is None or later or count <= shown:
            return ''
        if pick == 'cheapest' and self.rows_by_price(observation):
            return ''
        pages = -(-(count - shown) // shown)
        if pages > 2 and self.pages_read_by_code + 2 <= PAGES_READ_BY_CODE and (self.steps_left() - self.final_steps() - 2 >= 12):
            pages = 2
        if pages > 2 or self.pages_read_by_code + pages > PAGES_READ_BY_CODE or self.steps_left() - self.final_steps() - pages < 12:
            return f"\n[Oradon] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with {param}={_compact(value)}."
        texts = []
        for _ in range(pages):
            shape_problems: list[str] = []
            call = _coerce_arguments({**args, param: value}, self.schemas.get(name), 'args', shape_problems)
            if shape_problems:
                break
            reply = self.store_call(name, call)
            if reply is None or self.finished:
                break
            self.pages_read_by_code += 1
            seen = reply.get('observation')
            self.take_result(name, seen)
            self.paged_results.append(seen)
            shown_text = self.result_text(seen)
            self.result_cache[name + _compact(call)] = (self.task_version, self.step, shown_text)
            texts.append(f'{param}={_compact(value)}: {shown_text}')
            after = self.page_plan(name, seen)
            value = after[3] if after else None
            if value is None:
                break
        self.trace.append({'kind': 'pages', 'tool': name, 'read': len(texts)})
        rest = f"\n[Oradon] {count} listings matched and Oradon has read {shown * (len(texts) + 1)} rows of them; narrow the search with the requirement's value or page on to see the rest." if value is not None and texts else ''
        return '\n[Oradon read the next pages of this search itself, one step a page]\n' + '\n'.join(texts) + rest if texts else ''

    def price_cap_note(self, name: str, args: dict[str, Any], observation: Any) -> str:
        props = (self.schemas.get(name) or {}).get('properties') or {}

        def priced(key: str) -> bool:
            spec = props.get(key) if isinstance(props.get(key), dict) else {}
            return bool(re.search('price|cost|amount|budget|spend|money', key + ' ' + str(spec.get('description') or ''), re.I))
        caps = [k for (k, v) in args.items() if k in props and isinstance(v, (int, float)) and (not isinstance(v, bool)) and re.search('max|most|under|below|limit|cap|upto|up_to', k, re.I) and priced(k)]
        if not caps or not self.price_field:
            return ''
        caps.sort(key=lambda k: args[k])
        cap = args[caps[0]]
        over = [i for (i, _) in self.listings_in(observation) if (self.price(i) or 0) > cap]
        if not over:
            return ''
        modes = {k: v for (k, v) in args.items() if isinstance(v, str) and isinstance((props.get(k) or {}).get('enum'), list)}
        others = {k: [o for o in props[k]['enum'] if o != v] for (k, v) in modes.items()}
        hint = '; list the category with ' + ', '.join(f'{k}={_compact(o[0])}' for (k, o) in others.items() if o) + f' and the same {caps[0]} to see only cheaper listings' if any(others.values()) else ''
        return f'\n[Oradon] This search did not apply {caps[0]}={cap:g}: {len(over)} of its rows cost more{hint}.'

    def cheapest_rows_note(self, observations: list[Any]) -> str:
        if self.reading.get('pick') != 'cheapest' or not self.price_field:
            return ''
        found = {i for seen in observations for (i, _) in self.listings_in(seen)}
        rows = [(self.price(i), i) for i in found if self.purchasable(i) and self.price(i) is not None]
        if len(rows) < 2:
            return ''
        return '\n[Oradon] buyable rows here, cheapest first: ' + ', '.join(f'{i} ({p:g})' for (p, i) in sorted(rows)[:6])

    def wider_search_hold(self) -> str:
        if self.wider_search_held or self.reading.get('pick') != 'cheapest' or self.steps_left() < 14 + 2 * len(self.products):
            return ''
        self.wider_search_held = True
        lines = [f'{i} at {self.price(i):g}' for i in self.cart_listing_ids() if self.price(i) is not None]
        return f"Not run yet (no step used). {self.steps_left()} steps are left and the pick is the cheapest. Searches match a listing's title, so a requirement's value (often only in the attributes) finds few of the fitting listings, and rows come in no price order. Before checking out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line" + (f" ({'; '.join(lines)})" if lines else '') + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'

    def open_attributes(self, ids: list[str], keep: int) -> list[str]:
        tool = self.attributes_tool()
        target = self.sole_id_argument(tool)
        if not tool or not target:
            return []
        texts = []
        for listing_id in [i for i in ids if i in self.listings and i not in self.opened_records and (not self.is_sold_out(i))]:
            if self.steps_left() <= keep or self.finished or self.attributes_shown_by.get(tool) is False:
                break
            shape_problems: list[str] = []
            args = _coerce_arguments({target: listing_id}, self.schemas.get(tool), 'args', shape_problems)
            if shape_problems:
                break
            reply = self.store_call(tool, args)
            if reply is None:
                break
            seen = reply.get('observation')
            self.take_result(tool, seen)
            self.note_fields_shown(tool, seen)
            if isinstance(seen, dict) and self.listing_id_of(seen):
                self.attributes_shown_by[tool] = self.attributes_shown_by.get(tool, False) or bool(_attribute_block(seen))
            texts.append(f'{listing_id}: ' + self.result_text(seen)[:1500])
        if texts:
            self.trace.append({'kind': 'fill_facts', 'n': len(texts)})
        return texts

    def open_cheaper_rows(self) -> list[str]:
        if self.reading.get('pick') != 'cheapest':
            return []
        signature = self.cart_signature() + f'#{self.task_version}'
        category_column = self.category_field()
        if signature in self.cheaper_rows_done or not category_column:
            return []
        self.cheaper_rows_done.add(signature)
        cart = self.cart_listing_ids()
        selected: list[str] = []
        for line in cart:
            price = self.price(line)
            kind = str((self.listings.get(line) or {}).get(category_column) or '').strip().lower()
            if price is None or not kind:
                continue
            pool = [i for i in self.listing_order if i not in cart and i not in self.opened_records and self.purchasable(i) and (str((self.listings.get(i) or {}).get(category_column) or '').strip().lower() == kind) and ((self.price(i) if self.price(i) is not None else float('inf')) < price)]
            pool.sort(key=lambda i: self.price(i) or 0.0)
            selected += [i for i in pool[:2] if i not in selected]
        return self.open_attributes(selected[:CHEAPER_ROWS_OPENED], self.final_steps() + 4)

    def take_result(self, name: str, observation: Any) -> str:
        self.take_in(observation)
        lines = self.notice_lines(observation)
        if not lines:
            return ''
        meaning = self.read_notice(lines)
        if meaning:
            self.queued_notes.append(f'(seen on {name}) {meaning}')
        return meaning

    def open_candidates(self) -> None:
        tool = self.roles.get('compare')
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        target = next((k for (k, v) in props.items() if isinstance(v, dict) and 'array' in _schema_types(v)), None)
        if not tool or not target or any(k != target for k in schema.get('required') or []) or self.compare_is_bare():
            self.open_candidates_one_by_one()
            return
        if self.steps_left() <= 12:
            return
        cap = props[target].get('maxItems') if isinstance(props[target].get('maxItems'), int) else 4
        ranked = self.listings_by_product()
        share = max(1, cap // max(1, len(ranked)))
        selected: list[str] = []
        for ids in ranked:
            selected += [i for i in ids if i not in self.opened_records][:share]
        selected = selected[:cap]
        if len(selected) < 2:
            return
        shape_problems: list[str] = []
        args = _coerce_arguments({target: selected}, schema, 'args', shape_problems)
        if shape_problems:
            return
        before = len(self.opened_records)
        reply = self.store_call(tool, args)
        if reply is not None:
            seen = reply.get('observation')
            self.take_result(tool, seen)
            self.note_fields_shown(tool, seen)
            rows = self.listings_in(seen)
            if rows:
                self.attributes_shown_by[tool] = self.attributes_shown_by.get(tool, False) or any(_attribute_block(r) for (_, r) in rows)
            self.trace.append({'kind': 'widen', 'ids': selected, 'opened': len(self.opened_records) - before})
            if rows and self.attributes_shown_by.get(tool) is False:
                self.open_candidates_one_by_one()
                return
            if len(self.opened_records) == before and len(selected) > 2 and (self.steps_left() > 12):
                shape_problems = []
                args = _coerce_arguments({target: selected[:2]}, schema, 'args', shape_problems)
                if not shape_problems:
                    reply = self.store_call(tool, args)
                    if reply is not None:
                        self.take_result(tool, reply.get('observation'))

    def open_candidates_one_by_one(self) -> None:
        tool = self.attributes_tool()
        target = self.sole_id_argument(tool)
        if not tool or not target:
            return
        wanted = {w for (_, requirement) in self.live_requirements() for w in re.findall('[a-z0-9]{3,}', requirement.split(':', 1)[-1].lower())}
        each = OPENS_PER_PRODUCT if len(self.products) <= 2 else 4
        question_lanes: list[list[str]] = []
        heads: list[int] = []
        for ids in self.listings_by_product():
            opened = sum(1 for i in ids if i in self.opened_records)
            pool = [i for i in ids if i not in self.opened_records and self.purchasable(i)]
            if self.reading.get('pick') == 'cheapest':
                pool.sort(key=lambda i: self.price(i) if self.price(i) is not None else float('inf'))
            pool.sort(key=lambda i: -len(wanted & set(re.findall('[a-z0-9]{3,}', _compact(self.listings.get(i) or {}).lower()))))
            question_lanes.append(pool[:max(0, each - opened)])
            heads.append(max(0, 4 - opened))
        first = [question_lane[k] for k in range(4) for (question_lane, head) in zip(question_lanes, heads) if k < min(head, len(question_lane))]
        more = [question_lane[k] for k in range(each) for (question_lane, head) in zip(question_lanes, heads) if head <= k < len(question_lane)]
        cap = max(5, 2 * len(question_lanes) + 2)
        read = self.open_attributes(first[:cap][:8], 22)
        self.open_attributes((first[cap:] + more)[:max(0, 10 - len(read))], 30)

    def interview_due(self) -> bool:
        if self.interviewed or not self.roles.get('ask') or self.messages_free() <= 0:
            return False
        if self.steps_left() <= 8 or self.wrapping_up():
            return False
        opened = len(self.opened_records)
        if opened >= 3 * len(self.products) + 1:
            return True
        return self.step >= max(8, self.max_steps // 4)

    def message_arguments(self, text: str) -> dict[str, Any] | None:
        name = self.roles.get('ask')
        schema = self.schemas.get(name or '') or {}
        props = schema.get('properties') or {}
        texts = [k for (k, v) in props.items() if isinstance(v, dict) and 'string' in _schema_types(v) and ('enum' not in v)]
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
        shape_problems: list[str] = []
        args = _coerce_arguments(args, schema, 'args', shape_problems)
        return None if shape_problems else args

    def run_interview(self) -> str:
        self.interviewed = True
        if self.messages_free() <= 0:
            return ''
        self.open_candidates()
        question_lanes = self.plan_questions(self.messages_free())
        found = {n: 0 for n in question_lanes}
        spent = {n: 0 for n in question_lanes}
        used: dict[int, set[str]] = {n: set() for n in question_lanes}
        follow: list[tuple[int, list[str], bool]] = []
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        while self.messages_free() > 0 and (not self.finished) and (self.steps_left() > 6) and (not self.wrapping_up()):
            if follow:
                (product_number, labels, shown) = follow.pop(0)
            else:
                (product_number, labels) = self.next_question(question_lanes, found, spent, used)
                shown = False
                if not labels:
                    break
            used[product_number].update(_plain_words(label) for label in labels)
            batch = self.send_messages([(product_number, labels[0], self.question_text(product_number, labels, shown))])
            if not batch:
                break
            sent += batch
            spent[product_number] += 1
            had = [len(item.revealed) for item in self.products]
            heard = self.read_answers(batch, (FAST_MODEL, STRONG_MODEL))
            self.apply_answers(heard, told)
            fresh = [n for (n, item) in enumerate(self.products, 1) if len(item.revealed) > had[n - 1]]
            for n in fresh:
                found[n] = found.get(n, 0) + 1
            if fresh or 1 not in [_as_count(x) for x in _as_list(heard.get('crowded'))]:
                continue
            head = labels[0].split()[-1]
            if len(labels[0].split()) >= 2 and _plain_words(head) not in used[product_number]:
                follow.insert(0, (product_number, [head], False))
            elif not shown and len(self.sample_values(product_number, labels[0])) >= 2:
                follow.insert(0, (product_number, labels, True))
        found_items = [n for n in found if found[n] > 0]
        searched = self.search_revealed_values(found_items) if found_items else ''
        if told:
            self.task_version += 1
        lines = [f'Q: {q}\nA: {a}' for (_, q, a) in sent]
        summary = "Oradon's interviewer asked the shopper (each question used one step):\n" + '\n'.join(lines)
        summary += '\nNew firm requirements from these answers: ' + ('; '.join(told) if told else 'none')
        summary += '. Apply them to every pick (search again with the new values where needed).'
        pointed = [requirement for (_, requirement) in self.pointer_requirements if any(requirement in t for t in told)]
        if pointed:
            summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.'
        if searched:
            summary += '\n' + searched
        if self.queued_notes:
            summary += '\n[notes seen meanwhile] ' + ' | '.join(self.queued_notes)
            self.queued_notes = []
        return summary

    def search_form(self) -> tuple[str, dict[str, Any], str] | None:
        tool = self.roles.get('search')
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for (k, v) in props.items() if isinstance(v, dict) and 'string' in _schema_types(v) and ('enum' not in v)]
        named = [k for k in texts if re.search('(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword', k, re.I)]
        for args in reversed(self.searches_made):
            fields = [k for k in named or texts if isinstance(args.get(k), str) and re.search('[a-z]', args[k], re.I)]
            if not named:
                labels = {str(r.get(self.category_field()) or '').strip().lower() for r in self.listings.values()} - {''} if self.category_field() else set()
                fields = [k for k in fields if len({str(a.get(k)) for a in self.searches_made if a.get(k)}) >= 2 and any(len(str(a.get(k)).split()) >= 2 for a in self.searches_made) and (not any(str(a.get(k) or '').strip().lower() in labels for a in self.searches_made))]
            if fields and tool:
                field = fields[0]
                keep = {k: v for (k, v) in args.items() if k != field and (isinstance((props.get(k) or {}).get('enum'), list) or k in (schema.get('required') or []))}
                category_column = self.category_field()
                shelves = {_plain_words(item.category or '') for item in self.products} | {_plain_words(item.kind) for item in self.products}
                shelves |= {_plain_words(str(row.get(category_column) or '')) for row in self.listings.values()} if category_column else set()
                keep = {k: v for (k, v) in keep.items() if k in (schema.get('required') or []) or not isinstance(v, str) or (_plain_words(v) not in shelves and (not any({w for w in _stems(v) if len(w) >= 4} & {w for w in _stems(s) if len(w) >= 4} for s in shelves if s)))}
                worded = [a for a in self.searches_made if isinstance(a.get(field), str) and re.search('[a-z]', a[field], re.I)]
                for k in list(keep):
                    if isinstance((props.get(k) or {}).get('enum'), list):
                        used = [_compact(a[k]) for a in worded if k in a]
                        if used:
                            best = max(used, key=used.count)
                            keep[k] = next(a[k] for a in worded if k in a and _compact(a[k]) == best)
                return (tool, keep, field)
        return None

    def value_terms(self, product_number: int) -> list[str]:
        out = []
        for (n, requirement) in self.live_requirements():
            if n != product_number or requirement in [m for (_, m) in self.pointer_requirements] or ':' not in requirement:
                continue
            value = requirement.split(':', 1)[1].strip()
            if not value or re.search('\\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', value, re.I):
                continue
            out.append(value)
        return out

    def search_revealed_values(self, numbers: list[int]) -> str:
        plan = self.search_form()
        if not plan or self.steps_left() < 22:
            return ''
        (tool, base, field) = plan
        seen: list[str] = []
        budget = 8
        for product_number in numbers:
            values = self.value_terms(product_number)
            item = self.products[product_number - 1]
            fresh = [v for v in values if any(':' in t and t.split(':', 1)[1].strip() == v for t in item.revealed)]
            if not fresh or budget <= 2:
                continue
            stated = [i for (i, v) in enumerate(values) if v not in fresh]
            drop = stated[-1] if stated else 0
            tries = [values, values[:drop] + values[drop + 1:]] if len(values) > 1 else [values]
            for words in tries[:2]:
                if not words:
                    continue
                shape_problems: list[str] = []
                args = _coerce_arguments({**base, field: ' '.join(words)}, self.schemas.get(tool), 'args', shape_problems)
                if shape_problems or self.finished:
                    break
                reply = self.store_call(tool, args)
                budget -= 1
                if reply is None:
                    break
                observation = reply.get('observation')
                self.take_result(tool, observation)
                category_column = self.category_field()
                found = [i for (i, _) in self.listings_in(observation)]
                cap = self.budget if isinstance(self.budget, (int, float)) else None
                prices = [self.price(i) for i in found]
                if cap is not None and prices and all(p is not None and p > 10 * cap for p in prices):
                    cap = None
                rows = [i for i in found if self.purchasable(i) and (not category_column or self.label_belongs_to(item, str((self.listings.get(i) or {}).get(category_column) or ''))) and (cap is None or (self.price(i) or 0) <= cap)]
                seen.append(f'{_compact(args.get(field))}: ' + (', '.join(rows[:6]) or 'no row of that kind'))
                if rows:
                    opened = self.open_attributes([i for i in rows if i not in self.opened_records][:min(3, budget)], self.final_steps() + 16)
                    budget -= len(opened)
                    if opened:
                        seen.append('opened: ' + ' | '.join(opened)[:2400])
                    break
        if not seen:
            return ''
        self.trace.append({'kind': 'value_search', 'seen': [s[:200] for s in seen]})
        return "[Oradon searched each such item's requirement values together and read the first fitting-kind rows] " + '\n'.join(seen)

    def next_question(self, question_lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> tuple[int, list[str]]:
        first = {n: i for (i, n) in enumerate(self.hinted_products)}
        for product_number in sorted(question_lanes, key=lambda n: (found.get(n, 0) > 0, spent.get(n, 0), first.get(n, len(first) + n))):
            left = [label for label in question_lanes[product_number] if _plain_words(label) not in used[product_number]]
            if left:
                return (product_number, left[:1])
        return (0, [])

    def question_text(self, product_number: int, labels: list[str], shown: bool=False) -> str:
        if len(labels) >= 2:
            return f'What {labels[0]} or {labels[1]} would you like for the {self.products[product_number - 1].kind}?'
        return self.phrase_question(product_number, labels[0], self.sample_values(product_number, labels[0]) if shown else None)

    def product_keys(self, product_number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        item = self.products[product_number - 1]
        settled = [_stems(m.split(':', 1)[0]) for m in item.stated]
        rows = [(label, key, carried, len(values)) for (label, key, carried, values) in table if self.product_of_label(label) == product_number]
        measured = {key: sum(1 for v in values if re.match('^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$', v)) * 2 > len(values) for (_, key, _, values) in table}
        if len(self.products) == 1 and (not rows):
            rows = [(label, key, carried, len(values)) for (label, key, carried, values) in table]
        rows.sort(key=lambda r: (-r[2], measured.get(r[1], False), -r[3]))
        keys = [key for (_, key, _, _) in rows if not any(_names_property(property_name, _stems(key)) for property_name in settled) and (not re.search(NOT_A_PROPERTY, key))]
        return [key for key in keys if not re.search(BUNDLE_KEY, key)] + [key for key in keys if re.search(BUNDLE_KEY, key)]

    def product_of_label(self, label: str) -> int:
        if not label:
            return 0
        words = _plain_words(label)
        long = _root_words(label)
        (best, top) = (0, 0.0)
        for (product_number, item) in enumerate(self.products, 1):
            if words == _plain_words(item.category or ''):
                score = 3.0
            elif words == _plain_words(item.kind):
                score = 2.0
            else:
                mine = _root_words(item.category or '') | _root_words(item.kind)
                score = len(long & mine) / max(1, len(long | mine))
            if score > top:
                (best, top) = (product_number, score)
        return best

    def label_belongs_to(self, item: Product, label: str) -> bool:
        return bool(label) and item in self.products and (self.product_of_label(label) == self.products.index(item) + 1)

    def opened_count(self, product_number: int) -> int:
        category_column = self.category_field()
        item = self.products[product_number - 1]
        return sum(1 for record in self.opened_records.values() if category_column and self.label_belongs_to(item, str(record.get(category_column) or '')) and _attribute_block(record))

    def plan_questions(self, count: int) -> dict[int, list[str]]:
        table = self.attribute_table()
        settled = []
        for (product_number, item) in enumerate(self.products, 1):
            settled.append(f'item {product_number}: {item.kind} - settled by the request: ' + ('; '.join(item.stated) or 'nothing'))
        listing = []
        last = None
        for (index, (label, name, carried, values)) in enumerate(table, 1):
            if label != last:
                listing.append(f'category {label!r}:')
                last = label
            listing.append(f'  {index}. {name} (on {carried}): ' + ' | '.join(values) + self.other_property_mark(label, name, values) + (' [several features in one: ask last]' if re.search(BUNDLE_KEY, name) else ''))
        prompt = PLAN_PROMPT.replace('COUNT', str(max(1, count))) + '\n\nREQUEST:\n' + self.request + '\n\nITEMS:\n' + '\n'.join(settled)
        first = [str(x) for x in _as_list(self.reading.get('ask_first')) if str(x).strip()]
        if first:
            prompt += '\n\nASK FIRST (the request asks to be asked about these):\n' + '\n'.join(first)
        prompt += '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(listing) or '(no full listing records yet)')
        plan = self.json_answer(prompt, [STRONG_MODEL, FAST_MODEL], 3500) or {}
        question_lanes: dict[int, list[str]] = {n: [] for n in range(1, len(self.products) + 1)}
        hinted: dict[int, set[str]] = {n: set() for n in question_lanes}
        for pick in _as_list(plan.get('ask')):
            if not isinstance(pick, dict):
                continue
            product_number = min(max(1, _as_count(pick.get('item') or 1) or 1), len(self.products))
            label = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(pick.get('label') or ''))[0].strip().rstrip('?.')
            if not label or len(label.split()) > 5:
                continue
            if all(_plain_words(label) != _plain_words(other) for other in question_lanes[product_number]):
                question_lanes[product_number].append(label)
            if str(pick.get('hint') or '').strip():
                hinted[product_number].add(_plain_words(label))
                if product_number not in self.hinted_products:
                    self.hinted_products.append(product_number)
        for product_number in question_lanes:
            keys = self.product_keys(product_number, table)
            read = max(1, self.opened_count(product_number))
            own: dict[str, float] = {}
            for (label, key, carried, _) in table:
                if self.product_of_label(label) == product_number:
                    own[key] = max(own.get(key, 0.0), carried / read)

            def spelled(label: str) -> str:
                plain = lambda text: re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', _plain_words(text)))
                return next((k for k in own if plain(k) == plain(label)), label)

            def share_of(label: str) -> float:
                if _plain_words(label) in own:
                    return own[_plain_words(label)]
                near = [s for (k, s) in own.items() if _names_property(_stems(label), _stems(k)) or _names_property(_stems(k), _stems(label))]
                return max(near) if near else 0.0
            hinted[product_number] = {_plain_words(spelled(h)) for h in hinted[product_number]}
            question_lane = []
            for pick in question_lanes[product_number]:
                if all(_plain_words(spelled(pick)) != _plain_words(other) for other in question_lane):
                    question_lane.append(spelled(pick))
            for key in [k for k in keys if own.get(k, 0.0) >= CARRIED_SHARE_MIN and (not re.search(BUNDLE_KEY, k))][:CARRIED_KEYS_KEPT]:
                if any(_plain_words(key) == _plain_words(other) for other in question_lane[:3]):
                    continue
                weak = next((i for (i, other) in enumerate(question_lane[:3]) if _plain_words(other) not in hinted[product_number] and share_of(other) + CARRIED_SHARE_GAP <= own.get(key, 0.0)), None)
                if weak is None and len(question_lane) >= 3:
                    continue
                placed = [other for other in question_lane if _plain_words(other) != _plain_words(key)]
                if weak is not None and len(self.products) > 1:
                    moved = placed[weak]
                    placed[weak] = key
                    placed.insert(min(3, len(placed)), moved)
                else:
                    placed.insert(min(len(placed), weak if weak is not None else len(placed)), key)
                if any(_plain_words(x) in hinted[product_number] and all(_plain_words(x) != _plain_words(y) for y in placed[:3]) for x in question_lane[:3]):
                    continue
                question_lane = placed
            question_lanes[product_number] = question_lane
            for key in keys:
                if all(_plain_words(key) != _plain_words(other) for other in question_lanes[product_number]):
                    question_lanes[product_number].append(key)
            question_lanes[product_number] = [x for x in question_lanes[product_number] if not re.search(BUNDLE_KEY, _plain_words(x))] + [x for x in question_lanes[product_number] if re.search(BUNDLE_KEY, _plain_words(x))]
        self.trace.append({'kind': 'plan', 'ask': _clipped(plan.get('ask'), 120), 'lanes': {n: v[:8] for (n, v) in question_lanes.items()}})
        return question_lanes

    def phrase_question(self, product_number: int, label: str, samples: list[str] | None=None) -> str:
        text = f'What {label} would you like for the {self.products[product_number - 1].kind}'
        return text + (f', for example {samples[0]} or {samples[1]}?' if samples else '?')

    def sample_values(self, product_number: int, label: str) -> list[str]:
        want = _plain_words(label)
        settled = ' '.join(self.products[product_number - 1].stated).lower()
        for (_, name, _, values) in self.attribute_table():
            if name == want:
                return [v for v in values if v and v not in settled and (len(v) <= 30)][:2]
        return []

    def send_messages(self, questions: list[tuple[int, str, str]], floor: int=6) -> list[tuple[int, str, str]]:
        tool = self.roles['ask']
        sent: list[tuple[int, str, str]] = []
        for (product_number, _, text) in questions:
            if self.finished or self.steps_left() <= floor or self.messages_sent >= self.message_limit:
                break
            args = self.message_arguments(text)
            if not args:
                break
            reply = self.store_call(tool, args)
            if reply is None:
                break
            answer = reply.get('observation')
            if isinstance(answer, dict) and answer.get('error'):
                fault = ' '.join(str(answer.get(k) or '') for k in ('error', 'detail', 'message'))
                if re.search('budget|exceed|limit|no more|used up|too many|exhaust', fault, re.I):
                    self.message_limit = self.messages_sent  # the store takes no more messages
                self.trace.append({'kind': 'message_refused', 'q': text, 'error': fault[:200], 'limit': self.message_limit})
                break
            self.messages_sent += 1
            words = self.reply_words(answer)
            sent.append((product_number, text, words))
            self.action_record.append(f'step {self.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {self.cart_signature()})')
            self.action_record_steps.append(self.step)
            self.trace.append({'kind': 'ask', 'q': text, 'a': words[:300]})
        return sent

    def apply_answers(self, heard: dict[str, Any], told: list[str]) -> None:
        many = len(self.products) > 1
        for entry in _as_list(heard.get('needs')):
            if not isinstance(entry, dict) or not str(entry.get('need') or '').strip():
                continue
            product_number = min(max(1, _as_count(entry.get('item')) or 1), len(self.products))
            requirement = str(entry['need']).strip()
            if requirement not in self.products[product_number - 1].revealed:
                self.products[product_number - 1].revealed.append(requirement)
                told.append(f'item {product_number}: {requirement}' if many else requirement)
            if entry.get('points') is True and (product_number, requirement) not in self.pointer_requirements:
                self.pointer_requirements.append((product_number, requirement))
        for entry in _as_list(heard.get('drop')):
            if isinstance(entry, dict) and str(entry.get('property') or '').strip():
                item = self.products[min(max(1, _as_count(entry.get('item')) or 1), len(self.products)) - 1]
                gone = _plain_words(str(entry['property']).split(':')[0])
                hit = [m for m in item.stated + item.revealed if m not in item.waived and _plain_words(m.split(':')[0]) == gone]
                item.waived.extend(hit)
                told.extend(f'dropped {m}' for m in hit)
        for entry in _as_list(heard.get('free')):
            if isinstance(entry, dict) and entry.get('property'):
                product_number = min(max(1, _as_count(entry.get('item')) or 1), len(self.products))
                item = self.products[product_number - 1]
                item.indifferent.append(str(entry['property']))
                # "not fussed" about a property the request (or an earlier answer) requires: that requirement is waived
                gone = _stems(str(entry['property']).split(':')[0])
                hit = [m for m in item.stated + item.revealed if m not in item.waived and gone and _stems(m.split(':')[0]) == gone]
                if hit:
                    item.waived.extend(hit)
                    told.extend(f'dropped {m}' for m in hit)
        if isinstance(heard.get('budget'), (int, float)) and (not isinstance(heard.get('budget'), bool)) and (heard['budget'] > 0):
            self.budget = heard['budget']

    def read_answers(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(STRONG_MODEL, FAST_MODEL)) -> dict[str, Any]:
        prompt = ANSWERS_PROMPT + '\n\nITEMS:\n' + '\n'.join(f'item {n}: {it.kind}' for (n, it) in enumerate(self.products, 1))
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join(f'{i}. (about item {n}) Q: {q}\nA: {a}' for (i, (n, q, a)) in enumerate(sent, 1))
        return self.json_answer(prompt, list(models), 2500) or {}

    def process_gate(self, process_stage: str) -> tuple[bool, str]:
        steps = [p for p in self.process_steps if p['when'] == process_stage]
        if not steps or self.finished or (not self.process_due(process_stage)):
            return (True, '')
        ending = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        tools = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_compact(t['function'].get('parameters') or {})}" for t in self.tools if t['function']['name'] not in ending]
        said: list[str] = []
        reminded = False
        for _ in range(PROCESS_ROUNDS):
            prompt = PROCESS_PROMPT.replace('STAGE', 'place the order' if process_stage == 'order' else 'end without a purchase') + '\n\nREQUEST:\n' + self.request + '\n\nSTEPS DUE NOW:\n' + '\n'.join(f"- {p['step']}" for p in steps) + '\n\nLEDGER:\n' + self.requirements_text() + '\n\nCART NOW: ' + self.cart_description() + '\n\nRECORD (messages and step calls so far):\n' + ('\n'.join(f'{i}. {line}' for (i, line) in enumerate(self.action_record, 1)) or '(nothing yet)') + ('\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.changes[-4:]) if self.changes else '') + f"\n\nMESSAGES LEFT: {max(0, self.message_limit - self.messages_sent)}   STEPS LEFT: {self.steps_left()} (the {('order' if process_stage == 'order' else 'ending')} itself takes one)" + '\n\nTOOLS:\n' + '\n'.join(tools)
            json_reply = self.json_answer(prompt, [STRONG_MODEL, FAST_MODEL] if reminded else [FAST_MODEL, STRONG_MODEL], 3500)
            if json_reply is None:
                break
            told: list[str] = []
            hold = ''
            if said:
                self.apply_answers({'needs': json_reply.get('needs')}, told)
                hold = str(json_reply.get('hold') or '').strip()
            call = json_reply.get('call') if isinstance(json_reply.get('call'), dict) else None
            self.trace.append({'kind': 'gate', 'stage': process_stage, 'call': _clipped(call, 200), 'steps': _clipped(json_reply.get('steps'), 120), 'hold': hold[:200], 'told': told})
            if told or hold:
                self.task_version += 1
                return (False, ('Before going on, Oradon did the steps the request asks for:\n' + '\n'.join(said) + '\n' if said else '') + ("The shopper's reply changes the task: " + '; '.join(told + ([hold] if hold else [])) + '. Fix the cart for it, then check out again.'))
            name = str((call or {}).get('tool') or '')
            shown = [e for e in _as_list(json_reply.get('steps')) if isinstance(e, dict)]
            needs_message = process_stage == 'order' and any(p['messages'] > 0 for p in steps)
            lines = range(1, len(self.action_record) + 1)
            proof = {i for i in lines if not needs_message or (i <= len(self.action_record_steps) and self.action_record_steps[i - 1] > self.cart_changed_step)}
            unproven = [e for e in shown if e.get('done') is not True or not any((_as_count(n) or 0) in proof for n in _as_list(e.get('lines')))]
            if not call and unproven and (not said) and (not reminded) and self.roles.get('ask') and (self.messages_sent < self.message_limit) and (self.steps_left() >= 2):
                reminded = True
                self.trace.append({'kind': 'gate_unproven', 'steps': _clipped(unproven, 120)})
                self.action_record.append(f'(Oradon: no line made since the cart last changed at step {self.cart_changed_step} shows ' + '; '.join(str(e.get('step') or '') for e in unproven)[:200] + f' done, and {self.message_limit - self.messages_sent} message(s) and {self.steps_left()} steps are left: give the call that does it)')
                self.action_record_steps.append(-1)
                continue
            if not call or name not in self.schemas or name in ending:
                if not call:
                    self.process_cleared.add((process_stage, self.cart_signature() + f'#{self.task_version}'))
                break
            if self.steps_left() < 2 or (name == self.roles.get('ask') and self.messages_sent >= self.message_limit):
                break
            shape_problems: list[str] = []
            args = _coerce_arguments(call.get('args') if isinstance(call.get('args'), dict) else {}, self.schemas[name], 'args', shape_problems)
            if shape_problems:
                break
            if name == self.roles.get('ask'):
                text = next((v for v in args.values() if isinstance(v, str)), '')
                batch = self.send_messages([(1, '', text)], 1)
                if not batch:
                    break
                self.process_messages += 1
                said.append(f'Q: {text}\nA: {batch[0][2]}')
                continue
            reply = self.store_call(name, args)
            if reply is None:
                break
            seen = reply.get('observation')
            self.take_in(seen)
            self.action_record.append(f'step {self.step}: {name} {_compact(args)} -> {self.result_text(seen)[:300]} (cart then: {self.cart_signature()})')
            self.action_record_steps.append(self.step)
            said.append(f'{name} {_compact(args)} -> {self.result_text(seen)[:400]}')
        return (True, 'Before going on, Oradon did the steps the request asks for:\n' + '\n'.join(said) if said else '')

    def reply_words(self, observation: Any) -> str:
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
        return ' '.join(found)[:800] if found else _compact(observation)[:400]

    def notice_lines(self, observation: Any) -> list[str]:
        lines: list[str] = []
        if isinstance(observation, dict) and (not self.listing_id_of(observation)):
            for (key, value) in observation.items():
                if isinstance(value, str) and len(value.split()) >= 5:
                    lines.append(value)
                elif isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                    lines.extend(v for v in value if len(v.split()) >= 5)
                elif isinstance(value, dict) and (not self.listing_id_of(value, False, 1)):
                    for inner in value.values():
                        if isinstance(inner, str) and len(inner.split()) >= 6:
                            lines.append(inner)
        return [line for line in lines if line not in self.notices_read]

    def read_notice(self, lines: list[str]) -> str:
        self.notices_read.update(lines)
        prompt = NOTICE_PROMPT + '\n\nITEMS AND REQUIREMENTS NOW:\n' + self.requirements_text() + '\n\nTHE NOTE:\n' + '\n'.join(lines)
        json_reply = self.json_answer(prompt, [FAST_MODEL, BACKUP_MODEL], 2500) or {}
        gone = [str(g) for g in _as_list(json_reply.get('gone')) if isinstance(g, (str, int))]
        self.unavailable.update(g for g in gone if g in self.listings)
        self.notice_sold_out = bool(gone) or json_reply.get('sold_out') is True
        self.notice_named_listing = any(g in self.listings for g in gone)
        if self.notice_sold_out and len(lines) == 1:
            self.sold_out_notices.update(lines)
        if json_reply.get('kind') not in (None, 'change'):
            return ''
        changed = []
        budget = json_reply.get('budget')
        if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0) and (budget != self.budget):
            self.budget = budget
            changed.append(f'budget now {budget}')
        added: set[str] = set()
        drops = [e for e in _as_list(json_reply.get('drop')) if isinstance(e, dict) and str(e.get('need') or '').strip()]
        adds = [e for e in _as_list(json_reply.get('add')) if isinstance(e, dict) and str(e.get('need') or '').strip()]
        for entry in drops:
            item = self.products[min(max(1, _as_count(entry.get('item')) or 1), len(self.products)) - 1]
            target = str(entry['need']).strip()
            live = [m for m in item.stated + item.revealed if m not in item.waived]
            hit = [m for m in live if m == target]
            if not hit:
                hit = [m for m in live if _plain_words(m.split(':')[0]) == _plain_words(target.split(':')[0])][:1]
            item.waived.extend(hit or [target])
            changed.append(f'dropped {target}')
        for entry in adds:
            item = self.products[min(max(1, _as_count(entry.get('item')) or 1), len(self.products)) - 1]
            requirement = str(entry['need']).strip()
            if requirement in item.waived:
                item.waived.remove(requirement)
            if requirement not in item.stated + item.revealed:
                item.revealed.append(requirement)
            added.add(requirement)
            changed.append(f'added {requirement}')
        if gone:
            changed.append('unavailable: ' + ', '.join(gone))
        if json_reply.get('price_moved') is True:
            changed.append('a price changed')
        summary = str(json_reply.get('summary') or '').strip()
        if self.reading.get('pick') == 'cheapest' and (drops or gone or self.notice_sold_out or any(c.startswith('budget now') for c in changed)):
            self.wider_search_held = False
            self.cheaper_rows_done = set()
        if changed or summary:
            self.task_version += 1
            self.recent_calls = []
            self.changes.append(summary or '; '.join(changed))
        return (summary + (' (' + '; '.join(changed) + ')' if changed else '')).strip()

    def live_requirements(self) -> list[tuple[int, str]]:
        out = []
        for (product_number, item) in enumerate(self.products, 1):
            for requirement in item.stated + item.revealed:
                if requirement not in item.waived:
                    out.append((product_number, requirement))
        return out

    def evidence_lines(self, observation: Any) -> str:
        ids: list[str] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                listing_id = self.listing_id_of(node, False, depth)
                if listing_id and _attribute_block(node) and (listing_id not in ids):
                    ids.append(listing_id)
                    return
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1)
        walk(observation, 0)
        needs = self.live_requirements()
        if not ids or not needs:
            return ''
        many = len(self.products) > 1
        lines = []
        for listing_id in ids[:12]:
            block = _attribute_block(self.opened_records.get(listing_id) or {}) or {}
            parts = []
            for (product_number, requirement) in needs[:8]:
                property_name = requirement.split(':', 1)[0].strip()
                hits = [f'{k}={str(v)[:50]}' for (k, v) in block.items() if _names_property(_stems(property_name), _stems(k))]
                tag = f'(item {product_number}) ' if many else ''
                if hits:
                    parts.append(f'{tag}{property_name} -> ' + ', '.join(hits[:3]))
            if parts:
                lines.append(f'{listing_id}: ' + '; '.join(parts))
        if not lines:
            return ''
        return "[evidence: keys whose names contain a requirement's property; one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property] " + ' | '.join(lines)

    def cart_description(self) -> str:
        if self.cart_view_result is not None and self.cart_viewed_step >= self.cart_changed_step:
            return _compact(_clipped(self.cart_view_result, 300))[:3000]
        return "lines by Oradon's record of the cart calls (listing: units): " + _compact(self.cart_counts)

    def units_in_cart(self, listing_id: str) -> int:
        if self.cart_view_result is not None and self.cart_viewed_step >= self.cart_changed_step:
            stack: list[Any] = [self.cart_view_result]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    if listing_id in [v.strip() for v in node.values() if isinstance(v, str)]:
                        counts = [_as_count(v) for (k, v) in node.items() if not isinstance(v, str) and _as_count(v) is not None and re.search('\\b(?:qty|quantity|count|units?)\\b', _plain_words(k))]
                        if len(counts) == 1:
                            return counts[0]
                    stack += [v for v in node.values() if isinstance(v, (dict, list))]
                elif isinstance(node, list):
                    stack += node
        return self.cart_counts.get(listing_id, 1)

    def cart_listing_ids(self) -> list[str]:
        if self.cart_view_result is not None and self.cart_viewed_step >= self.cart_changed_step:
            text = _compact(self.cart_view_result)
            return [i for i in self.listings if i in text]
        return list(self.cart_counts)

    def opened_by_product(self, skip: list[str] | tuple = ()) -> list[tuple[int, list[str]]]:
        """Opened, not sold-out listings grouped by the product they are a listing of (0: a kind no product owns, kept only
        when no opened listing maps to a product); within a product the cheapest first under a cheapest rule, else the newest
        first."""
        category_column = self.category_field()
        groups: dict[int, list[str]] = {}
        for listing_id in reversed(self.listing_order):
            if listing_id not in self.opened_records or listing_id in skip or self.is_sold_out(listing_id):
                continue
            label = str((self.listings.get(listing_id) or {}).get(category_column) or '') if category_column else ''
            owner = self.product_of_label(label) if label else 0
            if not category_column and len(self.products) == 1:
                owner = 1
            groups.setdefault(owner, []).append(listing_id)
        if self.reading.get('pick') == 'cheapest':
            for ids in groups.values():
                ids.sort(key=lambda i: self.price(i) if self.price(i) is not None else float('inf'))
        owned = sorted((n, ids) for (n, ids) in groups.items() if n > 0)
        return owned or ([(0, groups[0])] if 0 in groups else [])

    def memory_records(self, skip: list[str] | tuple = (), opened_limit: int = 30, row_limit: int = 15, clip: int = 200, chars: int = 900) -> list[str]:
        """Records for a judgement over what was seen: every opened listing of a requested kind first, taken in turn from
        each product (so an early-opened fit of one product is never crowded out by later rows), then search rows, newest
        first."""
        groups = [list(ids) for (_, ids) in self.opened_by_product(skip)]
        opened: list[str] = []
        while len(opened) < opened_limit and any(groups):
            for ids in groups:
                if ids and len(opened) < opened_limit:
                    opened.append(ids.pop(0))
        lines = ['OPENED ' + _compact(_clipped(self.opened_records.get(i), clip))[:chars] for i in opened]
        rows = 0
        for listing_id in reversed(self.listing_order):
            if rows >= row_limit:
                break
            if listing_id in self.opened_records or listing_id in skip or self.is_sold_out(listing_id):
                continue
            lines.append('ROW ' + _compact(_clipped(self.listings.get(listing_id), 120))[:500])
            rows += 1
        return lines

    def opened_ledger(self) -> str:
        """The opened candidates per product, with price, stock and the values their attributes give for the product's
        requirements: kept in every status, so a trimmed history never loses what was opened."""
        groups = self.opened_by_product()
        if not groups:
            return ''
        needs = self.live_requirements()
        parts = []
        for (number, ids) in groups:
            own = [r.split(':', 1)[0].strip() for (n, r) in needs if n == number or number == 0]
            shown = []
            for listing_id in ids[:8]:
                block = _attribute_block(self.opened_records.get(listing_id) or {}) or {}
                values = [f'{k}={str(v)[:30]}' for prop in own for (k, v) in block.items() if _names_property(_stems(prop), _stems(k))][:3]
                price = self.price(listing_id)
                facts = ([format(price, 'g')] if price is not None else []) + ([] if self.purchasable(listing_id) else ['not buyable'])
                shown.append(listing_id + (f" ({', '.join(facts)})" if facts else '') + (': ' + ', '.join(values) if values else ''))
            more = f' and {len(ids) - 8} more' if len(ids) > 8 else ''
            head = f'item {number} ({self.products[number - 1].kind})' if number > 0 and len(self.products) > 1 else 'opened'
            parts.append(f'{head}: ' + '; '.join(shown) + more)
        return '\n[opened so far] ' + ' | '.join(parts)

    def record_text(self, listing_id: str) -> str:
        record = self.opened_records.get(listing_id) or self.listings.get(listing_id)
        return _compact(_clipped(record, 400))[:3000] if record else '(no record seen)'

    def other_records(self, skip: list[str], limit: int=30) -> str:
        return '\n'.join(self.memory_records(skip, opened_limit=max(limit, 20), row_limit=5, clip=160, chars=900))

    def compatibility_index(self) -> dict[str, Any]:
        marks: dict[str, dict[str, Any]] = {}
        notes: dict[str, set[str]] = {}
        for listing_id in self.listing_order:
            record = self.opened_records.get(listing_id) or self.listings.get(listing_id) or {}
            text = ' '.join(str(v) for v in record.values() if isinstance(v, str))
            text += ' ' + '. '.join(f'{k} {v}' for block in record.values() if isinstance(block, dict) for (k, v) in block.items() if isinstance(v, str))
            marks[listing_id] = _compatibility_marks(text)
            for (part, codes) in marks[listing_id]['also']:
                notes.setdefault(part, set()).update(codes)
        return {'marks': marks, 'notes': notes}

    def compatible_codes(self, listing_id: str, index: dict[str, Any]) -> set[str]:
        signature = index['marks'].get(listing_id) or {}
        found = set(signature.get('works') or [])
        if signature.get('own'):
            found |= index['notes'].get(signature['own'], set())
        return found

    def listing_ids_of(self, product_number: int) -> list[str]:
        category_column = self.category_field()
        item = self.products[product_number - 1]
        return [i for i in self.listing_order if category_column and self.label_belongs_to(item, str((self.listings.get(i) or {}).get(category_column) or ''))]

    def compatibility_faults(self) -> list[str]:
        if not self.compatibility_pairs:
            return []
        index = self.compatibility_index()
        cart = self.cart_listing_ids()
        shape_problems = []
        for (part, main) in self.compatibility_pairs:
            devices = [i for i in cart if i in self.listing_ids_of(main)]
            parts = [i for i in cart if i in self.listing_ids_of(part)]
            if len(devices) != 1 or not parts:
                continue
            codes = set(index['marks'].get(devices[0], {}).get('codes') or [])
            if not codes:
                continue
            for line in parts:
                fits = self.compatible_codes(line, index)
                if fits and (not fits & codes):
                    good = [i for i in self.listing_ids_of(part) if self.compatible_codes(i, index) & codes and (not self.is_sold_out(i)) and (i not in cart)][:4]
                    shape_problems.append(f"{line} ({self.products[part - 1].kind}) works with {', '.join(sorted(fits)[:4])}, not with {devices[0]} ({', '.join(sorted(codes))})" + (f"; seen parts of that kind that work with it: {', '.join(good)}" if good else '; no seen part of that kind works with it, so look for one or switch the device: ' + self.compatibility_plan(short=True)))
        return shape_problems

    def compatibility_plan(self, short: bool=False) -> str:
        if not self.compatibility_pairs:
            return ''
        index = self.compatibility_index()
        lines = []
        for main in sorted({m for (_, m) in self.compatibility_pairs}):
            parts = [p for (p, m) in self.compatibility_pairs if m == main]
            tally: dict[str, dict[int, int]] = {}
            for p in parts:
                for i in self.listing_ids_of(p):
                    if self.is_sold_out(i):
                        continue
                    for code in self.compatible_codes(i, index):
                        tally.setdefault(code, {}).setdefault(p, 0)
                        tally[code][p] += 1
            ranked = sorted(tally.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1].values())))[:4]
            devices = {code: [i for i in self.listing_ids_of(main) if code in (index['marks'].get(i, {}).get('codes') or [])][:2] for (code, _) in ranked}
            text = '; '.join(f'{code}: ' + ', '.join(f'{n} {self.products[p - 1].kind}' for (p, n) in sorted(per.items())) + (f" (device seen: {', '.join(devices[code])})" if devices[code] else ' (device not seen yet)') for (code, per) in ranked)
            if text:
                lines.append(f"codes the {self.products[main - 1].kind} candidates' parts name -> {text}")
        if not lines:
            return ''
        return ' | '.join(lines) if short else "[fit plan: a device fits a part when the part's text names the device's model code or another name it is sold as; codes are not searchable, so list the device's category and read the codes on its rows] " + ' | '.join(lines)

    def cart_line_faults(self) -> list[str]:
        category_column = self.category_field()
        ids = [i for i in self.cart_listing_ids() if (self.units_in_cart(i) or 0) > 0]
        if not ids or not category_column:
            return []
        label = {i: str((self.listings.get(i) or {}).get(category_column) or '') for i in ids}
        known = {n for (n, item) in enumerate(self.products, 1) if any(self.label_belongs_to(item, str(r.get(category_column) or '')) for r in self.listings.values())}
        owner = {i: [n for (n, item) in enumerate(self.products, 1) if self.label_belongs_to(item, label[i])] for i in ids}
        shape_problems: list[str] = []
        for (n, item) in enumerate(self.products, 1):
            if n not in known:
                continue
            lines = [i for i in ids if n in owner[i]]
            if not lines:
                shape_problems.append(f'item {n} ({item.kind}) has no line in the cart')
            elif len([i for i in lines if owner[i] == [n]]) > 1:
                shape_problems.append(f"item {n} ({item.kind}) has {len(lines)} lines ({', '.join(lines)}); keep one")
            for i in lines:
                have = self.units_in_cart(i)
                if owner[i] == [n] and have != item.quantity:
                    shape_problems.append(f'{i} is held at {have} unit(s); item {n} ({item.kind}) wants {item.quantity}')
        if len(known) == len(self.products):
            for i in ids:
                if not owner[i]:
                    shape_problems.append(f"{i} ({label[i] or 'unknown kind'}) is not a kind the shopper asked for: take it out")
        for i in ids:
            if self.is_sold_out(i):
                shape_problems.append(f'{i} is sold out')
        prices = {i: self.price(i) for i in ids}
        if isinstance(self.budget, (int, float)) and all(p is not None for p in prices.values()):
            if self.reading.get('budget_scope') == 'each':
                over = [i for i in ids if self.budget < prices[i] <= 10 * self.budget]
                shape_problems += [f'{i} costs {prices[i]:g}, over the {self.budget:g} limit for each item' for i in over]
            else:
                total = sum(prices[i] * self.units_in_cart(i) for i in ids)
                if self.budget + 1e-09 < total <= 10 * self.budget:
                    shape_problems.append(f'the total is {total:g}, over the budget of {self.budget:g}')
        return shape_problems

    def wrong_quantities(self) -> list[tuple[str, int, int]]:
        category_column = self.category_field()
        if not category_column:
            return []
        out = []
        for listing_id in self.cart_listing_ids():
            have = self.units_in_cart(listing_id) or 0
            owners = [n for (n, item) in enumerate(self.products, 1) if self.label_belongs_to(item, str((self.listings.get(listing_id) or {}).get(category_column) or ''))]
            mine = [i for i in self.cart_listing_ids() if (self.cart_counts.get(i, 1) or 0) > 0 and len(owners) == 1 and self.label_belongs_to(self.products[owners[0] - 1], str((self.listings.get(i) or {}).get(category_column) or ''))]
            if have > 0 and len(owners) == 1 and (len(mine) == 1) and (have != self.products[owners[0] - 1].quantity) and (not self.is_sold_out(listing_id)):
                out.append((listing_id, have, self.products[owners[0] - 1].quantity))
        return out

    def set_quantities(self, wrong: list[tuple[str, int, int]]) -> list[str]:
        name = self.roles.get('cart') or ''
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        texts = [k for (k, v) in props.items() if isinstance(v, dict) and 'string' in _schema_types(v) and ('enum' not in v)]
        target = next((k for (t, k) in self.id_arguments if t == name and k in texts), None) or next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        count = next((k for (k, v) in props.items() if isinstance(v, dict) and 'integer' in _schema_types(v)), None)
        mode = next((k for (k, v) in props.items() if isinstance(v, dict) and isinstance(v.get('enum'), list) and (len(v['enum']) >= 2)), None)
        if not name or not target or (not count):
            return []
        calls: list[dict[str, Any]] = []
        for (listing_id, have, want) in wrong:
            if self.cart_sets_quantity(name):
                calls.append({target: listing_id, count: want})
            elif mode:
                options = [str(o) for o in props[mode]['enum']]
                off = next((o for o in options if re.search('remov|delet|drop|clear|take', o, re.I)), options[1])
                on = next((o for o in options if o != off), options[0])
                calls += [{target: listing_id, mode: off, count: have}, {target: listing_id, mode: on, count: want}]
            else:
                return []
        if len(calls) + 3 > self.steps_left():
            return []
        done = []
        for args in calls:
            shape_problems: list[str] = []
            args = _coerce_arguments(args, schema, 'args', shape_problems)
            if shape_problems:
                break
            reply = self.store_call(name, args)
            if reply is None or self.finished:
                break
            self.track_cart(name, args)
            self.last_cart_reply = _compact(reply.get('observation'))
            self.take_result(name, reply.get('observation'))
            done.append(_compact(args) + ' -> ' + self.result_text(reply.get('observation'))[:300])
        self.trace.append({'kind': 'count_fix', 'calls': done})
        return done

    def review_key(self, ids: list[str]) -> str:
        return '|'.join(sorted(ids)) + f'#{self.task_version}'

    def review_cart(self, ids: list[str] | None=None, limit: float=SPEND_SOFT_CAP) -> str:
        lead = ids is not None
        ids = ids if lead else self.cart_listing_ids()
        key = self.review_key(ids)
        if key in self.reviewed_carts or not self.reviewer_models(limit) or (not ids):
            return ''
        self.reviewed_carts.add(key)
        prompt = CHECK_PROMPT + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.requirements_text()
        if lead:
            quantity = '; '.join(f'item {n}: quantity {it.quantity}' for (n, it) in enumerate(self.products, 1))
            prompt += '\n\nCART (planned, judge these listings as if they were carted at the requested quantities: ' + quantity + '):\n' + ', '.join(ids)
        else:
            prompt += '\n\nCART:\n' + self.cart_description()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + '\n'.join(f'{i}: ' + ('SOLD OUT (the store says it can no longer be bought). ' if self.is_sold_out(i) else '') + self.record_text(i) for i in ids)
        if self.reading.get('pick') in ('cheapest', 'most', 'least'):
            prompt += '\n\nOTHER LISTINGS SEEN (every opened listing of a requested kind first, item by item; then rows):\n' + self.other_records(ids, 20)
        json_reply = self.json_answer(prompt, self.reviewer_models(limit), 3000)
        self.trace.append({'kind': 'check', 'cart': ids, 'got': json_reply})
        self.review_verdicts[key] = bool(json_reply) and json_reply.get('ok') is not False
        if not json_reply or json_reply.get('ok') is not False:
            return ''
        self.review_candidates = [i for i in self.review_candidates if i not in ids]
        problems = [str(p) for p in _as_list(json_reply.get('problems'))][:5]
        better = [str(b) for b in _as_list(json_reply.get('better')) if str(b) in self.listings and (not self.is_sold_out(str(b)))][:4]
        text = 'A careful review says this cart is not right yet: ' + '; '.join(problems or ['see the requirements'])
        if better:
            text += '. Listings that look better: ' + ', '.join(better)
        return text + '. Fix the cart (or show the review is wrong by re-reading the records), then check out.'

    def follow_chains(self) -> tuple[str, bool]:
        live = {requirement for (_, requirement) in self.live_requirements()}
        pending = [requirement for (_, requirement) in self.pointer_requirements if requirement in live and requirement not in self.pointer_values]
        tool = self.roles.get('offer') or self.roles.get('open')
        target = self.sole_id_argument(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.chain_rounds < (6 if self.steps_left() > 15 else 4) and (self.spend < SPEND_HARD_CAP):
            self.chain_rounds += 1
            records = []
            for listing_id in reversed(self.listing_order):
                record = self.opened_records.get(listing_id) or self.listings.get(listing_id) if listing_id in self.opened_pages else None
                if record and len(records) < 45:
                    records.append('OPENED ' + _compact(_clipped(record, 300))[:1400])
            for listing_id in reversed(self.listing_order):
                if len(records) >= 60:
                    break
                if listing_id not in self.opened_pages or not (self.opened_records.get(listing_id) or self.listings.get(listing_id)):
                    records.append('ROW ' + _compact(_clipped(self.listings.get(listing_id), 80))[:300])
            prompt = CHAIN_PROMPT + '\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n' + '\n'.join(pending)
            prompt += '\n\nRECORDS (newest first):\n' + '\n'.join(records)
            json_reply = self.json_answer(prompt, [FAST_MODEL] + ([STRONG_MODEL] if self.spend + STRONG_MODEL_ROOM <= SPEND_HARD_CAP else []), 4500) or {}
            self.trace.append({'kind': 'trail', 'pending': pending, 'got': _clipped(json_reply, 150)})
            links_to_open: list[str] = []
            lost = []
            missing_links: list[tuple[str, str]] = []
            for chain in _as_list(json_reply.get('chains')):
                if not isinstance(chain, dict):
                    continue
                requirement = next((p for p in pending if p == str(chain.get('need') or '').strip()), None) or (pending[0] if len(pending) == 1 else None)
                if not requirement:
                    continue
                links = [l for l in _as_list(chain.get('links')) if isinstance(l, dict)]
                ids = [str(l.get('id') or '').strip() for l in links]
                if chain.get('done') is True and links and all(i in self.opened_pages for i in ids) and str(chain.get('value') or '').strip():
                    self.pointer_values[requirement] = str(chain['value']).strip()
                    self.task_version += 1
                    continue
                if links and all(i in self.opened_pages for i in ids) and (not str(links[-1].get('find') or '').strip()):
                    self.pointer_values[requirement] = str(chain.get('value') or '').strip() or f'as stated on {ids[-1]}'
                    self.task_version += 1
                    continue
                closed = [i for i in ids if i in self.listings and i not in self.opened_pages]
                if closed:
                    links_to_open += [i for i in closed if i not in links_to_open]
                else:
                    gap = next((l for (l, i) in zip(links, ids) if i not in self.listings), {})
                    link_description = str(gap.get('about') or '') or 'the listing it points at'
                    words = str(gap.get('find') or '').strip()
                    lost.append(f'{requirement} -> find {link_description}' + (f' (search: {words})' if words else ''))
                    missing_links.append((link_description, words))
            pending = [p for p in pending if p not in self.pointer_values]
            if not links_to_open and missing_links and tool and target and (self.steps_left() > 12) and (self.chain_searches < 3):
                links_to_open = self.search_chain_link(*missing_links[0])
                if self.trace and self.trace[-1].get('kind') == 'link_known':
                    seen.append(f"{self.trace[-1]['id']}: an opened page of the shop this chain names that shows the values its description gives; read the value from it")
                    lost = lost[1:]
            if not links_to_open or not tool or (not target) or (self.steps_left() < len(links_to_open) + 4) or self.finished:
                break
            for listing_id in links_to_open[:max(0, min(3, 6 - len(seen)))]:
                shape_problems: list[str] = []
                args = _coerce_arguments({target: listing_id}, self.schemas.get(tool), 'args', shape_problems)
                if shape_problems:
                    break
                reply = self.store_call(tool, args)
                if reply is None or self.finished:
                    break
                self.take_result(tool, reply.get('observation'))
                seen.append(f'{listing_id}: ' + self.result_text(reply.get('observation'))[:1500])
        text = ''
        if seen:
            text += '[Oradon opened the pages a requirement points at]\n' + '\n'.join(seen) + '\n'
        if self.pointer_values:
            text += '[values read from the pages requirements point at] ' + '; '.join(f'{k} -> {v}' for (k, v) in self.pointer_values.items()) + '\n'
        hold = bool(lost) and self.chain_holds < (3 if self.steps_left() > 15 else 2) and (self.steps_left() > 5)
        if hold:
            self.chain_holds += 1
            text += 'The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: ' + ' | '.join(lost[:3]) + ". Search for it by its shop's name together with the values its description gives (the shop's other listings of that kind are not it), open the row of that shop whose page shows those values, apply the value its page gives, then check out."
        return (text, hold)

    def rows_of_shop(self, link_description: str, ids: list[str]) -> list[str]:
        low = link_description.lower()
        out = []
        for listing_id in ids:
            row = self.listings.get(listing_id) or {}
            names = [str(v).strip() for (k, v) in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I)]
            if listing_id not in self.opened_pages and any(re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names):
                out.append(listing_id)
        return out

    def opened_of_shop(self, link_description: str) -> list[str]:
        low = link_description.lower()
        out = []
        for listing_id in self.opened_pages:
            row = self.listings.get(listing_id) or {}
            names = [str(v).strip() for (k, v) in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I)]
            if any(re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names):
                out.append(listing_id)
        return out

    def search_chain_link(self, link_description: str, words: str) -> list[str]:
        known: list[str] = []
        named = {w for listing_id in self.rows_of_shop(link_description, list(self.listings)) + self.opened_of_shop(link_description) for (k, v) in (self.listings.get(listing_id) or {}).items() if isinstance(v, str) and len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I) for w in re.findall('[a-z0-9]{3,}', v.lower())}
        values = {w for w in re.findall('[a-z0-9]+', (words or link_description).lower()) if len(w) >= 3 or w.isdigit()} - named - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for', 'of'}
        for listing_id in self.opened_of_shop(link_description):
            (stack, leaves) = ([self.opened_records.get(listing_id) or self.listings.get(listing_id) or {}], [])
            while stack:
                node = stack.pop()
                if isinstance(node, (dict, list)):
                    stack += list(node.values()) if isinstance(node, dict) else node
                else:
                    leaves.append(str(node))
            text = ' '.join(leaves).lower()
            if values and all(re.search('(?<![a-z0-9])' + re.escape(v) + '(?![a-z0-9])', text) for v in values):
                self.trace.append({'kind': 'link_known', 'id': listing_id})
                return []
        plan = self.search_form()
        words_key = ' '.join(words.lower().split())
        if words_key and words_key in self.chain_search_rows:
            self.chain_searches += 1
            known = self.rows_of_shop(link_description, self.chain_search_rows[words_key])
        elif plan and words:
            (tool, base, field) = plan
            shape_problems: list[str] = []
            args = _coerce_arguments({**base, field: words}, self.schemas.get(tool), 'args', shape_problems)
            if not shape_problems:
                self.chain_searches += 1
                reply = self.store_call(tool, args)
                if reply is not None:
                    observation = reply.get('observation')
                    self.take_result(tool, observation)
                    self.chain_search_rows[words_key] = [i for (i, _) in self.listings_in(observation)]
                    known = self.rows_of_shop(link_description, self.chain_search_rows[words_key])
                    self.trace.append({'kind': 'link_search', 'words': words[:80], 'rows': known[:4]})
        kind = {w for w in re.findall('[a-z0-9]{3,}', (words or link_description).lower())} - named - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for'}

        def of_kind(ids: list[str]) -> list[str]:
            return [i for i in ids if not kind or kind & set(re.findall('[a-z0-9]{3,}', _compact(self.listings.get(i) or {}).lower()))]
        return (of_kind(known) or of_kind(self.rows_of_shop(link_description, list(reversed(self.listing_order)))))[:2]

    def single_listing_tools(self) -> list[str]:
        """Tools that read one listing: the sole required text argument takes a listing id (the open or offer role, a tool
        seen called with one, or an argument named for an id), and no other role."""
        others = {self.roles.get(r) for r in ('search', 'cart', 'cart_view', 'buy', 'refuse', 'ask', 'compare')} - {None}
        out = []
        for name in self.schemas:
            target = self.sole_id_argument(name)
            if not target or name in others:
                continue
            if name in (self.roles.get('open'), self.roles.get('offer')) or (name, target) in self.id_arguments or re.search('id|item|product|listing|code|ref|sku', target, re.I):
                out.append(name)
        return out

    def open_lines_with_page_tools(self) -> str:
        """Before a checkout of a cart not seen this way yet: each cart line opened once with every single-listing page tool
        not yet called on it (one step each), while the steps leave the checkout and a repair. A store may count a line as
        opened only through one of its page tools (races 10-12: 5 of 180 v11.6 tasks lost on it)."""
        signature = self.cart_signature()
        if signature in self.lines_opened_for:
            return ''
        self.lines_opened_for.add(signature)
        tools = self.single_listing_tools()
        missing = [(t, line) for line in self.cart_listing_ids() for t in tools if (t, line) not in self.listing_tool_calls and not self.is_sold_out(line)]
        if not missing or len(missing) + self.final_steps() > self.steps_left() or self.time_left() < 5:
            return ''
        texts = []
        before = {line: (self.price(line), self.purchasable(line)) for line in self.cart_listing_ids()}
        for (tool, line) in missing:
            shape_problems: list[str] = []
            args = _coerce_arguments({self.sole_id_argument(tool): line}, self.schemas.get(tool), 'args', shape_problems)
            if shape_problems:
                continue
            reply = self.store_call(tool, args)
            if reply is None or self.finished:
                break
            self.take_result(tool, reply.get('observation'))
            texts.append(f'{line} with {tool}: ' + self.result_text(reply.get('observation'))[:800])
        self.trace.append({'kind': 'lines_opened', 'calls': [f'{t}:{l}' for (t, l) in missing], 'read': len(texts)})
        moved = [line for (line, (price, buyable)) in before.items() if self.purchasable(line) != buyable or (price is not None and self.price(line) not in (None, price))]
        if not texts or not moved:
            return ''  # nothing changed: the checkout goes on in the same call
        return "Oradon opened the cart's lines with the store's other page tools (a store may count a line as opened only that way):\n" + '\n'.join(texts) + '\nA line changed (stock or price): check the lines, then check out again.'

    def open_part_pages(self) -> str:
        """Before the first checkout: a cart line whose page says to see another part for the rest ("see part X") gets X's
        page opened (seen, else searched for by its code), once; the opened text is shown to the navigator."""
        if self.part_pages_held or self.steps_left() < 4:
            return ''
        tool = self.offer_tool_name()
        target = self.sole_id_argument(tool)
        if not tool or not target:
            return ''
        index = self.compatibility_index()
        wanted: list[str] = []
        for line in self.cart_listing_ids():
            code = (index['marks'].get(line) or {}).get('more')
            if code and code not in wanted:
                wanted.append(code)
        if not wanted:
            return ''
        self.part_pages_held = True
        texts: list[str] = []
        for code in wanted[:2]:
            pages = [i for (i, signature) in index['marks'].items() if signature.get('own') == code]
            if any(page in self.opened_pages for page in pages):
                continue
            plan = self.search_form()
            if not pages and plan and self.steps_left() >= 4:
                (search, base, field) = plan
                shape_problems: list[str] = []
                args = _coerce_arguments({**base, field: code}, self.schemas.get(search), 'args', shape_problems)
                if not shape_problems:
                    reply = self.store_call(search, args)
                    if reply is not None:
                        self.take_result(search, reply.get('observation'))
                    index = self.compatibility_index()
                    pages = [i for (i, signature) in index['marks'].items() if signature.get('own') == code]
            for page in pages[:1]:
                if self.steps_left() < 3 or self.finished:
                    break
                shape_problems = []
                args = _coerce_arguments({target: page}, self.schemas.get(tool), 'args', shape_problems)
                if shape_problems:
                    break
                reply = self.store_call(tool, args)
                if reply is None or self.finished:
                    break
                self.take_result(tool, reply.get('observation'))
                texts.append(f'{page} (part {code}): ' + self.result_text(reply.get('observation'))[:1500])
        self.trace.append({'kind': 'part_pages', 'codes': wanted[:2], 'opened': len(texts)})
        if not texts:
            return ''
        return "A cart line's page sends you to another part's page for the rest, so Oradon opened that page:\n" + '\n'.join(texts) + '\nCheck that the cart lines still fit with what it says, then check out.'

    def offer_tool_name(self) -> str | None:
        return self.roles.get('offer') or self.roles.get('open')

    def reread_cart_offers(self) -> str:
        tool = self.offer_tool_name()
        target = self.sole_id_argument(tool)
        if not tool or not target:
            return ''
        since = max(self.step - 2, self.refused_step)
        stale = [i for i in self.cart_listing_ids() if self.offer_read_step.get(i, -99) < since and i not in self.sold_out_step]
        if not stale or self.steps_left() < len(stale) + 3:
            return ''
        lines = []
        for listing_id in stale:
            shape_problems: list[str] = []
            args = _coerce_arguments({target: listing_id}, self.schemas.get(tool), 'args', shape_problems)
            if shape_problems:
                break
            before = dict(self.listings.get(listing_id) or {})
            reply = self.store_call(tool, args)
            if reply is None or self.finished:
                break
            observation = reply.get('observation')
            self.offer_read_step[listing_id] = self.step
            self.take_result(tool, observation)
            after = self.listings.get(listing_id) or {}
            moved = [k for (k, v) in after.items() if k in before and before[k] != v and (not isinstance(v, (dict, list)))]
            if moved:
                lines.append(f"{listing_id} changed ({', '.join(moved[:4])}): {_compact(_clipped(observation, 300))[:900]}")
        self.trace.append({'kind': 'reread', 'ids': stale, 'changed': len(lines)})
        if lines:
            self.task_version += 1
        return '\n'.join(lines)

    def is_sold_out(self, listing_id: str) -> bool:
        return listing_id in self.unavailable or listing_id in self.sold_out_step

    def sold_out_hint(self) -> str:
        lost = [i for i in self.cart_listing_ids() if i not in self.sold_out_step and self.is_sold_out(i)]
        if not lost:
            return ''
        for listing_id in lost:
            self.sold_out_step[listing_id] = self.step
        self.review_candidates = [i for i in self.review_candidates if i not in lost]
        self.task_version += 1
        selected = self.replacement_candidates(lost)
        text = f"[Oradon] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        if selected and self.steps_left() <= 6:
            return text + f" Few steps are left: in this turn add {selected[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(selected[1:]) or 'none'}."
        if selected and self.reading.get('pick') == 'cheapest':
            return text + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(selected)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {selected[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        if selected:
            return text + f" Listings already read that appear to fit that item instead: {', '.join(selected)}. In this turn take the sold-out line out of the cart and add {selected[0]} at the requested quantity, then check out."
        return text + ' Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits.'

    def find_sold_out_line(self) -> str:
        ids = [i for i in self.cart_listing_ids() if i not in self.sold_out_step]
        if any(self.is_sold_out(i) for i in ids):
            return ''
        if len(ids) == 1:
            self.unavailable.add(ids[0])
            return ''
        return self.reread_cart_offers() if ids else ''

    def replacement_candidates(self, lost: list[str]) -> list[str]:
        if not self.reviewer_models():
            return []
        records = self.memory_records(list(lost) + list(self.sold_out_step), opened_limit=20, row_limit=10, clip=200, chars=1000)
        if not records:
            return []
        prompt = REPLACEMENT_PROMPT + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.requirements_text()
        prompt += '\n\nSOLD-OUT CART LINE:\n' + '\n'.join(f'{i}: {self.record_text(i)}' for i in lost) + '\n\nCART NOW:\n' + self.cart_description()
        prompt += '\n\nOTHER LISTING RECORDS SEEN (opened listings of a requested kind first; then rows):\n' + '\n'.join(records)
        json_reply = self.json_answer(prompt, self.reviewer_models(), 3500)
        fits: list[tuple[float, str]] = []
        for entry in _as_list((json_reply or {}).get('fits')):
            listing_id = str(entry.get('id') if isinstance(entry, dict) else entry)
            price = entry.get('price') if isinstance(entry, dict) else None
            if listing_id in self.listings and listing_id not in lost and (listing_id not in self.sold_out_step) and (listing_id not in self.cart_listing_ids()) and (listing_id not in [i for (_, i) in fits]):
                fits.append((float(price) if isinstance(price, (int, float)) and (not isinstance(price, bool)) else float('inf'), listing_id))
        ranked = [i for (_, i) in sorted(fits, key=lambda f: f[0])] if self.reading.get('pick') == 'cheapest' else [i for (_, i) in fits]
        self.trace.append({'kind': 'swap', 'lost': lost, 'fits': fits})
        return ranked[:4]

    def audit_refusal(self) -> str:
        if self.refusal_audited or not self.reviewer_models():
            return ''
        self.refusal_audited = True
        json_reply = self.audit_answer()
        named = self.audit_candidates(json_reply)
        missing = self.products_without(named)
        if missing:
            rows = [str(i) for i in _as_list((json_reply or {}).get('open')) if str(i) in self.listings and str(i) not in self.opened_pages and (not self.is_sold_out(str(i)))]
            rows.sort(key=lambda i: self.row_owner(i) not in missing)  # rows of a product without a fit first
            if rows and self.open_candidate_rows(rows, most=3):
                json_reply = self.audit_answer()
                named = self.audit_candidates(json_reply) or named
        if not named:
            return ''
        self.review_candidates_by_product = named
        self.review_candidates = [i for ids in named.values() for i in ids]
        return 'A careful review thinks these listings may fit: ' + '; '.join(f"item {k}: {', '.join(v)}" for (k, v) in named.items()) + (f" ({json_reply.get('reason')})" if json_reply.get('reason') else '') + '. Open them and buy what truly fits; end without a purchase only if each one fails.'

    def rearm_refusal_checks(self, meaning: str) -> None:
        """The store refused an ending without a purchase (often with a change: a new budget, a dropped requirement): the
        task is not what the checks before the refusal judged, so each of them may run once more on the next one."""
        self.decline_rearms += 1
        self.refusal_audited = False
        self.refusal_chain_followed = False
        self.search_more_told = False
        self.late_refusal_held = False
        self.refusal_blocks = 0
        self.review_candidates = []
        self.review_candidates_by_product = {}
        self.confirmed_candidate = None
        self.completed_by_code = False
        self.task_version += 1
        self.trace.append({'kind': 'refusal_rearmed', 'meaning': meaning[:200], 'steps_left': self.steps_left()})

    def products_without(self, named: dict[str, list[str]]) -> list[int]:
        """Product numbers the audit named no listing for (a single product's key may be its name)."""
        keys = {k for (k, ids) in (named or {}).items() if ids}
        if len(self.products) == 1:
            return [] if keys else [1]
        numbers = {_as_count(k) for k in keys}
        return [n for n in range(1, len(self.products) + 1) if n not in numbers]

    def row_owner(self, listing_id: str) -> int:
        category_column = self.category_field()
        label = str((self.listings.get(listing_id) or {}).get(category_column) or '') if category_column else ''
        return self.product_of_label(label) if label else 0

    def audit_covers_every_product(self) -> bool:
        """The audit names at least one listing for every requested product (a single product's key may be its name)."""
        named = {k for (k, ids) in self.review_candidates_by_product.items() if ids}
        if len(self.products) == 1:
            return bool(named)
        numbers = {_as_count(k) for k in named}
        return all(n in numbers for n in range(1, len(self.products) + 1))

    def audit_answer(self) -> dict[str, Any] | None:
        records = self.memory_records(opened_limit=30, row_limit=15, clip=200, chars=1000)
        if not records:
            return None
        prompt = AUDIT_PROMPT + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.requirements_text()
        prompt += '\n\nLISTING RECORDS SEEN (every opened listing of a requested kind first, item by item; then search rows, newest first):\n' + '\n'.join(records)
        json_reply = self.json_answer(prompt, self.reviewer_models(), 4000)
        self.trace.append({'kind': 'audit', 'got': json_reply})
        return json_reply

    def audit_candidates(self, json_reply: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (json_reply or {}).get('fits')
        if not isinstance(fits, dict):
            return {}
        named = {str(k): [str(i) for i in _as_list(v) if str(i) in self.listings and (not self.is_sold_out(str(i)))][:4] for (k, v) in fits.items()}
        return {k: v for (k, v) in named.items() if v}

    def open_candidate_rows(self, rows: list[str], most: int=2) -> bool:
        (many, joiner) = (self.compare_list_argument(), self.roles.get('compare'))
        tool = self.attributes_tool()
        target = self.sole_id_argument(tool)
        if many and joiner and (not self.compare_is_bare()):
            plan = [(joiner, {many: rows[:self.compare_capacity()]})]
        elif tool and target:
            plan = [(tool, {target: i}) for i in rows[:most]]
        else:
            return False
        if self.steps_left() < len(plan) + max(5, len(self.products) + 2):
            return False
        before = len(self.opened_records)
        for (name, args) in plan:
            shape_problems: list[str] = []
            args = _coerce_arguments(args, self.schemas.get(name), 'args', shape_problems)
            if shape_problems:
                return False
            reply = self.store_call(name, args)
            if reply is None or self.finished:
                break
            self.take_result(name, reply.get('observation'))
        self.trace.append({'kind': 'audit_open', 'rows': rows, 'opened': len(self.opened_records) - before})
        return len(self.opened_records) > before

    def cart_call_plan(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        name = self.roles.get('cart') or ''
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        for (key, spec) in props.items():
            shape = spec.get('items') if isinstance(spec, dict) and 'array' in _schema_types(spec) else None
            if isinstance(shape, dict) and isinstance(shape.get('properties'), dict):
                inner = shape['properties']
                id_key = next((k for (k, s) in inner.items() if isinstance(s, dict) and 'string' in _schema_types(s) and ('enum' not in s)), None)
                count_key = next((k for (k, s) in inner.items() if isinstance(s, dict) and 'integer' in _schema_types(s)), None)
                if not id_key:
                    return None
                return [{key: [{id_key: i, **({count_key: n} if count_key else {})} for (i, n) in lines.items()]}]
        texts = [k for (k, s) in props.items() if isinstance(s, dict) and 'string' in _schema_types(s) and ('enum' not in s)]
        target = next((k for (t, k) in self.id_arguments if t == name and k in texts), None) or next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        count = next((k for (k, s) in props.items() if isinstance(s, dict) and 'integer' in _schema_types(s)), None)
        mode = next((k for (k, s) in props.items() if isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) >= 2)), None)
        if not target:
            return None
        calls: list[dict[str, Any]] = []
        for listing_id in drop:
            have = self.cart_counts.get(listing_id, 1)
            if mode:
                calls.append({target: listing_id, mode: props[mode]['enum'][1], **({count: have} if count else {})})
            elif count and self.cart_sets_quantity(name):
                calls.append({target: listing_id, count: 0})
            elif count and isinstance(props[count].get('minimum'), int) and (props[count]['minimum'] <= -have):
                calls.append({target: listing_id, count: -have})
            else:
                return None
        for (listing_id, want) in lines.items():
            have = self.cart_counts.get(listing_id, 0) if listing_id in self.cart_listing_ids() else 0
            if have == want:
                continue
            if have and (not self.cart_sets_quantity(name)):
                return None
            calls.append({target: listing_id, **({mode: props[mode]['enum'][0]} if mode else {}), **({count: want} if count else {})})
        return calls

    def complete_and_buy(self) -> str:
        (cart, buy) = (self.roles.get('cart'), self.roles.get('buy'))
        if self.completed_by_code or not self.confirmed_candidate or self.is_sold_out(self.confirmed_candidate) or (not cart) or (not buy):
            return ''
        self.completed_by_code = True
        category_column = self.category_field()

        def kind(listing_id: str) -> str:
            return str((self.listings.get(listing_id) or {}).get(category_column) or '').strip().lower() if category_column else listing_id
        plan: dict[str, int] = {}
        for (product_number, ids) in self.review_candidates_by_product.items():
            index = _as_count(product_number) or (1 if len(self.products) == 1 else None)
            pick = next((i for i in ids if not self.is_sold_out(i)), None)
            if pick and index and (1 <= index <= len(self.products)) and (kind(pick) not in {kind(p) for p in plan}):
                plan[pick] = self.products[index - 1].quantity
        if not plan:
            return ''
        if self.confirmed_candidate not in plan:
            self.confirmed_candidate = next(iter(plan))  # (the audit's leads were re-read: its pick for the first product)
        now = self.cart_listing_ids()
        keep = {i: self.cart_counts.get(i, 1) for i in now if i not in plan and (not self.is_sold_out(i)) and (kind(i) not in {kind(p) for p in plan})}
        drop = [i for i in now if i not in plan and i not in keep]
        if len(plan) + len({kind(i) for i in keep}) < len(self.products):
            return ''
        lines = {**keep, **plan}
        calls = self.cart_call_plan(lines, drop)
        if not calls or len(calls) + 1 > self.steps_left():
            return ''
        self.open_attributes(list(lines), len(calls) + 3)
        self.review_cart(list(lines), SPEND_HARD_CAP)
        if not self.review_verdicts.get(self.review_key(list(lines))):
            return ''
        done = []
        for args in calls:
            shape_problems: list[str] = []
            args = _coerce_arguments(args, self.schemas.get(cart), 'args', shape_problems)
            if shape_problems:
                return ''
            reply = self.store_call(cart, args)
            if reply is None or self.finished:
                return ''
            self.track_cart(cart, args)
            self.take_result(cart, reply.get('observation'))
            done.append(_compact(args))
        shape_problems = []
        args = _coerce_arguments(dict(self.reading.get('buy_args') or {}), self.schemas.get(buy), 'args', shape_problems)
        if shape_problems:
            return ''
        replies = []
        for attempt in range(2):
            if attempt:
                self.reread_cart_offers()
                self.review_cart(list(lines), SPEND_HARD_CAP)
                if not self.review_verdicts.get(self.review_key(list(lines))) or self.steps_left() < 1 or self.finished or any(self.is_sold_out(i) for i in lines):
                    break
            self.open_lines_with_page_tools()
            (go, said) = self.process_gate('order')
            if said:
                replies.append(said)
            if not go:
                break
            reply = self.store_call(buy, args)
            self.trace.append({'kind': 'completed', 'lines': lines, 'drop': drop, 'reply': _clipped(reply, 200)})
            if reply is None:
                break
            self.checkouts += 1
            if not reply.get('finished'):
                self.refused_step = self.step
            self.take_result(buy, reply.get('observation'))
            replies.append(self.result_text(reply.get('observation')))
            if self.finished or reply.get('finished'):
                break
        if not replies:
            return ''
        return 'Not run (Oradon acted instead): the independent check confirmed ' + self.confirmed_candidate + ', so Oradon set the cart (' + '; '.join(done) + ') and checked out. Store reply: ' + ' | then: '.join(replies)

    def is_checkout(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.roles.get('buy'):
            return False
        fixed = self.reading.get('buy_args') if isinstance(self.reading.get('buy_args'), dict) else {}
        return all(args.get(k) == v for (k, v) in fixed.items())

    def is_refusal(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.roles.get('refuse'):
            return False
        fixed = self.reading.get('refuse_args') if isinstance(self.reading.get('refuse_args'), dict) else {}
        return all(args.get(k) == v for (k, v) in fixed.items())

    def tools_offered(self) -> list[dict[str, Any]]:
        left = self.steps_left()
        ask = self.roles.get('ask')
        keep: set[str] | None = None
        ends = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        if left <= self.final_steps():
            keep = set(ends)
        elif left <= self.final_steps() + 3 or self.wrapping_up():
            keep = set(ends) | {self.roles.get('cart'), self.roles.get('cart_view')}
        elif left <= len(self.products) + 5 and self.checkouts == 0:
            keep = set(ends) | {self.roles.get('cart'), self.roles.get('cart_view'), self.roles.get('offer')}
        tools = [t for t in self.tools if t['function']['name'] != ask]
        if keep is not None and ends:
            narrowed = [t for t in tools if t['function']['name'] in keep]
            tools = narrowed or tools
        return tools

    def quiet_refusal(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        out = dict(args)
        for (key, spec) in props.items():
            if not isinstance(spec, dict) or 'string' not in _schema_types(spec) or 'enum' in spec:
                continue
            value = out.get(key)
            if value is None and key not in (schema.get('required') or []):
                continue
            text = str(value or '')
            if text and self.reading.get('quiet_decline') is False:
                continue  # the request asks for alternatives: the text that names them stays
            named = any(listing_id in text for listing_id in self.listings) or self.reading.get('quiet_decline') is not False
            if named or not text:
                out[key] = 'Sorry, nothing in the store meets all of your requirements, so I am not buying anything.'[:spec.get('maxLength', 200)]
        return out

    def ids_from_titles(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if name not in {self.roles.get(r) for r in ('cart', 'open', 'offer', 'compare')} - {None} or not self.id_field:
            return (args, '')
        props = (self.schemas.get(name) or {}).get('properties') or {}
        free = [k for (k, v) in props.items() if isinstance(v, dict) and ('string' in _schema_types(v) or 'array' in _schema_types(v)) and ('enum' not in v)]
        if len(free) != 1 or (name != self.roles.get('cart') and (name, free[0]) not in self.id_arguments):
            return (args, '')

        def fix(value: Any) -> tuple[Any, str]:
            if isinstance(value, str) and value.strip() in self.listings:
                return (value.strip(), '')
            if not isinstance(value, str) or _looks_like_id(value, True) or len(value.strip()) < 8:
                return (value, '')
            low = value.strip().lower()
            hits = [i for (i, row) in self.listings.items() if any(k != self.id_field and isinstance(v, str) and v.strip().lower().startswith(low) for (k, v) in row.items())]
            if len(hits) == 1:
                return (hits[0], '')
            if hits:
                return (value, f"several listings carry the text {value!r} ({', '.join(hits[:6])}); pass one listing id")
            return (value, f'{value!r} is not a listing id; pass the id from the result rows (their {self.id_field} field)')
        out = dict(args)
        for key in free:
            if key not in args:
                continue
            fixed = [fix(v) for v in args[key]] if isinstance(args[key], list) else [fix(args[key])]
            problem = next((p for (_, p) in fixed if p), '')
            if problem:
                return (args, problem)
            out[key] = [v for (v, _) in fixed] if isinstance(args[key], list) else fixed[0][0]
        return (out, '')

    def cart_sets_quantity(self, name: str) -> bool:
        props = self.schemas.get(name, {}).get('properties') or {}
        if any(isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) >= 2) for s in props.values()):
            return False
        return any(isinstance(s, dict) and 'integer' in _schema_types(s) and (s.get('minimum') == 0) for s in props.values())

    def track_cart(self, name: str, args: dict[str, Any]) -> None:
        self.cart_changed = True
        self.cart_changed_step = self.step
        spec = self.schemas.get(name, {}).get('properties') or {}
        removal = False
        for (key, value) in args.items():
            options = (spec.get(key) or {}).get('enum')
            if isinstance(options, list) and len(options) >= 2 and (value in options[1:]):
                removal = True
        loose_count = next((_as_count(v) for v in args.values() if _as_count(v) is not None and (not isinstance(v, str))), None)
        lines: list[tuple[str, int | None]] = []
        whole = False
        for (key, value) in args.items():
            if isinstance(value, list):
                shape = (spec.get(key) or {}).get('items') or {}
                if not value and isinstance(shape, dict) and (shape.get('type') == 'object' or 'properties' in shape):
                    whole = True
                for line in value:
                    if isinstance(line, dict):
                        whole = True
                        listing_id = next((v for v in line.values() if isinstance(v, str) and v in self.listings), None)
                        count = next((_as_count(v) for v in line.values() if _as_count(v) is not None and (not isinstance(v, str))), 1)
                        if listing_id:
                            lines.append((listing_id, count))
                    elif isinstance(line, str) and line in self.listings:
                        lines.append((line, loose_count))
            elif isinstance(value, str) and value in self.listings:
                lines.append((value, loose_count))
        if whole:
            self.cart_counts = {i: c for (i, c) in lines if c and c > 0}
            return
        sets = self.cart_sets_quantity(name)
        for (listing_id, count) in lines:
            have = self.cart_counts.get(listing_id, 0)
            if removal:
                have = have - count if count else 0
            elif sets:
                have = count if count is not None else 1
            else:
                have = have + (count if count is not None else 1)
            if have > 0:
                self.cart_counts[listing_id] = have
            else:
                self.cart_counts.pop(listing_id, None)

    def result_text(self, observation: Any) -> str:
        text = _compact(_clipped(observation, VALUE_CHARS))
        if len(text) > RESULT_CHARS:
            text = text[:RESULT_CHARS] + '...(cut)'
        return text

    def compare_list_argument(self) -> str | None:
        schema = self.schemas.get(self.roles.get('compare') or '') or {}
        props = schema.get('properties') or {}
        target = next((k for (k, v) in props.items() if isinstance(v, dict) and 'array' in _schema_types(v)), None)
        if not target or any(k != target for k in schema.get('required') or []):
            return None
        return target

    def compare_capacity(self) -> int:
        spec = ((self.schemas.get(self.roles.get('compare') or '') or {}).get('properties') or {}).get(self.compare_list_argument() or '') or {}
        if isinstance(spec.get('maxItems'), int):
            return spec['maxItems']
        if self.compare_failed:
            return max(2, min(self.compare_failed - 1, max(self.compare_worked, 2)))
        return max(4, self.compare_worked)

    def merge_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        (opener, joiner, target) = (self.roles.get('open'), self.roles.get('compare'), self.sole_id_argument(self.roles.get('open')))
        many = self.compare_list_argument()
        if not opener or not joiner or (not target) or (not many) or (opener == joiner):
            return calls
        if self.compare_is_bare():
            return calls
        selected: list[tuple[int, str]] = []
        for (index, call) in enumerate(calls):
            function = call.get('function') or {}
            args = _json_object(function.get('arguments')) if isinstance(function.get('arguments'), str) else None
            listing_id = str((args or {}).get(target) or '').strip()
            if function.get('name') == opener and args and (set(args) == {target}) and (listing_id in self.listings) and (listing_id not in self.sold_out_step):
                selected.append((index, listing_id))
        selected = selected[:self.compare_capacity()]
        if len(selected) < 2:
            return calls
        first = selected[0][0]
        out = []
        for (index, call) in enumerate(calls):
            if index == first:
                out.append({'id': call['id'], 'type': 'function', 'function': {'name': joiner, 'arguments': _compact({many: [i for (_, i) in selected]})}})
            elif index in {i for (i, _) in selected}:
                out.append({'id': call['id'], 'joined': True})
            else:
                out.append(call)
        self.trace.append({'kind': 'joined', 'ids': [i for (_, i) in selected]})
        return out

    def run_tool_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        calls = self.merge_opens(calls)
        for (index, call) in enumerate(calls):
            allowed = {t['function']['name'] for t in self.tools_offered()}
            call_id = call['id']
            if call.get('joined'):
                out.append((call_id, 'Shown in the side-by-side result above: Oradon opened these listings together in one step.'))
                continue
            function = call.get('function') or {}
            name = function.get('name')
            raw = function.get('arguments')
            if self.finished:
                out.append((call_id, 'Not run: the episode is over.'))
                continue
            if name == self.roles.get('ask'):
                out.append((call_id, "Not run (no step used): Oradon's interviewer handles every message to the shopper."))
                continue
            if name not in self.schemas:
                out.append((call_id, f"Not run (no step used): unknown tool. Tools: {', '.join(sorted(allowed))}."))
                continue
            if name not in allowed:
                out.append((call_id, 'Not run (no step used): too few steps are left for that; use: ' + ', '.join(sorted(allowed)) + '.'))
                continue
            if isinstance(raw, dict):
                args = raw
            else:
                args = _json_object(raw) if (raw or '').strip() else {}
                if args is None:
                    out.append((call_id, 'Not run (no step used): the arguments were not one JSON object.'))
                    continue
            shape_problems: list[str] = []
            args = _coerce_arguments(args, self.schemas[name], 'args', shape_problems)
            if shape_problems:
                out.append((call_id, 'Not run (no step used): ' + '; '.join(shape_problems[:6])))
                continue
            (args, wrong) = self.ids_from_titles(name, args)
            if wrong:
                out.append((call_id, 'Not run (no step used): ' + wrong + '.'))
                continue
            appended_note = ''
            many = self.compare_list_argument() if name == self.roles.get('compare') else None
            if many and isinstance(args.get(many), list) and (len(args[many]) > self.compare_capacity()) and self.compare_failed:
                room = self.compare_capacity()
                appended_note = f"\n[Oradon] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
                args = {**args, many: args[many][:room]}
            target = self.sole_id_argument(name)
            if target and name in (self.roles.get('open'), self.roles.get('offer')) and (str(args.get(target)) in self.sold_out_step):
                out.append((call_id, f'Not run (no step used): {args[target]} sold out at step {self.sold_out_step[str(args[target])]} and will not come back. Replace its cart line with another fitting listing.'))
                continue
            if name == self.roles.get('cart') and any((_as_count(v) or 0) < 0 for v in args.values() if not isinstance(v, str)):
                view = _compact(self.cart_view_result) if self.cart_viewed_step >= self.cart_changed_step else ''
                absent = [v for v in args.values() if isinstance(v, str) and v in self.listings and (v not in self.cart_counts) and (v not in view) and (v not in self.last_cart_reply)]
                if absent:
                    out.append((call_id, f'Not run (no step used): {absent[0]} is not in the cart, so there is nothing to take off. Cart now: ' + self.cart_description()))
                    continue
            ending = self.is_checkout(name, args) or self.is_refusal(name, args)
            if ending and len(calls) > 1:
                out.append((call_id, 'Not run (no step used): send the finishing call on its own, after reading your other results.'))
                continue
            if not ending and self.steps_left() <= self.final_steps() and (self.roles.get('buy') or self.roles.get('refuse')):
                out.append((call_id, "Not run (no step used): the last steps are kept for checking out (with the shopper's approval of the basket when it is due) or ending without a purchase."))
                continue
            signature = name + _compact(args)
            if not ending and name != self.roles.get('cart') and (signature in self.recent_calls):
                out.append((call_id, 'Not run (no step used): you just made this same call and nothing changed; its result is above.'))
                continue
            readers = {self.roles.get(r) for r in ('search', 'open', 'offer', 'compare')} - {None}
            cached = self.result_cache.get(signature)
            if name in readers and cached and (cached[0] == self.task_version):
                out.append((call_id, f'Not run (no step used): the same call at step {cached[1]} returned this, and nothing has changed since:\n' + cached[2][:3000]))
                continue
            if target and name in readers and (name != self.roles.get('compare')) and self.fully_shown(name, str(args.get(target))):
                out.append((call_id, f'Not run (no step used): {args[target]} has shown everything this view gives since the last change. Its record: ' + self.record_text(str(args[target]))))
                continue
            if many and isinstance(args.get(many), list) and args[many]:
                keep = [i for i in args[many] if not self.fully_shown(name, str(i))]
                if not keep:
                    out.append((call_id, 'Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n' + '\n'.join(f'{i}: {self.record_text(str(i))}' for i in args[many])[:4000]))
                    continue
                if len(keep) < len(args[many]):
                    appended_note += f"\n[Oradon] Left out {', '.join(str(i) for i in args[many] if i not in keep)}: shown in full already, nothing changed since."
                    args = {**args, many: keep}
                    signature = name + _compact(args)
            if self.is_refusal(name, args) and (not self.refusal_chain_followed) and self.steps_left() >= 4 and (not self.wrapping_up()):
                live = {requirement for (_, requirement) in self.live_requirements()}
                if any(r in live and r not in self.pointer_values for (_, r) in self.pointer_requirements):
                    self.refusal_chain_followed = True
                    (chain, _) = self.follow_chains()
                    self.trace.append({'kind': 'refusal_chain', 'found': bool(chain), 'steps_left': self.steps_left()})
                    if chain:
                        out.append((call_id, 'Not run yet. A requirement takes its value from another listing, and it was not resolved yet. ' + chain + 'Apply that value to your picks and decide again: a requirement that points at another listing is never by itself a reason to end without a purchase.'))
                        continue
            if self.is_checkout(name, args) and len(calls) == 1:
                lines_text = self.open_lines_with_page_tools()
                if lines_text:
                    out.append((call_id, 'Not run yet. ' + lines_text + '\nCart now: ' + self.cart_description()))
                    continue
            if ending and self.steps_left() > 2 and (not self.wrapping_up()):
                if self.is_checkout(name, args):
                    wider = self.wider_search_hold()
                    if wider:
                        out.append((call_id, wider + '\nCart now: ' + self.cart_description()))
                        continue
                    reread = self.reread_cart_offers()
                    if self.notice_sold_out:
                        self.find_sold_out_line()
                    if self.cart_changed and self.cart_viewed_step < self.cart_changed_step and self.roles.get('cart_view') and (self.steps_left() > 3):
                        shape_problems = []
                        view_args = _coerce_arguments({}, self.schemas.get(self.roles['cart_view']), 'args', shape_problems)
                        view = self.store_call(self.roles['cart_view'], view_args) if not shape_problems else None
                        if view is not None:
                            self.cart_view_result = view.get('observation')
                            self.cart_viewed_step = self.step
                            change = self.take_result(self.roles['cart_view'], self.cart_view_result)
                            if change:
                                self.queued_notes = []
                                out.append((call_id, 'Not run yet (no step used). Looking at the cart showed a change: ' + change + '\nCart now: ' + self.cart_description() + '\nApply the change, fix the cart if needed, then check out.'))
                                continue
                    swap = self.sold_out_hint()
                    held = [i for i in self.cart_listing_ids() if i in self.sold_out_step] if self.sold_out_holds < 2 else []
                    if swap or held:
                        self.sold_out_holds += 0 if swap else 1
                        out.append((call_id, 'Not run yet (no step used). ' + (swap or f"{', '.join(held)} in the cart is sold out: replace that line first.") + '\nCart now: ' + self.cart_description()))
                        continue
                    (chain, hold) = self.follow_chains()
                    if hold:
                        out.append((call_id, 'Not run yet (no step used). ' + chain + '\nCart now: ' + self.cart_description()))
                        continue
                    parts = self.open_part_pages()
                    if parts:
                        out.append((call_id, 'Not run yet. ' + parts + '\nCart now: ' + self.cart_description()))
                        continue
                    misfit = self.compatibility_faults() if self.compatibility_holds < 2 and self.steps_left() > self.final_steps() + 3 else []
                    if misfit:
                        self.compatibility_holds += 1
                        self.trace.append({'kind': 'fit_hold', 'faults': misfit[:4]})
                        out.append((call_id, 'Not run yet (no step used). A part in the cart does not work with the device in the cart: ' + '; '.join(misfit[:3]) + '. Swap the part (or the device), then check out.\nCart now: ' + self.cart_description()))
                        continue
                    shape_problems = self.cart_line_faults() if self.cart_line_holds < 2 and self.steps_left() > self.final_steps() + 2 else []
                    fixed = []
                    if shape_problems and all(' is held at ' in f for f in shape_problems):
                        fixed = self.set_quantities(self.wrong_quantities())
                        shape_problems = self.cart_line_faults() if fixed else shape_problems
                        if fixed and (not shape_problems):
                            self.cart_line_holds += 1
                            out.append((call_id, "Not run yet (no step used). Oradon set the cart's quantities itself: " + '; '.join(fixed) + '. If the store refused any of these, fix those lines; then check out.\nCart now: ' + self.cart_description()))
                            continue
                    if shape_problems:
                        self.cart_line_holds += 1
                        self.trace.append({'kind': 'line_hold', 'faults': shape_problems[:6]})
                        out.append((call_id, 'Not run yet (no step used). The cart does not match the request yet: ' + '; '.join(shape_problems[:6]) + ". Fix those lines (the cart tool's own way to add or take off units), then check out.\nCart now: " + self.cart_description()))
                        continue
                    read = self.open_attributes(self.cart_listing_ids(), self.final_steps() + 3) + self.open_cheaper_rows()
                    verdict = self.review_cart()
                    if verdict:
                        seen = "Oradon read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n' if read else ''
                        out.append((call_id, 'Not run yet. ' + chain + seen + verdict + '\nCart now: ' + self.cart_description()))
                        continue
                    if reread and self.spend > SPEND_SOFT_CAP:
                        out.append((call_id, "Not run yet (no step used). Oradon re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'))
                        continue
                elif self.steps_left() > 5:
                    review = self.audit_refusal()
                    if review and self.audit_covers_every_product():
                        self.confirmed_candidate = self.review_candidates[0]
                        self.trace.append({'kind': 'audit_buy', 'leads': dict(self.review_candidates_by_product), 'steps_left': self.steps_left()})
                        done = self.complete_and_buy()
                        if done:
                            out.append((call_id, done + (' [episode finished]' if self.finished else '')))
                            continue
                    if review:
                        out.append((call_id, 'Not run yet (no step used). ' + review))
                        continue
                    if self.review_candidates and self.steps_left() > 4 and (self.refusal_blocks == 0):
                        self.refusal_blocks = 1
                        self.trace.append({'kind': 'refuse_blocked', 'leads': list(self.review_candidates)})
                        out.append((call_id, 'Not run yet (no step used). The review found listings that appear to fit: ' + ', '.join(self.review_candidates) + '. Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again.'))
                        continue
                    if self.review_candidates and self.steps_left() > 4 and (self.refusal_blocks == 1):
                        self.refusal_blocks = 2
                        first = self.review_candidates[0]
                        self.review_cart([first], SPEND_HARD_CAP)
                        if self.review_verdicts.get(self.review_key([first])) and first in self.review_candidates:
                            self.confirmed_candidate = first
                            self.trace.append({'kind': 'refuse_blocked', 'confirmed': first})
                            out.append((call_id, f"Not run yet (no step used). An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."))
                            continue
            if self.is_refusal(name, args) and self.steps_left() <= 5 and (not self.late_refusal_held):
                lines = [i for i in self.built_lines if not self.is_sold_out(i)]
                fresh = self.cart_view_result is not None and self.cart_viewed_step >= self.cart_changed_step
                missing = [i for i in lines if i not in self.cart_listing_ids() or (i in self.removed_lines and (not fresh))]
                if len(lines) >= len(self.products) and len(missing) + 1 <= self.steps_left():
                    self.late_refusal_held = True
                    self.trace.append({'kind': 'late_decline_held', 'lines': lines, 'missing': missing})
                    redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else 'Check out the cart as it is.'
                    out.append((call_id, 'Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. ' + redo + ' End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again.'))
                    continue
            if self.is_refusal(name, args) and self.steps_left() <= 5 and (not self.refusal_audited) and self.steps_left() >= len(self.products) + 2 and (not self.wrapping_up()):
                review = self.audit_refusal()
                if review and self.review_candidates:
                    self.confirmed_candidate = self.review_candidates[0]
                    self.trace.append({'kind': 'late_refusal_audit', 'leads': list(self.review_candidates), 'steps_left': self.steps_left()})
            if self.is_refusal(name, args) and (not self.review_candidates) and (not self.search_more_told) and (self.steps_left() > self.max_steps // 3):
                self.search_more_told = True
                out.append((call_id, f"Not run yet (no step used). {self.steps_left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if self.is_refusal(name, args):
                done = self.complete_and_buy()
                if done:
                    out.append((call_id, done + (' [episode finished]' if self.finished else '')))
                    continue
                args = self.quiet_refusal(name, args)
            if ending and self.steps_left() >= 2 and self.time_left() > PROCESS_SECONDS:
                (go, said) = self.process_gate('order' if self.is_checkout(name, args) else 'end')
                if not go:
                    out.append((call_id, 'Not run yet. ' + said + '\nCart now: ' + self.cart_description()))
                    continue
                if said:
                    appended_note = '\n[Oradon] ' + said + appended_note
            reply = self.store_call(name, args)
            if reply is None:
                out.append((call_id, 'The call failed (network or server). You may try again.'))
                continue
            observation = reply.get('observation')
            if many and isinstance(args.get(many), list) and (len(args[many]) > 1):
                shown = _compact(observation)
                if not any(str(i) in shown for i in args[many]):
                    self.compare_failed = min(self.compare_failed or 999, len(args[many]))
                    appended_note += f'\n[Oradon] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_capacity()} per call.'
                else:
                    self.compare_worked = max(self.compare_worked, len(args[many]))
            self.recent_calls = (self.recent_calls + [signature])[-3:]
            self.id_arguments.update((name, k) for (k, v) in args.items() if (v if isinstance(v, list) else [v]) and all(isinstance(x, str) and x in self.listings for x in (v if isinstance(v, list) else [v])))
            self.take_in(observation)
            if name == self.roles.get('compare'):
                rows = self.listings_in(observation)
                if rows:
                    self.attributes_shown_by[name] = self.attributes_shown_by.get(name, False) or any(_attribute_block(r) for (_, r) in rows)
            if name in readers - {self.roles.get('search')}:
                self.note_fields_shown(name, observation)
            plain = {tool for (job, tool) in self.roles.items() if job != 'search'}
            paged = self.read_further_pages(name, args, observation) if name not in plain and (not ending) else ''
            if self.sole_id_argument(name) and name not in (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask')):
                listing = self.listing_id_of(observation) if isinstance(observation, dict) else None
                if listing:
                    self.attributes_shown_by[name] = self.attributes_shown_by.get(name, False) or bool(_attribute_block(observation))
                    if name == self.roles.get('open') and (not self.attributes_shown_by[name]):
                        other = next((t for t in self.schemas if t != name and self.sole_id_argument(t) and (self.attributes_shown_by.get(t) is not False) and (t not in (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask'), self.roles.get('search')))), None)
                        if other:
                            self.roles['open'] = other
                    if not self.attributes_shown_by[name] and name != self.roles.get('open') and (not self.roles.get('offer')):
                        self.roles['offer'] = name
                    if name == self.offer_tool_name():
                        self.offer_read_step[listing] = self.step
            if name == self.roles.get('cart'):
                self.track_cart(name, args)
                self.last_cart_reply = _compact(observation)
                self.recent_calls = []
                counts = [_as_count(v) for v in args.values() if not isinstance(v, str) and _as_count(v) is not None]
                for value in args.values():
                    if counts and isinstance(value, str) and (value in self.listings):
                        if counts[0] <= 0 and (not _id_in_rows(observation, value)):
                            self.removed_lines.add(value)
                        elif counts[0] > 0:
                            self.removed_lines.discard(value)
                lines = [i for i in self.cart_listing_ids() if not self.is_sold_out(i) and i not in self.removed_lines]
                if len(lines) >= len(self.built_lines):
                    self.built_lines = lines
            if name == self.roles.get('search') and self.listings_in(observation):
                self.searches_made = (self.searches_made + [dict(args)])[-8:]
            if name == self.roles.get('cart_view'):
                self.cart_view_result = observation
                self.cart_viewed_step = self.step
            if self.is_checkout(name, args):
                self.checkouts += 1
                if not reply.get('finished'):
                    self.refused_step = self.step
            text = self.result_text(observation) + paged + appended_note
            if name not in plain and (not ending):
                text += self.cheapest_rows_note([observation] + self.paged_results) + self.price_cap_note(name, args, observation)
            base = text
            proof = self.evidence_lines(observation)
            if proof:
                text += '\n' + proof
            if name == self.roles.get('cart') and any(_as_count(v) == 0 for v in args.values() if not isinstance(v, str)):
                if any(isinstance(v, str) and v in self.listings and _id_in_rows(observation, v) for v in args.values()):
                    text += '\n[Oradon] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'
            self.notice_sold_out = False
            lines = self.notice_lines(observation)
            if lines:
                meaning = self.read_notice(lines)
                if meaning:
                    text += '\n[note read] ' + meaning + '. Apply this change to your picks before checking out.'
            if self.is_refusal(name, args) and (not reply.get('finished')) and (not self.finished) and self.decline_rearms < 2:
                self.rearm_refusal_checks(meaning if lines else '')
            refused = self.is_checkout(name, args) and (not reply.get('finished'))
            again = refused and any(line in self.reply_words(observation) for line in self.sold_out_notices)
            loose = self.notice_sold_out and (refused or (not self.notice_named_listing and (not self.is_checkout(name, args))))
            if (loose or again) and self.cart_listing_ids() and (not self.finished):
                seen = self.find_sold_out_line()
                if seen:
                    text += "\n[Oradon re-read the cart's lines after the refusal]\n" + seen
            self.notice_sold_out = False
            swap = self.sold_out_hint()
            if swap:
                text += '\n' + swap
            if self.queued_notes:
                text += "\n[notes seen on Oradon's own calls] " + ' | '.join(self.queued_notes)
                self.queued_notes = []
            if name in readers:
                self.result_cache[signature] = (self.task_version, self.step, base)
            if reply.get('finished'):
                text += ' [episode finished]'
            out.append((call_id, text))
        return out

    def forced_finish(self, again: bool=False) -> str:
        if self.finished or self.store_credit_gone or self.steps_left() <= 0:
            return ''
        (buy, refuse) = (self.roles.get('buy'), self.roles.get('refuse'))
        fresh = self.cart_view_result is not None and self.cart_viewed_step >= self.cart_changed_step
        holding = bool(self.cart_listing_ids()) if fresh else bool(self.cart_counts)
        if buy and (holding or (self.cart_changed and (not fresh) and (self.steps_left() >= 2))):
            shape_problems: list[str] = []
            args = _coerce_arguments(dict(self.reading.get('buy_args') or {}), self.schemas.get(buy), 'args', shape_problems)
            if not shape_problems:
                self.open_lines_with_page_tools()
                if self.steps_left() >= 2 and self.time_left() > PROCESS_SECONDS:
                    self.process_gate('order')
                if self.finished or self.steps_left() <= 0:
                    return ''
                reply = self.store_call(buy, args)
                self.trace.append({'kind': 'forced_buy', 'reply': _clipped(reply, 200)})
                if self.finished or self.steps_left() <= 0:
                    return ''
                if reply is not None and again:
                    self.checkouts += 1
                    self.refused_step = self.step
                    self.take_result(buy, reply.get('observation'))
                    return 'the checkout was refused: ' + self.result_text(reply.get('observation'))
        if refuse:
            shape_problems = []
            args = _coerce_arguments(self.quiet_refusal(refuse, dict(self.reading.get('refuse_args') or {})), self.schemas.get(refuse), 'args', shape_problems)
            if not shape_problems:
                if self.steps_left() >= 2 and self.time_left() > PROCESS_SECONDS:
                    self.process_gate('end')
                if self.finished or self.steps_left() <= 0:
                    return ''
                reply = self.store_call(refuse, args)
                self.trace.append({'kind': 'forced_refuse', 'reply': _clipped(reply, 200)})
                if reply is not None and (not self.finished):
                    self.take_result(refuse, reply.get('observation'))
                    return 'ending without a purchase was refused: ' + self.result_text(reply.get('observation'))
        return ''

    def trim_history(self) -> None:
        size = sum(len(_compact(m.get('content'))) for m in self.conversation)
        if size <= HISTORY_CHARS:
            return
        tools = [i for (i, m) in enumerate(self.conversation) if m.get('role') == 'tool']
        for index in tools[:max(1, len(tools) // 2)]:
            content = self.conversation[index].get('content') or ''
            if len(content) > 600:
                ids = re.findall('"([A-Za-z0-9][A-Za-z0-9_\\-:.]{5,40})"', content)
                ids = [i for i in dict.fromkeys(ids) if i in self.listings][:20]
                self.conversation[index]['content'] = content[:400] + f"...(older result shortened; listings in it: {', '.join(ids)})"

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.read_task()
        self.trace.append({'kind': 'brief', 'brief': self.reading})
        opening = self.policy_view.get('opening')
        if opening is not None:
            self.take_in(opening)
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            intro += '\n\nWHAT THE STORE SHOWS FIRST:\n' + self.result_text(opening)
        intro += '\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n' + self.status_text()
        self.conversation = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': intro}]
        self.navigate(NAVIGATOR_TURNS, SPEND_HARD_CAP)
        if not self.finished:
            turned = self.forced_finish(again=True)
            if turned and (not self.finished) and (not self.store_credit_gone) and (not self.model_credit_gone) and (self.steps_left() > 1) and (self.time_left() > NAVIGATOR_SECONDS + 10):
                self.conversation.append({'role': 'user', 'content': '[Oradon] Oradon tried to end the episode and ' + turned + '\nApply what the store says, then finish.\n' + self.status_text()})
                self.navigate(self.navigator_turns + 10, SPEND_HARD_CAP + 0.01)
            if not self.finished:
                self.forced_finish()
        self.trace.append({'kind': 'end', 'finished': self.finished, 'steps': self.step, 'cost': round(self.spend, 5), 'calls': self.model_calls_made, 'seconds': round(time.monotonic() - self.started, 1)})
        return self.trace

    def navigate(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.finished and (not self.store_credit_gone) and (self.steps_left() > 0) and (self.navigator_turns < loop_cap) and (self.spend < spend_cap) and (self.time_left() > NAVIGATOR_SECONDS):
            if self.interview_due():
                summary = self.run_interview()
                if summary:
                    self.conversation.append({'role': 'user', 'content': summary + '\n' + self.status_text()})
                continue
            self.trim_history()
            self.navigator_turns += 1
            message = self.model_turn(self.conversation, [FAST_MODEL, BACKUP_MODEL], 3000, self.tools_offered())
            if message is None:
                self.model_misses += 1
                if self.model_credit_gone or self.model_calls_made >= MODEL_CALLS or self.model_misses > 2:
                    break
                continue
            calls = []
            for (product_number, call) in enumerate(message.get('tool_calls') or []):
                if not isinstance(call, dict):
                    continue
                function = call.get('function') if isinstance(call.get('function'), dict) else {}
                raw = function.get('arguments')
                parsed = raw if isinstance(raw, dict) else _json_object(raw) if isinstance(raw, str) and raw.strip() else {}
                calls.append({'id': str(call.get('id') or f'k{self.navigator_turns}_{product_number}'), 'type': 'function', 'function': {'name': str(function.get('name') or ''), 'arguments': _compact(parsed if isinstance(parsed, dict) else {})}})
            content = message.get('content')
            if not isinstance(content, str):
                content = _compact(content) if content else ''
            if not calls:
                idle += 1
                self.conversation.append({'role': 'assistant', 'content': content or '(no call)'})
                self.conversation.append({'role': 'user', 'content': 'Call one of the tools to continue.\n' + self.status_text()})
                if idle >= 3:
                    break
                continue
            idle = 0
            committing = {self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse')} - {None}
            if not self.interviewed and self.roles.get('ask') and (self.messages_free() > 0) and any(c['function']['name'] in committing for c in calls) and (self.steps_left() > 8):
                self.conversation.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
                for call in calls:
                    self.conversation.append({'role': 'tool', 'tool_call_id': call['id'], 'content': 'Not run (no step used): the shopper is asked first.'})
                summary = self.run_interview()
                self.conversation.append({'role': 'user', 'content': (summary or 'The interviewer had nothing to ask.') + '\n' + self.status_text()})
                continue
            self.conversation.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
            self.trace.append({'kind': 'turn', 'text': content[:300], 'calls': [c['function'] for c in calls]})
            replies = self.run_tool_calls(calls)
            for (index, (call_id, text)) in enumerate(replies):
                if index == len(replies) - 1:
                    text += '\n' + self.status_text()
                self.conversation.append({'role': 'tool', 'tool_call_id': call_id, 'content': text})

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Agent(problem_data)
    except Exception as exc:
        return [{'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]}]
    try:
        return episode.run()
    except Exception as exc:
        episode.trace.append({'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]})
        try:
            if not episode.finished:
                episode.forced_finish()
        except Exception as again:
            episode.trace.append({'kind': 'crash', 'error': f'{type(again).__name__}: {again}'[:400]})
        return episode.trace
__all__ = ['agent_main']
