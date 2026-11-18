"""CodeGod v4: buys for one shopper inside an ORO store world, asking one property of one item per message.

Lineage: a fork of the public YesYesYes v1 agent (itself from Know Thyself v1 and KYO v6). The store rules and most of
the model prompts come from there, and so does the idea of a light model walking the store while code keeps the ledger.
The program around them is CodeGod's own:

- A world reader: stores may answer with listing records as text blocks ("id: x | title: y | attributes_map: a=b; c=d")
  or with wrappers around the result; everything is turned into one normal form at the store boundary, and the model
  sees whole attributes instead of a text cut in the middle of them.
- Store shape, price, stock and page reading by code, with a model call only when the code cannot tell.
- A ranking of attribute keys as the likely hidden need, read from the episode's own records (`key_evidence`).
- A message guard: while a message tool exists and messages are unused, nothing commits, whatever the brief said.
- One property of one item per message, enforced by a last gate (`one_ask`).
- Item choice on multi-item requests from store evidence (`item_pull`) when messages are fewer than items.
- A model-free store hunt before a decline (`Hunter`) on the required values, with unread pages and a code reading of
  the opened listings; the same hunt serves the cheapest-pick sweep.
- A clock that learns call durations, so nothing starts that cannot finish inside the episode.
"""
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
HEAVY_KIN = (HEAVY, 'anthropic/claude-sonnet-5', 'anthropic/claude-sonnet-4.6')
LOOP_CALLS = 42
ALL_CALLS = 60
SPEND_SOFT = 0.052
SPEND_HARD = 0.078
HEAVY_ROOM = 0.02
REFUSAL_HUNT = 6
BACKOFF = (4, 10, 20)
BACKOFF_TOTAL = 40
BUSY_SECONDS = 20
EPISODE_SECONDS = 300
FINISH_MARGIN = 20
EXPLORE_MARGIN = 45
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3
AUTO_PAGES = 6
HUNT_STEPS = 12
SWEEP_STEPS = 6
MENU_KEYS = 14
DEFAULT_MESSAGES = 3
PLAN_PICKS = 6

NOT_PROPERTY = r'\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\w*|video|url|links?|images?|photos?|condition)\b'
CATCH_ALL = r'^(?!.*\b(?:types?|styles?|kinds?|shapes?)\b).*\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\w*|accessor\w*|extras?|misc\w*)\b'
ATTRIBUTE_FIELDS = r'attr|fact|spec|propert|detail|trait'
SHELF_KEYS = r'categor|shelf|department|aisle|section|product.?group'
TITLE_KEYS = r'title|headline|heading|^name$|product.?name'
NOTE_KEYS = r'remark|comment|notes?$|descri|overview|blurb|good.?to.?know|details?.?text|caption|summary|from.?the.?seller|pointers?'
SHOP_KEYS = r'seller|shop|store|vendor|brand|maker|merchant|producer|sold.?by'
PRICE_KEYS = r'price|cost|amount'
STOCK_KEYS = r'stock|on.?hand|avail|can.?buy|buyable|purchas'
COUNT_KEYS = r'count|total|overall|hits?|found|matches|num'
NEXT_KEYS = r'next|token|follow|cursor|continu|offset|page'
PAGE_ARGS = r'page|token|cursor|offset|follow|next|start|after|skip'
LIMIT_WORDS = r'\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or|anything but)\b'
NUMBER_WORDS = {'zero': 0, 'one': 1, 'a single': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6}
FIT_CODE = r'[A-Z][A-Za-z]{0,7}(?:[- ]?\d{3,6}|[- ]?\d{1,5}[xX]{1,4})'
PART_CODE = r'[A-Z]{1,5}\d{0,3}-?\d{3,7}'
PART_SERIES = r'[A-Z]{1,5}\d{1,3}'
FIT_WORDS = r'(?:works? (?:with|on)|compatible with|compatibility|fitment|fits?|to fit|for use with|usable with|designed for|made for|built for|intended for|suits?|goes with|pairs? with|replacement for|replaces|supports?)'
FIT_EXCEPT = r'\b(?:except(?:\s+for)?|excluding|other than|but not|apart from|not including|not for)\b'

_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=60, max_retries=2, retry_delay=1)

GUIDE = 'You are a careful buyer working for one shopper in an online store. You can act only by calling the store\'s tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.\n\nHOW THE STORE WORKS\n- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool\'s description and schema and use the exact argument names and allowed values.\n- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.\n- Search rows are summaries. Stores split a listing\'s record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once the desk knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.\n- Questions to the shopper are handled for you by the desk\'s interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.\n- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. The desk carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store\'s tools (product questions go through the desk\'s interviewer).\n\nWHAT COUNTS AS A FIT\n- The kind of product is decided by the listing\'s category label: only the label that matches the requested item as a whole counts, never a neighbouring one.\n- A requirement on a property is met when the listing\'s attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value\'s words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.\n- An attribute whose key names the requirement\'s property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. When several keys are that property (its bare name, or its name with the product\'s name added: finish, lamp finish, lamp_finish), one that states the value is enough. A key with other words in it (a strap finish for finish) is not that property. A key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing about the requirement. The desk adds [evidence] lines after opened listings with the keys whose names match a requirement.\n- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.\n- When the category label itself states a requirement\'s value (a season or audience in the label), that requirement is met by the category.\n- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing\'s value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.\n- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.\n- When one requested item must fit another (an accessory for a device), choose the two together from the listings\' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.\n- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.\n- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.\n\nCHOOSING\n- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and the desk reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.\n- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.\n- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.\n- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.\n- A request for several different items needs one cart line per item, each at its requested quantity.\n\nCHANGES WHILE YOU WORK\n- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool\'s result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.\n- Re-read a chosen listing\'s record shortly before checking out, because price and stock can change silently.\n\nFINISHING\n- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.\n- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request\'s wording for that ending, and when a remark is needed, keep it short and name no listing.\n- Every reply must call at least one tool. Keep your text short: what you know, then the next move.'
BRIEF = 'You prepare a shopping task for an agent. Read the shopper\'s request and the store\'s tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:\n\n"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request\'s words, "shelf": the store\'s category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request\'s own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property\'s value or limit outright; and never turn words that are part of what the product is called (a season or an audience named with the product) into musts: those belong to "name". A use or occasion the request makes a property of the product itself ("blazers for business wear", "a dress made for parties", "the occasion has to be party") IS a must on that property ("occasion: business"), unless the store\'s category label already names it; an event, person or setting the product is bought for ("to wear at his wedding", "for a colleague"), and how it should sit, feel or look in use, stay context. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.\n"budget": the most the shopper will pay in total as a number in the request\'s own unit, or null.\n"budget_unit": that unit in words (for example "cents"), or null.\n"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.\n"in_stock_only": true unless the request clearly accepts items that are not available.\n"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.\n"pick_measure": what is maximised or minimised for "most"/"least", else "".\n"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.\n"steps": how many steps, moves or tool calls the request allows (integer), or null.\n"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].\n"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.\n"wishes": a list of things the shopper would merely like.\n"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).\n"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper\'s approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps. A message that only tells the shopper nothing fits is not a step: the ending itself says so, so it takes 0 messages.\n"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.\n"fits": a list of {"item": item number, "with": item number} for each requested item that the request says must work with, fit or suit ANOTHER requested item (the one the agent picks); [] when none.\n"browse": how a tool lists one store category, when one can: {"tool": exact tool name, "category": the argument that takes the category label, "max_price": the argument that takes a maximum price, or null, "fixed": the other arguments that put the tool in that mode, else {}}; null when no tool lists a category.\n"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing\'s attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing\'s current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).\n"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.\n"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.\n'
PLAN = 'You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property; a property the shopper has no view on gets "don\'t mind", and one the shopper does not think about at all gets a puzzled reply, so every property named spends part of a message.\n\nYou get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item\'s category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.\n\nChoose the properties in this order:\n0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.\n1. Hinted properties. The request\'s story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product\'s looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.\nName a hinted property by a listed key of that item when one fits the hint (a hint about who will wear it: the key for gender or audience; about its looks: the listed key for colour, pattern or finish that most of its listings carry); a property no listing of the item carries comes after the item\'s three most carried keys.\n2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item\'s listings; once five or more listings were read, avoid keys seen on only one or two of them.\nSkip a property the request already settles for that item (a requirement names that property; a key whose values only repeat a required value, such as a pattern key showing the required decoration, is NOT settled), a second name for a property already chosen, a key whose values each list several purchase options at once, identifiers, model names, warranty and other logistics. Pack sizes, counts and quantities per pack are properties like any other, and a key marked [not the stated ...] is another property than that requirement, whatever words its name shares with it. Rank every item on its own: its hinted properties first, then its listed keys.\n"label" is the property\'s plain name: for a listed key, all of the key\'s words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property\'s name, no example values, no brackets, no second property.\nChoose up to COUNT for each item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story\'s words it comes from, or an empty string"}]}, each item\'s picks in rank order'
HEAR = 'A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "budget": new total limit as a number or null}\n- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3").\n- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop\'s product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer\'s own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.\n- "free": the property of each question the shopper said they do not mind about.\n- Folksy or roundabout wording that names a value still states that value.\n- A negative about the shopper\'s own flexibility (it won\'t budge, isn\'t up for trade, not negotiable, no room on it) wants the value it names: "material coming in as nylon isn\'t up for trade" states nylon. Write "not X" only when the answer turns X itself down.\n- The value is only what the property should be. Words that merely join a property to its value (such as matching, set to or coming in as) are not part of it: "size set to large" states large. This does not touch a value pointed at on another listing, which is copied whole.'
NOTE = 'While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:\n{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request\'s unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}\nA replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note.'
CHECK = 'You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.\nReading rules: a requirement is met when the listing\'s attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value\'s words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails, unless another key that is that property (its bare name, or its name with the product\'s name added) states the value, since one is enough; a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing\'s value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.\nWhen the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart\'s choice on that rule, the cart is wrong.\nReturn ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses.'
READING = 'the listing\'s attribute for a property decides when its key names that same property (when several keys do, as its bare name or its name with the product\'s name added, one stating the value is enough; any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value\'s words counts, such as "deep teal" for teal, limits include the edge); a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing\'s value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit.'
AUDIT_RULES = '\nRecords marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.\nThe requirements were written down by a quick reader, so first compare each one with the REQUEST and the ANSWERS FROM THE SHOPPER. A requirement that only repeats the story (who it is for, when, where or why it will be used, what went wrong before, how it should feel) and that the request does not demand of the product goes under "not_firm". A value that picked up words which only join a property to its value goes under "corrected" with the plain value. Judge the listings without the not_firm requirements and with the corrected values. A requirement the request or an answer states outright stays, even when no listing meets it: ending without a purchase is then right.\nReturn ONE JSON object and nothing else: {"not_firm": [{"item": item number, "need": the requirement copied exactly}], "corrected": [{"item": item number, "need": the requirement copied exactly, "value": "property: plain value"}], "fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "search": [at most 4 of {"item": item number, "query": "a keyword search of two or three words, not tried yet, that could surface a fitting listing: another name for the product with one required value (long queries often match nothing)"}], "reason": "one sentence"}'
UNMIX = 'A shopper\'s request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request\'s wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}'
TRAIL = 'A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing\'s page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement\'s chain from the requirement\'s own description, link by link, through the pages\' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).\nReturn ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing, "find": "for a link not found: search words (its shop and the values the description gives)"}], "done": true when the last link is an OPENED page that states the property\'s value itself (also when the requirement asks for a value other than or unlike that one: the page\'s value is what it is measured against), else false, "value": the property\'s value as stated on that last page, or ""}]}\nA record is a link only when it matches the whole description of that link: the shop or brand, the kind of product AND every property value the description gives (for example "a 10 pieces multipack" rules out the same shop\'s listing with 10 capsules and no multipack); when no record matches all of it, that link is not found yet, even if the same shop sells other listings of that kind. Give "find" for such a link: the shop\'s name and the description\'s values, as a few search words.\nList each chain\'s links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown.'
GATE = 'A shopping agent is about to STAGE. The shopper\'s request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product\'s properties do not do a step.\nWhen a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.\nReturn ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}'
SWAP_RULES = '\nReturn ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line\'s item, at most 6], "reason": "one sentence"}'
SHAPE = 'A store tool returned the result below. Name its parts by the exact keys the result and the tool\'s parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing\'s price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}'
AUDIT = 'A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: ' + READING + AUDIT_RULES
SWAP = "A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: " + READING + SWAP_RULES

def jdump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        return str(value)


def clipped(value: Any, limit: int, depth: int = 0) -> Any:
    """A copy with long strings cut; attribute blocks are left whole so a requirement's evidence always reaches the model."""
    if depth > 7:
        return '...'
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + '...'
    if isinstance(value, dict):
        return {k: clipped(v, limit, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [clipped(v, limit, depth + 1) for v in value]
    return value


def json_object(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    start, stop = text.find('{'), text.rfind('}')
    if not 0 <= start < stop:
        return None
    try:
        found = json.loads(text[start:stop + 1])
    except ValueError:
        return None
    return found if isinstance(found, dict) else None


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value] if value and isinstance(value, (str, dict)) else []


def whole_number(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and float(value).is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r'\s*\d+(\.0+)?\s*', value):
        return int(float(value))
    return None


def amount(value: Any) -> float | None:
    """A price read from a number, a numeric string, or a small price object."""
    if isinstance(value, dict):
        value = next((v for v in value.values() if isinstance(v, (int, float)) and not isinstance(v, bool)), None)
    if isinstance(value, str) and re.fullmatch(r'\s*\d+(\.\d+)?\s*', value):
        value = float(value)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def spaced(key: Any) -> str:
    """A key in plain lower-case words, whatever its spelling."""
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', str(key))
    return re.sub(r'[\s_.\-]+', ' ', text).strip().lower()


def singular(word: str) -> str:
    if len(word) > 3 and re.search(r'(?:ch|sh|ss|x|z)es$', word):
        return word[:-2]
    return word[:-1] if len(word) > 3 and word.endswith('s') and not word.endswith('ss') else word


def word_set(text: Any, shortest: int = 1) -> set[str]:
    return {singular(w) for w in re.findall(r'[a-z0-9]+', spaced(text)) if len(w) >= shortest}


def names_property(prop: Any, key: Any) -> bool:
    """Whether an attribute key names a property, by whole words in either spelling."""
    a, b = word_set(prop), word_set(key)
    return bool(a) and bool(b) and (a <= b or (len(b) >= 2 and b <= a))


def has_words(needle: Any, text: str) -> bool:
    words = ' '.join(re.findall(r'[a-z0-9]+', str(needle).lower()))
    return bool(words) and re.search(r'(?<![a-z0-9])' + re.escape(words) + r'(?![a-z0-9])', text) is not None


def catch_all(key: str) -> bool:
    return bool(re.search(CATCH_ALL, spaced(key)))


def message_count(request: str) -> int | None:
    """How many messages a request allows, read from its own words when they state it."""
    found = re.search(r'\b(\d+|zero|one|a single|two|three|four|five|six)\s+(?:\w+\s+){0,2}?(?:messages?|questions?)\b', request, re.I)
    if found:
        word = found.group(1).lower()
        return int(word) if word.isdigit() else NUMBER_WORDS.get(word)
    return None


def key_evidence(cover: float, spellings: int, values: dict[str, int], titled: bool, noted: bool, name: str, measured: float) -> float:
    """CodeGod's ranking of an attribute key as the shopper's unwritten need, from the listings read so far.

    A key earns weight when most of the item's listings carry it, when the store writes it under two spellings, when
    its values differ, when titles write its values and when sellers' notes name it. A key showing one value everywhere,
    a catch-all field and a bare measurement lose weight. Round weights, read from the store's records alone.
    Measured on recorded episodes: hidden keys carry two spellings 3.3 times as often as other keys (0.40 vs 0.12), sellers' notes
    name them 2.9 times as often (0.32 vs 0.11), and they are flat half as often (0.21 vs 0.59).
    """
    seen = sum(values.values())
    differ = len(values) >= 2
    spread = seen >= 2 and max(values.values() or [0]) * 2 < seen
    flat = len(values) == 1 and seen >= 2
    return (cover + 0.5 * (spellings >= 2) + 0.25 * differ + 0.25 * spread + 0.25 * bool(titled) + 0.5 * bool(noted)
            - 0.5 * flat - 0.5 * catch_all(name) - 0.25 * (measured > 0.5))


# ---- turning whatever a store answers into one normal form ----

def plain_scalar(text: str) -> Any:
    low = text.strip()
    if low in ('True', 'true'):
        return True
    if low in ('False', 'false'):
        return False
    if low in ('None', 'null', ''):
        return None
    if re.fullmatch(r'-?\d+', low):
        return int(low)
    if re.fullmatch(r'-?\d+\.\d+', low):
        return float(low)
    return low


def read_pairs(text: str, named: bool = False) -> dict[str, Any] | None:
    """Attributes written inline as text ("a=b; c=d"), or as JSON. A field the store calls attributes may also use a
    colon or hold one pair. A key with an empty value says nothing and is left out."""
    text = text.strip()
    if named and text.startswith('{'):
        try:
            found = json.loads(text)
            return found if isinstance(found, dict) else None
        except ValueError:
            return None
    for sign in ('=', ':') if named else ('=',):
        pieces = re.split(r'\s*[;,|]\s*(?=[^;,|=:]{1,60}' + sign + ')', text)
        if len(pieces) < (1 if named else 2) or not all(sign in p and len(p.partition(sign)[0]) <= 60 for p in pieces):
            continue
        pairs = {p.partition(sign)[0].strip(): p.partition(sign)[2].strip().rstrip(';') for p in pieces}
        return {k: v for k, v in pairs.items() if k and v} or None
    return None


def key_style(key: str) -> tuple[bool, bool]:
    """How a store writes its field names: with spaces or not, starting in capitals or not."""
    return ' ' in key, key[:1].isupper()


def split_fields(text: str) -> list[tuple[str, str]] | None:
    """The (key, value) fields of a record written as text, or None. The separator is the first of the usual ones that
    gives three or more "key: value" pieces, starting with the first piece. A separator inside a value (a title with " | "
    in it) does not start a field unless what follows is written like the record's first key, so such a piece goes back
    onto the value before it."""
    best: list[str] = []
    chosen = ''
    for mark in (' | ', ' || ', ' ; ', ' ~ ', ' • ', ' :: ', '\n', '\t'):
        pieces = text.split(mark)
        marked = [p for p in pieces if re.match(r'\s*[A-Za-z_][\w -]{0,40}\s*[:=] ?', p)]
        if len(marked) >= 3 and pieces[0] in marked:
            best, chosen = pieces, mark
            break
    if not best:
        return None
    first = re.match(r'\s*([A-Za-z_][\w -]{0,40}?)\s*[:=] ?', best[0])
    if not first:
        return None
    style = key_style(first.group(1))
    fields: list[tuple[str, str]] = []
    for piece in best:
        found = re.match(r'\s*([A-Za-z_][\w -]{0,40}?)\s*[:=] ?(.*)$', piece, re.S)
        if found and key_style(found.group(1)) == style:
            fields.append((found.group(1), found.group(2).strip()))
        elif fields:
            fields[-1] = (fields[-1][0], fields[-1][1] + chosen + piece)
        else:
            return None
    return fields


def read_block(text: Any) -> dict[str, Any] | None:
    """A listing written as one text ("id: x | title: y | ...") as a record, or None when the text is not one: it must
    have at least three fields, and one of them must hold an id."""
    if not isinstance(text, str) or len(text) < 12:
        return None
    fields = split_fields(text)
    if not fields or len(fields) < 3:
        return None
    named = [v for k, v in fields if re.search(r'(?:^| )(?:id|ref|sku|asin|code)(?: |$)', spaced(k)) and looks_like_id(v, True) and re.search(r'\d', v)]
    if not named and not any(looks_like_id(v) for _, v in fields[:2]):
        return None
    record: dict[str, Any] = {}
    for key, value in fields:
        record[key] = read_pairs(value, bool(re.search(ATTRIBUTE_FIELDS, spaced(key)))) or plain_scalar(value)
    return record


def restore_records(node: Any, depth: int = 0) -> Any:
    """The result with every listing written as text turned into a record; anything else comes back untouched, so a
    store that answers in structured records is not changed at all. A text field of a larger object is that object's
    own value (a description may look like a record), so only list entries and lone or wrapped results are read."""
    if depth > 8:
        return node
    if isinstance(node, str):
        return read_block(node) or node
    if isinstance(node, list):
        return [restore_records(v, depth + 1) for v in node]
    if isinstance(node, dict):
        if len(node) == 1:
            only = next(iter(node.values()))
            parsed = read_block(only)
            if parsed:
                return parsed
        return {k: v if isinstance(v, str) and len(node) > 2 else restore_records(v, depth + 1) for k, v in node.items()}
    return node


def take_off_wrapper(node: Any, remarks: list[str]) -> Any:
    """The result without its wrapper: a two-key object where one key holds plain sentences and the other the result.
    Whatever the keys are called, the sentences are the shopper or the store speaking; they go to `remarks`."""
    if isinstance(node, dict) and len(node) == 2:
        said = [k for k, v in node.items() if isinstance(v, list) and v and all(isinstance(x, str) and len(x.split()) >= 3 for x in v)]
        if len(said) == 1:
            remarks.extend(s for s in node[said[0]] if s.strip())
            return take_off_wrapper(next(v for k, v in node.items() if k != said[0]), remarks)
    return node


def normal_form(raw: Any, remarks: list[str]) -> Any:
    return take_off_wrapper(restore_records(raw), remarks)


def facts_block(node: Any) -> dict[str, Any] | None:
    """The largest object of plain values inside a record: its attributes."""
    if not isinstance(node, dict):
        return None
    best = None
    for key, value in node.items():
        if isinstance(value, dict) and value:
            plain = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
            enough = 1 if re.search(ATTRIBUTE_FIELDS, spaced(key)) else 2
            if len(plain) >= enough and (best is None or len(plain) > len(best)):
                best = plain
    return best


def looks_like_id(value: Any, named: bool = False) -> bool:
    if not isinstance(value, str) or ' ' in value.strip() or not 3 <= len(value.strip()) <= 64:
        return False
    return named or (bool(re.search(r'\d', value)) and bool(re.search(r'[A-Za-z]', value)) and bool(re.search(r'[-_:]', value)))


# ---- arguments checked against a tool's schema ----

def type_names(schema: dict[str, Any]) -> list[str]:
    kind = schema.get('type')
    return [kind] if isinstance(kind, str) else [k for k in kind or [] if isinstance(k, str)]


def is_kind(value: Any, kind: str) -> bool:
    numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
    return {'string': isinstance(value, str), 'integer': numeric and isinstance(value, int), 'number': numeric,
            'boolean': isinstance(value, bool), 'array': isinstance(value, list), 'object': isinstance(value, dict),
            'null': value is None}.get(kind, True)


def coerce(value: Any, kinds: list[str]) -> Any:
    for kind in kinds:
        if kind == 'integer' and isinstance(value, float) and value.is_integer():
            return int(value)
        if kind in ('integer', 'number') and isinstance(value, str) and re.fullmatch(r'\s*-?\d+(\.\d+)?\s*', value):
            number = float(value)
            return int(number) if kind == 'integer' or number.is_integer() else number
        if kind == 'string' and isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        if kind == 'boolean' and isinstance(value, str) and value.strip().lower() in ('true', 'false'):
            return value.strip().lower() == 'true'
        if kind == 'array' and value is not None and not isinstance(value, list):
            return [value]
    return value


def conform(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    """The value fitted to its schema where a fix is plain; what cannot be fitted is reported in `faults`."""
    if not isinstance(schema, dict):
        return value
    kinds = type_names(schema)
    if kinds and not any(is_kind(value, k) for k in kinds):
        value = coerce(value, kinds)
        if not any(is_kind(value, k) for k in kinds):
            faults.append(f"{where} should be {'/'.join(kinds)}")
            return value
    options = schema.get('enum')
    if isinstance(options, list) and value not in options:
        same_case = [o for o in options if isinstance(o, str) and isinstance(value, str) and o.lower() == value.strip().lower()]
        value = same_case[0] if same_case else value
        if value not in options:
            faults.append(f'{where}: allowed values are {options}')
    if isinstance(value, str):
        if isinstance(schema.get('maxLength'), int):
            value = value[:schema['maxLength']]
        if isinstance(schema.get('minLength'), int) and len(value) < schema['minLength']:
            faults.append(f'{where} is too short')
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        low, high = schema.get('minimum'), schema.get('maximum')
        if isinstance(low, (int, float)) and value < low:
            faults.append(f'{where} is below {low}')
        if isinstance(high, (int, float)) and value > high:
            faults.append(f'{where} is above {high}')
    elif isinstance(value, list):
        if isinstance(schema.get('maxItems'), int) and len(value) > schema['maxItems']:
            faults.append(f"{where} allows at most {schema['maxItems']} entries")
        if isinstance(schema.get('minItems'), int) and len(value) < schema['minItems']:
            faults.append(f"{where} needs at least {schema['minItems']} entries")
        if isinstance(schema.get('items'), dict):
            value = [conform(v, schema['items'], f'{where}[{i}]', faults) for i, v in enumerate(value)]
    elif isinstance(value, dict):
        props = schema.get('properties') if isinstance(schema.get('properties'), dict) else {}
        if schema.get('additionalProperties') is False:
            faults.extend(f'{where}.{n} is not an allowed argument' for n in value if n not in props)
        faults.extend(f'{where}.{n} is required' for n in schema.get('required') or [] if n not in value)
        value = {k: conform(v, props[k], f'{where}.{k}', faults) if k in props else v for k, v in value.items()}
    return value


def text_args(schema: dict[str, Any]) -> list[str]:
    props = schema.get('properties') or {}
    return [k for k, v in props.items() if isinstance(v, dict) and 'string' in type_names(v) and 'enum' not in v]


def lone_target(schema: dict[str, Any]) -> str | None:
    """The one free-text argument a tool needs, when it needs no other."""
    texts = text_args(schema)
    required = schema.get('required') or []
    target = next((k for k in required if k in texts), None) or (texts[0] if len(texts) == 1 else None)
    return target if target and not any(k != target for k in required) else None


# ---- what parts and devices say about fitting each other ----

def code_list(text: Any) -> list[str]:
    return [re.sub(r'[\s-]+', '', c).lower() for c in re.findall(r'\b' + FIT_CODE + r'\b', str(text or ''))]


def code_named(stated: str, code: str) -> bool:
    """A stated code names a model code exactly, or as a series with x's in place of digits."""
    tail = re.search(r'(?<=\d)x+$', stated)
    return code.startswith(stated[:tail.start()]) and len(code) == len(stated) if tail else stated == code


def fit_statements(text: str) -> dict[str, Any]:
    """Codes a listing's text names: its own, those it fits, those it says it does not, and the series it fits."""
    own = re.search(r'[\[(]\s*(?:part\s+)?(' + PART_CODE + r')\s*[\])]', text, re.I)
    marks: dict[str, Any] = {'own': own.group(1).upper() if own else '', 'codes': [], 'fits': [], 'not': [], 'series': []}
    for hit in re.finditer(r'\bmodel\s*(?:no\.?|number|#)?\s*[:.]?\s*(' + FIT_CODE + r')\b|\b(?:also sold as|also known as|sold as)\s+([^.;]+)', text, re.I):
        marks['codes'] += code_list(hit.group(1) or hit.group(2))
    for hit in re.finditer(r'\b(?<!not )(?<!never )' + FIT_WORDS + r'\s*:?\s+([^.;]+)', text, re.I):
        if re.match(r'part\s+series', hit.group(1), re.I):
            continue
        sides = re.split(FIT_EXCEPT, hit.group(1), maxsplit=1, flags=re.I)
        marks['fits'] += code_list(sides[0])
        marks['not'] += code_list(sides[1]) if len(sides) > 1 else []
    for hit in re.finditer(r'\bfits?\s+part\s+series\s*:?\s*([^.;]+)', text, re.I):
        marks['series'] += [c.upper() for c in re.findall(r'\b' + PART_SERIES + r'\b', hit.group(1))]
    return marks


def fit_verdict(part: dict[str, Any], device: dict[str, Any]) -> bool | None:
    """True or False when the part's statements can be held against this device, None when they say nothing about it."""
    if device['series'] and re.match(PART_SERIES + '-', part['own']):
        return any(part['own'] == s or part['own'].startswith(s + '-') for s in device['series'])
    family = lambda code: re.match(r'[a-z]*', code).group(0)
    named = {family(c) for c in part['fits']}
    codes = [c for c in device['codes'] if family(c) in named]
    if not codes:
        return None
    if any(code_named(d, c) for d in part['not'] for c in codes):
        return False
    return any(code_named(a, c) for a in part['fits'] for c in codes)


class Clock:
    """Episode time. Learns how long each kind of call takes, so that nothing starts that could not finish in time."""
    FIRST_GUESS = {LIGHT: 8.0, HEAVY: 20.0, SPARE: 10.0, 'store': 1.5}

    def __init__(self) -> None:
        self.began = time.monotonic()
        self.usual: dict[str, float] = dict(self.FIRST_GUESS)

    def left(self) -> float:
        return EPISODE_SECONDS - (time.monotonic() - self.began)

    def took(self, kind: str, seconds: float) -> None:
        self.usual[kind] = 0.5 * self.usual.get(kind, seconds) + 0.5 * seconds

    def needs(self, kind: str) -> float:
        return 1.5 * self.usual.get(kind, self.usual[HEAVY])

    def closing(self) -> float:
        """Seconds kept for the calls that end the episode."""
        return FINISH_MARGIN + 3 * self.needs('store')

    def affords(self, *kinds: str) -> bool:
        return self.left() > self.closing() + sum(self.needs(k) for k in kinds)

    def may_explore(self) -> bool:
        return self.left() > EXPLORE_MARGIN and self.affords(LIGHT, 'store')

    def try_limit(self, tries: int, ceiling: int) -> int:
        """The timeout of one try, such that all the client's tries end before the closing calls."""
        return max(5, min(ceiling, int((self.left() - self.closing()) / max(1, tries))))


class Wire:
    """Model calls: spending limits, a limited heavy model handing over to its kin, and no call the clock cannot cover."""

    def __init__(self, clock: Clock, trace: list[dict[str, Any]]) -> None:
        self.clock = clock
        self.trace = trace
        self.spent = 0.0
        self.calls = 0
        self.waited = 0.0
        self.no_credit = False
        self.resting: dict[str, float] = {}

    def judges(self, limit: float = SPEND_SOFT) -> list[str]:
        if self.spent <= limit and self.spent + HEAVY_ROOM <= SPEND_HARD:
            return [HEAVY, LIGHT]
        return [LIGHT] if self.spent <= SPEND_HARD else []

    def line_up(self, models: list[str]) -> list[tuple[str, bool]]:
        """Each model to try, with whether it stands in for the heavy one; the heavy one drops out when time is short."""
        order: list[tuple[str, bool]] = []
        for model in models:
            if model != HEAVY:
                order.append((model, False))
            elif self.clock.affords(HEAVY):
                order += [(kin, kin != HEAVY) for kin in HEAVY_KIN]
            elif LIGHT not in models:
                order.append((LIGHT, False))
        return order

    def reply(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None, deep: bool = False) -> dict[str, Any] | None:
        for model, stand_in in self.line_up(models):
            if self.no_credit or self.calls >= ALL_CALLS or not self.clock.affords(model if model in Clock.FIRST_GUESS else HEAVY):
                return None
            if time.monotonic() < self.resting.get(model, 0):
                continue
            message = self.with_backoff(model, messages, max_tokens, tools, 'low' if stand_in and not deep else None)
            if message is not None:
                return message
        return None

    def with_backoff(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None, effort: str | None) -> dict[str, Any] | None:
        self.calls += 1
        for pause in BACKOFF + (None,):
            message, status = self.once(model, messages, max_tokens, tools, effort)
            if message is not None:
                return message
            if status == 402:
                self.no_credit = True
                return None
            if status == 429 and model in HEAVY_KIN:
                self.resting[model] = time.monotonic() + BUSY_SECONDS
                self.calls -= 1
                return None
            passing = status is None or status in (408, 425, 429) or (isinstance(status, int) and status >= 500)
            if pause is None or not passing or self.waited + pause > BACKOFF_TOTAL or not self.clock.affords(model if model in Clock.FIRST_GUESS else HEAVY, 'store') or self.clock.left() < pause + self.clock.closing() + self.clock.needs(LIGHT):
                return None
            self.waited += pause
            time.sleep(pause * (0.8 + 0.4 * random.random()))
        return None

    def once(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None, effort: str | None) -> tuple[dict[str, Any] | None, Any]:
        sent = messages
        if model.startswith('anthropic/') and len(messages) > 1:
            sent = [dict(m) for m in messages]
            for index in {0, len(sent) - 1}:
                content = sent[index].get('content')
                if isinstance(content, str) and content:
                    sent[index]['content'] = [{'type': 'text', 'text': content, 'cache_control': {'type': 'ephemeral'}}]
        body: dict[str, Any] = {'model': model, 'messages': sent, 'max_tokens': max_tokens, 'temperature': 0}
        if tools:
            body.update(tools=tools, tool_choice='auto')
        if effort:
            body['reasoning'] = {'effort': effort}
        _llm.timeout = self.clock.try_limit(_llm.max_retries, 50)
        began = time.monotonic()
        result = _llm.post_verbose('/inference/chat/completions', body)
        self.clock.took(model if model in Clock.FIRST_GUESS else HEAVY, time.monotonic() - began)
        data = result.data
        if isinstance(data, dict) and data.get('choices'):
            cost = (data.get('usage') or {}).get('cost')
            self.spent += float(cost) if isinstance(cost, (int, float)) else 0.0
            message = data['choices'][0].get('message') or {}
            if message.get('content') or message.get('tool_calls'):
                return message, 200
            self.trace.append({'kind': 'llm_fail', 'model': model, 'status': 0})
            return None, 0
        error = result.error or {}
        status = error.get('status')
        if isinstance(data, dict) and isinstance(data.get('error'), dict):
            status = whole_number(data['error'].get('code'))
            error = {'body': data['error'].get('message')}
        self.trace.append({'kind': 'llm_fail', 'model': model, 'status': status, 'body': str(error.get('body') or '')[:200]})
        return None, status

    def ask_json(self, prompt: str, models: list[str], max_tokens: int = 1800, deep: bool = False) -> dict[str, Any] | None:
        for model in models:
            message = self.reply([{'role': 'user', 'content': prompt}], [model], max_tokens, deep=deep)
            found = json_object((message or {}).get('content'))
            if found is not None:
                return found
            if self.no_credit or self.calls >= ALL_CALLS:
                return None
        return None


class Reply:
    """One store answer: the normal form, the raw form and any sentences a wrapper carried."""

    def __init__(self, raw: Any, finished: bool) -> None:
        self.raw = raw
        self.finished = finished
        self.remarks: list[str] = []
        self.body = normal_form(raw, self.remarks)

    def failed(self) -> bool:
        return isinstance(self.body, dict) and any(re.search(r'error|problem|trouble', str(k), re.I) and v for k, v in self.body.items())


class Store:
    """Tool calls: the step count, the store's own refusals of a call, and arguments fitted to each tool."""

    def __init__(self, binding: dict[str, Any], view: dict[str, Any], clock: Clock, trace: list[dict[str, Any]]) -> None:
        self.binding = binding
        self.clock = clock
        self.trace = trace
        tools = [t for t in view.get('tools') or [] if isinstance(t, dict) and isinstance(t.get('function'), dict) and isinstance(t['function'].get('name'), str)]
        self.tools = tools
        self.schemas = {t['function']['name']: t['function'].get('parameters') if isinstance(t['function'].get('parameters'), dict) else {'type': 'object', 'properties': {}} for t in tools}
        steps = view.get('max_steps')
        steps = int(steps) if isinstance(steps, (int, float)) and not isinstance(steps, bool) else 0
        self.max_steps = steps if steps > 0 else 30
        self.step = 0
        self.finished = bool(view.get('finished'))
        self.broke = False

    def left(self) -> int:
        return self.max_steps - self.step

    def fitted(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        faults: list[str] = []
        fixed = conform(args, self.schemas.get(name), 'args', faults)
        return None if faults else fixed

    def call(self, name: str, args: dict[str, Any]) -> Reply | None:
        payload = {**self.binding, 'step': self.step + 1, 'action': {'name': name, 'args': args}}
        began = time.monotonic()
        _env.timeout = self.clock.try_limit(_env.max_retries, 60)
        for _ in range(4):
            result = _env.post_verbose('/environment/call', payload)
            if isinstance(result.data, dict):
                data = result.data
                self.clock.took('store', time.monotonic() - began)
                self.step = data['step'] if isinstance(data.get('step'), int) else self.step + 1
                self.finished = self.finished or bool(data.get('finished'))
                return Reply(data.get('observation'), bool(data.get('finished')))
            error = result.error or {}
            status, body = error.get('status'), str(error.get('body') or '')
            self.trace.append({'kind': 'env_fail', 'status': status, 'body': body[:200]})
            if status == 402:
                self.broke = True
            if status != 409:
                if status in (400, 401, 402, 403, 404, 422):
                    return None
                continue
            low = body.lower()
            expected = re.search(r'expected (\d+)', body)
            if re.search(r'finish|quarantin|finaliz', low):
                self.finished = True
                return None
            if expected:
                self.step = int(expected.group(1)) - 1
            elif 'reuse' in low or 'different action' in low:
                self.step += 1
            else:
                return None
            payload['step'] = self.step + 1
        return None


class Item:
    """One requested product: what the request settles, what the shopper said, and what they gave up."""

    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]) -> None:
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []
        self.asked: list[str] = []
        self.spent = 0
        self.firm = 0

    def live(self) -> list[str]:
        return [m for m in self.musts + self.told if m not in self.dropped]

    def settles(self, key: str) -> bool:
        return any(names_property(m.partition(':')[0], key) for m in self.musts)

    def nearly_settles(self, key: str) -> bool:
        """A key whose words are all in a stated requirement's property ("color" beside a stated "color family")."""
        words = word_set(key)
        return bool(words) and any(words <= word_set(m.partition(':')[0]) for m in self.musts)


class Book:
    """Every listing the episode has seen, in one normal form, with what can be read from them without a model."""

    def __init__(self, trace: list[dict[str, Any]]) -> None:
        self.trace = trace
        self.rows: dict[str, dict[str, Any]] = {}
        self.full: dict[str, dict[str, Any]] = {}
        self.order: list[str] = []
        self.opened: set[str] = set()
        self.unavailable: set[str] = set()
        self.gone: dict[str, int] = {}
        self.id_key: str | None = None
        self.price_key: str | None = None
        self.stock_key: str | None = None
        self.shelf_key: str | None = None
        self.shelf_mark = -1
        self.enums: set[str] = set()

    def listing_id(self, node: Any, in_list: bool = False) -> str | None:
        """The id of a listing record, or None. The id key is learned from the first clear record."""
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if self.id_key and looks_like_id(node.get(self.id_key), True):
            return node[self.id_key].strip()
        named = [k for k, v in node.items() if re.search(r'(?:^| )(?:id|ref|sku|asin|code)(?: |$)', spaced(k)) and looks_like_id(v, True) and re.search(r'\d', v)]
        shaped = [k for k, v in node.items() if looks_like_id(v)]
        chosen = named or (shaped if in_list or any(isinstance(v, dict) for v in node.values()) else [])
        if not chosen:
            return None
        if self.id_key is None:
            self.id_key = chosen[0]
        return node[chosen[0]].strip()

    def records_in(self, body: Any) -> list[tuple[str, dict[str, Any]]]:
        """Listing records of a result: the result itself, or the ones in its lists."""
        found: list[tuple[str, dict[str, Any]]] = []
        own = self.listing_id(body) if isinstance(body, dict) else None
        if own:
            return [(own, body)]
        stack = [(body, 0)]
        while stack:
            node, depth = stack.pop(0)
            if depth > 3:
                continue
            if isinstance(node, list):
                for entry in node:
                    ident = self.listing_id(entry, True)
                    if ident:
                        found.append((ident, entry))
                    elif isinstance(entry, (dict, list)):
                        stack.append((entry, depth + 1))
            elif isinstance(node, dict):
                stack += [(v, depth + 1) for v in node.values() if isinstance(v, (dict, list))]
        return found

    def absorb(self, body: Any) -> list[str]:
        ids = []
        for ident, record in self.records_in(body):
            ids.append(ident)
            if ident not in self.rows:
                self.order.append(ident)
                self.rows[ident] = {}
            row = self.rows[ident]
            if any(row.get(k) is True and v is False for k, v in record.items()):
                self.unavailable.add(ident)
            elif any(row.get(k) is False and v is True for k, v in record.items()):
                self.unavailable.discard(ident)
            row.update(record)
            if facts_block(record):
                self.full[ident] = dict(row)
            elif ident in self.full:
                self.full[ident].update(record)
            self.learn_row_keys(record)
        return ids

    def learn_row_keys(self, record: dict[str, Any]) -> None:
        if self.price_key is None:
            priced = [k for k, v in record.items() if re.search(PRICE_KEYS, spaced(k)) and amount(v) is not None]
            self.price_key = min(priced, key=len) if priced else None
        if self.stock_key is None:
            flags = [k for k, v in record.items() if re.search(STOCK_KEYS, spaced(k)) and (isinstance(v, bool) or isinstance(v, int))]
            self.stock_key = flags[0] if flags else None

    def record(self, ident: str) -> dict[str, Any]:
        return self.full.get(ident) or self.rows.get(ident) or {}

    def facts(self, ident: str) -> dict[str, Any]:
        return facts_block(self.full.get(ident) or {}) or {}

    def text_of(self, ident: str) -> str:
        """All the words a listing shows, lower case."""
        leaves: list[str] = []
        stack: list[Any] = [self.record(ident)]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                stack += list(node.values())
            elif isinstance(node, list):
                stack += node
            elif isinstance(node, str):
                leaves.append(node)
        return re.sub(r'[\s_]+', ' ', ' '.join(leaves).lower())

    def price(self, ident: str) -> float | None:
        return amount(self.rows.get(ident, {}).get(self.price_key)) if self.price_key else None

    def sold_out(self, ident: str) -> bool:
        return ident in self.unavailable or ident in self.gone

    def buyable(self, ident: str) -> bool:
        if self.sold_out(ident):
            return False
        stock = self.rows.get(ident, {}).get(self.stock_key) if self.stock_key else None
        return not (stock is False or (isinstance(stock, (int, float)) and not isinstance(stock, bool) and stock <= 0))

    def shelf_field(self) -> str | None:
        """The key that holds a listing's category: by name when the store says so, else by how its values repeat."""
        if self.shelf_mark == len(self.rows):
            return self.shelf_key
        self.shelf_mark = len(self.rows)
        counts: dict[str, list[str]] = {}
        for row in self.rows.values():
            for key, value in row.items():
                if isinstance(value, str) and 0 < len(value) <= 120 and not re.search(r'\d{3}', value):
                    counts.setdefault(key, []).append(value)
        common = {k: v for k, v in counts.items() if len(v) >= 0.8 * len(self.rows)}
        named = [k for k in common if re.search(SHELF_KEYS, spaced(k))]
        if named:
            self.shelf_key = min(named, key=len)
            return self.shelf_key

        def rank(key: str) -> tuple[float, int, float]:
            values = common[key]
            listed = sum(v.strip().lower() in self.enums for v in values) / len(values)
            return (-listed, len(set(values)), sum(map(len, values)) / len(values))
        kept = [k for k in common if len(set(common[k])) <= 40 and not re.search(TITLE_KEYS + '|' + NOTE_KEYS + '|' + SHOP_KEYS, spaced(k))]
        self.shelf_key = min(kept, key=rank) if kept else None
        return self.shelf_key

    def label(self, ident: str) -> str:
        field = self.shelf_field()
        text = str(self.rows.get(ident, {}).get(field) or '') if field else ''
        return re.split(r'\s*[>»]\s*', text.strip())[-1].strip()

    def shop_of(self, ident: str) -> list[str]:
        return [str(v).strip() for k, v in self.rows.get(ident, {}).items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search(SHOP_KEYS, spaced(k))]

    def row_text(self, ident: str, limit: int = 400) -> str:
        return jdump(clipped(self.record(ident), limit))[:3000] if self.record(ident) else '(no record seen)'


class Needs:
    """The request as read by the brief model: items, budget, pick rule and steps, and how listings map to items."""

    def __init__(self, book: Book) -> None:
        self.book = book
        self.items: list[Item] = []
        self.brief: dict[str, Any] = {}
        self.budget: float | None = None
        self.refs: list[tuple[int, str]] = []
        self.ref_values: dict[str, str] = {}
        self.bound: list[tuple[int, int]] = []
        self.process: list[dict[str, Any]] = []
        self.notes: list[str] = []
        self.version = 0
        self.adopted: dict[str, int] = {}

    def adopt(self, label: str) -> None:
        """A category no item's name resembles, carted anyway: the store files that item under this label after all."""
        if label and not self.owner(label):
            free = [n for n in range(1, len(self.items) + 1) if not self.ids_of(n)]
            if free:
                self.adopted[label] = free[0]

    def pick(self) -> Any:
        return self.brief.get('pick')

    def item_of(self, number: Any) -> Item:
        count = whole_number(number) or 1
        return self.items[min(max(1, count), len(self.items)) - 1]

    def now(self) -> list[tuple[int, str]]:
        return [(n, need) for n, item in enumerate(self.items, 1) for need in item.live()]

    def owner(self, label: str) -> int:
        """Which requested item a category label belongs to, 0 for none."""
        if not label:
            return 0
        if label in self.adopted:
            return self.adopted[label]
        plain, roots = spaced(label), word_set(label, 3)
        best, top = 0, 0.0
        for number, item in enumerate(self.items, 1):
            if plain == spaced(item.shelf or ''):
                score = 3.0
            elif plain == spaced(item.name):
                score = 2.0
            else:
                mine = word_set(item.shelf or '', 3) | word_set(item.name, 3)
                score = len(roots & mine) / max(1, len(roots | mine))
            if score > top:
                best, top = number, score
        return best

    def ids_of(self, number: int) -> list[str]:
        return [i for i in self.book.order if self.owner(self.book.label(i)) == number]

    def shelf_name(self, number: int) -> str:
        """The store's own label for an item, the one most of its rows show; else the brief's, else the item's name."""
        labels = [self.book.label(i) for i in self.ids_of(number)]
        item = self.items[number - 1]
        return max(set(labels), key=labels.count) if labels else (item.shelf or item.name)

    def values_of(self, number: int) -> list[str]:
        """Required values that can be searched for: plain values, no limits, exclusions or pointers."""
        out = []
        pointers = {need for _, need in self.refs}
        for owner, need in self.now():
            value = need.partition(':')[2].strip()
            if owner == number and need not in pointers and value and not re.search(LIMIT_WORDS, value, re.I):
                out.append(value)
        return out

    def own_keys(self, ident: str, prop: str, number: int) -> list[str]:
        """Attribute values under keys that are this property: the same words, whatever the spelling, once the item's own
        name is taken off the key ("lamp finish" on a lamp is its finish; "strap finish" is not)."""
        item = self.items[number - 1] if 1 <= number <= len(self.items) else None
        own = (word_set(item.name) | word_set(item.shelf)) if item else set()
        wanted = word_set(prop)
        return [str(v) for k, v in self.book.facts(ident).items() if word_set(k) == wanted or (word_set(k) - own) == wanted]

    def need_state(self, ident: str, need: str, number: int = 0) -> str:
        """What a listing's own record says of one requirement: 'met' under a key that is the property (one spelling
        stating the value is enough), 'broken' when such keys say something else, 'text' when only its words show the
        value, and 'open' when code cannot tell, which includes a key that merely has the property's name in it."""
        prop, _, value = need.partition(':')
        value = value.strip()
        choices = [c.strip() for c in re.split(r'\s+or\s+', value) if c.strip()]
        if not choices or any(re.search(LIMIT_WORDS, c, re.I) or re.search(r'\d', c) for c in choices):
            return 'open'
        wanted = [word_set(c) for c in choices]
        named = self.own_keys(ident, prop, number)
        if named:
            return 'met' if any(w <= word_set(v) for w in wanted for v in named) else 'broken'
        if any(names_property(prop, k) for k in self.book.facts(ident)):
            return 'open'
        shown = word_set(self.book.text_of(ident))
        return 'text' if any(w <= shown for w in wanted) else 'open'

    def standing(self, ident: str, number: int) -> tuple[int, int, int, int]:
        """How a listing stands against an item's requirements: counts of met, shown in its text only, open and broken."""
        marks = [self.need_state(ident, need, number) for owner, need in self.now() if owner == number]
        return marks.count('met'), marks.count('text'), marks.count('open'), marks.count('broken')

    def within_budget(self, ident: str) -> bool:
        price = self.book.price(ident)
        return self.budget is None or price is None or price <= self.budget or price > 10 * self.budget

    def key_menu(self, number: int) -> list[dict[str, Any]]:
        """The item's attribute keys with what its opened listings show, best evidence first; same-meaning keys are one."""
        book = self.book
        records = [i for i in book.order if i in book.full and book.facts(i) and (self.owner(book.label(i)) == number or len(self.items) == 1)]
        slots: dict[str, dict[str, Any]] = {}
        for ident in records:
            for key, value in book.facts(ident).items():
                name = spaced(key)
                text = str(value).strip().lower()
                if name and text:
                    slot = slots.setdefault(name, {'name': name, 'per': {}, 'spellings': set()})
                    slot['per'].setdefault(ident, text)
                    slot['spellings'].add(str(key))
        merged: list[dict[str, Any]] = []
        for slot in sorted(slots.values(), key=lambda s: (-len(s['per']), len(s['name']))):
            values = list(slot['per'].values())
            mean = sum(map(len, values)) / len(values)
            if mean > 40 or sum(' / ' in v for v in values) * 2 >= len(values):
                continue
            twin = next((m for m in merged if same_property(m, slot)), None)
            if twin is None:
                merged.append(slot)
            else:
                for ident, text in slot['per'].items():
                    twin['per'].setdefault(ident, text)
                twin['spellings'] |= slot['spellings'] | {'merged'}
        titles = ' '.join(str(v).lower() for i in records for k, v in book.rows[i].items() if isinstance(v, str) and re.search(TITLE_KEYS, spaced(k)))
        notes = ' '.join(re.sub(r'_', ' ', str(v).lower()) for i in records for k, v in book.rows[i].items() if isinstance(v, str) and re.search(NOTE_KEYS, spaced(k)))
        menu = []
        for slot in merged:
            tally: dict[str, int] = {}
            for text in slot['per'].values():
                tally[text] = tally.get(text, 0) + 1
            shown = [v for v in tally if len(v) >= 3]
            slot.update(n=len(slot['per']), values=list(tally)[:4], read=max(1, len(records)))
            slot['measured'] = sum(bool(re.match(r'^\s*[\d.,/x -]+\s*[a-z]{0,6}\s*$', v)) for v in slot['per'].values()) / slot['n']
            slot['score'] = key_evidence(slot['n'] / slot['read'], len(slot['spellings']), tally, any(has_words(v, titles) for v in shown), has_words(slot['name'], notes), slot['name'], slot['measured'])
            menu.append(slot)
        menu.sort(key=lambda s: (-s['score'], -s['n'], s['name']))
        return menu

    def open_keys(self, number: int) -> list[dict[str, Any]]:
        """Keys worth a question: not settled by the request, not an identifier or logistics."""
        item = self.items[number - 1]
        keys = [s for s in self.key_menu(number) if not item.settles(s['name']) and not re.search(NOT_PROPERTY, s['name'])]
        keys = [s for s in keys if not item.nearly_settles(s['name'])] + [s for s in keys if item.nearly_settles(s['name'])]
        return [s for s in keys if not catch_all(s['name'])] + [s for s in keys if catch_all(s['name'])]

    def ledger(self, facts_line: str = '') -> str:
        lines = []
        many = len(self.items) > 1
        for number, item in enumerate(self.items, 1):
            needs = [m + (f' (the page it points at gives: {self.ref_values[m]})' if m in self.ref_values else '') for m in item.live()]
            head = (f'item {number}: ' if many else 'item: ') + item.name
            head += f' (store label: {item.shelf})' if item.shelf else ''
            text = f"{head}, quantity {item.qty}; requirements: {'; '.join(needs) or 'none beyond the kind of product'}"
            lines.append(text + ('; shopper does not mind: ' + ', '.join(item.free) if item.free else ''))
        brief = self.brief
        if self.budget is not None:
            scope = 'for each item' if brief.get('budget_scope') == 'each' else 'in total'
            lines.append(f"budget: at most {self.budget:g} {brief.get('budget_unit') or ''} {scope}".replace('  ', ' '))
        pick = self.pick()
        if pick == 'cheapest':
            lines.append('pick: the cheapest listing that fits (lowest price)')
        elif pick in ('most', 'least'):
            lines.append(f"pick: the {pick} {brief.get('pick_measure') or ''} among fitting listings".replace('  ', ' '))
        else:
            lines.append('pick: any listing that fits everything')
        if brief.get('in_stock_only', True) is not False:
            lines.append('in stock only')
        lines += [f'wish (tie-breaker only): {w}' for w in as_list(brief.get('wishes'))]
        lines += [f'rule from the request: {r}' for r in as_list(brief.get('rules'))]
        if brief.get('quiet_decline') is True:
            lines.append('if nothing fits: end without naming or suggesting any listing')
        moment = {'choose': 'before choosing', 'order': 'at the order (the desk does it when you check out)', 'end': 'at an ending without a purchase (the desk does it then)'}
        lines += [f"step the request asks for, {moment[p['when']]}: {p['step']}" for p in self.process]
        if facts_line:
            lines.append(facts_line)
        return '\n'.join('- ' + line for line in lines)


def same_property(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Two keys are one property when they share a word and, on the listings carrying both, their values agree."""
    a, b = set(first['name'].split()), set(second['name'].split())
    if not a & b:
        return False

    def agree(x: str, y: str) -> bool:
        small, large = sorted((word_set(x), word_set(y)), key=len)
        return x == y or (bool(small) and small <= large and len(large) <= 2 * len(small) + 2)
    both = [i for i in first['per'] if i in second['per']]
    same = sum(agree(first['per'][i], second['per'][i]) for i in both)
    if not (a <= b or b <= a):
        return len(both) >= 2 and same == len(both)
    return same * 2 > len(both) if both else bool(set(first['per'].values()) & set(second['per'].values()))


def item_pull(two_spellings: int, stated: int, hinted: bool) -> float:
    """Which item to ask about first when messages are fewer than items: a story hint from the planner, then keys the
    store writes under two spellings, then fewer requirements already stated. Recorded episodes give this little to go on
    (112 multi-item tasks: 47.5% against 44.6% for a random item), so this only breaks a tie that must be broken somehow."""
    return 2.0 * hinted + 0.25 * min(two_spellings, 4) - 0.1 * min(stated, 3)


def one_ask(label: str, shelf: str) -> str:
    """The last gate before a planned question goes out: exactly one property of one item, in one plain sentence."""
    first = re.split(r'\s+(?:and|or)\s+|[/,;&?(]', str(label))[0].strip().rstrip('?.:')
    place = ' '.join(str(shelf).split()).rstrip('?.')
    if not first:
        return f'What kind of {place.lower()} would you like?'
    return f'What {first.lower()} would you like for the {place.lower()}?'


class Interviewer:
    """All messages to the shopper. Owes every allowed message before anything commits: see `owed`."""

    def __init__(self, errand: 'Errand') -> None:
        self.e = errand
        self.tool: str | None = None
        self.limit = 0
        self.used = 0
        self.kept = 0
        self.kept_used = 0
        self.stuck = False
        self.entries = 0
        self.record: list[str] = []
        self.record_steps: list[int] = []
        self.hinted: set[int] = set()

    def set_up(self, brief: dict[str, Any]) -> None:
        """Message tool and budget, trusting the brief only as far as the world bears it out: the tool is looked for in
        the schemas when the brief misses it, and the budget is the larger of what the brief and the request's own words
        say, since a message over the budget is only refused while one left unused can lose the episode."""
        e = self.e
        self.tool = e.roles.get('ask') or self.find_tool()
        if self.tool:
            e.roles['ask'] = self.tool
        stated = whole_number(brief.get('messages'))
        written = message_count(e.request) or 0
        self.limit = max(written, stated or 0) or (DEFAULT_MESSAGES if self.tool else 0)
        talk = [p['messages'] for p in e.needs.process if p['when'] == 'order' and p['messages'] > 0 and not self.is_notice(p['step'])]
        self.kept = min(max(talk), max(0, self.limit - 1)) if talk and self.tool else 0

    def find_tool(self) -> str | None:
        best, top = None, 0
        for tool in self.e.store.tools:
            fn = tool['function']
            name, about = spaced(fn['name']), str(fn.get('description') or '').lower()
            target = lone_target(self.e.store.schemas[fn['name']])
            if not target or fn['name'] in self.e.roles.values():
                continue
            score = 2 * bool(re.search(r'\b(?:ask|message|reply|contact|talk|chat)\b', name)) + 2 * bool(re.search(r'\b(?:shopper|buyer|user|customer|client)\b', name))
            score += bool(re.search(r'\b(?:message|ask|question|reply)\b', about)) + bool(re.search(r'\b(?:shopper|buyer|user|customer|client)\b', about))
            score += bool(re.search(r'msg|message|text|question|content|body|reply', target, re.I))
            if score > top and score >= 3:
                best, top = fn['name'], score
        return best

    @staticmethod
    def is_notice(step: str) -> bool:
        """A step that only says nothing was found, which the ending itself does."""
        return bool(re.search(r'\b(?:nothing|none|no\s+(?:match|fit|result|listing|product)\w*|not found|unavailable|substitute|alternative)', step, re.I))

    def free(self) -> int:
        return self.limit - self.used - max(0, self.kept - self.kept_used)

    def owed(self) -> bool:
        """The guarantee: true while a message could still go out, so callers hold every commit until it has."""
        e = self.e
        return bool(self.tool) and not self.stuck and self.free() > 0 and not e.store.finished and e.store.left() >= e.tail() + 2 and e.clock.affords(LIGHT, 'store')

    def ready(self) -> bool:
        e = self.e
        if not self.owed() or self.entries:
            return False
        opened = sum(1 for n in range(1, len(e.needs.items) + 1) if len([i for i in e.needs.ids_of(n) if i in e.book.full]) >= 2)
        if opened >= len(e.needs.items) and len(e.book.full) >= 3:
            return True
        return e.store.step >= max(8, e.store.max_steps // 4)

    def lanes(self) -> dict[int, list[str]]:
        """For each item the properties to ask, best first: the planner's ranked picks, which see the keys in evidence
        order, then the remaining keys in that order. The planner is asked for more picks than there are messages, so a
        pick code removes is replaced by its own next choice."""
        e, needs = self.e, self.e.needs
        menus = {n: needs.open_keys(n) for n in range(1, len(needs.items) + 1)}
        listing = []
        counter = 0
        for number, menu in menus.items():
            listing.append(f'category {needs.shelf_name(number)!r}:')
            for slot in menu[:MENU_KEYS]:
                counter += 1
                listing.append(f"  {counter}. {slot['name']} (on {slot['n']}): " + ' | '.join(v[:40] for v in slot['values']) + (' [several features in one: ask last]' if catch_all(slot['name']) else ''))
        settled = [f"item {n}: {it.name} - settled by the request: {'; '.join(it.musts) or 'nothing'}" for n, it in enumerate(needs.items, 1)]
        prompt = PLAN.replace('COUNT', str(max(PLAN_PICKS, self.free()))) + '\n\nREQUEST:\n' + e.request + '\n\nITEMS:\n' + '\n'.join(settled)
        first = [str(x) for x in as_list(needs.brief.get('ask_first')) if str(x).strip()]
        if first:
            prompt += '\n\nASK FIRST (the request asks to be asked about these):\n' + '\n'.join(first)
        prompt += '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(listing) if counter else '(no full listing records yet)')
        plan = e.wire.ask_json(prompt, [HEAVY, LIGHT], 3500, deep=True) or {}
        picks: dict[int, list[tuple[str, bool]]] = {n: [] for n in menus}
        for pick in as_list(plan.get('ask')):
            if isinstance(pick, dict):
                number = min(max(1, whole_number(pick.get('item')) or 1), len(needs.items))
                label = re.split(r'\s+(?:and|or)\s+|[/,;(]', str(pick.get('label') or ''))[0].strip().rstrip('?.')
                if label and len(label.split()) <= 5 and not needs.items[number - 1].settles(label) and not re.search(NOT_PROPERTY, spaced(label)):
                    told_by_story = bool(str(pick.get('hint') or '').strip())
                    picks[number].append((self.menu_spelling(label, menus[number]), told_by_story))
                    if told_by_story:
                        self.hinted.add(number)
        lanes = {}
        for number, menu in menus.items():
            lane = [p for p, _ in picks[number]] + [s['name'] for s in menu]
            seen: set[str] = set()
            lane = [x for x in lane if not (spaced(x) in seen or seen.add(spaced(x)))]
            # A story about who it is for pulls the planner to a bare gender question. In recorded episodes that was the
            # weakest common single question (about 16% firm against 22-28% for the other common ones), so it waits.
            if lane and spaced(lane[0]) == 'gender' and len(lane) > 2:
                lane.insert(2, lane.pop(0))
            item = needs.items[number - 1]
            lanes[number] = [x for x in lane if not item.nearly_settles(x)] + [x for x in lane if item.nearly_settles(x)]
        e.trace.append({'kind': 'plan', 'ask': clipped(plan.get('ask'), 120), 'lanes': {n: v[:6] for n, v in lanes.items()}})
        return lanes

    @staticmethod
    def menu_spelling(label: str, menu: list[dict[str, Any]]) -> str:
        return next((s['name'] for s in menu if word_set(s['name']) == word_set(label)), label)

    def item_rank(self, lanes: dict[int, list[str]]) -> list[int]:
        """Items in asking order. Every item gets its first question before any gets a second; with fewer messages than
        items, the store evidence decides which go first. An item already answered firmly waits behind the others."""
        needs = self.e.needs

        def pull(number: int) -> float:
            twins = sum(len(s['spellings']) >= 2 for s in needs.open_keys(number))
            return item_pull(twins, len(needs.items[number - 1].musts), number in self.hinted)
        waiting = [n for n in lanes if any(spaced(x) not in {spaced(a) for a in needs.items[n - 1].asked} for x in lanes[n])]
        return sorted(waiting, key=lambda n: (needs.items[n - 1].firm > 0, needs.items[n - 1].spent, -pull(n), n))

    def spoken(self, label: str) -> str:
        """A key as it is said aloud: a store's short prefix comes off (a word of one or two letters that starts several
        of the episode's keys). Keys and matching keep it; only the question's words change."""
        words = spaced(label).split()
        if len(words) < 2 or len(words[0]) > 2 or not words[0].isalpha():
            return label
        keys = {spaced(k) for i in self.e.book.full for k in self.e.book.facts(i)}
        return ' '.join(words[1:]) if sum(k.startswith(words[0] + ' ') for k in keys) >= 3 else label

    def fallback_label(self, number: int) -> str:
        """A property for a message that would otherwise go unused: any key of the item's kind not asked yet, else the
        plain question on the kind of product."""
        item = self.e.needs.items[number - 1]
        asked = {spaced(a) for a in item.asked}
        for slot in self.e.needs.key_menu(number):
            if slot['name'] not in asked and not re.search(NOT_PROPERTY, slot['name']):
                return slot['name']
        return '' if '' not in item.asked else '-'

    def run(self) -> str:
        """Open what the questions need, then spend every free message, one property of one item at a time."""
        e, needs = self.e, self.e.needs
        self.entries += 1
        if not self.owed():
            return ''
        e.opener.widen()
        lanes = self.lanes()
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        found_on: list[int] = []
        retry: tuple[int, str] | None = None
        while self.owed():
            if retry:
                number, label = retry
                retry = None
            else:
                order = self.item_rank(lanes)
                if order:
                    number = order[0]
                    asked = {spaced(a) for a in needs.items[number - 1].asked}
                    label = next(x for x in lanes[number] if spaced(x) not in asked)
                else:
                    number = min(range(1, len(needs.items) + 1), key=lambda n: needs.items[n - 1].spent)
                    label = self.fallback_label(number)
                    if label == '-':
                        break
            item = needs.items[number - 1]
            item.asked.append(label)
            item.spent += 1
            text = one_ask(self.spoken(label), needs.shelf_name(number))
            answer = self.send(text)
            if answer is None:
                break
            sent.append((number, text, answer))
            heard = self.hear([(number, text, answer)], [LIGHT, HEAVY])
            if e.wire.judges()[:1] == [HEAVY] and any(self.doubtful(entry) for entry in as_list(heard.get('needs'))):
                heard = self.hear([(number, text, answer)], [HEAVY]) or heard
            before = [len(it.told) for it in needs.items]
            self.take_heard(heard, told)
            fresh = [n for n, it in enumerate(needs.items, 1) if len(it.told) > before[n - 1]]
            for n in fresh:
                needs.items[n - 1].firm += 1
                found_on.append(n)
            if not fresh and 1 in [whole_number(x) for x in as_list(heard.get('crowded'))]:
                head = label.split()[-1]
                if len(label.split()) >= 2 and spaced(head) not in {spaced(a) for a in item.asked}:
                    retry = (number, head)
        self.stuck = self.stuck or self.free() > 0
        hunted = e.hunter.after_answers(list(dict.fromkeys(found_on))) if found_on else ''
        if told:
            needs.version += 1
        summary = "The desk's interviewer asked the shopper (each question used one step):\n" + '\n'.join(f'Q: {q}\nA: {a}' for _, q, a in sent)
        summary += '\nNew firm requirements from these answers: ' + ('; '.join(told) if told else 'none')
        summary += '. Apply them to every pick (search again with the new values where needed).'
        pointed = [need for _, need in needs.refs if any(need in t for t in told)]
        if pointed:
            summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.'
        return summary + ('\n' + hunted if hunted else '') + e.drain_notes()

    def send(self, text: str, floor: int = 0) -> str | None:
        """One message; the words of the reply, or None when it could not go out."""
        e = self.e
        if e.store.finished or e.store.left() <= floor or self.used >= self.limit or not self.tool:
            return None
        target = self.tool
        schema = e.store.schemas.get(target) or {}
        field = lone_target(schema) or next(iter(text_args(schema)), None)
        if not field:
            self.stuck = True
            return None
        args = {field: text}
        for key in schema.get('required') or []:
            if key not in args:
                options = ((schema.get('properties') or {}).get(key) or {}).get('enum')
                if not options:
                    self.stuck = True
                    return None
                args[key] = options[0]
        fitted = e.store.fitted(target, args)
        reply = e.store.call(target, fitted) if fitted is not None else None
        if reply is None:
            self.stuck = True
            return None
        self.used += 1
        e.heed(reply.remarks, target)
        words = self.words_of(reply.body)
        self.record.append(f'step {e.store.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {e.cart.mark()})')
        self.record_steps.append(e.store.step)
        e.trace.append({'kind': 'ask', 'q': text, 'a': words[:300]})
        if reply.failed() or re.search(r'\b(?:limit|no more messages|too many messages|exceeded|out of messages)\b', words, re.I):
            self.limit = self.used
        return words

    @staticmethod
    def words_of(body: Any) -> str:
        if isinstance(body, str):
            return body
        found: list[str] = []
        stack = [body]
        while stack:
            node = stack.pop()
            if isinstance(node, str) and len(node.split()) >= 2:
                found.append(node)
            elif isinstance(node, dict):
                stack += list(node.values())
            elif isinstance(node, list):
                stack += node
        return ' '.join(found)[:800] if found else jdump(body)[:400]

    def doubtful(self, entry: Any) -> bool:
        """A stated answer deserves a second reading when it points at another listing, or when its value has words no
        opened listing shows for that property, which is how words around a value creep into it."""
        if not isinstance(entry, dict):
            return False
        prop, _, value = str(entry.get('need') or '').partition(':')
        words = set(re.findall(r'[a-z0-9]+', value.lower()))
        if entry.get('points') is True:
            return True
        if len(words) < 2:
            return False
        shown: set[str] = set()
        for ident in self.e.book.full:
            for key, seen in self.e.book.facts(ident).items():
                if names_property(prop, key) or names_property(key, prop):
                    shown |= set(re.findall(r'[a-z0-9]+', str(seen).lower()))
        return bool(words - shown)

    def hear(self, sent: list[tuple[int, str, str]], models: list[str]) -> dict[str, Any]:
        needs = self.e.needs
        prompt = HEAR + '\n\nITEMS:\n' + '\n'.join(f'item {n}: {it.name}' for n, it in enumerate(needs.items, 1))
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join(f'{i}. (about item {n}) Q: {q}\nA: {a}' for i, (n, q, a) in enumerate(sent, 1))
        return self.e.wire.ask_json(prompt, models, 2500) or {}

    def take_heard(self, heard: dict[str, Any], told: list[str]) -> None:
        needs = self.e.needs
        many = len(needs.items) > 1
        for entry in as_list(heard.get('needs')):
            need = str(entry.get('need') or '').strip() if isinstance(entry, dict) else ''
            if not need:
                continue
            number = min(max(1, whole_number(entry.get('item')) or 1), len(needs.items))
            item = needs.items[number - 1]
            if need not in item.told:
                item.told.append(need)
                told.append(f'item {number}: {need}' if many else need)
            if entry.get('points') is True and (number, need) not in needs.refs:
                needs.refs.append((number, need))
        for entry in as_list(heard.get('drop')):
            if isinstance(entry, dict) and str(entry.get('property') or '').strip():
                item = needs.item_of(entry.get('item'))
                gone = spaced(str(entry['property']).split(':')[0])
                hits = [m for m in item.live() if spaced(m.split(':')[0]) == gone]
                item.dropped.extend(hits)
                told.extend(f'dropped {m}' for m in hits)
        for entry in as_list(heard.get('free')):
            if isinstance(entry, dict) and entry.get('property'):
                needs.item_of(entry.get('item')).free.append(str(entry['property']))
        limit = amount(heard.get('budget'))
        if limit:
            needs.budget = limit


class Opener:
    """Reading listings' attributes: single pages or side by side, whichever the store shows them in."""

    def __init__(self, errand: 'Errand') -> None:
        self.e = errand
        self.facts_from: dict[str, bool] = {}
        self.compare_ok = 0
        self.compare_bad = 0

    def single_tool(self) -> str | None:
        e = self.e
        known = next((t for t, ok in self.facts_from.items() if ok and lone_target(e.store.schemas.get(t) or {})), None)
        if known:
            return known
        tool = e.roles.get('open')
        return tool if tool and self.facts_from.get(tool) is not False and lone_target(e.store.schemas.get(tool) or {}) else None

    def compare_field(self) -> str | None:
        schema = self.e.store.schemas.get(self.e.roles.get('compare') or '') or {}
        props = schema.get('properties') or {}
        field = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in type_names(v)), None)
        return field if field and not any(k != field for k in schema.get('required') or []) else None

    def compare_room(self) -> int:
        field = self.compare_field()
        schema = self.e.store.schemas.get(self.e.roles.get('compare') or '') or {}
        spec = (schema.get('properties') or {}).get(field or '') or {}
        if isinstance(spec.get('maxItems'), int):
            return spec['maxItems']
        return max(2, min(self.compare_bad - 1, max(self.compare_ok, 2))) if self.compare_bad else max(4, self.compare_ok)

    def compare_useful(self) -> bool:
        """Side by side saves steps only where it shows attributes. One try shows whether it does, and that one step is
        repaid the first time it opens several listings."""
        e = self.e
        seen = self.facts_from.get(e.roles.get('compare') or '')
        if not self.compare_field():
            return False
        return seen if seen is not None else True

    def learn(self, name: str, reply: Reply) -> None:
        records = self.e.book.records_in(reply.body)
        if records:
            self.facts_from[name] = self.facts_from.get(name, False) or any(facts_block(r) for _, r in records)
            if name in self.e.page_tools():
                self.e.book.opened.update(i for i, _ in records)
        roles = self.e.roles
        if self.e.book.listing_id(reply.body) and lone_target(self.e.store.schemas.get(name) or {}) and name not in (roles.get('cart'), roles.get('buy'), roles.get('refuse'), roles.get('ask'), roles.get('search')):
            if name == roles.get('open') and not self.facts_from[name]:
                other = next((t for t in self.e.store.schemas if t != name and lone_target(self.e.store.schemas[t]) and self.facts_from.get(t) is not False and t not in roles.values()), None)
                if other:
                    roles['open'] = other
            if not self.facts_from[name] and name != roles.get('open') and not roles.get('offer'):
                roles['offer'] = name

    def read(self, ids: list[str], keep: int) -> list[str]:
        """Open these listings, several per step where the store allows. Returns what the views showed, for the model."""
        e = self.e
        wanted = [i for i in dict.fromkeys(ids) if i in e.book.rows and i not in e.book.full and not e.book.sold_out(i)]
        shown: list[str] = []
        room = self.compare_room()
        while wanted and e.store.left() > keep and not e.store.finished and e.clock.affords('store'):
            if self.compare_useful() and len(wanted) >= 2:
                batch, wanted = wanted[:room], wanted[room:]
                reply = e.use(e.roles['compare'], {self.compare_field(): batch}, keep)
                if reply is None:
                    break
                if not e.book.records_in(reply.body) and len(batch) > 2:
                    self.compare_bad = len(batch)
                    room = self.compare_room()
                    wanted = batch + wanted
                    continue
                self.compare_ok = max(self.compare_ok, len(batch))
                shown.append(e.show(reply)[:2500])
                if self.facts_from.get(e.roles['compare']) is False:
                    wanted = batch + wanted
                continue
            tool = self.single_tool()
            if not tool:
                break
            ident = wanted.pop(0)
            reply = e.use(tool, {lone_target(e.store.schemas[tool]): ident}, keep)
            if reply is None:
                break
            shown.append(f'{ident}: ' + e.show(reply)[:1500])
            if self.facts_from.get(tool) is False and tool == e.roles.get('open') and not self.single_tool():
                break
        if shown:
            e.trace.append({'kind': 'opened', 'n': len(shown)})
        return shown

    def widen(self) -> None:
        """Before questions: read a few listings of each item, those closest to its requirements first."""
        e, needs, book = self.e, self.e.needs, self.e.book
        if not e.clock.may_explore():
            return
        each = 6 if len(needs.items) <= 2 else 4
        lanes = []
        for number in range(1, len(needs.items) + 1):
            ids = needs.ids_of(number) or (book.order if len(needs.items) == 1 else [])
            done = sum(i in book.full for i in ids)
            words = word_set(' '.join(needs.values_of(number)), 3)
            pool = [i for i in ids if i not in book.full and book.buyable(i)]
            pool.sort(key=lambda i: (-len(words & word_set(book.text_of(i), 3)), book.price(i) or 0.0 if needs.pick() == 'cheapest' else 0.0))
            lanes.append(pool[:max(0, each - done)])
        turns = [lane[k] for k in range(each) for lane in lanes if k < len(lane)]
        self.read(turns[:10], max(e.tail() + 12, min(22, e.store.max_steps // 2)))


class Hunter:
    """Store searches run by code, with no model call: on the required values, on through unread pages, opening the rows
    whose own text shows those values. Most declines are wrong and many follow a store left half read; the same moves
    bring up cheaper rows for a cheapest pick and the rows a fresh answer calls for."""

    def __init__(self, errand: 'Errand') -> None:
        self.e = errand
        self.tried: set[str] = set()
        self.used_args: list[dict[str, Any]] = []
        self.unread: dict[str, dict[str, Any]] = {}
        self.shapes: dict[str, dict[str, Any] | None] = {}
        self.browse: dict[str, Any] | None = None
        self.browse_misses = 0
        self.hunts = 0
        self.swept: set[str] = set()
        self.pages_read = 0
        self.value_rows: set[str] = set()

    # how a result is paged

    def shape(self, name: str, body: Any) -> dict[str, Any] | None:
        """A search result's parts, read by code from the result and the tool's schema; the model is asked only when the
        result plainly pages on and the code cannot tell how."""
        if name in self.shapes:
            return self.shapes[name]
        if not isinstance(body, dict):
            return None
        lists = {k: v for k, v in body.items() if isinstance(v, list) and v and all(isinstance(r, dict) for r in v)}
        if not lists:
            return None
        rows = max(lists, key=lambda k: len(lists[k]))
        count = next((k for k, v in body.items() if isinstance(v, int) and not isinstance(v, bool) and re.search(COUNT_KEYS, spaced(k)) and not re.search(NEXT_KEYS, spaced(k))), None)
        following = next((k for k in body if k != rows and k != count and re.search(NEXT_KEYS, spaced(k))), None)
        props = (self.e.store.schemas.get(name) or {}).get('properties') or {}
        arg = next((k for k in props if re.search(PAGE_ARGS, spaced(k)) and not re.search(r'size|limit|per|count', spaced(k))), None)
        if following and not arg and self.e.clock.affords(LIGHT):
            sample = {k: v[:2] if k in lists else v for k, v in body.items()}
            prompt = SHAPE + '\n\nTOOL: ' + name + ' | parameters: ' + jdump(props) + '\n\nRESULT (first rows only):\n' + jdump(clipped(sample, 120))[:4000]
            got = self.e.wire.ask_json(prompt, [LIGHT], 800) or {}
            arg = got.get('next_arg') if got.get('next_arg') in props else None
        self.shapes[name] = {'rows': rows, 'count': count, 'next': following, 'arg': arg}
        return self.shapes[name]

    def note_result(self, name: str, args: dict[str, Any], body: Any) -> None:
        """Remember a search that has more pages, so a later hunt can read on from where it stopped."""
        shape = self.shape(name, body)
        if not shape or not shape['next'] or not shape['arg'] or not isinstance(body, dict):
            return
        base = {k: v for k, v in args.items() if k != shape['arg']}
        mark = name + jdump(base)
        following = body.get(shape['next'])
        if following in (None, '', False):
            self.unread.pop(mark, None)
            return
        ids = [i for i, _ in self.e.book.records_in(body)]
        entry = self.unread.setdefault(mark, {'tool': name, 'args': base, 'arg': shape['arg'], 'ids': []})
        entry['next'] = following
        entry['ids'] += ids

    def read_on(self, number: int, stop_at: int) -> list[str]:
        """The next page of the store's search that has shown the most rows for this item."""
        e = self.e
        words = word_set(' '.join(e.needs.values_of(number)), 3)

        def promise(entry: dict[str, Any]) -> tuple[int, int]:
            kind = [i for i in entry['ids'] if e.needs.owner(e.book.label(i)) == number]
            return (sum(bool(words & word_set(e.book.text_of(i), 3)) for i in kind), len(kind))
        ranked = sorted((x for x in self.unread.values() if promise(x)[1]), key=promise, reverse=True)
        if not ranked or e.store.step >= stop_at:
            return []
        entry = ranked[0]
        reply = e.use(entry['tool'], {**entry['args'], entry['arg']: entry['next']}, e.tail())
        if reply is None:
            self.unread = {k: v for k, v in self.unread.items() if v is not entry}
            return []
        self.pages_read += 1
        self.note_result(entry['tool'], entry['args'], reply.body)
        return [i for i, _ in e.book.records_in(reply.body)]

    def auto_pages(self, name: str, args: dict[str, Any], body: Any) -> str:
        """For a pick over all fitting listings, read a short result's next pages at once."""
        e = self.e
        shape = self.shape(name, body)
        if e.needs.pick() not in ('cheapest', 'most', 'least') or not shape or not shape['next'] or not shape['arg'] or not isinstance(body, dict):
            return ''
        count, following, rows = body.get(shape['count']), body.get(shape['next']), body.get(shape['rows'])
        given = args.get(shape['arg'])
        later = given not in (None, 0, 1, '')
        if not isinstance(count, int) or not rows or following in (None, '', False) or later or count <= len(rows):
            return ''
        prices = [e.book.price(i) for i, _ in e.book.records_in(body)]
        if e.needs.pick() == 'cheapest' and len(prices) >= 3 and None not in prices and prices == sorted(prices):
            return ''
        pages = -(-(count - len(rows)) // len(rows))
        if pages > 2 and self.pages_read + 2 > AUTO_PAGES:
            return f"\n[desk] This search matched {count} listings and showed {len(rows)}, in no price order. The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with {shape['arg']}={jdump(following)}."
        texts = []
        for _ in range(min(pages, 2)):
            if e.store.left() - e.tail() < 12:
                break
            reply = e.use(name, {**args, shape['arg']: following}, e.tail())
            if reply is None:
                break
            self.pages_read += 1
            texts.append(f"{shape['arg']}={jdump(following)}: {e.show(reply)}")
            self.note_result(name, args, reply.body)
            following = reply.body.get(shape['next']) if isinstance(reply.body, dict) else None
            if following in (None, '', False):
                break
        return '\n[the desk read the next pages of this search itself, one step a page]\n' + '\n'.join(texts) if texts else ''

    # searching

    def query_field(self) -> tuple[str, str, dict[str, Any]] | None:
        """The search tool, its free-text argument and the other arguments a worded search needs."""
        tool = self.e.roles.get('search')
        schema = self.e.store.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = text_args(schema)
        named = [k for k in texts if re.search(r'query|phrase|keyword|search|(?:^|_)q(?:_|$)|text|terms?|words?', k, re.I)]
        field = (named or (texts if len(texts) == 1 else []) or [None])[0]
        if not tool or not field:
            used = [k for k in texts if any(isinstance(a.get(k), str) and len(a[k].split()) >= 2 for a in self.used_args)]
            if not tool or not used:
                return None
            field = used[0]
        fixed: dict[str, Any] = {}
        for key, spec in props.items():
            if isinstance(spec, dict) and isinstance(spec.get('enum'), list) and (key in (schema.get('required') or []) or any(key in a for a in self.used_args)):
                shelf_arg = (self.browse or {}).get('category')
                past = [a[key] for a in self.used_args if key in a and field in a and shelf_arg not in a] or [a[key] for a in self.used_args if key in a and shelf_arg not in a]
                fixed[key] = max(past, key=past.count) if past else next((o for o in spec['enum'] if re.search(r'typed|text|word|keyword|query|search', str(o), re.I)), spec['enum'][0])
        return tool, field, fixed

    def search(self, words: str) -> list[str] | None:
        """One worded search. None when the store could not be asked; [] when it found nothing."""
        e = self.e
        plan = self.query_field()
        words = ' '.join(words.split())
        if not plan or not words:
            return None
        tool, field, fixed = plan
        ceiling = ((e.store.schemas[tool].get('properties') or {}).get(field) or {}).get('maxLength')
        if isinstance(ceiling, int) and len(words) > ceiling:
            words = words[:ceiling].rsplit(' ', 1)[0]
        if words.lower() in self.tried:
            return []
        self.tried.add(words.lower())
        reply = e.use(tool, {**fixed, field: words}, e.tail())
        if reply is None:
            return None
        return [i for i, _ in e.book.records_in(reply.body)]

    def note_search(self, args: dict[str, Any]) -> None:
        self.used_args = (self.used_args + [dict(args)])[-8:]
        tool = self.e.roles.get('search')
        for key in text_args(self.e.store.schemas.get(tool or '') or {}):
            if isinstance(args.get(key), str):
                self.tried.add(' '.join(args[key].split()).lower())

    def browse_from_schema(self) -> None:
        """A search tool with an argument for a category and one for a price limit can list a shelf under a price."""
        tool = self.e.roles.get('search')
        props = (self.e.store.schemas.get(tool or '') or {}).get('properties') or {}
        shelf = next((k for k, v in props.items() if isinstance(v, dict) and 'string' in type_names(v) and re.search(SHELF_KEYS + '|label', spaced(k))), None)
        cap = next((k for k, v in props.items() if isinstance(v, dict) and set(type_names(v)) & {'integer', 'number'} and re.search(PRICE_KEYS + '|spend|budget|cap', spaced(k)) and not re.search(PAGE_ARGS, spaced(k))), None)
        if tool and shelf:
            self.browse = {'tool': tool, 'category': shelf, 'price': cap, 'fixed': {}, 'learned': True}

    def browse_mode(self) -> dict[str, Any]:
        """The other arguments a shelf listing takes: as the model's own shelf listings set them; else a mode other than
        the one worded searches use, one whose word says fields or a category when there are several."""
        spec = self.browse or {}
        if not spec.get('learned'):
            return spec.get('fixed') or {}
        props = (self.e.store.schemas.get(spec['tool']) or {}).get('properties') or {}
        fixed: dict[str, Any] = {}
        for key, option in props.items():
            options = option.get('enum') if isinstance(option, dict) else None
            if not isinstance(options, list) or key in (spec['category'], spec['price']):
                continue
            seen = [a[key] for a in self.used_args if key in a and spec['category'] in a]
            worded = {a.get(key) for a in self.used_args if key in a and spec['category'] not in a}
            others = [o for o in options if o not in worded] or options
            fixed[key] = seen[-1] if seen else next((o for o in others if re.search(r'field|categor|depart|browse|filter|list', str(o), re.I)), others[0])
        return fixed

    def read_browse(self, spec: Any) -> None:
        schemas = self.e.store.schemas
        if not isinstance(spec, dict) or spec.get('tool') not in schemas:
            self.browse_from_schema()
            return
        props = schemas[spec['tool']].get('properties') or {}
        category, cap = spec.get('category'), spec.get('max_price')
        if category not in props:
            self.browse_from_schema()
        else:
            fixed = spec.get('fixed') if isinstance(spec.get('fixed'), dict) else {}
            self.browse = {'tool': spec['tool'], 'category': category, 'price': cap if cap in props else None, 'fixed': {k: v for k, v in fixed.items() if k in props and k not in (category, cap)}}

    def list_shelf(self, number: int, cap: float | None) -> list[str] | None:
        """The store's own listing of an item's category, under a price where it takes one. None when it has none."""
        spec = self.browse
        if not spec or self.browse_misses >= 2 or (cap is not None and not spec['price']):
            return None
        args = {**self.browse_mode(), spec['category']: self.e.needs.shelf_name(number)}
        if cap is not None:
            args[spec['price']] = cap
        reply = self.e.use(spec['tool'], args, self.e.tail())
        if reply is None or reply.failed():
            self.browse_misses += 1
            return None
        self.note_result(spec['tool'], args, reply.body)
        return [i for i, _ in self.e.book.records_in(reply.body)]

    def ladder(self, number: int, extra: list[str]) -> list[str]:
        """Searches for an item, most telling first: each value the shopper just gave with the product's name and alone,
        then values together, then the product's other names."""
        needs = self.e.needs
        item = needs.items[number - 1]
        values = needs.values_of(number)
        fresh = [v for v in values if any(t.partition(':')[2].strip() == v for t in item.told)]
        values = fresh + [v for v in values if v not in fresh]
        name = item.name
        shelf = needs.shelf_name(number)
        queries = list(extra)
        for value in values[:3]:
            queries += [f'{value} {name}', value]
        if len(values) >= 2:
            queries.append(f'{values[0]} {values[1]}')
        for value in values[:2]:
            if spaced(shelf) != spaced(name):
                queries.append(f'{value} {shelf}')
        queries += [name, shelf]
        return list(dict.fromkeys(q for q in queries if q.strip()))

    def promising(self, number: int, found: list[str], below: float | None = None) -> list[str]:
        """Unread rows of the item's kind that may be bought: rows a search on a required value brought up first, since a
        store's search may read attributes that rows do not show, then most of the values in their own text."""
        e, book = self.e, self.e.book
        values = e.needs.values_of(number)
        rows = []
        for ident in dict.fromkeys(found):
            label = book.label(ident)
            if ident in book.full or not book.buyable(ident) or not e.needs.within_budget(ident):
                continue
            if label and e.needs.owner(label) != number:
                continue
            price = book.price(ident)
            if below is not None and (price is None or price >= below):
                continue
            text = word_set(book.text_of(ident))
            rows.append((ident not in self.value_rows, -sum(word_set(v) <= text for v in values), -len(word_set(' '.join(values), 3) & text), price if price is not None else float('inf'), ident))
        return [row[-1] for row in sorted(rows)]

    def run(self, numbers: list[int], extra: dict[int, list[str]] | None = None, steps: int = HUNT_STEPS) -> list[str]:
        """Half the room goes to searching, one search per item in turn, and the rest to opening what those searches and
        unread pages brought up. Returns the listings newly opened."""
        e = self.e
        keep = e.tail() + len(e.needs.items) + 2
        room = min(steps, e.store.left() - keep)
        if room < 3 or not e.clock.may_explore() or not numbers:
            return []
        self.hunts += 1
        before = set(e.book.full)
        last = e.store.step + room
        search_until = e.store.step + max(1, room // 2 + room % 2)
        queues = {n: self.ladder(n, (extra or {}).get(n, [])) for n in numbers}
        pools: dict[int, list[str]] = {n: e.needs.ids_of(n) for n in numbers}
        for number in numbers:
            if self.browse and e.store.step < search_until:
                pools[number] += self.list_shelf(number, e.needs.budget if self.browse.get('price') else None) or []
        while e.store.step < search_until and any(queues.values()) and e.clock.may_explore():
            for number in numbers:
                if queues[number] and e.store.step < search_until:
                    words = queues[number].pop(0)
                    found = self.search(words)
                    if found is None:
                        queues[number] = []
                    else:
                        pools[number] += found
                        if any(v.lower() in words.lower() for v in e.needs.values_of(number)):
                            self.value_rows.update(found)
        for number in numbers:
            if e.store.step < search_until or not self.promising(number, pools[number]):
                pools[number] += self.read_on(number, last - 1)
        share = max(2, (last - e.store.step) // max(1, len(numbers)))
        for number in numbers:
            room_now = last - e.store.step
            if room_now > 0:
                picks = self.promising(number, pools[number])[:share * (e.opener.compare_room() if e.opener.compare_useful() else 1)]
                e.opener.read(picks, max(keep, e.store.left() - room_now))
        fresh = [i for i in e.book.full if i not in before]
        e.trace.append({'kind': 'hunt', 'items': numbers, 'opened': fresh[:12], 'steps': e.store.step - (last - room), 'tried': sorted(self.tried)[-6:]})
        return fresh

    def after_answers(self, numbers: list[int]) -> str:
        """After firm answers: search the new values at once and read the first rows that show them."""
        e = self.e
        if e.store.left() < e.tail() + 14 or not e.clock.may_explore():
            return ''
        steps = min(8, e.store.left() - e.tail() - 10)
        found = self.run(numbers, steps=steps)
        if not found:
            return ''
        lines = [f"{i}: {e.book.label(i)}, {self.fit_words(i)}" for i in found[:6]]
        return "[the desk searched each answered item's requirement values and read these rows] " + ' | '.join(lines)

    def fit_words(self, ident: str) -> str:
        needs = self.e.needs
        number = needs.owner(needs.book.label(ident))
        if not number:
            return 'kind unclear'
        met, text, unknown, broken = needs.standing(ident, number)
        return f'{met + text} requirements shown, {unknown} not settled by its record, {broken} broken'

    def fits_by_code(self, number: int, keyed_only: bool = False) -> list[str]:
        """Opened listings of an item whose record shows every requirement met, cheapest first; empty when the item has
        none to show. `keyed_only` wants each shown under a key that is the property, not merely in the listing's text."""
        needs, book = self.e.needs, self.e.book
        if not needs.items[number - 1].live():
            return []
        out = []
        for ident in needs.ids_of(number):
            if ident in book.full and book.buyable(ident) and needs.within_budget(ident):
                met, text, unknown, broken = needs.standing(ident, number)
                if (met or text) and not unknown and not broken and not (keyed_only and text):
                    out.append(ident)
        return sorted(out, key=lambda i: book.price(i) if book.price(i) is not None else float('inf'))

    def other_values(self, number: int) -> list[str]:
        """Values of the item's open keys: searched with the required values they bring up rows a plain search misses."""
        said = ' '.join(self.e.needs.values_of(number)).lower() + ' ' + self.e.needs.items[number - 1].name.lower()
        out = []
        for slot in self.e.needs.open_keys(number)[:3]:
            out += [v for v in slot['values'] if v and len(v) <= 20 and v not in said and re.search(r'[a-z0-9]', v)][:2]
        return out

    def sweep_cheaper(self, lines: list[str]) -> list[str]:
        """For a cheapest pick, once per cart: bring up rows priced below each line and read the cheapest of them."""
        e, needs, book = self.e, self.e.needs, self.e.book
        mark = e.cart.mark() + f'#{needs.version}'
        room = e.store.left() - e.spare() - 1
        if needs.pick() != 'cheapest' or mark in self.swept or room < 1 or not e.clock.may_explore():
            return []
        self.swept.add(mark)
        last = e.store.step + min(SWEEP_STEPS, room)
        shown: list[str] = []
        for line in lines:
            price, number = book.price(line), needs.owner(book.label(line))
            if price is None or not number or e.store.step >= last:
                continue
            found = self.list_shelf(number, int(price) - 1 if price > 1 else price * 0.9)
            if found is None:
                found = []
                base = ' '.join(needs.values_of(number)[:2] + [needs.items[number - 1].name])
                for other in self.other_values(number)[:3]:
                    if e.store.step < last - 1:
                        found += self.search(f'{other} {base}') or []
            found += [i for i in needs.ids_of(number) if i not in found]
            picks = self.promising(number, found, price)
            shown += e.opener.read(picks[:3 if len(lines) == 1 else 2], max(e.spare(), e.store.left() - (last - e.store.step)))
        e.trace.append({'kind': 'sweep', 'read': len(shown)})
        return shown


class Cart:
    """The cart as the episode's own calls and views show it."""

    def __init__(self, errand: 'Errand') -> None:
        self.e = errand
        self.lines: dict[str, int] = {}
        self.touched = False
        self.changed_at = -1
        self.viewed_at = -1
        self.view: Any = None
        self.last_said = ''
        self.dropped: set[str] = set()
        self.built: list[str] = []
        self.history: dict[str, list[tuple[str, int, bool]]] = {}
        self.frozen: dict[str, tuple[str, int]] = {}
        self.warned: set[tuple[str, int]] = set()
        self.moves = 0
        self.refused_moves = 0
        self.by_code = False
        self.turn_stopped = False
        self.unread = False
        self.code_moved: set[str] = set()
        self.disagreements = 0

    def planned_moves(self, name: str, args: dict[str, Any]) -> list[tuple[str, str]]:
        """Which listings a cart call would put in ('in') or take out ('out')."""
        rows = self.e.book.rows
        moves: list[tuple[str, str]] = []
        for value in args.values():
            lines = [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []
            if isinstance(value, list) and (lines or not value):
                listed = {v for line in lines for v in line.values() if isinstance(v, str) and v in rows}
                moves += [(i, 'in') for i in listed if i not in self.lines] + [(i, 'out') for i in self.lines if i not in listed]
        if moves:
            return moves
        out = self.e.takes_off(name, args)
        for value in args.values():
            for ident in value if isinstance(value, list) else [value]:
                if isinstance(ident, str) and ident in rows and (out or ident not in self.lines):
                    moves.append((ident, 'out' if out else 'in'))
        return moves

    def remember(self, moves: list[tuple[str, str]]) -> None:
        version = self.e.needs.version
        for ident, way in moves:
            self.history.setdefault(ident, []).append((way, version, self.by_code))
            if way == 'in':
                self.e.needs.adopt(self.e.book.label(ident))
            if self.by_code and way == 'in':
                self.code_moved.add(ident)
            elif not self.by_code and way == 'out' and ident in self.code_moved:
                self.disagreements += 1
        self.moves += 0 if self.by_code else len(moves)

    def settled_way(self, ident: str, way: str) -> str:
        """A listing has already changed direction once with nothing new learned, and is about to again. Where this ends
        is decided now and holds until something changes: a line the desk itself moved goes with the model's wish, since
        two disagreements mean the desk's reading is not to be trusted here; anything else stays in only if its record
        shows every requirement."""
        needs = self.e.needs
        if ident in self.code_moved:
            return way
        number = needs.owner(self.e.book.label(ident))
        met, text, unknown, broken = needs.standing(ident, number) if number else (0, 0, 1, 0)
        return 'in' if (met or text) and not unknown and not broken else 'out'

    def refusal(self, name: str, args: dict[str, Any]) -> str:
        """Why a cart call of the model should not run: moves that undo moves without anything new learned use up whole
        episodes, so each listing gets one change of mind. No step is used by a refusal."""
        e, needs, book = self.e, self.e.needs, self.e.book
        version = needs.version
        moves = self.planned_moves(name, args)
        coming_in = [i for i, way in moves if way == 'in']
        for ident, way in moves:
            if ident in book.gone and way == 'out':
                continue
            frozen = self.frozen.get(ident)
            if frozen and frozen[1] == version and frozen[0] != way:
                return self.refused(f'{ident} was settled {frozen[0]} of the cart after it went back and forth, and nothing has changed since. Check out with the cart as it stands, or choose a different listing for an item that has no line')
            past = [h for h in self.history.get(ident, []) if h[1] == version]
            turned = sum(a[0] != b[0] for a, b in zip(past, past[1:]))
            if past and past[-1][0] != way and turned >= 1:
                final = self.settled_way(ident, way)
                self.frozen[ident] = (final, version)
                e.trace.append({'kind': 'cart_settled', 'id': ident, 'way': final})
                if final != way:
                    return self.refused(f"{ident} has gone back and forth with nothing new learned. It stays {final} of the cart: its record {'shows every requirement' if final == 'in' else 'does not show every requirement'}. Check out, or choose a different listing")
            if way == 'out' and needs.pick() != 'most' and (ident, version) not in self.warned:
                number = needs.owner(book.label(ident))
                keyed = number and ident in e.hunter.fits_by_code(number, keyed_only=True)
                cheaper = [i for i in coming_in if keyed and needs.pick() == 'cheapest' and (book.price(i) if book.price(i) is not None else float('inf')) < (book.price(ident) or 0)]
                if keyed and not cheaper:
                    self.warned.add((ident, version))
                    return self.refused(f"{ident}'s record states every requirement in force, and requirements come from the ledger, not from guesses about the story. If its record breaks one, say which in your text and send this again; otherwise keep it and check out")
        for ident in coming_in:
            number = needs.owner(book.label(ident))
            held = [i for i in self.lines if needs.owner(book.label(i)) == number and self.frozen.get(i, ('', -1)) == ('in', version)]
            if number and held and ident not in held and not any(i == held[0] and way == 'out' for i, way in moves):
                return self.refused(f'{held[0]} was settled as the line for that item. Check out with it')
        if moves and self.moves >= 4 * len(needs.items) + 8:
            gaps = [i for i in coming_in if not any(needs.owner(book.label(c)) == needs.owner(book.label(i)) for c in self.lines)]
            if not gaps and not all(i in book.gone for i, way in moves if way == 'out'):
                return self.refused('the cart has been changed enough times for what is known. Check out with it as it stands')
        return ''

    def spare_lines(self) -> list[str]:
        """Second lines of one item: which to keep goes by its record, then by the pick rule, then by the first carted."""
        needs, book = self.e.needs, self.e.book
        spare: list[str] = []
        for number in range(1, len(needs.items) + 1):
            lines = [i for i in self.ids() if needs.owner(book.label(i)) == number]
            if len(lines) > 1:
                fits = self.e.hunter.fits_by_code(number)
                keep = next((i for i in fits if i in lines), lines[0])
                spare += [i for i in lines if i != keep]
        return spare

    def refused(self, text: str) -> str:
        self.refused_moves += 1
        self.turn_stopped = True
        return f'Not run (no step used): {text}. Cart now: {self.text()}'

    def fresh_view(self) -> bool:
        return self.view is not None and self.viewed_at >= self.changed_at

    def ids(self) -> list[str]:
        if self.fresh_view():
            text = jdump(self.view)
            return [i for i in self.e.book.rows if i in text]
        return list(self.lines)

    def mark(self) -> str:
        return jdump(sorted(self.lines.items()))

    def text(self) -> str:
        if self.fresh_view():
            return jdump(clipped(self.view, 300))[:3000]
        return "lines by the desk's record of the cart calls (listing: units): " + jdump(self.lines)

    def units(self, ident: str) -> int:
        if self.fresh_view():
            stack: list[Any] = [self.view]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    if ident in [v.strip() for v in node.values() if isinstance(v, str)]:
                        counts = [whole_number(v) for k, v in node.items() if not isinstance(v, str) and whole_number(v) is not None and re.search(r'\b(?:qty|quantity|count|units?|number|held)\b', spaced(k))]
                        if len(counts) == 1:
                            return counts[0]
                    stack += [v for v in node.values() if isinstance(v, (dict, list))]
                elif isinstance(node, list):
                    stack += node
        return self.lines.get(ident, 1)

    def sets_count(self, name: str) -> bool:
        """Whether the tool's count sets a line rather than adding to it: no mode argument and a count that may be 0."""
        props = (self.e.store.schemas.get(name) or {}).get('properties') or {}
        if any(isinstance(s, dict) and isinstance(s.get('enum'), list) and len(s['enum']) >= 2 for s in props.values()):
            return False
        return any(isinstance(s, dict) and 'integer' in type_names(s) and s.get('minimum') == 0 for s in props.values())

    def note(self, name: str, args: dict[str, Any], body: Any) -> None:
        book = self.e.book
        self.remember(self.planned_moves(name, args))
        self.touched = True
        self.changed_at = self.e.store.step
        self.last_said = jdump(body)
        spec = (self.e.store.schemas.get(name) or {}).get('properties') or {}
        removal = any(isinstance((spec.get(k) or {}).get('enum'), list) and len(spec[k]['enum']) >= 2 and v in spec[k]['enum'][1:] for k, v in args.items())
        count_arg = next((whole_number(v) for v in args.values() if not isinstance(v, str) and whole_number(v) is not None), None)
        seen: list[tuple[str, int | None]] = []
        whole_cart = False
        for key, value in args.items():
            if isinstance(value, list):
                shape = (spec.get(key) or {}).get('items') or {}
                whole_cart = whole_cart or (not value and isinstance(shape, dict) and (shape.get('type') == 'object' or 'properties' in shape))
                for line in value:
                    if isinstance(line, dict):
                        whole_cart = True
                        ident = next((v for v in line.values() if isinstance(v, str) and v in book.rows), None)
                        if ident:
                            seen.append((ident, next((whole_number(v) for v in line.values() if not isinstance(v, str) and whole_number(v) is not None), 1)))
                    elif isinstance(line, str) and line in book.rows:
                        seen.append((line, count_arg))
            elif isinstance(value, str) and value in book.rows:
                seen.append((value, count_arg))
        self.unread = self.unread or (not seen and not whole_cart)
        if whole_cart:
            self.lines = {i: c for i, c in seen if c and c > 0}
        else:
            setting = self.sets_count(name)
            for ident, count in seen:
                have = self.lines.get(ident, 0)
                if removal:
                    have = have - count if count else 0
                elif setting:
                    have = count if count is not None else 1
                else:
                    have += count if count is not None else 1
                if have > 0:
                    self.lines[ident] = have
                else:
                    self.lines.pop(ident, None)
        for ident, count in seen:
            if count is not None and count <= 0:
                self.dropped.add(ident)
            else:
                self.dropped.discard(ident)
        kept = [i for i in self.ids() if not book.sold_out(i) and i not in self.dropped]
        if len(kept) >= len(self.built):
            self.built = kept

    def calls_for(self, wanted: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        """The cart calls that reach these lines, or None when this tool's way of working cannot be read."""
        name = self.e.roles.get('cart') or ''
        schema = self.e.store.schemas.get(name) or {}
        props = schema.get('properties') or {}
        for key, spec in props.items():
            shape = spec.get('items') if isinstance(spec, dict) and 'array' in type_names(spec) else None
            if isinstance(shape, dict) and isinstance(shape.get('properties'), dict):
                inner = shape['properties']
                id_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'string' in type_names(s) and 'enum' not in s), None)
                count_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'integer' in type_names(s)), None)
                if not id_key:
                    return None
                return [{key: [{id_key: i, **({count_key: n} if count_key else {})} for i, n in wanted.items()]}]
        target = next((k for t, k in self.e.id_args if t == name and k in text_args(schema)), None) or lone_target(schema) or next(iter(k for k in schema.get('required') or [] if k in text_args(schema)), None)
        count = next((k for k, s in props.items() if isinstance(s, dict) and 'integer' in type_names(s)), None)
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get('enum'), list) and len(s['enum']) >= 2), None)
        if not target:
            return None
        calls: list[dict[str, Any]] = []
        for ident in drop:
            have = self.lines.get(ident, 1)
            if mode:
                calls.append({target: ident, mode: self.mode_word(props[mode]['enum'], True), **({count: have} if count else {})})
            elif count and self.sets_count(name):
                calls.append({target: ident, count: 0})
            elif count and isinstance(props[count].get('minimum'), int) and props[count]['minimum'] <= -have:
                calls.append({target: ident, count: -have})
            else:
                return None
        for ident, want in wanted.items():
            have = self.lines.get(ident, 0) if ident in self.ids() else 0
            if have == want:
                continue
            if have and mode and count:
                calls.append({target: ident, mode: self.mode_word(props[mode]['enum'], True), count: have})
                have = 0
            elif have and not self.sets_count(name):
                return None
            calls.append({target: ident, **({mode: self.mode_word(props[mode]['enum'], False)} if mode else {}), **({count: want} if count else {})})
        return calls

    @staticmethod
    def mode_word(options: list[Any], taking_off: bool) -> Any:
        words = [str(o) for o in options]
        off = next((o for o in words if re.search(r'remov|delet|drop|clear|take|discard|minus|subtract', o, re.I)), words[1])
        return off if taking_off else next(o for o in words if o != off)

    def faults(self) -> list[str]:
        """What is plainly wrong with the cart against the request, from the store's labels and prices."""
        e, needs, book = self.e, self.e.needs, self.e.book
        ids = [i for i in self.ids() if self.units(i) > 0]
        if not ids or not book.shelf_field():
            return []
        owners = {i: needs.owner(book.label(i)) for i in ids}
        known = {n for n in range(1, len(needs.items) + 1) if needs.ids_of(n)}
        found: list[str] = []
        for number in sorted(known):
            item = needs.items[number - 1]
            lines = [i for i in ids if owners[i] == number]
            if not lines:
                found.append(f'item {number} ({item.name}) has no line in the cart')
            elif len(lines) > 1:
                found.append(f"item {number} ({item.name}) has {len(lines)} lines ({', '.join(lines)}); keep one")
            for ident in lines:
                if self.units(ident) != item.qty:
                    found.append(f'{ident} is held at {self.units(ident)} unit(s); item {number} ({item.name}) wants {item.qty}')
        if len(known) == len(needs.items):
            found += [f"{i} ({book.label(i) or 'unknown kind'}) is not a kind the shopper asked for: take it out" for i in ids if not owners[i]]
        found += [f'{i} is sold out' for i in ids if book.sold_out(i)]
        prices = {i: book.price(i) for i in ids}
        if needs.budget is not None and None not in prices.values():
            if needs.brief.get('budget_scope') == 'each':
                found += [f'{i} costs {p:g}, over the {needs.budget:g} limit for each item' for i, p in prices.items() if needs.budget < p <= 10 * needs.budget]
            else:
                total = sum(p * self.units(i) for i, p in prices.items())
                if needs.budget + 1e-09 < total <= 10 * needs.budget:
                    found.append(f'the total is {total:g}, over the budget of {needs.budget:g}')
        return found

    def fix_units(self) -> list[str]:
        """Set wrong quantities right when each is the only line of its item; the rest is the model's to fix."""
        e, needs = self.e, self.e.needs
        wanted, wrong = {}, False
        for ident in self.ids():
            number = needs.owner(e.book.label(ident))
            if not number:
                return []
            wanted[ident] = needs.items[number - 1].qty
            wrong = wrong or self.units(ident) != wanted[ident]
        if not wrong or len(set(needs.owner(e.book.label(i)) for i in wanted)) != len(wanted):
            return []
        calls = self.calls_for(wanted, [])
        if not calls or len(calls) + 3 > e.store.left():
            return []
        done = []
        for args in calls:
            reply = e.use(e.roles['cart'], args)
            if reply is None:
                break
            self.note(e.roles['cart'], args, reply.body)
            done.append(jdump(args) + ' -> ' + e.show(reply)[:300])
        e.trace.append({'kind': 'unit_fix', 'calls': done})
        return done


class Desk:
    """Everything between a model's wish to end and the ending itself: checks before an order, a review before a decline."""

    def __init__(self, errand: 'Errand') -> None:
        self.e = errand
        self.gate_passed: set[tuple[str, str]] = set()
        self.checked: set[str] = set()
        self.verdicts: dict[str, bool] = {}
        self.offer_seen: dict[str, int] = {}
        self.sold_lines: set[str] = set()
        self.note_sold = False
        self.note_named = False
        self.leads: list[str] = []
        self.lead_items: dict[str, list[str]] = {}
        self.audited = False
        self.blocks = 0
        self.pushed = False
        self.late_held = False
        self.confirmed: str | None = None
        self.completed = False
        self.buys = 0
        self.refused_at = -1
        self.line_holds = 0
        self.fit_holds = 0
        self.trail_rounds = 0
        self.trail_blocks = 0
        self.link_searches = 0
        self.sold_holds = 0
        self.code_swaps: set[tuple[str, str]] = set()
        self.last_tried = False

    # steps the request asks for

    def gate_due(self, stage: str) -> bool:
        needs = self.e.needs
        return any(p['when'] == stage for p in needs.process) and (stage, self.e.cart.mark() + f'#{needs.version}') not in self.gate_passed

    def gate(self, stage: str) -> tuple[bool, str]:
        """Carry out the steps due before this ending; False when a reply changes the task."""
        e, needs, talk = self.e, self.e.needs, self.e.talk
        steps = [p for p in needs.process if p['when'] == stage]
        if not steps or e.store.finished or not self.gate_due(stage) or e.store.left() < 2 or not e.clock.affords(LIGHT):
            return True, ''
        ending = {e.roles.get('buy'), e.roles.get('refuse')} - {None}
        tools = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {jdump(t['function'].get('parameters') or {})}" for t in e.store.tools if t['function']['name'] not in ending]
        said: list[str] = []
        nudged = False
        for _ in range(GATE_ROUNDS):
            prompt = GATE.replace('STAGE', 'place the order' if stage == 'order' else 'end without a purchase')
            prompt += f"\n\nREQUEST:\n{e.request}\n\nSTEPS DUE NOW:\n" + '\n'.join(f"- {p['step']}" for p in steps) + f'\n\nLEDGER:\n{e.ledger()}\n\nCART NOW: {e.cart.text()}'
            prompt += '\n\nRECORD (messages and step calls so far):\n' + ('\n'.join(f'{i}. {line}' for i, line in enumerate(talk.record, 1)) or '(nothing yet)')
            prompt += ('\n\nSTORE NOTES SEEN:\n' + '\n'.join(needs.notes[-4:]) if needs.notes else '') + f"\n\nMESSAGES LEFT: {max(0, talk.limit - talk.used)}   STEPS LEFT: {e.store.left()} (the {'order' if stage == 'order' else 'ending'} itself takes one)\n\nTOOLS:\n" + '\n'.join(tools)
            got = e.wire.ask_json(prompt, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 3500)
            if got is None:
                break
            told: list[str] = []
            hold = ''
            if said:
                talk.take_heard({'needs': got.get('needs')}, told)
                hold = str(got.get('hold') or '').strip()
            call = got.get('call') if isinstance(got.get('call'), dict) else None
            e.trace.append({'kind': 'gate', 'stage': stage, 'call': clipped(call, 200), 'hold': hold[:200], 'told': told})
            if told or hold:
                needs.version += 1
                return False, ('Before going on, the desk did the steps the request asks for:\n' + '\n'.join(said) + '\n' if said else '') + "The shopper's reply changes the task: " + '; '.join(told + ([hold] if hold else [])) + '. Fix the cart for it, then check out again.'
            shown = [x for x in as_list(got.get('steps')) if isinstance(x, dict)]
            talking = stage == 'order' and any(p['messages'] > 0 for p in steps)
            proof = {i for i in range(1, len(talk.record) + 1) if not talking or (i <= len(talk.record_steps) and talk.record_steps[i - 1] > e.cart.changed_at)}
            unproven = [x for x in shown if x.get('done') is not True or not any((whole_number(n) or 0) in proof for n in as_list(x.get('lines')))]
            if not call and unproven and not said and not nudged and talk.tool and talk.used < talk.limit and e.store.left() >= 2:
                nudged = True
                talk.record.append(f'(the desk: no line made since the cart last changed at step {e.cart.changed_at} shows ' + '; '.join(str(x.get('step') or '') for x in unproven)[:200] + f' done, and {talk.limit - talk.used} message(s) and {e.store.left()} steps are left: give the call that does it)')
                talk.record_steps.append(-1)
                continue
            name = str((call or {}).get('tool') or '')
            if not call:
                self.gate_passed.add((stage, e.cart.mark() + f'#{needs.version}'))
                break
            if name not in e.store.schemas or name in ending or e.store.left() < 2:
                break
            args = e.store.fitted(name, call.get('args') if isinstance(call.get('args'), dict) else {})
            if args is None:
                break
            if name == talk.tool:
                text = next((v for v in args.values() if isinstance(v, str)), '')
                answer = None if self.asks_properties(text) else talk.send(text, 1)
                if answer is None:
                    break
                talk.kept_used += 1
                said.append(f'Q: {text}\nA: {answer}')
                continue
            reply = e.use(name, args, 1)
            if reply is None:
                break
            talk.record.append(f'step {e.store.step}: {name} {jdump(args)} -> {e.show(reply)[:300]} (cart then: {e.cart.mark()})')
            talk.record_steps.append(e.store.step)
            said.append(f'{name} {jdump(args)} -> {e.show(reply)[:400]}')
        return True, 'Before going on, the desk did the steps the request asks for:\n' + '\n'.join(said) if said else ''

    def asks_properties(self, text: str) -> bool:
        """Keeps a model-written process message to the same rule as the questions: it may not ask about two properties."""
        needs = self.e.needs
        low = text.lower()
        if '?' not in low or not re.search(r'\b(?:would you like|do you (?:need|want|prefer)|which|what|prefer)\b', low):
            return False
        keys = {s['name'] for n in range(1, len(needs.items) + 1) for s in needs.key_menu(n) if has_words(s['name'], low)}
        return len({k for k in keys if not any(k != o and k in o for o in keys)}) >= 2

    # what a cart is checked against

    def check(self, ids: list[str] | None = None, limit: float = SPEND_SOFT) -> str:
        e, book, needs = self.e, self.e.book, self.e.needs
        planned = ids is not None
        ids = ids if planned else e.cart.ids()
        key = self.check_key(ids)
        if key in self.checked or not e.wire.judges(limit) or not ids:
            return ''
        self.checked.add(key)
        prompt = f'{CHECK}\n\nREQUEST:\n{e.request}\n\nREQUIREMENTS IN FORCE:\n{e.ledger()}'
        if planned:
            prompt += '\n\nCART (planned, judge these listings as if they were carted at the requested quantities: ' + '; '.join(f'item {n}: quantity {it.qty}' for n, it in enumerate(needs.items, 1)) + '):\n' + ', '.join(ids)
        else:
            prompt += f'\n\nCART:\n{e.cart.text()}'
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + '\n'.join(f'{i}: ' + ('SOLD OUT (the store says it can no longer be bought). ' if book.sold_out(i) else '') + book.row_text(i) for i in ids)
        if needs.pick() in ('cheapest', 'most', 'least'):
            prompt += '\n\nOTHER LISTINGS SEEN (newest first):\n' + self.others(ids, 15)
        got = e.wire.ask_json(prompt, e.wire.judges(limit), 3000)
        e.trace.append({'kind': 'check', 'cart': ids, 'got': got})
        self.verdicts[key] = bool(got) and got.get('ok') is not False
        if not got or got.get('ok') is not False:
            return ''
        self.leads = [i for i in self.leads if i not in ids]
        better = [str(b) for b in as_list(got.get('better')) if str(b) in book.rows and not book.sold_out(str(b))][:4]
        text = 'A careful review says this cart is not right yet: ' + '; '.join([str(p) for p in as_list(got.get('problems'))][:5] or ['see the requirements'])
        return text + ('. Listings that look better: ' + ', '.join(better) if better else '') + '. Fix the cart (or show the review is wrong by re-reading the records), then check out.'

    def check_key(self, ids: list[str]) -> str:
        return '|'.join(sorted(ids)) + f'#{self.e.needs.version}'

    def others(self, skip: list[str], limit: int) -> str:
        book = self.e.book
        order = [i for i in reversed(book.order) if i not in skip and not book.sold_out(i)]
        if self.e.needs.pick() == 'cheapest':
            order.sort(key=lambda i: (i not in book.full, book.price(i) if book.price(i) is not None else float('inf')))
        return '\n'.join(jdump(clipped(book.record(i), 160))[:900] for i in order[:limit])

    def fit_faults(self) -> list[str]:
        """A part in the cart whose own words rule out the device in the cart."""
        e, needs, book = self.e, self.e.needs, self.e.book
        cart = e.cart.ids()
        found = []
        marks: dict[str, dict[str, Any]] = {}

        def marks_of(ident: str) -> dict[str, Any]:
            if ident not in marks:
                marks[ident] = fit_statements(book.text_of(ident))
            return marks[ident]
        for part, main in needs.bound:
            devices = [i for i in cart if needs.owner(book.label(i)) == main]
            parts = [i for i in cart if needs.owner(book.label(i)) == part]
            if len(devices) != 1:
                continue
            device = marks_of(devices[0])
            for line in parts:
                if fit_verdict(marks_of(line), device) is False:
                    good = [i for i in needs.ids_of(part) if fit_verdict(marks_of(i), device) and not book.sold_out(i) and i not in cart][:4]
                    found.append(f'{line} ({needs.items[part - 1].name}) does not name {devices[0]} among what it works with' + (f"; seen parts of that kind that do: {', '.join(good)}" if good else ''))
        return found

    # pages a requirement points at

    def trail(self) -> tuple[str, bool]:
        """Open the chain of pages each pointing requirement names; the value on the last page settles it."""
        e, needs, book = self.e, self.e.needs, self.e.book
        live = {need for _, need in needs.now()}
        pending = [need for _, need in needs.refs if need in live and need not in needs.ref_values]
        tool = e.offer_tool()
        target = lone_target(e.store.schemas.get(tool or '') or {})
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trail_rounds < (6 if e.store.left() > 15 else 4) and e.wire.spent < SPEND_HARD and e.clock.affords(LIGHT):
            self.trail_rounds += 1
            opened = [i for i in reversed(book.order) if i in book.opened][:45]
            rest = [i for i in reversed(book.order) if i not in book.opened][:max(0, 60 - len(opened))]
            prompt = TRAIL + '\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n' + '\n'.join(pending) + '\n\nRECORDS (newest first):\n'
            prompt += '\n'.join(['OPENED ' + jdump(clipped(book.record(i), 300))[:1400] for i in opened] + ['ROW ' + jdump(clipped(book.rows.get(i), 80))[:300] for i in rest])
            got = e.wire.ask_json(prompt, [LIGHT] + ([HEAVY] if e.wire.spent + HEAVY_ROOM <= SPEND_HARD else []), 4500) or {}
            e.trace.append({'kind': 'trail', 'pending': pending, 'got': clipped(got, 150)})
            reach: list[str] = []
            lost = []
            gaps: list[tuple[str, str]] = []
            for chain in as_list(got.get('chains')):
                if not isinstance(chain, dict):
                    continue
                need = next((p for p in pending if p == str(chain.get('need') or '').strip()), None) or (pending[0] if len(pending) == 1 else None)
                if not need:
                    continue
                links = [x for x in as_list(chain.get('links')) if isinstance(x, dict)]
                ids = [str(x.get('id') or '').strip() for x in links]
                value = str(chain.get('value') or '').strip()
                if links and all(i in book.opened for i in ids) and ((chain.get('done') is True and value) or not str(links[-1].get('find') or '').strip()):
                    needs.ref_values[need] = value or f'as stated on {ids[-1]}'
                    needs.version += 1
                    continue
                closed = [i for i in ids if i in book.rows and i not in book.opened]
                if closed:
                    reach += [i for i in closed if i not in reach]
                else:
                    gap = next((x for x, i in zip(links, ids) if i not in book.rows), {})
                    about, words = str(gap.get('about') or '') or 'the listing it points at', str(gap.get('find') or '').strip()
                    lost.append(f'{need} -> find {about}' + (f' (search: {words})' if words else ''))
                    gaps.append((about, words))
            pending = [p for p in pending if p not in needs.ref_values]
            if not reach and gaps and tool and target and e.store.left() > 12 and self.link_searches < 3:
                reach = self.find_link(*gaps[0])
            if not reach or not tool or not target or e.store.left() < len(reach) + 4 or e.store.finished:
                break
            for ident in reach[:max(0, min(3, 6 - len(seen)))]:
                reply = e.use(tool, {target: ident})
                if reply is None or e.store.finished:
                    break
                book.opened.add(ident)
                seen.append(f'{ident}: ' + e.show(reply)[:1500])
        text = '[the desk opened the pages a requirement points at]\n' + '\n'.join(seen) + '\n' if seen else ''
        if needs.ref_values:
            text += '[values read from the pages requirements point at] ' + '; '.join(f'{k} -> {v}' for k, v in needs.ref_values.items()) + '\n'
        hold = bool(lost) and self.trail_blocks < (3 if e.store.left() > 15 else 2) and e.store.left() > 5
        if hold:
            self.trail_blocks += 1
            text += 'The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: ' + ' | '.join(lost[:3]) + ". Search for it by its shop's name together with the values its description gives (the shop's other listings of that kind are not it), open the row of that shop whose page shows those values, apply the value its page gives, then check out."
        return text, hold

    def find_link(self, about: str, words: str) -> list[str]:
        """A listing a chain names: rows of the shop it names, after one search on the shop and values when none is known."""
        e, book = self.e, self.e.book
        low = about.lower()
        by_shop = lambda ids: [i for i in ids if i not in book.opened and any(has_words(s, low) for s in book.shop_of(i))]
        known = by_shop(list(reversed(book.order)))
        self.link_searches += 1
        if words:
            found = e.hunter.search(words)
            known = by_shop(found or []) + [i for i in known if i not in (found or [])]
            e.trace.append({'kind': 'link_search', 'words': words[:80], 'rows': known[:4]})
        wanted = {w for w in re.findall(r'[a-z0-9]{3,}', (words or about).lower())}
        return ([i for i in known if wanted & set(re.findall(r'[a-z0-9]{3,}', book.text_of(i)))] or known)[:2]

    # price and stock just before an order

    def reread_offers(self) -> str:
        e, book = self.e, self.e.book
        tool = e.offer_tool()
        target = lone_target(e.store.schemas.get(tool or '') or {})
        since = max(e.store.step - 2, self.refused_at)
        stale = [i for i in e.cart.ids() if self.offer_seen.get(i, -99) < since and i not in book.gone]
        if not tool or not target or not stale or e.store.left() < len(stale) + 3:
            return ''
        lines = []
        for ident in stale:
            before = dict(book.rows.get(ident) or {})
            reply = e.use(tool, {target: ident})
            if reply is None or e.store.finished:
                break
            self.offer_seen[ident] = e.store.step
            moved = [k for k, v in (book.rows.get(ident) or {}).items() if k in before and before[k] != v and not isinstance(v, (dict, list))]
            if moved:
                lines.append(f"{ident} changed ({', '.join(moved[:4])}): {e.show(reply)[:900]}")
        e.trace.append({'kind': 'reread', 'ids': stale, 'changed': len(lines)})
        if lines:
            e.needs.version += 1
        return '\n'.join(lines)

    def find_sold(self) -> str:
        """A note says something in the cart is gone without naming it: one line is that line; several are re-read."""
        book = self.e.book
        ids = [i for i in self.e.cart.ids() if i not in book.gone]
        if any(book.sold_out(i) for i in ids):
            return ''
        if len(ids) == 1:
            book.unavailable.add(ids[0])
            return ''
        return self.reread_offers() if ids else ''

    def swap_hint(self) -> str:
        e, book = self.e, self.e.book
        lost = [i for i in e.cart.ids() if i not in book.gone and book.sold_out(i)]
        if not lost:
            return ''
        for ident in lost:
            book.gone[ident] = e.store.step
        self.leads = [i for i in self.leads if i not in lost]
        e.needs.version += 1
        picks = self.replacements(lost)
        text = f"[desk] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        done = self.swap_lines(lost, picks[:1])
        if done:
            more = f" Other listings already read that appear to fit: {', '.join(picks[1:])}." if picks[1:] else ''
            return text + f' The desk has already changed the cart: {done}.{more} Cart now: {e.cart.text()}. ' + ('Check out again unless a cheaper fitting listing turns up.' if picks else 'Find a fitting listing for that item, cart it and check out.')
        if picks and e.store.left() <= 6:
            return text + f" Few steps are left: in this turn add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks[1:]) or 'none'}."
        if picks and e.needs.pick() == 'cheapest':
            return text + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(picks)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {picks[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        if picks:
            return text + f" Listings already read that appear to fit that item instead: {', '.join(picks)}. In this turn take the sold-out line out of the cart and add {picks[0]} at the requested quantity, then check out."
        return text + ' Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits.'

    def swap_lines(self, lost: list[str], picks: list[str]) -> str:
        """Take these lines out of the cart and put a replacement in, so a gone line can never go out with the order.
        Says what was done; empty when the store's cart tool cannot be worked or too few steps are left for it."""
        e, book, needs = self.e, self.e.book, self.e.needs
        tool = e.roles.get('cart')
        have = {i: (e.cart.units(i) or 1) for i in e.cart.ids() if i not in lost}
        new = {i: needs.item_of(needs.owner(book.label(i)) or 1).qty for i in picks if i not in have}
        drop = [i for i in lost if i in e.cart.ids()]
        calls = e.cart.calls_for({**have, **new}, drop) if tool and (drop or new) else None
        if not calls or len(calls) + 1 > e.store.left():
            return ''
        done = []
        e.cart.by_code = True
        for args in calls:
            reply = e.use(tool, args)
            if reply is None:
                break
            e.cart.note(tool, args, reply.body)
            done.append(jdump(args))
        e.cart.by_code = False
        e.trace.append({'kind': 'swap_lines', 'lost': lost, 'new': list(new), 'calls': done})
        return '; '.join(done)

    def cheaper_fit_swap(self) -> str:
        """For a cheapest pick: when an opened listing's record shows every requirement under its own keys and costs less
        than the cart's line, the cart takes that listing. This does not wait on the model's reading of the same record."""
        e, book, needs = self.e, self.e.book, self.e.needs
        if needs.pick() != 'cheapest' or e.cart.disagreements >= 2:
            return ''
        for line in e.cart.ids():
            number, price = needs.owner(book.label(line)), book.price(line)
            better = [i for i in e.hunter.fits_by_code(number, keyed_only=True) if i != line and (book.price(i) or 0) < (price or 0)] if number and price is not None else []
            better = [i for i in better if (line, i) not in self.code_swaps and not any(h[0] == 'out' and not h[2] for h in e.cart.history.get(i, []))]
            if better and e.store.left() > e.tail() + 2:
                self.code_swaps.add((line, better[0]))
                done = self.swap_lines([line], better[:1])
                if done:
                    return f"{better[0]} costs {book.price(better[0]):g} against {line}'s {price:g} and its record states every requirement, so the desk changed the cart: {done}."
        return ''

    def replacements(self, lost: list[str]) -> list[str]:
        """Code first: an opened same-kind listing whose record shows every requirement needs no model call."""
        e, book, needs = self.e, self.e.book, self.e.needs
        cart = e.cart.ids()
        numbers = [n for n in dict.fromkeys(needs.owner(book.label(i)) for i in lost) if n]

        def by_code() -> list[str]:
            found = [i for n in numbers for i in e.hunter.fits_by_code(n, keyed_only=True)]
            kept = list(dict.fromkeys(i for i in found if i not in lost and i not in cart and not book.sold_out(i)))
            # Lowest price first: on a cheapest pick that is the answer, on any other pick it costs nothing.
            return sorted(kept, key=lambda i: (book.price(i) is None, book.price(i) or 0))[:4]
        if not by_code() and numbers and e.store.left() > e.tail() + 3:
            # A short search only: the model still needs steps to read what it finds and to order again.
            e.hunter.run(numbers, steps=min(REFUSAL_HUNT, e.store.left() - e.tail() - 3))
        if by_code():
            return by_code()
        if not e.wire.judges():
            return []
        records = [jdump(clipped(book.record(i), 200))[:1200] for i in reversed(book.order) if i not in lost and not book.sold_out(i)][:25]
        if not records:
            return []
        prompt = f'{SWAP}\n\nREQUEST:\n{e.request}\n\nREQUIREMENTS IN FORCE:\n{e.ledger()}\n\nSOLD-OUT CART LINE:\n' + '\n'.join(f'{i}: {book.row_text(i)}' for i in lost) + f'\n\nCART NOW:\n{e.cart.text()}\n\nOTHER LISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
        got = e.wire.ask_json(prompt, e.wire.judges(), 3500) or {}
        fits: list[tuple[float, str]] = []
        for entry in as_list(got.get('fits')):
            ident = str(entry.get('id') if isinstance(entry, dict) else entry)
            number = needs.owner(book.label(ident))
            broken = needs.standing(ident, number)[3] if number else 0
            if ident in book.rows and ident not in lost and ident not in book.gone and ident not in cart and not broken and ident not in [i for _, i in fits]:
                fits.append((amount(entry.get('price') if isinstance(entry, dict) else None) or float('inf'), ident))
        e.trace.append({'kind': 'swap', 'lost': lost, 'fits': fits})
        fits = [(book.price(i) if book.price(i) is not None else p, i) for p, i in fits]
        ranked = sorted(fits) if needs.pick() == 'cheapest' else fits
        return [i for _, i in ranked][:4]

    # the two endings

    def before_order(self) -> str:
        """Checks before an order; a text holds the order and says why, an empty text lets it go."""
        e, book = self.e, self.e.book
        gone = [i for i in e.cart.ids() if i in book.gone]
        if gone:
            done = self.swap_lines(gone, [])
            if done:
                return f"{', '.join(gone)} sold out earlier and was still in the cart; the desk took it out: {done}. Cart now: {e.cart.text()}. Make sure each item has a line, then check out."
        hunted = e.hunter.sweep_cheaper(e.cart.ids())
        swapped = self.cheaper_fit_swap()
        if swapped:
            return swapped + f' Cart now: {e.cart.text()}. Check out again.'
        cart = e.cart
        reread = self.reread_offers() if e.chores_fit(len(cart.ids())) else ''
        if self.note_sold:
            self.find_sold()
        if cart.touched and cart.viewed_at < cart.changed_at and e.roles.get('cart_view') and e.chores_fit():
            reply = e.use(e.roles['cart_view'], {})
            if reply is not None:
                cart.view, cart.viewed_at = reply.body, e.store.step
                change = e.drain_notes()
                if change:
                    return 'Looking at the cart showed a change: ' + change + f'\nCart now: {cart.text()}\nApply the change, fix the cart if needed, then check out.'
        swap = self.swap_hint()
        held = [i for i in cart.ids() if i in book.gone] if self.sold_holds < 2 else []
        if swap or held:
            self.sold_holds += 0 if swap else 1
            return (swap or f"{', '.join(held)} in the cart is sold out: replace that line first.") + f'\nCart now: {cart.text()}'
        chain, hold = self.trail()
        if hold:
            return chain + f'\nCart now: {cart.text()}'
        misfit = self.fit_faults() if self.fit_holds < 2 and e.store.left() > e.tail() + 3 else []
        if misfit:
            self.fit_holds += 1
            e.trace.append({'kind': 'fit_hold', 'faults': misfit[:4]})
            return 'A part in the cart does not work with the device in the cart: ' + '; '.join(misfit[:3]) + f'. Swap the part (or the device), then check out.\nCart now: {cart.text()}'
        faults = cart.faults() if self.line_holds < 2 and e.store.left() > e.tail() + 2 else []
        if faults and all(' is held at ' in f for f in faults):
            fixed = cart.fix_units()
            faults = cart.faults() if fixed else faults
            if fixed and not faults:
                self.line_holds += 1
                return "The desk set the cart's quantities itself: " + '; '.join(fixed) + f'. If the store refused any of these, fix those lines; then check out.\nCart now: {cart.text()}'
        if faults:
            self.line_holds += 1
            e.trace.append({'kind': 'line_hold', 'faults': faults[:6]})
            return 'The cart does not match the request yet: ' + '; '.join(faults[:6]) + f". Fix those lines (the cart tool's own way to add or take off units), then check out.\nCart now: {cart.text()}"
        read = e.opener.read(cart.ids(), e.tail() + e.room_after_refusal() + 1)
        verdict = self.check()
        if verdict:
            seen = "The desk read these listings' attributes for the check:\n" + '\n'.join(read + hunted) + '\n' if read or hunted else ''
            return chain + seen + verdict + f'\nCart now: {cart.text()}'
        if reread and e.wire.spent > SPEND_SOFT:
            return f"The desk re-read the cart's listings just now:\n{reread}\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first."
        return ''

    def uncovered(self) -> list[int]:
        return [n for n in range(1, len(self.e.needs.items) + 1) if not self.e.hunter.fits_by_code(n)]

    def review_decline(self) -> str:
        """A decline stands only after the store was searched on the required values and a judge saw what that found."""
        e, needs, book = self.e, self.e.needs, self.e.book
        if self.audited or not e.wire.judges():
            return ''
        self.audited = True
        found = e.hunter.run(self.uncovered() or list(range(1, len(needs.items) + 1)))
        got = self.audit_call()
        self.take_review(got)
        named = self.audit_fits(got)
        missing = [n for n in range(1, len(needs.items) + 1) if not named.get(str(n))]
        if missing and got:
            proposals: dict[int, list[str]] = {}
            for entry in as_list(got.get('search')):
                if isinstance(entry, dict) and str(entry.get('query') or '').strip():
                    number = min(max(1, whole_number(entry.get('item')) or 1), len(needs.items))
                    proposals.setdefault(number, []).append(str(entry['query']).strip())
            rows = [str(i) for i in as_list(got.get('open')) if str(i) in book.rows and str(i) not in book.full and not book.sold_out(str(i))]
            opened = e.opener.read(rows[:3 * (e.opener.compare_room() if e.opener.compare_useful() else 1)], e.tail() + 5)
            again = e.hunter.run(missing, proposals, 6) if proposals else []
            if opened or again:
                got = self.audit_call()
                self.take_review(got)
                named = self.audit_fits(got) or named
        for number in missing:
            by_code = [i for i in e.hunter.fits_by_code(number) if i not in named.get(str(number), [])]
            if by_code and not named.get(str(number)):
                named[str(number)] = by_code[:3]
        e.trace.append({'kind': 'decline_review', 'hunted': found[:8], 'named': named})
        if not named:
            return ''
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        reason = f" ({got.get('reason')})" if got and got.get('reason') else ''
        return 'A careful review thinks these listings may fit: ' + '; '.join(f"item {k}: {', '.join(v)}" for k, v in named.items()) + f'{reason}. Open them and buy what truly fits; end without a purchase only if each one fails.'

    def audit_call(self) -> dict[str, Any] | None:
        e, needs, book = self.e, self.e.needs, self.e.book
        wanted = [i for i in reversed(book.order) if not book.sold_out(i) and needs.within_budget(i) and (not book.label(i) or needs.owner(book.label(i)))]
        pool = wanted or [i for i in reversed(book.order) if not book.sold_out(i)]
        opened = ['OPENED ' + jdump(clipped(book.full[i], 200))[:1000] for i in pool if i in book.full][:20]
        rows = ['ROW ' + jdump(clipped(book.rows[i], 120))[:500] for i in pool if i not in book.full][:15]
        if not opened and not rows:
            return None
        prompt = f'{AUDIT}\n\nREQUEST:\n{e.request}\n\nREQUIREMENTS IN FORCE:\n{e.ledger()}'
        answers = [line for line in e.talk.record if 'message to the shopper' in line]
        if answers:
            prompt += '\n\nANSWERS FROM THE SHOPPER:\n' + '\n'.join(answers)
        if e.hunter.tried:
            prompt += '\n\nSEARCHES ALREADY TRIED:\n' + ' | '.join(sorted(e.hunter.tried))
        prompt += '\n\nLISTING RECORDS SEEN (newest first):\n' + '\n'.join(opened + rows)
        got = e.wire.ask_json(prompt, e.wire.judges(), 4000)
        e.trace.append({'kind': 'audit', 'got': got})
        return got

    def take_review(self, got: dict[str, Any] | None) -> None:
        """Story set aside and values corrected by the review; a stated answer may be corrected but never set aside."""
        needs = self.e.needs
        pointers = {need for _, need in needs.refs}
        changed = []
        aside: set[int] = set()
        for entry in as_list((got or {}).get('not_firm')):
            number, need = (whole_number(entry.get('item')) or 0, str(entry.get('need') or '').strip()) if isinstance(entry, dict) else (0, '')
            if 1 <= number <= len(needs.items) and number not in aside and need in needs.items[number - 1].live() and need in needs.items[number - 1].musts and need not in pointers:
                aside.add(number)
                needs.items[number - 1].dropped.append(need)
                changed.append(f'set aside {need}')
        for entry in as_list((got or {}).get('corrected')):
            number, need, value = (whole_number(entry.get('item')) or 0, str(entry.get('need') or '').strip(), str(entry.get('value') or '').strip()) if isinstance(entry, dict) else (0, '', '')
            if 1 <= number <= len(needs.items) and need in needs.items[number - 1].live() and need not in pointers and ':' in value:
                needs.items[number - 1].dropped.append(need)
                needs.items[number - 1].told.append(value)
                changed.append(f'{need} read as {value}')
        if changed:
            needs.version += 1
            needs.notes.append('a review of the request: ' + '; '.join(changed))
            self.e.trace.append({'kind': 'review', 'changed': changed})

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get('fits')
        if not isinstance(fits, dict):
            return {}
        book = self.e.book
        named = {str(k): [str(i) for i in as_list(v) if str(i) in book.rows and not book.sold_out(str(i))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def before_decline(self) -> str:
        """A text holds the decline and says why; an empty text lets it go."""
        e = self.e
        left = e.store.left()
        if left > 5:
            review = self.review_decline()
            if review:
                return review
            if self.leads and left > 4 and self.blocks == 0:
                self.blocks = 1
                return 'The review found listings that appear to fit: ' + ', '.join(self.leads) + '. Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again.'
            if self.leads and left > 4 and self.blocks == 1:
                self.blocks = 2
                first = self.leads[0]
                self.check([first], SPEND_HARD)
                if self.verdicts.get(self.check_key([first])) and first in self.leads:
                    self.confirmed = first
                    e.trace.append({'kind': 'refuse_blocked', 'confirmed': first})
                    return f"An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."
        if left <= 5 and not self.late_held:
            lines = [i for i in e.cart.built if not e.book.sold_out(i)]
            missing = [i for i in lines if i not in e.cart.ids() or (i in e.cart.dropped and not e.cart.fresh_view())]
            if len(lines) >= len(e.needs.items) and len(missing) + 1 <= left:
                self.late_held = True
                redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else 'Check out the cart as it is.'
                return f'Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. {redo} End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again.'
        if not self.leads and not self.pushed and left > e.store.max_steps // 3 and e.clock.may_explore():
            self.pushed = True
            return f"{left} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."
        return ''

    def last_try(self) -> str:
        """An order was tried and refused, so the model thought something fits; a decline now is rarely right. When an opened
        listing's record shows every requirement for each item, the cart gets it instead. A decline that still follows,
        with that cart holding lines, becomes the order."""
        e, needs, book = self.e, self.e.needs, self.e.book
        if self.buys == 0 or e.store.left() < 2:
            return ''
        if self.last_tried:
            if not e.cart.ids() or not e.roles.get('buy'):
                return ''
            reply = e.use(e.roles['buy'], dict(needs.brief.get('buy_args') or {}))
            e.trace.append({'kind': 'last_try_order', 'reply': clipped(reply.raw if reply else None, 120)})
            if reply is None:
                return ''
            self.buys += 1
            return f'Not run (the desk ordered instead): the cart holds lines that show every requirement. Store reply: {e.show(reply)}' + (' [episode finished]' if e.store.finished else '')
        have = {needs.owner(book.label(i)) for i in e.cart.ids() if not book.sold_out(i)}
        picks = {}
        for number in range(1, len(needs.items) + 1):
            found = e.hunter.fits_by_code(number)
            if number not in have and not found:
                return ''
            if number not in have:
                picks[number] = found[0]
        self.last_tried = True
        done = self.swap_lines([], list(picks.values())) if picks else 'nothing'
        e.trace.append({'kind': 'last_try', 'picks': picks})
        return f'Not run yet (no step used). An order was tried and refused, and the records show listings that fit, so a decline is not right yet. The desk changed the cart: {done}. Cart now: {e.cart.text()}. Check out.'

    def complete_cart(self) -> str:
        """The independent check confirmed a listing and the model still wants to decline: buy the confirmed plan."""
        e, book, needs = self.e, self.e.book, self.e.needs
        cart_tool, buy = e.roles.get('cart'), e.roles.get('buy')
        if self.completed or not self.confirmed or book.sold_out(self.confirmed) or not cart_tool or not buy:
            return ''
        self.completed = True
        plan: dict[str, int] = {}
        for number, ids in self.lead_items.items():
            pick = next((i for i in ids if not book.sold_out(i)), None)
            index = whole_number(number)
            if pick and index and 1 <= index <= len(needs.items) and book.label(pick) not in {book.label(p) for p in plan}:
                plan[pick] = needs.items[index - 1].qty
        if self.confirmed not in plan:
            return ''
        now = e.cart.ids()
        keep = {i: e.cart.lines.get(i, 1) for i in now if i not in plan and not book.sold_out(i) and book.label(i) not in {book.label(p) for p in plan}}
        drop = [i for i in now if i not in plan and i not in keep]
        if len(plan) + len({book.label(i) for i in keep}) < len(needs.items):
            return ''
        wanted = {**keep, **plan}
        calls = e.cart.calls_for(wanted, drop)
        if not calls or len(calls) + 1 > e.store.left():
            return ''
        e.opener.read(list(wanted), len(calls) + 3)
        self.check(list(wanted), SPEND_HARD)
        if not self.verdicts.get(self.check_key(list(wanted))):
            return ''
        done = []
        for args in calls:
            reply = e.use(cart_tool, args)
            if reply is None or e.store.finished:
                return ''
            e.cart.note(cart_tool, args, reply.body)
            done.append(jdump(args))
        replies = []
        for attempt in range(2):
            if attempt:
                self.reread_offers()
                self.check(list(wanted), SPEND_HARD)
                if not self.verdicts.get(self.check_key(list(wanted))) or e.store.left() < 1 or e.store.finished or any(book.sold_out(i) for i in wanted):
                    break
            go, said = self.gate('order')
            replies += [said] if said else []
            if not go:
                break
            reply = e.use(buy, dict(needs.brief.get('buy_args') or {}))
            e.trace.append({'kind': 'completed', 'lines': wanted, 'reply': clipped(reply.raw if reply else None, 200)})
            if reply is None:
                break
            self.buys += 1
            if not reply.finished:
                self.refused_at = e.store.step
            replies.append(e.show(reply))
            if e.store.finished:
                break
        if not replies:
            return ''
        return f"Not run (the desk acted instead): the independent check confirmed {self.confirmed}, so the desk set the cart ({'; '.join(done)}) and checked out. Store reply: " + ' | then: '.join(replies)


class Errand:
    """One episode: reads the request, lets the light model walk the store, and answers each of its tool calls."""

    def __init__(self, problem: dict[str, Any]) -> None:
        environment = problem['environment']
        view = environment['policy_view']
        self.view = view
        self.request = str(view.get('query') or '')
        self.trace: list[dict[str, Any]] = []
        self.clock = Clock()
        self.wire = Wire(self.clock, self.trace)
        self.store = Store(dict(environment['binding']), view, self.clock, self.trace)
        self.book = Book(self.trace)
        self.needs = Needs(self.book)
        self.talk = Interviewer(self)
        self.opener = Opener(self)
        self.hunter = Hunter(self)
        self.cart = Cart(self)
        self.desk = Desk(self)
        self.roles: dict[str, str] = {}
        self.chat: list[dict[str, Any]] = []
        self.loop_calls = 0
        self.misses = 0
        self.heard: set[str] = set()
        self.queued: list[str] = []
        self.recent: list[str] = []
        self.id_args: set[tuple[str, str]] = set()
        self.read_cache: dict[str, tuple[int, int, str]] = {}
        self.view_keys: dict[str, tuple[int, set[str]]] = {}
        self.tool_keys: dict[str, set[str]] = {}
        self.turn_size = 1

    # the request

    def read_request(self) -> None:
        tools = '\n'.join(f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {jdump(t['function'].get('parameters') or {})}" for t in self.store.tools)
        prompt = f'{BRIEF}\nTOOLS:\n{tools}\n\nREQUEST:\n{self.request}'
        if self.view.get('opening') is not None:
            prompt += '\n\nOPENING NOTE FROM THE STORE:\n' + jdump(clipped(self.view.get('opening'), 1500))
        usable = lambda b: bool(as_list(b.get('items'))) and isinstance(b.get('roles'), dict) and bool(b['roles'].get('buy'))
        brief = self.wire.ask_json(prompt, [LIGHT], 4000) or {}
        if not self.wire.no_credit and not usable(brief):
            brief = self.wire.ask_json(prompt, [HEAVY, LIGHT], 2500) or brief
        elif not self.wire.no_credit and self.shelf_shared(brief):
            again = self.wire.ask_json(prompt, [HEAVY], 2500) or {}
            brief = again if usable(again) else brief
        self.needs.brief = brief
        roles = brief.get('roles') if isinstance(brief.get('roles'), dict) else {}
        self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.store.schemas}
        self.hunter.read_browse(brief.get('browse'))
        self.settle_endings(brief)
        self.read_items(brief)
        self.read_limits(brief)
        self.talk.set_up(brief)
        self.book.enums = self.schema_words()

    @staticmethod
    def shelf_shared(brief: dict[str, Any]) -> bool:
        shelves = [spaced(str(e.get('shelf') or '')) for e in as_list(brief.get('items')) if isinstance(e, dict)]
        shelves = [s for s in shelves if s]
        return len(shelves) != len(set(shelves))

    def settle_endings(self, brief: dict[str, Any]) -> None:
        for key in ('buy_args', 'refuse_args'):
            brief[key] = brief[key] if isinstance(brief.get(key), dict) else {}
        buy, refuse = self.roles.get('buy'), self.roles.get('refuse')
        if buy and buy == refuse and (not brief['buy_args'] or not brief['refuse_args'] or brief['buy_args'] == brief['refuse_args']):
            ask = f'The store tool {buy} both checks out and ends without a purchase, depending on its arguments. Its schema: {jdump(self.store.schemas.get(buy))}\nReturn ONE JSON object and nothing else: {{"buy_args": the fixed arguments that check out, "refuse_args": the fixed arguments that end without a purchase}} (only the arguments that choose the mode).'
            fixed = self.wire.ask_json(ask, [LIGHT, HEAVY], 1500) or {}
            if isinstance(fixed.get('buy_args'), dict) and isinstance(fixed.get('refuse_args'), dict) and fixed['buy_args'] != fixed['refuse_args']:
                brief['buy_args'], brief['refuse_args'] = fixed['buy_args'], fixed['refuse_args']

    def read_items(self, brief: dict[str, Any]) -> None:
        needs = self.needs
        merged_at: dict[int, int] = {}
        for old, entry in enumerate([x for x in as_list(brief.get('items')) if isinstance(x, dict)], 1):
            quantity = whole_number(entry.get('quantity'))
            item = Item(str(entry.get('name') or 'the product'), str(entry.get('shelf') or ''), quantity if quantity and quantity > 0 else 1, [str(m) for m in as_list(entry.get('musts')) if m])
            same = next((m for m in needs.items if item.shelf.strip() and m.qty == item.qty and spaced(m.shelf) == spaced(item.shelf) and not self.disagree(m, item)), None)
            if same is None:
                needs.items.append(item)
                merged_at[old] = len(needs.items)
            else:
                merged_at[old] = needs.items.index(same) + 1
                same.musts += [m for m in item.musts if m not in same.musts]
                self.trace.append({'kind': 'items_merged', 'into': same.name, 'dropped': item.name})
        if not needs.items:
            needs.items = [Item('the requested product', '', 1, [])]
        for pair in as_list(brief.get('fits')):
            if isinstance(pair, dict):
                part, main = merged_at.get(whole_number(pair.get('item')) or 0), merged_at.get(whole_number(pair.get('with')) or 0)
                if part and main and part != main and (part, main) not in needs.bound:
                    needs.bound.append((part, main))
        for lookup in as_list(brief.get('lookups')):
            text = str(lookup).strip()
            if not any(text in item.musts for item in needs.items):
                key = spaced(text.split(':')[0])
                text = next((m for item in needs.items for m in item.musts if spaced(m.split(':')[0]) == key), text)
            number = next((n for n, item in enumerate(needs.items, 1) if text in item.musts), 1)
            if text and (number, text) not in needs.refs:
                needs.refs.append((number, text))
        self.unmix()

    @staticmethod
    def disagree(a: Item, b: Item) -> bool:
        split = lambda need: (spaced(need.split(':')[0]), spaced(need.partition(':')[2]))
        return any(split(x)[0] == split(y)[0] and split(x)[1] != split(y)[1] for x in a.musts for y in b.musts)

    def unmix(self) -> None:
        """Words describing a pointed-at listing that the brief took for requirements on the product go back into the pointer."""
        needs = self.needs
        for number, item in enumerate(needs.items, 1):
            pointers = [need for n, need in needs.refs if n == number and need in item.musts]
            rest = [m for m in item.musts if m not in pointers]
            if not pointers or not rest:
                continue
            prompt = f'{UNMIX}\n\nREQUEST:\n{self.request}\n\nPRODUCT: {item.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n' + '\n'.join(pointers) + '\nOTHER REQUIREMENTS:\n' + '\n'.join(rest)
            got = self.wire.ask_json(prompt, [HEAVY, LIGHT], 1500) or {}
            moved = [m for m in rest if m in [str(x).strip() for x in as_list(got.get('describe_other'))]]
            if moved:
                joined = f"{pointers[0]} (that listing: {'; '.join(moved)})"
                item.musts = [joined if m == pointers[0] else m for m in item.musts if m not in moved]
                needs.refs = [(n, joined if (n, need) == (number, pointers[0]) else need) for n, need in needs.refs]
                self.trace.append({'kind': 'unmix', 'item': number, 'moved': moved})

    def read_limits(self, brief: dict[str, Any]) -> None:
        budget = amount(brief.get('budget'))
        self.needs.budget = budget if budget and budget > 0 else None
        steps = whole_number(brief.get('steps'))
        if steps is not None and 10 <= steps < self.store.max_steps:
            self.store.max_steps = steps
        for entry in as_list(brief.get('process')):
            if isinstance(entry, dict) and str(entry.get('step') or '').strip():
                when = str(entry.get('when') or '').strip().lower()
                self.needs.process.append({'step': str(entry['step']).strip(), 'when': when if when in ('choose', 'order', 'end') else 'order', 'messages': max(0, whole_number(entry.get('messages')) or 0)})

    def schema_words(self) -> set[str]:
        found: set[str] = set()
        stack: list[Any] = list(self.store.schemas.values())
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                found |= {o.strip().lower() for o in node.get('enum') or [] if isinstance(o, str)}
                stack += [v for v in node.values() if isinstance(v, dict)]
        return found

    # small shared helpers

    def is_buy(self, name: str, args: dict[str, Any]) -> bool:
        fixed = self.needs.brief.get('buy_args') or {}
        return name == self.roles.get('buy') and all(args.get(k) == v for k, v in fixed.items())

    def is_refuse(self, name: str, args: dict[str, Any]) -> bool:
        fixed = self.needs.brief.get('refuse_args') or {}
        return name == self.roles.get('refuse') and all(args.get(k) == v for k, v in fixed.items())

    def page_tools(self) -> set[str]:
        page = self.roles.get('offer') or self.roles.get('open')
        return {t for t in (page, self.roles.get('open'), self.roles.get('compare')) if t}

    def offer_tool(self) -> str | None:
        return self.roles.get('offer') or self.roles.get('open')

    def tail(self) -> int:
        """Steps kept for the end: the order, its approval message when one is due, and one more until a first refusal."""
        talking = any(p['when'] == 'order' and p['messages'] > 0 for p in self.needs.process)
        return (1 if self.desk.buys == 0 else 0) + (2 if talking and self.talk.used < self.talk.limit and self.desk.gate_due('order') else 1)

    def spare(self) -> int:
        """Steps exploring leaves alone. A first order is often refused with a note that the pick is gone, and fixing that
        takes a look, a removal, a new line and a second order for each item."""
        return self.tail() + (len(self.needs.items) + 5 if self.desk.buys == 0 else 0)

    def room_after_refusal(self) -> int:
        """Steps a refused first order needs: a removal and a new line per item, then the second order and one to spare."""
        return 2 * len(self.needs.items) + 2 if self.desk.buys == 0 else 0

    def chores_fit(self, count: int = 1) -> bool:
        """Whether reads before an order still leave the order and what a refusal would need."""
        return self.store.left() > self.tail() + self.room_after_refusal() + count

    def ledger(self) -> str:
        tool = self.opener.single_tool()
        line = ''
        if tool and (self.opener.facts_from.get(tool) is True or self.roles.get('offer')):
            bare = sorted(t for t, ok in self.opener.facts_from.items() if not ok)
            line = f'listing attributes come from {tool}' + (f"; {', '.join(bare)} show none" if bare else '')
        return self.needs.ledger(line)

    def use(self, name: str, args: dict[str, Any], keep: int = 0) -> Reply | None:
        """A store call made by the desk itself, read the same way as the model's, unless too little room is left for it."""
        if self.store.finished or self.store.left() <= keep or not self.clock.affords('store'):
            return None
        fitted = self.store.fitted(name, args)
        if fitted is None:
            return None
        reply = self.store.call(name, fitted)
        if reply is None:
            return None
        self.take_in(name, fitted, reply)
        return reply

    def take_in(self, name: str, args: dict[str, Any], reply: Reply) -> None:
        self.book.absorb(reply.body)
        self.opener.learn(name, reply)
        self.note_view(name, reply.body)
        if name == self.roles.get('search'):
            self.hunter.note_search(args)
            self.hunter.note_result(name, args, reply.body)
        self.heed(reply.remarks + self.loose_lines(reply.body), name)

    def heed(self, lines: list[str], source: str) -> str:
        """New sentences from the store or the shopper are read once; a change is queued for the model's next look."""
        fresh = [line for line in dict.fromkeys(lines) if line not in self.heard]
        if not fresh:
            return ''
        meaning = self.take_note(fresh)
        if meaning:
            self.queued.append(f'(seen on {source}) {meaning}')
        return meaning

    def drain_notes(self) -> str:
        if not self.queued:
            return ''
        text = '\n[notes seen meanwhile] ' + ' | '.join(self.queued)
        self.queued = []
        return text

    def loose_lines(self, body: Any) -> list[str]:
        lines: list[str] = []
        if isinstance(body, dict) and not self.book.listing_id(body):
            for value in body.values():
                if isinstance(value, str) and len(value.split()) >= 5:
                    lines.append(value)
                elif isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                    lines += [v for v in value if len(v.split()) >= 5]
                elif isinstance(value, dict) and not self.book.listing_id(value):
                    lines += [v for v in value.values() if isinstance(v, str) and len(v.split()) >= 6]
        return lines

    def take_note(self, lines: list[str]) -> str:
        needs, desk = self.needs, self.desk
        self.heard.update(lines)
        prompt = f'{NOTE}\n\nITEMS AND REQUIREMENTS NOW:\n{self.ledger()}\n\nTHE NOTE:\n' + '\n'.join(lines)
        got = self.wire.ask_json(prompt, [LIGHT, SPARE], 2500) or {}
        gone = [str(g) for g in as_list(got.get('gone')) if isinstance(g, (str, int))]
        self.book.unavailable.update(g for g in gone if g in self.book.rows)
        desk.note_sold = bool(gone) or got.get('sold_out') is True
        desk.note_named = any(g in self.book.rows for g in gone)
        if desk.note_sold and len(lines) == 1:
            desk.sold_lines.update(lines)
        if got.get('kind') not in (None, 'change'):
            return ''
        changed = []
        budget = amount(got.get('budget'))
        if budget and budget > 0 and budget != needs.budget:
            needs.budget = budget
            changed.append(f'budget now {budget:g}')
        for entry in [x for x in as_list(got.get('drop')) if isinstance(x, dict) and str(x.get('need') or '').strip()]:
            item = needs.item_of(entry.get('item'))
            target = str(entry['need']).strip()
            hits = [m for m in item.live() if m == target] or [m for m in item.live() if spaced(m.split(':')[0]) == spaced(target.split(':')[0])][:1]
            item.dropped.extend(hits or [target])
            changed.append(f'dropped {target}')
        for entry in [x for x in as_list(got.get('add')) if isinstance(x, dict) and str(x.get('need') or '').strip()]:
            item = needs.item_of(entry.get('item'))
            need = str(entry['need']).strip()
            if need in item.dropped:
                item.dropped.remove(need)
            if need not in item.musts + item.told:
                item.told.append(need)
            changed.append(f'added {need}')
        if gone:
            changed.append('unavailable: ' + ', '.join(gone))
        if got.get('price_moved') is True:
            changed.append('a price changed')
            self.desk.offer_seen.clear()
        summary = str(got.get('summary') or '').strip()
        if changed or summary:
            needs.version += 1
            self.recent = []
            needs.notes.append(summary or '; '.join(changed))
        return (summary + (' (' + '; '.join(changed) + ')' if changed else '')).strip()

    def show(self, reply: Reply) -> str:
        text = jdump(clipped(reply.body, SHOW_CHARS))
        text = text if len(text) <= OBS_LIMIT else text[:OBS_LIMIT] + '...(cut)'
        return text + (' [also said] ' + ' | '.join(reply.remarks) if reply.remarks else '')

    def note_view(self, name: str, body: Any) -> None:
        if name not in {self.roles.get(r) for r in ('open', 'offer', 'compare')}:
            return
        for ident, record in self.book.records_in(body):
            self.tool_keys.setdefault(name, set()).update(record)
            version, have = self.view_keys.get(ident, (-1, set()))
            self.view_keys[ident] = (self.needs.version, (have if version == self.needs.version else set()) | set(record))

    def shown_already(self, name: str, ident: str) -> bool:
        version, have = self.view_keys.get(ident, (-1, set()))
        keys = self.tool_keys.get(name)
        return bool(keys) and version == self.needs.version and keys <= have and ident not in self.book.gone

    def status(self) -> str:
        left, desk, needs = self.store.left(), self.desk, self.needs
        text = f'[desk] step {self.store.step} of {self.store.max_steps} used, {left} left.'
        if left <= self.tail() + 1:
            text += ' FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that).'
        elif left <= self.spare():
            text += ' Few steps left: set the cart to the chosen lines and check out now; a first order is often refused, and the steps kept here are for that.'
        elif desk.buys == 0 and left <= self.spare() + 6:
            text += ' Steps are running low: settle on your picks, fill the cart and check out within the next few steps.'
        elif desk.buys == 0 and not self.cart.touched and self.store.step >= self.store.max_steps // 2:
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        live = {need for _, need in needs.now()}
        lookups = [need for _, need in needs.refs if need in live and need not in needs.ref_values]
        if lookups and desk.buys == 0:
            text += '\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] ' + ' | '.join(lookups[:3])
        if desk.leads:
            text += '\n[review candidates] ' + ', '.join(desk.leads)
        if needs.notes:
            text += '\n[changes so far] ' + ' | '.join(needs.notes[-4:])
        if self.clock.left() < EXPLORE_MARGIN + 45:
            text += f'\n[time] about {max(0, int(self.clock.left()))} s of this episode are left: settle on your picks and check out within the next few calls.'
        return text + '\n[ledger]\n' + self.ledger()

    def offered(self) -> list[dict[str, Any]]:
        """The tools the model may call now: no message tool ever, and only the ending ones when steps run out."""
        left = self.store.left()
        ends = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        keep: set[Any] | None = None
        if left <= self.tail():
            keep = set(ends)
        elif left <= self.tail() + 3:
            keep = ends | {self.roles.get('cart'), self.roles.get('cart_view')}
        elif left <= self.spare():
            keep = ends | {self.roles.get('cart'), self.roles.get('cart_view')}
        tools = [t for t in self.store.tools if t['function']['name'] != self.talk.tool]
        if self.cart.refused_moves >= 4 and self.cart.lines:
            keep = set(ends)
        if keep is not None and ends:
            return [t for t in tools if t['function']['name'] in keep] or tools
        return tools

    def quiet_args(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """A decline's free text says nothing fits and names no listing, unless the request wants alternatives named."""
        schema = self.store.schemas.get(name) or {}
        out = dict(args)
        for key, spec in (schema.get('properties') or {}).items():
            if not isinstance(spec, dict) or 'string' not in type_names(spec) or 'enum' in spec:
                continue
            value = out.get(key)
            if value is None and key not in (schema.get('required') or []):
                continue
            text = str(value or '')
            if not text or any(i in text for i in self.book.rows) or self.needs.brief.get('quiet_decline') is not False:
                out[key] = 'Sorry, nothing in the store meets all of your requirements, so I am not buying anything.'[:spec.get('maxLength', 200)]
        return out

    def real_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """Where the model passed a title instead of an id, the id when the title is unmistakable; else what is wrong."""
        book = self.book
        if name not in {self.roles.get(r) for r in ('cart', 'open', 'offer', 'compare')} or not book.id_key:
            return args, ''
        props = (self.store.schemas.get(name) or {}).get('properties') or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and ('string' in type_names(v) or 'array' in type_names(v)) and 'enum' not in v]
        if len(free) != 1 or free[0] not in args:
            return args, ''

        def fix(value: Any) -> tuple[Any, str]:
            if not isinstance(value, str) or value.strip() in book.rows or looks_like_id(value, True) or len(value.strip()) < 8:
                return (value.strip() if isinstance(value, str) else value), ''
            low = value.strip().lower()
            hits = [i for i, row in book.rows.items() if any(k != book.id_key and isinstance(v, str) and v.strip().lower().startswith(low) for k, v in row.items())]
            if len(hits) == 1:
                return hits[0], ''
            if hits:
                return value, f"several listings carry the text {value!r} ({', '.join(hits[:6])}); pass one listing id"
            return value, f'{value!r} is not a listing id; pass the id from the result rows (their {book.id_key} field)'
        given = args[free[0]]
        fixed = [fix(v) for v in given] if isinstance(given, list) else [fix(given)]
        problem = next((p for _, p in fixed if p), '')
        if problem:
            return args, problem
        return {**args, free[0]: [v for v, _ in fixed] if isinstance(given, list) else fixed[0][0]}, ''

    # the model's tool calls

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Several single opens in one turn become one side-by-side call where that view shows attributes."""
        opener = self.roles.get('open')
        joiner, field, target = self.roles.get('compare'), self.opener.compare_field(), lone_target(self.store.schemas.get(opener or '') or {})
        if not opener or not joiner or not field or not target or opener == joiner or not self.opener.compare_useful():
            return calls
        picks: list[tuple[int, str]] = []
        for index, call in enumerate(calls):
            fn = call['function']
            args = json_object(fn.get('arguments'))
            ident = str((args or {}).get(target) or '').strip()
            if fn.get('name') == opener and args and set(args) == {target} and ident in self.book.rows and ident not in self.book.gone:
                picks.append((index, ident))
        picks = picks[:self.opener.compare_room()]
        if len(picks) < 2:
            return calls
        joined = {i for i, _ in picks}
        out = []
        for index, call in enumerate(calls):
            if index == picks[0][0]:
                out.append({'id': call['id'], 'type': 'function', 'function': {'name': joiner, 'arguments': jdump({field: [i for _, i in picks]})}})
            elif index in joined:
                out.append({'id': call['id'], 'joined': True})
            else:
                out.append(call)
        return out

    def refused_before_store(self, name: str, raw: Any) -> tuple[str, dict[str, Any]]:
        """A reason this call is not run at all (no step used), or the fitted arguments."""
        allowed = {t['function']['name'] for t in self.offered()}
        if self.store.finished:
            return 'Not run: the episode is over.', {}
        if name == self.talk.tool:
            return "Not run (no step used): the desk's interviewer handles every message to the shopper.", {}
        if name not in self.store.schemas:
            return f"Not run (no step used): unknown tool. Tools: {', '.join(sorted(allowed))}.", {}
        if name not in allowed:
            return 'Not run (no step used): too few steps are left for that; use: ' + ', '.join(sorted(allowed)) + '.', {}
        args = raw if isinstance(raw, dict) else (json_object(raw) if str(raw or '').strip() else {})
        if args is None:
            return 'Not run (no step used): the arguments were not one JSON object.', {}
        faults: list[str] = []
        args = conform(args, self.store.schemas[name], 'args', faults)
        if faults:
            return 'Not run (no step used): ' + '; '.join(faults[:6]), {}
        args, wrong = self.real_ids(name, args)
        if wrong:
            return f'Not run (no step used): {wrong}.', {}
        return '', args

    def no_need_to_run(self, name: str, args: dict[str, Any], ending: bool) -> str:
        """A reason a well-formed call would waste its step."""
        book, roles = self.book, self.roles
        target = lone_target(self.store.schemas.get(name) or {})
        readers = {roles.get(r) for r in ('search', 'open', 'offer', 'compare')} - {None}
        if target and name in (roles.get('open'), roles.get('offer')) and str(args.get(target)) in book.gone:
            return f'Not run (no step used): {args[target]} sold out at step {book.gone[str(args[target])]} and will not come back. Replace its cart line with another fitting listing.'
        if name == roles.get('cart'):
            if self.cart.turn_stopped:
                return f'Not run (no step used): the cart call before this one in your turn was refused, so this one was not sent either. Cart now: {self.cart.text()}'
            stop = self.cart.refusal(name, args)
            if stop:
                self.trace.append({'kind': 'cart_refused', 'args': clipped(args, 80), 'why': stop[:160]})
                return stop
        if name == roles.get('cart') and book.gone and not self.takes_off(name, args):
            back = [i for i in book.gone if i in jdump(args)]
            if back:
                return f'Not run (no step used): {back[0]} sold out at step {book.gone[back[0]]} and will not come back; choose another listing for that item. Cart now: {self.cart.text()}'
        if name == roles.get('cart') and any((whole_number(v) or 0) < 0 for v in args.values() if not isinstance(v, str)):
            absent = [v for v in args.values() if isinstance(v, str) and v in book.rows and v not in self.cart.lines and v not in self.cart.last_said]
            if absent:
                return f'Not run (no step used): {absent[0]} is not in the cart, so there is nothing to take off. Cart now: {self.cart.text()}'
        if ending and self.turn_size > 1:
            return 'Not run (no step used): send the finishing call on its own, after reading your other results.'
        if not ending and self.store.left() <= self.tail() and (roles.get('buy') or roles.get('refuse')):
            return "Not run (no step used): the last steps are kept for checking out (with the shopper's approval of the basket when it is due) or ending without a purchase."
        mark = name + jdump(args)
        if not ending and name != roles.get('cart') and mark in self.recent:
            return 'Not run (no step used): you just made this same call and nothing changed; its result is above.'
        cached = self.read_cache.get(mark)
        if name in readers and cached and cached[0] == self.needs.version:
            return f'Not run (no step used): the same call at step {cached[1]} returned this, and nothing has changed since:\n' + cached[2][:3000]
        if target and name in readers and name != roles.get('compare') and self.shown_already(name, str(args.get(target))):
            return f'Not run (no step used): {args[target]} has shown everything this view gives since the last change. Its record: {book.row_text(str(args[target]))}'
        return ''

    def takes_off(self, name: str, args: dict[str, Any]) -> bool:
        """Whether a cart call removes its line: its mode says so, or its count is zero or less."""
        props = (self.store.schemas.get(name) or {}).get('properties') or {}
        for key, value in args.items():
            options = (props.get(key) or {}).get('enum')
            if isinstance(options, list) and len(options) >= 2 and value == Cart.mode_word(options, True):
                return True
        return any((whole_number(v) if whole_number(v) is not None else 1) <= 0 for v in args.values() if not isinstance(v, (str, list, dict)))

    def before_ending(self, name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
        """What must happen before an ending goes out: a reason to hold it, the arguments, and a line for the model."""
        desk = self.desk
        said = ''
        if self.is_buy(name, args) and not self.cart.ids() and not self.cart.unread and desk.line_holds < 4:
            desk.line_holds += 1
            return 'Not run (no step used): the cart is empty, so an order cannot go through. Put a line in it for each item first.', args, ''
        extra = self.cart.spare_lines() if self.is_buy(name, args) else []
        if extra and self.store.left() >= 2 and desk.line_holds >= 2 and desk.swap_lines(extra, []):
            return f"Not run yet (no step used). An item had two lines, and the holds on that were used up; the desk took out {', '.join(extra)}. Cart now: {self.cart.text()}", args, ''
        gone = [i for i in self.cart.ids() if i in self.book.gone] if self.is_buy(name, args) else []
        if gone and self.store.left() >= 2 and desk.swap_lines(gone, []):
            return f"Not run yet (no step used). {', '.join(gone)} sold out and was still in the cart; the desk took it out. Cart now: {self.cart.text()}", args, ''
        if self.store.left() > 2 and self.clock.affords(LIGHT, 'store'):
            hold = desk.before_order() if self.is_buy(name, args) else desk.before_decline()
            if hold:
                return 'Not run yet (no step used). ' + hold, args, ''
        if self.is_refuse(name, args):
            hold = desk.last_try()
            if hold:
                return hold, args, ''
            done = desk.complete_cart()
            if done:
                return done + (' [episode finished]' if self.store.finished else ''), args, ''
            args = self.quiet_args(name, args)
        if self.store.left() >= 2 and self.clock.affords(LIGHT):
            go, said = desk.gate('order' if self.is_buy(name, args) else 'end')
            if not go:
                return f'Not run yet. {said}\nCart now: {self.cart.text()}', args, ''
        return '', args, f'\n[desk] {said}' if said else ''

    def run_one(self, call: dict[str, Any]) -> str:
        fn = call.get('function') or {}
        name = str(fn.get('name') or '')
        refusal, args = self.refused_before_store(name, fn.get('arguments'))
        if refusal:
            return refusal
        ending = self.is_buy(name, args) or self.is_refuse(name, args)
        refusal = self.no_need_to_run(name, args, ending)
        if refusal:
            return refusal
        extra = ''
        field = self.opener.compare_field() if name == self.roles.get('compare') else None
        if field and isinstance(args.get(field), list) and args[field]:
            room = self.opener.compare_room()
            if self.opener.compare_bad and len(args[field]) > room:
                extra += f"\n[desk] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[field][room:]))}."
                args = {**args, field: args[field][:room]}
            fresh = [i for i in args[field] if not self.shown_already(name, str(i))]
            if not fresh:
                return 'Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n' + '\n'.join(f'{i}: {self.book.row_text(str(i))}' for i in args[field])[:4000]
            if len(fresh) < len(args[field]):
                extra += f"\n[desk] Left out {', '.join(str(i) for i in args[field] if i not in fresh)}: shown in full already, nothing changed since."
                args = {**args, field: fresh}
        if ending:
            hold, args, line = self.before_ending(name, args)
            if hold:
                return hold
            extra = line + extra
        reply = self.store.call(name, args)
        if reply is None:
            return 'The call failed (network or server). You may try again.'
        return self.after_call(name, args, reply, ending, field, extra)

    def after_call(self, name: str, args: dict[str, Any], reply: Reply, ending: bool, field: str | None, extra: str) -> str:
        roles, book, desk = self.roles, self.book, self.desk
        body = reply.body
        if field and isinstance(args.get(field), list) and len(args[field]) > 1:
            if any(str(i) in jdump(body) for i in args[field]):
                self.opener.compare_ok = max(self.opener.compare_ok, len(args[field]))
            else:
                self.opener.compare_bad = min(self.opener.compare_bad or 999, len(args[field]))
                extra += f'\n[desk] The side-by-side view showed nothing for {len(args[field])} listings: send at most {self.opener.compare_room()} per call.'
        mark = name + jdump(args)
        self.recent = (self.recent + [mark])[-3:]
        desk.note_sold = False
        self.take_in(name, args, reply)
        for key, value in args.items():
            values = value if isinstance(value, list) else [value]
            if values and all(isinstance(x, str) and x in book.rows for x in values):
                self.id_args.add((name, key))
        if name == self.offer_tool():
            desk.offer_seen.update((i, self.store.step) for i, _ in book.records_in(body))
        is_cart = name == roles.get('cart')
        if is_cart:
            self.cart.note(name, args, body)
            self.recent = []
        if name == roles.get('cart_view'):
            self.cart.view, self.cart.viewed_at = body, self.store.step
        buying = self.is_buy(name, args)
        if buying:
            desk.buys += 1
            desk.refused_at = self.store.step if not reply.finished else desk.refused_at
        text = self.show(reply) + extra
        if buying and not reply.finished:
            text = '[desk] The order did NOT go through: the episode goes on, whatever the result says. ' + text
        plain = {tool for job, tool in roles.items() if job != 'search'}
        if name not in plain and not ending:
            text += self.hunter.auto_pages(name, args, body) + self.cheapest_rows(body)
        base = text
        evidence = self.evidence_line(body)
        text += '\n' + evidence if evidence else ''
        if is_cart and any(whole_number(v) == 0 for v in args.values() if not isinstance(v, str)):
            if any(isinstance(v, str) and v in jdump(body) and v in book.rows for v in args.values()):
                text += '\n[desk] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'
        said = self.drain_notes()
        if said:
            text += said + ' Apply each change to your picks before checking out.'
        refused = buying and not reply.finished
        again = refused and any(line in reply.remarks for line in desk.sold_lines)
        vague = desk.note_sold and (refused or (not desk.note_named and not buying))
        if (vague or again) and self.cart.ids() and not self.store.finished:
            seen = desk.find_sold()
            if seen:
                text += f"\n[the desk re-read the cart's lines after the refusal]\n{seen}"
        desk.note_sold = False
        swap = desk.swap_hint()
        if swap:
            text += '\n' + swap
        text += self.drain_notes()
        if name in {roles.get(r) for r in ('search', 'open', 'offer', 'compare')}:
            self.read_cache[mark] = (self.needs.version, self.store.step, base)
        return text + (' [episode finished]' if reply.finished else '')

    def cheapest_rows(self, body: Any) -> str:
        if self.needs.pick() != 'cheapest' or not self.book.price_key:
            return ''
        rows = sorted((self.book.price(i), i) for i, _ in self.book.records_in(body) if self.book.buyable(i) and self.book.price(i) is not None)
        return '\n[desk] buyable rows here, cheapest first: ' + ', '.join(f'{i} ({p:g})' for p, i in rows[:6]) if len(rows) >= 2 else ''

    def evidence_line(self, body: Any) -> str:
        """After an opened listing, the keys whose names match a requirement, and how code reads them."""
        needs, book = self.needs, self.book
        ids = [i for i, r in book.records_in(body) if facts_block(r)]
        lines = []
        for ident in ids[:12]:
            parts = []
            for number, need in needs.now()[:8]:
                prop = need.split(':', 1)[0].strip()
                hits = [f'{k}={str(v)[:50]}' for k, v in book.facts(ident).items() if names_property(prop, k)]
                if hits:
                    reading = {'met': ' [states it]', 'broken': ' [says otherwise]'}.get(needs.need_state(ident, need, number), '')
                    parts.append((f'(item {number}) ' if len(needs.items) > 1 else '') + f'{prop} -> ' + ', '.join(hits[:3]) + reading)
            if parts:
                lines.append(f'{ident}: ' + '; '.join(parts))
        if not lines:
            return ''
        return "[evidence: keys whose names contain a requirement's property; one naming a different property (another part or aspect) decides nothing, a key with another name may still hold the same property, and spellings of one key add up: one stating the value is enough, marked [states it]] " + ' | '.join(lines)

    # the whole episode

    def trim_history(self) -> None:
        if sum(len(jdump(m.get('content'))) for m in self.chat) <= HISTORY_LIMIT:
            return
        tool_turns = [m for m in self.chat if m.get('role') == 'tool']
        for message in tool_turns[:max(1, len(tool_turns) // 2)]:
            content = message.get('content') or ''
            if len(content) > 600:
                ids = [i for i in dict.fromkeys(re.findall(r'"([A-Za-z0-9][A-Za-z0-9_\-:.]{5,40})"', content)) if i in self.book.rows][:20]
                message['content'] = content[:400] + f"...(older result shortened; listings in it: {', '.join(ids)})"

    def finish(self, again: bool = False) -> str:
        """End the episode with what there is: the owed messages first, then the cart if it holds anything, else a decline."""
        store, roles, cart, desk = self.store, self.roles, self.cart, self.desk
        if store.finished or store.broke or store.left() <= 0:
            return ''
        if self.talk.owed():
            summary = self.talk.run()
            if summary and again and not store.finished:
                self.chat.append({'role': 'user', 'content': summary})
        buy, refuse = roles.get('buy'), roles.get('refuse')
        holding = bool(cart.ids()) if cart.fresh_view() else bool(cart.lines)
        if buy and (holding or (cart.touched and not cart.fresh_view() and store.left() >= 2)):
            args = store.fitted(buy, dict(self.needs.brief.get('buy_args') or {}))
            if args is not None:
                if store.left() >= 2:
                    desk.gate('order')
                if store.finished or store.left() <= 0:
                    return ''
                reply = store.call(buy, args)
                self.trace.append({'kind': 'forced_buy', 'reply': clipped(reply.raw if reply else None, 200)})
                if store.finished or store.left() <= 0:
                    return ''
                if reply is not None and again:
                    desk.buys += 1
                    desk.refused_at = store.step
                    self.take_in(buy, args, reply)
                    return f'the checkout was refused: {self.show(reply)}'
        if refuse:
            args = store.fitted(refuse, self.quiet_args(refuse, dict(self.needs.brief.get('refuse_args') or {})))
            if args is not None:
                if store.left() >= 2:
                    desk.gate('end')
                if store.finished or store.left() <= 0:
                    return ''
                reply = store.call(refuse, args)
                self.trace.append({'kind': 'forced_refuse', 'reply': clipped(reply.raw if reply else None, 200)})
                if reply is not None and not store.finished:
                    self.take_in(refuse, args, reply)
                    return f'ending without a purchase was refused: {self.show(reply)}'
        return ''

    def run(self) -> list[dict[str, Any]]:
        if self.store.finished:
            return self.trace
        self.read_request()
        self.trace.append({'kind': 'brief', 'brief': self.needs.brief})
        opening = self.view.get('opening')
        intro = f"SHOPPER'S REQUEST:\n{self.request}"
        if opening is not None:
            first = Reply(opening, False)
            self.book.absorb(first.body)
            intro += f'\n\nWHAT THE STORE SHOWS FIRST:\n{self.show(first)}'
        intro += '\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n' + self.status()
        self.chat = [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': intro}]
        self.walk(LOOP_CALLS, SPEND_HARD)
        if not self.store.finished:
            turned = self.finish(again=True)
            if turned and not self.store.finished and not self.store.broke and not self.wire.no_credit and self.store.left() > 1 and self.clock.may_explore():
                self.chat.append({'role': 'user', 'content': f'[desk] The desk tried to end the episode and {turned}\nApply what the store says, then finish.\n' + self.status()})
                self.walk(self.loop_calls + 10, SPEND_HARD + 0.01)
            if not self.store.finished:
                self.finish()
        self.trace.append({'kind': 'end', 'finished': self.store.finished, 'steps': self.store.step, 'cost': round(self.wire.spent, 5), 'calls': self.wire.calls, 'messages': self.talk.used, 'seconds': int(EPISODE_SECONDS - self.clock.left())})
        return self.trace

    def walk(self, loop_cap: int, spend_cap: float) -> None:
        """The light model's turns, until it ends the episode or a limit says to stop exploring."""
        idle = 0
        store = self.store
        while not store.finished and not store.broke and store.left() > 0 and self.loop_calls < loop_cap and self.wire.spent < spend_cap and self.clock.may_explore():
            if self.talk.ready():
                self.chat.append({'role': 'user', 'content': (self.talk.run() or 'The interviewer had nothing to ask.') + '\n' + self.status()})
                continue
            self.trim_history()
            self.loop_calls += 1
            message = self.wire.reply(self.chat, [LIGHT, SPARE], 3000, self.offered())
            if message is None:
                self.misses += 1
                if self.wire.no_credit or self.wire.calls >= ALL_CALLS or self.misses > 2:
                    break
                continue
            calls = []
            for number, call in enumerate(message.get('tool_calls') or []):
                if isinstance(call, dict):
                    fn = call.get('function') if isinstance(call.get('function'), dict) else {}
                    raw = fn.get('arguments')
                    parsed = raw if isinstance(raw, dict) else (json_object(raw) if isinstance(raw, str) else {})
                    calls.append({'id': str(call.get('id') or f'c{self.loop_calls}_{number}'), 'type': 'function', 'function': {'name': str(fn.get('name') or ''), 'arguments': jdump(parsed if isinstance(parsed, dict) else {})}})
            content = message.get('content')
            content = content if isinstance(content, str) else (jdump(content) if content else '')
            if not calls:
                idle += 1
                self.chat.append({'role': 'assistant', 'content': content or '(no call)'})
                self.chat.append({'role': 'user', 'content': 'Call one of the tools to continue.\n' + self.status()})
                if idle >= 3:
                    break
                continue
            idle = 0
            self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
            committing = {self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse')} - {None}
            if self.talk.owed() and any(c['function']['name'] in committing for c in calls):
                self.trace.append({'kind': 'message_guard', 'held': [c['function']['name'] for c in calls], 'used': self.talk.used, 'limit': self.talk.limit})
                for call in calls:
                    self.chat.append({'role': 'tool', 'tool_call_id': call['id'], 'content': 'Not run (no step used): the shopper is asked first.'})
                self.chat.append({'role': 'user', 'content': (self.talk.run() or 'The interviewer had nothing to ask.') + '\n' + self.status()})
                continue
            self.trace.append({'kind': 'turn', 'text': content[:300], 'calls': [c['function'] for c in calls]})
            calls_now = self.join_opens(calls)
            self.turn_size = len(calls)
            self.cart.turn_stopped = False
            for index, call in enumerate(calls_now):
                text = 'Shown in the side-by-side result above: the desk opened these listings together in one step.' if call.get('joined') else self.run_one(call)
                if index == len(calls_now) - 1:
                    text += '\n' + self.status()
                self.chat.append({'role': 'tool', 'tool_call_id': call['id'], 'content': text})


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        errand = Errand(problem_data)
    except Exception as exc:
        return [{'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]}]
    try:
        return errand.run()
    except Exception as exc:
        errand.trace.append({'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]})
        try:
            if not errand.store.finished:
                errand.talk.stuck = True
                errand.finish()
        except Exception as again:
            errand.trace.append({'kind': 'crash', 'error': f'{type(again).__name__}: {again}'[:400]})
        return errand.trace


__all__ = ['agent_main']
