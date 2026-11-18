from __future__ import annotations
import json
import random
import re
import sys
import time
from typing import Any
from src.agent.proxy_client import ProxyClient
LIGHT = 'anthropic/claude-haiku-5.5'
HEAVY = 'anthropic/claude-sonnet-5.5'
SPARE = 'openai/gpt-6-luna'
LOOP_CALLS = 42
ALL_CALLS = 60
SPEND_SOFT = 0.055
SPEND_HARD = 0.062
ROUND2_SPEND = 0.01
LOOP_SPEND = 0.05
HEAVY_ROOM = 0.02
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3
AUTO_PAGES = 6
CHEAPER_OPENS = 3
NOCREDIT_ON = True
LEVER_ON = True
SAYBACK_ON = True
SAYBACK_WORDS = 10
DSEARCH_ON = True
DSEARCH_RUNS = 3
DOPEN_ON = True
DOPEN_ROWS = 4
DOPEN_SHOWN = True
PRECHK_ON = True
PRECHK_TOP = 4
PRECHK_LINE = 2
PRECHK_OPENS = 4
EARLYREV_ON = True
EARLYREV_LEFT = 12
NDROP_ON = True
NDROP_RECORDS = 30
NDROP_FIRES = 2
KINDREV_ON = True
KINDREV_RECORDS = 30
KINDREV_TRIES = 1
HEARDROP_ON = True
OWNDROP_ON = True
DROPSRCH_ON = True
DROPCAP_ON = True
DROPMODE_ON = True
SWAPFIX_ON = True
SWAPFIX_SEEN = 8
CAP_SHARE = 0.75
DROPSRCH_FIRES = 1
LASTREV_ON = True
LASTREV_CAP = 0.072
LASTREV_END = True
REAUDIT_ON = True
REAUDIT_OUT = True
PRICE_ON = True
PRICE_FROM = 0.0275
END_FILL_ON = True
END_FILL_MAX = 200
HOLDROOM_ON = True
HOLD_ROOM = 0.01
HOLD_CEIL = SPEND_HARD
HOLD_CALLS = 4
FITPICK_END_SPEND = SPEND_SOFT
LIGHTROOM_ON = True
LIGHTROOM_TOKENS = 6000
LIGHTROOM_ALL = True
PROJECT_ON = True
PROJECT_HEAVY = 0.02
STATESTOP_ON = False
OFFERFIX_ON = True
OPENALL_ON = True
TRIM2_ON = True
AUDOLD_ON = True
AUDOLD_ROWS = 20
IDPROP_ON = True
FITPICK_ON = True
FITPICK_ROWS = 4
FITPICK_ALL = False
FITPICK_BARE = True
POINTNOW_ON = True
POINTNOW_MAX = 2
ENDID_ON = True
ENDROLE_ON = True
ENDFIX_ON = True
ENDASK_ON = True
ALLLINES_ON = True
FREENEXT_ON = True
BROAD_ON = False
BROAD_ROUNDS = 2
BROAD_LABELS = 8
BROAD_SHARE = 0.1
PLAINASK_ON = True
PLAIN_WORDS = 15
AFTERQ_ON = True
CARTLOOK_ON = True
CARTLOOK_LINES = 3
CARTLOOK_FIRES = 2
EDGECHEAP_ON = True
TOLDBUY_ON = True
TOLD3_ON = False
NOPACK_ON = True
NOPACK_AT = 4
REVEALGATE_ON = True
REVEAL_HOLDS = 0
REVEALBUY_ON = True
REVEALBUY_CAP = 0.08
REVEALBUY_ROWS = 16
CARTFILL_ON = True
BROADSTOP_ON = False
ASKROLE_ON = True
WHATWIDE_ON = True
VALSPLIT_ON = True
TEXTROW_ON = True
WIDE12_ON = True
MLINE_ON = False
MLINE_AT = 3
DECL2_ON = True
DECL2_AT = 2
LINEROT_ON = True
LINEROT_ROUNDS = 3
LABELSRC_ON = True
LABELSRC_TERMS = 5
PARTIAL_ON = True
SOMEFIRST = False
COUNTORDER = True
SRCLINE = True
SRCLINE2 = True
PAGENAMES_ON = True
PAGEHEAD_ROWS = 5
MIDLABEL_ON = True  
MIDLABEL_FROM = 5
MIDLABEL_CORE = 9
MIDLABEL_TO = 17
MIDLABEL_N = 1  
MIDLABEL_MIN = 1  
MIDLABEL_SPREAD_ON = True  
MIDLABEL_SHORT = 40
NEEDMULTI_ON = True  
NEED_K = 3  
NEEDONE1_ON = True  
NEEDDIFF_ON = True  
OPENATTR_ON = True  
OPENATTR_TOP = 5  
OPENATTR_PER = 3  
NAME2_ON = True  
NUMLAST_ON = True  
LINEOPEN_ON = True  
LINEOPEN_RUNS = 3  
HEARDRANK_ON = True  
ASK1OA_ON = True  
NEEDDIFF = ('Keys marked [differs across opened records] are where the listings opened so far disagree, so the '
            "shopper's answer on them decides which listing fits. Among the labels you choose for an item, put "
            'those keys first, before keys that read the same on every opened listing.')
PAGESHEET_ON = True
EDGEPLAIN_ON = True
SWEEP_ON = True
FORM572_ON = True
FORM_TERMS = 15
FORM_LABEL = 4
FORM_MIN = 9
BROADYIELD_ON = True
SWEEPSOLO_ON = True
FOLLOWWAIT_ON = True
NEVERANS_ON = True
AFTERSWEEP_ON = False  
NUMHINT_ON = True
OPENFIRST_ON = True
OPENFIRST_MAX = 8
ALLREC = True
SECONDEND_ON = True
SECONDEND_AT = 2
FULLONLY_ON = True
EMPTYEND_ON = True
CARTREADY_ON = True
STALEDONE_ON = True
POOLWIDE_ON = True
POOLWIDE_ROWS = 64
POOLWIDE_CHARS = 40000
POOLTRIM_ON = True
POOLTRIM_ROWS = 128
POOLTRIM_OPEN = (200, 600)
POOLTRIM_ROW = (120, 500)
POOLTRIM_CHARS = 56000
ROWORDER_ON = True
ROWTAG_ON = True
POOLGATE_ON = True
RECSEARCH_ON = True
RECSEARCH_EACH = 2
RECSEARCH_ROWS = 8
RECSEARCH_CHARS = 12000
RECARGS_ON = True
RECARGS_LOG = 24
RECARGS_SCHEMA = 3000
RECARGS = 'A shopping agent is about to end without buying because its searches found nothing that fits the items below. Two things most often hide a fitting listing: it sits on a later page of the very search the agent already ran for that item, or that search was narrowed by number limits or other filters the item does not need. Less often the store uses words the agent never searched. For each item write up to two new searches, in this order of preference: (1) the item\'s own earlier search again with the same words: when the search tool shown below has an argument that picks a later page of results, on its next page (never past page 3); when it has none, never invent one and change only the filters as follows; if that earlier search set any number limit, leave every number limit out, and leave out the other extra filters too; (2) only if no earlier search for the item set a number limit: the same search with the one number limit the request itself states, if it states one and the tool has a field for it; (3) a short query of two to five words taken from how the listings seen for the item describe this kind of product, not used by any search already run. Use only argument fields of the search tool shown below and never repeat a search already run exactly.\nReturn ONE JSON object and nothing else: {"searches": [{"item": item number, "args": {the full arguments for the search tool}}]}'
RECASK = 'A shopping agent is about to end without buying because its searches found nothing that fits the items below. A fitting listing is often missed because the store lists it under words the agent never searched. For each item, read the listings already seen for it and write up to two new search queries that use words those listings themselves use for this kind of product (their type, category or feature words) and that no search already run has used. Keep each query short (two to five words) and about the item itself, not a neighbouring kind of product.\nReturn ONE JSON object and nothing else: {"searches": [{"item": item number, "query": "the search text"}]}'
OPENRANK_ON = True
GAPFILL_ON = True
EDGE_OLD = ', "edge": "min" when the need gives only the lowest amount the shopper accepts, "max" when it gives only the highest amount they accept of something other than price, else null'
EDGE_PLAIN = (', "edge": "min" only when the reply itself says in words that the amount is the lowest the '
              'shopper accepts, "max" only when it says in words that the amount is the highest they accept of '
              'something other than price; a plain number or amount with no such words is the exact value '
              'wanted, so null; else null')
EDGE_NOTE = EDGE_PLAIN if EDGEPLAIN_ON else EDGE_OLD
QUIETFIX_ON = True
ENDPAIR_ON = True
SPENDTRIM_ON = True
USAGE_ON = True
AUDITLIGHT_ON = True
KEYASK_ON = False
ONEKIND_ON = True
AUDIT_HEAVY_UNTIL = 0.0
HEAVY_UNTIL = 0.03
_llm =ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
GUIDE = 'You are Kyo, a careful buyer working for one shopper in an online store. You can act only by calling the store\'s tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.\n\nHOW THE STORE WORKS\n- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool\'s description and schema and use the exact argument names and allowed values.\n- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.\n- Search rows are summaries. Stores split a listing\'s record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Kyo knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.\n- Questions to the shopper are handled for you by Kyo\'s interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.\n- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Kyo carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store\'s tools (product questions go through Kyo\'s interviewer).\n\nWHAT COUNTS AS A FIT\n- The kind of product is decided by the listing\'s category label: only the label that matches the requested item as a whole counts, never a neighbouring one.\n- A requirement on a property is met when the listing\'s attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value\'s words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.\n- An attribute whose key names the requirement\'s property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing about the requirement. Kyo adds [evidence] lines after opened listings with the keys whose names match a requirement.\n- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.\n- When the category label itself states a requirement\'s value (a season or audience in the label), that requirement is met by the category.\n- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing\'s value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.\n- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.\n- When one requested item must fit another (an accessory for a device), choose the two together from the listings\' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.\n- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.\n- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.\n\nCHOOSING\n- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Kyo reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.\n- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.\n- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.\n- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.\n- A request for several different items needs one cart line per item, each at its requested quantity.\n\nCHANGES WHILE YOU WORK\n- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool\'s result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.\n- Re-read a chosen listing\'s record shortly before checking out, because price and stock can change silently.\n\nFINISHING\n- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.\n- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request\'s wording for that ending, and when a remark is needed, keep it short and name no listing.\n- Every reply must call at least one tool. Keep your text short: what you know, then the next move.'
BRIEF = 'You prepare a shopping task for an agent. Read the shopper\'s request and the store\'s tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:\n\n"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request\'s words, "shelf": the store\'s category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request\'s own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property\'s value or limit outright; and never turn words that are part of what the product is called (a season or an audience named with the product) into musts: those belong to "name". A use or occasion the request makes a property of the product itself ("blazers for business wear", "a dress made for parties", "the occasion has to be party") IS a must on that property ("occasion: business"), unless the store\'s category label already names it; an event, person or setting the product is bought for ("to wear at his wedding", "for a colleague"), and how it should sit, feel or look in use, stay context. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.\n"budget": the most the shopper will pay in total as a number in the request\'s own unit, or null.\n"budget_unit": that unit in words (for example "cents"), or null.\n"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.\n"in_stock_only": true unless the request clearly accepts items that are not available.\n"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.\n"pick_measure": what is maximised or minimised for "most"/"least", else "".\n"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.\n"steps": how many steps, moves or tool calls the request allows (integer), or null.\n"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].\n"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.\n"wishes": a list of things the shopper would merely like.\n"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).\n"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper\'s approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.\n"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.\n"fits": a list of {"item": item number, "with": item number} for each requested item that the request says must work with, fit or suit ANOTHER requested item (the one the agent picks); [] when none.\n"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing\'s attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing\'s current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).\n"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.\n"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.\n'
PLAN = 'You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property; a property the shopper has no view on gets "don\'t mind", and one the shopper does not think about at all gets a puzzled reply, so every property named spends part of a message.\n\nYou get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item\'s category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.\n\nChoose the properties in this order:\n0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.\n1. Hinted properties. The request\'s story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product\'s looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.\nName a hinted property by a listed key of that item when one fits the hint (a hint about who will wear it: the key for gender or audience; about its looks: the listed key for colour, pattern or finish that most of its listings carry); a property no listing of the item carries comes after the item\'s three most carried keys.\n2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item\'s listings; once five or more listings were read, avoid keys seen on only one or two of them.\nSkip a property the request already settles for that item (a requirement names that property; a key whose values only repeat a required value, such as a pattern key showing the required decoration, is NOT settled), a second name for a property already chosen, a key whose values each list several purchase options at once, identifiers, model names, warranty and other logistics. Pack sizes, counts and quantities per pack are properties like any other, and a key marked [not the stated ...] is another property than that requirement, whatever words its name shares with it. Rank every item on its own: its hinted properties first, then its listed keys.\n"label" is the property\'s plain name: for a listed key, all of the key\'s words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property\'s name, no example values, no brackets, no second property.\nChoose up to COUNT for each item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story\'s words it comes from, or an empty string"}]}, each item\'s picks in rank order'
IDPROP = 'On identifiers: an identifier is a code that names the listing itself. A key whose values name the model of a part that the product contains, comes with or works with describes the product: it is a property, not an identifier, and may be asked even when its values look like codes.'
HEAR = 'A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "budget": new total limit as a number or null}\n- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3").\n- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop\'s product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer\'s own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.\n- "free": the property of each question the shopper said they do not mind about.\n- Folksy or roundabout wording that names a value still states that value.'
NOTE = 'While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:\n{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request\'s unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}\nA replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note.'
CHECK = 'You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.\nReading rules: a requirement is met when the listing\'s attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value\'s words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing\'s value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.\nWhen the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart\'s choice on that rule, the cart is wrong.\nReturn ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses.'
READING = 'the listing\'s attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value\'s words counts, such as "deep teal" for teal, limits include the edge); a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing\'s value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit.'
if NUMHINT_ON:
    READING = READING[:-1] + ('; a record whose text repeats a number the shopper told is not proof that it meets '
                              'that requirement: it meets it only when that number is the value of the attribute the '
                              'requirement names.')
CARTFILL = ('A shopping agent is about to call the store\'s cart tool to put one listing in the cart, but some required '
            'arguments of that tool are not filled yet. Fill them for this listing, using only values that appear in the '
            'LISTING RECORD below. Where the record offers several choices for an argument, take the one that best fits the '
            'request, the requirements in force and the budget; when they fit equally, the cheapest. Keep the arguments that '
            'are already set.\nReturn ONE JSON object and nothing else: {"args": {"argument name": "value"}}')
NEAR = ('The shopper has already answered questions about what they want, so this shopping episode must end with a purchase '
        'of the nearest listing, not with no purchase. For each requested item (numbered from 1; a single item is item 1), '
        'pick the ONE listing below that comes nearest to it: first the requested kind of product, then the shopper\'s '
        'answers, then the other requirements and the budget. A pick may miss a detail; pick it anyway. Prefer an opened '
        'page over a row when both are equally near. Give null for an item only when no listing below is that kind of '
        'product at all.\nReturn ONE JSON object and nothing else: {"near": {"item number": "listing id or null"}, '
        '"misses": {"item number": "the requirement the pick misses, or none"}}')
AUDIT = 'A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: ' + READING + '\nRecords marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.\nReturn ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}'
FIT = "A shopping agent is about to finish. The shopper answered its questions, and each answer is a requirement as firm as the request's own. Check each listing record below against EVERY numbered requirement, one at a time, with these reading rules: " + READING + '\nReturn ONE JSON object and nothing else: {"rows": {"listing id": {"meets": [numbers of the requirements its record shows met], "fails": [numbers of the requirements its record shows a different value for]}}}. A requirement its record does not address goes in neither list.'
RELOOK = "A note has just eased a shopping agent's task (a requirement was dropped or the budget was raised), so a listing passed over earlier may fit now. Check the listing records shown, all of the requested kinds, against the requirements now in force (only those), with these reading rules: " + READING + '\nRecords marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.\nReturn ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement now, best first, at most 4]}, "open": [ids of ROW listings whose page could show that they fit now, most promising first, at most 6], "reason": "one sentence"}'
UNMIX = 'A shopper\'s request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request\'s wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}'
TRAIL = 'A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing\'s page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement\'s chain from the requirement\'s own description, link by link, through the pages\' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).\nReturn ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing, "find": "for a link not found: search words (its shop and the values the description gives)"}], "done": true when the last link is an OPENED page that states the property\'s value itself (also when the requirement asks for a value other than or unlike that one: the page\'s value is what it is measured against), else false, "value": the property\'s value as stated on that last page, or ""}]}\nA record is a link only when it matches the whole description of that link: the shop or brand, the kind of product AND every property value the description gives (for example "a 10 pieces multipack" rules out the same shop\'s listing with 10 capsules and no multipack); when no record matches all of it, that link is not found yet, even if the same shop sells other listings of that kind. Give "find" for such a link: the shop\'s name and the description\'s values, as a few search words.\nList each chain\'s links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown.'
GATE = 'A shopping agent is about to STAGE. The shopper\'s request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product\'s properties do not do a step.\nWhen a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.\nReturn ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}'
SWAP = "A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: " + READING + '\nReturn ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line\'s item, at most 6], "reason": "one sentence"}'
SHAPE = 'A store tool returned the result below. Name its parts by the exact keys the result and the tool\'s parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing\'s price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}'
COMMON = frozenset('about also been could does from have into just like make more much only over really should some such than that their them then there these they this very want what when which will with would your yours'.split())
CAP_WORDS = frozenset('below max maximum most under upto'.split())
BOUNDS = frozenset('above approximately around at below between exactly ideally least less max maximum min minimum most must over prefer preferably preferred required should under up within'.split())
NEGATED = re.compile('\\b(?:not|no|never|without|except|unlike|avoid|anything but|other than|instead of)\\b')
GUARD_KEYS = 3
GUARD_SHARE = 0.0
GUARD_GAP = 0.34
NOT_PROPERTY = '\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b'
CATCH_ALL = '^(?!.*\\b(?:types?|styles?|kinds?|shapes?)\\b).*\\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\\w*|accessor\\w*|extras?|misc\\w*)\\b'
WIDEN_EACH = 6
FIT_CODE = '[A-Z][A-Za-z]{0,5} ?\\d{3,6}'
PART_CODE = '[A-Z]{1,4}\\d{3,6}'

def _codes(text: Any) -> list[str]:
    return [re.sub('\\s+', ' ', c).strip().lower() for c in re.findall('\\b' + FIT_CODE + '\\b', str(text or ''))]

def _fit_marks(text: Any) -> dict[str, Any]:
    text = str(text or '')
    own = re.search('[\\[(]\\s*(?:part\\s+)?(' + PART_CODE + ')\\s*[\\])]', text, re.I)
    codes: list[str] = []
    for hit in re.finditer('\\bmodel\\s*(?:no\\.?|number|#)\\s*[:.]?\\s*(' + FIT_CODE + ')\\b', text, re.I):
        codes += _codes(hit.group(1))
    for hit in re.finditer('\\b(?:also sold as|also known as|sold also as)\\s+([^.;]+)', text, re.I):
        codes += _codes(hit.group(1))
    works: list[str] = []
    for hit in re.finditer('\\b(?<!not )(?<!never )(?:works with|compatible with|fits)\\s*:?\\s+([^.;]+)', text, re.I):
        if not re.search('\\bpart\\s+' + PART_CODE + '\\s+also\\s+$', text[max(0, hit.start() - 40):hit.start()], re.I):
            works += _codes(hit.group(1))
    also = [(hit.group(1).upper(), _codes(hit.group(2))) for hit in re.finditer('\\bpart\\s+(' + PART_CODE + ')\\s+also fits\\s*:?\\s*([^.;]+)', text, re.I)]
    more = re.search('\\b(?:more|details|see|read)\\s+(?:on|at|under|about)?\\s*(?:the\\s+)?part\\s+(' + PART_CODE + ')', text, re.I)
    return {'own': own.group(1).upper() if own else '', 'codes': codes, 'works': works, 'also': also, 'more': more.group(1).upper() if more else ''}

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

def _numfield(per: dict[str, str]) -> bool:
    

    distinct = set(per.values())
    hits = 0
    for value in distinct:
        chars = [c for c in value if not c.isspace()]
        hits += bool(chars) and 2 * sum(c.isdigit() for c in chars) >= len(chars)
    return bool(distinct) and 2 * hits >= len(distinct)


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

def _digits(text: Any) -> bool:
    return any((ch.isdigit() for ch in str(text or '')))


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

_TEXTROWS = [0]


def _scalar_of(text: str) -> Any:
    value = text.strip()
    if re.fullmatch('-?(?:0|[1-9]\\d{0,5})', value):
        return int(value)
    if re.fullmatch('-?(?:0|[1-9]\\d{0,6})\\.\\d{1,4}', value):
        return float(value)
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    return value


def _row_parts(text: str) -> list[tuple[str, str]] | None:
    for sep in ('|', '\n'):
        if sep not in text:
            continue
        bits = [b for b in text.split(sep) if b.strip()]
        parts: list[list[str]] = []
        loose = 0
        for bit in bits:
            key, mark, value = bit.partition(':')
            if mark and value.strip() and len(key.split()) <= 4 and len(key.strip()) <= 40 and re.search('[A-Za-z]', key) and not re.search('[|=;{}\\[\\]"]', key):
                parts.append([key.strip(), value.strip()])
                continue
            loose += 1
            if parts:
                parts[-1][1] += ' | ' + bit.strip()
        if sep == '\n' and loose:
            continue
        if len(parts) >= 3 and len(parts) * 5 >= len(bits) * 3:
            return [(k, v) for k, v in parts]
    return None


def _pair_block(text: str) -> dict[str, Any] | None:
    if text.count('=') < 2:
        return None
    for sep in (';', ','):
        bits = [b for b in text.split(sep) if b.strip()]
        pairs: list[list[str]] = []
        for bit in bits:
            key, mark, value = bit.partition('=')
            if mark and len(key.strip()) <= 40 and re.search('[A-Za-z]', key):
                pairs.append([key.strip(), value.strip()])
            elif pairs:
                pairs[-1][1] += sep + bit
            else:
                break
        if len(pairs) >= 2 and len(pairs) * 5 >= len(bits) * 3:
            return {k: _scalar_of(v) for k, v in pairs}
    return None


def _text_row(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str) or len(text) < 12 or text.count(':') < 3:
        return None
    parts = _row_parts(text)
    if not parts:
        return None
    row: dict[str, Any] = {}
    for key, value in parts:
        if key not in row:
            row[key] = _pair_block(value) or _scalar_of(value)
    _TEXTROWS[0] += 1
    return row


def _keyed(node: Any, depth: int=0) -> Any:
    if depth > 6:
        return node
    if isinstance(node, str):
        return _text_row(node) or node
    if isinstance(node, list):
        if node and all((isinstance(v, str) for v in node)):
            rows = [_text_row(v) for v in node]
            if sum((r is not None for r in rows)) * 2 >= len(node):
                return [r if r is not None else v for r, v in zip(rows, node)]
            return node
        return [_keyed(v, depth + 1) for v in node]
    if isinstance(node, dict):
        if len(node) == 1:
            only = _text_row(next(iter(node.values())))
            if only is not None:
                return only
        return {k: v if k == 'user_message' else _keyed(v, depth + 1) for k, v in node.items()}
    return node

def _page_rows(node: Any) -> dict[str, int]:
    
    out: dict[str, int] = {}
    queue: list[Any] = [node]
    while queue and len(out) < 400:
        x = queue.pop(0)
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(v, (str, int, float)) and (not isinstance(v, bool)):
                    name = _words(str(k))
                    if name and name not in out:
                        out[name] = len(out)
                elif isinstance(v, (dict, list)):
                    queue.append(v)
        elif isinstance(x, list):
            queue.extend(x[:60])
    return out


def _mid_names(node: Any) -> dict[str, int]:
    

    out: dict[str, int] = {}
    seen = 0
    named: set[str] = set()
    queue: list[Any] = [node]
    while queue and seen < 400:
        x = queue.pop(0)
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(v, (str, int, float)) and (not isinstance(v, bool)):
                    name = _words(str(k))
                    if name and name not in named:
                        named.add(name)
                        if len(str(v).strip()) <= MIDLABEL_SHORT:
                            out[name] = seen
                        seen += 1
                elif isinstance(v, (dict, list)):
                    queue.append(v)
        elif isinstance(x, list):
            queue.extend(x[:60])
    return out


def _mid_values(node: Any) -> dict[str, str]:
    

    out: dict[str, str] = {}
    seen = 0
    named: set[str] = set()
    queue: list[Any] = [node]
    while queue and seen < 400:
        x = queue.pop(0)
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(v, (str, int, float)) and (not isinstance(v, bool)):
                    name = _words(str(k))
                    if name and name not in named:
                        named.add(name)
                        value = ' '.join(str(v).split()).lower()
                        if len(value) <= MIDLABEL_SHORT:
                            out[name] = value
                        seen += 1
                elif isinstance(v, (dict, list)):
                    queue.append(v)
        elif isinstance(x, list):
            queue.extend(x[:60])
    return out


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
        self.named = False
        self.call_names = 0
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
        self.need_last: bool | None = None
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
        self.reaudited = False
        self.audit_ids: set[str] = set()
        self.cart_since: dict[str, int] = {}
        self.taken_out: list[str] = []
        self.built: list[str] = []
        self.dropped: set[str] = set()
        self.late_held = False
        self.named_better: list[str] = []
        self.swapped = False
        self.lead_items: dict[str, list[str]] = {}
        self.verdicts: dict[str, bool] = {}
        self.confirmed: str | None = None
        self.completed = False
        self.chose: set[tuple[int, str]] = set()
        self.broad: dict[int, dict[str, Any]] = {}
        self.look_told: list[str] = []
        self.looks = 0
        self.told_held = False
        self.reveal_holds = 0
        self.reveal_last = False
        self.near_tried = False
        self.reveal_events = 0
        self.plan_picks: dict[int, int] = {}
        self.mline_done = False
        self.oattr_done = False
        self.oattr1_done = False
        self.broad_stay: int | None = None
        self.id_key: str | None = None
        self.facts_from: dict[str, bool] = {}
        self.queued: list[str] = []
        self.own_relax = False
        self.dropsrchs = 0
        self.drop_done: set[str] = set()
        self.buys = 0
        self.refusals = 0
        self.end_tries = 0
        self.second_end = False
        self.empty_tried = False
        self.gap_tried = False
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
        self.top_shown: set[str] = set()
        self.shown: dict[str, int] = {}
        self.best_at: dict[str, int] = {}
        self.top_marks: set[str] = set()
        self.top_read: list[str] = []
        self.wider_held = False
        self.paged_obs: list[Any] = []
        self.shortlist: dict[str, list[str]] = {}
        self.search_log: list[dict[str, Any]] = []
        self.found_for: dict[str, int] = {}
        self.src_item = 0
        self.dsearched = False
        self.dopened = False
        self.fit_held = False
        self.fit_declined = False
        self.hold_cap = 0.0
        self.hold_calls = 0
        self.early_done = False
        self.point_runs = 0
        self.point_mark = 0
        self.unfound: list[tuple[str, str]] = []
        self.state_stopped = False
        self.early_empty = False
        self.trimmed = False
        self.audit_window: list[str] = []
        self.end_turned = 0
        self.cite_held = False
        self.end_ids: dict[str, str] = {}
        self.end_reply: Any = None
        self.end_fixed: dict[str, Any] | None = None
        self.end_asked = False
        self.open_order: list[str] = []
        self.blind_obs: Any = None
        self.blind_mark = 0
        self.views = 0
        self.openfirst_n = 0
        self.kindrevs = 0
        self.relaxed = False
        self.relooks = 0
        self.note_leads: list[str] = []
        self.plan_order: list[int] = []
        self.line_holds = 0
        self.bound: list[tuple[int, int]] = []
        self.fit_holds = 0
        self.searched: list[dict[str, Any]] = []
        self.link_searches = 0
        self.link_seen: dict[str, list[str]] = {}

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

    def heavy_ok(self) -> bool:
        over = bool(LEVER_ON and SPENDTRIM_ON) and self.cost > HEAVY_UNTIL
        if over and not self.trimmed:
            self.trimmed = True
            self.trace.append({'kind': 'lever_fire', 'why': 'spendtrim', 'cost': round(self.cost, 4), 'step': self.step})
        return not over

    def judges(self, limit: float=SPEND_SOFT) -> list[str]:
        if self.cost <= limit and self.cost + HEAVY_ROOM <= SPEND_HARD and self.heavy_ok():
            return [HEAVY, LIGHT]
        return [LIGHT] if self.cost <= SPEND_HARD else []

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None=None) -> dict[str, Any] | None:
        for model in models:
            if self.calls >= ALL_CALLS or (LEVER_ON and NOCREDIT_ON and self.no_credit):
                return None
            self.calls += 1
            tries = 0
            while True:
                message, status = self.call_model(model, messages, max_tokens, tools)
                if message is not None:
                    return message
                passing = status is None or status in ((408, 425, 429) if LEVER_ON and NOCREDIT_ON else (402, 408, 425, 429)) or (isinstance(status, int) and status >= 500)
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
        payload = list(messages)
        if model.startswith('anthropic/') and len(payload) > 1:
            for spot in sorted({0, len(payload) - 1}):
                piece = payload[spot].get('content')
                if isinstance(piece, str) and piece:
                    payload[spot] = {**payload[spot], 'content': [{'type': 'text', 'text': piece, 'cache_control': {'type': 'ephemeral'}}]}
        body: dict[str, Any] = {'model': model, 'messages': payload, 'max_tokens': max_tokens, 'temperature': 0}
        if tools:
            body.update(tools=tools, tool_choice='auto')
        reply = _llm.post_verbose('/inference/chat/completions', body)
        data = reply.data
        if not (isinstance(data, dict) and data.get('choices')):
            fault = reply.error or {}
            code = fault.get('status')
            inner = data.get('error') if isinstance(data, dict) else None
            if isinstance(inner, dict):
                code = _count(inner.get('code'))
                fault = {'body': inner.get('message')}
            self.trace.append({'kind': 'llm_fail', 'model': model, 'status': code, 'body': str(fault.get('body') or '')[:200]})
            return (None, code)
        usage = data.get('usage') or {}
        price = usage.get('cost')
        if isinstance(price, (int, float)):
            self.cost += float(price)
        if USAGE_ON:
            self.usage_line(model, usage)
        message = data['choices'][0].get('message') or {}
        if not (message.get('content') or message.get('tool_calls')):
            self.trace.append({'kind': 'llm_fail', 'model': model, 'status': 0})
            return (None, 0)
        return (message, 200)

    def usage_line(self, model: str, usage: dict[str, Any]) -> None:
        cached = usage.get('prompt_tokens_details')
        row: dict[str, Any] = {'kind': 'usage', 'by': getattr(self, 'usage_by', '') or 'turn', 'heavy': model == HEAVY}
        row['usd'] = round(float(usage.get('cost') or 0), 6)
        row['tin'] = _count(usage.get('prompt_tokens'))
        row['tout'] = _count(usage.get('completion_tokens'))
        row['tcache'] = _count(cached.get('cached_tokens')) if isinstance(cached, dict) else None
        self.trace.append(row)

    def ask_json(self, prompt: str, models: list[str], max_tokens: int=1800) -> dict[str, Any] | None:
        self.usage_by = sys._getframe(1).f_code.co_name if USAGE_ON else ''
        try:
            return self.ask_json_once(prompt, models, max_tokens)
        finally:
            self.usage_by = ''

    def ask_json_once(self, prompt: str, models: list[str], max_tokens: int) -> dict[str, Any] | None:
        for n, model in enumerate(models):
            if LEVER_ON and PROJECT_ON and (model == HEAVY) and (self.cost + PROJECT_HEAVY > SPEND_HARD):
                if LIGHT in models[:n]:
                    self.trace.append({'kind': 'lever_fire', 'why': 'project_skip'})
                    return None
                model = LIGHT
                self.trace.append({'kind': 'lever_fire', 'why': 'project_light'})
            room = max_tokens
            if LEVER_ON and LIGHTROOM_ON and LIGHTROOM_ALL and (model == LIGHT) and (max_tokens < LIGHTROOM_TOKENS):
                room = LIGHTROOM_TOKENS
                self.trace.append({'kind': 'lever_fire', 'why': 'lightroom_all'})
            message = self.think([{'role': 'user', 'content': prompt}], [model], room)
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
        self.src_item = 0
        if LEVER_ON and SRCLINE and len(getattr(self, 'items', None) or []) > 1 and name == self.roles.get('search'):
            said = ' '.join((v for v in args.values() if isinstance(v, str)))
            self.src_item = self.owner_score(said) if said.strip() else 0
        if self.named:
            self.name_call(payload)
        for _ in range(4):
            result = _env.post_verbose('/environment/call', payload)
            if isinstance(result.data, dict):
                before = _TEXTROWS[0]
                data = self.unwrap(result.data)
                if TEXTROW_ON and 'observation' in data:
                    data['observation'] = _keyed(data['observation'])
                    if _TEXTROWS[0] > before:
                        self.text_rows = getattr(self, 'text_rows', 0) + _TEXTROWS[0] - before
                        if not getattr(self, 'textrow_fired', False):
                            self.textrow_fired = True
                            self.trace.append({'kind': 'lever_fire', 'why': 'textrow', 'rows': _TEXTROWS[0] - before, 'step': self.step})
                self.step = data['step'] if isinstance(data.get('step'), int) else self.step + 1
                if data.get('finished'):
                    self.finished = True
                seen = data.get('observation')
                if name in self.page_tools() and (not (isinstance(seen, dict) and seen.get('error'))):
                    for value in args.values():
                        self.opened.update((v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)))
                        self.open_order += [v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str) and v.strip() in self.rows]
                if LEVER_ON and OPENFIRST_ON and (not (isinstance(seen, dict) and seen.get('error'))):
                    self.blind_note(name, seen)
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
                    if self.named:
                        self.name_call(payload)
                    continue
                if 'reuse' in low or 'different action' in low:
                    self.step += 1
                    payload['step'] = self.step + 1
                    if self.named:
                        self.name_call(payload)
                    continue
                return None
            if status == 422 and re.search('finished|terminal observation', body):
                self.finished = True
                return None
            if status == 422 and (not self.named) and re.search('call_id|idempotency|turn must', body):
                self.named = True
                self.trace.append({'kind': 'lever_fire', 'why': 'call_stamp', 'step': self.step})
                self.name_call(payload)
                continue
            if status in (400, 401, 402, 403, 404, 422):
                if status == 402:
                    self.broke = True
                return None
        return None

    def name_call(self, payload: dict[str, Any]) -> None:
        self.call_names += 1
        turn = self.step + 1
        tag = f'k{turn}-{self.call_names}'
        payload.update(step=turn, turn=turn, call_id=tag, idempotency_key=tag)

    @staticmethod
    def unwrap(data: dict[str, Any]) -> dict[str, Any]:
        calls = data.get('calls')
        if not isinstance(calls, list) or len(calls) != 1 or (not isinstance(calls[0], dict)):
            return data
        wrap = calls[0].get('observation')
        if not (isinstance(wrap, dict) and 'observation' in wrap and (set(wrap) <= {'done', 'error', 'observation'})):
            return data
        seen = _keyed(wrap['observation']) if TEXTROW_ON else wrap['observation']
        if wrap.get('error') not in (None, False, '', [], {}):
            seen = dict(seen, error=wrap['error']) if isinstance(seen, dict) and 'error' not in seen else {'error': wrap['error'], 'observation': seen}
        said = data.get('user_message')
        words = said.get('content') if isinstance(said, dict) else said
        if isinstance(words, str) and words.strip():
            seen = dict(seen, user_message=[words]) if isinstance(seen, dict) else {'observation': seen, 'user_message': [words]}
        data['observation'] = seen
        if not isinstance(data.get('step'), int):
            count = next((data[k] for k in ('action_count', 'turn') if isinstance(data.get(k), int)), None)
            if count is not None:
                data['step'] = count
        if wrap.get('done') is True or calls[0].get('done') is True:
            data['finished'] = True
        return data

    def ask_role(self, brief: dict[str, Any]) -> None:
        taken = {v for k, v in self.roles.items() if k != 'ask'}
        lines = []
        for tool in self.tools:
            fn = tool['function']
            lines.append(f"- {fn['name']}: {str(fn.get('description') or '')[:400]} | parameters: {_dump(fn.get('parameters') or {})[:600]}")
        prompt = ('An agent does a shopping task in an online store on behalf of a shopper. Some stores give the agent a tool '
                  'that sends a message or a question to that shopper and returns what the shopper replies; other stores do not. '
                  'Below are this store\'s tools. Which ONE of them, if any, sends a message to the shopper? If none does, answer null.'
                  '\nTOOLS:\n' + '\n'.join(lines)
                  + '\nReturn ONE JSON object and nothing else: {"ask": "the exact tool name, or null"}')
        found = None
        for models in ([LIGHT], [HEAVY]):
            got = self.ask_json(prompt, models, 400) or {}
            name = got.get('ask') if isinstance(got, dict) else None
            props = ((self.schemas.get(name) or {}).get('properties') or {}) if isinstance(name, str) else {}
            if isinstance(name, str) and name in self.schemas and name not in taken and any('string' in _kinds(p or {}) for p in props.values()):
                found = name
                break
            if self.no_credit or self.broke:
                break
        zero = False
        if found:
            self.roles['ask'] = found
            brief['roles'] = {**(brief['roles'] if isinstance(brief.get('roles'), dict) else {}), 'ask': found}
            if _count(brief.get('messages')) == 0:
                brief['messages'] = None
                zero = True
        self.trace.append({'kind': 'lever_fire', 'why': 'askrole', 'found': bool(found), 'zero': zero, 'step': self.step})

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
        if LEVER_ON and ENDROLE_ON and (not self.no_credit) and self.roles.get('buy') and (not self.roles.get('refuse')):
            buy = self.roles['buy']
            ask = f'A shopping task in an online store ends with exactly one of two outcomes: a checkout, or an ending without a purchase. The checkout tool is {buy}. Which tool ends the episode without a purchase? It may be {buy} itself with other arguments.\nTOOLS:\n' + '\n'.join(lines) + '\nReturn ONE JSON object and nothing else: {"refuse": the exact tool name, or null when no tool does it, "refuse_args": the fixed arguments that make it end without a purchase, "buy_args": the fixed arguments that make ' + buy + ' check out}.'
            got = self.ask_json(ask, [LIGHT, HEAVY], 600) or {}
            name = got.get('refuse')
            if isinstance(name, str) and name in self.schemas:
                self.roles['refuse'] = name
                brief['roles'] = {**(brief['roles'] if isinstance(brief.get('roles'), dict) else {}), 'refuse': name}
                if isinstance(got.get('refuse_args'), dict):
                    brief['refuse_args'] = got['refuse_args']
                if name == buy and isinstance(got.get('buy_args'), dict) and (not brief['buy_args']):
                    brief['buy_args'] = got['buy_args']
            self.trace.append({'kind': 'lever_fire', 'why': 'end_role', 'found': bool(self.roles.get('refuse')), 'same': self.roles.get('refuse') == buy})
        if LEVER_ON and ASKROLE_ON and (not self.no_credit) and (not self.roles.get('ask')):
            self.ask_role(brief)
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
        self.offer_fix()

    def offer_fix(self) -> None:
        if not (LEVER_ON and OFFERFIX_ON) or self.roles.get('offer'):
            return
        jobs = {self.roles.get(k) for k in ('open', 'cart', 'cart_view', 'buy', 'refuse', 'ask', 'search', 'compare')}

        def numeric(tool: str) -> bool:
            props = (self.schemas.get(tool) or {}).get('properties') or {}
            return any(('number' in _kinds(v) or 'integer' in _kinds(v) for v in props.values() if isinstance(v, dict)))
        page = next((t for t in self.schemas if self.single_target(t) and t not in jobs and (not numeric(t))), None)
        if page:
            self.roles['offer'] = page
            self.offer_fixed = page
            self.trace.append({'kind': 'lever_fire', 'why': 'offerfix'})

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
        if LEVER_ON and REVEALGATE_ON and DECL2_ON and self.buys == 0 and self.asked and (not self.revealed()) and self.buy_push():
            text += '\n[finish] Finish with a purchase: the closest fitting listing for each item, then check out. An ending without a purchase rarely scores here.'
        if LEVER_ON and REVEALGATE_ON and self.buys == 0 and self.revealed() and self.buy_push():
            text += ('\n[shopper answered] The shopper has revealed what they want, so finish with a purchase: the closest '
                     'fitting listing for each item, then check out. An ending without a purchase scores nothing now.')
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
                        if getattr(self, 'src_item', 0):
                            self.found_for.setdefault(ident, self.src_item)
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
                if rows:
                    for ident in {self.ident(n, True, depth + 1) for n in node if isinstance(n, dict)} - {None, ''}:
                        self.shown[ident] = self.shown.get(ident, 0) + 1
                    for place, value in enumerate([n for n in node if isinstance(n, dict)]):
                        ident = self.ident(value, True, depth + 1)
                        if ident and place < self.best_at.get(ident, 1 << 30):
                            self.best_at[ident] = place
                if rows and PRECHK_ON:
                    for value in [n for n in node if isinstance(n, dict)][:PRECHK_TOP]:
                        ident = self.ident(value, True, depth + 1)
                        if ident:
                            self.top_shown.add(ident)
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
            picks += [i for i in pool[:2] if i not in picks]
        return self.fill_facts(picks[:CHEAPER_OPENS], self.tail() + 4)

    def top_rows(self) -> list[str]:
        if not (LEVER_ON and PRECHK_ON) or len(self.items) < 2 or self.cost > SPEND_SOFT:
            return []
        mark = self.cart_mark() + f'#{self.version}'
        cat_key = self.category_key()
        if mark in self.top_marks or not cat_key:
            return []
        self.top_marks.add(mark)
        cart = self.cart_ids()
        picks: list[str] = []
        for line in cart:
            kind = str((self.rows.get(line) or {}).get(cat_key) or '').strip().lower()
            if not kind:
                continue
            pool = [i for i in self.seen_order if i in self.top_shown and i not in cart and (i not in self.full) and (i not in self.opened) and (i not in picks) and self.buyable(i) and (str((self.rows.get(i) or {}).get(cat_key) or '').strip().lower() == kind)]
            picks += pool[:PRECHK_LINE]
        picks = picks[:PRECHK_OPENS]
        texts = self.fill_facts(picks, self.tail() + 4) if picks else []
        self.top_read = [i for i in picks if i in self.full]
        if picks:
            self.trace.append({'kind': 'lever_fire', 'why': 'precheck_top', 'pool': len(picks), 'read': len(self.top_read)})
        return texts

    def observe(self, name: str, observation: Any) -> str:
        self.absorb(observation)
        lines = self.loose_lines(observation)
        if not lines:
            return ''
        meaning = self.take_note(lines)
        if meaning:
            self.queued.append(f'(seen on {name}) {meaning}')
            if self.relaxed and OWNDROP_ON:
                self.own_relax = True
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
        self.widen()
        if LEVER_ON and LINEOPEN_ON and len(self.items) >= 2:
            self.line_open()
        lanes = self.plan_lanes(max(self.msgs_free(), BROAD_LABELS) if LEVER_ON and WIDE12_ON else self.msgs_free())
        found = {n: 0 for n in lanes}
        spent = {n: 0 for n in lanes}
        used: dict[int, set[str]] = {n: set() for n in lanes}
        follow: list[tuple[int, list[str], bool]] = []
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        crowded = False
        solo = LEVER_ON and ((NOPACK_ON and len(self.items) >= NOPACK_AT) or (SWEEP_ON and SWEEPSOLO_ON and len(self.items) > 1))
        if solo:
            self.trace.append({'kind': 'lever_fire', 'why': 'nopack', 'lines': len(self.items), 'step': self.step})
        while self.msgs_free() > 0 and (not self.finished) and (self.left() > 6):
            if LEVER_ON and MLINE_ON and (not self.mline_done) and (not follow) and len(self.items) >= MLINE_AT:
                self.mline_done = True
                went = self.mline_ask(lanes, found, spent, used, sent, told)
                if went is False:
                    break
                if went:
                    continue
            if LEVER_ON and ASK1OA_ON and (not self.oattr1_done) and self.msg_used == 0 and len(self.items) >= 3:
                self.oattr1_done = True
                went = self.oattr_ask(found, spent, used, sent, told, True)
                self.trace.append({'kind': 'lever_fire', 'why': 'ask1oa', 'sent': int(bool(went)), 'step': self.step})
                if went is False:
                    break
                if went:
                    continue
            if LEVER_ON and OPENATTR_ON and (not self.oattr_done) and self.msg_used == 1 and len(self.items) >= 2:
                self.oattr_done = True
                went = self.oattr_ask(found, spent, used, sent, told)
                if went is False:
                    break
                if went:
                    continue
            pack = self.pack_ask(lanes, found, spent, used) if LEVER_ON and ALLLINES_ON and (not follow) and (not crowded) and (not solo) else []
            if pack:
                for number, labels in pack:
                    used[number].update((_words(label) for label in labels))
                batch = self.send([(pack[0][0], pack[0][1][0], ' '.join((self.ask_text(number, labels) for number, labels in pack)))])
                if not batch:
                    break
                sent += batch
                for number, _ in pack:
                    spent[number] += 1
                had = [len(item.told) for item in self.items]
                heard = self.hear([(', '.join((str(number) for number, _ in pack)), q, a) for _, q, a in batch], (LIGHT, HEAVY))
                self.take_heard(heard, told)
                fresh = [n for n, item in enumerate(self.items, 1) if len(item.told) > had[n - 1]]
                for n in fresh:
                    found[n] = found.get(n, 0) + 1
                calm = {min(max(1, _count(e.get('item')) or 1), len(self.items)) for e in _items(heard.get('free')) if isinstance(e, dict) and e.get('property')} if FREENEXT_ON else set()
                for number, labels in pack:
                    if number not in fresh and number not in calm:
                        used[number].difference_update((_words(label) for label in labels))
                crowded = not fresh and 1 in [_count(x) for x in _items(heard.get('crowded'))]
                self.trace.append({'kind': 'lever_fire', 'why': 'alllines', 'items': len(pack), 'fresh': len(fresh), 'crowded': crowded, 'calm': len(calm - set(fresh))})
                continue
            wait = self.follow_wait(follow, lanes, found, spent, used)
            near = {n: lanes[n] for n in lanes if found.get(n, 0) or not spent.get(n, 0)} if wait else lanes
            pick = None if (follow and (not wait)) or not (LEVER_ON and BROAD_ON) else self.broad_pick(near, found, spent, used)
            if pick:
                if not self.broad_send(pick, sent, spent, found, used, told):
                    break
                back = {_words(x) for x in self.broad[pick[0]].pop('back', [])}
                if back:
                    lane = lanes[pick[0]]
                    lanes[pick[0]] = [x for x in lane if _words(x) not in back] + [x for x in lane if _words(x) in back]
                continue
            if follow and (not wait):
                number, labels, shown = follow.pop(0)
            else:
                number, labels = self.next_ask(lanes, found, spent, used)
                shown = False
                if not labels:
                    break
                labels = self.need_labels(number, labels, lanes, used)
            text = ''
            if LEVER_ON and NEEDMULTI_ON and len(labels) > 1:
                text, labels = self.broad_text(number, labels)
            used[number].update((_words(label) for label in labels))
            batch = self.send([(number, labels[0], text or self.ask_text(number, labels, shown))])
            if not batch:
                break
            sent += batch
            spent[number] += 1
            had = [len(item.told) for item in self.items]
            heard = self.hear(batch, (LIGHT, HEAVY))
            self.take_heard(heard, told)
            fresh = [n for n, item in enumerate(self.items, 1) if len(item.told) > had[n - 1]]
            for n in fresh:
                found[n] = found.get(n, 0) + 1
            self.need_last = bool(fresh)
            if fresh or 1 not in [_count(x) for x in _items(heard.get('crowded'))]:
                continue
            if LEVER_ON and AFTERQ_ON and self.counter(batch[-1][2]):
                self.trace.append({'kind': 'lever_fire', 'why': 'afterq', 'path': 'narrow', 'step': self.step})
                continue
            if LEVER_ON and VALSPLIT_ON and any((not spent.get(n) for n in lanes)):
                self.trace.append({'kind': 'lever_fire', 'why': 'coverfirst', 'step': self.step})
                continue
            head = labels[0].split()[-1]
            if len(labels[0].split()) >= 2 and _words(head) not in used[number]:
                follow.insert(0, (number, [head], False))
            elif not shown and not (LEVER_ON and FORM572_ON) and len(self.samples(number, labels[0])) >= 2 and ((number, _words(labels[0])) not in self.chose):
                follow.insert(0, (number, labels, True))
        found_items = [n for n in found if found[n] > 0]
        if LEVER_ON and VALSPLIT_ON:
            searched = self.line_search()
        else:
            searched = self.value_search(found_items) if found_items else ''
        if told:
            self.version += 1
        if OWNDROP_ON and self.own_relax:
            self.own_relax = False
            self.relaxed = True
        relook = self.note_relook() if (HEARDROP_ON or OWNDROP_ON) and self.relaxed else ''
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
        if relook:
            summary += '\n' + relook
        return summary

    def line_open(self) -> None:
        


        key = self.category_key()
        shape = (self.search_template() or self.first_template()) if key else None
        runs, read = 0, 0
        for number, item in enumerate(self.items, 1):
            if not key or self.finished or self.left() < 12 + self.tail():
                break
            if self.records_of_item(number):
                continue
            pool = [i for i in self.seen_order if i not in self.full and self.buyable(i)
                    and self.owns(item, str((self.rows.get(i) or {}).get(key) or ''))]
            if not pool and shape is not None and runs < LINEOPEN_RUNS:
                tool, base, field = shape
                query = ' '.join(str(item.name or '').split()[:4])
                faults: list[str] = []
                args = _shape({**base, field: query}, self.schemas.get(tool), 'args', faults)
                if not query or faults:
                    continue
                reply = self.act(tool, args)
                if reply is None:
                    break
                runs += 1
                seen = reply.get('observation')
                self.observe(tool, seen)
                pool = [i for i, _ in self.records_in(seen) if i not in self.full and self.buyable(i)
                        and self.owns(item, str((self.rows.get(i) or {}).get(key) or ''))]
            read += len(self.fill_facts(pool[:2], self.tail() + 10))
        self.trace.append({'kind': 'lever_fire', 'why': 'lineopen', 'runs': runs, 'read': read, 'step': self.step})

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

    def first_template(self) -> tuple[str, dict[str, Any], str] | None:
        
        tool = self.roles.get('search')
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        named = [k for k in texts if re.search('(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword', k, re.I)]
        if not tool or not named or [k for k in (schema.get('required') or []) if k != named[0]]:
            return None
        return (tool, {}, named[0])

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

    def line_search(self) -> str:
        plan = self.search_template() or self.first_template()
        if not plan or self.left() < 10 + self.tail():
            self.trace.append({'kind': 'lever_fire', 'why': 'valsplit_skip', 'plan': bool(plan), 'left': self.left(), 'step': self.step})
            return ''
        tool, base, field = plan
        rank = {n: i for i, n in enumerate(self.plan_order)}
        order = sorted(range(1, len(self.items) + 1), key=lambda n: (rank.get(n, len(rank) + n), n))
        past = {_words(str(a.get(field) or '')) for a in self.searched + self.search_log}
        notes: list[str] = []
        runs = 0
        for number in order:
            if self.finished or self.left() < 8 + self.tail():
                break
            item = self.items[number - 1]
            plain = [v for v in self.value_words(number) if not re.search('\\d', v)]
            replied = [v for v in plain if any((':' in t and t.split(':', 1)[1].strip() == v for t in item.told))]
            spoken = [v for v in plain if v not in replied]
            name = str(item.name or '').split()
            words = ' '.join(name + replied + spoken).split()
            if len(words) > 4:
                words = ' '.join(name + replied).split()
            if len(words) > 4 and len(name) > 1:
                words = ' '.join(name[-1:] + replied).split()
            text = ' '.join(words[:5])
            if not text or _words(text) in past:
                continue
            past.add(_words(text))
            faults: list[str] = []
            args = _shape({**base, field: text}, self.schemas.get(tool), 'args', faults)
            if faults:
                break
            reply = self.act(tool, args)
            if reply is None:
                break
            runs += 1
            observation = reply.get('observation')
            self.observe(tool, observation)
            cat_key = self.category_key()
            listed = [i for i, _ in self.records_in(observation)]
            cap = self.budget if isinstance(self.budget, (int, float)) else None
            costs = [self.price_of(i) for i in listed]
            if cap is not None and costs and all((c is not None and c > 10 * cap for c in costs)):
                cap = None
            fit = [i for i in listed if self.buyable(i) and (not cat_key or self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))) and (cap is None or (self.price_of(i) or 0) <= cap)]
            if LEVER_ON and HEARDRANK_ON and len(fit) > 1:
                own = _roots(' '.join(name))
                said, asked = _roots(' '.join(replied)) - own, _roots(' '.join(spoken)) - own
                bag = {i: _roots(re.sub('[\\W_]+', ' ', _dump(self.rows.get(i) or {}))) for i in fit}
                fit.sort(key=lambda i: (-len(said & bag[i]), -len(asked & bag[i])))
                self.trace.append({'kind': 'lever_fire', 'why': 'heardrank', 'rows': len(fit), 'step': self.step})
            notes.append(f'item {number} {_dump(text)}: ' + (', '.join(fit[:6]) or 'no row of that kind'))
            unread = [i for i in fit if i not in self.full][:2 if self.left() >= 14 else 1]
            if unread:
                pages = self.fill_facts(unread, self.tail() + 8)
                if pages:
                    notes.append('opened: ' + ' | '.join(pages)[:2000])
        self.trace.append({'kind': 'lever_fire', 'why': 'valsplit', 'items': len(order), 'searches': runs, 'step': self.step})
        if not notes:
            return ''
        return "[Kyo searched every item by its name plus the values from the answers and read up to two fitting rows of each] " + '\n'.join(notes)

    def broad_keys(self, number: int, lane: list[str]) -> list[str]:
        cat_key = self.category_key()
        item = self.items[number - 1]
        rows = [r for r in self.full.values() if cat_key and self.owns(item, str(r.get(cat_key) or ''))]
        if not rows and len(self.items) == 1:
            rows = list(self.full.values())
        src = False
        if not rows and len(self.items) > 1 and LEVER_ON and SRCLINE:
            rows = [r for i, r in self.full.items() if self.found_for.get(i) == number]
            src = bool(rows)
        mixed = False
        if not rows and len(self.items) > 1 and LEVER_ON and ALLREC:
            rows = list(self.full.values())
            src = mixed = bool(rows)
        tally: dict[str, dict[str, int]] = {}
        for record in rows:
            for key, value in (_fact_block(record) or {}).items():
                bucket = tally.setdefault(_words(key), {})
                value_text = str(value).strip().lower()
                bucket[value_text] = bucket.get(value_text, 0) + 1
        floor = BROAD_SHARE * max(1, len(rows))
        ranked = []
        for key in self.lane_keys(number, self.key_table(), loose=src):
            bucket = tally.get(_words(key)) or {}
            total = sum(bucket.values())
            if total and total >= floor:
                ranked.append((max(bucket.values()) / total, -total, len(ranked), key))
        ranked.sort()
        if LEVER_ON and COUNTORDER:
            ranked.sort(key=lambda r: (r[1], r[2]))
        partial = 0
        if LEVER_ON and PARTIAL_ON:
            carriers = sum((1 for record in rows if _fact_block(record)))
            if carriers >= 2:
                some = [r for r in ranked if -r[1] < carriers]
                partial = len(some)
                if SOMEFIRST:
                    ranked = some + [r for r in ranked if r not in some]
        picked: list[str] = []
        head = list(lane)[:self.plan_picks.get(number, 0)] if LEVER_ON and WIDE12_ON else []
        order = head + [r[3] for r in ranked] + list(lane)
        if LEVER_ON and LABELSRC_ON:
            terms = self.line_terms(number)
            order = terms[:LABELSRC_TERMS] + [r[3] for r in ranked] + head + terms[LABELSRC_TERMS:] + list(lane)
            if PARTIAL_ON:
                order = [r[3] for r in ranked] + head + terms + list(lane)
            self.trace.append({'kind': 'lever_fire', 'why': 'labelsrc', 'line': number, 'terms': len(terms), 'keys': len(ranked), 'head': len(head), 'partial': partial, 'count': int(COUNTORDER), 'src': len(rows) if src else 0, 'mixed': int(mixed), 'step': self.step})
        for key in order:
            if _words(key) and not any(_words(key) == _words(other) for other in picked) and (not (LEVER_ON and LABELSRC_ON and self.heard_label(number, key))):
                picked.append(key)
        if LEVER_ON and PAGENAMES_ON and len(self.items) > 1:
            picked = self.name_pick(number, picked)
        return picked

    def page_names(self) -> dict[int, dict[str, int]]:
        
        cat_key = self.category_key()
        out: dict[int, dict[str, int]] = {}
        self.name_sheets: dict[int, list[set[str]]] = {}
        for number, item in enumerate(self.items, 1):
            pages = [r for r in self.full.values() if cat_key and self.owns(item, str(r.get(cat_key) or ''))]
            if not pages:
                pages = [r for i, r in self.full.items() if self.found_for.get(i) == number]
            names: dict[str, int] = {}
            for page in pages:
                rows = _page_rows(page)
                if rows:
                    self.name_sheets.setdefault(number, []).append(set(rows))
                for name, place in rows.items():
                    names[name] = min(place, names.get(name, place))
            if names:
                out[number] = names
        return out

    def mid_labels(self, number: int, used: dict[int, set[str]] | None) -> list[str]:
        

        cat_key = self.category_key()
        item = self.items[number - 1]
        pages = [r for r in self.full.values() if cat_key and self.owns(item, str(r.get(cat_key) or ''))]
        if not pages:
            pages = [r for i, r in self.full.items() if self.found_for.get(i) == number]
        if not pages and len(self.items) == 1:
            pages = list(self.full.values())
        tried = set(self.__dict__.setdefault('mid_tried', {}).get(number, set()))
        if used:
            tried |= used.get(number, set())
        sheets = [set(v) for n, v in self.page_names().items() if n != number]
        every = set.intersection(*sheets) if sheets else set()
        best: dict[str, int] = {}
        vals: dict[str, set[str]] = {}
        for page in pages:
            short = _mid_values(page) if LEVER_ON and MIDLABEL_SPREAD_ON else {}
            for name, place in _mid_names(page).items():
                if MIDLABEL_FROM <= place < MIDLABEL_TO:
                    best[name] = min(place, best.get(name, place))
                    if name in short:
                        vals.setdefault(name, set()).add(short[name])
        pool = sorted(best, key=lambda n: (-len(vals.get(n, ())), best[n] < MIDLABEL_CORE, best[n], n))
        return [n for n in pool if n not in tried and n not in every and not self.heard_label(number, n)][:MIDLABEL_N]

    def name_pick(self, number: int, labels: list[str]) -> list[str]:
        

        pages = self.page_names()
        own = pages.get(number) or {}
        if not own:
            return labels
        others = [set(v) for n, v in pages.items() if n != number]
        sheets = [s for n, v in self.name_sheets.items() if n != number for s in v] if PAGESHEET_ON else others
        every = set.intersection(*sheets) if sheets else set()
        shared = set().union(*sheets) if sheets else set()
        mine: list[str] = []
        rest: list[str] = []
        plain: list[str] = []
        head = cut = 0
        for label in labels:
            name = _words(label)
            if name not in own:
                plain.append(label)
            elif own[name] < PAGEHEAD_ROWS:
                head += 1
            elif name in every:
                cut += 1
            elif name in shared:
                rest.append(label)
            else:
                mine.append(label)
        kept = bool(mine or rest)
        self.trace.append({'kind': 'lever_fire', 'why': 'pagenames', 'line': number, 'own': len(own), 'others': len(others),
                           'mine': len(mine), 'rest': len(rest), 'plain': len(plain), 'head': head, 'cut': cut, 'sheets': len(sheets),
                           'kept': int(kept), 'step': self.step})
        if not kept:
            return labels
        return mine + rest + plain

    def sweep_order(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int]) -> list[int]:
        




        if LEVER_ON and NEVERANS_ON and not AFTERSWEEP_ON:
            return sorted(lanes, key=lambda n: (spent.get(n, 0) > 0, found.get(n, 0) > 0, spent.get(n, 0), n))
        return sorted(lanes, key=lambda n: (spent.get(n, 0) > 0, (found.get(n, 0) > 0) != (spent.get(n, 0) > 0), spent.get(n, 0), n))

    def follow_wait(self, follow: list[tuple[int, list[str], bool]], lanes: dict[int, list[str]], found: dict[int, int],
                    spent: dict[int, int], used: dict[int, set[str]]) -> bool:
        



        if follow and LEVER_ON and SWEEP_ON and NEVERANS_ON and len(self.items) > 1:
            other = [n for n in lanes if not found.get(n, 0) and any((_words(label) not in used.get(n, set()) for label in lanes[n]))]
            gone = [f for f in follow if found.get(f[0], 0)] if other else []
            if gone:
                follow[:] = [f for f in follow if not found.get(f[0], 0)]
                self.trace.append({'kind': 'lever_fire', 'why': 'neverans', 'path': 'follow', 'dropped': len(gone), 'step': self.step})
        if not (follow and LEVER_ON and SWEEP_ON and FOLLOWWAIT_ON and len(self.items) > 1):
            return False
        line = follow[0][0]
        if found.get(line, 0) or not spent.get(line, 0):
            return False
        ready = [n for n in lanes if n != line and ((found.get(n, 0) and (AFTERSWEEP_ON or not NEVERANS_ON)) or not spent.get(n, 0))
                 and any((_words(label) not in used.get(n, set()) for label in lanes[n]))]
        if ready:
            self.trace.append({'kind': 'lever_fire', 'why': 'followwait', 'ahead': len(ready), 'step': self.step})
        return bool(ready)

    def broad_pick(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int],
                   used: dict[int, set[str]] | None = None) -> tuple[int, list[str], str] | None:
        rank = {n: i for i, n in enumerate(self.plan_order)}
        stay = self.broad_stay if LEVER_ON and LINEROT_ON else None
        self.broad_stay = None
        order = sorted(lanes, key=lambda n: (n != stay, found.get(n, 0) > 0, spent.get(n, 0), rank.get(n, len(rank) + n)))
        if LEVER_ON and SWEEP_ON and len(self.items) > 1:
            swept = self.sweep_order(lanes, found, spent)
            if swept[:1] != order[:1] or stay is not None:
                self.trace.append({'kind': 'lever_fire', 'why': 'sweep', 'path': 'broad', 'stay': int(stay is not None),
                                   'unasked': sum((1 for n in lanes if not spent.get(n, 0))), 'step': self.step})
            order = swept
        for number in order:
            state = self.broad.get(number)
            if state is None:
                state = self.broad[number] = {'round': 0, 'queue': self.broad_keys(number, lanes[number])}
            if LEVER_ON and LABELSRC_ON:
                state['queue'] = [label for label in state['queue'] if not self.heard_label(number, label)]
            if LEVER_ON and MIDLABEL_ON and spent.get(number, 0) and not found.get(number, 0) and state['round'] < self.broad_rounds():
                mid = self.mid_labels(number, used)
                if len(mid) >= MIDLABEL_MIN:
                    text, labels = self.broad_text(number, mid)
                    self.trace.append({'kind': 'lever_fire', 'why': 'midlabel', 'line': number, 'n': len(labels), 'step': self.step})
                    return (number, labels, text)
            if state['round'] < self.broad_rounds() and len(state['queue']) >= 2:
                text, labels = self.broad_text(number, state['queue'][:BROAD_LABELS])
                return (number, labels, text)
            if (LEVER_ON and SWEEP_ON and BROADYIELD_ON and used is not None and len(self.items) > 1 and (not spent.get(number, 0))
                    and any((_words(label) not in used.get(number, set()) for label in lanes[number]))):
                self.trace.append({'kind': 'lever_fire', 'why': 'broadyield', 'unasked': sum((1 for n in lanes if not spent.get(n, 0))),
                                   'step': self.step})
                return None
        return None

    def line_terms(self, number: int) -> list[str]:
        
        cat_key = self.category_key()
        item = self.items[number - 1]
        rows = [r for r in self.rows.values() if cat_key and self.owns(item, str(r.get(cat_key) or ''))]
        if not rows and len(self.items) == 1:
            rows = list(self.rows.values())
        if not rows and len(self.items) > 1 and LEVER_ON and SRCLINE2:
            rows = [r for i, r in self.rows.items() if self.found_for.get(i) == number]
        if len(rows) < 3:
            return []
        skip = _roots(self.request) | _roots(item.name) | _roots(item.shelf)
        shown: dict[str, dict[str, int]] = {}

        def tally(group: list[dict[str, Any]], show: bool) -> dict[str, int]:
            nonlocal skip
            count: dict[str, int] = {}
            for row in group:
                words: set[str] = set()
                for key, value in row.items():
                    skip |= _roots(str(key))
                    if not isinstance(value, str):
                        continue
                    if len(value.split()) <= 3:
                        skip |= _roots(value)
                        continue
                    for word in re.findall('[a-z]{3,}', value.lower()):
                        base = next(iter(_roots(word)), '')
                        if base:
                            words.add(base)
                            if show:
                                forms = shown.setdefault(base, {})
                                forms[word] = forms.get(word, 0) + 1
                for base in words:
                    count[base] = count.get(base, 0) + 1
            return count
        count = tally(rows, True)
        size = len(rows)
        keep = [base for base, df in count.items() if df >= 2 and base not in skip]
        keep.sort(key=lambda base: (count[base] >= size, abs(2 * count[base] - size), -count[base], base))
        return [max(shown[base].items(), key=lambda kv: (kv[1], kv[0]))[0] for base in keep[:2 * BROAD_LABELS]]

    def heard_label(self, number: int, label: str) -> bool:
        
        item = self.items[number - 1]
        heard = [m for m in item.musts + item.told if m not in item.dropped]
        if any((_names(_stem(m.split(':', 1)[0]), _stem(label)) for m in heard)):
            return True
        roots = _roots(label)
        if not roots:
            return False
        values = set().union(*(_roots(m.partition(':')[2]) for m in heard))
        nouns = _roots(item.name) | _roots(item.shelf)
        for args in self.searched + self.search_log:
            said = set().union(*(_roots(v) for v in args.values() if isinstance(v, str)))
            if said & _roots(item.name):
                nouns |= said
        return roots <= values or roots <= nouns

    def broad_rounds(self) -> int:
        return max(BROAD_ROUNDS, LINEROT_ROUNDS) if LEVER_ON and LINEROT_ON else BROAD_ROUNDS

    def broad_text(self, number: int, labels: list[str]) -> tuple[str, list[str]]:
        props = (self.schemas.get(self.roles.get('ask') or '') or {}).get('properties') or {}
        caps = [v['maxLength'] for v in props.values() if isinstance(v, dict) and isinstance(v.get('maxLength'), int)]
        line = self.line_name(number)
        names = list(labels)
        while True:
            if len(names) == 1:
                text = (f"What {names[0]} would you like for the {line}?" if LEVER_ON and WHATWIDE_ON
                        else f"For the {line}, what {names[0]} would you like?")
            elif LEVER_ON and WHATWIDE_ON:
                text = f"What {', '.join(names[:-1])} or {names[-1]} would you like for the {line}?"
            else:
                text = f"For the {line}, what {', '.join(names[:-1])} or {names[-1]} would you like?"
            if len(names) <= 2 or not caps or len(text) <= min(caps):
                return (text, names)
            names = names[:-1]

    def broad_send(self, pick: tuple[int, list[str], str], sent: list[tuple[int, str, str]], spent: dict[int, int], found: dict[int, int], used: dict[int, set[str]], told: list[str]) -> bool:
        number, labels, text = pick
        if LEVER_ON and MIDLABEL_ON:
            self.__dict__.setdefault('mid_tried', {}).setdefault(number, set()).update(_words(label) for label in labels)
        batch = self.send([(number, labels[0], text)])
        if not batch:
            return False
        sent.extend(batch)
        spent[number] += 1
        said = _roots(batch[-1][2])
        hit = [label for label in labels if _roots(label) and _roots(label) <= said]
        used[number].update(_words(label) for label in hit)
        state = self.broad[number]
        state['round'] += 1
        if state['round'] == 1 and (not (LEVER_ON and LINEROT_ON)):
            cut = labels.index(hit[-1]) + 1 if hit else min(2, len(labels))
            state['queue'] = [label for label in state['queue'] if label not in labels[:cut]]
            if LEVER_ON and WIDE12_ON:
                state['queue'] = [label for label in state['queue'] if label not in labels] + [label for label in labels[cut:] if label in state['queue']]
        self.trace.append({'kind': 'lever_fire', 'why': 'broad%d' % state['round'], 'n': len(labels), 'read': len(hit), 'step': self.step})
        had = [len(item.told) for item in self.items]
        self.take_heard(self.hear(batch, (LIGHT, HEAVY)), told)
        for n, item in enumerate(self.items, 1):
            if len(item.told) > had[n - 1]:
                found[n] = found.get(n, 0) + 1
        if LEVER_ON and LINEROT_ON:
            mine = len(self.items[number - 1].told) > had[number - 1]
            cut = labels.index(hit[-1]) + 1 if mine and hit else 1
            state['queue'] = [label for label in state['queue'] if label not in labels[:cut]]
            quiet = not any((len(item.told) > had[n] for n, item in enumerate(self.items)))
            if quiet and (not self.counter(batch[-1][2])):
                self.broad_stay = number
            self.trace.append({'kind': 'lever_fire', 'why': 'linerot', 'mine': mine, 'cut': cut, 'stay': self.broad_stay == number, 'left': len(state['queue']), 'step': self.step})
        if LEVER_ON and AFTERQ_ON and len(self.items[number - 1].told) == had[number - 1] and self.counter(batch[-1][2]):
            state['round'] = self.broad_rounds()
            state['back'] = list(labels)
            self.trace.append({'kind': 'lever_fire', 'why': 'afterq', 'path': 'broad', 'n': len(labels), 'step': self.step})
        if LEVER_ON and BROADSTOP_ON and state['round'] < self.broad_rounds() and len(self.items[number - 1].told) == had[number - 1]:
            state['round'] = self.broad_rounds()
            self.trace.append({'kind': 'lever_fire', 'why': 'broadstop', 'n': len(labels), 'step': self.step})
        return True

    def mline_ask(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]], sent: list[tuple[int, str, str]], told: list[str]) -> bool | None:
        per = 2 if 2 * len(self.items) <= BROAD_LABELS else 1
        parts: list[tuple[int, list[str]]] = []
        for number in sorted(lanes):
            state = self.broad.get(number)
            if state is None:
                state = self.broad[number] = {'round': 0, 'queue': self.broad_keys(number, lanes[number])}
            if state['queue']:
                parts.append((number, state['queue'][:per]))
        if len(parts) < 2:
            return None
        text, parts = self.mline_text(parts)
        if not text:
            return None
        for number, labels in parts:
            used[number].update((_words(label) for label in labels))
        batch = self.send([(parts[0][0], parts[0][1][0], text)])
        if not batch:
            return False
        sent.extend(batch)
        had = [len(item.told) for item in self.items]
        heard = self.hear([(', '.join((str(number) for number, _ in parts)), q, a) for _, q, a in batch], (LIGHT, HEAVY))
        self.take_heard(heard, told)
        fresh = [n for n, item in enumerate(self.items, 1) if len(item.told) > had[n - 1]]
        for n in fresh:
            found[n] = found.get(n, 0) + 1
        for number, labels in parts:
            spent[number] += 1
            state = self.broad[number]
            state['queue'] = [label for label in state['queue'] if label not in labels] + ([] if number in fresh else list(labels))
            if number not in fresh:
                used[number].difference_update((_words(label) for label in labels))
        self.trace.append({'kind': 'lever_fire', 'why': 'mline', 'lines': len(parts), 'labels': sum((len(x) for _, x in parts)), 'fresh': len(fresh), 'step': self.step})
        return True

    def mline_text(self, parts: list[tuple[int, list[str]]]) -> tuple[str, list[tuple[int, list[str]]]]:
        props = (self.schemas.get(self.roles.get('ask') or '') or {}).get('properties') or {}
        caps = [v['maxLength'] for v in props.values() if isinstance(v, dict) and isinstance(v.get('maxLength'), int)]
        cap = min(caps) if caps else 0
        while True:
            segs = [(' or '.join(labels), self.items[number - 1].name) for number, labels in parts]
            rest = [f'{what} for the {line}' for what, line in segs[1:]]
            text = f'What {segs[0][0]} would you like for the {segs[0][1]}'
            text += (', ' + ', '.join(rest[:-1]) + ',' if len(rest) > 1 else '') + f' and {rest[-1]}?'
            if not cap or len(text) <= cap:
                return (text, parts)
            if any((len(labels) > 1 for _, labels in parts)):
                parts = [(number, labels[:1]) for number, labels in parts]
            elif len(parts) > 2:
                parts = parts[:-1]
            else:
                return ('', parts)

    def buy_push(self) -> bool:
        if LEVER_ON and DECL2_ON:
            return self.reveal_events < DECL2_AT
        return bool(self.revealed())

    def counter(self, reply: str) -> bool:
        return str(reply or '').rstrip(' \n\'")').endswith('?')

    def open_attrs(self, used: dict[int, set[str]], lead_only: bool = False) -> list[tuple[int, list[str]]]:
        

        opened = [i for i in self.seen_order if self.full.get(i)]
        facts: dict[str, dict[str, str]] = {}
        for ident in opened:
            row = facts.setdefault(ident, {})
            for key, value in (_fact_block(self.full[ident]) or {}).items():
                name = _words(key)
                if name and name not in row:
                    row[name] = str(value).strip().lower()
        count = len(self.items)
        mine = {n: [i for i in self.item_ids(n) if i in facts] for n in range(1, count + 1)}
        owner: dict[str, int] = {}
        for n in range(1, count + 1):
            for ident in mine[n]:
                owner.setdefault(ident, n)

        def carry_of(n: int) -> dict[str, dict[str, str]]:
            carry: dict[str, dict[str, str]] = {}
            for ident in mine[n]:
                for name, value in facts[ident].items():
                    carry.setdefault(name, {})[ident] = value
            return carry

        def fit(n: int, name: str, per: dict[str, str]) -> bool:
            values = list(per.values())
            return (len(per) >= 2 and len(set(values)) >= 2 and sum(len(v) for v in values) <= 40 * len(values)
                    and name not in used.get(n, set()) and not self.heard_label(n, name))

        if count >= 3:
            lead = next((owner[i] for i in list(self.open_order) + opened if i in owner), 0)
            if lead:
                others = [set(carry_of(m)) for m in range(1, count + 1) if m != lead and mine[m]]
                shared = set.intersection(*others) if others else set()
                ranks = [(-len(per), k, name) for k, (name, per) in enumerate(carry_of(lead).items())
                         if name not in shared and fit(lead, name, per)]
                names = [r[-1] for r in sorted(ranks)][:OPENATTR_PER]
                if names:
                    self.oattr_loose = 0
                    return [(lead, names)]
        if lead_only:
            return []
        first = {n: k for k, n in enumerate(self.plan_order)}
        order = sorted(range(1, count + 1), key=lambda n: (len(self.items[n - 1].told) > 0, first.get(n, count + n)))
        if count == 2 and NAME2_ON:
            head = next((owner[i] for i in list(self.open_order) + opened if i in owner), 0)
            two = sorted(order, key=lambda n: n != head) if head else order
            keep: dict[int, list[str]] = {}
            spare: dict[int, list[str]] = {}
            for n in two:
                other = set(carry_of(3 - n)) if mine[3 - n] else set()
                ranks = sorted(((NUMLAST_ON and _numfield(per), -len(set(per.values())), -len(per), k, name)
                                for k, (name, per) in enumerate(carry_of(n).items()) if fit(n, name, per)))
                keep[n] = [r[-1] for r in ranks if r[-1] not in other]
                spare[n] = [r[-1] for r in ranks if r[-1] in other]
            pairs: dict[int, list[str]] = {n: [] for n in two}
            for lists, share in ((keep, (OPENATTR_PER, OPENATTR_PER)), (spare, (2, 1)), (spare, (OPENATTR_PER, OPENATTR_PER))):
                depth = 0
                while sum((len(p) for p in pairs.values())) < OPENATTR_PER and any((len(lists[n]) > depth for n in two)):
                    for k, n in enumerate(two):
                        got = pairs[n]
                        if (len(lists[n]) > depth and len(got) < share[min(k, 1)] and lists[n][depth] not in got
                                and sum((len(p) for p in pairs.values())) < OPENATTR_PER):
                            got.append(lists[n][depth])
                    depth += 1
            total2 = sum((len(p) for p in pairs.values()))
            self.trace.append({'kind': 'lever_fire', 'why': 'name2', 'kept': sum((len(keep[n]) for n in two)),
                               'names': total2, 'step': self.step})
            if total2:
                self.oattr_loose = 0
                return [(n, pairs[n]) for n in two if pairs[n]]
        picks: dict[int, list[str]] = {}
        for n in order:
            ranks = []
            for k, (name, per) in enumerate(carry_of(n).items()):
                if fit(n, name, per):
                    kinds = len(set(per.values()))
                    ranks.append((len(per) >= 3 and kinds == len(per), -kinds, -len(per), k, name))
            if ranks:
                picks[n] = [r[-1] for r in sorted(ranks)]
        groups: dict[int, list[str]] = {}
        total, depth = 0, 0
        while total < OPENATTR_PER and any(len(p) > depth for p in picks.values()):
            for n in order:
                if total < OPENATTR_PER and len(picks.get(n, [])) > depth:
                    groups.setdefault(n, []).append(picks[n][depth])
                    total += 1
            depth += 1
        if groups:
            self.oattr_loose = 0
            return [(n, groups[n]) for n in order if n in groups]
        pack, total = [], 0
        for n, labels in self.open_attrs_all(used):
            keep = labels[:max(0, OPENATTR_PER - total)]
            if keep:
                pack.append((n, keep))
                total += len(keep)
        return pack

    def open_attrs_all(self, used: dict[int, set[str]]) -> list[tuple[int, list[str]]]:
        
        opened = [i for i in self.seen_order if self.full.get(i)]
        carry: dict[str, dict[str, str]] = {}
        for ident in opened:
            for key, value in (_fact_block(self.full[ident]) or {}).items():
                name = _words(key)
                if name and ident not in carry.setdefault(name, {}):
                    carry[name][ident] = str(value).strip().lower()
        asked = set()
        for labels in used.values():
            asked |= set(labels)
        picks = [name for name, per in carry.items() if 2 <= len(per) < len(opened) and len(set(per.values())) > 1
                 and name not in asked]
        picks.sort(key=lambda name: -len(carry[name]))
        picks = picks[:OPENATTR_TOP]
        owned = {n: set(self.item_ids(n)) for n in range(1, len(self.items) + 1)}
        first = {n: i for i, n in enumerate(self.plan_order)}
        order = sorted(owned, key=lambda n: (len(self.items[n - 1].told) > 0, first.get(n, len(first) + n)))
        groups: dict[int, list[str]] = {}
        loose = 0
        for name in picks:
            lines = [n for n in order if owned[n] & set(carry[name]) and not self.heard_label(n, name)]
            if not lines and not any(owned[n] & set(carry[name]) for n in order):
                lines = order[:1]
                loose += 1
            for n in lines:
                groups.setdefault(n, []).append(name)
        self.oattr_loose = loose
        return [(n, groups[n]) for n in order if n in groups]

    def oattr_ask(self, found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]], sent: list[tuple[int, str, str]], told: list[str], lead_only: bool = False) -> bool | None:
        pack = self.open_attrs(used, lead_only)
        if not pack:
            self.trace.append({'kind': 'lever_fire', 'why': 'openattr', 'labels': 0, 'step': self.step})
            return None
        props = (self.schemas.get(self.roles.get('ask') or '') or {}).get('properties') or {}
        caps = [v['maxLength'] for v in props.values() if isinstance(v, dict) and isinstance(v.get('maxLength'), int)]
        while True:
            text = ' '.join(self.broad_text(n, labels)[0] for n, labels in pack)
            names = sum(len(labels) for _, labels in pack)
            if not caps or len(text) <= min(caps) or names <= 1:
                break
            most = max(range(len(pack)), key=lambda k: (len(pack[k][1]), k))
            n, labels = pack[most]
            pack = pack[:most] + ([(n, labels[:-1])] if len(labels) > 1 else []) + pack[most + 1:]
        for n, labels in pack:
            used[n].update(_words(label) for label in labels)
        batch = self.send([(pack[0][0], pack[0][1][0], text)])
        if not batch:
            return False
        sent += batch
        for n, _ in pack:
            spent[n] += 1
        had = [len(item.told) for item in self.items]
        heard = self.hear([(', '.join(str(n) for n, _ in pack), q, a) for _, q, a in batch], (LIGHT, HEAVY))
        self.take_heard(heard, told)
        fresh = [n for n, item in enumerate(self.items, 1) if len(item.told) > had[n - 1]]
        for n in fresh:
            found[n] = found.get(n, 0) + 1
        self.need_last = bool(fresh)
        self.trace.append({'kind': 'lever_fire', 'why': 'openattr', 'labels': names, 'lines': len(pack),
                           'loose': self.oattr_loose, 'fresh': len(fresh), 'step': self.step})
        return True

    def need_labels(self, number: int, labels: list[str], lanes: dict[int, list[str]], used: dict[int, set[str]]) -> list[str]:
        


        if not (LEVER_ON and NEEDMULTI_ON) or len(labels) != 1 or _digits(labels[0]):
            return labels
        if NEEDONE1_ON and len(self.items) == 1:
            self.trace.append({'kind': 'lever_fire', 'why': 'needone1', 'line': number, 'step': self.step})
            return labels
        if len(self.items) != 2 and self.need_last is False:
            self.trace.append({'kind': 'lever_fire', 'why': 'needone', 'line': number, 'step': self.step})
            return labels
        head = _words(labels[0])
        extra = [x for x in lanes.get(number, []) if _words(x) != head and _words(x) not in used.get(number, set())
                 and not _digits(x) and not self.heard_label(number, x)]
        pick = labels + extra[:max(0, NEED_K - 1)]
        self.trace.append({'kind': 'lever_fire', 'why': 'needmulti', 'line': number, 'n': len(pick),
                           'lines': len(self.items), 'after': -1 if self.need_last is None else int(self.need_last),
                           'step': self.step})
        return pick

    def next_ask(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> tuple[int, list[str]]:
        first = {n: i for i, n in enumerate(self.plan_order)}
        order = sorted(lanes, key=lambda n: (found.get(n, 0) > 0, spent.get(n, 0), first.get(n, len(first) + n)))
        if LEVER_ON and SWEEP_ON and len(self.items) > 1:
            swept = self.sweep_order(lanes, found, spent)
            ready = [n for n in order if any((_words(label) not in used[n] for label in lanes[n]))]
            if ready and [n for n in swept if n in ready][0] != ready[0]:
                self.trace.append({'kind': 'lever_fire', 'why': 'sweep', 'path': 'narrow', 'stay': 0,
                                   'unasked': sum((1 for n in lanes if not spent.get(n, 0))), 'step': self.step})
            order = swept
            if NEVERANS_ON and AFTERSWEEP_ON and ready:
                told = [n for n in sorted(lanes, key=lambda n: (spent.get(n, 0) > 0, found.get(n, 0) > 0, spent.get(n, 0), n)) if n in ready][0]
                if told != [n for n in swept if n in ready][0]:
                    self.trace.append({'kind': 'lever_fire', 'why': 'aftersweep', 'path': 'narrow', 'step': self.step})
            if NEVERANS_ON and ready:
                was = [n for n in sorted(lanes, key=lambda n: (spent.get(n, 0) > 0, (found.get(n, 0) > 0) != (spent.get(n, 0) > 0), spent.get(n, 0), n)) if n in ready][0]
                now = [n for n in swept if n in ready][0]
                if was != now:
                    self.trace.append({'kind': 'lever_fire', 'why': 'neverans', 'path': 'narrow', 'step': self.step})
        for number in order:
            left = [label for label in lanes[number] if _words(label) not in used[number]]
            if left:
                return (number, left[:1])
        return (0, [])

    def pack_ask(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> list[tuple[int, list[str]]]:
        free = self.msgs_free()
        if free <= 0:
            return []
        order = {n: i for i, n in enumerate(self.plan_order)}
        picks: list[tuple[int, list[str]]] = []
        for number in sorted(lanes, key=lambda n: order.get(n, len(order) + n)):
            if spent.get(number, 0) or found.get(number, 0):
                continue
            fresh = [label for label in lanes[number] if _words(label) not in used[number]]
            if fresh:
                picks.append((number, [fresh[0]]))
        return picks if len(picks) > free else []

    def ask_text(self, number: int, labels: list[str], shown: bool=False) -> str:
        first = labels[0]
        if len(labels) > 1:
            return self.plain_ask(number, lambda line: 'What {} or {} would you like for the {}?'.format(first, labels[1], line))
        return self.question(number, first, self.samples(number, first) if shown else None)

    def lane_keys(self, number: int, table: list[tuple[str, str, int, list[str]]], loose: bool=False) -> list[str]:
        item = self.items[number - 1]
        settled = [_stem(m.split(':', 1)[0]) for m in item.musts]
        rows = [(label, key, carried, len(values)) for label, key, carried, values in table if self.owner_of(label) == number]
        measured = {key: sum((1 for v in values if re.match('^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$', v))) * 2 > len(values) for _, key, _, values in table}
        if (len(self.items) == 1 or loose) and (not rows):
            rows = [(label, key, carried, len(values)) for label, key, carried, values in table]
        rows.sort(key=lambda r: (-r[2], measured.get(r[1], False), -r[3]))
        keys = [key for _, key, _, _ in rows if not any((_names(prop, _stem(key)) for prop in settled)) and (not re.search(NOT_PROPERTY, key))]
        return [key for key in keys if not re.search(CATCH_ALL, key)] + [key for key in keys if re.search(CATCH_ALL, key)]

    def owner_of(self, label: str) -> int:
        best = self.owner_score(label)
        lone = LEVER_ON and ONEKIND_ON and len(self.items) == 1
        if best or not label or not lone or self.any_owner():
            return best
        if not getattr(self, 'onekind_fired', False):
            self.onekind_fired = True
            self.trace.append({'kind': 'lever_fire', 'why': 'onekind', 'rows': len(self.rows), 'step': self.step})
        return 1

    def any_owner(self) -> bool:
        mark = (len(self.rows), len(self.full))
        if getattr(self, 'owner_mark', None) == mark:
            return self.owner_any
        key = self.category_key()
        hit = False
        if key:
            for row in self.rows.values():
                label = str(row.get(key) or '').strip()
                if label and self.owner_score(label):
                    hit = True
                    break
        self.owner_mark, self.owner_any = (mark, hit)
        return hit

    def owner_score(self, label: str) -> int:
        if not label:
            return 0
        plain, stems = _words(label), _roots(label)

        def fit(item: Item) -> float:
            if plain == _words(item.shelf or ''):
                return 3.0
            if plain == _words(item.name):
                return 2.0
            own = _roots(item.shelf or '') | _roots(item.name)
            return len(stems & own) / max(1, len(stems | own))
        scores = [fit(item) for item in self.items]
        top = max(scores, default=0.0)
        return scores.index(top) + 1 if top > 0 else 0

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
            listing.append(f'  {index}. {name} (on {carried}): ' + ' | '.join(values) + self.not_stated(label, name, values) + (' [differs across opened records]' if LEVER_ON and NEEDDIFF_ON and len(set(values)) >= 2 else '') + (' [several features in one: ask last]' if re.search(CATCH_ALL, name) else ''))
        prompt = PLAN.replace('COUNT', str(max(1, count))) + '\n\nREQUEST:\n' + self.request + '\n\nITEMS:\n' + '\n'.join(settled)
        first = [str(x) for x in _items(self.brief.get('ask_first')) if str(x).strip()]
        if first:
            prompt += '\n\nASK FIRST (the request asks to be asked about these):\n' + '\n'.join(first)
        prompt += '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(listing) or '(no full listing records yet)')
        if LEVER_ON and IDPROP_ON and table:
            prompt += '\n\n' + IDPROP
        if LEVER_ON and NEEDDIFF_ON and table:
            prompt += '\n\n' + NEEDDIFF
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
            self.plan_picks[number] = len(lane)
            for key in keys:
                if all((_words(key) != _words(other) for other in lanes[number])):
                    lanes[number].append(key)
            lanes[number] = [x for x in lanes[number] if not re.search(CATCH_ALL, _words(x))] + [x for x in lanes[number] if re.search(CATCH_ALL, _words(x))]
        self.trace.append({'kind': 'plan', 'ask': _clip(plan.get('ask'), 120), 'lanes': {n: v[:8] for n, v in lanes.items()}})
        return lanes

    def question(self, number: int, label: str, samples: list[str] | None=None) -> str:
        if LEVER_ON and FORM572_ON:
            return self.short_ask(number, label)
        if samples and LEVER_ON and PLAINASK_ON and any((_digits(s) for s in samples[:2])):
            self.trace.append({'kind': 'lever_fire', 'why': 'plainask', 'part': 'example', 'step': self.step})
            samples = None
        tail = f', for example {samples[0]} or {samples[1]}?' if samples else '?'
        return self.plain_ask(number, lambda line: f'What {label} would you like for the {line}' + tail)

    def short_ask(self, number: int, label: str) -> str:
        


        def clean(text: str) -> list[str]:
            return [t for t in str(text or '').split() if not _digits(t)]
        what = clean(label)[-FORM_LABEL:]
        name = clean(self.items[number - 1].name)
        room = FORM_TERMS - 6 - len(what)
        name = name[-room:] if room > 0 else []
        target = 'the ' + ' '.join(name) if name else 'it'
        text = 'What {} would you like for {}?'.format(' '.join(what), target) if what else 'What would you like for {}?'.format(target)
        text = ' '.join(text.split())
        short = len(text.split()) < FORM_MIN
        if short:
            text = 'What {} would you like me to pick for {}?'.format(' '.join(what), target) if what else 'What would you like me to pick for {}?'.format(target)
            text = ' '.join(text.split())
        if short or what != str(label or '').split() or name != str(self.items[number - 1].name or '').split():
            self.trace.append({'kind': 'lever_fire', 'why': 'form572', 'terms': len(text.split()), 'pad': int(short), 'step': self.step})
        return text

    def plain_ask(self, number: int, form: Any) -> str:
        cap = PLAIN_WORDS if LEVER_ON and PLAINASK_ON else 0
        depth = 0
        text = form(self.line_name(number))
        while cap and len(text.split()) >= cap and depth < 2:
            depth += 1
            text = form(self.line_name(number, depth))
        if depth:
            self.trace.append({'kind': 'lever_fire', 'why': 'plainask', 'part': 'line', 'depth': depth, 'words': len(text.split()), 'step': self.step})
        return text

    def line_name(self, number: int, depth: int=0) -> str:
        item = self.items[number - 1]
        if not (LEVER_ON and KEYASK_ON) or depth >= 2:
            return item.name
        tags, covered = self.line_tags(number, item)
        extra = self.line_values(number, item, covered) if depth == 0 else []
        if not tags and not extra:
            return item.name
        self.trace.append({'kind': 'lever_fire', 'why': 'keyask', 'labels': len(tags), 'extra': len(extra), 'step': self.step})
        inside = ([' or '.join(tags)] if tags else []) + ([', '.join(extra)] if extra else [])
        return f"{item.name} ({'; '.join(inside)})"

    def line_tags(self, number: int, item: Item) -> tuple[list[str], set[str]]:
        covered = set(_roots(item.name))
        base = _words(item.name)
        tags: list[str] = []
        for label in self.store_labels(number):
            norm = _words(label)
            if LEVER_ON and PLAINASK_ON and _digits(label):
                continue
            if _roots(label) and norm not in base and all(norm != _words(x) for x in tags):
                tags.append(label)
                covered |= _roots(label)
                if len(tags) == 2:
                    break
        return (tags, covered)

    def line_values(self, number: int, item: Item, covered: set[str]) -> list[str]:
        stated = {m.split(':', 1)[1].strip() for m in item.musts if ':' in m}
        picked: list[str] = []
        for value in self.value_words(number):
            stems = _roots(value)
            if value not in stated or not stems or stems <= covered or (LEVER_ON and PLAINASK_ON and _digits(value)):
                continue
            if len(' '.join(picked + [value]).split()) > 4:
                continue
            picked.append(value)
            covered |= stems
            if len(picked) == 2:
                break
        return picked

    def store_labels(self, number: int) -> list[str]:
        item = self.items[number - 1]
        key = self.category_key()
        counts: dict[str, int] = {}
        if key:
            for row in self.rows.values():
                label = str(row.get(key) or '').strip()
                if label and len(label) <= 60 and self.owns(item, label):
                    counts[label] = counts.get(label, 0) + 1
        shelf = (item.shelf or '').strip()
        home = _words(shelf)
        ordered = sorted(counts, key=lambda label: -counts[label])
        front = [label for label in ordered if _words(label) == home]
        ranked = front + [label for label in ordered if _words(label) != home]
        if shelf and len(shelf) <= 60 and not front:
            ranked.append(shelf)
        return ranked

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
            self.qa.append((text, words))
            self.done_log.append(f'step {self.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {self.cart_mark()})')
            self.done_steps.append(self.step)
            self.trace.append({'kind': 'ask', 'q': text, 'a': words[:300]})
            if isinstance(answer, dict) and answer.get('error'):
                break
        return sent

    def take_heard(self, heard: dict[str, Any], told: list[str]) -> None:
        many = len(self.items) > 1
        start = sum((len(it.told) for it in self.items))
        for entry in _items(heard.get('needs')):
            if not isinstance(entry, dict) or not str(entry.get('need') or '').strip():
                continue
            number = min(max(1, _count(entry.get('item')) or 1), len(self.items))
            need = str(entry['need']).strip()
            if LEVER_ON and EDGECHEAP_ON:
                self.edge_cheap(entry, need)
            if need not in self.items[number - 1].told:
                self.items[number - 1].told.append(need)
                told.append(f'item {number}: {need}' if many else need)
                if LEVER_ON and CARTLOOK_ON:
                    self.look_told.append(f'item {number}: {need}' if many else need)
            if entry.get('points') is True and (number, need) not in self.refs:
                self.refs.append((number, need))
        for entry in _items(heard.get('drop')):
            if isinstance(entry, dict) and str(entry.get('property') or '').strip():
                item = self.items[min(max(1, _count(entry.get('item')) or 1), len(self.items)) - 1]
                gone = _words(str(entry['property']).split(':')[0])
                hit = [m for m in item.musts + item.told if m not in item.dropped and _words(m.split(':')[0]) == gone]
                item.dropped.extend(hit)
                told.extend((f'dropped {m}' for m in hit))
                if hit and HEARDROP_ON:
                    self.relaxed = True
        for entry in _items(heard.get('free')):
            if isinstance(entry, dict) and entry.get('property'):
                number = min(max(1, _count(entry.get('item')) or 1), len(self.items))
                self.items[number - 1].free.append(str(entry['property']))
        if isinstance(heard.get('budget'), (int, float)) and (not isinstance(heard.get('budget'), bool)) and (heard['budget'] > 0):
            if HEARDROP_ON and isinstance(self.budget, (int, float)) and (not isinstance(self.budget, bool)) and (heard['budget'] > self.budget):
                self.relaxed = True
            self.budget = heard['budget']
        if sum((len(it.told) for it in self.items)) > start:
            self.reveal_events += 1

    def edge_cheap(self, entry: dict[str, Any], need: str) -> None:
        edge = str(entry.get('edge') or '').strip().lower()
        if edge not in ('min', 'max') or entry.get('points') is True or not _digits(need):
            return
        if self.brief.get('pick') in ('cheapest', 'most', 'least'):
            return
        self.brief['pick'] = 'cheapest'
        self.trace.append({'kind': 'lever_fire', 'why': 'edgecheap', 'edge': edge, 'step': self.step})

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(HEAVY, LIGHT)) -> dict[str, Any]:
        rules = HEAR.replace('else false}', 'else false' + EDGE_NOTE + '}', 1) if LEVER_ON and EDGECHEAP_ON else HEAR
        prompt = rules + '\n\nITEMS:\n' + '\n'.join((f'item {n}: {it.name}' for n, it in enumerate(self.items, 1)))
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f"{i}. (about item{('s' if isinstance(n, str) else '')} {n}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1)))
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
        self.relaxed = False
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
            if isinstance(self.budget, (int, float)) and budget > self.budget:
                self.relaxed = True
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
            self.relaxed = True
        for entry in adds:
            item = self.items[min(max(1, _count(entry.get('item')) or 1), len(self.items)) - 1]
            need = str(entry['need']).strip()
            if need in item.dropped:
                item.dropped.remove(need)
            if need not in item.musts + item.told:
                item.told.append(need)
            added.add(need)
            changed.append(f'added {need}')
            if self.note_leads and self.leads == self.note_leads:
                self.leads, self.lead_items, self.note_leads = ([], {}, [])
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

    def cart_look(self) -> str:
        news, self.look_told = (self.look_told, [])
        tool = self.facts_tool() or self.offer_tool()
        key = self.single_target(tool) if tool else None
        live = [i for i in self.cart_ids() if not self.sold_out(i)][:CARTLOOK_LINES]
        if not key or not live or self.looks >= CARTLOOK_FIRES:
            return ''
        if self.left() < len(live) + 4:
            self.trace.append({'kind': 'lever_fire', 'why': 'cartlook', 'opened': 0, 'lines': len(live), 'short': 1, 'step': self.step})
            return ''
        self.looks += 1
        pages: list[str] = []
        for ident in live:
            faults: list[str] = []
            args = _shape({key: ident}, self.schemas.get(tool), 'args', faults)
            if faults:
                break
            reply = self.act(tool, args)
            if reply is None or self.finished:
                break
            page = reply.get('observation')
            if tool == self.offer_tool():
                self.offer_seen[ident] = self.step
            self.observe(tool, page)
            proof = self.evidence(page)
            pages.append(f'{ident}: ' + self.show(page)[:4000] + (f'\n{proof}' if proof else ''))
        self.trace.append({'kind': 'lever_fire', 'why': 'cartlook', 'opened': len(pages), 'lines': len(live), 'short': 0, 'step': self.step})
        if not pages:
            return ''
        return ('[Kyo] Cart check after the new answers (' + '; '.join(news)[:600] + '): each cart line was opened again just now, one step each.\n'
                + '\n'.join(pages)
                + '\nA line stays only if its own page shows every new answer; if it does not, take it out and put in a listing of the same kind you have seen whose page shows them.')

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
            codes = set(index['marks'].get(devices[0], {}).get('codes') or [])
            if not codes:
                continue
            for line in parts:
                fits = self.fit_set(line, index)
                if fits and (not fits & codes):
                    good = [i for i in self.item_ids(part) if self.fit_set(i, index) & codes and (not self.sold_out(i)) and (i not in cart)][:4]
                    faults.append(f"{line} ({self.items[part - 1].name}) works with {', '.join(sorted(fits)[:4])}, not with {devices[0]} ({', '.join(sorted(codes))})" + (f"; seen parts of that kind that work with it: {', '.join(good)}" if good else '; no seen part of that kind works with it, so look for one or switch the device: ' + self.fit_plan(short=True)))
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
            devices = {code: [i for i in self.item_ids(main) if code in (index['marks'].get(i, {}).get('codes') or [])][:2] for code, _ in ranked}
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
        elif not lead and self.top_read and [i for i in self.top_read if i not in ids]:
            rows = [i for i in self.top_read if i not in ids and (not self.sold_out(i))]
            prompt += '\n\nThe listings below are of the cart\'s kinds and were read just now. They matter only when a cart line fails a requirement: then put in "better" each of them that meets every requirement for that line.\n\nOTHER LISTINGS SEEN (newest first):\n' + '\n'.join((_dump(_clip(self.full.get(i) or self.rows.get(i), 160))[:900] for i in rows))
        elif LEVER_ON and SWAPFIX_ON and (not lead):
            rows = self.item_others(ids)
            if rows:
                prompt += '\n\nOTHER LISTINGS SEEN (newest first):\n' + '\n'.join((f'{i}: ' + _dump(_clip(self.full.get(i) or self.rows.get(i), 160))[:900] for i in rows))
        got = self.ask_json(prompt, self.judges(limit), 3000)
        self.trace.append({'kind': 'check', 'cart': ids, 'got': got})
        self.verdicts[key] = bool(got) and got.get('ok') is not False
        if not got or got.get('ok') is not False:
            return ''
        self.leads = [i for i in self.leads if i not in ids]
        problems = [str(p) for p in _items(got.get('problems'))][:5]
        better = [str(b) for b in _items(got.get('better')) if str(b) in self.rows and (not self.sold_out(str(b)))][:4]
        self.named_better = better
        text = 'A careful review says this cart is not right yet: ' + '; '.join(problems or ['see the requirements'])
        if better:
            text += '. Listings that look better: ' + ', '.join(better)
        return text + '. Fix the cart (or show the review is wrong by re-reading the records), then check out.'

    def trail(self, early: bool=False) -> tuple[str, bool]:
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        tool = self.roles.get('offer') or self.roles.get('open')
        target = self.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        unfound: list[tuple[str, str]] = []
        rounds = 0
        while pending and (rounds < 2 if early else self.trails < (6 if self.left() > 15 else 4)) and (self.cost < SPEND_HARD):
            rounds += 1
            if not early:
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
            got = self.ask_json(prompt, [LIGHT] + ([HEAVY] if self.cost + HEAVY_ROOM <= SPEND_HARD and self.heavy_ok() else []), 4500) or {}
            self.trace.append({'kind': 'trail', 'pending': pending, 'got': _clip(got, 150)})
            reach: list[str] = []
            lost = []
            gaps: list[tuple[str, str]] = []
            unfound = []
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
                    unfound.append((about, words))
            pending = [p for p in pending if p not in self.ref_values]
            if not reach and gaps and tool and target and (self.left() > 12) and (self.link_searches < 3):
                reach = self.find_link(*gaps[0])
                if self.trace and self.trace[-1].get('kind') == 'link_known':
                    seen.append(f"{self.trace[-1]['id']}: an opened page of the shop this chain names that shows the values its description gives; read the value from it")
                    lost = lost[1:]
            if not reach or not tool or (not target) or (self.left() < len(reach) + 4) or self.finished:
                break
            for ident in reach[:max(0, min(3, (3 if early else 6) - len(seen)))]:
                faults: list[str] = []
                args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
                if faults:
                    break
                reply = self.act(tool, args)
                if reply is None or self.finished:
                    break
                self.observe(tool, reply.get('observation'))
                seen.append(f'{ident}: ' + self.show(reply.get('observation'))[:1500])
        self.unfound = unfound
        text = ''
        if seen:
            text += '[Kyo opened the pages a requirement points at]\n' + '\n'.join(seen) + '\n'
        if self.ref_values:
            text += '[values read from the pages requirements point at] ' + '; '.join((f'{k} -> {v}' for k, v in self.ref_values.items())) + '\n'
        if early:
            return (text, False)
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

    def point_now(self) -> str:
        if not (LEVER_ON and POINTNOW_ON) or self.finished or self.buys or (self.point_runs >= POINTNOW_MAX):
            return ''
        if len(self.seen_order) <= self.point_mark or self.left() < self.tail() + 8 or self.cost > SPEND_SOFT - 0.012:
            return ''
        live = {need for _, need in self.needs_now()}
        if not [need for _, need in self.refs if need in live and need not in self.ref_values]:
            return ''
        self.point_runs += 1
        before = len(self.ref_values)
        text, _ = self.trail(early=True)
        self.point_mark = len(self.seen_order)
        self.trace.append({'kind': 'point_now', 'n': len(self.ref_values) - before, 'read': len(self.unfound)})
        if self.unfound:
            text += '\n[pointed-to listings not found yet] ' + ' | '.join((about for about, _ in self.unfound[:3])) + ': find each by its shop, brand or words and open its page; the requirement takes the value the last page of its chain states.'
        return text.strip()

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

    def open_all(self) -> str:
        tool = self.roles.get('offer')
        target = self.single_target(tool)
        unopened = [i for i in self.cart_ids() if i not in self.opened and i not in self.gone]
        if not tool or not target or (not unopened) or (self.left() < len(unopened) + self.tail() + 2):
            return ''
        self.trace.append({'kind': 'lever_fire', 'why': 'openall', 'n': len(unopened)})
        lines = []
        for ident in unopened:
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
        late = self.cost > SPEND_SOFT
        if self.audited or (late and (not (LEVER_ON and LASTREV_ON and (self.cost <= LASTREV_CAP)))):
            return ''
        models = [LIGHT] if late else None
        if LEVER_ON and TRIM2_ON and self.early_empty:
            self.early_empty = False
            new = len(set(self.full) - self.audit_ids)
            if not new:
                self.audited = True
                self.trace.append({'kind': 'lever_fire', 'why': 'trim2_skip', 'late': late})
                return ''
            if not late and self.cost > PRICE_FROM:
                models = [LIGHT, HEAVY]
                self.trace.append({'kind': 'lever_fire', 'why': 'trim2_light', 'new': new})
        self.audited = True
        got = self.audit_call(models)
        named = self.audit_fits(got)
        if late:
            self.trace.append({'kind': 'lever_fire', 'why': 'late_audit', 'named': len(named)})
        if not named and (not late):
            rows = [str(i) for i in _items((got or {}).get('open')) if str(i) in self.rows and str(i) not in self.opened and (not self.sold_out(str(i)))]
            if rows and self.open_rows(rows):
                got = self.audit_call(self.review_models('reopen'))
                named = self.audit_fits(got)
        if not named and (not late):
            got = self.kind_review()
            named = self.audit_fits(got)
        if not named:
            return ''
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return 'A careful review thinks these listings may fit: ' + '; '.join((f"item {k}: {', '.join(v)}" for k, v in named.items())) + (f" ({got.get('reason')})" if got.get('reason') else '') + '. Open them and buy what truly fits; end without a purchase only if each one fails.'

    def audit_call(self, models: list[str] | None=None) -> dict[str, Any] | None:
        records = []
        self.audit_window = []
        self.audit_ids = set(self.full)
        order = list(reversed(self.seen_order))
        heading = 'LISTING RECORDS SEEN (newest first):\n'
        if LEVER_ON and LASTREV_ON:
            seen = set(order)
            mine = [i for i in reversed(list(self.full)) if i in seen] + [i for i in order if i not in self.full]
            out = [i for i in reversed(self.taken_out) if i in seen and (not self.sold_out(i))] if REAUDIT_ON and REAUDIT_OUT else []
            mine = out + [i for i in mine if i not in out]
            if [i for i in mine if not self.sold_out(i)][:30] != [i for i in order if not self.sold_out(i)][:30]:
                order = mine
                heading += ('(lines taken out of the cart first, newest first; then ' if out else '(') + 'opened pages first, newest opened first; then rows, newest seen first)\n'
                self.trace.append({'kind': 'lever_fire', 'why': 'audit_order'})
                if out:
                    self.trace.append({'kind': 'lever_fire', 'why': 'audit_out', 'out': len(out)})
        for ident in order:
            if self.sold_out(ident):
                continue
            record = self.full.get(ident) if ident in self.full else None
            records.append('OPENED ' + _dump(_clip(record, 200))[:1200] if record else 'ROW ' + _dump(_clip(self.rows.get(ident), 120))[:500])
            self.audit_window.append(ident)
            if len(records) >= 30:
                break
        if records and LEVER_ON and AUDOLD_ON:
            words = set(re.findall('[a-z]{3,}', self.request.lower()))
            pool = [i for i in reversed(self.seen_order) if i not in self.audit_window and i not in self.full and (i in self.rows) and (not self.sold_out(i))]
            hits = {i: len(words & set(re.findall('[a-z]{3,}', _dump(self.rows.get(i)).lower()))) for i in pool}
            older = sorted(pool, key=lambda i: -hits[i])[:AUDOLD_ROWS]
            if older:
                records.append('\nOLDER ROWS NEVER OPENED (most request words first):')
                records += ['ROW ' + _dump(_clip(self.rows.get(i), 120))[:500] for i in older]
                self.audit_window += older
                self.trace.append({'kind': 'lever_fire', 'why': 'audold', 'n': len(older)})
        if not records:
            return None
        prompt = AUDIT + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
        prompt += '\n\n' + heading + '\n'.join(records)
        if models is None and LEVER_ON and AUDITLIGHT_ON and self.cost > AUDIT_HEAVY_UNTIL:
            models = [LIGHT, HEAVY]
            self.trace.append({'kind': 'lever_fire', 'why': 'audit_light', 'cost': round(self.cost, 4)})
        room = LIGHTROOM_TOKENS if LEVER_ON and LIGHTROOM_ON and models and (models[0] == LIGHT) else 4000
        if room != 4000:
            self.trace.append({'kind': 'lever_fire', 'why': 'lightroom'})
        got = self.ask_json(prompt, models or ([HEAVY, LIGHT] if self.heavy_ok() else [LIGHT]), room)
        self.trace.append({'kind': 'audit', 'got': got})
        return got

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get('fits')
        if not isinstance(fits, dict):
            return {}
        named = {str(k): [str(i) for i in _items(v) if str(i) in self.rows and (not self.sold_out(str(i)))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def last_review(self) -> str:
        if self.left() < 3 or self.cost > LASTREV_CAP or (self.audited and (not self.lead_items)):
            return ''
        if self.audited:
            named, why = (dict(self.lead_items), 'last_leads')
        else:
            self.audited = True
            named, why = (self.audit_fits(self.audit_call([LIGHT])), 'last_review')
        self.trace.append({'kind': 'lever_fire', 'why': why, 'named': len(named)})
        lead = next((i for ids in named.values() for i in ids if not self.sold_out(i)), None)
        if not lead:
            return ''
        self.lead_items, self.leads = (named, [i for ids in named.values() for i in ids])
        self.fill_facts([lead], 2)
        if self.finished or self.sold_out(lead):
            return ''
        self.confirmed = lead
        done = self.complete_cart(LASTREV_CAP)
        if not done and (not self.finished):
            self.confirmed = None
        return done

    def reaudit(self) -> str:
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if not (LEVER_ON and REAUDIT_ON) or self.reaudited or holding or (self.cart_touched and (not fresh)) or self.confirmed or self.completed or (self.left() < 3) or (self.cost > LASTREV_CAP):
            return ''
        new = len(set(self.full) - self.audit_ids)
        if self.left() <= 5:
            if self.leads and self.refuse_blocks >= 2 or (self.audited and (not self.lead_items) and (not new)):
                return ''
            self.reaudited = True
            self.trace.append({'kind': 'lever_fire', 'why': 'reaudit_last', 'audited': self.audited, 'new': new})
            if self.audited and (not self.lead_items):
                self.audited = False
            return self.last_review()
        if not self.audited or self.leads or (not new):
            return ''
        self.reaudited = True
        got = self.audit_call([LIGHT])
        named = self.audit_fits(got)
        self.trace.append({'kind': 'lever_fire', 'why': 'reaudit', 'named': len(named), 'new': new})
        if not named:
            return ''
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        self.refuse_blocks = 0
        return 'Not run yet (no step used). A careful review of the listings seen, opened pages first, thinks these may fit: ' + '; '.join((f"item {k}: {', '.join(v)}" for k, v in named.items())) + (f" ({got.get('reason')})" if got.get('reason') else '') + '. Open them and buy what truly fits; end without a purchase only if each one fails.'

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

    def complete_cart(self, limit: float=SPEND_HARD, insist: bool=False) -> str:
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
        if (calls is None if LEVER_ON and CARTREADY_ON else not calls) or len(calls) + 1 > self.left():
            return ''
        self.fill_facts(list(lines), len(calls) + 3)
        if LEVER_ON and CARTFILL_ON and calls:
            calls = self.cart_fill(calls)
            if not calls:
                return ''
        if insist:
            if self.finished or any((self.sold_out(i) for i in lines)) or len(calls) + 1 > self.left():
                return ''
        else:
            self.check_cart(list(lines), limit)
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
        for attempt in range(1 if insist else 2):
            if attempt:
                self.reread_offers()
                self.check_cart(list(lines), limit)
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
        if insist:
            return 'Not run (Kyo acted instead): the shopper had revealed what they want, so Kyo put the nearest listing for each item in the cart (' + '; '.join(done) + ') and checked out. Store reply: ' + ' | then: '.join(replies)
        return 'Not run (Kyo acted instead): the independent check confirmed ' + self.confirmed + ', so Kyo set the cart (' + '; '.join(done) + ') and checked out. Store reply: ' + ' | then: '.join(replies)

    def told_hold(self) -> str:
        if LEVER_ON and DECL2_ON and (not self.buy_push()):
            return ''
        stated = [(n, it) for n, it in enumerate(self.items, 1) if [m for m in it.told if m not in it.dropped]]
        room = self.left()
        if self.told_held or self.buys or self.finished or not stated or room < len(self.items) + 3:
            return ''
        if LEVER_ON and TOLD3_ON and len(self.items) == 3:
            return ''
        self.told_held = True
        wide = room >= 10 + 2 * len(self.items)
        self.trace.append({'kind': 'lever_fire', 'why': 'toldbuy', 'wide': int(wide), 'lines': len(self.items), 'stated': len(stated), 'step': self.step})
        many = len(self.items) > 1
        heard = '; '.join(((f'item {n} ({it.name}): ' if many else '') + ', '.join([m for m in it.told if m not in it.dropped]) for n, it in stated))[:500]
        step_one = ('First run one short search per item made of the shopper\'s own answer words, and open the best new match for each item. '
                    if wide else '')
        return (f'Not run yet (no step used). The shopper already answered with what they want ({heard}), and an ending '
                f'without a purchase scores nothing when the store holds a listing that fits. {room} steps are left. ' + step_one
                + 'Then put the closest listing you opened for each item in the cart at the requested quantity and check out. '
                'End without a purchase only if every listing you opened for an item plainly contradicts one of those answers.')

    def second_end_swap(self) -> tuple[str, dict[str, Any]] | None:
        buy = self.roles.get('buy')
        if self.second_end or self.end_tries < 2 or self.buys or self.finished or (not buy) or self.reveal_events >= SECONDEND_AT or self.left() < 1:
            return None
        lines = [i for i in self.cart_ids() if not self.sold_out(i)]
        if not lines or (LEVER_ON and FULLONLY_ON and not self.full_lines(lines)):
            return None
        self.second_end = True
        faults: list[str] = []
        args = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), 'args', faults)
        self.trace.append({'kind': 'lever_fire', 'why': 'secondend', 'lines': len(lines), 'items': len(self.items), 'reveals': self.reveal_events, 'tries': self.end_tries, 'faults': len(faults), 'step': self.step})
        if faults:
            return None
        return (buy, args)

    def full_lines(self, lines: list[str]) -> bool:
        cat_key = self.category_key()
        kinds = {str((self.rows.get(i) or {}).get(cat_key) or '').strip().lower() or i for i in lines} if cat_key else set(lines)
        return len(kinds) >= len(self.items)

    def empty_end(self) -> str:
        cart, buy = (self.roles.get('cart'), self.roles.get('buy'))
        if self.empty_tried or self.end_tries < 2 or self.buys or self.finished or (not cart) or (not buy):
            return ''
        if self.reveal_events >= SECONDEND_AT or self.left() < len(self.items) + 2 or self.cost > REVEALBUY_CAP:
            return ''
        held = [i for i in self.cart_ids() if not self.sold_out(i)]
        if held and self.full_lines(held):
            return ''
        self.empty_tried = True
        named = self.near_pick()
        lead = next((i for ids in named.values() for i in ids if not self.sold_out(i)), None)
        self.trace.append({'kind': 'lever_fire', 'why': 'emptyend', 'named': len(named), 'held': len(held), 'items': len(self.items), 'reveals': self.reveal_events, 'tries': self.end_tries, 'step': self.step})
        if not lead or self.finished:
            return ''
        before = (self.lead_items, self.leads, self.confirmed, self.completed)
        self.lead_items, self.leads, self.confirmed = (named, [i for ids in named.values() for i in ids], lead)
        self.completed = False
        done = self.complete_cart(REVEALBUY_CAP, insist=True)
        if not done and (not self.finished):
            if not self.cart_touched_since(held):
                self.lead_items, self.leads, self.confirmed, self.completed = before
            self.trace.append({'kind': 'lever_fire', 'why': 'emptyend_none', 'step': self.step})
            return ''
        return done.replace('the shopper had revealed what they want', 'the ending without a purchase came again with the shopper having revealed little', 1)

    def gap_fill(self) -> str:
        



        cart, buy = (self.roles.get('cart'), self.roles.get('buy'))
        cat_key = self.category_key()
        if self.gap_tried or self.buys or self.finished or (not cart) or (not buy) or (not cat_key):
            return ''
        held = [i for i in self.cart_ids() if not self.sold_out(i)]
        if not held or self.full_lines(held) or len(self.items) < 2:
            return ''
        if self.left() < len(self.items) + 2 or self.cost > REVEALBUY_CAP:
            return ''
        self.gap_tried = True

        def kind(ident: str) -> str:
            return str((self.rows.get(ident) or self.full.get(ident) or {}).get(cat_key) or '').strip().lower()
        have = {kind(i) for i in held}
        named = self.near_pick()
        fill: dict[str, list[str]] = {}
        for number, ids in named.items():
            pick = next((i for i in ids if not self.sold_out(i)), None)
            label = kind(pick) if pick else ''
            if pick and label and label not in have and label not in {kind(v[0]) for v in fill.values()}:
                fill[number] = [pick]
        self.trace.append({'kind': 'lever_fire', 'why': 'gapfill', 'named': len(named), 'fill': len(fill), 'held': len(held), 'items': len(self.items), 'step': self.step})
        lead = next((v[0] for v in fill.values()), None)
        if not lead or self.finished:
            return ''
        before = (self.lead_items, self.leads, self.confirmed, self.completed)
        self.lead_items, self.leads, self.confirmed = (fill, [v[0] for v in fill.values()], lead)
        self.completed = False
        done = self.complete_cart(REVEALBUY_CAP, insist=True)
        if not done and (not self.finished):
            if not self.cart_touched_since(held):
                self.lead_items, self.leads, self.confirmed, self.completed = before
            self.trace.append({'kind': 'lever_fire', 'why': 'gapfill_none', 'step': self.step})
            return ''
        return done.replace('the shopper had revealed what they want, so Kyo put the nearest listing for each item in the cart',
                            'the cart lacked a line for an item, so Kyo added the nearest listing for each missing item', 1)

    def cart_touched_since(self, held: list[str]) -> bool:
        return [i for i in self.cart_ids() if not self.sold_out(i)] != held

    def revealed(self) -> int:
        return sum(1 for it in self.items if [m for m in it.told if m not in it.dropped])

    def reveal_gate(self) -> str:
        lines = self.revealed()
        room = self.left()
        if self.reveal_holds >= REVEAL_HOLDS or self.buys or self.finished or not lines or room < len(self.items) + 1:
            return ''
        self.reveal_holds += 1
        self.trace.append({'kind': 'lever_fire', 'why': 'revealgate', 'n': self.reveal_holds, 'lines': len(self.items), 'stated': lines, 'step': self.step})
        what = ('the closest listing you opened for each item' if len(self.items) > 1 else 'the closest listing you opened')
        return ('Not run yet (no step used). The shopper has already revealed what they want, so an ending without a purchase is '
                f'closed here: it scores nothing, while the closest listing usually passes. {room} steps are left. Put {what} '
                'in the cart at the requested quantity (open one first only if none is open yet), then check out.')

    def cart_fill(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        name = self.roles.get('cart') or ''
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        need = [k for k in schema.get('required') or [] if k in props]
        about = next((str(t['function'].get('description') or '') for t in self.tools if t['function']['name'] == name), '')
        out: list[dict[str, Any]] = []
        for call in calls:
            missing = [k for k in need if k not in call]
            if not missing:
                out.append(call)
                continue
            ident = next((v for v in call.values() if isinstance(v, str) and (v in self.full or v in self.rows)), None)
            if not ident:
                self.trace.append({'kind': 'lever_fire', 'why': 'cartfill_none', 'path': 'noid', 'step': self.step})
                return None
            record = self.full.get(ident) or self.rows.get(ident)
            whole = _dump(record)
            prompt = (CARTFILL + '\n\nCART TOOL: ' + about[:600] + '\nPARAMETERS: ' + _dump(schema)[:1500]
                      + '\nARGUMENTS ALREADY SET: ' + _dump(call) + '\nMISSING: ' + ', '.join(missing)
                      + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
                      + '\n\nLISTING RECORD:\n' + _dump(_clip(record, 300))[:4000])
            got = self.ask_json(prompt, [LIGHT, HEAVY], 1200)
            args = got.get('args') if isinstance(got, dict) and isinstance(got.get('args'), dict) else {}
            filled = dict(call)
            for key in missing:
                value = args.get(key)
                if isinstance(value, (int, float)) and (not isinstance(value, bool)):
                    value = str(value) if 'string' in _kinds(props.get(key) or {}) else value
                if isinstance(value, str):
                    value = value.strip()
                if value in (None, '') or (isinstance(value, str) and value not in whole):
                    self.trace.append({'kind': 'lever_fire', 'why': 'cartfill_none', 'path': 'unseen', 'n': len(missing), 'step': self.step})
                    return None
                filled[key] = value
            self.trace.append({'kind': 'lever_fire', 'why': 'cartfill', 'n': len(missing), 'step': self.step})
            out.append(filled)
        return out

    def near_pick(self) -> dict[str, list[str]]:
        pool = [i for i in reversed(list(self.full)) if not self.sold_out(i)]
        times: dict[str, int] = {}
        latest: dict[str, int] = {}
        ranked = bool(LEVER_ON and OPENRANK_ON)
        if ranked:
            for n, ident in enumerate(self.open_order):
                times[ident] = times.get(ident, 0) + 1
                latest[ident] = n
            pool.sort(key=lambda i: (-times.get(i, 0), -latest.get(i, -1)))
            self.trace.append({'kind': 'lever_fire', 'why': 'openrank', 'top': max(times.values(), default=0), 'opened': len(times), 'step': self.step})
        tail = [i for i in reversed(self.seen_order) if i in self.rows and i not in self.full and (not self.sold_out(i))]
        if LEVER_ON and ROWORDER_ON and tail:
            tail.sort(key=lambda i: (-self.shown.get(i, 0), self.best_at.get(i, 1 << 30)))
            self.trace.append({'kind': 'lever_fire', 'why': 'roworder', 'rows': len(tail), 'top': self.shown.get(tail[0], 0), 'step': self.step})
        pool += tail
        pool = list(dict.fromkeys(pool))
        wide = bool(LEVER_ON and POOLWIDE_ON) and self.cost <= (REVEALBUY_CAP if POOLGATE_ON else SPEND_SOFT)
        if wide and self.cost > SPEND_SOFT:
            self.trace.append({'kind': 'lever_fire', 'why': 'poolgate', 'spent': round(self.cost, 4), 'step': self.step})
        trim = wide and bool(POOLTRIM_ON)
        if wide and len(self.items) >= 3 and (not trim):
            pool = self.shelf_rounds(pool)
        pool = pool[:(POOLTRIM_ROWS if trim else POOLWIDE_ROWS) if wide else REVEALBUY_ROWS]
        (of, ol), (rf, rl) = (POOLTRIM_OPEN, POOLTRIM_ROW) if trim else ((200, 1200), (120, 500))
        budget = POOLTRIM_CHARS if trim else POOLWIDE_CHARS
        if not pool:
            return {}
        tagged = bool(LEVER_ON and ROWTAG_ON)

        def rowtag(ident: str) -> str:
            seen = self.shown.get(ident, 0) if tagged else 0
            return f's{seen} p{self.best_at.get(ident, 0) + 1} ' if seen else ''
        lines = [('OPENED ' + (f'x{times[i]} ' if times.get(i) else '') + _dump(_clip(self.full.get(i), of))[:ol]) if i in self.full else ('ROW ' + rowtag(i) + _dump(_clip(self.rows.get(i), rf))[:rl]) for i in pool]
        if wide:
            room, keep = (budget, 0)
            for line in lines:
                if keep and len(line) + 1 > room:
                    break
                room -= len(line) + 1
                keep += 1
            pool, lines = pool[:keep], lines[:keep]
            self.trace.append({'kind': 'lever_fire', 'why': 'poolwide', 'pool': keep, 'opened': sum((1 for i in pool if i in self.full)), 'chars': budget - room, 'trim': trim, 'lines': len(self.items), 'step': self.step})
        head = ('opened pages first: the most often opened first, x<n> = times opened, then the latest opened; then rows' if ranked else 'opened pages first, newest first; then rows')
        marks = sum((1 for ln in lines if ln.startswith('ROW s')))
        if tagged and marks:
            head += '; s<n> p<m> on a row = shown in n search result lists, best place m in one (1 = top)'
            self.trace.append({'kind': 'lever_fire', 'why': 'rowtag', 'rows': marks, 'step': self.step})
        prompt = (NEAR + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
                  + '\n\nLISTING RECORDS (' + head + '):\n' + '\n'.join(lines))
        got = self.ask_json(prompt, [LIGHT], 2500)
        self.trace.append({'kind': 'near', 'got': got})
        near = (got or {}).get('near') if isinstance(got, dict) else None
        if not isinstance(near, dict):
            return {}
        picked: dict[str, list[str]] = {}
        for number, value in near.items():
            pick = next((str(i) for i in _items(value) if str(i) in pool), None)
            if pick:
                picked[str(number)] = [pick]
        return picked

    def shelf_rounds(self, pool: list[str]) -> list[str]:
        

        key = self.category_key()
        if not key:
            return pool
        shelves: dict[str, list[str]] = {}
        for ident in pool:
            record = self.rows.get(ident) or self.full.get(ident) or {}
            shelves.setdefault(str(record.get(key) or '').strip().lower(), []).append(ident)
        out: list[str] = []
        while len(out) < len(pool):
            for queue in shelves.values():
                if queue:
                    out.append(queue.pop(0))
        return out

    def reveal_buy(self) -> str:
        stale = bool(self.completed and LEVER_ON and STALEDONE_ON and not self.cart_ids())
        if not (LEVER_ON and REVEALBUY_ON) or self.near_tried or self.buys or self.finished or (self.completed and not stale) or (not self.buy_push()):
            return ''
        if self.left() < len(self.items) + 2 or self.cost > REVEALBUY_CAP or (not self.roles.get('cart')) or (not self.roles.get('buy')):
            return ''
        self.near_tried = True
        named = {k: [i for i in v if not self.sold_out(i)] for k, v in (self.lead_items or {}).items()}
        named = {k: v for k, v in named.items() if v}
        how = 'leads'
        if len(named) < len(self.items):
            named, how = (self.near_pick(), 'near')
        lead = next((i for ids in named.values() for i in ids if not self.sold_out(i)), None)
        self.trace.append({'kind': 'lever_fire', 'why': 'revealbuy', 'how': how, 'named': len(named), 'lines': len(self.items), 'stated': self.revealed(), 'stale': stale, 'step': self.step})
        if not lead or self.finished:
            return ''
        before = (self.lead_items, self.leads, self.confirmed)
        self.lead_items, self.leads, self.confirmed = (named, [i for ids in named.values() for i in ids], lead)
        if stale:
            self.completed = False
        done = self.complete_cart(REVEALBUY_CAP, insist=True)
        if not done and (not self.finished):
            self.lead_items, self.leads, self.confirmed = before
            if stale:
                self.completed = True
            self.trace.append({'kind': 'lever_fire', 'why': 'revealbuy_none', 'step': self.step})
        return done

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
        keep = LEVER_ON and ENDID_ON and (self.end_turned > 0)
        hold = LEVER_ON and QUIETFIX_ON and self.brief.get('quiet_decline') is not True
        for key, spec in props.items():
            if not isinstance(spec, dict) or 'string' not in _kinds(spec) or 'enum' in spec:
                continue
            value = out.get(key)
            if value is None and key not in (schema.get('required') or []):
                continue
            if keep and isinstance(value, str) and (value.strip() in self.rows):
                out[key] = value.strip()
                continue
            text = str(value or '')
            if hold and text.strip():
                continue
            named = any((ident in text for ident in self.rows)) or self.brief.get('quiet_decline') is not False
            if named or not text:
                out[key] = 'Sorry, nothing in the store meets all of your requirements, so I am not buying anything.'[:spec.get('maxLength', 200)]
        if LEVER_ON and ENDPAIR_ON:
            out = self.pair_ids(name, out)
        return self.cite(name, out) if hold else out

    def id_props(self, name: str) -> list[str]:
        out: list[str] = []
        for key in self.text_props(name):
            column = [row[key] for row in self.rows.values() if key in row]
            if not column or not all(_id_like(v, True) for v in column):
                continue
            if len(set(column)) != len(column):
                continue
            if len(column) == 1 and not _id_like(column[0]):
                continue
            out.append(key)
        return out

    def pair_ids(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        keys = self.id_props(name)
        anchor = self.anchor_row(keys, args) if keys else None
        if not anchor:
            return args
        changes = {key: anchor[key] for key in keys if _id_like(anchor.get(key), True) and args.get(key) != anchor[key]}
        if not changes:
            return args
        self.trace.append({'kind': 'lever_fire', 'why': 'endpair', 'fixed': len(changes), 'keys': len(keys), 'step': self.step})
        return {**args, **changes}

    def anchor_row(self, keys: list[str], args: dict[str, Any]) -> dict[str, Any] | None:
        for key in keys:
            value = args.get(key)
            if not (isinstance(value, str) and value.strip()):
                continue
            ident = value.strip()
            row = self.rows.get(ident) or next((r for r in self.rows.values() if r.get(key) == ident), None)
            if row:
                return row
        near = self.closest()
        options = [i for i in near if not self.sold_out(i)] or near or [i for i in reversed(self.seen_order) if i in self.rows][:1]
        return self.rows.get(options[0]) if options else None

    def names_listing(self, args: dict[str, Any]) -> bool:
        for value in args.values():
            if isinstance(value, str) and any(ident in value for ident in self.rows):
                return True
        return False

    def closest(self) -> list[str]:
        pool = list(self.leads) + [i for i in reversed(self.seen_order) if i in self.opened] + list(reversed(self.built))
        return [i for i in dict.fromkeys(pool) if i in self.rows][:2]

    def text_props(self, name: str) -> list[str]:
        found: list[str] = []
        for key, spec in ((self.schemas.get(name) or {}).get('properties') or {}).items():
            if isinstance(spec, dict) and 'enum' not in spec and 'string' in _kinds(spec):
                found.append(key)
        return found

    def cite(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        near = self.closest()
        slots = [k for k in self.text_props(name) if self.prose(args.get(k))]
        if not near or not slots or self.names_listing({k: args[k] for k in slots}):
            return args
        slot = slots[0]
        note = ' Closest listings I checked: ' + ', '.join(near) + '; each one misses at least one of your requirements.'
        limit = ((((self.schemas.get(name) or {}).get('properties') or {}).get(slot)) or {}).get('maxLength')
        body = args[slot].rstrip()
        if isinstance(limit, int) and len(body) + len(note) > limit:
            keep = limit - len(note)
            if keep < 20:
                return args
            body = body[:keep].rstrip()
        self.trace.append({'kind': 'lever_fire', 'why': 'quietfix_cite', 'n': len(near), 'step': self.step})
        return {**args, slot: body + note}

    def prose(self, value: Any) -> bool:
        return isinstance(value, str) and len(value.split()) >= 2 and value.strip() not in self.rows

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
        before = set(self.cart_lines)
        if whole:
            self.cart_lines = {i: c for i, c in lines if c and c > 0}
            self.note_out(before)
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
        self.note_out(before)

    def note_out(self, before: set[str]) -> None:
        for ident in set(self.cart_lines) - before:
            self.cart_since.setdefault(ident, self.step)
            if ident in self.taken_out:
                self.taken_out.remove(ident)
        for ident in before - set(self.cart_lines):
            since = self.cart_since.pop(ident, self.step)
            if self.step - since >= 3 and ident not in self.taken_out:
                self.taken_out.append(ident)

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
            declining = False
            if LEVER_ON and ENDID_ON and self.is_refuse(name, args):
                self.end_ids.update({k: v.strip() for k, v in args.items() if isinstance(v, str) and v.strip() in self.rows})
            if ending and len(calls) > 1:
                out.append((cid, 'Not run (no step used): send the finishing call on its own, after reading your other results.'))
                continue
            if self.is_refuse(name, args):
                self.end_tries += 1
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
                    gap = self.gap_fill() if LEVER_ON and GAPFILL_ON else ''
                    if gap:
                        out.append((cid, gap))
                        continue
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows() + self.top_rows()
                    verdict = self.check_cart()
                    if verdict and LEVER_ON and SWAPFIX_ON and self.swap_fix():
                        trimmed += "\n[Kyo] The last check named listings that fit better than cart lines, so Kyo put each in place of its item's line and a fresh check passed. Cart now: " + self.cart_text()
                        verdict = ''
                    if verdict:
                        seen = "Kyo read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n' if read else ''
                        out.append((cid, 'Not run yet. ' + chain + seen + verdict + '\nCart now: ' + self.cart_text()))
                        continue
                    fit = self.fit_buy() if LEVER_ON and FITPICK_ON else ''
                    if fit:
                        out.append((cid, 'Not run yet (no step used). ' + fit + '\nCart now: ' + self.cart_text()))
                        continue
                    if reread and self.cost > SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). Kyo re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'))
                        continue
                    opened = self.open_all() if LEVER_ON and OPENALL_ON else ''
                    if opened:
                        out.append((cid, "Not run yet (no step used). Kyo opened the pages of the cart's listings not opened before:\n" + opened + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'))
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
                    fit = self.fit_decline() if LEVER_ON and FITPICK_ON else ''
                    if fit:
                        out.append((cid, 'Not run yet (no step used). ' + fit))
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
            if self.is_refuse(name, args) and (not self.leads) and (not self.dopened) and (self.left() > self.tail() + 8):
                found = self.decline_open()
                if found:
                    out.append((cid, 'Not run yet (no step used). ' + found))
                    continue
            if self.is_refuse(name, args) and (not self.leads) and (not self.dsearched) and (self.left() > self.tail() + 10):
                self.dsearched = True
                found = self.decline_search()
                if found:
                    out.append((cid, 'Not run yet (no step used). ' + found))
                    continue
            if self.is_refuse(name, args) and (not self.leads) and (not self.pushed) and (self.left() > self.max_steps // 3):
                self.pushed = True
                out.append((cid, f"Not run yet (no step used). {self.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if self.is_refuse(name, args):
                again = self.reaudit()
                if again or self.finished:
                    out.append((cid, (again or 'Not run: the episode is over.') + (' [episode finished]' if self.finished else '')))
                    continue
                done = self.complete_cart()
                if done:
                    out.append((cid, done + (' [episode finished]' if self.finished else '')))
                    continue
                held = (self.told_hold() if LEVER_ON and TOLDBUY_ON else '') or (self.reveal_gate() if LEVER_ON and REVEALGATE_ON else '')
                if held:
                    out.append((cid, held))
                    continue
                fresh_fill = not self.near_tried
                bought = self.reveal_buy()
                if bought or self.finished:
                    out.append((cid, (bought or 'Not run: the episode is over.') + (' [episode finished]' if self.finished else '')))
                    continue
                if LEVER_ON and QUIETFIX_ON and (not self.cite_held) and self.left() >= 3 and (self.brief.get('quiet_decline') is not True) and (not self.names_listing(args)) and self.closest() and (set(self.text_props(name)) - set(self.id_props(name) if ENDPAIR_ON else [])):
                    self.cite_held = True
                    self.trace.append({'kind': 'lever_fire', 'why': 'quietfix_hold', 'step': self.step})
                    out.append((cid, "Not run yet (no step used). In the ending call's text, name the closest listings you opened (their ids) and the requirement each one misses, then send the ending call again."))
                    continue
                filled = self.empty_end() if LEVER_ON and EMPTYEND_ON and not (fresh_fill and self.near_tried) else ''
                if filled or self.finished:
                    out.append((cid, (filled or 'Not run: the episode is over.') + (' [episode finished]' if self.finished else '')))
                    continue
                swap = self.second_end_swap() if LEVER_ON and SECONDEND_ON else None
                if swap:
                    name, args = swap
                    trimmed = ('\n[Kyo] This was a second ending without a purchase while the cart holds lines and the '
                               'shopper had revealed little, so Kyo checked out the cart as it is instead.') + trimmed
                else:
                    declining = True
                    args = self.quiet(name, args)
            if self.is_buy(name, args):
                args = self.say_back(name, args)
            if ending and self.left() >= 2:
                go, said = self.gate('order' if self.is_buy(name, args) else 'end')
                if not go:
                    out.append((cid, 'Not run yet. ' + said + '\nCart now: ' + self.cart_text()))
                    continue
                if said:
                    trimmed = '\n[Kyo] ' + said + trimmed
            lead = self.open_first() if name == self.roles.get('search') and (not ending) else ''
            if lead and self.finished:
                out.append((cid, lead + 'Not run: the episode is over.'))
                continue
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, lead + 'The call failed (network or server). You may try again.'))
                continue
            observation = reply.get('observation')
            if declining and (not self.finished) and LEVER_ON and ENDID_ON:
                self.end_turned += 1
                self.trace.append({'kind': 'lever_fire', 'why': 'end_turned', 'by': 'agent', 'n': self.end_turned, 'ids': len(self.end_ids), 'step': self.step})
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
            if name == self.roles.get('search'):
                self.search_log.append(dict(args))
            if self.single_target(name) and name not in (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask')):
                listing = self.ident(observation) if isinstance(observation, dict) else None
                if listing:
                    self.facts_from[name] = self.facts_from.get(name, False) or bool(_fact_block(observation))
                    if name == self.roles.get('open') and (not self.facts_from[name]):
                        other = next((t for t in self.schemas if t != name and self.single_target(t) and (self.facts_from.get(t) is not False) and (t not in (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask'), self.roles.get('search')))), None)
                        if other:
                            self.roles['open'] = other
                        if LEVER_ON and OFFERFIX_ON and other and (self.roles.get('offer') == other):
                            self.roles['offer'] = name
                            self.trace.append({'kind': 'lever_fire', 'why': 'offerswap'})
                    if not self.facts_from[name] and name != self.roles.get('open') and (not self.roles.get('offer')):
                        self.roles['offer'] = name
                    if name == self.offer_tool():
                        self.offer_seen[listing] = self.step
                elif LEVER_ON and OFFERFIX_ON and (name == self.roles.get('offer')) and (name == getattr(self, 'offer_fixed', None)) and (name not in self.facts_from) and (not (isinstance(observation, dict) and any((isinstance(v, dict) and self.ident(v) for v in observation.values())))):
                    self.roles.pop('offer', None)
                    self.offer_fixed = None
                    self.trace.append({'kind': 'lever_fire', 'why': 'offerundo'})
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
                    relook = self.note_relook()
                    if relook:
                        text += '\n' + relook
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
            if OWNDROP_ON and self.own_relax:
                self.own_relax = False
                self.relaxed = True
                relook = self.note_relook()
                if relook:
                    text += '\n' + relook
            if name in readers:
                self.read_cache[mark] = (self.version, self.step, base)
            if reply.get('finished'):
                text += ' [episode finished]'
            out.append((cid, lead + text))
        return out

    def blind_note(self, name: str, seen: Any) -> None:
        closing = {self.roles.get(r) for r in ('search', 'cart', 'buy', 'refuse', 'ask', 'cart_view')} - {None}
        if name == self.roles.get('search'):
            self.blind_obs, self.blind_mark = seen, self.views
        elif name not in closing and (self.single_target(name) or name == self.roles.get('compare')):
            self.views += 1

    def open_first(self) -> str:
        if not (LEVER_ON and OPENFIRST_ON) or self.blind_obs is None or self.openfirst_n >= OPENFIRST_MAX:
            return ''
        if self.views > self.blind_mark or self.left() <= self.tail() + max(6, self.max_steps // 5):
            return ''
        tool = self.facts_tool() or self.roles.get('open')
        target = self.single_target(tool)
        if not tool or not target or tool not in {t['function']['name'] for t in self.offered()}:
            return ''
        found = self.records_in(self.blind_obs)
        self.blind_obs = None
        pick = next((i for i, _ in found if i in self.rows and i not in self.opened and i not in self.full and self.buyable(i)), None)
        if not pick:
            return ''
        got = self.run_calls([{'id': f'of{self.step}', 'type': 'function', 'function': {'name': tool, 'arguments': _dump({target: pick})}}])
        said = got[0][1] if got else ''
        if not said or said.startswith('Not run') or said.startswith('The call failed'):
            return ''
        self.openfirst_n += 1
        self.trace.append({'kind': 'lever_fire', 'why': 'openfirst', 'n': self.openfirst_n, 'step': self.step})
        return f'[Kyo opened {pick}, the top unopened listing of the last search, before running this search]\n' + said + '\n\n[this search]\n'

    def need_words(self, needs: list[str]) -> list[str]:
        out: list[str] = []
        for need in needs:
            value = need.split(':', 1)[1] if ':' in need else need
            low = value.lower()
            if NEGATED.search(low) or re.search('\\d', low):
                continue
            out += [w for w in re.findall('[a-z][a-z-]{2,}', low) if w not in COMMON and w not in BOUNDS and (w not in out)]
        return out

    def item_search(self, item: Item) -> tuple[dict[str, Any], str, str] | None:
        tool = self.roles.get('search')
        props = (self.schemas.get(tool or '') or {}).get('properties') or {}
        required = set((self.schemas.get(tool or '') or {}).get('required') or [])
        page = (self.shapes.get(tool or '') or {}).get('next_arg')
        named = set(re.findall('[a-z]{3,}', item.name.lower()))
        chosen = None
        for args in reversed(self.search_log):
            text = ' '.join((str(v).lower() for v in args.values() if isinstance(v, str)))
            if named & set(re.findall('[a-z]{3,}', text)):
                chosen = args
                break
        if chosen is None and len(self.items) == 1:
            chosen = self.search_log[-1]
        if chosen is None:
            return None
        texts = [k for k, v in chosen.items() if k != page and isinstance(v, str) and v.strip() and isinstance(props.get(k), dict) and ('enum' not in props[k])]
        if not texts:
            return None
        key = max(texts, key=lambda k: (len(chosen[k].split()), len(named & set(re.findall('[a-z]{3,}', chosen[k].lower()))), k not in required))
        words = re.findall('[a-z][a-z-]{2,}', chosen[key].lower())
        noun = next((w for w in reversed(words) if w in named), None) or (words[-1] if words else '')
        if not noun:
            return None
        return ({k: v for k, v in chosen.items() if k != page}, key, noun)

    def decline_open(self) -> str:
        if not (LEVER_ON and DOPEN_ON) or self.cost > SPEND_SOFT:
            return ''
        cart = set(self.cart_ids())
        pool = [i for ids in self.item_groups() for i in ids if i not in self.full and i not in self.opened and (i not in cart) and self.buyable(i)]
        if not pool:
            return ''
        words = {w for item in self.items for w in self.need_words(list(item.told) + list(item.musts))}

        def rank(ident: str) -> tuple[int, int, float]:
            text = _dump(self.rows.get(ident)).lower()
            price = self.price_of(ident)
            return (-sum((1 for w in words if w in text)), -self.shown.get(ident, 0) if DOPEN_SHOWN else 0, price if price is not None else float('inf'))
        rows = sorted(pool, key=rank)[:DOPEN_ROWS]
        if DOPEN_SHOWN and set(rows) != set(sorted(pool, key=lambda i: rank(i)[::2])[:DOPEN_ROWS]):
            self.trace.append({'kind': 'lever_fire', 'why': 'dopen_shown', 'pool': len(pool)})
        before = len(self.full)
        if not self.open_rows(rows):
            return ''
        self.dopened = True
        self.audited = False
        review = self.audit()
        self.trace.append({'kind': 'lever_fire', 'why': 'decline_open', 'pool': len(pool), 'opened': len(self.full) - before, 'leads': len(self.leads)})
        if not review:
            return ''
        return 'Before ending without a purchase, Kyo opened listings of the requested kind that a search had shown but nobody had read. ' + review

    def rec_rows(self, number: int) -> list[str]:
        

        cat_key = self.category_key()
        item = self.items[number - 1]
        ids = [i for i, r in self.rows.items() if cat_key and self.owns(item, str(r.get(cat_key) or ''))]
        if not ids and len(self.items) == 1:
            ids = list(self.rows)
        if not ids and len(self.items) > 1:
            ids = [i for i in self.rows if self.found_for.get(i) == number]
        return sorted(ids, key=lambda i: (i not in self.full, -self.shown.get(i, 0)))[:RECSEARCH_ROWS]

    def rec_queries(self, tried: set[str]) -> dict[int, list[Any]]:
        




        if not (LEVER_ON and RECSEARCH_ON) or self.cost > REVEALBUY_CAP:
            return {}
        tool = self.roles.get('search')
        schema = self.schemas.get(tool) if tool else None
        whole = bool(RECARGS_ON and isinstance(schema, dict) and self.search_log)
        blocks = []
        for number, item in enumerate(self.items, 1):
            ids = self.rec_rows(number)
            if not ids and not whole:
                continue
            room = max(300, RECSEARCH_CHARS // max(1, len(self.items) * max(1, len(ids))))
            shown = [str(i) + ': ' + _dump(_clip(self.full.get(i) or self.rows.get(i) or {}, 200))[:room] for i in ids]
            blocks.append(f'ITEM {number}: {item.name}\nLISTINGS SEEN FOR IT:\n' + ('\n'.join(shown) or 'none'))
        if not blocks:
            return {}
        if whole:
            runs = [_dump(a) for a in self.search_log][-RECARGS_LOG:]
            prompt = (RECARGS + '\n\nREQUEST:\n' + str(self.request or '') + '\n\nSEARCH TOOL ' + str(tool)
                      + ' ARGUMENT SCHEMA:\n' + _dump(schema)[:RECARGS_SCHEMA]
                      + '\n\nSEARCHES ALREADY RUN (whole arguments, oldest first):\n' + '\n'.join(runs)
                      + '\n\n' + '\n\n'.join(blocks))
        else:
            prompt = (RECASK + '\n\nREQUEST:\n' + str(self.request or '') + '\n\nSEARCHES ALREADY RUN:\n'
                      + ('\n'.join(sorted(tried)) or 'none') + '\n\n' + '\n\n'.join(blocks))
        got = self.ask_json(prompt, [LIGHT], 700 if whole else 500) or {}
        props = schema.get('properties') if whole and isinstance(schema.get('properties'), dict) else {}
        done = {json.dumps(a, sort_keys=True, default=str) for a in self.search_log}
        out: dict[int, list[Any]] = {}
        for entry in _items(got.get('searches')):
            if not isinstance(entry, dict):
                continue
            try:
                number = int(entry.get('item'))
            except (TypeError, ValueError):
                continue
            if not 1 <= number <= len(self.items):
                continue
            have = out.setdefault(number, [])
            if len(have) >= RECSEARCH_EACH:
                continue
            given = entry.get('args')
            if whole and isinstance(given, dict):
                given = {k: v for k, v in given.items() if not props or k in props}
                faults: list[str] = []
                call = _shape(given, schema, 'args', faults)
                mark = json.dumps(call, sort_keys=True, default=str)
                if given and not faults and mark not in done:
                    done.add(mark)
                    have.append(call)
                continue
            query = ' '.join(str(entry.get('query') or '').lower().split())
            if not query or len(query) > 60 or len(query.split()) > 6:
                continue
            if query not in tried and query not in have:
                have.append(query)
        out = {n: q for n, q in out.items() if q}
        self.trace.append({'kind': 'lever_fire', 'why': 'recsearch', 'lines': len(blocks), 'whole': int(whole),
                           'got': sum((len(q) for q in out.values())),
                           'args': sum((1 for q in out.values() for v in q if isinstance(v, dict))), 'step': self.step})
        return out

    def decline_search(self) -> str:
        tool = self.roles.get('search')
        if not (LEVER_ON and DSEARCH_ON) or not tool or (not self.search_log):
            return ''
        tried = {' '.join(str(v).lower().split()) for a in self.search_log for v in a.values() if isinstance(v, str)}
        many = len(self.items) > 1
        extra = self.rec_queries(tried) if LEVER_ON and RECSEARCH_ON else {}
        queues: list[list[tuple[int, str, dict[str, Any]]]] = []
        for number, item in enumerate(self.items, 1):
            found = self.item_search(item)
            queue = [(number, q, q) for q in extra.get(number, []) if isinstance(q, dict)]
            if not found:
                if queue:
                    queues.append(queue)
                continue
            args, key, noun = found
            words = [w for w in self.need_words(list(item.told) + list(item.musts)) if w != noun]
            for query in [f'{w} {noun}' for w in words] + extra.get(number, []) + [noun]:
                if not isinstance(query, str) or query in tried or any((query == q for _, q, _ in queue)):
                    continue
                faults: list[str] = []
                call = _shape({**args, key: query}, self.schemas.get(tool), 'args', faults)
                if not faults:
                    queue.append((number, query, call))
            if queue:
                queues.append(queue)
        calls = [q[i] for i in range(DSEARCH_RUNS) for q in queues if i < len(q)]
        room = self.left() - self.tail() - 8
        calls = calls[:max(0, min(DSEARCH_RUNS, room))]
        texts: list[str] = []
        for number, query, call in calls:
            if self.finished:
                break
            reply = self.act(tool, call)
            if reply is None:
                break
            seen = reply.get('observation')
            self.observe(tool, seen)
            self.search_log.append(dict(call))
            mark = tool + _dump(call)
            text = self.show(seen)
            self.read_cache[mark] = (self.version, self.step, text)
            texts.append(('item ' + str(number) + ' ' if many else '') + _dump(query) + ': ' + text)
        self.trace.append({'kind': 'lever_fire', 'why': 'decline_search', 'ran': len(texts), 'planned': len(calls)})
        if not texts:
            return ''
        self.audited = False
        return 'Before ending without a purchase, Kyo ran searches not tried yet (one wanted value each, one step each):\n' + '\n'.join(texts) + '\nOpen the rows that may fit and buy what truly fits; end without a purchase only if each one fails a requirement.'

    def note_relook(self) -> str:
        if not (LEVER_ON and NDROP_ON) or not self.relaxed or self.finished or (self.relooks >= NDROP_FIRES) or (self.cost > SPEND_SOFT):
            return ''
        self.relaxed = False
        if self.buys:
            return ''
        if any((not self.sold_out(i) for i in self.cart_ids())):
            return self.drop_search()
        kinds = {i for ids in self.item_groups() for i in ids}
        pool = [i for i in reversed(self.seen_order) if i in kinds and self.buyable(i)][:NDROP_RECORDS]
        if not pool:
            return ''
        self.relooks += 1
        got = self.relook_call(pool)
        named = self.audit_fits(got)
        opened = 0
        if not named and self.left() > self.tail() + 10:
            rows = [str(i) for i in _items((got or {}).get('open')) if str(i) in pool and str(i) not in self.full and (str(i) not in self.opened)]
            before = len(self.full)
            if rows and self.open_rows(rows):
                opened = len(self.full) - before
                got = self.relook_call(pool)
                named = self.audit_fits(got)
        self.trace.append({'kind': 'lever_fire', 'why': 'note_relook', 'pool': len(pool), 'opened': opened, 'leads': sum((len(v) for v in named.values()))})
        if not named or self.finished:
            return ''
        if not self.leads:
            self.lead_items = named
            self.leads = [i for ids in named.values() for i in ids]
            self.note_leads = list(self.leads)
        head = '; '.join((f"item {k}: {', '.join(v)}" for k, v in named.items()))
        return '[Kyo re-checked the listings already seen against the requirements now in force] ' + head + ' appear to meet every requirement now. Open any you have not read since the note, put the best fitting one in the cart (respecting the pick rule) and check out.'

    def drop_search(self) -> str:
        tool = self.roles.get('search')
        if not (LEVER_ON and DROPSRCH_ON) or not tool or (not self.search_log) or (self.dropsrchs >= DROPSRCH_FIRES):
            return ''
        if self.brief.get('pick') != 'cheapest' or self.left() - self.tail() - 8 < 1:
            return ''
        required = set((self.schemas.get(tool) or {}).get('required') or [])
        plan = None
        for number, item in enumerate(self.items, 1):
            new = [m for m in item.dropped if m not in self.drop_done]
            self.drop_done.update(new)
            found = self.item_search(item) if new and plan is None else None
            if not found:
                continue
            args, key, noun = found
            gone = set(self.need_words(new)) - {noun}
            words = str(args[key]).split()
            kept = [w for w in words if w.lower().strip('.,;:!?()"\'') not in gone]
            call = {k: v for k, v in args.items() if not (k != key and k not in required and isinstance(v, str) and v.strip() and set(re.findall('[a-z][a-z-]{2,}', v.lower())) and (set(re.findall('[a-z][a-z-]{2,}', v.lower())) <= gone))}
            capped = self.cap_below(tool, item, call) if LEVER_ON and DROPCAP_ON else None
            if capped is None and len(kept) == len(words) and (len(call) == len(args)):
                continue
            call = capped or call
            call[key] = ' '.join(kept) or noun
            faults: list[str] = []
            call = _shape(call, self.schemas.get(tool), 'args', faults)
            if faults or tool + _dump(call) in self.read_cache:
                continue
            plan = (number, call)
        if plan is None:
            return ''
        self.dropsrchs += 1
        number, call = plan
        read = self.drop_read(tool, call, 'drop_search')
        if read is None or self.finished:
            return ''
        seens, found = ([read[0]], _dump(call) + ': ' + read[1])
        other = self.other_mode(tool, call, read[0]) if LEVER_ON and DROPMODE_ON else None
        if other is not None and self.left() - self.tail() - 8 >= 1:
            again = self.drop_read(tool, other, 'drop_mode')
            if self.finished:
                return ''
            if again is not None:
                seens.append(again[0])
                found += '\n' + _dump(other) + ': ' + again[1]
        many = len(self.items) > 1
        return '[Kyo searched again now that a requirement was dropped (' + ('one step' if len(seens) == 1 else 'two steps') + ')] ' + ('item ' + str(number) + ' ' if many else '') + found + self.cheapest_here(seens) + '\nThe pick rule is the cheapest: the order must hold the lowest-priced listing that meets every requirement now in force.' + ' Read the attributes of any cheaper row here before checking out and swap only for one that fits; otherwise keep the cart.'

    def cap_key(self, tool: str) -> str | None:
        props = (self.schemas.get(tool) or {}).get('properties') or {}
        page = (self.shapes.get(tool) or {}).get('next_arg')
        if not self.price_key:
            return None
        priced = set(_words(self.price_key).split())
        caps = [k for k, v in props.items() if k != page and isinstance(v, dict) and (v.get('type') in ('number', 'integer')) and set(_words(k).split()) & CAP_WORDS and set(_words(k).split()) & priced]
        return caps[0] if len(caps) == 1 else None

    def cap_below(self, tool: str, item: Item, call: dict[str, Any]) -> dict[str, Any] | None:
        props = (self.schemas.get(tool) or {}).get('properties') or {}
        cap_key = self.cap_key(tool)
        if cap_key is None:
            return None
        prices = [p for p in (self.price_of(i) for i in self.item_lines(item)) if p is not None]
        if not prices:
            return None
        price = min(prices) * CAP_SHARE
        cap: float | int = int(-(-price // 1)) - 1 if props[cap_key].get('type') == 'integer' else round(price, 2)
        old = call.get(cap_key)
        if cap <= 0 or (isinstance(old, (int, float)) and (not isinstance(old, bool)) and (old <= cap)):
            return None
        out = {**call, cap_key: cap}
        choices = [k for k, v in props.items() if isinstance(v, dict) and v.get('enum')]
        for past in reversed(self.search_log):
            if cap_key in past:
                out.update({k: past[k] for k in choices if k in past})
                break
        self.trace.append({'kind': 'lever_fire', 'why': 'drop_cap', 'n': 1, 'mode': int(any((k in out for k in choices)))})
        return out

    def other_mode(self, tool: str, call: dict[str, Any], seen: Any) -> dict[str, Any] | None:
        props = (self.schemas.get(tool) or {}).get('properties') or {}
        modes = [k for k, v in props.items() if isinstance(v, dict) and len(v.get('enum') or []) == 2]
        cap_key = self.cap_key(tool)
        if len(modes) != 1 or cap_key is None or modes[0] not in call:
            return None
        mode, cap = (modes[0], call.get(cap_key))
        values = list(props[mode]['enum'])
        if call[mode] not in values or not isinstance(cap, (int, float)) or isinstance(cap, bool):
            return None
        if not any((p is not None and p > cap for p in (self.price_of(i) for i, _ in self.records_in(seen)))):
            return None
        faults: list[str] = []
        out = _shape({**call, mode: values[1 - values.index(call[mode])]}, self.schemas.get(tool), 'args', faults)
        if faults or tool + _dump(out) in self.read_cache:
            return None
        return out

    def drop_read(self, tool: str, call: dict[str, Any], why: str) -> tuple[Any, str] | None:
        reply = self.act(tool, call)
        if reply is None:
            return None
        before = len(self.seen_order)
        seen = reply.get('observation')
        self.observe(tool, seen)
        self.search_log.append(dict(call))
        shown = self.show(seen)
        self.read_cache[tool + _dump(call)] = (self.version, self.step, shown)
        self.trace.append({'kind': 'lever_fire', 'why': why, 'n': len(self.seen_order) - before, 'read': 1})
        return (seen, shown)

    def item_lines(self, item: Item) -> list[str]:
        cart = [i for i in self.cart_ids() if not self.sold_out(i)]
        if len(self.items) == 1:
            return cart
        cat_key = self.category_key()
        long = {w for w in _stem(item.shelf or '') | _stem(item.name) if len(w) >= 4}
        out: list[str] = []
        for ident in cart:
            label = str((self.rows.get(ident) or {}).get(cat_key) or '') if cat_key else ''
            if label and (_words(label) in (_words(item.shelf or ''), _words(item.name)) or {w for w in _stem(label) if len(w) >= 4} & long):
                out.append(ident)
        return out

    def item_for(self, ident: str) -> Item | None:
        if len(self.items) == 1:
            return self.items[0]
        cat_key = self.category_key()
        label = str((self.rows.get(ident) or {}).get(cat_key) or '') if cat_key else ''
        mark = self.found_for.get(ident, 0) if LEVER_ON and SRCLINE2 else 0
        if not label:
            return self.items[mark - 1] if 0 < mark <= len(self.items) else None
        found = [item for item in self.items if _words(label) in (_words(item.shelf or ''), _words(item.name)) or {w for w in _stem(label) if len(w) >= 4} & {w for w in _stem(item.shelf or '') | _stem(item.name) if len(w) >= 4}]
        if not found and 0 < mark <= len(self.items):
            return self.items[mark - 1]
        return found[0] if len(found) == 1 else None

    def item_others(self, skip: list[str]) -> list[str]:
        wanted = {id(item) for item in (self.item_for(i) for i in skip) if item is not None}
        if not wanted:
            return []
        rows = [i for i in reversed(self.seen_order) if i not in skip and (not self.sold_out(i)) and (id(self.item_for(i)) in wanted)]
        rows.sort(key=lambda i: i not in self.full)
        return rows[:SWAPFIX_SEEN]

    def fit_terms(self, ident: str) -> frozenset[str]:
        record = self.full.get(ident) or self.rows.get(ident)
        return _stem(re.sub('[^0-9A-Za-z]+', ' ', _dump(record))) if record else frozenset()

    def fit_pool(self, number: int, skip: list[str], bare: bool=False) -> tuple[list[str], dict[str, frozenset[str]]]:
        item = self.items[number - 1]
        mine = [i for i in reversed(list(self.full)) if len(self.items) == 1 or self.item_for(i) is item]
        vocab = frozenset().union(*(self.fit_terms(i) for i in mine))
        words: dict[str, frozenset[str]] = {}
        for need in item.musts if bare else item.told:
            if need in item.dropped:
                continue
            value = _stem(re.sub('[^0-9A-Za-z]+', ' ', need.split(':', 1)[-1]))
            terms = frozenset((w for w in value if len(w) >= 3 or any((c.isdigit() for c in w)))) & vocab
            if terms:
                words[need] = terms
        pool = [i for i in mine if i not in skip and self.buyable(i) and (words or bare) and all((t <= self.fit_terms(i) if FITPICK_ALL else t & self.fit_terms(i) for t in words.values()))]
        pool.sort(key=lambda i: -sum((len(t & self.fit_terms(i)) for t in words.values())))
        return (pool, words)

    def fit_judge(self, number: int, ids: list[str], models: list[str] | None=None) -> tuple[list[tuple[str, bool]], dict[str, tuple[set[int], set[int]]]]:
        item = self.items[number - 1]
        needs = [(m, False) for m in item.musts if m not in item.dropped] + [(t, True) for t in item.told if t not in item.dropped]
        prompt = FIT + '\n\nREQUEST:\n' + self.request + f'\n\nREQUIREMENTS FOR ITEM {number} ({item.name}, quantity {item.qty}):\n'
        prompt += '\n'.join((f'{k}. {need}' + (" (the shopper's answer)" if told else '') for k, (need, told) in enumerate(needs, 1)))
        prompt += '\n\nALL REQUIREMENTS IN FORCE (for the budget and the pick rule):\n' + self.ledger()
        prompt += '\n\nRECORDS:\n' + '\n'.join((f'{i}: ' + self.record_of(i) for i in ids))
        got = self.ask_json(prompt, models or ([HEAVY, LIGHT] if self.heavy_ok() else [LIGHT]), 1500) or {}
        rows = got.get('rows') if isinstance(got.get('rows'), dict) else {}
        out: dict[str, tuple[set[int], set[int]]] = {}
        for ident in ids:
            row = rows.get(ident)
            if isinstance(row, dict):
                fails = {n for n in (_count(x) for x in _items(row.get('fails'))) if n}
                out[ident] = ({n for n in (_count(x) for x in _items(row.get('meets'))) if n} - fails, fails)
        self.trace.append({'kind': 'fit_judge', 'item': number, 'ids': ids, 'got': _clip(got, 120)})
        return (needs, out)

    def fit_order(self, fits: list[str]) -> list[str]:
        if self.brief.get('pick') == 'cheapest':
            return sorted(fits, key=lambda i: self.price_of(i) if self.price_of(i) is not None else float('inf'))
        return fits

    def fit_buy(self) -> str:
        if self.fit_held or self.cost > SPEND_SOFT or self.left() < 4:
            return ''
        cart = [i for i in self.cart_ids() if not self.sold_out(i)]
        for line in cart:
            item = self.items[0] if len(self.items) == 1 else self.item_for(line)
            if item is None:
                continue
            number = self.items.index(item) + 1
            pool, words = self.fit_pool(number, cart)
            have = self.fit_terms(line)
            if not pool or all((t <= have for t in words.values())):
                continue
            self.fit_held = True
            needs, rows = self.fit_judge(number, [line] + pool[:FITPICK_ROWS])
            told = {k for k, (_, heard) in enumerate(needs, 1) if heard}
            every = set(range(1, len(needs) + 1))
            failed = rows.get(line, (set(), set()))[1] & told
            fits = self.fit_order([i for i in pool[:FITPICK_ROWS] if i in rows and rows[i][0] >= every and (not rows[i][1])])
            if not failed or not fits:
                return ''
            self.trace.append({'kind': 'lever_fire', 'why': 'fitpick_buy', 'line': line, 'pick': fits[0], 'fails': len(failed)})
            return f"A check of each requirement one at a time, the shopper's answers included, says cart line {line} fails: " + '; '.join((needs[k - 1][0] for k in sorted(failed))) + f'. {fits[0]} (its page was read already) meets every one.' + f' In this turn take {line} out of the cart and add {fits[0]} at quantity {item.qty}, then check out;' + f' keep {line} only if its own record shows each of those requirements met.'
        return ''

    def fit_decline(self) -> str:
        if self.fit_declined or self.cost > FITPICK_END_SPEND or (not (FITPICK_BARE or any((item.told for item in self.items)))):
            return ''
        self.fit_declined = True
        cart = [i for i in self.cart_ids() if not self.sold_out(i)]
        covered = {id(self.items[0] if len(self.items) == 1 else self.item_for(i)) for i in cart}
        picks: list[tuple[int, str]] = []
        for number, item in enumerate(self.items, 1):
            if id(item) in covered:
                continue
            bare = FITPICK_BARE and (not [t for t in item.told if t not in item.dropped])
            pool = self.fit_pool(number, cart, bare)[0][:FITPICK_ROWS]
            if not pool:
                return ''
            needs, rows = self.fit_judge(number, pool, self.review_models('fit_bare') if bare else None)
            every = set(range(1, len(needs) + 1))
            fits = self.fit_order([i for i in pool if i in rows and rows[i][0] >= every and (not rows[i][1])])
            if not fits:
                return ''
            picks.append((number, fits[0]))
        if not picks:
            return ''
        self.trace.append({'kind': 'lever_fire', 'why': 'fitpick_decline', 'picks': [i for _, i in picks]})
        if HOLDROOM_ON:
            self.hold_cap = min(HOLD_CEIL, self.cost + HOLD_ROOM)
            self.hold_calls = self.loop_calls + HOLD_CALLS
        many = len(self.items) > 1
        named = ', '.join((f'{i} for item {n} at quantity {self.items[n - 1].qty}' if many else f'{i} at quantity {self.items[n - 1].qty}' for n, i in picks))
        return "A check of each requirement one at a time, the shopper's answers included, found opened listings that meet every one: " + named + '. Add them to the cart and check out; end without a purchase only if a record shows one of them missing a requirement,' + ' then send the ending call again.'

    def swap_fix(self) -> bool:
        cart_tool = self.roles.get('cart')
        cart = self.cart_ids()
        named = [b for b in self.named_better if b not in cart and (not self.sold_out(b))]
        if self.swapped or not named or (not cart_tool) or self.finished:
            return False
        self.swapped = True
        lines = {i: self.cart_lines.get(i, 1) for i in cart}
        out: list[str] = []
        used: set[int] = set()
        for b in named:
            item = self.item_for(b)
            if item is None or id(item) in used:
                continue
            held = [i for i in self.item_lines(item) if i in lines]
            if len(held) != 1:
                continue
            lines[b] = lines.pop(held[0])
            out.append(held[0])
            used.add(id(item))
        if not out:
            return False
        calls = self.cart_calls(lines, out)
        new = [b for b in lines if b not in cart]
        unread = [b for b in new if b not in self.full]
        if not calls or len(calls) + len(unread) + 2 > self.left():
            return False
        self.fill_facts(unread, len(calls) + 2)
        if self.finished or any((b not in self.full for b in new)):
            return False
        self.check_cart(list(lines), SPEND_HARD)
        if not self.verdicts.get(self.check_key(list(lines))):
            return False
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart_tool), 'args', faults)
            if faults:
                return False
            reply = self.act(cart_tool, args)
            if reply is None or self.finished:
                return False
            self.note_cart(cart_tool, args)
            self.observe(cart_tool, reply.get('observation'))
        self.trace.append({'kind': 'lever_fire', 'why': 'swap_fix', 'swapped': len(out), 'opened': len(unread)})
        return True

    def kind_review(self) -> dict[str, Any] | None:
        if not (LEVER_ON and KINDREV_ON) or self.kindrevs >= KINDREV_TRIES or self.cost > SPEND_SOFT:
            return None
        kinds = {i for ids in self.item_groups() for i in ids}
        order = [i for i in reversed(self.seen_order) if i in kinds and (not self.sold_out(i))]
        pool = ([i for i in order if i in self.full] + [i for i in order if i not in self.full])[:KINDREV_RECORDS]
        new = [i for i in pool if i not in self.audit_window]
        if not new:
            return None
        self.kindrevs += 1
        records = ['OPENED ' + _dump(_clip(self.full[i], 200))[:1200] if i in self.full else 'ROW ' + _dump(_clip(self.rows.get(i), 120))[:500] for i in pool]
        prompt = AUDIT + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
        prompt += '\n\nOnly listings of the requested kinds are listed below: those whose page was opened first, then search rows.'
        prompt += '\n\nLISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
        got = self.ask_json(prompt, self.review_models('kind') or [HEAVY, LIGHT], 1500)
        self.trace.append({'kind': 'audit', 'got': got})
        self.trace.append({'kind': 'lever_fire', 'why': 'kind_review', 'pool': len(pool), 'n': len(new), 'opened': sum((1 for i in new if i in self.full)), 'leads': len(self.audit_fits(got))})
        return got

    def relook_call(self, pool: list[str]) -> dict[str, Any] | None:
        records = []
        for ident in pool:
            record = self.full.get(ident)
            records.append('OPENED ' + _dump(_clip(record, 200))[:1200] if record else 'ROW ' + _dump(_clip(self.rows.get(ident), 120))[:500])
        prompt = RELOOK + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
        prompt += '\n\nLISTING RECORDS OF THE REQUESTED KINDS (newest first):\n' + '\n'.join(records)
        got = self.ask_json(prompt, self.review_models('relook') or [HEAVY, LIGHT], 1500)
        self.trace.append({'kind': 'relook', 'got': got})
        return got

    def review_models(self, site: str) -> list[str] | None:
        if not (LEVER_ON and PRICE_ON) or self.cost <= PRICE_FROM:
            return None
        self.trace.append({'kind': 'lever_fire', 'why': 'price_gate', 'site': site})
        return [LIGHT, HEAVY]

    def early_review(self) -> str:
        if not (LEVER_ON and EARLYREV_ON) or self.early_done or self.audited or self.leads or self.finished:
            return ''
        left = self.left()
        if left > EARLYREV_LEFT or left < self.tail() + 4 or self.buys or self.cart_touched or self.cart_ids():
            return ''
        self.early_done = True
        review = self.audit()
        self.trace.append({'kind': 'lever_fire', 'why': 'early_review', 'left': left, 'leads': len(self.leads)})
        if not review:
            self.early_empty = self.audited
            self.audited = False
            return ''
        first = next((i for i in self.leads if not self.sold_out(i)), None)
        if first and self.cost <= SPEND_HARD:
            self.check_cart([first], SPEND_HARD)
            if self.verdicts.get(self.check_key([first])) and first in self.leads:
                self.confirmed = first
                self.trace.append({'kind': 'lever_fire', 'why': 'early_confirmed'})
                review += f" An independent check of {first}'s records says it meets every requirement: add it to the cart at the requested quantity and check out."
        return '[Kyo, steps running low] ' + review

    def reply_words(self) -> list[str]:
        request = set(re.findall('[a-z]{4,}', self.request.lower()))
        out: list[str] = []
        for _, answer in self.qa:
            out += [w for w in re.findall('[a-z]{4,}', answer.lower()) if w not in request and w not in COMMON and (w not in out)]
        return out

    def say_back(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if not (LEVER_ON and SAYBACK_ON):
            return args
        said = self.reply_words()
        if not said:
            return args
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        fixed = self.brief.get('buy_args') if isinstance(self.brief.get('buy_args'), dict) else {}
        out = dict(args)
        for key, spec in props.items():
            if key in fixed or not isinstance(spec, dict) or 'enum' in spec or ('string' not in _kinds(spec)):
                continue
            text = out.get(key)
            empty = not isinstance(text, str) or not text.strip()
            if empty and key not in (schema.get('required') or []):
                continue
            if not empty and (len(text.split()) < 3 or text.strip() in self.rows):
                continue
            room = spec.get('maxLength') if isinstance(spec.get('maxLength'), int) else 400
            words = ' '.join(said[:SAYBACK_WORDS])
            if empty:
                out[key] = ('Ordering what the shopper asked for: ' + words + '.')[:room]
            else:
                have = set(re.findall('[a-z]{4,}', text.lower()))
                if sum((1 for w in said if w in have)) * 5 >= len(said):
                    continue
                out[key] = (text.rstrip(' .') + ' (the shopper asked for: ' + words + ').')[:room]
            self.trace.append({'kind': 'lever_fire', 'why': 'say_back', 'empty': empty})
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
        if (refuse or ENDFIX_ON) and LEVER_ON and EARLYREV_ON and self.confirmed and (not holding):
            done = self.complete_cart()
            if self.finished or self.left() <= 0:
                return ''
            if done and again:
                return 'the checkout of the confirmed listing was refused: ' + done
        if refuse and LEVER_ON and LASTREV_ON and LASTREV_END and (not holding) and (fresh or not self.cart_touched) and (not self.completed) and (not self.confirmed):
            done = self.last_review()
            if self.finished or self.left() <= 0:
                return ''
            if done and again:
                return 'the checkout of the reviewed listing was refused: ' + done
        if refuse and (not holding):
            done = self.reveal_buy()
            if self.finished or self.left() <= 0:
                return ''
            if done and again:
                return 'the checkout of the nearest listing was refused: ' + done
        if refuse and again and LEVER_ON and REVEALGATE_ON and (not self.reveal_last) and (not self.buys) and self.buy_push() and (self.left() >= len(self.items) + 2):
            self.reveal_last = True
            self.trace.append({'kind': 'lever_fire', 'why': 'revealgate', 'n': 0, 'lines': len(self.items), 'stated': self.revealed(), 'step': self.step})
            what = ('the closest listing you opened for each item' if len(self.items) > 1 else 'the closest listing you opened')
            return ('stopped short of an ending without a purchase: ' + ('the shopper has already revealed what they want, so that ending ' if self.revealed() else 'that ending ') +
                    f'scores nothing here. Put {what} in the cart at the requested quantity, then check out.')
        if refuse:
            faults = []
            args = self.end_args(refuse, faults)
            if not faults:
                if self.left() >= 2:
                    self.gate('end')
                if self.finished or self.left() <= 0:
                    return ''
                reply = self.act(refuse, args)
                self.trace.append({'kind': 'forced_refuse', 'reply': _clip(reply, 200)})
                if reply is not None and (not self.finished):
                    self.turned_down(reply)
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
        opening = _keyed(self.view.get('opening')) if TEXTROW_ON else self.view.get('opening')
        if opening is not None:
            self.absorb(opening)
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            intro += '\n\nWHAT THE STORE SHOWS FIRST:\n' + self.show(opening)
        intro += '\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n' + self.status()
        self.chat = [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': intro}]
        loop_spend = LOOP_SPEND if LEVER_ON else SPEND_HARD
        self.work(LOOP_CALLS, loop_spend)
        if not self.finished:
            turned = self.finish(again=True)
            if turned and (not self.finished) and (not self.broke) and (not self.no_credit) and (self.left() > 1):
                self.chat.append({'role': 'user', 'content': '[Kyo] Kyo tried to end the episode and ' + turned + '\nApply what the store says, then finish.\n' + self.status()})
                self.work(self.loop_calls + 10, max(loop_spend, self.cost) + ROUND2_SPEND)
            if not self.finished:
                self.finish()
            if not self.finished and LEVER_ON and END_FILL_ON:
                self.end_fill()
        self.trace.append({'kind': 'end', 'finished': self.finished, 'steps': self.step, 'cost': round(self.cost, 5), 'calls': self.calls})
        return self.trace

    def end_fill(self) -> None:
        refuse = self.roles.get('refuse')
        if not refuse or self.broke:
            return
        faults: list[str] = []
        args = self.end_args(refuse, faults)
        if faults:
            return
        fills = 0
        while not self.finished and (not self.broke) and (self.left() > 0) and (fills < END_FILL_MAX):
            if LEVER_ON and ENDASK_ON and (self.end_reply is not None) and (not self.end_asked):
                fixed = self.end_ask(refuse, args)
                if fixed is not None:
                    args = fixed
            fills += 1
            reply = self.act(refuse, args)
            if reply is None:
                break
            if not self.finished:
                self.turned_down(reply)
        if fills:
            self.trace.append({'kind': 'lever_fire', 'why': 'end_fill', 'fills': fills, 'finished': self.finished, 'step': self.step})

    def end_args(self, refuse: str, faults: list[str]) -> dict[str, Any]:
        args = dict(self.brief.get('refuse_args') or {})
        if LEVER_ON and ENDASK_ON and (self.end_fixed is not None):
            args = {**self.end_fixed, **args}
        elif LEVER_ON and ENDID_ON and self.end_turned and self.end_ids:
            props = (self.schemas.get(refuse) or {}).get('properties') or {}
            args = {**{k: v for k, v in self.end_ids.items() if k in props and (not self.sold_out(v))}, **args}
        return _shape(self.quiet(refuse, args), self.schemas.get(refuse), 'args', faults)

    def turned_down(self, reply: Any) -> None:
        if LEVER_ON and ENDID_ON:
            self.end_turned += 1
            self.end_reply = reply.get('observation') if isinstance(reply, dict) else None
            if self.end_turned <= 3:
                self.trace.append({'kind': 'lever_fire', 'why': 'end_turned', 'by': 'kyo', 'n': self.end_turned, 'ids': len(self.end_ids), 'step': self.step})

    def end_ask(self, refuse: str, sent: dict[str, Any]) -> dict[str, Any] | None:
        self.end_asked = True
        if self.no_credit or self.cost > 2 * SPEND_HARD:
            return None
        seen = list(dict.fromkeys(reversed(self.open_order))) + [i for i in self.leads if i not in self.open_order]
        seen = [i for i in seen if not self.sold_out(i)][:8]
        ask = f"An agent in an online store is ending the episode without a purchase with the store tool {refuse}. The store turned the call down.\nARGUMENTS SENT: {_dump(sent)}\nSTORE REPLY: {self.show(self.end_reply)[:1500]}\nTOOL SCHEMA: {_dump(self.schemas.get(refuse))}\nLISTINGS THE AGENT OPENED (newest first): {', '.join(seen) or 'none'}\nReturn ONE JSON object and nothing else: the arguments for {refuse} that end the episode without a purchase and that the store will take."
        got = self.ask_json(ask, [LIGHT], 1500)
        props = (self.schemas.get(refuse) or {}).get('properties') or {}
        fixed = {k: v for k, v in got.items() if k in props} if isinstance(got, dict) else {}
        self.trace.append({'kind': 'lever_fire', 'why': 'end_ask', 'got': len(fixed), 'step': self.step})
        if not fixed:
            return None
        self.end_fixed = fixed
        faults: list[str] = []
        args = self.end_args(refuse, faults)
        return None if faults else args

    def loop_stop(self, spend_cap: float) -> float:
        if LEVER_ON and STATESTOP_ON and (spend_cap == LOOP_SPEND) and (self.cost >= LOOP_SPEND) and self.cart_ids():
            if not self.state_stopped:
                self.state_stopped = True
                self.trace.append({'kind': 'lever_fire', 'why': 'statestop', 'lines': len(self.cart_ids())})
            return max(SPEND_HARD, self.hold_cap)
        return max(spend_cap, self.hold_cap)

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.finished and (not self.broke) and (self.left() > 0) and (self.loop_calls < max(loop_cap, self.hold_calls)) and (self.cost < self.loop_stop(spend_cap)):
            if self.ready_to_ask():
                summary = self.interview()
                if summary:
                    self.chat.append({'role': 'user', 'content': summary + '\n' + self.status()})
                continue
            if LEVER_ON and CARTLOOK_ON and self.look_told:
                note = self.cart_look()
                last = self.chat[-1] if self.chat else {}
                if note and last.get('role') == 'user' and isinstance(last.get('content'), str):
                    last['content'] += '\n\n' + note + '\n' + self.status()
                elif note:
                    self.chat.append({'role': 'user', 'content': note + '\n' + self.status()})
                if self.finished:
                    break
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
                self.chat.append({'role': 'user', 'content': (summary or 'The interviewer had nothing to ask.') + '\n' + self.status()})
                continue
            self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
            self.trace.append({'kind': 'turn', 'text': content[:300], 'calls': [c['function'] for c in calls]})
            replies = self.run_calls(calls)
            early = self.early_review() if replies else ''
            point = self.point_now() if replies and (not early) else ''
            for index, (cid, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += ('\n' + early if early else '') + ('\n' + point if point else '') + '\n' + self.status()
                self.chat.append({'role': 'tool', 'tool_call_id': cid, 'content': text})

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    _TEXTROWS[0] = 0  
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
