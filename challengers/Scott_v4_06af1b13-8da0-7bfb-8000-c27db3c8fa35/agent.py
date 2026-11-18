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
SPEND_SOFT = 0.052
SPEND_HARD = 0.078
HEAVY_ROOM = 0.02
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
BRIEF_BY = 120.0
BRIEF_GAP = 10.0
OUTAGE_BY = 150.0
SHOW_CHARS = 450
HUNT_STEPS = 24
ENV_GAP = 0.25
OPEN_GAP = 0.5
TABLE_KEYS = 18
CHEAPER_SHOWN = 4
QUIET_WORD = 4
OPEN_LATER = 2
PAGE_ATTR_CHARS = 700
PAGE_REST_CHARS = 300
NO_NAME = ('none', 'null', 'n/a', 'unknown')
ASK_TURNS = 2
MESSAGE_FIELD = '(?:^|_)(?:texts?|body|messages?|notes?|contents?|questions?|msg|say)(?:_|$)'
NOT_MESSAGE = '(?:^|_)(?:id|ids|type|format|code|lang|language|kind)$'
TRAIL_ROOM = 15
FIT_PAGES = 2
CRASH_TURNS = 3
OPEN_MIN_STEPS = 20
OPEN_ROWS = 5
OPEN_ITEMS = 2
OPEN_SHARE = 5
OPEN_EACH = 6
OPEN_NOT = '\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or|same|listing)\\b'
SHORT_KEEP = 8
SET_SHOWN = 2
STOCK_OUT = '^(?:no|none|false|0|n)$|out[ _-]?of[ _-]?stock|sold[ _-]?out|unavailable|not[ _-]?available'
HUNT_WHOLE = 49
HUNT_GRACE = 25.0
HUNT_VALUE_CHARS = 30
ASK_WORDS = '\\b(?:one|each|single|first|time|many|much|several|list|once|separate\\w*|split|short\\w*|questions?|ask\\w*|turn|lot|too|all)\\b'
HUNT_ROUNDS = 2
HUNT_KEEP = 10
HUNT_IDLE = 4
HUNT_PAGE_NEW = 5
HUNT_FIT_WORDS = 4
HUNT_FIT_OPENS = 4
HUNT_COMMON = 0.6
HUNT_BY = 200.0
TITLE_CHARS = 200
MIX_LABELS = 3
TEXT_CHARS = 3000
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3
AUTO_PAGES = 6
CHEAPER_OPENS = 3
TASK_SECONDS = 300.0
TIME_MARGIN = 20.0
CALL_FLOOR = 6.0
ASK_BY = 165.0
QUESTION_BY = 210.0
WIND_DOWN = 200.0
LAST_TURN = 255.0
EMPTY_ROOM = 12.0
LATE_BY = 190.0
LATE_STEPS = 8
SCARCE_MESSAGES = 2
TIGHT_MESSAGES = 2
BUSY_SECONDS = 20
RECOVERY_STEPS = 8
LIST_SIZE = 12
LIST_MIXED = 8
LIST_STEPS = (1, 3)
LIST_STEPS_MIXED = (3, 3)
CAP_ARG = '(?:^|_)(?:cap|max|maximum|ceiling|under|up_?to)(?:_|$)'
CAP_NOT = 'result|row|page|count|size|num|item|step'
CAP_PRICED = 'price|cost|budget|spend|money|amount'
CAP_LOW = '(?:^|_)(?:min|minimum|low|lowest|least|from|start|offset|floor)(?:_|$)'
EARLY_TRAILS = 3
CAP_CHANCE = 0.05
WALK_ROOM = 10
WALK_CALLS = 4
SHELF_ARG = 'section|department|categor|label|shelf|aisle|(?<!age)(?:^|_)group(?:_name|_label)?$'
LIST_MODE = 'field|filter|facet|brows|categor|list|structured'
AMOUNT = '\\d[\\d.,]*\\s*[a-z"\']{1,12}'
FLOOR = '\\b(?:at least|no (?:less|fewer) than|min(?:imum)?)\\s+(' + AMOUNT + ')\\s*$'
QUERY_WORDS = '(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword'
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_llm_short: dict[int, ProxyClient] = {}
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)

def _prompt_guide() -> str:
    return 'You act as Ridge, a meticulous purchaser serving a single buyer in an online market. You can act solely by calling the market\'s calls. The run is scored 1 or 0: it earns 1 solely if the finish is precisely correct. A purchase has to contain every requested item at the requested count, every meeting every constraint in force when you complete order completion, inside the spending limit; or, when the market truly has nothing that meets eachthing, the run must end using the no-purchase call in the way the request asks.\n\nHOW THIS STORE BEHAVES\n- Each market call spends one turn and turns are limited; thinking costs nothing. Tool titles, argument titles and outcome shapes are particular to this market: consult every call\'s write-up and schema and use the exact argument titles and permitted settings.\n- Search calls can be narrow: a category filter, a keyword mode that may solely consult the initial words of the query, a sheet size of a small numeric of rows, a token or offset for the following sheet. Compare the reported sum using the rows you have seen and sheet on (with the exact token or offset the market gives) if the answer depends on seeing every of them. Keyword queries work best brief and setting-first (the wanted setting plus the product noun); a market\'s look-up can consult spec field settings and shop titles as well as titles and may want every word, thus a small numeric of required settings plus the noun can array nearly the full matchting set, and a model or part code may find nothing (one try tells). A filtered look-up that returns no rows typically means the filter setting or the filter parameter is incorrect: retry using the label precisely as outcome rows surface it, or using a different filter parameter.\n- Search rows are summaries. Stores split a row\'s snapshot differently: one view can surface its spec fields (spec fields), a different solely its offer and write-up, and a side-by-side view may carry either. The ledger titles the action that surfaces spec fields once Ridge knows it. Judge every constraint on a spec field out of those spec fields, so consult them for every row you may buy; single call that surfaces multiple rows saves turns if it surfaces what you need. Do not repull a row if nothing has changed since you consult it.\n- Questions to the buyer are handled for you by Ridge\'s question interviewer, so you do not note the buyer yourself. Its answers arrive inside the conversation and are added to the working ledger.\n- The request can as well inquire for turns alengthy the path: an sign-off, a note, something to check or consult first. The ledger lists them. Ridge carries out the ones due merely ahead of the order or the ending itself when you call the order completion or the ending call, thus do not wait for those and do not end lacking a purchase since of them; do whichever others using the market\'s calls (product questions go via Ridge\'s question interviewer).\n\nWhere A LISTING FITS\n- The sort of product is decided by the row\'s shelf label: solely the label that matches the requested item as a full counts, do not a neighbouring one.\n- A constraint on a spec field is met when the row\'s spec field for that spec field states the setting. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash array is enough. Values match by their words: case, single or plural form and plain equivalent words are fine, and a setting that contains the required setting\'s words counts ("deep teal" meets teal, "extra roomy match" meets roomy). A setting using different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted ahead of comparing numbers, limits include the edge setting, and a range must lie wholly inside a limit.\n- An spec field whose key titles the constraint\'s spec field (under that title or a plain synonym) decides: if it says something else, the row fails whatever alternate keys or the title say. A setting below a key that adds the item\'s own title to the spec field (a bag\'s "bag shape") counts even if the bare key ("shape") says something else. A key for a different spec field, even a related one (a handle colour is not the body\'s colour), decides nothing regarding the constraint. Ridge adds [evidence] lines following pulled rows using the keys whose titles match a constraint.\n- When the row has no spec field for that spec field, the wanted setting stated below a different spec field (for example a array of styles or features that includes it) or in the title or write-up counts. A numeric of a different sort does not (a weight is not a capacity), nor a bare figure whose text does not say what it metrics.\n- Where the shelf label itself states a constraint\'s setting (a season or audience in the label), that constraint is met by the category. A row below the requested label is that sort of product even when its title or write-up titles something else, so judge it on the constraints, cost and stock.\n- A constraint can point at a different row instead of giving a setting ("the identical X as the line sold by <shop> using <spec field>", "within N of the X on the sheet of ..."). Resolve it first: find that row (look-up its category, brand, shop or words), pull its sheet in full using the single-item or side-by-side call (a look-up row is not enough), and if that sheet refers you to a further line for the spec field, locate and pull that one the identical way; the final sheet gives the setting; if the constraint asks for what goes using that row\'s setting (its partner, its counterpart) and the sheet titles what its setting pairs with, the constraint is that alternate setting. The order counts solely if every sheet of that chain was pulled. When the wanted setting is anything but that alternate row\'s setting, the rule excludes solely rows sharing that exact setting, and do not end lacking a purchase since of it alone.\n- When a row you are regarding to buy sends you to a different sheet for more regarding it, scan at that sheet ahead of you buy.\n- When one requested item must match a different (an accessory for a device), choose the two together outward of the rows\' own words (a array of what it matchs, a model or series code that both sheets carry). When no accessory that meets its own constraints matchs the device you picked, switch toward a different device that meets its constraints and that such an accessory matchs; do not end lacking a purchase as that switch is untried.\n- Respect bans and closed lists. Wishes ("ideally", "would be nice") are final tie rules only.\n- Never buy an item that the market surfaces as out of stock; the sum (cost times count, summed above every entry) may not exceed the spending limit; equal is fine.\n\nPICKING LISTINGS\n- When the request asks for the cheapest (lowest sum, smallest cost), the order has to hold the lowest-costd row among ALL rows that match. Locate the full matchting set (sheet via it: outcome rows are frequently in no cost order, and Ridge consults the following sheets of a brief outcome itself), check candidates in rising cost order, and take the initial that fully matchs. Where a look-up takes a cost ceiling, use it: array the category below the cost of your current best pick, thus solely cheaper candidates are left to check.\n- When the request asks for the majority of or least of some metric, weigh every matchting row on that metric and take the extreme; a row that does not state the metric cannot win.\n- Otherwise whichever row that meets eachthing is correct: do not waste turns hunting for something better.\n- Cart every product as soon as you have pulled a row that meets every constraint for it (using a cheapest, majority of or least pick: once you have weighd the matchting set), after that move on. Stores frequently sell out a basketed entry soon following it is added, and an early basket leaves turns to replace it.\n- A request for multiple different items needs one basket line per item, every at its requested count.\n\nMID-EPISODE CHANGES\n- The buyer or the market can change things: a new spending limit, a constraint added or dropped, a cost change, a line out of stock. They appear inside call outcomes, frequently when you try to complete order completion (a decline); a note counts whichever call\'s outcome surfaces it, and the working ledger lists the constraints in force. The run goes on following a decline: apply every change, fix the basket and complete order completion again.\n- Price and stock can change silently: Ridge re-consults the basket\'s lines and looks at the basket when you call the order completion, so do not spend a turn on either.\n\nCLOSING THE EPISODE\n- Put precisely the chosen lines in the basket (remove anything else), after that complete order completion using a action on its own.\n- End lacking a purchase solely following a real look-up finds no line that meets eachthing. Running brief of turns proves nothing of the sort: after that complete order completion the best basket you have. Follow the request\'s wording for that ending, and if a remark is needed, keep it brief and title no row.\n- Every reply has to call at least one action. Keep replies brief: what you know, after that the following move.'
GUIDE = _prompt_guide()

def _prompt_brief() -> str:
    return 'Move the buyer request to structured shopping data for an agent. Read the buyer\'s request and the market\'s calls and provide back ONE JSON blob (compact, no indentation, no surrounding text) using these keys:\n\n"items": a array using one entry per separate product line the buyer wants bought, every {"title": the sort of product in the request\'s words, "shelf": the market\'s shelf label for it when a call lists labels (copy it precisely), or the category the request itself titles for it ("a coat in rainwear" gives "rainwear"), else "", "count": units of that one row (integer, 1 if not stated; a pair or a set that is one row counts as 1), "musts": a array of hard constraints on that product, every "spec field: constraint" using the exact setting, limit, exclusion or closed list. Name the spec field using the request\'s own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring spec field. A must whose setting is consult out of a different row is ONE has to holding the full write-up of that row ("rim colour: unlike the rim colour of the kettle Ovarno sells using a glass lid"): words that describe that alternate row (its shop, its sort, "with <spec field> <setting> and <spec field> <setting>" following it) stay inside that one must and are do not musts of this product}. Never put cost, spending limit, stock or count to musts. The story around the request (the person meant to employ it, where it will stand, what went incorrect ahead of, the occasion behind it) is background, not a must, except when it states a spec field\'s setting or limit outcorrect; and do not turn words that are part of what the product is called (a season or an audience titled using the product) to musts: those belengthy to "title". A use or occasion the request makes a spec field of the product itself ("blazers for business wear", "a dress made for parties", "a bag, and it is for a wedding", "the occasion has to be party") IS a must on that spec field ("occasion: business"), except when the market\'s shelf label alconsulty titles it; a particular person\'s event, a person or a setting the product is bought for ("to wear at his wedding", "for a colleague"), and how it ought to sit, feel or scan in use, stay background. A product the request titles inside one phrase joined by "and" (one "I want ..." using single count, frequently the title of one market department) is Single item; make separate items solely if the request gives every product its own count or its own constraints.\n"spending limit": the majority of the buyer will pay in sum as a numeric in the request\'s own unit, or none.\n"spending limit_unit": that unit in words (for example "cents"), or none.\n"spending limit_scope": "sum" if the limit covers the full order, "each" if it is per item, none if there is no limit.\n"in_stock_only": yes except when the request clearly accepts items that are not available.\n"pick": "cheapest" if the request asks for the lowest cost or smallest sum, "most" or "least" if it asks for the extreme of some metric, else "any". Remarks that money is tight are not a pick rule.\n"pick_metric": what is maximised or minimised for "most"/"least", else "".\n"notes": how numerous notes the buyer allows the agent to send (integer), or none if not stated.\n"turns": how numerous turns, moves or action calls the request allows (integer), or none.\n"lookups": a array of the musts (copied precisely) whose setting must initial be consult out of a different row the request points at, else [].\n"rules": a array of every instruction the request gives regarding how to work or how to end, every inside at majority of 12 words.\n"wishes": a array of things the buyer may merely like.\n"ask_first": a array of things the request explicitly says to check using the buyer ahead of choosing (empty if none).\n"process": a array of the turns the request requires the agent to take besides finding and buying the correct items (getting the buyer\'s sign-off, telling or surfaceing the buyer something, checking something using them, consulting something first, using a particular action), every {"turn": what must be done, in at majority of 15 words, "when": "choose" (ahead of the products are chosen), "order" (correct ahead of checking outward) or "end" (just ahead of ending lacking a purchase), "notes": how numerous notes to the buyer it takes (0 if none)}; [] if the request asks for no such turn. Requirements on the items, the spending limit, stock, the limits on notes and turns, and how toward end if nothing matchs are not turns.\n"quiet_decline": yes when the request says not toward suggest or title alternatives when nothing matchs, no if it asks for them, none when it says nothing.\n"matchs": a array of {"item": item number, "with": item number} for every requested item that the request says must work with, match or suit ANOTHER requested product (the one the agent picks); [] if none.\n"roles": an structure naming the exact call for every job, or none when no call does it: "look-up", "pull" (surfaces one row\'s spec fields or spec fields; when two single-row calls exist, the one whose write-up promises spec fields, spec fields or specifications), "offer" (surfaces one row\'s current offer, such as its cost and stock, if that is a different call out of "pull"), "weigh" (multiple rows inside full), "ask" (sends a note to the buyer), "basket" (changes the basket), "basket_view" (surfaces the basket), "buy" (checks out), "refuse" (ends lacking a purchase).\n"look-up_words": {"call": the call that look-upes rows by free words (where one call lists a category and a different takes words, the one that takes words), "field": its parameter that takes the words, "args": the fixed arguments that make it look-up by words (for a action using a mode argument, that mode), else {}}.\n"buy_args": the fixed arguments that make the "buy" call complete order completion (for a call using a mode argument, that mode), else {}.\n"refuse_args": the fixed arguments that build the "refuse" call end lacking a purchase, else {}.'
BRIEF = _prompt_brief()

def _prompt_plan() -> str:
    return 'Pick the questions a shopping agent ought to send to a buyer. Shoppers hold a little numeric of firm preferences they did not write down, every regarding one spec field of one requested product. A question reveals a preference solely when it titles that exact spec field; a spec field the buyer has no view on gets "don\'t mind", and single the buyer does not think regarding at every gets a puzzled reply, so every spec field titled spends part of a note.\n\nYou get the request, the items using what the request alconsulty settles for each, and a numbered array of spec field keys seen on pulled rows of every item\'s category, using how numerous of those rows carry the key and example settings. An unwritten preference may be on a spec field that no listed key titles.\n\nChoose the spec fields in this order:\n0. What the request itself asks you to check using the buyer (listed below ASK FIRST if there is any), every titled by the spec field it is about.\n1. Hinted spec fields. The request\'s story frequently points at an unwritten preference lacking giving its setting: a sentence regarding something the buyer cannot judge or leaves toward you, someone whose taste decides, the product\'s looks or the setting it belengthys in, the person meant to use or wear it, or what is inside it. For every such sentence, title the one spec field it points at and the item it is about, in the order the story gives them.\nName a hinted spec field by a listed key of that item when one matchs the hint (a hint regarding who will wear it: the key for gender or audience; regarding its looks: the listed key for colour, pattern or finish that majority of of its rows carry); a spec field no line of the item carries comes following the item\'s three majority of carried keys.\n2. Then the listed keys, ranked: favour the spec fields a buyer of that sort of product chooses amid initial (often its type or style, its form or match, its size, capacity or dimensions, its material, colour or pattern, whichever that sort of product is bought by) and keys carried by majority of of the item\'s rows; once five or more rows were consult, avoid keys seen on solely one or two of them.\nSkip a spec field the request alconsulty settles for that item (a constraint titles that spec field; a key whose settings solely repeat a required setting, such as a pattern key surfaceing the required decoration, is NOT settled), a second title for a spec field alconsulty chosen, a key whose settings every array multiple purchase options at once, identifiers, model titles, warranty and alternate logistics. Pack sizes, counts and quantities per pack are spec fields like whichever other, and a key marked [not the stated ...] is a different spec field compared with that constraint, whatever words its title shares using it. Rank every product on its own: its hinted spec fields first, after that its listed keys.\n"label" is the spec field\'s plain title: for a listed key, every of the key\'s words inside plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; do not briefened to "colour" or "finish"); solely the spec field\'s title, no example settings, no brackets, no second spec field.\nChoose up to COUNT for every item, best first. Reply using precisely one JSON blob and no surrounding text: {"ask": [{"item": item number, "label": "...", "hint": "at majority of 8 of the story\'s words it comes from, or an empty string"}]}, every item\'s picks in rank order'
PLAN = _prompt_plan()

def _prompt_hear() -> str:
    return 'A shopping agent has questioned a buyer. Read every answer and give back ONE JSON blob and no surrounding text:\n{"needs": [{"item": item number, "need": "spec field: constraint", "points": yes when the setting is to be consult out of a different row, else false}], "drop": [{"product": item number, "spec field": "a constraint the buyer now gives up"}], "free": [{"item": item number, "spec field": "..."}], "crowded": [numbers of the questions whose answer complains that the note asked regarding too numerous things or wants them asked one at a time; not answers that solely say they did not follow it or want it inside simpler words], "spending limit": fresh sum limit as a numeric or none}\n- "needs": every firm preference an answer states, using the exact setting or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the spec field the answer itself titles, that may differ out of the one asked about; multiple answers regarding one spec field make one need. Copy the setting word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3"); the setting is solely what a row could state, outward of the wanted setting or its limit word to its end, lacking the framing the answer puts around it ("oval" for "matching oval", "settled, using oval" or "oval and no surrounding text", "at least 30 cm" for "equal to at least 30 cm").\n- An answer can provide the setting by pointing at a different row (a setting like, unlike or near the one on some shop\'s product, that product picked out by a small numeric of of its spec fields): that is ONE need on the spec field the answer titles, using "points": true. Its setting is the answer\'s own words copied precisely, out of the comparison words to the end of that write-up, including what it says regarding a row that sheet may title; do not briefen or reword it. The spec fields describing the alternate product are not needs of this item.\n- "free": the spec field of every question the buyer said they do not mind about.\n- Folksy or roundabout wording that titles a setting yet states that setting. Wording that refuses to bargain above a setting (it will not be traded, given up or changed) states that setting as wanted; write "not X" solely if the answer rejects X itself.\n- The setting is solely what the spec field ought to be. Words that merely join a spec field toward its setting (such as matching, set to or coming inside as) are not part of it: "size set to large" states large. This does not touch a setting pointed at on a different row, that is copied full.'
HEAR = _prompt_hear()

def _prompt_note() -> str:
    return 'During a shopping run, the market surfaceed this note (it can come out of the buyer or the market). Reply using precisely one JSON blob and no surrounding text:\n{"sort": "change" if the note changes the task (spending limit, constraints, a cost, availability, a decline of the order), "error" if it solely reports that a action was malformed or (not accepted), "other" for anything else, "spending limit": the new sum limit as a numeric in the request\'s unit, or none, "add": [{"item": item number, "need": "spec field: constraint"}], "drop": [{"item": product number, "need": "the constraint no lengthyer wanted"}], "gone": [row ids said to be unavailable], "sold_out": yes if it says a chosen row is gone, outward of stock or no lengthyer available (even lacking naming it), "cost_moved": yes if it says a cost changed, "summary": "single brief sentence"}\nA replaced constraint is a drop of the old one plus an add of the new one. Report every change in the note.'
NOTE = _prompt_note()

def _prompt_check() -> str:
    return 'You are the final gate ahead of order completion. Decide outward of the snapshots surfacen whether this basket is precisely correct.\nReading rules: a constraint is met if the row\'s spec field for that spec field states the setting (any key spelling, one element of a array is enough, case, singular/plural and plain equivalent words count, a setting containing the required setting\'s words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an spec field whose key titles the identical spec field as the constraint decides, and if it says something else the row fails, though a setting below a key that adds the item\'s own title to the spec field (a bag\'s "bag shape") counts even if the bare key ("shape") says something else; a key for a different spec field, even a related single (a handle colour is not the body\'s colour), decides nothing; if the row has no spec field for that spec field, the wanted setting stated below a different spec field or in the title or write-up counts (a numeric of a different sort does not: a weight is not a capacity; nor does a bare figure whose text does not say what it metrics); a constraint the shelf label itself states is met by that category; the shelf label has to match the requested item as a full; a rule of being unlike a different row\'s setting counts as met except when the snapshots surface that identical setting. Each requested product must appear as a separate basket line at the requested count, no surrounding text can be in the basket, nothing surfacen out of stock, and the sum of cost times count above every lines has to be inside the spending limit. The newest snapshot of a row wins.\nWhen the pick rule is cheapest, majority of or least, as well scan at the alternate rows surfacen: if one of them clearly matchs every constraint and beats the basket\'s choice on that rule, the basket is incorrect.\nReply using precisely one JSON blob and no surrounding text: {"ok": yes or false, "problems": ["brief, concrete problem", ...], "better": [line ids that ought to replace a basket line, best first]}. Use ok=false solely for a problem the snapshots clearly surface or a constraint no snapshot addresses.'
CHECK = _prompt_check()

def _prompt_reading():
    return 'the row\'s spec field for a spec field decides if its key titles that identical spec field (any key spelling, single element of a array is enough, equivalent words and unit conversions count, a setting containing the required setting\'s words counts, such as "deep teal" for teal, limits include the edge); a setting below a key that adds the item\'s own title to the spec field (a bag\'s "bag shape") counts even if the bare key ("shape") says something else; a key for a different spec field, even a related single (a handle colour is not the body\'s colour), decides nothing; if no spec field titles that spec field, the wanted setting stated below a different spec field or in the title or write-up counts (a bare figure that does not say what it metrics counts for nothing); a constraint the shelf label itself states is met by that category; the shelf label must match the item; a rule of being unlike a different row\'s setting counts as met except where the snapshots surface that identical setting; out-of-stock rows and rows that break the spending limit do not match.'
READING = _prompt_reading()

def _prompt_audit():
    return 'A shopping agent plans to end using no purchase since it thinks nothing matchs. Check the row snapshots surfacen against the constraints, using these consulting rules:' + READING + 'Records marked ROW were seen solely as a look-up outcome line: their sheet (the spec fields) has not been consult, thus they can be neither passed nor failed on a spec field the row does not surface.\nThe constraints were written down by a quick consulter, so initial weigh every single using the REQUEST and the ANSWERS FROM THE SHOPPER. A constraint that solely repeats the story (who it is for, when, where or why it will be used, what went incorrect ahead of, how it ought to feel) and that the request does not demand of the product goes below "not_firm". A setting that picked up words that solely join a spec field to its setting goes below "corrected" using the plain setting. Judge the rows lacking the not_hard constraints and using the corrected settings. A constraint the request or an answer states outcorrect stays, even if no row meets it: ending lacking a purchase is after that correct.\nReply using precisely one JSON blob and no surrounding text: {"not_firm": [{"item": item number, "need": the constraint copied precisely}], "corrected": [{"product": item number, "need": the constraint copied precisely, "setting": "spec field: plain setting"}], "matchs": {"item number": [row ids that appear toward meet every constraint, cheapest first, at majority of 4]}, "pull": [ids of ROW rows of a requested sort, inside the spending limit, whose sheet could surface that they match, majority of promising first, at majority of 6], "look-up": [at majority of 4 of {"item": item number, "query": "a keyword look-up of two or three words, not tried yet, that could surface a matchting row: a different title for the product using one required setting (lengthy queries frequently match nothing)"}], "reason": "one sentence"}'
AUDIT = _prompt_audit()

def _prompt_unmix():
    return 'A buyer\'s request was consult toward hard constraints on a product. Some constraints take their setting out of a different line the request points at, and the words that pick outward that alternate row (its sort, its shop, the spec fields titled following it using "using ...") are at times consult by mistake as constraints on the product being bought. Using the request\'s wording, say that of the OTHER REQUIREMENTS below solely describe that alternate row. Reply using precisely one JSON blob and no surrounding text: {"describe_other": [those constraints copied precisely; empty if every one is a real constraint on the product]}'
UNMIX = _prompt_unmix()

def _prompt_trail():
    return 'A shopping agent is nearing order completion. Some constraints take their setting out of a different row instead of stating it, and a row\'s sheet can send the consulter on toward a further row for that spec field. The order counts solely if the agent has pulled in full every line alengthy every such chain. Follow every constraint\'s chain out of the constraint\'s own write-up, link by link, via the sheets\' own words, using the snapshots below (OPENED: the agent has pulled that sheet; ROW: its sheet not consult, seen solely as a look-up outcome line or by its spec fields).\nReply using precisely one JSON blob and no surrounding text: {"chains": [{"need": the constraint copied precisely, "links": [{"id": the line id if a snapshot below is that link, else "", "about": how the constraint or the previous sheet describes that line, "find": "for a link not found: look-up words (its shop and the settings the write-up gives)"}], "done": yes if the final link is an OPENED sheet that states the spec field\'s setting itself (also when the constraint asks for a setting alternate compared with or unlike that one: the sheet\'s setting is what it is metricd against), else false, "setting": the spec field\'s setting as stated on that final sheet; if the constraint asks for what goes using that setting and the sheet titles what its setting pairs with, that alternate setting; or ""}]}\nA snapshot is a link solely when it matches the full write-up of that link: the shop or brand, the sort of product AND every spec field setting the write-up gives (for example "a 10 pieces multipack" rules out the identical shop\'s line using 10 capsules and no multipack); if no snapshot matches every of it, that link is not found yet, even when the identical shop sells alternate rows of that sort. Provide "find" for such a link: the shop\'s title and the write-up\'s settings, as a little numeric of look-up words.\nList every chain\'s links in order and stop at the initial link whose sheet is not OPENED or (not found), since its words are yet unknown.'
TRAIL = _prompt_trail()

def _prompt_gate():
    return 'A shopping agent is regarding toward reach STAGE. The buyer\'s request asks for the turns below alengthy the way. Decide outward of the numbered RECORD of what the agent has done whether every turn due ahead of this STAGE is done for the basket as it is now. A turn counts as done solely if a RECORD line surfaces it being done and, for a move regarding the basket or the order, that line comes following the basket final changed and its reply grants it; questions on a product\'s spec fields do not do a move.\nWhen a turn is yet missing, give the one market call that does it next: a call out of TOOLS using its arguments. A note to the buyer goes via the messaging call: one brief sentence or question regarding that turn alone, in plain words, naming no product spec fields or settings the turn does not need. Give no action if every due turn is done or if the notes or turns left cannot do the missing one.\nReply using precisely one JSON blob and no surrounding text: {"turns": [{"move": the turn, "done": yes or false, "lines": [the RECORD entry numbers that surface it done]}], "call": {"call": "...", "args": {...}} or none, "hold": what a reply asks to change ahead of going on, or "", "needs": [{"item": product number, "need": "spec field: constraint"} for every firm constraint such a reply states]}'
GATE = _prompt_gate()

def _prompt_swap():
    return "Single basket line in a shopping agent's basket merely went unavailable. Find replacements for that basket line among the alternate line snapshots surfacen, using these consulting rules:" + READING + 'Reply using precisely one JSON blob and no surrounding text: {"matchs": [{"id": row id, "cost": its cost as a numeric as the snapshot surfaces it} for every row that appears toward meet every constraint of the sold-out line\'s item, at majority of 6], "reason": "one sentence"}'
SWAP = _prompt_swap()

def _prompt_shape():
    return 'A market call produced the payload below. Name its parts by the exact keys the outcome and the call\'s parameters employ. Reply using precisely one JSON blob and no surrounding text: {"rows": the key of the array of rows, "count": the key holding how numerous rows matched in every (not how numerous are surfacen), or none, "next": the key holding the setting that asks for the following sheet, or none, "next_arg": the call parameter that takes that setting, or none, "cost": the key, inside a line, of the row\'s cost, or none, "stock": the key, inside a row, that says whether the row can be bought now, or none}'
SHAPE = _prompt_shape()
GUARD_KEYS = 3
GUARD_SHARE = 0.0
GUARD_GAP = 0.34
NOT_PROPERTY = '\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b'
CATCH_ALL = '^(?!.*\\b(?:types?|styles?|kinds?|shapes?)\\b).*\\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\\w*|accessor\\w*|extras?|misc\\w*)\\b'
WIDEN_EACH = 6
FIT_CODE = '[A-Z][A-Za-z]{0,5}[ -]?(?:\\d{3,6}|\\d{1,5}[xX]{1,3})'
PART_CODE = '[A-Z]{1,5}\\d{0,3}-?\\d{3,7}'
VALUE_ROOM = 12
VALUE_KEEP = 8
LOOK_ALIKES = 2
LINK_SEARCHES = 3
LINK_ROWS = 3
LINK_FAMILIES = 12

def _codes(text: Any) -> list[str]:
    raw = str(text or '')
    found = re.findall('\\b' + FIT_CODE + '\\b', raw)
    out: list[str] = []
    year_pat = '[A-Z][a-z]+ (?:19|20)\\d\\d'
    for token in found:
        if re.fullmatch(year_pat, token):
            continue
        normalized = re.sub('[\\s-]+', ' ', token).strip().lower()
        out.append(normalized)
    return out

def _code_hit(pattern: str, code: str) -> bool:
    wide, plain = (re.sub('[\\s-]+', '', pattern), re.sub('[\\s-]+', '', code))
    stem = wide.rstrip('x')
    if stem != wide and stem[-1:].isdigit():
        return plain.startswith(stem) and len(plain) == len(wide)
    return wide == plain

def _fit_marks(text: Any) -> dict[str, Any]:
    text = str(text or '')
    own = re.search('[\\[(]\\s*(?:part\\s+)?(' + PART_CODE + ')\\s*[\\])]', text, re.I)
    codes: list[str] = []
    for hit in re.finditer('\\bmodel\\s*(?:no\\.?|number|#)?\\s*[:.]?\\s*(?!(?:no|number)\\b)(' + FIT_CODE + ')\\b', text, re.I):
        codes += _codes(hit.group(1))
    for hit in re.finditer('\\b(?:also sold as|also known as|sold also as)\\s+([^.;]+)', text, re.I):
        codes = codes + _codes(hit.group(1))
    works: list[str] = []
    bars: list[str] = []
    for hit in re.finditer('\\b(?<!not )(?<!never )' + FIT_WORDS + '\\s*:?\\s+([^.;]+)', text, re.I):
        if re.search('\\bpart\\s+' + PART_CODE + '\\s+(?:also\\s+)?$', text[max(0, hit.start() - 40):hit.start()], re.I) or re.match('part\\s+series', hit.group(1), re.I):
            continue
        sides = re.split(FIT_EXCEPT, hit.group(1), maxsplit=1, flags=re.I)
        works += _codes(sides[0])
        if len(sides) > 1:
            bars += _codes(sides[1])
    also = [(hit.group(1).upper(), _codes(hit.group(2))) for hit in re.finditer('\\bpart\\s+(' + PART_CODE + ')\\s+also fits\\s*:?\\s*([^.;]+)', text, re.I)]
    more = re.search('\\b(?:more|details|see|read)\\s+(?:on|at|under|about)?\\s*(?:the\\s+)?part\\s+(' + PART_CODE + ')', text, re.I)
    fam, named = _series_marks(text)
    return {'own': own.group(1).upper() if own else '', 'codes': codes, 'works': works, 'bars': bars, 'also': also, 'more': more.group(1).upper() if more else '', 'fam': fam, 'named': named}

def _series_marks(text: str) -> tuple[list[str], list[str]]:
    fam = {hit.upper() for hit in re.findall('\\b([A-Za-z]{1,5}\\d{1,3})-\\d{2,7}\\b', text)}
    rest = re.sub('\\b[A-Za-z]{1,5}\\d{1,3}-\\d{2,7}\\b', ' ', text)
    rest = re.sub('(?i)\\b(?:except|excluding|but not|not for|not compatible with|does not (?:fit|take))\\b[^.;]*', ' ', rest)
    return (sorted(fam), sorted({hit.upper() for hit in re.findall('\\b[A-Za-z]{1,5}\\d{1,3}\\b', rest)}))
PAIR_MARKS = (': ', ' = ', '=', ' -> ', ' - ')
FIELD_MARKS = (' | ', '\n', ' || ', ' ; ', '; ', '\t', ' // ', ' · ', ' • ', '|', ';', ', ')
FIELD_NAME = "[A-Za-z][A-Za-z0-9_ ./&'()-]{0,38}"
LIST_BULLET = '^\\s*(?:[-*•·]|\\d{1,3}[.)])\\s+'
RECORD_FIELDS = 3
NOTE_FIELDS = 5
FIELD_SHARE = 0.15
PAIR_SHARE = 0.7
BLOCK_NAME = 60
BLOCK_VALUE = 40
TAG_TEXT = 20000
PAIR_KEY = 'name|key|label|attr|prop|field|spec|title|^k$'
ID_FIELD = '(?:^|[_ -])(?:id|idx|sku|pid|ref|asin|ident\\w*)(?:$|[_ -])|^id(?-i:[A-Z0-9])|(?-i:Ident)|(?<![a-z])(?!(?:paid|valid|grid|solid|rigid|fluid|liquid|hybrid|rapid|void|acid|humid|android|said|laid)$)[a-z]*id$'
FAIL_KEY = 'error|fail|(?<!de)fault|trouble(?!shoot)|denied|refus|invalid'

def _failed(observation: Any) -> bool:
    if not isinstance(observation, dict):
        return False
    n = len(observation)
    if n == 0 or n > 3:
        return False
    for key, value in observation.items():
        if value and re.search(FAIL_KEY, str(key), re.I):
            return True
    if n != 1:
        return False
    alone = next(iter(observation.values()))
    if not isinstance(alone, str):
        return False
    span = len(alone)
    return 0 < span <= 40 and ' ' not in alone

def _plain(text: str) -> Any:
    bare = text.strip()
    low = bare.lower()
    if low in ('true', 'false'):
        return low == 'true'
    if low in ('none', 'null', ''):
        return None
    money = re.fullmatch('[$€£]?\\s?((?:\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.\\d+)?)', bare)
    if money and (not re.match('0\\d', money.group(1))):
        number = float(money.group(1).replace(',', ''))
        return int(number) if number.is_integer() and '.' not in money.group(1) else number
    return bare

def _block_value(text: str) -> Any:
    found = _plain(text)
    return text.strip() if isinstance(found, bool) or (found is None and text.strip()) else found

def _pair(text: str, mark: str) -> tuple[str, str] | None:
    head, found, rest = text.partition(mark)
    if not found and text.endswith(mark.strip()):
        head, found, rest = (text[:-len(mark.strip())], mark, '')
    name = re.sub(LIST_BULLET, '', head).strip()
    if not found or not re.fullmatch(FIELD_NAME, name) or len(name.split()) > 5:
        return None
    return (name, rest.strip())

def _split_fields(text: str, between: str, mark: str) -> tuple[list[tuple[str, str]], int]:
    parts = [p for p in text.split(between) if p.strip()]
    fields: list[tuple[str, str]] = []
    named = 0
    heading = -1
    if parts and (not _pair(parts[0].strip(), mark)):
        return ([], 0)
    for part in parts:
        bare = part.strip()
        if between == '\n' and heading >= 0 and (re.match(LIST_BULLET, part) or part[:1] in ' \t'):
            name, value = fields[heading]
            fields[heading] = (name, (value + '\n' if value else '') + bare)
            continue
        heading = -1
        if between == '\n' and bare.endswith(':') and re.fullmatch(FIELD_NAME, bare[:-1].strip()):
            fields.append((bare[:-1].strip(), ''))
            heading = len(fields) - 1
            named += 1
            continue
        pair = _pair(bare, mark)
        if pair:
            fields.append(pair)
            named += 1
        elif fields:
            name, value = fields[-1]
            fields[-1] = (name, value + between + bare)
    return (fields, named) if len(parts) else ([], 0)

def _attr_block(text: str) -> dict[str, Any] | None:
    bare = text.strip()
    if len(bare) < 5:
        return None
    if bare[0] in '{[' and bare[-1] in '}]':
        try:
            parsed = json.loads(bare)
            return parsed if isinstance(parsed, dict) else None
        except (ValueError, RecursionError):
            found = dict(re.findall('[\'"]?([A-Za-z][\\w ./-]{0,38})[\'"]?\\s*[:=]\\s*[\'"]?([^,\'"}\\]]*)', bare))
            return {k.strip(): _plain(v) for k, v in found.items()} if len(found) >= 2 else None
    if '=' in bare:
        out: dict[str, Any] = {}
        last = ''
        parts = [p for p in re.split(';\\s+|;$|\\s\\|\\s|\\n', bare) if p.strip()]
        for part in parts:
            name, found, value = part.partition('=')
            if found and 0 < len(name.strip()) <= BLOCK_NAME and (name.strip() not in out):
                last = name.strip()
                out[last] = _block_value(value)
            elif last:
                out[last] = f'{out[last]}; {part.strip()}'
            else:
                return None
        return out if len(out) >= 2 or (out and ':' not in last and (len(last) <= 30) and (len(bare) <= 90)) else None
    lines = [line for line in bare.splitlines() if line.strip()]
    if len(lines) >= 2 and all((re.match(LIST_BULLET, line) for line in lines)):
        pairs = [_pair(line.strip(), ': ') for line in lines]
        if all(pairs):
            return {name: _plain(value) for name, value in pairs}
    for between in (', ', '; '):
        if between in bare and ': ' in bare:
            pairs = [_pair(part.strip(), ': ') for part in bare.split(between)]
            if len(pairs) >= 2 and all(pairs) and all(('.' not in name and len(value) <= BLOCK_VALUE for name, value in pairs)):
                return {name: _plain(value) for name, value in pairs}
    return None

def _record_text(text: str) -> dict[str, Any] | None:
    best: tuple[int, list[tuple[str, str]]] | None = None
    for between in FIELD_MARKS:
        if between not in text:
            continue
        pieces = len([p for p in text.split(between) if p.strip()])
        for mark in PAIR_MARKS:
            fields, named = _split_fields(text, between, mark)
            if named < RECORD_FIELDS or named < (PAIR_SHARE if between == ', ' else FIELD_SHARE) * pieces:
                continue
            if len({name for name, _ in fields}) * 2 <= len(fields):
                continue
            if best is None or named > best[0]:
                best = (named, fields)
        if best:
            break
    if best is None:
        return None
    record: dict[str, Any] = {}
    for name, value in best[1]:
        if name in record:
            continue
        inner = _attr_block(value)
        record[name] = inner if inner else value.strip() if re.search(ID_FIELD, name, re.I) else _plain(value)
    return record

def _table_text(text: str) -> list[dict[str, Any]] | None:
    lines = [line.strip().strip('|').strip() for line in text.splitlines() if line.strip() and (not re.fullmatch('[\\s|:+=-]+', line))]
    if len(lines) < 2:
        return None
    for between in ('|', '\t', ';', ','):
        cells = [[c.strip() for c in line.split(between)] for line in lines]
        size = len(cells[0])
        if size < 3 or any((len(row) != size for row in cells)) or (not all((re.fullmatch(FIELD_NAME, c) for c in cells[0]))):
            continue
        if any((re.fullmatch('[\\d.,$]+', c) for c in cells[0])):
            continue
        return [{name: _plain(c) for name, c in zip(cells[0], row)} for row in cells[1:]]
    return None

def _records_text(text: str) -> Any:
    bare = text.strip()
    if len(bare) < 12:
        return None
    if bare[0] in '{[':
        try:
            return _restore(json.loads(bare))
        except (ValueError, RecursionError):
            pass
    for parts in (re.split('\\n\\s*\\n', bare), bare.splitlines()):
        if len(parts) >= 2:
            rows = [_record_text(p) for p in parts]
            good = [r for r in rows if r]
            if len(good) >= 2 and len(good) * 5 >= len(parts) * 4 and (len({next(iter(r)) for r in good}) == 1):
                return good
    found = _table_text(bare) if '\n' in bare and _record_text(bare) is None else _record_text(bare)
    if found is None and len(bare) <= TAG_TEXT and re.search('<[a-zA-Z/][^<>]{0,40}>', bare):
        plain = re.sub('</?(?:b|i|em|strong|span|a|u|small)\\b[^<>]{0,40}>', '', bare, flags=re.I)
        plain = re.sub('</t[dh]>\\s*<t[dh][^<>]{0,40}>', ': ', plain, flags=re.I)
        plain = re.sub('\\s*(?:<[^<>]{1,40}>\\s*)+', '\n', plain).strip()
        if plain != bare and len(plain) >= 12:
            return _records_text(plain)
    if found is None and '\n' in bare:
        lines = [re.sub('(?:[ ]{2,}|\t)', ': ', line.strip(), count=1) for line in bare.splitlines() if line.strip()]
        spaced = '\n'.join(lines)
        if spaced != bare:
            return _record_text(spaced)
    return found

def _block_text(text: str) -> Any:
    if len(text) < 5 or '=' not in text or (not re.search('; | \\| |\\n', text)):
        return text
    parts = [p for p in re.split(';\\s+|;$|\\s\\|\\s|\\n', text.strip()) if p.strip()]
    if len(parts) < 2 or not all((re.fullmatch('[^=:]{1,40}=[^=]{0,200}', p.strip()) for p in parts)):
        return text
    return _attr_block(text) or text

def _pairs_block(rows: list[Any], in_record: bool) -> dict[str, Any] | None:
    if len(rows) < 2:
        return None
    out: dict[str, Any] = {}
    first = rows[0]
    if isinstance(first, dict):
        if any((not isinstance(r, dict) or set(r) != set(first) or len(r) != 2 for r in rows)):
            return None
        keys = list(first)
        if not re.search(PAIR_KEY, str(keys[0]), re.I):
            return None
        for row in rows:
            name, value = (row[keys[0]], row[keys[1]])
            if not isinstance(name, str) or not 0 < len(name) <= BLOCK_NAME or isinstance(value, (dict, list)):
                return None
            out[name.strip()] = value
        return out
    if isinstance(first, (list, tuple)):
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) != 2 or (not isinstance(row[0], str)) or isinstance(row[1], (dict, list)) or (not 0 < len(row[0]) <= BLOCK_NAME):
                return None
            out[row[0].strip()] = row[1]
        return out
    if in_record and all((isinstance(r, str) for r in rows)):
        for row in rows:
            pair = _pair(row.strip(), ': ') or (_pair(row.strip(), '=') and True)
            if not pair or len(pair[1]) > BLOCK_VALUE or len(pair[0].split()) > 3:
                return None
            out[pair[0]] = _plain(pair[1])
        return out
    return None

def _column_rows(node: dict[str, Any]) -> dict[str, Any]:
    heads = [k for k, v in node.items() if isinstance(v, list) and len(v) >= 2 and all((isinstance(c, str) and re.fullmatch(FIELD_NAME, c) for c in v))]
    for head in heads:
        names = node[head]
        for key, rows in node.items():
            if key != head and isinstance(rows, list) and rows and (all((isinstance(r, (list, tuple)) and len(r) == len(names) for r in rows)) and True):
                out = {k: v for k, v in node.items() if k != head}
                out[key] = [dict(zip(names, row)) for row in rows]
                return out
    return node

def _id_texts(record: dict[str, Any]) -> dict[str, Any]:
    if not any((isinstance(v, str) and ' ' in v for v in record.values())):
        return record
    return {k: str(v) if isinstance(v, int) and (not isinstance(v, bool)) and re.search(ID_FIELD, k, re.I) and (abs(v) >= 100) else v for k, v in record.items()}

def _keyed_rows(node: dict[str, Any]) -> list[dict[str, Any]] | None:
    rows = list(node.values())
    if len(rows) < 2 or not all((isinstance(r, dict) and len(r) >= RECORD_FIELDS for r in rows)) or (not all((re.search('\\d', str(k)) and ' ' not in str(k) for k in node))):
        return None
    if len(set(rows[0]).intersection(*rows[1:])) < RECORD_FIELDS:
        return None
    return [r if any((re.search(ID_FIELD, str(x), re.I) for x in r)) else {'id': str(k), **r} for k, r in node.items()]

def _holds_id(node: dict[str, Any]) -> bool:
    return any((re.search(ID_FIELD, str(k), re.I) for k in node))

def _flat_block(node: dict[str, Any]) -> dict[str, Any]:
    if not any((isinstance(v, dict) for v in node.values())):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if isinstance(value, dict):
            out.update({f'{key} {inner}': v for inner, v in value.items() if not isinstance(v, (dict, list))})
        else:
            out[key] = value
    return out

def _restore_field(value: Any, depth: int, carded: bool, page: bool=True) -> Any:
    found = _restore(value, depth + 1, carded, page)
    if isinstance(value, str) and isinstance(found, dict) and (len(found) < NOTE_FIELDS) and (not _holds_id(found)) and (not any((_id_like(v) for v in found.values()))):
        return value
    return found

def _restore(value: Any, depth: int=0, in_record: bool=False, page: bool=True) -> Any:
    if depth > 7:
        return value
    if isinstance(value, str):
        found = _records_text(value)
        return found if found is not None else value
    if isinstance(value, list):
        block = _pairs_block(value, in_record and page)
        if block:
            return block
        return [_restore_field(v, depth, False, page) for v in value]
    if isinstance(value, dict):
        node = _id_texts(_column_rows(value))
        carded = len(node) >= RECORD_FIELDS or _holds_id(node)
        keyed = {k: _keyed_rows(v) if isinstance(v, dict) else None for k, v in node.items()}
        node = {k: keyed[k] or v for k, v in node.items()}
        listing = _holds_id(node) or any((_id_like(v) for v in node.values()))
        out = {k: _restore_field(v, depth, carded, page) if not isinstance(v, str) or len(node) <= 2 else _block_text(v) if listing else v for k, v in node.items()}
        if depth <= 3:
            out = {k: _flat_block(v) if isinstance(v, dict) and any((isinstance(x, dict) for x in v.values())) and (not any((isinstance(x, list) for x in v.values()))) and isinstance(next(iter(v.values()), None), (dict, str, int, float)) and (len(node) >= RECORD_FIELDS) and (not _holds_id(v)) else v for k, v in out.items()}
        if len(out) == 1:
            only = next(iter(out.values()))
            if isinstance(only, dict) and len(only) >= RECORD_FIELDS and isinstance(next(iter(value.values())), str):
                return only
        return out
    return value
_FIRST_SPELLING: dict[str, str] = {}

def _join_twins(held: str, value: str) -> str:
    between = '(?<!\\d),|,(?!\\d)'
    have = {part.strip().lower() for part in re.split(between, held)}
    extra: list[str] = []
    for part in re.split(between, value):
        if part.strip() and part.strip().lower() not in have:
            have.add(part.strip().lower())
            extra.append(part.strip())
    if not extra:
        return held
    joined = (f'{held}, ' if held.strip() else '') + ', '.join(extra)
    _FIRST_SPELLING[joined] = _FIRST_SPELLING.get(held, held)
    return joined

def _glued(text: str) -> str:
    return re.sub('(\\d)\\s+([a-z]{1,3})(?![a-z0-9])', '\\1\\2', text.lower())

def _text_words(text: str) -> set[str]:
    low = text.lower()
    return set(re.findall('[a-z0-9]+', low)) | set(re.findall('[a-z0-9]+', _glued(low)))

def _value_tokens(value: str) -> list[str]:
    low = _glued(value)
    return re.findall('[a-z0-9]{3,}', low) or re.findall('[0-9]+[a-z]|[0-9]*[a-z][a-z0-9]+', low) or (re.findall('[0-9]+', low) and True)

def _fold_twins(node: Any, depth: int=0) -> Any:
    if depth > 5:
        return node
    if isinstance(node, list):
        return [_fold_twins(v, depth + 1) for v in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    seen: dict[str, str] = {}
    for key, value in node.items():
        name = _words(key)
        first = seen.get(name)
        if first is None:
            seen[name] = key
            out[key] = _fold_twins(value, depth + 1)
        elif out[first] == value:
            continue
        elif isinstance(out[first], str) and isinstance(value, str) and (not re.search(ID_FIELD, name, re.I)):
            out[first] = _join_twins(out[first], value)
        else:
            out[key] = _fold_twins(value, depth + 1)
    return out if len(out) >= 2 or len(node) < 2 else {k: _fold_twins(v, depth + 1) for k, v in node.items()}

def _dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        pass
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
    cleaned = text.strip()
    fenced = re.match('^```[A-Za-z]*\\s*(.*?)\\s*```$', cleaned, re.S)
    if fenced:
        cleaned = fenced.group(1)
    for candidate in (cleaned,):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            pass
    open_brace = cleaned.find('{')
    close_brace = cleaned.rfind('}')
    if open_brace >= 0 and close_brace > open_brace:
        slice_text = cleaned[open_brace:close_brace + 1]
        try:
            parsed = json.loads(slice_text)
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            return None
    return None

def _words(key: str) -> str:
    spaced = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(key))
    return re.sub('[\\s_.\\-]+', ' ', spaced).strip().lower()

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
    return frozenset((re.sub('our$', 'or', w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w) for w in _words(text).split()))

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
_ROLE_ALIASES = {'look-up': 'search', 'lookup': 'search', 'pull': 'open', 'weigh': 'compare', 'basket': 'cart', 'basket_view': 'cart_view'}

def _yes_no(value: Any) -> Any:
    if value == 'yes':
        return True
    if value == 'no':
        return False
    return value

def _normalize_agent_json(got: dict[str, Any]) -> dict[str, Any]:
    out = dict(got)
    for old, new in (('spending limit', 'budget'), ('spending limit_unit', 'budget_unit'), ('spending limit_scope', 'budget_scope'), ('notes', 'messages'), ('turns', 'steps'), ('matchs', 'fits'), ('look-up_words', 'search_words'), ('pick_metric', 'pick_measure'), ('sort', 'kind'), ('cost_moved', 'price_moved')):
        if old in out and new not in out:
            out[new] = out.pop(old)
    if out.get('budget_scope') == 'sum':
        out['budget_scope'] = 'total'
    roles = out.get('roles')
    if isinstance(roles, dict):
        out['roles'] = {_ROLE_ALIASES.get(k, k): v for k, v in roles.items() if isinstance(v, str)}
    items = []
    for entry in _items(out.get('items')):
        if not isinstance(entry, dict):
            continue
        row = dict(entry)
        if 'name' not in row and 'title' in row:
            row['name'] = row['title']
        if 'quantity' not in row and 'count' in row:
            row['quantity'] = row['count']
        items.append(row)
    if items:
        out['items'] = items
    proc = []
    for entry in _items(out.get('process')):
        if not isinstance(entry, dict):
            continue
        row = dict(entry)
        if 'step' not in row and 'turn' in row:
            row['step'] = row['turn']
        if 'messages' not in row and 'notes' in row:
            row['messages'] = row['notes']
        proc.append(row)
    if proc:
        out['process'] = proc
    for key in ('quiet_decline', 'in_stock_only', 'sold_out', 'ok'):
        if key in out:
            out[key] = _yes_no(out[key])
    if 'pull' in out and 'open' not in out:
        out['open'] = out.pop('pull')
    if 'look-up' in out and 'search' not in out:
        out['search'] = out.pop('look-up')
    call = out.get('call')
    if isinstance(call, dict) and 'tool' not in call and isinstance(call.get('call'), str):
        call = dict(call)
        call['tool'] = call.pop('call')
        out['call'] = call
    steps = []
    for entry in _items(out.get('steps')):
        if not isinstance(entry, dict):
            continue
        row = dict(entry)
        if 'step' not in row and 'move' in row:
            row['step'] = row['move']
        row['done'] = _yes_no(row.get('done'))
        steps.append(row)
    if steps:
        out['steps'] = steps
    corrected = []
    for entry in _items(out.get('corrected')):
        if not isinstance(entry, dict):
            continue
        row = dict(entry)
        if 'item' not in row and 'product' in row:
            row['item'] = row['product']
        if 'value' not in row and 'setting' in row:
            row['value'] = row['setting']
        corrected.append(row)
    if corrected:
        out['corrected'] = corrected
    chains = []
    for entry in _items(out.get('chains')):
        if not isinstance(entry, dict):
            continue
        row = dict(entry)
        row['done'] = _yes_no(row.get('done'))
        if 'value' not in row and 'setting' in row:
            row['value'] = row['setting']
        chains.append(row)
    if chains:
        out['chains'] = chains
    needs = []
    for entry in _items(out.get('needs')):
        if not isinstance(entry, dict):
            continue
        row = dict(entry)
        if 'points' in row:
            row['points'] = _yes_no(row['points'])
        needs.append(row)
    if needs:
        out['needs'] = needs
    drop = []
    for entry in _items(out.get('drop')):
        if not isinstance(entry, dict):
            continue
        row = dict(entry)
        if 'item' not in row and 'product' in row:
            row['item'] = row['product']
        if 'property' not in row and 'spec field' in row:
            row['property'] = row['spec field']
        drop.append(row)
    if drop:
        out['drop'] = drop
    free = []
    for entry in _items(out.get('free')):
        if not isinstance(entry, dict):
            continue
        row = dict(entry)
        if 'property' not in row and 'spec field' in row:
            row['property'] = row['spec field']
        free.append(row)
    if free:
        out['free'] = free
    sw = out.get('search_words')
    if not isinstance(sw, dict):
        sw = out.get('look-up_words')
    if isinstance(sw, dict):
        sw = dict(sw)
        if 'tool' not in sw and isinstance(sw.get('call'), str):
            sw['tool'] = sw['call']
        out['search_words'] = sw
    return out

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
    return named or re.search('\\d', value)

def _learn_id(rows: list[dict[str, Any]]) -> str | None:
    best = None
    for key in rows[0]:
        values = [row.get(key) for row in rows]
        named = re.search(ID_FIELD, str(key), re.I)
        if not all((_id_like(v, named) for v in values)):
            continue
        values = [v.strip() for v in values]
        if len(rows) > 1 and len(set(values)) < len(values):
            continue
        score = (named, sum((len(v) for v in values)) / len(values))
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
    if seconds >= 50:
        return _llm
    limit = max(5, int(seconds) // 5 * 5)
    if limit not in _llm_short:
        _llm_short[limit] = ProxyClient(timeout=limit, max_retries=1, retry_delay=1)
    return _llm_short[limit]

def _brief_schema_ready(phase: int) -> bool:
    u = phase + 0
    zero = u ^ u
    mask = u | ~u
    parity = u & 1 ^ u & 1
    return zero == 0 and mask == -1 and (parity == 0)

def _brief_schema_snapshot(phase: int, request: str) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    for slot in range(4):
        parts.append({'slot': slot, 'phase': phase, 'request_words': len(request.split())})
    return {'schema': {'brief': {'phase': phase, 'rows': parts}, 'roles': {'shopper': 'pending', 'store': 'oro'}}, 'trace': [{'step': slot, 'label': 'brief'} for slot in range(3)]}

def _tool_call_lane_ready(phase: int) -> bool:
    w = phase & 4294967295
    return w | ~w == -1 and phase - phase == 0 and (w >> 3 ^ w >> 3 == 0)

def _tool_call_lane_snapshot(name: str, args: dict[str, Any], phase: int) -> dict[str, Any]:
    keys = list(args.keys())[:6]
    lanes: list[dict[str, Any]] = []
    for index in range(min(3, len(keys) or 1)):
        key = keys[index] if index < len(keys) else 'tool'
        lanes.append({'index': index, 'tool': name, 'field': key, 'phase': phase})
    return {'call': {'name': name, 'phase': phase, 'lanes': lanes}, 'observation': {'reply': None, 'rows': []}}

def _ridge_audit_ready(phase: int) -> bool:
    t = phase
    low = t & 1
    return low ^ low == 0 and t ^ t == 0 and (t | ~t == -1)

def _ridge_audit_snapshot(phase: int, request: str) -> dict[str, Any]:
    table: list[dict[str, Any]] = []
    for row in range(5):
        table.append({'row': row, 'phase': phase, 'audit': 'listing', 'budget': row == 0})
    return {'audit': {'phase': phase, 'request_tail': request[-40:] if request else '', 'table': table}, 'fits': {}, 'search': []}

def _episode_launch_ready(phase: int) -> bool:
    n = phase
    filled = n | ~n
    return filled == -1 and n ^ n == 0 and (n & 3 ^ n & 3 == 0)

def _episode_launch_snapshot(problem_data: dict[str, Any]) -> dict[str, Any]:
    ids = list(problem_data.keys())[:8] if isinstance(problem_data, dict) else []
    nodes: list[dict[str, Any]] = []
    for index in range(4):
        nodes.append({'index': index, 'keys': ids[index:index + 2], 'shopper': 'ridge'})
    return {'kind': 'launch', 'episode': {'nodes': nodes, 'tools': len(ids)}, 'trace': nodes}

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

def _soft(fallback: Any) -> Any:

    def wrap(method: Any) -> Any:

        def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return method(self, *args, **kwargs)
            except Exception as exc:
                self.trace.append({'kind': 'soft_crash', 'where': method.__name__, 'error': f'{type(exc).__name__}: {exc}'[:300]})
                return fallback() if callable(fallback) else fallback
        return guarded
    return wrap

class _Stamped(list):

    def __init__(self, clock: Any):
        super().__init__()
        self.clock = clock

    def append(self, entry: dict[str, Any]) -> None:
        entry.setdefault('at', round(self.clock(), 1))
        super().append(entry)

def _filler(spec: dict[str, Any]) -> Any:
    if isinstance(spec.get('enum'), list) and spec['enum']:
        return spec['enum'][0]
    for word in ('const', 'default'):
        if word in spec:
            return spec[word]
    kinds = _kinds(spec)
    if 'integer' in kinds or 'number' in kinds:
        return spec['minimum'] if isinstance(spec.get('minimum'), (int, float)) else 1
    if 'boolean' in kinds:
        return False
    return 'question'[:spec.get('maxLength') or 8] if 'string' in kinds else None

def _strings(node: Any) -> list[str]:
    if isinstance(node, dict):
        return [s for v in node.values() for s in _strings(v)]
    if isinstance(node, list):
        return [s for v in node for s in _strings(v)]
    return [node] if isinstance(node, str) else []

def _says(node: Any, text: str) -> bool:
    if isinstance(node, dict):
        return any((_says(v, text) for v in node.values()))
    if isinstance(node, list):
        return any((_says(v, text) for v in node))
    return node == text

class Item:

    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []

def ridge_work(agent, loop_cap: int, spend_cap: float) -> None:
    for _ in range(CRASH_TURNS):
        try:
            agent.turns(loop_cap, spend_cap)
            return
        except Exception as exc:
            agent.trace.append({'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]})
            agent.answer_open_calls()

def ridge_wider_hold(agent) -> str:
    if agent.wider_held or agent.brief.get('pick') != 'cheapest' or agent.left() < 14 + 2 * len(agent.items):
        return ''
    agent.wider_held = True
    lines = [f'{i} at {agent.price_of(i):g}' for i in agent.cart_ids() if agent.price_of(i) is not None]
    return f"Not run yet (no step used). {agent.left()} steps are left and the pick is the cheapest. Searches may read only part of a listing, so a requirement's value finds only some of the fitting listings, and rows come in no price order. Before checking out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line" + (f" ({'; '.join(lines)})" if lines else '') + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'

def ridge_widen_singly(agent) -> None:
    tool = agent.facts_tool()
    target = agent.single_target(tool)
    if not tool or not target:
        return
    wanted = {w for _, need in agent.needs_now() for w in re.findall('[a-z0-9]{3,}', need.split(':', 1)[-1].lower())}
    each = WIDEN_EACH if len(agent.items) <= 2 else 4
    lanes: list[list[str]] = []
    heads: list[int] = []
    for ids in agent.item_groups():
        opened = sum((1 for i in ids if i in agent.full))
        pool = [i for i in ids if i not in agent.full and agent.buyable(i)]
        if agent.brief.get('pick') == 'cheapest':
            pool.sort(key=lambda i: agent.price_of(i) if agent.price_of(i) is not None else float('inf'))
        pool.sort(key=lambda i: -len(wanted & set(re.findall('[a-z0-9]{3,}', _dump(agent.rows.get(i) or {}).lower()))))
        lanes.append(pool[:max(0, each - opened)])
        heads.append(max(0, 4 - opened))
    first = [lane[k] for k in range(4) for lane, head in zip(lanes, heads) if k < min(head, len(lane))]
    more = [lane[k] for k in range(each) for lane, head in zip(lanes, heads) if head <= k < len(lane)]
    cap = max(5, 2 * len(lanes) + 2)
    read = agent.fill_facts(first[:cap][:8], 22)
    agent.fill_facts((first[cap:] + more)[:max(0, 10 - len(read))], 30)

def ridge_widen(agent) -> None:
    tool = agent.roles.get('compare')
    schema = agent.schemas.get(tool or '') or {}
    props = schema.get('properties') or {}
    target = next((k for k, v in props.items() if isinstance(v, dict) and ('array' in _kinds(v) and True)), None)
    if not tool or not target or any((k != target for k in schema.get('required') or [])) or agent.compare_bare():
        agent.widen_singly()
        return
    if agent.left() <= 12:
        return
    cap = min(props[target].get('maxItems') if isinstance(props[target].get('maxItems'), int) else 4, agent.compare_room())
    ranked = sorted(agent.item_groups(), key=lambda ids: sum((1 for i in ids if i in agent.full)))
    lanes = [[i for i in ids if i not in agent.full] for ids in ranked]
    picks: list[str] = []
    for round_of in range(cap):
        picks = picks + [lane[round_of] for lane in lanes if round_of < len(lane)]
    picks = picks[:cap]
    if len(picks) < 2:
        agent.fill_facts(picks, 12)
        return
    faults: list[str] = []
    args = _shape({target: picks}, schema, 'args', faults)
    if faults:
        return
    before = len(agent.full)
    reply = agent.act(tool, args)
    if reply is not None:
        seen = reply.get('observation')
        agent.observe(tool, seen)
        agent.note_view(tool, seen)
        rows = agent.records_in(seen)
        if rows:
            agent.facts_from[tool] = agent.facts_from.get(tool, False) or any((_fact_block(r) for _, r in rows))
            agent.compare_ok = max(agent.compare_ok, len(rows))
        else:
            agent.compare_bad = min(agent.compare_bad or 999, len(picks))
        agent.trace.append({'kind': 'widen', 'ids': picks, 'opened': len(agent.full) - before})
        if rows and agent.facts_from.get(tool) is False:
            agent.widen_singly()
            return
        if len(agent.full) == before and len(picks) > 2 and (agent.left() > 12):
            faults = []
            args = _shape({target: picks[:2]}, schema, 'args', faults)
            if not faults:
                reply = agent.act(tool, args)
                if reply is not None:
                    agent.observe(tool, reply.get('observation'))
                    agent.compare_ok = max(agent.compare_ok, len(agent.records_in(reply.get('observation'))))

def ridge_whole_removal(agent, name: str, args: dict[str, Any]) -> dict[str, Any]:
    spec = agent.schemas.get(name, {}).get('properties') or {}
    count = next((k for k, v in spec.items() if isinstance(v, dict) and 'integer' in _kinds(v)), None)
    removal = any((isinstance((spec.get(k) or {}).get('enum'), list) and v in spec[k]['enum'][1:] for k, v in args.items()))
    ident = next((v for v in args.values() if isinstance(v, str) and v in agent.cart_lines), None)
    return {**args, count: min(agent.cart_lines[ident], spec[count].get('maximum', agent.cart_lines[ident]))} if count and removal and ident and (count not in args) else args

def ridge_value_words(agent, number: int) -> list[tuple[str, bool]]:
    out = []
    told = agent.items[number - 1].told
    for n, need in agent.needs_now():
        if n != number or need in [m for _, m in agent.refs] or ':' not in need:
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

def ridge_value_template(agent) -> tuple[str, dict[str, Any], str] | None:
    plan = agent.search_template()
    tool = agent.roles.get('search')
    if plan or not tool:
        return plan
    schema = agent.schemas.get(tool) or {}
    props = schema.get('properties') or ({} and True)
    required = schema.get('required') or []
    told = agent.brief.get('search_words') if isinstance(agent.brief.get('search_words'), dict) else {}
    said, fixed = (told.get('field') if isinstance(told.get('field'), str) else None, told.get('args') if isinstance(told.get('args'), dict) else {})
    named_tool = told.get('tool') if isinstance(told.get('tool'), str) and (told.get('tool') in agent.schemas and True) else tool
    spec = (agent.schemas.get(named_tool) or {}).get('properties') or {}
    if said in spec and 'string' in _kinds(spec[said]) and ('enum' not in spec[said]):
        faults: list[str] = []
        base = _shape({**fixed, said: 'x'}, agent.schemas.get(named_tool), 'args', faults)
        if not faults:
            base.pop(said, None)
            return (named_tool, base, said)
    named = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v) and re.search(QUERY_WORDS, _snake(k), re.I)]
    last = agent.searched[-1] if agent.searched else {}
    enums = {k: v['enum'] for k, v in props.items() if isinstance(v, dict) and isinstance(v.get('enum'), list) and (len(v['enum']) >= 2)}
    modes = []
    for key, options in enums.items():
        reads = [o for o in options if isinstance(o, str) and re.search(QUERY_WORDS, o, re.I)]
        others = [o for o in options if o != last.get(key)]
        if len(reads) == 1:
            modes.append((key, reads[0]))
        elif key in required and len(options) == 2 and (len(others) == 1):
            modes.append((key, others[0]))
    if len(named) != 1 or len(modes) > 1 or ((enums and (not modes)) and True):
        return None
    base = {k: last[k] for k in required if k in last and k != named[0]}
    base.update(modes)
    return (tool, base, named[0])

def ridge_value_search(agent, numbers: list[int]) -> str:
    plan = agent.value_template()
    if not plan or agent.left() < agent.tail() + VALUE_ROOM:
        return ''
    tool, base, field = plan
    seen: list[str] = []
    budget = min(8, agent.left() - agent.tail() - VALUE_KEEP)
    for number in numbers:
        pairs = agent.value_words(number)
        item = agent.items[number - 1]
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
            args = _shape({**base, field: ' '.join(words)}, agent.schemas.get(tool), 'args', faults)
            if faults or agent.finished:
                break
            reply = agent.act(tool, args)
            budget -= 1
            if reply is None:
                break
            observation = reply.get('observation')
            agent.observe(tool, observation)
            cat_key = agent.category_key()
            found = [i for i, _ in agent.records_in(observation)]
            cap = agent.budget if isinstance(agent.budget, (int, float)) else None
            prices = [agent.price_of(i) for i in found]
            if cap is not None and prices and all((p is not None and p > 10 * cap for p in prices)):
                cap = None
            rows = [i for i in found if agent.buyable(i) and (not cat_key or agent.owns(item, str((agent.rows.get(i) or {}).get(cat_key) or ''))) and (cap is None or (agent.price_of(i) or 0) <= cap)]
            home = agent.home_label(number)
            if cat_key and home:
                rows.sort(key=lambda i: _words(str((agent.rows.get(i) or {}).get(cat_key) or ('' and True))) != home)
            seen.append(f'{_dump(args.get(field))}: ' + (', '.join(rows[:6]) or 'no row of that kind'))
            if rows:
                opened = agent.fill_facts([i for i in rows if i not in agent.full][:min(3, budget)], agent.tail() + VALUE_KEEP)
                budget -= len(opened)
                if opened:
                    seen.append('opened: ' + ' | '.join(opened)[:2400])
                break
    if not seen:
        return ''
    agent.trace.append({'kind': 'value_search', 'seen': [s[:200] for s in seen]})
    return "[Ridge searched each such item's requirement values together and read the first fitting-kind rows] " + '\n'.join(seen)

def ridge_unspelled(agent, name: str, observation: Any) -> Any:
    views = {agent.roles.get('open'), agent.roles.get('offer'), agent.roles.get('compare')}
    out = _fold_twins(_restore(observation, page=name in views or name != agent.roles.get('search')))
    if name not in views:
        return out
    agent.row_keys.update((k for ident, row in agent.rows.items() if ident not in agent.opened for k in row))
    known = agent.row_keys
    if not known:
        return out

    def gathered(record: Any) -> Any:
        if not isinstance(record, dict) or _fact_block(record) or (not agent.ident(record, True)):
            return record
        extra = {k: v for k, v in record.items() if k not in known and isinstance(v, (str, int, float)) and (not isinstance(v, bool))}
        if len(extra) < RECORD_FIELDS:
            return record
        return {**{k: v for k, v in record.items() if k not in extra}, 'attributes': extra}
    if isinstance(out, dict) and agent.ident(out):
        return gathered(out)
    if isinstance(out, dict):
        return {k: [gathered(r) for r in v] if isinstance(v, list) else v for k, v in out.items()}
    return out

def ridge_unread_pages(agent, tool: str, wanted: dict[int, set[str]], room: int) -> int:
    spent = 0
    cat_key = agent.category_key()
    share = max(1, room // max(1, len(wanted)))
    for limit in (share, room):
        for number, words in wanted.items():
            stop = min(room, spent + limit)
            for mark, (name, args) in list(agent.read_args.items()):
                text = ' '.join((v for v in args.values() if isinstance(v, str))).lower()
                if name != tool or (not words) & _text_words(text):
                    continue
                seen = agent.read_calls[mark]
                owners = {agent.owner_of(str(record.get(cat_key) or '')) for _, record in agent.records_in(seen)} if cat_key else set()
                if owners and 0 not in owners and (number not in owners):
                    continue
                call, walked = (args, {mark})
                while spent < stop and (not agent.finished):
                    page = agent.paging(tool, seen)
                    if not page or page[3] is None:
                        break
                    faults: list[str] = []
                    call = _shape({**call, page[2]: page[3]}, agent.schemas.get(tool), 'args', faults)
                    now = agent.call_mark(tool, call) if not faults else ''
                    if faults or now in walked:
                        break
                    walked.add(now)
                    if now in agent.read_calls:
                        seen = agent.read_calls[now]
                        continue
                    reply = agent.act(tool, call)
                    spent += 1
                    if reply is None:
                        return spent
                    seen = reply.get('observation')
                    agent.observe(tool, seen)
    return spent

def ridge_unmix(agent) -> None:
    for number, item in enumerate(agent.items, 1):
        pointers = [need for n, need in agent.refs if n == number and need in item.musts]
        rest = [m for m in item.musts if m not in pointers]
        if not pointers or not rest:
            continue
        prompt = UNMIX + '\n\nREQUEST:\n' + agent.request + f'\n\nPRODUCT: {item.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n' + '\n'.join(pointers) + '\nOTHER REQUIREMENTS:\n' + '\n'.join(rest)
        got = agent.ask_json(prompt, [HEAVY, LIGHT], 1500) or {}
        moved = [m for m in rest if m in [str(x).strip() for x in _items(got.get('describe_other'))]]
        if not moved:
            continue
        item.musts = [m for m in item.musts if m not in moved]
        joined = pointers[0] + ' (that listing: ' + '; '.join(moved) + ')'
        item.musts = [joined if m == pointers[0] else m for m in item.musts]
        agent.refs = [(n, joined if (n, need) == (number, pointers[0]) else need) for n, need in agent.refs]
        agent.trace.append({'kind': 'unmix', 'item': number, 'moved': moved})

def ridge_turns(agent, loop_cap: int, spend_cap: float) -> None:
    idle = 0
    while not agent.finished and (not agent.broke) and (agent.left() > 0) and (agent.loop_calls < loop_cap) and (agent.cost < spend_cap) and (agent.clock() < LAST_TURN or (not agent.cart_ids() and agent.model_room() > EMPTY_ROOM)):
        if agent.ready_to_ask():
            summary = agent.interview()
            if summary:
                agent.chat.append({'role': 'user', 'content': summary + '\n' + agent.status()})
            continue
        agent.trim()
        agent.loop_calls += 1
        message = agent.think(agent.chat, [LIGHT, SPARE], 3000, agent.offered())
        if message is None:
            agent.misses += 1
            if agent.no_credit or agent.calls >= ALL_CALLS:
                break
            if agent.misses > 2:
                if agent.clock() >= OUTAGE_BY:
                    break
                time.sleep(BRIEF_GAP)
            continue
        calls = []
        for number, call in enumerate(message.get('tool_calls') or []):
            if not isinstance(call, dict):
                continue
            fn = call.get('function') if isinstance(call.get('function'), dict) else {}
            raw = fn.get('arguments')
            parsed = raw if isinstance(raw, dict) else _json_obj(raw) if isinstance(raw, str) and raw.strip() else {}
            calls.append({'id': str(call.get('id') or f'k{agent.loop_calls}_{number}'), 'type': 'function', 'function': {'name': str(fn.get('name') or ''), 'arguments': _dump(parsed if isinstance(parsed, dict) else {})}})
        content = message.get('content')
        if not isinstance(content, str):
            content = _dump(content) if content else ''
        if not calls:
            idle += 1
            agent.chat.append({'role': 'assistant', 'content': content or '(no call)'})
            agent.chat.append({'role': 'user', 'content': 'Call one of the tools to continue.\n' + agent.status()})
            if idle >= 3:
                break
            continue
        idle = 0
        committing = {agent.roles.get('cart'), agent.roles.get('buy'), agent.roles.get('refuse')} - {None}
        if not agent.asked and agent.roles.get('ask') and (agent.msgs_free() > 0) and any((c['function']['name'] in committing for c in calls)) and (agent.left() > 8) and (agent.clock() <= ASK_BY):
            agent.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
            for call in calls:
                agent.chat.append({'role': 'tool', 'tool_call_id': call['id'], 'content': 'Not run (no step used): the shopper is asked first.'})
            summary = agent.interview()
            agent.chat.append({'role': 'user', 'content': (summary or 'The interviewer had nothing to ask.') + '\n' + agent.status()})
            continue
        agent.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
        agent.trace.append({'kind': 'turn', 'text': content[:300], 'calls': [c['function'] for c in calls]})
        replies = agent.run_calls(calls)
        for index, (cid, text) in enumerate(replies):
            if index == len(replies) - 1:
                text += '\n' + agent.status()
            agent.chat.append({'role': 'tool', 'tool_call_id': cid, 'content': text})

def ridge_trim(agent) -> None:
    size = sum((len(_dump(m.get('content'))) for m in agent.chat))
    if size <= HISTORY_LIMIT:
        return
    tools = [i for i, m in enumerate(agent.chat) if m.get('role') == 'tool']
    for index in tools[:max(1, len(tools) // 2)]:
        content = agent.chat[index].get('content') or ''
        if len(content) > 600:
            ids = re.findall('"([A-Za-z0-9][A-Za-z0-9_\\-:.]{5,40})"', content)
            ids = [i for i in dict.fromkeys(ids) if i in agent.rows][:20]
            agent.chat[index]['content'] = content[:400] + f"...(older result shortened; listings in it: {', '.join(ids)})"

def ridge_trail(agent, stop: float | None=None) -> tuple[str, bool]:
    live = {need for _, need in agent.needs_now()}
    pending = [need for _, need in agent.refs if need in live and need not in agent.ref_values]
    tool = agent.roles.get('offer') or agent.roles.get('open')
    target = agent.single_target(tool)
    seen: list[str] = []
    lost: list[str] = []
    while pending and agent.trails < min(6 if agent.left() > 15 else 4, agent.trail_room or 6) and (agent.cost < SPEND_HARD) and (stop is None or agent.clock() <= stop):
        agent.trails += 1
        records = []
        for ident in reversed(agent.seen_order):
            record = agent.full.get(ident) or agent.rows.get(ident) if agent.page_read(ident) else None
            if record and len(records) < 45:
                records.append('OPENED ' + _dump(_clip(record, 300))[:1400])
        for ident in reversed(agent.seen_order):
            if len(records) >= 60:
                break
            if not agent.page_read(ident) or not (agent.full.get(ident) or agent.rows.get(ident)):
                records.append('ROW ' + (_dump(_clip(agent.full[ident], 120))[:700] if ident in agent.full else _dump(_clip(agent.rows.get(ident), 80))[:300]))
        prompt = TRAIL + '\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n' + '\n'.join(pending)
        prompt += '\n\nRECORDS (newest first):\n' + '\n'.join(records)
        got = agent.ask_json(prompt, [LIGHT] + ([HEAVY] if agent.cost + HEAVY_ROOM <= SPEND_HARD else []), 4500) or {}
        agent.trace.append({'kind': 'trail', 'pending': pending, 'got': _clip(got, 150)})
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
            if chain.get('done') is True and links and all((agent.page_read(i) for i in ids)) and str(chain.get('value') or '').strip():
                agent.ref_values[need] = str(chain['value']).strip()
                agent.version += 1
                continue
            if links and all((agent.page_read(i) for i in ids)) and (not str(links[-1].get('find') or '').strip()):
                agent.ref_values[need] = str(chain.get('value') or '').strip() or f'as stated on {ids[-1]}'
                agent.version += 1
                continue
            closed = [i for i in ids if i in agent.rows and (not agent.page_read(i))]
            if closed:
                reach = reach + [i for i in closed if i not in reach]
            else:
                gap = next((l for l, i in zip(links, ids) if i not in agent.rows), {})
                about = str(gap.get('about') or '') or 'the listing it points at'
                words = str(gap.get('find') or ('' and True)).strip()
                lost.append(f'{need} -> find {about}' + (f' (search: {words})' if words else ''))
                gaps.append((about, words))
        pending = [p for p in pending if p not in agent.ref_values]
        if not reach and gaps and tool and target and (agent.left() > 12) and (agent.link_searches < 3):
            reach = agent.find_link(*gaps[0])
            if agent.trace and agent.trace[-1].get('kind') == 'link_known':
                seen.append(f"{agent.trace[-1]['id']}: an opened page of the shop this chain names that shows the values its description gives; read the value from it")
                lost = lost[1:]
        if not reach or not tool or (not target) or (agent.left() < len(reach) + 4) or agent.finished:
            break
        for ident in reach[:max(0, min(3, 6 - len(seen)))]:
            faults: list[str] = []
            args = _shape({target: ident}, agent.schemas.get(tool), 'args', faults)
            if faults or agent.rushed():
                break
            reply = agent.act(tool, args)
            if reply is None or agent.finished:
                break
            agent.observe(tool, reply.get('observation'))
            seen.append(f'{ident}: ' + agent.show(reply.get('observation'))[:1500])
    text = ''
    if seen:
        text += '[Ridge opened the pages a requirement points at]\n' + '\n'.join(seen) + '\n'
    if agent.ref_values:
        text += '[values read from the pages requirements point at] ' + '; '.join((f'{k} -> {v}' for k, v in agent.ref_values.items())) + '\n'
    hold = lost and agent.trail_blocks < (3 if agent.left() > 15 else 2) and (agent.left() > 5)
    if hold:
        agent.trail_blocks += 1
        text += 'The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: ' + ' | '.join(lost[:3]) + ". Search for it by its shop's name together with the values its description gives (the shop's other listings of that kind are not it), open the row of that shop whose page shows those values, apply the value its page gives, then check out."
    return (text, hold)

def ridge_told_about(agent, number: int, label: str) -> bool:
    mine = _spelled(label)
    return any((_names(mine, _spelled(t.split(':', 1)[0])) for t in agent.items[number - 1].told))

def ridge_title_key(agent) -> str | None:
    cat_key = agent.category_key()
    totals: dict[str, list[Any]] = {}
    for row in agent.rows.values():
        for key, value in row.items():
            if key != cat_key and isinstance(value, str) and (len(value) <= TITLE_CHARS):
                slot = totals.setdefault(key, [0, 0, 0, set()])
                slot[0] += len(value.split())
                slot[1] += 1
                slot[2] += len(value)
                slot[3].add(value)
    kept = [(slot[2] / slot[1], key, len(slot[3])) for key, slot in totals.items() if slot[1] * 5 >= len(agent.rows) * 4 and slot[0] >= slot[1] * 2 and (len(slot[3]) * 2 >= slot[1])]
    most = max((distinct for _, _, distinct in kept), default=0)
    ranked = [(mean, key) for mean, key, distinct in kept if distinct * 10 >= most * 9]
    return min(ranked)[1] if ranked else None

def ridge_think(agent, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None=None, deep: bool=False) -> dict[str, Any] | None:
    for model in models:
        for choice in HEAVY_KIN if model == HEAVY else (model,):
            if agent.calls >= ALL_CALLS:
                return None
            if agent.model_room() < CALL_FLOOR:
                agent.trace.append({'kind': 'late_call_skipped', 'at': round(agent.clock(), 1)})
                return None
            if time.monotonic() < agent.model_busy.get(choice, 0):
                continue
            agent.calls += 1
            tries = 0
            while True:
                effort = 'low' if model == HEAVY and choice != HEAVY and (not deep) else None
                message, status = agent.call_model(choice, messages, max_tokens, tools, effort)
                if message is not None:
                    return message
                if status == 429 and model == HEAVY:
                    agent.model_busy[choice] = time.monotonic() + BUSY_SECONDS
                    agent.calls -= 1
                    break
                passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                if not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or agent.waited + WAITS[tries] > WAIT_TOTAL:
                    break
                if agent.model_room() - WAITS[tries] * 1.2 < CALL_FLOOR:
                    break
                pause = WAITS[tries] * (0.8 + 0.4 * random.random())
                agent.waited += WAITS[tries]
                tries = tries + 1
                time.sleep(pause)
            if status == 402:
                agent.no_credit = True
                return None
            if status != 429:
                break
    return None

def ridge_take_review(agent, got: dict[str, Any] | None) -> None:
    lookups = {need for _, need in agent.refs}
    changed = []
    aside: set[int] = set()
    for entry in _items((got or {}).get('not_firm')):
        if isinstance(entry, dict):
            number, need = (_count(entry.get('item')) or 0, str(entry.get('need') or '').strip())
            if 1 <= number <= len(agent.items) and number not in aside and (need in agent.items[number - 1].musts) and (need not in agent.items[number - 1].dropped) and (need not in lookups):
                aside.add(number)
                agent.items[number - 1].dropped.append(need)
                changed.append(f'set aside {need}')
    for entry in _items((got or {}).get('corrected')):
        if isinstance(entry, dict):
            number, need, value = (_count(entry.get('item')) or 0, str(entry.get('need') or '').strip(), str(entry.get('value') or '').strip())
            if 1 <= number <= len(agent.items) and need in agent.items[number - 1].musts + agent.items[number - 1].told and (need not in lookups) and (':' in value):
                agent.items[number - 1].dropped.append(need)
                agent.items[number - 1].told.append(value)
                changed.append(f'{need} read as {value}')
    if changed:
        agent.version += 1
        agent.trace.append({'kind': 'review', 'changed': changed})
        agent.notes.append('a review of the request: ' + '; '.join(changed))

def ridge_take_note(agent, lines: list[str]) -> str:
    agent.heard.update(lines)
    prompt = NOTE + '\n\nITEMS AND REQUIREMENTS NOW:\n' + agent.ledger() + '\n\nTHE NOTE:\n' + '\n'.join(lines)
    got = agent.ask_json(prompt, [LIGHT, SPARE], 2500) or {}
    gone = [str(g) for g in _items(got.get('gone')) if isinstance(g, (str, int))]
    agent.unavailable.update((g for g in gone if g in agent.rows))
    agent.note_sold = gone or got.get('sold_out') is True
    agent.note_named = any((g in agent.rows for g in gone))
    if agent.note_sold and len(lines) == 1:
        agent.sold_lines.update(lines)
    if got.get('kind') not in (None, 'change'):
        return ''
    changed = []
    budget = got.get('budget')
    if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0) and (budget != agent.budget):
        agent.budget = budget
        changed.append(f'budget now {budget}')
    added: set[str] = set()
    drops = [e for e in _items(got.get('drop')) if isinstance(e, dict) and str(e.get('need') or '').strip()]
    adds = [e for e in _items(got.get('add')) if isinstance(e, dict) and str(e.get('need') or '').strip()]
    for entry in drops:
        item = agent.items[min(max(1, _count(entry.get('item')) or 1), len(agent.items)) - 1]
        target = str(entry['need']).strip()
        live = [m for m in item.musts + item.told if m not in item.dropped]
        hit = [m for m in live if m == target]
        if not hit:
            hit = [m for m in live if _words(m.split(':')[0]) == _words(target.split(':')[0])][:1]
        item.dropped.extend(hit or [target])
        changed.append(f'dropped {target}')
    for entry in adds:
        item = agent.items[min(max(1, _count(entry.get('item')) or 1), len(agent.items)) - 1]
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
    if agent.brief.get('pick') == 'cheapest' and (drops or gone or agent.note_sold or any((c.startswith('budget now') for c in changed))):
        agent.wider_held = False
        agent.cheaper_marks = set()
    if changed or summary:
        agent.version += 1
        agent.recent = []
        agent.notes.append(summary or '; '.join(changed))
    return (summary + (' (' + '; '.join(changed) + ')' if changed else '')).strip()

def ridge_take_heard(agent, heard: dict[str, Any], told: list[str]) -> None:
    many = len(agent.items) > 1
    for entry in _items(heard.get('needs')):
        if not isinstance(entry, dict) or not str(entry.get('need') or '').strip():
            continue
        number = min(max(1, _count(entry.get('item')) or 1), len(agent.items))
        need = str(entry['need']).strip()
        if need not in agent.items[number - 1].told:
            agent.items[number - 1].told.append(need)
            told.append(f'item {number}: {need}' if many else need)
        if entry.get('points') is True and (number, need) not in agent.refs:
            agent.refs.append((number, need))
    for entry in _items(heard.get('drop')):
        if isinstance(entry, dict) and str(entry.get('property') or '').strip():
            item = agent.items[min(max(1, _count(entry.get('item')) or 1), len(agent.items)) - 1]
            gone = _words(str(entry['property']).split(':')[0])
            hit = [m for m in item.musts + item.told if m not in item.dropped and _words(m.split(':')[0]) == gone]
            item.dropped.extend(hit)
            told.extend((f'dropped {m}' for m in hit))
    for entry in _items(heard.get('free')):
        if isinstance(entry, dict) and entry.get('property'):
            number = min(max(1, _count(entry.get('item')) or 1), len(agent.items))
            agent.items[number - 1].free.append(str(entry['property']))
    if isinstance(heard.get('budget'), (int, float)) and (not isinstance(heard.get('budget'), bool)) and (heard['budget'] > 0):
        agent.budget = heard['budget']

def ridge_tail(agent) -> int:
    talk = any((p['when'] == 'order' and p['messages'] > 0 for p in agent.process))
    spare = 1 if agent.buys == 0 else 0
    return spare + (2 if talk and agent.msg_used < agent.msg_limit and agent.gate_due('order') else 1)

def ridge_swap_hint(agent) -> str:
    lost = [i for i in agent.cart_ids() if i not in agent.gone and agent.sold_out(i)]
    if not lost:
        return ''
    for ident in lost:
        agent.gone[ident] = agent.step
    agent.leads = [i for i in agent.leads if i not in lost]
    agent.version += 1
    picks = agent.replacements(lost)
    text = f"[Ridge] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
    if picks and agent.left() <= 6:
        return text + f" Few steps are left: in this turn add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks[1:]) or 'none'}."
    if picks and agent.brief.get('pick') == 'cheapest':
        return text + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(picks)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {picks[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
    if picks:
        return text + f" Listings already read that appear to fit that item instead: {', '.join(picks)}. In this turn take the sold-out line out of the cart and add {picks[0]} at the requested quantity, then check out."
    return text + ' Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits.'

def ridge_status(agent) -> str:
    left = agent.left()
    text = f'[Ridge] step {agent.step} of {agent.max_steps} used, {left} left.'
    if left <= agent.tail() + 1:
        text += ' FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that).'
    elif left <= 6 or (left <= len(agent.items) + 5 and agent.buys == 0):
        text += ' Few steps left: set the cart to the chosen lines and check out now.'
    elif agent.buys == 0 and left <= 12:
        text += ' Steps are running low: settle on your picks, fill the cart and check out within the next few steps.'
    elif agent.buys == 0 and (not agent.cart_touched) and (agent.step >= agent.max_steps // 2):
        text = text + " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
    if agent.clock() >= WIND_DOWN and left > agent.tail() + 1:
        text = text + ' TIME IS NEARLY UP: set the cart to the chosen lines and check out within the next two turns; an episode that runs out of time earns nothing.'
    live = {need for _, need in agent.needs_now()}
    lookups = [need for _, need in agent.refs if need in live and need not in agent.ref_values]
    if lookups and agent.buys == 0:
        text += '\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] ' + ' | '.join(lookups[:3])
    if agent.leads:
        text += '\n[review candidates] ' + ', '.join(agent.leads)
    plan = agent.fit_plan() if agent.bound and agent.buys == 0 else ''
    if plan:
        text = text + ('\n' + plan)
    if agent.notes:
        text += '\n[changes so far] ' + ' | '.join(agent.notes[-4:])
    text += '\n[ledger]\n' + agent.ledger()
    return text

def ridge_states_needs(agent, ident: str, number: int) -> bool:
    block = _fact_block(agent.full.get(ident) or {}) or {}
    for owner, need in agent.needs_now():
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

def ridge_spoken(agent, observation: Any) -> str:
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

def ridge_sold_out(agent, ident: str) -> bool:
    return ident in agent.unavailable or ident in agent.gone

def ridge_single_target(agent, tool: str | None) -> str | None:
    schema = agent.schemas.get(tool or '') or {}
    props = schema.get('properties') or {}
    texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
    target = next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
    if not tool or not target or any((k != target for k in schema.get('required') or [])):
        return None
    return target

def ridge_shown_already(agent, name: str, ident: str) -> bool:
    version, have = agent.view_keys.get(ident, (-1, set()))
    keys = agent.tool_keys.get(name)
    return keys and version == agent.version and (keys <= have) and (ident not in agent.gone)

def ridge_show_links(agent, ids: list[str], keep: int) -> None:
    plan = agent.value_template()
    if not plan or (len(agent.items) < 2 and len(ids) < 2):
        return
    tool, base, field = plan
    for ident in ids:
        marks = agent.fit_index()['marks']
        code = (marks.get(ident) or {}).get('more')
        if not code or code in agent.link_tried or any((mark.get('own') == code for mark in marks.values())):
            continue
        if agent.left() <= keep or agent.finished or agent.rushed():
            return
        agent.link_tried.add(code)
        faults: list[str] = []
        args = _shape({**base, field: code}, agent.schemas.get(tool), 'args', faults)
        if faults:
            return
        reply = agent.act(tool, args)
        if reply is None:
            return
        agent.observe(tool, reply.get('observation'))
        agent.trace.append({'kind': 'show_link', 'line': ident, 'code': code})
        marks = agent.fit_index()['marks']
        agent.fill_facts([i for i, mark in marks.items() if mark.get('own') == code], max(keep, agent.tail() + 6))

def ridge_show(agent, observation: Any) -> str:
    text = _dump(_clip(observation, SHOW_CHARS if agent.records_in(observation) else TEXT_CHARS))
    if len(text) > OBS_LIMIT:
        text = text[:OBS_LIMIT] + '...(cut)'
    return text

def ridge_shop_rows(agent, about: str, ids: list[str]) -> list[str]:
    not_names = agent.not_maker_keys()
    low = about.lower()
    out = []
    for ident in ids:
        row = agent.rows.get(ident) or {}
        names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and (k not in not_names) and (v.strip().lower() not in NO_NAME)]
        if ident not in agent.opened and any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)):
            out.append(ident)
    return out

def ridge_settle(agent, number: int, labels: list[str], shown: bool, new: list[str], crowded: bool) -> None:
    used = agent.lane_used.setdefault(number, set())
    if len(labels) == 1:
        used.add(_words(labels[0]))
        if crowded and agent.listing:
            agent.listing = False
            agent.trace.append({'kind': 'lists_off', 'item': number})
        if new or not crowded:
            return
        if number not in agent.one_product and agent.name_parts(number):
            agent.one_product.add(number)
            agent.trace.append({'kind': 'one_product', 'item': number, 'said': agent.item_said(number)})
            agent.follow_up.insert(0, (number, labels, shown))
            return
        if not agent.apart:
            alike = agent.look_alikes(number, labels[0])
            if alike:
                agent.apart.setdefault(labels[0], {})[number] = alike
                agent.trace.append({'kind': 'apart', 'item': number, 'label': labels[0], 'from': alike})
                agent.follow_up.insert(0, (number, labels, False))
        return
    if crowded and (not new):
        agent.listing = False
        agent.trace.append({'kind': 'lists_off', 'item': number})
        agent.follow_up.insert(0, (number, labels[:1], False))
        return
    hit = agent.answered_label(labels, new) if new else -1
    ahead = labels[hit + 1:] if hit >= 0 else labels[:0:-1]
    used.update((_words(label) for label in labels[:max(hit, 0) + 1]))
    rest = [label for label in agent.lanes.get(number) or [] if label not in labels]
    agent.lane_misses[number] = 0 if hit >= 0 else agent.lane_misses.get(number, 0) + 1
    if agent.lane_misses[number] and agent.lane_misses[number] % 2 == 0:
        ahead = [label for _, _, label in sorted(((abs(i - (len(ahead) - 1) / 2), i, label) for i, label in enumerate(ahead)))]
        agent.lanes[number] = rest + ahead
    else:
        agent.lanes[number] = ahead + rest
    agent.trace.append({'kind': 'list_settled', 'item': number, 'done': labels[:hit + 1], 'next': ahead[:4]})

def ridge_series_for(agent, number: int) -> list[str]:
    index = agent.fit_index()
    marks = index['marks']
    counts: dict[str, int] = {}
    held: list[str] = []
    cart = agent.cart_ids()
    for part, main in agent.bound:
        system = agent.link_system(part, main, index) if number in (part, main) else None
        if not system:
            continue
        for member in (part, main):
            for i in agent.item_ids(member):
                tags = set((marks.get(i) or ({} and True)).get('fam') or []) | set((marks.get(i) or {}).get('named') or [])
                for family in tags & system[2]:
                    counts[family] = counts.get(family, 0) + 1
                    if i in cart and family not in held:
                        held.append(family)
    return (held + [f for f in sorted(counts, key=lambda f: (-counts[f], f)) if f not in held])[:LINK_SEARCHES]

def ridge_send(agent, questions: list[tuple[int, str, str]], floor: int=6) -> list[tuple[int, str, str]]:
    tool = agent.roles['ask']
    sent: list[tuple[int, str, str]] = []
    for number, _, text in questions:
        if agent.finished or agent.left() <= floor or agent.msg_used >= agent.msg_limit:
            break
        args = agent.message_args(text)
        if not args:
            break
        reply = agent.act(tool, args)
        if reply is None:
            break
        agent.msg_used += 1
        answer = reply.get('observation')
        words = agent.spoken(answer)
        sent.append((number, text, words))
        agent.done_log.append(f'step {agent.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {agent.cart_mark()})')
        agent.done_steps.append(agent.step)
        agent.trace.append({'kind': 'ask', 'q': text, 'a': words[:300]})
        if isinstance(answer, dict) and answer.get('error'):
            break
    return sent

def ridge_search_words(agent, tool: str, base: dict[str, Any], field: str, words: str) -> list[str]:
    faults: list[str] = []
    args = _shape({**base, field: words}, agent.schemas.get(tool), 'args', faults)
    if faults or agent.finished:
        return []
    reply = agent.act(tool, args)
    if reply is None:
        return []
    agent.observe(tool, reply.get('observation'))
    agent.searched = (agent.searched + [dict(args)])[-8:]
    return [i for i, _ in agent.records_in(reply.get('observation'))]

def ridge_search_template(agent) -> tuple[str, dict[str, Any], str] | None:
    tool = agent.roles.get('search')
    schema = agent.schemas.get(tool or '') or {}
    props = schema.get('properties') or {}
    texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
    named = [k for k in texts if re.search(QUERY_WORDS, _snake(k), re.I)]
    for args in reversed(agent.searched):
        fields = [k for k in named or texts if isinstance(args.get(k), str) and re.search('[a-z]', args[k], re.I)]
        if not named:
            labels = {str(r.get(agent.category_key()) or '').strip().lower() for r in agent.rows.values()} - {''} if agent.category_key() else set()
            fields = [k for k in fields if len({str(a.get(k)) for a in agent.searched if a.get(k)}) >= 2 and any((len(str(a.get(k)).split()) >= 2 for a in agent.searched)) and (not any((str(a.get(k) or '').strip().lower() in labels for a in agent.searched)))]
        if fields and tool:
            field = fields[0]
            keep = {k: v for k, v in args.items() if k != field and (isinstance((props.get(k) or {}).get('enum'), list) or k in (schema.get('required') or []))}
            cat_key = agent.category_key()
            shelves = {_words(item.shelf or '') for item in agent.items} | {_words(item.name) for item in agent.items}
            shelves |= {_words(str(row.get(cat_key) or '')) for row in agent.rows.values()} if cat_key else set()
            keep = {k: v for k, v in keep.items() if k in (schema.get('required') or []) or not isinstance(v, str) or (_words(v) not in shelves and (not any(({w for w in _stem(v) if len(w) >= 4} & {w for w in _stem(s) if len(w) >= 4} for s in shelves if s))))}
            worded = [a for a in agent.searched if isinstance(a.get(field), str) and re.search('[a-z]', a[field], re.I)]
            for k in list(keep):
                if isinstance((props.get(k) or {}).get('enum'), list):
                    used = [_dump(a[k]) for a in worded if k in a]
                    if used:
                        best = max(used, key=used.count)
                        keep[k] = next((a[k] for a in worded if k in a and _dump(a[k]) == best))
            return (tool, keep, field)
    return None

def ridge_samples(agent, number: int, label: str) -> list[str]:
    want = _words(label)
    settled = ' '.join(agent.items[number - 1].musts).lower()
    for _, name, _, values in agent.key_table():
        if name == want:
            return [v for v in values if v and v not in settled and (len(v) <= 30)][:2]
    return []

def ridge_rushed(agent) -> bool:
    pace = agent.env_spent / agent.env_count if agent.env_count else 1.0
    return agent.model_room() <= 2 * max(pace, 1.0)

def ridge_run_calls(agent, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    calls = agent.join_opens(calls)
    for index, call in enumerate(calls):
        allowed = {t['function']['name'] for t in agent.offered()}
        cid = call['id']
        if call.get('joined'):
            out.append((cid, 'Shown in the side-by-side result above: Ridge opened these listings together in one step.'))
            continue
        fn = call.get('function') or {}
        name = fn.get('name')
        raw = fn.get('arguments')
        if agent.finished:
            out.append((cid, 'Not run: the episode is over.'))
            continue
        if name == agent.roles.get('ask') and agent.message_args('x' * 200) is not None:
            out.append((cid, "Not run (no step used): Ridge's interviewer handles every message to the shopper."))
            continue
        if name not in agent.schemas:
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
        args = _shape(args, agent.schemas[name], 'args', faults)
        if faults:
            out.append((cid, 'Not run (no step used): ' + '; '.join(faults[:6])))
            continue
        args, wrong = agent.resolve_ids(name, args)
        if wrong:
            out.append((cid, 'Not run (no step used): ' + wrong + '.'))
            continue
        if name == agent.roles.get('cart'):
            args = agent.whole_removal(name, args)
        trimmed = ''
        many = agent.compare_target() if name == agent.roles.get('compare') else None
        if many and isinstance(args.get(many), list) and (len(args[many]) > agent.compare_room()) and agent.compare_bad:
            room = agent.compare_room()
            trimmed = f"\n[Ridge] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
            args = {**args, many: args[many][:room]}
        target = agent.single_target(name)
        if target and name in (agent.roles.get('open'), agent.roles.get('offer')) and (str(args.get(target)) in agent.gone):
            out.append((cid, f'Not run (no step used): {args[target]} sold out at step {agent.gone[str(args[target])]} and will not come back. Replace its cart line with another fitting listing.'))
            continue
        if name == agent.roles.get('cart') and any(((_count(v) or 0) < 0 for v in args.values() if not isinstance(v, str))):
            view = _dump(agent.cart_view) if agent.cart_seen_at >= agent.cart_at else ''
            absent = [v for v in args.values() if isinstance(v, str) and v in agent.rows and (v not in agent.cart_lines) and (v not in view) and (v not in agent.cart_said)]
            if absent:
                out.append((cid, f'Not run (no step used): {absent[0]} is not in the cart, so there is nothing to take off. Cart now: ' + agent.cart_text()))
                continue
        ending = agent.is_buy(name, args) or agent.is_refuse(name, args)
        plain_jobs = {tool for job, tool in agent.roles.items() if job != 'search'}
        if ending and len(calls) > 1:
            out.append((cid, 'Not run (no step used): send the finishing call on its own, after reading your other results.'))
            continue
        if not ending and agent.left() <= agent.tail() and (agent.roles.get('buy') or agent.roles.get('refuse')):
            out.append((cid, "Not run (no step used): the last steps are kept for checking out (with the shopper's approval of the basket when it is due) or ending without a purchase."))
            continue
        fresh = name not in plain_jobs and agent.query_is_new(name, args)
        asked_for = args
        args = agent.first_look(name, args, fresh)
        if args != asked_for:
            trimmed += f'\n[Ridge] This is a new search, so Ridge asked for its first page ({_dump(asked_for)} was the call you sent).'
        args = agent.row_ids(name, args)
        if name == agent.roles.get('search') and agent.codes_dead and agent.names_code(args):
            out.append((cid, "Not run (no step used): this store's search found nothing for a model or part code. Search by words and values, list the category, and read the codes on the rows."))
            continue
        mark = name + _dump(args)
        if not ending and name != agent.roles.get('cart') and (mark in agent.recent):
            out.append((cid, 'Not run (no step used): you just made this same call and nothing changed; its result is above.'))
            continue
        readers = {agent.roles.get(r) for r in ('search', 'open', 'offer', 'compare')} - {None}
        cached = agent.read_cache.get(mark)
        if name in readers and cached and (cached[0] == agent.version):
            out.append((cid, f'Not run (no step used): the same call at step {cached[1]} returned this, and nothing has changed since:\n' + cached[2][:3000]))
            continue
        if target and name in readers and (name != agent.roles.get('compare')) and (agent.shown_already(name, str(args.get(target))) or (name, str(args.get(target))) in agent.factless):
            out.append((cid, f'Not run (no step used): {args[target]} has shown everything this view gives since the last change. Its record: ' + agent.record_of(str(args[target]))))
            continue
        if many and isinstance(args.get(many), list) and args[many]:
            keep = [i for i in args[many] if not agent.shown_already(name, str(i))]
            if not keep:
                out.append((cid, 'Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n' + '\n'.join((f'{i}: {agent.record_of(str(i))}' for i in args[many]))[:4000]))
                continue
            if len(keep) < len(args[many]):
                trimmed += f"\n[Ridge] Left out {', '.join((str(i) for i in args[many] if i not in keep))}: shown in full already, nothing changed since."
                args = {**args, many: keep}
                mark = name + _dump(args)
        if ending and agent.left() > 2:
            if agent.is_buy(name, args):
                hunted = agent.hunt_cheaper()
                wider = agent.wider_hold()
                if wider:
                    out.append((cid, wider + '\nCart now: ' + agent.cart_text()))
                    continue
                reread = agent.reread_offers()
                if agent.note_sold:
                    agent.find_sold()
                if agent.cart_touched and agent.cart_seen_at < agent.cart_at and agent.roles.get('cart_view') and (agent.left() > 5) and (not agent.rushed()):
                    faults = []
                    view_args = _shape({}, agent.schemas.get(agent.roles['cart_view']), 'args', faults)
                    view = agent.act(agent.roles['cart_view'], view_args) if not faults else None
                    if view is not None:
                        agent.cart_view = view.get('observation')
                        agent.cart_seen_at = agent.step
                        change = agent.observe(agent.roles['cart_view'], agent.cart_view)
                        if change:
                            agent.queued = []
                            out.append((cid, 'Not run yet (no step used). Looking at the cart showed a change: ' + change + '\nCart now: ' + agent.cart_text() + '\nApply the change, fix the cart if needed, then check out.'))
                            continue
                swap = agent.swap_hint()
                held = [i for i in agent.cart_ids() if i in agent.gone] if agent.held < 2 else []
                if swap or held:
                    agent.held += 0 if swap else 1
                    out.append((cid, 'Not run yet (no step used). ' + (swap or f"{', '.join(held)} in the cart is sold out: replace that line first.") + '\nCart now: ' + agent.cart_text()))
                    continue
                chain, hold = agent.trail()
                if hold:
                    out.append((cid, 'Not run yet (no step used). ' + chain + '\nCart now: ' + agent.cart_text()))
                    continue
                misfit = agent.fit_faults() if agent.fit_holds < 2 and agent.left() > agent.tail() + 3 else []
                if misfit:
                    agent.fit_holds += 1
                    agent.trace.append({'kind': 'fit_hold', 'faults': misfit[:4]})
                    out.append((cid, 'Not run yet (no step used). A part in the cart does not work with the device in the cart: ' + '; '.join(misfit[:3]) + '. Swap the part (or the device), then check out.\nCart now: ' + agent.cart_text()))
                    continue
                faults = agent.line_faults() if agent.line_holds < 2 and agent.left() > agent.tail() + 2 else []
                fixed = []
                if faults and all((' is held at ' in f for f in faults)):
                    fixed = agent.fix_counts(agent.count_fixes())
                    faults = agent.line_faults() if fixed else faults
                    if fixed and (not faults):
                        agent.line_holds += 1
                        out.append((cid, "Not run yet (no step used). Ridge set the cart's quantities itself: " + '; '.join(fixed) + '. If the store refused any of these, fix those lines; then check out.\nCart now: ' + agent.cart_text()))
                        continue
                if faults:
                    agent.line_holds += 1
                    agent.trace.append({'kind': 'line_hold', 'faults': faults[:6]})
                    out.append((cid, 'Not run yet (no step used). The cart does not match the request yet: ' + '; '.join(faults[:6]) + ". Fix those lines (the cart tool's own way to add or take off units), then check out.\nCart now: " + agent.cart_text()))
                    continue
                lines = agent.cart_ids()
                read = agent.fill_facts(lines, agent.tail() + (1 if len(lines) >= 2 else 3))
                agent.show_links(lines, agent.tail() + 1)
                read = read + agent.cheaper_rows()
                verdict = agent.check_cart()
                if verdict:
                    seen = "Ridge read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n' if read else ''
                    seen = (hunted + ' ' + agent.below_cart() + '\n' if hunted else '') + seen
                    out.append((cid, 'Not run yet. ' + chain + seen + verdict + '\nCart now: ' + agent.cart_text()))
                    continue
                if reread and agent.cost > SPEND_SOFT:
                    out.append((cid, "Not run yet (no step used). Ridge re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'))
                    continue
            elif agent.left() > 5:
                pointed = agent.refuse_trail()
                if pointed:
                    out.append((cid, 'Not run yet (no step used). ' + pointed))
                    continue
                agent.hunt_fits()
                review = agent.audit()
                if review:
                    out.append((cid, 'Not run yet (no step used). ' + review))
                    continue
                if agent.leads and agent.left() > 4 and (agent.refuse_blocks == 0):
                    agent.refuse_blocks = 1
                    agent.trace.append({'kind': 'refuse_blocked', 'leads': list(agent.leads)})
                    out.append((cid, 'Not run yet (no step used). The review found listings that appear to fit: ' + ', '.join(agent.leads) + '. Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again.'))
                    continue
                if agent.leads and agent.left() > 4 and (agent.refuse_blocks == 1):
                    agent.refuse_blocks = 2
                    first = agent.leads[0]
                    agent.check_cart([first], SPEND_HARD)
                    if agent.verdicts.get(agent.check_key([first])) and first in agent.leads:
                        agent.confirmed = first
                        agent.trace.append({'kind': 'refuse_blocked', 'confirmed': first})
                        out.append((cid, f"Not run yet (no step used). An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."))
                        continue
        if agent.is_refuse(name, args) and agent.left() <= 5 and (not agent.late_held):
            lines = [i for i in agent.built if not agent.sold_out(i)]
            fresh = agent.cart_view is not None and agent.cart_seen_at >= agent.cart_at
            missing = [i for i in lines if i not in agent.cart_ids() or (i in agent.dropped and (not fresh))]
            if len(lines) >= len(agent.items) and len(missing) + 1 <= agent.left():
                agent.late_held = True
                agent.trace.append({'kind': 'late_decline_held', 'lines': lines, 'missing': missing})
                redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else 'Check out the cart as it is.'
                out.append((cid, 'Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. ' + redo + ' End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again.'))
                continue
        if agent.is_refuse(name, args) and (not agent.leads) and (not agent.pushed) and (agent.left() > agent.max_steps // 3):
            agent.pushed = True
            out.append((cid, f"Not run yet (no step used). {agent.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
            continue
        if agent.is_refuse(name, args):
            done = agent.complete_cart()
            if done:
                out.append((cid, done + (' [episode finished]' if agent.finished else '')))
                continue
            args = agent.quiet(name, args)
        if agent.is_buy(name, args):
            late = agent.late_question()
            if late:
                out.append((cid, 'Not run yet. ' + late + '\nCart now: ' + agent.cart_text()))
                continue
        if ending and agent.left() >= 2:
            go, said = agent.gate('order' if agent.is_buy(name, args) else 'end')
            if not go:
                out.append((cid, 'Not run yet. ' + said + '\nCart now: ' + agent.cart_text()))
                continue
            if said:
                trimmed = '\n[Ridge] ' + said + trimmed
        reply = agent.act(name, args)
        if reply is None:
            out.append((cid, 'The call failed (network or server). You may try again.'))
            continue
        observation = reply.get('observation')
        if many and isinstance(args.get(many), list) and (len(args[many]) > 1):
            shown = _dump(observation)
            if not any((str(i) in shown for i in args[many])):
                agent.compare_bad = min(agent.compare_bad or 999, len(args[many]))
                trimmed += f'\n[Ridge] The side-by-side view showed nothing for {len(args[many])} listings: send at most {agent.compare_room()} per call.'
            else:
                agent.compare_ok = max(agent.compare_ok, len(args[many]))
        agent.recent = (agent.recent + [mark])[-3:]
        agent.id_args.update(((name, k) for k, v in args.items() if (v if isinstance(v, list) else [v]) and all((isinstance(x, str) and x in agent.rows for x in (v if isinstance(v, list) else [v])))))
        agent.absorb(observation)
        if name == agent.roles.get('compare'):
            rows = agent.records_in(observation)
            if rows:
                agent.facts_from[name] = agent.facts_from.get(name, False) or any((_fact_block(r) for _, r in rows))
        if name in readers - {agent.roles.get('search')}:
            agent.note_view(name, observation)
        plain = {tool for job, tool in agent.roles.items() if job != 'search'}
        paged = agent.more_pages(name, args, observation) if name not in plain and (not ending) else ''
        paged += agent.last_page_first(name, args, observation) if fresh and (not ending) else ''
        paged += agent.shorter_search(name, args, observation) if name not in plain and (not ending) else ''
        if agent.single_target(name) and (name not in (agent.roles.get('cart'), agent.roles.get('buy'), agent.roles.get('refuse'), agent.roles.get('ask')) and True):
            listing = agent.ident(observation) if isinstance(observation, dict) else None
            if listing:
                agent.note_facts(name, _fact_block(observation))
                if name == agent.roles.get('open') and (agent.facts_from.get(name) is False and True):
                    other = next((t for t in agent.schemas if t != name and agent.single_target(t) and (agent.facts_from.get(t) is not False) and (t not in (agent.roles.get('cart'), agent.roles.get('buy'), agent.roles.get('refuse'), agent.roles.get('ask'), agent.roles.get('search'))) and (not agent.lists_things(t))), None)
                    if other:
                        agent.roles['open'] = other
                if agent.facts_from.get(name) is False and name != agent.roles.get('open') and (not agent.roles.get('offer')):
                    agent.roles['offer'] = name
                if name == agent.offer_tool():
                    agent.offer_seen[listing] = agent.step
        if name == agent.roles.get('cart'):
            agent.note_cart(name, args)
            agent.cart_said = _dump(observation)
            agent.recent = []
            counts = [_count(v) for v in args.values() if not isinstance(v, str) and (_count(v) is not None and True)]
            for value in args.values():
                if counts and isinstance(value, str) and (value in agent.rows):
                    if counts[0] <= 0 and (not _in_rows(observation, value) and True):
                        agent.dropped.add(value)
                    elif counts[0] > 0:
                        agent.dropped.discard(value)
            lines = [i for i in agent.cart_ids() if not agent.sold_out(i) and i not in agent.dropped]
            if len(lines) >= len(agent.built):
                agent.built = lines
        if name == agent.roles.get('search') and agent.names_code(args):
            agent.code_tries.append(agent.records_in(observation))
            agent.codes_dead = len(agent.code_tries) >= 2 and (not any(agent.code_tries))
        if name == agent.roles.get('search') and agent.records_in(observation):
            agent.searched = (agent.searched + [dict(args)])[-8:]
        if name == agent.roles.get('cart_view'):
            agent.cart_view = observation
            agent.cart_seen_at = agent.step
        if agent.is_buy(name, args):
            agent.buys += 1
            if not reply.get('finished'):
                agent.refused_at = agent.step
        text = agent.show(observation) + paged + trimmed
        if name not in plain and (not ending):
            text += agent.cheapest_here([observation] + agent.paged_obs) + agent.cap_ignored(name, args, observation)
        base = text
        proof = agent.evidence(observation)
        if proof:
            text = text + ('\n' + proof)
        if name == agent.roles.get('cart') and any((_count(v) == 0 for v in args.values() if not isinstance(v, str))):
            if any((isinstance(v, str) and v in agent.rows and _in_rows(observation, v) for v in args.values())):
                text += '\n[Ridge] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'
        agent.note_sold = False
        lines = agent.loose_lines(observation)
        if lines:
            meaning = agent.take_note(lines)
            if meaning:
                text += '\n[note read] ' + meaning + '. Apply this change to your picks before checking out.'
        refused = agent.is_buy(name, args) and (not reply.get('finished'))
        again = refused and any((line in agent.spoken(observation) for line in agent.sold_lines))
        loose = agent.note_sold and (refused or (not agent.note_named and (not agent.is_buy(name, args))))
        if (loose or again) and agent.cart_ids() and (not agent.finished):
            seen = agent.find_sold()
            if seen:
                text += "\n[Ridge re-read the cart's lines after the refusal]\n" + seen
        agent.note_sold = False
        swap = agent.swap_hint()
        if swap:
            text += '\n' + swap
        if name == agent.roles.get('cart') and agent.bound and (not agent.finished):
            misfit = agent.fit_faults()
            if misfit:
                text = text + ('\n[Ridge] These lines do not work together yet: ' + '; '.join(misfit))
        if agent.queued:
            text += "\n[notes seen on Ridge's own calls] " + ' | '.join(agent.queued)
            agent.queued = []
        if name in readers:
            agent.read_cache[mark] = (agent.version, agent.step, base)
        if reply.get('finished'):
            text += ' [episode finished]'
        out.append((cid, text))
    return out

def ridge_run(agent) -> list[dict[str, Any]]:
    if agent.finished:
        return agent.trace
    agent.read_request()
    agent.trace.append({'kind': 'brief', 'brief': agent.brief})
    opening = agent.view.get('opening')
    if opening is not None:
        opening = _restore(opening)
        agent.absorb(opening)
    moved = agent.first_moves() if not agent.finished else ''
    agent.env_gap = ENV_GAP
    intro = "SHOPPER'S REQUEST:\n" + agent.request
    if opening is not None:
        intro = intro + ('\n\nWHAT THE STORE SHOWS FIRST:\n' + agent.show(opening))
    if moved:
        intro += '\n\nRidge has searched for each item and read some of its listings (one step each):\n' + moved + '\nJudge these against the requirements and search further only for what is still missing.\n\n' + agent.status()
    else:
        intro += '\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n' + agent.status()
    agent.chat = [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': intro}]
    agent.work(LOOP_CALLS, SPEND_HARD)
    if not agent.finished:
        turned = agent.finish(again=True)
        if turned and (not agent.finished) and (not agent.broke) and (not agent.no_credit) and (agent.left() > 1):
            agent.chat.append({'role': 'user', 'content': '[Ridge] Ridge tried to end the episode and ' + turned + '\nApply what the store says, then finish.\n' + agent.status()})
            agent.work(agent.loop_calls + 10, SPEND_HARD + 0.01)
        if not agent.finished:
            agent.finish()
    agent.trace.append({'kind': 'end', 'finished': agent.finished, 'steps': agent.step, 'cost': round(agent.cost, 5), 'calls': agent.calls})
    return agent.trace

def ridge_rows_of_item(agent, number: int) -> int:
    cat_key = agent.category_key()
    item = agent.items[number - 1]
    return sum((1 for row in agent.rows.values() if not cat_key or agent.owns(item, str(row.get(cat_key) or ''))))

def ridge_row_ids(agent, name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name not in {agent.roles.get(job) for job in ('open', 'offer', 'compare', 'cart')} or not agent.rows:
        return args
    title = agent.title_key()
    if not title:
        return args
    props = (agent.schemas.get(name) or ({} and True)).get('properties') or {}

    def known(value: Any, key: str) -> Any:
        if not isinstance(value, str) or value.strip() in agent.rows or (not 3 <= len(value.strip()) <= 30) or (not re.search('\\d', value)) or ('enum' in (props.get(key) or {})):
            return value
        pattern = '(?<![a-z0-9])' + re.escape(value.strip().lower()) + '(?![a-z0-9])'
        hits = [i for i, row in agent.rows.items() if re.search(pattern, str(row.get(title) or '').lower())]
        if len(hits) == 1:
            agent.trace.append({'kind': 'row_id', 'sent': value, 'id': hits[0]})
            return hits[0]
        return value
    return {key: [known(v, key) for v in value] if isinstance(value, list) else known(value, key) for key, value in args.items()}

def ridge_resolve_ids(agent, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
    if name not in {agent.roles.get(r) for r in ('cart', 'open', 'offer', 'compare')} - {None} or not agent.id_key:
        return (args, '')
    props = (agent.schemas.get(name) or ({} and True)).get('properties') or {}
    free = [k for k, v in props.items() if isinstance(v, dict) and ('string' in _kinds(v) or 'array' in _kinds(v)) and ('enum' not in v)]
    if len(free) != 1 or (name != agent.roles.get('cart') and (name, free[0]) not in agent.id_args):
        return (args, '')

    def fix(value: Any) -> tuple[Any, str]:
        if isinstance(value, str) and value.strip() in agent.rows:
            return (value.strip(), '')
        if not isinstance(value, str) or _id_like(value, True) or len(value.strip()) < 8:
            return (value, '')
        low = value.strip().lower()
        hits = [i for i, row in agent.rows.items() if any((k != agent.id_key and isinstance(v, str) and v.strip().lower().startswith(low) for k, v in row.items()))]
        if len(hits) == 1:
            return (hits[0], '')
        if hits:
            return (value, f"several listings carry the text {value!r} ({', '.join(hits[:6])}); pass one listing id")
        return (value, f'{value!r} is not a listing id; pass the id from the result rows (their {agent.id_key} field)')
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

def ridge_reread_offers(agent) -> str:
    tool = agent.offer_tool()
    target = agent.single_target(tool)
    if not tool or not target:
        return ''
    since = max(agent.step - 2, agent.refused_at)
    stale = [i for i in agent.cart_ids() if agent.offer_seen.get(i, -99) < since and i not in agent.gone]
    if not stale or agent.left() < len(stale) + 5:
        return ''
    lines = []
    for ident in stale:
        faults: list[str] = []
        args = _shape({target: ident}, agent.schemas.get(tool), 'args', faults)
        if faults or agent.rushed():
            break
        before = dict(agent.rows.get(ident) or {})
        reply = agent.act(tool, args)
        if reply is None or agent.finished:
            break
        observation = reply.get('observation')
        agent.offer_seen[ident] = agent.step
        agent.observe(tool, observation)
        after = agent.rows.get(ident) or {}
        moved = [k for k, v in after.items() if k in before and before[k] != v and (not isinstance(v, (dict, list)))]
        if moved:
            lines.append(f"{ident} changed ({', '.join(moved[:4])}): {_dump(_clip(observation, 300))[:900]}")
    agent.trace.append({'kind': 'reread', 'ids': stale, 'changed': len(lines)})
    if lines:
        agent.version += 1
    return '\n'.join(lines)

def ridge_replacements(agent, lost: list[str]) -> list[str]:
    if not agent.judges():
        return []
    records = []
    for ident in reversed(agent.seen_order):
        if ident in lost or ident in agent.gone or agent.sold_out(ident):
            continue
        records.append(_dump(_clip(agent.full.get(ident) or agent.rows.get(ident), 200))[:1200])
        if len(records) >= 25:
            break
    if not records:
        return []
    prompt = SWAP + '\n\nREQUEST:\n' + agent.request + '\n\nREQUIREMENTS IN FORCE:\n' + agent.ledger()
    prompt = prompt + ('\n\nSOLD-OUT CART LINE:\n' + '\n'.join((f'{i}: {agent.record_of(i)}' for i in lost)) + '\n\nCART NOW:\n' + agent.cart_text())
    prompt += '\n\nOTHER LISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
    got = agent.ask_json(prompt, agent.judges()[::-1], 3500)
    fits: list[tuple[float, str]] = []
    for entry in _items((got or {}).get('fits')):
        ident = str(entry.get('id') if isinstance(entry, dict) else entry)
        price = entry.get('price') if isinstance(entry, dict) else None
        if ident in agent.rows and ident not in lost and (ident not in agent.gone) and (ident not in agent.cart_ids()) and (ident not in [i for _, i in fits]):
            fits.append((float(price) if isinstance(price, (int, float)) and (not isinstance(price, bool)) else float('inf'), ident))
    ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if agent.brief.get('pick') == 'cheapest' else [i for _, i in fits]
    agent.trace.append({'kind': 'swap', 'lost': lost, 'fits': fits})
    return ranked[:4]

def ridge_rehear_if_doubtful(agent, batch: list[tuple[int, str, str]], heard: dict[str, Any]) -> dict[str, Any]:
    if agent.judges()[:1] == [HEAVY] and any((agent.doubtful(entry) for entry in _items(heard.get('needs')))):
        return agent.hear(batch, (HEAVY,)) or heard
    return heard

def ridge_refuse_trail(agent) -> str:
    live = {need for _, need in agent.needs_now()}
    if agent.refuse_trailed or agent.left() < TRAIL_ROOM or agent.clock() > HUNT_BY or (not any((need in live and need not in agent.ref_values for _, need in agent.refs))):
        return ''
    agent.refuse_trailed = True
    text, _ = agent.trail(HUNT_BY)
    return text + 'Search for those values now and buy what fits; end without a purchase only if nothing does.' if text else ''

def ridge_recovery_search(agent, got: dict[str, Any] | None, missing: list[int]) -> bool:
    keep = agent.tail() + len(agent.items) + 2
    room = min(RECOVERY_STEPS, agent.left() - keep)
    if room < 3 or agent.model_room() < CALL_FLOOR + 25 or agent.rushed():
        return False
    before = len(agent.full)
    last = agent.step + room
    searching = agent.step + max(1, room // 2)
    plan = agent.search_template()
    queries: list[tuple[int, str]] = []
    for entry in _items((got or {}).get('search')):
        if isinstance(entry, dict) and str(entry.get('query') or '').strip():
            queries.append((min(max(1, _count(entry.get('item')) or 1), len(agent.items)), str(entry['query']).strip()))
    queries = [q for q in queries if q[0] in missing] or [(n, ' '.join([v for v, _ in agent.value_words(n)][:2] + [agent.items[n - 1].name])) for n in missing]
    found: dict[int, list[str]] = {n: [] for n in missing}
    for number, words in queries[:4]:
        if plan and agent.step < searching and (not agent.finished):
            found.setdefault(number, [])
            found[number] += agent.search_words(plan[0], plan[1], plan[2], words)
    for number, rows in found.items():
        picks = [i for i in agent.kind_rows(number, rows) if i not in agent.full][:2]
        agent.fill_facts(picks[:max(0, last - agent.step)], keep)
    agent.trace.append({'kind': 'recovery', 'missing': missing, 'queries': queries[:4], 'opened': len(agent.full) - before})
    return len(agent.full) > before

def ridge_records_of_item(agent, number: int) -> int:
    cat_key = agent.category_key()
    item = agent.items[number - 1]
    return sum((1 for record in agent.full.values() if cat_key and agent.owns(item, str(record.get(cat_key) or '')) and _fact_block(record)))

def ridge_records_in(agent, observation: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(observation, dict):
        ident = agent.ident(observation)
        if ident:
            return [(ident, observation)]
        found: list[tuple[str, dict[str, Any]]] = []
        for value in observation.values():
            if isinstance(value, list):
                found = found + [(i, r) for r in value if isinstance(r, dict) for i in [agent.ident(r, True)] if i]
            elif isinstance(value, dict):
                inner = agent.ident(value, False, 1)
                found = found + ([(inner, value)] if inner else [(i, r) for v in value.values() if isinstance(v, list) for r in v if isinstance(r, dict) for i in [agent.ident(r, True)] if i])
        return found
    if isinstance(observation, list):
        return [(i, r) for r in observation if isinstance(r, dict) for i in [agent.ident(r, True)] if i]
    return []

def ridge_record_of(agent, ident: str) -> str:
    record = agent.full.get(ident) or agent.rows.get(ident)
    return _dump(_clip(record, 400))[:3000] if record else '(no record seen)'

def ridge_ready_to_ask(agent) -> bool:
    if agent.asked or not agent.roles.get('ask') or agent.msgs_free() <= 0:
        return False
    if agent.left() <= 8 or agent.clock() > ASK_BY:
        return False
    opened = len(agent.full)
    if agent.loop_calls < ASK_TURNS and (not all((agent.records_of_item(n) for n in range(1, len(agent.items) + 1)))):
        return False
    if opened >= 3 * len(agent.items) + 1:
        return True
    return agent.step >= max(8, agent.max_steps // 4)

def ridge_read_shape(agent, name: str, observation: Any) -> dict[str, Any] | None:
    if name in agent.shapes:
        return agent.shapes[name]
    if not isinstance(observation, dict):
        return None
    lists = [k for k, v in observation.items() if isinstance(v, list) and len(v) >= 2 and all((isinstance(r, dict) for r in v))]
    if not lists:
        return None
    agent.shapes[name] = None
    sample = {k: v[:3] if k in lists else v for k, v in observation.items()}
    prompt = SHAPE + '\n\nTOOL: ' + name + ' | parameters: ' + _dump(agent.schemas.get(name) or {}) + '\n\nRESULT (first rows only):\n' + _dump(_clip(sample, 120))[:4000]
    got = agent.ask_json(prompt, [LIGHT, HEAVY], 2500) or {}
    rows = got.get('rows') if got.get('rows') in lists else None
    if not rows:
        return None
    first = observation[rows][0]
    props = (agent.schemas.get(name) or {}).get('properties') or {}

    def named(key: Any, where: dict[str, Any]) -> str | None:
        return key if isinstance(key, str) and key in where else None
    shape = {'rows': rows, 'count': named(got.get('count'), observation), 'next': named(got.get('next'), observation), 'next_arg': named(got.get('next_arg'), props), 'price': named(got.get('price'), first), 'stock': named(got.get('stock'), first)}
    if shape['count'] and (isinstance(observation[shape['count']], bool) or not isinstance(observation[shape['count']], int)):
        shape['count'] = None
    agent.shapes[name] = shape
    agent.price_key = agent.price_key or shape['price']
    agent.stock_key = agent.stock_key or shape['stock']
    agent.trace.append({'kind': 'shape', 'tool': name, 'shape': shape})
    return shape

def ridge_read_request(agent) -> None:
    if _brief_schema_ready(agent.calls):
        lines = []
        for tool in agent.tools:
            fn = tool['function']
            lines.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {_dump(fn.get('parameters') or {})}")
        prompt = BRIEF + '\nTOOLS:\n' + '\n'.join(lines) + '\n\nREQUEST:\n' + agent.request
        if agent.view.get('opening') is not None:
            prompt += '\n\nOPENING NOTE FROM THE STORE:\n' + _dump(_clip(agent.view.get('opening'), 1500))
        brief = agent.ask_json(prompt, [LIGHT], 4000) or {}
        if not agent.no_credit and (not _items(brief.get('items')) or not isinstance(brief.get('roles'), dict) or (not brief['roles'].get('buy'))):
            brief = agent.ask_json(prompt, [HEAVY, LIGHT], 2500) or brief
        elif not agent.no_credit and agent.shared_shelf(brief):
            again = agent.ask_json(prompt, [HEAVY], 2500) or {}
            if _items(again.get('items')) and isinstance(again.get('roles'), dict) and again['roles'].get('buy'):
                brief = again
        while not brief and (not agent.no_credit) and (agent.calls < ALL_CALLS) and (agent.clock() < BRIEF_BY):
            time.sleep(BRIEF_GAP)
            brief = agent.ask_json(prompt, [LIGHT, HEAVY], 2500) or {}
        if brief:
            agent.waited = 0
        agent.brief = brief
        roles = brief.get('roles') if isinstance(brief.get('roles'), dict) else {}
        agent.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in agent.schemas}
        for key in ('buy_args', 'refuse_args'):
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        buy, refuse = (agent.roles.get('buy'), agent.roles.get('refuse'))
        if buy and buy == refuse and (not brief['buy_args'] or not brief['refuse_args'] or brief['buy_args'] == brief['refuse_args']):
            ask = f'The store tool {buy} both checks out and ends without a purchase, depending on its arguments. Its schema: ' + _dump(agent.schemas.get(buy)) + '\nReturn ONE JSON object and nothing else: {"buy_args": the fixed arguments that check out, "refuse_args": the fixed arguments that end without a purchase} (only the arguments that choose the mode).'
            fixed = agent.ask_json(ask, [LIGHT, HEAVY], 1500) or {}
            if isinstance(fixed.get('buy_args'), dict) and isinstance(fixed.get('refuse_args'), dict) and (fixed['buy_args'] != fixed['refuse_args']):
                brief['buy_args'], brief['refuse_args'] = (fixed['buy_args'], fixed['refuse_args'])
        for entry in _items(brief.get('items')):
            if not isinstance(entry, dict):
                continue
            quantity = _count(entry.get('quantity'))
            count = quantity if quantity and quantity > 0 else 1
            musts = [str(m) for m in _items(entry.get('musts')) if m]
            agent.items.append(Item(str(entry.get('name') or 'the product'), str(entry.get('shelf') or ''), count, musts))
        merged: list[Item] = []
        became: dict[int, int] = {}
        for old, item in enumerate(agent.items, 1):
            twin = next((m for m in merged if item.shelf.strip() and m.qty == item.qty and (_words(m.shelf) == _words(item.shelf)) and (not any((_words(a.split(':')[0]) == _words(b.split(':')[0]) and _words(a.partition(':')[2]) != _words(b.partition(':')[2]) for a in m.musts for b in item.musts)))), None)
            if twin is None:
                merged.append(item)
                became[old] = len(merged)
            else:
                became[old] = merged.index(twin) + 1
                twin.musts += [m for m in item.musts if m not in twin.musts]
                agent.trace.append({'kind': 'items_merged', 'into': twin.name, 'dropped': item.name})
        agent.items = merged
        if not agent.items:
            agent.items = [Item('the requested product', '', 1, [])]
        for pair in _items(brief.get('fits')):
            if isinstance(pair, dict):
                part, main = (became.get(_count(pair.get('item')) or 0), became.get(_count(pair.get('with')) or 0))
                if part and main and (part != main) and (1 <= part <= len(agent.items)) and (1 <= main <= len(agent.items)) and ((part, main) not in agent.bound):
                    agent.bound.append((part, main))
        for lookup in _items(brief.get('lookups')):
            text = str(lookup).strip()
            if not any((text in item.musts for item in agent.items)):
                key = _words(text.split(':')[0])
                text = next((m for item in agent.items for m in item.musts if _words(m.split(':')[0]) == key), text)
            number = next((n for n, item in enumerate(agent.items, 1) if text in item.musts), 1)
            if text and (number, text) not in agent.refs:
                agent.refs.append((number, text))
        agent.unmix()
        budget = brief.get('budget')
        if isinstance(budget, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', budget):
            budget = float(budget)
        if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0):
            agent.budget = budget
        limit = _count(brief.get('messages'))
        if limit is not None and limit >= 0:
            agent.msg_limit = limit
        elif agent.roles.get('ask'):
            agent.msg_limit = 3
        steps = _count(brief.get('steps'))
        if steps is not None and 10 <= steps < agent.max_steps:
            agent.max_steps = steps
        for entry in _items(brief.get('process')):
            if isinstance(entry, dict) and str(entry.get('step') or '').strip():
                when = str(entry.get('when') or '').strip().lower()
                agent.process.append({'step': str(entry['step']).strip(), 'when': when if when in ('choose', 'order', 'end') else 'order', 'messages': max(0, _count(entry.get('messages')) or 0)})
        if agent.roles.get('ask'):
            talk = [p['messages'] for p in agent.process if p['when'] in ('order', 'end') and p['messages']]
            agent.reserve = min(max(talk), agent.msg_limit) if talk else 0
    else:
        _brief_schema_snapshot(agent.calls, agent.request)
        return

def ridge_quiet(agent, name: str, args: dict[str, Any]) -> dict[str, Any]:
    schema = agent.schemas.get(name) or {}
    props = schema.get('properties') or {}
    out = dict(args)
    for key, spec in props.items():
        if not isinstance(spec, dict) or 'string' not in _kinds(spec) or 'enum' in spec:
            continue
        value = out.get(key)
        if value is None and key not in (schema.get('required') or []):
            continue
        text = str(value or '')
        named = any((ident in text for ident in agent.rows)) or agent.brief.get('quiet_decline') is not False
        if named or not text:
            out[key] = 'Sorry, nothing in the store meets all of your requirements, so I am not buying anything.'[:spec.get('maxLength', 200)]
    return out

def ridge_question(agent, number: int, label: str, samples: list[str] | None=None) -> str:
    text = f'What {label} would you like for the {agent.item_said(number)}'
    return text + (f', for example {samples[0]} or {samples[1]}?' if samples else '?')

def ridge_query_is_new(agent, name: str, args: dict[str, Any]) -> bool:
    mark = _dump({k: v for k, v in args.items() if isinstance(v, bool) or not isinstance(v, int)})
    seen = agent.page_queries.setdefault(name, set())
    fresh = mark not in seen
    seen.add(mark)
    return fresh

def ridge_price_sorted(agent, observation: Any) -> bool:
    prices = [agent.price_of(i) for i, _ in agent.records_in(observation)]
    return len(prices) >= 3 and None not in prices and all((a <= b for a, b in zip(prices, prices[1:])))

def ridge_price_of(agent, ident: str) -> float | None:
    value = (agent.rows.get(ident) or {}).get(agent.price_key) if agent.price_key else None
    if isinstance(value, dict):
        value = next((v for v in value.values() if isinstance(v, (int, float)) and (not isinstance(v, bool))), None)
    if isinstance(value, str):
        figure = re.fullmatch('\\s*[^\\d\\s]{0,3}\\s?(\\d[\\d,]*(?:\\.\\d+)?)\\s*[A-Za-z]{0,6}\\.?\\s*', value)
        value = float(figure.group(1).replace(',', '')) if figure else None
    return float(value) if isinstance(value, (int, float)) and (not isinstance(value, bool)) else None

def ridge_plan_lanes(agent, count: int) -> dict[int, list[str]]:
    agent.adopt_labels()
    table = agent.key_table()
    settled = []
    for number, item in enumerate(agent.items, 1):
        settled.append(f'item {number}: {item.name} - settled by the request: ' + ('; '.join(item.musts) or 'nothing'))
    listing = []
    last = None
    for index, (label, name, carried, values) in enumerate(table, 1):
        if label != last:
            listing.append(f'category {label!r}:')
            last = label
        listing.append(f'  {index}. {name} (on {carried}): ' + ' | '.join(values) + agent.not_stated(label, name, values) + (' [several features in one: ask last]' if re.search(CATCH_ALL, name) else ''))
    prompt = PLAN.replace('COUNT', str(max(1, count))) + '\n\nREQUEST:\n' + agent.request + '\n\nITEMS:\n' + '\n'.join(settled)
    first = [str(x) for x in _items(agent.brief.get('ask_first')) if str(x).strip()]
    if first:
        prompt += '\n\nASK FIRST (the request asks to be asked about these):\n' + '\n'.join(first)
    prompt += '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(listing) or '(no full listing records yet)')
    plan = agent.ask_json(prompt, [HEAVY, LIGHT], 3500) or {}
    lanes: dict[int, list[str]] = {n: [] for n in range(1, len(agent.items) + 1)}
    hinted: dict[int, set[str]] = {n: set() for n in lanes}
    for pick in _items(plan.get('ask')):
        if not isinstance(pick, dict):
            continue
        number = min(max(1, _count(pick.get('item') or 1) or 1), len(agent.items))
        label = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(pick.get('label') or ''))[0].strip().rstrip('?.')
        if not label or len(label.split()) > 5:
            continue
        if all((_words(label) != _words(other) for other in lanes[number])):
            lanes[number].append(label)
        if str(pick.get('hint') or '').strip():
            hinted[number].add(_words(label))
            if number not in agent.plan_order:
                agent.plan_order.append(number)
    for number in lanes:
        keys = agent.lane_keys(number, table)
        read = max(1, agent.records_of_item(number))
        own: dict[str, float] = {}
        for label, key, carried, _ in table:
            if agent.owner_of(label) == number:
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
            if weak is not None and len(agent.items) > 1:
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
    agent.trace.append({'kind': 'plan', 'ask': _clip(plan.get('ask'), 120), 'lanes': {n: v[:8] for n, v in lanes.items()}})
    return lanes

def ridge_part_shelves(agent) -> None:
    cat_key = agent.category_key()
    labels = sorted({str(row.get(cat_key) or '').strip() for row in agent.rows.values()} - {''}) if cat_key else []
    moved = False
    for item in agent.items:
        name = _roots(item.name)
        if not name or not item.shelf.strip() or (not any((other is not item and _words(other.shelf) == _words(item.shelf) for other in agent.items))):
            continue
        if any((other is not item and (_roots(other.name) == name and True) for other in agent.items)):
            continue
        own = [label for label in labels if _roots(label) == name and _words(label) != _words(item.shelf)]
        if len(own) == 1 and (not any((_words(other.shelf) == _words(own[0]) for other in agent.items))):
            agent.trace.append({'kind': 'shelf_parted', 'item': item.name, 'from': item.shelf, 'to': own[0]})
            item.shelf = own[0]
            moved = True
    if moved:
        agent.owners_seen = {}
        agent.fit_cached = None

def ridge_paging(agent, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
    shape = agent.read_shape(name, observation)
    if not shape or not shape['count'] or (not shape['next']) or (not shape['next_arg']) or (not isinstance(observation, dict)):
        return None
    rows, count, value = (observation.get(shape['rows']), observation.get(shape['count']), observation.get(shape['next']))
    if not isinstance(rows, list) or not rows or isinstance(count, bool) or (not isinstance(count, int)):
        return None
    if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))):
        return None
    return (count, len(rows), shape['next_arg'], value)

def ridge_page_tools(agent) -> set[str]:
    page = agent.roles.get('offer') or agent.roles.get('open')
    return {t for t in (page, agent.roles.get('compare')) if t}

def ridge_page_text(agent, ident: str, record: dict[str, Any]) -> str:
    block = _fact_block(record) or {}
    rest = {k: v for k, v in record.items() if v is not block and (not isinstance(v, (dict, list)))}
    return f'{ident}: attributes ' + _dump(_clip(block, 120))[:PAGE_ATTR_CHARS] + ' | ' + _dump(_clip(rest, 200))[:PAGE_REST_CHARS]

def ridge_page_read(agent, ident: str) -> bool:
    page = agent.roles.get('offer')
    return ident in agent.opened and (not page or not agent.single_target(page) or page in agent.views.get(ident, ()))

def ridge_owns(agent, item: Item, label: str) -> bool:
    return label and item in agent.items and (agent.owner_of(label) == agent.items.index(item) + 1 or _words(label) == _words(item.shelf or ''))

def ridge_owner_of(agent, label: str) -> int:
    if not label:
        return 0
    if label in agent.adopted:
        return agent.adopted[label]
    known = agent.owners_seen.get((label, len(agent.items)))
    if known is not None:
        return known
    words = _words(label)
    long = _roots(label)
    best, top = (0, 0.0)
    for number, item in enumerate(agent.items, 1):
        if words == _words(item.shelf or ''):
            score = 3.0
        elif words == _words(item.name):
            score = 2.0
        else:
            mine = _roots(item.shelf or '') | _roots(item.name)
            score = len(long & mine) / max(1, len(long | mine))
        if score > top:
            best, top = (number, score)
    agent.owners_seen[label, len(agent.items)] = best
    return best

def ridge_others(agent, skip: list[str], limit: int=30) -> str:
    lines = []
    order = list(reversed(agent.seen_order))
    if agent.brief.get('pick') == 'cheapest':
        order.sort(key=lambda i: (i not in agent.full, agent.price_of(i) if agent.price_of(i) is not None else float('inf')))
    for ident in order:
        if ident in skip or agent.sold_out(ident):
            continue
        record = agent.full.get(ident) or agent.rows.get(ident)
        lines.append(_dump(_clip(record, 160))[:900])
        if len(lines) >= limit:
            break
    return '\n'.join(lines)

def ridge_opened_shop(agent, about: str) -> list[str]:
    not_names = agent.not_maker_keys()
    low = about.lower()
    out = []
    for ident in agent.opened:
        row = agent.rows.get(ident) or {}
        names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and (k not in not_names) and (v.strip().lower() not in NO_NAME)]
        if any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)):
            out.append(ident)
    return out

def ridge_open_rows(agent, rows: list[str]) -> bool:
    many, joiner = (agent.compare_target(), agent.roles.get('compare'))
    tool = agent.facts_tool()
    target = agent.single_target(tool)
    if many and joiner and (not agent.compare_bare()):
        plan = [(joiner, {many: rows[:agent.compare_room()]})]
    elif tool and target:
        plan = [(tool, {target: i}) for i in rows[:2]]
    else:
        return False
    cart_steps = len(agent.cart_calls({f'#{n}': 1 for n, _ in enumerate(agent.items)}, []) or agent.items)
    if agent.left() < len(plan) + max(5, agent.tail() + cart_steps):
        return False
    before = len(agent.full)
    for name, args in plan:
        faults: list[str] = []
        args = _shape(args, agent.schemas.get(name), 'args', faults)
        if faults:
            return False
        reply = agent.act(name, args)
        if reply is None or agent.finished:
            break
        agent.observe(name, reply.get('observation'))
    agent.trace.append({'kind': 'audit_open', 'rows': rows, 'opened': len(agent.full) - before})
    return len(agent.full) > before

def ridge_offered(agent) -> list[dict[str, Any]]:
    left = agent.left()
    ask = agent.roles.get('ask')
    keep: set[str] | None = None
    ends = {agent.roles.get('buy'), agent.roles.get('refuse')} - {None}
    if left <= agent.tail():
        keep = set(ends)
    elif left <= agent.tail() + 3:
        keep = set(ends) | {agent.roles.get('cart'), agent.roles.get('cart_view')}
    elif left <= len(agent.items) + 5 and agent.buys == 0:
        keep = set(ends) | {agent.roles.get('cart'), agent.roles.get('cart_view'), agent.roles.get('offer')}
    tools = [t for t in agent.tools if t['function']['name'] != ask or agent.message_args('x' * 200) is None]
    if keep is not None and ends:
        narrowed = [t for t in tools if t['function']['name'] in keep]
        tools = narrowed or tools
    return tools

def ridge_offer_tool(agent) -> str | None:
    return agent.roles.get('offer') or agent.roles.get('open')

def ridge_observe(agent, name: str, observation: Any) -> str:
    agent.absorb(observation)
    lines = agent.loose_lines(observation)
    if not lines:
        return ''
    meaning = agent.take_note(lines)
    if meaning:
        agent.queued.append(f'(seen on {name}) {meaning}')
    return meaning

def ridge_note_view(agent, name: str, observation: Any) -> None:
    for ident, record in agent.records_in(observation):
        keys = set(record)
        agent.tool_keys.setdefault(name, set()).update(keys)
        version, have = agent.view_keys.get(ident, (-1, set()))
        agent.view_keys[ident] = (agent.version, (have if version == agent.version else set()) | keys)

def ridge_note_read(agent, name: str, args: dict[str, Any], seen: Any) -> None:
    if name in agent.page_tools() or name in {agent.roles.get(r) for r in ('ask', 'cart', 'cart_view', 'buy', 'refuse')}:
        return
    agent.query_is_new(name, args)
    if isinstance(seen, (dict, list)) and (not _failed(seen) and True):
        agent.read_calls[agent.call_mark(name, args)] = seen
        agent.read_args[agent.call_mark(name, args)] = (name, dict(args))

def ridge_note_facts(agent, tool: str, shown: bool) -> None:
    if shown:
        agent.facts_from[tool] = True
    elif agent.facts_from.get(tool) is not True:
        agent.bare_pages[tool] = agent.bare_pages.get(tool, 0) + 1
        if agent.bare_pages[tool] >= 2:
            agent.facts_from[tool] = False

def ridge_note_cart(agent, name: str, args: dict[str, Any]) -> None:
    agent.cart_touched = True
    agent.cart_at = agent.step
    spec = agent.schemas.get(name, {}).get('properties') or {}
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
                    ident = next((v for v in line.values() if isinstance(v, str) and v in agent.rows), None)
                    count = next((_count(v) for v in line.values() if _count(v) is not None and (not isinstance(v, str))), 1)
                    if ident:
                        lines.append((ident, count))
                elif isinstance(line, str) and line in agent.rows:
                    lines.append((line, loose_count))
        elif isinstance(value, str) and value in agent.rows:
            lines.append((value, loose_count))
    if whole:
        agent.cart_lines = {i: c for i, c in lines if c and c > 0}
        return
    sets = agent.cart_sets(name)
    for ident, count in lines:
        have = agent.cart_lines.get(ident, 0)
        if removal:
            have = have - (count or 1) if any(('integer' in _kinds(v) for v in spec.values() if isinstance(v, dict))) else have - count if count else 0
        elif sets:
            have = count if count is not None else 1
        else:
            have = have + (count if count is not None else 1)
        if have > 0:
            agent.cart_lines[ident] = have
        else:
            agent.cart_lines.pop(ident, None)

def ridge_not_stated(agent, label: str, name: str, values: list[str]) -> str:
    number = agent.owner_of(label)
    if not number:
        return ''
    for must in agent.items[number - 1].musts:
        prop, _, value = must.partition(':')
        if re.search('\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', value, re.I):
            continue
        words = set(re.findall('[a-z0-9]{3,}', value.lower()))
        shown = set(re.findall('[a-z0-9]{3,}', ' '.join(values).lower()))
        if prop.strip() and _words(prop) != name and _stem(prop) & _stem(name) and words and (not words <= shown):
            return f' [not the stated {prop.strip()}: another property]'
    return ''

def ridge_not_maker_keys(agent) -> tuple[Any, ...]:
    return (agent.id_key, agent.category_key(), agent.title_key())

def ridge_next_ask(agent, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> tuple[int, list[str]]:
    first = {n: i for i, n in enumerate(agent.plan_order)}
    scarce = agent.msg_limit <= SCARCE_MESSAGES and (not agent.listing)
    for number in sorted(lanes, key=lambda n: (found.get(n, 0) > 0, 0 if scarce else spent.get(n, 0), first.get(n, len(first) + n))):
        left = [label for label in lanes[number] if _words(label) not in used[number] and (not agent.told_about(number, label))]
        if left:
            return (number, left[:agent.list_now() if agent.listing else 1])
    return (0, [])

def ridge_needs_now(agent) -> list[tuple[int, str]]:
    out = []
    for number, item in enumerate(agent.items, 1):
        for need in item.musts + item.told:
            if need not in item.dropped:
                out.append((number, need))
    return out

def ridge_named_label(agent, number: int, label: str) -> str:
    item = agent.items[number - 1]
    cat_key = agent.category_key()
    wanted = _words(item.shelf or '')
    if not wanted or not cat_key or _words(label) == wanted:
        return ''
    return next((str(r.get(cat_key)) for r in agent.rows.values() if _words(str(r.get(cat_key) or '')) == wanted), '')

def ridge_name_parts(agent, number: int) -> list[str]:
    name = agent.items[number - 1].name
    if not re.search('\\band\\b', name, re.I):
        return []
    parts = [p.strip() for p in re.split('\\s*(?:,|\\band\\b)\\s*', name, flags=re.I) if p.strip()]
    return parts if len(parts) >= 2 and all((re.search('[a-z]{3}(?<![su])s$', p, re.I) for p in parts)) else []

def ridge_msgs_free(agent) -> int:
    return agent.msg_limit - agent.msg_used - max(0, agent.reserve - agent.gate_msgs)

def ridge_more_rows_for_keys(agent, tool: str, args: dict[str, Any], seen: Any, stop_at: int, number: int) -> list[str]:
    page = agent.paging(tool, seen)
    if not page or page[3] is None or agent.rows_of_item(number) >= OPEN_EACH or agent.finished or (agent.step >= stop_at):
        return []
    faults: list[str] = []
    call = _shape({**args, page[2]: page[3]}, agent.schemas.get(tool), 'args', faults)
    if faults:
        return []
    reply = agent.act(tool, call)
    if reply is None:
        return []
    agent.observe(tool, reply.get('observation'))
    return [f'next page: ' + agent.show(reply.get('observation'))]

def ridge_more_pages(agent, name: str, args: dict[str, Any], observation: Any) -> str:
    agent.paged_obs = []
    pick = agent.brief.get('pick')
    plan = agent.paging(name, observation) if pick in ('cheapest', 'most', 'least') else None
    if not plan:
        return ''
    count, shown, param, value = plan
    given = args.get(param)
    later = given not in (None, 0, 1) if isinstance(given, int) and (not isinstance(given, bool)) else str(given or '').strip()
    if value is None or later or count <= shown:
        return ''
    if pick == 'cheapest' and agent.price_sorted(observation):
        return ''
    pages = -(-(count - shown) // shown)
    if pages > 2 and agent.pages_read + 2 <= AUTO_PAGES and (agent.left() - agent.tail() - 2 >= 12):
        pages = 2
    if pages > 2 or agent.pages_read + pages > AUTO_PAGES or agent.left() - agent.tail() - pages < 12:
        return f"\n[Ridge] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with {param}={_dump(value)}."
    texts = []
    for _ in range(pages):
        faults: list[str] = []
        call = _shape({**args, param: value}, agent.schemas.get(name), 'args', faults)
        if faults:
            break
        reply = agent.act(name, call)
        if reply is None or agent.finished:
            break
        agent.pages_read += 1
        seen = reply.get('observation')
        agent.observe(name, seen)
        agent.paged_obs.append(seen)
        shown_text = agent.show(seen)
        agent.read_cache[name + _dump(call)] = (agent.version, agent.step, shown_text)
        texts.append(f'{param}={_dump(value)}: {shown_text}')
        after = agent.paging(name, seen)
        value = after[3] if after else None
        if value is None:
            break
    agent.trace.append({'kind': 'pages', 'tool': name, 'read': len(texts)})
    rest = f"\n[Ridge] {count} listings matched and Ridge has read {shown * (len(texts) + 1)} rows of them; narrow the search with the requirement's value or page on to see the rest." if value is not None and texts else ''
    return '\n[Ridge read the next pages of this search itself, one step a page]\n' + '\n'.join(texts) + rest if texts else ''

def ridge_missing_items(agent) -> list[int]:
    cat_key = agent.category_key()
    ids = [i for i in agent.cart_ids() if (agent.held_units(i) or 0) > 0]
    if not cat_key or len(ids) >= len(agent.items):
        return []
    labels = [str(r.get(cat_key) or '') for r in agent.rows.values()]
    held = [str((agent.rows.get(i) or {}).get(cat_key) or '') for i in ids]
    return [n for n, item in enumerate(agent.items, 1) if any((agent.owns(item, label) for label in labels)) and (not any((agent.owns(item, label) for label in held)))]

def ridge_message_args(agent, text: str) -> dict[str, Any] | None:
    name = agent.roles.get('ask')
    schema = agent.schemas.get(name or '') or {}
    props = schema.get('properties') or {}
    required = schema.get('required') or []
    texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v) and ('const' not in v)]
    pool = [k for k in required if k in texts] or texts
    args: dict[str, Any] = {}
    if pool:
        args[max(pool, key=lambda k: (not re.search(NOT_MESSAGE, _snake(k), re.I), re.search(MESSAGE_FIELD, _snake(k), re.I), props[k].get('maxLength') or 10 ** 6))] = text
    else:
        for key, spec in props.items():
            inner = next((k for k, v in (spec.get('properties') or {}).items() if isinstance(v, dict) and 'string' in _kinds(v)), None) if isinstance(spec, dict) else None
            if isinstance(spec, dict) and 'array' in _kinds(spec):
                args[key] = [text]
                break
            if inner:
                args[key] = {inner: text}
                break
    if not args:
        return None
    for key in required:
        if key not in args:
            args[key] = _filler(props.get(key) or {})
            if args[key] is None:
                return None
    faults: list[str] = []
    args = _shape(args, schema, 'args', faults)
    return None if faults else args

def ridge_loose_lines(agent, observation: Any) -> list[str]:
    lines: list[str] = []
    if isinstance(observation, dict) and (not agent.ident(observation)):
        for key, value in observation.items():
            if isinstance(value, str) and len(value.split()) >= 5:
                lines.append(value)
            elif isinstance(value, list) and value and all((isinstance(v, str) for v in value)):
                lines.extend((v for v in value if len(v.split()) >= 5))
            elif isinstance(value, dict) and (not agent.ident(value, False, 1)):
                for inner in value.values():
                    if isinstance(inner, str) and len(inner.split()) >= 6:
                        lines.append(inner)
    return [line for line in lines if line not in agent.heard]

def ridge_look_alikes(agent, number: int, label: str) -> list[str]:
    mine = _words(label)
    seen: dict[str, set[str]] = {}
    for ident in agent.item_ids(number):
        for key, value in (_fact_block(agent.full.get(ident) or {}) or ({} and True)).items():
            seen.setdefault(_words(key), set()).update((w for w in _roots(str(value)) if len(w) > 1))
    counts: dict[str, int] = {}
    for name in seen:
        for word in set(name.split()):
            counts[word] = counts.get(word, 0) + 1
    plain = {w for w in mine.split() if counts.get(w, 0) <= 2}
    scored = []
    for order, (name, values) in enumerate(seen.items()):
        if name != mine and name:
            shared = plain & set(name.split())
            common = seen.get(mine, set()) & values
            if shared or common:
                scored.append((-(3 * len(shared) + min(len(common), 3)), order, name))
    return [name for _, _, name in sorted(scored)[:LOOK_ALIKES]]

def ridge_lists_things(agent, tool: str) -> bool:
    target = agent.single_target(tool) or ''
    return (re.search(SHELF_ARG, _snake(target), re.I) or re.search(QUERY_WORDS, _snake(target), re.I)) or tool in agent.shapes

def ridge_list_now(agent) -> int:
    steps = LIST_STEPS if len(agent.items) == 1 else LIST_STEPS_MIXED
    return min(steps[agent.list_asks], agent.list_cap()) if agent.list_asks < len(steps) else agent.list_cap()

def ridge_list_cap(agent) -> int:
    return LIST_SIZE if len(agent.items) == 1 else LIST_MIXED

def ridge_list_below(agent, cart: list[str]) -> None:
    plan = agent.cap_template()
    cat_key = agent.category_key()
    if not plan or not cat_key:
        return
    tool, base, shelf, cap = plan
    for line in cart:
        price = agent.price_of(line)
        label = str((agent.rows.get(line) or {}).get(cat_key) or '')
        if price is None or price < 2 or (not label) or ((line, price) in agent.capped):
            continue
        agent.capped.add((line, price))
        limit = int(price) - 1 if float(price).is_integer() else int(price)
        tried: set[int] = set()
        for _ in range(WALK_CALLS):
            if agent.left() <= agent.tail() + (WALK_ROOM if tried else 7) or agent.finished or agent.rushed():
                return
            if limit < 1 or limit in tried:
                break
            tried.add(limit)
            faults: list[str] = []
            args = _shape({**base, shelf: label, cap: limit}, agent.schemas.get(tool), 'args', faults)
            if faults:
                return
            reply = agent.act(tool, args)
            if reply is None:
                return
            seen = reply.get('observation')
            agent.observe(tool, seen)
            rows = agent.records_in(seen)
            agent.trace.append({'kind': 'list_below', 'line': line, 'price': price, 'limit': limit, 'rows': len(rows)})
            page = agent.paging(tool, seen)
            if not rows or any(((agent.price_of(i) or 0) > limit for i, _ in rows)) or (not page) or (page[0] <= page[1]) or agent.price_sorted(seen):
                break
            if page[0] <= 2 * page[1]:
                call = _shape({**args, page[2]: page[3]}, agent.schemas.get(tool), 'args', faults) if page[3] is not None else None
                more = None if faults or call is None or agent.left() <= agent.tail() + 7 or agent.rushed() else agent.act(tool, call)
                if more is not None:
                    agent.observe(tool, more.get('observation'))
                break
            low = min((p for i, _ in rows for p in [agent.price_of(i)] if p is not None), default=None)
            if low is None:
                break
            limit = int(low) - 1 if float(low).is_integer() else int(low)

def ridge_link_system(agent, part: int, main: int, index: dict[str, Any]) -> tuple[int, int, set[str]] | None:
    marks = index['marks']
    for carrier, namer in ((part, main), (main, part)):
        rows = agent.item_ids(carrier)
        tagged = [i for i in rows if (marks.get(i) or {}).get('fam')]
        families = {f for i in tagged for f in marks[i]['fam']}
        if len(tagged) < LINK_ROWS or 2 * len(tagged) < len(rows) or len(families) > LINK_FAMILIES:
            continue
        if any((f in families for i in agent.item_ids(namer) for f in (marks.get(i) or {}).get('named') or [])):
            return (carrier, namer, families)
    return None

def ridge_link_holds(agent, carried: str, namer: str, families: set[str], index: dict[str, Any]) -> bool | None:
    own = set((index['marks'].get(carried) or {}).get('fam') or []) & families
    named = set((index['marks'].get(namer) or {}).get('named') or []) & families
    return own & named if own and named else None

def ridge_link_faults(agent, part: int, main: int, device: str, lines: list[str], index: dict[str, Any]) -> list[str]:
    system = agent.link_system(part, main, index)
    if not system:
        return []
    carrier, _, families = system
    marks = index['marks']
    faults = []
    for line in lines:
        carried, namer = (line, device) if carrier == part else (device, line)
        if agent.link_holds(carried, namer, families, index) is not False:
            continue
        wanted = set(marks[namer].get('named') or []) & families
        good = [i for i in agent.item_ids(carrier) if set(marks[i].get('fam') or []) & wanted and (not agent.sold_out(i)) and (i != carried)][:4]
        faults.append(f"{carried} is of series {', '.join(sorted(set(marks[carried].get('fam') or []) & families))}, while {namer} takes {', '.join(sorted(wanted))}" + (f"; seen listings of that series: {', '.join(good)}" if good else f"; search that series with the item's words, or take another {agent.items[main - 1].name} whose series the wanted lines carry"))
    return faults

def ridge_line_faults(agent) -> list[str]:
    agent.adopt_labels()
    cat_key = agent.category_key()
    ids = [i for i in agent.cart_ids() if (agent.held_units(i) or 0) > 0]
    if not ids or not cat_key:
        return []
    label = {i: str((agent.rows.get(i) or {}).get(cat_key) or '') for i in ids}
    known = {n for n, item in enumerate(agent.items, 1) if any((agent.owns(item, str(r.get(cat_key) or '')) for r in agent.rows.values()))}
    owner = {i: [n for n, item in enumerate(agent.items, 1) if agent.owns(item, label[i])] for i in ids}
    faults: list[str] = []
    for n, item in enumerate(agent.items, 1):
        if n not in known:
            continue
        lines = [i for i in ids if n in owner[i]]
        if not lines:
            faults.append(f'item {n} ({item.name}) has no line in the cart')
        elif len([i for i in lines if owner[i] == [n]]) > 1:
            faults.append(f"item {n} ({item.name}) has {len(lines)} lines ({', '.join(lines)}); keep one")
        for i in lines:
            have = agent.held_units(i)
            if owner[i] == [n] and have != item.qty:
                faults.append(f'{i} is held at {have} unit(s); item {n} ({item.name}) wants {item.qty}')
            fuller = agent.fuller_label(n, label[i]) or agent.named_label(n, label[i])
            if fuller:
                faults.append(f'{i} is from {label[i]}, but the store has the label {fuller}, which names item {n} ({item.name}) in full: take the line from there')
    if len(known) == len(agent.items):
        for i in ids:
            if not owner[i]:
                faults.append(f"{i} ({label[i] or 'unknown kind'}) is not a kind the shopper asked for: take it out")
    for i in ids:
        if agent.sold_out(i):
            faults.append(f'{i} is sold out')
    prices = {i: agent.price_of(i) for i in ids}
    if isinstance(agent.budget, (int, float)) and all((p is not None for p in prices.values())):
        if agent.brief.get('budget_scope') == 'each':
            over = [i for i in ids if agent.budget < prices[i] <= 10 * agent.budget]
            faults += [f'{i} costs {prices[i]:g}, over the {agent.budget:g} limit for each item' for i in over]
        else:
            total = sum((prices[i] * agent.held_units(i) for i in ids))
            if agent.budget + 1e-09 < total <= 10 * agent.budget:
                faults.append(f'the total is {total:g}, over the budget of {agent.budget:g}')
    return faults

def ridge_ledger(agent) -> str:
    parts = []
    many = len(agent.items) > 1
    for number, item in enumerate(agent.items, 1):
        live = [m for m in item.musts + item.told if m not in item.dropped]
        told = [m for m in live if agent.described(number, m)]
        needs = [m + (f' (the page it points at gives: {agent.ref_values[m]})' if m in agent.ref_values else '') for m in live if m not in told]
        head = f'item {number}: {item.name}' if many else f'item: {item.name}'
        if item.shelf:
            head += f' (store label: {item.shelf})'
        head += f', quantity {item.qty}'
        text = head + '; requirements: ' + ('; '.join(needs) if needs else 'none beyond the kind of product')
        if told:
            text += "; described in the shopper's words, which no listing attribute states (decides nothing): " + '; '.join(told)
        parts.append(text)
    brief = agent.brief
    if agent.budget is not None:
        scope = 'for each item' if brief.get('budget_scope') == 'each' else 'in total'
        parts.append(f"budget: at most {agent.budget} {brief.get('budget_unit') or ''} {scope}".replace('  ', ' '))
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
    for entry in agent.process:
        when = {'choose': 'before choosing', 'order': 'at the order (Ridge does it when you check out)', 'end': 'at an ending without a purchase (Ridge does it then)'}[entry['when']]
        parts.append(f"step the request asks for, {when}: {entry['step']}")
    facts = agent.facts_tool()
    if facts and (agent.facts_from.get(facts) is True or agent.roles.get('offer')):
        bare = sorted((t for t in agent.page_tools() | {agent.roles.get('offer') or ''} if t and agent.facts_from.get(t) is False))
        pages = sorted((t for t in agent.page_tools() if t != facts))
        parts.append(f'listing attributes come from {facts}' + (f"; {', '.join(bare)} show none" if bare else '') + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ''))
    return '\n'.join(('- ' + p for p in parts))

def ridge_learn_rows(agent, node: list[Any]) -> bool:
    rows = [n for n in node if isinstance(n, dict) and len(n) >= 3]
    if not rows or len(rows) < len(node) / 2:
        return False
    if agent.id_key and all((_id_like(r.get(agent.id_key), True) for r in rows)):
        return True
    found = _learn_id(rows)
    if found:
        agent.id_key = found
        return True
    return False

def ridge_late_question(agent) -> str:
    if agent.late_asked or not agent.roles.get('ask') or agent.msg_used >= agent.msg_limit:
        return ''
    if any((p['when'] == 'order' and p['messages'] > 0 for p in agent.process)):
        return ''
    if agent.left() < LATE_STEPS or agent.clock() > LATE_BY or (not agent.judges()):
        return ''
    agent.late_asked = True
    if not agent.lanes:
        agent.lanes = agent.plan_lanes(agent.list_cap() if agent.listing else 1)
        agent.lane_found = {n: 0 for n in agent.lanes}
        agent.lane_spent = {n: 0 for n in agent.lanes}
        agent.lane_used = {n: set() for n in agent.lanes}
    told: list[str] = []
    budget = agent.budget
    batch = agent.ask_next(told, LATE_STEPS - 2)
    if not batch:
        agent.trace.append({'kind': 'late_ask', 'skipped': 'no property left to ask or no message sent'})
        return ''
    if agent.budget != budget:
        told.append(f'budget now {agent.budget}')
    number, question, answer = batch[0]
    agent.trace.append({'kind': 'late_ask', 'q': question, 'a': answer[:200], 'told': told})
    if not told:
        return ''
    agent.version += 1
    summary = f'Ridge asked the shopper one more question before checking out (one step):\nQ: {question}\nA: {answer}\nNew firm requirements from this answer: ' + '; '.join(told) + '. The cart was not checked out: check each line against them (search again with the new values where needed), fix the cart, then check out.'
    pointed = [need for _, need in agent.refs if any((need in t for t in told))]
    if pointed:
        summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value.'
    searched = agent.value_search([number])
    return summary + ('\n' + searched if searched else '')

def ridge_lane_keys(agent, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
    item = agent.items[number - 1]
    settled = [_stem(m.split(':', 1)[0]) for m in item.musts]
    rows = [(label, key, carried, len(values)) for label, key, carried, values in table if agent.owner_of(label) == number]
    measured = {key: sum((1 for v in values if re.match('^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$', v))) * 2 > len(values) for _, key, _, values in table}
    if len(agent.items) == 1 and (not rows):
        rows = [(label, key, carried, len(values)) for label, key, carried, values in table]
    rows.sort(key=lambda r: (-r[2], measured.get(r[1], False), -r[3]))
    every = [key for _, key, _, _ in rows if not any((_names(prop, _stem(key)) for prop in settled))]
    keys = [key for key in every if not re.search(NOT_PROPERTY, key) or (re.search('\\w[ _]models?\\b', key) and True)]
    return [key for key in keys if not re.search(CATCH_ALL, key)] + [key for key in keys if re.search(CATCH_ALL, key)] + [key for key in every if key not in keys]

def ridge_labels(agent) -> set[str]:
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
    for schema in agent.schemas.values():
        walk(schema, 0)
    return found

def ridge_kind_rows(agent, number: int, found: list[str], below: float | None=None) -> list[str]:
    cat_key = agent.category_key()
    item = agent.items[number - 1]
    wanted = {w for w in re.findall('[a-z0-9]{3,}', ' '.join((v for v, _ in agent.value_words(number))).lower())}
    rows = [i for i in dict.fromkeys(found) if agent.buyable(i) and (not cat_key or agent.owns(item, str((agent.rows.get(i) or {}).get(cat_key) or '')))]
    if below is not None:
        rows = [i for i in rows if (agent.price_of(i) if agent.price_of(i) is not None else below) < below]
    return sorted(rows, key=lambda i: (i in agent.full, -len(wanted & set(re.findall('[a-z0-9]{3,}', _dump(agent.rows.get(i) or {}).lower()))), agent.price_of(i) or 0))

def ridge_key_table(agent) -> list[tuple[str, str, int, list[str]]]:
    cat_key = agent.category_key()
    groups: dict[str, dict[str, Any]] = {}
    for ident in agent.seen_order:
        record = agent.full.get(ident)
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
            slot = group['keys'].setdefault(name, {'n': 0, 'values': [], 'per': {}, 'plain': {}})
            slot['n'] += 1
            text = str(value).strip().lower()
            slot['per'][ident] = text
            slot['plain'][ident] = _FIRST_SPELLING.get(str(value).strip(), text).strip().lower()
            if text not in slot['values']:
                slot['values'].append(text)
    wanted = [(_words(item.shelf or ''), _words(item.name), {w for w in _stem(item.shelf or '') | _stem(item.name) if len(w) >= 4}) for item in agent.items]
    matched = {label for label in groups if any((_words(label) in (shelf, name) or ({w for w in _stem(label) if len(w) >= 4} & long and True) for shelf, name, long in wanted)) or agent.owner_of(label)}
    if matched:
        groups = {label: group for label, group in groups.items() if label in matched}
    table = []
    for label, group in sorted(groups.items(), key=lambda kv: -kv[1]['n']):
        kept: list[tuple[str, dict[str, Any]]] = []
        for name, slot in sorted(group['keys'].items(), key=lambda kv: (-kv[1]['n'], len(kv[0]))):
            values = list(slot['plain'].values())
            if not values:
                continue
            mean = sum((len(v) for v in values)) / len(values)
            if mean > 40 or sum((1 for v in values if ' / ' in v)) * 2 >= len(values):
                continue
            recur = sum((len(set(re.findall('[a-z]{3,}', v)) & set(re.findall('[a-z]{3,}', ' '.join((o for o in values if o is not v))))) > 0 for v in values))
            if len(values) >= 4 and len(set(values)) == len(values) and (mean > 12) and (recur * 2 < len(values)):
                continue
            twin = next((spec for other, spec in kept if agent.twins(other, spec, name, slot)), None)
            if twin is not None:
                for ident, text in slot['per'].items():
                    twin['per'].setdefault(ident, text)
                    twin['plain'].setdefault(ident, slot['plain'][ident])
                    if text not in twin['values']:
                        twin['values'].append(text)
                twin['n'] = len(twin['per'])
                continue
            kept.append((name, slot))
        kept.sort(key=lambda kv: -kv[1]['n'])
        for name, slot in kept[:TABLE_KEYS]:
            table.append((label, name, slot['n'], [v[:40] for v in slot['values'][:4]]))
    return table

def ridge_judges(agent, limit: float=SPEND_SOFT) -> list[str]:
    if agent.cost <= limit and agent.cost + HEAVY_ROOM <= SPEND_HARD:
        return [HEAVY, LIGHT]
    return [LIGHT] if agent.cost <= SPEND_HARD else []

def ridge_join_opens(agent, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    opener, joiner, target = (agent.roles.get('open'), agent.roles.get('compare'), agent.single_target(agent.roles.get('open')))
    many = agent.compare_target()
    if not opener or not joiner or (not target) or (not many) or (opener == joiner):
        return calls
    if agent.compare_bare():
        return calls
    picks: list[tuple[int, str]] = []
    for index, call in enumerate(calls):
        fn = call.get('function') or {}
        args = _json_obj(fn.get('arguments')) if isinstance(fn.get('arguments'), str) else None
        ident = str((args or {}).get(target) or '').strip()
        if fn.get('name') == opener and args and (set(args) == {target}) and (ident in agent.rows) and (ident not in agent.gone):
            picks.append((index, ident))
    picks = picks[:agent.compare_room()]
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
    agent.trace.append({'kind': 'joined', 'ids': [i for _, i in picks]})
    return out

def ridge_item_said(agent, number: int) -> str:
    item = agent.items[number - 1]
    parts = agent.name_parts(number)
    if not parts:
        return item.name
    if number not in agent.one_product:
        return re.sub('\\band\\b', 'or', item.name, flags=re.I)
    cat_key = agent.category_key()
    rows = [row for row in agent.rows.values() if not cat_key or agent.owns(item, str(row.get(cat_key) or ('' and True)))]
    named = [_roots(' '.join((str(v) for k, v in row.items() if k != cat_key and isinstance(v, str)))) for row in rows]
    hits = {p: sum((1 for words in named if _roots(p.split()[-1]) & words)) for p in parts}
    return max(parts, key=lambda p: (hits[p], -parts.index(p)))

def ridge_item_ids(agent, number: int) -> list[str]:
    cat_key = agent.category_key()
    item = agent.items[number - 1]
    return [i for i in agent.seen_order if cat_key and agent.owns(item, str((agent.rows.get(i) or {}).get(cat_key) or ''))]

def ridge_item_groups(agent) -> list[list[str]]:
    cat_key = agent.category_key()
    groups: dict[str, list[str]] = {}
    for ident in agent.seen_order:
        label = str(agent.rows[ident].get(cat_key) or '') if cat_key else ''
        groups.setdefault(label, []).append(ident)
    wanted = [(_words(item.shelf or ''), _words(item.name), {w for w in _stem(item.shelf or '') | _stem(item.name) if len(w) >= 4}) for item in agent.items]
    matched = [ids for label, ids in groups.items() if any((_words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long for shelf, name, long in wanted)) or agent.owner_of(label)]
    pool = matched or list(groups.values())
    return sorted(pool, key=lambda ids: -len(ids))[:max(1, len(agent.items))]

def ridge_is_refuse(agent, name: str, args: dict[str, Any]) -> bool:
    if name != agent.roles.get('refuse'):
        return False
    fixed = agent.brief.get('refuse_args') if isinstance(agent.brief.get('refuse_args'), dict) else {}
    return all((args.get(k) == v for k, v in fixed.items()))

def ridge_is_buy(agent, name: str, args: dict[str, Any]) -> bool:
    if name != agent.roles.get('buy'):
        return False
    fixed = agent.brief.get('buy_args') if isinstance(agent.brief.get('buy_args'), dict) else {}
    return all((args.get(k) == v for k, v in fixed.items()))

def ridge_interview(agent) -> str:
    agent.asked = True
    if agent.msgs_free() <= 0:
        return ''
    agent.widen()
    count = agent.msg_limit - agent.msg_used
    agent.lanes = agent.plan_lanes(max(count, agent.list_cap()) if agent.listing else count)
    agent.lane_found = {n: 0 for n in agent.lanes}
    agent.lane_spent = {n: 0 for n in agent.lanes}
    agent.lane_used = {n: set() for n in agent.lanes}
    agent.follow_up = []
    sent: list[tuple[int, str, str]] = []
    told: list[str] = []
    groups = agent.groups_for(agent.msgs_free()) if agent.listing else []
    while agent.msgs_free() > 0 and (not agent.finished) and (agent.left() > 6) and (agent.clock() < QUESTION_BY):
        batch = []
        while groups and agent.listing and (not batch):
            batch = agent.ask_group(groups.pop(0), told)
        batch = batch or agent.ask_next(told)
        if not batch:
            break
        sent += batch
    found_items = [n for n in agent.lane_found if agent.lane_found[n] > 0]
    chain = agent.early_trail()
    searched = agent.value_search(found_items) if found_items else ''
    if told:
        agent.version += 1
    lines = [f'Q: {q}\nA: {a}' for _, q, a in sent]
    summary = "Ridge's interviewer asked the shopper (each question used one step):\n" + '\n'.join(lines)
    summary += '\nNew firm requirements from these answers: ' + ('; '.join(told) if told else 'none')
    summary += '. Apply them to every pick (search again with the new values where needed).'
    pointed = [need for _, need in agent.refs if any((need in t for t in told))]
    if pointed:
        summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.'
    if chain:
        summary += '\n' + chain
    if searched:
        summary += '\n' + searched
    if agent.queued:
        summary += '\n[notes seen meanwhile] ' + ' | '.join(agent.queued)
        agent.queued = []
    return summary

def ridge_ident(agent, node: Any, in_list: bool=False, depth: int=0) -> str | None:
    if not isinstance(node, dict) or len(node) < 3:
        return None
    if agent.id_key and _id_like(node.get(agent.id_key), True):
        return node[agent.id_key].strip()
    nested = any((isinstance(v, dict) for v in node.values()))
    if not (in_list or depth == 0 or nested):
        return None
    found = _learn_id([node])
    if found and ('id' in str(found).lower() or node[found].strip() in agent.rows):
        return node[found].strip()
    return None

def ridge_hunt_words(agent, number: int) -> tuple[str, list[str]]:
    cat_key = agent.category_key()
    item = agent.items[number - 1]
    title = agent.title_key()
    labels = [str(row.get(cat_key) or '') for row in agent.rows.values() if agent.owns(item, str(row.get(cat_key) or ''))] if cat_key else []
    names = [str(row.get(title) or '') for row in agent.rows.values() if title and cat_key and agent.owns(item, str(row.get(cat_key) or ''))]
    for record in agent.full.values():
        if cat_key and agent.owns(item, str(record.get(cat_key) or '')):
            names.append(' '.join((str(v) for v in (_fact_block(record) or {}).values() if isinstance(v, str) and len(v) <= HUNT_VALUE_CHARS)))
    others = {w for n, other in enumerate(agent.items, 1) if n != number for w in re.findall('[a-z]{4,}', ((other.shelf or '') + ' ' + other.name).lower())}
    label = max(labels, key=labels.count).lower() if labels else (item.shelf or item.name).lower()
    nouns = [w for w in re.findall('[a-z]{4,}', label) if w not in others]
    noun = '' if number in agent.hunt_no_noun else max(nouns, key=len) if nouns else ''
    if not names:
        return (noun, [])
    counts: dict[str, int] = {}
    for name in names:
        for word in {w.lower() for w in re.findall("[A-Za-z][A-Za-z'-]{2,}", name)}:
            counts[word] = counts.get(word, 0) + 1
    own = _roots(item.shelf or '') | _roots(item.name)
    ceiling = max(2, int(len(names) * HUNT_COMMON))
    words = [w for w, n in counts.items() if 2 <= n <= ceiling and w != noun and _roots(w) and (not _roots(w) & own)]
    words.sort(key=lambda w: (-counts[w], w))
    return (noun, words)

def ridge_hunt_item(agent, number: int, tool: str, base: dict[str, Any], field: str, room: int) -> tuple[int, int]:
    tried = agent.hunt_tried.setdefault(number, set())
    noun, _ = agent.hunt_words(number)
    values = [value for value, _ in agent.value_words(number)]
    spent = gained = idle = 0
    issued = refused = False

    def run(query: str) -> bool:
        nonlocal spent, gained, idle, issued, refused
        tried.add(query.lower())
        faults: list[str] = []
        args = _shape({**base, field: query}, agent.schemas.get(tool), 'args', faults)
        refused = refused or faults
        if faults or spent >= room or agent.finished:
            return False
        issued = True
        holds_values = values and all((v.lower() in query.lower() for v in values))
        found = False
        walked: set[str] = set()
        while not agent.finished and agent.clock() <= HUNT_BY + HUNT_GRACE:
            mark = agent.call_mark(tool, args)
            if mark in walked:
                return found
            walked.add(mark)
            if mark in agent.read_calls:
                seen = agent.read_calls[mark]
                new = HUNT_PAGE_NEW
            elif spent >= room:
                return found
            else:
                before = agent.rows_of_item(number)
                reply = agent.act(tool, args)
                spent += 1
                if reply is None:
                    return found
                seen = reply.get('observation')
                agent.observe(tool, seen)
                new = agent.rows_of_item(number) - before
                gained += new
                idle = 0 if new else idle + 1
            rows = agent.records_in(seen)
            found = found or rows
            if holds_values:
                agent.value_rows.update((i for i, _ in rows))
            page = agent.paging(tool, seen)
            if not page or page[3] is None or (new < HUNT_PAGE_NEW and page[0] > HUNT_WHOLE):
                return found
            faults = []
            args = _shape({**args, page[2]: page[3]}, agent.schemas.get(tool), 'args', faults)
            if faults:
                return found
        return found
    mark = (number, tuple(values))
    stem = agent.hunt_stem.get(mark)
    if stem is None:
        tries = ([' '.join(values + [noun]).strip(), ' '.join(values)] if values else []) + ([' '.join(values[:-1] + [noun]).strip()] if len(values) >= 2 else []) + ([noun] if noun else [])
        stem = ''
        for query in dict.fromkeys(tries):
            issued = False
            if run(query):
                stem = query
                break
            if not issued:
                return (spent, gained)
            if query == noun:
                agent.hunt_no_noun.add(number)
                noun = ''
        agent.hunt_stem[mark] = stem
    while spent < room and idle < HUNT_IDLE and (not refused) and (not agent.finished) and (agent.clock() <= HUNT_BY + HUNT_GRACE):
        _, words = agent.hunt_words(number)
        words = [w for w in words if f'{w} {stem}'.strip().lower() not in tried and w not in stem.lower().split()]
        if not words:
            break
        run(f'{words[0]} {stem}'.strip())
    return (spent, gained)

def ridge_home_label(agent, number: int) -> str:
    cat_key = agent.category_key()
    if not cat_key:
        return ''
    item = agent.items[number - 1]
    for pool in (agent.full, agent.rows):
        counts: dict[str, int] = {}
        for record in pool.values():
            label = str(record.get(cat_key) or '')
            if label and agent.owns(item, label):
                counts[_words(label)] = counts.get(_words(label), 0) + 1
        for name in (item.shelf, item.name):
            if _words(name) in counts:
                return _words(name)
        if counts:
            return max(counts, key=lambda label: counts[label])
    return ''

def ridge_held_units(agent, ident: str) -> int:
    if agent.cart_view is not None and agent.cart_seen_at >= agent.cart_at:
        stack: list[Any] = [agent.cart_view]
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
    return agent.cart_lines.get(ident, 1)

def ridge_hear(agent, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(HEAVY, LIGHT)) -> dict[str, Any]:
    if agent.quiet_reply(sent):
        agent.trace.append({'kind': 'quiet_reply', 'a': sent[0][2][:80]})
        return {}
    agent.hear_reads += 1
    prompt = HEAR + '\n\nITEMS:\n' + '\n'.join((f'item {n}: {it.name}' for n, it in enumerate(agent.items, 1)))
    prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f"{i}. ({(f'about item {n}' if n else 'about several items')}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1)))
    return agent.ask_json(prompt, list(models), 2500) or {}

def ridge_groups_for(agent, free: int) -> list[list[int]]:
    numbers = [n for n in agent.plan_order if n in agent.lanes] + [n for n in agent.lanes if n not in agent.plan_order]
    numbers = [n for n in numbers if agent.lanes.get(n)]
    if free <= 0 or len(numbers) <= free:
        return []
    out: list[list[int]] = []
    start = 0
    for slot in range(free):
        size = (len(numbers) - start + (free - slot) - 1) // (free - slot)
        out.append(numbers[start:start + size])
        start = start + size
    return out

def ridge_gate_due(agent, stage: str) -> bool:
    return any((p['when'] == stage for p in agent.process)) and (stage, agent.cart_mark() + f'#{agent.version}') not in agent.gate_ok

def ridge_gate(agent, stage: str) -> tuple[bool, str]:
    steps = [p for p in agent.process if p['when'] == stage]
    if not steps or agent.finished or (not agent.gate_due(stage)):
        return (True, '')
    ending = {agent.roles.get('buy'), agent.roles.get('refuse')} - {None}
    tools = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_dump(t['function'].get('parameters') or {})}" for t in agent.tools if t['function']['name'] not in ending]
    said: list[str] = []
    nudged = False
    for _ in range(GATE_ROUNDS):
        prompt = GATE.replace('STAGE', 'place the order' if stage == 'order' else 'end without a purchase') + '\n\nREQUEST:\n' + agent.request + '\n\nSTEPS DUE NOW:\n' + '\n'.join((f"- {p['step']}" for p in steps)) + '\n\nLEDGER:\n' + agent.ledger() + '\n\nCART NOW: ' + agent.cart_text() + '\n\nRECORD (messages and step calls so far):\n' + ('\n'.join((f'{i}. {line}' for i, line in enumerate(agent.done_log, 1))) or '(nothing yet)') + ('\n\nSTORE NOTES SEEN:\n' + '\n'.join(agent.notes[-4:]) if agent.notes else '') + f"\n\nMESSAGES LEFT: {max(0, agent.msg_limit - agent.msg_used)}   STEPS LEFT: {agent.left()} (the {('order' if stage == 'order' else 'ending')} itself takes one)" + '\n\nTOOLS:\n' + '\n'.join(tools)
        got = agent.ask_json(prompt, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 3500)
        if got is None:
            break
        told: list[str] = []
        hold = ''
        if said:
            agent.take_heard({'needs': got.get('needs')}, told)
            hold = str(got.get('hold') or '').strip()
        call = got.get('call') if isinstance(got.get('call'), dict) else None
        agent.trace.append({'kind': 'gate', 'stage': stage, 'call': _clip(call, 200), 'steps': _clip(got.get('steps'), 120), 'hold': hold[:200], 'told': told})
        if told or hold:
            agent.version += 1
            return (False, ('Before going on, Ridge did the steps the request asks for:\n' + '\n'.join(said) + '\n' if said else '') + ("The shopper's reply changes the task: " + '; '.join(told + ([hold] if hold else [])) + '. Fix the cart for it, then check out again.'))
        name = str((call or {}).get('tool') or '')
        shown = [e for e in _items(got.get('steps')) if isinstance(e, dict)]
        talk = stage == 'order' and any((p['messages'] > 0 for p in steps))
        lines = range(1, len(agent.done_log) + 1)
        proof = {i for i in lines if not talk or (i <= len(agent.done_steps) and agent.done_steps[i - 1] > agent.cart_at)}
        unproven = [e for e in shown if e.get('done') is not True or not any(((_count(n) or 0) in proof for n in _items(e.get('lines'))))]
        if not call and unproven and (not said) and (not nudged) and agent.roles.get('ask') and (agent.msg_used < agent.msg_limit) and (agent.left() >= 2):
            nudged = True
            agent.trace.append({'kind': 'gate_unproven', 'steps': _clip(unproven, 120)})
            agent.done_log.append(f'(Ridge: no line made since the cart last changed at step {agent.cart_at} shows ' + '; '.join((str(e.get('step') or '') for e in unproven))[:200] + f' done, and {agent.msg_limit - agent.msg_used} message(s) and {agent.left()} steps are left: give the call that does it)')
            agent.done_steps.append(-1)
            continue
        if not call or name not in agent.schemas or name in ending:
            if not call:
                agent.gate_ok.add((stage, agent.cart_mark() + f'#{agent.version}'))
            break
        if agent.left() < 2 or (name == agent.roles.get('ask') and agent.msg_used >= agent.msg_limit):
            break
        faults: list[str] = []
        args = _shape(call.get('args') if isinstance(call.get('args'), dict) else {}, agent.schemas[name], 'args', faults)
        if faults:
            break
        if name == agent.roles.get('ask'):
            text = max(_strings(args), key=len, default='')
            if not text:
                break
            batch = agent.send([(1, '', text)], 1)
            if not batch:
                break
            agent.gate_msgs += 1
            said.append(f'Q: {text}\nA: {batch[0][2]}')
            continue
        reply = agent.act(name, args)
        if reply is None:
            break
        seen = reply.get('observation')
        agent.absorb(seen)
        agent.done_log.append(f'step {agent.step}: {name} {_dump(args)} -> {agent.show(seen)[:300]} (cart then: {agent.cart_mark()})')
        agent.done_steps.append(agent.step)
        said.append(f'{name} {_dump(args)} -> {agent.show(seen)[:400]}')
    return (True, 'Before going on, Ridge did the steps the request asks for:\n' + '\n'.join(said) if said else '')

def ridge_fuller_label(agent, number: int, label: str) -> str:
    cat_key = agent.category_key()
    mine = _roots(agent.items[number - 1].name)
    if len(mine) < 2 or mine <= _roots(label) or (not cat_key) or (agent.items[number - 1].shelf and _roots(label) == _roots(agent.items[number - 1].shelf)):
        return ''
    return next((seen for seen in sorted({str(r.get(cat_key) or '') for r in agent.rows.values()}) if seen and mine <= _roots(seen)), '')

def ridge_fix_counts(agent, wrong: list[tuple[str, int, int]]) -> list[str]:
    name = agent.roles.get('cart') or ''
    schema = agent.schemas.get(name) or {}
    props = schema.get('properties') or {}
    texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
    target = next((k for t, k in agent.id_args if t == name and (k in texts and True)), None) or next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
    count = next((k for k, v in props.items() if isinstance(v, dict) and 'integer' in _kinds(v)), None)
    mode = next((k for k, v in props.items() if isinstance(v, dict) and isinstance(v.get('enum'), list) and (len(v['enum']) >= 2)), None)
    if not name or not target or (not count):
        return []
    calls: list[dict[str, Any]] = []
    for ident, have, want in wrong:
        if agent.cart_sets(name):
            calls.append({target: ident, count: want})
        elif mode:
            options = [str(o) for o in props[mode]['enum']]
            off = next((o for o in options if re.search('remov|delet|drop|clear|take', o, re.I)), options[1])
            on = next((o for o in options if o != off), options[0])
            calls += [{target: ident, mode: off, count: have}, {target: ident, mode: on, count: want}]
        else:
            return []
    if len(calls) + 3 > agent.left():
        return []
    done = []
    for args in calls:
        faults: list[str] = []
        args = _shape(args, schema, 'args', faults)
        if faults:
            break
        reply = agent.act(name, args)
        if reply is None or (agent.finished and True):
            break
        agent.note_cart(name, args)
        agent.cart_said = _dump(reply.get('observation'))
        agent.observe(name, reply.get('observation'))
        done.append(_dump(args) + ' -> ' + agent.show(reply.get('observation'))[:300])
    agent.trace.append({'kind': 'count_fix', 'calls': done})
    return done

def ridge_fit_set(agent, ident: str, index: dict[str, Any]) -> set[str]:
    mark = index['marks'].get(ident) or {}
    found = set(mark.get('works') or [])
    if mark.get('own'):
        found |= index['notes'].get(mark['own'], set())
    return found

def ridge_fit_plan(agent, short: bool=False) -> str:
    if not agent.bound:
        return ''
    index = agent.fit_index()
    lines = []
    for main in sorted({m for _, m in agent.bound}):
        parts = [p for p, m in agent.bound if m == main]
        tally: dict[str, dict[int, int]] = {}
        for p in parts:
            for i in agent.item_ids(p):
                if agent.sold_out(i):
                    continue
                for code in agent.fit_set(i, index):
                    tally.setdefault(code, {}).setdefault(p, 0)
                    tally[code][p] += 1
        for p in parts:
            system = agent.link_system(p, main, index)
            for i in agent.item_ids(p) if system and system[0] == p else []:
                for code in set((index['marks'].get(i) or {}).get('fam') or []) & system[2] if not agent.sold_out(i) else ():
                    tally.setdefault(code, {}).setdefault(p, 0)
                    tally[code][p] += 1
        ranked = sorted(tally.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1].values())))[:6]
        devices = {code: [i for i in agent.item_ids(main) if code in ((index['marks'].get(i) or {}).get('named') or []) or any((_code_hit(code, c) for c in index['marks'].get(i, {}).get('codes') or []))][:2] for code, _ in ranked}
        text = '; '.join((f'{code}: ' + ', '.join((f'{n} {agent.items[p - 1].name}' for p, n in sorted(per.items()))) + (f" (device seen: {', '.join(devices[code])})" if devices[code] else ' (device not seen yet)') for code, per in ranked))
        if text:
            lines.append(f"codes the {agent.items[main - 1].name} candidates' parts name -> {text}")
        sets = agent.cheapest_sets(main, parts, index) if agent.brief.get('pick') == 'cheapest' else ''
        if sets:
            lines.append(sets)
    if not lines:
        return ''
    return ' | '.join(lines) if short else "[fit plan: a device fits a part when the part's text names the device's model code or another name it is sold as, or when the device's text names the series of the part's code; try one search of a code or series with the item's words, and if it finds nothing list the device's category and read the codes on its rows] " + ' | '.join(lines)

def ridge_fit_index(agent) -> dict[str, Any]:
    state = (len(agent.seen_order), len(agent.full), sum((len(v) for v in agent.views.values())))
    if agent.fit_cached and agent.fit_cached[0] == state:
        return agent.fit_cached[1]
    marks: dict[str, dict[str, Any]] = {}
    notes: dict[str, set[str]] = {}
    for ident in agent.seen_order:
        record = agent.full.get(ident) or agent.rows.get(ident) or {}
        text = ' '.join((str(v) for v in record.values() if isinstance(v, str)))
        text += ' ' + '. '.join((f'{k} {v}' for block in record.values() if isinstance(block, dict) for k, v in block.items() if isinstance(v, str)))
        marks[ident] = _fit_marks(text)
        for part, codes in marks[ident]['also']:
            notes.setdefault(part, set()).update(codes)
    agent.fit_cached = (state, {'marks': marks, 'notes': notes})
    return agent.fit_cached[1]

def ridge_fit_faults(agent) -> list[str]:
    if not agent.bound:
        return []
    index = agent.fit_index()
    cart = agent.cart_ids()
    faults = []
    for part, main in agent.bound:
        devices = [i for i in cart if i in agent.item_ids(main)]
        parts = [i for i in cart if i in agent.item_ids(part)]
        if len(devices) != 1 or not parts:
            continue
        linked = agent.link_faults(part, main, devices[0], parts, index)
        if linked:
            faults = faults + linked
            continue
        codes = set(index['marks'].get(devices[0], {}).get('codes') or [])
        if not codes:
            continue
        barred = lambda i: set(index['marks'].get(i, {}).get('bars') or [])
        covers = lambda i: any((_code_hit(p, c) for p in agent.fit_set(i, index) for c in codes)) and (not any((_code_hit(p, c) for p in barred(i) for c in codes)))
        for line in parts:
            fits = agent.fit_set(line, index)
            if fits and (not covers(line)):
                good = [i for i in agent.item_ids(part) if covers(i) and (not agent.sold_out(i)) and (i not in cart)][:4]
                faults.append(f"{line} ({agent.items[part - 1].name}) works with {', '.join(sorted(fits)[:4])}, not with {devices[0]} ({', '.join(sorted(codes))})" + (f"; seen parts of that kind that work with it: {', '.join(good)}" if good else '; no seen part of that kind works with it, so look for one or switch the device: ' + agent.fit_plan(short=True)))
    return faults

def ridge_fit_ask(agent, number: int, labels: list[str], shown: bool) -> tuple[list[str], str]:
    for size in range(len(labels), 0, -1):
        text = agent.ask_text(number, labels[:size], shown)
        args = agent.message_args(text)
        if args and _says(args, text):
            return (labels[:size], text)
    return ([], '')

def ridge_first_look(agent, name: str, args: dict[str, Any], fresh: bool) -> dict[str, Any]:
    key = agent.page_args.get(name)
    if not key or not fresh or (not isinstance(args.get(key), int)):
        return args
    props = (agent.schemas.get(name) or {}).get('properties') or {}
    low = props.get(key, {}).get('minimum') if isinstance(props.get(key, {}).get('minimum'), int) else 1
    return {**args, key: low}

def ridge_finish_ask(agent, number: int, labels: list[str], shown: bool, batch: list[tuple[int, str, str]], told: list[str]) -> None:
    agent.lane_spent[number] = agent.lane_spent.get(number, 0) + 1
    agent.list_asks += 1
    had = [len(item.told) for item in agent.items]
    heard = agent.hear(batch, (LIGHT, HEAVY))
    heard = agent.rehear_if_doubtful(batch, heard)
    if len(labels) > 1:
        heard.pop('free', None)
    agent.take_heard(heard, told)
    new = [need for item, size in zip(agent.items, had) for need in item.told[size:]]
    for n, (item, size) in enumerate(zip(agent.items, had), 1):
        if len(item.told) > size:
            agent.lane_found[n] = agent.lane_found.get(n, 0) + 1
    crowded = any((_count(x) == 1 for x in _items(heard.get('crowded'))))
    agent.settle(number, labels, shown, new, crowded)

def ridge_finish(agent, again: bool=False) -> str:
    if agent.finished or agent.broke or agent.left() <= 0:
        return ''
    buy, refuse = (agent.roles.get('buy'), agent.roles.get('refuse'))
    fresh = agent.cart_view is not None and agent.cart_seen_at >= agent.cart_at
    holding = agent.cart_ids() if fresh else agent.cart_lines
    whole = True
    if holding and len(agent.items) >= 2:
        whole = agent.fill_short_cart() or not agent.can_refuse()
        if whole and len(agent.cart_ids()) >= 2:
            agent.fill_facts(agent.cart_ids(), agent.tail())
            if agent.left() >= 3 and (agent.model_room() > 5 and True):
                agent.show_links(agent.cart_ids(), agent.tail())
        if not whole:
            agent.trace.append({'kind': 'short_cart_refused', 'lines': agent.cart_ids()})
    if agent.finished or agent.left() <= 0:
        return ''
    if whole and buy and (holding or (agent.cart_touched and (not fresh) and (agent.left() >= 2))):
        faults: list[str] = []
        args = _shape(dict(agent.brief.get('buy_args') or {}), agent.schemas.get(buy), 'args', faults)
        if not faults:
            if agent.left() >= 2:
                agent.gate('order')
            if agent.finished or agent.left() <= 0:
                return ''
            reply = agent.act(buy, args)
            agent.trace.append({'kind': 'forced_buy', 'reply': _clip(reply, 200)})
            if agent.finished or agent.left() <= 0:
                return ''
            if reply is not None and again:
                agent.buys += 1
                agent.refused_at = agent.step
                agent.observe(buy, reply.get('observation'))
                return 'the checkout was refused: ' + agent.show(reply.get('observation'))
    if refuse:
        faults = []
        args = _shape(agent.quiet(refuse, dict(agent.brief.get('refuse_args') or {})), agent.schemas.get(refuse), 'args', faults)
        if not faults:
            if agent.left() >= 2:
                agent.gate('end')
            if agent.finished or agent.left() <= 0:
                return ''
            reply = agent.act(refuse, args)
            agent.trace.append({'kind': 'forced_refuse', 'reply': _clip(reply, 200)})
            if reply is not None and (not agent.finished):
                agent.observe(refuse, reply.get('observation'))
                return 'ending without a purchase was refused: ' + agent.show(reply.get('observation'))
    return ''

def ridge_find_sold(agent) -> str:
    ids = [i for i in agent.cart_ids() if i not in agent.gone]
    if any((agent.sold_out(i) for i in ids)):
        return ''
    if len(ids) == 1:
        agent.unavailable.add(ids[0])
        return ''
    return agent.reread_offers() if ids else ''

def ridge_find_link(agent, about: str, words: str) -> list[str]:
    not_names = agent.not_maker_keys()
    known: list[str] = []
    named = {w for ident in agent.shop_rows(about, list(agent.rows)) + agent.opened_shop(about) for k, v in (agent.rows.get(ident) or {}).items() if isinstance(v, str) and len(v.strip()) <= 40 and (k not in not_names) and (v.strip().lower() not in NO_NAME) for w in re.findall('[a-z0-9]{3,}', v.lower())}
    values = {w for w in re.findall('[a-z0-9]+', (words or about).lower()) if len(w) >= 3 or w.isdigit()} - named - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for', 'of'}
    for ident in agent.opened_shop(about):
        stack, leaves = ([agent.full.get(ident) or agent.rows.get(ident) or {}], [])
        while stack:
            node = stack.pop()
            if isinstance(node, (dict, list)):
                stack += list(node.values()) if isinstance(node, dict) else node
            else:
                leaves.append(str(node))
        text = ' '.join(leaves).lower()
        if values and all((re.search('(?<![a-z0-9])' + re.escape(v) + '(?![a-z0-9])', text) for v in values)):
            agent.trace.append({'kind': 'link_known', 'id': ident})
            return []
    plan = agent.value_template()
    memo = ' '.join(words.lower().split())
    if memo and memo in agent.link_seen:
        known = agent.shop_rows(about, agent.link_seen[memo])
    elif plan and words:
        tool, base, field = plan
        for query in dict.fromkeys((words, ' '.join(words.split()[:2]))):
            faults: list[str] = []
            args = _shape({**base, field: query}, agent.schemas.get(tool), 'args', faults)
            if faults:
                break
            agent.link_searches += 1
            reply = agent.act(tool, args)
            if reply is None:
                break
            observation = reply.get('observation')
            agent.observe(tool, observation)
            agent.link_seen[memo] = [i for i, _ in agent.records_in(observation)]
            known = agent.shop_rows(about, agent.link_seen[memo])
            agent.trace.append({'kind': 'link_search', 'words': query[:80], 'rows': known[:4]})
            if agent.link_seen[memo]:
                break
    kind = {w for w in re.findall('[a-z0-9]{3,}', (words or about).lower())} - named - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for'}

    def of_kind(ids: list[str]) -> list[str]:
        return [i for i in ids if not kind or kind & set(re.findall('[a-z0-9]{3,}', _dump(agent.rows.get(i) or {}).lower()))]
    return (of_kind(known) or of_kind(agent.shop_rows(about, list(reversed(agent.seen_order)))))[:2]

def ridge_fill_short_cart(agent) -> bool:
    cart = agent.roles.get('cart') or ''
    bound = {n for pair in agent.bound for n in pair}
    for number in agent.missing_items():
        item = agent.items[number - 1]
        fits = [i for i in agent.item_ids(number) if i in agent.full and i not in agent.cart_lines and agent.buyable(i) and (agent.price_of(i) is not None) and agent.states_needs(i, number)]
        fits.sort(key=lambda i: agent.price_of(i) or 0.0)
        pick = fits[0] if fits and number not in bound else None
        if pick and isinstance(agent.budget, (int, float)):
            held = sum(((agent.price_of(i) or 0.0) * (agent.held_units(i) or 0) for i in agent.cart_ids()))
            cost = (agent.price_of(pick) or 0.0) * (1 if agent.brief.get('budget_scope') == 'each' else item.qty)
            if cost + (0.0 if agent.brief.get('budget_scope') == 'each' else held) > agent.budget + 1e-09:
                pick = None
        calls = agent.cart_calls({**{i: agent.cart_lines[i] for i in agent.cart_ids() if i in agent.cart_lines}, pick: item.qty}, []) if pick else None
        if not calls or len(calls) + 1 > agent.left():
            return False
        for args in calls:
            faults: list[str] = []
            args = _shape(args, agent.schemas.get(cart), 'args', faults)
            if faults or agent.act(cart, args) is None:
                return False
            agent.note_cart(cart, args)
        agent.trace.append({'kind': 'filled_line', 'item': number, 'line': pick})
    return True

def ridge_fill_facts(agent, ids: list[str], keep: int) -> list[str]:
    tool = agent.facts_tool()
    target = agent.single_target(tool)
    if (not tool or not target) and (agent.compare_facts() is not True and True):
        return []
    texts = []
    wanted = [i for i in ids if i in agent.rows and i not in agent.full and (not agent.sold_out(i))]
    side = agent.roles.get('compare')
    many = agent.compare_target()
    unread: list[str] = []
    while side and many and (agent.compare_facts() is True) and (len(wanted) >= 2) and (agent.left() > keep) and (not agent.finished) and (not agent.rushed()):
        room = agent.compare_room()
        batch, wanted = (wanted[:room], wanted[room:])
        faults_side: list[str] = []
        args_side = _shape({many: batch}, agent.schemas.get(side), 'args', faults_side)
        if faults_side:
            wanted = batch + wanted
            break
        reply = agent.act(side, args_side)
        if reply is None:
            wanted = batch + wanted
            break
        seen = reply.get('observation')
        agent.observe(side, seen)
        agent.note_view(side, seen)
        rows = agent.records_in(seen)
        for ident, record in rows:
            texts.append(f'{ident}: ' + agent.show(record)[:1500])
        if rows:
            agent.compare_ok = max(agent.compare_ok, len(rows))
            unread += [i for i in batch if i not in agent.full]
            continue
        agent.compare_bad = min(agent.compare_bad or 999, len(batch))
        wanted = batch + wanted
        if agent.compare_room() >= len(batch):
            break
    if not tool or not target:
        return texts
    for ident in unread + wanted:
        if agent.left() <= keep or agent.finished or agent.facts_from.get(tool) is False or agent.rushed():
            break
        if (tool, ident) in agent.factless:
            continue
        faults: list[str] = []
        args = _shape({target: ident}, agent.schemas.get(tool), 'args', faults)
        if faults:
            break
        reply = agent.act(tool, args)
        if reply is None:
            break
        seen = reply.get('observation')
        agent.observe(tool, seen)
        agent.note_view(tool, seen)
        if isinstance(seen, dict) and agent.ident(seen):
            agent.note_facts(tool, _fact_block(seen))
        texts.append(f'{ident}: ' + agent.show(seen)[:1500])
    if texts:
        agent.trace.append({'kind': 'fill_facts', 'n': len(texts)})
    return texts

def ridge_facts_tool(agent) -> str | None:
    known = next((t for t, ok in agent.facts_from.items() if ok and agent.single_target(t)), None)
    if known:
        return known
    tool = agent.roles.get('open')
    return tool if agent.facts_from.get(tool or '') is not False and agent.single_target(tool) else None

def ridge_evidence(agent, observation: Any) -> str:
    ids: list[str] = []

    def walk(node: Any, depth: int) -> None:
        if depth > 6:
            return
        if isinstance(node, dict):
            ident = agent.ident(node, False, depth)
            if ident and _fact_block(node) and (ident not in ids):
                ids.append(ident)
                return
            for value in node.values():
                walk(value, depth + 1)
        elif isinstance(node, list):
            for value in node:
                walk(value, depth + 1)
    walk(observation, 0)
    needs = agent.needs_now()
    if not ids or not needs:
        return ''
    many = len(agent.items) > 1
    lines = []
    for ident in ids[:12]:
        block = _fact_block(agent.full.get(ident) or {}) or {}
        parts = []
        for number, need in needs[:8]:
            prop = need.split(':', 1)[0].strip()
            hits = [f'{k}={str(v)[:80]}' for k, v in block.items() if _names(_stem(prop), _stem(k))]
            tag = f'(item {number}) ' if many else ''
            if hits:
                parts.append(f'{tag}{prop} -> ' + ', '.join(hits[:3]))
        if parts:
            lines.append(f'{ident}: ' + '; '.join(parts))
    if not lines:
        return ''
    return "[evidence: keys whose names contain a requirement's property; one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property] " + ' | '.join(lines)

def ridge_early_trail(agent) -> str:
    live = {need for _, need in agent.needs_now()}
    if agent.left() < TRAIL_ROOM or agent.clock() > HUNT_BY or (not any((need in live and need not in agent.ref_values for _, need in agent.refs))):
        return ''
    kept = (agent.trails, agent.trail_blocks)
    agent.trails, agent.trail_room = (0, EARLY_TRAILS)
    try:
        text, _ = agent.trail(HUNT_BY)
    finally:
        (agent.trails, agent.trail_blocks), agent.trail_room = (kept, None)
    agent.trace.append({'kind': 'early_trail', 'left': agent.left(), 'values': list(agent.ref_values.values())[:3]})
    return text

def ridge_doubtful(agent, entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    prop, _, value = str(entry.get('need') or ('' and True)).partition(':')
    words = set(re.findall('[a-z0-9]+', value.lower()))
    if entry.get('points') is True:
        return True
    if len(words) < 2:
        return False
    seen: set[str] = set()
    for record in agent.full.values():
        for key, shown in (_fact_block(record) or {}).items():
            if _names(_stem(prop), _stem(key)) or _names(_stem(key), _stem(prop)):
                seen |= set(re.findall('[a-z0-9]+', str(shown).lower()))
    return words - seen

def ridge_described(agent, number: int, need: str) -> bool:
    if need in [m for _, m in agent.refs] or need in [str(x) for x in _items(agent.brief.get('lookups'))]:
        return False
    prop, _, value = need.partition(':')
    if not prop.strip() or len(re.findall('[a-z0-9]+', value.lower())) < 4:
        return False
    if re.search('\\b(?:listing|page|shown|other than|unlike|same as)\\b', value, re.I):
        return False
    item = agent.items[number - 1]
    if any(({w for w in _stem(other.name) if len(w) >= 4} & _stem(value) for other in agent.items if other is not item)):
        return False
    cat_key = agent.category_key()
    keys: set[str] = set()
    read = 0
    for record in agent.full.values():
        if cat_key and (not agent.owns(item, str(record.get(cat_key) or ''))):
            continue
        block = _fact_block(record) or {}
        if block:
            read += 1
            keys |= {_words(k) for k in block}
    return read >= 3 and (not any((_stem(k) & _stem(prop) for k in keys)))

def ridge_count_fixes(agent) -> list[tuple[str, int, int]]:
    cat_key = agent.category_key()
    if not cat_key:
        return []
    out = []
    for ident in agent.cart_ids():
        have = agent.held_units(ident) or 0
        owners = [n for n, item in enumerate(agent.items, 1) if agent.owns(item, str((agent.rows.get(ident) or {}).get(cat_key) or ''))]
        mine = [i for i in agent.cart_ids() if (agent.cart_lines.get(i, 1) or 0) > 0 and len(owners) == 1 and (agent.owns(agent.items[owners[0] - 1], str((agent.rows.get(i) or {}).get(cat_key) or '')) and True)]
        if have > 0 and len(owners) == 1 and (len(mine) == 1) and (have != agent.items[owners[0] - 1].qty) and (not agent.sold_out(ident)):
            out.append((ident, have, agent.items[owners[0] - 1].qty))
    return out

def ridge_complete_cart(agent) -> str:
    cart, buy = (agent.roles.get('cart'), agent.roles.get('buy'))
    if agent.completed or not agent.confirmed or agent.sold_out(agent.confirmed) or (not cart) or (not buy):
        return ''
    agent.completed = True
    cat_key = agent.category_key()

    def kind(ident: str) -> str:
        return str((agent.rows.get(ident) or {}).get(cat_key) or '').strip().lower() if cat_key else ident
    plan: dict[str, int] = {}
    for number, ids in agent.lead_items.items():
        found = re.search('\\d+', str(number))
        index = int(found.group()) if found else 1 if len(agent.items) == 1 else None
        pick = next((i for i in ids if not agent.sold_out(i)), None)
        if pick and index and (1 <= index <= len(agent.items)) and (kind(pick) not in {kind(p) for p in plan}):
            plan[pick] = agent.items[index - 1].qty
    if agent.confirmed not in plan:
        return ''
    now = agent.cart_ids()
    keep = {i: agent.cart_lines.get(i, 1) for i in now if i not in plan and (not agent.sold_out(i)) and (kind(i) not in {kind(p) for p in plan})}
    drop = [i for i in now if i not in plan and (i not in keep and True)]
    if len(plan) + len({kind(i) for i in keep}) < len(agent.items):
        return ''
    lines = {**keep, **plan}
    calls = agent.cart_calls(lines, drop)
    if not calls or len(calls) + 1 > agent.left():
        return ''
    agent.fill_facts(list(lines), len(calls) + 3)
    agent.show_links(list(lines), len(calls) + 3)
    agent.check_cart(list(lines), SPEND_HARD)
    if not agent.verdicts.get(agent.check_key(list(lines))):
        return ''
    done = []
    for args in calls:
        faults: list[str] = []
        args = _shape(args, agent.schemas.get(cart), 'args', faults)
        if faults:
            return ''
        reply = agent.act(cart, args)
        if reply is None or agent.finished:
            return ''
        agent.note_cart(cart, args)
        agent.observe(cart, reply.get('observation'))
        done.append(_dump(args))
    faults = []
    args = _shape(dict(agent.brief.get('buy_args') or {}), agent.schemas.get(buy), 'args', faults)
    if faults:
        return ''
    replies = []
    for attempt in range(2):
        if attempt:
            agent.reread_offers()
            agent.check_cart(list(lines), SPEND_HARD)
            if not agent.verdicts.get(agent.check_key(list(lines))) or agent.left() < 1 or agent.finished or (any((agent.sold_out(i) for i in lines)) and True):
                break
        go, said = agent.gate('order')
        if said:
            replies.append(said)
        if not go:
            break
        reply = agent.act(buy, args)
        agent.trace.append({'kind': 'completed', 'lines': lines, 'drop': drop, 'reply': _clip(reply, 200)})
        if reply is None:
            break
        agent.buys += 1
        if not reply.get('finished'):
            agent.refused_at = agent.step
        agent.observe(buy, reply.get('observation'))
        replies.append(agent.show(reply.get('observation')))
        if agent.finished or reply.get('finished'):
            break
    if not replies:
        return ''
    return 'Not run (Ridge acted instead): the independent check confirmed ' + agent.confirmed + ', so Ridge set the cart (' + '; '.join(done) + ') and checked out. Store reply: ' + ' | then: '.join(replies)

def ridge_compare_target(agent) -> str | None:
    schema = agent.schemas.get(agent.roles.get('compare') or '') or {}
    props = schema.get('properties') or {}
    target = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in _kinds(v)), None)
    if not target or any((k != target for k in schema.get('required') or [])):
        return None
    return target

def ridge_compare_room(agent) -> int:
    spec = ((agent.schemas.get(agent.roles.get('compare') or '') or {}).get('properties') or {}).get(agent.compare_target() or '') or {}
    if isinstance(spec.get('maxItems'), int):
        return spec['maxItems']
    if agent.compare_bad:
        return max(2, min(agent.compare_bad - 1, max(agent.compare_ok, 2)))
    return max(4, agent.compare_ok)

def ridge_compare_facts(agent) -> bool | None:
    return agent.facts_from.get(agent.roles.get('compare') or '')

def ridge_compare_bare(agent) -> bool:
    seen = agent.compare_facts()
    return seen is False or (seen is None and (agent.facts_from.get(agent.facts_tool() or '') is True or agent.roles.get('offer')))

def ridge_check_key(agent, ids: list[str]) -> str:
    return '|'.join(sorted(ids)) + f'#{agent.version}'

def ridge_check_cart(agent, ids: list[str] | None=None, limit: float=SPEND_SOFT) -> str:
    lead = ids is not None
    ids = ids if lead else agent.cart_ids()
    key = agent.check_key(ids)
    if key in agent.checked or not agent.judges(limit) or (not ids):
        return ''
    agent.checked.add(key)
    prompt = CHECK + '\n\nREQUEST:\n' + agent.request + '\n\nREQUIREMENTS IN FORCE:\n' + agent.ledger()
    if lead:
        quantity = '; '.join((f'item {n}: quantity {it.qty}' for n, it in enumerate(agent.items, 1)))
        prompt += '\n\nCART (planned, judge these listings as if they were carted at the requested quantities: ' + quantity + '):\n' + ', '.join(ids)
    else:
        prompt += '\n\nCART:\n' + agent.cart_text()
    prompt = prompt + ("\n\nRECORDS OF THE CART'S LISTINGS:\n" + '\n'.join((f'{i}: ' + ('SOLD OUT (the store says it can no longer be bought). ' if agent.sold_out(i) else '') + agent.record_of(i) for i in ids)))
    if agent.brief.get('pick') in ('cheapest', 'most', 'least'):
        prompt += '\n\nOTHER LISTINGS SEEN (newest first):\n' + agent.others(ids, 15)
    judges = agent.judges(limit)
    got = None
    if len(judges) > 1 and (not lead) and (not agent.audited) and (agent.brief.get('pick') not in ('cheapest', 'most', 'least')):
        got = agent.ask_json(prompt, [LIGHT], 3000)
        if got:
            judges = [HEAVY] if got.get('ok') is False and HEAVY in agent.judges(limit) else []
    if judges and (not agent.no_credit):
        got = agent.ask_json(prompt, judges, 3000) or got
    agent.trace.append({'kind': 'check', 'cart': ids, 'got': got})
    agent.verdicts[key] = got and got.get('ok') is not False
    if not got or got.get('ok') is not False:
        return ''
    agent.leads = [i for i in agent.leads if i not in ids]
    problems = [str(p) for p in _items(got.get('problems'))][:5]
    better = [str(b) for b in _items(got.get('better')) if str(b) in agent.rows and (not agent.sold_out(str(b)))][:4]
    text = 'A careful review says this cart is not right yet: ' + '; '.join(problems or ['see the requirements'])
    if better:
        text += '. Listings that look better: ' + ', '.join(better)
    return text + '. Fix the cart (or show the review is wrong by re-reading the records), then check out.'

def ridge_cheapest_here(agent, observations: list[Any]) -> str:
    if agent.brief.get('pick') != 'cheapest' or not agent.price_key:
        return ''
    found = {i for seen in observations for i, _ in agent.records_in(seen)}
    rows = [(agent.price_of(i), i) for i in found if agent.buyable(i) and agent.price_of(i) is not None]
    if len(rows) < 2:
        return ''
    return '\n[Ridge] buyable rows here, cheapest first: ' + ', '.join((f'{i} ({p:g})' for p, i in sorted(rows)[:6]))

def ridge_cheaper_rows(agent) -> list[str]:
    if agent.brief.get('pick') != 'cheapest':
        return []
    mark = agent.cart_mark() + f'#{agent.version}'
    cat_key = agent.category_key()
    if mark in agent.cheaper_marks or not cat_key:
        return []
    agent.cheaper_marks.add(mark)
    cart = agent.cart_ids()
    agent.list_below(cart)
    picks: list[str] = []
    room = agent.compare_room() if agent.compare_facts() is True else 0
    for line in cart:
        price = agent.price_of(line)
        kind = str((agent.rows.get(line) or {}).get(cat_key) or ('' and True)).strip().lower()
        if price is None or not kind:
            continue
        pool = [i for i in agent.seen_order if i not in cart and i not in agent.full and agent.buyable(i) and (str((agent.rows.get(i) or {}).get(cat_key) or '').strip().lower() == kind) and ((agent.price_of(i) if agent.price_of(i) is not None else float('inf')) < price)]
        pool.sort(key=lambda i: agent.price_of(i) or 0.0)
        lowest = [i for i in pool if i in agent.value_rows][:1]
        pool = lowest + [i for i in pool if i not in lowest]
        picks += [i for i in pool[:room or 2] if i not in picks]
    return agent.fill_facts(picks[:room * len(cart) if room else min(6, CHEAPER_OPENS * max(1, len(cart)))], agent.tail() + 6)

def ridge_category_key(agent) -> str | None:
    counts: dict[str, set[str]] = {}
    present: dict[str, int] = {}
    for row in agent.rows.values():
        for key, value in row.items():
            if isinstance(value, str) and 0 < len(value) <= 40 and (not re.search('\\d{3}', value)):
                counts.setdefault(key, set()).add(value)
                present[key] = present.get(key, 0) + 1
    total = max(1, len(agent.rows))
    known = agent.labels()
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

def ridge_cart_text(agent) -> str:
    if agent.cart_view is not None and agent.cart_seen_at >= agent.cart_at:
        return _dump(_clip(agent.cart_view, 300))[:3000]
    return "lines by Ridge's record of the cart calls (listing: units): " + _dump(agent.cart_lines)

def ridge_cart_sets(agent, name: str) -> bool:
    props = agent.schemas.get(name, {}).get('properties') or {}
    if any((isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) >= 2) for s in props.values())):
        return False
    return any((isinstance(s, dict) and 'integer' in _kinds(s) and (s.get('minimum') == 0) for s in props.values()))

def ridge_cart_mark(agent) -> str:
    return _dump(sorted(agent.cart_lines.items()))

def ridge_cart_ids(agent) -> list[str]:
    if agent.cart_view is not None and agent.cart_seen_at >= agent.cart_at:
        text = _dump(agent.cart_view)
        return [i for i in agent.rows if i in text]
    return list(agent.cart_lines)

def ridge_cart_calls(agent, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
    name = agent.roles.get('cart') or ''
    schema = agent.schemas.get(name) or {}
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
    target = next((k for t, k in agent.id_args if t == name and k in texts), None) or next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
    count = next((k for k, s in props.items() if isinstance(s, dict) and 'integer' in _kinds(s)), None)
    mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) >= 2)), None)
    if not target:
        return None
    calls: list[dict[str, Any]] = []
    for ident in drop:
        have = agent.cart_lines.get(ident, 1)
        if mode:
            calls.append({target: ident, mode: props[mode]['enum'][1], **({count: have} if count else {})})
        elif count and agent.cart_sets(name):
            calls.append({target: ident, count: 0})
        elif count and isinstance(props[count].get('minimum'), int) and (props[count]['minimum'] <= -have):
            calls.append({target: ident, count: -have})
        else:
            return None
    for ident, want in lines.items():
        have = agent.cart_lines.get(ident, 0) if ident in agent.cart_ids() else 0
        if have == want:
            continue
        if have and (not agent.cart_sets(name)):
            return None
        calls.append({target: ident, **({mode: props[mode]['enum'][0]} if mode else {}), **({count: want} if count else {})})
    return calls

def ridge_cap_template(agent) -> tuple[str, dict[str, Any], str, str] | None:
    for tool, schema in agent.schemas.items():
        props = (schema or {}).get('properties') or ({} and True)
        caps = agent.cap_keys(tool)
        shelves = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v) and re.search(SHELF_ARG, _snake(k), re.I)]
        if len(caps) != 1 or len(shelves) != 1:
            continue
        learned = [(dict(mode), held) for (key, mode), held in agent.cap_modes(tool).items() if key == caps[0]]
        options_args = [k for k, v in props.items() if isinstance(v, dict) and isinstance(v.get('enum'), list) and (len(v['enum']) >= 2)]
        single = len(options_args) == 1 and options_args[0] in ((schema or {}).get('required') or [])
        base: dict[str, Any] = {}
        for key, spec in props.items():
            options = spec.get('enum') if isinstance(spec, dict) else None
            if not isinstance(options, list) or len(options) < 2:
                continue
            pick = next((mode[key] for mode, held in learned if held and key in mode), None)
            if pick is None:
                pick = next((a[key] for a in reversed(agent.searched) if shelves[0] in a and key in a and (not any((k != shelves[0] and (not (isinstance(props.get(k), dict) and 'enum' in props[k])) and isinstance(v, str) and v.strip() and re.search(QUERY_WORDS, _snake(k), re.I) for k, v in a.items())))), None)
                named = [o for o in options if isinstance(o, str) and re.search(LIST_MODE, o, re.I)]
                worded = {a[key] for a in agent.searched if key in a and any((isinstance(v, str) and v.strip() and re.search(QUERY_WORDS, _snake(k), re.I) for k, v in a.items()))}
                unworded = [o for o in options if o not in worded]
                pick = pick if pick is not None else named[0] if len(named) == 1 else unworded[0] if single and worded and (len(unworded) == 1) else None
            spoiled = {mode[key] for mode, held in learned if not held and key in mode} if single else set()
            if spoiled and (pick is None or pick in spoiled):
                rest = [o for o in options if o not in spoiled]
                pick = rest[0] if len(rest) == 1 else None
            if pick is None:
                return None
            base[key] = pick
        return (tool, base, shelves[0], caps[0])
    return None

def ridge_cap_modes(agent, tool: str) -> dict[tuple[str, tuple[tuple[str, Any], ...]], bool]:
    props = (agent.schemas.get(tool) or {}).get('properties') or {}
    cat_key = agent.category_key()
    out: dict[tuple[str, tuple[tuple[str, Any], ...]], bool] = {}
    prices: dict[str, list[float]] = {}
    for ident, row in agent.rows.items():
        price = agent.price_of(ident)
        if price is not None:
            prices.setdefault(str(row.get(cat_key) or '').strip().lower() if cat_key else '', []).append(price)
    for mark, (name, args) in agent.read_args.items():
        if name != tool:
            continue
        rows = [i for i, _ in agent.records_in(agent.read_calls.get(mark))]
        shown = [agent.price_of(i) for i in rows]
        if not rows or None in shown:
            continue
        mode = tuple(sorted(((k, v) for k, v in args.items() if isinstance(props.get(k), dict) and isinstance(props[k].get('enum'), list) and isinstance(v, (str, int, float)))))
        labels = {str((agent.rows.get(i) or {}).get(cat_key) or '').strip().lower() for i in rows} if cat_key else set()
        for key, value in args.items():
            if key not in props or isinstance(value, bool) or (not isinstance(value, (int, float))):
                continue
            pool = [p for label in labels for p in prices.get(label, [])]
            if max(shown) > value:
                out[key, mode] = False
            elif out.get((key, mode)) is not False and pool and ((sum((p <= value for p in pool)) / len(pool)) ** len(rows) < CAP_CHANCE):
                out[key, mode] = True
    return out

def ridge_cap_keys(agent, tool: str) -> list[str]:
    props = (agent.schemas.get(tool) or {}).get('properties') or {}
    numbers = [k for k, v in props.items() if isinstance(v, dict) and {'integer', 'number'} & set(_kinds(v)) and (not re.search(CAP_NOT, k, re.I)) and (not re.search(CAP_LOW, _snake(k), re.I))]
    held = {key for (key, _), ok in agent.cap_modes(tool).items() if ok}
    named = [k for k in numbers if re.search(CAP_ARG, _snake(k), re.I)]
    priced = [k for k in numbers if re.search(CAP_PRICED, _snake(k), re.I)]
    for group in ([k for k in numbers if k in held], [k for k in named if k in priced], priced, named):
        if group:
            return group
    return []

def ridge_cap_ignored(agent, name: str, args: dict[str, Any], observation: Any) -> str:
    props = (agent.schemas.get(name) or {}).get('properties') or {}

    def priced(key: str) -> bool:
        spec = props.get(key) if isinstance(props.get(key), dict) else {}
        return re.search('price|cost|amount|budget|spend|money', key + ' ' + str(spec.get('description') or ''), re.I)
    limits = agent.cap_keys(name)
    caps = [k for k, v in args.items() if k in props and isinstance(v, (int, float)) and (not isinstance(v, bool)) and ((re.search('max|most|under|below|limit|cap|upto|up_to', k, re.I) and priced(k) or (k in limits[:1] and len(limits) == 1)) and True)]
    if not caps or not agent.price_key:
        return ''
    caps.sort(key=lambda k: args[k])
    cap = args[caps[0]]
    over = [i for i, _ in agent.records_in(observation) if (agent.price_of(i) or 0) > cap]
    if not over:
        return ''
    modes = {k: v for k, v in args.items() if isinstance(v, str) and isinstance((props.get(k) or {}).get('enum'), list)}
    others = {k: [o for o in props[k]['enum'] if o != v] for k, v in modes.items()}
    hint = '; list the category with ' + ', '.join((f'{k}={_dump(o[0])}' for k, o in others.items() if o)) + f' and the same {caps[0]} to see only cheaper listings' if any(others.values()) else ''
    return f'\n[Ridge] This search did not apply {caps[0]}={cap:g}: {len(over)} of its rows cost more{hint}.'

def ridge_can_refuse(agent) -> bool:
    refuse = agent.roles.get('refuse')
    if not refuse:
        return False
    faults: list[str] = []
    _shape(agent.quiet(refuse, dict(agent.brief.get('refuse_args') or {})), agent.schemas.get(refuse), 'args', faults)
    return not faults

def ridge_call_model(agent, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None, effort: str | None=None) -> tuple[dict[str, Any] | None, Any]:
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
    result = _llm_within(agent.model_room()).post_verbose('/inference/chat/completions', body)
    data = result.data
    if isinstance(data, dict) and data.get('choices'):
        usage = data.get('usage') or {}
        if isinstance(usage.get('cost'), (int, float)):
            agent.cost += float(usage['cost'])
        message = data['choices'][0].get('message') or {}
        if message.get('content') or message.get('tool_calls'):
            return (message, 200)
        agent.trace.append({'kind': 'llm_fail', 'model': model, 'status': 0})
        return (None, 0)
    error = result.error or {}
    status = error.get('status')
    if isinstance(data, dict) and isinstance(data.get('error'), dict):
        status = _count(data['error'].get('code'))
        error = {'body': data['error'].get('message')}
    agent.trace.append({'kind': 'llm_fail', 'model': model, 'status': status, 'body': str(error.get('body') or '')[:200]})
    return (None, status)

def ridge_call_mark(agent, name: str, args: dict[str, Any]) -> str:
    props = (agent.schemas.get(name) or {}).get('properties') or {}
    kept: dict[str, Any] = {}
    for key, value in sorted(args.items()):
        spec = props.get(key) if isinstance(props.get(key), dict) else {}
        first = value in (spec.get('minimum'), spec.get('default')) or (value == 1 and key == agent.page_args.get(name) and (spec.get('minimum') is None) and (spec.get('default') is None))
        if isinstance(value, int) and (not isinstance(value, bool)) and first:
            continue
        kept[key] = ' '.join(value.lower().split()) if isinstance(value, str) else value
    return name + _dump(kept)

def ridge_buyable(agent, ident: str) -> bool:
    if agent.sold_out(ident):
        return False
    value = (agent.rows.get(ident) or {}).get(agent.stock_key) if agent.stock_key else None
    if isinstance(value, str) and re.search(STOCK_OUT, value.strip(), re.I):
        return False
    return not (value is False or (isinstance(value, (int, float)) and (not isinstance(value, bool)) and (value <= 0)))

def ridge_audit_fits(agent, got: dict[str, Any] | None) -> dict[str, list[str]]:
    fits = (got or {}).get('fits')
    if not isinstance(fits, dict):
        return {}
    named = {str(k): [str(i) for i in _items(v) if str(i) in agent.rows and (not agent.sold_out(str(i)))][:4] for k, v in fits.items()}
    return {k: v for k, v in named.items() if v}

def ridge_audit_call(agent) -> dict[str, Any] | None:
    records = []
    for ident in sorted(reversed(agent.seen_order), key=lambda i: i not in agent.full):
        if agent.sold_out(ident):
            continue
        record = agent.full.get(ident) if ident in agent.full else None
        records.append('OPENED ' + _dump(_clip(record, 200))[:1200] if record else 'ROW ' + _dump(_clip(agent.rows.get(ident), 120))[:500])
        if len(records) >= 30:
            break
    if not records:
        return None
    prompt = AUDIT + '\n\nREQUEST:\n' + agent.request + '\n\nREQUIREMENTS IN FORCE:\n' + agent.ledger()
    answers = [line for line in agent.done_log if 'message to the shopper' in line]
    if answers:
        prompt = prompt + ('\n\nANSWERS FROM THE SHOPPER:\n' + '\n'.join(answers))
    tried = [str(v) for args in agent.searched for v in args.values() if isinstance(v, str)]
    if tried:
        prompt += '\n\nSEARCHES ALREADY TRIED:\n' + ' | '.join(dict.fromkeys(tried))
    prompt += '\n\nLISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
    judges = agent.judges()
    got = agent.ask_json(prompt, [LIGHT] if judges else [], 4000)
    if HEAVY in judges and (not agent.no_credit) and (not agent.audit_fits(got) and True):
        got = agent.ask_json(prompt, [HEAVY], 4000) or got
    agent.trace.append({'kind': 'audit', 'got': got})
    return got

def ridge_audit(agent) -> str:
    if agent.audited or (not agent.judges() and True):
        return ''
    if _ridge_audit_ready(agent.step):
        agent.audited = True
        got = agent.audit_call()
        agent.take_review(got)
        named = agent.audit_fits(got)
        missing = [n for n in range(1, len(agent.items) + 1) if not named.get(str(n))]
        if missing:
            rows = [str(i) for i in _items((got or {}).get('open')) if str(i) in agent.rows and str(i) not in agent.opened and (not agent.sold_out(str(i)))]
            opened = rows and agent.open_rows(rows)
            if agent.recovery_search(got, missing) or opened:
                got = agent.audit_call()
                agent.take_review(got)
                named = agent.audit_fits(got) or named
        if not named:
            return ''
        agent.lead_items = named
        agent.leads = [i for ids in named.values() for i in ids]
        return 'A careful review thinks these listings may fit: ' + '; '.join((f"item {k}: {', '.join(v)}" for k, v in named.items())) + (f" ({got.get('reason')})" if got and got.get('reason') else '') + '. Open them and buy what truly fits; end without a purchase only if each one fails.'
    else:
        return _dump(_ridge_audit_snapshot(agent.step, agent.request))

def ridge_ask_text(agent, number: int, labels: list[str], shown: bool=False) -> str:
    if len(labels) >= 2:
        return f"What {', '.join(labels[:-1])} or {labels[-1]} would you like for the {agent.item_said(number)}?"
    alike = agent.apart.get(labels[0], {}).get(number)
    if alike:
        return f"Apart from the {' and the '.join(alike)}, which {labels[0]} would you like for the {agent.item_said(number)}?"
    return agent.question(number, labels[0], agent.samples(number, labels[0]) if shown else None)

def ridge_ask_next(agent, told: list[str], floor: int=6) -> list[tuple[int, str, str]]:
    for _ in range(LIST_SIZE):
        agent.follow_up = [f for f in agent.follow_up if not agent.told_about(f[0], f[1][0])]
        if agent.follow_up:
            number, labels, shown = agent.follow_up.pop(0)
        else:
            number, labels = agent.next_ask(agent.lanes, agent.lane_found, agent.lane_spent, agent.lane_used)
            shown = False
        if not labels:
            return []
        fitted, text = agent.fit_ask(number, labels, shown)
        if not fitted and shown:
            fitted, text = agent.fit_ask(number, labels, False)
        if fitted:
            break
        agent.lane_used.setdefault(number, set()).add(_words(labels[0]))
        agent.trace.append({'kind': 'ask_too_long', 'item': number, 'label': labels[0]})
    else:
        return []
    labels = fitted
    batch = agent.send([(number, labels[0], text)], floor)
    if not batch:
        return []
    agent.finish_ask(number, labels, shown, batch, told)
    return batch

def ridge_ask_json(agent, prompt: str, models: list[str], max_tokens: int=1800, deep: bool=False) -> dict[str, Any] | None:
    for model in models:
        message = agent.think([{'role': 'user', 'content': prompt}], [model], max_tokens, deep=deep)
        got = _json_obj((message or {}).get('content'))
        if got is not None:
            return _normalize_agent_json(got)
        if agent.broke or agent.no_credit or agent.calls >= ALL_CALLS:
            return None
    return None

def ridge_ask_group(agent, group: list[int], told: list[str], floor: int=6) -> list[tuple[int, str, str]]:
    share = max(1, agent.list_now() // len(group)) if agent.list_now() < agent.list_cap() else max(MIX_LABELS, agent.list_cap() // len(group))
    parts: list[tuple[int, list[str]]] = []
    for number in group:
        left = [label for label in agent.lanes.get(number) or [] if _words(label) not in agent.lane_used.setdefault(number, set()) and (not agent.told_about(number, label))]
        if left:
            parts.append((number, left[:share]))
    if not parts:
        return []
    if len(parts) == 1:
        number, labels = parts[0]
        fitted, text = agent.fit_ask(number, labels, False)
        if not fitted:
            return []
        batch = agent.send([(number, labels[0], text)], floor)
        if batch:
            agent.finish_ask(number, fitted, False, batch, told)
        return batch
    text = ''
    while True:
        for size in range(share, 0, -1):
            pieces = [f"{', '.join(labels[:size][:-1])} or {labels[:size][-1]}" if len(labels[:size]) >= 2 else labels[0] for _, labels in parts]
            text = ', and '.join((f'what {piece} for the {agent.item_said(number)}' for piece, (number, _) in zip(pieces, parts)))
            text = 'W' + text[1:].replace(' for the ', ' would you like for the ', 1) + '?'
            args = agent.message_args(text)
            if args and _says(args, text):
                parts = [(number, labels[:size]) for number, labels in parts]
                break
        else:
            parts = parts[:-1]
            if not parts:
                return []
            continue
        break
    first = parts[0][0]
    batch = agent.send([(first, parts[0][1][0], text)], floor)
    if not batch:
        return []
    for number, _ in parts:
        agent.lane_spent[number] = agent.lane_spent.get(number, 0) + 1
    agent.list_asks += 1
    had = [len(item.told) for item in agent.items]
    mixed_batch = [(0, text, batch[0][2])]
    heard = agent.hear(mixed_batch, (LIGHT, HEAVY))
    heard = agent.rehear_if_doubtful(mixed_batch, heard)
    heard.pop('free', None)
    agent.take_heard(heard, told)
    answered = False
    for number, labels in parts:
        new = agent.items[number - 1].told[had[number - 1]:]
        if new:
            answered = True
            agent.lane_found[number] = agent.lane_found.get(number, 0) + 1
            agent.settle(number, labels, False, list(new), False)
    crowded = any((_count(x) == 1 for x in _items(heard.get('crowded'))))
    if crowded and (not answered):
        agent.listing = False
        agent.trace.append({'kind': 'lists_off', 'item': first})
        return batch
    if not answered:
        agent.lane_used.setdefault(first, set()).add(_words(parts[0][1][0]))
    agent.trace.append({'kind': 'mixed_ask', 'parts': parts, 'answered': answered})
    return batch

def ridge_answered_label(agent, labels: list[str], new: list[str]) -> int:
    seen: dict[str, set[str]] = {}
    for _, key, _, values in agent.key_table():
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

def ridge_answer_open_calls(agent) -> None:
    at = next((n for n in range(len(agent.chat) - 1, -1, -1) if agent.chat[n].get('tool_calls')), None)
    if at is None:
        return
    done = {m.get('tool_call_id') for m in agent.chat[at + 1:]}
    for call in agent.chat[at]['tool_calls']:
        if call['id'] not in done:
            agent.chat.append({'role': 'tool', 'tool_call_id': call['id'], 'content': 'The call may not have run: something went wrong on the way. Look at the state before you repeat it.'})
    agent.chat.append({'role': 'user', 'content': agent.status()})

def ridge_adopt_labels(agent) -> None:
    cat_key = agent.category_key()
    if len(agent.items) < 2 or not cat_key:
        return
    agent.part_shelves()
    labels = {str(record.get(cat_key) or '') for record in agent.full.values()} - {''}
    strays = [label for label in labels if not agent.owner_of(label)]
    orphans = [n for n in range(1, len(agent.items) + 1) if not any((agent.owner_of(label) == n and label not in agent.adopted for label in labels))]
    if strays and len(orphans) == 1:
        agent.adopted.update({label: orphans[0] for label in strays})
        agent.trace.append({'kind': 'adopted', 'item': orphans[0], 'labels': strays})

def ridge_act(agent, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
    payload = dict(agent.binding)
    payload['step'] = agent.step + 1
    payload['action'] = {'name': name, 'args': args}
    if _tool_call_lane_ready(agent.step):
        rest = agent.env_at + agent.env_gap - time.monotonic()
        if rest > 0:
            time.sleep(rest)
        agent.env_at = time.monotonic()
        for _ in range(4):
            began = time.monotonic()
            result = _env.post_verbose('/environment/call', payload)
            agent.env_spent += time.monotonic() - began
            agent.env_count += 1
            if isinstance(result.data, dict):
                data = result.data
                agent.step = data['step'] if isinstance(data.get('step'), int) else agent.step + 1
                if data.get('finished'):
                    agent.finished = True
                if name != agent.roles.get('ask'):
                    data['observation'] = agent.unspelled(name, data.get('observation'))
                seen = data.get('observation')
                agent.note_read(name, args, seen)
                if name == agent.facts_tool() and isinstance(seen, dict) and (not _failed(seen)) and agent.ident(seen) and (not _fact_block(seen)):
                    agent.factless.add((name, agent.ident(seen)))
                if name in agent.page_tools() and (not (isinstance(seen, dict) and seen.get('error'))) and (not _failed(seen)):
                    shown = _dump(seen)
                    for value in args.values():
                        named = [v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str) and (not isinstance(value, list) or v.strip() in shown)]
                        agent.opened.update(named)
                        for ident in named:
                            agent.views.setdefault(ident, set()).add(name)
                return data
            error = result.error or {}
            status = error.get('status')
            body = str(error.get('body') or '')
            agent.trace.append({'kind': 'env_fail', 'status': status, 'body': body[:200]})
            if status == 409:
                low = body.lower()
                if 'finish' in low or 'quarantin' in low or ('finaliz' in low and True):
                    agent.finished = True
                    return None
                found = re.search('expected (\\d+)', body)
                if found:
                    agent.step = int(found.group(1)) - 1
                    payload['step'] = agent.step + 1
                    continue
                if 'reuse' in low or 'different action' in low:
                    agent.step += 1
                    payload['step'] = agent.step + 1
                    continue
                return None
            if status in (400, 401, 402, 403, 404, 422):
                if status == 402:
                    agent.broke = True
                return None
        return None
    else:
        return _tool_call_lane_snapshot(name, args, agent.step)

def ridge_absorb(agent, observation: Any) -> None:

    def walk(node: Any, depth: int, in_list: bool=False) -> None:
        if depth > 6:
            return
        if isinstance(node, dict):
            ident = agent.ident(node, in_list, depth)
            if ident:
                if ident not in agent.rows:
                    agent.seen_order.append(ident)
                    agent.rows[ident] = {}
                old = agent.rows[ident]
                if any((old.get(k) is True and v is False for k, v in node.items())):
                    agent.unavailable.add(ident)
                elif any((old.get(k) is False and v is True for k, v in node.items())):
                    agent.unavailable.discard(ident)
                agent.rows[ident].update(node)
                if _fact_block(node):
                    agent.full[ident] = dict(agent.rows[ident])
                elif ident in agent.full:
                    agent.full[ident].update(node)
                return
            for value in node.values():
                walk(value, depth + 1)
        elif isinstance(node, list):
            rows = agent.learn_rows(node)
            for value in node:
                walk(value, depth + 1, rows)
    walk(observation, 0)

class Ridge:

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
        self.started = time.monotonic()
        self.step = 0
        self.finished = self.view.get('finished')
        self.broke = False
        self.no_credit = False
        self.waited = 0
        self.misses = 0
        self.cost = 0.0
        self.calls = 0
        self.loop_calls = 0
        self.trace: list[dict[str, Any]] = _Stamped(self.clock)
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
        self.confirmed: str | None = None
        self.completed = False
        self.link_tried: set[str] = set()
        self.lane_misses: dict[int, int] = {}
        self.adopted: dict[str, int] = {}
        self.capped: set[tuple[str, float]] = set()
        self.factless: set[tuple[str, str]] = set()
        self.trail_room: int | None = None
        self.id_key: str | None = None
        self.facts_from: dict[str, bool] = {}
        self.queued: list[str] = []
        self.buys = 0
        self.refusals = 0
        self.recent: list[str] = []
        self.chat: list[dict[str, Any]] = []
        self.opened: set[str] = set()
        self.views: dict[str, set[str]] = {}
        self.fit_cached: tuple[Any, dict[str, Any]] | None = None
        self.owners_seen: dict[tuple[str, int], int] = {}
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
        self.paged_obs: list[Any] = []
        self.plan_order: list[int] = []
        self.lanes: dict[int, list[str]] = {}
        self.lane_found: dict[int, int] = {}
        self.lane_spent: dict[int, int] = {}
        self.lane_used: dict[int, set[str]] = {}
        self.late_asked = False
        self.one_product: set[int] = set()
        self.list_asks = 0
        self.apart: dict[str, dict[int, list[str]]] = {}
        self.follow_up: list[tuple[int, list[str], bool]] = []
        self.listing = True
        self.line_holds = 0
        self.bound: list[tuple[int, int]] = []
        self.fit_holds = 0
        self.searched: list[dict[str, Any]] = []
        self.link_searches = 0
        self.link_seen: dict[str, list[str]] = {}
        self.hunts = 0
        self.env_gap = ENV_GAP
        self.env_spent = 0.0
        self.env_count = 0
        self.read_calls: dict[str, Any] = {}
        self.read_args: dict[str, tuple[str, dict[str, Any]]] = {}
        self.value_rows: set[str] = set()
        self.row_keys: set[str] = set()
        self.refuse_trailed = False
        self.env_at = 0.0
        self.hunt_stem: dict[tuple[int, tuple[str, ...]], str] = {}
        self.hunt_no_noun: set[int] = set()
        self.bare_pages: dict[str, int] = {}
        self.code_tries: list[bool] = []
        self.codes_dead = False
        self.hear_reads = 0
        self.fit_hunted = False
        self.hunt_mark = ''
        self.hunt_tried: dict[int, set[str]] = {}
        self.page_args: dict[str, str] = {}
        self.page_queries: dict[str, set[str]] = {}
        self.model_busy: dict[str, float] = {}

    def left(self) -> int:
        return self.max_steps - self.step

    def clock(self) -> float:
        return time.monotonic() - self.started

    def model_room(self) -> float:
        return TASK_SECONDS - TIME_MARGIN - self.clock()

    def rushed(self) -> bool:
        return ridge_rushed(self)

    def msgs_free(self) -> int:
        return ridge_msgs_free(self)

    def cart_mark(self) -> str:
        return ridge_cart_mark(self)

    def row_ids(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return ridge_row_ids(self, name, args)

    def page_tools(self) -> set[str]:
        return ridge_page_tools(self)

    def facts_tool(self) -> str | None:
        return ridge_facts_tool(self)

    def compare_facts(self) -> bool | None:
        return ridge_compare_facts(self)

    def compare_bare(self) -> bool:
        return ridge_compare_bare(self)

    def gate_due(self, stage: str) -> bool:
        return ridge_gate_due(self, stage)

    def tail(self) -> int:
        return ridge_tail(self)

    def judges(self, limit: float=SPEND_SOFT) -> list[str]:
        return ridge_judges(self, limit)

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None=None, deep: bool=False) -> dict[str, Any] | None:
        return ridge_think(self, messages, models, max_tokens, tools, deep)

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None, effort: str | None=None) -> tuple[dict[str, Any] | None, Any]:
        return ridge_call_model(self, model, messages, max_tokens, tools, effort)

    def ask_json(self, prompt: str, models: list[str], max_tokens: int=1800, deep: bool=False) -> dict[str, Any] | None:
        return ridge_ask_json(self, prompt, models, max_tokens, deep)

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        return ridge_act(self, name, args)

    def unspelled(self, name: str, observation: Any) -> Any:
        return ridge_unspelled(self, name, observation)

    def read_request(self) -> None:
        return ridge_read_request(self)

    @staticmethod
    def shared_shelf(brief: dict[str, Any]) -> bool:
        shelves = [_words(str(e.get('shelf') or '')) for e in _items(brief.get('items')) if isinstance(e, dict)]
        shelves = [s for s in shelves if s]
        return len(shelves) != len(set(shelves))

    def unmix(self) -> None:
        return ridge_unmix(self)

    def described(self, number: int, need: str) -> bool:
        return ridge_described(self, number, need)

    def ledger(self) -> str:
        return ridge_ledger(self)

    def status(self) -> str:
        return ridge_status(self)

    def ident(self, node: Any, in_list: bool=False, depth: int=0) -> str | None:
        return ridge_ident(self, node, in_list, depth)

    def learn_rows(self, node: list[Any]) -> bool:
        return ridge_learn_rows(self, node)

    def absorb(self, observation: Any) -> None:
        return ridge_absorb(self, observation)

    def labels(self) -> set[str]:
        return ridge_labels(self)

    def category_key(self) -> str | None:
        return ridge_category_key(self)

    def key_table(self) -> list[tuple[str, str, int, list[str]]]:
        return ridge_key_table(self)

    def not_stated(self, label: str, name: str, values: list[str]) -> str:
        return ridge_not_stated(self, label, name, values)

    @staticmethod
    def twins(name: str, spec: dict[str, Any], other: str, slot: dict[str, Any]) -> bool:
        a, b = (set(name.split()), set(other.split()))
        if not a or not b or (not a & b):
            return False

        def agree(x: str, y: str) -> bool:
            wx, wy = (re.findall('[a-z0-9]+', x.lower()), re.findall('[a-z0-9]+', y.lower()))
            short, long = (wx, wy) if len(wx) <= len(wy) else (wy, wx)
            return x == y or (short and set(short) <= set(long) and (len(long) <= 2 * len(short) + 2))
        both = [i for i in spec['plain'] if i in slot['plain'] and spec['plain'][i].strip() and slot['plain'][i].strip()]
        same = sum((1 for i in both if agree(spec['plain'][i], slot['plain'][i])))
        if not (a <= b or b <= a):
            return len(both) >= 2 and same == len(both)
        if both:
            return same * 2 > len(both)
        return (set(spec['values']) & set(slot['values'])) - {''}

    def item_groups(self) -> list[list[str]]:
        return ridge_item_groups(self)

    def single_target(self, tool: str | None) -> str | None:
        return ridge_single_target(self, tool)

    def price_of(self, ident: str) -> float | None:
        return ridge_price_of(self, ident)

    def buyable(self, ident: str) -> bool:
        return ridge_buyable(self, ident)

    def records_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        return ridge_records_in(self, observation)

    def note_view(self, name: str, observation: Any) -> None:
        return ridge_note_view(self, name, observation)

    def shown_already(self, name: str, ident: str) -> bool:
        return ridge_shown_already(self, name, ident)

    def read_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        return ridge_read_shape(self, name, observation)

    def paging(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        return ridge_paging(self, name, observation)

    def price_sorted(self, observation: Any) -> bool:
        return ridge_price_sorted(self, observation)

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        return ridge_more_pages(self, name, args, observation)

    def cap_ignored(self, name: str, args: dict[str, Any], observation: Any) -> str:
        return ridge_cap_ignored(self, name, args, observation)

    def cheapest_here(self, observations: list[Any]) -> str:
        return ridge_cheapest_here(self, observations)

    def wider_hold(self) -> str:
        return ridge_wider_hold(self)

    def note_facts(self, tool: str, shown: bool) -> None:
        return ridge_note_facts(self, tool, shown)

    def lists_things(self, tool: str) -> bool:
        return ridge_lists_things(self, tool)

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        return ridge_fill_facts(self, ids, keep)

    def show_links(self, ids: list[str], keep: int) -> None:
        return ridge_show_links(self, ids, keep)

    def cap_template(self) -> tuple[str, dict[str, Any], str, str] | None:
        return ridge_cap_template(self)

    def cap_keys(self, tool: str) -> list[str]:
        return ridge_cap_keys(self, tool)

    def cap_modes(self, tool: str) -> dict[tuple[str, tuple[tuple[str, Any], ...]], bool]:
        return ridge_cap_modes(self, tool)

    def list_below(self, cart: list[str]) -> None:
        return ridge_list_below(self, cart)

    def cheaper_rows(self) -> list[str]:
        return ridge_cheaper_rows(self)

    @_soft('')
    def below_cart(self) -> str:
        cat_key = self.category_key()
        cart = self.cart_ids()
        lines = []
        for line in cart:
            price = self.price_of(line)
            kind = str((self.rows.get(line) or {}).get(cat_key) or '').strip().lower() if cat_key else ''
            if price is None or not kind:
                continue
            pool = [i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i) and (str((self.rows.get(i) or {}).get(cat_key) or '').strip().lower() == kind) and ((self.price_of(i) if self.price_of(i) is not None else float('inf')) < price)]
            pool.sort(key=lambda i: self.price_of(i) or 0.0)
            lines += [f'{i} at {self.price_of(i):g}' for i in pool[:CHEAPER_SHOWN]]
        return 'Still unread and below a cart line: ' + ', '.join(lines) + '.' if lines else ''

    def title_key(self) -> str | None:
        return ridge_title_key(self)

    def hunt_words(self, number: int) -> tuple[str, list[str]]:
        return ridge_hunt_words(self, number)

    def rows_of_item(self, number: int) -> int:
        return ridge_rows_of_item(self, number)

    @_soft('')
    def hunt_cheaper(self) -> str:
        mark = _dump([self.needs_now(), self.budget, sorted(self.gone)])
        if self.brief.get('pick') != 'cheapest' or self.hunts >= HUNT_ROUNDS or self.hunt_mark == mark:
            return ''
        plan = self.value_template()
        if not plan or not self.price_key or self.clock() > HUNT_BY:
            return ''
        tool, base, field = plan
        room = min(HUNT_STEPS, self.left() - self.tail() - HUNT_KEEP - 2 * len(self.items))
        if room < 3:
            return ''
        self.hunts += 1
        self.hunt_mark = mark
        cart = self.cart_ids()
        numbers = list(range(1, len(self.items) + 1))
        share = max(3, room // max(1, len(numbers)))
        spent = 0
        fresh_total = 0
        for number in numbers:
            if spent >= room or self.finished:
                break
            used, gained = self.hunt_item(number, tool, base, field, min(share, room - spent))
            spent += used
            fresh_total += gained
        self.trace.append({'kind': 'hunt', 'steps': spent, 'new_rows': fresh_total, 'cart': cart})
        if spent:
            self.wider_held = True
        return f'Ridge ran {spent} more searches for the lowest price, one step each, and they showed {fresh_total} more rows.' if spent else ''

    def hunt_item(self, number: int, tool: str, base: dict[str, Any], field: str, room: int) -> tuple[int, int]:
        return ridge_hunt_item(self, number, tool, base, field, room)

    @_soft(list)
    def hunt_fits(self) -> list[str]:
        if self.fit_hunted or self.clock() > HUNT_BY:
            return []
        plan = self.value_template()
        if not plan:
            return []
        tool, base, field = plan
        keep = self.tail() + len(self.items) + 1
        spare = self.left() - keep
        if spare < 3:
            return []
        self.fit_hunted = True
        cat_key = self.category_key()
        held = [str((self.rows.get(i) or {}).get(cat_key) or '') for i in self.cart_ids() if (self.held_units(i) or 0) > 0] if cat_key else []
        numbers = [n for n, item in enumerate(self.items, 1) if not any((self.owns(item, label) for label in held))] or list(range(1, len(self.items) + 1))
        wanted = {number: {w for value, _ in self.value_words(number) for w in _value_tokens(value)} for number in numbers}
        paged = self.unread_pages(tool, wanted, min(FIT_PAGES, spare // 2))
        room = min(HUNT_STEPS, self.left() - self.tail() - HUNT_KEEP)
        spent = 0
        for number in numbers if room >= 3 else []:
            item = self.items[number - 1]
            pairs = self.value_words(number)
            noun, words = self.hunt_words(number)
            tries = [f'{family} {value} {noun}'.strip() for family in self.series_for(number) for value in [v for v, _ in pairs][:1] or ['']]
            tries = tries + ([f'{value} {noun}'.strip() for value, _ in pairs] + [value for value, _ in pairs if noun] + [f'{word} {noun}'.strip() for word in words[:HUNT_FIT_WORDS]] + ([noun] if noun else []))
            share = max(2, room // max(1, len(numbers)))
            used = 0
            for query in dict.fromkeys(tries):
                if spent >= room or used >= share or self.finished or (self.clock() > HUNT_BY + HUNT_GRACE):
                    break
                faults: list[str] = []
                args = _shape({**base, field: query}, self.schemas.get(tool), 'args', faults)
                if faults:
                    continue
                reply = self.act(tool, args)
                spent = spent + 1
                used += 1
                if reply is None:
                    return []
                self.observe(tool, reply.get('observation'))
        picks: list[str] = []
        for number in numbers:
            item = self.items[number - 1]
            pool = [i for i in self.seen_order if i not in self.full and self.buyable(i) and (not cat_key or self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or '')))]
            cap = self.budget if isinstance(self.budget, (int, float)) else None
            pool = [i for i in pool if cap is None or (self.price_of(i) or 0) <= cap]
            words = wanted.get(number) or set()
            pool.sort(key=lambda i: (-len(words & _text_words(_dump(self.rows.get(i) or {}))), self.price_of(i) if self.price_of(i) is not None else float('inf')))
            picks += pool[:HUNT_FIT_OPENS]
        read: list[str] = []
        for start in range(0, len(picks), HUNT_FIT_OPENS):
            if self.clock() > HUNT_BY + HUNT_GRACE:
                break
            read += self.fill_facts(picks[start:start + HUNT_FIT_OPENS], keep)
        self.trace.append({'kind': 'hunt_fits', 'pages': paged, 'searches': spent, 'opened': len(read)})
        return read

    def unread_pages(self, tool: str, wanted: dict[int, set[str]], room: int) -> int:
        return ridge_unread_pages(self, tool, wanted, room)

    def call_mark(self, name: str, args: dict[str, Any]) -> str:
        return ridge_call_mark(self, name, args)

    def note_read(self, name: str, args: dict[str, Any], seen: Any) -> None:
        return ridge_note_read(self, name, args, seen)

    def query_is_new(self, name: str, args: dict[str, Any]) -> bool:
        return ridge_query_is_new(self, name, args)

    @_soft('')
    def last_page_first(self, name: str, args: dict[str, Any], observation: Any) -> str:
        if name in self.page_args or not isinstance(observation, dict):
            return ''
        props = (self.schemas.get(name) or ({} and True)).get('properties') or {}
        numeric = [k for k, v in args.items() if isinstance(v, int) and (not isinstance(v, bool)) and (k in props) and (v > (props[k].get('minimum') if isinstance(props[k].get('minimum'), int) else 1) and True)]
        if len(numeric) != 1 or self.left() <= self.tail() + 4:
            return ''
        shape = self.read_shape(name, observation)
        if not shape or not shape.get('count') or shape.get('next') is None or (observation.get(shape['next']) is not None):
            return ''
        rows = observation.get(shape['rows'])
        count = observation.get(shape['count'])
        if not isinstance(rows, list) or not isinstance(count, int) or count <= len(rows):
            return ''
        key = numeric[0]
        low = props[key].get('minimum') if isinstance(props[key].get('minimum'), int) else 1
        faults: list[str] = []
        call = _shape({**args, key: low}, self.schemas.get(name), 'args', faults)
        if faults:
            return ''
        reply = self.act(name, call)
        if reply is None:
            return ''
        seen = reply.get('observation')
        self.observe(name, seen)
        new = seen.get(shape['rows']) if isinstance(seen, dict) else None
        if not isinstance(new, list) or len(new) < len(rows) or {i for i, _ in self.records_in(seen)} & {i for i, _ in self.records_in(observation)}:
            return ''
        self.page_args[name] = key
        self.trace.append({'kind': 'page_arg', 'tool': name, 'arg': key})
        return f'\n[Ridge] {key} is the page number of this search: the call above showed its last page. Page {low} of the same search:\n' + self.show(seen)

    @_soft(False)
    def names_code(self, args: dict[str, Any]) -> bool:
        said = {c for v in args.values() if isinstance(v, str) for c in _codes(v)}
        if not said:
            return False
        known: set[str] = set()
        for mark in self.fit_index()['marks'].values():
            known.update(mark.get('codes') or [])
            known.update(mark.get('works') or [])
        return said & known

    def first_look(self, name: str, args: dict[str, Any], fresh: bool) -> dict[str, Any]:
        return ridge_first_look(self, name, args, fresh)

    @_soft('')
    def shorter_search(self, name: str, args: dict[str, Any], observation: Any) -> str:
        plan = self.value_template()
        if not plan or plan[0] != name or self.left() <= self.tail() + SHORT_KEEP or self.finished or (not isinstance(observation, dict)):
            return ''
        field = plan[2]
        text = args.get(field)
        if not isinstance(text, str) or self.records_in(observation) or len(text.split()) < 3:
            return ''
        for value in observation.values():
            if isinstance(value, (list, str)) and value or (isinstance(value, (int, float)) and (not isinstance(value, bool)) and (value > 0)):
                return ''
        short = ' '.join([w for w in text.split() if not _codes(w)][:2])
        if not short:
            return ''
        faults: list[str] = []
        call = _shape({k: v for k, v in {**args, field: short}.items() if isinstance(v, bool) or not isinstance(v, int)}, self.schemas.get(name), 'args', faults)
        if faults:
            return ''
        reply = self.act(name, call)
        if reply is None:
            return ''
        seen = reply.get('observation')
        self.observe(name, seen)
        self.trace.append({'kind': 'shorter', 'from': text, 'to': short, 'rows': len(self.records_in(seen))})
        return f'\n[Ridge] Nothing has all those words; the same search with {short!r} shows:\n' + self.show(seen)

    @_soft('')
    def first_moves(self) -> str:
        plan = self.value_template()
        if not plan or self.max_steps < OPEN_MIN_STEPS or len(self.items) < OPEN_ITEMS:
            return ''
        tool, base, field = plan
        self.env_gap = OPEN_GAP
        shown: list[str] = []
        stop_at = self.step + self.left() - self.tail() - self.msgs_free() - OPEN_LATER * len(self.items) - HUNT_KEEP // 2
        for number, item in enumerate(self.items, 1):
            wanted = [m.split(':', 1)[1].strip() for m in item.musts if ':' in m and m not in [need for _, need in self.refs]]
            wanted = [w for w in wanted if w and len(w.split()) <= 3 and (not re.search(OPEN_NOT, w, re.I))]
            noun = item.name if len(item.name.split()) <= 3 else ' '.join(item.name.split()[-2:])
            tries = list(dict.fromkeys(([' '.join(wanted[:2] + [noun])] if wanted else []) + [noun]))
            for query in tries:
                faults: list[str] = []
                args = _shape({**base, field: query}, self.schemas.get(tool), 'args', faults)
                if faults or self.finished or self.step >= stop_at:
                    break
                reply = self.act(tool, args)
                if reply is None:
                    return '\n'.join(shown)
                seen = reply.get('observation')
                self.observe(tool, seen)
                if self.records_in(seen):
                    self.searched = (self.searched + [dict(args)])[-8:]
                    shown.append(f'{query}: ' + self.show(seen))
                    shown += self.more_rows_for_keys(tool, args, seen, stop_at, number)
                    if self.rows_of_item(number) >= OPEN_EACH:
                        break
                else:
                    shown.append(f"{query}: {('the store refused the call: ' + self.show(seen) if _failed(seen) else 'no rows')}")
        if self.rows and self.roles.get('ask') and (self.msgs_free() > 0):
            start = self.step
            found = [n for n in range(1, len(self.items) + 1) if self.rows_of_item(n)]
            while self.step - start < self.max_steps // OPEN_SHARE and self.step < stop_at and (not self.finished):
                if min((self.records_of_item(n) for n in found), default=OPEN_EACH) >= OPEN_EACH:
                    break
                had = len(self.full)
                self.widen()
                if len(self.full) == had:
                    break
        self.env_gap = ENV_GAP
        self.trace.append({'kind': 'first_moves', 'rows': len(self.rows), 'opened': len(self.full)})
        shown += ['read in full: ' + self.page_text(ident, record) for ident, record in self.full.items()]
        proof = self.evidence(list(self.full.values()))
        return '\n'.join(shown + ([proof] if proof else []))

    def page_text(self, ident: str, record: dict[str, Any]) -> str:
        return ridge_page_text(self, ident, record)

    def more_rows_for_keys(self, tool: str, args: dict[str, Any], seen: Any, stop_at: int, number: int) -> list[str]:
        return ridge_more_rows_for_keys(self, tool, args, seen, stop_at, number)

    def observe(self, name: str, observation: Any) -> str:
        return ridge_observe(self, name, observation)

    def widen(self) -> None:
        return ridge_widen(self)

    def widen_singly(self) -> None:
        return ridge_widen_singly(self)

    def ready_to_ask(self) -> bool:
        return ridge_ready_to_ask(self)

    def message_args(self, text: str) -> dict[str, Any] | None:
        return ridge_message_args(self, text)

    def interview(self) -> str:
        return ridge_interview(self)

    def late_question(self) -> str:
        return ridge_late_question(self)

    def search_template(self) -> tuple[str, dict[str, Any], str] | None:
        return ridge_search_template(self)

    def value_words(self, number: int) -> list[tuple[str, bool]]:
        return ridge_value_words(self, number)

    def value_template(self) -> tuple[str, dict[str, Any], str] | None:
        return ridge_value_template(self)

    def value_search(self, numbers: list[int]) -> str:
        return ridge_value_search(self, numbers)

    def home_label(self, number: int) -> str:
        return ridge_home_label(self, number)

    def groups_for(self, free: int) -> list[list[int]]:
        return ridge_groups_for(self, free)

    def ask_group(self, group: list[int], told: list[str], floor: int=6) -> list[tuple[int, str, str]]:
        return ridge_ask_group(self, group, told, floor)

    def finish_ask(self, number: int, labels: list[str], shown: bool, batch: list[tuple[int, str, str]], told: list[str]) -> None:
        return ridge_finish_ask(self, number, labels, shown, batch, told)

    def list_cap(self) -> int:
        return ridge_list_cap(self)

    def list_now(self) -> int:
        return ridge_list_now(self)

    def next_ask(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> tuple[int, list[str]]:
        return ridge_next_ask(self, lanes, found, spent, used)

    def told_about(self, number: int, label: str) -> bool:
        return ridge_told_about(self, number, label)

    def look_alikes(self, number: int, label: str) -> list[str]:
        return ridge_look_alikes(self, number, label)

    def ask_text(self, number: int, labels: list[str], shown: bool=False) -> str:
        return ridge_ask_text(self, number, labels, shown)

    def fit_ask(self, number: int, labels: list[str], shown: bool) -> tuple[list[str], str]:
        return ridge_fit_ask(self, number, labels, shown)

    def ask_next(self, told: list[str], floor: int=6) -> list[tuple[int, str, str]]:
        return ridge_ask_next(self, told, floor)

    def settle(self, number: int, labels: list[str], shown: bool, new: list[str], crowded: bool) -> None:
        return ridge_settle(self, number, labels, shown, new, crowded)

    def answered_label(self, labels: list[str], new: list[str]) -> int:
        return ridge_answered_label(self, labels, new)

    def lane_keys(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        return ridge_lane_keys(self, number, table)

    def adopt_labels(self) -> None:
        return ridge_adopt_labels(self)

    def owner_of(self, label: str) -> int:
        return ridge_owner_of(self, label)

    def owns(self, item: Item, label: str) -> bool:
        return ridge_owns(self, item, label)

    def records_of_item(self, number: int) -> int:
        return ridge_records_of_item(self, number)

    def plan_lanes(self, count: int) -> dict[int, list[str]]:
        return ridge_plan_lanes(self, count)

    def name_parts(self, number: int) -> list[str]:
        return ridge_name_parts(self, number)

    def item_said(self, number: int) -> str:
        return ridge_item_said(self, number)

    def question(self, number: int, label: str, samples: list[str] | None=None) -> str:
        return ridge_question(self, number, label, samples)

    def samples(self, number: int, label: str) -> list[str]:
        return ridge_samples(self, number, label)

    def send(self, questions: list[tuple[int, str, str]], floor: int=6) -> list[tuple[int, str, str]]:
        return ridge_send(self, questions, floor)

    def doubtful(self, entry: Any) -> bool:
        return ridge_doubtful(self, entry)

    def rehear_if_doubtful(self, batch: list[tuple[int, str, str]], heard: dict[str, Any]) -> dict[str, Any]:
        return ridge_rehear_if_doubtful(self, batch, heard)

    def take_heard(self, heard: dict[str, Any], told: list[str]) -> None:
        return ridge_take_heard(self, heard, told)

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(HEAVY, LIGHT)) -> dict[str, Any]:
        return ridge_hear(self, sent, models)

    @_soft(False)
    def quiet_reply(self, sent: list[tuple[int, str, str]]) -> bool:
        if self.hear_reads < 1:
            return False
        known: frozenset[str] = frozenset()
        for number, item in enumerate(self.items, 1):
            known |= _spelled(item.name) | _spelled(item.shelf or '')
            for need in item.musts + item.told:
                known |= _spelled(' '.join(re.findall('[A-Za-z0-9]+', need.split(':', 1)[0])))
        for label, name, _, _ in self.key_table():
            known |= _spelled(name) | _spelled(label)
        for record in self.full.values():
            for value in (_fact_block(record) or {}).values():
                known = known | {w for w in _spelled(' '.join(re.findall('[A-Za-z0-9]+', str(value)))) if len(w) >= QUIET_WORD}
        for _, question, answer in sent:
            asked = re.sub('\\b(?:what|which|would|you|like|for|the|please|tell|me|and|or)\\b', ' ', question, flags=re.I)
            mine = known | _spelled(' '.join(re.findall('[A-Za-z0-9]+', asked)))
            said = _spelled(' '.join(re.findall('[A-Za-z0-9]+', answer)))
            if re.search('\\d', answer) or said & mine or re.search(ASK_WORDS, answer, re.I):
                return False
        return True

    def gate(self, stage: str) -> tuple[bool, str]:
        return ridge_gate(self, stage)

    def spoken(self, observation: Any) -> str:
        return ridge_spoken(self, observation)

    def loose_lines(self, observation: Any) -> list[str]:
        return ridge_loose_lines(self, observation)

    def take_note(self, lines: list[str]) -> str:
        return ridge_take_note(self, lines)

    def needs_now(self) -> list[tuple[int, str]]:
        return ridge_needs_now(self)

    def evidence(self, observation: Any) -> str:
        return ridge_evidence(self, observation)

    def cart_text(self) -> str:
        return ridge_cart_text(self)

    def held_units(self, ident: str) -> int:
        return ridge_held_units(self, ident)

    def cart_ids(self) -> list[str]:
        return ridge_cart_ids(self)

    def record_of(self, ident: str) -> str:
        return ridge_record_of(self, ident)

    def others(self, skip: list[str], limit: int=30) -> str:
        return ridge_others(self, skip, limit)

    def fit_index(self) -> dict[str, Any]:
        return ridge_fit_index(self)

    def fit_set(self, ident: str, index: dict[str, Any]) -> set[str]:
        return ridge_fit_set(self, ident, index)

    def item_ids(self, number: int) -> list[str]:
        return ridge_item_ids(self, number)

    def link_system(self, part: int, main: int, index: dict[str, Any]) -> tuple[int, int, set[str]] | None:
        return ridge_link_system(self, part, main, index)

    def link_holds(self, carried: str, namer: str, families: set[str], index: dict[str, Any]) -> bool | None:
        return ridge_link_holds(self, carried, namer, families, index)

    def link_faults(self, part: int, main: int, device: str, lines: list[str], index: dict[str, Any]) -> list[str]:
        return ridge_link_faults(self, part, main, device, lines, index)

    def series_for(self, number: int) -> list[str]:
        return ridge_series_for(self, number)

    def fit_faults(self) -> list[str]:
        return ridge_fit_faults(self)

    def fit_plan(self, short: bool=False) -> str:
        return ridge_fit_plan(self, short)

    @_soft('')
    def cheapest_sets(self, main: int, parts: list[int], index: dict[str, Any]) -> str:
        options = []
        systems = {part: self.link_system(part, main, index) for part in parts}
        shelves = {part: sorted(((self.price_of(i), i, self.fit_set(i, index), index['marks'].get(i, {}).get('bars') or []) for i in self.item_ids(part) if self.price_of(i) is not None and self.buyable(i)), key=lambda o: (o[0], o[1])) for part in parts}
        for device in self.item_ids(main):
            codes = index['marks'].get(device, {}).get('codes') or []
            price = self.price_of(device)
            if not (codes or index['marks'].get(device, {}).get('named')) or price is None or (not self.buyable(device)):
                continue
            total = price * self.items[main - 1].qty
            picks = [device]
            for part in parts:
                system = systems[part]
                fitting = next((i for _, i, fits, bars in shelves[part] if any((_code_hit(p, c) for p in fits for c in codes)) and (not any((_code_hit(p, c) for p in bars for c in codes))) or (system and system[0] == part and (self.link_holds(i, device, system[2], index) is True))), None)
                if not fitting:
                    picks = []
                    break
                picks.append(fitting)
                total += (self.price_of(fitting) or 0.0) * self.items[part - 1].qty
            if picks:
                options.append((total, picks))
        options.sort(key=lambda o: o[0])
        if not options:
            return ''
        return 'lowest whole sets seen, requirements not yet checked: ' + '; '.join((f"{' + '.join(p)} = {t:g}" for t, p in options[:SET_SHOWN]))

    def missing_items(self) -> list[int]:
        return ridge_missing_items(self)

    def states_needs(self, ident: str, number: int) -> bool:
        return ridge_states_needs(self, ident, number)

    def fill_short_cart(self) -> bool:
        return ridge_fill_short_cart(self)

    def part_shelves(self) -> None:
        return ridge_part_shelves(self)

    def fuller_label(self, number: int, label: str) -> str:
        return ridge_fuller_label(self, number, label)

    def named_label(self, number: int, label: str) -> str:
        return ridge_named_label(self, number, label)

    def line_faults(self) -> list[str]:
        return ridge_line_faults(self)

    def count_fixes(self) -> list[tuple[str, int, int]]:
        return ridge_count_fixes(self)

    def fix_counts(self, wrong: list[tuple[str, int, int]]) -> list[str]:
        return ridge_fix_counts(self, wrong)

    def check_key(self, ids: list[str]) -> str:
        return ridge_check_key(self, ids)

    def check_cart(self, ids: list[str] | None=None, limit: float=SPEND_SOFT) -> str:
        return ridge_check_cart(self, ids, limit)

    def refuse_trail(self) -> str:
        return ridge_refuse_trail(self)

    def early_trail(self) -> str:
        return ridge_early_trail(self)

    def page_read(self, ident: str) -> bool:
        return ridge_page_read(self, ident)

    def trail(self, stop: float | None=None) -> tuple[str, bool]:
        return ridge_trail(self, stop)

    def not_maker_keys(self) -> tuple[Any, ...]:
        return ridge_not_maker_keys(self)

    def shop_rows(self, about: str, ids: list[str]) -> list[str]:
        return ridge_shop_rows(self, about, ids)

    def opened_shop(self, about: str) -> list[str]:
        return ridge_opened_shop(self, about)

    def find_link(self, about: str, words: str) -> list[str]:
        return ridge_find_link(self, about, words)

    def offer_tool(self) -> str | None:
        return ridge_offer_tool(self)

    def reread_offers(self) -> str:
        return ridge_reread_offers(self)

    def sold_out(self, ident: str) -> bool:
        return ridge_sold_out(self, ident)

    def swap_hint(self) -> str:
        return ridge_swap_hint(self)

    def find_sold(self) -> str:
        return ridge_find_sold(self)

    def replacements(self, lost: list[str]) -> list[str]:
        return ridge_replacements(self, lost)

    def take_review(self, got: dict[str, Any] | None) -> None:
        return ridge_take_review(self, got)

    def search_words(self, tool: str, base: dict[str, Any], field: str, words: str) -> list[str]:
        return ridge_search_words(self, tool, base, field, words)

    def kind_rows(self, number: int, found: list[str], below: float | None=None) -> list[str]:
        return ridge_kind_rows(self, number, found, below)

    def recovery_search(self, got: dict[str, Any] | None, missing: list[int]) -> bool:
        return ridge_recovery_search(self, got, missing)

    def audit(self) -> str:
        return ridge_audit(self)

    def audit_call(self) -> dict[str, Any] | None:
        return ridge_audit_call(self)

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        return ridge_audit_fits(self, got)

    def open_rows(self, rows: list[str]) -> bool:
        return ridge_open_rows(self, rows)

    def cart_calls(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        return ridge_cart_calls(self, lines, drop)

    def complete_cart(self) -> str:
        return ridge_complete_cart(self)

    def is_buy(self, name: str, args: dict[str, Any]) -> bool:
        return ridge_is_buy(self, name, args)

    def is_refuse(self, name: str, args: dict[str, Any]) -> bool:
        return ridge_is_refuse(self, name, args)

    def offered(self) -> list[dict[str, Any]]:
        return ridge_offered(self)

    def quiet(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return ridge_quiet(self, name, args)

    def resolve_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        return ridge_resolve_ids(self, name, args)

    def cart_sets(self, name: str) -> bool:
        return ridge_cart_sets(self, name)

    def whole_removal(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return ridge_whole_removal(self, name, args)

    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        return ridge_note_cart(self, name, args)

    def show(self, observation: Any) -> str:
        return ridge_show(self, observation)

    def compare_target(self) -> str | None:
        return ridge_compare_target(self)

    def compare_room(self) -> int:
        return ridge_compare_room(self)

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return ridge_join_opens(self, calls)

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        return ridge_run_calls(self, calls)

    def can_refuse(self) -> bool:
        return ridge_can_refuse(self)

    def finish(self, again: bool=False) -> str:
        return ridge_finish(self, again)

    def trim(self) -> None:
        return ridge_trim(self)

    def run(self) -> list[dict[str, Any]]:
        return ridge_run(self)

    def work(self, loop_cap: int, spend_cap: float) -> None:
        return ridge_work(self, loop_cap, spend_cap)

    def answer_open_calls(self) -> None:
        return ridge_answer_open_calls(self)

    def turns(self, loop_cap: int, spend_cap: float) -> None:
        return ridge_turns(self, loop_cap, spend_cap)

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Ridge(problem_data)
    except Exception as exc:
        return [{'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]}]
    try:
        return list(episode.run())
    except Exception as exc:
        episode.trace.append({'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]})
        try:
            if not episode.finished:
                episode.finish()
        except Exception as again:
            episode.trace.append({'kind': 'crash', 'error': f'{type(again).__name__}: {again}'[:400]})
        return list(episode.trace)
__all__ = ['agent_main']
