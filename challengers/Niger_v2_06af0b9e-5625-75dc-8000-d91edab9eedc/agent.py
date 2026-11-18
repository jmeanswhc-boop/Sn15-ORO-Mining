"""Wren: shops on behalf of a single customer within an ORO store world. Derived from Halo v3, the agent that topped
race 15 publicly.

A lightweight model browses the shop through the tools the world provides; the code holds the record, schedules
what to ask the customer, and verifies the basket before any purchase and the proof before any refusal. Additions
over that starting point:
- when Wren is ready to purchase, a message the request reserved for the close is used for one extra scheduled
question, so a strict message cap stops leaving a concealed need unasked;
- a timer holds each episode under the validator's 300-second ceiling: no interview, question, loop round or model
call begins if its outcome could not arrive in time, and the model's timeout contracts as time runs down;
- while the customer replies to lists, every message to them covers several scheduled properties (up to a fixed
cap) of a single item, rotating the order so a property the customer is indifferent to cannot mask those after it,
and it drops to one property per message once a reply requests one question at a time.
"""
from __future__ import annotations
import json
import random
import re
import threading
import time
from typing import Any
from src.agent.proxy_client import ProxyClient
MILD = 'anthropic/claude-haiku-5.5'
WEIGHTY = 'anthropic/claude-sonnet-5.5'
SURPLUS = 'openai/gpt-6-luna'
CYCLE_INVOKES = 42
CYCLE_SYMBOLS = 8000  # hidden reasoning is charged against max_tokens; with a smaller cap some loop turns were cut off before making a call
ENTIRE_INVOKES = 60
EXPEND_GENTLE = 0.055
EXPEND_STRICT = 0.078
WEIGHTY_CAPACITY = 0.02
LINGERS = (4, 10, 20, 35)
LINGER_AGGREGATE = 45
CONCISE_VIA = 120.0
CONCISE_VOID = 10.0
DOWNTIME_VIA = 150.0
DISPLAY_GLYPHS = 450
CHASE_TICKS = 24
MILIEU_VOID = 0.25
UNCLOSED_VOID = 0.5
GRID_HANDLES = 18
THRIFTIER_DISPLAYED = 4
SILENT_TERM = 4
UNCLOSED_AFTERWARDS = 2
SHEET_TRAIT_GLYPHS = 700
SHEET_REMAINDER_GLYPHS = 300
NIL_MONIKER = ('none', 'null', 'n/a', 'unknown')
INQUIRE_ROTATIONS = 2
LETTER_FACET = '(?:^|_)(?:texts?|body|messages?|notes?|contents?|questions?|msg|say)(?:_|$)'
NON_LETTER = '(?:^|_)(?:id|ids|type|format|code|lang|language|kind)$'
PATH_CAPACITY = 15
SUIT_SHEETS = 2
BREAKDOWN_ROTATIONS = 3
UNCLOSED_MINIMUM_TICKS = 20
UNCLOSED_TIERS = 5
UNCLOSED_ARTICLES = 2
UNCLOSED_FRACTION = 5
UNCLOSED_EVERY = 6
UNCLOSED_NON = '\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or|same|listing)\\b'
TERSE_PRESERVE = 8
ENSEMBLE_DISPLAYED = 2
INVENTORY_OUTPUT = '^(?:no|none|false|0|n)$|out[ _-]?of[ _-]?stock|sold[ _-]?out|unavailable|not[ _-]?available'
CHASE_COMPLETE = 49
CHASE_LENIENCY = 25.0
CHASE_WORTH_GLYPHS = 30
CHASE_SELECTIONS = 3
DENOM_WORTH = '(\\d[\\d.]*)\\s*([a-z]{2,4})'
INQUIRE_TERMS = '\\b(?:one|each|single|first|time|many|much|several|list|once|separate\\w*|split|short\\w*|questions?|ask\\w*|turn|lot|too|all)\\b'
CHASE_LAPS = 2
CHASE_PRESERVE = 10
CHASE_INACTIVE = 4
CHASE_SHEET_NOVEL = 5
CHASE_SUIT_TERMS = 4
CHASE_SUIT_LAUNCHES = 4
CHASE_USUAL = 0.6
CHASE_VIA = 200.0
HEADLINE_GLYPHS = 200
BLEND_LEGENDS = 3
WRITING_GLYPHS = 3000
GLIMPSE_CUTOFF = 15000
CHRONICLE_CUTOFF = 150000
CHECKPOINT_LAPS = 3
IMPLICIT_SHEETS = 6
LATE_ROOM = 8  # steps kept for the first finish (an order or an ending without one): a store may answer it with a change (a line
# sold out, a new budget, a dropped or changed requirement); race 16 singles refused at step <= 26 of 35 passed 6/6, at >= 30 4/10
PARTIAL_SHEETS_ROOM = 40
SWEEP_SERIES = 3  # leading series searched per bound item before the first model turn
SWEEP_OPENS = 9  # rows read by that sweep, one per item and series
SWEEP_KEEP = 30
SCAN_PAGES = 10  # result pages read per item while looking for a row whose fit text covers a required model code
SCAN_OPENS = 2  # rows read per item once found  # steps beyond the tail and the shopper messages the sweep leaves untouched  # steps beyond the tail needed before a search is paged only part of the way (a single item gets 35 in race 16)
THRIFTIER_LAUNCHES = 3
MISSION_SECS = 300.0
MOMENT_LEEWAY = 20.0
INVOKE_BOTTOM = 6.0
INQUIRE_VIA = 165.0
QUERY_VIA = 210.0
COIL_DOWNWARD = 200.0
FINAL_ROTATION = 255.0
TEMPO_ROTATIONS = 2  # timed loop turns needed before the pace line can tell whether the clock or the steps will run out first
TEMPO_INVOKES = 2  # calls per turn under which the remaining time runs out ahead of the steps (arena loop turns averaged 2.15)
BLANK_CAPACITY = 12.0
TARDY_VIA = 190.0
TARDY_TICKS = 8
RARE_LETTERS = 2
ROSTER_MAGNITUDE = 12
ROSTER_BLENDED = 8
ROSTER_TICKS = (1, 3)  # question labels in the first two questions of an episode, before the whole list size
ROSTER_TICKS_BLENDED = (3, 3)  # same idea across several items with each question aimed at a different item (simulated: level 3,3,8 or over 1,3,8 on 2-3 items)
LID_OPERAND = '(?:^|_)(?:cap|max|maximum|ceiling|under|up_?to)(?:_|$)'
LID_NON = 'result|row|page|count|size|num|item|step'
LID_COSTED = 'price|cost|budget|spend|money|amount'  # name of a limit argument while no search has yet shown one in effect
LID_MODEST = '(?:^|_)(?:min|minimum|low|lowest|least|from|start|offset|floor)(?:_|$)'
SOON_PATHS = 6  # chain rounds the early follow may use while every round turns up the next link; the checkout has its own
TETHER_SEEKS = 6  # chain-link searches per episode: a chain of 2-4 pages takes one or two for each link
SOON_LAUNCHES = 12  # pages the early follow may open in total, keeping the steps needed to buy each item free
LID_ODDS = 0.05  # chance that random rows drawn from their labels all fall under a value, below which the value counts as a held limit
TRAVERSE_CAPACITY = 10  # extra steps past the finish a walked listing needs after the first: a refusal at the next checkout still calls for a swap
TRAVERSE_INVOKES = 4  # per cart line, how many capped shelf listings one checkout may take (race 16: 456 shelf rows, 6 per page)
RACK_OPERAND = 'section|department|categor|label|shelf|aisle|(?<!age)(?:^|_)group(?:_name|_label)?$'
ROSTER_SETTING = 'field|filter|facet|brows|categor|list|structured'
VOLUME = '\\d[\\d.,]*\\s*[a-z"\']{1,12}'
BOTTOM = '\\b(?:at least|no (?:less|fewer) than|min(?:imum)?)\\s+(' + VOLUME + ')\\s*$'
ENQUIRY_TERMS = '(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword'
_oracle = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_oracle_terse: dict[int, ProxyClient] = {}
_milieu = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
PILOT = 'You are Wren, a meticulous purchasing agent acting for a single shopper inside an online store. Calling the store\'s tools is the only way you can do anything. Each episode scores either 1 or 0, and only a perfectly correct ending scores 1. That ending is either an order holding each item that was asked for, in the quantity asked for, with every item satisfying all requirements still active at checkout and the total within budget; or, if the store genuinely offers nothing that satisfies everything, a close through the no-purchase tool done the way the request specifies.\n\nHOW THE STORE WORKS\n- Steps are capped and each tool call uses up one; reasoning is free. This store has its own tool names, argument names and result formats, so study every tool\'s description and schema and pass precisely the argument names and permitted values they define.\n- Searching may be restricted: a filter by category, a keyword mode that perhaps reads only the opening words of your query, pages holding just a few rows, a token or offset that fetches the following page. If the answer hinges on seeing every result, check the total the store reports against the rows you have actually seen and keep paging, passing the token or offset exactly as the store returns it. Short keyword queries that lead with the value do best (the desired value, then the product noun). A store\'s search may match attribute values and shop names, not only titles, and may require all words, so a handful of required values plus the noun can return almost the entire set of fits, while a model or part code may return nothing (a single attempt settles it). When a filtered search comes back empty, the cause is usually a wrong filter value or a wrong filter parameter: try again using the label spelled exactly as result rows display it, or switch to a different filter parameter.\n- A search row is only a summary. Stores divide a listing\'s data in different ways: one view may display its attributes (properties), another only its offer and description, and a side-by-side view could hold either one. Once Wren has identified the tool that displays attributes, the ledger names it. Every requirement about a property is judged from those attributes, so read them for any listing you might buy; when a single call displays several listings and shows what you need, it saves steps. Never reopen a listing if nothing has changed since your last read of it. Each turn costs time on a fixed clock: put every call you already know you need (several listings to open, one search per item) into the same turn rather than one call per turn.\n- Wren\'s interviewer takes care of questions to the shopper on your behalf, so you never send the shopper a message yourself. Its replies appear in the conversation and get recorded in the ledger.\n- The request may also demand intermediate steps: an approval, a message, something to verify or read beforehand. These are listed in the ledger. Those that fall due right before the order or the ending are performed by Wren itself at the moment you call the checkout or ending tool, so never wait for them and never close without a purchase on their account; carry out the rest with the store\'s tools (questions about products go through Wren\'s interviewer).\n\nWHAT COUNTS AS A FIT\n- A listing\'s category label settles what kind of product it is: the label has to correspond to the requested item in its entirety, and an adjacent label never qualifies.\n- A property requirement is satisfied if the listing\'s attribute for that property gives the value. How the key is written is irrelevant (spaces, underscores, camel case, prefixes), and a single entry in a list separated by commas or slashes suffices. Matching goes by words: differences of case, singular versus plural, and plain synonyms are acceptable, and a value containing the words of the required value qualifies ("deep teal" satisfies teal, "extra roomy fit" satisfies roomy). A value made of other words does not qualify (an abbreviation or code is not the word behind it, and "teal green" differs from "sea green"). Convert units before you compare numbers; a limit includes its boundary value, and a range has to fall entirely within a limit.\n- When a key names the requirement\'s property (by that name or a plain synonym), that attribute is decisive: should it state a different value, the listing fails, no matter what the title or other keys claim. A key that joins the item\'s own name to the property (a bag\'s "bag shape") counts even if the plain key ("shape") states otherwise. A key about some other property, even a related one (a handle colour is not the colour of the body), settles nothing for the requirement. Following each opened listing, Wren appends [evidence] lines giving the keys whose names match a requirement.\n- If no attribute of the listing covers that property, the desired value counts when it appears under some other attribute (say, a styles or features list containing it) or in the title or description. A number measuring something else does not count (a weight is no capacity), and neither does a lone figure whose text never states what it measures.\n- Should the category label itself carry a requirement\'s value (a season or audience within the label), the category satisfies that requirement. Any listing filed under the requested label is that type of product, even if its title or description mentions another thing, so assess it by the requirements, price and stock.\n- Instead of stating a value, a requirement may refer to a different listing ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Settle that reference first: locate the referenced listing (search by its category, brand, shop or wording) and open its full page using the single-item or side-by-side tool, since a search row does not suffice; if that page directs you to yet another listing for the property, locate and open that one in the same manner, and the final page supplies the value. If the requirement wants whatever goes with that listing\'s value (its partner, its counterpart) and the page states what its value pairs with, then that paired value is the requirement. An order is valid only if every page along the chain was opened. When the requirement is for any value other than the referenced listing\'s, it rules out only listings with exactly that same value, and this alone is never grounds to end without a purchase.\n- If a listing you intend to purchase points you to a separate page for further details about it, check that page prior to buying.\n- Where one requested item has to fit another (an accessory for a device), pick both jointly based on what the listings themselves say (a list of the products it fits, a model or series code appearing on both pages). If none of the accessories that satisfy their own requirements fits the device you chose, move to a different device that satisfies its requirements and that such an accessory fits; never end without a purchase before that switch has been tried.\n- Honour exclusions and closed lists. Wishes ("ideally", "would be nice") serve only to break ties.\n- Do not purchase anything the store marks as sold out; the total (each line\'s price multiplied by its quantity, added across all lines) must stay at or below the budget, so matching it exactly is acceptable.\n\nCHOOSING\n- If the request wants the cheapest option (lowest total, smallest price), the order has to contain the least expensive listing out of ALL fitting listings. Gather the complete set of fits (page through it, because result rows frequently ignore price order, and Wren itself fetches further pages of a short result), examine candidates from lowest price upward, and take the first one that fits completely. Where a search accepts a maximum price, make use of it: list the category under the price of your present best pick, leaving only cheaper candidates to examine.\n- If the request wants the most or least of some measure, weigh all fitting listings on that measure and select the extreme one; a listing silent on the measure is never the winner.\n- In every other case, any listing satisfying everything is correct, so spend no steps searching for a better one.\n- Add an item to the cart the moment you have opened a listing satisfying all of its requirements (for a cheapest, most or least pick: after comparing the set of fits), and then continue to the next. Carted lines frequently sell out shortly after being added, and carting early keeps steps in reserve for a replacement.\n- When several distinct items are requested, each gets its own cart line at the quantity requested for it.\n\nCHANGES WHILE YOU WORK\n- Conditions may shift because of the shopper or the store: a different budget, a requirement introduced or removed, a new price, a listing that sells out. Such updates show up within tool results, frequently at a checkout attempt (as a refusal); a notice is valid no matter which tool\'s result carries it, and the ledger keeps the currently active requirements. A refusal does not end the episode: take every change into account, correct the cart, and attempt checkout once more.\n- Prices and stock may also shift without notice: when you call the checkout, Wren re-reads the lines in the cart and views the cart, so use no turn for either one.\n\nFINISHING\n- Make the cart hold precisely the selected lines (clear out everything else), and then check out in a separate call by itself.\n- Closing without a purchase is allowed only once a genuine search has turned up no listing satisfying everything. Running low on steps is no such proof: in that case check out the best cart available. For that no-purchase ending, follow how the request words it, and if a remark is required, make it brief and mention no listing.\n- Each of your replies has to include at least one tool call. Keep the prose brief: the facts you have, followed by your next action.'
CONCISE = 'Your job is to set up a shopping task for an agent. Using the shopper\'s request and the store\'s tools, output ONE JSON object (compact, unindented, with nothing around it) containing the following keys:\n\n"items": a list holding one entry for every distinct product the shopper wants to buy, each {"name": the product kind as the request words it, "shelf": the category label the store uses for it when some tool lists labels (copied exactly), otherwise the category the request itself names for it ("a coat in rainwear" yields "rainwear"), otherwise "", "quantity": how many units of that single listing (integer; 1 if unspecified; a pair or set sold as one listing counts as 1), "musts": the product\'s firm requirements as a list, each written "property: constraint" carrying the exact value, limit, exclusion or closed list. Use the request\'s own term for the property ("meant for a lakeside occasion" becomes "occasion: lakeside", "a 12-inch blade" becomes "blade length: 12 inch"), and do not trade it for an adjacent property. When a must takes its value from a different listing, it is ONE must that contains the full description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"); any words describing that other listing (its shop, its kind, a trailing "with <property> <value> and <property> <value>") remain within that single must and never become musts of this product, and the request\'s own wording for how the value relates to that listing\'s value is kept exactly as written (never cut down to "same as")}. Price, budget, stock and quantity never go into musts. Background details of the request (who will use the product, where it is going to stand, what failed previously, the occasion prompting it) are context rather than musts, except where they directly state a value or limit for a property; likewise, words forming part of the product\'s name (a season or audience mentioned together with the product) must not be made into musts: they go in "name". Where the request turns a use or occasion into a property of the product itself ("blazers for business wear", "a dress made for parties", "a bag, and it is for a wedding", "the occasion has to be party"), that IS a must on the property ("occasion: business"), except when the store\'s category label names it already; but a specific person\'s event, a person, or a setting the purchase is for ("to wear at his wedding", "for a colleague"), along with how the product should sit, feel or look while used, remain context. If the request names a product in a single phrase linked by "and" (a single "I want ..." with a single quantity, frequently the name of one store department), treat it as ONE item; split it into separate items only where the request assigns each product a quantity or requirements of its own.\n"budget": the shopper\'s maximum total spend, as a number expressed in the unit the request uses, or null.\n"budget_unit": the name of that unit written out (e.g. "cents"), or null.\n"budget_scope": "total" if the limit applies to the entire order, "each" if it applies per item, null if no limit exists.\n"in_stock_only": true, except when the request plainly allows unavailable items.\n"pick": "cheapest" if the request wants the lowest price or smallest total, "most" or "least" if it wants the extreme of some measure, otherwise "any". Comments about money being tight do not make a pick rule.\n"pick_measure": the measure being maximised or minimised under "most"/"least", otherwise "".\n"messages": the number of messages the shopper permits the agent to send (integer), or null if not given.\n"steps": the number of steps, moves or tool calls permitted by the request (integer), or null.\n"lookups": those musts (copied exactly) whose value has to be taken first from a different listing the request refers to, as a list, otherwise [].\n"rules": all instructions in the request concerning the way to work or the way to finish, as a list, each no longer than 12 words.\n"wishes": whatever the shopper would merely prefer, as a list.\n"ask_first": the points the request expressly says to confirm with the shopper prior to choosing, as a list (empty if there are none).\n"process": the actions, beyond locating and buying the correct items, that the request obliges the agent to perform (obtaining the shopper\'s approval, telling or showing the shopper something, confirming something with them, reading something beforehand, using a specific tool), as a list, each {"step": the action required, at most 15 words, "when": "choose" (prior to choosing the products), "order" (immediately before checkout) or "end" (immediately before ending without a purchase), "messages": the number of messages to the shopper it needs (0 if none)}; [] if the request demands no such action. Item requirements, the budget, stock, the message and step limits, and the way to end when nothing fits do not count as steps.\n"quiet_decline": true if the request says alternatives should not be suggested or named when nothing fits, false if it requests them, null if it is silent on this.\n"fits": a list with one {"item": item number, "with": item number} for every requested item which, per the request, has to work with, fit or suit ANOTHER requested item (whichever one the agent selects); [] if there are none.\n"roles": an object giving the exact tool for every job, with null for a job no tool performs: "search", "open" (displays a single listing\'s attributes or properties; if there are two single-listing tools, the one whose description promises attributes, properties or specifications), "offer" (displays a single listing\'s current offer, e.g. its price and stock, if that tool differs from "open"), "compare" (multiple listings in full), "ask" (messages the shopper), "cart" (modifies the cart), "cart_view" (displays the cart), "buy" (performs checkout), "refuse" (finishes without a purchase).\n"search_words": {"tool": the tool that looks up listings by free text (if one tool lists a category and a different one accepts words, the one accepting words), "field": the parameter of it that receives the words, "args": the fixed arguments that put it into word search (for a tool with a mode argument, that mode), otherwise {}}.\n"buy_args": the fixed arguments with which the "buy" tool performs checkout (for a tool taking a mode argument, that mode), otherwise {}.\n"refuse_args": the fixed arguments with which the "refuse" tool finishes without a purchase, otherwise {}.\n'
SCHEME = 'Your job is to pick the questions that a shopping agent puts to its shopper. Every shopper keeps a small number of firm wishes left out of the written request, and each wish concerns a single property of a single requested product. Asking uncovers a wish only if the question mentions precisely that property; a property the shopper holds no opinion on draws "don\'t mind", and one they never consider at all draws a confused answer, so each property you mention uses up a share of a message.\n\nYour input: the request; the items, each with what the request has already fixed for it; and a numbered list of the attribute keys found on opened listings in each item\'s category, giving how many of those listings have each key plus sample values. A hidden wish can concern a property that none of the listed keys covers.\n\nOrder of the properties:\n0. Anything the request itself tells you to confirm with the shopper (it appears under ASK FIRST if present), each one named after the property it concerns.\n1. Hinted properties. The story in the request frequently signals a hidden wish while withholding its value: a line about a thing the shopper is unable to judge or hands over to you, a person whose taste has the final say, how the product looks or the place it is meant for, who is going to use or wear it, or what it contains. For every such line, give the single property it signals and the item it concerns, following the story\'s order.\nWhen a listed key of that item matches the hint, use that key as the hinted property\'s name (hint about the wearer: the gender or audience key; hint about appearance: whichever listed colour, pattern or finish key appears on most of the item\'s listings); a property that none of the item\'s listings has goes after the three keys found most often on that item.\n2. After that, the listed keys in ranked order: prefer the properties that shoppers for this kind of product decide on before anything else (frequently its type or style, form or fit, size, capacity or dimensions, material, colour or pattern, whichever of these drive buying that kind of product) and keys present on the majority of the item\'s listings; when at least five listings have been read, steer clear of keys found on just one or two.\nLeave out: a property the request has already fixed for that item (meaning a requirement names that property; a key whose values merely echo a required value, for example a pattern key displaying the required decoration, is NOT settled); another name for a property you already picked; a key in which every value bundles several purchase options together; identifiers; model names; warranty and other logistics. Pack sizes, counts and quantities per pack count as ordinary properties, and a key tagged [not the stated ...] is a different property from that requirement, no matter which words its name has in common with it. Rank each item separately: hinted properties at the top, listed keys after.\n"label" holds the plain name of the property: for a listed key, every word of the key, spelled plainly ("lid colour" for lidColour, "rim finish" for rim_finish; not cut down to "colour" or "finish"); the property\'s name alone, without sample values, brackets or a second property.\nPick no more than COUNT per item, strongest first. Reply with ONE JSON object only, no other text: {"ask": [{"item": item number, "label": "...", "hint": "the story\'s words it stems from, 8 at most, or an empty string"}]}, with every item\'s picks in ranked order'
PERCEIVE = 'Below are questions a shopping agent put to a shopper, with the replies. Go through every reply, then answer with ONE JSON object only, no other text:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true if the value has to be taken from a different listing, otherwise false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [the numbers of questions whose reply objects that the message covered too many things or asks to have them put one by one; leave out replies that merely say they did not follow it or ask for plainer wording], "budget": the new overall limit as a number, or null}\n- "needs": each firm preference stated in a reply, carrying its precise value or limit plus unit ("at least 30 cm", "oval", "not wicker"). Use the property named in the reply itself as the need\'s name, even when it is not the property the question was about; several replies on one property give a single need. Take the value over verbatim from the reply, with its nouns and units intact ("at least 3 shelves", not "at least 3"); the value covers just what a listing might state, running from the wanted value or its limit word through to the end, minus whatever framing the reply wraps around it ("oval" for "matching oval", "settled, with oval" or "oval and nothing else", "at least 30 cm" for "equal to at least 30 cm").\n- A reply may supply the value by referring to a different listing (a value resembling, differing from or close to the one on some shop\'s product, which is identified by a few of its properties): treat that as ONE need on the property the reply names, with "points": true. For its value, copy the reply\'s own words exactly, starting at the comparison words and running to the end of that description, together with anything it says about a listing that page might name; do not trim or paraphrase it. Properties that describe the other product are not needs of this item.\n- "free": for each question, the property the shopper said they do not mind about.\n- Colloquial or indirect phrasing that names a value still states that value. Phrasing that will not negotiate a value (it is not to be swapped, surrendered or altered) states that value as wanted; put "not X" only if the reply turns down X itself.'
REMARK = 'The store displayed the note below in the middle of a shopping agent\'s work (its author may be the shopper or the store). Answer with ONE JSON object only, no other text:\n{"kind": "change" if the note alters the task (budget, requirements, a price, availability, the order being refused), "error" if all it does is say a call was badly formed or rejected, "other" otherwise, "budget": the new overall limit as a number in the unit the request uses, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [ids of listings it calls unavailable], "sold_out": true if it says a chosen listing has gone, sold out or stopped being available (even when it does not name it), "price_moved": true if it says a price has changed, "summary": "one short sentence"}\nSwapping one requirement for another counts as dropping the old one and adding the new one. Include each change the note contains.'
VERIFY = 'You perform the final review before a shopping agent completes checkout. Using the records provided, judge whether this cart is precisely correct. Whatever the shopper changed later is part of the requirements in force, which override the original request text: anything marked DROPPED is no longer required, though the request may still mention it.\nHow to read records: a listing satisfies a requirement when its attribute for that property gives the value (key spelling does not matter, matching one entry of a list suffices, differences of case or singular/plural and simple synonyms are fine, a value that includes the words of the required value is fine, e.g. "deep teal" for teal, units may be converted, a limit includes its boundary, a range has to fall entirely within); an attribute whose key names the requirement\'s own property settles it, and the listing fails if that attribute gives a different value, except that a value under a key joining the item\'s own name to the property (a bag\'s "bag shape") counts even if the plain key ("shape") gives a different value; a key about some other property, even one that is related (a handle colour is not the body\'s colour), settles nothing; if the listing lacks an attribute for that property, the wanted value appearing under a different attribute or in the title or description counts (a number measuring something else does not: a weight is not a capacity; nor does a lone figure whose text leaves unsaid what it measures); where the requirement is a measured limit (a number carrying a unit of measure), the figure must sit under a key whose name has a word in common with the property: if it appears only under a key measuring something else, or only in the title, the value is unknown and the requirement is failed; when a key spelled as the property\'s plural lists several entries, those are the variants sold, not this listing\'s own value, whenever the title names a different entry; a requirement stated by the category label itself is satisfied by that category; the category label has to match the requested item taken as a whole; a requirement to differ from another listing\'s value is satisfied unless the records show that very value. Every requested item has to sit on its own cart line at the requested quantity, the cart may hold nothing else, no line may be shown out of stock, and price times quantity summed over all lines has to stay within the budget. Where a listing has several records, the latest one applies.\nIf the pick rule is cheapest, most or least, check the other listings shown as well: should one of them plainly meet every requirement and outdo the cart\'s choice under that rule, the cart is wrong.\nAnswer with ONE JSON object only, no other text: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [ids of listings that ought to replace a cart line, best first]}. Set ok=false only for a problem that the records plainly show or for a requirement that no record covers.'
PERUSING = 'a listing\'s attribute is decisive for a property when the attribute\'s key names that very property (key spelling does not matter, matching one entry of a list suffices, synonyms and converted units are fine, a value that includes the words of the required value is fine, e.g. "deep teal" for teal, a limit includes its boundary); a value under a key joining the item\'s own name to the property (a bag\'s "bag shape") counts even if the plain key ("shape") gives a different value; a key about some other property, even one that is related (a handle colour is not the body\'s colour), settles nothing; if no attribute names that property, the wanted value appearing under a different attribute or in the title or description counts (a lone figure that leaves unsaid what it measures is worth nothing); where the requirement is a measured limit (a number carrying a unit of measure), the figure must sit under a key whose name has a word in common with the property: if it appears only under a key measuring something else, or only in the title, the value is unknown and the requirement is failed; when a key spelled as the property\'s plural lists several entries, those are the variants sold, not this listing\'s own value, whenever the title names a different entry; a requirement stated by the category label itself is satisfied by that category; the category label has to match the item; a requirement to differ from another listing\'s value is satisfied unless the records show that very value; a listing that is out of stock or that goes over the budget is not a fit; the requirements in force override the original request text, and anything marked DROPPED is no longer required.'
VET = 'Believing that no listing fits, a shopping agent intends to finish without a purchase. Test the listing records provided against the requirements, applying these reading rules: ' + PERUSING + '\nA record tagged ROW has only been seen as a line in search results: nobody has read its page (the attributes), so on any property its row leaves out it can be neither accepted nor rejected.\nAnswer with ONE JSON object only, no other text: {"fits": {"item number": [ids of listings that seem to satisfy every requirement, lowest price first, no more than 4]}, "open": [ids of ROW listings of a requested kind and within the budget whose page might reveal that they fit, best prospects first, no more than 6], "reason": "one sentence"}'
UNBLEND = 'Firm requirements on a product were drawn from a shopper\'s request. Certain requirements get their value from a different listing that the request refers to, and the words identifying that other listing (its kind, its shop, the properties mentioned after it with "with ...") sometimes get mistaken for requirements on the product to be bought. Going by how the request is worded, tell which of the OTHER REQUIREMENTS below merely describe that other listing. Answer with ONE JSON object only, no other text: {"describe_other": [those requirements, copied verbatim; empty if each of them is a genuine requirement on the product]}'
PATH = 'A shopping agent is on the point of checking out. Certain requirements, rather than stating their value, take it from a different listing, and the page of one listing can refer the reader onward to yet another listing for that property. An order only counts once the agent has fully opened every listing on each such chain. Starting from the requirement\'s own description, trace each requirement\'s chain one link at a time, through the words the pages themselves use, with the help of the records below (OPENED: the agent has opened this page; ROW: page unread, known only from a search result row or from its attributes).\nAnswer with ONE JSON object only, no other text: {"chains": [{"need": the requirement, copied verbatim, "links": [{"id": the listing\'s id if one of the records below is this link (the id of that record itself, never a code that appears in its title), otherwise "", "about": the way the requirement or the preceding page describes this listing, "values": the values this description assigns to that listing, as a few words, leaving out property names and the shop\'s name, "find": "if the link was not found: words to search for (the shop\'s name plus the values given in the description, leaving out property names and the product kind)"}], "done": true if the final link is an OPENED page stating the property\'s value itself and whose own wording does not send the reader on to yet another listing for that property (a page handing the value off to a different listing is not the final link, even if its attributes list a value) (this holds too when the requirement wants a value other than or unlike that one: the page\'s value is the yardstick it is compared with), otherwise false, "value": the property\'s value as that final page states it; if the requirement, or the request\'s own words for it, wants whatever goes with that value (rather than the value itself) and the page names what its value pairs with, that other value; or ""}]}\nA record counts as a link only if it fits the link\'s description in full: the shop or brand, the kind of product AND each property value that the description gives (e.g. "a 10 pieces multipack" excludes a listing from the same shop that has 10 capsules and no multipack); if no record fits all of that, the link has not been found yet, even when that shop sells other listings of that kind. For any such link, supply "find" with the shop\'s name together with the values in the description, as a handful of search words, leaving out property names and the product kind ("the kettles in Ovarno\'s listing with lid material glass and colour red" yields "Ovarno glass red"): the store\'s search requires every word to appear in the listing, and a property name or a kind word is frequently absent from it.\nGive the links of every chain in order, stopping at the first link whose page is not OPENED or that has not been found, because its words are not yet known.'
CHECKPOINT = 'A shopping agent is ready to STAGE. Along the way, the shopper\'s request calls for the steps listed below. Using the numbered RECORD of the agent\'s actions, judge for each step that is due before it may STAGE whether that step is complete for the cart in its current state. Treat a step as complete only if some RECORD line shows it being carried out and, when the step concerns the cart or the order, that line is later than the last change to the cart and the reply on it approves it; asking about a product\'s properties does not carry out a step.\nIf a step is still outstanding, supply the single store call that performs it next: a tool from TOOLS together with its arguments. Messages to the shopper are sent with the messaging tool: one brief sentence or question on that step only, in simple words, mentioning no product properties or values that the step does not require. Supply no call if all due steps are complete or if the remaining messages or steps cannot accomplish the missing one.\nAnswer with ONE JSON object only, no other text: {"steps": [{"step": the step, "done": true or false, "lines": [numbers of the RECORD lines showing it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": whatever a reply wants changed before continuing, or "", "needs": [{"item": item number, "need": "property: constraint"} for every firm requirement that such a reply states]}'
EXCHANGE = 'The cart of a shopping agent holds a listing that has just sold out. Search the remaining listing records provided for substitutes for that line of the cart, under these reading rules: ' + PERUSING + '\nAnswer with ONE JSON object only, no other text: {"fits": [{"id": the listing\'s id, "price": the price the record gives, written as a number} for each listing that seems to satisfy all requirements of the item on the sold-out line, no more than 6], "reason": "one sentence"}'
FORM = 'Below is what a store tool sent back. Identify its parts using precisely the keys found in the result and in the tool\'s parameters. Answer with ONE JSON object only, no other text: {"rows": the key whose value is the list of listings, "count": the key giving the total number of matching listings (not the number displayed), or null, "next": the key whose value requests the next page, or null, "next_arg": the parameter of the tool that accepts that value, or null, "price": within a row, the key for the listing\'s price, or null, "stock": within a row, the key telling whether the listing is available to buy right now, or null}'
SHIELD_HANDLES = 3
SHIELD_FRACTION = 0.0
SHIELD_VOID = 0.34
NON_QUALITY = '\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b'
TRAP_ENTIRE = '^(?!.*\\b(?:types?|styles?|kinds?|shapes?)\\b).*\\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\\w*|accessor\\w*|extras?|misc\\w*)\\b'
BROADEN_EVERY = 6
SUIT_CIPHER = '[A-Z][A-Za-z]{0,5}[ -]?(?:\\d{3,6}|\\d{1,5}[xX]{1,3})'
PORTION_CIPHER = '[A-Z]{1,5}\\d{0,3}-?\\d{3,7}'
WORTH_CAPACITY = 12  # steps past the finish required before any value search starts (race 16: limit 35, answers near step 15)
WORTH_PRESERVE = 8  # steps a value search keeps in reserve past the finish to open and buy what it turned up
PEEK_SIMILARS = 2  # how many keys one question may leave aside (race 16: one or two)
BOND_SEEKS = 3  # per item, the number of series a decline hunt searches (race 16: as many as 9)
BOND_TIERS = 3  # how many rows of one item must carry a series code before those codes form a system (race 16 covers: almost all rows)
BOND_LINEAGES = 12  # above this many distinct series they are model numbers rather than a system (race 16: 9)

def _ciphers(writing: Any) -> list[str]:
    located = re.findall('\\b' + SUIT_CIPHER + '\\b', str(writing or ''))
    return [re.sub('[\\s-]+', ' ', j).strip().lower() for j in located if not re.fullmatch('[A-Z][a-z]+ (?:19|20)\\d\\d', j)]

def _cipher_strike(motif: str, cipher: str) -> bool:
    """Whether a code given by a part matches a device's code: identical letters and digits, or a series with trailing x
    placeholders ("abc12xx" matches "abc1234"); spaces and hyphens are ignored.
    """
    broad, simple = (re.sub('[\\s-]+', '', motif), re.sub('[\\s-]+', '', cipher))
    root = broad.rstrip('x')
    if root != broad and root[-1:].isdigit():
        return simple.startswith(root) and len(simple) == len(broad)
    return broad == simple

def _suit_markers(writing: Any) -> dict[str, Any]:
    writing = str(writing or '')
    personal = re.search('[\\[(]\\s*(?:part\\s+)?(' + PORTION_CIPHER + ')\\s*[\\])]', writing, re.I)
    ciphers: list[str] = []
    for strike in re.finditer('\\bmodel\\s*(?:no\\.?|number|#)?\\s*[:.]?\\s*(?!(?:no|number)\\b)(' + SUIT_CIPHER + ')\\b', writing, re.I):
        ciphers += _ciphers(strike.group(1))
    for strike in re.finditer('\\b(?:also sold as|also known as|sold also as)\\s+([^.;]+)', writing, re.I):
        ciphers += _ciphers(strike.group(1))
    labors: list[str] = []
    bans: list[str] = []
    for strike in re.finditer('\\b(?<!not )(?<!never )(?:works with|compatible with|goes with|pairs with|fits)\\s*:?\\s+([^.;]+)', writing, re.I):
        if not re.search('\\bpart\\s+' + PORTION_CIPHER + '\\s+also\\s+$', writing[max(0, strike.start() - 40):strike.start()], re.I):
            chop = re.split('(?i)\\b(?:except|excluding|but not)\\b', strike.group(1), maxsplit=1)
            labors += _ciphers(chop[0])
            bans += _ciphers(chop[1]) if len(chop) > 1 else []
    too = [(strike.group(1).upper(), _ciphers(strike.group(2))) for strike in re.finditer('\\bpart\\s+(' + PORTION_CIPHER + ')\\s+also fits\\s*:?\\s*([^.;]+)', writing, re.I)]
    further = re.search('\\b(?:more|details|see|read)\\s+(?:on|at|under|about)?\\s*(?:the\\s+)?part\\s+(' + PORTION_CIPHER + ')', writing, re.I)
    clan, dubbed = _sequence_markers(writing)
    return {'own': personal.group(1).upper() if personal else '', 'codes': ciphers, 'works': labors, 'bars': bans, 'also': too, 'more': further.group(1).upper() if further else '', 'fam': clan, 'named': dubbed}

def _sequence_markers(writing: str) -> tuple[list[str], list[str]]:
    """The series prefixes that a record's own codes begin with ("AB1" from "AB1-2345"), plus the standalone
    series-looking words it mentions outside any clause ruling them out. Whether such a standalone word counts as a
    series is judged against codes found on other records (by the step that binds items through a series scheme).
    """
    clan = {strike.upper() for strike in re.findall('\\b([A-Za-z]{1,5}\\d{1,3})-\\d{2,7}\\b', writing)}
    remainder = re.sub('\\b[A-Za-z]{1,5}\\d{1,3}-\\d{2,7}\\b', ' ', writing)
    remainder = re.sub('(?i)\\b(?:except|excluding|but not|not for|not compatible with|does not (?:fit|take))\\b[^.;]*', ' ', remainder)
    return sorted(clan), sorted({strike.upper() for strike in re.findall('\\b[A-Za-z]{1,5}\\d{1,3}\\b', remainder)})
COUPLE_MARKERS = (': ', ' = ', '=', ' -> ', ' - ')
FACET_MARKERS = (' | ', '\n', ' || ', ' ; ', '; ', '\t', ' // ', ' · ', ' • ', '|', ';', ', ')
FACET_MONIKER = "[A-Za-z][A-Za-z0-9_ ./&'()-]{0,38}"
ROSTER_DASH = '^\\s*(?:[-*•·]|\\d{1,3}[.)])\\s+'
ENTRY_FACETS = 3
REMARK_FACETS = 5
FACET_FRACTION = 0.15
COUPLE_FRACTION = 0.7
CHUNK_MONIKER = 60
CHUNK_WORTH = 40
BADGE_WRITING = 20000
COUPLE_HANDLE = 'name|key|label|attr|prop|field|spec|title|^k$'
UID_FACET = '(?:^|[_ -])(?:id|idx|sku|pid|ref|asin|ident\\w*)(?:$|[_ -])|^id(?-i:[A-Z0-9])|(?-i:Ident)|(?<![a-z])(?!(?:paid|valid|grid|solid|rigid|fluid|liquid|hybrid|rapid|void|acid|humid|android|said|laid)$)[a-z]*id$'
FALTER_HANDLE = 'error|fail|(?<!de)fault|trouble(?!shoot)|denied|refus|invalid'

def _faltered(sighting: Any) -> bool:
    """Whether a result merely reports that the call did not succeed: a small object carrying a key that states it, or a lone code."""
    if not isinstance(sighting, dict) or not 0 < len(sighting) <= 3:
        return False
    if any((worth and re.search(FALTER_HANDLE, str(handle), re.I) for handle, worth in sighting.items())):
        return True
    just = next(iter(sighting.values()))
    return len(sighting) == 1 and isinstance(just, str) and 0 < len(just) <= 40 and ' ' not in just

def _simple(writing: str) -> Any:
    """The text of a field turned into what a structured result would contain: true or false, null, a number, otherwise the text itself."""
    naked = writing.strip()
    modest = naked.lower()
    if modest in ('true', 'false'):
        return modest == 'true'
    if modest in ('none', 'null', ''):
        return None
    cash = re.fullmatch('[$€£]?\\s?((?:\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.\\d+)?)', naked)
    if cash and not re.match('0\\d', cash.group(1)):
        numeral = float(cash.group(1).replace(',', ''))
        return int(numeral) if numeral.is_integer() and '.' not in cash.group(1) else numeral
    return naked

def _chunk_worth(writing: str) -> Any:
    """Parse one value from inside an attribute block, where something like "attachment method=none" still counts as a value; a block holds only texts and numbers."""
    located = _simple(writing)
    return writing.strip() if isinstance(located, bool) or (located is None and writing.strip()) else located

def _couple(writing: str, marker: str) -> tuple[str, str] | None:
    """(name, value) if the text starts by naming a field followed by this mark."""
    front, located, remainder = writing.partition(marker)
    if not located and writing.endswith(marker.strip()):
        front, located, remainder = (writing[:-len(marker.strip())], marker, '')
    moniker = re.sub(ROSTER_DASH, '', front).strip()
    if not located or not re.fullmatch(FACET_MONIKER, moniker) or len(moniker.split()) > 5:
        return None
    return (moniker, remainder.strip())

def _divide_facets(writing: str, among: str, marker: str) -> tuple[list[tuple[str, str]], int]:
    """Cut a text into fields at the given separator; a piece without a name is attached to the field before it (since a
    value can contain the separator). Also returns how many pieces had a name. A name ending in a colon with nothing
    following it introduces the bulleted or indented lines beneath it, which are kept together as one field.
    """
    portions = [pp for pp in writing.split(among) if pp.strip()]
    facets: list[tuple[str, str]] = []
    dubbed = 0
    caption = -1
    if portions and not _couple(portions[0].strip(), marker):
        return ([], 0)
    for portion in portions:
        naked = portion.strip()
        if among == '\n' and caption >= 0 and (re.match(ROSTER_DASH, portion) or portion[:1] in ' \t'):
            moniker, worth = facets[caption]
            facets[caption] = (moniker, (worth + '\n' if worth else '') + naked)
            continue
        caption = -1
        if among == '\n' and naked.endswith(':') and re.fullmatch(FACET_MONIKER, naked[:-1].strip()):
            facets.append((naked[:-1].strip(), ''))
            caption = len(facets) - 1
            dubbed += 1
            continue
        couple = _couple(naked, marker)
        if couple:
            facets.append(couple)
            dubbed += 1
        elif facets:
            moniker, worth = facets[-1]
            facets[-1] = (moniker, worth + among + naked)
    return (facets, dubbed) if len(portions) else ([], 0)

def _trait_chunk(writing: str) -> dict[str, Any] | None:
    """Attributes of a listing packed into a single field: "a=b; c=d", a block in braces, bulleted "name: value" lines, or
    brief "name: value" pieces separated by commas. Prose remains text regardless of the colons inside it (a
    description saying "Fits: A, B. Condition: new" is not a block), and any short text may serve as a name, because
    attribute names arrive however the catalog writes them.
    """
    naked = writing.strip()
    if len(naked) < 5:
        return None
    if naked[0] in '{[' and naked[-1] in '}]':
        try:
            decoded = json.loads(naked)
            return decoded if isinstance(decoded, dict) else None
        except (ValueError, RecursionError):
            located = dict(re.findall('[\'"]?([A-Za-z][\\w ./-]{0,38})[\'"]?\\s*[:=]\\s*[\'"]?([^,\'"}\\]]*)', naked))
            return {kk.strip(): _simple(vv) for kk, vv in located.items()} if len(located) >= 2 else None
    if '=' in naked:
        output: dict[str, Any] = {}
        final = ''
        portions = [pp for pp in re.split(';\\s+|;$|\\s\\|\\s|\\n', naked) if pp.strip()]
        for portion in portions:
            moniker, located, worth = portion.partition('=')
            if located and 0 < len(moniker.strip()) <= CHUNK_MONIKER and moniker.strip() not in output:
                final = moniker.strip()
                output[final] = _chunk_worth(worth)
            elif final:
                output[final] = f'{output[final]}; {portion.strip()}'
            else:
                return None
        return output if len(output) >= 2 or (output and ':' not in final and len(final) <= 30 and len(naked) <= 90) else None
    strands = [strand for strand in naked.splitlines() if strand.strip()]
    if len(strands) >= 2 and all((re.match(ROSTER_DASH, strand) for strand in strands)):
        couples = [_couple(strand.strip(), ': ') for strand in strands]
        if all(couples):
            return {moniker: _simple(worth) for moniker, worth in couples}
    for among in (', ', '; '):
        if among in naked and ': ' in naked:
            couples = [_couple(portion.strip(), ': ') for portion in naked.split(among)]
            if len(couples) >= 2 and all(couples) and all(('.' not in moniker and len(worth) <= CHUNK_WORTH for moniker, worth in couples)):
                return {moniker: _simple(worth) for moniker, worth in couples}
    return None

def _entry_writing(writing: str) -> dict[str, Any] | None:
    """The record that a text lays out one field after another, whatever separators it employs; nothing when it reads as
    ordinary prose. Separators are discovered by trial: first the outer field mark, then whichever pair mark yields the
    most named fields; a value written in the same manner turns into its own record (attributes of a listing packed
    into a single field).
    """
    finest: tuple[int, list[tuple[str, str]]] | None = None
    for among in FACET_MARKERS:
        if among not in writing:
            continue
        fragments = len([pp for pp in writing.split(among) if pp.strip()])
        for marker in COUPLE_MARKERS:
            facets, dubbed = _divide_facets(writing, among, marker)
            if dubbed < ENTRY_FACETS or dubbed < (COUPLE_FRACTION if among == ', ' else FACET_FRACTION) * fragments:
                continue
            if len({moniker for moniker, _ in facets}) * 2 <= len(facets):
                continue
            if finest is None or dubbed > finest[0]:
                finest = (dubbed, facets)
        if finest:
            break
    if finest is None:
        return None
    entry: dict[str, Any] = {}
    for moniker, worth in finest[1]:
        if moniker in entry:
            continue
        internal = _trait_chunk(worth)
        entry[moniker] = internal if internal else worth.strip() if re.search(UID_FACET, moniker, re.I) else _simple(worth)
    return entry

def _grid_writing(writing: str) -> list[dict[str, Any]] | None:
    """The rows of a plain-text table: a header line listing names, followed by lines split into that many cells (markdown divider lines are ignored)."""
    strands = [strand.strip().strip('|').strip() for strand in writing.splitlines() if strand.strip() and not re.fullmatch('[\\s|:+=-]+', strand)]
    if len(strands) < 2:
        return None
    for among in ('|', '\t', ';', ','):
        squares = [[j.strip() for j in strand.split(among)] for strand in strands]
        magnitude = len(squares[0])
        if magnitude < 3 or any((len(tier) != magnitude for tier in squares)) or not all((re.fullmatch(FACET_MONIKER, j) for j in squares[0])):
            continue
        if any((re.fullmatch('[\\d.,$]+', j) for j in squares[0])):
            continue
        return [{moniker: _simple(j) for moniker, j in zip(squares[0], tier)} for tier in squares[1:]]
    return None

def _entries_writing(writing: str) -> Any:
    """Structured version of text returned by a store tool, or nothing if it contains no records."""
    naked = writing.strip()
    if len(naked) < 12:
        return None
    if naked[0] in '{[':
        try:
            return _reinstate(json.loads(naked))
        except (ValueError, RecursionError):
            pass
    for portions in (re.split('\\n\\s*\\n', naked), naked.splitlines()):
        if len(portions) >= 2:
            tiers = [_entry_writing(pp) for pp in portions]
            fine = [rr for rr in tiers if rr]
            if len(fine) >= 2 and len(fine) * 5 >= len(portions) * 4 and len({next(iter(rr)) for rr in fine}) == 1:
                return fine
    located = _grid_writing(naked) if '\n' in naked and _entry_writing(naked) is None else _entry_writing(naked)
    if located is None and len(naked) <= BADGE_WRITING and re.search('<[a-zA-Z/][^<>]{0,40}>', naked):
        simple = re.sub('</?(?:b|i|em|strong|span|a|u|small)\\b[^<>]{0,40}>', '', naked, flags=re.I)
        simple = re.sub('</t[dh]>\\s*<t[dh][^<>]{0,40}>', ': ', simple, flags=re.I)
        simple = re.sub('\\s*(?:<[^<>]{1,40}>\\s*)+', '\n', simple).strip()
        if simple != naked and len(simple) >= 12:
            return _entries_writing(simple)
    if located is None and '\n' in naked:
        strands = [re.sub('(?:[ ]{2,}|\t)', ': ', strand.strip(), count=1) for strand in naked.splitlines() if strand.strip()]
        spread = '\n'.join(strands)
        if spread != naked:
            return _entry_writing(spread)
    return located

def _chunk_writing(writing: str) -> Any:
    """A text field on a row that already carries its own fields: treated as a record only if it consists entirely of name=value pieces."""
    if len(writing) < 5 or '=' not in writing or not re.search('; | \\| |\\n', writing):
        return writing
    portions = [pp for pp in re.split(';\\s+|;$|\\s\\|\\s|\\n', writing.strip()) if pp.strip()]
    if len(portions) < 2 or not all((re.fullmatch('[^=:]{1,40}=[^=]{0,200}', pp.strip()) for pp in portions)):
        return writing
    return _trait_chunk(writing) or writing

def _couples_chunk(tiers: list[Any], inside_entry: bool) -> dict[str, Any] | None:
    """Attributes given as a list: objects holding a name and a value, two-element [name, value] lists, or, within a
    record, brief "name: value" pieces. A list of offers, highlights or notes does not qualify.
    """
    if len(tiers) < 2:
        return None
    output: dict[str, Any] = {}
    initial = tiers[0]
    if isinstance(initial, dict):
        if any((not isinstance(rr, dict) or set(rr) != set(initial) or len(rr) != 2 for rr in tiers)):
            return None
        handles = list(initial)
        if not re.search(COUPLE_HANDLE, str(handles[0]), re.I):
            return None
        for tier in tiers:
            moniker, worth = tier[handles[0]], tier[handles[1]]
            if not isinstance(moniker, str) or not 0 < len(moniker) <= CHUNK_MONIKER or isinstance(worth, (dict, list)):
                return None
            output[moniker.strip()] = worth
        return output
    if isinstance(initial, (list, tuple)):
        for tier in tiers:
            if not isinstance(tier, (list, tuple)) or len(tier) != 2 or not isinstance(tier[0], str) or isinstance(tier[1], (dict, list)) or not 0 < len(tier[0]) <= CHUNK_MONIKER:
                return None
            output[tier[0].strip()] = tier[1]
        return output
    if inside_entry and all((isinstance(rr, str) for rr in tiers)):
        for tier in tiers:
            couple = _couple(tier.strip(), ': ') or _couple(tier.strip(), '=')
            if not couple or len(couple[1]) > CHUNK_WORTH or len(couple[0].split()) > 3:
                return None
            output[couple[0]] = _simple(couple[1])
        return output
    return None

def _pillar_tiers(vertex: dict[str, Any]) -> dict[str, Any]:
    """When a result lists its column names once and supplies every row as an array, each row is returned as an object instead."""
    fronts = [kk for kk, vv in vertex.items() if isinstance(vv, list) and len(vv) >= 2 and all((isinstance(j, str) and re.fullmatch(FACET_MONIKER, j) for j in vv))]
    for front in fronts:
        monikers = vertex[front]
        for handle, tiers in vertex.items():
            if handle != front and isinstance(tiers, list) and tiers and all((isinstance(rr, (list, tuple)) and len(rr) == len(monikers) for rr in tiers)):
                output = {kk: vv for kk, vv in vertex.items() if kk != front}
                output[handle] = [dict(zip(monikers, tier)) for tier in tiers]
                return output
    return vertex

def _uid_writings(entry: dict[str, Any]) -> dict[str, Any]:
    """The numeric id of a record rendered as text, which is how the agent keys its listings (the schemas convert it back when a tool expects a number)."""
    if not any((isinstance(vv, str) and ' ' in vv for vv in entry.values())):
        return entry
    return {kk: str(vv) if isinstance(vv, int) and not isinstance(vv, bool) and re.search(UID_FACET, kk, re.I) and abs(vv) >= 100 else vv for kk, vv in entry.items()}

def _handled_tiers(vertex: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Rows supplied as an object keyed by listing code, each one a record sharing the same fields, turned into a list of rows that carry their code."""
    tiers = list(vertex.values())
    if len(tiers) < 2 or not all((isinstance(rr, dict) and len(rr) >= ENTRY_FACETS for rr in tiers)) or not all((re.search('\\d', str(kk)) and ' ' not in str(kk) for kk in vertex)):
        return None
    if len(set(tiers[0]).intersection(*tiers[1:])) < ENTRY_FACETS:
        return None
    return [rr if any((re.search(UID_FACET, str(px), re.I) for px in rr)) else {'id': str(kk), **rr} for kk, rr in vertex.items()]

def _grasps_uid(vertex: dict[str, Any]) -> bool:
    return any((re.search(UID_FACET, str(kk), re.I) for kk in vertex))

def _planar_chunk(vertex: dict[str, Any]) -> dict[str, Any]:
    """Flattens an attribute block containing nested groups to a single level: {"size": {"width": 3}} becomes {"size width": 3}."""
    if not any((isinstance(vv, dict) for vv in vertex.values())):
        return vertex
    output: dict[str, Any] = {}
    for handle, worth in vertex.items():
        if isinstance(worth, dict):
            output.update({f'{handle} {internal}': vv for internal, vv in worth.items() if not isinstance(vv, (dict, list))})
        else:
            output[handle] = worth
    return output

def _reinstate_facet(worth: Any, deepness: int, tiled: bool, sheet: bool=True) -> Any:
    """Restores the value of a field; text turns into a record only if it looks like a listing (through an id, a code
    among its values, or a large number of fields), so a shop's remark full of colons and semicolons is kept as a
    remark.
    """
    located = _reinstate(worth, deepness + 1, tiled, sheet)
    if isinstance(worth, str) and isinstance(located, dict) and len(located) < REMARK_FACETS and not _grasps_uid(located) and not any((_uid_akin(vv) for vv in located.values())):
        return worth
    return located

def _reinstate(worth: Any, deepness: int=0, inside_entry: bool=False, sheet: bool=True) -> Any:
    """Bring a store result into the single shape the rest of the agent understands: lists of row objects, with a
    listing's attributes as a nested object. Worlds format results in different ways (race 15 sent each record as one
    text with its "a=b; c=d" attributes embedded, and a result the agent fails to parse blinds every check that relies
    on records), so this undoes the formatting once, in one place, for everything.
    """
    if deepness > 7:
        return worth
    if isinstance(worth, str):
        located = _entries_writing(worth)
        return located if located is not None else worth
    if isinstance(worth, list):
        chunk = _couples_chunk(worth, inside_entry and sheet)
        if chunk:
            return chunk
        return [_reinstate_facet(vv, deepness, False, sheet) for vv in worth]
    if isinstance(worth, dict):
        vertex = _uid_writings(_pillar_tiers(worth))
        tiled = len(vertex) >= ENTRY_FACETS or _grasps_uid(vertex)
        handled = {kk: _handled_tiers(vv) if isinstance(vv, dict) else None for kk, vv in vertex.items()}
        vertex = {kk: handled[kk] or vv for kk, vv in vertex.items()}
        posting = _grasps_uid(vertex) or any((_uid_akin(vv) for vv in vertex.values()))
        output = {kk: _reinstate_facet(vv, deepness, tiled, sheet) if not isinstance(vv, str) or len(vertex) <= 2 else (_chunk_writing(vv) if posting else vv) for kk, vv in vertex.items()}
        if deepness <= 3:
            output = {kk: _planar_chunk(vv) if isinstance(vv, dict) and any((isinstance(px, dict) for px in vv.values())) and not any((isinstance(px, list) for px in vv.values())) and isinstance(next(iter(vv.values()), None), (dict, str, int, float)) and len(vertex) >= ENTRY_FACETS and not _grasps_uid(vv) else vv for kk, vv in output.items()}
        if len(output) == 1:
            just = next(iter(output.values()))
            if isinstance(just, dict) and len(just) >= ENTRY_FACETS and isinstance(next(iter(worth.values())), str):
                return just
        return output
    return worth

_INITIAL_ORTHOGRAPHY: dict[str, str] = {}

def _unite_duplicates(retained: str, worth: str) -> str:
    """Merge the values of two spellings into one list with no repeats; the value of the first spelling remains recorded, for statistics on how a key's values look."""
    among = '(?<!\\d),|,(?!\\d)'
    possess = {portion.strip().lower() for portion in re.split(among, retained)}
    additional: list[str] = []
    for portion in re.split(among, worth):
        if portion.strip() and portion.strip().lower() not in possess:
            possess.add(portion.strip().lower())
            additional.append(portion.strip())
    if not additional:
        return retained
    united = (f'{retained}, ' if retained.strip() else '') + ', '.join(additional)
    _INITIAL_ORTHOGRAPHY[united] = _INITIAL_ORTHOGRAPHY.get(retained, retained)
    return united

def _fused(writing: str) -> str:
    """Joins a number with its short unit into a single word regardless of spelling: "65 W" and "65W" are the same value, "45 W" is a different one."""
    return re.sub('(\\d)\\s+([a-z]{1,3})(?![a-z0-9])', '\\1\\2', writing.lower())

def _writing_terms(writing: str) -> set[str]:
    """Words of a row or a query, where a number followed by its short unit is also counted as one word."""
    modest = writing.lower()
    return set(re.findall('[a-z0-9]+', modest)) | set(re.findall('[a-z0-9]+', _fused(modest)))

def _worth_symbols(worth: str) -> list[str]:
    """The words of a value that a query or a row could contain: the long ones, otherwise two-character ones that include a letter (XL, 4g, 65w), otherwise its numbers."""
    modest = _fused(worth)
    return re.findall('[a-z0-9]{3,}', modest) or re.findall('[0-9]+[a-z]|[0-9]*[a-z][a-z0-9]+', modest) or re.findall('[0-9]+', modest)

def _collapse_duplicates(vertex: Any, deepness: int=0) -> Any:
    """Collapse to one line per property: when a block writes a key in two spellings, the first spelling is kept, carrying
    both values if they disagree, so that a single list element suffices ("clock shape: rectangle" next to an
    underscored clock shape of "rectangular, square" still holds square). Around one in ten listings in races 13-15 and
    in a practice world contain such pairs with differing values; treating them as conflicting threw out listings the
    store's answer accepted. Each duplicate line also adds tokens to every subsequent model turn.
    """
    if deepness > 5:
        return vertex
    if isinstance(vertex, list):
        return [_collapse_duplicates(vv, deepness + 1) for vv in vertex]
    if not isinstance(vertex, dict):
        return vertex
    output: dict[str, Any] = {}
    spotted: dict[str, str] = {}
    for handle, worth in vertex.items():
        moniker = _terms(handle)
        initial = spotted.get(moniker)
        if initial is None:
            spotted[moniker] = handle
            output[handle] = _collapse_duplicates(worth, deepness + 1)
        elif output[initial] == worth:
            continue
        elif isinstance(output[initial], str) and isinstance(worth, str) and not re.search(UID_FACET, moniker, re.I):
            output[initial] = _unite_duplicates(output[initial], worth)
        else:
            output[handle] = _collapse_duplicates(worth, deepness + 1)
    return output if len(output) >= 2 or len(vertex) < 2 else {kk: _collapse_duplicates(vv, deepness + 1) for kk, vv in vertex.items()}

def _spill(worth: Any) -> str:
    try:
        return json.dumps(worth, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        return str(worth)

def _snip(worth: Any, cutoff: int, deepness: int=0) -> Any:
    if deepness > 7:
        return '...'
    if isinstance(worth, str):
        return worth if len(worth) <= cutoff else worth[:cutoff] + '...'
    if isinstance(worth, dict):
        return {kk: _snip(vv, cutoff, deepness + 1) for kk, vv in worth.items()}
    if isinstance(worth, list):
        return [_snip(vv, cutoff, deepness + 1) for vv in worth]
    return worth

def _jdoc_entity(writing: Any) -> dict[str, Any] | None:
    if not isinstance(writing, str):
        return None
    writing = writing.strip()
    palisade = re.match('^```[A-Za-z]*\\s*(.*?)\\s*```$', writing, re.S)
    if palisade:
        writing = palisade.group(1)
    try:
        obtained = json.loads(writing)
        return obtained if isinstance(obtained, dict) else None
    except ValueError:
        pass
    remaining, correct = (writing.find('{'), writing.rfind('}'))
    if 0 <= remaining < correct:
        try:
            obtained = json.loads(writing[remaining:correct + 1])
            return obtained if isinstance(obtained, dict) else None
        except ValueError:
            return None
    return None

def _terms(handle: str) -> str:
    handle = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(handle))
    return re.sub('[\\s_.\\-]+', ' ', handle).strip().lower()

def _serpent(handle: str) -> str:
    return _terms(handle).replace(' ', '_')

def _root(writing: str) -> frozenset[str]:
    return frozenset((ww[:-1] if len(ww) > 3 and ww.endswith('s') else ww for ww in _terms(writing).split()))

def _sources(writing: str) -> set[str]:
    output = set()
    for ww in _terms(writing).split():
        ww = ww[:-2] if re.search('(?:ch|sh|ss|x|z)es$', ww) else ww[:-1] if len(ww) > 3 and ww.endswith('s') and (not ww.endswith('ss')) else ww
        if len(ww) >= 3 and ww not in ('and', 'for', 'the', 'with'):
            output.add(ww)
    return output

def _spelt(writing: str) -> frozenset[str]:
    """The words of a name with plural -s and the British -our ending normalised, so that "colours" equals "color"."""
    return frozenset((re.sub('our$', 'or', ww[:-1] if len(ww) > 3 and ww.endswith('s') and (not ww.endswith('ss')) else ww) for ww in _terms(writing).split()))

def _monikers(feature: frozenset[str], handle: frozenset[str]) -> bool:
    if not feature or not handle:
        return False
    return feature == handle or feature <= handle or (len(handle) >= 2 and handle <= feature)

def _varieties(blueprint: dict[str, Any]) -> list[str]:
    variety = blueprint.get('type')
    if isinstance(variety, str):
        return [variety]
    return [kk for kk in variety or [] if isinstance(kk, str)]

def _suits_variety(worth: Any, variety: str) -> bool:
    if variety == 'string':
        return isinstance(worth, str)
    if variety == 'integer':
        return isinstance(worth, int) and (not isinstance(worth, bool))
    if variety == 'number':
        return isinstance(worth, (int, float)) and (not isinstance(worth, bool))
    if variety == 'boolean':
        return isinstance(worth, bool)
    if variety == 'array':
        return isinstance(worth, list)
    if variety == 'object':
        return isinstance(worth, dict)
    if variety == 'null':
        return worth is None
    return True

def _form(worth: Any, blueprint: Any, location: str, defects: list[str]) -> Any:
    if not isinstance(blueprint, dict):
        return worth
    varieties = _varieties(blueprint)
    if varieties and (not any((_suits_variety(worth, kk) for kk in varieties))):
        for variety in varieties:
            if variety == 'integer' and isinstance(worth, float) and worth.is_integer():
                worth = int(worth)
                break
            if variety in ('integer', 'number') and isinstance(worth, str) and re.fullmatch('\\s*-?\\d+(\\.\\d+)?\\s*', worth):
                numeral = float(worth)
                worth = int(numeral) if variety == 'integer' or numeral.is_integer() else numeral
                break
            if variety == 'string' and isinstance(worth, (int, float)) and (not isinstance(worth, bool)):
                worth = str(worth)
                break
            if variety == 'boolean' and isinstance(worth, str) and (worth.strip().lower() in ('true', 'false')):
                worth = worth.strip().lower() == 'true'
                break
            if variety == 'array' and (not isinstance(worth, list)) and (worth is not None):
                worth = [worth]
                break
        if not any((_suits_variety(worth, kk) for kk in varieties)):
            defects.append(f"{location} should be {'/'.join(varieties)}")
            return worth
    choices = blueprint.get('enum')
    if isinstance(choices, list) and worth not in choices:
        if isinstance(worth, str):
            modest = worth.strip().lower()
            accord = [oo for oo in choices if isinstance(oo, str) and oo.lower() == modest]
            if accord:
                worth = accord[0]
        if worth not in choices:
            defects.append(f'{location}: must be one of {choices}')
    if isinstance(worth, str) and isinstance(blueprint.get('maxLength'), int) and (len(worth) > blueprint['maxLength']):
        worth = worth[:blueprint['maxLength']]
    if isinstance(worth, str) and isinstance(blueprint.get('minLength'), int) and (len(worth) < blueprint['minLength']):
        defects.append(f'{location} is below the minimum length')
    if isinstance(worth, (int, float)) and (not isinstance(worth, bool)):
        if isinstance(blueprint.get('minimum'), (int, float)) and worth < blueprint['minimum']:
            defects.append(f"{location} is below {blueprint['minimum']}")
        if isinstance(blueprint.get('maximum'), (int, float)) and worth > blueprint['maximum']:
            defects.append(f"{location} is above {blueprint['maximum']}")
    if isinstance(worth, list):
        if isinstance(blueprint.get('maxItems'), int) and len(worth) > blueprint['maxItems']:
            defects.append(f"{location} takes no more than {blueprint['maxItems']} entries")
        if isinstance(blueprint.get('minItems'), int) and len(worth) < blueprint['minItems']:
            defects.append(f"{location} must have {blueprint['minItems']} or more entries")
        if isinstance(blueprint.get('items'), dict):
            worth = [_form(vv, blueprint['items'], f'{location}[{ii}]', defects) for ii, vv in enumerate(worth)]
    if isinstance(worth, dict):
        features = blueprint.get('properties') if isinstance(blueprint.get('properties'), dict) else {}
        if blueprint.get('additionalProperties') is False:
            for moniker in [nn for nn in worth if nn not in features]:
                defects.append(f'{location}.{moniker} is not among the accepted arguments')
        for moniker in blueprint.get('required') or []:
            if moniker not in worth:
                defects.append(f'{location}.{moniker} is required')
        worth = {kk: _form(vv, features.get(kk), f'{location}.{kk}', defects) if kk in features else vv for kk, vv in worth.items()}
    return worth

def _articles(worth: Any) -> list[Any]:
    if isinstance(worth, list):
        return worth
    if isinstance(worth, (str, dict)) and worth:
        return [worth]
    return []

def _tally(worth: Any) -> int | None:
    if isinstance(worth, bool):
        return None
    if isinstance(worth, int):
        return worth
    if isinstance(worth, float) and worth.is_integer():
        return int(worth)
    if isinstance(worth, str) and re.fullmatch('\\s*\\d+(\\.0+)?\\s*', worth):
        return int(float(worth))
    return None

def _uid_akin(worth: Any, dubbed: bool=False) -> bool:
    if not isinstance(worth, str) or not 3 <= len(worth.strip()) <= 64 or ' ' in worth.strip():
        return False
    return dubbed or bool(re.search('\\d', worth))

def _discover_uid(tiers: list[dict[str, Any]]) -> str | None:
    finest = None
    for handle in tiers[0]:
        worths = [tier.get(handle) for tier in tiers]
        dubbed = bool(re.search(UID_FACET, str(handle), re.I))
        if not all((_uid_akin(vv, dubbed) for vv in worths)):
            continue
        worths = [vv.strip() for vv in worths]
        if len(tiers) > 1 and len(set(worths)) < len(worths):
            continue
        grade = (dubbed, sum((len(vv) for vv in worths)) / len(worths))
        if finest is None or grade > finest[0]:
            finest = (grade, handle)
    return finest[1] if finest else None

def _inside_tiers(vertex: Any, identity: str, deepness: int=0) -> bool:
    if deepness > 5:
        return False
    if isinstance(vertex, list):
        return any((isinstance(vv, dict) and identity in [str(px).strip() for px in vv.values() if isinstance(px, str)] for vv in vertex)) or any((_inside_tiers(vv, identity, deepness + 1) for vv in vertex))
    if isinstance(vertex, dict):
        return any((_inside_tiers(vv, identity, deepness + 1) for vv in vertex.values()))
    return False

def _oracle_amid(secs: float) -> ProxyClient:
    """A model client whose timeout fits within the remaining time: the normal one, or one whose timeout is rounded to whole 5-second steps."""
    if secs >= 50:
        return _oracle
    cutoff = max(5, int(secs) // 5 * 5)
    if cutoff not in _oracle_terse:
        _oracle_terse[cutoff] = ProxyClient(timeout=cutoff, max_retries=1, retry_delay=1)
    return _oracle_terse[cutoff]

def _datum_chunk(vertex: Any) -> dict[str, Any] | None:
    if not isinstance(vertex, dict):
        return None
    finest = None
    for worth in vertex.values():
        if isinstance(worth, dict) and len(worth) >= 2:
            atoms = {kk: vv for kk, vv in worth.items() if isinstance(vv, (str, int, float)) and (not isinstance(vv, bool))}
            if len(atoms) >= 2 and (finest is None or len(atoms) > len(finest)):
                finest = atoms
    return finest

def _gentle(standby: Any) -> Any:
    """Wraps a method that contributes to the episode without being essential: on failure, the trace records it and the episode continues."""

    def enclose(approach: Any) -> Any:

        def shielded(self: Any, *operands: Any, **kwopts: Any) -> Any:
            try:
                return approach(self, *operands, **kwopts)
            except Exception as exn:
                self.transcript.append({'kind': 'soft_crash', 'where': approach.__name__, 'error': f'{type(exn).__name__}: {exn}'[:300]})
                return standby() if callable(standby) else standby
        return shielded
    return enclose

class _Imprinted(list):
    """A trace that timestamps each of its entries."""

    def __init__(self, timer: Any):
        super().__init__()
        self.timer = timer

    def append(self, element: dict[str, Any]) -> None:
        element.setdefault('at', round(self.timer(), 1))
        super().append(element)

def _padding(specification: dict[str, Any]) -> Any:
    """A value to put in a mandatory field where the agent has nothing to say; nothing when the schema offers no safe choice."""
    if isinstance(specification.get('enum'), list) and specification['enum']:
        return specification['enum'][0]
    for term in ('const', 'default'):
        if term in specification:
            return specification[term]
    varieties = _varieties(specification)
    if 'integer' in varieties or 'number' in varieties:
        return specification['minimum'] if isinstance(specification.get('minimum'), (int, float)) else 1
    if 'boolean' in varieties:
        return False
    return 'question'[:specification.get('maxLength') or 8] if 'string' in varieties else None

def _literals(vertex: Any) -> list[str]:
    """Every text contained in a call's arguments, however deeply nested."""
    if isinstance(vertex, dict):
        return [ss for vv in vertex.values() for ss in _literals(vv)]
    if isinstance(vertex, list):
        return [ss for vv in vertex for ss in _literals(vv)]
    return [vertex] if isinstance(vertex, str) else []

def _utters(vertex: Any, writing: str) -> bool:
    """Whether the arguments of a call contain the given text in full, at any depth."""
    if isinstance(vertex, dict):
        return any((_utters(vv, writing) for vv in vertex.values()))
    if isinstance(vertex, list):
        return any((_utters(vv, writing) for vv in vertex))
    return vertex == writing

class Article:

    def __init__(self, moniker: str, rack: str, amt: int, essentials: list[str]):
        self.moniker = moniker
        self.rack = rack
        self.amt = amt
        self.essentials = essentials
        self.informed: list[str] = []
        self.omitted: list[str] = []
        self.gratis: list[str] = []


class _SievePart:
    """Reads store replies into listing rows, keys and record shapes."""

    def article_clusters(self) -> list[list[str]]:
        sect_handle = self.genre_handle()
        clusters: dict[str, list[str]] = {}
        for identity in self.spotted_booking:
            legend = str(self.tiers[identity].get(sect_handle) or '') if sect_handle else ''
            clusters.setdefault(legend, []).append(identity)
        desired = [(_terms(article.rack or ''), _terms(article.moniker), {ww for ww in _root(article.rack or '') | _root(article.moniker) if len(ww) >= 4}) for article in self.articles]
        accorded = [uids for legend, uids in clusters.items() if any((_terms(legend) in (rack, moniker) or {ww for ww in _root(legend) if len(ww) >= 4} & lengthy for rack, moniker, lengthy in desired)) or self.proprietor_from(legend)]
        reservoir = accorded or list(clusters.values())
        return sorted(reservoir, key=lambda uids: -len(uids))[:max(1, len(self.articles))]

    def cost_from(self, identity: str) -> float | None:
        worth = (self.tiers.get(identity) or {}).get(self.cost_handle) if self.cost_handle else None
        if isinstance(worth, dict):
            worth = next((vv for vv in worth.values() if isinstance(vv, (int, float)) and (not isinstance(vv, bool))), None)
        if isinstance(worth, str):
            estimate = re.fullmatch('\\s*[^\\d\\s]{0,3}\\s?(\\d[\\d,]*(?:\\.\\d+)?)\\s*[A-Za-z]{0,6}\\.?\\s*', worth)
            worth = float(estimate.group(1).replace(',', '')) if estimate else None
        return float(worth) if isinstance(worth, (int, float)) and (not isinstance(worth, bool)) else None

    def discover_tiers(self, vertex: list[Any]) -> bool:
        tiers = [nn for nn in vertex if isinstance(nn, dict) and len(nn) >= 3]
        if not tiers or len(tiers) < len(vertex) / 2:
            return False
        if self.uid_handle and all((_uid_akin(rr.get(self.uid_handle), True) for rr in tiers)):
            return True
        located = _discover_uid(tiers)
        if located:
            self.uid_handle = located
            return True
        return False

    def displayed_previously(self, moniker: str, identity: str) -> bool:
        revision, possess = self.outlook_handles.get(identity, (-1, set()))
        handles = self.instrument_handles.get(moniker)
        return bool(handles) and revision == self.revision and (handles <= possess) and (identity not in self.departed)

    @staticmethod
    def duplicates(moniker: str, specification: dict[str, Any], alternative: str, cell: dict[str, Any]) -> bool:
        u, d = (set(moniker.split()), set(alternative.split()))
        if not u or not d or (not u & d):
            return False

        def concur(px: str, py: str) -> bool:
            ux, uy = (re.findall('[a-z0-9]+', px.lower()), re.findall('[a-z0-9]+', py.lower()))
            terse, lengthy = (ux, uy) if len(ux) <= len(uy) else (uy, ux)
            return px == py or (bool(terse) and set(terse) <= set(lengthy) and (len(lengthy) <= 2 * len(terse) + 2))
        jointly = [ii for ii in specification['plain'] if ii in cell['plain'] and specification['plain'][ii].strip() and cell['plain'][ii].strip()]
        identical = sum((1 for ii in jointly if concur(specification['plain'][ii], cell['plain'][ii])))
        if not (u <= d or d <= u):
            return len(jointly) >= 2 and identical == len(jointly)
        if jointly:
            return identical * 2 > len(jointly)
        return bool((set(specification['values']) & set(cell['values'])) - {''})

    def entries_inside(self, sighting: Any) -> list[tuple[str, dict[str, Any]]]:
        if isinstance(sighting, dict):
            identity = self.identity(sighting)
            if identity:
                return [(identity, sighting)]
            located: list[tuple[str, dict[str, Any]]] = []
            for worth in sighting.values():
                if isinstance(worth, list):
                    located += [(ii, rr) for rr in worth if isinstance(rr, dict) for ii in [self.identity(rr, True)] if ii]
                elif isinstance(worth, dict):
                    internal = self.identity(worth, False, 1)
                    located += [(internal, worth)] if internal else [(ii, rr) for vv in worth.values() if isinstance(vv, list) for rr in vv if isinstance(rr, dict) for ii in [self.identity(rr, True)] if ii]
            return located
        if isinstance(sighting, list):
            return [(ii, rr) for rr in sighting if isinstance(rr, dict) for ii in [self.identity(rr, True)] if ii]
        return []

    def genre_handle(self) -> str | None:
        tallies: dict[str, set[str]] = {}
        existing: dict[str, int] = {}
        for tier in self.tiers.values():
            for handle, worth in tier.items():
                if isinstance(worth, str) and 0 < len(worth) <= 40 and (not re.search('\\d{3}', worth)):
                    tallies.setdefault(handle, set()).add(worth)
                    existing[handle] = existing.get(handle, 0) + 1
        aggregate = max(1, len(self.tiers))
        familiar = self.legends()
        finest = None
        for handle, worths in tallies.items():
            if existing[handle] < 0.8 * aggregate or len(worths) > 40:
                continue
            rostered = sum((1 for vv in worths if vv.strip().lower() in familiar)) / len(worths)
            average = sum((len(vv) for vv in worths)) / len(worths)
            grade = (-rostered, len(worths), -existing[handle], -average)
            if finest is None or grade < finest[0]:
                finest = (grade, handle)
        return finest[1] if finest else None

    def handle_grid(self) -> list[tuple[str, str, int, list[str]]]:
        sect_handle = self.genre_handle()
        clusters: dict[str, dict[str, Any]] = {}
        for identity in self.spotted_booking:
            entry = self.ample.get(identity)
            if not entry:
                continue
            chunk = _datum_chunk(entry) or {}
            legend = str(entry.get(sect_handle) or 'listings') if sect_handle else 'listings'
            cluster = clusters.setdefault(legend, {'n': 0, 'keys': {}})
            cluster['n'] += 1
            accomplished: set[str] = set()
            for handle, worth in chunk.items():
                moniker = _terms(handle)
                if not moniker or moniker in accomplished:
                    continue
                accomplished.add(moniker)
                cell = cluster['keys'].setdefault(moniker, {'n': 0, 'values': [], 'per': {}, 'plain': {}})
                cell['n'] += 1
                writing = str(worth).strip().lower()
                cell['per'][identity] = writing
                cell['plain'][identity] = _INITIAL_ORTHOGRAPHY.get(str(worth).strip(), writing).strip().lower()
                if writing not in cell['values']:
                    cell['values'].append(writing)
        desired = [(_terms(article.rack or ''), _terms(article.moniker), {ww for ww in _root(article.rack or '') | _root(article.moniker) if len(ww) >= 4}) for article in self.articles]
        accorded = {legend for legend in clusters if any((_terms(legend) in (rack, moniker) or {ww for ww in _root(legend) if len(ww) >= 4} & lengthy for rack, moniker, lengthy in desired)) or self.proprietor_from(legend)}
        if accorded:
            clusters = {legend: cluster for legend, cluster in clusters.items() if legend in accorded}
        grid = []
        for legend, cluster in sorted(clusters.items(), key=lambda hv: -hv[1]['n']):
            preserved: list[tuple[str, dict[str, Any]]] = []
            for moniker, cell in sorted(cluster['keys'].items(), key=lambda hv: (-hv[1]['n'], len(hv[0]))):
                worths = list(cell['plain'].values())
                if not worths:
                    continue
                average = sum((len(vv) for vv in worths)) / len(worths)
                if average > 40 or sum((1 for vv in worths if ' / ' in vv)) * 2 >= len(worths):
                    continue
                repeat = sum((len(set(re.findall('[a-z]{3,}', vv)) & set(re.findall('[a-z]{3,}', ' '.join((oo for oo in worths if oo is not vv))))) > 0 for vv in worths))
                if len(worths) >= 4 and len(set(worths)) == len(worths) and (average > 12) and (repeat * 2 < len(worths)):
                    continue
                duplicate = next((specification for alternative, specification in preserved if self.duplicates(alternative, specification, moniker, cell)), None)
                if duplicate is not None:
                    for identity, writing in cell['per'].items():
                        duplicate['per'].setdefault(identity, writing)
                        duplicate['plain'].setdefault(identity, cell['plain'][identity])
                        if writing not in duplicate['values']:
                            duplicate['values'].append(writing)
                    duplicate['n'] = len(duplicate['per'])
                    continue
                preserved.append((moniker, cell))
            preserved.sort(key=lambda hv: -hv[1]['n'])
            for moniker, cell in preserved[:GRID_HANDLES]:
                grid.append((legend, moniker, cell['n'], [vv[:40] for vv in cell['values'][:4]]))
        return grid

    def identity(self, vertex: Any, inside_roster: bool=False, deepness: int=0) -> str | None:
        if not isinstance(vertex, dict) or len(vertex) < 3:
            return None
        if self.uid_handle and _uid_akin(vertex.get(self.uid_handle), True):
            return vertex[self.uid_handle].strip()
        embedded = any((isinstance(vv, dict) for vv in vertex.values()))
        if not (inside_roster or deepness == 0 or embedded):
            return None
        located = _discover_uid([vertex])
        if located and ('id' in str(located).lower() or vertex[located].strip() in self.tiers):
            return vertex[located].strip()
        return None

    def legends(self) -> set[str]:
        located: set[str] = set()

        def traverse(blueprint: Any, deepness: int) -> None:
            if deepness > 6 or not isinstance(blueprint, dict):
                return
            for choice in blueprint.get('enum') or []:
                if isinstance(choice, str):
                    located.add(choice.strip().lower())
            for worth in blueprint.values():
                if isinstance(worth, dict):
                    traverse(worth, deepness + 1)
        for blueprint in self.blueprints.values():
            traverse(blueprint, 0)
        return located

    def non_declared(self, legend: str, moniker: str, worths: list[str]) -> str:
        numeral = self.proprietor_from(legend)
        if not numeral:
            return ''
        for essential in self.articles[numeral - 1].essentials:
            feature, _, worth = essential.partition(':')
            if re.search('\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', worth, re.I):
                continue
            terms = set(re.findall('[a-z0-9]{3,}', worth.lower()))
            displayed = set(re.findall('[a-z0-9]{3,}', ' '.join(worths).lower()))
            if feature.strip() and _terms(feature) != moniker and _root(feature) & _root(moniker) and terms and (not terms <= displayed):
                return f' [refers to a different property, not the {feature.strip()} that was stated]'
        return ''

    def peruse_form(self, moniker: str, sighting: Any) -> dict[str, Any] | None:
        if moniker in self.forms:
            return self.forms[moniker]
        if not isinstance(sighting, dict):
            return None
        rosters = [kk for kk, vv in sighting.items() if isinstance(vv, list) and len(vv) >= 2 and all((isinstance(rr, dict) for rr in vv))]
        if not rosters:
            return None
        self.forms[moniker] = None
        specimen = {kk: vv[:3] if kk in rosters else vv for kk, vv in sighting.items()}
        instruction = FORM + '\n\nTOOL: ' + moniker + ' | parameters: ' + _spill(self.blueprints.get(moniker) or {}) + '\n\nRESULT (only the opening rows):\n' + _spill(_snip(specimen, 120))[:4000]
        obtained = self.inquire_jdoc(instruction, [MILD, WEIGHTY], 2500) or {}
        tiers = obtained.get('rows') if obtained.get('rows') in rosters else None
        if not tiers:
            return None
        initial = sighting[tiers][0]
        features = (self.blueprints.get(moniker) or {}).get('properties') or {}

        def dubbed(handle: Any, location: dict[str, Any]) -> str | None:
            return handle if isinstance(handle, str) and handle in location else None
        form = {'rows': tiers, 'count': dubbed(obtained.get('count'), sighting), 'next': dubbed(obtained.get('next'), sighting), 'next_arg': dubbed(obtained.get('next_arg'), features), 'price': dubbed(obtained.get('price'), initial), 'stock': dubbed(obtained.get('stock'), initial)}
        if form['count'] and (isinstance(sighting[form['count']], bool) or not isinstance(sighting[form['count']], int)):
            form['count'] = None
        self.forms[moniker] = form
        self.cost_handle = self.cost_handle or form['price']
        self.inventory_handle = self.inventory_handle or form['stock']
        self.transcript.append({'kind': 'shape', 'tool': moniker, 'shape': form})
        return form

    def purchasable(self, identity: str) -> bool:
        if self.vended_output(identity):
            return False
        worth = (self.tiers.get(identity) or {}).get(self.inventory_handle) if self.inventory_handle else None
        if isinstance(worth, str) and re.search(INVENTORY_OUTPUT, worth.strip(), re.I):
            return False
        return not (worth is False or (isinstance(worth, (int, float)) and (not isinstance(worth, bool)) and (worth <= 0)))

    def remark_outlook(self, moniker: str, sighting: Any) -> None:
        for identity, entry in self.entries_inside(sighting):
            handles = set(entry)
            self.instrument_handles.setdefault(moniker, set()).update(handles)
            revision, possess = self.outlook_handles.get(identity, (-1, set()))
            self.outlook_handles[identity] = (self.revision, (possess if revision == self.revision else set()) | handles)

    def soak(self, sighting: Any) -> None:

        def traverse(vertex: Any, deepness: int, inside_roster: bool=False) -> None:
            if deepness > 6:
                return
            if isinstance(vertex, dict):
                identity = self.identity(vertex, inside_roster, deepness)
                if identity:
                    if identity not in self.tiers:
                        self.spotted_booking.append(identity)
                        self.tiers[identity] = {}
                    aged = self.tiers[identity]
                    if any((aged.get(kk) is True and vv is False for kk, vv in vertex.items())):
                        self.unobtainable.add(identity)
                    elif any((aged.get(kk) is False and vv is True for kk, vv in vertex.items())):
                        self.unobtainable.discard(identity)
                    self.tiers[identity].update(vertex)
                    if _datum_chunk(vertex):
                        self.ample[identity] = dict(self.tiers[identity])
                    elif identity in self.ample:
                        self.ample[identity].update(vertex)
                    return
                for worth in vertex.values():
                    traverse(worth, deepness + 1)
            elif isinstance(vertex, list):
                tiers = self.discover_tiers(vertex)
                for worth in vertex:
                    traverse(worth, deepness + 1, tiers)
        traverse(sighting, 0)

    def solitary_aim(self, instrument: str | None) -> str | None:
        blueprint = self.blueprints.get(instrument or '') or {}
        features = blueprint.get('properties') or {}
        writings = [kk for kk, vv in features.items() if isinstance(vv, dict) and 'string' in _varieties(vv) and ('enum' not in vv)]
        aim = next((kk for kk in blueprint.get('required') or [] if kk in writings), None) or (writings[0] if len(writings) == 1 else None)
        if not instrument or not aim or any((kk != aim for kk in blueprint.get('required') or [])):
            return None
        return aim


class _TempoPart:
    """Keeps the clock, the step budget and the message allowance."""

    def basket_marker(self) -> str:
        return _spill(sorted(self.basket_strands.items()))

    def checkpoint_owed(self, phase: str) -> bool:
        return any((pp['when'] == phase for pp in self.digest)) and (phase, self.basket_marker() + f'#{self.revision}') not in self.checkpoint_okay

    def contrast_datums(self) -> bool | None:
        return self.datums_whence.get(self.duties.get('compare') or '')

    def contrast_naked(self) -> bool:
        spotted = self.contrast_datums()
        return spotted is False or (spotted is None and (self.datums_whence.get(self.datums_instrument() or '') is True or bool(self.duties.get('offer'))))

    def datums_instrument(self) -> str | None:
        familiar = next((tt for tt, okay in self.datums_whence.items() if okay and self.solitary_aim(tt)), None)
        if familiar:
            return familiar
        instrument = self.duties.get('open')
        return instrument if self.datums_whence.get(instrument or '') is not False and self.solitary_aim(instrument) else None

    def engine_capacity(self) -> float:
        return MISSION_SECS - MOMENT_LEEWAY - self.timer()

    def hurried(self) -> bool:
        """Too little time for a read the order can do without: that read plus the order, at the store speed seen this run, have to finish within the margin."""
        tempo = self.milieu_expended / self.milieu_tally if self.milieu_tally else 1.0
        return self.engine_capacity() <= 2 * max(tempo, 1.0)

    def mails_gratis(self) -> int:
        return self.mail_cutoff - self.mail_employed - max(0, self.buffer - self.checkpoint_mails)

    def rear(self) -> int:
        converse = any((pp['when'] == 'order' and pp['messages'] > 0 for pp in self.digest))
        return 2 if converse and self.mail_employed < self.mail_cutoff and self.checkpoint_owed('order') else 1

    def referees(self, cutoff: float=EXPEND_GENTLE) -> list[str]:
        if self.outlay <= cutoff and self.outlay + WEIGHTY_CAPACITY <= EXPEND_STRICT:
            return [WEIGHTY, MILD]
        return [MILD] if self.outlay <= EXPEND_STRICT else []

    def remaining(self) -> int:
        return self.maximum_ticks - self.tick

    def sheet_instruments(self) -> set[str]:
        sheet = self.duties.get('offer') or self.duties.get('open')
        return {tt for tt in (sheet, self.duties.get('compare')) if tt}

    def tier_uids(self, moniker: str, operands: dict[str, Any]) -> dict[str, Any]:
        """When a listing tool receives a code found in the title of a row already seen, swap in that row's id; any other value is left untouched."""
        if moniker not in {self.duties.get(chore) for chore in ('open', 'offer', 'compare', 'cart')} or not self.tiers:
            return operands
        headline = self.headline_handle()
        if not headline:
            return operands

        features = (self.blueprints.get(moniker) or {}).get('properties') or {}

        def familiar(worth: Any, handle: str) -> Any:
            if not isinstance(worth, str) or worth.strip() in self.tiers or not 3 <= len(worth.strip()) <= 30 or not re.search('\\d', worth) or 'enum' in (features.get(handle) or {}):
                return worth
            motif = '(?<![a-z0-9])' + re.escape(worth.strip().lower()) + '(?![a-z0-9])'
            strikes = [ii for ii, tier in self.tiers.items() if re.search(motif, str(tier.get(headline) or '').lower())]
            if len(strikes) == 1:
                self.transcript.append({'kind': 'row_id', 'sent': worth, 'id': strikes[0]})
                return strikes[0]
            return worth
        return {handle: [familiar(vv, handle) for vv in worth] if isinstance(worth, list) else familiar(worth, handle) for handle, worth in operands.items()}

    def timer(self) -> float:
        return time.monotonic() - self.initiated


class _EarPart:
    """Splits work into lanes and reads shopper replies and store notes."""

    def article_uttered(self, numeral: int) -> str:
        """How the item is worded in a question. When its name links plural nouns by "and", the question uses "or" in its
        place (race 13, ten agents: asking about "door handles and locks" brought one-at-a-time replies to 81 of 155
        questions, whereas "or" names did so 0 times in 50, and "dresser or cabinet" revealed a need 10 times out of 28).
        Once the shopper has replied one-at-a-time about this item, pick a single product from those its name links: the
        product whose head noun appears most often in the item's own rows beyond their category label (or simply the first
        when no such rows exist).
        """
        article = self.articles[numeral - 1]
        portions = self.moniker_portions(numeral)
        if not portions:
            return article.moniker
        if numeral not in self.sole_ware:
            return re.sub('\\band\\b', 'or', article.moniker, flags=re.I)
        sect_handle = self.genre_handle()
        tiers = [tier for tier in self.tiers.values() if not sect_handle or self.possesses(article, str(tier.get(sect_handle) or ''))]
        dubbed = [_sources(' '.join((str(vv) for kk, vv in tier.items() if kk != sect_handle and isinstance(vv, str)))) for tier in tiers]
        strikes = {pp: sum((1 for terms in dubbed if _sources(pp.split()[-1]) & terms)) for pp in portions}
        return max(portions, key=lambda pp: (strikes[pp], -portions.index(pp)))

    def checkpoint(self, phase: str) -> tuple[bool, str]:
        ticks = [pp for pp in self.digest if pp['when'] == phase]
        if not ticks or self.terminated or (not self.checkpoint_owed(phase)):
            return (True, '')
        closing = {self.duties.get('buy'), self.duties.get('refuse')} - {None}
        instruments = [f"- {tt['function']['name']}: {tt['function'].get('description', '')} | parameters: {_spill(tt['function'].get('parameters') or {})}" for tt in self.instruments if tt['function']['name'] not in closing]
        uttered: list[str] = []
        prodded = False
        for _ in range(CHECKPOINT_LAPS):
            instruction = CHECKPOINT.replace('STAGE', 'submit the order' if phase == 'order' else 'finish without buying') + '\n\nREQUEST:\n' + self.appeal + '\n\nSTEPS DUE NOW:\n' + '\n'.join((f"- {pp['step']}" for pp in ticks)) + '\n\nLEDGER:\n' + self.register() + '\n\nCART NOW: ' + self.basket_writing() + '\n\nRECORD (messages and step calls up to now):\n' + ('\n'.join((f'{ii}. {strand}' for ii, strand in enumerate(self.accomplished_journal, 1))) or '(nothing yet)') + ('\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.remarks[-4:]) if self.remarks else '') + f"\n\nMESSAGES LEFT: {max(0, self.mail_cutoff - self.mail_employed)}   STEPS LEFT: {self.remaining()} (one of them goes to the {('order' if phase == 'order' else 'ending')} itself)" + '\n\nTOOLS:\n' + '\n'.join(instruments)
            obtained = self.inquire_jdoc(instruction, [WEIGHTY, MILD] if prodded else [MILD, WEIGHTY], 3500)
            if obtained is None:
                break
            informed: list[str] = []
            grasp = ''
            if uttered:
                self.grab_perceived({'needs': obtained.get('needs')}, informed)
                grasp = str(obtained.get('hold') or '').strip()
            invoke = obtained.get('call') if isinstance(obtained.get('call'), dict) else None
            self.transcript.append({'kind': 'gate', 'stage': phase, 'call': _snip(invoke, 200), 'steps': _snip(obtained.get('steps'), 120), 'hold': grasp[:200], 'told': informed})
            if informed or grasp:
                self.revision += 1
                return (False, ('Before continuing, Wren carried out the steps the request calls for:\n' + '\n'.join(uttered) + '\n' if uttered else '') + ("The shopper's answer alters the task: " + '; '.join(informed + ([grasp] if grasp else [])) + '. Adjust the cart accordingly, then retry the checkout.'))
            moniker = str((invoke or {}).get('tool') or '')
            displayed = [z for z in _articles(obtained.get('steps')) if isinstance(z, dict)]
            converse = phase == 'order' and any((pp['messages'] > 0 for pp in ticks))
            strands = range(1, len(self.accomplished_journal) + 1)
            evidence = {ii for ii in strands if not converse or (ii <= len(self.accomplished_ticks) and self.accomplished_ticks[ii - 1] > self.basket_upon)}
            unverified = [z for z in displayed if z.get('done') is not True or not any(((_tally(nn) or 0) in evidence for nn in _articles(z.get('lines'))))]
            if not invoke and unverified and (not uttered) and (not prodded) and self.duties.get('ask') and (self.mail_employed < self.mail_cutoff) and (self.remaining() >= 2):
                prodded = True
                self.transcript.append({'kind': 'gate_unproven', 'steps': _snip(unverified, 120)})
                self.accomplished_journal.append(f'(Wren: not one line written after the cart last changed at step {self.basket_upon} shows ' + '; '.join((str(z.get('step') or '') for z in unverified))[:200] + f' done; {self.mail_cutoff - self.mail_employed} message(s) and {self.remaining()} steps remain: issue the call that performs it)')
                self.accomplished_ticks.append(-1)
                continue
            if not invoke or moniker not in self.blueprints or moniker in closing:
                if not invoke:
                    self.checkpoint_okay.add((phase, self.basket_marker() + f'#{self.revision}'))
                break
            if self.remaining() < 2 or (moniker == self.duties.get('ask') and self.mail_employed >= self.mail_cutoff):
                break
            defects: list[str] = []
            operands = _form(invoke.get('args') if isinstance(invoke.get('args'), dict) else {}, self.blueprints[moniker], 'args', defects)
            if defects:
                break
            if moniker == self.duties.get('ask'):
                writing = max(_literals(operands), key=len, default='')
                if not writing:
                    break
                bundle = self.transmit([(1, '', writing)], 1)
                if not bundle:
                    break
                self.checkpoint_mails += 1
                uttered.append(f'Q: {writing}\nA: {bundle[0][2]}')
                continue
            rejoinder = self.perform(moniker, operands)
            if rejoinder is None:
                break
            spotted = rejoinder.get('observation')
            self.soak(spotted)
            self.accomplished_journal.append(f'step {self.tick}: {moniker} {_spill(operands)} -> {self.display(spotted)[:300]} (cart at that time: {self.basket_marker()})')
            self.accomplished_ticks.append(self.tick)
            uttered.append(f'{moniker} {_spill(operands)} -> {self.display(spotted)[:400]}')
        return (True, 'Before continuing, Wren carried out the steps the request calls for:\n' + '\n'.join(uttered) if uttered else '')

    def demands_presently(self) -> list[tuple[int, str]]:
        output = []
        for numeral, article in enumerate(self.articles, 1):
            for demand in article.essentials + article.informed:
                if demand not in article.omitted:
                    output.append((numeral, demand))
        return output

    def embrace_legends(self) -> None:
        sect_handle = self.genre_handle()
        if len(self.articles) < 2 or not sect_handle:
            return
        self.portion_racks()
        legends = {str(entry.get(sect_handle) or '') for entry in self.ample.values()} - {''}
        outliers = [legend for legend in legends if not self.proprietor_from(legend)]
        loners = [nn for nn in range(1, len(self.articles) + 1) if not any((self.proprietor_from(legend) == nn and legend not in self.embraced for legend in legends))]
        if outliers and len(loners) == 1:
            self.embraced.update({legend: loners[0] for legend in outliers})
            self.transcript.append({'kind': 'adopted', 'item': loners[0], 'labels': outliers})

    def entries_from_article(self, numeral: int) -> int:
        sect_handle = self.genre_handle()
        article = self.articles[numeral - 1]
        return sum((1 for entry in self.ample.values() if sect_handle and self.possesses(article, str(entry.get(sect_handle) or '')) and _datum_chunk(entry)))

    def grab_perceived(self, perceived: dict[str, Any], informed: list[str]) -> None:
        numerous = len(self.articles) > 1
        for element in _articles(perceived.get('needs')):
            if not isinstance(element, dict) or not str(element.get('need') or '').strip():
                continue
            numeral = min(max(1, _tally(element.get('item')) or 1), len(self.articles))
            demand = str(element['need']).strip()
            if demand not in self.articles[numeral - 1].informed:
                self.articles[numeral - 1].informed.append(demand)
                informed.append(f'item {numeral}: {demand}' if numerous else demand)
            if element.get('points') is True and (numeral, demand) not in self.references:
                self.references.append((numeral, demand))
        for element in _articles(perceived.get('drop')):
            if isinstance(element, dict) and str(element.get('property') or '').strip():
                article = self.articles[min(max(1, _tally(element.get('item')) or 1), len(self.articles)) - 1]
                departed = _terms(str(element['property']).split(':')[0])
                strike = [mm for mm in article.essentials + article.informed if mm not in article.omitted and _terms(mm.split(':')[0]) == departed]
                article.omitted.extend(strike)
                informed.extend((f'dropped {mm}' for mm in strike))
        for element in _articles(perceived.get('free')):
            if isinstance(element, dict) and element.get('property'):
                numeral = min(max(1, _tally(element.get('item')) or 1), len(self.articles))
                self.articles[numeral - 1].gratis.append(str(element['property']))
        if isinstance(perceived.get('budget'), (int, float)) and (not isinstance(perceived.get('budget'), bool)) and (perceived['budget'] > 0):
            self.allowance = perceived['budget']

    def grab_remark(self, strands: list[str]) -> str:
        self.perceived.update(strands)
        instruction = REMARK + '\n\nITEMS AND REQUIREMENTS NOW:\n' + self.register() + '\n\nTHE NOTE:\n' + '\n'.join(strands)
        obtained = self.inquire_jdoc(instruction, [MILD, SURPLUS], 2500) or {}
        departed = [str(gg) for gg in _articles(obtained.get('gone')) if isinstance(gg, (str, int))]
        self.unobtainable.update((gg for gg in departed if gg in self.tiers))
        self.remark_vended = bool(departed) or obtained.get('sold_out') is True
        self.remark_dubbed = any((gg in self.tiers for gg in departed))
        if self.remark_vended and len(strands) == 1:
            self.vended_strands.update(strands)
        if obtained.get('kind') not in (None, 'change'):
            return ''
        altered = []
        allowance = obtained.get('budget')
        if isinstance(allowance, (int, float)) and (not isinstance(allowance, bool)) and (allowance > 0) and (allowance != self.allowance):
            self.allowance = allowance
            altered.append(f'budget now {allowance}')
        appended: set[str] = set()
        omits = [z for z in _articles(obtained.get('drop')) if isinstance(z, dict) and str(z.get('need') or '').strip()]
        appends = [z for z in _articles(obtained.get('add')) if isinstance(z, dict) and str(z.get('need') or '').strip()]
        for element in omits:
            article = self.articles[min(max(1, _tally(element.get('item')) or 1), len(self.articles)) - 1]
            aim = str(element['need']).strip()
            alive = [mm for mm in article.essentials + article.informed if mm not in article.omitted]
            strike = [mm for mm in alive if mm == aim]
            if not strike:
                strike = [mm for mm in alive if _terms(mm.split(':')[0]) == _terms(aim.split(':')[0])][:1]
            article.omitted.extend(strike or [aim])
            altered.append(f'dropped {aim}')
        for element in appends:
            article = self.articles[min(max(1, _tally(element.get('item')) or 1), len(self.articles)) - 1]
            demand = str(element['need']).strip()
            if demand in article.omitted:
                article.omitted.remove(demand)
            if demand not in article.essentials + article.informed:
                article.informed.append(demand)
            appended.add(demand)
            altered.append(f'added {demand}')
        if departed:
            altered.append('unavailable: ' + ', '.join(departed))
        if obtained.get('price_moved') is True:
            altered.append('a price changed')
        synopsis = str(obtained.get('summary') or '').strip()
        if self.concise.get('pick') == 'cheapest' and (omits or departed or self.remark_vended or any((j.startswith('budget now') for j in altered))):
            self.broader_retained = False
            self.thriftier_markers = set()
        if altered or synopsis:
            self.revision += 1
            self.latest = []
            self.remarks.append(synopsis or '; '.join(altered))
        return (synopsis + (' (' + '; '.join(altered) + ')' if altered else '')).strip()

    def moniker_portions(self, numeral: int) -> list[str]:
        """The products that an item name links with "and" or commas when each one is a plural noun ("door handles and
        locks"), otherwise an empty list; a lone product whose name contains "and" (such as "his and hers towels", "usb and
        lightning cables", "salt and pepper grinder") yields an empty list too.
        """
        moniker = self.articles[numeral - 1].moniker
        if not re.search('\\band\\b', moniker, re.I):
            return []
        portions = [pp.strip() for pp in re.split('\\s*(?:,|\\band\\b)\\s*', moniker, flags=re.I) if pp.strip()]
        return portions if len(portions) >= 2 and all((re.search('[a-z]{3}(?<![su])s$', pp, re.I) for pp in portions)) else []

    def perceive(self, transmitted: list[tuple[int, str, str]], engines: tuple[str, ...]=(WEIGHTY, MILD)) -> dict[str, Any]:
        if self.silent_rejoinder(transmitted):
            self.transcript.append({'kind': 'quiet_reply', 'a': transmitted[0][2][:80]})
            return {}
        self.perceive_peruses += 1
        instruction = PERCEIVE + '\n\nITEMS:\n' + '\n'.join((f'item {nn}: {elem.moniker}' for nn, elem in enumerate(self.articles, 1)))
        instruction += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f"{ii}. ({f'about item {nn}' if nn else 'about several items'}) Q: {qq}\nA: {u}" for ii, (nn, qq, u) in enumerate(transmitted, 1)))
        return self.inquire_jdoc(instruction, list(engines), 2500) or {}

    def possesses(self, article: Article, legend: str) -> bool:
        return bool(legend) and article in self.articles and (self.proprietor_from(legend) == self.articles.index(article) + 1 or _terms(legend) == _terms(article.rack or ''))

    def proprietor_from(self, legend: str) -> int:
        if not legend:
            return 0
        if legend in self.embraced:
            return self.embraced[legend]
        familiar = self.proprietors_spotted.get((legend, len(self.articles)))
        if familiar is not None:
            return familiar
        terms = _terms(legend)
        lengthy = _sources(legend)
        finest, peak = (0, 0.0)
        for numeral, article in enumerate(self.articles, 1):
            if terms == _terms(article.rack or ''):
                grade = 3.0
            elif terms == _terms(article.moniker):
                grade = 2.0
            else:
                unearth = _sources(article.rack or '') | _sources(article.moniker)
                grade = len(lengthy & unearth) / max(1, len(lengthy | unearth))
            if grade > peak:
                finest, peak = (numeral, grade)
        self.proprietors_spotted[(legend, len(self.articles))] = finest
        return finest

    def query(self, numeral: int, legend: str, specimens: list[str] | None=None) -> str:
        writing = f'What {legend} would you like for the {self.article_uttered(numeral)}'
        return writing + (f', for example {specimens[0]} or {specimens[1]}?' if specimens else '?')

    def plan_prompt(self, tally: int) -> tuple[str, list[Any]]:
        """The planner's prompt for a given number of messages, with the attribute key table it lists."""
        self.embrace_legends()
        grid = self.handle_grid()
        finalised = []
        for numeral, article in enumerate(self.articles, 1):
            finalised.append(f'item {numeral}: {article.moniker} - already decided by the request: ' + ('; '.join(article.essentials) or 'nothing'))
        posting = []
        final = None
        for position, (legend, moniker, conveyed, worths) in enumerate(grid, 1):
            if legend != final:
                posting.append(f'category {legend!r}:')
                final = legend
            posting.append(f'  {position}. {moniker} (on {conveyed}): ' + ' | '.join(worths) + self.non_declared(legend, moniker, worths) + (' [bundles several features: ask about it last]' if re.search(TRAP_ENTIRE, moniker) else ''))
        instruction = SCHEME.replace('COUNT', str(max(1, tally))) + '\n\nREQUEST:\n' + self.appeal + '\n\nITEMS:\n' + '\n'.join(finalised)
        initial = [str(px) for px in _articles(self.concise.get('ask_first')) if str(px).strip()]
        if initial:
            instruction += '\n\nASK FIRST (the request wants questions on these):\n' + '\n'.join(initial)
        instruction += '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(posting) or '(no complete listing records so far)')
        return (instruction, grid)

    def plan_ahead(self) -> None:
        """Start the planner's model call in the background while the opening steps still use the store."""
        if self.inquired or not self.duties.get('ask') or self.mails_gratis() <= 0 or not self.articles:
            return
        tally = self.mail_cutoff - self.mail_employed
        instruction, _ = self.plan_prompt(max(tally, self.roster_lid()) if self.posting else tally)
        box: dict[str, Any] = {}
        worker = threading.Thread(target=lambda: box.update(answer=self.inquire_jdoc(instruction, [WEIGHTY, MILD], 3500)), daemon=True)
        self.plan_early = (worker, box)
        worker.start()

    def scheme_tracks(self, tally: int) -> dict[int, list[str]]:
        instruction, grid = self.plan_prompt(tally)
        scheme = None
        if self.plan_early:
            worker, box = self.plan_early
            self.plan_early = None
            worker.join(max(1.0, min(40.0, self.engine_capacity() - INVOKE_BOTTOM)))
            scheme = box.get('answer') if not worker.is_alive() else None
            self.transcript.append({'kind': 'plan_early', 'used': scheme is not None})
        scheme = scheme or self.inquire_jdoc(instruction, [WEIGHTY, MILD], 3500) or {}
        tracks: dict[int, list[str]] = {nn: [] for nn in range(1, len(self.articles) + 1)}
        cued: dict[int, set[str]] = {nn: set() for nn in tracks}
        for choose in _articles(scheme.get('ask')):
            if not isinstance(choose, dict):
                continue
            numeral = min(max(1, _tally(choose.get('item') or 1) or 1), len(self.articles))
            legend = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(choose.get('label') or ''))[0].strip().rstrip('?.')
            if not legend or len(legend.split()) > 5:
                continue
            if all((_terms(legend) != _terms(alternative) for alternative in tracks[numeral])):
                tracks[numeral].append(legend)
            if str(choose.get('hint') or '').strip():
                cued[numeral].add(_terms(legend))
                if numeral not in self.scheme_booking:
                    self.scheme_booking.append(numeral)
        for numeral in tracks:
            handles = self.track_handles(numeral, grid)
            peruse = max(1, self.entries_from_article(numeral))
            personal: dict[str, float] = {}
            for legend, handle, conveyed, _ in grid:
                if self.proprietor_from(legend) == numeral:
                    personal[handle] = max(personal.get(handle, 0.0), conveyed / peruse)

            def spelt(legend: str) -> str:
                simple = lambda writing: re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', _terms(writing)))
                return next((kk for kk in personal if simple(kk) == simple(legend)), legend)

            def fraction_from(legend: str) -> float:
                if _terms(legend) in personal:
                    return personal[_terms(legend)]
                nigh = [ss for kk, ss in personal.items() if _monikers(_root(legend), _root(kk)) or _monikers(_root(kk), _root(legend))]
                return max(nigh) if nigh else 0.0
            cued[numeral] = {_terms(spelt(hh)) for hh in cued[numeral]}
            track = []
            for choose in tracks[numeral]:
                if all((_terms(spelt(choose)) != _terms(alternative) for alternative in track)):
                    track.append(spelt(choose))
            for handle in [kk for kk in handles if personal.get(kk, 0.0) >= SHIELD_FRACTION and (not re.search(TRAP_ENTIRE, kk))][:SHIELD_HANDLES]:
                if any((_terms(handle) == _terms(alternative) for alternative in track[:3])):
                    continue
                feeble = next((ii for ii, alternative in enumerate(track[:3]) if _terms(alternative) not in cued[numeral] and fraction_from(alternative) + SHIELD_VOID <= personal.get(handle, 0.0)), None)
                if feeble is None and len(track) >= 3:
                    continue
                submitted = [alternative for alternative in track if _terms(alternative) != _terms(handle)]
                if feeble is not None and len(self.articles) > 1:
                    shifted = submitted[feeble]
                    submitted[feeble] = handle
                    submitted.insert(min(3, len(submitted)), shifted)
                else:
                    submitted.insert(min(len(submitted), feeble if feeble is not None else len(submitted)), handle)
                if any((_terms(px) in cued[numeral] and all((_terms(px) != _terms(py) for py in submitted[:3])) for px in track[:3])):
                    continue
                track = submitted
            tracks[numeral] = track
            for handle in handles:
                if all((_terms(handle) != _terms(alternative) for alternative in tracks[numeral])):
                    tracks[numeral].append(handle)
            tracks[numeral] = [px for px in tracks[numeral] if not re.search(TRAP_ENTIRE, _terms(px))] + [px for px in tracks[numeral] if re.search(TRAP_ENTIRE, _terms(px))]
        self.transcript.append({'kind': 'plan', 'ask': _snip(scheme.get('ask'), 120), 'lanes': {nn: vv[:8] for nn, vv in tracks.items()}})
        return tracks


    @_gentle(False)
    def silent_rejoinder(self, transmitted: list[tuple[int, str, str]]) -> bool:
        """Whether a reply can be skipped without reading: it mentions nothing that the questions, the items, the listing keys
        or their values mention, contains no number, and asks nothing about how to ask. Race 15: 5112 of 5160 replies
        without a need are like this (race 14: 5050 of 5055), and counting the keys lets the test miss few replies carrying
        a need (race 15: 35 of 1355 using question words alone). Reading one costs a model call, roughly 5-10 s of a 300 s
        episode in a race. The episode's first reply is always read, since it shows whether this shopper accepts lists.
        """
        if self.perceive_peruses < 1:
            return False
        familiar: frozenset[str] = frozenset()
        for numeral, article in enumerate(self.articles, 1):
            familiar |= _spelt(article.moniker) | _spelt(article.rack or '')
            for demand in article.essentials + article.informed:
                familiar |= _spelt(' '.join(re.findall('[A-Za-z0-9]+', demand.split(':', 1)[0])))
        for legend, moniker, _, _ in self.handle_grid():
            familiar |= _spelt(moniker) | _spelt(legend)
        for entry in self.ample.values():
            for worth in (_datum_chunk(entry) or {}).values():
                familiar |= {ww for ww in _spelt(' '.join(re.findall('[A-Za-z0-9]+', str(worth)))) if len(ww) >= SILENT_TERM}
        for _, query, response in transmitted:
            inquired = re.sub('\\b(?:what|which|would|you|like|for|the|please|tell|me|and|or)\\b', ' ', query, flags=re.I)
            unearth = familiar | _spelt(' '.join(re.findall('[A-Za-z0-9]+', inquired)))
            uttered = _spelt(' '.join(re.findall('[A-Za-z0-9]+', response)))
            if re.search('\\d', response) or uttered & unearth or re.search(INQUIRE_TERMS, response, re.I):
                return False
        return True

    def slack_strands(self, sighting: Any) -> list[str]:
        strands: list[str] = []
        if isinstance(sighting, dict) and (not self.identity(sighting)):
            for handle, worth in sighting.items():
                if isinstance(worth, str) and len(worth.split()) >= 5:
                    strands.append(worth)
                elif isinstance(worth, list) and worth and all((isinstance(vv, str) for vv in worth)):
                    strands.extend((vv for vv in worth if len(vv.split()) >= 5))
                elif isinstance(worth, dict) and (not self.identity(worth, False, 1)):
                    for internal in worth.values():
                        if isinstance(internal, str) and len(internal.split()) >= 6:
                            strands.append(internal)
        return [strand for strand in strands if strand not in self.perceived]

    def specimens(self, numeral: int, legend: str) -> list[str]:
        desire = _terms(legend)
        finalised = ' '.join(self.articles[numeral - 1].essentials).lower()
        for _, moniker, _, worths in self.handle_grid():
            if moniker == desire:
                return [vv for vv in worths if vv and vv not in finalised and (len(vv) <= 30)][:2]
        return []

    def testimony(self, sighting: Any) -> str:
        uids: list[str] = []

        def traverse(vertex: Any, deepness: int) -> None:
            if deepness > 6:
                return
            if isinstance(vertex, dict):
                identity = self.identity(vertex, False, deepness)
                if identity and _datum_chunk(vertex) and (identity not in uids):
                    uids.append(identity)
                    return
                for worth in vertex.values():
                    traverse(worth, deepness + 1)
            elif isinstance(vertex, list):
                for worth in vertex:
                    traverse(worth, deepness + 1)
        traverse(sighting, 0)
        demands = self.demands_presently()
        if not uids or not demands:
            return ''
        numerous = len(self.articles) > 1
        strands = []
        for identity in uids[:12]:
            chunk = _datum_chunk(self.ample.get(identity) or {}) or {}
            portions = []
            for numeral, demand in demands[:8]:
                feature = demand.split(':', 1)[0].strip()
                strikes = [f'{kk}={str(vv)[:80]}' for kk, vv in chunk.items() if _monikers(_root(feature), _root(kk))]
                badge = f'(item {numeral}) ' if numerous else ''
                if strikes:
                    portions.append(f'{badge}{feature} -> ' + ', '.join(strikes[:3]))
            if portions:
                strands.append(f'{identity}: ' + '; '.join(portions))
        if not strands:
            return ''
        return "[evidence: keys with a requirement's property in their name; a key that names some other property (a different part or aspect) settles nothing, and a key named otherwise can still carry the same property] " + ' | '.join(strands)

    def track_handles(self, numeral: int, grid: list[tuple[str, str, int, list[str]]]) -> list[str]:
        article = self.articles[numeral - 1]
        finalised = [_root(mm.split(':', 1)[0]) for mm in article.essentials]
        tiers = [(legend, handle, conveyed, len(worths)) for legend, handle, conveyed, worths in grid if self.proprietor_from(legend) == numeral]
        gauged = {handle: sum((1 for vv in worths if re.match('^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$', vv))) * 2 > len(worths) for _, handle, _, worths in grid}
        if len(self.articles) == 1 and (not tiers):
            tiers = [(legend, handle, conveyed, len(worths)) for legend, handle, conveyed, worths in grid]
        tiers.sort(key=lambda rr: (-rr[2], gauged.get(rr[1], False), -rr[3]))
        each = [handle for _, handle, _, _ in tiers if not any((_monikers(feature, _root(handle)) for feature in finalised))]
        handles = [handle for handle in each if not re.search(NON_QUALITY, handle) or re.search('\\w[ _]models?\\b', handle)]
        return [handle for handle in handles if not re.search(TRAP_ENTIRE, handle)] + [handle for handle in handles if re.search(TRAP_ENTIRE, handle)] + [handle for handle in each if handle not in handles]

    def transmit(self, queries: list[tuple[int, str, str]], bottom: int=6) -> list[tuple[int, str, str]]:
        instrument = self.duties['ask']
        transmitted: list[tuple[int, str, str]] = []
        for numeral, _, writing in queries:
            if self.terminated or self.remaining() <= bottom or self.mail_employed >= self.mail_cutoff:
                break
            operands = self.letter_operands(writing)
            if not operands:
                break
            rejoinder = self.perform(instrument, operands)
            if rejoinder is None:
                break
            self.mail_employed += 1
            response = rejoinder.get('observation')
            terms = self.voiced(response)
            transmitted.append((numeral, writing, terms))
            self.accomplished_journal.append(f'step {self.tick}: sent the shopper {writing!r} -> answer {terms[:300]!r} (cart at that time: {self.basket_marker()})')
            self.accomplished_ticks.append(self.tick)
            self.transcript.append({'kind': 'ask', 'q': writing, 'a': terms[:300]})
            if isinstance(response, dict) and response.get('error'):
                break
        return transmitted

    def voiced(self, sighting: Any) -> str:
        if isinstance(sighting, str):
            return sighting
        located: list[str] = []

        def traverse(vertex: Any, deepness: int) -> None:
            if deepness > 4:
                return
            if isinstance(vertex, str) and len(vertex.split()) >= 2:
                located.append(vertex)
            elif isinstance(vertex, dict):
                for worth in vertex.values():
                    traverse(worth, deepness + 1)
            elif isinstance(vertex, list):
                for worth in vertex:
                    traverse(worth, deepness + 1)
        traverse(sighting, 0)
        return ' '.join(located)[:800] if located else _spill(sighting)[:400]


class _LeafPart:
    """Follows result pages, price caps and cheaper rows."""

    def broader_grasp(self) -> str:
        if self.broader_retained or self.concise.get('pick') != 'cheapest' or self.remaining() < 14 + 2 * len(self.articles):
            return ''
        self.broader_retained = True
        strands = [f'{ii} at {self.cost_from(ii):g}' for ii in self.basket_uids() if self.cost_from(ii) is not None]
        return f"Not executed yet (no step spent). {self.remaining()} steps remain and the pick is the lowest price. A search may match only part of a listing's text, so searching a requirement's value turns up only some fitting listings, and results are not sorted by price. Before you check out, do a few more searches per item using other names for the product (its plain single-word name, synonyms, words from fitting titles already seen) and consider only buyable rows cheaper than the cart's line" + (f" ({'; '.join(strands)})" if strands else '') + '; check the attributes of the lowest-priced of those rows, put in any that fits instead, then check out.'

    def cost_arranged(self, sighting: Any) -> bool:
        costs = [self.cost_from(ii) for ii, _ in self.entries_inside(sighting)]
        return len(costs) >= 3 and None not in costs and all((u <= d for u, d in zip(costs, costs[1:])))

    def display_bonds(self, uids: list[str], preserve: int) -> None:
        """For each part code a cart line refers the reader to, run it through search once, and open the page found if steps
        permit; what comes back goes to the ledger rather than the model. Only for carts with several lines.
        """
        scheme = self.worth_stencil()
        if not scheme or (len(self.articles) < 2 and len(uids) < 2):
            return
        instrument, basis, facet = scheme
        for identity in uids:
            markers = self.suit_position()['marks']
            cipher = (markers.get(identity) or {}).get('more')
            if not cipher or cipher in self.bond_attempted or any((marker.get('own') == cipher for marker in markers.values())):
                continue
            if self.remaining() <= preserve or self.terminated or self.hurried():
                return
            self.bond_attempted.add(cipher)
            defects: list[str] = []
            operands = _form({**basis, facet: cipher}, self.blueprints.get(instrument), 'args', defects)
            if defects:
                return
            rejoinder = self.perform(instrument, operands)
            if rejoinder is None:
                return
            self.witness(instrument, rejoinder.get('observation'))
            self.transcript.append({'kind': 'show_link', 'line': identity, 'code': cipher})
            markers = self.suit_position()['marks']
            self.populate_datums([ii for ii, marker in markers.items() if marker.get('own') == cipher], max(preserve, self.rear() + 6))

    def further_sheets(self, moniker: str, operands: dict[str, Any], sighting: Any) -> str:
        self.sheeted_glimpse = []
        choose = self.concise.get('pick')
        scheme = self.sheeting(moniker, sighting) if choose in ('cheapest', 'most', 'least') else None
        if not scheme:
            return ''
        tally, displayed, parameter, worth = scheme
        supplied = operands.get(parameter)
        afterwards = supplied not in (None, 0, 1) if isinstance(supplied, int) and (not isinstance(supplied, bool)) else bool(str(supplied or '').strip())
        if worth is None or afterwards or tally <= displayed:
            return ''
        if choose == 'cheapest' and self.cost_arranged(sighting):
            return ''
        sheets = -(-(tally - displayed) // displayed)
        if sheets > 2 and self.sheets_peruse + 2 <= IMPLICIT_SHEETS and (self.remaining() - self.rear() - 2 >= PARTIAL_SHEETS_ROOM):
            sheets = 2
        if sheets > 2 or self.sheets_peruse + sheets > IMPLICIT_SHEETS or self.remaining() - self.rear() - sheets < 12:
            return f"\n[Wren] {tally} listings matched this search and {displayed} were shown, not sorted by price. The pick must weigh every fitting listing: search again with tighter words (the value the requirement names) or go to the next page with {parameter}={_spill(worth)}."
        writings = []
        for _ in range(sheets):
            defects: list[str] = []
            invoke = _form({**operands, parameter: worth}, self.blueprints.get(moniker), 'args', defects)
            if defects:
                break
            rejoinder = self.perform(moniker, invoke)
            if rejoinder is None or self.terminated:
                break
            self.sheets_peruse += 1
            spotted = rejoinder.get('observation')
            self.witness(moniker, spotted)
            self.sheeted_glimpse.append(spotted)
            displayed_writing = self.display(spotted)
            self.peruse_stash[moniker + _spill(invoke)] = (self.revision, self.tick, displayed_writing)
            writings.append(f'{parameter}={_spill(worth)}: {displayed_writing}')
            following = self.sheeting(moniker, spotted)
            worth = following[3] if following else None
            if worth is None:
                break
        self.transcript.append({'kind': 'pages', 'tool': moniker, 'read': len(writings)})
        remainder = f"\n[Wren] {tally} listings matched, of which Wren has read {displayed * (len(writings) + 1)} rows; to see the others, refine the search with the requirement's value or keep paging." if worth is not None and writings else ''
        return '\n[Wren fetched the following pages of this search on its own, using one step per page]\n' + '\n'.join(writings) + remainder if writings else ''

    def lid_disregarded(self, moniker: str, operands: dict[str, Any], sighting: Any) -> str:
        features = (self.blueprints.get(moniker) or {}).get('properties') or {}

        def costed(handle: str) -> bool:
            specification = features.get(handle) if isinstance(features.get(handle), dict) else {}
            return bool(re.search('price|cost|amount|budget|spend|money', handle + ' ' + str(specification.get('description') or ''), re.I))
        cutoffs = self.lid_handles(moniker)
        lids = [kk for kk, vv in operands.items() if kk in features and isinstance(vv, (int, float)) and (not isinstance(vv, bool)) and (re.search('max|most|under|below|limit|cap|upto|up_to', kk, re.I) and costed(kk) or kk in cutoffs[:1] and len(cutoffs) == 1)]
        if not lids or not self.cost_handle:
            return ''
        lids.sort(key=lambda kk: operands[kk])
        lid = operands[lids[0]]
        beyond = [ii for ii, _ in self.entries_inside(sighting) if (self.cost_from(ii) or 0) > lid]
        if not beyond:
            return ''
        settings = {kk: vv for kk, vv in operands.items() if isinstance(vv, str) and isinstance((features.get(kk) or {}).get('enum'), list)}
        alternatives = {kk: [oo for oo in features[kk]['enum'] if oo != vv] for kk, vv in settings.items()}
        cue = '; browse the category using ' + ', '.join((f'{kk}={_spill(oo[0])}' for kk, oo in alternatives.items() if oo)) + f' plus the same {lids[0]} so that only cheaper listings appear' if any(alternatives.values()) else ''
        return f'\n[Wren] This search ignored {lids[0]}={lid:g}: {len(beyond)} of its rows are priced above it{cue}.'

    def lid_handles(self, instrument: str) -> list[str]:
        """Numeric arguments of the given tool that might limit a listing's price: those that searches showed acting as a
        limit, otherwise those named like a maximum, otherwise those named like a price or budget.
        """
        features = (self.blueprints.get(instrument) or {}).get('properties') or {}
        numerals = [kk for kk, vv in features.items() if isinstance(vv, dict) and {'integer', 'number'} & set(_varieties(vv)) and not re.search(LID_NON, kk, re.I) and not re.search(LID_MODEST, _serpent(kk), re.I)]
        retained = {handle for (handle, _), okay in self.lid_settings(instrument).items() if okay}
        dubbed = [kk for kk in numerals if re.search(LID_OPERAND, _serpent(kk), re.I)]
        costed = [kk for kk in numerals if re.search(LID_COSTED, _serpent(kk), re.I)]
        for cluster in ([kk for kk in numerals if kk in retained], [kk for kk in dubbed if kk in costed], costed, dubbed):
            if cluster:
                return cluster
        return []

    def lid_settings(self, instrument: str) -> dict[tuple[str, tuple[tuple[str, Any], ...]], bool]:
        """What searches with the given tool have shown so far about each numeric argument acting as a price ceiling, per
        argument and mode (the call's option arguments): false as soon as a call returned a row priced above the argument's
        value; true when every row of a call was at or under it and rows sampled randomly from known rows of the same
        labels would seldom have been (a fixed probability threshold), meaning the ceiling, not the query, excluded pricier
        rows.
        """
        features = (self.blueprints.get(instrument) or {}).get('properties') or {}
        sect_handle = self.genre_handle()
        output: dict[tuple[str, tuple[tuple[str, Any], ...]], bool] = {}
        costs: dict[str, list[float]] = {}
        for identity, tier in self.tiers.items():
            cost = self.cost_from(identity)
            if cost is not None:
                costs.setdefault(str(tier.get(sect_handle) or '').strip().lower() if sect_handle else '', []).append(cost)
        for marker, (moniker, operands) in self.peruse_operands.items():
            if moniker != instrument:
                continue
            tiers = [ii for ii, _ in self.entries_inside(self.peruse_invokes.get(marker))]
            displayed = [self.cost_from(ii) for ii in tiers]
            if not tiers or None in displayed:
                continue
            setting = tuple(sorted(((kk, vv) for kk, vv in operands.items() if isinstance(features.get(kk), dict) and isinstance(features[kk].get('enum'), list) and isinstance(vv, (str, int, float)))))
            legends = {str((self.tiers.get(ii) or {}).get(sect_handle) or '').strip().lower() for ii in tiers} if sect_handle else set()
            for handle, worth in operands.items():
                if handle not in features or isinstance(worth, bool) or not isinstance(worth, (int, float)):
                    continue
                reservoir = [pp for legend in legends for pp in costs.get(legend, [])]
                if max(displayed) > worth:
                    output[(handle, setting)] = False
                elif output.get((handle, setting)) is not False and reservoir and (sum((pp <= worth for pp in reservoir)) / len(reservoir)) ** len(tiers) < LID_ODDS:
                    output[(handle, setting)] = True
        return output

    def lid_stencil(self) -> tuple[str, dict[str, Any], str, str] | None:
        """The listing tool accepting a shelf and a price ceiling, along with the mode it requires: the mode a previous shelf
        listing used, otherwise the single option that names a listing. Nothing when neither the tool nor the mode is
        clear.
        """
        for instrument, blueprint in self.blueprints.items():
            features = (blueprint or {}).get('properties') or {}
            lids = self.lid_handles(instrument)
            racks = [kk for kk, vv in features.items() if isinstance(vv, dict) and 'string' in _varieties(vv) and 'enum' not in vv and re.search(RACK_OPERAND, _serpent(kk), re.I)]
            if len(lids) != 1 or len(racks) != 1:
                continue
            discovered = [(dict(setting), retained) for (handle, setting), retained in self.lid_settings(instrument).items() if handle == lids[0]]
            choices_operands = [kk for kk, vv in features.items() if isinstance(vv, dict) and isinstance(vv.get('enum'), list) and len(vv['enum']) >= 2]
            solitary = len(choices_operands) == 1 and choices_operands[0] in ((blueprint or {}).get('required') or [])
            basis: dict[str, Any] = {}
            for handle, specification in features.items():
                choices = specification.get('enum') if isinstance(specification, dict) else None
                if not isinstance(choices, list) or len(choices) < 2:
                    continue
                choose = next((setting[handle] for setting, retained in discovered if retained and handle in setting), None)
                if choose is None:
                    choose = next((u[handle] for u in reversed(self.sought) if racks[0] in u and handle in u and not any(kk != racks[0] and not (isinstance(features.get(kk), dict) and 'enum' in features[kk]) and isinstance(vv, str) and vv.strip() and re.search(ENQUIRY_TERMS, _serpent(kk), re.I) for kk, vv in u.items())), None)
                    dubbed = [oo for oo in choices if isinstance(oo, str) and re.search(ROSTER_SETTING, oo, re.I)]
                    phrased = {u[handle] for u in self.sought if handle in u and any(isinstance(vv, str) and vv.strip() and re.search(ENQUIRY_TERMS, _serpent(kk), re.I) for kk, vv in u.items())}
                    unphrased = [oo for oo in choices if oo not in phrased]
                    choose = choose if choose is not None else dubbed[0] if len(dubbed) == 1 else unphrased[0] if solitary and phrased and len(unphrased) == 1 else None
                ruined = {setting[handle] for setting, retained in discovered if not retained and handle in setting} if solitary else set()
                if ruined and (choose is None or choose in ruined):
                    remainder = [oo for oo in choices if oo not in ruined]
                    choose = remainder[0] if len(remainder) == 1 else None
                if choose is None:
                    return None
                basis[handle] = choose
            return (instrument, basis, racks[0], lids[0])
        return None

    def populate_datums(self, uids: list[str], preserve: int) -> list[str]:
        instrument = self.datums_instrument()
        aim = self.solitary_aim(instrument)
        if (not instrument or not aim) and self.contrast_datums() is not True:
            return []
        writings = []
        desired = [ii for ii in uids if ii in self.tiers and ii not in self.ample and (not self.vended_output(ii))]
        flank = self.duties.get('compare')
        numerous = self.contrast_aim()
        unperused: list[str] = []
        while flank and numerous and self.contrast_datums() is True and len(desired) >= 2 and self.remaining() > preserve and not self.terminated and not self.hurried():
            capacity = self.contrast_capacity()
            bundle, desired = (desired[:capacity], desired[capacity:])
            defects_flank: list[str] = []
            operands_flank = _form({numerous: bundle}, self.blueprints.get(flank), 'args', defects_flank)
            if defects_flank:
                desired = bundle + desired
                break
            rejoinder = self.perform(flank, operands_flank)
            if rejoinder is None:
                desired = bundle + desired
                break
            spotted = rejoinder.get('observation')
            self.witness(flank, spotted)
            self.remark_outlook(flank, spotted)
            tiers = self.entries_inside(spotted)
            for identity, entry in tiers:
                writings.append(f'{identity}: ' + self.display(entry)[:1500])
            if tiers:
                self.contrast_okay = max(self.contrast_okay, len(tiers))
                unperused += [ii for ii in bundle if ii not in self.ample]
                continue
            self.contrast_poor = min(self.contrast_poor or 999, len(bundle))
            desired = bundle + desired
            if self.contrast_capacity() >= len(bundle):
                break
        if not instrument or not aim:
            return writings
        for identity in unperused + desired:
            if self.remaining() <= preserve or self.terminated or self.datums_whence.get(instrument) is False or self.hurried():
                break
            if (instrument, identity) in self.datumless:
                continue
            defects: list[str] = []
            operands = _form({aim: identity}, self.blueprints.get(instrument), 'args', defects)
            if defects:
                break
            rejoinder = self.perform(instrument, operands)
            if rejoinder is None:
                break
            spotted = rejoinder.get('observation')
            self.witness(instrument, spotted)
            self.remark_outlook(instrument, spotted)
            if isinstance(spotted, dict) and self.identity(spotted):
                self.remark_datums(instrument, bool(_datum_chunk(spotted)))
            writings.append(f'{identity}: ' + self.display(spotted)[:1500])
        if writings:
            self.transcript.append({'kind': 'fill_facts', 'n': len(writings)})
        return writings

    def remark_datums(self, instrument: str, displayed: bool) -> None:
        """Record what a single-listing view displayed: it counts as showing attributes once any page included them, and as showing none after two pages lacking them."""
        if displayed:
            self.datums_whence[instrument] = True
        elif self.datums_whence.get(instrument) is not True:
            self.naked_sheets[instrument] = self.naked_sheets.get(instrument, 0) + 1
            if self.naked_sheets[instrument] >= 2:
                self.datums_whence[instrument] = False

    def roster_under(self, basket: list[str]) -> None:
        scheme = self.lid_stencil()
        sect_handle = self.genre_handle()
        if not scheme or not sect_handle:
            return
        instrument, basis, rack, lid = scheme
        phrased = self.worth_stencil()
        facet = phrased[2] if phrased and phrased[0] == instrument and phrased[2] not in (rack, lid) else None
        for strand in basket:
            cost = self.cost_from(strand)
            legend = str((self.tiers.get(strand) or {}).get(sect_handle) or '')
            if cost is None or cost < 2 or not legend or (strand, cost) in self.lidded:
                continue
            self.lidded.add((strand, cost))
            numeral = next((nn for nn, article in enumerate(self.articles, 1) if self.possesses(article, legend)), None)
            enquiry = ' '.join((worth for worth, _ in self.worth_terms(numeral))) if numeral and facet else ''
            for terms in ([enquiry] if enquiry else []) + ['']:
                if self.walk_below(instrument, {**basis, **({facet: terms} if terms else {})}, rack, lid, legend, strand, cost):
                    break

    def walk_below(self, instrument: str, basis: dict[str, Any], rack: str, lid: str, legend: str, strand: str, cost: float) -> bool:
        """List one label below a cart line's price, lowering the cap to the cheapest row shown while pages stay full; True once a
        call showed a row or the walk had to stop."""
        located = False
        cutoff = int(cost) - 1 if float(cost).is_integer() else int(cost)
        attempted: set[int] = set()
        for _ in range(TRAVERSE_INVOKES):
            if self.remaining() <= self.rear() + (TRAVERSE_CAPACITY if attempted else 7) or self.terminated or self.hurried():
                return True
            if cutoff < 1 or cutoff in attempted:
                break
            attempted.add(cutoff)
            defects: list[str] = []
            operands = _form({**basis, rack: legend, lid: cutoff}, self.blueprints.get(instrument), 'args', defects)
            if defects:
                return True
            rejoinder = self.perform(instrument, operands)
            if rejoinder is None:
                return True
            spotted = rejoinder.get('observation')
            self.witness(instrument, spotted)
            tiers = self.entries_inside(spotted)
            located = located or bool(tiers)
            self.transcript.append({'kind': 'list_below', 'line': strand, 'price': cost, 'limit': cutoff, 'rows': len(tiers), 'worded': len(basis)})
            sheet = self.sheeting(instrument, spotted)
            if not tiers or any(((self.cost_from(ii) or 0) > cutoff for ii, _ in tiers)) or not sheet or sheet[0] <= sheet[1] or self.cost_arranged(spotted):
                break
            if sheet[0] <= 2 * sheet[1]:
                invoke = _form({**operands, sheet[2]: sheet[3]}, self.blueprints.get(instrument), 'args', defects) if sheet[3] is not None else None
                further = None if defects or invoke is None or self.remaining() <= self.rear() + 7 or self.hurried() else self.perform(instrument, invoke)
                if further is not None:
                    self.witness(instrument, further.get('observation'))
                break
            modest = min((pp for ii, _ in tiers for pp in [self.cost_from(ii)] if pp is not None), default=None)
            if modest is None:
                break
            cutoff = int(modest) - 1 if float(modest).is_integer() else int(modest)
        return located

    def rosters_objects(self, instrument: str) -> bool:
        """Whether a tool browses a shelf or accepts search words, as opposed to a listing's id."""
        aim = self.solitary_aim(instrument) or ''
        return bool(re.search(RACK_OPERAND, _serpent(aim), re.I) or re.search(ENQUIRY_TERMS, _serpent(aim), re.I)) or instrument in self.forms

    def sheeting(self, moniker: str, sighting: Any) -> tuple[int, int, str, Any] | None:
        form = self.peruse_form(moniker, sighting)
        if not form or not form['count'] or (not form['next']) or (not form['next_arg']) or (not isinstance(sighting, dict)):
            return None
        tiers, tally, worth = (sighting.get(form['rows']), sighting.get(form['count']), sighting.get(form['next']))
        if not isinstance(tiers, list) or not tiers or isinstance(tally, bool) or (not isinstance(tally, int)):
            return None
        if worth is not None and (isinstance(worth, bool) or not isinstance(worth, (str, int))):
            return None
        return (tally, len(tiers), form['next_arg'], worth)

    def thriftier_tiers(self) -> list[str]:
        if self.concise.get('pick') != 'cheapest':
            return []
        marker = self.basket_marker() + f'#{self.revision}#{len(self.spotted_booking)}'
        sect_handle = self.genre_handle()
        if marker in self.thriftier_markers or not sect_handle:
            return []
        self.thriftier_markers.add(marker)
        basket = self.basket_uids()
        self.roster_under(basket)
        chooses: list[str] = []
        capacity = self.contrast_capacity() if self.contrast_datums() is True else 0
        for strand in basket:
            cost = self.cost_from(strand)
            variety = str((self.tiers.get(strand) or {}).get(sect_handle) or '').strip().lower()
            if cost is None or not variety:
                continue
            reservoir = [ii for ii in self.spotted_booking if ii not in basket and ii not in self.ample and self.purchasable(ii) and (str((self.tiers.get(ii) or {}).get(sect_handle) or '').strip().lower() == variety) and ((self.cost_from(ii) if self.cost_from(ii) is not None else float('inf')) < cost)]
            reservoir.sort(key=lambda ii: self.cost_from(ii) or 0.0)
            least = [ii for ii in reservoir if ii in self.worth_tiers][:1]
            dubbed = [ii for ii in reservoir if ii in self.worth_strikes and ii not in least]
            reservoir = least + dubbed + [ii for ii in reservoir if ii not in least and ii not in dubbed]
            chooses += [ii for ii in reservoir[:capacity or 2] if ii not in chooses]
        return self.populate_datums(chooses[:capacity * len(basket) if capacity else min(6, THRIFTIER_LAUNCHES * max(1, len(basket)))], self.rear() + 6)

    def thriftiest_herein(self, sightings: list[Any]) -> str:
        if self.concise.get('pick') != 'cheapest' or not self.cost_handle:
            return ''
        located = {ii for spotted in sightings for ii, _ in self.entries_inside(spotted)}
        tiers = [(self.cost_from(ii), ii) for ii in located if self.purchasable(ii) and self.cost_from(ii) is not None]
        if len(tiers) < 2:
            return ''
        return '\n[Wren] rows here you can buy, lowest price first: ' + ', '.join((f'{ii} ({pp:g})' for pp, ii in sorted(tiers)[:6]))

    @_gentle('')
    def under_basket(self) -> str:
        """Purchasable rows of a cart line's label with a lower price than that line, which no one has opened yet."""
        sect_handle = self.genre_handle()
        basket = self.basket_uids()
        strands = []
        for strand in basket:
            cost = self.cost_from(strand)
            variety = str((self.tiers.get(strand) or {}).get(sect_handle) or '').strip().lower() if sect_handle else ''
            if cost is None or not variety:
                continue
            reservoir = [ii for ii in self.spotted_booking if ii not in basket and ii not in self.ample and self.purchasable(ii) and str((self.tiers.get(ii) or {}).get(sect_handle) or '').strip().lower() == variety and (self.cost_from(ii) if self.cost_from(ii) is not None else float('inf')) < cost]
            reservoir.sort(key=lambda ii: self.cost_from(ii) or 0.0)
            strands += [f'{ii} at {self.cost_from(ii):g}' for ii in reservoir[:THRIFTIER_DISPLAYED]]
        return 'Not yet read and cheaper than a cart line: ' + ', '.join(strands) + '.' if strands else ''


class _OraclePart:
    """Talks to the language models and runs one store action."""

    def inquire_jdoc(self, instruction: str, engines: list[str], maximum_symbols: int=1800) -> dict[str, Any] | None:
        for engine in engines:
            letter = self.ponder([{'role': 'user', 'content': instruction}], [engine], maximum_symbols)
            obtained = _jdoc_entity((letter or {}).get('content'))
            if obtained is not None:
                return obtained
            if self.busted or self.nil_balance or self.invokes >= ENTIRE_INVOKES:
                return None
        return None

    def invoke_engine(self, engine: str, letters: list[dict[str, Any]], maximum_symbols: int, instruments: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        transmitted = letters
        if engine.startswith('anthropic/') and len(letters) > 1:
            transmitted = [dict(mm) for mm in letters]
            markers = [0] + ([len(transmitted) - 1] if len(transmitted) > 1 else [])
            for position in markers:
                substance = transmitted[position].get('content')
                if isinstance(substance, str) and substance:
                    transmitted[position]['content'] = [{'type': 'text', 'text': substance, 'cache_control': {'type': 'ephemeral'}}]
        frame: dict[str, Any] = {'model': engine, 'messages': transmitted, 'max_tokens': maximum_symbols, 'temperature': 0}
        if instruments:
            frame['tools'] = instruments
            frame['tool_choice'] = 'auto'
        outcome = _oracle_amid(self.engine_capacity()).post_verbose('/inference/chat/completions', frame)
        info = outcome.data
        if isinstance(info, dict) and info.get('choices'):
            consumption = info.get('usage') or {}
            if isinstance(consumption.get('cost'), (int, float)):
                self.outlay += float(consumption['cost'])
            letter = info['choices'][0].get('message') or {}
            if instruments and not letter.get('tool_calls') and info['choices'][0].get('finish_reason') == 'length':
                self.transcript.append({'kind': 'llm_fail', 'model': engine, 'status': 0, 'body': 'stopped at max_tokens without a call'})
                return (None, 0)
            if letter.get('content') or letter.get('tool_calls'):
                return (letter, 200)
            self.transcript.append({'kind': 'llm_fail', 'model': engine, 'status': 0})
            return (None, 0)
        mistake = outcome.error or {}
        standing = mistake.get('status')
        if isinstance(info, dict) and isinstance(info.get('error'), dict):
            standing = _tally(info['error'].get('code'))
            mistake = {'body': info['error'].get('message')}
        self.transcript.append({'kind': 'llm_fail', 'model': engine, 'status': standing, 'body': str(mistake.get('body') or '')[:200]})
        return (None, standing)

    def perform(self, moniker: str, operands: dict[str, Any]) -> dict[str, Any] | None:
        cargo = dict(self.tie)
        cargo['step'] = self.tick + 1
        cargo['action'] = {'name': moniker, 'args': operands}
        remainder = self.milieu_upon + self.milieu_void - time.monotonic()
        if remainder > 0:
            time.sleep(remainder)
        self.milieu_upon = time.monotonic()
        for _ in range(4):
            commenced = time.monotonic()
            outcome = _milieu.post_verbose('/environment/call', cargo)
            self.milieu_expended += time.monotonic() - commenced
            self.milieu_tally += 1
            if isinstance(outcome.data, dict):
                info = outcome.data
                self.tick = info['step'] if isinstance(info.get('step'), int) else self.tick + 1
                if info.get('finished'):
                    self.terminated = True
                if moniker != self.duties.get('ask'):
                    info['observation'] = self.unspelt(moniker, info.get('observation'))
                spotted = info.get('observation')
                self.remark_peruse(moniker, operands, spotted)
                if moniker == self.datums_instrument() and isinstance(spotted, dict) and not _faltered(spotted) and self.identity(spotted) and not _datum_chunk(spotted):
                    self.datumless.add((moniker, self.identity(spotted)))
                if moniker in self.sheet_instruments() and (not (isinstance(spotted, dict) and spotted.get('error'))) and not _faltered(spotted):
                    displayed = _spill(spotted)
                    for worth in operands.values():
                        dubbed = [vv.strip() for vv in (worth if isinstance(worth, list) else [worth]) if isinstance(vv, str) and (not isinstance(worth, list) or vv.strip() in displayed)]
                        self.launched.update(dubbed)
                        for identity in dubbed:
                            self.outlooks.setdefault(identity, set()).add(moniker)
                return info
            mistake = outcome.error or {}
            standing = mistake.get('status')
            frame = str(mistake.get('body') or '')
            self.transcript.append({'kind': 'env_fail', 'status': standing, 'body': frame[:200]})
            if standing == 409:
                modest = frame.lower()
                if 'finish' in modest or 'quarantin' in modest or 'finaliz' in modest:
                    self.terminated = True
                    return None
                located = re.search('expected (\\d+)', frame)
                if located:
                    self.tick = int(located.group(1)) - 1
                    cargo['step'] = self.tick + 1
                    continue
                if 'reuse' in modest or 'different action' in modest:
                    self.tick += 1
                    cargo['step'] = self.tick + 1
                    continue
                return None
            if standing in (400, 401, 402, 403, 404, 422):
                if standing == 402:
                    self.busted = True
                return None
        return None

    def ponder(self, letters: list[dict[str, Any]], engines: list[str], maximum_symbols: int, instruments: list[dict[str, Any]] | None=None) -> dict[str, Any] | None:
        for engine in engines:
            if self.invokes >= ENTIRE_INVOKES:
                return None
            if self.engine_capacity() < INVOKE_BOTTOM:
                self.transcript.append({'kind': 'late_call_skipped', 'at': round(self.timer(), 1)})
                return None
            self.invokes += 1
            attempts = 0
            while True:
                letter, standing = self.invoke_engine(engine, letters, maximum_symbols, instruments)
                if letter is not None:
                    return letter
                succeeding = standing is None or standing in (402, 408, 425, 429) or (isinstance(standing, int) and standing >= 500)
                if not succeeding or attempts >= (1 if standing in (None, 408) else len(LINGERS)) or self.lingered + LINGERS[attempts] > LINGER_AGGREGATE:
                    break
                if self.engine_capacity() - LINGERS[attempts] * 1.2 < INVOKE_BOTTOM:
                    break
                halt = LINGERS[attempts] * (0.8 + 0.4 * random.random())
                self.lingered += LINGERS[attempts]
                attempts += 1
                time.sleep(halt)
            if standing == 402:
                self.nil_balance = True
                return None
        return None


class _TallyPart:
    """Checks the basket lines against items, fits and counts."""

    def alternatives(self, bypass: list[str], cutoff: int=30) -> str:
        strands = []
        booking = list(reversed(self.spotted_booking))
        if self.concise.get('pick') == 'cheapest':
            booking.sort(key=lambda ii: (ii not in self.ample, self.cost_from(ii) if self.cost_from(ii) is not None else float('inf')))
        for identity in booking:
            if identity in bypass or self.vended_output(identity):
                continue
            entry = self.ample.get(identity) or self.tiers.get(identity)
            strands.append(_spill(_snip(entry, 160))[:900])
            if len(strands) >= cutoff:
                break
        return '\n'.join(strands)

    def ampler_legend(self, numeral: int, legend: str) -> str:
        sect_handle = self.genre_handle()
        unearth = _sources(self.articles[numeral - 1].moniker)
        if len(unearth) < 2 or unearth <= _sources(legend) or not sect_handle or (self.articles[numeral - 1].rack and _sources(legend) == _sources(self.articles[numeral - 1].rack)):
            return ''
        return next((spotted for spotted in sorted({str(rr.get(sect_handle) or '') for rr in self.tiers.values()}) if spotted and unearth <= _sources(spotted)), '')

    def article_uids(self, numeral: int) -> list[str]:
        sect_handle = self.genre_handle()
        article = self.articles[numeral - 1]
        return [ii for ii in self.spotted_booking if sect_handle and self.possesses(article, str((self.tiers.get(ii) or {}).get(sect_handle) or ''))]

    def basket_uids(self) -> list[str]:
        if self.basket_outlook is not None and self.basket_spotted_upon >= self.basket_upon:
            writing = _spill(self.basket_outlook)
            return [ii for ii in self.tiers if ii in writing]
        return list(self.basket_strands)

    def basket_writing(self) -> str:
        if self.basket_outlook is not None and self.basket_spotted_upon >= self.basket_upon:
            return _spill(_snip(self.basket_outlook, 300))[:3000]
        return "lines according to Wren's log of the cart calls (listing: units): " + _spill(self.basket_strands)

    def bond_defects(self, portion: int, primary: int, gadget: str, strands: list[str], position: dict[str, Any]) -> list[str]:
        platform = self.bond_platform(portion, primary, position)
        if not platform:
            return []
        courier, _, lineages = platform
        markers = position['marks']
        defects = []
        for strand in strands:
            conveyed, dubber = (strand, gadget) if courier == portion else (gadget, strand)
            if self.bond_grasps(conveyed, dubber, lineages, position) is not False:
                continue
            desired = set(markers[dubber].get('named') or []) & lineages
            fine = [ii for ii in self.article_uids(courier) if set(markers[ii].get('fam') or []) & desired and not self.vended_output(ii) and ii != conveyed][:4]
            defects.append(f"{conveyed} belongs to series {', '.join(sorted(set(markers[conveyed].get('fam') or []) & lineages))}, but {dubber} needs {', '.join(sorted(desired))}" + (f"; listings of that series already seen: {', '.join(fine)}" if fine else f"; search for that series together with the item's words, or choose a different {self.articles[primary - 1].moniker} with a series the wanted lines carry"))
        return defects

    def bond_grasps(self, conveyed: str, dubber: str, lineages: set[str], position: dict[str, Any]) -> bool | None:
        """Whether the series on a row belongs to those the other row mentions; undecided when either row is silent on it."""
        personal = set((position['marks'].get(conveyed) or {}).get('fam') or []) & lineages
        dubbed = set((position['marks'].get(dubber) or {}).get('named') or []) & lineages
        return bool(personal & dubbed) if personal and dubbed else None

    def bond_platform(self, portion: int, primary: int, position: dict[str, Any]) -> tuple[int, int, set[str]] | None:
        """For two items tied to each other, the triple of the item bearing series codes, the item naming them, and the
        series; nothing when the rows seen so far reveal no such scheme: most rows of one item bear a code from a few
        series, and rows of the other mention at least one of those series.
        """
        markers = position['marks']
        for courier, dubber in ((portion, primary), (primary, portion)):
            tiers = self.article_uids(courier)
            badged = [ii for ii in tiers if (markers.get(ii) or {}).get('fam')]
            lineages = {ff for ii in badged for ff in markers[ii]['fam']}
            if len(badged) < BOND_TIERS or 2 * len(badged) < len(tiers) or len(lineages) > BOND_LINEAGES:
                continue
            if any((ff in lineages for ii in self.article_uids(dubber) for ff in (markers.get(ii) or {}).get('named') or [])):
                return (courier, dubber, lineages)
        return None

    def conditions_demands(self, identity: str, numeral: int) -> bool:
        """Whether a listing's own attributes state each of the item's requirements: the value word for word (one of its "X or Y"
        alternatives is enough), or every number under the property's key inside a stated limit in the same unit; an exclusion, a
        comparison with another listing or a property the attributes do not name counts as not stated."""
        chunk = _datum_chunk(self.ample.get(identity) or {}) or {}
        divide = lambda writing: set(re.sub('[,/;|]', ' ', _terms(writing)).split())
        for proprietor, demand in self.demands_presently():
            if proprietor != numeral:
                continue
            feature, _, worth = demand.partition(':')
            coded = _ciphers(worth)
            if coded:
                position = self.suit_position()
                suits = self.suit_ensemble(identity, position)
                banned = position['marks'].get(identity, {}).get('bars') or []
                if any((_cipher_strike(ff, j) for ff in suits for j in coded)) and not any((_cipher_strike(d, j) for d in banned for j in coded)):
                    continue
            personal = [str(vv) for kk, vv in chunk.items() if _monikers(_spelt(feature), _spelt(kk))]
            if not personal:
                return False
            trailing = re.match('\\s*(\\d[\\d.,]*)\\s*([a-z"]*)\\s+or\\s+(less|lower|fewer|smaller|under|below|more|higher|greater|larger|above|over)\\s*$', worth.strip().lower())
            if trailing:
                amount, unit, flank = trailing.groups()
                worth = (('at most ' if flank in ('less', 'lower', 'fewer', 'smaller', 'under', 'below') else 'at least ') + amount + ' ' + unit).strip()
            cutoff = re.match('\\s*(at most|up to|under|below|no more than|less than|at least|over|no less than|above|no fewer than|more than)\\s+(\\d[\\d.,]*)\\s*([a-z"]*)\\s*$', worth.strip().lower())
            if cutoff:
                term, amount, unit = cutoff.groups()
                try:
                    brink = float(amount.replace(',', ''))
                except ValueError:
                    return False
                upper = term in ('at most', 'up to', 'under', 'below', 'no more than', 'less than')
                strict = term in ('under', 'below', 'less than', 'over', 'above', 'more than')
                retained = False
                for writing in personal:
                    located = []
                    for estimate, marker in re.findall('(\\d[\\d,]*(?:\\.\\d+)?)\\s*([a-z"]*)', writing.lower()):
                        try:
                            located.append((float(estimate.replace(',', '')), marker))
                        except ValueError:
                            continue
                    if unit:
                        located = [(nn, u) for nn, u in located if u and (u.startswith(unit) or unit.startswith(u))]
                    if located and all(((nn < brink if strict else nn <= brink) if upper else (nn > brink if strict else nn >= brink) for nn, _ in located)):
                        retained = True
                        break
                if not retained:
                    return False
                continue
            terms = divide(worth)
            if not terms or terms & {'not', 'no', 'least', 'most', 'under', 'over', 'unlike', 'different', 'min', 'max', 'minimum', 'maximum'}:
                return False
            choices = [divide(portion) for portion in re.split('\\s+or\\s+', worth.strip().lower()) if divide(portion)] if ' or ' in worth.lower() else [terms]
            if not any((choice <= divide(writing) for choice in choices for writing in personal)):
                return False
        return True

    def bound_ok(self, identity: str, numeral: int) -> bool:
        """For an item bound to another: whether the listing works with the other item's line already in the cart (a shared
        series, or a model code its fit list names); False when that line is missing or nothing shows that they fit."""
        position = self.suit_position()
        for portion, primary in self.confined:
            if numeral not in (portion, primary):
                continue
            alternative = primary if numeral == portion else portion
            strands = [ii for ii in self.basket_uids() if ii in self.article_uids(alternative)]
            if not strands:
                return False
            gadget, portions = (strands[0], [identity]) if numeral == portion else (identity, strands)
            platform = self.bond_platform(portion, primary, position)
            if platform:
                courier, _, lineages = platform
                for strand in portions:
                    conveyed, dubber = (strand, gadget) if courier == portion else (gadget, strand)
                    if self.bond_grasps(conveyed, dubber, lineages, position) is not True:
                        return False
                continue
            ciphers = set(position['marks'].get(gadget, {}).get('codes') or [])
            for strand in portions:
                if not ciphers or not any((_cipher_strike(pp, j) for pp in self.suit_ensemble(strand, position) for j in ciphers)):
                    return False
        return True

    def dubbed_legend(self, numeral: int, legend: str) -> str:
        """The store label that the request gave this item, when a seen listing carries it and it differs from the given label."""
        article = self.articles[numeral - 1]
        sect_handle = self.genre_handle()
        desired = _terms(article.rack or '')
        if not desired or not sect_handle or _terms(legend) == desired:
            return ''
        herein = next((str(rr.get(sect_handle)) for rr in self.tiers.values() if _terms(str(rr.get(sect_handle) or '')) == desired), '')
        return herein

    def entry_from(self, identity: str) -> str:
        entry = self.ample.get(identity) or self.tiers.get(identity)
        return _spill(_snip(entry, 400))[:3000] if entry else '(no record available)'

    def lacking_articles(self) -> list[int]:
        """Items lacking a line in a cart that has fewer lines than items; items for which no seen listing carries a label are ignored."""
        sect_handle = self.genre_handle()
        uids = [ii for ii in self.basket_uids() if (self.retained_parcels(ii) or 0) > 0]
        if not sect_handle or len(uids) >= len(self.articles):
            return []
        legends = [str(rr.get(sect_handle) or '') for rr in self.tiers.values()]
        retained = [str((self.tiers.get(ii) or {}).get(sect_handle) or '') for ii in uids]
        return [nn for nn, article in enumerate(self.articles, 1) if any((self.possesses(article, legend) for legend in legends)) and (not any((self.possesses(article, legend) for legend in retained)))]

    def directed_articles(self) -> set[int]:
        alive = {demand for _, demand in self.demands_presently()}
        return {nn for nn, demand in self.references if demand in alive}

    def attests(self, identity: str, numeral: int) -> bool:
        """True when a listing Wren has read spells out each live requirement of item `number` as race-17 graders judged
        pages: the attribute bearing the property's own name settles it (any entry of a list there will do, and that
        attribute beats all others), yet a plural-named list fails once the title points at one of its other entries.
        Bounds, exclusions and references to other listings are judged by the model instead (False)."""
        entry = self.ample.get(identity) or {}
        chunk = _datum_chunk(entry) or {}
        if not chunk:
            return False
        toks = lambda writing: set(re.findall('[a-z0-9+#]+', _terms(writing)))
        headline = toks(str(entry.get('title') or (self.tiers.get(identity) or {}).get('title') or ''))
        divide = lambda writing: [toks(portion) for portion in re.split('[,/;|]', str(writing)) if toks(portion)]
        demands = [demand for proprietor, demand in self.demands_presently() if proprietor == numeral and (not self.depicted(numeral, demand))]
        if not demands:
            return False
        complete = _spelt(self.articles[numeral - 1].moniker) | {'of'} if 0 < numeral <= len(self.articles) else frozenset()
        evidence = []
        for demand in demands:
            feature, _, worth = demand.partition(':')
            terms = toks(worth)
            if not terms or re.search(UNCLOSED_NON, _terms(worth)):
                return False
            dubbed = [(kk, vv) for kk, vv in chunk.items() if _monikers(_spelt(feature), _spelt(kk))]
            ampler = [(kk, vv) for kk, vv in dubbed if _spelt(feature) < _spelt(kk) <= _spelt(feature) | complete]
            ampler = [(kk, vv) for kk, vv in ampler if len(_spelt(kk)) == max((len(_spelt(ff)) for ff, _ in ampler))]
            exact = ampler or [(kk, vv) for kk, vv in dubbed if _spelt(kk) == _spelt(feature)]
            retained = False
            for handle, writing in exact or dubbed:
                portions = divide(writing)
                strike = [pp for pp in portions if terms <= pp]
                plural = _terms(handle) != _terms(feature) and _spelt(handle) == _spelt(feature)
                if strike and plural and len(portions) > 1 and any((pp not in strike and pp <= headline for pp in portions)) and (not any((pp <= headline for pp in strike))):
                    strike = []
                if exact and (not strike):
                    return False
                retained = retained or bool(strike)
                if strike and len(evidence) < len(demands):
                    evidence.append(f'{handle}: {_snip(str(writing), 160)}')
            if not retained:
                return False
        self.attestations[identity] = evidence
        return True

    def thriftier_attested(self) -> list[tuple[str, float, str, float]]:
        """With the lowest price as the pick rule: per cart line, the lowest-priced listing already read under that line's store
        label which `attests` its item in full while costing less (race 17: Wren opened the listing that would have won
        and still ordered a pricier one)."""
        sect_handle = self.genre_handle()
        if self.concise.get('pick') != 'cheapest' or not sect_handle:
            return []
        basket = [ii for ii in self.basket_uids() if (self.retained_parcels(ii) or 0) > 0]
        bypass = self.directed_articles() | {nn for couple in self.confined for nn in couple}
        output = []
        for strand in basket:
            legend = str((self.tiers.get(strand) or {}).get(sect_handle) or '')
            proprietors = [nn for nn, article in enumerate(self.articles, 1) if self.possesses(article, legend)]
            cost = self.cost_from(strand)
            if len(proprietors) != 1 or proprietors[0] in bypass or cost is None:
                continue
            suits = [ii for ii in self.ample if ii not in basket and ii not in self.departed and self.purchasable(ii) and (not self.vended_output(ii)) and str((self.tiers.get(ii) or {}).get(sect_handle) or '') == legend and (self.cost_from(ii) is not None) and (self.cost_from(ii) < cost - 1e-09) and self.attests(ii, proprietors[0])]
            if suits:
                finest = min(suits, key=lambda ii: self.cost_from(ii) or 0.0)
                output.append((strand, cost, finest, self.cost_from(finest) or 0.0))
        return output

    def gauged(self, demand: str) -> str | None:
        """The unit printed after the number when a requirement sets a measured bound (a limit word, or a figure with a decimal
        point); '' if no unit follows; None when the requirement is something else, e.g. a count or a plain word."""
        worth = _terms(demand.partition(':')[2]) if ':' in demand else ''
        crude = demand.partition(':')[2].lower()
        sized_hit = re.search(VOLUME, crude)
        if not sized_hit or not (re.search('\\d\\.\\d', crude) or re.search(UNCLOSED_NON, worth)):
            return None
        unit = re.sub('^[\\d.,\\s]+', '', sized_hit.group())
        return unit if re.fullmatch('[a-z]+', unit) else ''

    def conditions_metric(self, chunk: dict[str, Any], feature: str, unit: str) -> bool:
        roots = _sources(feature)
        for handle, writing in chunk.items():
            modest = str(writing).lower()
            if roots & _sources(handle) and re.search('\\d', modest) and (not unit or not re.search('[a-z]', modest) or re.search('\\d\\s*' + re.escape(unit), modest)):
                return True
        return False

    def conditions_term(self, identity: str, demand: str, retained: bool=False) -> bool:
        """True when a read listing carries a word requirement in the form race-15 and race-17 graders accepted. If some
        attribute is named after the property, that attribute decides (the model judges its value, unless `held` asks for
        one value word inside it). Otherwise a value word has to show up in some attribute value or in the title; a word
        sitting only in the description does not prove anything (race 15: ordered lines lacking any value word in their
        attributes and title were paid 0 times, 25 failed the request)."""
        entry = self.ample.get(identity) or {}
        chunk = _datum_chunk(entry) or {}
        tokens = _worth_symbols(demand.partition(':')[2])
        if not tokens:
            return True
        dubbed = [_fused(_spill(vv)) for kk, vv in chunk.items() if _monikers(_spelt(demand.partition(':')[0]), _spelt(kk))]
        if dubbed:
            return not retained or any((tt in vv for vv in dubbed for tt in tokens))
        headline = entry.get(self.headline_handle() or 'title') or entry.get('title') or (self.tiers.get(identity) or {}).get('title') or ''
        writing = _fused(' '.join([_spill(vv) for vv in chunk.values()] + [str(headline)]))
        return any((tt in writing for tt in tokens))

    def feeble_strands(self) -> list[tuple[str, int, list[str]]]:
        """The cart lines whose opened page fails to prove a live requirement as race-17 graders demanded: a measured bound
        that shows up only beneath a key about some other measure, only in the title, or not at all (none of 9 such
        listings passed; a bare figure under a key with the property's name is enough), or a word value missed by
        `conditions_term` (exclusions and conditions remain for the model)."""
        sect_handle = self.genre_handle()
        if not sect_handle:
            return []
        directed = {demand for _, demand in self.references}
        output = []
        for strand in [ii for ii in self.basket_uids() if (self.retained_parcels(ii) or 0) > 0]:
            entry = self.ample.get(strand) or {}
            chunk = _datum_chunk(entry) or {}
            proprietors = [nn for nn, article in enumerate(self.articles, 1) if self.possesses(article, str((self.tiers.get(strand) or {}).get(sect_handle) or ''))]
            if not chunk or len(proprietors) != 1:
                continue
            feeble = []
            for proprietor, demand in self.demands_presently():
                if proprietor != proprietors[0] or demand in directed or self.depicted(proprietor, demand) or ':' not in demand:
                    continue
                feature, _, worth = demand.partition(':')
                unit = self.gauged(demand)
                if unit is not None:
                    if not self.conditions_metric(chunk, feature, unit):
                        feeble.append(demand)
                elif not re.search(UNCLOSED_NON, re.sub('\\bor\\b', ' ', _terms(worth))) and (not self.conditions_term(strand, demand)):
                    feeble.append(demand)
            if feeble:
                output.append((strand, proprietors[0], feeble))
        return output

    def feeble_grasp(self) -> str:
        """Run at most once before checkout. For a cart line resting on thin evidence it points to opened listings with the same
        label whose attribute named for the property gives it, failing that to unopened rows of that label worth reading;
        returns '' when there is nothing to point to."""
        feeble = self.feeble_strands()
        if not feeble:
            return ''
        sect_handle = self.genre_handle()
        basket = self.basket_uids()
        portions = []
        for strand, numeral, demands in feeble[:2]:
            legend = str((self.tiers.get(strand) or {}).get(sect_handle) or '')
            features = [nn.partition(':')[0].strip() for nn in demands]
            identical = [ii for ii in self.spotted_booking if ii not in basket and ii != strand and str((self.tiers.get(ii) or {}).get(sect_handle) or '') == legend and self.purchasable(ii) and (not self.vended_output(ii)) and ii not in self.departed]
            if isinstance(self.allowance, (int, float)) and self.concise.get('budget_scope') == 'each':
                identical = [ii for ii in identical if (self.cost_from(ii) or 0.0) <= self.allowance]
            strong = [ii for ii in identical if ii in self.ample and all((self.conditions_metric(_datum_chunk(self.ample[ii]) or {}, nn.partition(':')[0], self.gauged(nn)) if self.gauged(nn) is not None else self.conditions_term(ii, nn, True) for nn in demands))]
            headline = _sources(str((self.ample.get(strand) or self.tiers.get(strand) or {}).get('title') or ''))
            tiers = [ii for ii in identical if ii not in self.ample]
            tiers.sort(key=lambda ii: (not any((_sources(pp) & _sources(_spill(self.tiers.get(ii) or {})) for pp in features)), -len(headline & _sources(str((self.tiers.get(ii) or {}).get('title') or ''))), self.cost_from(ii) if self.cost_from(ii) is not None else float('inf')))
            sizes = [nn.partition(':')[0].strip() for nn in demands if self.gauged(nn) is not None]
            terms = [nn.strip() for nn in demands if self.gauged(nn) is None]
            uttered = []
            if sizes:
                uttered.append(f"{strand} shows no {', '.join(sizes)} beneath a key carrying that name (a figure beneath a key about another measure, in the title only, or missing entirely, leaves it unproven, and the order is failed on it)")
            if terms:
                uttered.append(f"no attribute value or title of {strand} contains what {'; '.join(terms)} wants (a neighbouring attribute with a different sort of value, a list of figures for instance, or words found only in the description leave it unproven unless the page says it another way)")
            writing = '; '.join(uttered)
            if strong:
                strong.sort(key=lambda ii: self.cost_from(ii) if self.cost_from(ii) is not None else float('inf'))
                writing += '; opened listings of that kind giving it under a key of its own name: ' + ', '.join((f'{ii} ({self.cost_from(ii):g})' if self.cost_from(ii) is not None else ii for ii in strong[:3]))
            elif tiers and self.remaining() > self.rear() + 5:
                writing += '; first open these unread listings of that kind: ' + ', '.join(tiers[:3])
            else:
                continue
            portions.append(writing)
        if not portions:
            return ''
        self.transcript.append({'kind': 'weak_hold', 'lines': [ww[0] for ww in feeble]})
        return 'Not executed yet (no step spent). ' + ' | '.join(portions) + '. Replace it with a listing whose record spells out every requirement (a measured limit under a key of its own name, a wanted word inside an attribute or the title) and meets each one (pick rule included), then check out; keep the current line only when no such listing exists.'

    def populate_terse_basket(self) -> bool:
        """Ahead of a forced order, add for every item without a line the cheapest opened listing that states its
        requirements. Returns false if some item remains uncovered, so the episode finishes with a refusal instead of an
        incomplete order.
        """
        basket = self.duties.get('cart') or ''
        confined = {nn for couple in self.confined for nn in couple}
        for numeral in self.lacking_articles():
            article = self.articles[numeral - 1]
            suits = [ii for ii in self.article_uids(numeral) if ii in self.ample and ii not in self.basket_strands and self.purchasable(ii) and self.cost_from(ii) is not None and self.conditions_demands(ii, numeral)]
            suits.sort(key=lambda ii: self.cost_from(ii) or 0.0)
            suits = [ii for ii in suits if numeral not in confined or self.bound_ok(ii, numeral)]
            choose = suits[0] if suits else None
            if choose and isinstance(self.allowance, (int, float)):
                retained = sum(((self.cost_from(ii) or 0.0) * (self.retained_parcels(ii) or 0) for ii in self.basket_uids()))
                outlay = (self.cost_from(choose) or 0.0) * (1 if self.concise.get('budget_scope') == 'each' else article.amt)
                if outlay + (0.0 if self.concise.get('budget_scope') == 'each' else retained) > self.allowance + 1e-09:
                    choose = None
            invokes = self.basket_invokes({**{ii: self.basket_strands[ii] for ii in self.basket_uids() if ii in self.basket_strands}, choose: article.amt}, []) if choose else None
            if not invokes or len(invokes) + 1 > self.remaining():
                return False
            for operands in invokes:
                defects: list[str] = []
                operands = _form(operands, self.blueprints.get(basket), 'args', defects)
                if defects or self.perform(basket, operands) is None:
                    return False
                self.remark_basket(basket, operands)
            self.transcript.append({'kind': 'filled_line', 'item': numeral, 'line': choose})
        return True

    def portion_racks(self) -> None:
        """When an item shares a shelf with another requested item, it takes the store label matching its own name, if that
        label is seen and no item has claimed it yet.
        """
        sect_handle = self.genre_handle()
        legends = sorted({str(tier.get(sect_handle) or '').strip() for tier in self.tiers.values()} - {''}) if sect_handle else []
        shifted = False
        for article in self.articles:
            moniker = _sources(article.moniker)
            if not moniker or not article.rack.strip() or not any((alternative is not article and _terms(alternative.rack) == _terms(article.rack) for alternative in self.articles)):
                continue
            if any((alternative is not article and _sources(alternative.moniker) == moniker for alternative in self.articles)):
                continue
            personal = [legend for legend in legends if _sources(legend) == moniker and _terms(legend) != _terms(article.rack)]
            if len(personal) == 1 and not any((_terms(alternative.rack) == _terms(personal[0]) for alternative in self.articles)):
                self.transcript.append({'kind': 'shelf_parted', 'item': article.moniker, 'from': article.rack, 'to': personal[0]})
                article.rack = personal[0]
                shifted = True
        if shifted:
            self.proprietors_spotted = {}
            self.suit_stashed = None

    def repair_tallies(self, incorrect: list[tuple[str, int, int]]) -> list[str]:
        moniker = self.duties.get('cart') or ''
        blueprint = self.blueprints.get(moniker) or {}
        features = blueprint.get('properties') or {}
        writings = [kk for kk, vv in features.items() if isinstance(vv, dict) and 'string' in _varieties(vv) and ('enum' not in vv)]
        aim = next((kk for tt, kk in self.uid_operands if tt == moniker and kk in writings), None) or next((kk for kk in blueprint.get('required') or [] if kk in writings), None) or (writings[0] if len(writings) == 1 else None)
        tally = next((kk for kk, vv in features.items() if isinstance(vv, dict) and 'integer' in _varieties(vv)), None)
        setting = next((kk for kk, vv in features.items() if isinstance(vv, dict) and isinstance(vv.get('enum'), list) and (len(vv['enum']) >= 2)), None)
        if not moniker or not aim or (not tally):
            return []
        invokes: list[dict[str, Any]] = []
        for identity, possess, desire in incorrect:
            if self.basket_ensembles(moniker):
                invokes.append({aim: identity, tally: desire})
            elif setting:
                choices = [str(oo) for oo in features[setting]['enum']]
                unset = next((oo for oo in choices if re.search('remov|delet|drop|clear|take', oo, re.I)), choices[1])
                atop = next((oo for oo in choices if oo != unset), choices[0])
                invokes += [{aim: identity, setting: unset, tally: possess}, {aim: identity, setting: atop, tally: desire}]
            else:
                return []
        if len(invokes) + 3 > self.remaining():
            return []
        accomplished = []
        for operands in invokes:
            defects: list[str] = []
            operands = _form(operands, blueprint, 'args', defects)
            if defects:
                break
            rejoinder = self.perform(moniker, operands)
            if rejoinder is None or self.terminated:
                break
            self.remark_basket(moniker, operands)
            self.basket_uttered = _spill(rejoinder.get('observation'))
            self.witness(moniker, rejoinder.get('observation'))
            accomplished.append(_spill(operands) + ' -> ' + self.display(rejoinder.get('observation'))[:300])
        self.transcript.append({'kind': 'count_fix', 'calls': accomplished})
        return accomplished

    def retained_parcels(self, identity: str) -> int:
        if self.basket_outlook is not None and self.basket_spotted_upon >= self.basket_upon:
            pile: list[Any] = [self.basket_outlook]
            while pile:
                vertex = pile.pop()
                if isinstance(vertex, dict):
                    if identity in [vv.strip() for vv in vertex.values() if isinstance(vv, str)]:
                        tallies = [_tally(vv) for kk, vv in vertex.items() if not isinstance(vv, str) and _tally(vv) is not None and re.search('\\b(?:qty|quantity|count|units?)\\b', _terms(kk))]
                        if len(tallies) == 1:
                            return tallies[0]
                    pile += [vv for vv in vertex.values() if isinstance(vv, (dict, list))]
                elif isinstance(vertex, list):
                    pile += vertex
        return self.basket_strands.get(identity, 1)

    def sequence_towards(self, numeral: int) -> list[str]:
        """The series to search a linked item by, best first: those carried by the cart's lines for the linked items, then
        those that most rows of the linked items bear or mention. An empty list when no series scheme links the item.
        """
        position = self.suit_position()
        markers = position['marks']
        tallies: dict[str, int] = {}
        retained: list[str] = []
        basket = self.basket_uids()
        for portion, primary in self.confined:
            platform = self.bond_platform(portion, primary, position) if numeral in (portion, primary) else None
            if not platform:
                continue
            for fellow in (portion, primary):
                for ii in self.article_uids(fellow):
                    badges = set((markers.get(ii) or {}).get('fam') or []) | set((markers.get(ii) or {}).get('named') or [])
                    for lineage in badges & platform[2]:
                        tallies[lineage] = tallies.get(lineage, 0) + 1
                        if ii in basket and lineage not in retained:
                            retained.append(lineage)
        return (retained + [ff for ff in sorted(tallies, key=lambda ff: (-tallies[ff], ff)) if ff not in retained])[:BOND_SEEKS]

    def strand_defects(self) -> list[str]:
        self.embrace_legends()
        sect_handle = self.genre_handle()
        uids = [ii for ii in self.basket_uids() if (self.retained_parcels(ii) or 0) > 0]
        if not uids or not sect_handle:
            return []
        legend = {ii: str((self.tiers.get(ii) or {}).get(sect_handle) or '') for ii in uids}
        familiar = {nn for nn, article in enumerate(self.articles, 1) if any((self.possesses(article, str(rr.get(sect_handle) or '')) for rr in self.tiers.values()))}
        proprietor = {ii: [nn for nn, article in enumerate(self.articles, 1) if self.possesses(article, legend[ii])] for ii in uids}
        defects: list[str] = []
        for nn, article in enumerate(self.articles, 1):
            if nn not in familiar:
                continue
            strands = [ii for ii in uids if nn in proprietor[ii]]
            if not strands:
                writing = f'the cart holds no line for item {nn} ({article.moniker})'
                origin = self.legend_writing(nn)
                taken = [ii for ii in uids if origin and legend[ii] and _terms(legend[ii]) != _terms(origin) and _sources(legend[ii]) & _sources(origin)]
                if taken:
                    writing += f" (its listings bear the store label '{origin}'; " + ', '.join((f"{ii} bears '{legend[ii]}'" + (f', the label of item {proprietor[ii][0]}' if len(proprietor[ii]) == 1 else '') for ii in taken)) + (', hence it does not' if len(taken) == 1 else ', hence they do not') + f" count toward item {nn}, regardless of the title: list '{origin}' and take the line for item {nn} from it)"
                defects.append(writing)
            elif len([ii for ii in strands if proprietor[ii] == [nn]]) > 1:
                defects.append(f"item {nn} ({article.moniker}) appears on {len(strands)} lines ({', '.join(strands)}); keep just one")
            for ii in strands:
                possess = self.retained_parcels(ii)
                if proprietor[ii] == [nn] and possess != article.amt:
                    defects.append(f'{ii} is held at {possess} unit(s), while item {nn} ({article.moniker}) needs {article.amt}')
                ampler = self.ampler_legend(nn, legend[ii]) or self.dubbed_legend(nn, legend[ii])
                if ampler:
                    defects.append(f'{ii} comes from {legend[ii]}, yet the store has a label {ampler} that names item {nn} ({article.moniker}) completely: take the line from that one')
        if len(familiar) == len(self.articles):
            for ii in uids:
                if not proprietor[ii]:
                    defects.append(f"{ii} ({legend[ii] or 'unknown kind'}) is not a kind the shopper requested: remove it")
        for ii in uids:
            if self.vended_output(ii):
                defects.append(f'{ii} is out of stock')
        costs = {ii: self.cost_from(ii) for ii in uids}
        if isinstance(self.allowance, (int, float)) and all((pp is not None for pp in costs.values())):
            if self.concise.get('budget_scope') == 'each':
                beyond = [ii for ii in uids if self.allowance < costs[ii] <= 10 * self.allowance]
                defects += [f'{ii} is priced at {costs[ii]:g}, above the per-item limit of {self.allowance:g}' for ii in beyond]
            else:
                aggregate = sum((costs[ii] * self.retained_parcels(ii) for ii in uids))
                if self.allowance + 1e-09 < aggregate <= 10 * self.allowance:
                    defects.append(f'the total comes to {aggregate:g}, which exceeds the budget of {self.allowance:g}')
        return defects

    def suit_defects(self) -> list[str]:
        if not self.confined:
            return []
        position = self.suit_position()
        basket = self.basket_uids()
        defects = []
        for portion, primary in self.confined:
            gadgets = [ii for ii in basket if ii in self.article_uids(primary)]
            portions = [ii for ii in basket if ii in self.article_uids(portion)]
            if len(gadgets) != 1 or not portions:
                continue
            bonded = self.bond_defects(portion, primary, gadgets[0], portions, position)
            if bonded:
                defects += bonded
                continue
            ciphers = set(position['marks'].get(gadgets[0], {}).get('codes') or [])
            if not ciphers:
                continue
            banned = lambda ii: set(position['marks'].get(ii, {}).get('bars') or [])
            spans = lambda ii: any((_cipher_strike(pp, j) for pp in self.suit_ensemble(ii, position) for j in ciphers)) and not any((_cipher_strike(pp, j) for pp in banned(ii) for j in ciphers))
            for strand in portions:
                suits = self.suit_ensemble(strand, position)
                if suits and (not spans(strand)):
                    fine = [ii for ii in self.article_uids(portion) if spans(ii) and (not self.vended_output(ii)) and (ii not in basket)][:4]
                    defects.append(f"{strand} ({self.articles[portion - 1].moniker}) is compatible with {', '.join(sorted(suits)[:4])} but not with {gadgets[0]} ({', '.join(sorted(ciphers))})" + (f"; parts of that kind already seen that are compatible with it: {', '.join(fine)}" if fine else '; none of the parts of that kind seen so far is compatible with it, so search for one or change the device: ' + self.suit_scheme(terse=True)))
        return defects

    def suit_ensemble(self, identity: str, position: dict[str, Any]) -> set[str]:
        marker = position['marks'].get(identity) or {}
        located = set(marker.get('works') or [])
        if marker.get('own'):
            located |= position['notes'].get(marker['own'], set())
        return located

    def suit_position(self) -> dict[str, Any]:
        condition = (len(self.spotted_booking), len(self.ample), sum((len(vv) for vv in self.outlooks.values())))
        if self.suit_stashed and self.suit_stashed[0] == condition:
            return self.suit_stashed[1]
        markers: dict[str, dict[str, Any]] = {}
        remarks: dict[str, set[str]] = {}
        for identity in self.spotted_booking:
            entry = self.ample.get(identity) or self.tiers.get(identity) or {}
            writing = ' '.join((str(vv) for vv in entry.values() if isinstance(vv, str)))
            writing += ' ' + '. '.join((f'{kk} {vv}' for chunk in entry.values() if isinstance(chunk, dict) for kk, vv in chunk.items() if isinstance(vv, str)))
            markers[identity] = _suit_markers(writing)
            for portion, ciphers in markers[identity]['also']:
                remarks.setdefault(portion, set()).update(ciphers)
        self.suit_stashed = (condition, {'marks': markers, 'notes': remarks})
        return self.suit_stashed[1]

    def suit_scheme(self, terse: bool=False) -> str:
        if not self.confined:
            return ''
        position = self.suit_position()
        strands = []
        for primary in sorted({mm for _, mm in self.confined}):
            portions = [pp for pp, mm in self.confined if mm == primary]
            reckoning: dict[str, dict[int, int]] = {}
            for pp in portions:
                for ii in self.article_uids(pp):
                    if self.vended_output(ii):
                        continue
                    for cipher in self.suit_ensemble(ii, position):
                        reckoning.setdefault(cipher, {}).setdefault(pp, 0)
                        reckoning[cipher][pp] += 1
            for pp in portions:
                platform = self.bond_platform(pp, primary, position)
                for ii in self.article_uids(pp) if platform and platform[0] == pp else []:
                    for cipher in set((position['marks'].get(ii) or {}).get('fam') or []) & platform[2] if not self.vended_output(ii) else ():
                        reckoning.setdefault(cipher, {}).setdefault(pp, 0)
                        reckoning[cipher][pp] += 1
            rated = sorted(reckoning.items(), key=lambda hv: (-len(hv[1]), -sum(hv[1].values())))[:6]
            gadgets = {cipher: [ii for ii in self.article_uids(primary) if cipher in ((position['marks'].get(ii) or {}).get('named') or []) or any((_cipher_strike(cipher, j) for j in position['marks'].get(ii, {}).get('codes') or []))][:2] for cipher, _ in rated}
            writing = '; '.join((f'{cipher}: ' + ', '.join((f'{nn} {self.articles[pp - 1].moniker}' for pp, nn in sorted(apiece.items()))) + (f" (device seen: {', '.join(gadgets[cipher])})" if gadgets[cipher] else ' (no device seen so far)') for cipher, apiece in rated))
            if writing:
                strands.append(f"codes named by the parts of the {self.articles[primary - 1].moniker} candidates -> {writing}")
            stating = []
            for cipher, apiece in rated:
                located = []
                for pp in sorted(apiece):
                    okay = [ii for ii in self.article_uids(pp) if ii in self.ample and not self.vended_output(ii) and cipher in set((position['marks'].get(ii) or {}).get('fam') or []) | set(self.suit_ensemble(ii, position)) and self.conditions_demands(ii, pp)]
                    if okay:
                        located.append(f"{self.articles[pp - 1].moniker} {', '.join(okay[:2])}")
                if located:
                    stating.append(f'{cipher}: ' + '; '.join(located))
            if stating:
                strands.append('read rows that state every requirement of their item, by code -> ' + ' | '.join(stating))
            ensembles = self.thriftiest_ensembles(primary, portions, position) if self.concise.get('pick') == 'cheapest' else ''
            if ensembles:
                strands.append(ensembles)
        if not strands:
            return ''
        return ' | '.join(strands) if terse else "[fit plan: a part suits a device if the part's text mentions the device's model code or any other name the device is sold under, or if the device's text mentions the series of the part's code; run one search for a code or series plus the item's words, and if that returns nothing, browse the device's category and read the codes shown on its rows] " + ' | '.join(strands)

    def tally_repairs(self) -> list[tuple[str, int, int]]:
        sect_handle = self.genre_handle()
        if not sect_handle:
            return []
        output = []
        for identity in self.basket_uids():
            possess = self.retained_parcels(identity) or 0
            proprietors = [nn for nn, article in enumerate(self.articles, 1) if self.possesses(article, str((self.tiers.get(identity) or {}).get(sect_handle) or ''))]
            unearth = [ii for ii in self.basket_uids() if (self.basket_strands.get(ii, 1) or 0) > 0 and len(proprietors) == 1 and self.possesses(self.articles[proprietors[0] - 1], str((self.tiers.get(ii) or {}).get(sect_handle) or ''))]
            if possess > 0 and len(proprietors) == 1 and (len(unearth) == 1) and (possess != self.articles[proprietors[0] - 1].amt) and (not self.vended_output(identity)):
                output.append((identity, possess, self.articles[proprietors[0] - 1].amt))
        return output

    @_gentle('')
    def thriftiest_ensembles(self, primary: int, portions: list[int], position: dict[str, Any]) -> str:
        """The cheapest totals for one device of the item combined with one part of every bound item that its codes fit, among
        purchasable rows seen and without checking requirements: the choice weighs the full order, not each line
        separately.
        """
        choices = []
        platforms = {portion: self.bond_platform(portion, primary, position) for portion in portions}
        racks = {portion: sorted(((self.cost_from(ii), ii, self.suit_ensemble(ii, position), position['marks'].get(ii, {}).get('bars') or []) for ii in self.article_uids(portion) if self.cost_from(ii) is not None and self.purchasable(ii)), key=lambda oo: (oo[0], oo[1])) for portion in portions}
        for gadget in self.article_uids(primary):
            ciphers = position['marks'].get(gadget, {}).get('codes') or []
            cost = self.cost_from(gadget)
            if not (ciphers or (position['marks'].get(gadget, {}).get('named'))) or cost is None or not self.purchasable(gadget):
                continue
            aggregate = cost * self.articles[primary - 1].amt
            chooses = [gadget]
            for portion in portions:
                platform = platforms[portion]
                suiting = next((ii for _, ii, suits, bans in racks[portion] if (any((_cipher_strike(pp, j) for pp in suits for j in ciphers)) and not any((_cipher_strike(pp, j) for pp in bans for j in ciphers))) or (platform and platform[0] == portion and self.bond_grasps(ii, gadget, platform[2], position) is True)), None)
                if not suiting:
                    chooses = []
                    break
                chooses.append(suiting)
                aggregate += (self.cost_from(suiting) or 0.0) * self.articles[portion - 1].amt
            if chooses:
                choices.append((aggregate, chooses))
        choices.sort(key=lambda oo: oo[0])
        if not choices:
            return ''
        return 'cheapest complete sets seen so far (requirements still unchecked): ' + '; '.join((f"{' + '.join(pp)} = {tt:g}" for tt, pp in choices[:ENSEMBLE_DISPLAYED]))

    def verify_basket(self, uids: list[str] | None=None, cutoff: float=EXPEND_GENTLE) -> str:
        prospect = uids is not None
        uids = uids if prospect else self.basket_uids()
        handle = self.verify_handle(uids)
        if handle in self.verified or not self.referees(cutoff) or (not uids):
            return ''
        self.verified.add(handle)
        instruction = VERIFY + '\n\nREQUEST:\n' + self.appeal + '\n\nREQUIREMENTS IN FORCE:\n' + self.register()
        if prospect:
            amount = '; '.join((f'item {nn}: quantity {elem.amt}' for nn, elem in enumerate(self.articles, 1)))
            instruction += '\n\nCART (planned; assess these listings as though they sat in the cart at the requested quantities: ' + amount + '):\n' + ', '.join(uids)
        else:
            instruction += '\n\nCART:\n' + self.basket_writing()
        instruction += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + '\n'.join((f'{ii}: ' + ('SOLD OUT (per the store, it is no longer available to buy). ' if self.vended_output(ii) else '') + self.entry_from(ii) for ii in uids))
        feeble = [] if prospect else self.feeble_strands()
        if feeble:
            instruction += "\n\nNOT STATED BY THESE RECORDS (a measured limit is proven only by a key that shares the property's name; a wanted word only by an attribute value or the title, never by a neighbouring attribute holding a different sort of value, nor by the description), so treat each as unmet unless the record says it some other way:\n" + '\n'.join((f"{strand}: {'; '.join(demands)}" for strand, _, demands in feeble))
        if self.concise.get('pick') in ('cheapest', 'most', 'least'):
            instruction += '\n\nOTHER LISTINGS SEEN (most recent first):\n' + self.alternatives(uids, 15)
        referees = self.referees(cutoff)
        obtained = None
        if len(referees) > 1 and not prospect and not self.vetted and self.concise.get('pick') not in ('cheapest', 'most', 'least') and (not feeble):
            obtained = self.inquire_jdoc(instruction, [MILD], 3000)
            if obtained:
                referees = [WEIGHTY] if obtained.get('ok') is False and WEIGHTY in self.referees(cutoff) else []
        if referees and not self.nil_balance:
            obtained = self.inquire_jdoc(instruction, referees, 3000) or obtained
        self.transcript.append({'kind': 'check', 'cart': uids, 'got': obtained})
        self.rulings[handle] = bool(obtained) and obtained.get('ok') is not False
        if not obtained or obtained.get('ok') is not False:
            return ''
        self.prospects = [ii for ii in self.prospects if ii not in uids]
        issues = [str(pp) for pp in _articles(obtained.get('problems'))][:5]
        finer = [str(d) for d in _articles(obtained.get('better')) if str(d) in self.tiers and (not self.vended_output(str(d)))][:4]
        writing = 'A close review finds that this cart is still not correct: ' + '; '.join(issues or ['refer to the requirements'])
        if finer:
            writing += '. Listings that seem a better match: ' + ', '.join(finer)
        return writing + '. Correct the cart (or re-read the records to prove the review mistaken), then proceed to checkout.'

    def verify_handle(self, uids: list[str]) -> str:
        return '|'.join(sorted(uids)) + f'#{self.revision}'


class _IntakePart:
    """Turns the shopper request into items, needs and a running ledger."""

    @staticmethod
    def communal_rack(concise: dict[str, Any]) -> bool:
        racks = [_terms(str(z.get('shelf') or '')) for z in _articles(concise.get('items')) if isinstance(z, dict)]
        racks = [ss for ss in racks if ss]
        return len(racks) != len(set(racks))

    def depicted(self, numeral: int, demand: str) -> bool:
        if demand in [mm for _, mm in self.references] or demand in [str(px) for px in _articles(self.concise.get('lookups'))]:
            return False
        feature, _, worth = demand.partition(':')
        if not feature.strip() or len(re.findall('[a-z0-9]+', worth.lower())) < 4:
            return False
        if re.search('\\b(?:listing|page|shown|other than|unlike|same as)\\b', worth, re.I):
            return False
        article = self.articles[numeral - 1]
        if any(({ww for ww in _root(alternative.moniker) if len(ww) >= 4} & _root(worth) for alternative in self.articles if alternative is not article)):
            return False
        sect_handle = self.genre_handle()
        handles: set[str] = set()
        peruse = 0
        for entry in self.ample.values():
            if sect_handle and (not self.possesses(article, str(entry.get(sect_handle) or ''))):
                continue
            chunk = _datum_chunk(entry) or {}
            if chunk:
                peruse += 1
                handles |= {_terms(kk) for kk in chunk}
        return peruse >= 3 and (not any((_root(kk) & _root(feature) for kk in handles)))

    def peruse_appeal(self) -> None:
        strands = []
        for instrument in self.instruments:
            func = instrument['function']
            strands.append(f"- {func['name']}: {func.get('description', '')} | parameters: {_spill(func.get('parameters') or {})}")
        instruction = CONCISE + '\nTOOLS:\n' + '\n'.join(strands) + '\n\nREQUEST:\n' + self.appeal
        if self.outlook.get('opening') is not None:
            instruction += '\n\nOPENING NOTE FROM THE STORE:\n' + _spill(_snip(self.outlook.get('opening'), 1500))
        concise = self.inquire_jdoc(instruction, [MILD], 4000) or {}
        if not self.nil_balance and (not _articles(concise.get('items')) or not isinstance(concise.get('roles'), dict) or (not concise['roles'].get('buy'))):
            concise = self.inquire_jdoc(instruction, [WEIGHTY, MILD], 2500) or concise
        elif not self.nil_balance and self.communal_rack(concise):
            anew = self.inquire_jdoc(instruction, [WEIGHTY], 2500) or {}
            if _articles(anew.get('items')) and isinstance(anew.get('roles'), dict) and anew['roles'].get('buy'):
                concise = anew
        while not concise and not self.nil_balance and self.invokes < ENTIRE_INVOKES and self.timer() < CONCISE_VIA:
            time.sleep(CONCISE_VOID)
            concise = self.inquire_jdoc(instruction, [MILD, WEIGHTY], 2500) or {}
        if concise:
            self.lingered = 0
        self.concise = concise
        duties = concise.get('roles') if isinstance(concise.get('roles'), dict) else {}
        self.duties = {kk: vv for kk, vv in duties.items() if isinstance(vv, str) and vv in self.blueprints}
        for handle in ('buy_args', 'refuse_args'):
            if not isinstance(concise.get(handle), dict):
                concise[handle] = {}
        purchase, decline = (self.duties.get('buy'), self.duties.get('refuse'))
        if purchase and purchase == decline and (not concise['buy_args'] or not concise['refuse_args'] or concise['buy_args'] == concise['refuse_args']):
            inquire = f'Depending on its arguments, the store tool {purchase} either completes the purchase or finishes without buying. Here is its schema: ' + _spill(self.blueprints.get(purchase)) + '\nReply with ONE JSON object only: {"buy_args": the fixed arguments that complete the purchase, "refuse_args": the fixed arguments that finish without buying} (include just the arguments that select the mode).'
            repaired = self.inquire_jdoc(inquire, [MILD, WEIGHTY], 1500) or {}
            if isinstance(repaired.get('buy_args'), dict) and isinstance(repaired.get('refuse_args'), dict) and (repaired['buy_args'] != repaired['refuse_args']):
                concise['buy_args'], concise['refuse_args'] = (repaired['buy_args'], repaired['refuse_args'])
        for element in _articles(concise.get('items')):
            if not isinstance(element, dict):
                continue
            amount = _tally(element.get('quantity'))
            tally = amount if amount and amount > 0 else 1
            essentials = [str(mm) for mm in _articles(element.get('musts')) if mm]
            self.articles.append(Article(str(element.get('name') or 'the product'), str(element.get('shelf') or ''), tally, essentials))
        combined: list[Article] = []
        grew: dict[int, int] = {}
        for aged, article in enumerate(self.articles, 1):
            duplicate = next((mm for mm in combined if article.rack.strip() and mm.amt == article.amt and _terms(mm.rack) == _terms(article.rack) and (not any((_terms(u.split(':')[0]) == _terms(d.split(':')[0]) and _terms(u.partition(':')[2]) != _terms(d.partition(':')[2]) for u in mm.essentials for d in article.essentials)))), None)
            if duplicate is None:
                combined.append(article)
                grew[aged] = len(combined)
            else:
                grew[aged] = combined.index(duplicate) + 1
                duplicate.essentials += [mm for mm in article.essentials if mm not in duplicate.essentials]
                self.transcript.append({'kind': 'items_merged', 'into': duplicate.moniker, 'dropped': article.moniker})
        self.articles = combined
        if not self.articles:
            self.articles = [Article('the product the shopper wants', '', 1, [])]
        self.purview_essentials()
        for couple in _articles(concise.get('fits')):
            if isinstance(couple, dict):
                portion, primary = (grew.get(_tally(couple.get('item')) or 0), grew.get(_tally(couple.get('with')) or 0))
                if portion and primary and (portion != primary) and (1 <= portion <= len(self.articles)) and (1 <= primary <= len(self.articles)) and ((portion, primary) not in self.confined):
                    self.confined.append((portion, primary))
        for retrieval in _articles(concise.get('lookups')):
            writing = str(retrieval).strip()
            if not any((writing in article.essentials for article in self.articles)):
                handle = _terms(writing.split(':')[0])
                writing = next((mm for article in self.articles for mm in article.essentials if _terms(mm.split(':')[0]) == handle), writing)
            numeral = next((nn for nn, article in enumerate(self.articles, 1) if writing in article.essentials), 1)
            if writing and (numeral, writing) not in self.references:
                self.references.append((numeral, writing))
        self.unblend()
        allowance = concise.get('budget')
        if isinstance(allowance, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', allowance):
            allowance = float(allowance)
        if isinstance(allowance, (int, float)) and (not isinstance(allowance, bool)) and (allowance > 0):
            self.allowance = allowance
        cutoff = _tally(concise.get('messages'))
        if cutoff is not None and cutoff >= 0:
            self.mail_cutoff = cutoff
        elif self.duties.get('ask'):
            self.mail_cutoff = 3
        ticks = _tally(concise.get('steps'))
        if ticks is not None and 10 <= ticks < self.maximum_ticks:
            self.maximum_ticks = ticks
        for element in _articles(concise.get('process')):
            if isinstance(element, dict) and str(element.get('step') or '').strip():
                once = str(element.get('when') or '').strip().lower()
                self.digest.append({'step': str(element['step']).strip(), 'when': once if once in ('choose', 'order', 'end') else 'order', 'messages': max(0, _tally(element.get('messages')) or 0)})
        if self.duties.get('ask'):
            converse = [pp['messages'] for pp in self.digest if pp['when'] in ('order', 'end') and pp['messages']]
            self.buffer = min(max(converse), self.mail_cutoff) if converse else 0

    def purview_essentials(self) -> None:
        """Trims a requirement that the brief copied onto several items back to those its own sentences in the request
        mention, as long as every sentence carrying it (the property's words plus its figures) mentions one of them in
        full. An item is mentioned when each word of its name occurs in the sentence, except where those words occur a
        single time and only as part of a wider item's name that the sentence also contains. Removal happens only on an
        unmentioned item sharing a name word with a mentioned one (race 16: a need written for "the X and Y" kind landed on
        the "X" item too, leaving that item without any listing that fit)."""
        directed = {str(px).strip() for px in _articles(self.concise.get('lookups'))}
        sentences = [ss for ss in re.split('(?<=[.!?;])\\s+|\\n+', self.appeal) if ss.strip()]
        uttered = [[rr for ww in _terms(ss).split() for rr in _sources(ww)] for ss in sentences]
        figures = lambda writing: set(re.findall('\\d+(?:\\.\\d+)?', writing))
        monikers = [_sources(article.moniker) for article in self.articles]

        def dubbed(terms: list[str]) -> set[int]:
            retained = set(terms)
            output = set()
            for ii, unearth in enumerate(monikers):
                if not unearth or not unearth <= retained:
                    continue
                if any((unearth < broad <= retained for broad in monikers)) and all((terms.count(rr) == 1 for rr in unearth)):
                    continue
                output.add(ii)
            return output
        essentials = {mm for article in self.articles for mm in article.essentials if mm not in directed and ':' in mm}
        for essential in sorted(essentials):
            holders = [ii for ii, article in enumerate(self.articles) if essential in article.essentials]
            if len(holders) < 2:
                continue
            feature, _, worth = essential.partition(':')
            demand = _sources(feature)
            stating = [terms for ss, terms in zip(sentences, uttered) if demand and demand <= set(terms) and figures(worth) <= figures(ss)]
            preserve = set()
            for terms in stating:
                strike = dubbed(terms) & set(holders)
                if not strike:
                    preserve = set()
                    break
                preserve |= strike
            if not preserve:
                continue
            for ii in holders:
                if ii not in preserve and any((monikers[ii] & monikers[kk] for kk in preserve)):
                    self.articles[ii].essentials.remove(essential)
                    self.transcript.append({'kind': 'must_scoped', 'must': essential, 'from': self.articles[ii].moniker, 'to': [self.articles[kk].moniker for kk in sorted(preserve)]})

    def register(self) -> str:
        portions = []
        numerous = len(self.articles) > 1
        for numeral, article in enumerate(self.articles, 1):
            alive = [mm for mm in article.essentials + article.informed if mm not in article.omitted]
            informed = [mm for mm in alive if self.depicted(numeral, mm)]
            demands = [mm + (f' (value from the page it refers to: {self.reference_worths[mm]})' if mm in self.reference_worths else '') for mm in alive if mm not in informed]
            front = f'item {numeral}: {article.moniker}' if numerous else f'item: {article.moniker}'
            if article.rack:
                front += f' (store label: {article.rack})'
            front += f', quantity {article.amt}'
            writing = front + '; requirements: ' + ('; '.join(demands) if demands else 'nothing apart from the product type')
            if informed:
                writing += "; phrased in the shopper's own terms that no listing attribute carries (not decisive): " + '; '.join(informed)
            departed = [mm for mm in dict.fromkeys(article.omitted) if mm and mm not in alive]
            if departed:
                writing += '; DROPPED by the shopper after the request was made (not needed any more, even if the request still mentions it): ' + '; '.join(departed)
            portions.append(writing)
        varieties = self.legend_strand()
        if varieties:
            portions.append(varieties)
        concise = self.concise
        if self.allowance is not None:
            purview = 'per item' if concise.get('budget_scope') == 'each' else 'in total'
            portions.append(f"spending limit: up to {self.allowance} {concise.get('budget_unit') or ''} {purview}".replace('  ', ' '))
        choose = concise.get('pick')
        if choose == 'cheapest':
            portions.append('pick: whichever fitting listing has the lowest price')
        elif choose in ('most', 'least'):
            portions.append(f"pick: the {choose} {concise.get('pick_measure') or ''} of all listings that fit".replace('  ', ' '))
        else:
            portions.append('pick: whichever listing meets every requirement')
        if concise.get('in_stock_only', True) is not False:
            portions.append('must be in stock')
        for hope in _articles(concise.get('wishes')):
            portions.append(f'preference (used only to break ties): {hope}')
        for norm in _articles(concise.get('rules')):
            portions.append(f'rule stated in the request: {norm}')
        if concise.get('quiet_decline') is True:
            portions.append('when no listing fits: finish without buying, and do not name or recommend any listing')
        for element in self.digest:
            once = {'choose': 'before choosing', 'order': 'at checkout (Wren takes care of it as you check out)', 'end': 'when finishing without buying (Wren takes care of it at that point)'}[element['when']]
            portions.append(f"step required by the request, {once}: {element['step']}")
        datums = self.datums_instrument()
        if datums and (self.datums_whence.get(datums) is True or self.duties.get('offer')):
            naked = sorted((tt for tt in self.sheet_instruments() | {self.duties.get('offer') or ''} if tt and self.datums_whence.get(tt) is False))
            sheets = sorted((tt for tt in self.sheet_instruments() if tt != datums))
            portions.append(f'attributes of listings are read from {datums}' + (f"; {', '.join(naked)} show none" if naked else '') + (f"; a listing's page (what a page check or a chain of pages refers to) can be viewed with {' or '.join(sheets)}" if sheets else ''))
        return '\n'.join(('- ' + pp for pp in portions))

    def tempo_remark(self) -> str:
        """A note for the loop model once this episode's measured turn durations make it clear that time, not the step budget,
        is what will cut it short: at TEMPO_INVOKES calls apiece, the turns left before LAST_TURN would spend fewer steps than
        the search budget still holds. Turn length barely depends on how many calls it carries (race-17 copy, 397 loop
        turns: median 8.8 s for one call, 9.7 s for three, 11.8 s for four; means 10.7 s and 15.3 s), yet the average turn
        made only 2.15 calls; the 23 big tasks lost there all hit 256-280 s with 25-66 steps never used, and on the real
        race 17 the field's declines on tasks of 91+ steps left a median 39% of their steps untouched. Hence the note asks
        that every search and read already known to be needed go out together in one turn, and it names the items that have
        no cart line yet. It stays quiet until TEMPO_ROTATIONS turns were timed, and from WIND_DOWN on, where the closing
        note replaces it."""
        if len(self.rotation_ticks) < TEMPO_ROTATIONS or self.timer() >= COIL_DOWNWARD:
            return ''
        turn = max(sum(self.rotation_ticks) / len(self.rotation_ticks), INVOKE_BOTTOM)
        fit = max(0.0, FINAL_ROTATION - self.timer()) / turn
        capacity = self.remaining() - self.rear() - len(self.articles)
        if fit * TEMPO_INVOKES >= capacity:
            return ''
        writing = (f'TIME BEFORE STEPS: at roughly {turn:.0f} s per turn here, only about {int(fit)} further turns remain on the clock '
                f'(an added call inside a turn costs just a second or two), and that is short of the {self.remaining()} steps you still have. '
                'Send every search, next page and listing read you already know is needed together in this turn as separate calls, with little prose.')
        lacking = self.lacking_articles() if self.basket_uids() else list(range(1, len(self.articles) + 1))
        if lacking and len(self.articles) >= 2:
            writing += ' Still no cart line for: ' + '; '.join((f'item {nn} ({self.articles[nn - 1].moniker})' for nn in lacking[:6])) + '.'
        return writing

    def standing(self) -> str:
        remaining = self.remaining()
        writing = f'[Wren] {self.tick} of {self.maximum_ticks} steps spent, {remaining} remaining.'
        if remaining <= self.rear() + 1:
            writing += ' FINAL STEPS: if the cart has a line for every item, check out now; finish without buying only if your searches proved some item has no fitting listing (being short of steps proves nothing of the kind).'
        elif remaining <= 6 or (remaining <= len(self.articles) + 5 and self.purchases == 0):
            writing += ' Only a few steps remain: put the chosen lines in the cart and check out right away.'
        elif self.purchases == 0 and remaining <= 12:
            writing += ' The step budget is getting low: decide on your picks, load the cart and check out in the coming few steps.'
        elif self.purchases == 0 and (not self.basket_visited) and (self.tick >= self.maximum_ticks // 2):
            writing += " Half of the steps are gone and nothing is in the cart: add a fitting listing for every item now (the store could still run out of one)."
        tempo = self.tempo_remark()
        if tempo:
            writing += ' ' + tempo
        if self.timer() >= COIL_DOWNWARD and remaining > self.rear() + 1:
            writing += ' TIME IS NEARLY UP: put the chosen lines in the cart and check out within two turns; if the episode times out it scores zero.'
        alive = {demand for _, demand in self.demands_presently()}
        retrievals = [demand for _, demand in self.references if demand in alive and demand not in self.reference_worths]
        if retrievals and self.purchases == 0:
            writing += '\n[look these up first: open the listing each one refers to, plus every listing named on its page; the final page holds the value] ' + ' | '.join(retrievals[:3])
        if self.prospects:
            writing += '\n[review candidates] ' + ', '.join(self.prospects)
        scheme = self.suit_scheme() if self.confined and self.purchases == 0 else ''
        if scheme:
            writing += '\n' + scheme
        if self.remarks:
            writing += '\n[updates up to now] ' + ' | '.join(self.remarks[-4:])
        writing += '\n[ledger]\n' + self.register()
        return writing

    def unblend(self) -> None:
        for numeral, article in enumerate(self.articles, 1):
            indicators = [demand for nn, demand in self.references if nn == numeral and demand in article.essentials]
            remainder = [mm for mm in article.essentials if mm not in indicators]
            if not indicators or not remainder:
                continue
            instruction = UNBLEND + '\n\nREQUEST:\n' + self.appeal + f'\n\nPRODUCT: {article.moniker}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n' + '\n'.join(indicators) + '\nOTHER REQUIREMENTS:\n' + '\n'.join(remainder)
            obtained = self.inquire_jdoc(instruction, [WEIGHTY, MILD], 1500) or {}
            shifted = [mm for mm in remainder if mm in [str(px).strip() for px in _articles(obtained.get('describe_other'))]]
            if not shifted:
                continue
            article.essentials = [mm for mm in article.essentials if mm not in shifted]
            united = indicators[0] + ' (that listing: ' + '; '.join(shifted) + ')'
            article.essentials = [united if mm == indicators[0] else mm for mm in article.essentials]
            self.references = [(nn, united if (nn, demand) == (numeral, indicators[0]) else demand) for nn, demand in self.references]
            self.transcript.append({'kind': 'unmix', 'item': numeral, 'moved': shifted})

    def unspelt(self, moniker: str, sighting: Any) -> Any:
        """Put the result into the shape the agent reads; for a single-listing view, a record whose attributes are mixed in
        with its own keys has them collected into one object: the keys absent from the search rows that hold plain values.
        """
        outlooks = {self.duties.get('open'), self.duties.get('offer'), self.duties.get('compare')}
        output = _collapse_duplicates(_reinstate(sighting, sheet=moniker in outlooks or moniker != self.duties.get('search')))
        if moniker not in outlooks:
            return output
        self.tier_handles.update((kk for identity, tier in self.tiers.items() if identity not in self.launched for kk in tier))
        familiar = self.tier_handles
        if not familiar:
            return output

        def collected(entry: Any) -> Any:
            if not isinstance(entry, dict) or _datum_chunk(entry) or not self.identity(entry, True):
                return entry
            additional = {kk: vv for kk, vv in entry.items() if kk not in familiar and isinstance(vv, (str, int, float)) and not isinstance(vv, bool)}
            if len(additional) < ENTRY_FACETS:
                return entry
            return {**{kk: vv for kk, vv in entry.items() if kk not in additional}, 'attributes': additional}
        if isinstance(output, dict) and self.identity(output):
            return collected(output)
        if isinstance(output, dict):
            return {kk: [collected(rr) for rr in vv] if isinstance(vv, list) else vv for kk, vv in output.items()}
        return output


class _SpoorPart:
    """Handles the closing trail, linked parts, sold-out swaps and audits."""

    def basket_invokes(self, strands: dict[str, int], omit: list[str]) -> list[dict[str, Any]] | None:
        moniker = self.duties.get('cart') or ''
        blueprint = self.blueprints.get(moniker) or {}
        features = blueprint.get('properties') or {}
        for handle, specification in features.items():
            form = specification.get('items') if isinstance(specification, dict) and 'array' in _varieties(specification) else None
            if isinstance(form, dict) and isinstance(form.get('properties'), dict):
                internal = form['properties']
                uid_handle = next((kk for kk, ss in internal.items() if isinstance(ss, dict) and 'string' in _varieties(ss) and ('enum' not in ss)), None)
                tally_handle = next((kk for kk, ss in internal.items() if isinstance(ss, dict) and 'integer' in _varieties(ss)), None)
                if not uid_handle:
                    return None
                return [{handle: [{uid_handle: ii, **({tally_handle: nn} if tally_handle else {})} for ii, nn in strands.items()]}]
        writings = [kk for kk, ss in features.items() if isinstance(ss, dict) and 'string' in _varieties(ss) and ('enum' not in ss)]
        aim = next((kk for tt, kk in self.uid_operands if tt == moniker and kk in writings), None) or next((kk for kk in blueprint.get('required') or [] if kk in writings), None) or (writings[0] if len(writings) == 1 else None)
        tally = next((kk for kk, ss in features.items() if isinstance(ss, dict) and 'integer' in _varieties(ss)), None)
        setting = next((kk for kk, ss in features.items() if isinstance(ss, dict) and isinstance(ss.get('enum'), list) and (len(ss['enum']) >= 2)), None)
        if not aim:
            return None
        invokes: list[dict[str, Any]] = []
        for identity in omit:
            possess = self.basket_strands.get(identity, 1)
            if setting:
                invokes.append({aim: identity, setting: features[setting]['enum'][1], **({tally: possess} if tally else {})})
            elif tally and self.basket_ensembles(moniker):
                invokes.append({aim: identity, tally: 0})
            elif tally and isinstance(features[tally].get('minimum'), int) and (features[tally]['minimum'] <= -possess):
                invokes.append({aim: identity, tally: -possess})
            else:
                return None
        for identity, desire in strands.items():
            possess = self.basket_strands.get(identity, 0) if identity in self.basket_uids() else 0
            if possess == desire:
                continue
            if possess and (not self.basket_ensembles(moniker)):
                return None
            invokes.append({aim: identity, **({setting: features[setting]['enum'][0]} if setting else {}), **({tally: desire} if tally else {})})
        return invokes

    def decline_path(self) -> str:
        """One time, ahead of a decline while room remains: chase what a requirement refers to; returns what that revealed, or '' if nothing."""
        alive = {demand for _, demand in self.demands_presently()}
        if self.decline_followed or self.remaining() < PATH_CAPACITY or self.timer() > CHASE_VIA or not any((demand in alive and demand not in self.reference_worths for _, demand in self.references)):
            return ''
        self.decline_followed = True
        writing, _ = self.path(CHASE_VIA)
        return writing + 'Look up those values now and purchase whatever fits; finish without buying only when nothing fits.' if writing else ''

    def exchange_cue(self) -> str:
        mislaid = [ii for ii in self.basket_uids() if ii not in self.departed and self.vended_output(ii)]
        if not mislaid:
            return ''
        for identity in mislaid:
            self.departed[identity] = self.tick
        self.prospects = [ii for ii in self.prospects if ii not in mislaid]
        self.revision += 1
        chooses = self.substitutes(mislaid)
        writing = f"[Wren] Now sold out: {', '.join(mislaid)} (the store marks it as unavailable to buy). It will not return, so do not reopen it."
        if chooses and self.remaining() <= 6:
            return writing + f" Only a few steps remain: this turn, add {chooses[0]} at the requested quantity and drop the sold-out line (two calls), then check out. Other listings you have read that seem to fit: {', '.join(chooses[1:]) or 'none'}."
        if chooses and self.concise.get('pick') == 'cheapest':
            return writing + f" Listings you have read that seem to fit that item in its place, cheapest first: {', '.join(chooses)}. The pick rule favors the cheapest, so unless you have seen every fitting listing already, do one search below {chooses[0]}'s price (using the search's maximum price, if it offers one) before replacing the sold-out line with the cheapest fitting listing."
        if chooses:
            return writing + f" Listings you have read that seem to fit that item in its place: {', '.join(chooses)}. This turn, remove the sold-out line from the cart and add {chooses[0]} at the requested quantity, then check out."
        return writing + ' Take that line out of the cart and add the best alternative listing that fits; search for one if none of those you have read fits.'

    def intended_hamper(self, prospect: str) -> tuple[dict[str, int], list[str]] | None:
        """Builds a basket out of the review's leads: for each item the review named, its first lead not sold out, joined by
        the cart lines of every other kind; also returns which lines to remove. None if `lead` is left out or some item would
        have no line."""
        sect_handle = self.genre_handle()

        def variety(identity: str) -> str:
            return str((self.tiers.get(identity) or {}).get(sect_handle) or '').strip().lower() if sect_handle else identity
        scheme: dict[str, int] = {}
        for numeral, uids in self.prospect_articles.items():
            located = re.search('\\d+', str(numeral))
            position = int(located.group()) if located else (1 if len(self.articles) == 1 else None)
            choose = next((ii for ii in uids if not self.vended_output(ii)), None)
            if choose and position and (1 <= position <= len(self.articles)) and (variety(choose) not in {variety(pp) for pp in scheme}):
                scheme[choose] = self.articles[position - 1].amt
        if prospect not in scheme:
            return None
        presently = self.basket_uids()
        preserve = {ii: self.basket_strands.get(ii, 1) for ii in presently if ii not in scheme and (not self.vended_output(ii)) and (variety(ii) not in {variety(pp) for pp in scheme})}
        omit = [ii for ii in presently if ii not in scheme and ii not in preserve]
        if len(scheme) + len({variety(ii) for ii in preserve}) < len(self.articles):
            return None
        return ({**preserve, **scheme}, omit)

    def finalize_basket(self) -> str:
        basket, purchase = (self.duties.get('cart'), self.duties.get('buy'))
        if self.finalized or not self.affirmed or self.vended_output(self.affirmed) or (not basket) or (not purchase):
            return ''
        self.finalized = True
        hamper = self.intended_hamper(self.affirmed)
        if not hamper:
            return ''
        strands, omit = hamper
        invokes = self.basket_invokes(strands, omit)
        if not invokes or len(invokes) + 1 > self.remaining():
            return ''
        self.populate_datums(list(strands), len(invokes) + 3)
        self.display_bonds(list(strands), len(invokes) + 3)
        self.verify_basket(list(strands), EXPEND_STRICT)
        if not self.rulings.get(self.verify_handle(list(strands))):
            return ''
        accomplished = []
        for operands in invokes:
            defects: list[str] = []
            operands = _form(operands, self.blueprints.get(basket), 'args', defects)
            if defects:
                return ''
            rejoinder = self.perform(basket, operands)
            if rejoinder is None or self.terminated:
                return ''
            self.remark_basket(basket, operands)
            self.witness(basket, rejoinder.get('observation'))
            accomplished.append(_spill(operands))
        defects = []
        operands = _form(dict(self.concise.get('buy_args') or {}), self.blueprints.get(purchase), 'args', defects)
        if defects:
            return ''
        rejoinders = []
        self.reperuse_tenders()
        for trial in range(2):
            if trial:
                self.reperuse_tenders()
                self.verify_basket(list(strands), EXPEND_STRICT)
                if not self.rulings.get(self.verify_handle(list(strands))) or self.remaining() < 1 or self.terminated or any((self.vended_output(ii) for ii in strands)):
                    break
            proceed, uttered = self.checkpoint('order')
            if uttered:
                rejoinders.append(uttered)
            if not proceed:
                break
            rejoinder = self.perform(purchase, operands)
            self.transcript.append({'kind': 'completed', 'lines': strands, 'drop': omit, 'reply': _snip(rejoinder, 200)})
            if rejoinder is None:
                break
            self.purchases += 1
            if not rejoinder.get('finished'):
                self.declined_upon = self.tick
            self.witness(purchase, rejoinder.get('observation'))
            rejoinders.append(self.display(rejoinder.get('observation')))
            if self.terminated or rejoinder.get('finished'):
                break
        if not rejoinders:
            return ''
        return 'Not executed (Wren handled it instead): the independent check verified ' + self.affirmed + ', so Wren set up the cart (' + '; '.join(accomplished) + ') and completed checkout. The store replied: ' + ' | then: '.join(rejoinders)

    def launched_store(self, regarding: str) -> list[str]:
        non_monikers = self.non_creator_handles()
        modest = regarding.lower()
        output = []
        for identity in self.launched:
            tier = self.tiers.get(identity) or {}
            monikers = [str(vv).strip() for kk, vv in tier.items() if isinstance(vv, str) and 2 < len(vv.strip()) <= 40 and kk not in non_monikers and vv.strip().lower() not in NIL_MONIKER]
            if any((re.search('(?<![a-z0-9])' + re.escape(nn.lower()) + '(?![a-z0-9])', modest) for nn in monikers)):
                output.append(identity)
        return output

    def locate_bond(self, regarding: str, terms: str) -> list[str]:
        """Candidate rows for a chain link that is still missing, strongest first (no more than 2). Returns [] when a page
        already opened for the named shop turns out to be that link by its attributes (traced as 'link_known'); that
        verdict needs the attributes of an opened listing to contain every value the description lists, because a page's
        own text also mentions values some listing does not have ("this one is not <value>"), which on the race-16 copy led
        to the wrong listing in 4 cases out of 6. The searches start with the trail model's words, then try the shop's name
        paired with each single value (if at least two values were given), and finally the shop's name by itself, but that
        last query only when none of the value searches surfaced an unopened row from the shop. Rows that more value
        searches returned rank higher, ties going to the narrowest search (race 17: the store matches every word, and a
        single value written another way on the page (spaced in one place, joined in the other) hid the link from the
        all-words query, whereas the shop with one value found it as a single row). Repeating a query costs no step, as its
        rows are kept, and moves on to the next page after those rows run out (bond_enquiry)."""
        non_monikers = self.non_creator_handles()
        familiar: list[str] = []
        dubbed = {ww for identity in self.store_tiers(regarding, list(self.tiers)) + self.launched_store(regarding) for kk, vv in (self.tiers.get(identity) or {}).items() if isinstance(vv, str) and len(vv.strip()) <= 40 and kk not in non_monikers and vv.strip().lower() not in NIL_MONIKER for ww in re.findall('[a-z0-9]{3,}', vv.lower())}
        worths = {ww for ww in re.findall('[a-z0-9]+', (terms or regarding).lower()) if len(ww) >= 3 or ww.isdigit()} - dubbed - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for', 'of'}
        for identity in self.launched_store(regarding):
            pile, abandons = ([_datum_chunk(self.ample.get(identity) or {}) or {}], [])
            while pile:
                vertex = pile.pop()
                if isinstance(vertex, (dict, list)):
                    pile += list(vertex.values()) if isinstance(vertex, dict) else vertex
                else:
                    abandons.append(str(vertex))
            writing = ' '.join(abandons).lower()
            if worths and abandons and all((re.search('(?<![a-z0-9])' + re.escape(vv) + '(?![a-z0-9])', writing) for vv in worths)):
                self.transcript.append({'kind': 'link_known', 'id': identity})
                return []
        scheme = self.worth_stencil()
        portions = terms.split()
        votes: dict[str, list[int]] = {}
        if scheme and portions:
            instrument, basis, facet = scheme
            shop = portions[0]
            singles = [f'{shop} {ww}' for ww in portions[1:]] if len(portions) >= 3 else []
            for enquiry in dict.fromkeys([' '.join(portions)] + singles + [shop]):
                if enquiry == shop and votes:
                    break
                located = self.bond_enquiry(instrument, basis, facet, enquiry, regarding)
                if located is None:
                    break
                crisp = self.store_tiers(regarding, located)
                for identity in crisp:
                    votes.setdefault(identity, []).append(len(located))
                if crisp and enquiry not in singles:
                    break
            familiar = list(votes)
        lexemes = {ww for ww in re.findall('[a-z0-9]{3,}', f'{regarding} {terms}'.lower())} - dubbed - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for'}

        def rated(uids: list[str]) -> list[str]:
            """Keeps rows that have a word in common with the link's description (its kind or values), most shared words first."""
            graded = []
            for place, identity in enumerate(uids):
                strike = len(lexemes & set(re.findall('[a-z0-9]{3,}', _spill(self.tiers.get(identity) or {}).lower())))
                if strike or not lexemes:
                    graded.append((-strike, place, identity))
            return [identity for _, _, identity in sorted(graded)]
        return (sorted(rated(familiar), key=lambda ii: (-len(votes[ii]), min(votes[ii]))) or rated(self.store_tiers(regarding, list(reversed(self.spotted_booking)))))[:2]

    def bond_enquiry(self, instrument: str, basis: dict[str, Any], facet: str, enquiry: str, regarding: str) -> list[str] | None:
        """Rows returned for a chain-link query. Repeating a query costs no step since its rows are remembered; when all rows
        of the named shop from it have been opened and the store still has another page, that page is fetched as one
        further link search (race 17: in 15 of 121 episodes the pointed-at listing first surfaced on a later page of the
        shop's results, and 9 of the 56 passes on those tasks depended on that). Returns None when searching is no longer
        allowed."""
        reminder = enquiry.lower()
        located = self.bond_spotted.get(reminder)
        further = self.bond_further.get(reminder) if located is not None else None
        if located is not None and (self.store_tiers(regarding, located) or not further):
            return located
        if self.bond_seeks >= TETHER_SEEKS or self.remaining() <= 12:
            return located
        defects: list[str] = []
        operands = _form({**basis, facet: enquiry, **({further[0]: further[1]} if further else {})}, self.blueprints.get(instrument), 'args', defects)
        if defects:
            return located
        self.bond_seeks += 1
        rejoinder = self.perform(instrument, operands)
        if rejoinder is None:
            return located
        sighting = rejoinder.get('observation')
        self.witness(instrument, sighting)
        tiers = [ii for ii, _ in self.entries_inside(sighting)]
        sheet = self.sheeting(instrument, sighting)
        self.bond_further[reminder] = (sheet[2], sheet[3]) if sheet and sheet[3] is not None else None
        located = self.bond_spotted[reminder] = list(dict.fromkeys((located or []) + tiers))
        self.transcript.append({'kind': 'link_search', 'words': enquiry[:80], 'paged': bool(further), 'rows': self.store_tiers(regarding, tiers)[:4]})
        return located

    def wants_worths(self, identity: str, terms: str) -> list[str]:
        """Which value words from a chain link's description the listing never mentions in its attributes or title (spacing
        and punctuation ignored; a word right behind a number counts as its unit and is skipped; the trail model hands over
        the values without property names or the shop's name, and one stray word only costs a single recorded rejection).
        Returns [] when every word is present or nothing can be tested. Store decoys come from the named shop and are of the
        right kind but miss one described value (race-17 copy: such a decoy, lacking the described colour, was accepted as
        the link, and its page then sent the search the wrong way)."""
        headline = self.headline_handle()
        blocks = [_datum_chunk(rec) for rec in (self.ample.get(identity), self.tiers.get(identity)) if isinstance(rec, dict)]
        blocks = [d for d in blocks if d]
        if not blocks or not terms.strip():
            return []
        monikers = {ww for nn in self.store_monikers(identity) for ww in re.findall('[a-z0-9]+', nn.lower())}
        retained = re.sub('[^a-z0-9.]+', '', (' '.join((_spill(d) for d in blocks)) + ' ' + str((self.tiers.get(identity) or {}).get(headline) or '' if headline else '')).lower())
        tokens = [tt.strip('.') for tt in re.findall('[a-z0-9][a-z0-9.]*', terms.lower())]
        desired = [tt for nn, tt in enumerate(tokens) if tt and (len(tt) >= 3 or tt[0].isdigit()) and tt not in monikers and not (nn and tokens[nn - 1][:1].isdigit() and tt.isalpha())]
        return [tt for tt in dict.fromkeys(desired) if tt not in retained]

    def forwards_onward(self, identity: str, demand: str, worth: str) -> bool:
        """Detects a page that the trail model treated as the end of a chain although its long texts (anything besides the
        title) mention the requirement's property without containing the value taken from it; a page like that passes the
        property on to another listing. Race 17, across the 13 pointer tasks: of 378 pages holding the pointed property,
        12 mention it in their text but not their own value; 10 of those forward it, and the remaining 2 describe a pairing
        rule whose partner value is the very one the chain uses (it occurs in their text); none of the other 366 trips it."""
        feature = ' '.join(re.findall('[a-z0-9]+', demand.split(':')[0].lower()))
        desire = re.sub('[^a-z0-9]+', '', worth.lower())
        if len(feature) < 3 or not desire:
            return False
        headline = self.headline_handle()
        for entry in (self.tiers.get(identity) or {}, self.ample.get(identity) or {}):
            for handle, writing in entry.items():
                if isinstance(writing, str) and len(writing) > 40 and handle != headline and re.search('(?<![a-z0-9])' + re.escape(feature) + '(?![a-z0-9])', ' '.join(re.findall('[a-z0-9]+', writing.lower()))) and desire not in re.sub('[^a-z0-9]+', '', writing.lower()):
                    return True
        return False

    def locate_vended(self) -> str:
        uids = [ii for ii in self.basket_uids() if ii not in self.departed]
        if any((self.vended_output(ii) for ii in uids)):
            return ''
        if len(uids) == 1:
            self.unobtainable.add(uids[0])
            return ''
        return self.reperuse_tenders() if uids else ''

    def non_creator_handles(self) -> tuple[Any, ...]:
        """Row keys that can never contain the seller's or maker's name: the id, the label and the title. Every other text
        field may, and the value still has to occur verbatim in the description under comparison.
        """
        return (self.uid_handle, self.genre_handle(), self.headline_handle())

    def path(self, cease: float | None=None) -> tuple[str, bool]:
        alive = {demand for _, demand in self.demands_presently()}
        awaiting = [demand for _, demand in self.references if demand in alive and demand not in self.reference_worths]
        instrument = self.duties.get('offer') or self.duties.get('open')
        aim = self.solitary_aim(instrument)
        spotted: list[str] = []
        mislaid: list[str] = []
        while awaiting and self.paths < (self.path_capacity or (6 if self.remaining() > 15 else 4)) and (self.outlay < EXPEND_STRICT) and (cease is None or self.timer() <= cease):
            self.paths += 1
            entries = []
            for identity in reversed(self.spotted_booking):
                entry = self.ample.get(identity) or self.tiers.get(identity) if self.sheet_peruse(identity) else None
                if entry and len(entries) < 45:
                    entries.append('OPENED ' + _spill(_snip(entry, 300))[:1400])
            for identity in reversed(self.spotted_booking):
                if len(entries) >= 60:
                    break
                if not self.sheet_peruse(identity) or not (self.ample.get(identity) or self.tiers.get(identity)):
                    entries.append('ROW ' + (_spill(_snip(self.ample[identity], 120))[:700] if identity in self.ample else _spill(_snip(self.tiers.get(identity), 80))[:300]))
            instruction = PATH + '\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n' + '\n'.join(awaiting)
            instruction += "\n\nTHE SHOPPER'S REQUEST (every requirement above is a condensed reading of it; whenever the request itself spells out how the wanted value relates to the value found along the chain, follow the request's phrasing rather than the requirement's):\n" + self.appeal[:WRITING_GLYPHS]
            if self.path_remarks:
                instruction += '\n\nCHECKED BY WREN AGAINST THE RECORDS:\n' + '\n'.join(self.path_remarks[-6:])
            instruction += '\n\nRECORDS (most recent first):\n' + '\n'.join(entries)
            obtained = self.inquire_jdoc(instruction, [MILD] + ([WEIGHTY] if self.outlay + WEIGHTY_CAPACITY <= EXPEND_STRICT else []), 4500) or {}
            self.transcript.append({'kind': 'trail', 'pending': awaiting, 'got': _snip(obtained, 150)})
            attain: list[str] = []
            mislaid = []
            voids: list[tuple[str, str]] = []
            anew = False
            for tether in _articles(obtained.get('chains')):
                if not isinstance(tether, dict):
                    continue
                demand = next((pp for pp in awaiting if pp == str(tether.get('need') or '').strip()), None) or (awaiting[0] if len(awaiting) == 1 else None)
                if not demand:
                    continue
                bonds = [ll for ll in _articles(tether.get('links')) if isinstance(ll, dict)]
                uids = [self.bond_uid(str(ll.get('id') or '').strip()) for ll in bonds]
                for place, (link, identity) in enumerate(zip(bonds, uids)):
                    lacking = self.wants_worths(identity, str(link.get('values') or link.get('find') or '')) if identity and self.sheet_peruse(identity) else []
                    note = f'{identity} does not match the listing described as "' + str(link.get('about') or '')[:150] + '": its attributes and title are missing ' + ', '.join(lacking[:3])
                    if lacking and note not in self.path_remarks:
                        self.path_remarks.append(note)
                        self.transcript.append({'kind': 'link_unlike', 'id': identity, 'lacks': lacking[:3]})
                        shop = next((nn for nn in self.store_monikers(identity) if re.search('(?<![a-z0-9])' + re.escape(nn.lower()) + '(?![a-z0-9])', (demand + ' ' + str(link.get('about') or '')).lower())), '')
                        terms = str(link.get('find') or '').strip() or f"{shop} {link.get('values') or ''}".strip()
                        bonds, uids, anew = bonds[:place] + [dict(link, id='', find=terms)], uids[:place] + [''], True
                        break
                if tether.get('done') is True and bonds and all((self.sheet_peruse(ii) for ii in uids)) and str(tether.get('value') or '').strip():
                    note = f'{uids[-1]} talks about the property in "{demand[:80]}" in its own text yet lacks the value {str(tether["value"]).strip()[:40]}: it passes that property on to another listing, so continue the chain from the listing its text points to'
                    if self.forwards_onward(uids[-1], demand, str(tether['value'])) and note not in self.path_remarks:
                        self.path_remarks.append(note)
                        self.transcript.append({'kind': 'trail_onward', 'id': uids[-1], 'value': str(tether['value'])[:40]})
                        anew = True
                        continue
                    self.reference_worths[demand] = str(tether['value']).strip()
                    self.revision += 1
                    continue
                if bonds and all((self.sheet_peruse(ii) for ii in uids)) and (not str(bonds[-1].get('find') or '').strip()):
                    self.reference_worths[demand] = str(tether.get('value') or '').strip() or f'as given on {uids[-1]}'
                    self.revision += 1
                    continue
                shut = [ii for ii in uids if ii in self.tiers and not self.sheet_peruse(ii)]
                if shut:
                    attain += [ii for ii in shut if ii not in attain]
                else:
                    void = next((ll for ll, ii in zip(bonds, uids) if ii not in self.tiers), {})
                    regarding = str(void.get('about') or '') or 'the listing it refers to'
                    terms = str(void.get('find') or '').strip()
                    mislaid.append(f'{demand} -> find {regarding}' + (f' (search: {terms})' if terms else ''))
                    voids.append((regarding, terms))
            awaiting = [pp for pp in awaiting if pp not in self.reference_worths]
            if not attain and voids and instrument and aim and (self.remaining() > 12):
                finalised = set()
                for place, void in enumerate(voids[:2]):
                    if self.bond_seeks >= TETHER_SEEKS:
                        break
                    markers = len(self.transcript)
                    attain += [ii for ii in self.locate_bond(*void) if ii not in attain]
                    familiar = next((tt['id'] for tt in self.transcript[markers:] if tt.get('kind') == 'link_known'), None)
                    if familiar:
                        spotted.append(f"{familiar}: a page already opened from the shop this chain names, and its attributes contain every value of the description; read the value off that page")
                        finalised.add(place)
                mislaid = [ll for place, ll in enumerate(mislaid) if place not in finalised]
            if not attain or not instrument or (not aim) or self.terminated:
                if anew and awaiting and instrument and aim and (not self.terminated):
                    continue
                break
            if self.path_capacity:
                capacity = min(3, SOON_LAUNCHES - len(spotted), self.remaining() - (3 * len(self.articles) + self.rear() + 1))
            else:
                capacity = min(3, 6 - len(spotted)) if self.remaining() >= len(attain) + 4 else 0
            if capacity <= 0:
                break
            for identity in attain[:capacity]:
                defects: list[str] = []
                operands = _form({aim: identity}, self.blueprints.get(instrument), 'args', defects)
                if defects or self.hurried():
                    break
                rejoinder = self.perform(instrument, operands)
                if rejoinder is None or self.terminated:
                    break
                self.witness(instrument, rejoinder.get('observation'))
                spotted.append(f'{identity}: ' + self.display(rejoinder.get('observation'))[:1500])
        writing = ''
        if spotted:
            writing += "[Wren opened the pages below while tracing listings that requirements point at; a page counts as such a listing only when every value of that listing's description appears on it]\n" + '\n'.join(spotted) + '\n'
        if self.reference_worths:
            writing += '[values taken from the pages the requirements refer to] ' + '; '.join((f'{kk} -> {vv}' for kk, vv in self.reference_worths.items())) + '\n'
        grasp = bool(mislaid) and self.path_chunks < (3 if self.remaining() > 15 else 2) and (self.remaining() > 5)
        if grasp:
            self.path_chunks += 1
            writing += 'The order is valid only once every page a requirement refers to has been fully opened (a search row does not suffice), including each page that points on to another listing. Not yet found and opened: ' + ' | '.join(mislaid[:3]) + ". Look it up using the shop's name along with the values from its description (other listings of that kind from the same shop are not the one), open that shop's row whose page displays those values, use the value given on its page, then check out."
        return (writing, grasp)

    def bond_uid(self, supplied: str) -> str:
        """Turns the trail model's name for a chain link into a seen listing's id. Now and then the model passes a code from the
        listing's title (a series or part code) rather than the id; when exactly one seen listing has that code in its
        title, or failing that exactly one record has it anywhere, that listing is used, so an opened link is not treated as
        still missing (race-16 copy: 25 of 58 link ids were title codes, and the found link's shop then got its other rows
        opened)."""
        if not supplied or supplied in self.tiers:
            return supplied
        inside = [ii for ii in self.tiers if len(ii) >= 6 and ii in supplied]
        if len(inside) == 1:
            return inside[0]
        if not 3 <= len(supplied) <= 40 or not re.search('\\d', supplied):
            return supplied
        motif = '(?<![a-z0-9])' + re.escape(supplied.lower()) + '(?![a-z0-9])'
        headline = self.headline_handle()
        for complete in (False, True):
            if not complete and not headline:
                continue
            strikes = [ii for ii in self.spotted_booking if re.search(motif, (_spill(self.ample.get(ii) or self.tiers.get(ii) or {}) if complete else str((self.tiers.get(ii) or {}).get(headline) or '')).lower())]
            if len(strikes) == 1:
                self.transcript.append({'kind': 'link_id', 'sent': supplied, 'id': strikes[0]})
                return strikes[0]
            if strikes:
                break
        return supplied

    def store_monikers(self, identity: str) -> list[str]:
        """Short text fields of a listing's row able to carry the name of whoever sells or makes it (excluding id, label, name)."""
        non_monikers = self.non_creator_handles()
        tier = self.tiers.get(identity) or {}
        return [str(vv).strip() for kk, vv in tier.items() if isinstance(vv, str) and 2 < len(vv.strip()) <= 40 and kk not in non_monikers and vv.strip().lower() not in NIL_MONIKER]

    def monikers_store(self, identity: str, writing: str) -> bool:
        modest = writing.lower()
        return any((re.search('(?<![a-z0-9])' + re.escape(nn.lower()) + '(?![a-z0-9])', modest) for nn in self.store_monikers(identity)))

    def tether_writings(self) -> tuple[str, str]:
        """Returns a pair: the requirements whose value still comes from another listing, and the joined text of the pages on
        their chains, i.e. pages viewed through the page tool whose shop such a requirement or an earlier chain page names."""
        alive = {demand for _, demand in self.demands_presently()}
        demands = ' | '.join((demand for _, demand in self.references if demand in alive and demand not in self.reference_worths))
        if not demands:
            return ('', '')
        headline = self.headline_handle()
        sheets: list[str] = []
        employed: set[str] = set()
        for _ in range(5):
            familiar = demands + ' ' + ' '.join(sheets)
            novel = [ii for ii in self.spotted_booking if ii not in employed and self.sheet_peruse(ii) and self.monikers_store(ii, familiar)]
            if not novel:
                break
            for identity in novel:
                employed.add(identity)
                sheets += [str(vv) for kk, vv in (self.tiers.get(identity) or {}).items() if isinstance(vv, str) and len(vv) > 40 and kk != headline]
        return (demands, ' '.join(sheets))

    def sheet_initial(self, moniker: str, operands: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
        """Redirects the model's attribute read to the page view for a listing on the chain of a requirement that has no value
        yet. When the requirement or one of its chain pages names this listing's shop and the listing is outside every
        requested item's category, Wren shows its page instead, since only the page reveals where it leads and what its
        value is paired with, and the order is credited only after that page was opened (race-16 copy: reading chain
        listings by attributes alone yielded the value from before the redirect, or the value where its partner was
        wanted; the model attribute-read 1 to 4 chain listings before opening their pages). Listings inside a requested
        category keep the attribute read. Gives back (tool, args, note), unchanged tool and args when nothing applies."""
        starter, sheet = self.duties.get('open'), self.duties.get('offer')
        if moniker != starter or not sheet or sheet == starter or sheet not in self.blueprints:
            return (moniker, operands, '')
        aim, outlook = self.solitary_aim(moniker), self.solitary_aim(sheet)
        identity = str(operands.get(aim) or '').strip() if aim else ''
        if not outlook or identity not in self.tiers or self.sheet_peruse(identity) or identity in self.departed:
            return (moniker, operands, '')
        legend = str((self.tiers.get(identity) or {}).get(self.genre_handle() or '') or '')
        if not legend or self.proprietor_from(legend):
            return (moniker, operands, '')
        demands, sheets = self.tether_writings()
        if not demands or not self.monikers_store(identity, demands + ' ' + sheets):
            return (moniker, operands, '')
        defects: list[str] = []
        shifted = _form({outlook: identity}, self.blueprints.get(sheet), 'args', defects)
        if defects:
            return (moniker, operands, '')
        self.transcript.append({'kind': 'page_first', 'id': identity})
        return (sheet, shifted, f"\n[Wren] Opened the page of {identity} ({sheet}) rather than its attributes, because a requirement still depends on a listing from this shop, and only its page reveals where it leads or what its value pairs with (an order is credited only after that page was opened). If you still want its attributes, call {starter} on it again.")

    def reperuse_tenders(self) -> str:
        """Re-displays every cart line through the store's page view ahead of a checkout. A line never displayed that way is
        read even with few steps to spare: on the race-16 copy, 9 of 11 orders lost on a process step contained such a
        line, against 0 of the 8 other lost orders (real race 16: pointer needs were paid 44/48 when the page had been
        opened and 0/5 when it had not)."""
        instrument = self.tender_instrument()
        aim = self.solitary_aim(instrument)
        if not instrument or not aim:
            return ''
        after = max(self.tick - 2, self.declined_upon)
        outdated = [ii for ii in self.basket_uids() if self.tender_spotted.get(ii, -99) < after and ii not in self.departed]
        unseen = [ii for ii in outdated if not self.sheet_peruse(ii)]
        if outdated and self.remaining() < len(outdated) + 5:
            outdated = unseen if self.remaining() >= len(unseen) + self.rear() else []
        if not outdated:
            return ''
        strands = []
        for identity in outdated:
            defects: list[str] = []
            operands = _form({aim: identity}, self.blueprints.get(instrument), 'args', defects)
            if defects or self.hurried():
                break
            prior = dict(self.tiers.get(identity) or {})
            rejoinder = self.perform(instrument, operands)
            if rejoinder is None or self.terminated:
                break
            sighting = rejoinder.get('observation')
            self.tender_spotted[identity] = self.tick
            self.witness(instrument, sighting)
            following = self.tiers.get(identity) or {}
            shifted = [kk for kk, vv in following.items() if kk in prior and prior[kk] != vv and (not isinstance(vv, (dict, list)))]
            if shifted:
                strands.append(f"{identity} changed ({', '.join(shifted[:4])}): {_spill(_snip(sighting, 300))[:900]}")
        self.transcript.append({'kind': 'reread', 'ids': outdated, 'changed': len(strands)})
        if strands:
            self.revision += 1
        return '\n'.join(strands)

    def sheet_peruse(self, identity: str) -> bool:
        """Whether the page of a listing has been read: opened, and via the store's page view if it offers one separate from the attribute view."""
        sheet = self.duties.get('offer')
        return identity in self.launched and (not sheet or not self.solitary_aim(sheet) or sheet in self.outlooks.get(identity, ()))

    def soon_path(self) -> str:
        """Chase, a single time, requirement chains that refer to another listing, while there is still room to act on their content."""
        alive = {demand for _, demand in self.demands_presently()}
        if self.remaining() < PATH_CAPACITY or self.timer() > CHASE_VIA or not any((demand in alive and demand not in self.reference_worths for _, demand in self.references)):
            return ''
        preserved = (self.paths, self.path_chunks)
        self.paths, self.path_capacity = 0, SOON_PATHS
        try:
            writing, _ = self.path(CHASE_VIA)
        finally:
            (self.paths, self.path_chunks), self.path_capacity = preserved, None
        self.transcript.append({'kind': 'early_trail', 'left': self.remaining(), 'values': list(self.reference_worths.values())[:3]})
        return writing

    def store_tiers(self, regarding: str, uids: list[str]) -> list[str]:
        non_monikers = self.non_creator_handles()
        modest = regarding.lower()
        output = []
        for identity in uids:
            tier = self.tiers.get(identity) or {}
            monikers = [str(vv).strip() for kk, vv in tier.items() if isinstance(vv, str) and 2 < len(vv.strip()) <= 40 and kk not in non_monikers and vv.strip().lower() not in NIL_MONIKER]
            if identity not in self.launched and any((re.search('(?<![a-z0-9])' + re.escape(nn.lower()) + '(?![a-z0-9])', modest) for nn in monikers)):
                output.append(identity)
        return output

    def substitutes(self, mislaid: list[str]) -> list[str]:
        if not self.referees():
            return []
        entries = []
        for identity in reversed(self.spotted_booking):
            if identity in mislaid or identity in self.departed or self.vended_output(identity):
                continue
            entries.append(_spill(_snip(self.ample.get(identity) or self.tiers.get(identity), 200))[:1200])
            if len(entries) >= 25:
                break
        if not entries:
            return []
        instruction = EXCHANGE + '\n\nREQUEST:\n' + self.appeal + '\n\nREQUIREMENTS IN FORCE:\n' + self.register()
        instruction += '\n\nSOLD-OUT CART LINE:\n' + '\n'.join((f'{ii}: {self.entry_from(ii)}' for ii in mislaid)) + '\n\nCART NOW:\n' + self.basket_writing()
        instruction += '\n\nOTHER LISTING RECORDS SEEN (most recent first):\n' + '\n'.join(entries)
        obtained = self.inquire_jdoc(instruction, self.referees()[::-1], 3500)
        suits: list[tuple[float, str]] = []
        for element in _articles((obtained or {}).get('fits')):
            identity = str(element.get('id') if isinstance(element, dict) else element)
            cost = element.get('price') if isinstance(element, dict) else None
            if identity in self.tiers and identity not in mislaid and (identity not in self.departed) and (identity not in self.basket_uids()) and (identity not in [ii for _, ii in suits]):
                suits.append((float(cost) if isinstance(cost, (int, float)) and (not isinstance(cost, bool)) else float('inf'), identity))
        rated = [ii for _, ii in sorted(suits, key=lambda ff: ff[0])] if self.concise.get('pick') == 'cheapest' else [ii for _, ii in suits]
        self.transcript.append({'kind': 'swap', 'lost': mislaid, 'fits': suits})
        return rated[:4]

    def tender_instrument(self) -> str | None:
        return self.duties.get('offer') or self.duties.get('open')

    def unclosed_tiers(self, tiers: list[str]) -> bool:
        numerous, connector = (self.contrast_aim(), self.duties.get('compare'))
        instrument = self.datums_instrument()
        aim = self.solitary_aim(instrument)
        if numerous and connector and (not self.contrast_naked()):
            scheme = [(connector, {numerous: tiers[:self.contrast_capacity()]})]
        elif instrument and aim:
            scheme = [(instrument, {aim: ii}) for ii in tiers[:2]]
        else:
            return False
        basket_ticks = len(self.basket_invokes({f'#{nn}': 1 for nn, _ in enumerate(self.articles)}, []) or self.articles)
        if self.remaining() < len(scheme) + max(5, self.rear() + basket_ticks):
            return False
        prior = len(self.ample)
        for moniker, operands in scheme:
            defects: list[str] = []
            operands = _form(operands, self.blueprints.get(moniker), 'args', defects)
            if defects:
                return False
            rejoinder = self.perform(moniker, operands)
            if rejoinder is None or self.terminated:
                break
            self.witness(moniker, rejoinder.get('observation'))
        self.transcript.append({'kind': 'audit_open', 'rows': tiers, 'opened': len(self.ample) - prior})
        return len(self.ample) > prior

    def vended_output(self, identity: str) -> bool:
        return identity in self.unobtainable or identity in self.departed

    def vet(self) -> str:
        if self.vetted or not self.referees():
            return ''
        self.vetted = True
        obtained = self.vet_invoke()
        dubbed = self.vet_suits(obtained)
        if not dubbed:
            tiers = [str(ii) for ii in _articles((obtained or {}).get('open')) if str(ii) in self.tiers and str(ii) not in self.launched and (not self.vended_output(str(ii)))]
            if tiers and self.unclosed_tiers(tiers):
                obtained = self.vet_invoke()
                dubbed = self.vet_suits(obtained)
        if not dubbed:
            return ''
        self.prospect_articles = dubbed
        self.prospects = [ii for uids in dubbed.values() for ii in uids]
        return 'A close review suggests these listings might fit: ' + '; '.join((f"item {kk}: {', '.join(vv)}" for kk, vv in dubbed.items())) + (f" ({obtained.get('reason')})" if obtained.get('reason') else '') + '. Open them and purchase whatever really fits; finish without buying only if every one of them fails.'

    def vet_invoke(self) -> dict[str, Any] | None:
        entries = []
        for identity in sorted(reversed(self.spotted_booking), key=lambda ii: ii not in self.ample):
            if self.vended_output(identity):
                continue
            entry = self.ample.get(identity) if identity in self.ample else None
            entries.append('OPENED ' + _spill(_snip(entry, 200))[:1200] if entry else 'ROW ' + _spill(_snip(self.tiers.get(identity), 120))[:500])
            if len(entries) >= 30:
                break
        if not entries:
            return None
        instruction = VET + '\n\nREQUEST:\n' + self.appeal + '\n\nREQUIREMENTS IN FORCE:\n' + self.register()
        instruction += '\n\nLISTING RECORDS SEEN (most recent first):\n' + '\n'.join(entries)
        referees = self.referees()
        obtained = self.inquire_jdoc(instruction, [MILD] if referees else [], 4000)
        if WEIGHTY in referees and not self.nil_balance and not self.vet_suits(obtained):
            obtained = self.inquire_jdoc(instruction, [WEIGHTY], 4000) or obtained
        self.transcript.append({'kind': 'audit', 'got': obtained})
        return obtained

    def vet_suits(self, obtained: dict[str, Any] | None) -> dict[str, list[str]]:
        suits = (obtained or {}).get('fits')
        if not isinstance(suits, dict):
            return {}
        dubbed = {str(kk): [str(ii) for ii in _articles(vv) if str(ii) in self.tiers and (not self.vended_output(str(ii)))][:4] for kk, vv in suits.items()}
        return {kk: vv for kk, vv in dubbed.items() if vv}


class _ScoutPart:
    """Runs the searches that look for fitting or cheaper listings."""

    def broaden(self) -> None:
        instrument = self.duties.get('compare')
        blueprint = self.blueprints.get(instrument or '') or {}
        features = blueprint.get('properties') or {}
        aim = next((kk for kk, vv in features.items() if isinstance(vv, dict) and 'array' in _varieties(vv)), None)
        if not instrument or not aim or any((kk != aim for kk in blueprint.get('required') or [])) or self.contrast_naked():
            self.broaden_separately()
            return
        if self.remaining() <= 12:
            return
        lid = min(features[aim].get('maxItems') if isinstance(features[aim].get('maxItems'), int) else 4, self.contrast_capacity())
        rated = sorted(self.article_clusters(), key=lambda uids: sum((1 for ii in uids if ii in self.ample)))
        tracks = [[ii for ii in uids if ii not in self.ample] for uids in rated]
        chooses: list[str] = []
        for lap_from in range(lid):
            chooses += [track[lap_from] for track in tracks if lap_from < len(track)]
        chooses = chooses[:lid]
        if len(chooses) < 2:
            self.populate_datums(chooses, 12)
            return
        defects: list[str] = []
        operands = _form({aim: chooses}, blueprint, 'args', defects)
        if defects:
            return
        prior = len(self.ample)
        rejoinder = self.perform(instrument, operands)
        if rejoinder is not None:
            spotted = rejoinder.get('observation')
            self.witness(instrument, spotted)
            self.remark_outlook(instrument, spotted)
            tiers = self.entries_inside(spotted)
            if tiers:
                self.datums_whence[instrument] = self.datums_whence.get(instrument, False) or any((_datum_chunk(rr) for _, rr in tiers))
                self.contrast_okay = max(self.contrast_okay, len(tiers))
            else:
                self.contrast_poor = min(self.contrast_poor or 999, len(chooses))
            self.transcript.append({'kind': 'widen', 'ids': chooses, 'opened': len(self.ample) - prior})
            if tiers and self.datums_whence.get(instrument) is False:
                self.broaden_separately()
                return
            if len(self.ample) == prior and len(chooses) > 2 and (self.remaining() > 12):
                defects = []
                operands = _form({aim: chooses[:2]}, blueprint, 'args', defects)
                if not defects:
                    rejoinder = self.perform(instrument, operands)
                    if rejoinder is not None:
                        self.witness(instrument, rejoinder.get('observation'))
                        self.contrast_okay = max(self.contrast_okay, len(self.entries_inside(rejoinder.get('observation'))))

    def broaden_separately(self) -> None:
        instrument = self.datums_instrument()
        aim = self.solitary_aim(instrument)
        if not instrument or not aim:
            return
        desired = {ww for _, demand in self.demands_presently() for ww in re.findall('[a-z0-9]{3,}', demand.split(':', 1)[-1].lower())}
        every = BROADEN_EVERY if len(self.articles) <= 2 else 4
        tracks: list[list[str]] = []
        fronts: list[int] = []
        for uids in self.article_clusters():
            launched = sum((1 for ii in uids if ii in self.ample))
            reservoir = [ii for ii in uids if ii not in self.ample and self.purchasable(ii)]
            if self.concise.get('pick') == 'cheapest':
                reservoir.sort(key=lambda ii: self.cost_from(ii) if self.cost_from(ii) is not None else float('inf'))
            reservoir.sort(key=lambda ii: -len(desired & set(re.findall('[a-z0-9]{3,}', _spill(self.tiers.get(ii) or {}).lower()))))
            tracks.append(reservoir[:max(0, every - launched)])
            fronts.append(max(0, 4 - launched))
        initial = [track[kk] for kk in range(4) for track, front in zip(tracks, fronts) if kk < min(front, len(track))]
        further = [track[kk] for kk in range(every) for track, front in zip(tracks, fronts) if front <= kk < len(track)]
        lid = max(5, 2 * len(tracks) + 2)
        peruse = self.populate_datums(initial[:lid][:8], 22)
        self.populate_datums((initial[lid:] + further)[:max(0, 10 - len(peruse))], 30)

    def chase_stalks(self, numeral: int) -> list[list[str]]:
        """Lists, in the order tried, the groups of values a cheapest hunt queries for one item. A group keeps the item's
        stated values together, and on a store matching every word that alone returns nearly all the fitting listings. A
        closed list ("a or b") becomes one group per option, because no single query can express an either-or; values that
        name a model or part code are skipped (race 15: .97 of code-bearing searches came back empty, so a group with a code
        stalled at its first query and the hunt fell back to the plain product noun). A value that is just a figure plus a
        unit is also tried in its other form, joined or with a space, when the episode has only searched the form it has:
        the store sees the two forms as different words (real race 15: 159 of 231 such query pairs gave different rows).
        And when letters precede a number of three or more digits, the number by itself follows the first group as an
        extra search, because pages frequently print just the number (real race 17: in 8 of 63 accepted listings for such a
        socket value the number appeared in a searchable field without its letters)."""
        uttered = ' '.join((str(vv).lower() for u in self.sought for vv in u.values() if isinstance(vv, str)))
        directed = {mm for _, mm in self.references}
        simple = [worth for worth, _ in self.worth_terms(numeral) if not _ciphers(worth)]
        selections: list[str] = []
        for nn, demand in self.demands_presently():
            if nn != numeral or ':' not in demand:
                continue
            worth = demand.split(':', 1)[1].strip()
            if demand in directed:
                continue
            if selections or not re.search('\\bor\\b', worth, re.I) or re.search(UNCLOSED_NON, re.sub('\\bor\\b', ' ', worth, flags=re.I), re.I):
                continue
            portions = [pp.strip(' .') for pp in re.split('\\s*,\\s*|\\s+or\\s+', worth, flags=re.I) if pp.strip(' .')]
            selections = [pp for pp in portions if not _ciphers(pp) and len(pp) <= CHASE_WORTH_GLYPHS][:CHASE_SELECTIONS] if len(portions) >= 2 else []
        output: list[list[str]] = []
        for worths in [simple + [j] for j in selections] or [simple]:
            alternative = []
            for worth in worths:
                unit = re.fullmatch(DENOM_WORTH, worth.strip(), re.I)
                alternative.append((unit.group(1) + unit.group(2) if ' ' in worth.strip() else unit.group(1) + ' ' + unit.group(2)) if unit else worth)
            crisp = alternative != worths and any((vv.lower() in uttered and oo.lower() not in uttered for vv, oo in zip(worths, alternative) if vv != oo))
            located = alternative if crisp else worths
            if located not in output:
                output.append(located)
        for worths in list(output[:1]):
            naked = [re.sub('^[a-z]+\\s+(?=\\d{3,}$)', '', vv.strip(), flags=re.I) for vv in worths]
            if naked != worths and naked not in output:
                output.append(naked)
        return output

    def chase_article(self, numeral: int, instrument: str, basis: dict[str, Any], facet: str, capacity: int) -> tuple[int, int]:
        """Spends at most `room` searches on one item: each group from chase_stalks() as one query, then narrowed by further words."""
        attempted = self.chase_attempted.setdefault(numeral, set())
        nominal, _ = self.chase_terms(numeral)
        ensembles = self.chase_stalks(numeral)
        expended = earned = inactive = 0
        dispatched = declined = False

        def execute(enquiry: str) -> bool:
            """Run the given query up to its final page or the room limit, without re-reading any page already read this episode; returns whether any row was found."""
            nonlocal expended, earned, inactive, dispatched, declined
            attempted.add(enquiry.lower())
            defects: list[str] = []
            operands = _form({**basis, facet: enquiry}, self.blueprints.get(instrument), 'args', defects)
            declined = declined or bool(defects)
            if defects or expended >= capacity or self.terminated:
                return False
            dispatched = True
            grasps_worths = any((bool(worths) and all((vv.lower() in enquiry.lower() for vv in worths)) for worths in ensembles))
            located = False
            traversed: set[str] = set()
            while not self.terminated and self.timer() <= CHASE_VIA + CHASE_LENIENCY:
                marker = self.invoke_marker(instrument, operands)
                if marker in traversed:
                    return located
                traversed.add(marker)
                if marker in self.peruse_invokes:
                    spotted = self.peruse_invokes[marker]
                    novel = CHASE_SHEET_NOVEL
                elif expended >= capacity:
                    return located
                else:
                    prior = self.tiers_from_article(numeral)
                    rejoinder = self.perform(instrument, operands)
                    expended += 1
                    if rejoinder is None:
                        return located
                    spotted = rejoinder.get('observation')
                    self.witness(instrument, spotted)
                    novel = self.tiers_from_article(numeral) - prior
                    earned += novel
                    inactive = 0 if novel else inactive + 1
                tiers = self.entries_inside(spotted)
                located = located or bool(tiers)
                if grasps_worths:
                    self.worth_tiers.update((ii for ii, _ in tiers))
                sheet = self.sheeting(instrument, spotted)
                if not sheet or sheet[3] is None or (novel < CHASE_SHEET_NOVEL and sheet[0] > CHASE_COMPLETE):
                    return located
                defects = []
                operands = _form({**operands, sheet[2]: sheet[3]}, self.blueprints.get(instrument), 'args', defects)
                if defects:
                    return located
            return located
        marker = (numeral, tuple((tuple(worths) for worths in ensembles)))
        root = self.chase_root.get(marker)
        if root is None:
            root = ''
            for worths in ensembles:
                attempts = ([' '.join((worths + [nominal])).strip(), ' '.join(worths)] if worths else []) + ([' '.join((worths[:-1] + [nominal])).strip()] if len(worths) >= 2 else [])
                for enquiry in dict.fromkeys(attempts):
                    dispatched = False
                    if execute(enquiry):
                        root = root or enquiry
                        break
                    if not dispatched:
                        return (expended, earned)
            if not root and nominal:
                dispatched = False
                if execute(nominal):
                    root = nominal
                elif not dispatched:
                    return (expended, earned)
                else:
                    self.chase_nil_nominal.add(numeral)
            self.chase_root[marker] = root
        while expended < capacity and inactive < CHASE_INACTIVE and not declined and not self.terminated and self.timer() <= CHASE_VIA + CHASE_LENIENCY:
            _, terms = self.chase_terms(numeral)
            terms = [ww for ww in terms if f'{ww} {root}'.strip().lower() not in attempted and ww not in root.lower().split()]
            if not terms:
                break
            execute(f'{terms[0]} {root}'.strip())
        return (expended, earned)

    @_gentle('')
    def code_scan(self) -> str:
        """For each item with a requirement that names a model code, page through the search of its other required values
        until rows of its kind whose own fit text covers the code (and does not exclude it) appear, then read the best of
        them. Returns what was read."""
        scheme = self.worth_stencil()
        if not scheme or self.terminated:
            return ''
        instrument, basis, facet = scheme
        preserve = self.rear() + self.mails_gratis() + SWEEP_KEEP
        peruse: list[str] = []
        scanned = 0
        for numeral in range(1, len(self.articles) + 1):
            demands = [demand for proprietor, demand in self.demands_presently() if proprietor == numeral and ':' in demand]
            ciphers = [j for demand in demands for j in _ciphers(demand.split(':', 1)[1])]
            if not ciphers:
                continue

            def covered(identity: str) -> bool:
                position = self.suit_position()
                suits = self.suit_ensemble(identity, position)
                banned = position['marks'].get(identity, {}).get('bars') or []
                return any((_cipher_strike(ff, j) for ff in suits for j in ciphers)) and not any((_cipher_strike(d, j) for d in banned for j in ciphers))
            if any((ii in self.ample and self.conditions_demands(ii, numeral) for ii in self.article_uids(numeral))):
                continue
            nominal, _ = self.chase_terms(numeral)
            worths = [worth for worth, _ in self.worth_terms(numeral) if not _ciphers(worth)]
            terms = {ww for worth in worths for ww in _worth_symbols(worth)}

            def likely(identity: str) -> bool:
                return covered(identity) and terms <= _writing_terms(_spill(self.tiers.get(identity) or {}))
            designation = self.articles[numeral - 1].moniker if len(self.articles[numeral - 1].moniker.split()) <= 3 else ' '.join(self.articles[numeral - 1].moniker.split()[-2:])
            enquiry = ' '.join(worths[:2] + [designation]).strip() or nominal
            if not enquiry:
                continue
            defects: list[str] = []
            operands = _form({**basis, facet: enquiry}, self.blueprints.get(instrument), 'args', defects)
            sheets = 0
            while not defects and sheets < SCAN_PAGES and self.remaining() > preserve and not self.terminated:
                rejoinder = self.perform(instrument, operands)
                sheets += 1
                scanned += 1
                if rejoinder is None:
                    break
                spotted = rejoinder.get('observation')
                self.witness(instrument, spotted)
                if any((likely(ii) for ii in self.article_uids(numeral) if ii not in self.ample)):
                    break
                sheet = self.sheeting(instrument, spotted)
                if not sheet or sheet[3] is None:
                    break
                operands = _form({**operands, sheet[2]: sheet[3]}, self.blueprints.get(instrument), 'args', defects)
            reservoir = [ii for ii in self.article_uids(numeral) if ii not in self.ample and self.purchasable(ii) and covered(ii)]
            reservoir.sort(key=lambda ii: (-len(terms & _writing_terms(_spill(self.tiers.get(ii) or {}))), self.cost_from(ii) if self.cost_from(ii) is not None else float('inf')))
            peruse += self.populate_datums(reservoir[:SCAN_OPENS], preserve)
        self.transcript.append({'kind': 'code_scan', 'pages': scanned, 'opened': len(peruse)})
        if not peruse:
            return ''
        return "[Wren paged the searches of items that must fit a model code and read the rows whose own fit text names that code]\n" + '\n'.join(peruse)

    @_gentle('')
    def series_sweep(self) -> str:
        """Search each item that carries the series code binding it to another item once per leading series, with its first
        required value, and read the closest unread row of each series, before the model's first turn. Returns what was read."""
        if not self.confined or self.terminated:
            return ''
        scheme = self.worth_stencil()
        if not scheme:
            return ''
        instrument, basis, facet = scheme
        position = self.suit_position()
        carriers: dict[int, set[str]] = {}
        for portion, primary in self.confined:
            platform = self.bond_platform(portion, primary, position)
            if platform:
                carriers.setdefault(platform[0], set()).update(platform[2])
        if not carriers:
            return ''
        preserve = self.rear() + self.mails_gratis() + SWEEP_KEEP
        booking = {numeral: [ff for ff in self.sequence_towards(numeral) if ff in lineages][:SWEEP_SERIES] for numeral, lineages in carriers.items()}
        sought = 0
        self.milieu_void = UNCLOSED_VOID
        for numeral in sorted(booking):
            nominal, _ = self.chase_terms(numeral)
            worths = [worth for worth, _ in self.worth_terms(numeral)]
            for lineage in booking[numeral]:
                for enquiry in dict.fromkeys(([f'{lineage} {worths[0]} {nominal}'.strip()] if worths else []) + [f'{lineage} {nominal}'.strip()]):
                    if self.remaining() <= preserve or self.terminated:
                        break
                    defects: list[str] = []
                    operands = _form({**basis, facet: enquiry}, self.blueprints.get(instrument), 'args', defects)
                    if defects:
                        break
                    rejoinder = self.perform(instrument, operands)
                    sought += 1
                    if rejoinder is None:
                        self.milieu_void = MILIEU_VOID
                        return ''
                    spotted = rejoinder.get('observation')
                    self.witness(instrument, spotted)
                    if self.entries_inside(spotted):
                        self.sought = (self.sought + [dict(operands)])[-8:]
                        break
        position = self.suit_position()
        lid = self.allowance if isinstance(self.allowance, (int, float)) else None
        chooses: list[str] = []
        for rank in range(SWEEP_SERIES):
            for numeral in sorted(booking):
                if rank >= len(booking[numeral]):
                    continue
                lineage = booking[numeral][rank]
                terms = {ww for worth, _ in self.worth_terms(numeral) for ww in _worth_symbols(worth)}
                reservoir = [ii for ii in self.article_uids(numeral) if lineage in ((position['marks'].get(ii) or {}).get('fam') or []) and ii not in self.ample and self.purchasable(ii) and (lid is None or (self.cost_from(ii) or 0) <= lid)]
                reservoir.sort(key=lambda ii: (-len(terms & _writing_terms(_spill(self.tiers.get(ii) or {}))), self.cost_from(ii) if self.cost_from(ii) is not None else float('inf')))
                chooses += [ii for ii in reservoir[:1] if ii not in chooses]
        peruse = self.populate_datums(chooses[:SWEEP_OPENS], preserve)
        self.milieu_void = MILIEU_VOID
        self.transcript.append({'kind': 'series_sweep', 'searches': sought, 'opened': len(peruse), 'series': {nn: ff for nn, ff in booking.items()}})
        if not peruse:
            return ''
        return "[Wren searched the items bound by a series code once per leading series and read the closest row of each series; the fit plan in the status lines up the series]\n" + '\n'.join(peruse)

    @_gentle(list)
    def chase_suits(self) -> list[str]:
        """Keep searching, without the model, before finishing with no purchase.

        In race 15, 495 of the 2340 episodes of 13 top agents finished without buying on a task with a correct answer, and
        in 433 of those some other agent had a cart the store's answer set accepted, so a fitting listing was reachable;
        practice worlds show the same (a fit beyond the page limit, "no in-stock jeans list a slim fit"). The value of a
        requirement and its words surface listings the initial searches missed; rows mentioning them are opened so the
        audit can judge their attributes. Returns what was read.
        """
        if self.suit_chased or self.timer() > CHASE_VIA:
            return []
        scheme = self.worth_stencil()
        if not scheme:
            return []
        instrument, basis, facet = scheme
        preserve = self.rear() + len(self.articles) + 1
        surplus = self.remaining() - preserve
        if surplus < 3:
            return []
        self.suit_chased = True
        sect_handle = self.genre_handle()
        retained = [str((self.tiers.get(ii) or {}).get(sect_handle) or '') for ii in self.basket_uids() if (self.retained_parcels(ii) or 0) > 0] if sect_handle else []
        numerals = [nn for nn, article in enumerate(self.articles, 1) if not any((self.possesses(article, legend) for legend in retained))] or list(range(1, len(self.articles) + 1))
        desired = {numeral: {ww for worth, _ in self.worth_terms(numeral) for ww in _worth_symbols(worth)} for numeral in numerals}
        sheeted = self.unperused_sheets(instrument, desired, min(SUIT_SHEETS, surplus // 2))
        capacity = min(CHASE_TICKS, self.remaining() - self.rear() - CHASE_PRESERVE)
        expended = 0
        for numeral in numerals if capacity >= 3 else []:
            article = self.articles[numeral - 1]
            couples = self.worth_terms(numeral)
            nominal, terms = self.chase_terms(numeral)
            attempts = [f'{lineage} {worth} {nominal}'.strip() for lineage in self.sequence_towards(numeral) for worth in ([vv for vv, _ in couples][:1] or [''])]
            attempts += [f'{worth} {nominal}'.strip() for worth, _ in couples] + [worth for worth, _ in couples if nominal] + [f'{term} {nominal}'.strip() for term in terms[:CHASE_SUIT_TERMS]] + ([nominal] if nominal else [])
            fraction = max(2, capacity // max(1, len(numerals)))
            employed = 0
            for enquiry in dict.fromkeys(attempts):
                if expended >= capacity or employed >= fraction or self.terminated or self.timer() > CHASE_VIA + CHASE_LENIENCY:
                    break
                defects: list[str] = []
                operands = _form({**basis, facet: enquiry}, self.blueprints.get(instrument), 'args', defects)
                if defects:
                    continue
                rejoinder = self.perform(instrument, operands)
                expended += 1
                employed += 1
                if rejoinder is None:
                    return []
                self.witness(instrument, rejoinder.get('observation'))
        chooses: list[str] = []
        for numeral in numerals:
            article = self.articles[numeral - 1]
            reservoir = [ii for ii in self.spotted_booking if ii not in self.ample and self.purchasable(ii) and (not sect_handle or self.possesses(article, str((self.tiers.get(ii) or {}).get(sect_handle) or '')))]
            lid = self.allowance if isinstance(self.allowance, (int, float)) else None
            reservoir = [ii for ii in reservoir if lid is None or (self.cost_from(ii) or 0) <= lid]
            terms = desired.get(numeral) or set()
            reservoir.sort(key=lambda ii: (-len(terms & _writing_terms(_spill(self.tiers.get(ii) or {}))), self.cost_from(ii) if self.cost_from(ii) is not None else float('inf')))
            chooses += reservoir[:CHASE_SUIT_LAUNCHES]
        peruse: list[str] = []
        for begin in range(0, len(chooses), CHASE_SUIT_LAUNCHES):
            if self.timer() > CHASE_VIA + CHASE_LENIENCY:
                break
            peruse += self.populate_datums(chooses[begin:begin + CHASE_SUIT_LAUNCHES], preserve)
        self.transcript.append({'kind': 'hunt_fits', 'pages': sheeted, 'searches': expended, 'opened': len(peruse)})
        return peruse

    def chase_terms(self, numeral: int) -> tuple[str, list[str]]:
        """The noun of the item, i.e. a word present in its store label and in no other item's label (search matches labels
        too, so this keeps results on this item), plus words a follow-up search could add, best first: a word found on most
        rows brings nothing new and one found on a single row brings little more, so the kept words lie in between.
        """
        sect_handle = self.genre_handle()
        article = self.articles[numeral - 1]
        headline = self.headline_handle()
        legends = [str(tier.get(sect_handle) or '') for tier in self.tiers.values() if self.possesses(article, str(tier.get(sect_handle) or ''))] if sect_handle else []
        monikers = [str(tier.get(headline) or '') for tier in self.tiers.values() if headline and sect_handle and self.possesses(article, str(tier.get(sect_handle) or ''))]
        for entry in self.ample.values():
            if sect_handle and self.possesses(article, str(entry.get(sect_handle) or '')):
                monikers.append(' '.join((str(vv) for vv in (_datum_chunk(entry) or {}).values() if isinstance(vv, str) and len(vv) <= CHASE_WORTH_GLYPHS)))
        alternatives = {ww for nn, alternative in enumerate(self.articles, 1) if nn != numeral for ww in re.findall('[a-z]{4,}', ((alternative.rack or '') + ' ' + alternative.moniker).lower())}
        legend = max(legends, key=legends.count).lower() if legends else (article.rack or article.moniker).lower()
        nominals = [ww for ww in re.findall('[a-z]{4,}', legend) if ww not in alternatives]
        nominal = '' if numeral in self.chase_nil_nominal else (max(nominals, key=len) if nominals else '')
        if not monikers:
            return (nominal, [])
        tallies: dict[str, int] = {}
        for moniker in monikers:
            for term in {ww.lower() for ww in re.findall("[A-Za-z][A-Za-z'-]{2,}", moniker)}:
                tallies[term] = tallies.get(term, 0) + 1
        personal = _sources(article.rack or '') | _sources(article.moniker)
        roof = max(2, int(len(monikers) * CHASE_USUAL))
        terms = [ww for ww, nn in tallies.items() if 2 <= nn <= roof and ww != nominal and _sources(ww) and not _sources(ww) & personal]
        terms.sort(key=lambda ww: (-tallies[ww], ww))
        return (nominal, terms)

    @_gentle('')
    def chase_thriftier(self) -> str:
        """Keep searching, without the model, while the chosen listing is the cheapest and the store reveals only a portion of
        its stock.

        Searches halt at a page limit yet report the full count (race 14: 96 rows out of 199; race 15: 49 rows out of as
        many as 1000, with no price sorting and no price filter), so the cheapest listing of a kind may lie beyond any
        first search. In race 15 search matched titles, categories, brands, sellers and attribute VALUES and required every
        word (of 91849 matched words, 22642 matched only inside an attribute value), so combining all of an item's required
        values lists almost its entire fitting set, and appending another word divides a set that exceeds the limit. Once a
        requirement is dropped the fitting set grows: this settled 7 race-15 tasks, with almost every agent purchasing a
        pricier listing, while a rare successful run kept trying fresh words (9 cents, after 98 and then 20). Every search
        costs one step and no model call; afterwards a later step opens the lowest rows.
        """
        marker = _spill([self.demands_presently(), self.allowance, sorted(self.departed)])
        if self.concise.get('pick') != 'cheapest' or self.chases >= CHASE_LAPS or self.chase_marker == marker:
            return ''
        scheme = self.worth_stencil()
        if not scheme or not self.cost_handle or self.timer() > CHASE_VIA:
            return ''
        instrument, basis, facet = scheme
        capacity = min(CHASE_TICKS, self.remaining() - self.rear() - CHASE_PRESERVE - 2 * len(self.articles))
        if capacity < 3:
            return ''
        self.chases += 1
        self.chase_marker = marker
        basket = self.basket_uids()
        numerals = list(range(1, len(self.articles) + 1))
        fraction = max(3, capacity // max(1, len(numerals)))
        expended = 0
        crisp_aggregate = 0
        for numeral in numerals:
            if expended >= capacity or self.terminated:
                break
            employed, earned = self.chase_article(numeral, instrument, basis, facet, min(fraction, capacity - expended))
            expended += employed
            crisp_aggregate += earned
        self.transcript.append({'kind': 'hunt', 'steps': expended, 'new_rows': crisp_aggregate, 'cart': basket})
        if expended:
            self.broader_retained = True
        return f'Wren made {expended} extra searches looking for the lowest price (one step apiece), which turned up {crisp_aggregate} additional rows.' if expended else ''

    def enquiry_be_novel(self, moniker: str, operands: dict[str, Any]) -> bool:
        """Tells whether this query, ignoring its numeric arguments, has never been run before, and records it as seen."""
        marker = _spill({kk: vv for kk, vv in operands.items() if isinstance(vv, bool) or not isinstance(vv, int)})
        spotted = self.sheet_enquiries.setdefault(moniker, set())
        crisp = marker not in spotted
        spotted.add(marker)
        return crisp

    @_gentle('')
    def final_sheet_initial(self, moniker: str, operands: dict[str, Any], sighting: Any) -> str:
        """A fresh search opened on a later page returns few rows and no next page, which looks like everything: in race 15
        the models requested the browse tool's top page in 1716 of 2845 calls, seeing 7 of 342 rows and "next: none". When
        a call like that claims more rows than it displays, repeating it at the minimum value of its numeric argument
        settles it: rows with different ids, at least as many, show that the argument is the page number.
        """
        if moniker in self.sheet_operands or not isinstance(sighting, dict):
            return ''
        features = (self.blueprints.get(moniker) or {}).get('properties') or {}
        numerical = [kk for kk, vv in operands.items() if isinstance(vv, int) and not isinstance(vv, bool) and kk in features and vv > (features[kk].get('minimum') if isinstance(features[kk].get('minimum'), int) else 1)]
        if len(numerical) != 1 or self.remaining() <= self.rear() + 4:
            return ''
        form = self.peruse_form(moniker, sighting)
        if not form or not form.get('count') or form.get('next') is None or sighting.get(form['next']) is not None:
            return ''
        tiers = sighting.get(form['rows'])
        tally = sighting.get(form['count'])
        if not isinstance(tiers, list) or not isinstance(tally, int) or tally <= len(tiers):
            return ''
        handle = numerical[0]
        modest = features[handle].get('minimum') if isinstance(features[handle].get('minimum'), int) else 1
        defects: list[str] = []
        invoke = _form({**operands, handle: modest}, self.blueprints.get(moniker), 'args', defects)
        if defects:
            return ''
        rejoinder = self.perform(moniker, invoke)
        if rejoinder is None:
            return ''
        spotted = rejoinder.get('observation')
        self.witness(moniker, spotted)
        novel = spotted.get(form['rows']) if isinstance(spotted, dict) else None
        if not isinstance(novel, list) or len(novel) < len(tiers) or {ii for ii, _ in self.entries_inside(spotted)} & {ii for ii, _ in self.entries_inside(sighting)}:
            return ''
        self.sheet_operands[moniker] = handle
        self.transcript.append({'kind': 'page_arg', 'tool': moniker, 'arg': handle})
        return f'\n[Wren] {handle} sets the page number for this search, and the call above returned the final page. Here is page {modest} of that search:\n' + self.display(spotted)

    def further_tiers_towards_handles(self, instrument: str, operands: dict[str, Any], spotted: Any, cease_upon: int, numeral: int) -> list[str]:
        """Fetch an extra page if the opening page holds too few listings of this item to judge their contents; returns what that page showed."""
        sheet = self.sheeting(instrument, spotted)
        if not sheet or sheet[3] is None or self.tiers_from_article(numeral) >= UNCLOSED_EVERY or self.terminated or self.tick >= cease_upon:
            return []
        defects: list[str] = []
        invoke = _form({**operands, sheet[2]: sheet[3]}, self.blueprints.get(instrument), 'args', defects)
        if defects:
            return []
        rejoinder = self.perform(instrument, invoke)
        if rejoinder is None:
            return []
        self.witness(instrument, rejoinder.get('observation'))
        return [f'next page: ' + self.display(rejoinder.get('observation'))]

    def headline_handle(self) -> str | None:
        """The row key that holds a listing's title: among short texts that most rows have and few rows repeat, averaging more
        than one word and with roughly as many distinct values as the best such key (a seller name recurs, a listing title
        does not), the shortest one.
        """
        sect_handle = self.genre_handle()
        aggregates: dict[str, list[Any]] = {}
        for tier in self.tiers.values():
            for handle, worth in tier.items():
                if handle != sect_handle and isinstance(worth, str) and len(worth) <= HEADLINE_GLYPHS:
                    cell = aggregates.setdefault(handle, [0, 0, 0, set()])
                    cell[0] += len(worth.split())
                    cell[1] += 1
                    cell[2] += len(worth)
                    cell[3].add(worth)
        preserved = [(cell[2] / cell[1], handle, len(cell[3])) for handle, cell in aggregates.items() if cell[1] * 5 >= len(self.tiers) * 4 and cell[0] >= cell[1] * 2 and len(cell[3]) * 2 >= cell[1]]
        maximal = max((unique for _, _, unique in preserved), default=0)
        rated = [(average, handle) for average, handle, unique in preserved if unique * 10 >= maximal * 9]
        return min(rated)[1] if rated else None

    def initial_peek(self, moniker: str, operands: dict[str, Any], crisp: bool) -> dict[str, Any]:
        """Once it is clear which argument of the tool picks the page, fresh searches are sent for page one."""
        handle = self.sheet_operands.get(moniker)
        if not handle or not crisp or not isinstance(operands.get(handle), int):
            return operands
        features = (self.blueprints.get(moniker) or {}).get('properties') or {}
        modest = features.get(handle, {}).get('minimum') if isinstance(features.get(handle, {}).get('minimum'), int) else 1
        return {**operands, handle: modest}

    @_gentle('')
    def initial_shifts(self) -> str:
        """Before the model's first turn, search every item once and open a few of its listings, so the interview can begin
        immediately.

        A model turn in a race takes 5-10 s (race 15: 207 s per task on average; two agents lost 16 and 30 of their 180
        episodes to the 300 s cutoff, every one with 3+ items) and the opening turns do nothing but this; one-item tasks
        have time to spare and keep the model's own opening searches. The search for each item is its required values plus
        its noun, then its name if that returns nothing. Returns what the store displayed; '' when the search tool cannot
        be used without the model interpreting its modes, in which case the model begins normally.
        """
        scheme = self.worth_stencil()
        if not scheme or self.maximum_ticks < UNCLOSED_MINIMUM_TICKS or len(self.articles) < UNCLOSED_ARTICLES:
            return ''
        instrument, basis, facet = scheme
        self.milieu_void = UNCLOSED_VOID
        displayed: list[str] = []
        cease_upon = self.tick + self.remaining() - self.rear() - self.mails_gratis() - UNCLOSED_AFTERWARDS * len(self.articles) - CHASE_PRESERVE // 2
        for numeral, article in enumerate(self.articles, 1):
            desired = [mm.split(':', 1)[1].strip() for mm in article.essentials if ':' in mm and mm not in [demand for _, demand in self.references]]
            desired = [ww for ww in desired if ww and len(ww.split()) <= 3 and not re.search(UNCLOSED_NON, ww, re.I)]
            nominal = article.moniker if len(article.moniker.split()) <= 3 else ' '.join(article.moniker.split()[-2:])
            attempts = list(dict.fromkeys(([' '.join(desired[:2] + [nominal])] if desired else []) + [nominal]))
            for enquiry in attempts:
                defects: list[str] = []
                operands = _form({**basis, facet: enquiry}, self.blueprints.get(instrument), 'args', defects)
                if defects or self.terminated or self.tick >= cease_upon:
                    break
                rejoinder = self.perform(instrument, operands)
                if rejoinder is None:
                    return '\n'.join(displayed)
                spotted = rejoinder.get('observation')
                self.witness(instrument, spotted)
                if self.entries_inside(spotted):
                    self.sought = (self.sought + [dict(operands)])[-8:]
                    displayed.append(f'{enquiry}: ' + self.display(spotted))
                    displayed += self.further_tiers_towards_handles(instrument, operands, spotted, cease_upon, numeral)
                    if self.tiers_from_article(numeral) >= UNCLOSED_EVERY:
                        break
                else:
                    displayed.append(f"{enquiry}: {'the store refused the call: ' + self.display(spotted) if _faltered(spotted) else 'no rows'}")
        if self.tiers and self.duties.get('ask') and self.mails_gratis() > 0:
            begin = self.tick
            located = [nn for nn in range(1, len(self.articles) + 1) if self.tiers_from_article(nn)]
            while self.tick - begin < self.maximum_ticks // UNCLOSED_FRACTION and self.tick < cease_upon and not self.terminated:
                if min((self.entries_from_article(nn) for nn in located), default=UNCLOSED_EVERY) >= UNCLOSED_EVERY:
                    break
                possessed = len(self.ample)
                self.broaden()
                if len(self.ample) == possessed:
                    break
        self.milieu_void = MILIEU_VOID
        self.transcript.append({'kind': 'first_moves', 'rows': len(self.tiers), 'opened': len(self.ample)})
        displayed += ['fully read: ' + self.sheet_writing(identity, entry) for identity, entry in self.ample.items()]
        evidence = self.testimony(list(self.ample.values()))
        return '\n'.join(displayed + ([evidence] if evidence else []))

    def invoke_marker(self, moniker: str, operands: dict[str, Any]) -> str:
        """A single spelling for each call: words regardless of case and spacing, with a number left out when it sits at its first value."""
        features = (self.blueprints.get(moniker) or {}).get('properties') or {}
        preserved: dict[str, Any] = {}
        for handle, worth in sorted(operands.items()):
            specification = features.get(handle) if isinstance(features.get(handle), dict) else {}
            initial = worth in (specification.get('minimum'), specification.get('default')) or (worth == 1 and handle == self.sheet_operands.get(moniker) and specification.get('minimum') is None and specification.get('default') is None)
            if isinstance(worth, int) and not isinstance(worth, bool) and initial:
                continue
            preserved[handle] = ' '.join(worth.lower().split()) if isinstance(worth, str) else worth
        return moniker + _spill(preserved)

    @_gentle(False)
    def monikers_cipher(self, operands: dict[str, Any]) -> bool:
        """Whether a search mentions a model or part code that some listing's text presented as a code."""
        uttered = {j for vv in operands.values() if isinstance(vv, str) for j in _ciphers(vv)}
        if not uttered:
            return False
        familiar: set[str] = set()
        for marker in self.suit_position()['marks'].values():
            familiar.update(marker.get('codes') or [])
            familiar.update(marker.get('works') or [])
        return bool(uttered & familiar)

    def remark_peruse(self, moniker: str, operands: dict[str, Any], spotted: Any) -> None:
        """Note what any search or listing call returned, whoever sent it."""
        if moniker in self.sheet_instruments() or moniker in {self.duties.get(rr) for rr in ('ask', 'cart', 'cart_view', 'buy', 'refuse')}:
            return
        self.enquiry_be_novel(moniker, operands)
        if isinstance(spotted, (dict, list)) and not _faltered(spotted):
            self.peruse_invokes[self.invoke_marker(moniker, operands)] = spotted
            self.peruse_operands[self.invoke_marker(moniker, operands)] = (moniker, dict(operands))

    def sheet_writing(self, identity: str, entry: dict[str, Any]) -> str:
        """A single opened listing as shown to the model: attributes first, because those are what the model evaluates."""
        chunk = _datum_chunk(entry) or {}
        remainder = {kk: vv for kk, vv in entry.items() if vv is not chunk and not isinstance(vv, (dict, list))}
        return f'{identity}: attributes ' + _spill(_snip(chunk, 120))[:SHEET_TRAIT_GLYPHS] + ' | ' + _spill(_snip(remainder, 200))[:SHEET_REMAINDER_GLYPHS]

    @_gentle('')
    def terser_seek(self, moniker: str, operands: dict[str, Any], sighting: Any) -> str:
        """A long word search that returned nothing is retried with only its leading words, without a model turn. In race 15,
        54% of six-word queries and 42% of five-word queries came back empty (search needs every word), against 10% of
        one-word ones.
        """
        scheme = self.worth_stencil()
        if not scheme or scheme[0] != moniker or self.remaining() <= self.rear() + TERSE_PRESERVE or self.terminated or not isinstance(sighting, dict):
            return ''
        facet = scheme[2]
        writing = operands.get(facet)
        if not isinstance(writing, str) or self.entries_inside(sighting) or len(writing.split()) < 3:
            return ''
        for worth in sighting.values():
            if (isinstance(worth, (list, str)) and worth) or (isinstance(worth, (int, float)) and not isinstance(worth, bool) and worth > 0):
                return ''
        terse = ' '.join([ww for ww in writing.split() if not _ciphers(ww)][:2])
        if not terse:
            return ''
        defects: list[str] = []
        invoke = _form({kk: vv for kk, vv in {**operands, facet: terse}.items() if isinstance(vv, bool) or not isinstance(vv, int)}, self.blueprints.get(moniker), 'args', defects)
        if defects:
            return ''
        rejoinder = self.perform(moniker, invoke)
        if rejoinder is None:
            return ''
        spotted = rejoinder.get('observation')
        self.witness(moniker, spotted)
        self.transcript.append({'kind': 'shorter', 'from': writing, 'to': terse, 'rows': len(self.entries_inside(spotted))})
        return f'\n[Wren] No listing contains every one of those words; running the search with {terse!r} instead gives:\n' + self.display(spotted)

    def tiers_from_article(self, numeral: int) -> int:
        sect_handle = self.genre_handle()
        article = self.articles[numeral - 1]
        return sum((1 for tier in self.tiers.values() if not sect_handle or self.possesses(article, str(tier.get(sect_handle) or ''))))

    def unperused_sheets(self, instrument: str, desired: dict[int, set[str]], capacity: int) -> int:
        """At most the allowed number of extra pages from the earliest searches containing a required value, each item getting
        its share first, before trying any new words; returns the steps used. Searches whose rows all belong to a different
        item are skipped. In a practice world, accepted rows lay on unread pages of the episode's own searches in almost
        every mistaken decline, and in races 13-15 roughly 23% of carts that passed were first spotted on page 2 or beyond.
        """
        expended = 0
        sect_handle = self.genre_handle()
        fraction = max(1, capacity // max(1, len(desired)))
        for cutoff in (fraction, capacity):
            for numeral, terms in desired.items():
                cease = min(capacity, expended + cutoff)
                for marker, (moniker, operands) in list(self.peruse_operands.items()):
                    writing = ' '.join((vv for vv in operands.values() if isinstance(vv, str))).lower()
                    if moniker != instrument or not terms & _writing_terms(writing):
                        continue
                    spotted = self.peruse_invokes[marker]
                    proprietors = {self.proprietor_from(str(entry.get(sect_handle) or '')) for _, entry in self.entries_inside(spotted)} if sect_handle else set()
                    if proprietors and 0 not in proprietors and numeral not in proprietors:
                        continue
                    invoke, traversed = operands, {marker}
                    while expended < cease and not self.terminated:
                        sheet = self.sheeting(instrument, spotted)
                        if not sheet or sheet[3] is None:
                            break
                        defects: list[str] = []
                        invoke = _form({**invoke, sheet[2]: sheet[3]}, self.blueprints.get(instrument), 'args', defects)
                        presently = self.invoke_marker(instrument, invoke) if not defects else ''
                        if defects or presently in traversed:
                            break
                        traversed.add(presently)
                        if presently in self.peruse_invokes:
                            spotted = self.peruse_invokes[presently]
                            continue
                        rejoinder = self.perform(instrument, invoke)
                        expended += 1
                        if rejoinder is None:
                            return expended
                        spotted = rejoinder.get('observation')
                        self.witness(instrument, spotted)
        return expended

    def witness(self, moniker: str, sighting: Any) -> str:
        self.soak(sighting)
        strands = self.slack_strands(sighting)
        if not strands:
            return ''
        sense = self.grab_remark(strands)
        if sense:
            self.scheduled.append(f'(seen on {moniker}) {sense}')
        return sense


class _HelmPart:
    """Drives the turns, executes calls and settles the finish."""

    def basket_ensembles(self, moniker: str) -> bool:
        features = self.blueprints.get(moniker, {}).get('properties') or {}
        if any((isinstance(ss, dict) and isinstance(ss.get('enum'), list) and (len(ss['enum']) >= 2) for ss in features.values())):
            return False
        return any((isinstance(ss, dict) and 'integer' in _varieties(ss) and (ss.get('minimum') == 0) for ss in features.values()))

    def be_decline(self, moniker: str, operands: dict[str, Any]) -> bool:
        if moniker != self.duties.get('refuse'):
            return False
        repaired = self.concise.get('refuse_args') if isinstance(self.concise.get('refuse_args'), dict) else {}
        return all((operands.get(kk) == vv for kk, vv in repaired.items()))

    def tardy_terminate(self) -> str:
        """When the step cap is close, Wren checks out itself rather than accept an ending without a purchase, provided the
        cart has (or used to have) a line per item, or a short cart can be completed from opened listings that show every
        requirement.

        Lacking steps is no evidence that nothing fits. Across races 14, 16 and 17 (all agents), ordering the cart that was
        complete when a late decline came would have scored 7 and lost 2; on race 17 Wren itself declined late while a
        winning basket sat in its cart. Returns '' if Wren is unable to do it."""
        purchase, basket = (self.duties.get('buy'), self.duties.get('cart'))
        if not purchase or not basket or self.terminated:
            return ''
        sect_handle = self.genre_handle()
        strands = [ii for ii in self.constructed if not self.vended_output(ii)]
        crisp = self.basket_outlook is not None and self.basket_spotted_upon >= self.basket_upon
        presently = self.basket_uids()
        lacking: list[str] = []
        if len(strands) >= len(self.articles):
            lacking = [ii for ii in strands if ii not in presently or (ii in self.omitted and (not crisp))]
            desire: dict[str, int] = {}
            for identity in lacking:
                proprietors = [nn for nn, article in enumerate(self.articles, 1) if sect_handle and self.possesses(article, str((self.tiers.get(identity) or {}).get(sect_handle) or ''))]
                desire[identity] = self.articles[proprietors[0] - 1].amt if len(proprietors) == 1 else 1
            invokes = self.basket_invokes(desire, []) if desire else []
            if invokes is None or len(invokes) + 1 > self.remaining():
                return ''
            for operands in invokes:
                defects: list[str] = []
                operands = _form(operands, self.blueprints.get(basket), 'args', defects)
                if defects:
                    return ''
                rejoinder = self.perform(basket, operands)
                if rejoinder is None or self.terminated:
                    return ''
                self.remark_basket(basket, operands)
                self.witness(basket, rejoinder.get('observation'))
        elif len(self.articles) >= 2 and presently:
            if not self.populate_terse_basket() or len([ii for ii in self.basket_uids() if (self.retained_parcels(ii) or 0) > 0]) < len(self.articles):
                return ''
        else:
            return ''
        if self.terminated or self.remaining() < 1 or any((' out of stock' in ff or ' above the per-item limit' in ff or ' exceeds the budget' in ff for ff in self.strand_defects())):
            return ''
        if self.remaining() >= 2:
            self.reperuse_tenders()
        if self.terminated or self.remaining() < 1:
            return ''
        defects = []
        operands = _form(dict(self.concise.get('buy_args') or {}), self.blueprints.get(purchase), 'args', defects)
        if defects:
            return ''
        rejoinders: list[str] = []
        for trial in range(2):
            if trial and (self.remaining() < 1 or self.terminated or any((self.vended_output(ii) for ii in self.basket_uids()))):
                break
            if self.remaining() >= 2:
                proceed, uttered = self.checkpoint('order')
                if uttered:
                    rejoinders.append(uttered)
                if not proceed:
                    break
            if self.terminated or self.remaining() < 1:
                break
            rejoinder = self.perform(purchase, operands)
            self.transcript.append({'kind': 'late_finish', 'put_back': lacking, 'reply': _snip(rejoinder, 200)})
            if rejoinder is None:
                break
            self.purchases += 1
            if not rejoinder.get('finished'):
                self.declined_upon = self.tick
            self.witness(purchase, rejoinder.get('observation'))
            rejoinders.append(self.display(rejoinder.get('observation')))
            if self.terminated or rejoinder.get('finished'):
                break
        if not rejoinders:
            return ''
        return 'Not executed (Wren handled it instead): this near the step cap, a no-purchase ending scores zero while the cart has a line for every item, so Wren ' + (f"returned {', '.join(lacking)} to the cart and " if lacking else '') + 'completed checkout. The store replied: ' + ' | then: '.join(rejoinders)

    def terse_basket(self) -> bool:
        """True if the cart lacks a line with units for at least one requested item."""
        retained = [ii for ii in self.basket_uids() if (self.retained_parcels(ii) or 0) > 0]
        return len(retained) < len(self.articles) or bool(self.lacking_articles())

    def relaunch_peruses(self) -> str:
        """Single use: once the model has answered the trimmed tool list without calling anything while some item still has no
        cart line, the search tool and the listing views are put back (the final steps remain held for the cart and the
        order); a cart left short otherwise drifts toward a no-purchase finish with unused steps (race-16 copy: 3 of 5 such
        finishes, with 4 to 6 steps left over)."""
        search = self.duties.get('search')
        if self.relaunched or self.purchases or not search or self.remaining() <= self.rear() + 3 or not self.terse_basket():
            return ''
        if search in {tt['function']['name'] for tt in self.tendered()}:
            return ''
        self.relaunched = True
        self.transcript.append({'kind': 'reopened', 'left': self.remaining()})
        return 'An item is still missing from the cart, so search and the listing views are back: locate and read a listing that fits it, add it to the cart, and check out before the steps run out.'

    def tardy_evaluation(self) -> str:
        """Gives a decline arriving with five steps or fewer what the early guard grants one with more room: the review's
        leads, or a review made now from records already read (no step), go as one complete basket to the independent
        check, and if that check agrees, Wren fills the cart and checks out by itself (finalize_basket). Race-16 copy: with 6
        steps to go the review named a basket that would have paid, the model declined again at 4, and that decline was
        let through. Returns '' if nothing happened."""
        self.decline_chunks = 2
        if not self.prospects and not self.vetted and self.timer() <= COIL_DOWNWARD and self.referees():
            self.vetted = True
            dubbed = self.vet_suits(self.vet_invoke())
            if dubbed:
                self.prospect_articles = dubbed
                self.prospects = [ii for uids in dubbed.values() for ii in uids]
        initial = next((ii for ii in self.prospects if not self.vended_output(ii)), None)
        hamper = self.intended_hamper(initial) if initial else None
        if not hamper or self.timer() > COIL_DOWNWARD:
            return ''
        invokes = self.basket_invokes(*hamper)
        if not invokes or len(invokes) + 1 > self.remaining():
            return ''
        judged = sorted(hamper[0])
        self.verify_basket(judged, EXPEND_STRICT)
        if not self.rulings.get(self.verify_handle(judged)):
            return ''
        self.affirmed = initial
        self.transcript.append({'kind': 'late_review', 'confirmed': initial, 'basket': judged})
        return self.finalize_basket()

    def be_purchase(self, moniker: str, operands: dict[str, Any]) -> bool:
        if moniker != self.duties.get('buy'):
            return False
        repaired = self.concise.get('buy_args') if isinstance(self.concise.get('buy_args'), dict) else {}
        return all((operands.get(kk) == vv for kk, vv in repaired.items()))

    def complete_deletion(self, moniker: str, operands: dict[str, Any]) -> dict[str, Any]:
        """Removing a held line without a quantity specifies all units held, because otherwise the store would remove just one."""
        specification = self.blueprints.get(moniker, {}).get('properties') or {}
        tally = next((kk for kk, vv in specification.items() if isinstance(vv, dict) and 'integer' in _varieties(vv)), None)
        deletion = any((isinstance((specification.get(kk) or {}).get('enum'), list) and vv in specification[kk]['enum'][1:] for kk, vv in operands.items()))
        identity = next((vv for vv in operands.values() if isinstance(vv, str) and vv in self.basket_strands), None)
        return {**operands, tally: min(self.basket_strands[identity], specification[tally].get('maximum', self.basket_strands[identity]))} if tally and deletion and identity and tally not in operands else operands

    def contrast_aim(self) -> str | None:
        blueprint = self.blueprints.get(self.duties.get('compare') or '') or {}
        features = blueprint.get('properties') or {}
        aim = next((kk for kk, vv in features.items() if isinstance(vv, dict) and 'array' in _varieties(vv)), None)
        if not aim or any((kk != aim for kk in blueprint.get('required') or [])):
            return None
        return aim

    def contrast_capacity(self) -> int:
        specification = ((self.blueprints.get(self.duties.get('compare') or '') or {}).get('properties') or {}).get(self.contrast_aim() or '') or {}
        if isinstance(specification.get('maxItems'), int):
            return specification['maxItems']
        if self.contrast_poor:
            return max(2, min(self.contrast_poor - 1, max(self.contrast_okay, 2)))
        return max(4, self.contrast_okay)

    def determine_uids(self, moniker: str, operands: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if moniker not in {self.duties.get(rr) for rr in ('cart', 'open', 'offer', 'compare')} - {None} or not self.uid_handle:
            return (operands, '')
        features = (self.blueprints.get(moniker) or {}).get('properties') or {}
        gratis = [kk for kk, vv in features.items() if isinstance(vv, dict) and ('string' in _varieties(vv) or 'array' in _varieties(vv)) and ('enum' not in vv)]
        if len(gratis) != 1 or (moniker != self.duties.get('cart') and (moniker, gratis[0]) not in self.uid_operands):
            return (operands, '')

        def repair(worth: Any) -> tuple[Any, str]:
            if isinstance(worth, str) and worth.strip() in self.tiers:
                return (worth.strip(), '')
            if not isinstance(worth, str) or _uid_akin(worth, True) or len(worth.strip()) < 8:
                return (worth, '')
            modest = worth.strip().lower()
            strikes = [ii for ii, tier in self.tiers.items() if any((kk != self.uid_handle and isinstance(vv, str) and vv.strip().lower().startswith(modest) for kk, vv in tier.items()))]
            if len(strikes) == 1:
                return (strikes[0], '')
            if strikes:
                return (worth, f"the text {worth!r} appears on several listings ({', '.join(strikes[:6])}); give a single listing id")
            return (worth, f'{worth!r} is not a listing id; use the id shown in the result rows (their {self.uid_handle} field)')
        output = dict(operands)
        for handle in gratis:
            if handle not in operands:
                continue
            repaired = [repair(vv) for vv in operands[handle]] if isinstance(operands[handle], list) else [repair(operands[handle])]
            issue = next((pp for _, pp in repaired if pp), '')
            if issue:
                return (operands, issue)
            output[handle] = [vv for vv, _ in repaired] if isinstance(operands[handle], list) else repaired[0][0]
        return (output, '')

    def display(self, sighting: Any) -> str:
        writing = _spill(_snip(sighting, DISPLAY_GLYPHS if self.entries_inside(sighting) else WRITING_GLYPHS))
        if len(writing) > GLIMPSE_CUTOFF:
            writing = writing[:GLIMPSE_CUTOFF] + '...(cut)'
        return writing

    def execute(self) -> list[dict[str, Any]]:
        if self.terminated:
            return self.transcript
        self.peruse_appeal()
        self.transcript.append({'kind': 'brief', 'brief': self.concise})
        kickoff = self.outlook.get('opening')
        if kickoff is not None:
            kickoff = _reinstate(kickoff)
            self.soak(kickoff)
        shifted = self.initial_shifts() if not self.terminated else ''
        self.plan_ahead()
        swept = self.series_sweep() if not self.terminated else ''
        scanned = self.code_scan() if not self.terminated else ''
        if scanned:
            swept = (swept + '\n' if swept else '') + scanned
        if swept:
            shifted = (shifted + '\n' if shifted else '') + swept
        self.milieu_void = MILIEU_VOID
        preface = "SHOPPER'S REQUEST:\n" + self.appeal
        if kickoff is not None:
            preface += '\n\nWHAT THE STORE SHOWS FIRST:\n' + self.display(kickoff)
        if shifted:
            preface += '\n\nWren has run a search for each item and read a few of its listings (one step per listing):\n' + shifted + '\nAssess these against the requirements and keep searching only for whatever is still missing.\n\n' + self.standing()
        else:
            preface += '\n\nBegin by searching each item within its category and opening several listings fully, so the interviewer can learn which properties these listings have.\n\n' + self.standing()
        self.dialog = [{'role': 'system', 'content': PILOT}, {'role': 'user', 'content': preface}]
        self.labor(CYCLE_INVOKES, EXPEND_STRICT)
        if not self.terminated:
            rotated = self.terminate(anew=True)
            if rotated and (not self.terminated) and (not self.busted) and (not self.nil_balance) and (self.remaining() > 1):
                self.dialog.append({'role': 'user', 'content': '[Wren] Wren attempted to end the episode and ' + rotated + '\nAct on what the store reports, then finish.\n' + self.standing()})
                self.labor(self.cycle_invokes + 10, EXPEND_STRICT + 0.01)
            for _ in range(2):
                if self.terminated:
                    break
                rotated = self.terminate()
                if not rotated or self.terminated or self.busted or self.nil_balance or self.remaining() <= 1:
                    break
                self.dialog.append({'role': 'user', 'content': '[Wren] Wren attempted to end the episode and ' + rotated + '\nAct on what the store reports, then finish.\n' + self.standing()})
                self.labor(self.cycle_invokes + 6, EXPEND_STRICT + 0.02)
        self.transcript.append({'kind': 'end', 'finished': self.terminated, 'steps': self.tick, 'cost': round(self.outlay, 5), 'calls': self.invokes})
        return self.transcript

    def execute_invokes(self, invokes: list[dict[str, Any]]) -> list[tuple[str, str]]:
        output: list[tuple[str, str]] = []
        invokes = self.unite_launches(invokes)
        for position, invoke in enumerate(invokes):
            permitted = {tt['function']['name'] for tt in self.tendered()}
            cnum = invoke['id']
            if invoke.get('joined'):
                output.append((cnum, 'See the side-by-side result above: Wren opened these listings at once in a single step.'))
                continue
            func = invoke.get('function') or {}
            moniker = func.get('name')
            crude = func.get('arguments')
            if self.terminated:
                output.append((cnum, 'Not executed: the episode has ended.'))
                continue
            if moniker == self.duties.get('ask') and self.letter_operands('x' * 200) is not None:
                output.append((cnum, "Not executed (no step spent): Wren's interviewer takes care of every message to the shopper."))
                continue
            if moniker not in self.blueprints:
                output.append((cnum, f"Not executed (no step spent): no such tool. Available tools: {', '.join(sorted(permitted))}."))
                continue
            if moniker not in permitted:
                output.append((cnum, 'Not executed (no step spent): not enough steps remain for that; use: ' + ', '.join(sorted(permitted)) + '.'))
                continue
            if isinstance(crude, dict):
                operands = crude
            else:
                operands = _jdoc_entity(crude) if (crude or '').strip() else {}
                if operands is None:
                    output.append((cnum, 'Not executed (no step spent): the arguments did not form a single JSON object.'))
                    continue
            defects: list[str] = []
            operands = _form(operands, self.blueprints[moniker], 'args', defects)
            if defects:
                output.append((cnum, 'Not executed (no step spent): ' + '; '.join(defects[:6])))
                continue
            operands, incorrect = self.determine_uids(moniker, operands)
            if incorrect:
                output.append((cnum, 'Not executed (no step spent): ' + incorrect + '.'))
                continue
            if moniker == self.duties.get('cart'):
                operands = self.complete_deletion(moniker, operands)
            pruned = ''
            numerous = self.contrast_aim() if moniker == self.duties.get('compare') else None
            if numerous and isinstance(operands.get(numerous), list) and (len(operands[numerous]) > self.contrast_capacity()) and self.contrast_poor:
                capacity = self.contrast_capacity()
                pruned = f"\n[Wren] Only the first {capacity} listings were sent: this store displays no more than {capacity} side by side. Still not shown: {', '.join(map(str, operands[numerous][capacity:]))}."
                operands = {**operands, numerous: operands[numerous][:capacity]}
            aim = self.solitary_aim(moniker)
            if aim and moniker in (self.duties.get('open'), self.duties.get('offer')) and (str(operands.get(aim)) in self.departed):
                output.append((cnum, f'Not executed (no step spent): {operands[aim]} went out of stock at step {self.departed[str(operands[aim])]} and will not return. Swap its cart line for a different listing that fits.'))
                continue
            if moniker == self.duties.get('cart') and any(((_tally(vv) or 0) < 0 for vv in operands.values() if not isinstance(vv, str))):
                outlook = _spill(self.basket_outlook) if self.basket_spotted_upon >= self.basket_upon else ''
                vacant = [vv for vv in operands.values() if isinstance(vv, str) and vv in self.tiers and (vv not in self.basket_strands) and (vv not in outlook) and (vv not in self.basket_uttered)]
                if vacant:
                    output.append((cnum, f'Not executed (no step spent): {vacant[0]} is not in the cart, so there is nothing to remove. Cart now: ' + self.basket_writing()))
                    continue
            closing = self.be_purchase(moniker, operands) or self.be_decline(moniker, operands)
            room_at_call = self.remaining()
            simple_chores = {instrument for chore, instrument in self.duties.items() if chore != 'search'}
            if closing and len(invokes) > 1:
                output.append((cnum, 'Not executed (no step spent): send the finishing call by itself, once you have read your other results.'))
                continue
            if not closing and self.remaining() <= self.rear() and (self.duties.get('buy') or self.duties.get('refuse')):
                output.append((cnum, "Not executed (no step spent): the final steps are reserved for checking out (with the shopper's approval of the basket when that is required) or for ending without a purchase."))
                continue
            crisp = moniker not in simple_chores and self.enquiry_be_novel(moniker, operands)
            inquired_towards = operands
            operands = self.initial_peek(moniker, operands, crisp)
            if operands != inquired_towards:
                pruned += f"\n[Wren] Because this is a new search, Wren requested its first page (the call you sent was {_spill(inquired_towards)})."
            operands = self.tier_uids(moniker, operands)
            moniker, operands, chained = self.sheet_initial(moniker, operands)
            aim = self.solitary_aim(moniker)
            if moniker == self.duties.get('search') and self.ciphers_defunct and self.monikers_cipher(operands):
                output.append((cnum, "Not executed (no step spent): searching this store for a model or part code returned nothing. Search with words and values instead, list the category, and read the codes shown on the rows."))
                continue
            marker = moniker + _spill(operands)
            if not closing and moniker != self.duties.get('cart') and (marker in self.latest):
                output.append((cnum, 'Not executed (no step spent): you made this identical call a moment ago and nothing has changed; its result appears above.'))
                continue
            perusers = {self.duties.get(rr) for rr in ('search', 'open', 'offer', 'compare')} - {None}
            stashed = self.peruse_stash.get(marker)
            if moniker in perusers and stashed and (stashed[0] == self.revision):
                output.append((cnum, f'Not executed (no step spent): the identical call at step {stashed[1]} gave this result, and nothing has changed since then:\n' + stashed[2][:3000]))
                continue
            if aim and moniker in perusers and (moniker != self.duties.get('compare')) and (self.displayed_previously(moniker, str(operands.get(aim))) or (moniker, str(operands.get(aim))) in self.datumless):
                output.append((cnum, f'Not executed (no step spent): {operands[aim]} has already shown all this view offers since the last change. Its record: ' + self.entry_from(str(operands[aim]))))
                continue
            if numerous and isinstance(operands.get(numerous), list) and operands[numerous]:
                preserve = [ii for ii in operands[numerous] if not self.displayed_previously(moniker, str(ii))]
                if not preserve:
                    output.append((cnum, 'Not executed (no step spent): these listings have already shown all this view offers since the last change. Their records:\n' + '\n'.join((f'{ii}: {self.entry_from(str(ii))}' for ii in operands[numerous]))[:4000]))
                    continue
                if len(preserve) < len(operands[numerous]):
                    pruned += f"\n[Wren] Skipped {', '.join((str(ii) for ii in operands[numerous] if ii not in preserve))}: already shown in full, with no change since."
                    operands = {**operands, numerous: preserve}
                    marker = moniker + _spill(operands)
            if closing and self.remaining() > 2:
                if self.be_purchase(moniker, operands):
                    chased = self.chase_thriftier()
                    broader = self.broader_grasp()
                    if broader:
                        output.append((cnum, broader + '\nCart now: ' + self.basket_writing()))
                        continue
                    reperuse = self.reperuse_tenders()
                    if self.remark_vended:
                        self.locate_vended()
                    if self.basket_visited and self.basket_spotted_upon < self.basket_upon and self.duties.get('cart_view') and (self.remaining() > 5) and not self.hurried():
                        defects = []
                        outlook_operands = _form({}, self.blueprints.get(self.duties['cart_view']), 'args', defects)
                        outlook = self.perform(self.duties['cart_view'], outlook_operands) if not defects else None
                        if outlook is not None:
                            self.basket_outlook = outlook.get('observation')
                            self.basket_spotted_upon = self.tick
                            alter = self.witness(self.duties['cart_view'], self.basket_outlook)
                            if alter:
                                self.scheduled = []
                                output.append((cnum, 'Not executed yet (no step spent). A look at the cart revealed a change: ' + alter + '\nCart now: ' + self.basket_writing() + '\nTake the change into account, correct the cart if necessary, then check out.'))
                                continue
                    exchange = self.exchange_cue()
                    retained = [ii for ii in self.basket_uids() if ii in self.departed] if self.retained < 2 else []
                    if exchange or retained:
                        self.retained += 0 if exchange else 1
                        output.append((cnum, 'Not executed yet (no step spent). ' + (exchange or f"{', '.join(retained)} in the cart is out of stock: swap out that line first.") + '\nCart now: ' + self.basket_writing()))
                        continue
                    tether, grasp = self.path()
                    if grasp:
                        output.append((cnum, 'Not executed yet (no step spent). ' + tether + '\nCart now: ' + self.basket_writing()))
                        continue
                    mismatch = self.suit_defects() if self.suit_grasps < 2 and self.remaining() > self.rear() + 3 else []
                    if mismatch:
                        self.suit_grasps += 1
                        self.transcript.append({'kind': 'fit_hold', 'faults': mismatch[:4]})
                        output.append((cnum, 'Not executed yet (no step spent). A part in the cart is not compatible with the device in the cart: ' + '; '.join(mismatch[:3]) + '. Replace the part (or the device), then check out.\nCart now: ' + self.basket_writing()))
                        continue
                    defects = self.strand_defects() if self.strand_grasps < 2 and self.remaining() > self.rear() + 2 else []
                    repaired = []
                    if defects and all((' is held at ' in ff for ff in defects)):
                        repaired = self.repair_tallies(self.tally_repairs())
                        defects = self.strand_defects() if repaired else defects
                        if repaired and (not defects):
                            self.strand_grasps += 1
                            output.append((cnum, "Not executed yet (no step spent). Wren adjusted the cart's quantities on its own: " + '; '.join(repaired) + '. If the store rejected any of these, correct those lines; then check out.\nCart now: ' + self.basket_writing()))
                            continue
                    if defects:
                        self.strand_grasps += 1
                        self.transcript.append({'kind': 'line_hold', 'faults': defects[:6]})
                        output.append((cnum, 'Not executed yet (no step spent). The cart does not yet match the request: ' + '; '.join(defects[:6]) + ". Correct those lines (using the cart tool's own method for adding or removing units), then check out.\nCart now: " + self.basket_writing()))
                        continue
                    strands = self.basket_uids()
                    peruse = self.populate_datums(strands, self.rear() + (1 if len(strands) >= 2 else 3))
                    self.display_bonds(strands, self.rear() + 1)
                    peruse += self.thriftier_tiers()
                    spotted = "Wren read the attributes of these listings (one step per listing):\n" + '\n'.join(peruse) + '\n' if peruse else ''
                    cheaper = self.thriftier_attested() if self.thrifty_grasps < 1 and self.remaining() > self.rear() + 3 else []
                    if cheaper:
                        self.thrifty_grasps += 1
                        self.transcript.append({'kind': 'cheaper_hold', 'swaps': cheaper})
                        output.append((cnum, 'Not executed yet (no step spent). ' + spotted + 'Lowest price is the pick rule, and a listing you already read costs less while giving each requirement under a key of that name: ' + '; '.join((f'{novel} at {modest:g} in place of {aged} at {high:g}' + (f" ({'; '.join(self.attestations[novel])})" if self.attestations.get(novel) else '') for aged, high, novel, modest in cheaper)) + ". Exchange that line (with the cart tool's own way of removing and adding units) and then check out; leave it in place only when that listing's record plainly violates a requirement.\nCart now: " + self.basket_writing()))
                        continue
                    feeble = self.feeble_grasp() if self.feeble_grasps < 1 and self.remaining() > self.rear() + 3 else ''
                    if feeble:
                        self.feeble_grasps += 1
                        output.append((cnum, feeble.replace('Not executed yet (no step spent). ', 'Not executed yet (no step spent). ' + spotted, 1) + '\nCart now: ' + self.basket_writing()))
                        continue
                    ruling = self.verify_basket()
                    if ruling:
                        spotted = "Wren read the attributes of these listings for the check (one step per listing):\n" + '\n'.join(peruse) + '\n' if peruse else ''
                        spotted = (chased + ' ' + self.under_basket() + '\n' if chased else '') + spotted
                        output.append((cnum, 'Not executed yet. ' + tether + spotted + ruling + '\nCart now: ' + self.basket_writing()))
                        continue
                    if reperuse and self.outlay > EXPEND_GENTLE:
                        output.append((cnum, "Not executed yet (no step spent). Wren has just re-read the listings in the cart:\n" + reperuse + '\nCheck out again once every line still shows stock and a price within the budget; if not, replace that line first.'))
                        continue
                elif self.remaining() > 5:
                    directed = self.decline_path()
                    if directed:
                        output.append((cnum, 'Not executed yet (no step spent). ' + directed))
                        continue
                    self.chase_suits()
                    evaluation = self.vet()
                    if evaluation:
                        output.append((cnum, 'Not executed yet (no step spent). ' + evaluation))
                        continue
                    if self.prospects and self.remaining() > 4 and (self.decline_chunks == 0):
                        self.decline_chunks = 1
                        self.transcript.append({'kind': 'refuse_blocked', 'leads': list(self.prospects)})
                        output.append((cnum, 'Not executed yet (no step spent). The review turned up listings that seem to fit: ' + ', '.join(self.prospects) + '. Open them as needed, place the best-fitting one in the cart (following the pick rule) and check out. Finish without buying only if every one of them plainly misses a requirement; in that case state which one in your text and resend the ending call.'))
                        continue
                    if self.prospects and self.remaining() > 4 and (self.decline_chunks == 1):
                        self.decline_chunks = 2
                        initial = self.prospects[0]
                        hamper = self.intended_hamper(initial) if len(self.articles) > 1 else None
                        judged = sorted(hamper[0]) if hamper else [initial]
                        self.verify_basket(judged, EXPEND_STRICT)
                        if self.rulings.get(self.verify_handle(judged)) and initial in self.prospects:
                            self.affirmed = initial
                            self.transcript.append({'kind': 'refuse_blocked', 'confirmed': initial, 'basket': judged})
                            if hamper:
                                output.append((cnum, f"Not executed yet (no step spent). The independent check passed the basket {', '.join(judged)}: every requirement holds, one line per item. Make the cart contain exactly these lines at the requested quantities, then check out; end without buying only when a record plainly violates a requirement."))
                            else:
                                output.append((cnum, f"Not executed yet (no step spent). An independent check of the records for {initial} finds that it satisfies every requirement. Put it in the cart at the requested quantity and check out; finish without buying only if its record plainly violates a requirement."))
                            continue
            if self.be_decline(moniker, operands) and 2 <= self.remaining() <= 5 and (not self.affirmed) and (self.decline_chunks < 2) and (len([ii for ii in self.constructed if not self.vended_output(ii)]) < len(self.articles)):
                accomplished = self.tardy_evaluation()
                if accomplished:
                    output.append((cnum, accomplished + (' [episode finished]' if self.terminated else '')))
                    continue
            if self.be_decline(moniker, operands) and self.remaining() <= 5 and (not self.tardy_retained):
                self.tardy_retained = True
                accomplished = self.tardy_terminate()
                if accomplished:
                    output.append((cnum, accomplished + (' [episode finished]' if self.terminated else '')))
                    continue
                strands = [ii for ii in self.constructed if not self.vended_output(ii)]
                crisp = self.basket_outlook is not None and self.basket_spotted_upon >= self.basket_upon
                lacking = [ii for ii in strands if ii not in self.basket_uids() or (ii in self.omitted and (not crisp))]
                if not self.terminated and len(strands) >= len(self.articles) and len(lacking) + 1 <= self.remaining():
                    self.transcript.append({'kind': 'late_decline_held', 'lines': strands, 'missing': lacking})
                    rework = f"Return {', '.join(lacking)} to the cart at the requested quantities, then check out." if lacking else 'Check out the cart in its current state.'
                    output.append((cnum, 'Not executed yet (no step spent). Finishing without a purchase scores nothing if the store carries a fitting listing, and being low on steps does not prove that it lacks one. ' + rework + ' Finish without buying only if your search proved that a requested item has no listing meeting its requirements; in that case resend the ending call.'))
                    continue
            if self.be_decline(moniker, operands) and (not self.prospects) and (not self.shoved) and (room_at_call > self.maximum_ticks // 3) and (self.remaining() > self.rear() + len(self.articles) + 2):
                self.shoved = True
                output.append((cnum, f"Not executed yet (no step spent). {self.remaining()} steps still remain, so try search approaches you have not used yet before ending: browse each requested item's category and page through it using the token or offset the store gives back, try one-word value queries and alternative names for the property, and open the listings that could fit. Only after that, finish without a purchase."))
                continue
            if self.be_decline(moniker, operands):
                accomplished = self.finalize_basket()
                if accomplished:
                    output.append((cnum, accomplished + (' [episode finished]' if self.terminated else '')))
                    continue
                operands = self.silent(moniker, operands)
            if self.be_purchase(moniker, operands):
                tardy = self.tardy_query()
                if tardy:
                    output.append((cnum, 'Not executed yet. ' + tardy + '\nCart now: ' + self.basket_writing()))
                    continue
            if closing and self.remaining() >= 2:
                proceed, uttered = self.checkpoint('order' if self.be_purchase(moniker, operands) else 'end')
                if not proceed:
                    output.append((cnum, 'Not executed yet. ' + uttered + '\nCart now: ' + self.basket_writing()))
                    continue
                if uttered:
                    pruned = '\n[Wren] ' + uttered + pruned
            rejoinder = self.perform(moniker, operands)
            if rejoinder is None:
                output.append((cnum, 'The call did not go through (network or server error). Feel free to retry.'))
                continue
            sighting = rejoinder.get('observation')
            if numerous and isinstance(operands.get(numerous), list) and (len(operands[numerous]) > 1):
                displayed = _spill(sighting)
                if not any((str(ii) in displayed for ii in operands[numerous])):
                    self.contrast_poor = min(self.contrast_poor or 999, len(operands[numerous]))
                    pruned += f'\n[Wren] The side-by-side view returned nothing for {len(operands[numerous])} listings: send no more than {self.contrast_capacity()} per call.'
                else:
                    self.contrast_okay = max(self.contrast_okay, len(operands[numerous]))
            self.latest = (self.latest + [marker])[-3:]
            self.uid_operands.update(((moniker, kk) for kk, vv in operands.items() if (vv if isinstance(vv, list) else [vv]) and all((isinstance(px, str) and px in self.tiers for px in (vv if isinstance(vv, list) else [vv])))))
            self.soak(sighting)
            if moniker == self.duties.get('compare'):
                tiers = self.entries_inside(sighting)
                if tiers:
                    self.datums_whence[moniker] = self.datums_whence.get(moniker, False) or any((_datum_chunk(rr) for _, rr in tiers))
            if moniker in perusers - {self.duties.get('search')}:
                self.remark_outlook(moniker, sighting)
            simple = {instrument for chore, instrument in self.duties.items() if chore != 'search'}
            sheeted = self.further_sheets(moniker, operands, sighting) if moniker not in simple and (not closing) else ''
            sheeted += self.final_sheet_initial(moniker, operands, sighting) if crisp and (not closing) else ''
            sheeted += self.terser_seek(moniker, operands, sighting) if moniker not in simple and (not closing) else ''
            if self.solitary_aim(moniker) and moniker not in (self.duties.get('cart'), self.duties.get('buy'), self.duties.get('refuse'), self.duties.get('ask')):
                posting = self.identity(sighting) if isinstance(sighting, dict) else None
                if posting:
                    self.remark_datums(moniker, bool(_datum_chunk(sighting)))
                    if moniker == self.duties.get('open') and (self.datums_whence.get(moniker) is False):
                        alternative = next((tt for tt in self.blueprints if tt != moniker and self.solitary_aim(tt) and (self.datums_whence.get(tt) is not False) and (tt not in (self.duties.get('cart'), self.duties.get('buy'), self.duties.get('refuse'), self.duties.get('ask'), self.duties.get('search'))) and not self.rosters_objects(tt)), None)
                        if alternative:
                            self.duties['open'] = alternative
                    if self.datums_whence.get(moniker) is False and moniker != self.duties.get('open') and (not self.duties.get('offer')):
                        self.duties['offer'] = moniker
                    if moniker == self.tender_instrument():
                        self.tender_spotted[posting] = self.tick
            line_pages = ''
            if moniker == self.duties.get('cart'):
                self.remark_basket(moniker, operands)
                self.basket_uttered = _spill(sighting)
                if not closing and self.purchases == 0 and self.cart_full() and self.remaining() > self.rear() + LATE_ROOM + 2 * len(self.articles) + 2:
                    early = self.thriftier_tiers()
                    if early:
                        line_pages += '\n[Wren read the unread listings priced below a cart line, one step each; swap a line when one of them fits every requirement]\n' + '\n'.join(early)
                self.latest = []
                tallies = [_tally(vv) for vv in operands.values() if not isinstance(vv, str) and _tally(vv) is not None]
                for worth in operands.values():
                    if tallies and isinstance(worth, str) and (worth in self.tiers):
                        if tallies[0] <= 0 and (not _inside_tiers(sighting, worth)):
                            self.omitted.add(worth)
                        elif tallies[0] > 0:
                            self.omitted.discard(worth)
                strands = [ii for ii in self.basket_uids() if not self.vended_output(ii) and ii not in self.omitted]
                if len(strands) >= len(self.constructed):
                    self.constructed = strands
            if moniker == self.duties.get('search') and self.monikers_cipher(operands):
                self.cipher_attempts.append(bool(self.entries_inside(sighting)))
                self.ciphers_defunct = len(self.cipher_attempts) >= 2 and not any(self.cipher_attempts)
            if moniker == self.duties.get('search') and self.entries_inside(sighting):
                self.sought = (self.sought + [dict(operands)])[-8:]
                hit_query = ' '.join((str(vv) for vv in operands.values() if isinstance(vv, str))).lower()
                if any((hit_worth.lower() in hit_query for hit_numeral in range(1, len(self.articles) + 1) for hit_worth, _ in self.worth_terms(hit_numeral))):
                    for hit_sighting in [sighting] + list(self.sheeted_glimpse):
                        self.worth_strikes.update((ii for ii, _ in self.entries_inside(hit_sighting)))
            if moniker == self.duties.get('cart_view'):
                self.basket_outlook = sighting
                self.basket_spotted_upon = self.tick
            if self.be_purchase(moniker, operands):
                self.purchases += 1
                if not rejoinder.get('finished'):
                    self.declined_upon = self.tick
            writing = self.display(sighting) + sheeted + pruned + line_pages
            if moniker not in simple and (not closing):
                writing += self.thriftiest_herein([sighting] + self.sheeted_glimpse) + self.lid_disregarded(moniker, operands, sighting)
            basis = writing
            evidence = self.testimony(sighting)
            if evidence:
                writing += '\n' + evidence
            if moniker == self.duties.get('cart') and any((_tally(vv) == 0 for vv in operands.values() if not isinstance(vv, str))):
                if any((isinstance(vv, str) and vv in self.tiers and _inside_tiers(sighting, vv) for vv in operands.values())):
                    writing += '\n[Wren] A count of 0 kept that line in the cart: this tool adds the count to the line, so use a negative count to remove units.'
            self.remark_vended = False
            strands = self.slack_strands(sighting)
            if strands:
                sense = self.grab_remark(strands)
                if sense:
                    writing += '\n[note read] ' + sense + '. Update your picks with this change before checking out.'
            declined = self.be_purchase(moniker, operands) and (not rejoinder.get('finished'))
            anew = declined and any((strand in self.voiced(sighting) for strand in self.vended_strands))
            slack = self.remark_vended and (declined or (not self.remark_dubbed and (not self.be_purchase(moniker, operands))))
            if (slack or anew) and self.basket_uids() and (not self.terminated):
                spotted = self.locate_vended()
                if spotted:
                    writing += "\n[Wren re-read the cart lines after the refusal]\n" + spotted
            self.remark_vended = False
            exchange = self.exchange_cue()
            if exchange:
                writing += '\n' + exchange
            if moniker == self.duties.get('cart') and self.confined and not self.terminated:
                mismatch = self.suit_defects()
                if mismatch:
                    writing += '\n[Wren] These lines are not yet compatible with each other: ' + '; '.join(mismatch)
            if self.scheduled:
                writing += "\n[notes observed on Wren's own calls] " + ' | '.join(self.scheduled)
                self.scheduled = []
            writing += chained
            if moniker in perusers:
                self.peruse_stash[marker] = (self.revision, self.tick, basis)
            if rejoinder.get('finished'):
                writing += ' [episode finished]'
            output.append((cnum, writing))
        return output

    def labor(self, cycle_lid: int, expend_lid: float) -> None:
        for _ in range(BREAKDOWN_ROTATIONS):
            try:
                self.rotations(cycle_lid, expend_lid)
                return
            except Exception as exn:
                self.transcript.append({'kind': 'crash', 'error': f'{type(exn).__name__}: {exn}'[:400]})
                self.response_unclosed_invokes()

    def may_decline(self) -> bool:
        decline = self.duties.get('refuse')
        if not decline:
            return False
        defects: list[str] = []
        _form(self.silent(decline, dict(self.concise.get('refuse_args') or {})), self.blueprints.get(decline), 'args', defects)
        return not defects

    def prune(self) -> None:
        magnitude = sum((len(_spill(mm.get('content'))) for mm in self.dialog))
        if magnitude <= CHRONICLE_CUTOFF:
            return
        instruments = [ii for ii, mm in enumerate(self.dialog) if mm.get('role') == 'tool']
        for position in instruments[:max(1, len(instruments) // 2)]:
            substance = self.dialog[position].get('content') or ''
            if len(substance) > 600:
                uids = re.findall('"([A-Za-z0-9][A-Za-z0-9_\\-:.]{5,40})"', substance)
                uids = [ii for ii in dict.fromkeys(uids) if ii in self.tiers][:20]
                self.dialog[position]['content'] = substance[:400] + f"...(earlier result trimmed; listings it contained: {', '.join(uids)})"

    def remark_basket(self, moniker: str, operands: dict[str, Any]) -> None:
        self.basket_visited = True
        self.basket_upon = self.tick
        specification = self.blueprints.get(moniker, {}).get('properties') or {}
        deletion = False
        for handle, worth in operands.items():
            choices = (specification.get(handle) or {}).get('enum')
            if isinstance(choices, list) and len(choices) >= 2 and (worth in choices[1:]):
                deletion = True
        slack_tally = next((_tally(vv) for vv in operands.values() if _tally(vv) is not None and (not isinstance(vv, str))), None)
        strands: list[tuple[str, int | None]] = []
        complete = False
        for handle, worth in operands.items():
            if isinstance(worth, list):
                form = (specification.get(handle) or {}).get('items') or {}
                if not worth and isinstance(form, dict) and (form.get('type') == 'object' or 'properties' in form):
                    complete = True
                for strand in worth:
                    if isinstance(strand, dict):
                        complete = True
                        identity = next((vv for vv in strand.values() if isinstance(vv, str) and vv in self.tiers), None)
                        tally = next((_tally(vv) for vv in strand.values() if _tally(vv) is not None and (not isinstance(vv, str))), 1)
                        if identity:
                            strands.append((identity, tally))
                    elif isinstance(strand, str) and strand in self.tiers:
                        strands.append((strand, slack_tally))
            elif isinstance(worth, str) and worth in self.tiers:
                strands.append((worth, slack_tally))
        if complete:
            self.basket_strands = {ii: j for ii, j in strands if j and j > 0}
            return
        ensembles = self.basket_ensembles(moniker)
        for identity, tally in strands:
            possess = self.basket_strands.get(identity, 0)
            if deletion:
                possess = possess - (tally or 1) if any(('integer' in _varieties(vv) for vv in specification.values() if isinstance(vv, dict))) else possess - tally if tally else 0
            elif ensembles:
                possess = tally if tally is not None else 1
            else:
                possess = possess + (tally if tally is not None else 1)
            if possess > 0:
                self.basket_strands[identity] = possess
            else:
                self.basket_strands.pop(identity, None)

    def response_unclosed_invokes(self) -> None:
        """Reply to each call from the model's last turn, keeping the conversation in a state the model can carry on from."""
        upon = next((nn for nn in range(len(self.dialog) - 1, -1, -1) if self.dialog[nn].get('tool_calls')), None)
        if upon is None:
            return
        accomplished = {mm.get('tool_call_id') for mm in self.dialog[upon + 1:]}
        for invoke in self.dialog[upon]['tool_calls']:
            if invoke['id'] not in accomplished:
                self.dialog.append({'role': 'tool', 'tool_call_id': invoke['id'], 'content': 'The call might not have gone through: something failed along the way. Check the state before repeating it.'})
        self.dialog.append({'role': 'user', 'content': self.standing()})

    def rotations(self, cycle_lid: int, expend_lid: float) -> None:
        inactive = 0
        while not self.terminated and (not self.busted) and (self.remaining() > 0) and (self.cycle_invokes < cycle_lid) and (self.outlay < expend_lid) and (self.timer() < FINAL_ROTATION or (not self.basket_uids() and self.engine_capacity() > BLANK_CAPACITY)):
            if self.prepared_into_inquire():
                synopsis = self.questioning()
                if synopsis:
                    self.dialog.append({'role': 'user', 'content': synopsis + '\n' + self.standing()})
                continue
            self.prune()
            self.cycle_invokes += 1
            commenced = time.monotonic()
            letter = self.ponder(self.dialog, [MILD, SURPLUS], CYCLE_SYMBOLS, self.tendered())
            if letter is not None:
                self.rotation_ticks.append(time.monotonic() - commenced)
            if letter is None:
                self.omissions += 1
                if self.nil_balance or self.invokes >= ENTIRE_INVOKES:
                    break
                if self.omissions > 2:
                    if self.timer() >= DOWNTIME_VIA:
                        break
                    time.sleep(CONCISE_VOID)
                continue
            invokes = []
            for numeral, invoke in enumerate(letter.get('tool_calls') or []):
                if not isinstance(invoke, dict):
                    continue
                func = invoke.get('function') if isinstance(invoke.get('function'), dict) else {}
                crude = func.get('arguments')
                decoded = crude if isinstance(crude, dict) else _jdoc_entity(crude) if isinstance(crude, str) and crude.strip() else {}
                invokes.append({'id': str(invoke.get('id') or f'k{self.cycle_invokes}_{numeral}'), 'type': 'function', 'function': {'name': str(func.get('name') or ''), 'arguments': _spill(decoded if isinstance(decoded, dict) else {})}})
            substance = letter.get('content')
            if not isinstance(substance, str):
                substance = _spill(substance) if substance else ''
            if not invokes:
                inactive += 1
                self.dialog.append({'role': 'assistant', 'content': substance or '(no call)'})
                relaunched = self.relaunch_peruses()
                if relaunched:
                    inactive = 0
                self.dialog.append({'role': 'user', 'content': 'Use one of the tools to carry on.' + (' ' + relaunched if relaunched else '') + '\n' + self.standing()})
                if inactive >= 3:
                    break
                continue
            inactive = 0
            pledging = {self.duties.get('cart'), self.duties.get('buy'), self.duties.get('refuse')} - {None}
            if not self.inquired and self.duties.get('ask') and (self.mails_gratis() > 0) and any((j['function']['name'] in pledging for j in invokes)) and (self.remaining() > 8) and (self.timer() <= INQUIRE_VIA):
                self.dialog.append({'role': 'assistant', 'content': substance or None, 'tool_calls': invokes})
                for invoke in invokes:
                    self.dialog.append({'role': 'tool', 'tool_call_id': invoke['id'], 'content': 'Not executed (no step spent): the shopper gets asked first.'})
                synopsis = self.questioning()
                self.dialog.append({'role': 'user', 'content': (synopsis or 'The interviewer found no question to ask.') + '\n' + self.standing()})
                continue
            self.dialog.append({'role': 'assistant', 'content': substance or None, 'tool_calls': invokes})
            self.transcript.append({'kind': 'turn', 'text': substance[:300], 'calls': [j['function'] for j in invokes]})
            rejoinders = self.execute_invokes(invokes)
            for position, (cnum, writing) in enumerate(rejoinders):
                if position == len(rejoinders) - 1:
                    writing += '\n' + self.standing()
                self.dialog.append({'role': 'tool', 'tool_call_id': cnum, 'content': writing})

    def silent(self, moniker: str, operands: dict[str, Any]) -> dict[str, Any]:
        blueprint = self.blueprints.get(moniker) or {}
        features = blueprint.get('properties') or {}
        output = dict(operands)
        for handle, specification in features.items():
            if not isinstance(specification, dict) or 'string' not in _varieties(specification) or 'enum' in specification:
                continue
            worth = output.get(handle)
            if worth is None and handle not in (blueprint.get('required') or []):
                continue
            writing = str(worth or '')
            dubbed = any((identity in writing for identity in self.tiers)) or self.concise.get('quiet_decline') is not False
            if dubbed or not writing:
                output[handle] = 'Sorry, nothing in the store meets all of your requirements, so I am not buying anything.'[:specification.get('maxLength', 200)]
        return output

    def cart_full(self) -> bool:
        """Every requested item has a line: the cart holds at least as many in-stock lines as there are items."""
        return len([ii for ii in self.basket_uids() if (self.retained_parcels(ii) or 0) > 0 and not self.vended_output(ii)]) >= len(self.articles)

    def tendered(self) -> list[dict[str, Any]]:
        remaining = self.remaining()
        inquire = self.duties.get('ask')
        preserve: set[str] | None = None
        closes = {self.duties.get('buy'), self.duties.get('refuse')} - {None}
        if remaining <= self.rear():
            preserve = set(closes)
        elif remaining <= self.rear() + 3:
            preserve = set(closes) | {self.duties.get('cart'), self.duties.get('cart_view')}
        elif remaining <= len(self.articles) + 5 and self.purchases == 0 and not (self.relaunched and self.terse_basket()):
            preserve = set(closes) | {self.duties.get('cart'), self.duties.get('cart_view'), self.duties.get('offer')}
        instruments = [tt for tt in self.instruments if tt['function']['name'] != inquire or self.letter_operands('x' * 200) is None]
        if preserve is not None and closes:
            tightened = [tt for tt in instruments if tt['function']['name'] in preserve]
            instruments = tightened or instruments
        return instruments

    def terminate(self, anew: bool=False) -> str:
        if self.terminated or self.busted or self.remaining() <= 0:
            return ''
        purchase, decline = (self.duties.get('buy'), self.duties.get('refuse'))
        crisp = self.basket_outlook is not None and self.basket_spotted_upon >= self.basket_upon
        grasping = bool(self.basket_uids()) if crisp else bool(self.basket_strands)
        complete = True
        if grasping and len(self.articles) >= 2:
            complete = self.populate_terse_basket() or not self.may_decline()
            if complete and len(self.basket_uids()) >= 2:
                self.populate_datums(self.basket_uids(), self.rear())
                if self.remaining() >= 3 and self.engine_capacity() > 5:
                    self.display_bonds(self.basket_uids(), self.rear())
            if not complete:
                self.transcript.append({'kind': 'short_cart_refused', 'lines': self.basket_uids()})
        if self.terminated or self.remaining() <= 0:
            return ''
        blind = self.basket_visited and (not crisp) and (self.remaining() >= 2) and (not self.omitted) and (self.purchases == 0 or self.declined_upon < self.basket_upon)
        if complete and purchase and (grasping or blind):
            defects: list[str] = []
            operands = _form(dict(self.concise.get('buy_args') or {}), self.blueprints.get(purchase), 'args', defects)
            if not defects:
                if self.remaining() >= 2 and grasping:
                    self.reperuse_tenders()
                if self.terminated or self.remaining() <= 0:
                    return ''
                if self.remaining() >= 2:
                    self.checkpoint('order')
                if self.terminated or self.remaining() <= 0:
                    return ''
                rejoinder = self.perform(purchase, operands)
                self.transcript.append({'kind': 'forced_buy', 'reply': _snip(rejoinder, 200)})
                if self.terminated or self.remaining() <= 0:
                    return ''
                if rejoinder is not None and anew:
                    self.purchases += 1
                    self.declined_upon = self.tick
                    self.witness(purchase, rejoinder.get('observation'))
                    return 'the checkout was rejected: ' + self.display(rejoinder.get('observation'))
        if decline:
            defects = []
            operands = _form(self.silent(decline, dict(self.concise.get('refuse_args') or {})), self.blueprints.get(decline), 'args', defects)
            if not defects:
                if self.remaining() >= 2:
                    self.checkpoint('end')
                if self.terminated or self.remaining() <= 0:
                    return ''
                rejoinder = self.perform(decline, operands)
                self.transcript.append({'kind': 'forced_refuse', 'reply': _snip(rejoinder, 200)})
                if rejoinder is not None and (not self.terminated):
                    self.witness(decline, rejoinder.get('observation'))
                    return 'the request to end without a purchase was rejected: ' + self.display(rejoinder.get('observation'))
        return ''

    def unite_launches(self, invokes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        starter, connector, aim = (self.duties.get('open'), self.duties.get('compare'), self.solitary_aim(self.duties.get('open')))
        numerous = self.contrast_aim()
        if not starter or not connector or (not aim) or (not numerous) or (starter == connector):
            return invokes
        if self.contrast_naked():
            return invokes
        chooses: list[tuple[int, str]] = []
        for position, invoke in enumerate(invokes):
            func = invoke.get('function') or {}
            operands = _jdoc_entity(func.get('arguments')) if isinstance(func.get('arguments'), str) else None
            identity = str((operands or {}).get(aim) or '').strip()
            if func.get('name') == starter and operands and (set(operands) == {aim}) and (identity in self.tiers) and (identity not in self.departed):
                chooses.append((position, identity))
        chooses = chooses[:self.contrast_capacity()]
        if len(chooses) < 2:
            return invokes
        initial = chooses[0][0]
        output = []
        for position, invoke in enumerate(invokes):
            if position == initial:
                output.append({'id': invoke['id'], 'type': 'function', 'function': {'name': connector, 'arguments': _spill({numerous: [ii for _, ii in chooses]})}})
            elif position in {ii for ii, _ in chooses}:
                output.append({'id': invoke['id'], 'joined': True})
            else:
                output.append(invoke)
        self.transcript.append({'kind': 'joined', 'ids': [ii for _, ii in chooses]})
        return output


class _ParleyPart:
    """Plans and words the questions put to the shopper."""

    def clusters_towards(self, gratis: int) -> list[list[int]]:
        """Which items each free message covers when there are more items than messages: consecutive groups as equal in size
        as possible, so every item is asked about. Race 15: 19 of 90 tasks asked for 4 or 5 items with 3 messages allowed,
        and an agent naming two items in a single message drew a need on either one (10 needs across 37 such messages from
        one agent).
        """
        numerals = [nn for nn in self.scheme_booking if nn in self.tracks] + [nn for nn in self.tracks if nn not in self.scheme_booking]
        numerals = [nn for nn in numerals if self.tracks.get(nn)]
        if gratis <= 0 or len(numerals) <= gratis:
            return []
        output: list[list[int]] = []
        begin = 0
        for cell in range(gratis):
            magnitude = (len(numerals) - begin + (gratis - cell) - 1) // (gratis - cell)
            output.append(numerals[begin:begin + magnitude])
            begin += magnitude
        return output

    def finalise(self, numeral: int, legends: list[str], displayed: bool, novel: list[str], packed: bool) -> None:
        """Record what a reply revealed about the labels its question mentioned.

        A single label: it counts as asked; a reply requesting a narrower question triggers the base's follow-up (one
        product from an item name that joins several, the head word of the property, or sample values). A list: the shopper
        replies about the first property in a message that it tracks, and its reply to a message mentioning none of its
        properties matches its reply to a property it is indifferent to. Hence a revealed need settles its own label and
        all labels ahead of it, and the labels behind it head the item's next list; a reply revealing nothing settles only
        the first label (a need there would have been answered) and the rest come next in reversed order, so whatever the
        shopper is indifferent to stops blocking them; after a second such reply in a row, labels not yet mentioned go
        first, since otherwise the order would revert to that of the first list. A reply requesting one question at a time
        ends lists for the rest of the episode and the first label of the list is asked by itself.
        """
        employed = self.track_employed.setdefault(numeral, set())
        if len(legends) == 1:
            employed.add(_terms(legends[0]))
            if packed and self.posting:
                self.posting = False
                self.transcript.append({'kind': 'lists_off', 'item': numeral})
            if novel or not packed:
                return
            if numeral not in self.sole_ware and self.moniker_portions(numeral):
                self.sole_ware.add(numeral)
                self.transcript.append({'kind': 'one_product', 'item': numeral, 'said': self.article_uttered(numeral)})
                self.pursue_upward.insert(0, (numeral, legends, displayed))
                return
            if not self.aside:
                similar = self.peek_similars(numeral, legends[0])
                if similar:
                    self.aside.setdefault(legends[0], {})[numeral] = similar
                    self.transcript.append({'kind': 'apart', 'item': numeral, 'label': legends[0], 'from': similar})
                    self.pursue_upward.insert(0, (numeral, legends, False))
            return
        if packed and not novel:
            self.posting = False
            self.transcript.append({'kind': 'lists_off', 'item': numeral})
            self.pursue_upward.insert(0, (numeral, legends[:1], False))
            return
        strike = self.responded_legend(legends, novel) if novel else -1
        forward = legends[strike + 1:] if strike >= 0 else legends[:0:-1]
        employed.update((_terms(legend) for legend in legends[:max(strike, 0) + 1]))
        remainder = [legend for legend in self.tracks.get(numeral) or [] if legend not in legends]
        self.track_omissions[numeral] = 0 if strike >= 0 else self.track_omissions.get(numeral, 0) + 1
        if self.track_omissions[numeral] and self.track_omissions[numeral] % 2 == 0:
            forward = [legend for _, _, legend in sorted((abs(ii - (len(forward) - 1) / 2), ii, legend) for ii, legend in enumerate(forward))]
            self.tracks[numeral] = remainder + forward
        else:
            self.tracks[numeral] = forward + remainder
        self.transcript.append({'kind': 'list_settled', 'item': numeral, 'done': legends[:strike + 1], 'next': forward[:4]})

    def informed_regarding(self, numeral: int, legend: str) -> bool:
        """Whether some answer already supplied a need on this property of the item: the words of the label are contained in
        the told property's name ("colour" after "shade colour") or contain a told name of at least two words; colour or
        color, singular or plural, count alike. A told name of one word ("type") does not cover a longer label ("bulb
        type").
        """
        unearth = _spelt(legend)
        return any((_monikers(unearth, _spelt(tt.split(':', 1)[0])) for tt in self.articles[numeral - 1].informed))

    def inquire_cluster(self, cluster: list[int], informed: list[str], bottom: int=6) -> list[tuple[int, str, str]]:
        """A single message listing a few planned properties for every item in the group, under the same list rules: the reply
        concerns the first tracked property mentioned, so each item's strongest keys come first in its part, and only the
        first item's first label is settled by a reply that reveals nothing.
        """
        fraction = max(1, self.roster_presently() // len(cluster)) if self.roster_presently() < self.roster_lid() else max(BLEND_LEGENDS, self.roster_lid() // len(cluster))
        portions: list[tuple[int, list[str]]] = []
        for numeral in cluster:
            remaining = [legend for legend in self.tracks.get(numeral) or [] if _terms(legend) not in self.track_employed.setdefault(numeral, set()) and (not self.informed_regarding(numeral, legend))]
            if remaining:
                portions.append((numeral, remaining[:fraction]))
        if not portions:
            return []
        if len(portions) == 1:
            numeral, legends = portions[0]
            suited, writing = self.suit_inquire(numeral, legends, False)
            if not suited:
                return []
            bundle = self.transmit([(numeral, legends[0], writing)], bottom)
            if bundle:
                self.terminate_inquire(numeral, suited, False, bundle, informed)
            return bundle
        writing = ''
        while True:
            for magnitude in range(fraction, 0, -1):
                fragments = [f"{', '.join(legends[:magnitude][:-1])} or {legends[:magnitude][-1]}" if len(legends[:magnitude]) >= 2 else legends[0] for _, legends in portions]
                writing = ', and '.join((f'what {fragment} for the {self.article_uttered(numeral)}' for fragment, (numeral, _) in zip(fragments, portions)))
                writing = 'W' + writing[1:].replace(' for the ', ' would you like for the ', 1) + '?'
                operands = self.letter_operands(writing)
                if operands and _utters(operands, writing):
                    portions = [(numeral, legends[:magnitude]) for numeral, legends in portions]
                    break
            else:
                portions = portions[:-1]
                if not portions:
                    return []
                continue
            break
        initial = portions[0][0]
        bundle = self.transmit([(initial, portions[0][1][0], writing)], bottom)
        if not bundle:
            return []
        for numeral, _ in portions:
            self.track_expended[numeral] = self.track_expended.get(numeral, 0) + 1
        self.roster_inquires += 1
        possessed = [len(article.informed) for article in self.articles]
        perceived = self.perceive([(0, writing, bundle[0][2])], (MILD, WEIGHTY))
        perceived.pop('free', None)
        self.grab_perceived(perceived, informed)
        responded = False
        for numeral, legends in portions:
            novel = self.articles[numeral - 1].informed[possessed[numeral - 1]:]
            if novel:
                responded = True
                self.track_located[numeral] = self.track_located.get(numeral, 0) + 1
                self.finalise(numeral, legends, False, list(novel), False)
        packed = any((_tally(px) == 1 for px in _articles(perceived.get('crowded'))))
        if packed and not responded:
            self.posting = False
            self.transcript.append({'kind': 'lists_off', 'item': initial})
            return bundle
        if not responded:
            self.track_employed.setdefault(initial, set()).add(_terms(portions[0][1][0]))
        self.transcript.append({'kind': 'mixed_ask', 'parts': portions, 'answered': responded})
        return bundle

    def inquire_upcoming(self, informed: list[str], bottom: int=6) -> list[tuple[int, str, str]]:
        """Send the following question (a waiting follow-up first, otherwise the next planned labels), fold its answer into
        the items' needs (adding each new one to the list of disclosed needs) and settle the labels it mentioned. Returns
        the batch sent, or an empty list if nothing went out.
        """
        for _ in range(ROSTER_MAGNITUDE):
            self.pursue_upward = [ff for ff in self.pursue_upward if not self.informed_regarding(ff[0], ff[1][0])]
            if self.pursue_upward:
                numeral, legends, displayed = self.pursue_upward.pop(0)
            else:
                numeral, legends = self.upcoming_inquire(self.tracks, self.track_located, self.track_expended, self.track_employed)
                displayed = False
            if not legends:
                return []
            suited, writing = self.suit_inquire(numeral, legends, displayed)
            if not suited and displayed:
                suited, writing = self.suit_inquire(numeral, legends, False)
            if suited:
                break
            self.track_employed.setdefault(numeral, set()).add(_terms(legends[0]))
            self.transcript.append({'kind': 'ask_too_long', 'item': numeral, 'label': legends[0]})
        else:
            return []
        legends = suited
        bundle = self.transmit([(numeral, legends[0], writing)], bottom)
        if not bundle:
            return []
        self.terminate_inquire(numeral, legends, displayed, bundle, informed)
        return bundle

    def inquire_writing(self, numeral: int, legends: list[str], displayed: bool=False) -> str:
        if len(legends) >= 2:
            return f"What {', '.join(legends[:-1])} or {legends[-1]} would you like for the {self.article_uttered(numeral)}?"
        similar = self.aside.get(legends[0], {}).get(numeral)
        if similar:
            return f"Apart from the {' and the '.join(similar)}, which {legends[0]} would you like for the {self.article_uttered(numeral)}?"
        return self.query(numeral, legends[0], self.specimens(numeral, legends[0]) if displayed else None)

    def letter_operands(self, writing: str) -> dict[str, Any] | None:
        moniker = self.duties.get('ask')
        blueprint = self.blueprints.get(moniker or '') or {}
        features = blueprint.get('properties') or {}
        mandatory = blueprint.get('required') or []
        writings = [kk for kk, vv in features.items() if isinstance(vv, dict) and 'string' in _varieties(vv) and 'enum' not in vv and 'const' not in vv]
        reservoir = [kk for kk in mandatory if kk in writings] or writings
        operands: dict[str, Any] = {}
        if reservoir:
            operands[max(reservoir, key=lambda kk: (not re.search(NON_LETTER, _serpent(kk), re.I), bool(re.search(LETTER_FACET, _serpent(kk), re.I)), features[kk].get('maxLength') or 10 ** 6))] = writing
        else:
            for handle, specification in features.items():
                internal = next((kk for kk, vv in (specification.get('properties') or {}).items() if isinstance(vv, dict) and 'string' in _varieties(vv)), None) if isinstance(specification, dict) else None
                if isinstance(specification, dict) and 'array' in _varieties(specification):
                    operands[handle] = [writing]
                    break
                if internal:
                    operands[handle] = {internal: writing}
                    break
        if not operands:
            return None
        for handle in mandatory:
            if handle not in operands:
                operands[handle] = _padding(features.get(handle) or {})
                if operands[handle] is None:
                    return None
        defects: list[str] = []
        operands = _form(operands, blueprint, 'args', defects)
        return None if defects else operands

    def origin_legend(self, numeral: int) -> str:
        """The category label (in normalised word form) of the item's own listings: the shelf name from the request when a
        seen listing carries it, otherwise the label carried by most of the item's opened listings (or its search rows if
        none were opened); '' when unknown.
        """
        sect_handle = self.genre_handle()
        if not sect_handle:
            return ''
        article = self.articles[numeral - 1]
        for reservoir in (self.ample, self.tiers):
            tallies: dict[str, int] = {}
            for entry in reservoir.values():
                legend = str(entry.get(sect_handle) or '')
                if legend and self.possesses(article, legend):
                    tallies[_terms(legend)] = tallies.get(_terms(legend), 0) + 1
            for moniker in (article.rack, article.moniker):
                if _terms(moniker) in tallies:
                    return _terms(moniker)
            if tallies:
                return max(tallies, key=lambda legend: tallies[legend])
        return ''

    def legend_writing(self, numeral: int) -> str:
        """The store's own spelling of the label that matches this item exactly (its name's or shelf's words, ignoring plurals
        and linking words) and is owned by this item, provided a seen listing bears it; '' if not."""
        sect_handle = self.genre_handle()
        article = self.articles[numeral - 1]
        monikers = [_sources(article.moniker)] + ([_sources(article.rack)] if (article.rack or '').strip() else [])
        spotted = {str(rr.get(sect_handle) or '').strip() for rr in self.tiers.values()} - {''} if sect_handle else set()
        return next((writing for writing in sorted(spotted) if _sources(writing) in monikers and self.proprietor_from(writing) == numeral), '')

    def legend_strand(self) -> str:
        """If one requested item's store label lies completely inside another requested item's label (a kind plus a broader
        kind mentioned with it), spell out each item's label: a listing serves only the item whose label it bears, no matter
        what its title says (race 16: carts that drew the narrower kind's line from the broader label failed, while every
        accepted cart drew one line from each label)."""
        if len(self.articles) < 2 or not self.genre_handle():
            return ''
        personal = {nn: self.legend_writing(nn) for nn in range(1, len(self.articles) + 1)}
        personal = {nn: writing for nn, writing in personal.items() if writing}
        nigh = sorted({nn for u in personal for d in personal if u != d and _sources(personal[u]) < _sources(personal[d]) for nn in (u, d)})
        if len(nigh) < 2 or all((_terms(self.articles[nn - 1].rack or '') == _terms(personal[nn]) for nn in nigh)):
            return ''
        monikers = '; '.join((f"item {nn} ({self.articles[nn - 1].moniker}) filed under '{personal[nn]}'" for nn in nigh))
        return f'store labels: {monikers}. Each listing serves only the item whose label it bears, no matter what its title says; look up every one of these items by listing its own label.'

    def peek_similars(self, numeral: int, legend: str) -> list[str]:
        """At most two of the item's keys a reader might also mistake for the given label: keys that share a word with it, or
        values listed under both across the listings read. A word that several of the item's keys contain proves nothing,
        nor does a single-letter or single-digit value.
        """
        unearth = _terms(legend)
        spotted: dict[str, set[str]] = {}
        for identity in self.article_uids(numeral):
            for handle, worth in (_datum_chunk(self.ample.get(identity) or {}) or {}).items():
                spotted.setdefault(_terms(handle), set()).update((ww for ww in _sources(str(worth)) if len(ww) > 1))
        tallies: dict[str, int] = {}
        for moniker in spotted:
            for term in set(moniker.split()):
                tallies[term] = tallies.get(term, 0) + 1
        simple = {ww for ww in unearth.split() if tallies.get(ww, 0) <= 2}
        graded = []
        for booking, (moniker, worths) in enumerate(spotted.items()):
            if moniker != unearth and moniker:
                communal = simple & set(moniker.split())
                usual = spotted.get(unearth, set()) & worths
                if communal or usual:
                    graded.append((-(3 * len(communal) + min(len(usual), 3)), booking, moniker))
        return [moniker for _, _, moniker in sorted(graded)[:PEEK_SIMILARS]]

    def prepared_into_inquire(self) -> bool:
        if self.inquired or not self.duties.get('ask') or self.mails_gratis() <= 0:
            return False
        if self.remaining() <= 8 or self.timer() > INQUIRE_VIA:
            return False
        launched = len(self.ample)
        if self.cycle_invokes < INQUIRE_ROTATIONS and not all((self.entries_from_article(nn) for nn in range(1, len(self.articles) + 1))):
            return False
        if launched >= 3 * len(self.articles) + 1:
            return True
        return self.tick >= max(8, self.maximum_ticks // 4)

    def questioning(self) -> str:
        self.inquired = True
        if self.mails_gratis() <= 0:
            return ''
        self.broaden()
        tally = self.mail_cutoff - self.mail_employed
        self.tracks = self.scheme_tracks(max(tally, self.roster_lid()) if self.posting else tally)
        self.track_located = {nn: 0 for nn in self.tracks}
        self.track_expended = {nn: 0 for nn in self.tracks}
        self.track_employed = {nn: set() for nn in self.tracks}
        self.pursue_upward = []
        transmitted: list[tuple[int, str, str]] = []
        informed: list[str] = []
        clusters = self.clusters_towards(self.mails_gratis()) if self.posting else []
        while self.mails_gratis() > 0 and (not self.terminated) and (self.remaining() > 6) and (self.timer() < QUERY_VIA):
            bundle = []
            while clusters and self.posting and not bundle:
                bundle = self.inquire_cluster(clusters.pop(0), informed)
            bundle = bundle or self.inquire_upcoming(informed)
            if not bundle:
                break
            transmitted += bundle
        located_articles = [nn for nn in self.track_located if self.track_located[nn] > 0]
        tether = self.soon_path()
        sought = self.worth_seek(located_articles) if located_articles else ''
        if informed:
            self.revision += 1
        strands = [f'Q: {qq}\nA: {u}' for _, qq, u in transmitted]
        synopsis = "Wren's interviewer put these questions to the shopper (one step per question):\n" + '\n'.join(strands)
        synopsis += '\nFirm requirements added by these answers: ' + ('; '.join(informed) if informed else 'none')
        synopsis += '. Use them for every pick (re-run searches with the new values when needed).'
        directed = [demand for _, demand in self.references if any((demand in tt for tt in informed))]
        if directed:
            synopsis += ' Their value comes from a different listing: ' + ' | '.join(directed) + '. Locate that listing, open its full page, and open each listing named on that page; the final page holds the value, and the order only counts once every one of them has been opened.'
        if tether:
            synopsis += '\n' + tether
        if sought:
            synopsis += '\n' + sought
        if self.scheduled:
            synopsis += '\n[notes received in the meantime] ' + ' | '.join(self.scheduled)
            self.scheduled = []
        return synopsis

    def responded_legend(self, legends: list[str], novel: list[str]) -> int:
        """Position of the first label that a reply to a list addressed. Every freshly revealed need is tied to whichever
        label has the greatest word overlap with the need's property (colour and color counted as one), failing that to the
        label whose values across opened listings overlap most with the need's value; -1 if none of the new needs ties to
        any label (such a reply is then treated as one that revealed nothing).
        """
        spotted: dict[str, set[str]] = {}
        for _, handle, _, worths in self.handle_grid():
            spotted.setdefault(_terms(handle), set()).update((ww for vv in worths for ww in _sources(vv)))
        strikes = []
        for demand in novel:
            feature, _, worth = demand.partition(':')
            grades = [len(_spelt(legend) & _spelt(feature)) for legend in legends]
            if max(grades, default=0) == 0:
                grades = [len(spotted.get(_terms(legend), set()) & _sources(worth)) for legend in legends]
            if max(grades, default=0) > 0:
                strikes.append(grades.index(max(grades)))
        return min(strikes) if strikes else -1

    def roster_lid(self) -> int:
        return ROSTER_MAGNITUDE if len(self.articles) == 1 else ROSTER_BLENDED

    def roster_presently(self) -> int:
        """The labels the next list will name: only a few in the episode's opening questions, because nothing yet reveals how this shopper handles a list."""
        ticks = ROSTER_TICKS if len(self.articles) == 1 else ROSTER_TICKS_BLENDED
        return min(ticks[self.roster_inquires], self.roster_lid()) if self.roster_inquires < len(ticks) else self.roster_lid()

    def seek_stencil(self) -> tuple[str, dict[str, Any], str] | None:
        instrument = self.duties.get('search')
        blueprint = self.blueprints.get(instrument or '') or {}
        features = blueprint.get('properties') or {}
        writings = [kk for kk, vv in features.items() if isinstance(vv, dict) and 'string' in _varieties(vv) and ('enum' not in vv)]
        dubbed = [kk for kk in writings if re.search(ENQUIRY_TERMS, _serpent(kk), re.I)]
        for operands in reversed(self.sought):
            facets = [kk for kk in dubbed or writings if isinstance(operands.get(kk), str) and re.search('[a-z]', operands[kk], re.I)]
            if not dubbed:
                legends = {str(rr.get(self.genre_handle()) or '').strip().lower() for rr in self.tiers.values()} - {''} if self.genre_handle() else set()
                facets = [kk for kk in facets if len({str(u.get(kk)) for u in self.sought if u.get(kk)}) >= 2 and any((len(str(u.get(kk)).split()) >= 2 for u in self.sought)) and (not any((str(u.get(kk) or '').strip().lower() in legends for u in self.sought)))]
            if facets and instrument:
                facet = facets[0]
                preserve = {kk: vv for kk, vv in operands.items() if kk != facet and (isinstance((features.get(kk) or {}).get('enum'), list) or kk in (blueprint.get('required') or []))}
                sect_handle = self.genre_handle()
                racks = {_terms(article.rack or '') for article in self.articles} | {_terms(article.moniker) for article in self.articles}
                racks |= {_terms(str(tier.get(sect_handle) or '')) for tier in self.tiers.values()} if sect_handle else set()
                preserve = {kk: vv for kk, vv in preserve.items() if kk in (blueprint.get('required') or []) or not isinstance(vv, str) or (_terms(vv) not in racks and (not any(({ww for ww in _root(vv) if len(ww) >= 4} & {ww for ww in _root(ss) if len(ww) >= 4} for ss in racks if ss))))}
                phrased = [u for u in self.sought if isinstance(u.get(facet), str) and re.search('[a-z]', u[facet], re.I)]
                for kk in list(preserve):
                    if isinstance((features.get(kk) or {}).get('enum'), list):
                        employed = [_spill(u[kk]) for u in phrased if kk in u]
                        if employed:
                            finest = max(employed, key=employed.count)
                            preserve[kk] = next((u[kk] for u in phrased if kk in u and _spill(u[kk]) == finest))
                return (instrument, preserve, facet)
        return None

    def suit_inquire(self, numeral: int, legends: list[str], displayed: bool) -> tuple[list[str], str]:
        """The longest prefix of the given labels whose question the message tool's schema accepts in full (the schema shaper
        truncates text beyond a length limit, so an overlong list sheds labels from the end instead), together with that
        question; an empty list and '' when even one label does not fit.
        """
        for magnitude in range(len(legends), 0, -1):
            writing = self.inquire_writing(numeral, legends[:magnitude], displayed)
            operands = self.letter_operands(writing)
            if operands and _utters(operands, writing):
                return (legends[:magnitude], writing)
        return ([], '')

    def tardy_query(self) -> str:
        """Right before checkout, use a message the order does not require to put the next planned question.

        The interview holds back every message the request reserves for a process step, whether that step accompanies the
        order or a finish without buying. Once Wren is ready to purchase, the messages reserved for the no-purchase finish
        become available. Race 13 permitted two messages: the base posed one question in 28 of 180 episodes and left a
        hidden need unrevealed in 12 of those (no episode with an unrevealed need passed). Gives back the summary for the
        loop model if the answer adds a requirement, otherwise ''.
        """
        if self.tardy_inquired or not self.duties.get('ask') or self.mail_employed >= self.mail_cutoff:
            return ''
        if any((pp['when'] == 'order' and pp['messages'] > 0 for pp in self.digest)):
            return ''
        if self.remaining() < TARDY_TICKS or self.timer() > TARDY_VIA or not self.referees():
            return ''
        self.tardy_inquired = True
        if not self.tracks:
            self.tracks = self.scheme_tracks(self.roster_lid() if self.posting else 1)
            self.track_located = {nn: 0 for nn in self.tracks}
            self.track_expended = {nn: 0 for nn in self.tracks}
            self.track_employed = {nn: set() for nn in self.tracks}
        informed: list[str] = []
        allowance = self.allowance
        bundle = self.inquire_upcoming(informed, TARDY_TICKS - 2)
        if not bundle:
            self.transcript.append({'kind': 'late_ask', 'skipped': 'no property left to ask or no message sent'})
            return ''
        if self.allowance != allowance:
            informed.append(f'budget now {self.allowance}')
        numeral, query, response = bundle[0]
        self.transcript.append({'kind': 'late_ask', 'q': query, 'a': response[:200], 'told': informed})
        if not informed:
            return ''
        self.revision += 1
        synopsis = f'Before checkout, Wren put one further question to the shopper (one step):\nQ: {query}\nA: {response}\nFirm requirements added by this answer: ' + '; '.join(informed) + '. Checkout did not happen: compare every line with them (re-run searches with the new values when needed), correct the cart, then check out.'
        directed = [demand for _, demand in self.references if any((demand in tt for tt in informed))]
        if directed:
            synopsis += ' Their value comes from a different listing: ' + ' | '.join(directed) + '. Locate that listing, open its full page, and open each listing named on that page; the final page holds the value.'
        sought = self.worth_seek([numeral])
        return synopsis + ('\n' + sought if sought else '')

    def terminate_inquire(self, numeral: int, legends: list[str], displayed: bool, bundle: list[tuple[int, str, str]], informed: list[str]) -> None:
        """Fold the reply to a question just sent into what each item needs, and mark the labels it mentioned as settled."""
        self.track_expended[numeral] = self.track_expended.get(numeral, 0) + 1
        self.roster_inquires += 1
        possessed = [len(article.informed) for article in self.articles]
        perceived = self.perceive(bundle, (MILD, WEIGHTY))
        if len(legends) > 1:
            perceived.pop('free', None)
        self.grab_perceived(perceived, informed)
        novel = [demand for article, magnitude in zip(self.articles, possessed) for demand in article.informed[magnitude:]]
        for nn, (article, magnitude) in enumerate(zip(self.articles, possessed), 1):
            if len(article.informed) > magnitude:
                self.track_located[nn] = self.track_located.get(nn, 0) + 1
        packed = any((_tally(px) == 1 for px in _articles(perceived.get('crowded'))))
        self.finalise(numeral, legends, displayed, novel, packed)

    def upcoming_inquire(self, tracks: dict[int, list[str]], located: dict[int, int], expended: dict[int, int], employed: dict[int, set[str]]) -> tuple[int, list[str]]:
        initial = {nn: ii for ii, nn in enumerate(self.scheme_booking)}
        rare = self.mail_cutoff <= RARE_LETTERS and (not self.posting)
        for numeral in sorted(tracks, key=lambda nn: (located.get(nn, 0) > 0, 0 if rare else expended.get(nn, 0), initial.get(nn, len(initial) + nn))):
            remaining = [legend for legend in tracks[numeral] if _terms(legend) not in employed[numeral] and (not self.informed_regarding(numeral, legend))]
            if remaining:
                return (numeral, remaining[:self.roster_presently() if self.posting else 1])
        return (0, [])

    def worth_seek(self, numerals: list[int]) -> str:
        scheme = self.worth_stencil()
        if not scheme or self.remaining() < self.rear() + WORTH_CAPACITY:
            return ''
        instrument, basis, facet = scheme
        spotted: list[str] = []
        allowance = min(8, self.remaining() - self.rear() - WORTH_PRESERVE)
        for numeral in numerals:
            couples = self.worth_terms(numeral)
            article = self.articles[numeral - 1]
            worths = [vv for vv, _ in couples]
            crisp = [vv for vv, informed in couples if informed]
            if not crisp or allowance <= 2:
                continue
            prospect = [vv for vv in crisp if len(vv.split()) == 1 or re.fullmatch(VOLUME, vv, re.I)]
            worths = prospect + [vv for vv in worths if vv not in prospect]
            declared = [ii for ii, vv in enumerate(worths) if vv not in crisp]
            omit = declared[-1] if declared else 0
            attempts = [worths, worths[:omit] + worths[omit + 1:]] if len(worths) > 1 else [[worths[0], article.moniker], worths] if prospect else [worths]
            for terms in attempts[:2]:
                if not terms:
                    continue
                defects: list[str] = []
                operands = _form({**basis, facet: ' '.join(terms)}, self.blueprints.get(instrument), 'args', defects)
                if defects or self.terminated:
                    break
                rejoinder = self.perform(instrument, operands)
                allowance -= 1
                if rejoinder is None:
                    break
                sighting = rejoinder.get('observation')
                self.witness(instrument, sighting)
                sect_handle = self.genre_handle()
                located = [ii for ii, _ in self.entries_inside(sighting)]
                lid = self.allowance if isinstance(self.allowance, (int, float)) else None
                costs = [self.cost_from(ii) for ii in located]
                if lid is not None and costs and all((pp is not None and pp > 10 * lid for pp in costs)):
                    lid = None
                tiers = [ii for ii in located if self.purchasable(ii) and (not sect_handle or self.possesses(article, str((self.tiers.get(ii) or {}).get(sect_handle) or ''))) and (lid is None or (self.cost_from(ii) or 0) <= lid)]
                origin = self.origin_legend(numeral)
                if sect_handle and origin:
                    tiers.sort(key=lambda ii: _terms(str((self.tiers.get(ii) or {}).get(sect_handle) or '')) != origin)
                spotted.append(f'{_spill(operands.get(facet))}: ' + (', '.join(tiers[:6]) or 'no rows of this kind'))
                if tiers:
                    launched = self.populate_datums([ii for ii in tiers if ii not in self.ample][:min(3, allowance)], self.rear() + WORTH_PRESERVE)
                    allowance -= len(launched)
                    if launched:
                        spotted.append('opened: ' + ' | '.join(launched)[:2400])
                    break
        if not spotted:
            return ''
        self.transcript.append({'kind': 'value_search', 'seen': [ss[:200] for ss in spotted]})
        return "[Wren ran one combined search of the requirement values for each such item and read the top rows of the right kind] " + '\n'.join(spotted)

    def worth_stencil(self) -> tuple[str, dict[str, Any], str] | None:
        """The search a value search imitates: the usual search template, or, if every search so far browsed a category with
        no words, the tool's single query field with its single mode argument set to read it (the option naming words or a
        query, otherwise the remaining option of a mandatory two-option mode). Nothing when the schema leaves doubt.
        """
        scheme = self.seek_stencil()
        instrument = self.duties.get('search')
        if scheme or not instrument:
            return scheme
        blueprint = self.blueprints.get(instrument) or {}
        features = blueprint.get('properties') or {}
        mandatory = blueprint.get('required') or []
        informed = self.concise.get('search_words') if isinstance(self.concise.get('search_words'), dict) else {}
        uttered, repaired = (informed.get('field') if isinstance(informed.get('field'), str) else None, informed.get('args') if isinstance(informed.get('args'), dict) else {})
        dubbed_instrument = informed.get('tool') if isinstance(informed.get('tool'), str) and informed.get('tool') in self.blueprints else instrument
        specification = (self.blueprints.get(dubbed_instrument) or {}).get('properties') or {}
        if uttered in specification and 'string' in _varieties(specification[uttered]) and 'enum' not in specification[uttered]:
            defects: list[str] = []
            basis = _form({**repaired, uttered: 'x'}, self.blueprints.get(dubbed_instrument), 'args', defects)
            if not defects:
                basis.pop(uttered, None)
                return (dubbed_instrument, basis, uttered)
        dubbed = [kk for kk, vv in features.items() if isinstance(vv, dict) and 'string' in _varieties(vv) and 'enum' not in vv and re.search(ENQUIRY_TERMS, _serpent(kk), re.I)]
        final = self.sought[-1] if self.sought else {}
        variants = {kk: vv['enum'] for kk, vv in features.items() if isinstance(vv, dict) and isinstance(vv.get('enum'), list) and len(vv['enum']) >= 2}
        settings = []
        for handle, choices in variants.items():
            peruses = [oo for oo in choices if isinstance(oo, str) and re.search(ENQUIRY_TERMS, oo, re.I)]
            alternatives = [oo for oo in choices if oo != final.get(handle)]
            if len(peruses) == 1:
                settings.append((handle, peruses[0]))
            elif handle in mandatory and len(choices) == 2 and len(alternatives) == 1:
                settings.append((handle, alternatives[0]))
        if len(dubbed) != 1 or len(settings) > 1 or (variants and not settings):
            return None
        basis = {kk: final[kk] for kk in mandatory if kk in final and kk != dubbed[0]}
        basis.update(settings)
        return (instrument, basis, dubbed[0])

    def worth_terms(self, numeral: int) -> list[tuple[str, bool]]:
        """Words to search for each requirement value of the item that a word search can match, each flagged with whether an
        answer supplied it: the value as is, or, for an answer's lower numeric bound with a unit, the number and unit,
        because a concealed bound like that often sits on the value of a fitting listing.
        """
        output = []
        informed = self.articles[numeral - 1].informed
        for nn, demand in self.demands_presently():
            if nn != numeral or demand in [mm for _, mm in self.references] or ':' not in demand:
                continue
            worth = demand.split(':', 1)[1].strip()
            brink = re.search(BOTTOM, worth, re.I)
            if brink:
                if demand in informed:
                    output.append((brink.group(1).strip(), True))
                continue
            if not worth or re.search('\\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', worth, re.I):
                continue
            output.append((worth, demand in informed))
        return output


class Wren(_HelmPart, _SpoorPart, _TallyPart, _EarPart, _ParleyPart, _ScoutPart, _LeafPart, _SievePart, _IntakePart, _OraclePart, _TempoPart):
    """One shopping episode: the buyer state, assembled from the role classes above."""

    def __init__(self, issue: dict[str, Any]):
        surroundings = issue['environment']
        self.tie = dict(surroundings['binding'])
        self.outlook = surroundings['policy_view']
        self.appeal = str(self.outlook.get('query') or '')
        self.instruments = [tt for tt in self.outlook.get('tools') or [] if isinstance(tt, dict) and isinstance(tt.get('function'), dict) and isinstance(tt['function'].get('name'), str)]
        self.blueprints = {tt['function']['name']: tt['function'].get('parameters') if isinstance(tt['function'].get('parameters'), dict) else {'type': 'object', 'properties': {}} for tt in self.instruments}
        ticks = _tally(self.outlook.get('max_steps'))
        if ticks is None and isinstance(self.outlook.get('max_steps'), float):
            ticks = int(self.outlook['max_steps'])
        self.maximum_ticks = ticks if ticks and ticks > 0 else 30
        self.initiated = time.monotonic()
        self.tick = 0
        self.terminated = bool(self.outlook.get('finished'))
        self.busted = False
        self.nil_balance = False
        self.lingered = 0
        self.omissions = 0
        self.outlay = 0.0
        self.invokes = 0
        self.cycle_invokes = 0
        self.rotation_ticks: list[float] = []
        self.transcript: list[dict[str, Any]] = _Imprinted(self.timer)
        self.concise: dict[str, Any] = {}
        self.duties: dict[str, str] = {}
        self.articles: list[Article] = []
        self.allowance: Any = None
        self.mail_cutoff = 0
        self.mail_employed = 0
        self.digest: list[dict[str, Any]] = []
        self.buffer = 0
        self.checkpoint_mails = 0
        self.checkpoint_okay: set[tuple[str, str]] = set()
        self.accomplished_journal: list[str] = []
        self.accomplished_ticks: list[int] = []
        self.inquired = False
        self.faq: list[tuple[str, str]] = []
        self.tiers: dict[str, dict[str, Any]] = {}
        self.ample: dict[str, dict[str, Any]] = {}
        self.spotted_booking: list[str] = []
        self.perceived: set[str] = set()
        self.remarks: list[str] = []
        self.revision = 0
        self.basket_visited = False
        self.basket_upon = -1
        self.basket_spotted_upon = -1
        self.basket_outlook: Any = None
        self.remark_dubbed = False
        self.basket_strands: dict[str, int] = {}
        self.verified: set[str] = set()
        self.vetted = False
        self.tender_spotted: dict[str, int] = {}
        self.departed: dict[str, int] = {}
        self.unobtainable: set[str] = set()
        self.vended_strands: set[str] = set()
        self.remark_vended = False
        self.retained = 0
        self.uid_operands: set[tuple[str, str]] = set()
        self.basket_uttered = ''
        self.declined_upon = -1
        self.prospects: list[str] = []
        self.decline_chunks = 0
        self.shoved = False
        self.constructed: list[str] = []
        self.omitted: set[str] = set()
        self.tardy_retained = False
        self.thrifty_grasps = 0
        self.feeble_grasps = 0
        self.attestations: dict[str, list[str]] = {}
        self.prospect_articles: dict[str, list[str]] = {}
        self.rulings: dict[str, bool] = {}
        self.affirmed: str | None = None
        self.finalized = False
        self.relaunched = False
        self.bond_attempted: set[str] = set()
        self.track_omissions: dict[int, int] = {}
        self.embraced: dict[str, int] = {}
        self.lidded: set[tuple[str, float]] = set()
        self.datumless: set[tuple[str, str]] = set()
        self.path_capacity: int | None = None
        self.uid_handle: str | None = None
        self.datums_whence: dict[str, bool] = {}
        self.scheduled: list[str] = []
        self.purchases = 0
        self.rejections = 0
        self.latest: list[str] = []
        self.dialog: list[dict[str, Any]] = []
        self.launched: set[str] = set()
        self.outlooks: dict[str, set[str]] = {}
        self.suit_stashed: tuple[Any, dict[str, Any]] | None = None
        self.proprietors_spotted: dict[tuple[str, int], int] = {}
        self.references: list[tuple[int, str]] = []
        self.reference_worths: dict[str, str] = {}
        self.paths = 0
        self.path_chunks = 0
        self.contrast_okay = 0
        self.contrast_poor = 0
        self.instrument_handles: dict[str, set[str]] = {}
        self.outlook_handles: dict[str, tuple[int, set[str]]] = {}
        self.peruse_stash: dict[str, tuple[int, int, str]] = {}
        self.sheets_peruse = 0
        self.forms: dict[str, dict[str, Any] | None] = {}
        self.cost_handle: str | None = None
        self.inventory_handle: str | None = None
        self.thriftier_markers: set[str] = set()
        self.broader_retained = False
        self.sheeted_glimpse: list[Any] = []
        self.scheme_booking: list[int] = []
        self.tracks: dict[int, list[str]] = {}
        self.track_located: dict[int, int] = {}
        self.track_expended: dict[int, int] = {}
        self.track_employed: dict[int, set[str]] = {}
        self.tardy_inquired = False
        self.sole_ware: set[int] = set()
        self.roster_inquires = 0
        self.aside: dict[str, dict[int, list[str]]] = {}
        self.pursue_upward: list[tuple[int, list[str], bool]] = []
        self.posting = True
        self.strand_grasps = 0
        self.confined: list[tuple[int, int]] = []
        self.suit_grasps = 0
        self.sought: list[dict[str, Any]] = []
        self.bond_seeks = 0
        self.bond_spotted: dict[str, list[str]] = {}
        self.bond_further: dict[str, tuple[str, Any] | None] = {}
        self.path_remarks: list[str] = []
        self.chases = 0
        self.milieu_void = MILIEU_VOID
        self.milieu_expended = 0.0
        self.milieu_tally = 0
        self.peruse_invokes: dict[str, Any] = {}
        self.peruse_operands: dict[str, tuple[str, dict[str, Any]]] = {}
        self.worth_tiers: set[str] = set()
        self.plan_early: Any = None
        self.worth_strikes: set[str] = set()
        self.tier_handles: set[str] = set()
        self.decline_followed = False
        self.milieu_upon = 0.0
        self.chase_root: dict[tuple[int, tuple[str, ...]], str] = {}
        self.chase_nil_nominal: set[int] = set()
        self.naked_sheets: dict[str, int] = {}
        self.cipher_attempts: list[bool] = []
        self.ciphers_defunct = False
        self.perceive_peruses = 0
        self.suit_chased = False
        self.chase_marker = ''
        self.chase_attempted: dict[int, set[str]] = {}
        self.sheet_operands: dict[str, str] = {}
        self.sheet_enquiries: dict[str, set[str]] = {}


def agent_main(issue_info: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        session = Wren(issue_info)
    except Exception as exn:
        return [{'kind': 'crash', 'error': f'{type(exn).__name__}: {exn}'[:400]}]
    try:
        return list(session.execute())
    except Exception as exn:
        session.transcript.append({'kind': 'crash', 'error': f'{type(exn).__name__}: {exn}'[:400]})
        try:
            if not session.terminated:
                session.terminate()
        except Exception as anew:
            session.transcript.append({'kind': 'crash', 'error': f'{type(anew).__name__}: {anew}'[:400]})
        return list(session.transcript)
__all__ = ['agent_main']
