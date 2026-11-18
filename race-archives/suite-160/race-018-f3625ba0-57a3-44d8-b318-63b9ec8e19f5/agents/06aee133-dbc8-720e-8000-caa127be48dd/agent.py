from __future__ import annotations
import json
import random
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from src.agent.proxy_client import ProxyClient
SEAT_QUICK = 'anthropic/claude-haiku-5.5'
SEAT_DEEP = 'anthropic/claude-sonnet-5.5'
SEAT_ALT = 'openai/gpt-6-luna'
TURN_CAP = 42
MODEL_CALL_CAP = 60
BUDGET_EASE = 0.055
BUDGET_STOP = 0.078
BUDGET_HALT = 0.098
DEEP_MARGIN = 0.02
BACKOFF = (4, 10, 20, 35)
BACKOFF_TOTAL = 45
ROW_CHARS = 450
RESULT_CHARS = 15000
CHAT_CHARS = 150000
STEP_ROUNDS = 3
PAGE_AHEAD = 6
CHEAP_READS = 3
CLOCK_LIMIT = 235.0
FITMENT_RX = '\\b(?:compatible with|works with|works alongside|fits)\\s*:|\\bcompatible devices\\b|\\balso fits\\b|\\bevery [A-Za-z]{1,6}[ -]?\\d{1,5}x+\\b'
REASON_RX = '\\b(?:budget|cap|limit|afford|exceed\\w*|over the|more than|cents?|price[sd]?|costs?|total|sold out|out of stock|no stock|not in stock|unavailable)\\b'
_brain = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_store = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
CHARTER = 'You are Wren, a meticulous purchasing agent acting for a single shopper inside an online shop. Your only means of acting is to call the shop\'s tools. Scoring is binary, 1 or 0, and a 1 is awarded only for a finish that is exactly correct. That means either an order holding every item asked for, each at the quantity asked for, each satisfying all requirements that apply at the moment you check out, with the whole within budget; or, if the shop genuinely has nothing satisfying everything, closing the episode through the no-purchase tool in the manner the request specifies.\n\nSTORE MECHANICS\n- Each tool call uses up one of a limited number of steps; reasoning is free. This shop has its own tool names, argument names and result formats, so study every tool\'s description and schema and pass precisely the argument names and permitted values they define.\n- Searches may be restrictive: filtering by category, a keyword matcher that may consider only the opening words of your query, pages holding only a handful of rows, a next-page token or offset. Whenever the answer hinges on seeing every row, check the total the shop reports against the rows you have received and keep paging, using exactly the token or offset it supplies. Short keyword queries that lead with the value do best (the desired value followed by the product noun); put values in a query rather than the names of properties (a property name such as size or type rarely appears in a title). When a search naming the required values returns only a handful of rows of the item\'s category, open each affordable one: a title often hides what the attributes say. If a filtered search comes back empty, the cause is usually a wrong filter value or a wrong filter parameter: try again with the label spelled exactly as it appears in result rows, or switch to a different filter parameter.\n- A search row is only a summary. How a listing\'s record is divided varies by shop: one view may display the attributes (properties), a second view only its offer plus description, while a side-by-side comparison view may hold either. Once Wren has learned which tool displays attributes, the ledger names it. Base every property requirement on those attributes, which means reading them for any listing you might purchase; a single call covering several listings saves steps if it displays what you need. Do not reopen a listing if nothing about it has changed since your last read.\n- You never write to the shopper directly: Wren\'s interviewer handles any questions for you. Its answers show up in the conversation and get recorded in the ledger.\n- The request may also call for intermediate steps: an approval, a message, something to verify or read beforehand. These appear in the ledger. The ones due right before the order or the ending are performed by Wren itself at the moment you call checkout or the ending tool, so do not wait on them and do not end without buying on their account; handle the rest with the shop\'s tools (questions about products go via Wren\'s interviewer).\n\nWHEN A LISTING FITS\n- The listing\'s category label determines what kind of product it is: only a label matching the requested item as a whole qualifies, never an adjacent label.\n- A property requirement holds when the attribute the listing has for that property gives the value. Spelling of keys is irrelevant (spaces, underscores, camel case, prefixes), and a single member of a comma- or slash-separated list suffices. Values are compared by their words: differences of case, singular versus plural, and plain synonyms are acceptable, and a value containing the words of the required value qualifies ("deep teal" satisfies teal; "extra roomy fit" satisfies roomy). A value worded differently does not qualify (an abbreviation or code is not equal to the word it abbreviates, nor does "teal green" equal "sea green"). Convert units before comparing numbers; limits are inclusive of the boundary value; and a range must fall entirely within a limit.\n- If an attribute\'s key names the property of the requirement (by that name or a plain synonym), that attribute is decisive: should it give a different value, the listing fails regardless of the title or other keys. When several keys name the property, the most specific key decides (one that also names the product or its part, such as "clock shape" over a bare "shape"; spelling with spaces, underscores or dots makes no difference), and two equally specific keys (the same key spelled two ways, such as "dietary needs" and "dietary_needs") count together: the requirement is met when either of them gives the value. A key belonging to some other property, related or not (the colour of a handle says nothing about the body\'s colour), settles nothing for the requirement. After listings are opened, Wren appends [evidence] lines showing which keys have names matching a requirement.\n- If the listing lacks an attribute for that property, the desired value counts when it appears under another attribute (say, a styles or features list containing it), or appears in the title or the description. A number of a different kind does not count (a weight does not count as a capacity), and neither does a lone number with no text saying what quantity it is.\n- If the category label itself states the value of a requirement (for instance a season or audience inside the label), the category satisfies that requirement.\n- When a listing\'s own notes say one of its properties is to be taken from another listing ("take X from ..., not from this one", "for X, look at ..."), the value it prints for that property does not count: prefer a listing whose own record states the value, or open the listing it names and use the value there.\n- Instead of a value, a requirement may refer to another listing ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Settle that first: locate the referenced listing (search by its category, brand, shop or words), then open its full page with the single-item tool or the side-by-side tool (search rows are not enough), and if that page in turn points to yet another listing for the property, locate and open that one likewise; the value comes from the final page. The order only counts if every page in that chain was opened. When the desired value is anything other than the referenced listing\'s value, the rule excludes nothing but listings with that exact same value; on its own it is never a reason to end without buying.\n- If a listing you are on the point of buying points you to a further page with more detail about it, check that page before buying.\n- When one requested item must be compatible with another (say, an accessory for a device), pick the pair together based on the listings\' own wording (a list of what it is compatible with, a model or series code appearing on both pages). If none of the accessories meeting their own requirements fits the chosen device, move to a different device that satisfies its requirements and that an accessory meeting its own requirements does fit; do not end without buying while that switch remains untried.\n- Respect any exclusion or closed list. Treat wishes ("ideally", "would be nice") purely as tie-breakers.\n- Never buy anything the shop marks sold out. The total (each line\'s price multiplied by its quantity, summed over all lines) must stay within the budget; equalling it is fine.\n\nPICKING\n- If cheapest is what the request wants (lowest total, smallest price), your order has to contain the lowest-priced listing out of ALL fitting listings. Gather the entire fitting set (keep paging: rows frequently come in no price order, and Wren itself pulls further pages when a result is short), examine candidates from lowest price upward, and take the first that fits completely. If a search takes a price ceiling, use it: list the category below what your current best choice costs, so that only cheaper candidates remain to check.\n- If the request wants the most or the least of some measure, rank every fitting listing by it and pick the extreme one; a listing not stating the measure cannot win.\n- In every other case, any listing satisfying everything is correct: do not spend steps searching for something better.\n- Add each item to the cart the moment you have opened a listing satisfying all its requirements (for a cheapest, most or least choice, only after the fitting set has been compared), then go on to the next. Carted lines often sell out shortly after being added, and carting early leaves steps for a replacement.\n- A request covering several distinct items needs a separate cart line for each, each at the quantity requested.\n\nMID-TASK CHANGES\n- Things may change because of the shopper or the shop: a changed budget, a requirement added or removed, a price change, a listing selling out. These show up inside tool results, frequently when you attempt checkout (as a refusal); such a note counts no matter which tool\'s result carries it, and the requirements currently in force are listed in the ledger. A refusal does not end the episode: apply all the changes, correct the cart and check out once more.\n- Shortly before checkout, re-read the record of each listing you chose, since price and stock can change without notice.\n\nENDING\n- Make the cart hold exactly the chosen lines (remove everything else), view the cart once, then check out in a call by itself.\n- End without buying only once a genuine search has turned up no listing satisfying everything, and only when you can demonstrate the reason: a total above the budget, a stated limit that none of the listings meets, no stock. Being low on steps or having an unread page demonstrates nothing of that sort: in that case check out with the best cart you have built. Word that ending as the request directs: unless it forbids suggesting alternatives, name in the ending text the closest listing (giving its id, price and stock) along with what it fails on. The closest listing is the one breaking the fewest requirements, with the budget counted as one requirement; among ties, the cheapest.\n- Each reply must include at least one tool call. Keep text brief: what you know, then your next move.'
INTAKE_FORM = 'Your job is to turn a shopper\'s request into a brief that an agent will shop from. Study the request alongside the store\'s tool list, then return ONE JSON object, compact and unindented, with nothing else around it, holding the keys below:\n\n"items": a list holding one entry for every separate product the shopper wants to buy, each shaped {"name": what kind of product it is, worded the way the request words it, "shelf": the store\'s category label for that product if some tool lists labels (copied character for character), otherwise "", "quantity": how many units of that single listing (an integer; 1 if the request gives no number; a pair or set sold as a single listing is 1), "musts": the firm requirements on this product as a list, each written "property: constraint" carrying the precise value, limit, exclusion or closed list. Call the property by the word the request itself uses ("meant for a lakeside occasion" becomes "occasion: lakeside", "a 12-inch blade" becomes "blade length: 12 inch"), never replacing it with a nearby property. When a must takes its value from another listing, write it as ONE must that holds the full description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): whatever describes the other listing (its shop, its kind, a following "with <property> <value> and <property> <value>") remains inside that single must and never becomes a must of this product}. Price, budget, stock and quantity never go into musts. Treat the request\'s surrounding story (who will use the product, where it is going to stand, what failed last time, the occasion that prompted it) as context rather than a must, except where it spells out a property\'s value or limit directly. Words that form part of the product\'s own name (such as a season or an audience mentioned together with the product) are never musts either; they go in "name". But when the request states a property of the product in so many words ("the type of top is a tunik", "the style has to be wrap"), that is a must on that property ("type: tunik") with the plain product as the name ("top"), even if the value could also serve as a product name. When the request turns a use or occasion into a property of the product itself ("blazers for business wear", "a dress made for parties", "the occasion has to be party"), that IS a must on the property ("occasion: business"), except when the store\'s category label names it already. By contrast, the event, person or setting the purchase is for ("to wear at his wedding", "for a colleague"), remain context; but a fit, cut or feel the product is required to have (a loose or straight fit, a soft or firm feel, a slim cut) IS a must on that property, named the way the request names it. If the request names a product in a single phrase linked by "and" (a single "I want ..." with a single quantity, frequently the name of one store department), that is ONE item; split it into several items only if the request assigns a separate quantity or separate requirements to each product.\n"budget": the highest total the shopper is willing to pay, as a number in the unit the request uses, or null.\n"budget_unit": the name of that unit spelled out (such as "cents"), or null.\n"budget_scope": "total" if the limit applies to the entire order, "each" if it applies per item, null if no limit exists.\n"in_stock_only": true, unless the request plainly allows unavailable items.\n"pick": "cheapest" if the request wants the lowest price or the smallest total, "most" or "least" if it wants the maximum or minimum of some measure, otherwise "any". Comments about money being tight do not count as a pick rule.\n"pick_measure": the measure being maximised or minimised under "most"/"least", otherwise "".\n"messages": the number of messages the shopper lets the agent send (integer), or null if unstated.\n"steps": the number of steps, moves or tool calls permitted by the request (integer), or null.\n"lookups": a list of those musts (copied verbatim) whose value has to be read beforehand from another listing that the request refers to, otherwise [].\n"rules": a list of every instruction in the request about the way to work or the way to end, each one in 12 words or fewer.\n"wishes": a list of what the shopper would merely like to have.\n"ask_first": a list of the points the request expressly says to verify with the shopper before making a choice (empty if there are none).\n"process": the steps, beyond locating and buying the correct items, that the request obliges the agent to carry out (obtaining the shopper\'s approval, informing the shopper of something or showing them something, verifying something with them, reading something beforehand, using a specific tool), as a list of {"step": the action required, in 15 words or fewer, "when": "choose" (before choosing the products), "order" (immediately before checkout) or "end" (immediately before ending without a purchase), "messages": the number of messages to the shopper it needs (0 if none)}; [] if the request asks for no step of this kind. These are not steps: item requirements, the budget, stock, the message and step limits, and the way to end when nothing fits.\n"quiet_decline": true if the request forbids suggesting or naming alternatives in case nothing fits, false if it asks for alternatives, null if it is silent on this.\n"fits": a list with one {"item": item number, "with": item number} per requested item which, according to the request, has to work with, fit or suit ANOTHER of the requested items (whichever one the agent picks); [] if there is none.\n"roles": an object giving, for each job, the exact name of the tool that does it, or null if none does: "search", "open" (displays the attributes or properties of a single listing; if there are two tools that each show a single listing, take the tool whose description promises specifications, attributes or properties), "offer" (displays a single listing\'s current offer, e.g. its price and stock, when this tool is not the same as "open"), "compare" (shows several listings in full), "ask" (messages the shopper), "cart" (modifies the cart), "cart_view" (displays the cart), "buy" (performs checkout), "refuse" (ends without a purchase).\n"buy_args": the constant arguments with which the "buy" tool performs checkout (if the tool takes a mode argument, that mode), otherwise {}.\n"refuse_args": the constant arguments with which the "refuse" tool ends without a purchase, otherwise {}.\n'
ASK_FORM = 'Your task is to pick what a shopping agent will ask the shopper. Every shopper has a handful of firm preferences left out of the written request, each concerning a single property of a single requested product. A preference comes out only if the question names precisely that property; for a property the shopper has no opinion on the reply is "don\'t mind", and for one the shopper never considers the reply is puzzled, so each property named uses up part of a message.\n\nYour input is the request, the items together with what the request has already settled for each of them, and a numbered list of the attribute keys observed on opened listings in each item\'s category, giving for each key the count of those listings that carry it plus example values. A preference that was not written down can concern a property that none of the listed keys names.\n\nPick properties following this order:\n0. Anything the request itself tells you to verify with the shopper (shown under ASK FIRST if present), each one named by the property it concerns.\n1. Hinted properties. Frequently the story in the request hints at an unwritten preference without stating its value: a sentence mentioning something the shopper is unable to judge or hands over to you, a person whose taste is decisive, how the product looks or the setting it fits into, whoever will use or wear it, or what it contains. For every sentence like that, name the single property it hints at and the item it concerns, following the order in which the story presents them.\nIf one of that item\'s listed keys matches the hint, use that key to name the hinted property (a hint about the wearer: the gender or audience key; a hint about appearance: whichever listed colour, pattern or finish key the majority of the item\'s listings carry; a hint about the place it goes or the space it has to fit: the listed key for a dimension that place limits, such as its width, height or depth); a property carried by none of the item\'s listings is ranked after the three keys that the item\'s listings carry most.\n2. After those, the listed keys in ranked order: prefer the properties a buyer of this kind of product decides on first, judged by what the product is for and how it gets used (its kind or type, its fit, cut or form, its size, capacity, dimensions or other figures, its material, its main feature) every bit as much as by how it looks, and keys that most of the item\'s listings carry, whether the values seen so far vary or repeat. A key holding a figure ranks the same as any other key. Spend at most one of an item\'s picks on a property about its looks alone (colour, pattern, print, style, finish) unless the request\'s story is about how it looks. When five or more listings have been read, steer clear of keys present on just one or two of them.\nLeave out: a property the request has already settled for that item (settled means some requirement names that property, whereas a key whose values merely echo a required value, for instance a pattern key displaying the required decoration, is NOT settled); another name for a property you have already picked; a key whose every value lists several purchase options at once; brand, seller or shop, price, delivery, warranty, identifiers, model names or codes and similar logistics. Pack sizes, counts and per-pack quantities count as ordinary properties, while any key marked [not the stated ...] counts as a different property from that requirement, no matter which words its name has in common with it. Each item is ranked separately: hinted properties for it first, followed by its listed keys.\n"label" holds the plain name of the property: for a listed key, every word of the key spelled out plainly ("lid colour" for lidColour, "rim finish" for rim_finish; never cut down to "colour" or "finish"); just the name of the property, without example values, brackets or a second property.\nFor each item pick no more than COUNT, the best one first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story\'s words it comes from, or an empty string"}]}, listing each item\'s choices in ranked order'
LISTEN_FORM = 'Some questions went from a shopping agent to a shopper. Go through every reply, then return ONE JSON object and nothing else:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true if the value must be taken from a different listing, otherwise false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [question numbers whose reply objects that the message covered too many things or asks for just one question; leave out replies that merely say they did not understand it or ask for simpler, plainer or shorter words], "budget": the new overall limit as a number, or null}\n- "needs": each firm preference a reply states, including its precise value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Label the need with the property that the reply itself mentions, which can be different from the property the question was about; when several replies concern one property, together they form a single need. Reproduce the value verbatim as the reply gives it, retaining its nouns and units ("at least 3 shelves", not "at least 3"). Verbatim means the value itself, without the turns of phrase around it ("For me, pack type means single, plain and simple" gives "pack type: single"; "fruit type matching pineapple" gives "fruit type: pineapple").\n- A reply may supply the value by referring to some other listing (a value like, unlike or close to the one found on a product in some shop, with that product identified through a few of its properties): treat this as ONE need on the property the reply mentions, with "points": true. Its value is the reply\'s own wording reproduced exactly, starting at the comparison words and running to the end of that description, plus anything it says about a listing that the page may name; do not ever trim or paraphrase it. Whatever properties describe the other product do not count as needs of this item.\n- A reply saying that anything on a property other than one value gets a no ("anything on X other than Y gets a polite no thanks") wants that value and only it: need "X: Y". A plain exclusion ("not wicker", "anything but leather" said as a dislike of leather) stays an exclusion.\n- "free": for each question, the property the shopper said they do not care about.\n- A value named in homely or indirect phrasing is still a stated value; but a remark about the shopper\'s day, meal, trip or mood inside a reply that says they do not mind ("as long as I\'m home for the Sunday roast") states no need: that question belongs under "free".'
NOTICE_FORM = 'A note appeared from the store while a shopping agent was working (its source may be the shopper or the store). Return ONE JSON object and nothing else:\n{"kind": "change" if the note alters the task (for instance the budget, the requirements, a price, whether something is available, or the order being refused), "error" if all it does is report that a call was malformed or rejected, "other" for everything else, "budget": the new overall limit as a number in the unit of the request, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [ids of listings the note says are unavailable], "sold_out": true if it reports a chosen listing as gone, sold out or not available any more (even when it does not name it), "price_moved": true if it reports that a price changed, "summary": "one short sentence"}\nWhen a requirement is replaced, record the old one as a drop and the new one as an add. Every change the note contains must be reported.'
VERIFY_FORM = 'You are the final gate before a shopping agent completes checkout. Using the records shown, judge whether this cart is exactly correct.\nHow to read records: a requirement is satisfied when the attribute the listing has for that property gives the value (key spelling does not matter, a single element of a list suffices, case, singular/plural and plain synonyms are accepted, a value that contains the words of the required value is accepted, such as "deep teal" for teal, units are converted, limits include their edge, and a range has to fall entirely inside); an attribute keyed by the very property the requirement concerns is decisive, and if it gives something different the listing fails (when several keys name the property, the most specific key decides (one that also names the product or its part, such as "clock shape" over a bare "shape"; spelling with spaces, underscores or dots makes no difference), and two equally specific keys (the same key spelled two ways, such as "dietary needs" and "dietary_needs") count together: the requirement is met when either of them gives the value); a key naming another property, even a related one (the colour of a handle is not the colour of the body), settles nothing; if the listing has no attribute covering that property, the wanted value appearing under a different attribute, the title or the description is accepted (a figure of a different kind does not count — weight, for instance, is not capacity — and neither does a lone number whose text leaves unsaid what it measures); a requirement stated by the category label itself is satisfied by that category; the category label has to match the requested item in its entirety; a rule requiring a value unlike another listing\'s is treated as satisfied unless the records show that very value. Each requested item must be its own cart line at the requested quantity, the cart may hold nothing else, nothing may be shown as out of stock, and price times quantity summed across all lines must stay within the budget. For any listing, its most recent record wins.\nWhen cheapest, most or least is the pick rule, check the other listings shown as well: should any of them plainly meet every requirement and beat the cart\'s choice under that rule, the cart is wrong.\nReturn ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [ids of listings that should take the place of a cart line, best first]}. Answer ok=false solely when the records plainly show a problem or when no record covers a requirement. A line that the requirements list marks as "decides nothing" is the shopper\'s aside rather than a requirement: never fail a cart over it. When a cart listing\'s own notes say a required property is to be taken from another listing, its printed value for that property proves nothing.'
MATCH_RULES = 'a listing\'s attribute for a property is decisive when its key names that very property (the most specific such key decides, and two equally specific keys, such as one key spelled two ways, count together, so either of them giving the value is enough; key spelling does not matter, a single element of a list suffices, synonyms and unit conversions are accepted, a value that contains the words of the required value is accepted, such as "deep teal" for teal, limits include their edge); a key naming another property, even a related one (the colour of a handle is not the colour of the body), settles nothing; if none of the attributes names that property, the wanted value appearing under a different attribute, the title or the description is accepted (a lone number that leaves unsaid what it measures is worth nothing); a requirement stated by the category label itself is satisfied by that category; the category label has to match the item; a rule requiring a value unlike another listing\'s is treated as satisfied unless the records show that very value; listings that are out of stock or that exceed the budget are not a fit.'
REVIEW_FORM = 'A shopping agent intends to stop without buying, believing that nothing fits. Compare the listing records shown with the requirements, applying these reading rules: ' + MATCH_RULES + '\nRecords marked ROW appeared only as a row of search results: their page (the attributes) is still unread, so on any property not visible in the row they can neither pass nor fail.\nReturn ONE JSON object and nothing else: {"fits": {"item number": [ids of listings that seem to satisfy every requirement, cheapest first, 4 at most]}, "open": [ids of ROW listings whose kind was requested and that sit within the budget, whose page might reveal that they fit, most promising first, 6 at most], "closest": "the id of the listing that breaks the fewest requirements (going over the budget counts as one broken requirement; among equals the cheapest, in stock), or null", "reason": "one sentence"}'
SPLIT_FORM = 'A shopper\'s request has been turned into firm requirements on a product. Certain requirements take their value from a different listing the request refers to, and the words identifying that other listing (its shop, its kind, and the properties listed after it with "with ...") are at times wrongly read as requirements on the product being purchased. Based on the wording of the request, state which of the OTHER REQUIREMENTS below merely describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements, copied exactly; empty if every one of them is a genuine requirement on the product]}'
CHAIN_FORM = 'A shopping agent is close to checking out. Certain requirements do not state their value but take it from another listing, and the page of a listing may point the reader on to yet another listing for that property. The order is valid only if the agent has fully opened every listing on each such chain. Trace the chain of each requirement starting from that requirement\'s own description, one link at a time, through the words the pages themselves use, drawing on the records listed below (OPENED means the agent opened that page; ROW means it was only seen as a row in search results and its page is unread).\nReturn ONE JSON object and nothing else: {"chains": [{"need": the requirement, copied exactly, "links": [{"id": the listing id if a record below is that link, otherwise "", "about": the way the requirement or the preceding page describes that listing, "find": "for a link not found: search words (its shop and the values the description gives)"}], "done": true if the final link is an OPENED page that itself states the property\'s value (this holds too when the requirement wants a value that differs from or is unlike that one: the value on the page is what it is compared with), otherwise false, "value": the property\'s value as the final page states it, or ""}]}\nA record counts as a link only if it fits the link\'s entire description: the shop or brand, the product kind AND each property value that the description names (for instance "a 10 pieces multipack" excludes that shop\'s listing that has 10 capsules and no multipack); if no record fits all of it, the link has not been found yet, even when that shop sells other listings of the same kind. For such a link give "find": the shop name plus the values from the description, as a few search words.\nGive the links of each chain in order, halting at the earliest link that has not been found or whose page is not OPENED, because its words are not yet known.'
STEP_FORM = 'A shopping agent is getting ready to STAGE. Along the way, the shopper\'s request calls for the steps listed below. Using the numbered RECORD of the agent\'s actions, decide for each step that falls due before the agent can STAGE whether it has been done for the cart in its current state. A step is done only if some RECORD line shows it happening and, for steps concerning the order or the cart, that line is later than the last change to the cart and its reply approves it; asking about a product\'s properties does not perform a step.\nIf a step is still outstanding, give the single store call that performs it next: a tool from TOOLS together with its arguments. Messages to the shopper are sent through the messaging tool: a single short sentence or question concerning that step alone, in plain language, mentioning no product properties or values that the step does not require. Give no call when all due steps are done or when the remaining messages or steps are unable to perform the missing one.\nReturn ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [numbers of the RECORD lines showing it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": whatever a reply wants changed before proceeding, or "", "needs": [{"item": item number, "need": "property: constraint"} for every firm requirement stated in such a reply]}'
RESTOCK_FORM = 'A listing in the cart of a shopping agent has just sold out. Among the other listing records shown, find replacements for that cart line, applying these reading rules: ' + MATCH_RULES + '\nReturn ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price in number form as given in the record} for each listing that seems to satisfy every requirement of the item on the sold-out line, 6 at most], "reason": "one sentence"}'
LAYOUT_FORM = 'The result below came back from a store tool. Refer to its parts by the precise keys found in the result and in the tool\'s parameters. Return ONE JSON object and nothing else: {"rows": the key of the listings list, "count": the key giving the total number of listings that matched (not the number displayed), or null, "next": the key whose value asks for the next page, or null, "next_arg": the tool parameter that accepts that value, or null, "price": the key within a row for the listing\'s price, or null, "stock": the key within a row that tells whether the listing is buyable right now, or null}'
BACKUP_KEYS = 3
VOTER_WEIGHTS = (1.0, 1.0, 0.8)
VOTER_DEPTH = 4
DEL_WORDS = '\\b(?:del|rm|exclud\\w*|un[a-z]+|pull\\w*|remov\\w*|delet\\w*|drop\\w*|clear\\w*|take\\w*|releas\\w*|withdraw\\w*|discard\\w*|unhold|minus|decreas\\w*|cancel\\w*|subtract\\w*|out|off|less)\\b'
ADD_WORDS = '\\b(?:add\\w*|hold\\w*|place\\w*|put|includ\\w*|keep\\w*|insert\\w*|stash\\w*|reserv\\w*|plus|increas\\w*|in|more|set|buy|cart)\\b'
QTY_WORDS = '\\b(?:qty|quant\\w*|count\\w*|units?|pieces?|number|amount|how many|total|needed|wanted)\\b'
INDEX_WORDS = '\\b(?:line|index|position|slot|row|offset|page)\\b'
STORY_FORM = 'A shopper wrote the request below. Shoppers keep a few firm preferences they did not write down, each on ONE property of ONE requested product, and the request\'s story usually hints at them: who the product is for, where it will stand or be used, what went wrong before, how it should look or feel, what it holds or works with. For each item, rank up to COUNT properties by how likely the shopper holds an unwritten firm preference on it. Name each by one of the item\'s listed keys when one fits (all of its words, plain spelling), else by a plain property name. Skip properties the request already settles, identifiers, model names, warranty, price and stock. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "the story words it comes from, or an empty string"}]}, each item\'s picks most likely first.'
DIVIDE_FORM = 'A shopping agent may send only a few one-property questions to the shopper. Below are the request, the items with what the request already settles, and the attribute keys seen on listings of each item with example values. For each item, rank up to COUNT properties whose answer would most change WHICH listing the shopper accepts: properties whose values differ among listings that already meet the stated requirements, and that a buyer of that kind of product decides on. Name each by one of the listed keys (all of its words, plain spelling). Skip properties the request already settles, identifiers, model names, warranty, logistics, and keys whose values bundle several options. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": ""}]}, each item\'s picks best first.'
BACKUP_SHARE = 0.0
BACKUP_GAP = 0.34
NON_PROPERTY_RX = '\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b'
BUNDLE_KEY_RX = '^(?!.*\\b(?:types?|styles?|kinds?|shapes?)\\b).*\\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\\w*|accessor\\w*|extras?|misc\\w*)\\b'
READS_PER_ITEM = 6
WALK_PAGE_CAP = 13
WALK_OPEN_CAP = 4
PACK_SIZE = 10
PACK_LENGTH = 360
PROBE_FLOOR = 3
PROBE_WIDTH = 3
MODEL_CODE_RX = '[A-Z][A-Za-z]{0,5}[ -]?\\d{3,6}'
PART_ID_RX = '[A-Z]{1,4}\\d{3,6}'

def code_tokens(words_out: Any) -> list[str]:
    return [re.sub('[\\s-]+', '', c).strip().lower() for c in re.findall('\\b' + MODEL_CODE_RX + '\\b', str(words_out or ''))]

def fit_notes_single(words_out: Any) -> dict[str, Any]:
    words_out = str(words_out or '')
    own_map = re.search('[\\[(]\\s*(?:part\\s+)?(' + PART_ID_RX + ')\\s*[\\])]', words_out, re.I)
    model_codes: list[str] = []
    for rx_hit in re.finditer('\\bmodel\\s*(?:(?:no\\.?|number|#)\\s*[:.]?|[:.])\\s*(' + MODEL_CODE_RX + ')\\b', words_out, re.I):
        model_codes += code_tokens(rx_hit.group(1))
    for rx_hit in re.finditer('\\b(?:also sold as|also known as|sold also as)\\s+([^.;]+)', words_out, re.I):
        model_codes += code_tokens(rx_hit.group(1))
    works: list[str] = []
    for rx_hit in re.finditer('\\b(?<!not )(?<!never )(?:works with|compatible with|fits)\\s*:?\\s+([^.;]+)', words_out, re.I):
        if not re.search('\\bpart\\s+' + PART_ID_RX + '\\s+also\\s+$', words_out[max(0, rx_hit.start() - 40):rx_hit.start()], re.I):
            works += code_tokens(re.split('\\b(?:except|but not|apart from|excluding)\\b', rx_hit.group(1), flags=re.I)[0])
    also = [(rx_hit.group(1).upper(), code_tokens(rx_hit.group(2))) for rx_hit in re.finditer('\\bpart\\s+(' + PART_ID_RX + ')\\s+also fits\\s*:?\\s*([^.;]+)', words_out, re.I)]
    series = []
    for rx_hit in re.finditer('(?<!not )(?<!never )\\bevery\\s+([A-Za-z]{1,6})[ -]?(\\d{1,5})(x+)\\s+models?\\b(?:\\s*,?\\s*(?:except|but not|apart from|excluding)\\s+([^.;]+))?', words_out, re.I):
        series.append((rx_hit.group(1).lower() + rx_hit.group(2), len(rx_hit.group(3)), code_tokens(rx_hit.group(4) or '')))
    more = re.search('\\b(?:more|details|see|read)\\s+(?:on|at|under|about)?\\s*(?:the\\s+)?part\\s+(' + PART_ID_RX + ')', words_out, re.I)
    return {'own': own_map.group(1).upper() if own_map else '', 'codes': model_codes, 'works': works, 'also': also, 'more': more.group(1).upper() if more else '', 'series': series}

def series_covers(model_code: str, series: list[tuple[str, int, list[str]]]) -> bool:
    return any((model_code not in skip and re.fullmatch(re.escape(head_word) + '\\d{' + str(width) + '}', model_code) for head_word, width, skip in series))
MODEL_CODE_WIDE = '[A-Z][A-Za-z]{0,5}[- ]?\\d{3,6}'
PART_ID_WIDE = '[A-Z]{1,5}\\d{0,3}-?\\d{2,6}'

def code_key(model_code: Any) -> str:
    return re.sub('[\\s\\-]+', '', str(model_code or '')).lower()

def code_tokens_wide(words_out: Any) -> list[str]:
    return [code_key(c) for c in re.findall('\\b' + MODEL_CODE_WIDE + '\\b', str(words_out or ''))]

def fit_notes(words_out: Any) -> dict[str, Any]:
    words_out = str(words_out or '')
    own_map = re.search('[\\[(]\\s*(?:part\\s+)?(' + PART_ID_WIDE + ')\\s*[\\])]', words_out, re.I)
    model_codes: list[str] = []
    for rx_hit in re.finditer('\\bmodel\\s*(?:ref(?:erence)?\\.?|no\\.?|number|#|:)\\s*[:.]?\\s*(' + MODEL_CODE_WIDE + ')\\b', words_out, re.I):
        model_codes += code_tokens_wide(rx_hit.group(1))
    for rx_hit in re.finditer('\\b(?:also sold as|also known as|sold also as)\\s+([^.;]+)', words_out, re.I):
        model_codes += code_tokens_wide(rx_hit.group(1))
    works: list[str] = []
    excl: list[str] = []
    fams: list[str] = []
    for rx_hit in re.finditer('\\b(?<!not )(?<!never )(?:goes with|works with|compatible with|for use with|fits)\\s*:?\\s+([^.;]+)', words_out, re.I):
        if re.search('\\bpart\\s+' + PART_ID_WIDE + '\\s+also\\s+$', words_out[max(0, rx_hit.start() - 40):rx_hit.start()], re.I):
            continue
        head_word, *tail = re.split('\\b(?:except|excluding|but not|other than)\\b', rx_hit.group(1), maxsplit=1, flags=re.I)
        if tail:
            excl += code_tokens_wide(tail[0])
        for fam in re.finditer('\\b(?:every|all|any)\\s+([A-Z][A-Za-z]{0,5}[- ]?\\d*)([xX*#]{1,4})\\b', head_word):
            fams.append('^' + re.escape(code_key(fam.group(1))) + '\\d{' + str(len(fam.group(2))) + '}$')
        series = re.search('\\bpart\\s+series\\s+([A-Z0-9 ,\\-]+)', head_word, re.I)
        if series:
            fams += ['^' + re.escape(code_key(s)) for s in re.split('\\s*,\\s*|\\s+and\\s+', series.group(1)) if re.fullmatch('[A-Za-z]{1,5}-?\\d{0,3}', s.strip() or '-')]
        works += [c for c in code_tokens_wide(re.sub('\\b(?:every|all|any)\\s+[A-Z][A-Za-z]{0,5}[- ]?\\d*[xX*#]{1,4}\\b', ' ', head_word)) if c]
    also = [(code_key(rx_hit.group(1)), code_tokens_wide(rx_hit.group(2))) for rx_hit in re.finditer('\\bpart\\s+(' + PART_ID_WIDE + ')\\s+also fits\\s*:?\\s*([^.;]+)', words_out, re.I)]
    more = [code_key(rx_hit.group(1)) for rx_hit in re.finditer('\\b(?:more|details|see|read)\\s+(?:on|at|under|about)?\\s*(?:the\\s+)?part\\s+(' + PART_ID_WIDE + ')', words_out, re.I)]
    return {'own': code_key(own_map.group(1)) if own_map else '', 'codes': list(dict.fromkeys(model_codes)), 'works': list(dict.fromkeys(works)), 'excl': list(dict.fromkeys(excl)), 'fams': list(dict.fromkeys(fams)), 'also': also, 'more': list(dict.fromkeys(more))}

def as_text(val: Any) -> str:
    try:
        return json.dumps(val, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        return str(val)

def clipped_copy(val: Any, ceiling: int, level: int=0) -> Any:
    if level > 7:
        return '...'
    if isinstance(val, str):
        return val if len(val) <= ceiling else val[:ceiling] + '...'
    if isinstance(val, dict):
        return {k: clipped_copy(v, ceiling, level + 1) for k, v in val.items()}
    if isinstance(val, list):
        return [clipped_copy(v, ceiling, level + 1) for v in val]
    return val

def json_object(words_out: Any) -> dict[str, Any] | None:
    if not isinstance(words_out, str):
        return None
    words_out = words_out.strip()
    fence = re.match('^```[A-Za-z]*\\s*(.*?)\\s*```$', words_out, re.S)
    if fence:
        words_out = fence.group(1)
    try:
        json_got = json.loads(words_out)
        return json_got if isinstance(json_got, dict) else None
    except ValueError:
        pass
    left, right = (words_out.find('{'), words_out.rfind('}'))
    if 0 <= left < right:
        try:
            json_got = json.loads(words_out[left:right + 1])
            return json_got if isinstance(json_got, dict) else None
        except ValueError:
            return None
    return None

def plain_words(key: str) -> str:
    key = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(key))
    return re.sub('[\\s_.\\-]+', ' ', key).strip().lower()

def word_roots(words_out: str) -> frozenset[str]:
    return frozenset((w[:-1] if len(w) > 3 and w.endswith('s') else w for w in plain_words(words_out).split()))

def root_set(words_out: str) -> set[str]:
    acc = set()
    for w in plain_words(words_out).split():
        w = w[:-2] if re.search('(?:ch|sh|ss|x|z)es$', w) else w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w
        if len(w) >= 3 and w not in ('and', 'for', 'the', 'with'):
            acc.add(w)
    return acc

def name_covers(prop_words: frozenset[str], key: frozenset[str]) -> bool:
    if not prop_words or not key:
        return False
    return prop_words == key or prop_words <= key or (len(key) >= 2 and key <= prop_words)

def stem_word(word: str) -> str:
    w = word.lower()
    if len(w) >= 5 and w.endswith('ies'):
        return w[:-3] + 'y'
    if len(w) >= 6 and w.endswith('ing'):
        w = w[:-3]
    elif len(w) >= 5 and w.endswith('ed'):
        w = w[:-2]
    elif len(w) >= 5 and w.endswith('es') and (w[-3] in 'sxz') or (len(w) >= 6 and w.endswith(('ches', 'shes'))):
        w = w[:-2]
    elif len(w) >= 4 and w.endswith('s') and (not w.endswith('ss')):
        w = w[:-1]
    return w[:-1] if len(w) >= 5 and w[-1] == w[-2] else w

def value_stems(val: str) -> set[str]:
    return {stem_word(w) for w in re.findall('[a-z0-9]+', val.lower()) if len(w) >= 3 and w not in ('and', 'the', 'for', 'with', 'any', 'all')}

def text_stems(words_out: str) -> set[str]:
    return {stem_word(w) for w in re.findall('[a-z0-9]+', words_out.lower()) if len(w) >= 3}

def names_value(val: str, words_out: str) -> bool:
    want = value_stems(val)
    if not want:
        return False
    held_now = text_stems(words_out)
    for w in want:
        if w in held_now:
            continue
        if any((min(len(w), len(h)) >= 4 and abs(len(w) - len(h)) <= 2 and (h.startswith(w) or w.startswith(h)) for h in held_now)):
            continue
        if len(w) >= 5 and any((len(h) == len(w) and h[:-1] == w[:-1] for h in held_now)):
            continue
        return False
    return True

def schema_kinds(spec_doc: dict[str, Any]) -> list[str]:
    kind_txt = spec_doc.get('type')
    if isinstance(kind_txt, str):
        return [kind_txt]
    return [k for k in kind_txt or [] if isinstance(k, str)]

def value_fits_kind(val: Any, kind_txt: str) -> bool:
    if kind_txt == 'string':
        return isinstance(val, str)
    if kind_txt == 'integer':
        return isinstance(val, int) and (not isinstance(val, bool))
    if kind_txt == 'number':
        return isinstance(val, (int, float)) and (not isinstance(val, bool))
    if kind_txt == 'boolean':
        return isinstance(val, bool)
    if kind_txt == 'array':
        return isinstance(val, list)
    if kind_txt == 'object':
        return isinstance(val, dict)
    if kind_txt == 'null':
        return val is None
    return True

def conform(val: Any, spec_doc: Any, where: str, fault_list: list[str]) -> Any:
    if not isinstance(spec_doc, dict):
        return val
    kinds = schema_kinds(spec_doc)
    if kinds and (not any((value_fits_kind(val, k) for k in kinds))):
        for kind_txt in kinds:
            if kind_txt == 'integer' and isinstance(val, float) and val.is_integer():
                val = int(val)
                break
            if kind_txt in ('integer', 'number') and isinstance(val, str) and re.fullmatch('\\s*-?\\d+(\\.\\d+)?\\s*', val):
                slot_no = float(val)
                val = int(slot_no) if kind_txt == 'integer' or slot_no.is_integer() else slot_no
                break
            if kind_txt == 'string' and isinstance(val, (int, float)) and (not isinstance(val, bool)):
                val = str(val)
                break
            if kind_txt == 'boolean' and isinstance(val, str) and (val.strip().lower() in ('true', 'false')):
                val = val.strip().lower() == 'true'
                break
            if kind_txt == 'array' and (not isinstance(val, list)) and (val is not None):
                val = [val]
                break
        if not any((value_fits_kind(val, k) for k in kinds)):
            fault_list.append(f"{where} should be {'/'.join(kinds)}")
            return val
    opts = spec_doc.get('enum')
    if isinstance(opts, list) and val not in opts:
        if isinstance(val, str):
            low = val.strip().lower()
            match = [o for o in opts if isinstance(o, str) and o.lower() == low]
            if match:
                val = match[0]
        if val not in opts:
            fault_list.append(f'{where}: allowed values are {opts}')
    if isinstance(val, str) and isinstance(spec_doc.get('maxLength'), int) and (len(val) > spec_doc['maxLength']):
        val = val[:spec_doc['maxLength']]
    if isinstance(val, str) and isinstance(spec_doc.get('minLength'), int) and (len(val) < spec_doc['minLength']):
        fault_list.append(f'{where} is too short')
    if isinstance(val, (int, float)) and (not isinstance(val, bool)):
        if isinstance(spec_doc.get('minimum'), (int, float)) and val < spec_doc['minimum']:
            fault_list.append(f"{where} is below {spec_doc['minimum']}")
        if isinstance(spec_doc.get('maximum'), (int, float)) and val > spec_doc['maximum']:
            fault_list.append(f"{where} is above {spec_doc['maximum']}")
    if isinstance(val, list):
        if isinstance(spec_doc.get('maxItems'), int) and len(val) > spec_doc['maxItems']:
            fault_list.append(f"{where} allows at most {spec_doc['maxItems']} entries")
        if isinstance(spec_doc.get('minItems'), int) and len(val) < spec_doc['minItems']:
            fault_list.append(f"{where} needs at least {spec_doc['minItems']} entries")
        if isinstance(spec_doc.get('items'), dict):
            val = [conform(v, spec_doc['items'], f'{where}[{i}]', fault_list) for i, v in enumerate(val)]
    if isinstance(val, dict):
        prop_specs = spec_doc.get('properties') if isinstance(spec_doc.get('properties'), dict) else {}
        if spec_doc.get('additionalProperties') is False:
            for tool_name in [n for n in val if n not in prop_specs]:
                fault_list.append(f'{where}.{tool_name} is not an allowed argument')
        for tool_name in spec_doc.get('required') or []:
            if tool_name not in val:
                fault_list.append(f'{where}.{tool_name} is required')
        val = {k: conform(v, prop_specs.get(k), f'{where}.{k}', fault_list) if k in prop_specs else v for k, v in val.items()}
    return val

def as_list(val: Any) -> list[Any]:
    if isinstance(val, list):
        return val
    if isinstance(val, (str, dict)) and val:
        return [val]
    return []

def as_count(val: Any) -> int | None:
    if isinstance(val, bool):
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float) and val.is_integer():
        return int(val)
    if isinstance(val, str) and re.fullmatch('\\s*\\d+(\\.0+)?\\s*', val):
        return int(float(val))
    return None

def looks_like_id(val: Any, named_ids: bool=False) -> bool:
    if not isinstance(val, str) or not 3 <= len(val.strip()) <= 64 or ' ' in val.strip():
        return False
    return named_ids or bool(re.search('\\d', val))

def guess_id_key(row_list: list[dict[str, Any]]) -> str | None:
    top_pick = None
    for key in row_list[0]:
        vals = [row_doc.get(key) for row_doc in row_list]
        if not all((looks_like_id(v, 'id' in str(key).lower()) for v in vals)):
            continue
        vals = [v.strip() for v in vals]
        if len(row_list) > 1 and len(set(vals)) < len(vals):
            continue
        points = ('id' in str(key).lower(), sum((len(v) for v in vals)) / len(vals))
        if top_pick is None or points > top_pick[0]:
            top_pick = (points, key)
    return top_pick[1] if top_pick else None

def row_contains(obj: Any, pid: str, level: int=0) -> bool:
    if level > 5:
        return False
    if isinstance(obj, list):
        return any((isinstance(v, dict) and pid in [str(x).strip() for x in v.values() if isinstance(x, str)] for v in obj)) or any((row_contains(v, pid, level + 1) for v in obj))
    if isinstance(obj, dict):
        return any((row_contains(v, pid, level + 1) for v in obj.values()))
    return False
FIELD_RX = re.compile('^([a-z][a-z0-9_]{0,40}):\\s?(.*)$', re.S)
FIELD_RX_LOOSE = re.compile('^([A-Za-z][A-Za-z0-9_]{0,40}):\\s?(.*)$', re.S)
NUMBER_RX = re.compile('\\s*-?\\d+(?:\\.\\d+)?\\s*')

def plain_scalar(words_out: str) -> Any:
    low = words_out.strip().lower()
    if low in ('true', 'false'):
        return low == 'true'
    if low in ('none', 'null'):
        return None
    if NUMBER_RX.fullmatch(words_out):
        slot_no = float(words_out)
        return int(slot_no) if slot_no.is_integer() and '.' not in words_out else slot_no
    return words_out.strip()

def pairs_from_text(words_out: str) -> dict[str, Any] | None:
    if words_out.count('=') < 2:
        return None
    acc: dict[str, Any] = {}
    last = None
    for part in re.split(';\\s+', words_out.strip().rstrip(';')):
        key, eq, val = part.partition('=')
        if eq and 0 < len(key.strip()) <= 60 and ('|' not in key):
            last = key.strip()
            acc[last] = val.strip()
        elif last is not None:
            acc[last] = f'{acc[last]}; {part}'
        else:
            return None
    return acc if len(acc) >= 2 else None
FIELD_SEPARATORS = (' | ', '\n', '; ', ' || ', '\t')

def record_from_text(words_out: str, need_id: bool=False) -> dict[str, Any] | None:
    if not isinstance(words_out, str):
        return None
    for sep in FIELD_SEPARATORS:
        if words_out.count(sep) < 2:
            continue
        rec_doc = split_record(words_out.strip().split(sep), sep)
        if rec_doc is not None and (not need_id or any((isinstance(v, str) and looks_like_id(v) for v in rec_doc.values()))):
            return rec_doc
    return None

def split_record(pieces: list[str], sep: str) -> dict[str, Any] | None:
    for field in (FIELD_RX, FIELD_RX_LOOSE):
        first_val = field.match(pieces[0].strip())
        if not first_val:
            continue
        acc: dict[str, str] = {first_val.group(1): first_val.group(2)}
        last = first_val.group(1)
        for part in pieces[1:]:
            m = field.match(part.strip())
            if m and m.group(1) not in acc:
                last = m.group(1)
                acc[last] = m.group(2)
            else:
                acc[last] = acc[last] + sep + part
        if len(acc) >= 3:
            rec_doc: dict[str, Any] = {}
            for key, val in acc.items():
                rec_doc[key] = pairs_from_text(val) or plain_scalar(val)
            return rec_doc
    return None

def table_records(obj: dict[str, Any]) -> dict[str, Any] | None:
    heads = next((v for v in obj.values() if isinstance(v, list) and len(v) >= 3 and all((isinstance(h, str) and 0 < len(h) <= 40 for h in v))), None)
    if heads is None:
        return None
    for key, val in obj.items():
        if isinstance(val, list) and val and all((isinstance(r, list) and len(r) == len(heads) for r in val)):
            return {**{k: v for k, v in obj.items() if v is not heads and k != key}, key: [dict(zip(heads, r)) for r in val]}
    return None

def as_records(obj: Any, level: int=0, inside: bool=False) -> Any:
    if level > 8:
        return obj
    if isinstance(obj, dict):
        table = table_records(obj)
        if table is not None:
            obj = table
        rec_doc = len(obj) >= 3 and (not any((isinstance(v, dict) for v in obj.values())))
        return {k: as_records(v, level + 1, rec_doc) for k, v in obj.items()}
    if isinstance(obj, list):
        return [as_records(v, level + 1) for v in obj]
    if isinstance(obj, str):
        words_out = obj.strip()
        if words_out[:1] in '{[' and words_out[-1:] in '}]':
            try:
                return as_records(json.loads(words_out), level + 1)
            except ValueError:
                pass
        rec_doc = record_from_text(obj, need_id=not inside)
        if rec_doc is not None:
            return rec_doc
        if inside:
            pairs = pairs_from_text(obj)
            if pairs is not None:
                return pairs
    return obj

def facts_of(obj: Any) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    top_pick = None
    for val in obj.values():
        if isinstance(val, dict) and len(val) >= 2:
            scalars = {k: v for k, v in val.items() if isinstance(v, (str, int, float)) and (not isinstance(v, bool))}
            if len(scalars) >= 2 and (top_pick is None or len(scalars) > len(top_pick)):
                top_pick = scalars
    return top_pick
FILLER_WORDS = frozenset('a an the of in on for with to be is are it its as at by from that this'.split())
QUALIFIER_RX = re.compile('\\b(not|no|never|without|except|unlike|other than|anything but|rather than|instead of|at least|at most|up to|under|over|more than|less than|between|same as|similar|compatible|page|listing|sold by|or less|or more|minimum|maximum)\\b|[<>≤≥]')
JOINER_RX = re.compile('\\s*(?:,|/|;|\\||&|\\band\\b|\\bor\\b|\\+)\\s*')

def word_set(words_out: str) -> frozenset[str]:
    return frozenset((w for w in word_roots(words_out) if w not in FILLER_WORDS))

def one_letter_apart(a: str, b: str) -> bool:
    if a == b:
        return True
    if min(len(a), len(b)) < 5 or abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = (b, a)
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:] or (len(a) == len(b) >= 7 and a[i + 1:] == b[i + 1:])

def near_spelling(a: str, b: str) -> bool:
    if a == b or one_letter_apart(a, b):
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= 3 and (not short.isdigit()) and long.startswith(short) and (len(long) - len(short) <= 2)

def answer_says(need_txt: str, answer: str) -> bool:
    prop_words, _, val = need_txt.partition(':')
    if not val.strip():
        val, prop_words = (prop_words, '')

    def bare(words_out: str) -> set[str]:
        return {w[:-1] if len(w) > 3 and w.endswith('s') else w for w in re.findall('[a-z0-9]+(?:\\.[0-9]+)?', words_out.lower())} - FILLER_WORDS
    terms = bare(QUALIFIER_RX.sub(' ', val.lower())) - bare(prop_words)
    if not terms:
        return True
    heard = bare(answer)
    return all((any((near_spelling(w, h) for h in heard)) for w in terms))

def spelled_alike(want: str, val: Any) -> bool:
    a = re.sub('[^a-z0-9]', '', str(want).lower())
    for option in re.split('\\s*(?:,|/|;|\\|)\\s*', str(val).lower()):
        b = re.sub('[^a-z0-9]', '', option)
        if len(a) >= 5 and len(b) >= 5 and (a != b) and (a.startswith(b) or b.startswith(a) or a[:5] == b[:5]):
            return True
    return False

def just_below(price: float) -> float:
    return price - 1 if float(price).is_integer() else round(price - 0.01, 2)

def key_names_property(prop_words: str, key: str) -> bool:
    p, k = (word_roots(prop_words), word_roots(key))
    if name_covers(p, k):
        return True
    near = {w for w in p if any((one_letter_apart(w, x) for x in k))}
    return bool(p) and len(near) == len(p)
POINTER_PHRASE_RX = re.compile('\\b(?:see|taken from|read from|copied from|comes from|come from|look (?:at|to)|refer to|go by|check the|consult|from (?:another|a different|a separate) (?:product|listing|item)|instead of this|not (?:from|at|on) this|rather than this|take (?:it|its|the figure|the value|the answer|that) from)\\b', re.I)
PROPERTY_SPAN = '([a-z][a-z0-9\\-]*(?: [a-z0-9][a-z0-9\\-]*){0,3})'
POINTER_LEAD_RXS = [re.compile("^\\s*for\\s+(?:the\\s+|its\\s+|this\\s+item's\\s+)?" + PROPERTY_SPAN + '\\s*,', re.I), re.compile('^\\s*(?:its\\s+|the\\s+)?' + PROPERTY_SPAN + '\\s+(?:should|must|is to|are to|has to|needs to)\\s+be\\s+(?:taken|read|copied|checked)\\b', re.I), re.compile('^\\s*(?:its\\s+|the\\s+)?' + PROPERTY_SPAN + '\\s+comes?\\s+from\\b', re.I), re.compile('^\\s*take\\s+(?:its\\s+|the\\s+)?' + PROPERTY_SPAN + '\\s+(?:value\\s+|figure\\s+)?from\\b', re.I)]
NON_PROPERTY_WORDS = {'it', 'figure', 'value', 'answer', 'information', 'that', 'this', 'details', 'detail'}

def sent_elsewhere(words_out: Any) -> list[str]:
    acc: list[str] = []
    for sentence in re.split('(?<=[.!?;])\\s+', str(words_out or '')):
        if not POINTER_PHRASE_RX.search(sentence):
            continue
        for rx in POINTER_LEAD_RXS:
            m = rx.search(sentence)
            if m:
                prop_words = m.group(1).strip().lower()
                if prop_words not in NON_PROPERTY_WORDS and prop_words not in acc:
                    acc.append(prop_words)
                break
    return acc
CONNECTIVE_RX = re.compile('^(?:(?:matching|being|equal to|exactly|to be|set (?:to|at|on)|landing (?:on|at)|coming (?:in|out) as|comes? (?:in|out) as|turning out|turns out)\\s+)+', re.I)

def bare_need(need_txt: str) -> str:
    prop_words, colon, val = str(need_txt).partition(':')
    if not colon:
        return need_txt
    bare = CONNECTIVE_RX.sub('', val.strip()).strip()
    return f'{prop_words.strip()}: {bare}' if bare else need_txt

def spoken_key(tool_name: str) -> str:
    terms = plain_words(tool_name).split()
    while len(terms) > 1 and len(terms[0]) <= 2 and (not terms[0].isdigit()):
        terms = terms[1:]
    return ' '.join(terms)

class RecordJudge:

    @staticmethod
    def split(need_txt: str) -> tuple[str, str]:
        prop_words, _, val = str(need_txt).partition(':')
        return (prop_words.strip(), val.strip()) if val.strip() else ('', prop_words.strip())

    @staticmethod
    def listed_options(val: Any) -> list[frozenset[str]]:
        pieces = [p for p in JOINER_RX.split(str(val).lower()) if p.strip()]
        whole = word_set(str(val))
        hits_by_want = [word_set(p) for p in pieces]
        return [whole] + [t for t in hits_by_want if t and t != whole]

    @staticmethod
    def choices(want: str) -> list[frozenset[str]]:
        low = re.sub('(?<=\\d)[,/](?=\\d)', '\u2060', str(want).lower())
        whole = word_set(want)
        if not re.search('\\bor\\b|,|/|\\|', low):
            return [whole]
        hits_by_want = [word_set(p.replace('\u2060', '/')) for p in re.split('\\s*(?:,|/|\\||\\bor\\b)\\s*', low) if p.strip()]
        return [t for t in hits_by_want if t] or [whole]

    @classmethod
    def verdict(cls, need_txt: str, block: dict[str, Any], words_out: str, decide: str | None=None) -> tuple[str, str]:
        prop_words, want = cls.split(need_txt)
        low = want.lower()
        if not want or QUALIFIER_RX.search(low):
            return ('read', '')
        if not word_set(want):
            return ('read', '')
        allowed = cls.choices(want)
        numeric = bool(re.search('\\d', low))

        def held(val: Any) -> bool:
            return any((choice <= option for option in cls.listed_options(val) for choice in allowed))

        def offered(val: Any) -> bool:
            whole = word_set(str(val))
            pieces = [t for t in cls.listed_options(val)[1:] if t]
            return len(pieces) >= 3 and (not any((choice <= whole and len(choice) * 2 > len(whole) for choice in allowed)))
        named_ids = [(k, v) for k, v in block.items() if prop_words and key_names_property(prop_words, k)]
        if decide and len(named_ids) > 1:
            named_ids = [(k, v) for k, v in named_ids if plain_words(k) == decide] or named_ids
        for key, val in named_ids:
            if held(val):
                return ('read' if offered(val) else 'meets', f'{key}={str(val)[:60]}')
        if named_ids:
            key, val = named_ids[0]
            if numeric or any((spelled_alike(want, v) for _, v in named_ids)):
                return ('read', f'{key}={str(val)[:60]}')
            return ('differs', f'{key}={str(val)[:60]}')
        for key, val in block.items():
            if held(val) and (not offered(val)):
                return ('elsewhere', f'{key}={str(val)[:60]}')
        terms = word_set(words_out)
        if any((choice <= terms for choice in allowed)):
            return ('elsewhere', 'title/description')
        return ('read' if numeric else 'silent', '')

    @classmethod
    def sheet(cls, needs: list[str], rec_doc: dict[str, Any], deciders: dict[str, str] | None=None) -> list[tuple[str, str, str]]:
        block = facts_of(rec_doc) or {}
        words_out = ' '.join((str(v) for v in rec_doc.values() if isinstance(v, str)))[:4000]
        row_list = [(need_txt, *cls.verdict(need_txt, block, words_out, (deciders or {}).get(need_txt))) for need_txt in needs]
        moved = [p for v in rec_doc.values() if isinstance(v, str) for p in sent_elsewhere(v)]
        if not moved:
            return row_list
        acc = []
        for need_txt, verdict, where in row_list:
            prop_words = cls.split(need_txt)[0]
            delivered = next((p for p in moved if prop_words and (key_names_property(prop_words, p) or key_names_property(p, prop_words))), None)
            acc.append((need_txt, 'read', f'its own notes say to take {delivered} from another listing, so its printed value does not count') if delivered else (need_txt, verdict, where))
        return acc

    @staticmethod
    def clear(sheet: list[tuple[str, str, str]]) -> bool:
        return bool(sheet) and all((v in ('meets', 'elsewhere') for _, v, _ in sheet))

    @staticmethod
    def broken(sheet: list[tuple[str, str, str]]) -> list[str]:
        return [need_txt for need_txt, v, _ in sheet if v == 'differs']

    @staticmethod
    def card(pid: str, sheet: list[tuple[str, str, str]]) -> str:
        marks = {'meets': 'MEETS', 'differs': 'DIFFERS', 'elsewhere': 'STATED ELSEWHERE', 'silent': 'NOT STATED', 'read': 'TO READ'}
        pieces = [f'{need_txt} -> {marks[v]}' + (f' ({where})' if where else '') for need_txt, v, where in sheet]
        return f'{pid}: ' + '; '.join(pieces)

class Want:

    def __init__(self, tool_name: str, shelf: str, qty: int, musts: list[str]):
        self.name = tool_name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []

class Wire:

    def steps_left(self) -> int:
        return self.step_limit - self.steps_used

    def seconds_used(self) -> float:
        return time.monotonic() - self.started

    def commit_step(self) -> int:
        reserve = max(12, -(-3 * self.step_limit // 10)) + 2 * (max(1, len(self.want_list)) - 1)
        return max(self.step_limit // 2, self.step_limit - reserve)

    def past_clock(self, spare: float=0.0) -> bool:
        return time.monotonic() - self.started > CLOCK_LIMIT - spare

    def check_seats(self, ceiling: float=BUDGET_EASE) -> list[str]:
        if self.outlay <= ceiling and self.outlay + DEEP_MARGIN <= BUDGET_STOP:
            return [SEAT_DEEP, SEAT_QUICK]
        return [SEAT_QUICK] if self.outlay <= BUDGET_STOP else []

    def consult(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None=None) -> dict[str, Any] | None:
        for model in models:
            with self.counter_lock:
                if self.model_calls >= self.call_ceiling or self.outlay >= BUDGET_HALT or self.past_clock(-30):
                    return None
                self.model_calls += 1
            tries = 0
            effort = 'low' if tools and self.short_think and self.effort_allowed else None
            while True:
                message, status = self.post_model(model, messages, max_tokens, tools, effort)
                if message is not None:
                    return message
                if effort and (not self.effort_allowed):
                    effort = None
                    continue
                if status == 'cut' and effort is None and self.effort_allowed and (not self.past_clock(25)):
                    if tools:
                        self.short_think = True
                    effort = 'low'
                    with self.counter_lock:
                        if self.model_calls >= self.call_ceiling or self.outlay >= BUDGET_HALT:
                            return None
                        self.model_calls += 1
                    continue
                passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                if not passing or tries >= (1 if status in (None, 408) else len(BACKOFF)) or self.backoff_spent + BACKOFF[tries] > BACKOFF_TOTAL:
                    break
                pause = BACKOFF[tries] * (0.8 + 0.4 * random.random())
                with self.counter_lock:
                    self.backoff_spent += BACKOFF[tries]
                tries += 1
                time.sleep(pause)
            if status == 402:
                self.out_of_credit = True
                return None
        return None

    def post_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None, effort: str | None=None) -> tuple[dict[str, Any] | None, Any]:
        delivered = messages
        if model.startswith('anthropic/') and len(messages) > 1:
            delivered = [dict(m) for m in messages]
            marks = [0] + ([len(delivered) - 1] if len(delivered) > 1 else [])
            for pos in marks:
                body_text = delivered[pos].get('content')
                if isinstance(body_text, str) and body_text:
                    delivered[pos]['content'] = [{'type': 'text', 'text': body_text, 'cache_control': {'type': 'ephemeral'}}]
        body: dict[str, Any] = {'model': model, 'messages': delivered, 'max_tokens': max_tokens, 'temperature': 0}
        if effort and self.effort_allowed:
            body['reasoning'] = {'effort': effort}
        if tools:
            body['tools'] = tools
            body['tool_choice'] = 'auto'
        result = _brain.post_verbose('/inference/chat/completions', body)
        payload_data = result.data
        if isinstance(payload_data, dict) and payload_data.get('choices'):
            usage = payload_data.get('usage') or {}
            if isinstance(usage.get('cost'), (int, float)):
                with self.counter_lock:
                    self.outlay += float(usage['cost'])
            choice = payload_data['choices'][0] or {}
            message = choice.get('message') or {}
            spent_out = as_count(usage.get('completion_tokens')) or 0
            if (choice.get('finish_reason') == 'length' or spent_out >= max_tokens - 2) and (not message.get('tool_calls')) and (tools or json_object(message.get('content')) is None):
                with self.counter_lock:
                    self.cut_replies += 1
                self.journal.append({'kind': 'llm_cut', 'model': model, 'out': spent_out, 'effort': effort})
                return (None, 'cut')
            if message.get('content') or message.get('tool_calls'):
                return (message, 200)
            self.journal.append({'kind': 'llm_fail', 'model': model, 'status': 0})
            return (None, 0)
        error = result.error or {}
        status = error.get('status')
        if isinstance(payload_data, dict) and isinstance(payload_data.get('error'), dict):
            status = as_count(payload_data['error'].get('code'))
            error = {'body': payload_data['error'].get('message')}
        self.journal.append({'kind': 'llm_fail', 'model': model, 'status': status, 'body': str(error.get('body') or '')[:200]})
        if effort and status in (400, 422):
            self.effort_allowed = False
        return (None, status)

    def ask_for_json(self, ask_text_full: str, models: list[str], max_tokens: int=1800) -> dict[str, Any] | None:
        for model in models:
            message = self.consult([{'role': 'user', 'content': ask_text_full}], [model], max_tokens)
            json_got = json_object((message or {}).get('content'))
            if json_got is not None:
                return json_got
            if self.halted or self.out_of_credit or self.model_calls >= self.call_ceiling or (self.outlay >= BUDGET_HALT):
                return None
        return None

    def store_call(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(self.session)
        payload['step'] = self.steps_used + 1
        payload['action'] = {'name': tool_name, 'args': params}
        for _ in range(4):
            result = _store.post_verbose('/environment/call', payload)
            if isinstance(result.data, dict):
                payload_data = result.data
                if 'observation' in payload_data:
                    raw = payload_data.get('observation')
                    payload_data['observation'] = self.guarded(as_records, raw, raw)
                    if payload_data['observation'] is not raw and (not self.__dict__.get('shape_noted')) and (as_text(payload_data['observation']) != as_text(raw)):
                        self.shape_noted = True
                        self.journal.append({'kind': 'records_from_text', 'tool': tool_name})
                self.steps_used = payload_data['step'] if isinstance(payload_data.get('step'), int) else self.steps_used + 1
                if payload_data.get('finished'):
                    self.concluded = True
                met = payload_data.get('observation')
                if tool_name == self.tool_roles.get('search'):
                    self.pending_search = dict(params)
                if tool_name in self.page_tool_set() and (not (isinstance(met, dict) and met.get('error'))):
                    for val in params.values():
                        self.pages_opened.update((v.strip() for v in (val if isinstance(val, list) else [val]) if isinstance(v, str)))
                return payload_data
            error = result.error or {}
            status = error.get('status')
            body = str(error.get('body') or '')
            self.journal.append({'kind': 'env_fail', 'status': status, 'body': body[:200]})
            if status == 409:
                low = body.lower()
                if 'finish' in low or 'quarantin' in low or 'finaliz' in low:
                    self.concluded = True
                    return None
                hits_by_want = re.search('expected (\\d+)', body)
                if hits_by_want:
                    self.steps_used = int(hits_by_want.group(1)) - 1
                    payload['step'] = self.steps_used + 1
                    continue
                if 'reuse' in low or 'different action' in low:
                    self.steps_used += 1
                    payload['step'] = self.steps_used + 1
                    continue
                return None
            if status in (400, 401, 402, 403, 404, 422):
                if status == 402:
                    self.halted = True
                return None
        return None

    def guarded(self, fn: Any, default: Any, *params: Any) -> Any:
        try:
            return fn(*params)
        except Exception as err:
            self.journal.append({'kind': 'safely_error', 'step': getattr(fn, '__name__', '?'), 'error': f'{type(err).__name__}: {err}'[:200]})
            return default

    def render(self, tool_result: Any) -> str:
        words_out = as_text(clipped_copy(tool_result, ROW_CHARS))
        if len(words_out) > RESULT_CHARS:
            words_out = words_out[:RESULT_CHARS] + '...(cut)'
        return words_out

    def shrink_chat(self) -> None:
        size = sum((len(as_text(m.get('content'))) for m in self.dialogue))
        if size <= CHAT_CHARS:
            return
        tools = [i for i, m in enumerate(self.dialogue) if m.get('role') == 'tool']
        goal = int(CHAT_CHARS * 0.65)
        recent = set(tools[-6:])
        for pos in tools:
            if size <= goal:
                break
            body_text = self.dialogue[pos].get('content') or ''
            if pos in recent or len(body_text) <= 600 or '...(older result shortened;' in body_text:
                continue
            pids = re.findall('"([A-Za-z0-9][A-Za-z0-9_\\-:.]{5,40})"', body_text)
            pids = [i for i in dict.fromkeys(pids) if i in self.listings][:20]
            new = body_text[:400] + '...(older result shortened; listings in it: ' + ', '.join((f'"{i}"' for i in pids)) + ')'
            size -= len(as_text(body_text)) - len(as_text(new))
            self.dialogue[pos]['content'] = new

class Intake:

    def step_pending(self, stage: str) -> bool:
        return any((p['when'] == stage for p in self.required_steps)) and (stage, self.basket_key() + f'#{self.need_version}') not in self.steps_done

    def closing_room(self) -> int:
        talk = any((p['when'] == 'order' and p['messages'] > 0 for p in self.required_steps))
        return 2 if talk and self.messages_sent < self.message_cap and self.step_pending('order') else 1

    def take_brief(self) -> None:
        rows_out = []
        for tool_id in self.tool_specs:
            fn = tool_id['function']
            rows_out.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {as_text(fn.get('parameters') or {})}")
        ask_text_full = INTAKE_FORM + '\nTOOLS:\n' + '\n'.join(rows_out) + '\n\nREQUEST:\n' + self.request_text
        if self.policy.get('opening') is not None:
            ask_text_full += '\n\nOPENING NOTE FROM THE STORE:\n' + as_text(clipped_copy(self.policy.get('opening'), 1500))
        intake_doc = self.ask_for_json(ask_text_full, [SEAT_QUICK], 4000) or {}
        if not self.out_of_credit and (not as_list(intake_doc.get('items')) or not isinstance(intake_doc.get('roles'), dict) or (not intake_doc['roles'].get('buy'))):
            intake_doc = self.ask_for_json(ask_text_full, [SEAT_DEEP, SEAT_QUICK], 2500) or intake_doc
        elif not self.out_of_credit and self.shelf_shared(intake_doc):
            again = self.ask_for_json(ask_text_full, [SEAT_DEEP], 2500) or {}
            if as_list(again.get('items')) and isinstance(again.get('roles'), dict) and again['roles'].get('buy'):
                intake_doc = again
        self.intake = intake_doc
        roles = intake_doc.get('roles') if isinstance(intake_doc.get('roles'), dict) else {}
        self.tool_roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.tool_schemas}
        for key in ('buy_args', 'refuse_args'):
            if not isinstance(intake_doc.get(key), dict):
                intake_doc[key] = {}
        checkout_tool, refuse = (self.tool_roles.get('buy'), self.tool_roles.get('refuse'))
        if checkout_tool and checkout_tool == refuse and (not intake_doc['buy_args'] or not intake_doc['refuse_args'] or intake_doc['buy_args'] == intake_doc['refuse_args']):
            ask = f'The store tool {checkout_tool} both checks out and ends without a purchase, depending on its arguments. Its schema: ' + as_text(self.tool_schemas.get(checkout_tool)) + '\nReturn ONE JSON object and nothing else: {"buy_args": the fixed arguments that check out, "refuse_args": the fixed arguments that end without a purchase} (only the arguments that choose the mode).'
            fixed_args = self.ask_for_json(ask, [SEAT_QUICK, SEAT_DEEP], 1500) or {}
            if isinstance(fixed_args.get('buy_args'), dict) and isinstance(fixed_args.get('refuse_args'), dict) and (fixed_args['buy_args'] != fixed_args['refuse_args']):
                intake_doc['buy_args'], intake_doc['refuse_args'] = (fixed_args['buy_args'], fixed_args['refuse_args'])
        for rec in as_list(intake_doc.get('items')):
            if not isinstance(rec, dict):
                continue
            quantity = as_count(rec.get('quantity'))
            num = quantity if quantity and quantity > 0 else 1
            musts = [str(m) for m in as_list(rec.get('musts')) if m]
            self.want_list.append(Want(str(rec.get('name') or 'the product'), str(rec.get('shelf') or ''), num, musts))
        merged: list[Want] = []
        became: dict[int, int] = {}
        for old, want_obj in enumerate(self.want_list, 1):
            twin = next((m for m in merged if want_obj.shelf.strip() and plain_words(m.shelf) == plain_words(want_obj.shelf) and (not any((plain_words(a.split(':')[0]) == plain_words(b.split(':')[0]) and plain_words(a.partition(':')[2]) != plain_words(b.partition(':')[2]) for a in m.musts for b in want_obj.musts)))), None)
            if twin is None:
                merged.append(want_obj)
                became[old] = len(merged)
            else:
                became[old] = merged.index(twin) + 1
                twin.musts += [m for m in want_obj.musts if m not in twin.musts]
                self.journal.append({'kind': 'items_merged', 'into': twin.name, 'dropped': want_obj.name})
        self.want_list = merged
        if not self.want_list:
            self.want_list = [Want('the requested product', '', 1, [])]
        for pair in as_list(intake_doc.get('fits')):
            if isinstance(pair, dict):
                part, main = (became.get(as_count(pair.get('item')) or 0), became.get(as_count(pair.get('with')) or 0))
                if part and main and (part != main) and (1 <= part <= len(self.want_list)) and (1 <= main <= len(self.want_list)) and ((part, main) not in self.fit_pairs):
                    self.fit_pairs.append((part, main))
        for lookup in as_list(intake_doc.get('lookups')):
            words_out = str(lookup).strip()
            if not any((words_out in want_obj.musts for want_obj in self.want_list)):
                key = plain_words(words_out.split(':')[0])
                words_out = next((m for want_obj in self.want_list for m in want_obj.musts if plain_words(m.split(':')[0]) == key), words_out)
            slot_no = next((n for n, want_obj in enumerate(self.want_list, 1) if words_out in want_obj.musts), 1)
            if self.points_at_other_want(slot_no, words_out):
                self.journal.append({'kind': 'lookup_is_fit', 'item': slot_no, 'text': words_out[:120]})
                continue
            if words_out and (slot_no, words_out) not in self.chain_needs:
                self.chain_needs.append((slot_no, words_out))
        self.separate_requirements()
        budget = intake_doc.get('budget')
        if isinstance(budget, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', budget):
            budget = float(budget)
        if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0):
            self.spend_limit = budget
        ceiling = as_count(intake_doc.get('messages'))
        if ceiling is not None and ceiling >= 0:
            self.message_cap = ceiling
        elif self.tool_roles.get('ask'):
            self.message_cap = 3
        steps = as_count(intake_doc.get('steps'))
        if steps is not None and 10 <= steps < self.step_limit:
            self.step_limit = steps
        self.turn_ceiling = max(TURN_CAP, min(110, self.step_limit))
        self.call_ceiling = max(MODEL_CALL_CAP, self.turn_ceiling + 25)
        for rec in as_list(intake_doc.get('process')):
            if isinstance(rec, dict) and str(rec.get('step') or '').strip():
                when = str(rec.get('when') or '').strip().lower()
                self.required_steps.append({'step': str(rec['step']).strip(), 'when': when if when in ('choose', 'order', 'end') else 'order', 'messages': max(0, as_count(rec.get('messages')) or 0)})
        if self.tool_roles.get('ask'):
            talk = [p['messages'] for p in self.required_steps if p['when'] == 'order' and p['messages']]
            self.reserved_messages = min(max(talk), self.message_cap) if talk else 0

    def points_at_other_want(self, slot_no: int, words_out: str) -> bool:
        low = ' ' + plain_words(words_out) + ' '
        if re.search(' item [0-9]+ | (?:chosen|picked|selected) (?:as|for) ', low):
            return True
        if re.search(' (?:sold|sells|listed|offered|made) by | shop | seller | page ', low):
            return False
        linked = [n for n in range(1, len(self.want_list) + 1) if n != slot_no and ((slot_no, n) in self.fit_pairs or (n, slot_no) in self.fit_pairs)]
        return any((' ' + plain_words(self.want_list[n - 1].name) + ' ' in low for n in linked if plain_words(self.want_list[n - 1].name) not in ('the product', 'the requested product')))

    @staticmethod
    def shelf_shared(intake_doc: dict[str, Any]) -> bool:
        shelves = [plain_words(str(e.get('shelf') or '')) for e in as_list(intake_doc.get('items')) if isinstance(e, dict)]
        shelves = [s for s in shelves if s]
        return len(shelves) != len(set(shelves))

    def separate_requirements(self) -> None:
        for slot_no, want_obj in enumerate(self.want_list, 1):
            pointers = [need_txt for n, need_txt in self.chain_needs if n == slot_no and need_txt in want_obj.musts]
            rest = [m for m in want_obj.musts if m not in pointers]
            if not pointers or not rest:
                continue
            ask_text_full = SPLIT_FORM + '\n\nREQUEST:\n' + self.request_text + f'\n\nPRODUCT: {want_obj.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n' + '\n'.join(pointers) + '\nOTHER REQUIREMENTS:\n' + '\n'.join(rest)
            json_got = self.ask_for_json(ask_text_full, [SEAT_DEEP, SEAT_QUICK], 1500) or {}
            moved = [m for m in rest if m in [str(x).strip() for x in as_list(json_got.get('describe_other'))]]
            if not moved:
                continue
            want_obj.musts = [m for m in want_obj.musts if m not in moved]
            joined = pointers[0] + ' (that listing: ' + '; '.join(moved) + ')'
            want_obj.musts = [joined if m == pointers[0] else m for m in want_obj.musts]
            self.chain_needs = [(n, joined if (n, need_txt) == (slot_no, pointers[0]) else need_txt) for n, need_txt in self.chain_needs]
            self.journal.append({'kind': 'unmix', 'item': slot_no, 'moved': moved})

    def already_stated(self, slot_no: int, need_txt: str) -> bool:
        if need_txt in [m for _, m in self.chain_needs] or need_txt in [str(x) for x in as_list(self.intake.get('lookups'))]:
            return False
        prop_words, _, val = need_txt.partition(':')
        if not prop_words.strip() or len(re.findall('[a-z0-9]+', val.lower())) < 4:
            return False
        if re.search('\\b(?:listing|page|shown|other than|unlike|same as)\\b', val, re.I):
            return False
        want_obj = self.want_list[slot_no - 1]
        if any(({w for w in word_roots(alt.name) if len(w) >= 4} & word_roots(val) for alt in self.want_list if alt is not want_obj)):
            return False
        shelf_key = self.shelf_field()
        field_names: set[str] = set()
        read = 0
        for rec_doc in self.read_records.values():
            if shelf_key and (not self.belongs_to(want_obj, str(rec_doc.get(shelf_key) or ''))):
                continue
            block = facts_of(rec_doc) or {}
            if block:
                read += 1
                field_names |= {plain_words(k) for k in block}
        return read >= 3 and (not any((word_roots(k) & word_roots(prop_words) for k in field_names)))

    def requirement_sheet(self) -> str:
        pieces = []
        multi_arg = len(self.want_list) > 1
        for slot_no, want_obj in enumerate(self.want_list, 1):
            live = [m for m in want_obj.musts + want_obj.told if m not in want_obj.dropped]
            stated_now = [m for m in live if self.already_stated(slot_no, m)]
            needs = [m + (f' (the page it points at gives: {self.chain_values[m]})' if m in self.chain_values else '') for m in live if m not in stated_now]
            head_word = f'item {slot_no}: {want_obj.name}' if multi_arg else f'item: {want_obj.name}'
            if want_obj.shelf:
                head_word += f' (store label: {want_obj.shelf})'
            head_word += f', quantity {want_obj.qty}'
            whole = sorted((l for l in self.labels_seen() if self.label_matches_whole(slot_no, l)))
            if whole:
                head_word += f"; kind: only listings whose category label is {' or '.join((repr(l) for l in whole[:3]))} (a label naming the kind in part is another department and does not count)"
            words_out = head_word + '; requirements: ' + ('; '.join(needs) if needs else 'none beyond the kind of product')
            if stated_now:
                words_out += "; described in the shopper's words, which no listing attribute states (decides nothing): " + '; '.join(stated_now)
            if want_obj.free:
                words_out += '; shopper does not mind: ' + ', '.join(want_obj.free)
            pieces.append(words_out)
        intake_doc = self.intake
        if self.spend_limit is not None:
            scope = 'for each item' if intake_doc.get('budget_scope') == 'each' else 'in total'
            pieces.append(f"budget: at most {self.spend_limit} {intake_doc.get('budget_unit') or ''} {scope}".replace('  ', ' '))
        choice = intake_doc.get('pick')
        if choice == 'cheapest':
            pieces.append('pick: the cheapest listing that fits (lowest price)')
        elif choice in ('most', 'least'):
            pieces.append(f"pick: the {choice} {intake_doc.get('pick_measure') or ''} among fitting listings".replace('  ', ' '))
        else:
            pieces.append('pick: any listing that fits everything')
        if intake_doc.get('in_stock_only', True) is not False:
            pieces.append('in stock only')
        for wish in as_list(intake_doc.get('wishes')):
            pieces.append(f'wish (tie-breaker only): {wish}')
        for rule in as_list(intake_doc.get('rules')):
            pieces.append(f'rule from the request: {rule}')
        if intake_doc.get('quiet_decline') is True:
            pieces.append('if nothing fits: end without naming or suggesting any listing')
        for rec in self.required_steps:
            when = {'choose': 'before choosing', 'order': 'at the order (Wren does it when you check out)', 'end': 'at an ending without a purchase (Wren does it then)'}[rec['when']]
            pieces.append(f"step the request asks for, {when}: {rec['step']}")
        facts = self.detail_tool()
        if facts and (self.detail_sources.get(facts) is True or self.tool_roles.get('offer')):
            bare = sorted((t for t in self.page_tool_set() | {self.tool_roles.get('offer') or ''} if t and self.detail_sources.get(t) is False))
            pages = sorted((t for t in self.page_tool_set() if t != facts))
            pieces.append(f'listing attributes come from {facts}' + (f"; {', '.join(bare)} show none" if bare else '') + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ''))
        return '\n'.join(('- ' + p for p in pieces))

    def progress_note(self) -> str:
        left = self.steps_left()
        words_out = f'[Wren] step {self.steps_used} of {self.step_limit} used, {left} left.'
        if left <= self.closing_room() + 1:
            words_out += ' FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that).'
        elif left <= 6 or (left <= len(self.want_list) + 5 and self.checkouts == 0):
            words_out += ' Few steps left: set the cart to the chosen lines and check out now.'
        elif self.checkouts == 0 and left <= 12:
            words_out += ' Steps are running low: settle on your picks, fill the cart and check out within the next few steps.'
        elif self.checkouts == 0 and (not self.basket_touched) and (self.steps_used >= self.step_limit // 2):
            words_out += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        live = {need_txt for _, need_txt in self.live_needs()}
        lookups = [need_txt for _, need_txt in self.chain_needs if need_txt in live and need_txt not in self.chain_values]
        if lookups and self.checkouts == 0:
            words_out += '\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] ' + ' | '.join(lookups[:3])
        if self.commit_called and self.checkouts == 0 and self.basket_ids() and (left > self.closing_room() + 1):
            words_out += ' The commit point has passed: check out the cart in this turn unless a line clearly breaks a requirement (for a cheapest pick, the carted listing stands as the cheapest found; look for a cheaper one only after a refused checkout leaves steps for it).'
        if self.lead_ids:
            words_out += '\n[review candidates] ' + ', '.join(self.lead_ids)
        if self.checkouts == 0 and self.steps_left() > self.closing_room() + 2 and self.shelf_field():
            carted = set(self.basket_ids())
            named_ids = []
            for n in range(1, len(self.want_list) + 1):
                if any((i in carted for i in self.listings_for(n))):
                    continue
                row_list = self.rows_naming_values(n)[:3]
                if row_list:
                    named_ids.append((f'item {n}: ' if len(self.want_list) > 1 else '') + ', '.join(row_list))
            if named_ids:
                words_out += '\n[unopened rows of the kind whose text names each plain required value, in budget and in stock: ' + '; '.join(named_ids) + '. Open them before searching again or ending without a purchase.]'
        proposal = self.fitment_status() if self.checkouts == 0 and len(self.want_list) > 1 else ''
        if proposal:
            words_out += '\n' + proposal
        if self.notices:
            words_out += '\n[changes so far] ' + ' | '.join(self.notices[-4:])
        words_out += '\n[ledger]\n' + self.requirement_sheet()
        return words_out

    def run_step(self, stage: str) -> tuple[bool, str]:
        steps = [p for p in self.required_steps if p['when'] == stage]
        if not steps or self.concluded or (not self.step_pending(stage)):
            return (True, '')
        ending = {self.tool_roles.get('buy'), self.tool_roles.get('refuse')} - {None}
        tools = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {as_text(t['function'].get('parameters') or {})}" for t in self.tool_specs if t['function']['name'] not in ending]
        note_text: list[str] = []
        nudged = False
        for _ in range(STEP_ROUNDS):
            ask_text_full = STEP_FORM.replace('STAGE', 'place the order' if stage == 'order' else 'end without a purchase') + '\n\nREQUEST:\n' + self.request_text + '\n\nSTEPS DUE NOW:\n' + '\n'.join((f"- {p['step']}" for p in steps)) + '\n\nLEDGER:\n' + self.requirement_sheet() + '\n\nCART NOW: ' + self.basket_text() + '\n\nRECORD (messages and step calls so far):\n' + ('\n'.join((f'{i}. {row_line}' for i, row_line in enumerate(self.action_log, 1))) or '(nothing yet)') + ('\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.notices[-4:]) if self.notices else '') + f"\n\nMESSAGES LEFT: {max(0, self.message_cap - self.messages_sent)}   STEPS LEFT: {self.steps_left()} (the {('order' if stage == 'order' else 'ending')} itself takes one)" + '\n\nTOOLS:\n' + '\n'.join(tools)
            json_got = self.ask_for_json(ask_text_full, [SEAT_DEEP, SEAT_QUICK] if nudged else [SEAT_QUICK, SEAT_DEEP], 3500)
            if json_got is None:
                break
            stated_now: list[str] = []
            hold = ''
            if note_text:
                self.apply_answers({'needs': json_got.get('needs')}, stated_now)
                hold = str(json_got.get('hold') or '').strip()
            one_call = json_got.get('call') if isinstance(json_got.get('call'), dict) else None
            self.journal.append({'kind': 'gate', 'stage': stage, 'call': clipped_copy(one_call, 200), 'steps': clipped_copy(json_got.get('steps'), 120), 'hold': hold[:200], 'told': stated_now})
            if stated_now or hold:
                self.need_version += 1
                return (False, ('Before going on, Wren did the steps the request asks for:\n' + '\n'.join(note_text) + '\n' if note_text else '') + ("The shopper's reply changes the task: " + '; '.join(stated_now + ([hold] if hold else [])) + '. Fix the cart for it, then check out again.'))
            tool_name = str((one_call or {}).get('tool') or '')
            displayed = [e for e in as_list(json_got.get('steps')) if isinstance(e, dict)]
            talk = stage == 'order' and any((p['messages'] > 0 for p in steps))
            rows_out = range(1, len(self.action_log) + 1)
            proof = {i for i in rows_out if not talk or (i <= len(self.action_steps) and self.action_steps[i - 1] > self.basket_changed_at)}
            unproven = [e for e in displayed if e.get('done') is not True or not any(((as_count(n) or 0) in proof for n in as_list(e.get('lines'))))]
            if not one_call and unproven and (not note_text) and (not nudged) and self.tool_roles.get('ask') and (self.messages_sent < self.message_cap) and (self.steps_left() >= 2):
                nudged = True
                self.journal.append({'kind': 'gate_unproven', 'steps': clipped_copy(unproven, 120)})
                self.action_log.append(f'(Wren: no line made since the cart last changed at step {self.basket_changed_at} shows ' + '; '.join((str(e.get('step') or '') for e in unproven))[:200] + f' done, and {self.message_cap - self.messages_sent} message(s) and {self.steps_left()} steps are left: give the call that does it)')
                self.action_steps.append(-1)
                continue
            if not one_call or tool_name not in self.tool_schemas or tool_name in ending:
                if not one_call:
                    self.steps_done.add((stage, self.basket_key() + f'#{self.need_version}'))
                break
            if self.steps_left() < 2 or (tool_name == self.tool_roles.get('ask') and self.messages_sent >= self.message_cap):
                break
            fault_list: list[str] = []
            params = conform(one_call.get('args') if isinstance(one_call.get('args'), dict) else {}, self.tool_schemas[tool_name], 'args', fault_list)
            if fault_list:
                break
            if tool_name == self.tool_roles.get('ask'):
                words_out = next((v for v in params.values() if isinstance(v, str)), '')
                batch = self.deliver([(1, '', words_out)], 1)
                if not batch:
                    break
                self.step_messages += 1
                note_text.append(f'Q: {words_out}\nA: {batch[0][2]}')
                continue
            reply_obj = self.store_call(tool_name, params)
            if reply_obj is None:
                break
            met = reply_obj.get('observation')
            self.take_in(met)
            self.action_log.append(f'step {self.steps_used}: {tool_name} {as_text(params)} -> {self.render(met)[:300]} (cart then: {self.basket_key()})')
            self.action_steps.append(self.steps_used)
            note_text.append(f'{tool_name} {as_text(params)} -> {self.render(met)[:400]}')
        return (True, 'Before going on, Wren did the steps the request asks for:\n' + '\n'.join(note_text) if note_text else '')

    def live_needs(self) -> list[tuple[int, str]]:
        acc = []
        for slot_no, want_obj in enumerate(self.want_list, 1):
            for need_txt in want_obj.musts + want_obj.told:
                if need_txt not in want_obj.dropped:
                    acc.append((slot_no, need_txt))
        return acc

class Catalog:

    def page_tool_set(self) -> set[str]:
        page = self.tool_roles.get('offer') or self.tool_roles.get('open')
        return {t for t in (page, self.tool_roles.get('compare')) if t}

    def detail_tool(self) -> str | None:
        known = next((t for t, ok in self.detail_sources.items() if ok and self.id_argument(t)), None)
        if known:
            return known
        tool_id = self.tool_roles.get('open')
        return tool_id if self.detail_sources.get(tool_id or '') is not False and self.id_argument(tool_id) else None

    def multi_view_shows_facts(self) -> bool | None:
        return self.detail_sources.get(self.tool_roles.get('compare') or '')

    def multi_view_is_bare(self) -> bool:
        met = self.multi_view_shows_facts()
        return met is False or (met is None and (self.detail_sources.get(self.detail_tool() or '') is True or bool(self.tool_roles.get('offer'))))

    def id_of(self, obj: Any, in_list: bool=False, level: int=0) -> str | None:
        if not isinstance(obj, dict) or len(obj) < 3:
            return None
        if self.id_field and looks_like_id(obj.get(self.id_field), True):
            return obj[self.id_field].strip()
        nested = any((isinstance(v, dict) for v in obj.values()))
        if not (in_list or level == 0 or nested):
            return None
        hits_by_want = guess_id_key([obj])
        if hits_by_want and ('id' in str(hits_by_want).lower() or obj[hits_by_want].strip() in self.listings):
            return obj[hits_by_want].strip()
        return None

    def learn_row_shape(self, obj: list[Any]) -> bool:
        row_list = [n for n in obj if isinstance(n, dict) and len(n) >= 3]
        if not row_list or len(row_list) < len(obj) / 2:
            return False
        if self.id_field and all((looks_like_id(r.get(self.id_field), True) for r in row_list)):
            return True
        hits_by_want = guess_id_key(row_list)
        if hits_by_want:
            self.id_field = hits_by_want
            return True
        return False

    def take_in(self, tool_result: Any) -> None:

        def walk(obj: Any, level: int, in_list: bool=False) -> None:
            if level > 6:
                return
            if isinstance(obj, dict):
                pid = self.id_of(obj, in_list, level)
                if pid:
                    if pid not in self.listings:
                        self.listing_order.append(pid)
                        self.listings[pid] = {}
                    old = self.listings[pid]
                    if any((old.get(k) is True and v is False for k, v in obj.items())):
                        self.unavailable_ids.add(pid)
                    elif any((old.get(k) is False and v is True for k, v in obj.items())):
                        self.unavailable_ids.discard(pid)
                    self.listings[pid].update(obj)
                    if facts_of(obj):
                        self.read_records[pid] = dict(self.listings[pid])
                    elif pid in self.read_records:
                        self.read_records[pid].update(obj)
                    return
                for val in obj.values():
                    walk(val, level + 1)
            elif isinstance(obj, list):
                row_list = self.learn_row_shape(obj)
                for val in obj:
                    walk(val, level + 1, row_list)
        walk(tool_result, 0)

    def shelf_labels(self) -> set[str]:
        hits_by_want: set[str] = set()

        def walk(spec_doc: Any, level: int) -> None:
            if level > 6 or not isinstance(spec_doc, dict):
                return
            for option in spec_doc.get('enum') or []:
                if isinstance(option, str):
                    hits_by_want.add(option.strip().lower())
            for val in spec_doc.values():
                if isinstance(val, dict):
                    walk(val, level + 1)
        for spec_doc in self.tool_schemas.values():
            walk(spec_doc, 0)
        return hits_by_want

    def shelf_field(self) -> str | None:
        counts: dict[str, set[str]] = {}
        present: dict[str, int] = {}
        for row_doc in self.listings.values():
            for key, val in row_doc.items():
                if isinstance(val, str) and 0 < len(val) <= 40 and (not re.search('\\d{3}', val)):
                    counts.setdefault(key, set()).add(val)
                    present[key] = present.get(key, 0) + 1
        total = max(1, len(self.listings))
        known = self.shelf_labels()
        top_pick = None
        for key, vals in counts.items():
            if present[key] < 0.8 * total or len(vals) > 40:
                continue
            listed = sum((1 for v in vals if v.strip().lower() in known)) / len(vals)
            mean = sum((len(v) for v in vals)) / len(vals)
            points = (-listed, len(vals), -present[key], -mean)
            if top_pick is None or points < top_pick[0]:
                top_pick = (points, key)
        return top_pick[1] if top_pick else None

    def property_table(self) -> list[tuple[str, str, int, list[str]]]:
        shelf_key = self.shelf_field()
        groups: dict[str, dict[str, Any]] = {}
        for pid in self.listing_order:
            rec_doc = self.read_records.get(pid)
            if not rec_doc:
                continue
            block = facts_of(rec_doc) or {}
            label_txt = str(rec_doc.get(shelf_key) or 'listings') if shelf_key else 'listings'
            group = groups.setdefault(label_txt, {'n': 0, 'keys': {}})
            group['n'] += 1
            done: set[str] = set()
            for key, val in block.items():
                tool_name = plain_words(key)
                if not tool_name or tool_name in done:
                    continue
                done.add(tool_name)
                bucket = group['keys'].setdefault(tool_name, {'n': 0, 'values': [], 'per': {}})
                bucket['n'] += 1
                words_out = str(val).strip().lower()
                bucket['per'][pid] = words_out
                if words_out not in bucket['values']:
                    bucket['values'].append(words_out)
        wanted = [(plain_words(want_obj.shelf or ''), plain_words(want_obj.name), {w for w in word_roots(want_obj.shelf or '') | word_roots(want_obj.name) if len(w) >= 4}) for want_obj in self.want_list]
        matched = {label_txt for label_txt in groups if any((plain_words(label_txt) in (shelf, tool_name) or {w for w in word_roots(label_txt) if len(w) >= 4} & long for shelf, tool_name, long in wanted)) or self.want_for_label(label_txt)}
        if matched:
            groups = {label_txt: group for label_txt, group in groups.items() if label_txt in matched}
        table = []
        for label_txt, group in sorted(groups.items(), key=lambda kv: -kv[1]['n']):
            kept: list[tuple[str, dict[str, Any]]] = []
            for tool_name, bucket in sorted(group['keys'].items(), key=lambda kv: (-kv[1]['n'], len(kv[0]))):
                vals = list(bucket['per'].values())
                if not vals:
                    continue
                mean = sum((len(v) for v in vals)) / len(vals)
                if mean > 40 or sum((1 for v in vals if ' / ' in v)) * 2 >= len(vals):
                    continue
                recur = sum((len(set(re.findall('[a-z]{3,}', v)) & set(re.findall('[a-z]{3,}', ' '.join((o for o in vals if o is not v))))) > 0 for v in vals))
                if len(vals) >= 4 and len(set(vals)) == len(vals) and (mean > 12) and (recur * 2 < len(vals)):
                    continue
                twin = next((field_spec for alt, field_spec in kept if self.same_property(alt, field_spec, tool_name, bucket)), None)
                if twin is not None:
                    for pid, words_out in bucket['per'].items():
                        twin['per'].setdefault(pid, words_out)
                        if words_out not in twin['values']:
                            twin['values'].append(words_out)
                    twin['n'] = len(twin['per'])
                    continue
                kept.append((tool_name, bucket))
            kept.sort(key=lambda kv: -kv[1]['n'])
            for tool_name, bucket in kept[:14]:
                table.append((label_txt, tool_name, bucket['n'], [v[:40] for v in bucket['values'][:4]]))
        return table

    def unstated_mark(self, label_txt: str, tool_name: str, vals: list[str]) -> str:
        slot_no = self.want_for_label(label_txt)
        if not slot_no:
            return ''
        for must in self.want_list[slot_no - 1].musts:
            prop_words, _, val = must.partition(':')
            if re.search('\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', val, re.I):
                continue
            terms = set(re.findall('[a-z0-9]{3,}', val.lower()))
            displayed = set(re.findall('[a-z0-9]{3,}', ' '.join(vals).lower()))
            if prop_words.strip() and plain_words(prop_words) != tool_name and word_roots(prop_words) & word_roots(tool_name) and terms and (not terms <= displayed):
                return f' [not the stated {prop_words.strip()}: another property]'
        return ''

    @staticmethod
    def same_property(tool_name: str, field_spec: dict[str, Any], alt: str, bucket: dict[str, Any]) -> bool:
        a, b = (set(tool_name.split()), set(alt.split()))
        if not a or not b or (not a & b):
            return False

        def agree(x: str, y: str) -> bool:
            wx, wy = (re.findall('[a-z0-9]+', x.lower()), re.findall('[a-z0-9]+', y.lower()))
            short, long = (wx, wy) if len(wx) <= len(wy) else (wy, wx)
            return x == y or (bool(short) and set(short) <= set(long) and (len(long) <= 2 * len(short) + 2))
        both = [i for i in field_spec['per'] if i in bucket['per'] and field_spec['per'][i].strip() and bucket['per'][i].strip()]
        same = sum((1 for i in both if agree(field_spec['per'][i], bucket['per'][i])))
        if not (a <= b or b <= a):
            return len(both) >= 2 and same == len(both)
        if both:
            return same * 2 > len(both)
        return bool((set(field_spec['values']) & set(bucket['values'])) - {''})

    def want_groups(self) -> list[list[str]]:
        shelf_key = self.shelf_field()
        groups: dict[str, list[str]] = {}
        for pid in self.listing_order:
            label_txt = str(self.listings[pid].get(shelf_key) or '') if shelf_key else ''
            groups.setdefault(label_txt, []).append(pid)
        wanted = [(plain_words(want_obj.shelf or ''), plain_words(want_obj.name), {w for w in word_roots(want_obj.shelf or '') | word_roots(want_obj.name) if len(w) >= 4}) for want_obj in self.want_list]
        matched = [pids for label_txt, pids in groups.items() if any((plain_words(label_txt) in (shelf, tool_name) or {w for w in word_roots(label_txt) if len(w) >= 4} & long for shelf, tool_name, long in wanted)) or self.want_for_label(label_txt)]
        candidates = matched or list(groups.values())
        return sorted(candidates, key=lambda pids: -len(pids))[:max(1, len(self.want_list))]

    def id_argument(self, tool_id: str | None) -> str | None:
        spec_doc = self.tool_schemas.get(tool_id or '') or {}
        prop_specs = spec_doc.get('properties') or {}
        text_fields = [k for k, v in prop_specs.items() if isinstance(v, dict) and 'string' in schema_kinds(v) and ('enum' not in v)]
        id_param = next((k for k in spec_doc.get('required') or [] if k in text_fields), None) or (text_fields[0] if len(text_fields) == 1 else None)
        if not tool_id or not id_param or any((k != id_param for k in spec_doc.get('required') or [])):
            return None
        return id_param

    def price_for(self, pid: str) -> float | None:
        val = (self.listings.get(pid) or {}).get(self.price_field) if self.price_field else None
        if isinstance(val, dict):
            val = next((v for v in val.values() if isinstance(v, (int, float)) and (not isinstance(v, bool))), None)
        if isinstance(val, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', val):
            val = float(val)
        return float(val) if isinstance(val, (int, float)) and (not isinstance(val, bool)) else None

    def can_buy(self, pid: str) -> bool:
        if self.is_gone(pid):
            return False
        val = (self.listings.get(pid) or {}).get(self.stock_field) if self.stock_field else None
        return not (val is False or (isinstance(val, (int, float)) and (not isinstance(val, bool)) and (val <= 0)))

    def listing_records(self, tool_result: Any) -> list[tuple[str, dict[str, Any]]]:
        if isinstance(tool_result, dict):
            pid = self.id_of(tool_result)
            if pid:
                return [(pid, tool_result)]
            hits_by_want: list[tuple[str, dict[str, Any]]] = []
            for val in tool_result.values():
                if isinstance(val, list):
                    hits_by_want += [(i, r) for r in val if isinstance(r, dict) for i in [self.id_of(r, True)] if i]
                elif isinstance(val, dict):
                    hits_by_want += [(i, r) for v in val.values() if isinstance(v, list) for r in v if isinstance(r, dict) for i in [self.id_of(r, True)] if i]
            return hits_by_want
        if isinstance(tool_result, list):
            return [(i, r) for r in tool_result if isinstance(r, dict) for i in [self.id_of(r, True)] if i]
        return []

    def remember_view(self, tool_name: str, tool_result: Any) -> None:
        for pid, rec_doc in self.listing_records(tool_result):
            field_names = set(rec_doc)
            self.tool_fields.setdefault(tool_name, set()).update(field_names)
            version, held_now = self.view_fields.get(pid, (-1, set()))
            self.view_fields[pid] = (self.need_version, (held_now if version == self.need_version else set()) | field_names)

    def seen_in_view(self, tool_name: str, pid: str) -> bool:
        version, held_now = self.view_fields.get(pid, (-1, set()))
        field_names = self.tool_fields.get(tool_name)
        return bool(field_names) and version == self.need_version and (field_names <= held_now) and (pid not in self.gone_at)

    def result_layout(self, tool_name: str, tool_result: Any) -> dict[str, Any] | None:
        if tool_name in self.layouts:
            return self.layouts[tool_name]
        if not isinstance(tool_result, dict):
            return None
        lists = [k for k, v in tool_result.items() if isinstance(v, list) and len(v) >= 2 and all((isinstance(r, dict) for r in v))]
        if not lists:
            return None
        self.layouts[tool_name] = None
        sample = {k: v[:3] if k in lists else v for k, v in tool_result.items()}
        ask_text_full = LAYOUT_FORM + '\n\nTOOL: ' + tool_name + ' | parameters: ' + as_text(self.tool_schemas.get(tool_name) or {}) + '\n\nRESULT (first rows only):\n' + as_text(clipped_copy(sample, 120))[:4000]
        json_got = self.ask_for_json(ask_text_full, [SEAT_QUICK, SEAT_DEEP], 2500) or {}
        row_list = json_got.get('rows') if json_got.get('rows') in lists else None
        if not row_list:
            return None
        first_val = tool_result[row_list][0]
        prop_specs = (self.tool_schemas.get(tool_name) or {}).get('properties') or {}

        def named_ids(key: Any, where: dict[str, Any]) -> str | None:
            return key if isinstance(key, str) and key in where else None
        layout = {'rows': row_list, 'count': named_ids(json_got.get('count'), tool_result), 'next': named_ids(json_got.get('next'), tool_result), 'next_arg': named_ids(json_got.get('next_arg'), prop_specs), 'price': named_ids(json_got.get('price'), first_val), 'stock': named_ids(json_got.get('stock'), first_val)}
        if layout['count'] and (isinstance(tool_result[layout['count']], bool) or not isinstance(tool_result[layout['count']], int)):
            layout['count'] = None
        scalar = lambda v: isinstance(v, (str, int)) and (not isinstance(v, bool))
        if not layout['next']:
            layout['next'] = next((k for k, v in tool_result.items() if k not in lists and scalar(v) and re.search('next|cursor|token|after|continu', k, re.I)), None)
        if not layout['count']:
            layout['count'] = next((k for k, v in tool_result.items() if k not in lists and isinstance(v, int) and (not isinstance(v, bool)) and re.search('total|count|hits|match|found|num', k, re.I)), None)
        if layout['next'] and (not layout['next_arg']):
            layout['next_arg'] = self.next_page_argument(tool_name, tool_result.get(layout['next']))
            if layout['next_arg']:
                self.journal.append({'kind': 'shape_filled', 'tool': tool_name, 'next_arg': layout['next_arg']})
        money = lambda v: isinstance(v, (int, float)) and (not isinstance(v, bool)) or (isinstance(v, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', v) is not None)
        if not layout['price']:
            layout['price'] = next((k for k, v in first_val.items() if re.search('price|cost|amount|fee', k, re.I) and money(v)), None)
        if not layout['stock']:
            layout['stock'] = next((k for k, v in first_val.items() if re.search('stock|hand|avail', k, re.I) and (isinstance(v, bool) or isinstance(v, (int, float)))), None)
        self.layouts[tool_name] = layout
        self.price_field = self.price_field or layout['price']
        self.stock_field = self.stock_field or layout['stock']
        self.journal.append({'kind': 'shape', 'tool': tool_name, 'shape': layout})
        return layout

    def next_page_argument(self, tool_name: str, val: Any) -> str | None:
        prop_specs = (self.tool_schemas.get(tool_name) or {}).get('properties') or {}
        numeric = isinstance(val, int) and (not isinstance(val, bool))
        texty = isinstance(val, str)
        kinds = lambda field_spec: set(schema_kinds(field_spec))
        cands = [k for k, field_spec in prop_specs.items() if isinstance(field_spec, dict) and 'enum' not in field_spec and (bool({'integer', 'number'} & kinds(field_spec)) if numeric else 'string' in kinds(field_spec) if texty else bool({'integer', 'number', 'string'} & kinds(field_spec)))]
        byname = [k for k in cands if re.search('offset|page|begin|start|skip|cursor|token|after|next|continu', k + ' ' + str(prop_specs[k].get('description') or ''), re.I)]
        return byname[0] if byname else cands[0] if len(cands) == 1 else None

    def page_state(self, tool_name: str, tool_result: Any) -> tuple[int, int, str, Any] | None:
        layout = self.result_layout(tool_name, tool_result)
        if layout and layout['next'] and (not layout['next_arg']) and isinstance(tool_result, dict) and (tool_result.get(layout['next']) is not None):
            layout['next_arg'] = self.guarded(self.next_page_argument, None, tool_name, tool_result.get(layout['next']))
            if layout['next_arg']:
                self.journal.append({'kind': 'shape_filled', 'tool': tool_name, 'next_arg': layout['next_arg'], 'late': True})
        if not layout or not layout['count'] or (not layout['next']) or (not layout['next_arg']) or (not isinstance(tool_result, dict)):
            return None
        row_list, num, val = (tool_result.get(layout['rows']), tool_result.get(layout['count']), tool_result.get(layout['next']))
        if not isinstance(row_list, list) or not row_list or isinstance(num, bool) or (not isinstance(num, int)):
            return None
        if val is not None and (isinstance(val, bool) or not isinstance(val, (str, int))):
            return None
        return (num, len(row_list), layout['next_arg'], val)

    def is_price_ordered(self, tool_result: Any) -> bool:
        prices = [self.price_for(i) for i, _ in self.listing_records(tool_result)]
        return len(prices) >= 3 and None not in prices and all((a <= b for a, b in zip(prices, prices[1:])))

    def follow_pages(self, tool_name: str, params: dict[str, Any], tool_result: Any) -> str:
        self.paged_results = []
        choice = self.intake.get('pick')
        proposal = self.page_state(tool_name, tool_result) if choice in ('cheapest', 'most', 'least') else None
        if not proposal:
            return ''
        num, displayed, param, val = proposal
        given = params.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and (not isinstance(given, bool)) else bool(str(given or '').strip())
        if val is None or later or num <= displayed:
            return ''
        if choice == 'cheapest' and self.is_price_ordered(tool_result):
            return ''
        pages = -(-(num - displayed) // displayed)
        room = 4 if self.steps_left() - self.closing_room() >= 24 else 2
        if pages > room and self.page_reads + room <= PAGE_AHEAD and (self.steps_left() - self.closing_room() - room >= 12):
            pages = room
        if pages > room or self.page_reads + pages > PAGE_AHEAD or self.steps_left() - self.closing_room() - pages < 12:
            return f"\n[Wren] This search matched {num} listings and showed {displayed}, in no price order. The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with {param}={as_text(val)}."
        text_fields = []
        for _ in range(pages):
            fault_list: list[str] = []
            one_call = conform({**params, param: val}, self.tool_schemas.get(tool_name), 'args', fault_list)
            if fault_list:
                break
            reply_obj = self.store_call(tool_name, one_call)
            if reply_obj is None or self.concluded:
                break
            self.page_reads += 1
            met = reply_obj.get('observation')
            self.digest(tool_name, met)
            self.paged_results.append(met)
            shown_text = self.render(met)
            self.read_memo[tool_name + as_text(one_call)] = (self.need_version, self.steps_used, shown_text)
            text_fields.append(f'{param}={as_text(val)}: {shown_text}')
            after = self.page_state(tool_name, met)
            val = after[3] if after else None
            if val is None:
                break
        self.journal.append({'kind': 'pages', 'tool': tool_name, 'read': len(text_fields)})
        rest = f"\n[Wren] {num} listings matched and Wren has read {displayed * (len(text_fields) + 1)} rows of them; narrow the search with the requirement's value or page on to see the rest." if val is not None and text_fields else ''
        return '\n[Wren read the next pages of this search itself, one step a page]\n' + '\n'.join(text_fields) + rest if text_fields else ''

    def price_cap_ignored(self, tool_name: str, params: dict[str, Any], tool_result: Any) -> str:
        prop_specs = (self.tool_schemas.get(tool_name) or {}).get('properties') or {}

        def priced(key: str) -> bool:
            field_spec = prop_specs.get(key) if isinstance(prop_specs.get(key), dict) else {}
            return bool(re.search('price|cost|amount|budget|spend|money', key + ' ' + str(field_spec.get('description') or ''), re.I))
        caps = [k for k, v in params.items() if k in prop_specs and isinstance(v, (int, float)) and (not isinstance(v, bool)) and re.search('max|most|under|below|limit|cap|upto|up_to', k, re.I) and priced(k)]
        if not caps or not self.price_field:
            return ''
        caps.sort(key=lambda k: params[k])
        cap = params[caps[0]]
        over = [i for i, _ in self.listing_records(tool_result) if (self.price_for(i) or 0) > cap]
        if not over:
            return ''
        modes = {k: v for k, v in params.items() if isinstance(v, str) and isinstance((prop_specs.get(k) or {}).get('enum'), list)}
        others = {k: [o for o in prop_specs[k]['enum'] if o != v] for k, v in modes.items()}
        hint = '; list the category with ' + ', '.join((f'{k}={as_text(o[0])}' for k, o in others.items() if o)) + f' and the same {caps[0]} to see only cheaper listings' if any(others.values()) else ''
        return f'\n[Wren] This search did not apply {caps[0]}={cap:g}: {len(over)} of its rows cost more{hint}.'

    def cheapest_shown(self, observations: list[Any]) -> str:
        if self.intake.get('pick') != 'cheapest' or not self.price_field:
            return ''
        hits_by_want = {i for met in observations for i, _ in self.listing_records(met)}
        row_list = [(self.price_for(i), i) for i in hits_by_want if self.can_buy(i) and self.price_for(i) is not None]
        if len(row_list) < 2:
            return ''
        return '\n[Wren] buyable rows here, cheapest first: ' + ', '.join((f'{i} ({p:g})' for p, i in sorted(row_list)[:6]))

    def hold_for_more(self) -> str:
        if self.more_hold_done or self.intake.get('pick') != 'cheapest' or self.steps_left() < 14 + 2 * len(self.want_list):
            return ''
        self.more_hold_done = True
        rows_out = [f'{i} at {self.price_for(i):g}' for i in self.basket_ids() if self.price_for(i) is not None]
        return f"Not run yet (no step used). {self.steps_left()} steps are left and the pick is the cheapest. Searches match a listing's title, so a requirement's value (often only in the attributes) finds few of the fitting listings, and rows come in no price order. Before checking out, run a few more searches for each item, inside its category when the search takes one, under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line" + (f" ({'; '.join(rows_out)})" if rows_out else '') + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'

    def read_details(self, pids: list[str], kept_lines: int) -> list[str]:
        tool_id = self.detail_tool()
        id_param = self.id_argument(tool_id)
        if not tool_id or not id_param:
            return []
        text_fields = []
        for pid in [i for i in pids if i in self.listings and i not in self.read_records and (not self.is_gone(i))]:
            if self.steps_left() <= kept_lines or self.concluded or self.detail_sources.get(tool_id) is False:
                break
            fault_list: list[str] = []
            params = conform({id_param: pid}, self.tool_schemas.get(tool_id), 'args', fault_list)
            if fault_list:
                break
            reply_obj = self.store_call(tool_id, params)
            if reply_obj is None:
                break
            met = reply_obj.get('observation')
            self.digest(tool_id, met)
            self.remember_view(tool_id, met)
            if isinstance(met, dict) and self.id_of(met):
                self.detail_sources[tool_id] = self.detail_sources.get(tool_id, False) or bool(facts_of(met))
            text_fields.append(f'{pid}: ' + self.render(met)[:1500])
        if text_fields:
            self.journal.append({'kind': 'fill_facts', 'n': len(text_fields)})
        return text_fields

    def cheaper_candidates(self) -> list[str]:
        if self.intake.get('pick') != 'cheapest':
            return []
        stamp = self.basket_key() + f'#{self.need_version}'
        shelf_key = self.shelf_field()
        if stamp in self.cheap_marks or not shelf_key:
            return []
        self.cheap_marks.add(stamp)
        basket_tool = self.basket_ids()
        choices: list[str] = []
        for row_line in basket_tool:
            price = self.price_for(row_line)
            kind_txt = str((self.listings.get(row_line) or {}).get(shelf_key) or '').strip().lower()
            if price is None or not kind_txt:
                continue
            candidates = [i for i in self.listing_order if i not in basket_tool and i not in self.read_records and self.can_buy(i) and (str((self.listings.get(i) or {}).get(shelf_key) or '').strip().lower() == kind_txt) and ((self.price_for(i) if self.price_for(i) is not None else float('inf')) < price)]
            candidates.sort(key=lambda i: self.price_for(i) or 0.0)
            choices += [i for i in candidates[:2] if i not in choices]
        return self.read_details(choices[:CHEAP_READS], self.closing_room() + 4)

    def digest(self, tool_name: str, tool_result: Any) -> str:
        meaning = self.digest_rows(tool_name, tool_result)
        if tool_name == self.tool_roles.get('search') and self.pending_search is not None:
            pending, self.pending_search = (self.pending_search, None)
            try:
                pids = [i for i, _ in self.listing_records(tool_result)]
                if pids:
                    self.query_hits = (self.query_hits + [(pending, pids)])[-24:]
            except Exception as err:
                self.journal.append({'kind': 'hits_error', 'error': f'{type(err).__name__}: {err}'[:200]})
        return meaning

    def digest_rows(self, tool_name: str, tool_result: Any) -> str:
        self.take_in(tool_result)
        rows_out = self.stray_lines(tool_result)
        if not rows_out:
            return ''
        meaning = self.apply_notice(rows_out)
        if meaning:
            self.pending_notices.append(f'(seen on {tool_name}) {meaning}')
        return meaning

    def broaden(self) -> None:
        self.guarded(self.search_unseen_wants, None)
        tool_id = self.tool_roles.get('compare')
        spec_doc = self.tool_schemas.get(tool_id or '') or {}
        prop_specs = spec_doc.get('properties') or {}
        id_param = next((k for k, v in prop_specs.items() if isinstance(v, dict) and 'array' in schema_kinds(v)), None)
        if not tool_id or not id_param or any((k != id_param for k in spec_doc.get('required') or [])) or self.multi_view_is_bare():
            self.broaden_one()
            return
        if self.steps_left() <= 12:
            return
        cap = prop_specs[id_param].get('maxItems') if isinstance(prop_specs[id_param].get('maxItems'), int) else 4
        ranked = self.want_groups()
        share = max(1, cap // max(1, len(ranked)))
        choices: list[str] = []
        for pids in ranked:
            choices += [i for i in pids if i not in self.read_records][:share]
        choices = choices[:cap]
        if len(choices) < 2:
            return
        fault_list: list[str] = []
        params = conform({id_param: choices}, spec_doc, 'args', fault_list)
        if fault_list:
            return
        before = len(self.read_records)
        reply_obj = self.store_call(tool_id, params)
        if reply_obj is not None:
            met = reply_obj.get('observation')
            self.digest(tool_id, met)
            self.remember_view(tool_id, met)
            row_list = self.listing_records(met)
            if row_list:
                self.detail_sources[tool_id] = self.detail_sources.get(tool_id, False) or any((facts_of(r) for _, r in row_list))
            self.journal.append({'kind': 'widen', 'ids': choices, 'opened': len(self.read_records) - before})
            if row_list and self.detail_sources.get(tool_id) is False:
                self.broaden_one()
                return
            if len(self.read_records) == before and len(choices) > 2 and (self.steps_left() > 12):
                fault_list = []
                params = conform({id_param: choices[:2]}, spec_doc, 'args', fault_list)
                if not fault_list:
                    reply_obj = self.store_call(tool_id, params)
                    if reply_obj is not None:
                        self.digest(tool_id, reply_obj.get('observation'))

    def broaden_one(self) -> None:
        tool_id = self.detail_tool()
        id_param = self.id_argument(tool_id)
        if not tool_id or not id_param:
            return
        wanted = {w for _, need_txt in self.live_needs() for w in re.findall('[a-z0-9]{3,}', need_txt.split(':', 1)[-1].lower())}
        each = READS_PER_ITEM + 2 if len(self.want_list) == 1 else READS_PER_ITEM if len(self.want_list) <= 2 else 4
        queues: list[list[str]] = []
        heads: list[int] = []
        for pids in self.want_groups():
            opened = sum((1 for i in pids if i in self.read_records))
            candidates = [i for i in pids if i not in self.read_records and self.can_buy(i)]
            if self.intake.get('pick') == 'cheapest':
                candidates.sort(key=lambda i: self.price_for(i) if self.price_for(i) is not None else float('inf'))
            candidates.sort(key=lambda i: -len(wanted & set(re.findall('[a-z0-9]{3,}', as_text(self.listings.get(i) or {}).lower()))))
            queues.append(candidates[:max(0, each - opened)])
            heads.append(max(0, 4 - opened))
        first_val = [queue[k] for k in range(4) for queue, head_word in zip(queues, heads) if k < min(head_word, len(queue))]
        more = [queue[k] for k in range(each) for queue, head_word in zip(queues, heads) if head_word <= k < len(queue)]
        cap = max(5, 2 * len(queues) + 2)
        floor = self.closing_room() + self.messages_left() + 3
        read = self.read_details(first_val[:cap][:8], min(22, max(floor, self.steps_left() - 8)))
        self.read_details((first_val[cap:] + more)[:max(0, 10 - len(read))], min(30, max(floor + 2, self.steps_left() - 4)))

    def search_pattern(self) -> tuple[str, dict[str, Any], str] | None:
        tool_id = self.tool_roles.get('search')
        spec_doc = self.tool_schemas.get(tool_id or '') or {}
        prop_specs = spec_doc.get('properties') or {}
        text_fields = [k for k, v in prop_specs.items() if isinstance(v, dict) and 'string' in schema_kinds(v) and ('enum' not in v)]
        named_ids = [k for k in text_fields if re.search('(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword', k, re.I)]
        for params in reversed(self.searches_made):
            fields = [k for k in named_ids or text_fields if isinstance(params.get(k), str) and re.search('[a-z]', params[k], re.I)]
            if not named_ids:
                tags = {str(r.get(self.shelf_field()) or '').strip().lower() for r in self.listings.values()} - {''} if self.shelf_field() else set()
                fields = [k for k in fields if len({str(a.get(k)) for a in self.searches_made if a.get(k)}) >= 2 and any((len(str(a.get(k)).split()) >= 2 for a in self.searches_made)) and (not any((str(a.get(k) or '').strip().lower() in tags for a in self.searches_made)))]
            if fields and tool_id:
                field = fields[0]
                kept_lines = {k: v for k, v in params.items() if k != field and (isinstance((prop_specs.get(k) or {}).get('enum'), list) or k in (spec_doc.get('required') or []))}
                shelf_key = self.shelf_field()
                shelves = {plain_words(want_obj.shelf or '') for want_obj in self.want_list} | {plain_words(want_obj.name) for want_obj in self.want_list}
                shelves |= {plain_words(str(row_doc.get(shelf_key) or '')) for row_doc in self.listings.values()} if shelf_key else set()
                kept_lines = {k: v for k, v in kept_lines.items() if k in (spec_doc.get('required') or []) or not isinstance(v, str) or (plain_words(v) not in shelves and (not any(({w for w in word_roots(v) if len(w) >= 4} & {w for w in word_roots(s) if len(w) >= 4} for s in shelves if s))))}
                worded = [a for a in self.searches_made if isinstance(a.get(field), str) and re.search('[a-z]', a[field], re.I)]
                for k in list(kept_lines):
                    if isinstance((prop_specs.get(k) or {}).get('enum'), list):
                        used = [as_text(a[k]) for a in worded if k in a]
                        if used:
                            top_pick = max(used, key=used.count)
                            kept_lines[k] = next((a[k] for a in worded if k in a and as_text(a[k]) == top_pick))
                return (tool_id, kept_lines, field)
        return None

    def value_terms(self, slot_no: int) -> list[str]:
        acc = []
        for n, need_txt in self.live_needs():
            if n != slot_no or need_txt in [m for _, m in self.chain_needs] or ':' not in need_txt:
                continue
            val = need_txt.split(':', 1)[1].strip()
            if not val or re.search('\\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', val, re.I):
                continue
            if self.already_stated(slot_no, need_txt):
                continue
            val = re.sub('^(?:(?:must|should|has to|have to)\\s+(?:be|have|include)\\s+|includes?\\s+|including\\s+|contains?\\s+|containing\\s+|with\\s+|has\\s+|having\\s+|is\\s+|are\\s+|be\\s+)', '', val, flags=re.I).strip() or val
            acc.append(val)
        return acc

    def own_shelf(self, slot_no: int) -> str:
        shelf_key = self.shelf_field()
        if not shelf_key:
            return ''
        want_obj = self.want_list[slot_no - 1]
        if want_obj.shelf and any((plain_words(str(r.get(shelf_key) or '')) == plain_words(want_obj.shelf) for r in self.listings.values())):
            return plain_words(want_obj.shelf)
        tags = [plain_words(str(r.get(shelf_key) or '')) for r in self.read_records.values() if self.belongs_to(want_obj, str(r.get(shelf_key) or ''))]
        return max(set(tags), key=tags.count) if tags else ''

    def sweep_scope(self, slot_no: int, base: dict[str, Any], field: str='') -> dict[str, Any]:
        tool_id = self.tool_roles.get('search')
        prop_specs = (self.tool_schemas.get(tool_id or '') or {}).get('properties') or {}
        shelf_key = self.shelf_field()
        if not shelf_key:
            return base
        want_obj = self.want_list[slot_no - 1]
        tags = [str(r.get(shelf_key) or '') for r in self.read_records.values() if self.belongs_to(want_obj, str(r.get(shelf_key) or ''))]
        if not tags:
            return base
        label_txt = max(set(tags), key=tags.count)
        seen_labels = {plain_words(str(r.get(shelf_key) or '')) for r in self.listings.values()} - {''}
        shelves = seen_labels | {plain_words(it.shelf or '') for it in self.want_list} - {''}
        for params in reversed(self.searches_made):
            for key, val in params.items():
                field_spec = prop_specs.get(key) if isinstance(prop_specs.get(key), dict) else {}
                if key in base or key == field or (not isinstance(val, str)) or ('string' not in schema_kinds(field_spec)):
                    continue
                if plain_words(val) in shelves:
                    opts = field_spec.get('enum') if isinstance(field_spec.get('enum'), list) else None
                    if opts:
                        choice = next((o for o in opts if isinstance(o, str) and plain_words(o) == plain_words(label_txt)), None)
                        if choice is None:
                            continue
                        return {**base, key: choice}
                    return {**base, key: label_txt}
        return base

    def walk_category(self) -> str:
        if self.__dict__.get('walked') or self.intake.get('pick') != 'cheapest' or len(self.want_list) != 1 or self.concluded:
            return ''
        proposal = self.search_pattern()
        if not proposal or not self.detail_tool():
            return ''
        self.walked = True
        tool_id, base, field = proposal
        spec_doc = self.tool_schemas.get(tool_id) or {}
        prop_specs = spec_doc.get('properties') or {}
        params = dict(self.sweep_scope(1, base, field))
        shelf_key = self.shelf_field()
        tags = [str(r.get(shelf_key) or '') for r in self.read_records.values() if shelf_key and r.get(shelf_key)]
        label_txt = max(set(tags), key=tags.count) if tags else ''
        seen_labels = ({plain_words(str(r.get(shelf_key) or '')) for r in self.listings.values()} | {plain_words(label_txt)}) - {''}
        filtered = lambda a: any((k != field and isinstance(v, str) and (plain_words(v) in seen_labels) for k, v in a.items()))
        if not filtered(params) and label_txt:
            for past in reversed(self.searches_made):
                key = next((k for k, v in past.items() if k != field and isinstance(v, str) and (plain_words(v) in seen_labels) and isinstance(prop_specs.get(k), dict) and ('string' in schema_kinds(prop_specs[k])) and ('enum' not in prop_specs[k])), None)
                if key:
                    params[key] = past[key]
                    break
            if not filtered(params):
                key = next((k for k, field_spec in prop_specs.items() if k != field and isinstance(field_spec, dict) and ('string' in schema_kinds(field_spec)) and ('enum' not in field_spec) and re.search('categor|group|shelf|department|section|dept|aisle', k + ' ' + str(field_spec.get('description') or ''), re.I)), None)
                if key:
                    params[key] = label_txt
        if shelf_key and (not filtered(params)):
            self.journal.append({'kind': 'category_walk', 'stopped': 'no category argument'})
            return ''
        browse = False
        for key, field_spec in prop_specs.items():
            opts = field_spec.get('enum') if isinstance(field_spec, dict) else None
            choice = next((o for o in opts or [] if isinstance(o, str) and re.search('browse|list|all|categor|filter|shelf', o, re.I)), None)
            if choice is not None:
                params[key] = choice
                browse = True
        if browse and field not in (spec_doc.get('required') or []):
            params.pop(field, None)
        else:
            params[field] = self.want_list[0].name
        want_obj = self.want_list[0]
        needs = self.want_needs(1)
        clean = lambda i: i in self.read_records and bool(needs) and RecordJudge.clear(RecordJudge.sheet(needs, self.read_records[i]))
        own_map = self.own_shelf(1) or plain_words(label_txt)
        kind_ok = lambda i: not shelf_key or (plain_words(str((self.listings.get(i) or {}).get(shelf_key) or '')) == own_map if own_map else self.belongs_to(want_obj, str((self.listings.get(i) or {}).get(shelf_key) or '')))
        if self.steps_left() - self.closing_room() < WALK_OPEN_CAP + 6:
            return ''
        fault_list: list[str] = []
        one_call = conform(params, spec_doc, 'args', fault_list)
        self.journal.append({'kind': 'category_walk_args', 'args': one_call, 'faults': fault_list[:3], 'base': base, 'searched': self.searches_made[-3:]})
        if fault_list:
            return ''
        reply_obj = self.store_call(tool_id, one_call)
        if reply_obj is None:
            return ''
        met = reply_obj.get('observation')
        self.digest(tool_id, met)
        page = self.page_state(tool_id, met)
        total = page[0] if page else None
        pages = 1
        if not page or page[3] is None:
            self.journal.append({'kind': 'category_walk', 'total': total, 'pages': pages, 'stopped': 'one page'})
            return ''
        if isinstance(page[3], int) and (not isinstance(page[3], bool)):
            furthest = max((int(a.get(page[2]) or 0) for a in self.searches_made if isinstance(a.get(page[2]), int) and (not isinstance(a.get(page[2]), bool)) and all((a.get(k) == v for k, v in one_call.items() if k != page[2]))), default=-1)
            offsets = page[3] == page[1]
            if furthest >= page[3]:
                page = (page[0], page[1], page[2], furthest + (page[1] if offsets else 1))
        budget_pages = min(WALK_PAGE_CAP, self.steps_left() - self.closing_room() - WALK_OPEN_CAP - 4)
        if budget_pages < 2:
            self.journal.append({'kind': 'category_walk', 'total': total, 'pages': pages, 'stopped': 'steps'})
            return ''
        if isinstance(total, int) and page[1] and (total > 2 * budget_pages * page[1]):
            self.journal.append({'kind': 'category_walk', 'total': total, 'pages': pages, 'stopped': 'too large', 'reach': budget_pages * page[1]})
            return ''
        while page and page[3] is not None and (pages < budget_pages) and (not self.concluded) and (not isinstance(total, int) or not isinstance(page[3], int) or page[3] < total):
            fault_list = []
            more = conform({**one_call, page[2]: page[3]}, spec_doc, 'args', fault_list)
            if fault_list:
                break
            reply_obj = self.store_call(tool_id, more)
            if reply_obj is None:
                break
            met = reply_obj.get('observation')
            self.digest(tool_id, met)
            pages += 1
            page = self.page_state(tool_id, met)
        cap = self.spend_limit if isinstance(self.spend_limit, (int, float)) else None
        qty = want_obj.qty if self.intake.get('budget_scope') != 'each' else 1
        top_pick = min((self.price_for(i) for i in self.read_records if kind_ok(i) and clean(i) and self.can_buy(i) and (not self.is_gone(i)) and (self.price_for(i) is not None)), default=None)
        candidates = [i for i in self.listing_order if i not in self.read_records and i not in self.pages_opened and kind_ok(i) and self.can_buy(i) and (not self.is_gone(i)) and (self.price_for(i) is not None) and (cap is None or self.price_for(i) * qty <= cap) and (top_pick is None or self.price_for(i) < top_pick)]
        candidates.sort(key=lambda i: self.price_for(i))
        opened: list[str] = []
        rx_hit = None
        for pid in candidates[:WALK_OPEN_CAP]:
            if self.steps_left() - self.closing_room() < 4 or self.concluded:
                break
            if not self.read_details([pid], self.closing_room() + 3):
                break
            opened.append(pid)
            if clean(pid):
                rx_hit = pid
                break
        whole = page is None or page[3] is None or (isinstance(total, int) and isinstance(page[3], int) and (page[3] >= total))
        self.journal.append({'kind': 'category_walk', 'total': total, 'pages': pages, 'whole': whole, 'best_before': top_pick, 'pool': candidates[:8], 'opened': opened, 'clean': rx_hit})
        words_out = f"[Wren listed {('the whole' if whole else 'more of the')} {want_obj.name} category ({total} rows, {pages} more pages) and opened the unopened rows cheaper than the best fitting record, cheapest first] "
        if rx_hit:
            return words_out + f'{rx_hit} at {self.price_for(rx_hit):g} meets every requirement by its own attributes and is the cheapest such listing seen; prefer it unless its record says otherwise.'
        if opened:
            return words_out + 'opened ' + ', '.join(opened) + (f'; none met every requirement, so the best remains at {top_pick:g}' if top_pick is not None else '; none met every requirement') + '.'
        return words_out + ('no unopened row is cheaper than the best fitting record' if top_pick is not None else 'no affordable row of that label is unread') + '.'

    def search_values(self, numbers: list[int]) -> str:
        proposal = self.search_pattern()
        if not proposal or self.steps_left() < 22:
            return ''
        tool_id, base, field = proposal
        met: list[str] = []
        budget = 10 if len(self.want_list) == 1 else 8
        for slot_no in numbers:
            vals = self.value_terms(slot_no)
            want_obj = self.want_list[slot_no - 1]
            fresh = [v for v in vals if any((':' in t and t.split(':', 1)[1].strip() == v for t in want_obj.told))]
            if not vals or not want_obj.told or budget <= 2:
                continue
            base = self.sweep_scope(slot_no, proposal[1], field)
            vals = [v for v in vals if v in fresh] + [v for v in vals if v not in fresh]
            stated = [i for i, v in enumerate(vals) if v not in fresh]
            drop = stated[-1] if stated else 0
            tries = [vals, vals[:drop] + vals[drop + 1:]] if len(vals) > 1 else [vals]
            heads = [v.split(',')[0].strip() or v for v in vals]
            if heads != vals:
                tries.insert(1, heads)
            if base != proposal[1]:
                tries = [(terms, base) for terms in tries] + [(vals, proposal[1])]
            else:
                tries = [(terms, base) for terms in tries]
            done_tries: set[str] = set()
            for terms, scope in tries[:3]:
                if not terms or as_text([terms, scope]) in done_tries:
                    continue
                done_tries.add(as_text([terms, scope]))
                fault_list: list[str] = []
                params = conform({**scope, field: ' '.join(terms)}, self.tool_schemas.get(tool_id), 'args', fault_list)
                if fault_list or self.concluded:
                    break
                reply_obj = self.store_call(tool_id, params)
                budget -= 1
                if reply_obj is None:
                    break
                tool_result = reply_obj.get('observation')
                self.digest(tool_id, tool_result)
                shelf_key = self.shelf_field()
                hits_by_want = [i for i, _ in self.listing_records(tool_result)]
                layout = self.page_state(tool_id, tool_result)
                total = layout[0] if layout else len(hits_by_want)
                if layout and layout[3] is not None and (layout[1] < total <= 2 * layout[1]) and (budget > 3) and (self.steps_left() - self.closing_room() > 20):
                    more = conform({**params, layout[2]: layout[3]}, self.tool_schemas.get(tool_id), 'args', fault_list)
                    again = self.store_call(tool_id, more) if not fault_list else None
                    budget -= 1
                    if again is not None:
                        self.digest(tool_id, again.get('observation'))
                        hits_by_want += [i for i, _ in self.listing_records(again.get('observation')) if i not in hits_by_want]
                cap = self.spend_limit if isinstance(self.spend_limit, (int, float)) else None
                prices = [self.price_for(i) for i in hits_by_want]
                if cap is not None and prices and all((p is not None and p > 10 * cap for p in prices)):
                    cap = None
                qty = want_obj.qty if self.intake.get('budget_scope') != 'each' else 1
                own_map = self.own_shelf(slot_no)
                kind_ok = lambda i: not shelf_key or (plain_words(str((self.listings.get(i) or {}).get(shelf_key) or '')) == own_map if own_map else self.belongs_to(want_obj, str((self.listings.get(i) or {}).get(shelf_key) or '')))
                row_list = [i for i in hits_by_want if self.can_buy(i) and kind_ok(i) and (cap is None or (self.price_for(i) or 0) * qty <= cap)]
                met.append(f'{as_text(params.get(field))}: ' + (', '.join(row_list[:8]) or 'no row of that kind'))
                if row_list:
                    short = bool(layout) and len(hits_by_want) <= 24 and (total <= 24)
                    fresh_rows = [i for i in row_list if i not in self.read_records]
                    fresh_rows.sort(key=lambda i: self.price_for(i) if self.price_for(i) is not None else float('inf'))
                    room = max(0, self.steps_left() - self.closing_room() - 14)
                    take = min(len(fresh_rows), 6 if short else 3, budget, room)
                    opened = self.read_details(fresh_rows[:take], self.closing_room() + 14)
                    budget -= len(opened)
                    if opened:
                        met.append(('opened every affordable row of its kind in this short result, cheapest first: ' if short and len(opened) == len(fresh_rows) else 'opened: ') + ' | '.join(opened)[:3000])
                    needs = [need_txt for n, need_txt in self.live_needs() if n == slot_no and need_txt not in [m for _, m in self.chain_needs]]
                    if not needs or any((RecordJudge.clear(RecordJudge.sheet(needs, self.read_records[i])) for i in row_list if i in self.read_records)):
                        break
        if not met:
            return ''
        self.journal.append({'kind': 'value_search', 'seen': [s[:200] for s in met]})
        return "[Wren searched each such item's requirement values together inside its category and read its rows (titles often hide what the attributes say)] " + '\n'.join(met)

    def value_evidence(self, slot_no: int, pids: list[str] | None=None, every: bool=True) -> dict[str, dict[str, str]]:
        want_obj = self.want_list[slot_no - 1]
        vals: list[tuple[str, str]] = []
        for k, need_txt in self.live_needs():
            if k != slot_no or ':' not in need_txt or need_txt in [m for _, m in self.chain_needs]:
                continue
            prop_words, val = (part.strip() for part in need_txt.split(':', 1))
            if not val or re.search('\\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', val, re.I):
                continue
            vals.append((prop_words, val))
        if not vals:
            return {}
        shelf_key = self.shelf_field()
        cap = self.spend_limit if isinstance(self.spend_limit, (int, float)) and self.intake.get('budget_scope') != 'each' else None
        acc: dict[str, dict[str, str]] = {}
        for pid in pids if pids is not None else list(self.listing_order):
            row_doc = self.read_records.get(pid) or self.listings.get(pid) or {}
            if not row_doc or self.is_gone(pid) or (not self.can_buy(pid)):
                continue
            if shelf_key and (not self.belongs_to(want_obj, str(row_doc.get(shelf_key) or ''))):
                continue
            if cap is not None and (self.price_for(pid) or 0) * want_obj.qty > cap:
                continue
            block = facts_of(row_doc) or {}
            where: dict[str, str] = {}
            moved = sent_elsewhere(' '.join((str(v) for v in row_doc.values() if isinstance(v, str))))
            for prop_words, val in vals:
                stamp = ''
                delivered = any((key_names_property(prop_words, p) or key_names_property(p, prop_words) for p in moved))
                for k, v in [] if delivered else block.items():
                    if isinstance(v, (dict, list)):
                        continue
                    if name_covers(word_roots(prop_words), word_roots(k)):
                        stamp = 'key' if names_value(val, str(v)) else 'fails'
                        if stamp == 'key':
                            break
                    elif not stamp and names_value(val, str(v)):
                        stamp = 'other'
                if stamp == 'fails':
                    break
                if not stamp and (not delivered):
                    rest = ' '.join((str(v) for k, v in row_doc.items() if isinstance(v, str) and (block is None or k not in block)))
                    stamp = 'text' if names_value(val, rest) else ''
                if not stamp:
                    if every:
                        break
                    continue
                where[val] = stamp
            else:
                if where:
                    acc[pid] = where
        return acc

    def rows_naming_values(self, slot_no: int, pids: list[str] | None=None, every: bool=True) -> list[str]:
        marks = self.value_evidence(slot_no, pids, every)
        unread = [i for i in marks if i not in self.read_records and i not in self.pages_opened]
        unread.sort(key=lambda i: (-len(marks[i]), -sum((1 for m in marks[i].values() if m == 'key')), self.price_for(i) if self.price_for(i) is not None else 1e+18))
        return unread

    def recovery_variants(self, params: dict[str, Any], base: dict[str, Any], field: str, vals: list[str]) -> list[dict[str, Any]]:
        acc: list[dict[str, Any]] = []
        bare = {**base, field: vals[0]} if vals else None
        if bare and bare != params:
            acc.append(bare)
        prop_specs = (self.tool_schemas.get(self.tool_roles.get('search') or '') or {}).get('properties') or {}
        last = acc[-1] if acc else dict(params)
        for key, field_spec in prop_specs.items():
            opts = field_spec.get('enum') if isinstance(field_spec, dict) else None
            if isinstance(opts, list) and len(opts) >= 2 and (key in last):
                alt = next((o for o in opts if o != last[key]), None)
                if alt is not None:
                    acc.append({**last, key: alt})
                    break
        return acc[:2]

    def reread_by_value(self, numbers: list[int], tag: str, floor: int=0) -> list[str]:
        proposal = self.search_pattern()
        need_txt = max(self.closing_room() + 5, self.closing_room() + 1 + len(numbers) + 3, floor + 3)
        if not proposal or self.past_clock(30) or self.steps_left() < need_txt:
            return []
        tool_id, base, field = proposal
        opened: list[str] = []
        for slot_no in numbers:
            if (tag, slot_no) in self.recovery_done or self.steps_left() < need_txt:
                continue
            self.recovery_done.add((tag, slot_no))
            want_obj = self.want_list[slot_no - 1]
            vals = self.value_terms(slot_no)
            if not vals:
                continue
            noun = plain_words(want_obj.name).split()[-1:] if want_obj.name else []
            terms = ' '.join(vals[:2] + [w for w in noun if w not in ' '.join(vals[:2]).lower()])
            fault_list: list[str] = []
            params = conform({**base, field: terms}, self.tool_schemas.get(tool_id), 'args', fault_list)
            if fault_list or self.concluded:
                continue
            reply_obj = self.store_call(tool_id, params)
            if reply_obj is None:
                continue
            tool_result = reply_obj.get('observation')
            self.digest(tool_id, tool_result)
            if not self.listing_records(tool_result) and self.steps_left() >= need_txt + 1 and (not self.concluded):
                for retry in self.guarded(self.recovery_variants, [], params, base, field, vals):
                    fault_list = []
                    again = conform(retry, self.tool_schemas.get(tool_id), 'args', fault_list)
                    reply_obj = self.store_call(tool_id, again) if not fault_list else None
                    if reply_obj is None:
                        break
                    tool_result = reply_obj.get('observation')
                    self.digest(tool_id, tool_result)
                    params = again
                    if self.listing_records(tool_result) or self.steps_left() < need_txt + 1 or self.concluded:
                        break
            pages = 0
            page_ids = [i for i, _ in self.listing_records(tool_result)]
            row_list = [i for i in self.rows_naming_values(slot_no) if i in page_ids]
            while not row_list and (not self.value_evidence(slot_no, page_ids)) and (pages < 2) and (self.steps_left() >= need_txt + 2) and (not self.concluded):
                more = self.page_state(tool_id, tool_result)
                if not more or more[3] is None or more[0] <= more[1]:
                    break
                fault_list = []
                params = conform({**params, more[2]: more[3]}, self.tool_schemas.get(tool_id), 'args', fault_list)
                reply_obj = self.store_call(tool_id, params) if not fault_list else None
                if reply_obj is None:
                    break
                tool_result = reply_obj.get('observation')
                self.digest(tool_id, tool_result)
                pages += 1
                page_ids += [i for i, _ in self.listing_records(tool_result)]
                row_list = [i for i in self.rows_naming_values(slot_no) if i in page_ids]
            if not row_list:
                row_list = self.rows_naming_values(slot_no) or self.rows_naming_values(slot_no, page_ids, every=False)
            json_got = self.read_details(row_list[:2], max(self.closing_room() + 3, need_txt - 2))
            opened += [i for i in row_list[:2] if i in self.read_records or i in self.pages_opened]
            self.journal.append({'kind': 'recover_reads', 'tag': tag, 'item': slot_no, 'query': terms, 'pages': pages, 'opened': row_list[:2], 'read': len(json_got)})
        return opened

    def want_for_label(self, label_txt: str) -> int:
        if not label_txt:
            return 0
        terms = plain_words(label_txt)
        long = root_set(label_txt)
        top_pick, top = (0, 0.0)
        for slot_no, want_obj in enumerate(self.want_list, 1):
            if terms == plain_words(want_obj.shelf or ''):
                points = 3.0
            elif terms == plain_words(want_obj.name):
                points = 2.0
            else:
                mine = root_set(want_obj.shelf or '') | root_set(want_obj.name)
                points = len(long & mine) / max(1, len(long | mine))
            if points > top:
                top_pick, top = (slot_no, points)
        if top_pick and 0 < top < 0.99 and any((self.label_matches_whole(top_pick, alt) for alt in self.labels_seen())):
            return 0
        return top_pick

    def label_matches_whole(self, slot_no: int, label_txt: str) -> bool:
        want_obj = self.want_list[slot_no - 1]
        terms, long = (plain_words(label_txt), root_set(label_txt))
        return bool(label_txt) and (terms == plain_words(want_obj.shelf or '') or terms == plain_words(want_obj.name) or (bool(long) and long == (root_set(want_obj.shelf or '') or root_set(want_obj.name))))

    def labels_seen(self) -> set[str]:
        cache = getattr(self, '_label_cache', None)
        if cache and cache[0] == len(self.listings):
            return cache[1]
        key = self.shelf_field()
        hits_by_want = {str(row_doc.get(key) or '').strip() for row_doc in self.listings.values() if key} - {''}
        self._label_cache = (len(self.listings), hits_by_want)
        return hits_by_want

    def belongs_to(self, want_obj: Want, label_txt: str) -> bool:
        return bool(label_txt) and want_obj in self.want_list and (self.want_for_label(label_txt) == self.want_list.index(want_obj) + 1)

    def record_count_for(self, slot_no: int) -> int:
        shelf_key = self.shelf_field()
        want_obj = self.want_list[slot_no - 1]
        return sum((1 for rec_doc in self.read_records.values() if shelf_key and self.belongs_to(want_obj, str(rec_doc.get(shelf_key) or '')) and facts_of(rec_doc)))

    def evidence_lines(self, tool_result: Any) -> str:
        pids: list[str] = []

        def walk(obj: Any, level: int) -> None:
            if level > 6:
                return
            if isinstance(obj, dict):
                pid = self.id_of(obj, False, level)
                if pid and facts_of(obj) and (pid not in pids):
                    pids.append(pid)
                    return
                for val in obj.values():
                    walk(val, level + 1)
            elif isinstance(obj, list):
                for val in obj:
                    walk(val, level + 1)
        walk(tool_result, 0)
        needs = self.live_needs()
        if not pids or not needs:
            return ''
        multi_arg = len(self.want_list) > 1
        rows_out = []
        for pid in pids[:12]:
            block = facts_of(self.read_records.get(pid) or {}) or {}
            pieces = []
            for slot_no, need_txt in needs[:8]:
                prop_words = need_txt.split(':', 1)[0].strip()
                val = need_txt.split(':', 1)[1].strip() if ':' in need_txt else ''
                named_ids = sorted([k for k in block if name_covers(word_roots(prop_words), word_roots(k))], key=lambda k: -len(word_roots(k)))
                top = len(word_roots(named_ids[0])) if named_ids else 0
                hits = [f'{k}={str(block[k])[:50]}' + (' [names the required value]' if val and names_value(val, str(block[k])) else '') + ('' if len(word_roots(k)) == top else ' [general key: the more specific key decides]') for k in named_ids]
                tag = f'(item {slot_no}) ' if multi_arg else ''
                if hits:
                    pieces.append(f'{tag}{prop_words} -> ' + ', '.join(hits[:3]))
            if pieces:
                rows_out.append(f'{pid}: ' + '; '.join(pieces))
        if not rows_out:
            return ''
        return "[evidence: keys whose names contain a requirement's property; one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property] " + ' | '.join(rows_out)

    def record_text(self, pid: str) -> str:
        rec_doc = self.read_records.get(pid) or self.listings.get(pid)
        return as_text(clipped_copy(rec_doc, 400))[:3000] if rec_doc else '(no record seen)'

    def other_listings(self, skip: list[str], ceiling: int=30) -> str:
        rows_out = []
        order = list(reversed(self.listing_order))
        if self.intake.get('pick') == 'cheapest':
            order.sort(key=lambda i: (i not in self.read_records, self.price_for(i) if self.price_for(i) is not None else float('inf')))
        for pid in order:
            if pid in skip or self.is_gone(pid):
                continue
            rec_doc = self.read_records.get(pid) or self.listings.get(pid)
            rows_out.append(('OPENED ' if pid in self.read_records or pid in self.pages_opened else 'ROW ') + as_text(clipped_copy(rec_doc, 160))[:900])
            if len(rows_out) >= ceiling:
                break
        return '\n'.join(rows_out)

    def listings_for(self, slot_no: int) -> list[str]:
        shelf_key = self.shelf_field()
        want_obj = self.want_list[slot_no - 1]
        return [i for i in self.listing_order if shelf_key and self.belongs_to(want_obj, str((self.listings.get(i) or {}).get(shelf_key) or ''))]

    def listings_in_scope(self, slot_no: int) -> list[str]:
        pids = self.listings_for(slot_no)
        if pids or len(self.want_list) != 1:
            return pids
        shelf_key = self.shelf_field()
        if not shelf_key:
            return list(self.listing_order)
        tags = [str(r.get(shelf_key) or '') for r in self.read_records.values() if r.get(shelf_key)]
        if not tags:
            return []
        label_txt = plain_words(max(set(tags), key=tags.count))
        return [i for i in self.listing_order if plain_words(str((self.listings.get(i) or {}).get(shelf_key) or '')) == label_txt]

    def price_band_reads(self, lost_ids: list[str]) -> list[str]:
        if self.intake.get('pick') != 'cheapest' or len(lost_ids) != 1 or (not self.detail_tool()):
            return []
        sold = self.price_for(lost_ids[0])
        numbers = [n for n in range(1, len(self.want_list) + 1) if lost_ids[0] in self.listings_in_scope(n)]
        if sold is None or len(numbers) != 1:
            return []
        slot_no = numbers[0]
        needs = self.want_needs(slot_no)
        clean = lambda i: i in self.read_records and bool(needs) and RecordJudge.clear(RecordJudge.sheet(needs, self.read_records[i]))
        backup_prices = [self.price_for(i) for i in self.listings_in_scope(slot_no) if i != lost_ids[0] and clean(i) and self.can_buy(i) and (not self.is_gone(i)) and (self.price_for(i) is not None)]
        ceiling = min(backup_prices) if backup_prices else None
        stems = {w for v in self.value_terms(slot_no) for w in word_roots(v) if len(w) >= 3}
        from_query = {i for params, pids in self.query_hits if any((stems & word_roots(str(v)) for v in params.values() if isinstance(v, str))) for i in pids}
        named_ids = set(self.rows_naming_values(slot_no, every=False))
        candidates = [i for i in self.listings_in_scope(slot_no) if i not in self.read_records and i not in self.pages_opened and (i != lost_ids[0]) and self.can_buy(i) and (not self.is_gone(i)) and (self.price_for(i) is not None) and (self.price_for(i) > sold) and (ceiling is None or self.price_for(i) < ceiling) and (i in from_query or i in named_ids)]
        candidates.sort(key=lambda i: self.price_for(i))
        room = self.steps_left() - self.closing_room() - 4
        take = min(len(candidates), 2, max(0, room))
        if take <= 0:
            return []
        json_got = self.read_details(candidates[:take], self.closing_room() + 4)
        hits_by_want = sorted((i for i in candidates[:take] if clean(i)), key=lambda i: self.price_for(i))
        self.journal.append({'kind': 'band_reads', 'sold': sold, 'ceiling': ceiling, 'pool': candidates[:6], 'opened': json_got, 'clean': hits_by_want})
        return hits_by_want

    @staticmethod
    def word_tokens(words_out: str) -> set[str]:
        return {w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w for w in re.findall('[a-z0-9]+', str(words_out).lower()) if len(w) >= 3 and w not in ('and', 'the', 'for', 'with', 'any')}

    def read_rows(self, row_list: list[str]) -> bool:
        multi_arg, joiner = (self.multi_view_arg(), self.tool_roles.get('compare'))
        tool_id = self.detail_tool()
        id_param = self.id_argument(tool_id)
        if multi_arg and joiner and (not self.multi_view_is_bare()):
            proposal = [(joiner, {multi_arg: row_list[:self.multi_view_size()]})]
        elif tool_id and id_param:
            proposal = [(tool_id, {id_param: i}) for i in row_list[:2]]
        else:
            return False
        if self.steps_left() < len(proposal) + 5:
            return False
        before = len(self.read_records)
        for tool_name, params in proposal:
            fault_list: list[str] = []
            params = conform(params, self.tool_schemas.get(tool_name), 'args', fault_list)
            if fault_list:
                return False
            reply_obj = self.store_call(tool_name, params)
            if reply_obj is None or self.concluded:
                break
            self.digest(tool_name, reply_obj.get('observation'))
        self.journal.append({'kind': 'audit_open', 'rows': row_list, 'opened': len(self.read_records) - before})
        return len(self.read_records) > before

    def fix_ids(self, tool_name: str, params: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if tool_name not in {self.tool_roles.get(r) for r in ('cart', 'open', 'offer', 'compare')} - {None} or not self.id_field:
            return (params, '')
        prop_specs = (self.tool_schemas.get(tool_name) or {}).get('properties') or {}
        free = [k for k, v in prop_specs.items() if isinstance(v, dict) and ('string' in schema_kinds(v) or 'array' in schema_kinds(v)) and ('enum' not in v)]
        if len(free) != 1 or (tool_name != self.tool_roles.get('cart') and (tool_name, free[0]) not in self.id_arguments):
            return (params, '')

        def fix(val: Any) -> tuple[Any, str]:
            if isinstance(val, str) and val.strip() in self.listings:
                return (val.strip(), '')
            if not isinstance(val, str) or looks_like_id(val, True) or len(val.strip()) < 8:
                return (val, '')
            low = val.strip().lower()
            hits = [i for i, row_doc in self.listings.items() if any((k != self.id_field and isinstance(v, str) and v.strip().lower().startswith(low) for k, v in row_doc.items()))]
            if len(hits) == 1:
                return (hits[0], '')
            if hits:
                return (val, f"several listings carry the text {val!r} ({', '.join(hits[:6])}); pass one listing id")
            return (val, f'{val!r} is not a listing id; pass the id from the result rows (their {self.id_field} field)')
        acc = dict(params)
        for key in free:
            if key not in params:
                continue
            fixed_args = [fix(v) for v in params[key]] if isinstance(params[key], list) else [fix(params[key])]
            problem = next((p for _, p in fixed_args if p), '')
            if problem:
                return (params, problem)
            acc[key] = [v for v, _ in fixed_args] if isinstance(params[key], list) else fixed_args[0][0]
        return (acc, '')

    def multi_view_arg(self) -> str | None:
        spec_doc = self.tool_schemas.get(self.tool_roles.get('compare') or '') or {}
        prop_specs = spec_doc.get('properties') or {}
        id_param = next((k for k, v in prop_specs.items() if isinstance(v, dict) and 'array' in schema_kinds(v)), None)
        if not id_param or any((k != id_param for k in spec_doc.get('required') or [])):
            return None
        return id_param

    def multi_view_size(self) -> int:
        field_spec = ((self.tool_schemas.get(self.tool_roles.get('compare') or '') or {}).get('properties') or {}).get(self.multi_view_arg() or '') or {}
        if isinstance(field_spec.get('maxItems'), int):
            return field_spec['maxItems']
        if self.multi_view_bad:
            return max(2, min(self.multi_view_bad - 1, max(self.multi_view_ok, 2)))
        return max(4, self.multi_view_ok)

    def merge_reads(self, call_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        opener, joiner, id_param = (self.tool_roles.get('open'), self.tool_roles.get('compare'), self.id_argument(self.tool_roles.get('open')))
        multi_arg = self.multi_view_arg()
        if not opener or not joiner or (not id_param) or (not multi_arg) or (opener == joiner):
            return call_list
        if self.multi_view_is_bare():
            return call_list
        choices: list[tuple[int, str]] = []
        for pos, one_call in enumerate(call_list):
            fn = one_call.get('function') or {}
            params = json_object(fn.get('arguments')) if isinstance(fn.get('arguments'), str) else None
            pid = str((params or {}).get(id_param) or '').strip()
            if fn.get('name') == opener and params and (set(params) == {id_param}) and (pid in self.listings) and (pid not in self.gone_at):
                choices.append((pos, pid))
        choices = choices[:self.multi_view_size()]
        if len(choices) < 2:
            return call_list
        first_val = choices[0][0]
        acc = []
        for pos, one_call in enumerate(call_list):
            if pos == first_val:
                acc.append({'id': one_call['id'], 'type': 'function', 'function': {'name': joiner, 'arguments': as_text({multi_arg: [i for _, i in choices]})}})
            elif pos in {i for i, _ in choices}:
                acc.append({'id': one_call['id'], 'joined': True})
            else:
                acc.append(one_call)
        self.journal.append({'kind': 'joined', 'ids': [i for _, i in choices]})
        return acc

class Interviewer:

    def messages_left(self) -> int:
        return self.message_cap - self.messages_sent - max(0, self.reserved_messages - self.step_messages)

    def search_unseen_wants(self) -> None:
        if len(self.want_list) < 2:
            return
        tool_id = self.tool_roles.get('search')
        spec_doc = self.tool_schemas.get(tool_id or '') or {}
        prop_specs = spec_doc.get('properties') or {}
        proposal = self.search_pattern()
        for slot_no, want_obj in enumerate(self.want_list, 1):
            if self.listings_for(slot_no) or self.steps_left() <= 14 or self.concluded:
                continue
            if proposal:
                _, base, field = proposal
                others = {plain_words(o.shelf or '') for o in self.want_list if o is not want_obj} | {plain_words(o.name) for o in self.want_list if o is not want_obj}
                params = {k: want_obj.shelf or want_obj.name if isinstance(v, str) and plain_words(v) in others else v for k, v in base.items()}
                params[field] = want_obj.name
            else:
                field = next((k for k, v in prop_specs.items() if isinstance(v, dict) and 'string' in schema_kinds(v) and ('enum' not in v)), None)
                if not tool_id or not field:
                    return
                params = {field: want_obj.name}
                for k in spec_doc.get('required') or []:
                    opts = (prop_specs.get(k) or {}).get('enum')
                    if k not in params and isinstance(opts, list) and opts:
                        params[k] = opts[0]
            fault_list: list[str] = []
            one_call = conform(params, spec_doc, 'args', fault_list)
            if fault_list or not tool_id:
                continue
            reply_obj = self.store_call(tool_id, one_call)
            if reply_obj is None:
                return
            self.digest(tool_id, reply_obj.get('observation'))
            self.journal.append({'kind': 'lane_fill', 'item': slot_no, 'args': one_call, 'rows': len(self.listings_for(slot_no))})

    def time_to_ask(self) -> bool:
        if self.interviewed or not self.tool_roles.get('ask') or self.messages_left() <= 0:
            return False
        if self.steps_left() <= self.messages_left() + 3:
            return False
        opened = len(self.read_records)
        if opened >= 3 * len(self.want_list) + 1:
            return True
        return self.steps_used >= max(8, self.step_limit // 4)

    def message_payload(self, words_out: str) -> dict[str, Any] | None:
        tool_name = self.tool_roles.get('ask')
        spec_doc = self.tool_schemas.get(tool_name or '') or {}
        prop_specs = spec_doc.get('properties') or {}
        text_fields = [k for k, v in prop_specs.items() if isinstance(v, dict) and 'string' in schema_kinds(v) and ('enum' not in v)]
        id_param = next((k for k in spec_doc.get('required') or [] if k in text_fields), None) or (text_fields[0] if text_fields else None)
        if not id_param:
            return None
        params: dict[str, Any] = {id_param: words_out}
        for key in spec_doc.get('required') or []:
            if key not in params:
                field_spec = prop_specs.get(key) or {}
                if isinstance(field_spec.get('enum'), list) and field_spec['enum']:
                    params[key] = field_spec['enum'][0]
                else:
                    return None
        fault_list: list[str] = []
        params = conform(params, spec_doc, 'args', fault_list)
        return None if fault_list else params

    def hold_interview(self) -> str:
        self.interviewed = True
        if self.messages_left() <= 0:
            return ''
        self.broaden()
        self.scarce_messages = self.messages_left() <= len(self.want_list) - len(self.tied_wants()) + 1
        queues = self.question_queues(self.messages_left())
        self.journal.append({'kind': 'ask_plan', 'tight': self.scarce_messages, 'bound': sorted(self.tied_wants()), 'messages': self.messages_left()})
        hits_by_want = {n: 0 for n in queues}
        spent = {n: 0 for n in queues}
        used: dict[int, set[str]] = {n: set() for n in queues}
        follow: list[tuple[int, list[str], bool]] = []
        delivered: list[tuple[int, str, str]] = []
        stated_now: list[str] = []
        mode = 'probe' if self.messages_left() >= PROBE_FLOOR and self.message_cap >= PROBE_FLOOR else 'single'
        if mode == 'probe' and len(self.want_list) > 1:
            mode = 'pack'
        while self.messages_left() > 0 and (not self.concluded) and (self.steps_left() > max(3, self.closing_room() + 2)) and (not self.past_clock(40)):
            joint: list[tuple[int, list[str]]] = []
            if follow:
                slot_no, tags, displayed = follow.pop(0)
            else:
                width = PROBE_WIDTH if mode == 'probe' and self.messages_left() >= 2 else PACK_SIZE if mode == 'pack' else 1
                if mode == 'pack' and self.messages_left() == 1:
                    joint = self.shared_question(queues, spent, used)
                slot_no, tags = joint[0] if joint else self.pick_next_question(queues, hits_by_want, spent, used, width)
                displayed = False
                if not tags:
                    if self.messages_left() > 0 and (not self.__dict__.get('asked_open')) and (self.steps_left() > max(3, self.closing_room() + 2)):
                        self.asked_open = True
                        id_param = min(range(1, len(self.want_list) + 1), key=lambda n: (hits_by_want.get(n, 0) > 0, n))
                        words_out = f'Is there anything else the {self.short_name(id_param)} must be or have, that I should know before choosing?'
                        batch = self.deliver([(id_param, 'anything else', words_out)], max(3, self.closing_room() + 2))
                        if batch:
                            delivered += batch
                            had = [len(want_obj.told) for want_obj in self.want_list]
                            heard = self.read_answers(batch, (SEAT_QUICK, SEAT_DEEP))
                            heard = self.exact_value_reply(heard, batch)
                            self.apply_answers(heard, stated_now)
                            self.journal.append({'kind': 'open_question', 'item': id_param, 'fresh': [n for n, want_obj in enumerate(self.want_list, 1) if len(want_obj.told) > had[n - 1]]})
                    break
            if joint:
                words_out, joint = self.shared_wording(joint)
                slot_no, tags = joint[0]
                for n, field_names in joint:
                    used[n].update((plain_words(label_txt) for label_txt in field_names))
            else:
                words_out, tags = self.packed_question(slot_no, tags, displayed)
                used[slot_no].update((plain_words(label_txt) for label_txt in tags))
            batch = self.deliver([(slot_no, tags[0], words_out)], max(3, self.closing_room() + 2))
            if not batch:
                break
            delivered += batch
            for n in [n for n, _ in joint] if joint else [slot_no]:
                spent[n] = spent.get(n, 0) + 1
            had = [len(want_obj.told) for want_obj in self.want_list]
            topic = [(' and '.join((str(n) for n, _ in joint)) + '; give each need the number of the one item it is about' if joint else n, q, a) for n, q, a in batch]
            heard = self.read_answers(topic, (SEAT_QUICK, SEAT_DEEP))
            if joint:
                heard = self.shared_reply_owners(heard, joint)
            heard = self.exact_value_reply(heard, batch)
            self.apply_answers(heard, stated_now)
            fresh = [n for n, want_obj in enumerate(self.want_list, 1) if len(want_obj.told) > had[n - 1]]
            for n in fresh:
                hits_by_want[n] = hits_by_want.get(n, 0) + 1
                for need_txt in self.want_list[n - 1].told[had[n - 1]:]:
                    used.setdefault(n, set()).add(plain_words(need_txt.split(':', 1)[0]))
            crowded = 1 in [as_count(x) for x in as_list(heard.get('crowded'))]
            if joint:
                self.journal.append({'kind': 'joint_ask', 'items': [n for n, _ in joint], 'fresh': fresh})
                continue
            mine = lambda x: isinstance(x, dict) and (as_count(x.get('item')) or 1) == slot_no
            minded = any((mine(x) and any((name_covers(word_roots(str(x.get('property') or '')), word_roots(label_txt)) or name_covers(word_roots(label_txt), word_roots(str(x.get('property') or ''))) for label_txt in tags)) for x in as_list(heard.get('free'))))
            stated = any((mine(x) and str(x.get('need') or '').strip() for x in as_list(heard.get('needs'))))
            if len(tags) >= 2:
                mode = 'single' if crowded and (not fresh) else 'pack'
                self.journal.append({'kind': 'pack_mode', 'after': len(tags), 'mode': mode, 'fresh': bool(fresh), 'minded': minded, 'crowded': crowded})
                if crowded and (not fresh):
                    used[slot_no] -= {plain_words(label_txt) for label_txt in tags}
                    used[slot_no].add(plain_words(tags[0]))
                    follow.insert(0, (slot_no, [tags[0]], False))
                continue
            if fresh or not crowded or (self.scarce_messages and len(tags) < 2):
                continue
            head_word = tags[0].split()[-1]
            if len(tags[0].split()) >= 2 and plain_words(head_word) not in used[slot_no]:
                follow.insert(0, (slot_no, [head_word], False))
            elif not displayed and len(self.example_values(slot_no, tags[0])) >= 2:
                follow.insert(0, (slot_no, tags, True))
        found_items = [n for n in hits_by_want if hits_by_want[n] > 0]
        searched = self.search_values(found_items) if found_items else ''
        walked = self.guarded(self.walk_category, '')
        if walked:
            searched = (searched + '\n' + walked).strip()
        swept = self.guarded(self.fitment_sweep, '')
        if swept:
            searched = (searched + '\n' + swept).strip()
        if stated_now:
            self.need_version += 1
        rows_out = [f'Q: {q}\nA: {a}' for _, q, a in delivered]
        report = "Wren's interviewer asked the shopper (each question used one step):\n" + '\n'.join(rows_out)
        report += '\nNew firm requirements from these answers: ' + ('; '.join(stated_now) if stated_now else 'none')
        report += '. Apply them to every pick (search again with the new values where needed).'
        pointed = [need_txt for _, need_txt in self.chain_needs if any((need_txt in t for t in stated_now))]
        if pointed:
            report += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.'
        if searched:
            report += '\n' + searched
        if self.pending_notices:
            report += '\n[notes seen meanwhile] ' + ' | '.join(self.pending_notices)
            self.pending_notices = []
        return report

    def tied_wants(self) -> set[int]:
        bound = {part for part, _ in self.fit_pairs}
        for slot_no in range(1, len(self.want_list) + 1):
            docs = [r for r in self.read_records.values() if self.belongs_to(self.want_list[slot_no - 1], str(r.get(self.shelf_field() or '') or ''))]
            if docs and sum((1 for r in docs if re.search(FITMENT_RX, as_text(r), re.I))) * 2 > len(docs):
                bound.add(slot_no)
        return set() if len(bound) >= len(self.want_list) else bound

    def pick_next_question(self, queues: dict[int, list[str]], hits_by_want: dict[int, int], spent: dict[int, int], used: dict[int, set[str]], width: int=1) -> tuple[int, list[str]]:
        first_val = {n: i for i, n in enumerate(self.want_order)}
        if width > 1:
            share = lambda n: min(1.0, width / max(1, len(self.unasked_keys(queues, used, n))))
            for slot_no in sorted(queues, key=lambda n: (hits_by_want.get(n, 0) > 0, spent.get(n, 0), first_val.get(n, len(first_val)), -share(n), n)):
                left = self.unasked_keys(queues, used, slot_no)
                if left:
                    return (slot_no, left[:width])
            return (0, [])
        if self.scarce_messages:
            bound = self.tied_wants()
            left = {n: [label_txt for label_txt in queues[n] if plain_words(label_txt) not in used[n]] for n in queues}
            ranked = sorted(queues, key=lambda n: (n in bound, first_val.get(n, len(first_val) + n)))
            free = [n for n in ranked if n not in bound]
            if len(free) >= 2:
                for slot_no in sorted(queues, key=lambda n: (n in bound, hits_by_want.get(n, 0) > 0, spent.get(n, 0), first_val.get(n, len(first_val) + n))):
                    if left.get(slot_no):
                        self.focus_want = slot_no
                        return (slot_no, left[slot_no][:1])
                return (0, [])
            if self.focus_want and hits_by_want.get(self.focus_want, 0) == 0 and left.get(self.focus_want):
                return (self.focus_want, left[self.focus_want][:1])
            for slot_no in [n for n in ranked if hits_by_want.get(n, 0) == 0 and n not in bound] + [self.focus_want] + ranked:
                if slot_no and left.get(slot_no):
                    self.focus_want = slot_no
                    return (slot_no, left[slot_no][:1])
            return (0, [])
        for slot_no in sorted(queues, key=lambda n: (hits_by_want.get(n, 0) > 0, spent.get(n, 0), first_val.get(n, len(first_val) + n))):
            left = self.unasked_keys(queues, used, slot_no)
            if left:
                return (slot_no, left[:1])
        return (0, [])

    def unasked_keys(self, queues: dict[int, list[str]], used: dict[int, set[str]], slot_no: int) -> list[str]:
        return [label_txt for label_txt in queues.get(slot_no) or [] if plain_words(label_txt) not in used.get(slot_no, set()) and (not any((word_roots(label_txt) == word_roots(u) or (len(word_roots(u)) >= 2 and word_roots(u) <= word_roots(label_txt)) for u in used.get(slot_no, set()) if u)))]

    def shared_question(self, queues: dict[int, list[str]], spent: dict[int, int], used: dict[int, set[str]]) -> list[tuple[int, list[str]]]:
        first_val = {n: i for i, n in enumerate(self.want_order)}
        waiting = [n for n in sorted(queues, key=lambda n: first_val.get(n, len(first_val) + n)) if spent.get(n, 0) == 0 and self.unasked_keys(queues, used, n)]
        if len(waiting) < 2:
            return []
        share = max(2, PACK_SIZE // len(waiting))
        return [(n, self.unasked_keys(queues, used, n)[:share]) for n in waiting]

    def shared_reply_owners(self, heard: dict[str, Any], joint: list[tuple[int, list[str]]]) -> dict[str, Any]:
        numbers = [n for n, _ in joint]
        for field in ('needs', 'free', 'drop'):
            for rec in as_list(heard.get(field)):
                if not isinstance(rec, dict) or as_count(rec.get('item')) in numbers:
                    continue
                note_text = word_roots(str(rec.get('need') or rec.get('property') or ''))
                owner = next((n for n, field_names in joint if any((name_covers(word_roots(k), note_text) for k in field_names)) or name_covers(word_roots(self.short_name(n)), note_text)), numbers[0])
                rec['item'] = owner
        return heard

    def shared_wording(self, joint: list[tuple[int, list[str]]]) -> tuple[str, list[tuple[int, list[str]]]]:
        cap = min(PACK_LENGTH, self.message_limit())
        kept_lines = [(n, list(field_names)) for n, field_names in joint]
        while True:
            pieces = [f'for the {self.short_name(n)}, ' + (', '.join(field_names[:-1]) + f' or {field_names[-1]}' if len(field_names) >= 2 else field_names[0]) for n, field_names in kept_lines if field_names]
            words_out = 'What would you like ' + '; and '.join(pieces) + '?'
            if len(words_out) <= cap:
                return (words_out, [(n, k) for n, k in kept_lines if k])
            if all((len(k) <= 1 for _, k in kept_lines)):
                kept_lines.pop()
                if len(kept_lines) == 1:
                    words_out, field_names = self.packed_question(kept_lines[0][0], kept_lines[0][1])
                    return (words_out, [(kept_lines[0][0], field_names)])
                continue
            longest = max(range(len(kept_lines)), key=lambda i: len(kept_lines[i][1]))
            kept_lines[longest][1].pop()

    def question_text(self, slot_no: int, tags: list[str], displayed: bool=False) -> str:
        return self.packed_question(slot_no, tags, displayed)[0]

    def packed_question(self, slot_no: int, tags: list[str], displayed: bool=False) -> tuple[str, list[str]]:
        if len(tags) >= 2:
            tool_name = self.short_name(slot_no)
            cap = min(PACK_LENGTH, self.message_limit())
            kept_lines = list(tags)
            while len(kept_lines) >= 2:
                words_out = f'What {kept_lines[0]} or {kept_lines[1]} would you like for the {tool_name}?' if len(kept_lines) == 2 else f'For the {tool_name}, what ' + ', '.join(kept_lines[:-1]) + f' or {kept_lines[-1]} would you like?'
                if len(words_out) <= cap:
                    return (words_out, kept_lines)
                kept_lines.pop()
            return (self.phrase_question(slot_no, tags[0], None), tags[:1])
        return (self.phrase_question(slot_no, tags[0], self.example_values(slot_no, tags[0]) if displayed else None), tags[:1])

    def message_limit(self) -> int:
        spec_doc = self.tool_schemas.get(self.tool_roles.get('ask') or '') or {}
        prop_specs = spec_doc.get('properties') or {}
        text_fields = [k for k, v in prop_specs.items() if isinstance(v, dict) and 'string' in schema_kinds(v) and ('enum' not in v)]
        id_param = next((k for k in spec_doc.get('required') or [] if k in text_fields), None) or (text_fields[0] if text_fields else None)
        ceiling = (prop_specs.get(id_param) or {}).get('maxLength') if id_param else None
        return ceiling if isinstance(ceiling, int) and ceiling > 20 else 10000

    def candidate_properties(self, slot_no: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        want_obj = self.want_list[slot_no - 1]
        settled = [word_roots(m.split(':', 1)[0]) for m in want_obj.musts]
        row_list = [(label_txt, key, carried, len(vals)) for label_txt, key, carried, vals in table if self.want_for_label(label_txt) == slot_no]
        measured = {key: sum((1 for v in vals if re.match('^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$', v))) * 2 > len(vals) for _, key, _, vals in table}
        if len(self.want_list) == 1 and (not row_list):
            row_list = [(label_txt, key, carried, len(vals)) for label_txt, key, carried, vals in table]
        row_list.sort(key=lambda r: -r[2])
        stated = [frozenset(re.findall('[a-z0-9]+', m.split(':', 1)[1].lower())) for m in want_obj.musts if ':' in m]
        stated = [v for v in stated if v]
        key_values = {key: vals for _, key, _, vals in table}

        def settles(key: str) -> bool:
            if re.search('\\b(?:colou?rs?|patterns?|prints?|styles?|finish(?:es)?|designs?|looks?)\\b', key):
                return False
            return any((frozenset(re.findall('[a-z0-9]+', str(v).lower())) == want for v in key_values.get(key, []) for want in stated if sum((len(w) for w in want if not w.isdigit())) >= 4 and (not want <= {'yes', 'no', 'true', 'false', 'none', 'n', 'y'})))
        field_names = [key for _, key, _, _ in row_list if not any((name_covers(prop_words, word_roots(key)) for prop_words in settled)) and (not re.search(NON_PROPERTY_RX, key)) and (not settles(key))]
        return [key for key in field_names if not re.search(BUNDLE_KEY_RX, key)] + [key for key in field_names if re.search(BUNDLE_KEY_RX, key)]

    def story_sentences(self) -> list[str]:
        plain = ' '.join(self.request_text.split())
        pieces = [p.strip() for p in re.split('(?<=[.!?;])\\s+(?=[A-Za-z(])', plain) if p.strip()]
        boiler = re.compile('\\b(?:budget|cents?|cts|dollars?|usd|price|costs?|spend|pay|total|afford|penny|pennies|loans?|messages?|questions?|\\d+ (?:moves|steps|calls|tool calls)|steps?|in stock|on hand|stock|website|listed|substitute|alternative|come up empty|empty-handed|nothing fits|say so|cheapest|least expensive|costs? the least|lowest|most expensive|check with me|run (?:it|that) by me|confirm|approv\\w*|tell me|let me know)\\b|[$£€]', re.I)
        stated = ' '.join((m for want_obj in self.want_list for m in want_obj.musts)) + ' ' + ' '.join((want_obj.name for want_obj in self.want_list))
        stated_stems = value_stems(stated)
        acc = []
        for p in pieces:
            if boiler.search(p) or len(p.split()) < 4:
                continue
            terms = text_stems(p)
            if stated_stems and len(terms & stated_stems) >= max(2, len(terms) // 3):
                continue
            acc.append(p)
        return acc[:6]

    def quotes_the_story(self, hint: str) -> bool:
        terms = [w for w in value_stems(hint) if len(w) >= 4 and w not in ('that', 'this', 'with', 'from', 'have', 'they', 'them', 'will', 'what', 'when', 'your', 'about', 'been', 'into', 'would', 'like')]
        if len(terms) < 2:
            return False
        for s in self.story_sentences():
            held_now = text_stems(s)
            if sum((1 for w in terms if w in held_now)) >= max(2, len(terms) - 1):
                return True
        return False

    def question_queues(self, num: int) -> dict[int, list[str]]:
        table = self.property_table()
        settled = []
        for slot_no, want_obj in enumerate(self.want_list, 1):
            settled.append(f'item {slot_no}: {want_obj.name} - settled by the request: ' + ('; '.join(want_obj.musts) or 'nothing'))
        listing = []
        last = None
        for pos, (label_txt, tool_name, carried, vals) in enumerate(table, 1):
            if label_txt != last:
                listing.append(f'category {label_txt!r}:')
                last = label_txt
            listing.append(f'  {pos}. {tool_name} (on {carried}): ' + ' | '.join(vals) + self.unstated_mark(label_txt, tool_name, vals) + (' [several features in one: ask last]' if re.search(BUNDLE_KEY_RX, tool_name) else ''))
        body = '\n\nREQUEST:\n' + self.request_text + '\n\nITEMS:\n' + '\n'.join(settled)
        story = self.story_sentences()
        if story:
            body += '\n\nSTORY SENTENCES (each may hint at ONE unwritten preference on ONE property; name that property for the item it concerns, quoting the sentence in "hint"):\n' + '\n'.join((f'- {s}' for s in story))
        first_val = [str(x) for x in as_list(self.intake.get('ask_first')) if str(x).strip()]
        if first_val:
            body += '\n\nASK FIRST (the request asks to be asked about these):\n' + '\n'.join(first_val)
        body += '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(listing) or '(no full listing records yet)')
        proposal = self.vote_properties(body, max(1, num), table)
        queues: dict[int, list[str]] = {n: [] for n in range(1, len(self.want_list) + 1)}
        hinted: dict[int, set[str]] = {n: set() for n in queues}
        for choice in as_list(proposal.get('ask')):
            if not isinstance(choice, dict):
                continue
            slot_no = min(max(1, as_count(choice.get('item') or 1) or 1), len(self.want_list))
            label_txt = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(choice.get('label') or ''))[0].strip().rstrip('?.')
            if not label_txt or len(label_txt.split()) > 5:
                continue
            if all((plain_words(label_txt) != plain_words(alt) for alt in queues[slot_no])):
                queues[slot_no].append(label_txt)
            if str(choice.get('hint') or '').strip():
                hinted[slot_no].add(plain_words(label_txt))
                if slot_no not in self.want_order:
                    self.want_order.append(slot_no)
        for slot_no in queues:
            field_names = self.candidate_properties(slot_no, table)
            read = max(1, self.record_count_for(slot_no))
            own_map: dict[str, float] = {}
            for label_txt, key, carried, _ in table:
                if self.want_for_label(label_txt) == slot_no:
                    own_map[key] = max(own_map.get(key, 0.0), carried / read)

            def spelled(label_txt: str) -> str:
                plain = lambda words_out: re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', plain_words(words_out)))
                return next((k for k in own_map if plain(k) == plain(label_txt)), label_txt)

            def share_of(label_txt: str) -> float:
                if plain_words(label_txt) in own_map:
                    return own_map[plain_words(label_txt)]
                near = [s for k, s in own_map.items() if name_covers(word_roots(label_txt), word_roots(k)) or name_covers(word_roots(k), word_roots(label_txt))]
                return max(near) if near else 0.0
            hinted[slot_no] = {plain_words(spelled(h)) for h in hinted[slot_no]}
            queue = []
            for choice in queues[slot_no]:
                if all((plain_words(spelled(choice)) != plain_words(alt) for alt in queue)):
                    queue.append(spelled(choice))
            for key in [k for k in field_names if own_map.get(k, 0.0) >= BACKUP_SHARE and (not re.search(BUNDLE_KEY_RX, k))][:0 if self.scarce_messages and queue else BACKUP_KEYS]:
                if any((plain_words(key) == plain_words(alt) for alt in queue[:3])):
                    continue
                weak = next((i for i, alt in enumerate(queue[:3]) if plain_words(alt) not in hinted[slot_no] and share_of(alt) + BACKUP_GAP <= own_map.get(key, 0.0)), None)
                if weak is None and len(queue) >= 3:
                    continue
                placed = [alt for alt in queue if plain_words(alt) != plain_words(key)]
                if weak is not None and len(self.want_list) > 1:
                    moved = placed[weak]
                    placed[weak] = key
                    placed.insert(min(3, len(placed)), moved)
                else:
                    placed.insert(min(len(placed), weak if weak is not None else len(placed)), key)
                if any((plain_words(x) in hinted[slot_no] and all((plain_words(x) != plain_words(y) for y in placed[:3])) for x in queue[:3])):
                    continue
                queue = placed
            queues[slot_no] = queue
            for key in field_names:
                if all((plain_words(key) != plain_words(alt) for alt in queues[slot_no])):
                    queues[slot_no].append(key)
            queues[slot_no] = [x for x in queues[slot_no] if not re.search(BUNDLE_KEY_RX, plain_words(x))] + [x for x in queues[slot_no] if re.search(BUNDLE_KEY_RX, plain_words(x))]
            queues[slot_no] = self.single_look(queues[slot_no], hinted[slot_no])
        self.journal.append({'kind': 'plan', 'ask': clipped_copy(proposal.get('ask'), 120), 'lanes': {n: v[:8] for n, v in queues.items()}})
        return queues

    def short_name(self, slot_no: int) -> str:
        tool_name = re.sub('\\s+', ' ', self.want_list[slot_no - 1].name).strip()
        pieces = re.split('\\s+(?:and|or)\\s+|(?<=[A-Za-z])\\s*[&/]\\s*(?=[A-Za-z])', tool_name, maxsplit=1)
        if len(pieces) < 2 or not pieces[0].strip() or (not pieces[1].strip()) or (not re.search('s\\b', pieces[1].strip().split()[-1])):
            return tool_name
        head_word, rest = (pieces[0].strip(), pieces[1].strip().split())
        return head_word + ' ' + rest[-1] if len(head_word.split()) == 1 and len(rest) >= 2 else head_word

    def vote_properties(self, body: str, num: int, table: list[tuple[str, str, int, list[str]]]) -> dict[str, Any]:
        seats = [(ASK_FORM, [SEAT_DEEP, SEAT_QUICK], 3500)]
        if self.outlay < BUDGET_EASE and (not self.past_clock(60)):
            seats += [(STORY_FORM, [SEAT_QUICK, SEAT_ALT], 1500), (DIVIDE_FORM, [SEAT_ALT, SEAT_QUICK], 2500)]
        prompts = [(form.replace('COUNT', str(max(num, 2))) + body, models, cap) for form, models, cap in seats]
        if len(prompts) == 1:
            votes = [self.ask_for_json(*prompts[0]) or {}]
        else:
            with ThreadPoolExecutor(max_workers=len(prompts)) as candidates:
                votes = [f.result() or {} for f in [candidates.submit(self.ask_for_json, *p) for p in prompts]]
        field_names = {}
        for label_txt, tool_name, _, _ in table:
            field_names.setdefault(self.want_for_label(label_txt), []).append(tool_name)
        points: dict[tuple[int, str], float] = {}
        hints: dict[tuple[int, str], str] = {}
        backing: dict[tuple[int, str], float] = {}
        displayed: dict[tuple[int, str], str] = {}
        for weight, vote in zip(VOTER_WEIGHTS, votes):
            rank: dict[int, int] = {}
            for choice in as_list(vote.get('ask')):
                if not isinstance(choice, dict):
                    continue
                slot_no = min(max(1, as_count(choice.get('item') or 1) or 1), len(self.want_list))
                label_txt = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(choice.get('label') or ''))[0].strip().rstrip('?.')
                if not label_txt or len(label_txt.split()) > 5:
                    continue
                plain = lambda words_out: re.sub('our\\b', 'or', ' '.join((w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w for w in plain_words(words_out).split())))
                same = [k for k in field_names.get(slot_no, []) if plain(k) == plain(label_txt)]
                wider = [k for k in field_names.get(slot_no, []) if set(plain(label_txt).split()) < set(plain(k).split())]
                own_map = same[0] if same else wider[0] if len(wider) == 1 else label_txt
                bucket = (slot_no, plain(own_map))
                place = rank.get(slot_no, 0)
                rank[slot_no] = place + 1
                if place >= VOTER_DEPTH:
                    continue
                points[bucket] = points.get(bucket, 0.0) + weight * (VOTER_DEPTH - place)
                displayed.setdefault(bucket, own_map)
                hint = str(choice.get('hint') or '').strip()
                if hint and self.quotes_the_story(hint):
                    hints.setdefault(bucket, hint)
                    backing[bucket] = backing.get(bucket, 0.0) + weight
                    points[bucket] += weight * VOTER_DEPTH
        first_val = [root_set(str(x)) for x in as_list(self.intake.get('ask_first')) if str(x).strip()]
        asked_first = lambda s: any((len(root_set(s[1]) & f) >= 1 for f in first_val))
        merged = sorted(points, key=lambda s: (s[0], not asked_first(s), backing.get(s, 0.0) < 1.8, -points[s]))
        self.journal.append({'kind': 'panel', 'voters': len(votes), 'scores': {f'{n}:{k}': round(v, 1) for (n, k), v in sorted(points.items(), key=lambda kv: -kv[1])[:10]}})
        return {'ask': [{'item': n, 'label': displayed[n, k], 'hint': hints.get((n, k), '')} for n, k in merged]}

    @staticmethod
    def single_look(queue: list[str], hinted: set[str]) -> list[str]:
        look = lambda label_txt: bool(re.search('\\b(?:colou?rs?|patterns?|prints?|styles?|finish(?:es)?|designs?|looks?)\\b', plain_words(label_txt))) and plain_words(label_txt) not in hinted
        kept: list[str] = []
        later: list[str] = []
        met = 0
        for label_txt in queue:
            if len(kept) < 3 and look(label_txt):
                met += 1
                if met > 1:
                    later.append(label_txt)
                    continue
            kept.append(label_txt)
        if not later or len([x for x in kept if not look(x)]) < 2:
            return queue
        return kept[:3] + later + kept[3:]

    def phrase_question(self, slot_no: int, label_txt: str, samples: list[str] | None=None) -> str:
        words_out = f'What {label_txt} would you like for the {self.short_name(slot_no)}'
        return words_out + (f', for example {samples[0]} or {samples[1]}?' if samples else '?')

    def example_values(self, slot_no: int, label_txt: str) -> list[str]:
        want = plain_words(label_txt)
        settled = ' '.join(self.want_list[slot_no - 1].musts).lower()
        for _, tool_name, _, vals in self.property_table():
            if tool_name == want:
                return [v for v in vals if v and v not in settled and (len(v) <= 30)][:2]
        return []

    def deliver(self, questions: list[tuple[int, str, str]], floor: int=6) -> list[tuple[int, str, str]]:
        tool_id = self.tool_roles['ask']
        delivered: list[tuple[int, str, str]] = []
        for slot_no, _, words_out in questions:
            if self.concluded or self.steps_left() <= floor or self.messages_sent >= self.message_cap:
                break
            params = self.message_payload(words_out)
            if not params:
                break
            reply_obj = self.store_call(tool_id, params)
            if reply_obj is None:
                break
            self.messages_sent += 1
            answer = reply_obj.get('observation')
            terms = self.reply_text(answer)
            delivered.append((slot_no, words_out, terms))
            self.action_log.append(f'step {self.steps_used}: message to the shopper {words_out!r} -> reply {terms[:300]!r} (cart then: {self.basket_key()})')
            self.action_steps.append(self.steps_used)
            self.journal.append({'kind': 'ask', 'q': words_out, 'a': terms[:300]})
            if isinstance(answer, dict) and answer.get('error'):
                break
        return delivered

    def apply_answers(self, heard: dict[str, Any], stated_now: list[str]) -> None:
        multi_arg = len(self.want_list) > 1
        for rec in as_list(heard.get('needs')):
            if not isinstance(rec, dict) or not str(rec.get('need') or '').strip():
                continue
            slot_no = min(max(1, as_count(rec.get('item')) or 1), len(self.want_list))
            need_txt = str(rec['need']).strip()
            if rec.get('points') is not True:
                need_txt = bare_need(need_txt)
            if need_txt not in self.want_list[slot_no - 1].told:
                self.want_list[slot_no - 1].told.append(need_txt)
                stated_now.append(f'item {slot_no}: {need_txt}' if multi_arg else need_txt)
            if rec.get('points') is True and (slot_no, need_txt) not in self.chain_needs:
                self.chain_needs.append((slot_no, need_txt))
        for rec in as_list(heard.get('drop')):
            if isinstance(rec, dict) and str(rec.get('property') or '').strip():
                want_obj = self.want_list[min(max(1, as_count(rec.get('item')) or 1), len(self.want_list)) - 1]
                gone = plain_words(str(rec['property']).split(':')[0])
                rx_hit = [m for m in want_obj.musts + want_obj.told if m not in want_obj.dropped and plain_words(m.split(':')[0]) == gone]
                want_obj.dropped.extend(rx_hit)
                stated_now.extend((f'dropped {m}' for m in rx_hit))
        for rec in as_list(heard.get('free')):
            if isinstance(rec, dict) and rec.get('property'):
                slot_no = min(max(1, as_count(rec.get('item')) or 1), len(self.want_list))
                self.want_list[slot_no - 1].free.append(str(rec['property']))
        if isinstance(heard.get('budget'), (int, float)) and (not isinstance(heard.get('budget'), bool)) and (heard['budget'] > 0):
            self.spend_limit = heard['budget']

    def exact_value_reply(self, heard: dict[str, Any], batch: list[tuple[int, str, str]]) -> dict[str, Any]:
        try:
            pattern = re.compile('\\b(?:anything|any\\s+\\w+|whatever)\\s+(?:on|with|where|about|for|in)\\s+(?:the\\s+)?([a-z][a-z0-9 &\\-]{1,60}?)\\s+(?:comes in\\s+|is\\s+|are\\s+)?(?:other than|apart from)\\s+(?:a |an |the )?((?:[^,.;:!?]|\\.(?=\\d)){1,60}?)(?=\\s+(?:gets?|earns?|would|will|is|means|draws?|goes|because|as)\\b|[,;:!?]|\\.(?!\\d)|$)', re.I)
            refusal = re.compile('\\b(?:no|nope|binned|bin|chuck\\w*|toss\\w*|refus\\w*|turn\\w*\\s+(?:it\\s+)?down|not for me|won.?t|never|no room|don.?t\\s+(?:go\\s+)?show\\w*|not having)\\b', re.I)
            needs = [dict(e) for e in as_list(heard.get('needs')) if isinstance(e, dict)]
            for slot_no, question, answer in batch:
                hits_by_want = pattern.search(str(answer or ''))
                if not hits_by_want:
                    continue
                words_out = str(answer or '')
                after, before = (words_out[hits_by_want.end(2):hits_by_want.end(2) + 70], words_out[max(0, hits_by_want.start() - 40):hits_by_want.start()])
                if re.search('\\b(?:fine|ok|okay|happy|good|great|will do|works|suits)\\b', after, re.I) or not (refusal.search(after) or refusal.search(before)):
                    continue
                val = hits_by_want.group(2).strip()
                if re.search('\\b(?:page|listing|same as|whatever|within|either way|unlike|like the)\\b', str(answer)[hits_by_want.start():], re.I):
                    continue
                asked = re.search('what (.+?) would you like', question, re.I)
                prop_words = (hits_by_want.group(1) or (asked.group(1) if asked else '') or '').strip()
                prop_words = re.sub('\\s+(?:of|for)\\s+(?:the|your|my)\\s+.+$', '', prop_words).strip()
                if not val or not prop_words:
                    continue
                fixed_args = False
                for rec in needs:
                    need_txt = str(rec.get('need') or '')
                    if (as_count(rec.get('item')) or 1) == slot_no and re.search('\\b(?:other than|besides|except|apart from|anything but|not)\\b', need_txt, re.I) and (set(plain_words(val).split()) <= set(plain_words(need_txt).split())):
                        rec['need'] = f"{need_txt.split(':', 1)[0].strip() or prop_words}: {val}"
                        fixed_args = True
                if not fixed_args and (not any(((as_count(e.get('item')) or 1) == slot_no and set(plain_words(val).split()) <= set(plain_words(str(e.get('need') or '')).split()) for e in needs))):
                    needs.append({'item': slot_no, 'need': f'{prop_words}: {val}', 'points': False})
                    fixed_args = True
                if fixed_args:
                    heard['free'] = [e for e in as_list(heard.get('free')) if not (isinstance(e, dict) and plain_words(str(e.get('property') or '')) == plain_words(prop_words))]
                    self.journal.append({'kind': 'only_value', 'item': slot_no, 'need': f'{prop_words}: {val}'})
            heard['needs'] = needs
        except Exception as err:
            self.journal.append({'kind': 'only_value_error', 'error': f'{type(err).__name__}: {err}'[:200]})
        return heard

    def read_answers(self, delivered: list[tuple[int, str, str]], models: tuple[str, ...]=(SEAT_DEEP, SEAT_QUICK)) -> dict[str, Any]:
        ask_text_full = LISTEN_FORM + '\n\nITEMS:\n' + '\n'.join((f'item {n}: {it.name}' for n, it in enumerate(self.want_list, 1)))
        ask_text_full += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f'{i}. (about item {n}) Q: {q}\nA: {a}' for i, (n, q, a) in enumerate(delivered, 1)))
        return self.ask_for_json(ask_text_full, list(models), 2500) or {}

    def reply_text(self, tool_result: Any) -> str:
        if isinstance(tool_result, str):
            return tool_result
        hits_by_want: list[str] = []

        def walk(obj: Any, level: int) -> None:
            if level > 4:
                return
            if isinstance(obj, str) and len(obj.split()) >= 2:
                hits_by_want.append(obj)
            elif isinstance(obj, dict):
                for val in obj.values():
                    walk(val, level + 1)
            elif isinstance(obj, list):
                for val in obj:
                    walk(val, level + 1)
        walk(tool_result, 0)
        return ' '.join(hits_by_want)[:800] if hits_by_want else as_text(tool_result)[:400]

    def stray_lines(self, tool_result: Any) -> list[str]:
        rows_out: list[str] = []
        if isinstance(tool_result, dict) and (not self.id_of(tool_result)):
            for key, val in tool_result.items():
                if isinstance(val, str) and len(val.split()) >= 5:
                    rows_out.append(val)
                elif isinstance(val, list) and val and all((isinstance(v, str) for v in val)):
                    rows_out.extend((v for v in val if len(v.split()) >= 5))
                elif isinstance(val, dict) and (not self.id_of(val, False, 1)):
                    for inner in val.values():
                        if isinstance(inner, str) and len(inner.split()) >= 6:
                            rows_out.append(inner)
        return [row_line for row_line in rows_out if row_line not in self.answers_seen]

    def apply_notice(self, rows_out: list[str]) -> str:
        self.answers_seen.update(rows_out)
        ask_text_full = NOTICE_FORM + '\n\nITEMS AND REQUIREMENTS NOW:\n' + self.requirement_sheet() + '\n\nTHE NOTE:\n' + '\n'.join(rows_out)
        json_got = self.ask_for_json(ask_text_full, [SEAT_QUICK, SEAT_ALT], 2500) or {}
        gone = [str(g) for g in as_list(json_got.get('gone')) if isinstance(g, (str, int))]
        self.unavailable_ids.update((g for g in gone if g in self.listings))
        self.notice_gone = bool(gone) or json_got.get('sold_out') is True
        self.notice_named = any((g in self.listings for g in gone))
        if self.notice_gone and len(rows_out) == 1:
            self.gone_lines.update(rows_out)
        if json_got.get('kind') not in (None, 'change'):
            return ''
        changed = []
        budget = json_got.get('budget')
        if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0) and (budget != self.spend_limit):
            self.spend_limit = budget
            changed.append(f'budget now {budget}')
        added: set[str] = set()
        drops = [e for e in as_list(json_got.get('drop')) if isinstance(e, dict) and str(e.get('need') or '').strip()]
        adds = [e for e in as_list(json_got.get('add')) if isinstance(e, dict) and str(e.get('need') or '').strip()]
        for rec in drops:
            want_obj = self.want_list[min(max(1, as_count(rec.get('item')) or 1), len(self.want_list)) - 1]
            id_param = str(rec['need']).strip()
            live = [m for m in want_obj.musts + want_obj.told if m not in want_obj.dropped]
            rx_hit = [m for m in live if m == id_param]
            if not rx_hit:
                rx_hit = [m for m in live if plain_words(m.split(':')[0]) == plain_words(id_param.split(':')[0])][:1]
            want_obj.dropped.extend(rx_hit or [id_param])
            changed.append(f'dropped {id_param}')
        for rec in adds:
            want_obj = self.want_list[min(max(1, as_count(rec.get('item')) or 1), len(self.want_list)) - 1]
            need_txt = str(rec['need']).strip()
            if need_txt in want_obj.dropped:
                want_obj.dropped.remove(need_txt)
            if need_txt not in want_obj.musts + want_obj.told:
                want_obj.told.append(need_txt)
            added.add(need_txt)
            changed.append(f'added {need_txt}')
        if gone:
            changed.append('unavailable: ' + ', '.join(gone))
        if json_got.get('price_moved') is True:
            changed.append('a price changed')
        report = str(json_got.get('summary') or '').strip()
        if self.intake.get('pick') == 'cheapest' and (drops or gone or self.notice_gone or (json_got.get('price_moved') is True) or any((c.startswith('budget now') for c in changed))):
            self.more_hold_done = False
            self.cheap_marks = set()
        if changed or report:
            self.need_version += 1
            self.recent_calls = []
            self.notices.append(report or '; '.join(changed))
        return (report + (' (' + '; '.join(changed) + ')' if changed else '')).strip()

class Fitment:

    def fitment_index(self) -> dict[str, Any]:
        marks: dict[str, dict[str, Any]] = {}
        notes: dict[str, set[str]] = {}
        for pid in self.listing_order:
            rec_doc = self.read_records.get(pid) or self.listings.get(pid) or {}
            words_out = ' '.join((str(v) for v in rec_doc.values() if isinstance(v, str)))
            words_out += ' ' + '. '.join((f'{k} {v}' for block in rec_doc.values() if isinstance(block, dict) for k, v in block.items() if isinstance(v, str)))
            marks[pid] = fit_notes(words_out)
            for part, model_codes in marks[pid]['also']:
                notes.setdefault(part, set()).update(model_codes)
        return {'marks': marks, 'notes': notes}

    def fits_for(self, pid: str, pos: dict[str, Any]) -> set[str]:
        stamp = pos['marks'].get(pid) or {}
        hits_by_want = set(stamp.get('works') or [])
        if stamp.get('own'):
            hits_by_want |= pos['notes'].get(stamp['own'], set())
        if stamp.get('more'):
            for alt, met in pos['marks'].items():
                if alt != pid and met.get('own') == stamp['more']:
                    hits_by_want |= set(met.get('works') or []) | pos['notes'].get(stamp['more'], set())
        return hits_by_want

    def fits_model(self, pid: str, model_codes: set[str], pos: dict[str, Any]) -> bool:
        series = (pos['marks'].get(pid) or {}).get('series') or []
        return bool(self.fits_for(pid, pos) & model_codes) or any((series_covers(model_code, series) for model_code in model_codes))

    def fitment_stated(self, pid: str, pos: dict[str, Any]) -> bool:
        return bool(self.fits_for(pid, pos) or (pos['marks'].get(pid) or {}).get('series'))

    def device_marks(self, pid: str, pos: dict[str, Any]) -> set[str]:
        stamp = pos['marks'].get(pid) or {}
        return set(stamp.get('codes') or []) | ({stamp['own']} if stamp.get('own') else set())

    def part_fits(self, part: str, device: str, pos: dict[str, Any]) -> bool | None:
        model_codes = self.device_marks(device, pos)
        if not model_codes:
            return None
        stamp = pos['marks'].get(part) or {}
        if model_codes & set(stamp.get('excl') or []):
            return False
        fits = self.fits_for(part, pos)
        if model_codes & fits:
            return True
        fams = list(stamp.get('fams') or [])
        if any((re.search(fam, model_code) for fam in fams for model_code in model_codes)):
            return True
        if fits or fams:
            return False
        return None

    def guess_anchor(self) -> None:
        if self.fit_pairs or len(self.want_list) < 2 or self.__dict__.get('bound_tried') == len(self.listing_order):
            return
        self.bound_tried = len(self.listing_order)
        pos = self.fitment_index()
        top_pick: tuple[int, int, list[int]] | None = None
        for main in range(1, len(self.want_list) + 1):
            model_codes: set[str] = set()
            for pid in self.listings_for(main):
                model_codes |= self.device_marks(pid, pos)
            if not model_codes:
                continue
            pieces = []
            hits_total = 0
            for part in range(1, len(self.want_list) + 1):
                if part == main:
                    continue
                hits = sum((1 for pid in self.listings_for(part) if self.fits_for(pid, pos) & model_codes))
                if hits >= 2:
                    pieces.append(part)
                    hits_total += hits
            if pieces and (top_pick is None or (len(pieces), hits_total) > (len(top_pick[2]), top_pick[1])):
                top_pick = (main, hits_total, pieces)
        if top_pick:
            self.fit_pairs = [(part, top_pick[0]) for part in top_pick[2]]
            self.journal.append({'kind': 'bound_inferred', 'bound': self.fit_pairs})

    def want_needs(self, slot_no: int) -> list[str]:
        refs = [m for _, m in self.chain_needs]
        return [need_txt for n, need_txt in self.live_needs() if n == slot_no and need_txt not in refs]

    def need_score(self, pid: str, slot_no: int) -> tuple[int, bool]:
        rec_doc = self.read_records.get(pid) or self.listings.get(pid) or {}
        sheet = RecordJudge.sheet(self.want_needs(slot_no), rec_doc) if rec_doc else []
        return (sum((1 for _, v, _ in sheet if v in ('meets', 'elsewhere'))), any((v == 'differs' for _, v, _ in sheet)))

    def solve_fitment(self) -> dict[str, Any]:
        self.guess_anchor()
        if not self.fit_pairs:
            return {}
        pos = self.fitment_index()
        acc: dict[str, Any] = {'sets': []}
        for main in sorted({m for _, m in self.fit_pairs}):
            pieces = [p for p, m in self.fit_pairs if m == main]
            top_pick = None
            for device in self.listings_for(main):
                if self.is_gone(device) or not self.can_buy(device) or (not self.device_marks(device, pos)):
                    continue
                meets, broken = self.need_score(device, main)
                if broken:
                    continue
                rows_out = {main: device}
                missing = []
                evidence = meets
                for part in pieces:
                    cands = []
                    for pid in self.listings_for(part):
                        if pid == device or self.is_gone(pid) or (not self.can_buy(pid)) or (not self.part_fits(pid, device, pos)):
                            continue
                        json_got, bad = self.need_score(pid, part)
                        if not bad:
                            price = self.price_for(pid)
                            cands.append((-json_got, price if price is not None else 1000000000000.0, pid))
                    if not cands:
                        missing.append(part)
                        continue
                    cands.sort()
                    rows_out[part] = cands[0][2]
                    evidence -= cands[0][0]
                total = sum(((self.price_for(i) or 0) * self.want_list[n - 1].qty for n, i in rows_out.items()))
                over = isinstance(self.spend_limit, (int, float)) and self.intake.get('budget_scope') != 'each' and (total > self.spend_limit)
                points = (len(pieces) - len(missing), 0 if over else 1, evidence, -total)
                if top_pick is None or points > top_pick[0]:
                    top_pick = (points, rows_out, missing, over)
            if top_pick:
                acc['sets'].append({'main': main, 'lines': top_pick[1], 'missing': top_pick[2], 'over': top_pick[3]})
        return acc

    def fitment_status(self) -> str:
        try:
            sol = self.solve_fitment()
            if not self.fit_pairs:
                return ''
            pos = self.fitment_index()
            parts_text = []
            for chosen in sol.get('sets') or []:
                device = chosen['lines'][chosen['main']]
                model_codes = ', '.join(sorted(self.device_marks(device, pos))[:3])
                parts_text.append(f"item {chosen['main']} {self.want_list[chosen['main'] - 1].name}: {device} (known as {model_codes})")
                for part, pid in chosen['lines'].items():
                    if part != chosen['main']:
                        parts_text.append(f'item {part} {self.want_list[part - 1].name}: {pid} (its text names {model_codes})')
                for part in chosen['missing']:
                    parts_text.append(f'item {part} {self.want_list[part - 1].name}: no listing seen yet names {model_codes}; list its category and read the rows')
                if chosen['over']:
                    parts_text.append('this set is over the budget: look for cheaper listings that name the same codes')
            head_word = '[fit solver: from every listing seen, a part fits a device when its text names a code the device is known by] '
            if parts_text:
                return head_word + 'best set so far: ' + '; '.join(parts_text) + ". Before checking out, open each chosen listing and any page it points to ('more on part ...')."
            proposal = self.fitment_overview(short=True)
            return head_word + ('no device with a known code has been seen yet' + (': ' + proposal if proposal else "; list the devices' category and read the codes on its rows"))
        except Exception as err:
            self.journal.append({'kind': 'solver_error', 'error': f'{type(err).__name__}: {err}'[:200]})
            return ''

    def fitment_sweep(self) -> str:
        if self.__dict__.get('swept_fit') or len(self.want_list) < 2:
            return ''
        self.guess_anchor()
        if not self.fit_pairs:
            return ''
        self.swept_fit = True
        proposal = self.search_pattern()
        if not proposal:
            return ''
        tool_id, base, field = proposal
        spec_doc = self.tool_schemas.get(tool_id) or {}
        prop_specs = spec_doc.get('properties') or {}
        sol = self.solve_fitment()
        targets: list[int] = []
        for chosen in sol.get('sets') or []:
            targets += chosen['missing']
        if not sol.get('sets'):
            targets = sorted({m for _, m in self.fit_pairs}) + [p for p, _ in self.fit_pairs]
        text_fields = []
        call_list = 0
        for slot_no in dict.fromkeys(targets):
            if call_list >= 6 or self.steps_left() - self.closing_room() < 20 or self.concluded:
                break
            params = dict(self.sweep_scope(slot_no, base, field))
            browse = False
            if params != base:
                for key, field_spec in prop_specs.items():
                    opts = field_spec.get('enum') if isinstance(field_spec, dict) else None
                    choice = next((o for o in opts or [] if isinstance(o, str) and re.search('browse|list|all|categor|filter|shelf', o, re.I)), None)
                    if choice is not None:
                        params[key] = choice
                        browse = True
            if browse and field not in (spec_doc.get('required') or []):
                params.pop(field, None)
            else:
                params[field] = self.want_list[slot_no - 1].name
            fault_list: list[str] = []
            one_call = conform(params, spec_doc, 'args', fault_list)
            if fault_list:
                continue
            reply_obj = self.store_call(tool_id, one_call)
            call_list += 1
            if reply_obj is None:
                break
            met = reply_obj.get('observation')
            self.digest(tool_id, met)
            row_list = [i for i, _ in self.listing_records(met)]
            page = self.page_state(tool_id, met)
            if page and page[3] is not None and (call_list < 6) and (self.steps_left() - self.closing_room() >= 20):
                more = conform({**one_call, page[2]: page[3]}, spec_doc, 'args', fault_list)
                again = self.store_call(tool_id, more) if not fault_list else None
                call_list += 1
                if again is not None:
                    self.digest(tool_id, again.get('observation'))
                    row_list += [i for i, _ in self.listing_records(again.get('observation'))]
            text_fields.append(f'item {slot_no} ({self.want_list[slot_no - 1].name}): {len(row_list)} rows listed')
        if not text_fields:
            return ''
        self.journal.append({'kind': 'fit_sweep', 'texts': text_fields})
        return '[Wren listed the categories of the items that must fit together, to read their fit text] ' + '; '.join(text_fields) + '\n' + self.fitment_status()

    def pointer_pages(self, pids: list[str]) -> list[str]:
        if len(self.want_list) < 2:
            return []
        pos = self.fitment_index()
        owners = {m.get('own'): i for i, m in pos['marks'].items() if m.get('own')}
        want: list[str] = []
        unknown: list[str] = []
        for pid in pids:
            for model_code in (pos['marks'].get(pid) or {}).get('more') or []:
                known = owners.get(model_code)
                if known and known not in self.pages_opened and (known not in want):
                    want.append(known)
                elif not known and model_code not in unknown:
                    unknown.append(model_code)
        proposal = self.search_pattern() if unknown else None
        for model_code in unknown[:2]:
            if not proposal or self.steps_left() <= self.closing_room() + 6 or self.concluded:
                break
            tool_id, base, field = proposal
            fault_list: list[str] = []
            one_call = conform({**base, field: model_code}, self.tool_schemas.get(tool_id), 'args', fault_list)
            if fault_list:
                break
            reply_obj = self.store_call(tool_id, one_call)
            if reply_obj is None:
                break
            self.digest(tool_id, reply_obj.get('observation'))
            pos = self.fitment_index()
            hits_by_want = next((i for i, m in pos['marks'].items() if m.get('own') == model_code), None)
            if hits_by_want and hits_by_want not in self.pages_opened and (hits_by_want not in want):
                want.append(hits_by_want)
        text_fields = []
        tool_id = self.detail_tool()
        id_param = self.id_argument(tool_id)
        for pid in want[:3]:
            if not tool_id or not id_param or self.steps_left() <= self.closing_room() + 3 or self.concluded:
                break
            fault_list = []
            params = conform({id_param: pid}, self.tool_schemas.get(tool_id), 'args', fault_list)
            if fault_list:
                break
            reply_obj = self.store_call(tool_id, params)
            if reply_obj is None:
                break
            self.digest(tool_id, reply_obj.get('observation'))
            text_fields.append(f'{pid} (a page a cart line points to): ' + self.render(reply_obj.get('observation'))[:800])
        if text_fields:
            self.journal.append({'kind': 'fit_pointers', 'opened': want[:3]})
        return text_fields

    def solved_picks(self) -> dict[int, str]:
        try:
            sol = self.solve_fitment()
            choices: dict[int, str] = {}
            for chosen in sol.get('sets') or []:
                if chosen['missing'] or chosen['over']:
                    return {}
                choices.update(chosen['lines'])
            if not choices:
                return {}
            rest = self.clean_choices(partial=True)
            for slot_no in range(1, len(self.want_list) + 1):
                if slot_no not in choices and slot_no in rest:
                    choices[slot_no] = rest[slot_no]
            return choices if len(choices) == len(self.want_list) else {}
        except Exception as err:
            self.journal.append({'kind': 'solver_picks_error', 'error': f'{type(err).__name__}: {err}'[:200]})
            return {}

    def fitment_lines(self, pids: list[str]) -> list[str]:
        if not self.fit_pairs:
            return []
        pos = self.fitment_index()
        rows_out = []
        for part, main in self.fit_pairs:
            devices = [i for i in pids if i in self.listings_for(main)]
            for device in devices[:1]:
                model_codes = ', '.join(sorted(self.device_marks(device, pos))[:3]) or 'no code seen'
                for pid in [i for i in pids if i in self.listings_for(part)]:
                    verdict = self.part_fits(pid, device, pos)
                    named_ids = ', '.join(sorted(self.fits_for(pid, pos))[:4])
                    rows_out.append(f'FIT {pid} with {device} ({model_codes}): ' + ('FITS (its text names that code)' if verdict else 'DOES NOT FIT (it names ' + (named_ids or 'other codes') + ')' if verdict is False else 'NOT STATED'))
        return rows_out

    def fitment_problems(self) -> list[str]:
        self.guess_anchor()
        if not self.fit_pairs:
            return []
        pos = self.fitment_index()
        basket_tool = self.basket_ids()
        fault_list = []
        for part, main in self.fit_pairs:
            devices = [i for i in basket_tool if i in self.listings_for(main)]
            pieces = [i for i in basket_tool if i in self.listings_for(part)]
            if len(devices) != 1 or not pieces:
                continue
            device = devices[0]
            model_codes = ', '.join(sorted(self.device_marks(device, pos))[:3])
            for row_line in pieces:
                if self.part_fits(row_line, device, pos) is False:
                    named_ids = ', '.join(sorted(self.fits_for(row_line, pos))[:4]) or 'other codes'
                    good = [i for i in self.listings_for(part) if i not in basket_tool and (not self.is_gone(i)) and self.part_fits(i, device, pos)][:4]
                    fault_list.append(f'{row_line} ({self.want_list[part - 1].name}) names {named_ids}, not {device} ({model_codes})' + (f"; seen parts of that kind that name it: {', '.join(good)}" if good else '; no seen part of that kind names it, so list that category or switch the device. ' + self.fitment_status()))
        return fault_list

    def fitting_set(self) -> dict[str, int]:
        solved: dict[str, int] = {}
        for chosen in (self.guarded(self.solve_fitment, {}) or {}).get('sets') or []:
            if chosen['missing'] or chosen['over']:
                solved = {}
                break
            solved.update({pid: self.want_list[n - 1].qty for n, pid in chosen['lines'].items()})
        total = sum(((self.price_for(i) or 0) * q for i, q in solved.items()))
        if solved and (not (isinstance(self.spend_limit, (int, float)) and self.intake.get('budget_scope') != 'each' and (total > self.spend_limit))):
            return solved
        if not self.fit_pairs:
            return {}
        pos = self.fitment_index()
        price = lambda i: self.price_for(i) if self.price_for(i) is not None else 1e+18

        def shows(slot_no: int, pid: str) -> int:
            want_obj = self.want_list[slot_no - 1]
            words_out = as_text(self.read_records.get(pid) or self.listings.get(pid)).lower()
            wants = [m for m in want_obj.musts + want_obj.told if m not in want_obj.dropped]
            named_ids = sum((1 for w in set(re.findall('[a-z0-9]{2,}', want_obj.name.lower())) if re.search('\\b' + re.escape(w), words_out)))
            return 2 * sum((1 for m in wants if all((w in words_out for w in re.findall('[a-z0-9]{3,}', m.partition(':')[2].lower())[:3])))) + named_ids

        def best_of(slot_no: int, pids: list[str]) -> list[str]:
            top = max((shows(slot_no, i) for i in pids), default=0)
            return [i for i in pids if shows(slot_no, i) == top]
        top_pick: dict[str, int] = {}
        best_total = None
        for main in sorted({m for _, m in self.fit_pairs}):
            pieces = [p for p, m in self.fit_pairs if m == main]
            devices = best_of(main, [i for i in self.listings_for(main) if self.can_buy(i) and (pos['marks'].get(i) or {}).get('codes')])
            for device in sorted(devices, key=lambda i: (i not in self.basket_ids(), price(i)))[:12]:
                model_codes = set(pos['marks'][device]['codes'])
                rows_out = {device: self.want_list[main - 1].qty}
                for part in pieces:
                    fitting = best_of(part, [i for i in self.listings_for(part) if self.can_buy(i) and self.fits_model(i, model_codes, pos)])
                    if not fitting:
                        rows_out = {}
                        break
                    choice = min(fitting, key=lambda i: (i not in self.basket_ids(), i not in self.read_records, price(i)))
                    rows_out[choice] = self.want_list[part - 1].qty
                if not rows_out:
                    continue
                total = sum((price(i) * q for i, q in rows_out.items()))
                if self.spend_limit is not None and self.intake.get('budget_scope') != 'each' and (total > self.spend_limit):
                    continue
                if best_total is None or total < best_total:
                    top_pick, best_total = (rows_out, total)
        return top_pick

    def mend_fitment(self) -> bool:
        basket_tool = self.tool_roles.get('cart')
        if not basket_tool or not self.fit_pairs or (not self.fitment_problems()):
            return False
        combo = self.fitting_set()
        if not combo:
            return False
        rows_out = {i: q for i, q in self.basket_lines.items() if i in self.basket_ids()}
        shelf_key = self.shelf_field()
        kind_txt = lambda i: str((self.listings.get(i) or {}).get(shelf_key) or i)
        taken = {kind_txt(i) for i in combo}
        rows_out = {**{i: q for i, q in rows_out.items() if kind_txt(i) not in taken}, **combo}
        drop = [i for i in self.basket_ids() if i not in rows_out]
        call_list = self.basket_calls(rows_out, drop)
        if not call_list or len(call_list) + 2 > self.steps_left():
            return False
        for params in call_list:
            fault_list: list[str] = []
            params = conform(params, self.tool_schemas.get(basket_tool), 'args', fault_list)
            if fault_list:
                return False
            reply_obj = self.store_call(basket_tool, params)
            if reply_obj is None or self.concluded:
                return False
            self.record_basket(basket_tool, params)
            self.digest(basket_tool, reply_obj.get('observation'))
        self.journal.append({'kind': 'fit_repair', 'lines': rows_out})
        return True

    def fitment_overview(self, short: bool=False) -> str:
        if not self.fit_pairs:
            return ''
        pos = self.fitment_index()
        rows_out = []
        for main in sorted({m for _, m in self.fit_pairs}):
            pieces = [p for p, m in self.fit_pairs if m == main]
            tally: dict[str, dict[int, int]] = {}
            for p in pieces:
                for i in self.listings_for(p):
                    if self.is_gone(i):
                        continue
                    for model_code in self.fits_for(i, pos):
                        tally.setdefault(model_code, {}).setdefault(p, 0)
                        tally[model_code][p] += 1
            ranked = sorted(tally.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1].values())))[:4]
            devices = {model_code: [i for i in self.listings_for(main) if model_code in (pos['marks'].get(i, {}).get('codes') or [])][:2] for model_code, _ in ranked}
            words_out = '; '.join((f'{model_code}: ' + ', '.join((f'{n} {self.want_list[p - 1].name}' for p, n in sorted(per.items()))) + (f" (device seen: {', '.join(devices[model_code])})" if devices[model_code] else ' (device not seen yet)') for model_code, per in ranked))
            if words_out:
                rows_out.append(f"codes the {self.want_list[main - 1].name} candidates' parts name -> {words_out}")
        if not rows_out:
            return ''
        return ' | '.join(rows_out) if short else "[fit plan: a device fits a part when the part's text names the device's model code or another name it is sold as; codes are not searchable, so list the device's category and read the codes on its rows] " + ' | '.join(rows_out)

class Basket:

    def basket_key(self) -> str:
        return as_text(sorted(self.basket_lines.items()))

    def basket_text(self) -> str:
        if self.basket_view is not None and self.basket_viewed_at >= self.basket_changed_at:
            return as_text(clipped_copy(self.basket_view, 300))[:3000]
        return "lines by Wren's record of the cart calls (listing: units): " + as_text(self.basket_lines)

    def units_held(self, pid: str) -> int:
        if self.basket_view is not None and self.basket_viewed_at >= self.basket_changed_at:
            stack: list[Any] = [self.basket_view]
            while stack:
                obj = stack.pop()
                if isinstance(obj, dict):
                    if pid in [v.strip() for v in obj.values() if isinstance(v, str)]:
                        counts = [as_count(v) for k, v in obj.items() if not isinstance(v, str) and as_count(v) is not None and re.search('\\b(?:qty|quantity|count|units?)\\b', plain_words(k))]
                        if len(counts) == 1:
                            return counts[0]
                    stack += [v for v in obj.values() if isinstance(v, (dict, list))]
                elif isinstance(obj, list):
                    stack += obj
        return self.basket_lines.get(pid, 1)

    def wanted_units(self, pid: str) -> int:
        return next((want_obj.qty for slot_no, want_obj in enumerate(self.want_list, 1) if pid in self.listings_for(slot_no)), 1)

    def basket_ids(self) -> list[str]:
        if self.basket_view is not None and self.basket_viewed_at >= self.basket_changed_at:
            words_out = as_text(self.basket_view)
            return [i for i in self.listings if i in words_out]
        return list(self.basket_lines)

    def check_neighbours(self, pids: list[str]) -> str:
        shelf_key = self.shelf_field()
        kind_txt = lambda i: str((self.read_records.get(i) or self.listings.get(i) or {}).get(shelf_key) or '') if shelf_key else ''
        kinds = {kind_txt(i) for i in pids}
        choice = self.intake.get('pick')
        cheapest = choice == 'cheapest'
        line_price = {kind_txt(i): self.price_for(i) or 0 for i in pids}
        order = list(reversed(self.listing_order))
        if cheapest:
            order.sort(key=lambda i: self.price_for(i) if self.price_for(i) is not None else float('inf'))
        opened, row_list = ([], [])
        for pid in order:
            if pid in pids or self.is_gone(pid) or kind_txt(pid) not in kinds:
                continue
            rec_doc = self.read_records.get(pid) or self.listings.get(pid) or {}
            if pid in self.read_records or pid in self.pages_opened:
                if len(opened) < 8:
                    opened.append('OPENED ' + as_text(clipped_copy(rec_doc, 110))[:650])
            elif choice in ('cheapest', 'most', 'least') and len(row_list) < 6 and (self.price_for(pid) is not None) and (choice != 'cheapest' or self.price_for(pid) < line_price.get(kind_txt(pid), 0)):
                title = next((v for k, v in rec_doc.items() if isinstance(v, str) and k.lower() in ('title', 'name', 'product_name', 'listing_title')), '')
                row_list.append(f'ROW {pid} price {self.price_for(pid)}: ' + str(title)[:140])
        return '\n'.join(opened + row_list)

    def proven_by_code(self, pids: list[str]) -> bool:
        if self.intake.get('pick') not in (None, 'any') or self.chain_needs or self.fit_pairs or (not pids) or (len(pids) != len(self.want_list)):
            return False
        shelf_key = self.shelf_field()
        owners = []
        total = 0.0
        for pid in pids:
            rec_doc = self.read_records.get(pid)
            if not rec_doc or self.is_gone(pid) or (not self.can_buy(pid)):
                return False
            slot_no = (self.want_for_label(str(rec_doc.get(shelf_key) or '')) if shelf_key else 0) or (1 if len(self.want_list) == 1 else 0)
            price = self.price_for(pid)
            if not slot_no or price is None:
                return False
            owners.append(slot_no)
            total += price * self.want_list[slot_no - 1].qty
            needs = [need_txt for n, need_txt in self.live_needs() if n == slot_no]
            if not needs or not RecordJudge.clear(RecordJudge.sheet(needs, rec_doc)):
                return False
        if sorted(owners) != list(range(1, len(self.want_list) + 1)):
            return False
        if isinstance(self.spend_limit, (int, float)) and (total if self.intake.get('budget_scope') != 'each' else max((self.price_for(i) or 0 for i in pids))) > self.spend_limit:
            return False
        return True

    def line_problems(self) -> list[str]:
        shelf_key = self.shelf_field()
        pids = [i for i in self.basket_ids() if (self.units_held(i) or 0) > 0]
        if not pids or not shelf_key:
            return []
        label_txt = {i: str((self.listings.get(i) or {}).get(shelf_key) or '') for i in pids}
        known = {n for n, want_obj in enumerate(self.want_list, 1) if any((self.belongs_to(want_obj, str(r.get(shelf_key) or '')) for r in self.listings.values()))}
        owner = {i: [n for n, want_obj in enumerate(self.want_list, 1) if self.belongs_to(want_obj, label_txt[i])] for i in pids}
        fault_list: list[str] = []
        for n, want_obj in enumerate(self.want_list, 1):
            if n not in known:
                continue
            rows_out = [i for i in pids if n in owner[i]]
            if not rows_out:
                fault_list.append(f'item {n} ({want_obj.name}) has no line in the cart')
            elif len([i for i in rows_out if owner[i] == [n]]) > 1:
                fault_list.append(f"item {n} ({want_obj.name}) has {len(rows_out)} lines ({', '.join(rows_out)}); keep one")
            for i in rows_out:
                held_now = self.units_held(i)
                if owner[i] == [n] and held_now != want_obj.qty:
                    fault_list.append(f'{i} is held at {held_now} unit(s); item {n} ({want_obj.name}) wants {want_obj.qty}')
        if len(known) == len(self.want_list):
            for i in pids:
                if not owner[i]:
                    fault_list.append(f"{i} ({label_txt[i] or 'unknown kind'}) is not a kind the shopper asked for: take it out")
        for i in pids:
            if self.is_gone(i):
                fault_list.append(f'{i} is sold out')
        prices = {i: self.price_for(i) for i in pids}
        if isinstance(self.spend_limit, (int, float)) and all((p is not None for p in prices.values())):
            if self.intake.get('budget_scope') == 'each':
                over = [i for i in pids if self.spend_limit < prices[i] <= 10 * self.spend_limit]
                fault_list += [f'{i} costs {prices[i]:g}, over the {self.spend_limit:g} limit for each item' for i in over]
            else:
                total = sum((prices[i] * self.units_held(i) for i in pids))
                if self.spend_limit + 1e-09 < total <= 10 * self.spend_limit:
                    fault_list.append(f'the total is {total:g}, over the budget of {self.spend_limit:g}')
        return fault_list

    def quantity_gaps(self) -> list[tuple[str, int, int]]:
        shelf_key = self.shelf_field()
        if not shelf_key:
            return []
        acc = []
        for pid in self.basket_ids():
            held_now = self.units_held(pid) or 0
            owners = [n for n, want_obj in enumerate(self.want_list, 1) if self.belongs_to(want_obj, str((self.listings.get(pid) or {}).get(shelf_key) or ''))]
            mine = [i for i in self.basket_ids() if (self.basket_lines.get(i, 1) or 0) > 0 and len(owners) == 1 and self.belongs_to(self.want_list[owners[0] - 1], str((self.listings.get(i) or {}).get(shelf_key) or ''))]
            if held_now > 0 and len(owners) == 1 and (len(mine) == 1) and (held_now != self.want_list[owners[0] - 1].qty) and (not self.is_gone(pid)):
                acc.append((pid, held_now, self.want_list[owners[0] - 1].qty))
        return acc

    def set_quantities(self, wrong: list[tuple[str, int, int]]) -> list[str]:
        tool_name = self.tool_roles.get('cart') or ''
        spec_doc = self.tool_schemas.get(tool_name) or {}
        prop_specs = spec_doc.get('properties') or {}
        text_fields = [k for k, v in prop_specs.items() if isinstance(v, dict) and 'string' in schema_kinds(v) and ('enum' not in v)]
        id_param = next((k for t, k in self.id_arguments if t == tool_name and k in text_fields), None) or next((k for k in spec_doc.get('required') or [] if k in text_fields), None) or (text_fields[0] if len(text_fields) == 1 else None)
        ints = [k for k, v in prop_specs.items() if isinstance(v, dict) and 'integer' in schema_kinds(v)]
        named_ids = [k for k in ints if re.search(QTY_WORDS, plain_words(k)) and (not re.search(INDEX_WORDS, plain_words(k)))]
        num = named_ids[0] if named_ids else ints[0] if len(ints) == 1 else None
        mode, on, off = self.cart_mode_words(prop_specs)
        if not tool_name or not id_param or (not num):
            return []
        call_list: list[dict[str, Any]] = []
        for pid, held_now, want in wrong:
            if self.basket_replaces(tool_name):
                call_list.append({id_param: pid, num: want})
            elif mode:
                call_list += [{id_param: pid, mode: off, num: held_now}, {id_param: pid, mode: on, num: want}]
            else:
                return []
        if len(call_list) + 3 > self.steps_left():
            return []
        done = []
        for params in call_list:
            fault_list: list[str] = []
            params = conform(params, spec_doc, 'args', fault_list)
            if fault_list:
                break
            reply_obj = self.store_call(tool_name, params)
            if reply_obj is None or self.concluded:
                break
            self.record_basket(tool_name, params)
            self.basket_said = as_text(reply_obj.get('observation'))
            self.digest(tool_name, reply_obj.get('observation'))
            done.append(as_text(params) + ' -> ' + self.render(reply_obj.get('observation'))[:300])
        self.journal.append({'kind': 'count_fix', 'calls': done})
        return done

    def check_id(self, pids: list[str]) -> str:
        return '|'.join(sorted(pids)) + f'#{self.need_version}'

    def judged_text(self, pids: list[str]) -> str:
        try:
            shelf_key = self.shelf_field()
            rows_out = []
            for pid in pids:
                rec_doc = self.read_records.get(pid) or self.listings.get(pid) or {}
                slot_no = (self.want_for_label(str(rec_doc.get(shelf_key) or '')) if shelf_key else 0) or 1
                needs = [need_txt for n, need_txt in self.live_needs() if n == slot_no and need_txt not in [m for _, m in self.chain_needs]]
                if needs and rec_doc:
                    rows_out.append(RecordJudge.card(pid, RecordJudge.sheet(needs, rec_doc)))
            rows_out += self.fitment_lines(pids)
            return '\n\nCOMPUTED EVIDENCE (code located each requirement on each record; it does not judge synonyms, units or numbers, so read TO READ and NOT STATED yourself and overrule a line only when the record clearly says otherwise):\n' + '\n'.join(rows_out) if rows_out else ''
        except Exception as err:
            self.journal.append({'kind': 'judged_error', 'error': f'{type(err).__name__}: {err}'[:200]})
            return ''

    def verify_basket(self, pids: list[str] | None=None, ceiling: float=BUDGET_EASE) -> str:
        lead = pids is not None
        pids = pids if lead else self.basket_ids()
        key = self.check_id(pids)
        if key in self.checked_ids or not self.check_seats(ceiling) or (not pids):
            return ''
        self.checked_ids.add(key)
        ask_text_full = VERIFY_FORM + '\n\nREQUEST:\n' + self.request_text + '\n\nREQUIREMENTS IN FORCE:\n' + self.requirement_sheet()
        if lead:
            quantity = '; '.join((f'item {n}: quantity {it.qty}' for n, it in enumerate(self.want_list, 1)))
            ask_text_full += '\n\nCART (planned, judge these listings as if they were carted at the requested quantities: ' + quantity + '):\n' + ', '.join(pids)
        else:
            ask_text_full += '\n\nCART:\n' + self.basket_text()
        ask_text_full += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + '\n'.join((f'{i}: ' + ('SOLD OUT (the store says it can no longer be bought). ' if self.is_gone(i) else '') + self.record_text(i) for i in pids))
        ask_text_full += self.judged_text(pids)
        ask_text_full += '\n\nOTHER LISTINGS SEEN OF THE SAME KINDS (OPENED = full record read, ROW = search row only):\n' + (self.guarded(self.check_neighbours, '', pids) or '(none)')
        ask_text_full += '\n\nAlso add to the JSON object "backup": {"<cart listing id>": [up to 3 ids of OTHER listings marked OPENED that meet every requirement of that cart line\'s item, best by the pick rule first]} so a line can be replaced at once if it sells out; listings marked ROW never count as a backup.'
        seats = self.check_seats(ceiling)
        if len(seats) > 1 and seats[0] == SEAT_DEEP and self.guarded(self.proven_by_code, False, pids):
            json_got = self.ask_for_json(ask_text_full, [SEAT_QUICK], 3000)
            if not json_got or json_got.get('ok') is False:
                self.journal.append({'kind': 'check_escalated', 'cart': pids})
                json_got = self.ask_for_json(ask_text_full, seats, 3000) or json_got
        else:
            json_got = self.ask_for_json(ask_text_full, seats, 3000)
        self.journal.append({'kind': 'check', 'cart': pids, 'got': json_got})
        for row_line, alts in ((json_got or {}).get('backup') or {}).items() if isinstance((json_got or {}).get('backup'), dict) else []:
            kept_lines = [str(i) for i in as_list(alts) if (str(i) in self.read_records or str(i) in self.pages_opened) and self.can_buy(str(i)) and (str(i) not in pids) and (str(i) != str(row_line))]
            if self.intake.get('pick') == 'cheapest':
                kept_lines.sort(key=lambda i: self.price_for(i) if self.price_for(i) is not None else 1e+18)
            if kept_lines and str(row_line) in pids:
                self.backups[str(row_line)] = (self.requirement_sheet(), kept_lines[:3])
        for row_line in pids:
            if self.backups.get(row_line, ('', []))[0] != self.requirement_sheet() or not self.backups[row_line][1]:
                alts = self.guarded(self.coded_backups, [], row_line, pids)
                if alts:
                    self.backups[row_line] = (self.requirement_sheet(), alts[:3])
                    self.journal.append({'kind': 'code_backups', 'line': row_line, 'alts': alts[:3]})
        self.basket_verdicts[key] = bool(json_got) and json_got.get('ok') is not False
        if not json_got or json_got.get('ok') is not False:
            return ''
        self.lead_ids = [i for i in self.lead_ids if i not in pids]
        problems = [str(p) for p in as_list(json_got.get('problems'))][:5]
        better = [str(b) for b in as_list(json_got.get('better')) if str(b) in self.listings and (not self.is_gone(str(b)))][:4]
        words_out = 'A careful review says this cart is not right yet: ' + '; '.join(problems or ['see the requirements'])
        if better:
            words_out += '. Listings that look better: ' + ', '.join(better)
        return words_out + '. Fix the cart (or show the review is wrong by re-reading the records), then check out.'

    def follow_chain(self) -> tuple[str, bool]:
        live = {need_txt for _, need_txt in self.live_needs()}
        pending = [need_txt for _, need_txt in self.chain_needs if need_txt in live and need_txt not in self.chain_values]
        tool_id = self.tool_roles.get('offer') or self.tool_roles.get('open')
        id_param = self.id_argument(tool_id)
        met: list[str] = []
        lost_ids: list[str] = []
        while pending and self.chain_checks < (6 if self.steps_left() > 15 else 4) and (self.outlay < BUDGET_STOP):
            self.chain_checks += 1
            docs = []
            for pid in reversed(self.listing_order):
                rec_doc = self.read_records.get(pid) or self.listings.get(pid) if pid in self.pages_opened else None
                if rec_doc and len(docs) < 45:
                    docs.append('OPENED ' + as_text(clipped_copy(rec_doc, 300))[:1400])
            for pid in reversed(self.listing_order):
                if len(docs) >= 60:
                    break
                if pid not in self.pages_opened or not (self.read_records.get(pid) or self.listings.get(pid)):
                    docs.append('ROW ' + as_text(clipped_copy(self.listings.get(pid), 80))[:300])
            ask_text_full = CHAIN_FORM + '\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n' + '\n'.join(pending)
            ask_text_full += '\n\nRECORDS (newest first):\n' + '\n'.join(docs)
            json_got = self.ask_for_json(ask_text_full, [SEAT_QUICK] + ([SEAT_DEEP] if self.outlay + DEEP_MARGIN <= BUDGET_STOP else []), 4500) or {}
            self.journal.append({'kind': 'trail', 'pending': pending, 'got': clipped_copy(json_got, 150)})
            reach: list[str] = []
            lost_ids = []
            gaps: list[tuple[str, str]] = []
            for chain in as_list(json_got.get('chains')):
                if not isinstance(chain, dict):
                    continue
                need_txt = next((p for p in pending if p == str(chain.get('need') or '').strip()), None) or (pending[0] if len(pending) == 1 else None)
                if not need_txt:
                    continue
                links = [l for l in as_list(chain.get('links')) if isinstance(l, dict)]
                pids = [str(l.get('id') or '').strip() for l in links]
                if chain.get('done') is True and links and all((i in self.pages_opened for i in pids)) and str(chain.get('value') or '').strip():
                    self.chain_values[need_txt] = str(chain['value']).strip()
                    self.need_version += 1
                    continue
                if links and all((i in self.pages_opened for i in pids)) and (not str(links[-1].get('find') or '').strip()):
                    self.chain_values[need_txt] = str(chain.get('value') or '').strip() or f'as stated on {pids[-1]}'
                    self.need_version += 1
                    continue
                closed = [i for i in pids if i in self.listings and i not in self.pages_opened]
                if closed:
                    reach += [i for i in closed if i not in reach]
                else:
                    gap = next((l for l, i in zip(links, pids) if i not in self.listings), {})
                    topic = str(gap.get('about') or '') or 'the listing it points at'
                    terms = str(gap.get('find') or '').strip()
                    lost_ids.append(f'{need_txt} -> find {topic}' + (f' (search: {terms})' if terms else ''))
                    gaps.append((topic, terms))
            pending = [p for p in pending if p not in self.chain_values]
            if not reach and gaps and tool_id and id_param and (self.steps_left() > 12) and (self.reference_searches < 3):
                reach = self.find_reference(*gaps[0])
                if self.journal and self.journal[-1].get('kind') == 'link_known':
                    met.append(f"{self.journal[-1]['id']}: an opened page of the shop this chain names that shows the values its description gives; read the value from it")
                    lost_ids = lost_ids[1:]
            if not reach or not tool_id or (not id_param) or (self.steps_left() < len(reach) + 4) or self.concluded:
                break
            for pid in reach[:max(0, min(3, 6 - len(met)))]:
                fault_list: list[str] = []
                params = conform({id_param: pid}, self.tool_schemas.get(tool_id), 'args', fault_list)
                if fault_list:
                    break
                reply_obj = self.store_call(tool_id, params)
                if reply_obj is None or self.concluded:
                    break
                self.digest(tool_id, reply_obj.get('observation'))
                met.append(f'{pid}: ' + self.render(reply_obj.get('observation'))[:1500])
        words_out = ''
        if met:
            words_out += '[Wren opened the pages a requirement points at]\n' + '\n'.join(met) + '\n'
        if self.chain_values:
            words_out += '[values read from the pages requirements point at] ' + '; '.join((f'{k} -> {v}' for k, v in self.chain_values.items())) + '\n'
        hold = bool(lost_ids) and self.chain_holds < (3 if self.steps_left() > 15 else 2) and (self.steps_left() > 5)
        if hold:
            self.chain_holds += 1
            words_out += 'The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: ' + ' | '.join(lost_ids[:3]) + ". Search for it by its shop's name together with the values its description gives (the shop's other listings of that kind are not it), open the row of that shop whose page shows those values, apply the value its page gives, then check out."
        return (words_out, hold)

    def rows_of_shop(self, topic: str, pids: list[str]) -> list[str]:
        low = topic.lower()
        acc = []
        for pid in pids:
            row_doc = self.listings.get(pid) or {}
            names = [str(v).strip() for k, v in row_doc.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I)]
            if pid not in self.pages_opened and any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)):
                acc.append(pid)
        return acc

    def opened_from_shop(self, topic: str) -> list[str]:
        low = topic.lower()
        acc = []
        for pid in self.pages_opened:
            row_doc = self.listings.get(pid) or {}
            names = [str(v).strip() for k, v in row_doc.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I)]
            if any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)):
                acc.append(pid)
        return acc

    def find_reference(self, topic: str, terms: str) -> list[str]:
        known: list[str] = []
        named_ids = {w for pid in self.rows_of_shop(topic, list(self.listings)) + self.opened_from_shop(topic) for k, v in (self.listings.get(pid) or {}).items() if isinstance(v, str) and len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I) for w in re.findall('[a-z0-9]{3,}', v.lower())}
        vals = {w for w in re.findall('[a-z0-9]+', (terms or topic).lower()) if len(w) >= 3 or w.isdigit()} - named_ids - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for', 'of'}
        for pid in self.opened_from_shop(topic):
            stack, leaves = ([self.read_records.get(pid) or self.listings.get(pid) or {}], [])
            while stack:
                obj = stack.pop()
                if isinstance(obj, (dict, list)):
                    stack += list(obj.values()) if isinstance(obj, dict) else obj
                else:
                    leaves.append(str(obj))
            words_out = ' '.join(leaves).lower()
            if vals and all((re.search('(?<![a-z0-9])' + re.escape(v) + '(?![a-z0-9])', words_out) for v in vals)):
                self.journal.append({'kind': 'link_known', 'id': pid})
                return []
        proposal = self.search_pattern()
        memo = ' '.join(terms.lower().split())
        if memo and memo in self.references_seen:
            self.reference_searches += 1
            known = self.rows_of_shop(topic, self.references_seen[memo])
        elif proposal and terms:
            tool_id, base, field = proposal
            fault_list: list[str] = []
            params = conform({**base, field: terms}, self.tool_schemas.get(tool_id), 'args', fault_list)
            if not fault_list:
                self.reference_searches += 1
                reply_obj = self.store_call(tool_id, params)
                if reply_obj is not None:
                    tool_result = reply_obj.get('observation')
                    self.digest(tool_id, tool_result)
                    self.references_seen[memo] = [i for i, _ in self.listing_records(tool_result)]
                    known = self.rows_of_shop(topic, self.references_seen[memo])
                    self.journal.append({'kind': 'link_search', 'words': terms[:80], 'rows': known[:4]})
        kind_txt = {w for w in re.findall('[a-z0-9]{3,}', (terms or topic).lower())} - named_ids - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for'}

        def of_kind(pids: list[str]) -> list[str]:
            return [i for i in pids if not kind_txt or kind_txt & set(re.findall('[a-z0-9]{3,}', as_text(self.listings.get(i) or {}).lower()))]
        return (of_kind(known) or of_kind(self.rows_of_shop(topic, list(reversed(self.listing_order)))))[:2]

    def offer_reader(self) -> str | None:
        return self.tool_roles.get('offer') or self.tool_roles.get('open')

    def refresh_offers(self) -> str:
        tool_id = self.offer_reader()
        id_param = self.id_argument(tool_id)
        if not tool_id or not id_param:
            return ''
        since = max(self.steps_used - 2, self.refused_step)
        stale = [i for i in self.basket_ids() if self.offer_read_at.get(i, -99) < since and i not in self.gone_at]
        if not stale or self.steps_left() < len(stale) + 3:
            return ''
        rows_out = []
        for pid in stale:
            fault_list: list[str] = []
            params = conform({id_param: pid}, self.tool_schemas.get(tool_id), 'args', fault_list)
            if fault_list:
                break
            before = dict(self.listings.get(pid) or {})
            reply_obj = self.store_call(tool_id, params)
            if reply_obj is None or self.concluded:
                break
            tool_result = reply_obj.get('observation')
            self.offer_read_at[pid] = self.steps_used
            self.digest(tool_id, tool_result)
            after = self.listings.get(pid) or {}
            moved = [k for k, v in after.items() if k in before and before[k] != v and (not isinstance(v, (dict, list)))]
            if moved:
                rows_out.append(f"{pid} changed ({', '.join(moved[:4])}): {as_text(clipped_copy(tool_result, 300))[:900]}")
        self.journal.append({'kind': 'reread', 'ids': stale, 'changed': len(rows_out)})
        if rows_out:
            self.need_version += 1
        return '\n'.join(rows_out)

    def is_gone(self, pid: str) -> bool:
        return pid in self.unavailable_ids or pid in self.gone_at

    def replacement_hint(self) -> str:
        lost_ids = [i for i in self.basket_ids() if i not in self.gone_at and self.is_gone(i)]
        if not lost_ids:
            return ''
        for pid in lost_ids:
            self.gone_at[pid] = self.steps_used
        self.lead_ids = [i for i in self.lead_ids if i not in lost_ids]
        self.need_version += 1
        now = self.requirement_sheet()
        ladder = [i for l in lost_ids for snap, alts in [self.backups.get(l) or ('', [])] if snap == now for i in alts if not self.is_gone(i) and i not in self.basket_ids() and (i not in lost_ids)]
        band = self.guarded(self.price_band_reads, [], lost_ids)
        choices = band + [i for i in ladder if i not in band]
        if not choices:
            choices = self.substitutes(lost_ids)
        if not choices and self.reread_by_value(sorted({n for n in range(1, len(self.want_list) + 1) for i in lost_ids if i in self.listings_for(n)}), 'sold'):
            choices = self.substitutes(lost_ids)
        if ladder:
            self.journal.append({'kind': 'ladder_used', 'lost': lost_ids, 'picks': ladder})
        words_out = f"[Wren] Sold out now: {', '.join(lost_ids)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        swapped = self.substitute_now(lost_ids, choices)
        if swapped:
            return words_out + swapped
        if choices and self.steps_left() <= 6:
            return words_out + f" Few steps are left: in this turn add {choices[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(choices[1:]) or 'none'}."
        if choices and self.intake.get('pick') == 'cheapest':
            return words_out + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(choices)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {choices[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        if choices:
            return words_out + f" Listings already read that appear to fit that item instead: {', '.join(choices)}. In this turn take the sold-out line out of the cart and add {choices[0]} at the requested quantity, then check out."
        return words_out + ' Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits.'

    def coded_backups(self, row_line: str, basket_tool: list[str]) -> list[str]:
        numbers = [n for n in range(1, len(self.want_list) + 1) if row_line in self.listings_in_scope(n)]
        if len(numbers) != 1:
            return []
        needs = self.want_needs(numbers[0])
        if not needs:
            return []
        acc = []
        for pid in self.listings_in_scope(numbers[0]):
            if pid in basket_tool or pid == row_line or pid not in self.read_records or self.is_gone(pid) or (not self.can_buy(pid)):
                continue
            if RecordJudge.clear(RecordJudge.sheet(needs, self.read_records[pid])):
                acc.append(pid)
        acc.sort(key=lambda i: self.price_for(i) if self.price_for(i) is not None else 1e+18)
        return acc

    def value_reads_before_decline(self) -> str:
        if self.__dict__.get('decline_read') or self.steps_left() < 10:
            return ''
        self.decline_read = True
        proposal = self.search_pattern()
        if not proposal:
            return ''
        tool_id, base, field = proposal
        cap = self.spend_limit if isinstance(self.spend_limit, (int, float)) else None
        hits_by_want: list[str] = []
        opened: list[str] = []
        for slot_no, want_obj in enumerate(self.want_list, 1):
            needs = self.want_needs(slot_no)
            if not needs or self.steps_left() < 10:
                continue
            clear = lambda i: i in self.read_records and RecordJudge.clear(RecordJudge.sheet(needs, self.read_records[i]))
            if any((clear(i) and self.can_buy(i) and (not self.is_gone(i)) for i in self.listings_in_scope(slot_no))):
                continue
            vals = self.value_terms(slot_no)
            tries = [vals] + ([vals[:2]] if len(vals) > 2 else [])
            candidates: list[str] = []
            for terms in tries:
                if not terms or self.steps_left() < 10:
                    continue
                fault_list: list[str] = []
                one_call = conform({**base, field: ' '.join(terms)}, self.tool_schemas.get(tool_id), 'args', fault_list)
                if fault_list:
                    continue
                reply_obj = self.store_call(tool_id, one_call)
                if reply_obj is None or self.concluded:
                    break
                met = reply_obj.get('observation')
                self.digest(tool_id, met)
                pids = [i for i, _ in self.listing_records(met)]
                candidates += [i for i in pids if i in self.listings_in_scope(slot_no) and i not in self.read_records and (i not in candidates) and self.can_buy(i) and (not self.is_gone(i)) and (cap is None or self.price_for(i) is None or self.price_for(i) * want_obj.qty <= cap)]
            candidates.sort(key=lambda i: self.price_for(i) if self.price_for(i) is not None else 1e+18)
            json_got = self.read_details(candidates[:4], self.closing_room() + 4)
            opened += [i for i in candidates[:4] if i in self.read_records]
            hits_by_want += [i for i in candidates[:4] if clear(i)]
            self.journal.append({'kind': 'decline_reads', 'item': slot_no, 'values': vals, 'pool': candidates[:6], 'opened': len(json_got), 'clear': [i for i in candidates[:4] if clear(i)]})
        if not hits_by_want:
            return ''
        return 'Wren searched the requirement values once more and opened these listings: ' + ', '.join(opened) + '. By their own records ' + ', '.join(hits_by_want) + ' meet every requirement and fit the budget. Put the best of them in the cart (respecting the pick rule) and check out; end without a purchase only if its record clearly breaks a requirement.'

    def swap_unproven_line(self) -> str:
        pids = self.basket_ids()
        done = self.__dict__.setdefault('clean_swaps', set())
        stamp = self.check_id(pids)
        if not pids or stamp in done or self.steps_left() < 6 or (self.intake.get('pick') in ('most', 'least')):
            return ''
        done.add(stamp)
        for row_line in pids:
            numbers = [n for n in range(1, len(self.want_list) + 1) if row_line in self.listings_in_scope(n)]
            if len(numbers) != 1 or row_line not in self.read_records:
                continue
            needs = self.want_needs(numbers[0])
            if not needs:
                continue
            bad = [need_txt for need_txt, v, where in RecordJudge.sheet(needs, self.read_records[row_line]) if v in ('differs', 'silent') or (v == 'read' and str(where).startswith('its own notes say'))]
            if not bad:
                continue
            for choice in self.coded_backups(row_line, pids)[:3]:
                note_text = self.substitute_now([row_line], [choice])
                if note_text:
                    self.journal.append({'kind': 'clean_swap', 'line': row_line, 'pick': choice, 'unproven': bad[:3]})
                    return f"Wren replaced {row_line}: its own record does not state {'; '.join(bad[:2])}, while {choice}'s record meets every requirement." + note_text
        return ''

    def substitute_now(self, lost_ids: list[str], choices: list[str]) -> str:
        basket_tool = self.tool_roles.get('cart')
        if not choices or len(lost_ids) != 1 or (not basket_tool) or (self.steps_left() < 4):
            return ''
        rows_out = {i: self.basket_lines.get(i, 1) for i in self.basket_ids() if i not in lost_ids}
        rows_out[choices[0]] = self.basket_lines.get(lost_ids[0], 1)
        if choices[0] not in self.read_records and choices[0] not in self.pages_opened and self.detail_tool():
            if not self.read_details([choices[0]], self.closing_room() + 4):
                return ''
            self.checked_ids.discard(self.check_id(list(rows_out)))
            self.verify_basket(list(rows_out), BUDGET_STOP)
            if not self.basket_verdicts.get(self.check_id(list(rows_out))):
                return ''
        each = self.intake.get('budget_scope') == 'each'
        if self.spend_limit is not None and ((self.price_for(choices[0]) or 0) > self.spend_limit if each else sum(((self.price_for(i) or 0) * q for i, q in rows_out.items())) > self.spend_limit):
            return ''
        pos = self.fitment_index()
        devices = {model_code for i in rows_out if i != choices[0] for model_code in pos['marks'].get(i, {}).get('codes') or []}
        if devices and self.fitment_stated(choices[0], pos) and (not self.fits_model(choices[0], devices, pos)):
            return ''
        call_list = self.basket_calls(rows_out, lost_ids)
        if not call_list or len(call_list) + 2 > self.steps_left():
            return ''
        for params in call_list:
            fault_list: list[str] = []
            params = conform(params, self.tool_schemas.get(basket_tool), 'args', fault_list)
            if fault_list:
                return ''
            reply_obj = self.store_call(basket_tool, params)
            if reply_obj is None or self.concluded:
                return ''
            self.record_basket(basket_tool, params)
            self.digest(basket_tool, reply_obj.get('observation'))
        self.journal.append({'kind': 'swap_in', 'lost': lost_ids, 'pick': choices[0]})
        self.planned_lines = list(rows_out)
        self.late_hold_done = False
        cheaper = self.intake.get('pick') == 'cheapest' and self.steps_left() > 10
        return f" Wren already put {choices[0]} in its place ({('the cheapest listing read that appears to fit' if self.intake.get('pick') == 'cheapest' else 'a listing read that appears to fit')}; other candidates: {', '.join(choices[1:]) or 'none'}). Cart now: " + self.basket_text() + '. ' + ('If you have not yet seen every fitting listing, look once for a cheaper fit under its price; otherwise check out.' if cheaper else 'Check out now unless it clearly breaks a requirement.')

    def gone_in_basket(self) -> str:
        pids = [i for i in self.basket_ids() if i not in self.gone_at]
        if any((self.is_gone(i) for i in pids)):
            return ''
        if len(pids) == 1:
            self.unavailable_ids.add(pids[0])
            return ''
        return self.refresh_offers() if pids else ''

    def shares_kind(self, pid: str, lost_ids: list[str]) -> bool:
        shelf_key = self.shelf_field()
        if not shelf_key:
            return True
        owners = {self.want_for_label(str((self.listings.get(i) or {}).get(shelf_key) or '')) for i in lost_ids} - {0}
        return not owners or self.want_for_label(str((self.listings.get(pid) or {}).get(shelf_key) or '')) in owners

    def substitutes(self, lost_ids: list[str]) -> list[str]:
        if not self.check_seats():
            return []
        docs = []
        for pid in reversed(self.listing_order):
            if pid in lost_ids or pid in self.gone_at or self.is_gone(pid):
                continue
            docs.append(('OPENED ' if pid in self.read_records or pid in self.pages_opened else 'ROW ') + as_text(clipped_copy(self.read_records.get(pid) or self.listings.get(pid), 200))[:1200])
            if len(docs) >= 25:
                break
        if not docs:
            return []
        ask_text_full = RESTOCK_FORM + '\n\nREQUEST:\n' + self.request_text + '\n\nREQUIREMENTS IN FORCE:\n' + self.requirement_sheet()
        ask_text_full += '\n\nSOLD-OUT CART LINE:\n' + '\n'.join((f'{i}: {self.record_text(i)}' for i in lost_ids)) + '\n\nCART NOW:\n' + self.basket_text()
        ask_text_full += '\n\nOTHER LISTING RECORDS SEEN (newest first; OPENED = full record read, ROW = search row only, which can neither pass nor fail a property it does not show):\n' + '\n'.join(docs)
        json_got = self.ask_for_json(ask_text_full, self.check_seats(), 3500)
        fits: list[tuple[float, str]] = []
        for rec in as_list((json_got or {}).get('fits')):
            pid = str(rec.get('id') if isinstance(rec, dict) else rec)
            price = rec.get('price') if isinstance(rec, dict) else None
            if pid in self.listings and pid not in lost_ids and (pid not in self.gone_at) and (pid not in self.basket_ids()) and (pid not in [i for _, i in fits]) and self.shares_kind(pid, lost_ids):
                known = self.price_for(pid)
                fits.append((known if known is not None else float(price) if isinstance(price, (int, float)) and (not isinstance(price, bool)) else float('inf'), pid))
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.intake.get('pick') == 'cheapest' else [i for _, i in fits]
        self.journal.append({'kind': 'swap', 'lost': lost_ids, 'fits': fits})
        return ranked[:4]

    def contradicts(self, pid: str, need_txt: str) -> bool | None:
        verdict = RecordJudge.sheet([need_txt], self.read_records.get(pid) or {})[0][1]
        if verdict == 'differs':
            return True
        if verdict in ('meets', 'elsewhere'):
            return False
        return None
        prop_words, _, val = need_txt.partition(':')
        val = val.strip().lower()
        if not prop_words.strip() or not val or re.search('\\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or|same|similar|near)\\b|\\d', val):
            return None
        block = facts_of(self.read_records.get(pid) or {}) or {}
        field_names = [k for k in block if name_covers(word_roots(prop_words), word_roots(k))]
        if not field_names:
            return None
        note_text = ' '.join((str(block[k]) for k in field_names)).lower()
        wanted = self.word_tokens(val)
        if not wanted:
            return None
        return not bool(wanted & self.word_tokens(note_text))

    def clean_choices(self, partial: bool=False) -> dict[int, str]:
        shelf_key = self.shelf_field()
        choices: dict[int, str] = {}
        cap = self.spend_limit if isinstance(self.spend_limit, (int, float)) else None
        used: set[str] = set()
        for slot_no, want_obj in enumerate(self.want_list, 1):
            needs = [need_txt for n, need_txt in self.live_needs() if n == slot_no]
            scored = []
            for pid in self.read_records:
                if pid in used or not self.can_buy(pid) or pid in self.unavailable_ids:
                    continue
                if shelf_key and (not self.belongs_to(want_obj, str((self.listings.get(pid) or {}).get(shelf_key) or ''))):
                    continue
                price = self.price_for(pid)
                qty = want_obj.qty if self.intake.get('budget_scope') != 'each' else 1
                if cap is not None and (price is None or price * qty > cap):
                    continue
                if self.basket_verdicts.get(self.check_id([pid])) is False:
                    continue
                verdicts = [self.contradicts(pid, need_txt) for need_txt in needs]
                if any((v is True for v in verdicts)):
                    continue
                sheet = RecordJudge.sheet([need_txt for need_txt in needs if need_txt not in [m for _, m in self.chain_needs]], self.read_records.get(pid) or {})
                if any((v in ('differs', 'silent') for _, v, _ in sheet)):
                    continue
                words_out = as_text({k: v for k, v in (self.read_records.get(pid) or {}).items() if isinstance(v, str)}).lower()
                stated_now = [need_txt for need_txt in want_obj.told if need_txt not in want_obj.dropped]
                if any((self.contradicts(pid, need_txt) is not False and (not self.word_tokens(need_txt.partition(':')[2]) & self.word_tokens(words_out)) for need_txt in stated_now)):
                    continue
                stated_now = sum((1 for need_txt, v in zip(needs, verdicts) if v is None and self.word_tokens(need_txt.partition(':')[2]) & self.word_tokens(words_out)))
                stated = sum((1 for v in verdicts if v is False))
                scored.append((-stated, -stated_now, price if self.intake.get('pick') == 'cheapest' and price is not None else 0.0, pid))
            if scored:
                choices[slot_no] = min(scored)[3]
                used.add(choices[slot_no])
        return choices if partial or len(choices) == len(self.want_list) else {}

    def turn_decline_into_order(self) -> str:
        if self.__dict__.get('decline_turned') or not self.tool_roles.get('cart') or (not self.tool_roles.get('buy')):
            return ''
        choices = self.solved_picks() or self.clean_choices()
        if not choices or len(choices) != len(self.want_list):
            return ''
        self.decline_turned = True
        rows_out = {pid: self.want_list[slot_no - 1].qty for slot_no, pid in choices.items()}
        drop = [i for i in self.basket_ids() if i not in rows_out]
        call_list = self.basket_calls(rows_out, drop)
        if not call_list or len(call_list) + 2 > self.steps_left():
            return ''
        basket_tool = self.tool_roles['cart']
        for one_call in call_list:
            fault_list: list[str] = []
            one_call = conform(one_call, self.tool_schemas.get(basket_tool), 'args', fault_list)
            if fault_list:
                return ''
            reply_obj = self.store_call(basket_tool, one_call)
            if reply_obj is None or self.concluded:
                return ''
            self.record_basket(basket_tool, one_call)
            self.digest(basket_tool, reply_obj.get('observation'))
        self.planned_lines = list(rows_out)
        self.journal.append({'kind': 'decline_converted', 'lines': rows_out})
        note_text = self.close_out()
        return 'Not run: ending without a purchase earns nothing while the records hold a listing for every item that breaks no requirement. Wren carted that set instead (' + ', '.join((f'item {n}: {i}' for n, i in sorted(choices.items()))) + ') and attempted the checkout.' + (' The store answered: ' + note_text if note_text else '')

    def order_rescue(self) -> str:
        if self.order_salvaged or self.basket_ids() or self.concluded or (not self.tool_roles.get('cart')) or (not self.tool_roles.get('buy')):
            return ''
        self.order_salvaged = True
        choices = self.clean_choices() or (self.solved_picks() if len(self.want_list) > 1 else {})
        if not choices:
            return ''
        rows_out = {pid: self.want_list[slot_no - 1].qty for slot_no, pid in choices.items()}
        call_list = self.basket_calls(rows_out, [])
        if not call_list or len(call_list) + 1 > self.steps_left():
            return ''
        for params in call_list:
            fault_list: list[str] = []
            params = conform(params, self.tool_schemas.get(self.tool_roles['cart']), 'args', fault_list)
            if fault_list:
                return ''
            reply_obj = self.store_call(self.tool_roles['cart'], params)
            if reply_obj is None or self.concluded:
                return ''
            self.record_basket(self.tool_roles['cart'], params)
            self.digest(self.tool_roles['cart'], reply_obj.get('observation'))
        self.journal.append({'kind': 'rescue_order', 'lines': rows_out})
        return 'carted ' + ', '.join(rows_out)

    def review_decline(self) -> str:
        if self.reviewed or not self.check_seats():
            return ''
        self.reviewed = True
        row_list = [i for n in range(1, len(self.want_list) + 1) for i in self.rows_naming_values(n)[:2]]
        if row_list and self.steps_left() >= self.closing_room() + 6:
            self.read_rows(row_list[:2])
            self.journal.append({'kind': 'audit_named', 'rows': row_list[:2]})
        json_got = self.review_call()
        named_ids = self.review_picks(json_got)
        if not named_ids:
            row_list = [str(i) for i in as_list((json_got or {}).get('open')) if str(i) in self.listings and str(i) not in self.pages_opened and (not self.is_gone(str(i)))]
            if row_list and self.read_rows(row_list):
                json_got = self.review_call()
                named_ids = self.review_picks(json_got)
        if not named_ids:
            return ''
        self.leads_by_want = named_ids
        self.lead_ids = [i for pids in named_ids.values() for i in pids]
        return 'A careful review thinks these listings may fit: ' + '; '.join((f"item {k}: {', '.join(v)}" for k, v in named_ids.items())) + (f" ({json_got.get('reason')})" if json_got.get('reason') else '') + '. Open them and buy what truly fits; end without a purchase only if each one fails.'

    def review_call(self, models: list[str] | None=None) -> dict[str, Any] | None:
        docs = []
        for pid in reversed(self.listing_order):
            if self.is_gone(pid):
                continue
            rec_doc = self.read_records.get(pid) if pid in self.read_records else None
            docs.append('OPENED ' + as_text(clipped_copy(rec_doc, 200))[:1200] if rec_doc else 'ROW ' + as_text(clipped_copy(self.listings.get(pid), 120))[:500])
            if len(docs) >= 30:
                break
        if not docs:
            return None
        ask_text_full = REVIEW_FORM + '\n\nREQUEST:\n' + self.request_text + '\n\nREQUIREMENTS IN FORCE (a line marked as deciding nothing is an aside, not a requirement):\n' + self.requirement_sheet()
        ask_text_full += '\n\nLISTING RECORDS SEEN (newest first):\n' + '\n'.join(docs)
        json_got = self.ask_for_json(ask_text_full, models or self.check_seats(), 4000)
        self.journal.append({'kind': 'audit', 'got': json_got})
        if isinstance(json_got, dict) and str(json_got.get('closest') or '') in self.listings:
            self.nearest = str(json_got['closest'])
        return json_got

    def review_picks(self, json_got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (json_got or {}).get('fits')
        if not isinstance(fits, dict):
            return {}
        named_ids = {str(k): [str(i) for i in as_list(v) if str(i) in self.listings and (not self.is_gone(str(i)))][:4] for k, v in fits.items()}
        return {k: v for k, v in named_ids.items() if v}

    def basket_calls(self, rows_out: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        tool_name = self.tool_roles.get('cart') or ''
        spec_doc = self.tool_schemas.get(tool_name) or {}
        prop_specs = spec_doc.get('properties') or {}
        for key, field_spec in prop_specs.items():
            layout = field_spec.get('items') if isinstance(field_spec, dict) and 'array' in schema_kinds(field_spec) else None
            if isinstance(layout, dict) and isinstance(layout.get('properties'), dict):
                inner = layout['properties']
                id_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'string' in schema_kinds(s) and ('enum' not in s)), None)
                count_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'integer' in schema_kinds(s)), None)
                if not id_key:
                    return None
                return [{key: [{id_key: i, **({count_key: n} if count_key else {})} for i, n in rows_out.items()]}]
        text_fields = [k for k, s in prop_specs.items() if isinstance(s, dict) and 'string' in schema_kinds(s) and ('enum' not in s)]
        id_param = next((k for t, k in self.id_arguments if t == tool_name and k in text_fields), None) or next((k for k in spec_doc.get('required') or [] if k in text_fields), None) or (text_fields[0] if len(text_fields) == 1 else None)
        ints = [k for k, s in prop_specs.items() if isinstance(s, dict) and 'integer' in schema_kinds(s)]
        named_ids = [k for k in ints if re.search(QTY_WORDS, plain_words(k)) and (not re.search(INDEX_WORDS, plain_words(k)))]
        num = named_ids[0] if named_ids else ints[0] if len(ints) == 1 else None
        mode, adds, dels = self.cart_mode_words(prop_specs)
        if not id_param:
            return None
        call_list: list[dict[str, Any]] = []
        for pid in drop:
            held_now = self.basket_lines.get(pid, 1)
            if mode:
                call_list.append({id_param: pid, mode: dels, **({num: held_now} if num else {})})
            elif num and self.basket_replaces(tool_name):
                call_list.append({id_param: pid, num: 0})
            elif num and isinstance(prop_specs[num].get('minimum'), int) and (prop_specs[num]['minimum'] <= -held_now):
                call_list.append({id_param: pid, num: -held_now})
            else:
                return None
        for pid, want in rows_out.items():
            held_now = self.basket_lines.get(pid, 0) if pid in self.basket_ids() else 0
            if held_now == want:
                continue
            if held_now and (not self.basket_replaces(tool_name)):
                return None
            call_list.append({id_param: pid, **({mode: adds} if mode else {}), **({num: want} if num else {})})
        return call_list

    @staticmethod
    def cart_mode_words(prop_specs: dict[str, Any]) -> tuple[str | None, Any, Any]:
        for key, field_spec in prop_specs.items():
            opts = [o for o in field_spec.get('enum') or [] if isinstance(o, str)] if isinstance(field_spec, dict) else []
            dels = [o for o in opts if re.search(DEL_WORDS, plain_words(o))]
            adds = [o for o in opts if o not in dels and re.search(ADD_WORDS, plain_words(o))]
            if len(opts) >= 2 and dels and adds:
                return (key, adds[0], dels[0])
        for key, field_spec in prop_specs.items():
            opts = [o for o in field_spec.get('enum') or [] if isinstance(o, str)] if isinstance(field_spec, dict) else []
            if len(opts) != 2:
                continue
            dels = [o for o in opts if re.search(DEL_WORDS, plain_words(o))]
            adds = [o for o in opts if o not in dels and re.search(ADD_WORDS, plain_words(o))]
            if dels and (not adds):
                return (key, next((o for o in opts if o not in dels)), dels[0])
            if adds and (not dels):
                return (key, adds[0], next((o for o in opts if o not in adds)))
        typed = any((re.search(DEL_WORDS + '|' + ADD_WORDS, plain_words(str(o))) for s in prop_specs.values() if isinstance(s, dict) for o in s.get('enum') or []))
        key = next((k for k, s in prop_specs.items() if isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) == 2)), None)
        if key and (not typed):
            return (key, prop_specs[key]['enum'][0], prop_specs[key]['enum'][1])
        return (None, None, None)

    def finish_basket(self) -> str:
        basket_tool, checkout_tool = (self.tool_roles.get('cart'), self.tool_roles.get('buy'))
        if self.basket_finished or not self.approved_pick or self.is_gone(self.approved_pick) or (not basket_tool) or (not checkout_tool):
            return ''
        self.basket_finished = True
        shelf_key = self.shelf_field()

        def kind_txt(pid: str) -> str:
            return str((self.listings.get(pid) or {}).get(shelf_key) or '').strip().lower() if shelf_key else pid
        proposal: dict[str, int] = {}
        for slot_no, pids in self.leads_by_want.items():
            pos = as_count(slot_no)
            choice = next((i for i in pids if not self.is_gone(i)), None)
            if choice and pos and (1 <= pos <= len(self.want_list)) and (kind_txt(choice) not in {kind_txt(p) for p in proposal}):
                proposal[choice] = self.want_list[pos - 1].qty
        if self.approved_pick not in proposal:
            return ''
        now = self.basket_ids()
        kept_lines = {i: self.basket_lines.get(i, 1) for i in now if i not in proposal and (not self.is_gone(i)) and (kind_txt(i) not in {kind_txt(p) for p in proposal})}
        drop = [i for i in now if i not in proposal and i not in kept_lines]
        if len(proposal) + len({kind_txt(i) for i in kept_lines}) < len(self.want_list):
            return ''
        rows_out = {**kept_lines, **proposal}
        call_list = self.basket_calls(rows_out, drop)
        if not call_list or len(call_list) + 1 > self.steps_left():
            return ''
        self.read_details(list(rows_out), len(call_list) + 3)
        self.verify_basket(list(rows_out), BUDGET_STOP)
        if not self.basket_verdicts.get(self.check_id(list(rows_out))):
            return ''
        done = []
        for params in call_list:
            fault_list: list[str] = []
            params = conform(params, self.tool_schemas.get(basket_tool), 'args', fault_list)
            if fault_list:
                return ''
            reply_obj = self.store_call(basket_tool, params)
            if reply_obj is None or self.concluded:
                return ''
            self.record_basket(basket_tool, params)
            self.digest(basket_tool, reply_obj.get('observation'))
            done.append(as_text(params))
        fault_list = []
        params = conform(dict(self.intake.get('buy_args') or {}), self.tool_schemas.get(checkout_tool), 'args', fault_list)
        if fault_list:
            return ''
        replies = []
        for attempt in range(2):
            if attempt:
                self.refresh_offers()
                self.verify_basket(list(rows_out), BUDGET_STOP)
                if not self.basket_verdicts.get(self.check_id(list(rows_out))) or self.steps_left() < 1 or self.concluded or any((self.is_gone(i) for i in rows_out)):
                    break
            go, note_text = self.run_step('order')
            if note_text:
                replies.append(note_text)
            if not go:
                break
            reply_obj = self.store_call(checkout_tool, params)
            self.journal.append({'kind': 'completed', 'lines': rows_out, 'drop': drop, 'reply': clipped_copy(reply_obj, 200)})
            if reply_obj is None:
                break
            self.checkouts += 1
            if not reply_obj.get('finished'):
                self.refused_step = self.steps_used
            self.digest(checkout_tool, reply_obj.get('observation'))
            replies.append(self.render(reply_obj.get('observation')))
            if self.concluded or reply_obj.get('finished'):
                break
        if not replies:
            return ''
        return 'Not run (Wren acted instead): the independent check confirmed ' + self.approved_pick + ', so Wren set the cart (' + '; '.join(done) + ') and checked out. Store reply: ' + ' | then: '.join(replies)

    def is_checkout(self, tool_name: str, params: dict[str, Any]) -> bool:
        if tool_name != self.tool_roles.get('buy'):
            return False
        fixed_args = self.intake.get('buy_args') if isinstance(self.intake.get('buy_args'), dict) else {}
        return all((params.get(k) == v for k, v in fixed_args.items()))

    def is_decline(self, tool_name: str, params: dict[str, Any]) -> bool:
        if tool_name != self.tool_roles.get('refuse'):
            return False
        fixed_args = self.intake.get('refuse_args') if isinstance(self.intake.get('refuse_args'), dict) else {}
        return all((params.get(k) == v for k, v in fixed_args.items()))

    def decline_text(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        spec_doc = self.tool_schemas.get(tool_name) or {}
        prop_specs = spec_doc.get('properties') or {}
        acc = dict(params)
        for key, field_spec in prop_specs.items():
            if not isinstance(field_spec, dict) or 'string' not in schema_kinds(field_spec) or 'enum' in field_spec:
                continue
            val = acc.get(key)
            if val is None and key not in (spec_doc.get('required') or []) and (self.intake.get('quiet_decline') is True or not re.search('justif|reason|remark|note|message|explain|comment|why|text', key, re.I)):
                continue
            words_out = str(val or '')
            if key in (self.intake.get('refuse_args') or {}):
                continue
            if self.intake.get('quiet_decline') is True:
                acc[key] = 'Sorry, nothing in the store meets all of your requirements, so I am not buying anything.'[:field_spec.get('maxLength', 200)]
                continue
            ceiling = field_spec.get('maxLength', 1000)
            if not any((pid in words_out for pid in self.listings)) or (self.nearest and self.nearest not in words_out):
                note = self.nearest_text()[:ceiling]
                words_out = (words_out or 'Nothing in the store meets every one of your requirements, so I am not buying anything.')[:max(0, ceiling - len(note))] + note
            acc[key] = words_out[:ceiling]
        return acc

    def nearest_text(self) -> str:
        candidates = [i for i in ([self.nearest] if self.nearest else []) + self.lead_ids + self.basket_ids() + self.planned_lines if i in self.listings]
        if not candidates and self.want_list:
            candidates = [i for i in self.listings_for(1) if i in self.read_records and (not self.is_gone(i))]
        if not candidates:
            return ''
        pid = candidates[0]
        row_doc = self.read_records.get(pid) or self.listings.get(pid) or {}
        title = next((str(v) for k, v in row_doc.items() if isinstance(v, str) and re.search('title|heading', k, re.I)), '') or next((str(v) for k, v in row_doc.items() if isinstance(v, str) and re.search('name', k, re.I) and (not re.search('brand|shop|seller|maker', k, re.I))), '')
        price = self.price_for(pid)
        return f' The closest listing is {pid} ({title[:80]}' + (f', price {price:g}' if price is not None else '') + (', in stock' if self.can_buy(pid) else '') + '), which misses at least one requirement.'

    def basket_replaces(self, tool_name: str) -> bool:
        prop_specs = self.tool_schemas.get(tool_name, {}).get('properties') or {}
        if any((isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) >= 2) for s in prop_specs.values())):
            return False
        return any((isinstance(s, dict) and 'integer' in schema_kinds(s) and (s.get('minimum') == 0) for s in prop_specs.values()))

    def held_lines_kept(self, tool_name: str, params: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if len(self.want_list) < 2 or not self.basket_lines:
            return (params, '')
        prop_specs = (self.tool_schemas.get(tool_name) or {}).get('properties') or {}
        for key, field_spec in prop_specs.items():
            layout = field_spec.get('items') if isinstance(field_spec, dict) and 'array' in schema_kinds(field_spec) else None
            if not (isinstance(layout, dict) and isinstance(layout.get('properties'), dict)) or not isinstance(params.get(key), list) or (not params[key]):
                continue
            inner = layout['properties']
            id_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'string' in schema_kinds(s) and ('enum' not in s)), None)
            count_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'integer' in schema_kinds(s)), None)
            if not id_key:
                return (params, '')
            owner = lambda i: next((n for n in range(1, len(self.want_list) + 1) if i in self.listings_for(n)), 0)
            delivered = [str(row_line.get(id_key) or '') for row_line in params[key] if isinstance(row_line, dict)]
            covered = {owner(i) for i in delivered} - {0}
            kept_lines = [i for i, c in self.basket_lines.items() if i not in delivered and c and (not self.is_gone(i)) and owner(i) and (owner(i) not in covered)]
            if not kept_lines:
                return (params, '')
            rows_out = list(params[key]) + [{id_key: i, **({count_key: self.basket_lines[i]} if count_key else {})} for i in kept_lines]
            self.journal.append({'kind': 'cart_lines_kept', 'kept': kept_lines})
            return ({**params, key: rows_out}, f"\n[Wren] This cart tool replaces the whole selection, so the lines already carted for the other items ({', '.join(kept_lines)}) were kept in it.")
        return (params, '')

    def record_basket(self, tool_name: str, params: dict[str, Any]) -> None:
        self.basket_touched = True
        self.basket_changed_at = self.steps_used
        field_spec = self.tool_schemas.get(tool_name, {}).get('properties') or {}
        mode, _, dels = self.cart_mode_words(field_spec)
        note_text = plain_words(str(params.get(mode))) if mode and params.get(mode) is not None else ''
        removal = bool(mode) and (params.get(mode) == dels or bool(re.search(DEL_WORDS, note_text)))
        set_word = bool(re.search('\\b(?:set\\w*|updat\\w*|chang\\w*|replac\\w*)\\b', note_text))
        loose_count = next((as_count(v) for v in params.values() if as_count(v) is not None and (not isinstance(v, str))), None)
        rows_out: list[tuple[str, int | None]] = []
        whole = False
        for key, val in params.items():
            if isinstance(val, list):
                layout = (field_spec.get(key) or {}).get('items') or {}
                if not val and isinstance(layout, dict) and (layout.get('type') == 'object' or 'properties' in layout):
                    whole = True
                for row_line in val:
                    if isinstance(row_line, dict):
                        whole = True
                        pid = next((v for v in row_line.values() if isinstance(v, str) and v in self.listings), None)
                        num = next((as_count(v) for v in row_line.values() if as_count(v) is not None and (not isinstance(v, str))), 1)
                        if pid:
                            rows_out.append((pid, num))
                    elif isinstance(row_line, str) and row_line in self.listings:
                        rows_out.append((row_line, loose_count))
            elif isinstance(val, str) and val in self.listings:
                rows_out.append((val, loose_count))
        if whole:
            self.basket_lines = {i: c for i, c in rows_out if c and c > 0}
            return
        sets = set_word or self.basket_replaces(tool_name)
        for pid, num in rows_out:
            held_now = self.basket_lines.get(pid, 0)
            if removal:
                held_now = held_now - num if num else 0
            elif sets:
                held_now = num if num is not None else 1
            else:
                held_now = held_now + (num if num is not None else 1)
            if held_now > 0:
                self.basket_lines[pid] = held_now
            else:
                self.basket_lines.pop(pid, None)

    def salvage_basket(self) -> None:
        basket_tool = self.tool_roles.get('cart')
        if self.salvaged or self.concluded or self.halted or (not basket_tool) or (not self.tool_roles.get('buy')) or self.basket_ids() or self.basket_lines:
            return
        self.salvaged = True
        if self.steps_left() < len(self.want_list) + 3 or self.past_clock(-30):
            return
        choices: dict[str, int] = dict(self.fitting_set())
        json_got = self.review_call([SEAT_QUICK]) if self.check_seats() and (not self.past_clock()) else None
        named_ids = self.review_picks(json_got)
        shelf_key = self.shelf_field()
        kinds: set[str] = set()
        shelf_key = self.shelf_field()
        kinds = {str((self.listings.get(i) or {}).get(shelf_key) or i) for i in choices}
        for slot_no, want_obj in enumerate(self.want_list, 1):
            if any((i in self.listings_for(slot_no) for i in choices)):
                continue
            pids = [i for i in named_ids.get(str(slot_no), []) if not self.is_gone(i)]
            if not pids:
                pids = [i for i in self.planned_lines if i in self.listings_for(slot_no) and self.can_buy(i)]
            if not pids:
                mine = [i for i in self.listings_for(slot_no) if self.can_buy(i)]
                candidates = [i for i in mine if i in self.read_records]
                wants = [m for m in want_obj.musts + want_obj.told if m not in want_obj.dropped]
                shows = lambda i: sum((1 for m in wants if all((w in as_text(self.read_records.get(i)).lower() for w in re.findall('[a-z0-9]{3,}', m.partition(':')[2].lower())[:3]))))
                pids = sorted(candidates, key=lambda i: (-shows(i), self.price_for(i) if self.price_for(i) is not None else 1e+18))
            choice = next((i for i in pids if str((self.listings.get(i) or {}).get(shelf_key) or i) not in kinds), None)
            if not choice:
                self.journal.append({'kind': 'rescue', 'missing': slot_no})
                return
            kinds.add(str((self.listings.get(choice) or {}).get(shelf_key) or choice))
            choices[choice] = want_obj.qty
        each = self.intake.get('budget_scope') == 'each'
        if self.spend_limit is not None and (any(((self.price_for(i) or 0) > self.spend_limit for i in choices)) if each else sum(((self.price_for(i) or 0) * q for i, q in choices.items())) > self.spend_limit):
            self.journal.append({'kind': 'rescue', 'over_budget': choices})
            return
        call_list = self.basket_calls(choices, [])
        if not call_list or len(call_list) + 2 > self.steps_left():
            return
        for params in call_list:
            fault_list: list[str] = []
            params = conform(params, self.tool_schemas.get(basket_tool), 'args', fault_list)
            if fault_list:
                return
            reply_obj = self.store_call(basket_tool, params)
            if reply_obj is None or self.concluded:
                return
            self.record_basket(basket_tool, params)
            self.digest(basket_tool, reply_obj.get('observation'))
        self.journal.append({'kind': 'rescue', 'cart': choices})

class Helm:

    def commit_ready_now(self) -> bool:
        pids = self.basket_ids()
        if not pids or self.intake.get('pick') not in (None, 'any') or self.chain_needs or self.fit_pairs:
            return False
        if self.messages_left() > 0 and (not self.interviewed):
            return False
        if self.step_pending('choose') or self.step_pending('order'):
            return False
        if not self.basket_verdicts.get(self.check_id(pids)):
            return False
        return bool(self.proven_by_code(pids))

    def tool_menu(self) -> list[dict[str, Any]]:
        left = self.steps_left()
        ask = self.tool_roles.get('ask')
        kept_lines: set[str] | None = None
        ends = {self.tool_roles.get('buy'), self.tool_roles.get('refuse')} - {None}
        if left <= self.closing_room():
            kept_lines = set(ends)
        elif left <= self.closing_room() + 3:
            kept_lines = set(ends) | {self.tool_roles.get('cart'), self.tool_roles.get('cart_view')}
        elif left <= len(self.want_list) + 5 and self.checkouts == 0:
            kept_lines = set(ends) | {self.tool_roles.get('cart'), self.tool_roles.get('cart_view'), self.tool_roles.get('offer')}
        tools = [t for t in self.tool_specs if t['function']['name'] != ask]
        if kept_lines is not None and ends:
            narrowed = [t for t in tools if t['function']['name'] in kept_lines]
            tools = narrowed or tools
        return tools

    def execute_calls(self, call_list: list[dict[str, Any]]) -> list[tuple[str, str]]:
        acc: list[tuple[str, str]] = []
        call_list = self.merge_reads(call_list)
        for pos, one_call in enumerate(call_list):
            allowed = {t['function']['name'] for t in self.tool_menu()}
            call_id = one_call['id']
            if one_call.get('joined'):
                acc.append((call_id, 'Shown in the side-by-side result above: Wren opened these listings together in one step.'))
                continue
            fn = one_call.get('function') or {}
            tool_name = fn.get('name')
            raw = fn.get('arguments')
            if self.concluded:
                acc.append((call_id, 'Not run: the episode is over.'))
                continue
            if tool_name == self.tool_roles.get('ask'):
                acc.append((call_id, "Not run (no step used): Wren's interviewer handles every message to the shopper."))
                continue
            if tool_name not in self.tool_schemas:
                acc.append((call_id, f"Not run (no step used): unknown tool. Tools: {', '.join(sorted(allowed))}."))
                continue
            if tool_name not in allowed:
                acc.append((call_id, 'Not run (no step used): too few steps are left for that; use: ' + ', '.join(sorted(allowed)) + '.'))
                continue
            if isinstance(raw, dict):
                params = raw
            else:
                params = json_object(raw) if (raw or '').strip() else {}
                if params is None:
                    acc.append((call_id, 'Not run (no step used): the arguments were not one JSON object.'))
                    continue
            fault_list: list[str] = []
            params = conform(params, self.tool_schemas[tool_name], 'args', fault_list)
            if fault_list:
                acc.append((call_id, 'Not run (no step used): ' + '; '.join(fault_list[:6])))
                continue
            params, wrong = self.fix_ids(tool_name, params)
            if wrong:
                acc.append((call_id, 'Not run (no step used): ' + wrong + '.'))
                continue
            trimmed = ''
            multi_arg = self.multi_view_arg() if tool_name == self.tool_roles.get('compare') else None
            if multi_arg and isinstance(params.get(multi_arg), list) and (len(params[multi_arg]) > self.multi_view_size()) and self.multi_view_bad:
                room = self.multi_view_size()
                trimmed = f"\n[Wren] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, params[multi_arg][room:]))}."
                params = {**params, multi_arg: params[multi_arg][:room]}
            id_param = self.id_argument(tool_name)
            if id_param and tool_name in (self.tool_roles.get('open'), self.tool_roles.get('offer')) and (str(params.get(id_param)) in self.gone_at):
                acc.append((call_id, f'Not run (no step used): {params[id_param]} sold out at step {self.gone_at[str(params[id_param])]} and will not come back. Replace its cart line with another fitting listing.'))
                continue
            if tool_name == self.tool_roles.get('cart') and any(((as_count(v) or 0) < 0 for v in params.values() if not isinstance(v, str))):
                view = as_text(self.basket_view) if self.basket_viewed_at >= self.basket_changed_at else ''
                absent = [v for v in params.values() if isinstance(v, str) and v in self.listings and (v not in self.basket_lines) and (v not in view) and (v not in self.basket_said)]
                if absent:
                    acc.append((call_id, f'Not run (no step used): {absent[0]} is not in the cart, so there is nothing to take off. Cart now: ' + self.basket_text()))
                    continue
            if tool_name == self.tool_roles.get('cart'):
                prop_specs = (self.tool_schemas.get(tool_name) or {}).get('properties') or {}
                mode, on, off = self.cart_mode_words(prop_specs)
                pids = [v for v in params.values() if isinstance(v, str) and v in self.listings]
                view = as_text(self.basket_view) if self.basket_viewed_at >= self.basket_changed_at else ''
                removing = bool(mode) and params.get(mode) == off
                adding = (not mode or params.get(mode) == on) and (not any(((as_count(v) or 0) < 0 for v in params.values() if not isinstance(v, str))))
                if pids and removing and all((i not in self.basket_lines and i not in view and (i not in self.basket_said) for i in pids)):
                    acc.append((call_id, f'Not run (no step used): {pids[0]} is not in the cart now, so there is nothing to take off. Cart now: ' + self.basket_text()))
                    continue
                if pids and adding and (not self.basket_replaces(tool_name)) and all((i in self.basket_lines and self.units_held(i) == self.wanted_units(i) for i in pids)):
                    acc.append((call_id, f'Not run (no step used): {pids[0]} is already in the cart at the requested quantity; adding it again would double that line. Cart now: ' + self.basket_text()))
                    continue
            if tool_name == self.tool_roles.get('cart'):
                params, kept_note = self.guarded(self.held_lines_kept, (params, ''), tool_name, params)
                trimmed += kept_note
            ending = self.is_checkout(tool_name, params) or self.is_decline(tool_name, params)
            if ending and len(call_list) > 1:
                acc.append((call_id, 'Not run (no step used): send the finishing call on its own, after reading your other results.'))
                continue
            if not ending and self.steps_left() <= self.closing_room() and (self.tool_roles.get('buy') or self.tool_roles.get('refuse')):
                acc.append((call_id, "Not run (no step used): the last steps are kept for checking out (with the shopper's approval of the basket when it is due) or ending without a purchase."))
                continue
            if not self.interviewed and self.tool_roles.get('ask') and (self.messages_left() > 0) and (tool_name in {self.tool_roles.get(r) for r in ('search', 'open', 'offer', 'compare')} - {None}) and (self.steps_left() <= self.messages_left() + 5):
                acc.append((call_id, "Not run (no step used): the remaining steps are kept for the shopper's questions, which come next; Wren asks them now."))
                continue
            if not self.interviewed and self.tool_roles.get('ask') and (self.messages_left() > 0) and (tool_name in {self.tool_roles.get(r) for r in ('search', 'open', 'offer', 'compare')} - {None}) and self.time_to_ask():
                self.due_calls = self.__dict__.get('due_calls', 0) + 1
                if self.due_calls > 3:
                    acc.append((call_id, "Not run (no step used): the shopper's questions are due and come next; Wren asks them now, then this can run."))
                    continue
            stamp = tool_name + as_text(params)
            if not ending and tool_name != self.tool_roles.get('cart') and (stamp in self.recent_calls):
                acc.append((call_id, 'Not run (no step used): you just made this same call and nothing changed; its result is above.'))
                continue
            readers = {self.tool_roles.get(r) for r in ('search', 'open', 'offer', 'compare')} - {None}
            cached = self.read_memo.get(stamp)
            if tool_name in readers and cached and (cached[0] == self.need_version):
                acc.append((call_id, f'Not run (no step used): the same call at step {cached[1]} returned this, and nothing has changed since:\n' + cached[2][:3000]))
                continue
            if id_param and tool_name in readers and (tool_name != self.tool_roles.get('compare')) and self.seen_in_view(tool_name, str(params.get(id_param))):
                acc.append((call_id, f'Not run (no step used): {params[id_param]} has shown everything this view gives since the last change. Its record: ' + self.record_text(str(params[id_param]))))
                continue
            if multi_arg and isinstance(params.get(multi_arg), list) and params[multi_arg]:
                kept_lines = [i for i in params[multi_arg] if not self.seen_in_view(tool_name, str(i))]
                if not kept_lines:
                    acc.append((call_id, 'Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n' + '\n'.join((f'{i}: {self.record_text(str(i))}' for i in params[multi_arg]))[:4000]))
                    continue
                if len(kept_lines) < len(params[multi_arg]):
                    trimmed += f"\n[Wren] Left out {', '.join((str(i) for i in params[multi_arg] if i not in kept_lines))}: shown in full already, nothing changed since."
                    params = {**params, multi_arg: kept_lines}
                    stamp = tool_name + as_text(params)
            if ending and self.steps_left() > 2:
                if self.is_checkout(tool_name, params):
                    wider = self.hold_for_more()
                    if wider:
                        acc.append((call_id, wider + '\nCart now: ' + self.basket_text()))
                        continue
                    reread = self.refresh_offers()
                    if self.notice_gone:
                        self.gone_in_basket()
                    if self.basket_touched and self.basket_viewed_at < self.basket_changed_at and self.tool_roles.get('cart_view') and (self.steps_left() > 3):
                        fault_list = []
                        view_args = conform({}, self.tool_schemas.get(self.tool_roles['cart_view']), 'args', fault_list)
                        view = self.store_call(self.tool_roles['cart_view'], view_args) if not fault_list else None
                        if view is not None:
                            self.basket_view = view.get('observation')
                            self.basket_viewed_at = self.steps_used
                            change = self.digest(self.tool_roles['cart_view'], self.basket_view)
                            if change:
                                self.pending_notices = []
                                acc.append((call_id, 'Not run yet (no step used). Looking at the cart showed a change: ' + change + '\nCart now: ' + self.basket_text() + '\nApply the change, fix the cart if needed, then check out.'))
                                continue
                    swap = self.replacement_hint()
                    held = [i for i in self.basket_ids() if i in self.gone_at] if self.holds < 2 else []
                    if swap or held:
                        self.holds += 0 if swap else 1
                        acc.append((call_id, 'Not run yet (no step used). ' + (swap or f"{', '.join(held)} in the cart is sold out: replace that line first.") + '\nCart now: ' + self.basket_text()))
                        continue
                    chain, hold = self.follow_chain()
                    if hold:
                        acc.append((call_id, 'Not run yet (no step used). ' + chain + '\nCart now: ' + self.basket_text()))
                        continue
                    misfit = self.guarded(self.fitment_problems, []) if self.fitment_holds < 2 and self.steps_left() > self.closing_room() + 3 else []
                    if misfit:
                        self.fitment_holds += 1
                        self.journal.append({'kind': 'fit_hold', 'faults': misfit[:4]})
                        acc.append((call_id, 'Not run yet (no step used). A part in the cart does not work with the device in the cart: ' + '; '.join(misfit[:3]) + '. Swap the part (or the device), then check out.\nCart now: ' + self.basket_text()))
                        continue
                    fault_list = self.line_problems() if self.line_hold_count < 2 and self.steps_left() > self.closing_room() + 2 else []
                    fixed_args = []
                    if fault_list and all((' is held at ' in f for f in fault_list)):
                        fixed_args = self.set_quantities(self.quantity_gaps())
                        fault_list = self.line_problems() if fixed_args else fault_list
                        if fixed_args and (not fault_list):
                            self.line_hold_count += 1
                            acc.append((call_id, "Not run yet (no step used). Wren set the cart's quantities itself: " + '; '.join(fixed_args) + '. If the store refused any of these, fix those lines; then check out.\nCart now: ' + self.basket_text()))
                            continue
                    if fault_list:
                        self.line_hold_count += 1
                        self.journal.append({'kind': 'line_hold', 'faults': fault_list[:6]})
                        acc.append((call_id, 'Not run yet (no step used). The cart does not match the request yet: ' + '; '.join(fault_list[:6]) + ". Fix those lines (the cart tool's own way to add or take off units), then check out.\nCart now: " + self.basket_text()))
                        continue
                    read = self.guarded(self.pointer_pages, [], self.basket_ids()) + self.read_details(self.basket_ids(), self.closing_room() + 3) + self.cheaper_candidates()
                    cleaned = self.guarded(self.swap_unproven_line, '')
                    if cleaned:
                        acc.append((call_id, 'Not run yet (no step used). ' + cleaned))
                        continue
                    verdict = self.verify_basket()
                    if verdict:
                        met = "Wren read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n' if read else ''
                        acc.append((call_id, 'Not run yet. ' + chain + met + verdict + '\nCart now: ' + self.basket_text()))
                        continue
                    if reread and self.outlay > BUDGET_EASE:
                        acc.append((call_id, "Not run yet (no step used). Wren re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'))
                        continue
                elif self.steps_left() > 5:
                    live = {n for n in range(1, len(self.want_list) + 1) for i in self.basket_ids() if i in self.listings_for(n) and (not self.is_gone(i))}
                    fresh = self.reread_by_value([n for n in range(1, len(self.want_list) + 1) if n not in live], 'decline', floor=6)
                    if fresh:
                        self.late_hold_done = True
                        acc.append((call_id, 'Not run yet (no step used). Wren searched the requirement values once more and read these listings: ' + ', '.join(fresh) + '. Check them against every requirement; buy one that fits, and end without a purchase only if each of them fails.\nCart now: ' + self.basket_text()))
                        continue
                    hits_by_want = self.guarded(self.value_reads_before_decline, '')
                    if hits_by_want:
                        acc.append((call_id, 'Not run yet (no step used). ' + hits_by_want + '\nCart now: ' + self.basket_text()))
                        continue
                    review = self.review_decline()
                    if review:
                        acc.append((call_id, 'Not run yet (no step used). ' + review))
                        continue
                    if self.lead_ids and self.steps_left() > 4 and (self.decline_holds == 0):
                        self.decline_holds = 1
                        self.journal.append({'kind': 'refuse_blocked', 'leads': list(self.lead_ids)})
                        acc.append((call_id, 'Not run yet (no step used). The review found listings that appear to fit: ' + ', '.join(self.lead_ids) + '. Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again.'))
                        continue
                    if self.lead_ids and self.steps_left() > 4 and (self.decline_holds == 1):
                        self.decline_holds = 2
                        first_val = self.lead_ids[0]
                        self.verify_basket([first_val], BUDGET_STOP)
                        if self.basket_verdicts.get(self.check_id([first_val])) and first_val in self.lead_ids:
                            self.approved_pick = first_val
                            self.journal.append({'kind': 'refuse_blocked', 'confirmed': first_val})
                            acc.append((call_id, f"Not run yet (no step used). An independent check of {first_val}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."))
                            continue
            if self.is_decline(tool_name, params) and self.steps_left() <= 5 and (not self.late_hold_done):
                rows_out = [i for i in self.planned_lines if not self.is_gone(i)]
                fresh = self.basket_view is not None and self.basket_viewed_at >= self.basket_changed_at
                missing = [i for i in rows_out if i not in self.basket_ids() or (i in self.removed_lines and (not fresh))]
                if len(rows_out) >= len(self.want_list) and len(missing) + 1 <= self.steps_left():
                    self.late_hold_done = True
                    self.journal.append({'kind': 'late_decline_held', 'lines': rows_out, 'missing': missing})
                    redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else 'Check out the cart as it is.'
                    acc.append((call_id, 'Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. ' + redo + ' End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again.'))
                    continue
            said_why = ' '.join((str(v) for v in params.values() if isinstance(v, str)))
            free_text = any((isinstance(v, dict) and 'string' in schema_kinds(v) and ('enum' not in v) and (k not in (self.intake.get('refuse_args') or {})) for k, v in ((self.tool_schemas.get(tool_name) or {}).get('properties') or {}).items()))
            if self.is_decline(tool_name, params) and free_text and (not self.reason_hold_done) and (self.steps_left() > 6) and (not self.past_clock(25)) and (not re.search(REASON_RX, said_why, re.I)):
                self.reason_hold_done = True
                self.journal.append({'kind': 'proof_hold', 'said': said_why[:200]})
                acc.append((call_id, 'Not run yet (no step used). Ending without a purchase scores nothing unless the store provably has nothing that fits: a total over the budget, a stated limit that no listing meets, or no stock. If that is the reason, give it with the numbers in the ending text and send the ending call again. Otherwise put the closest fitting listing for each requested item in the cart and check out.'))
                continue
            if self.is_decline(tool_name, params) and len(self.want_list) > 1 and (not self.__dict__.get('fit_held')) and (self.steps_left() > len(self.want_list) + 3):
                choices = self.solved_picks()
                if choices:
                    self.fit_held = True
                    self.journal.append({'kind': 'fit_hold_decline', 'picks': choices})
                    acc.append((call_id, 'Not run yet (no step used). The listings seen hold a set whose parts name the device they must fit, and no record breaks a requirement: ' + '; '.join((f'item {n} {self.want_list[n - 1].name}: {i}' for n, i in sorted(choices.items()))) + '. Put them in the cart at the requested quantities, open any page they point to, and check out; end without a purchase only if a record clearly breaks a requirement in force (say which), then send the ending again.'))
                    continue
            if self.is_decline(tool_name, params) and (not self.salvage_held) and (not self.basket_ids()) and (self.steps_left() > len(self.want_list) + 3):
                choices = self.clean_choices()
                if choices:
                    self.salvage_held = True
                    self.journal.append({'kind': 'rescue_hold', 'picks': choices})
                    acc.append((call_id, 'Not run yet (no step used). These opened listings break no requirement in their records: ' + '; '.join((f'item {n}: {i}' for n, i in choices.items())) + '. An ending without a purchase earns nothing while such a listing exists. Put them in the cart at the requested quantities and check out; end without a purchase only if a record clearly breaks a requirement in force (say which), then send the ending again.'))
                    continue
            if self.is_decline(tool_name, params) and (not self.lead_ids) and (not self.search_pushed) and (self.steps_left() > self.step_limit // 3):
                self.search_pushed = True
                acc.append((call_id, f"Not run yet (no step used). {self.steps_left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if self.is_decline(tool_name, params) and len(self.want_list) > 1 and (self.steps_left() > len(self.want_list) + 3) and (not self.past_clock(25)):
                turned = self.guarded(self.turn_decline_into_order, '')
                if turned:
                    acc.append((call_id, turned + (' [episode finished]' if self.concluded else '')))
                    continue
            if self.is_decline(tool_name, params):
                done = self.finish_basket()
                if done:
                    acc.append((call_id, done + (' [episode finished]' if self.concluded else '')))
                    continue
                params = self.decline_text(tool_name, params)
            if ending and self.steps_left() >= 2:
                go, note_text = self.run_step('order' if self.is_checkout(tool_name, params) else 'end')
                if not go:
                    acc.append((call_id, 'Not run yet. ' + note_text + '\nCart now: ' + self.basket_text()))
                    continue
                if note_text:
                    trimmed = '\n[Wren] ' + note_text + trimmed
            reply_obj = self.store_call(tool_name, params)
            if reply_obj is None:
                acc.append((call_id, 'The call failed (network or server). You may try again.'))
                continue
            tool_result = reply_obj.get('observation')
            if multi_arg and isinstance(params.get(multi_arg), list) and (len(params[multi_arg]) > 1):
                displayed = as_text(tool_result)
                if not any((str(i) in displayed for i in params[multi_arg])):
                    self.multi_view_bad = min(self.multi_view_bad or 999, len(params[multi_arg]))
                    trimmed += f'\n[Wren] The side-by-side view showed nothing for {len(params[multi_arg])} listings: send at most {self.multi_view_size()} per call.'
                else:
                    self.multi_view_ok = max(self.multi_view_ok, len(params[multi_arg]))
            self.recent_calls = (self.recent_calls + [stamp])[-3:]
            self.id_arguments.update(((tool_name, k) for k, v in params.items() if (v if isinstance(v, list) else [v]) and all((isinstance(x, str) and x in self.listings for x in (v if isinstance(v, list) else [v])))))
            self.take_in(tool_result)
            if tool_name == self.tool_roles.get('compare'):
                row_list = self.listing_records(tool_result)
                if row_list:
                    self.detail_sources[tool_name] = self.detail_sources.get(tool_name, False) or any((facts_of(r) for _, r in row_list))
            if tool_name in readers - {self.tool_roles.get('search')}:
                self.remember_view(tool_name, tool_result)
            plain = {tool_id for job, tool_id in self.tool_roles.items() if job != 'search'}
            paged = self.follow_pages(tool_name, params, tool_result) if tool_name not in plain and (not ending) else ''
            if self.id_argument(tool_name) and tool_name not in (self.tool_roles.get('cart'), self.tool_roles.get('buy'), self.tool_roles.get('refuse'), self.tool_roles.get('ask')):
                listing = self.id_of(tool_result) if isinstance(tool_result, dict) else None
                if listing:
                    self.detail_sources[tool_name] = self.detail_sources.get(tool_name, False) or bool(facts_of(tool_result))
                    if tool_name == self.tool_roles.get('open') and (not self.detail_sources[tool_name]):
                        alt = next((t for t in self.tool_schemas if t != tool_name and self.id_argument(t) and (self.detail_sources.get(t) is not False) and (t not in (self.tool_roles.get('cart'), self.tool_roles.get('buy'), self.tool_roles.get('refuse'), self.tool_roles.get('ask'), self.tool_roles.get('search')))), None)
                        if alt:
                            self.tool_roles['open'] = alt
                    if not self.detail_sources[tool_name] and tool_name != self.tool_roles.get('open') and (not self.tool_roles.get('offer')):
                        self.tool_roles['offer'] = tool_name
                    if tool_name == self.offer_reader():
                        self.offer_read_at[listing] = self.steps_used
            if tool_name == self.tool_roles.get('cart'):
                self.record_basket(tool_name, params)
                self.basket_said = as_text(tool_result)
                self.recent_calls = []
                counts = [as_count(v) for v in params.values() if not isinstance(v, str) and as_count(v) is not None]
                for val in params.values():
                    if counts and isinstance(val, str) and (val in self.listings):
                        if counts[0] <= 0 and (not row_contains(tool_result, val)):
                            self.removed_lines.add(val)
                        elif counts[0] > 0:
                            self.removed_lines.discard(val)
                rows_out = [i for i in self.basket_ids() if not self.is_gone(i) and i not in self.removed_lines]
                if len(rows_out) >= len(self.planned_lines):
                    self.planned_lines = rows_out
            if tool_name == self.tool_roles.get('search') and self.listing_records(tool_result):
                self.searches_made = (self.searches_made + [dict(params)])[-8:]
            if tool_name == self.tool_roles.get('cart_view'):
                self.basket_view = tool_result
                self.basket_viewed_at = self.steps_used
            if self.is_checkout(tool_name, params):
                self.checkouts += 1
                if not reply_obj.get('finished'):
                    self.refused_step = self.steps_used
            words_out = self.render(tool_result) + paged + trimmed
            if tool_name not in plain and (not ending):
                words_out += self.cheapest_shown([tool_result] + self.paged_results) + self.price_cap_ignored(tool_name, params, tool_result)
            base = words_out
            proof = self.evidence_lines(tool_result)
            if proof:
                words_out += '\n' + proof
            if tool_name == self.tool_roles.get('cart') and any((as_count(v) == 0 for v in params.values() if not isinstance(v, str))):
                if any((isinstance(v, str) and v in self.listings and row_contains(tool_result, v) for v in params.values())):
                    words_out += '\n[Wren] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'
            self.notice_gone = False
            rows_out = self.stray_lines(tool_result)
            if rows_out:
                meaning = self.apply_notice(rows_out)
                if meaning:
                    words_out += '\n[note read] ' + meaning + '. Apply this change to your picks before checking out.'
            refused = self.is_checkout(tool_name, params) and (not reply_obj.get('finished'))
            again = refused and any((row_line in self.reply_text(tool_result) for row_line in self.gone_lines))
            loose = self.notice_gone and (refused or (not self.notice_named and (not self.is_checkout(tool_name, params))))
            if (loose or again) and self.basket_ids() and (not self.concluded):
                met = self.gone_in_basket()
                if met:
                    words_out += "\n[Wren re-read the cart's lines after the refusal]\n" + met
            self.notice_gone = False
            swap = self.replacement_hint()
            if swap:
                words_out += '\n' + swap
            if self.pending_notices:
                words_out += "\n[notes seen on Wren's own calls] " + ' | '.join(self.pending_notices)
                self.pending_notices = []
            if tool_name in readers:
                self.read_memo[stamp] = (self.need_version, self.steps_used, base)
            if reply_obj.get('finished'):
                words_out += ' [episode finished]'
            acc.append((call_id, words_out))
        return acc

    def close_out(self, again: bool=False) -> str:
        if self.concluded or self.halted or self.steps_left() <= 0:
            return ''
        checkout_tool, refuse = (self.tool_roles.get('buy'), self.tool_roles.get('refuse'))
        if not self.basket_ids() and (not self.basket_lines) and (self.steps_left() >= len(self.want_list) + 1):
            try:
                self.order_rescue()
            except Exception as err:
                self.journal.append({'kind': 'rescue_error', 'error': f'{type(err).__name__}: {err}'[:200]})
            if self.concluded or self.steps_left() <= 0:
                return ''
        fresh = self.basket_view is not None and self.basket_viewed_at >= self.basket_changed_at
        holding = bool(self.basket_ids()) if fresh else bool(self.basket_lines)
        if checkout_tool and (holding or (self.basket_touched and (not fresh) and (self.steps_left() >= 2))):
            fault_list: list[str] = []
            params = conform(dict(self.intake.get('buy_args') or {}), self.tool_schemas.get(checkout_tool), 'args', fault_list)
            if not fault_list:
                if self.fit_pairs and self.steps_left() >= 4:
                    self.mend_fitment()
                if self.steps_left() >= 2 and (not self.past_clock(-20)):
                    self.run_step('order')
                if self.concluded or self.steps_left() <= 0:
                    return ''
                reply_obj = self.store_call(checkout_tool, params)
                self.journal.append({'kind': 'forced_buy', 'reply': clipped_copy(reply_obj, 200)})
                if self.concluded or self.steps_left() <= 0:
                    return ''
                if reply_obj is not None and again:
                    self.checkouts += 1
                    self.refused_step = self.steps_used
                    self.digest(checkout_tool, reply_obj.get('observation'))
                    return 'the checkout was refused: ' + self.render(reply_obj.get('observation'))
        if refuse:
            fault_list = []
            params = conform(self.decline_text(refuse, dict(self.intake.get('refuse_args') or {})), self.tool_schemas.get(refuse), 'args', fault_list)
            if not fault_list:
                if self.steps_left() >= 2 and (not self.past_clock(-20)):
                    self.run_step('end')
                if self.concluded or self.steps_left() <= 0:
                    return ''
                reply_obj = self.store_call(refuse, params)
                self.journal.append({'kind': 'forced_refuse', 'reply': clipped_copy(reply_obj, 200)})
                if reply_obj is not None and (not self.concluded):
                    self.digest(refuse, reply_obj.get('observation'))
                    return 'ending without a purchase was refused: ' + self.render(reply_obj.get('observation'))
        return ''

    def conduct(self) -> list[dict[str, Any]]:
        if self.concluded:
            return self.journal
        self.take_brief()
        self.journal.append({'kind': 'brief', 'brief': self.intake})
        opening = self.policy.get('opening')
        if opening is not None:
            self.take_in(opening)
        intro = "SHOPPER'S REQUEST:\n" + self.request_text
        if opening is not None:
            intro += '\n\nWHAT THE STORE SHOWS FIRST:\n' + self.render(opening)
        intro += '\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n' + self.progress_note()
        self.dialogue = [{'role': 'system', 'content': CHARTER}, {'role': 'user', 'content': intro}]
        self.drive(self.turn_ceiling, BUDGET_STOP)
        if not self.concluded:
            self.salvage_basket()
            turned = self.close_out(again=True)
            if turned and (not self.concluded) and (not self.halted) and (not self.out_of_credit) and (self.steps_left() > 1):
                self.dialogue.append({'role': 'user', 'content': '[Wren] Wren tried to end the episode and ' + turned + '\nApply what the store says, then finish.\n' + self.progress_note()})
                self.drive(self.turns + 10, BUDGET_STOP + 0.01)
            if not self.concluded:
                self.close_out()
        self.journal.append({'kind': 'end', 'finished': self.concluded, 'steps': self.steps_used, 'cost': round(self.outlay, 5), 'calls': self.model_calls})
        return self.journal

    def drive(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.concluded and (not self.halted) and (self.steps_left() > 0) and (self.turns < loop_cap) and (self.outlay < spend_cap) and (not self.past_clock()):
            if not self.commit_called and self.checkouts == 0 and (self.steps_used >= self.commit_step() or self.guarded(self.commit_ready_now, False)) and (self.steps_left() > self.closing_room() + 2):
                self.commit_called = True
                self.more_hold_done = True
                held = self.basket_ids() or list(self.basket_lines)
                self.journal.append({'kind': 'commit_point', 'step': self.steps_used, 'left': self.steps_left(), 'cart': held})
                self.dialogue.append({'role': 'user', 'content': f'[Wren] Commit point: {self.steps_left()} steps are left and nothing has been checked out. A sold-out line or a refused checkout needs several steps to fix, so ' + ('check out the cart now unless a line clearly breaks a requirement.' if held else "cart each item's listing that you have read and that meets every requirement, and check out now; if your search showed that some item has no such listing, end without a purchase instead.") + '\n' + self.progress_note()})
            if self.time_to_ask():
                try:
                    report = self.hold_interview()
                except Exception as err:
                    self.journal.append({'kind': 'interview_error', 'error': f'{type(err).__name__}: {err}'[:300]})
                    report = '[the interviewer stopped early; ask the shopper yourself if messages are left, one property per message]' if self.messages_left() > 0 else ''
                if report:
                    self.dialogue.append({'role': 'user', 'content': report + '\n' + self.progress_note()})
                continue
            self.shrink_chat()
            self.turns += 1
            message = self.consult(self.dialogue, [SEAT_QUICK, SEAT_ALT], 4000, self.tool_menu())
            if message is None:
                self.empty_turns += 1
                if self.out_of_credit or self.model_calls >= self.call_ceiling or self.empty_turns > 2:
                    break
                continue
            call_list = []
            for slot_no, one_call in enumerate(message.get('tool_calls') or []):
                if not isinstance(one_call, dict):
                    continue
                fn = one_call.get('function') if isinstance(one_call.get('function'), dict) else {}
                raw = fn.get('arguments')
                parsed = raw if isinstance(raw, dict) else json_object(raw) if isinstance(raw, str) and raw.strip() else {}
                call_list.append({'id': str(one_call.get('id') or f'k{self.turns}_{slot_no}'), 'type': 'function', 'function': {'name': str(fn.get('name') or ''), 'arguments': as_text(parsed if isinstance(parsed, dict) else {})}})
            body_text = message.get('content')
            if not isinstance(body_text, str):
                body_text = as_text(body_text) if body_text else ''
            if not call_list:
                idle += 1
                self.dialogue.append({'role': 'assistant', 'content': body_text or '(no call)'})
                self.dialogue.append({'role': 'user', 'content': 'Call one of the tools to continue.\n' + self.progress_note()})
                if idle >= 3:
                    break
                continue
            idle = 0
            committing = {self.tool_roles.get('cart'), self.tool_roles.get('buy'), self.tool_roles.get('refuse')} - {None}
            if not self.interviewed and self.tool_roles.get('ask') and (self.messages_left() > 0) and any((c['function']['name'] in committing for c in call_list)) and (self.steps_left() > self.messages_left() + 3):
                self.dialogue.append({'role': 'assistant', 'content': body_text or None, 'tool_calls': call_list})
                for one_call in call_list:
                    self.dialogue.append({'role': 'tool', 'tool_call_id': one_call['id'], 'content': 'Not run (no step used): the shopper is asked first.'})
                report = self.hold_interview()
                self.dialogue.append({'role': 'user', 'content': (report or 'The interviewer had nothing to ask.') + '\n' + self.progress_note()})
                continue
            self.dialogue.append({'role': 'assistant', 'content': body_text or None, 'tool_calls': call_list})
            self.journal.append({'kind': 'turn', 'text': body_text[:300], 'calls': [c['function'] for c in call_list]})
            replies = self.execute_calls(call_list)
            for pos, (call_id, words_out) in enumerate(replies):
                if pos == len(replies) - 1:
                    words_out += '\n' + self.progress_note()
                self.dialogue.append({'role': 'tool', 'tool_call_id': call_id, 'content': words_out})

class Steward(Helm, Basket, Fitment, Interviewer, Catalog, Intake, Wire):

    def __init__(self, problem: dict[str, Any]):
        environment = problem['environment']
        self.session = dict(environment['binding'])
        self.policy = environment['policy_view']
        self.request_text = str(self.policy.get('query') or '')
        self.tool_specs = [t for t in self.policy.get('tools') or [] if isinstance(t, dict) and isinstance(t.get('function'), dict) and isinstance(t['function'].get('name'), str)]
        self.tool_schemas = {t['function']['name']: t['function'].get('parameters') if isinstance(t['function'].get('parameters'), dict) else {'type': 'object', 'properties': {}} for t in self.tool_specs}
        steps = as_count(self.policy.get('max_steps'))
        if steps is None and isinstance(self.policy.get('max_steps'), float):
            steps = int(self.policy['max_steps'])
        self.step_limit = steps if steps and steps > 0 else 30
        self.started = time.monotonic()
        self.scarce_messages = False
        self.focus_want = 0
        self.reason_hold_done = False
        self.salvaged = False
        self.nearest: str | None = None
        self.backups: dict[str, Any] = {}
        self.recovery_done: set[Any] = set()
        self.commit_called = False
        self.counter_lock = threading.Lock()
        self.steps_used = 0
        self.concluded = bool(self.policy.get('finished'))
        self.halted = False
        self.out_of_credit = False
        self.backoff_spent = 0
        self.empty_turns = 0
        self.short_think = False
        self.effort_allowed = True
        self.cut_replies = 0
        self.outlay = 0.0
        self.model_calls = 0
        self.turns = 0
        self.journal: list[dict[str, Any]] = []
        self.intake: dict[str, Any] = {}
        self.tool_roles: dict[str, str] = {}
        self.want_list: list[Want] = []
        self.spend_limit: Any = None
        self.message_cap = 0
        self.messages_sent = 0
        self.required_steps: list[dict[str, Any]] = []
        self.reserved_messages = 0
        self.step_messages = 0
        self.steps_done: set[tuple[str, str]] = set()
        self.action_log: list[str] = []
        self.action_steps: list[int] = []
        self.interviewed = False
        self.exchanges: list[tuple[str, str]] = []
        self.listings: dict[str, dict[str, Any]] = {}
        self.read_records: dict[str, dict[str, Any]] = {}
        self.listing_order: list[str] = []
        self.answers_seen: set[str] = set()
        self.notices: list[str] = []
        self.need_version = 0
        self.basket_touched = False
        self.basket_changed_at = -1
        self.basket_viewed_at = -1
        self.basket_view: Any = None
        self.notice_named = False
        self.basket_lines: dict[str, int] = {}
        self.checked_ids: set[str] = set()
        self.reviewed = False
        self.offer_read_at: dict[str, int] = {}
        self.gone_at: dict[str, int] = {}
        self.unavailable_ids: set[str] = set()
        self.gone_lines: set[str] = set()
        self.notice_gone = False
        self.holds = 0
        self.id_arguments: set[tuple[str, str]] = set()
        self.basket_said = ''
        self.refused_step = -1
        self.lead_ids: list[str] = []
        self.decline_holds = 0
        self.search_pushed = False
        self.planned_lines: list[str] = []
        self.removed_lines: set[str] = set()
        self.late_hold_done = False
        self.leads_by_want: dict[str, list[str]] = {}
        self.basket_verdicts: dict[str, bool] = {}
        self.approved_pick: str | None = None
        self.basket_finished = False
        self.id_field: str | None = None
        self.detail_sources: dict[str, bool] = {}
        self.pending_notices: list[str] = []
        self.checkouts = 0
        self.declines_sent = 0
        self.recent_calls: list[str] = []
        self.dialogue: list[dict[str, Any]] = []
        self.pages_opened: set[str] = set()
        self.chain_needs: list[tuple[int, str]] = []
        self.chain_values: dict[str, str] = {}
        self.chain_checks = 0
        self.chain_holds = 0
        self.multi_view_ok = 0
        self.multi_view_bad = 0
        self.tool_fields: dict[str, set[str]] = {}
        self.view_fields: dict[str, tuple[int, set[str]]] = {}
        self.read_memo: dict[str, tuple[int, int, str]] = {}
        self.page_reads = 0
        self.layouts: dict[str, dict[str, Any] | None] = {}
        self.price_field: str | None = None
        self.stock_field: str | None = None
        self.cheap_marks: set[str] = set()
        self.more_hold_done = False
        self.paged_results: list[Any] = []
        self.want_order: list[int] = []
        self.line_hold_count = 0
        self.fit_pairs: list[tuple[int, int]] = []
        self.fitment_holds = 0
        self.searches_made: list[dict[str, Any]] = []
        self.query_hits: list[tuple[dict[str, Any], list[str]]] = []
        self.pending_search: dict[str, Any] | None = None
        self.reference_searches = 0
        self.references_seen: dict[str, list[str]] = {}
        self.turn_ceiling = max(TURN_CAP, min(110, self.step_limit))
        self.call_ceiling = max(MODEL_CALL_CAP, self.turn_ceiling + 25)
        self.salvage_held = False
        self.order_salvaged = False

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Steward(problem_data)
    except Exception as exc:
        return [{'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]}]
    try:
        return episode.conduct()
    except Exception as exc:
        episode.journal.append({'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]})
        try:
            if not episode.concluded:
                episode.close_out()
        except Exception as again:
            episode.journal.append({'kind': 'crash', 'error': f'{type(again).__name__}: {again}'[:400]})
        return episode.journal
__all__ = ['agent_main']
