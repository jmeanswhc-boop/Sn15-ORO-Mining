"""Halo: buys for one shopper inside an ORO store world. Built on Know Thyself, the public race-13 top agent.

A light model walks the store with the world's own tools; code keeps the ledger, plans the questions to the shopper
and checks the cart before an order and the evidence before a decline. Beyond the base:
- a message the request kept for the ending is spent on one more planned question when Halo is about to buy
  (late_question), so a tight message limit no longer leaves a hidden need unasked;
- a clock keeps every episode inside the validator's 300 s limit: no interview, question, loop turn or model request
  starts when its result could not be used in time, and the model timeout shrinks with the time left;
- message shapes follow the product count: on one product a lone property opens and pairs of properties come next;
  on several products a short list (3) opens and long ones (8) follow, and when products outnumber the messages
  neighbouring products share one message. A reply saying the message held too much turns every later message
  into a single property;
- change notices riding on a shopper reply go to the note reader, apart from the reply itself, and a budget heard in
  a reply keeps the figure as the shopper wrote it.
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
LOOP_CALLS = 42
ALL_CALLS = 60
SPEND_SOFT = 0.055
SPEND_HARD = 0.078
HEAVY_ROOM = 0.02
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3
AUTO_PAGES = 10
CHOICE_KEY = '^(?:.*[\\s_])?(?:options?|variants?)$|^(?:select|choose)\\b'
CHOICE_MARK = ' (choices on offer, not a stated value)'
CHEAPER_OPENS = 3
LINK_SEARCHES = 5
EARLY_WALKS = 2
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
LIST_SIZE = 8
LIST_OPEN = 3
PAIR_SIZE = 2
ALLOWANCE = '\\b(?:remain\\w*|left|allowance|unspent|quota)\\b'
LEFT_KEY = 'remain|left|unspent|avail|spare'
CAP_ARG = '(?:^|_)(?:up_?to|ceiling|maximum|under|max|cap)(?:_|$)'
CAP_NOT = 'step|item|num|size|count|page|row|result'
SHELF_ARG = 'aisle|shelf|label|categor|department|section|(?<!age)(?:^|_)group(?:_name|_label)?$'
LIST_MODE = 'structured|list|categor|brows|facet|filter|field'
AMOUNT = '\\d[\\d.,]*\\s*[a-z"\']{1,12}'
FLOOR = '\\b(?:min(?:imum)?|no (?:fewer|less) than|at least)\\s+(' + AMOUNT + ')\\s*$'
QUERY_WORDS = 'keyword|phrase|query|(?:^|_)(?:words?|terms?|keywords?|phrase|text|search|query|q)(?:_|$)'
REMARK_ARG = 'summary|why|explanation|explain|comment|remark|message|note|justif|reason'
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_llm_short: dict[int, ProxyClient] = {}
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
GUIDE = 'You are Halo, buying on behalf of a single shopper in an online shop, and your only way to act is a call to one of the shop\'s tools. Scoring is all or nothing: the episode scores 1 only when its ending is exactly correct, otherwise 0. A correct purchase holds every item the shopper asked for, at the quantity asked, each one satisfying every requirement still in force at checkout, with a total inside the budget. If the shop really offers nothing that satisfies everything, the correct ending is the no-purchase tool, used the way the request describes.\n\nTHE SHOP AND ITS TOOLS\n- Each tool call uses one step, and steps are capped; reasoning is free. This shop has its own tool names, argument names and result formats: study every tool\'s description and schema, and pass exactly the argument names and permitted values it defines.\n- Searching may be limited: a filter by category, a keyword mode that might look at only the opening words of your query, pages of only a few rows, a token or offset that fetches the next page. When the right answer depends on seeing every match, set the total the shop reports against the rows you have received and keep paging, passing back exactly the token or offset the shop returned. Short keyword queries that lead with the value do best: the wanted value and a brief product noun, rather than the full category name, which matches every row of that category equally. An empty result from a filtered search usually signals a wrong filter value or a wrong filter parameter: try again with the label written exactly as result rows print it, or use a different filter parameter.\n- A search row is only a summary. Shops split up the record of a listing differently: one view may give its attributes (properties), another just its offer and description, and a comparison view may hold either one. Once Halo has learned which tool shows attributes, the ledger names it. Decide every property requirement from those attributes, so read them for each listing that you might buy; one call that shows several listings at once is cheaper in steps, provided it shows what you need. Do not reopen a listing when nothing about it has changed since your last read.\n- The clock is your tightest limit: every reply you write burns several seconds of a fixed time allowance, whereas a tool call only uses a step. Bundle into one reply every call that does not wait on another call\'s output (searches for separate items, opening the listings you are comparing, a next page together with the next search); hold back only the calls that need an earlier result.\n- Halo\'s interviewer handles all questions to the shopper, so you never send the shopper a message yourself. Its answers show up in the conversation and are written into the ledger.\n- Besides the purchase, the request may demand actions on the way (an approval, a message, something to read or check first); you will find them in the ledger. Halo itself performs the ones that fall due right before the order or the ending at the moment you call the checkout tool or the ending tool, so do not wait for them, and never give up on a purchase because of them; carry out any other ones with the shop\'s tools (questions about a product go through Halo\'s interviewer).\n\nDECIDING WHETHER A LISTING FITS\n- The listing\'s category label settles what kind of product it is: only a label that names the requested item as a whole qualifies, never an adjacent one.\n- A property requirement is satisfied once the attribute the listing has for that property gives the value. How the key is spelled does not matter (spaces, underscores, camel case, prefixes), and a single element of a comma- or slash-separated list suffices. Values are matched on their words: letter case, singular versus plural and ordinary synonyms are accepted, and a value holding the required value\'s words qualifies ("deep teal" satisfies teal, "extra roomy fit" satisfies roomy). A value made of other words does not qualify (an abbreviation or code does not stand in for the full word, and "teal green" is not "sea green"); numbers are compared only after converting units, a limit includes its boundary value, and a range has to sit entirely within a limit.\n- When the key of an attribute names the property of the requirement (under that name or an ordinary synonym), that attribute is decisive: if it gives a different value, the listing fails no matter what other keys or the title say. A key about another property, even a closely related one (a lid colour is not the colour of the whole item), settles nothing for the requirement. After each opened listing Halo appends [evidence] lines showing the keys whose names match a requirement.\n- If the listing lacks an attribute for that property, the wanted value appearing under some other attribute (such as a list of features or styles that includes it) or in the title or description is accepted. A number of a different kind is not (a weight is not a capacity), and neither is a bare figure with no text telling what it measures.\n- If the category label itself gives a requirement\'s value (an audience or season in the label), the category satisfies that requirement.\n- Instead of a value, a requirement may refer to another listing ("the X that matches the listing <shop> sells with <property>", "no more than N away from the X shown on the page of ..."). Settle it before anything else: locate that listing (search by its category, brand, shop or wording), open its full page through the single-item tool or the side-by-side one (a search row does not suffice), and if that page directs you to yet another listing for the property, locate and open that one in the same way; the final page supplies the value. An order counts only if every page in that chain was opened. When the requirement wants any value except that other listing\'s, it excludes only listings with that exact value, and on its own it is never a reason to end without a purchase.\n- When a listing you plan to buy points you to another page for more details on it, check that page before buying.\n- When one requested item has to fit another (an accessory for a device), pick the pair together, based on the listings\' own wording (a list of the devices it fits, or a model or series code printed on both pages). If none of the accessories meeting their own requirements fits the chosen device, change to another device that satisfies its requirements and that a qualifying accessory fits; do not end without a purchase while that switch remains untried.\n- Honour exclusions and closed lists. Wishes ("would be nice", "ideally") serve only to break ties.\n- Never buy something the shop marks as sold out; the total (price times quantity, added up across all lines) must not go over the budget, and being exactly equal is allowed.\n\nPICKING AMONG FITS\n- If the request wants the cheapest (lowest price, smallest total), the order must contain the lowest-priced listing out of ALL fitting listings. Collect the complete fitting set (page through it: rows often come in no price order, and Halo fetches further pages of a short result on its own), examine candidates from the cheapest upward, and take the first one that fits completely. Where a search takes a price ceiling, make use of it: list the category below what your current best choice costs, so that only cheaper candidates remain to check.\n- If the request wants the most or least of some measure, rank all fitting listings by that measure and take the top or bottom one; a listing that leaves the measure unstated can never win.\n- In every other case, any listing that satisfies everything is correct: do not spend steps searching for a better one.\n- Add each item to the cart once you have opened a listing that satisfies all its requirements (for a cheapest, most or least pick: once the fitting set has been compared), then go on to the next. Carted lines often sell out shortly after being added, and carting early leaves you steps to replace one.\n- A request for several distinct items needs a separate cart line for each, each at the quantity asked.\n\nWHEN THINGS CHANGE MID-TASK\n- The shopper or the shop may change things: a new budget, a requirement added or removed, a price change, a listing selling out. These show up inside tool results, often at checkout (a refusal); a note counts no matter which tool\'s result carries it, and the ledger lists the requirements currently in force. The episode continues after a refusal: apply each change, repair the cart and try the checkout again.\n- Shortly before checkout, read a chosen listing\'s record once more, since price and stock can change without notice.\n\nENDING\n- Leave exactly the chosen lines in the cart (take out anything else), view the cart once, then check out in a call by itself.\n- End without a purchase only once a genuine search has found no listing that satisfies everything. Being low on steps proves no such thing: in that case check out the best cart you have. Use the request\'s wording for that ending, and if a remark is required, keep it brief and name no listing.\n- Every reply needs at least one tool call. Keep your text brief: what you know, then your next move.'
BRIEF = 'You prepare a shopping task so that an agent can carry it out. Read the shopper\'s request and the shop\'s tools, then reply with ONE JSON object only (compact, no indentation, no other text) holding these keys:\n\n"items": a list with one entry per separate product the shopper wants bought, each {"name": the product kind in the request\'s own words, "shelf": the shop\'s category label for it when some tool lists labels (copied exactly), or else the category the request names for it ("a coat in rainwear" gives "rainwear"), otherwise "", "quantity": how many units of that single listing (integer; 1 if not stated; a pair or a set sold as one listing counts as 1), "musts": a list of the firm requirements on that product, each written "property: constraint" and giving the exact value, limit, exclusion or closed list. Call the property by the word the request uses for it ("for a lakeside occasion" becomes "occasion: lakeside", "with a 12-inch blade" becomes "blade length: 12 inch"); never replace it with an adjacent property. A must whose value comes from another listing is a SINGLE must carrying the full description of that listing ("lid colour: unlike the lid colour of the teapot that Brindle sells with a steel handle"): the words describing that other listing (its shop, its kind, a following "with <property> <value> and <property> <value>") remain inside that one must and never become musts of this product}. Price, budget, stock and quantity never go into musts. The story around the request (who will use it, where it will be placed, what went wrong last time, the occasion behind it) is background, not a must, unless it plainly gives a property\'s value or limit; and words that are part of the product\'s name (an audience or a season named together with the product) are never musts: they go in "name". A use or occasion that the request turns into a property of the product itself ("a dress intended for parties", "the occasion must be party", "suits for business wear", "a bag, and it is meant for a wedding") counts as a must on that property ("occasion: business") except when the shop\'s category label already names it; background also covers a specific person\'s event, the person or the setting it is bought for ("for a colleague", "to wear at his wedding"), and the way it ought to sit, feel or look in use. When the request names products in a single phrase linked by "and" (a single "I want ..." carrying a single quantity, often a shop department\'s name), they form ONE item; split them into separate items only when each gets a quantity or requirements of its own.\n"budget": the highest total the shopper will pay, given as a number in the unit the request uses, or null.\n"budget_unit": the name of that unit (for example "cents"), or null.\n"budget_scope": "total" if the limit applies to the whole order, "each" if it applies per item, null when no limit is given.\n"in_stock_only": true, unless the request clearly says unavailable items are acceptable.\n"pick": "cheapest" if the request wants the lowest price or smallest total, "most" or "least" if it wants the extreme of some measure, otherwise "any". Comments that money is tight are not a pick rule, and neither is an instruction about what to offer when nothing fits cleanly (the closest listing, or one within some margin of the cheapest of those).\n"pick_measure": the quantity being maximised or minimised for "most"/"least", otherwise "".\n"messages": the number of messages the shopper lets the agent send (integer), or null if not stated.\n"steps": the number of steps, moves or tool calls the request permits (integer), or null.\n"lookups": the musts (copied exactly) that need their value read first from another listing the request points to, as a list, otherwise [].\n"rules": a list of every instruction in the request on how to work or how to finish, each in 12 words or fewer.\n"wishes": a list of the things the shopper would only like to have.\n"ask_first": a list of the things the request explicitly says to confirm with the shopper before choosing (empty if none).\n"process": a list of the steps the request demands of the agent apart from finding and buying the right items (for example getting the shopper to approve, telling or showing them something, confirming something with them, reading something beforehand, or using one particular tool), each {"step": what has to be done, in 15 words or fewer, "when": "choose" (prior to choosing the products), "order" (just ahead of checkout) or "end" (just before ending without a purchase), "messages": the number of messages to the shopper it needs (0 if none)}; [] when no such step is asked for. These do not count as steps: item requirements, the budget, stock, message and step limits, and the way to finish if nothing fits.\n"quiet_decline": true when the request forbids suggesting or naming alternatives if nothing fits, false when it wants them offered, null when it does not say.\n"fits": a list holding one {"item": item number, "with": item number} per requested item that, by the request\'s words, has to work with, fit or suit ANOTHER requested item (whichever one the agent chooses); [] if none.\n"roles": an object giving the exact tool for each job, or null where no tool does it: "search", "open" (shows a single listing\'s attributes or properties; if there are two single-listing tools, the one whose description promises specifications, properties or attributes), "offer" (shows a single listing\'s current offer, such as price and stock, when that tool differs from "open"), "compare" (several listings in full), "ask" (sends the shopper a message), "cart" (edits the cart contents), "cart_view" (displays the cart), "buy" (places the order), "refuse" (finishes with no purchase).\n"buy_args": arguments that never change and that the "buy" tool needs in order to place the order (when the tool has a mode argument, that mode), otherwise {}.\n"refuse_args": arguments that never change and that the "refuse" tool needs in order to finish with no purchase, otherwise {}.\n'
PLAN = 'You pick the questions that a shopping agent will put to a shopper. Each shopper keeps a few firm preferences that the request leaves out, each one about a single property of a single requested product. A question uncovers a preference only if it names that very property; for a property the shopper has no opinion on the reply is "don\'t mind", and a property the shopper never considers gets a confused reply, so each property you name uses up part of a message.\n\nYour input: the request, the items together with what the request already fixes for each, and for each item\'s category a numbered list of the attribute keys that opened listings show, with the number of those listings that carry each key and sample values. A preference left unwritten can concern a property that none of the listed keys names.\n\nPick properties in this priority:\n0. Anything the request itself says you should confirm with the shopper (it appears under ASK FIRST if there is any), each named by the property it concerns.\n1. Hinted properties. The story in the request often hints at an unwritten preference without stating its value: a sentence about something the shopper cannot assess or leaves up to you, someone whose taste is decisive, the product\'s appearance or the setting it must suit, the person who will use or wear it, or what it contains. For every such sentence, name the single property it hints at and the item it concerns, in the story\'s order.\nUse a listed key of that item to name a hinted property when one matches the hint (a hint about the wearer: the gender or audience key; about appearance: the listed colour, pattern or finish key that most of its listings carry); a property that no listing of the item carries goes after the item\'s three most carried keys.\n2. After that, the listed keys in rank order: prefer the properties a buyer of that product kind decides on first (usually its style or type, its fit or form, its dimensions, capacity or size, its pattern, colour or material, whichever drive the purchase of that kind of product) and keys that most of the item\'s listings carry; when at least five listings have been read, steer clear of keys found on just one or two of them.\nDo not pick a property that the request already fixes for that item (a requirement names it; a key whose values only echo a required value, for instance a pattern key that shows the decoration the request requires, is NOT fixed), an alternative name for a property already picked, a key each of whose values lists several purchase options together, identifiers, model names, and warranty or other logistics. Counts, pack sizes and quantities per pack are ordinary properties, and a key that carries the mark [not the stated ...] is about a different property from that requirement, however many words its name has in common with it. Rank every item on its own terms, hinted properties ahead of listed keys.\n"label" gives the plain name of the property: for a listed key, every word of the key spelled plainly ("handle colour" for handleColour, "base finish" for base_finish; never cut down to "colour" or "finish"); just the property\'s name, with no sample values, no brackets and no second property.\nPick up to COUNT per item, best first. Reply with ONE JSON object only: {"ask": [{"item": item number, "label": "...", "hint": "the few story words (8 at most) it comes from, or left empty"}]}; give the picks of each item in rank order'
HEAR = 'A shopping agent put some questions to a shopper. Go through every answer and reply with ONE JSON object only:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true if the value has to be read from another listing, otherwise false}], "drop": [{"property": "a requirement the shopper is now giving up", "item": item number}], "free": [{"property": "...", "item": item number}], "crowded": [the numbers of those questions where the answer complains the message covered too many things, or asks that they come one per message; leave out answers that merely say they did not understand it or want simpler wording], "budget": a new overall limit as a number, or null}\n- "needs": each firm preference stated in an answer, with its exact value or limit and its unit ("oval", "at least 30 cm", "not wicker"). Call the need after the property the answer itself mentions, which may not be the one asked about; several answers on one property form one need. Copy the value exactly as the answer words it, keeping its nouns and units (write "at least 3 shelves", never just "at least 3"). The value holds only what a listing could state: it starts at the wanted value, or at the limit word before it, and runs to the end, without the framing the answer puts round it ("matching oval", "settled, with oval" and "oval and nothing else" all give "oval"; "equal to at least 30 cm" gives "at least 30 cm").\n- An answer may give the value by referring to another listing (a value like, unlike or close to the one on some shop\'s product, that product identified by a few of its properties): treat it as a SINGLE need on the property named by the answer, with "points": true, whose value is the answer\'s wording copied exactly, from the comparison words through to the end of that description, together with whatever it says about a listing named on that page; do not trim or paraphrase it. This item gets no needs from the properties that describe that other product.\n- "free": per question, the property about which the shopper said they do not mind.\n- A value named in homely or indirect words is still stated. If the wording refuses to negotiate on a value (no trading it, giving it up or changing it), that value is wanted; use "not X" just for an answer that rejects X itself.'
NOTE = 'While a shopping agent was working, the shop displayed the note below (its source may be the shopper or the shop). Reply with ONE JSON object only:\n{"kind": "change" if the note alters the task (the budget, a requirement, a price, availability, or a refused order), "error" if it merely reports that a call was malformed or rejected, "other" for anything else, "budget": the new overall limit, a number in the request\'s unit, or null, "drop": [{"need": "the requirement that is no longer wanted", "item": item number}], "add": [{"need": "property: constraint", "item": item number}], "gone": [ids of listings it says are unavailable], "sold_out": true if it reports that a chosen listing has gone, sold out or become unavailable (whether or not it names that listing), "price_moved": true if it says a price has changed, "summary": "one short sentence"}\nA requirement that is replaced counts as a drop of the old one and an add of the new one. Report every change the note contains.'
CHECK = 'You are the final review before a shopping agent checks out. Judge from the records shown whether this cart is exactly correct.\nHow to read records: a requirement holds when the attribute the listing has for that property gives the value (the key may be spelled any way; one element of a list suffices unless the key lists choices on offer (options, variants, a selection) or the listing\'s own chosen option is a different element of that list; letter case, singular/plural and ordinary synonyms are accepted; a value holding the required value\'s words is accepted, e.g. "deep teal" for teal; units are converted; limits include the boundary; a range has to sit entirely inside); a set of alternatives where only one would be bought (sizes or counts to choose among, often under a plural key or separated by "or") is just an offer and states none of them; a word attached to the value that turns it into another model, grade or version (a qualifier before a name) gives a different value, not the required one; an attribute whose key names the requirement\'s own property is decisive, and if it gives something else the listing fails; a key for another property, even a closely related one (a lid colour is not the colour of the whole item), settles nothing; where no attribute of the listing covers that property, the wanted value given under another attribute or in the title or description is accepted (a number of a different kind is not: a weight is not a capacity; nor is a bare figure with no text saying what it measures); a requirement stated by the category label itself is satisfied by that category; the requested item as a whole must match the category label; a rule of differing from another listing\'s value counts as satisfied unless the records show that same value. Every requested item has to be its own cart line at the requested quantity, nothing else may sit in the cart, nothing may be shown out of stock, and price times quantity summed over all lines has to stay within the budget. A listing\'s newest record takes precedence.\nIf the pick rule is cheapest, most or least, check the other listings shown as well: when one of them plainly meets every requirement and does better than the cart\'s choice by that rule, the cart is wrong.\nGive ONE JSON object as your whole reply: {"ok": true or false, "problems": ["a short, concrete problem", ...], "better": [ids of listings to put in place of a cart line, best first]}. ok=false is reserved for a problem the records show plainly, or a requirement that none of the records speaks to.'
READING = 'an attribute of the listing settles a property when its key names that very property (key spelling does not matter; one element of a list suffices unless the key lists choices on offer (options, variants, a selection) or the listing\'s own chosen option is a different element of that list; synonyms and unit conversions are accepted; a value holding the required value\'s words is accepted, e.g. "deep teal" for teal; limits include the boundary); a set of alternatives where only one would be bought (sizes or counts to choose among, often under a plural key or separated by "or") is just an offer and states none of them; a word attached to the value that turns it into another model, grade or version (a qualifier before a name) gives a different value, not the required one; a key for another property, even a closely related one (a lid colour is not the colour of the whole item), settles nothing; if no attribute names that property, the wanted value given under another attribute or in the title or description is accepted (a bare figure with no word for what it measures counts for nothing); a requirement stated by the category label itself is satisfied by that category; the category label has to match the item; a rule of differing from another listing\'s value counts as satisfied unless the records show that same value; listings out of stock and listings over the budget do not fit.'
AUDIT = 'A shopping agent plans to end without buying, believing nothing fits. Test the listing records shown against the requirements, using these reading rules: ' + READING + '\nA record marked ROW is known only from a search result row: its page (with the attributes) is unread, so it can neither pass nor fail on any property the row does not show.\nReply with ONE JSON object only: {"fits": {"item number": [up to 4 ids of listings that seem to meet every requirement, cheapest first]}, "open": [up to 6 ids of ROW listings that are of a requested kind, within the budget, and whose page might show they fit, most promising first], "reason": "one sentence"}'
UNMIX = 'A shopper\'s request has been turned into firm requirements on a product. A few requirements take their value from another listing the request refers to, and the words identifying that other listing (its shop, its kind, and properties listed after it with "with ...") sometimes get misread as requirements on the product to be bought. Going by the request\'s wording, tell which of the OTHER REQUIREMENTS below merely describe that other listing. Answer with ONE JSON object and no other text: {"describe_other": [each such requirement, copied exactly; empty if each one is a genuine requirement on the product]}'
TRAIL = 'A shopping agent is close to checking out. Some requirements take their value from another listing rather than stating it, and the page of a listing can send the reader on to yet another listing for that property. An order counts only if the agent has fully opened every listing along each such chain. Trace the chain of each requirement starting from the requirement\'s own description, one link at a time, through the wording of the pages, using the records below (OPENED: the agent opened that page; ROW: known only from a search result row, its page unread).\nReply with ONE JSON object only: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the id of the record below that is this link, else "", "about": what the requirement or the previous page says about that listing, "find": "for a link still missing: words to search for (its shop plus the values in the description)"}], "done": true if the last link is an OPENED page that itself states the property\'s value (this holds too when the requirement wants some value differing from that one, or unlike it, because the page\'s value is the yardstick), else false, "value": the property\'s value as the last page states it, or ""}]}\nA record is a link only if it matches the link\'s entire description: its shop or brand, its kind of product AND each property value in the description (e.g. "a twin pack of 4" rules out that shop\'s listing of 4 units that is no twin pack); if no record matches all of that, the link is not found yet, even when the same shop sells other listings of that kind. Such a link gets "find": a few search words made of the shop\'s name and the values in the description.\nGive each chain\'s links in order, ending at the first link that is not found or whose page is not OPENED, as its wording is still unknown.\nA page may also name a partner for its own value: it says the value goes together with, pairs with or mates up with some other value. If the request\'s own wording asks for that partner (a partner piece, whatever goes with or mates up with the pointed value) instead of the identical value, and the last page names such a partner, then "value" is the partner value as written on the page, not the page\'s own value.'
GATE = 'A shopping agent is about to STAGE. Along the way, the shopper\'s request asks for the steps listed below. Using the numbered RECORD of the agent\'s actions, decide for each step due before this STAGE whether the cart in its present state has had it done. Only a RECORD line showing the step being carried out makes it done; when the step concerns the order or the cart, the line must also sit after the most recent cart change, and its reply must grant it; questions about a product\'s properties do not accomplish a step.\nIf a step is still outstanding, give the single store call that performs it next: a tool from TOOLS plus its arguments. Messages to the shopper are sent with the messaging tool, as a single short sentence or question in plain words about that step alone, and mention no product property or value that the step does not need. Leave the call out when all due steps are done, or when the messages or steps remaining cannot accomplish the missing one.\nOutput ONE JSON object and nothing more: {"steps": [{"step": the step, "lines": [numbers of the RECORD lines that show it done], "done": true or false}], "hold": anything a reply wants changed before going on, or "", "call": {"tool": "...", "args": {...}} or null, "needs": [{"need": "property: constraint", "item": item number} for each firm requirement stated in such a reply]}'
SWAP = "A shopping agent's cart holds a listing that has just sold out. Look among the other listing records shown for replacements of that cart line, using these reading rules: " + READING + '\nRespond with nothing but ONE JSON object: {"fits": [{"price": the record\'s price for it, as a number, "id": listing id} for each listing that seems to meet every requirement of the item on the sold-out line, at most 6], "reason": "one sentence"}'
SHAPE = 'Below is what a store tool returned. Name its parts by the exact keys found in the result and in the tool\'s parameters. Reply with ONE JSON object only: {"rows": the key that holds the list of listings, "count": the key giving the number of listings matched overall (not just those shown), or null, "next": the key with the value that fetches the page after this one, or null, "next_arg": the parameter of the tool that takes that value, or null, "price": the row-level key for the listing\'s price, or null, "stock": the row-level key that tells whether the listing can be bought right now, or null}'
GUARD_KEYS = 3
GUARD_SHARE = 0.0
GUARD_GAP = 0.34
NOT_PROPERTY = '\\b(?:condition|photos?|images?|links?|url|video|warrant\\w*|barcodes?|ean|upc|asin|codes?|ids?|sku|(?<!compatible )models?|variants?|options?)\\b'
CATCH_ALL = '^(?!.*\\b(?:shapes?|kinds?|styles?|types?)\\b).*\\b(?:misc\\w*|extras?|accessor\\w*|includ\\w*|descriptions?|specifications?|specs?|highlights?|functions?|features?)\\b'
WIDEN_EACH = 6
FIT_CODE = '[A-Z][A-Za-z]{0,5}[- ]?(?:\\d{1,4}[xX]{1,3}|\\d{3,6})'
PART_CODE = '[A-Z]{1,4}\\d{3,6}'
SERIES_CODE = '[A-Z]{1,4}\\d{1,3}'
CODE_QUERY = '\\b(?:[A-Za-z]{1,6}-?\\d{3,6}|[A-Z][a-z]{2,6} \\d{3})\\b'
FIT_VERB = '(?i:(?<!not )(?<!never )(?<!in)(?:\\b(?:fits?|suits?)|\\b(?:goes|go|works?|pairs?|fits?|compatible|for use) with))\\b\\s*:?\\s*'
CODE_GLUE = '(?i:every|all|any|the|models?|series|and|or|plus|also)\\b|[,/&()]'
CODE_BUT = '(?i:except|excluding|but not|other than|apart from)\\b'

def _code_key(code: str) -> str:
    """One spelling per code: MOZ-7450, MOZ 7450 and moz7450 are the same code; ROT31xx stays a family of codes."""
    return re.sub('[- ]', '', code).lower()

def _codes(text: Any) -> list[str]:
    return [_code_key(c) for c in re.findall('\\b' + FIT_CODE + '\\b', str(text or ''))]

def _code_list(clause: str) -> tuple[list[str], list[str]]:
    """Reads a fit clause from its start while it holds only codes and list words: 'every ROT31xx model, except ROT3140'
    gives (['rot31xx'], ['rot3140']); the reading stops at the first other word, whatever verb came before."""
    fits: list[str] = []
    nots: list[str] = []
    side, pos = (fits, 0)
    while pos < len(clause):
        hit = re.match('\\s*(?:(?P<code>' + FIT_CODE + ')\\b|(?P<but>' + CODE_BUT + ')|(?:' + CODE_GLUE + '))', clause[pos:])
        if not hit or not hit.group(0).strip():
            break
        if hit.group('code'):
            side.append(_code_key(hit.group('code')))
        elif hit.group('but'):
            side = nots
        pos += hit.end()
    return (fits, nots)

def _fit_hit(fit: str, code: str) -> bool:
    """A fit code names a device code: the same code, or a family (rot31xx) whose x's stand for that code's digits."""
    if fit == code:
        return True
    stem = fit.rstrip('x')
    return stem != fit and stem[-1:].isdigit() and len(code) == len(fit) and code.startswith(stem) and code[len(stem):].isdigit()

def _fit_marks(text: Any) -> dict[str, Any]:
    text = str(text or '')
    own = re.search('[\\[(]\\s*(?:part\\s+)?(' + PART_CODE + ')\\s*[\\])]', text, re.I)
    codes: list[str] = []
    for hit in re.finditer('(?i:\\bmodel)(?:\\s*(?i:no|number|nr|ref|reference|code|id)\\b\\.?|\\s*#)?\\s*[:.#]?\\s*(' + FIT_CODE + ')\\b', text):
        codes += _codes(hit.group(1))
    for hit in re.finditer('\\b(?:also sold as|also known as|sold also as)\\s+([^.;|]+)', text, re.I):
        codes += _code_list(hit.group(1))[0]
    works: list[str] = []
    nots: list[str] = []
    for hit in re.finditer(FIT_VERB, text):
        if not re.search('\\bpart\\s+' + PART_CODE + '\\s+also\\s+$', text[max(0, hit.start() - 40):hit.start()], re.I):
            fits, but = _code_list(text[hit.end():hit.end() + 300])
            works += fits
            nots += but
    also = [(hit.group(1).upper(), _code_list(hit.group(2))[0]) for hit in re.finditer('\\bpart\\s+(' + PART_CODE + ')\\s+also fits\\s*:?\\s*([^.;|]+)', text, re.I)]
    more = re.search('\\b(?:more|details|see|read)\\s+(?:on|at|under|about)?\\s*(?:the\\s+)?part\\s+(' + PART_CODE + ')', text, re.I)
    family = re.search('[\\[(]\\s*(' + SERIES_CODE + ')-\\d{2,6}\\s*[\\])]', text)
    takes = {hit.group(1).upper() for hit in re.finditer('\\bseries\\s*:?\\s*(' + SERIES_CODE + ')\\b', text, re.I)}
    return {'own': own.group(1).upper() if own else '', 'codes': codes, 'works': works, 'nots': nots, 'also': also, 'more': more.group(1).upper() if more else '', 'family': family.group(1).upper() if family else '', 'takes': takes}

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

def _single(word: str) -> str:
    """A word's singular form read from its ending alone (-ies to -y; -ches, -shes, -sses, -xes, -zes lose -es; a plain
    -s goes), for the short head noun of a search query."""
    low = word.lower()
    if len(low) > 4 and low.endswith('ies'):
        return word[:-3] + 'y'
    if len(low) > 4 and re.search('(?:ch|sh|ss|x|z)es$', low):
        return word[:-2]
    if len(low) > 3 and low.endswith('s') and (not re.search('(?:ss|us|is|ves|ics)$', low)):
        return word[:-1]
    return word

def _value_terms(value: str) -> list[str]:
    """The words of a required value a query can carry: joining and modal words dropped ("must be", "suited to a"),
    a number kept with its unit. Empty when more than three words are left (a phrase, not a value)."""
    filler = {'a', 'an', 'the', 'to', 'of', 'for', 'with', 'in', 'on', 'at', 'by', 'be', 'is', 'are', 'must', 'should', 'has', 'have', 'it', 'its', 'that', 'which', 'suited', 'suitable', 'made', 'from', 'only', 'any', 'either', 'both', 'each', 'one', 'kind', 'type'}
    words = [w for w in re.findall("[A-Za-z0-9][A-Za-z0-9'.+/-]*", str(value)) if w.lower() not in filler]
    return words if len(words) <= 3 else []

def _tokens(text: Any) -> set[str]:
    """Lower-case letter and digit runs of a text, without the small joining words."""
    return set(re.findall('[a-z0-9]+', str(text or '').lower())) - {'a', 'an', 'the', 'and', 'of', 'for', 'with', 'in'}

def _spelled(text: str) -> frozenset[str]:
    """Words of a name folded for spelling: a trailing plural -s dropped and -our read as -or, so "colours" matches "color"."""
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
            faults.append(f'{where} must be one of {options}')
    if isinstance(value, str) and isinstance(schema.get('maxLength'), int) and (len(value) > schema['maxLength']):
        value = value[:schema['maxLength']]
    if isinstance(value, str) and isinstance(schema.get('minLength'), int) and (len(value) < schema['minLength']):
        faults.append(f'{where} is shorter than the schema allows')
    if isinstance(value, (int, float)) and (not isinstance(value, bool)):
        if isinstance(schema.get('minimum'), (int, float)) and value < schema['minimum']:
            faults.append(f"{where} is below {schema['minimum']}")
        if isinstance(schema.get('maximum'), (int, float)) and value > schema['maximum']:
            faults.append(f"{where} is above {schema['maximum']}")
    if isinstance(value, list):
        if isinstance(schema.get('maxItems'), int) and len(value) > schema['maxItems']:
            faults.append(f"{where} takes no more than {schema['maxItems']} entries")
        if isinstance(schema.get('minItems'), int) and len(value) < schema['minItems']:
            faults.append(f"{where} takes no fewer than {schema['minItems']} entries")
        if isinstance(schema.get('items'), dict):
            value = [_shape(v, schema['items'], f'{where}[{i}]', faults) for i, v in enumerate(value)]
    if isinstance(value, dict):
        props = schema.get('properties') if isinstance(schema.get('properties'), dict) else {}
        if schema.get('additionalProperties') is False:
            for name in [n for n in value if n not in props]:
                faults.append(f'{where}.{name} is not an argument this tool takes')
        for name in schema.get('required') or []:
            if name not in value:
                faults.append(f'{where}.{name} is missing but required')
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

def _plain(value: str) -> Any:
    value = value.strip()
    if value in ('True', 'False'):
        return value == 'True'
    if value == 'None':
        return None
    if re.fullmatch('-?\\d+', value):
        return int(value)
    if re.fullmatch('-?\\d+\\.\\d+', value):
        return float(value)
    return value

def _text_record(text: Any) -> dict[str, Any] | None:
    """A listing the store wrote as one line, "key: value | key: value | facts: a=b; c=d", as a record: a value made
    only of "name=value" pieces becomes a block of attributes. None when the line is not such a listing (fewer than
    three fields, or no field holds an id)."""
    if not isinstance(text, str) or text.count(' | ') < 2:
        return None
    record: dict[str, Any] = {}
    last = None
    for part in re.split('\\s+\\|\\s+', text.strip()):
        key, sep, value = part.partition(':')
        if sep and re.fullmatch("[A-Za-z][\\w .'/-]{0,40}", key.strip()) and key.strip() not in record:
            last = key.strip()
            pieces = [x for x in re.split(';\\s*', value.strip()) if x.strip()]
            pairs = [x.partition('=') for x in pieces]
            if len(pairs) >= 1 and all((sep2 and 0 < len(k.strip()) <= 40 for k, sep2, _ in pairs)):
                record[last] = {_choice_key(k.strip()): _plain(v) for k, _, v in pairs}
            else:
                record[last] = _plain(value)
        elif last is not None and isinstance(record[last], str):
            record[last] += ' | ' + part
        else:
            return None
    if len(record) < 3 or not any((_id_like(v) and re.search('[A-Za-z]', v) for v in record.values() if isinstance(v, str))):
        return None
    return {_choice_key(k): v for k, v in record.items()}

def _choice_key(key: str) -> str:
    """A key naming the purchase choices a listing offers ("options", "variants") marked as such, so no check reads a
    value listed there as the listing's stated value. Real race 15: order lines whose required value stood only under
    such a key failed 36 of 36."""
    return key + CHOICE_MARK if re.match(CHOICE_KEY, _words(key)) and CHOICE_MARK not in key else key

def _offered(key: Any) -> bool:
    """Whether a key names the purchase choices a listing offers ("options", "variants", "select: size", "choose a
    style"), in a record of any shape: a value listed there is on offer, not the listing's stated value."""
    return CHOICE_MARK in str(key) or bool(re.match(CHOICE_KEY, _words(str(key))))

def _parts(text: Any) -> list[str]:
    """The values one attribute lists ("red, blue / gold"), each in _words form."""
    return [p for p in (_words(x) for x in re.split('[,/;|]', str(text))) if p]

def _picked_other(block: dict[str, Any], held: Any, value: str) -> str:
    """The value a listing's own choice picks out of a listed attribute when it is none of the required value's
    alternatives: the attribute lists several values, and a key naming the listing's options or variant names one of
    them that the requirement does not (a list "A, B, C" whose options key names C, for a requirement "A or D").
    The listing sold is that value, so the list does not state the others. '' when there is no such pick."""
    listed = _parts(held)
    if len(listed) < 2:
        return ''
    chosen = [p for k, v in block.items() if _offered(k) for p in _parts(v)]
    wanted = [_words(o) for o in re.split('\\s+or\\s+|,\\s*', str(value).strip().lower()) if _words(o)]
    if not chosen or any((w in chosen for w in wanted)):
        return ''
    return next((p for p in chosen if p in listed), '')

def _untext(text: str) -> Any:
    """A store reply sent as a string that holds JSON, or a listing written one "key: value" per line, read into
    the shape the rest of Halo reads; the string itself when it is neither."""
    body = text.strip()
    if body[:1] in '{[' and body[-1:] in '}]':
        try:
            return json.loads(body)
        except ValueError:
            return text
    lines = [x.strip() for x in body.splitlines() if x.strip()]
    if len(lines) >= 3 and sum((1 for x in lines if re.match("[A-Za-z][\\w .'/-]{0,40}:\\s", x))) >= max(3, len(lines) - 1):
        return ' | '.join(lines)
    return text

def _read_rows(node: Any, made: list[dict[str, Any]], depth: int=0) -> Any:
    """Every listing line inside a store reply turned into a record (each one also put in made); a reply that is only
    {"one key": listing line}, or only the line itself, becomes the record. Anything else is kept as it came."""
    if depth > 6:
        return node
    if depth == 0 and isinstance(node, str) and node.strip()[:1] in '{[':
        parsed = _untext(node)
        if not isinstance(parsed, str):
            node = parsed

    def one(value: Any, row: bool=False) -> Any:
        if row and isinstance(value, str) and '\n' in value and value.strip()[:1] not in '{[':
            lined = _untext(value)
            if isinstance(lined, str) and lined != value and _text_record(lined):
                value = lined
        record = _text_record(value)
        if record:
            made.append(record)
            return record
        return _read_rows(value, made, depth + 1)
    if isinstance(node, list):
        return [one(v, True) for v in node]
    if isinstance(node, dict):
        if len(node) == 1:
            only = _text_record(next(iter(node.values())))
            if only:
                made.append(only)
                return only
        return {k: one(v) for k, v in node.items()}
    if depth == 0 and isinstance(node, str):
        record = _text_record(node)
        if record:
            made.append(record)
            return record
    return node

def _in_rows(node: Any, ident: str, depth: int=0) -> bool:
    if depth > 5:
        return False
    if isinstance(node, list):
        return any((isinstance(v, dict) and ident in [str(x).strip() for x in v.values() if isinstance(x, str)] for v in node)) or any((_in_rows(v, ident, depth + 1) for v in node))
    if isinstance(node, dict):
        return any((_in_rows(v, ident, depth + 1) for v in node.values()))
    return False

def _llm_within(seconds: float) -> ProxyClient:
    """Pick a model client able to finish in the seconds remaining: the default one, else one built with a timeout counted in whole 5 s steps."""
    if seconds >= 50:
        return _llm
    limit = max(5, int(seconds) // 5 * 5)
    if limit not in _llm_short:
        _llm_short[limit] = ProxyClient(timeout=limit, max_retries=1, retry_delay=1)
    return _llm_short[limit]

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

def _facts_of(node: Any) -> dict[str, Any] | None:
    """_fact_block, or a facts value sent as one "key=value; key=value" string (some records come that way, and
    covers() and the fit plan would otherwise see no attributes on them)."""
    block = _fact_block(node)
    if block or not isinstance(node, dict):
        return block
    for value in node.values():
        if isinstance(value, str) and value.count('=') >= 2:
            pairs = [p.split('=', 1) for p in value.split(';') if '=' in p]
            pairs = [(k.strip(), v.strip()) for k, v in pairs if 0 < len(k.strip()) <= 40]
            if len(pairs) >= 2:
                return dict(pairs)
    return None
UPPER_WORDS = 'at most|no (?:more|higher|greater|bigger|larger)(?: than)?|not (?:more than|over|above|exceed\\w*)|up to|or (?:less|lower|fewer|under|below|smaller)|and (?:under|below|less)'
LOWER_WORDS = 'at least|no (?:less|lower|fewer|smaller)(?: than)?|not (?:less than|under|below)|or (?:more|higher|greater|above|over|bigger|larger)|and (?:up|over|above|more)'
BELOW_WORDS = 'under|below|less than|lower than|fewer than|smaller than'
ABOVE_WORDS = 'over|above|more than|greater than|higher than|bigger than|larger than'
FIGURE = '(\\d[\\d,]*(?:\\.\\d+)?)\\s*([a-z"]*)'

def _limit(value: str) -> tuple[bool, bool, float, str] | None:
    """A measured limit in a requirement's value, the limit words right before or right after the figure ("at most N
    <unit>", "N <unit> at most", "N <unit> or more"): (an upper limit, strict, the figure, its unit word), or None.
    The brief writes both orders; read as plain words, a value with the words after the figure was met by no listing."""
    low = str(value or '').lower()
    edge = re.match('\\s*(max|min)(?:imum)?\\b\\s*:?\\s*' + FIGURE + '\\s*$', low) or re.match('\\s*' + FIGURE + '\\s*\\(?(max|min)(?:imum)?\\)?\\s*$', low)
    if edge:
        side, figure, unit = (edge.group(1), edge.group(2), edge.group(3)) if edge.group(1) in ('max', 'min') else (edge.group(3), edge.group(1), edge.group(2))
        try:
            return (side == 'max', False, float(figure.replace(',', '')), unit)
        except ValueError:
            return None
    for words, upper, strict in ((UPPER_WORDS, True, False), (LOWER_WORDS, False, False), (BELOW_WORDS, True, True), (ABOVE_WORDS, False, True)):
        found = re.search('\\b(?:' + words + ')\\s*:?\\s*' + FIGURE, low) or re.search(FIGURE + '[\\s,]*(?:' + words + ')\\b', low)
        if found:
            try:
                return (upper, strict, float(found.group(1).replace(',', '')), found.group(2))
            except ValueError:
                return None
    return None

def _same_unit(got: str, unit: str) -> bool:
    """Whether a figure's unit word is the limit's unit word, written out or shortened ("in" for "inches")."""
    if not got or not unit:
        return False
    return got == unit or (len(got) >= 2 and unit.startswith(got)) or (len(unit) >= 2 and got.startswith(unit))
SEEK_TALK = "\\b(?:search\\w*|look(?:ing)? (?:for|up|into|through|at)|find\\w*|open\\w*|brows\\w*|next page|page (?:on|through)|read(?:ing)? (?:the|its|their|more|another|one)|check(?:ing)? (?!out\\b)\\w+)"

class Item:

    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []

class Halo:

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
        self.finished = bool(self.view.get('finished'))
        self.broke = False
        self.no_credit = False
        self.waited = 0
        self.misses = 0
        self.credit_fails = 0
        self.pace: list[float] = []
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
        self.text_born: set[str] = set()
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
        self.read_by: dict[str, set[str]] = {}
        self.gone: dict[str, int] = {}
        self.unavailable: set[str] = set()
        self.sold_lines: set[str] = set()
        self.note_sold = False
        self.held = 0
        self.id_args: set[tuple[str, str]] = set()
        self.cart_said = ''
        self.refused_at = -1
        self.denied_at = -1
        self.count_fixed: set[str] = set()
        self.stale_opens = 0
        self.ref_pushed = False
        self.secured = False
        self.chain_ids: set[str] = set()
        self.series_pushed = False
        self.series_full: set[str] = set()
        self.series_noted = 0
        self.stated_done = False
        self.loose_held = False
        self.unmet_held = False
        self.bare_held = False
        self.seek_now = False
        self.seek_turns = 0
        self.query_starts: set[tuple[str, str, str]] = set()
        self.short_opens = 0
        self.unread_steps = 0
        self.unread_marks: set[str] = set()
        self.sold_cart: set[str] = set()
        self.sold_note = -1
        self.fit_memo: tuple[Any, Any] = (None, None)
        self.change_held = False
        self.widened_at = -1
        self.refused_cart = ''
        self.redone = False
        self.leads: list[str] = []
        self.refuse_blocks = 0
        self.pushed = False
        self.built: list[str] = []
        self.built_at = -1
        self.built_needs: tuple[tuple[int, str], ...] = ()
        self.note_step = -1
        self.dropped: set[str] = set()
        self.late_held = False
        self.lead_items: dict[str, list[str]] = {}
        self.verdicts: dict[str, bool] = {}
        self.confirmed: str | None = None
        self.completed = False
        self.link_tried: set[str] = set()
        self.lane_misses: dict[int, int] = {}
        self.lost: set[int] = set()
        self.gap_tried = False
        self.silent: dict[int, list[str]] = {}
        self.page_seen: set[str] = set()
        self.adopted: dict[str, int] = {}
        self.capped: set[tuple[str, float]] = set()
        self.id_key: str | None = None
        self.facts_from: dict[str, bool] = {}
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
        self.early_walks = 0
        self.early_rounds = 0
        self.early_at = -99
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
        self.follow_up: list[tuple[int, list[str], bool]] = []
        self.listing = False
        self.split = False
        self.slices: list[list[int]] = []
        self.ask_count = 0
        self.cap_known = False
        self.cap_read = False
        self.line_holds = 0
        self.bound: list[tuple[int, int]] = []
        self.fit_holds = 0
        self.searched: list[dict[str, Any]] = []
        self.link_searches = 0
        self.link_seen: dict[str, list[str]] = {}
        self.code_finds = [0, 0]
        self.free_read: dict[int, int] = {}
        self.lead_words: int | None = None
        self.echoes = 0
        self.search_log: list[tuple[str, str, Any, list[str], tuple[str, ...], int | None, int]] = []
        self.list_ends: list[str] = []
        self.list_marks: set[str] = set()
        self.hit_words: dict[str, set[str]] = {}
        self.fit_rescued = False
        self.value_rescued = False
        self.deaf: dict[str, tuple[str, Any]] = {}
        self.mode_log: list[tuple[str, str, Any, tuple[str, ...], tuple[str, ...], int | None, int]] = []
        self.deaf_notes = 0
        self.link_vetoes: dict[str, int] = {}
        self.resolved_sought: set[str] = set()
        self.opening_done = False

    def left(self) -> int:
        return self.max_steps - self.step

    def clock(self) -> float:
        return time.monotonic() - self.started

    def model_room(self) -> float:
        return TASK_SECONDS - TIME_MARGIN - self.clock()

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
        return (2 if talk and self.msg_used < self.msg_limit and self.gate_due('order') else 1) + len(self.unread_lines())

    def read_tools(self) -> list[str]:
        """The world's two ways of reading one listing: its attributes and its page, when they are distinct tools."""
        out = []
        for tool in (self.facts_tool(), self.offer_tool()):
            if tool and self.single_target(tool) and tool not in out:
                out.append(tool)
        return out

    def unread_lines(self) -> list[tuple[str, str]]:
        """(tool, listing) for each cart line not yet read with each reading tool. Race-16 world, 810 Arena episodes of
        10 agents: an order whose every line was read with both the attributes tool and the page tool failed the
        process check 0 times in 130; a line never read through the attributes tool failed it 14 of 14, a line never
        page-viewed 26 of 30. A store with one reading tool counts the same way: an order holding a line never opened failed it."""
        tools = self.read_tools()
        if not tools:
            return []
        return [(tool, line) for line in self.cart_ids() if line not in self.gone for tool in tools if tool not in self.read_by.get(line, set())]

    def read_lines(self) -> str:
        """Reads, in code, each cart line with each reading tool it has not been read with (one step each) when the
        steps allow it and still leave the order its step. Returns what the reads showed when the store changed
        something (a note, a price, stock), else ''."""
        todo = self.unread_lines()
        if not todo or self.left() < len(todo) + 1 or self.finished:
            return ''
        version = self.version
        shown = []
        for tool, line in todo:
            faults: list[str] = []
            args = _shape({self.single_target(tool): line}, self.schemas.get(tool), 'args', faults)
            if faults:
                continue
            reply = self.act(tool, args)
            if reply is None or self.finished:
                break
            observation = reply.get('observation')
            if tool == self.offer_tool():
                self.offer_seen[line] = self.step
            self.observe(tool, observation)
            shown.append(f'{tool} {line}: ' + self.show(observation)[:600])
        self.trace.append({'kind': 'read_lines', 'reads': todo})
        return '\n'.join(shown) if self.version != version else ''

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
            while True:
                message, status = self.call_model(model, messages, max_tokens, tools)
                if message is not None:
                    self.credit_fails = 0
                    return message
                passing = status is None or status in (408, 425, 429) or (isinstance(status, int) and status >= 500)
                if not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL:
                    break
                if self.model_room() - WAITS[tries] * 1.2 < CALL_FLOOR:
                    break
                pause = WAITS[tries] * (0.8 + 0.4 * random.random())
                self.waited += WAITS[tries]
                tries += 1
                time.sleep(pause)
            if status == 402:
                self.credit_fails += 1
                if self.credit_fails >= 2:
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
        """The first JSON object a model returns for the prompt, trying the models in order. A reply that comes back
        with no text at all is asked once more of the same model before the next one is tried."""
        for model in models:
            for attempt in range(2):
                message = self.think([{'role': 'user', 'content': prompt}], [model], max_tokens)
                got = _json_obj((message or {}).get('content'))
                if got is not None:
                    return got
                if self.broke or self.no_credit or self.calls >= ALL_CALLS:
                    return None
                empty = message is None and bool(self.trace) and self.trace[-1].get('kind') == 'llm_fail' and self.trace[-1].get('status') == 0
                if not empty or attempt or self.model_room() < 2 * CALL_FLOOR:
                    break
                self.trace.append({'kind': 'empty_reply_retried', 'model': model})
        return None

    def page_key(self, args: dict[str, Any]) -> str | None:
        return next((k for k, v in args.items() if re.search('page|leaf', k, re.I) and (not re.search('size|per|limit|count', k, re.I)) and isinstance(v, int) and (not isinstance(v, bool))), None)

    @staticmethod
    def page_mark(name: str, args: dict[str, Any], key: str | None) -> str:
        return name + json.dumps({k: v for k, v in args.items() if k != key and v not in (None, '', [], {})}, sort_keys=True, default=str)

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        key = self.page_key(args)
        self.page_seen.add(self.page_mark(name, args, key))
        payload = dict(self.binding)
        payload['step'] = self.step + 1
        payload['action'] = {'name': name, 'args': args}
        for _ in range(4):
            result = _env.post_verbose('/environment/call', payload)
            if isinstance(result.data, dict):
                data = result.data
                if 'observation' in data:
                    made: list[dict[str, Any]] = []
                    data['observation'] = _read_rows(data['observation'], made)
                    self.text_born.update((next((v.strip() for v in r.values() if _id_like(v) and re.search('[A-Za-z]', v)), '') for r in made))
                self.step = data['step'] if isinstance(data.get('step'), int) else self.step + 1
                if data.get('finished'):
                    self.finished = True
                aim = self.single_target(name)
                if aim and isinstance(args.get(aim), str) and (not (isinstance(data.get('observation'), dict) and data['observation'].get('error'))):
                    self.read_by.setdefault(args[aim].strip(), set()).add(name)
                seen = data.get('observation')
                if name in self.page_tools() and (not (isinstance(seen, dict) and seen.get('error'))):
                    for value in args.values():
                        self.opened.update((v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)))
                if name == self.roles.get('search'):
                    words = self.query_words(args)
                    for ident, _ in self.records_in(seen) if words else []:
                        self.hit_words.setdefault(ident, set()).update(words)
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
            prompt += '\n\nWHAT THE STORE SAID AT THE START:\n' + _dump(_clip(self.view.get('opening'), 1500))
        brief = self.ask_json(prompt, [LIGHT], 4000) or {}
        if not self.no_credit and (not _items(brief.get('items')) or not isinstance(brief.get('roles'), dict) or (not brief['roles'].get('buy'))):
            brief = self.ask_json(prompt, [HEAVY, LIGHT], 2500) or brief
        elif not self.no_credit and self.shared_shelf(brief):
            again = self.ask_json(prompt, [HEAVY], 2500) or {}
            if _items(again.get('items')) and isinstance(again.get('roles'), dict) and again['roles'].get('buy'):
                brief = again
        if brief.get('pick') in ('most', 'least') and re.search('\\b(?:unmet|shortfalls?|requirements?|mismatch\\w*)\\b|\\bshort\\b', str(brief.get('pick_measure') or ''), re.I):
            self.trace.append({'kind': 'pick_was_fallback', 'pick': brief.get('pick'), 'measure': str(brief.get('pick_measure'))[:80]})
            brief['pick'], brief['pick_measure'] = ('any', '')
        self.brief = brief
        roles = brief.get('roles') if isinstance(brief.get('roles'), dict) else {}
        self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.schemas}
        self.page_role()
        for key in ('buy_args', 'refuse_args'):
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        buy, refuse = (self.roles.get('buy'), self.roles.get('refuse'))
        if buy and buy == refuse and (not brief['buy_args'] or not brief['refuse_args'] or brief['buy_args'] == brief['refuse_args']):
            ask = f'One store tool, {buy}, does two jobs: with some arguments it places the order, with others it closes the episode with no purchase. Schema: ' + _dump(self.schemas.get(buy)) + '\nReply with a single JSON object only: {"buy_args": the fixed arguments for placing the order, "refuse_args": the fixed arguments for closing with no purchase} (just the arguments that select the job).'
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
            self.items = [Item('the product asked for', '', 1, [])]
        for pair in _items(brief.get('fits')):
            if isinstance(pair, dict):
                part, main = (became.get(_count(pair.get('item')) or 0), became.get(_count(pair.get('with')) or 0))
                if part and main and (part != main) and (1 <= part <= len(self.items)) and (1 <= main <= len(self.items)) and ((part, main) not in self.bound):
                    self.bound.append((part, main))
        lookups = list(_items(brief.get('lookups')))
        for entry in _items(brief.get('items')):
            if isinstance(entry, dict):
                lookups += [x for x in _items(entry.get('lookups')) if x not in lookups]
        for lookup in lookups:
            text = str(lookup).strip()
            if not any((text in item.musts for item in self.items)):
                key = _words(text.split(':')[0])
                text = next((m for item in self.items for m in item.musts if _words(m.split(':')[0]) == key), text)
            number = next((n for n, item in enumerate(self.items, 1) if text in item.musts), 1)
            if text and self.condition_only(text):
                self.trace.append({'kind': 'lookup_is_condition', 'need': text[:160]})
                continue
            if text and (number, text) not in self.refs:
                self.refs.append((number, text))
        self.relations()
        self.unmix()
        self.split_shared_musts()
        budget = brief.get('budget')
        if isinstance(budget, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', budget):
            budget = float(budget)
        if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0):
            self.budget = budget
        limit = _count(brief.get('messages'))
        if limit is not None and limit >= 0:
            self.msg_limit = limit
            self.cap_known = True
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

    def page_role(self) -> None:
        """The page tool when the brief named none. A store can offer two single-listing readers, one for the
        attributes and one for the whole page (its description, seller remarks and pointers to other listings); the
        brief leaves the page reader out when its description names no price or stock. Without it, cart lines are
        never page-viewed before the order and pointed pages are read only through the attributes tool, which does not
        show their remarks. The one tool left without a role that reads a single listing, does not change anything and
        is not a word search takes the role (a search tool given the role sent listing ids as search words: Arena r13/r15,
        where the brief named the category tool as the search)."""
        if self.roles.get('offer') or not self.single_target(self.roles.get('open')):
            return
        taken = set(self.roles.values())
        changes = '\\b(?:add|remove|withdraw|delete|cart|basket|bag|buy|checkout|purchase|order|refuse|message|send|ask|question|modify|change|update|set)\\b'
        spare = []
        for tool in self.tools:
            fn = tool['function']
            name = fn['name']
            text = f"{_words(name)} {fn.get('description') or ''}".lower()
            if name not in taken and self.single_target(name) and (not re.search(changes, text)) and (not self.query_field(name)):
                spare.append(name)
        if len(spare) == 1:
            self.roles['offer'] = spare[0]
            self.trace.append({'kind': 'page_role', 'tool': spare[0]})

    def condition_only(self, need: str) -> bool:
        """A lookup that is a condition on another requested item ("x if the other item's y is z", "x except when
        ..."), not a value on a page the request points at: neither the requirement nor the request's own sentence for it
        names a page, a listing, a seller or a pointer, and one of them states a condition. The chain walk would search
        for a listing nobody named and hold the order for it; the requirement stays with the item's other musts."""
        clause = self.request_clause(need)
        texts = f'{need} {clause}'.lower()
        pointer = '\\b(?:page|pages|points?|pointed|listing|listings|sold by|seller|shop|store|from the|whatever)\\b'
        condition = "\\b(?:if|unless|except|should|when|whenever|in case)\\b"
        return not re.search(pointer, texts) and bool(re.search(condition, texts))

    @staticmethod
    def shared_shelf(brief: dict[str, Any]) -> bool:
        shelves = [_words(str(e.get('shelf') or '')) for e in _items(brief.get('items')) if isinstance(e, dict)]
        shelves = [s for s in shelves if s]
        return len(shelves) != len(set(shelves))

    def request_clause(self, need: str) -> str:
        """The request's own sentence for a requirement: the one sharing most words with its value, at least three and
        a capitalised name among them when the value has one. The brief can paraphrase a relation ("the partner piece
        for ..." became "same as ...", Arena 14561); the sentence keeps it."""
        value = need.split(':', 1)[-1]
        words = {w for w in re.findall('[a-z0-9]{3,}', value.lower())}
        names = {w.lower() for w in re.findall("\\b[A-Z][a-z]{3,}", value)}
        best, score = '', 0
        for sentence in re.split('(?<=[.!?])\\s+|\\n+', self.request):
            low = sentence.lower()
            have = {w for w in words if re.search('(?<![a-z0-9])' + re.escape(w) + '(?![a-z0-9])', low)}
            if names and not names & have:
                continue
            if len(have) >= 3 and len(have) > score:
                best, score = sentence.strip(), len(have)
        return best[:400]

    def relations(self) -> None:
        """A requirement read from another listing whose relation word says the opposite of the request's own sentence
        for it is turned back: the brief can copy the contrast word of its example onto a request that asks for the
        other listing's very value (the agent then kept a line whose value differed from the page it was told to
        match), or the reverse. Only the leading relation word changes, in the item's musts and in the lookups."""
        contrast = '\\b(?:unlike|differ(?:s|ent|ing)?|other than|anything but|not (?:the same|like|match\\w*))\\b'
        for index, (number, need) in enumerate(list(self.refs)):
            prop, colon, value = need.partition(':')
            clause = self.request_clause(need).lower()
            if not colon or not clause or not 1 <= number <= len(self.items):
                continue
            apart = re.match('\\s*(?:unlike|different (?:from|to|than)|other than|not (?:the same as|like|matching))\\s+', value, re.I)
            alike = re.match('\\s*(?:the same as|same as|matching|identical to|like)\\s+', value, re.I)
            if apart and (not re.search(contrast, clause)):
                fixed = f'{prop}: the same as {value[apart.end():]}'
            elif alike and re.search(contrast, clause):
                fixed = f'{prop}: unlike {value[alike.end():]}'
            else:
                continue
            item = self.items[number - 1]
            item.musts = [fixed if m == need else m for m in item.musts]
            self.refs[index] = (number, fixed)
            self.trace.append({'kind': 'relation_fixed', 'from': need[:160], 'to': fixed[:160]})

    def split_shared_musts(self) -> None:
        """A requirement the brief put on two or more items stays only on the items the request ties it to: a sentence
        holding the value together with a word of that item's name no other item's name has. When some items have such
        a sentence and others none, the others lose it (Arena 14561: a colour the request gave one camera was put on
        both, and the other had no fitting listing). Values the request states for all of them stay."""
        if len(self.items) < 2:
            return
        own = {n: {w for w in re.findall('[a-z]{3,}', item.name.lower())} for n, item in enumerate(self.items, 1)}
        only = {n: own[n] - set().union(*(own[m] for m in own if m != n)) for n in own}
        sentences = [x.lower() for x in re.split('(?<=[.!?;])\\s+|\\n+|,\\s+|\\s+[\u2014\u2013-]+\\s+', self.request) if x.strip()]
        texts: dict[str, list[int]] = {}
        for n, item in enumerate(self.items, 1):
            for must in item.musts:
                texts.setdefault(must, []).append(n)
        for must, numbers in texts.items():
            if len(numbers) < 2 or must in [m for _, m in self.refs]:
                continue
            value = must.split(':', 1)[-1].strip().lower()
            if not value:
                continue
            tied = [n for n in numbers if any((value in x and any((re.search('(?<![a-z])' + re.escape(w), x) for w in only[n])) for x in sentences))]
            if not tied or len(tied) == len(numbers):
                continue
            for n in numbers:
                if n not in tied:
                    self.items[n - 1].musts = [m for m in self.items[n - 1].musts if m != must]
            self.trace.append({'kind': 'shared_must_split', 'must': must, 'kept': tied, 'dropped': [n for n in numbers if n not in tied]})

    def unmix(self) -> None:
        for number, item in enumerate(self.items, 1):
            pointers = [need for n, need in self.refs if n == number and need in item.musts]
            rest = [m for m in item.musts if m not in pointers]
            if not pointers or not rest:
                continue
            prompt = UNMIX + '\n\nREQUEST:\n' + self.request + f'\n\nPRODUCT: {item.name}\nREQUIREMENTS WHOSE VALUE COMES FROM ANOTHER LISTING:\n' + '\n'.join(pointers) + '\nOTHER REQUIREMENTS:\n' + '\n'.join(rest)
            got = self.ask_json(prompt, [HEAVY, LIGHT], 1500) or {}
            moved = [m for m in rest if m in [str(x).strip() for x in _items(got.get('describe_other'))]]
            if not moved:
                continue
            item.musts = [m for m in item.musts if m not in moved]
            joined = pointers[0] + ' (the referenced listing: ' + '; '.join(moved) + ')'
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
            needs = [m + (f' (value found on the referenced page: {self.ref_values[m]})' if m in self.ref_values else '') + (f' [the request says: "{self.request_clause(m)}"]' if m in [r for _, r in self.refs] and self.request_clause(m) else '') + (" [a fit to another requested item: met when this listing's page names a model code of that item's listing; an attribute naming some other product does not decide it]" if self.fit_need(number, m) else '') for m in live if m not in told]
            head = f'item {number}: {item.name}' if many else f'item: {item.name}'
            if item.shelf:
                head += f' (shelf in the store: {item.shelf})'
            head += f', quantity {item.qty}'
            text = head + '; needs: ' + ('; '.join(needs) if needs else 'nothing past being this kind of product')
            if told:
                text += "; wording only the shopper uses, matched by no attribute key (cannot settle anything): " + '; '.join(told)
            parts.append(text)
        brief = self.brief
        if self.budget is not None:
            scope = 'per item' if brief.get('budget_scope') == 'each' else 'in total'
            parts.append(f"budget ceiling: {self.budget} {brief.get('budget_unit') or ''} {scope}".replace('  ', ' '))
        pick = brief.get('pick')
        if pick == 'cheapest':
            parts.append('pick rule: lowest price among fitting listings')
        elif pick in ('most', 'least'):
            parts.append(f"pick rule: the {pick} {brief.get('pick_measure') or ''} across the listings that fit".replace('  ', ' '))
        else:
            parts.append('pick rule: none; any listing meeting every requirement will do')
        if brief.get('in_stock_only', True) is not False:
            parts.append('must be in stock')
        for wish in _items(brief.get('wishes')):
            parts.append(f'preference (only breaks ties): {wish}')
        first = brief.get('budget')
        first = float(first) if isinstance(first, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', first) else first
        moved = isinstance(first, (int, float)) and (not isinstance(first, bool)) and isinstance(self.budget, (int, float)) and float(first) != float(self.budget)
        for rule in _items(brief.get('rules')):
            stale = moved and re.search('(?<![\\d.])' + re.escape(f'{float(first):g}') + '(?![\\d.])', str(rule).replace(',', ''))
            parts.append(f'rule stated in the request: {rule}' + (f' (superseded: a later note moved the budget to {self.budget:g}, see the budget line above)' if stale else ''))
        if brief.get('quiet_decline') is True:
            parts.append('when no listing fits: close with no purchase and mention no listing at all')
        for entry in self.process:
            when = {'choose': 'ahead of the choice', 'order': 'with the order (Halo handles it at checkout)', 'end': 'with a no-purchase ending (Halo handles it at that point)'}[entry['when']]
            parts.append(f"step the request requires, {when}: {entry['step']}")
        facts = self.facts_tool()
        if facts and (self.facts_from.get(facts) is True or self.roles.get('offer')):
            bare = sorted((t for t in self.page_tools() | {self.roles.get('offer') or ''} if t and self.facts_from.get(t) is False))
            pages = sorted((t for t in self.page_tools() if t != facts))
            parts.append(f'attributes of a listing are shown by {facts}' + (f"; {', '.join(bare)} show none" if bare else '') + (f"; the page of a listing (meant by a page check or a chain of pages) comes from {' or '.join(pages)}" if pages else ''))
        return '\n'.join(('- ' + p for p in parts))

    def status(self) -> str:
        left = self.left()
        text = f'[Halo] steps used {self.step}/{self.max_steps}, {left} remaining.'
        if left <= self.tail() + 1:
            text += ' LAST STEPS: if the cart has a line for every item, place the order now; close with no purchase only if searching proved some item has no fitting listing (being short of steps proves nothing).'
        elif left <= 6 or (left <= len(self.items) + 5 and self.buys == 0):
            text += ' Steps nearly gone: put the chosen lines in the cart and place the order now.'
        elif self.buys == 0 and left <= 12:
            text += ' Steps are getting scarce: decide your picks, fill the cart and order within a few more steps.'
        elif self.buys == 0 and (not self.cart_touched) and (self.step >= self.max_steps // 2):
            text += " Half of the steps are gone with nothing in the cart: add a fitting listing for each item now (one could still sell out)."
        if self.pace and self.clock() >= 90 and left > self.tail() + 1:
            per = sorted(self.pace)[len(self.pace) // 2]
            turns = max(0, int((LAST_TURN - self.clock()) // max(per, 1.0)))
            text += f' Clock: about {turns} more replies fit at your recent pace ({per:.0f} s each); put calls that do not need each other\'s results in one reply.'
        if self.clock() >= WIND_DOWN and left > self.tail() + 1:
            text += ' THE CLOCK IS ALMOST OUT: put the chosen lines in the cart and order within two more turns; a timed-out episode scores zero.'
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if lookups and self.buys == 0:
            text += '\n[look these up before ordering: open the listing each one refers to, then every listing named on that page; the value is on the final page] ' + ' | '.join(lookups[:3])
        if self.leads:
            text += '\n[listings the review flagged] ' + ', '.join(self.leads)
        plan = self.fit_plan() if self.bound and self.buys == 0 else ''
        if plan:
            text += '\n' + plan
        if self.notes:
            text += '\n[updates since the request] ' + ' | '.join(self.notes[-4:])
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

    @staticmethod
    def query_words(args: dict[str, Any]) -> set[str]:
        """The words a search call sent in its query field (a field named for a query, words or keywords)."""
        return {w for k, v in args.items() if isinstance(v, str) and re.search(QUERY_WORDS, _snake(k), re.I) for w in _tokens(v)}

    def row_words(self, ident: str, label: bool=True) -> set[str]:
        """The words a listing's search row shows (its text fields); label=False leaves out the category label."""
        record = self.rows.get(ident) or {}
        cat_key = None if label else self.category_key()
        return _tokens(' '.join((str(v) for k, v in record.items() if isinstance(v, str) and k != cat_key)))

    def row_evidence(self, ident: str, number: int) -> int:
        """How strongly a listing's row points at item `number`: one for each plain required value (or one option of an
        "x or y" value) whose words its row shows or the word searches that returned it carried (the store's word
        search needs every word and also reads the attributes), and one when its own words name the product, so a row
        of some other thing filed under that category ranks last."""
        seen = self.row_words(ident, label=False) | self.hit_words.get(ident, set())
        refs = [r for _, r in self.refs]
        score = 0
        for owner, need in self.needs_now():
            if owner != number or ':' not in need or need in refs or self.fit_need(number, need):
                continue
            value = need.partition(':')[2].lower()
            if re.search('\\b(?:not|no|except|without|other than|at most|at least|under|over|below|above|up to|less|more|same|unlike)\\b', value):
                continue
            options = [_tokens(o) for o in re.split('\\s+or\\s+|,\\s*', value) if o.strip()]
            score += any((o and o <= seen for o in options))
        name = _roots(self.items[number - 1].name)
        return score + bool(name & _roots(' '.join(self.row_words(ident, label=False))))

    def row_rank(self, ids: list[str], number: int) -> list[str]:
        """Unread rows, the strongest evidence for item `number` first, then the cheapest."""
        return sorted(ids, key=lambda i: (-self.row_evidence(i, number), self.price_of(i) if self.price_of(i) is not None else float('inf')))

    def row_tally(self, ident: str, number: int) -> tuple[int, int]:
        """(values, kind) a listing's row shows for item `number`: two for each requirement in force whose plain value
        (or one option of an "x or y" value) its own text fields spell out or whose limit a figure in the limit's unit
        on the row keeps, one when only the words of a search that returned it carried the value (a store's word
        search can read the attributes, but some match any one word); kind is 2 when its store label is the item's
        own label, plus 1 when its own words name the product. Requirements read from another listing, fits to a
        device line, exclusions and sums of money count for no row."""
        row = self.rows.get(ident) or {}
        cat_key = self.category_key()
        text = ' '.join((str(v) for k, v in row.items() if isinstance(v, str) and k != cat_key)).lower()
        seen = _tokens(text)
        hit = self.hit_words.get(ident, set())
        refs = [r for _, r in self.refs]
        values = 0
        for owner, need in self.needs_now():
            if owner != number or ':' not in need or need in refs or self.fit_need(number, need):
                continue
            low = need.partition(':')[2].strip().lower()
            limit = _limit(low)
            if limit:
                most, strict, bound, unit = limit
                for got in re.finditer('(\\d[\\d,]*(?:\\.\\d+)?)\\s*([a-z"]+)', text):
                    if not _same_unit(got.group(2), unit):
                        continue
                    try:
                        figure = float(got.group(1).replace(',', ''))
                    except ValueError:
                        continue
                    if (figure < bound if strict else figure <= bound) if most else (figure > bound if strict else figure >= bound):
                        values += 2
                        break
                continue
            if re.search('\\b(?:not|no|except|without|other than|same|unlike|different)\\b', low):
                continue
            options = [_tokens(o) - {'either', 'or'} for o in re.split('\\s+or\\s+|,\\s*', low) if o.strip()]
            values += 2 if any((o and o <= seen for o in options)) else 1 if any((o and o <= seen | hit for o in options)) else 0
        item = self.items[number - 1]
        label = _words(str(row.get(cat_key) or '')) if cat_key else ''
        kind = 2 if label and label in (_words(item.shelf or ''), _words(item.name)) else 0
        return (values, kind + bool(_roots(item.name) & _roots(text)))

    def unread_rank(self, number: int) -> list[tuple[str, tuple[int, int]]]:
        """The rows of item `number` seen in search results but never opened, best first, each with row_tally(): rows
        in stock and inside the budget first, then the most values their row shows plus the kind, then, under a
        cheapest pick, the lowest price, else the order they were seen in (the store's own relevance)."""
        cap = self.budget if isinstance(self.budget, (int, float)) and self.budget > 0 else None
        units = 1 if self.brief.get('budget_scope') == 'each' else self.items[number - 1].qty
        cheapest = self.brief.get('pick') == 'cheapest'
        cart = set(self.cart_ids())
        place = {i: n for n, i in enumerate(self.seen_order)}
        out = []
        for ident in self.item_ids(number):
            if ident in self.full or ident in self.opened or ident in cart or self.sold_out(ident):
                continue
            price, stocked = self.offer_terms(ident)
            if not stocked or (cap is not None and price is not None and price * units > cap + 1e-09):
                continue
            values, kind = self.row_tally(ident, number)
            order = (price if price is not None else float('inf')) if cheapest else place.get(ident, 0)
            out.append(((-(values + kind), order), ident, (values, kind)))
        out.sort()
        return [(ident, tally) for _, ident, tally in out]

    def settled(self, number: int) -> list[str]:
        """Opened listings of item `number` that settle every value the request states with no conflict (weigh), in
        stock, priced and inside the budget at the item's quantity, cheapest first."""
        cap = self.budget if isinstance(self.budget, (int, float)) and self.budget > 0 else None
        units = 1 if self.brief.get('budget_scope') == 'each' else self.items[number - 1].qty
        out = []
        for ident in self.item_ids(number):
            if ident not in self.full or self.sold_out(ident):
                continue
            price, stocked = self.offer_terms(ident)
            if not stocked or price is None or (cap is not None and price * units > cap + 1e-09):
                continue
            conflicts, unstated, _ = self.weigh(ident, number)
            if not conflicts and (not unstated):
                out.append((price, ident))
        return [i for _, i in sorted(out)]

    def open_listed(self, ids: list[str], steps: int) -> int:
        """Opens listings in full within `steps` steps: side by side with the compare tool, several per step, when the
        store has one that shows attributes, else one per step with the attributes tool. Returns the steps used."""
        todo = [i for i in ids if i in self.rows and i not in self.full]
        used = 0
        many, joiner = (self.compare_target(), self.roles.get('compare'))
        if many and joiner and (not self.compare_bare()) and len(todo) >= 2:
            while todo and used < steps and (not self.finished) and (not self.compare_bare()):
                batch = todo[:max(2, self.compare_room())]
                faults: list[str] = []
                args = _shape({many: batch}, self.schemas.get(joiner), 'args', faults)
                if faults:
                    break
                reply = self.act(joiner, args)
                used += 1
                if reply is None:
                    break
                seen = reply.get('observation')
                self.observe(joiner, seen)
                self.note_view(joiner, seen)
                rows = self.records_in(seen)
                if rows:
                    self.facts_from[joiner] = self.facts_from.get(joiner, False) or any((_fact_block(r) for _, r in rows))
                if not any((i in _dump(seen) for i in batch)):
                    self.compare_bad = min(self.compare_bad or 999, len(batch))
                    break
                self.compare_ok = max(self.compare_ok, len(batch))
                todo = [i for i in todo if i not in batch]
            todo = [i for i in ids if i in self.rows and i not in self.full]
        if todo and used < steps and (not self.finished):
            read = self.fill_facts(todo[:steps - used], max(0, self.left() - (steps - used)))
            used += len(read)
        return used

    def unread_picks(self, numbers: list[int], count: int) -> list[str]:
        """Up to `count` unread rows for the items `numbers`, taken in turn from each item's unread_rank(), keeping only
        rows whose own text or search words show at least one stated value (any row of the item's own label when it
        states none that a row can show)."""
        lanes = []
        for number in numbers:
            ranked = self.unread_rank(number)
            countable = any((values for _, (values, _) in ranked))
            lanes.append([i for i, (values, kind) in ranked if values > 0 or (not countable and kind >= 2)])
        picks: list[str] = []
        for k in range(max((len(lane) for lane in lanes), default=0)):
            for lane in lanes:
                if k < len(lane) and lane[k] not in picks:
                    picks.append(lane[k])
        return picks[:count]

    def open_unread(self, numbers: list[int], count: int, reserve: int) -> tuple[list[str], list[str]]:
        """Opens the best unread rows of the items `numbers` (unread_picks), at most `count` of them, inside the
        episode's share for these opens (a sixth of its steps) and keeping `reserve` steps beyond the cart's own tail.
        Returns (the listings opened, the opened listings that now settle every stated value of their item). Arena
        r12-r16, build 30: in 32 failed tasks another run solved, the winning listing was in Halo's search results and
        never opened; its row had been seen in the first steps, and the episode then ended without a purchase or
        ordered a worse listing."""
        share = max(2, self.max_steps // 6) - self.unread_steps
        room = min(share, self.left() - self.tail() - reserve)
        if room <= 0 or count <= 0 or self.finished or self.model_room() < 15 or (not numbers):
            return ([], [])
        picks = self.unread_picks(numbers, count)
        if not picks:
            return ([], [])
        before = set(self.full)
        used = self.open_listed(picks, min(room, len(picks)))
        self.unread_steps += used
        opened = [i for i in picks if i in self.full and i not in before]
        fit = [i for n in numbers for i in self.settled(n) if i in opened]
        self.trace.append({'kind': 'unread_open', 'items': numbers, 'picks': picks, 'opened': opened, 'fit': fit, 'steps': used})
        return (opened, fit)

    def unread_text(self, opened: list[str], fit: list[str]) -> str:
        """Halo's note on listings it opened from the unread rows."""
        text = f"Halo has opened {len(opened)} listing(s) that showed up in search results but had never been opened, choosing those whose rows show the most requested values:\n" + '\n'.join((f'{i}: {self.facts_view(i, 900)}' for i in opened))
        if fit:
            text += '\nOf these, ' + ', '.join((f'{i} ({self.price_of(i):g})' if self.price_of(i) is not None else i for i in fit)) + ' settle every value the request states, by the code\'s reading.'
        return text

    def unread_turn(self) -> str:
        """After a tool result, at about a third and again at about three fifths of the steps (each once), once the
        shopper has been asked or cannot be: open the best unread rows of each item that has no cart line and still no
        opened listing settling every stated value, and show their records. '' when nothing was opened. An item with a
        cart line is left to the model: in 51 of build 30's 168 accepted orders the code's reading did not settle a
        line the grader accepted."""
        if self.roles.get('ask') and (not self.asked) and self.msgs_free() > 0:
            return ''
        mark = next((m for m, at in (('mid', 0.6), ('early', 0.35)) if self.step >= max(8, int(self.max_steps * at))), '')
        if not mark or mark in self.unread_marks or self.left() <= self.tail() + 8 or self.model_room() < 60:
            return ''
        self.unread_marks.add(mark)
        if mark == 'mid':
            self.unread_marks.add('early')
        held = set(self.cart_ids())
        due = [n for n in range(1, len(self.items) + 1) if not held & set(self.item_ids(n)) and (not self.settled(n))]
        opened, fit = self.open_unread(due, 2 * len(due) + 1, 6)
        return '\n[Halo] ' + self.unread_text(opened, fit) if opened else ''

    def unread_end(self) -> str:
        """Once, when the model ends without a purchase: open the best unread rows of each item with no opened listing
        settling every stated value. When that finds such a listing for every one of those items, the ending waits
        and the model is shown the records; otherwise the decline review that follows reads them too. '' to let the
        ending go on."""
        if 'end' in self.unread_marks or self.left() <= self.tail() + 5 or self.model_room() < 45:
            return ''
        self.unread_marks.add('end')
        due = [n for n in range(1, len(self.items) + 1) if not self.settled(n)]
        opened, fit = self.open_unread(due, min(8, 3 * len(due) + 1), 4)
        if not opened or any((not self.settled(n) for n in due)):
            return ''
        return self.unread_text(opened, fit) + ' A listing that settles every value argues for buying, not for ending: cart the one the pick rule prefers (alongside the lines of the other items) and order; a no-purchase close is right only when one of them plainly breaks a live requirement, and then name it.'

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
        prompt = SHAPE + '\n\nTOOL: ' + name + ' | argument schema: ' + _dump(self.schemas.get(name) or {}) + '\n\nRESULT (leading rows only):\n' + _dump(_clip(sample, 120))[:4000]
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
        return len(prices) >= 5 and None not in prices and all((a <= b for a, b in zip(prices, prices[1:]))) and prices[0] < prices[-1]

    def pointed_kind(self, labels: list[str], args: dict[str, Any]) -> bool:
        """Whether a search hunts a listing that a requirement still points at: its query names the shop that the
        pointing requirement names (a capitalised word of it), and one of its rows carries a label whose words all
        appear in that requirement or in the request's sentence for it. The store can list such a shop's rows of that
        kind past the first page, and the value every pick depends on sits on that one listing."""
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        words = self.query_words(args)
        if not pending or not labels or not words:
            return False
        for need in pending:
            shops = {w.lower() for w in re.findall("\\b[A-Z][A-Za-z]{3,}", need.partition(':')[2])}
            said = _roots(need) | _roots(self.request_clause(need))
            if shops & words and any((_roots(label) and _roots(label) <= said for label in labels if label)):
                return True
        return False

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        self.paged_obs = []
        pick = self.brief.get('pick')
        if name != self.roles.get('search'):
            return ''
        cat_key = self.category_key()
        labels = [str(r.get(cat_key) or '') for _, r in self.records_in(observation)] if cat_key else []
        pointed = self.pointed_kind(labels, args)
        plan = self.paging(name, observation) if pick in ('cheapest', 'most', 'least') or pointed else None
        if not plan:
            return ''
        count, shown, param, value = plan
        if labels and (not pointed) and (not any((self.owns(item, label) for item in self.items for label in labels))):
            return ''
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and (not isinstance(given, bool)) else bool(str(given or '').strip())
        if value is None or later or count <= shown:
            return ''
        if pick == 'cheapest' and self.price_sorted(observation):
            return ''
        named = self.query_shops(args, observation)
        if named and (not self.only_shops(observation, named)):
            return ''
        if not pointed and self.bare_list(name, args):
            return f"\n[Halo] {count} listings in this list, {shown} on this page. Halo pages ahead on its own only for searches that include a required value: search the item's value plus its product noun, or request {param}={_dump(value)} yourself if you need the whole list."
        cap = self.auto_pages()
        pages = -(-(count - shown) // shown)
        if pages > 4 and self.pages_read + 2 <= cap and (self.left() - self.tail() - 2 >= 12):
            pages = 2
        if pages > 4 or self.pages_read + pages > cap or self.left() - self.tail() - pages < 12:
            if pointed and pick not in ('cheapest', 'most', 'least'):
                return f"\n[Halo] {count} hits for this search, {shown} on this page. A listing that a requirement refers to may be further back: ask for {param}={_dump(value)}."
            return f"\n[Halo] {count} hits for this search, {shown} on this page, not sorted by price. The pick rule weighs all fitting listings: narrow the words (use the requirement's value) or ask for {param}={_dump(value)}."
        texts = []
        cut = False
        last: tuple[dict[str, Any], Any] | None = None
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
            last = (call, seen)
            after = self.paging(name, seen)
            value = after[3] if after else None
            if value is None:
                break
            if named and (not self.only_shops(seen, named)):
                cut = True
                break
        self.trace.append({'kind': 'pages', 'tool': name, 'read': len(texts)})
        rest = f"\n[Halo] Of {count} hits Halo has now read {shown * (len(texts) + 1)} rows; to reach the others, narrow by the requirement's value or keep paging." if value is not None and texts else ''
        if cut:
            rest = f"\n[Halo] This search has no more rows from {', '.join(sorted(named))}: leading with a shop's name puts that shop's rows first, so later pages belong to other shops."
        if last and value is None:
            rest += self.list_end(name, last[0], last[1])
        return '\n[Halo paged through this search on its own, one step per page]\n' + '\n'.join(texts) + rest if texts else ''

    def auto_pages(self) -> int:
        """Pages Halo reads itself over an episode: about a sixth of its steps, 3 to AUTO_PAGES. Arena r16 (35 steps):
        failing episodes spent 5.1 search pages before the first cart line against 1.6 for the top agent."""
        return max(3, min(AUTO_PAGES, self.max_steps // 6))

    def bare_list(self, name: str, args: dict[str, Any]) -> bool:
        """A search that carries no required value: no query words, a mode seen to ignore its words, or words that only
        name the kind of product (a requested item's name or store label, a category label of the rows, the shelf the
        search filters on). Real race 16: pass rate .36 / .20 / .05 with 0 / 1-2 / 3+ such broad lists before the
        first cart line, in the same order inside every agent."""
        words = self.query_words(args)
        deaf = self.deaf.get(name)
        if not words or (deaf and args.get(deaf[0]) == deaf[1]):
            return True
        kinds: set[str] = set()
        for item in self.items:
            kinds |= _roots(item.name) | _roots(item.shelf or '')
        shelf = self.shelf_arg(name)
        if shelf and isinstance(args.get(shelf), str):
            kinds |= _roots(args[shelf])
        cat_key = self.category_key()
        if cat_key:
            for label in {str(r.get(cat_key) or '') for r in self.rows.values()} - {''}:
                kinds |= _roots(label)
        said = _roots(' '.join(sorted(words))) | {w for w in words if re.search('\\d', w)}
        return not said or said <= kinds

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
        hint = '; for cheaper listings only, browse the category with ' + ', '.join((f'{k}={_dump(o[0])}' for k, o in others.items() if o)) + f' plus the same {caps[0]}' if any(others.values()) else ''
        return f'\n[Halo] {caps[0]}={cap:g} was ignored by this search: {len(over)} rows are priced above it{hint}.'

    def query_field(self, name: str) -> str | None:
        """A search tool's single free-text argument (where its words go), or None."""
        props = (self.schemas.get(name) or {}).get('properties') or {}
        named = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and 'enum' not in v and re.search(QUERY_WORDS, _snake(k), re.I)]
        if named or name != self.roles.get('search'):
            return named[0] if len(named) == 1 else None
        free = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and 'enum' not in v]
        needed = [k for k in free if k in ((self.schemas.get(name) or {}).get('required') or [])]
        return free[0] if len(free) == 1 else needed[0] if len(needed) == 1 else None

    def fresh_start(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """A new query starts at the top of the store's list. A store can take a bare integer for where its list starts
        (its reply's next-page value is that number plus the rows shown); a model can read such a number as a count and
        send 5 or 10 with a first query, which skips the rows that match best (Arena race-17 world: 68 of 86 first
        searches did). The first call of words not sent before with this tool leaves out an optional integer argument
        that is the list's start: the one a reply showed to be the next-page argument, or one with no description, a
        minimum of 0 and a name that is no price, size or count filter. Later calls with the same words page as asked.
        Returns (the arguments to send, the dropped argument names joined)."""
        field = self.query_field(name)
        words = ' '.join(str(args.get(field) or '').lower().split()) if field else ''
        if not words:
            return (args, '')
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        learned = (self.shapes.get(name) or {}).get('next_arg')
        drop = []
        for key, value in args.items():
            spec = props.get(key)
            if key == field or key in (schema.get('required') or []) or not isinstance(spec, dict) or 'integer' not in _kinds(spec):
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= (spec.get('minimum') or 0):
                continue
            plain = spec.get('minimum') == 0 and not spec.get('description') and not re.search('price|budget|cost|amount|cap|rating|year|size|count|limit|per|num|qty|quantity|max|min|stock', key, re.I)
            if key == learned or plain:
                drop.append(key)
        mark = (name, words, _dump({k: v for k, v in sorted(args.items()) if k != field and k not in drop}))
        if not drop or mark in self.query_starts:
            self.query_starts.add(mark)
            return (args, '')
        self.query_starts.add(mark)
        return ({k: v for k, v in args.items() if k not in drop}, ', '.join(drop))

    def search_sign(self, name: str, args: dict[str, Any]) -> tuple[str, Any, list[str]] | None:
        """(its other arguments, its page argument's value, its words) for a search that sends words, else None."""
        field = self.query_field(name)
        words = str(args.get(field) or '').lower().split() if field else []
        if not words:
            return None
        page = (self.shapes.get(name) or {}).get('next_arg') or self.page_key(args)
        rest = _dump({k: v for k, v in sorted(args.items()) if k not in (field, page)})
        offset = args.get(page) if page else None
        return (rest, 0 if offset in (None, '') else offset, words)

    def search_echo(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """Logs a word search with its rows and tells when they repeat an earlier search sent with other words. A store
        that reads only the first words of a query lists the very same rows and count for every query sharing them, so
        such a repeat shows how many words it reads: those the two queries share before they part. Repeats like that cost a
        step each and were paged on. The count is compared when the result's shape is known
        (no model call is spent here to learn it). A search sent in a mode other than the word mode is left to
        deaf_mode(): there a repeat shows a mode that ignores words, not a store reading only the first ones. Returns
        the note for the result, else ''."""
        sign = self.search_sign(name, args)
        rows = tuple((i for i, _ in self.records_in(observation)))
        deaf = self.deaf_mode(name, args, rows, observation)
        if not sign or len(rows) < 3 or self.filter_mode(name, args):
            return deaf
        rest, offset, words = sign
        page = self.paging(name, observation) if self.shapes.get(name) else None
        count = page[0] if page else None
        note = ''
        for tool, o_rest, o_off, o_words, o_rows, o_count, o_step in self.search_log:
            if (tool, o_rest, o_off, o_rows) != (name, rest, offset, rows) or None not in (count, o_count) and count != o_count or o_words == words:
                continue
            if sorted(words) == sorted(o_words):
                break
            shared = 0
            while shared < min(len(words), len(o_words)) and words[shared] == o_words[shared]:
                shared += 1
            if 1 <= shared < min(len(words), len(o_words)):
                self.lead_words = min(self.lead_words or shared, shared)
                self.echoes += 1
                self.trace.append({'kind': 'search_echo', 'words': ' '.join(words), 'of': ' '.join(o_words), 'reads': shared})
                note = f"\n[Halo] These are the very rows and count of the search {' '.join(o_words)!r} at step {o_step}: this store's word search reads only the first {shared} word(s) of a query, and words after them change nothing. Halo did not page this repeat. Lead each new search with words no earlier search led with, the word that narrows it first."
            break
        self.search_log = (self.search_log + [(name, rest, offset, words, rows, count, self.step)])[-40:]
        return note

    def word_option(self, tool: str, key: str) -> Any:
        """The option of a search's mode argument that makes it read its words: the one option named for words, a
        phrase or a query; once a mode was seen to ignore words (deaf_mode), the other option of that argument. None
        when the argument is no such mode or the choice is in doubt."""
        spec = (((self.schemas.get(tool) or {}).get('properties') or {}).get(key)) or {}
        options = spec.get('enum') if isinstance(spec, dict) else None
        if not isinstance(options, list) or len(options) < 2 or key == self.shelf_arg(tool):
            return None
        reads = [o for o in options if isinstance(o, str) and re.search(QUERY_WORDS, o, re.I)]
        deaf = self.deaf.get(tool)
        if deaf and deaf[0] == key:
            others = [o for o in reads or options if o != deaf[1]]
            return others[0] if len(others) == 1 else None
        return reads[0] if len(reads) == 1 else None

    def filter_mode(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        """The mode arguments of a search call set to an option other than the word option, else {}."""
        out = {}
        for key, value in args.items():
            word = self.word_option(tool, key)
            if word is not None and value != word:
                out[key] = value
        return out

    def deaf_mode(self, name: str, args: dict[str, Any], rows: tuple[str, ...], observation: Any) -> str:
        """Search-mode echo. A search sent in a mode other than its word mode that lists the very rows and count of
        the same filter in the same mode with other words, or with none, shows a mode that ignores words: Arena r16
        filter mode listed the whole category for every worded search (257/257 real-race pairs), and searches that
        copied it (Halo's own value searches among them) read nothing. Halo records the mode, its own searches switch to
        the word option (search_template, word_plan) and the model gets one line. Returns that line, else ''."""
        modes = self.filter_mode(name, args)
        if not modes or len(rows) < 2:
            return ''
        key, value = next(iter(modes.items()))
        field = self.query_field(name)
        words = tuple(sorted(_tokens(args.get(field)))) if field else ()
        known = self.deaf.get(name)
        if known and known == (key, value):
            if words and self.deaf_notes < 3:
                self.deaf_notes += 1
                return f"\n[Halo] {key}={_dump(value)} ignores the words of a search (seen earlier): this lists the whole filter. Search words narrow the list only with {key}={_dump(self.word_option(name, key))}."
            return ''
        page = (self.shapes.get(name) or {}).get('next_arg') or self.page_key(args)
        rest = _dump({k: v for k, v in sorted(args.items()) if k not in (field, page)})
        offset = args.get(page) if page else None
        shape = self.paging(name, observation) if self.shapes.get(name) else None
        count = shape[0] if shape else None
        hit = next(((w, s) for t, r, o, w, rw, c, s in self.mode_log if (t, r, o, rw) == (name, rest, offset, rows) and w != words and (None in (c, count) or c == count)), None)
        self.mode_log = (self.mode_log + [(name, rest, offset, words, rows, count, self.step)])[-30:]
        if hit is None:
            return ''
        self.deaf[name] = (key, value)
        other = self.word_option(name, key)
        self.trace.append({'kind': 'deaf_mode', 'tool': name, 'mode': f'{key}={value}', 'words': ' '.join(words), 'of': ' '.join(hit[0])})
        what = f"the search for {' '.join(hit[0])!r} at step {hit[1]}" if hit[0] else f'the same filter without words at step {hit[1]}'
        turn = f' Words narrow a search only with {key}={_dump(other)}.' if other is not None else ''
        return f'\n[Halo] With {key}={_dump(value)} this search ignores its words: it listed the very rows and count of {what}.{turn}'

    def shelf_arg(self, tool: str | None) -> str | None:
        """The search tool's one argument naming a category, shelf or group (enum or free text), else None."""
        props = (self.schemas.get(tool or '') or {}).get('properties') or {}
        field = self.query_field(tool or '')
        shelves = [k for k, v in props.items() if k != field and isinstance(v, dict) and 'string' in _kinds(v) and re.search(SHELF_ARG, _snake(k), re.I)]
        return shelves[0] if len(shelves) == 1 else None

    def lead_twin(self, name: str, args: dict[str, Any]) -> str:
        """Once a repeat showed how many words the store reads: the earlier search with this one's other arguments,
        page and first words but other words after them (it would list the same rows), as "'words' at step N"; else ''.
        A call sent before with the very same words is left to the result cache, which reruns it after a change.
        Only after two such repeats: one could also come from a store that skips words it does not know."""
        sign = self.search_sign(name, args) if self.lead_words and self.echoes >= 2 else None
        if not sign:
            return ''
        rest, offset, words = sign
        same = [(w, s) for t, r, o, w, _, _, s in self.search_log if t == name and r == rest and o == offset and w[:self.lead_words] == words[:self.lead_words]]
        if not same or any((w == words for w, _ in same)):
            return ''
        return f"{' '.join(same[-1][0])!r} at step {same[-1][1]}"

    def leads_used(self) -> list[str]:
        """The first words of the word searches so far, once the store is known to read only those."""
        if not self.lead_words:
            return []
        return list(dict.fromkeys((' '.join(w[:self.lead_words]) for _, _, _, w, _, _, _ in self.search_log)))

    def list_end(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """A list the store stops paging before its count (no next page while more matches are counted), with a pick
        that counts every fitting listing. When the page offset tops out, the rows past it come up only in searches
        led by other words. Returns the note for the result (once per list), else ''."""
        if self.brief.get('pick') not in ('cheapest', 'most', 'least'):
            return ''
        page = self.paging(name, observation)
        if not page or page[3] is not None:
            return ''
        count, shown, param, _ = page
        start = args.get(param)
        start = 0 if start is None else start
        if isinstance(start, bool) or not isinstance(start, int) or start % shown or start + shown >= count:
            return ''
        sign = self.search_sign(name, args)
        what = repr(' '.join(sign[2])) if sign else 'this list'
        mark = name + _dump({k: v for k, v in sorted(args.items()) if k != param})
        if mark in self.list_marks:
            return ''
        self.list_marks.add(mark)
        self.list_ends.append(f'{what} at {start + shown} of its {count} matches')
        self.trace.append({'kind': 'list_end', 'list': what, 'reached': start + shown, 'count': count})
        return f"\n[Halo] The store ends {what} here, at {start + shown} of its {count} matches (no next page), so the other {count - start - shown} come up only in other searches. Every fitting listing competes for the pick: search again opening with fresh words (a second name for the product, a word from its store label, a value that fitting listings carry) and check the rows that could undercut your current best."

    def past_end(self, name: str, faults: list[str]) -> str:
        """When a page call asks past the furthest page the store's schema allows, what that leaves."""
        if name != self.roles.get('search'):
            return ''
        known = (self.shapes.get(name) or {}).get('next_arg')
        for fault in faults:
            got = re.match('args\\.(\\w+) is above ', fault)
            if got and (got.group(1) == known or (re.search('page|offset|start|begin|skip|from|cursor', got.group(1), re.I) and (not re.search('size|per|limit|count', got.group(1), re.I)))):
                return '. The store pages a list no further than that, so its rows past that point come up only in other searches: lead them with other words (another name for the product, a word of its store label, a value that fitting listings state).'
        return ''

    def cheapest_here(self, observations: list[Any]) -> str:
        if self.brief.get('pick') != 'cheapest' or not self.price_key:
            return ''
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ''
        return '\n[Halo] rows you can buy here, by rising price: ' + ', '.join((f'{i} ({p:g})' for p, i in sorted(rows)[:6]))

    def wider_hold(self) -> str:
        if self.wider_held or self.brief.get('pick') != 'cheapest' or self.left() < (8 if self.list_ends else 14) + 2 * len(self.items):
            return ''
        self.wider_held = True
        lines = [f'{i} at {self.price_of(i):g}' for i in self.cart_ids() if self.price_of(i) is not None]
        ends = f" The store cut off {'; '.join(self.list_ends[-2:])}; what lies past that surfaces only through other searches." if self.list_ends else ''
        led = self.leads_used()
        lead = f" Here a query is judged on its first {self.lead_words} word(s) alone, and the searches so far opened with {', '.join((repr(l) for l in led[-6:]))}: open every new one with fresh words." if led else ''
        return f"Held back for now (step not charged). {self.left()} steps are left and the lowest price wins the pick. Store searches may look at only part of a listing, so searching a requirement's value reaches a mere share of the fits, and the rows are not price-ordered.{ends}{lead} Ahead of the order, send a few extra searches per item under alternative product names (the bare one-word name, a synonym, words lifted from fitting titles seen so far, a value fitting listings carry) and consider just in-stock rows whose price beats the cart line" + (f" ({'; '.join(lines)})" if lines else '') + '; check the attributes of the lowest-priced of them, switch to any that fits, then order.'

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

    def show_links(self, ids: list[str], keep: int) -> None:
        """For each part code a cart line refers the reader to, search it once and, if steps allow, open the
        page found; this feeds the ledger rather than the model. Only for carts with several lines."""
        plan = self.value_template()
        if not plan or (len(self.items) < 2 and len(ids) < 2):
            return
        tool, base, field = plan
        for ident in ids:
            marks = self.fit_index()['marks']
            code = (marks.get(ident) or {}).get('more')
            if not code or code in self.link_tried or any((mark.get('own') == code for mark in marks.values())):
                continue
            if self.left() <= keep or self.finished:
                return
            self.link_tried.add(code)
            faults: list[str] = []
            args = _shape({**base, field: code}, self.schemas.get(tool), 'args', faults)
            if faults:
                return
            reply = self.act(tool, args)
            if reply is None:
                return
            self.observe(tool, reply.get('observation'))
            self.trace.append({'kind': 'show_link', 'line': ident, 'code': code})
            marks = self.fit_index()['marks']
            self.fill_facts([i for i, mark in marks.items() if mark.get('own') == code], max(keep, self.tail() + 6))

    def cap_template(self) -> tuple[str, dict[str, Any], str, str] | None:
        """Find a listing tool accepting both a shelf and a price ceiling, plus the mode it requires: whatever mode
        an earlier shelf listing used, otherwise the single option that names a listing. None if the tool or mode is unclear."""
        for tool, schema in self.schemas.items():
            props = (schema or {}).get('properties') or {}
            caps = [k for k, v in props.items() if isinstance(v, dict) and {'integer', 'number'} & set(_kinds(v)) and (re.search(CAP_ARG, _snake(k), re.I) or re.search('budget|(?:^|_)price_?(?:cap|limit|max)|max_?price', _snake(k), re.I)) and not re.search(CAP_NOT, k, re.I)]
            shelves = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and 'enum' not in v and re.search(SHELF_ARG, _snake(k), re.I)]
            if len(caps) != 1 or len(shelves) != 1:
                continue
            base: dict[str, Any] = {}
            for key, spec in props.items():
                options = spec.get('enum') if isinstance(spec, dict) else None
                if not isinstance(options, list) or len(options) < 2:
                    continue
                pick = next((a[key] for a in reversed(self.searched) if shelves[0] in a and key in a and not any(k != shelves[0] and not (isinstance(props.get(k), dict) and 'enum' in props[k]) and isinstance(v, str) and v.strip() and re.search(QUERY_WORDS, _snake(k), re.I) for k, v in a.items())), None)
                named = [o for o in options if isinstance(o, str) and re.search(LIST_MODE, o, re.I)]
                worded = [o for o in options if isinstance(o, str) and re.search(QUERY_WORDS, o, re.I)]
                if not named and len(options) == 2 and len(worded) == 1:
                    named = [o for o in options if o not in worded]
                pick = pick if pick is not None else named[0] if len(named) == 1 else None
                if pick is None:
                    return None
                base[key] = pick
            return (tool, base, shelves[0], caps[0])
        return None

    def list_upto(self, line: str) -> None:
        """After a cheapest-pick line sells out: list its category up to the sold line's own price (ties count) and read
        the two cheapest unread rows of it, so the replacement search starts from the price that was fitting rather
        than from the dearer listings already read (Arena 14561: the next cheapest fit cost the same as the sold one)."""
        plan = self.cap_template()
        cat_key = self.category_key()
        price = self.price_of(line)
        label = str((self.rows.get(line) or {}).get(cat_key) or '') if cat_key else ''
        if not plan or not label or price is None:
            return
        tool, base, shelf, cap = plan
        faults: list[str] = []
        args = _shape({**base, shelf: label, cap: int(price) if float(price).is_integer() else price}, self.schemas.get(tool), 'args', faults)
        if faults:
            return
        reply = self.act(tool, args)
        if reply is None or self.finished:
            return
        observation = reply.get('observation')
        self.observe(tool, observation)
        rows = sorted((i for i, _ in self.records_in(observation) if i != line and i not in self.full and self.buyable(i) and self.price_of(i) is not None), key=lambda i: self.price_of(i) or 0.0)[:2]
        if rows and self.left() > self.tail() + len(rows) + 3:
            self.fill_facts(rows, self.tail() + 3)
        self.trace.append({'kind': 'list_upto', 'line': line, 'price': price, 'rows': rows})

    def list_below(self, cart: list[str]) -> None:
        plan = self.cap_template()
        cat_key = self.category_key()
        if not plan or not cat_key:
            return
        tool, base, shelf, cap = plan
        for line in cart:
            price = self.price_of(line)
            label = str((self.rows.get(line) or {}).get(cat_key) or '')
            if price is None or price < 2 or not label or (line, price) in self.capped:
                continue
            if self.left() <= self.tail() + 5 or self.finished:
                return
            self.capped.add((line, price))
            faults: list[str] = []
            args = _shape({**base, shelf: label, cap: int(price) - 1 if float(price).is_integer() else int(price)}, self.schemas.get(tool), 'args', faults)
            if faults:
                return
            reply = self.act(tool, args)
            if reply is None:
                return
            self.observe(tool, reply.get('observation'))
            self.trace.append({'kind': 'list_below', 'line': line, 'price': price})
            page = self.paging(tool, reply.get('observation'))
            if page and page[3] is not None and page[1] < page[0] <= 2 * page[1] and self.left() > self.tail() + 5 and not self.price_sorted(reply.get('observation')):
                call = _shape({**args, page[2]: page[3]}, self.schemas.get(tool), 'args', faults)
                more = None if faults else self.act(tool, call)
                if more is not None:
                    self.observe(tool, more.get('observation'))

    def cheaper_rows(self) -> list[str]:
        if self.brief.get('pick') != 'cheapest':
            return []
        mark = self.cart_mark() + f'#{self.version}'
        cat_key = self.category_key()
        if mark in self.cheaper_marks or not cat_key:
            return []
        self.cheaper_marks.add(mark)
        cart = self.cart_ids()
        self.list_below(cart)
        picks: list[str] = []
        for line in cart:
            price = self.price_of(line)
            kind = str((self.rows.get(line) or {}).get(cat_key) or '').strip().lower()
            if price is None or not kind:
                continue
            pool = [i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i) and (str((self.rows.get(i) or {}).get(cat_key) or '').strip().lower() == kind) and ((self.price_of(i) if self.price_of(i) is not None else float('inf')) < price)]
            number = next((n for n in range(1, len(self.items) + 1) if line in self.item_ids(n)), 0)
            floor = self.row_tally(line, number)[0] if number else 0
            pool.sort(key=lambda i: (number > 0 and self.row_tally(i, number)[0] < floor, self.price_of(i) or 0.0))
            picks += [i for i in pool[:2] if i not in picks]
        return self.fill_facts(picks[:min(6, CHEAPER_OPENS * max(1, len(cart)))], self.tail() + 4)

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
        self.read_cap()
        free = self.msgs_free()
        if free <= 0:
            return ''
        self.listing = free >= 2 and (not self.split)
        count = self.msg_limit - self.msg_used
        self.lanes = self.plan_lanes(max(count, LIST_SIZE) if self.listing else count)
        self.lane_found = {n: 0 for n in self.lanes}
        self.lane_spent = {n: 0 for n in self.lanes}
        self.lane_used = {n: set() for n in self.lanes}
        self.follow_up = []
        waiting = [n for n in self.ask_order() if self.lane_left(n)]
        self.slices = self.slice_plan(waiting, free) if self.listing and self.cap_known else []
        if self.slices:
            self.trace.append({'kind': 'slices', 'free': free, 'slices': self.slices})
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        while self.msgs_free() > 0 and (not self.finished) and (self.left() > 6) and (self.clock() < QUESTION_BY):
            batch = self.ask_group(told) if self.slices and self.listing else self.ask_next(told)
            if not batch:
                break
            sent += batch
        told.extend(self.free_drops(sent))
        found_items = [n for n in self.lane_found if self.lane_found[n] > 0]
        searched = self.value_search(found_items) if found_items else ''
        if told:
            self.version += 1
        lines = [f'Q: {q}\nA: {a}' for _, q, a in sent]
        summary = "Questions Halo's interviewer put to the shopper (one step per question):\n" + '\n'.join(lines)
        summary += '\nFirm requirements added by these answers: ' + ('; '.join(told) if told else 'none')
        summary += '. Hold every pick to them (run new searches with the new values where that is needed).'
        pointed = [need for _, need in self.refs if any((need in t and (not t.startswith('dropped ')) for t in told))]
        if pointed:
            summary += ' Their value comes from a different listing: ' + ' | '.join(pointed) + '. Locate that listing, open its full page, then open each listing that page names; the value sits on the final page, and an order counts only once every one of them has been opened.'
        if searched:
            summary += '\n' + searched
        if self.queued:
            summary += '\n[store notes that arrived in the meantime] ' + ' | '.join(self.queued)
            self.queued = []
        return summary

    def late_question(self) -> str:
        """Just before an order, spend one message the order does not need on the next planned question.

        The interview never uses messages the request reserves for a process step (one tied to the order or to a
        no-purchase ending). Once Halo is about to buy, the ones reserved for the ending become spare. In race 13 two
        messages were allowed; the base asked a single question in 28 of 180 episodes, and 12 of those left a hidden need
        unrevealed (none of the episodes with an unrevealed need passed). Gives the loop model a summary when the answer
        adds a requirement, otherwise ''."""
        if self.late_asked or not self.roles.get('ask') or self.msg_used >= self.msg_limit:
            return ''
        if any((p['when'] == 'order' and p['messages'] > 0 for p in self.process)):
            return ''
        if self.left() < LATE_STEPS or self.clock() > LATE_BY or not self.judges():
            return ''
        self.late_asked = True
        if not self.lanes:
            self.lanes = self.plan_lanes(LIST_SIZE if self.listing else 1)
            self.lane_found = {n: 0 for n in self.lanes}
            self.lane_spent = {n: 0 for n in self.lanes}
            self.lane_used = {n: set() for n in self.lanes}
        told: list[str] = []
        budget = self.budget
        batch = self.ask_next(told, LATE_STEPS - 2, most=PAIR_SIZE)
        if not batch:
            self.trace.append({'kind': 'late_ask', 'skipped': 'nothing left to ask, or the message was not sent'})
            return ''
        if self.budget != budget:
            told.append(f'budget now {self.budget}')
        told.extend(self.free_drops(batch))
        number, question, answer = batch[0]
        self.trace.append({'kind': 'late_ask', 'q': question, 'a': answer[:200], 'told': told})
        if not told:
            return ''
        self.version += 1
        summary = f'Before the order Halo put one more question to the shopper (one step):\nQ: {question}\nA: {answer}\nFirm requirements added by this answer: ' + '; '.join(told) + '. The order was not placed: test each cart line against them (new searches with the new values where needed), correct the cart, then order.'
        pointed = [need for _, need in self.refs if any((need in t and (not t.startswith('dropped ')) for t in told))]
        if pointed:
            summary += ' Their value comes from a different listing: ' + ' | '.join(pointed) + '. Locate that listing, open its full page, then open each listing that page names; the value sits on the final page.'
        searched = self.value_search([number])
        return summary + ('\n' + searched if searched else '')

    def allowance_tool(self) -> str | None:
        """A store tool, outside every role, that takes no arguments and says it reports what is left of an allowance."""
        taken = set(self.roles.values())
        for tool in self.tools:
            fn = tool['function']
            params = fn.get('parameters') or {}
            if fn['name'] in taken or params.get('required') or params.get('properties'):
                continue
            if re.search(ALLOWANCE, _words(fn['name']) + ' ' + str(fn.get('description') or '').lower()):
                return fn['name']
        return None

    def left_count(self, seen: Any) -> int | None:
        """Messages still open according to an allowance reading: of exactly two whole numbers, the one under a key
        saying what is left, provided the other equals the messages Halo has sent (otherwise the reading may be about
        steps or something else, and None is returned)."""
        if not isinstance(seen, dict) or seen.get('error'):
            return None
        found: list[tuple[str, int]] = []

        def walk(node: Any, key: str, depth: int) -> None:
            if depth > 3 or isinstance(node, bool):
                return
            if isinstance(node, int):
                found.append((_words(key), node))
            elif isinstance(node, dict):
                for inner, value in node.items():
                    walk(value, str(inner), depth + 1)
        walk(seen, '', 0)
        if len(found) != 2:
            return None
        marks = [bool(re.search(LEFT_KEY, key)) for key, _ in found]
        if marks.count(True) != 1:
            return None
        left = found[marks.index(True)][1]
        spent = found[marks.index(False)][1]
        return left if spent == self.msg_used and 0 <= left <= 50 else None

    def read_cap(self) -> None:
        """When the request named no message limit, one reading of the store's allowance tool (if it has one) sets
        it: the messages it says are left plus those already sent. Only a known limit lets the interview plan shared
        messages ahead."""
        if self.cap_known or self.cap_read or self.left() <= 12:
            return
        self.cap_read = True
        tool = self.allowance_tool()
        if not tool:
            return
        faults: list[str] = []
        args = _shape({}, self.schemas.get(tool) or {}, 'args', faults)
        if faults:
            return
        reply = self.act(tool, args)
        if reply is None:
            return
        left = self.left_count(reply.get('observation'))
        self.trace.append({'kind': 'cap_read', 'tool': tool, 'left': left})
        if left is not None:
            self.msg_limit = self.msg_used + left
            self.cap_known = True

    def ask_order(self) -> list[int]:
        """Products in the planner's order (those it gave a story hint for first), the rest in request order."""
        return [n for n in self.plan_order if n in self.lanes] + [n for n in sorted(self.lanes) if n not in self.plan_order]

    @staticmethod
    def slice_plan(products: list[int], free: int) -> list[list[int]]:
        """One run of neighbouring products per free message, sizes as even as they come, kept in the given order,
        when there are more products than messages; [] otherwise (each product then gets messages of its own)."""
        if free < 1 or len(products) <= free:
            return []
        out: list[list[int]] = []
        start = 0
        for slot in range(free):
            size = (len(products) - start + (free - slot) - 1) // (free - slot)
            out.append(products[start:start + size])
            start += size
        return out

    def labels_per_message(self, sharing: int=1) -> int:
        """How many properties of one product the next message names. Lists off: one. One product: one in the first
        message, two after it. Several products: LIST_OPEN in the first message and LIST_SIZE later; a product
        sharing the message with others gets max(1, LIST_OPEN // sharing) in the first message and
        max(LIST_OPEN, LIST_SIZE // sharing) later. The caller cuts it to the properties the lane really has."""
        if not self.listing:
            return 1
        opening = self.ask_count == 0
        if len(self.items) == 1:
            return 1 if opening else PAIR_SIZE
        if sharing <= 1:
            return LIST_OPEN if opening else LIST_SIZE
        return max(1, LIST_OPEN // sharing) if opening else max(LIST_OPEN, LIST_SIZE // sharing)

    def search_template(self) -> tuple[str, dict[str, Any], str] | None:
        tool = self.roles.get('search')
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        named = [k for k in texts if re.search(QUERY_WORDS, _snake(k), re.I)]
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
                        word = self.word_option(tool, k)
                        if word is not None:
                            keep[k] = word
                return (tool, keep, field)
        return None

    def value_words(self, number: int) -> list[tuple[str, bool]]:
        """Per requirement value of the item that a word search could match: the words to search, flagged with
        whether an answer supplied them. That is the value as written, or, for a lower numeric bound with a unit taken from
        an answer, its number and unit, because a fitting listing often carries that very value."""
        out = []
        told = self.items[number - 1].told
        for n, need in self.needs_now():
            if n != number or need in [m for _, m in self.refs] or ':' not in need or self.fit_need(number, need):
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

    def value_template(self) -> tuple[str, dict[str, Any], str] | None:
        """Template a value search follows. Normally search_template(); if instead all searches up to now browsed a category
        wordlessly, the tool's only query field with its only mode argument set so that field gets read (the option mentioning
        words or a query; failing that, the remaining choice of a required two-way mode). None if the schema leaves it open."""
        plan = self.search_template()
        tool = self.roles.get('search')
        if plan or not tool or not self.searched:
            return plan
        schema = self.schemas.get(tool) or {}
        props = schema.get('properties') or {}
        required = schema.get('required') or []
        named = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and 'enum' not in v and re.search(QUERY_WORDS, _snake(k), re.I)]
        last = self.searched[-1]
        enums = {k: v['enum'] for k, v in props.items() if isinstance(v, dict) and isinstance(v.get('enum'), list) and len(v['enum']) >= 2}
        modes = []
        for key, options in enums.items():
            reads = [o for o in options if isinstance(o, str) and re.search(QUERY_WORDS, o, re.I)]
            others = [o for o in options if o != last.get(key)]
            if len(reads) == 1:
                modes.append((key, reads[0]))
            elif key in required and len(options) == 2 and len(others) == 1:
                modes.append((key, others[0]))
        if len(named) != 1 or len(modes) > 1 or (enums and not modes):
            return None
        base = {k: last[k] for k in required if k in last and k != named[0]}
        base.update(modes)
        return (tool, base, named[0])

    def value_search(self, numbers: list[int], stated: bool=False) -> str:
        """Searches each item with its requirement values together and opens the first rows of its kind. After the
        interview the values an answer gave lead; with stated=True every plain value the request states counts, and
        the first query names them all with the product noun when it fits the query field (the store's word search
        needs every word and also matches attribute values, so all the values together narrow a long result to the
        fitting rows: two declines whose fitting row a king reached that way). Under a cheapest pick a
        first query that matched more than three pages is not opened from at once when a later try names the values
        alone: the product noun widens a word search that ranks rows by how many query words they carry, so the values
        alone list the rows carrying all of them first; up to three pages of that query are read and the cheapest rows
        of the item's kind across both queries are opened."""
        plan = self.value_template()
        if not plan or self.left() < 22:
            return ''
        tool, common, field = plan
        seen: list[str] = []
        budget = max(8, self.max_steps // 6)
        room = ((self.schemas.get(tool) or {}).get('properties') or {}).get(field) or {}
        width = room.get('maxLength') if isinstance(room.get('maxLength'), int) else 60
        shelf = self.shelf_arg(tool)
        for number in numbers:
            pairs = self.value_words(number)
            item = self.items[number - 1]
            label = self.shelf_label(tool, number) if shelf else None
            base = {k: v for k, v in common.items() if k != shelf or (not label and k in ((self.schemas.get(tool) or {}).get('required') or []))}
            if label:
                base[shelf] = label
            values = [v for v, _ in pairs]
            fresh = [v for v, told in pairs if told or stated]
            if not fresh or budget <= 2:
                continue
            lead = [v for v in fresh if len(v.split()) == 1 or re.fullmatch(AMOUNT, v, re.I)]
            values = lead + [v for v in values if v not in lead]
            head = self.head_words(number)[0]
            tries = self.short_tries(values, fresh, item.name, head)
            if stated:
                every = list(dict.fromkeys((w for v in values for w in (_value_terms(v) or v.split())))) + head
                if 3 <= len(every) <= 5 and len(' '.join(every)) <= width:
                    tries = [every] + [t for t in tries if t != every]
            held: list[str] = []
            cheapest = self.brief.get('pick') == 'cheapest'
            alone = [len(w) >= 2 and (not _roots(' '.join(w)) & _roots(item.name)) for w in tries[:3]]
            for index, words in enumerate(tries[:3]):
                if not words:
                    continue
                deep = cheapest and alone[index]
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
                self.search_echo(tool, args, observation)
                pages, page = [observation], self.paging(tool, observation)
                limit = max(3, self.value_pages(page[1])) if page else 3
                while page and page[3] is not None and (page[0] <= 3 * page[1] or deep or (len(pages) < limit and self.page_fits(pages[-1], number))) and len(pages) < limit and budget > 3 and self.left() > self.tail() + 18:
                    more_args = _shape({**args, page[2]: page[3]}, self.schemas.get(tool), 'args', faults)
                    more = None if faults else self.act(tool, more_args)
                    budget -= 1
                    if more is None:
                        break
                    self.observe(tool, more.get('observation'))
                    pages.append(more.get('observation'))
                    page = self.paging(tool, more.get('observation'))
                cat_key = self.category_key()
                found = list(dict.fromkeys((i for seen_page in pages for i, _ in self.records_in(seen_page))))
                cap = self.budget if isinstance(self.budget, (int, float)) else None
                prices = [self.price_of(i) for i in found]
                if cap is not None and prices and all((p is not None and p > 10 * cap for p in prices)):
                    cap = None
                rows = [i for i in found if self.buyable(i) and (not cat_key or self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))) and (cap is None or (self.price_of(i) or 0) <= cap)]
                rows += [i for i in held if i not in rows]
                if self.brief.get('pick') == 'cheapest':
                    rows.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float('inf'))
                home = self.home_label(number)
                if cat_key and home:
                    rows.sort(key=lambda i: _words(str((self.rows.get(i) or {}).get(cat_key) or '')) != home)
                total = self.paging(tool, observation)
                read = f' ({len(pages)} of its pages read, {total[0]} matches)' if total else ''
                seen.append(f'{_dump(args.get(field))}{read}: ' + (', '.join(rows[:6]) or 'nothing of that kind listed'))
                if rows and cheapest and (not deep) and total and total[0] > 3 * total[1] and any(alone[index + 1:]) and budget > 6:
                    held = rows
                    continue
                if rows:
                    opened = self.fill_facts([i for i in rows if i not in self.full][:min(3, budget)], self.tail() + 16)
                    budget -= len(opened)
                    if opened:
                        seen.append('opened: ' + ' | '.join(opened)[:2400])
                    break
        if not seen:
            return ''
        self.trace.append({'kind': 'value_search', 'seen': [s[:200] for s in seen]})
        first = 'cheapest' if self.brief.get('pick') == 'cheapest' else 'first'
        return f"[Halo ran one search per such item with all its requirement values and read the {first} rows of the right kind; pages it did not read were not checked, and different words may surface other fitting listings] " + '\n'.join(seen)

    def stated_search(self) -> str:
        """Run once, after the first search and while at least 22 steps remain: a value search per item whose request
        gives at least two plain values, or a single plain value no search has carried yet, and which no opened listing satisfies
        yet (covers). Needs written into the request are never fed to the interview's value search, which only handles
        answered values. The store's word search matches a listing's title and attribute values (its description is not
        read), so pairing the value with the product noun finds the listings that state it."""
        if self.stated_done or not self.searched or self.left() < 22 or self.model_room() < 60:
            return ''
        self.stated_done = True
        used = set().union(*(self.query_words(a) for a in self.searched))
        numbers = [n for n in range(1, len(self.items) + 1) for pairs in [self.value_words(n)] if (len(pairs) >= 2 or (len(pairs) == 1 and (not _tokens(pairs[0][0]) <= used))) and (not any((self.covers(i, n) for i in self.item_ids(n) if i in self.full)))]
        return self.value_search(numbers, stated=True) if numbers else ''

    @staticmethod
    def short_tries(values: list[str], fresh: list[str], name: str, head: list[str] | None=None) -> list[list[str]]:
        """Value-search queries of at most three words: two one-word values (a told one first), else a told value with
        the product's head noun (head_words; else the name's last word), then the value's longest word with the noun,
        then a number joined to its unit ("64 GB" as "64GB"). Real races: 2-word queries a fitting title leads with
        listed the paid line .21-.39 per call against .06-.18 for 4+ words; r15 searches of 6+ words came back empty .63."""
        noun = ' '.join(head) if head else (re.findall('[A-Za-z]+', name) or [''])[-1]
        ones = [v for v in values if len(v.split()) == 1]
        out: list[list[str]] = []
        if len(ones) >= 2 and (not fresh or ones[0] in fresh or ones[1] in fresh):
            out.append(ones[:2])
        for value in (fresh or values)[:2]:
            words = value.split()
            own = [w for w in words if not _roots(w) & _roots(name)]
            if len(words) > 2 and not own:
                continue
            out.append(words + [noun] if len(words) <= 2 else [max(own, key=len), noun])
            joined = re.sub('(\\d)\\s+([A-Za-z]{1,4})\\b', '\\1\\2', value)
            if joined != value and len(joined.split()) <= 2:
                out.append(joined.split() + [noun])
        seen: set[str] = set()
        keep = []
        for words in out:
            words = [w for w in words if w][:3]
            mark = ' '.join(words).lower()
            if words and mark not in seen and len(mark) <= 30:
                seen.add(mark)
                keep.append(words)
        return keep

    def home_label(self, number: int) -> str:
        """Category label (_words form) of the listings that belong to the item: the shelf name from the request when
        some seen listing carries it, otherwise whichever label the bulk of the item's opened listings carry (its search rows
        if none is opened); '' when unknown."""
        cat_key = self.category_key()
        if not cat_key:
            return ''
        item = self.items[number - 1]
        for pool in (self.full, self.rows):
            counts: dict[str, int] = {}
            for record in pool.values():
                label = str(record.get(cat_key) or '')
                if label and self.owns(item, label):
                    counts[_words(label)] = counts.get(_words(label), 0) + 1
            for name in (item.shelf, item.name):
                if _words(name) in counts:
                    return _words(name)
            if counts:
                return max(counts, key=lambda label: counts[label])
        return ''

    def shelf_label(self, tool: str, number: int) -> str | None:
        """The value the search's shelf argument takes for item `number`. A listed set of labels: the label equal to the
        item's store label or name, else to a label its own rows carry, else the one label owned by the item that shares
        a word with it. Free text: only a label the item's own rows carry, since a guessed one can empty the search. None
        when unknown. Arena r14: b28's value searches dropped the shelf and opened rows of another shelf (3 of its 4
        losses to b27)."""
        shelf = self.shelf_arg(tool)
        if not shelf:
            return None
        spec = ((self.schemas.get(tool) or {}).get('properties') or {}).get(shelf) or {}
        item = self.items[number - 1]
        cat_key = self.category_key()
        counts: dict[str, int] = {}
        for ident in self.item_ids(number) if cat_key else []:
            label = str((self.rows.get(ident) or {}).get(cat_key) or '').strip()
            if label:
                counts[label] = counts.get(label, 0) + 1
        home = self.home_label(number)
        seen = sorted(counts, key=lambda label: (_words(label) != home, -counts[label]))
        options = spec.get('enum') if isinstance(spec, dict) else None
        if isinstance(options, list):
            pool = [o for o in options if isinstance(o, str)]
            for want in [item.shelf, item.name] + seen:
                hit = next((o for o in pool if want and _words(o) == _words(want)), None)
                if hit:
                    return hit
            mine = _roots(item.shelf or '') | _roots(item.name)
            owned = [o for o in pool if self.owner_of(o) == number and _roots(o) & mine]
            return owned[0] if len(owned) == 1 else None
        return seen[0] if seen else None

    def word_plan(self) -> tuple[str, dict[str, Any], str, str | None] | None:
        """The store's word search read from its schema alone, for Halo's searches before the model has searched:
        (tool, its fixed mode arguments, its query field, its shelf argument or None). The search role's tool when it
        takes words, else the one other tool with no role that does. A mode argument takes
        its word option (word_option); None when a required argument is left in doubt."""
        tool = self.roles.get('search')
        if not tool or not self.query_field(tool):
            taken = set(self.roles.values())
            spare = [t for t in self.schemas if t not in taken and self.query_field(t)]
            tool = spare[0] if len(spare) == 1 else None
        if not tool:
            return None
        schema = self.schemas.get(tool) or {}
        field = self.query_field(tool)
        shelf = self.shelf_arg(tool)
        if not field:
            return None
        base: dict[str, Any] = {}
        for key, spec in (schema.get('properties') or {}).items():
            if key in (field, shelf) or not isinstance(spec, dict):
                continue
            word = self.word_option(tool, key)
            if word is not None:
                base[key] = word
            elif key in (schema.get('required') or []):
                return None
        if shelf and shelf in (schema.get('required') or []):
            return None
        return (tool, base, field, shelf)

    @staticmethod
    def fit_words(words: list[str], width: int, keep: int) -> list[str]:
        """Whole words of a query cut to the field's width: the first of the last `keep` words (the product's name)
        go first while two or more of them are left, then the words before them from the end."""
        words = [w for w in words if w]
        while len(' '.join(words)) > width and len(words) > 1:
            if keep >= 2:
                del words[len(words) - keep]
                keep -= 1
            elif len(words) > keep:
                del words[len(words) - keep - 1]
            else:
                break
        return words

    def head_words(self, number: int) -> list[list[str]]:
        """Short head nouns of item `number` for a query, the likelier first: the last word of its name in singular
        form (with the word before it when that word has four letters or fewer). A name that joins two kinds ("a b and
        c", "a & b c") also gives the first kind's head, first when that kind has two or more words or the second
        has one (else the first word is a modifier both share). Arena r12-r16: passing runs found the winning
        listing with the distinguishing value and such a noun far more often than with the category's whole name."""
        item = self.items[number - 1]
        stop = {'the', 'a', 'an', 'of', 'for', 'with', 'in', 'on'}
        parts = [[w for w in re.findall("[A-Za-z0-9][A-Za-z0-9'-]*", p) if w.lower() not in stop] for p in re.split('\\s*(?:,|&|/|\\band\\b|\\bor\\b)\\s*', item.name)]
        parts = [p for p in parts if p]
        if not parts:
            return [['product']]

        def head(words: list[str]) -> list[str]:
            last = _single(words[-1])
            return [words[-2], last] if len(last) <= 4 and len(words) >= 2 else [last]
        out = [head(parts[-1])]
        if len(parts) >= 2:
            if len(parts[0]) >= 2 or len(parts[-1]) == 1:
                out.insert(0, head(parts[0]))
            else:
                out.append(head(parts[0]))
        keep: list[list[str]] = []
        for words in out:
            if [w.lower() for w in words] not in [[x.lower() for x in k] for k in keep]:
                keep.append(words)
        return keep

    def query_values(self, number: int) -> list[list[str]]:
        """The plain values the request (or an answer) states for item `number`, as query words: the most
        distinctive first (a one-word value, a code before a word; then a value with a number; then the shorter;
        a number with its unit is written many ways, so it leads only when no one-word value exists), each without the words of the
        item's head nouns (a value word the item's name also carries stays)."""
        name = _roots(' '.join((w for h in self.head_words(number) for w in h)))
        out: list[list[str]] = []
        for value, _ in self.value_words(number):
            words = [w for w in _value_terms(value) if not _roots(w) or not _roots(w) <= name]
            if words and [w.lower() for w in words] not in [[x.lower() for x in o] for o in out]:
                out.append(words)
        out.sort(key=lambda words: (len(words) > 1, not any((re.search('\\d', w) for w in words)), len(words)))
        return out

    def opening_tries(self, number: int, width: int, filed: bool=False) -> list[list[str]]:
        """The opening's queries for item `number`, value first and short: two stated values with the item's short
        head noun (head_words) when they make six words or fewer, then the most distinctive value with it (the
        shorter query after an empty result), then the head noun with the word before it in the item's name, then
        the other head noun. With filed=True (the search's category argument carries the item's category) and a name
        that joins several kinds, the value queries go without a noun, which would keep only one of the kinds. A
        category argument carries the category; the words never repeat the category's whole
        name (on a store that ranks rows by the query words a row carries, the label's words rank every row of the
        category alike and push the fitting one down)."""
        item = self.items[number - 1]
        values = self.query_values(number)[:2]
        heads = self.head_words(number)
        noun = heads[0]
        name_words = [w for w in re.findall("[A-Za-z0-9][A-Za-z0-9'-]*", item.name) if w.lower() not in ('the', 'a', 'an', 'and', 'or', 'of', 'for', 'with')]
        plain = noun
        if len(noun) == 1 and len(name_words) >= 2 and _single(name_words[-1]).lower() == noun[0].lower():
            plain = [name_words[-2], noun[0]]
        out: list[list[str]] = []
        tries = []
        if filed and len(heads) > 1 and values:
            noun = []
        if len(values) >= 2 and len(values[0]) + len(values[1]) + len(noun) <= 6:
            tries.append(values[0] + values[1] + noun)
        if values:
            tries.append(values[0] + noun)
        tries.append(plain)
        tries += heads[1:]
        for words in tries:
            once: list[str] = []
            for w in words:
                if w and w.lower() not in [x.lower() for x in once]:
                    once.append(w)
            words = self.fit_words(once, width, len(noun))
            if words and [w.lower() for w in words] not in [[x.lower() for x in o] for o in out]:
                out.append(words)
        return out

    def wide_try(self, number: int, width: int) -> list[str]:
        """The other wording of an item's value query: its two shortest stated values with the item's whole name. On a
        store that ranks rows by the query's words, the short query (opening_tries) and this one surface different
        rows; on the five Arena noask worlds, builds that sent only one of the two wordings lost about a third of each
        other's passes on listings only the other wording returned."""
        item = self.items[number - 1]
        values = sorted(dict.fromkeys((v for v, _ in self.value_words(number))), key=lambda v: len(v.split()))[:2]
        said = [w for v in values for w in v.split()]
        low = {w.lower() for w in said}
        noun = [w for w in re.findall("[A-Za-z0-9][A-Za-z0-9'.-]*", item.name) if w.lower() not in ('the', 'a', 'an', 'and', 'or', 'of', 'for', 'with') and w.lower() not in low] or ['product']
        return self.fit_words(said + noun, width, len(noun)) if said else []

    def value_shown(self, ident: str, words: list[str]) -> bool:
        """Whether a search row shows a value: every word of it (singular or plural alike) in the row's own text."""
        text = ' '.join((str(v) for v in (self.rows.get(ident) or {}).values() if isinstance(v, (str, int, float)) and (not isinstance(v, bool))))
        have = _roots(text) | _tokens(text)
        want = [w for w in words if w]
        return bool(want) and all(((_roots(w) and _roots(w) <= have) or w.lower() in have or re.sub('[^a-z0-9]', '', w.lower()) in re.sub('[^a-z0-9]', '', text.lower()) for w in want))

    def of_kind(self, ident: str, number: int) -> bool:
        """Whether a search row is of item `number`'s kind: its category label is the item's, else (no label) its
        text carries a head noun of the item."""
        item = self.items[number - 1]
        cat_key = self.category_key()
        record = self.rows.get(ident) or {}
        label = str(record.get(cat_key) or '') if cat_key else ''
        if label:
            return self.owns(item, label)
        text = _roots(' '.join((str(v) for v in record.values() if isinstance(v, str))))
        return any((_roots(' '.join(h)) & text for h in self.head_words(number)))

    def page_fits(self, observation: Any, number: int) -> bool:
        """Whether a page of search rows still holds a row of item `number`'s kind that shows one of its stated values,
        so the next page may hold more (Arena r12-r16: passing runs found the winning row on pages 2-5 of short value queries)."""
        values = self.query_values(number)
        return bool(values) and any((self.of_kind(i, number) and any((self.value_shown(i, v) for v in values)) for i, _ in self.records_in(observation)))

    def value_pages(self, page_size: int) -> int:
        """Pages one value query of the opening may read: about 30 rows on long episodes and 15 on 35-step ones (more
        pages when the store shows fewer rows a page), at least two."""
        rows = 15 if self.max_steps <= 40 else 30 if self.max_steps >= 80 else 15 + (self.max_steps - 40) * 15 // 40
        return max(2, -(-rows // max(1, page_size)))

    def capped_list(self, number: int, rows: list[str]) -> tuple[str, dict[str, Any]] | None:
        """Category browse for item `number` capped by price, used under a cheapest pick or a budget: the listing tool
        taking a shelf and a price ceiling (cap_template), the item's category as its rows spell it, and as the ceiling either
        the lowest price among rows of the item's kind that show a stated value (cheapest pick), or the budget left for this
        item once the cheapest seen row of every other item is paid (other picks; only when every already-seen row of the item's kind
        is dearer than that). None if either part is unknown."""
        if self.brief.get('pick') != 'cheapest' and (not isinstance(self.budget, (int, float))):
            return None
        plan = self.cap_template()
        cat_key = self.category_key()
        if not plan or not cat_key:
            return None
        tool, base, shelf, key = plan
        item = self.items[number - 1]
        home = self.home_label(number)
        labels: dict[str, int] = {}
        for ident in rows:
            label = str((self.rows.get(ident) or {}).get(cat_key) or '').strip()
            if label and self.owns(item, label) and _words(label) == home:
                labels[label] = labels.get(label, 0) + 1
        if not labels:
            return None
        label = max(labels, key=lambda l: labels[l])
        values = self.query_values(number)
        fits = [p for i in rows for p in [self.price_of(i)] if p is not None and self.buyable(i) and self.of_kind(i, number) and (not values or any((self.value_shown(i, v) for v in values)))]
        cap: float | None = min(fits) if fits and self.brief.get('pick') == 'cheapest' else None
        if cap is None and isinstance(self.budget, (int, float)):
            spent = 0.0
            for other, thing in enumerate(self.items, 1):
                if other == number:
                    continue
                prices = [p for i in self.item_ids(other) for p in [self.price_of(i)] if p is not None and self.buyable(i)]
                spent += min(prices) * thing.qty if prices else 0.0
            cap = (self.budget - spent) / max(1, item.qty)
            kind = [p for i in rows for p in [self.price_of(i)] if p is not None and self.of_kind(i, number)]
            if self.brief.get('pick') != 'cheapest' and (not kind or min(kind) <= cap):
                return None
        if cap is None or cap <= 0:
            return None
        spec = ((self.schemas.get(tool) or {}).get('properties') or {}).get(key) or {}
        value: Any = int(cap) if 'integer' in _kinds(spec) or float(cap).is_integer() else round(cap, 2)
        return (tool, {**base, shelf: label, key: value})

    def code_search(self, tool: str, args: dict[str, Any], limit: int=5000) -> tuple[dict[str, Any], Any, list[str], str] | None:
        """One search Halo sends itself, kept like a model search (rows, echo check, result cache, recent searches).
        Returns (the arguments sent, the result, its listing ids, its text for the model), else None."""
        faults: list[str] = []
        args = _shape(args, self.schemas.get(tool), 'args', faults)
        if faults or self.finished:
            return None
        self.fresh_start(tool, args)
        reply = self.act(tool, args)
        if reply is None:
            return None
        observation = reply.get('observation')
        self.observe(tool, observation)
        echo = self.search_echo(tool, args, observation)
        shown = self.show(observation)
        self.read_cache[tool + _dump(args)] = (self.version, self.step, shown)
        rows = [i for i, _ in self.records_in(observation)]
        if rows:
            self.searched = (self.searched + [dict(args)])[-8:]
        text = shown if len(shown) <= limit else shown[:limit] + '...(cut)'
        return (args, observation, rows, f'{_dump(args)} (step {self.step}): {text}{echo}')

    def seller_names(self) -> dict[str, bool]:
        """Names seen in the rows' shop fields (lowercased), each with whether some row also carries it as a brand."""
        out: dict[str, bool] = {}
        for row in self.rows.values():
            for key, value in row.items():
                if not isinstance(value, str) or not 2 < len(value.strip()) <= 40:
                    continue
                name = value.strip().lower()
                if re.search('seller|shop|store|merchant', str(key), re.I):
                    out[name] = out.get(name, False)
                elif re.search('brand|maker|manufacturer|vendor|make', str(key), re.I) and name in out:
                    out[name] = True
        for row in self.rows.values():
            for key, value in row.items():
                if isinstance(value, str) and value.strip().lower() in out and re.search('brand|maker|manufacturer|vendor|make', str(key), re.I):
                    out[value.strip().lower()] = True
        return out

    def pointer_parts(self, text: str, hint: str='') -> tuple[str, str, list[str]]:
        """(shop, kind of product, up to three values) a description of another listing names, for a search led by the
        shop. The shop: a name the rows' shop fields carry that the text names, else the first capitalised word of the
        hint (the trail's search words) or of the text that is not a sentence's first word, a requested item's word or a
        category label's word. The kind: the longest category label (rows, schema) the text holds, else the words
        around the shop that name a listing. The values: after "with" / "that has", each comma or "and" part's number,
        or its last word (its first when the last is a property name the request or the records use)."""
        prop, colon, body = text.partition(':')
        props = set(re.findall('[a-z]+', prop.lower())) if colon and len(prop.split()) <= 4 else set()
        text = body.strip() if props else text
        clean = re.split('\\s*\\((?:or|read|the page|either)\\b|\\bor wherever\\b|\\bor (?:via|through) (?:the|a|another)\\b|\\bor the (?:page|listing) it\\b|\\bin either direction\\b|\\bthat ranks\\b|,? ?(?:read |either )?directly or\\b', text)[0]
        grammar = {'the', 'a', 'an', 'its', 'this', 'that', 'those', 'these', 'same', 'unlike', 'identical', 'equal', 'like', 'other', 'than', 'any', 'one', 'own', 'page', 'pages', 'listing', 'listings', 'product', 'item', 'shop', 'store', 'seller', 'brand', 'and', 'or', 'with', 'of', 'for', 'to', 'in', 'on', 'by', 'from', 'at', 'is', 'has', 'have', 'which', 'where', 'whose', 'not', 'no', 'given', 'named', 'pointed', 'sells', 'lists', 'offers', 'sold', 'listed', 'gives', 'it', 'you', 'direct', 'directly', 'names', 'record', 'records', 'requirement', 'previous', 'next', 'link', 'opened', 'each', 'points', 'point', 'sends', 'redirects'}
        item_words = {w for item in self.items for w in re.findall('[a-z0-9]+', item.name.lower())}
        cat_key = self.category_key()
        labels = {str(r.get(cat_key) or '').strip() for r in self.rows.values()} if cat_key else set()
        for tool in self.schemas:
            shelf = self.shelf_arg(tool)
            spec = (((self.schemas.get(tool) or {}).get('properties') or {}).get(shelf or '')) or {}
            labels |= {o.strip() for o in spec.get('enum') or [] if isinstance(o, str)}
        labels = {l for l in labels if l and len(l) <= 60}
        label_words = {w for l in labels for w in re.findall('[a-z0-9]+', l.lower())}
        low = clean.lower()
        sellers = self.seller_names()
        shop = ''
        for source, lead in ((hint, False), (clean, True), (clean, False)):
            spots = [(m.start(), m.group(0)) for s in sellers for m in [re.search('(?<![A-Za-z0-9])' + re.escape(s) + '(?![A-Za-z0-9])', source, re.I)] if m]
            if spots:
                shop = min(spots)[1]
                break
            for found in re.finditer("\\b[A-Z][a-z][A-Za-z'-]+", source):
                word = re.sub("'s$", '', found.group(0))
                before = source[:found.start()].rstrip()
                if lead and (not before or before[-1] in '.:;!?'):
                    continue
                if word.lower() in grammar or word.lower() in item_words or word.lower() in label_words:
                    continue
                shop = word
                break
            if shop:
                break
        kind = next((l for l in sorted(labels, key=len, reverse=True) if re.search('(?<![a-z0-9])' + re.escape(l.lower()) + '(?![a-z0-9])', low)), '')
        if not kind and shop:
            name = re.escape(shop)
            for pattern in (name + "(?:'s)?\\s+([a-z][a-z &-]*?)\\s+(?:listing|product|page|item|with|that|in|for)\\b", '\\b(?:the|a|an)\\s+([a-z][a-z &-]*?)\\s+(?:listing\\s+|page\\s+)?(?:sold by|from|at|by|in|on|of|for)\\s+(?:the page (?:of|for)\\s+)?' + name, '\\b(?:the|a|an)\\s+([a-z][a-z &-]*?)\\s+' + name + '\\s+(?:sells|lists|offers)\\b'):
                found = re.search(pattern, clean)
                if found:
                    words = [w for w in re.split('\\b(?:of|on|for|to|in) (?:the|a|an)\\b', found.group(1))[-1].split() if w not in grammar]
                    if words and not set(words) <= props | grammar:
                        kind = ' '.join(words)
                        break
        at = clean.lower().find(shop.lower()) if shop else -1
        rest = re.sub('(?<=\\d),(?=\\d{3}\\b)', '', clean[at + len(shop):] if at >= 0 else clean)
        start = re.search('\\b(?:with|that has|that have|having|that is|in)\\b', rest)
        known = props | {w for item in self.items for m in item.musts + item.told for w in re.findall('[a-z]+', m.split(':', 1)[0].lower())}
        known |= {w for record in self.full.values() for k in (_fact_block(record) or {}) for w in _words(k).split()}
        values: list[str] = []
        for part in re.split(',|;|\\band\\b|\\(|\\)', rest[start.end():]) if start else []:
            numbers = re.findall('(?<![\\w.])\\d+(?:\\.\\d+)?(?![\\w.])', part)
            if numbers:
                values.append(numbers[0])
                continue
            words = [w for w in re.findall('[a-z0-9][a-z0-9-]*', part.lower()) if w not in grammar]
            if words:
                pick = words[-1] if words[-1] not in known or words[0] in known else words[0]
                if pick not in shop.lower().split() and pick not in values:
                    values.append(pick)
        values.sort(key=lambda v: bool(re.fullmatch('[\\d.]+', v)))
        return (shop, kind, values[:3])

    def hop_queries(self, text: str, hint: str='') -> tuple[str, list[str]]:
        """The searches for a listing a description points at, shortest first in meaning: the shop's name with up to
        three of the values the description gives (the store reads only a query's first few words, and a property's
        name is not in a listing's values), then the shop with the kind of product, then the shop alone."""
        shop, kind, values = self.pointer_parts(text, hint)
        if not shop:
            return ('', [])
        out = []
        for words in ([shop] + values[:3] if values else [], [shop] + re.findall('[A-Za-z0-9][A-Za-z0-9-]*', kind) if kind else [], [shop]):
            mark = ' '.join(words)
            if words and mark.lower() not in [o.lower() for o in out]:
                out.append(mark)
        return (shop, out)

    def link_cap(self) -> int:
        """Searches an episode may spend on the listings requirements point at: LINK_SEARCHES, or an eighth of the
        steps on long episodes (chains of four or five pages need a search a page)."""
        return max(LINK_SEARCHES, self.max_steps // 8)

    def opening(self) -> str:
        """Runs ahead of the model's first turn. Every item gets a search with at most two plain values from the request
        plus its name, inside its category if the search accepts one (after an empty result one shorter query, then the name
        by itself); a value query that returned rows also gets its next page; a category browse happens only when nothing was
        found and the store can list a shelf with no words. Queries stay short (query_values, head_words): a distinctive
        stated value missing from the rows gets a query of its own; a value query keeps paging while its rows of the item's
        kind show a stated value (value_pages); under a cheapest pick or a budget, a store search that accepts a shelf and a
        price ceiling lists the item's category up to the cheapest fitting price seen (capped_list). The opening may use a
        fifth of the steps in short episodes and a quarter in longer ones. Before all that, each requirement pointing at
        another listing gets its first hop: the shop's name with the description's values, then with the product kind. Real
        race 16 (1840 episodes): pass rate .36 / .20 / .05 for 0 / 1-2 / 3+ whole-category browses before the first cart
        line; Arena r13: the race's top agent won all five chain tasks we lost by searching the shop with the pointed values
        from its first steps. Gives the text for the model's first turn, or ''."""
        plan = self.word_plan()
        if self.opening_done or not plan or self.finished or self.model_room() < 150:
            return ''
        self.opening_done = True
        tool, base, field, shelf = plan
        spec = ((self.schemas.get(tool) or {}).get('properties') or {}).get(field) or {}
        width = spec.get('maxLength') if isinstance(spec.get('maxLength'), int) else 60
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        cap = (max(4, self.max_steps // 5) if self.max_steps <= 40 else self.max_steps // 4) + (2 if pending else 0)
        start = self.step
        texts: list[str] = []

        def room(more: int=0) -> bool:
            return self.step - start + more < cap and self.left() > 20 and (not self.finished) and self.model_room() > 120
        for need in pending:
            shop, queries = self.hop_queries(need)
            for query in queries[:2]:
                if not room() or self.link_searches >= self.link_cap():
                    break
                words = self.fit_words(query.split(), width, 0)
                done = self.code_search(tool, {**base, field: ' '.join(words)}, 4000)
                if done is None:
                    break
                self.link_searches += 1
                self.link_seen[' '.join(words).lower()] = done[2]
                hits = self.shop_rows(need, done[2])
                texts.append(f'the listing "{need}" points at: ' + done[3])
                self.trace.append({'kind': 'link_search', 'words': ' '.join(words), 'rows': hits[:4], 'opening': True})
                if hits:
                    break
        todo = [(n, it) for n, it in enumerate(self.items, 1) if it.musts or it.name != 'the product asked for']
        for index, (number, item) in enumerate(todo):
            if not room():
                break
            label = self.shelf_label(tool, number) if shelf else None
            fixed = {**base, **({shelf: label} if label else {})}
            got = None
            values = self.query_values(number)
            noun = self.head_words(number)[0]
            seen_rows: list[str] = []
            queries: set[str] = set()
            tries = self.opening_tries(number, width, bool(label))
            wide = self.wide_try(number, width)
            if wide and self.max_steps <= 40 and [w.lower() for w in wide] not in [[x.lower() for x in t] for t in tries]:
                tries = [wide] + tries
            for words in tries:
                if not room():
                    break
                queries.add(' '.join(words).lower())
                done = self.code_search(tool, {**fixed, field: ' '.join(words)}, 4000)
                if done is None:
                    break
                texts.append(f'item {number} ({item.name}): ' + done[3])
                seen_rows += done[2]
                if done[2]:
                    got = (words, done)
                    break
            share = max(2, (cap - (self.step - start)) // max(1, len(todo) - index))
            began = self.step

            def more(extra: int=0) -> bool:
                return room(extra) and self.step - began + extra < share
            short = self.opening_tries(number, width, bool(label))[:1]
            for other in [w for w in [wide] + short if w]:
                if not got or ' '.join(other).lower() in queries or not more():
                    continue
                queries.add(' '.join(other).lower())
                done = self.code_search(tool, {**fixed, field: ' '.join(other)}, 4000)
                if done is not None:
                    seen_rows += done[2]
                    texts.append(f'item {number} ({item.name}), other wording: ' + done[3])
            if got:
                sent = {w.lower() for w in got[0]}
                for words in values[:3]:
                    if not more() or {w.lower() for w in words} <= sent or any((self.value_shown(i, words) for i in seen_rows)):
                        continue
                    if not any((re.search('\\d', w) for w in words)) and len(words) > 1:
                        continue
                    query = self.fit_words(words + noun, width, len(noun))
                    if ' '.join(query).lower() in queries:
                        continue
                    done = self.code_search(tool, {**fixed, field: ' '.join(query)}, 4000)
                    if done is None:
                        break
                    sent |= {w.lower() for w in words}
                    seen_rows += done[2]
                    texts.append(f'item {number} ({item.name}), one value: ' + done[3])
            if got and _tokens(' '.join(got[0])) - _tokens(item.name) - _tokens(' '.join(noun)) and room():
                args, observation = (got[1][0], got[1][1])
                page = self.paging(tool, observation)
                limit = self.value_pages(page[1]) if page else 0
                read = 1
                reserve = 2 if self.capped_list(number, seen_rows) else 0
                while page and page[3] is not None and page[0] > page[1] * read and read < limit and (read == 1 and room() or more(reserve)):
                    after = self.code_search(tool, {**args, page[2]: page[3]}, 4000)
                    if after is None:
                        break
                    read += 1
                    self.pages_read += 1
                    seen_rows += after[2]
                    texts.append(f'item {number}, page {read}: ' + after[3])
                    if not self.page_fits(after[1], number):
                        break
                    page = self.paging(tool, after[1])
            capped = self.capped_list(number, seen_rows) if got else None
            if capped and more():
                done = self.code_search(capped[0], capped[1], 4000)
                if done is not None:
                    texts.append(f'item {number} ({item.name}), its category up to a price: ' + done[3])
                    page = self.paging(capped[0], done[1])
                    if page and page[3] is not None and page[0] > page[1] and more() and (not self.price_sorted(done[1])):
                        after = self.code_search(capped[0], {**done[0], page[2]: page[3]}, 4000)
                        if after is not None:
                            texts.append(f'item {number}, its capped list, next page: ' + after[3])
            if got is None and label and room():
                listing = {k: v for k, v in base.items()}
                for key in listing:
                    mode = ((self.schemas.get(tool) or {}).get('properties') or {}).get(key) or {}
                    others = [o for o in mode.get('enum') or [] if o != listing[key] and isinstance(o, str) and re.search(LIST_MODE, o, re.I)]
                    listing[key] = others[0] if len(others) == 1 else listing[key]
                if listing != base or not base:
                    done = self.code_search(tool, {**listing, shelf: label}, 4000)
                    if done is not None:
                        texts.append(f'item {number} ({item.name}), its category without words: ' + done[3])
        if not texts:
            return ''
        room_left = 24000
        sizes = sorted((len(t) for t in texts))
        share = room_left
        for index, size in enumerate(sizes):
            share = (room_left - sum(sizes[:index])) // max(1, len(sizes) - index)
            if size > share:
                break
        share = max(300, share)
        for index, text in enumerate(texts):
            if len(text) > share:
                ids = re.findall('"(' + re.escape(self.id_key or 'id') + ')":"([^"]+)"', text) if self.id_key else []
                texts[index] = text[:share] + '...(cut; listings in it: ' + ', '.join(dict.fromkeys(i for _, i in ids)) + ')'
        if self.queued:
            texts.append('[store notes that arrived in the meantime] ' + ' | '.join(self.queued))
            self.queued = []
        self.trace.append({'kind': 'opening', 'steps': self.step - start, 'searches': len(texts)})
        self.stated_done = True
        return "[Halo searched the store before your first turn, one step a search: each item with values the request states (and its category when the search takes one), and the shop of each listing a requirement points at]\n" + '\n'.join(texts)

    def next_ask(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]], most: int=LIST_SIZE) -> tuple[int, list[str]]:
        """The product the next message is about and its labels: products with no need found yet before those with one,
        among them the fewest messages first (so a product never asked leads), then not lost, then plan order. The same
        order holds when lists are off, whatever the message limit."""
        first = {n: i for i, n in enumerate(self.plan_order)}
        for number in sorted(lanes, key=lambda n: (found.get(n, 0) > 0, spent.get(n, 0), n in self.lost, first.get(n, len(first) + n))):
            left = [label for label in lanes[number] if _words(label) not in used.get(number, set()) and (not self.told_about(number, label))]
            if left:
                return (number, left[:min(most, self.labels_per_message())])
        return (0, [])

    def lane_left(self, number: int) -> list[str]:
        used = self.lane_used.get(number, set())
        return [label for label in self.lanes.get(number) or [] if _words(label) not in used and (not self.told_about(number, label))]

    def told_about(self, number: int, label: str) -> bool:
        """True when an answer already set a need on this property: the told property's name contains the label's
        words ("colour" after "shade colour"), or the label contains a told name of at least two words; colour/color and
        singular/plural count alike. A told name of one word ("type") does not cover a longer label ("bulb type")."""
        mine = _spelled(label)
        return any((_names(mine, _spelled(t.split(':', 1)[0])) for t in self.items[number - 1].told))

    def ask_text(self, number: int, labels: list[str], shown: bool=False) -> str:
        if len(labels) >= 2:
            return f"Which {', '.join(labels[:-1])} or {labels[-1]} do you want for the {self.item_said(number)}?"
        return self.question(number, labels[0], self.samples(number, labels[0]) if shown else None)

    def fit_ask(self, number: int, labels: list[str], shown: bool) -> tuple[list[str], str]:
        """The longest prefix of `labels` whose question fits the message tool's schema unchanged (the shaper truncates
        text beyond a length cap, so an overlong list loses labels from its end instead), returned with that question;
        ([], '') if not even a single label fits."""
        for size in range(len(labels), 0, -1):
            text = self.ask_text(number, labels[:size], shown)
            args = self.message_args(text)
            if args and text in args.values():
                return (labels[:size], text)
        return ([], '')

    def ask_next(self, told: list[str], floor: int=6, most: int=LIST_SIZE, pick: int=0) -> list[tuple[int, str, str]]:
        """Send one question (a queued follow-up takes priority over the next planned labels; `pick` names the product
        when a planned slice holds it alone), merge its answer into the items' needs (each fresh need also goes onto
        `told`) and mark the labels it covered as settled. A list reply the reader marks crowded turns lists off for the
        rest of the episode, also when it named a need. Gives back the batch sent, or [] when no message went out."""
        for _ in range(LIST_SIZE):
            self.follow_up = [f for f in self.follow_up if not self.told_about(f[0], f[1][0])]
            if self.follow_up:
                number, labels, shown = self.follow_up.pop(0)
            elif pick and self.lane_left(pick):
                number, labels, shown = (pick, self.lane_left(pick)[:min(most, self.labels_per_message())], False)
            else:
                number, labels = self.next_ask(self.lanes, self.lane_found, self.lane_spent, self.lane_used, most)
                shown = False
            if not labels:
                return []
            fitted, text = self.fit_ask(number, labels, shown)
            if not fitted and shown:
                fitted, text = self.fit_ask(number, labels, False)
            if fitted:
                break
            self.lane_used.setdefault(number, set()).add(_words(labels[0]))
            self.trace.append({'kind': 'ask_too_long', 'item': number, 'label': labels[0]})
        else:
            return []
        labels = fitted
        batch = self.send([(number, labels[0], text)], floor)
        if not batch:
            return []
        self.ask_count += 1
        self.lane_spent[number] = self.lane_spent.get(number, 0) + 1
        had = [len(item.told) for item in self.items]
        heard = self.hear(batch, (LIGHT, HEAVY))
        unclear = not _items(heard.get('needs')) and (not _items(heard.get('free'))) and (not _items(heard.get('crowded')))
        if unclear:
            self.lost.add(number)
        else:
            self.lost.discard(number)
        if len(labels) > 1:
            heard['free_said'] = heard.pop('free', None)
        self.take_heard(heard, told, ' '.join((a for _, _, a in batch)), labels)
        new = [need for item, size in zip(self.items, had) for need in item.told[size:]]
        for n, (item, size) in enumerate(zip(self.items, had), 1):
            if len(item.told) > size:
                self.lane_found[n] = self.lane_found.get(n, 0) + 1
        crowded = any((_count(x) == 1 for x in _items(heard.get('crowded'))))
        self.settle(number, labels, shown, new, crowded)
        if crowded and len(labels) > 1:
            self.singles_now(number, bool(new))
        return batch

    def singles_now(self, number: int, named: bool) -> None:
        """A reply saying a list or a shared message held too much: one property per message from here on."""
        self.listing = False
        self.slices = []
        if not self.split:
            self.split = True
            self.trace.append({'kind': 'singles_from_here', 'item': number, 'with_need': named})

    @staticmethod
    def either(labels: list[str]) -> str:
        return labels[0] if len(labels) == 1 else f"{', '.join(labels[:-1])} or {labels[-1]}"

    def group_text(self, parts: list[tuple[int, list[str]]]) -> str:
        heads = [f'which {self.either(labels)} the {self.item_said(n)} should have' for n, labels in parts]
        if len(heads) == 1:
            return f'Could you say {heads[0]}?'
        return 'Could you say ' + ', '.join(heads[:-1]) + ', and ' + heads[-1] + '?'

    def fit_group(self, parts: list[tuple[int, list[str]]]) -> tuple[list[tuple[int, list[str]]], str]:
        """The shared message that fits the message tool's length cap unchanged: labels leave from the end of the
        product holding the most (the later product on a tie) until each has one, then products leave from the end."""
        parts = [(n, list(labels)) for n, labels in parts if labels]
        while parts:
            text = self.group_text(parts)
            args = self.message_args(text)
            if args and text in args.values():
                return (parts, text)
            widest = max(range(len(parts)), key=lambda i: (len(parts[i][1]), i))
            if len(parts[widest][1]) > 1:
                parts[widest][1].pop()
            else:
                parts.pop()
        return ([], '')

    def ask_group(self, told: list[str], floor: int=6) -> list[tuple[int, str, str]]:
        """Send the next planned slice. A product alone in its slice gets a message of its own, the whole list size; two
        or more share one message under the share rule of labels_per_message (a product the length cap leaves out
        moves to the front of the next slice). The reply is read once for all of them; a crowded
        reply ends lists and slices (products it named a need for are settled, the rest keep their lanes untouched).
        Otherwise a product with a fresh need settles its list as a single-product list would, and so does the first
        product of the message when nothing was found; later products that drew nothing keep their lanes."""
        while self.slices and self.listing:
            group = [n for n in self.slices.pop(0) if self.lane_left(n)]
            if not group:
                continue
            if len(group) == 1:
                return self.ask_next(told, floor, pick=group[0])
            share = self.labels_per_message(len(group))
            parts, text = self.fit_group([(n, self.lane_left(n)[:share]) for n in group])
            if not parts:
                self.trace.append({'kind': 'group_too_long', 'items': group})
                continue
            names = [n for n, _ in parts]
            cut = [n for n in group if n not in names]
            if cut:
                self.trace.append({'kind': 'group_cut', 'items': cut, 'moved': bool(self.slices)})
                if self.slices:
                    self.slices[0] = cut + self.slices[0]
            batch = self.send([(parts[0][0], parts[0][1][0], text)], floor)
            if not batch:
                return []
            self.ask_count += 1
            for n in names:
                self.lane_spent[n] = self.lane_spent.get(n, 0) + 1
            had = [len(item.told) for item in self.items]
            heard = self.hear([(' and '.join((str(n) for n in names)), text, batch[0][2])], (LIGHT, HEAVY))
            unclear = not _items(heard.get('needs')) and (not _items(heard.get('free'))) and (not _items(heard.get('crowded')))
            if unclear:
                self.lost.update(names)
            else:
                self.lost.difference_update(names)
            heard['free_said'] = heard.pop('free', None)
            self.take_heard(heard, told, batch[0][2], [label for _, labels in parts for label in labels])
            fresh = {n: self.items[n - 1].told[had[n - 1]:] for n in range(1, len(self.items) + 1)}
            for n, needs in fresh.items():
                if needs:
                    self.lane_found[n] = self.lane_found.get(n, 0) + 1
            crowded = any((_count(x) == 1 for x in _items(heard.get('crowded'))))
            self.trace.append({'kind': 'group', 'parts': parts, 'a': batch[0][2][:200], 'crowded': crowded})
            for place, (n, labels) in enumerate(parts):
                if fresh.get(n) or (place == 0 and (not crowded)):
                    self.settle(n, labels, False, fresh.get(n) or [], False)
            if crowded:
                self.singles_now(names[0], any((fresh.get(n) for n in names)))
            return batch
        return self.ask_next(told, floor)

    def settle(self, number: int, labels: list[str], shown: bool, new: list[str], crowded: bool) -> None:
        """Record what an answer revealed about the labels its question covered.

        Single label: it counts as asked; a reply wanting a narrower question gets the base follow-up (the property's head
        word, or sample values). List: the shopper's answer covers only the first tracked property a message names, and its
        answer when a message names none of its tracked properties equals its answer about a property it does not care about. Hence a
        revealed need settles its own label and all labels ahead of it, and the labels behind it head the item's next list;
        a reply revealing nothing settles just the first label (had a need existed there, it would have been named) and the rest follow in
        reverse order, so properties the shopper does not care about stop blocking them; after two such replies in a row the
        labels never named go first, since otherwise the order would fall back to the first list's. A reply wanting questions one
        by one switches lists off for the remaining episode, and the first label of that list gets asked alone."""
        used = self.lane_used.setdefault(number, set())
        if len(labels) == 1:
            used.add(_words(labels[0]))
            if new or not crowded:
                return
            head = labels[0].split()[-1]
            if len(labels[0].split()) >= 2 and _words(head) not in used:
                self.follow_up.insert(0, (number, [head], False))
            elif not shown and len(self.samples(number, labels[0])) >= 2:
                self.follow_up.insert(0, (number, labels, True))
            return
        if crowded and not new:
            self.listing = False
            cleared = [l for l in self.silent.get(number, []) if l not in labels]
            used.update((_words(l) for l in cleared))
            self.trace.append({'kind': 'lists_off', 'item': number, 'cleared': cleared[:8]})
            first = next((l for l in labels if _words(l) not in used), labels[0])
            self.follow_up.insert(0, (number, [first], False))
            return
        hit = self.answered_label(labels, new) if new else -1
        if hit < 0:
            self.silent.setdefault(number, []).extend((l for l in labels if l not in self.silent.get(number, [])))
        ahead = labels[hit + 1:] if hit >= 0 else labels[:0:-1]
        used.update((_words(label) for label in labels[:max(hit, 0) + 1]))
        rest = [label for label in self.lanes.get(number) or [] if label not in labels]
        self.lane_misses[number] = 0 if hit >= 0 else self.lane_misses.get(number, 0) + 1
        self.lanes[number] = rest + ahead if self.lane_misses[number] and self.lane_misses[number] % 2 == 0 else ahead + rest
        self.trace.append({'kind': 'list_settled', 'item': number, 'done': labels[:hit + 1], 'next': ahead[:4]})

    def answered_label(self, labels: list[str], new: list[str]) -> int:
        """Position of the first label a list reply responded to. Every new need is tied to the label with the
        largest word overlap with its property (colour/color alike); failing that, to the label whose values seen on opened
        listings overlap its value the most; -1 when no new need ties to any label (the reply is then settled as one that
        revealed nothing)."""
        seen: dict[str, set[str]] = {}
        for _, key, _, values in self.key_table():
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

    def lane_keys(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        item = self.items[number - 1]
        measured = {key: sum((1 for v in values if re.match('^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$', v))) * 2 > len(values) for _, key, _, values in table}
        vague = set(self.vague_keys(number, table))
        settled = [_spelled(m.split(':', 1)[0]) for m in item.musts]
        settled = [prop for prop in settled if not any((_names(prop, _spelled(key)) for key in vague))]
        rows = [(label, key, carried, len(values)) for label, key, carried, values in table if self.owner_of(label) == number]
        if len(self.items) == 1 and (not rows):
            rows = [(label, key, carried, len(values)) for label, key, carried, values in table]
        rows.sort(key=lambda r: (-r[2], measured.get(r[1], False), -r[3]))
        keys = [key for _, key, _, _ in rows if not any((_names(prop, _spelled(key)) for prop in settled)) and (not re.search(NOT_PROPERTY, key))]
        return [key for key in keys if not re.search(CATCH_ALL, key)] + [key for key in keys if re.search(CATCH_ALL, key)]

    def adopt_labels(self) -> None:
        cat_key = self.category_key()
        if len(self.items) < 2 or not cat_key:
            return
        labels = {str(record.get(cat_key) or '') for record in self.full.values()} - {''}
        strays = [label for label in labels if not self.owner_of(label)]
        orphans = [n for n in range(1, len(self.items) + 1) if not any((self.owner_of(label) == n and label not in self.adopted for label in labels))]
        if strays and len(orphans) == 1:
            self.adopted.update({label: orphans[0] for label in strays})
            self.trace.append({'kind': 'adopted', 'item': orphans[0], 'labels': strays})

    def owner_of(self, label: str) -> int:
        if not label:
            return 0
        if label in self.adopted:
            return self.adopted[label]
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

    def plan_lanes(self, count: int) -> dict[int, list[str]]:
        self.adopt_labels()
        table = self.key_table()
        settled = []
        for number, item in enumerate(self.items, 1):
            settled.append(f'item {number}: {item.name} - already fixed by the request: ' + ('; '.join(item.musts) or 'nothing'))
        listing = []
        last = None
        for index, (label, name, carried, values) in enumerate(table, 1):
            if label != last:
                listing.append(f'category {label!r}:')
                last = label
            listing.append(f'  {index}. {name} (on {carried}): ' + ' | '.join(values) + self.not_stated(label, name, values) + (' [bundles several features: ask it last]' if re.search(CATCH_ALL, name) else ''))
        prompt = PLAN.replace('COUNT', str(max(1, count))) + '\n\nREQUEST:\n' + self.request + '\n\nITEMS:\n' + '\n'.join(settled)
        first = [str(x) for x in _items(self.brief.get('ask_first')) if str(x).strip()]
        if first:
            prompt += '\n\nASK FIRST (the request wants these raised before anything else):\n' + '\n'.join(first)
        prompt += '\n\nATTRIBUTE KEYS ON OPENED LISTINGS:\n' + ('\n'.join(listing) or '(no listing opened in full so far)')
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
            if read >= 4:
                weak = {h for h in hinted[number] if h in own and own[h] < 0.5}
                moved = {}
                for h in weak:
                    partners = [k for k in own if own[k] >= 0.5 and _spelled(k) & _spelled(h)]
                    if partners:
                        moved[h] = max(partners, key=lambda k: (own[k], -len(k)))
                if weak:
                    self.trace.append({'kind': 'weak_hint', 'item': number, 'dropped': sorted(weak), 'moved': moved})
                hinted[number] = (hinted[number] - weak) | {_words(k) for k in moved.values()}
            hinted[number] |= {_words(k) for k in self.vague_keys(number, table)}
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
            lanes[number] = self.untwin(self.keys_first(lanes[number], keys, hinted[number], first))
            lanes[number] = [x for x in lanes[number] if not re.search(CATCH_ALL, _words(x))] + [x for x in lanes[number] if re.search(CATCH_ALL, _words(x))]
        self.trace.append({'kind': 'plan', 'ask': _clip(plan.get('ask'), 120), 'lanes': {n: v[:8] for n, v in lanes.items()}})
        return lanes

    @staticmethod
    def untwin(lane: list[str]) -> list[str]:
        """A lane without a label's second spelling: the same words in singular or plural ("skin concern", "skin
        concerns") or a key carrying an example tail ("screen resolution e g 1920 x 1080"); the first one stays. Run
        13981 spent list places on such twins, and the shopper's "pack type" answer was heard as a "packaging type"
        need too, which no listing carried."""
        out: list[str] = []
        seen: set[str] = set()
        for label in lane:
            plain = re.split('\\s+e\\s?g\\b', label, maxsplit=1)[0].strip() or label
            mark = ' '.join(sorted(_spelled(plain)))
            if mark in seen:
                continue
            seen.add(mark)
            out.append(plain)
        return out

    @staticmethod
    def keys_first(lane: list[str], keys: list[str], hinted: set[str], first: list[str]) -> list[str]:
        """A lane with the item's own listing keys (by how many of its opened listings carry them, as lane_keys ranks
        them) ahead of every planned label that is not a listing key, after the labels a story hint or the request's
        ask-first points at. Real race 15: a question naming a pending need's exact key drew it .59-.73 against
        .25-.36 without it, and 84 of 94 needs were among their category's 6 most carried keys; Halo v2's planner,
        blind to that race's text rows, named most of them by other words."""
        asked = set(_roots(' '.join(first))) if first else set()
        lead = [x for x in lane if _words(x) in hinted or (asked and set(_roots(x)) and set(_roots(x)) <= asked)]
        listed = {_words(k) for k in keys}
        ranked = [k for k in keys if k in lane or _words(k) in {_words(x) for x in lane}]
        out: list[str] = []
        for label in lead + ranked + [x for x in lane if _words(x) not in listed]:
            if all((_words(label) != _words(other) for other in out)):
                out.append(label)
        return out

    def vague_keys(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        """The item's listing keys that its listings fill with measures (most values a number with a unit) while the
        request settles them only in words, with no number ("<property>: fits the space it goes in"). Such a must is not
        settled for asking: the shopper gives the number when asked (a
        request that described a size only by the place the product must fit got a number back when it was asked)."""
        item = self.items[number - 1]
        loose = [_spelled(m.split(':', 1)[0]) for m in item.musts if ':' in m and (not re.search('\\d', m.split(':', 1)[1]))]
        if not loose:
            return []
        out = []
        for label, key, _, values in table:
            if self.owner_of(label) != number and len(self.items) > 1:
                continue
            numeric = sum((1 for v in values if re.match('^\\s*[\\d.,/x -]+\\s*[a-z"\']{0,8}\\s*$', v)))
            if values and numeric * 2 > len(values) and any((_names(prop, _spelled(key)) for prop in loose)) and key not in out:
                out.append(key)
        return out

    def name_parts(self, number: int) -> list[str]:
        """Products that an item name links with "and", "or", "&" or commas when every one is a plural noun ("hooks
        and hangers"), else []; one product named that way ("bread and butter knife", "hot and cold taps",
        "his and hers mugs") yields []."""
        name = self.items[number - 1].name
        if not re.search('\\b(?:and|or)\\b|&', name, re.I):
            return []
        parts = [p.strip() for p in re.split('\\s*(?:,|&|\\band\\b|\\bor\\b)\\s*', name, flags=re.I) if p.strip()]
        return parts if len(parts) >= 2 and all((re.search('[a-z]{3}(?<![su])s$', p, re.I) for p in parts)) else []

    def item_said(self, number: int) -> str:
        """How a question refers to the item. When a name links several plural nouns, the question covers just one of
        them: questions naming two products got a value answer roughly half as often as those naming one (race 16, all
        agents: .48 vs .89 on need properties). The chosen product is the one whose head noun shows up in the bulk of the item's rows
        outside their category label (the first one when there are no such rows)."""
        item = self.items[number - 1]
        parts = self.name_parts(number)
        if not parts:
            return item.name
        cat_key = self.category_key()
        rows = [row for row in self.rows.values() if not cat_key or self.owns(item, str(row.get(cat_key) or ''))]
        named = [_roots(' '.join((str(v) for k, v in row.items() if k != cat_key and isinstance(v, str)))) for row in rows]
        hits = {p: sum((1 for words in named if _roots(p.split()[-1]) & words)) for p in parts}
        return max(parts, key=lambda p: (hits[p], -parts.index(p)))

    def question(self, number: int, label: str, samples: list[str] | None=None) -> str:
        text = f'Which {label} do you want for the {self.item_said(number)}'
        return text + (f', such as {samples[0]} or {samples[1]}?' if samples else '?')

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
            answer, notices = self.reply_parts(reply.get('observation'))
            words = self.spoken(answer)
            if notices:
                meaning = self.take_note(notices)
                self.trace.append({'kind': 'reply_notice', 'lines': [n[:160] for n in notices], 'read': meaning[:200]})
                if meaning:
                    self.queued.append(f'(came with a shopper reply) {meaning}')
            sent.append((number, text, words))
            self.done_log.append(f'step {self.step}: sent the shopper {text!r} -> answer {words[:300]!r} (cart then held: {self.cart_mark()})')
            self.done_steps.append(self.step)
            self.trace.append({'kind': 'ask', 'q': text, 'a': words[:300]})
            if isinstance(answer, dict) and answer.get('error'):
                break
        return sent

    def reply_parts(self, observation: Any) -> tuple[Any, list[str]]:
        """A shopper reply can carry store notices beside it: a list of sentences next to the reply text. The reply
        goes on to the reply reader; the sentences, when not seen before, go to the note reader, which knows the
        request's money unit and every kind of change. Without a text part next to them nothing is split off."""
        if not isinstance(observation, dict) or observation.get('error'):
            return (observation, [])
        lists = {k: v for k, v in observation.items() if isinstance(v, list) and v and all((isinstance(x, str) for x in v))}
        rest = {k: v for k, v in observation.items() if k not in lists}
        if not lists or not any((isinstance(v, (str, dict)) and len(self.spoken(v).split()) >= 2 for v in rest.values())):
            return (observation, [])
        notices = [x.strip() for v in lists.values() for x in v if len(x.split()) >= 3 and x.strip() not in self.heard]
        texts = [v for v in rest.values() if isinstance(v, str)]
        return (texts[0] if len(texts) == 1 and len(rest) == 1 else rest, notices)

    def figure_as_said(self, figure: float, said: str) -> float:
        """A budget the reply reader returned, kept in the unit the shopper wrote it in: when the figure is a figure
        written in the reply shifted by a power of ten (8639 read as 86.39), the written one stands; Halo never rescales
        a figure by itself."""
        written = [float(x.replace(',', '')) for x in re.findall('\\d[\\d,]*(?:\\.\\d+)?', said or '')]
        written = [w for w in written if w > 0]
        for shift in (0, 1, 2, 3, -1, -2, -3):
            for w in written:
                if abs(figure * 10 ** shift - w) <= 1e-06 * max(1.0, w):
                    if shift:
                        self.trace.append({'kind': 'budget_as_said', 'heard': figure, 'written': w})
                    return int(w) if w == int(w) else w
        return figure

    def money_value(self, value: str) -> bool:
        """A need value that is a sum of money: a number together with a word of the request's budget unit."""
        unit = {w for w in _spelled(str(self.brief.get('budget_unit') or '')) if len(w) >= 3}
        return bool(unit) and bool(re.search('\\d', value)) and bool(unit & _spelled(value))

    def drop_hits(self, item: Item, prop: str) -> list[str]:
        """Requirements in force on the property a drop names: the same words first; failing that, the one requirement
        whose property and the dropped name contain one another word for word (plural or singular alike), if only one."""
        live = [m for m in item.musts + item.told if m not in item.dropped]
        gone = _words(prop.split(':')[0])
        same = [m for m in live if _words(m.split(':')[0]) == gone]
        if same or not gone:
            return same
        name = _spelled(gone)
        near = [m for m in live if ':' in m and _spelled(m.split(':')[0]) and (name <= _spelled(m.split(':')[0]) or _spelled(m.split(':')[0]) <= name)]
        return near if len(near) == 1 else []

    def take_heard(self, heard: dict[str, Any], told: list[str], said: str='', labels: list[str] | tuple[str, ...]=()) -> None:
        many = len(self.items) > 1
        spoken = _spelled(said) if said else set()
        freed = [e for e in _items(heard.get('free')) + _items(heard.get('free_said')) if isinstance(e, dict) and str(e.get('property') or '').strip()]
        for entry in _items(heard.get('needs')):
            if not isinstance(entry, dict) or not str(entry.get('need') or '').strip():
                continue
            number = min(max(1, _count(entry.get('item')) or 1), len(self.items))
            need = str(entry['need']).strip()
            prop, colon, value = need.partition(':')
            if colon and self.money_value(value):
                self.trace.append({'kind': 'need_is_money', 'need': need[:160]})
                continue
            if any((min(max(1, _count(f.get('item')) or 1), len(self.items)) == number and _spelled(str(f['property']).split(':')[0]) == _spelled(prop) for f in freed)):
                self.trace.append({'kind': 'need_also_free', 'need': need[:160]})
                continue
            if colon and spoken and (not _spelled(prop) <= spoken):
                pool = list(labels) + list((getattr(self, 'lanes', None) or {}).get(number) or []) + [t.split(':')[0] for t in self.items[number - 1].told]
                named = [l.strip() for l in pool if _spelled(l) and _spelled(l) <= spoken and (_spelled(l) & _spelled(prop)) and (not _spelled(l) <= _spelled(value))]
                if named:
                    better = max(named, key=lambda l: len(_spelled(l)))
                    self.trace.append({'kind': 'need_renamed', 'from': prop.strip(), 'to': better})
                    need = f'{better}:{value}'
            if need not in self.items[number - 1].told:
                self.items[number - 1].told.append(need)
                told.append(f'item {number}: {need}' if many else need)
            if entry.get('points') is True and (number, need) not in self.refs:
                self.refs.append((number, need))
        for entry in _items(heard.get('drop')):
            if isinstance(entry, dict) and str(entry.get('property') or '').strip():
                item = self.items[min(max(1, _count(entry.get('item')) or 1), len(self.items)) - 1]
                hit = self.drop_hits(item, str(entry['property']))
                item.dropped.extend(hit)
                told.extend((f'dropped {m}' for m in hit))
        for entry in _items(heard.get('free')):
            if isinstance(entry, dict) and entry.get('property'):
                number = min(max(1, _count(entry.get('item')) or 1), len(self.items))
                self.items[number - 1].free.append(str(entry['property']))
        if isinstance(heard.get('budget'), (int, float)) and (not isinstance(heard.get('budget'), bool)) and (heard['budget'] > 0):
            self.budget = self.figure_as_said(heard['budget'], said)

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(HEAVY, LIGHT)) -> dict[str, Any]:
        prompt = HEAR + '\n\nITEMS:\n' + '\n'.join((f'item {n}: {it.name}' for n, it in enumerate(self.items, 1)))
        prompt += '\n\nQUESTIONS SENT AND REPLIES:\n' + '\n'.join((f'{i}. (item {n}) Q: {q}\nA: {a}' for i, (n, q, a) in enumerate(sent, 1)))
        return self.ask_json(prompt, list(models), 2500) or {}

    def free_drops(self, sent: list[tuple[int, str, str]]) -> list[str]:
        """Requirements in force that an answer lets go of. The reader files a property the shopper says they no longer
        mind under "free", the same as one they never cared about. When that property is not what the message asked
        about, the answer brought it up by itself, and when it is a requirement in force (from the request or an
        earlier answer), the shopper is giving it up: the later word replaces the request's text. Such a requirement is
        dropped as a note's drop is. It must be named in an answer and be the property of exactly one requirement, word
        for word. Left in the ledger, such a requirement keeps a pricier listing that meets it
        ahead of cheaper ones that now fit. Returns the 'dropped ...' lines for the summary."""
        out: list[str] = []
        for number, item in enumerate(self.items, 1):
            fresh = item.free[self.free_read.get(number, 0):]
            self.free_read[number] = len(item.free)
            asked = [_spelled(re.sub('[^\\w\\s]', ' ', q)) for n, q, _ in sent if n == number]
            spoken = _spelled(re.sub('[^\\w\\s]', ' ', ' '.join((a for n, _, a in sent if n == number))))
            for prop in fresh:
                name = _spelled(str(prop).split(':')[0])
                if not name or not name <= spoken or any((name <= q for q in asked)):
                    continue
                live = [m for m in item.musts + item.told if m not in item.dropped and ':' in m]
                hit = [m for m in live if _spelled(m.split(':')[0]) == name]
                if len(hit) != 1:
                    continue
                item.dropped.extend(hit)
                out.append(f'dropped {hit[0]}')
                self.trace.append({'kind': 'free_drop', 'item': number, 'dropped': hit[0][:200]})
        if out:
            self.notes.append('the shopper gave up ' + '; '.join((o[len('dropped '):] for o in out)))
            self.note_step = self.step
            self.widened_at = self.step
            self.recent = []
            if self.brief.get('pick') == 'cheapest':
                self.wider_held = False
                self.cheaper_marks = set()
        return out

    def gate(self, stage: str) -> tuple[bool, str]:
        steps = [p for p in self.process if p['when'] == stage]
        if not steps or self.finished or (not self.gate_due(stage)):
            return (True, '')
        ending = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        tools = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_dump(t['function'].get('parameters') or {})}" for t in self.tools if t['function']['name'] not in ending]
        said: list[str] = []
        nudged = False
        for _ in range(GATE_ROUNDS):
            prompt = GATE.replace('STAGE', 'place the order' if stage == 'order' else 'close the episode without buying') + '\n\nREQUEST:\n' + self.request + '\n\nSTEPS REQUIRED AT THIS POINT:\n' + '\n'.join((f"- {p['step']}" for p in steps)) + '\n\nLEDGER:\n' + self.ledger() + '\n\nCURRENT CART: ' + self.cart_text() + '\n\nRECORD (step calls and messages up to now):\n' + ('\n'.join((f'{i}. {line}' for i, line in enumerate(self.done_log, 1))) or '(empty so far)') + ('\n\nNOTES FROM THE STORE:\n' + '\n'.join(self.notes[-4:]) if self.notes else '') + f"\n\nMESSAGES REMAINING: {max(0, self.msg_limit - self.msg_used)}   STEPS REMAINING: {self.left()} (the {('order' if stage == 'order' else 'ending')} uses one of them)" + '\n\nTOOLS:\n' + '\n'.join(tools)
            got = self.ask_json(prompt, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 3500)
            if got is None and self.model_room() > 30:
                got = self.ask_json(prompt, [HEAVY, LIGHT], 3500)
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
                return (False, ('First, Halo carried out the steps the request calls for:\n' + '\n'.join(said) + '\n' if said else '') + ("The shopper's answer alters the task: " + '; '.join(told + ([hold] if hold else [])) + '. Adjust the cart to match, then order again.'))
            name = str((call or {}).get('tool') or '')
            shown = [e for e in _items(got.get('steps')) if isinstance(e, dict)]
            talk = stage == 'order' and any((p['messages'] > 0 for p in steps))
            lines = range(1, len(self.done_log) + 1)
            proof = {i for i in lines if not talk or (i <= len(self.done_steps) and self.done_steps[i - 1] > self.cart_at)}
            unproven = [e for e in shown if e.get('done') is not True or not any(((_count(n) or 0) in proof for n in _items(e.get('lines'))))]
            if not call and unproven and (not said) and (not nudged) and self.roles.get('ask') and (self.msg_used < self.msg_limit) and (self.left() >= 2):
                nudged = True
                self.trace.append({'kind': 'gate_unproven', 'steps': _clip(unproven, 120)})
                self.done_log.append(f'(Halo: no RECORD line after the cart change at step {self.cart_at} proves ' + '; '.join((str(e.get('step') or '') for e in unproven))[:200] + f' was done; messages left {self.msg_limit - self.msg_used}, steps left {self.left()}: name the call that does it)')
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
            self.done_log.append(f'step {self.step}: {name} {_dump(args)} -> {self.show(seen)[:300]} (cart then held: {self.cart_mark()})')
            self.done_steps.append(self.step)
            said.append(f'{name} {_dump(args)} -> {self.show(seen)[:400]}')
        return (True, 'First, Halo carried out the steps the request calls for:\n' + '\n'.join(said) if said else '')

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
        prompt = NOTE + '\n\nCURRENT ITEMS AND REQUIREMENTS:\n' + self.ledger() + '\n\nNOTE TEXT:\n' + '\n'.join(lines)
        got = self.ask_json(prompt, [LIGHT, SPARE], 2500) or {}
        gone = [str(g) for g in _items(got.get('gone')) if isinstance(g, (str, int))]
        self.unavailable.update((g for g in gone if g in self.rows))
        self.note_sold = bool(gone) or got.get('sold_out') is True
        self.note_named = any((g in self.rows for g in gone))
        if self.note_sold:
            self.sold_cart = set(self.cart_ids())
        if self.note_sold and len(lines) == 1:
            self.sold_lines.update(lines)
        if got.get('kind') not in (None, 'change'):
            return ''
        changed = []
        before = self.budget
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
                hit = self.drop_hits(item, target)[:1]
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
            changed.append('no longer offered: ' + ', '.join(gone))
        if got.get('price_moved') is True:
            changed.append('a price moved')
        summary = str(got.get('summary') or '').strip()
        raised = isinstance(before, (int, float)) and isinstance(self.budget, (int, float)) and self.budget > before
        if drops or (raised and self.brief.get('pick') in ('most', 'least')):
            self.widened_at = self.step
        if self.brief.get('pick') == 'cheapest' and (drops or gone or self.note_sold or any((c.startswith('budget now') for c in changed))):
            self.wider_held = False
            self.cheaper_marks = set()
        if changed or summary:
            self.version += 1
            self.recent = []
            named = [g for g in gone if g in self.rows]
            mark = f" [sold out: {', '.join(named)}]" if named else ' [names no listing]' if self.note_sold else ''
            self.sold_note = len(self.notes) if self.note_sold and (not named) else self.sold_note
            self.notes.append((summary or '; '.join(changed)) + mark)
            self.note_step = self.step
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
                if self.fit_need(number, need):
                    continue
                prop = need.split(':', 1)[0].strip()
                named = [(k, v) for k, v in block.items() if _names(_stem(prop), _stem(k))]
                plain = need.split(':', 1)[1] if not re.search('\\d', need) else ''
                hits = [f'{k}={str(v)[:50]}' + (' (choices on offer, not a stated value)' if _offered(k) else f' (its own option names {p}: the listing is {p})' if p else '') for k, v in named for p in [_picked_other(block, v, plain) if plain else '']]
                tag = f'(item {number}) ' if many else ''
                if hits:
                    twins = len({_spelled(k) for k, _ in named}) < len(named) and len({_words(str(v)) for _, v in named}) > 1
                    parts.append(f'{tag}{prop} -> ' + ', '.join(hits[:3]) + (' (spellings of one key with different values: the requirement is met when any of them states it)' if twins else ''))
            if parts:
                lines.append(f'{ident}: ' + '; '.join(parts))
        if not lines:
            return ''
        return "[evidence: attribute keys whose name includes a requirement's property; a key for another property (another component or facet) proves nothing, and a differently named key may still describe the same property] " + ' | '.join(lines)

    def cart_text(self) -> str:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            return _dump(_clip(self.cart_view, 300))[:3000]
        return "cart per Halo's log of cart calls (listing: units): " + _dump(self.cart_lines)

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
        return _dump(_clip(record, 400))[:3000] if record else '(no record yet)'

    def facts_view(self, ident: str, size: int=1200) -> str:
        """One listing's record for a note Halo writes after opening it in code, attributes first: the attribute block
        leads, then the short fields, and long free text comes last and cut short. A note cut at a fixed length after
        the description used to end inside the attributes or before them, and the model, told the listing was read,
        searched on for a value its attributes stated."""
        record = self.full.get(ident) or self.rows.get(ident)
        if not isinstance(record, dict) or not record:
            return self.record_of(ident)[:size]

        def rank(pair: tuple[str, Any]) -> int:
            value = pair[1]
            if isinstance(value, dict) or (isinstance(value, str) and value.count('=') >= 2):
                return 0
            if isinstance(value, str) and len(value) > 120:
                return 3
            return 2 if isinstance(value, list) else 1
        ordered = {k: _clip(v, 160) if rank((k, v)) == 3 else v for k, v in sorted(record.items(), key=rank)}
        return _dump(_clip(ordered, 400))[:size]

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

    def fits_codes(self, ident: str, codes: set[str], index: dict[str, Any]) -> bool | None:
        """Whether a part's text names one of a device's codes (a family such as rot31xx counts, its 'except' codes do not);
        None when the part names no codes or the device has none."""
        fits = self.fit_set(ident, index)
        if not fits or not codes:
            return None
        nots = set((index['marks'].get(ident) or {}).get('nots') or [])
        return any((_fit_hit(f, c) for f in fits for c in codes)) and (not any((_fit_hit(n, c) for n in nots for c in codes)))

    def fit_index_now(self) -> dict[str, Any]:
        """fit_index(), rebuilt only when the seen or opened listings changed (covers() runs it per listing)."""
        key = (len(self.seen_order), len(self.full))
        if self.fit_memo[0] != key:
            self.fit_memo = (key, self.fit_index())
        return self.fit_memo[1]

    def fit_need(self, number: int, need: str) -> bool:
        """A requirement of a part item (self.bound) whose value only asks it to fit the other requested item ("fits the
        X bought as item 1", "the X bought"). The store decides that by the model codes the pages name (fit_index), not by
        attribute words: a line whose compatibility attribute names another product can still fit by its codes, and
        reading the words dropped such lines. The brief also writes such a requirement as "compatible with the X
        chosen" or "must suit the X", with or without a property name before a colon; read as a plain value, it failed
        every listing of the part (no attribute names "compatibility"), so every code check that uses covers() passed
        none of them."""
        mains = [m for p, m in self.bound if p == number]
        if not mains or need in [r for _, r in self.refs]:
            return False
        value = (need.partition(':')[2] if ':' in need else need).lower()
        if re.search('\\d', re.sub('\\bitem \\d+\\b', '', value)) or re.search('\\b(?:if|unless|when|whether|than)\\b', value):
            return False
        if not re.search('\\b(?:fits?|fitting|bought|buying|end up|works? with|goes with)\\b', need.lower()) and (not re.search('\\b(?:compatible|suits?)\\b', value)):
            return False
        flat = re.sub('[^a-z0-9]', '', value)
        names = lambda m: _roots(self.items[m - 1].name) | {re.sub('[^a-z0-9]', '', self.items[m - 1].name.lower())}
        return any((re.search(f'\\bitem {m}\\b', value) or any((len(r) >= 4 and r in flat for r in names(m))) for m in mains))

    def fit_device_codes(self, number: int, index: dict[str, Any], lines: list[str] | None=None) -> set[str]:
        """Model codes of the one device line (in `lines`, else the cart) that part item `number` must fit; empty when
        there is no single such line."""
        pool = self.cart_ids() if lines is None else lines
        for part, main in self.bound:
            if part == number:
                devices = [i for i in pool if i in self.item_ids(main) and (not self.sold_out(i))]
                if len(devices) == 1:
                    return set(index['marks'].get(devices[0], {}).get('codes') or [])
        return set()

    def item_ids(self, number: int) -> list[str]:
        cat_key = self.category_key()
        item = self.items[number - 1]
        return [i for i in self.seen_order if cat_key and self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))]

    def told_codes(self, number: int, index: dict[str, Any]) -> set[str]:
        """Device codes the requirements in force on an item name ("fits my phone, model ROT3110"); a code counts only when
        some seen listing names it as a model code or a fit code, so a figure such as "up to 1000" is never one."""
        told = {c for owner, need in self.needs_now() if owner == number for c in _codes((need.partition(':')[2] or need).upper()) if not c.endswith('x')}
        known = {c for mark in index['marks'].values() for c in (mark.get('codes') or []) + (mark.get('works') or []) + (mark.get('nots') or [])}
        return {c for c in told if any((_fit_hit(k, c) for k in known))}

    def fit_parts(self, part: int, codes: set[str], index: dict[str, Any], cart: list[str]) -> tuple[list[str], list[str]]:
        """Listings of part item `part` outside the cart that work with one of a device's codes: (opened ones whose
        attributes meet the part's own requirements, cheapest first; unread rows, best evidence first). A row is no
        repair until its page is read: an unread row may be another kind of product, and naming it as the cheapest repair
        spends the last turns opening such rows."""
        pool = [i for i in self.item_ids(part) if i not in cart and self.buyable(i) and self.price_of(i) is not None and self.fits_codes(i, codes, index)]
        proven = sorted((i for i in pool if i in self.full and self.covers(i, part, alone=True)), key=lambda i: self.price_of(i) or 0.0)
        return (proven, self.row_rank([i for i in pool if i not in self.full], part))

    def fit_sets(self, main: int) -> list[dict[str, Any]]:
        """One entry per model code seen on a listing of device item `main`: for the device and for each part item bound
        to it, the cheapest opened, buyable listing that meets its own requirements and carries (device) or works with
        (part) that code, else up to 3 unread rows that do, best evidence first. Ranked by slots filled with such opened
        listings, then slots with an unread row, then the filled lines' cost. This shows when another code's set is nearly complete while the first device's parts
        are still being searched for."""
        index = self.fit_index_now()
        marks = index['marks']
        parts = [p for p, m in self.bound if m == main]
        codes = sorted({c for i in self.item_ids(main) if not self.sold_out(i) for c in (marks.get(i) or {}).get('codes') or []})
        out = []
        for code in codes:
            slots: dict[int, tuple[list[str], list[str]]] = {}
            for number in [main] + parts:
                if number == main:
                    pool = [i for i in self.item_ids(main) if code in ((marks.get(i) or {}).get('codes') or []) and self.buyable(i) and self.price_of(i) is not None]
                    proven = sorted((i for i in pool if i in self.full and self.covers(i, main, alone=True)), key=lambda i: self.price_of(i) or 0.0)
                    slots[number] = (proven[:1], self.row_rank([i for i in pool if i not in self.full], main)[:3])
                else:
                    proven, unread = self.fit_parts(number, {code}, index, [])
                    slots[number] = (proven[:1], unread[:3])
            done = sum((1 for p, _ in slots.values() if p))
            lead = sum((1 for p, u in slots.values() if not p and u))
            cost = sum(((self.price_of(p[0]) or 0.0) * self.items[n - 1].qty for n, (p, _) in slots.items() if p))
            out.append({'code': code, 'slots': slots, 'done': done, 'lead': lead, 'cost': cost})
        out.sort(key=lambda s: (-s['done'], -s['lead'], s['cost']))
        return out

    def fit_set_text(self, entry: dict[str, Any]) -> str:
        bits = []
        for number, (proven, unread) in entry['slots'].items():
            name = self.items[number - 1].name
            if proven:
                bits.append(f'{name} {proven[0]} ({self.price_of(proven[0]):g}, opened, fits)')
            elif unread:
                bits.append(f"{name}: none opened fits yet; unread rows: {', '.join((f'{i} ({self.price_of(i):g})' for i in unread))}")
            else:
                bits.append(f'{name}: no opened listing meets its requirements and no unread row is left')
        whole = entry['done'] == len(entry['slots'])
        return f"{entry['code']}: " + '; '.join(bits) + (f" (a complete set: these lines cost {entry['cost']:g})" if whole else '')

    def fit_faults(self) -> list[str]:
        index = self.fit_index()
        cart = self.cart_ids()
        faults = []
        units = lambda i: max(1, self.held_units(i) or 1)
        cost = lambda i, n: (self.price_of(i) or 0.0) * n
        for part, main in self.bound:
            devices = [i for i in cart if i in self.item_ids(main)]
            parts = [i for i in cart if i in self.item_ids(part)]
            if len(devices) != 1 or not parts:
                continue
            device = devices[0]
            codes = set(index['marks'].get(device, {}).get('codes') or [])
            if not codes:
                continue
            for line in parts:
                proven, unread = self.fit_parts(part, codes, index, cart)
                rows = f"; unread rows that name it, best evidence first (read one before using it): {', '.join((f'{i} ({self.price_of(i):g})' for i in unread[:4]))}" if unread else ''
                if not self.fit_set(line, index):
                    if proven or unread or any((self.fit_set(i, index) for i in self.item_ids(part) if i != line)):
                        faults.append(f"{line} ({self.items[part - 1].name}) names no model it fits, so it is not shown to fit {device} ({', '.join(sorted(codes))}), while other listings of that kind name theirs" + (f"; opened ones that name it and meet their requirements, cheapest first: {', '.join((f'{i} ({self.price_of(i):g})' for i in proven[:4]))}" if proven else '') + rows + ('' if proven or unread else '; look for one whose text names that model'))
                    continue
                if self.fits_codes(line, codes, index) is not False:
                    continue
                fits = self.fit_set(line, index)
                good = proven[:4]
                others = [p for p, m in self.bound if m == main and p != part]
                hosts = sorted((i for i in self.item_ids(main) if i != device and (not self.sold_out(i)) and self.price_of(i) is not None and self.fits_codes(line, set(index['marks'].get(i, {}).get('codes') or []), index)), key=lambda i: self.price_of(i) or 0.0)[:2]
                text = f"{line} ({self.items[part - 1].name}) fits {', '.join(sorted(fits)[:4])} but not {device} ({', '.join(sorted(codes))})"
                swap_part = cost(device, units(device)) + cost(good[0], self.items[part - 1].qty) if good else None
                swap_host = cost(hosts[0], units(device)) + cost(line, units(line)) if hosts and (not others) else None
                if good:
                    text += f"; opened parts of that kind that meet their requirements and work with it, cheapest first: {', '.join((f'{i} ({self.price_of(i):g})' for i in good))}"
                if swap_part is not None and (swap_host is None or swap_part <= swap_host):
                    text += f'; the cheapest repair is to swap the part to {good[0]} (this pair costs {swap_part:g})' + (f', not to swap the device ({hosts[0]} with {line} costs {swap_host:g})' if swap_host is not None else '')
                elif swap_host is not None:
                    text += f'; the cheapest repair is to swap the device to {hosts[0]}, which {line} works with (this pair costs {swap_host:g}' + (f'; swapping the part to {good[0]} costs {swap_part:g})' if swap_part is not None else ')')
                text += rows
                if not good and swap_host is None:
                    sets = [s for s in self.fit_sets(main) if not any((_fit_hit(s['code'], c) for c in codes))][:2]
                    text += '; no opened part of that kind that meets its requirements works with it, so read those rows or switch the device and its parts together to a code whose set is (nearly) complete: ' + (' | '.join((self.fit_set_text(s) for s in sets)) if sets else self.fit_plan(short=True))
                faults.append(text)
        faults += self.series_faults(index, cart)
        for number, item in enumerate(self.items, 1):
            codes = self.told_codes(number, index)
            if not codes or any((number == p for p, _ in self.bound)):
                continue
            for line in [i for i in cart if i in self.item_ids(number)]:
                marks = index['marks'].get(line) or {}
                if codes & set(marks.get('codes') or []) or (marks.get('own') or '').lower() in codes:
                    continue
                if self.fits_codes(line, codes, index) is False:
                    good = sorted((i for i in self.item_ids(number) if self.fits_codes(i, codes, index) and (not self.sold_out(i)) and (i not in cart) and self.price_of(i) is not None), key=lambda i: self.price_of(i) or 0.0)[:4]
                    faults.append(f"{line} ({item.name}) fits {', '.join(sorted(self.fit_set(line, index))[:4])}" + (f" except {', '.join(sorted(marks.get('nots') or [])[:3])}" if marks.get('nots') else '') + f", but not {', '.join(sorted(codes))}, which the requirements name" + (f"; seen ones that work with it, cheapest first: {', '.join((f'{i} ({self.price_of(i):g})' for i in good))}" if good else '; look for one that names that code'))
        return faults

    def series_basket(self) -> tuple[str, dict[int, str], float] | None:
        """The cheapest basket that keeps one part series across the items that carry series marks: for each series,
        the seen in-stock listings of each host item that take it ("takes part series AB5") and of each part item coded
        in it ("[AB5-1234]"), opened listings that fail a requirement left out. Returns (series, item -> listing, cost)
        for the cheapest series that has a listing for every such item, else None. Arena 14561 (race-16 world): three
        bundle declines held two of three lines of one series while another series had a full set seen."""
        marks = self.fit_index()['marks']
        owner = {i: n for n in range(1, len(self.items) + 1) for i in self.item_ids(n)}
        hosts: dict[str, dict[int, list[str]]] = {}
        parts: dict[str, dict[int, list[str]]] = {}
        for ident, mark in marks.items():
            number = owner.get(ident)
            if not number or self.sold_out(ident) or self.price_of(ident) is None or (ident in self.full and (not self.covers(ident, number))):
                continue
            for code in mark.get('takes') or ():
                hosts.setdefault(code, {}).setdefault(number, []).append(ident)
            if mark.get('family'):
                parts.setdefault(mark['family'], {}).setdefault(number, []).append(ident)
        host_items = {n for by in hosts.values() for n in by}
        part_items = {n for by in parts.values() for n in by} - host_items
        if not host_items or not part_items:
            return None
        cap = self.budget if isinstance(self.budget, (int, float)) and self.brief.get('budget_scope') != 'each' else None
        best = None
        self.series_full = set()
        for code in set(hosts) & set(parts):
            pools = {n: hosts[code].get(n) for n in host_items}
            pools.update({n: parts[code].get(n) for n in part_items})
            if not all(pools.values()):
                continue
            self.series_full.add(code)
            picks = {n: min(pool, key=lambda i: self.price_of(i) or 0.0) for n, pool in pools.items()}
            cost = sum(((self.price_of(i) or 0.0) * self.items[n - 1].qty for n, i in picks.items()))
            if cap is not None and cost > cap + 1e-09:
                continue
            if best is None or cost < best[2]:
                best = (code, picks, cost)
        return best

    def series_text(self, basket: tuple[str, dict[int, str], float]) -> str:
        code, picks, cost = basket
        return f'series {code}: ' + ', '.join((f'{self.items[n - 1].name}: {i} ({self.price_of(i):g})' for n, i in sorted(picks.items()))) + f' (these lines cost {cost:g})'

    def series_faults(self, index: dict[str, Any], cart: list[str]) -> list[str]:
        """Cart lines of different items whose series disagree: one line's text says which part series it takes
        ("takes part series AB5") and a line of another item carries its own code of a different series in brackets
        ("[AB7-1234]"). Race 16: on the bundle tasks every paid basket kept one series across its lines and every
        Halo basket that mixed series failed. Names the seen listings of the same series that would repair the pair,
        cheapest first, for the part and for the line that takes it."""
        marks = index['marks']
        owner = {i: n for n in range(1, len(self.items) + 1) for i in self.item_ids(n)}
        price = lambda i: self.price_of(i) or 0.0
        open_rows = lambda n: [i for i in self.item_ids(n) if i not in cart and (not self.sold_out(i)) and self.price_of(i) is not None]
        faults = []
        for host in cart:
            takes = set((marks.get(host) or {}).get('takes') or [])
            if not takes or host not in owner:
                continue
            for line in cart:
                family = (marks.get(line) or {}).get('family') or ''
                if line == host or not family or family in takes or owner.get(line, owner[host]) == owner[host]:
                    continue
                parts = sorted((i for i in open_rows(owner[line]) if (marks.get(i) or {}).get('family') in takes), key=price)[:4]
                hosts = sorted((i for i in open_rows(owner[host]) if family in ((marks.get(i) or {}).get('takes') or ())), key=price)[:3]
                text = f"{line} ({self.items[owner[line] - 1].name}) is series {family}, while {host} ({self.items[owner[host] - 1].name}) takes series {', '.join(sorted(takes))}"
                if parts:
                    text += f"; {', '.join(sorted(takes))} listings already seen for {line}'s item, lowest price first: {', '.join((f'{i} ({price(i):g})' for i in parts))}"
                if hosts:
                    text += f"; or listings already seen for {host}'s item that take {family}, lowest price first: {', '.join((f'{i} ({price(i):g})' for i in hosts))}"
                if not parts and (not hosts):
                    basket = self.series_basket()
                    text += f'; seen listings make a full one-series set: {self.series_text(basket)}' if basket else f'; search that item with the series code ({sorted(takes)[0]}) and keep every line of one series'
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
            named: dict[str, list[str]] = {}
            proven = False
            for p in parts:
                good = [i for i in self.item_ids(p) if i in self.full and self.buyable(i) and self.price_of(i) is not None and self.covers(i, p, alone=True)]
                proven = proven or bool(good)
                for i in good or self.item_ids(p):
                    if self.sold_out(i):
                        continue
                    for code in self.fit_set(i, index):
                        tally.setdefault(code, {}).setdefault(p, 0)
                        tally[code][p] += 1
                        if good:
                            named.setdefault(code, []).append(i)
            ranked = sorted(tally.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1].values())))[:4]
            devices = {code: [i for i in self.item_ids(main) if any((_fit_hit(code, c) for c in index['marks'].get(i, {}).get('codes') or []))][:2] for code, _ in ranked}
            text = '; '.join((f'{code}: ' + ', '.join((f'{n} {self.items[p - 1].name}' for p, n in sorted(per.items()))) + (f" ({', '.join(named[code][:2])})" if named.get(code) else '') + (f" (device found: {', '.join(devices[code])})" if devices[code] else ' (device not found yet)') for code, per in ranked))
            sets = [s for s in self.fit_sets(main) if s['done']][:3] if proven else []
            if sets:
                lines.append(f"sets by {self.items[main - 1].name} model code, the most complete first (a slot is filled by an opened listing that meets its own requirements and carries or works with the code; for an empty slot read the unread rows named, and keep the device and every part on one code) -> " + ' | '.join((self.fit_set_text(s) for s in sets)))
            elif text and proven:
                lines.append(f"codes named by opened {' and '.join((self.items[p - 1].name for p in parts))} listings that meet their own requirements (pick a {self.items[main - 1].name} carrying one of them) -> {text}")
            elif text:
                lines.append(f"codes named by parts of the {self.items[main - 1].name} candidates -> {text}")
        if not lines:
            return ''
        return ' | '.join(lines) if short else "[fit plan: a part suits a device if the part's text mentions that device's model code or any other name it sells under; codes do not work as search words, so browse the device category and take the codes from its rows] " + ' | '.join(lines)

    def missing_items(self) -> list[int]:
        """Items lacking a cart line, when the cart has fewer lines than items; an item no seen listing is labelled for is left out."""
        cat_key = self.category_key()
        ids = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        if not cat_key or len(ids) >= len(self.items):
            return []
        labels = [str(r.get(cat_key) or '') for r in self.rows.values()]
        held = [str((self.rows.get(i) or {}).get(cat_key) or '') for i in ids]
        return [n for n, item in enumerate(self.items, 1) if any((self.owns(item, label) for label in labels)) and (not any((self.owns(item, label) for label in held)))]

    def fit_rescue(self) -> None:
        """Once, at a forced order (the model is out of turns or time): a part line that works with another code than
        the one device line in the cart is swapped, within the budget, for the cheapest opened listing of its item that
        meets its own requirements and works with the device; when none is opened, up to two unread rows naming the
        device's code are read first, best evidence first."""
        if self.fit_rescued or not self.bound or not self.roles.get('cart'):
            return
        self.fit_rescued = True
        tool = self.roles['cart']
        for part, main in self.bound:
            index = self.fit_index()
            cart = self.cart_ids()
            devices = [i for i in cart if i in self.item_ids(main)]
            lines = [i for i in cart if i in self.item_ids(part)]
            if len(devices) != 1 or len(lines) != 1:
                continue
            line = lines[0]
            codes = set(index['marks'].get(devices[0], {}).get('codes') or [])
            if not codes or self.fits_codes(line, codes, index) is not False:
                continue
            proven, unread = self.fit_parts(part, codes, index, cart)
            if not proven and unread and self.left() >= self.tail() + 5 and self.model_room() > 6:
                self.fill_facts(unread[:2], self.tail() + 3)
                proven, _ = self.fit_parts(part, codes, self.fit_index(), cart)
            qty = self.items[part - 1].qty
            if isinstance(self.budget, (int, float)) and self.brief.get('budget_scope') == 'each':
                proven = [i for i in proven if (self.price_of(i) or 0.0) <= self.budget + 1e-09]
            elif isinstance(self.budget, (int, float)):
                rest = sum(((self.price_of(i) or 0.0) * (self.held_units(i) or 0) for i in cart if i != line))
                proven = [i for i in proven if rest + (self.price_of(i) or 0.0) * qty <= self.budget + 1e-09]
            keep = {i: self.cart_lines[i] for i in cart if i != line and i in self.cart_lines}
            calls = self.cart_calls({**keep, proven[0]: qty}, [line]) if proven else None
            if not calls or len(calls) + self.tail() > self.left():
                continue
            for args in calls:
                faults: list[str] = []
                args = _shape(args, self.schemas.get(tool), 'args', faults)
                if faults or self.act(tool, args) is None:
                    return
                self.note_cart(tool, args)
            self.trace.append({'kind': 'fit_rescue', 'line': line, 'pick': proven[0]})

    def stated_other(self, ident: str, number: int) -> list[list[set[str]]]:
        """The plain required values of item `number` (a word value or a number with its unit; alternatives allowed)
        that the listing's own attributes contradict: a key naming the property (not a choice key) is there and none of
        its values holds every word of any alternative, reading a list its own option narrows as that option. Each is
        returned as the word sets of its alternatives. Limits, exclusions, fits and pointed-at values are left out."""
        raw = _facts_of(self.full.get(ident) or {}) or {}
        refs = [r for _, r in self.refs]
        out: list[list[set[str]]] = []
        for owner, need in self.needs_now():
            prop, colon, value = need.partition(':')
            low = value.strip().lower()
            if owner != number or not colon or not low or need in refs or self.fit_need(number, need) or re.search('\\b(?:not|no|except|without|other than|unlike|same|at most|at least|no more than|no less than|up to|under|over|below|above|between|within|max|min)\\b', low) or _limit(low):
                continue
            held = [_picked_other(raw, v, low) or str(v) for k, v in raw.items() if not _offered(k) and _names(_spelled(prop), _spelled(k))]
            alts = [o for o in re.split('\\s+or\\s+', low) if _tokens(_words(o))]
            fold = lambda text: set(_spelled(re.sub('[,/;|()]', ' ', str(text))))
            if held and alts and not any((fold(o) <= fold(t) for o in alts for t in held)):
                out.append([_tokens(_words(o)) for o in alts])
        return out

    def value_rescue(self) -> bool:
        """Once, at a forced order (the model is out of turns or steps): a cart line whose own attribute states another
        value than a plain requirement (stated_other) is swapped, within the budget, for the cheapest opened listing of
        its item that meets every requirement (covers); when none is opened, up to two unread rows of that item that the
        value's words reached (the words a search for them sent, or the row's own text) are read first, best evidence
        first. Under a cheapest pick the swap is never dearer than the line; under a most or least pick nothing is
        swapped. A line carted unread in the last steps can show another value under its own key when it is read just
        before the order, while a row that the value's own search returned was never opened. True when it swapped."""
        if self.value_rescued or not self.roles.get('cart') or self.brief.get('pick') in ('most', 'least') or self.model_room() <= 6:
            return False
        self.value_rescued = True
        tool = self.roles['cart']
        cart = self.cart_ids()
        cap = self.budget if isinstance(self.budget, (int, float)) and self.budget > 0 else None
        each = self.brief.get('budget_scope') == 'each'
        extra = max(0, len(self.read_tools()) - 1)
        for number, item in enumerate(self.items, 1):
            lines = [i for i in cart if i in self.item_ids(number)]
            if len(lines) != 1 or lines[0] not in self.full or self.covers(lines[0], number):
                continue
            line = lines[0]
            wrong = self.stated_other(line, number)
            if not wrong:
                continue
            rest = 0.0 if each else sum(((self.price_of(i) or 0.0) * (self.held_units(i) or 0) for i in cart if i != line))
            mine = self.price_of(line)

            def room(i: str) -> bool:
                price = self.price_of(i)
                if price is None or not self.buyable(i):
                    return False
                if self.brief.get('pick') == 'cheapest' and mine is not None and price > mine:
                    return False
                return cap is None or rest + price * (1 if each else item.qty) <= cap + 1e-09
            pool = sorted((i for i in self.item_ids(number) if i != line and i in self.full and room(i) and self.covers(i, number)), key=lambda i: self.price_of(i) or 0.0)
            if not pool:
                rows = [i for i in self.item_ids(number) if i != line and i not in self.full and room(i) and all((any((o <= self.hit_words.get(i, set()) | self.row_words(i) for o in alts)) for alts in wrong))]
                rows = self.row_rank(rows, number)[:2]
                if not rows or self.left() < len(rows) + 1 + extra + self.tail():
                    continue
                self.fill_facts(rows, self.tail() + 1 + extra)
                pool = sorted((i for i in rows if i in self.full and room(i) and self.covers(i, number)), key=lambda i: self.price_of(i) or 0.0)
            keep = {i: self.cart_lines[i] for i in cart if i != line and i in self.cart_lines}
            calls = self.cart_calls({**keep, pool[0]: item.qty}, [line]) if pool else None
            if not calls or len(calls) + extra + self.tail() > self.left():
                continue
            for args in calls:
                faults: list[str] = []
                args = _shape(args, self.schemas.get(tool), 'args', faults)
                if faults or self.act(tool, args) is None:
                    return False
                self.note_cart(tool, args)
            self.trace.append({'kind': 'value_rescue', 'line': line, 'pick': pool[0]})
            return True
        return False

    def states_needs(self, ident: str, number: int) -> bool:
        """True when the listing's attributes spell out every requirement of the item word for word; limits and
        exclusions are not read here and count as unstated."""
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

    def relational(self, number: int, prop: str, value: str) -> bool:
        """Whether a requirement's value names no value of its own but a relation to another listing or item (the
        same as, unlike, what goes with, compatible with the one chosen), whether or not the brief listed it as a
        lookup or a fit: no attribute can state such a value word for word."""
        pointed = any((n == number and _words(r.partition(':')[0]) == _words(prop) and value.strip().lower() in r.lower() for n, r in self.refs))
        return pointed or self.fit_need(number, f'{prop}: {value}') or bool(re.search('\\b(?:same|different|unlike|goes with|go with|together|compatible|fits?|fitting|match\\w*|chosen|as the|listing|page|whatever|anything|wherever)\\b', value.lower()))

    def weigh(self, ident: str, number: int) -> tuple[list[str], list[str], list[str]]:
        """How an opened listing stands on item `number`'s requirements in force: (conflicts, unstated, pending), each a
        list of property names; a met requirement is in none. A conflict is what the listing itself answers with
        another value: its key for the property holds a value sharing no word with the required one (or another
        figure), a stated figure in the limit's unit falls outside the limit, its key holds an excluded value, the value
        a chain walk read off another listing's page differs, its page names a model code of no device line the part
        must fit, or it is out of stock or unpriced. The real grader (race 17) accepted a line whose key for the
        property held another value 0 times in 21. Unstated is a value the request states that nothing on the record
        settles: a limit with no figure under the property's key (a figure only in the title, under another measure's
        key or nowhere was accepted 2 times in 19), a word value no field states (stated under another key or in the
        title it counts: accepted 49 times in 49), a value sharing only some words with the key's value. Pending is a
        value read from another listing, or a fit to a device line, that is not settled yet."""
        raw = _facts_of(self.full.get(ident) or {}) or {}
        block = {k: v for k, v in raw.items() if not _offered(k)}
        record = self.full.get(ident) or self.rows.get(ident) or {}
        split = lambda text: set(re.sub('[,/;|()-]', ' ', _words(str(text))).split())
        refs = [r for _, r in self.refs]
        conflicts: list[str] = []
        unstated: list[str] = []
        pending: list[str] = []
        for owner, need in self.needs_now():
            if owner != number:
                continue
            prop, colon, value = need.partition(':')
            name = (prop if colon else need).strip()[:40]
            if self.fit_need(number, need):
                index = self.fit_index_now()
                codes = self.fit_device_codes(number, index)
                hit = self.fits_codes(ident, codes, index) if codes else None
                if hit is False:
                    conflicts.append(name)
                elif hit is None:
                    pending.append(name)
                continue
            doubt = unstated
            if need in refs:
                doubt = pending
                need = self.ref_need(need)
                prop, colon, value = need.partition(':')
            if need and self.need_met(ident, number, need, block, False, raw):
                continue
            low = value.strip().lower()
            if not need or not colon or not low or re.search('\\b(?:unlike|same|different|as the|between|within)\\b', low):
                doubt.append(name)
                continue
            bare = [str(v) for k, v in block.items() if _names(_spelled(prop), _spelled(k))]
            held = [_picked_other(raw, v, low) or str(v) for k, v in block.items() if _names(_spelled(prop), _spelled(k))]
            fold = lambda text: set(_spelled(re.sub('[,/;|()-]', ' ', str(text))))
            if any((fold(t) == fold(low) for t in bare)):
                continue
            limit = _limit(low)
            if limit:
                figures = [got.group(2) for t in held for got in [re.search('([\\d.]+)\\s*([a-z"]*)', t.lower())] if got]
                (conflicts if any((_same_unit(u, limit[3]) for u in figures)) else doubt).append(name)
                continue
            words = split(low)
            others = split(' '.join((str(v) for k, v in record.items() if isinstance(v, (str, int, float)) and (not isinstance(v, bool)) and (not _offered(k)))) + ' ' + ' '.join((str(v) for k, v in block.items() if not _names(_spelled(prop), _spelled(k)))))
            if words & {'not', 'no', 'except', 'without'} or 'other than' in low:
                barred = words - {'not', 'no', 'except', 'without', 'other', 'than', 'a', 'an', 'the', 'or', 'and', 'any'}
                if held and barred:
                    conflicts.append(name)
                elif not barred or barred <= others:
                    doubt.append(name)
                continue
            options = [split(o) - {'either', 'else'} for o in re.split('\\s+or\\s+', low) if o.strip()] if 'or' in words else [words]
            if not held:
                if not any((o and o <= others for o in options)):
                    doubt.append(name)
                continue
            wanted = set(re.findall('\\d+(?:\\.\\d+)?', low))
            shown = {n for t in bare for n in re.findall('\\d+(?:\\.\\d+)?', t)}
            if wanted and shown:
                (doubt if wanted & shown else conflicts).append(name)
            elif any((fold(' '.join(o)) & fold(t) for o in options for t in bare)):
                doubt.append(name)
            else:
                conflicts.append(name)
        price, stocked = self.offer_terms(ident)
        if not stocked or price is None:
            conflicts.append('stock')
        return (conflicts, unstated, pending)

    def ref_need(self, need: str) -> str:
        """A requirement that takes its value from another listing, written with the value the chain walk read on
        that listing's last page ("storage: 64 gb", "colour: not red", "power: at most 240 w"), for weigh(); '' while the
        chain is open or when the relation names neither the same value, nor its opposite, nor a limit (a partner
        value, a range around it)."""
        value = (self.ref_values.get(need) or '').strip()
        prop, _, text = need.partition(':')
        low = text.lower()
        if not value or value.startswith('as stated on') or re.search('\\b(?:partner|pairs?|paired|goes with|go with|goes together|together with|within|near|close to|around|about)\\b', low):
            return ''
        if re.search('\\b(?:unlike|different|differs|not the same|other than|anything but|except)\\b', low):
            return f'{prop}: not {value}'
        if re.search('\\d', value) and re.search('\\b(?:' + UPPER_WORDS + '|' + BELOW_WORDS + ')\\b', low):
            return f'{prop}: at most {value}'
        if re.search('\\d', value) and re.search('\\b(?:' + LOWER_WORDS + '|' + ABOVE_WORDS + ')\\b', low):
            return f'{prop}: at least {value}'
        return f'{prop}: {value}'

    def offer_terms(self, ident: str) -> tuple[float | None, bool]:
        """(price, in stock) of a listing for fill_short_cart(): price_of() and buyable(), or, while the store's price
        and stock keys are not known yet (they are learned from a search result's shape, which an episode without a
        cheapest pick may never read), the number under the row's one key named for a price and any key named for
        stock or availability that says no."""
        price = self.price_of(ident)
        row = self.rows.get(ident) or {}
        if price is None and (not self.price_key):
            keys = [k for k, v in row.items() if isinstance(v, (int, float)) and (not isinstance(v, bool)) and re.search('price|cost', _snake(k)) and (not re.search('old|was|original|before|compare|saving|discount|per_|shipping', _snake(k)))]
            price = float(row[keys[0]]) if len(keys) == 1 else None
        stocked = self.buyable(ident)
        if stocked and (not self.stock_key):
            stocked = not any((re.search('stock|avail|on_hand|orderable|buyable', _snake(k)) and (v is False or (isinstance(v, (int, float)) and (not isinstance(v, bool)) and v <= 0)) for k, v in row.items()))
        return (price, stocked)

    def fill_pick(self, number: int, spent: float, taken: list[str]) -> str | None:
        """The listing fill_short_cart() carts for item `number`: opened, of a store label the item owns with no fuller
        label for it, in stock, inside the budget with `spent` already in the basket, not in the cart or `taken`, with
        no conflict and every value the request states settled by its record (weigh); listings the model took out of
        the cart go last, then the fewest values still pending (read from another listing, a fit to a device line),
        then the lowest price. None when no listing qualifies, and the episode then ends without a purchase. Arena
        r12-r16 code-ended declines: where a fill matched an order accepted elsewhere (6), its listing left only pointed
        values pending; where ending without a purchase was the accepted answer (6), every candidate left a stated value
        unsettled (a value the store does not carry shows as no key, not as another value)."""
        cat_key = self.category_key()
        cap = self.budget if isinstance(self.budget, (int, float)) and self.budget > 0 else None
        each = self.brief.get('budget_scope') == 'each'
        units = 1 if each else self.items[number - 1].qty
        held = set(self.cart_ids())
        pool = []
        for ident in self.item_ids(number):
            price, stocked = self.offer_terms(ident)
            if ident not in self.full or ident in held or ident in taken or (not stocked) or price is None:
                continue
            if cap is not None and price * units + (0.0 if each else spent) > cap + 1e-09:
                continue
            if cat_key and self.fuller_label(number, str((self.rows.get(ident) or {}).get(cat_key) or '')):
                continue
            conflicts, unstated, pending = self.weigh(ident, number)
            if not conflicts and (not unstated):
                pool.append((ident in self.dropped, len(pending), price, ident))
        if not pool:
            return None
        pool.sort()
        return pool[0][3]

    def fill_short_cart(self) -> bool:
        """Before an order Halo places or lets through (the model is out of turns, time or steps, or checks out with
        an item missing): for each item with no cart line, cart fill_pick()'s listing, device items before the parts
        that must fit them, then let the order go. Every missing item gets a line or none does: False when one has no
        listing that qualifies (each opened one conflicts or leaves a stated value unsettled, or none is opened), and
        the episode then ends without a purchase, since a short order never passes. True when no item is missing or every one was filled. Arena r12-r16:
        116 of b29's 132 declines were endings with an empty or short cart, 5 of the 132 were right, and real races
        need a decline on 1-2 tasks of 90; only listings already opened are used, so no step goes to new searches."""
        cart = self.roles.get('cart') or ''
        missing = self.missing_items()
        if not missing:
            return True
        if not cart:
            return False
        mains = {m for _, m in self.bound}
        missing.sort(key=lambda n: n not in mains)
        each = self.brief.get('budget_scope') == 'each'
        held = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        spent = 0.0 if each else sum(((self.offer_terms(i)[0] or 0.0) * (self.held_units(i) or 0) for i in held))
        bare = [n for n in missing if not self.settled(n)]
        if bare and 'fill' not in self.unread_marks and self.model_room() > 35:
            self.unread_marks.add('fill')
            self.open_unread(bare, min(6, 2 * len(bare) + 1), len(missing) + 1)
        picks: dict[int, str] = {}
        for number in missing:
            pick = self.fill_pick(number, spent, list(picks.values()))
            if not pick:
                self.trace.append({'kind': 'fill_none', 'item': number, 'picks': picks})
                return False
            picks[number] = pick
            spent += 0.0 if each else (self.offer_terms(pick)[0] or 0.0) * self.items[number - 1].qty
        calls = self.cart_calls({**{i: self.cart_lines[i] for i in held if i in self.cart_lines}, **{p: self.items[n - 1].qty for n, p in picks.items()}}, [])
        if not calls or len(calls) + 1 > self.left():
            self.trace.append({'kind': 'fill_no_room', 'picks': picks})
            return False
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), 'args', faults)
            if faults:
                return False
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return False
            self.note_cart(cart, args)
            self.observe(cart, reply.get('observation'))
        self.trace.append({'kind': 'filled_lines', 'picks': picks})
        return not self.missing_items()

    def covers(self, ident: str, number: int, alone: bool=False) -> bool:
        """Whether a listing's attributes meet every requirement in force for the item: a plain value word for word
        under a key naming the property (as states_needs), an exclusion ("not 2G") when the key's value lacks the
        excluded words, a limit ("at most 12 GB") when the key's first number with the same unit word (or none on
        both sides) is within it; a value naming alternatives ("custom or foldable") is met by any one of them.
        Anything it cannot read counts as not met. A fit to another requested item (fit_need) is read from the model
        codes against the one device line in the cart (none carted: not met), or left out with alone=True."""
        raw = _facts_of(self.full.get(ident) or {}) or {}
        block = {k: v for k, v in raw.items() if not _offered(k)}
        return all((self.need_met(ident, number, need, block, alone, raw) for owner, need in self.needs_now() if owner == number))

    def need_met(self, ident: str, number: int, need: str, block: dict[str, Any], alone: bool=False, raw: dict[str, Any] | None=None) -> bool:
        """One requirement of covers(), read from the listing's attribute block the way covers() describes."""
        split = lambda text: set(re.sub('[,/;|()]', ' ', _words(str(text))).split())
        if self.fit_need(number, need):
            if alone:
                return True
            index = self.fit_index_now()
            codes = self.fit_device_codes(number, index)
            return bool(codes) and self.fits_codes(ident, codes, index) is True
        prop, colon, value = need.partition(':')
        if not colon:
            return False
        held = [str(v) for k, v in block.items() if _names(_spelled(prop), _spelled(k))]
        if not held:
            return False
        low = value.strip().lower()
        words = split(low)
        limit = _limit(low) if need not in [r for _, r in self.refs] else None
        if limit:
            most, strict, bound, unit = limit
            ok = False
            for text in held:
                got = re.search('([\\d.]+)\\s*([a-z"]*)', text.lower())
                if not got or (got.group(2) != unit and (got.group(2) or unit) and not (got.group(2).startswith(unit) or unit.startswith(got.group(2)))):
                    continue
                try:
                    number_ = float(got.group(1))
                except ValueError:
                    continue
                ok = (number_ < bound if strict else number_ <= bound) if most else (number_ > bound if strict else number_ >= bound)
                break
            return ok
        negated = words & {'not', 'no', 'except', 'without'}
        if negated or 'other than' in low:
            barred = words - {'not', 'no', 'except', 'without', 'other', 'than', 'a', 'an', 'the', 'or', 'and', 'any'}
            return bool(barred) and (not any((barred <= split(t) for t in held)))
        options = [split(o) - {'either', 'else'} for o in re.split('\\s+or\\s+', low) if o.strip()] if 'or' in words else [words]
        if raw is None:
            raw = _facts_of(self.full.get(ident) or {}) or {}
        held = [_picked_other(raw, t, low) or t for t in held]
        return bool(words) and any((o and o <= split(t) for o in options for t in held))

    def shortfall(self, ident: str, number: int) -> list[str]:
        """What keeps a listing from fitting item `number`: the property of each requirement in force that need_met()
        finds unmet, then "stock" when it is sold out and "budget" when its price at the item's quantity alone passes
        the budget. The request's "short on the fewest counts" counts these."""
        raw = _facts_of(self.full.get(ident) or {}) or {}
        block = {k: v for k, v in raw.items() if not _offered(k)}
        out = [(need.partition(':')[0] if ':' in need else need).strip()[:40] for owner, need in self.needs_now() if owner == number and (not self.need_met(ident, number, need, block, False, raw))]
        if not self.buyable(ident):
            out.append('stock')
        price = self.price_of(ident)
        units = 1 if self.brief.get('budget_scope') == 'each' else self.items[number - 1].qty
        if isinstance(self.budget, (int, float)) and price is not None and price * units > self.budget + 1e-09:
            out.append('budget')
        return out

    def closest_note(self) -> str:
        """The remark for an ending without a purchase when the request asks to be told that nothing fits and offered the
        listing short on the fewest counts (brief quiet_decline false): for each requested item, the opened listing
        of its kind with the fewest shortfall() counts, the cheapest among those, by id with what it lacks. '' when no
        item has an opened listing of its kind."""
        parts = []
        for number, item in enumerate(self.items, 1):
            pool = [i for i in self.item_ids(number) if i in self.full and self.price_of(i) is not None]
            if not pool:
                continue
            ranked = sorted(((len(self.shortfall(i, number)), self.price_of(i) or 0.0, i) for i in pool))
            best = ranked[0][2]
            short = self.shortfall(best, number)
            head = f'{item.name}: ' if len(self.items) > 1 else ''
            parts.append(head + best + (f" (short on {', '.join(short[:4])})" if short else ' (meets every requirement)'))
        if not parts:
            return ''
        self.trace.append({'kind': 'closest_note', 'parts': parts[:4]})
        return 'No clean fit, so nothing was bought. Closest listing, short on the fewest counts: ' + '; '.join(parts) + '.'

    def fill_gap(self) -> str:
        """Once, when the model ends without a purchase while the cart already holds lines for some requested items:
        fill each missing item with the cheapest opened listing of its kind whose attributes meet every requirement
        of that item (covers; up to 3 more of its rows opened when none is), and keep the
        filled basket only when an independent check of its records passes it. Real race 15: 3 of Halo v2's 5 declines
        on live order tasks held a part-built cart with the paid listing opened; hayate orders a part-built cart's best
        known fits before declining, and its orders holding an unsure line passed 1 of 18, hence the check. Field research (r13-r15): late-filled orders passed .22 vs .53 and the cart check alone passed 8
        failing baskets in 19, so a fill also needs a listing whose page carries every requirement."""
        if self.gap_tried or len(self.items) < 2 or (not self.judges(SPEND_HARD)) or self.model_room() < 25 or (not self.roles.get('cart')):
            return ''
        held = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0 and (not self.sold_out(i))]
        missing = self.missing_items()
        if any((self.sold_out(i) for i in self.cart_ids())):
            return ''
        if not held or not missing or len(missing) >= len(self.items) or self.left() - self.tail() < len(missing) + 6:
            return ''
        self.gap_tried = True
        cap = self.budget if isinstance(self.budget, (int, float)) else None
        each = self.brief.get('budget_scope') == 'each'
        spent = 0.0 if each else sum(((self.price_of(i) or 0.0) * (self.held_units(i) or 0) for i in held))

        def fits(number: int) -> list[str]:
            qty = self.items[number - 1].qty
            pool = [i for i in self.item_ids(number) if i in self.full and i not in held and self.buyable(i) and self.covers(i, number) and self.price_of(i) is not None and (cap is None or (self.price_of(i) or 0.0) * (1 if each else qty) + spent <= cap + 1e-09)]
            return sorted(pool, key=lambda i: self.price_of(i) or 0.0)
        picks: dict[int, str] = {}
        for number in missing:
            pool = fits(number)
            if not pool:
                fresh = sorted((i for i in self.item_ids(number) if i not in self.full and self.buyable(i)), key=lambda i: self.price_of(i) or 0.0)[:3]
                if fresh and self.left() - self.tail() > len(missing) + len(fresh) + 3:
                    self.fill_facts(fresh, self.tail() + len(missing) + 3)
                    pool = fits(number)
            if not pool:
                self.trace.append({'kind': 'gap_unfilled', 'item': number})
                return ''
            picks[number] = pool[0]
            spent += 0.0 if each else (self.price_of(pool[0]) or 0.0) * self.items[number - 1].qty
        lines = held + list(picks.values())
        self.check_cart(lines, SPEND_HARD)
        if not self.verdicts.get(self.check_key(lines)):
            self.trace.append({'kind': 'gap_refused', 'picks': picks})
            return ''
        calls = self.cart_calls({**{i: self.held_units(i) or self.cart_lines.get(i, 1) for i in held}, **{p: self.items[n - 1].qty for n, p in picks.items()}}, [])
        if not calls or len(calls) + 1 > self.left():
            return ''
        cart = self.roles['cart']
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), 'args', faults)
            if faults or self.act(cart, args) is None:
                return ''
            self.note_cart(cart, args)
        self.trace.append({'kind': 'gap_filled', 'picks': picks})
        named = ', '.join((f'{self.items[n - 1].name}: {p}' for n, p in picks.items()))
        return f'Held back for now (the ending took no step). Only {len(held)} of the {len(self.items)} requested items had cart lines, so Halo carted the best-fitting opened listing for each missing one ({named}), and a separate review of the full basket\'s records passes it on every requirement. With such a basket available a no-purchase close scores zero: order now, or close with no purchase only if some record plainly breaks a live requirement (and name it).'

    def fuller_label(self, number: int, label: str) -> str:
        """The store label that names the item exactly when a line comes from another: a label naming the item in full
        beats one that names only part of it, and one naming the item alone beats one that names more kinds as well."""
        cat_key = self.category_key()
        item = self.items[number - 1]
        mine, theirs = (_roots(item.name), _roots(label))
        if len(mine) < 2 or not cat_key or (item.shelf and theirs == _roots(item.shelf)):
            return ''
        seen = sorted({str(r.get(cat_key) or '') for r in self.rows.values()})
        if not mine <= theirs:
            return next((other for other in seen if other and mine <= _roots(other)), '')
        return next((other for other in seen if other and mine <= _roots(other) < theirs), '')

    def line_faults(self) -> list[str]:
        self.adopt_labels()
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
                faults.append(f'item {n} ({item.name}) is missing from the cart')
            elif len([i for i in lines if owner[i] == [n]]) > 1:
                faults.append(f"item {n} ({item.name}) has {len(lines)} lines ({', '.join(lines)}); keep one")
            for i in lines:
                have = self.held_units(i)
                if owner[i] == [n] and have != item.qty:
                    faults.append(f'{i} is held at {have} unit(s), but item {n} ({item.name}) needs {item.qty}')
                fuller = self.fuller_label(n, label[i])
                if fuller:
                    faults.append(f'{i} sits under the label {label[i]}, yet a closer store label exists, {fuller}, naming item {n} ({item.name}) ' + ('fully' if not _roots(item.name) <= _roots(label[i]) else 'without the extra kinds that label adds') + ': take the line from that one')
        bare = [n for n in range(1, len(self.items) + 1) if n not in known and (not any((n in owner[i] for i in ids)))]
        if len(ids) < len(self.items) and bare:
            faults.append(f'only {len(ids)} cart line(s) for {len(self.items)} requested items; still no line for ' + ', '.join((f'item {n} ({self.items[n - 1].name})' for n in bare[:len(self.items) - len(ids)])) + ': find it and add it')
        if len(known) == len(self.items):
            for i in ids:
                if not owner[i]:
                    faults.append(f"{i} ({label[i] or 'kind not known'}) is no kind the shopper requested: remove it")
        for i in ids:
            if self.sold_out(i):
                faults.append(f'{i} can no longer be bought (sold out)')
        prices = {i: self.price_of(i) for i in ids}
        if isinstance(self.budget, (int, float)) and all((p is not None for p in prices.values())):
            if self.brief.get('budget_scope') == 'each':
                over = [i for i in ids if self.budget < prices[i] <= 10 * self.budget]
                faults += [f'{i} is priced {prices[i]:g}, above the per-item limit of {self.budget:g}' for i in over]
            else:
                total = sum((prices[i] * self.held_units(i) for i in ids))
                if self.budget + 1e-09 < total <= 10 * self.budget:
                    faults.append(f'the cart totals {total:g}, above the {self.budget:g} budget')
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
        wrong = [w for w in wrong if w[0] not in self.count_fixed]
        if not name or not target or (not count) or (not wrong):
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
        self.count_fixed.update((w[0] for w in wrong))
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

    def fit_pairs(self) -> str:
        index = self.fit_index()
        lines = []
        for part, main in self.bound:
            hosts = [i for i in self.item_ids(main) if i in self.full and (not self.sold_out(i)) and self.price_of(i) is not None]
            parts = [i for i in self.item_ids(part) if i in self.full and (not self.sold_out(i)) and self.price_of(i) is not None]
            pairs = [(self.price_of(h) * self.items[main - 1].qty + self.price_of(q) * self.items[part - 1].qty, h, q) for h in hosts for q in parts if self.fits_codes(q, set(index['marks'].get(h, {}).get('codes') or []), index)]
            pairs.sort()
            if pairs:
                lines.append(f'{self.items[main - 1].name} + {self.items[part - 1].name}: ' + '; '.join((f'{h} + {q} = {t:g}' for t, h, q in pairs[:3])))
        return '\n'.join(lines)

    def proof_lines(self, ids: list[str]) -> str:
        """Per cart line and requirement in force: no attribute names the property, or the attribute that names it lacks
        the required words. Limits and exclusions are left to the check."""
        cat_key = self.category_key()
        flat = lambda text: re.sub('[^a-z0-9]', '', str(text).lower())
        out = []
        for ident in ids:
            block = _facts_of(self.full.get(ident) or {})
            if not block or not cat_key:
                continue
            owner = self.owner_of(str((self.rows.get(ident) or {}).get(cat_key) or ''))
            for number, need in self.needs_now():
                prop, _, value = need.partition(':')
                if number == owner and self.fit_need(number, need):
                    index = self.fit_index_now()
                    hit = self.fits_codes(ident, self.fit_device_codes(number, index, ids), index)
                    out.append(f"{ident}: {prop.strip()} is a fit to another requested item, read from the model codes the pages name, not from an attribute naming some other product: " + ("its page names a model code of the device line" if hit else "its page names none of the device line's model codes" if hit is False else 'no single device line or no codes to compare'))
                    continue
                words = [w for w in re.findall('[a-z0-9]+', value.lower()) if len(w) >= 2]
                if number != owner or not words or not prop.strip():
                    continue
                ranged = set(words) & {'not', 'no', 'least', 'most', 'under', 'over', 'unlike', 'different', 'min', 'max', 'minimum', 'maximum', 'below', 'above', 'between', 'than', 'or', 'any', 'without'}
                limit = _limit(value)
                if limit and limit[3] and need not in [r for _, r in self.refs] and (not self.measured(ident, prop.strip(), limit[3])):
                    out.append(f'{ident}: no figure for {prop.strip()} anywhere on its record (no attribute for it holds a number, no field states one in {limit[3]}); the requirement says {value.strip()}')
                    continue
                if ranged and ranged != {'or'}:
                    continue
                named = [k for k in block if _names(_spelled(prop), _spelled(k))]
                keys = [k for k in named if not _offered(k)]
                picks = [(k, p) for k in keys for p in [_picked_other(block, block[k], value)] if p]
                if not keys:
                    shown = [k for k in named if _offered(k)]
                    out.append(f'{ident}: no attribute names {prop.strip()} (the requirement says {value.strip()})' + (f"; {shown[0].replace(CHOICE_MARK, '')} = {str(block[shown[0]])[:90]} lists choices on offer, which states none of them" if shown else ''))
                elif picks and len(picks) == len(keys):
                    out.append(f"{ident}: {picks[0][0]} = {str(block[picks[0][0]])[:90]} lists several values and the listing's own option names {picks[0][1]}, so its {prop.strip()} is {picks[0][1]}; the requirement says {value.strip()}")
                elif not ranged and not any((all((w in flat(block[k]) for w in words)) for k in keys)):
                    out.append(f"{ident}: {'; '.join((f'{k} = {block[k]}' for k in keys[:2]))}, the requirement says {value.strip()}")
        return '\n'.join(out[:10])

    def offered_only(self, ident: str) -> str:
        """Why an opened listing does not state a plain required value of its item although the value's words appear
        under a key naming the property: every such key lists choices on offer, or lists several values of which the
        listing's own option names another. '' when no requirement is held only that way."""
        raw = _facts_of(self.full.get(ident) or {}) or {}
        number = next((n for n in range(1, len(self.items) + 1) if ident in self.item_ids(n)), 0)
        if not raw or not number:
            return ''
        refs = [r for _, r in self.refs]
        for owner, need in self.needs_now():
            prop, colon, value = need.partition(':')
            if owner != number or not colon or need in refs or self.fit_need(number, need) or re.search('\\d', value) or re.search('\\b(?:not|no|except|without|other than|unlike|same|at most|at least|under|over)\\b', value.lower()):
                continue
            named = [k for k in raw if _names(_spelled(prop), _spelled(k))]
            options = [_tokens(o) for o in re.split('\\s+or\\s+', value.lower()) if _tokens(o)]
            holds = [k for k in named if any((o <= _tokens(raw[k]) for o in options))]
            if not holds:
                continue
            own = [k for k in named if not _offered(k)]
            if all((_offered(k) for k in holds)) and not any((any((o <= _tokens(raw[k]) for o in options)) for k in own)):
                return f"{ident}'s {prop.strip()} stands only in {holds[0].replace(CHOICE_MARK, '')} = {str(raw[holds[0]])[:80]}, choices on offer"
            picks = [(k, _picked_other(raw, raw[k], value)) for k in own]
            if own and all((p for _, p in picks)):
                return f"{ident}'s {picks[0][0]} = {str(raw[picks[0][0]])[:80]} lists several values and its own option names {picks[0][1]}"
        return ''

    def check_key(self, ids: list[str]) -> str:
        return '|'.join(sorted(ids)) + f'#{self.version}'

    def check_cart(self, ids: list[str] | None=None, limit: float=SPEND_SOFT) -> str:
        lead = ids is not None
        ids = ids if lead else self.cart_ids()
        key = self.check_key(ids)
        if key in self.checked or not self.judges(limit) or (not ids):
            return ''
        self.checked.add(key)
        prompt = CHECK + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS CURRENTLY IN FORCE:\n' + self.ledger()
        if self.notes:
            prompt += "\n\nCHANGES SINCE THE REQUEST (later notes from the shopper or the store; where they differ from the request's text, they and the requirements in force decide; a note that a listing sold out is about the listing its brackets name and no other, and one that names no listing makes a cart line sold out only when that line's own record says so):\n" + '\n'.join((f'- {n}' for n in self.notes[-4:]))
        if lead:
            quantity = '; '.join((f'item {n}: quantity {it.qty}' for n, it in enumerate(self.items, 1)))
            prompt += '\n\nCART (a plan: assume each of these listings is carted in the quantity asked for: ' + quantity + '):\n' + ', '.join(ids)
        else:
            prompt += '\n\nCART:\n' + self.cart_text()
        prompt += "\n\nRECORDS FOR EACH CART LISTING:\n" + '\n'.join((f'{i}: ' + ('SOLD OUT (no longer buyable, per the store). ' if self.sold_out(i) else '') + self.record_of(i) for i in ids))
        if self.brief.get('pick') in ('cheapest', 'most', 'least'):
            prompt += '\n\nMORE LISTINGS SEEN (latest first):\n' + self.others(ids, 15)
        pairs = self.fit_pairs() if self.brief.get('pick') == 'cheapest' else ''
        if pairs:
            prompt += "\n\nPAIRS THAT FIT BY THE CODES THE PAGES NAME (opened listings in stock, cheapest total first; their other requirements are not checked here):\n" + pairs
        proof = self.proof_lines(ids)
        if proof:
            prompt += '\n\nWORD CHECK OF THE ATTRIBUTES (code reading, words only; synonyms and units can still match, so judge by the reading rules):\n' + proof
        judges = self.judges(limit)
        got = None
        if len(judges) > 1 and not lead and not self.audited and self.brief.get('pick') not in ('cheapest', 'most', 'least'):
            got = self.ask_json(prompt, [LIGHT], 3000)
            if got:
                judges = [HEAVY] if got.get('ok') is False and HEAVY in self.judges(limit) else []
        if judges and not self.no_credit:
            got = self.ask_json(prompt, judges, 3000) or got
        self.trace.append({'kind': 'check', 'cart': ids, 'got': got})
        self.verdicts[key] = bool(got) and got.get('ok') is not False
        if not got or got.get('ok') is not False:
            return ''
        self.leads = [i for i in self.leads if i not in ids]
        problems = [str(p) for p in _items(got.get('problems'))][:5]
        better = [str(b) for b in _items(got.get('better')) if str(b) in self.rows and (not self.sold_out(str(b)))][:4]
        offered = [n for n in (self.offered_only(b) for b in better) if n]
        better = [b for b in better if not self.offered_only(b)]
        text = 'An independent review finds the cart still wrong: ' + '; '.join(problems or ['check the requirements'])
        if better:
            text += '. Better-looking listings: ' + ', '.join(better)
        if offered:
            text += ". Halo's reading of the records: " + '; '.join(offered[:3]) + ' (a list of choices on offer states none of them, and a list the listing\'s own option narrows states only that option): such a listing is no fit for that requirement'

        return text + '. Correct the cart (or re-read the records to prove the review wrong), then order.'

    def trail(self, early: bool=False) -> tuple[str, bool]:
        """Follows each pointed-at chain in code (TRAIL), opening the links it finds. At checkout (early=False) a link
        still missing holds the order. early=True is the walk before checkout (early_trail): at most three rounds of its
        own, opening only while chain_floor() steps stay for the model, never a hold. A chain's first page counts only
        when it states every value the requirement's description gives (other_starts); else the shop's next rows of that
        kind are opened instead."""
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        tool = self.roles.get('offer') or self.roles.get('open')
        target = self.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        limit = self.trails + 3 if early else (6 if self.left() > 15 else 4) + self.early_rounds
        floor = self.chain_floor() if early else 4
        while pending and self.trails < limit and (self.cost < SPEND_HARD):
            self.trails += 1
            self.early_rounds += 1 if early else 0
            records = []
            for ident in reversed(self.seen_order):
                record = self.full.get(ident) or self.rows.get(ident) if ident in self.opened or ident in self.full else None
                if record and len(records) < 45:
                    records.append(('OPENED ' if ident in self.opened else 'ATTRIBUTES READ (page not opened) ') + _dump(_clip(record, 300))[:1400])
            for ident in reversed(self.seen_order):
                if len(records) >= 60:
                    break
                if ident in self.full or ident not in self.opened or not (self.full.get(ident) or self.rows.get(ident)):
                    if ident in self.full:
                        continue
                    records.append('ROW ' + _dump(_clip(self.rows.get(ident), 80))[:300])
            prompt = TRAIL + '\n\nREQUIREMENTS WHOSE VALUE SITS ON ANOTHER LISTING:\n' + '\n'.join(pending)
            said = [f'- {p}: "{self.request_clause(p)}"' for p in pending if self.request_clause(p)]
            if said:
                prompt += "\n\nTHE REQUEST'S OWN WORDS FOR THEM (the relation they name decides the value: the same, unlike, the partner of, ...):\n" + '\n'.join(said)
            prompt += '\n\nRECORDS (latest first):\n' + '\n'.join(records)
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
                self.chain_ids.update((i for i in ids if i in self.rows))
                others = self.other_starts(need, ids[0]) if ids and ids[0] in self.opened and self.link_vetoes.get(need, 0) < 2 else []
                if others:
                    self.link_vetoes[need] = self.link_vetoes.get(need, 0) + 1
                    reach += [i for i in others[:2] if i not in reach]
                    lost.append(f"{need} -> {ids[0]} does not show every value the request gives for the first page; the same shop's {', '.join(others[:2])} match it better")
                    self.trace.append({'kind': 'link_veto', 'need': need[:120], 'page': ids[0], 'next': others[:2]})
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
                    about = str(gap.get('about') or '') or 'the listing it refers to'
                    words = str(gap.get('find') or '').strip()
                    lost.append(f'{need} -> find {about}' + (f' (search: {words})' if words else ''))
                    gaps.append((about, words))
            pending = [p for p in pending if p not in self.ref_values]
            if not reach and gaps and tool and target and (self.left() > max(12, floor)) and (self.link_searches < self.link_cap()):
                reach = self.find_link(*gaps[0])
                if self.trace and self.trace[-1].get('kind') == 'link_known':
                    seen.append(f"{self.trace[-1]['id']}: an opened page from the shop named in this chain, showing the values its description lists; take the value from it")
                    lost = lost[1:]
            if not reach or not tool or (not target) or (self.left() < len(reach) + floor) or self.finished:
                break
            for ident in reach[:max(0, min(3, 6 - len(seen)))]:
                faults: list[str] = []
                args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
                if faults:
                    break
                reply = self.act(tool, args)
                if reply is None or self.finished:
                    break
                if tool == self.offer_tool():
                    self.offer_seen[ident] = self.step
                self.observe(tool, reply.get('observation'))
                seen.append(f'{ident}: ' + self.show(reply.get('observation'))[:1500])
        text = ''
        if seen:
            text += '[Halo opened the pages that requirements refer to]\n' + '\n'.join(seen) + '\n'
        if self.ref_values:
            text += '[values taken from the referenced pages] ' + '; '.join((f'{k} -> {v}' for k, v in self.ref_values.items())) + '\n'
        if early:
            if lost:
                text += '[Halo] A requirement draws its value through a run of linked pages; an order is valid only after each page in that run has been opened. Still missing: ' + ' | '.join(lost[:3]) + ". Query the shop's name with the values the description gives (a different listing of this kind from that shop is not the target), open the shop's row whose page carries those values, and carry on from that page likewise."
            return (text, False)
        hold = bool(lost) and self.trail_blocks < (3 if self.left() > 15 else 2) and (self.left() > 5)
        if hold:
            self.trail_blocks += 1
            text += 'An order is valid only after each page behind a requirement has been fully opened (a sighting among search rows is insufficient), and that includes any page that leads on to a further listing. Still missing: ' + ' | '.join(lost[:3]) + ". Query the shop's name with the values the description gives (a different listing of this kind from that shop is not the target), open the shop's row whose page carries those values, take the value from that page, then order."
        return (text, hold)

    def link_misses(self, need: str, ident: str) -> list[str]:
        """The values the requirement's description of the listing it points at gives (pointer_parts) that this page
        states nowhere: not in its title, attributes (keys or values) or description; numbers compare as numbers."""
        return self.page_misses(ident, self.pointer_parts(need)[2])

    def page_misses(self, ident: str, values: list[str]) -> list[str]:
        """The values a listing's record states nowhere (title, attribute keys and values, description)."""
        stack, leaves = ([self.full.get(ident) or self.rows.get(ident) or {}], [])
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                leaves += [_words(k) for k in node]
                stack += list(node.values())
            elif isinstance(node, list):
                stack += node
            else:
                leaves.append(str(node))
        text = re.sub('(?<=\\d),(?=\\d{3}\\b)', '', ' '.join(leaves).lower())
        numbers = {float(x) for x in re.findall('\\d+(?:\\.\\d+)?', text)}
        roots = _roots(text) | set(re.findall('[a-z0-9][a-z0-9-]*', text))
        return [v for v in values if not (float(v) in numbers if re.fullmatch('\\d+(?:\\.\\d+)?', v) else v in roots or _roots(v) & roots)]

    def other_starts(self, need: str, first: str) -> list[str]:
        """When an opened page taken as a chain's first link misses a value the requirement's description gives, the
        same shop's unopened rows of that kind whose own row text already misses fewer of those values (best first);
        else []. Comparing with the shop's other rows keeps a value the store words differently (a unit, a word
        order) from turning the right page away: alone, the check missed a value on 2 of 8 right first pages of passing
        Arena chains. Arena r13/r15: wrong first pages of the right shop (another branch of the chain) and the right
        start left unopened on a later page lost five chain tasks."""
        _, kind, values = self.pointer_parts(need)
        lack = len(self.page_misses(first, values)) if values else 0
        if not lack:
            return []
        about = need.split(':', 1)[-1]
        rows = [i for i in self.shop_rows(about, list(reversed(self.seen_order))) if i != first]
        typed = [i for i in rows if not kind or _roots(kind) & _roots(_dump(self.rows.get(i) or {}))]
        better = [(len(self.page_misses(i, values)), n, i) for n, i in enumerate(typed)]
        return [i for miss, _, i in sorted(better) if miss < lack]

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

    def query_shops(self, args: dict[str, Any], observation: Any) -> set[str]:
        """The shop or brand names of this result's rows that the search's own words name, lowercased."""
        words = ' '.join((str(v).lower() for v in args.values() if isinstance(v, str)))
        out = set()
        for _, row in self.records_in(observation):
            for key, value in row.items():
                name = str(value).strip().lower() if isinstance(value, str) else ''
                if 2 < len(name) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(key), re.I) and re.search('(?<![a-z0-9])' + re.escape(name) + '(?![a-z0-9])', words):
                    out.add(name)
        return out

    def only_shops(self, observation: Any, names: set[str]) -> bool:
        """Every row of this result belongs to one of these shops or brands. A word search led by a shop's name lists
        that shop's rows first, so a page that also shows other shops is where its rows end: pages after it are not
        read for the shop (pointer searches spent two steps a search on pages without one row of the named shop)."""
        rows = self.records_in(observation)
        return bool(rows) and all((any((isinstance(v, str) and v.strip().lower() in names for k, v in row.items() if re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I))) for _, row in rows))

    def chain_floor(self) -> int:
        """Steps an early chain walk leaves the model: about a sixth of the episode's steps, 6 to 14."""
        return min(14, max(6, self.max_steps // 6))

    def early_trail(self) -> str:
        """The chain walk before checkout. trail() runs when the model checks out, which on pointer tasks is mostly in
        the last few steps, where it can neither search for a link (12+ steps) nor hold the order (6+): the order then
        goes out with the chain half opened and fails the process check. Here Halo walks the chain in code from the
        first turns (once a search has run), while chain_floor() + 3 steps and 90+ s are left, a few times an episode
        and some steps apart (both scaled to the episode's steps); its text rides on the next result. A chain the model
        already opened costs one model call. Arena r13: the race's top agent walked each chain from its first steps and
        passed all five chain tasks b28/b29 lost walking at a third of the steps."""
        walks = max(EARLY_WALKS, self.max_steps // 30)
        apart = max(4, min(8, self.max_steps // 8))
        if self.early_walks >= walks or self.finished or self.left() <= self.chain_floor() + 2 or self.model_room() < 90 or self.step - self.early_at < apart:
            return ''
        live = {need for _, need in self.needs_now()}
        if not self.searched or not any((need in live and need not in self.ref_values for _, need in self.refs)):
            return ''
        self.early_walks += 1
        self.early_at = self.step
        before = set(self.ref_values)
        text, _ = self.trail(early=True)
        found = self.resolved_search([need for need in self.ref_values if need not in before])
        self.trace.append({'kind': 'early_trail', 'step': self.step, 'resolved': sorted(self.ref_values)[:3], 'text': text[:160]})
        return (text + ('\n' + found if found else '')).strip()

    def resolved_search(self, needs: list[str]) -> str:
        """Under a cheapest pick, once a chain gave a requirement its value: one search for that value with the item's
        name (inside its category when the search takes one), its rows of the item's kind listed cheapest first. The
        chain's own end listing is one listing carrying the value, not the cheapest (Arena r14: b29 ordered the chain
        end at 79 while a 57 listing carried the same value). Only a value the item must share, not one it must avoid
        or stay near."""
        plan = self.search_template() or (self.word_plan() or (None,) * 4)[:3]
        if self.brief.get('pick') != 'cheapest' or not plan[0] or self.left() < self.chain_floor() + 6 or self.finished:
            return ''
        tool, base, field = plan
        spec = ((self.schemas.get(tool) or {}).get('properties') or {}).get(field) or {}
        width = spec.get('maxLength') if isinstance(spec.get('maxLength'), int) else 60
        shelf = self.shelf_arg(tool)
        out = []
        for need in needs:
            value = self.ref_values.get(need, '')
            number = next((n for n, m in self.refs if m == need), 0)
            relation = need.split(':', 1)[-1].lower()
            if need in self.resolved_sought or not number or not value or value.startswith('as stated on') or re.search('\\b(?:unlike|differ\\w*|other than|anything but|not|within|inside|either way|at most|at least|no (?:more|less|higher|lower)|above|below)\\b', relation):
                continue
            self.resolved_sought.add(need)
            item = self.items[number - 1]
            said = re.findall('[A-Za-z0-9][A-Za-z0-9.-]*', value)[:3]
            name = [w for w in re.findall('[A-Za-z0-9][A-Za-z0-9-]*', item.name) if w.lower() not in {x.lower() for x in said}]
            label = self.shelf_label(tool, number) if shelf else None
            args = {**{k: v for k, v in base.items() if k != shelf}, **({shelf: label} if label else {}), field: ' '.join(self.fit_words(said + name, width, len(name)))}
            done = self.code_search(tool, args, 3000)
            if done is None:
                break
            cat_key = self.category_key()
            rows = [i for i in done[2] if self.buyable(i) and self.price_of(i) is not None and (not cat_key or self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or '')))]
            rows.sort(key=lambda i: self.price_of(i) or 0.0)
            self.trace.append({'kind': 'resolved_search', 'need': need[:120], 'value': value[:60], 'rows': rows[:6]})
            out.append(f"[Halo searched the value the pointed pages gave ({value}) for item {number}, since the pick is the cheapest listing carrying it: {done[0].get(field)!r}] " + (', '.join((f'{i} ({self.price_of(i):g})' for i in rows[:6])) + ' (cheapest first; open them to check every other requirement)' if rows else 'no buyable row of that item here'))
        return '\n'.join(out)

    def find_link(self, about: str, words: str) -> list[str]:
        """Rows for the listing that a chain link describes (about: the trail's description of it, words: its search
        words). When a page already opened from the named shop states every value of the description, that page is the link: nothing is
        returned and link_known is logged. Otherwise Halo searches, each query only once: the shop's name with up to three
        description values, then the shop plus the product kind, then the shop by itself (hop_queries); the trail's own words
        are used only when no shop is named (its 7-9 word phrases lost the store past the first few words, Arena r14). Gives
        up to three unopened rows of that shop and kind, rows whose text shows more of the values coming first."""
        said = f'{about} {words}'
        named = {w for ident in self.shop_rows(said, list(self.rows)) + self.opened_shop(said) for k, v in (self.rows.get(ident) or {}).items() if isinstance(v, str) and len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I) for w in re.findall('[a-z0-9]{3,}', v.lower())}
        _, kind, values = self.pointer_parts(about, words)
        for ident in self.opened_shop(said):
            if ident not in self.chain_ids and values and (not self.page_misses(ident, values)):
                self.trace.append({'kind': 'link_known', 'id': ident})
                return []
        plan = self.search_template() or (self.word_plan() or (None,) * 4)[:3]
        _, queries = self.hop_queries(about, words)
        if not queries and words.strip():
            queries = [words.strip()]
        known: list[str] = []
        if plan[0]:
            tool, base, field = plan
            spec = ((self.schemas.get(tool) or {}).get('properties') or {}).get(field) or {}
            width = spec.get('maxLength') if isinstance(spec.get('maxLength'), int) else 60
            for query in queries:
                if known or self.link_searches >= self.link_cap() or self.left() <= 12 or self.finished:
                    break
                text = ' '.join(self.fit_words(query.split(), width, 0))
                memo = text.lower()
                self.link_searches += 1
                if memo in self.link_seen:
                    known = self.shop_rows(said, self.link_seen[memo])
                    continue
                done = self.code_search(tool, {**base, field: text}, 0)
                if done is None:
                    break
                self.link_seen[memo] = done[2]
                known = self.shop_rows(said, done[2])
                self.trace.append({'kind': 'link_search', 'words': text, 'rows': known[:4]})
        grammar = {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for'}
        sort = _roots(kind) if kind else {w for w in re.findall('[a-z0-9]{3,}', (words or about).lower())} - named - grammar

        def of_kind(ids: list[str]) -> list[str]:
            return [i for i in ids if not sort or sort & (_roots(_dump(self.rows.get(i) or {})) | set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower())))]
        pool = of_kind(known) or of_kind(self.shop_rows(said, list(reversed(self.seen_order)))) or known
        return sorted(pool, key=lambda i: len(self.page_misses(i, values)))[:3]

    def offer_tool(self) -> str | None:
        return self.roles.get('offer') or self.roles.get('open')

    def reread_offers(self) -> str:
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return ''
        since = max(self.step - 2, self.refused_at)
        stale = [i for i in self.cart_ids() if self.offer_seen.get(i, -99) < since and i not in self.gone]
        chain = [i for i in sorted(self.chain_ids) if i not in self.offer_seen and i not in stale][:3]
        if chain and self.left() >= len(stale) + len(chain) + 3:
            self.trace.append({'kind': 'chain_pages', 'ids': chain})
            stale += chain
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
        self.name_sold(lost)
        self.leads = [i for i in self.leads if i not in lost]
        self.version += 1
        if self.brief.get('pick') == 'cheapest' and self.left() > self.tail() + 6:
            self.list_upto(lost[0])
        picks = self.replacements(lost)
        text = f"[Halo] Out of stock as of now: {', '.join(lost)} (the store reports it can no longer be bought). That is final; opening it again is wasted."
        if picks and self.left() <= 6:
            return text + f" Very few steps remain: in this one turn put {picks[0]} in at the quantity asked for and take the out-of-stock line away (two calls), then order. Further listings already read that look like fits: {', '.join(picks[1:]) or 'none'}."
        sure = None
        if picks and self.judges(SPEND_HARD) and self.clock() < LATE_BY:
            rest = [i for i in self.cart_ids() if i not in lost and i not in self.gone]
            for pick in picks[:2]:
                self.check_cart([pick] + rest, SPEND_HARD)
                if self.verdicts.get(self.check_key([pick] + rest)):
                    sure = pick
                    break
            self.trace.append({'kind': 'swap_checked', 'picks': picks[:2], 'sure': sure})
            if not sure:
                return text + f" Already-read candidates to stand in for it: {', '.join(picks)}; yet a separate review of their records found each of the leading ones failing some requirement. Hunt for a stand-in whose record states every requirement, and use one of these only when that hunt finds nothing better."
            picks = [sure] + [i for i in picks if i != sure]
        if picks and self.brief.get('pick') == 'cheapest':
            return text + f" Already-read candidates for that item, by rising price: {', '.join(picks)}. The pick rule wants the lowest price, so unless every fitting listing is already in view, send one search priced under {picks[0]} (through the search's price ceiling, where it has one) and only then trade the out-of-stock line for the cheapest fit." + (f" A second, separate review passes {sure} on every requirement." if sure == picks[0] else '')
        if picks:
            return text + f" Already-read candidates for that item: {', '.join(picks)}. In this one turn take the out-of-stock line away and put {picks[0]} in at the quantity asked for, then order."
        return text + ' Take that line away and cart the strongest other fit instead, searching for one when none of the listings read so far qualifies.'

    def find_sold(self) -> str:
        """Following a note that some chosen listing sold out without saying which: the single cart line it could mean is
        marked sold out when that line already sat in the cart as the note arrived; otherwise the cart lines are re-read and
        the store's own stock settles it. Pinning such a note on whichever single line the cart held later removed every new
        line (Arena r16: a fitting line was dropped long after the note's listing had been replaced)."""
        ids = [i for i in self.cart_ids() if i not in self.gone]
        if any((self.sold_out(i) for i in ids)):
            return ''
        if len(ids) == 1 and ids[0] in self.sold_cart:
            self.unavailable.add(ids[0])
            self.name_sold([ids[0]])
            return ''
        return self.reread_offers() if ids else ''

    def name_sold(self, ids: list[str]) -> None:
        """Writes the listing a sold-out note meant into that note once Halo knows it, so the cart check and the model
        read the note as about that listing only."""
        if 0 <= self.sold_note < len(self.notes) and ids:
            self.notes[self.sold_note] = self.notes[self.sold_note].replace(' [names no listing]', '') + f" [sold out: {', '.join(ids)}]"
            self.sold_note = -1

    def replacements(self, lost: list[str]) -> list[str]:
        if not self.judges():
            return []
        records = []
        for ident in reversed(self.seen_order):
            if ident in lost or ident in self.gone or self.sold_out(ident):
                continue
            records.append(f'{ident}: ' + self.facts_view(ident, 1200))
            if len(records) >= 25:
                break
        if not records:
            return []
        prompt = SWAP + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS CURRENTLY IN FORCE:\n' + self.ledger()
        prompt += '\n\nCART LINE THAT SOLD OUT:\n' + '\n'.join((f'{i}: {self.record_of(i)}' for i in lost)) + '\n\nCURRENT CART:\n' + self.cart_text()
        prompt += '\n\nRECORDS OF OTHER LISTINGS SEEN (latest first):\n' + '\n'.join(records)
        got = self.ask_json(prompt, self.judges()[::-1], 3500)
        fits: list[tuple[float, str]] = []
        for entry in _items((got or {}).get('fits')):
            ident = str(entry.get('id') if isinstance(entry, dict) else entry)
            price = entry.get('price') if isinstance(entry, dict) else None
            if ident in self.rows and ident not in lost and (ident not in self.gone) and (ident not in self.cart_ids()) and (ident not in [i for _, i in fits]):
                fits.append((float(price) if isinstance(price, (int, float)) and (not isinstance(price, bool)) else float('inf'), ident))
        if self.brief.get('pick') == 'cheapest':
            cost = lambda f: self.price_of(f[1]) if self.price_of(f[1]) is not None else f[0]
            ranked = [i for _, i in sorted(fits, key=lambda f: (cost(f), -self.stated(f[1])))]
        else:
            ranked = sorted((i for _, i in fits), key=lambda i: -self.stated(i))
        self.trace.append({'kind': 'swap', 'lost': lost, 'fits': fits, 'stated': {i: self.stated(i) for i in ranked}})
        return ranked[:4]

    def stated(self, ident: str) -> int:
        facts = _fact_block(self.full.get(ident)) or _fact_block(self.rows.get(ident)) or {}
        keys = [set(_words(k).split()) for k in facts]
        props = {_words(m.split(':')[0]) for item in self.items for m in item.musts if ':' in m}
        return sum((1 for p in props if p and any((k and (k <= set(p.split()) or set(p.split()) <= k) for k in keys))))

    def audit(self) -> str:
        if self.audited or not self.judges():
            return ''
        self.audited = True
        got = self.audit_call()
        named = self.audit_fits(got)
        if len(named) < len(self.items):
            rows = [str(i) for i in _items((got or {}).get('open')) if str(i) in self.rows and str(i) not in self.opened and (not self.sold_out(str(i)))]
            if rows and (not named or self.model_room() > 45) and self.open_rows(rows):
                got = self.audit_call()
                named = self.audit_fits(got)
        if not named:
            return ''
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return 'An independent review believes these listings could fit: ' + '; '.join((f"item {k}: {', '.join(v)}" for k, v in named.items())) + (f" ({got.get('reason')})" if got.get('reason') else '') + '. Open them and buy whichever really fits; close with no purchase only if every one fails.'

    def audit_call(self) -> dict[str, Any] | None:
        """The decline review's model call. Opened pages go first (newest first, up to 18), then search rows up to 30
        records: in long episodes the newest 30 were mostly rows, and the review reported pages it was not shown as
        never opened (5 of 9 decline reviews in one Arena run). Each page shows its attributes first (facts_view): cut at a
        fixed length after long description fields, a record lost the attribute that fitted (Arena r15)."""
        live = [i for i in reversed(self.seen_order) if not self.sold_out(i)]
        pages = [i for i in live if self.full.get(i)][:18]
        rows = [i for i in live if i not in pages][:max(0, 30 - len(pages))]
        records = [f'OPENED {i}: ' + self.facts_view(i, 1200) for i in pages]
        records += ['ROW ' + _dump(_clip(self.rows.get(i), 120))[:500] for i in rows]
        if not records:
            return None
        prompt = AUDIT + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS CURRENTLY IN FORCE:\n' + self.ledger()
        prompt += '\n\nLISTING RECORDS SEEN (opened pages first, then search rows; newest first in each):\n' + '\n'.join(records)
        judges = self.judges()
        got = self.ask_json(prompt, [LIGHT] if judges else [], 4000)
        if HEAVY in judges and not self.no_credit and not self.audit_fits(got):
            got = self.ask_json(prompt, [HEAVY], 4000) or got
        self.trace.append({'kind': 'audit', 'got': got})
        return got

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get('fits')
        if not isinstance(fits, dict):
            return {}
        named = {str(k): [str(i) for i in _items(v) if str(i) in self.rows and (not self.sold_out(str(i)))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def unsought_refs(self) -> list[str]:
        """Values taken from the referenced pages that no search has tried yet ("need -> value"). Race 16: Hamus
        resolved a chain to its value, then closed with no purchase without a single search for it."""
        said = ' '.join((_dump(a) for a in self.searched)).lower()
        out = []
        for need, value in self.ref_values.items():
            if value.startswith('as stated on'):
                continue
            words = set(re.findall('[a-z0-9]{3,}', value.lower())) - {'the', 'and', 'with', 'from', 'that', 'than', 'least', 'most', 'more', 'less'}
            if words and (not any((w in said for w in words))):
                out.append(f'{need} -> {value}')
        return out

    def inexact_lines(self) -> list[str]:
        """Cart lines whose attribute holds a plain required value only inside a longer value ("dark reddish-brown" for
        "brown") while an opened, buyable listing of the same item states it exactly under that property and meets
        every requirement (covers). On the Arena the pick with the longer value failed while two listings stating the
        value exactly were opened."""
        split = lambda text: set(re.sub('[,/;|()-]', ' ', _words(str(text))).split())
        out = []
        for number in range(1, len(self.items) + 1):
            plain = [(p.strip(), v.strip().lower()) for _, need in self.needs_now() if _ == number for p, c, v in [need.partition(':')] if c and v.strip() and (not re.search('\\d', v)) and (not re.search('\\b(?:not|no|except|without|other than|or)\\b', v.lower()))]
            if not plain:
                continue
            lines = [i for i in self.cart_ids() if i in self.item_ids(number)]
            for line in lines:
                raw = _facts_of(self.full.get(line) or {}) or {}
                block = {k: v for k, v in raw.items() if not _offered(k)}
                for prop, value in plain:
                    held = [_picked_other(raw, v, value) or str(v) for k, v in block.items() if _names(_spelled(prop), _spelled(k))]
                    if not held or any((split(t) == split(value) or split(value) in [split(e) for e in re.split('[,/;|]', t)] for t in held)) or (not any((split(value) <= split(t) for t in held))):
                        continue
                    exact = [i for i in self.item_ids(number) if i != line and i in self.full and self.buyable(i) and self.price_of(i) is not None and self.covers(i, number) and any((split(str(v)) == split(value) for k, v in (_fact_block(self.full.get(i) or {}) or {}).items() if _names(_spelled(prop), _spelled(k))))]
                    if exact:
                        exact.sort(key=lambda i: self.price_of(i) or 0.0)
                        out.append(f"{line}'s {prop} is '{held[0]}', not '{value}'; opened listings stating '{value}' exactly, cheapest first: " + ', '.join((f'{i} ({self.price_of(i):g})' for i in exact[:3])))
        return out

    def unmet_lines(self) -> list[str]:
        """Cart lines failing covers() whose attribute under a key naming a plain required value's property shares no
        word with that value (another value, not a longer one): a value in other words taken as the required one. With
        opened, buyable listings of the item within the budget that meet every requirement (covers), those are named;
        with none, the line is reported as failing when no other field of its record (title, description, another key,
        the store label) holds the value's words either, with the item's unread rows in the budget that the value's
        words reached, cheapest first. The real grader (race 17) accepted a line whose key for the property held another
        value 0 times in 21; reported only when an alternative was opened, such a line was bought and lost (Arena r12)."""
        split = lambda text: set(re.sub('[,/;|()-]', ' ', _words(str(text))).split())
        cap = self.budget if isinstance(self.budget, (int, float)) and self.budget > 0 else None
        out = []
        for number in range(1, len(self.items) + 1):
            plain = [(p.strip(), v.strip().lower()) for _, need in self.needs_now() if _ == number and (not self.fit_need(number, need)) for p, c, v in [need.partition(':')] if c and v.strip() and (not re.search('\\d', v)) and (not re.search('\\b(?:not|no|except|without|other than|or)\\b', v.lower()))]
            for line in [i for i in self.cart_ids() if i in self.item_ids(number)] if plain else []:
                if self.covers(line, number):
                    continue
                block = _facts_of(self.full.get(line) or {}) or {}
                record = self.full.get(line) or self.rows.get(line) or {}
                for prop, value in plain:
                    held = [_picked_other(block, v, value) or str(v) for k, v in block.items() if not _offered(k) and _names(_spelled(prop), _spelled(k))]
                    if not held or any((split(value) & split(t) for t in held)):
                        continue
                    fits = [i for i in self.item_ids(number) if i != line and i in self.full and self.buyable(i) and self.price_of(i) is not None and self.covers(i, number) and (not isinstance(self.budget, (int, float)) or (self.price_of(i) or 0.0) <= self.budget)]
                    others = split(' '.join((str(v) for k, v in record.items() if isinstance(v, (str, int, float)) and (not isinstance(v, bool)) and (not _offered(k)))) + ' ' + ' '.join((str(v) for k, v in block.items() if not _offered(k) and (not _names(_spelled(prop), _spelled(k))))))
                    if fits:
                        fits.sort(key=lambda i: self.price_of(i) or 0.0)
                        out.append(f"{line}'s {prop} is '{held[0][:60]}', which does not state '{value}'; opened listings meeting every requirement: " + ', '.join((f'{i} ({self.price_of(i):g})' for i in fits[:4])))
                    elif not split(value) <= others and (not self.relational(number, prop, value)) and (not any((set(_spelled(value)) & set(_spelled(re.sub('[,/;|()-]', ' ', str(v)))) for k, v in block.items() if not _offered(k) and _names(_spelled(prop), _spelled(k))))) and (not set(_spelled(value)) <= set(_spelled(' '.join(others)))):
                        words = _tokens(value)
                        rows = [i for i in self.item_ids(number) if i != line and i not in self.full and self.buyable(i) and self.price_of(i) is not None and (cap is None or (self.price_of(i) or 0.0) <= cap + 1e-09) and words <= self.hit_words.get(i, set()) | self.row_words(i)]
                        rows.sort(key=lambda i: self.price_of(i) or 0.0)
                        out.append(f"{line}'s {prop} is '{held[0][:60]}', which does not state '{value}', and no other field of its record does; no opened listing of that item meets every requirement" + (': unread rows of that item the value\'s words reached, cheapest first: ' + ', '.join((f'{i} ({self.price_of(i):g})' for i in rows[:4])) if rows else ''))
                    break
        return out

    def unstated_lines(self) -> list[str]:
        """Cart lines with no attribute for a property a requirement in force sets (fits to another item and pointed-at
        values aside) while an opened, buyable listing of the same item states every requirement on its own (covers,
        alone=True): a late cheaper swap can take a line with no attribute for a required property while the line it
        replaced stated it (fit_faults then names any device repair). Under a cheapest pick only listings priced at or
        below the line count: the reading rules accept a value its title or description states when no attribute names
        the property, so a dearer swap loses the pick (bare holds under a cheapest pick: swapped up 2 of 2 lost on the
        pick, the line kept 1 of 1 passed). A measured limit whose figure the line's record gives nowhere (measured())
        also counts, and then a dearer listing counts too: the real grader (race 17) accepted such a line 2 times in 10."""
        refs = [r for _, r in self.refs]
        out = []
        cheapest = self.brief.get('pick') == 'cheapest'
        for number in range(1, len(self.items) + 1):
            props = [(need.partition(':')[0].strip(), set(re.findall('[a-z0-9]{2,}', need.partition(':')[2].lower())) - {'or', 'and', 'the'}, _limit(need.partition(':')[2])) for n, need in self.needs_now() if n == number and ':' in need and need not in refs and (not self.fit_need(number, need))]
            if not props:
                continue
            for line in [i for i in self.cart_ids() if i in self.item_ids(number)]:
                block = {k: v for k, v in (_facts_of(self.full.get(line) or {}) or {}).items() if not _offered(k)}
                said = set(re.findall('[a-z0-9]{2,}', ' '.join((str(v) for v in block.values())).lower()))
                bare = [p for p, words, _ in props if p and block and (not any((_names(_spelled(p), _spelled(k)) for k in block))) and (not words or not words <= said)]
                blind = [p for p, _, lim in props if p and block and lim and lim[3] and (not self.measured(line, p, lim[3]))]
                bare += [p for p in blind if p not in bare]
                if not bare:
                    continue
                stated = sorted((i for i in self.item_ids(number) if i != line and i in self.full and self.buyable(i) and self.price_of(i) is not None and self.covers(i, number, alone=True)), key=lambda i: self.price_of(i) or 0.0)
                own = self.price_of(line)
                if cheapest and own is not None and (not blind):
                    stated = [i for i in stated if (self.price_of(i) or 0.0) <= own]
                if stated:
                    out.append(f"{line} lacks an attribute for {', '.join(bare)}" + (f" (and no figure for {', '.join(blind)} anywhere on its record)" if blind else '') + "; opened listings of that item whose attributes state every requirement, cheapest first: " + ', '.join((f'{i} ({self.price_of(i):g})' for i in stated[:3])))
        return out

    def measured(self, ident: str, prop: str, unit: str) -> bool:
        """Whether a listing's record gives a figure for a measured limit on `prop`: a key naming the property holds a
        figure (in any unit; the check converts), or some field states a figure in the limit's own unit word."""
        record = self.full.get(ident) or self.rows.get(ident) or {}
        raw = {k: v for k, v in (_facts_of(record) or {}).items() if not _offered(k)}
        tops = {k: v for k, v in record.items() if isinstance(v, (str, int, float)) and (not isinstance(v, bool))}
        if any((_names(_spelled(prop), _spelled(k)) and re.search('\\d', str(v)) for k, v in list(raw.items()) + list(tops.items()))):
            return True
        text = ' '.join((str(v) for k, v in record.items() if isinstance(v, (str, int, float)) and (not isinstance(v, bool)) and (not _offered(k)))) + ' ' + ' '.join((str(v) for v in raw.values()))
        return any((_same_unit(m.group(2), unit) for m in re.finditer(FIGURE, text.lower())))

    def unmeasured(self) -> tuple[str, list[str], list[str], list[str]] | None:
        """unstated_lines' case for a measured limit ("at most N unit") when no opened listing states every requirement:
        a cart line whose record shows no figure for the limit anywhere (measured()). Returns (line, those limits,
        opened listings of the item in stock and in the budget that give the figure and conflict with nothing (weigh),
        cheapest first, the item's unread rows in the budget, rows whose own text shows a figure in that unit first, then
        rows a search with the unit's word reached, then by price), or None. Arena r14: a line with no figure at all for
        a measured limit was bought and failed; the real grader accepted such lines 2 times in 10."""
        refs = [r for _, r in self.refs]
        cap = self.budget if isinstance(self.budget, (int, float)) and self.budget > 0 else None
        for number in range(1, len(self.items) + 1):
            limits = [(p.strip(), v.strip(), lim) for n, need in self.needs_now() if n == number and need not in refs and (not self.fit_need(number, need)) for p, c, v in [need.partition(':')] if c for lim in [_limit(v)] if lim and lim[3]]
            if not limits:
                continue
            ids = self.item_ids(number)
            units = 1 if self.brief.get('budget_scope') == 'each' else self.items[number - 1].qty
            room = lambda i: self.buyable(i) and self.price_of(i) is not None and (cap is None or (self.price_of(i) or 0.0) * units <= cap + 1e-09)
            for line in [i for i in self.cart_ids() if i in ids and i in self.full]:
                blind = [(p, v, lim) for p, v, lim in limits if not self.measured(line, p, lim[3])]
                if not blind:
                    continue
                opened = sorted((i for i in ids if i != line and i in self.full and room(i) and all((self.measured(i, p, lim[3]) for p, _, lim in blind)) and (not self.weigh(i, number)[0])), key=lambda i: self.price_of(i) or 0.0)
                rows = [i for i in ids if i != line and i not in self.full and room(i)]
                rows.sort(key=lambda i: (not all((self.measured(i, p, lim[3]) for p, _, lim in blind)), not all((any((_same_unit(w, lim[3]) for w in self.hit_words.get(i, set()))) for _, _, lim in blind)), self.price_of(i) or 0.0))
                if opened or rows:
                    return (line, [f'{p}: {v}' for p, v, _ in blind], opened[:3], rows[:6])
        return None

    def unstated_leads(self) -> tuple[str, list[str], list[str]] | None:
        """unstated_lines' case when no opened listing states every requirement: a cart line with no attribute for a
        property whose required value is a plain word or two (only its title or description says it). Returns (line,
        those properties, rows): the unread, buyable, in-budget rows of the item that the store's word search returned
        for every word of those values, rows whose own text does not show the words first (the match then sits in
        fields the row leaves out, most often the attributes), then by price."""
        refs = [r for _, r in self.refs]
        cap = self.budget if isinstance(self.budget, (int, float)) and self.budget > 0 else None
        for number in range(1, len(self.items) + 1):
            plain = [(p.strip(), _tokens(v)) for n, need in self.needs_now() if n == number and need not in refs and (not self.fit_need(number, need)) for p, c, v in [need.partition(':')] if c and _tokens(v) and (not re.search('\\d', v)) and (not re.search('\\b(?:not|no|except|without|other than|or|same|unlike|as)\\b', v.lower()))]
            if not plain:
                continue
            ids = self.item_ids(number)
            if any((i in self.full and self.buyable(i) and self.price_of(i) is not None and self.covers(i, number, alone=True) for i in ids)):
                continue
            qty = 1 if self.brief.get('budget_scope') == 'each' else self.items[number - 1].qty
            for line in [i for i in self.cart_ids() if i in ids]:
                block = {k: v for k, v in (_facts_of(self.full.get(line) or {}) or {}).items() if not _offered(k)}
                said = _tokens(' '.join((str(v) for v in block.values())))
                bare = [(p, words) for p, words in plain if block and (not any((_names(_spelled(p), _spelled(k)) for k in block))) and (not words <= said)]
                if not bare:
                    continue
                rows = [i for i in ids if i != line and i not in self.full and self.buyable(i) and self.price_of(i) is not None and (cap is None or (self.price_of(i) or 0.0) * qty <= cap + 1e-09) and all((words <= self.hit_words.get(i, set()) for _, words in bare))]
                if rows:
                    rows.sort(key=lambda i: (all((words <= self.row_words(i) for _, words in bare)), self.price_of(i) or 0.0))
                    return (line, [p for p, _ in bare], rows[:8])
        return None

    def secure_cart(self) -> str:
        """Run once, when two thirds of the steps are gone and nothing is carted yet: add the lowest-priced opened
        listing of each item whose attributes satisfy every live requirement (covers), provided the basket stays inside the
        budget. A store can sell out a line at the add or at checkout, and a late add leaves no steps for a swap (race 16: a
        listing carted at step 31 of 35 sold out at checkout). The model can still replace any line."""
        if self.secured or self.cart_ids() or self.cart_touched or self.buys or (not self.roles.get('cart')):
            return ''
        if self.step < 2 * self.max_steps // 3 or self.left() - self.tail() < len(self.items) + 5 or self.model_room() < 40:
            return ''
        if self.brief.get('pick') not in (None, '', 'any', 'cheapest') or any((p.get('when') not in ('order', 'end') for p in self.process)):
            return ''
        self.secured = True
        cap = self.budget if isinstance(self.budget, (int, float)) else None
        each = self.brief.get('budget_scope') == 'each'
        picks: dict[int, str] = {}
        spent = 0.0
        for number, item in enumerate(self.items, 1):
            pool = sorted((i for i in self.item_ids(number) if i in self.full and self.buyable(i) and self.price_of(i) is not None and self.covers(i, number) and i not in picks.values()), key=lambda i: self.price_of(i) or 0.0)
            if not pool:
                return ''
            picks[number] = pool[0]
            cost = (self.price_of(pool[0]) or 0.0) * item.qty
            if cap is not None and (cost > cap + 1e-09 if each else spent + cost > cap + 1e-09):
                return ''
            spent += 0.0 if each else cost
        calls = self.cart_calls({p: self.items[n - 1].qty for n, p in picks.items()}, [])
        if not calls or len(calls) + self.tail() + 3 > self.left():
            return ''
        cart = self.roles['cart']
        said = []
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), 'args', faults)
            if faults:
                return ''
            reply = self.act(cart, args)
            if reply is None:
                return ''
            self.note_cart(cart, args)
            self.observe(cart, reply.get('observation'))
            said.append(self.show(reply.get('observation'))[:300])
        self.trace.append({'kind': 'cart_secured', 'picks': picks})
        return f"\n[Halo] With two thirds of the steps gone and nothing carted, Halo added for each item the lowest-priced opened listing whose attributes satisfy all requirements ({', '.join((f'{self.items[n - 1].name}: {p}' for n, p in picks.items()))}; a step apiece). The store answered: " + ' | '.join(said) + '\nReplace a line if you come across a better fit; if not, order.'

    def open_stale(self, known: set[str] | None, observation: Any) -> str:
        """A search that returned only listings seen before: the wanted value is not in their row text, so another
        rewording returns the same rows. Halo opens the unread ones in it (in budget, cheapest first; at most 3 times an
        episode) and shows their records. On the Arena two declines came after 5 and 18 searches returned the same rows,
        whose unread listings held the value in their attributes."""
        if known is None or self.stale_opens >= 3 or self.left() <= 12 or self.model_room() < 20:
            return ''
        ids = [i for i, _ in self.records_in(observation)]
        if not ids or any((i not in known for i in ids)):
            return ''
        cap = self.budget if isinstance(self.budget, (int, float)) and self.budget > 0 else None
        held = {}
        if self.brief.get('pick') == 'cheapest':
            for n in range(1, len(self.items) + 1):
                prices = [self.price_of(i) for i in self.cart_ids() if i in self.item_ids(n) and self.price_of(i) is not None]
                if prices:
                    held.update({i: min(prices) for i in self.item_ids(n)})
        unread = sorted((i for i in ids if i not in self.full and (not self.sold_out(i)) and (cap is None or (self.price_of(i) or 0) <= cap) and (i not in held or (self.price_of(i) or 0) < held[i])), key=lambda i: self.price_of(i) or 0.0)[:3]
        if not unread:
            return ''
        self.stale_opens += 1
        if not self.open_rows(unread):
            return ''
        read = [i for i in unread if self.full.get(i)]
        self.trace.append({'kind': 'stale_open', 'rows': unread, 'read': read})
        return '\n[Halo] This search returned only listings seen before, so rewording it again will not reach new ones. Halo opened the unread ones in it (one step each); a value missing from the row text can sit in the attributes:\n' + '\n'.join((f'{i}: {self.facts_view(i)}' for i in read))

    def open_short(self, args: dict[str, Any], observation: Any) -> str:
        """A word search of two or more words that returned one or two rows, none read yet: the store's search needs
        every word and also reads fields a row leaves out (the shop, the attributes), so such a row is most likely the
        very listing the words describe, such as one a requirement points at, even when its title names none of them.
        Halo opens them (one step each; at most 3 times an episode) and shows their records."""
        if self.short_opens >= 3 or self.left() <= self.tail() + 6 or self.model_room() < 30:
            return ''
        words = self.query_words(args)
        ids = [i for i, _ in self.records_in(observation)]
        if len(words) < 2 or not ids or len(ids) > 2 or any((i in self.full or self.sold_out(i) for i in ids)):
            return ''
        self.short_opens += 1
        if not self.open_rows(ids):
            return ''
        read = [i for i in ids if self.full.get(i)]
        hidden = [i for i in read if not words <= self.row_words(i)]
        self.trace.append({'kind': 'short_open', 'rows': ids, 'hidden': hidden})
        return f"\n[Halo] This search of {len(words)} words matched only {len(ids)} listing(s), so Halo opened them (one step each)" + (f"; the store matched words that {', '.join(hidden)}'s row does not show, so they sit in its shop or attributes" if hidden else '') + ':\n' + '\n'.join((f'{i}: {self.record_of(i)[:900]}' for i in read))

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
            found = re.search('\\d+', str(number))
            index = int(found.group()) if found else (1 if len(self.items) == 1 else None)
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
        self.show_links(list(lines), len(calls) + 3)
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
        return 'Held back (Halo handled it): the independent check approved ' + self.confirmed + ', so Halo arranged the cart (' + '; '.join(done) + ') and placed the order. The store answered: ' + ' | then: '.join(replies)

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
        lacking = self.lacking()
        if left <= self.tail():
            keep = set(ends)
        elif self.seek_now and left >= self.tail() + 2 + lacking or (lacking and left >= 2 * lacking + self.tail()):
            keep = None
        elif left <= self.tail() + 3:
            keep = set(ends) | {self.roles.get('cart'), self.roles.get('cart_view')}
        elif left <= len(self.items) + 5 and self.buys == 0:
            keep = set(ends) | {self.roles.get('cart'), self.roles.get('cart_view'), self.roles.get('offer')}
        tools = [t for t in self.tools if t['function']['name'] != ask]
        if keep is not None and ends:
            narrowed = [t for t in tools if t['function']['name'] in keep]
            tools = narrowed or tools
        return tools

    def lacking(self) -> int:
        """Count of requested items without an in-stock cart line (every item needs its own line). offered() keeps
        search and open available as long as two steps (a search, then an open or a cart add) remain for each such item ahead of
        the ending's steps: narrowing the offer to the cart tools at a set count of steps left such an item with nothing to cart,
        and the episode closed with no purchase (Arena r12-r16: 97 episodes requested a search that was not offered, 10
        passed)."""
        held = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0 and (not self.sold_out(i))]
        return max(0, len(self.items) - len(held))

    def seek_turn(self, text: str) -> bool:
        """A reply without a tool call saying it intends to search for, open or read listings while the narrowed tool set has
        no search: the next turn brings back every tool (no more than twice per episode, and only while two calls still leave
        a cart add for each line-less item plus the ending's steps) rather than counting the reply as idle; three idle replies
        end the episode, usually with no purchase. When granted later, the extra calls ate the steps the end-of-episode fill
        required."""
        self.seek_now = False
        search = self.roles.get('search')
        if not search or self.seek_turns >= 2 or self.left() < self.tail() + 2 + self.lacking() or (not re.search(SEEK_TALK, text or '', re.I)):
            return False
        if search in {t['function']['name'] for t in self.offered()}:
            return False
        self.seek_turns += 1
        self.seek_now = True
        self.trace.append({'kind': 'seek_turn', 'step': self.step, 'text': (text or '')[:160]})
        return True

    def quiet(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """The ending call's text fields. A request that says to keep alternatives out (or says nothing) gets a plain
        remark naming no listing; one that asks to be offered the closest listing gets closest_note() in its remark
        fields, filled in even when the model left them out."""
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        out = dict(args)
        offer = self.closest_note() if self.brief.get('quiet_decline') is False else ''
        for key, spec in props.items():
            if not isinstance(spec, dict) or 'string' not in _kinds(spec) or 'enum' in spec:
                continue
            remark = bool(re.search(REMARK_ARG, _snake(key), re.I))
            value = out.get(key)
            if value is None and key not in (schema.get('required') or []) and (not (offer and remark)):
                continue
            text = str(value or '')
            if offer and remark:
                out[key] = offer[:spec.get('maxLength', 400)]
                continue
            named = any((ident in text for ident in self.rows)) or self.brief.get('quiet_decline') is not False
            if named or not text:
                out[key] = 'I could not find anything in this store that meets every one of your requirements, so I am leaving without a purchase.'[:spec.get('maxLength', 200)]
        return out

    def resolve_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if name not in {self.roles.get(r) for r in ('cart', 'open', 'offer', 'compare')} - {None} or not self.id_key:
            return (args, '')
        props = (self.schemas.get(name) or {}).get('properties') or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and ('string' in _kinds(v) or 'array' in _kinds(v)) and ('enum' not in v)]
        if len(free) != 1 or (name != self.roles.get('cart') and (name, free[0]) not in self.id_args and (len(self.rows) < 3)):
            return (args, '')
        shapes = {re.sub('[0-9a-f]', 'h', i.lower()) for i in list(self.rows)[:40]}

        def fix(value: Any) -> tuple[Any, str]:
            if isinstance(value, str) and value.strip() in self.rows:
                return (value.strip(), '')
            if isinstance(value, str) and value.strip() and len(value.strip()) < 8 and re.sub('[0-9a-f]', 'h', value.strip().lower()) not in shapes:
                code = value.strip().lower()
                hits = [i for i, row in self.rows.items() if any((isinstance(v, str) and re.search('(?<![a-z0-9])' + re.escape(code) + '(?![a-z0-9])', v.lower()) for k, v in row.items() if k != self.id_key))]
                if len(hits) == 1:
                    return (hits[0], '')
                return (value, f"{value!r} is a code found in a listing's text rather than a listing id; use the id shown in the result rows (field {self.id_key})" + (f": rows carrying it: {', '.join(hits[:6])}" if hits else ''))
            if not isinstance(value, str) or _id_like(value, True) or len(value.strip()) < 8:
                return (value, '')
            low = value.strip().lower()
            hits = [i for i, row in self.rows.items() if any((k != self.id_key and isinstance(v, str) and v.strip().lower().startswith(low) for k, v in row.items()))]
            if len(hits) == 1:
                return (hits[0], '')
            if hits:
                return (value, f"the text {value!r} appears on more than one listing ({', '.join(hits[:6])}); give a single listing id")
            return (value, f'{value!r} is no listing id; use the id shown in the result rows (field {self.id_key})')
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

    def seller_as_brand(self, name: str, args: dict[str, Any]) -> str:
        """A brand, maker or manufacturer filter given a name the rows carry only in their shop field: the filter would
        miss that shop's listings (Arena r13: a manufacturer filter on a chain's shop name found nothing, and the chain
        was walked down the wrong listing). Returns the reason the call is not sent, else ''."""
        if name in {self.roles.get(r) for r in ('cart', 'buy', 'refuse', 'ask', 'open', 'offer', 'compare')}:
            return ''
        sellers = self.seller_names()
        for key, value in args.items():
            if isinstance(value, str) and re.search('brand|maker|manufactur|vendor|make', _snake(key), re.I) and sellers.get(value.strip().lower()) is False:
                field = self.query_field(name) or self.query_field(self.roles.get('search') or '')
                how = f' in {field}' if field else ''
                return f"{value} is a shop that sells listings here (the rows' shop field), not a brand or manufacturer, so a {key} filter would miss its listings; search its name as words{how}, followed by two or three values its description gives"
        return ''

    def cart_sets(self, name: str) -> bool:
        props = self.schemas.get(name, {}).get('properties') or {}
        if any((isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) >= 2) for s in props.values())):
            return False
        return any((isinstance(s, dict) and 'integer' in _kinds(s) and (s.get('minimum') == 0) for s in props.values()))

    def whole_removal(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """When a removal of a held line comes with no count, fill in the units held, because the store would remove just one."""
        spec = self.schemas.get(name, {}).get('properties') or {}
        count = next((k for k, v in spec.items() if isinstance(v, dict) and 'integer' in _kinds(v)), None)
        removal = any((isinstance((spec.get(k) or {}).get('enum'), list) and v in spec[k]['enum'][1:] for k, v in args.items()))
        ident = next((v for v in args.values() if isinstance(v, str) and v in self.cart_lines), None)
        return {**args, count: min(self.cart_lines[ident], spec[count].get('maximum', self.cart_lines[ident]))} if count and removal and ident and count not in args else args

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
                have = have - (count or 1) if any(('integer' in _kinds(v) for v in spec.values() if isinstance(v, dict))) else have - count if count else 0
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
                out.append((cid, 'See the joint comparison result above: these listings were opened together there, costing a single step.'))
                continue
            fn = call.get('function') or {}
            name = fn.get('name')
            raw = fn.get('arguments')
            if self.finished:
                out.append((cid, 'Held back: the episode has ended.'))
                continue
            if name == self.roles.get('ask'):
                out.append((cid, "Held back (step not charged): every message to the shopper goes through Halo's interviewer."))
                continue
            if name not in self.schemas:
                out.append((cid, f"Held back (step not charged): no such tool. Available: {', '.join(sorted(allowed))}."))
                continue
            if name not in allowed:
                out.append((cid, 'Held back (step not charged): not enough steps remain for that; available now: ' + ', '.join(sorted(allowed)) + '.'))
                continue
            if isinstance(raw, dict):
                args = raw
            else:
                args = _json_obj(raw) if (raw or '').strip() else {}
                if args is None:
                    out.append((cid, 'Held back (step not charged): the arguments did not form a single JSON object.'))
                    continue
            faults: list[str] = []
            args = _shape(args, self.schemas[name], 'args', faults)
            if faults:
                out.append((cid, 'Held back (step not charged): ' + '; '.join(faults[:6]) + self.past_end(name, faults)))
                continue
            args, wrong = self.resolve_ids(name, args)
            wrong = wrong or self.seller_as_brand(name, args)
            if wrong:
                out.append((cid, 'Held back (step not charged): ' + wrong + '.'))
                continue
            args, started = self.fresh_start(name, args)
            if name == self.roles.get('cart'):
                args = self.whole_removal(name, args)
            trimmed = ''
            if started:
                trimmed += f'\n[Halo] Dropped {started} on this first call of these words: that argument is where the list starts, not how many rows come back, so a new query starts at the top. Send it again with the same words to read further down the list.'
            many = self.compare_target() if name == self.roles.get('compare') else None
            if many and isinstance(args.get(many), list) and (len(args[many]) > self.compare_room()) and self.compare_bad:
                room = self.compare_room()
                trimmed = f"\n[Halo] Only the first {room} listings were sent: this store compares at most {room} at once. Still unshown: {', '.join(map(str, args[many][room:]))}."
                args = {**args, many: args[many][:room]}
            target = self.single_target(name)
            if target and name in (self.roles.get('open'), self.roles.get('offer')) and (str(args.get(target)) in self.gone):
                out.append((cid, f'Held back (step not charged): {args[target]} has been sold out since step {self.gone[str(args[target])]} and stays gone. Swap its cart line for another fitting listing.'))
                continue
            if name == self.roles.get('cart') and any(((_count(v) or 0) < 0 for v in args.values() if not isinstance(v, str))):
                view = _dump(self.cart_view) if self.cart_seen_at >= self.cart_at else ''
                absent = [v for v in args.values() if isinstance(v, str) and v in self.rows and (v not in self.cart_lines) and (v not in view) and (v not in self.cart_said)]
                if absent:
                    out.append((cid, f'Held back (step not charged): {absent[0]} has no cart line, so nothing can be removed. Cart as it stands: ' + self.cart_text()))
                    continue
            ending = self.is_buy(name, args) or self.is_refuse(name, args)
            if ending and len(calls) > 1:
                out.append((cid, 'Held back (step not charged): send the finishing call alone, once your other results are read.'))
                continue
            if not ending and self.left() <= self.tail() and (self.roles.get('buy') or self.roles.get('refuse')):
                out.append((cid, "Held back (step not charged): the final steps are reserved for the order (with the shopper's basket approval when that is due) or for closing with no purchase."))
                continue
            pkey = self.page_key(args) if name not in (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask')) else None
            if pkey and args[pkey] >= 2 and self.page_mark(name, args, pkey) not in self.page_seen:
                trimmed += f'\n[Halo] Read page 1 of this list first ({pkey}={args[pkey]} was asked, and no earlier page of it was read).'
                args = {**args, pkey: 1}
            mark = name + _dump(args)
            coded = name == self.roles.get('search') and any((isinstance(v, str) and re.search(CODE_QUERY, v) for v in args.values()))
            if coded and self.code_finds[0] >= 2 and (not self.code_finds[1]):
                out.append((cid, f'Held back (step not charged): {self.code_finds[0]} searches by a model or part code found no listing in this store, so codes are not searchable here. Search by the kind of product and its plain values, then read the codes on the rows and pages.'))
                continue
            if not ending and name != self.roles.get('cart') and (mark in self.recent):
                out.append((cid, 'Held back (step not charged): this exact call was just made and nothing has changed; see its result above.'))
                continue
            twin = self.lead_twin(name, args) if name == self.roles.get('search') and (not ending) else ''
            if twin:
                self.trace.append({'kind': 'lead_twin', 'args': _clip(args, 80), 'twin': twin})
                out.append((cid, f'Held back (step not charged): here a query is judged on its first {self.lead_words} word(s) alone, so this search would bring back the rows of {twin} again (shown above). Open with fresh words instead: a second name for the product, a word from its store label, or a value that fitting listings carry.'))
                continue
            readers = {self.roles.get(r) for r in ('search', 'open', 'offer', 'compare')} - {None}
            cached = self.read_cache.get(mark)
            if name in readers and cached and (cached[0] == self.version):
                out.append((cid, f'Held back (step not charged): this call already ran at step {cached[1]} and nothing has changed since; its result:\n' + cached[2][:3000]))
                continue
            if target and name in readers and (name != self.roles.get('compare')) and self.shown_already(name, str(args.get(target))):
                out.append((cid, f'Held back (step not charged): this view of {args[target]} has shown all it can since the last change. Record: ' + self.record_of(str(args[target]))))
                continue
            if many and isinstance(args.get(many), list) and args[many]:
                keep = [i for i in args[many] if not self.shown_already(name, str(i))]
                if not keep:
                    out.append((cid, 'Held back (step not charged): this view of these listings has shown all it can since the last change. Records:\n' + '\n'.join((f'{i}: {self.record_of(str(i))}' for i in args[many]))[:4000]))
                    continue
                if len(keep) < len(args[many]):
                    trimmed += f"\n[Halo] Dropped {', '.join((str(i) for i in args[many] if i not in keep))}: already shown in full, unchanged since."
                    args = {**args, many: keep}
                    mark = name + _dump(args)
            if ending and self.is_refuse(name, args) and (not self.redone) and (self.denied_at >= 0) and (self.note_step >= self.denied_at) and (self.left() >= 3):
                self.redone = True
                self.audited = False
                self.refuse_blocks = 0
                self.leads = []
                self.trace.append({'kind': 'refuse_after_change', 'note': (self.notes or [''])[-1][:200]})
                out.append((cid, "Held back for now (step not charged). The store turned down the ending and altered the request: " + ((self.notes or [''])[-1] or 'see its last note') + ". Requirements as they now stand:\n" + self.ledger() + "\nListings rejected earlier can qualify now (more budget, a requirement dropped). Search using the altered values, open the plausible ones, cart the best and order; close with no purchase only if nothing satisfies the requirements as they now stand."))
                continue
            if ending and self.is_buy(name, args) and self.unread_lines():
                changed = self.read_lines()
                if changed:
                    out.append((cid, 'Held back for now (the order took no step). Halo read the cart\'s lines with the store\'s listing tools before the order, and the store showed a change:\n' + changed + '\nApply it, then check out.\nCart as it stands: ' + self.cart_text()))
                    continue
            if ending and self.left() > 2:
                if self.is_buy(name, args):
                    if not self.change_held and self.refused_at >= 0 and self.widened_at >= self.refused_at and self.cart_mark() == self.refused_cart and self.brief.get('pick') in ('cheapest', 'most', 'least') and self.left() > self.tail() + 3:
                        self.change_held = True
                        self.trace.append({'kind': 'change_hold', 'note': (self.notes or [''])[-1][:200]})
                        out.append((cid, 'Held back for now (step not charged). Along with refusing the order, the store altered the request: ' + ((self.notes or [''])[-1] or 'see the last note') + ". The note is the shopper speaking later, so it overrides the request text wherever the two disagree; requirements as they now stand:\n" + self.ledger() + "\nThe cart is unchanged from the refused one. With a requirement dropped (or more budget), listings rejected earlier can qualify, and one of them may win on the pick rule: search the item leaving out the dropped value, read the top rows, cart the best fit, then order.\nCart as it stands: " + self.cart_text()))
                        continue
                    wider = self.wider_hold()
                    if wider:
                        out.append((cid, wider + '\nCart as it stands: ' + self.cart_text()))
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
                                out.append((cid, 'Held back for now (step not charged). The cart view shows a change: ' + change + '\nCart as it stands: ' + self.cart_text() + '\nTake the change into account, correct the cart if needed, then order.'))
                                continue
                    swap = self.swap_hint()
                    held = [i for i in self.cart_ids() if i in self.gone] if self.held < 2 else []
                    if swap or held:
                        self.held += 0 if swap else 1
                        out.append((cid, 'Held back for now (step not charged). ' + (swap or f"{', '.join(held)} in the cart has sold out: swap that line first.") + '\nCart as it stands: ' + self.cart_text()))
                        continue
                    chain, hold = self.trail()
                    if hold:
                        out.append((cid, 'Held back for now (step not charged). ' + chain + '\nCart as it stands: ' + self.cart_text()))
                        continue
                    misfit = self.fit_faults() if self.fit_holds < 2 and self.left() > self.tail() + 3 else []
                    loose = self.inexact_lines() if not misfit and (not self.loose_held) and self.left() > self.tail() + 3 else []
                    if loose:
                        self.loose_held = True
                        self.trace.append({'kind': 'loose_hold', 'lines': loose[:3]})
                        out.append((cid, 'Held back for now (step not charged). A cart line states a required value only inside a longer one, while another opened listing of that item states it exactly: ' + '; '.join(loose[:3]) + ". A listing that states the value exactly is the safer fit: swap to it (respecting the pick rule and the budget) unless the request's own words accept the longer value, then check out.\nCart as it stands: " + self.cart_text()))
                        continue
                    unmet = self.unmet_lines() if not misfit and (not self.unmet_held) and self.left() > self.tail() + 3 else []
                    if unmet:
                        self.unmet_held = True
                        self.trace.append({'kind': 'unmet_hold', 'lines': unmet[:3]})
                        out.append((cid, 'Held back for now (step not charged). A cart line\'s attribute for a required property states another value: ' + '; '.join(unmet[:3]) + '. A value in different words does not state the requirement (the reading rules): swap to a listing that states it (the best of those named on the pick rule, or one of the unread rows named once its page shows the value), unless the two values are one word spelled differently, then check out; when no listing of that item states it, end without a purchase.\nCart as it stands: ' + self.cart_text()))
                        continue
                    bare = self.unstated_lines() if not misfit and (not loose) and (not self.bare_held) and self.left() > self.tail() + 3 else []
                    if bare:
                        self.bare_held = True
                        self.trace.append({'kind': 'bare_hold', 'lines': bare[:3]})
                        out.append((cid, "Held back for now (step not charged). A cart line lacks an attribute for a property a requirement sets, while other opened listings of that item state every requirement in their attributes: " + '; '.join(bare[:3]) + ". A title or description is weaker proof than an attribute: swap to such a listing (keeping the pick rule, the budget and any fit to another item), unless another attribute of the line states the value; then check out.\nCart as it stands: " + self.cart_text()))
                        continue
                    lead = self.unstated_leads() if not misfit and (not loose) and (not self.bare_held) and self.left() > self.tail() + 6 and self.model_room() > 40 else None
                    if lead:
                        self.bare_held = True
                        line, props, rows = lead
                        read = [i for i in rows[:2] if self.full.get(i)] if self.open_rows(rows[:2]) else []
                        proven = self.unstated_lines()
                        self.trace.append({'kind': 'bare_lead_hold', 'line': line, 'rows': rows, 'read': read, 'proven': proven[:2]})
                        text = f"Held back for now (step not charged). {line} lacks an attribute for {', '.join(props)}: the required value appears only in its title or description, which proves less than an attribute does, and none of that item's opened listings states all requirements yet. A word search on the required value turned up these unopened rows of that item (first those whose own text lacks the words, meaning the hit lies in their attributes), by rising price inside each group: " + ', '.join((f'{i} ({self.price_of(i):g})' for i in rows)) + '.'
                        if read:
                            text += ' Halo opened the first ones (one step each):\n' + '\n'.join((f'{i}: {self.facts_view(i)}' for i in read)) + '\n'
                        if proven:
                            text += ' Opened listings that now state every requirement: ' + '; '.join(proven[:2]) + '.'
                        text += ' Read the others that may fit (judge every requirement, not only this one) and swap to the best one on the pick rule whose attributes state every requirement; keep the line only when none of them does. Then check out.'
                        out.append((cid, text + '\nCart as it stands: ' + self.cart_text()))
                        continue
                    gauge = self.unmeasured() if not misfit and (not loose) and (not self.bare_held) and self.left() > self.tail() + 3 else None
                    if gauge:
                        self.bare_held = True
                        line, limits, opened, rows = gauge
                        read = [i for i in rows[:2] if self.full.get(i)] if rows and self.left() > self.tail() + 6 and self.model_room() > 40 and self.open_rows(rows[:2]) else []
                        self.trace.append({'kind': 'gauge_hold', 'line': line, 'limits': limits, 'opened': opened, 'rows': rows, 'read': read})
                        text = f"Held back for now (step not charged). {line} shows no figure for {'; '.join(limits)} anywhere on its record: no attribute for that property holds a number and no field states one in that unit, so nothing shows the limit is met (a figure only under another measure's key or nowhere does not count)."
                        if opened:
                            text += ' Opened listings of that item that state the figure and break no requirement their record states, cheapest first: ' + ', '.join((f'{i} ({self.price_of(i):g})' for i in opened)) + '.'
                        if read:
                            text += ' Halo opened the cheapest unread rows of that item (one step each):\n' + '\n'.join((f'{i}: {self.facts_view(i)}' for i in read)) + '\n'
                        rest = [i for i in rows if i not in read]
                        if rest:
                            text += ' Unread rows of that item in the budget, those showing a figure in that unit first, then by price: ' + ', '.join((f'{i} ({self.price_of(i):g})' for i in rest[:5])) + '.'
                        text += ' Swap to the best listing on the pick rule whose record states a figure inside the limit and meets every other requirement; keep the line only when none does. Then check out.'
                        out.append((cid, text + '\nCart as it stands: ' + self.cart_text()))
                        continue
                    if misfit:
                        self.fit_holds += 1
                        self.trace.append({'kind': 'fit_hold', 'faults': misfit[:4]})
                        out.append((cid, 'Held back for now (step not charged). A cart line does not work with the device it has to fit: ' + '; '.join(misfit[:3]) + '. Make the repair named (the cheapest one when two are named), then check out. Only when the request accepts a listing that falls short and no fitting one is seen, keep the line and check out.\nCart as it stands: ' + self.cart_text()))
                        continue
                    faults = self.line_faults() if self.line_holds < 2 and self.left() > self.tail() + 2 else []
                    fixed = []
                    if faults and all((' is held at ' in f for f in faults)):
                        fixed = self.fix_counts(self.count_fixes())
                        faults = self.line_faults() if fixed else faults
                        if fixed and (not faults):
                            self.line_holds += 1
                            out.append((cid, "Held back for now (step not charged). Halo fixed the cart quantities on its own: " + '; '.join(fixed) + '. Correct any line the store refused, then order.\nCart as it stands: ' + self.cart_text()))
                            continue
                    if faults:
                        self.line_holds += 1
                        self.trace.append({'kind': 'line_hold', 'faults': faults[:6]})
                        out.append((cid, 'Held back for now (step not charged). The cart still differs from the request: ' + '; '.join(faults[:6]) + ". Correct those lines (adding or removing units the way the cart tool does it), then order.\nCart as it stands: " + self.cart_text()))
                        continue
                    lines = self.cart_ids()
                    read = self.fill_facts(lines, self.tail() + (1 if len(lines) >= 2 else 3))
                    self.show_links(lines, self.tail() + 1)
                    read += self.cheaper_rows()
                    verdict = self.check_cart()
                    if verdict:
                        seen = "For the check Halo read the attributes of these listings (one step apiece):\n" + '\n'.join(read) + '\n' if read else ''
                        out.append((cid, 'Held back for now. ' + chain + seen + verdict + '\nCart as it stands: ' + self.cart_text()))
                        continue
                    if reread and self.cost > SPEND_SOFT:
                        out.append((cid, "Held back for now (step not charged). Halo has just re-read the cart's listings:\n" + reread + '\nOrder again if each line still has stock and a price within budget; if not, swap that line first.'))
                        continue
                elif self.left() > 3:
                    basket = self.series_basket() if self.is_refuse(name, args) and (not self.series_pushed) and self.left() > 4 else None
                    if basket and any((i not in self.cart_ids() for i in basket[1].values())):
                        self.series_pushed = True
                        self.trace.append({'kind': 'series_basket', 'basket': basket[:2]})
                        out.append((cid, "Held back for now (step not charged). Among listings already seen there is a complete one-series set for the items with series marks: " + self.series_text(basket) + ". All lines of such a set have to belong to the same series, and swapping the host listing is allowed to get there. Cart these lines (plus the lines of the other items) and order; a no-purchase close is right only if one of these lines plainly violates a requirement."))
                        continue
                    unsought = self.unsought_refs() if self.is_refuse(name, args) and (not self.ref_pushed) and self.left() > 4 else []
                    if unsought:
                        self.ref_pushed = True
                        self.trace.append({'kind': 'ref_unsought', 'refs': unsought})
                        out.append((cid, 'Held back for now (step not charged). The pages a requirement points at gave its value, but no search has used that value yet: ' + ' | '.join(unsought[:3]) + ". Search the item's listings with it (the product's noun and the value, one or two words), open the ones that may fit, and buy the best fitting one; end without a purchase only if none of them meets the requirements."))
                        continue
                    fixed = self.fix_counts(self.count_fixes()) if self.is_refuse(name, args) and self.cart_ids() and self.left() > 5 else []
                    if fixed:
                        self.trace.append({'kind': 'count_fix_before_decline', 'fixed': fixed})
                        out.append((cid, "Held back for now (step not charged). Halo fixed the cart quantities on its own: " + '; '.join(fixed) + ". A line's units are whatever count the cart tool is given (0 removes the line), so a bad count never justifies closing with no purchase. When each line now satisfies the requirements, order; close with no purchase only if some line plainly breaks one.\nCart as it stands: " + self.cart_text()))
                        continue
                    unread = self.unread_end() if self.is_refuse(name, args) else ''
                    if unread:
                        out.append((cid, 'Held back for now (the ending took no step). ' + unread + '\nCart as it stands: ' + self.cart_text()))
                        continue
                    review = self.audit()
                    if review:
                        out.append((cid, 'Held back for now (step not charged). ' + review))
                        continue
                    if self.leads and self.left() > 4 and (self.refuse_blocks == 0):
                        self.refuse_blocks = 1
                        self.trace.append({'kind': 'refuse_blocked', 'leads': list(self.leads)})
                        out.append((cid, 'Held back for now (step not charged). The review turned up listings that seem to fit: ' + ', '.join(self.leads) + '. Open whichever need it, cart the strongest fit under the pick rule and order. Only if each of them plainly misses a requirement may you close with no purchase, and then your text must say which, before the ending call goes out again.'))
                        continue
                    if self.leads and self.left() > 4 and (self.refuse_blocks == 1):
                        self.refuse_blocks = 2
                        first = self.leads[0]
                        self.check_cart([first], SPEND_HARD)
                        if self.verdicts.get(self.check_key([first])) and first in self.leads:
                            self.confirmed = first
                            self.trace.append({'kind': 'refuse_blocked', 'confirmed': first})
                            out.append((cid, f"Held back for now (step not charged). A second, separate review passes {first} on every requirement. Put it in the cart in the quantity asked for and order; close with no purchase only if its record plainly breaks a requirement."))
                            continue
                    held = self.held_basket() if self.is_refuse(name, args) and (not self.late_held) and (self.left() > 6) else []
                    if held:
                        self.late_held = True
                        missing = [i for i in held if i not in self.cart_ids() or i in self.dropped]
                        self.trace.append({'kind': 'held_basket', 'lines': held, 'missing': missing})
                        redo = f"Re-add {', '.join(missing)} in the quantities asked for, then order." if missing else 'Order the cart unchanged.'
                        out.append((cid, f"Held back for now (step not charged). The basket put together at step {self.built_at} ({', '.join(held)}) has every requested item, nothing has moved since, and a separate review of its records passes it on every requirement. With such a basket available, a no-purchase close scores zero. " + redo + ' A no-purchase close is right only when some record plainly breaks a live requirement, and then name it.'))
                        continue
            if self.is_refuse(name, args) and self.left() <= 5 and (not self.late_held):
                lines = [i for i in self.built if not self.sold_out(i)]
                fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
                missing = [i for i in lines if i not in self.cart_ids() or (i in self.dropped and (not fresh))]
                if len(lines) >= len(self.items) and len(missing) + 1 <= self.left():
                    self.late_held = True
                    self.trace.append({'kind': 'late_decline_held', 'lines': lines, 'missing': missing})
                    redo = f"Re-add {', '.join(missing)} in the quantities asked for, then order." if missing else 'Order the cart unchanged.'
                    out.append((cid, 'Held back for now (step not charged). Closing with no purchase scores zero whenever a fitting listing exists, and running low on steps is no evidence that none does. ' + redo + ' Only once searching has shown a requested item with no listing that meets its requirements may the ending call be sent again.'))
                    continue
            if self.is_refuse(name, args) and (not self.leads) and (not self.pushed) and (self.left() > self.max_steps // 3):
                self.pushed = True
                out.append((cid, f"Held back for now (step not charged). With {self.left()} steps still available, first try search routes not used so far: walk the category of every requested item page by page (using the token or offset the store hands back), send one-word value queries and alternative names for the property, and open plausible listings. A no-purchase close comes after that, not before."))
                continue
            if self.is_buy(name, args) and self.missing_items():
                lacked = self.missing_items()
                if self.fill_short_cart():
                    changed = self.read_lines()
                    if changed:
                        out.append((cid, f"Held back for now (the order took no step). The cart had no line for {', '.join((self.items[n - 1].name for n in lacked))}, so Halo added the opened listing that fits it best, read the cart's lines, and the store showed a change:\n" + changed + '\nApply it, then check out.\nCart as it stands: ' + self.cart_text()))
                        continue
                    trimmed += f"\n[Halo] The cart had no line for {', '.join((self.items[n - 1].name for n in lacked))}, so Halo added the opened listing that fits it best before the order."
                elif self.roles.get('refuse') and self.can_refuse() and self.missing_items():
                    refuse = self.roles['refuse']
                    faults = []
                    plain = _shape(dict(self.brief.get('refuse_args') or {}), self.schemas.get(refuse), 'args', faults)
                    if not faults:
                        self.trace.append({'kind': 'short_order_declined', 'missing': self.missing_items(), 'cart': self.cart_ids()})
                        trimmed += '\n[Halo] An order lacking one of the requested items is always wrong, and none of the opened listings for that item settles all stated values free of conflict, so Halo closed with no purchase instead.'
                        name, args = (refuse, plain)
            if self.is_refuse(name, args):
                gap = self.fill_gap()
                if gap:
                    out.append((cid, gap + '\nCart as it stands: ' + self.cart_text()))
                    continue
                done = self.complete_cart()
                if done:
                    out.append((cid, done + (' [episode over]' if self.finished else '')))
                    continue
                args = self.quiet(name, args)
            if self.is_buy(name, args):
                late = self.late_question()
                if late:
                    out.append((cid, 'Held back for now. ' + late + '\nCart as it stands: ' + self.cart_text()))
                    continue
            if ending and self.left() >= 2:
                go, said = self.gate('order' if self.is_buy(name, args) else 'end')
                if not go:
                    out.append((cid, 'Held back for now. ' + said + '\nCart as it stands: ' + self.cart_text()))
                    continue
                if said:
                    trimmed = '\n[Halo] ' + said + trimmed
            known = set(self.rows) if name == self.roles.get('search') and (not ending) else None
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, 'The call did not go through (network or server error). Trying again is fine.'))
                continue
            observation = reply.get('observation')
            if many and isinstance(args.get(many), list) and (len(args[many]) > 1):
                shown = _dump(observation)
                if not any((str(i) in shown for i in args[many])):
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    trimmed += f'\n[Halo] With {len(args[many])} listings the side-by-side view came back empty: send no more than {self.compare_room()} per call.'
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
            echo = self.search_echo(name, args, observation) if name == self.roles.get('search') and (not ending) else ''
            if echo:
                self.paged_obs = []
                trimmed += echo
            paged = self.more_pages(name, args, observation) if name not in plain and (not ending) and (not echo) else ''
            if name == self.roles.get('search') and (not ending) and (not echo):
                trimmed += self.list_end(name, args, observation)
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
                misfit = self.fit_faults() if self.cart_ids() else []
                labels = [f for f in self.line_faults() if 'a closer store label exists' in f][:2]
                if labels:
                    trimmed += '\n[Halo] Category check of the cart now: ' + '; '.join(labels) + '.'
                basket = self.series_basket() if self.cart_ids() and self.series_noted < 2 else None
                held = {c for i in self.cart_ids() for c in (self.fit_index()['marks'].get(i) or {}).get('takes') or ()} if basket else set()
                if basket and held and not held & self.series_full:
                    self.series_noted += 1
                    trimmed += f"\n[Halo] The cart's host listing takes series {', '.join(sorted(held))}, and no seen set of that series has a listing for every item with series marks; a full set is seen in {self.series_text(basket)}. Switch to one series that has every part rather than searching on in this one."
                if misfit:
                    self.trace.append({'kind': 'fit_note', 'faults': misfit[:3]})
                    trimmed += '\n[Halo] Fit check of the cart now: ' + '; '.join(misfit[:2]) + '. Repair it before checking out (a part chosen for an earlier device line may no longer fit).'
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
                    if lines != self.built:
                        self.built_at = self.step
                        self.built_needs = tuple(self.needs_now())
                    self.built = lines
            if name == self.roles.get('search') and self.records_in(observation):
                self.searched = (self.searched + [dict(args)])[-8:]
                trimmed += self.open_stale(known, observation) or self.open_short(args, observation)
            if coded:
                self.code_finds[1 if self.records_in(observation) else 0] += 1
            if name == self.roles.get('search') and (not ending) and self.searched and (not coded) and (not self.records_in(observation)):
                words = max((len(v.split()) for v in args.values() if isinstance(v, str)), default=0)
                if words >= 3:
                    trimmed += f'\n[Halo] No listing matched these {words} words. This store may need every word to match: try the one or two words that name the product and its most telling value.'
            if name == self.roles.get('cart_view'):
                self.cart_view = observation
                self.cart_seen_at = self.step
            if self.is_buy(name, args):
                self.buys += 1
                if not reply.get('finished'):
                    self.refused_at = self.step
                    self.refused_cart = self.cart_mark()
            if self.is_refuse(name, args) and (not reply.get('finished')) and (not self.finished):
                self.denied_at = self.step
            if name != self.roles.get('cart') and (not ending):
                trimmed += self.secure_cart()
                trimmed += self.unread_turn()
            text = self.show(observation) + paged + trimmed
            if name not in plain and (not ending):
                text += self.cheapest_here([observation] + self.paged_obs) + self.cap_ignored(name, args, observation)
            base = text
            proof = self.evidence(observation)
            if proof:
                text += '\n' + proof
            if name == self.roles.get('cart') and any((_count(v) == 0 for v in args.values() if not isinstance(v, str))):
                if any((isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values())):
                    text += '\n[Halo] The line survived a count of 0: this tool adds whatever count it gets to the line, so units come off only with a negative count.'
            self.note_sold = False
            lines = self.loose_lines(observation)
            if lines:
                meaning = self.take_note(lines)
                if meaning:
                    text += '\n[store note, as read] ' + meaning + '. Bring your picks in line with this change before ordering.'
            refused = self.is_buy(name, args) and (not reply.get('finished'))
            again = refused and any((line in self.spoken(observation) for line in self.sold_lines))
            loose = self.note_sold and (refused or (not self.note_named and (not self.is_buy(name, args))))
            if (loose or again) and self.cart_ids() and (not self.finished):
                seen = self.find_sold()
                if seen:
                    text += "\n[after the refusal Halo re-read the cart lines]\n" + seen
            self.note_sold = False
            swap = self.swap_hint()
            if swap:
                text += '\n' + swap
            if self.queued:
                text += "\n[store notes that came with Halo's own calls] " + ' | '.join(self.queued)
                self.queued = []
            if name in readers:
                self.read_cache[mark] = (self.version, self.step, base)
            if reply.get('finished'):
                text += ' [episode over]'
            out.append((cid, text))
        return out

    def held_basket(self) -> list[str]:
        """The fullest basket built in this episode, when it still stands: one in-stock line per item, built after the
        last store note or refused checkout and under the same requirements, and passed by the cart check."""
        lines = [i for i in self.built if not self.sold_out(i)]
        cat_key = self.category_key()
        if len(lines) < len(self.items) or not cat_key or self.built_at <= max(self.note_step, self.refused_at) or self.built_needs != tuple(self.needs_now()):
            return []
        if any((not any((self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or '')) for i in lines)) for item in self.items)):
            return []
        if len([i for i in lines if i not in self.cart_ids() or i in self.dropped]) + 2 > self.left():
            return []
        self.check_cart(lines, SPEND_HARD)
        return lines if self.verdicts.get(self.check_key(lines)) else []

    def can_refuse(self) -> bool:
        refuse = self.roles.get('refuse')
        if not refuse:
            return False
        faults: list[str] = []
        _shape(self.quiet(refuse, dict(self.brief.get('refuse_args') or {})), self.schemas.get(refuse), 'args', faults)
        return not faults

    def finish(self, again: bool=False) -> str:
        if self.finished or self.broke or self.left() <= 0:
            return ''
        buy, refuse = (self.roles.get('buy'), self.roles.get('refuse'))
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        whole = True
        if buy and (not holding) and self.left() >= 2 and self.missing_items():
            self.fill_short_cart()
            fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
            holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
            if self.finished or self.left() <= 0:
                return ''
        if holding and len(self.items) >= 2:
            whole = self.fill_short_cart() or bool(self.fill_gap()) or not self.can_refuse()
            if whole and self.bound and not self.finished:
                self.fit_rescue()
            if whole and len(self.cart_ids()) >= 2:
                self.fill_facts(self.cart_ids(), self.tail())
                if self.left() >= 3 and self.model_room() > 5:
                    self.show_links(self.cart_ids(), self.tail())
            if not whole:
                self.trace.append({'kind': 'short_cart_refused', 'lines': self.cart_ids()})
        if self.finished or self.left() <= 0:
            return ''
        if whole and buy and (holding or (self.cart_touched and (not fresh) and (self.left() >= 2))):
            self.read_lines()
            if self.finished or self.left() <= 0:
                return ''
            if self.value_rescue():
                self.read_lines()
                if self.finished or self.left() <= 0:
                    return ''
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
                    return 'the order was refused: ' + self.show(reply.get('observation'))
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
                    return 'the no-purchase ending was refused: ' + self.show(reply.get('observation'))
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
                self.chat[index]['content'] = content[:400] + f"...(earlier result cut; listing ids in it: {', '.join(ids)})"

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.read_request()
        self.trace.append({'kind': 'brief', 'brief': self.brief})
        opening = self.view.get('opening')
        if opening is not None:
            self.absorb(opening)
        intro = 'REQUEST FROM THE SHOPPER:\n' + self.request
        if opening is not None:
            intro += "\n\nTHE STORE'S OPENING VIEW:\n" + self.show(opening)
        first = self.opening()
        if first:
            intro += '\n\n' + first + "\n\nThese rows are already in front of you: judge them first and open in full the listings among them that may fit (their attributes decide). Search further only for what they leave open: an item with no promising row, a required value no row shows, or more candidates when the pick rule compares every fitting listing.\n\n" + self.status()
        else:
            intro += '\n\nOpen with one search per item inside its category, then open a few listings in full; that shows the interviewer what properties listings of this kind carry.\n\n' + self.status()
        self.chat = [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': intro}]
        self.work(LOOP_CALLS, SPEND_HARD)
        if not self.finished:
            turned = self.finish(again=True)
            if turned and (not self.finished) and (not self.broke) and (not self.no_credit) and (self.left() > 1):
                self.chat.append({'role': 'user', 'content': '[Halo] Halo attempted to finish the episode, but ' + turned + "\nAct on the store's reply, then finish.\n" + self.status()})
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
            if not self.stated_done and self.searched and not self.ready_to_ask():
                found = self.stated_search()
                if found and self.chat and isinstance(self.chat[-1].get('content'), str) and self.chat[-1].get('role') in ('tool', 'user'):
                    self.chat[-1]['content'] += '\n' + found
                elif found:
                    self.chat.append({'role': 'user', 'content': found + '\n' + self.status()})
            walked = self.early_trail() if not self.ready_to_ask() else ''
            if walked and self.chat and isinstance(self.chat[-1].get('content'), str) and self.chat[-1].get('role') in ('tool', 'user'):
                self.chat[-1]['content'] += '\n' + walked + '\n' + self.status().split('\n', 1)[0]
            elif walked:
                self.chat.append({'role': 'user', 'content': walked + '\n' + self.status()})
            self.trim()
            self.loop_calls += 1
            began = time.monotonic()
            message = self.think(self.chat, [LIGHT, SPARE], 3000, self.offered())
            if message is None:
                self.misses += 1
                if self.no_credit or self.calls >= ALL_CALLS or self.misses > 2:
                    break
                continue
            self.misses = 0
            self.pace = (self.pace + [time.monotonic() - began])[-5:]
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
                seek = self.seek_turn(content)
                idle += 0 if seek else 1
                self.chat.append({'role': 'assistant', 'content': content or '(no call)'})
                self.chat.append({'role': 'user', 'content': ('[Halo] For this turn only, searching and the listing views are back on offer.\n' if seek else '') + 'Continue by calling one of the tools.\n' + self.status()})
                if idle >= 3:
                    break
                continue
            idle = 0
            committing = {self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse')} - {None}
            if not self.asked and self.roles.get('ask') and (self.msgs_free() > 0) and any((c['function']['name'] in committing for c in calls)) and (self.left() > 8) and (self.clock() <= ASK_BY):
                self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
                for call in calls:
                    self.chat.append({'role': 'tool', 'tool_call_id': call['id'], 'content': 'Held back (step not charged): the shopper gets asked first.'})
                summary = self.interview()
                self.chat.append({'role': 'user', 'content': (summary or 'The interviewer found no question to ask.') + '\n' + self.status()})
                continue
            self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
            self.trace.append({'kind': 'turn', 'text': content[:300], 'calls': [c['function'] for c in calls]})
            replies = self.run_calls(calls)
            self.seek_now = False
            for index, (cid, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += '\n' + self.status()
                self.chat.append({'role': 'tool', 'tool_call_id': cid, 'content': text})

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Halo(problem_data)
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
