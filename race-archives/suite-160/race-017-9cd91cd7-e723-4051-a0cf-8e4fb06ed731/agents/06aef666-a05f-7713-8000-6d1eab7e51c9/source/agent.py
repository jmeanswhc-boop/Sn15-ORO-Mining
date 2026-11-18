"""Peace: an agent that shops for one person in a store world it can reach only through the world's tools.
agent_main builds a Peace from the problem data and calls run, which returns the trace of the episode.
An episode has four working parts. The brief: read_request sends the request, the tool list and any opening
note to the light model (the heavy model retries a brief that lacks items or a buy tool) and reads back the
items with quantities and firm requirements, the budget and its scope, the pick rule, message and step
limits, requirements that point at other listings, process steps due before choosing, ordering or ending,
and which tool does which job. The tool loop (work): the light model, with SPARE as fallback, reads the
GUIDE system prompt and a status block after every turn and proposes store calls; run_calls fits their
arguments to each schema, answers repeats, cached reads and views that can show nothing new without
spending a step, and keeps the last steps for the finish. The ledger in code: absorb files every listing
record a result shows (search rows, opened records with an attribute block, stock flags that flip),
take_note reads loose text for budget, requirement and stock changes, and ledger and status render the
requirements in force, the pick rule, the tool that shows attributes and the steps left for the model.
The interviewer: once enough listings are open, or when the model tries to cart or end before it ran,
interview opens more listings (widen), plan_lanes has the heavy model rank property labels per item from
the attribute keys seen, and ask_next sends list questions and reads the answers with hear; settle rotates
the labels when a list misses and falls back to one property per message when the shopper says a question
asked too much. late_question may spend one more message just before a checkout.
Before a checkout run_calls re-reads stale offers, looks at the cart, swaps sold-out lines, resolves
pointer chains (trail), checks part and device fits and the cart lines in code, opens unread lines, looks
for cheaper rows and has a model review the cart (check_cart); gate then carries out the request's own
steps due at the order. Before a decline it audits the records seen, holds the ending while candidates
remain, may confirm one and buy it itself (complete_cart), and rewrites the decline text (quiet). When the
loop stops with the episode still open, finish checks out the cart held or ends without a purchase.
Clock: Peace assumes TASK_SECONDS for the episode. Model calls stop once fewer than CALL_FLOOR seconds remain
before TIME_MARGIN of that limit; ASK_BY, QUESTION_BY and LATE_BY close the interview windows, WIND_DOWN adds
a time warning to the status, and LAST_TURN ends the loop unless the cart is empty and EMPTY_ROOM seconds
of model room remain. Spend: the cost the proxy reports is summed; SPEND_SOFT, SPEND_HARD and HEAVY_ROOM
decide which reviewers may run and when the loop stops, and LOOP_CALLS and ALL_CALLS cap loop turns and
model calls. Constant groups: models (LIGHT, HEAVY, SPARE); retries and rounds (WAITS, WAIT_TOTAL,
GATE_ROUNDS, AUTO_PAGES, CHEAPER_OPENS, WIDEN_EACH, LATE_STEPS); text sizes (SHOW_CHARS, OBS_LIMIT,
HISTORY_LIMIT); interview tuning (LIST_SIZE, SCARCE_MESSAGES, GUARD_KEYS, GUARD_SHARE, GUARD_GAP,
NOT_PROPERTY, CATCH_ALL); argument and value patterns (CAP_ARG, CAP_NOT, SHELF_ARG, LIST_MODE, QUERY_WORDS,
AMOUNT, FLOOR, FIT_CODE, PART_CODE); the proxy clients (_llm, _llm_short, _env); and the prompts: GUIDE for
the loop, BRIEF, UNMIX, SHAPE, PLAN, HEAR, NOTE, GATE, TRAIL, CHECK, AUDIT and SWAP, with READING holding
the attribute-reading rules that AUDIT and SWAP share.

Beyond the base it was built from, Peace reads listing ids that are numbers, attributes given as lists of name
and value records, and note lines that ride on a shopper's reply; it names a property under both of its store
spellings in a question; and it orders each list by how likely each property still is to be wanted after the
earlier answers (_prior, _read), with up to LIST_SIZE properties a list. It also reads records that a store writes as lines of text (_structured), so a world that answers that way
keeps a working ledger."""
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
AUTO_PAGES = 6
CHEAPER_OPENS = 3
TASK_SECONDS = 300.0
TIME_MARGIN = 20.0
CALL_FLOOR = 6.0
ASK_BY = 165.0
ASK_EARLY = 95.0
QUESTION_BY = 210.0
WIND_DOWN = 200.0
LAST_TURN = 255.0
EMPTY_ROOM = 12.0
LATE_BY = 190.0
LATE_STEPS = 8
SCARCE_MESSAGES = 2
LIST_SIZE = 10
LIST_MESSAGES = 3
RECOVER_STEPS = 8
_SELLER_KEY = re.compile('^(?:seller|shop|store|vendor|merchant|retailer|sold by)(?: name)?$')
_SELLERISH = 'seller|shop|store|vendor|merchant|retail|brand|sold'
CAP_ARG = '(?:^|_)(?:cap|max|maximum|ceiling|under|up_?to)(?:_|$)'
CAP_NOT = 'result|row|page|count|size|num|item|step'
SHELF_ARG = 'section|department|categor|label|shelf|aisle|(?<!age)(?:^|_)group(?:_name|_label)?$'
LIST_MODE = 'field|filter|facet|brows|categor|list|structured'
AMOUNT = '\\d[\\d.,]*\\s*[a-z"\']{1,12}'
FLOOR = '\\b(?:at least|no (?:less|fewer) than|min(?:imum)?)\\s+(' + AMOUNT + ')\\s*$'
QUERY_WORDS = '(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword'
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_llm_short: dict[int, ProxyClient] = {}
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)


def _env_post(binding: dict[str, Any], step: int, name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Send one store call and return its result data, or a _failed record when the call did not come back as data.
    The payload is the episode binding plus the step number and the action (tool name and arguments). A failure is
    returned as {'_failed': {'status': HTTP status, 'body': the error body as text}} so act can retry or stop on it.
    Every store call goes through this function, which is the hook a test world wraps."""
    payload = dict(binding)
    payload['step'] = step
    payload['action'] = {'name': name, 'args': args}
    result = _env.post_verbose('/environment/call', payload)
    if isinstance(result.data, dict):
        return result.data
    error = result.error or {}
    return {'_failed': {'status': error.get('status'), 'body': str(error.get('body') or '')}}
GUIDE = 'You are Peace, a careful buyer working for one shopper in an online store. You can act only by calling the store\'s tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.\n\nHOW THE STORE WORKS\n- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool\'s description and schema and use the exact argument names and allowed values.\n- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.\n- Search rows are summaries. Stores split a listing\'s record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Peace knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.\n- Questions to the shopper are handled for you by Peace\'s interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.\n- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Peace carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store\'s tools (product questions go through Peace\'s interviewer).\n\nWHAT COUNTS AS A FIT\n- The kind of product is decided by the listing\'s category label: only the label that matches the requested item as a whole counts, never a neighbouring one.\n- A requirement on a property is met when the listing\'s attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value\'s words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.\n- An attribute whose key names the requirement\'s property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing about the requirement. Peace adds [evidence] lines after opened listings with the keys whose names match a requirement.\n- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.\n- When the category label itself states a requirement\'s value (a season or audience in the label), that requirement is met by the category.\n- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing\'s value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.\n- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.\n- When one requested item must fit another (an accessory for a device), choose the two together from the listings\' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.\n- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.\n- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.\n\nCHOOSING\n- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Peace reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.\n- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.\n- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.\n- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.\n- A request for several different items needs one cart line per item, each at its requested quantity.\n\nCHANGES WHILE YOU WORK\n- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool\'s result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.\n- Re-read a chosen listing\'s record shortly before checking out, because price and stock can change silently.\n\nFINISHING\n- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.\n- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request\'s wording for that ending, and when a remark is needed, keep it short and name no listing.\n- Every reply must call at least one tool. Keep your text short: what you know, then the next move.'
BRIEF = 'You prepare a shopping task for an agent. Read the shopper\'s request and the store\'s tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:\n\n"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request\'s words, "shelf": the store\'s category label for it when a tool lists labels (copy it exactly), or the category the request itself names for it ("a coat in rainwear" gives "rainwear"), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request\'s own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property\'s value or limit outright; and never turn words that are part of what the product is called (a season or an audience named with the product) into musts: those belong to "name". A use or occasion the request makes a property of the product itself ("blazers for business wear", "a dress made for parties", "a bag, and it is for a wedding", "the occasion has to be party") IS a must on that property ("occasion: business"), unless the store\'s category label already names it; a particular person\'s event, a person or a setting the product is bought for ("to wear at his wedding", "for a colleague"), and how it should sit, feel or look in use, stay context. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.\n"budget": the most the shopper will pay in total as a number in the request\'s own unit, or null.\n"budget_unit": that unit in words (for example "cents"), or null.\n"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.\n"in_stock_only": true unless the request clearly accepts items that are not available.\n"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.\n"pick_measure": what is maximised or minimised for "most"/"least", else "".\n"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.\n"steps": how many steps, moves or tool calls the request allows (integer), or null.\n"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].\n"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.\n"wishes": a list of things the shopper would merely like.\n"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).\n"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper\'s approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.\n"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.\n"fits": a list of {"item": item number, "with": item number} for each requested item that the request says must work with, fit or suit ANOTHER requested item (the one the agent picks); [] when none.\n"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing\'s attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing\'s current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).\n"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.\n"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.\n'
PLAN = 'You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property; a property the shopper has no view on gets "don\'t mind", and one the shopper does not think about at all gets a puzzled reply, so every property named spends part of a message.\n\nYou get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item\'s category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.\n\nChoose the properties in this order:\n0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.\n1. Hinted properties. The request\'s story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product\'s looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.\nName a hinted property by a listed key of that item when one fits the hint (a hint about who will wear it: the key for gender or audience; about its looks: the listed key for colour, pattern or finish that most of its listings carry); a property no listing of the item carries comes after the item\'s three most carried keys.\n2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item\'s listings; once five or more listings were read, avoid keys seen on only one or two of them.\nSkip a property the request already settles for that item (a requirement names that property; a key whose values only repeat a required value, such as a pattern key showing the required decoration, is NOT settled), a second name for a property already chosen, a key whose values each list several purchase options at once, identifiers, model names, warranty and other logistics. Pack sizes, counts and quantities per pack are properties like any other, and a key marked [not the stated ...] is another property than that requirement, whatever words its name shares with it. Rank every item on its own: its hinted properties first, then its listed keys.\n"label" is the property\'s plain name: for a listed key, all of the key\'s words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property\'s name, no example values, no brackets, no second property.\nChoose up to COUNT for each item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story\'s words it comes from, or an empty string"}]}, each item\'s picks in rank order'
HEAR = 'A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants them asked one at a time; not answers that only say they did not follow it or want it in simpler words], "budget": new total limit as a number or null}\n- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3"); the value is only what a listing could state, from the wanted value or its limit word to its end, without the framing the answer puts around it ("oval" for "matching oval", "settled, with oval" or "oval and nothing else", "at least 30 cm" for "equal to at least 30 cm").\n- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop\'s product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer\'s own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.\n- "free": the property of each question the shopper said they do not mind about.\n- Folksy or roundabout wording that names a value still states that value. Wording that refuses to bargain over a value (it will not be traded, given up or changed) states that value as wanted; write \"not X\" only when the answer rejects X itself.'
NOTE = 'While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:\n{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request\'s unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}\nA replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note.'
CHECK = 'You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.\nReading rules: a requirement is met when the listing\'s attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value\'s words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing\'s value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.\nWhen the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart\'s choice on that rule, the cart is wrong.\nReturn ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses.'
READING = 'the listing\'s attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value\'s words counts, such as "deep teal" for teal, limits include the edge); a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing\'s value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit.'
AUDIT = 'A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: ' + READING + '\nRecords marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.\nReturn ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}'
UNMIX = 'A shopper\'s request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request\'s wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}'
TRAIL = 'A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing\'s page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement\'s chain from the requirement\'s own description, link by link, through the pages\' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).\nReturn ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing, "find": "for a link not found: search words (its shop and the values the description gives)"}], "done": true when the last link is an OPENED page that states the property\'s value itself (also when the requirement asks for a value other than or unlike that one: the page\'s value is what it is measured against), else false, "value": the property\'s value as stated on that last page, or ""}]}\nA record is a link only when it matches the whole description of that link: the shop or brand, the kind of product AND every property value the description gives (for example "a 10 pieces multipack" rules out the same shop\'s listing with 10 capsules and no multipack); when no record matches all of it, that link is not found yet, even if the same shop sells other listings of that kind. Give "find" for such a link: the shop\'s name and the description\'s values, as a few search words.\nList each chain\'s links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown.'
GATE = 'A shopping agent is about to STAGE. The shopper\'s request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product\'s properties do not do a step.\nWhen a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.\nReturn ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}'
SWAP = "A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: " + READING + '\nReturn ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line\'s item, at most 6], "reason": "one sentence"}'
SHAPE = 'A store tool returned the result below. Name its parts by the exact keys the result and the tool\'s parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing\'s price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}'
GUARD_KEYS = 3
GUARD_SHARE = 0.0
GUARD_GAP = 0.34
NOT_PROPERTY = '\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b'
CATCH_ALL = '^(?!.*\\b(?:types?|styles?|kinds?|shapes?)\\b).*\\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\\w*|accessor\\w*|extras?|misc\\w*)\\b'
WIDEN_EACH = 6
FIT_CODE = '[A-Z][A-Za-z]{0,5} ?\\d{3,6}'
PART_CODE = '[A-Z]{1,4}\\d{3,6}'

def _codes(text: Any) -> list[str]:
    """Return the model or series codes found in a text, normalised for comparison.
    Each match of FIT_CODE at word boundaries (a capital letter, up to five more letters, an optional space and
    three to six digits) has its inner whitespace collapsed to one space and is lower-cased. None or an empty
    value gives an empty list; other values are read through str()."""
    return [re.sub('\\s+', ' ', c).strip().lower() for c in re.findall('\\b' + FIT_CODE + '\\b', str(text or ''))]

def _fit_marks(text: Any) -> dict[str, Any]:
    """Read the fitting marks in a listing's text that pair an accessory with a device.
    Returns a dict with five entries; each phrase it reads runs to the next dot or semicolon, and case is ignored.
    own: the listing's own part code (a PART_CODE inside round or square brackets, optionally after the word
    part), upper-cased, or an empty string.
    codes: the codes given after a model number label (model no, model number, model #) or after also sold as,
    also known as or sold also as.
    works: the codes after works with, compatible with or fits, unless not or never comes right before, and
    leaving out the fits that belong to a sentence of the form part CODE also fits.
    also: a list of (part code, codes) pairs read from such part CODE also fits sentences.
    more: the part code the text sends the reader on to (more, details, see or read, then part CODE), or ''.
    The code lists go through _codes, so they are lower-cased like the codes of other listings."""
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
    """Encode a value as compact JSON (no spaces after separators, non-ASCII kept), or as str() when it cannot.
    Peace uses this one form for prompt sections, cache keys, cart marks, call arguments and log text, so
    equal data always gives equal text."""
    try:
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        return str(value)

def _clip(value: Any, limit: int, depth: int=0) -> Any:
    """Return a copy of nested data with long strings cut, so one big field cannot flood a prompt.
    Strings over limit characters keep their first limit characters plus three dots; dicts and lists are copied
    element by element with the same limit, and past seven levels of nesting a whole branch becomes three dots.
    Other values pass unchanged, and the input is never modified."""
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
    """Pull one JSON object out of a model reply, allowing a code fence or stray text around it.
    Non-strings give None. A Markdown code fence that wraps the whole reply is removed and the text is parsed;
    when that fails, the span from the first opening brace to the last closing brace is tried. Only a dict is
    returned: a list, a bare scalar or text that never parses gives None."""
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
    """Normalise a key or label to lower-case words separated by single spaces.
    A lower-case letter or digit followed by a capital is split, so camel case turns into words, and
    runs of whitespace, underscores, dots and hyphens become one space. Peace compares names in this
    form however a store spells them."""
    key = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(key))
    return re.sub('[\\s_.\\-]+', ' ', key).strip().lower()

def _snake(key: str) -> str:
    """Spell a key in snake case: the words from _words joined by underscores.
    CAP_ARG, SHELF_ARG and QUERY_WORDS are matched against this form, so camel-case and spaced parameter
    names are tested the same way."""
    return _words(key).replace(' ', '_')

def _read(place: int) -> float:
    """Chance that the shopper reads a property named exactly at this place of a list (1-based).
    About 0.9 first, 0.6 second and third, 0.3 later: the rates seen in live race 14 and qualifying episodes of
    every agent that asked lists."""
    return 0.9 if place <= 1 else 0.6 if place <= 3 else 0.3


def _prior(rank: int) -> float:
    """Starting weight of a lane label by its rank at planning time (0-based).
    1 for the first four, then falling (0.5 at rank 6, 0.25 at rank 12), the way the shopper's unstated wishes
    spread over the planner's ranking in live episodes."""
    return 1.0 / (1.0 + max(0, rank - 3) / 3.0)


def _stem(text: str) -> frozenset[str]:
    """Return the words of a text as a frozenset of rough stems, a word of four or more letters losing a final s.
    The words come from _words. Callers compare such sets by intersection and subset tests (_names among them)
    to match property names, labels and keys across singular and plural forms."""
    return frozenset((w[:-1] if len(w) > 3 and w.endswith('s') else w for w in _words(text).split()))

def _roots(text: str) -> set[str]:
    """Return the set of rough word roots of a text, used for overlap scores between labels and item names.
    Words come from _words. A word ending in ches, shes, sses, xes or zes loses the es; any other word of four or
    more letters ending in a single s loses the s. Roots under three letters and the words and, for, the and
    with are left out. Unlike _stem the result is a plain set and short words are dropped."""
    out = set()
    for w in _words(text).split():
        w = w[:-2] if re.search('(?:ch|sh|ss|x|z)es$', w) else w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w
        if len(w) >= 3 and w not in ('and', 'for', 'the', 'with'):
            out.add(w)
    return out

def _spelled(text: str) -> frozenset[str]:
    """Return a frozenset of a text's words with plural s and British our spellings evened out.
    Each word from _words of four or more letters that ends in s (but not ss) drops the s, and a word ending in
    our then ends in or, so plural and British forms of a property name equal the plain form. The interviewer
    compares labels with requirement properties and key names in this form."""
    return frozenset((re.sub('our$', 'or', w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w) for w in _words(text).split()))

def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    """Decide whether a key names the same property as a requirement, with both given as word sets.
    An empty set on either side means no. They match when the sets are equal, when every word of the property is
    in the key, or when the key has two or more words and all of them are in the property. A one-word key
    inside a longer property name therefore does not match."""
    if not prop or not key:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)

def _kinds(schema: dict[str, Any]) -> list[str]:
    """Return the JSON-schema type of a parameter as a list of type names.
    A single string becomes a one-item list and a list keeps its string entries. A missing type gives an empty
    list, which _shape treats as accepting any value."""
    kind = schema.get('type')
    if isinstance(kind, str):
        return [kind]
    return [k for k in kind or [] if isinstance(k, str)]

def _fits_kind(value: Any, kind: str) -> bool:
    """Say whether a value already has the JSON-schema type named by kind.
    integer takes int and number takes int or float, both refusing booleans (a bool is an int in Python);
    string, boolean, array, object and null map to str, bool, list, dict and None. An unknown type name accepts
    anything, so an unusual schema never blocks a call."""
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
    """Fit a value to a JSON schema where that is safe, and append every remaining problem to faults.
    where names the spot in fault messages; a schema that is not a dict accepts the value as it is. A value of
    the wrong type gets the first conversion that applies: an integral float to int, a numeric string to a
    number (an int when the schema wants an integer, truncating, or when the number is whole), a number to text,
    the text true or false to a boolean, or a lone value wrapped in a list. If the type still does not fit, a
    type fault is added and the value is returned with no further checks. An enum value equal to an option apart
    from case takes the store's spelling, otherwise it is a fault. Text over maxLength is cut; text under
    minLength, numbers outside minimum or maximum and lists outside minItems or maxItems are faults. List
    entries are fitted to the items schema; for an object, unknown names (when additionalProperties is false)
    and missing required names are faults, and known properties are fitted recursively."""
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
    """Coerce a reply field into a list: a list as it is, a non-empty string or dict as a one-item list, else [].
    Every list field of a model reply is read through it, so the loops over those fields need no type checks."""
    if isinstance(value, list):
        return value
    if isinstance(value, (str, dict)) and value:
        return [value]
    return []

def _count(value: Any) -> int | None:
    """Read a whole number from a loosely typed value, or return None.
    Booleans give None. An int is returned as it is, a float only when it is whole, and a string only when it is
    digits with an optional dot and zeros, spaces allowed around it; a sign or any other text gives None.
    Quantities, limits from the brief, item numbers in model replies and cart counts are read through it."""
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
    """Say whether a value could be a listing id: a string of 3 to 64 characters with no space once stripped.
    When named is False (the key's name does not say id) it must also contain a digit, which keeps plain words
    such as colours or brand names from passing for ids.
    A positive integer counts only when named is True (a key whose name marks it as an id)."""
    if isinstance(value, int) and (not isinstance(value, bool)):
        # a numbered listing: only under a key known to hold ids
        return named and value > 0
    if not isinstance(value, str) or not 3 <= len(value.strip()) <= 64 or ' ' in value.strip():
        return False
    return named or bool(re.search('\\d', value))

def _learn_id(rows: list[dict[str, Any]]) -> str | None:
    """Work out which key of a non-empty list of result rows carries the listing id, or return None.
    Only the first row's keys are tried. A key qualifies when every row's value under it passes _id_like (a key
    whose name contains id needs no digits) and, with more than one row, the stripped values are all different.
    Among qualifying keys, one whose name contains id wins, then the one with the longer mean value. With a
    single row, as ident calls it, uniqueness cannot be tested and the score alone decides.
    Keys holding text ids are tried first; keys holding numbers (a store that numbers its listings) count
    only when no text key qualifies, so a store with text ids is read exactly as before."""
    best = None
    # text ids first; numbers count only when no key holds text ids
    for numbers in (False, True):
        for key in rows[0]:
            values = [row.get(key) for row in rows]
            if any((isinstance(v, int) and (not isinstance(v, bool)) for v in values)) != numbers:
                continue
            if not all((_id_like(v, 'id' in str(key).lower()) for v in values)):
                continue
            values = [str(v).strip() for v in values]
            if len(rows) > 1 and len(set(values)) < len(values):
                continue
            score = ('id' in str(key).lower(), sum((len(v) for v in values)) / len(values))
            if best is None or score > best[0]:
                best = (score, key)
        if best:
            break
    return best[1] if best else None

def _in_rows(node: Any, ident: str, depth: int=0) -> bool:
    """Say whether a listing id appears as a string value of a row somewhere in nested data.
    Only dicts that sit in a list are tested, by their direct string values after stripping; the search goes
    down through lists and dict values to depth 5. run_calls uses it to see whether a cart reply still shows a
    line after a call with a count of zero or less."""
    if depth > 5:
        return False
    if isinstance(node, list):
        return any((isinstance(v, dict) and ident in [str(x).strip() for x in v.values() if isinstance(x, (str, int)) and (not isinstance(x, bool))] for v in node)) or any((_in_rows(v, ident, depth + 1) for v in node))
    if isinstance(node, dict):
        return any((_in_rows(v, ident, depth + 1) for v in node.values()))
    return False

def _llm_within(seconds: float) -> ProxyClient:
    """Return a proxy client whose timeout fits in the given number of seconds.
    With 50 seconds or more the shared client _llm (50 s timeout) serves. Otherwise the timeout is the seconds
    rounded down to a multiple of five, at least five; one client per such timeout is made on first use and kept
    in _llm_short. call_model passes model_room(), so a late call cannot outlast the episode clock."""
    if seconds >= 50:
        return _llm
    limit = max(5, int(seconds) // 5 * 5)
    if limit not in _llm_short:
        _llm_short[limit] = ProxyClient(timeout=limit, max_retries=1, retry_delay=1)
    return _llm_short[limit]

def _scalars(value: Any) -> dict[str, Any]:
    """Keep the plain values of a mapping: text keys whose value is a string or a number (never a boolean).
    Anything that is not a dict gives an empty dict."""
    if not isinstance(value, dict):
        return {}
    return {k: v for k, v in value.items() if isinstance(k, str) and isinstance(v, (str, int, float)) and (not isinstance(v, bool))}


def _pairs(value: Any) -> dict[str, Any]:
    """Read attributes given as a list of small records instead of a name to value map, or return an empty dict.
    The records must share one text field holding short, non-identifier names (at least two different ones; a
    repeated name keeps its first value) and exactly one other plain field, the value. Text in Name: value form is
    not read, since a listing's own description is often written that way."""
    if isinstance(value, list) and len(value) >= 2 and all((isinstance(v, dict) for v in value)):
        common = sorted(set(value[0]).intersection(*[set(v) for v in value[1:]]))
        for name_key in common:
            names = [v.get(name_key) for v in value]
            # names may repeat (one property under two spellings that read the same): the first wins
            if not all((isinstance(n, str) and 0 < len(n) <= 60 and (not _id_like(n)) for n in names)) or len(set(names)) < 2:
                continue
            others = [k for k in common if k != name_key and all((isinstance(v.get(k), (str, int, float)) and (not isinstance(v.get(k), bool)) for v in value))]
            if len(others) == 1:
                pairs: dict[str, Any] = {}
                for v in value:
                    pairs.setdefault(v[name_key], v[others[0]])
                return pairs
    return {}


_TEXT_KEY = re.compile('^([A-Za-z][A-Za-z0-9_]*(?: [A-Za-z0-9_]+){0,3}):(?: (.*)|)$', re.S)
_PAIR_KEY = re.compile('^[A-Za-z0-9][^=;|]{0,59}$')


def _text_value(text: str) -> Any:
    """Type one value read from a text record: True, False and None by their words, whole numbers and decimals as
    numbers, and a run of 'key=value' parts joined by '; ' (an attribute list) as a dict, a part without '=' joining the
    value before it. Anything else comes back as the text it was."""
    if text in ('True', 'False'):
        return text == 'True'
    if text == 'None':
        return None
    if re.fullmatch('-?\\d{1,15}', text):
        return int(text)
    if re.fullmatch('-?\\d{1,12}\\.\\d{1,6}', text):
        return float(text)
    if '=' not in text or '; ' not in text:
        return text
    pairs: dict[str, str] = {}
    last = None
    for part in text.split('; '):
        key, eq, value = part.partition('=')
        if eq and _PAIR_KEY.match(key.strip()) and key.strip() not in pairs:
            last = key.strip()
            pairs[last] = value.strip()
        elif last is not None:
            # a value that itself holds '; '
            pairs[last] += '; ' + part
        else:
            return text
    return pairs if len(pairs) >= 2 else text


def _text_record(text: Any) -> dict[str, Any] | None:
    """Read one record that a store writes as a line of text, 'key: value | key: value | ...', into a dict, or None.
    A part that does not start with a new key belongs to the value before it (a title can hold the separator). It must
    give three or more keys, one of them holding an id-like value, so a reply or a description is never taken for a
    record; the values are typed by _text_value."""
    if not isinstance(text, str) or ' | ' not in text or len(text) > 20000:
        return None
    out: dict[str, str] = {}
    last = None
    for part in text.split(' | '):
        piece = part.strip()
        lead = ''
        while piece.startswith('|'):
            # a value that ended with the separator itself ('... Jersey |')
            lead += '|'
            piece = piece[1:].strip()
        found = _TEXT_KEY.match(piece)
        if found and found.group(1) not in out:
            if lead and last is not None:
                out[last] += ' ' + lead
            last = found.group(1)
            out[last] = (found.group(2) or '').strip()
        elif last is not None:
            # a value that itself holds ' | ' (a title with separators)
            out[last] += ' | ' + part
        else:
            return None
    if len(out) < 3 or not any((_id_like(v) for v in out.values())):
        return None
    return {k: _text_value(v) for k, v in out.items()}


def _structured(node: Any, depth: int=0) -> Any:
    """Return an observation with every record written as text turned into a dict, everything else as it was.
    A whole observation that is one such line, a list whose items are mostly such lines, and the one text of a small
    wrapper (three keys or fewer) are read with _text_record; a wrapper that holds nothing but that line comes back as
    the record itself, the way a JSON store gives a page, so the listing is found at the top. Text inside a record's own
    fields is never read, so a store that answers in JSON objects comes back unchanged."""
    if depth > 6:
        return node
    if isinstance(node, str) and depth == 0:
        # a whole observation that is one record written as text
        return _text_record(node) or node
    if isinstance(node, list):
        parsed = [_text_record(v) for v in node]
        if node and sum((1 for p in parsed if p)) * 2 >= len(node):
            return [p if p else v for p, v in zip(parsed, node)]
        return [_structured(v, depth + 1) for v in node]
    if isinstance(node, dict):
        texts = [k for k, v in node.items() if isinstance(v, str)]
        if len(node) <= 3 and len(texts) == 1:
            record = _text_record(node[texts[0]])
            if record and len(node) == 1:
                # a page that is only its record, as a JSON store gives it: the listing is read at the top
                return record
            if record:
                return {k: (record if k == texts[0] else _structured(v, depth + 1)) for k, v in node.items()}
        return {k: (v if isinstance(v, str) else _structured(v, depth + 1)) for k, v in node.items()}
    return node


_DUE_KEY = re.compile('(?:^| )(?:arriv[a-z]*|deliver[a-z]*|ship|ships|shipping|shipped|dispatch[a-z]*|eta)(?: |$)')
_NOT_DUE_KEY = re.compile('(?:^| )(?:power|weight|member[a-z]*|relation[a-z]*|scale|method|cost|costs|fee|fees|charge|charges|price|free|address|country|region|zone|area|type|mode|option|options|size)(?: |$)')
_DUE_VALUE = re.compile('(?:^|[^a-z])(?:before|by|within|until|no later|in time|tomorrow|today|tonight|next|monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekend|january|february|march|april|june|july|august|september|october|november|december|[0-9]+ ?(?:st|nd|rd|th)|[0-9]+ ?(?:business |working )?(?:day|days|hour|hours|week|weeks)|[0-9]+[/.-][0-9]+)(?:$|[^a-z])', re.I)
_SHOWN_KEY = re.compile('(?:^| )(?:arriv[a-z]*|deliver[a-z]*|ship[a-z]*|dispatch[a-z]*|eta|lead time|handling|transit|courier|carrier)(?: |$)')
_SHOWN_TEXT = re.compile('(?:^|[^a-z])(?:ships?|dispatch(?:es|ed)?|deliver(?:s|ed|y)?|arriv(?:es|al)|estimated delivery)(?: in| within| by| on| from| out|$)', re.I)


def _deadline(must: str) -> bool:
    """Say whether a requirement is a delivery deadline: its property is about arriving, delivery, shipping or dispatch
    (not a product property that only shares the word, such as power delivery, shipping weight, a delivery method or
    cost), and its value names a time: before, by, within, until, a weekday, a month, an ordinal day, a number of days,
    hours or weeks, or a date."""
    key, _, value = must.partition(':')
    words = _words(key)
    return bool(_DUE_KEY.search(words)) and (not _NOT_DUE_KEY.search(words)) and bool(_DUE_VALUE.search(value))


def _shows_delivery(value: Any, depth: int=0) -> bool:
    """Say whether a listing record shows anything about delivery: a field, at any of the first levels of nesting, named
    for arrival, delivery, shipping, dispatch, ETA, lead time, handling, transit or the carrier, or a short field value
    such as 'Ships in 2 days'. Long free text (descriptions) is not read."""
    if depth > 3:
        return False
    if isinstance(value, dict):
        return any((_SHOWN_KEY.search(_words(str(k))) or _shows_delivery(v, depth + 1) for k, v in value.items()))
    if isinstance(value, list):
        return any((_shows_delivery(v, depth + 1) for v in value[:50]))
    return isinstance(value, str) and len(value) <= 60 and bool(_SHOWN_TEXT.search(value))


def _faulted(seen: Any) -> bool:
    # a store reply that reports a fault: any field named like an error or a failure that holds something, or ok false
    """Say whether a store reply reports a fault: ok is False, or a top-level field named like an error, a failure, a
    fault or trouble holds something. Field names differ from world to world, so no world's own name is used."""
    return isinstance(seen, dict) and (seen.get('ok') is False or any((v and re.search('error|fail|fault|trouble', str(k), re.I) for k, v in seen.items())))


def _fact_block(node: Any) -> dict[str, Any] | None:
    """Find the attribute block of a listing record: its richest nested dict of plain values.
    For a dict node, each value that is itself a dict with two or more entries is reduced to its string and
    number entries (booleans and nested values left out); the reduced dict with the most entries, needing at
    least two, is returned, the first one winning a tie. Anything else gives None. A record with such a block
    counts as opened in full: absorb keeps it in self.full, and evidence, key_table and the checks read it.
    A value that is a one-key group around the attributes is looked inside; when no mapping qualifies, a list
    of small name and value records is read through _pairs."""
    if not isinstance(node, dict):
        return None
    best = None
    for value in node.values():
        scalars = _scalars(value)
        if len(scalars) < 2 and isinstance(value, dict) and len(value) == 1:
            scalars = _scalars(next(iter(value.values())))
        if len(scalars) >= 2 and (best is None or len(scalars) > len(best)):
            best = scalars
    if best is None:
        for value in node.values():
            pairs = _pairs(value)
            if len(pairs) >= 2 and (best is None or len(pairs) > len(best)):
                best = pairs
    return best

class Item:

    """One requested product as Peace tracks it through the episode.
    name and shelf come from the brief (shelf is the store's category label or the category the request names,
    and may be empty) and qty is the number of units wanted. musts are the firm requirements read from the
    request; told gathers requirements learned later (shopper answers, gate replies, notes that add one);
    dropped lists requirements no longer in force; free records properties the shopper said they do not mind,
    which no other code reads. The requirements in force are musts plus told, minus dropped (see needs_now)."""
    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        """Store the name, shelf label, quantity and request requirements; start empty told, dropped and free lists."""
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []

class Peace:

    """The shopping agent for one episode: brief, interviewer, tool loop, code-side ledger and pre-finish checks.
    One instance serves one problem. It holds the store binding and tools, the brief and the items, every
    listing record seen, Peace's own record of the cart, the interview lanes, the spend, call and clock
    counters, and a trace list of events that run returns as the episode's log.
    Main state: rows maps each listing id to everything any result showed about it, full keeps the listings
    seen with an attribute block, seen_order the order of first sight, and opened the ids passed to page tools.
    cart_lines is Peace's own count per cart line, kept from the cart calls it saw; cart_view is the last cart
    the store showed, trusted only while cart_seen_at is not older than cart_at. unavailable and gone hold
    sold-out listings (gone ones are also stamped with the step when Peace noticed). The version counter goes
    up whenever what is known changes (new or dropped requirements, notes, sold-out lines, changed offers,
    resolved pointer chains); the read cache, view records, gate marks and review keys all carry it, so such a
    change makes them stale at once."""
    def __init__(self, problem: dict[str, Any]):
        """Set up the episode state from the problem data; no store or model call is made here.
        From problem['environment'] it keeps a copy of the binding and the policy view, the request (the view's
        query), the tools that have a function with a string name, and one parameter schema per tool (an empty
        object schema when none is given). max_steps comes from the view (a float is truncated; 30 when missing or
        not positive), finished from the view's own flag, and the clock starts. Everything else starts empty or at
        zero: step and call counters and cost, the brief, roles and items, budget and message counts, process steps
        and gate marks, the listing stores (rows, full, seen_order, opened), the cart state (cart_lines, cart_view,
        and the steps when the cart last changed and was last seen, both -1), the stock sets (unavailable, gone),
        the read caches and learned shapes, the interview lanes and their counters, and the counters that limit how
        often run_calls may hold an ending call."""
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
        self.heard_text = ''
        self.heard_words: set[str] = set()
        self.notes: list[str] = []
        self.version = 0
        self.cart_touched = False
        self.cart_at = -1
        self.called_at: dict[tuple[str, str], int] = {}
        self.changed_at = -1
        self.offered_ids: set[str] = set()
        self.unshown: list[str] = []
        self.offer_tried: set[str] = set()
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
        self.early_held = False
        self.lead_items: dict[str, list[str]] = {}
        self.verdicts: dict[str, bool] = {}
        self.confirmed: str | None = None
        self.completed = False
        self.link_tried: set[str] = set()
        self.lane_misses: dict[int, int] = {}
        self.lane_weight: dict[int, dict[str, float]] = {}
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
        self.spellings: dict[str, str] = {}
        self.lane_found: dict[int, int] = {}
        self.lane_spent: dict[int, int] = {}
        self.lane_used: dict[int, set[str]] = {}
        self.late_asked = False
        self.one_product: set[int] = set()
        self.follow_up: list[tuple[int, list[str], bool]] = []
        self.listing = True
        self.line_holds = 0
        self.bound: list[tuple[int, int]] = []
        self.fit_holds = 0
        self.searched: list[dict[str, Any]] = []
        self.link_searches = 0
        self.link_seen: dict[str, list[str]] = {}

    def left(self) -> int:
        """Return the store steps still left: max_steps minus the steps used."""
        return self.max_steps - self.step

    def clock(self) -> float:
        """Return the seconds since the episode started, by the monotonic clock."""
        return time.monotonic() - self.started

    def model_room(self) -> float:
        """Return the seconds still usable for model calls: TASK_SECONDS less TIME_MARGIN less clock().
        think refuses a call when this falls below CALL_FLOOR, and _llm_within bounds each call's timeout by it."""
        return TASK_SECONDS - TIME_MARGIN - self.clock()

    def msgs_free(self) -> int:
        """Return how many shopper messages the interviewer may still spend.
        That is msg_limit less msg_used, less the part of the reserve for order and ending steps that
        gate has not used yet (reserve minus gate_msgs, never below zero). The result can be negative.
        read_request sets the reserve from the order and end steps that need messages, and gate counts
        its own messages in gate_msgs."""
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)

    def cart_mark(self) -> str:
        """Return a fingerprint of Peace's own cart record: its sorted (listing, units) pairs as compact JSON.
        gate_due, gate and cheaper_rows pair it with the version number to tell whether the cart has changed, and
        send and gate stamp it on every record line."""
        return _dump(sorted(self.cart_lines.items()))

    def page_tools(self) -> set[str]:
        """Return the tools whose calls count as opening a listing's page.
        That is the offer tool (the open tool when there is no offer role) together with the compare tool, leaving
        out roles that are not set. act marks every string id passed to these tools as opened."""
        page = self.roles.get('offer') or self.roles.get('open')
        return {t for t in (page, self.roles.get('compare')) if t}

    def facts_tool(self) -> str | None:
        """Return the single-listing tool that shows attributes, or None.
        A tool already seen showing an attribute block (facts_from True) that takes one id wins. Otherwise the open
        role serves, provided it is not known to show none (facts_from False) and takes a single id.
        facts_from is learned from replies: fill_facts, widen and run_calls record per tool whether a single-listing
        or side-by-side reply carried an attribute block, and run_calls moves the open role to another single-id
        tool when the open tool shows none."""
        known = next((t for t, ok in self.facts_from.items() if ok and self.single_target(t)), None)
        if known:
            return known
        tool = self.roles.get('open')
        return tool if self.facts_from.get(tool or '') is not False and self.single_target(tool) else None

    def compare_facts(self) -> bool | None:
        """Return whether the compare tool showed attribute blocks: True, False, or None when not known.
        False means it returned listing records without any block; None also covers a store with no compare tool."""
        return self.facts_from.get(self.roles.get('compare') or '')

    def compare_bare(self) -> bool:
        """Say whether the compare tool should be treated as one that shows no attributes.
        True when it is known to show none, or when nothing is known about it yet while the facts tool is known to
        show attributes or the store has a separate offer tool. widen, open_rows and join_opens then avoid it."""
        seen = self.compare_facts()
        return seen is False or (seen is None and (self.facts_from.get(self.facts_tool() or '') is True or bool(self.roles.get('offer'))))

    def gate_due(self, stage: str) -> bool:
        """Say whether the request's process steps for a stage still need a gate run for the current cart.
        True when some process step has this when value (order or end) and the pair of the stage with the cart mark
        and version number is not yet in gate_ok, where gate records a stage it found complete."""
        return any((p['when'] == stage for p in self.process)) and (stage, self.cart_mark() + f'#{self.version}') not in self.gate_ok

    def tail(self) -> int:
        """Return how many final steps Peace keeps for finishing the episode: 1, or 2 when a message is due.
        It is 2 when an order-stage process step needs messages, a message is still allowed and the order gate is
        still due, so one step can go to that message before the checkout. run_calls refuses other calls inside
        the tail, and offered and status narrow the tools and sharpen the warnings near it."""
        talk = any((p['when'] == 'order' and p['messages'] > 0 for p in self.process))
        return 2 if talk and self.msg_used < self.msg_limit and self.gate_due('order') else 1

    def judges(self, limit: float=SPEND_SOFT) -> list[str]:
        """Return the models that may run a review now, given the spend so far.
        HEAVY and LIGHT, in that order, while the cost is within limit (SPEND_SOFT by default) and one heavy call
        (HEAVY_ROOM) would still fit under SPEND_HARD; only LIGHT while the cost is within SPEND_HARD; else none.
        check_cart, audit, replacements and late_question use it to decide whether a review can be afforded."""
        if self.cost <= limit and self.cost + HEAVY_ROOM <= SPEND_HARD:
            return [HEAVY, LIGHT]
        return [LIGHT] if self.cost <= SPEND_HARD else []

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None=None) -> dict[str, Any] | None:
        """Ask the given models in turn for one chat reply and return the first message with content or tool calls.
        It gives up with None once ALL_CALLS model calls were made, or when model_room() is below CALL_FLOOR (logged
        as late_call_skipped). Each model tried counts as one call. A failed call is retried after a pause when its
        status is missing, 402, 408, 425, 429 or 500 and above: at most once for a missing status or 408, else up to
        len(WAITS) times. Pauses follow WAITS with a random factor between 0.8 and 1.2, and stop when the nominal
        waits would pass WAIT_TOTAL over the episode or leave less than CALL_FLOOR seconds of room. An empty reply
        (status 0) is not retried. A 402 that outlasts its retries sets no_credit and ends the search; any other
        failure moves on to the next model, and None comes back when every model failed. tools, when given, are
        passed on to call_model for a tool-calling turn.
        work uses it for loop turns with tools, and ask_json for every prompt that wants a JSON object back."""
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
                    return message
                passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                if not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL:
                    break
                if self.model_room() - WAITS[tries] * 1.2 < CALL_FLOOR:
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
        """Make one chat completion request through the proxy and return (message, status).
        For anthropic models with more than one message, the first and the last message have their plain text
        content wrapped in a text block marked for ephemeral prompt caching (on a copy; the chat is not changed).
        The body asks for temperature 0 and max_tokens, and carries the tools with tool_choice auto when given.
        The client comes from _llm_within(model_room()), and the usage cost the proxy reports is added to cost.
        A choice with content or tool calls gives (message, 200); an empty choice is logged and gives (None, 0); an
        error gives (None, status), the status taken from the client error or from the code in the body's error
        object, and is logged as llm_fail with up to 200 characters of the error text."""
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
        """Send one prompt as a single user message and return the first reply that parses to a JSON object.
        The models are tried one at a time through think with max_tokens, and the reply text goes through _json_obj.
        It returns None when every model fails, and stops early once the store reported no funds (broke), the proxy
        reported no credit, or ALL_CALLS was reached."""
        for model in models:
            message = self.think([{'role': 'user', 'content': prompt}], [model], max_tokens)
            got = _json_obj((message or {}).get('content'))
            if got is not None:
                return got
            if self.broke or self.no_credit or self.calls >= ALL_CALLS:
                return None
        return None

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        """Send one action to the store and return the response data, or None when it did not go through.
        The payload is the binding plus the next step number and the action (tool name and arguments). On a dict
        reply the step counter follows the reply's step (or goes up by one), finished is set when the reply says so,
        and when the tool is one of page_tools and the observation is not an error, every string argument value
        (also inside lists) joins self.opened. Failures are logged as env_fail and, within four attempts: a 409
        whose text mentions finishing, quarantine or finalizing marks the episode finished; a 409 naming the
        expected step resyncs the step counter and retries; a 409 about reuse or a different action moves the step
        on by one and retries; any other 409 gives up. 400, 401, 403, 404 and 422 give up at once, and 402 also
        sets broke. Other failures, such as network or server errors, are retried right away without a pause.
        Each call goes through the module-level _env_post, which returns the result data or a _failed record
        carrying the HTTP status and body; the retries and step resyncs work on that record.
        A store that writes its records as lines of text has them turned into dicts first (_structured), so the
        ledger, the key table and the checks read them, and show clips each field on its own rather than the whole line.
        A clean reply of a read tool (search, page, side-by-side, cart view, the facts tool) is kept in called_at under its
        tool and arguments, for the gate; any other successful call stamps changed_at, since it may have changed the store's
        state (a remove tool without a role, a cart call the gate made)."""
        sent = self.step + 1
        for _ in range(4):
            data = _env_post(self.binding, sent, name, args)
            if '_failed' not in data:
                if 'observation' in data:
                    # a store that writes records as lines of text: read them as records (a JSON store is unchanged)
                    data['observation'] = _structured(data['observation'])
                self.step = data['step'] if isinstance(data.get('step'), int) else self.step + 1
                if data.get('finished'):
                    self.finished = True
                seen = data.get('observation')
                # reads that came back clean may be reused by the gate; any other tool may have changed the store's state
                reads = {self.roles.get(r) for r in ('search', 'open', 'offer', 'compare', 'cart_view')} | {self.facts_tool()}
                fault = _faulted(seen)
                if name in reads - {None}:
                    if seen is not None and (not fault):
                        self.called_at[(name, json.dumps(args, sort_keys=True, default=str))] = self.step
                else:
                    self.changed_at = self.step
                if name == self.roles.get('offer') and (not _faulted(seen)):
                    self.offered_ids.update((str(v).strip() for v in args.values() if isinstance(v, (str, int)) and (not isinstance(v, bool))))
                if name in self.page_tools() and (not (isinstance(seen, dict) and seen.get('error'))):
                    for value in args.values():
                        self.opened.update((str(v).strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str) or (isinstance(v, int) and (not isinstance(v, bool)))))
                return data
            error = data['_failed'] if isinstance(data.get('_failed'), dict) else {}
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
                    sent = self.step + 1
                    continue
                if 'reuse' in low or 'different action' in low:
                    self.step += 1
                    sent = self.step + 1
                    continue
                return None
            if status in (400, 401, 402, 403, 404, 422):
                if status == 402:
                    self.broke = True
                return None
        return None

    def read_request(self) -> None:
        """Have a model read the request into the brief, then set up items, roles, budget and limits from it.
        The BRIEF prompt carries every tool (name, description, parameter schema), the request and, when present,
        the store's opening note clipped to 1500 characters. LIGHT answers first; when that brief lacks items, a
        roles object or a buy tool, HEAVY then LIGHT try again; when two items share a shelf label
        (shared_shelf), HEAVY is asked once more and its brief replaces the first if it has items, roles and a
        buy tool. Roles are kept only when they name a real tool. When buy and refuse are the same tool and their
        fixed arguments are missing or equal, a short extra prompt asks which arguments select the checkout and
        which the decline. Items get quantity 1 when none is given; an item whose shelf label equals an earlier
        item's, with no property given two different values between them, is merged into that item (logged as
        items_merged), and a placeholder item stands in when none was read. Fit pairs become self.bound
        (renumbered after merging), lookups become self.refs (matched to a must by property name when not copied
        exactly), and unmix runs. The budget is kept when positive (a numeric string counts). msg_limit is the
        brief's message count, or 3 when an ask tool exists and no count was given; a step count from 10 up to
        below max_steps lowers max_steps. Process steps are kept with when set to choose, order or end (order by
        default) and their message counts, and with an ask tool the reserve is the largest message count among
        the order and end steps, capped at msg_limit.
        Every question names one property, whatever the message limit: most race worlds answer a message that names
        two or more of their subjects with "one at a time", which costs the whole message."""
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
        # lists only where exactly LIST_MESSAGES messages are allowed; single questions otherwise
        self.listing = False
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

    @staticmethod
    def shared_shelf(brief: dict[str, Any]) -> bool:
        """Say whether two or more items of a brief name the same non-empty shelf label, compared by _words.
        read_request asks the heavy model for a second brief when this is true."""
        shelves = [_words(str(e.get('shelf') or '')) for e in _items(brief.get('items')) if isinstance(e, dict)]
        shelves = [s for s in shelves if s]
        return len(shelves) != len(set(shelves))

    def unmix(self) -> None:
        """Move words that only describe a pointed-at listing out of an item's own requirements.
        For each item that has both pointer requirements (refs among its musts) and other musts, the
        UNMIX prompt with the request, the product name and both lists goes to HEAVY, then LIGHT. Musts
        the reply copies exactly under describe_other are taken off the item and appended to its first
        pointer requirement as a parenthesised note on that listing; the matching entry of self.refs is
        renamed to the joined text, and the change is logged as unmix. read_request runs it once, after
        the lookups are matched to musts."""
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
        """Say whether a requirement is worded in the shopper's own terms that no attribute of this kind states.
        The ledger lists such requirements apart, as deciding nothing. The answer is False for pointer requirements
        (refs or brief lookups), for one with no property name or fewer than four words in its value, for a value
        that mentions a listing, a page, shown, other than, unlike or same as, and for a value that shares a word of
        four or more letters with another item's name. Otherwise the attribute blocks of the opened records of this
        item's category (all opened records when no category key is known) are read, and the answer is True only
        when at least three were read and none of their keys shares a stem with the requirement's property."""
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
        """Render the task as Peace now understands it, one fact per line, each line starting with a dash.
        For each item: its number (when there are several), name, store label and quantity, and the requirements in
        force (musts plus told, minus dropped); a pointer requirement whose value trail has resolved gets that value
        appended, and requirements that described() flags are listed apart as deciding nothing. Then the budget
        with its unit and scope, the pick rule (cheapest; most or least with its measure; otherwise any listing
        that fits), in stock only unless the brief says otherwise, wishes as tie-breakers, the request's rules, the
        quiet decline rule, each process step with when it is due, and, once known, which tool shows listing
        attributes, which page tools show none and which tool shows a listing's page. The status block and the
        gate, note, check, audit and swap prompts all embed it.
        After the items come the delivery deadlines that no listing in this store can show (unshown), listed as no requirement."""
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
            parts.append(text)
        if self.unshown:
            parts.append('not requirements (no listing in this store states a delivery or arrival time, so none can be checked): ' + '; '.join(self.unshown))
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
            when = {'choose': 'before choosing', 'order': 'at the order (Peace does it when you check out)', 'end': 'at an ending without a purchase (Peace does it then)'}[entry['when']]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        facts = self.facts_tool()
        if facts and (self.facts_from.get(facts) is True or self.roles.get('offer')):
            bare = sorted((t for t in self.page_tools() | {self.roles.get('offer') or ''} if t and self.facts_from.get(t) is False))
            pages = sorted((t for t in self.page_tools() if t != facts))
            parts.append(f'listing attributes come from {facts}' + (f"; {', '.join(bare)} show none" if bare else '') + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ''))
        return '\n'.join(('- ' + p for p in parts))

    def status(self) -> str:
        """Build the status block added to the last tool reply of each turn and to Peace's own user messages.
        It opens with the steps used and left, then gives at most one step warning: final steps (left within the
        tail plus one), few steps (six or fewer, or few enough for the item count before any buy), running low
        (twelve or fewer before any buy), or half the steps used while the cart was never touched. After WIND_DOWN
        seconds a time warning is added unless the final-steps warning applies. Before any buy it lists up to three
        pointer requirements still unresolved and the fit plan for paired items; it also shows review candidates
        (leads), the last four changes from notes, and the ledger.
        The warnings only advise the model; the tail itself is enforced by run_calls and offered."""
        left = self.left()
        text = f'[Peace] step {self.step} of {self.max_steps} used, {left} left.'
        if left <= self.tail() + 1:
            text += ' FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that).'
        elif left <= 6 or (left <= len(self.items) + 5 and self.buys == 0):
            text += ' Few steps left: set the cart to the chosen lines and check out now.'
        elif self.buys == 0 and left <= 12:
            text += ' Steps are running low: settle on your picks, fill the cart and check out within the next few steps.'
        elif self.buys == 0 and (not self.cart_touched) and (self.step >= self.max_steps // 2):
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        if self.clock() >= WIND_DOWN and left > self.tail() + 1:
            text += ' TIME IS NEARLY UP: set the cart to the chosen lines and check out within the next two turns; an episode that runs out of time earns nothing.'
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
        text += '\n[ledger]\n' + self.ledger()
        return text

    def ident(self, node: Any, in_list: bool=False, depth: int=0) -> str | None:
        """Return the listing id of a dict that looks like a listing record, or None.
        A record needs at least three keys. When id_key is known and the node's value under it passes _id_like,
        that value, stripped, is the id. Otherwise only a node inside a result list, the top node (depth 0) or a
        node that holds a nested dict is tried: _learn_id on the node alone picks a key, accepted when its name
        contains id or its value is already a known listing id.
        absorb, records_in, loose_lines, evidence and fill_facts use it to tell listing records from other data.
        A numbered id comes back in its text form, so every record of a listing is filed under one key."""
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if self.id_key and _id_like(node.get(self.id_key), True):
            return str(node[self.id_key]).strip()
        nested = any((isinstance(v, dict) for v in node.values()))
        if not (in_list or depth == 0 or nested):
            return None
        found = _learn_id([node])
        if found and ('id' in str(found).lower() or str(node[found]).strip() in self.rows):
            return str(node[found]).strip()
        return None

    def learn_rows(self, node: list[Any]) -> bool:
        """Decide whether a list in a result holds listing rows, learning the id key on the way.
        At least half of the entries must be dicts with three or more keys. They count as rows when the known
        id_key holds an id-like value in every one of them, or when _learn_id finds a key, which then becomes
        self.id_key (replacing any earlier guess)."""
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

        """File every listing record found in a store result into Peace's stores.
        The result is walked to depth 6. A dict that ident recognises is merged into self.rows (a new id is also
        appended to seen_order); a key whose value flips from True to False marks the listing unavailable, and one
        that flips back clears the mark. When the dict has an attribute block (_fact_block), self.full gets a copy
        of the merged row; when it has none but the listing is already in full, the full record is updated too.
        The walk does not go inside a recognised record. For a list, learn_rows decides whether its entries are
        result rows, which lets ident accept them below the top level.
        rows therefore gathers search rows and pages alike, the newest value of a key winning, while full only holds
        listings once seen with an attribute block. Every store reply passes through it (via observe, run_calls or
        gate), and so does the opening view.
        After each result, shelf_from_rows checks the brief's seller requirements against the category labels seen.
        Then drop_unshown_deadlines drops a delivery-date requirement that no listing in this store can show."""
        def walk(node: Any, depth: int, in_list: bool=False) -> None:
            """Visit one node at the given depth; in_list tells ident that the node is an entry of a row list."""
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
        self.shelf_from_rows()
        self.drop_unshown_deadlines()

    def drop_unshown_deadlines(self) -> None:
        """Set aside the delivery deadlines (_deadline: 'arrival: before the 15th') among the requirements in force
        when no listing in this store can show one: five or more listings seen, a listing page opened or a full record read
        (and an offer page opened, where the store keeps offers on a page of their own), and no search row or full record
        showing any delivery, shipping or arrival detail (_shows_delivery). A deadline from the request leaves musts and one
        from a reply joins dropped; each is listed in unshown, which the ledger shows as no requirement, and a note tells the
        model why. The version goes up, so earlier check verdicts are not reused. Where any listing shows such a detail, or no
        page has been read yet, the deadline stays for the model and the reviewers to judge."""
        dated = [(item, must) for item in self.items for must in item.musts + item.told if must not in item.dropped and _deadline(must)]
        if not dated or len(self.rows) < 5 or not (self.opened or self.full):
            return
        offer = self.roles.get('offer')
        if offer and offer != self.facts_tool() and (not self.offered_ids):
            return
        if any((_shows_delivery(record) for record in list(self.rows.values()) + list(self.full.values()))):
            return
        for item, must in dated:
            if must in item.musts:
                item.musts.remove(must)
            else:
                item.dropped.append(must)
            if must not in self.unshown:
                self.unshown.append(must)
            self.trace.append({'kind': 'deadline_dropped', 'must': must})
            self.queued.append(f"(Peace) no listing in this store states a delivery or arrival time, so '{must}' cannot be checked against any listing and is not a requirement")
        self.version += 1

    def shelf_from_rows(self) -> None:
        """Drop a brief requirement that names a seller when the store shows it is the item's category instead.
        A must on a seller, shop, store, vendor, merchant or retailer (or who it is sold by) qualifies when the request says
        'from <value>', listings carry <value> as their category label, and no listing gives it as a seller, shop or brand.
        Then the must goes, <value> becomes the item's shelf when the brief left that empty and its name when the brief only
        said listing, item or product, and a note for the loop says why. A request such as '1 of the same listing from
        Point & Shoot Cameras' can be read as a seller requirement that no listing meets. A key that names sellers, shops or
        brands never counts as the category label, even when every listing seen so far shares one value."""
        sellers = [(item, must) for item in self.items for must in item.musts if _SELLER_KEY.match(_words(must.partition(':')[0])) and must.partition(':')[2].strip()]
        if not sellers or not self.rows:
            return
        cat_key = self.category_key()
        label_keys = {k for row in self.rows.values() for k in row if 'categor' in _words(k)} | ({cat_key} if cat_key else set())
        # a key naming sellers, shops or brands is never the category label, even when every row seen shares one value
        label_keys = {k for k in label_keys if 'categor' in _words(k) or not re.search(_SELLERISH, _words(k))}
        if not label_keys:
            return
        labels = {_words(row[k]) for row in self.rows.values() for k in label_keys if isinstance(row.get(k), str)}
        named = {_words(v) for row in self.rows.values() for k, v in row.items() if isinstance(v, str) and re.search(_SELLERISH, _words(k))}
        for item, must in sellers:
            value = must.partition(':')[2].strip()
            if _words(value) not in labels or _words(value) in named or not re.search('from[ ]+(?:the[ ]+)?' + re.escape(value), self.request, re.I):
                continue
            item.musts.remove(must)
            if not item.shelf.strip():
                item.shelf = value
            if _words(item.name) in ('listing', 'item', 'product', 'the listing', 'the item', 'the product', 'the requested product'):
                item.name = value.lower()
            self.trace.append({'kind': 'shelf_not_seller', 'item': item.name, 'must': must})
            self.queued.append(f"(Peace) '{value}' is the store's category label of these listings, not a seller: the request asks for an item from that category, so '{must}' is no requirement")

    def labels(self) -> set[str]:
        """Collect every enum option of every tool schema, stripped and lower-cased.
        The schemas are walked through nested dicts to depth 6 (lists such as anyOf are not entered). category_key
        uses the set to prefer the record key whose values the store's own filters list."""
        found: set[str] = set()

        def walk(schema: Any, depth: int) -> None:
            """Add the string enum options of one schema node and recurse into its dict values."""
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
        """Guess which key of the listing rows holds the category label, or return None.
        For every key it gathers the short string values (1 to 40 characters, no run of three
        digits) and counts the rows that carry one. Keys found in fewer than 80 percent of the rows,
        or with more than 40 distinct values, are skipped. The rest rank by the share of their
        values that are schema enum labels, then by fewer distinct values, then by presence in more
        rows, then by a longer mean value. It is worked out afresh on every call from all rows seen
        so far. Without it the code-side cart checks (line_faults, missing_items, count_fixes) and
        the cheaper-row search do nothing."""
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
        """Summarise the attribute keys of the opened listings, per category, for the interview planner.
        Opened records (self.full, in the order first seen) are grouped by category label (one group called
        listings when no category key is known). For each key name, in _words form, it counts the listings that
        carry it and keeps each listing's value, lower-cased. When some group labels match an item (equal to the
        item's shelf or name, sharing a stem of four or more letters with them, or claimed by owner_of), only those
        groups are kept. Within a group, keys are taken by count, most first; it skips keys whose values average
        over 40 characters, keys where at least half of the values contain a spaced slash (lists of options), and
        free-text keys (four or more values, all different, averaging over 12 characters, with fewer than half
        sharing a word with the other values). A key that twins() pairs with one already kept is merged into it.
        The 14 most carried keys of each group come back as (label, key name, number of listings carrying it, up
        to four values cut to 40 characters), the largest groups first.
        plan_lanes (which hands it to lane_keys), answered_label and samples read it; it is rebuilt on every call.
        The first twin folded into a kept key is remembered as that key's other spelling (spellings), so a
        question can name both."""
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
                    twin.setdefault('other', name)
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
                if slot.get('other'):
                    self.spellings[_words(name)] = slot['other']
        return table

    def not_stated(self, label: str, name: str, values: list[str]) -> str:
        """Return a marker for a key that resembles, but is not, the property a requirement states; else ''.
        For the item that owns the category label, it looks at the requirements with a plain value (no negation,
        comparison, range or or-words). When such a requirement's property differs from the key name but shares a
        stem with it, and the requirement's value words are not all among the key's sample values, the marker
        names that requirement's property as another property. plan_lanes adds it to the key list, and the PLAN
        prompt treats a marked key as a property of its own."""
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
        """Say whether two key-table entries look like one property under two spellings.
        The two names must share a word. agree() counts two values as the same when they are equal or the shorter
        one's words all appear in the longer one, which may have at most twice as many words plus two. Over the
        listings that carry both keys with non-empty values: when neither name's words contain the other's, at
        least two such listings must all agree; when one name contains the other, a majority must agree, and with
        no shared listing a common non-empty value is enough."""
        a, b = (set(name.split()), set(other.split()))
        if not a or not b or (not a & b):
            return False

        def agree(x: str, y: str) -> bool:
            """Say whether two values match: equal, or the shorter one's words inside a longer one of at most 2n+2 words."""
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
        """Group the seen listings by category label and return the groups that matter for the items.
        All listings in seen_order are grouped by their row's category label (a single group when no category key
        is known). Groups whose label matches an item, as in key_table, are preferred; with no match all groups
        count. The largest groups come first, at most one per item and at least one. widen and widen_singly draw
        the listings they open from these groups."""
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
        """Return the one parameter through which a tool takes a single listing id, or None.
        Candidates are the free-text string parameters (no enum). The required one among them is chosen, or else the
        only one; the tool qualifies only when no other parameter is required. A missing tool also gives None."""
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        target = next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not tool or not target or any((k != target for k in schema.get('required') or [])):
            return None
        return target

    def price_of(self, ident: str) -> float | None:
        """Return a listing's price from its row as a float, or None.
        It reads the price key learned by read_shape. A dict value gives its first plain number, a numeric string is
        converted, and booleans or anything else give None. Without a known price key every price is None."""
        value = (self.rows.get(ident) or {}).get(self.price_key) if self.price_key else None
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and (not isinstance(v, bool))), None)
        if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', value):
            value = float(value)
        return float(value) if isinstance(value, (int, float)) and (not isinstance(value, bool)) else None

    def buyable(self, ident: str) -> bool:
        """Say whether a listing can be bought as far as Peace knows.
        False when sold_out says so, or when the row's stock key (learned by read_shape) holds False or a number of
        zero or less. Unknown stock counts as buyable."""
        if self.sold_out(ident):
            return False
        value = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        return not (value is False or (isinstance(value, (int, float)) and (not isinstance(value, bool)) and (value <= 0)))

    def records_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        """Return the (id, record) pairs that a store result shows near its top level.
        A dict result that is itself a listing gives just that pair. Otherwise records come from the dict's list
        values and from lists one level further inside its dict values; a list result gives the records among its
        entries. Deeper records are not searched here (absorb walks deeper)."""
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
        """Remember which keys a tool showed for each listing in a result, for the current version.
        tool_keys[name] collects every key the tool has ever shown, and view_keys keeps for each listing the keys
        seen since the version last changed (an older entry starts afresh). shown_already compares the two."""
        for ident, record in self.records_in(observation):
            keys = set(record)
            self.tool_keys.setdefault(name, set()).update(keys)
            version, have = self.view_keys.get(ident, (-1, set()))
            self.view_keys[ident] = (self.version, (have if version == self.version else set()) | keys)

    def shown_already(self, name: str, ident: str) -> bool:
        """Say whether reading a listing with this tool again could show nothing new.
        True only when the tool has shown keys before, the listing was viewed in the current version, every key the
        tool shows is among the keys already seen for it, and the listing has not been marked gone."""
        version, have = self.view_keys.get(ident, (-1, set()))
        keys = self.tool_keys.get(name)
        return bool(keys) and version == self.version and (keys <= have) and (ident not in self.gone)

    def read_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        """Learn once per tool how its list results are laid out, with one model call, and return that shape.
        A cached answer, including a cached failure, comes back at once. The result must be a dict holding a list of
        at least two dicts; otherwise None is returned without caching. A sample (row lists cut to three rows,
        values clipped, at most 4000 characters) goes with the tool's parameters to the SHAPE prompt for LIGHT,
        then HEAVY. The answer is kept only where its keys exist: rows must name one of the row lists, count and
        next must be keys of the result (count only when it holds an integer), next_arg a parameter of the tool,
        and price and stock keys of the first row. The shape is cached and logged, and it sets price_key and
        stock_key when they are not known yet.
        paging calls it, so a tool's shape is learned the first time more_pages or list_below needs it."""
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
            """Return the key when it is a string present in the given dict, else None."""
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
        """Return (total matched, rows shown, next-page parameter, next value) for a list result, or None.
        It needs a shape with count, next and next_arg keys, a non-empty row list and an integer total. The next
        value must be None, a string or an integer; None means there is no further page."""
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
        """Say whether the records of a result are already in ascending price order.
        It needs at least three records, every one with a known price."""
        prices = [self.price_of(i) for i, _ in self.records_in(observation)]
        return len(prices) >= 3 and None not in prices and all((a <= b for a, b in zip(prices, prices[1:])))

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """Read further pages of a search result when the pick rule needs the whole fitting set.
        Only for cheapest, most and least picks and a result whose shape allows paging; paged_obs is reset first.
        Nothing happens when there is no next value, the call itself asked for a later page, every match is shown,
        or (for cheapest) the rows are already sorted by price. The pages needed are capped at two when that fits
        within AUTO_PAGES and leaves 12 steps beyond the tail; a larger need, or too few steps, gives back a hint
        to narrow the search or page on with the given value instead. Each page call is fitted to the schema,
        counted in pages_read, observed, kept in paged_obs and read_cache, and summarised; reading stops when a
        page has no next value. The page texts come back under a header, with a note when pages remain unread.
        run_calls calls it after any call that is not an ending, made with the search tool or a tool with no role."""
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
            return f"\n[Peace] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with {param}={_dump(value)}."
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
        rest = f"\n[Peace] {count} listings matched and Peace has read {shown * (len(texts) + 1)} rows of them; narrow the search with the requirement's value or page on to see the rest." if value is not None and texts else ''
        return '\n[Peace read the next pages of this search itself, one step a page]\n' + '\n'.join(texts) + rest if texts else ''

    def cap_ignored(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """Warn when a search's price cap was not applied to the rows it returned.
        Cap arguments are numeric arguments of the call whose names suggest a maximum and whose name or description
        is about money (priced). With the smallest such cap and a known price key, the result's listings priced
        above it are counted. When there are any, the note names the cap and the count and, when the call set enum
        string arguments, suggests listing the category with another option of those enums and the same cap."""
        props = (self.schemas.get(name) or {}).get('properties') or {}

        def priced(key: str) -> bool:
            """Say whether a parameter's name or description mentions price, cost, amount, budget, spend or money."""
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
        return f'\n[Peace] This search did not apply {caps[0]}={cap:g}: {len(over)} of its rows cost more{hint}.'

    def cheapest_here(self, observations: list[Any]) -> str:
        """For a cheapest pick, list the buyable priced rows in the given results, cheapest first.
        It needs a price key and at least two such rows, and names up to six with their prices. run_calls adds it to
        search replies, counting the pages that more_pages read."""
        if self.brief.get('pick') != 'cheapest' or not self.price_key:
            return ''
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ''
        return '\n[Peace] buyable rows here, cheapest first: ' + ', '.join((f'{i} ({p:g})' for p, i in sorted(rows)[:6]))

    def wider_hold(self) -> str:
        """Return a one-time message that holds the first checkout of a cheapest pick for a wider search.
        It applies once (wider_held), only for a cheapest pick and only with at least 14 plus twice the item count
        steps left. The text says the checkout did not run and asks for more searches under other names for each
        product, looking only at buyable rows priced below the cart's lines (listed with their prices), before the
        checkout. take_note clears wider_held after a change that can move the cheapest answer."""
        if self.wider_held or self.brief.get('pick') != 'cheapest' or self.left() < 14 + 2 * len(self.items):
            return ''
        self.wider_held = True
        lines = [f'{i} at {self.price_of(i):g}' for i in self.cart_ids() if self.price_of(i) is not None]
        return f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest. Searches may read only part of a listing, so a requirement's value finds only some of the fitting listings, and rows come in no price order. Before checking out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line" + (f" ({'; '.join(lines)})" if lines else '') + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        """Open listings whose attributes are not read yet, one facts-tool call each, and return their texts.
        It uses facts_tool and its single id parameter on the given ids that are known rows, not in self.full and
        not sold out. It stops when only keep steps are left, the episode finished, the tool turned out to show no
        attributes, or a call does not fit the schema or fails. Each reply goes through observe and note_view; when
        it is a single listing, facts_from for the tool becomes True if it carried an attribute block, and False
        when it did not and the tool was not already known to show them. The texts are id plus the shown reply,
        each cut to 1500 characters; a non-empty batch is logged as fill_facts.
        Callers: widen_singly, value_search, show_links, cheaper_rows, complete_cart, finish and the checkout path of
        run_calls, each passing the number of steps it wants to keep."""
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
        """Follow the part references of listings by searching the part code, then open the parts found.
        It needs a value_template search plan and either two or more items or two or more ids. For each id
        whose text sends the reader on to a part code (the more mark of _fit_marks) that was not tried
        before and that no seen listing carries as its own code, it searches the code in the plan's text
        field (one step), logs show_link, and opens with fill_facts the listings that now carry that code
        as their own (keeping at least the tail plus six steps). It stops at keep steps left, when
        finished, or on a call that does not fit or fails. complete_cart, finish and the checkout path of
        run_calls call it on the cart lines."""
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
        """Find a tool that lists a category under a price cap, with the fixed arguments it needs.
        A tool qualifies with exactly one numeric parameter whose snake-case name matches CAP_ARG and not
        CAP_NOT, and exactly one free-text parameter whose name matches SHELF_ARG. Every enum parameter with two
        or more options needs a value: the one used in the latest search that set the shelf parameter and that
        key with no free-text query besides the shelf, else the single option matching LIST_MODE. When no such
        value is found the whole lookup gives None. Returns (tool, fixed args, shelf parameter, cap parameter)
        for the first qualifying tool."""
        for tool, schema in self.schemas.items():
            props = (schema or {}).get('properties') or {}
            caps = [k for k, v in props.items() if isinstance(v, dict) and {'integer', 'number'} & set(_kinds(v)) and re.search(CAP_ARG, _snake(k), re.I) and not re.search(CAP_NOT, k, re.I)]
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
                pick = pick if pick is not None else named[0] if len(named) == 1 else None
                if pick is None:
                    return None
                base[key] = pick
            return (tool, base, shelves[0], caps[0])
        return None

    def list_below(self, cart: list[str]) -> None:
        """List each cart line's category below that line's price, to bring cheaper candidates into view.
        It needs cap_template and a category key. For each line with a known price of at least 2 and a category
        label, not yet listed at that price (self.capped), it calls the tool with the label and a cap just under the
        price (one less for a whole price, else the price rounded down) and observes the reply. When the reply has a
        next page, its total is at most twice the rows shown and the rows are not sorted by price, one more page is
        read. It stops when only the tail plus five steps are left or the episode finished."""
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

    def search_below(self, cart: list[str]) -> None:
        """For a cheapest pick in a store that cannot list a category under a price: run up to two fresh searches for the
        cart's items (the item's requirement values with its name, else with its shelf; a query already tried is skipped), so
        cheaper fitting rows the earlier searches never brought up can be read before the checkout. It keeps room for the swap
        and the checkout and records each search."""
        plan = self.search_template() or self.value_template()
        cat_key = self.category_key()
        if not plan or not cat_key:
            return
        search, base, field = plan
        tried = self.tried_queries(search, field)
        done = 0
        asked: set[int] = set()
        for line in cart:
            number = self.owner_of(str((self.rows.get(line) or {}).get(cat_key) or ''))
            if not number or number in asked or self.price_of(line) is None or done >= 2 or self.left() <= self.tail() + 6 + len(cart) or self.finished:
                continue
            asked.add(number)
            query = next((q for q in self.item_queries(number) if _words(q) not in tried), '')
            if not query:
                continue
            tried.add(_words(query))
            faults: list[str] = []
            args = _shape(self.item_search_args(number, base, field, query), self.schemas.get(search), 'args', faults)
            if faults:
                continue
            reply = self.act(search, args)
            done += 1
            if reply is None:
                return
            self.observe(search, reply.get('observation'))
            self.trace.append({'kind': 'search_below', 'line': line, 'query': query})

    def cheaper_rows(self) -> list[str]:
        """Open unread rows that undercut the cart's lines before a cheapest-pick checkout, and return their texts.
        It runs once per cart mark and version and needs a category key. list_below runs first. Then, for each cart
        line with a price and a label, the two cheapest seen rows with the same label that are not in the cart, not
        opened, buyable and priced below the line are chosen; fill_facts opens up to min(6, CHEAPER_OPENS per cart
        line) of them, keeping the tail plus four steps.
        run_calls adds its texts to the cart review before a checkout.
        Without a capped category listing, search_below searches again first."""
        if self.brief.get('pick') != 'cheapest':
            return []
        mark = self.cart_mark() + f'#{self.version}'
        cat_key = self.category_key()
        if mark in self.cheaper_marks or not cat_key:
            return []
        self.cheaper_marks.add(mark)
        cart = self.cart_ids()
        self.list_below(cart)
        if self.cap_template() is None:
            self.search_below(cart)
        picks: list[str] = []
        for line in cart:
            price = self.price_of(line)
            kind = str((self.rows.get(line) or {}).get(cat_key) or '').strip().lower()
            if price is None or not kind:
                continue
            pool = [i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i) and (str((self.rows.get(i) or {}).get(cat_key) or '').strip().lower() == kind) and ((self.price_of(i) if self.price_of(i) is not None else float('inf')) < price)]
            pool.sort(key=lambda i: self.price_of(i) or 0.0)
            picks += [i for i in pool[:2] if i not in picks]
        return self.fill_facts(picks[:min(6, CHEAPER_OPENS * max(1, len(cart)))], self.tail() + 4)

    def observe(self, name: str, observation: Any) -> str:
        """File a result's records and read any loose note text in it; return what the note meant, or ''.
        absorb stores the records; loose_lines finds note-like text not read before; take_note interprets it. A
        meaning is also queued, tagged with the tool's name, so the next reply to the model can mention it."""
        self.absorb(observation)
        lines = self.loose_lines(observation)
        if not lines:
            return ''
        meaning = self.take_note(lines)
        if meaning:
            self.queued.append(f'(seen on {name}) {meaning}')
        return meaning

    def widen(self) -> None:
        """Before the interview, open more listings in one side-by-side call so the planner sees more attribute keys.
        It needs a compare tool whose only required parameter is an array and that compare_bare does not rule out;
        otherwise widen_singly runs instead. With twelve or fewer steps left nothing is done. Up to the array's
        maxItems (4 when not given) unopened ids are drawn evenly from item_groups, and at least two are needed.
        The reply is observed and passed to note_view, and facts_from for the tool is updated from its records.
        When the tool turns out to show records without attributes, widen_singly runs; when nothing new was opened
        from more than two ids and steps allow, the call is repeated with the first two ids.
        interview calls it once, before planning the questions."""
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
        """Open listings one at a time with the facts tool, spread over the item groups, before the interview.
        Each group from item_groups gets a lane of unopened buyable listings, ordered by price for a cheapest pick
        and then (stable) by how many words of the current requirement values the row's text contains. A lane holds
        up to WIDEN_EACH (4 with more than two items) minus the listings of that group already opened, and its
        first 4 minus opened entries form the head. The heads, taken round-robin over the lanes, are opened first,
        up to max(5, 2 per lane plus 2) and at most eight, while more than 22 steps remain; the rest, round-robin
        too, follow up to ten reads in all while more than 30 steps remain."""
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
        """Say whether the interview should start now.
        Never after it ran, without an ask tool or free messages, with eight or fewer steps left or after ASK_BY
        seconds. Otherwise yes once self.full holds at least three opened listings per item plus one, or once the
        step count reaches max(8, a quarter of max_steps).
        work checks it at the start of every loop turn.
        Once ASK_EARLY seconds have passed it starts whatever the ledger holds: a store whose records never fill the
        ledger, or slow turns, would otherwise run past ASK_BY with the messages unused."""
        if self.asked or not self.roles.get('ask') or self.msgs_free() <= 0:
            return False
        if self.left() <= 8 or self.clock() > ASK_BY:
            return False
        opened = len(self.full)
        if opened >= 3 * len(self.items) + 1:
            return True
        if self.clock() >= ASK_EARLY:
            # too slow to wait for more records: ask while there is still time for every message
            return True
        return self.step >= max(8, self.max_steps // 4)

    def message_args(self, text: str) -> dict[str, Any] | None:
        """Build the arguments of one message to the shopper, or None when the ask tool cannot carry it.
        The text goes into the required free-text parameter, or else the first free-text parameter; every other
        required parameter must be an enum and gets its first option. The arguments are fitted with _shape, and any
        fault gives None. _shape cuts text at maxLength, which fit_ask detects by looking for the text unchanged."""
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
        """Run the interview: ask the shopper about unwritten preferences and return a summary for the loop model.
        It marks the interview as done (asked) and returns '' when no message is free. widen first opens more
        listings. plan_lanes then ranks property labels per item, at least LIST_SIZE per item while list questions
        are on, else as many as messages remain, and the lane counters and follow-ups are reset. ask_next sends one
        question at a time while messages are free, the episode is open, more than six steps are left and the clock
        is before QUESTION_BY. Items whose lanes found new requirements then get a value_search. When anything new
        was learned the version goes up, which voids the read cache and the gate marks. The summary lists each
        question and answer, the new firm requirements with an instruction to apply them, the pointer requirements
        among them that must be resolved through other listings, the value search results, and the notes queued
        meanwhile (the queue is then cleared).
        work calls it when ready_to_ask says so, or first when the model tries to cart, buy or decline before the
        interview ran. lanes maps each item to its labels; lane_found counts answers that gave a requirement,
        lane_spent the messages spent and lane_used the labels already asked, per item.
        Every lane label starts with the weight _prior gives its rank (lane_weight); settle lowers it for each
        list that names the label without an answer about it.
        When the plan and the key table leave every lane empty (an opened page that showed one attribute or none),
        bare_lanes plans again from the products themselves, so the allowed messages are not left unused."""
        self.asked = True
        if self.msgs_free() <= 0:
            return ''
        self.widen()
        count = self.msg_limit - self.msg_used
        self.lanes = self.plan_lanes(max(count, LIST_SIZE) if self.listing else count)
        if not any(self.lanes.values()):
            # nothing to ask from the plan or the table: ask what buyers of these products choose between
            self.lanes = self.bare_lanes(max(count, LIST_SIZE) if self.listing else count)
        self.lane_found = {n: 0 for n in self.lanes}
        self.lane_spent = {n: 0 for n in self.lanes}
        self.lane_used = {n: set() for n in self.lanes}
        self.lane_weight = {n: {_words(label): _prior(rank) for rank, label in enumerate(lane)} for n, lane in self.lanes.items()}
        self.follow_up = []
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        while self.msgs_free() > 0 and (not self.finished) and (self.left() > 6) and (self.clock() < QUESTION_BY):
            batch = self.ask_next(told)
            if not batch:
                break
            sent += batch
        found_items = [n for n in self.lane_found if self.lane_found[n] > 0]
        searched = self.value_search(found_items) if found_items else ''
        if told:
            self.version += 1
        lines = [f'Q: {q}\nA: {a}' for _, q, a in sent]
        summary = "Peace's interviewer asked the shopper (each question used one step):\n" + '\n'.join(lines)
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

    def late_question(self) -> str:
        """Spend one more shopper message just before a checkout, and return a hold message when it adds requirements.
        It runs once (late_asked) and needs an ask tool with a message left, no order-stage process step that needs
        messages, at least LATE_STEPS steps, a clock within LATE_BY and an affordable reviewer (judges). When no
        lanes were planned yet, plan_lanes runs for LIST_SIZE labels (one when list questions are off). ask_next
        then sends one question, keeping LATE_STEPS minus two steps, and a budget change from the answer counts as
        news. Without news the result is '' and the checkout goes ahead. Otherwise the version goes up and the text
        gives the question, the answer and the new requirements, says the cart was not checked out and must be
        checked against them, names pointer requirements to resolve, and adds a value search for the item asked
        about. Each attempt is logged as late_ask.
        run_calls asks for it when the model checks out, after the other checkout holds have passed.
        When it plans the lanes itself, it also gives every label its starting weight (_prior of its rank).
        It falls back to bare_lanes the same way when its own plan comes back empty."""
        if self.late_asked or not self.roles.get('ask') or self.msg_used >= self.msg_limit:
            return ''
        if any((p['when'] == 'order' and p['messages'] > 0 for p in self.process)):
            return ''
        if self.left() < LATE_STEPS or self.clock() > LATE_BY or not self.judges():
            return ''
        self.late_asked = True
        if not self.lanes:
            self.lanes = self.plan_lanes(LIST_SIZE if self.listing else 1)
            if not any(self.lanes.values()):
                self.lanes = self.bare_lanes(LIST_SIZE if self.listing else 1)
            self.lane_found = {n: 0 for n in self.lanes}
            self.lane_spent = {n: 0 for n in self.lanes}
            self.lane_used = {n: set() for n in self.lanes}
            self.lane_weight = {n: {_words(label): _prior(rank) for rank, label in enumerate(lane)} for n, lane in self.lanes.items()}
        told: list[str] = []
        budget = self.budget
        batch = self.ask_next(told, LATE_STEPS - 2)
        if not batch:
            self.trace.append({'kind': 'late_ask', 'skipped': 'no property left to ask or no message sent'})
            return ''
        if self.budget != budget:
            told.append(f'budget now {self.budget}')
        number, question, answer = batch[0]
        self.trace.append({'kind': 'late_ask', 'q': question, 'a': answer[:200], 'told': told})
        if not told:
            return ''
        self.version += 1
        summary = f'Peace asked the shopper one more question before checking out (one step):\nQ: {question}\nA: {answer}\nNew firm requirements from this answer: ' + '; '.join(told) + '. The cart was not checked out: check each line against them (search again with the new values where needed), fix the cart, then check out.'
        pointed = [need for _, need in self.refs if any((need in t for t in told))]
        if pointed:
            summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value.'
        searched = self.value_search([number])
        return summary + ('\n' + searched if searched else '')

    def search_template(self) -> tuple[str, dict[str, Any], str] | None:
        """Work out from past searches how to send a keyword search, as (tool, fixed args, text field), or None.
        It uses the search role. Text fields are its free-text string parameters, preferring those whose snake-case
        names match QUERY_WORDS. Past searches are read newest first, and the first with a worded value in such a
        field gives the plan. When no parameter name looks like a query, a field counts only if past searches put at
        least two different values in it, one of them two or more words long, and never a category label seen on
        the rows. The fixed args keep the call's enum and required arguments other than the field, minus
        non-required strings that name an item's shelf or name or a seen category label; each enum argument takes
        the value used most often among the worded searches.
        value_template and find_link build their searches from it."""
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
                return (tool, keep, field)
        return None

    def value_words(self, number: int) -> list[tuple[str, bool]]:
        """List the values of an item's requirements in force that could be typed into a search, with a told flag.
        Pointer requirements and needs without a colon are skipped. A lower bound (FLOOR: at least, no less or no
        fewer than, min or minimum, then an amount at the end) gives just the amount, and only when the shopper
        told it. Values with negation, comparison, range, pointer or or-words are skipped. Each entry is (value
        text, True when the requirement came from item.told)."""
        out = []
        told = self.items[number - 1].told
        for n, need in self.needs_now():
            if n != number or need in [m for _, m in self.refs] or ':' not in need:
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
        """Return the search plan used for value searches: search_template's, or one built from the schema.
        When search_template has no plan, there must be a search tool and at least one past search;
        otherwise None. The tool then needs exactly one free-text parameter named like a query
        (QUERY_WORDS). For each enum parameter with two or more options a mode is chosen: the single
        option whose own text matches QUERY_WORDS, or, for a required two-option enum, the option the last
        search did not use. More than one mode, or enums with no mode at all, give None. The fixed args
        are the last search's required arguments except the query, plus the modes. value_search and
        show_links build their searches from it."""
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

    def value_search(self, numbers: list[int]) -> str:
        """Search with the values the shopper just gave for the given items, and open the first fitting-kind rows.
        It needs value_template and at least 22 steps left, and spends at most eight calls in all. For each item
        with a told value (while more than two calls remain), one-word values and amounts lead the query and the
        item's other values follow. Up to two queries are tried: all values, then all but the last value the
        request stated (the first value when none was stated); a single told value that is one word or an amount
        is tried with the item's name first, then alone. Each query is one search. Its buyable rows of the item's
        category within the budget (the budget is ignored when every price is over ten times it) are listed,
        home-label rows first, and up to three unread ones are opened with fill_facts, keeping the tail plus 16
        steps; the first query that finds such rows ends the item. The lines come back under a header and are
        logged as value_search, or '' when nothing was searched.
        interview runs it for the items whose lanes found something, late_question for the item it asked about."""
        plan = self.value_template()
        if not plan or self.left() < 22:
            return ''
        tool, base, field = plan
        seen: list[str] = []
        budget = 8
        for number in numbers:
            pairs = self.value_words(number)
            item = self.items[number - 1]
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
                home = self.home_label(number)
                if cat_key and home:
                    rows.sort(key=lambda i: _words(str((self.rows.get(i) or {}).get(cat_key) or '')) != home)
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
        return "[Peace searched each such item's requirement values together and read the first fitting-kind rows] " + '\n'.join(seen)

    def home_label(self, number: int) -> str:
        """Return the category label, in _words form, where an item's listings mostly sit, or ''.
        Opened records are counted first, then all rows. Among the labels owned by the item, its shelf or its name
        wins when present; else the most frequent label of that pool. value_search sorts rows by it."""
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

    def next_ask(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> tuple[int, list[str]]:
        """Choose the item and labels for the next question from the lanes, or (0, []) when nothing is left.
        Items are ordered by whether their lane already found a requirement (those go last), then by the messages
        spent on them (ignored when messages are scarce: msg_limit at most SCARCE_MESSAGES with list questions off),
        then by plan_order, which holds the items with hinted labels. The first item with labels neither used nor
        already told gives up to LIST_SIZE of them while list questions are on, else one."""
        first = {n: i for i, n in enumerate(self.plan_order)}
        scarce = self.msg_limit <= SCARCE_MESSAGES and (not self.listing)
        for number in sorted(lanes, key=lambda n: (found.get(n, 0) > 0, 0 if scarce else spent.get(n, 0), first.get(n, len(first) + n))):
            left = [label for label in lanes[number] if _words(label) not in used[number] and (not self.told_about(number, label))]
            if left:
                return (number, left[:LIST_SIZE if self.listing else 1])
        return (0, [])

    def told_about(self, number: int, label: str) -> bool:
        """Say whether the shopper already gave the item a requirement on the property a label names.
        The label and the property of each told requirement are compared with _spelled and _names."""
        mine = _spelled(label)
        return any((_names(mine, _spelled(t.split(':', 1)[0])) for t in self.items[number - 1].told))

    def ask_text(self, number: int, labels: list[str], shown: bool=False) -> str:
        """Phrase a question about one or more labels for an item.
        Two or more labels make one list question (What a, b or c would you like for the item?); a single label
        uses question(), offering two sample values when shown is set.
        Each property is written by with_spelling, so a merged key carries its other spelling in brackets."""
        if len(labels) >= 2:
            named = [self.with_spelling(label) for label in labels]
            return f"What {', '.join(named[:-1])} or {named[-1]} would you like for the {self.item_said(number)}?"
        return self.question(number, labels[0], self.samples(number, labels[0]) if shown else None)

    def fit_ask(self, number: int, labels: list[str], shown: bool) -> tuple[list[str], str]:
        """Find the longest leading run of labels whose question the ask tool can carry, and return (labels, text).
        A text fits when message_args accepts it and carries it unchanged, not cut at a maximum length. When even a
        single label does not fit, the result is ([], '')."""
        for size in range(len(labels), 0, -1):
            text = self.ask_text(number, labels[:size], shown)
            args = self.message_args(text)
            if args and text in args.values():
                return (labels[:size], text)
        return ([], '')

    def ask_next(self, told: list[str], floor: int=6) -> list[tuple[int, str, str]]:
        """Send the next interview question, read the answer into requirements, and return what was sent.
        Up to LIST_SIZE times it takes a follow-up (dropping those the shopper already answered) or else the next
        labels from next_ask, and tries fit_ask with sample values, then without; a label whose question never fits
        is marked used and logged as ask_too_long. With no labels left, or no fitting text, it returns []. The
        question goes out through send, keeping floor steps, and the item's spent count goes up. hear reads the
        answer with LIGHT, then HEAVY; for a list question the do-not-mind entries are ignored. take_heard applies
        the result and adds the new requirements to told, and every item that gained one has its lane_found count
        raised. Whether the answer called the question crowded is read from the reply's crowded list, and settle
        updates the lane. Returns the (item, question, answer) entries sent.
        interview calls it in a loop, and late_question once."""
        for _ in range(LIST_SIZE):
            self.follow_up = [f for f in self.follow_up if not self.told_about(f[0], f[1][0])]
            if self.follow_up:
                number, labels, shown = self.follow_up.pop(0)
            else:
                number, labels = self.next_ask(self.lanes, self.lane_found, self.lane_spent, self.lane_used)
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
        self.lane_spent[number] = self.lane_spent.get(number, 0) + 1
        had = [len(item.told) for item in self.items]
        heard = self.hear(batch, (LIGHT, HEAVY))
        if len(labels) > 1:
            heard.pop('free', None)
        self.take_heard(heard, told)
        new = [need for item, size in zip(self.items, had) for need in item.told[size:]]
        for n, (item, size) in enumerate(zip(self.items, had), 1):
            if len(item.told) > size:
                self.lane_found[n] = self.lane_found.get(n, 0) + 1
        crowded = any((_count(x) == 1 for x in _items(heard.get('crowded'))))
        self.settle(number, labels, shown, new, crowded)
        return batch

    def settle(self, number: int, labels: list[str], shown: bool, new: list[str], crowded: bool) -> None:
        """Record what an answer showed about the labels its question named, and order the item's lane for the next list.
        A single label is marked asked. When its reply gave nothing new and asked for a simpler question (crowded), a
        follow-up is queued: the same label about one product of an item name that joins several (once per item), else
        the label's last word when it has several and that word was not asked, else the same label with example values. A list whose reply only asked for one question at a time switches lists off for the
        episode and queues the list's first label alone. Otherwise the label the reply answered (answered_label) is
        marked asked, or the list's first label when nothing new came (it was read with a chance of about 0.9). The
        shopper reads a list up to the property it answers, so every label before the answered one, or every label of a
        list that drew nothing new, has its weight multiplied by the chance it was missed at its place (1 - _read).
        The item's remaining lane labels are then sorted heaviest first, keeping their order on ties, so the next list
        leads with the properties most likely still wanted."""
        used = self.lane_used.setdefault(number, set())
        if len(labels) == 1:
            used.add(_words(labels[0]))
            if new or not crowded:
                return
            if number not in self.one_product and self.name_parts(number):
                self.one_product.add(number)
                self.trace.append({'kind': 'one_product', 'item': number, 'said': self.item_said(number)})
                self.follow_up.insert(0, (number, labels, shown))
                return
            head = labels[0].split()[-1]
            if len(labels[0].split()) >= 2 and _words(head) not in used:
                self.follow_up.insert(0, (number, [head], False))
            elif not shown and len(self.samples(number, labels[0])) >= 2:
                self.follow_up.insert(0, (number, labels, True))
            return
        if crowded and not new:
            self.listing = False
            self.trace.append({'kind': 'lists_off', 'item': number})
            self.follow_up.insert(0, (number, labels[:1], False))
            return
        hit = self.answered_label(labels, new) if new else -1
        used.add(_words(labels[max(hit, 0)]))
        weight = self.lane_weight.setdefault(number, {})
        # the shopper read the list up to the label it answered (all of it when nothing new came): each label read and
        # passed over is less likely to be wanted
        for place, label in enumerate(labels[:hit] if hit >= 0 else labels, 1):
            weight[_words(label)] = weight.get(_words(label), 1.0) * (1.0 - _read(place))
        lane = [label for label in self.lanes.get(number) or [] if _words(label) not in used]
        self.lane_misses[number] = 0 if hit >= 0 else self.lane_misses.get(number, 0) + 1
        self.lanes[number] = sorted(lane, key=lambda label: -weight.get(_words(label), 1.0))
        self.trace.append({'kind': 'list_settled', 'item': number, 'done': labels[:hit + 1], 'next': self.lanes[number][:4]})

    def answered_label(self, labels: list[str], new: list[str]) -> int:
        """Return the index of the first label of a list question that a new requirement answers, or -1.
        For each new requirement, labels are scored by the _spelled words they share with its property;
        when no label shares any, by the roots shared between the requirement's value and the sample
        values that key_table shows for that label. The best-scoring label of each requirement is a hit,
        and the lowest hit index is returned."""
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
        """Rank the attribute keys of an item's category as interview labels to fall back on.
        It takes the key_table rows whose label belongs to the item (all rows when there is one item and none
        match). Rows sort by how many listings carry the key, then plain keys before measured ones (keys whose
        values are mostly a number with a short unit), then by more sample values. Keys that a request requirement
        already names, and keys matching NOT_PROPERTY (ids, codes, links, images, warranty and the like), are
        dropped; catch-all keys (CATCH_ALL) are moved to the end."""
        item = self.items[number - 1]
        settled = [_stem(m.split(':', 1)[0]) for m in item.musts]
        rows = [(label, key, carried, len(values)) for label, key, carried, values in table if self.owner_of(label) == number]
        measured = {key: sum((1 for v in values if re.match('^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$', v))) * 2 > len(values) for _, key, _, values in table}
        if len(self.items) == 1 and (not rows):
            rows = [(label, key, carried, len(values)) for label, key, carried, values in table]
        rows.sort(key=lambda r: (-r[2], measured.get(r[1], False), -r[3]))
        keys = [key for _, key, _, _ in rows if not any((_names(prop, _stem(key)) for prop in settled)) and (not re.search(NOT_PROPERTY, key))]
        return [key for key in keys if not re.search(CATCH_ALL, key)] + [key for key in keys if re.search(CATCH_ALL, key)]

    def adopt_labels(self) -> None:
        """Hand the category labels no item claims to the one item that has none, when that is unambiguous.
        With two or more items and a category key, labels of opened records that owner_of gives to no item are
        strays, and items with no label of their own (adopted labels do not count) are orphans. When there are
        strays and exactly one orphan, every stray is mapped to it in self.adopted and the mapping is logged.
        plan_lanes and line_faults call it first."""
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
        """Return the number of the item a category label belongs to, or 0.
        An adopted label gives its item. Otherwise each item is scored: 3 when the label equals its shelf (by
        _words), 2 when it equals its name, else the overlap of _roots between the label and the item's shelf and
        name, divided by their union. The highest positive score wins, and a tie goes to the earlier item.
        owns, key_table, item_groups, lane_keys and the cart checks decide which item a label belongs to through it."""
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
        """Say whether a non-empty category label belongs to the given item, by owner_of."""
        return bool(label) and item in self.items and (self.owner_of(label) == self.items.index(item) + 1)

    def records_of_item(self, number: int) -> int:
        """Count the item's opened records: those with an attribute block whose category label it owns.
        The count is 0 when no category key is known."""
        cat_key = self.category_key()
        item = self.items[number - 1]
        return sum((1 for record in self.full.values() if cat_key and self.owns(item, str(record.get(cat_key) or '')) and _fact_block(record)))

    def bare_lanes(self, count: int) -> dict[int, list[str]]:
        """Plan the interview when nothing else gave a property to ask: one planner call that says no attribute keys were
        seen and asks, for each item, for the properties a buyer of that kind of product usually chooses between. The
        labels are cut and kept as plan_lanes keeps them; the lanes go into the trace."""
        settled = [f'item {n}: {item.name} - settled by the request: ' + ('; '.join(item.musts) or 'nothing') for n, item in enumerate(self.items, 1)]
        prompt = PLAN.replace('COUNT', str(max(1, count))) + '\n\nREQUEST:\n' + self.request + '\n\nITEMS:\n' + '\n'.join(settled)
        prompt += ('\n\nATTRIBUTE KEYS SEEN:\n(none yet: no opened listing of these items showed its attributes; name for each item '
                   'the properties a buyer of that kind of product usually chooses between, in plain words)')
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 3500) or {}
        lanes: dict[int, list[str]] = {n: [] for n in range(1, len(self.items) + 1)}
        for pick in _items(plan.get('ask')):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, _count(pick.get('item') or 1) or 1), len(self.items))
            label = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(pick.get('label') or ''))[0].strip().rstrip('?.')
            if label and len(label.split()) <= 5 and all((_words(label) != _words(other) for other in lanes[number])):
                lanes[number].append(label)
        self.trace.append({'kind': 'bare_plan', 'lanes': _clip(lanes, 60)})
        return lanes

    def plan_lanes(self, count: int) -> dict[int, list[str]]:
        """Plan the interview: an ordered list of property labels to ask about for each item (its lane).
        adopt_labels runs, then key_table. The PLAN prompt, with COUNT set to the given number, gets the
        request, each item with the requirements the request settles, the ASK FIRST list from the brief, and
        the numbered key list per category with how many listings carry each key, sample values, the
        not_stated marker and a note on catch-all keys; HEAVY, then LIGHT, answers. Picks are cut at the first
        and, or, slash, comma, semicolon or bracket, dropped when empty or longer than five words, and
        deduplicated per item; a pick with a hint is recorded as hinted and puts its item in plan_order. Per
        item, picks are respelled to the table key with the same plain form when one exists. Then up to
        GUARD_KEYS of the item's most carried keys (lane_keys order, share of the item's opened records at
        least GUARD_SHARE, not catch-all) are guarded into the first three places: when one of those places
        holds an unhinted pick whose share is GUARD_GAP or more below the key's, the key takes it (with
        several items the displaced pick moves to fourth place) or, with one item, is inserted there; a key is
        also added when the lane has fewer than three labels; a placement that would push a hinted pick out of
        the first three is skipped. The remaining lane_keys follow, catch-all labels go last, and the plan is
        logged. interview and late_question call it."""
        self.adopt_labels()
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
            for key in keys:
                if all((_words(key) != _words(other) for other in lanes[number])):
                    lanes[number].append(key)
            lanes[number] = [x for x in lanes[number] if not re.search(CATCH_ALL, _words(x))] + [x for x in lanes[number] if re.search(CATCH_ALL, _words(x))]
        self.trace.append({'kind': 'plan', 'ask': _clip(plan.get('ask'), 120), 'lanes': {n: v[:8] for n, v in lanes.items()}})
        return lanes

    def name_parts(self, number: int) -> list[str]:
        """Split an item name that joins several plural products with and (or commas) into its parts.
        The name must contain the word and. It is split at commas and at and, and the parts come back only when there
        are two or more and each ends in a plural s after three letters (not ss or us); otherwise []."""
        name = self.items[number - 1].name
        if not re.search('\\band\\b', name, re.I):
            return []
        parts = [p.strip() for p in re.split('\\s*(?:,|\\band\\b)\\s*', name, flags=re.I) if p.strip()]
        return parts if len(parts) >= 2 and all((re.search('[a-z]{3}(?<![su])s$', p, re.I) for p in parts)) else []

    def item_said(self, number: int) -> str:
        """Return how a question names an item.
        A name without plural parts is used as it is. For a name with parts, and becomes or, unless the
        shopper called a question crowded (the item is in one_product): then the one part whose last word
        shares the most roots with the item's category rows (their string values other than the label) is
        used, the earlier part winning a tie."""
        item = self.items[number - 1]
        parts = self.name_parts(number)
        if not parts:
            return item.name
        if number not in self.one_product:
            return re.sub('\\band\\b', 'or', item.name, flags=re.I)
        cat_key = self.category_key()
        rows = [row for row in self.rows.values() if not cat_key or self.owns(item, str(row.get(cat_key) or ''))]
        named = [_roots(' '.join((str(v) for k, v in row.items() if k != cat_key and isinstance(v, str)))) for row in rows]
        hits = {p: sum((1 for words in named if _roots(p.split()[-1]) & words)) for p in parts}
        return max(parts, key=lambda p: (hits[p], -parts.index(p)))

    def question(self, number: int, label: str, samples: list[str] | None=None) -> str:
        """Phrase a one-property question for an item, optionally offering two example values.
        The form is What LABEL would you like for the ITEM, ending in a question mark, or offering the first two
        samples as for example A or B.
        The property is written by with_spelling."""
        text = f'What {self.with_spelling(label)} would you like for the {self.item_said(number)}'
        return text + (f', for example {samples[0]} or {samples[1]}?' if samples else '?')

    def with_spelling(self, prop: str) -> str:
        """Name a property for a question: its own name, with the other spelling key_table kept for it in brackets.
        For a merged key this reads like fit (pants fit type), so the shopper's reader finds the property under
        whichever of the two names it knows; any other property comes back unchanged.
        The other spelling is added only when it reads as a property name: at most four words, each made of two or
        more letters (a key such as screen resolution e g 1920 x 1080 is left out)."""
        known = self.spellings.get(_words(prop))
        plain = bool(known) and len(known.split()) <= 4 and all((w.isalpha() and len(w) > 1 for w in known.split()))
        return f'{prop} ({known})' if plain and _words(known) != _words(prop) else prop

    def samples(self, number: int, label: str) -> list[str]:
        """Return up to two example values for a label from the key table, for a question that offers examples.
        The key whose name equals the label's words supplies them; values already present in the item's request
        requirements and values over 30 characters are left out. [] when no key matches."""
        want = _words(label)
        settled = ' '.join(self.items[number - 1].musts).lower()
        for _, name, _, values in self.key_table():
            if name == want:
                return [v for v in values if v and v not in settled and (len(v) <= 30)][:2]
        return []

    def send(self, questions: list[tuple[int, str, str]], floor: int=6) -> list[tuple[int, str, str]]:
        """Send interview or gate messages through the ask tool and return what was sent.
        Each question is (item number, label, text). Sending stops when the episode finished, floor or fewer
        steps are left, the message limit is used up, the arguments cannot be built, or a call fails. Each
        message raises msg_used, and the reply text comes from spoken(). A line with the step, the question, the
        reply (up to 300 characters) and the cart mark is appended to done_log for the gate, its step to
        done_steps, and the exchange to the trace. A reply that is an error object stops the batch after it.
        Returns (item, text, reply) for each message sent. ask_next sends interview questions through it, and
        gate the messages a process step needs.
        Note lines that come with a reply (a list of sentences next to it, such as a shopper easing off a
        requirement) go to take_note, and what they mean is queued for the loop like a note on any other tool result."""
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
            notes = [line for value in (answer.values() if isinstance(answer, dict) else []) if isinstance(value, list) for line in value if isinstance(line, str) and len(line.split()) >= 5 and line not in self.heard]
            if notes:
                meaning = self.take_note(notes)
                if meaning:
                    self.queued.append(f'(seen with a shopper reply) {meaning}')
            sent.append((number, text, words))
            self.done_log.append(f'step {self.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {self.cart_mark()})')
            self.done_steps.append(self.step)
            self.trace.append({'kind': 'ask', 'q': text, 'a': words[:300]})
            if isinstance(answer, dict) and answer.get('error'):
                break
        return sent

    def take_heard(self, heard: dict[str, Any], told: list[str]) -> None:
        """Apply a parsed shopper answer: new requirements, dropped ones, do-not-mind properties and budget.
        Each need is added to its item's told list when new, and to the caller's told list (prefixed with the item
        number when there are several items); a need marked as pointing at another listing joins self.refs. A drop
        marks every live requirement on that property (by _words of the part before the colon) as dropped and
        reports it. Free entries are recorded on the item. A positive numeric budget replaces self.budget. Item
        numbers are clamped to the valid range, defaulting to 1.
        A property name that comes back with the bracketed second spelling a question used keeps only its first
        name; brackets inside a value stay.
        A need whose property neither the question nor the answer names is not kept unless it points at another
        listing (an answer of no preference can carry chatter a reader takes for a wish); a need on price or budget that
        comes with a new budget is the budget itself; and a new budget passes sane_budget."""
        many = len(self.items) > 1
        for entry in _items(heard.get('needs')):
            if not isinstance(entry, dict) or not str(entry.get('need') or '').strip():
                continue
            number = min(max(1, _count(entry.get('item')) or 1), len(self.items))
            need = str(entry['need']).strip()
            prop, colon, value = need.partition(':')
            named = {w for w in re.findall('[a-z0-9]+', prop.lower()) if len(w) >= 3}
            if self.heard_words and named and (not self.names_heard(named)) and entry.get('points') is not True:
                # a property neither the question nor the answer names: not something the shopper stated
                continue
            if heard.get('budget') and set(_words(prop).split()) <= {'price', 'budget', 'cost', 'total', 'spend', 'amount'}:
                # a new spending limit stated in the reply: it is the budget, not a requirement on the listing
                continue
            if colon and '(' in prop:
                # questions name a property with its other spelling in brackets; a requirement keeps one name
                need = re.sub('\\s*\\([^()]*\\)', '', prop).strip() + ':' + value
            if need not in self.items[number - 1].told:
                self.items[number - 1].told.append(need)
                told.append(f'item {number}: {need}' if many else need)
            if entry.get('points') is True and (number, need) not in self.refs:
                self.refs.append((number, need))
        for entry in _items(heard.get('drop')):
            if isinstance(entry, dict) and str(entry.get('property') or '').strip():
                item = self.items[min(max(1, _count(entry.get('item')) or 1), len(self.items)) - 1]
                gone = _words(re.sub('\\s*\\([^()]*\\)', '', str(entry['property']).split(':')[0]))
                hit = [m for m in item.musts + item.told if m not in item.dropped and _words(m.split(':')[0]) == gone]
                item.dropped.extend(hit)
                told.extend((f'dropped {m}' for m in hit))
        for entry in _items(heard.get('free')):
            if isinstance(entry, dict) and entry.get('property'):
                number = min(max(1, _count(entry.get('item')) or 1), len(self.items))
                self.items[number - 1].free.append(re.sub('\\s*\\([^()]*\\)', '', str(entry['property'])).strip())
        if isinstance(heard.get('budget'), (int, float)) and (not isinstance(heard.get('budget'), bool)) and (heard['budget'] > 0):
            self.budget = self.sane_budget(heard['budget'], self.heard_text)

    def names_heard(self, words: set[str]) -> bool:
        """Say whether any of the given property words appears among the words of the last questions and answers.
        British and American spellings (colour, color) count as one, and two words that share their first five letters
        match (accessory, accessories)."""
        fold = lambda w: re.sub('our$', 'or', w)
        heard = {fold(h) for h in self.heard_words}
        return any((fold(w) == h or (len(w) >= 5 and len(h) >= 5 and fold(w)[:5] == h[:5]) for w in words for h in heard))

    def budget_in_text(self, text: str) -> float | None:
        """Find a new spending limit a note states in the budget's own unit, or None.
        It needs a budget in force, a unit from the brief, and a note that says nothing about a price; then exactly one
        amount followed within three words by a word of that unit (cents for a cents budget), within five times the
        budget in force. take_note uses it when the note reader returned no budget and no price change."""
        unit = [w[:-1] if len(w) > 3 and w.endswith('s') else w for w in re.findall('[a-z]+', str(self.brief.get('budget_unit') or '').lower())]
        old = self.budget if isinstance(self.budget, (int, float)) and (not isinstance(self.budget, bool)) and self.budget > 0 else None
        if not unit or not old or re.search('\\bpric|\\bcharg|\\bcost', text, re.I):
            return None
        found = set()
        for number, words in re.findall('(\\d[\\d,]*(?:\\.\\d+)?)\\s+((?:[a-z]+\\s+){0,2}[a-z]+)', text.lower()):
            said = [w[:-1] if len(w) > 3 and w.endswith('s') else w for w in words.split()]
            if any((w in unit for w in said)):
                found.add(float(number.replace(',', '')))
        if len(found) != 1:
            return None
        value = found.pop()
        return value if value > 0 and max(value / old, old / value) <= 5 else None

    def sane_budget(self, value: float, text: str) -> float:
        """Check a new spending limit read from a reply or a note against the text it came from.
        A value the text itself states, or one within three times the budget in force, is kept; otherwise the text's
        own number nearest the budget in force is used (a model that converted 2230 cents into 22.3 would otherwise
        leave every listing over budget). With no budget in force the value is kept as it is."""
        numbers = [float(x.replace(',', '')) for x in re.findall('\\d[\\d,]*(?:\\.\\d+)?', text or '')]
        numbers = [n for n in numbers if n > 0]
        old = self.budget if isinstance(self.budget, (int, float)) and (not isinstance(self.budget, bool)) and self.budget > 0 else None
        if not old or value in numbers:
            return value
        ratio = lambda c: max(c / old, old / c)
        if ratio(value) <= 3:
            return value
        best = min(numbers, key=ratio, default=value)
        return best if ratio(best) < ratio(value) else value

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(HEAVY, LIGHT)) -> dict[str, Any]:
        """Have a model read the shopper's answers into needs, drops, free properties, crowded complaints and budget.
        The HEAR prompt lists the items and the numbered questions and answers, each tagged with its item. The models
        are tried in the given order with up to 2500 tokens; an empty dict comes back on failure.
        ask_next passes LIGHT first, then HEAVY; the default order puts HEAVY first.
        It keeps the answers' text (heard_text) and the words of the questions and answers (heard_words) for
        take_heard's checks."""
        self.heard_text = ' '.join((a for _, _, a in sent))
        self.heard_words = {w for _, q, a in sent for w in re.findall('[a-z0-9]+', (q + ' ' + a).lower())}
        prompt = HEAR + '\n\nITEMS:\n' + '\n'.join((f'item {n}: {it.name}' for n, it in enumerate(self.items, 1)))
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f'{i}. (about item {n}) Q: {q}\nA: {a}' for i, (n, q, a) in enumerate(sent, 1)))
        return self.ask_json(prompt, list(models), 2500) or {}

    def open_offers(self) -> list[str]:
        # returns the cart lines an offer page showed sold out, so the checkout can be held for a swap
        """Before a checkout, open each cart line's offer page with the offer tool when the store has one apart from its
        attribute, cart, search and ending tools and the line has not been opened (or tried) with it yet. One step per line,
        and only when the checkout and the order gate's own steps (tail) still fit after it. A store that splits the page
        counts a line as looked at only on its offer page. Returns the lines an offer page showed sold out."""
        tool = self.roles.get('offer')
        others = {self.roles.get(r) for r in ('open', 'cart', 'cart_view', 'buy', 'refuse', 'ask', 'search', 'compare')}
        if not tool or tool in others or tool not in self.schemas or self.finished:
            return []
        target = self.single_target(tool)
        missing = [i for i in self.cart_ids() if i not in self.offered_ids and i not in self.offer_tried and (not self.sold_out(i))]
        # the checkout and every step the order gate still needs (an approval message) come first
        if not target or not missing or self.left() < len(missing) + self.tail():
            return []
        gone = []
        for ident in missing:
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
            if faults or self.left() < self.tail() + 1:
                break
            self.offer_tried.add(ident)
            reply = self.act(tool, args)
            if reply is None or self.finished:
                break
            self.observe(tool, reply.get('observation'))
            self.trace.append({'kind': 'offer_view', 'line': ident})
            if self.sold_out(ident):
                gone.append(ident)
        return gone

    def gate(self, stage: str) -> tuple[bool, str]:
        """Carry out the request's own process steps due at the order or at a decline, and say whether to go on.
        Returns (True, '') when no step has that stage, the episode is finished, or the stage already passed for
        this cart and version. Otherwise it runs up to GATE_ROUNDS rounds. The GATE prompt gets the request, the
        due steps, the ledger, the cart, the numbered record of messages and step calls (done_log), recent notes,
        the messages and steps left, and the tools other than the ending ones; LIGHT answers first, HEAVY first
        after a nudge. Once something was done in an earlier round, needs and a hold in the reply are applied: if
        either is present the version goes up and (False, text) comes back, so the model fixes the cart first.
        For an order step that needs messages, only record lines made after the cart last changed count as proof.
        When the reply gives no call although some step lacks proof, nothing was done yet, and a message and two
        steps remain, Peace adds a nudge line to the record and asks again with HEAVY first. A reply with no call
        marks the stage as passed in gate_ok. A message call goes out through send (counted in gate_msgs); any
        other tool is called, and its result is absorbed and added to the record. The rounds also stop on an
        unknown or ending tool, fewer than two steps, a message call with none left, or arguments that do not fit.
        At the end it returns (True, a summary of what Peace did, or '').
        A read-only call the gate's model asks for (not the cart, checkout, ending or message tool) that Peace already made
        with the same arguments after the cart and the store's state last changed (cart_at, changed_at) is not repeated: the earlier call (called_at) goes into the record
        as done, so a page opened just before the checkout does not cost a second step."""
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
                return (False, ('Before going on, Peace did the steps the request asks for:\n' + '\n'.join(said) + '\n' if said else '') + ("The shopper's reply changes the task: " + '; '.join(told + ([hold] if hold else [])) + '. Fix the cart for it, then check out again.'))
            name = str((call or {}).get('tool') or '')
            shown = [e for e in _items(got.get('steps')) if isinstance(e, dict)]
            talk = stage == 'order' and any((p['messages'] > 0 for p in steps))
            lines = range(1, len(self.done_log) + 1)
            proof = {i for i in lines if not talk or (i <= len(self.done_steps) and self.done_steps[i - 1] > self.cart_at)}
            unproven = [e for e in shown if e.get('done') is not True or not any(((_count(n) or 0) in proof for n in _items(e.get('lines'))))]
            if not call and unproven and (not said) and (not nudged) and self.roles.get('ask') and (self.msg_used < self.msg_limit) and (self.left() >= 2):
                nudged = True
                self.trace.append({'kind': 'gate_unproven', 'steps': _clip(unproven, 120)})
                self.done_log.append(f'(Peace: no line made since the cart last changed at step {self.cart_at} shows ' + '; '.join((str(e.get('step') or '') for e in unproven))[:200] + f' done, and {self.msg_limit - self.msg_used} message(s) and {self.left()} steps are left: give the call that does it)')
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
            changing = {self.roles.get(r) for r in ('cart', 'buy', 'refuse', 'ask')} - {None}
            earlier = self.called_at.get((name, json.dumps(args, sort_keys=True, default=str)))
            if name not in changing and earlier is not None and earlier > max(self.cart_at, self.changed_at):
                # the same read was already made after the cart last changed: record it instead of repeating it
                self.done_log.append(f'step {earlier}: {name} {_dump(args)} (made by Peace after the cart last changed; cart then: {self.cart_mark()})')
                self.done_steps.append(earlier)
                self.trace.append({'kind': 'gate_reused', 'call': name, 'at': earlier})
                continue
            reply = self.act(name, args)
            if reply is None:
                break
            seen = reply.get('observation')
            self.absorb(seen)
            self.done_log.append(f'step {self.step}: {name} {_dump(args)} -> {self.show(seen)[:300]} (cart then: {self.cart_mark()})')
            self.done_steps.append(self.step)
            said.append(f'{name} {_dump(args)} -> {self.show(seen)[:400]}')
        return (True, 'Before going on, Peace did the steps the request asks for:\n' + '\n'.join(said) if said else '')

    def spoken(self, observation: Any) -> str:
        """Return the readable text of a store reply, for shopper answers and refusals.
        A string comes back whole. Otherwise every string of two or more words within four levels is collected and
        joined, cut to 800 characters; when there is none, the reply's JSON cut to 400 characters is used.
        send uses it for answers, and run_calls to see whether a refused checkout repeats a known sale-out note."""
        if isinstance(observation, str):
            return observation
        found: list[str] = []

        def walk(node: Any, depth: int) -> None:
            """Collect the strings of two or more words in one node, recursing to depth 4."""
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
        """Return the note-like text lines of a store result that Peace has not read before.
        Only a dict result that is not itself a listing record is looked at: its string values of five or more
        words, such strings inside a list of strings, and strings of six or more words inside nested dicts that are
        not listing records. Lines already in self.heard are left out.
        observe and run_calls hand its lines to take_note."""
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
        """Read a note the store showed with a model and apply the changes it reports; return a short meaning.
        The lines are marked as heard. The NOTE prompt with the ledger and the lines goes to LIGHT, then SPARE.
        Listings named as gone that are known rows become unavailable; note_sold is set when any are named or the
        note says a chosen listing sold out, and note_named when a named one is a known row. A one-line sale-out
        note is kept in sold_lines so a repeat of it can be recognised. Unless the kind is change (or missing),
        nothing more is done. A new positive budget replaces the old one; each drop marks the exact live
        requirement, else the first one on the same property, else the text itself as dropped; each add is taken
        off dropped and appended to told when new. For a cheapest pick, a drop, a sale-out or a budget change
        resets the wider search hold and the cheaper-row marks. When anything changed or the note has a summary,
        the version goes up, the recent-call memory is cleared and the summary joins self.notes. Returns the
        summary with the changes in parentheses.
        observe calls it for results of Peace's own calls, and run_calls for the replies to the model's calls.
        A new budget in a note passes sane_budget first, so a limit converted into another unit is read back in the
        store's own.
        When the reader found no budget and no price change but the note states one amount in the budget's unit
        (budget_in_text), that amount is the new budget and the note counts as a change."""
        self.heard.update(lines)
        prompt = NOTE + '\n\nITEMS AND REQUIREMENTS NOW:\n' + self.ledger() + '\n\nTHE NOTE:\n' + '\n'.join(lines)
        got = self.ask_json(prompt, [LIGHT, SPARE], 2500) or {}
        if got.get('budget') is None and (not got.get('price_moved')):
            stated = self.budget_in_text(' '.join(lines))
            if stated is not None:
                got['budget'] = stated
                if got.get('kind') not in (None, 'change'):
                    got['kind'] = 'change'
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
            budget = self.sane_budget(budget, ' '.join(lines))
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
        """Return (item number, requirement) for every requirement in force: musts and told, minus dropped."""
        out = []
        for number, item in enumerate(self.items, 1):
            for need in item.musts + item.told:
                if need not in item.dropped:
                    out.append((number, need))
        return out

    def evidence(self, observation: Any) -> str:
        """Point out, for each opened listing in a result, the attribute keys whose names match a requirement.
        Listing records with an attribute block are found to depth 6. For up to 12 of them and the first eight
        requirements in force, the keys of the stored record's block whose stems name the requirement's property
        (_names) are listed with their values (cut to 50 characters, at most three per requirement, tagged with
        the item number when there are several items). Returns one evidence line whose header warns that a matching
        name can still be another property and that another name can hold the same one, or ''.
        run_calls adds it to the reply of every call; the copy kept in the read cache leaves it out."""
        ids: list[str] = []

        def walk(node: Any, depth: int) -> None:
            """Collect ids of listing records with an attribute block, without entering a record once found."""
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
        """Describe the cart for prompts: the last cart view when it is fresh, else Peace's own record of lines.
        A view is fresh when it was seen at or after the last cart change; it is
        clipped and cut to 3000 characters."""
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            return _dump(_clip(self.cart_view, 300))[:3000]
        return "lines by Peace's record of the cart calls (listing: units): " + _dump(self.cart_lines)

    def held_units(self, ident: str) -> int:
        """Return how many units of a listing the cart holds.
        With a fresh cart view, a dict in it that carries the id as a string value and exactly one quantity-like key
        (qty, quantity, count or units) with a count gives the answer. Otherwise Peace's record is used, and a
        listing missing from it counts as 1."""
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
        """Return the listing ids in the cart: known ids found in a fresh cart view's JSON text, else Peace's record."""
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            text = _dump(self.cart_view)
            return [i for i in self.rows if i in text]
        return list(self.cart_lines)

    def record_of(self, ident: str) -> str:
        """Return a listing's record as clipped JSON (the opened record first, else its row), or a no-record note."""
        record = self.full.get(ident) or self.rows.get(ident)
        return _dump(_clip(record, 400))[:3000] if record else '(no record seen)'

    def others(self, skip: list[str], limit: int=30) -> str:
        """List other seen listings as clipped JSON lines, for reviews that compare the cart against them.
        Newest first; for a cheapest pick, opened listings come first and then by price (unknown prices last).
        Listings in skip and sold-out ones are left out, and at most limit lines are given."""
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
        """Read the fitting marks of every seen listing and collect the also-fits notes by part code.
        Each listing's text is its record's string values plus the key and value pairs of its nested dicts. marks
        maps each id to its _fit_marks; notes maps a part code to the codes any listing says that part also fits."""
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
        """Return the device codes a listing works with: its own works codes plus those noted for its own part code."""
        mark = index['marks'].get(ident) or {}
        found = set(mark.get('works') or [])
        if mark.get('own'):
            found |= index['notes'].get(mark['own'], set())
        return found

    def item_ids(self, number: int) -> list[str]:
        """Return the seen listing ids whose category label belongs to the item; none without a category key."""
        cat_key = self.category_key()
        item = self.items[number - 1]
        return [i for i in self.seen_order if cat_key and self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))]

    def fit_faults(self) -> list[str]:
        """Find cart parts that do not work with the cart's device, for each pair of items that must fit together.
        For each pair in self.bound the check needs exactly one cart line of the device item and
        at least one of the part item, and the device's record must show model codes. A part
        line whose fit set is not empty but names none of those codes is a fault; its text says
        what the part works with and names either up to four seen parts of that kind that fit
        the device, or, when there is none, the short fit plan for switching the device.
        run_calls holds a checkout over these faults at most twice (fit_holds), with more than
        the tail plus three steps left."""
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
        """Summarise which device codes the seen parts name, to guide the choice of a device and its parts.
        For each device item, the codes in the fit sets of the unsold listings of its paired part items are tallied
        per part item; the four codes named by the most part items (then by the most listings) are shown with their
        part counts and up to two seen devices that carry the code. short gives just the lines; otherwise a header
        explains how codes pair parts with devices. '' when there are no pairs or nothing was found.
        Codes with equal counts are ordered by the code itself, so the text does not depend on Python's hash seed (which
        changes the order a set of strings is walked in) and the same episode writes the same plan in every run."""
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
            ranked = sorted(tally.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1].values()), kv[0]))[:4]
            devices = {code: [i for i in self.item_ids(main) if code in (index['marks'].get(i, {}).get('codes') or [])][:2] for code, _ in ranked}
            text = '; '.join((f'{code}: ' + ', '.join((f'{n} {self.items[p - 1].name}' for p, n in sorted(per.items()))) + (f" (device seen: {', '.join(devices[code])})" if devices[code] else ' (device not seen yet)') for code, per in ranked))
            if text:
                lines.append(f"codes the {self.items[main - 1].name} candidates' parts name -> {text}")
        if not lines:
            return ''
        return ' | '.join(lines) if short else "[fit plan: a device fits a part when the part's text names the device's model code or another name it is sold as; codes are not searchable, so list the device's category and read the codes on its rows] " + ' | '.join(lines)

    def missing_items(self) -> list[int]:
        """Return the numbers of items the store has shown listings for but the cart holds no line of.
        It needs a category key and fewer held lines (units above zero) than items; otherwise []."""
        cat_key = self.category_key()
        ids = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        if not cat_key or len(ids) >= len(self.items):
            return []
        labels = [str(r.get(cat_key) or '') for r in self.rows.values()]
        held = [str((self.rows.get(i) or {}).get(cat_key) or '') for i in ids]
        return [n for n, item in enumerate(self.items, 1) if any((self.owns(item, label) for label in labels)) and (not any((self.owns(item, label) for label in held)))]

    def states_needs(self, ident: str, number: int) -> bool:
        """Say whether an opened listing's attributes plainly state every requirement in force for an item.
        For each requirement, its value words (split at commas, slashes, semicolons and bars) must all appear in the
        value of a key that names the same property (_spelled and _names). An empty value, or one with negation or
        limit words (not, no, least, most, under, over, unlike, different, min, max and the like), makes it False.
        fill_short_cart relies on it to add only lines it can vouch for without a model."""
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

    def fill_short_cart(self) -> bool:
        """Add a line for each item missing from a multi-item cart, using opened listings that plainly fit.
        For each missing item, opened listings not in the cart that are buyable, priced and pass
        states_needs are sorted by price and the cheapest is taken, but never for an item in a fit pair.
        With a numeric budget the pick is dropped when its cost (one unit for a per-item budget; else
        price times quantity plus the held total) would go over. The cart calls come from cart_calls;
        each is fitted, sent and recorded with note_cart, and the line is logged as filled_line. It
        returns False as soon as an item cannot be filled (no pick, no calls, too few steps, a fault or a
        failed call), and True when nothing was missing or every missing item was added. finish runs it
        for carts of two or more items."""
        cart = self.roles.get('cart') or ''
        bound = {n for pair in self.bound for n in pair}
        for number in self.missing_items():
            item = self.items[number - 1]
            fits = [i for i in self.item_ids(number) if i in self.full and i not in self.cart_lines and self.buyable(i) and self.price_of(i) is not None and self.states_needs(i, number)]
            fits.sort(key=lambda i: self.price_of(i) or 0.0)
            pick = fits[0] if fits and number not in bound else None
            if pick and isinstance(self.budget, (int, float)):
                held = sum(((self.price_of(i) or 0.0) * (self.held_units(i) or 0) for i in self.cart_ids()))
                cost = (self.price_of(pick) or 0.0) * (1 if self.brief.get('budget_scope') == 'each' else item.qty)
                if cost + (0.0 if self.brief.get('budget_scope') == 'each' else held) > self.budget + 1e-09:
                    pick = None
            calls = self.cart_calls({**{i: self.cart_lines[i] for i in self.cart_ids() if i in self.cart_lines}, pick: item.qty}, []) if pick else None
            if not calls or len(calls) + 1 > self.left():
                return False
            for args in calls:
                faults: list[str] = []
                args = _shape(args, self.schemas.get(cart), 'args', faults)
                if faults or self.act(cart, args) is None:
                    return False
                self.note_cart(cart, args)
            self.trace.append({'kind': 'filled_line', 'item': number, 'line': pick})
        return True

    def fuller_label(self, number: int, label: str) -> str:
        """Return a seen category label that names an item more fully than its cart line's label, or ''.
        Only for an item name with two or more roots that the line's label does not all contain, when a category
        key is known and the label is not the item's own shelf. The first seen label, in sorted order, that holds
        all the name's roots is returned."""
        cat_key = self.category_key()
        mine = _roots(self.items[number - 1].name)
        if len(mine) < 2 or mine <= _roots(label) or not cat_key or (self.items[number - 1].shelf and _roots(label) == _roots(self.items[number - 1].shelf)):
            return ''
        return next((seen for seen in sorted({str(r.get(cat_key) or '') for r in self.rows.values()}) if seen and mine <= _roots(seen)), '')

    def line_faults(self) -> list[str]:
        """Check the cart lines against the items in code and return the faults found.
        adopt_labels runs first, and the check needs a category key and lines with units above zero. For
        each item whose kind the store has shown: no line is a fault; more than one line belonging only
        to it must be cut to one; a line of only that item held at a count other than the item's
        quantity is a fault; a line whose label fuller_label can improve is a fault. When every item's
        kind has been seen, lines of no requested kind must go. Sold-out lines are faults. When every
        line's price is known and a budget is set, a line over a per-item budget, or a total over the
        total budget, is a fault, unless the excess passes ten times the budget (read as a unit
        mismatch). run_calls holds a checkout over these faults at most twice (line_holds), and only
        with more than the tail plus two steps left."""
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
                faults.append(f'item {n} ({item.name}) has no line in the cart')
            elif len([i for i in lines if owner[i] == [n]]) > 1:
                faults.append(f"item {n} ({item.name}) has {len(lines)} lines ({', '.join(lines)}); keep one")
            for i in lines:
                have = self.held_units(i)
                if owner[i] == [n] and have != item.qty:
                    faults.append(f'{i} is held at {have} unit(s); item {n} ({item.name}) wants {item.qty}')
                fuller = self.fuller_label(n, label[i])
                if fuller:
                    faults.append(f'{i} is from {label[i]}, but the store has the label {fuller}, which names item {n} ({item.name}) in full: take the line from there')
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
        """Return (listing, units held, units wanted) for cart lines whose count differs from their item's quantity.
        A line qualifies only when it belongs to exactly one item, is that item's only line in the cart, holds units
        and is not sold out. It needs a category key."""
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
        """Set the right unit counts on cart lines through the cart tool, and return the calls made.
        The id parameter is one seen carrying ids for this tool, else the required free-text parameter, else the
        only one; the count parameter is the first integer one, and the mode parameter the first enum with two or
        more options. A cart that sets counts (cart_sets) gets one call per line with the wanted count; a cart with
        a mode gets a removal of the held units (the option that reads like remove, delete, drop, clear or take,
        else the second option) followed by an add of the wanted units; any other cart gives up. Three steps beyond
        the calls are needed. Each call is fitted, sent, recorded with note_cart and observed, the last cart reply
        is kept, and the batch is logged as count_fix; the result lines are the arguments with the shown reply.
        run_calls applies it when the only cart faults left are wrong counts."""
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
        """Return the key for one review of a set of listings: the sorted ids joined by bars, plus the version."""
        return '|'.join(sorted(ids)) + f'#{self.version}'

    def check_cart(self, ids: list[str] | None=None, limit: float=SPEND_SOFT) -> str:
        """Have a model review the cart, or a planned set of listings, and return a hold message when it fails.
        With ids given (a planned set) the prompt asks to judge them as if carted at the requested quantities;
        otherwise the cart's ids and cart_text are used. Nothing happens when this key was reviewed already, no
        reviewer is affordable within limit, or there are no ids. The CHECK prompt carries the request, the ledger,
        the cart, each listing's record (marked when sold out) and, for cheapest, most and least picks, 15 other
        seen listings. For a real cart with an any pick, when two reviewers are allowed and no audit ran, LIGHT goes
        first and HEAVY only rechecks a failing verdict; otherwise the reviewers run in order. verdicts records a
        pass (a reply without ok set to false), and the review is logged. A failing verdict drops the ids from
        leads and returns up to five problems and up to four better listings (known and not sold out), asking the
        model to fix the cart or show from the records that the review is wrong.
        run_calls calls it on the cart before a checkout and on a single lead before a decline (at SPEND_HARD), and
        complete_cart on its planned lines (at SPEND_HARD)."""
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
        text = 'A careful review says this cart is not right yet: ' + '; '.join(problems or ['see the requirements'])
        if better:
            text += '. Listings that look better: ' + ', '.join(better)
        return text + '. Fix the cart (or show the review is wrong by re-reading the records), then check out.'

    def trail(self) -> tuple[str, bool]:
        """Resolve requirements whose value comes from another listing before a checkout; return (text, hold).
        Pending are the pointer requirements in force with no resolved value. While some remain, for at most six
        rounds (four when 15 or fewer steps are left) and with spend under SPEND_HARD, the TRAIL prompt gets the
        pending requirements and up to 60 records, newest first: opened ones (up to 45) before plain rows. LIGHT
        answers, with HEAVY as fallback when affordable. A chain whose links are all opened is resolved when it
        is done with a value, or when its last link needs no search (its value, or a pointer to the last page,
        is kept): ref_values gets it and the version goes up. Links that are known rows not yet opened are
        opened next with the offer or open tool (at most three per round, and about six pages in all); a link
        not seen at all is listed as still to find. When there is nothing to open, find_link searches for the
        first missing link (at most three link searches), and an opened page of that shop that already shows the
        values is reported instead. The rounds also stop without a single-id page tool, with too few steps or
        once finished. The text lists the pages opened and the values read; hold is True while links remain
        unfound (at most three times, two when 15 or fewer steps are left, and only with more than five steps),
        with instructions on finding them."""
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        tool = self.roles.get('offer') or self.roles.get('open')
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
            gaps: list[tuple[str, str]] = []
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
            pending = [p for p in pending if p not in self.ref_values]
            if not reach and gaps and tool and target and (self.left() > 12) and (self.link_searches < 3):
                reach = self.find_link(*gaps[0])
                if self.trace and self.trace[-1].get('kind') == 'link_known':
                    seen.append(f"{self.trace[-1]['id']}: an opened page of the shop this chain names that shows the values its description gives; read the value from it")
                    lost = lost[1:]
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
        text = ''
        if seen:
            text += '[Peace opened the pages a requirement points at]\n' + '\n'.join(seen) + '\n'
        if self.ref_values:
            text += '[values read from the pages requirements point at] ' + '; '.join((f'{k} -> {v}' for k, v in self.ref_values.items())) + '\n'
        hold = bool(lost) and self.trail_blocks < (3 if self.left() > 15 else 2) and (self.left() > 5)
        if hold:
            self.trail_blocks += 1
            text += 'The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: ' + ' | '.join(lost[:3]) + ". Search for it by its shop's name together with the values its description gives (the shop's other listings of that kind are not it), open the row of that shop whose page shows those values, apply the value its page gives, then check out."
        return (text, hold)

    def shop_rows(self, about: str, ids: list[str]) -> list[str]:
        """Return the unopened listings among ids whose shop or brand name appears in a link description.
        A row's shop names are its string values (3 to 40 characters) under keys that mention seller, shop, store,
        vendor, brand, maker or merchant; a name must appear in the description as a whole word run."""
        low = about.lower()
        out = []
        for ident in ids:
            row = self.rows.get(ident) or {}
            names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I)]
            if ident not in self.opened and any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)):
                out.append(ident)
        return out

    def opened_shop(self, about: str) -> list[str]:
        """Return the opened listings whose shop or brand name appears in a link description, tested as in shop_rows."""
        low = about.lower()
        out = []
        for ident in self.opened:
            row = self.rows.get(ident) or {}
            names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I)]
            if any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)):
                out.append(ident)
        return out

    def find_link(self, about: str, words: str) -> list[str]:
        """Look for the listing that a pointer chain's link describes, and return up to two rows to open.
        Value words are taken from the search words (or the description), leaving out the shop's own name words
        and filler words. When an opened listing of that shop already contains all those words, it is logged as
        link_known and [] comes back, so trail can point at it. Otherwise one search is made with the words through
        search_template (the same words again reuse the rows found before; both count as link searches), and the
        rows of the named shop in it are kept. Candidates must share a kind word with the description when there
        are such words; when the search gives none, rows of that shop seen earlier, newest first, are used."""
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
            """Keep the ids whose row text shares one of the description's kind words (all ids when there are none)."""
            return [i for i in ids if not kind or kind & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower()))]
        return (of_kind(known) or of_kind(self.shop_rows(about, list(reversed(self.seen_order)))))[:2]

    def offer_tool(self) -> str | None:
        """Return the tool that shows a listing's current offer: the offer role, else the open role."""
        return self.roles.get('offer') or self.roles.get('open')

    def reread_offers(self) -> str:
        """Re-read the offers of cart lines not looked at lately, and report what changed.
        It uses offer_tool and its id parameter. A line is stale when its offer was last seen
        before the later of two steps ago and the last refused checkout, and it is not gone. It
        needs three steps more than the stale lines. Each read is observed and stamped in
        offer_seen, and changed plain fields of the row are reported with the reply. The pass is
        logged as reread, and the version goes up when anything changed. Returns the change lines
        joined, or ''. run_calls calls it before every checkout, find_sold to locate a sale-out,
        and complete_cart before its second attempt."""
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
        """Say whether a listing is known to be sold out: marked unavailable or gone."""
        return ident in self.unavailable or ident in self.gone

    def swap_hint(self) -> str:
        """When cart lines have just sold out, mark them gone and return instructions for replacing them.
        Lines newly sold out are stamped in self.gone with the step and removed from leads, and the version goes up.
        replacements suggests read listings that may fit instead. The text says not to open the line again and,
        by case: with six or fewer steps left, swap to the first suggestion in this turn; for a cheapest pick, list
        the suggestions cheapest first and ask for one search below the best one's price; otherwise swap to the
        first one; with no suggestion, remove the line and find another. '' when no line just sold out.
        run_calls asks for it before a checkout and after every store call."""
        lost = [i for i in self.cart_ids() if i not in self.gone and self.sold_out(i)]
        if not lost:
            return ''
        for ident in lost:
            self.gone[ident] = self.step
        self.leads = [i for i in self.leads if i not in lost]
        self.version += 1
        picks = self.replacements(lost)
        text = f"[Peace] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        if picks and self.left() <= 6:
            return text + f" Few steps are left: in this turn add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks[1:]) or 'none'}."
        if picks and self.brief.get('pick') == 'cheapest':
            return text + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(picks)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {picks[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        if picks:
            return text + f" Listings already read that appear to fit that item instead: {', '.join(picks)}. In this turn take the sold-out line out of the cart and add {picks[0]} at the requested quantity, then check out."
        return text + ' Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits.'

    def find_sold(self) -> str:
        """Work out which cart line a sale-out note meant when it named none.
        Nothing is done when the cart is empty or a line is already known to be sold out. With one line,
        that line is marked unavailable; with several, their offers are re-read and the change lines
        returned. run_calls calls it before a checkout when a note reported a sale-out, and after a reply
        that reports one: a refused checkout (also when it repeats an earlier sale-out note), or any other
        call when the note named no known listing."""
        ids = [i for i in self.cart_ids() if i not in self.gone]
        if any((self.sold_out(i) for i in ids)):
            return ''
        if len(ids) == 1:
            self.unavailable.add(ids[0])
            return ''
        return self.reread_offers() if ids else ''

    def replacements(self, lost: list[str]) -> list[str]:
        """Ask a model which read listings could replace sold-out cart lines, and return up to four ids.
        It needs an affordable reviewer. Up to 25 other seen listings, newest first and not sold out, go into the
        SWAP prompt with the request, the ledger, the sold-out records and the cart; LIGHT answers first, then HEAVY.
        Known listings that are not lost, gone or in the cart are kept in the reply's order, or by the stated price
        for a cheapest pick (no price sorts last). Logged as swap.
        swap_hint asks for it when cart lines have just sold out."""
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
        got = self.ask_json(prompt, self.judges()[::-1], 3500)
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
        """Review the records once before a decline, and return a hold message naming listings that may fit.
        It runs once (audited) and only when a reviewer is affordable. audit_call asks for fits; when it names none,
        the unopened rows it suggests (known, not sold out) are opened with open_rows, and when that opened anything
        the audit is asked again. With fits, lead_items and leads are set, and the text names them per item with
        the reviewer's reason, asking the model to open them and buy what truly fits; '' otherwise.
        The leads drive the next declines in run_calls: the next one is held once more over them, the one after
        that has the first lead reviewed alone by check_cart, and a pass sets confirmed, so that complete_cart can
        buy it when the model declines yet again.
        When the review leaves an item without a fit, recover searches again and opens new rows for it, and the review is
        asked once more; its new fits are kept when they cover at least as many items."""
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
        missing = self.unfitted(named)
        if missing and self.recover(missing):
            # the re-review sees the earlier leads and the rows recover opened first, and its fits are merged per item
            again_got = self.audit_call([i for ids in named.values() for i in ids] + self.recovered)
            again = self.audit_fits(again_got)
            if again:
                named = self.merge_fits(named, again)
                got = again_got
        if not named:
            return ''
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return 'A careful review thinks these listings may fit: ' + '; '.join((f"item {k}: {', '.join(v)}" for k, v in named.items())) + (f" ({got.get('reason')})" if got.get('reason') else '') + '. Open them and buy what truly fits; end without a purchase only if each one fails.'

    def fit_number(self, key: str) -> int | None:
        """The item number a review's fits key names ('1', 'item 2'), or 1 for a key without digits on a one-item request;
        None when it names no item of this request."""
        found = re.search('\\d+', str(key))
        number = int(found.group()) if found else (1 if len(self.items) == 1 else None)
        return number if number and 1 <= number <= len(self.items) else None

    def unfitted(self, named: dict[str, list[str]]) -> list[int]:
        """The item numbers a review's fits leave without a listing."""
        covered = {self.fit_number(k) for k in named}
        return [n for n in range(1, len(self.items) + 1) if n not in covered]

    def merge_fits(self, old: dict[str, list[str]], new: dict[str, list[str]]) -> dict[str, list[str]]:
        """Merge two reviews' fits per item number, earlier ids first, up to four per item."""
        merged: dict[str, list[str]] = {}
        for fits in (old, new):
            for key, ids in fits.items():
                number = self.fit_number(key)
                slot = str(number) if number else str(key)
                merged[slot] = list(dict.fromkeys(merged.get(slot, []) + list(ids)))[:4]
        return merged

    def tried_queries(self, tool: str, field: str) -> set[str]:
        """Every text the search tool has been called with in this episode, from the search history and from called_at,
        so a query that came back empty is not sent again."""
        tried = {_words(str(a.get(field) or '')) for a in self.searched}
        for (name, dumped) in self.called_at:
            if name != tool:
                continue
            try:
                args = json.loads(dumped)
            except ValueError:
                continue
            if isinstance(args, dict) and isinstance(args.get(field), str):
                tried.add(_words(args[field]))
        return tried

    def item_label(self, number: int) -> str:
        """The category label of a requested item: the one most of its rows show, else the brief's shelf."""
        cat_key = self.category_key()
        item = self.items[number - 1]
        labels = [str(r.get(cat_key) or '') for r in self.rows.values()] if cat_key else []
        labels = [label for label in labels if label and self.owns(item, label)]
        return max(set(labels), key=labels.count) if labels else item.shelf

    def item_queries(self, number: int) -> list[str]:
        """Up to two search texts for an item: its requirement values with its name, then the leading value with its shelf
        (yes/no values left out, at most eight words each)."""
        item = self.items[number - 1]
        values = [v for v, _ in self.value_words(number) if re.search('[a-z]', v, re.I) and _words(v) not in ('yes', 'no', 'true', 'false')]
        options = [' '.join(values + [item.name]), ' '.join(values[:1] + [item.shelf or item.name])] if values else [item.shelf or item.name]
        return [' '.join(q.split()[:8]) for q in options if q.strip()]

    def item_search_args(self, number: int, base: dict[str, Any], field: str, query: str) -> dict[str, Any]:
        # a category argument the last search carried (a required group or shelf filter) is set to this item's own
        """Arguments for a search on one item's behalf: the template's fixed arguments, with a category argument (a required
        group or shelf filter) set to this item's own label, and the query text."""
        args = dict(base)
        label = self.item_label(number)
        for key, value in base.items():
            if key != field and isinstance(value, str) and label and re.search(SHELF_ARG, _snake(key), re.I):
                args[key] = label
        args[field] = query
        return args

    def recover(self, missing: list[int]) -> bool:
        """Search again for items the decline review left without a fit, and say whether a new full record came of it.
        For each such item it runs up to two fresh searches with the store's search template (the item's requirement values with
        its name, then the leading value with its shelf; a query already tried is skipped), then opens up to two unread rows of
        the item's kind per item, most requirement words first and cheapest next, within budget. It spends at most RECOVER_STEPS
        steps, keeps room for the cart, one line per item and the checkout, and does nothing past WIND_DOWN seconds."""
        plan = self.search_template() or self.value_template()
        tool = self.facts_tool()
        keep = self.tail() + len(self.items) + 2
        room = min(RECOVER_STEPS, self.left() - keep)
        if not plan or not tool or not self.single_target(tool) or room < 3 or self.clock() > WIND_DOWN or self.finished:
            return False
        search, base, field = plan
        before = set(self.full)
        tried = self.tried_queries(search, field)
        budget = self.budget if isinstance(self.budget, (int, float)) else None
        searches = max(1, room // 2)
        found: list[tuple[int, str]] = []
        used = 0
        plans = {number: self.item_queries(number) for number in missing}
        order = [(number, query) for k in range(2) for number in missing for query in plans[number][k:k + 1]]
        for number, query in order:
            if used >= searches or self.finished:
                break
            if True:
                if _words(query) in tried:
                    continue
                tried.add(_words(query))
                faults: list[str] = []
                args = _shape(self.item_search_args(number, base, field, query), self.schemas.get(search), 'args', faults)
                if faults:
                    continue
                reply = self.act(search, args)
                used += 1
                if reply is None:
                    break
                self.observe(search, reply.get('observation'))
                found += [(number, i) for i, _ in self.records_in(reply.get('observation'))]
        picks: list[str] = []
        cat_key = self.category_key()   # after the searches: more rows make the guess firmer
        for number in missing:
            item = self.items[number - 1]
            words = {w for v, _ in self.value_words(number) for w in re.findall('[a-z0-9]{3,}', v.lower())}
            rows = [i for n, i in dict.fromkeys(found) if n == number and i not in self.full and self.buyable(i) and (not cat_key or self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))) and (budget is None or (self.price_of(i) or 0) <= budget)]
            rows.sort(key=lambda i: (-len(words & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower()))), self.price_of(i) or 0))
            picks += [i for i in rows[:2] if i not in picks]
        if picks and room - used > 0:
            self.fill_facts(picks[:room - used], self.left() - (room - used))
        self.recovered = [i for i in picks if i in self.full and i not in before]
        self.trace.append({'kind': 'recover', 'items': missing, 'searches': used, 'opened': len(self.recovered)})
        return bool(self.recovered)

    def covers(self, lines: list[str]) -> bool:
        """Say whether a set of listings has a line for every requested item, by the store's category labels (owns).
        With one item, or before the category key is known, any set counts. The early decline hold in run_calls uses it, so
        two lines for one item do not pass for a cart that covered every item."""
        cat_key = self.category_key()
        if len(self.items) < 2 or not cat_key:
            return True
        labels = [str((self.full.get(i) or self.rows.get(i) or {}).get(cat_key) or '') for i in lines]
        return all((any((self.owns(item, label) for label in labels)) for item in self.items))

    def audit_call(self, keep: list[str] | None=None) -> dict[str, Any] | None:
        """Send the AUDIT prompt over up to 30 seen listings and return the parsed reply.
        Listings come newest first, sold-out ones skipped: opened ones as OPENED with their record, the rest as ROW.
        LIGHT answers first when a reviewer is affordable, and HEAVY asks again when it is allowed and LIGHT named
        no fit. None when there are no records; the reply is logged as audit."""
        records = []
        for ident in dict.fromkeys([i for i in keep or [] if i in self.rows] + list(reversed(self.seen_order))):
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
        judges = self.judges()
        got = self.ask_json(prompt, [LIGHT] if judges else [], 4000)
        if HEAVY in judges and not self.no_credit and not self.audit_fits(got):
            got = self.ask_json(prompt, [HEAVY], 4000) or got
        self.trace.append({'kind': 'audit', 'got': got})
        return got

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        """Read an audit reply's fits as item key to up to four known, unsold listing ids, dropping empty entries."""
        fits = (got or {}).get('fits')
        if not isinstance(fits, dict):
            return {}
        named = {str(k): [str(i) for i in _items(v) if str(i) in self.rows and (not self.sold_out(str(i)))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def open_rows(self, rows: list[str]) -> bool:
        """Open rows the audit asked to see, and say whether any new opened record came of it.
        A side-by-side call with up to compare_room ids is used when the compare tool takes an id array and is not
        ruled out by compare_bare; otherwise the first two rows are opened one by one with the facts tool. It needs
        five steps more than the calls, stops on a fault, a failed call or the end, and logs audit_open."""
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
        """Build the cart tool calls that leave exactly the given lines in the cart, or None when that cannot be done.
        lines maps listing ids to wanted units and drop lists ids to take out. A cart tool that takes an array of
        objects gets one call carrying every line (the first string field for the id, the first integer field for
        the count), and drop plays no part. Otherwise the id parameter is chosen as in fix_counts. Each dropped
        line is removed with the mode's second option and the held count, or with a count of 0 on a cart that sets
        counts, or with a negative count when the parameter's minimum allows it; with none of these the result is
        None. Each wanted line not already held at its count is added with the mode's first option and the count;
        a line held at another count can be changed only on a cart that sets counts, else the result is None.
        fill_short_cart and complete_cart send the calls it builds."""
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
        """Set the cart and check out itself once a review confirmed a listing the model wanted to decline over.
        It runs once and needs a confirmed lead that is not sold out, plus cart and buy tools. The plan takes, for
        each item the audit named, its first unsold lead at the item's quantity (one per category label), and must
        include the confirmed listing; cart lines of other labels are kept and the rest dropped, and together they
        must cover every item. cart_calls builds the calls, which must fit in the steps left. The planned lines are
        opened where unread, their part links followed, and check_cart reviews them at SPEND_HARD; only a pass goes
        on. The cart calls run, then up to two checkout attempts follow: each passes the order gate first, and the
        second re-reads offers and rechecks the cart, stopping when the check fails, a line sold out, steps ran out
        or the episode ended. Each attempt counts a buy and stamps refused_at when the store does not finish. The
        result tells the model what Peace did and what the store said, or is '' when it did nothing.
        run_calls tries it when the model sends a decline, before the decline text is quieted."""
        cart, buy = (self.roles.get('cart'), self.roles.get('buy'))
        if self.completed or not self.confirmed or self.sold_out(self.confirmed) or (not cart) or (not buy):
            return ''
        self.completed = True
        cat_key = self.category_key()

        def kind(ident: str) -> str:
            """Return a listing's category label in lower case, or the id itself when no category key is known."""
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
            self.open_offers()
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
        return 'Not run (Peace acted instead): the independent check confirmed ' + self.confirmed + ', so Peace set the cart (' + '; '.join(done) + ') and checked out. Store reply: ' + ' | then: '.join(replies)

    def is_buy(self, name: str, args: dict[str, Any]) -> bool:
        """Say whether a call is the checkout: the buy tool with every fixed buy argument at its brief value."""
        if name != self.roles.get('buy'):
            return False
        fixed = self.brief.get('buy_args') if isinstance(self.brief.get('buy_args'), dict) else {}
        return all((args.get(k) == v for k, v in fixed.items()))

    def is_refuse(self, name: str, args: dict[str, Any]) -> bool:
        """Say whether a call is the decline: the refuse tool with every fixed refuse argument at its brief value."""
        if name != self.roles.get('refuse'):
            return False
        fixed = self.brief.get('refuse_args') if isinstance(self.brief.get('refuse_args'), dict) else {}
        return all((args.get(k) == v for k, v in fixed.items()))

    def offered(self) -> list[dict[str, Any]]:
        """Return the tools offered to the loop model this turn.
        The ask tool is never offered, since the interviewer sends every message. Near the end the list narrows:
        only the ending tools within tail steps, plus the cart and cart view tools within the tail plus three, and
        also the offer tool when no buy happened yet and the steps left are within the item count plus five.
        Without ending tools, or when narrowing would leave nothing, the full list stays.
        run_calls also uses the list to refuse calls to tools that are not offered at that moment."""
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
        """Rewrite the free-text arguments of a decline so that they name no listing.
        Each free-text string parameter that is set (or required) gets a fixed short apology, cut to its maximum
        length (200 when none is given), when its text names a known listing id, when it is empty, or whenever the
        brief does not say that the request wants alternatives (quiet_decline false).
        run_calls applies it to the model's decline, and finish and can_refuse to the brief's fixed arguments."""
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
        """Turn listing names passed where an id belongs into ids, or explain why the call cannot run.
        Only for the cart, open, offer and compare tools once the id key is known, when the tool has exactly one
        free-text or array parameter and that parameter was seen carrying ids (always for the cart tool). A known
        id passes stripped; non-strings, id-like strings and strings under eight characters pass as they are.
        Longer text is matched against the start of the rows' other string values: one match gives its id, several
        or none give a problem text. Returns (args, problem), keeping the original args when there is a problem.
        run_calls calls it after _shape; the parameters it trusts to carry ids are learned in run_calls (id_args)."""
        if name not in {self.roles.get(r) for r in ('cart', 'open', 'offer', 'compare')} - {None} or not self.id_key:
            return (args, '')
        props = (self.schemas.get(name) or {}).get('properties') or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and ('string' in _kinds(v) or 'array' in _kinds(v)) and ('enum' not in v)]
        if len(free) != 1 or (name != self.roles.get('cart') and (name, free[0]) not in self.id_args):
            return (args, '')

        def fix(value: Any) -> tuple[Any, str]:
            """Map one argument value to a listing id, returning (value, problem text or '')."""
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
        """Say whether the cart tool sets a line's units outright: no enum of two or more options, an integer with minimum 0."""
        props = self.schemas.get(name, {}).get('properties') or {}
        if any((isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) >= 2) for s in props.values())):
            return False
        return any((isinstance(s, dict) and 'integer' in _kinds(s) and (s.get('minimum') == 0) for s in props.values()))

    def whole_removal(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Fill in the count of a cart removal that leaves it out, so the whole held line is taken off.
        When an enum argument holds a non-first option (read as a removal), a string argument names a line in
        Peace's cart record and the integer count parameter is missing, the count becomes the units held, capped
        at the parameter's maximum. Any other call is returned unchanged."""
        spec = self.schemas.get(name, {}).get('properties') or {}
        count = next((k for k, v in spec.items() if isinstance(v, dict) and 'integer' in _kinds(v)), None)
        removal = any((isinstance((spec.get(k) or {}).get('enum'), list) and v in spec[k]['enum'][1:] for k, v in args.items()))
        ident = next((v for v in args.values() if isinstance(v, str) and v in self.cart_lines), None)
        return {**args, count: min(self.cart_lines[ident], spec[count].get('maximum', self.cart_lines[ident]))} if count and removal and ident and count not in args else args

    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        """Update Peace's own record of the cart after a cart call, from the arguments sent.
        It sets cart_touched and stamps cart_at. A call with an enum argument at a non-first option counts as a
        removal. Lines are read from the arguments: list entries that are objects (id: the first string value that
        is a known listing; count: the first non-string count, else 1) or known ids, and plain string arguments that
        are known ids, these two with the call's loose count. A list of objects, or an empty list of an object
        schema, replaces the whole record. Otherwise, per line, a removal lowers the units by the count (by one
        when no count is given and the tool has an integer parameter, to nothing when it has none), a cart that
        sets counts takes the count (1 when missing), and any other cart adds the count (1 when missing); lines at
        zero or below leave the record.
        run_calls, fill_short_cart, fix_counts and complete_cart call it after each cart call they send."""
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
        """Render a result for the model: compact JSON with strings clipped to SHOW_CHARS, cut at OBS_LIMIT with a mark."""
        text = _dump(_clip(observation, SHOW_CHARS))
        if len(text) > OBS_LIMIT:
            text = text[:OBS_LIMIT] + '...(cut)'
        return text

    def compare_target(self) -> str | None:
        """Return the compare tool's array parameter when no other parameter is required, else None."""
        schema = self.schemas.get(self.roles.get('compare') or '') or {}
        props = schema.get('properties') or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in _kinds(v)), None)
        if not target or any((k != target for k in schema.get('required') or [])):
            return None
        return target

    def compare_room(self) -> int:
        """Return how many listings one side-by-side call may carry.
        The schema's maxItems when given. After a call showed nothing for some number of ids, one less than the
        smallest such number, kept between 2 and the larger of 2 and the biggest call that worked. Otherwise the
        biggest call that worked, at least 4.
        run_calls learns compare_ok and compare_bad from whether a side-by-side reply mentions the ids it was given."""
        spec = ((self.schemas.get(self.roles.get('compare') or '') or {}).get('properties') or {}).get(self.compare_target() or '') or {}
        if isinstance(spec.get('maxItems'), int):
            return spec['maxItems']
        if self.compare_bad:
            return max(2, min(self.compare_bad - 1, max(self.compare_ok, 2)))
        return max(4, self.compare_ok)

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge several single-listing open calls of one turn into one side-by-side compare call.
        It needs distinct open and compare tools, a single id parameter on open, an id array on compare that is its
        only required parameter, and a compare tool not ruled out by compare_bare. Open calls whose only argument is
        a known, not gone id are picked, up to compare_room. With two or more, the first picked call becomes the
        compare call with all their ids (keeping its call id), and the other picked calls become placeholders
        marked joined, which run_calls answers without a step. Logged as joined."""
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
        """Run one model turn's tool calls through Peace's checks and return (call id, reply text) for each.
        join_opens first merges single opens. A call is answered without spending a step when the episode is over,
        it is a message (the interviewer sends those), the tool is unknown or not offered now, its arguments are
        not one JSON object, do not fit the schema (_shape) or name a listing that resolve_ids cannot turn into an
        id, it opens a gone listing, it takes units off a line not in the cart, it is an ending call sent along
        with others, it would spend a tail step on anything but an ending, it repeats one of the last three calls,
        or it is a read whose cached result is still current or whose listing has shown everything. Compare calls
        drop ids already shown in full, and are cut to compare_room after a side-by-side call that failed.
        A checkout with more than two steps left meets the holds in turn, each answered as not run: the one-time
        wider search (wider_hold), a cart view that shows a change, sold-out lines (swap_hint), unresolved pointer
        chains (trail), part and device mismatches (fit_faults), cart line faults (line_faults, where pure count
        faults are fixed in place by fix_counts), the cart review after fill_facts, show_links and cheaper_rows
        (check_cart), and offer changes found once spend is past SPEND_SOFT. A decline with more than five steps
        meets the audit and up to two holds over its leads, the second confirming the first lead with check_cart.
        A late decline after a cart that covered every item is held once, an early decline with no leads is
        pushed once toward more searching, complete_cart may buy a confirmed lead, and quiet cleans the decline
        text. A checkout may wait for late_question, and every ending with two steps left passes gate.
        After the store call it learns from the reply: compare size limits, recent calls, which arguments carry
        ids, the records (absorb), which tools show attributes (moving the open and offer roles when needed),
        views (note_view), extra pages (more_pages), Peace's cart record (note_cart, dropped and built lines),
        recent searches, the cart view and buys. The reply text gains pages, trims, cheapest rows, ignored caps,
        evidence, a zero-count warning, read notes, a sale-out re-read, swap hints and queued notes, and reads
        are cached for the current version.
        Replies that begin with Not run tell the model that its own call spent no step; calls Peace makes itself
        during a hold (reads, cart views, questions, cart fixes) still use store steps.
        A decline with three to five steps left, after a review named a lead for every item, has the first lead
        checked independently at once (the lead follow-ups above need more room); when the check passes, complete_cart
        buys it on the same decline, after checking the whole planned cart and passing the order gate. The review's lead lists
        are first cut to the leads still standing, so the lead checked is the one complete_cart plans for its item.
        A decline with more than five steps left after a cart that held a line for every item (covers, when the
        store's category labels tell items apart) is held once on its own flag, so the late hold stays for the last steps;
        it is not held while a confirmed lead waits for complete_cart. The hold names that cart's lines and asks to re-read
        their records before ending."""
        out: list[tuple[str, str]] = []
        calls = self.join_opens(calls)
        for index, call in enumerate(calls):
            allowed = {t['function']['name'] for t in self.offered()}
            cid = call['id']
            if call.get('joined'):
                out.append((cid, 'Shown in the side-by-side result above: Peace opened these listings together in one step.'))
                continue
            fn = call.get('function') or {}
            name = fn.get('name')
            raw = fn.get('arguments')
            if self.finished:
                out.append((cid, 'Not run: the episode is over.'))
                continue
            if name == self.roles.get('ask'):
                out.append((cid, "Not run (no step used): Peace's interviewer handles every message to the shopper."))
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
            if name == self.roles.get('cart'):
                args = self.whole_removal(name, args)
            trimmed = ''
            many = self.compare_target() if name == self.roles.get('compare') else None
            if many and isinstance(args.get(many), list) and (len(args[many]) > self.compare_room()) and self.compare_bad:
                room = self.compare_room()
                trimmed = f"\n[Peace] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
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
                    trimmed += f"\n[Peace] Left out {', '.join((str(i) for i in args[many] if i not in keep))}: shown in full already, nothing changed since."
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
                            out.append((cid, "Not run yet (no step used). Peace set the cart's quantities itself: " + '; '.join(fixed) + '. If the store refused any of these, fix those lines; then check out.\nCart now: ' + self.cart_text()))
                            continue
                    if faults:
                        self.line_holds += 1
                        self.trace.append({'kind': 'line_hold', 'faults': faults[:6]})
                        out.append((cid, 'Not run yet (no step used). The cart does not match the request yet: ' + '; '.join(faults[:6]) + ". Fix those lines (the cart tool's own way to add or take off units), then check out.\nCart now: " + self.cart_text()))
                        continue
                    lines = self.cart_ids()
                    read = self.fill_facts(lines, self.tail() + (1 if len(lines) >= 2 else 3))
                    self.show_links(lines, self.tail() + 1)
                    read += self.cheaper_rows()
                    verdict = self.check_cart()
                    if verdict:
                        seen = "Peace read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n' if read else ''
                        out.append((cid, 'Not run yet. ' + chain + seen + verdict + '\nCart now: ' + self.cart_text()))
                        continue
                    if reread and self.cost > SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). Peace re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'))
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
            if self.is_refuse(name, args) and self.left() > 5 and (not self.early_held) and (not (self.confirmed and (not self.completed))):
                lines = [i for i in self.built if not self.sold_out(i)]
                fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
                missing = [i for i in lines if i not in self.cart_ids() or (i in self.dropped and (not fresh))]
                if len(lines) >= len(self.items) and self.covers(lines) and len(missing) + 1 <= self.left():
                    self.early_held = True
                    self.trace.append({'kind': 'early_decline_held', 'lines': lines, 'missing': missing})
                    redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else 'Check out the cart as it is.'
                    out.append((cid, 'Not run yet (no step used). Earlier in this episode the cart held a line for every requested item: ' + ', '.join(lines) + '. Ending without a purchase earns nothing when the store has a fitting listing, and almost no request has none. Re-read the records of those lines against the requirements in force: a line fails when its record states a different value, or when it leaves out something the shopper asked for. ' + redo + ' End without a purchase only if a record breaks a requirement and your search found no other listing for that item; then send the ending call again.'))
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
            if self.is_refuse(name, args) and (not self.leads) and (not self.pushed) and (self.left() > self.max_steps // 3):
                self.pushed = True
                out.append((cid, f"Not run yet (no step used). {self.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if self.is_refuse(name, args) and self.leads and (not self.confirmed) and (self.refuse_blocks < 2) and (3 <= self.left() <= 5):
                # the review's lists keep only the leads still standing (a failed check drops one from leads), so the
                # lead checked here is the one complete_cart plans for its item
                kept = {k: [i for i in v if i in self.leads] for k, v in self.lead_items.items()}
                kept = {k: v for k, v in kept.items() if v}
                first = next((i for i in next(iter(kept.values()), []) if not self.sold_out(i)), None)
                if first and len(kept) >= len(self.items):
                    self.lead_items = kept
                    self.refuse_blocks = 2
                    self.check_cart([first], SPEND_HARD)
                    if self.verdicts.get(self.check_key([first])) and first in self.leads:
                        self.confirmed = first
                        self.trace.append({'kind': 'late_lead_confirmed', 'confirmed': first})
            if self.is_refuse(name, args):
                done = self.complete_cart()
                if done:
                    out.append((cid, done + (' [episode finished]' if self.finished else '')))
                    continue
                args = self.quiet(name, args)
            if self.is_buy(name, args):
                late = self.late_question()
                if late:
                    out.append((cid, 'Not run yet. ' + late + '\nCart now: ' + self.cart_text()))
                    continue
            if self.is_buy(name, args):
                gone = self.open_offers()
                if gone and self.left() > self.tail() + 2:
                    out.append((cid, 'Not run yet. The offer page of ' + ', '.join(gone) + ' shows it can no longer be bought: replace that line, then check out.\nCart now: ' + self.cart_text()))
                    continue
            if ending and self.left() >= 2:
                go, said = self.gate('order' if self.is_buy(name, args) else 'end')
                if not go:
                    out.append((cid, 'Not run yet. ' + said + '\nCart now: ' + self.cart_text()))
                    continue
                if said:
                    trimmed = '\n[Peace] ' + said + trimmed
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, 'The call failed (network or server). You may try again.'))
                continue
            observation = reply.get('observation')
            if many and isinstance(args.get(many), list) and (len(args[many]) > 1):
                shown = _dump(observation)
                if not any((str(i) in shown for i in args[many])):
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    trimmed += f'\n[Peace] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_room()} per call.'
                else:
                    self.compare_ok = max(self.compare_ok, len(args[many]))
            self.recent = (self.recent + [mark])[-3:]
            self.id_args.update(((name, k) for k, v in args.items() if (v if isinstance(v, list) else [v]) and all(((isinstance(x, str) or (isinstance(x, int) and (not isinstance(x, bool)))) and str(x) in self.rows for x in (v if isinstance(v, list) else [v])))))
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
                if len(lines) >= len(self.built) and (not _faulted(observation)):
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
                    text += '\n[Peace] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'
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
                    text += "\n[Peace re-read the cart's lines after the refusal]\n" + seen
            self.note_sold = False
            swap = self.swap_hint()
            if swap:
                text += '\n' + swap
            if self.queued:
                text += "\n[notes seen on Peace's own calls] " + ' | '.join(self.queued)
                self.queued = []
            if name in readers:
                self.read_cache[mark] = (self.version, self.step, base)
            if reply.get('finished'):
                text += ' [episode finished]'
            out.append((cid, text))
        return out

    def can_refuse(self) -> bool:
        """Say whether a decline can be sent: a refuse tool exists and its fixed arguments, quieted, fit its schema."""
        refuse = self.roles.get('refuse')
        if not refuse:
            return False
        faults: list[str] = []
        _shape(self.quiet(refuse, dict(self.brief.get('refuse_args') or {})), self.schemas.get(refuse), 'args', faults)
        return not faults

    def finish(self, again: bool=False) -> str:
        """Close an episode the loop left open: check out the cart held, or end without a purchase.
        Nothing is done once finished, out of funds or out of steps. The cart counts as held by a fresh view or by
        Peace's record. For two or more items, fill_short_cart tries to add missing lines, and a cart it cannot
        complete is not bought when a decline is possible (logged as short_cart_refused). A complete cart of two or
        more lines has its unread lines opened and, with three steps and five seconds of model room, its part links
        followed. A checkout is sent, after the order gate when two steps remain, when the cart is held, or when it
        was changed and not viewed since and two steps remain. With again set and a reply that left the episode
        open, the refusal is counted and observed and its text returned, so run can give the loop another turn.
        Otherwise the decline is sent next with quieted arguments after the end gate, and a decline the store did
        not accept is reported the same way. Returns '' when nothing needs the loop again.
        run calls it after the loop (first with again set), and agent_main after a crash."""
        if self.finished or self.broke or self.left() <= 0:
            return ''
        buy, refuse = (self.roles.get('buy'), self.roles.get('refuse'))
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        whole = True
        if holding and len(self.items) >= 2:
            whole = self.fill_short_cart() or not self.can_refuse()
            if whole and len(self.cart_ids()) >= 2:
                self.fill_facts(self.cart_ids(), self.tail())
                if self.left() >= 3 and self.model_room() > 5:
                    self.show_links(self.cart_ids(), self.tail())
            if not whole:
                self.trace.append({'kind': 'short_cart_refused', 'lines': self.cart_ids()})
        if self.finished or self.left() <= 0:
            return ''
        if whole and buy and (holding or (self.cart_touched and (not fresh) and (self.left() >= 2))):
            faults: list[str] = []
            args = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), 'args', faults)
            if not faults:
                self.open_offers()
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
                    return 'ending without a purchase was refused: ' + self.show(reply.get('observation'))
        return ''

    def trim(self) -> None:
        """Shorten old tool results once the chat grows past HISTORY_LIMIT characters.
        Within the older half of the tool messages (at least one), those over 600 characters are cut to 400 and
        followed by a list of up to 20 known listing ids quoted in them, so the model still knows which
        listings those results showed."""
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
        """Run the whole episode and return the trace.
        An episode already finished returns at once. read_request writes the brief (logged), the opening view is
        absorbed, and the chat starts with GUIDE as the system prompt and a user message holding the request, the
        opening view, an instruction to search each item and open several listings so the interviewer has keys to
        work from, and the status. work runs the loop within LOOP_CALLS and SPEND_HARD. If the episode is still
        open, finish runs with again set; when the store refused that ending and steps, funds and credit remain,
        the refusal goes to the model and work gets ten more loop calls and a slightly higher spend cap. A last
        finish closes whatever is still open, and an end entry with steps, cost and calls is logged."""
        if self.finished:
            return self.trace
        self.read_request()
        self.trace.append({'kind': 'brief', 'brief': self.brief})
        opening = _structured(self.view.get('opening'))
        if opening is not None:
            self.absorb(opening)
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            intro += '\n\nWHAT THE STORE SHOWS FIRST:\n' + self.show(opening)
        intro += '\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n' + self.status()
        self.chat = [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': intro}]
        self.work(LOOP_CALLS, SPEND_HARD)
        if not self.finished:
            turned = self.finish(again=True)
            if turned and (not self.finished) and (not self.broke) and (not self.no_credit) and (self.left() > 1):
                self.chat.append({'role': 'user', 'content': '[Peace] Peace tried to end the episode and ' + turned + '\nApply what the store says, then finish.\n' + self.status()})
                self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
            if not self.finished:
                self.finish()
        self.trace.append({'kind': 'end', 'finished': self.finished, 'steps': self.step, 'cost': round(self.cost, 5), 'calls': self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        """The tool loop: one model turn after another until the episode ends or a limit is reached.
        It runs while the episode is open, funds and steps remain, loop calls stay under loop_cap and spend under
        spend_cap, and the clock is before LAST_TURN (or the cart is empty and EMPTY_ROOM seconds of model room
        remain). When ready_to_ask says so, the interview runs and its summary joins the chat. Otherwise the
        history is trimmed and LIGHT, with SPARE as fallback, is asked with the offered tools; three failed turns
        in all, no credit or ALL_CALLS end the loop. Tool calls are normalised (missing ids filled in, arguments
        re-encoded as JSON objects). A reply without calls gets a reminder, and three such replies in a row end
        the loop. When the model tries to cart, buy or decline before the interview while messages, more than
        eight steps and time before ASK_BY remain, its calls are answered as not run and the interview runs first.
        Otherwise run_calls handles the calls, and the status is added to the last tool reply.
        Each model turn is stored as an assistant message with its tool calls, followed by one tool message per
        call id, and is logged as a turn with its text and calls."""
        idle = 0
        while not self.finished and (not self.broke) and (self.left() > 0) and (self.loop_calls < loop_cap) and (self.cost < spend_cap) and (self.clock() < LAST_TURN or (not self.cart_ids() and self.model_room() > EMPTY_ROOM)):
            if self.ready_to_ask():
                summary = self.interview()
                if summary:
                    self.chat.append({'role': 'user', 'content': summary + '\n' + self.status()})
                continue
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
            if not self.asked and self.roles.get('ask') and (self.msgs_free() > 0) and any((c['function']['name'] in committing for c in calls)) and (self.left() > 8) and (self.clock() <= ASK_BY):
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
    """Entry point: build a Peace for the problem, run the episode and return its trace.
    A failure in the constructor gives a trace with a single crash entry. A failure during the run is
    logged and, while the episode is still open, finish is tried once more (its own failure logged too);
    the trace is returned either way."""
    try:
        episode = Peace(problem_data)
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
