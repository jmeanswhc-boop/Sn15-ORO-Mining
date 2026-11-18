"""Zenith: buys for one shopper in an ORO store world. A fork of the public Halo v2, which extends the public
Know Thyself agent (top of race 13).

Inherited: a light model works the store through the world's own tools; code owns the ledger and the question
plan, and vets the cart ahead of an order and the evidence ahead of a decline. A clock holds each episode under
the validator's 300 s cap. late_question spends a message kept for the ending on one more planned question when
a purchase is near. A message can bundle at most LIST_SIZE of an item's planned properties; settle rotates them
so ones the shopper is indifferent to never block the rest. Zenith's own changes:
- multi-property questions only when the request allows exactly 3 messages: worlds allowing 2, 4 or 5 messages
  refuse them with a "one at a time" complaint that costs a message;
- a two-part shopper reply (answer plus unprompted remarks such as a new budget or a dropped requirement)
  reaches the reader with its parts apart;
- a heard budget change is held in the request's unit (a x100 slip is rescaled);
- in every world the next question goes to the least-asked item (items with nothing found yet first).
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
HISTORY_LIMIT = 150 * 1000
SHORTENED = '...(older result shortened; listings in it: '
HISTORY_CEILING = 2 * HISTORY_LIMIT
GATE_ROUNDS = 3
AUTO_PAGES = 6
WHOLE_PAGES = 6
WHOLE_TOTAL = 12
WHOLE_ROOM = 12
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
LIST_SIZE = 8
LIST_WORLD = 3
RESCALE = 6.0
CAP_ARG = '(?:^|_)(?:cap|max|maximum|ceiling|under|up_?to)(?:_|$)'
CAP_NOT = 'result|row|page|count|size|num|item|step'
SHELF_ARG = 'section|department|categor|label|shelf|aisle|(?<!age)(?:^|_)group(?:_name|_label)?$'
LIST_MODE = 'field|filter|facet|brows|categor|list|structured'
AMOUNT = '\\d[\\d.,]*\\s*[a-z"\']{1,12}'
FLOOR = '\\b(?:at least|no (?:less|fewer) than|min(?:imum)?)\\s+(' + AMOUNT + ')\\s*$'
QUERY_WORDS = '(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword'
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_llm_short: dict[int, ProxyClient] = {}
_TEXT_WORLD = [False]

class _Store:
    """The store's transport: the proxy client's post_verbose, the one call the agent makes on it, except that a
    listing row the store sends as one string reaches the rest of the code as the dict a dict store would send (see
    _text_records). The rest of the result comes back as the client returned it."""

    def __init__(self, client: ProxyClient):
        self.client = client

    def post_verbose(self, path: str, json_data: Any=None) -> Any:
        result = self.client.post_verbose(path, json_data)
        data = result.data
        if isinstance(data, dict) and 'observation' in data:
            try:
                rows = _text_records(data['observation'])
                if rows is not data['observation']:
                    _TEXT_WORLD[0] = True
                data['observation'] = rows
            except Exception:
                pass
        return result
_env = _Store(ProxyClient(timeout=95, max_retries=3, retry_delay=1))
GUIDE = 'You are Zenith, a careful buyer working for one shopper in an online store. You can act only by calling the store\'s tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.\n\nHOW THE STORE WORKS\n- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool\'s description and schema and use the exact argument names and allowed values.\n- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.\n- Search rows are summaries. Stores split a listing\'s record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Zenith knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.\n- Questions to the shopper are handled for you by Zenith\'s interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.\n- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Zenith carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store\'s tools (product questions go through Zenith\'s interviewer).\n\nWHAT COUNTS AS A FIT\n- The kind of product is decided by the listing\'s category label: only the label that matches the requested item as a whole counts, never a neighbouring one.\n- A requirement on a property is met when the listing\'s attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value\'s words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.\n- An attribute whose key names the requirement\'s property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing about the requirement. Zenith adds [evidence] lines after opened listings with the keys whose names match a requirement.\n- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.\n- When the category label itself states a requirement\'s value (a season or audience in the label), that requirement is met by the category.\n- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing\'s value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.\n- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.\n- When one requested item must fit another (an accessory for a device), choose the two together from the listings\' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.\n- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.\n- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.\n\nCHOOSING\n- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Zenith reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.\n- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.\n- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.\n- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.\n- A request for several different items needs one cart line per item, each at its requested quantity.\n\nCHANGES WHILE YOU WORK\n- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool\'s result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.\n- Re-read a chosen listing\'s record shortly before checking out, because price and stock can change silently.\n\nFINISHING\n- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.\n- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request\'s wording for that ending, and when a remark is needed, keep it short and name no listing.\n- Every reply must call at least one tool. Keep your text short: what you know, then the next move.'
BRIEF = 'Job: brief a buying agent on one shopping task. Inputs: the store\'s tools plus the shopper\'s request. Output: ONE JSON object, compact and unindented, with nothing around it. Keys:\n\n"items": list, one entry per distinct product to be bought, each {"name": the product kind, worded as in the request, "shelf": the store\'s category label for it, copied verbatim, if some tool lists labels; otherwise the category the request names for it ("a coat in rainwear" gives "rainwear"); otherwise "", "quantity": integer units of that single listing (1 if unstated; a pair or set sold as one listing is 1), "musts": list of firm requirements on the product, each written "property: constraint" and holding the exact limit, value, exclusion or closed list. The property name must be the request\'s own word ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"). Never substitute an adjacent property. When a must takes its value from another listing, it stays ONE must containing the full description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"). Words naming that other listing (the shop, the kind, a trailing "with <property> <value> and <property> <value>") stay inside that single must; they are never musts of the product being bought}. Price, budget, stock and quantity never go into musts. The request\'s surrounding story (intended user, where it will stand, earlier failures, the occasion behind it) counts as context, never as a must, unless it spells out a property\'s value or limit directly. Words forming part of the product\'s name (a season or audience named together with the product) must never become musts; they belong in "name". If the request makes a use or occasion a property belonging to the product itself ("blazers for business wear", "a dress made for parties", "a bag, and it is for a wedding", "the occasion has to be party"), that use or occasion then IS a must for that property ("occasion: business"); exception: the category label of the store names it already. These remain context: one particular person\'s event, a person or setting the purchase is meant for ("to wear at his wedding", "for a colleague"), and how it should look, sit or feel in use. Products named in a single phrase joined by "and" (one "I want ..." with a single quantity, often a store department\'s name) form ONE item. Several items only when every product has a quantity or requirements of its own.\n"budget": the shopper\'s total spending ceiling, as a number in the unit the request itself uses; or null.\n"budget_unit": that unit spelled out (e.g. "cents"); or null.\n"budget_scope": "total" if the limit applies to the entire order, "each" if it applies per item, null if no limit exists.\n"in_stock_only": true, unless the request plainly accepts unavailable items.\n"pick": "cheapest" if the request wants the minimum price or smallest total; "most" or "least" if it wants the extreme of some measure; else "any". A remark that money is tight is no pick rule.\n"pick_measure": the measure maximised or minimised under "most"/"least"; else "".\n"messages": integer count of messages the shopper lets the agent send; null if unstated.\n"steps": integer limit on steps, moves or tool calls set by the request; or null.\n"lookups": list of the musts (verbatim) whose value has to be read first from another listing the request refers to; else [].\n"rules": list of every instruction in the request about the way to proceed or the way to finish, each 12 words at most.\n"wishes": a list of soft preferences: what the shopper would like but does not demand.\n"ask_first": a list of points the request expressly wants confirmed with the shopper before choosing (empty if none).\n"process": list of steps the request demands beyond locating and buying the right items (get the shopper\'s approval; tell or show the shopper something; check something with them; read something first; use a given tool), each {"step": the required action, 15 words at most, "when": "choose" (before products are chosen), "order" (immediately before checkout) or "end" (immediately before ending without a purchase), "messages": number of messages to the shopper it needs (0 if none)}; [] if the request demands no such step. Not steps: item requirements, the budget, stock, message and step limits, the way to finish when nothing fits.\n"quiet_decline": true if the request forbids suggesting or naming alternatives when nothing fits; false if it asks for them; null if silent.\n"fits": list of {"item": item number, "with": item number}, one per requested item that, per the request, has to work with, suit or fit ANOTHER requested item (whichever the agent picks); [] if none.\n"roles": object mapping each job to its exact tool name; null where no tool does the job: "search", "open" (displays one listing\'s attributes or properties; with two single-listing tools, choose the tool whose description says it shows attributes, properties or specifications), "offer" (displays one listing\'s current offer, e.g. price and stock, when that tool differs from "open"), "compare" (several listings in full), "ask" (messages the shopper), "cart" (modifies the cart), "cart_view" (displays the cart), "buy" (checks out), "refuse" (ends without a purchase).\n"buy_args": fixed arguments with which the "buy" tool checks out (for a tool taking a mode argument, that mode); else {}.\n"refuse_args": fixed arguments with which the "refuse" tool ends without a purchase; else {}.\n'
PLAN = 'Job: pick the questions a shopping agent will send its shopper. Each shopper holds a few firm, unwritten preferences, each tied to a single property of one requested product. A preference comes out only when a question names that exact property. Where the shopper has no view, the reply is "don\'t mind"; where the shopper never thinks about the property at all, the reply is puzzled. Every named property therefore uses up part of a message.\n\nInput: the request; the items, and what the request settles already for each; a numbered list of the attribute keys found on opened listings in each item\'s category, with the count of those listings carrying the key, plus example values. An unwritten preference may concern a property absent from every listed key.\n\nSelection order:\n0. Whatever the request itself says to check with the shopper (shown under ASK FIRST if present), each identified by the property involved.\n1. Hinted properties. The story in a request often hints at an unwritten preference but leaves out its value: a sentence on something beyond the shopper\'s judgement or left to you, on someone whose taste decides, on the product\'s appearance or the setting it fits, on the person who will use or wear it, or on what it contains. Per such sentence, name the single property it hints at and the item it concerns, in story order.\nUse a listed key of that item as the hinted property\'s name when one fits the hint (hint on who will wear it: the gender or audience key; hint on looks: whichever listed colour, pattern or finish key most of its listings carry). Any property carried by none of the item\'s listings ranks below the three keys the item\'s listings carry most.\n2. Next, the listed keys, ranked. Prefer properties a buyer of that product kind decides on first (often type or style, form or fit, size, capacity or dimensions, material, colour or pattern: whichever that product kind is bought by), and keys most of the item\'s listings carry. After five or more listings have been read, avoid keys that only one or two of them carry.\nSkip: any property already settled by the request for that item (settled = some requirement names it; NOT settled = a key whose values merely repeat a required value, e.g. a pattern key that displays the required decoration); a second name for an already chosen property; a key whose every value lists several purchase options at once; also identifiers, model names, warranty and similar logistics. Pack sizes, counts and per-pack quantities are ordinary properties. Any key tagged [not the stated ...] names a property other than that requirement, even if its name shares words with it. Rank each item separately: hinted properties first, listed keys after.\n"label" holds the plain name of the property. For a listed key: all its words, spelled plainly ("lid colour" for lidColour, "rim finish" for rim_finish; never cut to "colour" or "finish"). The name alone: no sample values, no brackets, never a second property.\nPick at most COUNT per item, best first. Output ONE JSON object, no other text: {"ask": [{"item": item number, "label": "...", "hint": "the story\'s words it comes from, 8 at most, or an empty string"}]}, listing each item\'s choices in rank order'
HEAR = 'A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants them asked one at a time; not answers that only say they did not follow it or want it in simpler words], "budget": new total limit as a number or null}\n- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3"); the value is only what a listing could state, from the wanted value or its limit word to its end, without the framing the answer puts around it ("oval" for "matching oval", "settled, with oval" or "oval and nothing else", "at least 30 cm" for "equal to at least 30 cm").\n- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop\'s product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer\'s own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.\n- "free": the property of each question the shopper said they do not mind about.\n- Folksy or roundabout wording that names a value still states that value. Wording that refuses to bargain over a value (it will not be traded, given up or changed) states that value as wanted; write "not X" only when the answer rejects X itself.\n- Some replies also carry remarks the shopper volunteered. They come after the answer, inside square brackets opening with "said besides the answer:" (this marker text appears verbatim in the reply). Such remarks answer no question. A new spending limit goes to "budget"; a requirement given up goes to "drop"; a new firm wish goes to "needs", filed under the property the remark names. Never turn a remark\'s amount or value into a need on the property the question asked about.\n- State "budget" in the unit of the budget in force (given as "BUDGET IN FORCE: <amount> <unit>" when known). Convert the shopper\'s figure into that unit; cents stay cents.\n- Refusing all values but one = a need for that one value ("other than walnut, nothing for me" -> "walnut"). Use "not X" only when X itself is the value refused.'
NOTE = 'While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:\n{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request\'s unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}\nA replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note.'
CHECK = 'Role: final review ahead of a shopping agent\'s checkout. From the records shown, judge whether the cart is exactly right.\nHow to read records: a requirement passes if the listing\'s attribute for the property gives the value. Key spelling is free; one list element suffices; letter case, singular/plural and plain synonyms are equal; a value that contains the required value\'s words passes (e.g. "deep teal" for teal); convert units; limits are inclusive; a range must fall entirely inside the bound. When an attribute\'s key names the requirement\'s own property, that attribute decides; a different value there fails the listing. A key for some other property, even one that is related (the colour of a handle is not the colour of the body), settles nothing. With no attribute for the property, the wanted value counts if it appears under a different attribute, in the title or in the description. Not accepted: a number of another kind (weight never proves capacity), or a bare figure whose text never says what it measures. A requirement stated by the category label itself is met by that category. That label must also match the entire requested item. A rule to be unlike another listing\'s value passes unless that same value appears in the records. Cart rules: one separate cart line per requested item, at the requested quantity; no extra lines; no line marked out of stock; price times quantity, summed across all lines, within the budget. For any listing, its newest record wins.\nPick rule cheapest, most or least: inspect the other listings shown too. If one of them clearly satisfies every requirement and does better than the cart\'s pick under that rule, the cart fails.\nOutput ONE JSON object, no other text: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [ids of listings to swap in for a cart line, best first]}. ok=false is reserved for a problem the records plainly show, or a requirement no record covers.'
READING = 'the listing\'s attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value\'s words counts, such as "deep teal" for teal, limits include the edge); a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing\'s value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit.'
AUDIT = 'A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: ' + READING + '\nRecords marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.\nReturn ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}'
UNMIX = 'Context: a shopper\'s request was parsed into firm requirements on one product. Some requirements get their value from a second listing that the request refers to. The words identifying that second listing (its kind, its shop, properties listed after it with "with ...") are at times misread as requirements on the product to buy. List, by the request\'s wording, those OTHER REQUIREMENTS below that do nothing but describe the second listing. Output ONE JSON object, no other text: {"describe_other": [those requirements, copied verbatim; empty if each one is a genuine requirement on the product]}'
TRAIL = 'Situation: a shopping agent will check out next. Some requirements do not state their value; they take it from another listing. A listing\'s page may point the reader onward to yet another listing for the same property. The order is valid only if the agent has fully opened every listing on each such chain. Trace each requirement\'s chain: start at the requirement\'s own description and go link by link, guided by what the pages themselves say. Use the records below (OPENED = page opened by the agent; ROW = only a search result row was seen, page unread).\nOutput ONE JSON object, no other text: {"chains": [{"need": the requirement, copied verbatim, "links": [{"id": the listing id if that link is one of the records below, else "", "about": how the requirement or the preceding page describes that listing, "find": "search words for a link not found (its shop plus the values from the description)"}], "done": true if the final link is an OPENED page stating the property\'s value itself (also true if the requirement wants a value unlike or other than the stated one: that page\'s value is the reference to measure against), else false, "value": the property\'s value as that final page states it, or ""}]}\nA record counts as a link only if it matches the link\'s full description: shop or brand, product kind AND each property value the description gives. Example: "a 10 pieces multipack" excludes a listing from the same shop that has 10 capsules but no multipack. If no record matches everything, the link counts as not yet found, even though that shop may sell other listings of the same kind. Such a link gets "find" = a handful of search words: the shop\'s name plus the values from the description.\nList the links of every chain in order. Halt at the earliest link that is not found or whose page is not OPENED; what it says is still unknown.'
GATE = 'Context: the shopping agent will STAGE next. On the way, the shopper\'s request demands the steps below. Using the numbered RECORD of the agent\'s actions, decide for each step due before it may STAGE whether that step is done for the cart in its current state. Done means a RECORD line shows the step performed; for a cart or order step, that line must also come after the cart\'s last change, and its reply must grant the step. Questions on product properties never perform a step.\nFor a step still missing, name the next store call that performs it: a tool from TOOLS plus its arguments. Shopper messages use the messaging tool: a single short sentence or question in plain words about that step only, naming no product property or value the step does not need. Name no call if every due step is done, or if the remaining messages or steps cannot perform the missing one.\nOutput ONE JSON object, no other text: {"steps": [{"step": the step, "done": true or false, "lines": [RECORD line numbers showing it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": a change a reply requests before going further, or "", "needs": [{"item": item number, "need": "property: constraint"} one per firm requirement that such a reply states]}'
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
    return [re.sub('\\s+', ' ', c).strip().lower() for c in re.findall('\\b' + FIT_CODE + '\\b', str(text or ''))]

def _fit_marks(text: Any) -> dict[str, Any]:
    text = str(text or '')
    own = _hit_part(re.search(MARK_OWN, text, re.I))
    codes = _hit_codes(re.finditer(MARK_MODEL, text, re.I)) + _hit_codes(re.finditer(MARK_ALIAS, text, re.I))
    works = _hit_codes((hit for hit in re.finditer(MARK_WORKS, text, re.I) if not _other_part_before(text, hit.start())))
    also = [(hit.group(1).upper(), _codes(hit.group(2))) for hit in re.finditer(MARK_ALSO, text, re.I)]
    more = _hit_part(re.search(MARK_MORE, text, re.I))
    return {'own': own, 'codes': codes, 'works': works, 'also': also, 'more': more}

def _dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        return str(value)

def _clip(value: Any, limit: int, depth: int=0) -> Any:
    if depth > 7:
        return '...'
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + '...'
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
    tries = [text]
    left, right = (text.find('{'), text.rfind('}'))
    if 0 <= left < right:
        tries.append(text[left:right + 1])
    for attempt in tries:
        try:
            got = json.loads(attempt)
        except ValueError:
            continue
        return got if isinstance(got, dict) else None
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

def _spelled(text: str) -> frozenset[str]:
    """A name's word set, folding British -our and plural -s, so "colours" and "color" compare equal."""
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
    if kind not in KIND_TYPES:
        return True
    if isinstance(value, bool) and kind in ('integer', 'number'):
        return False
    return isinstance(value, KIND_TYPES[kind])

def _shape(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    if not isinstance(schema, dict):
        return value
    kinds = _kinds(schema)
    if kinds and (not _fits_one_of(value, kinds)):
        value = _shape_cast(value, kinds)
        if not _fits_one_of(value, kinds):
            faults.append(f"{where} should be {'/'.join(kinds)}")
            return value
    value = _shape_enum(value, schema, where, faults)
    if isinstance(value, str):
        return _shape_text(value, schema, where, faults)
    if isinstance(value, list):
        return _shape_list(value, schema, where, faults)
    if isinstance(value, dict):
        return _shape_object(value, schema, where, faults)
    if _schema_number(value):
        _shape_bounds(value, schema, where, faults)
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
    scores = {key: _id_key_score(rows, key) for key in rows[0]}
    usable = [key for key, score in scores.items() if score is not None]
    return max(usable, key=scores.get, default=None)

def _in_rows(node: Any, ident: str, depth: int=0) -> bool:
    if depth > 5:
        return False
    if isinstance(node, dict):
        children = node.values()
    elif isinstance(node, list):
        if any((isinstance(row, dict) and _row_holds(row, ident) for row in node)):
            return True
        children = node
    else:
        return False
    return any((_in_rows(child, ident, depth + 1) for child in children))

def _llm_within(seconds: float) -> ProxyClient:
    """Model client fitted to the time left: the default one, or a client whose timeout is cut to whole 5 s steps."""
    if seconds >= 50:
        return _llm
    limit = max(5, int(seconds) // 5 * 5)
    if limit not in _llm_short:
        _llm_short[limit] = ProxyClient(timeout=limit, max_retries=1, retry_delay=1)
    return _llm_short[limit]

def _fact_block(node: Any) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    blocks = [_scalar_facts(value) for value in node.values() if isinstance(value, dict)]
    return max((block for block in blocks if len(block) >= 2), key=len, default=None)

def _text_value(text: str, block: bool=True) -> Any:
    """One value of a listing row sent as text, typed as a dict row holds it: None, True or False, an int or a float
    for a plain numeral, a dict for a fact block ("k=v; k=v": two or more pairs, and a part with no "=" belongs to the
    value before it), else the text as it stands. A value inside a fact block is never read as a block again
    (block=False)."""
    bare = text.strip()
    if bare in ('None', 'null'):
        return None
    if bare in ('True', 'true'):
        return True
    if bare in ('False', 'false'):
        return False
    if re.fullmatch('[-+]?(?:0|[1-9]\\d*)', bare):
        return int(bare)
    if re.fullmatch('[-+]?(?:0|[1-9]\\d*)\\.\\d+', bare):
        return float(bare)
    if not block:
        return text
    parts = re.split('; (?=[^;=]+=)', text)
    if len(parts) < 2 or '=' not in parts[0]:
        return text
    facts: dict[str, str] = {}
    last = ''
    for part in parts:
        key, _, value = part.partition('=')
        if not key.strip() or ': ' in key or '\n' in key:
            return text
        if key in facts:
            facts[last] += '; ' + part
            continue
        facts[key] = value
        last = key
    return {key: _text_value(value, False) for key, value in facts.items()}

def _pipe_record(text: Any) -> dict[str, Any] | None:
    """The race-15 layout of a listing row sent as one string, read into a dict: three or more "key: value" parts joined by
    " | ", each key letters, digits, _ or - from a letter, and one value a single token of letters and digits (the
    listing id, which a product description written as "label: value" parts lacks). A part that does not open with
    a new key belongs to the value before it, since titles hold " | " too; so does one whose key repeats a key of the
    row, or has capitals when the row's first key has none (a "FREE: ..." inside a title). Anything else gives None,
    and so do "key: value" parts one per line, as product descriptions in dict stores hold such lines."""
    if not isinstance(text, str) or ' | ' not in text or (not re.match('[A-Za-z][A-Za-z0-9_-]*: ', text)):
        return None
    lower = not re.match('[A-Za-z0-9_-]*[A-Z]', text)
    fields: dict[str, str] = {}
    last = ''
    for part in re.split(' \\| (?=[A-Za-z][A-Za-z0-9_-]*: )', text):
        key, _, value = part.partition(': ')
        if last and (key in fields or (lower and re.search('[A-Z]', key))):
            fields[last] += ' | ' + part
            continue
        fields[key] = value
        last = key
    if len(fields) < 3 or not any((re.fullmatch('(?=.*[A-Za-z])(?=.*\\d)[A-Za-z0-9_-]{3,64}', v.strip()) for v in fields.values())):
        return None
    return {key: _text_value(value) for key, value in fields.items()}
_ROW_ID = '(?=.*[A-Za-z])(?=.*\\d)[A-Za-z0-9_-]{3,64}'
_ROW_KEY = '[a-z][A-Za-z0-9_-]{0,40}'
_ROW_SPLITS = ((' | ', '='), ('; ', '='), (', ', '='), ('; ', ': '), ('\n', ': '), ('\n', '='))
_ROW_MIN = 4

def _row_like(fields: dict[str, Any]) -> bool:
    """The mark of a listing row read from text in a layout other than the race-15 one: _ROW_MIN or more fields, one
    of them an id-like single token of letters and digits."""
    return len(fields) >= _ROW_MIN and any((isinstance(v, str) and re.fullmatch(_ROW_ID, v.strip()) for v in fields.values()))

def _json_record(text: str) -> dict[str, Any] | None:
    """A row sent as a JSON object, or as one written with bare None, True and False: the dict, when it reads as a
    row (_row_like)."""
    bare = text.strip()
    if not (bare.startswith('{') and bare.endswith('}')):
        return None
    words = {'None': 'null', 'True': 'true', 'False': 'false'}
    plain = re.sub('"(?:[^"\\\\]|\\\\.)*"|\\b(None|True|False)\\b', lambda m: m.group(0) if m.group(1) is None else words[m.group(1)], bare)
    try:
        value = json.loads(plain)
    except ValueError:
        return None
    if isinstance(value, dict) and all((isinstance(key, str) for key in value)) and _row_like(value):
        return value
    return None

def _split_record(text: str, cut: str, mark: str) -> dict[str, Any] | None:
    """A row sent as text in another layout: "key<mark>value" parts joined by <cut>, every key a name that starts
    with a lower-case letter (a code field name, which a product text's "Colour: red" lines lack). A part that does
    not open with such a key, or repeats a key of the row, belongs to the value before it."""
    fields: dict[str, str] = {}
    last = ''
    for part in re.split(re.escape(cut) + '(?=' + _ROW_KEY + re.escape(mark) + ')', text):
        key, found, value = part.partition(mark)
        if not last and (not found or not re.fullmatch(_ROW_KEY, key)):
            return None
        if last and key in fields:
            fields[last] += cut + part
            continue
        fields[key] = value
        last = key
    if not _row_like(fields):
        return None
    block = mark == ': ' and cut == '\n'
    return {key: _text_value(value, block) for key, value in fields.items()}

def _text_record(text: Any) -> dict[str, Any] | None:
    """A listing row the store sent as one string, read into the dict a dict store would send: the race-15 layout
    first (_pipe_record), else a JSON object (_json_record), else another "key<mark>value" layout (_split_record; of
    those that read it, the one whose first cut comes first, since a block of facts nested in a field comes after
    that field's start). None when no layout reads it as a row."""
    if not isinstance(text, str):
        return None
    record = _pipe_record(text)
    if record is not None:
        return record
    record = _json_record(text)
    if record is not None:
        return record
    best, first = (None, len(text))
    for cut, mark in _ROW_SPLITS:
        at = re.search(re.escape(cut) + '(?=' + _ROW_KEY + re.escape(mark) + ')', text)
        if at is not None and at.start() < first:
            got = _split_record(text, cut, mark)
            if got is not None:
                best, first = (got, at.start())
    return best

def _text_rows(text: str) -> Any:
    """Several rows sent as one text, a row a line ("1." or "-" before a row is dropped): the list of rows when every
    line reads as one; when two or more lines do and at most two others do not, {"rows": [...], "notes": [...]}, so
    no line is lost; else None."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if len(lines) < 2:
        return None
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    for line in lines:
        row = _text_record(re.sub('^(?:\\d{1,3}[.)]|[-*])\\s+', '', line))
        if row is None:
            notes.append(line)
        else:
            rows.append(row)
    if len(rows) < 2 or len(notes) > 2:
        return None
    return rows if not notes else {'rows': rows, 'notes': notes}

def _text_table(value: dict[str, Any]) -> dict[str, Any] | None:
    """A table: one list of _ROW_MIN or more distinct names (the columns) and, under another key, lists as long as it,
    each holding an id-like value: those lists become rows (dicts) under the same key, and the names stay. None when
    the dict holds no such table."""
    heads = [key for key, inner in value.items() if isinstance(inner, list) and len(inner) >= _ROW_MIN and (len(set(map(str, inner))) == len(inner)) and all((isinstance(s, str) and re.fullmatch('[A-Za-z][A-Za-z0-9 _-]{0,40}', s) for s in inner))]
    if len(heads) != 1:
        return None
    names = value[heads[0]]
    new = dict(value)
    for key, inner in value.items():
        if key != heads[0] and isinstance(inner, list) and inner and all((isinstance(r, list) and len(r) == len(names) for r in inner)):
            rows = [dict(zip(names, r)) for r in inner]
            if all((_row_like(row) for row in rows)):
                new[key] = rows
    return new if any((new[key] is not value[key] for key in value)) else None

def _text_records(value: Any, depth: int=0) -> Any:
    """The value with every string that _text_record reads as a listing row replaced by its dict (or that _text_rows
    reads as several rows, by them), and every table (_text_table) by its rows, through dicts and lists down to the
    depth _clip keeps. A dict whose one key holds such a row gives the row itself, as a dict store
    shows a page; a list whose strings are not all rows (notes with a row among them) stays as it came. The very same
    object comes back when nothing changes, so a store that sends dict rows is left alone."""
    if depth > 7:
        return value
    if isinstance(value, str):
        if '\n' in value:
            rows = _text_rows(value)
            if rows is not None:
                return rows
        record = _text_record(value)
        return value if record is None else record
    if isinstance(value, dict):
        new = {key: _text_records(inner, depth + 1) for key, inner in value.items()}
        if all((new[key] is inner for key, inner in value.items())):
            return _text_table(value) or value
        if len(value) == 1 and isinstance(next(iter(value.values())), str):
            return next(iter(new.values()))
        return _text_table(new) or new
    if isinstance(value, list):
        rows = [_text_records(inner, depth + 1) for inner in value]
        if all((a is b for a, b in zip(rows, value))):
            return value
        if any((isinstance(b, str) and a is b for a, b in zip(rows, value))):
            return value
        return rows
    return value

class Item:

    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []
UNSTATED_WORDS = frozenset({'not', 'no', 'least', 'most', 'under', 'over', 'unlike', 'different', 'min', 'max', 'minimum', 'maximum'})

def _lane_has(lane: list[str], label: str) -> bool:
    """Whether `lane` holds a label with the words of `label` (case, spacing and separators aside)."""
    words = _words(label)
    return any((_words(other) == words for other in lane))

def _lane_join(lane: list[str], label: str) -> None:
    """Append `label` to `lane` unless the lane holds it already."""
    if not _lane_has(lane, label):
        lane.append(label)

def _lane_broad(text: str) -> bool:
    """Whether a key names several features in one (CATCH_ALL); such keys are asked last."""
    return bool(re.search(CATCH_ALL, text))

def _lane_label(value: Any) -> str:
    """A planned label cut to the first property it names (before "and"/"or", a slash, comma, semicolon or bracket);
    '' when nothing is left or more than five words are."""
    label = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(value or ''))[0].strip().rstrip('?.')
    return label if len(label.split()) <= 5 else ''

def _lane_plain(text: str) -> str:
    """`text` in words with a plural -s (after three letters) dropped and -our spelled -or."""
    return re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', _words(text)))

def _lane_spelling(own: dict[str, float], label: str) -> str:
    """The item's own key that spells `label` the same up to a plural -s or a British -our, else `label`."""
    plain = _lane_plain(label)
    return next((key for key in own if _lane_plain(key) == plain), label)

def _lane_share(own: dict[str, float], label: str) -> float:
    """The share of the item's listings that carry `label`: its own key's, else the highest among the keys naming it or
    named by it, else 0."""
    words = _words(label)
    if words in own:
        return own[words]
    stem = _stem(label)
    near = [share for key, share in own.items() if _names(stem, _stem(key)) or _names(_stem(key), stem)]
    return max(near, default=0.0)

def _lane_guard(lane: list[str], key: str, own: dict[str, float], hints: set[str], swap: bool) -> list[str]:
    """`lane` with one of the item's best-carried keys given a place among its first three labels, else `lane` itself.

    The key takes the place of the first label there that was not hinted and whose share is at least GUARD_GAP below
    the key's (with `swap` that label moves to fourth place, else it and the labels after it move down one), or ends a
    lane of fewer than three labels; a later copy of the key leaves. Nothing moves when the key is among the first
    three already, when a full top three holds no such weak label, or when a hinted label would leave the top three."""
    top = lane[:3]
    if _lane_has(top, key):
        return lane
    share = own.get(key, 0.0)
    weak = next((i for i, other in enumerate(top) if _words(other) not in hints and _lane_share(own, other) + GUARD_GAP <= share), None)
    if weak is None and len(lane) >= 3:
        return lane
    placed = [other for other in lane if _words(other) != _words(key)]
    if weak is None:
        placed.append(key)
    elif swap:
        moved, placed[weak] = (placed[weak], key)
        placed.insert(min(3, len(placed)), moved)
    else:
        placed.insert(weak, key)
    if any((_words(label) in hints and (not _lane_has(placed[:3], label)) for label in top)):
        return lane
    return placed

def _lane_amounts(values: list[str]) -> bool:
    """Whether most of a key's values are amounts: digits and separators with at most a short unit."""
    amounts = sum((1 for value in values if re.match('^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$', value)))
    return amounts * 2 > len(values)

def _top_index(scores: list[float]) -> int | None:
    """The index of the first highest score; None when no score is above zero."""
    top = max(scores, default=0)
    return scores.index(top) if top > 0 else None

def _shelf_fit(words: str, long: set[str], item: Item) -> float:
    """How well a category label (its words and root words) fits an item: 3 for the item's shelf label, 2 for its
    name, else the share of root words the label and the item's shelf and name have in common."""
    if words == _words(item.shelf or ''):
        return 3.0
    if words == _words(item.name):
        return 2.0
    mine = _roots(item.shelf or '') | _roots(item.name)
    return len(long & mine) / max(1, len(long | mine))

def _shelf_label(record: dict[str, Any] | None, cat_key: str) -> str:
    """The category label a listing record carries under `cat_key`; '' when it carries none."""
    return str((record or {}).get(cat_key) or '')

def _answer_slot(labels: list[str], need: str, seen: dict[str, set[str]]) -> int | None:
    """The index of the label a new need answers: the one sharing most words with the need's property, else the one
    whose values seen on listings share most words with its value; None when it shares no word with any label."""
    prop, _, value = need.partition(':')
    by_name = _top_index([len(_spelled(label) & _spelled(prop)) for label in labels])
    if by_name is not None:
        return by_name
    roots = _roots(value)
    return _top_index([len(seen.get(_words(label), set()) & roots) for label in labels])

def _need_words(text: str) -> set[str]:
    """The words of a value, commas, slashes, semicolons and bars read as spaces."""
    return set(re.sub('[,/;|]', ' ', _words(text)).split())

def _states_need(block: dict[str, Any], need: str) -> bool:
    """Whether a listing's attributes state one requirement word for word: a key naming its property holds every word
    of its value. A value with a limit or an exclusion word counts as not stated."""
    prop, _, value = need.partition(':')
    words = _need_words(value)
    if not words or words & UNSTATED_WORDS:
        return False
    named = _spelled(prop)
    return any((_names(named, _spelled(key)) and words <= _need_words(str(held)) for key, held in block.items()))

def _message_target(props: dict[str, Any], required: list[str]) -> str | None:
    """The message tool's text field: the first required plain-text property, else the first plain-text one."""
    texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
    return next((k for k in required if k in texts), None) or (texts[0] if texts else None)
MARK_OWN = '[\\[(]\\s*(?:part\\s+)?(' + PART_CODE + ')\\s*[\\])]'
MARK_MODEL = '\\bmodel\\s*(?:no\\.?|number|#)\\s*[:.]?\\s*(' + FIT_CODE + ')\\b'
MARK_ALIAS = '\\b(?:also sold as|also known as|sold also as)\\s+([^.;]+)'
MARK_WORKS = '\\b(?<!not )(?<!never )(?:works with|compatible with|fits)\\s*:?\\s+([^.;]+)'
MARK_OTHER_PART = '\\bpart\\s+' + PART_CODE + '\\s+also\\s+$'
MARK_ALSO = '\\bpart\\s+(' + PART_CODE + ')\\s+also fits\\s*:?\\s*([^.;]+)'
MARK_MORE = '\\b(?:more|details|see|read)\\s+(?:on|at|under|about)?\\s*(?:the\\s+)?part\\s+(' + PART_CODE + ')'
KIND_TYPES = {'string': str, 'integer': int, 'number': (int, float), 'boolean': bool, 'array': list, 'object': dict, 'null': type(None)}
RELATIVE_VALUE = '\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b'

def _hit_codes(hits: Any) -> list[str]:
    """The fit codes named in the first group of each regex hit, in order."""
    return [code for hit in hits for code in _codes(hit.group(1))]

def _hit_part(hit: Any) -> str:
    """The part code a regex hit captured, upper-cased; empty without a hit."""
    return hit.group(1).upper() if hit else ''

def _other_part_before(text: str, at: int) -> bool:
    """True when the 40 characters before position `at` end in "part <code> also": the phrase starting there speaks
    of that other part (MARK_ALSO reads it)."""
    return bool(re.search(MARK_OTHER_PART, text[max(0, at - 40):at], re.I))

def _schema_number(value: Any) -> bool:
    """True for what a JSON schema calls a number: an int or a float, never a bool."""
    return isinstance(value, (int, float)) and (not isinstance(value, bool))

def _fits_one_of(value: Any, kinds: list[str]) -> bool:
    return any((_fits_kind(value, kind) for kind in kinds))

def _shape_cast(value: Any, kinds: list[str]) -> Any:
    """The value converted to the first listed kind it plainly stands for: a whole float as an integer, numeric text
    as an integer or number, a number as text, "true" or "false" as a boolean, a single value as a one-entry array.
    Unchanged when no kind applies."""
    for kind in kinds:
        if kind == 'integer' and isinstance(value, float) and value.is_integer():
            return int(value)
        if kind in ('integer', 'number') and isinstance(value, str) and re.fullmatch('\\s*-?\\d+(\\.\\d+)?\\s*', value):
            number = float(value)
            return int(number) if kind == 'integer' or number.is_integer() else number
        if kind == 'string' and _schema_number(value):
            return str(value)
        if kind == 'boolean' and isinstance(value, str) and (value.strip().lower() in ('true', 'false')):
            return value.strip().lower() == 'true'
        if kind == 'array' and (not isinstance(value, list)) and (value is not None):
            return [value]
    return value

def _shape_enum(value: Any, schema: dict[str, Any], where: str, faults: list[str]) -> Any:
    """A value checked against the schema's options: text naming an option up to case and outer spaces becomes that
    option, and any other value outside them is a fault."""
    options = schema.get('enum')
    if not isinstance(options, list) or value in options:
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        value = next((option for option in options if isinstance(option, str) and option.lower() == low), value)
    if value not in options:
        faults.append(f'{where}: allowed values are {options}')
    return value

def _shape_text(value: str, schema: dict[str, Any], where: str, faults: list[str]) -> str:
    """Text cut to maxLength; text shorter than minLength is a fault."""
    if isinstance(schema.get('maxLength'), int) and len(value) > schema['maxLength']:
        value = value[:schema['maxLength']]
    if isinstance(schema.get('minLength'), int) and len(value) < schema['minLength']:
        faults.append(f'{where} is too short')
    return value

def _shape_bounds(value: int | float, schema: dict[str, Any], where: str, faults: list[str]) -> None:
    """A number below the schema's minimum or above its maximum is a fault."""
    if isinstance(schema.get('minimum'), (int, float)) and value < schema['minimum']:
        faults.append(f"{where} is below {schema['minimum']}")
    if isinstance(schema.get('maximum'), (int, float)) and value > schema['maximum']:
        faults.append(f"{where} is above {schema['maximum']}")

def _shape_list(value: list[Any], schema: dict[str, Any], where: str, faults: list[str]) -> list[Any]:
    """An array checked against maxItems and minItems, each entry shaped by the items schema when there is one."""
    if isinstance(schema.get('maxItems'), int) and len(value) > schema['maxItems']:
        faults.append(f"{where} allows at most {schema['maxItems']} entries")
    if isinstance(schema.get('minItems'), int) and len(value) < schema['minItems']:
        faults.append(f"{where} needs at least {schema['minItems']} entries")
    entry = schema.get('items')
    if not isinstance(entry, dict):
        return value
    return [_shape(v, entry, f'{where}[{i}]', faults) for i, v in enumerate(value)]

def _shape_object(value: dict[str, Any], schema: dict[str, Any], where: str, faults: list[str]) -> dict[str, Any]:
    """An object checked for keys the schema does not allow and required keys it lacks, each listed property shaped
    by its own schema (other keys pass unchanged)."""
    props = schema.get('properties') if isinstance(schema.get('properties'), dict) else {}
    if schema.get('additionalProperties') is False:
        for name in value:
            if name not in props:
                faults.append(f'{where}.{name} is not an allowed argument')
    for name in schema.get('required') or []:
        if name not in value:
            faults.append(f'{where}.{name} is required')
    return {k: _shape(v, props[k], f'{where}.{k}', faults) if k in props else v for k, v in value.items()}

def _id_key_score(rows: list[dict[str, Any]], key: Any) -> tuple[bool, float] | None:
    """How well a key of the first row serves as the listing id: (its name says id, the mean length of its values),
    or None when a row's value is not id-like or two rows share one value."""
    named = 'id' in str(key).lower()
    values = [row.get(key) for row in rows]
    if not all((_id_like(value, named) for value in values)):
        return None
    values = [value.strip() for value in values]
    if len(set(values)) < len(values):
        return None
    return (named, sum((len(value) for value in values)) / len(values))

def _row_holds(row: dict[str, Any], ident: Any) -> bool:
    """True when one of the row's text values, trimmed, is the ident."""
    return any((isinstance(value, str) and value.strip() == ident for value in row.values()))

def _scalar_facts(block: dict[str, Any]) -> dict[str, Any]:
    """The entries of a nested dict whose values are text or numbers (never bools), in order."""
    return {key: value for key, value in block.items() if isinstance(value, str) or _schema_number(value)}

def _name_stems(text: str) -> set[str]:
    """The word stems of four or more letters in a name."""
    return {w for w in _stem(text) if len(w) >= 4}

def _tally_facts(slots: dict[str, dict[str, Any]], ident: str, facts: dict[str, Any]) -> None:
    """Count one record's facts into its category's slots. Keys that read as the same words count once, by the first
    of them; a value is kept trimmed and lower-cased, as this record's value and once in the slot's value list."""
    named: dict[str, str] = {}
    for key, value in facts.items():
        name = _words(key)
        if name and name not in named:
            named[name] = str(value).strip().lower()
    for name, text in named.items():
        slot = slots.setdefault(name, {'n': 0, 'values': [], 'per': {}})
        slot['n'] += 1
        slot['per'][ident] = text
        if text not in slot['values']:
            slot['values'].append(text)

def _free_text(values: list[str]) -> bool:
    """True when a key's values read as free text rather than as one property: over 40 characters on average, half
    or more of them listing parts (" / "), or four or more values, all distinct and over 12 characters on average,
    fewer than half of which share a word with the others."""
    mean = sum((len(v) for v in values)) / len(values)
    if mean > 40 or sum((1 for v in values if ' / ' in v)) * 2 >= len(values):
        return True
    return len(values) >= 4 and len(set(values)) == len(values) and (mean > 12) and (_shared_words(values) * 2 < len(values))

def _shared_words(values: list[str]) -> int:
    """How many values share a word of three or more letters with the other values (copies of the same text object
    do not count as others)."""
    words = [set(re.findall('[a-z]{3,}', v)) for v in values]
    shared = 0
    for value, own in zip(values, words):
        others = set().union(*(theirs for other, theirs in zip(values, words) if other is not value))
        if own & others:
            shared += 1
    return shared

def _merge_slot(twin: dict[str, Any], slot: dict[str, Any]) -> None:
    """Fold a key's slot into the twin slot kept before it: the twin takes each record's value it lacks and each new
    distinct value, and its count becomes the number of records it now covers."""
    for ident, text in slot['per'].items():
        twin['per'].setdefault(ident, text)
        if text not in twin['values']:
            twin['values'].append(text)
    twin['n'] = len(twin['per'])

def _stated_words(text: str) -> set[str]:
    """The words and numbers of three or more characters in a text, lower-cased."""
    return set(re.findall('[a-z0-9]{3,}', text.lower()))

def _sibling_property(prop: str, name: str) -> bool:
    """True when a requirement's property is another attribute than the key name but shares a word stem with it."""
    return bool(prop.strip()) and _words(prop) != name and bool(_stem(prop) & _stem(name))

def _chat_body(model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> dict[str, Any]:
    """The proxy request for one chat completion. An Anthropic model's chat of two or more messages carries
    prompt-cache marks."""
    if model.startswith('anthropic/') and len(messages) > 1:
        messages = _cache_marked(messages)
    body: dict[str, Any] = {'model': model, 'messages': messages, 'max_tokens': max_tokens, 'temperature': 0}
    if tools:
        body['tools'] = tools
        body['tool_choice'] = 'auto'
    return body

def _cache_marked(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A copy of a chat of two or more messages whose first and last messages hold their text as one block marked
    for the prompt cache."""
    marked = [dict(m) for m in messages]
    for message in (marked[0], marked[-1]):
        content = message.get('content')
        if isinstance(content, str) and content:
            message['content'] = [{'type': 'text', 'text': content, 'cache_control': {'type': 'ephemeral'}}]
    return marked

def _failure_of(result: Any) -> tuple[Any, str]:
    """The status and the body (first 200 characters) of a failed model request. An error object inside the reply's
    data overrides the transport's error."""
    error = result.error or {}
    status = error.get('status')
    said = error.get('body')
    data = result.data
    if isinstance(data, dict) and isinstance(data.get('error'), dict):
        status = _count(data['error'].get('code'))
        said = data['error'].get('message')
    return (status, str(said or '')[:200])

def _passes(status: Any) -> bool:
    """Whether a failed model request may work when sent again: no status (the connection failed or the code was
    unreadable), lost credit (402), a timeout (408), too early (425), a rate limit (429) or a server error."""
    return status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)

def _model_turn(message: dict[str, Any], turn: int) -> tuple[str, list[dict[str, Any]]]:
    """A model message as the chat keeps it: its text (other content as JSON) and its tool calls. Each call gets an id
    (k<turn>_<position> when the model gave none), a name, and its arguments as one JSON object ({} when they are not
    one)."""
    calls = []
    for number, call in enumerate(message.get('tool_calls') or []):
        if not isinstance(call, dict):
            continue
        fn = call.get('function')
        if not isinstance(fn, dict):
            fn = {}
        args = fn.get('arguments')
        if isinstance(args, str):
            args = _json_obj(args) if args.strip() else {}
        if not isinstance(args, dict):
            args = {}
        calls.append({'id': str(call.get('id') or f'k{turn}_{number}'), 'type': 'function', 'function': {'name': str(fn.get('name') or ''), 'arguments': _dump(args)}})
    content = message.get('content')
    if not isinstance(content, str):
        content = _dump(content) if content else ''
    return (content, calls)

def _arg_texts(args: dict[str, Any]) -> list[str]:
    """Every text argument, and every text inside a list argument, stripped: the listing ids a page call names."""
    texts = []
    for value in args.values():
        for part in value if isinstance(value, list) else [value]:
            if isinstance(part, str):
                texts.append(part.strip())
    return texts
STEP_WHEN = {'choose': 'before choosing', 'order': 'at the order (Zenith does it when you check out)', 'end': 'at an ending without a purchase (Zenith does it then)'}

def _long_stems(text: str) -> set[str]:
    """The stemmed words of a text that have four letters or more."""
    return {w for w in _stem(text) if len(w) >= 4}

def _texts_agree(x: str, y: str) -> bool:
    """Two values agree when they are equal, or when every word of the shorter one is in the longer one and the longer
    one has at most twice as many words, plus two."""
    if x == y:
        return True
    wx, wy = (re.findall('[a-z0-9]+', x.lower()), re.findall('[a-z0-9]+', y.lower()))
    short, long = (wx, wy) if len(wx) <= len(wy) else (wy, wx)
    return bool(short) and set(short) <= set(long) and (len(long) <= 2 * len(short) + 2)

def _enum_labels(schema: Any, depth: int=0) -> list[str]:
    """The string options of every enum in a tool schema, stripped and lower-cased, in schema order, looking into
    nested objects (not lists) down to depth 6."""
    if depth > 6 or not isinstance(schema, dict):
        return []
    found = [option.strip().lower() for option in schema.get('enum') or [] if isinstance(option, str)]
    for value in schema.values():
        if isinstance(value, dict):
            found += _enum_labels(value, depth + 1)
    return found

def _label_rank(values: set[str], carried: int, known: set[str]) -> tuple[float, int, int, float]:
    """Sort key of a candidate category column, best first: the largest share of values that are tool enum labels,
    then the fewest distinct values, the most rows carrying it, the longest values."""
    listed = sum((1 for v in values if v.strip().lower() in known)) / len(values)
    mean = sum((len(v) for v in values)) / len(values)
    return (-listed, len(values), -carried, -mean)

def _fit_text(record: dict[str, Any]) -> str:
    """The text fit marks are read from: a record's own string fields, then the string attributes of its nested blocks
    as "key value" sentences."""
    own = ' '.join((str(v) for v in record.values() if isinstance(v, str)))
    nested = '. '.join((f'{k} {v}' for block in record.values() if isinstance(block, dict) for k, v in block.items() if isinstance(v, str)))
    return own + ' ' + nested
WALK_HOPS = 6
WALK_START = 20
WALK_ROOM = 14
WALK_BATCH = 3
WALK_OPENS = 14
WALK_SEARCHES = 10
WALK_CLOCK = 175.0
HOP = 'A shopping agent is following a chain of listings in a store. One requirement on the product it buys does not state its value: it describes another listing (the LISTING SOUGHT below), and the page of that listing either gives the value of the PROPERTY or names a further listing to take it from, which may in turn name another.\nReturn ONE JSON object and nothing else:\n{"sought": {"store": false when the LISTING SOUGHT is another product of this same order (one the shopper buys, picks or owns along with it) rather than a listing of the store told apart by its own details, else true, "shop": the shop, seller or maker the LISTING SOUGHT names, or "", "kind": the kind of product it names, in at most 3 words, or "", "words": up to 3 search words, each a single word from a different value the description gives (prefer a number or an uncommon word; never the shop, the kind, a unit or a filler word)}, "match": the id of the page below that is the listing sought, or "", "value": the PROPERTY\'s value as the matching page states it for itself, or "", "next": null, or, when the matching page sends the reader to another listing for the PROPERTY, {"text": the page\'s words that describe that listing, "shop": ..., "kind": ..., "words": [...]} (filled as in "sought")}\n- A page is the listing sought only when the page itself (its remarks and its listed details, not just its title, which can mislead) agrees with the description on the shop, the kind of product and every value the description gives. A page of the same shop and kind with a different value is a different listing; when no page agrees on everything, "match" is "".\n- When the matching page says to take the PROPERTY from another listing, fill "next" and leave "value" empty, even when the page also lists a value for that property. A remark that sends the reader elsewhere for some other property does not count.\n- With no pages below, fill only "sought".'

def _walk_words(text: Any) -> set[str]:
    """The words of a text for loose matching: lower case, letters and digits apart ("50ml" gives 50 and ml), a
    plural s dropped."""
    out: set[str] = set()
    for word in re.findall('[a-z]+|\\d+(?:\\.\\d+)?', str(text).lower()):
        out.add(word[:-1] if len(word) > 3 and word.endswith('s') else word)
    return out

class Zenith:

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
        self.id_key: str | None = None
        self.facts_from: dict[str, bool] = {}
        self.queued: list[str] = []
        self.buys = 0
        self.refusals = 0
        self.recent: list[str] = []
        self.chat: list[dict[str, Any]] = []
        self.trim_mark = 0
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
        self.whole_read = 0
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
        self.follow_up: list[tuple[int, list[str], bool]] = []
        self.listing = True
        self.line_holds = 0
        self.bound: list[tuple[int, int]] = []
        self.fit_holds = 0
        self.searched: list[dict[str, Any]] = []
        self.link_searches = 0
        self.link_seen: dict[str, list[str]] = {}
        self.walked: set[str] = set()
        self.walk_pages: dict[str, Any] = {}

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
        return 2 if talk and self.msg_used < self.msg_limit and self.gate_due('order') else 1

    def judges(self, limit: float=SPEND_SOFT) -> list[str]:
        if self.cost <= limit and self.cost + HEAVY_ROOM <= SPEND_HARD:
            return [HEAVY, LIGHT]
        return [LIGHT] if self.cost <= SPEND_HARD else []

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None=None) -> dict[str, Any] | None:
        for model in models:
            if not self.call_allowed():
                return None
            self.calls += 1
            message, status = self.request_with_waits(model, messages, max_tokens, tools)
            if message is not None:
                return message
            if status == 402:
                self.no_credit = True
                return None
        return None

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        body = _chat_body(model, messages, max_tokens, tools)
        result = _llm_within(self.model_room()).post_verbose('/inference/chat/completions', body)
        data = result.data
        if isinstance(data, dict) and data.get('choices'):
            return self.booked_answer(model, data)
        status, said = _failure_of(result)
        self.trace.append({'kind': 'llm_fail', 'model': model, 'status': status, 'body': said})
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
        payload = {**self.binding, 'step': self.step + 1, 'action': {'name': name, 'args': args}}
        for _ in range(4):
            result = _env.post_verbose('/environment/call', payload)
            if isinstance(result.data, dict):
                return self.booked_step(name, args, result.data)
            if not self.store_retry(result):
                return None
            payload['step'] = self.step + 1
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
        self.listing = self.msg_limit == LIST_WORLD
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
        shelves = [_words(str(e.get('shelf') or '')) for e in _items(brief.get('items')) if isinstance(e, dict)]
        shelves = [s for s in shelves if s]
        return len(shelves) != len(set(shelves))

    def unmix(self) -> None:
        for number, item in enumerate(self.items, 1):
            pointers = [need for n, need in self.refs if n == number and need in item.musts]
            rest = [m for m in item.musts if m not in pointers]
            if not pointers or not rest:
                continue
            moved = self.unmix_moved(item, pointers, rest)
            if moved:
                self.unmix_fold(number, item, pointers[0], moved)

    def described(self, number: int, need: str) -> bool:
        if self.points_elsewhere(need):
            return False
        prop, _, value = need.partition(':')
        if not prop.strip() or len(re.findall('[a-z0-9]+', value.lower())) < 4:
            return False
        if re.search('\\b(?:listing|page|shown|other than|unlike|same as)\\b', value, re.I):
            return False
        item = self.items[number - 1]
        value_stems = _stem(value)
        if any((_long_stems(other.name) & value_stems for other in self.items if other is not item)):
            return False
        blocks = self.item_fact_blocks(item)
        if len(blocks) < 3:
            return False
        keys = {_words(k) for block in blocks for k in block}
        prop_stems = _stem(prop)
        return not any((_stem(key) & prop_stems for key in keys))

    def ledger(self) -> str:
        many = len(self.items) > 1
        parts = [self.ledger_item(number, item, many) for number, item in enumerate(self.items, 1)]
        parts += self.ledger_terms()
        for entry in self.process:
            when = STEP_WHEN[entry['when']]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        facts = self.ledger_facts()
        if facts:
            parts.append(facts)
        return '\n'.join(('- ' + p for p in parts))

    def status(self) -> str:
        left = self.left()
        tail = self.tail()
        text = f'[Zenith] step {self.step} of {self.max_steps} used, {left} left.' + self.status_warning(left, tail)
        if self.clock() >= WIND_DOWN and left > tail + 1:
            text += ' TIME IS NEARLY UP: set the cart to the chosen lines and check out within the next two turns; an episode that runs out of time earns nothing.'
        return text + self.status_notes() + '\n[ledger]\n' + self.ledger()

    def ident(self, node: Any, in_list: bool=False, depth: int=0) -> str | None:
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if self.keyed_by_id([node]):
            return node[self.id_key].strip()
        nested = any((isinstance(v, dict) for v in node.values()))
        if in_list or depth == 0 or nested:
            return self.guessed_id(node)
        return None

    def learn_rows(self, node: list[Any]) -> bool:
        rows = [n for n in node if isinstance(n, dict) and len(n) >= 3]
        if not rows or len(rows) < len(node) / 2:
            return False
        if self.keyed_by_id(rows):
            return True
        found = _learn_id(rows)
        if found:
            self.id_key = found
        return bool(found)

    def absorb(self, observation: Any) -> None:
        self.absorb_node(observation, 0, False)

    def labels(self) -> set[str]:
        return {label for schema in self.schemas.values() for label in _enum_labels(schema)}

    def category_key(self) -> str | None:
        texts, carried = self.short_texts()
        floor = 0.8 * max(1, len(self.rows))
        known = self.labels()
        ranked = [(key, _label_rank(values, carried[key], known)) for key, values in texts.items() if carried[key] >= floor and len(values) <= 40]
        if not ranked:
            return None
        return min(ranked, key=lambda pair: pair[1])[0]

    def key_table(self) -> list[tuple[str, str, int, list[str]]]:
        groups = self.requested_groups(self.key_groups())
        table = []
        for label, group in sorted(groups.items(), key=lambda kv: -kv[1]['n']):
            for name, slot in self.kept_keys(group['keys']):
                table.append((label, name, slot['n'], [v[:40] for v in slot['values'][:4]]))
        return table

    def not_stated(self, label: str, name: str, values: list[str]) -> str:
        number = self.owner_of(label)
        if not number:
            return ''
        shown = _stated_words(' '.join(values))
        for must in self.items[number - 1].musts:
            prop, _, value = must.partition(':')
            if re.search(RELATIVE_VALUE, value, re.I) or not _sibling_property(prop, name):
                continue
            words = _stated_words(value)
            if words and (not words <= shown):
                return f' [not the stated {prop.strip()}: another property]'
        return ''

    @staticmethod
    def twins(name: str, spec: dict[str, Any], other: str, slot: dict[str, Any]) -> bool:
        a, b = (set(name.split()), set(other.split()))
        if not a or not b or (not a & b):
            return False
        mine, theirs = (spec['per'], slot['per'])
        both = [i for i in mine if i in theirs and mine[i].strip() and theirs[i].strip()]
        same = sum((1 for i in both if _texts_agree(mine[i], theirs[i])))
        if a <= b or b <= a:
            if both:
                return same * 2 > len(both)
            shared = set(spec['values']) & set(slot['values'])
            return bool(shared - {''})
        return len(both) >= 2 and same == len(both)

    def item_groups(self) -> list[list[str]]:
        groups = self.ids_by_label()
        wanted = [(_words(item.shelf or ''), _words(item.name), _long_stems(item.shelf or '') | _long_stems(item.name)) for item in self.items]
        matched = [ids for label, ids in groups.items() if self.label_wanted(label, wanted)]
        pool = matched or list(groups.values())
        return sorted(pool, key=len, reverse=True)[:max(1, len(self.items))]

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
        if isinstance(observation, list):
            return self.listed_records(observation)
        if not isinstance(observation, dict):
            return []
        ident = self.ident(observation)
        if ident:
            return [(ident, observation)]
        found: list[tuple[str, dict[str, Any]]] = []
        for value in observation.values():
            if isinstance(value, list):
                found += self.listed_records(value)
            elif isinstance(value, dict):
                for inner in value.values():
                    if isinstance(inner, list):
                        found += self.listed_records(inner)
        return found

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
        whole = self.whole_pages(pages)
        if not whole and pages > 2 and (self.pages_read + 2 <= AUTO_PAGES) and (self.left() - self.tail() - 2 >= 12):
            pages = 2
        if not whole and (pages > 2 or self.pages_read + pages > AUTO_PAGES or self.left() - self.tail() - pages < 12):
            return f"\n[Zenith] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with {param}={_dump(value)}."
        texts = []
        for _ in range(pages):
            faults: list[str] = []
            call = _shape({**args, param: value}, self.schemas.get(name), 'args', faults)
            if faults:
                break
            reply = self.act(name, call)
            if reply is None or self.finished:
                break
            if whole:
                self.whole_read += 1
            else:
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
        rest = f"\n[Zenith] {count} listings matched and Zenith has read {shown * (len(texts) + 1)} rows of them; narrow the search with the requirement's value or page on to see the rest." if value is not None and texts else ''
        if whole and value is None and (len(texts) == pages):
            rest = f'\n[Zenith] These are all {count} listings this search matched.'
        return '\n[Zenith read the next pages of this search itself, one step a page]\n' + '\n'.join(texts) + rest if texts else ''

    def whole_pages(self, pages: int) -> bool:
        """Whether a search for a pick that counts every fitting listing (cheapest, most, least) is read to its
        end: the rest fits in WHOLE_PAGES further pages, the task's whole reads stay within WHOLE_TOTAL pages, and
        the steps left after the read keep WHOLE_ROOM plus four a requested item. These reads do not count against
        AUTO_PAGES, so the two-page reads of later searches keep their own allowance."""
        return pages <= WHOLE_PAGES and self.whole_read + pages <= WHOLE_TOTAL and (self.left() - self.tail() - pages >= WHOLE_ROOM + 4 * len(self.items))

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
        return f'\n[Zenith] This search did not apply {caps[0]}={cap:g}: {len(over)} of its rows cost more{hint}.'

    def cheapest_here(self, observations: list[Any]) -> str:
        if self.brief.get('pick') != 'cheapest' or not self.price_key:
            return ''
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ''
        return '\n[Zenith] buyable rows here, cheapest first: ' + ', '.join((f'{i} ({p:g})' for p, i in sorted(rows)[:6]))

    def wider_hold(self) -> str:
        if self.wider_held or self.brief.get('pick') != 'cheapest' or self.left() < 14 + 2 * len(self.items):
            return ''
        self.wider_held = True
        lines = [f'{i} at {self.price_of(i):g}' for i in self.cart_ids() if self.price_of(i) is not None]
        return f"Not run yet (no step used). {self.left()} steps are left; the pick rule is cheapest. A search may match against only part of a listing, so searching a requirement's value surfaces just some fitting listings, and rows arrive unsorted by price. Before checkout, per item, run a few more searches under alternative product names (the bare one-word product name, synonyms, terms taken from fitting titles seen so far); consider only buyable rows priced under the cart's line" + (f" ({'; '.join(lines)})" if lines else '') + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'

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
        """Search each part code a cart line refers the reader to (one search per code) and, steps permitting, open
        the page found. Output goes to the ledger, not to the model. Multi-line carts only."""
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
        """Listing tool with a shelf argument and a price-cap argument, plus the mode it requires: the mode an earlier
        shelf listing used, else the single option that names a listing. None if tool or mode is unclear."""
        for tool, schema in self.schemas.items():
            props = (schema or {}).get('properties') or {}
            caps = [k for k, v in props.items() if isinstance(v, dict) and {'integer', 'number'} & set(_kinds(v)) and re.search(CAP_ARG, _snake(k), re.I) and (not re.search(CAP_NOT, k, re.I))]
            shelves = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v) and re.search(SHELF_ARG, _snake(k), re.I)]
            if len(caps) != 1 or len(shelves) != 1:
                continue
            base: dict[str, Any] = {}
            for key, spec in props.items():
                options = spec.get('enum') if isinstance(spec, dict) else None
                if not isinstance(options, list) or len(options) < 2:
                    continue
                pick = next((a[key] for a in reversed(self.searched) if shelves[0] in a and key in a and (not any((k != shelves[0] and (not (isinstance(props.get(k), dict) and 'enum' in props[k])) and isinstance(v, str) and v.strip() and re.search(QUERY_WORDS, _snake(k), re.I) for k, v in a.items())))), None)
                named = [o for o in options if isinstance(o, str) and re.search(LIST_MODE, o, re.I)]
                pick = pick if pick is not None else named[0] if len(named) == 1 else None
                if pick is None:
                    return None
                base[key] = pick
            return (tool, base, shelves[0], caps[0])
        return None

    def list_below(self, cart: list[str]) -> None:
        plan = self.cap_template()
        cat_key = self.category_key()
        if not plan or not cat_key:
            return
        tool, base, shelf, cap = plan
        for line in cart:
            price = self.price_of(line)
            label = str((self.rows.get(line) or {}).get(cat_key) or '')
            if price is None or price < 2 or (not label) or ((line, price) in self.capped):
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
            if page and page[3] is not None and (page[1] < page[0] <= 2 * page[1]) and (self.left() > self.tail() + 5) and (not self.price_sorted(reply.get('observation'))):
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
            pool.sort(key=lambda i: self.price_of(i) or 0.0)
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
        schema = self.schemas.get(self.roles.get('ask') or '') or {}
        props = schema.get('properties') or {}
        required = schema.get('required') or []
        target = _message_target(props, required)
        if not target:
            return None
        args: dict[str, Any] = {target: text}
        for key in required:
            if key in args:
                continue
            options = (props.get(key) or {}).get('enum')
            if not isinstance(options, list) or not options:
                return None
            args[key] = options[0]
        faults: list[str] = []
        args = _shape(args, schema, 'args', faults)
        return None if faults else args

    def interview(self) -> str:
        self.asked = True
        if self.msgs_free() <= 0:
            return ''
        self.widen()
        room = self.msg_limit - self.msg_used
        self.start_lanes(max(room, LIST_SIZE) if self.listing else room)
        self.follow_up = []
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        while self.interview_room():
            batch = self.ask_next(told)
            if not batch:
                break
            sent += batch
        found = [n for n, hits in self.lane_found.items() if hits > 0]
        searched = self.value_search(found) if found else ''
        if told:
            self.version += 1
        summary = self.interview_summary(sent, told)
        if searched:
            summary += '\n' + searched
        if self.queued:
            summary += '\n[notes seen meanwhile] ' + ' | '.join(self.queued)
            self.queued = []
        return summary

    def late_question(self) -> str:
        """Pre-checkout: put a message the order does not need toward the next planned question.

        The interview skips every message the request reserves for a process step, be it a step that comes with
        the order or one that comes with an ending without a purchase. Once Zenith is about to buy, messages
        reserved for the ending are spare. In race 13 (two messages allowed) the base sent a single question in 28 of
        180 episodes; a hidden need went undisclosed in 12 of them, and no episode with a missed need passed. Returns a
        summary for the loop model if the answer adds a requirement, '' otherwise."""
        if not self.may_ask_late():
            return ''
        self.late_asked = True
        if not self.lanes:
            self.start_lanes(LIST_SIZE if self.listing else 1)
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
        summary = f'Zenith asked the shopper one more question before checking out (one step):\nQ: {question}\nA: {answer}\nNew firm requirements from this answer: ' + '; '.join(told) + '. The cart was not checked out: check each line against them (search again with the new values where needed), fix the cart, then check out.' + self.pointed_note(told, '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value.')
        searched = self.value_search([number])
        return summary + ('\n' + searched if searched else '')

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
                return (tool, keep, field)
        return None

    def value_words(self, number: int) -> list[tuple[str, bool]]:
        """For one item: search words for every requirement value a words search can match, each flagged with
        whether an answer supplied it. That is the value itself; for an answer-given minimum with a unit (a numeric
        floor), only its number and unit, because such a hidden limit often equals the value of a fitting listing."""
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
        """Template for a value search to copy: search_template(); or, if all searches so far listed a category with no words,
        the tool's lone query field plus its lone mode argument, switched to reading it (the option that names words or
        a query; else, of a required two-option mode, the option the last search did not use). None if the schema leaves
        any doubt."""
        plan = self.search_template()
        tool = self.roles.get('search')
        if plan or not tool or (not self.searched):
            return plan
        schema = self.schemas.get(tool) or {}
        props = schema.get('properties') or {}
        required = schema.get('required') or []
        named = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v) and re.search(QUERY_WORDS, _snake(k), re.I)]
        last = self.searched[-1]
        enums = {k: v['enum'] for k, v in props.items() if isinstance(v, dict) and isinstance(v.get('enum'), list) and (len(v['enum']) >= 2)}
        modes = []
        for key, options in enums.items():
            reads = [o for o in options if isinstance(o, str) and re.search(QUERY_WORDS, o, re.I)]
            others = [o for o in options if o != last.get(key)]
            if len(reads) == 1:
                modes.append((key, reads[0]))
            elif key in required and len(options) == 2 and (len(others) == 1):
                modes.append((key, others[0]))
        if len(named) != 1 or len(modes) > 1 or (enums and (not modes)):
            return None
        base = {k: last[k] for k in required if k in last and k != named[0]}
        base.update(modes)
        return (tool, base, named[0])

    def value_search(self, numbers: list[int]) -> str:
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
        return "[Zenith searched each such item's requirement values together and read the first fitting-kind rows] " + '\n'.join(seen)

    def home_label(self, number: int) -> str:
        """Category label (in _words form) carried by the item's own listings. It is the request's shelf name if a seen
        listing carries it; else the majority label among the item's opened listings (among its search rows if none
        is opened); '' when unknown."""
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
        first = {n: i for i, n in enumerate(self.plan_order)}
        for number in sorted(lanes, key=lambda n: (found.get(n, 0) > 0, spent.get(n, 0), first.get(n, len(first) + n))):
            left = [label for label in lanes[number] if _words(label) not in used[number] and (not self.told_about(number, label))]
            if left:
                return (number, left[:LIST_SIZE if self.listing else 1])
        return (0, [])

    def told_about(self, number: int, label: str) -> bool:
        """True if some answer has already supplied a need for this item property: the told property's name contains the
        label's words ("colour" after "shade colour"), or the label contains a multi-word told name; colour/color and
        singular/plural count alike. A one-word told name ("type") never hides a longer label ("bulb type")."""
        mine = _spelled(label)
        return any((_names(mine, _spelled(t.split(':', 1)[0])) for t in self.items[number - 1].told))

    def ask_text(self, number: int, labels: list[str], shown: bool=False) -> str:
        if len(labels) >= 2:
            return f"What {', '.join(labels[:-1])} or {labels[-1]} would you like for the {self.item_said(number)}?"
        return self.question(number, labels[0], self.samples(number, labels[0]) if shown else None)

    def fit_ask(self, number: int, labels: list[str], shown: bool) -> tuple[list[str], str]:
        """Longest prefix of `labels` whose question survives the message tool's schema whole, returned with that question.
        The schema shaper truncates text past a length cap, so an over-long list sheds labels from its end instead.
        ([], '') if no single label fits."""
        for size in range(len(labels), 0, -1):
            text = self.ask_text(number, labels[:size], shown)
            args = self.message_args(text)
            if args and text in args.values():
                return (labels[:size], text)
        return ([], '')

    def ask_next(self, told: list[str], floor: int=6) -> list[tuple[int, str, str]]:
        """Ask the next question (any pending follow-up goes before the next planned labels). Fold the answer into the
        items' needs (each new need is appended to `told`) and settle the labels the question named. Returns the batch
        sent; [] if nothing went out."""
        question = self.fitted_question()
        if question is None:
            return []
        number, labels, shown, text = question
        batch = self.send([(number, labels[0], text)], floor)
        if not batch:
            return []
        self.lane_spent[number] = self.lane_spent.get(number, 0) + 1
        new, crowded = self.take_answer(batch, labels, told)
        self.settle(number, labels, shown, new, crowded)
        return batch

    def settle(self, number: int, labels: list[str], shown: bool, new: list[str], crowded: bool) -> None:
        """Record what an answer revealed about the labels its question named.

        Single label: mark it asked. A reply that wants a narrower question triggers the base's follow-up: a single
        product out of a multi-product item name, the property's head word, or sample values. List: the shopper answers
        on the first property in a message that it tracks, and a message that names none of its properties draws the
        same reply as an indifferent property. Hence a disclosed need settles its own label and every earlier one, and
        the later labels head the item's next list. A reply disclosing nothing settles the first label only (had there
        been a need, it would have come out); the rest come next in reverse order, so the properties the shopper does
        not mind stop standing in front. Two such replies in a row put the still-unnamed labels in front, since
        otherwise the order would revert to that of the first list. A one-question-at-a-time request turns lists off
        for the rest of the episode; the list's first label then goes out alone."""
        used = self.lane_used.setdefault(number, set())
        pushback = crowded and (not new)
        if len(labels) == 1:
            used.add(_words(labels[0]))
            if pushback:
                self.narrow_follow_up(number, labels, shown, used)
            return
        if pushback:
            self.listing = False
            self.trace.append({'kind': 'lists_off', 'item': number})
            self.follow_up.insert(0, (number, labels[:1], False))
            return
        self.settle_list(number, labels, new, used)

    def answered_label(self, labels: list[str], new: list[str]) -> int:
        """Index of the first label a list reply answered. A new need maps to the label with the most words in common with
        its property (colour equals color); else to the label whose opened-listing values overlap most with its value.
        -1 if no new need maps to any label; the reply then counts as one that disclosed nothing."""
        seen = self.label_roots()
        hits = [_answer_slot(labels, need, seen) for need in new]
        return min((hit for hit in hits if hit is not None), default=-1)

    def lane_keys(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        settled = [_stem(must.split(':', 1)[0]) for must in self.items[number - 1].musts]
        rows = [row for row in table if self.owner_of(row[0]) == number]
        amounts = {key: _lane_amounts(values) for _, key, _, values in table}
        if not rows and len(self.items) == 1:
            rows = list(table)
        rows.sort(key=lambda row: (-row[2], amounts[row[1]], -len(row[3])))
        keys = [key for _, key, _, _ in rows if not any((_names(prop, _stem(key)) for prop in settled)) and (not re.search(NOT_PROPERTY, key))]
        return sorted(keys, key=_lane_broad)

    def adopt_labels(self) -> None:
        cat_key = self.category_key()
        if len(self.items) < 2 or not cat_key:
            return
        labels = {_shelf_label(record, cat_key) for record in self.full.values()} - {''}
        strays = [label for label in labels if not self.owner_of(label)]
        orphans = self.unlabelled_items(labels)
        if not strays or len(orphans) != 1:
            return
        for label in strays:
            self.adopted[label] = orphans[0]
        self.trace.append({'kind': 'adopted', 'item': orphans[0], 'labels': strays})

    def owner_of(self, label: str) -> int:
        if not label:
            return 0
        if label in self.adopted:
            return self.adopted[label]
        words, long = (_words(label), _roots(label))
        best = _top_index([_shelf_fit(words, long, item) for item in self.items])
        return 0 if best is None else best + 1

    def owns(self, item: Item, label: str) -> bool:
        return bool(label) and item in self.items and (self.owner_of(label) == self.items.index(item) + 1)

    def records_of_item(self, number: int) -> int:
        cat_key = self.category_key()
        item = self.items[number - 1]
        return sum((1 for record in self.full.values() if cat_key and self.owns(item, str(record.get(cat_key) or '')) and _fact_block(record)))

    def plan_lanes(self, count: int) -> dict[int, list[str]]:
        self.adopt_labels()
        table = self.key_table()
        plan = self.ask_json(self.plan_prompt(count, table), [HEAVY, LIGHT], 3500) or {}
        picks, hinted = self.plan_picks(plan.get('ask'))
        lanes = {number: self.plan_lane(number, table, labels, hinted[number]) for number, labels in picks.items()}
        self.trace.append({'kind': 'plan', 'ask': _clip(plan.get('ask'), 120), 'lanes': {n: v[:8] for n, v in lanes.items()}})
        return lanes

    def name_parts(self, number: int) -> list[str]:
        """Products an item name joins by "and" or commas, provided each is a plural noun ("door handles and locks");
        otherwise []. A single product whose name contains "and" gives [] ("salt and pepper grinder",
        "usb and lightning cables", "his and hers towels")."""
        name = self.items[number - 1].name
        if not re.search('\\band\\b', name, re.I):
            return []
        parts = []
        for piece in re.split('\\s*(?:,|\\band\\b)\\s*', name, flags=re.I):
            piece = piece.strip()
            if piece:
                parts.append(piece)
        if len(parts) < 2:
            return []
        plural = all((re.search('[a-z]{3}(?<![su])s$', part, re.I) for part in parts))
        return parts if plural else []

    def item_said(self, number: int) -> str:
        """How a question names the item. Plural nouns joined by "and" become joined by "or" in the question (race 13,
        10 agents: "door handles and locks" drew one-at-a-time replies to 81 of 155 questions, while names joined by
        "or" drew them for 0 of 50, and "dresser or cabinet" revealed a need 10 times in 28). Following a
        one-at-a-time reply on the item: just one joined product, the one whose head noun is named outside the
        category label by the largest number of the item's own rows (the first, if there are none)."""
        item = self.items[number - 1]
        parts = self.name_parts(number)
        if not parts:
            return item.name
        if number not in self.one_product:
            return re.sub('\\band\\b', 'or', item.name, flags=re.I)
        return self.likeliest_part(item, parts)

    def question(self, number: int, label: str, samples: list[str] | None=None) -> str:
        text = f'What {label} would you like for the {self.item_said(number)}'
        return text + (f', for example {samples[0]} or {samples[1]}?' if samples else '?')

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
            reply = self.act(tool, args) if args else None
            if reply is None:
                break
            self.msg_used += 1
            answer = reply.get('observation')
            words = self.spoken(answer)
            sent.append((number, text, words))
            self.record_message(text, words)
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
            self.budget = self.rescaled(heard['budget'])

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(HEAVY, LIGHT)) -> dict[str, Any]:
        prompt = HEAR + '\n\nITEMS:\n' + '\n'.join((f'item {n}: {it.name}' for n, it in enumerate(self.items, 1)))
        if isinstance(self.budget, (int, float)) and (not isinstance(self.budget, bool)):
            prompt += f"\n\nBUDGET IN FORCE: {self.budget:g} {self.brief.get('budget_unit') or ''}".rstrip()
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f'{i}. (about item {n}) Q: {q}\nA: {a}' for i, (n, q, a) in enumerate(sent, 1)))
        return self.ask_json(prompt, list(models), 2500) or {}

    def gate(self, stage: str) -> tuple[bool, str]:
        steps = [p for p in self.process if p['when'] == stage]
        if not steps or self.finished or (not self.gate_due(stage)):
            return (True, '')
        ending = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        menu = self.gate_menu(ending)
        head = self.gate_intro(stage, steps)
        said: list[str] = []
        nudged = False
        for _ in range(GATE_ROUNDS):
            got = self.ask_json(head + self.gate_record(stage) + menu, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 3500)
            if got is None:
                break
            told, hold = self.gate_heard(got) if said else ([], '')
            call = got.get('call') if isinstance(got.get('call'), dict) else None
            self.trace.append({'kind': 'gate', 'stage': stage, 'call': _clip(call, 200), 'steps': _clip(got.get('steps'), 120), 'hold': hold[:200], 'told': told})
            if told or hold:
                self.version += 1
                changes = told + [hold] if hold else told
                return (False, self.gate_report(said) + '\n' + "The shopper's reply changes the task: " + '; '.join(changes) + '. Fix the cart for it, then check out again.')
            unproven = self.unproven_steps(stage, steps, got)
            if not call:
                if unproven and (not said) and (not nudged) and self.roles.get('ask') and (self.msg_used < self.msg_limit) and (self.left() >= 2):
                    nudged = True
                    self.gate_nudge(unproven)
                    continue
                self.gate_ok.add((stage, self.cart_mark() + f'#{self.version}'))
                break
            name = str(call.get('tool') or '')
            asking = name == self.roles.get('ask')
            if name not in self.schemas or name in ending or self.left() < 2 or (asking and self.msg_used >= self.msg_limit):
                break
            faults: list[str] = []
            args = _shape(call.get('args') if isinstance(call.get('args'), dict) else {}, self.schemas[name], 'args', faults)
            if faults:
                break
            line = self.gate_message(args) if asking else self.gate_store_call(name, args)
            if line is None:
                break
            said.append(line)
        return (True, self.gate_report(said))

    def spoken(self, observation: Any) -> str:
        if isinstance(observation, str):
            return observation
        if isinstance(observation, dict):
            answers = [v for v in observation.values() if isinstance(v, str) and len(v.split()) >= 2]
            asides = [s for v in observation.values() if isinstance(v, list) for s in v if isinstance(s, str) and len(s.split()) >= 2]
            if len(answers) == 1 and asides:
                return answers[0] + ' [said besides the answer: ' + ' | '.join(asides) + ']'
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
        if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0) and (self.rescaled(budget) != self.budget):
            self.budget = self.rescaled(budget)
            changed.append(f'budget now {self.budget:g}')
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

    def rescaled(self, amount: float) -> float:
        """A new budget the shopper gave, in the unit of the request's budget. Readers sometimes turn '22068 cents'
        into 220.68 or the reverse, so the amount is taken at the first scale (x1, x100, /100) that leaves it within a
        factor of RESCALE of the budget in force; an amount no scale brings that close is taken as said."""
        old = self.budget
        if not isinstance(old, (int, float)) or isinstance(old, bool) or old <= 0:
            return amount
        for scale in (1, 100, 0.01):
            if old / RESCALE <= amount * scale <= old * RESCALE:
                return amount * scale
        return amount

    def needs_now(self) -> list[tuple[int, str]]:
        out = []
        for number, item in enumerate(self.items, 1):
            for need in item.musts + item.told:
                if need not in item.dropped:
                    out.append((number, need))
        return out

    def evidence(self, observation: Any) -> str:
        ids: list[str] = []
        self.evidence_ids(observation, 0, ids)
        needs = self.needs_now()
        if not ids or not needs:
            return ''
        many = len(self.items) > 1
        lines = []
        for ident in ids[:12]:
            line = self.evidence_line(ident, needs[:8], many)
            if line:
                lines.append(line)
        if not lines:
            return ''
        return "[evidence: keys whose names contain a requirement's property; one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property] " + ' | '.join(lines)

    def cart_text(self) -> str:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            return _dump(_clip(self.cart_view, 300))[:3000]
        return "lines by Zenith's record of the cart calls (listing: units): " + _dump(self.cart_lines)

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
        for ident in self.others_order():
            if ident in skip or self.sold_out(ident):
                continue
            record = self.full.get(ident) or self.rows.get(ident)
            lines.append(_dump(_clip(record, 160))[:900])
            if len(lines) >= limit:
                break
        return '\n'.join(lines)

    def fit_index(self) -> dict[str, Any]:
        marks: dict[str, dict[str, Any]] = {}
        for ident in self.seen_order:
            record = self.full.get(ident) or self.rows.get(ident) or {}
            marks[ident] = _fit_marks(_fit_text(record))
        notes: dict[str, set[str]] = {}
        for mark in marks.values():
            for part, codes in mark['also']:
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
        parts_of: dict[int, list[int]] = {}
        for part, main in self.bound:
            parts_of.setdefault(main, []).append(part)
        lines = []
        for main in sorted(parts_of):
            line = self.fit_plan_line(main, parts_of[main], index)
            if line:
                lines.append(line)
        if not lines:
            return ''
        plan = ' | '.join(lines)
        return plan if short else "[fit plan: a part fits a device if the part's text names that device's model code or any other name the device is sold under; search cannot find codes, so browse the device's category and take the codes from its rows] " + plan

    def missing_items(self) -> list[int]:
        """Items lacking a cart line while the cart has fewer lines than items; skips items no seen listing is labelled with."""
        cat_key = self.category_key()
        ids = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        if not cat_key or len(ids) >= len(self.items):
            return []
        seen = [_shelf_label(row, cat_key) for row in self.rows.values()]
        held = [_shelf_label(self.rows.get(i), cat_key) for i in ids]
        missing = []
        for number, item in enumerate(self.items, 1):
            if not any((self.owns(item, label) for label in seen)):
                continue
            if any((self.owns(item, label) for label in held)):
                continue
            missing.append(number)
        return missing

    def states_needs(self, ident: str, number: int) -> bool:
        """True if the listing's attributes state every requirement of the item verbatim. Limits and exclusions are not
        parsed here; they count as unstated."""
        block = _fact_block(self.full.get(ident) or {}) or {}
        return all((_states_need(block, need) for owner, need in self.needs_now() if owner == number))

    def fill_short_cart(self) -> bool:
        """Forced-order prep: for every item lacking a line, add the cheapest opened listing whose attributes state its
        requirements. Returns False if any item remains uncovered; the episode then ends in a refusal, not a short
        order."""
        cart = self.roles.get('cart') or ''
        bound = {n for pair in self.bound for n in pair}
        for number in self.missing_items():
            item = self.items[number - 1]
            fits = [i for i in self.item_ids(number) if i in self.full and i not in self.cart_lines and self.buyable(i) and (self.price_of(i) is not None) and self.states_needs(i, number)]
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
        cat_key = self.category_key()
        mine = _roots(self.items[number - 1].name)
        if len(mine) < 2 or mine <= _roots(label) or (not cat_key) or (self.items[number - 1].shelf and _roots(label) == _roots(self.items[number - 1].shelf)):
            return ''
        return next((seen for seen in sorted({str(r.get(cat_key) or '') for r in self.rows.values()}) if seen and mine <= _roots(seen)), '')

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
        judges = self.judges(limit)
        got = None
        if len(judges) > 1 and (not lead) and (not self.audited) and (self.brief.get('pick') not in ('cheapest', 'most', 'least')):
            got = self.ask_json(prompt, [LIGHT], 3000)
            if got:
                judges = [HEAVY] if got.get('ok') is False and HEAVY in self.judges(limit) else []
        if judges and (not self.no_credit):
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
            text += '[Zenith opened the pages a requirement points at]\n' + '\n'.join(seen) + '\n'
        if self.ref_values:
            text += '[values read from the pages requirements point at] ' + '; '.join((f'{k} -> {v}' for k, v in self.ref_values.items())) + '\n'
        hold = bool(lost) and self.trail_blocks < (3 if self.left() > 15 else 2) and (self.left() > 5)
        if hold:
            self.trail_blocks += 1
            text += 'The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: ' + ' | '.join(lost[:3]) + ". Find it by searching its shop's name plus the values its description lists (other same-kind listings of that shop are not it); open that shop's row whose page carries those values, apply the value read there, then check out."
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
        text = f"[Zenith] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
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
        if self.audited or not self.judges():
            return ''
        self.audited = True
        got = self.audit_call()
        named = self.audit_fits(got)
        if not named and self.audit_opens(got):
            got = self.audit_call()
            named = self.audit_fits(got)
        if not named:
            return ''
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return self.audit_message(named, got.get('reason'))

    def audit_call(self) -> dict[str, Any] | None:
        records = self.audit_records()
        if not records:
            return None
        prompt = self.audit_prompt(records)
        judges = self.judges()
        got = self.ask_json(prompt, [LIGHT] if judges else [], 4000)
        if HEAVY in judges and (not self.no_credit) and (not self.audit_fits(got)):
            got = self.ask_json(prompt, [HEAVY], 4000) or got
        self.trace.append({'kind': 'audit', 'got': got})
        return got

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get('fits')
        if not isinstance(fits, dict):
            return {}
        named = {str(item): self.audit_ids(ids)[:4] for item, ids in fits.items()}
        return {item: ids for item, ids in named.items() if ids}

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
            index = int(found.group()) if found else 1 if len(self.items) == 1 else None
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
        return 'Not run (Zenith acted instead): the independent check confirmed ' + self.confirmed + ', so Zenith set the cart (' + '; '.join(done) + ') and checked out. Store reply: ' + ' | then: '.join(replies)

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

    def whole_removal(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """A count-less removal of a held line gets the held units as its count; the store would otherwise remove just one."""
        spec = self.schemas.get(name, {}).get('properties') or {}
        count = next((k for k, v in spec.items() if isinstance(v, dict) and 'integer' in _kinds(v)), None)
        removal = any((isinstance((spec.get(k) or {}).get('enum'), list) and v in spec[k]['enum'][1:] for k, v in args.items()))
        ident = next((v for v in args.values() if isinstance(v, str) and v in self.cart_lines), None)
        return {**args, count: min(self.cart_lines[ident], spec[count].get('maximum', self.cart_lines[ident]))} if count and removal and ident and (count not in args) else args

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
                out.append((cid, 'Shown in the side-by-side result above: Zenith opened these listings together in one step.'))
                continue
            fn = call.get('function') or {}
            name = fn.get('name')
            raw = fn.get('arguments')
            if self.finished:
                out.append((cid, 'Not run: the episode is over.'))
                continue
            if name == self.roles.get('ask'):
                out.append((cid, "Not run (no step used): Zenith's interviewer handles every message to the shopper."))
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
                trimmed = f"\n[Zenith] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
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
                    trimmed += f"\n[Zenith] Left out {', '.join((str(i) for i in args[many] if i not in keep))}: shown in full already, nothing changed since."
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
                            out.append((cid, "Not run yet (no step used). Zenith set the cart's quantities itself: " + '; '.join(fixed) + '. If the store refused any of these, fix those lines; then check out.\nCart now: ' + self.cart_text()))
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
                        seen = "Zenith read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n' if read else ''
                        out.append((cid, 'Not run yet. ' + chain + seen + verdict + '\nCart now: ' + self.cart_text()))
                        continue
                    if reread and self.cost > SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). Zenith re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'))
                        continue
                elif self.left() > 5:
                    review = self.audit()
                    if review:
                        out.append((cid, 'Not run yet (no step used). ' + review))
                        continue
                    if self.leads and self.left() > 4 and (self.refuse_blocks == 0):
                        self.refuse_blocks = 1
                        self.trace.append({'kind': 'refuse_blocked', 'leads': list(self.leads)})
                        out.append((cid, 'Not run yet (no step used). The review found listings that appear to fit: ' + ', '.join(self.leads) + '. Open them as needed, cart the best-fitting one (the pick rule applies) and check out. Ending without a purchase is allowed only if each of them clearly breaks a requirement; in that case name the failing requirement in your text and resend the ending call.'))
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
                out.append((cid, f"Not run yet (no step used). {self.left()} steps are still left. Before ending, search in untried ways: browse each requested item's category and page through it with the store's returned token or offset; try one-word value queries and other names for the property; open the listings that could fit. Only after that may you end without a purchase."))
                continue
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
            if ending and self.left() >= 2:
                go, said = self.gate('order' if self.is_buy(name, args) else 'end')
                if not go:
                    out.append((cid, 'Not run yet. ' + said + '\nCart now: ' + self.cart_text()))
                    continue
                if said:
                    trimmed = '\n[Zenith] ' + said + trimmed
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, 'The call failed (network or server). You may try again.'))
                continue
            observation = reply.get('observation')
            if many and isinstance(args.get(many), list) and (len(args[many]) > 1):
                shown = _dump(observation)
                if not any((str(i) in shown for i in args[many])):
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    trimmed += f'\n[Zenith] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_room()} per call.'
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
                    text += '\n[Zenith] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'
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
                    text += "\n[Zenith re-read the cart's lines after the refusal]\n" + seen
            self.note_sold = False
            swap = self.swap_hint()
            if swap:
                text += '\n' + swap
            if self.queued:
                text += "\n[notes seen on Zenith's own calls] " + ' | '.join(self.queued)
                self.queued = []
            if name in readers:
                self.read_cache[mark] = (self.version, self.step, base)
            if reply.get('finished'):
                text += ' [episode finished]'
            out.append((cid, text))
        return out

    def can_refuse(self) -> bool:
        refuse = self.roles.get('refuse')
        if not refuse:
            return False
        faults: list[str] = []
        _shape(self.quiet(refuse, dict(self.brief.get('refuse_args') or {})), self.schemas.get(refuse), 'args', faults)
        return not faults

    def finish(self, again: bool=False) -> str:
        if self.broke or self.out_of_play():
            return ''
        buy, refuse = (self.roles.get('buy'), self.roles.get('refuse'))
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids())
        whole = True
        if holding and len(self.items) >= 2:
            whole = self.whole_cart_ready()
        if self.out_of_play():
            return ''
        to_order = holding or (self.cart_touched and (not fresh) and (self.left() >= 2))
        if whole and buy and to_order:
            fits, reply = self.closing_call(buy, dict(self.brief.get('buy_args') or {}), 'order', 'forced_buy')
            if fits and self.out_of_play():
                return ''
            if reply is not None and again:
                self.buys += 1
                self.refused_at = self.step
                return self.refusal_text(buy, reply, 'the checkout was refused: ')
        if not refuse:
            return ''
        _, reply = self.closing_call(refuse, self.quiet(refuse, dict(self.brief.get('refuse_args') or {})), 'end', 'forced_refuse')
        if reply is None or self.finished:
            return ''
        return self.refusal_text(refuse, reply, 'ending without a purchase was refused: ')

    def trim(self) -> None:
        """Shorten the older half of the tool results once the history passes HISTORY_LIMIT, and again only after
        it has grown by HISTORY_LIMIT more, or on every call while it is over HISTORY_CEILING; each result is
        shortened once. Editing an old message makes the next model call re-write the prompt cache from that message
        on, so cutting a little on every call while the history stayed over the limit cost more than it saved, and
        cutting a shortened result again lost its listing ids. The ceiling keeps a world with very large results
        about as short as cutting on every call kept it, well inside the model's context window."""
        if sum((len(_dump(m.get('content'))) for m in self.chat)) <= min(max(HISTORY_LIMIT, self.trim_mark), HISTORY_CEILING):
            return
        results = [m for m in self.chat if m.get('role') == 'tool']
        for message in results[:max(1, len(results) // 2)]:
            content = message.get('content') or ''
            if len(content) > 600 and SHORTENED not in content:
                message['content'] = self.short_result(content)
        self.trim_mark = sum((len(_dump(m.get('content'))) for m in self.chat)) + HISTORY_LIMIT

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.read_request()
        self.trace.append({'kind': 'brief', 'brief': self.brief})
        self.chat = [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': self.opening_message()}]
        self.work(LOOP_CALLS, SPEND_HARD)
        if not self.finished:
            self.close_episode()
        self.trace.append({'kind': 'end', 'finished': self.finished, 'steps': self.step, 'cost': round(self.cost, 5), 'calls': self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while self.loop_may_turn(loop_cap, spend_cap):
            if self.walk_due():
                self.walk_note(self.walk_pointers())
                continue
            if self.ready_to_ask():
                summary = self.interview()
                if summary:
                    self.chat_user(summary + '\n')
                continue
            self.trim()
            self.loop_calls += 1
            message = self.think(self.chat, [LIGHT, SPARE], 3000, self.offered())
            if message is None:
                self.misses += 1
                if self.no_credit or self.calls >= ALL_CALLS or self.misses > 2:
                    return
                continue
            content, calls = _model_turn(message, self.loop_calls)
            if not calls:
                idle += 1
                self.chat.append({'role': 'assistant', 'content': content or '(no call)'})
                self.chat_user('Call one of the tools to continue.\n')
                if idle >= 3:
                    return
                continue
            idle = 0
            self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
            if self.interview_first(calls):
                self.hold_calls(calls)
            else:
                self.play_calls(content, calls)

    def plan_prompt(self, count: int, table: list[tuple[str, str, int, list[str]]]) -> str:
        """The PLAN prompt: the request, what it settles for each item, the properties it asks to be asked about, and
        the attribute keys seen on full listings."""
        settled = [f'item {number}: {item.name} - settled by the request: ' + ('; '.join(item.musts) or 'nothing') for number, item in enumerate(self.items, 1)]
        keys = self.plan_key_lines(table)
        first = [str(x) for x in _items(self.brief.get('ask_first')) if str(x).strip()]
        parts = [PLAN.replace('COUNT', str(max(1, count))), '\n\nREQUEST:\n', self.request, '\n\nITEMS:\n', '\n'.join(settled)]
        if first:
            parts += ['\n\nASK FIRST (the request asks to be asked about these):\n', '\n'.join(first)]
        parts += ['\n\nATTRIBUTE KEYS SEEN:\n', '\n'.join(keys) or '(no full listing records yet)']
        return ''.join(parts)

    def plan_key_lines(self, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        """The lines under ATTRIBUTE KEYS SEEN: a heading for each category label, then each key numbered, with how many
        listings carry it, its values, and marks for a key that is not the stated property or holds several features."""
        lines: list[str] = []
        heading = None
        for index, (label, name, carried, values) in enumerate(table, 1):
            if label != heading:
                heading = label
                lines.append(f'category {label!r}:')
            line = f'  {index}. {name} (on {carried}): ' + ' | '.join(values) + self.not_stated(label, name, values)
            if _lane_broad(name):
                line += ' [several features in one: ask last]'
            lines.append(line)
        return lines

    def plan_picks(self, asks: Any) -> tuple[dict[int, list[str]], dict[int, set[str]]]:
        """The plan's labels for each item number, in plan order and without repeats, and the words of those it gave a
        hint for; an item with a hinted label joins plan_order."""
        lanes: dict[int, list[str]] = {n: [] for n in range(1, len(self.items) + 1)}
        hinted: dict[int, set[str]] = {n: set() for n in lanes}
        for pick in _items(asks):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, _count(pick.get('item') or 1) or 1), len(self.items))
            label = _lane_label(pick.get('label'))
            if not label:
                continue
            _lane_join(lanes[number], label)
            if not str(pick.get('hint') or '').strip():
                continue
            hinted[number].add(_words(label))
            if number not in self.plan_order:
                self.plan_order.append(number)
        return (lanes, hinted)

    def plan_lane(self, number: int, table: list[tuple[str, str, int, list[str]]], picks: list[str], hints: set[str]) -> list[str]:
        """Item `number`'s labels in asking order: the plan's picks in the spelling of the item's own keys, its
        best-carried keys guarded into the first three places (_lane_guard), then its other keys; catch-all keys last."""
        keys = self.lane_keys(number, table)
        own = self.plan_shares(number, table)
        hinted = {_words(_lane_spelling(own, hint)) for hint in hints}
        lane: list[str] = []
        for pick in picks:
            _lane_join(lane, _lane_spelling(own, pick))
        guards = [key for key in keys if own.get(key, 0.0) >= GUARD_SHARE and (not _lane_broad(key))]
        swap = len(self.items) > 1
        for key in guards[:GUARD_KEYS]:
            lane = _lane_guard(lane, key, own, hinted, swap)
        for key in keys:
            _lane_join(lane, key)
        return sorted(lane, key=lambda label: _lane_broad(_words(label)))

    def plan_shares(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> dict[str, float]:
        """Each key of item `number`'s own categories with the highest share of the item's read listings carrying it."""
        read = max(1, self.records_of_item(number))
        own: dict[str, float] = {}
        for label, key, carried, _ in table:
            if self.owner_of(label) == number:
                own[key] = max(own.get(key, 0.0), carried / read)
        return own

    def unlabelled_items(self, labels: set[str]) -> list[int]:
        """The numbers of the items that own none of `labels`; a label adopted earlier counts for no item."""
        owners = {self.owner_of(label) for label in labels if label not in self.adopted}
        return [n for n in range(1, len(self.items) + 1) if n not in owners]

    def label_roots(self) -> dict[str, set[str]]:
        """Each key of the key table (in words) with the root words of every value it shows."""
        seen: dict[str, set[str]] = {}
        for _, key, _, values in self.key_table():
            roots = seen.setdefault(_words(key), set())
            for value in values:
                roots.update(_roots(value))
        return seen

    def narrow_follow_up(self, number: int, labels: list[str], shown: bool, used: set[str]) -> None:
        """Queue the follow-up to a one-label question whose reply asked for a narrower one: the item asked as one of the
        products its name joins (once per item), else the label's head word alone (unless asked already), else the
        label with example values (when none were shown and two are seen)."""
        if number not in self.one_product and self.name_parts(number):
            self.one_product.add(number)
            self.trace.append({'kind': 'one_product', 'item': number, 'said': self.item_said(number)})
            self.follow_up.insert(0, (number, labels, shown))
            return
        words = labels[0].split()
        head = words[-1]
        if len(words) >= 2 and _words(head) not in used:
            self.follow_up.insert(0, (number, [head], False))
        elif not shown and len(self.samples(number, labels[0])) >= 2:
            self.follow_up.insert(0, (number, labels, True))

    def settle_list(self, number: int, labels: list[str], new: list[str], used: set[str]) -> None:
        """Settle a list's reply by the rules in settle: mark the settled labels used, count the replies in a row that
        disclosed nothing, and put the labels still ahead before the item's other labels (behind them after every
        second such reply)."""
        hit = self.answered_label(labels, new) if new else -1
        if hit >= 0:
            settled, ahead = (labels[:hit + 1], labels[hit + 1:])
            self.lane_misses[number] = 0
        else:
            settled, ahead = (labels[:1], labels[:0:-1])
            self.lane_misses[number] = self.lane_misses.get(number, 0) + 1
        used.update((_words(label) for label in settled))
        misses = self.lane_misses[number]
        rest = [label for label in self.lanes.get(number) or [] if label not in labels]
        if misses and misses % 2 == 0:
            self.lanes[number] = rest + ahead
        else:
            self.lanes[number] = ahead + rest
        self.trace.append({'kind': 'list_settled', 'item': number, 'done': labels[:hit + 1], 'next': ahead[:4]})

    def gate_intro(self, stage: str, steps: list[dict[str, Any]]) -> str:
        """The fixed head of the gate prompt: the instructions for this stage, the request and the steps due now."""
        goal = 'place the order' if stage == 'order' else 'end without a purchase'
        return GATE.replace('STAGE', goal) + '\n\nREQUEST:\n' + self.request + '\n\nSTEPS DUE NOW:\n' + '\n'.join((f"- {p['step']}" for p in steps))

    def gate_record(self, stage: str) -> str:
        """The part of the gate prompt that changes from round to round: the ledger, the cart, the record of messages
        and step calls, the latest store notes, and the messages and steps left."""
        sections = ['\n\nLEDGER:\n' + self.ledger(), '\n\nCART NOW: ' + self.cart_text()]
        record = '\n'.join((f'{i}. {line}' for i, line in enumerate(self.done_log, 1)))
        sections.append('\n\nRECORD (messages and step calls so far):\n' + (record or '(nothing yet)'))
        if self.notes:
            sections.append('\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.notes[-4:]))
        final = 'order' if stage == 'order' else 'ending'
        sections.append(f'\n\nMESSAGES LEFT: {max(0, self.msg_limit - self.msg_used)}   STEPS LEFT: {self.left()} (the {final} itself takes one)')
        return ''.join(sections)

    def gate_menu(self, ending: set[str]) -> str:
        """The TOOLS section of the gate prompt: every tool but the ending ones, with its description and parameters."""
        lines = []
        for tool in self.tools:
            fn = tool['function']
            if fn['name'] in ending:
                continue
            lines.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {_dump(fn.get('parameters') or {})}")
        return '\n\nTOOLS:\n' + '\n'.join(lines)

    def gate_heard(self, got: dict[str, Any]) -> tuple[list[str], str]:
        """Take the needs the gate model read in the shopper's replies into the items. Returns them as told, with the
        hold the model gave."""
        told: list[str] = []
        self.take_heard({'needs': got.get('needs')}, told)
        return (told, str(got.get('hold') or '').strip())

    def unproven_steps(self, stage: str, steps: list[dict[str, Any]], got: dict[str, Any]) -> list[dict[str, Any]]:
        """The steps the gate model lists as not done, or as done without citing a record line that proves it. When an
        order step needs messages, only lines made after the cart last changed are proof."""
        proof = set(range(1, len(self.done_log) + 1))
        if stage == 'order' and any((p['messages'] > 0 for p in steps)):
            proof = {i for i in proof if i <= len(self.done_steps) and self.done_steps[i - 1] > self.cart_at}
        unproven = []
        for entry in _items(got.get('steps')):
            if not isinstance(entry, dict):
                continue
            proven = entry.get('done') is True and any(((_count(n) or 0) in proof for n in _items(entry.get('lines'))))
            if not proven:
                unproven.append(entry)
        return unproven

    def gate_nudge(self, unproven: list[dict[str, Any]]) -> None:
        """Add a line to the record telling the gate model which due steps no line proves done since the cart last
        changed, and what is left to do them with."""
        self.trace.append({'kind': 'gate_unproven', 'steps': _clip(unproven, 120)})
        missing = '; '.join((str(e.get('step') or '') for e in unproven))[:200]
        self.done_log.append(f'(Zenith: no line made since the cart last changed at step {self.cart_at} shows ' + missing + f' done, and {self.msg_limit - self.msg_used} message(s) and {self.left()} steps are left: give the call that does it)')
        self.done_steps.append(-1)

    def gate_message(self, args: dict[str, Any]) -> str | None:
        """Send the message the gate model asked for. Returns its record line, or None when it was not sent."""
        text = next((v for v in args.values() if isinstance(v, str)), '')
        batch = self.send([(1, '', text)], 1)
        if not batch:
            return None
        self.gate_msgs += 1
        return f'Q: {text}\nA: {batch[0][2]}'

    def gate_store_call(self, name: str, args: dict[str, Any]) -> str | None:
        """Make the store call the gate model asked for and log it in the record. Returns its line, or None when the
        call failed."""
        reply = self.act(name, args)
        if reply is None:
            return None
        seen = reply.get('observation')
        self.absorb(seen)
        self.done_log.append(f'step {self.step}: {name} {_dump(args)} -> {self.show(seen)[:300]} (cart then: {self.cart_mark()})')
        self.done_steps.append(self.step)
        return f'{name} {_dump(args)} -> {self.show(seen)[:400]}'

    @staticmethod
    def gate_report(said: list[str]) -> str:
        """What the gate did before going on, one line per message or call; '' when it did nothing."""
        if not said:
            return ''
        return 'Before going on, Zenith did the steps the request asks for:\n' + '\n'.join(said)

    def fitted_question(self) -> tuple[int, list[str], bool, str] | None:
        """The next question whose text the message tool takes whole: (item number, the labels it names, the shown
        flag its labels came with, text). Labels too long to ask are marked used and the next ones tried, up to
        LIST_SIZE times; None when no labels are left or none fit."""
        for _ in range(LIST_SIZE):
            number, labels, shown = self.next_labels()
            if not labels:
                return None
            fitted, text = self.fit_ask(number, labels, shown)
            if not fitted and shown:
                fitted, text = self.fit_ask(number, labels, False)
            if fitted:
                return (number, fitted, shown, text)
            self.lane_used.setdefault(number, set()).add(_words(labels[0]))
            self.trace.append({'kind': 'ask_too_long', 'item': number, 'label': labels[0]})
        return None

    def next_labels(self) -> tuple[int, list[str], bool]:
        """The labels to ask about next: the first pending follow-up, once those an answer already covered are
        dropped, else the next planned labels. Returns (item number, labels, shown)."""
        self.follow_up = [f for f in self.follow_up if not self.told_about(f[0], f[1][0])]
        if self.follow_up:
            return self.follow_up.pop(0)
        number, labels = self.next_ask(self.lanes, self.lane_found, self.lane_spent, self.lane_used)
        return (number, labels, False)

    def take_answer(self, batch: list[tuple[int, str, str]], labels: list[str], told: list[str]) -> tuple[list[str], bool]:
        """Read the answer to a sent question into the items' needs (each new one also goes to `told`; a question
        naming several labels takes no property the shopper does not mind from it) and count a find for every item
        that gained a need. Returns the new needs and whether the shopper asked for one question at a time."""
        had = [len(item.told) for item in self.items]
        heard = self.hear(batch, (LIGHT, HEAVY))
        if len(labels) > 1:
            heard.pop('free', None)
        self.take_heard(heard, told)
        new: list[str] = []
        for number, (item, size) in enumerate(zip(self.items, had), 1):
            gained = item.told[size:]
            if gained:
                new += gained
                self.lane_found[number] = self.lane_found.get(number, 0) + 1
        crowded = any((_count(x) == 1 for x in _items(heard.get('crowded'))))
        return (new, crowded)

    def interview_room(self) -> bool:
        """Whether the interview may ask once more: a message is free, the episode goes on, and steps and time are
        left."""
        return self.msgs_free() > 0 and (not self.finished) and (self.left() > 6) and (self.clock() < QUESTION_BY)

    def start_lanes(self, count: int) -> None:
        """Plan the question lanes for `count` messages, with nothing found, spent or used in any lane yet."""
        self.lanes = self.plan_lanes(count)
        self.lane_found = dict.fromkeys(self.lanes, 0)
        self.lane_spent = dict.fromkeys(self.lanes, 0)
        self.lane_used = {number: set() for number in self.lanes}

    def interview_summary(self, sent: list[tuple[int, str, str]], told: list[str]) -> str:
        """The interview's account for the loop model: each question with its answer, the new firm requirements, and
        the new needs whose value another listing holds."""
        talk = '\n'.join((f'Q: {q}\nA: {a}' for _, q, a in sent))
        needs = '; '.join(told) if told else 'none'
        return "Zenith's interviewer asked the shopper (each question used one step):\n" + talk + '\nNew firm requirements from these answers: ' + needs + '. Apply them to every pick (search again with the new values where needed).' + self.pointed_note(told, '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.')

    def pointed_note(self, told: list[str], ending: str) -> str:
        """The note naming the new needs in `told` whose value another listing holds, closed by `ending`; '' when there
        are none."""
        pointed = [need for _, need in self.refs if any((need in t for t in told))]
        if not pointed:
            return ''
        return ' These take their value from another listing: ' + ' | '.join(pointed) + ending

    def may_ask_late(self) -> bool:
        """Whether a late question may go out: none went out yet, a message is left that no order step keeps, and
        there are steps, time and model budget left to use its answer."""
        if self.late_asked or not self.roles.get('ask') or self.msg_used >= self.msg_limit:
            return False
        order_talks = any((p['when'] == 'order' and p['messages'] > 0 for p in self.process))
        if order_talks or self.left() < LATE_STEPS or self.clock() > LATE_BY:
            return False
        return bool(self.judges())

    def record_message(self, text: str, words: str) -> None:
        """Log a message and the shopper's reply: in the record the gate reads (with the step it took) and in the
        trace."""
        self.done_log.append(f'step {self.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {self.cart_mark()})')
        self.done_steps.append(self.step)
        self.trace.append({'kind': 'ask', 'q': text, 'a': words[:300]})

    def likeliest_part(self, item: Item, parts: list[str]) -> str:
        """The part whose head noun most of the item's own rows name outside their category label; the earliest part
        on a tie, so the first one when no row names any."""
        cat_key = self.category_key()
        named = []
        for row in self.rows.values():
            if cat_key and (not self.owns(item, str(row.get(cat_key) or ''))):
                continue
            named.append(_roots(' '.join((str(v) for k, v in row.items() if k != cat_key and isinstance(v, str)))))
        scores = []
        for part in parts:
            head = _roots(part.split()[-1])
            scores.append(sum((1 for words in named if head & words)))
        return parts[scores.index(max(scores))]

    def fit_plan_line(self, main: int, parts: list[int], index: dict[str, Any]) -> str:
        """The fit-plan line of one main item: the (up to four) codes its parts' in-stock candidates name, the most
        widely named first, each with how many candidates of each part name it and up to two of the main item's own
        listings that carry it; '' when no candidate names a code."""
        tally = self.fit_code_tally(parts, index)
        ranked = sorted(tally.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1].values())))[:4]
        if not ranked:
            return ''
        devices = self.item_ids(main)
        entries = []
        for code, per in ranked:
            counts = ', '.join((f'{n} {self.items[p - 1].name}' for p, n in sorted(per.items())))
            seen = [i for i in devices if code in (index['marks'].get(i, {}).get('codes') or [])][:2]
            where = f" (device seen: {', '.join(seen)})" if seen else ' (device not seen yet)'
            entries.append(f'{code}: ' + counts + where)
        return f"codes the {self.items[main - 1].name} candidates' parts name -> " + '; '.join(entries)

    def fit_code_tally(self, parts: list[int], index: dict[str, Any]) -> dict[str, dict[int, int]]:
        """For each code the in-stock candidates of `parts` name: how many candidates of each part name it."""
        tally: dict[str, dict[int, int]] = {}
        for part in parts:
            for ident in self.item_ids(part):
                if self.sold_out(ident):
                    continue
                for code in self.fit_set(ident, index):
                    per_part = tally.setdefault(code, {})
                    per_part[part] = per_part.get(part, 0) + 1
        return tally

    def key_groups(self) -> dict[str, dict[str, Any]]:
        """The opened listings' attribute keys grouped by category label, in first-seen order: per label the number of
        records ('n') and its slots ('keys'); a slot counts the records stating its key ('n') and holds the distinct
        values ('values') and each record's value ('per')."""
        cat_key = self.category_key()
        groups: dict[str, dict[str, Any]] = {}
        for ident in self.seen_order:
            record = self.full.get(ident)
            if not record:
                continue
            facts = _fact_block(record) or {}
            label = str(record.get(cat_key) or 'listings') if cat_key else 'listings'
            group = groups.setdefault(label, {'n': 0, 'keys': {}})
            group['n'] += 1
            _tally_facts(group['keys'], ident, facts)
        return groups

    def requested_groups(self, groups: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """The groups whose label matches a requested item, in order: in words it is the item's shelf or name, or it
        shares a stem of four or more letters with them, or owner_of gives it an item. All groups when none matches."""
        wanted = [(_words(item.shelf or ''), _words(item.name), _name_stems(item.shelf or '') | _name_stems(item.name)) for item in self.items]
        kept: dict[str, dict[str, Any]] = {}
        for label, group in groups.items():
            words, stems = (_words(label), _name_stems(label))
            if any((words in (shelf, name) or stems & long for shelf, name, long in wanted)) or self.owner_of(label):
                kept[label] = group
        return kept or groups

    def kept_keys(self, slots: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
        """One category's attribute keys for the table, most carried first, at most 14: a key whose values read as free
        text is left out, and a key that twins one kept before it merges into that one."""
        kept: list[tuple[str, dict[str, Any]]] = []
        for name, slot in sorted(slots.items(), key=lambda kv: (-kv[1]['n'], len(kv[0]))):
            if _free_text(list(slot['per'].values())):
                continue
            twin = next((spec for other, spec in kept if self.twins(other, spec, name, slot)), None)
            if twin is None:
                kept.append((name, slot))
            else:
                _merge_slot(twin, slot)
        kept.sort(key=lambda kv: -kv[1]['n'])
        return kept[:14]

    def call_allowed(self) -> bool:
        """Whether one more model request may start: the call cap is not reached and its answer can still come in time
        (a request the clock rules out is traced)."""
        if self.calls >= ALL_CALLS:
            return False
        if self.model_room() < CALL_FLOOR:
            self.trace.append({'kind': 'late_call_skipped', 'at': round(self.clock(), 1)})
            return False
        return True

    def request_with_waits(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        """Send one model request, and send it again after a jittered pause while wait_before_retry allows. Returns the
        last (message, status)."""
        tries = 0
        while True:
            message, status = self.call_model(model, messages, max_tokens, tools)
            if message is not None:
                return (message, status)
            wait = self.wait_before_retry(status, tries)
            if wait is None:
                return (None, status)
            pause = wait * (0.8 + 0.4 * random.random())
            self.waited += wait
            tries += 1
            time.sleep(pause)

    def wait_before_retry(self, status: Any, tries: int) -> int | None:
        """The wait (from WAITS) before a failed request goes again, or None when it does not: the status does not
        pass, its retries are spent (one after a lost connection or a timeout, len(WAITS) otherwise), or the wait
        would overrun the episode's wait budget or leave the next request too little time."""
        if not _passes(status):
            return None
        if tries >= (1 if status in (None, 408) else len(WAITS)):
            return None
        wait = WAITS[tries]
        if self.waited + wait > WAIT_TOTAL or self.model_room() - wait * 1.2 < CALL_FLOOR:
            return None
        return wait

    def booked_answer(self, model: str, data: dict[str, Any]) -> tuple[dict[str, Any] | None, int]:
        """Add a reply's reported cost, then return its message with status 200, or (None, 0) when the message has
        neither text nor tool calls."""
        usage = data.get('usage') or {}
        cost = usage.get('cost')
        if isinstance(cost, (int, float)):
            self.cost += float(cost)
        message = data['choices'][0].get('message') or {}
        if message.get('content') or message.get('tool_calls'):
            return (message, 200)
        self.trace.append({'kind': 'llm_fail', 'model': model, 'status': 0})
        return (None, 0)

    def booked_step(self, name: str, args: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Take in a store reply: its step count, its end flag, and the listings a page tool opened (unless the page
        came back with an error). Returns the reply."""
        step = data.get('step')
        self.step = step if isinstance(step, int) else self.step + 1
        if data.get('finished'):
            self.finished = True
        if name in self.page_tools():
            seen = data.get('observation')
            if not (isinstance(seen, dict) and seen.get('error')):
                self.opened.update(_arg_texts(args))
        return data

    def store_retry(self, result: Any) -> bool:
        """Trace a failed store call and say whether to send it again: a 409 goes to step_conflict, a client error
        stops it (402 also marks the account out of credit), anything else is sent again."""
        error = result.error or {}
        status = error.get('status')
        body = str(error.get('body') or '')
        self.trace.append({'kind': 'env_fail', 'status': status, 'body': body[:200]})
        if status == 409:
            return self.step_conflict(body)
        if status == 402:
            self.broke = True
        return status not in (400, 401, 402, 403, 404, 422)

    def step_conflict(self, body: str) -> bool:
        """Read a 409 from the store. True when the call should go again at the step the store expects (the number it
        names, or one past a step it says is used); False when the episode is over or the conflict is unknown."""
        low = body.lower()
        if 'finish' in low or 'quarantin' in low or 'finaliz' in low:
            self.finished = True
            return False
        found = re.search('expected (\\d+)', body)
        if found:
            self.step = int(found.group(1)) - 1
            return True
        if 'reuse' in low or 'different action' in low:
            self.step += 1
            return True
        return False

    def audit_opens(self, got: dict[str, Any] | None) -> bool:
        """Open the listings the review asked to see that are still unopened and on sale; True when a page was
        added."""
        rows = [i for i in self.audit_ids((got or {}).get('open')) if i not in self.opened]
        return bool(rows) and self.open_rows(rows)

    def audit_message(self, named: dict[str, list[str]], reason: Any) -> str:
        """The review's finding for the model: the listings it thinks may fit, per item, and its reason."""
        listed = '; '.join((f"item {k}: {', '.join(v)}" for k, v in named.items()))
        because = f' ({reason})' if reason else ''
        return 'A careful review thinks these listings may fit: ' + listed + because + '. Open them and buy what truly fits; end without a purchase only if each one fails.'

    def audit_records(self) -> list[str]:
        """Up to 30 records of the listings seen, newest first, sold-out ones left out."""
        records: list[str] = []
        for ident in reversed(self.seen_order):
            if self.sold_out(ident):
                continue
            records.append(self.audit_record(ident))
            if len(records) >= 30:
                break
        return records

    def audit_record(self, ident: str) -> str:
        """One listing as the review reads it: OPENED with its page when one was read, else ROW with its search
        row."""
        page = self.full.get(ident)
        if page:
            return 'OPENED ' + _dump(_clip(page, 200))[:1200]
        return 'ROW ' + _dump(_clip(self.rows.get(ident), 120))[:500]

    def audit_prompt(self, records: list[str]) -> str:
        """The review prompt: the request, the requirements in force and the listing records."""
        return AUDIT + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger() + '\n\nLISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)

    def audit_ids(self, values: Any) -> list[str]:
        """The ids among `values` (one value or a list, read as text) of listings a search showed that are not sold
        out, in order."""
        ids = []
        for value in _items(values):
            ident = str(value)
            if ident in self.rows and (not self.sold_out(ident)):
                ids.append(ident)
        return ids

    def out_of_play(self) -> bool:
        """Whether the episode is over or has no step left."""
        return self.finished or self.left() <= 0

    def whole_cart_ready(self) -> bool:
        """Ready a cart for two or more items for a forced order: fill_short_cart adds a line for each item without
        one, and a whole cart of two or more lines has its lines' facts and part links read. False (traced) when the
        cart stays short and the episode can end without a purchase instead."""
        if not self.fill_short_cart() and self.can_refuse():
            self.trace.append({'kind': 'short_cart_refused', 'lines': self.cart_ids()})
            return False
        if len(self.cart_ids()) >= 2:
            self.fill_facts(self.cart_ids(), self.tail())
            if self.left() >= 3 and self.model_room() > 5:
                self.show_links(self.cart_ids(), self.tail())
        return True

    def closing_call(self, tool: str, args: dict[str, Any], stage: str, kind: str) -> tuple[bool, dict[str, Any] | None]:
        """Send a call that ends the episode, its arguments shaped to the tool's schema, after the request's process
        steps for `stage` when two steps are left. Returns (fits, reply): fits is False when the arguments do not fit
        the schema; reply is None when nothing was sent or the store did not answer."""
        faults: list[str] = []
        args = _shape(args, self.schemas.get(tool), 'args', faults)
        if faults:
            return (False, None)
        if self.left() >= 2:
            self.gate(stage)
        if self.out_of_play():
            return (True, None)
        reply = self.act(tool, args)
        self.trace.append({'kind': kind, 'reply': _clip(reply, 200)})
        return (True, reply)

    def refusal_text(self, tool: str, reply: dict[str, Any], lead: str) -> str:
        """Take in the store's answer to a refused ending and put it in words for the model."""
        self.observe(tool, reply.get('observation'))
        return lead + self.show(reply.get('observation'))

    def short_result(self, content: str) -> str:
        """An older tool result cut to its first 400 characters, followed by the seen listings it named (up to 20)."""
        quoted = re.findall('"([A-Za-z0-9][A-Za-z0-9_\\-:.]{5,40})"', content)
        known = [i for i in dict.fromkeys(quoted) if i in self.rows][:20]
        return content[:400] + SHORTENED + ', '.join(known) + ')'

    def opening_message(self) -> str:
        """The first user message: the request, what the store shows first (also taken into the records), how to
        start, and the status."""
        text = "SHOPPER'S REQUEST:\n" + self.request
        opening = self.view.get('opening')
        if opening is not None:
            self.absorb(opening)
            text += '\n\nWHAT THE STORE SHOWS FIRST:\n' + self.show(opening)
        return text + '\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n' + self.status()

    def close_episode(self) -> None:
        """End an episode the loop left open: force the ending; when the store refuses it, give the model a few more
        turns with the store's answer; then force the ending once more if the episode is still open."""
        answer = self.finish(again=True)
        if answer and (not self.finished) and (not self.broke) and (not self.no_credit) and (self.left() > 1):
            self.chat_user('[Zenith] Zenith tried to end the episode and ' + answer + '\nApply what the store says, then finish.\n')
            self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
        if not self.finished:
            self.finish()

    def loop_may_turn(self, loop_cap: int, spend_cap: float) -> bool:
        """Whether the loop takes another turn: the episode is open with steps left, the turn and spend caps are not
        reached, and it is before LAST_TURN (later only while the cart is empty and the model has EMPTY_ROOM)."""
        if self.broke or self.out_of_play():
            return False
        if not (self.loop_calls < loop_cap and self.cost < spend_cap):
            return False
        return self.clock() < LAST_TURN or (not self.cart_ids() and self.model_room() > EMPTY_ROOM)

    def interview_first(self, calls: list[dict[str, Any]]) -> bool:
        """Whether a turn that would change the cart or end the episode waits for the shopper's interview: none has
        happened, a message is free, and steps and time allow one."""
        if self.asked or not self.roles.get('ask') or self.msgs_free() <= 0:
            return False
        committing = {self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse')} - {None}
        if not any((c['function']['name'] in committing for c in calls)):
            return False
        return self.left() > 8 and self.clock() <= ASK_BY

    def hold_calls(self, calls: list[dict[str, Any]]) -> None:
        """Answer every call of the turn with 'not run' and interview the shopper first."""
        for call in calls:
            self.chat_tool(call['id'], 'Not run (no step used): the shopper is asked first.')
        summary = self.interview()
        self.chat_user((summary or 'The interviewer had nothing to ask.') + '\n')

    def play_calls(self, content: str, calls: list[dict[str, Any]]) -> None:
        """Trace the turn, run its calls, and put each reply in the chat, the status after the last one."""
        self.trace.append({'kind': 'turn', 'text': content[:300], 'calls': [c['function'] for c in calls]})
        replies = self.run_calls(calls)
        last = len(replies) - 1
        for index, (cid, text) in enumerate(replies):
            if index == last:
                text += '\n' + self.status()
            self.chat_tool(cid, text)

    def chat_user(self, head: str) -> None:
        """Add a user message: `head`, then the current status."""
        self.chat.append({'role': 'user', 'content': head + self.status()})

    def chat_tool(self, cid: str, text: str) -> None:
        """Add the reply to one tool call."""
        self.chat.append({'role': 'tool', 'tool_call_id': cid, 'content': text})

    def unmix_moved(self, item: Item, pointers: list[str], rest: list[str]) -> list[str]:
        """The item's other requirements that the model says only describe the listing a pointer requirement names."""
        prompt = UNMIX + '\n\nREQUEST:\n' + self.request + f'\n\nPRODUCT: {item.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n' + '\n'.join(pointers) + '\nOTHER REQUIREMENTS:\n' + '\n'.join(rest)
        got = self.ask_json(prompt, [HEAVY, LIGHT], 1500) or {}
        named = [str(x).strip() for x in _items(got.get('describe_other'))]
        return [m for m in rest if m in named]

    def unmix_fold(self, number: int, item: Item, pointer: str, moved: list[str]) -> None:
        """Fold the describing requirements into the pointer requirement they describe, in the musts and in the refs."""
        joined = pointer + ' (that listing: ' + '; '.join(moved) + ')'
        item.musts = [joined if m == pointer else m for m in item.musts if m not in moved]
        self.refs = [(n, joined if (n, need) == (number, pointer) else need) for n, need in self.refs]
        self.trace.append({'kind': 'unmix', 'item': number, 'moved': moved})

    def points_elsewhere(self, need: str) -> bool:
        """True for a requirement whose value comes from another listing: a known pointer or one of the brief's lookups."""
        if need in [m for _, m in self.refs]:
            return True
        return need in [str(x) for x in _items(self.brief.get('lookups'))]

    def item_fact_blocks(self, item: Item) -> list[dict[str, Any]]:
        """The attribute blocks of the full records labelled for this item (of every full record while no category key
        is known)."""
        cat_key = self.category_key()
        blocks = []
        for record in self.full.values():
            if cat_key and (not self.owns(item, str(record.get(cat_key) or ''))):
                continue
            block = _fact_block(record)
            if block:
                blocks.append(block)
        return blocks

    def ledger_item(self, number: int, item: Item, many: bool) -> str:
        """An item's ledger line: its kind, store label and quantity, the requirements in force, then those stated only
        in the shopper's words (which decide nothing)."""
        told: list[str] = []
        needs: list[str] = []
        for need in item.musts + item.told:
            if need in item.dropped:
                continue
            if self.described(number, need):
                told.append(need)
            elif need in self.ref_values:
                needs.append(need + f' (the page it points at gives: {self.ref_values[need]})')
            else:
                needs.append(need)
        head = f'item {number}: {item.name}' if many else f'item: {item.name}'
        if item.shelf:
            head += f' (store label: {item.shelf})'
        head += f', quantity {item.qty}'
        text = head + '; requirements: ' + ('; '.join(needs) if needs else 'none beyond the kind of product')
        if told:
            text += "; described in the shopper's words, which no listing attribute states (decides nothing): " + '; '.join(told)
        return text

    def ledger_terms(self) -> list[str]:
        """The ledger lines for the order as a whole: budget, pick rule, stock, wishes, the request's rules and a quiet
        ending."""
        brief = self.brief
        lines = []
        if self.budget is not None:
            scope = 'for each item' if brief.get('budget_scope') == 'each' else 'in total'
            lines.append(f"budget: at most {self.budget} {brief.get('budget_unit') or ''} {scope}".replace('  ', ' '))
        lines.append(self.ledger_pick())
        if brief.get('in_stock_only', True) is not False:
            lines.append('in stock only')
        lines += [f'wish (tie-breaker only): {wish}' for wish in _items(brief.get('wishes'))]
        lines += [f'rule from the request: {rule}' for rule in _items(brief.get('rules'))]
        if brief.get('quiet_decline') is True:
            lines.append('if nothing fits: end without naming or suggesting any listing')
        return lines

    def ledger_pick(self) -> str:
        """The ledger line for the request's pick rule."""
        pick = self.brief.get('pick')
        if pick == 'cheapest':
            return 'pick: the cheapest listing that fits (lowest price)'
        if pick in ('most', 'least'):
            return f"pick: the {pick} {self.brief.get('pick_measure') or ''} among fitting listings".replace('  ', ' ')
        return 'pick: any listing that fits everything'

    def ledger_facts(self) -> str:
        """The ledger line naming the tool that shows listing attributes once that is known, else ''."""
        facts = self.facts_tool()
        if not facts or not (self.facts_from.get(facts) is True or self.roles.get('offer')):
            return ''
        pages = self.page_tools()
        bare = sorted((t for t in pages | {self.roles.get('offer') or ''} if t and self.facts_from.get(t) is False))
        shown_by = sorted((t for t in pages if t != facts))
        line = f'listing attributes come from {facts}'
        if bare:
            line += f"; {', '.join(bare)} show none"
        if shown_by:
            line += f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(shown_by)}"
        return line

    def status_warning(self, left: int, tail: int) -> str:
        """The status line's warning for the steps left ('' while there is room)."""
        if left <= tail + 1:
            return ' FINAL STEPS: if the cart has a line for every item, check out now. End without a purchase only if your search proved some item lacks any fitting listing (being out of steps proves nothing of the sort).'
        if left <= 6 or (left <= len(self.items) + 5 and self.buys == 0):
            return ' Few steps left: set the cart to the chosen lines and check out now.'
        if self.buys == 0 and left <= 12:
            return ' Steps are running low: settle on your picks, fill the cart and check out within the next few steps.'
        if self.buys == 0 and (not self.cart_touched) and (self.step >= self.max_steps // 2):
            return " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        return ''

    def status_notes(self) -> str:
        """The status sections between the step line and the ledger: lookups still open, candidates to review, the fit
        plan and the latest changes."""
        idle = self.buys == 0
        text = ''
        lookups = self.status_lookups() if idle else []
        if lookups:
            text += '\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] ' + ' | '.join(lookups[:3])
        if self.leads:
            text += '\n[review candidates] ' + ', '.join(self.leads)
        plan = self.fit_plan() if self.bound and idle else ''
        if plan:
            text += '\n' + plan
        if self.notes:
            text += '\n[changes so far] ' + ' | '.join(self.notes[-4:])
        return text

    def status_lookups(self) -> list[str]:
        """Pointer requirements in force whose value no page has given yet, in the order they were found."""
        live = {need for _, need in self.needs_now()}
        return [need for _, need in self.refs if need in live and need not in self.ref_values]

    def keyed_by_id(self, rows: list[Any]) -> bool:
        """True when the learned id key holds a usable id in every one of these rows."""
        return bool(self.id_key) and all((_id_like(row.get(self.id_key), True) for row in rows))

    def guessed_id(self, node: dict[str, Any]) -> str | None:
        """The id of a lone record under the key that would be learned from it alone, kept only when that key's name
        says id or the id is a row already seen."""
        found = _learn_id([node])
        if not found:
            return None
        value = node[found].strip()
        if 'id' in str(found).lower() or value in self.rows:
            return value
        return None

    def absorb_node(self, node: Any, depth: int, in_list: bool) -> None:
        """Walk one part of a store reply (down to depth 6) and merge every listing record in it into the rows; a list
        that reads as rows teaches the id key before its records are read."""
        if depth > 6:
            return
        if isinstance(node, list):
            rows = self.learn_rows(node)
            for value in node:
                self.absorb_node(value, depth + 1, rows)
            return
        if not isinstance(node, dict):
            return
        ident = self.ident(node, in_list, depth)
        if ident:
            self.absorb_record(ident, node)
            return
        for value in node.values():
            self.absorb_node(value, depth + 1, False)

    def absorb_record(self, ident: str, node: dict[str, Any]) -> None:
        """Merge one listing record into its row and, once it shows attributes, its full record. A flag turning from
        True to False marks the listing unavailable; one turning from False to True clears that."""
        if ident not in self.rows:
            self.seen_order.append(ident)
            self.rows[ident] = {}
        row = self.rows[ident]
        went_out = any((row.get(k) is True and v is False for k, v in node.items()))
        came_back = any((row.get(k) is False and v is True for k, v in node.items()))
        if went_out:
            self.unavailable.add(ident)
        elif came_back:
            self.unavailable.discard(ident)
        row.update(node)
        if _fact_block(node):
            self.full[ident] = dict(row)
        elif ident in self.full:
            self.full[ident].update(node)

    def short_texts(self) -> tuple[dict[str, set[str]], dict[str, int]]:
        """For each row key, in order of first sight: the distinct short texts it holds (1 to 40 characters, no run of
        three digits) and how many times rows hold one."""
        texts: dict[str, set[str]] = {}
        carried: dict[str, int] = {}
        for row in self.rows.values():
            for key, value in row.items():
                if not isinstance(value, str) or not 0 < len(value) <= 40 or re.search('\\d{3}', value):
                    continue
                texts.setdefault(key, set()).add(value)
                carried[key] = carried.get(key, 0) + 1
        return (texts, carried)

    def ids_by_label(self) -> dict[str, list[str]]:
        """The seen listing ids grouped by category label, in order of first sight (one '' group while no category key
        is known)."""
        cat_key = self.category_key()
        groups: dict[str, list[str]] = {}
        for ident in self.seen_order:
            label = str(self.rows[ident].get(cat_key) or '') if cat_key else ''
            groups.setdefault(label, []).append(ident)
        return groups

    def label_wanted(self, label: str, wanted: list[tuple[str, str, set[str]]]) -> bool:
        """True when a category label names a requested item: it spells an item's store label or name, shares a long
        word with them, or belongs to an item."""
        words = _words(label)
        long = _long_stems(label)
        if any((words in (shelf, name) or long & item_long for shelf, name, item_long in wanted)):
            return True
        return bool(self.owner_of(label))

    def listed_records(self, rows: list[Any]) -> list[tuple[str, dict[str, Any]]]:
        """(id, record) for each record of a list that carries an id, in list order."""
        found = []
        for row in rows:
            ident = self.ident(row, True) if isinstance(row, dict) else None
            if ident:
                found.append((ident, row))
        return found

    def evidence_ids(self, node: Any, depth: int, ids: list[str]) -> None:
        """Add to ids, in reading order, each listing of a store reply (down to depth 6) shown with an attribute block;
        the walk does not look inside such a record the first time it meets it."""
        if depth > 6:
            return
        if isinstance(node, dict):
            ident = self.ident(node, False, depth)
            if ident and _fact_block(node) and (ident not in ids):
                ids.append(ident)
                return
            children = node.values()
        elif isinstance(node, list):
            children = node
        else:
            return
        for value in children:
            self.evidence_ids(value, depth + 1, ids)

    def evidence_line(self, ident: str, needs: list[tuple[int, str]], many: bool) -> str:
        """One listing's evidence entry: for each requirement, up to three of its attributes whose keys name the
        requirement's property ('' when none do)."""
        block = _fact_block(self.full.get(ident) or {}) or {}
        parts = []
        for number, need in needs:
            prop = need.split(':', 1)[0].strip()
            prop_stems = _stem(prop)
            hits = [f'{k}={str(v)[:50]}' for k, v in block.items() if _names(prop_stems, _stem(k))]
            if hits:
                tag = f'(item {number}) ' if many else ''
                parts.append(f'{tag}{prop} -> ' + ', '.join(hits[:3]))
        if not parts:
            return ''
        return f'{ident}: ' + '; '.join(parts)

    def others_order(self) -> list[str]:
        """The seen listing ids newest first; for a cheapest pick, the ones opened in full first, each part by price."""
        order = self.seen_order[::-1]
        if self.brief.get('pick') == 'cheapest':
            order.sort(key=self.cheap_rank)
        return order

    def cheap_rank(self, ident: str) -> tuple[bool, float]:
        """Sort key for a cheapest pick: listings opened in full first, then by price, unpriced ones last."""
        price = self.price_of(ident)
        return (ident not in self.full, float('inf') if price is None else price)

    def walk_due(self) -> bool:
        """Whether a pointer chain waits for its walk: the store has sent listing rows as text (see _TEXT_WORLD), a
        live requirement that points at another listing has no value yet and no walk took it up, nothing was ordered
        yet, and the interview is over or will not come."""
        if not _TEXT_WORLD[0] or self.finished or self.broke or self.no_credit or self.buys:
            return False
        if self.roles.get('ask') and (not self.asked) and (self.msgs_free() > 0) and (self.left() > 8) and (self.clock() <= ASK_BY):
            return False
        return bool(self.walk_needs())

    def walk_needs(self) -> list[str]:
        """The live pointer requirements with no value read yet that no walk took up, in the order they came."""
        live = {need for _, need in self.needs_now()}
        out: list[str] = []
        for _, need in self.refs:
            if need in live and need not in self.ref_values and (need not in self.walked) and (need not in out):
                out.append(need)
        return out

    def walk_room(self, steps: int=WALK_ROOM) -> bool:
        """Whether the walk may take another step: more than `steps` steps left, the clock before WALK_CLOCK and the
        spend under SPEND_SOFT, so that the order still has its room."""
        return not self.finished and (not self.broke) and (self.left() > steps) and (self.clock() < WALK_CLOCK) and (self.cost < SPEND_SOFT)

    def walk_pointers(self) -> str:
        """Walk the chain of every requirement walk_needs() lists (each is taken up once, walked or not) while room
        lasts; the notes for the worker, or ''."""
        needs = self.walk_needs()
        self.walked.update(needs)
        notes = []
        for need in needs:
            if not self.walk_room(WALK_START):
                break
            notes.append(self.walk_chain(need))
        return '\n'.join((n for n in notes if n))

    def walk_note(self, text: str) -> None:
        """Give the worker the walk's notes and the status: joined to the user message that has not been sent yet,
        else as a new one."""
        if not text:
            return
        note = text + '\n' + self.status()
        last = self.chat[-1] if self.chat else None
        if isinstance(last, dict) and last.get('role') == 'user' and isinstance(last.get('content'), str):
            last['content'] += '\n\n' + note
        else:
            self.chat.append({'role': 'user', 'content': note})

    def walk_chain(self, need: str) -> str:
        """Follow one requirement's chain hop by hop from its own description; on reaching a page that states the
        value, store it in ref_values. The note for the worker."""
        prop = need.partition(':')[0].strip() or need
        desc = need
        first = self.walk_judge(prop, desc, []).get('sought')
        if isinstance(first, dict) and first.get('store') is False:
            self.trace.append({'kind': 'walk', 'need': need[:160], 'skipped': 'not a store listing'})
            return ''
        sought = self.walk_sought(first)
        chain: list[str] = []
        value = ''
        budget = {'opens': WALK_OPENS, 'searches': WALK_SEARCHES}
        steps, calls = (self.step, self.calls)
        for _ in range(WALK_HOPS):
            if not (sought[0] or sought[2]) or not self.walk_room():
                break
            found = self.walk_hop(prop, desc, sought, chain, budget)
            if found is None:
                break
            ident, said, onward = found
            chain.append(ident)
            if onward is None:
                value = said
                break
            desc, sought = onward
        self.trace.append({'kind': 'walk', 'need': need[:160], 'chain': chain, 'value': value[:80], 'steps': self.step - steps, 'calls': self.calls - calls})
        if value:
            self.version += self.ref_values.get(need) != value
            self.ref_values[need] = value
            return f'[chain walk] {need}\nOpened along its chain: ' + ' -> '.join(chain) + f". The last page gives {prop}: {value}. Measure the requirement against that value as the requirement words it: the same value, unless it asks for a different one. The chain's pages were read only for it: order another listing of the requested kind that meets the requirement and the others, not one of them."
        opened = ' -> '.join(chain) if chain else 'nothing yet'
        return f"[chain walk] {need}\nNot finished. Opened along its chain: {opened}. Still to find and open: {desc[:300]}. Look for it by its shop's name, alone or with one of the values it is described by, and read the value from its page."

    def walk_hop(self, prop: str, desc: str, sought: tuple[str, str, list[str]], chain: list[str], budget: dict[str, int]) -> tuple[str, str, tuple[str, tuple[str, str, list[str]]] | None] | None:
        """Find the listing `desc` describes: first among the shop's pages opened before (no step), then through
        searches (the shop's name with one value word each, then the shop's name alone and its next page), opening
        the likeliest rows WALK_BATCH at a time. (id, value, (next description, next sought) or None) or None."""
        shop, kind, words = sought
        judged = set(chain)
        if shop:
            known = [i for i in self.walk_rank(self.walk_shop_ids(shop, list(self.seen_order)), kind, words) if i in self.opened]
            found = self.walk_try(prop, desc, known[:2 * WALK_BATCH], judged, budget)
            if found:
                return found
            queries = [f'{shop} {w}' for w in words[:2]] + [shop]
        else:
            queries = [' '.join(words[:2]), ' '.join(([kind] if kind else []) + words[:1])]
        for query in list(dict.fromkeys((q for q in queries if q.strip()))):
            page = None
            for _ in range(2):
                if budget['searches'] <= 0 or not self.walk_room():
                    return None
                budget['searches'] -= 1
                ids, page = self.walk_search(query, page)
                pool = [i for i in self.walk_rank(self.walk_shop_ids(shop, ids) if shop else ids, kind, words) if i not in judged]
                for start in range(0, min(len(pool), 2 * WALK_BATCH), WALK_BATCH):
                    found = self.walk_try(prop, desc, pool[start:start + WALK_BATCH], judged, budget)
                    if found:
                        return found
                if query != shop or page is None:
                    break
        return None

    def walk_try(self, prop: str, desc: str, ids: list[str], judged: set[str], budget: dict[str, int]) -> tuple[str, str, tuple[str, tuple[str, str, list[str]]] | None] | None:
        """Open those of `ids` not opened yet (within the budget), then ask which of them is the listing sought."""
        batch: list[str] = []
        for ident in ids:
            if ident in judged:
                continue
            if ident not in self.opened:
                if budget['opens'] <= 0 or not self.walk_room() or (not self.walk_open(ident)):
                    break
                budget['opens'] -= 1
            batch.append(ident)
        if not batch:
            return None
        judged.update(batch)
        got = self.walk_judge(prop, desc, batch)
        match = str(got.get('match') or '').strip()
        if match not in batch:
            return None
        onward = got.get('next')
        if isinstance(onward, dict) and str(onward.get('text') or '').strip():
            return (match, '', (' '.join(str(onward['text']).split()), self.walk_sought(onward)))
        return (match, ' '.join(str(got.get('value') or '').split()), None)

    def walk_judge(self, prop: str, desc: str, ids: list[str]) -> dict[str, Any]:
        """One small model call: the description read into shop, kind and search words, and which of the pages `ids`
        is that listing, with the value it gives or the listing it sends the reader on to. {} when no room."""
        if self.calls >= ALL_CALLS - 12 or self.cost >= SPEND_SOFT or self.clock() >= WALK_CLOCK:
            return {}
        pages = '\n'.join((f'{i}: {self.page_text(i)}' for i in ids)) if ids else '(none)'
        prompt = HOP + '\n\nPROPERTY: ' + prop + '\nLISTING SOUGHT: ' + desc + '\n\nPAGES:\n' + pages
        got = self.ask_json(prompt, [LIGHT], 2500) or {}
        self.trace.append({'kind': 'walk_judge', 'ids': ids, 'got': _clip(got, 120)})
        return got

    @staticmethod
    def walk_sought(raw: Any) -> tuple[str, str, list[str]]:
        """A description read by walk_judge as (shop, kind, up to 3 search words)."""
        raw = raw if isinstance(raw, dict) else {}
        shop = ' '.join(str(raw.get('shop') or '').split())[:60]
        kind = ' '.join(str(raw.get('kind') or '').split())[:60]
        words: list[str] = []
        for word in _items(raw.get('words')):
            word = ' '.join(str(word).split())
            if word and len(word) <= 30 and (word.lower() != shop.lower()) and (word.lower() not in [w.lower() for w in words]):
                words.append(word)
        return (shop, kind, words[:3])

    def walk_search(self, query: str, page: tuple[str, Any] | None) -> tuple[list[str], tuple[str, Any] | None]:
        """Run one store search for `query` (its next page when `page` is given): the ids of its rows, and for a
        first page with more rows than it shows, the argument and value that ask for the next page."""
        plan = self.search_template() or self.walk_query_plan()
        if not plan or self.finished:
            return ([], None)
        tool, base, field = plan
        args = {**base, field: query}
        if page is not None:
            args[page[0]] = page[1]
        faults: list[str] = []
        args = _shape(args, self.schemas.get(tool), 'args', faults)
        if faults:
            return ([], None)
        reply = self.act(tool, args)
        if reply is None or self.finished:
            return ([], None)
        observation = reply.get('observation')
        self.observe(tool, observation)
        ids = list(dict.fromkeys((i for i, _ in self.records_in(observation))))
        more = None
        if page is None and ids:
            paged = self.paging(tool, observation)
            if paged and paged[3] is not None and (paged[0] > paged[1]):
                more = (paged[2], paged[3])
        self.trace.append({'kind': 'walk_search', 'q': query[:80], 'page': None if page is None else page[1], 'rows': len(ids)})
        return (ids, more)

    def walk_query_plan(self) -> tuple[str, dict[str, Any], str] | None:
        """The search tool and its words argument when no words search ran yet: the text argument whose name says
        query, search, text or words (else the only text argument), when no other argument is required."""
        tool = self.roles.get('search')
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        named = [k for k in texts if re.search(QUERY_WORDS, _snake(k), re.I)]
        field = named[0] if named else texts[0] if len(texts) == 1 else None
        if not tool or not field or any((k != field for k in schema.get('required') or [])):
            return None
        return (tool, {}, field)

    def walk_open(self, ident: str) -> bool:
        """Open one listing's page with the page tool the checkout walk uses; whether it was opened."""
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not tool or not target or self.finished:
            return False
        faults: list[str] = []
        args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
        if faults:
            return False
        reply = self.act(tool, args)
        if reply is None or self.finished:
            return False
        self.observe(tool, reply.get('observation'))
        self.walk_pages[ident] = reply.get('observation')
        self.trace.append({'kind': 'walk_open', 'id': ident})
        return ident in self.opened

    def walk_shop_ids(self, shop: str, ids: list[str]) -> list[str]:
        """Those of `ids` whose row names the shop: a short text field equal to it, or holding it as whole words."""
        name = ' '.join(shop.lower().split())
        if not name:
            return []
        near = re.compile('(?<![a-z0-9])' + re.escape(name) + '(?![a-z0-9])')
        out: list[str] = []
        for ident in dict.fromkeys(ids):
            for value in (self.rows.get(ident) or {}).values():
                if isinstance(value, str) and len(value) <= 60:
                    text = ' '.join(value.lower().split())
                    if text == name or (len(text) <= 40 and near.search(text)):
                        out.append(ident)
                        break
        return out

    def walk_rank(self, ids: list[str], kind: str, words: list[str]) -> list[str]:
        """`ids` with rows naming the kind first, then by how many search words their rows hold, else in store order."""
        kinds = {w for w in _walk_words(kind) if len(w) >= 3}
        values: set[str] = set()
        for word in words:
            values |= _walk_words(word)

        def score(item: tuple[int, str]) -> tuple[int, int, int]:
            index, ident = item
            text = _walk_words(_dump(self.rows.get(ident) or {}))
            return (0 if kinds & text else 1, -len(values & text), index)
        return [i for _, i in sorted(enumerate(dict.fromkeys(ids)), key=score)]

    def page_text(self, ident: str) -> str:
        """A listing's page for walk_judge: short fields and detail blocks first, long texts cut and last. A page the
        walk opened but the record reader did not take in is shown as the store sent it, never as its search row."""
        record = self.full.get(ident) or self.walk_pages.get(ident) or self.rows.get(ident) or {}
        if not isinstance(record, dict):
            return str(record)[:2000]
        short = {k: v for k, v in record.items() if not (isinstance(v, str) and len(v) > 500)}
        long = {k: v[:200] + '...' for k, v in record.items() if isinstance(v, str) and len(v) > 500}
        return _dump(_clip({**short, **long}, 500))[:2000]

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Zenith(problem_data)
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
