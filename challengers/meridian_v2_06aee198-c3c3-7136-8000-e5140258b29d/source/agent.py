"""meridian v3: an ORO SN15 shopping agent that buys for one shopper inside a store world, built on the public
race-15 winner Halo v2 (itself built on Know Thyself, the public race-13 winner) with meridian's own changes below.

How an episode runs:
- read_request(): the light model reads the request and the store's tools into a brief (items with their stated
  requirements, quantities and store labels, budget, pick rule, message and step limits, steps the request wants
  done along the way, and which tool searches, opens, compares, messages, carts, buys or declines).
- work(): the light model (SPARE as fallback) drives the store's tools under the GUIDE prompt. Code keeps a ledger of
  every listing record seen, the requirements in force and the cart, and adds it to each turn (status()).
- interview(): once enough listings are open, plan_lanes() asks the heavy model for each item's likeliest unwritten
  preferences, ranked, and the shopper is asked in lists of up to LIST_SIZE properties of one item while it answers
  lists, one property per message after a reply that asks for one question at a time. settle() rotates the lists so a
  property the shopper does not mind cannot hide the ones behind it; late_question() spends a message kept for an
  ending on one more question before a checkout.
- Before a checkout, run_calls() re-reads offers, follows reference chains (trail()), checks accessory fit and cart
  lines, and asks a judge (check_cart()); before ending without a purchase it audits the listings seen (audit()) and
  holds a decline once when fitting listings appear. finish() forces a checkout or a decline at the end.
- Limits: TASK_SECONDS per episode with TIME_MARGIN kept, the ASK_BY / QUESTION_BY / LATE_BY / WIND_DOWN / LAST_TURN
  clock gates, model spend SPEND_SOFT / SPEND_HARD per episode, and the LOOP_CALLS / ALL_CALLS call caps.

meridian's changes (each found on our replica of the race-14 world, where the base declined with a fitting listing
already open; paired run on 25 tasks, one model cache: 17 -> 19 passed; strict race-12 replica, 20 tasks: 9 = 9):
- name_needs(): a disclosed need keeps the property name the shopper's own words use ("pack type", not the reader's
  "packaging type"; "pants length", not "length"; words compared without punctuation, side notes left out, and never
  a shorter name the question did not ask, so "storage type," stays "storage type"), and agrees() drops a reading
  that excludes a value the item already holds for the same property (a double negative read as "other than X" next
  to "X"); a reply that turns down anything other than a value ("anything on P other than V gets a polite no
  thanks") asks for that value, so a reading "P: not V" of it is turned back into "P: V" (_turned_down_but()),
  unless V itself is a negation ("other than not available").
- side_notes(): notes that come back beside the shopper's reply (a requirement dropped, a budget change) go through
  the store-note reader, as notes on every other tool's result already did.
- Store rules: a requirement that takes its value through a rule the store states (a table of which value goes with
  which) is judged by the value the rule gives, in the chain reader (TRAIL), the cart check (CHECK) and the decline
  audit and sold-out swap (READING).
- Store surfaces the base could not read (tests on our recorded episodes; behaviour on text-id stores unchanged, free
  replays identical on two packs): where some tool takes a whole-number id (int_ids, _takes_int_ids), listing ids
  that are whole numbers under an id-named key are tracked like text ids (as_id, _id_named, _int_ids), never one
  under a key naming something else (search_id, order_id, cart_id: _context_id), never a container of rows, and never
  a price or a total in the cart view (_ints_under); in other stores every id path reads text ids only, as in the
  base. A listing whose attributes come as a list of name/value records or of "Name: value" lines gets them read
  as its attribute map (_listed_facts) when it has no map of its own.
- One-line listing records: a store that writes each search row and listing page as one string ("ref_ref: p-... |
  title: ... | price: 14284 | category: ... | facts: battery capacity=5000 mah; network=5g | ..."), as the race-15
  store did, gets those strings turned into maps in act() before anything reads them (_with_records, _text_record).
  In the base, such strings left the ledger empty (no attribute keys for the interview, no records for the cart check)
  and the display cut each one at SHOW_CHARS, hiding about half of a race-15 page's attributes (18,197 of 37,044
  entries in Halo's own 180 race-15 episodes). On local worlds given that surface: q160 base 11/15, this file 13/15;
  pack 14 base 8/25 (15 tasks lost on undisclosed needs), this file 20/25. A whole-record string can also be JSON
  text holding a listing, or "key=value; ..." parts with an id (_string_record).
- The interview starts by ASK_SOON (110 s) at the latest (ready_to_ask()): in live race 15 the base sent no question
  in 8 of 180 episodes, and the love line, whose cutoff falls near 135 s, in 87; a preference the request leaves
  out is only ever drawn out by a question.
- An interview whose lanes run dry with messages left plans once more, naming what was already asked (extend_lanes()),
  while the episode is open, steps remain and the clock is under REPLAN_BY (180 s): in live race 15, 36 of the base's
  180 episodes sent one or two of three messages because their lanes held 1-4 labels (no attribute keys could be
  read).
- Calls Halo's own code builds (page reads, reference chains, offer re-reads, the forced checkout and decline, cart
  repairs, follow-up searches, the interview's messages) give a required parameter the base never fills a safe value
  (with_required(), fill_value()): a reading tool's choice takes the option naming all / full / complete / default,
  a preview-only switch ("dryRun", "rehearse", "simulate") is false, a confirm switch true, a page size 20 and a page
  number the first page, the checkout's note a neutral line; choices and switches of the cart, checkout and decline
  tools are never guessed. Without this, a store whose page tool requires a view choice, or whose finish tool requires
  a note or a preview switch, silently turned those calls off (our make_newworld reqargs / negbool worlds).
- check_roles(): the shopper messages never go to the checkout or decline tool; when the brief names a closing tool
  for them, they move to the one free-text tool left, or the interview is dropped.

Entry point: agent_main(problem_data), which returns the episode's trace.
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
AUTO_PAGES = 6
CHEAPER_OPENS = 3
TASK_SECONDS = 300.0
TIME_MARGIN = 20.0
CALL_FLOOR = 6.0
ASK_BY = 165.0
# without an interview yet, start it at the latest once this many seconds have passed (see ready_to_ask())
ASK_SOON = 110.0
REPLAN_BY = 180.0
NOOP_FLAG = re.compile('dry.?run|preview|rehears|simulat|sandbox|what.?if|validat\\w* only|check(?:ed)? only|checked but not|not (?:be )?carried out|nothing is (?:applied|changed|saved|bought)|without (?:committing|applying|changing|saving)|keeps everything as it was|test (?:only|mode|run)', re.I)
CONFIRM_FLAG = re.compile('(?:^| )(?:confirm(?:ed)?|commit|apply|execute|proceed)(?: |$)')
CHOICE_DEFAULT = re.compile('(?:^| )(?:all|full|complete|everything|default|standard)(?: |$)')
SIZE_ARG = re.compile('(?:^| )(?:limit|size|per|rows|results|count)(?: |$)')
PAGE_ARG = re.compile('(?:^| )(?:page|leaf|offset|start|first|from|skip|cursor)(?: |$)')
NOT_SIZE = re.compile('price|cost|budget|cap|amount|qty|quantity|units|stock|needed|wanted')
NOTE_ARG = re.compile('(?:^| )(?:note|comment|message|reason|explanation|remark|remarks|memo|text)(?: |$)')
QUESTION_BY = 210.0
WIND_DOWN = 200.0
LAST_TURN = 255.0
EMPTY_ROOM = 12.0
LATE_BY = 190.0
LATE_STEPS = 8
SCARCE_MESSAGES = 2
LIST_SIZE = 8
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
GUIDE = 'You are Halo, a careful buyer working for one shopper in an online store. You can act only by calling the store\'s tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.\n\nHOW THE STORE WORKS\n- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool\'s description and schema and use the exact argument names and allowed values.\n- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.\n- Search rows are summaries. Stores split a listing\'s record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Halo knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.\n- Questions to the shopper are handled for you by Halo\'s interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.\n- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Halo carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store\'s tools (product questions go through Halo\'s interviewer).\n\nWHAT COUNTS AS A FIT\n- The kind of product is decided by the listing\'s category label: only the label that matches the requested item as a whole counts, never a neighbouring one.\n- A requirement on a property is met when the listing\'s attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value\'s words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.\n- An attribute whose key names the requirement\'s property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing about the requirement. Halo adds [evidence] lines after opened listings with the keys whose names match a requirement.\n- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.\n- When the category label itself states a requirement\'s value (a season or audience in the label), that requirement is met by the category.\n- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing\'s value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.\n- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.\n- When one requested item must fit another (an accessory for a device), choose the two together from the listings\' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.\n- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.\n- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.\n\nCHOOSING\n- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Halo reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.\n- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.\n- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.\n- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.\n- A request for several different items needs one cart line per item, each at its requested quantity.\n\nCHANGES WHILE YOU WORK\n- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool\'s result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.\n- Re-read a chosen listing\'s record shortly before checking out, because price and stock can change silently.\n\nFINISHING\n- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.\n- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request\'s wording for that ending, and when a remark is needed, keep it short and name no listing.\n- Every reply must call at least one tool. Keep your text short: what you know, then the next move.'
BRIEF = 'You prepare a shopping task for an agent. Read the shopper\'s request and the store\'s tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:\n\n"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request\'s words, "shelf": the store\'s category label for it when a tool lists labels (copy it exactly), or the category the request itself names for it ("a coat in rainwear" gives "rainwear"), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request\'s own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property\'s value or limit outright; and never turn words that are part of what the product is called (a season or an audience named with the product) into musts: those belong to "name". A use or occasion the request makes a property of the product itself ("blazers for business wear", "a dress made for parties", "a bag, and it is for a wedding", "the occasion has to be party") IS a must on that property ("occasion: business"), unless the store\'s category label already names it; a particular person\'s event, a person or a setting the product is bought for ("to wear at his wedding", "for a colleague"), and how it should sit, feel or look in use, stay context. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.\n"budget": the most the shopper will pay in total as a number in the request\'s own unit, or null.\n"budget_unit": that unit in words (for example "cents"), or null.\n"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.\n"in_stock_only": true unless the request clearly accepts items that are not available.\n"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.\n"pick_measure": what is maximised or minimised for "most"/"least", else "".\n"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.\n"steps": how many steps, moves or tool calls the request allows (integer), or null.\n"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].\n"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.\n"wishes": a list of things the shopper would merely like.\n"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).\n"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper\'s approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.\n"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.\n"fits": a list of {"item": item number, "with": item number} for each requested item that the request says must work with, fit or suit ANOTHER requested item (the one the agent picks); [] when none.\n"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing\'s attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing\'s current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).\n"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.\n"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.\n'
PLAN = 'You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property; a property the shopper has no view on gets "don\'t mind", and one the shopper does not think about at all gets a puzzled reply, so every property named spends part of a message.\n\nYou get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item\'s category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.\n\nChoose the properties in this order:\n0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.\n1. Hinted properties. The request\'s story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product\'s looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.\nName a hinted property by a listed key of that item when one fits the hint (a hint about who will wear it: the key for gender or audience; about its looks: the listed key for colour, pattern or finish that most of its listings carry); a property no listing of the item carries comes after the item\'s three most carried keys.\n2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item\'s listings; once five or more listings were read, avoid keys seen on only one or two of them.\nSkip a property the request already settles for that item (a requirement names that property; a key whose values only repeat a required value, such as a pattern key showing the required decoration, is NOT settled), a second name for a property already chosen, a key whose values each list several purchase options at once, identifiers, model names, warranty and other logistics. Pack sizes, counts and quantities per pack are properties like any other, and a key marked [not the stated ...] is another property than that requirement, whatever words its name shares with it. Rank every item on its own: its hinted properties first, then its listed keys.\n"label" is the property\'s plain name: for a listed key, all of the key\'s words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property\'s name, no example values, no brackets, no second property.\nChoose up to COUNT for each item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story\'s words it comes from, or an empty string"}]}, each item\'s picks in rank order'
HEAR = 'A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants them asked one at a time; not answers that only say they did not follow it or want it in simpler words], "budget": new total limit as a number or null}\n- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3"); the value is only what a listing could state, from the wanted value or its limit word to its end, without the framing the answer puts around it ("oval" for "matching oval", "settled, with oval" or "oval and nothing else", "at least 30 cm" for "equal to at least 30 cm").\n- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop\'s product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer\'s own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.\n- "free": the property of each question the shopper said they do not mind about.\n- Folksy or roundabout wording that names a value still states that value. Wording that refuses to bargain over a value (it will not be traded, given up or changed) states that value as wanted; write \"not X\" only when the answer rejects X itself.'
NOTE = 'While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:\n{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request\'s unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}\nA replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note.'
CHECK = 'You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.\nReading rules: a requirement is met when the listing\'s attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value\'s words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing\'s value counts as met unless the records show that same value; a requirement that takes its value from a rule the store states (a table of which value goes with or pairs with which) is met by the value that rule gives. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.\nWhen the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart\'s choice on that rule, the cart is wrong.\nReturn ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses.'
READING = 'the listing\'s attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value\'s words counts, such as "deep teal" for teal, limits include the edge); a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing\'s value counts as met unless the records show that same value; a requirement that takes its value from a rule the store states (a table of which value goes with or pairs with which) is met by the value that rule gives; out-of-stock listings and listings that break the budget do not fit.'
AUDIT = 'A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: ' + READING + '\nRecords marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.\nReturn ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}'
UNMIX = 'A shopper\'s request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request\'s wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}'
TRAIL = 'A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing\'s page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement\'s chain from the requirement\'s own description, link by link, through the pages\' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).\nReturn ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing, "find": "for a link not found: search words (its shop and the values the description gives)"}], "done": true when the last link is an OPENED page that states the property\'s value itself (also when the requirement asks for a value other than or unlike that one: the page\'s value is what it is measured against), else false, "value": the property\'s value as stated on that last page (when the requirement asks for the value that goes with, pairs with or matches that one under a rule the page or the store states, the value that rule gives), or ""}]}\nA record is a link only when it matches the whole description of that link: the shop or brand, the kind of product AND every property value the description gives (for example "a 10 pieces multipack" rules out the same shop\'s listing with 10 capsules and no multipack); when no record matches all of it, that link is not found yet, even if the same shop sells other listings of that kind. Give "find" for such a link: the shop\'s name and the description\'s values, as a few search words.\nList each chain\'s links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown.'
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
    """Returns every model or part code found in the text, as FIT_CODE defines it (a capital letter, up to five more
    letters, an optional space, then 3 to 6 digits, on word boundaries), with inner whitespace collapsed and
    lower-cased. Gives an empty list when the text is empty or None or holds no code."""
    return [re.sub('\\s+', ' ', c).strip().lower() for c in re.findall('\\b' + FIT_CODE + '\\b', str(text or ''))]

def _fit_marks(text: Any) -> dict[str, Any]:
    """Scans a listing's text for compatibility marks and returns a dict: 'own' (a part code in brackets or parentheses,
    upper-cased, else ''), 'codes' (codes after 'model no./number/#' or 'also sold as/known as'), 'works' (codes
    after 'works with', 'compatible with' or 'fits', skipping negated hits and 'part X also fits' phrases), 'also'
    ((part code, codes) pairs from those phrases) and 'more' (a part code cited for details, else '')."""
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
    """Serializes a value to compact JSON (no spaces after separators, non-ASCII characters kept as they are) and falls
    back to str(value) when the value cannot be encoded."""
    try:
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        return str(value)

def _clip(value: Any, limit: int, depth: int=0) -> Any:
    """Returns a copy of a nested value in which every string longer than limit is cut to limit characters followed by
    '...', walking through dicts and lists. Anything nested more than seven levels down is replaced by the string
    '...'; other scalars come back unchanged."""
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
    """Parses a model reply into a JSON object: trims it, unwraps a triple-backtick fence around the whole text, tries
    to parse it all, and when that is not valid JSON parses the span from the first '{' to the last '}'. Returns the
    dict, or None for non-string input, unparseable text, or JSON that is not an object."""
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
    """Normalizes a key or phrase into lower-case space-separated words: puts a space at camelCase boundaries (a
    lower-case letter or digit before a capital), turns runs of whitespace, underscores, dots and hyphens into single
    spaces, and trims the result, so 'lidColour' and 'lid_colour' both become 'lid colour'."""
    key = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(key))
    return re.sub('[\\s_.\\-]+', ' ', key).strip().lower()

def _snake(key: str) -> str:
    """Returns a key in snake_case: the lower-case words produced by _words joined with underscores, so 'lidColour' and
    'lid colour' both become 'lid_colour'."""
    return _words(key).replace(' ', '_')

def _stem(text: str) -> frozenset[str]:
    """Returns a frozenset of the text's _words tokens, each word longer than three letters losing a trailing 's'. This
    rough singular form lets property names, attribute keys and category labels be compared by shared words
    regardless of plurals."""
    return frozenset((w[:-1] if len(w) > 3 and w.endswith('s') else w for w in _words(text).split()))

def _roots(text: str) -> set[str]:
    """Returns the set of content-word roots in a text: each _words token drops a plural 'es' after ch, sh, ss, x or z,
    or else a trailing 's' when it is longer than three letters and does not end in 'ss'. Roots shorter than three
    letters and the words and, for, the and with are left out."""
    out = set()
    for w in _words(text).split():
        w = w[:-2] if re.search('(?:ch|sh|ss|x|z)es$', w) else w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w
        if len(w) >= 3 and w not in ('and', 'for', 'the', 'with'):
            out.add(w)
    return out

def _spelled(text: str) -> frozenset[str]:
    """Returns a frozenset of the text's words in a spelling-neutral form: a plural 's' is dropped from words longer
    than three letters that do not end in 'ss', then a final 'our' becomes 'or', so 'Colours' and 'color' give the
    same word."""
    return frozenset((re.sub('our$', 'or', w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w) for w in _words(text).split()))

def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    """Tells whether a requirement's property words and an attribute key's words name the same property: False when
    either set is empty; True when the sets are equal, when every property word appears in the key, or when a key of
    at least two words lies wholly inside the property."""
    if not prop or not key:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)

def _kinds(schema: dict[str, Any]) -> list[str]:
    """Returns the JSON-schema type names declared by a schema as a list: a single string 'type' becomes a one-element
    list, a list of types keeps only its string entries, and a missing type gives an empty list."""
    kind = schema.get('type')
    if isinstance(kind, str):
        return [kind]
    return [k for k in kind or [] if isinstance(k, str)]

def _fits_kind(value: Any, kind: str) -> bool:
    """Tells whether a value matches one JSON-schema type name (string, integer, number, boolean, array, object or
    null), never counting booleans as integers or numbers. An unrecognized type name accepts any value."""
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
    """Coerces a value toward a JSON schema, recursing into array items and object properties, and returns the result.
    It repairs integral floats, numeric strings, numbers given for text, 'true'/'false' strings, a lone value for an
    array, enum case and over-long strings; every rule still broken (type, enum, minLength, minimum, maximum, item
    counts, disallowed extra keys, missing required keys) appends a message naming its path to faults."""
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
    """Returns a value as a list: a list unchanged, a non-empty string or dict wrapped in a one-element list, and
    anything else (None, empty strings or dicts, numbers, booleans) as an empty list."""
    if isinstance(value, list):
        return value
    if isinstance(value, (str, dict)) and value:
        return [value]
    return []

def _count(value: Any) -> int | None:
    """Reads a whole count from a value: an int (booleans excluded), a float with no fractional part, or a string of
    digits optionally ending in '.0' with surrounding spaces allowed. Returns None for anything else, such as
    booleans, fractional numbers or other text."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.0+)?\\s*', value):
        return int(float(value))
    return None

def _as_id(value: Any) -> str | None:
    """A listing id in its text form: a string stripped, a whole number (not a bool) written out; None otherwise."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return None

def _int_ids(props: dict[str, Any]) -> list[str]:
    """Whole-number parameters whose name holds the word id (item_id, ItemId): where a store's listing ids are numbers."""
    return [k for k, v in props.items() if isinstance(v, dict) and 'integer' in _kinds(v) and 'enum' not in v and 'id' in _words(k).split()]

def _id_named(key: Any) -> bool:
    """Whether a key holds the word 'id' (item_id, ItemId, product_id_ref, id; not width or valid): a number counts as an
    id only under such a key."""
    return 'id' in _words(str(key)).split()

def _id_like(value: Any, named: bool=False, ints: bool=False) -> bool:
    """Tells whether a value could be a listing id: a string whose trimmed length is 3 to 64 characters, with no space
    inside, that contains a digit; when named is True (the key says it is an id) the digit is not required. A whole
    number (not a bool) counts only in a store whose ids are numbers (ints), under an id-named key (named), and when
    not negative."""
    if isinstance(value, int) and not isinstance(value, bool):
        return ints and named and value >= 0
    if not isinstance(value, str) or not 3 <= len(value.strip()) <= 64 or ' ' in value.strip():
        return False
    return named or bool(re.search('\\d', value))

def _learn_id(rows: list[dict[str, Any]], ints: bool=False) -> str | None:
    """Chooses which key of the first row holds listing ids: every row's value under it must pass _id_like (no digit
    needed when the key contains 'id'; whole numbers only when ints is True, under a key holding the word id that
    does not name something else such as search_id or order_id, _context_id) and, with several rows, the values in
    text form (_as_id) must all differ. Prefers keys containing 'id', then the longest average value; returns None
    when no key qualifies."""
    best = None
    for key in rows[0]:
        values = [row.get(key) for row in rows]
        if not all((_id_like(v, _id_named(key) and (not _context_id(key)), ints) if isinstance(v, int) and (not isinstance(v, bool)) else _id_like(v, 'id' in str(key).lower()) for v in values)):
            continue
        values = [_as_id(v) for v in values]
        if len(rows) > 1 and len(set(values)) < len(values):
            continue
        score = ('id' in str(key).lower(), sum((len(v) for v in values)) / len(values))
        if best is None or score > best[0]:
            best = (score, key)
    return best[1] if best else None

def _in_rows(node: Any, ident: str, depth: int=0, ints: bool=False) -> bool:
    """Tells whether ident is a value of some dict that sits in a list anywhere in a nested result (lists and dict
    values walked down to depth 5): a string value (trimmed), or, when ints is True (the store's ids are numbers), a
    whole number under an id-named key, never a price or a count. Scalars, and nodes deeper than that, give False."""
    if depth > 5:
        return False
    if isinstance(node, list):
        return any((isinstance(v, dict) and ident in [_as_id(x) for k, x in v.items() if isinstance(x, str) or (ints and _id_named(k) and _as_id(x) is not None)] for v in node)) or any((_in_rows(v, ident, depth + 1, ints) for v in node))
    if isinstance(node, dict):
        return any((_in_rows(v, ident, depth + 1, ints) for v in node.values()))
    return False

def _llm_within(seconds: float) -> ProxyClient:
    """Returns a ProxyClient whose timeout fits the seconds left: the shared 50-second client _llm when seconds is at
    least 50, otherwise a single-retry client with the timeout rounded down to a multiple of 5 (never under 5),
    created on first use and cached in _llm_short."""
    if seconds >= 50:
        return _llm
    limit = max(5, int(seconds) // 5 * 5)
    if limit not in _llm_short:
        _llm_short[limit] = ProxyClient(timeout=limit, max_retries=1, retry_delay=1)
    return _llm_short[limit]

def _fact_block(node: Any) -> dict[str, Any] | None:
    """Returns a record's attribute block: among the dict values directly under node, the one with the most scalar
    entries (strings and numbers, not booleans), reduced to those entries. When no child dict has at least two scalar
    entries, the largest attribute list a child value holds (_listed_facts: name/value records or "Name: value"
    lines) stands in; None when node is not a dict or neither form is found."""
    if not isinstance(node, dict):
        return None
    best = None
    for value in node.values():
        if isinstance(value, dict) and len(value) >= 2:
            scalars = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and (not isinstance(v, bool))}
            if len(scalars) >= 2 and (best is None or len(scalars) > len(best)):
                best = scalars
    if best is None:
        for value in node.values():
            listed = _listed_facts(value)
            if listed and (best is None or len(listed) > len(best)):
                best = listed
    return best

FACT_NAME = ('name', 'key', 'label', 'attribute', 'property', 'field', 'title', 'spec')
FACT_VALUE = ('value', 'values', 'val', 'text', 'detail', 'setting')

def _listed_facts(value: Any) -> dict[str, Any] | None:
    """A list of two or more name/value records, or of strings at least half of which (and two or more) read
    'Name: value', as a map {name: value}; None for any other value. Names must be 1 to 5 words; values must be plain
    (text, numbers, or a list of plain values)."""
    if not isinstance(value, list) or len(value) < 2:
        return None
    out: dict[str, Any] = {}
    skipped = 0
    for entry in value:
        if isinstance(entry, dict):
            keys = {_words(k): k for k in entry}
            name_key = next((keys[w] for w in FACT_NAME if w in keys), None)
            value_key = next((keys[w] for w in FACT_VALUE if w in keys), None)
            if not name_key or not value_key:
                return None
            name, item = (entry[name_key], entry[value_key])
            if isinstance(item, list) and all((isinstance(x, (str, int, float)) and (not isinstance(x, bool)) for x in item)):
                item = ', '.join((str(x) for x in item))
        elif isinstance(entry, str):
            hit = re.match('\\s*([A-Za-z][A-Za-z0-9 _/&+.()-]{0,40}?)\\s*:\\s*(.*)$', entry, re.S)
            if not hit or not 1 <= len(_words(hit.group(1)).split()) <= 5:
                skipped += 1
                continue
            name, item = (hit.group(1), hit.group(2).strip())
        else:
            return None
        if not isinstance(name, str) or not 1 <= len(_words(name).split()) <= 5:
            return None
        if not isinstance(item, (str, int, float)) or isinstance(item, bool):
            return None
        out[name.strip()] = item
    return out if len(out) >= 2 and skipped * 2 <= len(value) else None

def _folded(text: Any) -> str:
    """Words of a text with plural -s and a final -our folded ("colours" and "color" alike), single-spaced."""
    out = []
    for word in _words(str(text or '')).split():
        word = word[:-1] if len(word) > 3 and word.endswith('s') and not word.endswith('ss') else word
        out.append(re.sub('our$', 'or', word))
    return ' '.join(out)

def _negated(value: str) -> tuple[bool, str]:
    """(True, rest) for a value that excludes something ("not X", "other than X", "anything but X"), else
    (False, value)."""
    text = value.strip()
    hit = re.match('(?:not|no|never|other than|anything but|anything other than|except)\\s+(.*)$', text, re.I)
    return (True, hit.group(1).strip()) if hit else (False, text)

DOUBLE_NO = re.compile("\\b(?:anything|everything)\\b[^.;!?]{0,80}?\\bother than\\s+(?P<v>[^,.;!?]+?)\\s+"
                       "(?:(?:gets?|earns?|is|will get|would get)\\s+(?:a\\s+)?(?:polite\\s+|firm\\s+|flat\\s+|big\\s+)?"
                       "(?:no|nope|pass|no-go|no go|thumbs down|rejection|refusal|veto)\\b"
                       "|(?:won't|will not|wouldn't|would not)\\s+do\\b|is\\s+out\\b|is\\s+off the table\\b)", re.I)

def _turned_down_but(reply: str) -> str:
    """The value V of a reply that turns down anything other than V ("anything on P other than V gets a polite no
    thanks", "everything other than V won't do"), or ''."""
    hit = DOUBLE_NO.search(str(reply or ''))
    return hit.group('v').strip() if hit else ''

def _same_value(a: str, b: str) -> bool:
    """Whether two values name the same thing: equal words, or the shorter's words (at least one) all in the other."""
    x, y = (set(re.findall('[a-z0-9]+', a.lower())), set(re.findall('[a-z0-9]+', b.lower())))
    if not x or not y:
        return False
    return x == y or (x <= y if len(x) <= len(y) else y <= x)

CONTEXT_IDS = frozenset('next prev previous page cursor offset search query request session order orders cart basket trolley bag user customer shopper buyer seller vendor shop store merchant brand category department message thread event trace run task episode step call total count parent group result response transaction payment invoice shipment address account'.split())

def _context_id(key: Any) -> bool:
    """Whether an id-named key belongs to something other than a listing (search_id, order_id, cart_id, next_id,
    seller_id, ...): a whole number under it is never a listing id."""
    return bool(set(_words(str(key)).split()) & CONTEXT_IDS)

def _takes_int_ids(schema: Any, depth: int=0) -> bool:
    """Whether a tool's parameters take a whole-number id: an integer (or a list of integers) whose name holds the word
    id, at any level of the arguments. Only in such a store are whole numbers read as listing ids."""
    if depth > 4 or not isinstance(schema, dict):
        return False
    props = schema.get('properties') if isinstance(schema.get('properties'), dict) else {}
    for key, spec in props.items():
        if not isinstance(spec, dict):
            continue
        kinds = _kinds(spec)
        inner = spec.get('items') if isinstance(spec.get('items'), dict) else {}
        named = bool({'id', 'ids'} & set(_words(str(key)).split()))
        if named and 'enum' not in spec and ('integer' in kinds or ('array' in kinds and 'integer' in _kinds(inner))):
            return True
        if _takes_int_ids(spec, depth + 1) or _takes_int_ids(inner, depth + 1):
            return True
    return False

def _rows_inside(node: dict[str, Any], key: str) -> bool:
    """Whether a map holds a list of records that carry `key` themselves: a container of listing rows, not a listing."""
    return any((isinstance(v, list) and any((isinstance(r, dict) and key in r for r in v[:20])) for v in node.values()))

def _ints_under(node: Any, key: str | None, depth: int=0) -> set[str]:
    """The whole numbers a result holds under an id-named key (not one naming something else) or under `key`, as text."""
    found: set[str] = set()
    if depth > 6:
        return found
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key or (_id_named(k) and (not _context_id(k))):
                for x in v if isinstance(v, list) else [v]:
                    if isinstance(x, int) and (not isinstance(x, bool)):
                        found.add(str(x))
            if isinstance(v, (dict, list)):
                found |= _ints_under(v, key, depth + 1)
    elif isinstance(node, list):
        for v in node:
            found |= _ints_under(v, key, depth + 1)
    return found

def _texts_of(node: Any, depth: int=0) -> list[str]:
    """Every text value in a result (not its keys)."""
    if depth > 6:
        return []
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        return [t for v in node.values() for t in _texts_of(v, depth + 1)]
    if isinstance(node, list):
        return [t for v in node for t in _texts_of(v, depth + 1)]
    return []

def _phrase(text: Any) -> str:
    """A text folded for phrase matching: punctuation removed, then _folded() (plural -s and -our folded)."""
    return _folded(re.sub('[^0-9A-Za-z]+', ' ', str(text or '')))

RECORD_KEY = re.compile('^([A-Za-z0-9_][A-Za-z0-9_ ./()\\[\\]+-]{0,40}?):(?: (.*))?$', re.S)
FACT_KEYS = re.compile('(?:^|_| )(?:facts?|attributes?|specs?|specifications?|details?|traits?|properties)(?:$|_| )')

def _plain(text: str, fact: bool=False) -> Any:
    """A text value in its plain type: None, true / false, a whole or decimal number, or the text itself. Inside an
    attribute map (fact) "None" stays text, since it can be the attribute's own value."""
    value = text.strip()
    if value in ('None', 'null'):
        return value if fact else None
    if value in ('True', 'False'):
        return value == 'True'
    if re.fullmatch('-?\\d{1,15}', value):
        return int(value)
    if re.fullmatch('-?\\d{1,12}\\.\\d{1,6}', value):
        return float(value)
    return value

def _text_record(text: Any) -> dict[str, Any] | None:
    """A listing record written as one line ("key: value | key: value | ...") as a map, or None. At least three
    segments must read "key: value", the first segment among them, and one value must look like a listing id; a
    segment without a key continues the value before it. An attribute value of two or more "name=value" parts
    separated by ";" becomes a map of its own."""
    if not isinstance(text, str) or ' | ' not in text or len(text) > 20000:
        return None
    parts = text.split(' | ')
    first = RECORD_KEY.match(parts[0].strip())
    if not first:
        return None
    record: dict[str, Any] = {}
    key = None
    for part in parts:
        bare = part.strip()
        lead = len(bare) - len(bare.lstrip('|'))
        hit = RECORD_KEY.match(bare[lead:].strip())
        if hit and hit.group(1) not in record:
            if lead and key is not None:
                record[key] = record[key] + ' |' * lead
            key = hit.group(1)
            record[key] = hit.group(2) or ''
        elif key is not None:
            record[key] = record[key] + ' | ' + part
    if len(record) < 3 or not any((_id_like(v.strip()) for v in record.values())):
        return None
    out: dict[str, Any] = {}
    for key, value in record.items():
        facts = _fact_pairs(value) if FACT_KEYS.search(key) else None
        out[key] = facts if facts is not None else _plain(value)
    return out

def _fact_pairs(text: str) -> dict[str, Any] | None:
    """Attributes written as "name=value; name=value" as a map, or None (fewer than two pairs). A part without "=",
    or whose name is over six words, continues the value before it, so a value may hold "; " or "="."""
    pairs: list[list[str]] = []
    for part in text.split(';'):
        name, sep, item = part.partition('=')
        if sep and name.strip() and len(name.split()) <= 6:
            pairs.append([name.strip(), item])
        elif pairs:
            pairs[-1][1] += ';' + part
        elif part.strip():
            return None
    return {name: _plain(item, True) for name, item in pairs} if len(pairs) >= 2 else None

def _string_record(text: str, depth: int) -> Any:
    """What a whole-record string stands for, or None: a "key: value | ..." line (_text_record), JSON text of a map or a
    list (decoded, then read like any result), or "key=value; ..." parts with an id-like value among them."""
    record = _text_record(text)
    if record is not None:
        return record
    bare = text.strip()
    if bare[:1] in ('{', '[') and len(bare) <= 200000:
        try:
            decoded = json.loads(bare)
        except ValueError:
            decoded = None
        if isinstance(decoded, (dict, list)) and _holds_listing(decoded):
            return _with_records(decoded, depth + 1)
    if ' | ' not in bare and bare.count('=') >= 3 and len(bare) <= 20000:
        pairs = _fact_pairs(bare)
        if pairs and len(pairs) >= 3 and all((len(str(k).split()) <= 6 for k in pairs)) and any((isinstance(v, str) and _id_like(v) for v in pairs.values())):
            return pairs
    return None

def _holds_listing(node: Any, depth: int=0) -> bool:
    """Whether decoded JSON holds a listing record: a map of three or more keys with an id-like text value, itself or
    within two levels of lists and maps."""
    if depth > 2:
        return False
    if isinstance(node, dict):
        if len(node) >= 3 and any((isinstance(v, str) and _id_like(v.strip()) and re.search('[A-Za-z]', v) for v in node.values())):
            return True
        return any((_holds_listing(v, depth + 1) for v in node.values() if isinstance(v, (dict, list))))
    if isinstance(node, list):
        return any((_holds_listing(v, depth + 1) for v in node[:20] if isinstance(v, (dict, list))))
    return False

def _with_records(node: Any, depth: int=0, whole: bool=True) -> Any:
    """A tool result with every one-line listing record (_text_record) turned into a map; anything else as it was (the
    same objects when nothing changes). Only a string that stands for a whole record is read: the result itself, an
    item of a list, or a value of a small wrapper (a map of at most two keys, such as {"value": ...}); text inside a
    record's own fields (a description) is never read as a record."""
    if depth > 6:
        return node
    if isinstance(node, str):
        record = _string_record(node, depth) if whole else None
        return record if record is not None else node
    if isinstance(node, list):
        new = [_with_records(v, depth + 1, True) for v in node]
        return new if any((a is not b for a, b in zip(new, node))) else node
    if isinstance(node, dict):
        small = len(node) <= 2
        new = {k: _with_records(v, depth + 1, small) for k, v in node.items()}
        return new if any((new[k] is not node[k] for k in node)) else node
    return node

class Item:

    """Represents one product the shopper wants bought: what it is called, the store shelf it belongs on, how many
    units, and the requirements in force for it. Halo keeps one per distinct requested product, numbered from 1 in
    request order, and builds the ledger and the interview plan from these records."""
    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        """Stores the product name, its shelf (store category label, possibly empty), the quantity qty and the request's
        firm requirements musts, and starts three empty lists: told for requirements learned later from the shopper
        or store notes, dropped for requirements no longer wanted, and free for properties the shopper does not mind."""
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []

class Halo:

    """Runs one shopping episode for a single shopper. run() reads the request into a brief and per-item ledger, then
    loops the LIGHT model (SPARE as fallback) over the store's tools under the GUIDE prompt while code interviews the
    shopper, absorbs listing records and store notes, carries out steps the request mandates, reviews the cart with
    judge models, and forces a final checkout or no-purchase ending; it returns the event trace."""
    def __init__(self, problem: dict[str, Any]):
        """Sets up one episode from problem['environment']: copies the binding, keeps the policy view, the request text
        (its 'query'), the tools carrying a named function and each one's parameter schema (an empty object schema
        when absent), and max_steps (30 when missing or not positive). Starts the clock, takes finished from the
        view, and sets every counter, cart, record and interview structure to empty."""
        environment = problem['environment']
        self.binding = dict(environment['binding'])
        self.view = environment['policy_view']
        if self.view.get('opening') is not None:
            self.view = {**self.view, 'opening': _with_records(self.view.get('opening'))}
        self.request = str(self.view.get('query') or '')
        self.tools = [t for t in self.view.get('tools') or [] if isinstance(t, dict) and isinstance(t.get('function'), dict) and isinstance(t['function'].get('name'), str)]
        self.schemas = {t['function']['name']: t['function'].get('parameters') if isinstance(t['function'].get('parameters'), dict) else {'type': 'object', 'properties': {}} for t in self.tools}
        self.int_ids = any((_takes_int_ids(s) for s in self.schemas.values()))
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
        self.lane_found: dict[int, int] = {}
        self.lane_spent: dict[int, int] = {}
        self.lane_used: dict[int, set[str]] = {}
        self.late_asked = False
        self.replanned = False
        self.reply_own = ''
        self.arg_used: dict[tuple[str, str], Any] = {}
        self.filled: set[tuple[Any, tuple[str, ...]]] = set()
        self.one_product: set[int] = set()
        self.follow_up: list[tuple[int, list[str], bool]] = []
        self.listing = True
        self.line_holds = 0
        self.bound: list[tuple[int, int]] = []
        self.fit_holds = 0
        self.searched: list[dict[str, Any]] = []
        self.link_searches = 0
        self.link_seen: dict[str, list[str]] = {}

    def as_id(self, value: Any) -> str | None:
        """_as_id() for this store: a whole number is an id only where the store's ids are numbers (int_ids)."""
        if isinstance(value, int) and (not isinstance(value, bool)) and (not self.int_ids):
            return None
        return _as_id(value)

    def left(self) -> int:
        """Returns how many store steps remain in the episode: max_steps minus the steps used so far (step); it reaches
        zero or below once the step budget is spent."""
        return self.max_steps - self.step

    def clock(self) -> float:
        """Returns the seconds elapsed since this episode object was created, measured on the monotonic clock so system
        time changes do not affect it."""
        return time.monotonic() - self.started

    def model_room(self) -> float:
        """Returns the seconds still available for model calls: TASK_SECONDS minus TIME_MARGIN minus the elapsed
        clock(). It turns negative once the time budget less its safety margin is used up."""
        return TASK_SECONDS - TIME_MARGIN - self.clock()

    def msgs_free(self) -> int:
        """Returns how many shopper messages the interviewer may still send: msg_limit minus msg_used, less the messages
        still held back for order- or end-time process steps (reserve minus those the gate has already sent,
        gate_msgs, never below zero)."""
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)

    def cart_mark(self) -> str:
        """Returns a fingerprint of the cart as Halo tracks it: the (listing id, quantity) pairs of cart_lines, sorted
        and dumped as compact JSON. Paired with the change counter version, it keys which cart states the gate has
        already cleared."""
        return _dump(sorted(self.cart_lines.items()))

    def page_tools(self) -> set[str]:
        """Returns the set of tool names that show a listing's own page: the offer tool (the open tool when no offer
        role was found) and the compare tool, leaving out roles the brief did not fill."""
        page = self.roles.get('offer') or self.roles.get('open')
        return {t for t in (page, self.roles.get('compare')) if t}

    def facts_tool(self) -> str | None:
        """Names the tool for reading listing attributes: the first tool already seen returning an attribute block that
        takes a single listing argument (single_target), else the open role's tool when it is single-target and not
        known to lack attributes. Returns None when neither applies."""
        known = next((t for t, ok in self.facts_from.items() if ok and self.single_target(t)), None)
        if known:
            return known
        tool = self.roles.get('open')
        return tool if self.facts_from.get(tool or '') is not False and self.single_target(tool) else None

    def compare_facts(self) -> bool | None:
        """Returns what is known about the compare tool's attributes from facts_from: True once its results have shown
        an attribute block, False when they have shown none, and None when it has not been seen or no compare tool
        exists."""
        return self.facts_from.get(self.roles.get('compare') or '')

    def compare_bare(self) -> bool:
        """Tells whether to treat the compare tool as showing no attributes: True when its results lacked an attribute
        block, or when that is still unknown and either the attribute tool (facts_tool) is known to show them or a
        separate offer tool exists."""
        seen = self.compare_facts()
        return seen is False or (seen is None and (self.facts_from.get(self.facts_tool() or '') is True or bool(self.roles.get('offer'))))

    def gate_due(self, stage: str) -> bool:
        """Tells whether the request lists a process step for stage ('choose', 'order' or 'end') that the gate has not
        yet cleared for the current cart fingerprint (cart_mark) and change counter (version), as recorded in
        gate_ok."""
        return any((p['when'] == stage for p in self.process)) and (stage, self.cart_mark() + f'#{self.version}') not in self.gate_ok

    def tail(self) -> int:
        """Returns how many final steps to hold back: 2 when an order-time process step needs shopper messages, a
        message is still left and the order gate is due (one for that message, one for the checkout), otherwise 1."""
        talk = any((p['when'] == 'order' and p['messages'] > 0 for p in self.process))
        return 2 if talk and self.msg_used < self.msg_limit and self.gate_due('order') else 1

    def judges(self, limit: float=SPEND_SOFT) -> list[str]:
        """Returns the models allowed for a judgement at the current spend: [HEAVY, LIGHT] while cost is within limit
        (SPEND_SOFT by default) and still leaves HEAVY_ROOM under SPEND_HARD, [LIGHT] while cost is within
        SPEND_HARD, and an empty list beyond that."""
        if self.cost <= limit and self.cost + HEAVY_ROOM <= SPEND_HARD:
            return [HEAVY, LIGHT]
        return [LIGHT] if self.cost <= SPEND_HARD else []

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None=None) -> dict[str, Any] | None:
        """Tries each model in turn and returns the first assistant message, or None. Gives up once ALL_CALLS calls are
        made or model_room() falls under CALL_FLOOR (tracing 'late_call_skipped'). Retries 402, 425, 429 and 5xx up
        to len(WAITS) times and missing-status or 408 failures once, after jittered WAITS pauses within the
        episode-wide WAIT_TOTAL and the time left; a lasting 402 sets no_credit and stops."""
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
        """Sends one chat-completion request at temperature 0, offering tools with tool_choice auto, through a client
        whose timeout fits model_room(); for anthropic models with several messages it marks the first and last
        string contents as ephemeral cache points. Adds the reported cost to cost and returns (message, 200), (None,
        0) for an empty reply, or (None, status) on errors, tracing 'llm_fail'."""
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
        """Sends prompt as a single user message to each model in turn (one think call apiece) and returns the first
        reply that parses as a JSON object. Returns None when every model fails, or early once broke or no_credit is
        set or ALL_CALLS is reached."""
        for model in models:
            message = self.think([{'role': 'user', 'content': prompt}], [model], max_tokens)
            got = _json_obj((message or {}).get('content'))
            if got is not None:
                return got
            if self.broke or self.no_credit or self.calls >= ALL_CALLS:
                return None
        return None

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        """Posts one store tool call (binding, next step number, action) to /environment/call and returns the response
        dict, updating step and finished from it; each argument value sent is kept per tool (arg_used) for
        with_required(); when a page tool's result carries no error, its string and whole-number arguments, in text
        form, join opened. Resyncs the step on 409 clashes (marking finished when the episode is over), returns None
        on 400-404 or 422 (402 sets broke), and gives up after four attempts."""
        payload = dict(self.binding)
        payload['step'] = self.step + 1
        payload['action'] = {'name': name, 'args': args}
        for key, value in (args or {}).items():
            self.arg_used[(name, key)] = value
        for _ in range(4):
            result = _env.post_verbose('/environment/call', payload)
            if isinstance(result.data, dict):
                data = result.data
                if 'observation' in data:
                    data['observation'] = _with_records(data['observation'])
                self.step = data['step'] if isinstance(data.get('step'), int) else self.step + 1
                if data.get('finished'):
                    self.finished = True
                seen = data.get('observation')
                if name in self.page_tools() and (not (isinstance(seen, dict) and seen.get('error'))):
                    for value in args.values():
                        self.opened.update((self.as_id(v) for v in (value if isinstance(value, list) else [value]) if self.as_id(v) is not None))
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
        """Reads the request and tools into the brief via the BRIEF prompt (LIGHT; HEAVY again when items or a buy role
        are missing or two items share a shelf), then sets roles (check_roles(): the shopper messages never go to the
        checkout or decline tool), distinct buy and refuse mode arguments, Items with same-shelf twins merged, fit
        pairs (bound), lookups (refs, then unmix), budget, message limit (3 by default with an ask tool), a tighter
        step limit, process steps and the message reserve. In order: One model call reads the request together with
        every tool's name, description and schema, so the brief can name the exact tool that does each job (roles) as
        well as the items, budget and limits. A brief without items, roles or a checkout tool is asked again of the
        heavy model; one that gives two items the same store shelf is re-read by the heavy model too, and kept only
        if that reading is complete. When one tool both checks out and declines (a mode argument), the two fixed
        argument sets must differ; a short model call reads them from that tool's schema when the brief left them
        equal or empty. Two items on the same shelf whose requirements never give the same property different values
        are one product named twice: they are merged, and 'became' maps old item numbers to the merged list.
        Requirements whose value has to be read from another listing (lookups) are recorded in refs under the item
        that holds them, matched by property name when the brief copied the text loosely. No message limit in the
        request but a messaging tool present: three messages are assumed. A step limit from the request applies only
        between 10 and the store's own maximum. Messages the request's own steps need at the order or at an ending
        (an approval, a statement) are reserved, so the interview does not spend them."""
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
        self.check_roles()
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
        """Tells whether two or more items in a brief name the same non-empty shelf label, compared after _words
        normalization; read_request then asks HEAVY for a second reading of the request."""
        shelves = [_words(str(e.get('shelf') or '')) for e in _items(brief.get('items')) if isinstance(e, dict)]
        shelves = [s for s in shelves if s]
        return len(shelves) != len(set(shelves))

    def unmix(self) -> None:
        """Separates words that only describe a pointed-at listing from an item's own requirements: for each item whose
        musts hold both a requirement from refs and others, asks HEAVY (then LIGHT) with UNMIX which others merely
        describe that listing, moves them into the first pointing requirement as '(that listing: ...)', updates refs
        to match and logs an 'unmix' trace entry."""
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
        """Tells whether a requirement is in the shopper's own words that no listing attribute states: it must not be a
        lookup, its value needs four or more words without listing-reference words or another item's name, and at
        least three opened records of the item's category must carry an attribute block, none of whose keys shares a
        stem with the property."""
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
        """Builds the ledger text, one '- ' line per entry: each item's name, store label, quantity and live
        requirements (musts and told minus dropped, with values resolved from pointed-at pages), listing those
        described() flags apart; then budget with unit and scope, the pick rule, in-stock-only, wishes, rules, a
        quiet-decline note, process steps with their timing and, once known, which tool shows listing attributes."""
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
            when = {'choose': 'before choosing', 'order': 'at the order (Halo does it when you check out)', 'end': 'at an ending without a purchase (Halo does it then)'}[entry['when']]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        facts = self.facts_tool()
        if facts and (self.facts_from.get(facts) is True or self.roles.get('offer')):
            bare = sorted((t for t in self.page_tools() | {self.roles.get('offer') or ''} if t and self.facts_from.get(t) is False))
            pages = sorted((t for t in self.page_tools() if t != facts))
            parts.append(f'listing attributes come from {facts}' + (f"; {', '.join(bare)} show none" if bare else '') + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ''))
        return '\n'.join(('- ' + p for p in parts))

    def status(self) -> str:
        """Builds the status text Halo adds to each turn: steps used and left; one warning chosen by what remains
        (within tail()+1 or 6 steps, or before any checkout attempt within item count plus 5 or 12 steps, or half
        gone with the cart untouched); a time warning past WIND_DOWN; unresolved lookups and the fit plan before any
        checkout; review candidates (leads); the last four change notes; the ledger."""
        left = self.left()
        text = f'[Halo] step {self.step} of {self.max_steps} used, {left} left.'
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
        """Returns the listing id of a dict record with at least three keys, or None; a whole-number id comes back in
        text form ("1003") and only where the store's ids are numbers (int_ids, through as_id). The learned id_key is
        used when its value looks like an id; otherwise, only for a node at depth 0, one flagged as a row-list entry
        or one holding a nested dict, _learn_id guesses a key, accepted only when its name contains "id" or its value
        is already a known row id. A map holding a list of records keyed the same way is a container (a search result
        with its own id), never a listing, when that id is a whole number."""
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if self.id_key and _id_like(node.get(self.id_key), True, self.int_ids) and (not (isinstance(node.get(self.id_key), int) and _rows_inside(node, self.id_key))):
            return self.as_id(node[self.id_key])
        nested = any((isinstance(v, dict) for v in node.values()))
        if not (in_list or depth == 0 or nested):
            return None
        found = _learn_id([node], self.int_ids)
        if found and isinstance(node[found], int) and _rows_inside(node, found):
            found = None
        if found and ('id' in str(found).lower() or self.as_id(node[found]) in self.rows):
            return self.as_id(node[found])
        return None

    def learn_rows(self, node: list[Any]) -> bool:
        """Decides whether a list holds listing rows; at least half its entries must be dicts with three or more keys.
        Returns True when id_key gives an id-like value in every such dict, or when _learn_id (whole numbers only
        where int_ids) finds a key with distinct id-like values across them, which then becomes id_key; otherwise
        returns False."""
        rows = [n for n in node if isinstance(n, dict) and len(n) >= 3]
        if not rows or len(rows) < len(node) / 2:
            return False
        if self.id_key and all((_id_like(r.get(self.id_key), True, self.int_ids) for r in rows)):
            return True
        found = _learn_id(rows, self.int_ids)
        if found:
            self.id_key = found
            return True
        return False

    def absorb(self, observation: Any) -> None:

        """Merges every listing record found in a tool result into the agent's state: new ids join seen_order, fields
        are merged into rows, a field turning from True to False marks the id unavailable (False to True clears it),
        a record carrying a fact block is copied into full, and a later view without one updates an existing full
        record."""
        def walk(node: Any, depth: int, in_list: bool=False) -> None:
            """Visits one node of the result, giving up past depth 6. A dict that ident() names as a listing is merged
            as absorb describes and not searched further; any other dict has its values walked one level deeper. A
            list is first tested with learn_rows, whose answer is passed to its entries as in_list."""
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
        """Returns the set of every string enum option in the tools' parameter schemas, stripped and lower-cased;
        category_key compares row values with it to spot category labels. Each schema is walked through nested dict
        values down to depth 6, so options inside lists of schemas (anyOf and the like) are not collected."""
        found: set[str] = set()

        def walk(schema: Any, depth: int) -> None:
            """Adds the stripped, lower-cased string options of one schema node's enum to the found set, then recurses
            into each value that is itself a dict; returns at once for a non-dict or past depth 6."""
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
        """Guesses which row field holds the category label. Counts short string values (1 to 40 characters, no run of
        three digits) per key across rows; a key qualifies when at least 80% of rows carry one and it has at most 40
        distinct values. Prefers the largest share of distinct values found in labels(), then fewer distinct values,
        wider presence and longer values; None when nothing qualifies."""
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
        """Builds the interview's attribute-key table from the fact blocks of fully opened listings, grouped by category
        label ('listings' without a category key), keeping groups that match a requested item when any do. Drops keys
        with long, mostly ' / '-listed or free-text values, merges twins(), and returns up to 14 (label, plain key
        name, listings carrying it, first four values) tuples per group, biggest group and key first."""
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
        """Returns ' [not the stated P: another property]' for a key-table key whose name shares a word with the
        property P of one of the owning item's request musts without being that name, when the must has a plain value
        (no negation, limit or "or" wording) whose words are not all among the key's sample values. Returns '' when
        owner_of finds no item or no must qualifies."""
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
        """Decides whether two key-table keys are one property under two names; the names must share a word. When one
        name's words contain the other's, more than half the listings carrying both must agree (with none shared, the
        value lists must have a value in common); otherwise at least two shared listings are needed and all must
        agree."""
        a, b = (set(name.split()), set(other.split()))
        if not a or not b or (not a & b):
            return False

        def agree(x: str, y: str) -> bool:
            """Returns True when two values are equal, or when the shorter one's words (at least one) all appear in the
            longer one and the longer has no more than twice as many words plus two."""
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
        """Groups every seen listing id, in first-seen order, by category label (one group when no category key is
        known), keeps the groups whose label matches a requested item (equal to its shelf or name in words, sharing a
        word of four or more letters, or owner_of), or all groups when none does, and returns the largest of them, no
        more groups than there are items."""
        cat_key = self.category_key()
        groups: dict[str, list[str]] = {}
        for ident in self.seen_order:
            label = str(self.rows[ident].get(cat_key) or '') if cat_key else ''
            groups.setdefault(label, []).append(ident)
        wanted = [(_words(item.shelf or ''), _words(item.name), {w for w in _stem(item.shelf or '') | _stem(item.name) if len(w) >= 4}) for item in self.items]
        matched = [ids for label, ids in groups.items() if any((_words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long for shelf, name, long in wanted)) or self.owner_of(label)]
        pool = matched or list(groups.values())
        return sorted(pool, key=lambda ids: -len(ids))[:max(1, len(self.items))]

    def check_roles(self) -> None:
        """Keeps the shopper messages off a closing tool: when the brief names the checkout or decline tool for "ask" too,
        "ask" goes to the one tool left without a role whose required arguments are all free text, or is dropped (no
        interview, rather than a question sent through a checkout or a decline)."""
        ask = self.roles.get('ask')
        if not ask or ask not in {self.roles.get('buy'), self.roles.get('refuse')}:
            return
        taken = set(self.roles.values())
        spare = []
        for name, schema in self.schemas.items():
            props = (schema or {}).get('properties') or {}
            need = (schema or {}).get('required') or []
            texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
            if name not in taken and need and all((k in texts for k in need)):
                spare.append(name)
        self.trace.append({'kind': 'ask_on_closing_tool', 'was': ask, 'spare': spare[:4]})
        if len(spare) == 1:
            self.roles['ask'] = spare[0]
        else:
            self.roles.pop('ask', None)

    def fill_value(self, name: str | None, key: str, spec: Any) -> Any:
        """A safe value for required parameter `key` of tool `name` that a call built by Halo's code leaves out, or None
        (see with_required()). Choices and other switches are never guessed on the cart, checkout or decline tools."""
        if not isinstance(spec, dict):
            return None
        kinds = _kinds(spec)
        words = _words(key)
        about = words + ' ' + str(spec.get('description') or '').lower()
        acting = name in {self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse')} - {None}
        used = self.arg_used.get((name or '', key))
        options = spec.get('enum') if isinstance(spec.get('enum'), list) and spec['enum'] else None
        if options:
            if acting:
                return None
            named = [o for o in options if isinstance(o, str) and CHOICE_DEFAULT.search(_words(o))]
            if named:
                return named[0]
            return used if used in options else options[0]
        if 'boolean' in kinds:
            if NOOP_FLAG.search(about):
                return False
            if CONFIRM_FLAG.search(words):
                return True
            return None
        if 'integer' in kinds or 'number' in kinds:
            low = spec.get('minimum') if isinstance(spec.get('minimum'), (int, float)) and (not isinstance(spec.get('minimum'), bool)) else None
            high = spec.get('maximum') if isinstance(spec.get('maximum'), (int, float)) and (not isinstance(spec.get('maximum'), bool)) else None
            if NOT_SIZE.search(words):
                return None
            if SIZE_ARG.search(words) and name != self.roles.get('cart'):
                size = int(min(20, high)) if high is not None else 20
                return int(max(size, low)) if low is not None else size
            if PAGE_ARG.search(words):
                start = 1 if re.search('page|leaf', words) else 0
                return int(max(start, low)) if low is not None else start
            return None
        if 'string' in kinds:
            if name == self.roles.get('buy') and NOTE_ARG.search(words):
                return 'Placing the order for the items chosen.'[:spec['maxLength'] if isinstance(spec.get('maxLength'), int) else 200]
            if not acting and name != self.roles.get('ask') and isinstance(used, str) and used.strip():
                return used
        return None

    def with_required(self, name: str | None, args: dict[str, Any]) -> dict[str, Any]:
        """`args` of a call Halo's own code built for tool `name`, plus a value for each required parameter it leaves out
        where fill_value() has a safe one: a reading tool's choice (the option naming all / full / complete / default,
        else the loop model's last value), a preview-only switch set to false, a confirm switch set to true, a page size
        of 20 or the first page, the checkout's note. Anything else stays missing and the call is skipped as before."""
        schema = self.schemas.get(name or '') or {}
        props = schema.get('properties') if isinstance(schema.get('properties'), dict) else {}
        missing = [k for k in schema.get('required') or [] if k not in args and k in props]
        if not missing:
            return args
        out = dict(args)
        for key in missing:
            value = self.fill_value(name, key, props.get(key))
            if value is not None:
                out[key] = value
        added = sorted((k for k in out if k not in args))
        if added and (name, tuple(added)) not in self.filled:
            self.filled.add((name, tuple(added)))
            self.trace.append({'kind': 'filled_required', 'tool': name, 'keys': added})
        return out

    def fillable(self, tool: str | None, key: str) -> bool:
        """Whether required parameter `key` of a page or compare tool (open, offer, compare) can be filled by
        with_required(), so the tool still counts as one that reads listings by id."""
        if not tool or tool not in {self.roles.get('open'), self.roles.get('offer'), self.roles.get('compare')}:
            return False
        props = (self.schemas.get(tool) or {}).get('properties') or {}
        return self.fill_value(tool, key, props.get(key)) is not None

    def single_target(self, tool: str | None) -> str | None:
        """Returns the argument through which a tool takes a single listing id: the first required free string parameter
        (string type, no enum), or the only such parameter when none is required; a tool with no free string
        parameter offers its whole-number parameters named as ids (_int_ids) instead. Returns None for a missing
        tool, when no such parameter exists, or when the schema requires another argument that with_required() cannot
        fill (fillable(): only a page or compare tool's choice, preview switch, page size or page number can be
        filled)."""
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)] or _int_ids(props)
        target = next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not tool or not target or any((k != target and (not self.fillable(tool, k)) for k in schema.get('required') or [])):
            return None
        return target

    def price_of(self, ident: str) -> float | None:
        """Returns a listing's price as a float from its merged row under price_key: a number, a plain numeric string,
        or the first number inside a nested price dict. Returns None when price_key is unknown, the listing is
        unseen, or the value is not numeric (booleans do not count)."""
        value = (self.rows.get(ident) or {}).get(self.price_key) if self.price_key else None
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and (not isinstance(v, bool))), None)
        if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', value):
            value = float(value)
        return float(value) if isinstance(value, (int, float)) and (not isinstance(value, bool)) else None

    def buyable(self, ident: str) -> bool:
        """Returns False for a listing that sold_out() reports (in unavailable or gone) or whose row value under
        stock_key is False or a number of zero or less. A missing value, any other value or an unknown stock_key
        counts as buyable."""
        if self.sold_out(ident):
            return False
        value = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        return not (value is False or (isinstance(value, (int, float)) and (not isinstance(value, bool)) and (value <= 0)))

    def records_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        """Returns (id, record) pairs for the listings a tool result shows: the result itself when ident() names it;
        else the dicts ident() accepts inside its top-level lists and inside lists one dict deeper; for a list
        result, its entries. Gives an empty list when none is recognised, and changes no state."""
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
        """Remembers which fields a viewing tool shows. For each listing records_in finds in the result, adds the
        record's keys to tool_keys[name] and to the listing's view_keys set for the current version, starting a fresh
        set when the version changed since its last recorded view; shown_already reads both to skip repeat views."""
        for ident, record in self.records_in(observation):
            keys = set(record)
            self.tool_keys.setdefault(name, set()).update(keys)
            version, have = self.view_keys.get(ident, (-1, set()))
            self.view_keys[ident] = (self.version, (have if version == self.version else set()) | keys)

    def shown_already(self, name: str, ident: str) -> bool:
        """Returns True when calling tool name on listing ident would show nothing new: the tool has shown keys before,
        every key it has ever shown is already in the listing's view_keys set for the current version, and the
        listing is not in gone."""
        version, have = self.view_keys.get(ident, (-1, set()))
        keys = self.tool_keys.get(name)
        return bool(keys) and version == self.version and (keys <= have) and (ident not in self.gone)

    def read_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        """Learns, once per tool, how a list result is laid out: for a dict result with a list of two or more row dicts,
        asks LIGHT (HEAVY as fallback) with SHAPE for the rows, count, next-page, price and stock keys, keeping only
        names that exist (count must hold an integer). Caches the shape in shapes (None after a failed reading),
        fills unset price_key and stock_key, and returns it or None."""
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
            """Returns key when it is a string present in where (the result, its first row or the tool's parameters),
            else None, so a key named by the model counts only if it really exists."""
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
        """Returns (total matched, rows shown, next-page parameter, next-page value) for a search result, from the
        tool's shape learned by read_shape; the value is None when the store gives no next page. Returns None when
        the shape lacks a count, next key or next parameter, the rows are missing or empty, the count is not an
        integer, or the value is neither string nor integer."""
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
        """Returns True when a result shows at least three listings, all with a known price (read through price_of from
        rows), in non-decreasing order of price; otherwise False."""
        prices = [self.price_of(i) for i, _ in self.records_in(observation)]
        return len(prices) >= 3 and None not in prices and all((a <= b for a, b in zip(prices, prices[1:])))

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """Reads the next pages of a first-page search result itself (clearing paged_obs first) when the pick is
        cheapest, most or least and more listings matched than were shown, unless a cheapest search is already
        price-sorted: at most two pages, within AUTO_PAGES and keeping 12 steps beyond tail(), each observed and kept
        in paged_obs and read_cache. Returns their text, a hint to narrow or page on when it may not, or ''. In
        order: At most two pages are read at once while AUTO_PAGES allows and at least 12 steps would remain; a
        longer result is left to the loop with a note on how to page on or narrow it."""
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
            return f"\n[Halo] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with {param}={_dump(value)}."
        texts = []
        for _ in range(pages):
            faults: list[str] = []
            call = _shape(self.with_required(name, {**args, param: value}), self.schemas.get(name), 'args', faults)
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
        rest = f"\n[Halo] {count} listings matched and Halo has read {shown * (len(texts) + 1)} rows of them; narrow the search with the requirement's value or page on to see the rest." if value is not None and texts else ''
        return '\n[Halo read the next pages of this search itself, one step a page]\n' + '\n'.join(texts) + rest if texts else ''

    def cap_ignored(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """Warns when a search ignored a price ceiling it was given: takes the smallest numeric argument whose name
        suggests a maximum (max, most, under, below, limit, cap, up to) and that priced() ties to price, then counts
        result rows priced above it. Returns a note with that count, hinting at another option for each enum argument
        used; '' without such a cap, a price_key or a row above it."""
        props = (self.schemas.get(name) or {}).get('properties') or {}

        def priced(key: str) -> bool:
            """Returns True when a parameter's name or schema description mentions price, cost, amount, budget, spend or
            money, ignoring case; this marks the numeric caps that limit price."""
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
        return f'\n[Halo] This search did not apply {caps[0]}={cap:g}: {len(over)} of its rows cost more{hint}.'

    def cheapest_here(self, observations: list[Any]) -> str:
        """Returns, for a cheapest pick with a known price_key, a line naming up to six buyable, priced listings found
        across the given results (a search and the pages Halo read after it) with their prices, cheapest first.
        Returns '' for other picks or when fewer than two such listings appear."""
        if self.brief.get('pick') != 'cheapest' or not self.price_key:
            return ''
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ''
        return '\n[Halo] buyable rows here, cheapest first: ' + ', '.join((f'{i} ({p:g})' for p, i in sorted(rows)[:6]))

    def wider_hold(self) -> str:
        """Holds a cheapest-pick checkout while wider_held is unset (take_note clears it after some changes) and at
        least 14 steps plus two per item remain: sets wider_held and returns a no-step reply asking for more searches
        under other product names and a look at buyable rows priced below the cart lines, listed with their prices.
        Returns '' otherwise."""
        if self.wider_held or self.brief.get('pick') != 'cheapest' or self.left() < 14 + 2 * len(self.items):
            return ''
        self.wider_held = True
        lines = [f'{i} at {self.price_of(i):g}' for i in self.cart_ids() if self.price_of(i) is not None]
        return f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest. Searches may read only part of a listing, so a requirement's value finds only some of the fitting listings, and rows come in no price order. Before checking out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line" + (f" ({'; '.join(lines)})" if lines else '') + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        """Opens, one step each, the given listings that are known rows not yet fully read and not sold out, through
        facts_tool() and its single id argument, while more than keep steps remain, the episode runs and that tool
        has not proved to show no attributes. Each result is observed, passed to note_view and used to update
        facts_from. Returns 'id: result' texts cut to 1500 characters; [] without a usable tool."""
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return []
        texts = []
        for ident in [i for i in ids if i in self.rows and i not in self.full and (not self.sold_out(i))]:
            if self.left() <= keep or self.finished or self.facts_from.get(tool) is False:
                break
            faults: list[str] = []
            args = _shape(self.with_required(tool, {target: ident}), self.schemas.get(tool), 'args', faults)
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
        """Follows "see part X" style pointers on the given listings: for each pointed part code not tried yet and not
        the own code of any seen listing, searches the code with value_template()'s search (one step), then opens the
        listings carrying it as their own code, keeping max(keep, tail() + 6) steps. Does nothing without a template,
        or with one item and fewer than two ids; stops at keep steps left."""
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
            args = _shape(self.with_required(tool, {**base, field: code}), self.schemas.get(tool), 'args', faults)
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
        """Finds a search tool that lists a category under a price cap: the first schema with exactly one numeric cap
        parameter (CAP_ARG, not CAP_NOT) and exactly one free string category parameter (SHELF_ARG). Enums of two or
        more options copy the newest search that set the category without query text, else take their only LIST_MODE
        option. Returns (tool, fixed args, category, cap), or None when none qualifies or an enum stays open."""
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
        """Lists each cart line's category below its price via cap_template()'s tool, once a category key is known: one
        observed step per line that costs 2 or more, has a category label and was not yet listed at that price,
        capped at the price minus one when whole, else rounded down. Reads one more page if the rest fits on it and
        rows are not price-sorted. Stops at tail() + 5 steps left."""
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
            args = _shape(self.with_required(tool, {**base, shelf: label, cap: int(price) - 1 if float(price).is_integer() else int(price)}), self.schemas.get(tool), 'args', faults)
            if faults:
                return
            reply = self.act(tool, args)
            if reply is None:
                return
            self.observe(tool, reply.get('observation'))
            self.trace.append({'kind': 'list_below', 'line': line, 'price': price})
            page = self.paging(tool, reply.get('observation'))
            if page and page[3] is not None and page[1] < page[0] <= 2 * page[1] and self.left() > self.tail() + 5 and not self.price_sorted(reply.get('observation')):
                call = _shape(self.with_required(tool, {**args, page[2]: page[3]}), self.schemas.get(tool), 'args', faults)
                more = None if faults else self.act(tool, call)
                if more is not None:
                    self.observe(tool, more.get('observation'))

    def cheaper_rows(self) -> list[str]:
        """Hunts for cheaper fits before a cheapest-pick checkout, skipping a cart mark and version already in
        cheaper_marks: runs list_below on the cart, then opens with fill_facts the two cheapest unopened, buyable
        seen listings priced below each line in its category, at most min(6, CHEAPER_OPENS times the line count),
        keeping tail() + 4 steps. Returns the opened texts; [] for other picks, a repeated mark or no category key."""
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
        """Handles the result of a call Halo made itself: absorbs its listing records, then passes any unread free-text
        lines found by loose_lines to take_note. Queues the note's meaning, tagged with the tool name, for the next
        summary and returns it; returns '' when there is no new line or take_note reports nothing."""
        self.absorb(observation)
        lines = self.loose_lines(observation)
        if not lines:
            return ''
        meaning = self.take_note(lines)
        if meaning:
            self.queued.append(f'(seen on {name}) {meaning}')
        return meaning

    def widen(self) -> None:
        """Opens more listings before the interview so their attribute keys are known. With a compare tool taking one
        array argument (anything else it requires fillable by with_required()) and compare_bare() false, sends one
        side-by-side call of two or more unopened ids spread over item_groups(), up to maxItems (default 4), when
        more than 12 steps remain, retrying with two ids if nothing new opened. Otherwise, or when compare shows no
        attributes, calls widen_singly()."""
        tool = self.roles.get('compare')
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in _kinds(v)), None)
        if not tool or not target or any((k != target and (not self.fillable(tool, k)) for k in schema.get('required') or [])) or self.compare_bare():
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
        args = _shape(self.with_required(tool, {target: picks}), schema, 'args', faults)
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
                args = _shape(self.with_required(tool, {target: picks[:2]}), schema, 'args', faults)
                if not faults:
                    reply = self.act(tool, args)
                    if reply is not None:
                        self.observe(tool, reply.get('observation'))

    def widen_singly(self) -> None:
        """Opens listings one by one with facts_tool(), aiming at WIDEN_EACH opened records per item group (4 with three
        or more items). Unopened buyable ids with the most requirement-value words in their row go first (cheaper
        first among equals for a cheapest pick); a round-robin round brings each group toward four opened (at most 8
        opens, keeping 22 steps), then the rest follow, up to ten opens in all, keeping 30 steps."""
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
        """Returns True when the interview should start now: it has not run, a messaging tool exists, msgs_free() is
        positive, more than 8 steps remain, the clock is within ASK_BY, and either full holds at least three records
        per item plus one or a quarter of max_steps (at least 8) are used. At the latest, once ASK_SOON seconds have
        passed without an interview, it says yes as well, so a slow episode still asks before ASK_BY."""
        if self.asked or not self.roles.get('ask') or self.msgs_free() <= 0:
            return False
        if self.left() <= 8 or self.clock() > ASK_BY:
            return False
        opened = len(self.full)
        if opened >= 3 * len(self.items) + 1:
            return True
        return self.step >= max(8, self.max_steps // 4) or self.clock() >= ASK_SOON

    def message_args(self, text: str) -> dict[str, Any] | None:
        """Builds the messaging tool's arguments for one message: text goes into the first required free string
        parameter (else the first such parameter) and each other required parameter gets its first enum option, or
        the safe value fill_value() gives it (a preview-only switch false, a confirm switch true, a page size or page
        number). Returns None when there is no text parameter, a required one has neither, or _shape reports a fault;
        a maxLength in the schema may cut the text."""
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
                    value = self.fill_value(name, key, spec)
                    if value is None:
                        return None
                    args[key] = value
        faults: list[str] = []
        args = _shape(args, schema, 'args', faults)
        return None if faults else args

    def interview(self) -> str:
        """Runs the shopper interview once (sets asked; '' when no message is free): calls widen(), plans question lanes
        with plan_lanes, and sends questions via ask_next while messages are free, more than 6 steps remain and the
        clock is under QUESTION_BY. Then runs value_search for items that gained needs, bumps version when
        requirements changed, and returns a summary of the exchanges, new requirements, searches and queued notes,
        clearing queued. In order: Questions continue while messages, steps (more than six) and time (before
        QUESTION_BY) remain and the plan still has labels to ask. When the lanes run dry while a message is still
        free, the episode is open, more than six steps remain and the clock is under REPLAN_BY, extend_lanes() plans
        once more (naming the properties already asked) and the questions go on; at most once per episode."""
        self.asked = True
        if self.msgs_free() <= 0:
            return ''
        self.widen()
        count = self.msg_limit - self.msg_used
        self.lanes = self.plan_lanes(max(count, LIST_SIZE) if self.listing else count)
        self.lane_found = {n: 0 for n in self.lanes}
        self.lane_spent = {n: 0 for n in self.lanes}
        self.lane_used = {n: set() for n in self.lanes}
        self.follow_up = []
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        while self.msgs_free() > 0 and (not self.finished) and (self.left() > 6) and (self.clock() < QUESTION_BY):
            batch = self.ask_next(told)
            if not batch:
                if self.replanned or self.msgs_free() <= 0 or self.finished or self.left() <= 6 or self.clock() >= REPLAN_BY or (not self.extend_lanes()):
                    break
                continue
            sent += batch
        found_items = [n for n in self.lane_found if self.lane_found[n] > 0]
        searched = self.value_search(found_items) if found_items else ''
        if told:
            self.version += 1
        lines = [f'Q: {q}\nA: {a}' for _, q, a in sent]
        summary = "Halo's interviewer asked the shopper (each question used one step):\n" + '\n'.join(lines)
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
        """Asks one last question just before checkout, once (sets late_asked), when a message is still unused, no
        order-time step needs messages, at least LATE_STEPS steps remain, the clock is within LATE_BY and judges()
        still offers a model. Plans lanes if none exist and sends the next question via ask_next. When the answer
        changes a requirement or the budget, bumps version and returns a hold summary plus a value_search; else ''.
        In order: The late question needs LATE_STEPS steps, time before LATE_BY and room in the model budget, and is
        asked at most once."""
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
        summary = f'Halo asked the shopper one more question before checking out (one step):\nQ: {question}\nA: {answer}\nNew firm requirements from this answer: ' + '; '.join(told) + '. The cart was not checked out: check each line against them (search again with the new values where needed), fix the cart, then check out.'
        pointed = [need for _, need in self.refs if any((need in t for t in told))]
        if pointed:
            summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value.'
        searched = self.value_search([number])
        return summary + ('\n' + searched if searched else '')

    def search_template(self) -> tuple[str, dict[str, Any], str] | None:
        """Derives a free-text query template from recent searches: the newest search with letters in a query-named
        string parameter gives the field (without one, a free string parameter that varied, once held several words
        and never a category label). Keeps that call's enum and required arguments, minus non-required category-like
        strings, setting each enum to its most used value among worded searches. Returns (tool, base args, field), or
        None."""
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
        """Returns (value, told) pairs usable as search words for the given item's live requirements: the text after
        each need's colon, skipping lookups listed in refs, needs without a colon, and values with negation,
        comparison or limit words. A FLOOR minimum such as "at least 30 cm" yields just its amount, and only for a
        told need; told marks needs from the item's told list rather than the request."""
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
        """Returns search_template()'s (tool, fixed args, query field) plan if it has one; otherwise, given a search
        tool and a recorded search, builds one from the schema: the one free-text parameter named like QUERY_WORDS,
        the last search's required arguments and, when enums exist, one mode (the lone option worded like a query, or
        the option the last search left unused in a required two-option enum). Returns None when any part is missing
        or ambiguous."""
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
        """Searches for each given item with its requirement values joined into one query (value_template()), when 22 or
        more steps are left, using at most 8 steps on searches and opens. Items lacking a told value are skipped;
        one-word or amount told values lead. If no buyable, in-budget row of the item's category comes back, a
        shorter retry may follow, usually minus the last request-stated value. Opens up to three found rows; returns
        the report or ''. In order: Each item gets at most two searches, the told values together and then without
        the last stated value, and the first fitting-kind rows of a search that finds any are opened at once."""
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
                args = _shape(self.with_required(tool, {**base, field: ' '.join(words)}), self.schemas.get(tool), 'args', faults)
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
        return "[Halo searched each such item's requirement values together and read the first fitting-kind rows] " + '\n'.join(seen)

    def home_label(self, number: int) -> str:
        """Returns the normalised category label that best stands for the given item: it counts labels the item owns
        among opened records, or among all seen rows when none qualify, and prefers one equal to the item's shelf or
        name, else the most frequent. Returns '' when no category key is known or the item owns no label."""
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
        """Chooses the next item and labels to ask about. Items that have yielded no requirement yet come first, then
        those with fewer questions spent (not counted when messages are scarce: at most SCARCE_MESSAGES with lists
        off), then plan_order and item number. Returns the first item whose lane still has labels neither used nor
        told_about, cut to LIST_SIZE labels while lists are on and to one label once they are off, or (0, []) when
        none remain."""
        first = {n: i for i, n in enumerate(self.plan_order)}
        scarce = self.msg_limit <= SCARCE_MESSAGES and (not self.listing)
        for number in sorted(lanes, key=lambda n: (found.get(n, 0) > 0, 0 if scarce else spent.get(n, 0), first.get(n, len(first) + n))):
            left = [label for label in lanes[number] if _words(label) not in used[number] and (not self.told_about(number, label))]
            if left:
                return (number, left[:LIST_SIZE if self.listing else 1])
        return (0, [])

    def told_about(self, number: int, label: str) -> bool:
        """Returns True when one of the given item's told needs already covers this label: with plural s and -our
        spellings folded, the label's words equal the need's property words, all lie within them, or include every
        word of a property name that has two or more words."""
        mine = _spelled(label)
        return any((_names(mine, _spelled(t.split(':', 1)[0])) for t in self.items[number - 1].told))

    def ask_text(self, number: int, labels: list[str], shown: bool=False) -> str:
        """Builds the question for the given item: two or more labels make one list question, "What A, B or C would you
        like for the <item>?", with the item named by item_said(); a single label goes through question(), with
        example values from samples() when shown is True."""
        if len(labels) >= 2:
            return f"What {', '.join(labels[:-1])} or {labels[-1]} would you like for the {self.item_said(number)}?"
        return self.question(number, labels[0], self.samples(number, labels[0]) if shown else None)

    def fit_ask(self, number: int, labels: list[str], shown: bool) -> tuple[list[str], str]:
        """Finds the longest leading slice of labels whose ask_text() question the message tool accepts whole:
        message_args() must succeed and still carry the exact text, so a maxLength cut rules it out. Returns those
        labels with the text, or ([], '') when not even a single label fits."""
        for size in range(len(labels), 0, -1):
            text = self.ask_text(number, labels[:size], shown)
            args = self.message_args(text)
            if args and text in args.values():
                return (labels[:size], text)
        return ([], '')

    def ask_next(self, told: list[str], floor: int=6) -> list[tuple[int, str, str]]:
        """Asks the shopper one question and takes in the answer. Uses the first pending follow_up not yet told, else
        next_ask(), trying up to LIST_SIZE picks until fit_ask() fits one (an unfit pick's first label is marked
        used). After send(), hear() reads the reply (LIGHT first; 'free' ignored for lists), name_needs() (on the
        shopper's own words, reply_own) and take_heard() record new needs into told, lane counters update and
        settle() runs. Returns the sent batch, or []."""
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
        self.name_needs(heard, self.reply_own or batch[0][2], labels)
        self.take_heard(heard, told)
        new = [need for item, size in zip(self.items, had) for need in item.told[size:]]
        for n, (item, size) in enumerate(zip(self.items, had), 1):
            if len(item.told) > size:
                self.lane_found[n] = self.lane_found.get(n, 0) + 1
        crowded = any((_count(x) == 1 for x in _items(heard.get('crowded'))))
        self.settle(number, labels, shown, new, crowded)
        return batch

    def settle(self, number: int, labels: list[str], shown: bool, new: list[str], crowded: bool) -> None:
        """Updates lane state after a question. A single label is marked used; a crowded reply with no need queues a
        front follow-up (the item narrowed to one product, the label's last word, or examples). A crowded reply to a
        list with no need turns lists off for the episode and queues the list's first label alone. Any other list
        reply marks the labels up to the first answered one used (only the first when nothing was disclosed) and
        requeues the rest: the labels after the answered one lead; after a reply that disclosed nothing the remaining
        labels go reversed in front of the untried ones, or behind them after a second such reply in a row."""
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
        ahead = labels[hit + 1:] if hit >= 0 else labels[:0:-1]
        used.update((_words(label) for label in labels[:max(hit, 0) + 1]))
        rest = [label for label in self.lanes.get(number) or [] if label not in labels]
        self.lane_misses[number] = 0 if hit >= 0 else self.lane_misses.get(number, 0) + 1
        self.lanes[number] = rest + ahead if self.lane_misses[number] and self.lane_misses[number] % 2 == 0 else ahead + rest
        self.trace.append({'kind': 'list_settled', 'item': number, 'done': labels[:hit + 1], 'next': ahead[:4]})

    def answered_label(self, labels: list[str], new: list[str]) -> int:
        """Returns the position of the earliest label in a sent list that one of the new needs answers, or -1 when none
        matches. Each need goes to the label sharing the most folded words with its property; with no such overlap,
        to the label whose key_table example values share the most word roots with the need's value."""
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
        """Returns the attribute keys worth asking about for the given item, from key_table rows of categories it owns
        (every row for a lone item with none). Sorts by how many listings carry the key, then keys whose examples are
        not mostly numbers with short units, then more example values; drops keys a request must's property already
        names and NOT_PROPERTY keys, and puts CATCH_ALL keys last."""
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
        """Assigns category labels of opened records that no item owns (owner_of gives 0) to the only item that owns
        none of those labels outright, saving the mapping in self.adopted. Acts only with two or more items, a known
        category key, at least one such stray label and exactly one such unlabelled item."""
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
        """Returns the 1-based number of the item a category label belongs to, or 0 for an empty label or one sharing
        nothing with any item. An adopted label gives its recorded item; otherwise each item scores 3 when the
        normalised label equals its shelf, 2 when it equals its name, else the Jaccard overlap of word roots with
        shelf and name, and the first top scorer wins."""
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
        """Returns True when the label is non-empty, the item is one of self.items, and owner_of() maps the label to
        that item's 1-based position."""
        return bool(label) and item in self.items and (self.owner_of(label) == self.items.index(item) + 1)

    def records_of_item(self, number: int) -> int:
        """Counts the opened records in self.full whose category label the given item owns and that carry an attribute
        block (_fact_block); returns 0 when no category key is known."""
        cat_key = self.category_key()
        item = self.items[number - 1]
        return sum((1 for record in self.full.values() if cat_key and self.owns(item, str(record.get(cat_key) or '')) and _fact_block(record)))

    def extend_lanes(self) -> bool:
        """Plans once more when the lanes ran dry with messages left: the plan prompt also gets the properties already
        asked per item (ALREADY ASKED) and each item's lane becomes its new labels that were neither asked nor told.
        True when some item gained a label to ask."""
        self.replanned = True
        asked = {n: sorted(self.lane_used.get(n, set())) for n in range(1, len(self.items) + 1)}
        more = self.plan_lanes(max(1, self.msgs_free()) * (LIST_SIZE if self.listing else 1), asked)
        grew = False
        for number, labels in more.items():
            fresh = [x for x in labels if _words(x) not in self.lane_used.get(number, set()) and (not self.told_about(number, x))]
            if fresh:
                self.lanes[number] = fresh
                grew = True
        self.trace.append({'kind': 'replanned', 'lanes': {n: v[:8] for n, v in self.lanes.items()}, 'grew': grew})
        return grew

    def plan_lanes(self, count: int, asked: dict[int, list[str]] | None=None) -> dict[int, list[str]]:
        """Returns each item's ordered lane of properties to ask about. After adopt_labels(), asks HEAVY (then LIGHT)
        with PLAN (COUNT set to count), the request, settled musts, ask_first entries and the key_table listing.
        Picks, cleaned and respelled to the item's own keys, lead (hinted ones add their item to plan_order); up to
        GUARD_KEYS top lane_keys may take the slot of an unhinted top-three pick with a listing share GUARD_GAP or
        more lower; other lane_keys follow, CATCH_ALL last. In order: The plan prompt gets the request, what the
        request already settles for each item, and the attribute keys of opened listings per category with how often
        each is carried and example values. Each pick becomes one label in its item's lane: the label is cut at its
        first 'and', 'or' or punctuation, labels over five words are dropped, and a pick with a hint marks its item
        as asked first. Guard: the item's GUARD_KEYS most carried keys are moved into the first three places when a
        weaker, unhinted label holds such a place, so a list does not open on keys few listings carry. Every
        remaining key of the item's category follows the planned labels, and labels that bundle several features
        (CATCH_ALL) go last, so later lists still have properties to name."""
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
        if asked and any(asked.values()):
            prompt += '\n\nALREADY ASKED (the shopper has been asked about these; choose other properties):\n' + '\n'.join((f"item {n}: {', '.join(v)}" for n, v in asked.items() if v))
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
                """Returns the key in own (the item's carried attribute keys) that matches the label once both get the
                same light respelling (lower-cased words, a final s after three letters dropped, a final -our made
                -or), so planned labels take the store's spelling; returns the label itself when no key matches."""
                plain = lambda text: re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', _words(text)))
                return next((k for k in own if plain(k) == plain(label)), label)

            def share_of(label: str) -> float:
                """Returns the share of the item's opened listings that carry the label as a key: own's value when the
                label's words are a key there, else the largest share among keys whose stemmed words contain, or lie
                within, the label's, else 0.0."""
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
        """Returns the parts of the given item's name, split at commas and 'and', when it joins two or more
        plural-looking nouns (each part ending in s after three letters, not in -ss or -us), as in "plates and
        bowls"; returns [] when the name has no 'and' or a part fails that test."""
        name = self.items[number - 1].name
        if not re.search('\\band\\b', name, re.I):
            return []
        parts = [p.strip() for p in re.split('\\s*(?:,|\\band\\b)\\s*', name, flags=re.I) if p.strip()]
        return parts if len(parts) >= 2 and all((re.search('[a-z]{3}(?<![su])s$', p, re.I) for p in parts)) else []

    def item_said(self, number: int) -> str:
        """Returns how questions refer to the given item: its name, or for a compound name (name_parts) the name with
        'and' turned into 'or'. Once the item is in one_product, returns the single part whose last word's roots
        appear in the most seen rows of the item's category (their string fields besides the category), the earliest
        part winning ties."""
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
        """Returns the single-property question "What <label> would you like for the <item>?", naming the item with
        item_said(); when samples are given, it ends ", for example <first> or <second>?" instead, using the first
        two."""
        text = f'What {label} would you like for the {self.item_said(number)}'
        return text + (f', for example {samples[0]} or {samples[1]}?' if samples else '?')

    def samples(self, number: int, label: str) -> list[str]:
        """Returns up to two example values for the label from the first key_table entry whose key words equal it:
        non-empty values of at most 30 characters that do not occur within the given item's joined request musts.
        Returns [] when no entry has that key."""
        want = _words(label)
        settled = ' '.join(self.items[number - 1].musts).lower()
        for _, name, _, values in self.key_table():
            if name == want:
                return [v for v in values if v and v not in settled and (len(v) <= 30)][:2]
        return []

    def send(self, questions: list[tuple[int, str, str]], floor: int=6) -> list[tuple[int, str, str]]:
        """Sends each (item, label, text) question with the ask tool, stopping once the episode has finished, steps left
        fall to floor or below, msg_limit is reached, message_args() fails, a call returns nothing, or after a reply
        that carries an error. Each sent message counts in msg_used, is logged to done_log and done_steps, and has
        its side_notes read by take_note(); the shopper's own words of the last reply (own_words: the top-level text,
        not side notes) are kept in reply_own. Returns (item, text, reply words) triples."""
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
            self.reply_own = self.own_words(answer) or words
            sent.append((number, text, words))
            extra = self.side_notes(answer)
            meaning = self.take_note(extra) if extra else ''
            if meaning:
                self.queued.append(f'(seen with the shopper\'s reply) {meaning}')
            self.done_log.append(f'step {self.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {self.cart_mark()})')
            self.done_steps.append(self.step)
            self.trace.append({'kind': 'ask', 'q': text, 'a': words[:300]})
            if isinstance(answer, dict) and answer.get('error'):
                break
        return sent

    def take_heard(self, heard: dict[str, Any], told: list[str]) -> None:
        """Applies a parsed answer to the ledger, clamping item numbers to range. Each need that agrees() accepts joins
        its item's told list and the told report (prefixed 'item N:' when there are several items); pointing needs
        also join refs. A drop marks every live must or told need on that property dropped and reports it; free
        properties are stored on the item, and a positive budget replaces self.budget."""
        many = len(self.items) > 1
        for entry in _items(heard.get('needs')):
            if not isinstance(entry, dict) or not str(entry.get('need') or '').strip():
                continue
            number = min(max(1, _count(entry.get('item')) or 1), len(self.items))
            need = str(entry['need']).strip()
            if not self.agrees(number, need):
                continue
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

    def known_names(self, labels: list[str]) -> dict[str, str]:
        """Property names this episode has seen, folded for phrase matching (_phrase: punctuation removed, plural -s and
        -our dropped) -> plain form: attribute keys of opened listings, planned and asked labels, and the properties
        of every item's requirements; names of more than five words are left out."""
        names = [key for _, key, _, _ in self.key_table()] + list(labels)
        names += [label for lane in self.lanes.values() for label in lane]
        names += [need.split(':', 1)[0] for item in self.items for need in item.musts + item.told]
        out: dict[str, str] = {}
        for name in names:
            plain = _words(name)
            if plain and len(plain.split()) <= 5:
                out.setdefault(_phrase(plain), plain)
        return out

    def name_needs(self, heard: dict[str, Any], reply: str, labels: list[str]) -> None:
        """Renames each need the reader took from a reply to the property name the shopper's own words use (reply: the
        top-level reply, not side notes; words compared without punctuation, _phrase): the longest known name in the
        reply that holds the reader's property as a phrase, or, when the reader's property is not in the reply at
        all, the one known name the reply holds, except a name inside the reader's own property name that the
        question did not ask. Anything less clear keeps the reader's name. A reading "P: not V" from a reply that
        turns down anything other than V (_turned_down_but) becomes "P: V", unless V itself starts with a negation
        ("other than not available")."""
        said = f' {_phrase(reply)} '
        known = self.known_names(labels)
        present = [k for k in known if f' {k} ' in said]
        for entry in _items(heard.get('needs')):
            if not isinstance(entry, dict):
                continue
            prop, sep, value = str(entry.get('need') or '').partition(':')
            mine = _phrase(prop)
            if not sep or not mine:
                continue
            if f' {mine} ' in said:
                longer = [k for k in present if k != mine and f' {mine} ' in f' {k} ']
                pick = max(longer, key=len) if longer else None
            else:
                asked = {_phrase(x) for x in labels}
                tops = [k for k in present if not any(k != o and f' {k} ' in f' {o} ' for o in present) and (k in asked or not set(k.split()) <= set(mine.split()))]
                pick = tops[0] if len(tops) == 1 else None
            if pick and pick != mine:
                entry['need'] = f'{known[pick]}: {value.strip()}'
                self.trace.append({'kind': 'need_renamed', 'was': prop.strip(), 'now': known[pick]})
            negated, base = _negated(value)
            wanted = _turned_down_but(reply)
            if negated and wanted and (not _negated(wanted)[0]) and _same_value(base, wanted):
                kept = str(entry['need']).partition(':')[0]
                entry['need'] = f'{kept}: {base}'
                self.trace.append({'kind': 'double_no', 'need': entry['need'][:160]})

    def agrees(self, number: int, need: str) -> bool:
        """False for a need that excludes the very value the item already holds for that property ("P: not V" or
        "P: other than V" next to "P: V"); a plain "P: V" that arrives after such an exclusion drops the exclusion.
        Values compare by their words; the shorter one's words must all be in the other."""
        item = self.items[number - 1]
        prop, _, value = need.partition(':')
        negated, base = _negated(value)
        for held in [m for m in item.musts + item.told if m not in item.dropped]:
            other, _, held_value = held.partition(':')
            if _words(other) != _words(prop):
                continue
            held_negated, held_base = _negated(held_value)
            if negated == held_negated or not _same_value(base, held_base):
                continue
            if negated:
                self.trace.append({'kind': 'negation_skipped', 'need': need[:160], 'held': held[:160]})
                return False
            item.dropped.append(held)
            self.trace.append({'kind': 'negation_dropped', 'need': need[:160], 'held': held[:160]})
        return True

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(HEAVY, LIGHT)) -> dict[str, Any]:
        """Asks the given models in order (HEAVY then LIGHT by default) to read the sent questions and answers with the
        HEAR prompt and the numbered item names. Returns the parsed JSON object (needs, drop, free, crowded, budget),
        or {} when no model returns valid JSON."""
        prompt = HEAR + '\n\nITEMS:\n' + '\n'.join((f'item {n}: {it.name}' for n, it in enumerate(self.items, 1)))
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f'{i}. (about item {n}) Q: {q}\nA: {a}' for i, (n, q, a) in enumerate(sent, 1)))
        return self.ask_json(prompt, list(models), 2500) or {}

    def gate(self, stage: str) -> tuple[bool, str]:
        """Performs the request's process steps due at stage ('order' or 'end') unless gate_ok already covers this cart
        and version. Over up to GATE_ROUNDS model rounds (GATE prompt, LIGHT first) it sends the named message via
        send() or runs the named non-ending tool, logging to done_log, and nudges once, HEAVY-first, when no record
        proves a step. Returns (False, text) when a reply changes the task, else (True, a report or ''); a no-call
        verdict adds gate_ok. In order: Up to GATE_ROUNDS rounds: a model call reads the record of messages and calls
        against the steps due at this stage, and either names the one store call that does a missing step or finds
        them all done. A reply that adds a requirement or asks for a change before going on stops the checkout or
        ending: the caller is told what changed so the cart can be fixed first. A step marked done with no record
        line after the cart last changed is not taken on trust once: the record gets a note saying so and the gate
        asks again, the heavy model first this time. A message to the shopper goes through send(), which counts it
        against the limit; any other call is made directly and logged in the record so later rounds can see it."""
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
                return (False, ('Before going on, Halo did the steps the request asks for:\n' + '\n'.join(said) + '\n' if said else '') + ("The shopper's reply changes the task: " + '; '.join(told + ([hold] if hold else [])) + '. Fix the cart for it, then check out again.'))
            name = str((call or {}).get('tool') or '')
            shown = [e for e in _items(got.get('steps')) if isinstance(e, dict)]
            talk = stage == 'order' and any((p['messages'] > 0 for p in steps))
            lines = range(1, len(self.done_log) + 1)
            proof = {i for i in lines if not talk or (i <= len(self.done_steps) and self.done_steps[i - 1] > self.cart_at)}
            unproven = [e for e in shown if e.get('done') is not True or not any(((_count(n) or 0) in proof for n in _items(e.get('lines'))))]
            if not call and unproven and (not said) and (not nudged) and self.roles.get('ask') and (self.msg_used < self.msg_limit) and (self.left() >= 2):
                nudged = True
                self.trace.append({'kind': 'gate_unproven', 'steps': _clip(unproven, 120)})
                self.done_log.append(f'(Halo: no line made since the cart last changed at step {self.cart_at} shows ' + '; '.join((str(e.get('step') or '') for e in unproven))[:200] + f' done, and {self.msg_limit - self.msg_used} message(s) and {self.left()} steps are left: give the call that does it)')
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
        return (True, 'Before going on, Halo did the steps the request asks for:\n' + '\n'.join(said) if said else '')

    def side_notes(self, observation: Any) -> list[str]:
        """The notes a message tool's result carries beside the reply: strings of five or more words inside its lists
        and nested records (the reply itself is a top-level string), not read before."""
        if not isinstance(observation, dict):
            return []
        found: list[str] = []

        def walk(node: Any, depth: int) -> None:
            """Visits list entries and dict values recursively down to depth 4, appending to found each string of five
            or more words that is not already in self.heard or in found."""
            if depth > 4:
                return
            if isinstance(node, str):
                if len(node.split()) >= 5 and node not in self.heard and node not in found:
                    found.append(node)
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1)
            elif isinstance(node, dict):
                for value in node.values():
                    walk(value, depth + 1)
        for value in observation.values():
            if isinstance(value, (list, dict)):
                walk(value, 1)
        return found

    def own_words(self, observation: Any) -> str:
        """The shopper's own reply in a message tool's result: the result itself when it is text, else its top-level texts
        of two or more words (side notes sit in nested lists and records); '' when there are none."""
        if isinstance(observation, str):
            return observation
        if isinstance(observation, dict):
            return ' '.join((v for v in observation.values() if isinstance(v, str) and len(v.split()) >= 2))[:800]
        return ''

    def spoken(self, observation: Any) -> str:
        """Returns the words of a tool result such as the shopper's reply: the result itself when it is a string, else
        every string of two or more words found down to depth 4, joined with spaces and cut to 800 characters; when
        none is found, the result's compact JSON cut to 400 characters."""
        if isinstance(observation, str):
            return observation
        found: list[str] = []

        def walk(node: Any, depth: int) -> None:
            """Visits dict values and list entries recursively down to depth 4, appending to found, in visiting order,
            every string of two or more words; other values are ignored."""
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
        """Collects note-like text from a tool result that is a dict but not itself a listing record: top-level strings
        of five or more words, such strings from all-string lists, and strings of six or more words directly inside
        nested non-listing dicts. Returns those not already in self.heard, or [] for any other result."""
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
        """Sends note lines to LIGHT (then SPARE) with the NOTE prompt and ledger, adding them to self.heard. Gone ids
        among known rows become unavailable; note_sold, note_named and sold_lines are updated. Unless the note is
        classed as non-change, it applies a new budget, drops and adds (into told), resets wider_held and
        cheaper_marks for a cheapest pick after drops, sell-outs or budget changes, and logs to notes with a version
        bump. Returns the summary plus changes, or ''. In order: Only a note read as a change touches the ledger; an
        error or other note returns nothing, though listings it names as gone are still marked unavailable above. A
        dropped requirement is matched in full first and by its property name otherwise; an added one goes to the
        item's told needs unless it is already held, and is taken out of dropped. With a cheapest pick, anything that
        can change which listing is cheapest (a drop, a sell-out, a budget change) re-arms the wider search and the
        cheaper-row checks before the next checkout."""
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
        """Returns the requirements in force as (item number, need) pairs, item by item: the request's musts first, then
        told needs, leaving out any in the item's dropped list."""
        out = []
        for number, item in enumerate(self.items, 1):
            for need in item.musts + item.told:
                if need not in item.dropped:
                    out.append((number, need))
        return out

    def evidence(self, observation: Any) -> str:
        """Builds the [evidence] line for a tool result: for up to 12 listings the result shows with an attribute block,
        and each of the first eight requirements in force, lists up to three keys of the stored record whose stemmed
        names match the requirement's property (values cut to 50 characters), tagged by item when there are several.
        Returns '' when there is no such listing, requirement or matching key."""
        ids: list[str] = []

        def walk(node: Any, depth: int) -> None:
            """Walks the result's dicts and lists down to depth 6, appending to ids the id of each dict that ident()
            reads as a listing and that carries an attribute block (_fact_block), without descending into a newly
            collected listing."""
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
        """Returns the cart as text for prompts and replies: the last cart view as compact JSON (strings clipped to 300
        characters, the whole to 3000) when it was taken at or after the latest cart change, else Halo's cart_lines
        record of listing ids and units from its own cart calls."""
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            return _dump(_clip(self.cart_view, 300))[:3000]
        return "lines by Halo's record of the cart calls (listing: units): " + _dump(self.cart_lines)

    def held_units(self, ident: str) -> int:
        """Returns the units of listing ident in the cart. With a cart view at least as fresh as the last cart change,
        it searches the view for a dict holding the id as a string value, or, where the store's ids are numbers, as a
        whole number under id_key or an id-named key (so a price equal to an id is never taken for it), with exactly
        one numeric count under a qty, quantity, count or unit(s) key; otherwise, or when none is found, it uses
        cart_lines, giving 1 for an unrecorded listing."""
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            stack: list[Any] = [self.cart_view]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    if ident in [self.as_id(v) for k, v in node.items() if self.as_id(v) is not None and (isinstance(v, str) or k == self.id_key or _id_named(k))]:
                        counts = [_count(v) for k, v in node.items() if not isinstance(v, str) and _count(v) is not None and re.search('\\b(?:qty|quantity|count|units?)\\b', _words(k))]
                        if len(counts) == 1:
                            return counts[0]
                    stack += [v for v in node.values() if isinstance(v, (dict, list))]
                elif isinstance(node, list):
                    stack += node
        return self.cart_lines.get(ident, 1)

    def cart_ids(self) -> list[str]:
        """Returns the ids of the listings in the cart. When a cart view has been read since the last cart change
        (cart_seen_at >= cart_at), these are the known listing ids whose text occurs in that view's JSON; an
        all-digit id counts only as a whole number under id_key or an id-named key where the store's ids are numbers
        (_ints_under), or as a whole number inside the view's text values (_texts_of), never as a price, a total or
        part of a longer number. Otherwise they are the keys of cart_lines, Halo's own record of its cart calls."""
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            text = _dump(self.cart_view)
            if not any((i.isdigit() for i in self.rows)):
                return [i for i in self.rows if i in text]
            numbers = _ints_under(self.cart_view, self.id_key) if self.int_ids else set()
            words = self.cart_view if isinstance(self.cart_view, str) else ' | '.join(_texts_of(self.cart_view))
            return [i for i in self.rows if ((i in numbers or bool(re.search('(?<![0-9.])' + i + '(?![0-9.])', words))) if i.isdigit() else i in text)]
        return list(self.cart_lines)

    def record_of(self, ident: str) -> str:
        """Returns the listing's full record (or, failing that, its search row) as compact JSON with string values
        clipped to 400 characters and the whole text cut at 3000; returns '(no record seen)' when neither is known."""
        record = self.full.get(ident) or self.rows.get(ident)
        return _dump(_clip(record, 400))[:3000] if record else '(no record seen)'

    def others(self, skip: list[str], limit: int=30) -> str:
        """Returns the records of other seen listings, one compact JSON line each (strings clipped to 160 characters,
        lines cut at 900), skipping ids in skip and sold-out ones, newest first, at most limit lines (default 30).
        Under the cheapest pick, listings with full records come first, each group by ascending price with unpriced
        ones last; '' when none qualify."""
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
        """Runs _fit_marks over the text of every seen listing (full record or row: its top-level strings plus 'key
        value' pairs from nested dicts) to find its own part code, model codes, works-with codes, 'part X also fits'
        notes and any pointer to another part's page. Returns {'marks': marks per listing id, 'notes': part code ->
        every device code a listing says that part also fits}."""
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
        """Returns the device codes a listing is said to work with, read from a fit_index() result: the codes in its own
        works-with, compatible-with or fits text, plus any codes that 'part X also fits' notes give for its own part
        code. Returns an empty set when none are known."""
        mark = index['marks'].get(ident) or {}
        found = set(mark.get('works') or [])
        if mark.get('own'):
            found |= index['notes'].get(mark['own'], set())
        return found

    def item_ids(self, number: int) -> list[str]:
        """Returns the seen listing ids, in the order first seen, whose row's category label (under category_key())
        belongs to item number (1-based) according to owns(); returns [] when no category key has been identified."""
        cat_key = self.category_key()
        item = self.items[number - 1]
        return [i for i in self.seen_order if cat_key and self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))]

    def fit_faults(self) -> list[str]:
        """Checks carted parts against the carted device for each fit pair in bound (part item, device item). With
        exactly one device line carrying model codes, each carted part whose fit_set is non-empty but shares no code
        with it yields a fault naming up to 4 seen, unsold, uncarted parts that fit, or a short fit_plan when none
        do. Returns the faults, [] when none."""
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
        """Summarises, per device item in bound, which device codes the unsold seen parts say they fit: the top 4 codes,
        ranked by how many part items and then listings name them, with per-item counts and up to 2 seen devices
        carrying each code (or 'device not seen yet'). Returns the lines joined by ' | ', prefixed by an explanatory
        header unless short; '' with no pairs or codes."""
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

    def missing_items(self) -> list[int]:
        """Returns the 1-based numbers of the requested items that have at least one seen listing in their category but
        no cart line holding units of it. Returns [] when no category key is known or the cart already has at least
        as many lines holding units as there are items."""
        cat_key = self.category_key()
        ids = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        if not cat_key or len(ids) >= len(self.items):
            return []
        labels = [str(r.get(cat_key) or '') for r in self.rows.values()]
        held = [str((self.rows.get(i) or {}).get(cat_key) or '') for i in ids]
        return [n for n, item in enumerate(self.items, 1) if any((self.owns(item, label) for label in labels)) and (not any((self.owns(item, label) for label in held)))]

    def states_needs(self, ident: str, number: int) -> bool:
        """Returns True when the attribute block of a listing's full record (_fact_block) plainly states every live
        requirement of item number: each requirement's value words must all appear in the value of an attribute whose
        key names its property. Returns False at the first miss, or for a requirement with no value words or with a
        limit or negation word (not, under, over, most, unlike, max and similar)."""
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
        """Carts, for each missing_items() item, the cheapest buyable, priced, uncarted listing of its category whose
        full record states all its requirements (states_needs), unless the item is in a fit pair or the line breaks a
        numeric budget (unit price for 'each' scope, else price times quantity plus the cart total). Returns False at
        the first item left unfilled (no pick or plan, too few steps, a failed call); True otherwise."""
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
                args = _shape(self.with_required(cart, args), self.schemas.get(cart), 'args', faults)
                if faults or self.act(cart, args) is None:
                    return False
                self.note_cart(cart, args)
            self.trace.append({'kind': 'filled_line', 'item': number, 'line': pick})
        return True

    def fuller_label(self, number: int, label: str) -> str:
        """Returns a fuller category label for item number when a cart line's label lacks some root words of the item's
        name: the alphabetically first seen label containing all of them. Returns '' when the name has fewer than two
        root words, the label covers them or matches the item's shelf, no category key is known, or no such label was
        seen."""
        cat_key = self.category_key()
        mine = _roots(self.items[number - 1].name)
        if len(mine) < 2 or mine <= _roots(label) or not cat_key or (self.items[number - 1].shelf and _roots(label) == _roots(self.items[number - 1].shelf)):
            return ''
        return next((seen for seen in sorted({str(r.get(cat_key) or '') for r in self.rows.values()}) if seen and mine <= _roots(seen)), '')

    def line_faults(self) -> list[str]:
        """Lists faults of the cart lines holding units, matched to items by category label after adopt_labels: for
        items with seen listings, a missing or extra line, a wrong unit count or an improvable label (fuller_label);
        unrequested kinds once all items have seen listings; sold-out lines; a price or total over budget (up to 10
        times it, all prices known). Returns [] for an empty cart or no category key."""
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
        """Returns (listing id, units held, units wanted) for each cart line to correct: not sold out, holding at least
        one unit, belonging to exactly one requested item, being that item's only cart line, and holding a count
        other than the item's quantity. Returns [] when no category key is known."""
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
        """Corrects the (id, have, want) quantities in wrong via the cart tool: one call setting want when it sets
        counts (cart_sets), else a removal-mode call for have units then an add-mode call for want. Records each call
        (note_cart, cart_said, observe) and returns 'args -> reply' lines, stopping at the first failure; [] without
        id or count arguments (the id may be a whole-number parameter named as an id; the count is another integer
        parameter), without setting or a mode, or when the calls plus 3 exceed left()."""
        name = self.roles.get('cart') or ''
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)] or _int_ids(props)
        target = next((k for t, k in self.id_args if t == name and k in texts), None) or next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        count = next((k for k, v in props.items() if isinstance(v, dict) and 'integer' in _kinds(v) and k != target), None)
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
            args = _shape(self.with_required(name, args), schema, 'args', faults)
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
        """Returns the memo key for judging a set of listings: the ids sorted and joined by '|', then '#' and the
        current version, so the same set is judged afresh once the version changes."""
        return '|'.join(sorted(ids)) + f'#{self.version}'

    def check_cart(self, ids: list[str] | None=None, limit: float=SPEND_SOFT) -> str:
        """Asks a judge (CHECK prompt) whether the cart, or the planned ids at the requested quantities, is exactly
        right; skips a repeated check_key, empty ids or empty judges(limit). A real plain-pick cart before any audit
        goes to LIGHT first, and to HEAVY only if LIGHT rejects it or fails. Sets verdicts[check_key]; on ok=false
        prunes leads and returns a fix request (up to 5 problems, 4 better listings), else ''. In order: The judge
        sees the request, the ledger, the cart (or the planned lines when called with ids) and the full records of
        the cart's listings; with a cheapest, most or least pick it also sees the other listings. When the light
        model passes an ordinary cart it is accepted without the heavy judge; a fail from the light model is checked
        again by the heavy one before it stops the checkout."""
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
        """Resolves live refs (requirements valued from another listing) missing from ref_values. Runs up to 6 rounds
        per episode (4 at 15 or fewer steps left) below SPEND_HARD: a TRAIL query traces each chain, fully opened
        chains' values are stored, and up to 3 unopened links are opened or one is searched (find_link). Returns
        (report, hold); hold means a link stays unfound, under the trail_blocks cap, with over 5 steps left. In
        order: Each round asks the chain reader to follow every pending pointed requirement link by link through the
        opened records, then opens the rows a chain names but that were never opened, up to six rounds. A chain whose
        last link is an opened page gives the requirement its value (ref_values); a chain stopped at a listing never
        seen yields a 'find' description used for a search. While a link is still missing the checkout is held (at
        most three times, two when few steps are left) with a message saying what to find and how to search for it."""
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
                args = _shape(self.with_required(tool, {target: ident}), self.schemas.get(tool), 'args', faults)
                if faults:
                    break
                reply = self.act(tool, args)
                if reply is None or self.finished:
                    break
                self.observe(tool, reply.get('observation'))
                seen.append(f'{ident}: ' + self.show(reply.get('observation'))[:1500])
        text = ''
        if seen:
            text += '[Halo opened the pages a requirement points at]\n' + '\n'.join(seen) + '\n'
        if self.ref_values:
            text += '[values read from the pages requirements point at] ' + '; '.join((f'{k} -> {v}' for k, v in self.ref_values.items())) + '\n'
        hold = bool(lost) and self.trail_blocks < (3 if self.left() > 15 else 2) and (self.left() > 5)
        if hold:
            self.trail_blocks += 1
            text += 'The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: ' + ' | '.join(lost[:3]) + ". Search for it by its shop's name together with the values its description gives (the shop's other listings of that kind are not it), open the row of that shop whose page shows those values, apply the value its page gives, then check out."
        return (text, hold)

    def shop_rows(self, about: str, ids: list[str]) -> list[str]:
        """Returns those of ids that are not yet opened and whose row has a seller, shop, store, vendor, brand, maker or
        merchant field (a 3 to 40 character string) named as a whole word or phrase in the description about,
        ignoring case. Keeps the given order; [] when none match."""
        low = about.lower()
        out = []
        for ident in ids:
            row = self.rows.get(ident) or {}
            names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I)]
            if ident not in self.opened and any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)):
                out.append(ident)
        return out

    def opened_shop(self, about: str) -> list[str]:
        """Returns the opened listings (self.opened) whose row has a seller, shop, store, vendor, brand, maker or
        merchant field (a 3 to 40 character string) named as a whole word or phrase in the description about,
        ignoring case; [] when none. Unlike shop_rows it scans every opened id rather than a given list."""
        low = about.lower()
        out = []
        for ident in self.opened:
            row = self.rows.get(ident) or {}
            names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I)]
            if any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)):
                out.append(ident)
        return out

    def find_link(self, about: str, words: str) -> list[str]:
        """Finds candidates for an unfound chain link (about, words). If an opened listing of the named shop shows every
        value word of the link, traces 'link_known' with its id and returns []. Else reuses link_seen or runs one
        search from search_template (counted in link_searches) and returns up to 2 unopened rows of that shop sharing
        a word with the link, from those results or else all seen rows, or []."""
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
            args = _shape(self.with_required(tool, {**base, field: words}), self.schemas.get(tool), 'args', faults)
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
            """Keeps the ids whose row, dumped as lowercase JSON, shares at least one kind word with the link
            (three-plus-character words of the search words, or of the description when there are none, minus
            shop-name words and filler); keeps every id when there are no kind words."""
            return [i for i in ids if not kind or kind & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower()))]
        return (of_kind(known) or of_kind(self.shop_rows(about, list(reversed(self.seen_order)))))[:2]

    def offer_tool(self) -> str | None:
        """Returns the tool for re-reading a listing's current offer (price and stock): the 'offer' role, else the
        'open' role; None when neither is set."""
        return self.roles.get('offer') or self.roles.get('open')

    def reread_offers(self) -> str:
        """Re-opens, with offer_tool and its single id argument, each cart line not in gone whose offer has not been
        read since max(step - 2, refused_at), to refresh prices and stock. Returns '' without such a tool, with
        nothing stale, or with fewer than the stale count plus 3 steps left; else updates offer_seen and rows, bumps
        version on any change, and returns a line per listing whose scalar fields changed."""
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
            args = _shape(self.with_required(tool, {target: ident}), self.schemas.get(tool), 'args', faults)
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
        """Returns True when the listing is in unavailable (a row flag that turned from true to false, a store note
        naming it, or find_sold's guess) or in gone (cart lines swap_hint already handled as sold out); False
        otherwise."""
        return ident in self.unavailable or ident in self.gone

    def swap_hint(self) -> str:
        """Handles cart lines newly sold out: records them in gone at this step, drops them from leads, bumps version
        and asks replacements() for substitutes. Returns '' when there are none; else a do-not-reopen notice advising
        a two-call swap to the first pick at 6 or fewer steps left, a search below its price under the cheapest pick,
        a plain swap, or, without picks, removing the line and finding another."""
        lost = [i for i in self.cart_ids() if i not in self.gone and self.sold_out(i)]
        if not lost:
            return ''
        for ident in lost:
            self.gone[ident] = self.step
        self.leads = [i for i in self.leads if i not in lost]
        self.version += 1
        picks = self.replacements(lost)
        text = f"[Halo] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        if picks and self.left() <= 6:
            return text + f" Few steps are left: in this turn add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks[1:]) or 'none'}."
        if picks and self.brief.get('pick') == 'cheapest':
            return text + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(picks)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {picks[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        if picks:
            return text + f" Listings already read that appear to fit that item instead: {', '.join(picks)}. In this turn take the sold-out line out of the cart and add {picks[0]} at the requested quantity, then check out."
        return text + ' Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits.'

    def find_sold(self) -> str:
        """Works out which cart line a sold-out note meant: returns '' when a cart line not yet in gone is already known
        sold out, marks the only such line unavailable when there is just one (returning ''), and with several lines
        re-reads their offers through reread_offers and returns its change report; '' for an empty cart."""
        ids = [i for i in self.cart_ids() if i not in self.gone]
        if any((self.sold_out(i) for i in ids)):
            return ''
        if len(ids) == 1:
            self.unavailable.add(ids[0])
            return ''
        return self.reread_offers() if ids else ''

    def replacements(self, lost: list[str]) -> list[str]:
        """Asks a judge (SWAP prompt; judges() reversed, so LIGHT first) for listings that could replace the sold-out
        lines in lost, showing the request, ledger, cart and up to 25 of the newest other available listings. Keeps
        known ids not lost, gone or carted, sorted by reported price under the cheapest pick (unpriced last), and
        returns up to 4; [] when judges() is empty or no other listing remains."""
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
        """Runs once per episode (setting audited) when the agent tries to end without buying: asks audit_call which
        seen listings fit and, if none is named, opens the unopened unsold rows the reply suggests (open_rows) and
        asks again when that added full records. When fits are named, stores them in lead_items and leads and returns
        a message listing them per item with the judge's reason; otherwise ''."""
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
        if not named:
            return ''
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return 'A careful review thinks these listings may fit: ' + '; '.join((f"item {k}: {', '.join(v)}" for k, v in named.items())) + (f" ({got.get('reason')})" if got.get('reason') else '') + '. Open them and buy what truly fits; end without a purchase only if each one fails.'

    def audit_call(self) -> dict[str, Any] | None:
        """Asks the AUDIT judge which seen listings fit, showing the request, ledger and up to 30 of the newest unsold
        listings as OPENED full records or ROW search rows. Asks LIGHT, then HEAVY when it is affordable, credit
        remains and LIGHT's reply names no fit (keeping LIGHT's reply if HEAVY gives none). Returns the parsed reply;
        None without records, judges or an answer."""
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
        judges = self.judges()
        got = self.ask_json(prompt, [LIGHT] if judges else [], 4000)
        if HEAVY in judges and not self.no_credit and not self.audit_fits(got):
            got = self.ask_json(prompt, [HEAVY], 4000) or got
        self.trace.append({'kind': 'audit', 'got': got})
        return got

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        """Returns an audit reply's 'fits' mapping (item key -> listing ids) with string keys, keeping only ids seen in
        rows and not sold out, at most 4 per item, and dropping items left empty; returns {} when the reply is None
        or its 'fits' is not an object."""
        fits = (got or {}).get('fits')
        if not isinstance(fits, dict):
            return {}
        named = {str(k): [str(i) for i in _items(v) if str(i) in self.rows and (not self.sold_out(str(i)))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def open_rows(self, rows: list[str]) -> bool:
        """Opens search rows so their attributes can be judged: one side-by-side call with up to compare_room() ids when
        the compare tool takes an id array and is not judged bare (compare_bare), else single facts_tool calls for
        the first two rows. Returns True only when new full records arrived; False when no tool fits, fewer than the
        planned calls plus 5 steps remain, or the arguments break the schema."""
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
            args = _shape(self.with_required(name, args), self.schemas.get(name), 'args', faults)
            if faults:
                return False
            reply = self.act(name, args)
            if reply is None or self.finished:
                break
            self.observe(name, reply.get('observation'))
        self.trace.append({'kind': 'audit_open', 'rows': rows, 'opened': len(self.full) - before})
        return len(self.full) > before

    def cart_calls(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        """Plans cart tool calls that set lines (id -> units) and remove drop: a single call listing all lines when the
        tool takes an array of line objects, else one removal per dropped id (mode option, zero or negative count)
        and one add per line whose held units differ. The id field may be a whole-number parameter named as an id
        when the tool has no free string field, and the count is never taken from the id field. Returns None when no
        id field or way to remove fits, or a held line's count would change without cart_sets."""
        name = self.roles.get('cart') or ''
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        for key, spec in props.items():
            shape = spec.get('items') if isinstance(spec, dict) and 'array' in _kinds(spec) else None
            if isinstance(shape, dict) and isinstance(shape.get('properties'), dict):
                inner = shape['properties']
                id_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'string' in _kinds(s) and ('enum' not in s)), None) or next(iter(_int_ids(inner)), None)
                count_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'integer' in _kinds(s) and k != id_key), None)
                if not id_key:
                    return None
                return [{key: [{id_key: i, **({count_key: n} if count_key else {})} for i, n in lines.items()]}]
        texts = [k for k, s in props.items() if isinstance(s, dict) and 'string' in _kinds(s) and ('enum' not in s)] or _int_ids(props)
        target = next((k for t, k in self.id_args if t == name and k in texts), None) or next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        count = next((k for k, s in props.items() if isinstance(s, dict) and 'integer' in _kinds(s) and k != target), None)
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
        """Tries buying instead of declining, once per episode, when a lead was confirmed: plans each item's first
        unsold lead (one per kind, at its quantity), keeps unsold cart lines of other kinds, drops the rest, needing
        the confirmed lead and every item covered. After fill_facts, show_links and a passing check_cart
        (SPEND_HARD), sets the cart and buys via the order gate, retrying once after a refusal. Returns a report or
        ''. In order: Builds the cart from the decline audit's leads: one listing per item, the confirmed listing
        among them, keeping other cart lines only when they are of a kind no lead replaces. Checks out at most twice:
        before a retry the offers are re-read and the cart checked again, and the order's own steps (the gate) run
        before each attempt."""
        cart, buy = (self.roles.get('cart'), self.roles.get('buy'))
        if self.completed or not self.confirmed or self.sold_out(self.confirmed) or (not cart) or (not buy):
            return ''
        self.completed = True
        cat_key = self.category_key()

        def kind(ident: str) -> str:
            """Returns a listing's kind for the plan: its row's category label under cat_key, stripped and lowercased
            ('' when absent), or the id itself when no category key is known, so each listing is then its own kind."""
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
            args = _shape(self.with_required(cart, args), self.schemas.get(cart), 'args', faults)
            if faults:
                return ''
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return ''
            self.note_cart(cart, args)
            self.observe(cart, reply.get('observation'))
            done.append(_dump(args))
        faults = []
        args = _shape(self.with_required(buy, dict(self.brief.get('buy_args') or {})), self.schemas.get(buy), 'args', faults)
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
        return 'Not run (Halo acted instead): the independent check confirmed ' + self.confirmed + ', so Halo set the cart (' + '; '.join(done) + ') and checked out. Store reply: ' + ' | then: '.join(replies)

    def is_buy(self, name: str, args: dict[str, Any]) -> bool:
        """Returns True when a call names the checkout ('buy') tool and its arguments match every fixed argument in the
        brief's buy_args, which tells checkout apart from ending when one tool does both; False for any other tool or
        a mismatching argument."""
        if name != self.roles.get('buy'):
            return False
        fixed = self.brief.get('buy_args') if isinstance(self.brief.get('buy_args'), dict) else {}
        return all((args.get(k) == v for k, v in fixed.items()))

    def is_refuse(self, name: str, args: dict[str, Any]) -> bool:
        """Returns True when a call is the no-purchase ending: the tool is the one in the 'refuse' role and args hold
        every fixed key/value pair from the brief's refuse_args with equal values, so a tool shared with checkout
        counts only in its decline mode. A missing or non-dict refuse_args lets any call to that tool qualify; a call
        to any other tool gives False."""
        if name != self.roles.get('refuse'):
            return False
        fixed = self.brief.get('refuse_args') if isinstance(self.brief.get('refuse_args'), dict) else {}
        return all((args.get(k) == v for k, v in fixed.items()))

    def offered(self) -> list[dict[str, Any]]:
        """Returns the tool definitions the light model may call this turn: every store tool except the messaging
        ('ask') one, narrowed as steps run out. At or below tail() steps left only the buy and refuse tools stay;
        within tail()+3 the cart and cart-view tools are kept too; and within len(items)+5 steps before any checkout
        attempt (buys == 0) the offer tool joins them. Narrowing needs a buy or refuse role and falls back to the
        full list when no tool matches."""
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
        """Returns a copy of a decline call's args in which free-text fields may be swapped for a fixed, neutral apology
        naming no listing. Each string parameter without an enum that is present or required gets that sentence, cut
        to its maxLength (200 when unset), when its text is empty, mentions a known listing id, or the brief's
        quiet_decline is anything but False; all other values are kept."""
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
        """Rewrites title-like text in the listing argument of a cart, open, offer or compare call into a listing id,
        returning (args, problem). It applies only once id_key is known and the tool has exactly one free string or
        array parameter without an enum, which for tools other than the cart must have carried ids before (id_args);
        otherwise args pass through with ''. Each value, or list element, goes through fix(): the first problem
        returns the original args with that message, else the rewritten copy comes back with ''."""
        if name not in {self.roles.get(r) for r in ('cart', 'open', 'offer', 'compare')} - {None} or not self.id_key:
            return (args, '')
        props = (self.schemas.get(name) or {}).get('properties') or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and ('string' in _kinds(v) or 'array' in _kinds(v)) and ('enum' not in v)]
        if len(free) != 1 or (name != self.roles.get('cart') and (name, free[0]) not in self.id_args):
            return (args, '')

        def fix(value: Any) -> tuple[Any, str]:
            """Maps one argument value to a listing id and returns (value, problem). A known id comes back stripped;
            non-strings, single-token id-like strings and strings under 8 characters pass unchanged. Other text is
            matched, ignoring case, against the start of every non-id string field in the seen rows: one match
            returns that listing's id, several return the value with a message naming up to six of them, and none
            returns a message pointing at the id_key field."""
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
        """Guesses from the named cart tool's schema whether it sets a line's quantity outright instead of adding to it.
        Returns False when any parameter is an enum with two or more options (an add/remove mode); otherwise True
        only when some integer parameter declares minimum 0, so a count of zero can clear a line."""
        props = self.schemas.get(name, {}).get('properties') or {}
        if any((isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) >= 2) for s in props.values())):
            return False
        return any((isinstance(s, dict) and 'integer' in _kinds(s) and (s.get('minimum') == 0) for s in props.values()))

    def whole_removal(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Returns cart-call args with the unit count filled in when a removal leaves it out, so the whole line comes
        off. When an enum argument takes any value after its first option (read as a remove mode), an argument (a
        string or a whole number, in text form) names a listing in cart_lines, and the tool's first integer parameter
        is absent, that count is set to the units on record, capped at the parameter's maximum; otherwise args are
        returned as given."""
        spec = self.schemas.get(name, {}).get('properties') or {}
        count = next((k for k, v in spec.items() if isinstance(v, dict) and 'integer' in _kinds(v)), None)
        removal = any((isinstance((spec.get(k) or {}).get('enum'), list) and v in spec[k]['enum'][1:] for k, v in args.items()))
        ident = next((self.as_id(v) for v in args.values() if self.as_id(v) in self.cart_lines), None)
        return {**args, count: min(self.cart_lines[ident], spec[count].get('maximum', self.cart_lines[ident]))} if count and removal and ident and count not in args else args

    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        """Updates Halo's record of the cart (cart_lines: listing id to units) after a cart call, setting cart_touched
        and stamping cart_at with the current step. A call carrying an array of line objects, or an empty array of
        them, replaces the record with its listed lines of positive count. Otherwise each known listing id in the
        args (a string or a whole number, read in text form; an id value is never read as the count) is reduced in
        remove mode (an enum value past the first option), set when cart_sets() reports set semantics, or raised by
        the given count (1 when none); lines reaching zero or less are dropped."""
        self.cart_touched = True
        self.cart_at = self.step
        spec = self.schemas.get(name, {}).get('properties') or {}
        removal = False
        for key, value in args.items():
            options = (spec.get(key) or {}).get('enum')
            if isinstance(options, list) and len(options) >= 2 and (value in options[1:]):
                removal = True
        loose_count = next((_count(v) for v in args.values() if _count(v) is not None and (not isinstance(v, str)) and self.as_id(v) not in self.rows), None)
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
                        ident = next((self.as_id(v) for v in line.values() if self.as_id(v) in self.rows), None)
                        count = next((_count(v) for v in line.values() if _count(v) is not None and (not isinstance(v, str)) and self.as_id(v) not in self.rows), 1)
                        if ident:
                            lines.append((ident, count))
                    elif self.as_id(line) in self.rows:
                        lines.append((self.as_id(line), loose_count))
            elif self.as_id(value) in self.rows:
                lines.append((self.as_id(value), loose_count))
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
        """Returns a store observation as compact JSON text for the chat: every string in it is cut to SHOW_CHARS (450)
        characters, values nested more than seven levels down become '...', and the whole text is capped at OBS_LIMIT
        (15000) characters with a '...(cut)' marker."""
        text = _dump(_clip(observation, SHOW_CHARS))
        if len(text) > OBS_LIMIT:
            text = text[:OBS_LIMIT] + '...(cut)'
        return text

    def compare_target(self) -> str | None:
        """Returns the name of the compare tool's list parameter, taken as the first array-typed property in its schema.
        Returns None when there is no compare tool or array property, or when the schema requires another parameter
        that with_required() cannot fill (fillable())."""
        schema = self.schemas.get(self.roles.get('compare') or '') or {}
        props = schema.get('properties') or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in _kinds(v)), None)
        if not target or any((k != target and (not self.fillable(self.roles.get('compare'), k)) for k in schema.get('required') or [])):
            return None
        return target

    def compare_room(self) -> int:
        """Returns how many listings to send in one side-by-side compare call. A maxItems on the compare tool's list
        parameter wins; once a call showed none of its listings (compare_bad, the smallest such size), it is one less
        than that size, capped at the largest size known to work (compare_ok, but at least 2) and never below 2;
        otherwise it is the larger of 4 and compare_ok."""
        spec = ((self.schemas.get(self.roles.get('compare') or '') or {}).get('properties') or {}).get(self.compare_target() or '') or {}
        if isinstance(spec.get('maxItems'), int):
            return spec['maxItems']
        if self.compare_bad:
            return max(2, min(self.compare_bad - 1, max(self.compare_ok, 2)))
        return max(4, self.compare_ok)

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Folds several single-listing open calls from one turn into one side-by-side compare call. Opens passing only
        the id argument for a known listing not marked sold out (gone) are collected up to compare_room(); with two
        or more, the first becomes a compare call over their ids, the rest become placeholders flagged 'joined', and
        the merge is traced. Calls return unchanged when the open or compare role is missing or both are one tool,
        either lacks a usable id parameter, or compare_bare() says the compare view lacks attributes."""
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
        """Runs a turn's tool calls (after join_opens()), returning a (call id, reply text) pair for each. Calls with
        bad arguments or ids, to unknown, unoffered or messaging tools, opening sold-out listings, removing absent
        lines, repeating calls with nothing changed, bundling a finish with others or spending the last tail() steps
        on anything else are answered without running. A checkout is held for wider_hold, a changed cart view,
        sold-out lines, pending trail() chains, fit or line faults, a check_cart objection or a late question; a
        decline for audit leads, many steps left or a fully built cart, unless complete_cart() buys the confirmed
        lead. Both endings pass gate(), the decline with quiet() text. Executed calls update Halo's records, roles
        and caches and reply with the clipped observation plus Halo's notes. In order: Every tool call the loop model
        sent is either answered without using a step (with the reason) or made; finishing calls pass through the
        checks below first. Messages to the shopper belong to the interviewer alone; calls with an unknown tool, a
        tool withheld for lack of steps, or arguments that are not one JSON object are refused at no step cost.
        Arguments are shaped to the schema (types, enums, lengths) and listing ids are resolved from title text;
        anything still wrong is reported back instead of spending a step on a refusal. A finishing call must come
        alone, and the last steps are kept for checking out or declining, so other calls are refused then. Repeated
        reads cost nothing: the same call just made, a read whose result has not changed since (read_cache), and a
        page that has already shown everything its view gives are answered from memory. Before a checkout: the wider
        cheapest search, an offer re-read, a fresh look at the cart, sold-out swaps, reference chains, accessory fit,
        cart line faults and the judge's check, each of which can hold it. A late decline with a cart that once held
        a line per item is held once: running short of steps does not show that nothing fits. A decline with many
        steps left and no review candidates is held once and the loop is told how to search more widely first. A
        decline that gets through is turned into a checkout when the audit confirmed a fitting listing; otherwise its
        remark is made quiet (no listing named). Steps the request wants done at the order or at an ending run just
        before the finishing call; a reply that changes the task stops it. A side-by-side view that showed none of
        the listings sent marks that count as too many for this store, so later compare calls send fewer. A
        single-listing view teaches which tool shows attributes: an open tool that shows none hands the open role to
        another single-listing tool, and a tool showing only offers becomes the offer tool. Cart calls update Halo's
        own record of the cart lines; a line set to zero that the store's reply no longer shows counts as dropped.
        The result text gains the cheapest buyable rows and a warning when a price cap was ignored, then evidence
        lines naming the attribute keys that match each requirement's property. Loose notes in the result go to the
        note reader; a refusal or sell-out note without a listing id makes Halo re-read the cart's offers to find
        which line it meant."""
        out: list[tuple[str, str]] = []
        calls = self.join_opens(calls)
        for index, call in enumerate(calls):
            allowed = {t['function']['name'] for t in self.offered()}
            cid = call['id']
            if call.get('joined'):
                out.append((cid, 'Shown in the side-by-side result above: Halo opened these listings together in one step.'))
                continue
            fn = call.get('function') or {}
            name = fn.get('name')
            raw = fn.get('arguments')
            if self.finished:
                out.append((cid, 'Not run: the episode is over.'))
                continue
            if name == self.roles.get('ask'):
                out.append((cid, "Not run (no step used): Halo's interviewer handles every message to the shopper."))
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
                trimmed = f"\n[Halo] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
                args = {**args, many: args[many][:room]}
            target = self.single_target(name)
            if target and name in (self.roles.get('open'), self.roles.get('offer')) and (str(args.get(target)) in self.gone):
                out.append((cid, f'Not run (no step used): {args[target]} sold out at step {self.gone[str(args[target])]} and will not come back. Replace its cart line with another fitting listing.'))
                continue
            if name == self.roles.get('cart') and any(((_count(v) or 0) < 0 for v in args.values() if not isinstance(v, str) and self.as_id(v) not in self.rows)):
                view = _dump(self.cart_view) if self.cart_seen_at >= self.cart_at else ''
                absent = [self.as_id(v) for v in args.values() if self.as_id(v) in self.rows and (self.as_id(v) not in self.cart_lines) and (self.as_id(v) not in view) and (self.as_id(v) not in self.cart_said)]
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
                    trimmed += f"\n[Halo] Left out {', '.join((str(i) for i in args[many] if i not in keep))}: shown in full already, nothing changed since."
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
                        view_args = _shape(self.with_required(self.roles['cart_view'], {}), self.schemas.get(self.roles['cart_view']), 'args', faults)
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
                            out.append((cid, "Not run yet (no step used). Halo set the cart's quantities itself: " + '; '.join(fixed) + '. If the store refused any of these, fix those lines; then check out.\nCart now: ' + self.cart_text()))
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
                        seen = "Halo read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n' if read else ''
                        out.append((cid, 'Not run yet. ' + chain + seen + verdict + '\nCart now: ' + self.cart_text()))
                        continue
                    if reread and self.cost > SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). Halo re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'))
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
                    trimmed = '\n[Halo] ' + said + trimmed
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, 'The call failed (network or server). You may try again.'))
                continue
            observation = reply.get('observation')
            if many and isinstance(args.get(many), list) and (len(args[many]) > 1):
                shown = _dump(observation)
                if not any((str(i) in shown for i in args[many])):
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    trimmed += f'\n[Halo] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_room()} per call.'
                else:
                    self.compare_ok = max(self.compare_ok, len(args[many]))
            self.recent = (self.recent + [mark])[-3:]
            self.id_args.update(((name, k) for k, v in args.items() if (v if isinstance(v, list) else [v]) and all((self.as_id(x) in self.rows for x in (v if isinstance(v, list) else [v])))))
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
                counts = [_count(v) for v in args.values() if not isinstance(v, str) and _count(v) is not None and self.as_id(v) not in self.rows]
                for value in args.values():
                    if counts and self.as_id(value) in self.rows:
                        if counts[0] <= 0 and (not _in_rows(observation, self.as_id(value), ints=self.int_ids)):
                            self.dropped.add(self.as_id(value))
                        elif counts[0] > 0:
                            self.dropped.discard(self.as_id(value))
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
            if name == self.roles.get('cart') and any((_count(v) == 0 for v in args.values() if not isinstance(v, str) and self.as_id(v) not in self.rows)):
                if any((self.as_id(v) in self.rows and _in_rows(observation, self.as_id(v), ints=self.int_ids) for v in args.values())):
                    text += '\n[Halo] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'
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
                    text += "\n[Halo re-read the cart's lines after the refusal]\n" + seen
            self.note_sold = False
            swap = self.swap_hint()
            if swap:
                text += '\n' + swap
            if self.queued:
                text += "\n[notes seen on Halo's own calls] " + ' | '.join(self.queued)
                self.queued = []
            if name in readers:
                self.read_cache[mark] = (self.version, self.step, base)
            if reply.get('finished'):
                text += ' [episode finished]'
            out.append((cid, text))
        return out

    def can_refuse(self) -> bool:
        """Returns True when Halo could send the no-purchase ending by itself: a tool holds the 'refuse' role and the
        brief's refuse_args, after quiet() fills the free-text fields with the canned decline, pass _shape() against
        that tool's schema without faults. Returns False when there is no refuse tool or those arguments miss a
        required field or break the schema."""
        refuse = self.roles.get('refuse')
        if not refuse:
            return False
        faults: list[str] = []
        _shape(self.with_required(refuse, self.quiet(refuse, dict(self.brief.get('refuse_args') or {}))), self.schemas.get(refuse), 'args', faults)
        return not faults

    def finish(self, again: bool=False) -> str:
        """Forces an ending without the model: a checkout when a cart is held, else (or when that checkout gets no
        reply, or is refused while again is False) the no-purchase ending. For two or more items, fill_short_cart()
        first adds missing lines and the lines' attributes and linked pages are read; a still-short cart is bought
        only if can_refuse() rules out a decline. Each ending runs gate() when two or more steps remain and uses
        buy_args or quiet() refuse_args, with any other required parameter filled where safe (with_required(): a
        preview switch false, the note). Returns 'the checkout was refused: ...' when again is set (counting it in
        buys and refused_at), the refusal text for a refused decline, and '' otherwise. In order: A multi-item cart
        missing an item is filled with the cheapest opened listing that states every requirement word for word; if
        one stays missing and a decline is possible, the episode declines instead of ordering short. Forced ending:
        check out when the cart holds lines (after the order's own steps); otherwise, or when that is refused,
        decline with a quiet remark."""
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
            args = _shape(self.with_required(buy, dict(self.brief.get('buy_args') or {})), self.schemas.get(buy), 'args', faults)
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
        if refuse:
            faults = []
            args = _shape(self.with_required(refuse, self.quiet(refuse, dict(self.brief.get('refuse_args') or {}))), self.schemas.get(refuse), 'args', faults)
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
        """Shrinks the chat history once the summed JSON length of the message contents exceeds HISTORY_LIMIT (150000
        characters); below that it does nothing. Of the older half of the tool-result messages (at least one), each
        longer than 600 characters is cut to its first 400 plus a note naming up to 20 known listing ids found quoted
        in it; other roles are never touched."""
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
        """Plays the episode and returns its trace (at once if already finished). It builds the brief with
        read_request(), absorbs the opening view, seeds the chat with GUIDE and an intro of request, opening and
        status(), then runs work() capped at LOOP_CALLS turns and SPEND_HARD. If still open, finish(again=True)
        forces an ending; a refusal with more than one step and credit left goes back to the model for up to 10 more
        turns (budget SPEND_HARD + 0.01) before a last finish(). A closing 'end' entry logs finished, steps, cost and
        calls."""
        if self.finished:
            return self.trace
        self.read_request()
        self.trace.append({'kind': 'brief', 'brief': self.brief})
        opening = self.view.get('opening')
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
                self.chat.append({'role': 'user', 'content': '[Halo] Halo tried to end the episode and ' + turned + '\nApply what the store says, then finish.\n' + self.status()})
                self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
            if not self.finished:
                self.finish()
        self.trace.append({'kind': 'end', 'finished': self.finished, 'steps': self.step, 'cost': round(self.cost, 5), 'calls': self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        """Drives the light-model tool loop until the episode finishes, Halo is broke, steps run out, loop_cap turns or
        spend_cap is reached, or the clock passes LAST_TURN (an empty cart may go on while model_room() exceeds
        EMPTY_ROOM). Each pass runs interview() when ready_to_ask() allows, else trims the history and asks LIGHT,
        then SPARE, for calls among offered(); a credit error, ALL_CALLS or a third missed reply stops it, as do
        three call-less replies in a row. Cart, buy or refuse calls before any interview (over 8 steps left, before
        ASK_BY) are turned back while the shopper is asked; other calls go to run_calls(), their results returning as
        tool messages with status() on the last. In order: The interview runs as soon as ready_to_ask() allows,
        outside the loop model's turns; its summary joins the chat as a user turn. A turn with no tool call gets a
        reminder; three such turns in a row end the loop."""
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
    """Builds a Halo episode from problem_data and returns the trace its run() produces. If construction fails it
    returns a one-entry trace of kind 'crash' with the error type and message cut to 400 characters; if run() raises,
    the crash is appended, finish() is tried once on an unfinished episode (a second error is logged as well), and
    the trace so far is returned."""
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
