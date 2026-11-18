"""Space (s line): the released race-10 winner's procedure with the race-11 lessons built in.

A light model walks the store with the world's own tools. Code keeps the
ledger (items, needs, budget, listings seen), plans the questions to the
shopper, and stops wasted steps. A heavy model reads the request once and
checks the cart before an order and the evidence before a decline.
What this line adds, from the race-11 forensics of the k line and the field:
a wall clock with a watchdog thread that sends the finish itself before the
sandbox's 300 s kill, time-aware optional stages, a spend ledger sized to
the shared $5 key, a least-contradicting order instead of an ending without
a purchase (91% of the field's declines were wrong), code-token marks for
compatibility requirements, and stricter literal reading of attributes.
"""
from __future__ import annotations

import json
import random
import re
import threading
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

LIGHT = "anthropic/claude-haiku-5.5"
HEAVY = "anthropic/claude-haiku-5.5"          # fixed-form calls: level with the larger model on the benches at a fraction of the spend
HEAVY_EFFORT = "low"      # reasoning effort for the heavy model's fixed-form calls; the provider default (high) costs the most
HEAR_EFFORT = "medium"    # the shopper's answers are read with more care: a misread property name or a negation turned into an exclusion loses the episode
SPARE = "openai/gpt-6-luna"

LOOP_CALLS = 52             # race 11: tasks allowed 52-113 steps; 42 turns ran out before the steps did
ALL_CALLS = 70
SPEND_SOFT = 0.045
SPEND_HARD = 0.055          # 90 episodes share one $5 key (race 11: a 0.0566 average ran it dry): 90 x 0.055 stays under it; raised 11-08 from 0.042 so long multi-item
                            # episodes early and those orders passed 0.1-0.2 against 0.6, while the field spends 0.036-0.055 an episode
SPEND_FINAL = 0.006         # spend kept for the last check and the finish (allowed past SPEND_HARD)
DEADLINE_S = 285.0          # the sandbox kills the episode at 300 s
GUARD_S = 245.0             # the watchdog thread sends the finish itself at this point whatever the main thread is doing
STAGE_RESERVE = 66.0        # seconds the finish needs: no optional stage (re-reads, trail, checks, paging, interview) starts with less room
# waits before resending a failed model call; an episode has 300 s in all, so at most 45 s go to waiting
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3             # store calls the process gate may make before one checkout or ending
AUTO_PAGES = 6              # further search pages Kyo reads itself for a cheapest, most or least pick
CHEAPER_OPENS = 3           # cheaper rows of a cart line's kind whose attributes Kyo reads before a cheapest checkout

# one attempt per call (think() resends): a stuck call costs 50 s, not minutes, inside the 300 s
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=50, max_retries=2, retry_delay=1)   # a store call that stalls is given up before it costs the episode


GUIDE = """You are Kyo, a careful buyer working for one shopper in an online store. You can act only by calling the store's tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.

HOW THE STORE WORKS
- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool's description and schema and use the exact argument names and allowed values.
- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.
- Search rows are summaries. Stores split a listing's record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Kyo knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.
- Questions to the shopper are handled for you by Kyo's interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.
- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Kyo carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store's tools (product questions go through Kyo's interviewer).

WHAT COUNTS AS A FIT
- The kind of product is decided by the listing's category label: only the label that matches the requested item as a whole counts, never a neighbouring one.
- A requirement on a property is met when the listing's attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value's words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.
- A wish in the request that names no value a listing could state ("legs cut comfortably", "should not stick out far") is not a requirement: the interviewer asks the shopper about that property, and when the answer gives no value, any listing meeting the stated requirements fits. Never decline or reject a listing over an unstated value you inferred yourself.
- An attribute whose key names the requirement's property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body's colour), decides nothing about the requirement. Kyo adds [evidence] lines after opened listings with the keys whose names match a requirement.
- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.
- When the category label itself states a requirement's value (a season or audience in the label), that requirement is met by the category.
- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing's value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.
- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.
- When one requested item must fit another (an accessory for a device), choose the two together from the listings' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.
- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.
- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.

CHOOSING
- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Kyo reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.
- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.
- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.
- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.
- A request for several different items needs one cart line per item, each at its requested quantity.

CHANGES WHILE YOU WORK
- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool's result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.
- Re-read a chosen listing's record shortly before checking out, because price and stock can change silently.

FINISHING
- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.
- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request's wording for that ending, and when a remark is needed, keep it short and name no listing.
- Every reply must call at least one tool. Keep your text short: what you know, then the next move."""


BRIEF = """You prepare a shopping task for an agent. Read the shopper's request and the store's tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:

"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request's words, "shelf": the store's category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request's own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property's value or limit outright; and never turn words that are part of what the product is called (a season, an audience or a use named with the product) into musts: those belong to "name". A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.
"budget": the most the shopper will pay in total as a number in the request's own unit, or null.
"budget_unit": that unit in words (for example "cents"), or null.
"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.
"in_stock_only": true unless the request clearly accepts items that are not available.
"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.
"pick_measure": what is maximised or minimised for "most"/"least", else "".
"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.
"steps": how many steps, moves or tool calls the request allows (integer), or null.
"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].
"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.
"wishes": a list of things the shopper would merely like.
"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).
"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper's approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.
"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.
"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing's attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing's current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).
"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.
"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.
"""


PLAN = """You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property. A question may list several properties of one product: the shopper answers about the first listed property they hold a preference on, and the chance of an answer falls the further down the list that property sits, so rank the properties by how likely a preference is. A settled requirement whose value is only a wish in words (no figure, no value listings use) is a hint that the shopper holds an exact preference on that property: list that property first.

You get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item's category, with how many of those listings carry the key and example values. Unwritten preferences sit on properties the item's listings carry as keys: a property that no listed key names is not worth a message, since the shopper's reply to it is only confusion.

Choose the properties in this order:
0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.
1. Hinted properties. The request's story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product's looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them, and only when a listed key of that item carries that property; a hint that no key answers to is left aside.
Name a hinted property by its plain name: a hint about who will wear it points at gender, a hint about its looks at colour or finish (whichever the request leaves open).
2. Then the listed keys, ranked: keys carried by most of the item's listings come first, and among keys of similar reach a property whose values are words (a type, a material, a colour, a style, a finish, a pattern) goes before one whose values are figures or measurements, because unwritten preferences sit on word-valued properties several times more often than on measured ones; the most carried measurements stay on the list after the word-valued properties; avoid keys seen on only one or two of them.
Skip a property the request already settles for that item (and a key that names the same measure or aspect under another word, such as dimensions or size when a width or height is settled), a second name for a property already chosen, a key that lists purchase variants (a mix of sizes, colours and pack counts), identifiers, warranty and other logistics. With several items, put every item's hinted properties first, then take turns between the items.
"label" is the property's plain name: for a listed key, all of the key's words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property's name, no example values, no brackets, no second property.
Choose up to COUNT, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story's words it comes from, or an empty string"}]}"""


ALLOC = """

HOW MANY MESSAGES: MESSAGES in total, ITEMS items, one property of one item per message. The split between items is yours: begin with the item whose circumstances speak loudest, and while messages remain, reach every item once before returning to any of them. List the picks in sending order. An item that gets no message is chosen later from how well a listing matches the circumstances, so spend the messages where guessing wrong hurts most."""

WISHPICK = """You advise a shopping assistant. The listings below all pass the written requirements for the ITEM in the request that follows. The shopper keeps one firm wish unwritten, nearly always about a single property, and the circumstances in the request (the person it is for, the place and the use, the earlier disappointment, what the shopper insists on) usually betray it. From those circumstances alone, which listing is the shopper most likely to be happy with? Answer with ONE JSON object only: {"pick": "listing id", "property": "the property the unwritten wish is probably about", "reason": "one sentence"}"""


REFILL = """A shopping agent may send the shopper one more message naming up to seven properties of one requested product; the shopper answers about the first listed property they hold a firm preference on. Return ONE JSON object and nothing else: {"labels": ["property", ...]}: up to seven property names a shopper of this product plausibly holds a firm preference on, likeliest first, in the words such listings use as attribute keys, none already asked or settled (a key naming the same measure or aspect as one asked or settled, under another word, counts as asked). Prefer properties the request hints at, then those most such listings state. Return [] when nothing plausible is left."""

HEAR = """A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "unclear": [{"item": item number, "property": "a property the shopper was asked about and could not or would not answer (confusion, a shrug, no view on it)"}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "replies": [{"question": number, "kind": "value|no_preference|confusion|crowded|restated|other", "property": "the property the answer speaks about, in the answer's own words, or null"}], "next": {"form": "list|single", "properties": ["property", ...]}, "budget": new total limit as a number or null}
- "replies": one entry per question, read from the answer alone: value = it states a preference; no_preference = they do not mind, in any idiom that leaves the choice to the agent (no idea, any will do, you pick, no view on it, go with the flow); confusion = they did not follow or ask what was meant; crowded = they complain the message asked too many things; restated = they say the request already fixed that property ("I told you at the start ..."); other = anything else.
- "next": your advice for the agent's next message to the same product, read from how this shopper answers: "form" is "list" when they answer a list of properties (a value or a plain no-preference came back), "single" when they want one thing at a time, did not follow a list, or repeat the request; "properties" are the properties worth asking next, likeliest first, none the shopper has settled, waved off or been asked about already (empty when nothing is left).
- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3").
- When the answer names the property itself ("the pack type", "on sleeves", "size class of the dresser"), that name, in the answer's own words, is the need's property even when the question asked about another property; a question that listed several properties is answered about the one the answer names.
- An answer that turns down every value except one ("anything other than coconut gets a no", "nothing but oak", "only linen will do") states that one value as a positive need ("fruit type: coconut"), never as an exclusion. Only an answer that rules a value out without naming the wanted one ("not wicker", "anything but red") is an exclusion, written "not <value>".
- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop's product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.
- "free": the property of each question the shopper said they do not mind about.
- Folksy or roundabout wording that names a value still states that value."""


NOTE = """While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:
{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}
A replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note."""


CHECK = """You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.
Reading rules: a requirement is met when the listing's attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value's words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing's value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.
Every attribute the request itself states about an item (material, size or count words, colour, type, model or edition words, what it is for) is a requirement even when the requirements list does not repeat it: a cart line that satisfies the shopper's answers but not one of the request's own words is not the best option, and a listing shown that meets all of them is better.
When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.
The grader reads the listing's attribute table literally: an abbreviation or code does not state the value (an "L" is not "large"), an attribute for the property that names another value fails the listing, and a listing with no attribute at all for a stated property is doubtful; when another listing shown states that value outright in its attribute, the cart is wrong and that listing goes under "better".
Beyond what the request spells out, the shopper keeps one firm wish unwritten, nearly always about a single property of one item, and the circumstances in the request (the person it is for, the place and the use, the earlier disappointment, what the shopper insists on) usually betray it. If a listing shown besides the cart line satisfies every requirement in force and matches that wish where the cart line does not, put its id under "prefer"; under a cheapest, most or least rule do so only when the circumstances make the cart line an unlikely match. Without such a signal in the request, "prefer" stays empty.
Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first], "prefer": [ids of listings satisfying every requirement that match the unwritten wish where the cart line does not, strongest match first], "doubtful": true if the judgement leans on a requirement that no record speaks to, or on wording that leaves the match open; otherwise false, "support": {"<listing id>": [{"requirement": "<a requirement in force on that line, as listed>", "key": "<the record's attribute key that decides it, or title / description, or null when no record text decides it>", "value": "<the deciding text copied exactly from the record, or null>"}]}}. One support row for EVERY requirement in force on each cart line (stated by the request and answered by the shopper), key and value copied exactly from the record shown; a requirement with no deciding record text gets null and makes the cart not right. For an exclusion ("not X", "anything but X") cite the record's own value for that property (the row shows it is not X). A wish in words that names no value a listing could state needs no row. Use ok=false only for a problem the records clearly show or a requirement no record addresses."""


READING = """the listing's attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value's words counts, such as "deep teal" for teal, limits include the edge); a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing's value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit; an abbreviation or code is not the value it stands for, and a listing whose attribute for the property states the value outright beats one that only implies it. A requirement counts as met only by the listing's own attribute for that property stating the value, exactly for a number (30.3 is not 33) or within the stated bound; a value found only in the title or description, or under another attribute, counts only when no listing seen states it under its own attribute. Between a listing that states every requirement in its table and one that leaves a requirement unstated or merely close, the first is the fit and the second is not the best option."""

AUDIT = """A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: """ + READING + """
Records marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.
Return ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}"""

UNMIX = """A shopper's request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request's wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}"""

TRAIL = """A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing's page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement's chain from the requirement's own description, link by link, through the pages' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).
Return ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing}], "done": true when the last link is an OPENED page that states the property's value itself, else false, "value": the property's value as stated on that last page, or ""}]}
List each chain's links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown."""

GATE = """A shopping agent is about to STAGE. The shopper's request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product's properties do not do a step.
When a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.
Return ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}"""

SWAP = """A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: """ + READING + """
Return ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line's item, at most 6], "reason": "one sentence"}"""

PICK = """A shopping agent must check out now and has no listing it is sure of. An ending without a purchase is wrong whenever the store holds a fitting listing, so choose, for each requested item, the listing among the records shown that meets the most of its requirements (reading rules: """ + READING + """), preferring one that meets every stated requirement, then one whose attributes state the values outright, then the cheaper one. Only listings of the requested kind of product, in stock and inside the budget may be chosen.
Return ONE JSON object and nothing else: {"picks": {"item number": listing id or null}, "reason": "one sentence"}; null only when no record is of that item's kind."""

SHAPE = """A store tool returned the result below. Name its parts by the exact keys the result and the tool's parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing's price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}"""


def _dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _clip(value: Any, limit: int, depth: int = 0) -> Any:
    if depth > 7:
        return "..."
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "..."
    if isinstance(value, dict):
        return {k: _clip(v, limit, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip(v, limit, depth + 1) for v in value]
    return value


def _json_obj(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    text = text.strip()
    fence = re.match(r"^```[A-Za-z]*\s*(.*?)\s*```$", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        got = json.loads(text)
        return got if isinstance(got, dict) else None
    except ValueError:
        pass
    left, right = text.find("{"), text.rfind("}")
    if 0 <= left < right:
        try:
            got = json.loads(text[left:right + 1])
            return got if isinstance(got, dict) else None
        except ValueError:
            return None
    return None


def _words(key: str) -> str:
    key = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(key))
    return re.sub(r"[\s_.\-]+", " ", key).strip().lower()


def _stem(text: str) -> frozenset[str]:
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in _words(text).split())


_CODE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{1,5}-?\d{2,}[A-Za-z0-9-]*(?![A-Za-z0-9])")


def _code_tokens(text: str) -> list[str]:
    """Model-like codes in a requirement ("LUK1720", "MX-5", "RX100"): letters followed by digits, which stores keep in
    descriptions and compatibility lists rather than in attribute tables."""
    return list(dict.fromkeys(t for t in _CODE.findall(str(text)) if sum(c.isdigit() for c in t) >= 2))



def _code_like(values: list[str]) -> bool:
    """Most of the values are codes (one token mixing letters and digits, or a dashed number), as identifiers are; a measurement
    ("33.1 cm") or a word value is not."""
    codes = sum(1 for v in values if re.fullmatch(r"[a-z]{0,4}\d[a-z0-9-]{2,}|\d[\d-]{3,}", str(v).strip().lower()))
    return codes * 2 > len(values)


def _vague(value: str) -> bool:
    """A requirement value that is a wish in words (no figure, more than three words): the shopper holds the exact preference, so the
    property is still worth a question."""
    v = str(value).strip()
    return not re.search(r"\d", v) and len(v.split()) > 3

def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    if not prop or not key:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)


_FIGURE = r"\s+(?P<num>\d+(?:[.,]\d+)?)\s*(?P<unit>[a-z\"%°/]+)?\.?$"
_FLOOR = re.compile(r"^(?:at least|minimum|more than|over|above)" + _FIGURE)
_CEIL = re.compile(r"^(?:at most|maximum|less than|up to|under|below)" + _FIGURE)
_EXCL = re.compile(r"^(?:not|other than|anything but|anything other than|never|except)\s+(?P<rest>.+)$")


_INCH_MARKS = chr(34) + chr(8243)   # the straight double quote and the double prime, both written for inches
_FOOT_MARK = chr(39)                 # the apostrophe, written for feet


def _canon_unit(u: str) -> str:
    """One spelling for a unit word by rule (no vocabulary table): symbols, long names and plurals fold onto the short form."""
    u = str(u or "").lower().strip().rstrip(".")
    if u in _INCH_MARKS or u in ("in", "inch", "inches"):
        return "inch"
    if u == _FOOT_MARK or u in ("feet", "foot", "ft"):
        return "ft"
    u = re.sub(r"^(?:centimet(?:er|re)s?)$", "cm", u)
    u = re.sub(r"^(?:millimet(?:er|re)s?)$", "mm", u)
    u = re.sub(r"^(?:met(?:er|re)s?)$", "m", u)
    u = re.sub(r"^(?:kilograms?|kgs)$", "kg", u)
    u = re.sub(r"^(?:grams?)$", "g", u)
    u = re.sub(r"^(?:pounds?|lbs)$", "lb", u)
    u = re.sub(r"^(?:ounces?)$", "oz", u)
    u = re.sub(r"^(?:millilit(?:er|re)s?)$", "ml", u)
    u = re.sub(r"^(?:lit(?:er|re)s?)$", "l", u)
    u = re.sub(r"^(?:gigabytes?)$", "gb", u)
    u = re.sub(r"^(?:terabytes?)$", "tb", u)
    u = re.sub(r"^(?:megapixels?)$", "mp", u)
    u = re.sub(r"^(?:watts?)$", "w", u)
    u = re.sub(r"^(?:volts?)$", "v", u)
    u = re.sub(r"^(?:hours?|hrs?)$", "h", u)
    u = re.sub(r"^(?:percent)$", "%", u)
    u = re.sub(r"^(?:pieces?|pcs?)$", "piece", u)
    return u


def _canon_measure(text: str) -> str:
    """Numbers and units in one spelling: a price with a currency sign and thousands separators becomes a plain number with usd,
    an inch or foot mark becomes the unit word, a long or plural unit name becomes its short form."""
    s = str(text)
    s = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", s)
    s = re.sub(r"(\d) (\d{3})(?!\d)", r"\1\2", s)
    s = re.sub(r"[$]\s*(\d+(?:\.\d+)?)", r"\1 usd", s)
    s = re.sub(r"(\d)\s*(?:usd|dollars?)\b", r"\1 usd", s, flags=re.I)
    s = re.sub(r"(\d)\s*[" + re.escape(_INCH_MARKS) + r"]", r"\1 inch", s)
    s = re.sub(r"(\d)\s*" + re.escape(_FOOT_MARK) + r"(?!\w)", r"\1 ft", s)
    s = re.sub(r"(\d(?:\.\d+)?)\s*([A-Za-z%]{1,12})\b", lambda m: m.group(1) + " " + _canon_unit(m.group(2)), s)
    return s


def _unit(text: str) -> str:
    """A unit word reduced for comparison: lower case, no plural, no trailing dot; two units agree when one begins the other."""
    u = _canon_unit(str(text or ""))
    return u[:-1] if len(u) > 2 and u.endswith("s") else u


def _value_words(value: Any) -> frozenset[str]:
    """A value's words for a literal comparison: a number and its unit are one word, one- and two-letter words carry nothing."""
    low = re.sub(r"[^a-z0-9.%/]+", " ", _canon_measure(str(value)).lower())
    joined = re.sub(r"(\d)\s*([a-z]{1,6})\b", lambda m: m.group(1) + m.group(2).rstrip("s"), low)
    return frozenset(w for w in _stem(joined) if len(w) > 2 or any(c.isdigit() for c in w))


def _constraint(value: str) -> tuple[str, Any]:
    """A requirement value as a grader reads it: ("min" | "max", (figure, unit)), ("not", excluded words) or ("plain", words)."""
    low = re.sub(r"\s+", " ", _canon_measure(str(value)).lower()).strip().rstrip(".")
    for mode, pattern in (("min", _FLOOR), ("max", _CEIL)):
        m = pattern.match(low)
        if m:
            return mode, (float(m.group("num").replace(",", ".")), _unit(m.group("unit") or ""))
    m = _EXCL.match(low)
    if m:
        return "not", _value_words(m.group("rest"))
    return "plain", _value_words(low)


def _meets(mode: str, core: Any, shown: Any) -> bool | None:
    """Whether an attribute's value satisfies a constraint; None when it cannot be compared (no figure, another unit)."""
    text = _canon_measure(str(shown)).lower()
    if mode == "plain":
        return bool(core) and core <= _value_words(text)
    if mode == "not":
        return not (core and core <= _value_words(text))
    nums = re.findall(r"\d+(?:[.,]\d+)?", text)
    if not nums:
        return None
    want, unit = core
    m = re.search(r"\d+(?:[.,]\d+)?\s*([a-z\"%°/]+)", text)
    have = _unit(m.group(1)) if m else ""
    if unit and have and unit != have and not (unit.startswith(have[:2]) or have.startswith(unit[:2])):
        return None
    got = float(nums[0].replace(",", "."))
    return got >= want if mode == "min" else got <= want


def _aligned_keys(prop: str, keys: list[str]) -> list[str]:
    """The one attribute key spelt like the property (every property word shares its first four letters with a key word), for the
    shopper's wording the exact match missed; several candidates decide nothing."""
    want = [w[:4] for w in _words(prop).split() if len(w) >= 3]
    if not want:
        return []
    hits = [k for k in keys if all(any(w == kw[:4] for kw in _words(k).split() if len(kw) >= 3) for w in want)]
    return hits if len(hits) == 1 else []


def _kinds(schema: dict[str, Any]) -> list[str]:
    kind = schema.get("type")
    if isinstance(kind, str):
        return [kind]
    return [k for k in kind or [] if isinstance(k, str)]


def _fits_kind(value: Any, kind: str) -> bool:
    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "array":
        return isinstance(value, list)
    if kind == "object":
        return isinstance(value, dict)
    if kind == "null":
        return value is None
    return True


def _shape(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    """Bend a value toward its JSON schema; list what cannot be fixed."""
    if not isinstance(schema, dict):
        return value
    kinds = _kinds(schema)
    if kinds and not any(_fits_kind(value, k) for k in kinds):
        for kind in kinds:
            if kind == "integer" and isinstance(value, float) and value.is_integer():
                value = int(value)
                break
            if kind in ("integer", "number") and isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
                number = float(value)
                value = int(number) if kind == "integer" or number.is_integer() else number
                break
            if kind == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
                value = str(value)
                break
            if kind == "boolean" and isinstance(value, str) and value.strip().lower() in ("true", "false"):
                value = value.strip().lower() == "true"
                break
            if kind == "array" and not isinstance(value, list) and value is not None:
                value = [value]
                break
        if not any(_fits_kind(value, k) for k in kinds):
            faults.append(f"{where} should be {'/'.join(kinds)}")
            return value
    options = schema.get("enum")
    if isinstance(options, list) and value not in options:
        if isinstance(value, str):
            low = value.strip().lower()
            match = [o for o in options if isinstance(o, str) and o.lower() == low]
            if match:
                value = match[0]
        if value not in options:
            faults.append(f"{where}: allowed values are {options}")
    if isinstance(value, str) and isinstance(schema.get("maxLength"), int) and len(value) > schema["maxLength"]:
        value = value[: schema["maxLength"]]
    if isinstance(value, str) and isinstance(schema.get("minLength"), int) and len(value) < schema["minLength"]:
        faults.append(f"{where} is too short")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(schema.get("minimum"), (int, float)) and value < schema["minimum"]:
            faults.append(f"{where} is below {schema['minimum']}")
        if isinstance(schema.get("maximum"), (int, float)) and value > schema["maximum"]:
            faults.append(f"{where} is above {schema['maximum']}")
    if isinstance(value, list):
        if isinstance(schema.get("maxItems"), int) and len(value) > schema["maxItems"]:
            faults.append(f"{where} allows at most {schema['maxItems']} entries")
        if isinstance(schema.get("minItems"), int) and len(value) < schema["minItems"]:
            faults.append(f"{where} needs at least {schema['minItems']} entries")
        if isinstance(schema.get("items"), dict):
            value = [_shape(v, schema["items"], f"{where}[{i}]", faults) for i, v in enumerate(value)]
    if isinstance(value, dict):
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        if schema.get("additionalProperties") is False:
            for name in [n for n in value if n not in props]:
                faults.append(f"{where}.{name} is not an allowed argument")
        for name in schema.get("required") or []:
            if name not in value:
                faults.append(f"{where}.{name} is required")
        value = {k: _shape(v, props.get(k), f"{where}.{k}", faults) if k in props else v for k, v in value.items()}
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
    if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.0+)?\s*", value):
        return int(float(value))
    return None


def _id_like(value: Any, named: bool = False) -> bool:
    """An identifier-looking value: a number, or a short token with a digit; under a key named like an id, any short token."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return True
    if not isinstance(value, str):
        return False
    v = value.strip()
    if not 2 <= len(v) <= 64:
        return False
    if named:
        return True
    return " " not in v and (bool(re.search(r"\d", v)) or bool(re.fullmatch(r"[A-Za-z]{1,4}[-_][A-Za-z0-9]{3,}", v)))


def _learn_id(rows: list[dict[str, Any]]) -> str | None:
    """The key that identifies rows: string values with a digit and no spaces, unique per row, longest on average."""
    best = None
    for key in rows[0]:
        values = [row.get(key) for row in rows]
        if not all(_id_like(v, "id" in str(key).lower()) for v in values):
            continue
        values = [str(v).strip() for v in values]
        if len(rows) > 1 and len(set(values)) < len(values):
            continue
        score = ("id" in str(key).lower(), sum(len(v) for v in values) / len(values))
        if best is None or score > best[0]:
            best = (score, key)
    return best[1] if best else None


def _in_rows(node: Any, ident: str, depth: int = 0) -> bool:
    """Whether a list in the result holds a record carrying this id (a cart listing still showing the line)."""
    if depth > 5:
        return False
    if isinstance(node, list):
        return any(isinstance(v, dict) and ident in [str(x).strip() for x in v.values() if isinstance(x, str)] for v in node) or any(_in_rows(v, ident, depth + 1) for v in node)
    if isinstance(node, dict):
        return any(_in_rows(v, ident, depth + 1) for v in node.values())
    return False


_TEXT_PAIR = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]{0,30}):\s(.*)$", re.S)


def _textval(raw: str) -> Any:
    if raw in ("None", "null", ""):
        return None
    if raw in ("True", "true"):
        return True
    if raw in ("False", "false"):
        return False
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    if re.fullmatch(r"-?\d+\.\d+", raw):
        return float(raw)
    return raw


def _textrow(text: Any) -> dict[str, Any] | None:
    """A record the store serves as one text line ("ref_ref: p-1 | title: ... | price: 1099 | facts: colour=red; size=large | ...")
    read back into a dict, a "k=v; k=v" part into a nested dict; None for ordinary text (fewer than three "key: value" parts)."""
    if not isinstance(text, str) or " | " not in text or ": " not in text[:60]:
        return None
    out: dict[str, Any] = {}
    last: str | None = None
    for part in re.split(r"\s\|\s", text):
        m = _TEXT_PAIR.match(part)
        if m:
            key = m.group(1).strip().lower()
            raw = m.group(2).strip()
            pairs = [kv for kv in re.split(r";\s*", raw) if "=" in kv] if "=" in raw else []
            if pairs and (len(pairs) >= 2 or key in ("facts", "attributes", "specs", "details", "properties")) and all(len(kv.split("=", 1)[0]) <= 40 for kv in pairs):
                out[key] = {kv.split("=", 1)[0].strip().lower(): kv.split("=", 1)[1].strip() for kv in pairs}
            else:
                out[key] = _textval(raw)
            last = key
        elif last is not None:
            if isinstance(out[last], dict):
                for kv in re.split(r";\s*", part):
                    if "=" in kv:
                        k2, v2 = kv.split("=", 1)
                        out[last][k2.strip().lower()] = v2.strip()
            else:
                out[last] = f"{out[last]} | {part}" if out[last] is not None else part
    return out if len(out) >= 3 else None


FORMAT_MEMORY: dict[str, Any] = {}   # the record layout a model described during this run (one store serves a whole run; a fresh process knows nothing)

FORMAT_PROMPT = """A store answered a tool call with the text below. It describes one product listing or a list of listings in some text layout. Return ONE JSON object and nothing else:
{"records": [{"id": "the listing's identifier as written", "title": "its title as written", "price": a number or null, "fields": {"every other field name exactly as written": "its value as written"}, "attributes": {"attribute name as written": "its value as written"}}],
 "format": {"record_sep": "the exact text between two listings, or null when the text holds one listing", "field_sep": "the exact text between two fields", "kv_sep": "the exact text between a field's name and its value", "attributes_field": "the field whose value lists the listing's attributes, or null", "attr_sep": "the exact text between two attributes, or null", "attr_kv": "the exact text between an attribute's name and its value, or null"}}
Add "attributes_shape": one of "dict" (name: value inside one field), "pairs" (a list of name/value entries), "kv_string" (name=value items in one text), "prefixed" (attribute names carry a common prefix), "none"; and "rows_shape": one of "text" (each listing one text), "dicts", "arrays_with_header".
Copy every separator exactly, spaces included. Keep field and attribute names exactly as the text spells them. Read every listing the text holds."""


def _kv_lines(s: str) -> dict[str, Any] | None:
    """'key: value' or 'key = value' lines, one field a line (three or more of them)."""
    lines = [ln for ln in s.splitlines() if ln.strip()]
    if len(lines) < 3:
        return None
    out: dict[str, Any] = {}
    for ln in lines:
        m = re.match(r"^\s*[-*\u2022]?\s*([A-Za-z_][A-Za-z0-9_ ]{0,40}?)\s*[:=]\s*(.*)$", ln)
        if not m:
            continue
        key = m.group(1).strip().lower()
        raw = m.group(2).strip()
        pairs = [kv for kv in re.split(r";\s*", raw) if "=" in kv] if "=" in raw else []
        out[key] = {kv.split("=", 1)[0].strip().lower(): kv.split("=", 1)[1].strip() for kv in pairs} if len(pairs) >= 2 else _textval(raw)
    return out if len(out) >= 3 and len(out) >= 0.6 * len(lines) else None


def _decode_with(s: str, spec: dict[str, Any]) -> Any:
    """Read text with the separators a model described: {record_sep, field_sep, kv_sep, attributes_field, attr_sep, attr_kv}."""
    try:
        fs = str(spec.get("field_sep") or "")
        kv = str(spec.get("kv_sep") or "")
        if not fs or not kv or fs == kv:
            return None
        rs = spec.get("record_sep")
        chunks = [c for c in (s.split(str(rs)) if rs else [s]) if c.strip()]
        af = str(spec.get("attributes_field") or "").strip().lower()
        asep = str(spec.get("attr_sep") or "")
        akv = str(spec.get("attr_kv") or "")
        recs = []
        for chunk in chunks:
            out: dict[str, Any] = {}
            for part in chunk.split(fs):
                if kv not in part:
                    continue
                key, raw = part.split(kv, 1)
                key = key.strip().lower()
                raw = raw.strip()
                if not key or len(key) > 40:
                    continue
                if af and key == af and asep and akv:
                    out[key] = {p.split(akv, 1)[0].strip().lower(): p.split(akv, 1)[1].strip() for p in raw.split(asep) if akv in p}
                else:
                    out[key] = _textval(raw)
            if len(out) >= 3:
                recs.append(out)
        if not recs:
            return None
        return recs if (len(recs) > 1 or rs) else recs[0]
    except Exception:
        return None


def _decode_text(text: Any) -> Any:
    """Text-serialised record(s) read back into dict(s): JSON, 'k: v | k: v' lines, one-field-a-line text, then the layout a model
    described earlier in this run (FORMAT_MEMORY); None for ordinary prose such as a shopper's answer."""
    if not isinstance(text, str) or len(text) < 20:
        return None
    s = text.strip()
    spec = FORMAT_MEMORY.get("spec")
    if isinstance(spec, dict):   # the layout the model described for this store comes first
        got = _decode_with(s, spec)
        if got is not None:
            return got
    if s[:1] in "[{":
        try:
            got = json.loads(s)
            if isinstance(got, (dict, list)):
                return got
        except Exception:
            pass
    row = _textrow(s)
    if row is not None:
        return row
    row = _kv_lines(s)
    if row is not None:
        return row
    return None


def _clean_keys(rows: list[Any]) -> bool:
    """Field names a real layout produces: short, without JSON punctuation (a layout that splits a JSON dump yields keys with quotes/braces)."""
    bad = "{}[]" + chr(34)
    keys = [str(k) for r in rows if isinstance(r, dict) for k in r]
    return bool(keys) and all(len(k) <= 40 and not any(ch in k for ch in bad) for k in keys)


RAW_BY_ID: dict[str, str] = {}   # the store's own text of a record the code decoded, by its identifier (for the attribute follow-up)


def _remember_raw(decoded: Any, text: str) -> None:
    rec = decoded if isinstance(decoded, dict) else (decoded[0] if isinstance(decoded, list) and decoded and isinstance(decoded[0], dict) else None)
    if not isinstance(rec, dict):
        return
    for key, value in rec.items():
        if isinstance(value, str) and 2 <= len(value) <= 64 and " " not in value and (re.search(r"\d", value) or "id" in str(key).lower()):
            RAW_BY_ID[value.strip()] = text[:4000]
            if len(RAW_BY_ID) > 400:
                RAW_BY_ID.pop(next(iter(RAW_BY_ID)))
            return


def _split_pairs(text: str) -> dict[str, Any] | None:
    """Name/value items inside one field read by their own repeated separators (used only after the model's attribute follow-up
    failed): a name/value mark repeated two or more times and an item mark that splits the text into those items."""
    if not isinstance(text, str) or len(text) < 7 or len(text) > 3000:
        return None
    for kv in ("=", "~", ": ", "->", "=>", ":"):
        if text.count(kv) < 2:
            continue
        for sep in ("; ", " / ", " | ", ", ", chr(10), ";", "/"):
            items = [x.strip() for x in text.split(sep) if x.strip()]
            if len(items) < 2 or not all(kv in x for x in items):
                continue
            out = {}
            for x in items:
                name, value = x.split(kv, 1)
                name = name.strip().lower()
                if not name or len(name) > 30 or len(name.split()) > 3 or re.search(r"\d{4,}", name):
                    out = {}
                    break
                out[name] = value.strip()
            if len(out) >= 2:
                return out
    return None


def _unread_block(rec: Any) -> bool:
    """True when a record carries a string field that reads like an attribute block (repeated name/value marks) which no decoder has
    turned into a dict yet: the only situation worth a model question about the attribute layout. Search rows without a block are not."""
    if not isinstance(rec, dict) or any(isinstance(v, dict) and len(v) >= 2 for v in rec.values()):
        return False
    return any(isinstance(v, str) and len(v) >= 12 and _split_pairs(v) for v in rec.values())


def _structured_texts(node: Any, depth: int = 0) -> list[str]:
    """Every record-like text inside an observation (a page served as one line, rows served as lines), longest first."""
    out: list[str] = []
    if depth > 3:
        return out
    if isinstance(node, str):
        if _looks_structured(node):
            out.append(node)
    elif isinstance(node, dict):
        for v in node.values():
            out += _structured_texts(v, depth + 1)
    elif isinstance(node, list):
        for v in node[:6]:
            out += _structured_texts(v, depth + 1)
    return sorted(out, key=len, reverse=True)


def _looks_structured(text: str) -> bool:
    """Text that reads like a serialised record rather than prose: several field separators and a digit."""
    if not isinstance(text, str) or len(text) < 120 or text.lstrip()[:1] in "[{":
        return False
    marks = len(re.findall(r"\w\s*[:=]\s*\S", text)) + text.count(" | ") + text.count("\n")
    return marks >= 4 and bool(re.search(r"\d", text))


def _normalize_record(rec: dict[str, Any]) -> dict[str, Any]:
    """One record in the shape the code keeps: an attribute block as a nested dict whether it came as a dict, a list of {name, value}
    pairs, a 'k=v; k=v' string or flat prefixed keys (attr_colour …); rows given as arrays next to a header list become dicts; a price
    written as text ('$1,299.00') becomes a number; a numeric identifier becomes text."""
    out: dict[str, Any] = dict(rec)
    shape = str((FORMAT_MEMORY.get("spec") or {}).get("attributes_shape") or "")
    header = next((v for v in out.values() if isinstance(v, list) and len(v) >= 3 and all(isinstance(x, str) for x in v)), None)
    for key, value in list(out.items()):
        if isinstance(value, list) and value and all(isinstance(x, dict) for x in value) and len(value[0]) <= 4 and shape in ("", "pairs"):
            names = [k for k in value[0] if re.search(r"name|key|attr|prop|label|field|spec", str(k), re.I)]
            vals = [k for k in value[0] if k not in names and re.search(r"value|val|content|text|data", str(k), re.I)]
            if names and vals:
                out[key] = {str(x.get(names[0])).strip().lower(): x.get(vals[0]) for x in value if isinstance(x, dict) and x.get(names[0]) is not None}
        elif isinstance(value, list) and value and header is not None and all(isinstance(x, list) and len(x) == len(header) for x in value):
            out[key] = [dict(zip(header, row)) for row in value]
        elif isinstance(value, str) and _decode_text(value) is not None:
            out[key] = _decode_text(value)   # a record (or list of records) served as text under this key
            _remember_raw(out[key], value)
        elif isinstance(value, str) and value.count("=") >= 2 and len(value) < 3000 and " | " not in value and shape in ("", "kv_string", "dict"):
            pairs = [kv for kv in re.split(r";\s*|,\s*(?=[^=,;]{1,40}=)", value) if "=" in kv]
            if len(pairs) >= 2 and all(0 < len(kv.split("=", 1)[0].strip()) <= 40 for kv in pairs):
                out[key] = {kv.split("=", 1)[0].strip().lower(): kv.split("=", 1)[1].strip() for kv in pairs}
        elif isinstance(value, str) and re.search(r"price|cost|amount|total", str(key), re.I):
            m = re.fullmatch(r"\s*(?:[A-Z]{3}|[$€£¥₩])?\s*(-?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)\s*(?:[A-Z]{3}|[$€£¥₩])?\s*", value)
            if m:
                number = float(re.sub(r"[,\s]", "", m.group(1)))
                out[key] = int(number) if number.is_integer() else number
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and re.search(r"(^|_)(id|code|ref|sku|number)$|identifier", str(key), re.I):
            out[key] = str(value)
    if (FORMAT_MEMORY.get("attr_tries", 0) >= 1 and not (FORMAT_MEMORY.get("spec") or {}).get("attributes_field")
            and not any(isinstance(v, dict) for v in out.values())):
        # the model's attribute follow-up named no field: the string field whose own repeated separators yield the most name/value items is the block
        found = [(len(pairs), key, pairs) for key, value in out.items() if isinstance(value, str) for pairs in [_split_pairs(value)] if pairs]
        if found:
            found.sort(key=lambda x: -x[0])
            out[found[0][1]] = found[0][2]
    groups: dict[str, dict[str, Any]] = {}
    for key, value in out.items():
        m = re.match(r"^(attr|attribute|spec|specs|prop|property|feature)[_\-:. ](.+)$", str(key), re.I)
        if m and not isinstance(value, (dict, list)):
            groups.setdefault(m.group(1).lower(), {})[m.group(2).strip().lower()] = value
    for prefix, table in groups.items():
        if len(table) >= 2 and shape in ("", "prefixed") and not any(isinstance(v, dict) and len(v) >= 2 for v in out.values()):
            out[prefix + "_table"] = table
    return out


def _decode_deep(node: Any, depth: int = 0) -> Any:
    """Text-serialised records anywhere inside an observation read back into structures (lists of lines, a page as one line)."""
    if depth > 5:
        return node
    if isinstance(node, str):
        got = _decode_text(node)
        return got if got is not None else node
    if isinstance(node, list):
        return [_decode_deep(x, depth + 1) for x in node]
    if isinstance(node, dict):
        return {k: _decode_deep(v, depth + 1) for k, v in node.items()}
    return node


def _fact_block(node: Any) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    best = None
    for value in node.values():
        if isinstance(value, dict) and len(value) >= 2:
            scalars = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
            if len(scalars) >= 2 and (best is None or len(scalars) > len(best)):
                best = scalars
    return best


def _roles_by_words(enders: list[str], tools: list[dict[str, Any]], got: dict[str, Any]) -> dict[str, Any]:
    """Last resort without a model: the ending tools told apart by the words of their names and descriptions."""
    text = {t["function"]["name"]: re.sub(r"[^a-z]+", " ", (t["function"]["name"] + " " + str(t["function"].get("description") or "")).lower()) for t in tools}
    buy_words = ("order", "buy", "checkout", "purchase", "settle", "pay", "confirm")
    no_words = ("abstain", "refuse", "declin", "give up", "cancel", "quit", "without")
    def side(name: str) -> str | None:
        words = " " + text.get(name, name.lower()) + " "
        b = any(w in words for w in buy_words); n = any(w in words for w in no_words)
        return "buy" if b and not n else ("refuse" if n and not b else None)
    sides = {n: side(n) for n in enders}
    buys = [n for n, s in sides.items() if s == "buy"]; nos = [n for n, s in sides.items() if s == "refuse"]
    if len(enders) == 2 and len(buys) + len(nos) == 1:              # one side named: the other tool is the other ending
        other = next(n for n in enders if n not in buys + nos)
        (nos if buys else buys).append(other)
    got = dict(got)
    if len(buys) == 1 and not got.get("buy"):
        got["buy"] = buys[0]
    if len(nos) == 1 and not got.get("refuse"):
        got["refuse"] = nos[0]
    return got


ACCOUNT = """A shopping agent is placing the order below and owes the store a short account of it in plain prose. Write two to five sentences: name each ordered listing by its id and title with its price, quantity and stock as its record shows; for every requirement in force give the value the ordered listing's own record states for it (copy the record's wording; where a requirement takes its value from another listing's page, give that page's value too and say how the ordered value stands to it); then the budget in force and the order's total; and when the shopper or the store announced a change (a new budget, a requirement given up, a listing gone), say how the order honours it. Use only what the records and the ledger below show, and never restate a requirement without the value that meets it. Return ONE JSON object and nothing else: {"account": "..."}"""

SWEEP = """A shopping agent is about to check out. Listings that fit the request may still be unseen because long or specific search phrases hide them. Propose up to 3 NEW short search phrases for the cart's items: first the item's plain kind words exactly as the request names the item (two words), then the kind word plus ONE requirement value from the requirements in force, written the way a listing prints it (a number joined to its unit; a colour, brand or material as a plain word); never a value the requirements do not name. Do not repeat a phrase already searched. Return ONE JSON object and nothing else: {"phrases": ["...", "..."]}"""


RUN_SHOPPER: dict[str, int] = {"complaints": 0, "list_first": 0, "list_deep": 0, "singles": 0}   # how this process's shopper has met list questions so far (one shopper a run; a fresh process knows nothing)


def run_shopper_seen() -> dict[str, int]:
    return dict(RUN_SHOPPER)


def label_spot(labels: list[str], named: str) -> int | None:
    """Position in `labels` of the property an answer spoke about, by the words the reader quoted; None when none of them matches."""
    try:
        plain = _words(named)
        if not plain:
            return None
        for index, label in enumerate(labels):
            mine = _words(label)
            if mine == plain or _names(_stem(mine), _stem(plain)) or _names(_stem(plain), _stem(mine)):
                return index
    except Exception:
        pass
    return None


def run_shopper_note(labels: list[str], kind: str, named: str, complained: bool) -> None:
    """Record what one answer taught about this run's shopper: a complaint about a list, an answer about a list's first property only,
    an answer about a later listed property (the list was read whole), or an answered one-property question. Never fails."""
    try:
        if len(labels) > 1:
            if complained or kind == "crowded":
                RUN_SHOPPER["complaints"] += 1
            elif kind == "value":
                spot = label_spot(labels, named)
                RUN_SHOPPER["list_deep" if spot is not None and spot >= 1 else "list_first"] += 1
        elif kind in ("value", "no_preference"):
            RUN_SHOPPER["singles"] += 1
    except Exception:
        pass


def run_shopper_lists() -> bool | None:
    """True when this run's shopper has answered about a later property of a list (lists are read whole and complaints are rare);
    False after a complaint, or after three lists answered only on their first property, with no such answer; None while unknown."""
    deep, complaints, first = RUN_SHOPPER["list_deep"], RUN_SHOPPER["complaints"], RUN_SHOPPER["list_first"]
    if complaints >= 2 and complaints > deep:
        return False
    return True    # an answer about a list's first property says nothing about lists (a shopper reading the whole list answers the first one it minds too)


class Item:
    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []
        self.restated: list[str] = []   # properties the shopper said the request already fixed (asked in a list, repeated instead of answered)


class Kyo:
    @property
    def version(self) -> int:
        return self._version

    @version.setter
    def version(self, value: int) -> None:
        # the step of the last change of the requirement state: a cart line read before it must be read again before the order
        self._version = value
        self.version_step = getattr(self, "step", -99) if value else -99

    def __init__(self, problem: dict[str, Any]):
        environment = problem["environment"]
        self.binding = dict(environment["binding"])
        self.view = environment["policy_view"]
        self.request = str(self.view.get("query") or "")
        self.tools = [t for t in self.view.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict) and isinstance(t["function"].get("name"), str)]
        self.schemas = {t["function"]["name"]: t["function"].get("parameters") if isinstance(t["function"].get("parameters"), dict) else {"type": "object", "properties": {}} for t in self.tools}
        steps = _count(self.view.get("max_steps"))
        if steps is None and isinstance(self.view.get("max_steps"), float):
            steps = int(self.view["max_steps"])
        self.max_steps = steps if steps and steps > 0 else 30
        self.t0: float = time.monotonic()
        self.llm_secs: list[float] = []
        self.store_lat: list[float] = []
        self.in_env = 0
        self.closing = False
        self.guard_fired = False
        self.guard_thread: threading.Thread | None = None
        self.note_shapes: set[str] = set()
        self.notes_read = 0
        self.fallbacks = 0
        self.step = 0
        self.finished = bool(self.view.get("finished"))
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
        self.story_swapped = False
        self.doubt_opened = False
        self.qa: list[tuple[str, str]] = []
        self.rows: dict[str, dict[str, Any]] = {}
        self.full: dict[str, dict[str, Any]] = {}
        self.raw_texts: list[str] = []
        self.format_tries = 0
        self.learning_format = False
        self.store_secs: list[float] = []
        self.model_secs: list[float] = []
        self.dict_without_facts: dict[str, Any] | None = None
        self.seen_order: list[str] = []
        self.heard: set[str] = set()
        self.notes: list[str] = []
        self.version_step = -99
        self.version = 0
        self.cart_touched = False
        self.cart_at = -1
        self.cart_seen_at = -1
        self.cart_view: Any = None
        self.cart_lines: dict[str, int] = {}
        self.checked: set[str] = set()
        self.audited = False
        self.offer_seen: dict[str, int] = {}
        self.gone: dict[str, int] = {}
        self.unavailable: set[str] = set()
        self.sold_lines: set[str] = set()
        self.note_sold = False
        self.note_price = False   # the last note read said a price moved
        self.held = 0
        self.pending_changes: list[str] = []
        self.settle_holds = 0
        self.literal_holds = 0
        self.recent_notes: list[str] = []
        self.id_args: set[tuple[str, str]] = set()
        self.cart_said = ""
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
        self.chain_searches = 0   # searches made for a chain link nobody has found (two at most)
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
        self.search_log: list[tuple[str, dict[str, Any]]] = []
        self.swept: set[str] = set()
        self.wider_held = False
        self.paged_obs: list[Any] = []

    # ---------- plumbing ----------
    def left(self) -> int:
        return self.max_steps - self.step

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def room(self) -> float:
        """What is left of the sandbox's 300 s, less a margin."""
        return DEADLINE_S - self.elapsed()

    def llm_pace(self) -> float:
        """A model call's typical duration here: the median of the last five (6 s until any was made)."""
        recent = sorted(self.llm_secs[-5:])
        return max(2.0, recent[len(recent) // 2]) if recent else 6.0

    def pace(self) -> float:
        """A store call's typical duration here: the median of the last six (3 s until any was made)."""
        recent = sorted(self.store_lat[-6:])
        return max(1.0, recent[len(recent) // 2]) if recent else 3.0

    def allow(self, seconds: float) -> bool:
        """An optional stage that takes about `seconds` may start: the finish keeps STAGE_RESERVE after it."""
        return not self.closing and self.room() - seconds > STAGE_RESERVE

    def spend_ok(self, est: float = 0.004, final: bool = False) -> bool:
        """A model call costing about `est` fits the episode's share of the key; the final check and finish may use the reserve."""
        cap = SPEND_HARD + (SPEND_FINAL if final else 0.0)
        return self.cost + est <= cap

    def msgs_free(self) -> int:
        """Messages the interviewer may send: the ones the request's steps at the order or the ending take stay back."""
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)

    def cart_mark(self) -> str:
        return _dump(sorted(self.cart_lines.items()))

    def page_tools(self) -> set[str]:
        """Tools that show a listing's page: the offer view when the store has one apart from the attributes view
        (an attributes list alone is not the page), else the single-listing view; and the side-by-side view."""
        page = self.roles.get("offer") or self.roles.get("open")
        return {t for t in (page, self.roles.get("compare")) if t}

    def facts_tool(self) -> str | None:
        """The single-listing tool that shows attributes: one that has shown them, else the brief's attributes view."""
        known = next((t for t, ok in self.facts_from.items() if ok and self.single_target(t)), None)
        if known:
            return known
        tool = self.roles.get("open")
        return tool if self.facts_from.get(tool or "") is not False and self.single_target(tool) else None

    def compare_facts(self) -> bool | None:
        """Whether the side-by-side view shows attributes: True or False once seen, None before."""
        return self.facts_from.get(self.roles.get("compare") or "")

    def compare_bare(self) -> bool:
        """The side-by-side view is no way to read attributes: it showed none, or, not tried yet, the store has a separate
        attributes view (one that showed them, or one apart from the offer view)."""
        seen = self.compare_facts()
        return seen is False or (seen is None and (self.facts_from.get(self.facts_tool() or "") is True or bool(self.roles.get("offer"))))

    def gate_due(self, stage: str) -> bool:
        # steps due at this stage that the gate has not passed for the cart as it is now
        return any(p["when"] == stage for p in self.process) and (stage, self.cart_mark() + f"#{self.version}") not in self.gate_ok

    def tail(self) -> int:
        """Steps kept for the finish: the checkout, and one more while a step due at the order needs a message."""
        talk = any(p["when"] == "order" and p["messages"] > 0 for p in self.process)
        return 2 if talk and self.msg_used < self.msg_limit and self.gate_due("order") else 1

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None, final: bool = False, reasoning: dict[str, Any] | None = None) -> dict[str, Any] | None:
        for model in models:
            # no room, no credit or no budget: None, and the finish by code follows
            if self.calls >= ALL_CALLS or self.no_credit or self.closing or self.room() < STAGE_RESERVE or not self.spend_ok(0.004, final):
                return None
            self.calls += 1
            tries = 0
            while True:
                message, status = self.call_model(model, messages, max_tokens, tools, reasoning)
                if message is None and status == 0 and str((reasoning or {}).get("effort") or "") not in ("low", "none"):
                    # a reply with no content: the model's default thinking used the whole completion (race 14 smoke: the walker's call,
                    # twice in a row, then a forced finish); the same call once more with brief thinking
                    message, status = self.call_model(model, messages, max_tokens, tools, {"effort": "none" if model.startswith("anthropic/") else "low"})
                if message is not None:
                    return message
                if status == 402:
                    self.no_credit = True                     # an empty key stays empty: no wait helps, the finish by code follows
                    return None
                pause = self.backoff(tries, status)
                if pause is None:
                    break
                self.waited += WAITS[tries]
                tries += 1
                time.sleep(pause)
        return None

    def backoff(self, tries: int, status: Any) -> float | None:
        """Seconds to wait before the same call goes out again, or None when it should not: only rate limits, timeouts and server
        errors earn a retry, a timeout (which already took the client's 50 s) one at most, never past the waiting allowance, and
        never into the room the finish needs."""
        worth_it = status is None or status in {408, 425, 429} or (isinstance(status, int) and status >= 500)
        cap = 1 if status in (None, 408) else len(WAITS)
        if not worth_it or tries >= cap or self.waited + WAITS[tries] > WAIT_TOTAL:
            return None
        pause = WAITS[tries] * (0.8 + 0.4 * random.random())
        return pause if self.room() - pause >= STAGE_RESERVE + 10 else None

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None, reasoning: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None, Any]:
        """One inference call: the reply message, or None and the failure's status (0 for a reply without content)."""
        sent = messages
        if model.startswith("anthropic/") and len(messages) > 1:
            sent = [dict(m) for m in messages]
            marks = [0] + ([len(sent) - 1] if len(sent) > 1 else [])
            for index in marks:
                content = sent[index].get("content")
                if isinstance(content, str) and content:
                    sent[index]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        body: dict[str, Any] = {"model": model, "messages": sent, "max_tokens": max_tokens, "temperature": 0}
        if reasoning is None and model == HEAVY and HEAVY_EFFORT and not tools:   # fixed-form calls only; the walker keeps the default
            reasoning = {"effort": HEAVY_EFFORT}
        if reasoning:
            body["reasoning"] = reasoning
        if str((reasoning or {}).get("effort") or "") != "none":
            body["max_tokens"] = max(max_tokens, 2500)      # race 14 smoke: 400- and 1800-token replies came back as thinking alone (finish "length", no content)
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        started = time.monotonic()
        try:
            result = _llm.post_verbose("/inference/chat/completions", body)
        except Exception as exc:                              # a client that raises is a failed call, not the episode's end
            self.llm_secs.append(time.monotonic() - started)
            self.trace.append({"kind": "llm_fail", "model": model, "status": None, "body": f"{type(exc).__name__}: {exc}"[:200]})
            return None, None
        self.llm_secs.append(time.monotonic() - started)
        data = result.data
        if isinstance(data, dict) and data.get("choices"):
            usage = data.get("usage") or {}
            paid = usage.get("cost")
            if not paid and isinstance(usage.get("cost_details"), dict):
                paid = usage["cost_details"].get("upstream_inference_cost")   # a BYOK key: the proxy reports cost 0 and the real charge here
            if isinstance(paid, (int, float)):
                self.cost += float(paid)
            message = data["choices"][0].get("message") or {}
            if message.get("content") or message.get("tool_calls"):
                return message, 200
            details = usage.get("completion_tokens_details") if isinstance(usage.get("completion_tokens_details"), dict) else {}
            self.trace.append({"kind": "llm_fail", "model": model, "status": 0, "finish": data["choices"][0].get("finish_reason"), "reasoning_tokens": details.get("reasoning_tokens"), "completion_tokens": usage.get("completion_tokens")})
            return None, 0
        error = result.error or {}
        status = error.get("status")
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            # an error carried inside a 200 reply, with the provider's own code
            status = _count(data["error"].get("code"))
            error = {"body": data["error"].get("message")}
        self.trace.append({"kind": "llm_fail", "model": model, "status": status, "body": str(error.get("body") or "")[:200]})
        return None, status

    def ask_json(self, prompt: str, models: list[str], max_tokens: int = 1800, final: bool = False, reasoning: dict[str, Any] | None = None) -> dict[str, Any] | None:
        for model in models:
            message = self.think([{"role": "user", "content": prompt}], [model], max_tokens, final=final, reasoning=reasoning)
            got = _json_obj((message or {}).get("content"))
            if got is not None:
                return got
            if self.broke or self.no_credit or self.calls >= ALL_CALLS:
                return None
            if reasoning and str(reasoning.get("effort") or "") not in ("", "low", "none"):
                # a reasoning reply cut off at the token limit arrives without content (race 14 smoke: two empty reads in a row): the same
                # model once more with less thought and twice the room
                message = self.think([{"role": "user", "content": prompt}], [model], max_tokens * 2, final=final, reasoning={"effort": "none" if model.startswith("anthropic/") else "low"})
                got = _json_obj((message or {}).get("content"))
                if got is not None:
                    return got
        return None

    def act(self, name: str, args: dict[str, Any], guard: bool = False) -> dict[str, Any] | None:
        if self.closing and not guard:
            return None                                       # the watchdog has taken over the ending: the main thread sends nothing more
        payload = {**self.binding, "step": self.step + 1, "action": {"name": name, "args": args}}
        for attempt in range(4):
            if attempt and self.room() < 6:
                return None
            result = self.store_call(payload)
            if isinstance(result.data, dict):
                data = result.data
                self.step = data["step"] if isinstance(data.get("step"), int) else self.step + 1
                if data.get("finished"):
                    self.finished = True
                seen = data.get("observation")
                if name in self.page_tools() and not (isinstance(seen, dict) and seen.get("error")):
                    # listings whose full page the agent has opened (a search row is not that)
                    for value in args.values():
                        self.opened.update(v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str))
                return data
            error = result.error or {}
            status = error.get("status")
            body = str(error.get("body") or "")
            self.trace.append({"kind": "env_fail", "status": status, "body": body[:200]})
            if status == 409:
                low = body.lower()
                if "finish" in low or "quarantin" in low or "finaliz" in low:
                    self.finished = True
                    return None
                found = re.search(r"expected (\d+)", body)
                if found:
                    self.step = int(found.group(1)) - 1
                    payload["step"] = self.step + 1
                    continue
                if "reuse" in low or "different action" in low:
                    self.step += 1
                    payload["step"] = self.step + 1
                    continue
                return None
            if status in (400, 401, 402, 403, 404, 422):
                if status == 402:
                    self.broke = True
                return None
        return None

    def store_call(self, payload: dict[str, Any]) -> Any:
        """One round trip to the store, timed for pace(); the watchdog waits while one is in flight."""
        self.in_env += 1
        t = time.perf_counter()
        try:
            return _env.post_verbose("/environment/call", payload)
        finally:
            self.in_env -= 1
            self.store_lat.append(time.perf_counter() - t)

    # ---------- reading the request ----------
    def read_request(self) -> None:
        lines = []
        for tool in self.tools:
            fn = tool["function"]
            lines.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {_dump(fn.get('parameters') or {})}")
        prompt = BRIEF + "\nTOOLS:\n" + "\n".join(lines) + "\n\nREQUEST:\n" + self.request
        if self.view.get("opening") is not None:
            prompt += "\n\nOPENING NOTE FROM THE STORE:\n" + _dump(_clip(self.view.get("opening"), 1500))
        # the light model reads the request; the heavy one only when that reading names no item or no tool for a job
        # room for the light model's thinking: a reply cut at its limit sent a tenth of race 6's briefs to the heavy one
        brief = self.ask_json(prompt, [LIGHT], 4000) or {}
        if not self.no_credit and (not _items(brief.get("items")) or not isinstance(brief.get("roles"), dict) or not brief["roles"].get("buy")):
            brief = self.ask_json(prompt, [HEAVY, LIGHT], 2500) or brief
        self.brief = brief
        roles = brief.get("roles") if isinstance(brief.get("roles"), dict) else {}
        self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.schemas}
        for key in ("buy_args", "refuse_args"):
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        buy, refuse = self.roles.get("buy"), self.roles.get("refuse")
        if buy and buy == refuse and (not brief["buy_args"] or not brief["refuse_args"] or brief["buy_args"] == brief["refuse_args"]):
            ask = (f"The store tool {buy} both checks out and ends without a purchase, depending on its arguments. Its schema: "
                   + _dump(self.schemas.get(buy)) + "\nReturn ONE JSON object and nothing else: {\"buy_args\": the fixed arguments that check out, \"refuse_args\": the fixed arguments that end without a purchase} (only the arguments that choose the mode).")
            fixed = self.ask_json(ask, [LIGHT, HEAVY], 600) or {}
            if isinstance(fixed.get("buy_args"), dict) and isinstance(fixed.get("refuse_args"), dict) and fixed["buy_args"] != fixed["refuse_args"]:
                brief["buy_args"], brief["refuse_args"] = fixed["buy_args"], fixed["refuse_args"]
        for entry in _items(brief.get("items")):
            if not isinstance(entry, dict):
                continue
            quantity = _count(entry.get("quantity"))
            count = quantity if quantity and quantity > 0 else 1
            musts = [str(m) for m in _items(entry.get("musts")) if m]
            self.items.append(Item(str(entry.get("name") or "the product"), str(entry.get("shelf") or ""), count, musts))
        if not self.items:
            self.items = [Item("the requested product", "", 1, [])]
        for lookup in _items(brief.get("lookups")):
            text = str(lookup).strip()
            if not any(text in item.musts for item in self.items):
                # the lookup copied loosely: the must on the same property stands for it
                key = _words(text.split(":")[0])
                text = next((m for item in self.items for m in item.musts if _words(m.split(":")[0]) == key), text)
            number = next((n for n, item in enumerate(self.items, 1) if text in item.musts), 1)
            if text and (number, text) not in self.refs:
                self.refs.append((number, text))
        self.unmix()
        budget = brief.get("budget")
        if isinstance(budget, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", budget):
            budget = float(budget)
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0:
            self.budget = budget
        limit = _count(brief.get("messages"))
        if limit is not None and limit >= 0:
            self.msg_limit = limit
        elif self.roles.get("ask"):
            self.msg_limit = 3
        steps = _count(brief.get("steps"))
        if steps is not None and 10 <= steps < self.max_steps:
            self.max_steps = steps
        for entry in _items(brief.get("process")):
            if isinstance(entry, dict) and str(entry.get("step") or "").strip():
                when = str(entry.get("when") or "").strip().lower()
                self.process.append({"step": str(entry["step"]).strip(), "when": when if when in ("choose", "order", "end") else "order",
                                     "messages": max(0, _count(entry.get("messages")) or 0)})
        if self.roles.get("ask"):
            # the messages the steps at the order and the ending take, one at least for each of the two that needs any
            talk = [p["messages"] for p in self.process if p["when"] in ("order", "end") and p["messages"]]
            self.reserve = min(max(talk), self.msg_limit) if talk else 0
        self.roles_from_shape()

    def roles_from_shape(self) -> None:
        """The finishing roles when the brief named none (or a tool the store does not have): read off the schemas, since a
        world's names are new every race. A tool that takes no free text and no listing id (nothing, or only enums, numbers and
        booleans) ends the episode. One such tool with a two-way enum is both roles, its modes named by a model call over the
        schema; two such tools are told apart the same way; a lone argument-less tool is the order."""
        if self.roles.get("buy") and self.roles.get("refuse"):
            return
        taken = {v for k, v in self.roles.items() if k not in ("buy", "refuse") and v}
        plain = []
        for name, schema in self.schemas.items():
            props = schema.get("properties") or {}
            free = any(isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v for v in props.values())
            if name in taken or free:
                continue
            plain.append(name)
        if not plain:
            return
        self.trace.append({"kind": "roles_from_shape", "candidates": plain})
        if len(plain) == 1:
            name = plain[0]
            self.roles["buy"] = self.roles.get("buy") or name
            self.roles["refuse"] = self.roles.get("refuse") or name
            self.resolve_modes(name)
            return
        listing = "\n".join(f"- {n}: {_dump(self.schemas.get(n))}" for n in plain)
        ask = ("These store tools take no free text and no listing id, so they end the task. Which one checks out (buys the basket) and which ends "
               "without a purchase? Return ONE JSON object and nothing else: {\"buy\": tool name or null, \"refuse\": tool name or null, "
               "\"buy_args\": the fixed arguments that make the buy tool check out, else {}, \"refuse_args\": the fixed arguments that end without a purchase, else {}}\n" + listing)
        got = self.ask_json(ask, [LIGHT, HEAVY], 500) or {}
        if not (got.get("buy") or got.get("refuse")):
            got = _roles_by_words(plain, self.tools, got)       # no model answered: the names and descriptions decide
        for role in ("buy", "refuse"):
            tool = got.get(role)
            if isinstance(tool, str) and tool in plain and not self.roles.get(role):
                self.roles[role] = tool
        for key in ("buy_args", "refuse_args"):
            if isinstance(got.get(key), dict) and not self.brief.get(key):
                self.brief[key] = got[key]
        if self.roles.get("buy") and self.roles.get("buy") == self.roles.get("refuse"):
            self.resolve_modes(self.roles["buy"])

    def resolve_modes(self, tool: str) -> None:
        """One tool that both checks out and ends without a purchase: the model names the arguments of each mode from the schema."""
        brief = self.brief
        for key in ("buy_args", "refuse_args"):
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        schema = self.schemas.get(tool) or {}
        enums = [k for k, v in (schema.get("properties") or {}).items() if isinstance(v, dict) and isinstance(v.get("enum"), list) and len(v["enum"]) >= 2]
        if not enums or (brief["buy_args"] and brief["refuse_args"] and brief["buy_args"] != brief["refuse_args"]):
            return
        ask = (f"The store tool {tool} both checks out and ends without a purchase, depending on its arguments. Its schema: " + _dump(schema)
               + "\nReturn ONE JSON object and nothing else: {\"buy_args\": the fixed arguments that check out, \"refuse_args\": the fixed arguments that end without a purchase} (only the arguments that choose the mode).")
        fixed = self.ask_json(ask, [LIGHT, HEAVY], 600) or {}
        if isinstance(fixed.get("buy_args"), dict) and isinstance(fixed.get("refuse_args"), dict) and fixed["buy_args"] != fixed["refuse_args"]:
            brief["buy_args"], brief["refuse_args"] = fixed["buy_args"], fixed["refuse_args"]
            self.trace.append({"kind": "modes_resolved", "tool": tool})

    def unmix(self) -> None:
        """Musts that only describe the listing a lookup points at (read as musts of the product) move into that lookup."""
        for number, item in enumerate(self.items, 1):
            pointers = [need for n, need in self.refs if n == number and need in item.musts]
            rest = [m for m in item.musts if m not in pointers]
            if not pointers or not rest:
                continue
            prompt = (UNMIX + "\n\nREQUEST:\n" + self.request + f"\n\nPRODUCT: {item.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n"
                      + "\n".join(pointers) + "\nOTHER REQUIREMENTS:\n" + "\n".join(rest))
            got = self.ask_json(prompt, [HEAVY, LIGHT], 500) or {}
            moved = [m for m in rest if m in [str(x).strip() for x in _items(got.get("describe_other"))]]
            if not moved:
                continue
            item.musts = [m for m in item.musts if m not in moved]
            joined = pointers[0] + " (that listing: " + "; ".join(moved) + ")"
            item.musts = [joined if m == pointers[0] else m for m in item.musts]
            self.refs = [(n, joined if (n, need) == (number, pointers[0]) else need) for n, need in self.refs]
            self.trace.append({"kind": "unmix", "item": number, "moved": moved})

    def ledger(self) -> str:
        parts = []
        many = len(self.items) > 1
        for number, item in enumerate(self.items, 1):
            needs = [m + (f" (the page it points at gives: {self.ref_values[m]})" if m in self.ref_values else "") for m in item.musts + item.told if m not in item.dropped]
            head = f"item {number}: {item.name}" if many else f"item: {item.name}"
            if item.shelf:
                head += f" (store label: {item.shelf})"
            head += f", quantity {item.qty}"
            text = head + "; requirements: " + ("; ".join(needs) if needs else "none beyond the kind of product")
            if item.free:
                text += "; shopper does not mind: " + ", ".join(item.free)
            parts.append(text)
        brief = self.brief
        if self.budget is not None:
            scope = "for each item" if brief.get("budget_scope") == "each" else "in total"
            parts.append(f"budget: at most {self.budget} {brief.get('budget_unit') or ''} {scope}".replace("  ", " "))
        pick = brief.get("pick")
        if pick == "cheapest":
            parts.append("pick: the cheapest listing that fits (lowest price)")
        elif pick in ("most", "least"):
            parts.append(f"pick: the {pick} {brief.get('pick_measure') or ''} among fitting listings".replace("  ", " "))
        else:
            parts.append("pick: any listing that fits everything")
        if brief.get("in_stock_only", True) is not False:
            parts.append("in stock only")
        for wish in _items(brief.get("wishes")):
            parts.append(f"wish (tie-breaker only): {wish}")
        for rule in _items(brief.get("rules")):
            parts.append(f"rule from the request: {rule}")
        if brief.get("quiet_decline") is True:
            parts.append("if nothing fits: end without naming or suggesting any listing")
        for entry in self.process:
            when = {"choose": "before choosing", "order": "at the order (Kyo does it when you check out)", "end": "at an ending without a purchase (Kyo does it then)"}[entry["when"]]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        facts = self.facts_tool()
        if facts and (self.facts_from.get(facts) is True or self.roles.get("offer")):
            bare = sorted(t for t in self.page_tools() | {self.roles.get("offer") or ""} if t and self.facts_from.get(t) is False)
            pages = sorted(t for t in self.page_tools() if t != facts)
            parts.append(f"listing attributes come from {facts}" + (f"; {', '.join(bare)} show none" if bare else "")
                         + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ""))
        return "\n".join("- " + p for p in parts)

    def status(self) -> str:
        left = self.left()
        text = f"[Kyo] step {self.step} of {self.max_steps} used, {left} left."
        if left <= self.tail() + 1:
            text += " FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that)."
        elif left <= 6 or (left <= len(self.items) + 9 and self.buys == 0):
            # the first order keeps steps for what follows it: the store or the shopper answers most first orders with a change (a price, the
            # budget, a sell-out), and reading the lines again, fixing the cart and repeating the order take about five steps
            text += " Few steps left: set the cart to the chosen lines and check out now, so that steps remain for a change the store may answer with."
        elif self.buys == 0 and left <= 16:
            text += " Steps are running low: settle on your picks, fill the cart and check out within the next few steps."
        elif self.buys == 0 and not self.cart_touched and self.step >= self.max_steps // 2:
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        missing = self.missing_kinds()
        if missing and (left <= 14 or self.room() < 110):
            text += " No cart line yet for: " + "; ".join(missing[:3]) + " (each needs a search, an open and a cart call)."
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if lookups and self.buys == 0:
            text += "\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] " + " | ".join(lookups[:3])
        if self.leads:
            text += "\n[review candidates] " + ", ".join(self.leads)
        if self.notes:
            text += "\n[changes so far] " + " | ".join(self.notes[-4:])
        text += "\n[ledger]\n" + self.ledger()
        return text

    # ---------- listings ----------
    def ident(self, node: Any, in_list: bool = False, depth: int = 0) -> str | None:
        """A listing's id, read with the id key learned from result rows (or named by the structure learner); None for anything else."""
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if not self.id_key and FORMAT_MEMORY.get("id_key") in node and _id_like(node.get(FORMAT_MEMORY["id_key"]), True):
            self.id_key = str(FORMAT_MEMORY["id_key"])
        if self.id_key and _id_like(node.get(self.id_key), True):
            return str(node[self.id_key]).strip()
        nested = any(isinstance(v, dict) for v in node.values())
        if not (in_list or depth == 0 or nested):
            return None
        found = _learn_id([node])
        if found and ("id" in str(found).lower() or str(node[found]).strip() in self.rows):
            return str(node[found]).strip()
        return None

    def learn_rows(self, node: list[Any]) -> bool:
        rows = [n for n in node if isinstance(n, dict) and len(n) >= 3]
        if not rows or len(rows) < len(node) / 2:
            return False
        if self.id_key and all(_id_like(r.get(self.id_key), True) for r in rows):
            return True
        found = _learn_id(rows)
        if found:
            self.id_key = found
            return True
        return False

    def absorb(self, observation: Any) -> None:
        def walk(node: Any, depth: int, in_list: bool = False) -> None:
            if depth > 6:
                return
            if isinstance(node, str):   # a record served as text (race 15: one 'k: v | k: v' line): the model describes the layout first, code reads with it
                if _looks_structured(node) and not FORMAT_MEMORY.get("spec") and FORMAT_MEMORY.get("asked", 0) < 2 and not getattr(self, "learning_format", False):
                    self.learn_layout(node)
                row = _decode_text(node)
                spec_now = FORMAT_MEMORY.get("spec")
                if (isinstance(row, dict) and isinstance(spec_now, dict) and not spec_now.get("attributes_field") and not _fact_block(_normalize_record(row))
                        and FORMAT_MEMORY.get("attr_tries", 0) < 2 and not getattr(self, "learning_format", False)
                        and _unread_block(row)):
                    FORMAT_MEMORY["attr_tries"] = FORMAT_MEMORY.get("attr_tries", 0) + 1   # a page read without an attribute block: ask once which field holds it
                    self.learning_format = True
                    try:
                        self.learn_attributes(node, spec_now)
                    finally:
                        self.learning_format = False
                    row = _decode_text(node)
                if row is not None:
                    walk(row, depth, in_list)
                elif len(node) >= 150 and depth <= 3:
                    self.note_raw(node)
                return
            if isinstance(node, dict):
                node = _normalize_record(node)
                ident = self.ident(node, in_list, depth)
                if ident:
                    if ident not in self.rows:
                        self.seen_order.append(ident)
                        self.rows[ident] = {}
                    # a flag of the listing's own record turning from true to false is read as "can no longer be bought"
                    old = self.rows[ident]
                    if any(old.get(k) is True and v is False for k, v in node.items()):
                        self.unavailable.add(ident)
                    elif any(old.get(k) is False and v is True for k, v in node.items()):
                        self.unavailable.discard(ident)
                    spec_now = FORMAT_MEMORY.get("spec")
                    if (not _fact_block(node) and isinstance(spec_now, dict) and not spec_now.get("attributes_field") and ident in RAW_BY_ID
                            and FORMAT_MEMORY.get("attr_tries", 0) < 2 and not getattr(self, "learning_format", False)
                            and _unread_block(node)):
                        FORMAT_MEMORY["attr_tries"] = FORMAT_MEMORY.get("attr_tries", 0) + 1   # a page read without an attribute block: ask once which field holds it
                        self.learning_format = True
                        try:
                            self.learn_attributes(RAW_BY_ID[ident], spec_now)
                        finally:
                            self.learning_format = False
                        again = _decode_text(RAW_BY_ID[ident])
                        if isinstance(again, dict):
                            node = _normalize_record(again)
                    self.rows[ident].update(node)
                    if _fact_block(node):
                        self.full[ident] = dict(self.rows[ident])
                    elif ident in self.full:
                        self.full[ident].update(node)
                    elif len(node) >= 6 and depth <= 1 and self.id_key and self.id_key in node:
                        self.dict_without_facts = node       # a page-sized listing record (the store's own id key) with no attribute block the code can read
                    return
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                node = [(_decode_text(x) if isinstance(x, str) else None) or x for x in node]
                rows = self.learn_rows(node)
                for value in node:
                    walk(value, depth + 1, rows)

        walk(observation, 0)
        self.maybe_learn_format()

    def note_raw(self, text: str) -> None:
        """Long store text no decoder could read: kept for the format learner and as the planner's evidence of last resort."""
        self.raw_texts = (getattr(self, "raw_texts", []) + [text[:4000]])[-4:]

    def raw_hint(self) -> str:
        """What the planner sees when no listing record could be parsed: the store's own text of the last listings."""
        raws = getattr(self, "raw_texts", [])
        if not raws:
            return "(no full listing records yet)"
        self.trace.append({"kind": "raw_hint", "n": len(raws)})
        return ("(no listing record could be parsed from the store's replies; the store's own text of the last listings follows: read the attribute names it states)\n"
                + "\n---\n".join(r[:1500] for r in raws[-2:]))

    def flatten_model_record(self, rec: dict[str, Any]) -> dict[str, Any] | None:
        """A record as the format learner returns it -> the flat dict the record store keeps (id, title, price, other fields, attributes)."""
        ident = str(rec.get("id") or "").strip()
        if not ident:
            return None
        flat: dict[str, Any] = {"id": ident}
        if rec.get("title") is not None:
            flat["title"] = rec.get("title")
        if rec.get("price") is not None:
            flat["price"] = rec.get("price")
        fields = rec.get("fields") if isinstance(rec.get("fields"), dict) else {}
        for key, value in fields.items():
            flat[str(key).strip().lower()] = value
        attrs = rec.get("attributes") if isinstance(rec.get("attributes"), dict) else {}
        if attrs:
            flat["attributes"] = {str(k).strip().lower(): v for k, v in attrs.items()}
        return flat if len(flat) >= 3 else None

    def learn_layout(self, sample: str) -> bool:
        """One light call on the first record-like text of the run: the model returns the layout (separators, attribute and row shapes)
        and the records it read; the layout is kept for the run when it reads the sample back, else the code's decoders take over."""
        FORMAT_MEMORY["asked"] = FORMAT_MEMORY.get("asked", 0) + 1
        self.learning_format = True
        try:
            got = self.ask_json(FORMAT_PROMPT + "\n\nSTORE TEXT:\n" + sample[:3000], [LIGHT, HEAVY], 2000) or {}
            spec = got.get("format") if isinstance(got.get("format"), dict) else None
            if spec:
                for key in ("attributes_shape", "rows_shape"):
                    if got.get(key):
                        spec[key] = str(got.get(key))
                read = _decode_with(sample, spec)
                rows = read if isinstance(read, list) else ([read] if isinstance(read, dict) else [])
                if rows and _clean_keys(rows) and any(_learn_id([r]) or any(_id_like(v, True) and "id" in str(k).lower() for k, v in r.items()) for r in rows if isinstance(r, dict)):
                    FORMAT_MEMORY["spec"] = spec
                    FORMAT_MEMORY["stale"] = False
                    if not spec.get("attributes_field") and str(spec.get("attributes_shape") or "") != "none":
                        self.learn_attributes(sample, spec)   # the attribute block matters most to the planner: one focused call when the layout left it out
                    self.trace.append({"kind": "format_learned", "first": True, "spec": str(spec)[:240], "sample": sample[:100]})
                    return True
            self.trace.append({"kind": "format_fallback", "n": len(_items(got.get("records")))})
            for rec in _items(got.get("records")):
                flat = self.flatten_model_record(rec) if isinstance(rec, dict) else None
                if flat:
                    self.absorb(flat)
            return False
        finally:
            self.learning_format = False

    def learn_attributes(self, sample: str, spec: dict[str, Any]) -> None:
        """A layout whose attribute block the model did not name: one focused call asks which field lists the listing's attributes and
        what separates the items and names from values; kept only when it reads two or more attributes from the sample."""
        prompt = ("A store serves each product listing as text. Its fields are separated by " + repr(spec.get("field_sep")) + " and a field's name and value by " + repr(spec.get("kv_sep")) + ". "
                  "One field holds the listing's ATTRIBUTES (several name/value items such as colour, size, material). Return ONE JSON object and nothing else: "
                  "{\"attributes_field\": \"that field's name exactly as written, or null when no field lists attributes\", \"attr_sep\": \"the exact text between two attribute items\", \"attr_kv\": \"the exact text between an attribute's name and its value\"}\n\nLISTING TEXT:\n" + sample[:2500])
        got = self.ask_json(prompt, [LIGHT, HEAVY], 600) or {}
        trial = dict(spec)
        for key in ("attributes_field", "attr_sep", "attr_kv"):
            if got.get(key) is not None:
                trial[key] = str(got.get(key))
        read = _decode_with(sample, trial)
        rows = read if isinstance(read, list) else ([read] if isinstance(read, dict) else [])
        if any(isinstance(r, dict) and any(isinstance(v, dict) and len(v) >= 2 for v in r.values()) for r in rows):
            spec.update({k: trial[k] for k in ("attributes_field", "attr_sep", "attr_kv") if k in trial})
            self.trace.append({"kind": "attributes_learned", "field": spec.get("attributes_field"), "sep": spec.get("attr_sep"), "kv": spec.get("attr_kv")})
        else:
            self.trace.append({"kind": "attributes_failed", "got": str(got)[:160]})

    def maybe_learn_format(self) -> None:
        """A store whose records no decoder reads: a light model reads one such text into records and describes its separators; the code
        then reads the rest of this run with that description (FORMAT_MEMORY), asking again only when a description fails."""
        raws = [r for r in getattr(self, "raw_texts", []) if _looks_structured(r)]
        json_sample = False
        if not raws and getattr(self, "dict_without_facts", None) and not self.full:
            raws = [_dump(self.dict_without_facts)[:3000]]   # a record the code identified but whose attributes it could not read
            json_sample = True
        if not raws or self.full or getattr(self, "learning_format", False) or getattr(self, "format_tries", 0) >= 2:
            return
        if FORMAT_MEMORY.get("spec") and not FORMAT_MEMORY.get("stale"):
            return                                         # a learned layout exists and _decode_text already tried it on this text
        if FORMAT_MEMORY.get("failed", 0) >= 4:
            return
        self.learning_format = True
        self.format_tries = getattr(self, "format_tries", 0) + 1
        try:
            sample = raws[-1][:3000]
            got = self.ask_json(FORMAT_PROMPT + "\n\nSTORE TEXT:\n" + sample, [LIGHT, HEAVY], 2000) or {}
            spec = got.get("format") if isinstance(got.get("format"), dict) else None
            records = [r for r in _items(got.get("records")) if isinstance(r, dict)]
            read_v = _decode_with(sample, spec) if spec else None
            if spec and not json_sample and read_v is not None and _clean_keys(read_v if isinstance(read_v, list) else [read_v]):
                FORMAT_MEMORY["spec"] = spec
                FORMAT_MEMORY["stale"] = False
                self.trace.append({"kind": "format_learned", "spec": str(spec)[:240]})
                for raw in list(raws):
                    decoded = _decode_text(raw)
                    if decoded is not None:
                        self.absorb(decoded)
            else:
                FORMAT_MEMORY["failed"] = FORMAT_MEMORY.get("failed", 0) + 1
                if FORMAT_MEMORY.get("spec"):
                    FORMAT_MEMORY["stale"] = True
                self.trace.append({"kind": "format_records_only", "n": len(records)})
                for rec in records:
                    flat = self.flatten_model_record(rec)
                    if flat:
                        self.absorb(flat)
        finally:
            self.learning_format = False


    def fill_required(self, tool: str | None, args: dict[str, Any], kind: str) -> dict[str, Any]:
        """Required arguments of a finishing tool the brief did not fix: an enum takes its first option, a free text takes the code's own
        one-line account of the ending, a number 0, a flag true, a list [] (a store that wants a justification gets one)."""
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        out = dict(args)
        for key in schema.get("required") or []:
            if key in out and out[key] not in (None, ""):
                continue
            spec = props.get(key) or {}
            kinds = _kinds(spec) if isinstance(spec, dict) else []
            if isinstance(spec, dict) and isinstance(spec.get("enum"), list) and spec["enum"]:
                out[key] = spec["enum"][0]
            elif "string" in kinds or not kinds:
                out[key] = self.finish_text(kind)
            elif "integer" in kinds or "number" in kinds:
                out[key] = 0
            elif "boolean" in kinds:
                out[key] = True
            elif "array" in kinds:
                out[key] = []
            elif "object" in kinds:
                out[key] = {}
        return out

    def finish_text(self, kind: str) -> str:
        """A plain account of the ending for a store that asks for one."""
        if kind == "buy":
            lines = [f"{ident} x{count}" for ident, count in (self.cart_lines or {}).items()] or list(self.cart_ids())
            return ("Checking out " + (", ".join(lines) if lines else "the cart") + ": each line is the listing whose attributes state the request's requirements and the shopper's answers.")[:400]
        return "Ending without a purchase: no listing seen states every requirement of the request, so nothing is bought."

    def learn_reply_key(self, observation: Any, question: str) -> None:
        """A reply whose shape leaves several candidate texts: one light call names the field that holds the shopper's words (kept for the run)."""
        if FORMAT_MEMORY.get("reply_key") or FORMAT_MEMORY.get("reply_key_tries", 0) >= 2 or not isinstance(observation, dict):
            return
        FORMAT_MEMORY["reply_key_tries"] = FORMAT_MEMORY.get("reply_key_tries", 0) + 1
        prompt = ("A shopping agent asked a shopper a question and the store returned the JSON below. Return ONE JSON object and nothing else: "
                  "{\"reply_key\": \"the top-level key whose value is the shopper's own answer\", \"notice_keys\": [\"keys whose values are the store's or shopper's side notices, if any\"]}\n\n"
                  "QUESTION:\n" + question[:300] + "\n\nREPLY JSON:\n" + _dump(_clip(observation, 1200))[:3000])
        got = self.ask_json(prompt, [LIGHT, HEAVY], 600) or {}
        key = str(got.get("reply_key") or "").strip()
        if key and key in observation and isinstance(observation.get(key), str):
            FORMAT_MEMORY["reply_key"] = key
            self.trace.append({"kind": "reply_key_learned", "key": key})

    def labels(self) -> set[str]:
        found: set[str] = set()

        def walk(schema: Any, depth: int) -> None:
            if depth > 6 or not isinstance(schema, dict):
                return
            for option in schema.get("enum") or []:
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
                if isinstance(value, str) and 0 < len(value) <= 40 and not re.search(r"\d{3}", value):
                    counts.setdefault(key, set()).add(value)
                    present[key] = present.get(key, 0) + 1
        total = max(1, len(self.rows))
        known = self.labels()
        best = None
        for key, values in counts.items():
            if present[key] < 0.8 * total or len(values) > 40:
                continue
            listed = sum(1 for v in values if v.strip().lower() in known) / len(values)
            mean = sum(len(v) for v in values) / len(values)
            score = (-listed, len(values), -present[key], -mean)
            if best is None or score < best[0]:
                best = (score, key)
        return best[1] if best else None

    def key_table(self) -> list[tuple[str, str, int, list[str]]]:
        """(category, key name, listings carrying it, sample values), most carried first per category."""
        cat_key = self.category_key()
        groups: dict[str, dict[str, Any]] = {}
        for ident in self.seen_order:
            record = self.full.get(ident)
            if not record:
                continue
            block = _fact_block(record) or {}
            label = str(record.get(cat_key) or "listings") if cat_key else "listings"
            group = groups.setdefault(label, {"n": 0, "keys": {}})
            group["n"] += 1
            done: set[str] = set()
            for key, value in block.items():
                name = _words(key)
                if not name or name in done:
                    continue
                done.add(name)
                slot = group["keys"].setdefault(name, {"n": 0, "values": [], "per": {}})
                slot["n"] += 1
                text = str(value).strip().lower()
                slot["per"][ident] = text
                if text not in slot["values"]:
                    slot["values"].append(text)
        wanted = [(_words(item.shelf or ""), _words(item.name), {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}) for item in self.items]
        matched = {label for label in groups if any(_words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long for shelf, name, long in wanted)}
        if matched:
            groups = {label: group for label, group in groups.items() if label in matched}
        table = []
        stats: dict[str, tuple[int, int, bool]] = {}
        values_of: dict[str, set[str]] = {}
        for label, group in sorted(groups.items(), key=lambda kv: -kv[1]["n"]):
            kept: list[tuple[str, dict[str, Any]]] = []
            for name, slot in sorted(group["keys"].items(), key=lambda kv: -kv[1]["n"]):
                values = list(slot["per"].values())
                if not values:
                    continue
                mean = sum(len(v) for v in values) / len(values)
                if mean > 40 or sum(1 for v in values if " / " in v) * 2 >= len(values):
                    continue
                if len(values) >= 4 and len(set(values)) == len(values) and mean > 12:
                    continue
                twin = any(name.startswith(other + " ") and slot["per"] == spec["per"] for other, spec in kept)
                if twin:
                    continue
                kept.append((name, slot))
            for name, slot in kept[:20]:
                table.append((label, name, slot["n"], [v[:40] for v in slot["values"][:4]]))
                # (listings carrying the key, distinct values, code-shaped values): an identifier is distinct on every listing AND coded;
                # a measurement is distinct too but is a property a shopper may want (race 14: depth, width)
                stats[name] = (slot["n"], len(set(slot["per"].values())), _code_like(list(slot["per"].values())))
                values_of.setdefault(name, set()).update(slot["per"].values())
        self.key_stats = stats
        self.key_values = values_of
        return table

    def item_groups(self) -> list[list[str]]:
        """Listing ids per category, keeping the categories that match a requested item (largest first)."""
        cat_key = self.category_key()
        groups: dict[str, list[str]] = {}
        for ident in self.seen_order:
            label = str(self.rows[ident].get(cat_key) or "") if cat_key else ""
            groups.setdefault(label, []).append(ident)
        wanted = [(_words(item.shelf or ""), _words(item.name), {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}) for item in self.items]
        matched = [ids for label, ids in groups.items() if any(_words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long for shelf, name, long in wanted)]
        pool = matched or list(groups.values())
        return sorted(pool, key=lambda ids: -len(ids))[: max(1, len(self.items))]

    def single_target(self, tool: str | None) -> str | None:
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        target = next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not tool or not target or any(k != target for k in schema.get("required") or []):
            return None
        return target

    def price_of(self, ident: str) -> float | None:
        """The listing's price as its newest record shows it, under the price key the model named for result rows."""
        value = (self.rows.get(ident) or {}).get(self.price_key) if self.price_key else None
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and not isinstance(v, bool)), None)
        if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", value):
            value = float(value)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def buyable(self, ident: str) -> bool:
        """Not sold out by any record: no note or refusal named it, and its stock flag (the key the model named) is not no."""
        if self.sold_out(ident):
            return False
        value = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        return not (value is False or (isinstance(value, (int, float)) and not isinstance(value, bool) and value <= 0))

    def records_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        """The listing records a result carries: the result itself, or the rows of its lists (two levels deep)."""
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
        """Which fields each viewing tool shows, and which a listing has shown since the last change: a view whose
        fields were all shown already is not run again."""
        for ident, record in self.records_in(observation):
            keys = set(record)
            self.tool_keys.setdefault(name, set()).update(keys)
            version, have = self.view_keys.get(ident, (-1, set()))
            self.view_keys[ident] = (self.version, (have if version == self.version else set()) | keys)

    def shown_already(self, name: str, ident: str) -> bool:
        version, have = self.view_keys.get(ident, (-1, set()))
        keys = self.tool_keys.get(name)
        return bool(keys) and version == self.version and keys <= have and ident not in self.gone

    # ---------- result pages ----------
    def read_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        """Which keys of a tool's row-list result hold the rows, the match count, the next-page value (and the parameter
        taking it), a row's price and the flag saying it can be bought: named once per tool by the light model, checked against the result."""
        if name in self.shapes:
            return self.shapes[name]
        if not isinstance(observation, dict):
            return None
        lists = [k for k, v in observation.items() if isinstance(v, list) and len(v) >= 2 and all(isinstance(r, dict) for r in v)]
        if not lists:
            return None
        self.shapes[name] = None
        sample = {k: (v[:3] if k in lists else v) for k, v in observation.items()}
        prompt = (SHAPE + "\n\nTOOL: " + name + " | parameters: " + _dump(self.schemas.get(name) or {})
                  + "\n\nRESULT (first rows only):\n" + _dump(_clip(sample, 120))[:4000])
        got = self.ask_json(prompt, [LIGHT, HEAVY], 1500) or {}
        rows = got.get("rows") if got.get("rows") in lists else None
        if not rows:
            return None
        first = observation[rows][0]
        props = (self.schemas.get(name) or {}).get("properties") or {}

        def named(key: Any, where: dict[str, Any]) -> str | None:
            return key if isinstance(key, str) and key in where else None

        shape = {"rows": rows, "count": named(got.get("count"), observation), "next": named(got.get("next"), observation),
                 "next_arg": named(got.get("next_arg"), props), "price": named(got.get("price"), first), "stock": named(got.get("stock"), first)}
        if shape["count"] and (isinstance(observation[shape["count"]], bool) or not isinstance(observation[shape["count"]], int)):
            shape["count"] = None
        self.shapes[name] = shape
        self.price_key = self.price_key or shape["price"]
        self.stock_key = self.stock_key or shape["stock"]
        self.trace.append({"kind": "shape", "tool": name, "shape": shape})
        return shape

    def paging(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        """(matches reported, rows shown, the parameter that asks for the next page, its value or None on the last page)
        by the tool's named result shape; None when the result has no such parts."""
        shape = self.read_shape(name, observation)
        if not shape or not shape["count"] or not shape["next"] or not shape["next_arg"] or not isinstance(observation, dict):
            return None
        rows, count, value = observation.get(shape["rows"]), observation.get(shape["count"]), observation.get(shape["next"])
        if not isinstance(rows, list) or not rows or isinstance(count, bool) or not isinstance(count, int):
            return None
        if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))):
            return None
        return count, len(rows), shape["next_arg"], value

    def price_sorted(self, observation: Any) -> bool:
        prices = [self.price_of(i) for i, _ in self.records_in(observation)]
        return len(prices) >= 3 and None not in prices and all(a <= b for a, b in zip(prices, prices[1:]))

    def more_pages(self, name: str, args: dict[str, Any], observation: Any, deep: bool = False) -> str:
        """A search that matched more listings than it showed, under a cheapest, most or least pick: Kyo reads a short
        rest itself (a step a page, no model turn); a long rest is named so the agent can narrow the search."""
        self.paged_obs = []
        pick = self.brief.get("pick")
        plan = self.paging(name, observation) if pick in ("cheapest", "most", "least") else None
        if not plan:
            return ""
        count, shown, param, value = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and not isinstance(given, bool) else bool(str(given or "").strip())
        if value is None or later or count <= shown:
            # the last page, or the agent is paging itself
            return ""
        if pick == "cheapest" and self.price_sorted(observation):
            return ""
        limit = 4 if deep else 2                               # race 13: the cheapest fitting rug sat on page 4 of the value search
        pages = min(-(-(count - shown) // shown), limit)
        if pages > limit or self.pages_read + pages > AUTO_PAGES + (2 if deep else 0) or self.left() - self.tail() - pages < 12:
            return (f"\n[Kyo] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: "
                    f"search with narrower words (the requirement's value) or page on with {param}={_dump(value)}.")
        texts = []
        for _ in range(pages):
            if not self.allow(self.pace() * 2 + 4):
                break
            faults: list[str] = []
            call = _shape({**args, param: value}, self.schemas.get(name), "args", faults)
            if faults:
                break
            reply = self.act(name, call)
            if reply is None or self.finished:
                break
            self.pages_read += 1
            seen = reply.get("observation")
            self.observe(name, seen)
            self.paged_obs.append(seen)
            shown_text = self.show(seen)
            self.read_cache[name + _dump(call)] = (self.version, self.step, shown_text)
            texts.append(f"{param}={_dump(value)}: {shown_text}")
            after = self.paging(name, seen)
            value = after[3] if after else None
            if value is None:
                break
        self.trace.append({"kind": "pages", "tool": name, "read": len(texts)})
        return ("\n[Kyo read the rest of this search itself, one step a page]\n" + "\n".join(texts)) if texts else ""

    def cheapest_here(self, observations: list[Any]) -> str:
        """Under a cheapest pick: the buyable rows of a search result (and the pages Kyo read on) in price order, since
        stores list them in no such order."""
        if self.brief.get("pick") != "cheapest" or not self.price_key:
            return ""
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ""
        return "\n[Kyo] buyable rows here, cheapest first: " + ", ".join(f"{i} ({p:g})" for p, i in sorted(rows)[:6])

    def code_marks(self, observations: list[Any]) -> str:
        """Requirements that carry a model-like code: which rows of this result mention it anywhere in their text. Stores keep
        compatibility codes in descriptions and lists, where a filter or an attribute table never shows them."""
        codes: list[tuple[int, str]] = []
        for number, need in self.needs_now():
            codes += [(number, c) for c in _code_tokens(need) if (number, c) not in codes]
        if not codes:
            return ""
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        if not found:
            return ""
        parts = []
        for number, code in codes[:4]:
            low = code.lower()
            hits = [i for i in found if low in _dump(self.rows.get(i) or {}).lower()]
            tag = f"item {number}: " if len(self.items) > 1 else ""
            if hits:
                parts.append(f"{tag}rows whose text carries {code}: " + ", ".join(hits[:6]))
            else:
                parts.append(f"{tag}no row here carries {code} in its text")
        return "\n[Kyo] " + " | ".join(parts) + ". A code like this lives in descriptions and compatibility lists: judge it from the row's or page's text, and keep paging or searching the plain product noun until a row carries it."

    def wider_hold(self) -> str:
        """Once, before a cheapest checkout with many steps left: a store's keyword search reads titles, so fitting
        listings whose titles use other words are missed by searches for a requirement's value."""
        if self.wider_held or self.brief.get("pick") != "cheapest" or self.left() < 14 + 2 * len(self.items):
            return ""
        self.wider_held = True
        lines = [f"{i} at {self.price_of(i):g}" for i in self.cart_ids() if self.price_of(i) is not None]
        return (f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest. Searches match a listing's title,"
                " so a requirement's value (often only in the attributes) finds few of the fitting listings, and rows come in no price order."
                " Before checking out, run a few more searches for each item under other names for the product (its plain one-word name,"
                " synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line"
                + (f" ({'; '.join(lines)})" if lines else "") + "; read the attributes of the cheapest such rows, swap in any that fits, then check out.")

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        """Read the attributes of listings about to be judged that no view has shown yet, keeping `keep` steps."""
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return []
        texts = []
        for ident in [i for i in ids if i in self.rows and i not in self.full and not self.sold_out(i)]:
            if self.left() <= keep or self.finished or self.facts_from.get(tool) is False or not self.allow(self.pace() + 3):
                break
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
            if faults:
                break
            reply = self.act(tool, args)
            if reply is None:
                break
            seen = reply.get("observation")
            self.observe(tool, seen)
            self.note_view(tool, seen)
            if isinstance(seen, dict) and self.ident(seen):
                self.facts_from[tool] = self.facts_from.get(tool, False) or bool(_fact_block(seen))
            texts.append(f"{ident}: " + self.show(seen)[:1500])
        if texts:
            self.trace.append({"kind": "fill_facts", "n": len(texts)})
        return texts

    def cheaper_rows(self) -> list[str]:
        """Before a cheapest checkout: buyable rows of a cart line's own kind priced below it, whose attributes were
        never read, get read (a few, cheapest first) so the last check can weigh them against the cart's choice."""
        if self.brief.get("pick") != "cheapest":
            return []
        mark = self.cart_mark() + f"#{self.version}"
        cat_key = self.category_key()
        if mark in self.cheaper_marks or not cat_key:
            return []
        self.cheaper_marks.add(mark)
        cart = self.cart_ids()
        picks: list[str] = []
        for line in cart:
            price = self.price_of(line)
            kind = str((self.rows.get(line) or {}).get(cat_key) or "").strip().lower()
            if price is None or not kind:
                continue
            pool = [i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i)
                    and str((self.rows.get(i) or {}).get(cat_key) or "").strip().lower() == kind
                    and (self.price_of(i) if self.price_of(i) is not None else float("inf")) < price]
            pool.sort(key=lambda i: self.price_of(i) or 0.0)
            picks += [i for i in pool[:2] if i not in picks]
        return self.fill_facts(picks[:CHEAPER_OPENS], self.tail() + 4)

    def sweep_cheaper(self) -> list[str]:
        """Before a checkout, once per cart (any pick rule): the light model names a few short search phrases (the plain kind words, the kind word plus one
        requirement value) for the cart's items and Kyo runs them itself (a step each; the paging rule may add a page or two), so
        that cheaper fitting listings hidden from long phrases reach the rows the check and cheaper_rows weigh."""
        search = self.roles.get("search")
        if not search or not self.cart_ids():
            return []
        mark = self.cart_mark() + f"#{self.version}"
        if mark in self.swept or not self.allow(self.pace() * 7 + 20) or self.left() - self.tail() < 12 or not self.spend_ok(0.003):
            return []
        self.swept.add(mark)
        shape = self.shapes.get(search) or {}
        props = (self.schemas.get(search) or {}).get("properties") or {}
        used = [k for _, a in self.search_log for k, v in a.items() if isinstance(v, str) and k != shape.get("next_arg")]
        param = max(set(used), key=used.count) if used else next((k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "string" and k != shape.get("next_arg")), None)
        if not param:
            return []
        done = {str(a.get(param)).strip().lower() for _, a in self.search_log if isinstance(a.get(param), str)}
        prompt = (SWEEP + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
                  + "\n\nPHRASES ALREADY SEARCHED:\n" + (", ".join(sorted(done)) or "none")
                  + "\n\nCART LINES (find cheaper fits for these items):\n" + "\n".join(f"{i}: " + self.record_of(i) for i in self.cart_ids()))
        got = self.ask_json(prompt, [LIGHT, SPARE], 600) or {}
        phrases = [str(p).strip() for p in _items(got.get("phrases")) if str(p).strip() and str(p).strip().lower() not in done][:3]
        self.trace.append({"kind": "sweep", "phrases": phrases})
        seen: list[str] = []
        for phrase in phrases:
            if not self.allow(self.pace() * 3 + 8) or self.left() - self.tail() < 8:
                break
            faults: list[str] = []
            call = _shape({param: phrase}, self.schemas.get(search), "args", faults)
            if faults:
                break
            reply = self.act(search, call)
            if reply is None or self.finished:
                break
            observation = reply.get("observation")
            self.search_log.append((search, dict(call)))
            self.observe(search, observation)
            self.more_pages(search, call, observation)
            seen.append(f"Kyo searched {param}={phrase!r} for fitting listings the earlier phrases may have hidden: {len(self.records_in(observation))} rows")
        return seen

    def observe(self, name: str, observation: Any) -> str:
        """Absorb a result of Kyo's own call and read any note in it; returns what the note changed."""
        self.absorb(observation)
        lines = self.loose_lines(observation)
        if not lines:
            return ""
        meaning = self.take_note(lines)
        if meaning:
            self.queued.append(f"(seen on {name}) {meaning}")
        return meaning

    def widen(self) -> None:
        """Open more listings side by side (one step) so the key counts are not read off two or three listings."""
        tool = self.roles.get("compare")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and "array" in _kinds(v)), None)
        # a side-by-side view without attributes shows the planner no keys: the attributes view one by one instead
        if not tool or not target or any(k != target for k in schema.get("required") or []) or self.compare_bare():
            self.widen_singly()
            return
        if self.left() <= 12:
            return
        cap = props[target].get("maxItems") if isinstance(props[target].get("maxItems"), int) else 4
        ranked = self.item_groups()
        share = max(1, cap // max(1, len(ranked)))
        picks: list[str] = []
        for ids in ranked:
            picks += [i for i in ids if i not in self.full][:share]
        picks = picks[:cap]
        if len(picks) < 2:
            return
        faults: list[str] = []
        args = _shape({target: picks}, schema, "args", faults)
        if faults:
            return
        before = len(self.full)
        reply = self.act(tool, args)
        if reply is not None:
            seen = reply.get("observation")
            self.observe(tool, seen)
            self.note_view(tool, seen)
            rows = self.records_in(seen)
            if rows:
                self.facts_from[tool] = self.facts_from.get(tool, False) or any(_fact_block(r) for _, r in rows)
            self.trace.append({"kind": "widen", "ids": picks, "opened": len(self.full) - before})
            if rows and self.facts_from.get(tool) is False:
                # the side-by-side view shows no attributes here: read them singly instead
                self.widen_singly()
                return
            if len(self.full) == before and len(picks) > 2 and self.left() > 12:
                faults = []
                args = _shape({target: picks[:2]}, schema, "args", faults)
                if not faults:
                    reply = self.act(tool, args)
                    if reply is not None:
                        self.observe(tool, reply.get("observation"))

    def widen_singly(self) -> None:
        """No side-by-side view that shows attributes: read a few more listings' attributes singly."""
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return
        # turns between the item groups, up to two each; with a cheapest pick the cheapest buyable rows first
        lanes: list[list[str]] = []
        for ids in self.item_groups():
            opened = sum(1 for i in ids if i in self.full)
            pool = [i for i in ids if i not in self.full and self.buyable(i)]
            if self.brief.get("pick") == "cheapest":
                pool.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
            lanes.append(pool[: max(0, min(2, 8 - opened))])
        queue = [lane[k] for k in range(2) for lane in lanes if k < len(lane)]
        # while steps are plentiful: 22 stay for the rest of the episode
        self.fill_facts(queue[:5], 22)

    def ready_to_ask(self) -> bool:
        if self.asked or not self.roles.get("ask") or self.msgs_free() <= 0 or not self.allow(self.pace() * 6 + 40):
            return False
        if self.left() <= 8:
            return False
        opened = len(self.full)
        if opened >= 3 * len(self.items) + 1:
            return True
        return self.step >= max(8, self.max_steps // 4)

    def message_args(self, text: str) -> dict[str, Any] | None:
        name = self.roles.get("ask")
        schema = self.schemas.get(name or "") or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        target = next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if texts else None)
        if not target:
            return None
        args: dict[str, Any] = {target: text}
        for key in schema.get("required") or []:
            if key not in args:
                spec = props.get(key) or {}
                if isinstance(spec.get("enum"), list) and spec["enum"]:
                    args[key] = spec["enum"][0]
                else:
                    return None
        faults: list[str] = []
        args = _shape(args, schema, "args", faults)
        return None if faults else args

    def interview(self) -> str:
        self.asked = True
        count = self.msgs_free()
        if count <= 0:
            return ""
        self.widen()
        table = self.key_table()
        settled = []
        for number, item in enumerate(self.items, 1):
            settled.append(f"item {number}: {item.name} - settled by the request: " + ("; ".join(item.musts) or "nothing"))
        listing = []
        last = None
        for index, (label, name, carried, values) in enumerate(table, 1):
            if label != last:
                listing.append(f"category {label!r}:")
                last = label
            listing.append(f"  {index}. {name} (on {carried}): " + " | ".join(values))
        # several ranked picks per item: a message lists up to seven of them (race 14: the top agent's lists averaged 6.7 properties and
        # revealed a need on 0.33 of its questions against 0.19-0.26 for one property a message)
        prompt = PLAN.replace("COUNT", str(min(21, max(count + 1, 14 if len(self.items) == 1 else 7 * len(self.items))))) + (ALLOC.replace("MESSAGES", str(count)).replace("ITEMS", str(len(self.items))) if len(self.items) > 1 else "") + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
        first = [str(x) for x in _items(self.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) or self.raw_hint())
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 3000) or {}
        ordered: list[tuple[int, str]] = []
        for pick in _items(plan.get("ask")):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, _count(pick.get("item") or 1) or 1), len(self.items))
            # one property per question: a label naming two is cut to the first
            label = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(pick.get("label") or ""))[0].strip().rstrip("?.")
            if not label or len(label.split()) > 5:
                continue
            if all(_words(label) != _words(other) for n, other in ordered if n == number):
                ordered.append((number, label))
        # a question names a property the listings carry: a label matching no listed key is dropped while enough picks remain
        keys = [_words(name) for _, name, _, _ in table]
        if keys:
            def listed(label: str) -> bool:
                plain = _words(label)
                if any(plain in str(x).lower() or str(x).lower() in plain for x in first):
                    return True
                stems = {w[:4] for w in plain.split() if len(w) >= 3}
                return any(stems & {w[:4] for w in k.split() if len(w) >= 3} for k in keys)
            kept = [(n, lab) for n, lab in ordered if listed(lab)]
            dropped = [(n, lab) for n, lab in ordered if not listed(lab)]
            ordered = kept + dropped[: max(0, count - len(kept))]
        # a message names seven properties, so the planner's picks run out before the messages do: each item's list goes on with the
        # most carried keys of its own category that nothing settled yet, enough for every message (an identifier, a coded value distinct
        # on every listing, is skipped; a measurement distinct on every listing is not); a short queue goes on with keys seen on one listing
        stats = getattr(self, "key_stats", {})
        for number, item in enumerate(self.items, 1):
            have = {_words(lab) for n, lab in ordered if n == number}
            mine = {w for w in _stem(getattr(item, "shelf", "") or "") | _stem(item.name) if len(w) >= 4}
            for floor in (2, 1):
                for label, name, carried, values in table:
                    seen_on, distinct, coded = stats.get(name, (carried, len(values), False))
                    if len(have) >= 21 or carried < floor or name in have or (seen_on >= 5 and distinct >= seen_on and coded):
                        continue
                    shown = [str(v).strip() for v in values]
                    if (len(set(shown)) <= 1 and carried >= 3) or any("<" in v or v.endswith("px") for v in shown) or sum(1 for v in shown if not v or v in ("0", "none")) > len(shown) // 2:
                        continue   # a key with one value on every listing, page markup, or mostly empty says nothing a shopper could want
                    if len(self.items) > 1 and not ({w for w in _stem(label) if len(w) >= 4} & mine or _words(label) in (_words(getattr(item, "shelf", "") or ""), _words(item.name))):
                        continue
                    if self.settled_label(number, name):
                        continue
                    ordered.append((number, name))
                    have.add(name)
        self.trace.append({"kind": "plan", "ask": _clip(plan.get("ask"), 120), "queue": _clip(ordered, 80)})
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        # each message lists up to seven ranked properties of one item (the shopper answers about the first listed one they care about);
        # a shopper who complains about a list gets one property a message from then on; items take turns, the planner's first item first
        asked: dict[int, list[str]] = {}
        turns: dict[int, int] = {}
        one_only = False
        lists_ok = run_shopper_lists()     # earlier episodes of this run: True = lists are read whole, False = one property a message, None = unknown yet
        if lists_ok is False:
            one_only = True
            self.trace.append({"kind": "run_shopper", "form": "singles", "seen": run_shopper_seen()})
        probed = False            # the one three-property probe an episode sends while the shopper's way with lists is unknown
        self.reworded = set()     # (item, property) asked once more in other words after a reply that did not follow the first wording
        probe_single = False      # the model's advice: one property next
        probe_next = False        # a list met confusion in a bundle: the next message is a one-property probe
        probing = False           # the message just sent was that probe
        refills = 0
        while self.msgs_free() > 0 and not self.finished and self.left() > 6:
            pending = []
            for number in dict.fromkeys(n for n, _ in ordered):
                labels = [lab for n, lab in ordered if n == number and lab not in asked.get(number, []) and not self.settled_label(number, lab)]
                if labels:
                    pending.append((turns.get(number, 0), number, labels))
            if not pending:
                # the queue ran dry with a message left (race 14 smoke: 2 of 3 messages used): the model names more properties of the item asked least
                if refills >= 2:
                    break
                refills += 1
                more = self.refill_labels(asked, turns)
                if not more:
                    break
                ordered += more
                continue
            _, number, labels = min(pending, key=lambda p: (p[0], [n for n, _ in ordered].index(p[1])))
            if one_only or probe_single or probe_next:
                labels = labels[:1]
            elif lists_ok is True:
                labels = labels[:8 if RUN_SHOPPER["complaints"] == 0 else 3]    # race 16: the complaint rate rose with the list's length (48% at two labels, 79% at six or more)
            elif lists_ok is None and sent and not probed and len(labels) >= 2:
                labels = labels[:3]    # unknown shopper: the second message lists three properties once; an answer about a later one, or a complaint, teaches the run
                probed = True
            else:
                labels = labels[:1]    # race 16: the shopper answered a list's first property only (or complained); one property a message loses nothing
            batch = self.send([(number, " / ".join(labels), self.question_list(number, labels))])
            probing, probe_next = probe_next, False
            if not batch:
                break
            sent += batch
            turns[number] = turns.get(number, 0) + 1
            asked.setdefault(number, []).extend(labels)
            heard = self.hear(batch)
            self.take_heard(heard, told)
            first = next((r for r in _items(heard.get("replies")) if isinstance(r, dict)), {})
            kind_of_reply = str(first.get("kind") or "")
            run_shopper_note(labels, kind_of_reply, str(first.get("property") or ""), bool(_items(heard.get("crowded"))))
            if not one_only:
                lists_ok = run_shopper_lists()
            if len(labels) > 1 and kind_of_reply not in ("crowded", "restated") and not _items(heard.get("crowded")):
                # the shopper answers about the first listed property it holds a view on: a reply that discloses nothing settles the
                # first label only, a disclosure settles the labels up to the one answered; the others return to the queue in their order
                spot = label_spot(labels, str(first.get("property") or "")) if (kind_of_reply == "value" or _items(heard.get("needs"))) else None
                keep = labels[: (spot + 1) if spot is not None else 1]
                asked[number] = [lab for lab in asked[number] if lab not in labels or lab in keep]
                self.trace.append({"kind": "list_settle", "item": number, "settled": keep, "back": len(labels) - len(keep)})
            if probing:
                probing = False
                if kind_of_reply in ("value", "no_preference") or _items(heard.get("needs")) or _items(heard.get("free")):
                    one_only = True          # one property drew an answer where the list drew none: this shopper takes one property a message
                    self.trace.append({"kind": "probe", "verdict": "singles"})
                else:
                    self.trace.append({"kind": "probe", "verdict": "lists"})
            elif len(labels) > 1 and kind_of_reply in ("confusion", "other") and not _items(heard.get("needs")) and len(self.items) > 1 and RUN_SHOPPER["complaints"] > 0:
                probe_next = True            # a bundle: confusion may mean no listed property is a need, so one cheap probe decides (race-13 persona arm: lists alone collapsed 25 -> 6)
            elif len(labels) > 1 and kind_of_reply in ("confusion", "other") and not _items(heard.get("needs")) and len(self.items) == 1 and RUN_SHOPPER["complaints"] > 0:
                # a one-item request: a shopper who answers lists meets a list with a value or "no preference" (race 14: confusion 0% there), so
                # confusion means this shopper does not take lists; one property a message from here (in a bundle confusion also means no listed
                # property is a need, so lists go on there)
                one_only = True                            # race 15: the labels already listed stay asked; unasked properties come first (46 of 96 re-asks were waste)
                self.trace.append({"kind": "probe", "verdict": "singles"})
            elif len(labels) == 1 and len(self.items) == 1 and kind_of_reply == "confusion" and not _items(heard.get("needs")) and not self.reworded and self.msgs_free() > 0:
                # race 16: in a one-item request confusion never met a question about a non-facet (no preference did), so it is a misread: the same
                # property asked once more in other words drew the answer 78 times in 100; with several items confusion mostly means the property
                # is not that item's facet (64%), and the message goes to another property instead
                self.reworded.add((number, labels[0]))
                asked[number] = [lab for lab in asked[number] if lab != labels[0]]
                self.trace.append({"kind": "reword", "item": number, "property": labels[0]})
            advice = heard.get("next") if isinstance(heard.get("next"), dict) else {}
            probe_single = str(advice.get("form") or "") == "single" and not one_only and RUN_SHOPPER["complaints"] > 0      # the model's advice for the next message counts once this run's shopper has complained about a list
            wanted = [_words(str(x)) for x in _items(advice.get("properties")) if str(x).strip()]
            if wanted:   # the properties the model names next go first in this item's queue
                mine = [(n, lab) for n, lab in ordered if n == number and lab not in asked.get(number, [])]
                front = [p for p in mine if any(_words(p[1]) == w or _names(_stem(_words(p[1])), _stem(w)) for w in wanted)]
                ordered = [p for p in ordered if p not in mine] + front + [p for p in mine if p not in front]
            if len(labels) > 1 and (_items(heard.get("crowded")) or kind_of_reply == "crowded"):
                # the list was turned down unanswered: its properties are asked again one a message
                one_only = True
                asked[number] = [lab for lab in asked[number] if lab not in labels]
            elif len(labels) > 1 and str(first.get("kind") or "") == "restated":
                # the shopper repeated a requirement the request already fixed instead of answering: that property is settled for good
                # and the other listed properties go back in the queue (race 14 probe: the reply type decides the next message, not a rule)
                named = _words(str(first.get("property") or ""))
                item = self.items[number - 1]
                hit = [lab for lab in labels if named and (_words(lab) == named or _names(_stem(_words(lab)), _stem(named)) or _names(_stem(named), _stem(_words(lab))))]
                item.restated = getattr(item, "restated", []) + (hit or ([labels[0]] if not named else []))
                asked[number] = [lab for lab in asked[number] if lab not in labels or lab in item.restated]
                self.trace.append({"kind": "restated", "item": number, "property": named, "settled": hit})
        if told:
            self.version += 1
        lines = [f"Q: {q}\nA: {a}" for _, q, a in sent]
        summary = "Kyo's interviewer asked the shopper (each question used one step):\n" + "\n".join(lines)
        summary += "\nNew firm requirements from these answers: " + ("; ".join(told) if told else "none")
        summary += ". Apply them to every pick (search again with the new values where needed)."
        if told:
            summary += self.recheck_note()
        pointed = [need for _, need in self.refs if any(need in t for t in told)]
        if pointed:
            summary += (" These take their value from another listing: " + " | ".join(pointed) + ". Find that listing, open its page in full,"
                        " and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.")
        if self.queued:
            summary += "\n[notes seen meanwhile] " + " | ".join(self.queued)
            self.queued = []
        return summary

    def page_deeper(self, number: int) -> bool:
        """Four pages instead of two for the item's value search: a cheapest pick, and no seen row of the item is both cheaper than
        its cart line and a literal fit."""
        if self.brief.get("pick") != "cheapest":
            return False
        held = [self.cost_of(i) for i in self.cart_ids() if self.item_of(i) == number and self.cost_of(i) is not None]
        if not held:
            return True
        ceiling = min(held)
        for ident in self.rows:
            cost = self.cost_of(ident)
            if self.item_of(ident) == number and cost is not None and cost < ceiling and self.literal_fit(ident, number) is True:
                return False
        return True

    def recheck_note(self) -> str:
        """After new requirements: the opened listings of each item sorted by how their records read against every plain
        requirement now in force (stating all of them / stating another value), so a fit already read is chosen before more searching."""
        try:
            budget = self.budget if isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool) else None
            pools = self.kind_pools(budget)
            parts = []
            search = self.roles.get("search")
            shape = (self.shapes.get(search) or {}) if search else {}
            props = ((self.schemas.get(search) or {}).get("properties") or {}) if search else {}
            used = [k for _, a in self.search_log for k, v in a.items() if isinstance(v, str) and k != shape.get("next_arg")]
            param = (max(set(used), key=used.count) if used else next((k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "string" and k != shape.get("next_arg")), None)) if search else None
            searches = 0
            for number, item in enumerate(self.items, 1):
                opened = [i for i in pools.get(number, []) if i in self.full]
                states = [i for i in opened if self.literal_fit(i, number) is True]
                clashes = [i for i in opened if self.literal_fit(i, number) is False]
                if states or clashes:
                    part = f"item {number}: " + (f"already-read listings stating every known requirement: {', '.join(states[:6])}" if states else "no already-read listing states every known requirement")
                    if clashes:
                        part += f"; stating another value for one: {', '.join(clashes[:6])}"
                    parts.append(part)
                if states or not param or searches >= 2 or not self.allow(self.pace() * 3 + 20) or self.left() - self.tail() < 8:
                    continue
                values = [v for _, v in self.literal_needs(number) if v in {t.split(":", 1)[1].strip() for t in item.told if ":" in t}]
                if not values:
                    continue
                faults: list[str] = []
                args = _shape({param: " ".join(item.name.split()[:2] + values[-1].split())[:120]}, self.schemas.get(search), "args", faults)
                if faults:
                    continue
                reply = self.act(search, args)
                searches += 1
                if reply is None or self.finished:
                    break
                self.observe(search, reply.get("observation"))
                self.search_log.append((search, dict(args)))
                self.more_pages(search, args, reply.get("observation"), deep=self.page_deeper(number))   # a cheapest pick: the cheapest fit may sit pages deep (race 13: page 4)
                rows = [i for i, _ in self.records_in(reply.get("observation"))][:8]
                self.trace.append({"kind": "value_search", "item": number, "args": args, "rows": len(rows)})
                parts.append(f"item {number}: searched the new value and saw {len(rows)} rows" + (f" ({', '.join(rows[:5])}): open the promising ones before choosing" if rows else ""))
            return (" [Kyo, from the records read so far] " + " | ".join(parts) + ". A listing that states every requirement is a fit to cart; search further only for an item with none.") if parts else ""
        except Exception as exc:
            self.trace.append({"kind": "recheck_error", "error": f"{type(exc).__name__}: {exc}"[:160]})
            return ""

    def question(self, number: int, label: str, samples: list[str] | None = None) -> str:
        text = f"What {label} would you like for the {self.items[number - 1].name}"
        return text + (f", for example {samples[0]} or {samples[1]}?" if samples else "?")

    def refill_labels(self, asked: dict[int, list[str]], turns: dict[int, int]) -> list[tuple[int, str]]:
        """Up to seven more properties of the item asked about least, named by the model from the request, the keys the listings carry
        and what was asked or settled already; a label matching a settled or asked property is dropped."""
        try:
            number = min(range(1, len(self.items) + 1), key=lambda n: (turns.get(n, 0), n))
            item = self.items[number - 1]
            keys = [name for _, name, _, _ in self.key_table()]
            prompt = (REFILL + "\n\nREQUEST:\n" + self.request + f"\n\nPRODUCT: {item.name}" + "\nSETTLED BY THE REQUEST OR ANSWERED: " + ("; ".join(item.musts + item.told + item.free) or "nothing")
                      + "\nASKED ALREADY: " + (", ".join(asked.get(number, [])) or "nothing") + "\nATTRIBUTE KEYS LISTINGS CARRY: " + (", ".join(keys[:40]) or self.raw_hint()[:1800]))
            got = self.ask_json(prompt, [LIGHT, HEAVY], 1200, reasoning={"effort": "none"}) or {}   # a list of names needs no deliberation
            out: list[tuple[int, str]] = []
            for raw in _items(got.get("labels")):
                label = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(raw))[0].strip().rstrip("?.").lower()
                if not label or len(label.split()) > 5 or _words(label) in {_words(a) for a in asked.get(number, [])} or self.settled_label(number, label):
                    continue
                if all(_words(label) != _words(lab) for _, lab in out):
                    out.append((number, label))
            self.trace.append({"kind": "refill", "item": number, "labels": [lab for _, lab in out][:7]})
            return out[:7]
        except Exception as exc:
            self.trace.append({"kind": "refill_error", "error": f"{type(exc).__name__}: {exc}"[:160]})
            return []

    def question_list(self, number: int, labels: list[str]) -> str:
        """One message naming the item and up to seven of its properties, the likeliest first."""
        if len(labels) <= 1:
            if labels and (number, labels[0]) in getattr(self, "reworded", set()):
                return self.question_again(number, labels[0])
            return self.question(number, labels[0]) if labels else ""
        return f"For the {self.items[number - 1].name}, what {', '.join(labels[:-1])} or {labels[-1]} would you like?"

    def question_again(self, number: int, label: str) -> str:
        # the same property in other words, with two values the listings state for it, after a reply that did not follow the first wording
        values: list[str] = []
        try:
            for _, name, _, shown in self.key_table():
                if _words(name) == _words(label) or _names(_stem(_words(name)), _stem(_words(label))):
                    values = [v for v in shown if v and v not in values][:2]
                    break
        except Exception:
            values = []
        item = self.items[number - 1].name
        if len(values) >= 2:
            return f"Sorry, I meant the {item}'s {label} as its listings state it, such as {values[0]} or {values[1]}: do you hold out for one of those, or is any fine?"
        return f"Sorry, I meant the {label} the listings state for the {item}: do you hold out for a particular one, or is any fine?"

    def settled_label(self, number: int, label: str) -> bool:
        """A property the shopper already answered about, declined to care about, or the request settled."""
        item = self.items[number - 1]
        plain = _words(label)
        known = ({_words(n.split(":", 1)[0]) for n in item.musts if ":" in n and not _vague(n.split(":", 1)[1])}
                 | {_words(n.split(":", 1)[0]) for n in item.told if ":" in n} | {_words(f) for f in item.free} | {_words(f) for f in getattr(item, "restated", [])})
        if plain in known or any(_names(_stem(plain), _stem(k)) for k in known if k):
            return True
        # the key abbreviates a settled property's words ("os" for "operating system")
        if any(len(k.split()) >= 2 and "".join(w[0] for w in k.split()) == plain for k in known if k):
            return True
        # a settled requirement whose property matches no key by name: the key whose values show the stated value is that property's key
        # (the request says "android", the listings carry it under another key name), so a question on it would be answered with nothing new
        values = getattr(self, "key_values", {}).get(plain) or set()
        if values:
            names = list(getattr(self, "key_stats", {}))
            for need in item.musts + item.told:
                if ":" not in need:
                    continue
                prop, value = (x.strip() for x in need.split(":", 1))
                if any(_names(_stem(_words(prop)), _stem(n)) for n in names):
                    continue
                want = _value_words(value)
                if want and len(want) <= 3 and any(want <= _value_words(v) for v in values):
                    return True
        return False

    def samples(self, number: int, label: str) -> list[str]:
        """Two values the listings show for this property: they say which property a question means."""
        want = _words(label)
        settled = " ".join(self.items[number - 1].musts).lower()
        for _, name, _, values in self.key_table():
            if name == want:
                return [v for v in values if v and v not in settled and len(v) <= 30][:2]
        return []

    def send(self, questions: list[tuple[int, str, str]], floor: int = 6) -> list[tuple[int, str, str]]:
        tool = self.roles["ask"]
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
            answer = reply.get("observation")
            if isinstance(answer, dict) and not FORMAT_MEMORY.get("reply_key") and sum(1 for v in answer.values() if isinstance(v, str) and len(v.split()) >= 2) + sum(1 for v in answer.values() if isinstance(v, list) and v and all(isinstance(x, str) for x in v)) >= 2:
                self.learn_reply_key(answer, text)
            words = self.spoken(answer)
            sent.append((number, text, words))
            self.done_log.append(f"step {self.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {self.cart_mark()})")
            self.done_steps.append(self.step)
            self.trace.append({"kind": "ask", "q": text, "a": words[:300]})
            if isinstance(answer, dict) and answer.get("error"):
                break
        return sent

    def take_heard(self, heard: dict[str, Any], told: list[str]) -> None:
        """Add what the answers stated to the ledger; `told` collects the new requirements for the summary."""
        many = len(self.items) > 1
        for entry in _items(heard.get("needs")):
            if not isinstance(entry, dict) or not str(entry.get("need") or "").strip():
                continue
            number = min(max(1, _count(entry.get("item")) or 1), len(self.items))
            need = str(entry["need"]).strip()
            if need not in self.items[number - 1].told:
                self.items[number - 1].told.append(need)
                told.append(f"item {number}: {need}" if many else need)
            if entry.get("points") is True and (number, need) not in self.refs:
                self.refs.append((number, need))
        for entry in _items(heard.get("drop")):
            if isinstance(entry, dict) and str(entry.get("property") or "").strip():
                item = self.items[min(max(1, _count(entry.get("item")) or 1), len(self.items)) - 1]
                gone = _words(str(entry["property"]).split(":")[0])
                hit = [m for m in item.musts + item.told if m not in item.dropped and _words(m.split(":")[0]) == gone]
                item.dropped.extend(hit)
                told.extend(f"dropped {m}" for m in hit)
        for entry in _items(heard.get("free")):
            if isinstance(entry, dict) and entry.get("property"):
                number = min(max(1, _count(entry.get("item")) or 1), len(self.items))
                self.items[number - 1].free.append(str(entry["property"]))
        if isinstance(heard.get("budget"), (int, float)) and not isinstance(heard.get("budget"), bool) and heard["budget"] > 0:
            self.budget = heard["budget"]

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...] = (HEAVY, LIGHT)) -> dict[str, Any]:
        prompt = HEAR + "\n\nITEMS:\n" + "\n".join(f"item {n}: {it.name}" for n, it in enumerate(self.items, 1))
        prompt += "\n\nQUESTIONS AND ANSWERS:\n" + "\n".join(f"{i}. (about item {n}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1))
        return self.ask_json(prompt, list(models), 3000, reasoning={"effort": HEAR_EFFORT}) or {}

    def gate(self, stage: str) -> tuple[bool, str]:
        """Right before a checkout ("order") or an ending without a purchase ("end"): the steps the request asks for at
        that point, done by the model's choice of store calls (a message to the shopper, a page, any store tool) until
        it finds none missing for the cart as it is now. Returns whether to go on, and what to tell the agent."""
        steps = [p for p in self.process if p["when"] == stage]
        if not steps or self.finished or not self.gate_due(stage):
            return True, ""
        ending = {self.roles.get("buy"), self.roles.get("refuse")} - {None}
        tools = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_dump(t['function'].get('parameters') or {})}"
                 for t in self.tools if t["function"]["name"] not in ending]
        said: list[str] = []
        nudged = False
        for _ in range(GATE_ROUNDS):
            if not self.allow(self.pace() + 12):
                break
            prompt = (GATE.replace("STAGE", "place the order" if stage == "order" else "end without a purchase")
                      + "\n\nREQUEST:\n" + self.request + "\n\nSTEPS DUE NOW:\n" + "\n".join(f"- {p['step']}" for p in steps)
                      + "\n\nLEDGER:\n" + self.ledger() + "\n\nCART NOW: " + self.cart_text()
                      + "\n\nRECORD (messages and step calls so far):\n" + ("\n".join(f"{i}. {line}" for i, line in enumerate(self.done_log, 1)) or "(nothing yet)")
                      + ("\n\nSTORE NOTES SEEN:\n" + "\n".join(self.notes[-4:]) if self.notes else "")
                      + f"\n\nMESSAGES LEFT: {max(0, self.msg_limit - self.msg_used)}   STEPS LEFT: {self.left()} (the {'order' if stage == 'order' else 'ending'} itself takes one)"
                      + "\n\nTOOLS:\n" + "\n".join(tools))
            # the light model decides; after a nudge, the heavy one
            got = self.ask_json(prompt, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 2000)
            if got is None:
                break
            told: list[str] = []
            hold = ""
            if said:
                # what a reply to the gate's own call asks for (earlier answers are the interviewer's to read)
                self.take_heard({"needs": got.get("needs")}, told)
                hold = str(got.get("hold") or "").strip()
            call = got.get("call") if isinstance(got.get("call"), dict) else None
            self.trace.append({"kind": "gate", "stage": stage, "call": _clip(call, 200), "steps": _clip(got.get("steps"), 120), "hold": hold[:200], "told": told})
            if told or hold:
                self.version += 1
                return False, ("Before going on, Kyo did the steps the request asks for:\n" + "\n".join(said) + "\n" if said else "") + (
                    "The shopper's reply changes the task: " + "; ".join(told + ([hold] if hold else [])) + ". Fix the cart for it, then check out again.")
            name = str((call or {}).get("tool") or "")
            # a step called done must point at the record line that did it; at the order, a step that takes a
            # message is about the final cart, so its line must come after the cart last changed
            shown = [e for e in _items(got.get("steps")) if isinstance(e, dict)]
            talk = stage == "order" and any(p["messages"] > 0 for p in steps)
            lines = range(1, len(self.done_log) + 1)
            proof = {i for i in lines if not talk or (i <= len(self.done_steps) and self.done_steps[i - 1] > self.cart_at)}
            unproven = [e for e in shown if e.get("done") is not True or not any((_count(n) or 0) in proof for n in _items(e.get("lines")))]
            if not call and unproven and not said and not nudged and self.roles.get("ask") and self.msg_used < self.msg_limit and self.left() >= 2:
                nudged = True
                self.trace.append({"kind": "gate_unproven", "steps": _clip(unproven, 120)})
                self.done_log.append(f"(Kyo: no line made since the cart last changed at step {self.cart_at} shows " + "; ".join(str(e.get("step") or "") for e in unproven)[:200]
                                     + f" done, and {self.msg_limit - self.msg_used} message(s) and {self.left()} steps are left: give the call that does it)")
                self.done_steps.append(-1)
                continue
            if not call or name not in self.schemas or name in ending:
                if not call:
                    self.gate_ok.add((stage, self.cart_mark() + f"#{self.version}"))
                break
            if self.left() < 2 or (name == self.roles.get("ask") and self.msg_used >= self.msg_limit):
                break
            faults: list[str] = []
            args = _shape(call.get("args") if isinstance(call.get("args"), dict) else {}, self.schemas[name], "args", faults)
            if faults:
                break
            if name == self.roles.get("ask"):
                text = next((v for v in args.values() if isinstance(v, str)), "")
                batch = self.send([(1, "", text)], 1)
                if not batch:
                    break
                self.gate_msgs += 1
                said.append(f"Q: {text}\nA: {batch[0][2]}")
                continue
            reply = self.act(name, args)
            if reply is None:
                break
            seen = reply.get("observation")
            self.absorb(seen)
            self.done_log.append(f"step {self.step}: {name} {_dump(args)} -> {self.show(seen)[:300]} (cart then: {self.cart_mark()})")
            self.done_steps.append(self.step)
            said.append(f"{name} {_dump(args)} -> {self.show(seen)[:400]}")
        if stage == "order" and any(line.startswith("Q: ") for line in said) and self.left() >= len(self.cart_ids()) + 2 and self.allow(self.pace() * (len(self.cart_ids()) + 1) + 20):
            self.reread_offers()                              # the lines are read again after the shopper's reply (race 14: orders not re-read after the last answer passed 0.17 vs 0.62)
        return True, ("Before going on, Kyo did the steps the request asks for:\n" + "\n".join(said)) if said else ""

    def spoken(self, observation: Any) -> str:
        """The shopper's words in a reply: the field the structure learner named, else the payload's text (strings that stand
        alone, before any list of notice lines), else the whole reply."""
        if isinstance(observation, str):
            return observation
        key = FORMAT_MEMORY.get("reply_key")
        if isinstance(observation, dict) and key and isinstance(observation.get(key), str):
            return observation[key]
        found: list[str] = []
        listed: list[str] = []

        def walk(node: Any, depth: int, in_list: bool) -> None:
            if depth > 4:
                return
            if isinstance(node, str) and len(node.split()) >= 2 and _decode_text(node) is None:
                (listed if in_list else found).append(node)
            elif isinstance(node, dict):
                for value in node.values():
                    walk(value, depth + 1, False)
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1, True)

        walk(observation, 0, False)
        if found:
            return " ".join(found)[:800]
        if listed:
            return " ".join(listed)[:800]
        return _dump(observation)[:400]

    # ---------- notes inside tool results ----------
    def loose_lines(self, observation: Any) -> list[str]:
        """The shopper's or store's loose sentences in a result (a budget change, a dropped requirement, a sell-out), wherever the
        envelope puts them: not a listing record, not the text of one, not a value inside a record."""
        lines: list[str] = []
        def note_like(text: str) -> bool:
            return len(text.split()) >= 5 and _decode_text(text) is None
        if isinstance(observation, str):
            lines = [observation] if note_like(observation) else []
        elif isinstance(observation, dict) and not self.ident(observation):
            for key, value in observation.items():
                if isinstance(value, str) and note_like(value):
                    lines.append(value)
                elif isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                    lines.extend(v for v in value if note_like(v))
                elif isinstance(value, dict) and not self.ident(value, False, 1):
                    for inner in value.values():
                        if isinstance(inner, str) and len(inner.split()) >= 6 and note_like(inner):
                            lines.append(inner)
                        elif isinstance(inner, list) and inner and all(isinstance(v, str) for v in inner):
                            lines.extend(v for v in inner if note_like(v))
        elif isinstance(observation, list) and observation and all(isinstance(v, str) for v in observation):
            lines = [v for v in observation if note_like(v)]
        return [line for line in lines if line not in self.heard]

    def take_note(self, lines: list[str]) -> str:
        self.heard.update(lines)
        self.recent_notes = [str(l)[:160] for l in lines][-2:]   # quoted in the finish text when the model's reading of them is unavailable
        # a banner whose only change is a number ("Showing 5 of 37 ...") is the same note again: no model call for it
        fresh = [l for l in lines if re.sub(r"\d+", "#", l) not in self.note_shapes]
        self.note_shapes.update(re.sub(r"\d+", "#", l) for l in lines)
        if not fresh or self.notes_read >= 12 or not self.spend_ok(0.002):
            return ""
        lines = fresh
        self.notes_read += 1
        prompt = NOTE + "\n\nITEMS AND REQUIREMENTS NOW:\n" + self.ledger() + "\n\nTHE NOTE:\n" + "\n".join(lines)
        got = self.ask_json(prompt, [LIGHT, SPARE], 1200) or {}
        gone = [str(g) for g in _items(got.get("gone")) if isinstance(g, (str, int))]
        self.unavailable.update(g for g in gone if g in self.rows)
        self.note_sold = bool(gone) or got.get("sold_out") is True
        if self.note_sold and len(lines) == 1:
            # only a note that is nothing but the sell-out is remembered: the same words later mean the same thing
            self.sold_lines.update(lines)
        if got.get("kind") not in (None, "change"):
            return ""
        changed = []
        budget = got.get("budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 and budget != self.budget:
            self.budget = budget
            changed.append(f"budget now {budget}")
        added: set[str] = set()
        drops = [e for e in _items(got.get("drop")) if isinstance(e, dict) and str(e.get("need") or "").strip()]
        adds = [e for e in _items(got.get("add")) if isinstance(e, dict) and str(e.get("need") or "").strip()]
        for entry in drops:
            item = self.items[min(max(1, _count(entry.get("item")) or 1), len(self.items)) - 1]
            target = str(entry["need"]).strip()
            live = [m for m in item.musts + item.told if m not in item.dropped]
            hit = [m for m in live if m == target]
            if not hit:
                hit = [m for m in live if _words(m.split(":")[0]) == _words(target.split(":")[0])][:1]
            item.dropped.extend(hit or [target])
            changed.append(f"dropped {target}")
        for entry in adds:
            item = self.items[min(max(1, _count(entry.get("item")) or 1), len(self.items)) - 1]
            need = str(entry["need"]).strip()
            if need in item.dropped:
                item.dropped.remove(need)
            if need not in item.musts + item.told:
                item.told.append(need)
            added.add(need)
            changed.append(f"added {need}")
        if gone:
            changed.append("unavailable: " + ", ".join(gone))
        self.note_price = got.get("price_moved") is True
        if self.note_price:
            changed.append("a price moved")
        summary = str(got.get("summary") or "").strip()
        if changed or summary:
            self.version += 1
            self.recent = []
            self.notes.append(summary or "; ".join(changed))
            for entry in changed:
                if entry.startswith(("budget", "dropped", "added", "a price")):
                    self.pending_changes.append(entry)
        return (summary + (" (" + "; ".join(changed) + ")" if changed else "")).strip()

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
                if ident and _fact_block(node) and ident not in ids:
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
            return ""
        many = len(self.items) > 1
        lines = []
        for ident in ids[:12]:
            block = _fact_block(self.full.get(ident) or {}) or {}
            parts = []
            for number, need in needs[:8]:
                prop = need.split(":", 1)[0].strip()
                hits = [f"{k}={str(v)[:50]}" for k, v in block.items() if _names(_stem(prop), _stem(k))]
                tag = f"(item {number}) " if many else ""
                if hits:
                    parts.append(f"{tag}{prop} -> " + ", ".join(hits[:3]))
            if parts:
                lines.append(f"{ident}: " + "; ".join(parts))
        if not lines:
            return ""
        return "[evidence: keys whose names contain a requirement's property; one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property] " + " | ".join(lines)

    # ---------- checks ----------
    def cart_text(self) -> str:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            return _dump(_clip(self.cart_view, 300))[:3000]
        # a cart view older than the last cart change would show lines that are gone
        return "lines by Kyo's record of the cart calls (listing: units): " + _dump(self.cart_lines)

    def cart_ids(self) -> list[str]:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            text = _dump(self.cart_view)
            shown = [i for i in self.rows if i in text]
            # a view that names no listing id (titles or line numbers only) is read through the record of the cart calls
            return shown if (shown or not self.cart_lines) else list(self.cart_lines)
        return list(self.cart_lines)

    def record_of(self, ident: str) -> str:
        record = self.full.get(ident) or self.rows.get(ident)
        return _dump(_clip(record, 400))[:3000] if record else "(no record seen)"

    def others(self, skip: list[str], limit: int = 30) -> str:
        lines = []
        order = list(reversed(self.seen_order))
        if self.brief.get("pick") == "cheapest":
            # listings whose attributes were read first, each group cheapest first
            order.sort(key=lambda i: (i not in self.full, self.price_of(i) if self.price_of(i) is not None else float("inf")))
        for ident in order:
            if ident in skip or self.sold_out(ident):
                continue
            record = self.full.get(ident) or self.rows.get(ident)
            lines.append(_dump(_clip(record, 160))[:900])
            if len(lines) >= limit:
                break
        return "\n".join(lines)

    def check_key(self, ids: list[str]) -> str:
        return "|".join(sorted(ids)) + f"#{self.version}"

    def wish_steer(self, got: dict[str, Any] | None, ids: list[str], key: str, lead: bool, limit: float) -> str | None:
        """After a passing review: the listing the review names as matching the shopper's unwritten wish replaces the cart line (one time an
        episode); a review marked doubtful earns one extra candidate read and a second review. None when there is nothing to steer."""
        if not got or got.get("ok") is False:
            return None
        wished = []
        for raw in _items(got.get("prefer")):
            ident = str(raw)
            if ident in self.rows and ident not in ids and not self.sold_out(ident) and ident not in wished and self.literal_fit(ident, self.item_of(ident) or 1) is not False:
                wished.append(ident)
        if wished and not self.story_swapped:
            self.story_swapped = True
            self.trace.append({"kind": "wish_steer", "cart": ids, "to": wished[0]})
            return (f"The review passes the cart on the written requirements, yet {wished[0]} also satisfies all of them and answers the shopper's "
                    f"unwritten wish better than the current line. Swap that line for {wished[0]} at the same quantity unless its record shows a failing, then check out.")
        if got.get("doubtful") is True and not self.doubt_opened and self.left() > 4 and self.room() > 110:
            candidates = wished + [str(b) for b in _items(got.get("better")) if str(b) in self.rows]
            fresh = [i for i in dict.fromkeys(candidates) if i not in ids and i not in getattr(self, "full", {}) and not self.sold_out(i)]
            if fresh:
                self.doubt_opened = True
                self.trace.append({"kind": "doubt_read", "cart": ids, "read": fresh[0]})
                if self.open_rows([fresh[0]]):
                    self.checked.discard(key)
                    return self.check_cart(ids if lead else None, limit)
        return None

    def wish_choice(self, ids: list[str]) -> str:
        """The cart line set against the opened listings of the same shelf that also pass the written requirements, as a direct question
        which one the shopper's circumstances point to. A different answer swaps the line, one time an episode."""
        label_key = self.category_key()
        opened = [i for i in reversed(list(self.full)) if i not in ids and not self.sold_out(i)]
        for line in ids:
            shelf = str((self.rows.get(line) or {}).get(label_key) or "") if label_key else ""
            number = next((n for n, it in enumerate(self.items, 1) if shelf and (it.name.lower() in shelf.lower() or shelf.lower() in it.name.lower())), 1)
            pool = [i for i in opened if (not shelf or str((self.rows.get(i) or {}).get(label_key) or "") == shelf) and self.literal_fit(i, number) is True][:3]
            if not pool or self.cost + 0.002 > SPEND_HARD or self.room() < 80:
                continue
            prompt = (WISHPICK.replace("ITEM", self.items[number - 1].name) + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
                      + "\n\nCANDIDATES (each passes the written requirements):\n" + "\n".join(f"{i}: " + self.record_of(i) for i in [line] + pool))
            got = self.ask_json(prompt, [LIGHT, SPARE], 1200) or {}
            choice = str(got.get("pick") or "").strip()
            self.trace.append({"kind": "wish_choice", "line": line, "pool": pool, "pick": choice, "property": str(got.get("property") or "")[:60]})
            if choice in pool:
                self.story_swapped = True
                return (f"Set against the other opened listings that pass the written requirements, {choice} answers the shopper's circumstances"
                        f" (an unwritten wish, probably about {str(got.get('property') or 'one property')}) better than {line}. Swap the line for {choice}"
                        " at the same quantity, then check out.")
            if choice == line:
                self.story_swapped = True
        return ""

    def check_cart(self, ids: list[str] | None = None, limit: float = SPEND_SOFT) -> str:
        lead = ids is not None
        ids = ids if lead else self.cart_ids()
        key = self.check_key(ids)
        if key in self.checked or self.cost > limit or not ids:
            return ""
        self.checked.add(key)
        prompt = CHECK + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        if lead:
            quantity = "; ".join(f"item {n}: quantity {it.qty}" for n, it in enumerate(self.items, 1))
            prompt += "\n\nCART (planned, judge these listings as if they were carted at the requested quantities: " + quantity + "):\n" + ", ".join(ids)
        else:
            prompt += "\n\nCART:\n" + self.cart_text()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + "\n".join(f"{i}: " + ("SOLD OUT (the store says it can no longer be bought). " if self.sold_out(i) else "") + self.record_of(i) for i in ids)
        if self.brief.get("pick") in ("cheapest", "most", "least"):
            prompt += "\n\nOTHER LISTINGS SEEN (newest first):\n" + self.others(ids, 15)
        got = self.ask_json(prompt, [HEAVY, LIGHT], 2500)
        if isinstance(got, dict):
            # a citation the record text does not hold, or none at all, leaves the requirement unshown and the cart not right
            # (race 14: orders on a revealed need whose record never stated it)
            unsupported: list[str] = []
            support = got.get("support") if isinstance(got.get("support"), dict) else {}
            for pid in ids:
                text = self.record_of(pid).lower()
                for row in _items(support.get(pid)):
                    if not isinstance(row, dict):
                        continue
                    requirement = str(row.get("requirement") or "")
                    if _vague(requirement.split(":", 1)[1] if ":" in requirement else requirement):
                        continue                                  # a wish in words settles nothing and needs no record text
                    key = str(row.get("key") or "").strip().lower(); value = str(row.get("value") or "").strip().lower()
                    parts = re.findall(r"[a-z0-9.]+", value)
                    if re.search(r"\b(?:not|never|except|without)\b|other than|anything but", requirement.lower()) and key not in ("", "null", "none"):
                        if all(w in text for w in re.findall(r"[a-z0-9]+", key)[:3]):
                            continue                              # an exclusion: the record's own value under that key is the evidence
                    if not (value and key not in ("", "null", "none") and (value in text or (parts and all(w in text for w in parts[:5])))):
                        unsupported.append(f"{pid}: {str(row.get('requirement') or '?')[:60]}")
            if unsupported:
                got["ok"] = False
                got["problems"] = [str(x) for x in _items(got.get("problems"))][:3] + ["no record text decides: " + "; ".join(unsupported[:4])]
                self.trace.append({"kind": "unsupported", "rows": unsupported[:6]})
        self.trace.append({"kind": "check", "cart": ids, "got": _clip(got, 400)})
        self.verdicts[key] = bool(got) and got.get("ok") is not False
        if got and got.get("ok") is not False and not self.story_swapped and self.brief.get("pick") not in ("cheapest", "most", "least"):
            swapped = self.wish_choice(ids)
            if swapped:
                return swapped
        steered = self.wish_steer(got, ids, key, lead, limit)
        if steered is not None:
            return steered
        if not got or got.get("ok") is not False:
            return ""
        self.leads = [i for i in self.leads if i not in ids]
        problems = [str(p) for p in _items(got.get("problems"))][:5]
        better = [str(b) for b in _items(got.get("better")) if str(b) in self.rows and not self.sold_out(str(b))][:4]
        text = "A careful review says this cart is not right yet: " + "; ".join(problems or ["see the requirements"])
        if better:
            text += ". Listings that look better: " + ", ".join(better)
        return text + ". Fix the cart (or show the review is wrong by re-reading the records), then check out."

    def early_trail(self, summary: str) -> str:
        # the pages a requirement points at are walked right after the interview, while steps remain (race 16: the first page of a chain
        # opened at step 23 of 35 left no room for its second; the agents that passed had both open by step 24)
        try:
            live = {need for _, need in self.needs_now()}
            if not any(need in live and need not in self.ref_values for _, need in self.refs) or self.left() <= 10 or self.finished or not self.allow(self.pace() * 3 + 20):
                return summary
            walked, _ = self.trail(early=True)
            return (summary + "\n" + walked) if walked else summary
        except Exception as exc:
            self.trace.append({"kind": "early_trail_error", "error": f"{type(exc).__name__}: {exc}"[:160]})
            return summary

    def trail(self, early: bool = False) -> tuple[str, bool]:
        """Before a checkout: each requirement whose value is on another listing has every page of its chain opened.
        Kyo opens the links it can name itself; a link nobody has found yet holds the checkout back (twice at most).
        Returns what it saw and whether to hold the checkout."""
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        # a chain's links are pages: the page view (the offer view when it is apart from the attributes view)
        tool = self.roles.get("offer") or self.roles.get("open")
        target = self.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trails < 4 and self.spend_ok(0.004) and self.allow(self.pace() * 3 + 15):
            self.trails += 1
            records = []
            for ident in reversed(self.seen_order):
                # an opened page is the page view's record (with the attributes when they were read too)
                record = (self.full.get(ident) or self.rows.get(ident)) if ident in self.opened else None
                if record:
                    records.append("OPENED " + _dump(_clip(record, 300))[:1400])
                else:
                    records.append("ROW " + _dump(_clip(self.rows.get(ident), 80))[:300])
                if len(records) >= 60:
                    break
            prompt = TRAIL + "\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n" + "\n".join(pending)
            prompt += "\n\nRECORDS (newest first):\n" + "\n".join(records)
            got = self.ask_json(prompt, [LIGHT, HEAVY], 1800) or {}
            self.trace.append({"kind": "trail", "pending": pending, "got": _clip(got, 150)})
            reach: list[str] = []
            lost = []
            for chain in _items(got.get("chains")):
                if not isinstance(chain, dict):
                    continue
                need = next((p for p in pending if p == str(chain.get("need") or "").strip()), None) or (pending[0] if len(pending) == 1 else None)
                if not need:
                    continue
                links = [l for l in _items(chain.get("links")) if isinstance(l, dict)]
                ids = [str(l.get("id") or "").strip() for l in links]
                if chain.get("done") is True and links and all(i in self.opened for i in ids) and str(chain.get("value") or "").strip():
                    self.ref_values[need] = str(chain["value"]).strip()
                    self.version += 1
                    continue
                closed = [i for i in ids if i in self.rows and i not in self.opened]
                if closed:
                    reach += [i for i in closed if i not in reach]
                else:
                    about = next((str(l.get("about") or "") for l, i in zip(links, ids) if i not in self.rows), "") or "the listing it points at"
                    lost.append(f"{need} -> find {about}")
            pending = [p for p in pending if p not in self.ref_values]
            if not reach and lost and self.chain_searches < 2 and self.left() >= 8 and self.allow(self.pace() * 2 + 15):
                # the link is not among the rows seen: one search by the words that describe it, then the walk reads what the search brought
                self.chain_searches += 1
                if self.search_about(lost[0].split(" -> find ", 1)[-1]):
                    continue
            if not reach or not tool or not target or self.left() < len(reach) + 4 or self.finished:
                break
            for ident in reach[: max(0, min(3, 6 - len(seen)))]:
                faults: list[str] = []
                args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
                if faults:
                    break
                reply = self.act(tool, args)
                if reply is None or self.finished:
                    break
                self.observe(tool, reply.get("observation"))
                seen.append(f"{ident}: " + self.show(reply.get("observation"))[:1500])
        text = ""
        if seen:
            text += "[Kyo opened the pages a requirement points at]\n" + "\n".join(seen) + "\n"
        if self.ref_values:
            text += "[values read from the pages requirements point at] " + "; ".join(f"{k} -> {v}" for k, v in self.ref_values.items()) + "\n"
        hold = bool(lost) and self.trail_blocks < 2 and self.left() > 5
        if hold or (early and lost):
            if hold and not early:
                self.trail_blocks += 1      # an early walk does not use up the checkout's two holds
            text += ("The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: "
                     + " | ".join(lost[:3]) + ". Search for it by its shop, brand or words, open it, apply the value its page gives, then check out.")
        return text, hold

    def search_about(self, about: str) -> bool:
        # one search for a listing a chain points at, by the words that describe it (its shop, brand, kind) as the page or the request names them
        try:
            search = self.roles.get("search")
            if not search:
                return False
            shape = self.shapes.get(search) or {}
            props = (self.schemas.get(search) or {}).get("properties") or {}
            used = [k for _, a in self.search_log for k, v in a.items() if isinstance(v, str) and k != shape.get("next_arg")]
            param = max(set(used), key=used.count) if used else next((k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "string" and k != shape.get("next_arg")), None)
            words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", about) if len(w) >= 3]   # the search engine weighs the words itself
            phrase = " ".join(words[:6])
            if param is None or len(phrase) < 3:
                return False
            faults: list[str] = []
            call = _shape({param: phrase}, self.schemas.get(search), "args", faults)
            if faults:
                return False
            reply = self.act(search, call)
            if reply is None or self.finished:
                return False
            self.observe(search, reply.get("observation"))
            self.trace.append({"kind": "chain_search", "phrase": phrase, "rows": len(self.records_in(reply.get("observation")))})
            return True
        except Exception as exc:
            self.trace.append({"kind": "chain_search_error", "error": f"{type(exc).__name__}: {exc}"[:160]})
            return False

    def offer_tool(self) -> str | None:
        """Tool for re-reading a listing's live offer: the offer view when the store has one, else the single-listing view. Never the
        search tool: a search for a listing's id shows a result row, not the listing's page, and the grader counts only the page."""
        offer = self.roles.get("offer")
        if offer and offer != self.roles.get("search"):
            return offer
        return self.roles.get("open")

    def reread_offers(self, changed_only: bool = False) -> str:
        """Re-read each cart line with the store's offer view just before checkout: stock and price can change unannounced."""
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return ""
        since = self.version_step if changed_only else max(self.step - 2, self.refused_at, self.version_step)
        stale = [i for i in self.cart_ids() if self.offer_seen.get(i, -99) < since and i not in self.gone and (not changed_only or i not in self.full or since > 0)]
        if not stale or self.left() < len(stale) + 3:
            return ""
        lines = []
        for ident in stale:
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
            if faults:
                break
            before = dict(self.rows.get(ident) or {})
            reply = self.act(tool, args)
            if reply is None or self.finished:
                break
            observation = reply.get("observation")
            self.offer_seen[ident] = self.step
            self.observe(tool, observation)
            # compare the stored record, so a re-read that arrives wrapped together with a note still shows its change
            after = self.rows.get(ident) or {}
            moved = [k for k, v in after.items() if k in before and before[k] != v and not isinstance(v, (dict, list))]
            if moved:
                lines.append(f"{ident} changed ({', '.join(moved[:4])}): {_dump(_clip(observation, 300))[:900]}")
        self.trace.append({"kind": "reread", "ids": stale, "changed": len(lines)})
        if lines:
            self.version += 1
        return "\n".join(lines)

    def sold_out(self, ident: str) -> bool:
        """A note or a refusal named the listing as gone, or a flag of its own record turned from true to false."""
        return ident in self.unavailable or ident in self.gone

    def swap_hint(self) -> str:
        """A cart line whose newest record says it is sold out: name replacements among listings already read, once per line."""
        lost = [i for i in self.cart_ids() if i not in self.gone and self.sold_out(i)]
        if not lost:
            return ""
        for ident in lost:
            self.gone[ident] = self.step
        self.leads = [i for i in self.leads if i not in lost]
        self.version += 1
        picks = self.replacements(lost)
        text = f"[Kyo] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        if picks and self.left() <= 6:
            return text + f" Few steps are left: in this turn add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks[1:]) or 'none'}."
        if picks and self.brief.get("pick") == "cheapest":
            return text + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(picks)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {picks[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        if picks:
            return text + f" Listings already read that appear to fit that item instead: {', '.join(picks)}. In this turn take the sold-out line out of the cart and add {picks[0]} at the requested quantity, then check out."
        return text + " Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits."

    def find_sold(self) -> str:
        """A refusal said a chosen listing sold out without naming it: a lone cart line is that listing; several lines
        are re-read with the offer view so their records show which one. Returns what the re-read showed."""
        ids = [i for i in self.cart_ids() if i not in self.gone]
        if any(self.sold_out(i) for i in ids):
            return ""
        # a refusal is a change notice until a re-read shows the line gone (race 9: a refusal worded like a sell-out marked a live line gone)
        return self.reread_offers() if ids and self.allow(self.pace() * (len(ids) + 1) + 6) else ""

    def replacements(self, lost: list[str]) -> list[str]:
        if self.cost > SPEND_SOFT:
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
        prompt = SWAP + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        prompt += "\n\nSOLD-OUT CART LINE:\n" + "\n".join(f"{i}: {self.record_of(i)}" for i in lost) + "\n\nCART NOW:\n" + self.cart_text()
        prompt += "\n\nOTHER LISTING RECORDS SEEN (newest first):\n" + "\n".join(records)
        got = self.ask_json(prompt, [HEAVY, LIGHT], 1200)
        fits: list[tuple[float, str]] = []
        for entry in _items((got or {}).get("fits")):
            ident = str(entry.get("id") if isinstance(entry, dict) else entry)
            price = entry.get("price") if isinstance(entry, dict) else None
            if ident in self.rows and ident not in lost and ident not in self.gone and ident not in self.cart_ids() and ident not in [i for _, i in fits]:
                fits.append((float(price) if isinstance(price, (int, float)) and not isinstance(price, bool) else float("inf"), ident))
        # sorted here: the model names fits well but orders them by price unreliably
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.brief.get("pick") == "cheapest" else [i for _, i in fits]
        self.trace.append({"kind": "swap", "lost": lost, "fits": fits})
        return ranked[:4]

    def audit(self) -> str:
        if self.audited or self.cost > SPEND_SOFT:
            return ""
        self.audited = True
        got = self.audit_call()
        named = self.audit_fits(got)
        if not named:
            # rows whose page was never read can still fit: open the promising ones (one step) and look again
            rows = [str(i) for i in _items((got or {}).get("open")) if str(i) in self.rows and str(i) not in self.opened and not self.sold_out(str(i))]
            if rows and self.open_rows(rows):
                got = self.audit_call()
                named = self.audit_fits(got)
        if not named:
            return ""
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return "A careful review thinks these listings may fit: " + "; ".join(f"item {k}: {', '.join(v)}" for k, v in named.items()) + (f" ({got.get('reason')})" if got.get("reason") else "") + ". Open them and buy what truly fits; end without a purchase only if each one fails."

    def audit_call(self) -> dict[str, Any] | None:
        records = []
        for ident in reversed(self.seen_order):
            if self.sold_out(ident):
                continue
            record = self.full.get(ident) if ident in self.full else None
            records.append(("OPENED " + _dump(_clip(record, 200))[:1200]) if record else ("ROW " + _dump(_clip(self.rows.get(ident), 120))[:500]))
            if len(records) >= 30:
                break
        if not records:
            return None
        prompt = AUDIT + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        prompt += "\n\nLISTING RECORDS SEEN (newest first):\n" + "\n".join(records)
        got = self.ask_json(prompt, [HEAVY, LIGHT], 1500)
        self.trace.append({"kind": "audit", "got": got})
        return got

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get("fits")
        if not isinstance(fits, dict):
            return {}
        named = {str(k): [str(i) for i in _items(v) if str(i) in self.rows and not self.sold_out(str(i))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def open_rows(self, rows: list[str]) -> bool:
        """Open listings seen only as search rows: side by side in one step when the store can, else the first two one by one."""
        many, joiner = self.compare_target(), self.roles.get("compare")
        tool = self.facts_tool()
        target = self.single_target(tool)
        if many and joiner and not self.compare_bare():
            plan = [(joiner, {many: rows[: self.compare_room()]})]
        elif tool and target:
            plan = [(tool, {target: i}) for i in rows[:2]]
        else:
            return False
        if self.left() < len(plan) + 5:
            return False
        before = len(self.full)
        for name, args in plan:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(name), "args", faults)
            if faults:
                return False
            reply = self.act(name, args)
            if reply is None or self.finished:
                break
            self.observe(name, reply.get("observation"))
        self.trace.append({"kind": "audit_open", "rows": rows, "opened": len(self.full) - before})
        return len(self.full) > before

    def cart_calls(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        """Arguments that make the cart hold `lines` and none of `drop`, read off the cart tool's schema; None when it cannot be said."""
        name = self.roles.get("cart") or ""
        schema = self.schemas.get(name) or {}
        props = schema.get("properties") or {}
        for key, spec in props.items():
            shape = spec.get("items") if isinstance(spec, dict) and "array" in _kinds(spec) else None
            if isinstance(shape, dict) and isinstance(shape.get("properties"), dict):
                inner = shape["properties"]
                id_key = next((k for k, s in inner.items() if isinstance(s, dict) and "string" in _kinds(s) and "enum" not in s), None)
                count_key = next((k for k, s in inner.items() if isinstance(s, dict) and "integer" in _kinds(s)), None)
                if not id_key:
                    return None
                return [{key: [{id_key: i, **({count_key: n} if count_key else {})} for i, n in lines.items()]}]
        texts = [k for k, s in props.items() if isinstance(s, dict) and "string" in _kinds(s) and "enum" not in s]
        target = (next((k for t, k in self.id_args if t == name and k in texts), None)
                  or next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None))
        count = next((k for k, s in props.items() if isinstance(s, dict) and "integer" in _kinds(s)), None)
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2), None)
        if not target:
            return None
        calls: list[dict[str, Any]] = []
        for ident in drop:
            have = self.cart_lines.get(ident, 1)
            if mode:
                calls.append({target: ident, mode: props[mode]["enum"][1], **({count: have} if count else {})})
            elif count and self.cart_sets(name):
                calls.append({target: ident, count: 0})
            elif count and isinstance(props[count].get("minimum"), int) and props[count]["minimum"] <= -have:
                calls.append({target: ident, count: -have})
            else:
                return None
        for ident, want in lines.items():
            have = self.cart_lines.get(ident, 0) if ident in self.cart_ids() else 0
            if have == want:
                continue
            if have and not self.cart_sets(name):
                return None
            add = {target: ident, **({mode: props[mode]["enum"][0]} if mode else {})}
            # a cart tool without a count argument raises the line by one per call: one call per unit still wanted
            calls.extend([{**add, count: want}] if count else [dict(add) for _ in range(max(1, want - have))])
        return calls

    def complete_cart(self) -> str:
        """The agent ends without buying although the independent check confirmed a reviewed listing: Kyo carts the reviewed
        lines itself (one per item, keeping cart lines of the other items) and checks out, when a fresh check passes that cart."""
        cart, buy = self.roles.get("cart"), self.roles.get("buy")
        if self.completed or not self.confirmed or self.sold_out(self.confirmed) or not cart or not buy:
            return ""
        self.completed = True
        cat_key = self.category_key()

        def kind(ident: str) -> str:
            return str((self.rows.get(ident) or {}).get(cat_key) or "").strip().lower() if cat_key else ident

        plan: dict[str, int] = {}
        for number, ids in self.lead_items.items():
            index = _count(number)
            pick = next((i for i in ids if not self.sold_out(i)), None)
            if pick and index and 1 <= index <= len(self.items) and kind(pick) not in {kind(p) for p in plan}:
                plan[pick] = self.items[index - 1].qty
        if self.confirmed not in plan:
            return ""
        now = self.cart_ids()
        keep = {i: self.cart_lines.get(i, 1) for i in now if i not in plan and not self.sold_out(i) and kind(i) not in {kind(p) for p in plan}}
        drop = [i for i in now if i not in plan and i not in keep]
        if len(plan) + len({kind(i) for i in keep}) < len(self.items):
            return ""
        lines = {**keep, **plan}
        calls = self.cart_calls(lines, drop)
        if not calls or len(calls) + 1 > self.left():
            return ""
        # acting costs no model call; a cart the checks have not passed yet gets one check, allowed up to the hard spend cap
        self.fill_facts(list(lines), len(calls) + 3)
        self.check_cart(list(lines), SPEND_HARD)
        if not self.verdicts.get(self.check_key(list(lines))):
            return ""
        done = []
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), "args", faults)
            if faults:
                return ""
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return ""
            self.note_cart(cart, args)
            self.observe(cart, reply.get("observation"))
            done.append(_dump(args))
        faults = []
        args = _shape(self.fill_required(buy, dict(self.brief.get("buy_args") or {}), "buy"), self.schemas.get(buy), "args", faults)
        if faults:
            return ""
        replies = []
        # a refused checkout (a price, budget or stock change) gets one retry after re-reading the lines and a fresh check
        for attempt in range(2):
            if attempt:
                self.reread_offers()
                self.check_cart(list(lines), SPEND_HARD)
                if (not self.verdicts.get(self.check_key(list(lines))) or self.left() < 1 or self.finished
                        or any(self.sold_out(i) for i in lines)):
                    break
            go, said = self.gate("order")
            if said:
                replies.append(said)
            if not go:
                break
            reply = self.act(buy, args)
            self.trace.append({"kind": "completed", "lines": lines, "drop": drop, "reply": _clip(reply, 200)})
            if reply is None:
                break
            self.buys += 1
            if not reply.get("finished"):
                self.refused_at = self.step
            self.observe(buy, reply.get("observation"))
            replies.append(self.show(reply.get("observation")))
            if self.finished or reply.get("finished"):
                break
        if not replies:
            return ""
        return ("Not run (Kyo acted instead): the independent check confirmed " + self.confirmed + ", so Kyo set the cart (" + "; ".join(done)
                + ") and checked out. Store reply: " + " | then: ".join(replies))

    # ---------- tool use ----------
    def is_buy(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.roles.get("buy"):
            return False
        fixed = self.brief.get("buy_args") if isinstance(self.brief.get("buy_args"), dict) else {}
        return all(args.get(k) == v for k, v in fixed.items())

    def is_refuse(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.roles.get("refuse"):
            return False
        fixed = self.brief.get("refuse_args") if isinstance(self.brief.get("refuse_args"), dict) else {}
        return all(args.get(k) == v for k, v in fixed.items())

    def offered(self) -> list[dict[str, Any]]:
        left = self.left()
        ask = self.roles.get("ask")
        keep: set[str] | None = None
        ends = {self.roles.get("buy"), self.roles.get("refuse")} - {None}
        if left <= self.tail():
            keep = set(ends)
        elif left <= self.tail() + 3:
            keep = set(ends) | {self.roles.get("cart"), self.roles.get("cart_view")}
        elif left <= len(self.items) + 9 and self.buys == 0:
            keep = set(ends) | {self.roles.get("cart"), self.roles.get("cart_view"), self.roles.get("offer")}
        tools = [t for t in self.tools if t["function"]["name"] != ask]
        if keep is not None and ends:
            narrowed = [t for t in tools if t["function"]["name"] in keep]
            tools = narrowed or tools
        return tools

    def grounded(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        """Free text of the finishing call, written from Kyo's records when the model left it empty and completed when it misses the
        budget now in force or a cart line: the lines (id, price, units, stock), the requirements in force, what was dropped, the
        budget and the total. Race 13: after a change at the checkout no order passed without such a text. Never throws."""
        try:
            if tool != self.roles.get("buy"):
                return args
            props = (self.schemas.get(tool) or {}).get("properties") or {}
            free = [k for k, s in props.items() if isinstance(s, dict) and "string" in _kinds(s) and "enum" not in s]
            if not free:
                return args
            field = free[0]
            limit = int(props[field].get("maxLength") or 900)
            budget = self.budget if isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool) else None
            ids = self.live_lines()
            total = 0.0
            said: list[str] = []
            for ident in ids:
                row = self.rows.get(ident) or {}
                cost = self.cost_of(ident)
                units = self.cart_lines.get(ident, 1)
                total += (cost or 0.0) * units
                name = ""
                for value in row.values():
                    if isinstance(value, str) and 12 <= len(value) <= 120 and not _id_like(value, True):
                        name = value[:60]
                        break
                piece = ident + (f" ({name})" if name else "")
                piece += f" at {cost:g} x {units}" if cost is not None else f" x {units}"
                if self.buyable(ident):
                    piece += ", in stock by its latest record"
                said.append(piece)
            wants = []
            for _, need in self.needs_now():
                prop, _, value = need.partition(":")
                wants.append(f"{prop.strip()} = {value.strip()}" if value else need)
            gone = [d for item in self.items for d in item.dropped]
            scope = "for each item" if self.brief.get("budget_scope") == "each" else "in total"
            money = f"Budget now {budget:g} {scope}; the total is {total:g}, within it." if budget is not None else f"No budget was set; the total is {total:g}."
            wants = list(dict.fromkeys(wants))
            text = str(args.get(field) or "").strip()
            if text.startswith("Checking out ") and "each line is the listing whose attributes state" in text:
                text = ""                                     # the code's own stock sentence (a required field filled before this point) is no account
            if not text:
                text = self.account_of(ids, said, gone, money)
            if not text:
                text = "Ordering " + "; ".join(said) + ". Requirements in force: " + ("; ".join(wants) or "none beyond the item itself") + "."
                if gone:
                    text += f" Dropped on the shopper's word: {', '.join(gone[:4])}."
                text += " " + money
            else:
                if budget is not None and f"{budget:g}" not in text and str(int(budget)) not in text:
                    text += " " + money
                absent = [piece for ident, piece in zip(ids, said) if ident not in text]
                if absent:
                    text += " Lines: " + "; ".join(absent) + "."
            latest = " ".join(self.recent_notes)
            if latest and latest[:40] not in text and not any(d in text for d in re.findall(r"\d+", latest)):
                text += " Shopper's latest note: " + latest      # the changed figure is cited even when the note could not be read in time
            changed = dict(args)
            changed[field] = text[:limit]
            return changed
        except Exception as exc:
            self.trace.append({"kind": "grounded_error", "error": f"{type(exc).__name__}: {exc}"[:160]})
            return args

    def account_of(self, ids: list[str], said: list[str], gone: list[str], money: str) -> str:
        # the order's account written by the model from the records (race 16: 2 of 115 accounts written by code passed, 16 of 54 written in prose);
        # skipped when the watchdog is finishing, when the pace leaves no room, or when the spend reserve is used up
        try:
            if self.closing or not ids or not self.allow(self.pace() + 12) or not self.spend_ok(0.01, final=True):
                return ""
            others = [i for i in self.seen_order if i in self.opened and i in self.full and i not in ids][-4:]
            prompt = (ACCOUNT + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
                      + "\n\nORDER LINES:\n" + "\n".join(said) + "\n\nRECORDS OF THE ORDERED LISTINGS:\n" + "\n".join(f"{i}: {self.record_of(i)}" for i in ids)
                      + ("\n\nOTHER PAGES OPENED (a requirement may take its value from one of them):\n" + "\n".join(f"{i}: {self.record_of(i)}" for i in others) if others else "")
                      + "\n\nGIVEN UP BY THE SHOPPER: " + (", ".join(gone[:4]) or "nothing") + "\nMONEY: " + money + "\nLATEST NOTES: " + (" ".join(self.recent_notes) or "none"))
            got = self.ask_json(prompt, [HEAVY, LIGHT], 700, final=True, reasoning={"effort": "none"}) or {}
            account = re.sub(r"\s+", " ", str(got.get("account") or "")).strip()
            if len(account) < 40:
                return ""
            self.trace.append({"kind": "account", "chars": len(account)})
            return account
        except Exception as exc:
            self.trace.append({"kind": "account_error", "error": f"{type(exc).__name__}: {exc}"[:160]})
            return ""

    def quiet(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        props = schema.get("properties") or {}
        out = dict(args)
        for key, spec in props.items():
            if not isinstance(spec, dict) or "string" not in _kinds(spec) or "enum" in spec:
                continue
            value = out.get(key)
            if value is None and key not in (schema.get("required") or []):
                continue
            text = str(value or "")
            named = any(ident in text for ident in self.rows) or self.brief.get("quiet_decline") is True
            if named or not text:
                out[key] = "Sorry, nothing in the store meets all of your requirements, so I am not buying anything."[: spec.get("maxLength", 200)]
        return out

    def resolve_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """A listing given by its title where a tool takes listing ids: use the id when one seen row has that title,
        otherwise say what is wrong, so no step is spent on a call the store cannot match."""
        if name not in {self.roles.get(r) for r in ("cart", "open", "offer", "compare")} - {None} or not self.id_key:
            return args, ""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and ("string" in _kinds(v) or "array" in _kinds(v)) and "enum" not in v]
        # only the tool's one free argument, and only where it is known to take ids (the cart, or a call seen carrying one)
        if len(free) != 1 or (name != self.roles.get("cart") and (name, free[0]) not in self.id_args):
            return args, ""

        def fix(value: Any) -> tuple[Any, str]:
            if isinstance(value, str) and value.strip() in self.rows:
                return value.strip(), ""
            if not isinstance(value, str) or _id_like(value, True) or len(value.strip()) < 8:
                return value, ""
            low = value.strip().lower()
            hits = [i for i, row in self.rows.items() if any(k != self.id_key and isinstance(v, str) and v.strip().lower().startswith(low) for k, v in row.items())]
            if len(hits) == 1:
                return hits[0], ""
            if hits:
                return value, f"several listings carry the text {value!r} ({', '.join(hits[:6])}); pass one listing id"
            return value, f"{value!r} is not a listing id; pass the id from the result rows (their {self.id_key} field)"

        out = dict(args)
        for key in free:
            if key not in args:
                continue
            fixed = [fix(v) for v in args[key]] if isinstance(args[key], list) else [fix(args[key])]
            problem = next((p for _, p in fixed if p), "")
            if problem:
                return args, problem
            out[key] = [v for v, _ in fixed] if isinstance(args[key], list) else fixed[0][0]
        return out, ""

    def cart_sets(self, name: str) -> bool:
        """A cart tool whose count may be exactly 0 and that has no mode for taking lines off sets a line's quantity: 0 removes it."""
        props = self.schemas.get(name, {}).get("properties") or {}
        if any(isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2 for s in props.values()):
            return False
        return any(isinstance(s, dict) and "integer" in _kinds(s) and s.get("minimum") == 0 for s in props.values())

    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        self.cart_touched = True
        self.cart_at = self.step
        spec = self.schemas.get(name, {}).get("properties") or {}
        removal = False
        for key, value in args.items():
            options = (spec.get(key) or {}).get("enum")
            if isinstance(options, list) and len(options) >= 2 and value in options[1:]:
                removal = True
        loose_count = next((_count(v) for v in args.values() if _count(v) is not None and not isinstance(v, str)), None)
        lines: list[tuple[str, int | None]] = []
        whole = False
        for key, value in args.items():
            if isinstance(value, list):
                shape = (spec.get(key) or {}).get("items") or {}
                if not value and isinstance(shape, dict) and (shape.get("type") == "object" or "properties" in shape):
                    whole = True
                for line in value:
                    if isinstance(line, dict):
                        whole = True
                        ident = next((v for v in line.values() if isinstance(v, str) and v in self.rows), None)
                        count = next((_count(v) for v in line.values() if _count(v) is not None and not isinstance(v, str)), 1)
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
                have = have - count if count else 0
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
            text = text[:OBS_LIMIT] + "...(cut)"
        return text

    def compare_target(self) -> str | None:
        schema = self.schemas.get(self.roles.get("compare") or "") or {}
        props = schema.get("properties") or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and "array" in _kinds(v)), None)
        if not target or any(k != target for k in schema.get("required") or []):
            return None
        return target

    def compare_room(self) -> int:
        """How many listings one side-by-side call may carry: the schema's limit, else what the store has shown to work."""
        spec = ((self.schemas.get(self.roles.get("compare") or "") or {}).get("properties") or {}).get(self.compare_target() or "") or {}
        if isinstance(spec.get("maxItems"), int):
            return spec["maxItems"]
        if self.compare_bad:
            return max(2, min(self.compare_bad - 1, max(self.compare_ok, 2)))
        return max(4, self.compare_ok)

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Several single-listing opens in one turn become one side-by-side call: one step instead of several."""
        opener, joiner, target = self.roles.get("open"), self.roles.get("compare"), self.single_target(self.roles.get("open"))
        many = self.compare_target()
        if not opener or not joiner or not target or not many or opener == joiner:
            return calls
        # only into a side-by-side view that shows what the opens would (attributes)
        if self.compare_bare():
            return calls
        picks: list[tuple[int, str]] = []
        for index, call in enumerate(calls):
            fn = call.get("function") or {}
            args = _json_obj(fn.get("arguments")) if isinstance(fn.get("arguments"), str) else None
            ident = str((args or {}).get(target) or "").strip()
            if fn.get("name") == opener and args and set(args) == {target} and ident in self.rows and ident not in self.gone:
                picks.append((index, ident))
        picks = picks[: self.compare_room()]
        if len(picks) < 2:
            return calls
        first = picks[0][0]
        out = []
        for index, call in enumerate(calls):
            if index == first:
                out.append({"id": call["id"], "type": "function", "function": {"name": joiner, "arguments": _dump({many: [i for _, i in picks]})}})
            elif index in {i for i, _ in picks}:
                out.append({"id": call["id"], "joined": True})
            else:
                out.append(call)
        self.trace.append({"kind": "joined", "ids": [i for _, i in picks]})
        return out

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        calls = self.join_opens(calls)
        for index, call in enumerate(calls):
            allowed = {t["function"]["name"] for t in self.offered()}
            cid = call["id"]
            if call.get("joined"):
                out.append((cid, "Shown in the side-by-side result above: Kyo opened these listings together in one step."))
                continue
            fn = call.get("function") or {}
            name = fn.get("name")
            raw = fn.get("arguments")
            if self.finished:
                out.append((cid, "Not run: the episode is over."))
                continue
            if name == self.roles.get("ask"):
                out.append((cid, "Not run (no step used): Kyo's interviewer handles every message to the shopper."))
                continue
            if name not in self.schemas:
                out.append((cid, f"Not run (no step used): unknown tool. Tools: {', '.join(sorted(allowed))}."))
                continue
            if name not in allowed:
                out.append((cid, "Not run (no step used): too few steps are left for that; use: " + ", ".join(sorted(allowed)) + "."))
                continue
            if isinstance(raw, dict):
                args = raw
            else:
                args = _json_obj(raw) if (raw or "").strip() else {}
                if args is None:
                    out.append((cid, "Not run (no step used): the arguments were not one JSON object."))
                    continue
            faults: list[str] = []
            args = _shape(args, self.schemas[name], "args", faults)
            if faults:
                out.append((cid, "Not run (no step used): " + "; ".join(faults[:6])))
                continue
            args, wrong = self.resolve_ids(name, args)
            if wrong:
                out.append((cid, "Not run (no step used): " + wrong + "."))
                continue
            trimmed = ""
            many = self.compare_target() if name == self.roles.get("compare") else None
            if many and isinstance(args.get(many), list) and len(args[many]) > self.compare_room() and self.compare_bad:
                room = self.compare_room()
                trimmed = f"\n[Kyo] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
                args = {**args, many: args[many][:room]}
            target = self.single_target(name)
            if target and name in (self.roles.get("open"), self.roles.get("offer")) and str(args.get(target)) in self.gone:
                out.append((cid, f"Not run (no step used): {args[target]} sold out at step {self.gone[str(args[target])]} and will not come back. Replace its cart line with another fitting listing."))
                continue
            if name == self.roles.get("cart") and any((_count(v) or 0) < 0 for v in args.values() if not isinstance(v, str)):
                view = _dump(self.cart_view) if self.cart_seen_at >= self.cart_at else ""
                absent = [v for v in args.values() if isinstance(v, str) and v in self.rows and v not in self.cart_lines and v not in view and v not in self.cart_said]
                if absent:
                    out.append((cid, f"Not run (no step used): {absent[0]} is not in the cart, so there is nothing to take off. Cart now: " + self.cart_text()))
                    continue
            ending = self.is_buy(name, args) or self.is_refuse(name, args)
            if ending and len(calls) > 1:
                out.append((cid, "Not run (no step used): send the finishing call on its own, after reading your other results."))
                continue
            if not ending and self.left() <= self.tail() and (self.roles.get("buy") or self.roles.get("refuse")):
                out.append((cid, "Not run (no step used): the last steps are kept for checking out (with the shopper's approval of the basket when it is due) or ending without a purchase."))
                continue
            mark = name + _dump(args)
            if not ending and name != self.roles.get("cart") and mark in self.recent:
                out.append((cid, "Not run (no step used): you just made this same call and nothing changed; its result is above."))
                continue
            readers = {self.roles.get(r) for r in ("search", "open", "offer", "compare")} - {None}
            cached = self.read_cache.get(mark)
            if name in readers and cached and cached[0] == self.version:
                out.append((cid, f"Not run (no step used): the same call at step {cached[1]} returned this, and nothing has changed since:\n" + cached[2][:3000]))
                continue
            if target and name in readers and name != self.roles.get("compare") and self.shown_already(name, str(args.get(target))):
                out.append((cid, f"Not run (no step used): {args[target]} has shown everything this view gives since the last change. Its record: " + self.record_of(str(args[target]))))
                continue
            if many and isinstance(args.get(many), list) and args[many]:
                keep = [i for i in args[many] if not self.shown_already(name, str(i))]
                if not keep:
                    out.append((cid, "Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n"
                                + "\n".join(f"{i}: {self.record_of(str(i))}" for i in args[many])[:4000]))
                    continue
                if len(keep) < len(args[many]):
                    trimmed += f"\n[Kyo] Left out {', '.join(str(i) for i in args[many] if i not in keep)}: shown in full already, nothing changed since."
                    args = {**args, many: keep}
                    mark = name + _dump(args)
            if ending and self.left() > 2 and self.allow(self.pace() * 2 + 12):
                if self.is_buy(name, args):
                    wider = self.wider_hold() if self.allow(60) else ""
                    if wider:
                        out.append((cid, wider + "\nCart now: " + self.cart_text()))
                        continue
                    reread = self.reread_offers() if self.allow(self.pace() * (len(self.cart_ids()) + 1) + 6) else ""
                    if self.cart_touched and self.cart_seen_at < self.cart_at and self.roles.get("cart_view") and self.left() > 3 and self.allow(self.pace() + 4):
                        faults = []
                        view_args = _shape({}, self.schemas.get(self.roles["cart_view"]), "args", faults)
                        view = self.act(self.roles["cart_view"], view_args) if not faults else None
                        if view is not None:
                            self.cart_view = view.get("observation")
                            self.cart_seen_at = self.step
                            change = self.observe(self.roles["cart_view"], self.cart_view)
                            if change:
                                self.queued = []
                                out.append((cid, "Not run yet (no step used). Looking at the cart showed a change: " + change + "\nCart now: " + self.cart_text() + "\nApply the change, fix the cart if needed, then check out."))
                                continue
                    swap = self.swap_hint()
                    # the checkout is held back at most twice for a sold-out line, so a wrong cart record cannot stall the episode
                    held = [i for i in self.cart_ids() if i in self.gone] if self.held < 2 else []
                    if swap or held:
                        self.held += 0 if swap else 1
                        out.append((cid, "Not run yet (no step used). " + (swap or f"{', '.join(held)} in the cart is sold out: replace that line first.") + "\nCart now: " + self.cart_text()))
                        continue
                    chain, hold = self.trail() if self.allow(45) else ("", False)
                    if hold:
                        out.append((cid, "Not run yet (no step used). " + chain + "\nCart now: " + self.cart_text()))
                        continue
                    # the check judges attributes: read the cart lines' own (and, for a cheapest pick, a few cheaper rows')
                    read = (self.sweep_cheaper() + self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()) if self.allow(self.pace() * 4 + 10) else []
                    verdict = self.check_cart(limit=SPEND_HARD + SPEND_FINAL) if self.allow(30) else ""
                    if verdict:
                        seen = ("Kyo read these listings' attributes for the check (one step each):\n" + "\n".join(read) + "\n") if read else ""
                        out.append((cid, "Not run yet. " + chain + seen + verdict + "\nCart now: " + self.cart_text()))
                        continue
                    if reread and self.cost > SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). Kyo re-read the cart's listings just now:\n" + reread + "\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first."))
                        continue
                else:
                    if self.left() > 5 and self.allow(35):
                        review = self.audit()
                        if review:
                            out.append((cid, "Not run yet (no step used). " + review))
                            continue
                        if self.leads and self.left() > 4 and self.refuse_blocks == 0:
                            self.refuse_blocks = 1
                            self.trace.append({"kind": "refuse_blocked", "leads": list(self.leads)})
                            out.append((cid, "Not run yet (no step used). The review found listings that appear to fit: " + ", ".join(self.leads) + ". Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again."))
                            continue
                        if self.leads and self.left() > 4 and self.refuse_blocks == 1:
                            self.refuse_blocks = 2
                            first = self.leads[0]
                            self.check_cart([first], SPEND_HARD)
                            if self.verdicts.get(self.check_key([first])) and first in self.leads:
                                self.confirmed = first
                                self.trace.append({"kind": "refuse_blocked", "confirmed": first})
                                out.append((cid, f"Not run yet (no step used). An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."))
                                continue
            if self.is_refuse(name, args) and self.left() <= 5 and not self.late_held:
                # too late for the review: a decline here usually throws away a built cart because time ran out
                lines = [i for i in self.built if not self.sold_out(i)]
                fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
                missing = [i for i in lines if i not in self.cart_ids() or (i in self.dropped and not fresh)]
                if len(lines) >= len(self.items) and len(missing) + 1 <= self.left():
                    self.late_held = True
                    self.trace.append({"kind": "late_decline_held", "lines": lines, "missing": missing})
                    redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else "Check out the cart as it is."
                    out.append((cid, "Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. "
                                + redo + " End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again."))
                    continue
            if self.is_refuse(name, args) and not self.leads and not self.pushed and self.left() > self.max_steps // 3:
                self.pushed = True
                out.append((cid, f"Not run yet (no step used). {self.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if self.is_refuse(name, args):
                done = self.complete_cart()
                if done:
                    out.append((cid, done + (" [episode finished]" if self.finished else "")))
                    continue
                # an ending without a purchase while a buyable in-budget listing of a requested kind was seen: the field's
                # declines were wrong 9 times in 10 (race 11), so the least-contradicting listing is bought instead
                if self.left() >= 3 and self.fallback_order():
                    out.append((cid, "Not run as sent: a listing of the requested kind was in stock and inside the budget, so Kyo carted the one that meets the most requirements and checked out." + (" [episode finished]" if self.finished else "")))
                    continue
                args = self.quiet(name, args)
            if ending and self.is_buy(name, args) and self.left() >= 3:
                settled = self.settle_changes() if self.pending_changes else ""
                if settled and self.settle_holds < 2:
                    self.settle_holds += 1
                    out.append((cid, "Not run yet. " + settled + ".\nCart now: " + self.cart_text() + "\nCheck the lines against the requirements now in force, then check out."))
                    continue
                swapped = self.literal_swap()
                if swapped:
                    trimmed = "\n[Kyo] " + swapped + trimmed
                elif self.literal_holds < 2 and self.left() >= 5 and self.allow(self.pace() * 4 + 20):
                    failing = self.literal_report()
                    if failing:
                        self.literal_holds += 1
                        out.append((cid, "Not run yet (no step used). " + failing + "\nCart now: " + self.cart_text()))
                        continue
            if ending and self.left() >= 2:
                go, said = self.gate("order" if self.is_buy(name, args) else "end")
                if not go:
                    out.append((cid, "Not run yet. " + said + "\nCart now: " + self.cart_text()))
                    continue
                if said:
                    trimmed = "\n[Kyo] " + said + trimmed
            if ending and self.is_buy(name, args):
                args = self.grounded(name, args)
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, "The call failed (network or server). You may try again."))
                continue
            observation = reply.get("observation")
            if many and isinstance(args.get(many), list) and len(args[many]) > 1:
                shown = _dump(observation)
                if not any(str(i) in shown for i in args[many]):
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    trimmed += f"\n[Kyo] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_room()} per call."
                else:
                    self.compare_ok = max(self.compare_ok, len(args[many]))
            self.recent = (self.recent + [mark])[-3:]
            self.id_args.update((name, k) for k, v in args.items() if (v if isinstance(v, list) else [v]) and all(isinstance(x, str) and x in self.rows for x in (v if isinstance(v, list) else [v])))
            self.absorb(observation)
            if name == self.roles.get("compare"):
                rows = self.records_in(observation)
                if rows:
                    self.facts_from[name] = self.facts_from.get(name, False) or any(_fact_block(r) for _, r in rows)
            if name in readers - {self.roles.get("search")}:
                self.note_view(name, observation)
            # every job but searching: only a search result is paged
            plain = {tool for job, tool in self.roles.items() if job != "search"}
            if name == self.roles.get("search") and isinstance(args, dict):
                self.search_log.append((name, dict(args)))
            paged = self.more_pages(name, args, observation) if name not in plain and not ending else ""
            if self.single_target(name) and name not in (self.roles.get("cart"), self.roles.get("buy"), self.roles.get("refuse"), self.roles.get("ask"), self.roles.get("search")):
                listing = self.ident(observation) if isinstance(observation, dict) else None
                if listing:
                    self.facts_from[name] = self.facts_from.get(name, False) or bool(_fact_block(observation))
                    if name == self.roles.get("open") and not self.facts_from[name]:
                        other = next((t for t in self.schemas if t != name and self.single_target(t) and self.facts_from.get(t) is not False
                                      and t not in (self.roles.get("cart"), self.roles.get("buy"), self.roles.get("refuse"), self.roles.get("ask"), self.roles.get("search"))), None)
                        if other:
                            self.roles["open"] = other
                    if not self.facts_from[name] and name not in (self.roles.get("open"), self.roles.get("search")) and not self.roles.get("offer"):
                        self.roles["offer"] = name
                    if name == self.offer_tool():
                        self.offer_seen[listing] = self.step
            if name == self.roles.get("cart"):
                self.note_cart(name, args)
                self.cart_said = _dump(observation)
                self.recent = []
                # lines taken off by a zero or negative count (unless the reply still lists them), and the fullest
                # cart built so far, which is offered back when a late decline would throw it away
                counts = [_count(v) for v in args.values() if not isinstance(v, str) and _count(v) is not None]
                for value in args.values():
                    if counts and isinstance(value, str) and value in self.rows:
                        if counts[0] <= 0 and not _in_rows(observation, value):
                            self.dropped.add(value)
                        elif counts[0] > 0:
                            self.dropped.discard(value)
                lines = [i for i in self.cart_ids() if not self.sold_out(i) and i not in self.dropped]
                if len(lines) >= len(self.built):
                    self.built = lines
            if name == self.roles.get("cart_view"):
                self.cart_view = observation
                self.cart_seen_at = self.step
            if self.is_buy(name, args):
                self.buys += 1
                if not reply.get("finished"):
                    self.refused_at = self.step
            text = self.show(observation) + paged + trimmed
            if name not in plain and not ending:
                text += self.cheapest_here([observation] + self.paged_obs) + self.code_marks([observation] + self.paged_obs)
            base = text
            proof = self.evidence(observation)
            if proof:
                text += "\n" + proof
            if name == self.roles.get("cart") and any(_count(v) == 0 for v in args.values() if not isinstance(v, str)):
                if any(isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values()):
                    text += "\n[Kyo] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off."
            self.note_sold = False
            self.note_price = False
            lines = self.loose_lines(observation)
            if lines:
                meaning = self.take_note(lines)
                if meaning and self.note_price and not self.note_sold:
                    text += "\n[note read] " + meaning + ". A moved price on a cart line changes nothing else: the cart stays unless the new price breaks the budget."
                elif meaning:
                    text += "\n[note read] " + meaning + ". Apply this change to your picks before checking out."
            if self.is_buy(name, args) and not reply.get("finished") and (self.note_sold or any(line in self.spoken(observation) for line in self.sold_lines)):
                seen = self.find_sold()
                if seen:
                    text += "\n[Kyo re-read the cart's lines after the refusal]\n" + seen
            elif self.is_buy(name, args) and not reply.get("finished") and self.note_price and self.left() >= len(self.cart_ids()) + 3 and self.allow(self.pace() * (len(self.cart_ids()) + 1) + 15):
                # race 16: every one of our 27 cart changes after a notice failed; after a moved price the top agent re-read the line and ordered it again naming the new price (12 of 14 passed)
                seen = self.reread_offers()
                self.trace.append({"kind": "price_reread", "changed": bool(seen)})
                text += ("\n[Kyo re-read the cart's lines after the refusal: a price moved]\n" + (seen or "the records read as before")
                         + "\nOrder the same cart again naming the price now shown; replace a line only when its new price breaks the budget.")
            swap = self.swap_hint()
            if swap:
                text += "\n" + swap
            if self.queued:
                text += "\n[notes seen on Kyo's own calls] " + " | ".join(self.queued)
                self.queued = []
            if name in readers:
                self.read_cache[mark] = (self.version, self.step, base)
            if reply.get("finished"):
                text += " [episode finished]"
            out.append((cid, text))
        return out

    def finish(self, again: bool = False) -> str:
        """End the episode: check out a held cart, else end without a purchase. With `again`, a refused checkout is
        returned (as the store's reply) for the agent to deal with instead of being followed by an ending without a purchase."""
        if self.finished or self.broke or self.left() <= 0:
            return ""
        buy, refuse = self.roles.get("buy"), self.roles.get("refuse")
        live = self.live_lines()
        if buy and live and len(self.items) > 1:
            self.fill_missing_kinds()                         # a basket without a line for some requested kind cannot pass; a candidate of that kind still can
            live = self.live_lines()
        if buy and live:
            faults: list[str] = []
            args = _shape(self.fill_required(buy, dict(self.brief.get("buy_args") or {}), "buy"), self.schemas.get(buy), "args", faults)
            if not faults:
                # a reply that adds a requirement does not stop a forced finish: no steps are left to act on it
                if self.left() >= 3 and self.allow(self.pace() * 3 + 8):
                    if self.pending_changes:
                        self.settle_changes()
                    self.literal_swap()
                if self.left() >= 2 and self.allow(self.pace() + 12):
                    self.gate("order")
                if self.finished or self.left() <= 0:
                    return ""
                reply = self.settle(buy, args, repeat=not again)
                if self.finished or self.left() <= 0 or reply is None:
                    return ""
                return ("the checkout was refused: " + self.show(reply.get("observation"))) if again else ""
        if buy and not live and self.fallback_order():
            return ""
        if refuse:
            faults = []
            args = _shape(self.fill_required(refuse, self.quiet(refuse, dict(self.brief.get("refuse_args") or {})), "refuse"), self.schemas.get(refuse), "args", faults)
            if not faults:
                if self.left() >= 2 and self.allow(self.pace() + 12):
                    self.gate("end")
                if self.finished or self.left() <= 0:
                    return ""
                reply = self.act(refuse, args)
                self.trace.append({"kind": "forced_refuse", "reply": _clip(reply, 200)})
                if reply is not None and not self.finished:
                    self.observe(refuse, reply.get("observation"))
                    return "ending without a purchase was refused: " + self.show(reply.get("observation"))
        return ""

    def settle(self, tool: str, args: dict[str, Any], repeat: bool, guard: bool = False) -> dict[str, Any] | None:
        """The finishing call: sent once more after a failed call, and, with `repeat`, once more after a refusal (the store
        accepts the repeated finishing call; a stale order still beats no order). Refusals are read into the ledger."""
        reply = None
        for attempt in range(3):
            if self.finished or self.left() <= 0 or self.room() < 5:
                break
            if tool == self.roles.get("buy") and self.left() >= len(self.cart_ids()) + 2 and self.room() > self.pace() * (len(self.cart_ids()) + 1) + 20:
                self.reread_offers(changed_only=True)             # a line read before the last answer or notice is read again (race 13: 0/7 passed without that)
            args = self.grounded(tool, args)
            reply = self.act(tool, args, guard=guard)
            self.trace.append({"kind": "settle", "tool": tool, "attempt": attempt, "reply": _clip(reply, 160)})
            if reply is None:
                if attempt == 0:
                    continue                                  # a network blip: the same call once more
                break
            if reply.get("finished") or self.finished:
                break
            self.buys += 1
            self.refused_at = self.step
            self.observe(tool, reply.get("observation"))
            if not repeat or attempt >= 1:
                break
            if any(c.startswith("a price") for c in self.pending_changes) and self.left() >= len(self.cart_ids()) + 4 and self.room() > self.pace() * (len(self.cart_ids()) + 1) + 20:
                self.reread_offers()                          # a moved price is read from the line's page first, so the re-fit and the order's account use the new figure
            if self.pending_changes and self.left() >= 4 and self.room() > 30:
                self.settle_changes()                         # the refusal carried a change: re-fit the cart before the order is repeated
            if self.left() >= len(self.cart_ids()) + 2 and self.room() > self.pace() * (len(self.cart_ids()) + 1) + 20:
                self.reread_offers(changed_only=True)         # the ordered lines are read again after the change (race 13: 4/49 passed without that, 47/96 with it)
        return reply

    def live_lines(self) -> list[str]:
        """Cart lines the store can still sell: a cart of sold-out lines only is an empty cart."""
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        ids = self.cart_ids() if fresh else list(self.cart_lines)
        return [i for i in ids if not self.sold_out(i)]

    def kind_label(self, ident: str) -> str:
        cat_key = self.category_key()
        return str((self.rows.get(ident) or {}).get(cat_key) or "").strip().lower() if cat_key else ""

    def item_of(self, ident: str) -> int | None:
        """The requested item a listing belongs to, by its category label's words against the items' names and shelves."""
        label = self.kind_label(ident)
        words = {w for w in _stem(label) if len(w) >= 4} if label else set()
        best: tuple[int, int] | None = None
        for number, item in enumerate(self.items, 1):
            want = {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}
            score = len(words & want) if words else 0
            if _words(label) and _words(label) in (_words(item.shelf or ""), _words(item.name)):
                score += 3
            if score and (best is None or score > best[1]):
                best = (number, score)
        if best:
            return best[0]
        # no category label, or one that names none of the items: the row's own words (title, label) decide
        text = " ".join(str(v) for v in (self.rows.get(ident) or {}).values() if isinstance(v, str) and len(v) <= 200)
        have = _stem(text)
        for number, item in enumerate(self.items, 1):
            want = {w for w in _stem(item.name) | _stem(item.shelf or "") if len(w) >= 4}
            hits = sum(1 for w in want if w in have or any(h.startswith(w[:5]) for h in have if len(w) >= 5))
            if want and hits >= max(1, (len(want) + 1) // 2):
                return number
        return None

    def stock_says_no(self, ident: str) -> bool:
        """Without a named stock key: the newest record carries a false flag (or a zero count) under a key that speaks of stock,
        availability or buying."""
        for key, value in (self.rows.get(ident) or {}).items():
            if re.search(r"stock|avail|buy|purchas|orderable|sale", str(key), re.I) and (value is False or (isinstance(value, (int, float)) and not isinstance(value, bool) and value <= 0)):
                return True
        return False

    def cost_of(self, ident: str) -> float | None:
        """The listing's price: the named price key when the store's row shape is known, else the first numeric field of its newest
        record whose key reads like a price."""
        known = self.price_of(ident)
        if known is not None:
            return known
        for key, value in (self.rows.get(ident) or {}).items():
            if not re.search(r"pric|cost|amount|cent|penn|fee|total", str(key), re.I):
                continue
            if isinstance(value, dict):
                value = next((v for v in value.values() if isinstance(v, (int, float)) and not isinstance(v, bool)), None)
            if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", value):
                value = float(value)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
        return None

    def missing_kinds(self) -> list[str]:
        """Requested items with no live cart line of their kind."""
        have = {self.item_of(i) for i in self.live_lines()}
        return [f"item {n} ({it.name})" for n, it in enumerate(self.items, 1) if n not in have]

    def kind_pools(self, budget: float | None) -> dict[int, list[str]]:
        """Buyable listings of each requested item among the records seen, inside the budget when one is in force (a listing whose
        price is unknown cannot be shown to fit a budget, so it is left out)."""
        pools: dict[int, list[str]] = {}
        for ident in self.seen_order:
            number = self.item_of(ident)
            if number is None or self.sold_out(ident) or not self.buyable(ident) or (not self.stock_key and self.stock_says_no(ident)):
                continue
            price = self.cost_of(ident)
            qty = self.items[number - 1].qty
            if budget is not None and (price is None or price * qty > budget):
                continue
            pools.setdefault(number, []).append(ident)
        return pools

    def fill_missing_kinds(self) -> None:
        """Before a forced buy of a basket that lacks a line for some requested item: cart that item's cheapest buyable candidate
        (opened listings first) while the budget, the steps and the clock allow. Never throws."""
        try:
            cart = self.roles.get("cart")
            live = self.live_lines()
            have = {self.item_of(i) for i in live}
            missing = [n for n in range(1, len(self.items) + 1) if n not in have]
            if not cart or not missing:
                return
            budget = self.budget if isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool) else None
            each = self.brief.get("budget_scope") == "each"
            pools = self.kind_pools(budget)
            spent = sum((self.cost_of(i) or 0.0) * self.cart_lines.get(i, 1) for i in live)
            adds: dict[str, int] = {}
            for number in missing:
                pool = sorted(pools.get(number, []), key=lambda i: (i not in self.full, self.cost_of(i) if self.cost_of(i) is not None else float("inf")))
                if not pool:
                    continue
                pick = pool[0]; qty = self.items[number - 1].qty
                price = self.cost_of(pick) or 0.0
                if budget is not None and not each and spent + price * qty > budget:
                    continue
                spent += price * qty
                adds[pick] = qty
            if not adds:
                self.trace.append({"kind": "fill_missing_none", "missing": missing})
                return
            calls = self.cart_calls({**{i: self.cart_lines.get(i, 1) for i in live}, **adds}, [])
            calls = [c for c in calls if any(v in adds for v in c.values() if isinstance(v, str))]
            if not calls or len(calls) + 1 > self.left() or self.room() < self.pace() * (len(calls) + 1) + 4:
                return
            for step_args in calls:
                faults: list[str] = []
                step_args = _shape(step_args, self.schemas.get(cart), "args", faults)
                if faults:
                    return
                reply = self.act(cart, step_args)
                if reply is None or self.finished:
                    return
                self.note_cart(cart, step_args)
                self.observe(cart, reply.get("observation"))
            self.trace.append({"kind": "fill_missing", "added": adds})
        except Exception as exc:
            self.trace.append({"kind": "fill_missing_error", "error": f"{type(exc).__name__}: {exc}"[:200]})

    # ---------- literal evidence: the grader's reading of a record, by code ----------
    def literal_needs(self, number: int) -> list[tuple[str, str]]:
        """An item's plain requirements as (property, value). A bound, an exclusion, a list of options or a pointer to another
        listing is left to the model's reading."""
        out: list[tuple[str, str]] = []
        item = self.items[number - 1]
        for need in item.musts + item.told:
            if need in item.dropped or ":" not in need:
                continue
            prop, value = need.split(":", 1)
            value = str(self.ref_values.get(need) or value).strip()
            low = value.lower()
            tokens = low.split()
            # a plain value is short and names the value itself; a longer phrase, a figure with qualifiers, a negation, an
            # alternative or a pointer to another listing is a constraint the model reads (a misread here fails closed: no
            # record can state such a phrase word for word, so nothing is swapped on it)
            if not prop.strip() or not tokens or " the " in f" {low} " or re.search(r"\b(or|but|unlike|same|whatever|page|listing)\b", low):
                continue
            mode, _ = _constraint(value)
            # a bound ("at least 320 ml") and an exclusion ("not wicker") are read as the grader reads them; a plain value is short and names
            # the value itself, anything longer is left to the model
            if mode == "plain" and (len(tokens) > 3 or (len(tokens) >= 3 and re.search(r"\d", low)) or re.search(r"\b(not|no|than)\b", low)):
                continue
            out.append((prop.strip(), value))
        return out

    def literal_fit(self, ident: str, number: int) -> bool | None:
        """Whether an opened record states every plain requirement of the item as a grader reads it: the attribute whose key names
        the property carries the value's words (a number and its unit as one word); with no such key, the title or description does.
        False only when an attribute for the property states another value; a record with no such attribute is not held against (the grader passed such listings); None when the record was never opened or no requirement is plain."""
        record = self.full.get(ident)
        needs = self.literal_needs(number)
        if not record or not needs:
            return None
        block = _fact_block(record) or {}
        text = _dump(record)
        answered = set()
        for need in self.items[number - 1].told:
            if ":" in need:
                prop, value = need.split(":", 1)
                answered.add((prop.strip(), str(self.ref_values.get(need) or value).strip()))
        unknown = False
        for prop, value in needs:
            mode, core = _constraint(value)
            if mode == "plain" and not core:
                continue
            keys = [k for k in block if _names(_stem(prop), _stem(k))] or _aligned_keys(prop, list(block))
            if keys:
                verdicts = [_meets(mode, core, block[k]) for k in keys]
                if any(v is True for v in verdicts):
                    continue
                if all(v is False for v in verdicts) or (prop, value) in answered:
                    return False   # the attribute states another value, or a figure the shopper's bound is not met by
                unknown = True
            elif (prop, value) in answered:
                # the shopper's own words with no attribute for them on this record: the grader reads the table literally (race 14: every such
                # order was a miss), so the record passes only when it states the value somewhere else
                if not (mode == "plain" and core <= _value_words(text)):
                    return False
            elif mode != "plain" or not core <= _value_words(text):
                unknown = True   # a stated value the record states nowhere: neither a fit nor a contradiction
        return None if unknown else True

    def settle_changes(self) -> str:
        """A change arrived (budget, a dropped or added requirement) since the cart was built: each cart line it unsettles is
        replaced by the cheapest row already seen for that item which the requirements now in force do not rule out, after
        reading the cheapest unread candidates (three at most); when nothing seen can serve, one search is run first. Never throws."""
        try:
            changes = self.pending_changes[:]
            self.pending_changes.clear()
            cart = self.roles.get("cart")
            if not (changes and cart) or self.left() < 4 or not self.allow(self.pace() * 3 + 20):
                return ""
            budget = self.budget if isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool) else None
            per_item = self.brief.get("budget_scope") == "each"
            cheapest = self.brief.get("pick") == "cheapest"
            needs_moved = any(c.startswith(("dropped", "added")) for c in changes)
            cost = lambda i: self.cost_of(i)
            live = [i for i in self.cart_ids() if not self.sold_out(i)]
            spent = sum((cost(i) or 0.0) * self.cart_lines.get(i, 1) for i in live)
            replaced: dict[str, str] = {}
            report: list[str] = []
            index = 0
            while index < len(live):
                old = live[index]
                index += 1
                number = self.item_of(old)
                if number is None:
                    continue
                qty = self.items[number - 1].qty
                held = self.cart_lines.get(old, qty)
                old_cost = cost(old) or 0.0
                breaks = budget is not None and (old_cost * qty > budget if per_item else spent > budget + 1e-9)
                if not breaks and not (needs_moved and self.literal_fit(old, number) is False):
                    continue   # race 14: a line that still states every requirement stays (every swap away from a fitting line was a miss)
                allowance = None if budget is None else (budget if per_item else budget - (spent - old_cost * held))
                taken = set(live) | set(replaced.values())

                def serves(i: str) -> bool:
                    c = cost(i)
                    if i in taken or c is None or not self.buyable(i) or self.item_of(i) != number:
                        return False
                    if allowance is not None and c * qty > allowance + 1e-9:
                        return False
                    if cheapest and not breaks and c >= old_cost:
                        return False
                    return self.literal_fit(i, number) is not False

                found = [i for i in self.rows if serves(i)]
                if not found:
                    found = [i for i in self.search_once(number) if serves(i)]
                found.sort(key=lambda i: cost(i) or 0.0)
                if not found:
                    report.append(f"item {number}: nothing seen of that kind fits the changed requirements within the budget; search for one")
                    continue
                self.fill_facts([i for i in found[:3] if i not in self.full], self.tail() + 3)
                choice = next((i for i in found[:3] if self.literal_fit(i, number) is True), None)
                if choice is None and (breaks or (cheapest and not self.literal_needs(number))):
                    # a record silent on a plain need serves only against a sure fail (the line breaks the budget) or when a cheapest pick has nothing plain left
                    choice = next((i for i in found[:3] if i in self.full and self.literal_fit(i, number) is None), None)
                if choice is None:
                    report.append(f"item {number}: the cheaper rows {', '.join(found[:3])} could not be read as fits; judge them yourself")
                    continue
                replaced[old] = choice
                spent += ((cost(choice) or 0.0) - old_cost) * held
                report.append(f"item {number}: {old} ({old_cost:g}) gives way to {choice} ({cost(choice):g}), the cheapest seen row the requirements now allow")
            if replaced:
                lines = {replaced.get(i, i): self.cart_lines.get(i, 1) for i in live}
                calls = self.cart_calls(lines, [o for o in replaced if o not in lines]) or []
                if not calls:
                    report = [r.replace("gives way to", "should give way to") for r in report] + ["Kyo could not change the cart itself: make these replacements with the cart tool"]
                for call in calls[:4]:
                    faults: list[str] = []
                    call = _shape(call, self.schemas.get(cart), "args", faults)
                    if faults or self.left() < 3:
                        break
                    reply = self.act(cart, call)
                    if reply is None or self.finished:
                        break
                    self.note_cart(cart, call)
                    self.observe(cart, reply.get("observation"))
                self.trace.append({"kind": "settle_changes", "changes": changes[:4], "replaced": replaced, "calls": len(calls)})
            return ("[Kyo re-fitted the cart to the change] " + "; ".join(report)) if report else ""
        except Exception as exc:
            self.trace.append({"kind": "settle_error", "error": f"{type(exc).__name__}: {exc}"[:160]})
            return ""

    def search_once(self, number: int) -> list[str]:
        """One search naming the item and its plain requirement values, for a cart line nothing seen can replace."""
        search = self.roles.get("search")
        if not search or self.left() - self.tail() < 6 or not self.allow(self.pace() * 3 + 20):
            return []
        shape = self.shapes.get(search) or {}
        props = (self.schemas.get(search) or {}).get("properties") or {}
        used = [k for _, a in self.search_log for k, v in a.items() if isinstance(v, str) and k != shape.get("next_arg")]
        param = max(set(used), key=used.count) if used else next((k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "string" and k != shape.get("next_arg")), None)
        if param is None:
            return []
        phrase = " ".join(self.items[number - 1].name.split()[:3] + [v for _, v in self.literal_needs(number)][:2])[:120]
        faults: list[str] = []
        call = _shape({param: phrase}, self.schemas.get(search), "args", faults)
        if faults:
            return []
        reply = self.act(search, call)
        if reply is None or self.finished:
            return []
        self.observe(search, reply.get("observation"))
        return [i for i, _ in self.records_in(reply.get("observation"))]

    def literal_report(self) -> str:
        """Cart lines whose records contradict a requirement the shopper stated, or lack the attribute for it, with what the record shows,
        so the walker replaces them before the order instead of buying a sure miss. Never throws."""
        try:
            notes: list[str] = []
            for ident in self.live_lines():
                number = self.item_of(ident)
                if number is None or self.literal_fit(ident, number) is not False:
                    continue
                block = _fact_block(self.full.get(ident) or {}) or {}
                for prop, value in self.literal_needs(number):
                    mode, core = _constraint(value)
                    keys = [k for k in block if _names(_stem(prop), _stem(k))] or _aligned_keys(prop, list(block))
                    if keys and any(_meets(mode, core, block[k]) is True for k in keys):
                        continue
                    shown = "; ".join(f"{k}: {block[k]}" for k in keys[:2]) if keys else "no attribute for it"
                    notes.append(f"{ident} (item {number}) does not state {prop}: {value} ({shown})")
            if not notes:
                return ""
            self.trace.append({"kind": "literal_hold", "lines": notes[:4]})
            return ("These cart lines fail a requirement read literally from their attribute tables: " + "; ".join(notes[:4])
                    + ". A listing whose table lacks the attribute or shows another value is graded as a miss. Replace each such line with a listing whose table "
                      "states the value (search the category with the value's words, open candidates and read their tables), then check out.")
        except Exception as exc:
            self.trace.append({"kind": "literal_report_error", "error": f"{type(exc).__name__}: {exc}"[:160]})
            return ""

    def literal_swap(self) -> str:
        """Before a checkout: a cart line whose record contradicts or never states a plain requirement gives way to an opened,
        buyable listing of the same item that states them all, while the budget holds. Never throws; returns a note for the model."""
        try:
            cart = self.roles.get("cart")
            if not cart or self.left() < 3:
                return ""
            budget = self.budget if isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool) else None
            each = self.brief.get("budget_scope") == "each"
            live = self.live_lines()
            if not live:
                return ""
            spent = sum((self.cost_of(i) or 0.0) * self.cart_lines.get(i, 1) for i in live)
            pools = self.kind_pools(budget)
            swaps: dict[str, str] = {}
            for ident in live:
                number = self.item_of(ident)
                if number is None or self.literal_fit(ident, number) is not False:
                    continue
                qty = self.items[number - 1].qty
                fits = sorted((i for i in pools.get(number, []) if i in self.full and i not in live and i not in swaps.values() and self.literal_fit(i, number)),
                              key=lambda i: self.cost_of(i) if self.cost_of(i) is not None else float("inf"))
                for alt in fits:
                    total = spent - (self.cost_of(ident) or 0.0) * self.cart_lines.get(ident, qty) + (self.cost_of(alt) or 0.0) * qty
                    if budget is not None and not each and total > budget:
                        continue
                    swaps[ident] = alt
                    spent = total
                    break
            if not swaps:
                return ""
            lines = {i: self.cart_lines.get(i, 1) for i in live if i not in swaps}
            for old, alt in swaps.items():
                lines[alt] = self.items[(self.item_of(old) or 1) - 1].qty
            calls = self.cart_calls(lines, list(swaps))
            if not calls or len(calls) + 1 > self.left() or self.room() < self.pace() * (len(calls) + 1) + 4:
                return ""
            for args in calls:
                faults: list[str] = []
                args = _shape(args, self.schemas.get(cart), "args", faults)
                if faults:
                    return ""
                reply = self.act(cart, args)
                if reply is None or self.finished:
                    return ""
                self.note_cart(cart, args)
                self.observe(cart, reply.get("observation"))
            self.trace.append({"kind": "literal_swap", "swaps": swaps})
            return ("Kyo replaced cart lines whose records did not state a requirement with opened listings that state every requirement: "
                    + ", ".join(f"{old} -> {alt}" for old, alt in swaps.items()) + ". Cart now: " + self.cart_text())
        except Exception as exc:
            self.trace.append({"kind": "literal_swap_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
            return ""

    def fallback_order(self, quick: bool = False, guard: bool = False) -> bool:
        """The least-contradicting order instead of an ending without a purchase: for each item a buyable in-budget listing of
        its kind among the records seen (the audit's lead when it named one, else the heavy model's pick over the opened records,
        else the cheapest buyable row of the kind), carted at the requested quantity and checked out. False when some item has
        no such listing at all: then an ending without a purchase is the right ending. Never throws."""
        try:
            return self.fallback_steps(quick, guard)
        except Exception as exc:
            self.trace.append({"kind": "fallback_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
            return False

    def fallback_steps(self, quick: bool = False, guard: bool = False) -> bool:
        cart, buy = self.roles.get("cart"), self.roles.get("buy")
        if not cart or not buy or self.fallbacks or self.left() < 2 or self.room() < self.pace() * 2 + 2:
            return False
        self.fallbacks += 1
        budget = self.budget if isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool) else None
        each = self.brief.get("budget_scope") == "each"
        pools = self.kind_pools(budget)
        if any(n not in pools for n in range(1, len(self.items) + 1)):
            self.trace.append({"kind": "fallback_none", "pools": {k: len(v) for k, v in pools.items()}})
            return False
        if budget is not None and not each:
            floor = sum(min(self.cost_of(i) or 0.0 for i in pools[n]) * self.items[n - 1].qty for n in pools)
            if floor > budget:
                self.trace.append({"kind": "fallback_floor", "floor": floor, "budget": budget})
                return False                                  # even the cheapest buyable rows together break the budget: nothing fits, the ending stands
        basket: dict[str, int] = {}
        for number, item in enumerate(self.items, 1):
            literal = [i for i in pools[number] if i in self.full and self.literal_fit(i, number)]   # a record stating every plain requirement comes before the audit's lead and the model's pick
            if literal:
                basket[min(literal, key=lambda i: self.cost_of(i) if self.cost_of(i) is not None else float("inf"))] = item.qty
                continue
            lead = next((i for i in (self.lead_items.get(str(number)) or []) if i in pools[number]), None)
            if lead:
                basket[lead] = item.qty
        asked = [n for n in range(1, len(self.items) + 1) if not any(self.item_of(i) == n for i in basket)]
        if asked and not quick and self.spend_ok(0.006, final=True) and self.allow(25):
            records = []
            for number in asked:
                opened = [i for i in pools[number] if i in self.full][:10] or pools[number][:6]
                records.append(f"item {number}: " + "\n".join(f"{i}: {self.record_of(i)}" for i in opened))
            prompt = PICK + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger() + "\n\nRECORDS BY ITEM:\n" + "\n".join(records)
            try:
                got = self.ask_json(prompt, [HEAVY, LIGHT], 600, final=True) or {}
            except Exception as exc:                          # a failed pick leaves the cheapest-row choice below in place
                got = {}
                self.trace.append({"kind": "fallback_pick_error", "error": f"{type(exc).__name__}: {exc}"[:160]})
            self.trace.append({"kind": "fallback_pick", "got": _clip(got, 200)})
            for number in asked:
                pick = str((got.get("picks") or {}).get(str(number)) or "").strip() if isinstance(got.get("picks"), dict) else ""
                if pick in pools[number]:
                    basket[pick] = self.items[number - 1].qty
        for number in range(1, len(self.items) + 1):
            if not any(self.item_of(i) == number for i in basket):
                # no verdict: the cheapest buyable row of the kind, read listings first
                pool = sorted(pools[number], key=lambda i: (i not in self.full, self.cost_of(i) if self.cost_of(i) is not None else float("inf")))
                basket[pool[0]] = self.items[number - 1].qty
        total = sum((self.cost_of(i) or 0.0) * q for i, q in basket.items())
        if budget is not None and not each and total > budget:
            self.trace.append({"kind": "fallback_over_budget", "basket": basket, "total": total})
            return False
        surplus = [i for i in self.cart_ids() if i not in basket]
        calls = self.cart_calls(basket, surplus)
        if not calls or len(calls) + 1 > self.left() or self.room() < self.pace() * (len(calls) + 1) + 3:
            return False
        for step_args in calls:
            faults: list[str] = []
            step_args = _shape(step_args, self.schemas.get(cart), "args", faults)
            if faults:
                return False
            reply = self.act(cart, step_args, guard=guard)
            if reply is None or self.finished:
                return False
            self.note_cart(cart, step_args)
            self.observe(cart, reply.get("observation"))
        faults = []
        buy_args = _shape(self.fill_required(buy, dict(self.brief.get("buy_args") or {}), "buy"), self.schemas.get(buy), "args", faults)
        if faults:
            return False
        reply = self.settle(buy, buy_args, repeat=True, guard=guard)
        self.trace.append({"kind": "fallback_order", "basket": basket, "quick": quick, "reply": _clip(reply, 200)})
        return self.finished

    # ---------- the watchdog ----------
    def arm_guard(self) -> None:
        if self.guard_thread is not None:
            return
        self.guard_thread = threading.Thread(target=self._guard, daemon=True)
        self.guard_thread.start()

    def _guard(self) -> None:
        while not self.finished and not self.broke:
            left = GUARD_S - (time.monotonic() - self.t0)
            if left <= 0:
                break
            time.sleep(min(1.0, left))
        if not self.finished and not self.broke:
            self.guard_finish()

    def guard_finish(self) -> None:
        """The finish sent by the watchdog: the main thread may be held in a model call or a long chain of store calls, and the
        sandbox ends the episode at 300 s with nothing sent. The order when the cart holds a live line, else the ending."""
        self.closing = True
        self.guard_fired = True
        until = time.monotonic() + 10.0                       # a store call in flight on the main thread finishes first (step numbering)
        while self.in_env and time.monotonic() < until:
            time.sleep(0.2)
        try:
            self.finish_now()
        except Exception as exc:
            self.trace.append({"kind": "guard_error", "error": f"{type(exc).__name__}: {exc}"[:300]})

    def finish_now(self) -> None:
        """The plain finish by code, no gate and no check: buy when a live line is held, else end without a purchase."""
        buy, refuse = self.roles.get("buy"), self.roles.get("refuse")
        if self.finished or self.left() <= 0:
            return
        if buy and self.live_lines():
            faults: list[str] = []
            args = _shape(self.fill_required(buy, dict(self.brief.get("buy_args") or {}), "buy"), self.schemas.get(buy), "args", faults)
            if not faults:
                self.settle(buy, args, repeat=True, guard=True)
                self.trace.append({"kind": "guard_buy", "finished": self.finished})
                if self.finished:
                    return
        elif buy and self.room() > self.pace() * 2 + 3 and self.fallback_order(quick=True, guard=True):
            return
        if refuse:
            faults = []
            args = _shape(self.fill_required(refuse, self.quiet(refuse, dict(self.brief.get("refuse_args") or {})), "refuse"), self.schemas.get(refuse), "args", faults)
            if not faults and self.left() > 0:
                reply = self.act(refuse, args, guard=True)
                self.trace.append({"kind": "guard_refuse", "reply": _clip(reply, 160)})

    def trim(self) -> None:
        size = sum(len(_dump(m.get("content"))) for m in self.chat)
        if size <= HISTORY_LIMIT:
            return
        tools = [i for i, m in enumerate(self.chat) if m.get("role") == "tool"]
        for index in tools[: max(1, len(tools) // 2)]:
            content = self.chat[index].get("content") or ""
            if len(content) > 600:
                ids = re.findall(r'"([A-Za-z0-9][A-Za-z0-9_\-:.]{5,40})"', content)
                ids = [i for i in dict.fromkeys(ids) if i in self.rows][:20]
                self.chat[index]["content"] = content[:400] + f"...(older result shortened; listings in it: {', '.join(ids)})"

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.arm_guard()
        self.read_request()
        self.trace.append({"kind": "brief", "brief": self.brief})
        opening = self.view.get("opening")
        if opening is not None:
            self.absorb(opening)
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            intro += "\n\nWHAT THE STORE SHOWS FIRST:\n" + self.show(opening)
        intro += "\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n" + self.status()
        self.chat = [{"role": "system", "content": GUIDE}, {"role": "user", "content": intro}]
        self.work(LOOP_CALLS, SPEND_HARD)
        if not self.finished:
            turned = self.finish(again=True)
            # the store turned the ending down (a change from the shopper, a sold-out line): a short second round deals with it
            if turned and not self.finished and not self.broke and not self.no_credit and self.left() > 1:
                self.chat.append({"role": "user", "content": "[Kyo] Kyo tried to end the episode and " + turned + "\nApply what the store says, then finish.\n" + self.status()})
                self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
            if not self.finished and not self.closing:
                self.finish()
        if self.guard_fired and self.guard_thread is not None and self.guard_thread.is_alive():
            self.guard_thread.join(timeout=max(1.0, min(25.0, self.room())))    # the watchdog's finish lands before the process goes
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls,
                           "seconds": round(time.monotonic() - self.t0, 1), "guard": self.guard_fired})
        return self.trace

    def reserve_now(self) -> float:
        """Room the loop leaves for the finish: the stage reserve plus a model turn's worth while no order has been attempted yet,
        because in race 13 the first order attempt brought every budget change and drop and the re-fit, re-read and grounded text
        after it need about six steps and forty seconds."""
        base = max(STAGE_RESERVE + 8, self.llm_pace() * 2 + 30)
        return base + (self.pace() * 3 + self.llm_pace() if self.buys == 0 else 0.0)

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.finished and not self.broke and not self.closing and self.left() > 0 and self.loop_calls < loop_cap and self.cost < spend_cap and self.room() > self.reserve_now() and (self.buys > 0 or self.left() >= 8):
            if self.ready_to_ask():
                summary = self.early_trail(self.interview())
                if summary:
                    self.chat.append({"role": "user", "content": summary + "\n" + self.status()})
                continue
            self.trim()
            self.loop_calls += 1
            message = self.think(self.chat, [LIGHT, SPARE], 3000, self.offered())
            if message is None:
                self.misses += 1
                if self.no_credit or self.calls >= ALL_CALLS or self.misses > 2 or self.closing or self.room() < STAGE_RESERVE + 8:
                    break
                continue
            calls = []
            for number, call in enumerate(message.get("tool_calls") or []):
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                raw = fn.get("arguments")
                parsed = raw if isinstance(raw, dict) else (_json_obj(raw) if isinstance(raw, str) and raw.strip() else {})
                calls.append({"id": str(call.get("id") or f"k{self.loop_calls}_{number}"), "type": "function",
                              "function": {"name": str(fn.get("name") or ""), "arguments": _dump(parsed if isinstance(parsed, dict) else {})}})
            content = message.get("content")
            if not isinstance(content, str):
                content = _dump(content) if content else ""
            if not calls:
                idle += 1
                self.chat.append({"role": "assistant", "content": content or "(no call)"})
                self.chat.append({"role": "user", "content": "Call one of the tools to continue.\n" + self.status()})
                if idle >= 3:
                    break
                continue
            idle = 0
            committing = {self.roles.get("cart"), self.roles.get("buy"), self.roles.get("refuse")} - {None}
            if not self.asked and self.roles.get("ask") and self.msgs_free() > 0 and any(c["function"]["name"] in committing for c in calls) and self.left() > 8:
                self.chat.append({"role": "assistant", "content": content or None, "tool_calls": calls})
                for call in calls:
                    self.chat.append({"role": "tool", "tool_call_id": call["id"], "content": "Not run (no step used): the shopper is asked first."})
                summary = self.early_trail(self.interview())
                self.chat.append({"role": "user", "content": (summary or "The interviewer had nothing to ask.") + "\n" + self.status()})
                continue
            self.chat.append({"role": "assistant", "content": content or None, "tool_calls": calls})
            self.trace.append({"kind": "turn", "text": content[:300], "calls": [c["function"] for c in calls]})
            replies = self.run_calls(calls)
            for index, (cid, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += "\n" + self.status()
                self.chat.append({"role": "tool", "tool_call_id": cid, "content": text})


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Kyo(problem_data)
    except Exception as exc:
        return [{"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]}]
    try:
        return episode.run()
    except Exception as exc:
        episode.trace.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]})
        try:
            if not episode.finished and not episode.closing:
                episode.finish_now()
        except Exception as again:
            episode.trace.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:400]})
        return episode.trace


__all__ = ["agent_main"]
